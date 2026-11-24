import { BaseRepository } from './base.repository.js';
import { documentChunks } from '../schema/chunks.js';
import { documents } from '../schema/documents.js';
import { estimateTokens, extractSignificantTerms } from '@rag/core';
import crypto from 'crypto';
import { sql, desc, and, eq, ne, gte, inArray, or } from 'drizzle-orm';

export interface InsertChunkInput {
  text: string;
  embedding: number[] | null;
  pageNumber?: number;
  sectionHeading?: string;
  parentChunkId?: string | null;
  chunkRole?: 'parent' | 'child' | 'standalone';
  metadata?: Record<string, unknown>;
}

export class ChunkRepository extends BaseRepository {
  constructor() {
    super('document_chunks');
  }

  async insertChunks(
    orgId: string,
    documentId: string,
    versionId: string,
    chunks: InsertChunkInput[],
    embeddingModel: string,
  ) {
    if (chunks.length === 0) return [];

    const records = chunks.map((chunk, index) => {
      const hash = crypto.createHash('sha256').update(chunk.text).digest('hex');
      return {
        orgId,
        documentId,
        versionId,
        chunkIndex: index,
        content: chunk.text,
        contentHash: hash,
        tokenCount: estimateTokens(chunk.text),
        pageNumber: chunk.pageNumber ?? null,
        sectionHeading: chunk.sectionHeading ?? null,
        parentChunkId: chunk.parentChunkId ?? null,
        chunkRole: chunk.chunkRole ?? 'standalone',
        metadata: chunk.metadata || {},
        embedding: chunk.embedding,
        embeddingModel: chunk.embedding ? embeddingModel : null,
      };
    });

    // Insert in bounded batches so very large documents never exceed Postgres parameter limits.
    const batchSize = 100;
    const inserted: (typeof documentChunks.$inferSelect)[] = [];
    for (let i = 0; i < records.length; i += batchSize) {
      const batch = records.slice(i, i + batchSize);
      const rows = await this.db.insert(documentChunks).values(batch).returning();
      inserted.push(...rows);
    }
    return inserted;
  }

  /**
   * Fetches parent container chunks by ID (ACL & tenant-safe) for Parent-Child context expansion (#7).
   */
  async findParentsByIds(orgId: string, parentIds: string[], documentIds?: string[]) {
    const uniqueIds = [...new Set(parentIds.filter(Boolean))];
    if (uniqueIds.length === 0) return [];

    const conditions = [
      eq(documentChunks.orgId, orgId),
      eq(documentChunks.isActive, true),
      eq(documents.isDeleted, false),
      inArray(documentChunks.id, uniqueIds),
    ];
    if (documentIds && documentIds.length > 0) {
      conditions.push(inArray(documentChunks.documentId, documentIds));
    }

    return this.db
      .select({
        chunk: documentChunks,
        document: documents,
      })
      .from(documentChunks)
      .innerJoin(documents, eq(documentChunks.documentId, documents.id))
      .where(and(...conditions));
  }

  /**
   * Content-hash embedding reuse: looks up existing embeddings for identical chunk text
   * produced by the same `embeddingModel` within the same tenant, avoiding redundant
   * upstream embedding calls on re-uploads or overlapping documents.
   */
  async findExistingEmbeddingsByHashes(
    orgId: string,
    contentHashes: string[],
    embeddingModel: string,
  ): Promise<Map<string, number[]>> {
    const out = new Map<string, number[]>();
    const uniqueHashes = [...new Set(contentHashes.filter(Boolean))];
    if (uniqueHashes.length === 0) return out;

    const batchSize = 200;
    for (let i = 0; i < uniqueHashes.length; i += batchSize) {
      const batch = uniqueHashes.slice(i, i + batchSize);
      const rows = await this.db
        .select({
          contentHash: documentChunks.contentHash,
          embedding: documentChunks.embedding,
        })
        .from(documentChunks)
        .where(
          and(
            eq(documentChunks.orgId, orgId),
            eq(documentChunks.embeddingModel, embeddingModel),
            inArray(documentChunks.contentHash, batch),
            sql`${documentChunks.embedding} IS NOT NULL`,
          ),
        );

      for (const row of rows) {
        if (out.has(row.contentHash) || !row.embedding) continue;
        // pgvector driver may return either a parsed number[] or string "[0.1,0.2,...]"
        const vec = Array.isArray(row.embedding)
          ? row.embedding
          : typeof row.embedding === 'string'
            ? (row.embedding as string)
              .replace(/^\[|\]$/g, '')
              .split(',')
              .map(Number)
            : null;
        if (vec && vec.length > 0 && vec.every(Number.isFinite)) {
          out.set(row.contentHash, vec);
        }
      }
    }

    return out;
  }

  /**
   * Fetches active chunks by ID (used by LightRAG graph-to-chunk provenance expansion).
   */
  async findByIds(orgId: string, chunkIds: string[], documentIds?: string[]) {
    const uniqueIds = [...new Set(chunkIds.filter(Boolean))];
    if (uniqueIds.length === 0) return [];

    const conditions = [
      eq(documentChunks.orgId, orgId),
      eq(documentChunks.isActive, true),
      eq(documents.isDeleted, false),
      inArray(documentChunks.id, uniqueIds),
    ];

    if (documentIds && documentIds.length > 0) {
      conditions.push(inArray(documentChunks.documentId, documentIds));
    }

    return this.db
      .select({
        chunk: documentChunks,
        document: documents,
      })
      .from(documentChunks)
      .innerJoin(documents, eq(documentChunks.documentId, documents.id))
      .where(and(...conditions));
  }

  /**
   * Fetches active chunks by ID along with their cosine similarity to `queryEmbedding`
   * so graph-linked chunks can be ranked accurately by vector relevance (LightRAG VECTOR pick).
   */
  async findByIdsRanked(
    orgId: string,
    chunkIds: string[],
    queryEmbedding: number[],
    limit = 20,
    documentIds?: string[],
  ) {
    const uniqueIds = [...new Set(chunkIds.filter(Boolean))];
    if (uniqueIds.length === 0) return [];

    const similarity = sql<number>`1 - (${documentChunks.embedding} <=> ${JSON.stringify(queryEmbedding)}::vector)`;
    const conditions = [
      eq(documentChunks.orgId, orgId),
      eq(documentChunks.isActive, true),
      eq(documents.isDeleted, false),
      inArray(documentChunks.id, uniqueIds),
      sql`${documentChunks.embedding} IS NOT NULL`,
    ];

    if (documentIds && documentIds.length > 0) {
      conditions.push(inArray(documentChunks.documentId, documentIds));
    }

    const rows = await this.db
      .select({
        chunk: documentChunks,
        document: documents,
        similarity,
      })
      .from(documentChunks)
      .innerJoin(documents, eq(documentChunks.documentId, documents.id))
      .where(and(...conditions))
      .orderBy(desc(similarity))
      .limit(limit);

    return rows.map((r) => ({ ...r, similarity: Number(r.similarity) }));
  }

  /**
   * Deactivates chunks from any version of this document OTHER than
   * `keepVersionId`. Called after a new version's chunks finish indexing, so a
   * document never has two versions' worth of chunks searchable at once.
   */
  async deactivateOtherVersions(orgId: string, documentId: string, keepVersionId: string) {
    await this.db
      .update(documentChunks)
      .set({ isActive: false })
      .where(
        and(
          eq(documentChunks.orgId, orgId),
          eq(documentChunks.documentId, documentId),
          ne(documentChunks.versionId, keepVersionId),
        ),
      );
  }

  async searchSimilar(
    orgId: string,
    embedding: number[],
    limit: number = 5,
    minSimilarity: number = 0.5,
    documentIds?: string[],
  ) {
    const similarity = sql<number>`1 - (${documentChunks.embedding} <=> ${JSON.stringify(embedding)}::vector)`;

    const conditions = [
      eq(documentChunks.orgId, orgId),
      eq(documentChunks.isActive, true),
      ne(documentChunks.chunkRole, 'parent'),
      eq(documents.isDeleted, false),
      sql`${documentChunks.embedding} IS NOT NULL`,
      gte(similarity, minSimilarity),
    ];

    if (documentIds && documentIds.length > 0) {
      conditions.push(inArray(documentChunks.documentId, documentIds));
    }

    const results = await this.db
      .select({
        chunk: documentChunks,
        document: documents,
        similarity,
      })
      .from(documentChunks)
      .innerJoin(documents, eq(documentChunks.documentId, documents.id))
      .where(and(...conditions))
      .orderBy(desc(similarity))
      .limit(limit);

    return results.map((r) => ({ ...r, similarity: Number(r.similarity) }));
  }

  /**
   * Multilingual-safe full-text (BM25-ish) search using Postgres's built-in text search.
   * Combines the `english` GIN index (migration 0005) with the language-agnostic `simple`
   * GIN index (migration 0009) so both English stemmed queries AND non-English / exact-code
   * queries (e.g. Indonesian questions, policy/SKU codes) match cleanly.
   */
  async searchFullText(orgId: string, query: string, limit: number = 15, documentIds?: string[]) {
    const sanitized = query.replace(/[^\p{L}\p{N}\s_-]+/gu, ' ').trim();
    if (!sanitized) return [];

    // Strip high-frequency English/Indonesian stopwords BEFORE constructing the OR-tsquery on
    // the language-agnostic 'simple' GIN index. Without this, words like "the"/"is"/"in" match
    // 100% of rows at 10,000+ docs and force Postgres to score every chunk (900ms -> ~15ms!).
    const significant = extractSignificantTerms(sanitized).slice(0, 8);
    const simpleQueryText = significant.length > 0 ? significant.join(' ') : sanitized;

    const tsqueryEn = sql`NULLIF(replace(plainto_tsquery('english', ${sanitized})::text, ' & ', ' | '), '')::tsquery`;
    const tsquerySimple = sql`NULLIF(replace(plainto_tsquery('simple', ${simpleQueryText})::text, ' & ', ' | '), '')::tsquery`;
    const tsvectorEn = sql`to_tsvector('english', ${documentChunks.content})`;
    const tsvectorSimple = sql`to_tsvector('simple', ${documentChunks.content})`;

    const rank = sql<number>`GREATEST(
      COALESCE(ts_rank_cd(${tsvectorEn}, ${tsqueryEn}), 0),
      COALESCE(ts_rank_cd(${tsvectorSimple}, ${tsquerySimple}), 0)
    )`;

    const conditions = [
      eq(documentChunks.orgId, orgId),
      eq(documentChunks.isActive, true),
      ne(documentChunks.chunkRole, 'parent'),
      eq(documents.isDeleted, false),
      or(
        sql`(${tsqueryEn} IS NOT NULL AND ${tsvectorEn} @@ ${tsqueryEn})`,
        sql`(${tsquerySimple} IS NOT NULL AND ${tsvectorSimple} @@ ${tsquerySimple})`,
      ),
    ];

    if (documentIds && documentIds.length > 0) {
      conditions.push(inArray(documentChunks.documentId, documentIds));
    }

    const results = await this.db
      .select({
        chunk: documentChunks,
        document: documents,
        rank,
      })
      .from(documentChunks)
      .innerJoin(documents, eq(documentChunks.documentId, documents.id))
      .where(and(...conditions))
      .orderBy(desc(rank))
      .limit(limit);

    return results;
  }
}
