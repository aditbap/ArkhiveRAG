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

export interface GovernanceRetrievalFilter {
  includeDrafts?: boolean;
  includeArchived?: boolean;
  asOfDate?: Date | null;
}

function buildGovernanceConditions(filter?: GovernanceRetrievalFilter) {
  const conds = [];
  if (!filter?.includeDrafts) {
    conds.push(ne(documents.governanceStatus, 'draft'));
  }
  if (!filter?.includeArchived) {
    conds.push(ne(documents.governanceStatus, 'archived'));
  }
  if (filter?.asOfDate) {
    const iso = filter.asOfDate.toISOString();
    conds.push(sql`(${documents.effectiveAt} IS NULL OR ${documents.effectiveAt} <= ${iso}::timestamptz)`);
    conds.push(sql`(${documents.expiresAt} IS NULL OR ${documents.expiresAt} > ${iso}::timestamptz)`);
  }
  return conds;
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
    governance?: GovernanceRetrievalFilter,
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
      ...buildGovernanceConditions(governance),
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
    governance?: GovernanceRetrievalFilter,
  ) {
    const similarity = sql<number>`1 - (${documentChunks.embedding} <=> ${JSON.stringify(embedding)}::vector)`;

    const conditions = [
      eq(documentChunks.orgId, orgId),
      eq(documentChunks.isActive, true),
      ne(documentChunks.chunkRole, 'parent'),
      eq(documents.isDeleted, false),
      sql`${documentChunks.embedding} IS NOT NULL`,
      gte(similarity, minSimilarity),
      ...buildGovernanceConditions(governance),
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
   * Multilingual-safe, 10k-doc-scalable full-text search using the precomputed `content_tsv`
   * GIN index (migration 0014):
   * 1. Runs strict AND `plainto_tsquery` over code-prioritized significant terms first (<20ms at 10k docs).
   * 2. Falls back to stopword-filtered OR-tsquery only when fewer than 3 chunks match.
   */
  async searchFullText(
    orgId: string,
    query: string,
    limit: number = 15,
    documentIds?: string[],
    governance?: GovernanceRetrievalFilter,
  ) {
    const sanitized = query.replace(/[^\p{L}\p{N}\s_-]+/gu, ' ').trim();
    if (!sanitized) return [];

    const significant = extractSignificantTerms(sanitized).slice(0, 8);
    const simpleQueryText = significant.length > 0 ? significant.join(' ') : sanitized;

    const runQuery = async (useAnd: boolean, useStoredTsv: boolean) => {
      const tsvSimple = useStoredTsv
        ? sql`${documentChunks}.content_tsv`
        : sql`to_tsvector('simple', ${documentChunks.content})`;
      const tsvEn = sql`to_tsvector('english', ${documentChunks.content})`;
      const tsquerySimple = useAnd
        ? sql`NULLIF(plainto_tsquery('simple', ${simpleQueryText})::text, '')::tsquery`
        : sql`NULLIF(replace(plainto_tsquery('simple', ${simpleQueryText})::text, ' & ', ' | '), '')::tsquery`;
      const tsqueryEn = useAnd
        ? sql`NULLIF(plainto_tsquery('english', ${sanitized})::text, '')::tsquery`
        : sql`NULLIF(replace(plainto_tsquery('english', ${sanitized})::text, ' & ', ' | '), '')::tsquery`;

      const rank = useStoredTsv
        ? sql<number>`COALESCE(ts_rank_cd(${tsvSimple}, ${tsquerySimple}), 0)`
        : sql<number>`GREATEST(
            COALESCE(ts_rank_cd(${tsvEn}, ${tsqueryEn}), 0),
            COALESCE(ts_rank_cd(${tsvSimple}, ${tsquerySimple}), 0)
          )`;

      const matchCond = useStoredTsv
        ? sql`(${tsquerySimple} IS NOT NULL AND ${tsvSimple} @@ ${tsquerySimple})`
        : or(
            sql`(${tsqueryEn} IS NOT NULL AND ${tsvEn} @@ ${tsqueryEn})`,
            sql`(${tsquerySimple} IS NOT NULL AND ${tsvSimple} @@ ${tsquerySimple})`,
          );

      const conditions = [
        eq(documentChunks.orgId, orgId),
        eq(documentChunks.isActive, true),
        ne(documentChunks.chunkRole, 'parent'),
        eq(documents.isDeleted, false),
        ...buildGovernanceConditions(governance),
        matchCond,
      ];

      if (documentIds && documentIds.length > 0) {
        conditions.push(inArray(documentChunks.documentId, documentIds));
      }

      return this.db
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
    };

    try {
      const strictHits = await runQuery(true, true);
      if (strictHits.length >= Math.min(3, limit)) return strictHits;
      return await runQuery(false, true);
    } catch {
      // Graceful fallback if migration 0014 (`content_tsv`) has not yet been applied
      return runQuery(false, false);
    }
  }
}
