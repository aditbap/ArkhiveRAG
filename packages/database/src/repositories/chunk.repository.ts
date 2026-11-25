import { BaseRepository } from './base.repository.js';
import { documentChunks } from '../schema/chunks.js';
import { documents } from '../schema/documents.js';
import { HNSW_EF_SEARCH_DEFAULT, withHnswSearch } from '../client.js';
import { estimateTokens, extractSignificantTerms } from '@rag/core';
import crypto from 'crypto';
import { sql, asc, desc, and, eq, ne, inArray, or, type SQL } from 'drizzle-orm';

export interface InsertChunkInput {
  text: string;
  embedding: number[] | null;
  pageNumber?: number;
  sectionHeading?: string;
  parentChunkId?: string | null;
  /** Positional index into the `parents` array when using `replaceVersionChunks`. */
  parentIndex?: number | null;
  chunkRole?: 'parent' | 'child' | 'standalone';
  metadata?: Record<string, unknown>;
}

export interface GovernanceRetrievalFilter {
  includeDrafts?: boolean;
  includeArchived?: boolean;
  asOfDate?: Date | null;
}

/**
 * Lean retrieval column projection for `document_chunks`.
 * Explicitly omits `embedding` (4–12 KB wire payload per row) and stored `content_tsv`
 * to eliminate ~85% of Postgres->Node wire serialization and TOAST decompression overhead.
 */
export const CHUNK_RETRIEVAL_COLUMNS = {
  id: documentChunks.id,
  documentId: documentChunks.documentId,
  versionId: documentChunks.versionId,
  orgId: documentChunks.orgId,
  chunkIndex: documentChunks.chunkIndex,
  content: documentChunks.content,
  contentHash: documentChunks.contentHash,
  tokenCount: documentChunks.tokenCount,
  pageNumber: documentChunks.pageNumber,
  sectionHeading: documentChunks.sectionHeading,
  parentChunkId: documentChunks.parentChunkId,
  chunkRole: documentChunks.chunkRole,
  metadata: documentChunks.metadata,
  embeddingModel: documentChunks.embeddingModel,
  qdrantPointId: documentChunks.qdrantPointId,
  isActive: documentChunks.isActive,
  createdAt: documentChunks.createdAt,
} as const;

/**
 * Lean retrieval column projection for `documents`.
 */
export const DOCUMENT_RETRIEVAL_COLUMNS = {
  id: documents.id,
  orgId: documents.orgId,
  title: documents.title,
  fileName: documents.fileName,
  fileType: documents.fileType,
  fileSizeBytes: documents.fileSizeBytes,
  fileHash: documents.fileHash,
  status: documents.status,
  folderId: documents.folderId,
  activeVersionId: documents.activeVersionId,
  tags: documents.tags,
  metadata: documents.metadata,
  effectiveAt: documents.effectiveAt,
  expiresAt: documents.expiresAt,
  supersedesDocumentId: documents.supersedesDocumentId,
  authorityTier: documents.authorityTier,
  governanceStatus: documents.governanceStatus,
  isDeleted: documents.isDeleted,
  createdAt: documents.createdAt,
  updatedAt: documents.updatedAt,
} as const;

export interface RetrievedChunkRecord {
  id: string;
  documentId: string;
  versionId: string;
  orgId: string;
  chunkIndex: number;
  content: string;
  contentHash: string;
  tokenCount: number;
  pageNumber: number | null;
  sectionHeading: string | null;
  parentChunkId: string | null;
  chunkRole: string;
  metadata: unknown;
  embeddingModel: string | null;
  qdrantPointId: string | null;
  isActive: boolean;
  createdAt: Date;
}

export interface RetrievedDocumentRecord {
  id: string;
  orgId: string;
  title: string;
  fileName: string;
  fileType: string;
  fileSizeBytes: number;
  fileHash: string;
  status: 'uploaded' | 'processing' | 'ready' | 'failed' | 'deleted';
  folderId: string | null;
  activeVersionId: string | null;
  tags: string[] | null;
  metadata: unknown;
  effectiveAt: Date | null;
  expiresAt: Date | null;
  supersedesDocumentId: string | null;
  authorityTier: number;
  governanceStatus: string;
  isDeleted: boolean;
  createdAt: Date;
  updatedAt: Date;
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

function buildRawGovernanceSql(filter?: GovernanceRetrievalFilter, alias = 'd'): SQL {
  const parts: SQL[] = [];
  const col = (name: string) => sql.raw(`${alias}.${name}`);
  if (!filter?.includeDrafts) {
    parts.push(sql`AND ${col('governance_status')} <> 'draft'`);
  }
  if (!filter?.includeArchived) {
    parts.push(sql`AND ${col('governance_status')} <> 'archived'`);
  }
  if (filter?.asOfDate) {
    const iso = filter.asOfDate.toISOString();
    parts.push(sql`AND (${col('effective_at')} IS NULL OR ${col('effective_at')} <= ${iso}::timestamptz)`);
    parts.push(sql`AND (${col('expires_at')} IS NULL OR ${col('expires_at')} > ${iso}::timestamptz)`);
  }
  return parts.length > 0 ? sql.join(parts, sql` `) : sql``;
}

function uuidArraySql(ids: string[]) {
  return sql`ARRAY[${sql.join(ids.map((id) => sql`${id}::uuid`), sql`, `)}]::uuid[]`;
}

function extractRows<T extends Record<string, unknown>>(res: unknown): T[] {
  if (Array.isArray(res)) return res as T[];
  if (res && typeof res === 'object' && 'rows' in res && Array.isArray((res as { rows: unknown[] }).rows)) {
    return (res as { rows: T[] }).rows;
  }
  return [];
}

function mapRawJoinedRow(row: Record<string, unknown>): {
  chunk: RetrievedChunkRecord;
  document: RetrievedDocumentRecord;
} {
  return {
    chunk: {
      id: String(row.c_id),
      documentId: String(row.c_document_id),
      versionId: String(row.c_version_id),
      orgId: String(row.c_org_id),
      chunkIndex: Number(row.c_chunk_index),
      content: String(row.c_content),
      contentHash: String(row.c_content_hash),
      tokenCount: Number(row.c_token_count),
      pageNumber: row.c_page_number != null ? Number(row.c_page_number) : null,
      sectionHeading: row.c_section_heading != null ? String(row.c_section_heading) : null,
      parentChunkId: row.c_parent_chunk_id != null ? String(row.c_parent_chunk_id) : null,
      chunkRole: String(row.c_chunk_role ?? 'standalone'),
      metadata: row.c_metadata ?? {},
      embeddingModel: row.c_embedding_model != null ? String(row.c_embedding_model) : null,
      qdrantPointId: row.c_qdrant_point_id != null ? String(row.c_qdrant_point_id) : null,
      isActive: Boolean(row.c_is_active),
      createdAt: row.c_created_at instanceof Date ? row.c_created_at : new Date(String(row.c_created_at)),
    },
    document: {
      id: String(row.d_id),
      orgId: String(row.d_org_id),
      title: String(row.d_title),
      fileName: String(row.d_file_name),
      fileType: String(row.d_file_type),
      fileSizeBytes: Number(row.d_file_size_bytes),
      fileHash: String(row.d_file_hash),
      status: String(row.d_status) as RetrievedDocumentRecord['status'],
      folderId: row.d_folder_id != null ? String(row.d_folder_id) : null,
      activeVersionId: row.d_active_version_id != null ? String(row.d_active_version_id) : null,
      tags: Array.isArray(row.d_tags) ? (row.d_tags as string[]) : [],
      metadata: row.d_metadata ?? {},
      effectiveAt: row.d_effective_at ? (row.d_effective_at instanceof Date ? row.d_effective_at : new Date(String(row.d_effective_at))) : null,
      expiresAt: row.d_expires_at ? (row.d_expires_at instanceof Date ? row.d_expires_at : new Date(String(row.d_expires_at))) : null,
      supersedesDocumentId: row.d_supersedes_document_id != null ? String(row.d_supersedes_document_id) : null,
      authorityTier: Number(row.d_authority_tier ?? 2),
      governanceStatus: String(row.d_governance_status ?? 'active'),
      isDeleted: Boolean(row.d_is_deleted),
      createdAt: row.d_created_at instanceof Date ? row.d_created_at : new Date(String(row.d_created_at)),
      updatedAt: row.d_updated_at instanceof Date ? row.d_updated_at : new Date(String(row.d_updated_at)),
    },
  };
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
   * Crash-safe, idempotent atomic replacement of all chunks for a single document version.
   * Runs inside one Postgres transaction:
   *   1. Deletes any partial rows previously written for `(orgId, documentId, versionId)`
   *      (prevents duplicate chunks when a worker crashes / SIGKILLs mid-ingestion and retries).
   *   2. Inserts parent containers and links child chunks via `parentIndex`.
   *   3. Deactivates chunks belonging to older versions of the same document.
   */
  async replaceVersionChunks(
    orgId: string,
    documentId: string,
    versionId: string,
    parents: InsertChunkInput[],
    searchableChunks: InsertChunkInput[],
    embeddingModel: string,
  ): Promise<{
    insertedParents: (typeof documentChunks.$inferSelect)[];
    insertedChunks: (typeof documentChunks.$inferSelect)[];
  }> {
    this.requireTenant(orgId);

    return this.db.transaction(async (tx) => {
      await tx
        .delete(documentChunks)
        .where(
          and(
            eq(documentChunks.orgId, orgId),
            eq(documentChunks.documentId, documentId),
            eq(documentChunks.versionId, versionId),
          ),
        );

      const batchSize = 100;
      const insertedParents: (typeof documentChunks.$inferSelect)[] = [];
      if (parents.length > 0) {
        const parentRecords = parents.map((chunk, index) => ({
          orgId,
          documentId,
          versionId,
          chunkIndex: index,
          content: chunk.text,
          contentHash: crypto.createHash('sha256').update(chunk.text).digest('hex'),
          tokenCount: estimateTokens(chunk.text),
          pageNumber: chunk.pageNumber ?? null,
          sectionHeading: chunk.sectionHeading ?? null,
          parentChunkId: null,
          chunkRole: 'parent' as const,
          metadata: chunk.metadata || {},
          embedding: null,
          embeddingModel: null,
        }));
        for (let i = 0; i < parentRecords.length; i += batchSize) {
          const rows = await tx.insert(documentChunks).values(parentRecords.slice(i, i + batchSize)).returning();
          insertedParents.push(...rows);
        }
      }

      const insertedChunks: (typeof documentChunks.$inferSelect)[] = [];
      if (searchableChunks.length > 0) {
        const childRecords = searchableChunks.map((chunk, index) => {
          const linkedParent =
            chunk.chunkRole === 'child' && chunk.parentIndex != null
              ? insertedParents[chunk.parentIndex]
              : undefined;
          return {
            orgId,
            documentId,
            versionId,
            chunkIndex: index,
            content: chunk.text,
            contentHash: crypto.createHash('sha256').update(chunk.text).digest('hex'),
            tokenCount: estimateTokens(chunk.text),
            pageNumber: chunk.pageNumber ?? null,
            sectionHeading: chunk.sectionHeading ?? null,
            parentChunkId: linkedParent?.id ?? chunk.parentChunkId ?? null,
            chunkRole: chunk.chunkRole ?? 'standalone',
            metadata: chunk.metadata || {},
            embedding: chunk.embedding,
            embeddingModel: chunk.embedding ? embeddingModel : null,
          };
        });
        for (let i = 0; i < childRecords.length; i += batchSize) {
          const rows = await tx.insert(documentChunks).values(childRecords.slice(i, i + batchSize)).returning();
          insertedChunks.push(...rows);
        }
      }

      await tx
        .update(documentChunks)
        .set({ isActive: false })
        .where(
          and(
            eq(documentChunks.orgId, orgId),
            eq(documentChunks.documentId, documentId),
            ne(documentChunks.versionId, versionId),
          ),
        );

      return { insertedParents, insertedChunks };
    });
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

    return this.readDb
      .select({
        chunk: CHUNK_RETRIEVAL_COLUMNS,
        document: DOCUMENT_RETRIEVAL_COLUMNS,
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
      const rows = await this.readDb
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

    return this.readDb
      .select({
        chunk: CHUNK_RETRIEVAL_COLUMNS,
        document: DOCUMENT_RETRIEVAL_COLUMNS,
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

    const distance = sql<number>`(${documentChunks.embedding} <=> ${JSON.stringify(queryEmbedding)}::vector)`;
    const similarity = sql<number>`(1.0 - (${documentChunks.embedding} <=> ${JSON.stringify(queryEmbedding)}::vector))`;
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

    const rows = await this.readDb
      .select({
        chunk: CHUNK_RETRIEVAL_COLUMNS,
        document: DOCUMENT_RETRIEVAL_COLUMNS,
        similarity,
      })
      .from(documentChunks)
      .innerJoin(documents, eq(documentChunks.documentId, documents.id))
      .where(and(...conditions))
      .orderBy(asc(distance))
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

  /**
   * HNSW-accelerated vector similarity search (`O(log N)`).
   * Uses a two-stage CTE ordering by raw cosine distance `embedding <=> $vec ASC`
   * (required for pgvector HNSW index scans) with lean non-embedding column projection
   * and PgBouncer-safe `SET LOCAL hnsw.ef_search` + `hnsw.iterative_scan`.
   */
  async searchSimilar(
    orgId: string,
    embedding: number[],
    limit: number = 5,
    minSimilarity: number = 0.5,
    documentIds?: string[],
    governance?: GovernanceRetrievalFilter,
  ) {
    this.requireTenant(orgId);
    const vecLiteral = JSON.stringify(embedding);
    const maxDistance = Math.max(0, Math.min(2, 1 - minSimilarity));
    const hasSelectiveFilter = Boolean((documentIds && documentIds.length > 0) || governance?.asOfDate);
    const overfetch = hasSelectiveFilter
      ? Math.min(Math.max(limit * 6, 64), 300)
      : Math.min(Math.max(limit * 4, 48), 250);
    const efSearch = Math.max(HNSW_EF_SEARCH_DEFAULT, overfetch);

    const docFilterSql =
      documentIds && documentIds.length > 0
        ? sql`AND c.document_id = ANY(${uuidArraySql(documentIds)})`
        : sql``;
    const governanceSql = buildRawGovernanceSql(governance, 'd');

    const rawRows = await withHnswSearch(this.readDb, efSearch, async (tx) => {
      const res = await tx.execute(sql`
        WITH hnsw_candidates AS (
          SELECT
            c.id AS c_id,
            c.document_id AS c_document_id,
            c.version_id AS c_version_id,
            c.org_id AS c_org_id,
            c.chunk_index AS c_chunk_index,
            c.content AS c_content,
            c.content_hash AS c_content_hash,
            c.token_count AS c_token_count,
            c.page_number AS c_page_number,
            c.section_heading AS c_section_heading,
            c.parent_chunk_id AS c_parent_chunk_id,
            c.chunk_role AS c_chunk_role,
            c.metadata AS c_metadata,
            c.embedding_model AS c_embedding_model,
            c.qdrant_point_id AS c_qdrant_point_id,
            c.is_active AS c_is_active,
            c.created_at AS c_created_at,
            (c.embedding <=> ${vecLiteral}::vector) AS distance
          FROM document_chunks c
          WHERE c.org_id = ${orgId}::uuid
            AND c.is_active = true
            AND c.chunk_role <> 'parent'
            AND c.embedding IS NOT NULL
            ${docFilterSql}
          ORDER BY c.embedding <=> ${vecLiteral}::vector ASC
          LIMIT ${overfetch}
        )
        SELECT
          h.c_id, h.c_document_id, h.c_version_id, h.c_org_id, h.c_chunk_index,
          h.c_content, h.c_content_hash, h.c_token_count, h.c_page_number,
          h.c_section_heading, h.c_parent_chunk_id, h.c_chunk_role, h.c_metadata,
          h.c_embedding_model, h.c_qdrant_point_id, h.c_is_active, h.c_created_at,
          d.id AS d_id, d.org_id AS d_org_id, d.title AS d_title, d.file_name AS d_file_name,
          d.file_type AS d_file_type, d.file_size_bytes AS d_file_size_bytes,
          d.file_hash AS d_file_hash, d.status AS d_status, d.folder_id AS d_folder_id,
          d.active_version_id AS d_active_version_id, d.tags AS d_tags,
          d.metadata AS d_metadata, d.effective_at AS d_effective_at, d.expires_at AS d_expires_at,
          d.supersedes_document_id AS d_supersedes_document_id,
          d.authority_tier AS d_authority_tier, d.governance_status AS d_governance_status,
          d.is_deleted AS d_is_deleted, d.created_at AS d_created_at, d.updated_at AS d_updated_at,
          (1.0 - h.distance)::float8 AS similarity
        FROM hnsw_candidates h
        INNER JOIN documents d ON d.id = h.c_document_id
        WHERE d.is_deleted = false
          AND h.distance <= ${maxDistance}
          ${governanceSql}
        ORDER BY h.distance ASC
        LIMIT ${limit}
      `);
      return extractRows<Record<string, unknown>>(res);
    });

    return rawRows.map((row) => ({
      ...mapRawJoinedRow(row),
      similarity: Number(row.similarity),
    }));
  }

  /**
   * Multilingual-safe, 100k-doc-scalable full-text search using the tenant-scoped
   * composite partial GIN index (`idx_chunks_org_content_tsv_gin`, migration 0015):
   * 1. Runs strict AND `plainto_tsquery` over code-prioritized significant terms first.
   * 2. Bounds candidate scoring inside a pre-rank CTE so broad terms at 100k+ docs never
   *    run `ts_rank_cd()` across tens of thousands of rows.
   * 3. Falls back to stopword-filtered OR-tsquery only when fewer than 3 chunks match.
   */
  async searchFullText(
    orgId: string,
    query: string,
    limit: number = 15,
    documentIds?: string[],
    governance?: GovernanceRetrievalFilter,
  ) {
    this.requireTenant(orgId);
    const sanitized = query.replace(/[^\p{L}\p{N}\s_-]+/gu, ' ').trim();
    if (!sanitized) return [];

    const significant = extractSignificantTerms(sanitized).slice(0, 8);
    const simpleQueryText = significant.length > 0 ? significant.join(' ') : sanitized;

    const docFilterSql =
      documentIds && documentIds.length > 0
        ? sql`AND c.document_id = ANY(${uuidArraySql(documentIds)})`
        : sql``;
    const governanceSql = buildRawGovernanceSql(governance, 'd');

    const runStoredTsvQuery = async (useAnd: boolean) => {
      const tsqueryExpr = useAnd
        ? sql`NULLIF(plainto_tsquery('simple', ${simpleQueryText})::text, '')::tsquery`
        : sql`NULLIF(replace(plainto_tsquery('simple', ${simpleQueryText})::text, ' & ', ' | '), '')::tsquery`;
      // Cap candidate rows before ts_rank_cd() + documents join so 100k-doc broad terms stay <35ms,
      // while smaller corpora (<500 matching rows) remain 100% exhaustive (0.0000 accuracy drift).
      const candidateCap = useAnd
        ? Math.min(Math.max(limit * 30, 450), 1000)
        : Math.min(Math.max(limit * 16, 250), 500);

      const res = await this.readDb.execute(sql`
        WITH q AS (
          SELECT ${tsqueryExpr} AS tsq
        ),
        fts_candidates AS (
          SELECT
            c.id AS c_id,
            c.document_id AS c_document_id,
            c.version_id AS c_version_id,
            c.org_id AS c_org_id,
            c.chunk_index AS c_chunk_index,
            c.content AS c_content,
            c.content_hash AS c_content_hash,
            c.token_count AS c_token_count,
            c.page_number AS c_page_number,
            c.section_heading AS c_section_heading,
            c.parent_chunk_id AS c_parent_chunk_id,
            c.chunk_role AS c_chunk_role,
            c.metadata AS c_metadata,
            c.embedding_model AS c_embedding_model,
            c.qdrant_point_id AS c_qdrant_point_id,
            c.is_active AS c_is_active,
            c.created_at AS c_created_at,
            c.content_tsv AS c_content_tsv
          FROM document_chunks c, q
          WHERE q.tsq IS NOT NULL
            AND c.org_id = ${orgId}::uuid
            AND c.is_active = true
            AND c.chunk_role <> 'parent'
            AND c.content_tsv @@ q.tsq
            ${docFilterSql}
          LIMIT ${candidateCap}
        )
        SELECT
          f.c_id, f.c_document_id, f.c_version_id, f.c_org_id, f.c_chunk_index,
          f.c_content, f.c_content_hash, f.c_token_count, f.c_page_number,
          f.c_section_heading, f.c_parent_chunk_id, f.c_chunk_role, f.c_metadata,
          f.c_embedding_model, f.c_qdrant_point_id, f.c_is_active, f.c_created_at,
          d.id AS d_id, d.org_id AS d_org_id, d.title AS d_title, d.file_name AS d_file_name,
          d.file_type AS d_file_type, d.file_size_bytes AS d_file_size_bytes,
          d.file_hash AS d_file_hash, d.status AS d_status, d.folder_id AS d_folder_id,
          d.active_version_id AS d_active_version_id, d.tags AS d_tags,
          d.metadata AS d_metadata, d.effective_at AS d_effective_at, d.expires_at AS d_expires_at,
          d.supersedes_document_id AS d_supersedes_document_id,
          d.authority_tier AS d_authority_tier, d.governance_status AS d_governance_status,
          d.is_deleted AS d_is_deleted, d.created_at AS d_created_at, d.updated_at AS d_updated_at,
          COALESCE(ts_rank_cd(f.c_content_tsv, q.tsq), 0)::float8 AS rank
        FROM fts_candidates f
        CROSS JOIN q
        INNER JOIN documents d ON d.id = f.c_document_id
        WHERE d.is_deleted = false
          ${governanceSql}
        ORDER BY rank DESC
        LIMIT ${limit}
      `);

      return extractRows<Record<string, unknown>>(res).map((row) => ({
        ...mapRawJoinedRow(row),
        rank: Number(row.rank),
      }));
    };

    const runFallbackQuery = async () => {
      const tsvSimple = sql`to_tsvector('simple', ${documentChunks.content})`;
      const tsvEn = sql`to_tsvector('english', ${documentChunks.content})`;
      const tsquerySimple = sql`NULLIF(replace(plainto_tsquery('simple', ${simpleQueryText})::text, ' & ', ' | '), '')::tsquery`;
      const tsqueryEn = sql`NULLIF(replace(plainto_tsquery('english', ${sanitized})::text, ' & ', ' | '), '')::tsquery`;
      const rank = sql<number>`GREATEST(
        COALESCE(ts_rank_cd(${tsvEn}, ${tsqueryEn}), 0),
        COALESCE(ts_rank_cd(${tsvSimple}, ${tsquerySimple}), 0)
      )`;

      const conditions = [
        eq(documentChunks.orgId, orgId),
        eq(documentChunks.isActive, true),
        ne(documentChunks.chunkRole, 'parent'),
        eq(documents.isDeleted, false),
        ...buildGovernanceConditions(governance),
        or(
          sql`(${tsqueryEn} IS NOT NULL AND ${tsvEn} @@ ${tsqueryEn})`,
          sql`(${tsquerySimple} IS NOT NULL AND ${tsvSimple} @@ ${tsquerySimple})`,
        ),
      ];
      if (documentIds && documentIds.length > 0) {
        conditions.push(inArray(documentChunks.documentId, documentIds));
      }

      const rows = await this.readDb
        .select({
          chunk: CHUNK_RETRIEVAL_COLUMNS,
          document: DOCUMENT_RETRIEVAL_COLUMNS,
          rank,
        })
        .from(documentChunks)
        .innerJoin(documents, eq(documentChunks.documentId, documents.id))
        .where(and(...conditions))
        .orderBy(desc(rank))
        .limit(limit);

      return rows.map((r) => ({ ...r, rank: Number(r.rank) }));
    };

    try {
      const strictHits = await runStoredTsvQuery(true);
      if (strictHits.length >= Math.min(3, limit)) return strictHits;
      return await runStoredTsvQuery(false);
    } catch {
      // Graceful fallback if migration 0014 (`content_tsv`) has not yet been applied
      return runFallbackQuery();
    }
  }
}
