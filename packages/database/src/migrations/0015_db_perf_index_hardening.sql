-- V2.7 Database Performance & Concurrency Hardening
-- 1) Drop 2 obsolete redundant expression GIN FTS indexes on `document_chunks`
--    (cuts FTS write amplification by ~67% during ingestion and frees buffer RAM)
-- 2) Replace unscoped `idx_chunks_content_tsv_gin` with tenant-scoped partial composite
--    `btree_gin` index `(org_id, content_tsv)` excluding parent containers
-- 3) Tighten partial HNSW index predicate on `document_chunks` to exclude parent containers
--    so it matches the exact `WHERE` predicate of HNSW vector retrieval queries
-- 4) Add composite active-child document lookup index for selective ACL-scoped queries

CREATE EXTENSION IF NOT EXISTS "btree_gin";
--> statement-breakpoint
DROP INDEX IF EXISTS "idx_chunks_content_fts";
--> statement-breakpoint
DROP INDEX IF EXISTS "idx_chunks_content_fts_simple";
--> statement-breakpoint
DROP INDEX IF EXISTS "idx_chunks_content_tsv_gin";
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_chunks_org_content_tsv_gin"
  ON "document_chunks" USING GIN ("org_id", "content_tsv")
  WHERE "is_active" = true AND "chunk_role" <> 'parent';
--> statement-breakpoint
DROP INDEX IF EXISTS "idx_chunks_embedding_hnsw";
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_chunks_embedding_hnsw"
  ON "document_chunks" USING hnsw ("embedding" vector_cosine_ops)
  WITH (m = 16, ef_construction = 64)
  WHERE "is_active" = true AND "chunk_role" <> 'parent' AND "embedding" IS NOT NULL;
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_chunks_org_doc_active_searchable"
  ON "document_chunks" ("org_id", "document_id")
  WHERE "is_active" = true AND "chunk_role" <> 'parent';
