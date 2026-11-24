-- Phase 1-4 LightRAG+ upgrades:
-- 1) Hybrid Lexical + Vector search indexes on kg_entities & kg_relations
-- 2) Document-level kg_indexed_hash for O(1) unchanged-document skip on re-upload
-- 3) Migrate embedding columns to vector(1024) for BGE-M3 (@cf/baai/bge-m3) with
--    native full-precision pgvector HNSW indexes (no halfvec quantization needed).

ALTER TABLE "documents"
  ADD COLUMN IF NOT EXISTS "kg_indexed_hash" varchar(64);
--> statement-breakpoint
-- Hybrid KG lexical GIN indexes (language-agnostic 'simple' config preserves exact policy/SOP codes & multilingual names)
CREATE INDEX IF NOT EXISTS "idx_kg_entities_fts_simple"
  ON "kg_entities" USING GIN (to_tsvector('simple', "entity_name" || ' ' || "description"));
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_kg_relations_fts_simple"
  ON "kg_relations" USING GIN (to_tsvector('simple', "keywords" || ' ' || "source_entity" || ' ' || "target_entity" || ' ' || "description"));
--> statement-breakpoint
-- Drop optional 2048-dim halfvec indexes from migration 0009 before resizing vector columns to 1024
DROP INDEX IF EXISTS "idx_chunks_embedding_halfvec_hnsw";
--> statement-breakpoint
DROP INDEX IF EXISTS "idx_kg_entities_embedding_halfvec_hnsw";
--> statement-breakpoint
DROP INDEX IF EXISTS "idx_kg_relations_embedding_halfvec_hnsw";
--> statement-breakpoint
-- Safely transition vector columns to vector(1024): keeps existing 1024-dim vectors, nulls retired 2048-dim ones
ALTER TABLE "document_chunks"
  ALTER COLUMN "embedding" TYPE vector(1024)
  USING (CASE WHEN "embedding" IS NOT NULL AND vector_dims("embedding") = 1024 THEN "embedding"::vector(1024) ELSE NULL END);
--> statement-breakpoint
ALTER TABLE "kg_entities"
  ALTER COLUMN "embedding" TYPE vector(1024)
  USING (CASE WHEN "embedding" IS NOT NULL AND vector_dims("embedding") = 1024 THEN "embedding"::vector(1024) ELSE NULL END);
--> statement-breakpoint
ALTER TABLE "kg_relations"
  ALTER COLUMN "embedding" TYPE vector(1024)
  USING (CASE WHEN "embedding" IS NOT NULL AND vector_dims("embedding") = 1024 THEN "embedding"::vector(1024) ELSE NULL END);
--> statement-breakpoint
-- Native full-precision HNSW cosine indexes (1024 <= 2000 pgvector HNSW limit)
CREATE INDEX IF NOT EXISTS "idx_chunks_embedding_hnsw"
  ON "document_chunks" USING hnsw ("embedding" vector_cosine_ops)
  WITH (m = 16, ef_construction = 64)
  WHERE "is_active" AND "embedding" IS NOT NULL;
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_kg_entities_embedding_hnsw"
  ON "kg_entities" USING hnsw ("embedding" vector_cosine_ops)
  WITH (m = 16, ef_construction = 64)
  WHERE "embedding" IS NOT NULL;
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_kg_relations_embedding_hnsw"
  ON "kg_relations" USING hnsw ("embedding" vector_cosine_ops)
  WITH (m = 16, ef_construction = 64)
  WHERE "embedding" IS NOT NULL;
