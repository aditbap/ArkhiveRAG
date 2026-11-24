-- LightRAG-inspired Knowledge Graph (Entities + Relationships) and persistent LLM Cache
-- stored natively inside PostgreSQL + pgvector, strictly isolated per tenant (org_id).

CREATE TABLE IF NOT EXISTS "kg_entities" (
  "id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
  "org_id" uuid NOT NULL REFERENCES "organizations"("id") ON DELETE CASCADE,
  "entity_name" varchar(256) NOT NULL,
  "canonical_name" varchar(256) NOT NULL,
  "entity_type" varchar(64) NOT NULL DEFAULT 'Other',
  "description" text NOT NULL,
  "source_chunk_ids" uuid[] NOT NULL DEFAULT '{}',
  "document_ids" uuid[] NOT NULL DEFAULT '{}',
  "degree" integer NOT NULL DEFAULT 0,
  "embedding_model" varchar(100),
  "embedding" vector(2048),
  "created_at" timestamptz NOT NULL DEFAULT now(),
  "updated_at" timestamptz NOT NULL DEFAULT now()
);
--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS "idx_kg_entities_org_canonical"
  ON "kg_entities" ("org_id", "canonical_name");
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_kg_entities_org_degree"
  ON "kg_entities" ("org_id", "degree" DESC);
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_kg_entities_doc_ids"
  ON "kg_entities" USING GIN ("document_ids");
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_kg_entities_chunk_ids"
  ON "kg_entities" USING GIN ("source_chunk_ids");
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "kg_relations" (
  "id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
  "org_id" uuid NOT NULL REFERENCES "organizations"("id") ON DELETE CASCADE,
  "source_entity" varchar(256) NOT NULL,
  "target_entity" varchar(256) NOT NULL,
  "source_canonical" varchar(256) NOT NULL,
  "target_canonical" varchar(256) NOT NULL,
  "keywords" text NOT NULL DEFAULT '',
  "description" text NOT NULL,
  "weight" real NOT NULL DEFAULT 1,
  "source_chunk_ids" uuid[] NOT NULL DEFAULT '{}',
  "document_ids" uuid[] NOT NULL DEFAULT '{}',
  "embedding_model" varchar(100),
  "embedding" vector(2048),
  "created_at" timestamptz NOT NULL DEFAULT now(),
  "updated_at" timestamptz NOT NULL DEFAULT now()
);
--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS "idx_kg_relations_org_pair"
  ON "kg_relations" ("org_id", "source_canonical", "target_canonical");
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_kg_relations_org_source"
  ON "kg_relations" ("org_id", "source_canonical");
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_kg_relations_org_target"
  ON "kg_relations" ("org_id", "target_canonical");
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_kg_relations_doc_ids"
  ON "kg_relations" USING GIN ("document_ids");
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_kg_relations_chunk_ids"
  ON "kg_relations" USING GIN ("source_chunk_ids");
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "llm_cache" (
  "org_id" uuid NOT NULL REFERENCES "organizations"("id") ON DELETE CASCADE,
  "cache_key" varchar(128) NOT NULL,
  "cache_type" varchar(32) NOT NULL,
  "model" varchar(128) NOT NULL,
  "response_text" text NOT NULL,
  "created_at" timestamptz NOT NULL DEFAULT now(),
  "updated_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("org_id", "cache_key")
);
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_llm_cache_org_type"
  ON "llm_cache" ("org_id", "cache_type");
--> statement-breakpoint
-- Content-hash lookup index so unchanged chunks reuse existing embeddings on re-ingestion
CREATE INDEX IF NOT EXISTS "idx_chunks_org_content_hash"
  ON "document_chunks" ("org_id", "content_hash", "embedding_model")
  WHERE "embedding" IS NOT NULL;
--> statement-breakpoint
-- Language-agnostic ('simple') GIN index complementing the 'english' FTS index so
-- non-English documents (e.g. Indonesian) and exact product/policy codes match reliably.
CREATE INDEX IF NOT EXISTS "idx_chunks_content_fts_simple"
  ON "document_chunks" USING GIN (to_tsvector('simple', "content"));
--> statement-breakpoint
-- Optional halfvec HNSW ANN indexes for 2048-dim vectors (supported on pgvector >= 0.7).
-- Safely skipped if the database runs an older pgvector without halfvec.
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_type WHERE typname = 'halfvec') THEN
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_chunks_embedding_halfvec_hnsw ON document_chunks USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops) WITH (m = 16, ef_construction = 64) WHERE is_active AND embedding IS NOT NULL';
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_kg_entities_embedding_halfvec_hnsw ON kg_entities USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops) WITH (m = 16, ef_construction = 64) WHERE embedding IS NOT NULL';
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_kg_relations_embedding_halfvec_hnsw ON kg_relations USING hnsw ((embedding::halfvec(2048)) halfvec_cosine_ops) WITH (m = 16, ef_construction = 64) WHERE embedding IS NOT NULL';
  END IF;
EXCEPTION WHEN OTHERS THEN
  NULL;
END $$;
