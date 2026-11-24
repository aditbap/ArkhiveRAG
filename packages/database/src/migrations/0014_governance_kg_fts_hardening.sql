-- V2.2 Enterprise Governance, KG Intelligence, and 10k-Doc FTS Scale Hardening
-- 1) First-class document governance lifecycle (draft | approved | active | superseded | archived)
-- 2) KG entity/relation confidence, relation_type, directionality, and normalized aliases
-- 3) Precomputed stored generated content_tsv + GIN index for <20ms FTS at 10,000+ documents

ALTER TABLE "documents"
  ADD COLUMN IF NOT EXISTS "governance_status" varchar(24) DEFAULT 'active' NOT NULL;
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_documents_org_governance"
  ON "documents" ("org_id", "governance_status", "effective_at" DESC)
  WHERE "is_deleted" = false;
--> statement-breakpoint
ALTER TABLE "kg_entities"
  ADD COLUMN IF NOT EXISTS "confidence" real DEFAULT 0.85 NOT NULL,
  ADD COLUMN IF NOT EXISTS "normalized_aliases" text[] DEFAULT '{}'::text[] NOT NULL;
--> statement-breakpoint
ALTER TABLE "kg_relations"
  ADD COLUMN IF NOT EXISTS "relation_type" varchar(32) DEFAULT 'related' NOT NULL,
  ADD COLUMN IF NOT EXISTS "is_directed" boolean DEFAULT false NOT NULL,
  ADD COLUMN IF NOT EXISTS "confidence" real DEFAULT 0.85 NOT NULL;
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_kg_relations_org_type"
  ON "kg_relations" ("org_id", "relation_type");
--> statement-breakpoint
ALTER TABLE "document_chunks"
  ADD COLUMN IF NOT EXISTS "content_tsv" tsvector GENERATED ALWAYS AS (
    to_tsvector('simple', COALESCE("section_heading", '') || ' ' || "content")
  ) STORED;
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_chunks_content_tsv_gin"
  ON "document_chunks" USING GIN ("content_tsv")
  WHERE "is_active" = true;
