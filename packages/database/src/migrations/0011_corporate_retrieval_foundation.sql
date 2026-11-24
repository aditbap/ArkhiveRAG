-- Phase A: Corporate Retrieval Foundation
-- 1) pg_trgm extension + kg_entity_aliases for fuzzy & semantic Entity Resolution (#5)
-- 2) Temporal & Authority Conflict Resolution metadata on documents (#10)
-- 3) Parent-Child chunking support (chunk_role + parent_chunk_id index) (#7)

CREATE EXTENSION IF NOT EXISTS pg_trgm;
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "kg_entity_aliases" (
  "id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
  "org_id" uuid NOT NULL REFERENCES "organizations"("id") ON DELETE CASCADE,
  "alias_canonical" varchar(256) NOT NULL,
  "alias_name" varchar(256) NOT NULL,
  "target_canonical" varchar(256) NOT NULL,
  "similarity" real DEFAULT 1 NOT NULL,
  "created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS "ux_kg_entity_aliases_org_alias"
  ON "kg_entity_aliases" ("org_id", "alias_canonical");
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_kg_entity_aliases_org_target"
  ON "kg_entity_aliases" ("org_id", "target_canonical");
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_kg_entities_canonical_trgm"
  ON "kg_entities" USING GIN ("canonical_name" gin_trgm_ops);
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_kg_entities_name_trgm"
  ON "kg_entities" USING GIN ("entity_name" gin_trgm_ops);
--> statement-breakpoint
ALTER TABLE "documents"
  ADD COLUMN IF NOT EXISTS "effective_at" timestamp with time zone,
  ADD COLUMN IF NOT EXISTS "expires_at" timestamp with time zone,
  ADD COLUMN IF NOT EXISTS "supersedes_document_id" uuid REFERENCES "documents"("id") ON DELETE SET NULL,
  ADD COLUMN IF NOT EXISTS "authority_tier" smallint DEFAULT 2 NOT NULL;
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_documents_org_temporal"
  ON "documents" ("org_id", "authority_tier" DESC, "effective_at" DESC)
  WHERE "is_deleted" = false;
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_documents_supersedes"
  ON "documents" ("org_id", "supersedes_document_id")
  WHERE "supersedes_document_id" IS NOT NULL;
--> statement-breakpoint
ALTER TABLE "document_chunks"
  ADD COLUMN IF NOT EXISTS "chunk_role" varchar(16) DEFAULT 'standalone' NOT NULL;
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_chunks_org_parent"
  ON "document_chunks" ("org_id", "parent_chunk_id")
  WHERE "parent_chunk_id" IS NOT NULL;
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_chunks_org_role_active"
  ON "document_chunks" ("org_id", "chunk_role")
  WHERE "is_active" = true;
