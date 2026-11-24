-- Phase D: RAPTOR + Graph Community Summaries (#8) with strict ACL scope tagging & HNSW index

CREATE TABLE IF NOT EXISTS "kg_communities" (
  "id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
  "org_id" uuid NOT NULL REFERENCES "organizations"("id") ON DELETE CASCADE,
  "level" smallint DEFAULT 0 NOT NULL,
  "cluster_key" varchar(64) NOT NULL,
  "cluster_hash" varchar(64) NOT NULL,
  "title" varchar(256) NOT NULL,
  "summary" text NOT NULL,
  "entity_canonicals" text[] DEFAULT '{}' NOT NULL,
  "document_ids" uuid[] DEFAULT '{}' NOT NULL,
  "embedding_model" varchar(100),
  "embedding" vector(1024),
  "created_at" timestamp with time zone DEFAULT now() NOT NULL,
  "updated_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS "ux_kg_communities_org_level_key"
  ON "kg_communities" ("org_id", "level", "cluster_key");
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_kg_communities_org_docs_gin"
  ON "kg_communities" USING GIN ("document_ids");
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_kg_communities_embedding_hnsw"
  ON "kg_communities" USING hnsw ("embedding" vector_cosine_ops)
  WITH (m = 16, ef_construction = 64)
  WHERE "embedding" IS NOT NULL;
