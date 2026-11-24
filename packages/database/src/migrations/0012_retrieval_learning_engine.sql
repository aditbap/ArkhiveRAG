-- Phase C: RetrievalLearningEngine (versioned configs, compact telemetry, feedback, shadow & regression experiments)

CREATE TABLE IF NOT EXISTS "retrieval_configs" (
  "id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
  "org_id" uuid REFERENCES "organizations"("id") ON DELETE CASCADE,
  "version" integer NOT NULL,
  "status" varchar(24) DEFAULT 'draft' NOT NULL,
  "config_json" jsonb NOT NULL,
  "eval_metrics_json" jsonb,
  "parent_config_id" uuid,
  "created_by" uuid REFERENCES "users"("id") ON DELETE SET NULL,
  "notes" text,
  "promoted_at" timestamp with time zone,
  "created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_retrieval_configs_org_status"
  ON "retrieval_configs" ("org_id", "status", "version" DESC);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "retrieval_telemetry" (
  "id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
  "org_id" uuid NOT NULL REFERENCES "organizations"("id") ON DELETE CASCADE,
  "message_id" uuid REFERENCES "messages"("id") ON DELETE SET NULL,
  "config_version" integer DEFAULT 1 NOT NULL,
  "query_hash" varchar(64) NOT NULL,
  "query_text" text NOT NULL,
  "intent" varchar(32) NOT NULL,
  "strategy_used" varchar(32) NOT NULL,
  "effective_mode" varchar(16) NOT NULL,
  "crag_verdict" varchar(16) DEFAULT 'HIGH' NOT NULL,
  "crag_action" varchar(24) DEFAULT 'proceed' NOT NULL,
  "candidate_features_json" jsonb DEFAULT '[]' NOT NULL,
  "timings_json" jsonb DEFAULT '{}' NOT NULL,
  "latency_ms" integer DEFAULT 0 NOT NULL,
  "context_tokens" integer DEFAULT 0 NOT NULL,
  "prompt_tokens" integer DEFAULT 0 NOT NULL,
  "completion_tokens" integer DEFAULT 0 NOT NULL,
  "created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_retrieval_telemetry_org_created"
  ON "retrieval_telemetry" ("org_id", "created_at" DESC);
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_retrieval_telemetry_org_intent"
  ON "retrieval_telemetry" ("org_id", "intent", "created_at" DESC);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "retrieval_feedback" (
  "id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
  "org_id" uuid NOT NULL REFERENCES "organizations"("id") ON DELETE CASCADE,
  "telemetry_id" uuid REFERENCES "retrieval_telemetry"("id") ON DELETE SET NULL,
  "message_id" uuid REFERENCES "messages"("id") ON DELETE SET NULL,
  "user_id" uuid REFERENCES "users"("id") ON DELETE SET NULL,
  "rating" smallint NOT NULL,
  "cited_chunk_ids" uuid[] DEFAULT '{}' NOT NULL,
  "reason_code" varchar(64),
  "comment" text,
  "created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_retrieval_feedback_org_created"
  ON "retrieval_feedback" ("org_id", "created_at" DESC);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "retrieval_experiments" (
  "id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
  "org_id" uuid NOT NULL REFERENCES "organizations"("id") ON DELETE CASCADE,
  "experiment_type" varchar(24) DEFAULT 'shadow' NOT NULL,
  "baseline_config_id" uuid REFERENCES "retrieval_configs"("id") ON DELETE SET NULL,
  "challenger_config_id" uuid REFERENCES "retrieval_configs"("id") ON DELETE SET NULL,
  "strategy_name" varchar(64) NOT NULL,
  "query_count" integer DEFAULT 0 NOT NULL,
  "metrics_json" jsonb DEFAULT '{}' NOT NULL,
  "gate_passed" boolean,
  "gate_report_json" jsonb,
  "created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "idx_retrieval_experiments_org_created"
  ON "retrieval_experiments" ("org_id", "created_at" DESC);
