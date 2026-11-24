import {
  pgTable,
  uuid,
  varchar,
  text,
  integer,
  smallint,
  boolean,
  timestamp,
  jsonb,
} from 'drizzle-orm/pg-core';
import { organizations } from './organizations.js';
import { users } from './users.js';
import { messages } from './chat.js';

export const retrievalConfigs = pgTable('retrieval_configs', {
  id: uuid('id').defaultRandom().primaryKey(),
  orgId: uuid('org_id').references(() => organizations.id, { onDelete: 'cascade' }),
  version: integer('version').notNull(),
  status: varchar('status', { length: 24 }).default('draft').notNull(),
  configJson: jsonb('config_json').notNull(),
  evalMetricsJson: jsonb('eval_metrics_json'),
  parentConfigId: uuid('parent_config_id'),
  createdBy: uuid('created_by').references(() => users.id, { onDelete: 'set null' }),
  notes: text('notes'),
  promotedAt: timestamp('promoted_at', { withTimezone: true }),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
});

export const retrievalTelemetry = pgTable('retrieval_telemetry', {
  id: uuid('id').defaultRandom().primaryKey(),
  orgId: uuid('org_id')
    .references(() => organizations.id, { onDelete: 'cascade' })
    .notNull(),
  messageId: uuid('message_id').references(() => messages.id, { onDelete: 'set null' }),
  configVersion: integer('config_version').default(1).notNull(),
  queryHash: varchar('query_hash', { length: 64 }).notNull(),
  queryText: text('query_text').notNull(),
  intent: varchar('intent', { length: 32 }).notNull(),
  strategyUsed: varchar('strategy_used', { length: 32 }).notNull(),
  effectiveMode: varchar('effective_mode', { length: 16 }).notNull(),
  cragVerdict: varchar('crag_verdict', { length: 16 }).default('HIGH').notNull(),
  cragAction: varchar('crag_action', { length: 24 }).default('proceed').notNull(),
  candidateFeaturesJson: jsonb('candidate_features_json').default('[]').notNull(),
  timingsJson: jsonb('timings_json').default('{}').notNull(),
  latencyMs: integer('latency_ms').default(0).notNull(),
  contextTokens: integer('context_tokens').default(0).notNull(),
  promptTokens: integer('prompt_tokens').default(0).notNull(),
  completionTokens: integer('completion_tokens').default(0).notNull(),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
});

export const retrievalFeedback = pgTable('retrieval_feedback', {
  id: uuid('id').defaultRandom().primaryKey(),
  orgId: uuid('org_id')
    .references(() => organizations.id, { onDelete: 'cascade' })
    .notNull(),
  telemetryId: uuid('telemetry_id').references(() => retrievalTelemetry.id, { onDelete: 'set null' }),
  messageId: uuid('message_id').references(() => messages.id, { onDelete: 'set null' }),
  userId: uuid('user_id').references(() => users.id, { onDelete: 'set null' }),
  rating: smallint('rating').notNull(),
  citedChunkIds: uuid('cited_chunk_ids').array().default([]).notNull(),
  reasonCode: varchar('reason_code', { length: 64 }),
  comment: text('comment'),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
});

export const retrievalExperiments = pgTable('retrieval_experiments', {
  id: uuid('id').defaultRandom().primaryKey(),
  orgId: uuid('org_id')
    .references(() => organizations.id, { onDelete: 'cascade' })
    .notNull(),
  experimentType: varchar('experiment_type', { length: 24 }).default('shadow').notNull(),
  baselineConfigId: uuid('baseline_config_id').references(() => retrievalConfigs.id, { onDelete: 'set null' }),
  challengerConfigId: uuid('challenger_config_id').references(() => retrievalConfigs.id, { onDelete: 'set null' }),
  strategyName: varchar('strategy_name', { length: 64 }).notNull(),
  queryCount: integer('query_count').default(0).notNull(),
  metricsJson: jsonb('metrics_json').default('{}').notNull(),
  gatePassed: boolean('gate_passed'),
  gateReportJson: jsonb('gate_report_json'),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
});
