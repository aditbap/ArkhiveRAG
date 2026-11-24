import {
  pgTable,
  uuid,
  varchar,
  text,
  integer,
  real,
  timestamp,
  primaryKey,
  customType,
} from 'drizzle-orm/pg-core';
import { organizations } from './organizations.js';

const vector1024 = customType<{ data: number[]; driverData: string }>({
  dataType() {
    return 'vector(1024)';
  },
  toDriver(value: number[]): string {
    return `[${value.join(',')}]`;
  },
});

export const kgEntities = pgTable('kg_entities', {
  id: uuid('id').defaultRandom().primaryKey(),
  orgId: uuid('org_id')
    .references(() => organizations.id, { onDelete: 'cascade' })
    .notNull(),
  entityName: varchar('entity_name', { length: 256 }).notNull(),
  canonicalName: varchar('canonical_name', { length: 256 }).notNull(),
  entityType: varchar('entity_type', { length: 64 }).default('Other').notNull(),
  description: text('description').notNull(),
  sourceChunkIds: uuid('source_chunk_ids').array().default([]).notNull(),
  documentIds: uuid('document_ids').array().default([]).notNull(),
  degree: integer('degree').default(0).notNull(),
  embeddingModel: varchar('embedding_model', { length: 100 }),
  embedding: vector1024('embedding'),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).defaultNow().notNull(),
});

export const kgRelations = pgTable('kg_relations', {
  id: uuid('id').defaultRandom().primaryKey(),
  orgId: uuid('org_id')
    .references(() => organizations.id, { onDelete: 'cascade' })
    .notNull(),
  sourceEntity: varchar('source_entity', { length: 256 }).notNull(),
  targetEntity: varchar('target_entity', { length: 256 }).notNull(),
  sourceCanonical: varchar('source_canonical', { length: 256 }).notNull(),
  targetCanonical: varchar('target_canonical', { length: 256 }).notNull(),
  keywords: text('keywords').default('').notNull(),
  description: text('description').notNull(),
  weight: real('weight').default(1).notNull(),
  sourceChunkIds: uuid('source_chunk_ids').array().default([]).notNull(),
  documentIds: uuid('document_ids').array().default([]).notNull(),
  embeddingModel: varchar('embedding_model', { length: 100 }),
  embedding: vector1024('embedding'),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).defaultNow().notNull(),
});

export const kgEntityAliases = pgTable('kg_entity_aliases', {
  id: uuid('id').defaultRandom().primaryKey(),
  orgId: uuid('org_id')
    .references(() => organizations.id, { onDelete: 'cascade' })
    .notNull(),
  aliasCanonical: varchar('alias_canonical', { length: 256 }).notNull(),
  aliasName: varchar('alias_name', { length: 256 }).notNull(),
  targetCanonical: varchar('target_canonical', { length: 256 }).notNull(),
  similarity: real('similarity').default(1).notNull(),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
});

export const llmCache = pgTable(
  'llm_cache',
  {
    orgId: uuid('org_id')
      .references(() => organizations.id, { onDelete: 'cascade' })
      .notNull(),
    cacheKey: varchar('cache_key', { length: 128 }).notNull(),
    cacheType: varchar('cache_type', { length: 32 }).notNull(),
    model: varchar('model', { length: 128 }).notNull(),
    responseText: text('response_text').notNull(),
    createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
    updatedAt: timestamp('updated_at', { withTimezone: true }).defaultNow().notNull(),
  },
  (table) => ({
    pk: primaryKey({ columns: [table.orgId, table.cacheKey] }),
  }),
);
