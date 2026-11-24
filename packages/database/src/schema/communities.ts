import {
  pgTable,
  uuid,
  varchar,
  text,
  smallint,
  timestamp,
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

export const kgCommunities = pgTable('kg_communities', {
  id: uuid('id').defaultRandom().primaryKey(),
  orgId: uuid('org_id')
    .references(() => organizations.id, { onDelete: 'cascade' })
    .notNull(),
  level: smallint('level').default(0).notNull(),
  clusterKey: varchar('cluster_key', { length: 64 }).notNull(),
  clusterHash: varchar('cluster_hash', { length: 64 }).notNull(),
  title: varchar('title', { length: 256 }).notNull(),
  summary: text('summary').notNull(),
  entityCanonicals: text('entity_canonicals').array().default([]).notNull(),
  documentIds: uuid('document_ids').array().default([]).notNull(),
  embeddingModel: varchar('embedding_model', { length: 100 }),
  embedding: vector1024('embedding'),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow().notNull(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).defaultNow().notNull(),
});
