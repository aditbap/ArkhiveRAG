import { BaseRepository } from './base.repository.js';
import { kgCommunities } from '../schema/communities.js';
import { kgEntities, kgRelations } from '../schema/knowledge-graph.js';
import { HNSW_EF_SEARCH_DEFAULT, withHnswSearch } from '../client.js';
import { and, desc, eq, sql } from 'drizzle-orm';

function extractRows<T extends Record<string, unknown>>(res: unknown): T[] {
  if (Array.isArray(res)) return res as T[];
  if (res && typeof res === 'object' && 'rows' in res && Array.isArray((res as { rows: unknown[] }).rows)) {
    return (res as { rows: T[] }).rows;
  }
  return [];
}

export interface UpsertCommunityInput {
  level: number;
  clusterKey: string;
  clusterHash: string;
  title: string;
  summary: string;
  entityCanonicals: string[];
  documentIds: string[];
  embeddingModel: string;
  embedding: number[];
}

export class CommunityRepository extends BaseRepository {
  constructor() {
    super('kg_communities');
  }

  async getExistingCommunityHashes(orgId: string, level = 0): Promise<Map<string, string>> {
    this.requireTenant(orgId);
    const rows = await this.readDb
      .select({
        clusterKey: kgCommunities.clusterKey,
        clusterHash: kgCommunities.clusterHash,
      })
      .from(kgCommunities)
      .where(and(eq(kgCommunities.orgId, orgId), eq(kgCommunities.level, level)));
    return new Map(rows.map((r) => [r.clusterKey, r.clusterHash]));
  }

  async loadGraphForClustering(orgId: string, limitEntities = 2000, limitRelations = 6000) {
    this.requireTenant(orgId);
    const [entities, relations] = await Promise.all([
      this.readDb
        .select({
          canonicalName: kgEntities.canonicalName,
          entityName: kgEntities.entityName,
          entityType: kgEntities.entityType,
          description: kgEntities.description,
          documentIds: kgEntities.documentIds,
        })
        .from(kgEntities)
        .where(eq(kgEntities.orgId, orgId))
        .orderBy(desc(kgEntities.degree))
        .limit(limitEntities),
      this.readDb
        .select({
          sourceCanonical: kgRelations.sourceCanonical,
          targetCanonical: kgRelations.targetCanonical,
          weight: kgRelations.weight,
          keywords: kgRelations.keywords,
          description: kgRelations.description,
          documentIds: kgRelations.documentIds,
        })
        .from(kgRelations)
        .where(eq(kgRelations.orgId, orgId))
        .orderBy(desc(kgRelations.weight))
        .limit(limitRelations),
    ]);
    return { entities, relations };
  }

  async upsertCommunities(orgId: string, items: UpsertCommunityInput[]): Promise<void> {
    this.requireTenant(orgId);
    if (items.length === 0) return;

    for (const item of items) {
      await this.db
        .insert(kgCommunities)
        .values({
          orgId,
          level: item.level,
          clusterKey: item.clusterKey,
          clusterHash: item.clusterHash,
          title: item.title.slice(0, 256),
          summary: item.summary,
          entityCanonicals: item.entityCanonicals,
          documentIds: item.documentIds,
          embeddingModel: item.embeddingModel,
          embedding: item.embedding,
          updatedAt: new Date(),
        })
        .onConflictDoUpdate({
          target: [kgCommunities.orgId, kgCommunities.level, kgCommunities.clusterKey],
          set: {
            clusterHash: sql`excluded.cluster_hash`,
            title: sql`excluded.title`,
            summary: sql`excluded.summary`,
            entityCanonicals: sql`excluded.entity_canonicals`,
            documentIds: sql`excluded.document_ids`,
            embeddingModel: sql`excluded.embedding_model`,
            embedding: sql`excluded.embedding`,
            updatedAt: sql`now()`,
          },
        });
    }
  }

  /**
   * ACL-safe vector search over RAPTOR community summaries (#8).
   * SECURITY INVARIANT: When `documentIds` ACL scope is provided, uses strict array containment
   * (`kg_communities.document_ids <@ allowed_document_ids`) so a community summary containing
   * ANY unauthorized document is NEVER returned to a restricted caller.
   */
  async searchCommunities(
    orgId: string,
    embedding: number[],
    limit = 4,
    minSimilarity = 0.25,
    documentIds?: string[],
  ) {
    this.requireTenant(orgId);
    const vecLiteral = JSON.stringify(embedding);
    const maxDistance = Math.max(0, Math.min(2, 1 - minSimilarity));
    const overfetch = Math.min(Math.max(limit * 4, 24), 120);
    const efSearch = Math.max(HNSW_EF_SEARCH_DEFAULT, overfetch);

    const aclSql =
      documentIds && documentIds.length > 0
        ? sql`AND document_ids <@ ARRAY[${sql.join(documentIds.map((id) => sql`${id}::uuid`), sql`, `)}]::uuid[]`
        : sql``;

    const rows = await withHnswSearch(this.readDb, efSearch, async (tx) => {
      const res = await tx.execute(sql`
        WITH hnsw_candidates AS (
          SELECT
            id, level, title, summary, document_ids,
            (embedding <=> ${vecLiteral}::vector) AS distance
          FROM kg_communities
          WHERE org_id = ${orgId}::uuid
            AND embedding IS NOT NULL
            ${aclSql}
          ORDER BY embedding <=> ${vecLiteral}::vector ASC
          LIMIT ${overfetch}
        )
        SELECT
          id, level, title, summary, document_ids,
          (1.0 - distance)::float8 AS similarity
        FROM hnsw_candidates
        WHERE distance <= ${maxDistance}
        ORDER BY distance ASC
        LIMIT ${limit}
      `);
      return extractRows<Record<string, unknown>>(res);
    });

    return rows.map((r) => ({
      id: String(r.id),
      level: Number(r.level),
      title: String(r.title),
      summary: String(r.summary),
      documentIds: Array.isArray(r.document_ids) ? (r.document_ids as string[]) : [],
      similarity: Number(r.similarity),
    }));
  }
}
