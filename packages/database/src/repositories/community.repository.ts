import { BaseRepository } from './base.repository.js';
import { kgCommunities } from '../schema/communities.js';
import { kgEntities, kgRelations } from '../schema/knowledge-graph.js';
import { and, desc, eq, gte, sql } from 'drizzle-orm';

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
    const rows = await this.db
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
      this.db
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
      this.db
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
    const similarity = sql<number>`1 - (${kgCommunities.embedding} <=> ${JSON.stringify(embedding)}::vector)`;
    const conditions = [
      eq(kgCommunities.orgId, orgId),
      sql`${kgCommunities.embedding} IS NOT NULL`,
      gte(similarity, minSimilarity),
    ];

    if (documentIds && documentIds.length > 0) {
      const allowedArr = sql`ARRAY[${sql.join(documentIds.map((id) => sql`${id}::uuid`), sql`, `)}]::uuid[]`;
      conditions.push(sql`${kgCommunities.documentIds} <@ ${allowedArr}`);
    }

    const rows = await this.db
      .select({
        id: kgCommunities.id,
        level: kgCommunities.level,
        title: kgCommunities.title,
        summary: kgCommunities.summary,
        documentIds: kgCommunities.documentIds,
        similarity,
      })
      .from(kgCommunities)
      .where(and(...conditions))
      .orderBy(desc(similarity))
      .limit(limit);

    return rows.map((r) => ({ ...r, similarity: Number(r.similarity) }));
  }
}
