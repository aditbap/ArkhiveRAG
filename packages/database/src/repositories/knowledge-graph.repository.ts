import { BaseRepository } from './base.repository.js';
import { kgEntities, kgEntityAliases, kgRelations, llmCache } from '../schema/knowledge-graph.js';
import { documents } from '../schema/documents.js';
import {
  CONSTANTS,
  weightedReciprocalRankFusion,
  type EntityAliasMapping,
  type MergedEntityInput,
  type MergedRelationInput,
  type PprEdge,
  type RetrievedEntity,
  type RetrievedRelation,
} from '@rag/core';
import { sql, and, eq, desc, gte, inArray, or } from 'drizzle-orm';

function capUuidList(ids: string[], max = CONSTANTS.KG_MAX_SOURCE_IDS_PER_RECORD): string[] {
  const unique = [...new Set(ids.filter(Boolean))];
  return unique.length > max ? unique.slice(0, max) : unique;
}

function uuidArraySql(ids: string[]) {
  return sql`ARRAY[${sql.join(ids.map((id) => sql`${id}::uuid`), sql`, `)}]::uuid[]`;
}

export class KnowledgeGraphRepository extends BaseRepository {
  constructor() {
    super('kg_entities');
  }

  /** Fast existence check so libraries without KG data skip KG retrieval with negligible overhead. */
  async hasGraphData(orgId: string, documentIds?: string[]): Promise<boolean> {
    this.requireTenant(orgId);
    const conditions = [eq(kgEntities.orgId, orgId)];
    if (documentIds && documentIds.length > 0) {
      conditions.push(sql`${kgEntities.documentIds} && ${uuidArraySql(documentIds)}`);
    }
    const rows = await this.db
      .select({ id: kgEntities.id })
      .from(kgEntities)
      .where(and(...conditions))
      .limit(1);
    return rows.length > 0;
  }

  async getExistingEntities(orgId: string, canonicalNames: string[]) {
    this.requireTenant(orgId);
    if (canonicalNames.length === 0) return [];
    return this.db
      .select()
      .from(kgEntities)
      .where(and(eq(kgEntities.orgId, orgId), inArray(kgEntities.canonicalName, canonicalNames)));
  }

  async getExistingRelations(orgId: string, pairs: Array<[string, string]>) {
    this.requireTenant(orgId);
    if (pairs.length === 0) return [];
    const sources = [...new Set(pairs.map(([s]) => s))];
    const candidates = await this.db
      .select()
      .from(kgRelations)
      .where(and(eq(kgRelations.orgId, orgId), inArray(kgRelations.sourceCanonical, sources)));
    const pairSet = new Set(pairs.map(([s, t]) => `${s}::${t}`));
    return candidates.filter((r) => pairSet.has(`${r.sourceCanonical}::${r.targetCanonical}`));
  }

  async upsertEntities(orgId: string, entities: MergedEntityInput[], embeddingModel: string): Promise<void> {
    this.requireTenant(orgId);
    if (entities.length === 0) return;

    const batchSize = 50;
    for (let i = 0; i < entities.length; i += batchSize) {
      const batch = entities.slice(i, i + batchSize);
      const values = batch.map((e) => ({
        orgId,
        entityName: e.entityName.slice(0, CONSTANTS.KG_ENTITY_NAME_MAX_LENGTH),
        canonicalName: e.canonicalName.slice(0, CONSTANTS.KG_ENTITY_NAME_MAX_LENGTH),
        entityType: e.entityType.slice(0, 64) || 'Other',
        description: e.description,
        sourceChunkIds: capUuidList(e.sourceChunkIds),
        documentIds: capUuidList(e.documentIds),
        embeddingModel,
        embedding: e.embedding ?? null,
        updatedAt: new Date(),
      }));

      await this.db
        .insert(kgEntities)
        .values(values)
        .onConflictDoUpdate({
          target: [kgEntities.orgId, kgEntities.canonicalName],
          set: {
            entityName: sql`excluded.entity_name`,
            entityType: sql`CASE WHEN excluded.entity_type <> 'Other' THEN excluded.entity_type ELSE ${kgEntities.entityType} END`,
            description: sql`excluded.description`,
            sourceChunkIds: sql`excluded.source_chunk_ids`,
            documentIds: sql`excluded.document_ids`,
            embeddingModel: sql`excluded.embedding_model`,
            embedding: sql`COALESCE(excluded.embedding, ${kgEntities.embedding})`,
            updatedAt: sql`now()`,
          },
        });
    }
  }

  async upsertRelations(orgId: string, relations: MergedRelationInput[], embeddingModel: string): Promise<void> {
    this.requireTenant(orgId);
    if (relations.length === 0) return;

    const touchedCanonicals = new Set<string>();
    const batchSize = 50;

    for (let i = 0; i < relations.length; i += batchSize) {
      const batch = relations.slice(i, i + batchSize);
      const values = batch.map((r) => {
        touchedCanonicals.add(r.sourceCanonical);
        touchedCanonicals.add(r.targetCanonical);
        return {
          orgId,
          sourceEntity: r.sourceEntity.slice(0, CONSTANTS.KG_ENTITY_NAME_MAX_LENGTH),
          targetEntity: r.targetEntity.slice(0, CONSTANTS.KG_ENTITY_NAME_MAX_LENGTH),
          sourceCanonical: r.sourceCanonical.slice(0, CONSTANTS.KG_ENTITY_NAME_MAX_LENGTH),
          targetCanonical: r.targetCanonical.slice(0, CONSTANTS.KG_ENTITY_NAME_MAX_LENGTH),
          keywords: r.keywords,
          description: r.description,
          weight: r.weight,
          sourceChunkIds: capUuidList(r.sourceChunkIds),
          documentIds: capUuidList(r.documentIds),
          embeddingModel,
          embedding: r.embedding ?? null,
          updatedAt: new Date(),
        };
      });

      await this.db
        .insert(kgRelations)
        .values(values)
        .onConflictDoUpdate({
          target: [kgRelations.orgId, kgRelations.sourceCanonical, kgRelations.targetCanonical],
          set: {
            sourceEntity: sql`excluded.source_entity`,
            targetEntity: sql`excluded.target_entity`,
            keywords: sql`excluded.keywords`,
            description: sql`excluded.description`,
            weight: sql`excluded.weight`,
            sourceChunkIds: sql`excluded.source_chunk_ids`,
            documentIds: sql`excluded.document_ids`,
            embeddingModel: sql`excluded.embedding_model`,
            embedding: sql`COALESCE(excluded.embedding, ${kgRelations.embedding})`,
            updatedAt: sql`now()`,
          },
        });
    }

    if (touchedCanonicals.size > 0) {
      await this.refreshEntityDegrees(orgId, [...touchedCanonicals]);
    }
  }

  async refreshEntityDegrees(orgId: string, canonicalNames?: string[]): Promise<void> {
    this.requireTenant(orgId);
    if (canonicalNames && canonicalNames.length > 0) {
      const nameList = sql.join(canonicalNames.map((n) => sql`${n}`), sql`, `);
      await this.db.execute(sql`
        UPDATE kg_entities e
        SET degree = COALESCE(deg.cnt, 0), updated_at = now()
        FROM (
          SELECT c AS canonical_name, count(*)::int AS cnt
          FROM (
            SELECT source_canonical AS c FROM kg_relations WHERE org_id = ${orgId}::uuid
            UNION ALL
            SELECT target_canonical AS c FROM kg_relations WHERE org_id = ${orgId}::uuid
          ) edges
          WHERE c IN (${nameList})
          GROUP BY c
        ) deg
        WHERE e.org_id = ${orgId}::uuid
          AND e.canonical_name = deg.canonical_name
      `);
      return;
    }

    await this.db.execute(sql`
      UPDATE kg_entities e
      SET degree = COALESCE((
        SELECT count(*)::int
        FROM kg_relations r
        WHERE r.org_id = ${orgId}::uuid
          AND (r.source_canonical = e.canonical_name OR r.target_canonical = e.canonical_name)
      ), 0),
      updated_at = now()
      WHERE e.org_id = ${orgId}::uuid
    `);
  }

  /**
   * Local retrieval branch: vector similarity search over `kg_entities`
   * ordered by cosine similarity with entity degree as secondary tie-breaker.
   */
  async searchEntities(
    orgId: string,
    embedding: number[],
    limit = 20,
    minSimilarity = 0.2,
    documentIds?: string[],
  ): Promise<RetrievedEntity[]> {
    this.requireTenant(orgId);
    const similarity = sql<number>`1 - (${kgEntities.embedding} <=> ${JSON.stringify(embedding)}::vector)`;
    const conditions = [
      eq(kgEntities.orgId, orgId),
      sql`${kgEntities.embedding} IS NOT NULL`,
      gte(similarity, minSimilarity),
    ];
    if (documentIds && documentIds.length > 0) {
      conditions.push(sql`${kgEntities.documentIds} && ${uuidArraySql(documentIds)}`);
    }

    const rows = await this.db
      .select({
        id: kgEntities.id,
        entityName: kgEntities.entityName,
        canonicalName: kgEntities.canonicalName,
        entityType: kgEntities.entityType,
        description: kgEntities.description,
        degree: kgEntities.degree,
        sourceChunkIds: kgEntities.sourceChunkIds,
        documentIds: kgEntities.documentIds,
        similarity,
      })
      .from(kgEntities)
      .where(and(...conditions))
      .orderBy(desc(similarity), desc(kgEntities.degree))
      .limit(limit);

    return rows.map((r) => ({ ...r, similarity: Number(r.similarity) }));
  }

  /**
   * Lexical full-text search over `kg_entities` using the language-agnostic 'simple' GIN index.
   * Catches exact policy/SOP codes (POL-SEC-409, SOP-JKT-882) and non-English proper nouns.
   */
  async searchEntitiesFullText(
    orgId: string,
    query: string,
    limit = 20,
    documentIds?: string[],
  ): Promise<RetrievedEntity[]> {
    this.requireTenant(orgId);
    const sanitized = query.replace(/[^\p{L}\p{N}\s_-]+/gu, ' ').trim();
    if (!sanitized) return [];

    const tsquery = sql`NULLIF(replace(plainto_tsquery('simple', ${sanitized})::text, ' & ', ' | '), '')::tsquery`;
    const tsvector = sql`to_tsvector('simple', ${kgEntities.entityName} || ' ' || ${kgEntities.description})`;
    const rank = sql<number>`COALESCE(ts_rank_cd(${tsvector}, ${tsquery}), 0)`;

    const conditions = [
      eq(kgEntities.orgId, orgId),
      sql`(${tsquery} IS NOT NULL AND ${tsvector} @@ ${tsquery})`,
    ];
    if (documentIds && documentIds.length > 0) {
      conditions.push(sql`${kgEntities.documentIds} && ${uuidArraySql(documentIds)}`);
    }

    const rows = await this.db
      .select({
        id: kgEntities.id,
        entityName: kgEntities.entityName,
        canonicalName: kgEntities.canonicalName,
        entityType: kgEntities.entityType,
        description: kgEntities.description,
        degree: kgEntities.degree,
        sourceChunkIds: kgEntities.sourceChunkIds,
        documentIds: kgEntities.documentIds,
        similarity: rank,
      })
      .from(kgEntities)
      .where(and(...conditions))
      .orderBy(desc(rank), desc(kgEntities.degree))
      .limit(limit);

    return rows.map((r) => ({ ...r, similarity: Number(r.similarity) }));
  }

  /**
   * Hybrid Entity Search: runs Vector similarity + Lexical FTS in parallel and fuses
   * via weighted RRF so exact codes and multilingual entity names never drop out.
   */
  async searchEntitiesHybrid(
    orgId: string,
    queryText: string,
    embedding: number[],
    limit = 20,
    minSimilarity = 0.2,
    documentIds?: string[],
  ): Promise<RetrievedEntity[]> {
    const [dense, lexical] = await Promise.all([
      this.searchEntities(orgId, embedding, limit, minSimilarity, documentIds),
      this.searchEntitiesFullText(orgId, queryText, limit, documentIds).catch(() => []),
    ]);
    if (lexical.length === 0) return dense;
    const fused = weightedReciprocalRankFusion<RetrievedEntity>(
      [
        { items: dense, weight: 1.0 },
        { items: lexical, weight: 0.35 },
      ],
      (e) => e.canonicalName,
      60,
    );
    return fused.slice(0, limit).map(({ item, score }) => ({ ...item, similarity: score }));
  }

  /**
   * Global retrieval branch: vector similarity search over `kg_relations`
   * ordered by cosine similarity with relation weight as secondary tie-breaker.
   */
  async searchRelations(
    orgId: string,
    embedding: number[],
    limit = 20,
    minSimilarity = 0.2,
    documentIds?: string[],
  ): Promise<RetrievedRelation[]> {
    this.requireTenant(orgId);
    const similarity = sql<number>`1 - (${kgRelations.embedding} <=> ${JSON.stringify(embedding)}::vector)`;
    const conditions = [
      eq(kgRelations.orgId, orgId),
      sql`${kgRelations.embedding} IS NOT NULL`,
      gte(similarity, minSimilarity),
    ];
    if (documentIds && documentIds.length > 0) {
      conditions.push(sql`${kgRelations.documentIds} && ${uuidArraySql(documentIds)}`);
    }

    const rows = await this.db
      .select({
        id: kgRelations.id,
        sourceEntity: kgRelations.sourceEntity,
        targetEntity: kgRelations.targetEntity,
        sourceCanonical: kgRelations.sourceCanonical,
        targetCanonical: kgRelations.targetCanonical,
        keywords: kgRelations.keywords,
        description: kgRelations.description,
        weight: kgRelations.weight,
        sourceChunkIds: kgRelations.sourceChunkIds,
        documentIds: kgRelations.documentIds,
        similarity,
      })
      .from(kgRelations)
      .where(and(...conditions))
      .orderBy(desc(similarity), desc(kgRelations.weight))
      .limit(limit);

    return rows.map((r) => ({ ...r, similarity: Number(r.similarity) }));
  }

  /**
   * Lexical full-text search over `kg_relations` using the 'simple' GIN index.
   */
  async searchRelationsFullText(
    orgId: string,
    query: string,
    limit = 20,
    documentIds?: string[],
  ): Promise<RetrievedRelation[]> {
    this.requireTenant(orgId);
    const sanitized = query.replace(/[^\p{L}\p{N}\s_-]+/gu, ' ').trim();
    if (!sanitized) return [];

    const tsquery = sql`NULLIF(replace(plainto_tsquery('simple', ${sanitized})::text, ' & ', ' | '), '')::tsquery`;
    const tsvector = sql`to_tsvector('simple', ${kgRelations.keywords} || ' ' || ${kgRelations.sourceEntity} || ' ' || ${kgRelations.targetEntity} || ' ' || ${kgRelations.description})`;
    const rank = sql<number>`COALESCE(ts_rank_cd(${tsvector}, ${tsquery}), 0)`;

    const conditions = [
      eq(kgRelations.orgId, orgId),
      sql`(${tsquery} IS NOT NULL AND ${tsvector} @@ ${tsquery})`,
    ];
    if (documentIds && documentIds.length > 0) {
      conditions.push(sql`${kgRelations.documentIds} && ${uuidArraySql(documentIds)}`);
    }

    const rows = await this.db
      .select({
        id: kgRelations.id,
        sourceEntity: kgRelations.sourceEntity,
        targetEntity: kgRelations.targetEntity,
        sourceCanonical: kgRelations.sourceCanonical,
        targetCanonical: kgRelations.targetCanonical,
        keywords: kgRelations.keywords,
        description: kgRelations.description,
        weight: kgRelations.weight,
        sourceChunkIds: kgRelations.sourceChunkIds,
        documentIds: kgRelations.documentIds,
        similarity: rank,
      })
      .from(kgRelations)
      .where(and(...conditions))
      .orderBy(desc(rank), desc(kgRelations.weight))
      .limit(limit);

    return rows.map((r) => ({ ...r, similarity: Number(r.similarity) }));
  }

  /**
   * Hybrid Relation Search: fuses Vector similarity + Lexical FTS via weighted RRF
   * and gives a small bridge boost to cross-document relationships.
   */
  async searchRelationsHybrid(
    orgId: string,
    queryText: string,
    embedding: number[],
    limit = 20,
    minSimilarity = 0.2,
    documentIds?: string[],
  ): Promise<RetrievedRelation[]> {
    const [dense, lexical] = await Promise.all([
      this.searchRelations(orgId, embedding, limit, minSimilarity, documentIds),
      this.searchRelationsFullText(orgId, queryText, limit, documentIds).catch(() => []),
    ]);
    if (lexical.length === 0) return dense;
    const fused = weightedReciprocalRankFusion<RetrievedRelation>(
      [
        { items: dense, weight: 1.0 },
        { items: lexical, weight: 0.35 },
      ],
      (r) => `${r.sourceCanonical}::${r.targetCanonical}`,
      60,
    );
    return fused.slice(0, limit).map(({ item, score }) => ({ ...item, similarity: score }));
  }

  /**
   * 1-hop graph neighborhood expansion for local entities -> connected relations.
   * Prioritizes cross-document bridge relations + edge weight.
   */
  async getRelationsForEntities(
    orgId: string,
    canonicalNames: string[],
    limit = 30,
    documentIds?: string[],
  ): Promise<RetrievedRelation[]> {
    this.requireTenant(orgId);
    if (canonicalNames.length === 0) return [];

    const conditions = [
      eq(kgRelations.orgId, orgId),
      or(
        inArray(kgRelations.sourceCanonical, canonicalNames),
        inArray(kgRelations.targetCanonical, canonicalNames),
      ),
    ];
    if (documentIds && documentIds.length > 0) {
      conditions.push(sql`${kgRelations.documentIds} && ${uuidArraySql(documentIds)}`);
    }

    const docSpread = sql<number>`COALESCE(array_length(${kgRelations.documentIds}, 1), 1)`;
    const rows = await this.db
      .select({
        id: kgRelations.id,
        sourceEntity: kgRelations.sourceEntity,
        targetEntity: kgRelations.targetEntity,
        sourceCanonical: kgRelations.sourceCanonical,
        targetCanonical: kgRelations.targetCanonical,
        keywords: kgRelations.keywords,
        description: kgRelations.description,
        weight: kgRelations.weight,
        sourceChunkIds: kgRelations.sourceChunkIds,
        documentIds: kgRelations.documentIds,
      })
      .from(kgRelations)
      .where(and(...conditions))
      .orderBy(desc(docSpread), desc(kgRelations.weight))
      .limit(limit);

    return rows.map((r) => ({ ...r, similarity: 0.5 }));
  }

  /**
   * 1-hop endpoint lookup for global relations -> connected entities.
   * Preserves caller-supplied canonical order (matching LightRAG's relation-similarity ordering)
   * instead of sorting by degree DESC, which previously let generic hub nodes crowd out
   * query-specific entities.
   */
  async getEntitiesByCanonicalNames(
    orgId: string,
    canonicalNames: string[],
    documentIds?: string[],
  ): Promise<RetrievedEntity[]> {
    this.requireTenant(orgId);
    if (canonicalNames.length === 0) return [];

    const conditions = [
      eq(kgEntities.orgId, orgId),
      inArray(kgEntities.canonicalName, canonicalNames),
    ];
    if (documentIds && documentIds.length > 0) {
      conditions.push(sql`${kgEntities.documentIds} && ${uuidArraySql(documentIds)}`);
    }

    const rows = await this.db
      .select({
        id: kgEntities.id,
        entityName: kgEntities.entityName,
        canonicalName: kgEntities.canonicalName,
        entityType: kgEntities.entityType,
        description: kgEntities.description,
        degree: kgEntities.degree,
        sourceChunkIds: kgEntities.sourceChunkIds,
        documentIds: kgEntities.documentIds,
      })
      .from(kgEntities)
      .where(and(...conditions));

    const byCanonical = new Map(rows.map((r) => [r.canonicalName, r]));
    const ordered: RetrievedEntity[] = [];
    for (const name of canonicalNames) {
      const row = byCanonical.get(name);
      if (row) ordered.push({ ...row, similarity: 0.5 });
    }
    return ordered;
  }

  /**
   * Resolves alias canonical names to their surviving canonical entity targets (#5 Entity Resolution).
   */
  async resolveEntityAliases(orgId: string, canonicalNames: string[]): Promise<Map<string, string>> {
    this.requireTenant(orgId);
    const out = new Map<string, string>();
    const unique = [...new Set(canonicalNames.filter(Boolean))];
    if (unique.length === 0) return out;

    const rows = await this.db
      .select({
        aliasCanonical: kgEntityAliases.aliasCanonical,
        targetCanonical: kgEntityAliases.targetCanonical,
      })
      .from(kgEntityAliases)
      .where(and(eq(kgEntityAliases.orgId, orgId), inArray(kgEntityAliases.aliasCanonical, unique)));

    for (const row of rows) {
      out.set(row.aliasCanonical, row.targetCanonical);
    }
    return out;
  }

  /**
   * Trigram fuzzy entity match (`pg_trgm`) for cross-document Entity Resolution (#5).
   */
  async findFuzzyEntityCandidates(
    orgId: string,
    canonicalName: string,
    minSimilarity = 0.72,
    limit = 3,
  ): Promise<Array<{ canonicalName: string; entityName: string; entityType: string; similarity: number }>> {
    this.requireTenant(orgId);
    if (!canonicalName || canonicalName.length < 4) return [];

    const sim = sql<number>`similarity(${kgEntities.canonicalName}, ${canonicalName})`;
    const rows = await this.db
      .select({
        canonicalName: kgEntities.canonicalName,
        entityName: kgEntities.entityName,
        entityType: kgEntities.entityType,
        similarity: sim,
      })
      .from(kgEntities)
      .where(and(eq(kgEntities.orgId, orgId), gte(sim, minSimilarity)))
      .orderBy(desc(sim))
      .limit(limit);

    return rows.map((r) => ({ ...r, similarity: Number(r.similarity) }));
  }

  async upsertEntityAliases(orgId: string, aliases: EntityAliasMapping[]): Promise<void> {
    this.requireTenant(orgId);
    if (aliases.length === 0) return;

    const deduped = new Map<string, EntityAliasMapping>();
    for (const a of aliases) {
      if (!a.aliasCanonical || !a.targetCanonical || a.aliasCanonical === a.targetCanonical) continue;
      deduped.set(a.aliasCanonical, a);
    }
    const items = [...deduped.values()];
    if (items.length === 0) return;

    const batchSize = 100;
    for (let i = 0; i < items.length; i += batchSize) {
      const batch = items.slice(i, i + batchSize).map((a) => ({
        orgId,
        aliasCanonical: a.aliasCanonical.slice(0, CONSTANTS.KG_ENTITY_NAME_MAX_LENGTH),
        aliasName: a.aliasName.slice(0, CONSTANTS.KG_ENTITY_NAME_MAX_LENGTH),
        targetCanonical: a.targetCanonical.slice(0, CONSTANTS.KG_ENTITY_NAME_MAX_LENGTH),
        similarity: a.similarity,
      }));

      await this.db
        .insert(kgEntityAliases)
        .values(batch)
        .onConflictDoUpdate({
          target: [kgEntityAliases.orgId, kgEntityAliases.aliasCanonical],
          set: {
            targetCanonical: sql`excluded.target_canonical`,
            similarity: sql`excluded.similarity`,
          },
        });
    }
  }

  /**
   * Loads a bounded, tenant-scoped, ACL-filtered multi-hop subgraph around `seedCanonicalNames`
   * for Graph-Aware Personalized PageRank (#1).
   * SECURITY INVARIANT: `org_id` and `documentIds` ACL filters are applied in SQL BEFORE any
   * edge or entity chunk provenance enters PPR propagation.
   */
  async getTenantSubgraph(
    orgId: string,
    seedCanonicalNames: string[],
    options: { maxHops?: number; maxEdges?: number; documentIds?: string[] } = {},
  ): Promise<{ edges: PprEdge[]; entityChunks: Map<string, string[]> }> {
    this.requireTenant(orgId);
    const maxHops = Math.min(2, Math.max(1, options.maxHops ?? 2));
    const maxEdges = options.maxEdges ?? 240;
    const documentIds = options.documentIds;

    const frontier = new Set(seedCanonicalNames.filter(Boolean));
    if (frontier.size === 0) {
      return { edges: [], entityChunks: new Map() };
    }

    const visitedNodes = new Set<string>(frontier);
    const edgeMap = new Map<string, PprEdge>();

    for (let hop = 0; hop < maxHops && frontier.size > 0 && edgeMap.size < maxEdges; hop++) {
      const currentSeeds = [...frontier].slice(0, 60);
      frontier.clear();

      const conditions = [
        eq(kgRelations.orgId, orgId),
        or(
          inArray(kgRelations.sourceCanonical, currentSeeds),
          inArray(kgRelations.targetCanonical, currentSeeds),
        ),
      ];
      if (documentIds && documentIds.length > 0) {
        conditions.push(sql`${kgRelations.documentIds} && ${uuidArraySql(documentIds)}`);
      }

      const rows = await this.db
        .select({
          sourceCanonical: kgRelations.sourceCanonical,
          targetCanonical: kgRelations.targetCanonical,
          weight: kgRelations.weight,
          sourceChunkIds: kgRelations.sourceChunkIds,
        })
        .from(kgRelations)
        .where(and(...conditions))
        .orderBy(desc(kgRelations.weight))
        .limit(maxEdges - edgeMap.size);

      for (const r of rows) {
        const key = `${r.sourceCanonical}::${r.targetCanonical}`;
        if (!edgeMap.has(key)) {
          edgeMap.set(key, {
            sourceCanonical: r.sourceCanonical,
            targetCanonical: r.targetCanonical,
            weight: r.weight,
            sourceChunkIds: r.sourceChunkIds,
          });
        }
        if (!visitedNodes.has(r.sourceCanonical)) {
          visitedNodes.add(r.sourceCanonical);
          frontier.add(r.sourceCanonical);
        }
        if (!visitedNodes.has(r.targetCanonical)) {
          visitedNodes.add(r.targetCanonical);
          frontier.add(r.targetCanonical);
        }
      }
    }

    const allNodes = [...visitedNodes];
    const entityConditions = [eq(kgEntities.orgId, orgId), inArray(kgEntities.canonicalName, allNodes)];
    if (documentIds && documentIds.length > 0) {
      entityConditions.push(sql`${kgEntities.documentIds} && ${uuidArraySql(documentIds)}`);
    }

    const entityRows = await this.db
      .select({
        canonicalName: kgEntities.canonicalName,
        sourceChunkIds: kgEntities.sourceChunkIds,
      })
      .from(kgEntities)
      .where(and(...entityConditions));

    const entityChunks = new Map<string, string[]>();
    for (const row of entityRows) {
      entityChunks.set(row.canonicalName, row.sourceChunkIds);
    }

    return { edges: [...edgeMap.values()], entityChunks };
  }

  async getDocumentKgHash(orgId: string, documentId: string): Promise<string | null> {
    this.requireTenant(orgId);
    const rows = await this.db
      .select({ kgIndexedHash: documents.kgIndexedHash })
      .from(documents)
      .where(and(eq(documents.orgId, orgId), eq(documents.id, documentId)))
      .limit(1);
    return rows[0]?.kgIndexedHash ?? null;
  }

  async setDocumentKgHash(orgId: string, documentId: string, fileHash: string): Promise<void> {
    this.requireTenant(orgId);
    await this.db
      .update(documents)
      .set({ kgIndexedHash: fileHash, updatedAt: new Date() })
      .where(and(eq(documents.orgId, orgId), eq(documents.id, documentId)));
  }

  /**
   * Incremental cleanup when a document is deleted or replaced by a new version:
   * strips references to `documentId` (and optionally outdated chunk IDs) from
   * entities & relations, and deletes nodes/edges that have no surviving documents.
   */
  async removeDocumentFromGraph(orgId: string, documentId: string): Promise<void> {
    this.requireTenant(orgId);

    // Strip documentId and any chunks belonging to that document from relations
    await this.db.execute(sql`
      UPDATE kg_relations
      SET document_ids = array_remove(document_ids, ${documentId}::uuid),
          source_chunk_ids = COALESCE((
            SELECT array_agg(cid)
            FROM unnest(source_chunk_ids) AS cid
            WHERE cid NOT IN (
              SELECT id FROM document_chunks WHERE org_id = ${orgId}::uuid AND document_id = ${documentId}::uuid
            )
          ), '{}'::uuid[]),
          updated_at = now()
      WHERE org_id = ${orgId}::uuid
        AND ${documentId}::uuid = ANY(document_ids)
    `);

    await this.db.execute(sql`
      DELETE FROM kg_relations
      WHERE org_id = ${orgId}::uuid
        AND COALESCE(array_length(document_ids, 1), 0) = 0
    `);

    // Strip documentId and its chunks from entities
    await this.db.execute(sql`
      UPDATE kg_entities
      SET document_ids = array_remove(document_ids, ${documentId}::uuid),
          source_chunk_ids = COALESCE((
            SELECT array_agg(cid)
            FROM unnest(source_chunk_ids) AS cid
            WHERE cid NOT IN (
              SELECT id FROM document_chunks WHERE org_id = ${orgId}::uuid AND document_id = ${documentId}::uuid
            )
          ), '{}'::uuid[]),
          updated_at = now()
      WHERE org_id = ${orgId}::uuid
        AND ${documentId}::uuid = ANY(document_ids)
    `);

    await this.db.execute(sql`
      DELETE FROM kg_entities
      WHERE org_id = ${orgId}::uuid
        AND COALESCE(array_length(document_ids, 1), 0) = 0
    `);

    await this.refreshEntityDegrees(orgId);
  }

  async getLlmCache(orgId: string, cacheKey: string): Promise<string | null> {
    this.requireTenant(orgId);
    const rows = await this.db
      .select({ responseText: llmCache.responseText })
      .from(llmCache)
      .where(and(eq(llmCache.orgId, orgId), eq(llmCache.cacheKey, cacheKey)))
      .limit(1);
    return rows[0]?.responseText ?? null;
  }

  async setLlmCache(
    orgId: string,
    cacheKey: string,
    cacheType: 'extract' | 'keywords' | 'summary',
    model: string,
    responseText: string,
  ): Promise<void> {
    this.requireTenant(orgId);
    await this.db
      .insert(llmCache)
      .values({ orgId, cacheKey, cacheType, model, responseText, updatedAt: new Date() })
      .onConflictDoUpdate({
        target: [llmCache.orgId, llmCache.cacheKey],
        set: {
          cacheType: sql`excluded.cache_type`,
          model: sql`excluded.model`,
          responseText: sql`excluded.response_text`,
          updatedAt: sql`now()`,
        },
      });
  }
}
