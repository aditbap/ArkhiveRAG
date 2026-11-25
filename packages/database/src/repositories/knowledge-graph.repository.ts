import { BaseRepository } from './base.repository.js';
import { kgEntities, kgEntityAliases, kgRelations, llmCache } from '../schema/knowledge-graph.js';
import { documents } from '../schema/documents.js';
import { HNSW_EF_SEARCH_DEFAULT, withHnswSearch } from '../client.js';
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

function extractRows<T extends Record<string, unknown>>(res: unknown): T[] {
  if (Array.isArray(res)) return res as T[];
  if (res && typeof res === 'object' && 'rows' in res && Array.isArray((res as { rows: unknown[] }).rows)) {
    return (res as { rows: T[] }).rows;
  }
  return [];
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
    const rows = await this.readDb
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
        confidence: typeof e.confidence === 'number' ? Math.max(0.1, Math.min(1, e.confidence)) : 0.85,
        normalizedAliases: e.normalizedAliases ?? [],
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
            confidence: sql`GREATEST(${kgEntities.confidence}, excluded.confidence)`,
            normalizedAliases: sql`excluded.normalized_aliases`,
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
          relationType: (r.relationType || 'related').slice(0, 32),
          isDirected: Boolean(r.isDirected),
          confidence: typeof r.confidence === 'number' ? Math.max(0.1, Math.min(1, r.confidence)) : 0.85,
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
            relationType: sql`CASE WHEN excluded.relation_type <> 'related' THEN excluded.relation_type ELSE ${kgRelations.relationType} END`,
            isDirected: sql`excluded.is_directed OR ${kgRelations.isDirected}`,
            confidence: sql`GREATEST(${kgRelations.confidence}, excluded.confidence)`,
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
   * Local retrieval branch: HNSW-accelerated vector similarity search over `kg_entities`
   * ordered by raw cosine distance `embedding <=> vec ASC` with entity degree as secondary tie-breaker.
   */
  async searchEntities(
    orgId: string,
    embedding: number[],
    limit = 20,
    minSimilarity = 0.2,
    documentIds?: string[],
  ): Promise<RetrievedEntity[]> {
    this.requireTenant(orgId);
    const vecLiteral = JSON.stringify(embedding);
    const maxDistance = Math.max(0, Math.min(2, 1 - minSimilarity));
    const overfetch = Math.min(Math.max(limit * 4, 48), 250);
    const efSearch = Math.max(HNSW_EF_SEARCH_DEFAULT, overfetch);
    const docFilterSql =
      documentIds && documentIds.length > 0
        ? sql`AND document_ids && ${uuidArraySql(documentIds)}`
        : sql``;

    const rows = await withHnswSearch(this.readDb, efSearch, async (tx) => {
      const res = await tx.execute(sql`
        WITH hnsw_candidates AS (
          SELECT
            id, entity_name, canonical_name, entity_type, description,
            degree, source_chunk_ids, document_ids,
            (embedding <=> ${vecLiteral}::vector) AS distance
          FROM kg_entities
          WHERE org_id = ${orgId}::uuid
            AND embedding IS NOT NULL
            ${docFilterSql}
          ORDER BY embedding <=> ${vecLiteral}::vector ASC
          LIMIT ${overfetch}
        )
        SELECT
          id, entity_name, canonical_name, entity_type, description,
          degree, source_chunk_ids, document_ids,
          (1.0 - distance)::float8 AS similarity
        FROM hnsw_candidates
        WHERE distance <= ${maxDistance}
        ORDER BY distance ASC, degree DESC
        LIMIT ${limit}
      `);
      return extractRows<Record<string, unknown>>(res);
    });

    return rows.map((r) => ({
      id: String(r.id),
      entityName: String(r.entity_name),
      canonicalName: String(r.canonical_name),
      entityType: String(r.entity_type),
      description: String(r.description),
      degree: Number(r.degree),
      sourceChunkIds: Array.isArray(r.source_chunk_ids) ? (r.source_chunk_ids as string[]) : [],
      documentIds: Array.isArray(r.document_ids) ? (r.document_ids as string[]) : [],
      similarity: Number(r.similarity),
    }));
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

    const rows = await this.readDb
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
   * Global retrieval branch: HNSW-accelerated vector similarity search over `kg_relations`
   * ordered by raw cosine distance `embedding <=> vec ASC` with relation weight as secondary tie-breaker.
   */
  async searchRelations(
    orgId: string,
    embedding: number[],
    limit = 20,
    minSimilarity = 0.2,
    documentIds?: string[],
  ): Promise<RetrievedRelation[]> {
    this.requireTenant(orgId);
    const vecLiteral = JSON.stringify(embedding);
    const maxDistance = Math.max(0, Math.min(2, 1 - minSimilarity));
    const overfetch = Math.min(Math.max(limit * 4, 48), 250);
    const efSearch = Math.max(HNSW_EF_SEARCH_DEFAULT, overfetch);
    const docFilterSql =
      documentIds && documentIds.length > 0
        ? sql`AND document_ids && ${uuidArraySql(documentIds)}`
        : sql``;

    const rows = await withHnswSearch(this.readDb, efSearch, async (tx) => {
      const res = await tx.execute(sql`
        WITH hnsw_candidates AS (
          SELECT
            id, source_entity, target_entity, source_canonical, target_canonical,
            keywords, description, weight, source_chunk_ids, document_ids,
            (embedding <=> ${vecLiteral}::vector) AS distance
          FROM kg_relations
          WHERE org_id = ${orgId}::uuid
            AND embedding IS NOT NULL
            ${docFilterSql}
          ORDER BY embedding <=> ${vecLiteral}::vector ASC
          LIMIT ${overfetch}
        )
        SELECT
          id, source_entity, target_entity, source_canonical, target_canonical,
          keywords, description, weight, source_chunk_ids, document_ids,
          (1.0 - distance)::float8 AS similarity
        FROM hnsw_candidates
        WHERE distance <= ${maxDistance}
        ORDER BY distance ASC, weight DESC
        LIMIT ${limit}
      `);
      return extractRows<Record<string, unknown>>(res);
    });

    return rows.map((r) => ({
      id: String(r.id),
      sourceEntity: String(r.source_entity),
      targetEntity: String(r.target_entity),
      sourceCanonical: String(r.source_canonical),
      targetCanonical: String(r.target_canonical),
      keywords: String(r.keywords),
      description: String(r.description),
      weight: Number(r.weight),
      sourceChunkIds: Array.isArray(r.source_chunk_ids) ? (r.source_chunk_ids as string[]) : [],
      documentIds: Array.isArray(r.document_ids) ? (r.document_ids as string[]) : [],
      similarity: Number(r.similarity),
    }));
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

    const rows = await this.readDb
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
    const rows = await this.readDb
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

    const rows = await this.readDb
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
   * Reverse provenance lookup: fetches KG entities anchored in a set of high-confidence
   * chunk IDs for Anchor-Guided Pseudo-Relevance Graph Bridging (PRGB).
   * Strictly tenant-scoped and document-ACL filtered, ordered by specificity (rarest doc spread first).
   */
  async getEntitiesBySourceChunkIds(
    orgId: string,
    chunkIds: string[],
    limit = 24,
    documentIds?: string[],
  ): Promise<RetrievedEntity[]> {
    this.requireTenant(orgId);
    const unique = [...new Set(chunkIds.filter(Boolean))];
    if (unique.length === 0) return [];

    const conditions = [
      eq(kgEntities.orgId, orgId),
      sql`${kgEntities.sourceChunkIds} && ${uuidArraySql(unique)}`,
    ];
    if (documentIds && documentIds.length > 0) {
      conditions.push(sql`${kgEntities.documentIds} && ${uuidArraySql(documentIds)}`);
    }

    const docSpread = sql<number>`COALESCE(array_length(${kgEntities.documentIds}, 1), 1)`;
    const rows = await this.readDb
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
      .where(and(...conditions))
      .orderBy(docSpread, kgEntities.degree)
      .limit(limit);

    return rows.map((r) => ({ ...r, similarity: 0.75 }));
  }

  /**
   * Resolves alias canonical names to their surviving canonical entity targets (#5 Entity Resolution).
   */
  async resolveEntityAliases(orgId: string, canonicalNames: string[]): Promise<Map<string, string>> {
    this.requireTenant(orgId);
    const out = new Map<string, string>();
    const unique = [...new Set(canonicalNames.filter(Boolean))];
    if (unique.length === 0) return out;

    const rows = await this.readDb
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
    const rows = await this.readDb
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

      const rows = await this.readDb
        .select({
          sourceCanonical: kgRelations.sourceCanonical,
          targetCanonical: kgRelations.targetCanonical,
          weight: kgRelations.weight,
          relationType: kgRelations.relationType,
          isDirected: kgRelations.isDirected,
          confidence: kgRelations.confidence,
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
            relationType: r.relationType,
            isDirected: r.isDirected,
            confidence: r.confidence,
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

    const entityRows = await this.readDb
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
    const rows = await this.readDb
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

    // Clean up orphaned alias pointers and dangling relations referencing removed entities
    await this.db.execute(sql`
      DELETE FROM kg_entity_aliases a
      WHERE a.org_id = ${orgId}::uuid
        AND NOT EXISTS (
          SELECT 1 FROM kg_entities e
          WHERE e.org_id = a.org_id AND e.canonical_name = a.target_canonical
        )
    `);

    await this.db.execute(sql`
      DELETE FROM kg_relations r
      WHERE r.org_id = ${orgId}::uuid
        AND (
          NOT EXISTS (SELECT 1 FROM kg_entities e WHERE e.org_id = r.org_id AND e.canonical_name = r.source_canonical)
          OR NOT EXISTS (SELECT 1 FROM kg_entities e WHERE e.org_id = r.org_id AND e.canonical_name = r.target_canonical)
        )
    `);

    await this.refreshEntityDegrees(orgId);
  }

  /**
   * Automatic cosine-similarity synonymy edge discovery (HippoRAG-style) using pgvector:
   * links semantically synonymous entities (`sim >= minSimilarity`, default 0.82) across
   * documents so PPR bridges multilingual aliases and codenames without merging descriptions.
   */
  async upsertCosineSynonymyEdges(
    orgId: string,
    entities: MergedEntityInput[],
    embeddingModel: string,
    minSimilarity = 0.82,
    maxPerEntity = 2,
  ): Promise<number> {
    this.requireTenant(orgId);
    const synonymEdges: MergedRelationInput[] = [];
    const seen = new Set<string>();

    for (const e of entities) {
      if (!e.embedding || e.embedding.length === 0) continue;
      const neighbors = await this.searchEntities(orgId, e.embedding, maxPerEntity + 2, minSimilarity).catch(() => []);
      let added = 0;
      for (const n of neighbors) {
        if (n.canonicalName === e.canonicalName) continue;
        if (added >= maxPerEntity) break;
        const [sC, tC, sN, tN] =
          e.canonicalName < n.canonicalName
            ? [e.canonicalName, n.canonicalName, e.entityName, n.entityName]
            : [n.canonicalName, e.canonicalName, n.entityName, e.entityName];
        const key = `${sC}::${tC}`;
        if (seen.has(key)) continue;
        seen.add(key);
        synonymEdges.push({
          sourceEntity: sN,
          targetEntity: tN,
          sourceCanonical: sC,
          targetCanonical: tC,
          keywords: 'synonym, alias',
          description: `Semantic synonymy edge (cosine=${n.similarity.toFixed(3)})`,
          weight: Number((n.similarity * 0.85).toFixed(4)),
          relationType: 'synonym',
          isDirected: false,
          confidence: Number(n.similarity.toFixed(4)),
          sourceChunkIds: capUuidList([...e.sourceChunkIds, ...n.sourceChunkIds], 6),
          documentIds: capUuidList([...e.documentIds, ...n.documentIds], 6),
        });
        added++;
      }
    }

    if (synonymEdges.length > 0) {
      await this.upsertRelations(orgId, synonymEdges, embeddingModel);
    }
    return synonymEdges.length;
  }

  /**
   * Graph integrity audit: counts orphan entities, dangling relation endpoints, self-loops,
   * and stale entity aliases for tenant health monitoring.
   */
  async checkGraphIntegrity(orgId: string): Promise<{
    orphanEntities: number;
    danglingRelations: number;
    selfLoops: number;
    staleAliases: number;
    healthy: boolean;
  }> {
    this.requireTenant(orgId);
    const [orphans, dangling, selfLoops, stale] = await Promise.all([
      this.db.execute(sql`
        SELECT count(*)::int AS cnt FROM kg_entities
        WHERE org_id = ${orgId}::uuid AND COALESCE(array_length(document_ids, 1), 0) = 0
      `),
      this.db.execute(sql`
        SELECT count(*)::int AS cnt FROM kg_relations r
        WHERE r.org_id = ${orgId}::uuid AND (
          NOT EXISTS (SELECT 1 FROM kg_entities e WHERE e.org_id = r.org_id AND e.canonical_name = r.source_canonical)
          OR NOT EXISTS (SELECT 1 FROM kg_entities e WHERE e.org_id = r.org_id AND e.canonical_name = r.target_canonical)
        )
      `),
      this.db.execute(sql`
        SELECT count(*)::int AS cnt FROM kg_relations
        WHERE org_id = ${orgId}::uuid AND source_canonical = target_canonical
      `),
      this.db.execute(sql`
        SELECT count(*)::int AS cnt FROM kg_entity_aliases a
        WHERE a.org_id = ${orgId}::uuid AND NOT EXISTS (
          SELECT 1 FROM kg_entities e WHERE e.org_id = a.org_id AND e.canonical_name = a.target_canonical
        )
      `),
    ]);

    const readCnt = (res: unknown): number => {
      const rows = Array.isArray(res) ? res : (res as { rows?: Array<{ cnt?: number }> })?.rows ?? [];
      return Number((rows[0] as { cnt?: number } | undefined)?.cnt ?? 0);
    };

    const orphanEntities = readCnt(orphans);
    const danglingRelations = readCnt(dangling);
    const selfLoopCount = readCnt(selfLoops);
    const staleAliases = readCnt(stale);

    return {
      orphanEntities,
      danglingRelations,
      selfLoops: selfLoopCount,
      staleAliases,
      healthy: orphanEntities === 0 && danglingRelations === 0 && selfLoopCount === 0 && staleAliases === 0,
    };
  }

  /** Prunes orphan entities, dangling edges, self-loops, and stale aliases, then recomputes degrees. */
  async pruneGraphOrphans(orgId: string): Promise<void> {
    this.requireTenant(orgId);
    await this.db.execute(sql`
      DELETE FROM kg_relations WHERE org_id = ${orgId}::uuid AND source_canonical = target_canonical
    `);
    await this.db.execute(sql`
      DELETE FROM kg_entities WHERE org_id = ${orgId}::uuid AND COALESCE(array_length(document_ids, 1), 0) = 0
    `);
    await this.db.execute(sql`
      DELETE FROM kg_relations r
      WHERE r.org_id = ${orgId}::uuid AND (
        NOT EXISTS (SELECT 1 FROM kg_entities e WHERE e.org_id = r.org_id AND e.canonical_name = r.source_canonical)
        OR NOT EXISTS (SELECT 1 FROM kg_entities e WHERE e.org_id = r.org_id AND e.canonical_name = r.target_canonical)
      )
    `);
    await this.db.execute(sql`
      DELETE FROM kg_entity_aliases a
      WHERE a.org_id = ${orgId}::uuid AND NOT EXISTS (
        SELECT 1 FROM kg_entities e WHERE e.org_id = a.org_id AND e.canonical_name = a.target_canonical
      )
    `);
    await this.refreshEntityDegrees(orgId);
  }

  async getLlmCache(orgId: string, cacheKey: string): Promise<string | null> {
    this.requireTenant(orgId);
    const rows = await this.readDb
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
