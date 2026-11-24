import crypto from 'crypto';
import {
  AsyncSemaphore,
  CONSTANTS,
  areEntityTypesCompatible,
  buildEntityExtractionSystemPrompt,
  buildEntityExtractionUserPrompt,
  buildSummarizeDescriptionsPrompt,
  buildTupleEntityExtractionSystemPrompt,
  buildTupleEntityExtractionUserPrompt,
  countDescriptionFragments,
  ENTITY_CONTINUE_EXTRACTION_USER_PROMPT,
  estimateTokens,
  extractEntityCode,
  mergeDescriptionFragments,
  mergeKeywords,
  parseExtractionOutput,
  resolveInDocumentEntities,
  type ChunkExtractionResult,
  type EmbeddingProvider,
  type EntityAliasMapping,
  type KgExtractFormat,
  type MergedEntityInput,
  type MergedRelationInput,
} from '@rag/core';
import type { LLMMessage, LLMProvider } from '../interfaces/llm.js';

export interface ExtractableChunk {
  id: string;
  text: string;
  sectionHeading?: string | null;
}

export interface ExistingEntitySnapshot {
  canonicalName: string;
  entityName: string;
  entityType: string;
  description: string;
  sourceChunkIds: string[];
  documentIds: string[];
  embedding?: number[] | null;
}

export interface ExistingRelationSnapshot {
  sourceCanonical: string;
  targetCanonical: string;
  sourceEntity: string;
  targetEntity: string;
  keywords: string;
  description: string;
  weight: number;
  sourceChunkIds: string[];
  documentIds: string[];
  embedding?: number[] | null;
}

export interface KgPersistenceBridge {
  getLlmCache(orgId: string, cacheKey: string): Promise<string | null>;
  setLlmCache(
    orgId: string,
    cacheKey: string,
    cacheType: 'extract' | 'keywords' | 'summary',
    model: string,
    responseText: string,
  ): Promise<void>;
  getExistingEntities(orgId: string, canonicalNames: string[]): Promise<ExistingEntitySnapshot[]>;
  getExistingRelations(orgId: string, pairs: Array<[string, string]>): Promise<ExistingRelationSnapshot[]>;
  upsertEntities(orgId: string, entities: MergedEntityInput[], embeddingModel: string): Promise<void>;
  upsertRelations(orgId: string, relations: MergedRelationInput[], embeddingModel: string): Promise<void>;
  resolveEntityAliases?(orgId: string, canonicalNames: string[]): Promise<Map<string, string>>;
  findFuzzyEntityCandidates?(
    orgId: string,
    canonicalName: string,
    minSimilarity?: number,
    limit?: number,
  ): Promise<Array<{ canonicalName: string; entityName: string; entityType: string; similarity: number }>>;
  upsertEntityAliases?(orgId: string, aliases: EntityAliasMapping[]): Promise<void>;
}

export interface KgExtractionOptions {
  maxAsync?: number;
  maxGleaning?: number;
  extractFormat?: KgExtractFormat;
  summaryFragmentThreshold?: number;
  summaryMaxTokens?: number;
  embedBatchSize?: number;
  onProgress?: (stats: { chunksProcessed: number; chunksTotal: number }) => Promise<void> | void;
}

export interface KgExtractionSummary {
  chunksProcessed: number;
  entitiesUpserted: number;
  relationsUpserted: number;
  cacheHits: number;
  llmCalls: number;
}

function resolveExtractFormat(requested: KgExtractFormat | undefined, _modelName: string): 'tuple' | 'json' {
  const mode = requested ?? ((process.env.KG_EXTRACT_FORMAT as KgExtractFormat | undefined) || 'json');
  if (mode === 'tuple') return 'tuple';
  // Live BGE-M3 + Llama-3.1-8B benchmarks showed JSON extraction produces 2.5x more relations
  // (211 vs 84) in half the latency (49s vs 105s) compared to untuned tuple prompts.
  return 'json';
}

/**
 * LightRAG+ Knowledge Graph Extraction & Incremental Merge Pipeline:
 * 1. Adaptive format selection (`tuple` for 7B-14B models, `json` for frontier models)
 *    with bounded concurrency (`AsyncSemaphore`), section breadcrumbs, persistent LLM
 *    cache, and optional gleaning.
 * 2. Deterministic sorted description merging so concurrent runs never trigger
 *    spurious re-embeddings on unchanged entities/relations.
 * 3. Automatic LLM description synthesis when fragments exceed threshold.
 * 4. Single combined batch embedding pass across BOTH entities and relations.
 */
export class KnowledgeGraphExtractor {
  private readonly extractLlm: LLMProvider;
  private readonly summaryLlm: LLMProvider;
  private readonly embedder: EmbeddingProvider;
  private readonly bridge: KgPersistenceBridge;

  constructor(deps: {
    extractLlm: LLMProvider;
    summaryLlm?: LLMProvider;
    embedder: EmbeddingProvider;
    bridge: KgPersistenceBridge;
  }) {
    this.extractLlm = deps.extractLlm;
    this.summaryLlm = deps.summaryLlm ?? deps.extractLlm;
    this.embedder = deps.embedder;
    this.bridge = deps.bridge;
  }

  private hashKey(prefix: string, payload: string): string {
    return `${prefix}:${crypto.createHash('sha256').update(payload).digest('hex')}`;
  }

  private async extractSingleChunk(
    orgId: string,
    chunk: ExtractableChunk,
    systemPrompt: string,
    format: 'tuple' | 'json',
    maxGleaning: number,
  ): Promise<{ result: ChunkExtractionResult; cacheHit: boolean; llmCalls: number }> {
    const cacheKey = this.hashKey(
      'extract',
      `${this.extractLlm.model}|${format}|g${maxGleaning}|${chunk.sectionHeading ?? ''}|${chunk.text}`,
    );

    try {
      const cached = await this.bridge.getLlmCache(orgId, cacheKey);
      if (cached) {
        return {
          result: parseExtractionOutput(cached),
          cacheHit: true,
          llmCalls: 0,
        };
      }
    } catch {
      /* cache is best-effort */
    }

    const userPrompt =
      format === 'tuple'
        ? buildTupleEntityExtractionUserPrompt(chunk.text, chunk.sectionHeading)
        : buildEntityExtractionUserPrompt(chunk.text, chunk.sectionHeading);

    const messages: LLMMessage[] = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt },
    ];

    let llmCalls = 0;
    const firstResponse = await this.extractLlm.generate(messages);
    llmCalls++;

    let merged = parseExtractionOutput(firstResponse.content);
    let combinedRaw = firstResponse.content;

    if (maxGleaning > 0 && merged.entities.length > 0) {
      const gleanMessages: LLMMessage[] = [
        ...messages,
        { role: 'assistant', content: firstResponse.content },
        { role: 'user', content: ENTITY_CONTINUE_EXTRACTION_USER_PROMPT },
      ];

      for (let pass = 0; pass < maxGleaning; pass++) {
        try {
          const gleanResponse = await this.extractLlm.generate(gleanMessages);
          llmCalls++;
          const extra = parseExtractionOutput(gleanResponse.content);
          if (extra.entities.length === 0 && extra.relations.length === 0) break;
          merged = this.combineChunkExtractions(merged, extra);
          combinedRaw = JSON.stringify({
            entities: merged.entities.map((e) => ({ name: e.name, type: e.type, description: e.description })),
            relationships: merged.relations.map((r) => ({
              source: r.source,
              target: r.target,
              keywords: r.keywords.join(', '),
              description: r.description,
            })),
          });
        } catch {
          break;
        }
      }
    }

    try {
      await this.bridge.setLlmCache(orgId, cacheKey, 'extract', this.extractLlm.model, combinedRaw);
    } catch {
      /* cache write is best-effort */
    }

    return { result: merged, cacheHit: false, llmCalls };
  }

  private combineChunkExtractions(
    base: ChunkExtractionResult,
    extra: ChunkExtractionResult,
  ): ChunkExtractionResult {
    const entityMap = new Map(base.entities.map((e) => [e.canonicalName, { ...e }]));
    for (const e of extra.entities) {
      const cur = entityMap.get(e.canonicalName);
      if (cur) {
        cur.description = mergeDescriptionFragments(cur.description, [e.description]);
        if (cur.type === 'Other' && e.type !== 'Other') cur.type = e.type;
      } else {
        entityMap.set(e.canonicalName, { ...e });
      }
    }

    const relMap = new Map(
      base.relations.map((r) => [`${r.sourceCanonical}::${r.targetCanonical}`, { ...r }]),
    );
    for (const r of extra.relations) {
      const key = `${r.sourceCanonical}::${r.targetCanonical}`;
      const cur = relMap.get(key);
      if (cur) {
        cur.weight += r.weight;
        cur.description = mergeDescriptionFragments(cur.description, [r.description]);
        cur.keywords = mergeKeywords(cur.keywords.join(', '), r.keywords)
          .split(',')
          .map((k) => k.trim())
          .filter(Boolean);
      } else {
        relMap.set(key, { ...r });
      }
    }

    return {
      entities: [...entityMap.values()],
      relations: [...relMap.values()],
    };
  }

  private async maybeSummarizeDescription(
    orgId: string,
    kind: 'Entity' | 'Relationship',
    name: string,
    description: string,
    fragmentThreshold: number,
    maxTokens: number,
  ): Promise<{ text: string; llmCalled: boolean }> {
    const fragments = description
      .split(CONSTANTS.KG_FIELD_SEP)
      .map((s) => s.trim())
      .filter(Boolean);
    if (fragments.length < fragmentThreshold && estimateTokens(description) <= maxTokens) {
      return { text: description, llmCalled: false };
    }

    const cacheKey = this.hashKey('summary', `${this.summaryLlm.model}|${kind}|${name}|${description}`);
    try {
      const cached = await this.bridge.getLlmCache(orgId, cacheKey);
      if (cached) return { text: cached, llmCalled: false };
    } catch {
      /* ignore */
    }

    try {
      const prompt = buildSummarizeDescriptionsPrompt({ kind, name, descriptions: fragments });
      const res = await this.summaryLlm.generate([{ role: 'user', content: prompt }]);
      const summarized = res.content.trim();
      if (summarized) {
        await this.bridge
          .setLlmCache(orgId, cacheKey, 'summary', this.summaryLlm.model, summarized)
          .catch(() => undefined);
        return { text: summarized, llmCalled: true };
      }
    } catch {
      /* keep raw merged description if summary LLM fails */
    }
    return { text: description, llmCalled: false };
  }

  async extractAndIndexDocument(
    orgId: string,
    documentId: string,
    chunks: ExtractableChunk[],
    options: KgExtractionOptions = {},
  ): Promise<KgExtractionSummary> {
    if (chunks.length === 0) {
      return { chunksProcessed: 0, entitiesUpserted: 0, relationsUpserted: 0, cacheHits: 0, llmCalls: 0 };
    }

    const maxAsync = options.maxAsync ?? parseInt(process.env.KG_MAX_ASYNC || String(CONSTANTS.KG_DEFAULT_MAX_ASYNC), 10);
    const maxGleaning = options.maxGleaning ?? parseInt(process.env.KG_MAX_GLEANING || String(CONSTANTS.KG_DEFAULT_MAX_GLEANING), 10);
    const format = resolveExtractFormat(options.extractFormat, this.extractLlm.model);
    const summaryThreshold = options.summaryFragmentThreshold ?? CONSTANTS.KG_FORCE_LLM_SUMMARY_ON_MERGE;
    const summaryMaxTokens = options.summaryMaxTokens ?? CONSTANTS.KG_SUMMARY_MAX_TOKENS;
    const embedBatchSize = options.embedBatchSize ?? 32;

    const systemPrompt =
      format === 'tuple'
        ? buildTupleEntityExtractionSystemPrompt()
        : buildEntityExtractionSystemPrompt();
    const semaphore = new AsyncSemaphore(maxAsync);

    let chunksProcessed = 0;
    let cacheHits = 0;
    let llmCalls = 0;

    // 1. Parallel bounded-concurrency extraction across all chunks
    const perChunkResults = await Promise.all(
      chunks.map((chunk) =>
        semaphore.run(async () => {
          try {
            const extracted = await this.extractSingleChunk(orgId, chunk, systemPrompt, format, maxGleaning);
            if (extracted.cacheHit) cacheHits++;
            llmCalls += extracted.llmCalls;
            chunksProcessed++;
            if (options.onProgress) {
              await options.onProgress({ chunksProcessed, chunksTotal: chunks.length });
            }
            return { chunkId: chunk.id, extracted: extracted.result };
          } catch (err) {
            chunksProcessed++;
            console.warn(`[KG] Extraction failed for chunk ${chunk.id}:`, err instanceof Error ? err.message : err);
            return { chunkId: chunk.id, extracted: { entities: [], relations: [] } };
          }
        }),
      ),
    );

    // 2. Aggregate within this document + Entity Resolution (#5)
    const rawEntitiesWithChunks: Array<{ entity: ChunkExtractionResult['entities'][number]; chunkId: string }> = [];
    const rawRelationsWithChunks: Array<{ relation: ChunkExtractionResult['relations'][number]; chunkId: string }> = [];

    for (const { chunkId, extracted } of perChunkResults) {
      for (const e of extracted.entities) rawEntitiesWithChunks.push({ entity: e, chunkId });
      for (const r of extracted.relations) rawRelationsWithChunks.push({ relation: r, chunkId });
    }

    const inDocResolved = resolveInDocumentEntities(
      rawEntitiesWithChunks.map((x) => x.entity),
      rawRelationsWithChunks.map((x) => x.relation),
    );
    const aliasRedirects = new Map<string, string>(
      inDocResolved.aliases.map((a) => [a.aliasCanonical, a.targetCanonical]),
    );
    const resolvedAliasesToPersist: EntityAliasMapping[] = [...inDocResolved.aliases];

    // Optional cross-document alias & pg_trgm fuzzy entity resolution against existing tenant KG
    if (this.bridge.resolveEntityAliases) {
      const existingAliasMap = await this.bridge
        .resolveEntityAliases(
          orgId,
          inDocResolved.entities.map((e) => e.canonicalName),
        )
        .catch(() => new Map<string, string>());
      for (const [alias, target] of existingAliasMap.entries()) {
        aliasRedirects.set(alias, target);
      }
    }

    if (this.bridge.findFuzzyEntityCandidates) {
      const candidateLookups = inDocResolved.entities.filter(
        (e) => e.canonicalName.length >= 6 && !aliasRedirects.has(e.canonicalName),
      );
      await Promise.all(
        candidateLookups.map((e) =>
          semaphore.run(async () => {
            const matches = await this.bridge
              .findFuzzyEntityCandidates!(orgId, e.canonicalName, 0.78, 3)
              .catch(() => []);
            const eCode = extractEntityCode(e.name);
            for (const m of matches) {
              if (m.canonicalName === e.canonicalName) break;
              if (!areEntityTypesCompatible(m.entityType, e.type)) continue;
              const mCode = extractEntityCode(m.entityName);
              if (eCode && mCode && eCode !== mCode) continue;
              aliasRedirects.set(e.canonicalName, m.canonicalName);
              resolvedAliasesToPersist.push({
                aliasCanonical: e.canonicalName,
                aliasName: e.name,
                targetCanonical: m.canonicalName,
                similarity: m.similarity,
              });
              break;
            }
          }),
        ),
      );
    }

    const resolveCanonical = (c: string): string => {
      let cur = c;
      for (let hops = 0; hops < 4; hops++) {
        const next = aliasRedirects.get(cur);
        if (!next || next === cur) break;
        cur = next;
      }
      return cur;
    };

    const entityAcc = new Map<string, MergedEntityInput>();
    const relAcc = new Map<string, MergedRelationInput>();

    for (const { chunkId, entity: e } of rawEntitiesWithChunks) {
      const targetCanonical = resolveCanonical(e.canonicalName);
      const cur = entityAcc.get(targetCanonical);
      if (cur) {
        cur.description = mergeDescriptionFragments(cur.description, [e.description]);
        if (cur.entityType === 'Other' && e.type !== 'Other') cur.entityType = e.type;
        cur.sourceChunkIds.push(chunkId);
      } else {
        entityAcc.set(targetCanonical, {
          entityName: e.name,
          canonicalName: targetCanonical,
          entityType: e.type,
          description: mergeDescriptionFragments(null, [e.description]),
          sourceChunkIds: [chunkId],
          documentIds: [documentId],
        });
      }
    }

    for (const { chunkId, relation: r } of rawRelationsWithChunks) {
      let srcC = resolveCanonical(r.sourceCanonical);
      let tgtC = resolveCanonical(r.targetCanonical);
      if (!srcC || !tgtC || srcC === tgtC) continue;
      let srcName = entityAcc.get(srcC)?.entityName ?? r.source;
      let tgtName = entityAcc.get(tgtC)?.entityName ?? r.target;
      if (srcC > tgtC) {
        [srcC, tgtC] = [tgtC, srcC];
        [srcName, tgtName] = [tgtName, srcName];
      }
      const key = `${srcC}::${tgtC}`;
      const cur = relAcc.get(key);
      if (cur) {
        cur.weight += r.weight;
        cur.keywords = mergeKeywords(cur.keywords, r.keywords);
        cur.description = mergeDescriptionFragments(cur.description, [r.description]);
        cur.sourceChunkIds.push(chunkId);
      } else {
        relAcc.set(key, {
          sourceEntity: srcName,
          targetEntity: tgtName,
          sourceCanonical: srcC,
          targetCanonical: tgtC,
          keywords: mergeKeywords(null, r.keywords),
          description: mergeDescriptionFragments(null, [r.description]),
          weight: r.weight,
          sourceChunkIds: [chunkId],
          documentIds: [documentId],
        });
      }
    }

    if (entityAcc.size === 0 && relAcc.size === 0) {
      return { chunksProcessed, entitiesUpserted: 0, relationsUpserted: 0, cacheHits, llmCalls };
    }

    // 3. Incremental merge with existing graph records in Postgres
    const existingEntities = await this.bridge.getExistingEntities(orgId, [...entityAcc.keys()]);
    const existingEntityEmbedText = new Map<string, string>();
    for (const existing of existingEntities) {
      const incoming = entityAcc.get(existing.canonicalName);
      if (!incoming) continue;
      const canonicalExistingDesc = mergeDescriptionFragments(existing.description, []);
      if (existing.embedding && existing.embedding.length > 0) {
        existingEntityEmbedText.set(
          existing.canonicalName,
          `${existing.entityName} (${existing.entityType})\n${canonicalExistingDesc.replaceAll(CONSTANTS.KG_FIELD_SEP, '; ')}`,
        );
      }
      incoming.description = mergeDescriptionFragments(canonicalExistingDesc, [incoming.description]);
      if (incoming.entityType === 'Other' && existing.entityType !== 'Other') {
        incoming.entityType = existing.entityType;
      }
      incoming.sourceChunkIds = [...new Set([...existing.sourceChunkIds, ...incoming.sourceChunkIds])];
      incoming.documentIds = [...new Set([...existing.documentIds, ...incoming.documentIds])];
    }

    const relPairs: Array<[string, string]> = [...relAcc.values()].map((r) => [
      r.sourceCanonical,
      r.targetCanonical,
    ]);
    const existingRelations = await this.bridge.getExistingRelations(orgId, relPairs);
    const existingRelationEmbedText = new Map<string, string>();
    for (const existing of existingRelations) {
      const key = `${existing.sourceCanonical}::${existing.targetCanonical}`;
      const incoming = relAcc.get(key);
      if (!incoming) continue;
      const canonicalExistingDesc = mergeDescriptionFragments(existing.description, []);
      if (existing.embedding && existing.embedding.length > 0) {
        existingRelationEmbedText.set(
          key,
          `${mergeKeywords(existing.keywords, [])}\t${existing.sourceEntity} - ${existing.targetEntity}\n${canonicalExistingDesc.replaceAll(CONSTANTS.KG_FIELD_SEP, '; ')}`,
        );
      }
      incoming.weight += existing.weight;
      incoming.keywords = mergeKeywords(existing.keywords, [incoming.keywords]);
      incoming.description = mergeDescriptionFragments(canonicalExistingDesc, [incoming.description]);
      incoming.sourceChunkIds = [...new Set([...existing.sourceChunkIds, ...incoming.sourceChunkIds])];
      incoming.documentIds = [...new Set([...existing.documentIds, ...incoming.documentIds])];
    }

    // 4. Optional LLM summarization for entities/relations with many accumulated fragments
    const entitiesList = [...entityAcc.values()];
    const relationsList = [...relAcc.values()];

    await Promise.all([
      ...entitiesList
        .filter((e) => countDescriptionFragments(e.description) >= summaryThreshold || estimateTokens(e.description) > summaryMaxTokens)
        .map((e) =>
          semaphore.run(async () => {
            const res = await this.maybeSummarizeDescription(
              orgId,
              'Entity',
              e.entityName,
              e.description,
              summaryThreshold,
              summaryMaxTokens,
            );
            e.description = res.text;
            if (res.llmCalled) llmCalls++;
          }),
        ),
      ...relationsList
        .filter((r) => countDescriptionFragments(r.description) >= summaryThreshold || estimateTokens(r.description) > summaryMaxTokens)
        .map((r) =>
          semaphore.run(async () => {
            const res = await this.maybeSummarizeDescription(
              orgId,
              'Relationship',
              `${r.sourceEntity} - ${r.targetEntity}`,
              r.description,
              summaryThreshold,
              summaryMaxTokens,
            );
            r.description = res.text;
            if (res.llmCalled) llmCalls++;
          }),
        ),
    ]);

    // 5. Combined single-pass batch embedding across BOTH entities and relations (halves HTTP round-trips)
    const embedTasks: Array<{ text: string; assign: (vec: number[]) => void }> = [];
    for (const e of entitiesList) {
      const text = `${e.entityName} (${e.entityType})\n${e.description.replaceAll(CONSTANTS.KG_FIELD_SEP, '; ')}`;
      if (existingEntityEmbedText.get(e.canonicalName) === text) continue;
      embedTasks.push({ text, assign: (vec) => { e.embedding = vec; } });
    }
    for (const r of relationsList) {
      const key = `${r.sourceCanonical}::${r.targetCanonical}`;
      const text = `${r.keywords}\t${r.sourceEntity} - ${r.targetEntity}\n${r.description.replaceAll(CONSTANTS.KG_FIELD_SEP, '; ')}`;
      if (existingRelationEmbedText.get(key) === text) continue;
      embedTasks.push({ text, assign: (vec) => { r.embedding = vec; } });
    }

    for (let i = 0; i < embedTasks.length; i += embedBatchSize) {
      const batch = embedTasks.slice(i, i + embedBatchSize);
      const embedRes = await this.embedder.embed(batch.map((b) => b.text));
      for (let j = 0; j < batch.length; j++) {
        const task = batch[j];
        const vec = embedRes.embeddings[j];
        if (task && vec) task.assign(vec);
      }
    }

    // 6. Upsert merged KG & resolved aliases into PostgreSQL
    await this.bridge.upsertEntities(orgId, entitiesList, this.embedder.model);
    await this.bridge.upsertRelations(orgId, relationsList, this.embedder.model);
    if (this.bridge.upsertEntityAliases && resolvedAliasesToPersist.length > 0) {
      await this.bridge.upsertEntityAliases(orgId, resolvedAliasesToPersist).catch(() => undefined);
    }

    return {
      chunksProcessed,
      entitiesUpserted: entitiesList.length,
      relationsUpserted: relationsList.length,
      cacheHits,
      llmCalls,
    };
  }
}
