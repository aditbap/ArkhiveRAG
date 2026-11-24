import crypto from 'crypto';
import {
  CONSTANTS,
  DEFAULT_RETRIEVAL_CONFIG,
  assembleLightRagContext,
  buildSlimKeywordsPrompt,
  buildUnifiedDecomposePrompt,
  buildUnifiedQueryAnalysisPrompt,
  classifyQueryIntent,
  computeExactCodeMatchScore,
  computeHeadingMatchScore,
  computePersonalizedPageRank,
  evaluateCragConfidence,
  extractSignificantTerms,
  parseDecomposeOutput,
  parseQueryAnalysisOutput,
  parseTemporalIntent,
  propagatePprToChunks,
  resolveTemporalAndAuthorityConflicts,
  roundRobinMerge,
  scoreCandidateWithLtr,
  shouldDecomposeQuery,
  weightedReciprocalRankFusion,
  type AssembledContext,
  type ConcreteRagQueryMode,
  type ContextChunkInput,
  type CragEvaluation,
  type EmbeddingProvider,
  type QueryAnalysisResult,
  type QueryIntent,
  type RagQueryMode,
  type RerankerProvider,
  type RetrievalConfigPayload,
  type RetrievalFeatureVector,
  type RetrievalStrategyName,
  type RetrievedEntity,
  type RetrievedRelation,
  type ScoredChunk,
  type TokenBudgetConfig,
} from '@rag/core';
import { ChunkRepository, KnowledgeGraphRepository } from '@rag/database';
import type { LLMProvider } from '@rag/providers';

export interface CandidateFeatureRecord {
  chunkId: string;
  documentId: string;
  finalScore: number;
  features: RetrievalFeatureVector;
}

export interface CommunitySummaryHit {
  id: string;
  level: number;
  title: string;
  summary: string;
  documentIds: string[];
  similarity: number;
}

export interface CommunitySearchBridge {
  searchCommunities(
    orgId: string,
    embedding: number[],
    limit?: number,
    minSimilarity?: number,
    documentIds?: string[],
  ): Promise<CommunitySummaryHit[]>;
}

export interface LightRagRetrievalInput {
  orgId: string;
  userQuery: string;
  history?: Array<{ role: string; content: string }>;
  mode?: RagQueryMode;
  documentIds?: string[];
  topK?: number;
  chunkTopK?: number;
  rerankLimit?: number;
  rerankMinScore?: number;
  cosineThreshold?: number;
  budget?: Partial<TokenBudgetConfig>;
  /** Optional active config from RetrievalLearningEngine (defaults to DEFAULT_RETRIEVAL_CONFIG). */
  retrievalConfig?: RetrievalConfigPayload;
  /** Optional explicit strategy override (used by ShadowEvaluatorWorker). */
  strategyOverride?: RetrievalStrategyName;
  /** Disable CRAG retry loop when benchmarking a fixed single-pass strategy. */
  disableCragRetry?: boolean;
}

export interface LightRagRetrievalResult {
  requestedMode: RagQueryMode;
  effectiveMode: ConcreteRagQueryMode;
  routingReason: string;
  intent: QueryIntent;
  strategy: RetrievalStrategyName;
  configVersion: number;
  subQueries: string[];
  crag: CragEvaluation;
  communitiesUsed: CommunitySummaryHit[];
  candidateFeatures: CandidateFeatureRecord[];
  analysis: QueryAnalysisResult;
  assembled: AssembledContext;
  timingsMs: {
    queryAnalysis: number;
    embedding: number;
    retrieval: number;
    ppr: number;
    rerank: number;
    total: number;
  };
}

export interface FastCacheBridge {
  get(key: string): Promise<string | null>;
  setex(key: string, ttlSeconds: number, value: string): Promise<unknown>;
}

export interface LightRagRetrieverDeps {
  chunkRepo: ChunkRepository;
  kgRepo: KnowledgeGraphRepository;
  embedder: EmbeddingProvider;
  keywordsLlm: LLMProvider;
  reranker?: RerankerProvider | null;
  fastCache?: FastCacheBridge | null;
  communityRepo?: CommunitySearchBridge | null;
}

interface InternalCandidate extends ContextChunkInput {
  denseSim: number;
  bm25Norm: number;
  entityHit: number;
  relationHit: number;
  pprScore: number;
  maxsimScore: number;
  crossEncoderScore: number;
  siblingHits: number;
  createdAt?: Date | string | null;
  updatedAt?: Date | string | null;
}

export class LightRagRetriever {
  private readonly chunkRepo: ChunkRepository;
  private readonly kgRepo: KnowledgeGraphRepository;
  private readonly embedder: EmbeddingProvider;
  private readonly keywordsLlm: LLMProvider;
  private readonly reranker: RerankerProvider | null;
  private readonly fastCache: FastCacheBridge | null;
  private readonly communityRepo: CommunitySearchBridge | null;

  constructor(deps: LightRagRetrieverDeps) {
    this.chunkRepo = deps.chunkRepo;
    this.kgRepo = deps.kgRepo;
    this.embedder = deps.embedder;
    this.keywordsLlm = deps.keywordsLlm;
    this.reranker = deps.reranker ?? null;
    this.fastCache = deps.fastCache ?? null;
    this.communityRepo = deps.communityRepo ?? null;
  }

  private async analyzeQuery(
    orgId: string,
    userQuery: string,
    history: Array<{ role: string; content: string }>,
    needKeywords: boolean,
    wantDecompose: boolean,
  ): Promise<{ analysis: QueryAnalysisResult; subQueries: string[] }> {
    if (!needKeywords && !wantDecompose && history.length === 0) {
      return {
        analysis: {
          standaloneQuery: userQuery,
          highLevelKeywords: [],
          lowLevelKeywords: [],
          usedFallback: false,
        },
        subQueries: [userQuery],
      };
    }

    const singleTurn = history.length === 0;
    const prefix = wantDecompose ? 'decomp1' : 'kw2';
    const cacheKey = singleTurn
      ? `${prefix}:${crypto.createHash('sha256').update(`${this.keywordsLlm.model}|${userQuery.toLowerCase().trim()}`).digest('hex')}`
      : '';

    const parseRaw = (raw: string): { analysis: QueryAnalysisResult; subQueries: string[] } => {
      if (wantDecompose) {
        const d = parseDecomposeOutput(raw, userQuery);
        return {
          analysis: {
            standaloneQuery: d.standaloneQuery,
            highLevelKeywords: d.highLevelKeywords,
            lowLevelKeywords: d.lowLevelKeywords,
            usedFallback: d.usedFallback,
          },
          subQueries: d.subQueries,
        };
      }
      const parsed = parseQueryAnalysisOutput(raw, userQuery);
      return { analysis: parsed, subQueries: [parsed.standaloneQuery || userQuery] };
    };

    if (singleTurn) {
      if (this.fastCache) {
        try {
          const hit = await this.fastCache.get(`rag:${orgId}:${cacheKey}`);
          if (hit) return parseRaw(hit);
        } catch {
          /* best-effort */
        }
      }
      try {
        const cached = await this.kgRepo.getLlmCache(orgId, cacheKey);
        if (cached) {
          if (this.fastCache) {
            void this.fastCache.setex(`rag:${orgId}:${cacheKey}`, 86400, cached).catch(() => undefined);
          }
          return parseRaw(cached);
        }
      } catch {
        /* best-effort */
      }
    }

    try {
      const prompt = wantDecompose
        ? buildUnifiedDecomposePrompt({ query: userQuery, history })
        : singleTurn
          ? buildSlimKeywordsPrompt(userQuery)
          : buildUnifiedQueryAnalysisPrompt({ query: userQuery, history });

      const res = await this.keywordsLlm.generate([{ role: 'user', content: prompt }]);
      const result = parseRaw(res.content);
      if (singleTurn && !result.analysis.usedFallback) {
        void this.kgRepo
          .setLlmCache(orgId, cacheKey, 'keywords', this.keywordsLlm.model, res.content)
          .catch(() => undefined);
        if (this.fastCache) {
          void this.fastCache.setex(`rag:${orgId}:${cacheKey}`, 86400, res.content).catch(() => undefined);
        }
      }
      return result;
    } catch {
      return {
        analysis: {
          standaloneQuery: userQuery,
          highLevelKeywords: [],
          lowLevelKeywords: [],
          usedFallback: true,
        },
        subQueries: [userQuery],
      };
    }
  }

  private async batchEmbedQueries(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    if (this.embedder.embedQueries) {
      return this.embedder.embedQueries(texts);
    }
    return Promise.all(texts.map((t) => this.embedder.embedQuery(t)));
  }

  private async runHybridChunkSearch(
    orgId: string,
    queryText: string,
    queryEmbedding: number[],
    chunkTopK: number,
    cosineThreshold: number,
    denseWeight: number,
    bm25Weight: number,
    documentIds?: string[],
  ): Promise<InternalCandidate[]> {
    const [denseResults, ftsResults] = await Promise.all([
      this.chunkRepo.searchSimilar(orgId, queryEmbedding, chunkTopK, cosineThreshold, documentIds),
      this.chunkRepo.searchFullText(orgId, queryText, chunkTopK, documentIds),
    ]);

    const denseById = new Map(denseResults.map((r) => [r.chunk.id, r.similarity]));
    const maxFtsRank = Math.max(1e-6, ...ftsResults.map((r) => Number(r.rank) || 0));
    const ftsById = new Map(ftsResults.map((r) => [r.chunk.id, (Number(r.rank) || 0) / maxFtsRank]));

    type ChunkWithDoc = {
      chunk: (typeof denseResults)[number]['chunk'];
      document: (typeof denseResults)[number]['document'];
    };

    const fused = weightedReciprocalRankFusion<ChunkWithDoc>(
      [
        { items: denseResults, weight: denseWeight },
        { items: ftsResults, weight: bm25Weight },
      ],
      (item) => item.chunk.id,
      60,
    );

    return fused.slice(0, chunkTopK).map(({ item, score }) => {
      const doc = item.document as typeof item.document & {
        effectiveAt?: Date | string | null;
        expiresAt?: Date | string | null;
        supersedesDocumentId?: string | null;
        authorityTier?: number | null;
      };
      const ch = item.chunk as typeof item.chunk & {
        parentChunkId?: string | null;
        chunkRole?: 'parent' | 'child' | 'standalone';
      };
      const denseSim = denseById.get(item.chunk.id) ?? 0;
      const bm25Norm = ftsById.get(item.chunk.id) ?? 0;
      return {
        id: item.chunk.id,
        documentId: item.document.id,
        documentTitle: item.document.title,
        fileName: item.document.fileName,
        fileType: item.document.fileType,
        content: item.chunk.content,
        sectionHeading: item.chunk.sectionHeading,
        pageNumber: item.chunk.pageNumber,
        parentChunkId: ch.parentChunkId ?? null,
        chunkRole: ch.chunkRole ?? 'standalone',
        effectiveAt: doc.effectiveAt ?? null,
        expiresAt: doc.expiresAt ?? null,
        supersedesDocumentId: doc.supersedesDocumentId ?? null,
        authorityTier: doc.authorityTier ?? null,
        createdAt: item.document.createdAt,
        updatedAt: item.document.updatedAt,
        similarity: score,
        source: 'hybrid' as const,
        denseSim,
        bm25Norm,
        entityHit: 0,
        relationHit: 0,
        pprScore: 0,
        maxsimScore: denseSim,
        crossEncoderScore: 0,
        siblingHits: 1,
      };
    });
  }

  async retrieve(input: LightRagRetrievalInput): Promise<LightRagRetrievalResult> {
    return this.executeRetrievalPass(input, false);
  }

  private async executeRetrievalPass(
    input: LightRagRetrievalInput,
    isCragRetry: boolean,
  ): Promise<LightRagRetrievalResult> {
    const totalStart = Date.now();
    const config = input.retrievalConfig ?? DEFAULT_RETRIEVAL_CONFIG;
    const requestedMode: RagQueryMode = input.mode ?? CONSTANTS.DEFAULT_RAG_MODE;
    const history = input.history ?? [];

    // 1. Sub-1ms Intent Classification & Strategy Resolution
    const plan = classifyQueryIntent(requestedMode, input.userQuery, history, config);
    const intent = plan.intent;
    const profile = plan.profile;
    const strategy: RetrievalStrategyName = input.strategyOverride ?? plan.strategy;

    const topK = input.topK ?? CONSTANTS.DEFAULT_KG_TOP_K;
    const chunkTopK = input.chunkTopK ?? CONSTANTS.DEFAULT_CHUNK_TOP_K;
    const rerankLimit = input.rerankLimit ?? profile.rerankLimit ?? CONSTANTS.DEFAULT_RERANK_LIMIT;
    const rerankMinScore = input.rerankMinScore ?? profile.minRelevanceScore ?? 0.1;
    const cosineThreshold = input.cosineThreshold ?? CONSTANTS.DEFAULT_KG_COSINE_THRESHOLD;

    let effectiveMode: ConcreteRagQueryMode = isCragRetry
      ? 'mix'
      : input.strategyOverride === 'fast_hybrid'
        ? 'naive'
        : input.strategyOverride
          ? 'mix'
          : plan.effectiveMode;
    let routingReason = isCragRetry ? 'crag_escalate_retry' : plan.reason;

    if (effectiveMode !== 'naive') {
      const hasKg = await this.kgRepo.hasGraphData(input.orgId, input.documentIds).catch(() => false);
      if (!hasKg) {
        effectiveMode = 'naive';
        routingReason = 'empty_kg_fallback';
      }
    }

    const useKg = effectiveMode !== 'naive';
    const wantLocal = effectiveMode === 'local' || effectiveMode === 'hybrid' || effectiveMode === 'mix';
    const wantGlobal = effectiveMode === 'global' || effectiveMode === 'hybrid' || effectiveMode === 'mix';
    const wantDirectChunks = effectiveMode === 'naive' || effectiveMode === 'mix';
    const wantDecompose =
      useKg &&
      (strategy === 'decomposed_ppr' || (requestedMode === 'auto' && shouldDecomposeQuery(intent, input.userQuery)));
    const wantDrift = strategy === 'drift_global' && this.communityRepo != null;

    // 2. Query Analysis (Unified single-call decomposition + keywords, OR speculative parallel single-turn)
    let analysis: QueryAnalysisResult;
    let subQueries: string[];
    let queryAnalysisMs: number;
    let precomputedQueryEmbedding: number[] | undefined;
    let precomputedDirectChunks: InternalCandidate[] | undefined;

    if (useKg && wantDirectChunks && history.length === 0) {
      const specStart = Date.now();
      const [analysisRes, specChunkRes] = await Promise.all([
        this.analyzeQuery(input.orgId, input.userQuery, history, true, wantDecompose),
        (async () => {
          const [qVec] = await this.batchEmbedQueries([input.userQuery]);
          if (!qVec) return { qVec: undefined, chunks: [] as InternalCandidate[] };
          const chunks = await this.runHybridChunkSearch(
            input.orgId,
            input.userQuery,
            qVec,
            chunkTopK,
            cosineThreshold,
            profile.stage1.dense,
            profile.stage1.bm25,
            input.documentIds,
          );
          return { qVec, chunks };
        })(),
      ]);
      analysis = analysisRes.analysis;
      subQueries = analysisRes.subQueries;
      precomputedQueryEmbedding = specChunkRes.qVec;
      precomputedDirectChunks = specChunkRes.chunks;
      queryAnalysisMs = Date.now() - specStart;
    } else {
      const analysisStart = Date.now();
      const analysisRes = await this.analyzeQuery(input.orgId, input.userQuery, history, useKg, wantDecompose);
      analysis = analysisRes.analysis;
      subQueries = analysisRes.subQueries;
      queryAnalysisMs = Date.now() - analysisStart;
    }

    const standaloneQuery = analysis.standaloneQuery || input.userQuery;
    const rawLl = analysis.lowLevelKeywords.join(', ');
    const rawHl = analysis.highLevelKeywords.join(', ');
    const entityQuery = wantLocal ? rawLl || rawHl || standaloneQuery : '';
    const relationQuery = wantGlobal ? rawHl || rawLl || standaloneQuery : '';

    // 3. Single-batch query embedding across standaloneQuery, KG keywords, and decomposed sub-queries
    const embedStart = Date.now();
    const uniqueTexts: string[] = [];
    const textToIndex = new Map<string, number>();
    const registerText = (t: string): number => {
      if (!t) return -1;
      if (precomputedQueryEmbedding && t === input.userQuery) return -2;
      const existing = textToIndex.get(t);
      if (existing !== undefined) return existing;
      const idx = uniqueTexts.length;
      uniqueTexts.push(t);
      textToIndex.set(t, idx);
      return idx;
    };

    const qIdx = registerText(standaloneQuery);
    const llIdx = registerText(entityQuery);
    const hlIdx = registerText(relationQuery);
    const extraSubQueries = subQueries.filter((sq) => sq.toLowerCase() !== standaloneQuery.toLowerCase()).slice(0, 2);
    const subIdxList = extraSubQueries.map((sq) => ({ query: sq, idx: registerText(sq) }));

    const embeddings = await this.batchEmbedQueries(uniqueTexts);
    const resolveVec = (idx: number): number[] | undefined =>
      idx === -2 ? precomputedQueryEmbedding : idx >= 0 ? embeddings[idx] : undefined;

    const queryEmbedding = resolveVec(qIdx);
    const llEmbedding = resolveVec(llIdx);
    const hlEmbedding = resolveVec(hlIdx);
    const embeddingMs = Date.now() - embedStart;

    // 4. Parallel multi-store retrieval (Local Entities + Global Relations + Direct/Sub-query Chunks + RAPTOR Communities)
    const retrievalStart = Date.now();

    const localBranch = async (): Promise<{ entities: RetrievedEntity[]; relations: RetrievedRelation[] }> => {
      if (!wantLocal || !llEmbedding) return { entities: [], relations: [] };
      const entities =
        typeof this.kgRepo.searchEntitiesHybrid === 'function'
          ? await this.kgRepo.searchEntitiesHybrid(
            input.orgId,
            entityQuery,
            llEmbedding,
            topK,
            cosineThreshold,
            input.documentIds,
          )
          : await this.kgRepo.searchEntities(
            input.orgId,
            llEmbedding,
            topK,
            cosineThreshold,
            input.documentIds,
          );
      const canonicals = entities.map((e) => e.canonicalName);
      const relations =
        canonicals.length > 0
          ? await this.kgRepo.getRelationsForEntities(input.orgId, canonicals, topK, input.documentIds)
          : [];
      return { entities, relations };
    };

    const globalBranch = async (): Promise<{ entities: RetrievedEntity[]; relations: RetrievedRelation[] }> => {
      if (!wantGlobal || !hlEmbedding) return { entities: [], relations: [] };
      const relations =
        typeof this.kgRepo.searchRelationsHybrid === 'function'
          ? await this.kgRepo.searchRelationsHybrid(
            input.orgId,
            relationQuery,
            hlEmbedding,
            topK,
            cosineThreshold,
            input.documentIds,
          )
          : await this.kgRepo.searchRelations(
            input.orgId,
            hlEmbedding,
            topK,
            cosineThreshold,
            input.documentIds,
          );
      const endpointCanonicals = [...new Set(relations.flatMap((r) => [r.sourceCanonical, r.targetCanonical]))];
      const entities =
        endpointCanonicals.length > 0
          ? await this.kgRepo.getEntitiesByCanonicalNames(input.orgId, endpointCanonicals, input.documentIds)
          : [];
      return { entities, relations };
    };

    const communityBranch = async (): Promise<CommunitySummaryHit[]> => {
      if (!this.communityRepo || !queryEmbedding) return [];
      if (!wantDrift && intent !== 'global_synthesis' && effectiveMode !== 'global') return [];
      return this.communityRepo
        .searchCommunities(input.orgId, queryEmbedding, 4, 0.25, input.documentIds)
        .catch(() => []);
    };

    const subQueryChunksBranch = async (): Promise<InternalCandidate[][]> => {
      if (!wantDirectChunks || subIdxList.length === 0) return [];
      return Promise.all(
        subIdxList.map(async ({ query, idx }) => {
          const vec = resolveVec(idx);
          if (!vec) return [];
          return this.runHybridChunkSearch(
            input.orgId,
            query,
            vec,
            Math.max(8, Math.floor(chunkTopK * 0.75)),
            cosineThreshold,
            profile.stage1.dense,
            profile.stage1.bm25,
            input.documentIds,
          );
        }),
      );
    };

    const [localRes, globalRes, directChunks, subQueryChunkLists, communitiesUsed] = await Promise.all([
      localBranch(),
      globalBranch(),
      precomputedDirectChunks !== undefined
        ? Promise.resolve(precomputedDirectChunks)
        : wantDirectChunks && queryEmbedding
          ? this.runHybridChunkSearch(
            input.orgId,
            standaloneQuery,
            queryEmbedding,
            chunkTopK,
            cosineThreshold,
            profile.stage1.dense,
            profile.stage1.bm25,
            input.documentIds,
          )
          : Promise.resolve<InternalCandidate[]>([]),
      subQueryChunksBranch(),
      communityBranch(),
    ]);

    // DRIFT-style adaptive follow-up (#11): if community summaries surface documents not yet in directChunks,
    // pull top local chunks scoped to those community documents within strict budget.
    let driftChunks: InternalCandidate[] = [];
    if (wantDrift && communitiesUsed.length > 0 && queryEmbedding) {
      const communityDocIds = [...new Set(communitiesUsed.flatMap((c) => c.documentIds))];
      const allowedCommunityDocs =
        input.documentIds && input.documentIds.length > 0
          ? communityDocIds.filter((id) => input.documentIds!.includes(id))
          : communityDocIds;
      if (allowedCommunityDocs.length > 0) {
        driftChunks = await this.runHybridChunkSearch(
          input.orgId,
          standaloneQuery,
          queryEmbedding,
          Math.min(10, chunkTopK),
          cosineThreshold,
          profile.stage1.dense,
          profile.stage1.bm25,
          allowedCommunityDocs.slice(0, 40),
        );
      }
    }

    // 5. Merge Local + Global KG entities & relations
    const mergedEntities = roundRobinMerge<RetrievedEntity>(
      [localRes.entities, globalRes.entities],
      (e) => e.canonicalName,
    );
    const mergedRelations = roundRobinMerge<RetrievedRelation>(
      [localRes.relations, globalRes.relations],
      (r) => `${r.sourceCanonical}::${r.targetCanonical}`,
    );

    // 5b. Graph-Aware Personalized PageRank (#1) over ACL-filtered 2-hop tenant subgraph
    const pprStart = Date.now();
    let pprChunkScores = new Map<string, number>();
    if (useKg && mergedEntities.length > 0 && typeof this.kgRepo.getTenantSubgraph === 'function') {
      const seedWeights = new Map<string, number>();
      mergedEntities.slice(0, 15).forEach((entity, idx) => {
        const w = Math.max(0.05, entity.similarity) * (1 / (1 + idx * 0.15));
        seedWeights.set(entity.canonicalName, (seedWeights.get(entity.canonicalName) ?? 0) + w);
      });
      try {
        const subgraph = await this.kgRepo.getTenantSubgraph(input.orgId, [...seedWeights.keys()], {
          maxHops: 2,
          maxEdges: 240,
          documentIds: input.documentIds,
        });
        if (subgraph.edges.length > 0) {
          const ppr = computePersonalizedPageRank(subgraph.edges, seedWeights, {
            damping: 0.85,
            iterations: 12,
          });
          pprChunkScores = propagatePprToChunks({
            nodeScores: ppr.nodeScores,
            entityChunks: subgraph.entityChunks,
            edges: subgraph.edges,
          });
        }
      } catch {
        /* PPR is additive; gracefully degrade if subgraph lookup fails */
      }
    }
    const pprMs = Date.now() - pprStart;

    // 6. Provenance chunk collection across Entities, Relations, and PPR bridge chunks
    const entityChunkStats = new Map<string, { bestRank: number; freq: number }>();
    mergedEntities.forEach((entity, rank) => {
      for (const cid of entity.sourceChunkIds.slice(0, CONSTANTS.DEFAULT_RELATED_CHUNKS_PER_ITEM)) {
        const cur = entityChunkStats.get(cid);
        if (cur) {
          cur.freq++;
          cur.bestRank = Math.min(cur.bestRank, rank);
        } else {
          entityChunkStats.set(cid, { bestRank: rank, freq: 1 });
        }
      }
    });

    const relationChunkStats = new Map<string, { bestRank: number; freq: number; bridgeBoost: number }>();
    mergedRelations.forEach((rel, rank) => {
      const isCrossDocBridge = rel.documentIds.length > 1;
      for (const cid of rel.sourceChunkIds.slice(0, CONSTANTS.DEFAULT_RELATED_CHUNKS_PER_ITEM)) {
        const cur = relationChunkStats.get(cid);
        if (cur) {
          cur.freq++;
          cur.bestRank = Math.min(cur.bestRank, rank);
          if (isCrossDocBridge) cur.bridgeBoost = Math.max(cur.bridgeBoost, 0.12);
        } else {
          relationChunkStats.set(cid, {
            bestRank: rank,
            freq: 1,
            bridgeBoost: isCrossDocBridge ? 0.12 : 0,
          });
        }
      }
    });

    const topPprChunkIds = [...pprChunkScores.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 24)
      .map(([id]) => id);

    const kgChunkIds = [...new Set([...entityChunkStats.keys(), ...relationChunkStats.keys(), ...topPprChunkIds])];
    const entityChunks: InternalCandidate[] = [];
    const relationChunks: InternalCandidate[] = [];
    const pprChunks: InternalCandidate[] = [];

    if (kgChunkIds.length > 0 && queryEmbedding) {
      const kgFetchQuota = Math.max(
        36,
        Math.min(kgChunkIds.length, Math.floor((mergedEntities.length + mergedRelations.length) * 2.5) + 12),
      );
      const rankedKgRows = await this.chunkRepo.findByIdsRanked(
        input.orgId,
        kgChunkIds,
        queryEmbedding,
        kgFetchQuota,
        input.documentIds,
      );

      const maxEntityFreq = Math.max(1, ...[...entityChunkStats.values()].map((s) => s.freq));
      const maxRelFreq = Math.max(1, ...[...relationChunkStats.values()].map((s) => s.freq));

      for (const row of rankedKgRows) {
        const doc = row.document as typeof row.document & {
          effectiveAt?: Date | string | null;
          expiresAt?: Date | string | null;
          supersedesDocumentId?: string | null;
          authorityTier?: number | null;
        };
        const ch = row.chunk as typeof row.chunk & {
          parentChunkId?: string | null;
          chunkRole?: 'parent' | 'child' | 'standalone';
        };
        const eStat = entityChunkStats.get(row.chunk.id);
        const rStat = relationChunkStats.get(row.chunk.id);
        const pprScore = pprChunkScores.get(row.chunk.id) ?? 0;

        const baseCandidate = {
          id: row.chunk.id,
          documentId: row.document.id,
          documentTitle: row.document.title,
          fileName: row.document.fileName,
          fileType: row.document.fileType,
          content: row.chunk.content,
          sectionHeading: row.chunk.sectionHeading,
          pageNumber: row.chunk.pageNumber,
          parentChunkId: ch.parentChunkId ?? null,
          chunkRole: ch.chunkRole ?? 'standalone',
          effectiveAt: doc.effectiveAt ?? null,
          expiresAt: doc.expiresAt ?? null,
          supersedesDocumentId: doc.supersedesDocumentId ?? null,
          authorityTier: doc.authorityTier ?? null,
          createdAt: row.document.createdAt,
          updatedAt: row.document.updatedAt,
          denseSim: row.similarity,
          bm25Norm: 0,
          entityHit: eStat ? 1 / (1 + eStat.bestRank * 0.25) : 0,
          relationHit: rStat ? 1 / (1 + rStat.bestRank * 0.25) : 0,
          pprScore,
          maxsimScore: row.similarity,
          crossEncoderScore: 0,
          siblingHits: 1,
        };

        if (eStat) {
          const rankScore = 1 / (1 + eStat.bestRank * 0.25);
          const freqScore = eStat.freq / maxEntityFreq;
          const combined = 0.72 * row.similarity + 0.10 * rankScore + 0.06 * freqScore + 0.12 * pprScore;
          entityChunks.push({ ...baseCandidate, similarity: combined, source: 'entity' });
        }

        if (rStat && !eStat) {
          const rankScore = 1 / (1 + rStat.bestRank * 0.25);
          const freqScore = rStat.freq / maxRelFreq;
          const combined =
            0.70 * row.similarity + 0.10 * rankScore + 0.05 * freqScore + rStat.bridgeBoost * 0.4 + 0.12 * pprScore;
          relationChunks.push({ ...baseCandidate, similarity: combined, source: 'relation' });
        }

        if (pprScore >= 0.15) {
          const combined = 0.55 * row.similarity + 0.45 * pprScore;
          pprChunks.push({ ...baseCandidate, similarity: combined, source: 'ppr' });
        }
      }

      entityChunks.sort((a, b) => b.similarity - a.similarity);
      relationChunks.sort((a, b) => b.similarity - a.similarity);
      pprChunks.sort((a, b) => b.similarity - a.similarity);
    }

    // Fallback to direct hybrid if KG-only mode yielded zero chunks
    let fallbackChunks = directChunks;
    if (!wantDirectChunks && entityChunks.length === 0 && relationChunks.length === 0 && pprChunks.length === 0 && queryEmbedding) {
      fallbackChunks = await this.runHybridChunkSearch(
        input.orgId,
        standaloneQuery,
        queryEmbedding,
        chunkTopK,
        cosineThreshold,
        profile.stage1.dense,
        profile.stage1.bm25,
        input.documentIds,
      );
    }

    // 6b. Stage 1 Adaptive Weighted Fusion (#3) + Multi-Query MaxSim Aggregation (#9)
    const unifiedById = new Map<string, InternalCandidate>();
    const absorbCandidate = (c: InternalCandidate) => {
      const existing = unifiedById.get(c.id);
      if (!existing) {
        unifiedById.set(c.id, { ...c });
      } else {
        existing.denseSim = Math.max(existing.denseSim, c.denseSim);
        existing.bm25Norm = Math.max(existing.bm25Norm, c.bm25Norm);
        existing.entityHit = Math.max(existing.entityHit, c.entityHit);
        existing.relationHit = Math.max(existing.relationHit, c.relationHit);
        existing.pprScore = Math.max(existing.pprScore, c.pprScore);
        // Multi-query MaxSim (#9): retains peak similarity across decomposed sub-queries
        existing.maxsimScore = Math.max(existing.maxsimScore, c.maxsimScore, c.denseSim);
        existing.siblingHits += 1;
      }
    };

    const allLists: Array<{ items: InternalCandidate[]; weight: number }> = [
      { items: fallbackChunks, weight: Math.max(0.1, (profile.stage1.dense + profile.stage1.bm25) * 0.5) },
      { items: entityChunks, weight: Math.max(0.05, profile.stage1.entity) },
      { items: relationChunks, weight: Math.max(0.05, profile.stage1.relation) },
      { items: pprChunks, weight: Math.max(0.05, profile.stage1.ppr) },
      { items: driftChunks, weight: Math.max(0.05, profile.stage1.community) },
      ...subQueryChunkLists.map((list) => ({ items: list, weight: Math.max(0.2, profile.stage1.maxsim) })),
    ];

    for (const { items } of allLists) {
      for (const item of items) absorbCandidate(item);
    }

    const stage1Fused = weightedReciprocalRankFusion<InternalCandidate>(allLists, (c) => c.id, 60);
    const maxStage1Score = Math.max(1e-6, stage1Fused[0]?.score ?? 1);

    // Diversity-first doc ordering before Stage 2 rerank
    const firstPassByDoc: InternalCandidate[] = [];
    const overflowChunks: InternalCandidate[] = [];
    const seenDocCounts = new Map<string, number>();
    for (const { item, score } of stage1Fused) {
      const enriched = unifiedById.get(item.id) ?? item;
      enriched.similarity = score / maxStage1Score;
      const count = seenDocCounts.get(enriched.documentId) ?? 0;
      if (count === 0) firstPassByDoc.push(enriched);
      else overflowChunks.push(enriched);
      seenDocCounts.set(enriched.documentId, count + 1);
    }

    let candidates = [...firstPassByDoc, ...overflowChunks].slice(0, Math.max(chunkTopK, rerankLimit * 3));
    const retrievalMs = Date.now() - retrievalStart;

    // 7. Selective Cross-Encoder Reranking (#2):
    // Skip the external reranker HTTP round-trip on high-confidence exact/factual fast-path hits
    // (saves ~400-500ms p50 latency when BGE-M3 + BM25 already have unambiguous consensus).
    const rerankStart = Date.now();
    let usedCrossEncoder = false;
    const top1Dense = candidates[0]?.denseSim ?? 0;
    const top2Dense = candidates[1]?.denseSim ?? 0;
    const fastPathConfident =
      effectiveMode === 'naive' &&
      (intent === 'exact_lookup' || intent === 'factual') &&
      ((plan.features.hasExactCode &&
        candidates[0] != null &&
        computeExactCodeMatchScore(plan.features.exactCodes, candidates[0].content, candidates[0].documentTitle) >= 1) ||
        (top1Dense >= 0.62 && top1Dense - top2Dense >= 0.04));

    if (this.reranker && candidates.length > 1 && !fastPathConfident) {
      const scored: ScoredChunk[] = candidates.map((c) => ({
        id: c.id,
        content: c.sectionHeading ? `${c.sectionHeading}\n${c.content}` : c.content,
        documentId: c.documentId,
        score: c.similarity,
      }));
      try {
        const reranked = await this.reranker.rerank(standaloneQuery, scored, Math.min(candidates.length, rerankLimit * 2));
        const rerankScoreMap = new Map(reranked.map((r) => [r.id, Math.max(0, Math.min(1, r.score ?? 0))]));
        usedCrossEncoder = rerankScoreMap.size > 0;
        for (const c of candidates) {
          c.crossEncoderScore = rerankScoreMap.get(c.id) ?? 0;
        }
      } catch {
        usedCrossEncoder = false;
      }
    }
    const rerankMs = Date.now() - rerankStart;

    // 8. Temporal & Authority Conflict Resolution (#10) + 12-Feature LTR Scoring (#12)
    const temporalIntent = parseTemporalIntent(standaloneQuery);
    const temporalAnnotations = resolveTemporalAndAuthorityConflicts(candidates, temporalIntent);
    const queryTerms = extractSignificantTerms(standaloneQuery);

    // Compute parent-coherence support (count of sibling child chunks hitting the same parent)
    const parentHitCounts = new Map<string, number>();
    for (const c of candidates) {
      if (c.parentChunkId) {
        parentHitCounts.set(c.parentChunkId, (parentHitCounts.get(c.parentChunkId) ?? 0) + 1);
      }
    }

    const scoredWithFeatures: Array<{ candidate: InternalCandidate; features: RetrievalFeatureVector; finalScore: number }> = [];
    for (const c of candidates) {
      const tAnn = temporalAnnotations.get(c.documentId);
      const versionActive = tAnn ? (tAnn.isSuperseded ? 0.0 : tAnn.isExpired ? 0.3 : 1.0) : 1.0;
      const parentHits = c.parentChunkId ? parentHitCounts.get(c.parentChunkId) ?? 1 : c.siblingHits;

      const features: RetrievalFeatureVector = {
        denseSim: Math.max(0, Math.min(1, c.denseSim)),
        bm25Norm: Math.max(0, Math.min(1, c.bm25Norm)),
        entityHit: Math.max(0, Math.min(1, c.entityHit)),
        relationHit: Math.max(0, Math.min(1, c.relationHit)),
        pprScore: Math.max(0, Math.min(1, c.pprScore)),
        crossEncoderScore: usedCrossEncoder ? c.crossEncoderScore : Math.max(0, Math.min(1, c.denseSim)),
        recencyScore: tAnn?.recencyScore ?? 0.7,
        authorityScore: tAnn?.authorityScore ?? 0.5,
        versionActive,
        headingMatch: computeHeadingMatchScore(queryTerms, c.sectionHeading),
        exactCodeMatch: computeExactCodeMatchScore(plan.features.exactCodes, c.content, c.documentTitle),
        parentCoherence: Math.min(1, parentHits / 3),
      };

      const rawLtr = scoreCandidateWithLtr(features, profile.ltr);
      const finalScore = Math.max(0, Math.min(1, rawLtr * (tAnn?.temporalMultiplier ?? 1.0)));

      if (tAnn?.badge) {
        c.temporalBadge = tAnn.badge;
      }
      c.similarity = finalScore;
      scoredWithFeatures.push({ candidate: c, features, finalScore });
    }

    scoredWithFeatures.sort((a, b) => b.finalScore - a.finalScore);

    // Prune noisy tail below minRelevanceScore (keeps token usage lean & prevents irrelevant padding)
    let keptScored = scoredWithFeatures.filter((s) => s.finalScore >= rerankMinScore).slice(0, rerankLimit);
    if (keptScored.length === 0 && scoredWithFeatures.length > 0) {
      keptScored = scoredWithFeatures.slice(0, Math.min(3, rerankLimit));
    }

    // 9. Bounded CRAG Confidence Gate (#6 — Zero LLM grading, max 1 escalation retry)
    const crag = evaluateCragConfidence(
      standaloneQuery,
      keptScored.map((s) => ({ id: s.candidate.id, content: s.candidate.content, score: s.finalScore })),
      {
        highConfidenceThreshold: config.crag.highConfidenceThreshold,
        lowConfidenceThreshold: config.crag.lowConfidenceThreshold,
        minTermCoverageForHigh: config.crag.minTermCoverageForHigh,
        allowRetry: config.crag.enableRetry && !input.disableCragRetry,
      },
      isCragRetry,
    );

    if (crag.action === 'retry_escalate' && !isCragRetry && effectiveMode === 'naive') {
      const hasKg = await this.kgRepo.hasGraphData(input.orgId, input.documentIds).catch(() => false);
      if (hasKg) {
        return this.executeRetrievalPass(input, true);
      }
    }

    if (crag.action === 'prune_tail' && keptScored.length > 3) {
      const cutoff = (keptScored[0]?.finalScore ?? 0) * 0.55;
      keptScored = keptScored.filter((s, idx) => idx < 2 || s.finalScore >= cutoff);
    }

    // 10. Token-Efficient Parent-Child Context Expansion (#7)
    let finalChunks: ContextChunkInput[] = keptScored.map((s) => s.candidate);
    const topChildScore = keptScored[0]?.finalScore ?? 0;
    const shouldExpandParents =
      finalChunks.some((c) => c.chunkRole === 'child' && c.parentChunkId) &&
      !(
        (intent === 'exact_lookup' || intent === 'factual') &&
        topChildScore >= profile.skipParentExpandAbove
      );

    if (shouldExpandParents && typeof this.chunkRepo.findParentsByIds === 'function') {
      const parentIdsToFetch = [
        ...new Set(
          finalChunks
            .filter((c) => c.chunkRole === 'child' && c.parentChunkId)
            .map((c) => c.parentChunkId as string),
        ),
      ];
      if (parentIdsToFetch.length > 0) {
        const parentRows = await this.chunkRepo
          .findParentsByIds(input.orgId, parentIdsToFetch, input.documentIds)
          .catch(() => []);
        const parentMap = new Map(parentRows.map((r) => [r.chunk.id, r]));
        const expanded: ContextChunkInput[] = [];
        const usedParents = new Set<string>();

        for (const chunk of finalChunks) {
          if (chunk.chunkRole === 'child' && chunk.parentChunkId) {
            if (usedParents.has(chunk.parentChunkId)) continue; // Deduplicate sibling children!
            const pRow = parentMap.get(chunk.parentChunkId);
            if (pRow) {
              usedParents.add(chunk.parentChunkId);
              expanded.push({
                ...chunk,
                id: pRow.chunk.id,
                content: pRow.chunk.content,
                sectionHeading: pRow.chunk.sectionHeading ?? chunk.sectionHeading,
                pageNumber: pRow.chunk.pageNumber ?? chunk.pageNumber,
                chunkRole: 'parent',
              });
              continue;
            }
          }
          expanded.push(chunk);
        }
        finalChunks = expanded;
      }
    }

    // Prepend compact RAPTOR community summaries as high-level context items on global_synthesis
    const effectiveBudget: TokenBudgetConfig = {
      maxEntityTokens: input.budget?.maxEntityTokens ?? plan.budget.maxEntityTokens,
      maxRelationTokens: input.budget?.maxRelationTokens ?? plan.budget.maxRelationTokens,
      maxTotalTokens: input.budget?.maxTotalTokens ?? plan.budget.maxTotalTokens,
    };

    const assembled = assembleLightRagContext({
      entities: mergedEntities,
      relations: mergedRelations,
      chunks: finalChunks,
      budget: effectiveBudget,
    });

    if (communitiesUsed.length > 0) {
      const commLines = communitiesUsed
        .slice(0, 3)
        .map((c, idx) => `[Community L${c.level} #${idx + 1}] ${c.title}: ${c.summary}`)
        .join('\n');
      assembled.contextText = `### Corpus Community Summaries\n${commLines}\n\n${assembled.contextText}`;
    }

    const candidateFeatures: CandidateFeatureRecord[] = scoredWithFeatures.slice(0, 12).map((s) => ({
      chunkId: s.candidate.id,
      documentId: s.candidate.documentId,
      finalScore: Number(s.finalScore.toFixed(4)),
      features: s.features,
    }));

    return {
      requestedMode,
      effectiveMode,
      routingReason,
      intent,
      strategy,
      configVersion: config.version,
      subQueries,
      crag,
      communitiesUsed,
      candidateFeatures,
      analysis,
      assembled,
      timingsMs: {
        queryAnalysis: queryAnalysisMs,
        embedding: embeddingMs,
        retrieval: retrievalMs,
        ppr: pprMs,
        rerank: rerankMs,
        total: Date.now() - totalStart,
      },
    };
  }
}
