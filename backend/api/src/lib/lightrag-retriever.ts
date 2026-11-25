import crypto from 'crypto';
import {
  CONSTANTS,
  DEFAULT_RETRIEVAL_CONFIG,
  assembleLightRagContext,
  buildSlimKeywordsPrompt,
  buildUnifiedDecomposePrompt,
  buildUnifiedQueryAnalysisPrompt,
  calibrateDenseScores,
  classifyQueryIntent,
  computeAdaptiveTopK,
  computeExactCodeMatchScore,
  computeHeadingMatchScore,
  computePersonalizedPageRank,
  detectEvidenceContradictions,
  detectQueryAmbiguity,
  enrichPronounQueryWithHistory,
  evaluateCragConfidence,
  extractSignificantTerms,
  isTopicContinuation,
  lockExactCodesInQuery,
  parseDecomposeOutput,
  parseQueryAnalysisOutput,
  parseTemporalIntent,
  propagatePprToChunks,
  resolveTemporalAndAuthorityConflicts,
  roundRobinMerge,
  sanitizeConversationHistory,
  scoreCandidateWithLtr,
  selectDiverseNonDuplicateCandidates,
  shouldDecomposeQuery,
  SingleflightGroup,
  traceTopPprPaths,
  verifyMultiHopEvidenceCoverage,
  weightedReciprocalRankFusion,
  type AmbiguityAssessment,
  type AssembledContext,
  type ConcreteRagQueryMode,
  type ContextChunkInput,
  type ContradictionAnalysis,
  type CragEvaluation,
  type EmbeddingProvider,
  type MultiHopCoverageReport,
  type PprPathTrace,
  type QueryAnalysisResult,
  type QueryFeatures,
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
  /** Optional governance visibility overrides (drafts are excluded by default). */
  includeDrafts?: boolean;
  /** Optional point-in-time validity date (`asOfDate`) for historical policy lookups. */
  asOfDate?: Date | null;
}

export interface LightRagRetrievalResult {
  requestedMode: RagQueryMode;
  effectiveMode: ConcreteRagQueryMode;
  routingReason: string;
  intent: QueryIntent;
  facets: QueryFeatures;
  strategy: RetrievalStrategyName;
  configVersion: number;
  subQueries: string[];
  crag: CragEvaluation;
  ambiguity: AmbiguityAssessment;
  contradictions: ContradictionAnalysis;
  multiHopCoverage: MultiHopCoverageReport;
  pprPaths: PprPathTrace[];
  historyInjectionsNeutralized: number;
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

const HAS_GRAPH_CACHE_TTL_MS = 30_000;
const HAS_GRAPH_CACHE_MAX_ENTRIES = 500;
const hasGraphCacheByRepo = new WeakMap<object, Map<string, { value: boolean; expiresAt: number }>>();
const chunkSearchFlightByRepo = new WeakMap<object, SingleflightGroup>();

function hashScopeIds(documentIds?: string[]): string {
  if (!documentIds || documentIds.length === 0) return 'all';
  const sorted = [...new Set(documentIds)].sort().join(',');
  return crypto.createHash('sha1').update(sorted).digest('hex').slice(0, 16);
}

function getGraphCache(repo: object) {
  let cache = hasGraphCacheByRepo.get(repo);
  if (!cache) {
    cache = new Map();
    hasGraphCacheByRepo.set(repo, cache);
  }
  return cache;
}

function getChunkSearchFlight(repo: object) {
  let group = chunkSearchFlightByRepo.get(repo);
  if (!group) {
    group = new SingleflightGroup();
    chunkSearchFlightByRepo.set(repo, group);
  }
  return group;
}

/** Clears cached `hasGraphData` entries for a repository (useful on KG ingestion or in tests). */
export function invalidateGraphDataCache(kgRepo: object, orgId?: string): void {
  const cache = hasGraphCacheByRepo.get(kgRepo);
  if (!cache) return;
  if (!orgId) {
    cache.clear();
    return;
  }
  const prefix = `${orgId}|`;
  for (const key of cache.keys()) {
    if (key.startsWith(prefix)) cache.delete(key);
  }
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

  private async checkHasGraphData(orgId: string, documentIds?: string[]): Promise<boolean> {
    const scopeHash = hashScopeIds(documentIds);
    const localKey = `${orgId}|${scopeHash}`;
    const cache = getGraphCache(this.kgRepo);
    const now = Date.now();
    const cached = cache.get(localKey);
    if (cached && cached.expiresAt > now) {
      return cached.value;
    }

    const flight = getChunkSearchFlight(this.kgRepo);
    return flight.run(`hasGraph:${localKey}`, async () => {
      const value = await this.kgRepo.hasGraphData(orgId, documentIds).catch(() => false);
      if (cache.size >= HAS_GRAPH_CACHE_MAX_ENTRIES) {
        const oldestKey = cache.keys().next().value;
        if (oldestKey !== undefined) cache.delete(oldestKey);
      }
      cache.set(localKey, { value, expiresAt: Date.now() + HAS_GRAPH_CACHE_TTL_MS });
      return value;
    });
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
    governance?: { includeDrafts?: boolean; includeArchived?: boolean; asOfDate?: Date | null },
  ): Promise<InternalCandidate[]> {
    // Tenant + ACL + governance isolated singleflight coalesces identical burst queries
    // without ever sharing results across tenants or permission scopes.
    const govKey = `${governance?.includeDrafts ? 1 : 0}:${governance?.includeArchived ? 1 : 0}:${governance?.asOfDate ? governance.asOfDate.toISOString() : ''}`;
    const flightKey = `hybrid:${orgId}|${hashScopeIds(documentIds)}|${govKey}|${chunkTopK}|${cosineThreshold.toFixed(3)}|${denseWeight.toFixed(2)}|${bm25Weight.toFixed(2)}|${queryText.trim().toLowerCase()}`;
    const flight = getChunkSearchFlight(this.chunkRepo);

    const results = await flight.run(flightKey, async () => {
      const [denseResults, ftsResults] = await Promise.all([
        this.chunkRepo.searchSimilar(orgId, queryEmbedding, chunkTopK, cosineThreshold, documentIds, governance),
        this.chunkRepo.searchFullText(orgId, queryText, chunkTopK, documentIds, governance),
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
          governanceStatus?: string | null;
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
          governanceStatus: doc.governanceStatus ?? 'active',
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
    });

    // Return per-caller copies so downstream scoring/mutation never aliases across coalesced callers.
    return results.map((c) => ({ ...c }));
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
    // Sanitize conversation history against injected directives & prevent topic-shift contamination
    const { sanitized: sanitizedHistory, injectionsNeutralized: historyInjectionsNeutralized } =
      sanitizeConversationHistory(input.history ?? []);
    const topicContinues = isTopicContinuation(input.userQuery, sanitizedHistory);
    const history = topicContinues ? sanitizedHistory : [];

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
      const hasKg = await this.checkHasGraphData(input.orgId, input.documentIds);
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

    const governanceFilter = {
      includeDrafts: Boolean(input.includeDrafts),
      includeArchived: plan.features.allowHistorical,
      asOfDate: input.asOfDate ?? null,
    };

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
            governanceFilter,
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

    const rawStandalone = analysis.standaloneQuery || input.userQuery;
    const historyResolved =
      history.length > 0 &&
      plan.features.hasHistoryPronoun &&
      rawStandalone.trim().toLowerCase() === input.userQuery.trim().toLowerCase()
        ? enrichPronounQueryWithHistory(rawStandalone, history)
        : rawStandalone;
    // Deterministic pre-LLM identifier lock: guarantees rewrites never drop exact corporate codes
    const standaloneQuery = lockExactCodesInQuery(input.userQuery, historyResolved);
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
    // Preserve original raw userQuery as a batched dual-query branch whenever history resolution
    // or rewriting altered standaloneQuery so multi-clause follow-ups never drop second-clause recall.
    const candidateSubQueries =
      standaloneQuery.trim().toLowerCase() !== input.userQuery.trim().toLowerCase()
        ? [...subQueries, input.userQuery]
        : subQueries;
    const extraSubQueries = [...new Set(candidateSubQueries)]
      .filter((sq) => sq.trim().toLowerCase() !== standaloneQuery.trim().toLowerCase())
      .slice(0, 2);
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
            governanceFilter,
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

    // 5b. Graph-Aware Personalized PageRank (#1) + Anchor-Guided Pseudo-Relevance Graph Bridging (PRGB)
    // Seeds PPR from BOTH IDF-weighted query entities AND high-specificity bridge entities anchored
    // in the top direct hybrid chunks (strictly skipped in `local`/`hybrid` modes so KG-only modes stay self-sufficient).
    const pprStart = Date.now();
    let pprChunkScores = new Map<string, number>();
    let pprPaths: PprPathTrace[] = [];
    const codeAnchorIds =
      wantDirectChunks && plan.features.exactCodes.length > 0
        ? directChunks
            .filter(
              (c) =>
                computeExactCodeMatchScore(plan.features.exactCodes, c.content, c.documentTitle) > 0,
            )
            .map((c) => c.id)
        : [];
    const anchorChunkIds = wantDirectChunks
      ? [...new Set([...codeAnchorIds, ...directChunks.slice(0, 3).map((c) => c.id)])].slice(0, 4)
      : [];
    let anchorEntities: RetrievedEntity[] = [];
    if (
      useKg &&
      wantDirectChunks &&
      anchorChunkIds.length > 0 &&
      typeof this.kgRepo.getEntitiesBySourceChunkIds === 'function'
    ) {
      try {
        anchorEntities = await this.kgRepo.getEntitiesBySourceChunkIds(
          input.orgId,
          anchorChunkIds,
          18,
          input.documentIds,
        );
      } catch {
        anchorEntities = [];
      }
    }

    if (useKg && (mergedEntities.length > 0 || anchorEntities.length > 0) && typeof this.kgRepo.getTenantSubgraph === 'function') {
      const seedWeights = new Map<string, number>();

      // Primary query-matched entity seeds (bounded IDF prevents mega-hubs without crushing key entities)
      mergedEntities.slice(0, 12).forEach((entity, idx) => {
        const docSpread = Math.max(1, entity.documentIds?.length ?? entity.sourceChunkIds.length);
        const idf = Math.max(0.25, Math.min(1.0, 1 / Math.log2(2 + docSpread)));
        const w = Math.max(0.05, entity.similarity) * (1 / (1 + idx * 0.25)) * idf;
        seedWeights.set(entity.canonicalName, (seedWeights.get(entity.canonicalName) ?? 0) + w);
      });

      // Anchor-chunk bridge entity seeds: only specific entities (docSpread <= 5) so bridges jump cleanly
      const anchorRankByChunk = new Map(anchorChunkIds.map((id, i) => [id, i]));
      for (const entity of anchorEntities) {
        const docSpread = Math.max(1, entity.documentIds?.length ?? entity.sourceChunkIds.length);
        if (docSpread > 5) continue;
        let bestAnchorRank = 2;
        for (const cid of entity.sourceChunkIds) {
          const r = anchorRankByChunk.get(cid);
          if (r !== undefined && r < bestAnchorRank) bestAnchorRank = r;
        }
        const idf = Math.max(0.25, Math.min(1.0, 1 / Math.log2(2 + docSpread)));
        const anchorBoost = (0.95 / (1 + bestAnchorRank * 0.45)) * idf;
        seedWeights.set(entity.canonicalName, (seedWeights.get(entity.canonicalName) ?? 0) + anchorBoost);
      }

      try {
        const subgraph = await this.kgRepo.getTenantSubgraph(input.orgId, [...seedWeights.keys()], {
          maxHops: 2,
          maxEdges: 240,
          documentIds: input.documentIds,
        });
        if (subgraph.edges.length > 0) {
          // Adaptive hub dampening (omitting hubDampening lets PPR calibrate to graph density in [0.35, 0.70])
          const ppr = computePersonalizedPageRank(subgraph.edges, seedWeights, {
            damping: 0.85,
            iterations: 12,
          });
          pprChunkScores = propagatePprToChunks({
            nodeScores: ppr.nodeScores,
            entityChunks: subgraph.entityChunks,
            edges: subgraph.edges,
          });
          pprPaths = traceTopPprPaths(subgraph.edges, [...seedWeights.keys()], ppr.nodeScores, 5);
        }
      } catch {
        /* PPR is additive; gracefully degrade if subgraph lookup fails */
      }
    }
    const pprMs = Date.now() - pprStart;

    // 6. IDF-Dampened Provenance Chunk Collection across Entities, Relations, and PPR Bridge Chunks
    const entityChunkStats = new Map<string, { bestScore: number; bestRank: number; freq: number }>();
    mergedEntities.slice(0, 12).forEach((entity, rank) => {
      const docSpread = Math.max(1, entity.documentIds?.length ?? entity.sourceChunkIds.length);
      const idf = 1 / Math.log2(2 + docSpread);
      const precisionHit = (1 / (1 + rank * 0.35)) * Math.min(1, idf * 1.45);
      for (const cid of entity.sourceChunkIds.slice(0, CONSTANTS.DEFAULT_RELATED_CHUNKS_PER_ITEM)) {
        const cur = entityChunkStats.get(cid);
        if (cur) {
          cur.freq++;
          cur.bestRank = Math.min(cur.bestRank, rank);
          cur.bestScore = Math.max(cur.bestScore, precisionHit);
        } else {
          entityChunkStats.set(cid, { bestScore: precisionHit, bestRank: rank, freq: 1 });
        }
      }
    });

    const relationChunkStats = new Map<string, { bestScore: number; bestRank: number; freq: number; bridgeBoost: number }>();
    mergedRelations.slice(0, 12).forEach((rel, rank) => {
      const isCrossDocBridge = rel.documentIds.length > 1;
      const docSpread = Math.max(1, rel.documentIds?.length ?? rel.sourceChunkIds.length);
      const idf = 1 / Math.log2(2 + docSpread);
      const precisionHit = (1 / (1 + rank * 0.35)) * Math.min(1, idf * 1.45);
      for (const cid of rel.sourceChunkIds.slice(0, CONSTANTS.DEFAULT_RELATED_CHUNKS_PER_ITEM)) {
        const cur = relationChunkStats.get(cid);
        if (cur) {
          cur.freq++;
          cur.bestRank = Math.min(cur.bestRank, rank);
          cur.bestScore = Math.max(cur.bestScore, precisionHit);
          if (isCrossDocBridge) cur.bridgeBoost = Math.max(cur.bridgeBoost, 0.12);
        } else {
          relationChunkStats.set(cid, {
            bestScore: precisionHit,
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
        governanceFilter,
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
          entityHit: eStat ? eStat.bestScore : 0,
          relationHit: rStat ? rStat.bestScore : 0,
          pprScore,
          maxsimScore: row.similarity,
          crossEncoderScore: 0,
          siblingHits: 1,
        };

        if (eStat) {
          const freqScore = eStat.freq / maxEntityFreq;
          const combined = 0.72 * row.similarity + 0.10 * eStat.bestScore + 0.06 * freqScore + 0.12 * pprScore;
          entityChunks.push({ ...baseCandidate, similarity: combined, source: 'entity' });
        }

        if (rStat && !eStat) {
          const freqScore = rStat.freq / maxRelFreq;
          const combined =
            0.70 * row.similarity + 0.10 * rStat.bestScore + 0.05 * freqScore + rStat.bridgeBoost * 0.4 + 0.12 * pprScore;
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

    // 8. Temporal & Authority Conflict Resolution (#10) + Query-Adaptive Calibration + 12-Feature LTR (#12)
    const temporalIntent = parseTemporalIntent(standaloneQuery);
    const temporalAnnotations = resolveTemporalAndAuthorityConflicts(candidates, temporalIntent);
    const queryTerms = extractSignificantTerms(standaloneQuery);
    const effectiveExactCodes = [
      ...new Set([
        ...plan.features.exactCodes,
        ...[...standaloneQuery.matchAll(/\b([A-Z]{2,6}-[A-Z0-9]{2,}(?:-[A-Z0-9]{2,})?)\b/g)]
          .map((m) => m[1]!)
          .filter(Boolean),
      ]),
    ];

    // Calibrate dense similarities per query to counter embedding anisotropy (narrow cosine cones)
    const calibratedDense = calibrateDenseScores(candidates.map((c) => c.denseSim));

    // Compute parent-coherence support (strictly for real parent-child chunks, neutral 0.5 for standalone)
    const parentHitCounts = new Map<string, number>();
    for (const c of candidates) {
      if (c.parentChunkId) {
        parentHitCounts.set(c.parentChunkId, (parentHitCounts.get(c.parentChunkId) ?? 0) + 1);
      }
    }

    const scoredWithFeatures: Array<{ candidate: InternalCandidate; features: RetrievalFeatureVector; finalScore: number }> = [];
    for (let idx = 0; idx < candidates.length; idx++) {
      const c = candidates[idx]!;
      const tAnn = temporalAnnotations.get(c.documentId);
      const versionActive = tAnn ? tAnn.versionMatchScore : 1.0;
      const parentCoherence = c.parentChunkId
        ? Math.min(1, (parentHitCounts.get(c.parentChunkId) ?? 1) / 3)
        : 0.5;
      const denseCal = calibratedDense[idx] ?? Math.max(0, Math.min(1, c.denseSim));

      const features: RetrievalFeatureVector = {
        denseSim: denseCal,
        bm25Norm: Math.max(0, Math.min(1, c.bm25Norm)),
        entityHit: Math.max(0, Math.min(1, c.entityHit)),
        relationHit: Math.max(0, Math.min(1, c.relationHit)),
        pprScore: Math.max(0, Math.min(1, c.pprScore)),
        crossEncoderScore: usedCrossEncoder
          ? c.crossEncoderScore
          : Math.max(0, Math.min(1, 0.5 * denseCal + 0.5 * Math.max(0, Math.min(1, c.bm25Norm)))),
        recencyScore: tAnn?.recencyScore ?? 0.7,
        authorityScore: tAnn?.authorityScore ?? 0.5,
        versionActive,
        headingMatch: computeHeadingMatchScore(queryTerms, c.sectionHeading),
        exactCodeMatch: computeExactCodeMatchScore(effectiveExactCodes, c.content, c.documentTitle),
        parentCoherence,
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

    // Post-LTR near-duplicate suppression, per-document diversity cap, and adaptive Top-K
    const aboveFloor = scoredWithFeatures.filter((s) => s.finalScore >= rerankMinScore);
    const poolForSelection = aboveFloor.length > 0 ? aboveFloor : scoredWithFeatures.slice(0, Math.min(3, rerankLimit));

    const adaptiveLimit = computeAdaptiveTopK(
      intent,
      rerankLimit,
      poolForSelection.map((s) => s.finalScore),
      (poolForSelection[0]?.features.exactCodeMatch ?? 0) >= 1,
    );
    const maxPerDocument =
      intent === 'exact_lookup' || intent === 'factual' ? 3 : 2;

    let keptScored = selectDiverseNonDuplicateCandidates(
      poolForSelection.map((s) => ({
        id: s.candidate.id,
        documentId: s.candidate.documentId,
        content: s.candidate.content,
        finalScore: s.finalScore,
        ref: s,
      })),
      { limit: adaptiveLimit, maxPerDocument, nearDuplicateThreshold: 0.85 },
    ).map((x) => x.ref);

    // 9. Bounded CRAG Confidence Gate (#6 — Zero LLM grading, max 1 escalation retry)
    // Uses unstretched raw denseSim cap alongside LTR finalScore so flat low-sim noise never fools CRAG.
    const crag = evaluateCragConfidence(
      standaloneQuery,
      keptScored.map((s) => ({
        id: s.candidate.id,
        content: s.candidate.content,
        score: Math.min(s.finalScore, Math.max(s.candidate.denseSim, s.candidate.bm25Norm * 0.75, s.features.exactCodeMatch)),
      })),
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

    const ambiguity = detectQueryAmbiguity({
      query: input.userQuery,
      hasHistoryAnchor: history.length > 0,
      exactCodes: effectiveExactCodes,
      topCandidates: keptScored.slice(0, 3).map((s) => ({
        documentId: s.candidate.documentId,
        documentTitle: s.candidate.documentTitle,
        finalScore: s.finalScore,
      })),
    });

    const contradictions = detectEvidenceContradictions(
      keptScored.map((s) => ({
        id: s.candidate.id,
        documentId: s.candidate.documentId,
        documentTitle: s.candidate.documentTitle,
        content: s.candidate.content,
        temporalBadge: s.candidate.temporalBadge,
      })),
    );

    const multiHopCoverage = verifyMultiHopEvidenceCoverage(
      intent,
      pprPaths,
      keptScored.map((s) => ({ documentId: s.candidate.documentId, content: s.candidate.content })),
    );

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
      facets: plan.features,
      strategy,
      configVersion: config.version,
      subQueries,
      crag,
      ambiguity,
      contradictions,
      multiHopCoverage,
      pprPaths,
      historyInjectionsNeutralized,
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
