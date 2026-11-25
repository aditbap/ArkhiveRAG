import crypto from 'crypto';
import {
  DEFAULT_RETRIEVAL_CONFIG,
  optimizeRetrievalWeights,
  runRegressionGate,
  type LearningEvalCase,
  type QueryIntent,
  type RegressionGateReport,
  type RegressionGateThresholds,
  type RetrievalConfigPayload,
  type RetrievalEvaluationSummary,
  type RetrievalStrategyName,
  type WeightOptimizationOptions,
} from '@rag/core';
import { LearningRepository, type RecordFeedbackInput } from '@rag/database';
import type { FastCacheBridge, LightRagRetrievalResult, LightRagRetriever } from './lightrag-retriever.js';

export interface ShadowStrategyOutcome {
  strategy: RetrievalStrategyName;
  effectiveMode: string;
  latencyMs: number;
  contextTokens: number;
  topDocIds: string[];
  cragVerdict: string;
  cragConfidence: number;
  overlapWithBaseline: number;
}

export interface OptimizeAndGateResult {
  configId: string | null;
  version: number;
  promoted: boolean;
  baselineSummary: RetrievalEvaluationSummary;
  challengerSummary: RetrievalEvaluationSummary;
  gateReport: RegressionGateReport;
  updatedIntents: QueryIntent[];
}

/**
 * `RetrievalLearningEngine` — Closed-Loop Adaptive Retrieval Orchestration & Learning System.
 *
 * Implements a strictly decoupled 3-plane corporate design so learning NEVER penalizes
 * live user latency or burns generation tokens:
 * 1. Online Serving Plane: loads active versioned config from Redis (60s TTL) -> Postgres,
 *    and records compact retrieval telemetry asynchronously.
 * 2. Async Shadow Exploration Plane (default 2% sample, retrieval-only = 0 generation tokens):
 *    replays sampled queries across challenger strategies off the critical path.
 * 3. Offline Optimizer + Regression Gate Plane: runs L2-regularized coordinate ascent with
 *    bounded step clamps (`|Δw| <= 0.15`), validates every quality/latency/token/ACL gate,
 *    and supports atomic promotion + 1-click rollback.
 */
export class RetrievalLearningEngine {
  private readonly repo: LearningRepository;
  private readonly fastCache: FastCacheBridge | null;
  private readonly shadowSampleRate: number;
  private readonly dailyShadowCap: number;
  private dailyShadowCount = 0;
  private dailyShadowDay = new Date().toISOString().slice(0, 10);

  constructor(deps: {
    repo?: LearningRepository;
    fastCache?: FastCacheBridge | null;
    shadowSampleRate?: number;
    dailyShadowCap?: number;
  } = {}) {
    this.repo = deps.repo ?? new LearningRepository();
    this.fastCache = deps.fastCache ?? null;
    this.shadowSampleRate =
      deps.shadowSampleRate ?? parseFloat(process.env.SHADOW_SAMPLE_RATE || '0.02');
    this.dailyShadowCap =
      deps.dailyShadowCap ?? parseInt(process.env.LEARNING_DAILY_SHADOW_CAP || '200', 10);
  }

  private configCacheKey(orgId: string): string {
    return `rag:learning:active-config:${orgId}`;
  }

  /**
   * Loads the active immutable retrieval config for `orgId` (`<2ms` via Redis cache).
   */
  async getActiveConfig(orgId: string): Promise<{ id: string | null; config: RetrievalConfigPayload }> {
    const cacheKey = this.configCacheKey(orgId);
    if (this.fastCache) {
      try {
        const cached = await this.fastCache.get(cacheKey);
        if (cached) {
          return JSON.parse(cached) as { id: string | null; config: RetrievalConfigPayload };
        }
      } catch {
        /* best-effort */
      }
    }

    try {
      const row = await this.repo.getActiveConfigRow(orgId);
      const result = row
        ? { id: row.id, config: row.configJson as RetrievalConfigPayload }
        : { id: null, config: DEFAULT_RETRIEVAL_CONFIG };

      if (this.fastCache) {
        void this.fastCache.setex(cacheKey, 60, JSON.stringify(result)).catch(() => undefined);
      }
      return result;
    } catch {
      return { id: null, config: DEFAULT_RETRIEVAL_CONFIG };
    }
  }

  private async invalidateConfigCache(orgId: string): Promise<void> {
    if (!this.fastCache) return;
    try {
      await this.fastCache.setex(this.configCacheKey(orgId), 1, '');
    } catch {
      /* ignore */
    }
  }

  /**
   * Records non-blocking retrieval telemetry after a query executes.
   */
  async recordQueryTelemetry(params: {
    orgId: string;
    messageId?: string | null;
    queryText: string;
    retrieval: LightRagRetrievalResult;
    promptTokens?: number;
    completionTokens?: number;
  }): Promise<string | null> {
    const queryHash = crypto
      .createHash('sha256')
      .update(params.queryText.trim().toLowerCase())
      .digest('hex');

    try {
      return await this.repo.recordTelemetry({
        orgId: params.orgId,
        messageId: params.messageId ?? null,
        configVersion: params.retrieval.configVersion,
        queryHash,
        queryText: params.queryText,
        intent: params.retrieval.intent,
        strategyUsed: params.retrieval.strategy,
        effectiveMode: params.retrieval.effectiveMode,
        cragVerdict: params.retrieval.crag.verdict,
        cragAction: params.retrieval.crag.action,
        candidateFeatures: params.retrieval.candidateFeatures,
        timings: params.retrieval.timingsMs,
        latencyMs: params.retrieval.timingsMs.total,
        contextTokens: params.retrieval.assembled.estimatedTokens.total,
        promptTokens: params.promptTokens ?? 0,
        completionTokens: params.completionTokens ?? 0,
      });
    } catch {
      return null;
    }
  }

  /**
   * Decides whether to schedule an async shadow comparison (respecting sample rate + daily cap).
   */
  shouldRunShadow(config: RetrievalConfigPayload): boolean {
    const today = new Date().toISOString().slice(0, 10);
    if (today !== this.dailyShadowDay) {
      this.dailyShadowDay = today;
      this.dailyShadowCount = 0;
    }
    if (this.dailyShadowCount >= this.dailyShadowCap) return false;
    const rate = Math.min(this.shadowSampleRate, config.shadowSampleRate ?? this.shadowSampleRate);
    if (rate <= 0) return false;
    if (Math.random() < rate) {
      this.dailyShadowCount++;
      return true;
    }
    return false;
  }

  /**
   * Runs retrieval-only multi-strategy shadow evaluation (0 generation LLM tokens,
   * zero impact on live user latency when invoked asynchronously).
   */
  async runShadowComparison(params: {
    orgId: string;
    query: string;
    retriever: LightRagRetriever;
    baselineResult: LightRagRetrievalResult;
    challengerStrategies?: RetrievalStrategyName[];
    documentIds?: string[];
  }): Promise<ShadowStrategyOutcome[]> {
    const strategies: RetrievalStrategyName[] = params.challengerStrategies ?? [
      'fast_hybrid',
      'ppr_mix',
      'temporal_hybrid',
      'decomposed_ppr',
    ];
    const baselineDocs = new Set(params.baselineResult.assembled.chunksUsed.map((c) => c.documentId));
    const challengers = strategies.filter((s) => s !== params.baselineResult.strategy);

    const outcomes: ShadowStrategyOutcome[] = [];
    for (const strategy of challengers) {
      try {
        const res = await params.retriever.retrieve({
          orgId: params.orgId,
          userQuery: params.query,
          documentIds: params.documentIds,
          strategyOverride: strategy,
          disableCragRetry: true,
        });
        const topDocIds = [...new Set(res.assembled.chunksUsed.map((c) => c.documentId))].slice(0, 5);
        const overlapCount = topDocIds.filter((id) => baselineDocs.has(id)).length;
        const overlap = topDocIds.length > 0 ? overlapCount / topDocIds.length : 1;

        outcomes.push({
          strategy,
          effectiveMode: res.effectiveMode,
          latencyMs: res.timingsMs.total,
          contextTokens: res.assembled.estimatedTokens.total,
          topDocIds,
          cragVerdict: res.crag.verdict,
          cragConfidence: Number(res.crag.confidence.toFixed(4)),
          overlapWithBaseline: Number(overlap.toFixed(4)),
        });
      } catch {
        /* shadow failures never affect production */
      }
    }

    if (outcomes.length > 0) {
      await this.repo
        .recordExperiment({
          orgId: params.orgId,
          experimentType: 'shadow',
          strategyName: `${params.baselineResult.strategy}_vs_challengers`,
          queryCount: 1,
          metrics: {
            query: params.query.slice(0, 200),
            intent: params.baselineResult.intent,
            baseline: {
              strategy: params.baselineResult.strategy,
              latencyMs: params.baselineResult.timingsMs.total,
              contextTokens: params.baselineResult.assembled.estimatedTokens.total,
              cragConfidence: params.baselineResult.crag.confidence,
            },
            challengers: outcomes,
          },
          gatePassed: null,
        })
        .catch(() => undefined);
    }

    return outcomes;
  }

  async recordFeedback(input: RecordFeedbackInput) {
    return this.repo.recordFeedback(input);
  }

  /**
   * Controlled Weight Optimization + Mandatory Regression Gate (Steps 5-7):
   * Optimizes per-intent LTR weights over golden cases (and/or feedback-labeled telemetry),
   * runs every quality/latency/token/security regression gate, persists the config & report,
   * and promotes atomically ONLY if `autoPromote=true` AND `gateReport.passed === true`.
   */
  async optimizeAndGate(params: {
    orgId: string;
    cases: LearningEvalCase[];
    createdBy?: string | null;
    autoPromote?: boolean;
    optimizationOptions?: WeightOptimizationOptions;
    gateThresholds?: RegressionGateThresholds;
    notes?: string;
  }): Promise<OptimizeAndGateResult> {
    const active = await this.getActiveConfig(params.orgId);
    const opt = optimizeRetrievalWeights(params.cases, active.config, params.optimizationOptions);
    const gateReport = runRegressionGate(opt.baselineSummary, opt.challengerSummary, params.gateThresholds);

    const nextVersion = await this.repo.getNextVersion(params.orgId);
    opt.candidateConfig.version = nextVersion;

    const saved = await this.repo.createConfig({
      orgId: params.orgId,
      version: nextVersion,
      status: 'draft',
      config: opt.candidateConfig,
      evalMetrics: opt.challengerSummary,
      parentConfigId: active.id,
      createdBy: params.createdBy ?? null,
      notes: params.notes ?? gateReport.summary,
    });

    await this.repo.recordExperiment({
      orgId: params.orgId,
      experimentType: 'regression',
      baselineConfigId: active.id,
      challengerConfigId: saved?.id ?? null,
      strategyName: `ltr_opt_v${nextVersion}`,
      queryCount: params.cases.length,
      metrics: {
        baseline: opt.baselineSummary,
        challenger: opt.challengerSummary,
        updatedIntents: opt.updatedIntents,
        weightDeltas: opt.weightDeltas,
      },
      gatePassed: gateReport.passed,
      gateReport,
    });

    let promoted = false;
    if (params.autoPromote && gateReport.passed && saved?.id) {
      const res = await this.repo.promoteConfig(params.orgId, saved.id);
      if (res) {
        promoted = true;
        await this.invalidateConfigCache(params.orgId);
      }
    }

    return {
      configId: saved?.id ?? null,
      version: nextVersion,
      promoted,
      baselineSummary: opt.baselineSummary,
      challengerSummary: opt.challengerSummary,
      gateReport,
      updatedIntents: opt.updatedIntents,
    };
  }

  async promoteConfig(orgId: string, configId: string) {
    const promoted = await this.repo.promoteConfig(orgId, configId);
    if (promoted) {
      await this.invalidateConfigCache(orgId);
    }
    return promoted;
  }

  async listConfigs(orgId: string) {
    return this.repo.listConfigs(orgId);
  }

  async getMetricsSummary(orgId: string, windowDays = 7) {
    return this.repo.getMetricsSummary(orgId, windowDays);
  }
}
