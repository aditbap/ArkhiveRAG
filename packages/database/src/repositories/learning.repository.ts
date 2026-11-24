import { BaseRepository } from './base.repository.js';
import {
  retrievalConfigs,
  retrievalTelemetry,
  retrievalFeedback,
  retrievalExperiments,
} from '../schema/learning.js';
import type { RetrievalConfigPayload, RetrievalEvaluationSummary, RegressionGateReport } from '@rag/core';
import { and, desc, eq, gte, isNull, or, sql } from 'drizzle-orm';

export interface RecordTelemetryInput {
  orgId: string;
  messageId?: string | null;
  configVersion: number;
  queryHash: string;
  queryText: string;
  intent: string;
  strategyUsed: string;
  effectiveMode: string;
  cragVerdict: string;
  cragAction: string;
  candidateFeatures: unknown[];
  timings: Record<string, number>;
  latencyMs: number;
  contextTokens: number;
  promptTokens?: number;
  completionTokens?: number;
}

export interface RecordFeedbackInput {
  orgId: string;
  telemetryId?: string | null;
  messageId?: string | null;
  userId?: string | null;
  rating: -1 | 1;
  citedChunkIds?: string[];
  reasonCode?: string | null;
  comment?: string | null;
}

export interface CreateExperimentInput {
  orgId: string;
  experimentType: 'shadow' | 'regression' | 'ablation';
  baselineConfigId?: string | null;
  challengerConfigId?: string | null;
  strategyName: string;
  queryCount: number;
  metrics: Record<string, unknown>;
  gatePassed?: boolean | null;
  gateReport?: RegressionGateReport | null;
}

export class LearningRepository extends BaseRepository {
  constructor() {
    super('retrieval_configs');
  }

  async getActiveConfigRow(orgId: string) {
    this.requireTenant(orgId);
    const rows = await this.db
      .select()
      .from(retrievalConfigs)
      .where(
        and(
          eq(retrievalConfigs.status, 'active'),
          or(eq(retrievalConfigs.orgId, orgId), isNull(retrievalConfigs.orgId)),
        ),
      )
      .orderBy(sql`CASE WHEN ${retrievalConfigs.orgId} IS NOT NULL THEN 0 ELSE 1 END`, desc(retrievalConfigs.version))
      .limit(1);
    return rows[0] ?? null;
  }

  async listConfigs(orgId: string, limit = 20) {
    this.requireTenant(orgId);
    return this.db
      .select()
      .from(retrievalConfigs)
      .where(or(eq(retrievalConfigs.orgId, orgId), isNull(retrievalConfigs.orgId)))
      .orderBy(desc(retrievalConfigs.version), desc(retrievalConfigs.createdAt))
      .limit(limit);
  }

  async getNextVersion(orgId: string): Promise<number> {
    this.requireTenant(orgId);
    const rows = await this.db
      .select({ maxVer: sql<number>`COALESCE(MAX(${retrievalConfigs.version}), 1)` })
      .from(retrievalConfigs)
      .where(eq(retrievalConfigs.orgId, orgId));
    return Number(rows[0]?.maxVer ?? 1) + 1;
  }

  async createConfig(params: {
    orgId: string;
    version: number;
    status: 'draft' | 'shadow' | 'active' | 'rolled_back';
    config: RetrievalConfigPayload;
    evalMetrics?: RetrievalEvaluationSummary | null;
    parentConfigId?: string | null;
    createdBy?: string | null;
    notes?: string | null;
  }) {
    this.requireTenant(params.orgId);
    const [row] = await this.db
      .insert(retrievalConfigs)
      .values({
        orgId: params.orgId,
        version: params.version,
        status: params.status,
        configJson: params.config,
        evalMetricsJson: params.evalMetrics ?? null,
        parentConfigId: params.parentConfigId ?? null,
        createdBy: params.createdBy ?? null,
        notes: params.notes ?? null,
        promotedAt: params.status === 'active' ? new Date() : null,
      })
      .returning();
    return row;
  }

  async promoteConfig(orgId: string, configId: string) {
    this.requireTenant(orgId);
    return this.db.transaction(async (tx) => {
      const [target] = await tx
        .select()
        .from(retrievalConfigs)
        .where(and(eq(retrievalConfigs.id, configId), eq(retrievalConfigs.orgId, orgId)))
        .limit(1);
      if (!target) return null;

      await tx
        .update(retrievalConfigs)
        .set({ status: 'rolled_back' })
        .where(and(eq(retrievalConfigs.orgId, orgId), eq(retrievalConfigs.status, 'active')));

      const [promoted] = await tx
        .update(retrievalConfigs)
        .set({ status: 'active', promotedAt: new Date() })
        .where(and(eq(retrievalConfigs.id, configId), eq(retrievalConfigs.orgId, orgId)))
        .returning();

      return promoted ?? null;
    });
  }

  async recordTelemetry(input: RecordTelemetryInput) {
    this.requireTenant(input.orgId);
    const [row] = await this.db
      .insert(retrievalTelemetry)
      .values({
        orgId: input.orgId,
        messageId: input.messageId ?? null,
        configVersion: input.configVersion,
        queryHash: input.queryHash,
        queryText: input.queryText.slice(0, 2000),
        intent: input.intent,
        strategyUsed: input.strategyUsed,
        effectiveMode: input.effectiveMode,
        cragVerdict: input.cragVerdict,
        cragAction: input.cragAction,
        candidateFeaturesJson: input.candidateFeatures,
        timingsJson: input.timings,
        latencyMs: Math.round(input.latencyMs),
        contextTokens: Math.round(input.contextTokens),
        promptTokens: Math.round(input.promptTokens ?? 0),
        completionTokens: Math.round(input.completionTokens ?? 0),
      })
      .returning({ id: retrievalTelemetry.id });
    return row?.id ?? null;
  }

  async recordFeedback(input: RecordFeedbackInput) {
    this.requireTenant(input.orgId);
    const [row] = await this.db
      .insert(retrievalFeedback)
      .values({
        orgId: input.orgId,
        telemetryId: input.telemetryId ?? null,
        messageId: input.messageId ?? null,
        userId: input.userId ?? null,
        rating: input.rating,
        citedChunkIds: input.citedChunkIds ?? [],
        reasonCode: input.reasonCode ?? null,
        comment: input.comment ?? null,
      })
      .returning();
    return row;
  }

  async recordExperiment(input: CreateExperimentInput) {
    this.requireTenant(input.orgId);
    const [row] = await this.db
      .insert(retrievalExperiments)
      .values({
        orgId: input.orgId,
        experimentType: input.experimentType,
        baselineConfigId: input.baselineConfigId ?? null,
        challengerConfigId: input.challengerConfigId ?? null,
        strategyName: input.strategyName,
        queryCount: input.queryCount,
        metricsJson: input.metrics,
        gatePassed: input.gatePassed ?? null,
        gateReportJson: input.gateReport ?? null,
      })
      .returning();
    return row;
  }

  async listRecentTelemetryWithFeedback(orgId: string, limit = 200) {
    this.requireTenant(orgId);
    const rows = await this.db
      .select({
        telemetry: retrievalTelemetry,
        feedback: retrievalFeedback,
      })
      .from(retrievalTelemetry)
      .leftJoin(retrievalFeedback, eq(retrievalFeedback.telemetryId, retrievalTelemetry.id))
      .where(eq(retrievalTelemetry.orgId, orgId))
      .orderBy(desc(retrievalTelemetry.createdAt))
      .limit(limit);
    return rows;
  }

  async getMetricsSummary(orgId: string, windowDays = 7) {
    this.requireTenant(orgId);
    const since = new Date(Date.now() - windowDays * 86_400_000);

    const telemetryRows = await this.db
      .select({
        intent: retrievalTelemetry.intent,
        strategyUsed: retrievalTelemetry.strategyUsed,
        cragVerdict: retrievalTelemetry.cragVerdict,
        latencyMs: retrievalTelemetry.latencyMs,
        contextTokens: retrievalTelemetry.contextTokens,
      })
      .from(retrievalTelemetry)
      .where(and(eq(retrievalTelemetry.orgId, orgId), gte(retrievalTelemetry.createdAt, since)))
      .limit(5000);

    const feedbackRows = await this.db
      .select({ rating: retrievalFeedback.rating })
      .from(retrievalFeedback)
      .where(and(eq(retrievalFeedback.orgId, orgId), gte(retrievalFeedback.createdAt, since)))
      .limit(5000);

    const experiments = await this.db
      .select()
      .from(retrievalExperiments)
      .where(eq(retrievalExperiments.orgId, orgId))
      .orderBy(desc(retrievalExperiments.createdAt))
      .limit(10);

    const totalQueries = telemetryRows.length;
    const latencies = telemetryRows.map((r) => r.latencyMs).sort((a, b) => a - b);
    const avgLatencyMs = totalQueries > 0 ? latencies.reduce((a, b) => a + b, 0) / totalQueries : 0;
    const p95LatencyMs = totalQueries > 0 ? latencies[Math.min(totalQueries - 1, Math.floor(totalQueries * 0.95))] ?? 0 : 0;
    const avgContextTokens =
      totalQueries > 0 ? telemetryRows.reduce((s, r) => s + r.contextTokens, 0) / totalQueries : 0;

    const byIntent: Record<string, number> = {};
    const byStrategy: Record<string, number> = {};
    const byCragVerdict: Record<string, number> = {};
    for (const r of telemetryRows) {
      byIntent[r.intent] = (byIntent[r.intent] ?? 0) + 1;
      byStrategy[r.strategyUsed] = (byStrategy[r.strategyUsed] ?? 0) + 1;
      byCragVerdict[r.cragVerdict] = (byCragVerdict[r.cragVerdict] ?? 0) + 1;
    }

    const positiveFeedback = feedbackRows.filter((f) => f.rating > 0).length;
    const negativeFeedback = feedbackRows.filter((f) => f.rating < 0).length;

    return {
      windowDays,
      totalQueries,
      avgLatencyMs: Math.round(avgLatencyMs),
      p95LatencyMs: Math.round(p95LatencyMs),
      avgContextTokens: Math.round(avgContextTokens),
      byIntent,
      byStrategy,
      byCragVerdict,
      feedback: {
        total: feedbackRows.length,
        positive: positiveFeedback,
        negative: negativeFeedback,
        positiveRate: feedbackRows.length > 0 ? Number((positiveFeedback / feedbackRows.length).toFixed(4)) : null,
      },
      recentExperiments: experiments,
    };
  }
}
