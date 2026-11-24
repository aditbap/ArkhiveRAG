import type { RetrievalEvaluationSummary } from './optimizer.js';

export interface RegressionGateThresholds {
  /** Minimum NDCG@5 delta vs baseline (e.g. -0.002 allows non-regression; +0.005 requires gain). */
  minNdcgDelta?: number;
  /** Maximum allowed Full-Recall@5 drop vs baseline. Default 0.005. */
  maxRecallDrop?: number;
  /** Minimum absolute faithfulness required. Default 0.90. */
  minFaithfulness?: number;
  /** Maximum allowed p95 latency multiplier vs baseline. Default 1.15 (+15%). */
  maxP95LatencyRatio?: number;
  /** Maximum allowed avg context token multiplier vs baseline (Section 3.5 contract: <= +5%). */
  maxTokenRatio?: number;
  /** Require zero multi-tenant / ACL security leaks. Default true. */
  requireZeroAclLeaks?: boolean;
}

export interface RegressionGateCheck {
  name: string;
  passed: boolean;
  baseline: number;
  challenger: number;
  threshold: string;
  reason: string;
}

export interface RegressionGateReport {
  passed: boolean;
  checks: RegressionGateCheck[];
  summary: string;
}

/**
 * Multi-Metric Deployment Regression Gate for `RetrievalLearningEngine` (Step 7):
 * Blocks promotion of any candidate retrieval configuration unless ALL quality,
 * faithfulness, latency, token-efficiency, and security invariants pass.
 */
export function runRegressionGate(
  baseline: RetrievalEvaluationSummary,
  challenger: RetrievalEvaluationSummary,
  thresholds: RegressionGateThresholds = {},
): RegressionGateReport {
  const minNdcgDelta = thresholds.minNdcgDelta ?? -0.002;
  const maxRecallDrop = thresholds.maxRecallDrop ?? 0.005;
  const minFaithfulness = thresholds.minFaithfulness ?? 0.90;
  const maxP95LatencyRatio = thresholds.maxP95LatencyRatio ?? 1.15;
  const maxTokenRatio = thresholds.maxTokenRatio ?? 1.05;
  const requireZeroAclLeaks = thresholds.requireZeroAclLeaks ?? true;

  const checks: RegressionGateCheck[] = [];

  // 1. NDCG@5 non-regression / gain
  const ndcgDelta = challenger.ndcgAt5 - baseline.ndcgAt5;
  checks.push({
    name: 'ndcg@5',
    passed: ndcgDelta >= minNdcgDelta - 1e-6,
    baseline: baseline.ndcgAt5,
    challenger: challenger.ndcgAt5,
    threshold: `delta >= ${minNdcgDelta}`,
    reason: `delta=${ndcgDelta.toFixed(4)}`,
  });

  // 2. Full-Recall@5 non-regression
  const recallDrop = baseline.fullRecallAt5 - challenger.fullRecallAt5;
  checks.push({
    name: 'full_recall@5',
    passed: recallDrop <= maxRecallDrop + 1e-6,
    baseline: baseline.fullRecallAt5,
    challenger: challenger.fullRecallAt5,
    threshold: `drop <= ${maxRecallDrop}`,
    reason: `drop=${recallDrop.toFixed(4)}`,
  });

  // 3. Faithfulness floor
  checks.push({
    name: 'faithfulness',
    passed: challenger.faithfulness >= minFaithfulness - 1e-6,
    baseline: baseline.faithfulness,
    challenger: challenger.faithfulness,
    threshold: `>= ${minFaithfulness}`,
    reason: `score=${challenger.faithfulness.toFixed(4)}`,
  });

  // 4. p95 Latency ceiling
  const latencyCap = baseline.p95LatencyMs > 0 ? baseline.p95LatencyMs * maxP95LatencyRatio : Infinity;
  checks.push({
    name: 'p95_latency_ms',
    passed: challenger.p95LatencyMs <= latencyCap + 1e-6,
    baseline: baseline.p95LatencyMs,
    challenger: challenger.p95LatencyMs,
    threshold: `<= ${Number.isFinite(latencyCap) ? latencyCap.toFixed(1) : 'N/A'}ms (${maxP95LatencyRatio}x)`,
    reason: `challenger=${challenger.p95LatencyMs.toFixed(1)}ms`,
  });

  // 5. Token-Efficiency ceiling (<= +5% avg context tokens unless quality jumped dramatically)
  const tokenCap = baseline.avgContextTokens > 0 ? baseline.avgContextTokens * maxTokenRatio : Infinity;
  const dramaticQualityJump = ndcgDelta >= 0.03 && challenger.avgContextTokens <= baseline.avgContextTokens * 1.12;
  checks.push({
    name: 'avg_context_tokens',
    passed: challenger.avgContextTokens <= tokenCap + 1e-6 || dramaticQualityJump,
    baseline: baseline.avgContextTokens,
    challenger: challenger.avgContextTokens,
    threshold: `<= ${Number.isFinite(tokenCap) ? tokenCap.toFixed(1) : 'N/A'} (${maxTokenRatio}x)`,
    reason: dramaticQualityJump
      ? `allowed via +${ndcgDelta.toFixed(3)} NDCG jump`
      : `challenger=${challenger.avgContextTokens.toFixed(1)} tok`,
  });

  // 6. Zero multi-tenant / ACL leak security invariant
  if (requireZeroAclLeaks) {
    checks.push({
      name: 'acl_tenant_isolation',
      passed: challenger.aclLeakCount === 0,
      baseline: baseline.aclLeakCount,
      challenger: challenger.aclLeakCount,
      threshold: '== 0 leaks',
      reason: `leaks=${challenger.aclLeakCount}`,
    });
  }

  const failed = checks.filter((c) => !c.passed);
  const passed = failed.length === 0;
  const summary = passed
    ? `PASSED all ${checks.length} gates (NDCG@5: ${baseline.ndcgAt5} -> ${challenger.ndcgAt5}, Tokens: ${baseline.avgContextTokens} -> ${challenger.avgContextTokens})`
    : `FAILED ${failed.length}/${checks.length} gates: ${failed.map((f) => `${f.name} (${f.reason})`).join(', ')}`;

  return { passed, checks, summary };
}
