import {
  LTR_FEATURE_KEYS,
  scoreCandidateWithLtr,
  type QueryIntent,
  type RetrievalConfigPayload,
  type RetrievalFeatureVector,
} from '../retrieval/intent-and-ltr.js';

export interface LearningEvalCandidate {
  chunkId: string;
  documentId: string;
  features: RetrievalFeatureVector;
}

export interface LearningEvalCase {
  query: string;
  intent: QueryIntent;
  expectedDocIds: string[];
  candidates: LearningEvalCandidate[];
  latencyMs: number;
  contextTokens: number;
  /** Optional 0..1 faithfulness score from offline judge or golden label. */
  faithfulness?: number;
  /** Optional user feedback rating (+1 / -1) used as weak supervision when golden labels are absent. */
  userRating?: number;
}

export interface RetrievalEvaluationSummary {
  caseCount: number;
  ndcgAt5: number;
  fullRecallAt5: number;
  mrr: number;
  faithfulness: number;
  avgLatencyMs: number;
  p95LatencyMs: number;
  avgContextTokens: number;
  qualityPer1kTokens: number;
  compositeScore: number;
  aclLeakCount: number;
}

function dcgAtK(relevances: number[], k: number): number {
  let sum = 0;
  for (let i = 0; i < Math.min(k, relevances.length); i++) {
    sum += (relevances[i] ?? 0) / Math.log2(i + 2);
  }
  return sum;
}

function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[idx] ?? 0;
}

/**
 * Evaluates a `RetrievalConfigPayload` across a set of labeled/logged retrieval cases
 * using the unified corporate Pareto objective:
 * `0.45*NDCG@5 + 0.30*FullRecall@5 + 0.15*Faithfulness - 0.05*LatencyNorm - 0.05*TokenNorm`
 */
export function evaluateRetrievalConfig(
  cases: LearningEvalCase[],
  config: RetrievalConfigPayload,
  options: { aclLeakCount?: number; targetLatencyMs?: number; targetTokens?: number } = {},
): RetrievalEvaluationSummary {
  if (cases.length === 0) {
    return {
      caseCount: 0,
      ndcgAt5: 0,
      fullRecallAt5: 0,
      mrr: 0,
      faithfulness: 1,
      avgLatencyMs: 0,
      p95LatencyMs: 0,
      avgContextTokens: 0,
      qualityPer1kTokens: 0,
      compositeScore: 0,
      aclLeakCount: options.aclLeakCount ?? 0,
    };
  }

  let ndcgSum = 0;
  let fullRecallSum = 0;
  let mrrSum = 0;
  let faithSum = 0;
  let faithCount = 0;
  let tokenSum = 0;
  const latencies: number[] = [];

  for (const c of cases) {
    const profile = config.intents[c.intent];
    const ranked = c.candidates
      .map((cand) => ({
        cand,
        score: scoreCandidateWithLtr(cand.features, profile.ltr),
      }))
      .sort((a, b) => b.score - a.score);

    // Deduplicate to top-5 unique documents for document-level IR metrics
    const topDocIds: string[] = [];
    for (const r of ranked) {
      if (!topDocIds.includes(r.cand.documentId)) {
        topDocIds.push(r.cand.documentId);
        if (topDocIds.length >= 5) break;
      }
    }

    const expectedSet = new Set(c.expectedDocIds);
    const rels = topDocIds.map((id) => (expectedSet.has(id) ? 1 : 0));
    const idealCount = Math.min(5, expectedSet.size);
    const idealRels = Array.from({ length: idealCount }, () => 1);
    const idcg = dcgAtK(idealRels, 5);
    const ndcg = idcg > 0 ? dcgAtK(rels, 5) / idcg : 1;

    const hits = topDocIds.filter((id) => expectedSet.has(id)).length;
    const fullRecall = expectedSet.size > 0 ? hits / expectedSet.size : 1;

    const firstHitIdx = rels.findIndex((r) => r > 0);
    const mrr = firstHitIdx >= 0 ? 1 / (firstHitIdx + 1) : 0;

    ndcgSum += ndcg;
    fullRecallSum += fullRecall;
    mrrSum += mrr;

    if (typeof c.faithfulness === 'number') {
      faithSum += Math.max(0, Math.min(1, c.faithfulness));
      faithCount++;
    }
    tokenSum += Math.max(0, c.contextTokens);
    latencies.push(Math.max(0, c.latencyMs));
  }

  const n = cases.length;
  const ndcgAt5 = ndcgSum / n;
  const fullRecallAt5 = fullRecallSum / n;
  const mrr = mrrSum / n;
  const faithfulness = faithCount > 0 ? faithSum / faithCount : 0.95;
  const avgLatencyMs = latencies.reduce((a, b) => a + b, 0) / n;
  const p95LatencyMs = percentile(latencies, 95);
  const avgContextTokens = tokenSum / n;

  const targetLat = options.targetLatencyMs ?? 1500;
  const targetTok = options.targetTokens ?? 3000;
  const latNorm = Math.min(1, avgLatencyMs / targetLat);
  const tokNorm = Math.min(1, avgContextTokens / targetTok);

  const compositeScore =
    0.45 * ndcgAt5 +
    0.30 * fullRecallAt5 +
    0.15 * faithfulness -
    0.05 * latNorm -
    0.05 * tokNorm;

  const qualityPer1kTokens = avgContextTokens > 0 ? ndcgAt5 / (avgContextTokens / 1000) : ndcgAt5;

  return {
    caseCount: n,
    ndcgAt5: Number(ndcgAt5.toFixed(4)),
    fullRecallAt5: Number(fullRecallAt5.toFixed(4)),
    mrr: Number(mrr.toFixed(4)),
    faithfulness: Number(faithfulness.toFixed(4)),
    avgLatencyMs: Number(avgLatencyMs.toFixed(1)),
    p95LatencyMs: Number(p95LatencyMs.toFixed(1)),
    avgContextTokens: Number(avgContextTokens.toFixed(1)),
    qualityPer1kTokens: Number(qualityPer1kTokens.toFixed(4)),
    compositeScore: Number(compositeScore.toFixed(4)),
    aclLeakCount: options.aclLeakCount ?? 0,
  };
}

export interface WeightOptimizationOptions {
  /** Maximum allowed absolute weight change per feature per cycle (safety clamp). Default 0.15. */
  maxStepPerWeight?: number;
  /** L2 regularization toward baseline weights to prevent overfitting. Default 0.02. */
  l2Penalty?: number;
  /** Minimum cases for an intent before updating its weights. Default 3. */
  minCasesPerIntent?: number;
  /** Number of coordinate-ascent passes. Default 2. */
  passes?: number;
}

export interface WeightOptimizationResult {
  candidateConfig: RetrievalConfigPayload;
  baselineSummary: RetrievalEvaluationSummary;
  challengerSummary: RetrievalEvaluationSummary;
  updatedIntents: QueryIntent[];
  weightDeltas: Partial<Record<QueryIntent, Partial<Record<keyof RetrievalFeatureVector, number>>>>;
}

/**
 * Controlled Offline Weight Optimizer for `RetrievalLearningEngine`:
 * Performs L2-regularized coordinate ascent over each intent's 12 LTR weights using
 * golden datasets & feedback-labeled telemetry. Strictly clamps every weight update to
 * `|Δw_i| <= maxStepPerWeight` (default 0.15) so weights evolve smoothly and safely.
 */
export function optimizeRetrievalWeights(
  cases: LearningEvalCase[],
  baseConfig: RetrievalConfigPayload,
  options: WeightOptimizationOptions = {},
): WeightOptimizationResult {
  const maxStep = options.maxStepPerWeight ?? 0.15;
  const l2Penalty = options.l2Penalty ?? 0.02;
  const minCasesPerIntent = options.minCasesPerIntent ?? 3;
  const passes = options.passes ?? 2;

  const baselineSummary = evaluateRetrievalConfig(cases, baseConfig);

  // Deep clone config for challenger
  const candidateConfig: RetrievalConfigPayload = JSON.parse(JSON.stringify(baseConfig)) as RetrievalConfigPayload;
  candidateConfig.version = baseConfig.version + 1;

  const byIntent = new Map<QueryIntent, LearningEvalCase[]>();
  for (const c of cases) {
    const arr = byIntent.get(c.intent) ?? [];
    arr.push(c);
    byIntent.set(c.intent, arr);
  }

  const updatedIntents: QueryIntent[] = [];
  const weightDeltas: Partial<Record<QueryIntent, Partial<Record<keyof RetrievalFeatureVector, number>>>> = {};
  const stepCandidates = [-0.08, -0.04, -0.02, 0.02, 0.04, 0.08];

  for (const [intent, intentCases] of byIntent.entries()) {
    if (intentCases.length < minCasesPerIntent) continue;

    const baseWeights = baseConfig.intents[intent].ltr;
    const curWeights = candidateConfig.intents[intent].ltr;

    const regularizedScore = (): number => {
      const evalRes = evaluateRetrievalConfig(intentCases, candidateConfig);
      let l2 = 0;
      for (const key of LTR_FEATURE_KEYS) {
        const diff = (curWeights[key] ?? 0) - (baseWeights[key] ?? 0);
        l2 += diff * diff;
      }
      return evalRes.compositeScore - l2Penalty * l2;
    };

    let bestObj = regularizedScore();
    let intentChanged = false;

    for (let pass = 0; pass < passes; pass++) {
      for (const key of LTR_FEATURE_KEYS) {
        const origVal = curWeights[key] ?? 0;
        const baseVal = baseWeights[key] ?? 0;
        let bestVal = origVal;

        for (const delta of stepCandidates) {
          const proposed = Math.max(0, Math.min(1, Number((origVal + delta).toFixed(4))));
          if (Math.abs(proposed - baseVal) > maxStep + 1e-6) continue;
          curWeights[key] = proposed;
          const obj = regularizedScore();
          if (obj > bestObj + 1e-5) {
            bestObj = obj;
            bestVal = proposed;
          }
        }

        curWeights[key] = bestVal;
        if (Math.abs(bestVal - baseVal) > 1e-5) {
          intentChanged = true;
        }
      }
    }

    if (intentChanged) {
      updatedIntents.push(intent);
      const deltas: Partial<Record<keyof RetrievalFeatureVector, number>> = {};
      for (const key of LTR_FEATURE_KEYS) {
        const diff = Number(((curWeights[key] ?? 0) - (baseWeights[key] ?? 0)).toFixed(4));
        if (Math.abs(diff) > 1e-5) deltas[key] = diff;
      }
      weightDeltas[intent] = deltas;
    }
  }

  const challengerSummary = evaluateRetrievalConfig(cases, candidateConfig);

  return {
    candidateConfig,
    baselineSummary,
    challengerSummary,
    updatedIntents,
    weightDeltas,
  };
}
