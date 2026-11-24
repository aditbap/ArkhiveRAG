import type { ConcreteRagQueryMode, RagQueryMode, TokenBudgetConfig } from '../types/lightrag.js';
import { parseTemporalIntent } from '../lightrag/temporal.js';

export type QueryIntent =
  | 'exact_lookup'
  | 'factual'
  | 'temporal_policy'
  | 'local_multihop'
  | 'global_synthesis'
  | 'comparative_complex';

export type RetrievalStrategyName =
  | 'fast_hybrid'
  | 'temporal_hybrid'
  | 'ppr_mix'
  | 'decomposed_ppr'
  | 'drift_global';

export interface QueryFeatures {
  wordCount: number;
  hasExactCode: boolean;
  exactCodes: string[];
  hasTemporalCue: boolean;
  allowHistorical: boolean;
  targetYear: number | null;
  hasMultiClause: boolean;
  hasComparisonCue: boolean;
  hasSynthesisCue: boolean;
  hasHistoryPronoun: boolean;
}

export interface RetrievalFeatureVector {
  denseSim: number;
  bm25Norm: number;
  entityHit: number;
  relationHit: number;
  pprScore: number;
  crossEncoderScore: number;
  recencyScore: number;
  authorityScore: number;
  versionActive: number;
  headingMatch: number;
  exactCodeMatch: number;
  parentCoherence: number;
}

export const LTR_FEATURE_KEYS: ReadonlyArray<keyof RetrievalFeatureVector> = [
  'denseSim',
  'bm25Norm',
  'entityHit',
  'relationHit',
  'pprScore',
  'crossEncoderScore',
  'recencyScore',
  'authorityScore',
  'versionActive',
  'headingMatch',
  'exactCodeMatch',
  'parentCoherence',
] as const;

export interface Stage1FusionWeights {
  dense: number;
  bm25: number;
  entity: number;
  relation: number;
  ppr: number;
  community: number;
  maxsim: number;
}

export interface IntentWeightProfile {
  strategy: RetrievalStrategyName;
  mode: ConcreteRagQueryMode;
  stage1: Stage1FusionWeights;
  ltr: Record<keyof RetrievalFeatureVector, number>;
  /** Hard per-intent context token ceiling (Section 3.5 token efficiency contract). */
  maxContextTokens: number;
  maxEntityTokens: number;
  maxRelationTokens: number;
  rerankLimit: number;
  /** Minimum post-scoring relevance required to include a chunk (prunes low-signal tail). */
  minRelevanceScore: number;
  /** Skip full parent expansion when child score >= this threshold on simple lookups. */
  skipParentExpandAbove: number;
}

export interface RetrievalConfigPayload {
  version: number;
  intents: Record<QueryIntent, IntentWeightProfile>;
  crag: {
    highConfidenceThreshold: number;
    lowConfidenceThreshold: number;
    minTermCoverageForHigh: number;
    enableRetry: boolean;
  };
  shadowSampleRate: number;
}

export const DEFAULT_RETRIEVAL_CONFIG: RetrievalConfigPayload = {
  version: 1,
  intents: {
    exact_lookup: {
      strategy: 'fast_hybrid',
      mode: 'naive',
      stage1: { dense: 1.0, bm25: 1.25, entity: 0.35, relation: 0.1, ppr: 0.15, community: 0.0, maxsim: 0.35 },
      ltr: {
        denseSim: 0.42,
        bm25Norm: 0.24,
        entityHit: 0.03,
        relationHit: 0.01,
        pprScore: 0.02,
        crossEncoderScore: 0.12,
        recencyScore: 0.01,
        authorityScore: 0.01,
        versionActive: 0.04,
        headingMatch: 0.02,
        exactCodeMatch: 0.28,
        parentCoherence: 0.01,
      },
      maxContextTokens: 1800,
      maxEntityTokens: 350,
      maxRelationTokens: 250,
      rerankLimit: 5,
      minRelevanceScore: 0.12,
      skipParentExpandAbove: 0.85,
    },
    factual: {
      strategy: 'fast_hybrid',
      mode: 'naive',
      stage1: { dense: 1.1, bm25: 0.9, entity: 0.45, relation: 0.2, ppr: 0.25, community: 0.0, maxsim: 0.4 },
      ltr: {
        denseSim: 0.52,
        bm25Norm: 0.20,
        entityHit: 0.05,
        relationHit: 0.02,
        pprScore: 0.04,
        crossEncoderScore: 0.15,
        recencyScore: 0.01,
        authorityScore: 0.01,
        versionActive: 0.04,
        headingMatch: 0.03,
        exactCodeMatch: 0.10,
        parentCoherence: 0.02,
      },
      maxContextTokens: 1900,
      maxEntityTokens: 450,
      maxRelationTokens: 350,
      rerankLimit: 5,
      minRelevanceScore: 0.12,
      skipParentExpandAbove: 0.85,
    },
    temporal_policy: {
      strategy: 'temporal_hybrid',
      mode: 'mix',
      stage1: { dense: 1.05, bm25: 0.95, entity: 0.65, relation: 0.4, ppr: 0.45, community: 0.1, maxsim: 0.3 },
      ltr: {
        denseSim: 0.38,
        bm25Norm: 0.14,
        entityHit: 0.06,
        relationHit: 0.04,
        pprScore: 0.05,
        crossEncoderScore: 0.14,
        recencyScore: 0.12,
        authorityScore: 0.08,
        versionActive: 0.20,
        headingMatch: 0.03,
        exactCodeMatch: 0.12,
        parentCoherence: 0.02,
      },
      maxContextTokens: 2600,
      maxEntityTokens: 650,
      maxRelationTokens: 550,
      rerankLimit: 6,
      minRelevanceScore: 0.12,
      skipParentExpandAbove: 0.95,
    },
    local_multihop: {
      strategy: 'ppr_mix',
      mode: 'mix',
      stage1: { dense: 1.15, bm25: 0.85, entity: 0.95, relation: 0.80, ppr: 0.90, community: 0.15, maxsim: 0.45 },
      ltr: {
        denseSim: 0.48,
        bm25Norm: 0.12,
        entityHit: 0.10,
        relationHit: 0.09,
        pprScore: 0.12,
        crossEncoderScore: 0.14,
        recencyScore: 0.01,
        authorityScore: 0.01,
        versionActive: 0.04,
        headingMatch: 0.02,
        exactCodeMatch: 0.10,
        parentCoherence: 0.03,
      },
      maxContextTokens: 3200,
      maxEntityTokens: 900,
      maxRelationTokens: 850,
      rerankLimit: 7,
      minRelevanceScore: 0.10,
      skipParentExpandAbove: 0.98,
    },
    comparative_complex: {
      strategy: 'decomposed_ppr',
      mode: 'mix',
      stage1: { dense: 1.10, bm25: 0.85, entity: 0.90, relation: 0.85, ppr: 0.85, community: 0.30, maxsim: 0.55 },
      ltr: {
        denseSim: 0.46,
        bm25Norm: 0.12,
        entityHit: 0.09,
        relationHit: 0.10,
        pprScore: 0.11,
        crossEncoderScore: 0.15,
        recencyScore: 0.01,
        authorityScore: 0.02,
        versionActive: 0.04,
        headingMatch: 0.03,
        exactCodeMatch: 0.10,
        parentCoherence: 0.04,
      },
      maxContextTokens: 3800,
      maxEntityTokens: 1000,
      maxRelationTokens: 1000,
      rerankLimit: 8,
      minRelevanceScore: 0.10,
      skipParentExpandAbove: 0.98,
    },
    global_synthesis: {
      strategy: 'drift_global',
      mode: 'mix',
      stage1: { dense: 1.10, bm25: 0.70, entity: 0.75, relation: 1.05, ppr: 0.75, community: 0.90, maxsim: 0.30 },
      ltr: {
        denseSim: 0.50,
        bm25Norm: 0.10,
        entityHit: 0.07,
        relationHit: 0.14,
        pprScore: 0.10,
        crossEncoderScore: 0.14,
        recencyScore: 0.01,
        authorityScore: 0.02,
        versionActive: 0.04,
        headingMatch: 0.04,
        exactCodeMatch: 0.04,
        parentCoherence: 0.04,
      },
      maxContextTokens: 4000,
      maxEntityTokens: 950,
      maxRelationTokens: 1050,
      rerankLimit: 8,
      minRelevanceScore: 0.08,
      skipParentExpandAbove: 0.98,
    },
  },
  crag: {
    highConfidenceThreshold: 0.55,
    lowConfidenceThreshold: 0.24,
    minTermCoverageForHigh: 0.35,
    enableRetry: true,
  },
  shadowSampleRate: 0.02,
};

const EXACT_CODE_GLOBAL_RE = /\b([A-Z]{2,6}-[A-Z0-9]{2,}(?:-[A-Z0-9]{2,})?)\b/g;
const COMPARISON_RE =
  /\b(?:compare|contrast|versus|vs\.?|difference\s+between|differ\s+from|bandingkan|perbedaan\s+antara)\b/i;
const SYNTHESIS_RE =
  /\b(?:summarize\s+all|summarise\s+all|overall|overview\s+of\s+all|across\s+all|all\s+(?:security|policies|incidents|systems|guarantees|departments|teams)|roadmap\s+priorities|key\s+themes|ringkasan\s+menyeluruh|secara\s+keseluruhan)\b/i;
const MULTIHOP_RE =
  /\b(?:connect|relationship|between|impact\s+of|who\s+manages\b.*\bthat\b|which\b.*\band\s+how|report\s+to\b.*\band\b|guarantees\s+covering|depends\s+on|leads\s+to|hubungan\s+antara)\b/i;
const TEMPORAL_POLICY_RE =
  /\b(?:latest|current|currently|active|effective|superseded|deprecated|updated|version|policy|sop|kebijakan|berlaku|terbaru|saat\s+ini)\b/i;
const MULTI_CLAUSE_RE =
  /\b(?:and|as\s+well\s+as|plus|serta|dan)\b.*\b(?:what|who|how|when|where|which|why|berapa|siapa|bagaimana|apa)\b/i;
const PRONOUN_RE = /\b(?:it|its|they|them|their|that|those|her|his|tersebut|itu)\b/i;

export interface ClassifiedQueryPlan {
  intent: QueryIntent;
  strategy: RetrievalStrategyName;
  effectiveMode: ConcreteRagQueryMode;
  reason: string;
  features: QueryFeatures;
  profile: IntentWeightProfile;
  budget: TokenBudgetConfig;
}

/**
 * Deterministic `<1ms` query intent classifier that selects the single best strategy,
 * retrieval mode, per-intent weight profile, and strict token budget ceiling.
 */
export function classifyQueryIntent(
  requestedMode: RagQueryMode,
  query: string,
  history: Array<{ role: string; content: string }> = [],
  config: RetrievalConfigPayload = DEFAULT_RETRIEVAL_CONFIG,
): ClassifiedQueryPlan {
  const trimmed = query.trim();
  const words = trimmed ? trimmed.split(/\s+/) : [];
  const exactCodes = [...trimmed.matchAll(EXACT_CODE_GLOBAL_RE)].map((m) => m[1]!).filter(Boolean);
  const temporal = parseTemporalIntent(trimmed);

  const features: QueryFeatures = {
    wordCount: words.length,
    hasExactCode: exactCodes.length > 0,
    exactCodes,
    hasTemporalCue: TEMPORAL_POLICY_RE.test(trimmed) || temporal.targetYear !== null || temporal.allowHistorical,
    allowHistorical: temporal.allowHistorical,
    targetYear: temporal.targetYear,
    hasMultiClause: MULTI_CLAUSE_RE.test(trimmed),
    hasComparisonCue: COMPARISON_RE.test(trimmed),
    hasSynthesisCue: SYNTHESIS_RE.test(trimmed),
    hasHistoryPronoun: history.length > 0 && PRONOUN_RE.test(trimmed),
  };

  let intent: QueryIntent;
  let reason: string;

  if (features.hasSynthesisCue) {
    intent = 'global_synthesis';
    reason = 'global_synthesis_cue';
  } else if (features.hasComparisonCue || (features.hasMultiClause && features.wordCount >= 12)) {
    intent = 'comparative_complex';
    reason = features.hasComparisonCue ? 'comparison_cue' : 'multi_clause_compound';
  } else if (MULTIHOP_RE.test(trimmed) || features.hasMultiClause || features.hasHistoryPronoun) {
    intent = 'local_multihop';
    reason = features.hasHistoryPronoun ? 'multiturn_reference' : 'multihop_cue';
  } else if (features.hasExactCode && features.wordCount <= 16) {
    intent = 'exact_lookup';
    reason = 'exact_code_lookup';
  } else if (features.hasTemporalCue && (temporal.allowHistorical || temporal.targetYear !== null || /\b(?:latest|current|active|effective|terbaru|berlaku|superseded)\b/i.test(trimmed))) {
    intent = 'temporal_policy';
    reason = 'temporal_policy_cue';
  } else if (features.wordCount <= 14) {
    intent = 'factual';
    reason = 'direct_factual';
  } else {
    intent = 'local_multihop';
    reason = 'long_complex_query';
  }

  const profile = config.intents[intent];
  const effectiveMode: ConcreteRagQueryMode = requestedMode === 'auto' ? profile.mode : requestedMode;

  return {
    intent,
    strategy: profile.strategy,
    effectiveMode,
    reason: requestedMode === 'auto' ? reason : `explicit:${requestedMode}`,
    features,
    profile,
    budget: {
      maxEntityTokens: profile.maxEntityTokens,
      maxRelationTokens: profile.maxRelationTokens,
      maxTotalTokens: profile.maxContextTokens,
    },
  };
}

const STOPWORDS = new Set([
  'the', 'is', 'at', 'which', 'on', 'in', 'a', 'an', 'and', 'or', 'for', 'to', 'of', 'with',
  'from', 'by', 'as', 'are', 'was', 'were', 'be', 'been', 'what', 'who', 'how', 'when', 'where',
  'why', 'does', 'do', 'did', 'can', 'could', 'should', 'would', 'about', 'into', 'our', 'their',
  'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'pada', 'adalah', 'apa', 'siapa', 'bagaimana', 'berapa',
]);

export function extractSignificantTerms(text: string): string[] {
  const cleaned = text.toLowerCase().replace(/[^\p{L}\p{N}-]+/gu, ' ');
  const terms = cleaned
    .split(/\s+/)
    .filter((w) => w.length >= 3 && !STOPWORDS.has(w));
  return [...new Set(terms)];
}

/**
 * Computes lexical overlap between query terms and a section heading in `[0, 1]`.
 */
export function computeHeadingMatchScore(queryTerms: string[], sectionHeading?: string | null): number {
  if (!sectionHeading || queryTerms.length === 0) return 0;
  const lower = sectionHeading.toLowerCase();
  let hits = 0;
  for (const term of queryTerms) {
    if (lower.includes(term)) hits++;
  }
  return Math.min(1, hits / Math.min(4, queryTerms.length));
}

/**
 * Computes exact corporate code match score in `[0, 1]`.
 */
export function computeExactCodeMatchScore(exactCodes: string[], content: string, title: string): number {
  if (exactCodes.length === 0) return 0;
  const haystack = `${title} ${content}`.toLowerCase();
  let hits = 0;
  for (const code of exactCodes) {
    if (haystack.includes(code.toLowerCase())) hits++;
  }
  return hits / exactCodes.length;
}

/**
 * Computes normalized linear Learning-to-Rank score over the 12-feature vector.
 * Runs in <0.01ms per candidate with zero external ML runtime dependency.
 */
export function scoreCandidateWithLtr(
  features: RetrievalFeatureVector,
  weights: Record<keyof RetrievalFeatureVector, number>,
): number {
  let dot = 0;
  let positiveWeightSum = 0;
  for (const key of LTR_FEATURE_KEYS) {
    const w = weights[key] ?? 0;
    const f = Math.max(0, Math.min(1, features[key] ?? 0));
    dot += w * f;
    if (w > 0) positiveWeightSum += w;
  }
  if (positiveWeightSum <= 0) return 0;
  return Math.max(0, Math.min(1, dot / positiveWeightSum));
}
