import { extractSignificantTerms } from './intent-and-ltr.js';

export type CragVerdict = 'HIGH' | 'AMBIGUOUS' | 'LOW';
export type CragAction = 'proceed' | 'prune_tail' | 'retry_escalate' | 'abstain_hint';

export interface CragCandidate {
  id: string;
  content: string;
  score: number;
}

export interface CragEvaluation {
  verdict: CragVerdict;
  action: CragAction;
  confidence: number;
  topScore: number;
  scoreMargin: number;
  termCoverage: number;
  reason: string;
}

export interface CragThresholds {
  highConfidenceThreshold?: number;
  lowConfidenceThreshold?: number;
  minTermCoverageForHigh?: number;
  allowRetry?: boolean;
}

/**
 * Prompt directive appended ONLY when CRAG still rates evidence as `LOW` after the
 * single bounded corrective retry—prevents hallucinations on unanswerable queries.
 */
export const CRAG_LOW_CONFIDENCE_DIRECTIVE =
  'IMPORTANT EVIDENCE CHECK: The retrieved context has LOW confidence or incomplete coverage for this question. ' +
  'Answer ONLY with facts explicitly supported by the citations above. If the authorized documents do not contain ' +
  'sufficient information to answer part or all of the question, state clearly what is missing instead of guessing.';

/**
 * Zero-LLM-Token Bounded CRAG Evaluator (#6):
 * Assesses retrieval evidence reliability in `<2ms` using:
 * 1. Top candidate calibrated LTR/reranker score
 * 2. Score separation (margin between top-1 and tail median)
 * 3. Query significant-term coverage across the top-3 chunks
 *
 * Guarantees at most 1 corrective retry (`retry_escalate`), never looping unboundedly
 * and never burning 2,000+ tokens on per-chunk LLM grading.
 */
export function evaluateCragConfidence(
  query: string,
  candidates: CragCandidate[],
  thresholds: CragThresholds = {},
  alreadyRetried = false,
): CragEvaluation {
  const highThresh = thresholds.highConfidenceThreshold ?? 0.55;
  const lowThresh = thresholds.lowConfidenceThreshold ?? 0.24;
  const minCov = thresholds.minTermCoverageForHigh ?? 0.35;
  const allowRetry = thresholds.allowRetry ?? true;

  if (candidates.length === 0) {
    return {
      verdict: 'LOW',
      action: !alreadyRetried && allowRetry ? 'retry_escalate' : 'abstain_hint',
      confidence: 0,
      topScore: 0,
      scoreMargin: 0,
      termCoverage: 0,
      reason: 'no_candidates_retrieved',
    };
  }

  const sorted = [...candidates].sort((a, b) => b.score - a.score);
  const topScore = Math.max(0, Math.min(1, sorted[0]?.score ?? 0));
  const tailIdx = Math.min(sorted.length - 1, 4);
  const tailScore = Math.max(0, Math.min(1, sorted[tailIdx]?.score ?? 0));
  const scoreMargin = Math.max(0, topScore - tailScore);

  const queryTerms = extractSignificantTerms(query);
  const topText = sorted
    .slice(0, 3)
    .map((c) => c.content.toLowerCase())
    .join(' ');

  let matchedTerms = 0;
  for (const term of queryTerms) {
    if (topText.includes(term)) matchedTerms++;
  }
  const termCoverage = queryTerms.length > 0 ? matchedTerms / queryTerms.length : 1;

  // Composite reliability score in [0, 1]
  const confidence = Math.max(0, Math.min(1, 0.65 * topScore + 0.25 * termCoverage + 0.10 * Math.min(1, scoreMargin * 2)));

  if (confidence >= highThresh && termCoverage >= minCov) {
    return {
      verdict: 'HIGH',
      action: 'proceed',
      confidence,
      topScore,
      scoreMargin,
      termCoverage,
      reason: 'strong_score_and_term_coverage',
    };
  }

  if (confidence < lowThresh || (termCoverage < 0.15 && topScore < 0.45)) {
    return {
      verdict: 'LOW',
      action: !alreadyRetried && allowRetry ? 'retry_escalate' : 'abstain_hint',
      confidence,
      topScore,
      scoreMargin,
      termCoverage,
      reason: alreadyRetried ? 'low_confidence_after_retry' : 'low_score_or_term_coverage',
    };
  }

  return {
    verdict: 'AMBIGUOUS',
    action: 'prune_tail',
    confidence,
    topScore,
    scoreMargin,
    termCoverage,
    reason: 'moderate_confidence_prune_noisy_tail',
  };
}
