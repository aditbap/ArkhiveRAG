import { extractSignificantTerms, type QueryIntent } from './intent-and-ltr.js';
import type { PprPathTrace } from '../lightrag/ppr.js';

export interface ReliabilityChunkInput {
  id: string;
  documentId: string;
  documentTitle: string;
  content: string;
  temporalBadge?: string;
}

export interface EvidenceContradiction {
  docA: { documentId: string; documentTitle: string; value: string; badge?: string };
  docB: { documentId: string; documentTitle: string; value: string; badge?: string };
  kind: 'numeric_conflict' | 'polarity_conflict' | 'version_conflict';
  sharedTopicTerms: string[];
  summary: string;
}

export interface ContradictionAnalysis {
  hasContradictions: boolean;
  contradictions: EvidenceContradiction[];
  promptDirective?: string;
}

const QUANTIFIED_FACT_RE =
  /(?:\$\s*\d[\d,]*(?:\.\d+)?|\b\d+(?:\.\d+)?\s*(?:%|percent|days?|hari|hours?|jam|minutes?|menit|years?|tahun|months?|bulan|ms|gb|mb|usd)\b)/gi;

const ALLOW_RE = /\b(?:allowed|permitted|enabled|approved|mandatory|required|wajib|diperbolehkan)\b/i;
const DENY_RE = /\b(?:prohibited|forbidden|disallowed|not\s+allowed|banned|only\s+for\s+none|dilarang|tidak\s+diperbolehkan)\b/i;

function extractNormalizedQuantities(text: string): string[] {
  const matches = [...text.matchAll(QUANTIFIED_FACT_RE)].map((m) =>
    m[0].toLowerCase().replace(/\s+/g, ' ').trim(),
  );
  return [...new Set(matches)];
}

function quantityUnit(q: string): string {
  if (q.startsWith('$')) return 'currency_usd';
  if (q.endsWith('%') || q.includes('percent')) return 'percent';
  if (/\b(?:day|days|hari)\b/.test(q)) return 'days';
  if (/\b(?:hour|hours|jam)\b/.test(q)) return 'hours';
  if (/\b(?:minute|minutes|menit)\b/.test(q)) return 'minutes';
  if (/\b(?:year|years|tahun)\b/.test(q)) return 'years';
  if (/\b(?:month|months|bulan)\b/.test(q)) return 'months';
  return 'other';
}

/**
 * Zero-LLM-cost contradiction detector across top retrieved evidence chunks (<2ms).
 * Detects:
 * 1. Explicit version conflicts (`SUPERSEDED` vs active edition on the same topic).
 * 2. Numeric/limit conflicts sharing the same unit & high topical overlap (e.g. `$160` vs `$280` hotel cap).
 * 3. Direct polarity conflicts (`allowed/mandatory` vs `prohibited/forbidden`) on overlapping policy terms.
 */
export function detectEvidenceContradictions(chunks: ReliabilityChunkInput[]): ContradictionAnalysis {
  const top = chunks.slice(0, 6);
  if (top.length < 2) {
    return { hasContradictions: false, contradictions: [] };
  }

  const parsed = top.map((c) => ({
    chunk: c,
    terms: new Set(extractSignificantTerms(`${c.documentTitle} ${c.content}`)),
    quantities: extractNormalizedQuantities(c.content),
    hasAllow: ALLOW_RE.test(c.content),
    hasDeny: DENY_RE.test(c.content),
    isSuperseded: Boolean(c.temporalBadge?.includes('SUPERSEDED')),
  }));

  const contradictions: EvidenceContradiction[] = [];
  const seenPairs = new Set<string>();

  for (let i = 0; i < parsed.length; i++) {
    for (let j = i + 1; j < parsed.length; j++) {
      const a = parsed[i]!;
      const b = parsed[j]!;
      if (a.chunk.documentId === b.chunk.documentId) continue;

      const shared: string[] = [];
      for (const t of a.terms) {
        if (b.terms.has(t)) shared.push(t);
      }
      if (shared.length < 3) continue;

      const pairKey = [a.chunk.documentId, b.chunk.documentId].sort().join('::');
      if (seenPairs.has(pairKey)) continue;

      // 1. Explicit superseded vs active policy conflict
      if (a.isSuperseded !== b.isSuperseded) {
        seenPairs.add(pairKey);
        contradictions.push({
          docA: {
            documentId: a.chunk.documentId,
            documentTitle: a.chunk.documentTitle,
            value: a.isSuperseded ? 'SUPERSEDED edition' : 'ACTIVE edition',
            badge: a.chunk.temporalBadge,
          },
          docB: {
            documentId: b.chunk.documentId,
            documentTitle: b.chunk.documentTitle,
            value: b.isSuperseded ? 'SUPERSEDED edition' : 'ACTIVE edition',
            badge: b.chunk.temporalBadge,
          },
          kind: 'version_conflict',
          sharedTopicTerms: shared.slice(0, 6),
          summary: `Active vs. superseded edition detected between "${a.chunk.documentTitle}" and "${b.chunk.documentTitle}".`,
        });
        continue;
      }

      // 2. Numeric conflict on identical unit (e.g. $160 vs $280) with strong shared topic overlap (>= 5 terms)
      if (shared.length >= 5) {
        let foundNumeric = false;
        for (const qa of a.quantities) {
          const ua = quantityUnit(qa);
          if (ua === 'other') continue;
          for (const qb of b.quantities) {
            if (qa !== qb && quantityUnit(qb) === ua && !b.quantities.includes(qa) && !a.quantities.includes(qb)) {
              seenPairs.add(pairKey);
              contradictions.push({
                docA: { documentId: a.chunk.documentId, documentTitle: a.chunk.documentTitle, value: qa, badge: a.chunk.temporalBadge },
                docB: { documentId: b.chunk.documentId, documentTitle: b.chunk.documentTitle, value: qb, badge: b.chunk.temporalBadge },
                kind: 'numeric_conflict',
                sharedTopicTerms: shared.slice(0, 6),
                summary: `Conflicting ${ua} figures (${qa} vs ${qb}) across "${a.chunk.documentTitle}" and "${b.chunk.documentTitle}".`,
              });
              foundNumeric = true;
              break;
            }
          }
          if (foundNumeric) break;
        }
        if (foundNumeric) continue;
      }

      // 3. Polarity conflict (one allows/mandates while the other prohibits on >= 5 shared topic terms)
      if (shared.length >= 5 && ((a.hasAllow && !a.hasDeny && b.hasDeny && !b.hasAllow) || (b.hasAllow && !b.hasDeny && a.hasDeny && !a.hasAllow))) {
        seenPairs.add(pairKey);
        contradictions.push({
          docA: { documentId: a.chunk.documentId, documentTitle: a.chunk.documentTitle, value: a.hasAllow ? 'allowed/required' : 'prohibited', badge: a.chunk.temporalBadge },
          docB: { documentId: b.chunk.documentId, documentTitle: b.chunk.documentTitle, value: b.hasAllow ? 'allowed/required' : 'prohibited', badge: b.chunk.temporalBadge },
          kind: 'polarity_conflict',
          sharedTopicTerms: shared.slice(0, 6),
          summary: `Conflicting policy directive (allowed vs prohibited) across "${a.chunk.documentTitle}" and "${b.chunk.documentTitle}".`,
        });
      }
    }
  }

  if (contradictions.length === 0) {
    return { hasContradictions: false, contradictions: [] };
  }

  const lines = contradictions.slice(0, 3).map((c, idx) => `${idx + 1}. ${c.summary}`).join(' ');
  return {
    hasContradictions: true,
    contradictions,
    promptDirective:
      `CONTRADICTION CHECK: Retrieved documents contain potentially conflicting statements (${lines}) ` +
      'Prefer the latest ACTIVE/effective edition for current rules (unless the user explicitly asked for historical rules), and clearly cite which edition each figure comes from.',
  };
}

export interface ClaimCitationCheck {
  claim: string;
  citedIndices: number[];
  grounded: boolean;
  reason: 'grounded' | 'forged_citation' | 'uncited_factual_claim' | 'low_evidence_overlap';
}

export interface AnswerCitationValidation {
  valid: boolean;
  totalClaims: number;
  groundedClaims: number;
  forgedCitations: number[];
  unsupportedClaims: ClaimCitationCheck[];
  groundingRatio: number;
}

const CITATION_MARKER_RE = /\[(?:Citation\s*)?(\d+)\]/gi;

/**
 * Post-generation per-claim citation validator (<2ms, zero LLM calls).
 * Verifies:
 * 1. Every `[Citation N]` reference is strictly in-bounds (`1 <= N <= chunksUsed.length`) — catches forged citations.
 * 2. Each cited factual sentence shares concrete numeric or lexical overlap with its cited chunk(s).
 */
export function validateAnswerCitations(
  answerText: string,
  chunksUsed: Array<{ id: string; content: string; documentTitle?: string }>,
): AnswerCitationValidation {
  const maxIdx = chunksUsed.length;
  const forgedSet = new Set<number>();

  // Scan entire answer for out-of-bounds citation numbers
  for (const m of answerText.matchAll(CITATION_MARKER_RE)) {
    const n = Number(m[1]);
    if (!Number.isInteger(n) || n < 1 || n > maxIdx) {
      forgedSet.add(n);
    }
  }

  // Split into non-trivial sentences/claims (ignoring short greetings or headings)
  const rawSentences = answerText
    .replace(/\r/g, '\n')
    .split(/(?<=[.!?])\s+|\n+/)
    .map((s) => s.trim())
    .filter((s) => s.length >= 24 && !s.startsWith('#'));

  const chunkHaystacks = chunksUsed.map((c) => `${c.documentTitle ?? ''} ${c.content}`.toLowerCase());
  const checks: ClaimCitationCheck[] = [];

  for (const sentence of rawSentences) {
    const citedIndices = [
      ...new Set([...sentence.matchAll(CITATION_MARKER_RE)].map((m) => Number(m[1])).filter(Number.isFinite)),
    ];

    const hasOutOfBounds = citedIndices.some((n) => n < 1 || n > maxIdx);
    if (hasOutOfBounds) {
      checks.push({ claim: sentence, citedIndices, grounded: false, reason: 'forged_citation' });
      continue;
    }

    // Abstention / meta sentences don't require document citations
    if (/\b(?:do\s+not\s+contain|not\s+mentioned|insufficient\s+information|cannot\s+confirm|tidak\s+tersedia)\b/i.test(sentence)) {
      continue;
    }

    const cleanClaim = sentence.replace(CITATION_MARKER_RE, ' ');
    const claimTerms = extractSignificantTerms(cleanClaim);
    if (claimTerms.length < 2) continue;

    if (citedIndices.length === 0) {
      // Uncited claim containing specific numbers/codes is flagged as unsupported
      const hasHardFact = /\d/.test(cleanClaim) || /\b[A-Z]{2,6}-[A-Z0-9]{2,}\b/.test(cleanClaim);
      if (hasHardFact) {
        checks.push({ claim: sentence, citedIndices: [], grounded: false, reason: 'uncited_factual_claim' });
      }
      continue;
    }

    const combinedCitedText = citedIndices.map((n) => chunkHaystacks[n - 1] ?? '').join(' ');
    const claimNumbers = [...cleanClaim.matchAll(/\b\d+(?:[.,]\d+)?%?\b/g)].map((m) => m[0].toLowerCase());
    const numberHits = claimNumbers.filter((num) => combinedCitedText.includes(num)).length;
    const termHits = claimTerms.filter((t) => combinedCitedText.includes(t)).length;
    const termOverlap = termHits / claimTerms.length;

    const numbersOk = claimNumbers.length === 0 || numberHits / claimNumbers.length >= 0.5;
    const grounded = termOverlap >= 0.25 && numbersOk;

    checks.push({
      claim: sentence,
      citedIndices,
      grounded,
      reason: grounded ? 'grounded' : 'low_evidence_overlap',
    });
  }

  const totalClaims = checks.length;
  const groundedClaims = checks.filter((c) => c.grounded).length;
  const unsupportedClaims = checks.filter((c) => !c.grounded);
  const forgedCitations = [...forgedSet];
  const groundingRatio = totalClaims > 0 ? groundedClaims / totalClaims : forgedCitations.length === 0 ? 1 : 0;

  return {
    valid: forgedCitations.length === 0 && unsupportedClaims.length === 0,
    totalClaims,
    groundedClaims,
    forgedCitations,
    unsupportedClaims,
    groundingRatio: Number(groundingRatio.toFixed(4)),
  };
}

export interface MultiHopCoverageReport {
  required: boolean;
  sufficient: boolean;
  distinctDocuments: number;
  tracedPathsCount: number;
  missingHopHint?: string;
}

/**
 * Verifies that multi-hop or comparative queries retain evidence across distinct documents
 * and traced graph paths (rather than collapsing onto a single hop-1 document).
 */
export function verifyMultiHopEvidenceCoverage(
  intent: QueryIntent,
  pprPaths: PprPathTrace[],
  chunksUsed: Array<{ documentId: string; content: string }>,
): MultiHopCoverageReport {
  const required = intent === 'local_multihop' || intent === 'comparative_complex' || intent === 'global_synthesis';
  const distinctDocuments = new Set(chunksUsed.map((c) => c.documentId)).size;
  if (!required) {
    return { required: false, sufficient: true, distinctDocuments, tracedPathsCount: pprPaths.length };
  }

  const sufficient = distinctDocuments >= 2;
  return {
    required: true,
    sufficient,
    distinctDocuments,
    tracedPathsCount: pprPaths.length,
    missingHopHint: sufficient
      ? undefined
      : 'Only one source document was retrieved for a multi-hop/synthesis question; explicitly state if connecting evidence for another hop is missing.',
  };
}

export interface CalibratedConfidence {
  score: number;
  band: 'HIGH' | 'MEDIUM' | 'LOW' | 'ABSTAIN';
  factors: {
    retrievalConfidence: number;
    citationGrounding: number;
    contradictionPenalty: number;
    multiHopPenalty: number;
  };
}

/**
 * Combines retrieval CRAG confidence, contradiction penalty, multi-hop coverage, and
 * post-generation citation grounding into a single calibrated reliability score in `[0, 1]`.
 */
export function calibrateAnswerConfidence(params: {
  cragConfidence: number;
  cragAction: string;
  contradictionCount: number;
  multiHopSufficient: boolean;
  citationGroundingRatio?: number;
  forgedCitationCount?: number;
}): CalibratedConfidence {
  const retrievalConfidence = Math.max(0, Math.min(1, params.cragConfidence));
  const citationGrounding =
    params.citationGroundingRatio !== undefined
      ? Math.max(0, Math.min(1, params.citationGroundingRatio))
      : 1.0;
  const contradictionPenalty = Math.min(0.25, params.contradictionCount * 0.08);
  const multiHopPenalty = params.multiHopSufficient ? 0 : 0.12;
  const forgedPenalty = (params.forgedCitationCount ?? 0) > 0 ? 0.35 : 0;

  const raw =
    0.55 * retrievalConfidence +
    0.45 * citationGrounding -
    contradictionPenalty -
    multiHopPenalty -
    forgedPenalty;
  const score = Number(Math.max(0, Math.min(1, raw)).toFixed(4));

  let band: CalibratedConfidence['band'];
  if (params.cragAction === 'abstain_hint' || score < 0.32) band = 'ABSTAIN';
  else if (score < 0.58) band = 'LOW';
  else if (score < 0.8) band = 'MEDIUM';
  else band = 'HIGH';

  return {
    score,
    band,
    factors: {
      retrievalConfidence,
      citationGrounding,
      contradictionPenalty,
      multiHopPenalty,
    },
  };
}
