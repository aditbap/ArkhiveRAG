import { estimateTokens } from '../lightrag/index.js';
import { extractSignificantTerms } from './intent-and-ltr.js';

export interface ConversationTurn {
  role: string;
  content: string;
}

const EXACT_CODE_RE = /\b([A-Z]{2,6}-[A-Z0-9]{2,}(?:-[A-Z0-9]{2,})?)\b/g;
const PRONOUN_FOLLOWUP_RE =
  /\b(?:it|its|they|them|their|that|those|this|these|her|his|above|previous|former|latter|tersebut|itu|ini|dia|nya)\b/i;

const HISTORY_INJECTION_PATTERNS: RegExp[] = [
  /ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|prompts|rules)/gi,
  /disregard\s+(?:all\s+)?(?:previous|prior|system)\s+(?:instructions|prompts)/gi,
  /(?:reveal|print|output|show)\s+(?:your\s+)?(?:system\s+prompt|hidden\s+instructions)/gi,
  /you\s+are\s+now\s+(?:dan|in\s+developer\s+mode|unrestricted)/gi,
  /^\s*(?:system|developer)\s*:\s*/gim,
];

/**
 * Strips/neutralizes prompt-injection override phrases embedded in prior conversation turns
 * before history is fed to query analysis or generation. Never mutates benign corporate text.
 */
export function sanitizeConversationHistory(history: ConversationTurn[]): {
  sanitized: ConversationTurn[];
  injectionsNeutralized: number;
} {
  let injectionsNeutralized = 0;
  const sanitized = history.map((turn) => {
    let cleaned = turn.content;
    for (const pattern of HISTORY_INJECTION_PATTERNS) {
      const before = cleaned;
      cleaned = cleaned.replace(pattern, '[filtered-directive]');
      if (cleaned !== before) injectionsNeutralized++;
    }
    return {
      role: turn.role === 'assistant' ? 'assistant' : 'user',
      content: cleaned.trim(),
    };
  });
  return { sanitized, injectionsNeutralized };
}

/**
 * Topic-shift detector: determines whether `currentQuery` actually continues the previous
 * conversation turn (via pronoun/deictic cue OR shared significant terms/codes) vs. switching
 * to an unrelated new topic. Prevents stale history from contaminating fresh retrieval queries.
 */
export function isTopicContinuation(currentQuery: string, history: ConversationTurn[]): boolean {
  if (history.length === 0) return false;
  if (PRONOUN_FOLLOWUP_RE.test(currentQuery)) return true;

  const qTerms = new Set(extractSignificantTerms(currentQuery));
  if (qTerms.size === 0) return true;

  const recentText = history
    .slice(-2)
    .map((m) => m.content)
    .join(' ');
  const hTerms = new Set(extractSignificantTerms(recentText));
  if (hTerms.size === 0) return false;

  let overlap = 0;
  for (const t of qTerms) {
    if (hTerms.has(t)) overlap++;
  }
  return overlap / Math.min(qTerms.size, 6) >= 0.16;
}

/**
 * Token-budgeted conversation memory compression:
 * - Preserves the most recent `keepRecentTurns` turns verbatim (default 4 messages = 2 pairs).
 * - Deterministically compresses older turns into a single compact memory summary preserving
 *   exact corporate codes and top topic terms so long chats stay within `maxHistoryTokens`.
 */
export function compressConversationHistory(
  history: ConversationTurn[],
  options: { keepRecentMessages?: number; maxHistoryTokens?: number } = {},
): ConversationTurn[] {
  const keepRecent = Math.max(2, options.keepRecentMessages ?? 4);
  const maxTokens = Math.max(200, options.maxHistoryTokens ?? 1200);
  if (history.length === 0) return [];

  const totalTokens = history.reduce((sum, m) => sum + estimateTokens(m.content), 0);
  if (history.length <= keepRecent && totalTokens <= maxTokens) {
    return history;
  }

  const older = history.slice(0, Math.max(0, history.length - keepRecent));
  const recent = history.slice(-keepRecent);

  const output: ConversationTurn[] = [];
  if (older.length > 0) {
    const olderText = older.map((m) => m.content).join(' ');
    const codes = [...new Set([...olderText.matchAll(EXACT_CODE_RE)].map((m) => m[1]!).filter(Boolean))].slice(0, 8);
    const terms = extractSignificantTerms(olderText)
      .filter((t) => !codes.some((c) => c.toLowerCase() === t))
      .slice(0, 12);
    const summaryParts: string[] = [];
    if (codes.length > 0) summaryParts.push(`Referenced codes: ${codes.join(', ')}`);
    if (terms.length > 0) summaryParts.push(`Earlier topics: ${terms.join(', ')}`);
    if (summaryParts.length > 0) {
      output.push({
        role: 'assistant',
        content: `[Earlier conversation context summary] ${summaryParts.join(' | ')}`,
      });
    }
  }

  // Enforce hard token ceiling across remaining recent messages (newest prioritized)
  let usedTokens = output.reduce((s, m) => s + estimateTokens(m.content), 0);
  const keptRecent: ConversationTurn[] = [];
  for (let i = recent.length - 1; i >= 0; i--) {
    const msg = recent[i]!;
    const t = estimateTokens(msg.content);
    if (usedTokens + t <= maxTokens || keptRecent.length === 0) {
      const clipped =
        usedTokens + t > maxTokens
          ? msg.content.slice(0, Math.max(120, (maxTokens - usedTokens) * 3))
          : msg.content;
      keptRecent.unshift({ role: msg.role, content: clipped });
      usedTokens += Math.min(t, maxTokens - usedTokens);
    }
  }

  return [...output, ...keptRecent];
}

export interface AmbiguityAssessment {
  isAmbiguous: boolean;
  reason?: 'underspecified_short_query' | 'competing_policy_editions' | 'low_term_specificity';
  clarificationHint?: string;
}

/**
 * Detects genuinely ambiguous enterprise queries without blocking clear questions:
 * 1. Ultra-short queries (<= 2 significant terms) with zero codes and no conversation anchor.
 * 2. Competing distinct high-scoring documents with nearly tied scores and different policy titles
 *    when the query doesn't disambiguate which policy/scope is intended.
 */
export function detectQueryAmbiguity(params: {
  query: string;
  hasHistoryAnchor: boolean;
  exactCodes: string[];
  topCandidates?: Array<{ documentId: string; documentTitle: string; finalScore: number }>;
}): AmbiguityAssessment {
  const words = params.query.trim().split(/\s+/).filter(Boolean);
  const sigTerms = extractSignificantTerms(params.query);

  if (words.length <= 3 && sigTerms.length <= 1 && params.exactCodes.length === 0 && !params.hasHistoryAnchor) {
    return {
      isAmbiguous: true,
      reason: 'underspecified_short_query',
      clarificationHint:
        'The query is very brief and may match multiple topics. Answer the most likely interpretation from the retrieved evidence and offer 1–2 concise clarifying follow-up options.',
    };
  }

  const top = params.topCandidates ?? [];
  if (
    top.length >= 2 &&
    params.exactCodes.length === 0 &&
    words.length <= 6 &&
    top[0]!.documentId !== top[1]!.documentId &&
    top[0]!.finalScore >= 0.6 &&
    top[0]!.finalScore - top[1]!.finalScore <= 0.025
  ) {
    const t1 = top[0]!.documentTitle;
    const t2 = top[1]!.documentTitle;
    if (t1.toLowerCase() !== t2.toLowerCase()) {
      return {
        isAmbiguous: true,
        reason: 'competing_policy_editions',
        clarificationHint: `Multiple equally relevant documents matched ("${t1}" and "${t2}"). Present key grounded points from each and ask the user which scope or policy they want to focus on.`,
      };
    }
  }

  return { isAmbiguous: false };
}
