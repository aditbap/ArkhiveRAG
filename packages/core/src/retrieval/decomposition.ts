import type { QueryIntent } from './intent-and-ltr.js';

export interface DecomposedQueryResult {
  standaloneQuery: string;
  subQueries: string[];
  highLevelKeywords: string[];
  lowLevelKeywords: string[];
  usedFallback: boolean;
}

/**
 * Token-efficient trigger gate for Query Decomposition (#4):
 * Only activates on genuinely comparative or multi-clause compound queries (~15-20% of traffic).
 */
export function shouldDecomposeQuery(intent: QueryIntent, query: string): boolean {
  if (intent === 'comparative_complex') return true;
  if (intent === 'local_multihop' && query.trim().split(/\s+/).length >= 14) {
    return /\b(?:and|both|compare|versus|vs\.?|as well as|serta|dan)\b/i.test(query);
  }
  return false;
}

/**
 * Single unified LLM prompt combining query rewrite + decomposition (max 3 atomic sub-queries)
 * + dual-level LightRAG keywords in ONE call so decomposition never adds a second LLM round-trip.
 */
export function buildUnifiedDecomposePrompt(params: {
  query: string;
  history?: Array<{ role: string; content: string }>;
}): string {
  const historyBlock =
    params.history && params.history.length > 0
      ? `\nConversation History:\n${params.history
          .slice(-4)
          .map((m) => `${m.role}: ${m.content.slice(0, 300)}`)
          .join('\n')}\n`
      : '';

  return `Decompose this enterprise question into 2 to 3 concise, self-contained atomic sub-queries for parallel document retrieval, and extract search keywords. Preserve exact policy/SOP/incident codes verbatim. Keep the same language as the query.${historyBlock}
Query: ${params.query}

Return ONLY valid JSON:
{"standalone_query":"...","sub_queries":["sub-query 1","sub-query 2"],"high_level_keywords":["theme1"],"low_level_keywords":["entity1","code1"]}`;
}

export function parseDecomposeOutput(raw: string, fallbackQuery: string): DecomposedQueryResult {
  const fallback: DecomposedQueryResult = {
    standaloneQuery: fallbackQuery,
    subQueries: [fallbackQuery],
    highLevelKeywords: [],
    lowLevelKeywords: [],
    usedFallback: true,
  };

  if (!raw || typeof raw !== 'string') return fallback;

  try {
    const start = raw.indexOf('{');
    const end = raw.lastIndexOf('}');
    if (start === -1 || end === -1 || end <= start) return fallback;
    const parsed = JSON.parse(raw.slice(start, end + 1)) as Record<string, unknown>;

    const standaloneQuery =
      typeof parsed.standalone_query === 'string' && parsed.standalone_query.trim()
        ? parsed.standalone_query.trim()
        : fallbackQuery;

    const rawSubs = Array.isArray(parsed.sub_queries) ? parsed.sub_queries : [];
    const subQueries = [...new Set(
      rawSubs
        .filter((s): s is string => typeof s === 'string' && s.trim().length >= 4)
        .map((s) => s.trim()),
    )].slice(0, 3);

    const normalizeList = (v: unknown): string[] =>
      Array.isArray(v)
        ? [...new Set(v.filter((x): x is string => typeof x === 'string' && x.trim().length > 0).map((x) => x.trim()))]
        : [];

    return {
      standaloneQuery,
      subQueries: subQueries.length > 0 ? subQueries : [standaloneQuery],
      highLevelKeywords: normalizeList(parsed.high_level_keywords),
      lowLevelKeywords: normalizeList(parsed.low_level_keywords),
      usedFallback: false,
    };
  } catch {
    return fallback;
  }
}
