import { CONSTANTS } from '../constants.js';
import type {
  AssembledContext,
  ContextChunkInput,
  RetrievedEntity,
  RetrievedRelation,
  TokenBudgetConfig,
} from '../types/lightrag.js';

/**
 * Fast, dependency-free token estimator calibrated for both Latin/Indonesian/English
 * (~4 chars/token) and CJK scripts (~1.5 chars/token). Avoids loading heavy WASM
 * tokenizers on hot API request paths while staying within ~10% of cl100k_base.
 */
export function estimateTokens(text: string): number {
  if (!text) return 0;
  let cjkChars = 0;
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    if (
      (code >= 0x4e00 && code <= 0x9fff) || // CJK Unified Ideographs
      (code >= 0x3040 && code <= 0x30ff) || // Hiragana & Katakana
      (code >= 0xac00 && code <= 0xd7af)    // Hangul
    ) {
      cjkChars++;
    }
  }
  const nonCjkChars = text.length - cjkChars;
  return Math.ceil(nonCjkChars / 4 + cjkChars / 1.5);
}

/**
 * Cursor-based round-robin interleaving merge across N ranked lists with deduplication.
 * Mirrors LightRAG's `_perform_kg_search` / `_merge_all_chunks` strategy, and improves on
 * index-locked interleaving by advancing past duplicates so overlapping branches never
 * lose their turn before token truncation.
 */
export function roundRobinMerge<T>(lists: T[][], getKey: (item: T) => string): T[] {
  const merged: T[] = [];
  const seen = new Set<string>();
  const cursors = lists.map(() => 0);

  let progressed = true;
  while (progressed) {
    progressed = false;
    for (let li = 0; li < lists.length; li++) {
      const list = lists[li];
      if (!list) continue;
      let idx = cursors[li] ?? 0;
      while (idx < list.length) {
        const item = list[idx++];
        if (!item) continue;
        const key = getKey(item);
        if (!key || seen.has(key)) continue;
        seen.add(key);
        merged.push(item);
        progressed = true;
        break;
      }
      cursors[li] = idx;
    }
  }
  return merged;
}

/**
 * Weighted Reciprocal Rank Fusion (RRF) across arbitrary candidate lists.
 * Combines heterogeneous scoring scales (pgvector cosine, Postgres ts_rank_cd, KG scores).
 */
export function weightedReciprocalRankFusion<T>(
  lists: Array<{ items: T[]; weight?: number }>,
  getKey: (item: T) => string,
  k: number = 60,
): Array<{ item: T; score: number }> {
  const scores = new Map<string, number>();
  const byKey = new Map<string, T>();

  for (const { items, weight = 1 } of lists) {
    items.forEach((item, rank) => {
      const key = getKey(item);
      if (!byKey.has(key)) byKey.set(key, item);
      scores.set(key, (scores.get(key) ?? 0) + weight / (k + rank + 1));
    });
  }

  return [...scores.entries()]
    .sort(([, a], [, b]) => b - a)
    .flatMap(([key, score]) => {
      const item = byKey.get(key);
      return item ? [{ item, score }] : [];
    });
}

/**
 * Truncates an ordered list of items to fit within `maxTokens`, always preserving
 * highest-ranked items first.
 */
export function truncateByTokenBudget<T>(
  items: T[],
  formatItem: (item: T, index: number) => string,
  maxTokens: number,
): { kept: T[]; formatted: string[]; tokensUsed: number } {
  const kept: T[] = [];
  const formatted: string[] = [];
  let tokensUsed = 0;

  if (maxTokens <= 0) return { kept, formatted, tokensUsed };

  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    if (!item) continue;
    const text = formatItem(item, kept.length);
    const cost = estimateTokens(text);
    if (kept.length > 0 && tokensUsed + cost > maxTokens) {
      break;
    }
    // Even if the very first item is slightly larger than maxTokens, include it truncated
    if (kept.length === 0 && cost > maxTokens) {
      const maxChars = Math.max(200, maxTokens * 3);
      kept.push(item);
      formatted.push(text.slice(0, maxChars) + '...');
      tokensUsed += estimateTokens(formatted[0] ?? '');
      break;
    }
    kept.push(item);
    formatted.push(text);
    tokensUsed += cost;
  }

  return { kept, formatted, tokensUsed };
}

function compactText(text: string, maxChars: number): string {
  const cleaned = text.replaceAll(CONSTANTS.KG_FIELD_SEP, '; ').replace(/\s+/g, ' ').trim();
  return cleaned.length > maxChars ? `${cleaned.slice(0, maxChars - 1).trimEnd()}…` : cleaned;
}

/**
 * Grounded Bridge-First KG Filtering:
 * Eliminates noisy distractor KG items from unselected low-rank documents while keeping:
 *   1. Cross-document bridge items touching at least one retrieved chunk (highest multi-hop value)
 *   2. High-relevance uncovered KG items (for KG-only modes or when a chunk didn't fit budget)
 *   3. At most `maxRedundantSignposts` concise items touching retrieved chunks when Tiers 1-2 are sparse.
 */
function prioritizeNonRedundantKg<
  T extends { sourceChunkIds: string[]; documentIds: string[]; similarity?: number },
>(
  items: T[],
  coveredChunkIds: Set<string>,
  maxRedundantSignposts = 1,
): T[] {
  if (items.length === 0 || coveredChunkIds.size === 0) return items;

  const crossDocBridges: T[] = [];
  const highSignalUncovered: T[] = [];
  const coveredSignposts: T[] = [];

  for (const item of items) {
    const touchesCovered = item.sourceChunkIds.some((cid) => coveredChunkIds.has(cid));
    const isCrossDocBridge = item.documentIds.length > 1;

    if (touchesCovered && isCrossDocBridge) {
      crossDocBridges.push(item);
    } else if (touchesCovered) {
      coveredSignposts.push(item);
    } else if (item.sourceChunkIds.length === 0 || (item.similarity ?? 0) >= 0.55) {
      // Keep only genuinely high-relevance uncovered KG facts (or synthetic provenance-less items),
      // rejecting low-similarity distractors from documents LTR already filtered out.
      highSignalUncovered.push(item);
    }
  }

  const keptUncovered = highSignalUncovered.slice(0, crossDocBridges.length > 0 ? 1 : 2);
  const needSignpost = crossDocBridges.length === 0 && keptUncovered.length === 0;
  const keptSignposts = coveredSignposts.slice(0, needSignpost ? Math.max(1, maxRedundantSignposts) : 0);

  return [...crossDocBridges, ...keptUncovered, ...keptSignposts];
}

/**
 * Assembles LightRAG's structured dual-level context (compact Knowledge Graph Entities +
 * Relationships + grounded Document Chunks with `[Citation N]` markers):
 * - Prioritizes grounded `[Citation N]` document chunks first.
 * - Filters KG context to grounded cross-document bridges & high-signal novel facts,
 *   eliminating unselected-doc KG distractors (~60% leaner than verbose JSON-per-item).
 */
export function assembleLightRagContext(params: {
  entities: RetrievedEntity[];
  relations: RetrievedRelation[];
  chunks: ContextChunkInput[];
  budget?: Partial<TokenBudgetConfig>;
}): AssembledContext {
  const maxEntityTokens = params.budget?.maxEntityTokens ?? CONSTANTS.DEFAULT_MAX_ENTITY_TOKENS;
  const maxRelationTokens = params.budget?.maxRelationTokens ?? CONSTANTS.DEFAULT_MAX_RELATION_TOKENS;
  const maxTotalTokens = params.budget?.maxTotalTokens ?? CONSTANTS.DEFAULT_MAX_TOTAL_TOKENS;

  const entityCap = Math.min(maxEntityTokens, Math.floor(maxTotalTokens * 0.25));
  const relationCap = Math.min(maxRelationTokens, Math.floor(maxTotalTokens * 0.25));
  const overheadTokens = 64;

  // 1. Reserve primary budget for grounded `[Citation N]` document chunks first
  const chunkBudget = Math.max(512, maxTotalTokens - Math.min(entityCap + relationCap, Math.floor(maxTotalTokens * 0.35)) - overheadTokens);
  const chunkSlice = truncateByTokenBudget(
    params.chunks,
    (c, idx) => {
      const metaParts = [c.documentTitle];
      if (c.sectionHeading && c.sectionHeading.toLowerCase() !== c.documentTitle.toLowerCase()) {
        metaParts.push(`Section: ${c.sectionHeading}`);
      }
      if (c.pageNumber != null) metaParts.push(`Page: ${c.pageNumber}`);
      if (c.temporalBadge) metaParts.push(c.temporalBadge);
      return `[Citation ${idx + 1}] ${metaParts.join(' | ')}\n${c.content}`;
    },
    chunkBudget,
  );

  // 2. Grounded bridge-first KG filtering & compact single-line serialization
  const coveredChunkIds = new Set(chunkSlice.kept.map((c) => c.id));
  const prunedEntities = prioritizeNonRedundantKg(params.entities, coveredChunkIds, 1);
  const prunedRelations = prioritizeNonRedundantKg(params.relations, coveredChunkIds, 1);

  const remainingForKg = Math.max(0, maxTotalTokens - chunkSlice.tokensUsed - overheadTokens);
  const effEntityCap = Math.min(entityCap, Math.floor(remainingForKg * 0.5));

  const entitySlice = truncateByTokenBudget(
    prunedEntities,
    (e) => `- ${e.entityName} (${e.entityType}): ${compactText(e.description, 100)}`,
    effEntityCap,
  );

  const effRelationCap = Math.min(relationCap, Math.max(0, remainingForKg - entitySlice.tokensUsed));
  const relationSlice = truncateByTokenBudget(
    prunedRelations,
    (r) => `- ${r.sourceEntity} -> ${r.targetEntity} [${compactText(r.keywords, 32)}]: ${compactText(r.description, 95)}`,
    effRelationCap,
  );

  const sections: string[] = [];

  if (entitySlice.formatted.length > 0) {
    sections.push(`### Knowledge Graph Data (Entities)\n${entitySlice.formatted.join('\n')}`);
  }

  if (relationSlice.formatted.length > 0) {
    sections.push(`### Knowledge Graph Data (Relationships)\n${relationSlice.formatted.join('\n')}`);
  }

  if (chunkSlice.formatted.length > 0) {
    sections.push(`### Document Chunks\n\n${chunkSlice.formatted.join('\n\n')}`);
  }

  const contextText = sections.join('\n\n');
  const totalTokens = entitySlice.tokensUsed + relationSlice.tokensUsed + chunkSlice.tokensUsed;

  return {
    contextText,
    entitiesUsed: entitySlice.kept,
    relationsUsed: relationSlice.kept,
    chunksUsed: chunkSlice.kept,
    estimatedTokens: {
      entities: entitySlice.tokensUsed,
      relations: relationSlice.tokensUsed,
      chunks: chunkSlice.tokensUsed,
      total: totalTokens,
    },
  };
}
