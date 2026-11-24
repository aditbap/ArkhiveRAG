import type { ConcreteRagQueryMode, RagQueryMode } from '../types/lightrag.js';

export interface AdaptiveRoutingDecision {
  mode: ConcreteRagQueryMode;
  reason: string;
}

const EXACT_CODE_RE = /\b[A-Z]{2,5}-[A-Z0-9]{2,}(?:-[A-Z0-9]{2,})?\b/;
const SIMPLE_GREETING_RE = /^(?:hi|hello|hey|halo|selamat\s+(?:pagi|siang|sore|malam)|thanks|thank\s+you|terima\s+kasih|ok|okay)\b[!?. ]*$/i;
const MULTI_HOP_OR_SYNTHESIS_RE =
  /\b(?:summarize\s+all|summarise\s+all|compare|contrast|across|connect|relationship|between|impact\s+of|how\s+do(?:es)?\b.*\band\b|who\s+manages\b.*\bthat\b|which\b.*\band\s+how|report\s+to\b.*\band\b|guarantees\s+covering|roadmap\s+priorities|overall|overview\s+of\s+all|all\s+(?:security|policies|incidents|systems|guarantees)|ringkasan\s+menyeluruh|hubungan\s+antara)\b/i;

/**
 * Deterministic sub-1ms adaptive query mode router (`mode: 'auto'`).
 *
 * Routes simple direct lookups and exact-code queries to `'naive'` (Hybrid Dense +
 * Multilingual BM25 + RRF, ~250ms, 0 extra LLM calls), while routing multi-hop,
 * cross-document synthesis, thematic, and ambiguous follow-up queries to `'mix'`
 * (Knowledge Graph + Hybrid Chunks).
 */
export function resolveAdaptiveMode(
  requestedMode: RagQueryMode,
  query: string,
  history: Array<{ role: string; content: string }> = [],
): AdaptiveRoutingDecision {
  if (requestedMode !== 'auto') {
    return { mode: requestedMode, reason: 'explicit' };
  }

  const trimmed = query.trim();
  if (!trimmed || SIMPLE_GREETING_RE.test(trimmed)) {
    return { mode: 'naive', reason: 'greeting_or_empty' };
  }

  // Complex multi-hop, cross-document, or broad thematic questions benefit most from KG + Hybrid ('mix')
  if (MULTI_HOP_OR_SYNTHESIS_RE.test(trimmed)) {
    return { mode: 'mix', reason: 'synthesis_or_multihop' };
  }

  // Multi-turn follow-ups with pronouns/cross-references benefit from unified analysis + KG links
  if (history.length > 0 && /\b(?:it|its|they|them|their|that|those|her|his|tersebut|itu)\b/i.test(trimmed)) {
    return { mode: 'mix', reason: 'multiturn_reference' };
  }

  // Compound questions asking two distinct things ("X ... and Y ...?") often span multiple docs
  const hasCompoundAnd = /\band\b.*\b(?:what|who|how|when|where|which|why|berapa|siapa|bagaimana)\b/i.test(trimmed);
  if (hasCompoundAnd) {
    return { mode: 'mix', reason: 'compound_multihop' };
  }

  // Exact policy/SOP/incident codes without multi-hop clauses are answered fastest & most accurately via Hybrid Dense+BM25
  if (EXACT_CODE_RE.test(trimmed) && trimmed.split(/\s+/).length <= 14) {
    return { mode: 'naive', reason: 'exact_code_lookup' };
  }

  // Short direct factual questions (<= 14 words, no synthesis markers) -> fast Hybrid Dense+BM25
  if (trimmed.split(/\s+/).length <= 14) {
    return { mode: 'naive', reason: 'direct_factual_lookup' };
  }

  return { mode: 'mix', reason: 'complex_query' };
}
