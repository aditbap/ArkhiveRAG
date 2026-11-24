import { CONSTANTS } from '../constants.js';
import type {
  ChunkExtractionResult,
  ExtractedEntity,
  ExtractedRelation,
  QueryAnalysisResult,
} from '../types/lightrag.js';

/**
 * Normalizes an entity name for consistent deduplication and graph matching.
 * Strips surrounding quotes/brackets, collapses whitespace, and caps length.
 */
export function normalizeEntityName(raw: string): string {
  const cleaned = raw
    .replace(/[\r\n\t]+/g, ' ')
    .replace(/^["'`«»„“”<>()[\]]+|["'`«»„“”<>()[\]]+$/g, '')
    .replace(/\s+/g, ' ')
    .trim();
  if (!cleaned) return '';
  return cleaned.slice(0, CONSTANTS.KG_ENTITY_NAME_MAX_LENGTH);
}

/** Case-folded canonical identifier used as the unique key in PostgreSQL. */
export function canonicalizeEntityName(name: string): string {
  return normalizeEntityName(name).toLowerCase();
}

/** Canonical undirected edge key so (A, B) and (B, A) always merge into one edge. */
export function canonicalEdgePair(a: string, b: string): [string, string] {
  const ca = canonicalizeEntityName(a);
  const cb = canonicalizeEntityName(b);
  return ca <= cb ? [ca, cb] : [cb, ca];
}

/**
 * Deduplicates and merges description fragments separated by `<SEP>`, sorting
 * deterministically so concurrent merge order never causes false-positive re-embeddings.
 */
export function mergeDescriptionFragments(existing: string | null | undefined, incoming: string[]): string {
  const byKey = new Map<string, string>();

  const add = (raw: string) => {
    for (const piece of raw.split(CONSTANTS.KG_FIELD_SEP)) {
      const trimmed = piece.trim();
      if (!trimmed) continue;
      const key = trimmed.toLowerCase();
      if (!byKey.has(key)) {
        byKey.set(key, trimmed);
      }
    }
  };

  if (existing) add(existing);
  for (const desc of incoming) add(desc);

  return [...byKey.entries()]
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .map(([, val]) => val)
    .join(CONSTANTS.KG_FIELD_SEP);
}

/** Counts distinct description fragments separated by `<SEP>`. */
export function countDescriptionFragments(description: string): number {
  if (!description.trim()) return 0;
  return description
    .split(CONSTANTS.KG_FIELD_SEP)
    .map((p) => p.trim())
    .filter(Boolean).length;
}

/** Deduplicates comma-separated relation keywords while preserving order. */
export function mergeKeywords(existing: string | null | undefined, incoming: string[]): string {
  const seen = new Set<string>();
  const ordered: string[] = [];

  const add = (raw: string) => {
    for (const kw of raw.split(/[,，;；]/)) {
      const cleaned = kw.trim();
      if (!cleaned) continue;
      const key = cleaned.toLowerCase();
      if (!seen.has(key)) {
        seen.add(key);
        ordered.push(cleaned);
      }
    }
  };

  if (existing) add(existing);
  for (const item of incoming) add(item);

  return ordered.join(', ');
}

/**
 * Extracts the outermost JSON object from an LLM response, tolerating markdown
 * code fences or stray leading/trailing prose.
 */
export function extractJsonObject(raw: unknown): Record<string, unknown> | null {
  if (raw && typeof raw === 'object' && !Array.isArray(raw)) {
    return raw as Record<string, unknown>;
  }
  if (typeof raw !== 'string') return null;
  const trimmed = raw.trim();
  if (!trimmed) return null;

  const candidates: string[] = [trimmed];

  // Strip ```json ... ``` fences if present
  const fenceMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenceMatch?.[1]) {
    candidates.push(fenceMatch[1].trim());
  }

  // Fallback: slice from first `{` to last `}`
  const firstBrace = trimmed.indexOf('{');
  const lastBrace = trimmed.lastIndexOf('}');
  if (firstBrace !== -1 && lastBrace > firstBrace) {
    candidates.push(trimmed.slice(firstBrace, lastBrace + 1));
  }

  for (const candidate of candidates) {
    try {
      const parsed: unknown = JSON.parse(candidate);
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      // Try next candidate
    }
  }
  return null;
}

function normalizeEntityType(raw: unknown): string {
  if (typeof raw !== 'string') return 'Other';
  const cleaned = raw.replace(/[^a-zA-Z0-9 _-]/g, '').trim().slice(0, 64);
  return cleaned || 'Other';
}

function isPlaceholder(val: string): boolean {
  return /^<[^>]+>$/.test(val.trim());
}

/**
 * Parses LLM output from entity/relationship extraction.
 * Supports LightRAG's primary JSON format AND falls back to LightRAG's `<|#|>`
 * tuple format if the model ignores the JSON contract.
 */
export function parseExtractionOutput(raw: unknown): ChunkExtractionResult {
  const rawStr = typeof raw === 'string' ? raw : raw != null ? JSON.stringify(raw) : '';
  const entitiesByCanonical = new Map<string, ExtractedEntity>();
  const relationsByKey = new Map<string, ExtractedRelation>();

  const addEntity = (nameRaw: unknown, typeRaw: unknown, descRaw: unknown) => {
    if (typeof nameRaw !== 'string') return;
    const name = normalizeEntityName(nameRaw);
    if (!name || isPlaceholder(name)) return;
    const canonicalName = canonicalizeEntityName(name);
    if (!canonicalName) return;

    const description = typeof descRaw === 'string' ? descRaw.trim() : '';
    if (isPlaceholder(description)) return;
    const type = normalizeEntityType(typeRaw);

    const existing = entitiesByCanonical.get(canonicalName);
    if (existing) {
      existing.description = mergeDescriptionFragments(existing.description, description ? [description] : []);
      if (existing.type === 'Other' && type !== 'Other') existing.type = type;
    } else {
      entitiesByCanonical.set(canonicalName, {
        name,
        canonicalName,
        type,
        description: description || name,
      });
    }
  };

  const addRelation = (
    srcRaw: unknown,
    tgtRaw: unknown,
    keywordsRaw: unknown,
    descRaw: unknown,
    weightRaw?: unknown,
  ) => {
    if (typeof srcRaw !== 'string' || typeof tgtRaw !== 'string') return;
    const rawSrc = normalizeEntityName(srcRaw);
    const rawTgt = normalizeEntityName(tgtRaw);
    if (!rawSrc || !rawTgt || isPlaceholder(rawSrc) || isPlaceholder(rawTgt)) return;

    const srcCanon = canonicalizeEntityName(rawSrc);
    const tgtCanon = canonicalizeEntityName(rawTgt);
    if (!srcCanon || !tgtCanon || srcCanon === tgtCanon) return; // drop self-loops

    // Canonical undirected ordering
    const swapped = srcCanon > tgtCanon;
    const sourceCanonical = swapped ? tgtCanon : srcCanon;
    const targetCanonical = swapped ? srcCanon : tgtCanon;
    const source = entitiesByCanonical.get(sourceCanonical)?.name ?? (swapped ? rawTgt : rawSrc);
    const target = entitiesByCanonical.get(targetCanonical)?.name ?? (swapped ? rawSrc : rawTgt);

    const keywordStr = Array.isArray(keywordsRaw)
      ? keywordsRaw.filter((k): k is string => typeof k === 'string').join(', ')
      : typeof keywordsRaw === 'string'
        ? keywordsRaw
        : '';
    const keywords = mergeKeywords(null, [keywordStr])
      .split(',')
      .map((k) => k.trim())
      .filter((k) => k && !isPlaceholder(k));

    const description = typeof descRaw === 'string' ? descRaw.trim() : '';
    const weight = typeof weightRaw === 'number' && Number.isFinite(weightRaw) && weightRaw > 0 ? weightRaw : 1.0;

    const key = `${sourceCanonical}::${targetCanonical}`;
    const existing = relationsByKey.get(key);
    if (existing) {
      existing.weight += weight;
      existing.description = mergeDescriptionFragments(existing.description, description ? [description] : []);
      existing.keywords = mergeKeywords(existing.keywords.join(', '), keywords)
        .split(',')
        .map((k) => k.trim())
        .filter(Boolean);
    } else {
      relationsByKey.set(key, {
        source,
        target,
        sourceCanonical,
        targetCanonical,
        keywords,
        description: description || `${source} is related to ${target}`,
        weight,
      });
    }
  };

  // 1. Primary path: JSON structured output
  const jsonObj = extractJsonObject(raw);
  if (jsonObj && (Array.isArray(jsonObj.entities) || Array.isArray(jsonObj.relationships) || Array.isArray(jsonObj.relations))) {
    const rawEntities = Array.isArray(jsonObj.entities) ? jsonObj.entities : [];
    for (const item of rawEntities) {
      if (!item || typeof item !== 'object') continue;
      const e = item as Record<string, unknown>;
      addEntity(e.name ?? e.entity_name, e.type ?? e.entity_type, e.description ?? e.entity_description);
    }

    const rawRelations = Array.isArray(jsonObj.relationships)
      ? jsonObj.relationships
      : Array.isArray(jsonObj.relations)
        ? jsonObj.relations
        : [];
    for (const item of rawRelations) {
      if (!item || typeof item !== 'object') continue;
      const r = item as Record<string, unknown>;
      addRelation(
        r.source ?? r.source_entity ?? r.src_id,
        r.target ?? r.target_entity ?? r.tgt_id,
        r.keywords ?? r.relationship_keywords,
        r.description ?? r.relationship_description,
        r.weight,
      );
    }
  } else {
    // 2. Fallback path: LightRAG `<|#|>` tuple format
    const sep = CONSTANTS.KG_TUPLE_DELIMITER;
    for (const line of rawStr.split('\n')) {
      const cleaned = line.trim();
      if (!cleaned || cleaned === CONSTANTS.KG_COMPLETION_DELIMITER) continue;
      const parts = cleaned.split(sep).map((p) => p.trim());
      if (parts.length >= 4 && parts[0]?.toLowerCase() === 'entity') {
        addEntity(parts[1], parts[2], parts[3]);
      } else if (parts.length >= 5 && (parts[0]?.toLowerCase() === 'relation' || parts[0]?.toLowerCase() === 'relationship')) {
        addRelation(parts[1], parts[2], parts[3], parts[4]);
      }
    }
  }

  // Ensure every relation endpoint exists in entities (matches LightRAG invariant)
  for (const rel of relationsByKey.values()) {
    if (!entitiesByCanonical.has(rel.sourceCanonical)) {
      entitiesByCanonical.set(rel.sourceCanonical, {
        name: rel.source,
        canonicalName: rel.sourceCanonical,
        type: 'Other',
        description: rel.description,
      });
    }
    if (!entitiesByCanonical.has(rel.targetCanonical)) {
      entitiesByCanonical.set(rel.targetCanonical, {
        name: rel.target,
        canonicalName: rel.targetCanonical,
        type: 'Other',
        description: rel.description,
      });
    }
  }

  return {
    entities: [...entitiesByCanonical.values()].slice(0, CONSTANTS.KG_MAX_EXTRACTION_ENTITIES),
    relations: [...relationsByKey.values()].slice(0, CONSTANTS.KG_MAX_EXTRACTION_RECORDS),
  };
}

function cleanKeywordArray(raw: unknown, maxItems: number): string[] {
  if (!Array.isArray(raw)) return [];
  const seen = new Set<string>();
  const out: string[] = [];
  for (const item of raw) {
    if (typeof item !== 'string') continue;
    const cleaned = item.replace(/[\r\n]+/g, ' ').trim();
    if (!cleaned || isPlaceholder(cleaned)) continue;
    const key = cleaned.toLowerCase();
    if (!seen.has(key)) {
      seen.add(key);
      out.push(cleaned);
      if (out.length >= maxItems) break;
    }
  }
  return out;
}

/**
 * Parses the unified query analysis response (standalone query + high/low keywords).
 * Falls back gracefully to the original query when JSON parsing fails.
 */
export function parseQueryAnalysisOutput(raw: unknown, originalQuery: string): QueryAnalysisResult {
  const parsed = extractJsonObject(raw);
  if (!parsed) {
    return {
      standaloneQuery: originalQuery,
      highLevelKeywords: [],
      lowLevelKeywords: [],
      usedFallback: true,
    };
  }

  const standaloneRaw = typeof parsed.standalone_query === 'string' ? parsed.standalone_query.trim() : '';
  const standaloneQuery = standaloneRaw && !isPlaceholder(standaloneRaw) ? standaloneRaw : originalQuery;
  const highLevelKeywords = cleanKeywordArray(parsed.high_level_keywords ?? parsed.hl, 6);
  const lowLevelKeywords = cleanKeywordArray(parsed.low_level_keywords ?? parsed.ll, 8);

  return {
    standaloneQuery,
    highLevelKeywords,
    lowLevelKeywords,
    usedFallback: false,
  };
}
