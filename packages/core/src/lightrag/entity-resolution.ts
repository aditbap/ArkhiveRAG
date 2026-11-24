import { CONSTANTS } from '../constants.js';
import type { ExtractedEntity, ExtractedRelation } from '../types/lightrag.js';
import { mergeDescriptionFragments, mergeKeywords } from './parser.js';

const CORPORATE_PREFIX_RE =
  /^(?:the|policy|standard|procedure|sop|runbook|document|doc|system|platform|service|team|kebijakan|prosedur|dokumen|sistem|tim|panduan)\s+/i;
const EXACT_DOC_CODE_RE = /\b([A-Z]{2,6}-[A-Z0-9]{1,8}(?:-[A-Z0-9]{1,8})*)\b/i;

/**
 * Extracts a canonical corporate identifier code if present (e.g. `POL-SEC-409`, `SOP-JKT-882`, `INC-2026-014`).
 * Requires at least one digit so ordinary hyphenated words (`Zero-Trust`, `Cold-Chain`, `Multi-Tenant`)
 * are never mistaken for distinct corporate codes.
 */
export function extractEntityCode(name: string): string | null {
  for (const match of name.matchAll(new RegExp(EXACT_DOC_CODE_RE.source, 'gi'))) {
    const candidate = match[1];
    if (candidate && /\d/.test(candidate)) {
      return candidate.toLowerCase();
    }
  }
  return null;
}

const ABBREVIATION_EXPANSIONS: Record<string, string> = {
  sec: 'security',
  ops: 'operations',
  infra: 'infrastructure',
  auth: 'authentication',
  mgmt: 'management',
  eng: 'engineering',
  fin: 'finance',
  hr: 'human resources',
  jkt: 'jakarta',
  sby: 'surabaya',
  mks: 'makassar',
  bpn: 'balikpapan',
};

/**
 * Produces a normalized alias key for corporate entity resolution.
 * Collapses casing, punctuation, extra spaces, common Indonesian/English wrapper words,
 * and standard corporate abbreviations (e.g., `"Policy POL-SEC-409"` -> `"pol-sec-409"`).
 */
export function normalizeEntityAliasKey(name: string): string {
  const cleaned = name
    .trim()
    .toLowerCase()
    .replace(/['"`()[\]{}]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  const stripped = cleaned.replace(CORPORATE_PREFIX_RE, '').trim() || cleaned;
  // Preserve exact hyphenated codes (containing digits) untouched; expand standalone abbreviations in prose names
  if (extractEntityCode(stripped)) return stripped;
  return stripped
    .split(' ')
    .map((tok) => ABBREVIATION_EXPANSIONS[tok] ?? tok)
    .join(' ');
}

export function cosineSimilarityVectors(a: number[], b: number[]): number {
  const len = Math.min(a.length, b.length);
  if (len === 0) return 0;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < len; i++) {
    const va = a[i] ?? 0;
    const vb = b[i] ?? 0;
    dot += va * vb;
    na += va * va;
    nb += vb * vb;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom > 0 ? dot / denom : 0;
}

/**
 * Context-embedding entity disambiguation guard: prevents homonyms (same/similar surface
 * name with incompatible types or divergent contextual embeddings) from falsely merging.
 */
export function shouldMergeEntitiesWithContext(params: {
  nameA: string;
  typeA: string;
  embeddingA?: number[] | null;
  nameB: string;
  typeB: string;
  embeddingB?: number[] | null;
  minContextCosine?: number;
}): boolean {
  if (!areEntityTypesCompatible(params.typeA, params.typeB)) return false;
  const codeA = extractEntityCode(params.nameA);
  const codeB = extractEntityCode(params.nameB);
  if (codeA && codeB && codeA !== codeB) return false;
  if (params.embeddingA && params.embeddingB && params.embeddingA.length > 0 && params.embeddingB.length > 0) {
    const cos = cosineSimilarityVectors(params.embeddingA, params.embeddingB);
    return cos >= (params.minContextCosine ?? 0.78);
  }
  return true;
}

export interface ClassifiedRelationMeta {
  relationType: 'supersedes' | 'governs' | 'owns' | 'depends_on' | 'reports_to' | 'synonym' | 'related';
  isDirected: boolean;
}

/**
 * Deterministic relation-type and directionality classifier (<0.1ms, zero LLM cost).
 */
export function classifyRelationType(keywords: string, description: string): ClassifiedRelationMeta {
  const text = `${keywords} ${description}`.toLowerCase();
  if (/\b(?:supersedes|replaces|obsoletes|menggantikan)\b/.test(text)) {
    return { relationType: 'supersedes', isDirected: true };
  }
  if (/\b(?:governs|mandates|requires|enforces|regulates|mengatur|mewajibkan)\b/.test(text)) {
    return { relationType: 'governs', isDirected: true };
  }
  if (/\b(?:owns|manages|leads|chairs|operates|maintains|mengelola|memimpin)\b/.test(text)) {
    return { relationType: 'owns', isDirected: true };
  }
  if (/\b(?:reports\s+to|escalates\s+to|melapor\s+kepada)\b/.test(text)) {
    return { relationType: 'reports_to', isDirected: true };
  }
  if (/\b(?:depends\s+on|uses|integrates\s+with|routes\s+to|menggunakan|bergantung)\b/.test(text)) {
    return { relationType: 'depends_on', isDirected: true };
  }
  if (/\b(?:synonym|alias|also\s+known\s+as|aka|dikenal\s+sebagai)\b/.test(text)) {
    return { relationType: 'synonym', isDirected: false };
  }
  return { relationType: 'related', isDirected: false };
}

export interface SynonymyEntityNode {
  canonicalName: string;
  entityName: string;
  entityType: string;
  embedding: number[];
  sourceChunkIds: string[];
  documentIds: string[];
}

export interface SynonymyEdgeCandidate {
  sourceCanonical: string;
  targetCanonical: string;
  sourceEntity: string;
  targetEntity: string;
  similarity: number;
  weight: number;
  sourceChunkIds: string[];
  documentIds: string[];
}

/**
 * Discovers automatic cosine-similarity synonymy edges (HippoRAG-style) across entities
 * with compatible types and high semantic similarity (`>= threshold`, default `0.82`),
 * while strictly refusing to bridge two DIFFERENT explicit corporate codes.
 */
export function buildCosineSynonymyEdges(
  entities: SynonymyEntityNode[],
  options: { minSimilarity?: number; maxEdgesPerNode?: number } = {},
): SynonymyEdgeCandidate[] {
  const minSim = options.minSimilarity ?? 0.82;
  const maxPerNode = options.maxEdgesPerNode ?? 2;
  const out: SynonymyEdgeCandidate[] = [];
  const seenPairs = new Set<string>();

  for (let i = 0; i < entities.length; i++) {
    const a = entities[i]!;
    if (!a.embedding || a.embedding.length === 0) continue;
    const codeA = extractEntityCode(a.entityName);
    const matches: Array<{ target: SynonymyEntityNode; sim: number }> = [];

    for (let j = 0; j < entities.length; j++) {
      if (i === j) continue;
      const b = entities[j]!;
      if (!b.embedding || b.embedding.length === 0) continue;
      if (a.canonicalName === b.canonicalName) continue;
      if (!areEntityTypesCompatible(a.entityType, b.entityType)) continue;
      const codeB = extractEntityCode(b.entityName);
      if (codeA && codeB && codeA !== codeB) continue;

      const sim = cosineSimilarityVectors(a.embedding, b.embedding);
      if (sim >= minSim) {
        matches.push({ target: b, sim });
      }
    }

    matches.sort((x, y) => y.sim - x.sim);
    for (const m of matches.slice(0, maxPerNode)) {
      const [s, t] =
        a.canonicalName < m.target.canonicalName ? [a, m.target] : [m.target, a];
      const key = `${s.canonicalName}::${t.canonicalName}`;
      if (seenPairs.has(key)) continue;
      seenPairs.add(key);
      out.push({
        sourceCanonical: s.canonicalName,
        targetCanonical: t.canonicalName,
        sourceEntity: s.entityName,
        targetEntity: t.entityName,
        similarity: m.sim,
        weight: Number((m.sim * 0.85).toFixed(4)),
        sourceChunkIds: [...new Set([...s.sourceChunkIds, ...t.sourceChunkIds])].slice(0, 6),
        documentIds: [...new Set([...s.documentIds, ...t.documentIds])].slice(0, 6),
      });
    }
  }

  return out;
}

/**
 * Character trigram Dice similarity (pure TS mirror of Postgres `pg_trgm` similarity)
 * for fast deterministic entity alias matching without external deps.
 */
export function trigramSimilarity(a: string, b: string): number {
  const s1 = `  ${a.toLowerCase().trim()} `;
  const s2 = `  ${b.toLowerCase().trim()} `;
  if (s1 === s2) return 1;
  if (s1.length < 3 || s2.length < 3) return 0;

  const grams1 = new Set<string>();
  for (let i = 0; i <= s1.length - 3; i++) grams1.add(s1.slice(i, i + 3));
  const grams2 = new Set<string>();
  for (let i = 0; i <= s2.length - 3; i++) grams2.add(s2.slice(i, i + 3));

  let intersection = 0;
  for (const g of grams1) {
    if (grams2.has(g)) intersection++;
  }
  return (2 * intersection) / (grams1.size + grams2.size);
}

/**
 * Checks whether two entity types are compatible for automatic resolution.
 * `'Other'` / `'Concept'` are wildcards that can merge into a more specific type.
 */
export function areEntityTypesCompatible(typeA: string, typeB: string): boolean {
  const a = typeA.trim().toLowerCase();
  const b = typeB.trim().toLowerCase();
  if (a === b) return true;
  if (a === 'other' || b === 'other') return true;
  // Policy/Document/Regulation/Standard often get tagged interchangeably by 8B extractors
  const docGroup = new Set(['policy', 'document', 'regulation', 'standard', 'procedure', 'artifact', 'concept']);
  if (docGroup.has(a) && docGroup.has(b)) return true;
  return false;
}

export interface EntityAliasMapping {
  aliasCanonical: string;
  aliasName: string;
  targetCanonical: string;
  similarity: number;
}

export interface InDocResolutionResult {
  entities: ExtractedEntity[];
  relations: ExtractedRelation[];
  aliases: EntityAliasMapping[];
}

/**
 * Resolves and deduplicates entity variants within a single extraction batch before
 * hitting PostgreSQL. Merges:
 * 1. Entities sharing an exact corporate code (e.g. `POL-SEC-409` & `Security Policy POL-SEC-409`)
 * 2. Entities sharing the same prefix-stripped alias key (`Policy 409` & `The Policy 409`)
 * 3. High-similarity trigram variants (`similarity >= 0.84`) with compatible entity types.
 *
 * All relation endpoints are automatically rewritten to the surviving canonical names
 * and duplicate/self-loop edges created by merging are collapsed cleanly.
 */
export function resolveInDocumentEntities(
  entities: ExtractedEntity[],
  relations: ExtractedRelation[],
  trigramThreshold = 0.84,
): InDocResolutionResult {
  if (entities.length <= 1) {
    return { entities, relations, aliases: [] };
  }

  // Prefer richer/longer non-'Other' names as canonical representatives (or exact code holders)
  const sorted = [...entities].sort((a, b) => {
    const typeScoreA = a.type !== 'Other' ? 1 : 0;
    const typeScoreB = b.type !== 'Other' ? 1 : 0;
    if (typeScoreB !== typeScoreA) return typeScoreB - typeScoreA;
    if (b.description.length !== a.description.length) return b.description.length - a.description.length;
    return a.canonicalName.localeCompare(b.canonicalName);
  });

  const representatives: ExtractedEntity[] = [];
  const repByCode = new Map<string, ExtractedEntity>();
  const repByAliasKey = new Map<string, ExtractedEntity>();
  const redirectMap = new Map<string, { target: ExtractedEntity; similarity: number }>();
  const aliases: EntityAliasMapping[] = [];

  for (const entity of sorted) {
    const code = extractEntityCode(entity.name);
    const aliasKey = normalizeEntityAliasKey(entity.name);

    let matchedRep: ExtractedEntity | undefined;
    let matchedSim = 0;

    // 1. Exact corporate code match (strongest signal)
    if (code && repByCode.has(code)) {
      const candidate = repByCode.get(code)!;
      if (areEntityTypesCompatible(candidate.type, entity.type)) {
        matchedRep = candidate;
        matchedSim = 0.98;
      }
    }

    // 2. Prefix-stripped normalized key match
    if (!matchedRep && aliasKey.length >= 3 && repByAliasKey.has(aliasKey)) {
      const candidate = repByAliasKey.get(aliasKey)!;
      if (areEntityTypesCompatible(candidate.type, entity.type)) {
        matchedRep = candidate;
        matchedSim = 0.95;
      }
    }

    // 3. High trigram similarity match (guard against short acronyms colliding like "S3" vs "S4")
    if (!matchedRep && aliasKey.length >= 6) {
      let bestSim = trigramThreshold;
      for (const rep of representatives) {
        if (!areEntityTypesCompatible(rep.type, entity.type)) continue;
        // Never merge two DIFFERENT explicit corporate codes (e.g. POL-SEC-409 vs POL-SEC-410)
        const repCode = extractEntityCode(rep.name);
        if (code && repCode && code !== repCode) continue;
        const sim = trigramSimilarity(aliasKey, normalizeEntityAliasKey(rep.name));
        if (sim >= bestSim) {
          bestSim = sim;
          matchedRep = rep;
          matchedSim = sim;
        }
      }
    }

    if (matchedRep && matchedRep.canonicalName !== entity.canonicalName) {
      matchedRep.description = mergeDescriptionFragments(matchedRep.description, [entity.description]);
      if (matchedRep.type === 'Other' && entity.type !== 'Other') {
        matchedRep.type = entity.type;
      }
      redirectMap.set(entity.canonicalName, { target: matchedRep, similarity: matchedSim });
      aliases.push({
        aliasCanonical: entity.canonicalName,
        aliasName: entity.name,
        targetCanonical: matchedRep.canonicalName,
        similarity: matchedSim,
      });
    } else if (!matchedRep) {
      const clone: ExtractedEntity = { ...entity };
      representatives.push(clone);
      if (code && !repByCode.has(code)) repByCode.set(code, clone);
      if (aliasKey.length >= 3 && !repByAliasKey.has(aliasKey)) repByAliasKey.set(aliasKey, clone);
    } else {
      // Same canonicalName duplicate
      matchedRep.description = mergeDescriptionFragments(matchedRep.description, [entity.description]);
    }
  }

  // Rewrite and deduplicate relations using resolved canonical targets
  const relMap = new Map<string, ExtractedRelation>();
  for (const rel of relations) {
    const srcRedirect = redirectMap.get(rel.sourceCanonical)?.target;
    const tgtRedirect = redirectMap.get(rel.targetCanonical)?.target;

    let srcCanonical = srcRedirect ? srcRedirect.canonicalName : rel.sourceCanonical;
    let tgtCanonical = tgtRedirect ? tgtRedirect.canonicalName : rel.targetCanonical;
    let srcName = srcRedirect ? srcRedirect.name : rel.source;
    let tgtName = tgtRedirect ? tgtRedirect.name : rel.target;

    // Drop self-loops created by merging two aliases of the same entity
    if (!srcCanonical || !tgtCanonical || srcCanonical === tgtCanonical) continue;

    // Maintain canonical undirected edge orientation
    if (srcCanonical > tgtCanonical) {
      [srcCanonical, tgtCanonical] = [tgtCanonical, srcCanonical];
      [srcName, tgtName] = [tgtName, srcName];
    }

    const key = `${srcCanonical}::${tgtCanonical}`;
    const existing = relMap.get(key);
    if (existing) {
      existing.weight += rel.weight;
      existing.description = mergeDescriptionFragments(existing.description, [rel.description]);
      existing.keywords = mergeKeywords(existing.keywords.join(', '), rel.keywords)
        .split(',')
        .map((k) => k.trim())
        .filter(Boolean);
    } else {
      relMap.set(key, {
        source: srcName.slice(0, CONSTANTS.KG_ENTITY_NAME_MAX_LENGTH),
        target: tgtName.slice(0, CONSTANTS.KG_ENTITY_NAME_MAX_LENGTH),
        sourceCanonical: srcCanonical.slice(0, CONSTANTS.KG_ENTITY_NAME_MAX_LENGTH),
        targetCanonical: tgtCanonical.slice(0, CONSTANTS.KG_ENTITY_NAME_MAX_LENGTH),
        keywords: [...rel.keywords],
        description: rel.description,
        weight: rel.weight,
      });
    }
  }

  return {
    entities: representatives,
    relations: [...relMap.values()],
    aliases,
  };
}
