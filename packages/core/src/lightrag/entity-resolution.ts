import { CONSTANTS } from '../constants.js';
import type { ExtractedEntity, ExtractedRelation } from '../types/lightrag.js';
import { mergeDescriptionFragments, mergeKeywords } from './parser.js';

const CORPORATE_PREFIX_RE = /^(?:the|policy|standard|procedure|sop|document|doc|kebijakan|prosedur|dokumen)\s+/i;
const EXACT_DOC_CODE_RE = /\b([A-Z]{2,6}-[A-Z0-9]{2,}(?:-[A-Z0-9]{2,})?)\b/i;

/**
 * Produces a normalized alias key for corporate entity resolution.
 * Collapses casing, punctuation, extra spaces, and common wrapper words
 * (e.g., `"Policy POL-SEC-409"` -> `"pol-sec-409"`, `"Security Policy (POL-SEC-409)"`).
 */
export function normalizeEntityAliasKey(name: string): string {
  const cleaned = name
    .trim()
    .toLowerCase()
    .replace(/['"`()[\]{}]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  return cleaned.replace(CORPORATE_PREFIX_RE, '').trim() || cleaned;
}

/**
 * Extracts a canonical corporate identifier code if present (e.g. `POL-SEC-409`, `SOP-JKT-882`, `INC-2026-014`).
 */
export function extractEntityCode(name: string): string | null {
  const match = name.match(EXACT_DOC_CODE_RE);
  return match?.[1] ? match[1].toLowerCase() : null;
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
