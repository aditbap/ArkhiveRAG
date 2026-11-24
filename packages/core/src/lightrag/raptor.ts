import crypto from 'crypto';

export interface CommunityNodeInput {
  canonicalName: string;
  entityName: string;
  entityType: string;
  description: string;
  documentIds: string[];
}

export interface CommunityEdgeInput {
  sourceCanonical: string;
  targetCanonical: string;
  weight: number;
  keywords: string;
  description: string;
  documentIds: string[];
}

export interface DetectedCommunity {
  clusterKey: string;
  clusterHash: string;
  level: number;
  title: string;
  entityCanonicals: string[];
  entityNames: string[];
  documentIds: string[];
  relationDescriptions: string[];
  entityDescriptions: string[];
}

/**
 * Deterministic Weighted Label Propagation community detection (pure TypeScript, O(E * iters),
 * zero external graph libraries). Groups tightly connected entities into coherent corporate
 * topic clusters for RAPTOR / GraphRAG community summaries (#8).
 */
export function detectGraphCommunities(
  entities: CommunityNodeInput[],
  relations: CommunityEdgeInput[],
  options: { maxIterations?: number; minClusterSize?: number; maxClusterSize?: number } = {},
): DetectedCommunity[] {
  const maxIterations = options.maxIterations ?? 15;
  const minClusterSize = options.minClusterSize ?? 2;
  const maxClusterSize = options.maxClusterSize ?? 25;

  if (entities.length < minClusterSize) return [];

  const entityMap = new Map(entities.map((e) => [e.canonicalName, e]));
  const adjacency = new Map<string, Array<{ target: string; weight: number }>>();

  for (const e of entities) adjacency.set(e.canonicalName, []);
  for (const r of relations) {
    if (!entityMap.has(r.sourceCanonical) || !entityMap.has(r.targetCanonical)) continue;
    if (r.sourceCanonical === r.targetCanonical) continue;
    const w = Number.isFinite(r.weight) && r.weight > 0 ? r.weight : 1;
    adjacency.get(r.sourceCanonical)!.push({ target: r.targetCanonical, weight: w });
    adjacency.get(r.targetCanonical)!.push({ target: r.sourceCanonical, weight: w });
  }

  // Deterministic sorted order
  const sortedNodes = [...entityMap.keys()].sort();
  const labels = new Map<string, string>();
  for (const n of sortedNodes) labels.set(n, n);

  for (let iter = 0; iter < maxIterations; iter++) {
    let changed = false;
    for (const node of sortedNodes) {
      const neighbors = adjacency.get(node);
      if (!neighbors || neighbors.length === 0) continue;

      const labelWeights = new Map<string, number>();
      for (const n of neighbors) {
        const lbl = labels.get(n.target) ?? n.target;
        labelWeights.set(lbl, (labelWeights.get(lbl) ?? 0) + n.weight);
      }

      let bestLabel = labels.get(node) ?? node;
      let bestWeight = -1;
      for (const [lbl, w] of labelWeights.entries()) {
        if (w > bestWeight || (w === bestWeight && lbl < bestLabel)) {
          bestWeight = w;
          bestLabel = lbl;
        }
      }

      if (bestLabel !== labels.get(node)) {
        labels.set(node, bestLabel);
        changed = true;
      }
    }
    if (!changed) break;
  }

  // Group nodes by final community label
  const groups = new Map<string, string[]>();
  for (const [node, lbl] of labels.entries()) {
    const arr = groups.get(lbl) ?? [];
    arr.push(node);
    groups.set(lbl, arr);
  }

  const communities: DetectedCommunity[] = [];
  for (const [, members] of groups.entries()) {
    if (members.length < minClusterSize) continue;

    // Split very large clusters into bounded chunks for concise summaries
    for (let i = 0; i < members.length; i += maxClusterSize) {
      const slice = members.slice(i, i + maxClusterSize).sort();
      if (slice.length < minClusterSize) continue;
      const memberSet = new Set(slice);

      const entityNodes = slice.map((c) => entityMap.get(c)!).filter(Boolean);
      const internalRels = relations.filter(
        (r) => memberSet.has(r.sourceCanonical) && memberSet.has(r.targetCanonical),
      );

      // Pick highest-connectivity entity as cluster anchor title
      const degCount = new Map<string, number>();
      for (const r of internalRels) {
        degCount.set(r.sourceCanonical, (degCount.get(r.sourceCanonical) ?? 0) + 1);
        degCount.set(r.targetCanonical, (degCount.get(r.targetCanonical) ?? 0) + 1);
      }
      const topEntities = [...entityNodes]
        .sort((a, b) => (degCount.get(b.canonicalName) ?? 0) - (degCount.get(a.canonicalName) ?? 0))
        .slice(0, 3);
      const title = topEntities.map((e) => e.entityName).join(' / ').slice(0, 200);

      const docIdSet = new Set<string>();
      for (const e of entityNodes) for (const d of e.documentIds) docIdSet.add(d);
      for (const r of internalRels) for (const d of r.documentIds) docIdSet.add(d);

      const clusterKey = crypto.createHash('sha256').update(slice.join('|')).digest('hex').slice(0, 24);
      const hashInput = [
        ...entityNodes.map((e) => `${e.canonicalName}:${e.description.slice(0, 200)}`),
        ...internalRels.map((r) => `${r.sourceCanonical}->${r.targetCanonical}:${r.description.slice(0, 160)}`),
      ].join('\n');
      const clusterHash = crypto.createHash('sha256').update(hashInput).digest('hex');

      communities.push({
        clusterKey,
        clusterHash,
        level: 0,
        title,
        entityCanonicals: slice,
        entityNames: entityNodes.map((e) => e.entityName),
        documentIds: [...docIdSet].sort(),
        entityDescriptions: entityNodes.map((e) => `- ${e.entityName} (${e.entityType}): ${e.description.slice(0, 220)}`),
        relationDescriptions: internalRels
          .slice(0, 20)
          .map((r) => `- ${r.sourceCanonical} <-> ${r.targetCanonical} [${r.keywords}]: ${r.description.slice(0, 180)}`),
      });
    }
  }

  return communities;
}

/**
 * Token-efficient RAPTOR community summary prompt (outputs ~150-220 tokens replacing 3,000+ raw tokens).
 */
export function buildCommunitySummaryPrompt(community: DetectedCommunity): string {
  return `Write a concise, factual executive summary (3 to 4 sentences, under 180 words) synthesizing the key corporate policies, systems, people, and relationships in this knowledge cluster. Preserve exact codes, numbers, and dates verbatim. Do NOT invent facts outside the cluster data.

Cluster Anchor: ${community.title}

Entities:
${community.entityDescriptions.slice(0, 15).join('\n')}

Relationships:
${community.relationDescriptions.slice(0, 15).join('\n')}

Return ONLY valid JSON:
{"title":"concise 4-8 word cluster title","summary":"3-4 sentence grounded summary"}`;
}

/**
 * Pure cosine similarity helper for Sentence/Proposition Multi-Vector MaxSim (#9).
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  const len = Math.min(a.length, b.length);
  if (len === 0) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < len; i++) {
    const va = a[i] ?? 0;
    const vb = b[i] ?? 0;
    dot += va * vb;
    normA += va * va;
    normB += vb * vb;
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom > 0 ? dot / denom : 0;
}

/**
 * Sentence/Proposition-Level ColBERT-Style Late Interaction MaxSim (#9):
 * Given M query sub-vectors (e.g., decomposed clauses) and N child-chunk vectors belonging
 * to a parent document section, computes `mean_{i=1..M} max_{j=1..N} cos(q_i, d_j)`.
 */
export function computePropositionMaxSim(querySubVectors: number[][], childVectors: number[][]): number {
  if (querySubVectors.length === 0 || childVectors.length === 0) return 0;
  let sumMax = 0;
  for (const qv of querySubVectors) {
    let maxSim = 0;
    for (const cv of childVectors) {
      const sim = cosineSimilarity(qv, cv);
      if (sim > maxSim) maxSim = sim;
    }
    sumMax += maxSim;
  }
  return sumMax / querySubVectors.length;
}
