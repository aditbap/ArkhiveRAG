export type KgRelationType =
  | 'supersedes'
  | 'governs'
  | 'owns'
  | 'depends_on'
  | 'reports_to'
  | 'synonym'
  | 'related';

export interface PprEdge {
  sourceCanonical: string;
  targetCanonical: string;
  weight: number;
  sourceChunkIds?: string[];
  relationType?: KgRelationType | string;
  confidence?: number;
  isDirected?: boolean;
}

export interface PprOptions {
  /** Damping factor (probability of continuing the random walk vs teleporting back to seeds). Default 0.85. */
  damping?: number;
  /** Maximum power-iteration steps. Default 12 (converges well within 8-12 steps on 2-hop subgraphs). */
  iterations?: number;
  /** L1 convergence threshold to stop early. Default 1e-5. */
  tolerance?: number;
  /**
   * Specificity dampening exponent on node degree to prevent generic hub nodes from dominating.
   * When omitted, automatically adapts to subgraph density in `[0.35, 0.70]`.
   */
  hubDampening?: number;
}

export interface PprPathTrace {
  nodes: string[];
  score: number;
}

export interface PprResult {
  /** Map of canonical entity name -> normalized PPR score in [0, 1]. */
  nodeScores: Map<string, number>;
  iterationsRun: number;
  converged: boolean;
  /** Effective hub dampening applied (adaptive when not explicitly specified). */
  hubDampeningUsed: number;
}

const RELATION_TYPE_MULTIPLIERS: Record<string, number> = {
  supersedes: 1.25,
  governs: 1.25,
  owns: 1.2,
  depends_on: 1.15,
  reports_to: 1.15,
  synonym: 0.9,
  related: 1.0,
};

export function relationTypeMultiplier(relationType?: string | null): number {
  if (!relationType) return 1.0;
  return RELATION_TYPE_MULTIPLIERS[relationType.toLowerCase()] ?? 1.0;
}

/**
 * Computes graph-density-adaptive hub dampening in `[0.35, 0.70]`:
 * - Sparse subgraphs (avgDegree <= 2) stay gentle (~0.40) to preserve multi-hop bridges.
 * - Dense enterprise subgraphs (avgDegree >= 10) dampen harder (~0.66) to suppress mega-hubs.
 */
export function computeAdaptiveHubDampening(nodeCount: number, edgeCount: number): number {
  if (nodeCount <= 1 || edgeCount <= 0) return 0.35;
  const avgDegree = (2 * edgeCount) / nodeCount;
  return Math.max(0.35, Math.min(0.7, 0.35 + 0.25 * (avgDegree / 8)));
}

/**
 * Graph-Aware Personalized PageRank (HippoRAG / FastGraphRAG style) over a tenant-scoped,
 * ACL-filtered subgraph. Runs in pure TypeScript over sparse adjacency maps (<5ms on
 * 2-hop neighborhoods up to 500 nodes / 2,000 edges) with zero external graph DB needed.
 */
export function computePersonalizedPageRank(
  edges: PprEdge[],
  seedWeights: Map<string, number>,
  options: PprOptions = {},
): PprResult {
  const damping = options.damping ?? 0.85;
  const maxIterations = options.iterations ?? 12;
  const tolerance = options.tolerance ?? 1e-5;

  const nodes = new Set<string>();
  const adjacency = new Map<string, Array<{ target: string; weight: number }>>();
  let validEdgeCount = 0;

  for (const seed of seedWeights.keys()) {
    nodes.add(seed);
  }

  for (const edge of edges) {
    const s = edge.sourceCanonical;
    const t = edge.targetCanonical;
    if (!s || !t || s === t) continue;
    const baseW = Number.isFinite(edge.weight) && edge.weight > 0 ? edge.weight : 1;
    const conf =
      typeof edge.confidence === 'number' && Number.isFinite(edge.confidence)
        ? Math.max(0.2, Math.min(1, edge.confidence))
        : 1.0;
    const typeMult = relationTypeMultiplier(edge.relationType);
    const forwardW = baseW * conf * typeMult;
    // Directed edges still permit damped reverse traversal (0.65x) so reverse multi-hop queries succeed
    const reverseW = edge.isDirected ? forwardW * 0.65 : forwardW;

    nodes.add(s);
    nodes.add(t);
    validEdgeCount++;

    let listS = adjacency.get(s);
    if (!listS) {
      listS = [];
      adjacency.set(s, listS);
    }
    listS.push({ target: t, weight: forwardW });

    let listT = adjacency.get(t);
    if (!listT) {
      listT = [];
      adjacency.set(t, listT);
    }
    listT.push({ target: s, weight: reverseW });
  }

  const hubDampening = options.hubDampening ?? computeAdaptiveHubDampening(nodes.size, validEdgeCount);

  if (nodes.size === 0 || seedWeights.size === 0) {
    return { nodeScores: new Map(), iterationsRun: 0, converged: true, hubDampeningUsed: hubDampening };
  }

  // Normalize personalization vector (restart distribution)
  const personalization = new Map<string, number>();
  let totalSeedWeight = 0;
  for (const [node, rawWeight] of seedWeights.entries()) {
    const w = Math.max(0, rawWeight);
    if (w > 0) {
      personalization.set(node, w);
      totalSeedWeight += w;
    }
  }
  if (totalSeedWeight <= 0) {
    const uniform = 1 / nodes.size;
    for (const node of nodes) personalization.set(node, uniform);
  } else {
    for (const [node, w] of personalization.entries()) {
      personalization.set(node, w / totalSeedWeight);
    }
  }

  // Precompute outgoing weight totals with optional mild hub dampening
  const outWeightSum = new Map<string, number>();
  for (const [node, neighbors] of adjacency.entries()) {
    let sum = 0;
    for (const n of neighbors) {
      const targetDeg = adjacency.get(n.target)?.length ?? 1;
      const effWeight = hubDampening > 0 ? n.weight / Math.pow(targetDeg, hubDampening) : n.weight;
      n.weight = effWeight;
      sum += effWeight;
    }
    outWeightSum.set(node, sum);
  }

  // Initialize rank vector to personalization vector
  let rank = new Map<string, number>();
  for (const node of nodes) {
    rank.set(node, personalization.get(node) ?? 0);
  }

  let iterationsRun = 0;
  let converged = false;

  for (let iter = 0; iter < maxIterations; iter++) {
    iterationsRun = iter + 1;
    const next = new Map<string, number>();
    let danglingMass = 0;

    for (const node of nodes) {
      const p = personalization.get(node) ?? 0;
      next.set(node, (1 - damping) * p);
    }

    for (const node of nodes) {
      const r = rank.get(node) ?? 0;
      if (r === 0) continue;
      const neighbors = adjacency.get(node);
      const sum = outWeightSum.get(node) ?? 0;
      if (!neighbors || neighbors.length === 0 || sum <= 0) {
        danglingMass += r;
        continue;
      }
      const distributed = damping * r;
      for (const n of neighbors) {
        next.set(n.target, (next.get(n.target) ?? 0) + distributed * (n.weight / sum));
      }
    }

    // Redistribute dangling mass back along personalization vector
    if (danglingMass > 0) {
      const bonus = damping * danglingMass;
      for (const [node, p] of personalization.entries()) {
        next.set(node, (next.get(node) ?? 0) + bonus * p);
      }
    }

    // L1 convergence check
    let diff = 0;
    for (const node of nodes) {
      diff += Math.abs((next.get(node) ?? 0) - (rank.get(node) ?? 0));
    }
    rank = next;
    if (diff < tolerance) {
      converged = true;
      break;
    }
  }

  // Min-max normalize scores to [0, 1] for stable downstream fusion/LTR features
  let maxScore = 0;
  for (const score of rank.values()) {
    if (score > maxScore) maxScore = score;
  }
  const normalized = new Map<string, number>();
  if (maxScore > 0) {
    for (const [node, score] of rank.entries()) {
      normalized.set(node, score / maxScore);
    }
  }

  return { nodeScores: normalized, iterationsRun, converged, hubDampeningUsed: hubDampening };
}

/**
 * Extracts top highest-relevance 1-hop and 2-hop reasoning paths from seed nodes
 * across PPR-scored neighbors for explainability and path-relevance verification.
 */
export function traceTopPprPaths(
  edges: PprEdge[],
  seedNodes: string[],
  nodeScores: Map<string, number>,
  maxPaths = 5,
): PprPathTrace[] {
  const adj = new Map<string, Array<{ target: string; weight: number }>>();
  for (const e of edges) {
    if (!e.sourceCanonical || !e.targetCanonical || e.sourceCanonical === e.targetCanonical) continue;
    const w = (e.weight > 0 ? e.weight : 1) * relationTypeMultiplier(e.relationType);
    let a = adj.get(e.sourceCanonical);
    if (!a) {
      a = [];
      adj.set(e.sourceCanonical, a);
    }
    a.push({ target: e.targetCanonical, weight: w });
    let b = adj.get(e.targetCanonical);
    if (!b) {
      b = [];
      adj.set(e.targetCanonical, b);
    }
    b.push({ target: e.sourceCanonical, weight: e.isDirected ? w * 0.65 : w });
  }

  const seedSet = new Set(seedNodes);
  const paths: PprPathTrace[] = [];
  const seen = new Set<string>();

  for (const seed of seedNodes.slice(0, 6)) {
    const sScore = nodeScores.get(seed) ?? 0;
    if (sScore <= 0) continue;
    const hop1 = (adj.get(seed) ?? [])
      .map((n) => ({ node: n.target, score: nodeScores.get(n.target) ?? 0 }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 4);

    for (const h1 of hop1) {
      if (h1.score <= 0) continue;
      const key1 = `${seed}->${h1.node}`;
      if (!seen.has(key1) && !seedSet.has(h1.node)) {
        seen.add(key1);
        paths.push({ nodes: [seed, h1.node], score: Math.sqrt(sScore * h1.score) });
      }
      const hop2 = (adj.get(h1.node) ?? [])
        .filter((n) => n.target !== seed)
        .map((n) => ({ node: n.target, score: nodeScores.get(n.target) ?? 0 }))
        .sort((a, b) => b.score - a.score)
        .slice(0, 2);

      for (const h2 of hop2) {
        if (h2.score <= 0) continue;
        const key2 = `${seed}->${h1.node}->${h2.node}`;
        if (!seen.has(key2)) {
          seen.add(key2);
          paths.push({
            nodes: [seed, h1.node, h2.node],
            score: Math.cbrt(sScore * h1.score * h2.score),
          });
        }
      }
    }
  }

  return paths.sort((a, b) => b.score - a.score).slice(0, maxPaths);
}

/**
 * Projects entity & edge PPR scores onto their provenance chunk IDs, returning a
 * normalized `[0, 1]` PPR relevance map per chunk ID.
 */
export function propagatePprToChunks(params: {
  nodeScores: Map<string, number>;
  entityChunks: Map<string, string[]>;
  edges: PprEdge[];
}): Map<string, number> {
  const chunkScores = new Map<string, number>();

  for (const [canonical, score] of params.nodeScores.entries()) {
    if (score <= 0) continue;
    const chunks = params.entityChunks.get(canonical);
    if (!chunks || chunks.length === 0) continue;
    // Bounded IDF specificity [0.22, 1.0]: suppresses mega-hubs without crushing key medium-frequency entities
    const specificity = Math.max(0.22, Math.min(1.0, 1 / Math.pow(chunks.length, 0.7)));
    const contrib = score * specificity;
    for (const chunkId of chunks) {
      chunkScores.set(chunkId, (chunkScores.get(chunkId) ?? 0) + contrib);
    }
  }

  // Bridge edges connecting TWO high-PPR nodes get a specificity-dampened multi-hop bonus
  for (const edge of params.edges) {
    if (!edge.sourceChunkIds || edge.sourceChunkIds.length === 0) continue;
    const sScore = params.nodeScores.get(edge.sourceCanonical) ?? 0;
    const tScore = params.nodeScores.get(edge.targetCanonical) ?? 0;
    if (sScore <= 0 || tScore <= 0) continue;
    const edgeSpecificity = 1 / Math.sqrt(edge.sourceChunkIds.length);
    const bridgeScore = Math.sqrt(sScore * tScore) * 0.45 * edgeSpecificity;
    for (const chunkId of edge.sourceChunkIds) {
      chunkScores.set(chunkId, (chunkScores.get(chunkId) ?? 0) + bridgeScore);
    }
  }

  // Sublinear log1p compression prevents hub-heavy chunks with 10 noisy edges from swamping 2-hop bridges
  let maxChunk = 0;
  for (const [id, v] of chunkScores.entries()) {
    const compressed = Math.log1p(v);
    chunkScores.set(id, compressed);
    if (compressed > maxChunk) maxChunk = compressed;
  }
  if (maxChunk > 0) {
    for (const [id, v] of chunkScores.entries()) {
      chunkScores.set(id, v / maxChunk);
    }
  }

  return chunkScores;
}
