export interface PprEdge {
  sourceCanonical: string;
  targetCanonical: string;
  weight: number;
  sourceChunkIds?: string[];
}

export interface PprOptions {
  /** Damping factor (probability of continuing the random walk vs teleporting back to seeds). Default 0.85. */
  damping?: number;
  /** Maximum power-iteration steps. Default 12 (converges well within 8-12 steps on 2-hop subgraphs). */
  iterations?: number;
  /** L1 convergence threshold to stop early. Default 1e-5. */
  tolerance?: number;
  /** Specificity dampening exponent on node degree to prevent generic hub nodes from dominating. Default 0.25. */
  hubDampening?: number;
}

export interface PprResult {
  /** Map of canonical entity name -> normalized PPR score in [0, 1]. */
  nodeScores: Map<string, number>;
  iterationsRun: number;
  converged: boolean;
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
  const hubDampening = options.hubDampening ?? 0.25;

  const nodes = new Set<string>();
  const adjacency = new Map<string, Array<{ target: string; weight: number }>>();

  for (const seed of seedWeights.keys()) {
    nodes.add(seed);
  }

  for (const edge of edges) {
    const s = edge.sourceCanonical;
    const t = edge.targetCanonical;
    if (!s || !t || s === t) continue;
    const w = Number.isFinite(edge.weight) && edge.weight > 0 ? edge.weight : 1;
    nodes.add(s);
    nodes.add(t);

    let listS = adjacency.get(s);
    if (!listS) {
      listS = [];
      adjacency.set(s, listS);
    }
    listS.push({ target: t, weight: w });

    let listT = adjacency.get(t);
    if (!listT) {
      listT = [];
      adjacency.set(t, listT);
    }
    listT.push({ target: s, weight: w });
  }

  if (nodes.size === 0 || seedWeights.size === 0) {
    return { nodeScores: new Map(), iterationsRun: 0, converged: true };
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

  return { nodeScores: normalized, iterationsRun, converged };
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
    // Discount overly broad entities that appear in dozens of chunks
    const specificity = 1 / Math.sqrt(chunks.length);
    const contrib = score * specificity;
    for (const chunkId of chunks) {
      chunkScores.set(chunkId, (chunkScores.get(chunkId) ?? 0) + contrib);
    }
  }

  // Bridge edges connecting TWO high-PPR nodes get an extra multi-hop bonus
  for (const edge of params.edges) {
    if (!edge.sourceChunkIds || edge.sourceChunkIds.length === 0) continue;
    const sScore = params.nodeScores.get(edge.sourceCanonical) ?? 0;
    const tScore = params.nodeScores.get(edge.targetCanonical) ?? 0;
    if (sScore <= 0 || tScore <= 0) continue;
    const bridgeScore = Math.sqrt(sScore * tScore) * 0.65;
    for (const chunkId of edge.sourceChunkIds) {
      chunkScores.set(chunkId, (chunkScores.get(chunkId) ?? 0) + bridgeScore);
    }
  }

  let maxChunk = 0;
  for (const v of chunkScores.values()) {
    if (v > maxChunk) maxChunk = v;
  }
  if (maxChunk > 0) {
    for (const [id, v] of chunkScores.entries()) {
      chunkScores.set(id, v / maxChunk);
    }
  }

  return chunkScores;
}
