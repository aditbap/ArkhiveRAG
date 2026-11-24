import type { RerankerProvider, ScoredChunk } from '@rag/core';
import { CloudflareRerankerProvider } from './cloudflare/reranker.js';
import { CohereRerankerProvider } from './cohere/reranker.js';

export type RerankerProviderKind = 'cloudflare' | 'cohere' | 'auto' | 'none';

/**
 * Timeout & circuit-breaker wrapper around any Cross-Encoder RerankerProvider.
 * - Enforces a strict per-call timeout (default 1500ms) so slow/rate-limited upstream APIs
 *   never block chat latency.
 * - Opens a short 30s circuit breaker after `failureThreshold` consecutive failures so
 *   fallback to calibrated LTR ordering is instant and deterministic.
 */
export class ResilientRerankerProvider implements RerankerProvider {
  readonly name: string;
  readonly model: string;
  private readonly inner: RerankerProvider;
  private readonly timeoutMs: number;
  private readonly failureThreshold: number;
  private readonly cooldownMs: number;
  private consecutiveFailures = 0;
  private circuitOpenUntil = 0;

  constructor(
    inner: RerankerProvider,
    options: { timeoutMs?: number; failureThreshold?: number; cooldownMs?: number } = {},
  ) {
    this.inner = inner;
    this.name = inner.name;
    this.model = inner.model;
    this.timeoutMs = options.timeoutMs ?? parseInt(process.env.RERANKER_TIMEOUT_MS || '1500', 10);
    this.failureThreshold = options.failureThreshold ?? 3;
    this.cooldownMs = options.cooldownMs ?? 30_000;
  }

  isCircuitOpen(now = Date.now()): boolean {
    return now < this.circuitOpenUntil;
  }

  async rerank(query: string, chunks: ScoredChunk[], topN: number): Promise<ScoredChunk[]> {
    if (this.isCircuitOpen()) {
      throw new Error(`Reranker circuit open (${this.name})`);
    }
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      const timeoutPromise = new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error(`Reranker timeout after ${this.timeoutMs}ms`)), this.timeoutMs);
      });
      const result = await Promise.race([this.inner.rerank(query, chunks, topN), timeoutPromise]);
      this.consecutiveFailures = 0;
      return result;
    } catch (err) {
      this.consecutiveFailures++;
      if (this.consecutiveFailures >= this.failureThreshold) {
        this.circuitOpenUntil = Date.now() + this.cooldownMs;
      }
      throw err;
    } finally {
      if (timer) clearTimeout(timer);
    }
  }
}

/**
 * Provider-agnostic Cross-Encoder Reranker factory.
 * - `cloudflare`: Uses `@cf/baai/bge-reranker-base` (high concurrency, no 10-RPM trial cap).
 * - `cohere`: Uses Cohere `rerank-v3.5` (with local `COHERE_MAX_RPM` guard).
 * - `auto` (default): Prefers Cloudflare BGE reranker when Cloudflare credentials exist,
 *   otherwise falls back to Cohere when `COHERE_API_KEY` is configured.
 * - `none`: Disables cross-encoder reranking.
 */
export function createRerankerProvider(explicitKind?: RerankerProviderKind): RerankerProvider | null {
  const kind = (explicitKind || process.env.RERANKER_PROVIDER || 'auto').toLowerCase() as RerankerProviderKind;
  if (kind === 'none' || kind === ('off' as RerankerProviderKind)) {
    return null;
  }

  const wrap = (provider: RerankerProvider): RerankerProvider => new ResilientRerankerProvider(provider);

  if (kind === 'cloudflare') {
    try {
      return wrap(new CloudflareRerankerProvider());
    } catch {
      return null;
    }
  }

  if (kind === 'cohere') {
    try {
      return wrap(new CohereRerankerProvider());
    } catch {
      return null;
    }
  }

  // auto: prefer Cloudflare BGE Reranker for throughput, then Cohere
  if (process.env.CLOUDFLARE_ACCOUNT_ID && process.env.CLOUDFLARE_API_TOKEN) {
    try {
      return wrap(new CloudflareRerankerProvider());
    } catch {
      /* fall through */
    }
  }

  if (process.env.COHERE_API_KEY) {
    try {
      return wrap(new CohereRerankerProvider());
    } catch {
      /* fall through */
    }
  }

  return null;
}
