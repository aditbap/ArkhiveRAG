import type { RerankerProvider } from '@rag/core';
import { CloudflareRerankerProvider } from './cloudflare/reranker.js';
import { CohereRerankerProvider } from './cohere/reranker.js';

export type RerankerProviderKind = 'cloudflare' | 'cohere' | 'auto' | 'none';

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

  if (kind === 'cloudflare') {
    try {
      return new CloudflareRerankerProvider();
    } catch {
      return null;
    }
  }

  if (kind === 'cohere') {
    try {
      return new CohereRerankerProvider();
    } catch {
      return null;
    }
  }

  // auto: prefer Cloudflare BGE Reranker for throughput, then Cohere
  if (process.env.CLOUDFLARE_ACCOUNT_ID && process.env.CLOUDFLARE_API_TOKEN) {
    try {
      return new CloudflareRerankerProvider();
    } catch {
      /* fall through */
    }
  }

  if (process.env.COHERE_API_KEY) {
    try {
      return new CohereRerankerProvider();
    } catch {
      /* fall through */
    }
  }

  return null;
}
