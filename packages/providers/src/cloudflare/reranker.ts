import { retryWithBackoff, type RerankerProvider, type ScoredChunk } from '@rag/core';

interface CloudflareRerankItem {
  id: number;
  score: number;
}

interface CloudflareRerankResponse {
  success?: boolean;
  result?: {
    response?: CloudflareRerankItem[];
  };
  errors?: Array<{ message?: string }>;
}

/**
 * High-throughput Cross-Encoder Reranker powered by Cloudflare Workers AI
 * (`@cf/baai/bge-reranker-base`). Eliminates Cohere Trial 10-RPM bottlenecks
 * while delivering multilingual cross-encoder precision and zero-cost token
 * pruning (0 LLM tokens consumed).
 */
export class CloudflareRerankerProvider implements RerankerProvider {
  public readonly name = 'cloudflare';
  public readonly model: string;
  private readonly accountId: string;
  private readonly apiToken: string;
  private readonly timeoutMs: number;

  constructor(accountId?: string, apiToken?: string, modelName?: string, timeoutMs?: number) {
    this.accountId = accountId || process.env.CLOUDFLARE_ACCOUNT_ID || '';
    this.apiToken = apiToken || process.env.CLOUDFLARE_API_TOKEN || '';
    this.model = modelName || process.env.CLOUDFLARE_RERANKER_MODEL || '@cf/baai/bge-reranker-base';
    this.timeoutMs = timeoutMs ?? parseInt(process.env.CLOUDFLARE_RERANKER_TIMEOUT_MS || '15000', 10);

    if (!this.accountId || !this.apiToken) {
      throw new Error('CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN are required for CloudflareRerankerProvider');
    }
  }

  async rerank(query: string, candidates: ScoredChunk[], topN: number = 5): Promise<ScoredChunk[]> {
    if (!candidates || candidates.length === 0) return [];

    const url = `https://api.cloudflare.com/client/v4/accounts/${this.accountId}/ai/run/${this.model}`;
    const contexts = candidates.map((c) => ({ text: c.content.slice(0, 1800) }));

    try {
      const data = await retryWithBackoff(
        async () => {
          const controller = new AbortController();
          const timer = setTimeout(() => controller.abort(), this.timeoutMs);
          try {
            const res = await fetch(url, {
              method: 'POST',
              headers: {
                Authorization: `Bearer ${this.apiToken}`,
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                query,
                contexts,
                top_k: Math.min(topN, candidates.length),
              }),
              signal: controller.signal,
            });
            if (!res.ok) {
              const errText = await res.text().catch(() => '');
              throw new Error(`Cloudflare reranker HTTP ${res.status}: ${errText.slice(0, 200)}`);
            }
            return (await res.json()) as CloudflareRerankResponse;
          } finally {
            clearTimeout(timer);
          }
        },
        { maxAttempts: 2, baseDelayMs: 300 },
      );

      const items = data.result?.response;
      if (!Array.isArray(items) || items.length === 0) {
        return candidates.slice(0, topN);
      }

      const ranked: ScoredChunk[] = [];
      for (const item of items) {
        const candidate = candidates[item.id];
        if (!candidate) continue;
        // BGE reranker returns raw logits or sigmoid scores; normalize logits >1 or <0 via sigmoid
        const raw = Number(item.score);
        const normalizedScore =
          raw >= 0 && raw <= 1 ? raw : 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, raw))));
        ranked.push({ ...candidate, score: normalizedScore });
      }
      ranked.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
      return ranked.slice(0, topN);
    } catch (error) {
      console.warn(
        '[CloudflareReranker] Falling back to fused order:',
        error instanceof Error ? error.message : String(error),
      );
      return candidates.slice(0, topN);
    }
  }
}
