import { EmbeddingProvider, EmbeddingResult, retryWithBackoff } from '@rag/core';

/**
 * Cloudflare Workers AI Embedding Provider (default: `@cf/baai/bge-m3`, 1024-dim
 * multilingual dense embeddings supporting 100+ languages including English & Indonesian).
 */
export class CloudflareEmbeddingProvider implements EmbeddingProvider {
  public readonly name = 'cloudflare';
  public readonly model: string;
  public readonly dimensions: number;
  private readonly accountId: string;
  private readonly apiToken: string;
  private readonly timeoutMs: number;
  private readonly maxBatchSize: number;

  constructor(opts?: {
    accountId?: string;
    apiToken?: string;
    modelName?: string;
    dimensions?: number;
    timeoutMs?: number;
  }) {
    this.accountId = opts?.accountId || process.env.CLOUDFLARE_ACCOUNT_ID || '';
    this.apiToken = opts?.apiToken || process.env.CLOUDFLARE_API_TOKEN || '';
    this.model = opts?.modelName || process.env.EMBEDDING_MODEL || '@cf/baai/bge-m3';
    this.dimensions = opts?.dimensions ?? parseInt(process.env.EMBEDDING_DIMENSIONS || '1024', 10);
    this.timeoutMs = opts?.timeoutMs ?? parseInt(process.env.EMBEDDING_TIMEOUT_MS || '45000', 10);
    this.maxBatchSize = 32;

    if (!this.accountId || !this.apiToken) {
      throw new Error('Cloudflare credentials (CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN) are required');
    }
  }

  private get url() {
    return `https://api.cloudflare.com/client/v4/accounts/${this.accountId}/ai/run/${this.model}`;
  }

  private async requestVectors(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    const allVectors: number[][] = [];

    for (let i = 0; i < texts.length; i += this.maxBatchSize) {
      const batch = texts.slice(i, i + this.maxBatchSize).map((t) => (t.trim() ? t.slice(0, 6000) : ' '));
      const vectors = await retryWithBackoff(
        async () => {
          const response = await fetch(this.url, {
            method: 'POST',
            headers: {
              Authorization: `Bearer ${this.apiToken}`,
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: batch }),
            signal: AbortSignal.timeout(this.timeoutMs),
          });

          if (!response.ok) {
            const err = await response.text();
            throw new Error(`Cloudflare embeddings error: ${response.status} ${err}`);
          }

          const data = (await response.json()) as {
            result?: { data?: number[][] };
          };
          const vecs = data.result?.data;
          if (!Array.isArray(vecs) || vecs.length !== batch.length) {
            throw new Error(`Cloudflare embeddings returned unexpected shape for batch of ${batch.length}`);
          }
          return vecs;
        },
        { maxAttempts: 4, baseDelayMs: 500 },
      );
      allVectors.push(...vectors);
    }

    return allVectors;
  }

  async embed(texts: string[]): Promise<EmbeddingResult> {
    const start = Date.now();
    const embeddings = await this.requestVectors(texts);
    const estTokens = texts.reduce((sum, t) => sum + Math.ceil(t.length / 4), 0);

    return {
      embeddings,
      model: this.model,
      usage: { prompt_tokens: estTokens, total_tokens: estTokens },
      latency_ms: Date.now() - start,
    };
  }

  async embedQuery(text: string): Promise<number[]> {
    const [vec] = await this.requestVectors([text]);
    if (!vec) throw new Error('Failed to generate query embedding via Cloudflare');
    return vec;
  }

  async embedQueries(texts: string[]): Promise<number[][]> {
    return this.requestVectors(texts);
  }
}
