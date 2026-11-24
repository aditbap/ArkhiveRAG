import { EmbeddingProvider, EmbeddingResult, retryWithBackoff } from '@rag/core';

/**
 * OpenAI-compatible Embeddings Provider (works with OpenAI, Together, Fireworks,
 * Ollama `/v1/embeddings`, vLLM, LiteLLM, etc.).
 */
export class OpenAICompatibleEmbeddingProvider implements EmbeddingProvider {
  public readonly name = 'openai-compatible';
  public readonly model: string;
  public readonly dimensions: number;
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly timeoutMs: number;

  constructor(opts?: {
    baseUrl?: string;
    apiKey?: string;
    modelName?: string;
    dimensions?: number;
    timeoutMs?: number;
  }) {
    this.baseUrl = (opts?.baseUrl || process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1').replace(/\/+$/, '');
    this.apiKey = opts?.apiKey || process.env.OPENAI_API_KEY || '';
    this.model = opts?.modelName || process.env.EMBEDDING_MODEL || 'text-embedding-3-small';
    this.dimensions = opts?.dimensions ?? parseInt(process.env.EMBEDDING_DIMENSIONS || '1024', 10);
    this.timeoutMs = opts?.timeoutMs ?? parseInt(process.env.EMBEDDING_TIMEOUT_MS || '30000', 10);

    if (!this.apiKey) {
      throw new Error('OPENAI_API_KEY is required for OpenAICompatibleEmbeddingProvider');
    }
  }

  async embed(texts: string[]): Promise<EmbeddingResult> {
    if (texts.length === 0) {
      return {
        embeddings: [],
        model: this.model,
        usage: { prompt_tokens: 0, total_tokens: 0 },
        latency_ms: 0,
      };
    }

    const start = Date.now();
    const data = await retryWithBackoff(async () => {
      const response = await fetch(`${this.baseUrl}/embeddings`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: this.model,
          input: texts,
          ...(this.model.startsWith('text-embedding-3') ? { dimensions: this.dimensions } : {}),
        }),
        signal: AbortSignal.timeout(this.timeoutMs),
      });

      if (!response.ok) {
        const err = await response.text();
        throw new Error(`OpenAI-compatible embeddings error: ${response.status} ${err}`);
      }

      return (await response.json()) as {
        data: Array<{ embedding: number[] }>;
        usage?: { prompt_tokens?: number; total_tokens?: number };
      };
    });

    return {
      embeddings: data.data.map((d) => d.embedding),
      model: this.model,
      usage: {
        prompt_tokens: data.usage?.prompt_tokens ?? 0,
        total_tokens: data.usage?.total_tokens ?? 0,
      },
      latency_ms: Date.now() - start,
    };
  }

  async embedQuery(text: string): Promise<number[]> {
    const res = await this.embed([text]);
    if (!res.embeddings[0]) throw new Error('Failed to generate query embedding');
    return res.embeddings[0];
  }

  async embedQueries(texts: string[]): Promise<number[][]> {
    const res = await this.embed(texts);
    return res.embeddings;
  }
}
