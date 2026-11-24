import { EmbeddingProvider, EmbeddingResult, retryWithBackoff } from '@rag/core';

export class NvidiaEmbeddingProvider implements EmbeddingProvider {
  public readonly name = 'nvidia';
  public readonly model: string;
  public readonly dimensions: number;
  private readonly apiKey: string;
  private readonly timeoutMs: number;

  constructor(apiKey?: string, modelName?: string) {
    this.apiKey = apiKey || process.env.NVIDIA_API_KEY || '';
    let modelStr = modelName || process.env.EMBEDDING_MODEL || 'nvidia/nv-embedqa-e5-v5';
    if (!modelStr.includes('/')) {
      modelStr = `nvidia/${modelStr}`;
    }
    this.model = modelStr;
    this.dimensions = parseInt(process.env.EMBEDDING_DIMENSIONS || '1024', 10);
    this.timeoutMs = parseInt(process.env.EMBEDDING_TIMEOUT_MS || '30000', 10);

    if (!this.apiKey) {
      throw new Error('NVIDIA_API_KEY is required');
    }
  }

  private async requestEmbeddings(
    texts: string[],
    inputType: 'passage' | 'query',
  ): Promise<EmbeddingResult> {
    if (texts.length === 0) {
      return {
        embeddings: [],
        model: this.model,
        usage: { prompt_tokens: 0, total_tokens: 0 },
        latency_ms: 0,
      };
    }

    const start = Date.now();
    const payload = {
      input: texts,
      model: this.model,
      input_type: inputType,
      encoding_format: 'float',
      truncate: 'END',
    };

    // Log shape only - the payload contains full document text.
    console.debug(`[NVIDIA] embedding ${texts.length} text(s) as '${inputType}' with ${this.model}`);

    const data = await retryWithBackoff(async () => {
      let response: Response;
      try {
        response = await fetch('https://integrate.api.nvidia.com/v1/embeddings', {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${this.apiKey}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(payload),
          signal: AbortSignal.timeout(this.timeoutMs),
        });
      } catch (err) {
        const name = err instanceof Error ? err.name : undefined;
        if (name === 'TimeoutError' || name === 'AbortError') {
          throw new Error(
            `NVIDIA embeddings request timed out after ${this.timeoutMs}ms (model: ${this.model})`,
            { cause: err },
          );
        }
        throw err;
      }

      if (!response.ok) {
        const err = await response.text();
        throw new Error(`NVIDIA API error: ${response.status} ${err}`);
      }

      return (await response.json()) as {
        data: { embedding: number[] }[];
        usage?: { prompt_tokens?: number; total_tokens?: number };
      };
    });

    const embeddings = data.data.map((item) => item.embedding);
    const latency_ms = Date.now() - start;

    return {
      embeddings,
      model: this.model,
      usage: {
        prompt_tokens: data.usage?.prompt_tokens || 0,
        total_tokens: data.usage?.total_tokens || 0,
      },
      latency_ms,
    };
  }

  /** Embeds document passages (concurrency-safe: no shared mutable state). */
  async embed(texts: string[]): Promise<EmbeddingResult> {
    return this.requestEmbeddings(texts, 'passage');
  }

  /** Embeds a single query string without mutating instance state. */
  async embedQuery(text: string): Promise<number[]> {
    const result = await this.requestEmbeddings([text], 'query');
    if (!result.embeddings[0]) {
      throw new Error('Failed to generate embedding for query');
    }
    return result.embeddings[0];
  }

  /**
   * Batch-embeds multiple queries (e.g. [standaloneQuery, llKeywords, hlKeywords])
   * in ONE upstream API round-trip, matching LightRAG's batch query pre-computation.
   */
  async embedQueries(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    const result = await this.requestEmbeddings(texts, 'query');
    if (result.embeddings.length !== texts.length) {
      throw new Error(
        `NVIDIA embedQueries returned ${result.embeddings.length} vectors for ${texts.length} inputs`,
      );
    }
    return result.embeddings;
  }
}
