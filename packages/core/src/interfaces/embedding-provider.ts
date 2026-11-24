export interface EmbeddingResult {
  embeddings: number[][];
  model: string;
  usage: { prompt_tokens: number; total_tokens: number };
  latency_ms: number;
}

export interface EmbeddingProvider {
  readonly name: string;
  readonly model: string;
  readonly dimensions: number;

  embed(texts: string[]): Promise<EmbeddingResult>;
  embedQuery(text: string): Promise<number[]>;
  /**
   * Batch-embeds multiple query strings in a single upstream request.
   * Inspired by LightRAG's single-batch pre-computation of [query, ll_keywords, hl_keywords]
   * to avoid 2-3 sequential embedding round-trips per chat turn.
   */
  embedQueries?(texts: string[]): Promise<number[][]>;
}
