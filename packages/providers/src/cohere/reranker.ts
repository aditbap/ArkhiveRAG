import { CohereClient } from 'cohere-ai';
import type { RerankerProvider, ScoredChunk } from '@rag/core';

export class CohereRerankerProvider implements RerankerProvider {
  public readonly name = 'cohere';
  public readonly model: string;
  private cohere: CohereClient;
  private readonly maxRpm: number;
  private readonly callTimestamps: number[] = [];

  constructor(apiKey?: string, modelName?: string, maxRpm?: number) {
    const token = apiKey || process.env.COHERE_API_KEY;
    if (!token) {
      throw new Error('COHERE_API_KEY is required');
    }
    this.cohere = new CohereClient({ token });
    this.model = modelName || process.env.RERANKER_MODEL || 'rerank-v3.5';
    // Optional local RPM guard (e.g. COHERE_MAX_RPM=9 for trial keys) to avoid noisy 429 bursts.
    this.maxRpm = maxRpm ?? parseInt(process.env.COHERE_MAX_RPM || '0', 10);
  }

  private withinBudget(): boolean {
    if (this.maxRpm <= 0) return true;
    const cutoff = Date.now() - 60_000;
    while (this.callTimestamps.length > 0 && (this.callTimestamps[0] ?? 0) < cutoff) {
      this.callTimestamps.shift();
    }
    if (this.callTimestamps.length >= this.maxRpm) {
      return false;
    }
    this.callTimestamps.push(Date.now());
    return true;
  }

  async rerank(query: string, candidates: ScoredChunk[], topN: number = 5): Promise<ScoredChunk[]> {
    if (!candidates || candidates.length === 0) {
      return [];
    }

    if (!this.withinBudget()) {
      // Gracefully preserve caller's fused order instead of hammering Cohere with 429s.
      return candidates.slice(0, topN);
    }

    try {
      const response = await this.cohere.rerank({
        model: this.model,
        query,
        documents: candidates.map((c) => c.content),
        topN,
      });

      return response.results.flatMap((result) => {
        const candidate = candidates[result.index];
        return candidate ? [{ ...candidate, score: result.relevanceScore }] : [];
      });
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      if (msg.includes('429') || msg.toLowerCase().includes('toomanyrequests')) {
        console.warn('Cohere rerank rate-limited (429); falling back to fused ranking');
        return candidates.slice(0, topN);
      }
      console.error('Error reranking documents with Cohere:', error);
      throw error;
    }
  }
}
