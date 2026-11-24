import {
  AsyncSemaphore,
  buildCommunitySummaryPrompt,
  detectGraphCommunities,
  type CommunityEdgeInput,
  type CommunityNodeInput,
  type EmbeddingProvider,
} from '@rag/core';
import type { LLMProvider } from '../interfaces/llm.js';

export interface CommunityBuilderBridge {
  loadGraphForClustering(orgId: string): Promise<{
    entities: CommunityNodeInput[];
    relations: CommunityEdgeInput[];
  }>;
  getExistingCommunityHashes(orgId: string, level?: number): Promise<Map<string, string>>;
  upsertCommunities(
    orgId: string,
    items: Array<{
      level: number;
      clusterKey: string;
      clusterHash: string;
      title: string;
      summary: string;
      entityCanonicals: string[];
      documentIds: string[];
      embeddingModel: string;
      embedding: number[];
    }>,
  ): Promise<void>;
}

export interface CommunityBuildSummary {
  communitiesDetected: number;
  communitiesUpdated: number;
  communitiesSkippedUnchanged: number;
  llmCalls: number;
}

/**
 * Incremental RAPTOR / Graph Community Summary Builder (#8).
 * Uses `cluster_hash` dirty checking so unchanged clusters cost 0 LLM & 0 embedding tokens.
 */
export class CommunitySummaryBuilder {
  private readonly summaryLlm: LLMProvider;
  private readonly embedder: EmbeddingProvider;
  private readonly bridge: CommunityBuilderBridge;

  constructor(deps: {
    summaryLlm: LLMProvider;
    embedder: EmbeddingProvider;
    bridge: CommunityBuilderBridge;
  }) {
    this.summaryLlm = deps.summaryLlm;
    this.embedder = deps.embedder;
    this.bridge = deps.bridge;
  }

  async rebuildForOrg(orgId: string, options: { maxAsync?: number } = {}): Promise<CommunityBuildSummary> {
    const { entities, relations } = await this.bridge.loadGraphForClustering(orgId);
    const detected = detectGraphCommunities(entities, relations);
    if (detected.length === 0) {
      return { communitiesDetected: 0, communitiesUpdated: 0, communitiesSkippedUnchanged: 0, llmCalls: 0 };
    }

    const existingHashes = await this.bridge.getExistingCommunityHashes(orgId, 0);
    const dirty = detected.filter((c) => existingHashes.get(c.clusterKey) !== c.clusterHash);
    const skipped = detected.length - dirty.length;

    if (dirty.length === 0) {
      return {
        communitiesDetected: detected.length,
        communitiesUpdated: 0,
        communitiesSkippedUnchanged: skipped,
        llmCalls: 0,
      };
    }

    const semaphore = new AsyncSemaphore(options.maxAsync ?? 4);
    let llmCalls = 0;

    const summarized = await Promise.all(
      dirty.map((community) =>
        semaphore.run(async () => {
          let title = community.title;
          let summary = community.entityDescriptions.slice(0, 4).join(' ');
          try {
            const prompt = buildCommunitySummaryPrompt(community);
            const res = await this.summaryLlm.generate([{ role: 'user', content: prompt }]);
            llmCalls++;
            const raw = res.content;
            const s = raw.indexOf('{');
            const e = raw.lastIndexOf('}');
            if (s !== -1 && e > s) {
              const parsed = JSON.parse(raw.slice(s, e + 1)) as { title?: string; summary?: string };
              if (typeof parsed.title === 'string' && parsed.title.trim()) title = parsed.title.trim();
              if (typeof parsed.summary === 'string' && parsed.summary.trim()) summary = parsed.summary.trim();
            } else if (raw.trim()) {
              summary = raw.trim().slice(0, 1200);
            }
          } catch {
            /* graceful extractive fallback */
          }
          return { community, title, summary };
        }),
      ),
    );

    const embedTexts = summarized.map((s) => `${s.title}\n${s.summary}`);
    const embedRes = await this.embedder.embed(embedTexts);

    const toUpsert = summarized.flatMap((item, idx) => {
      const vec = embedRes.embeddings[idx];
      if (!vec) return [];
      return [
        {
          level: item.community.level,
          clusterKey: item.community.clusterKey,
          clusterHash: item.community.clusterHash,
          title: item.title,
          summary: item.summary,
          entityCanonicals: item.community.entityCanonicals,
          documentIds: item.community.documentIds,
          embeddingModel: this.embedder.model,
          embedding: vec,
        },
      ];
    });

    await this.bridge.upsertCommunities(orgId, toUpsert);

    return {
      communitiesDetected: detected.length,
      communitiesUpdated: toUpsert.length,
      communitiesSkippedUnchanged: skipped,
      llmCalls,
    };
  }
}
