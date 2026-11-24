import type { EmbeddingProvider } from '@rag/core';
import { CloudflareEmbeddingProvider } from './cloudflare/embedder.js';
import { NvidiaEmbeddingProvider } from './nvidia/embedder.js';
import { GeminiEmbeddingProvider } from './gemini/embedder.js';
import { OpenAICompatibleEmbeddingProvider } from './openai/embedder.js';

export type EmbeddingProviderKind = 'cloudflare' | 'nvidia' | 'gemini' | 'openai';

/**
 * Resolves the active embedding provider from `EMBEDDING_PROVIDER` or available
 * credentials. Prefers Cloudflare `@cf/baai/bge-m3` when `EMBEDDING_PROVIDER=cloudflare`
 * or when `EMBEDDING_MODEL` starts with `@cf/`.
 */
export function resolveEmbeddingProviderKind(env: NodeJS.ProcessEnv = process.env): EmbeddingProviderKind {
  const explicit = env.EMBEDDING_PROVIDER?.trim().toLowerCase();
  if (explicit === 'cloudflare' || explicit === 'nvidia' || explicit === 'gemini' || explicit === 'openai') {
    return explicit;
  }

  const model = env.EMBEDDING_MODEL?.trim() ?? '';
  if (model.startsWith('@cf/') && env.CLOUDFLARE_ACCOUNT_ID && env.CLOUDFLARE_API_TOKEN) {
    return 'cloudflare';
  }
  if (env.CLOUDFLARE_ACCOUNT_ID && env.CLOUDFLARE_API_TOKEN && !env.NVIDIA_API_KEY) {
    return 'cloudflare';
  }
  if (env.NVIDIA_API_KEY) return 'nvidia';
  if (env.CLOUDFLARE_ACCOUNT_ID && env.CLOUDFLARE_API_TOKEN) return 'cloudflare';
  if (env.GEMINI_API_KEY) return 'gemini';
  if (env.OPENAI_API_KEY) return 'openai';
  throw new Error(
    'No embedding provider configured (set EMBEDDING_PROVIDER and credentials for cloudflare, nvidia, gemini, or openai)',
  );
}

export function createEmbeddingProvider(env: NodeJS.ProcessEnv = process.env): EmbeddingProvider {
  const kind = resolveEmbeddingProviderKind(env);
  switch (kind) {
    case 'cloudflare':
      return new CloudflareEmbeddingProvider({
        accountId: env.CLOUDFLARE_ACCOUNT_ID,
        apiToken: env.CLOUDFLARE_API_TOKEN,
        modelName: env.EMBEDDING_MODEL?.startsWith('@cf/') ? env.EMBEDDING_MODEL : '@cf/baai/bge-m3',
      });
    case 'gemini':
      return new GeminiEmbeddingProvider(env.GEMINI_API_KEY);
    case 'openai':
      return new OpenAICompatibleEmbeddingProvider({
        baseUrl: env.OPENAI_BASE_URL,
        apiKey: env.OPENAI_API_KEY,
        modelName: env.EMBEDDING_MODEL,
      });
    case 'nvidia':
    default:
      return new NvidiaEmbeddingProvider(env.NVIDIA_API_KEY, env.EMBEDDING_MODEL);
  }
}
