import type { LLMRole } from '@rag/core';
import type { LLMProvider } from './interfaces/llm.js';
import { CloudflareLLMProvider } from './cloudflare/llm.js';
import { GeminiLLMProvider } from './gemini/llm.js';
import { OpenAICompatibleLLMProvider } from './openai/llm.js';

export type LLMProviderKind = 'cloudflare' | 'gemini' | 'openai';

const ROLE_MODEL_ENV: Record<LLMRole, string> = {
  query: 'LLM_QUERY_MODEL',
  extract: 'LLM_EXTRACT_MODEL',
  keywords: 'LLM_KEYWORDS_MODEL',
  summary: 'LLM_SUMMARY_MODEL',
};

const FAST_FALLBACK_MODELS: Record<LLMProviderKind, string> = {
  cloudflare: '@cf/meta/llama-3.1-8b-instruct',
  gemini: 'gemini-2.5-flash',
  openai: 'gpt-4o-mini',
};

/**
 * Resolves the active LLM provider kind from `LLM_PROVIDER` or available credentials,
 * preserving Cloudflare as the default when Cloudflare credentials are configured.
 */
export function resolveLLMProviderKind(env: NodeJS.ProcessEnv = process.env): LLMProviderKind | null {
  const explicit = env.LLM_PROVIDER?.trim().toLowerCase();
  if (explicit === 'cloudflare' || explicit === 'gemini' || explicit === 'openai') {
    return explicit;
  }
  if (env.CLOUDFLARE_ACCOUNT_ID && env.CLOUDFLARE_API_TOKEN) return 'cloudflare';
  if (env.GEMINI_API_KEY) return 'gemini';
  if (env.OPENAI_API_KEY) return 'openai';
  return null;
}

export function hasLLMConfigured(env: NodeJS.ProcessEnv = process.env): boolean {
  return resolveLLMProviderKind(env) !== null;
}

/**
 * Decoupled Role-Based LLM Factory (inspired by LightRAG's `llm_roles.py`).
 *
 * Allows running a lightweight/fast model for high-frequency tasks (`extract`,
 * `keywords`, `summary`) and a higher-capacity model for final answer synthesis
 * (`query`), while remaining 100% backward-compatible with `LLM_PRIMARY_MODEL`.
 */
export function createLLMForRole(
  role: LLMRole,
  env: NodeJS.ProcessEnv = process.env,
): LLMProvider {
  const kind = resolveLLMProviderKind(env);
  if (!kind) {
    throw new Error(
      'No LLM provider credentials configured (set CLOUDFLARE_ACCOUNT_ID+CLOUDFLARE_API_TOKEN, GEMINI_API_KEY, or OPENAI_API_KEY)',
    );
  }

  const roleSpecificModel = env[ROLE_MODEL_ENV[role]]?.trim();
  const primaryModel = env.LLM_PRIMARY_MODEL?.trim();
  const model =
    roleSpecificModel ||
    (role === 'query' ? primaryModel : primaryModel || FAST_FALLBACK_MODELS[kind]) ||
    FAST_FALLBACK_MODELS[kind];

  switch (kind) {
    case 'gemini':
      return new GeminiLLMProvider(env.GEMINI_API_KEY, model);
    case 'openai':
      return new OpenAICompatibleLLMProvider({
        baseUrl: env.OPENAI_BASE_URL,
        apiKey: env.OPENAI_API_KEY,
        model,
      });
    case 'cloudflare':
    default:
      return new CloudflareLLMProvider(env.CLOUDFLARE_ACCOUNT_ID, env.CLOUDFLARE_API_TOKEN, model);
  }
}
