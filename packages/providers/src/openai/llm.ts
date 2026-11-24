import { retryWithBackoff } from '@rag/core';
import { LLMProvider, LLMMessage, LLMResponse, LLMStreamChunk } from '../interfaces/llm.js';

/**
 * OpenAI-compatible Chat Completions provider (works with OpenAI, Groq, DeepSeek,
 * Together, OpenRouter, vLLM, Ollama, LiteLLM, etc.).
 */
export class OpenAICompatibleLLMProvider implements LLMProvider {
  public readonly name = 'openai-compatible';
  public readonly model: string;
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly timeoutMs: number;

  constructor(opts?: { baseUrl?: string; apiKey?: string; model?: string; timeoutMs?: number }) {
    this.baseUrl = (opts?.baseUrl || process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1').replace(/\/+$/, '');
    this.apiKey = opts?.apiKey || process.env.OPENAI_API_KEY || '';
    this.model = opts?.model || process.env.LLM_PRIMARY_MODEL || 'gpt-4o-mini';
    this.timeoutMs = opts?.timeoutMs ?? parseInt(process.env.LLM_TIMEOUT_MS || '60000', 10);

    if (!this.apiKey) {
      throw new Error('OPENAI_API_KEY is required for OpenAICompatibleLLMProvider');
    }
  }

  private get endpoint() {
    return `${this.baseUrl}/chat/completions`;
  }

  private get headers() {
    return {
      Authorization: `Bearer ${this.apiKey}`,
      'Content-Type': 'application/json',
    };
  }

  async generate(messages: LLMMessage[]): Promise<LLMResponse> {
    const start = Date.now();
    const data = await retryWithBackoff(async () => {
      const response = await fetch(this.endpoint, {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify({
          model: this.model,
          messages: messages.map((m) => ({ role: m.role, content: m.content })),
        }),
        signal: AbortSignal.timeout(this.timeoutMs),
      });

      if (!response.ok) {
        const err = await response.text();
        throw new Error(`OpenAI-compatible API error: ${response.status} ${err}`);
      }

      return (await response.json()) as {
        choices?: Array<{ message?: { content?: string } }>;
        usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number };
      };
    });

    return {
      content: data.choices?.[0]?.message?.content || '',
      model: this.model,
      usage: {
        prompt_tokens: data.usage?.prompt_tokens ?? 0,
        completion_tokens: data.usage?.completion_tokens ?? 0,
        total_tokens: data.usage?.total_tokens ?? 0,
      },
      latency_ms: Date.now() - start,
    };
  }

  async *generateStream(messages: LLMMessage[]): AsyncIterable<LLMStreamChunk> {
    const response = await retryWithBackoff(async () => {
      const res = await fetch(this.endpoint, {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify({
          model: this.model,
          messages: messages.map((m) => ({ role: m.role, content: m.content })),
          stream: true,
        }),
        signal: AbortSignal.timeout(this.timeoutMs),
      });

      if (!res.ok) {
        const err = await res.text();
        throw new Error(`OpenAI-compatible API error: ${res.status} ${err}`);
      }
      return res;
    });

    if (!response.body) {
      throw new Error('No response body from OpenAI-compatible API');
    }

    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    const reader = (response.body as ReadableStream<Uint8Array>).getReader();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed === 'data: [DONE]' || !trimmed.startsWith('data: ')) continue;
        try {
          const parsed = JSON.parse(trimmed.slice(6)) as {
            choices?: Array<{ delta?: { content?: string } }>;
            usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number };
          };
          const content = parsed.choices?.[0]?.delta?.content || '';
          if (content || parsed.usage) {
            yield {
              content,
              isFinal: false,
              model: this.model,
              usage: parsed.usage
                ? {
                    prompt_tokens: parsed.usage.prompt_tokens ?? 0,
                    completion_tokens: parsed.usage.completion_tokens ?? 0,
                    total_tokens: parsed.usage.total_tokens ?? 0,
                  }
                : undefined,
            };
          }
        } catch {
          /* ignore keepalive or partial SSE lines */
        }
      }
    }

    yield { content: '', isFinal: true, model: this.model };
  }
}
