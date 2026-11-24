import { retryWithBackoff } from '@rag/core';
import { LLMProvider, LLMMessage, LLMResponse, LLMStreamChunk } from '../interfaces/llm.js';

export class CloudflareLLMProvider implements LLMProvider {
  public readonly name = 'cloudflare';
  public readonly model: string;
  private readonly accountId: string;
  private readonly apiToken: string;
  private readonly timeoutMs: number;
  private readonly maxTokens: number;

  constructor(accountId?: string, apiToken?: string, modelName?: string, timeoutMs?: number, maxTokens?: number) {
    this.accountId = accountId || process.env.CLOUDFLARE_ACCOUNT_ID || '';
    this.apiToken = apiToken || process.env.CLOUDFLARE_API_TOKEN || '';
    this.model = modelName || process.env.LLM_PRIMARY_MODEL || '@cf/meta/llama-3.1-8b-instruct';
    this.timeoutMs = timeoutMs ?? parseInt(process.env.LLM_TIMEOUT_MS || '60000', 10);
    // Cloudflare Workers AI defaults to 256 max_tokens when omitted, which truncates
    // JSON entity extraction and long RAG answers. Default to 2048.
    this.maxTokens = maxTokens ?? parseInt(process.env.LLM_MAX_TOKENS || '2048', 10);

    if (!this.accountId || !this.apiToken) {
      throw new Error('Cloudflare credentials (CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN) are required');
    }
  }

  private get url() {
    return `https://api.cloudflare.com/client/v4/accounts/${this.accountId}/ai/run/${this.model}`;
  }

  private get headers() {
    return {
      Authorization: `Bearer ${this.apiToken}`,
      'Content-Type': 'application/json',
    };
  }

  async generate(messages: LLMMessage[]): Promise<LLMResponse> {
    const start = Date.now();

    const data = await retryWithBackoff(async () => {
      const response = await fetch(this.url, {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify({
          messages: messages.map((m) => ({
            role: m.role,
            content: m.content,
          })),
          max_tokens: this.maxTokens,
        }),
        signal: AbortSignal.timeout(this.timeoutMs),
      });

      if (!response.ok) {
        const err = await response.text();
        throw new Error(`Cloudflare API error: ${response.status} ${err}`);
      }

      return (await response.json()) as {
        result?: {
          response?: unknown;
          usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number };
        };
      };
    });

    const result = data.result ?? {};
    const latency_ms = Date.now() - start;

    // Cloudflare Workers AI auto-deserializes JSON-shaped responses into plain objects;
    // serialize back to string so downstream parsers always receive text.
    const rawResponse = result.response;
    const content =
      typeof rawResponse === 'string'
        ? rawResponse
        : rawResponse != null
          ? JSON.stringify(rawResponse)
          : '';

    return {
      content,
      model: this.model,
      usage: {
        prompt_tokens: result.usage?.prompt_tokens ?? 0,
        completion_tokens: result.usage?.completion_tokens ?? 0,
        total_tokens: result.usage?.total_tokens ?? 0,
      },
      latency_ms,
    };
  }

  async *generateStream(messages: LLMMessage[]): AsyncIterable<LLMStreamChunk> {
    const response = await retryWithBackoff(async () => {
      const res = await fetch(this.url, {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify({
          messages: messages.map((m) => ({
            role: m.role,
            content: m.content,
          })),
          max_tokens: this.maxTokens,
          stream: true,
        }),
        signal: AbortSignal.timeout(this.timeoutMs),
      });

      if (!res.ok) {
        const err = await res.text();
        throw new Error(`Cloudflare API error: ${res.status} ${err}`);
      }
      return res;
    });

    if (!response.body) {
      throw new Error('No response body from Cloudflare API');
    }

    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    const parseLines = (lines: string[]): LLMStreamChunk[] => {
      const out: LLMStreamChunk[] = [];
      for (const line of lines) {
        if (line.trim() === '' || line.trim() === 'data: [DONE]') continue;
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6)) as { response?: string };
            out.push({
              content: data.response || '',
              isFinal: false,
              model: this.model,
            });
          } catch {
            /* ignore non-JSON SSE lines */
          }
        }
      }
      return out;
    };

    if ('getReader' in response.body) {
      const reader = (response.body as ReadableStream<Uint8Array>).getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const chunk of parseLines(lines)) yield chunk;
      }
    } else {
      for await (const chunk of response.body as unknown as AsyncIterable<Uint8Array>) {
        buffer += decoder.decode(chunk, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const item of parseLines(lines)) yield item;
      }
    }

    yield {
      content: '',
      isFinal: true,
      model: this.model,
    };
  }
}
