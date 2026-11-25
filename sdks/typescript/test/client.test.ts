import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { ArkhiveRAGClient, ArkhiveRAGError } from '../src/index.js';

describe('@arkhiverag/sdk ArkhiveRAGClient', () => {
  it('sends X-Api-Key and X-Org-Id headers and parses JSON responses', async () => {
    const calls: Array<{ url: string; headers: Record<string, string>; body?: string }> = [];
    const client = new ArkhiveRAGClient({
      baseUrl: 'http://localhost:3002/',
      apiKey: 'ak_live_001122334455_aabb',
      orgId: '00000000-0000-4000-8000-000000000001',
      fetch: async (input, init) => {
        calls.push({
          url: String(input),
          headers: (init?.headers ?? {}) as Record<string, string>,
          body: typeof init?.body === 'string' ? init.body : undefined,
        });
        return new Response(
          JSON.stringify({
            data: {
              decision: 'execute',
              role: 'member',
              refused: false,
              answer: '3 active docs',
              steps: [{ step: 1, tool: 'query_document_inventory', outcome: 'allow', summary: '3 docs' }],
              structuredData: [],
              citations: [],
              timingsMs: { total: 12 },
            },
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        );
      },
    });

    const res = await client.agent.query({ query: 'How many active docs?' });
    assert.equal(res.data.answer, '3 active docs');
    assert.equal(calls.length, 1);
    assert.equal(calls[0]?.url, 'http://localhost:3002/api/v1/agent/query');
    assert.equal(calls[0]?.headers['X-Api-Key'], 'ak_live_001122334455_aabb');
    assert.equal(calls[0]?.headers['X-Org-Id'], '00000000-0000-4000-8000-000000000001');
  });

  it('streams SSE events cleanly in chatStream()', async () => {
    const encoder = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode('data: {"type":"chunk","content":"Hello "}\n\n'));
        controller.enqueue(encoder.encode('data: {"type":"chunk","content":"ArkhiveRAG"}\n\n'));
        controller.enqueue(encoder.encode('data: [DONE]\n\n'));
        controller.close();
      },
    });

    const client = new ArkhiveRAGClient({
      baseUrl: 'http://localhost:3002',
      token: 'jwt-token',
      fetch: async () => new Response(stream, { status: 200 }),
    });

    const parts: string[] = [];
    for await (const ev of client.chatStream({
      mode: 'auto',
      messages: [{ role: 'user', content: 'Hi' }],
    })) {
      if (ev.type === 'chunk' && ev.content) parts.push(ev.content);
    }
    assert.equal(parts.join(''), 'Hello ArkhiveRAG');
  });

  it('throws structured ArkhiveRAGError on non-2xx responses', async () => {
    const client = new ArkhiveRAGClient({
      baseUrl: 'http://localhost:3002',
      apiKey: 'ak_live_invalid',
      fetch: async () =>
        new Response(
          JSON.stringify({
            error: { code: 'UNAUTHORIZED', message: 'Invalid API key', requestId: 'req-123' },
          }),
          { status: 401 },
        ),
    });

    await assert.rejects(
      () => client.documents.list(),
      (err: unknown) =>
        err instanceof ArkhiveRAGError &&
        err.status === 401 &&
        err.code === 'UNAUTHORIZED' &&
        err.requestId === 'req-123',
    );
  });
});
