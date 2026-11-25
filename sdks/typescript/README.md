# `@arkhiverag/sdk` (TypeScript / JavaScript SDK)

Zero-dependency TypeScript client for **ArkhiveRAG** using native `fetch`, `FormData`, and `ReadableStream` (works in Node.js 20+, Edge runtimes, and browsers).

## Quick Start

```ts
import { ArkhiveRAGClient } from '@arkhiverag/sdk';

const client = new ArkhiveRAGClient({
  baseUrl: 'http://localhost:3002',
  apiKey: process.env.ARKHIVERAG_API_KEY, // ak_live_... (or pass `token` for Keycloak JWT)
});

// 1. Adaptive RAG chat
const answer = await client.chat({
  mode: 'auto',
  agentic: true,
  messages: [{ role: 'user', content: 'Summarize our active reimbursement policy.' }],
});
console.log(answer.choices?.[0]?.message?.content);

// 2. Streaming SSE chat
for await (const event of client.chatStream({
  mode: 'mix',
  messages: [{ role: 'user', content: 'List Q3 security updates.' }],
})) {
  if (event.type === 'chunk' && event.content) process.stdout.write(event.content);
}

// 3. RBAC Company Agent
const agentRes = await client.agent.query({
  query: 'How many active documents and pending ingestion jobs do we have?',
});
console.log(agentRes.data.answer, agentRes.data.steps);
```
