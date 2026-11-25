# `@rag/api`

Fastify HTTP API for the Corporate Adaptive RAG Platform (`port 3002`, Prometheus metrics on `port 9100`).

## Responsibilities

- **Authenticated RAG API (`/api/v1`):**
  - `/documents` — Multipart uploads (MinIO + ClamAV + BullMQ ingestion), version history, ACLs, soft delete
  - `/chat/completions` — Adaptive retrieval (`naive`, `local`, `global`, `hybrid`, `mix`, `auto`), optional `agentic: true` routing, SSE streaming & JSON modes, cited responses
  - `/agent` — V3.1 RBAC-enforced Company Agent (`/tools`, `/query`, owner-only `/audit`)
  - `/libraries`, `/folders`, `/conversations` — Multi-tenant workspace & conversation management
  - `/retrieval` — Self-learning feedback & regression gate endpoints
- **Public Guest Chat (`/api/v1/guest/chat`):** Rate-limited plain LLM chat with no document/library access.
- **Health & Observability:** `/health`, `/ready`, and Prometheus `/metrics` with W3C `traceparent` + `X-Request-Id` propagation.

## Key Modules

- [`src/server.ts`](src/server.ts) — Fastify bootstrap, IP rate-limiting, hooks, route registration
- [`src/lib/lightrag-retriever.ts`](src/lib/lightrag-retriever.ts) — Hybrid dense/BM25, KG PPR traversal, temporal resolution, CRAG & LTR reranking
- [`src/lib/company-agent.ts`](src/lib/company-agent.ts) — Bounded (`<= 3` hops) RBAC agent orchestrator
- [`src/lib/learning-engine.ts`](src/lib/learning-engine.ts) — Policy calibration & 6-metric regression gate
- [`src/plugins/metrics.ts`](src/plugins/metrics.ts) — Prometheus counters & histograms

## Scripts

```bash
pnpm --filter @rag/api dev              # Start API in watch mode
pnpm --filter @rag/api build            # Compile TypeScript to dist/
pnpm --filter @rag/api typecheck        # Run tsc --noEmit
pnpm --filter @rag/api lint             # ESLint check
pnpm --filter @rag/api test:unit        # Fast unit test suite
pnpm --filter @rag/api test:integration # Requires Postgres, Redis, MinIO & Keycloak
pnpm --filter @rag/api requeue:stuck    # Re-enqueue stalled ingestion jobs
```
