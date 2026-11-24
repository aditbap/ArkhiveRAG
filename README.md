# RAG Platform (Corporate Adaptive RAG V2)

Open-source, multi-tenant **Bring-Your-Own-Key (BYOK)** enterprise RAG platform built entirely on **PostgreSQL 16 (`pgvector` + `pg_trgm` + multilingual FTS) + Redis + MinIO + Keycloak** — no external graph database or heavy search cluster required.

## Highlights

- **Adaptive Multi-Algorithm Retrieval Engine:**
  - **PostgreSQL-native Knowledge Graph** (`kg_entities`, `kg_relations`, `kg_entity_aliases`) with `pg_trgm` + exact-code Entity Resolution and O(1) document-hash skip on re-index.
  - **Graph-Aware Personalized PageRank (PPR)** over ACL-filtered 2-hop tenant subgraphs for multi-hop reasoning.
  - **Hybrid Dense + Multilingual BM25** (`vector(1024)` HNSW + stopword-filtered `'simple'`/`'english'` GIN), **Parent-Child Chunking**, and **Proposition Multi-Vector MaxSim**.
  - **Temporal & Authority Conflict Resolution** so active/superseding policies automatically outrank superseded or expired editions.
  - **Unified Query Decomposition**, **Adaptive Weighted Fusion**, **12-Feature Learning-to-Rank (LTR)**, and **Bounded CRAG Confidence Gate** (0 LLM-per-chunk grading, max 1 retry + grounded abstention).
  - **ACL-Safe RAPTOR Community Summaries & DRIFT Traversal** enforcing strict `<@` document-scope containment so restricted documents never leak across roles.
- **`RetrievalLearningEngine`:**
  - 3-plane design: `<5 ms` online intent routing with versioned `retrieval_configs` (Redis-cached), async retrieval-only shadow comparisons (`0` generation tokens), and offline L2-regularized weight optimization gated by a **6-metric Regression Gate** (NDCG, Recall, Faithfulness, p95 latency, `<= +5%` token budget, `0` ACL leaks) with instant rollback.
- **Enterprise Governance:** Keycloak OIDC + RBAC (`owner`/`admin`/`member`/`viewer`), strict tenant & document-level ACL isolation, ClamAV virus scanning, PaddleOCR sidecar, Langfuse + Prometheus + Grafana telemetry.

## Bring Your Own API Key (BYOK)

No keys are bundled. Copy `.env.example` to `.env` and configure **any** supported 1024-dim provider:

| Provider | Env Variables to Set | Default Models (1024-dim) |
|---|---|---|
| **Cloudflare Workers AI** *(all-in-one)* | `LLM_PROVIDER=cloudflare`, `EMBEDDING_PROVIDER=cloudflare`, `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN` | `@cf/meta/llama-3.1-8b-instruct`, `@cf/baai/bge-m3`, `@cf/baai/bge-reranker-base` |
| **OpenAI / Ollama / vLLM / OpenRouter** | `LLM_PROVIDER=openai`, `EMBEDDING_PROVIDER=openai`, `OPENAI_API_KEY` *(+ optional `OPENAI_BASE_URL`)* | `gpt-4o-mini` / custom, `text-embedding-3-small` (1024 dims) |
| **Google Gemini** | `LLM_PROVIDER=gemini`, `EMBEDDING_PROVIDER=gemini`, `GEMINI_API_KEY` | `gemini-3-flash-preview`, `gemini-embedding-001` (1024 dims) |
| **Cohere Reranker** *(optional)* | `RERANKER_PROVIDER=cohere`, `COHERE_API_KEY` | `rerank-v3.5` |

## Architecture

```
Browser ──► Caddy (HTTPS) ─┬─► web      Next.js 16 + Auth.js (Keycloak OIDC)
                           ├─► api      Fastify: libraries, documents, adaptive chat (SSE), learning engine, RBAC
                           └─► keycloak identity provider

api ──► postgres (pgvector HNSW + KG + PPR + RAPTOR + Learning) · redis (BullMQ) · minio (S3) · BYOK LLM/Embed/Rerank
worker ◄─ redis queue ─ ingestion: ClamAV → OCR → Parent-Child chunk → embed → KG + Entity Resolution → postgres
prometheus ◄─ api:9100/metrics ─► grafana
```

| Path | Purpose |
|---|---|
| `apps/api` | HTTP API (Fastify), `LightRagRetriever`, `RetrievalLearningEngine` |
| `apps/worker` | Document ingestion, parent-child chunking, KG & entity resolution worker (BullMQ) |
| `apps/web` | Web UI (Next.js 16) with adaptive retrieval mode selector |
| `packages/core` | Pure algorithms: PPR, entity resolution, temporal resolution, LTR, CRAG, RAPTOR, optimizer, regression gate |
| `packages/database` | Drizzle schemas, repositories, SQL migrations (`0000`–`0013`) + runner |
| `packages/providers` | BYOK LLM, embedding, reranker (Cloudflare/OpenAI/Gemini/NVIDIA/Cohere), MinIO, OCR, ClamAV adapters |
| `packages/auth` | Role-based access control & JWT verification |
| `infra/docker`, `infra/deploy` | Docker Compose, Dockerfiles, Caddyfile, Prometheus/Grafana, Dokploy guide |

## Quick start (local)

Requires Node 22, pnpm, and Docker. See [CONTRIBUTING.md](CONTRIBUTING.md) for full details.

```bash
pnpm install
docker compose -f infra/docker/docker-compose.yml up -d
cp .env.example .env           # Add your own LLM/Embedding API key(s)
infra/keycloak/setup-realm.sh --dev-user
pnpm db:migrate && pnpm dev
```

## Deploy

Single-server production stack: [Docker Compose + Caddy](infra/deploy/README.md) or [Dokploy](infra/deploy/DOKPLOY.md).
CI (`.github/workflows/ci.yml`) typechecks, lints, tests, builds, and scans Docker images on every PR.

## Quality checks

```bash
pnpm typecheck && pnpm lint && pnpm test:unit && pnpm build
```

## Benchmarks

See **[BENCHMARK.md](BENCHMARK.md)** for the full 5-way head-to-head benchmark against `LlamaIndex`, `LightRAG`, `Haystack`, and `HippoRAG 2`, plus live 10,000-document PostgreSQL scalability, security red-team, and token cost results.

Security policy: [SECURITY.md](SECURITY.md). License: [LICENSE](LICENSE).
