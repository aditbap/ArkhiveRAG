# ArkhiveRAG (V3.1)

[![CI](https://github.com/aditbap/rag/actions/workflows/ci.yml/badge.svg)](https://github.com/aditbap/rag/actions/workflows/ci.yml)
[![Release](https://img.shields.io/badge/release-v3.1.0-blue.svg)](CHANGELOG.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Node.js](https://img.shields.io/badge/Node.js-22%2B-339933.svg)](.nvmrc)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15%2B%20pgvector-336791.svg)](docs/ARCHITECTURE.md)

Open-source, multi-tenant **Bring-Your-Own-Key (BYOK)** enterprise RAG platform built entirely on **PostgreSQL 15+ (`pgvector` HNSW + `pg_trgm` + tenant-scoped `btree_gin` FTS) + PgBouncer + Redis + MinIO + Keycloak** — no external graph database or separate search cluster required.

## Highlights

- **Verified #1 All-Rounder Retrieval Quality & Efficiency:**
  - In a controlled 5-way head-to-head benchmark ([BENCHMARK.md](BENCHMARK.md)) against `LlamaIndex`, `LightRAG`, `Haystack`, and `HippoRAG 2` over identical corpus, vectors, and KG triples: **`auto` + `RetrievalLearningEngine`** leads with **FullRec@5 `0.9444`**, **NDCG@5 `0.9482`**, **MRR `1.0000` (24/24)**, **FactCov `0.9792`** in **`569` avg tokens**; **`naive`** achieves **`2.101` Quality / 1k Tokens (#1 overall)** in **`447` avg tokens**.
- **PostgreSQL-Native Knowledge Graph & Adaptive Retrieval:**
  - `kg_entities`, `kg_relations`, and `kg_entity_aliases` with `pg_trgm` + exact-code Entity Resolution and O(1) `kg_indexed_hash` skip on re-index.
  - **Hub-Proof Adaptive Personalized PageRank (PPR)** over ACL-filtered 2-hop tenant subgraphs + Anchor-Guided PRGB + cosine synonymy edges.
  - **Hybrid Dense + Multilingual BM25** (`vector(1024)` HNSW ordered by raw `<=> ASC` + composite partial `btree_gin` `(org_id, content_tsv)`), **Parent-Child Chunking**, and **Proposition Multi-Vector MaxSim**.
  - **Temporal & Authority Conflict Resolution** so active/superseding policies automatically outrank superseded or expired editions, while historical *"before YYYY"* queries resolve accurately.
  - **Bounded CRAG Confidence Gate** (0 LLM-per-chunk grading), **12-Feature Learning-to-Rank (LTR)**, **ACL-Safe RAPTOR Community Summaries** (`<@` strict containment), and **`RetrievalLearningEngine`** (6-metric regression gate + instant rollback).
- **Proven `100,000`-Document Scale, High Concurrency & Resilience:**
  - **`7.93 ms` HNSW `p50`**, **`17.01 ms` composite GIN FTS `p50`**, **`14.74 ms` Hybrid `p50`** at `100,000` 1024-dim documents; **`165.1 req/s`** direct / **`249.6 req/s`** via PgBouncer (`:6432`) at `500` concurrent VUs (`0 / 970` errors).
  - **Crash-Safe & Chaos-Tested:** Fail-open Redis cache (`<= 511 ms` under outage), atomic single-transaction `replaceVersionChunks` (`0` duplicates on `SIGKILL`), `statement_timeout` protection, automated `EXPLAIN (ANALYZE, BUFFERS)` plan gate, and SHA-256 verified backup/restore drills.
- **Enterprise Governance, RBAC Agent & Observability:**
  - Keycloak OIDC + RBAC (`owner`/`admin`/`member`/`viewer`), strict SQL tenant & document ACL isolation, ClamAV virus scanning, PaddleOCR sidecar, Pino JSON logs with `X-Request-Id` / W3C `traceparent`, Prometheus alerts, pre-provisioned Grafana dashboard, and optional Langfuse tracing.
  - **V3.1 RBAC Company Agent (`/api/v1/agent` & `agentic: true` chat):** Bounded (`<= 3` hops) deterministic orchestrator deciding when to invoke `LightRagRetriever`, when to query allowlisted read-only PostgreSQL tools (`query_document_inventory`, `query_governance_audit`, `query_workspace_members`, `query_ingestion_jobs`, `query_security_audit_logs`), and when to fail-closed refuse unauthorized, cross-tenant, raw-SQL, or write probes with full `audit_logs` & Prometheus telemetry.

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
Browser ──► Caddy (HTTPS) ─┬─► web       Next.js 16 + Auth.js (Keycloak OIDC)
                           ├─► api:3002  Fastify: retrieval (Singleflight + HNSW + PPR), RBAC, SSE, learning engine
                           └─► keycloak  OIDC identity provider

api / worker ──► pgbouncer (tx mode) ──► postgres (pgvector HNSW + btree_gin + KG + RAPTOR) [+ optional read replica]
             ├──► redis (fail-open HTTP cache + BullMQ queue)
             └──► minio (S3) · clamav · paddleocr
prometheus (api:9100 + worker:9092 + alerts.yml) ──► grafana:3003
```

| Path | Purpose |
|---|---|
| [`apps/api`](apps/api/README.md) | HTTP API (Fastify), `LightRagRetriever`, V3.1 `CompanyAgentOrchestrator`, `RetrievalLearningEngine`, Prometheus `/metrics` |
| [`apps/worker`](apps/worker/README.md) | Crash-safe ingestion worker (BullMQ), parent-child chunking, KG + entity resolution, `:9092` health/metrics |
| [`apps/web`](apps/web/README.md) | Web UI (Next.js 16 + Auth.js) with adaptive retrieval mode selector & citations |
| [`packages/core`](packages/core/README.md) | Pure algorithms: PPR, entity/temporal resolution, LTR, CRAG, RAPTOR, `SingleflightGroup`, V3.1 RBAC `agent/` |
| [`packages/database`](packages/database/README.md) | Drizzle schemas, read/write split repositories, SQL migrations (`0000`–`0015`) + runner |
| [`packages/providers`](packages/providers/README.md) | BYOK LLM, embedding, reranker (Cloudflare/OpenAI/Gemini/NVIDIA/Cohere), MinIO, OCR, ClamAV adapters |
| [`packages/auth`](packages/auth/README.md) | RBAC (`owner`/`admin`/`member`/`viewer`) & Keycloak RS256 JWT verification |
| [`infra/`](infra/README.md) | Docker Compose stacks, PgBouncer, tuned `postgresql.conf`, Caddyfile, Prometheus alerts, Grafana dashboard |
| [`scripts/`](scripts/README.md) | Operational gates (`ops:*`), scale load benchmarks, SHA-256 backup/restore drills, and local dev seeding |
| [`docs/`](docs/README.md) | Architecture, OpenAPI 3.1 + API Reference, Deployment Guide, and Operations Runbook |
| [`examples/`](examples/README.md) | Quickstart CLI walkthrough for adaptive chat and V3.1 RBAC Company Agent queries |

## Quick Start (Local)

Requires Node 22, pnpm, and Docker. See [CONTRIBUTING.md](CONTRIBUTING.md) for contributor conventions.

```bash
pnpm install
docker compose -f infra/docker/docker-compose.yml up -d
cp .env.example .env           # Add your own LLM/Embedding API key(s)
infra/keycloak/setup-realm.sh --dev-user
pnpm db:migrate && pnpm dev
```

*(Optional local PgBouncer transaction pooling: add `--profile pooler` and point `DATABASE_URL` to port `6432`.)*

## Quality & Production Gates

```bash
pnpm typecheck && pnpm lint && pnpm test:unit && pnpm build

# Against running Postgres / PgBouncer:
pnpm ops:verify-plans       # Asserts EXPLAIN (ANALYZE, BUFFERS) uses HNSW + composite GIN (fails on Seq Scan)
pnpm ops:verify-resilience  # Verifies 40-query tenant/ACL isolation, idempotent chunk commits & statement_timeout
pnpm ops:eval-agent         # Runs live 24-case RBAC Company Agent & policy refusal evaluation gate
```

## Documentation & Community

- **[Architecture Overview](docs/ARCHITECTURE.md)** — Topology, adaptive retrieval pipeline, database tuning & security model
- **[API Reference](docs/API.md)** & **[OpenAPI 3.1 Spec](docs/openapi.yaml)** — Auth, RBAC, endpoints, and SSE/JSON schemas
- **[Deployment Guide](docs/DEPLOYMENT.md)** & **[Docker Compose + Caddy](infra/deploy/README.md)** / **[Dokploy](infra/deploy/DOKPLOY.md)** — Setup, PgBouncer, read replicas & sizing
- **[Operations Runbook](docs/OPERATIONS.md)** & **[Rollback Playbook](infra/deploy/ROLLBACK.md)** — SLOs, alert triage, HNSW maintenance & DR drills
- **[Benchmarks](BENCHMARK.md)** — 5-way head-to-head results & `10k → 100k`-document Postgres/PgBouncer telemetry
- **[Changelog](CHANGELOG.md)** · **[Contributing](CONTRIBUTING.md)** · **[Code of Conduct](CODE_OF_CONDUCT.md)** · **[Security Policy](SECURITY.md)** · **[License (MIT)](LICENSE)**
