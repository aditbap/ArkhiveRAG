# Scripts Index

Developer utilities, operational verification gates, and disaster-recovery scripts.

## Operational & Verification Gates (`scripts/ops/`)

Run via root `package.json` aliases or directly from the repo root:

| Script | Root Command | Purpose |
|---|---|---|
| [`ops/verify-query-plans.mts`](ops/verify-query-plans.mts) | `pnpm ops:verify-plans` | Runs `EXPLAIN (ANALYZE, BUFFERS)` against live Postgres to assert HNSW (`Index Scan`) and tenant-scoped composite `btree_gin` plans |
| [`ops/verify-resilience.mts`](ops/verify-resilience.mts) | `pnpm ops:verify-resilience` | Verifies 40-query tenant/ACL isolation, idempotent single-transaction chunk commits, and `statement_timeout` enforcement |
| [`ops/eval-agentic-rag.mts`](ops/eval-agentic-rag.mts) | `pnpm ops:eval-agent` | Live 24-case RBAC Company Agent & fail-closed policy refusal evaluation suite |
| [`ops/bench-scale-load.mts`](ops/bench-scale-load.mts) | — | Scale & high-concurrency load benchmark (`10k–100k` docs, direct vs PgBouncer `:6432`) |
| [`ops/backup.sh`](ops/backup.sh) | — | PostgreSQL + MinIO backup with SHA-256 checksums |
| [`ops/restore.sh`](ops/restore.sh) | — | Automated restore drill with row-count and HNSW/GIN index parity verification |

## Developer & QA Data (`scripts/dev/`, `scripts/`)

| File | Root Command | Purpose |
|---|---|---|
| [`dev/seed.ts`](dev/seed.ts) | `pnpm db:seed` | Seeds baseline `Test Org` and `Test User` rows into local PostgreSQL |
| [`golden-qa.example.json`](golden-qa.example.json) | — | Example Q&A evaluation fixture schema |
