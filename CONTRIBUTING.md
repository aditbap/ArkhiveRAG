# Contributing

We welcome contributions across retrieval algorithms, database performance, connectors, observability, and documentation! Please review the [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

## Local Setup

```bash
nvm use                                   # Node 22 (.nvmrc)
pnpm install
docker compose -f infra/docker/docker-compose.yml up -d
cp .env.example .env                      # then fill in BYOK provider keys (never commit .env)
infra/keycloak/setup-realm.sh --dev-user  # local only
pnpm db:migrate && pnpm dev
```

## Before Opening a PR

`pnpm typecheck && pnpm lint && pnpm test:unit && pnpm build` must pass (CI runs the same checks on every PR and push to main).

If your change touches database schemas, SQL queries, caching, or concurrency:
- Ensure Postgres is running (`docker compose -f infra/docker/docker-compose.yml up -d postgres`).
- Run `pnpm ops:verify-plans` (`EXPLAIN (ANALYZE, BUFFERS)` HNSW + GIN index gate) and `pnpm ops:verify-resilience` (multi-tenant/ACL isolation, idempotency, and timeout checks).
- Integration tests (`pnpm --filter @rag/api test:integration`) require the local infrastructure stack and Keycloak running.

## Conventions

- **Database changes:** `pnpm db:new <name>` creates a numbered SQL migration in `packages/database/src/migrations/`; all migrations must be **forward-only and re-runnable** (`IF NOT EXISTS`).
- **Multi-tenancy:** Every repository query must enforce `orgId` (`requireTenant`) and document-level ACLs before ranking or caching.
- **Repo structure:** Keep operational scripts in `scripts/ops/`, documentation in `docs/`, and never commit `.env` or machine-specific paths.
