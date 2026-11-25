# `@rag/database`

PostgreSQL 15+ (`pgvector`, `pg_trgm`, `btree_gin`) data layer using Drizzle ORM, read/write pool separation, and forward-only SQL migrations.

## Features

- **Read/Write Split & PgBouncer Ready (`src/client.ts`):** Exports `db` (primary writer) and `readDb` (`DATABASE_READ_URL` read replica, falling back to `db`). Automatically sets `prepare: false` for PgBouncer transaction pooling (`:6432` or `DATABASE_POOLER_MODE=transaction`) and probes `pg_extension.extversion >= 0.8` for `pgvector` iterative HNSW scans.
- **Repositories (`src/repositories/`):** Strictly tenant-isolated (`orgId`) & document-ACL filtered queries across chunks (`vector(1024)` two-stage CTE ordered by `<=> ASC`), Knowledge Graph (`kg_entities`, `kg_relations`, `kg_entity_aliases`), communities, documents, libraries, conversations, and `AgentToolsRepository` (allowlisted parameterized queries + persistent `audit_logs`).
- **Forward-Only SQL Migrations (`src/migrations/`):** Numbered `0000`–`0015` migrations serialized with `pg_advisory_lock`.

## Scripts

```bash
pnpm --filter @rag/database build          # Compile TypeScript & copy SQL migrations to dist/
pnpm --filter @rag/database db:migrate     # Apply pending migrations (or --baseline=<tag>)
pnpm --filter @rag/database db:new <name>  # Scaffold a new numbered SQL migration
pnpm --filter @rag/database typecheck
pnpm --filter @rag/database lint
pnpm --filter @rag/database test:unit      # Migration journal & SQL hygiene tests
```
