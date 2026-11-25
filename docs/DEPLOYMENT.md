# Deployment Guide — Corporate Adaptive RAG

This guide covers single-server production deployment, PgBouncer pooling, PostgreSQL memory/SSD tuning, forward-only migrations, and capacity sizing.

- Quick Compose reference: [infra/deploy/README.md](../infra/deploy/README.md)
- Dokploy deployment: [infra/deploy/DOKPLOY.md](../infra/deploy/DOKPLOY.md)
- Rollback & DR playbook: [infra/deploy/ROLLBACK.md](../infra/deploy/ROLLBACK.md)

---

## 1. System Prerequisites

| Scale Tier | Corpus Size | Recommended Host RAM | `shared_buffers` | `effective_cache_size` | `shm_size` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Small / Starter** | `<= 10,000` docs (~`150 MB`) | `4–8 GB` | `512 MB` (default) | `1,536 MB` (default) | `1g` |
| **Medium** | `10,000–50,000` docs (~`750 MB`) | `8–16 GB` | `2 GB` | `6 GB` | `1g` |
| **Large / High Concurrency** | `50,000–100,000+` docs (~`1.5 GB`) | `16+ GB` + NVMe SSD | `4 GB` | `12 GB` | `1g–2g` |

> **Important:** Full stack (`OCR` `2 GB` + `ClamAV` `2 GB` + `Keycloak` `1 GB` + `Postgres` + `API`/`Worker`/`Web`) requires at least **`8 GB` host RAM** in production. `docker-compose.yml` and `docker-compose.deploy.yml` mount [infra/deploy/postgresql.conf](../infra/deploy/postgresql.conf) and configure `shm_size: 1g` so parallel `CREATE INDEX` / `REINDEX` operations with `maintenance_work_mem=512MB` never exhaust `/dev/shm`.

---

## 2. Environment Configuration (BYOK + Production Safety)

1. Copy the production template and generate strong random secrets (`openssl rand -hex 24`):
   ```bash
   cp .env.production.example .env.production
   ```
2. Set **`EMBEDDING_DIMENSIONS=1024`** matching the PostgreSQL `vector(1024)` HNSW schema:
   - **Cloudflare Workers AI** (`LLM_PROVIDER=cloudflare`, `EMBEDDING_PROVIDER=cloudflare`): `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`
   - **OpenAI / vLLM / Ollama / OpenRouter** (`LLM_PROVIDER=openai`, `EMBEDDING_PROVIDER=openai`): `OPENAI_API_KEY` *(+ optional `OPENAI_BASE_URL`)*
   - **Google Gemini** (`LLM_PROVIDER=gemini`, `EMBEDDING_PROVIDER=gemini`): `GEMINI_API_KEY`
3. Verify your Compose config resolves cleanly before booting:
   ```bash
   docker compose --env-file .env.production -f infra/docker/docker-compose.deploy.yml config --quiet
   ```

---

## 3. Initial Production Deploy (Docker Compose + Caddy)

```bash
DC="docker compose --env-file .env.production -f infra/docker/docker-compose.deploy.yml"

# 1. Start Postgres & Keycloak first so the OIDC realm and web client secret exist
$DC up -d postgres keycloak

KEYCLOAK_CONTAINER=$($DC ps -q keycloak) \
KEYCLOAK_ADMIN_PASSWORD=<same-as-env-production> \
WEB_URL=https://<APP_DOMAIN> \
KEYCLOAK_PUBLIC_URL=https://<AUTH_DOMAIN> \
ENV_FILE=.env.production \
infra/keycloak/setup-realm.sh

# 2. Run migrations, API, Worker, Web, PgBouncer, Caddy, Prometheus & Grafana
$DC up -d --build

# 3. Verify readiness & running services
$DC ps
curl -fsS https://<API_DOMAIN>/health
curl -fsS https://<API_DOMAIN>/ready
```

---

## 4. Database Pooling, PgBouncer & Read Replicas

- **PgBouncer (`transaction` mode):** Enabled by default in `docker-compose.deploy.yml` (`DATABASE_URL=postgresql://...@pgbouncer:5432/rag_platform`, `DATABASE_POOLER_MODE=transaction`). `packages/database/src/client.ts` automatically sets `prepare: false` whenever `DATABASE_POOLER_MODE=transaction` or port `:6432` is detected.
- **Direct Session Connection for Migrations:** The `migrate` service connects directly to `postgres:5432` (not PgBouncer) because PostgreSQL advisory locks (`pg_advisory_lock`) and DDL require session-level affinity.
- **Optional Read-Replica Offloading:** Set `DATABASE_READ_URL` (e.g., to a streaming replica or read-only PgBouncer pool) in `.env.production` to offload all vector/FTS/KG read queries from the primary writer pool with zero code changes.

---

## 5. Forward-Only Migrations & Baselining Existing Databases

Migrations live in `packages/database/src/migrations/` (`0000`–`0015`) and run via `@rag/database` (`node dist/migrate.js` in containers, or `pnpm db:migrate` locally).

If adopting the migration runner on an existing database created before `schema_migrations` existed, baseline up to the last applied tag once:

```bash
node packages/database/dist/migrate.js --baseline=0015_db_perf_index_hardening
```

All migrations are **forward-only, transactional, and additive** (`IF NOT EXISTS`), enabling zero-downtime `< 2 minute` application rollbacks without destructive schema drops.

---

## 6. Upgrades & Rollback

- **Upgrade:** Pin `IMAGE_TAG` in `.env.production` (or `git pull && $DC up -d --build`), then run `pnpm ops:verify-plans`.
- **Rollback:** Follow [infra/deploy/ROLLBACK.md](../infra/deploy/ROLLBACK.md) (`IMAGE_TAG=<prev_tag> $DC up -d --no-deps api worker web`).
