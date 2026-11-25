# Production Deploy (Single Server, Docker Compose)

> Using Dokploy? See [DOKPLOY.md](DOKPLOY.md). For full architecture, database tuning, and sizing details see [docs/DEPLOYMENT.md](../../docs/DEPLOYMENT.md) and [ROLLBACK.md](ROLLBACK.md).

Stack: Caddy (auto-HTTPS) → `web` / `api` / `keycloak`; `pgbouncer` (transaction pooling) → tuned `postgres` (`pgvector` + `infra/deploy/postgresql.conf`), `redis`, `minio`, `ocr`, `clamav`, `prometheus` (with `alerts.yml`), and `grafana` (SSH tunnel `127.0.0.1:3003` only) stay internal.

## Prerequisites
- Linux server with Docker + Compose v2, `>= 8 GB` RAM (OCR `2g`, ClamAV `2g`, Keycloak `1g`, Postgres `1g+` `shm_size=1g`), ports `80`/`443` open.
- Three DNS A records → server IP: `APP_DOMAIN`, `API_DOMAIN`, `AUTH_DOMAIN`.
- BYOK 1024-dim LLM/Embedding credentials (e.g., Cloudflare Workers AI, OpenAI/Ollama/vLLM, or Google Gemini).

## First Deploy
```bash
cp .env.production.example .env.production      # fill in secrets & BYOK API keys (openssl rand -hex 24)
DC="docker compose --env-file .env.production -f infra/docker/docker-compose.deploy.yml"

$DC up -d postgres keycloak                       # Keycloak needs its realm before web can log anyone in
KEYCLOAK_CONTAINER=$($DC ps -q keycloak) \
KEYCLOAK_ADMIN_PASSWORD=<same as .env.production> \
WEB_URL=https://<APP_DOMAIN> KEYCLOAK_PUBLIC_URL=https://<AUTH_DOMAIN> \
ENV_FILE=.env.production infra/keycloak/setup-realm.sh   # writes AUTH_KEYCLOAK_SECRET (+ AUTH_SECRET if empty)

$DC up -d --build                                 # runs forward-only migrations, then api/worker/web/caddy
$DC ps && curl -fsS https://<API_DOMAIN>/health
```
Create real users in the Keycloak admin console (`https://<AUTH_DOMAIN>`); do NOT use `--dev-user`.

## Update & Rollback
- **Update:** `git pull && $DC up -d --build` (forward-only migrations run automatically in `migrate`). Changing `API_DOMAIN` requires rebuilding `web`.
- **Rollback (< 2 min):** See [ROLLBACK.md](ROLLBACK.md) (`IMAGE_TAG=<prev_tag> $DC up -d --no-deps api worker web`).

## Backups & Restore Verification
Use the verified scripts in `scripts/ops/` (with SHA-256 checksums, distroless MinIO export, and row-count/index parity checks):
```bash
./scripts/ops/backup.sh ./backups
./scripts/ops/restore.sh ./backups/rag-backup-<TIMESTAMP> --target-db rag_restore_verify --drop-after-verify
```
