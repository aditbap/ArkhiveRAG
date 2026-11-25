# Deployment Rollback & Disaster Recovery Playbook

This playbook defines safe, repeatable rollback procedures for **Corporate RAG** (`api`, `worker`, `web`, PostgreSQL/`pgvector`, Redis, and MinIO).

---

## 1. Golden Rule: Expand-and-Contract Database Migrations

All migrations in `packages/database/src/migrations/` follow **backward-compatible additive evolution** (`ADD COLUMN IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`, nullable or defaulted columns). Because the previous application version ignores newly added columns/indexes, **application rollback does NOT require destructive `DROP COLUMN` rollbacks**.

---

## 2. Pre-Deployment Safety Checklist

Before rolling out a new release tag (`vX.Y.Z`):

1. **Snapshot database & object storage:**
   ```bash
   ./scripts/ops/backup.sh ./backups
   ```
2. **Record the last known good Git SHA / container image tag:**
   ```bash
   export PREV_TAG="v2.5.0" # or previous immutable commit SHA
   ```
3. **Verify backup integrity (optional non-destructive drill):**
   ```bash
   ./scripts/ops/restore.sh ./backups/rag-backup-<TIMESTAMP> --target-db rag_restore_verify --drop-after-verify
   ```

---

## 3. Fast Application Rollback (< 2 Minutes, Zero Data Loss)

If error rate, `p95` latency, `/ready` probes, or retrieval quality regress after deploy:

1. **Pin services back to the previous image tag (or Git commit):**
   ```bash
   # Option A — Pinned Docker Compose images
   IMAGE_TAG="${PREV_TAG}" docker compose -f infra/docker/docker-compose.deploy.yml up -d --no-deps api worker web

   # Option B — Git source deployment
   git checkout "${PREV_TAG}"
   pnpm install --frozen-lockfile && pnpm build
   ```
2. **Wait for readiness probes on both API and Worker:**
   ```bash
   curl -fsS http://localhost:3001/ready
   curl -fsS http://localhost:9092/ready
   ```
3. **Flush stale response caches cleanly (bumps org cache version without wiping BullMQ queues):**
   - Because chat cache keys are versioned by `rag:cache-version:<orgId>` and `rag:acl-version:<orgId>`, operators can force fresh retrieval for any org via `INCR rag:cache-version:<orgId>` without restarting Redis.
4. **Post-Rollback Smoke Verification:**
   - `GET /health` → `200` with `X-Request-Id` & W3C `traceparent` headers
   - `GET /ready` → `200` (`postgres: ok`, `redis: ok`, `storage: ok`)
   - `GET :9092/metrics` → `worker_active_jobs` & `worker_uptime_seconds` updating
   - Execute 1 authenticated retrieval query and confirm `[Citation N]` grounding & `0` errors in logs.

---

## 4. Full Database & Storage Restore (Disaster Recovery)

Use only if data corruption or destructive accidental deletion occurred:

1. **Stop writers first** (`api` and `worker`) so no in-flight mutations race with restore:
   ```bash
   docker compose -f infra/docker/docker-compose.deploy.yml stop api worker
   ```
2. **Verify checksums and restore PostgreSQL:**
   ```bash
   ./scripts/ops/restore.sh ./backups/rag-backup-<TIMESTAMP> --target-db rag_platform
   ```
3. **Restore MinIO objects (if needed, distroless-compatible via `docker cp`):**
   ```bash
   mkdir -p /tmp/minio-restore && tar -xzf ./backups/rag-backup-<TIMESTAMP>/minio-data.tar.gz -C /tmp/minio-restore
   docker cp /tmp/minio-restore/. rag-minio:/data
   rm -rf /tmp/minio-restore
   ```
4. **Restart services and verify `/ready`:**
   ```bash
   docker compose -f infra/docker/docker-compose.deploy.yml start api worker
   curl -fsS http://localhost:3001/ready && curl -fsS http://localhost:9092/ready
   ```
