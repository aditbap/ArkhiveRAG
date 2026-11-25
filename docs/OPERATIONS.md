# Operations Runbook — ArkhiveRAG

This runbook defines measurable **Service Level Objectives (SLOs)**, alert triage steps for [infra/deploy/alerts.yml](../infra/deploy/alerts.yml), database maintenance, and verified backup/restore drills.

---

## 1. Service Level Objectives (SLOs)

Measured and verified against `100,000` 1024-dim documents on PostgreSQL 15.20 (`pgvector 0.8.7`) and PgBouncer `1.23.1`:

| Signal | Target SLO | V2.7 Verified Baseline (`100k` docs) |
| :--- | :---: | :---: |
| **API Availability (`/ready`)** | `>= 99.9%` | `0 / 970` errors across `10–500` VUs |
| **API 5xx Error Ratio** | `< 0.5%` (Alert at `> 2%`) | `0.00%` |
| **Dense HNSW Retrieval Latency** | `p95 <= 60 ms`, `p99 <= 100 ms` | `7.93 ms` p50 · `41.75 ms` p95 · `44.46 ms` p99 |
| **Composite GIN FTS Latency** | `p95 <= 80 ms`, `p99 <= 120 ms` | `17.01 ms` p50 · `23.80 ms` p95 · `26.52 ms` p99 |
| **Hybrid Dense+FTS Stage Latency** | `p95 <= 120 ms` (Alert at `> 500 ms` stage p95) | `14.74 ms` p50 · `26.09 ms` p95 · `26.16 ms` p99 |
| **Concurrent Query Burst (`50` VUs @ `100k` docs)** | `p95 <= 1.0 s`, `>= 40 req/s` | `414.8 ms` p50 · `468.7 ms` p95 · `105.6 req/s` |
| **Redis Outage Fail-Open Bound** | HTTP overhead `<= 600 ms`, `0` hung routes | `< 511 ms` fail-open, `5/5` HTTP requests succeed |

---

## 2. Observability Stack

- **Structured Logs & Correlation:** Fastify logs JSON via Pino (`LOG_LEVEL`) with `X-Request-Id` and W3C `traceparent` echoed on every response (`backend/api/src/server.ts`).
- **Metrics Endpoints:**
  - API internal port: `http://api:9100/metrics` (`METRICS_PORT=9100` in prod; `/metrics` in dev on `:3002`).
  - Worker internal port: `http://worker:9092/metrics` (plus `/health` and `/ready`).
- **Prometheus & Grafana:**
  - Production scrape + rules: [infra/deploy/prometheus.yml](../infra/deploy/prometheus.yml) + [infra/deploy/alerts.yml](../infra/deploy/alerts.yml).
  - Pre-provisioned Grafana dashboard: `ArkhiveRAG V3.0 Overview` at `http://127.0.0.1:3003` (reach over SSH tunnel `ssh -L 3003:127.0.0.1:3003 <server>`).
- **LLM & Tracing Extension Point:** Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` for token/span tracing in Langfuse. Notice W3C `traceparent` headers propagate through Fastify so an OpenTelemetry HTTP/OTLP collector sidecar can be attached without altering application APIs.

---

## 3. Alert Triage Runbook (`infra/deploy/alerts.yml`)

| Alert | Probable Cause | Immediate Triage |
| :--- | :--- | :--- |
| **`RagApiDown`** | API crashed, OOM, or `assertProductionConfig` failed on boot | `docker compose logs --tail=100 api` and inspect `GET /ready`. Verify `.env.production` secrets & Postgres/Redis health. |
| **`RagWorkerDown`** | Worker crashed or `:9092` health server unreachable | `docker compose logs --tail=100 worker` and check `curl http://worker:9092/ready`. |
| **`RagHighHttp5xxRate`** | Upstream DB error, unhandled route exception, or expired Keycloak keys | Filter API JSON logs by `res.statusCode >= 500` and correlate with `requestId` / `traceparent`. Check `GET /ready`. |
| **`RagRetrievalLatencySloBreach`** | HNSW index bypass, stale table stats, or cold buffer cache | Run `pnpm ops:verify-plans` to verify `Index Scan[idx_chunks_embedding_hnsw]`. Inspect `pg_stat_statements` top total_exec_time queries and run `ANALYZE document_chunks`. |
| **`RagDbPoolSaturation`** | Burst concurrency saturating Postgres backends or slow queries | Verify PgBouncer is healthy, check `pg_stat_activity` for `wait_event`, raise `PGBOUNCER_DEFAULT_POOL_SIZE` / `DATABASE_POOL_MAX`, or offload reads via `DATABASE_READ_URL`. |
| **`RagRedisDegradedMode`** | Redis down, restarting, or slow (`>500ms` cache timeout) | Inspect Redis container RAM/AOF persistence; API stays up fail-open until Redis recovers. |
| **`RagRerankerFallbackSpike`** | Cloudflare/Cohere reranker throttling or timeouts | Check `rag_reranker_fallbacks_total` reason labels (`timeout`, `rate_limit`); confirm upstream provider status. |
| **`RagIngestionQueueBacklog`** | Worker throughput < upload rate or OCR/embedding bottleneck | Check `worker_active_jobs` & OCR container health; scale worker replicas or `WORKER_CONCURRENCY`. |
| **`RagWorkerStalledOrFailingJobs`** | Worker `SIGKILL`/OOM mid-job or corrupt file failing retries | Inspect `ingestion_jobs` where `status = 'failed'`; note `replaceVersionChunks` guarantees zero duplicate chunks on retries. |
| **`RagAgentPolicyDenialSpike`** | Role boundary probing, prompt injection, or misconfigured UI role | Inspect `GET /api/v1/agent/audit` (owner only) or `audit_logs` where `action = 'agent:policy_deny'` correlated with `userId` and `requestId`. |

---

## 4. Database Verification & Maintenance

Run automated verification gates on live environments at any time (creates isolated orgs and cleans up automatically):

```bash
# Confirm EXPLAIN (ANALYZE, BUFFERS) uses HNSW + composite GIN (fails on Seq Scan)
pnpm ops:verify-plans

# Confirm 40-query tenant/ACL isolation, idempotent chunk replacement, and statement_timeout
pnpm ops:verify-resilience

# Run live 24-case RBAC Company Agent evaluation across all 4 roles & 6 categories
pnpm ops:eval-agent
```

### Compacting HNSW Tombstones After Bulk Purges
PostgreSQL's `pgvector` HNSW index does not immediately shrink on massive deletes (`> 30%` rows deleted in a tenant purge). After large bulk deletions, schedule an online zero-downtime index rebuild:

```sql
REINDEX INDEX CONCURRENTLY idx_chunks_embedding_hnsw;
ANALYZE document_chunks;
```

### Inspecting Live Slow Queries (`pg_stat_statements`)
```sql
SELECT calls, round(mean_exec_time::numeric, 2) AS mean_ms, round(max_exec_time::numeric, 2) AS max_ms, left(query, 100)
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 15;
```

---

## 5. Verified Backup, Restore Drill & Rollback

```bash
# 1. Create SHA-256 verified Postgres + MinIO backup
./scripts/ops/backup.sh ./backups

# 2. Non-destructive restore drill (verifies SHA-256, 100% row parity, and HNSW + GIN indexes)
./scripts/ops/restore.sh ./backups/rag-backup-<TIMESTAMP> --target-db rag_restore_verify --drop-after-verify
```

For fast `< 2 minute` application rollback or full disaster recovery, follow **[docs/ROLLBACK.md](ROLLBACK.md)**.
