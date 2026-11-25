# Changelog

All notable changes to **Corporate Adaptive RAG** are documented in this file.

## [v3.0.0] - 2026-11-25

### Added
- **Observability & Alerting:** Added 9 Prometheus SLO/health alert rules in `infra/deploy/alerts.yml` (verified via `promtool`), `rag-worker` (`:9092/metrics`) scraping in `infra/deploy/prometheus.yml`, fixed local dev API scrape port (`:3002`), and pre-provisioned Grafana dashboard `Corporate RAG V3.0 Overview`.
- **Operations & CI Gates:** Added repeatable `scripts/ops/verify-resilience.mts` (`pnpm ops:verify-resilience`) and `pnpm ops:verify-plans` (`scripts/ops/verify-query-plans.mts`), and wired both into `.github/workflows/integration.yml`.
- **Complete Documentation Suite:** Added `docs/ARCHITECTURE.md`, `docs/API.md`, `docs/openapi.yaml` (OpenAPI 3.1.0), `docs/DEPLOYMENT.md`, and `docs/OPERATIONS.md`.
- **Open-Source Community Files:** Added `CHANGELOG.md`, `CODE_OF_CONDUCT.md`, `.github/PULL_REQUEST_TEMPLATE.md`, and GitHub issue templates.

### Fixed
- Aligned `infra/docker/docker-compose.deploy.yml` and `.env.production.example` to `vector(1024)` (`EMBEDDING_DIMENSIONS=1024`, `@cf/baai/bge-m3`), added reproducible `IMAGE_TAG` pinning, added worker HTTP `/health` check (`:9092`), and fixed `:3001 → :3002` port references in `infra/deploy/ROLLBACK.md`.

## [v2.7.0] - 2026-11-25 — Database Performance & Concurrency

### Changed & Optimized
- **Restored True `O(log N)` pgvector HNSW Scans:** Rewrote `ChunkRepository`, `KnowledgeGraphRepository`, and `CommunityRepository` vector searches to order by raw `embedding <=> $vec ASC` inside a two-stage filtered CTE with configurable `hnsw.ef_search` and automatic `hnsw.iterative_scan` probing (`pgvector >= 0.8`).
- **Eliminated ~85% Embedding Wire Over-Fetch:** Introduced `CHUNK_RETRIEVAL_COLUMNS` and `DOCUMENT_RETRIEVAL_COLUMNS` projections omitting heavy `embedding` and `content_tsv` columns on retrieval queries.
- **Migration `0015_db_perf_index_hardening.sql`:** Dropped 2 redundant GIN FTS indexes, consolidated onto tenant-scoped composite partial `btree_gin` `(org_id, content_tsv)`, tightened partial HNSW predicates (`chunk_role <> 'parent'`), and bounded `ts_rank_cd()` candidate scoring.
- **Tuned PostgreSQL + PgBouncer + Read Replica Support:** Added `infra/deploy/postgresql.conf` (`shm_size: 1g`), PgBouncer transaction pooling (auto `prepare: false`), optional `DATABASE_READ_URL` read-replica pool, 30s `hasGraphData` cache, and tenant/ACL-isolated `SingleflightGroup` query coalescing.
- **Verified `100k`-Doc Gains:** HNSW `p50` `356.4 ms → 7.93 ms` (**44.9× faster**), FTS `p50` `263.1 ms → 17.01 ms` (**15.5× faster**), Hybrid `p50` `544.5 ms → 14.74 ms` (**36.9× faster**), and `500` VU throughput `2.4 → 165.1 req/s` (`249.6 req/s` via PgBouncer `:6432`) with `0.0000` V2.5 retrieval regression.

## [v2.6.0] - 2026-11-25 — Production Scalability & Resilience

- Split fast fail-open HTTP cache Redis (`commandTimeout: 500ms`, `enableOfflineQueue: false`) from BullMQ blocking queue connection with exponential retries (`attempts: 3`).
- Added single-transaction `ChunkRepository.replaceVersionChunks` (`0` duplicate chunks across worker `SIGKILL` crashes), startup stale-job reconciler, and `statement_timeout` bounds.
- Added `X-Request-Id` + W3C `traceparent` propagation, `rag_db_pool_connections`, `rag_redis_degraded_total`, zero-dependency worker `:9092` observability server, `scripts/ops/backup.sh`, `scripts/ops/restore.sh`, and `infra/deploy/ROLLBACK.md`.
- Passed all 5 live Docker chaos & multi-tenant security drills plus `10k → 100k` load stress (`0 / 970` errors).

## [v2.5.0] - 2026-11-24 — Retrieval Accuracy & Token Optimization

- Delivered Query-Adaptive Dense Calibration, Hub-Proof Adaptive Personalized PageRank, Point-in-Time Temporal & Authority Resolution, Anchor-Guided PRGB, ACL-Safe RAPTOR Communities, Bounded CRAG Gate, `RetrievalLearningEngine` (6-metric regression gate), Grounded KG Compression, and Score-Ratio Tail Pruning.
- Swept #1 overall in the controlled 5-way RAG benchmark (`0.9482` NDCG@5, `0.9444` FullRec@5, `1.0000` MRR, `2.101` Quality / 1k Tokens in `naive` mode).
