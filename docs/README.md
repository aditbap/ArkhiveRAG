# Documentation Index

Navigation hub for architecture, API contracts, deployment, and production operations.

## By Audience

- **Developers & Integrators**
  - [Architecture Overview](ARCHITECTURE.md) — Service topology, adaptive retrieval pipeline, database tuning, and security model
  - [API Reference](API.md) — Authentication, RBAC headers, endpoints, and SSE/JSON payload guides
  - [OpenAPI 3.1 Specification](openapi.yaml) — Machine-readable REST API schema
  - [Examples](../examples/README.md) — Runnable CLI walkthrough for chat, adaptive retrieval, and the V3.1 RBAC Company Agent
- **Operators & SREs**
  - [Deployment Guide](DEPLOYMENT.md) — Hardware sizing, BYOK setup, PgBouncer pooling, and forward-only migrations
  - [Docker Compose + Caddy Deploy](DOCKER_DEPLOY.md) — Single-server HTTPS deployment steps
  - [Dokploy Deployment Guide](DOKPLOY.md) — Deploying with Dokploy Traefik or Cloudflare Tunnel
  - [Operations Runbook](OPERATIONS.md) — SLOs, Prometheus alert triage, HNSW maintenance, and backup/restore drills
  - [Rollback Playbook](ROLLBACK.md) — Sub-2-minute application and DB rollback procedures
- **Quality, Benchmarks & Governance**
  - [5-Way Benchmarks & Scale Telemetry](BENCHMARK.md) — Head-to-head accuracy (LlamaIndex, LightRAG, Haystack, HippoRAG 2) and 100k-doc latency/load benchmarks
  - [Contributing Guide](../.github/CONTRIBUTING.md) · [Changelog](CHANGELOG.md) · [Security Policy](../.github/SECURITY.md) · [Code of Conduct](../.github/CODE_OF_CONDUCT.md)
