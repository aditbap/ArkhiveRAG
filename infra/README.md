# Infrastructure Index

Deployment manifests, reverse-proxy configs, database tuning, observability dashboards, and sidecar Docker builds.

## Directory Layout

| Path | Contents |
|---|---|
| [`docker/`](docker) | Compose stacks: `docker-compose.yml` (local dev + optional `--profile pooler` PgBouncer), `docker-compose.deploy.yml` (standalone Compose + Caddy + PgBouncer + Prometheus/Grafana), `docker-compose.dokploy.yml` (Dokploy variant), Grafana provisioning *(service Dockerfiles live in `frontend/`, `backend/api/`, `backend/worker/`)* |
| [`deploy/`](deploy) | Production runtime configs: `postgresql.conf`, `postgres-init.sql`, `Caddyfile`, `prometheus.yml`, `alerts.yml` *(deployment/rollback docs live in [`../docs/`](../docs/README.md))* |
| [`keycloak/`](keycloak) | Custom Keycloak 26.0 image + theme and [`setup-realm.sh`](keycloak/setup-realm.sh) bootstrap script ([keycloak/README.md](keycloak/README.md)) |
| [`ocr/`](ocr) | Python PaddleOCR HTTP sidecar (`Dockerfile`, `app.py`, `requirements.txt`) for scanned PDFs and images |

For full sizing and bootstrap instructions see [docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md).
