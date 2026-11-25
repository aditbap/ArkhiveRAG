# Dokploy Deployment Guide

Use [docker-compose.dokploy.yml](../infra/docker/docker-compose.dokploy.yml) when deploying onto a [Dokploy](https://dokploy.com/) host where Dokploy's Traefik reverse proxy (or a Cloudflare Tunnel sidecar) terminates TLS instead of the standalone `caddy` service in [docker-compose.deploy.yml](../infra/docker/docker-compose.deploy.yml).

For full sizing, BYOK configuration, PgBouncer pooling, and database tuning details, see [DEPLOYMENT.md](DEPLOYMENT.md) and [ROLLBACK.md](ROLLBACK.md).

## 1. Create the Compose App in Dokploy

- **Repository:** your fork or upstream repository (`main` or a release tag).
- **Compose Path:** `./infra/docker/docker-compose.dokploy.yml`
- **Build Context:** repository root (`.`).

## 2. Configure Environment Variables

Copy the contents of [.env.production.example](../.env.production.example) into the Dokploy **Environment** tab and fill in:

- Strong random secrets (`openssl rand -hex 24`) for `POSTGRES_PASSWORD`, `MINIO_SECRET_KEY`, `KEYCLOAK_ADMIN_PASSWORD`, `AUTH_SECRET`, and `GRAFANA_ADMIN_PASSWORD`.
- Public domains: `APP_DOMAIN`, `API_DOMAIN`, and `AUTH_DOMAIN`.
- Your BYOK `1024`-dim provider keys (`CLOUDFLARE_*`, `OPENAI_*`, `GEMINI_API_KEY`, and optional `COHERE_API_KEY` / `NVIDIA_API_KEY`).
- Optional `TUNNEL_TOKEN` if routing traffic through the `cloudflared` sidecar instead of Dokploy Traefik domains.

> **Important:** Full stack (`ocr` + `clamav` + `keycloak` + `postgres` `shm_size=1g`) requires at least `8 GB` host RAM.

## 3. Configure Routing Domains

- **Option A — Dokploy Traefik Domains:** Attach domains under **Domains** in Dokploy:
  - `APP_DOMAIN` -> `web` (`3000`)
  - `API_DOMAIN` -> `api` (`3002`)
  - `AUTH_DOMAIN` -> `keycloak` (`8080`)
  - *(Optional, restrict behind SSO/VPN)* `grafana` (`3000`)
- **Option B — Cloudflare Tunnel:** Point `APP_DOMAIN` -> `web:3000`, `API_DOMAIN` -> `api:3002`, and `AUTH_DOMAIN` -> `keycloak:8080` in your Cloudflare Zero Trust tunnel config and set `TUNNEL_TOKEN`.

## 4. Bootstrap Keycloak Realm

After the initial deploy brings `postgres` and `keycloak` up, run the realm bootstrap script on the Dokploy host to create the `rag-platform` realm and emit `AUTH_KEYCLOAK_SECRET`:

```bash
KEYCLOAK_CONTAINER=$(docker ps --filter "name=keycloak" -q | head -n1) \
KEYCLOAK_ADMIN_PASSWORD=<same as Dokploy env> \
WEB_URL=https://<APP_DOMAIN> \
KEYCLOAK_PUBLIC_URL=https://<AUTH_DOMAIN> \
infra/keycloak/setup-realm.sh
```

Copy the printed `AUTH_KEYCLOAK_SECRET` into the Dokploy Environment tab and redeploy `web`.

## 5. Updating & Rollback

- **Update:** Trigger a rebuild/deploy in Dokploy; the `migrate` init service applies forward-only migrations directly against `postgres:5432` before `api` and `worker` boot via `pgbouncer`.
- **Rollback (< 2 min):** Set `IMAGE_TAG=<previous_tag>` in the Environment tab and redeploy, or follow [ROLLBACK.md](ROLLBACK.md).
