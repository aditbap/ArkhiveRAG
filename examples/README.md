# Examples

Quick runnable walkthroughs for interacting with the **ArkhiveRAG** API and V3.1 RBAC Company Agent.

## Prerequisites

1. Start the local stack (`docker compose -f infra/docker/docker-compose.yml up -d`), run `pnpm db:migrate`, and start the API (`pnpm dev` or `pnpm --filter @rag/api dev`).
2. Obtain a valid Keycloak JWT access token for a seeded or logged-in user and export it:
   ```bash
   export RAG_TOKEN="<your-keycloak-jwt>"
   export API_URL="http://localhost:3002"   # optional, defaults to http://localhost:3002
   ```
3. *(Optional)* Pass `X_ORG_ID="<library-uuid>"` if targeting a specific shared library.

## Run

```bash
./examples/quickstart.sh
```

The script demonstrates:
1. Liveness probe (`GET /health`)
2. Workspace/library discovery (`GET /api/v1/libraries`)
3. Adaptive RAG chat (`POST /api/v1/chat/completions` with `mode: "auto"`, `agentic: true`)
4. RBAC Company Agent tool catalog (`GET /api/v1/agent/tools`)
5. Company Agent read-only DB query & out-of-authority fail-closed refusal (`POST /api/v1/agent/query`)
