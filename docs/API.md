# API Reference — Corporate Adaptive RAG

Machine-readable OpenAPI 3.1.0 specification: **[openapi.yaml](openapi.yaml)**.

---

## 1. Base URLs, Auth & Correlation Headers

- **Default local API URL:** `http://localhost:3002`
- **Authentication:** Every `/api/v1/*` route except `/api/v1/guest/chat` requires a Keycloak OIDC `Authorization: Bearer <jwt>` token.
- **Organization / Library Selection:** Set `X-Organization-Id: <uuid>` (or query within the user's active organization claim/personal workspace). All membership permissions are resolved against the target organization.
- **Distributed Correlation Headers:** Send `X-Request-Id` and/or W3C `traceparent`; the API validates/propagates both on every response and propagates `traceId` into BullMQ ingestion jobs.
- **Standard Error Envelope:**
  ```json
  {
    "error": {
      "code": "BAD_REQUEST",
      "message": "Human-readable explanation",
      "requestId": "optional-correlation-id"
    }
  }
  ```

---

## 2. RBAC Permission Matrix (`packages/auth/src/rbac.ts`)

| Action | `owner` | `admin` | `member` | `viewer` |
| :--- | :---: | :---: | :---: | :---: |
| `documents:read`, `chat:create`, `chat:read`, `members:read` | Yes | Yes | Yes | Yes |
| `documents:create` | Yes | Yes | Yes | No |
| `documents:update` / `documents:delete` | Any | Any | Own uploads only | No |
| `folders:manage`, `members:manage`, `library:update`, `settings:read` | Yes | Yes | No | No |
| `settings:update`, `library:delete` | Yes | No | No | No |

Additionally, `includeDrafts: true` in chat retrieval is restricted strictly to `owner` and `admin`.

---

## 3. Operational Probes & Metrics

| Method & Path | Port | Auth | Description |
| :--- | :---: | :---: | :--- |
| `GET /health` | `3002` (API), `9092` (Worker) | None | Process liveness (`200 ok`). |
| `GET /ready` | `3002` (API), `9092` (Worker) | None | Live checks for `postgres`, `redis`, and `storage` (`200` ready / `503` unavailable). |
| `GET /metrics` | `9100` (API prod), `3002` (API dev), `9092` (Worker) | Internal network | Prometheus exposition endpoint. |

---

## 4. Authenticated API Endpoints (`/api/v1`)

### Chat & Conversations
- `POST /api/v1/guest/chat` — Public guest-only SSE stream (no login, no document access; strictly rate-limited per IP/minute, per IP/day, and global/day; returns `403` when `GUEST_CHAT_ENABLED=false`).
- `POST /api/v1/chat/completions` (`chat:create`) — Adaptive RAG completion returning JSON or Server-Sent Events (`stream: true`). Supports `mode` (`auto` | `mix` | `hybrid` | `local` | `global` | `naive`), `conversationId`, `documentIds` ACL scope, `includeDrafts` (`owner`/`admin` only), and `agentic` (`boolean`, defaults to `AGENT_DEFAULT_ENABLED=false`) to enable RBAC-gated structured DB + document routing.
- `GET /api/v1/conversations` (`chat:read`) — List active (non-archived) conversations for current user & org.
- `GET /api/v1/conversations/:id/messages` (`chat:read`) — Fetch conversation messages with hydrated citation chips.
- `DELETE /api/v1/conversations/:id` (`chat:create`) — Delete user's conversation.

### RBAC Company Agent (`/api/v1/agent`)
- `GET /api/v1/agent/tools` (`chat:read`) — Lists tools authorized and restricted for the authenticated caller's role (`owner` | `admin` | `member` | `viewer`).
- `POST /api/v1/agent/query` (`chat:create`) — Executes the bounded (`<= 3` steps) RBAC agent across document search and allowlisted read-only DB tools (`query_document_inventory`, `query_governance_audit`, `query_workspace_members`, `query_ingestion_jobs`, `query_security_audit_logs`). Returns `{ decision, refused, refusalCode, steps, structuredData, citations, answer }`.
- `GET /api/v1/agent/audit` (`settings:update`, `owner` only) — Inspects recent `agent:tool_allow` and `agent:policy_deny` security audit log entries.

### Documents & Folders
- `GET /api/v1/documents` (`documents:read`) — Paginated document list (`limit` 1..200, `offset`, `folderId`).
- `POST /api/v1/documents/upload` (`documents:create`) — Multipart file upload; stores in MinIO, creates version 1, enqueues BullMQ ingestion job (`201`). Rolls back storage and returns `503` if queue/storage is unavailable.
- `POST /api/v1/documents/:id/versions` (`documents:update` / `:own`) — Upload replacement file of same file family; creates new version and atomically replaces active chunks upon completion.
- `GET /api/v1/documents/:id/versions` (`documents:read`) — List all versions of a document.
- `PATCH /api/v1/documents/:id` (`documents:update` / `:own`) — Rename document display name.
- `PATCH /api/v1/documents/:id/move` (`documents:update` / `:own`) — Move document into a folder (or `null` for root).
- `PATCH /api/v1/documents/:id/notes` (`documents:update` / `:own`) — Update document notes.
- `PATCH /api/v1/documents/:id/governance` (`documents:update`) — Transition governance lifecycle (`draft` | `approved` | `active` | `superseded` | `archived`), set `effectiveAt`, `expiresAt`, `supersedesDocumentId`, `authorityTier` (`1..4`), and flush org retrieval cache.
- `DELETE /api/v1/documents/:id` (`documents:delete` / `:own`) — Soft-delete document, prune references from Knowledge Graph, and bump cache version.
- `GET /api/v1/documents/:id/content` (`documents:read`) — Stream original file with `X-Content-Type-Options: nosniff` and sandbox CSP headers.
- `GET /api/v1/folders`, `POST /api/v1/folders`, `PATCH /api/v1/folders/:id`, `DELETE /api/v1/folders/:id` — Manage folder hierarchy.

### Shared Libraries, Membership & Invites
- `GET /api/v1/libraries` — List libraries current user belongs to (`personal` & `shared`).
- `POST /api/v1/libraries` — Create shared library (`owner`).
- `GET /api/v1/libraries/join/:code` — Preview invite metadata & usability.
- `POST /api/v1/libraries/join` — Redeem invite code to join library.
- `GET /api/v1/libraries/:id`, `PATCH /api/v1/libraries/:id`, `DELETE /api/v1/libraries/:id` — Read, rename, or archive shared library (`DELETE` bumps ACL cache version).
- `GET /api/v1/libraries/:id/members`, `PATCH /api/v1/libraries/:id/members/:userId`, `DELETE /api/v1/libraries/:id/members/:userId` — View/update roles (`owner`/`admin`/`member`/`viewer`) or remove/leave member (bumps ACL cache version).
- `GET /api/v1/libraries/:id/invites`, `POST /api/v1/libraries/:id/invites`, `DELETE /api/v1/libraries/:id/invites/:inviteId` — Create/list/revoke library invite codes.

### `RetrievalLearningEngine` & RAPTOR Communities (`/api/v1/retrieval`)
- `POST /api/v1/retrieval/feedback` (`chat:create`) — Record thumbs up/down (`1` or `-1`), `telemetryId`, `messageId`, `citedChunkIds`, `reasonCode`.
- `GET /api/v1/retrieval/metrics?days=7` (`chat:read`) — Retrieval quality, latency, intent mix, token usage, CRAG verdicts, and recent experiments (`days` 1..90).
- `GET /api/v1/retrieval/configs` (`chat:read`) — Active config version and immutable history for this org.
- `POST /api/v1/retrieval/optimize` (`documents:delete`) — Run bounded weight optimization + 6-metric regression gate over `LearningEvalCase[]`.
- `POST /api/v1/retrieval/configs/:id/promote` (`documents:delete`) — Promote or roll back to a specific immutable retrieval config UUID.
- `POST /api/v1/retrieval/communities/rebuild` (`documents:delete`) — Incrementally rebuild ACL-safe RAPTOR community summaries for the org.
