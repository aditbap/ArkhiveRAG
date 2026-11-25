# `@rag/auth`

Keycloak OIDC JWT verification and multi-tenant Role-Based Access Control (RBAC) middleware for Fastify.

## Modules

- [`src/jwt-verify.ts`](src/jwt-verify.ts) — RS256 JWKS verification against Keycloak (`KEYCLOAK_URL` / `KEYCLOAK_REALM`) via `jose`
- [`src/middleware.ts`](src/middleware.ts) — `authenticate`, `requirePermissionMiddleware`, `requireLibraryPermission`, `requireUser` request guards and org membership resolution
- [`src/rbac.ts`](src/rbac.ts) — Role hierarchy (`owner > admin > member > viewer`) and fine-grained permission checks (`document:*`, `chat:*`, `settings:update`, etc.)

## Scripts

```bash
pnpm --filter @rag/auth build
pnpm --filter @rag/auth typecheck
pnpm --filter @rag/auth lint
pnpm --filter @rag/auth test:unit
```
