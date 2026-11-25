# syntax=docker/dockerfile:1
#
# Image for frontend (Next.js standalone output). Build from the repository root:
#   docker build -f frontend/Dockerfile \
#     --build-arg NEXT_PUBLIC_API_URL=https://api.example.com/api/v1 -t arkhiverag-web .
#
# NEXT_PUBLIC_API_URL is inlined into the browser bundle at build time, so changing the API
# domain means rebuilding this image. AUTH_* variables are read at runtime.

ARG NODE_VERSION=22
FROM node:${NODE_VERSION}-slim AS base
ENV PNPM_HOME=/pnpm PATH=/pnpm:$PATH
RUN corepack enable
WORKDIR /repo

FROM base AS build
ARG NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL} NEXT_TELEMETRY_DISABLED=1
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml turbo.json tsconfig.base.json ./
COPY frontend/package.json frontend/
COPY backend/api/package.json backend/api/
COPY backend/worker/package.json backend/worker/
COPY packages/core/package.json packages/core/
COPY packages/database/package.json packages/database/
COPY packages/providers/package.json packages/providers/
COPY packages/auth/package.json packages/auth/
RUN --mount=type=cache,id=pnpm-store,target=/pnpm/store \
    pnpm install --frozen-lockfile --filter "web..."
COPY frontend frontend
RUN pnpm --filter web run build

FROM node:${NODE_VERSION}-slim AS runtime
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 PORT=3000 HOSTNAME=0.0.0.0
WORKDIR /app
COPY --from=build --chown=node:node /repo/frontend/.next/standalone ./
COPY --from=build --chown=node:node /repo/frontend/.next/static ./frontend/.next/static
COPY --from=build --chown=node:node /repo/frontend/public ./frontend/public
USER node
EXPOSE 3000
CMD ["node", "frontend/server.js"]
