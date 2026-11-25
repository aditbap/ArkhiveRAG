# syntax=docker/dockerfile:1
#
# Image for backend/api. Build from the repository root:
#   docker build -f backend/api/Dockerfile -t arkhiverag-api .
#
# Stages: install + build workspace deps, `pnpm deploy` production dependencies into /out,
# then copy that into a slim non-root runtime image.

ARG NODE_VERSION=22
FROM node:${NODE_VERSION}-slim AS base
ENV PNPM_HOME=/pnpm PATH=/pnpm:$PATH
RUN corepack enable
WORKDIR /repo

FROM base AS build
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml turbo.json tsconfig.base.json ./
COPY frontend/package.json frontend/
COPY backend/api/package.json backend/api/
COPY backend/worker/package.json backend/worker/
COPY packages/core/package.json packages/core/
COPY packages/database/package.json packages/database/
COPY packages/providers/package.json packages/providers/
COPY packages/auth/package.json packages/auth/
RUN --mount=type=cache,id=pnpm-store,target=/pnpm/store \
    pnpm install --frozen-lockfile --filter "@rag/api..."
COPY backend/api backend/api
COPY packages packages
RUN pnpm --filter "@rag/api..." run build
RUN pnpm --filter "@rag/api" deploy --prod --legacy /out

FROM node:${NODE_VERSION}-slim AS runtime
ENV NODE_ENV=production
WORKDIR /app
COPY --from=build --chown=node:node /out /app
USER node
CMD ["node", "dist/index.js"]
