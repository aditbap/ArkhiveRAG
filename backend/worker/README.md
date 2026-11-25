# `@rag/worker`

Crash-safe BullMQ document ingestion and maintenance worker (`health & Prometheus metrics` on `port 9092`).

## Responsibilities

- **Document Ingestion Pipeline (`src/worker.ts`):**
  1. Downloads uploaded files from MinIO S3 storage (`@rag/providers`)
  2. Parses PDF, DOCX, HTML, or text (delegating scanned PDFs/images to the PaddleOCR sidecar)
  3. Builds hierarchical parent-child chunks and atomic propositions
  4. Extracts Knowledge Graph entities, typed relations, aliases, and community summaries (`O(1)` `kg_indexed_hash` skip when unchanged)
  5. Persists chunks + vectors (`vector(1024)`) atomically inside a single `replaceVersionChunks` database transaction (`0` partial chunks on crash/`SIGKILL`)
- **Maintenance & Health (`src/maintenance.ts`, `src/health-server.ts`):** Exposes `/health`, `/ready`, and `/metrics` for Prometheus scraping and Docker healthchecks.

## Scripts

```bash
pnpm --filter @rag/worker dev       # Start worker in watch mode
pnpm --filter @rag/worker build     # Compile TypeScript to dist/
pnpm --filter @rag/worker start     # Run compiled worker
pnpm --filter @rag/worker typecheck # Run tsc --noEmit
pnpm --filter @rag/worker lint      # ESLint check
```
