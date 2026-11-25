# `@rag/core`

Pure, zero-I/O domain algorithms, types, interfaces, and policy logic shared across `apps/api`, `apps/worker`, and `@rag/database`.

## Modules

- [`src/agent/`](src/agent) — V3.1 RBAC Company Agent domain logic:
  - 9-tool role capability matrix (`owner`, `admin`, `member`, `viewer`)
  - Fail-closed `detectOutOfAuthorityRequest` + `authorizeToolCall` policy guard
  - Bounded (`AGENT_MAX_STEPS = 3`) deterministic planner (`router.ts`) and offline/online evaluation harness (`evaluator.ts`)
- [`src/retrieval/`](src/retrieval) — 12-feature Learning-to-Rank (LTR) fusion, CRAG confidence gate, temporal/authority conflict resolution, RAPTOR tree utilities, citation validation, history compression
- [`src/lightrag/`](src/lightrag) — Hub-proof Personalized PageRank (PPR), entity resolution, query/keyword routing, context assembly
- [`src/learning/`](src/learning) — 6-metric non-regression gate and adaptive parameter calibration
- [`src/utils/singleflight.ts`](src/utils/singleflight.ts) — Request coalescing for concurrent identical queries
- [`src/interfaces/`](src/interfaces) & [`src/types/`](src/types) — Provider contracts (`LLMProvider`, `EmbeddingProvider`, `RerankerProvider`, storage, OCR) and domain models

## Scripts

```bash
pnpm --filter @rag/core build
pnpm --filter @rag/core typecheck
pnpm --filter @rag/core lint
```
