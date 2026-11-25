# `@rag/providers`

Bring-Your-Own-Key (BYOK) infrastructure and model provider adapters implementing `@rag/core` interfaces.

## Adapters

- **LLM (`src/llm-roles.ts`):** Role-based model routing (`query`, `keywords`, `extraction`, `community`) across Cloudflare Workers AI (`src/cloudflare/`), OpenAI / vLLM / Ollama / OpenRouter (`src/openai/`), and Google Gemini (`src/gemini/`).
- **Embeddings (`src/embedding-factory.ts`):** 1024-dim vector adapters for Cloudflare Workers AI, OpenAI-compatible endpoints, Google Gemini, and NVIDIA NIM (`src/nvidia/`).
- **Rerankers (`src/reranker-factory.ts`):** Cloudflare BGE reranker, Cohere (`src/cohere/`), or automatic fallback.
- **Document Processing & Security:**
  - Parsers (`src/parsers/`): PDF, DOCX, HTML, plain text
  - OCR (`src/ocr/`): PaddleOCR sidecar client
  - Chunkers (`src/chunkers/`): Recursive / hierarchical parent-child splitters
  - Storage (`src/storage/`): MinIO S3 object storage client
  - Security (`src/security/`): ClamAV TCP stream scanner
  - Observability (`src/langfuse.ts`): Optional Langfuse tracing client

## Scripts

```bash
pnpm --filter @rag/providers build
pnpm --filter @rag/providers typecheck
pnpm --filter @rag/providers lint
```
