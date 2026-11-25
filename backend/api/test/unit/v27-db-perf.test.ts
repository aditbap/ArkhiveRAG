import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SingleflightGroup } from '@rag/core';
import { CHUNK_RETRIEVAL_COLUMNS, resolvePoolOptions } from '@rag/database';
import { LightRagRetriever, invalidateGraphDataCache } from '../../src/lib/lightrag-retriever.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '../../../..');

test('V2.7 Phase 1: CHUNK_RETRIEVAL_COLUMNS omits heavy embedding & content_tsv columns', () => {
  const keys = Object.keys(CHUNK_RETRIEVAL_COLUMNS);
  assert.equal(keys.includes('embedding'), false, 'embedding must never be projected in retrieval queries');
  assert.equal(keys.includes('contentTsv'), false, 'contentTsv must never be projected in retrieval queries');
  for (const required of [
    'id',
    'documentId',
    'orgId',
    'content',
    'sectionHeading',
    'pageNumber',
    'parentChunkId',
    'chunkRole',
    'tokenCount',
  ]) {
    assert.equal(keys.includes(required), true, `required column ${required} missing from CHUNK_RETRIEVAL_COLUMNS`);
  }
});

test('V2.7 Phase 1: vector repositories use raw <=> ASC HNSW ordering (never ORDER BY 1-(embedding<=>) DESC)', () => {
  const repoFiles = [
    'packages/database/src/repositories/chunk.repository.ts',
    'packages/database/src/repositories/knowledge-graph.repository.ts',
    'packages/database/src/repositories/community.repository.ts',
  ];

  for (const relPath of repoFiles) {
    const src = fs.readFileSync(path.join(REPO_ROOT, relPath), 'utf-8');
    // Guard against the V2.6 HNSW killer pattern:
    assert.equal(
      /orderBy\(\s*desc\(\s*similarity\s*\)/i.test(src),
      false,
      `${relPath} must not use orderBy(desc(similarity)) on pgvector cosine similarity`,
    );
    assert.equal(
      /ORDER BY\s+.*embedding\s*<=>.*ASC/i.test(src),
      true,
      `${relPath} must order by raw embedding <=> ... ASC for pgvector HNSW index scans`,
    );
  }
});

test('V2.7 Phase 3: resolvePoolOptions disables prepared statements under PgBouncer transaction mode', () => {
  const direct = resolvePoolOptions('postgresql://postgres:secret@postgres:5432/rag_platform');
  assert.equal(direct.poolerMode, 'session');
  assert.equal(direct.prepare, true);
  assert.equal(direct.max, 25);

  // Auto-detect standard PgBouncer port :6432
  const autoPort = resolvePoolOptions('postgresql://postgres:secret@localhost:6432/rag_platform');
  assert.equal(autoPort.poolerMode, 'transaction');
  assert.equal(autoPort.prepare, false);

  // Auto-detect ?pgbouncer=true query parameter
  const autoQuery = resolvePoolOptions('postgresql://postgres:secret@db.internal:5432/rag_platform?pgbouncer=true');
  assert.equal(autoQuery.poolerMode, 'transaction');
  assert.equal(autoQuery.prepare, false);

  // Explicit env override for container-internal `pgbouncer:5432`
  const explicit = resolvePoolOptions('postgresql://postgres:secret@pgbouncer:5432/rag_platform', {
    poolerModeEnv: 'transaction',
    poolMaxEnv: '40',
  });
  assert.equal(explicit.poolerMode, 'transaction');
  assert.equal(explicit.prepare, false);
  assert.equal(explicit.max, 40);
});

test('V2.7 Phase 4: SingleflightGroup coalesces identical concurrent calls and isolates distinct tenants/errors', async () => {
  const sf = new SingleflightGroup();
  let callsA = 0;
  let callsB = 0;

  const task = (tenant: string, inc: () => number) => async () => {
    const n = inc();
    await new Promise((r) => setTimeout(r, 25));
    return `${tenant}:${n}`;
  };

  const results = await Promise.all([
    ...Array.from({ length: 8 }, () => sf.run('org-A|query-1', task('org-A', () => ++callsA))),
    ...Array.from({ length: 4 }, () => sf.run('org-B|query-1', task('org-B', () => ++callsB))),
  ]);

  assert.equal(callsA, 1, '8 identical concurrent org-A calls must execute once');
  assert.equal(callsB, 1, '4 identical concurrent org-B calls must execute independently from org-A');
  assert.equal(sf.coalescedCount(), 10);
  assert.equal(sf.inFlightCount(), 0);
  assert.deepEqual(results.slice(0, 8), Array(8).fill('org-A:1'));
  assert.deepEqual(results.slice(8), Array(4).fill('org-B:1'));

  // Errors must never poison subsequent calls
  let attempt = 0;
  await assert.rejects(
    sf.run('org-A|flaky', async () => {
      attempt++;
      throw new Error('transient DB timeout');
    }),
    /transient DB timeout/,
  );
  const recovered = await sf.run('org-A|flaky', async () => {
    attempt++;
    return 'ok';
  });
  assert.equal(recovered, 'ok');
  assert.equal(attempt, 2);
});

test('V2.7 Phase 4: LightRagRetriever caches hasGraphData per tenant+ACL and coalesces identical hybrid searches', async () => {
  let hasGraphCalls = 0;
  let searchSimilarCalls = 0;
  let searchFullTextCalls = 0;

  const mockChunkRepo = {
    async searchSimilar(orgId: string) {
      searchSimilarCalls++;
      await new Promise((r) => setTimeout(r, 20));
      return [
        {
          chunk: {
            id: `chunk-${orgId}`,
            documentId: `doc-${orgId}`,
            versionId: 'v1',
            orgId,
            chunkIndex: 0,
            content: `POL-SEC-409 hardware MFA policy for ${orgId}`,
            contentHash: 'h1',
            tokenCount: 12,
            pageNumber: 1,
            sectionHeading: 'Security',
            parentChunkId: null,
            chunkRole: 'standalone',
            metadata: {},
            embeddingModel: 'bge-m3',
            qdrantPointId: null,
            isActive: true,
            createdAt: new Date(),
          },
          document: {
            id: `doc-${orgId}`,
            orgId,
            title: `Security Policy ${orgId}`,
            fileName: 'policy.md',
            fileType: 'md',
            fileSizeBytes: 100,
            fileHash: 'fh1',
            status: 'ready' as const,
            folderId: null,
            activeVersionId: null,
            tags: [],
            metadata: {},
            effectiveAt: null,
            expiresAt: null,
            supersedesDocumentId: null,
            authorityTier: 3,
            governanceStatus: 'active',
            isDeleted: false,
            createdAt: new Date(),
            updatedAt: new Date(),
          },
          similarity: 0.92,
        },
      ];
    },
    async searchFullText() {
      searchFullTextCalls++;
      await new Promise((r) => setTimeout(r, 20));
      return [];
    },
    async findByIdsRanked() {
      return [];
    },
    async findParentsByIds() {
      return [];
    },
  };

  const mockKgRepo = {
    async hasGraphData() {
      hasGraphCalls++;
      return false;
    },
  };

  const mockEmbedder = {
    model: 'test-embed',
    dimensions: 4,
    async embedQuery() {
      return [0.5, 0.5, 0.5, 0.5];
    },
    async embedQueries(texts: string[]) {
      return texts.map(() => [0.5, 0.5, 0.5, 0.5]);
    },
    async embedDocuments(texts: string[]) {
      return texts.map(() => [0.5, 0.5, 0.5, 0.5]);
    },
  };

  const mockLlm = {
    model: 'test-llm',
    async chat() {
      return { content: '{}', usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 } };
    },
    async *streamChat() {
      yield { content: '' };
    },
  };

  const makeRetriever = () =>
    new LightRagRetriever({
      chunkRepo: mockChunkRepo as never,
      kgRepo: mockKgRepo as never,
      embedder: mockEmbedder as never,
      keywordsLlm: mockLlm as never,
    });

  // 5 concurrent identical requests for org-1 + 2 for org-2
  const [org1Results, org2Results] = await Promise.all([
    Promise.all(
      Array.from({ length: 5 }, () =>
        makeRetriever().retrieve({
          orgId: 'org-1',
          userQuery: 'What does POL-SEC-409 require?',
          mode: 'mix',
          disableCragRetry: true,
        }),
      ),
    ),
    Promise.all(
      Array.from({ length: 2 }, () =>
        makeRetriever().retrieve({
          orgId: 'org-2',
          userQuery: 'What does POL-SEC-409 require?',
          mode: 'mix',
          disableCragRetry: true,
        }),
      ),
    ),
  ]);

  assert.equal(hasGraphCalls, 2, 'hasGraphData must execute once per distinct tenant (cached across concurrent bursts)');
  assert.equal(searchSimilarCalls, 2, 'identical concurrent hybrid searches must coalesce per tenant');
  assert.equal(searchFullTextCalls, 2);
  assert.ok(org1Results.every((r) => r.assembled.chunksUsed.length > 0 && r.assembled.chunksUsed.every((c) => c.documentId === 'doc-org-1')));
  assert.ok(org2Results.every((r) => r.assembled.chunksUsed.length > 0 && r.assembled.chunksUsed.every((c) => c.documentId === 'doc-org-2')));

  // Second sequential wave for org-1 hits hasGraphData cache (0 new calls)
  await makeRetriever().retrieve({
    orgId: 'org-1',
    userQuery: 'Another query about POL-SEC-409',
    mode: 'mix',
    disableCragRetry: true,
  });
  assert.equal(hasGraphCalls, 2, 'subsequent org-1 query within 30s TTL must reuse cached hasGraphData');

  // Explicit cache invalidation forces re-check on next query
  invalidateGraphDataCache(mockKgRepo, 'org-1');
  await makeRetriever().retrieve({
    orgId: 'org-1',
    userQuery: 'Query after KG ingestion invalidation',
    mode: 'mix',
    disableCragRetry: true,
  });
  assert.equal(hasGraphCalls, 3, 'invalidateGraphDataCache must evict cached tenant entry');
});
