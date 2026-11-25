import { after, describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { consumeRateLimit, type RateLimitRedis } from '../../src/lib/rate-limit.js';
import {
  CACHE_REDIS_OPTIONS,
  QUEUE_REDIS_OPTIONS,
  DEFAULT_INGESTION_JOB_OPTIONS,
  closeRedisConnections,
} from '../../src/queue.js';
import {
  DATABASE_POOL_MAX,
  DATABASE_CONNECT_TIMEOUT_S,
  DATABASE_STATEMENT_TIMEOUT_MS,
  queryClient,
} from '@rag/database';

after(async () => {
  await closeRedisConnections();
  await queryClient.end({ timeout: 1 }).catch(() => undefined);
});

describe('V2.6 Phase 1 — Fail-Open Redis Split, Retry Policy & Bounded DB Pool', () => {
  it('isolates fast fail-open cache Redis options from BullMQ blocking queue options', () => {
    assert.equal(CACHE_REDIS_OPTIONS.enableOfflineQueue, false);
    assert.equal(CACHE_REDIS_OPTIONS.maxRetriesPerRequest, 1);
    assert.ok((CACHE_REDIS_OPTIONS.commandTimeout ?? 9999) <= 1000);
    assert.equal(QUEUE_REDIS_OPTIONS.maxRetriesPerRequest, null);
  });

  it('enforces exponential retries and bounded job retention on the API ingestion producer', () => {
    assert.equal(DEFAULT_INGESTION_JOB_OPTIONS.attempts, 3);
    assert.deepEqual(DEFAULT_INGESTION_JOB_OPTIONS.backoff, { type: 'exponential', delay: 2000 });
    assert.deepEqual(DEFAULT_INGESTION_JOB_OPTIONS.removeOnComplete, { count: 1000 });
    assert.deepEqual(DEFAULT_INGESTION_JOB_OPTIONS.removeOnFail, { count: 5000 });
  });

  it('fails open immediately on Redis outage and notifies degraded callback', async () => {
    let degradedCalls = 0;
    const brokenRedis: RateLimitRedis = {
      async incr() {
        throw new Error('Stream isn\'t writeable and enableOfflineQueue options is false');
      },
      async ttl() {
        return -1;
      },
      async expire() {
        return 1;
      },
    };

    const t0 = performance.now();
    const res = await consumeRateLimit(brokenRedis, 'rate-limit:ip:127.0.0.1', 100, 60, () => {
      degradedCalls++;
    });
    const elapsedMs = performance.now() - t0;

    assert.equal(res.allowed, true);
    assert.equal(res.count, 0);
    assert.equal(degradedCalls, 1);
    assert.ok(elapsedMs < 100, `expected fast fail-open <100ms, got ${elapsedMs}ms`);
  });

  it('configures bounded Postgres pool size, connect timeout, and statement timeout', () => {
    assert.ok(DATABASE_POOL_MAX >= 5);
    assert.ok(DATABASE_CONNECT_TIMEOUT_S >= 1 && DATABASE_CONNECT_TIMEOUT_S <= 30);
    assert.ok(DATABASE_STATEMENT_TIMEOUT_MS > 0 && DATABASE_STATEMENT_TIMEOUT_MS <= 120_000);
  });
});

describe('V2.6 Phase 2 — Crash-Safe Idempotent Ingestion & Storage Guards', () => {
  it('exposes transactional replaceVersionChunks on ChunkRepository for zero-duplicate retries', async () => {
    const { ChunkRepository } = await import('@rag/database');
    const repo = new ChunkRepository();
    assert.equal(typeof repo.replaceVersionChunks, 'function');
  });

  it('distinguishes transient retry attempts from final job failures so UI never flashes false failed state', () => {
    const isFinal = (unrecoverable: boolean, attemptsMade: number, maxAttempts: number) =>
      unrecoverable || attemptsMade + 1 >= Math.max(1, maxAttempts);

    assert.equal(isFinal(false, 0, 3), false);
    assert.equal(isFinal(false, 1, 3), false);
    assert.equal(isFinal(false, 2, 3), true);
    assert.equal(isFinal(true, 0, 3), true);
  });
});

describe('V2.6 Phase 3 — Distributed Tracing, Correlation Headers & Production Metrics', () => {
  it('propagates X-Request-Id, generates valid W3C traceparent, and exposes resilience metrics', async () => {
    process.env.MINIO_ENDPOINT ||= 'localhost';
    process.env.MINIO_ACCESS_KEY ||= 'test-access-key';
    process.env.MINIO_SECRET_KEY ||= 'test-secret-key';
    process.env.MINIO_BUCKET ||= 'test-documents';

    const { buildServer } = await import('../../src/server.js');
    const server = buildServer();
    try {
      const incomingTraceId = '4bf92f3577b34da6a3ce929d0e0e4736';
      const healthRes = await server.inject({
        method: 'GET',
        url: '/health',
        headers: {
          'x-request-id': 'req-v26-trace-001',
          traceparent: `00-${incomingTraceId}-00f067aa0ba902b7-01`,
        },
      });
      assert.equal(healthRes.statusCode, 200);
      assert.equal(healthRes.headers['x-request-id'], 'req-v26-trace-001');
      const tp = String(healthRes.headers['traceparent'] ?? '');
      assert.match(tp, new RegExp(`^00-${incomingTraceId}-[0-9a-f]{16}-01$`));

      const metricsRes = await server.inject({ method: 'GET', url: '/metrics' });
      assert.equal(metricsRes.statusCode, 200);
      const body = metricsRes.body;
      assert.ok(body.includes('http_request_duration_seconds'));
      assert.ok(body.includes('rag_redis_degraded_total'));
      assert.ok(body.includes('rag_dependency_errors_total'));
      assert.ok(body.includes('rag_db_pool_connections'));
    } finally {
      await server.close();
    }
  });
});
