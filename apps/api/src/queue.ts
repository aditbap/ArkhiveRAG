import { Queue, type JobsOptions } from 'bullmq';
import { Redis, type RedisOptions } from 'ioredis';
import * as dotenv from 'dotenv';

dotenv.config({ path: '../../.env' });

export const INGESTION_QUEUE_NAME = 'document-ingestion';

const redisUrl = process.env.REDIS_URL || 'redis://localhost:6379';
const commandTimeoutMs = Math.max(100, parseInt(process.env.REDIS_COMMAND_TIMEOUT_MS || '500', 10) || 500);

/**
 * Fast fail-open Redis options for synchronous HTTP rate-limiting & cache paths.
 * Disabling offline queuing and bounding commandTimeout ensures a Redis outage never
 * hangs HTTP requests — they reject in <500ms so callers fail open gracefully.
 */
export const CACHE_REDIS_OPTIONS: RedisOptions = {
  maxRetriesPerRequest: 1,
  commandTimeout: commandTimeoutMs,
  connectTimeout: 1000,
  enableOfflineQueue: false,
  retryStrategy: (times: number) => Math.min(times * 200, 5000),
};

/**
 * Dedicated blocking Redis options required by BullMQ producers/consumers.
 */
export const QUEUE_REDIS_OPTIONS: RedisOptions = {
  maxRetriesPerRequest: null,
  connectTimeout: 2000,
  retryStrategy: (times: number) => Math.min(times * 250, 5000),
};

/**
 * Unified BullMQ job retry & retention policy enforced on the API producer so every
 * enqueued document automatically retries transient embedding/storage/DB failures.
 */
export const DEFAULT_INGESTION_JOB_OPTIONS: JobsOptions = {
  attempts: 3,
  backoff: {
    type: 'exponential',
    delay: 2000,
  },
  removeOnComplete: { count: 1000 },
  removeOnFail: { count: 5000 },
};

function attachErrorGuard(client: Redis, label: string): Redis {
  let lastLoggedAt = 0;
  client.on('error', (err) => {
    const now = Date.now();
    if (now - lastLoggedAt >= 10_000) {
      lastLoggedAt = now;
      console.warn(`[Redis:${label}] connection error (degraded mode): ${err.message}`);
    }
  });
  return client;
}

/** Fast fail-open client used by HTTP routes (rate limits, chat cache, ACL versions). */
export const cacheRedisConnection = attachErrorGuard(new Redis(redisUrl, CACHE_REDIS_OPTIONS), 'cache');

/** Backward-compatible export pointing HTTP consumers to the fast fail-open client. */
export const redisConnection = cacheRedisConnection;

/** Dedicated BullMQ connection isolated from HTTP cache timeouts. */
export const queueRedisConnection = attachErrorGuard(new Redis(redisUrl, QUEUE_REDIS_OPTIONS), 'queue');

export const ingestionQueue = new Queue(INGESTION_QUEUE_NAME, {
  connection: queueRedisConnection,
  defaultJobOptions: DEFAULT_INGESTION_JOB_OPTIONS,
});

export async function closeRedisConnections(): Promise<void> {
  await Promise.allSettled([
    ingestionQueue.close(),
    cacheRedisConnection.quit().catch(() => cacheRedisConnection.disconnect()),
    queueRedisConnection.quit().catch(() => queueRedisConnection.disconnect()),
  ]);
}
