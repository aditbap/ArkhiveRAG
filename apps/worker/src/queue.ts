import { Queue } from 'bullmq';
import { Redis } from 'ioredis';
import * as dotenv from 'dotenv';

dotenv.config({ path: '../../.env' });

export const INGESTION_QUEUE_NAME = 'document-ingestion';

const redisUrl = process.env.REDIS_URL || 'redis://localhost:6379';

export const redisConnection = new Redis(redisUrl, {
  maxRetriesPerRequest: null,
  connectTimeout: 2000,
  retryStrategy: (times: number) => Math.min(times * 250, 5000),
});

let lastRedisErrLog = 0;
redisConnection.on('error', (err) => {
  const now = Date.now();
  if (now - lastRedisErrLog >= 10_000) {
    lastRedisErrLog = now;
    console.warn(`[Worker:Redis] connection error: ${err.message}`);
  }
});

export const ingestionQueue = new Queue(INGESTION_QUEUE_NAME, {
  connection: redisConnection,
  defaultJobOptions: {
    attempts: 3,
    backoff: {
      type: 'exponential',
      delay: 2000,
    },
    removeOnComplete: { count: 1000 },
    removeOnFail: { count: 5000 },
  },
});
