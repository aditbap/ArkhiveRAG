import crypto from 'node:crypto';
import fastify, { FastifyInstance } from 'fastify';
import cors from '@fastify/cors';
import multipart from '@fastify/multipart';
import { authenticate } from '@rag/auth';
import { db, sql } from '@rag/database';
import { MinioStorageProvider } from '@rag/providers';
import { documentRoutes } from './routes/documents.js';
import { chatRoutes } from './routes/chat.js';
import { guestChatRoutes } from './routes/guest-chat.js';
import { conversationsRoutes } from './routes/conversations.js';
import folderRoutes from './routes/folders.js';
import { libraryRoutes } from './routes/libraries.js';
import { learningRoutes } from './routes/learning.js';
import { agentRoutes } from './routes/agent.js';
import { ragRedisDegraded, registerMetrics } from './plugins/metrics.js';
import { redisConnection } from './queue.js';
import { consumeRateLimit } from './lib/rate-limit.js';
import * as dotenv from 'dotenv';

dotenv.config({ path: '../../.env' });

const W3C_TRACEPARENT_RE = /^00-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}$/i;

function buildTraceparent(incomingHeader?: string): string {
  if (incomingHeader) {
    const m = W3C_TRACEPARENT_RE.exec(incomingHeader.trim());
    if (m?.[1]) {
      const spanId = crypto.randomBytes(8).toString('hex');
      return `00-${m[1].toLowerCase()}-${spanId}-01`;
    }
  }
  return `00-${crypto.randomBytes(16).toString('hex')}-${crypto.randomBytes(8).toString('hex')}-01`;
}

export function buildServer(): FastifyInstance {
  const server = fastify({
    ignoreTrailingSlash: true,
    genReqId: (req) => {
      const incoming = req.headers['x-request-id'];
      if (typeof incoming === 'string' && /^[a-zA-Z0-9._-]{1,128}$/.test(incoming.trim())) {
        return incoming.trim();
      }
      return crypto.randomUUID();
    },
    logger: {
      level: process.env.LOG_LEVEL || 'info',
    },
    // Applies to JSON/text bodies only (file uploads stream through @fastify/multipart, capped by
    // MAX_FILE_SIZE_MB below) — so a 100MB JSON body can't be used to exhaust memory.
    bodyLimit: parseInt(process.env.JSON_BODY_LIMIT_MB || '2', 10) * 1024 * 1024,
    // Behind the reverse proxy: take the client address from X-Forwarded-For so per-IP limits and
    // logs see real clients rather than the proxy.
    trustProxy: process.env.TRUST_PROXY === 'true',
  });

  // Echo request correlation & W3C traceparent headers on every response for distributed tracing.
  server.addHook('onRequest', async (request, reply) => {
    const incomingTp = typeof request.headers['traceparent'] === 'string' ? request.headers['traceparent'] : undefined;
    const tp = buildTraceparent(incomingTp);
    reply.header('x-request-id', request.id);
    reply.header('traceparent', tp);
  });

  // Plugins
  server.register(cors, {
    origin: process.env.CORS_ORIGINS?.split(',').map((o) => o.trim()) || '*',
  });
  
  server.register(multipart, {
    limits: {
      fileSize: parseInt(process.env.MAX_FILE_SIZE_MB || '100') * 1024 * 1024,
    },
  });

  // Global Auth Decorators
  server.decorateRequest('user', null);
  server.decorateRequest('userRole', null);

  // Liveness: the process is up and serving.
  server.get('/health', async () => {
    return { status: 'ok', timestamp: new Date().toISOString() };
  });

  // Readiness: the dependencies this API cannot work without. 503 takes the instance out of
  // rotation (load balancer / compose healthcheck) instead of serving errors.
  const storage = MinioStorageProvider.fromEnv();
  const withTimeout = <T>(p: Promise<T>, ms = 2000) =>
    Promise.race([p, new Promise<never>((_, rej) => setTimeout(() => rej(new Error('timeout')), ms))]);
  server.get('/ready', async (_request, reply) => {
    const checks: Record<string, 'ok' | string> = {};
    await Promise.all([
      ['postgres', () => db.execute(sql`select 1`)],
      ['redis', () => redisConnection.ping()],
      ['storage', () => storage.ping()],
    ].map(async ([name, probe]) => {
      try {
        await withTimeout((probe as () => Promise<unknown>)());
        checks[name as string] = 'ok';
      } catch (err) {
        checks[name as string] = err instanceof Error ? err.message : 'failed';
      }
    }));
    const ready = Object.values(checks).every((v) => v === 'ok');
    return reply.status(ready ? 200 : 503).send({ status: ready ? 'ready' : 'unavailable', checks });
  });

  // Coarse per-IP flood guard in front of everything (per-user limits for chat/upload/join are
  // finer-grained and live in the routes). Fails open if Redis is down.
  const ipLimit = parseInt(process.env.RATE_LIMIT_IP_PER_MINUTE || '600', 10);
  server.addHook('onRequest', async (request, reply) => {
    if (request.url === '/health' || request.url === '/ready' || request.url === '/metrics') return;
    const result = await consumeRateLimit(
      redisConnection,
      `rate-limit:ip:${request.ip}`,
      ipLimit,
      60,
      () => ragRedisDegraded.inc({ operation: 'ip_rate_limit' }),
    );
    if (!result.allowed) {
      return reply.status(429).send({ error: { code: 'TOO_MANY_REQUESTS', message: 'Too many requests from this address.' } });
    }
  });

  registerMetrics(server);

  // Public routes (no login): guests may chat without documents. Registered outside the
  // authenticated scope below — keep everything that touches a library out of here.
  server.register(guestChatRoutes, { prefix: '/api/v1/guest/chat' });

  // API Routes (v1)
  server.register(async (api) => {
    // Add auth hook for all routes in this context
    api.addHook('onRequest', authenticate);
    
    api.register(documentRoutes, { prefix: '/documents' });
    api.register(chatRoutes, { prefix: '/chat' });
    api.register(conversationsRoutes, { prefix: '/conversations' });
    api.register(folderRoutes, { prefix: '/folders' });
    api.register(libraryRoutes, { prefix: '/libraries' });
    api.register(learningRoutes, { prefix: '/retrieval' });
    api.register(agentRoutes, { prefix: '/agent' });
  }, { prefix: '/api/v1' });

  return server;
}
