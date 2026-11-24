import { FastifyPluginAsync } from 'fastify';
import { requirePermissionMiddleware } from '@rag/auth';
import type { LearningEvalCase } from '@rag/core';
import { CommunityRepository } from '@rag/database';
import { CommunitySummaryBuilder, createEmbeddingProvider, createLLMForRole } from '@rag/providers';
import { RetrievalLearningEngine } from '../lib/learning-engine.js';
import { redisConnection } from '../queue.js';

const learningEngine = new RetrievalLearningEngine({ fastCache: redisConnection });
const communityRepo = new CommunityRepository();
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export const learningRoutes: FastifyPluginAsync = async (fastify) => {
  // Record explicit user feedback (thumbs up/down + cited chunks) for closed-loop learning
  fastify.post(
    '/feedback',
    { preHandler: requirePermissionMiddleware('chat:create') },
    async (request, reply) => {
      const orgId = request.user?.orgId || request.user?.id;
      const userId = request.user?.id;
      if (!orgId || !userId) {
        return reply.status(403).send({ error: { code: 'FORBIDDEN', message: 'Organization required' } });
      }

      const body = (request.body ?? {}) as {
        telemetryId?: string;
        messageId?: string;
        rating?: number;
        citedChunkIds?: string[];
        reasonCode?: string;
        comment?: string;
      };

      if (body.rating !== 1 && body.rating !== -1) {
        return reply.status(400).send({ error: { code: 'BAD_REQUEST', message: 'rating must be 1 or -1' } });
      }

      const saved = await learningEngine.recordFeedback({
        orgId,
        userId,
        telemetryId: body.telemetryId && UUID_RE.test(body.telemetryId) ? body.telemetryId : null,
        messageId: body.messageId && UUID_RE.test(body.messageId) ? body.messageId : null,
        rating: body.rating,
        citedChunkIds: Array.isArray(body.citedChunkIds)
          ? body.citedChunkIds.filter((id) => typeof id === 'string' && UUID_RE.test(id))
          : [],
        reasonCode: typeof body.reasonCode === 'string' ? body.reasonCode.slice(0, 64) : null,
        comment: typeof body.comment === 'string' ? body.comment.slice(0, 1000) : null,
      });

      return reply.status(201).send({ data: saved });
    },
  );

  // Live retrieval metrics, intent mix, token usage, latency, CRAG verdicts, and recent experiments
  fastify.get(
    '/metrics',
    { preHandler: requirePermissionMiddleware('chat:read') },
    async (request, reply) => {
      const orgId = request.user?.orgId || request.user?.id;
      if (!orgId) {
        return reply.status(403).send({ error: { code: 'FORBIDDEN', message: 'Organization required' } });
      }
      const query = request.query as { days?: string };
      const days = Math.min(90, Math.max(1, parseInt(query.days || '7', 10) || 7));
      const summary = await learningEngine.getMetricsSummary(orgId, days);
      const active = await learningEngine.getActiveConfig(orgId);
      return reply.status(200).send({
        data: {
          activeConfigId: active.id,
          activeConfigVersion: active.config.version,
          ...summary,
        },
      });
    },
  );

  // List immutable retrieval config versions for this tenant
  fastify.get(
    '/configs',
    { preHandler: requirePermissionMiddleware('chat:read') },
    async (request, reply) => {
      const orgId = request.user?.orgId || request.user?.id;
      if (!orgId) {
        return reply.status(403).send({ error: { code: 'FORBIDDEN', message: 'Organization required' } });
      }
      const configs = await learningEngine.listConfigs(orgId);
      const active = await learningEngine.getActiveConfig(orgId);
      return reply.status(200).send({
        data: {
          activeConfig: active.config,
          activeConfigId: active.id,
          versions: configs,
        },
      });
    },
  );

  // Admin: Run controlled weight optimization + mandatory regression gate over golden eval cases
  fastify.post(
    '/optimize',
    { preHandler: requirePermissionMiddleware('documents:delete') },
    async (request, reply) => {
      const orgId = request.user?.orgId || request.user?.id;
      const userId = request.user?.id;
      if (!orgId) {
        return reply.status(403).send({ error: { code: 'FORBIDDEN', message: 'Organization required' } });
      }
      const body = (request.body ?? {}) as {
        cases?: LearningEvalCase[];
        autoPromote?: boolean;
        notes?: string;
      };
      if (!Array.isArray(body.cases) || body.cases.length === 0) {
        return reply.status(400).send({
          error: { code: 'BAD_REQUEST', message: 'Non-empty cases array is required for regression-gated optimization' },
        });
      }

      const result = await learningEngine.optimizeAndGate({
        orgId,
        cases: body.cases,
        createdBy: userId ?? null,
        autoPromote: Boolean(body.autoPromote),
        notes: typeof body.notes === 'string' ? body.notes.slice(0, 500) : undefined,
      });

      return reply.status(200).send({ data: result });
    },
  );

  // Admin: Promote or rollback to a specific immutable config version
  fastify.post(
    '/configs/:id/promote',
    { preHandler: requirePermissionMiddleware('documents:delete') },
    async (request, reply) => {
      const orgId = request.user?.orgId || request.user?.id;
      const { id } = request.params as { id: string };
      if (!orgId) {
        return reply.status(403).send({ error: { code: 'FORBIDDEN', message: 'Organization required' } });
      }
      if (!UUID_RE.test(id)) {
        return reply.status(400).send({ error: { code: 'BAD_REQUEST', message: 'Invalid config ID' } });
      }
      const promoted = await learningEngine.promoteConfig(orgId, id);
      if (!promoted) {
        return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Config version not found' } });
      }
      return reply.status(200).send({ data: promoted });
    },
  );

  // Admin: Incrementally rebuild RAPTOR community summaries (#8) for this tenant
  fastify.post(
    '/communities/rebuild',
    { preHandler: requirePermissionMiddleware('documents:delete') },
    async (request, reply) => {
      const orgId = request.user?.orgId || request.user?.id;
      if (!orgId) {
        return reply.status(403).send({ error: { code: 'FORBIDDEN', message: 'Organization required' } });
      }
      const builder = new CommunitySummaryBuilder({
        summaryLlm: createLLMForRole('summary'),
        embedder: createEmbeddingProvider(),
        bridge: communityRepo,
      });
      const summary = await builder.rebuildForOrg(orgId);
      return reply.status(200).send({ data: summary });
    },
  );
};
