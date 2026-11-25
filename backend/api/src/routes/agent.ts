import { FastifyPluginAsync } from 'fastify';
import { requirePermissionMiddleware } from '@rag/auth';
import { listToolsForRole, normalizeAgentRole } from '@rag/core';
import { AgentToolsRepository, ChunkRepository, CommunityRepository, KnowledgeGraphRepository } from '@rag/database';
import { createEmbeddingProvider, createLLMForRole, createRerankerProvider } from '@rag/providers';
import { CompanyAgentOrchestrator } from '../lib/company-agent.js';
import { LightRagRetriever } from '../lib/lightrag-retriever.js';
import { ragAgentPolicyDenials, ragAgentSteps, ragAgentToolCalls } from '../plugins/metrics.js';
import { redisConnection } from '../queue.js';

const agentToolsRepo = new AgentToolsRepository();
const chunkRepo = new ChunkRepository();
const kgRepo = new KnowledgeGraphRepository();
const communityRepo = new CommunityRepository();

function createAgentOrchestrator() {
  const retriever = new LightRagRetriever({
    chunkRepo,
    kgRepo,
    embedder: createEmbeddingProvider(),
    keywordsLlm: createLLMForRole('keywords'),
    reranker: createRerankerProvider(),
    fastCache: redisConnection,
    communityRepo,
  });

  return new CompanyAgentOrchestrator({
    retriever,
    toolsRepo: agentToolsRepo,
    llm: createLLMForRole('query'),
    telemetry: {
      onToolCall: (tool, outcome) => ragAgentToolCalls.inc({ tool, outcome }),
      onPolicyDenial: (role, code) => ragAgentPolicyDenials.inc({ role, code }),
      onStepsCompleted: (steps) => ragAgentSteps.observe(steps),
    },
  });
}

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export const agentRoutes: FastifyPluginAsync = async (fastify) => {
  // Discover authorized vs restricted agent tools for the caller's role
  fastify.get(
    '/tools',
    { preHandler: requirePermissionMiddleware('chat:read') },
    async (request, reply) => {
      const orgId = request.user?.orgId || request.user?.id;
      if (!orgId) {
        return reply.status(403).send({ error: { code: 'FORBIDDEN', message: 'Organization required' } });
      }
      const role = normalizeAgentRole(request.userRole);
      return {
        data: {
          orgId,
          role,
          ...listToolsForRole(role),
        },
      };
    },
  );

  // Execute RBAC-enforced Company Agent query (document retrieval + allowlisted DB tools + policy refusal)
  fastify.post(
    '/query',
    { preHandler: requirePermissionMiddleware('chat:create') },
    async (request, reply) => {
      const orgId = request.user?.orgId || request.user?.id;
      const userId = request.user?.id;
      if (!orgId || !userId) {
        return reply.status(403).send({ error: { code: 'FORBIDDEN', message: 'Organization required' } });
      }

      const body = (request.body ?? {}) as {
        query?: unknown;
        documentIds?: unknown;
        synthesize?: unknown;
      };
      if (typeof body.query !== 'string' || !body.query.trim() || body.query.length > 4000) {
        return reply.status(400).send({
          error: { code: 'BAD_REQUEST', message: 'query must be a non-empty string (<= 4000 chars)' },
        });
      }

      let documentIds: string[] | undefined;
      if (body.documentIds !== undefined) {
        if (!Array.isArray(body.documentIds) || !body.documentIds.every((id): id is string => typeof id === 'string' && UUID_RE.test(id))) {
          return reply.status(400).send({
            error: { code: 'BAD_REQUEST', message: 'documentIds must be an array of valid UUIDs' },
          });
        }
        documentIds = body.documentIds;
      }

      const role = normalizeAgentRole(request.userRole);
      const orchestrator = createAgentOrchestrator();
      const result = await orchestrator.run({
        orgId,
        userId,
        role,
        userQuery: body.query.trim(),
        documentIds,
        requestId: request.id,
        ipAddress: request.ip,
        synthesizeAnswer: body.synthesize !== false,
      });

      return reply.status(200).send({
        data: {
          decision: result.decision,
          role: result.role,
          refused: result.refused,
          refusalCode: result.refusalCode,
          refusalReason: result.refusalReason,
          requiredRoles: result.requiredRoles,
          answer: result.answer,
          steps: result.steps,
          structuredData: result.results
            .filter((r) => r.tool !== 'search_documents' && r.tool !== 'search_draft_documents')
            .map((r) => ({ tool: r.tool, summary: r.summary, data: r.data })),
          citations:
            result.retrieval?.assembled.chunksUsed.map((chunk, idx) => ({
              index: idx + 1,
              document: chunk.documentTitle,
              documentId: chunk.documentId,
              fileName: chunk.fileName,
              score: chunk.similarity,
              pageNumber: chunk.pageNumber ?? undefined,
              sectionHeading: chunk.sectionHeading ?? undefined,
              snippet: chunk.content.slice(0, 120) + '...',
            })) ?? [],
          timingsMs: result.timingsMs,
        },
      });
    },
  );

  // Owner-only: inspect recent agent allow/deny security audit trail
  fastify.get(
    '/audit',
    { preHandler: requirePermissionMiddleware('settings:update') },
    async (request, reply) => {
      const orgId = request.user?.orgId || request.user?.id;
      if (!orgId) {
        return reply.status(403).send({ error: { code: 'FORBIDDEN', message: 'Organization required' } });
      }
      const { limit } = request.query as { limit?: string };
      const parsedLimit = Math.min(50, Math.max(1, parseInt(limit || '25', 10) || 25));
      const rows = await agentToolsRepo.querySecurityAuditLogs(orgId, {
        actionPrefix: 'agent:',
        limit: parsedLimit,
      });
      return { data: rows };
    },
  );
};
