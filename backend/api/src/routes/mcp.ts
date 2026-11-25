import { FastifyPluginAsync } from 'fastify';
import { requirePermissionMiddleware } from '@rag/auth';
import { normalizeAgentRole } from '@rag/core';
import {
  AgentToolsRepository,
  ChunkRepository,
  CommunityRepository,
  KnowledgeGraphRepository,
  auditLogs,
  db,
} from '@rag/database';
import { createEmbeddingProvider, createLLMForRole, createRerankerProvider } from '@rag/providers';
import { CompanyAgentOrchestrator } from '../lib/company-agent.js';
import { LightRagRetriever } from '../lib/lightrag-retriever.js';
import { handleMcpJsonRpc, type JsonRpcRequest } from '../lib/mcp-handler.js';
import { ragAgentPolicyDenials, ragAgentSteps, ragAgentToolCalls } from '../plugins/metrics.js';
import { redisConnection } from '../queue.js';

const agentToolsRepo = new AgentToolsRepository();
const chunkRepo = new ChunkRepository();
const kgRepo = new KnowledgeGraphRepository();
const communityRepo = new CommunityRepository();

function createRetriever() {
  return new LightRagRetriever({
    chunkRepo,
    kgRepo,
    embedder: createEmbeddingProvider(),
    keywordsLlm: createLLMForRole('keywords'),
    reranker: createRerankerProvider(),
    fastCache: redisConnection,
    communityRepo,
  });
}

export const mcpRoutes: FastifyPluginAsync = async (fastify) => {
  fastify.get('/', async (_request, reply) => {
    return reply.status(405).send({
      error: {
        code: 'METHOD_NOT_ALLOWED',
        message: 'Send MCP JSON-RPC 2.0 requests via POST /mcp with Authorization: Bearer <token|api_key>.',
      },
    });
  });

  fastify.post(
    '/',
    { preHandler: requirePermissionMiddleware('chat:read') },
    async (request, reply) => {
      const orgId = request.user?.orgId || request.user?.id;
      const userId = request.user?.id;
      if (!orgId || !userId) {
        return reply.status(403).send({ error: { code: 'FORBIDDEN', message: 'Organization required' } });
      }

      const role = normalizeAgentRole(request.userRole);
      const retriever = createRetriever();
      const orchestrator = new CompanyAgentOrchestrator({
        retriever,
        toolsRepo: agentToolsRepo,
        llm: createLLMForRole('query'),
        telemetry: {
          onToolCall: (tool, outcome) => ragAgentToolCalls.inc({ tool, outcome }),
          onPolicyDenial: (r, code) => ragAgentPolicyDenials.inc({ role: r, code }),
          onStepsCompleted: (steps) => ragAgentSteps.observe(steps),
        },
      });

      const body = (request.body ?? {}) as JsonRpcRequest;
      const outcome = await handleMcpJsonRpc(body, {
        orgId,
        userId,
        role,
        requestId: request.id,
        ipAddress: request.ip,
        searchKnowledgeBase: async ({ query, mode, documentIds }) => {
          const retrieved = await retriever.retrieve({
            orgId,
            userQuery: query,
            mode,
            documentIds,
            includeDrafts: role === 'owner' || role === 'admin',
          });
          return {
            mode: retrieved.effectiveMode,
            crag: retrieved.crag,
            chunks: retrieved.assembled.chunksUsed.map((c, idx) => ({
              index: idx + 1,
              documentId: c.documentId,
              documentTitle: c.documentTitle,
              fileName: c.fileName,
              pageNumber: c.pageNumber ?? null,
              sectionHeading: c.sectionHeading ?? null,
              score: c.similarity,
              content: c.content,
            })),
          };
        },
        askCompanyAgent: async ({ query, synthesize, documentIds }) => {
          const result = await orchestrator.run({
            orgId,
            userId,
            role,
            userQuery: query,
            documentIds,
            requestId: request.id,
            ipAddress: request.ip,
            synthesizeAnswer: synthesize,
          });
          return {
            decision: result.decision,
            role: result.role,
            refused: result.refused,
            refusalCode: result.refusalCode,
            refusalReason: result.refusalReason,
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
                pageNumber: chunk.pageNumber ?? null,
                sectionHeading: chunk.sectionHeading ?? null,
              })) ?? [],
          };
        },
        listDocuments: async ({ limit }) =>
          agentToolsRepo.queryDocumentInventory(orgId, {
            includeDrafts: role === 'owner' || role === 'admin',
            limit,
          }),
        getGovernanceAudit: async ({ limit }) =>
          agentToolsRepo.queryGovernanceAudit(orgId, { limit }),
        getSecurityAuditLogs: async ({ limit }) =>
          agentToolsRepo.querySecurityAuditLogs(orgId, { limit }),
        recordAudit: async ({ action, tool, allowed, reason }) => {
          await db
            .insert(auditLogs)
            .values({
              orgId,
              userId,
              action,
              resourceType: 'mcp_tool',
              details: { tool, role, allowed, reason: reason ?? null, apiKeyId: request.apiKey?.id ?? null },
              ipAddress: request.ip,
            })
            .catch(() => undefined);
        },
      });

      if (outcome.payload === null) {
        return reply.status(outcome.status).send();
      }
      return reply.status(outcome.status).send(outcome.payload);
    },
  );
};
