import { FastifyPluginAsync } from 'fastify';
import { requirePermissionMiddleware } from '@rag/auth';
import {
  db,
  conversations,
  messages,
  citations,
  ChunkRepository,
  KnowledgeGraphRepository,
  CommunityRepository,
  eq,
  and,
} from '@rag/database';
import {
  createEmbeddingProvider,
  createRerankerProvider,
  createLLMForRole,
  type LLMProvider,
  type LLMMessage,
  getLangfuse,
  flushLangfuse,
} from '@rag/providers';
import {
  CONSTANTS,
  CRAG_LOW_CONFIDENCE_DIRECTIVE,
  buildLightRagSystemPrompt,
  isRagQueryMode,
  type EmbeddingProvider,
  type RagQueryMode,
  type RerankerProvider,
} from '@rag/core';
import crypto from 'crypto';
import { redisConnection } from '../queue.js';
import { consumeRateLimit } from '../lib/rate-limit.js';
import { sseCorsHeaders } from '../lib/sse.js';
import { LightRagRetriever } from '../lib/lightrag-retriever.js';
import { RetrievalLearningEngine } from '../lib/learning-engine.js';
import { ragPipelineDuration, ragCacheHits } from '../plugins/metrics.js';

const chunkRepo = new ChunkRepository();
const kgRepo = new KnowledgeGraphRepository();
const communityRepo = new CommunityRepository();
const learningEngine = new RetrievalLearningEngine({ fastCache: redisConnection });

// Lazy-initialize providers so startup doesn't crash when env vars aren't loaded at import time
let embeddingProvider: EmbeddingProvider;
let queryLlmProvider: LLMProvider;
let keywordsLlmProvider: LLMProvider;
let rerankerProvider: RerankerProvider | null | undefined;

function getEmbeddingProvider(): EmbeddingProvider {
  if (!embeddingProvider) embeddingProvider = createEmbeddingProvider();
  return embeddingProvider;
}

function getQueryLlmProvider(): LLMProvider {
  if (!queryLlmProvider) queryLlmProvider = createLLMForRole('query');
  return queryLlmProvider;
}

function getKeywordsLlmProvider(): LLMProvider {
  if (!keywordsLlmProvider) keywordsLlmProvider = createLLMForRole('keywords');
  return keywordsLlmProvider;
}

function getRerankerProvider(): RerankerProvider | null {
  if (rerankerProvider !== undefined) return rerankerProvider;
  rerankerProvider = createRerankerProvider();
  if (!rerankerProvider) {
    console.warn('No cross-encoder reranker configured (Cloudflare/Cohere); reranking disabled');
  }
  return rerankerProvider;
}

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

interface IncomingMessage {
  role: string;
  content: string;
}

const isIncomingMessage = (m: unknown): m is IncomingMessage =>
  typeof m === 'object' &&
  m !== null &&
  typeof (m as IncomingMessage).role === 'string' &&
  typeof (m as IncomingMessage).content === 'string';

const RERANK_MIN_SCORE = parseFloat(process.env.RERANK_MIN_SCORE || '0.1');
const RERANK_LIMIT = parseInt(process.env.RERANK_LIMIT || String(CONSTANTS.DEFAULT_RERANK_LIMIT), 10);
const CHAT_RATE_LIMIT_PER_MINUTE = parseInt(process.env.RATE_LIMIT_CHAT_PER_MINUTE || '20', 10);
const DEFAULT_RAG_MODE: RagQueryMode = isRagQueryMode(process.env.RAG_DEFAULT_MODE)
  ? process.env.RAG_DEFAULT_MODE
  : CONSTANTS.DEFAULT_RAG_MODE;
const MAX_ENTITY_TOKENS = parseInt(
  process.env.RAG_MAX_ENTITY_TOKENS || String(CONSTANTS.DEFAULT_MAX_ENTITY_TOKENS),
  10,
);
const MAX_RELATION_TOKENS = parseInt(
  process.env.RAG_MAX_RELATION_TOKENS || String(CONSTANTS.DEFAULT_MAX_RELATION_TOKENS),
  10,
);
const MAX_TOTAL_TOKENS = parseInt(
  process.env.RAG_MAX_TOTAL_TOKENS || String(CONSTANTS.DEFAULT_MAX_TOTAL_TOKENS),
  10,
);

export const chatRoutes: FastifyPluginAsync = async (fastify) => {
  fastify.post(
    '/completions',
    {
      preHandler: requirePermissionMiddleware('chat:create'),
    },
    async (request, reply) => {
      const orgId = request.user?.orgId || request.user?.id;
      if (!orgId) {
        return reply
          .status(403)
          .send({ error: { code: 'FORBIDDEN', message: 'User does not belong to an organization' } });
      }

      const userId = request.user?.id;
      if (!userId) {
        return reply.status(403).send({ error: { code: 'FORBIDDEN', message: 'User ID is required' } });
      }

      const body = (request.body ?? {}) as {
        messages?: unknown;
        stream?: boolean;
        conversationId?: unknown;
        documentIds?: unknown;
        mode?: unknown;
      };
      const { stream } = body;
      const rawMessages = body.messages;
      if (!Array.isArray(rawMessages) || rawMessages.length === 0) {
        return reply
          .status(400)
          .send({ error: { code: 'BAD_REQUEST', message: 'Missing or empty messages array' } });
      }
      if (!rawMessages.every(isIncomingMessage)) {
        return reply
          .status(400)
          .send({ error: { code: 'BAD_REQUEST', message: 'Each message needs a string role and content' } });
      }
      const incomingMessages: IncomingMessage[] = rawMessages;

      let ragMode: RagQueryMode = DEFAULT_RAG_MODE;
      if (body.mode !== undefined) {
        if (!isRagQueryMode(body.mode)) {
          return reply.status(400).send({
            error: {
              code: 'BAD_REQUEST',
              message: 'Invalid mode. Expected one of: auto, mix, hybrid, local, global, naive',
            },
          });
        }
        ragMode = body.mode;
      }

      if (
        body.conversationId !== undefined &&
        (typeof body.conversationId !== 'string' || !UUID_RE.test(body.conversationId))
      ) {
        return reply
          .status(400)
          .send({ error: { code: 'BAD_REQUEST', message: 'Invalid conversation ID format' } });
      }
      let conversationId = body.conversationId as string | undefined;
      let isNewConversation = false;

      let documentIds: string[] | undefined;
      if (body.documentIds !== undefined) {
        if (!Array.isArray(body.documentIds)) {
          return reply
            .status(400)
            .send({ error: { code: 'BAD_REQUEST', message: 'documentIds must be an array' } });
        }
        if (!body.documentIds.every((id): id is string => typeof id === 'string' && UUID_RE.test(id))) {
          return reply
            .status(400)
            .send({ error: { code: 'BAD_REQUEST', message: 'Invalid document ID format' } });
        }
        documentIds = body.documentIds;
      }

      if (conversationId) {
        const existing = await db.query.conversations.findFirst({
          where: and(
            eq(conversations.id, conversationId),
            eq(conversations.orgId, orgId),
            eq(conversations.userId, userId as string),
          ),
        });
        if (!existing) {
          request.log.info({ conversationId }, 'Returning 404 because conversation was not found in DB');
          return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Conversation not found' } });
        }
      } else {
        conversationId = crypto.randomUUID();
        isNewConversation = true;
      }

      const lastMessage = incomingMessages.at(-1);
      if (!lastMessage || lastMessage.role !== 'user') {
        return reply
          .status(400)
          .send({ error: { code: 'BAD_REQUEST', message: 'Last message must be from user' } });
      }

      // Rate limiting
      const limit = await consumeRateLimit(
        redisConnection,
        `rate-limit:chat:${orgId}:${userId}`,
        CHAT_RATE_LIMIT_PER_MINUTE,
        60,
      );
      if (!limit.allowed) {
        return reply.status(429).send({
          error: { code: 'TOO_MANY_REQUESTS', message: 'Rate limit exceeded. Please try again in a minute.' },
        });
      }

      const userQuery = lastMessage.content;
      const priorHistory = incomingMessages.slice(0, -1);

      const trace = getLangfuse().trace({
        name: 'chat-completion',
        sessionId: conversationId,
        userId: userId as string,
        tags: ['RAG', `mode:${ragMode}`],
        input: userQuery,
      });

      const rootSpan = trace.span({
        name: 'chat-interaction',
        input: userQuery,
      });

      // Cache key includes org version, document scope, RAG mode, AND recent turn context
      // (prevents follow-up messages like "Why?" from hitting unrelated conversations' cache!)
      const docIdsStr =
        Array.isArray(documentIds) && documentIds.length ? [...documentIds].sort().join(',') : 'all';
      const historyFingerprint =
        priorHistory.length > 0
          ? priorHistory
              .slice(-4)
              .map((m) => `${m.role}:${m.content.trim()}`)
              .join('|')
          : '';
      const queryHash = crypto
        .createHash('sha256')
        .update(`${ragMode}|${historyFingerprint}|${userQuery.toLowerCase().trim()}`)
        .digest('hex');

      let cacheVersion = '0';
      try {
        cacheVersion = (await redisConnection.get(`rag:cache-version:${orgId}`)) ?? '0';
      } catch {
        /* cache is best-effort */
      }
      const cacheKey = `cache:chat:org:${orgId}:v${cacheVersion}:mode:${ragMode}:docs:${docIdsStr}:query:${queryHash}`;

      try {
        const cachedResult = await redisConnection.get(cacheKey);
        if (cachedResult) {
          const parsedCache = JSON.parse(cachedResult) as {
            content: string;
            citations: unknown;
            retrievalMeta?: unknown;
          };

          try {
            if (isNewConversation) {
              await db.insert(conversations).values({
                id: conversationId,
                orgId,
                userId: userId as string,
                title: userQuery.trim().slice(0, 60) || 'New Conversation',
              });
            }
            await db.insert(messages).values([
              { id: crypto.randomUUID(), conversationId, orgId, role: 'user', content: userQuery },
              {
                id: crypto.randomUUID(),
                conversationId,
                orgId,
                role: 'assistant',
                content: parsedCache.content,
                model: 'cache',
                tokenCount: 0,
                latencyMs: 0,
              },
            ]);
          } catch (persistErr) {
            request.log.error(persistErr, 'Failed to persist cached chat turn');
          }

          request.log.info({ cacheKey }, 'Query Cache Hit');
          ragCacheHits.inc();

          if (stream) {
            reply.hijack();
            reply.raw.writeHead(200, {
              'Content-Type': 'text/event-stream',
              'Cache-Control': 'no-cache',
              Connection: 'keep-alive',
              ...sseCorsHeaders(request.headers.origin),
            });

            reply.raw.write(
              `data: ${JSON.stringify({ type: 'meta', conversationId, retrieval: parsedCache.retrievalMeta })}\n\n`,
            );
            reply.raw.write(`data: ${JSON.stringify({ type: 'citations', citations: parsedCache.citations })}\n\n`);
            reply.raw.write(`data: ${JSON.stringify({ type: 'chunk', content: parsedCache.content })}\n\n`);
            reply.raw.write(`data: [DONE]\n\n`);
            reply.raw.end();

            rootSpan.end({ output: parsedCache.content });
            trace.update({ output: parsedCache.content });
            flushLangfuse();
            return;
          } else {
            rootSpan.end({ output: parsedCache.content });
            trace.update({ output: parsedCache.content });
            flushLangfuse();
            return reply.status(200).send({
              id: crypto.randomUUID(),
              conversationId,
              role: 'assistant',
              content: parsedCache.content,
              citations: parsedCache.citations,
              retrieval: parsedCache.retrievalMeta,
              cached: true,
            });
          }
        }
      } catch (err) {
        request.log.warn(err, 'Cache check failed, proceeding to RAG pipeline');
      }

      // Load tenant's active RetrievalLearningEngine configuration (<2ms from Redis)
      const activeConfig = await learningEngine.getActiveConfig(orgId);

      // LightRAG V2 Adaptive Retrieval (Intent Routing + PPR + Temporal/Authority + LTR + CRAG + Parent-Child)
      const retrievalSpan = rootSpan.span({ name: 'lightrag-retrieval', input: { userQuery, mode: ragMode } });
      const retriever = new LightRagRetriever({
        chunkRepo,
        kgRepo,
        embedder: getEmbeddingProvider(),
        keywordsLlm: getKeywordsLlmProvider(),
        reranker: getRerankerProvider(),
        fastCache: redisConnection,
        communityRepo,
      });

      const retrieval = await retriever.retrieve({
        orgId,
        userQuery,
        history: priorHistory,
        mode: ragMode,
        documentIds,
        rerankLimit: RERANK_LIMIT,
        rerankMinScore: RERANK_MIN_SCORE,
        retrievalConfig: activeConfig.config,
        budget: {
          maxEntityTokens: MAX_ENTITY_TOKENS,
          maxRelationTokens: MAX_RELATION_TOKENS,
          maxTotalTokens: MAX_TOTAL_TOKENS,
        },
      });

      ragPipelineDuration.observe({ stage: 'query_analysis' }, retrieval.timingsMs.queryAnalysis / 1000);
      ragPipelineDuration.observe({ stage: 'retrieval' }, (retrieval.timingsMs.embedding + retrieval.timingsMs.retrieval) / 1000);
      ragPipelineDuration.observe({ stage: 'rerank' }, retrieval.timingsMs.rerank / 1000);

      const { assembled } = retrieval;
      const retrievalMeta = {
        mode: retrieval.effectiveMode,
        requestedMode: retrieval.requestedMode,
        routingReason: retrieval.routingReason,
        intent: retrieval.intent,
        strategy: retrieval.strategy,
        configVersion: retrieval.configVersion,
        crag: {
          verdict: retrieval.crag.verdict,
          action: retrieval.crag.action,
          confidence: Number(retrieval.crag.confidence.toFixed(3)),
        },
        entitiesCount: assembled.entitiesUsed.length,
        relationsCount: assembled.relationsUsed.length,
        chunksCount: assembled.chunksUsed.length,
        estimatedTokens: assembled.estimatedTokens.total,
        keywords: {
          highLevel: retrieval.analysis.highLevelKeywords,
          lowLevel: retrieval.analysis.lowLevelKeywords,
        },
      };

      retrievalSpan.end({ output: retrievalMeta });
      request.log.info(
        {
          mode: retrieval.effectiveMode,
          intent: retrieval.intent,
          strategy: retrieval.strategy,
          crag: retrieval.crag.verdict,
          entities: assembled.entitiesUsed.length,
          relations: assembled.relationsUsed.length,
          chunks: assembled.chunksUsed.length,
          timingsMs: retrieval.timingsMs,
        },
        'LightRAG V2 retrieval complete',
      );

      // Fire-and-forget async shadow evaluation on a tiny sampled fraction (0ms user latency, 0 gen tokens)
      if (learningEngine.shouldRunShadow(activeConfig.config)) {
        void learningEngine
          .runShadowComparison({
            orgId,
            query: userQuery,
            retriever,
            baselineResult: retrieval,
            documentIds,
          })
          .catch(() => undefined);
      }

      const baseSystemPrompt = buildLightRagSystemPrompt({
        contextText: assembled.contextText,
        hasGraphData: assembled.entitiesUsed.length > 0 || assembled.relationsUsed.length > 0,
      });
      const systemPrompt =
        retrieval.crag.action === 'abstain_hint'
          ? `${baseSystemPrompt}\n\n${CRAG_LOW_CONFIDENCE_DIRECTIVE}`
          : baseSystemPrompt;

      const queryLlm = getQueryLlmProvider();
      const mainGen = rootSpan.generation({
        name: 'main-llm-response',
        model: queryLlm.model,
        input: systemPrompt,
      });

      const llmMessages: LLMMessage[] = [
        { role: 'system', content: systemPrompt },
        ...incomingMessages.map((m) => ({
          role: (m.role === 'user' ? 'user' : 'assistant') as 'user' | 'assistant',
          content: m.content,
        })),
      ];

      const userMessageId = crypto.randomUUID();
      const assistantMessageId = crypto.randomUUID();

      if (isNewConversation) {
        try {
          await db.insert(conversations).values({
            id: conversationId,
            orgId,
            userId: userId as string,
            title: 'New Conversation',
          });
        } catch (err) {
          request.log.error(err, 'Failed to create conversation entry in DB');
        }
      }

      const saveToDb = async (
        fullContent: string,
        usage: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number } | undefined,
        latencyMs: number,
      ) => {
        try {
          await db.insert(messages).values({
            id: userMessageId,
            conversationId,
            orgId,
            role: 'user',
            content: userQuery,
          });

          await db.insert(messages).values({
            id: assistantMessageId,
            conversationId,
            orgId,
            role: 'assistant',
            content: fullContent,
            model: queryLlm.model,
            tokenCount: usage?.total_tokens || 0,
            latencyMs,
            metadata: { retrieval: retrievalMeta },
          });

          // Non-blocking compact telemetry for RetrievalLearningEngine
          void learningEngine
            .recordQueryTelemetry({
              orgId,
              messageId: assistantMessageId,
              queryText: userQuery,
              retrieval,
              promptTokens: usage?.prompt_tokens ?? 0,
              completionTokens: usage?.completion_tokens ?? 0,
            })
            .catch(() => undefined);

          if (isNewConversation) {
            getKeywordsLlmProvider()
              .generate([
                {
                  role: 'user',
                  content: `Generate a very short, concise title (max 5 words) for a conversation that starts with this message: "${userQuery}". Do not use quotes or prefixes.`,
                },
              ])
              .then(async (titleResponse) => {
                const title = titleResponse.content.trim().replace(/^["']|["']$/g, '');
                if (title) {
                  await db.update(conversations).set({ title }).where(eq(conversations.id, conversationId));
                }
              })
              .catch((titleErr) => {
                request.log.warn(titleErr, 'Failed to generate title');
              });
          }

          if (assembled.chunksUsed.length > 0) {
            const usedCitationIndices = new Set<number>();
            const matches = fullContent.matchAll(/\[Citation\s*([\d,\s]+)\]/gi);
            for (const match of matches) {
              if (match[1]) {
                for (const part of match[1].split(',')) {
                  const idx = parseInt(part.trim(), 10);
                  if (Number.isFinite(idx) && idx > 0) usedCitationIndices.add(idx);
                }
              }
            }

            const matchedChunks = assembled.chunksUsed
              .map((chunk, i) => ({ chunk, index: i + 1 }))
              .filter((item) => usedCitationIndices.has(item.index));

            // Fallback to top retrieved chunks when LLM answers without explicit [Citation N] brackets
            const usedChunks =
              matchedChunks.length > 0
                ? matchedChunks
                : assembled.chunksUsed.slice(0, 5).map((chunk, i) => ({ chunk, index: i + 1 }));

            if (usedChunks.length > 0) {
              const citationRecords = usedChunks.map(({ chunk, index }) => ({
                id: crypto.randomUUID(),
                messageId: assistantMessageId,
                chunkId: chunk.id,
                documentId: chunk.documentId,
                orgId,
                citationIndex: index,
                relevanceScore: chunk.similarity,
                snippet: chunk.content.substring(0, 200),
                pageNumber: chunk.pageNumber ?? null,
                sectionHeading: chunk.sectionHeading ?? null,
              }));
              await db.insert(citations).values(citationRecords);
            }
          }
        } catch (err) {
          request.log.error(err, 'Failed to save chat to DB');
        }
      };

      const saveToCache = async (content: string, citationsOutput: unknown) => {
        try {
          const cacheData = JSON.stringify({ content, citations: citationsOutput, retrievalMeta });
          await redisConnection.setex(cacheKey, 3600, cacheData);
        } catch (err) {
          request.log.warn(err, 'Failed to save to Redis cache');
        }
      };

      const citationsOutput = assembled.chunksUsed.map((chunk, index) => ({
        index: index + 1,
        document: chunk.documentTitle,
        documentId: chunk.documentId,
        fileName: chunk.fileName,
        fileType: chunk.fileType,
        score: chunk.similarity,
        pageNumber: chunk.pageNumber ?? undefined,
        sectionHeading: chunk.sectionHeading ?? undefined,
        snippet: chunk.content.substring(0, 120) + '...',
      }));

      if (stream) {
        reply.hijack();
        reply.raw.writeHead(200, {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          Connection: 'keep-alive',
          ...sseCorsHeaders(request.headers.origin),
        });

        reply.raw.write(
          `data: ${JSON.stringify({ type: 'meta', conversationId, retrieval: retrievalMeta })}\n\n`,
        );
        reply.raw.write(`data: ${JSON.stringify({ type: 'citations', citations: citationsOutput })}\n\n`);

        let fullContent = '';
        let finalUsage = undefined;
        const startMs = Date.now();

        try {
          const streamIterable = queryLlm.generateStream(llmMessages);
          for await (const chunk of streamIterable) {
            if (chunk.content) {
              fullContent += chunk.content;
              reply.raw.write(`data: ${JSON.stringify({ type: 'chunk', content: chunk.content })}\n\n`);
            }
            if (chunk.usage) {
              finalUsage = chunk.usage;
            }
          }
        } catch (err) {
          request.log.error(err, 'Streaming failed');
          const errMsg = err instanceof Error && err.message ? err.message : 'Generation failed';
          reply.raw.write(`data: ${JSON.stringify({ type: 'error', message: errMsg })}\n\n`);
        }

        const latencyMs = Date.now() - startMs;
        ragPipelineDuration.observe({ stage: 'generation' }, latencyMs / 1000);
        mainGen.end({
          output: fullContent,
          usage: finalUsage
            ? {
                promptTokens: finalUsage.prompt_tokens,
                completionTokens: finalUsage.completion_tokens,
                totalTokens: finalUsage.total_tokens,
              }
            : undefined,
        });
        rootSpan.end({ output: fullContent });
        trace.update({ output: fullContent });
        flushLangfuse();
        await saveToDb(fullContent, finalUsage, latencyMs);
        if (fullContent.trim()) {
          await saveToCache(fullContent, citationsOutput);
        }

        reply.raw.write(`data: [DONE]\n\n`);
        reply.raw.end();
        return;
      }

      // Non-streaming fallback
      const llmResponse = await queryLlm.generate(llmMessages);
      ragPipelineDuration.observe({ stage: 'generation' }, llmResponse.latency_ms / 1000);
      await saveToDb(llmResponse.content, llmResponse.usage, llmResponse.latency_ms);
      mainGen.end({
        output: llmResponse.content,
        usage: llmResponse.usage
          ? {
              promptTokens: llmResponse.usage.prompt_tokens,
              completionTokens: llmResponse.usage.completion_tokens,
              totalTokens: llmResponse.usage.total_tokens,
            }
          : undefined,
      });
      rootSpan.end({ output: llmResponse.content });
      trace.update({ output: llmResponse.content });
      flushLangfuse();

      if (llmResponse.content.trim()) {
        await saveToCache(llmResponse.content, citationsOutput);
      }

      return reply.status(200).send({
        id: assistantMessageId,
        conversationId,
        role: 'assistant',
        content: llmResponse.content,
        citations: citationsOutput,
        retrieval: retrievalMeta,
        usage: llmResponse.usage,
      });
    },
  );
};
