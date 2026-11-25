import crypto from 'crypto';
import { Worker, Job, UnrecoverableError } from 'bullmq';
import { INGESTION_QUEUE_NAME, redisConnection } from './queue.js';
import { workerMetrics } from './health-server.js';
import {
  JobRepository,
  ChunkRepository,
  DocumentRepository,
  KnowledgeGraphRepository,
  type InsertChunkInput,
} from '@rag/database';
import {
  PdfParser,
  DocxParser,
  TxtParser,
  HtmlParser,
  RecursiveChunker,
  createEmbeddingProvider,
  MinioStorageProvider,
  PaddleOcrProvider,
  needsOcr,
  scanBuffer,
  KnowledgeGraphExtractor,
  createLLMForRole,
  hasLLMConfigured,
} from '@rag/providers';
import { CONSTANTS, retryWithBackoff, type ParsedDocument } from '@rag/core';

const IMAGE_TYPES = ['jpeg', 'png', 'webp'];
const ENABLE_VIRUS_SCAN = process.env.ENABLE_VIRUS_SCAN === 'true';
const ENABLE_KG_INDEXING = process.env.ENABLE_KG_INDEXING !== 'false';
const KG_STRICT_INDEXING = process.env.KG_STRICT_INDEXING === 'true';
const ENABLE_PARENT_CHILD_CHUNKING = process.env.ENABLE_PARENT_CHILD_CHUNKING !== 'false';
const CHUNK_SIZE = parseInt(process.env.CHUNK_SIZE || String(CONSTANTS.DEFAULT_CHUNK_SIZE), 10);
const CHUNK_OVERLAP = parseInt(process.env.CHUNK_OVERLAP || String(CONSTANTS.CHUNK_OVERLAP), 10);
const STORAGE_DOWNLOAD_TIMEOUT_MS = Math.max(
  2000,
  parseInt(process.env.STORAGE_DOWNLOAD_TIMEOUT_MS || '30000', 10) || 30000,
);

const jobRepo = new JobRepository();
const chunkRepo = new ChunkRepository();
const docRepo = new DocumentRepository();
const kgRepo = new KnowledgeGraphRepository();
const storage = MinioStorageProvider.fromEnv();

async function streamToBufferWithTimeout(
  stream: NodeJS.ReadableStream,
  timeoutMs: number,
): Promise<Buffer> {
  return new Promise<Buffer>((resolve, reject) => {
    const chunks: Buffer[] = [];
    const timer = setTimeout(() => {
      if ('destroy' in stream && typeof stream.destroy === 'function') {
        stream.destroy(new Error(`Storage download timed out after ${timeoutMs}ms`));
      }
      reject(new Error(`Storage download timed out after ${timeoutMs}ms`));
    }, timeoutMs);
    timer.unref();

    stream.on('data', (chunk) => chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)));
    stream.on('end', () => {
      clearTimeout(timer);
      resolve(Buffer.concat(chunks));
    });
    stream.on('error', (err) => {
      clearTimeout(timer);
      reject(err);
    });
  });
}

async function downloadObjectSafely(objectKey: string): Promise<Buffer> {
  return retryWithBackoff(
    async () => {
      const stream = await Promise.race([
        storage.download(objectKey),
        new Promise<never>((_, rej) =>
          setTimeout(() => rej(new Error(`Storage open timed out after ${STORAGE_DOWNLOAD_TIMEOUT_MS}ms`)), STORAGE_DOWNLOAD_TIMEOUT_MS),
        ),
      ]);
      return streamToBufferWithTimeout(stream, STORAGE_DOWNLOAD_TIMEOUT_MS);
    },
    { maxAttempts: 2, baseDelayMs: 500 },
  );
}

export function isFinalJobAttempt(
  error: unknown,
  attemptsMade: number,
  maxAttempts?: number,
): boolean {
  if (error instanceof UnrecoverableError) return true;
  const total = Math.max(1, maxAttempts ?? 1);
  return attemptsMade + 1 >= total;
}

export const WORKER_BULLMQ_OPTIONS = {
  concurrency: parseInt(process.env.WORKER_CONCURRENCY || '5', 10),
  lockDuration: 120_000,
  stalledInterval: 30_000,
  maxStalledCount: 2,
};

export interface IngestionJobData {
  jobId: string;
  documentId: string;
  versionId: string;
  orgId: string;
  filePath: string;
  fileType: string;
  traceId?: string;
}

export const ingestionWorker = new Worker<IngestionJobData>(
  INGESTION_QUEUE_NAME,
  async (job: Job<IngestionJobData>) => {
    const { jobId, documentId, orgId, fileType, filePath: objectKey, traceId } = job.data;
    const logPrefix = traceId ? `[Worker][trace=${traceId}]` : '[Worker]';
    const jobStartMs = Date.now();
    workerMetrics.activeJobs++;

    console.log(`${logPrefix} Starting job ${jobId} (attempt ${job.attemptsMade + 1}/${job.opts.attempts ?? 1}) for document ${documentId}`);

    try {
      await docRepo.updateStatus(orgId, documentId, 'processing');
      const buffer = await downloadObjectSafely(objectKey);

      // 0. Virus scan (opt-in via ENABLE_VIRUS_SCAN)
      if (ENABLE_VIRUS_SCAN) {
        await jobRepo.updateStatus(orgId, jobId, 'parsing', 'scanning_for_viruses', {});
        const scanResult = await scanBuffer(buffer);
        if (scanResult.infected) {
          await storage
            .delete(objectKey)
            .catch((err) => console.error(`[Worker] Could not delete rejected file ${objectKey}:`, err));
          throw new UnrecoverableError(
            `File rejected: virus scan matched signature "${scanResult.signature}"`,
          );
        }
      }

      // 1. Parsing Phase
      await jobRepo.updateStatus(orgId, jobId, 'parsing', 'parsing_document', { chunks_total: 0 });
      console.log(`[Worker] Parsing ${objectKey}...`);

      let parsed: ParsedDocument;
      if (IMAGE_TYPES.includes(fileType)) {
        await jobRepo.updateStatus(orgId, jobId, 'parsing', 'ocr_processing', {});
        parsed = await new PaddleOcrProvider().run(buffer, `document.${fileType}`);
      } else {
        let parser;
        if (fileType === 'pdf') parser = new PdfParser();
        else if (fileType === 'docx') parser = new DocxParser();
        else if (fileType === 'html') parser = new HtmlParser();
        else parser = new TxtParser();

        parsed = await parser.parse(buffer, fileType);

        if (fileType === 'pdf' && needsOcr(parsed)) {
          console.log(`[Worker] PDF text layer too sparse, routing to OCR...`);
          await jobRepo.updateStatus(orgId, jobId, 'parsing', 'ocr_processing', {});
          parsed = await new PaddleOcrProvider().run(buffer, 'document.pdf');
        }
      }

      // 2. Structure-, Semantic- & Parent-Child-aware Chunking Phase (#7)
      await jobRepo.updateStatus(orgId, jobId, 'chunking', 'creating_chunks', { chunks_total: 0 });
      console.log(
        `[Worker] Chunking document (size=${CHUNK_SIZE}, overlap=${CHUNK_OVERLAP}, parentChild=${ENABLE_PARENT_CHILD_CHUNKING})...`,
      );

      const chunker = new RecursiveChunker(CHUNK_SIZE, CHUNK_OVERLAP);
      const batch = ENABLE_PARENT_CHILD_CHUNKING
        ? await chunker.chunkDocumentParentChild(parsed)
        : { parents: [], children: [], standalone: await chunker.chunkDocument(parsed) };

      const searchableChunks = [...batch.children, ...batch.standalone];
      const chunks_total = searchableChunks.length;

      if (chunks_total === 0) {
        throw new Error(
          'No text could be extracted from this document (it may be a scanned/image-only file or empty)',
        );
      }

      // 3. Embedding Phase (only child & standalone chunks are embedded; parents are zero-embed containers)
      await jobRepo.updateStatus(orgId, jobId, 'embedding', 'generating_embeddings', {
        chunks_total,
        chunks_embedded: 0,
      });
      console.log(
        `[Worker] Embedding ${chunks_total} searchable chunks (${batch.parents.length} parents, ${batch.children.length} children, ${batch.standalone.length} standalone)...`,
      );

      const embedder = createEmbeddingProvider();
      const hashes = searchableChunks.map((c) => crypto.createHash('sha256').update(c.text).digest('hex'));
      const existingEmbeddings = await chunkRepo
        .findExistingEmbeddingsByHashes(orgId, hashes, embedder.model)
        .catch(() => new Map<string, number[]>());

      // Collect unique texts whose embeddings do not exist yet
      const missingByHash = new Map<string, string>();
      for (let i = 0; i < searchableChunks.length; i++) {
        const hash = hashes[i];
        const chunk = searchableChunks[i];
        if (hash && chunk && !existingEmbeddings.has(hash) && !missingByHash.has(hash)) {
          missingByHash.set(hash, chunk.text);
        }
      }

      const missingEntries = [...missingByHash.entries()];
      const reusedCount = chunks_total - missingEntries.length;
      if (reusedCount > 0) {
        console.log(`[Worker] Reusing ${reusedCount}/${chunks_total} cached chunk embeddings`);
      }

      const batchSize = 64;
      let embeddedMissing = 0;
      for (let i = 0; i < missingEntries.length; i += batchSize) {
        const slice = missingEntries.slice(i, i + batchSize);
        const result = await embedder.embed(slice.map(([, text]) => text));
        for (let j = 0; j < slice.length; j++) {
          const entry = slice[j];
          const vec = result.embeddings[j];
          if (entry && vec) {
            existingEmbeddings.set(entry[0], vec);
          }
        }
        embeddedMissing += slice.length;
        await jobRepo.updateStatus(orgId, jobId, 'embedding', 'generating_embeddings', {
          chunks_total,
          chunks_embedded: Math.min(chunks_total, reusedCount + embeddedMissing),
        });
      }

      // 4. Crash-safe, idempotent atomic Indexing Phase (delete partial version rows -> insert parents -> insert children -> deactivate older versions)
      await jobRepo.updateStatus(orgId, jobId, 'indexing', 'saving_to_pgvector', {
        chunks_total,
        chunks_indexed: 0,
      });

      const versionId = job.data.versionId;
      const parentInputs: InsertChunkInput[] = batch.parents.map((p) => ({
        text: p.text,
        embedding: null,
        pageNumber: p.pageNumber,
        sectionHeading: p.sectionHeading,
        chunkRole: 'parent' as const,
        metadata: p.metadata,
      }));

      const chunksWithEmbeddings: InsertChunkInput[] = [];
      for (let i = 0; i < searchableChunks.length; i++) {
        const chunk = searchableChunks[i];
        const hash = hashes[i];
        const embedding = hash ? existingEmbeddings.get(hash) : undefined;
        if (chunk && embedding) {
          chunksWithEmbeddings.push({
            text: chunk.text,
            embedding,
            pageNumber: chunk.pageNumber,
            sectionHeading: chunk.sectionHeading,
            parentIndex: chunk.chunkRole === 'child' ? (chunk.parentIndex ?? null) : null,
            chunkRole: chunk.chunkRole ?? 'standalone',
            metadata: chunk.metadata,
          });
        }
      }

      console.log(
        `${logPrefix} Atomically replacing ${chunksWithEmbeddings.length} embedded chunks (+ ${parentInputs.length} parents) for version ${versionId}...`,
      );
      const { insertedChunks } = await chunkRepo.replaceVersionChunks(
        orgId,
        documentId,
        versionId,
        parentInputs,
        chunksWithEmbeddings,
        embedder.model,
      );

      // 5. LightRAG Knowledge Graph Extraction & Incremental Indexing Phase
      let kgStats: { entitiesUpserted: number; relationsUpserted: number; cacheHits: number; skippedUnchanged?: boolean } | null = null;
      if (ENABLE_KG_INDEXING && hasLLMConfigured()) {
        const docRecord = await docRepo.findById(orgId, documentId).catch(() => null);
        const alreadyIndexedHash = docRecord?.kgIndexedHash ?? null;

        // O(1) document-hash short-circuit: if this exact file content already has KG indexed,
        // skip KG re-extraction completely (0 LLM calls, 0 KG embedding calls).
        if (docRecord?.fileHash && alreadyIndexedHash && alreadyIndexedHash === docRecord.fileHash) {
          console.log(`[Worker] Skipping KG extraction for ${documentId}: fileHash unchanged (${alreadyIndexedHash.slice(0, 12)})`);
          kgStats = { entitiesUpserted: 0, relationsUpserted: 0, cacheHits: insertedChunks.length, skippedUnchanged: true };
        } else {
          await jobRepo.updateStatus(orgId, jobId, 'indexing', 'extracting_knowledge_graph', {
            chunks_total,
            chunks_indexed: insertedChunks.length,
            kg_chunks_processed: 0,
          });
          console.log(`[Worker] Extracting Knowledge Graph for ${insertedChunks.length} chunks...`);

          try {
            // Clean any prior version's KG references for this document before upserting the new version
            await kgRepo.removeDocumentFromGraph(orgId, documentId);

            const extractor = new KnowledgeGraphExtractor({
              extractLlm: createLLMForRole('extract'),
              summaryLlm: createLLMForRole('summary'),
              embedder,
              bridge: kgRepo,
            });

            const summary = await extractor.extractAndIndexDocument(
              orgId,
              documentId,
              insertedChunks.map((c) => ({
                id: c.id,
                text: c.content,
                sectionHeading: c.sectionHeading,
              })),
              {
                onProgress: async ({ chunksProcessed }) => {
                  if (chunksProcessed % 5 === 0 || chunksProcessed === insertedChunks.length) {
                    await jobRepo
                      .updateStatus(orgId, jobId, 'indexing', 'extracting_knowledge_graph', {
                        chunks_total,
                        chunks_indexed: insertedChunks.length,
                        kg_chunks_processed: chunksProcessed,
                      })
                      .catch(() => undefined);
                  }
                },
              },
            );
            if (docRecord?.fileHash) {
              await kgRepo.setDocumentKgHash(orgId, documentId, docRecord.fileHash).catch(() => undefined);
            }
            kgStats = {
              entitiesUpserted: summary.entitiesUpserted,
              relationsUpserted: summary.relationsUpserted,
              cacheHits: summary.cacheHits,
            };
            console.log(
              `[Worker] KG indexed for ${documentId}: ${summary.entitiesUpserted} entities, ${summary.relationsUpserted} relations (${summary.cacheHits} cache hits, ${summary.llmCalls} LLM calls)`,
            );
          } catch (kgErr) {
            console.warn(
              `[Worker] Knowledge Graph extraction failed for ${documentId} (chunks remain searchable):`,
              kgErr instanceof Error ? kgErr.message : kgErr,
            );
            if (KG_STRICT_INDEXING) throw kgErr;
          }
        }
      } else if (ENABLE_KG_INDEXING) {
        console.warn('[Worker] Skipping KG extraction: no LLM provider credentials configured in worker env');
      }

      // 6. Completion
      await jobRepo.updateStatus(orgId, jobId, 'completed', 'done', {
        chunks_total,
        chunks_indexed: insertedChunks.length,
        ...(kgStats
          ? {
              kg_entities: kgStats.entitiesUpserted,
              kg_relations: kgStats.relationsUpserted,
              kg_cache_hits: kgStats.cacheHits,
            }
          : {}),
      });
      await docRepo.updateStatus(orgId, documentId, 'ready');
      await redisConnection.incr(`rag:cache-version:${orgId}`).catch(() => undefined);
      console.log(`${logPrefix} Job ${jobId} completed successfully.`);
      workerMetrics.completedJobs++;
      workerMetrics.lastJobDurationSeconds = (Date.now() - jobStartMs) / 1000;
    } catch (error) {
      const finalAttempt = isFinalJobAttempt(error, job.attemptsMade, job.opts.attempts);
      if (finalAttempt) workerMetrics.failedJobs++;
      else workerMetrics.retriedJobs++;
      workerMetrics.lastJobDurationSeconds = (Date.now() - jobStartMs) / 1000;
      console.error(
        `${logPrefix} Job ${jobId} failed (attempt ${job.attemptsMade + 1}/${job.opts.attempts ?? 1}, final=${finalAttempt}):`,
        error,
      );
      if (error instanceof Error) {
        if (finalAttempt) {
          await jobRepo.failJob(orgId, jobId, error).catch(() => undefined);
          await docRepo.updateStatus(orgId, documentId, 'failed', error.message).catch(() => undefined);
        } else {
          await jobRepo
            .updateStatus(orgId, jobId, 'pending', 'retrying_after_transient_error', {
              last_error: error.message,
              attempt: job.attemptsMade + 1,
            })
            .catch(() => undefined);
        }
      }
      throw error;
    } finally {
      workerMetrics.activeJobs = Math.max(0, workerMetrics.activeJobs - 1);
    }
  },
  {
    connection: redisConnection,
    ...WORKER_BULLMQ_OPTIONS,
  },
);

ingestionWorker.on('stalled', (jobId) => {
  workerMetrics.stalledJobs++;
  console.warn(`[Worker] Reclaimed stalled BullMQ job ${jobId}`);
});

ingestionWorker.on('completed', (job) => {
  console.log(`Job ${job.id} has completed!`);
});

ingestionWorker.on('failed', (job, err) => {
  console.log(`Job ${job?.id} has failed with ${err.message}`);
});
