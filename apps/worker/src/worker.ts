import crypto from 'crypto';
import { Worker, Job, UnrecoverableError } from 'bullmq';
import { INGESTION_QUEUE_NAME, redisConnection } from './queue.js';
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
import { CONSTANTS, type ParsedDocument } from '@rag/core';

const IMAGE_TYPES = ['jpeg', 'png', 'webp'];
const ENABLE_VIRUS_SCAN = process.env.ENABLE_VIRUS_SCAN === 'true';
const ENABLE_KG_INDEXING = process.env.ENABLE_KG_INDEXING !== 'false';
const KG_STRICT_INDEXING = process.env.KG_STRICT_INDEXING === 'true';
const ENABLE_PARENT_CHILD_CHUNKING = process.env.ENABLE_PARENT_CHILD_CHUNKING !== 'false';
const CHUNK_SIZE = parseInt(process.env.CHUNK_SIZE || String(CONSTANTS.DEFAULT_CHUNK_SIZE), 10);
const CHUNK_OVERLAP = parseInt(process.env.CHUNK_OVERLAP || String(CONSTANTS.CHUNK_OVERLAP), 10);

const jobRepo = new JobRepository();
const chunkRepo = new ChunkRepository();
const docRepo = new DocumentRepository();
const kgRepo = new KnowledgeGraphRepository();
const storage = MinioStorageProvider.fromEnv();

async function streamToBuffer(stream: NodeJS.ReadableStream): Promise<Buffer> {
  const chunks: Buffer[] = [];
  for await (const chunk of stream) {
    chunks.push(chunk as Buffer);
  }
  return Buffer.concat(chunks);
}

export interface IngestionJobData {
  jobId: string;
  documentId: string;
  versionId: string;
  orgId: string;
  filePath: string;
  fileType: string;
}

export const ingestionWorker = new Worker<IngestionJobData>(
  INGESTION_QUEUE_NAME,
  async (job: Job<IngestionJobData>) => {
    const { jobId, documentId, orgId, fileType, filePath: objectKey } = job.data;

    console.log(`[Worker] Starting job ${jobId} for document ${documentId}`);

    try {
      await docRepo.updateStatus(orgId, documentId, 'processing');
      const buffer = await streamToBuffer(await storage.download(objectKey));

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

      // 4. Indexing Phase (Save parent containers first so child rows can link via parent_chunk_id)
      await jobRepo.updateStatus(orgId, jobId, 'indexing', 'saving_to_pgvector', {
        chunks_total,
        chunks_indexed: 0,
      });

      const versionId = job.data.versionId;
      const insertedParents =
        batch.parents.length > 0
          ? await chunkRepo.insertChunks(
              orgId,
              documentId,
              versionId,
              batch.parents.map((p) => ({
                text: p.text,
                embedding: null,
                pageNumber: p.pageNumber,
                sectionHeading: p.sectionHeading,
                chunkRole: 'parent' as const,
                metadata: p.metadata,
              })),
              embedder.model,
            )
          : [];

      const chunksWithEmbeddings: InsertChunkInput[] = [];
      for (let i = 0; i < searchableChunks.length; i++) {
        const chunk = searchableChunks[i];
        const hash = hashes[i];
        const embedding = hash ? existingEmbeddings.get(hash) : undefined;
        if (chunk && embedding) {
          const parentRow =
            chunk.chunkRole === 'child' && chunk.parentIndex != null
              ? insertedParents[chunk.parentIndex]
              : undefined;
          chunksWithEmbeddings.push({
            text: chunk.text,
            embedding,
            pageNumber: chunk.pageNumber,
            sectionHeading: chunk.sectionHeading,
            parentChunkId: parentRow?.id ?? null,
            chunkRole: chunk.chunkRole ?? 'standalone',
            metadata: chunk.metadata,
          });
        }
      }

      console.log(
        `[Worker] Saving ${chunksWithEmbeddings.length} embedded chunks (+ ${insertedParents.length} parents) to Postgres...`,
      );
      const insertedChunks = await chunkRepo.insertChunks(
        orgId,
        documentId,
        versionId,
        chunksWithEmbeddings,
        embedder.model,
      );
      await chunkRepo.deactivateOtherVersions(orgId, documentId, versionId);

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
      console.log(`[Worker] Job ${jobId} completed successfully.`);
    } catch (error) {
      console.error(`[Worker] Job ${jobId} failed:`, error);
      if (error instanceof Error) {
        await jobRepo.failJob(orgId, jobId, error);
        await docRepo.updateStatus(orgId, documentId, 'failed', error.message);
      }
      throw error;
    }
  },
  {
    connection: redisConnection,
    concurrency: parseInt(process.env.WORKER_CONCURRENCY || '5', 10),
  },
);

ingestionWorker.on('completed', (job) => {
  console.log(`Job ${job.id} has completed!`);
});

ingestionWorker.on('failed', (job, err) => {
  console.log(`Job ${job?.id} has failed with ${err.message}`);
});
