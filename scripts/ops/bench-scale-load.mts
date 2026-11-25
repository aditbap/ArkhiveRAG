/**
 * V2.6 Phase 4 — Production Infrastructure Scale & Concurrency Load Benchmark
 *
 * Benchmarks real PostgreSQL + pgvector HNSW (1024-dim normalized vectors) +
 * migration 0014 stored `content_tsv` GIN FTS + 2-hop KG PPR across:
 *   - Corpus Scale Ladder: 10,000 -> 50,000 -> 100,000 documents
 *   - Concurrency Waves:   10, 50, 100, 250, 500 concurrent virtual users
 *   - Mixed Workload:      Simultaneous reader burst + transactional `replaceVersionChunks` ingestion
 *   - Telemetry:           p50 / p95 / p99 latencies, DB pool saturation (`pg_stat_activity`), disk & RAM
 *
 * Runs inside an isolated benchmark org and cleans up completely in `finally`.
 */
import crypto from 'node:crypto';
import fs from 'node:fs';
import { performance } from 'node:perf_hooks';
import { execSync } from 'node:child_process';
import * as dotenv from 'dotenv';
import {
  ChunkRepository,
  KnowledgeGraphRepository,
  getDatabasePoolStats,
  queryClient,
} from '../../packages/database/dist/index.js';
import {
  computePersonalizedPageRank,
  propagatePprToChunks,
  weightedReciprocalRankFusion,
} from '../../packages/core/dist/index.js';

dotenv.config();

const BENCH_ORG_ID = '00000000-0000-4000-8000-000000000026';
const EMBED_DIM = 1024;
const TIERS = (process.env.BENCH_TIERS || '10000,50000,100000')
  .split(',')
  .map((s) => parseInt(s.trim(), 10))
  .filter((n) => Number.isFinite(n) && n > 0);
const VU_LEVELS = (process.env.BENCH_VUS || '10,50,100,250,500')
  .split(',')
  .map((s) => parseInt(s.trim(), 10))
  .filter((n) => Number.isFinite(n) && n > 0);
const REPORT_OUT = process.env.BENCH_OUT || '';

const TOPICS = [
  { code: 'POL-SEC-409', dept: 'Security', text: 'Zero-trust MFA hardware token rotation and production SSH bastion audit policy' },
  { code: 'POL-TRV-500', dept: 'Finance', text: 'Corporate travel hotel reimbursement cap and international per diem expense policy' },
  { code: 'SOP-JKT-882', dept: 'Logistics', text: 'Jakarta port fleet manifest scanning OCR retention and customs clearance procedure' },
  { code: 'INC-2026-014', dept: 'SRE', text: 'Production PostgreSQL pgvector latency spike postmortem and connection pool remediation' },
  { code: 'SLA-ENT-2026', dept: 'Enterprise', text: 'Enterprise workspace 99.95 uptime guarantee PagerDuty escalation and service credit schedule' },
  { code: 'POL-IAM-770', dept: 'IAM', text: 'Vault-KMS-Prod cryptographic key envelope rotation and least-privilege RBAC mandate' },
  { code: 'SOP-OPS-200', dept: 'Operations', text: 'Disaster recovery pg_dump backup verification MinIO replication and rollback checklist' },
  { code: 'ARC-RAG-901', dept: 'AI', text: 'Hybrid BM25 dense retrieval Personalized PageRank knowledge graph and citation grounding' },
  { code: 'POL-PRV-310', dept: 'Legal', text: 'GDPR data residency tenant isolation retention purge and audit trail compliance' },
  { code: 'RUN-K8S-112', dept: 'Platform', text: 'Redis BullMQ worker graceful shutdown stalled job recovery and horizontal scaling' },
];

const QUERIES = [
  'What does POL-SEC-409 require for hardware MFA token rotation?',
  'Current POL-TRV-500 hotel reimbursement cap and expense policy',
  'Jakarta port manifest scanning OCR retention SOP-JKT-882',
  'Root cause of INC-2026-014 pgvector connection pool latency spike',
  'Enterprise uptime SLA-ENT-2026 PagerDuty escalation and service credits',
  'Vault-KMS-Prod key rotation mandate under POL-IAM-770',
  'Disaster recovery pg_dump backup and rollback SOP-OPS-200',
  'Hybrid BM25 Personalized PageRank citation grounding ARC-RAG-901',
];

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normalize(vec: Float32Array): Float32Array {
  let sum = 0;
  for (let i = 0; i < vec.length; i++) sum += vec[i]! * vec[i]!;
  const norm = Math.sqrt(sum) || 1;
  for (let i = 0; i < vec.length; i++) vec[i] = vec[i]! / norm;
  return vec;
}

const rng = mulberry32(20261125);
const topicCentroids: Float32Array[] = TOPICS.map(() => {
  const v = new Float32Array(EMBED_DIM);
  for (let i = 0; i < EMBED_DIM; i++) v[i] = rng() * 2 - 1;
  return normalize(v);
});

function makeClusteredVector(topicIdx: number): string {
  const c = topicCentroids[topicIdx % topicCentroids.length]!;
  const v = new Float32Array(EMBED_DIM);
  for (let i = 0; i < EMBED_DIM; i++) {
    v[i] = c[i]! * 0.85 + (rng() * 2 - 1) * 0.15;
  }
  normalize(v);
  // 4 decimals keeps payload compact (~7KB vs 20KB) with <1e-4 cosine error
  const parts = new Array<string>(EMBED_DIM);
  for (let i = 0; i < EMBED_DIM; i++) parts[i] = v[i]!.toFixed(4);
  return `[${parts.join(',')}]`;
}

function makeQueryVector(topicIdx: number): number[] {
  const c = topicCentroids[topicIdx % topicCentroids.length]!;
  const v = new Float32Array(EMBED_DIM);
  for (let i = 0; i < EMBED_DIM; i++) {
    v[i] = c[i]! * 0.9 + (rng() * 2 - 1) * 0.1;
  }
  normalize(v);
  return Array.from(v);
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * p));
  return Number((sorted[idx] ?? 0).toFixed(2));
}

function summarizeLatencies(samples: number[]) {
  const s = [...samples].sort((a, b) => a - b);
  return {
    p50: percentile(s, 0.5),
    p95: percentile(s, 0.95),
    p99: percentile(s, 0.99),
    max: Number((s[s.length - 1] ?? 0).toFixed(2)),
  };
}

function getPostgresContainerRam(): string {
  try {
    return execSync('docker stats --no-stream --format "{{.MemUsage}}" rag-postgres', {
      encoding: 'utf-8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }).trim();
  } catch {
    return 'unknown';
  }
}

async function seedBenchOrg() {
  await queryClient`
    INSERT INTO organizations (id, name, slug, kind, is_active, max_documents, max_storage_mb)
    VALUES (${BENCH_ORG_ID}::uuid, 'V2.6 Scale Bench Org', 'v26-scale-bench', 'shared', true, 500000, 100000)
    ON CONFLICT (id) DO NOTHING
  `;
}

async function seedKnowledgeGraph(sampleChunkIds: string[], sampleDocIds: string[]) {
  const entityCount = 400;
  const edgeCount = 900;
  const entNames = Array.from({ length: entityCount }, (_, i) => ({
    canonical: i < TOPICS.length ? TOPICS[i]!.code.toLowerCase() : `service-node-${i}`,
    name: i < TOPICS.length ? TOPICS[i]!.code : `ServiceNode-${i}`,
    type: i < TOPICS.length ? 'Policy' : 'System',
  }));

  for (let i = 0; i < entNames.length; i += 100) {
    const batch = entNames.slice(i, i + 100);
    const ids = batch.map(() => crypto.randomUUID());
    const names = batch.map((e) => e.name);
    const canonicals = batch.map((e) => e.canonical);
    const types = batch.map((e) => e.type);
    const descs = batch.map((e, idx) => `Enterprise node ${e.name} in cluster ${(i + idx) % TOPICS.length}`);
    const vecs = batch.map((_, idx) => makeClusteredVector((i + idx) % TOPICS.length));
    const cids = batch.map((_, idx) => `{${sampleChunkIds[(i + idx) % sampleChunkIds.length]!}}`);
    const dids = batch.map((_, idx) => `{${sampleDocIds[(i + idx) % sampleDocIds.length]!}}`);

    await queryClient.unsafe(
      `INSERT INTO kg_entities (id, org_id, entity_name, canonical_name, entity_type, description, embedding, source_chunk_ids, document_ids, degree)
       SELECT u.id::uuid, $1::uuid, u.name, u.canonical, u.etype, u.descr, u.emb::vector, u.cids::uuid[], u.dids::uuid[], 4
       FROM unnest($2::text[], $3::text[], $4::text[], $5::text[], $6::text[], $7::text[], $8::text[], $9::text[])
         AS u(id, name, canonical, etype, descr, emb, cids, dids)
       ON CONFLICT (org_id, canonical_name) DO NOTHING`,
      [BENCH_ORG_ID, ids, names, canonicals, types, descs, vecs, cids, dids],
    );
  }

  const edgeRows: Array<{
    id: string;
    src: string;
    tgt: string;
    srcC: string;
    tgtC: string;
    kw: string;
    desc: string;
    vec: string;
    cids: string;
    dids: string;
  }> = [];
  const seenEdges = new Set<string>();
  for (let i = 0; i < edgeCount; i++) {
    const aIdx = i < 120 ? i % TOPICS.length : Math.floor(rng() * entityCount);
    let bIdx = (aIdx + 1 + Math.floor(rng() * 18)) % entityCount;
    if (aIdx === bIdx) bIdx = (bIdx + 1) % entityCount;
    let a = entNames[aIdx]!;
    let b = entNames[bIdx]!;
    if (a.canonical > b.canonical) [a, b] = [b, a];
    const key = `${a.canonical}::${b.canonical}`;
    if (seenEdges.has(key)) continue;
    seenEdges.add(key);
    edgeRows.push({
      id: crypto.randomUUID(),
      src: a.name,
      tgt: b.name,
      srcC: a.canonical,
      tgtC: b.canonical,
      kw: 'governs, depends_on',
      desc: `${a.name} connects to ${b.name} for enterprise compliance`,
      vec: makeClusteredVector(aIdx % TOPICS.length),
      cids: `{${sampleChunkIds[i % sampleChunkIds.length]!}}`,
      dids: `{${sampleDocIds[i % sampleDocIds.length]!}}`,
    });
  }

  for (let i = 0; i < edgeRows.length; i += 100) {
    const batch = edgeRows.slice(i, i + 100);
    await queryClient.unsafe(
      `INSERT INTO kg_relations (id, org_id, source_entity, target_entity, source_canonical, target_canonical, keywords, description, weight, embedding, source_chunk_ids, document_ids)
       SELECT u.id::uuid, $1::uuid, u.src, u.tgt, u.src_c, u.tgt_c, u.kw, u.descr, 1.0, u.emb::vector, u.cids::uuid[], u.dids::uuid[]
       FROM unnest($2::text[], $3::text[], $4::text[], $5::text[], $6::text[], $7::text[], $8::text[], $9::text[], $10::text[], $11::text[])
         AS u(id, src, tgt, src_c, tgt_c, kw, descr, emb, cids, dids)
       ON CONFLICT (org_id, source_canonical, target_canonical) DO NOTHING`,
      [
        BENCH_ORG_ID,
        batch.map((e) => e.id),
        batch.map((e) => e.src),
        batch.map((e) => e.tgt),
        batch.map((e) => e.srcC),
        batch.map((e) => e.tgtC),
        batch.map((e) => e.kw),
        batch.map((e) => e.desc),
        batch.map((e) => e.vec),
        batch.map((e) => e.cids),
        batch.map((e) => e.dids),
      ],
    );
  }
}

async function insertSyntheticBatch(startIdx: number, count: number): Promise<{ docIds: string[]; chunkIds: string[] }> {
  const batchSize = 250;
  const sampleDocIds: string[] = [];
  const sampleChunkIds: string[] = [];

  for (let offset = 0; offset < count; offset += batchSize) {
    const n = Math.min(batchSize, count - offset);
    const docIds: string[] = [];
    const verIds: string[] = [];
    const chunkIds: string[] = [];
    const titles: string[] = [];
    const fileNames: string[] = [];
    const hashes: string[] = [];
    const contents: string[] = [];
    const embeddings: string[] = [];

    for (let i = 0; i < n; i++) {
      const idx = startIdx + offset + i;
      const tIdx = idx % TOPICS.length;
      const topic = TOPICS[tIdx]!;
      const docId = crypto.randomUUID();
      const verId = crypto.randomUUID();
      const chunkId = crypto.randomUUID();
      const content = `${topic.code} (${topic.dept} Doc #${idx}): ${topic.text}. Control revision R-${idx % 500} applies to cluster-${idx % 40} with verified audit checksum ${idx}.`;

      docIds.push(docId);
      verIds.push(verId);
      chunkIds.push(chunkId);
      titles.push(`${topic.code} ${topic.dept} Guide #${idx}`);
      fileNames.push(`${topic.code.toLowerCase()}-${idx}.md`);
      hashes.push(crypto.createHash('sha256').update(content).digest('hex'));
      contents.push(content);
      embeddings.push(makeClusteredVector(tIdx));

      if (sampleDocIds.length < 50) {
        sampleDocIds.push(docId);
        sampleChunkIds.push(chunkId);
      }
    }

    await queryClient.unsafe(
      `INSERT INTO documents (id, org_id, title, file_name, file_type, file_size_bytes, file_hash, status, governance_status, authority_tier, effective_at)
       SELECT u.id::uuid, $1::uuid, u.title, u.fname, 'md', 1024, u.fhash, 'ready', 'active', 3, NOW()
       FROM unnest($2::text[], $3::text[], $4::text[], $5::text[]) AS u(id, title, fname, fhash)`,
      [BENCH_ORG_ID, docIds, titles, fileNames, hashes],
    );

    await queryClient.unsafe(
      `INSERT INTO document_versions (id, document_id, org_id, version_number, file_path, file_hash, file_size_bytes)
       SELECT u.vid::uuid, u.did::uuid, $1::uuid, 1, 'bench/' || u.vid || '.md', u.fhash, 1024
       FROM unnest($2::text[], $3::text[], $4::text[]) AS u(vid, did, fhash)`,
      [BENCH_ORG_ID, verIds, docIds, hashes],
    );

    await queryClient.unsafe(
      `INSERT INTO document_chunks (id, org_id, document_id, version_id, chunk_index, content, content_hash, token_count, chunk_role, embedding, embedding_model, is_active)
       SELECT u.cid::uuid, $1::uuid, u.did::uuid, u.vid::uuid, 0, u.content, u.fhash, 35, 'standalone', u.emb::vector, 'synthetic-1024', true
       FROM unnest($2::text[], $3::text[], $4::text[], $5::text[], $6::text[], $7::text[]) AS u(cid, did, vid, content, fhash, emb)`,
      [BENCH_ORG_ID, chunkIds, docIds, verIds, contents, hashes, embeddings],
    );
  }

  return { docIds: sampleDocIds, chunkIds: sampleChunkIds };
}

async function cleanupBenchOrg() {
  console.log('\n[Cleanup] Removing isolated benchmark organization & reclaiming space...');
  await queryClient`DELETE FROM kg_relations WHERE org_id = ${BENCH_ORG_ID}::uuid`;
  await queryClient`DELETE FROM kg_entities WHERE org_id = ${BENCH_ORG_ID}::uuid`;
  await queryClient`DELETE FROM document_chunks WHERE org_id = ${BENCH_ORG_ID}::uuid`;
  await queryClient`DELETE FROM documents WHERE org_id = ${BENCH_ORG_ID}::uuid`;
  await queryClient`DELETE FROM organizations WHERE id = ${BENCH_ORG_ID}::uuid`;
  await queryClient`ANALYZE document_chunks`;
}

async function main() {
  const chunkRepo = new ChunkRepository();
  const kgRepo = new KnowledgeGraphRepository();
  const queryVecs = QUERIES.map((_, i) => makeQueryVector(i % TOPICS.length));

  const report: Record<string, unknown> = {
    startedAt: new Date().toISOString(),
    embedDimensions: EMBED_DIM,
    tiers: [],
    concurrency: [],
    mixedWorkload: null,
  };

  try {
    await cleanupBenchOrg().catch(() => undefined);
    await seedBenchOrg();

    let currentCount = 0;
    let kgSeeded = false;

    for (const targetDocs of TIERS) {
      const toAdd = targetDocs - currentCount;
      console.log(`\n=== Scale Tier: ${targetDocs.toLocaleString()} Documents (+${toAdd.toLocaleString()} new 1024-dim chunks) ===`);
      const insStart = performance.now();
      const samples = await insertSyntheticBatch(currentCount, toAdd);
      const insSec = (performance.now() - insStart) / 1000;
      currentCount = targetDocs;

      if (!kgSeeded && samples.chunkIds.length > 0) {
        await seedKnowledgeGraph(samples.chunkIds, samples.docIds);
        kgSeeded = true;
      }

      await queryClient`ANALYZE document_chunks`;
      await queryClient`ANALYZE documents`;

      // Warmup
      await chunkRepo.searchSimilar(BENCH_ORG_ID, queryVecs[0]!, 10, 0.1);
      await chunkRepo.searchFullText(BENCH_ORG_ID, QUERIES[0]!, 10);

      const hnswLats: number[] = [];
      const ftsLats: number[] = [];
      const hybridLats: number[] = [];
      const pprLats: number[] = [];
      const rounds = 24;

      for (let i = 0; i < rounds; i++) {
        const qIdx = i % QUERIES.length;
        const qText = QUERIES[qIdx]!;
        const qVec = queryVecs[qIdx]!;

        const t0 = performance.now();
        await chunkRepo.searchSimilar(BENCH_ORG_ID, qVec, 10, 0.1);
        hnswLats.push(performance.now() - t0);

        const t1 = performance.now();
        await chunkRepo.searchFullText(BENCH_ORG_ID, qText, 10);
        ftsLats.push(performance.now() - t1);

        const t2 = performance.now();
        const [dRes, fRes] = await Promise.all([
          chunkRepo.searchSimilar(BENCH_ORG_ID, qVec, 10, 0.1),
          chunkRepo.searchFullText(BENCH_ORG_ID, qText, 10),
        ]);
        weightedReciprocalRankFusion(
          [{ items: dRes, weight: 1.0 }, { items: fRes, weight: 1.0 }],
          (x) => x.chunk.id,
          60,
        );
        hybridLats.push(performance.now() - t2);

        const t3 = performance.now();
        const seedCanonical = TOPICS[qIdx % TOPICS.length]!.code.toLowerCase();
        const sub = await kgRepo.getTenantSubgraph(BENCH_ORG_ID, [seedCanonical], { maxHops: 2, maxEdges: 240 });
        const ppr = computePersonalizedPageRank(sub.edges, new Map([[seedCanonical, 1.0]]), {
          damping: 0.85,
          iterations: 12,
        });
        propagatePprToChunks({ nodeScores: ppr.nodeScores, entityChunks: sub.entityChunks, edges: sub.edges });
        pprLats.push(performance.now() - t3);
      }

      const [sizeRow] = await queryClient<Array<{ total_mb: string; table_mb: string; index_mb: string }>>`
        SELECT
          round(pg_total_relation_size('document_chunks') / 1048576.0, 1)::text AS total_mb,
          round(pg_relation_size('document_chunks') / 1048576.0, 1)::text AS table_mb,
          round(pg_indexes_size('document_chunks') / 1048576.0, 1)::text AS index_mb
      `;
      const ram = getPostgresContainerRam();

      const tierResult = {
        docs: targetDocs,
        insertThroughputDocsPerSec: Number((toAdd / insSec).toFixed(1)),
        insertElapsedSec: Number(insSec.toFixed(1)),
        hnswMs: summarizeLatencies(hnswLats),
        ftsStoredTsvMs: summarizeLatencies(ftsLats),
        hybridRrfMs: summarizeLatencies(hybridLats),
        ppr2HopMs: summarizeLatencies(pprLats),
        totalDiskMb: Number(sizeRow?.total_mb ?? 0),
        tableMb: Number(sizeRow?.table_mb ?? 0),
        indexMb: Number(sizeRow?.index_mb ?? 0),
        postgresRam: ram,
      };
      (report.tiers as unknown[]).push(tierResult);
      console.log(
        `  [${targetDocs.toLocaleString()} docs] Insert=${tierResult.insertThroughputDocsPerSec} docs/s | HNSW p50/p95/p99=${tierResult.hnswMs.p50}/${tierResult.hnswMs.p95}/${tierResult.hnswMs.p99}ms | FTS(0014) p50/p95/p99=${tierResult.ftsStoredTsvMs.p50}/${tierResult.ftsStoredTsvMs.p95}/${tierResult.ftsStoredTsvMs.p99}ms | Hybrid p50/p95/p99=${tierResult.hybridRrfMs.p50}/${tierResult.hybridRrfMs.p95}/${tierResult.hybridRrfMs.p99}ms | PPR p50=${tierResult.ppr2HopMs.p50}ms | Disk=${tierResult.totalDiskMb}MB | RAM=${ram}`,
      );
    }

    // Concurrency waves at peak corpus size
    console.log(`\n=== Concurrency Waves at ${currentCount.toLocaleString()} Documents ===`);
    for (const vus of VU_LEVELS) {
      const lats: number[] = [];
      let errors = 0;
      let peakActive = 0;
      let peakWaiting = 0;
      let peakTotal = 0;

      const sampler = setInterval(async () => {
        try {
          const st = await getDatabasePoolStats();
          peakActive = Math.max(peakActive, st.active);
          peakWaiting = Math.max(peakWaiting, st.waiting);
          peakTotal = Math.max(peakTotal, st.total);
        } catch {
          /* ignore transient sample error */
        }
      }, 50);

      const waveStart = performance.now();
      await Promise.all(
        Array.from({ length: vus }, async (_, idx) => {
          const qIdx = idx % QUERIES.length;
          const t0 = performance.now();
          try {
            const [dRes, fRes] = await Promise.all([
              chunkRepo.searchSimilar(BENCH_ORG_ID, queryVecs[qIdx]!, 10, 0.1),
              chunkRepo.searchFullText(BENCH_ORG_ID, QUERIES[qIdx]!, 10),
            ]);
            weightedReciprocalRankFusion(
              [{ items: dRes, weight: 1.0 }, { items: fRes, weight: 1.0 }],
              (x) => x.chunk.id,
              60,
            );
            lats.push(performance.now() - t0);
          } catch {
            errors++;
          }
        }),
      );
      clearInterval(sampler);
      const waveMs = performance.now() - waveStart;
      const st = await getDatabasePoolStats().catch(() => ({ active: 0, waiting: 0, total: 0, poolMax: 25 }));
      peakActive = Math.max(peakActive, st.active);
      peakWaiting = Math.max(peakWaiting, st.waiting);
      peakTotal = Math.max(peakTotal, st.total);

      const waveResult = {
        concurrentUsers: vus,
        totalRequests: vus,
        errors,
        errorRatePct: Number(((errors / vus) * 100).toFixed(2)),
        throughputReqPerSec: Number(((vus / waveMs) * 1000).toFixed(1)),
        latencyMs: summarizeLatencies(lats),
        dbPoolPeak: { active: peakActive, waiting: peakWaiting, total: peakTotal, poolMax: st.poolMax },
      };
      (report.concurrency as unknown[]).push(waveResult);
      console.log(
        `  [${String(vus).padStart(3)} VUs] errors=${errors}/${vus} (${waveResult.errorRatePct}%) | throughput=${waveResult.throughputReqPerSec} req/s | p50/p95/p99=${waveResult.latencyMs.p50}/${waveResult.latencyMs.p95}/${waveResult.latencyMs.p99}ms | DB peak active=${peakActive} waiting=${peakWaiting} total=${peakTotal}/${st.poolMax}`,
      );
    }

    // Mixed concurrent ingestion (transactional replaceVersionChunks) + retrieval burst
    console.log(`\n=== Mixed Workload: 50 Concurrent Readers + 10 Concurrent Transactional Writers ===`);
    const readerLats: number[] = [];
    const writerLats: number[] = [];
    let readerErrors = 0;
    let writerErrors = 0;

    // Pre-create 10 doc+version shells for transactional replaceVersionChunks writes
    const writeTargets: Array<{ docId: string; verId: string }> = [];
    for (let i = 0; i < 10; i++) {
      const docId = crypto.randomUUID();
      const verId = crypto.randomUUID();
      await queryClient`
        INSERT INTO documents (id, org_id, title, file_name, file_type, file_size_bytes, file_hash, status, governance_status, authority_tier, effective_at)
        VALUES (${docId}::uuid, ${BENCH_ORG_ID}::uuid, ${`Mixed Write Doc ${i}`}, ${`mixed-${i}.md`}, 'md', 512, ${`hash-${i}`}, 'processing', 'active', 3, NOW())
      `;
      await queryClient`
        INSERT INTO document_versions (id, document_id, org_id, version_number, file_path, file_hash, file_size_bytes)
        VALUES (${verId}::uuid, ${docId}::uuid, ${BENCH_ORG_ID}::uuid, 1, ${`bench/mixed-${i}.md`}, ${`hash-${i}`}, 512)
      `;
      writeTargets.push({ docId, verId });
    }

    await Promise.all([
      ...Array.from({ length: 50 }, async (_, idx) => {
        const qIdx = idx % QUERIES.length;
        const t0 = performance.now();
        try {
          await Promise.all([
            chunkRepo.searchSimilar(BENCH_ORG_ID, queryVecs[qIdx]!, 10, 0.1),
            chunkRepo.searchFullText(BENCH_ORG_ID, QUERIES[qIdx]!, 10),
          ]);
          readerLats.push(performance.now() - t0);
        } catch {
          readerErrors++;
        }
      }),
      ...writeTargets.map(async (target, idx) => {
        const t0 = performance.now();
        try {
          const children = Array.from({ length: 5 }, (_, cIdx) => ({
            text: `Mixed ingestion transactional chunk ${idx}-${cIdx} for POL-SEC-409 validation.`,
            embedding: makeQueryVector((idx + cIdx) % TOPICS.length),
            chunkRole: 'standalone' as const,
            pageNumber: cIdx + 1,
          }));
          // Call replaceVersionChunks TWICE to also prove zero-duplicate idempotency under live load!
          await chunkRepo.replaceVersionChunks(BENCH_ORG_ID, target.docId, target.verId, [], children, 'synthetic-1024');
          await chunkRepo.replaceVersionChunks(BENCH_ORG_ID, target.docId, target.verId, [], children, 'synthetic-1024');
          writerLats.push(performance.now() - t0);
        } catch {
          writerErrors++;
        }
      }),
    ]);

    // Verify exact chunk count = 10 * 5 = 50 (0 duplicates despite double-running every write!)
    const [cntRow] = await queryClient<Array<{ cnt: string }>>`
      SELECT count(*)::text AS cnt FROM document_chunks
      WHERE org_id = ${BENCH_ORG_ID}::uuid AND document_id = ANY(${writeTargets.map((t) => t.docId)}::uuid[])
    `;
    const actualMixedChunks = parseInt(cntRow?.cnt ?? '0', 10);

    report.mixedWorkload = {
      readers: { total: 50, errors: readerErrors, latencyMs: summarizeLatencies(readerLats) },
      writers: {
        total: 10,
        idempotentDoubleRuns: 20,
        errors: writerErrors,
        latencyMs: summarizeLatencies(writerLats),
        expectedChunks: 50,
        actualChunks: actualMixedChunks,
        duplicateChunks: actualMixedChunks - 50,
      },
    };
    console.log(
      `  Readers (50): errors=${readerErrors}, p50/p95/p99=${summarizeLatencies(readerLats).p50}/${summarizeLatencies(readerLats).p95}/${summarizeLatencies(readerLats).p99}ms`,
    );
    console.log(
      `  Writers (10x2 idempotent tx): errors=${writerErrors}, p50/p95=${summarizeLatencies(writerLats).p50}/${summarizeLatencies(writerLats).p95}ms, chunks=${actualMixedChunks}/50 (0 duplicates=${actualMixedChunks === 50})`,
    );

    report.finishedAt = new Date().toISOString();
    if (REPORT_OUT) {
      fs.writeFileSync(REPORT_OUT, JSON.stringify(report, null, 2));
      console.log(`\nSaved Phase 4 benchmark JSON to ${REPORT_OUT}`);
    }
  } finally {
    await cleanupBenchOrg().catch((err) => console.error('Cleanup error:', err));
    await queryClient.end({ timeout: 5 });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
