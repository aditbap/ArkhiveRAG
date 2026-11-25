/**
 * V2.7 Phase 5 — Automated PostgreSQL EXPLAIN (ANALYZE, BUFFERS) Plan Regression Gate
 *
 * Verifies that:
 *   1. Migration 0015 index hygiene is active (single tenant-scoped composite GIN FTS index,
 *      obsolete redundant GIN indexes dropped, partial HNSW index present).
 *   2. `ChunkRepository.searchSimilar` uses `Index Scan` on `idx_chunks_embedding_hnsw`
 *      (fails non-zero if PostgreSQL falls back to `Seq Scan` on `document_chunks`).
 *   3. `ChunkRepository.searchFullText` uses `Bitmap Index Scan` on `idx_chunks_org_content_tsv_gin`
 *      (fails non-zero if PostgreSQL falls back to `Seq Scan` on `document_chunks`).
 *   4. Retrieval rows omit the heavy `embedding` column (lean wire projection).
 *   5. Reports PostgreSQL memory/SSD planner configuration (`shared_buffers`, `random_page_cost`, etc.).
 *
 * Usage:
 *   npx tsx scripts/ops/verify-query-plans.mts                  # seeds isolated 5,000-doc org & cleans up
 *   npx tsx scripts/ops/verify-query-plans.mts --org=<uuid>     # inspects plans against an existing org
 */
import crypto from 'node:crypto';
import * as dotenv from 'dotenv';
import {
  ChunkRepository,
  queryClient,
  supportsHnswIterativeScan,
} from '../../packages/database/dist/index.js';

dotenv.config();

const DEFAULT_ORG_ID = '00000000-0000-4000-8000-000000000027';
const EMBED_DIM = 1024;
const SEED_COUNT = Math.max(1000, parseInt(process.env.BENCH_PLAN_DOCS || '5000', 10) || 5000);

interface PlanNode {
  'Node Type': string;
  'Relation Name'?: string;
  'Index Name'?: string;
  'Actual Total Time'?: number;
  'Actual Rows'?: number;
  'Shared Hit Blocks'?: number;
  'Shared Read Blocks'?: number;
  Plans?: PlanNode[];
}

function collectNodes(node: PlanNode, out: PlanNode[] = []): PlanNode[] {
  out.push(node);
  if (Array.isArray(node.Plans)) {
    for (const child of node.Plans) collectNodes(child, out);
  }
  return out;
}

function makeNormalizedVector(seed: number): number[] {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const v = new Array<number>(EMBED_DIM);
  let sum = 0;
  for (let i = 0; i < EMBED_DIM; i++) {
    const val = next() * 2 - 1;
    v[i] = val;
    sum += val * val;
  }
  const norm = Math.sqrt(sum) || 1;
  for (let i = 0; i < EMBED_DIM; i++) v[i] = Number((v[i]! / norm).toFixed(4));
  return v;
}

async function seedIsolatedOrg(orgId: string, count: number) {
  console.log(`[PlanCheck] Seeding ${count.toLocaleString()} synthetic chunks in isolated org ${orgId}...`);
  await queryClient`
    INSERT INTO organizations (id, name, slug, kind, is_active, max_documents, max_storage_mb)
    VALUES (${orgId}::uuid, 'V2.7 Plan Verifier Org', 'v27-plan-verify', 'shared', true, 100000, 10000)
    ON CONFLICT (id) DO NOTHING
  `;

  const topics = [
    'POL-SEC-409 Zero-trust hardware MFA token rotation and SSH bastion audit policy',
    'POL-TRV-500 Corporate travel hotel reimbursement cap and international per diem expense policy',
    'SOP-JKT-882 Jakarta port fleet manifest scanning OCR retention and customs clearance procedure',
    'INC-2026-014 Production PostgreSQL pgvector HNSW index tuning and PgBouncer remediation',
    'SLA-ENT-2026 Enterprise workspace 99.95 uptime guarantee PagerDuty escalation schedule',
  ];

  const batchSize = 250;
  for (let offset = 0; offset < count; offset += batchSize) {
    const n = Math.min(batchSize, count - offset);
    const docIds: string[] = [];
    const verIds: string[] = [];
    const chunkIds: string[] = [];
    const titles: string[] = [];
    const fnames: string[] = [];
    const hashes: string[] = [];
    const contents: string[] = [];
    const vecs: string[] = [];

    for (let i = 0; i < n; i++) {
      const idx = offset + i;
      const text = `${topics[idx % topics.length]} revision R-${idx % 200} checksum-${idx}`;
      docIds.push(crypto.randomUUID());
      verIds.push(crypto.randomUUID());
      chunkIds.push(crypto.randomUUID());
      titles.push(`Policy Guide #${idx}`);
      fnames.push(`policy-${idx}.md`);
      hashes.push(crypto.createHash('sha256').update(text).digest('hex'));
      contents.push(text);
      vecs.push(`[${makeNormalizedVector(1000 + (idx % 50)).join(',')}]`);
    }

    await queryClient.unsafe(
      `INSERT INTO documents (id, org_id, title, file_name, file_type, file_size_bytes, file_hash, status, governance_status, authority_tier, effective_at)
       SELECT u.id::uuid, $1::uuid, u.title, u.fname, 'md', 1024, u.fhash, 'ready', 'active', 3, NOW()
       FROM unnest($2::text[], $3::text[], $4::text[], $5::text[]) AS u(id, title, fname, fhash)`,
      [orgId, docIds, titles, fnames, hashes],
    );

    await queryClient.unsafe(
      `INSERT INTO document_versions (id, document_id, org_id, version_number, file_path, file_hash, file_size_bytes)
       SELECT u.vid::uuid, u.did::uuid, $1::uuid, 1, 'plan/' || u.vid || '.md', u.fhash, 1024
       FROM unnest($2::text[], $3::text[], $4::text[]) AS u(vid, did, fhash)`,
      [orgId, verIds, docIds, hashes],
    );

    await queryClient.unsafe(
      `INSERT INTO document_chunks (id, org_id, document_id, version_id, chunk_index, content, content_hash, token_count, chunk_role, embedding, embedding_model, is_active)
       SELECT u.cid::uuid, $1::uuid, u.did::uuid, u.vid::uuid, 0, u.content, u.fhash, 30, 'standalone', u.emb::vector, 'synthetic-1024', true
       FROM unnest($2::text[], $3::text[], $4::text[], $5::text[], $6::text[], $7::text[]) AS u(cid, did, vid, content, fhash, emb)`,
      [orgId, chunkIds, docIds, verIds, contents, hashes, vecs],
    );
  }

  await queryClient`ANALYZE documents`;
  await queryClient`ANALYZE document_chunks`;
}

async function cleanupIsolatedOrg(orgId: string) {
  await queryClient`DELETE FROM document_chunks WHERE org_id = ${orgId}::uuid`;
  await queryClient`DELETE FROM documents WHERE org_id = ${orgId}::uuid`;
  await queryClient`DELETE FROM organizations WHERE id = ${orgId}::uuid`;
  await queryClient`ANALYZE document_chunks`;
}

async function main() {
  const orgArg = process.argv.find((a) => a.startsWith('--org='))?.split('=')[1];
  const orgId = orgArg || DEFAULT_ORG_ID;
  const shouldCleanup = !orgArg;
  const failures: string[] = [];

  try {
    // 1. Verify PostgreSQL tuning settings & extension capabilities
    const settings = await queryClient<Array<{ name: string; setting: string; unit: string | null }>>`
      SELECT name, setting, unit
      FROM pg_settings
      WHERE name IN (
        'shared_buffers', 'effective_cache_size', 'work_mem', 'maintenance_work_mem',
        'random_page_cost', 'effective_io_concurrency', 'max_connections', 'shared_preload_libraries'
      )
      ORDER BY name
    `;
    const iterativeSupported = await supportsHnswIterativeScan(queryClient);
    console.log('\n=== PostgreSQL Runtime Configuration ===');
    for (const s of settings) {
      console.log(`  ${s.name.padEnd(26)} = ${s.setting}${s.unit ? ` (${s.unit})` : ''}`);
    }
    console.log(`  ${'hnsw.iterative_scan'.padEnd(26)} = ${iterativeSupported ? 'SUPPORTED (pgvector >= 0.8)' : 'not available'}`);

    // 2. Verify Migration 0015 index hygiene on document_chunks
    const indexes = await queryClient<Array<{ indexname: string; indexdef: string }>>`
      SELECT indexname, indexdef
      FROM pg_indexes
      WHERE tablename = 'document_chunks'
      ORDER BY indexname
    `;
    const idxNames = new Set(indexes.map((i) => i.indexname));
    console.log('\n=== document_chunks Indexes ===');
    for (const idx of indexes) {
      console.log(`  - ${idx.indexname}`);
    }
    for (const required of ['idx_chunks_embedding_hnsw', 'idx_chunks_org_content_tsv_gin']) {
      if (!idxNames.has(required)) {
        failures.push(`Missing required index: ${required}`);
      }
    }
    for (const obsolete of ['idx_chunks_content_fts', 'idx_chunks_content_fts_simple', 'idx_chunks_content_tsv_gin']) {
      if (idxNames.has(obsolete)) {
        failures.push(`Obsolete redundant FTS index still present: ${obsolete}`);
      }
    }

    if (shouldCleanup) {
      await seedIsolatedOrg(orgId, SEED_COUNT);
    }

    const qVec = makeNormalizedVector(1005);
    const vecLiteral = `[${qVec.join(',')}]`;

    // 3. EXPLAIN (ANALYZE, BUFFERS) — HNSW Vector Search
    const hnswExplain = await queryClient.begin(async (tx) => {
      await tx.unsafe(`SET LOCAL hnsw.ef_search = 64`);
      if (iterativeSupported) {
        await tx.unsafe(`SET LOCAL hnsw.iterative_scan = 'relaxed_order'`);
      }
      const rows = await tx.unsafe<Array<{ 'QUERY PLAN': Array<{ Plan: PlanNode; 'Execution Time': number }> }>>(
        `EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
         WITH hnsw_candidates AS (
           SELECT
             c.id, c.document_id, c.content,
             (c.embedding <=> $1::vector) AS distance
           FROM document_chunks c
           WHERE c.org_id = $2::uuid
             AND c.is_active = true
             AND c.chunk_role <> 'parent'
             AND c.embedding IS NOT NULL
           ORDER BY c.embedding <=> $1::vector ASC
           LIMIT 60
         )
         SELECT h.id, d.title, (1.0 - h.distance)::float8 AS similarity
         FROM hnsw_candidates h
         INNER JOIN documents d ON d.id = h.document_id
         WHERE d.is_deleted = false
           AND d.governance_status <> 'draft'
           AND d.governance_status <> 'archived'
           AND h.distance <= 0.85
         ORDER BY h.distance ASC
         LIMIT 15`,
        [vecLiteral, orgId],
      );
      return rows[0]!['QUERY PLAN'][0]!;
    });

    const hnswNodes = collectNodes(hnswExplain.Plan);
    const usedHnswIdx = hnswNodes.some((n) => n['Index Name'] === 'idx_chunks_embedding_hnsw');
    const hnswSeqScan = hnswNodes.some(
      (n) => n['Node Type'] === 'Seq Scan' && n['Relation Name'] === 'document_chunks',
    );
    console.log('\n=== EXPLAIN (ANALYZE, BUFFERS): HNSW Vector Search ===');
    console.log(`  Execution Time : ${hnswExplain['Execution Time'].toFixed(2)} ms`);
    console.log(`  Plan Nodes     : ${hnswNodes.map((n) => `${n['Node Type']}${n['Index Name'] ? `[${n['Index Name']}]` : ''}`).join(' -> ')}`);
    console.log(`  Uses HNSW Index: ${usedHnswIdx ? 'YES' : 'NO'}`);
    if (!usedHnswIdx) failures.push('HNSW Vector Search did NOT use idx_chunks_embedding_hnsw');
    if (hnswSeqScan) failures.push('HNSW Vector Search fell back to Seq Scan on document_chunks');

    // 4. EXPLAIN (ANALYZE, BUFFERS) — Tenant-Scoped Composite GIN FTS Search
    const ftsExplainRows = await queryClient.unsafe<
      Array<{ 'QUERY PLAN': Array<{ Plan: PlanNode; 'Execution Time': number }> }>
    >(
      `EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
       WITH q AS (
         SELECT NULLIF(plainto_tsquery('simple', $1)::text, '')::tsquery AS tsq
       ),
       fts_candidates AS (
         SELECT c.id, c.document_id, c.content, c.content_tsv
         FROM document_chunks c, q
         WHERE q.tsq IS NOT NULL
           AND c.org_id = $2::uuid
           AND c.is_active = true
           AND c.chunk_role <> 'parent'
           AND c.content_tsv @@ q.tsq
         LIMIT 450
       )
       SELECT f.id, d.title, COALESCE(ts_rank_cd(f.content_tsv, q.tsq), 0)::float8 AS rank
       FROM fts_candidates f
       CROSS JOIN q
       INNER JOIN documents d ON d.id = f.document_id
       WHERE d.is_deleted = false
         AND d.governance_status <> 'draft'
         AND d.governance_status <> 'archived'
       ORDER BY rank DESC
       LIMIT 15`,
      ['POL-SEC-409 MFA token rotation', orgId],
    );
    const ftsExplain = ftsExplainRows[0]!['QUERY PLAN'][0]!;
    const ftsNodes = collectNodes(ftsExplain.Plan);
    const usedGinIdx = ftsNodes.some((n) => n['Index Name'] === 'idx_chunks_org_content_tsv_gin');
    const ftsSeqScan = ftsNodes.some(
      (n) => n['Node Type'] === 'Seq Scan' && n['Relation Name'] === 'document_chunks',
    );
    console.log('\n=== EXPLAIN (ANALYZE, BUFFERS): Composite GIN FTS Search ===');
    console.log(`  Execution Time : ${ftsExplain['Execution Time'].toFixed(2)} ms`);
    console.log(`  Plan Nodes     : ${ftsNodes.map((n) => `${n['Node Type']}${n['Index Name'] ? `[${n['Index Name']}]` : ''}`).join(' -> ')}`);
    console.log(`  Uses GIN Index : ${usedGinIdx ? 'YES' : 'NO'}`);
    if (!usedGinIdx) failures.push('FTS Search did NOT use idx_chunks_org_content_tsv_gin');
    if (ftsSeqScan) failures.push('FTS Search fell back to Seq Scan on document_chunks');

    // 5. Verify Lean Projection on live ChunkRepository calls (no `embedding` wire over-fetch)
    const chunkRepo = new ChunkRepository();
    const [vecHits, ftsHits] = await Promise.all([
      chunkRepo.searchSimilar(orgId, qVec, 5, 0.1),
      chunkRepo.searchFullText(orgId, 'POL-SEC-409 MFA token rotation', 5),
    ]);
    console.log('\n=== Live Repository Lean-Projection Check ===');
    console.log(`  searchSimilar hits: ${vecHits.length}, searchFullText hits: ${ftsHits.length}`);
    if (vecHits.length === 0) failures.push('searchSimilar returned 0 hits on seeded org');
    if (ftsHits.length === 0) failures.push('searchFullText returned 0 hits on seeded org');
    for (const hit of [...vecHits, ...ftsHits]) {
      if ('embedding' in hit.chunk) {
        failures.push('Lean projection violated: `embedding` property found on retrieved chunk');
        break;
      }
      if ('content_tsv' in hit.chunk || 'contentTsv' in hit.chunk) {
        failures.push('Lean projection violated: `content_tsv` property found on retrieved chunk');
        break;
      }
    }

    if (failures.length > 0) {
      console.error('\n[FAIL] Query Plan Regression Gate FAILED:');
      for (const f of failures) console.error(`  - ${f}`);
      process.exitCode = 1;
    } else {
      console.log('\n[PASS] All V2.7 query plans, indexes, and lean projections verified!');
    }
  } finally {
    if (shouldCleanup) {
      await cleanupIsolatedOrg(orgId).catch((err) => console.error('Cleanup error:', err));
    }
    await queryClient.end({ timeout: 5 });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
