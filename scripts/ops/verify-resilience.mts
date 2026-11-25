/**
 * ArkhiveRAG V3.0 — Repeatable Resilience, Multi-Tenant Security & Timeout Verifier
 *
 * Exercises critical production invariants directly against PostgreSQL or PgBouncer
 * (`:5432` or `:6432`) with zero external LLM/embedding API keys required:
 *   1. Multi-Tenant, Draft & Document-ACL Isolation (40 concurrent mixed queries using identical 1.0 vectors)
 *   2. Atomic `replaceVersionChunks` Idempotency (simulates crash partial writes + 2x recovery retries -> 0 duplicates)
 *   3. PostgreSQL `statement_timeout` Guard (verifies runaway query aborts cleanly with SQLSTATE `57014`)
 *
 * Always seeds isolated UUID organizations (`...0030`, `...0031`) and cleans up in `finally`.
 *
 * Usage:
 *   pnpm ops:verify-resilience
 *   DATABASE_URL=postgresql://postgres:postgres@localhost:6432/rag_platform pnpm ops:verify-resilience
 */
import crypto from 'node:crypto';
import * as dotenv from 'dotenv';
import postgres from 'postgres';
import {
  ChunkRepository,
  DATABASE_POOLER_MODE,
  queryClient,
} from '../../packages/database/dist/index.js';

dotenv.config();

const ORG_A = '00000000-0000-4000-8000-000000000030';
const ORG_B = '00000000-0000-4000-8000-000000000031';
const EMBED_DIM = 1024;

function makeUnitBasisVector(): number[] {
  return Array.from({ length: EMBED_DIM }, (_, i) => (i === 0 ? 1 : 0));
}

async function seedIsolatedTenants() {
  for (const [id, name, slug] of [
    [ORG_A, 'V30 Resilience Tenant A', 'v30-res-tenant-a'],
    [ORG_B, 'V30 Resilience Tenant B', 'v30-res-tenant-b'],
  ] as const) {
    await queryClient`
      INSERT INTO organizations (id, name, slug, kind, is_active, max_documents, max_storage_mb)
      VALUES (${id}::uuid, ${name}, ${slug}, 'shared', true, 1000, 1000)
      ON CONFLICT (id) DO NOTHING
    `;
  }
}

async function cleanupIsolatedTenants() {
  for (const orgId of [ORG_A, ORG_B]) {
    await queryClient`DELETE FROM document_chunks WHERE org_id = ${orgId}::uuid`;
    await queryClient`DELETE FROM documents WHERE org_id = ${orgId}::uuid`;
    await queryClient`DELETE FROM organizations WHERE id = ${orgId}::uuid`;
  }
}

async function verifyMultiTenantAndAclIsolation(chunkRepo: ChunkRepository) {
  const vec = makeUnitBasisVector();
  const vecStr = `[${vec.join(',')}]`;
  const docA1 = crypto.randomUUID();
  const docA2 = crypto.randomUUID();
  const docADraft = crypto.randomUUID();
  const docB1 = crypto.randomUUID();

  const records = [
    [docA1, ORG_A, 'Tenant A Public Policy', 'active', 'SECRET-TOKEN-ALPHA-PUBLIC POL-SEC-409'],
    [docA2, ORG_A, 'Tenant A Restricted Policy', 'active', 'SECRET-TOKEN-ALPHA-RESTRICTED POL-SEC-409'],
    [docADraft, ORG_A, 'Tenant A Draft Policy', 'draft', 'SECRET-TOKEN-ALPHA-DRAFT POL-SEC-409'],
    [docB1, ORG_B, 'Tenant B Confidential', 'active', 'SECRET-TOKEN-BETA-ONLY POL-SEC-409'],
  ] as const;

  for (const [did, oid, title, gov, content] of records) {
    const vid = crypto.randomUUID();
    const cid = crypto.randomUUID();
    const hash = crypto.createHash('sha256').update(content).digest('hex');
    await queryClient`
      INSERT INTO documents (id, org_id, title, file_name, file_type, file_size_bytes, file_hash, status, governance_status, authority_tier, effective_at)
      VALUES (${did}::uuid, ${oid}::uuid, ${title}, ${`${did}.md`}, 'md', 128, ${hash}, 'ready', ${gov}, 3, NOW())
    `;
    await queryClient`
      INSERT INTO document_versions (id, document_id, org_id, version_number, file_path, file_hash, file_size_bytes)
      VALUES (${vid}::uuid, ${did}::uuid, ${oid}::uuid, 1, ${`res/${did}.md`}, ${hash}, 128)
    `;
    await queryClient.unsafe(
      `INSERT INTO document_chunks (id, org_id, document_id, version_id, chunk_index, content, content_hash, token_count, chunk_role, embedding, embedding_model, is_active)
       VALUES ($1::uuid, $2::uuid, $3::uuid, $4::uuid, 0, $5, $6, 12, 'standalone', $7::vector, 'resilience-test', true)`,
      [cid, oid, did, vid, content, hash, vecStr],
    );
  }

  let crossTenantLeaks = 0;
  let draftLeaks = 0;
  let aclLeaks = 0;
  let totalHits = 0;

  await Promise.all(
    Array.from({ length: 40 }, async (_, i) => {
      const mode = i % 3;
      if (mode === 0) {
        const [v, f] = await Promise.all([
          chunkRepo.searchSimilar(ORG_A, vec, 10, 0.1),
          chunkRepo.searchFullText(ORG_A, 'POL-SEC-409', 10),
        ]);
        totalHits += v.length + f.length;
        for (const hit of [...v, ...f]) {
          if (hit.chunk.orgId !== ORG_A || hit.chunk.content.includes('BETA')) crossTenantLeaks++;
          if (hit.chunk.content.includes('DRAFT') || hit.document.governanceStatus === 'draft') draftLeaks++;
        }
      } else if (mode === 1) {
        const [v, f] = await Promise.all([
          chunkRepo.searchSimilar(ORG_A, vec, 10, 0.1, [docA1]),
          chunkRepo.searchFullText(ORG_A, 'POL-SEC-409', 10, [docA1]),
        ]);
        totalHits += v.length + f.length;
        for (const hit of [...v, ...f]) {
          if (hit.chunk.orgId !== ORG_A) crossTenantLeaks++;
          if (hit.document.id !== docA1 || hit.chunk.content.includes('RESTRICTED') || hit.chunk.content.includes('DRAFT')) {
            aclLeaks++;
          }
        }
      } else {
        const [v, f] = await Promise.all([
          chunkRepo.searchSimilar(ORG_B, vec, 10, 0.1),
          chunkRepo.searchFullText(ORG_B, 'POL-SEC-409', 10),
        ]);
        totalHits += v.length + f.length;
        for (const hit of [...v, ...f]) {
          if (hit.chunk.orgId !== ORG_B || hit.chunk.content.includes('ALPHA')) crossTenantLeaks++;
        }
      }
    }),
  );

  const ok = totalHits > 0 && crossTenantLeaks === 0 && draftLeaks === 0 && aclLeaks === 0;
  console.log(
    `  [1/3] Multi-Tenant & ACL Concurrency (40 queries, ${DATABASE_POOLER_MODE}): hits=${totalHits}, tenantLeaks=${crossTenantLeaks}, draftLeaks=${draftLeaks}, aclLeaks=${aclLeaks} -> ${ok ? 'PASS' : 'FAIL'}`,
  );
  if (!ok) {
    throw new Error(`Tenant/ACL isolation check failed (hits=${totalHits}, tenant=${crossTenantLeaks}, draft=${draftLeaks}, acl=${aclLeaks})`);
  }
}

async function verifyIdempotentReplacement(chunkRepo: ChunkRepository) {
  const docId = crypto.randomUUID();
  const verId = crypto.randomUUID();
  const vec = makeUnitBasisVector();
  const hash = crypto.createHash('sha256').update('idempotent-test-doc').digest('hex');

  await queryClient`
    INSERT INTO documents (id, org_id, title, file_name, file_type, file_size_bytes, file_hash, status, governance_status, authority_tier, effective_at)
    VALUES (${docId}::uuid, ${ORG_A}::uuid, 'Crash Recovery Doc', 'crash.md', 'md', 100, ${hash}, 'processing', 'active', 2, NOW())
  `;
  await queryClient`
    INSERT INTO document_versions (id, document_id, org_id, version_number, file_path, file_hash, file_size_bytes)
    VALUES (${verId}::uuid, ${docId}::uuid, ${ORG_A}::uuid, 1, 'res/crash.md', ${hash}, 100)
  `;

  // Simulate partial crashed writes before retry
  await chunkRepo.insertChunks(
    ORG_A,
    docId,
    verId,
    [
      { text: 'Partial orphan chunk 1 before SIGKILL', embedding: vec, chunkRole: 'standalone' },
      { text: 'Partial orphan chunk 2 before SIGKILL', embedding: vec, chunkRole: 'standalone' },
    ],
    'resilience-test',
  );

  const finalChunks = Array.from({ length: 5 }, (_, idx) => ({
    text: `Recovered clean chunk #${idx}`,
    embedding: vec,
    chunkRole: 'standalone' as const,
  }));

  // Execute recovery twice (simulating duplicate BullMQ redelivery)
  await chunkRepo.replaceVersionChunks(ORG_A, docId, verId, [], finalChunks, 'resilience-test');
  await chunkRepo.replaceVersionChunks(ORG_A, docId, verId, [], finalChunks, 'resilience-test');

  const countRows = await queryClient<Array<{ cnt: string }>>`
    SELECT count(*)::text AS cnt
    FROM document_chunks
    WHERE org_id = ${ORG_A}::uuid AND document_id = ${docId}::uuid AND is_active = true
  `;
  const activeCount = parseInt(countRows[0]?.cnt || '0', 10);
  const ok = activeCount === 5;
  console.log(`  [2/3] Atomic Idempotent replaceVersionChunks (partial write + 2x retry): activeChunks=${activeCount}/5 -> ${ok ? 'PASS' : 'FAIL'}`);
  if (!ok) {
    throw new Error(`Expected 5 active chunks after idempotent recovery, found ${activeCount}`);
  }
}

async function verifyStatementTimeoutGuard() {
  const databaseUrl = process.env.DATABASE_URL || 'postgresql://postgres:postgres@localhost:5432/rag_platform';
  const shortClient = postgres(databaseUrl, {
    max: 1,
    connect_timeout: 5,
    prepare: DATABASE_POOLER_MODE !== 'transaction',
  });

  const startedAt = Date.now();
  let code = '';
  try {
    await shortClient.begin(async (tx) => {
      await tx.unsafe(`SET LOCAL statement_timeout = '300ms'`);
      await tx.unsafe(`SELECT pg_sleep(2)`);
    });
  } catch (err) {
    code = (err as { code?: string })?.code ?? '';
  } finally {
    await shortClient.end({ timeout: 5 });
  }

  const elapsedMs = Date.now() - startedAt;
  const ok = code === '57014' && elapsedMs < 1500;
  console.log(`  [3/3] PostgreSQL statement_timeout Guard: SQLSTATE=${code || 'NONE'} in ${elapsedMs}ms -> ${ok ? 'PASS' : 'FAIL'}`);
  if (!ok) {
    throw new Error(`statement_timeout guard failed (SQLSTATE=${code}, elapsedMs=${elapsedMs})`);
  }
}

async function main() {
  console.log(`\n=== ArkhiveRAG V3.0 Resilience & Isolation Gate (poolerMode=${DATABASE_POOLER_MODE}) ===`);
  const chunkRepo = new ChunkRepository();

  try {
    await cleanupIsolatedTenants();
    await seedIsolatedTenants();
    await verifyMultiTenantAndAclIsolation(chunkRepo);
    await verifyIdempotentReplacement(chunkRepo);
    await verifyStatementTimeoutGuard();
    console.log('\n[PASS] All V3.0 resilience, isolation, and timeout checks passed!');
  } finally {
    await cleanupIsolatedTenants().catch((err) => console.error('Cleanup error:', err));
    await queryClient.end({ timeout: 5 });
  }
}

main().catch((err) => {
  console.error('\n[FAIL] V3.0 resilience gate failed:', err);
  process.exit(1);
});
