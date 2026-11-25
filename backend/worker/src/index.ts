import './setup.js';
import { assertProductionConfig } from '@rag/core';
import { MinioStorageProvider } from '@rag/providers';
import { db, queryClient, sql } from '@rag/database';
import { purgeExpired } from './maintenance.js';
import { startWorkerHealthServer } from './health-server.js';
import { ingestionQueue, redisConnection } from './queue.js';

assertProductionConfig('worker');

// Imported after the config check so a bad configuration is reported instead of a downstream error.
const { ingestionWorker } = await import('./worker.js');

console.log('Worker process starting...');

/**
 * Startup reconciler: if a previous worker replica was SIGKILL'd and left documents stuck in
 * `processing` for >15 minutes with no active/waiting BullMQ job, surface them cleanly as failed
 * so UI spinners do not hang indefinitely.
 */
async function reconcileStaleProcessingDocs(): Promise<void> {
  const locked = await redisConnection.set('rag:stale-reconcile-lock', String(process.pid), 'EX', 120, 'NX').catch(() => null);
  if (!locked) return;
  try {
    const activeJobs = await ingestionQueue.getJobs(['active', 'waiting', 'delayed']);
    const liveDocIds = new Set(activeJobs.map((j) => j.data?.documentId).filter(Boolean));
    const staleRows = await db.execute<{ id: string }>(sql`
      SELECT id FROM documents
      WHERE status = 'processing'
        AND is_deleted = false
        AND updated_at < NOW() - INTERVAL '15 minutes'
      LIMIT 100
    `);
    let reconciled = 0;
    for (const row of staleRows) {
      if (liveDocIds.has(row.id)) continue;
      await db.execute(sql`
        UPDATE documents
        SET status = 'failed',
            error_message = 'Ingestion interrupted by worker restart; please re-upload or requeue',
            updated_at = NOW()
        WHERE id = ${row.id} AND status = 'processing'
      `);
      reconciled++;
    }
    if (reconciled > 0) {
      console.warn(`[Worker] Reconciled ${reconciled} stale processing document(s) after restart`);
    }
  } catch (err) {
    console.warn('[Worker] Stale processing reconciliation skipped:', err instanceof Error ? err.message : err);
  } finally {
    await redisConnection.del('rag:stale-reconcile-lock').catch(() => undefined);
  }
}
void reconcileStaleProcessingDocs();

// Retention maintenance: permanently purge soft-deleted documents/libraries after a grace period.
// A Redis lock keeps multiple worker replicas from running it at once.
const retentionDays = parseInt(process.env.PURGE_RETENTION_DAYS || '30', 10);
const storage = MinioStorageProvider.fromEnv();

const workerMetricsPort = parseInt(process.env.WORKER_METRICS_PORT || '9092', 10);
const healthServer = Number.isFinite(workerMetricsPort) && workerMetricsPort > 0
  ? startWorkerHealthServer({
      port: workerMetricsPort,
      redis: redisConnection,
      storagePing: () => storage.ping(),
    })
  : null;
if (healthServer) {
  console.log(`[Worker] Health & metrics server listening on :${workerMetricsPort} (/health, /ready, /metrics)`);
}
const runMaintenance = async () => {
  const locked = await redisConnection.set('rag:maintenance-lock', String(process.pid), 'EX', 3600, 'NX');
  if (!locked) return;
  try {
    const r = await purgeExpired(storage, retentionDays);
    if (r.librariesPurged || r.documentsPurged) console.log('[Maintenance] purged', JSON.stringify(r));
  } catch (err) {
    console.error('[Maintenance] purge failed:', err);
  } finally {
    await redisConnection.del('rag:maintenance-lock').catch(() => undefined);
  }
};
const firstRun = setTimeout(runMaintenance, 60_000);
const interval = setInterval(runMaintenance, 6 * 3600_000);

const shutdown = async () => {
  console.log('Shutting down worker process gracefully...');
  clearTimeout(firstRun);
  clearInterval(interval);
  healthServer?.close();
  await ingestionWorker.close();
  await redisConnection.quit().catch(() => undefined);
  await queryClient.end({ timeout: 5 }).catch(() => undefined);
  process.exit(0);
};

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
