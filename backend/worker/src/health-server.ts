import http from 'node:http';
import { db, sql } from '@rag/database';
import type { Redis } from 'ioredis';

export interface WorkerMetricsState {
  startedAtMs: number;
  activeJobs: number;
  completedJobs: number;
  failedJobs: number;
  retriedJobs: number;
  stalledJobs: number;
  lastJobDurationSeconds: number;
}

export const workerMetrics: WorkerMetricsState = {
  startedAtMs: Date.now(),
  activeJobs: 0,
  completedJobs: 0,
  failedJobs: 0,
  retriedJobs: 0,
  stalledJobs: 0,
  lastJobDurationSeconds: 0,
};

function renderPrometheusText(): string {
  const uptimeSeconds = Math.max(0, (Date.now() - workerMetrics.startedAtMs) / 1000);
  return [
    '# HELP worker_uptime_seconds Worker process uptime in seconds',
    '# TYPE worker_uptime_seconds gauge',
    `worker_uptime_seconds ${uptimeSeconds.toFixed(2)}`,
    '# HELP worker_active_jobs Currently executing ingestion jobs',
    '# TYPE worker_active_jobs gauge',
    `worker_active_jobs ${workerMetrics.activeJobs}`,
    '# HELP worker_jobs_total Total ingestion jobs processed by terminal or retry outcome',
    '# TYPE worker_jobs_total counter',
    `worker_jobs_total{status="completed"} ${workerMetrics.completedJobs}`,
    `worker_jobs_total{status="failed"} ${workerMetrics.failedJobs}`,
    `worker_jobs_total{status="retried"} ${workerMetrics.retriedJobs}`,
    '# HELP worker_stalled_jobs_total Total BullMQ stalled jobs reclaimed',
    '# TYPE worker_stalled_jobs_total counter',
    `worker_stalled_jobs_total ${workerMetrics.stalledJobs}`,
    '# HELP worker_last_job_duration_seconds Duration of the most recently finished ingestion job',
    '# TYPE worker_last_job_duration_seconds gauge',
    `worker_last_job_duration_seconds ${workerMetrics.lastJobDurationSeconds.toFixed(3)}`,
    '',
  ].join('\n');
}

function withTimeout<T>(promise: Promise<T>, ms = 2000): Promise<T> {
  return Promise.race([
    promise,
    new Promise<never>((_, rej) => setTimeout(() => rej(new Error('timeout')), ms)),
  ]);
}

/**
 * Lightweight zero-dependency observability server for `@rag/worker` exposing:
 *   - GET /health  (liveness)
 *   - GET /ready   (Postgres + Redis + MinIO dependency check)
 *   - GET /metrics (Prometheus text format)
 */
export function startWorkerHealthServer(opts: {
  port: number;
  host?: string;
  redis: Pick<Redis, 'ping'>;
  storagePing: () => Promise<unknown>;
}): http.Server {
  const server = http.createServer(async (req, res) => {
    if (req.method !== 'GET') {
      res.writeHead(405).end();
      return;
    }

    if (req.url === '/health') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(
        JSON.stringify({
          status: 'ok',
          activeJobs: workerMetrics.activeJobs,
          uptimeSeconds: Math.round((Date.now() - workerMetrics.startedAtMs) / 1000),
          timestamp: new Date().toISOString(),
        }),
      );
      return;
    }

    if (req.url === '/ready') {
      const checks: Record<string, string> = {};
      await Promise.all(
        (
          [
            ['postgres', () => db.execute(sql`select 1`)],
            ['redis', () => opts.redis.ping()],
            ['storage', () => opts.storagePing()],
          ] as const
        ).map(async ([name, probe]) => {
          try {
            await withTimeout(probe());
            checks[name] = 'ok';
          } catch (err) {
            checks[name] = err instanceof Error ? err.message : 'failed';
          }
        }),
      );
      const ready = Object.values(checks).every((v) => v === 'ok');
      res.writeHead(ready ? 200 : 503, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: ready ? 'ready' : 'unavailable', checks }));
      return;
    }

    if (req.url === '/metrics') {
      res.writeHead(200, { 'Content-Type': 'text/plain; version=0.0.4; charset=utf-8' });
      res.end(renderPrometheusText());
      return;
    }

    res.writeHead(404).end();
  });

  server.listen(opts.port, opts.host ?? '0.0.0.0');
  return server;
}
