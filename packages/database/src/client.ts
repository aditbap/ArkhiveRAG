import { drizzle } from 'drizzle-orm/postgres-js';
import { sql } from 'drizzle-orm';
import postgres from 'postgres';
import * as schema from './schema/index.js';

export type DatabasePoolerMode = 'session' | 'transaction';

export interface ResolvedPoolOptions {
  connectionString: string;
  max: number;
  connectTimeoutS: number;
  statementTimeoutMs: number;
  poolerMode: DatabasePoolerMode;
  prepare: boolean;
}

/**
 * Resolves `postgres.js` pool options from environment variables and URL.
 * Critical for PgBouncer transaction pooling: automatically disables unnamed/named
 * prepared statements (`prepare: false`) when `poolerMode === 'transaction'` or when
 * connecting to the default PgBouncer port (`:6432`).
 */
export function resolvePoolOptions(
  connectionString: string,
  opts: {
    poolMaxEnv?: string;
    defaultPoolMax?: number;
    connectTimeoutEnv?: string;
    statementTimeoutEnv?: string;
    poolerModeEnv?: string;
  } = {},
): ResolvedPoolOptions {
  const defaultPoolMax = opts.defaultPoolMax ?? 25;
  const max = Math.max(5, parseInt(opts.poolMaxEnv || String(defaultPoolMax), 10) || defaultPoolMax);
  const connectTimeoutS = Math.max(1, parseInt(opts.connectTimeoutEnv || '5', 10) || 5);
  const rawStatementTimeout = parseInt(opts.statementTimeoutEnv || '30000', 10);
  const statementTimeoutMs =
    Number.isFinite(rawStatementTimeout) && rawStatementTimeout >= 0 ? rawStatementTimeout : 30000;

  const rawMode = (opts.poolerModeEnv || 'auto').trim().toLowerCase();
  let poolerMode: DatabasePoolerMode = 'session';
  if (rawMode === 'transaction' || rawMode === 'pgbouncer') {
    poolerMode = 'transaction';
  } else if (rawMode === 'session' || rawMode === 'direct') {
    poolerMode = 'session';
  } else {
    // Auto-detect standard PgBouncer port (:6432) or `?pgbouncer=true` query hint
    try {
      const parsed = new URL(connectionString);
      if (parsed.port === '6432' || parsed.searchParams.get('pgbouncer') === 'true') {
        poolerMode = 'transaction';
      }
    } catch {
      if (connectionString.includes(':6432/') || connectionString.includes('pgbouncer=true')) {
        poolerMode = 'transaction';
      }
    }
  }

  return {
    connectionString,
    max,
    connectTimeoutS,
    statementTimeoutMs,
    poolerMode,
    prepare: poolerMode !== 'transaction',
  };
}

function createPostgresClient(resolved: ResolvedPoolOptions) {
  return postgres(resolved.connectionString, {
    max: resolved.max,
    idle_timeout: 20,
    max_lifetime: 60 * 30,
    connect_timeout: resolved.connectTimeoutS,
    prepare: resolved.prepare,
    // PgBouncer transaction mode rejects startup `statement_timeout` unless configured
    // in `ignore_startup_parameters`; only pass it as a startup parameter in session mode
    // (or when PgBouncer explicitly allows it).
    ...(resolved.statementTimeoutMs > 0 && resolved.poolerMode === 'session'
      ? { connection: { statement_timeout: resolved.statementTimeoutMs } }
      : {}),
  });
}

// Primary (read+write) connection
const primaryConnectionString =
  process.env.DATABASE_URL || 'postgresql://postgres:postgres@localhost:5432/rag_platform';

export const PRIMARY_POOL_CONFIG = resolvePoolOptions(primaryConnectionString, {
  poolMaxEnv: process.env.DATABASE_POOL_MAX,
  defaultPoolMax: 25,
  connectTimeoutEnv: process.env.DATABASE_CONNECT_TIMEOUT_S,
  statementTimeoutEnv: process.env.DATABASE_STATEMENT_TIMEOUT_MS,
  poolerModeEnv: process.env.DATABASE_POOLER_MODE,
});

export const DATABASE_POOL_MAX = PRIMARY_POOL_CONFIG.max;
export const DATABASE_CONNECT_TIMEOUT_S = PRIMARY_POOL_CONFIG.connectTimeoutS;
export const DATABASE_STATEMENT_TIMEOUT_MS = PRIMARY_POOL_CONFIG.statementTimeoutMs;
export const DATABASE_POOLER_MODE = PRIMARY_POOL_CONFIG.poolerMode;

export const queryClient = createPostgresClient(PRIMARY_POOL_CONFIG);
export const db = drizzle(queryClient, { schema });

// Optional Read-Replica connection (defaults seamlessly to primary when DATABASE_READ_URL is unset)
const rawReadUrl = (process.env.DATABASE_READ_URL || '').trim();
export const HAS_READ_REPLICA = rawReadUrl.length > 0 && rawReadUrl !== primaryConnectionString;

export const READ_POOL_CONFIG = HAS_READ_REPLICA
  ? resolvePoolOptions(rawReadUrl, {
    poolMaxEnv: process.env.DATABASE_READ_POOL_MAX || process.env.DATABASE_POOL_MAX,
    defaultPoolMax: 30,
    connectTimeoutEnv: process.env.DATABASE_CONNECT_TIMEOUT_S,
    statementTimeoutEnv: process.env.DATABASE_STATEMENT_TIMEOUT_MS,
    poolerModeEnv: process.env.DATABASE_READ_POOLER_MODE || process.env.DATABASE_POOLER_MODE,
  })
  : PRIMARY_POOL_CONFIG;

export const readQueryClient = HAS_READ_REPLICA ? createPostgresClient(READ_POOL_CONFIG) : queryClient;
export const readDb = HAS_READ_REPLICA ? drizzle(readQueryClient, { schema }) : db;

// ---- pgvector HNSW Runtime Search Parameters ----
export const HNSW_EF_SEARCH_DEFAULT = Math.max(
  16,
  Math.min(400, parseInt(process.env.HNSW_EF_SEARCH || '64', 10) || 64),
);
export const HNSW_ITERATIVE_SCAN_MODE = (process.env.HNSW_ITERATIVE_SCAN || 'relaxed_order').trim().toLowerCase();

let cachedIterativeScanSupport: boolean | null = null;
let iterativeScanProbePromise: Promise<boolean> | null = null;

/**
 * Lazily probes whether the connected PostgreSQL + pgvector instance supports
 * `hnsw.iterative_scan` (pgvector >= 0.8.0). Cached per process so filtered
 * multi-tenant HNSW queries get 100% recall on pgvector >= 0.8 without ever
 * aborting a transaction on pgvector 0.7.
 */
export async function supportsHnswIterativeScan(
  client: typeof queryClient = readQueryClient,
): Promise<boolean> {
  if (cachedIterativeScanSupport !== null) return cachedIterativeScanSupport;
  if (HNSW_ITERATIVE_SCAN_MODE === 'off' || HNSW_ITERATIVE_SCAN_MODE === 'false') {
    cachedIterativeScanSupport = false;
    return false;
  }
  if (!iterativeScanProbePromise) {
    // Note: PostgreSQL only populates `pg_settings` with `hnsw.*` after `vector.so` loads
    // in that backend, so checking `pg_extension.extversion >= 0.8` is reliable on fresh connections.
    iterativeScanProbePromise = client<Array<{ extversion: string }>>`
      SELECT extversion
      FROM pg_extension
      WHERE extname = 'vector'
    `
      .then((rows) => {
        const ver = rows[0]?.extversion || '';
        const [majorStr, minorStr] = ver.split('.');
        const major = parseInt(majorStr || '0', 10) || 0;
        const minor = parseInt(minorStr || '0', 10) || 0;
        const supported = major > 0 || minor >= 8;
        cachedIterativeScanSupport = supported;
        return supported;
      })
      .catch(() => {
        cachedIterativeScanSupport = false;
        return false;
      });
  }
  return iterativeScanProbePromise;
}

export type DrizzleDb = typeof db;
export type DrizzleTx = Parameters<Parameters<DrizzleDb['transaction']>[0]>[0];

/**
 * Executes a pgvector HNSW query inside a PgBouncer-transaction-safe `SET LOCAL`
 * scope (`hnsw.ef_search` + `hnsw.iterative_scan` when supported + `statement_timeout`
 * when running through PgBouncer transaction mode).
 */
export async function withHnswSearch<T>(
  dbInstance: DrizzleDb,
  efSearch: number,
  fn: (tx: DrizzleTx | DrizzleDb) => Promise<T>,
  opts: { enableIterativeScan?: boolean; poolerMode?: DatabasePoolerMode } = {},
): Promise<T> {
  const clampedEf = Math.max(16, Math.min(400, Math.round(efSearch)));
  const poolerMode = opts.poolerMode ?? READ_POOL_CONFIG.poolerMode;
  const useIterative =
    opts.enableIterativeScan !== false &&
    HNSW_ITERATIVE_SCAN_MODE !== 'off' &&
    HNSW_ITERATIVE_SCAN_MODE !== 'false' &&
    (await supportsHnswIterativeScan());

  if (typeof dbInstance.transaction !== 'function') {
    return fn(dbInstance);
  }

  return dbInstance.transaction(async (tx) => {
    const mode = HNSW_ITERATIVE_SCAN_MODE === 'strict_order' ? 'strict_order' : 'relaxed_order';
    if (useIterative && poolerMode === 'transaction' && READ_POOL_CONFIG.statementTimeoutMs > 0) {
      await tx.execute(
        sql`SELECT
          set_config('hnsw.ef_search', ${String(clampedEf)}, true),
          set_config('hnsw.iterative_scan', ${mode}, true),
          set_config('statement_timeout', ${`${READ_POOL_CONFIG.statementTimeoutMs}ms`}, true)`,
      );
    } else if (useIterative) {
      await tx.execute(
        sql`SELECT
          set_config('hnsw.ef_search', ${String(clampedEf)}, true),
          set_config('hnsw.iterative_scan', ${mode}, true)`,
      );
    } else if (poolerMode === 'transaction' && READ_POOL_CONFIG.statementTimeoutMs > 0) {
      await tx.execute(
        sql`SELECT
          set_config('hnsw.ef_search', ${String(clampedEf)}, true),
          set_config('statement_timeout', ${`${READ_POOL_CONFIG.statementTimeoutMs}ms`}, true)`,
      );
    } else {
      await tx.execute(sql`SELECT set_config('hnsw.ef_search', ${String(clampedEf)}, true)`);
    }
    return fn(tx);
  });
}

export interface DatabasePoolStats {
  active: number;
  idle: number;
  idleInTransaction: number;
  waiting: number;
  total: number;
  poolMax: number;
  poolerMode: DatabasePoolerMode;
  hasReadReplica: boolean;
}

/**
 * Queries PostgreSQL `pg_stat_activity` for current-database connection saturation telemetry.
 */
export async function getDatabasePoolStats(): Promise<DatabasePoolStats> {
  const rows = await queryClient<Array<{ state: string | null; wait_event_type: string | null; count: string }>>`
    SELECT state, wait_event_type, count(*)::text AS count
    FROM pg_stat_activity
    WHERE datname = current_database()
    GROUP BY state, wait_event_type
  `;

  let active = 0;
  let idle = 0;
  let idleInTransaction = 0;
  let waiting = 0;
  let total = 0;

  for (const row of rows) {
    const n = parseInt(row.count, 10) || 0;
    total += n;
    if (row.state === 'active') active += n;
    else if (row.state === 'idle') idle += n;
    else if (row.state && row.state.startsWith('idle in transaction')) idleInTransaction += n;
    if (row.wait_event_type && row.state === 'active') waiting += n;
  }

  return {
    active,
    idle,
    idleInTransaction,
    waiting,
    total,
    poolMax: DATABASE_POOL_MAX,
    poolerMode: DATABASE_POOLER_MODE,
    hasReadReplica: HAS_READ_REPLICA,
  };
}
