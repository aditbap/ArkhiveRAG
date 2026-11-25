import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import * as schema from './schema/index.js';

// Setup connection string
const connectionString = process.env.DATABASE_URL || 'postgresql://postgres:postgres@localhost:5432/rag_platform';

// Bounded connection pool & timeouts prevent slow queries or 50+ VU bursts from wedging Postgres.
export const DATABASE_POOL_MAX = Math.max(5, parseInt(process.env.DATABASE_POOL_MAX || '25', 10) || 25);
export const DATABASE_CONNECT_TIMEOUT_S = Math.max(
  1,
  parseInt(process.env.DATABASE_CONNECT_TIMEOUT_S || '5', 10) || 5,
);
const rawStatementTimeout = parseInt(process.env.DATABASE_STATEMENT_TIMEOUT_MS || '30000', 10);
export const DATABASE_STATEMENT_TIMEOUT_MS = Number.isFinite(rawStatementTimeout) && rawStatementTimeout >= 0
  ? rawStatementTimeout
  : 30000;

export const queryClient = postgres(connectionString, {
  max: DATABASE_POOL_MAX,
  idle_timeout: 20,
  max_lifetime: 60 * 30,
  connect_timeout: DATABASE_CONNECT_TIMEOUT_S,
  ...(DATABASE_STATEMENT_TIMEOUT_MS > 0
    ? { connection: { statement_timeout: DATABASE_STATEMENT_TIMEOUT_MS } }
    : {}),
});

// Initialize drizzle
export const db = drizzle(queryClient, { schema });

export interface DatabasePoolStats {
  active: number;
  idle: number;
  idleInTransaction: number;
  waiting: number;
  total: number;
  poolMax: number;
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
  };
}
