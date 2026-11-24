import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import * as schema from './schema/index.js';

// Setup connection string
const connectionString = process.env.DATABASE_URL || 'postgresql://postgres:postgres@localhost:5432/rag_platform';

// Create postgres connection pool (default 25 prevents 50-VU queueing at 10k+ docs)
const poolMax = Math.max(5, parseInt(process.env.DATABASE_POOL_MAX || '25', 10) || 25);
export const queryClient = postgres(connectionString, {
  max: poolMax,
  idle_timeout: 20,
  max_lifetime: 60 * 30,
});

// Initialize drizzle
export const db = drizzle(queryClient, { schema });
