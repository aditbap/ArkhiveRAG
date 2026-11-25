import { db, readDb } from '../client.js';

export abstract class BaseRepository {
  /** Primary read/write Drizzle client. */
  protected readonly db = db;
  /** Read-replica Drizzle client (aliases primary `db` when `DATABASE_READ_URL` is unset). */
  protected readonly readDb = readDb;

  /**
   * Base constructor
   * @param tableName Name of the table for logging purposes
   */
  constructor(protected readonly tableName: string) {}

  /**
   * Helper to ensure tenant isolation is explicitly passed to every repository method.
   * This is a defense-in-depth mechanism.
   * @param tenantId The organization ID to filter by
   */
  protected requireTenant(tenantId: string): void {
    if (!tenantId) {
      throw new Error(`Tenant ID is required for operations on ${this.tableName}`);
    }
  }
}
