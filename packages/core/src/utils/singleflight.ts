/**
 * In-flight request coalescer (singleflight) for high-concurrency read paths.
 *
 * When multiple concurrent virtual users execute the exact same tenant-scoped,
 * ACL-scoped read query simultaneously, they share a single underlying DB promise
 * instead of stampeding the connection pool.
 *
 * Safety invariants:
 * - Callers MUST include `orgId` and any ACL scope (`documentIds` hash) in `key`.
 * - Entries are ALWAYS removed in `finally` so errors never poison subsequent calls
 *   and completed results are never served stale.
 */
export class SingleflightGroup {
  private readonly inFlight = new Map<string, Promise<unknown>>();
  private coalesced = 0;

  async run<T>(key: string, fn: () => Promise<T>): Promise<T> {
    const existing = this.inFlight.get(key) as Promise<T> | undefined;
    if (existing) {
      this.coalesced++;
      return existing;
    }

    const promise = Promise.resolve()
      .then(fn)
      .finally(() => {
        if (this.inFlight.get(key) === promise) {
          this.inFlight.delete(key);
        }
      });

    this.inFlight.set(key, promise);
    return promise;
  }

  inFlightCount(): number {
    return this.inFlight.size;
  }

  coalescedCount(): number {
    return this.coalesced;
  }

  clear(): void {
    this.inFlight.clear();
  }
}
