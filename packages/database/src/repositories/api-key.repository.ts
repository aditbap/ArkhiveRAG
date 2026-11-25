import crypto from 'node:crypto';
import { and, desc, eq, isNull } from 'drizzle-orm';
import { BaseRepository } from './base.repository.js';
import { apiKeys } from '../schema/api-keys.js';

export type ApiKeyRole = 'admin' | 'member' | 'viewer';

export interface ApiKeyMetadata {
  id: string;
  orgId: string;
  createdBy: string;
  name: string;
  keyPrefix: string;
  role: string;
  expiresAt: Date | null;
  revokedAt: Date | null;
  lastUsedAt: Date | null;
  createdAt: Date;
}

export interface VerifiedApiKey {
  id: string;
  orgId: string;
  createdBy: string;
  name: string;
  keyPrefix: string;
  role: string;
}

const API_KEY_PATTERN = /^(ak_live_[0-9a-f]{12})_[0-9a-f]{48}$/;

/** Extract the lookup prefix (`ak_live_<12-hex>`) from a well-formed raw key, or null. */
export function parseApiKeyPrefix(rawKey: string): string | null {
  const match = API_KEY_PATTERN.exec(rawKey.trim());
  return match?.[1] ?? null;
}

/** Deterministic SHA-256 hex digest used for constant-time storage verification. */
export function hashApiKey(rawKey: string): string {
  return crypto.createHash('sha256').update(rawKey.trim(), 'utf8').digest('hex');
}

export class ApiKeyRepository extends BaseRepository {
  constructor() {
    super('api_keys');
  }

  /** Mint a new org-scoped API key; the raw secret is returned once and never persisted. */
  async createKey(params: {
    orgId: string;
    createdBy: string;
    name: string;
    role: ApiKeyRole;
    expiresAt?: Date | null;
  }): Promise<{ key: string; record: ApiKeyMetadata }> {
    this.requireTenant(params.orgId);

    const prefix = `ak_live_${crypto.randomBytes(6).toString('hex')}`;
    const secret = crypto.randomBytes(24).toString('hex');
    const rawKey = `${prefix}_${secret}`;
    const keyHash = hashApiKey(rawKey);

    const [row] = await this.db
      .insert(apiKeys)
      .values({
        orgId: params.orgId,
        createdBy: params.createdBy,
        name: params.name.trim(),
        keyPrefix: prefix,
        keyHash,
        role: params.role,
        expiresAt: params.expiresAt ?? null,
      })
      .returning({
        id: apiKeys.id,
        orgId: apiKeys.orgId,
        createdBy: apiKeys.createdBy,
        name: apiKeys.name,
        keyPrefix: apiKeys.keyPrefix,
        role: apiKeys.role,
        expiresAt: apiKeys.expiresAt,
        revokedAt: apiKeys.revokedAt,
        lastUsedAt: apiKeys.lastUsedAt,
        createdAt: apiKeys.createdAt,
      });

    if (!row) {
      throw new Error('Failed to insert API key');
    }

    return { key: rawKey, record: row };
  }

  /** List org keys ordered newest-first; never exposes `keyHash`. */
  async listByOrg(orgId: string): Promise<ApiKeyMetadata[]> {
    this.requireTenant(orgId);
    return this.readDb
      .select({
        id: apiKeys.id,
        orgId: apiKeys.orgId,
        createdBy: apiKeys.createdBy,
        name: apiKeys.name,
        keyPrefix: apiKeys.keyPrefix,
        role: apiKeys.role,
        expiresAt: apiKeys.expiresAt,
        revokedAt: apiKeys.revokedAt,
        lastUsedAt: apiKeys.lastUsedAt,
        createdAt: apiKeys.createdAt,
      })
      .from(apiKeys)
      .where(eq(apiKeys.orgId, orgId))
      .orderBy(desc(apiKeys.createdAt));
  }

  /** Soft-revoke an active API key inside `orgId`. */
  async revokeKey(orgId: string, keyId: string): Promise<boolean> {
    this.requireTenant(orgId);
    const updated = await this.db
      .update(apiKeys)
      .set({ revokedAt: new Date() })
      .where(and(eq(apiKeys.id, keyId), eq(apiKeys.orgId, orgId), isNull(apiKeys.revokedAt)))
      .returning({ id: apiKeys.id });

    return updated.length > 0;
  }

  /** Constant-time verification of a raw `ak_live_...` credential against active non-expired keys. */
  async verifyKey(rawKey: string): Promise<VerifiedApiKey | null> {
    const keyPrefix = parseApiKeyPrefix(rawKey);
    if (!keyPrefix) return null;

    const [row] = await this.readDb
      .select({
        id: apiKeys.id,
        orgId: apiKeys.orgId,
        createdBy: apiKeys.createdBy,
        name: apiKeys.name,
        keyPrefix: apiKeys.keyPrefix,
        keyHash: apiKeys.keyHash,
        role: apiKeys.role,
        expiresAt: apiKeys.expiresAt,
      })
      .from(apiKeys)
      .where(and(eq(apiKeys.keyPrefix, keyPrefix), isNull(apiKeys.revokedAt)))
      .limit(1);

    if (!row) return null;
    if (row.expiresAt && row.expiresAt.getTime() <= Date.now()) return null;

    const candidate = Buffer.from(hashApiKey(rawKey), 'hex');
    const stored = Buffer.from(row.keyHash, 'hex');
    if (candidate.length !== stored.length || !crypto.timingSafeEqual(candidate, stored)) {
      return null;
    }

    // Best-effort non-blocking last_used_at stamp on writer pool
    this.db
      .update(apiKeys)
      .set({ lastUsedAt: new Date() })
      .where(eq(apiKeys.id, row.id))
      .catch(() => undefined);

    return {
      id: row.id,
      orgId: row.orgId,
      createdBy: row.createdBy,
      name: row.name,
      keyPrefix: row.keyPrefix,
      role: row.role,
    };
  }
}
