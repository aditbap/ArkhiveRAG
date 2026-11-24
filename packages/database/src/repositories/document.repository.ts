import { eq, and, isNull, desc, getTableColumns, sql } from 'drizzle-orm';
import { BaseRepository } from './base.repository.js';
import { documents, documentVersions, users } from '../schema/index.js';

export class DocumentRepository extends BaseRepository {
  constructor() {
    super('documents');
  }

  async findById(tenantId: string, id: string) {
    this.requireTenant(tenantId);
    return this.db.query.documents.findFirst({
      where: and(
        eq(documents.id, id),
        eq(documents.orgId, tenantId),
        eq(documents.isDeleted, false)
      )
    });
  }

  async listVersions(tenantId: string, documentId: string) {
    this.requireTenant(tenantId);
    return this.db.query.documentVersions.findMany({
      where: and(
        eq(documentVersions.documentId, documentId),
        eq(documentVersions.orgId, tenantId)
      ),
      orderBy: (v, { desc }) => [desc(v.versionNumber)],
    });
  }

  async getActiveVersion(tenantId: string, documentId: string) {
    this.requireTenant(tenantId);
    return this.db.query.documentVersions.findFirst({
      where: and(
        eq(documentVersions.documentId, documentId),
        eq(documentVersions.orgId, tenantId),
        eq(documentVersions.isActive, true)
      )
    });
  }

  async list(tenantId: string, limit = 20, offset = 0, folderId?: string | null) {
    this.requireTenant(tenantId);
    
    const conditions = [
      eq(documents.orgId, tenantId),
      eq(documents.isDeleted, false)
    ];

    if (folderId === null) {
      conditions.push(isNull(documents.folderId));
    } else if (folderId !== undefined) {
      conditions.push(eq(documents.folderId, folderId));
    }

    return this.db.query.documents.findMany({
      where: and(...conditions),
      limit,
      offset,
      orderBy: (docs, { desc }) => [desc(docs.createdAt)],
    });
  }
  
  /** Like list(), but each row carries the uploader's display name (for shared libraries). */
  async listWithUploader(tenantId: string, limit = 20, offset = 0, folderId?: string | null) {
    this.requireTenant(tenantId);

    const conditions = [eq(documents.orgId, tenantId), eq(documents.isDeleted, false)];
    if (folderId === null) conditions.push(isNull(documents.folderId));
    else if (folderId !== undefined) conditions.push(eq(documents.folderId, folderId));

    return this.db
      .select({ ...getTableColumns(documents), uploaderName: users.fullName, uploaderEmail: users.email })
      .from(documents)
      .leftJoin(users, eq(users.id, documents.uploadedBy))
      .where(and(...conditions))
      .orderBy(desc(documents.createdAt))
      .limit(limit)
      .offset(offset);
  }

  async create(tenantId: string, data: typeof documents.$inferInsert) {
    this.requireTenant(tenantId);
    const [result] = await this.db.insert(documents).values({
      ...data,
      orgId: tenantId,
    }).returning();
    return result;
  }

  async softDelete(tenantId: string, id: string) {
    this.requireTenant(tenantId);
    const [result] = await this.db.update(documents)
      .set({ isDeleted: true, deletedAt: new Date() })
      .where(and(eq(documents.id, id), eq(documents.orgId, tenantId)))
      .returning();
    return result;
  }

  async move(tenantId: string, id: string, folderId: string | null) {
    this.requireTenant(tenantId);
    const [result] = await this.db.update(documents)
      .set({ folderId, updatedAt: new Date() })
      .where(and(eq(documents.id, id), eq(documents.orgId, tenantId)))
      .returning();
    return result;
  }

  /** Live documents and bytes stored (every version's object counts) for quota enforcement. */
  async usage(tenantId: string): Promise<{ documentCount: number; storageBytes: number }> {
    this.requireTenant(tenantId);
    const [docs] = await this.db
      .select({ n: sql<number>`count(*)::int` })
      .from(documents)
      .where(and(eq(documents.orgId, tenantId), eq(documents.isDeleted, false)));
    const [bytes] = await this.db
      .select({ n: sql<number>`coalesce(sum(${documentVersions.fileSizeBytes}), 0)::float8` })
      .from(documentVersions)
      .innerJoin(documents, eq(documents.id, documentVersions.documentId))
      .where(and(eq(documentVersions.orgId, tenantId), eq(documents.isDeleted, false)));
    return { documentCount: docs?.n ?? 0, storageBytes: Number(bytes?.n ?? 0) };
  }

  async rename(tenantId: string, id: string, name: string) {
    this.requireTenant(tenantId);
    const [result] = await this.db.update(documents)
      .set({ title: name, fileName: name, updatedAt: new Date() })
      .where(and(eq(documents.id, id), eq(documents.orgId, tenantId), eq(documents.isDeleted, false)))
      .returning();
    return result;
  }

  async updateStatus(tenantId: string, id: string, status: typeof documents.$inferSelect['status'], errorMessage?: string | null) {
    this.requireTenant(tenantId);
    const [result] = await this.db.update(documents)
      .set({ status, errorMessage: errorMessage ?? null, updatedAt: new Date() })
      .where(and(eq(documents.id, id), eq(documents.orgId, tenantId)))
      .returning();
    return result;
  }

  /**
   * Updates enterprise document governance lifecycle (`draft | approved | active | superseded | archived`),
   * effective validity dates, and authority tier, appends an immutable audit trail entry in `metadata`,
   * and atomically marks any `supersedesDocumentId` predecessor as `superseded` when activated.
   */
  async updateGovernance(
    tenantId: string,
    id: string,
    patch: {
      governanceStatus: string;
      effectiveAt?: Date | null;
      expiresAt?: Date | null;
      supersedesDocumentId?: string | null;
      authorityTier?: number;
      actorUserId: string;
      previousStatus: string;
      existingMetadata?: Record<string, unknown>;
      note?: string;
    },
  ) {
    this.requireTenant(tenantId);
    const priorHistory = Array.isArray(patch.existingMetadata?.governanceHistory)
      ? (patch.existingMetadata?.governanceHistory as unknown[])
      : [];
    const auditEntry = {
      from: patch.previousStatus,
      to: patch.governanceStatus,
      actorUserId: patch.actorUserId,
      at: new Date().toISOString(),
      ...(patch.note ? { note: patch.note } : {}),
    };
    const nextMetadata = {
      ...(patch.existingMetadata ?? {}),
      governanceHistory: [...priorHistory.slice(-19), auditEntry],
    };

    const setValues: Record<string, unknown> = {
      governanceStatus: patch.governanceStatus,
      metadata: nextMetadata,
      updatedAt: new Date(),
    };
    if (patch.effectiveAt !== undefined) setValues.effectiveAt = patch.effectiveAt;
    if (patch.expiresAt !== undefined) setValues.expiresAt = patch.expiresAt;
    if (patch.supersedesDocumentId !== undefined) setValues.supersedesDocumentId = patch.supersedesDocumentId;
    if (patch.authorityTier !== undefined) setValues.authorityTier = patch.authorityTier;

    const [updated] = await this.db
      .update(documents)
      .set(setValues)
      .where(and(eq(documents.id, id), eq(documents.orgId, tenantId), eq(documents.isDeleted, false)))
      .returning();

    // Automatically mark predecessor superseded when activating a replacement edition
    if (updated && patch.governanceStatus === 'active' && patch.supersedesDocumentId && patch.supersedesDocumentId !== id) {
      await this.db
        .update(documents)
        .set({ governanceStatus: 'superseded', updatedAt: new Date() })
        .where(
          and(
            eq(documents.id, patch.supersedesDocumentId),
            eq(documents.orgId, tenantId),
            eq(documents.isDeleted, false),
          ),
        );
    }

    return updated;
  }

  /**
   * Atomically deactivates the current active version (if any) and inserts a
   * new one, with the version number incremented from whatever came before.
   * Used both for the initial upload (no prior version — number starts at 1)
   * and for re-uploading a replacement file for an existing document.
   */
  async createNewVersion(
    tenantId: string,
    documentId: string,
    data: { filePath: string; fileHash: string; fileSizeBytes: number }
  ) {
    this.requireTenant(tenantId);
    return this.db.transaction(async (tx) => {
      const [last] = await tx
        .select({ versionNumber: documentVersions.versionNumber })
        .from(documentVersions)
        .where(and(eq(documentVersions.documentId, documentId), eq(documentVersions.orgId, tenantId)))
        .orderBy(desc(documentVersions.versionNumber))
        .limit(1);

      await tx
        .update(documentVersions)
        .set({ isActive: false })
        .where(and(
          eq(documentVersions.documentId, documentId),
          eq(documentVersions.orgId, tenantId),
          eq(documentVersions.isActive, true)
        ));

      const [version] = await tx.insert(documentVersions).values({
        documentId,
        orgId: tenantId,
        versionNumber: (last?.versionNumber ?? 0) + 1,
        filePath: data.filePath,
        fileHash: data.fileHash,
        fileSizeBytes: data.fileSizeBytes,
        isActive: true,
      }).returning();

      if (!version) throw new Error('Failed to create document version');
      return version;
    });
  }

  /** Syncs the documents row's denormalized file info after a new version replaces it. */
  async updateFileInfo(tenantId: string, id: string, data: { fileHash: string; fileSizeBytes: number; status: typeof documents.$inferSelect['status'] }) {
    this.requireTenant(tenantId);
    const [result] = await this.db.update(documents)
      .set({ fileHash: data.fileHash, fileSizeBytes: data.fileSizeBytes, status: data.status, errorMessage: null, updatedAt: new Date() })
      .where(and(eq(documents.id, id), eq(documents.orgId, tenantId)))
      .returning();
    return result;
  }

  async updateNotes(tenantId: string, id: string, notes: string | null) {
    this.requireTenant(tenantId);
    const [result] = await this.db.update(documents)
      .set({ notes, updatedAt: new Date() })
      .where(and(eq(documents.id, id), eq(documents.orgId, tenantId)))
      .returning();
    return result;
  }
}
