import { and, desc, eq, ne, sql } from 'drizzle-orm';
import { BaseRepository } from './base.repository.js';
import { documents } from '../schema/documents.js';
import { ingestionJobs } from '../schema/jobs.js';
import { libraryInvites, memberships, users } from '../schema/users.js';
import { auditLogs } from '../schema/audit.js';

export interface DocumentInventoryResult {
  totalDocuments: number;
  totalStorageBytes: number;
  byStatus: Record<string, number>;
  byGovernanceStatus: Record<string, number>;
  draftsIncluded: boolean;
  items: Array<{
    id: string;
    title: string;
    fileType: string;
    status: string;
    governanceStatus: string;
    authorityTier: number;
    effectiveAt: string | null;
    expiresAt: string | null;
    updatedAt: string;
  }>;
}

export interface GovernanceAuditResult {
  byGovernanceStatus: Record<string, number>;
  supersedingDocuments: Array<{
    id: string;
    title: string;
    governanceStatus: string;
    supersedesDocumentId: string;
    effectiveAt: string | null;
  }>;
  recentTransitions: Array<{
    documentId: string;
    documentTitle: string;
    from: string;
    to: string;
    actorUserId: string;
    at: string;
    note?: string;
  }>;
}

export interface WorkspaceMembersResult {
  totalMembers: number;
  byRole: Record<string, number>;
  activeInvites: number;
  members: Array<{
    userId: string;
    email: string;
    fullName: string | null;
    role: string;
    joinedAt: string;
    documentCount: number;
  }>;
}

export interface IngestionJobsSummaryResult {
  totalJobs: number;
  byStatus: Record<string, number>;
  jobs: Array<{
    id: string;
    documentId: string;
    status: string;
    attempts: number;
    errorMessage: string | null;
    createdAt: string;
    updatedAt: string;
  }>;
}

export interface SecurityAuditLogRecord {
  id: string;
  action: string;
  userId: string | null;
  resourceType: string | null;
  details: unknown;
  createdAt: string;
}

/**
 * Parameterized, strictly tenant-scoped read-only database tools for the V3.1
 * RBAC Company Agent, plus auditable allow/deny logging into `audit_logs`.
 * All reads execute on `this.readDb` so high-concurrency agent queries stay off
 * the primary writer pool when `DATABASE_READ_URL` is configured.
 */
export class AgentToolsRepository extends BaseRepository {
  constructor() {
    super('agent_tools');
  }

  async queryDocumentInventory(
    orgId: string,
    opts: {
      includeDrafts?: boolean;
      status?: string;
      governanceStatus?: string;
      expiringWithinDays?: number;
      limit?: number;
    } = {},
  ): Promise<DocumentInventoryResult> {
    this.requireTenant(orgId);
    const includeDrafts = Boolean(opts.includeDrafts);
    const limit = Math.max(1, Math.min(50, Math.round(opts.limit ?? 25)));

    const baseConditions = [eq(documents.orgId, orgId), eq(documents.isDeleted, false)];
    if (!includeDrafts) {
      baseConditions.push(ne(documents.governanceStatus, 'draft'));
    }

    const [aggRows, statusRows, govRows] = await Promise.all([
      this.readDb
        .select({
          totalDocuments: sql<number>`count(*)::int`,
          totalStorageBytes: sql<number>`coalesce(sum(${documents.fileSizeBytes}), 0)::float8`,
        })
        .from(documents)
        .where(and(...baseConditions)),
      this.readDb
        .select({
          status: documents.status,
          cnt: sql<number>`count(*)::int`,
        })
        .from(documents)
        .where(and(...baseConditions))
        .groupBy(documents.status),
      this.readDb
        .select({
          governanceStatus: documents.governanceStatus,
          cnt: sql<number>`count(*)::int`,
        })
        .from(documents)
        .where(and(...baseConditions))
        .groupBy(documents.governanceStatus),
    ]);

    const itemConditions = [...baseConditions];
    if (opts.status) {
      itemConditions.push(eq(documents.status, opts.status as typeof documents.$inferSelect.status));
    }
    if (opts.governanceStatus) {
      itemConditions.push(eq(documents.governanceStatus, opts.governanceStatus));
    }
    if (typeof opts.expiringWithinDays === 'number' && opts.expiringWithinDays > 0) {
      const horizon = new Date(Date.now() + opts.expiringWithinDays * 86_400_000).toISOString();
      itemConditions.push(
        sql`${documents.expiresAt} IS NOT NULL AND ${documents.expiresAt} >= now() AND ${documents.expiresAt} <= ${horizon}::timestamptz`,
      );
    }

    const itemRows = await this.readDb
      .select({
        id: documents.id,
        title: documents.title,
        fileType: documents.fileType,
        status: documents.status,
        governanceStatus: documents.governanceStatus,
        authorityTier: documents.authorityTier,
        effectiveAt: documents.effectiveAt,
        expiresAt: documents.expiresAt,
        updatedAt: documents.updatedAt,
      })
      .from(documents)
      .where(and(...itemConditions))
      .orderBy(desc(documents.updatedAt))
      .limit(limit);

    const byStatus: Record<string, number> = {};
    for (const r of statusRows) byStatus[r.status] = Number(r.cnt);

    const byGovernanceStatus: Record<string, number> = {};
    for (const r of govRows) byGovernanceStatus[r.governanceStatus] = Number(r.cnt);

    return {
      totalDocuments: Number(aggRows[0]?.totalDocuments ?? 0),
      totalStorageBytes: Number(aggRows[0]?.totalStorageBytes ?? 0),
      byStatus,
      byGovernanceStatus,
      draftsIncluded: includeDrafts,
      items: itemRows.map((r) => ({
        id: r.id,
        title: r.title,
        fileType: r.fileType,
        status: r.status,
        governanceStatus: r.governanceStatus,
        authorityTier: r.authorityTier,
        effectiveAt: r.effectiveAt ? r.effectiveAt.toISOString() : null,
        expiresAt: r.expiresAt ? r.expiresAt.toISOString() : null,
        updatedAt: r.updatedAt.toISOString(),
      })),
    };
  }

  async queryGovernanceAudit(
    orgId: string,
    opts: { limit?: number } = {},
  ): Promise<GovernanceAuditResult> {
    this.requireTenant(orgId);
    const limit = Math.max(1, Math.min(50, Math.round(opts.limit ?? 25)));

    const [govCounts, docRows] = await Promise.all([
      this.readDb
        .select({
          governanceStatus: documents.governanceStatus,
          cnt: sql<number>`count(*)::int`,
        })
        .from(documents)
        .where(and(eq(documents.orgId, orgId), eq(documents.isDeleted, false)))
        .groupBy(documents.governanceStatus),
      this.readDb
        .select({
          id: documents.id,
          title: documents.title,
          governanceStatus: documents.governanceStatus,
          supersedesDocumentId: documents.supersedesDocumentId,
          effectiveAt: documents.effectiveAt,
          metadata: documents.metadata,
        })
        .from(documents)
        .where(and(eq(documents.orgId, orgId), eq(documents.isDeleted, false)))
        .orderBy(desc(documents.updatedAt))
        .limit(Math.max(limit * 2, 40)),
    ]);

    const byGovernanceStatus: Record<string, number> = {};
    for (const r of govCounts) byGovernanceStatus[r.governanceStatus] = Number(r.cnt);

    const supersedingDocuments: GovernanceAuditResult['supersedingDocuments'] = [];
    const recentTransitions: GovernanceAuditResult['recentTransitions'] = [];

    for (const doc of docRows) {
      if (doc.supersedesDocumentId && supersedingDocuments.length < limit) {
        supersedingDocuments.push({
          id: doc.id,
          title: doc.title,
          governanceStatus: doc.governanceStatus,
          supersedesDocumentId: doc.supersedesDocumentId,
          effectiveAt: doc.effectiveAt ? doc.effectiveAt.toISOString() : null,
        });
      }
      const meta = (doc.metadata ?? {}) as { governanceHistory?: unknown };
      if (Array.isArray(meta.governanceHistory)) {
        for (const raw of meta.governanceHistory) {
          if (!raw || typeof raw !== 'object') continue;
          const entry = raw as Record<string, unknown>;
          recentTransitions.push({
            documentId: doc.id,
            documentTitle: doc.title,
            from: String(entry.from ?? ''),
            to: String(entry.to ?? ''),
            actorUserId: String(entry.actorUserId ?? ''),
            at: String(entry.at ?? ''),
            ...(typeof entry.note === 'string' ? { note: entry.note } : {}),
          });
        }
      }
    }

    recentTransitions.sort((a, b) => b.at.localeCompare(a.at));

    return {
      byGovernanceStatus,
      supersedingDocuments,
      recentTransitions: recentTransitions.slice(0, limit),
    };
  }

  async queryWorkspaceMembers(
    orgId: string,
    opts: { limit?: number } = {},
  ): Promise<WorkspaceMembersResult> {
    this.requireTenant(orgId);
    const limit = Math.max(1, Math.min(50, Math.round(opts.limit ?? 50)));

    const [roleRows, inviteRows, memberRows] = await Promise.all([
      this.readDb
        .select({
          role: memberships.role,
          cnt: sql<number>`count(*)::int`,
        })
        .from(memberships)
        .where(and(eq(memberships.orgId, orgId), eq(memberships.isActive, true)))
        .groupBy(memberships.role),
      this.readDb
        .select({ cnt: sql<number>`count(*)::int` })
        .from(libraryInvites)
        .where(
          and(
            eq(libraryInvites.orgId, orgId),
            sql`${libraryInvites.revokedAt} IS NULL`,
            sql`(${libraryInvites.expiresAt} IS NULL OR ${libraryInvites.expiresAt} > now())`,
          ),
        ),
      this.readDb
        .select({
          userId: users.id,
          email: users.email,
          fullName: users.fullName,
          role: memberships.role,
          joinedAt: memberships.createdAt,
          documentCount: sql<number>`(select count(*)::int from ${documents} d where d.org_id = ${memberships.orgId} and d.uploaded_by = ${users.id} and d.is_deleted = false)`,
        })
        .from(memberships)
        .innerJoin(users, eq(users.id, memberships.userId))
        .where(and(eq(memberships.orgId, orgId), eq(memberships.isActive, true)))
        .orderBy(memberships.createdAt)
        .limit(limit),
    ]);

    const byRole: Record<string, number> = {};
    let totalMembers = 0;
    for (const r of roleRows) {
      const n = Number(r.cnt);
      byRole[r.role] = n;
      totalMembers += n;
    }

    return {
      totalMembers,
      byRole,
      activeInvites: Number(inviteRows[0]?.cnt ?? 0),
      members: memberRows.map((m) => ({
        userId: m.userId,
        email: m.email,
        fullName: m.fullName,
        role: m.role,
        joinedAt: m.joinedAt.toISOString(),
        documentCount: Number(m.documentCount ?? 0),
      })),
    };
  }

  async queryIngestionJobs(
    orgId: string,
    opts: { status?: string; limit?: number } = {},
  ): Promise<IngestionJobsSummaryResult> {
    this.requireTenant(orgId);
    const limit = Math.max(1, Math.min(50, Math.round(opts.limit ?? 20)));

    const statusCounts = await this.readDb
      .select({
        status: ingestionJobs.status,
        cnt: sql<number>`count(*)::int`,
      })
      .from(ingestionJobs)
      .where(eq(ingestionJobs.orgId, orgId))
      .groupBy(ingestionJobs.status);

    const jobConditions = [eq(ingestionJobs.orgId, orgId)];
    if (opts.status) {
      jobConditions.push(eq(ingestionJobs.status, opts.status as typeof ingestionJobs.$inferSelect.status));
    }

    const jobs = await this.readDb
      .select({
        id: ingestionJobs.id,
        documentId: ingestionJobs.documentId,
        status: ingestionJobs.status,
        attempts: ingestionJobs.attempts,
        errorMessage: ingestionJobs.errorMessage,
        createdAt: ingestionJobs.createdAt,
        updatedAt: ingestionJobs.updatedAt,
      })
      .from(ingestionJobs)
      .where(and(...jobConditions))
      .orderBy(desc(ingestionJobs.createdAt))
      .limit(limit);

    const byStatus: Record<string, number> = {};
    let totalJobs = 0;
    for (const r of statusCounts) {
      const n = Number(r.cnt);
      byStatus[r.status] = n;
      totalJobs += n;
    }

    return {
      totalJobs,
      byStatus,
      jobs: jobs.map((j) => ({
        id: j.id,
        documentId: j.documentId,
        status: j.status,
        attempts: j.attempts,
        errorMessage: j.errorMessage,
        createdAt: j.createdAt.toISOString(),
        updatedAt: j.updatedAt.toISOString(),
      })),
    };
  }

  async querySecurityAuditLogs(
    orgId: string,
    opts: { actionPrefix?: string; limit?: number } = {},
  ): Promise<SecurityAuditLogRecord[]> {
    this.requireTenant(orgId);
    const limit = Math.max(1, Math.min(50, Math.round(opts.limit ?? 20)));
    const conditions = [eq(auditLogs.orgId, orgId)];
    if (opts.actionPrefix) {
      conditions.push(sql`${auditLogs.action} LIKE ${`${opts.actionPrefix}%`}`);
    }

    const rows = await this.readDb
      .select({
        id: auditLogs.id,
        action: auditLogs.action,
        userId: auditLogs.userId,
        resourceType: auditLogs.resourceType,
        details: auditLogs.details,
        createdAt: auditLogs.createdAt,
      })
      .from(auditLogs)
      .where(and(...conditions))
      .orderBy(desc(auditLogs.createdAt))
      .limit(limit);

    return rows.map((r) => ({
      id: r.id,
      action: r.action,
      userId: r.userId,
      resourceType: r.resourceType,
      details: r.details,
      createdAt: r.createdAt.toISOString(),
    }));
  }

  async recordAgentAuditLog(entry: {
    orgId: string;
    userId?: string | null;
    action: 'agent:tool_allow' | 'agent:policy_deny';
    tool: string;
    role: string;
    allowed: boolean;
    code?: string;
    reason?: string;
    requestId?: string;
    latencyMs?: number;
    ipAddress?: string;
  }): Promise<void> {
    this.requireTenant(entry.orgId);
    try {
      await this.db.insert(auditLogs).values({
        orgId: entry.orgId,
        userId: entry.userId ?? null,
        action: entry.action,
        resourceType: 'agent_tool',
        details: {
          tool: entry.tool,
          role: entry.role,
          allowed: entry.allowed,
          ...(entry.code ? { code: entry.code } : {}),
          ...(entry.reason ? { reason: entry.reason } : {}),
          ...(entry.requestId ? { requestId: entry.requestId } : {}),
          ...(entry.latencyMs !== undefined ? { latencyMs: entry.latencyMs } : {}),
        },
        ipAddress: entry.ipAddress?.slice(0, 45) ?? null,
      });
    } catch {
      /* Audit logging is best-effort and must never fail authorized requests */
    }
  }
}
