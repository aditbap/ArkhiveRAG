export interface TemporalQueryIntent {
  /** User explicitly asks for current / latest / active rules. */
  preferLatest: boolean;
  /** User explicitly asks about historical / superseded / older versions. */
  allowHistorical: boolean;
  /** Specific target year requested in query (e.g. 2024), if any. */
  targetYear: number | null;
  /** Explicit upper-bound exclusion year (e.g. "before the 2026 update" -> 2026), if any. */
  beforeYear: number | null;
}

const LATEST_CUE_RE =
  /\b(?:latest|current|currently|active|effective|now|today|newest|updated|recent|terbaru|berlaku|saat\s+ini|terkini|sekarang)\b/i;
const HISTORICAL_CUE_RE =
  /\b(?:previous|prior|old|older|former|historical|history|superseded|deprecated|archived|legacy|before\s+(?:the\s+)?20\d\d|prior\s+to\s+(?:the\s+)?20\d\d|sebelum(?:nya)?\s+(?:tahun\s+)?20\d\d|sebelumnya|lama|terdahulu|riwayat)\b/i;
const BEFORE_YEAR_RE = /\b(?:before|prior\s+to|sebelum)\s+(?:the\s+|tahun\s+)?(20[1-3]\d)\b/i;
const YEAR_RE = /\b(20[1-3]\d)\b/g;

/**
 * Extracts temporal intent from English & Indonesian enterprise queries in <0.5ms.
 * Properly distinguishes historical point-in-time queries ("previous 2024 ... before the 2026 update")
 * from current-version queries.
 */
export function parseTemporalIntent(query: string): TemporalQueryIntent {
  const preferLatest = LATEST_CUE_RE.test(query);
  const allowHistorical = HISTORICAL_CUE_RE.test(query);

  const beforeMatch = query.match(BEFORE_YEAR_RE);
  const beforeYear = beforeMatch ? Number(beforeMatch[1]) : null;

  const years: number[] = [];
  for (const m of query.matchAll(YEAR_RE)) {
    const y = Number(m[1]);
    if (Number.isFinite(y)) years.push(y);
  }

  let targetYear: number | null = null;
  if (years.length > 0) {
    if (allowHistorical) {
      const nonUpperBound = beforeYear !== null ? years.filter((y) => y !== beforeYear) : years;
      targetYear = nonUpperBound.length > 0 ? Math.min(...nonUpperBound) : null;
    } else {
      targetYear = Math.max(...years);
    }
  }

  return {
    preferLatest: !allowHistorical && (preferLatest || targetYear === null),
    allowHistorical,
    targetYear,
    beforeYear,
  };
}

/**
 * Corporate authority tier mapping:
 * 4 = Binding Policy / Standard / Charter / Regulation
 * 3 = SOP / Runbook / SLA / Architecture Spec
 * 2 = Report / Memo / Guide / General Document (default)
 * 1 = Draft / Meeting Notes / Scratch / Unverified
 */
export function inferAuthorityTier(
  title: string,
  fileName?: string,
  metadata?: Record<string, unknown>,
): number {
  const explicit = metadata?.authorityTier;
  if (typeof explicit === 'number' && explicit >= 1 && explicit <= 4) {
    return Math.round(explicit);
  }
  const combined = `${title} ${fileName ?? ''}`.toLowerCase();
  if (/\b(?:draft|wip|notes|scratch|brainstorm|meeting\s+minutes)\b/.test(combined)) return 1;
  if (/\b(?:pol-[a-z0-9-]+|policy|kebijakan|charter|regulation|compliance|governance|security\s+standard)\b/.test(combined)) {
    return 4;
  }
  if (/\b(?:sop-[a-z0-9-]+|sop|runbook|sla|procedure|prosedur|architecture|playbook|handbook)\b/.test(combined)) {
    return 3;
  }
  return 2;
}

export function authorityTierLabel(tier: number): string {
  switch (tier) {
    case 4:
      return 'Policy';
    case 3:
      return 'SOP/Standard';
    case 1:
      return 'Draft/Notes';
    default:
      return 'Document';
  }
}

export type DocumentGovernanceStatus = 'draft' | 'approved' | 'active' | 'superseded' | 'archived';

export const VALID_GOVERNANCE_STATUSES: readonly DocumentGovernanceStatus[] = [
  'draft',
  'approved',
  'active',
  'superseded',
  'archived',
] as const;

export function isDocumentGovernanceStatus(v: unknown): v is DocumentGovernanceStatus {
  return typeof v === 'string' && (VALID_GOVERNANCE_STATUSES as readonly string[]).includes(v);
}

const ALLOWED_GOVERNANCE_TRANSITIONS: Record<DocumentGovernanceStatus, DocumentGovernanceStatus[]> = {
  draft: ['approved', 'active', 'archived'],
  approved: ['active', 'draft', 'archived'],
  active: ['superseded', 'archived'],
  superseded: ['archived', 'active'],
  archived: ['draft'],
};

/**
 * Validates enterprise document approval workflow transitions with RBAC enforcement:
 * - Only `owner` or `admin` may approve, activate, supersede, or archive documents.
 * - Enforces legal lifecycle state transitions (e.g. `draft -> approved -> active -> superseded -> archived`).
 */
export function validateGovernanceTransition(
  fromStatus: DocumentGovernanceStatus,
  toStatus: DocumentGovernanceStatus,
  actorRole: string,
): { allowed: boolean; reason?: string } {
  if (fromStatus === toStatus) {
    return { allowed: true };
  }
  const role = actorRole.toLowerCase();
  if (role !== 'owner' && role !== 'admin') {
    return {
      allowed: false,
      reason: `Role "${actorRole}" cannot transition governance status to "${toStatus}" (owner/admin required)`,
    };
  }
  const allowedTargets = ALLOWED_GOVERNANCE_TRANSITIONS[fromStatus] ?? [];
  if (!allowedTargets.includes(toStatus)) {
    return {
      allowed: false,
      reason: `Illegal governance transition "${fromStatus}" -> "${toStatus}" (allowed: ${allowedTargets.join(', ')})`,
    };
  }
  return { allowed: true };
}

/**
 * Builds a version-aware, ACL-version-aware, role-visibility-scoped chat cache key.
 * Prevents stale answers from leaking after document edits, governance changes, or
 * library member permission revocations/role downgrades.
 */
export function buildVersionedChatCacheKey(params: {
  orgId: string;
  docCacheVersion: string;
  aclVersion: string;
  includeDrafts: boolean;
  ragMode: string;
  documentIds?: string[];
  asOfDate?: Date | null;
  queryHash: string;
}): string {
  const docScope =
    Array.isArray(params.documentIds) && params.documentIds.length > 0
      ? [...params.documentIds].sort().join(',')
      : 'all';
  const draftFlag = params.includeDrafts ? 'd1' : 'd0';
  const asOfFlag = params.asOfDate ? params.asOfDate.toISOString().slice(0, 10) : 'now';
  return `cache:chat:org:${params.orgId}:dv${params.docCacheVersion}:av${params.aclVersion}:${draftFlag}:asof_${asOfFlag}:mode:${params.ragMode}:docs:${docScope}:q:${params.queryHash}`;
}

export interface TemporalCandidate {
  documentId: string;
  documentTitle: string;
  effectiveAt?: Date | string | null;
  expiresAt?: Date | string | null;
  /** Optional aliases for effectiveAt / expiresAt (`effective_from` / `effective_until`). */
  effectiveFrom?: Date | string | null;
  effectiveUntil?: Date | string | null;
  supersedesDocumentId?: string | null;
  authorityTier?: number | null;
  governanceStatus?: DocumentGovernanceStatus | string | null;
  createdAt?: Date | string | null;
  updatedAt?: Date | string | null;
}

/**
 * Point-in-time validity check: returns true if the document's `[effectiveFrom, effectiveUntil)`
 * window covers `asOfDate`. Documents without explicit bounds are treated as open-ended.
 */
export function isDocumentEffectiveAsOf(doc: TemporalCandidate, asOfDate: Date): boolean {
  const targetMs = asOfDate.getTime();
  const startMs = toTimestamp(doc.effectiveAt ?? doc.effectiveFrom);
  const endMs = toTimestamp(doc.expiresAt ?? doc.effectiveUntil);
  if (startMs !== null && targetMs < startMs) return false;
  if (endMs !== null && targetMs >= endMs) return false;
  return true;
}

export interface TemporalResolutionAnnotation {
  isSuperseded: boolean;
  supersededByDocId?: string;
  isExpired: boolean;
  /** Normalized recency/temporal-fit score in [0, 1]. */
  recencyScore: number;
  /** Normalized authority score in [0, 1]. */
  authorityScore: number;
  /** Version-alignment feature in [0, 1]: prefers active versions normally, or target historical versions when requested. */
  versionMatchScore: number;
  /** Multiplier applied when temporal conflict resolution adjusts score. */
  temporalMultiplier: number;
  /** Compact metadata badge injected into prompt context for conflict clarity. */
  badge?: string;
}

function toTimestamp(value: Date | string | null | undefined): number | null {
  if (!value) return null;
  const d = value instanceof Date ? value : new Date(value);
  const ms = d.getTime();
  return Number.isFinite(ms) ? ms : null;
}

/**
 * Resolves temporal & authority conflicts across a candidate set:
 * - Detects documents explicitly superseded by another document (`supersedesDocumentId`)
 *   OR expired (`expiresAt < now`).
 * - Computes normalized `recencyScore` (exponential half-life decay over 365 days)
 *   and `authorityScore` (`authorityTier / 4`).
 * - Demotes superseded/expired documents unless the user explicitly asked for historical info
 *   or a specific past year matching that document.
 * - Produces explicit badges (`[Policy | Effective 2026-01-01]` / `[SUPERSEDED]`) so the
 *   generator LLM resolves conflicts transparently instead of mixing old and new rules.
 */
export function resolveTemporalAndAuthorityConflicts<T extends TemporalCandidate>(
  candidates: T[],
  intent: TemporalQueryIntent,
  now: Date = new Date(),
): Map<string, TemporalResolutionAnnotation> {
  const nowMs = now.getTime();
  const supersededBy = new Map<string, string>();

  for (const c of candidates) {
    if (c.supersedesDocumentId) {
      supersededBy.set(c.supersedesDocumentId, c.documentId);
    }
  }

  const out = new Map<string, TemporalResolutionAnnotation>();
  const seenDocs = new Set<string>();

  for (const c of candidates) {
    if (seenDocs.has(c.documentId)) continue;
    seenDocs.add(c.documentId);

    const effMs =
      toTimestamp(c.effectiveAt ?? c.effectiveFrom) ??
      toTimestamp(c.updatedAt) ??
      toTimestamp(c.createdAt);
    const expMs = toTimestamp(c.expiresAt ?? c.effectiveUntil);
    const govStatus = (c.governanceStatus ?? 'active').toLowerCase();
    const isExpired = expMs !== null && expMs < nowMs;
    const isSuperseded = supersededBy.has(c.documentId) || govStatus === 'superseded';
    const isDraft = govStatus === 'draft';
    const isArchived = govStatus === 'archived';
    const tier = c.authorityTier ?? inferAuthorityTier(c.documentTitle);
    const authorityScore = Math.max(0.25, Math.min(1, tier / 4));

    // Exponential recency decay: 1.0 for today, ~0.5 after 365 days, floors at 0.2
    let recencyScore = 0.7;
    const docYear = effMs !== null ? new Date(effMs).getUTCFullYear() : null;
    if (effMs !== null) {
      const ageDays = Math.max(0, (nowMs - effMs) / 86_400_000);
      recencyScore = Math.max(0.2, Math.exp(-Math.LN2 * (ageDays / 365)));
    }

    // Specific year & historical point-in-time alignment
    let yearMultiplier = 1.0;
    let versionMatchScore = isSuperseded ? 0.0 : isExpired ? 0.3 : 1.0;

    if (intent.allowHistorical) {
      if (intent.targetYear !== null && docYear !== null) {
        if (docYear === intent.targetYear) {
          yearMultiplier = 1.25;
          versionMatchScore = 1.0;
          recencyScore = 0.9;
        } else if (intent.beforeYear !== null && docYear >= intent.beforeYear) {
          yearMultiplier = 0.72;
          versionMatchScore = 0.2;
          recencyScore = 0.2;
        } else {
          versionMatchScore = isSuperseded ? 0.85 : 0.65;
        }
      } else if (intent.beforeYear !== null && docYear !== null) {
        if (docYear < intent.beforeYear) {
          yearMultiplier = 1.18;
          versionMatchScore = 1.0;
          recencyScore = 0.85;
        } else {
          yearMultiplier = 0.72;
          versionMatchScore = 0.2;
          recencyScore = 0.2;
        }
      } else {
        versionMatchScore = isSuperseded ? 0.9 : 0.75;
      }
    } else if (intent.targetYear !== null && docYear !== null) {
      if (docYear === intent.targetYear) {
        yearMultiplier = 1.15;
      } else if (Math.abs(docYear - intent.targetYear) >= 2) {
        yearMultiplier = 0.85;
      }
    }

    let temporalMultiplier = yearMultiplier;
    if (isDraft) {
      // Unapproved drafts are strongly demoted unless explicitly allowed
      temporalMultiplier *= 0.25;
      versionMatchScore = Math.min(versionMatchScore, 0.15);
    } else if (!intent.allowHistorical) {
      if (isSuperseded) {
        // Keep a small floor if user explicitly asked for that doc's year; otherwise strongly demote
        temporalMultiplier *= intent.targetYear !== null && docYear === intent.targetYear ? 0.9 : 0.35;
      } else if (isArchived) {
        temporalMultiplier *= 0.3;
        versionMatchScore = Math.min(versionMatchScore, 0.2);
      } else if (isExpired) {
        temporalMultiplier *= 0.55;
      }
    }

    const badgeParts: string[] = [];
    if (tier >= 3 || tier === 1) badgeParts.push(authorityTierLabel(tier));
    if (c.effectiveAt || c.effectiveFrom) {
      const effDate = new Date(effMs!).toISOString().slice(0, 10);
      badgeParts.push(`Effective ${effDate}`);
    }
    if (isDraft) badgeParts.push('DRAFT');
    else if (isSuperseded) badgeParts.push('SUPERSEDED');
    else if (isArchived) badgeParts.push('ARCHIVED');
    else if (isExpired) badgeParts.push('EXPIRED');

    out.set(c.documentId, {
      isSuperseded,
      supersededByDocId: supersededBy.get(c.documentId),
      isExpired,
      recencyScore,
      authorityScore,
      versionMatchScore,
      temporalMultiplier,
      badge: badgeParts.length > 0 ? badgeParts.join(' | ') : undefined,
    });
  }

  return out;
}
