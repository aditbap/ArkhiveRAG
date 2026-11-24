export interface TemporalQueryIntent {
  /** User explicitly asks for current / latest / active rules. */
  preferLatest: boolean;
  /** User explicitly asks about historical / superseded / older versions. */
  allowHistorical: boolean;
  /** Specific year mentioned in query (e.g. 2024, 2025, 2026), if any. */
  targetYear: number | null;
}

const LATEST_CUE_RE =
  /\b(?:latest|current|currently|active|effective|now|today|newest|updated|recent|terbaru|berlaku|saat\s+ini|terkini|sekarang)\b/i;
const HISTORICAL_CUE_RE =
  /\b(?:previous|prior|old|older|former|historical|history|superseded|deprecated|archived|legacy|before\s+20\d\d|sebelumnya|lama|terdahulu|riwayat)\b/i;
const YEAR_RE = /\b(20[1-3]\d)\b/g;

/**
 * Extracts temporal intent from English & Indonesian enterprise queries in <0.5ms.
 */
export function parseTemporalIntent(query: string): TemporalQueryIntent {
  const preferLatest = LATEST_CUE_RE.test(query);
  const allowHistorical = HISTORICAL_CUE_RE.test(query);

  const years: number[] = [];
  for (const m of query.matchAll(YEAR_RE)) {
    const y = Number(m[1]);
    if (Number.isFinite(y)) years.push(y);
  }
  const targetYear = years.length > 0 ? Math.max(...years) : null;

  return {
    preferLatest: preferLatest || (!allowHistorical && targetYear === null),
    allowHistorical,
    targetYear,
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

export interface TemporalCandidate {
  documentId: string;
  documentTitle: string;
  effectiveAt?: Date | string | null;
  expiresAt?: Date | string | null;
  supersedesDocumentId?: string | null;
  authorityTier?: number | null;
  createdAt?: Date | string | null;
  updatedAt?: Date | string | null;
}

export interface TemporalResolutionAnnotation {
  isSuperseded: boolean;
  supersededByDocId?: string;
  isExpired: boolean;
  /** Normalized recency score in [0, 1]. */
  recencyScore: number;
  /** Normalized authority score in [0, 1]. */
  authorityScore: number;
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

    const effMs = toTimestamp(c.effectiveAt) ?? toTimestamp(c.updatedAt) ?? toTimestamp(c.createdAt);
    const expMs = toTimestamp(c.expiresAt);
    const isExpired = expMs !== null && expMs < nowMs;
    const isSuperseded = supersededBy.has(c.documentId);
    const tier = c.authorityTier ?? inferAuthorityTier(c.documentTitle);
    const authorityScore = Math.max(0.25, Math.min(1, tier / 4));

    // Exponential recency decay: 1.0 for today, ~0.5 after 365 days, floors at 0.2
    let recencyScore = 0.7;
    if (effMs !== null) {
      const ageDays = Math.max(0, (nowMs - effMs) / 86_400_000);
      recencyScore = Math.max(0.2, Math.exp(-Math.LN2 * (ageDays / 365)));
    }

    // Specific year match boost/penalty
    let yearMultiplier = 1.0;
    if (intent.targetYear !== null && effMs !== null) {
      const docYear = new Date(effMs).getUTCFullYear();
      if (docYear === intent.targetYear) {
        yearMultiplier = 1.15;
      } else if (Math.abs(docYear - intent.targetYear) >= 2 && !intent.allowHistorical) {
        yearMultiplier = 0.85;
      }
    }

    let temporalMultiplier = yearMultiplier;
    if (!intent.allowHistorical) {
      if (isSuperseded) {
        // Keep a small floor if user explicitly asked for that doc's year; otherwise strongly demote
        const docYear = effMs !== null ? new Date(effMs).getUTCFullYear() : null;
        temporalMultiplier *= intent.targetYear !== null && docYear === intent.targetYear ? 0.9 : 0.35;
      } else if (isExpired) {
        temporalMultiplier *= 0.55;
      }
    }

    const badgeParts: string[] = [];
    if (tier >= 3 || tier === 1) badgeParts.push(authorityTierLabel(tier));
    if (c.effectiveAt) {
      const effDate = new Date(effMs!).toISOString().slice(0, 10);
      badgeParts.push(`Effective ${effDate}`);
    }
    if (isSuperseded) badgeParts.push('SUPERSEDED');
    else if (isExpired) badgeParts.push('EXPIRED');

    out.set(c.documentId, {
      isSuperseded,
      supersededByDocId: supersededBy.get(c.documentId),
      isExpired,
      recencyScore,
      authorityScore,
      temporalMultiplier,
      badge: badgeParts.length > 0 ? badgeParts.join(' | ') : undefined,
    });
  }

  return out;
}
