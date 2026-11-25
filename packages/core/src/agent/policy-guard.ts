import type {
  AgentPolicyDecision,
  AgentRefusalCode,
  AgentRole,
  AgentToolName,
  AgentToolSpec,
} from './types.js';

const ALL_ROLES: readonly AgentRole[] = ['owner', 'admin', 'member', 'viewer'];
const WRITE_MEMBERS: readonly AgentRole[] = ['owner', 'admin', 'member'];
const ADMINS: readonly AgentRole[] = ['owner', 'admin'];
const OWNER_ONLY: readonly AgentRole[] = ['owner'];

export const TOOL_POLICY_MATRIX: Record<AgentToolName, AgentToolSpec> = {
  search_documents: {
    name: 'search_documents',
    description: 'Hybrid HNSW + composite GIN + KG/PPR retrieval over active authorized documents.',
    allowedRoles: ALL_ROLES,
    readOnly: true,
    maxRows: 30,
  },
  search_draft_documents: {
    name: 'search_draft_documents',
    description: 'Search unpublished draft documents in addition to active policies (Admin/Owner only).',
    allowedRoles: ADMINS,
    readOnly: true,
    maxRows: 20,
  },
  query_document_inventory: {
    name: 'query_document_inventory',
    description: 'Parameterized read-only inventory: document counts by status/governance, storage usage, and expiring docs.',
    allowedRoles: ALL_ROLES,
    readOnly: true,
    maxRows: 50,
  },
  query_governance_audit: {
    name: 'query_governance_audit',
    description: 'Inspect approval workflow transitions, superseding chains, and governance metadata audit history (Admin/Owner only).',
    allowedRoles: ADMINS,
    readOnly: true,
    maxRows: 50,
  },
  query_workspace_members: {
    name: 'query_workspace_members',
    description: 'Read workspace member roles, join dates, upload counts, and active invite counts (Admin/Owner only).',
    allowedRoles: ADMINS,
    readOnly: true,
    maxRows: 50,
  },
  query_ingestion_jobs: {
    name: 'query_ingestion_jobs',
    description: 'Inspect BullMQ document ingestion queue status counts, attempts, and failure diagnostics (Member+ only).',
    allowedRoles: WRITE_MEMBERS,
    readOnly: true,
    maxRows: 50,
  },
  query_security_audit_logs: {
    name: 'query_security_audit_logs',
    description: 'Read tenant security & agent policy denial audit logs (Owner only).',
    allowedRoles: OWNER_ONLY,
    readOnly: true,
    maxRows: 50,
  },
  direct_answer: {
    name: 'direct_answer',
    description: 'Respond to greetings or explain the tools authorized for the caller role without querying storage.',
    allowedRoles: ALL_ROLES,
    readOnly: true,
    maxRows: 1,
  },
  refuse_out_of_scope: {
    name: 'refuse_out_of_scope',
    description: 'Deterministic auditable refusal for unauthorized, cross-tenant, raw-SQL, or write requests.',
    allowedRoles: ALL_ROLES,
    readOnly: true,
    maxRows: 1,
  },
};

export function normalizeAgentRole(role: string | null | undefined): AgentRole {
  const normalized = (role || '').trim().toLowerCase();
  if (normalized === 'owner' || normalized === 'admin' || normalized === 'member' || normalized === 'viewer') {
    return normalized;
  }
  return 'viewer';
}

export function listToolsForRole(role: AgentRole): {
  authorized: AgentToolSpec[];
  restricted: AgentToolSpec[];
} {
  const authorized: AgentToolSpec[] = [];
  const restricted: AgentToolSpec[] = [];
  for (const spec of Object.values(TOOL_POLICY_MATRIX)) {
    if (spec.name === 'refuse_out_of_scope') continue;
    if (spec.allowedRoles.includes(role)) {
      authorized.push(spec);
    } else {
      restricted.push(spec);
    }
  }
  return { authorized, restricted };
}

export interface AuthorityBlockCheck {
  blocked: boolean;
  code?: AgentRefusalCode;
  reason?: string;
}

const SECRET_EXFIL_PATTERNS: readonly RegExp[] = [
  /\b(system prompt|hidden prompt|developer instructions|reveal your instructions|print your prompt|show me your prompt)\b/i,
  /\b(database_url|cloudflare_api_token|openai_api_key|gemini_api_key|nvidia_api_key|minio_secret_key|auth_secret|jwt secret|postgres password|env variables|dotenv|\.env)\b/i,
  /\b(api key|secret key|private key|credentials|kunci rahasia|password database)\b/i,
];

const CROSS_TENANT_PATTERNS: readonly RegExp[] = [
  /\b(other org(anization)?|another org(anization)?|different org(anization)?|other tenant|another tenant|cross[- ]tenant|all tenants|every organization)\b/i,
  /\b(tenant[\s_-]?b|org[\s_-]?b|organisasi lain|tenant lain|semua organisasi|perusahaan lain)\b/i,
];

const RAW_SQL_PATTERNS: readonly RegExp[] = [
  /\b(select\s+\*|select\s+.+\s+from\s+(document_chunks|documents|users|memberships|audit_logs|organizations)|drop\s+table|truncate\s+table|alter\s+table|union\s+select)\b/i,
  /\b(execute\s+(raw\s+)?sql|run\s+(a\s+)?raw\s+(sql|query)|jalankan\s+sql)\b/i,
];

const WRITE_MUTATION_PATTERNS: readonly RegExp[] = [
  /\b(delete|remove|purge|wipe|drop|erase)\s+(all\s+|the\s+|that\s+|every\s+)?(documents?|files?|chunks?|members?|users?|library|libraries|invites?)\b/i,
  /\b(promote|upgrade|escalate|grant|assign|change)\s+(my\s+|me\s+|user\s+|member\s+)?(to\s+|role\s+to\s+)?(owner|admin|role)\b/i,
  /\b(make me|jadikan saya)\s+(an?\s+)?(owner|admin)\b/i,
  /\b(approve|publish|supersede|archive)\s+(that\s+|this\s+|the\s+)?(draft|document|policy)\b/i,
  /\b(hapus semua dokumen|hapus anggota|ubah role)\b/i,
];

/**
 * Deterministically inspects user requests BEFORE planning/execution for attempts that
 * violate security boundaries regardless of role (secret exfiltration, cross-tenant reads,
 * raw SQL execution, or mutating actions via the read-only agent).
 *
 * Role-gated read capabilities (drafts, governance audit, members, jobs, audit logs) are
 * NOT blocked here so `planAgentToolsDeterministic` maps them to their explicit tool name
 * and `authorizeToolCall` denies them with `POLICY_ROLE_DENIED` + required roles.
 */
export function detectOutOfAuthorityRequest(query: string, _role: AgentRole): AuthorityBlockCheck {
  const q = query.trim();
  if (!q) return { blocked: false };

  if (SECRET_EXFIL_PATTERNS.some((re) => re.test(q))) {
    return {
      blocked: true,
      code: 'POLICY_SECRET_EXFIL_FORBIDDEN',
      reason: 'Refused: system instructions, environment variables, and credentials are never accessible through the agent.',
    };
  }

  if (CROSS_TENANT_PATTERNS.some((re) => re.test(q))) {
    return {
      blocked: true,
      code: 'POLICY_CROSS_TENANT_FORBIDDEN',
      reason: 'Refused: cross-organization / cross-tenant data access is strictly forbidden; all queries are bound to your verified organization.',
    };
  }

  if (RAW_SQL_PATTERNS.some((re) => re.test(q))) {
    return {
      blocked: true,
      code: 'POLICY_RAW_SQL_FORBIDDEN',
      reason: 'Refused: arbitrary or raw SQL execution is prohibited; only allowlisted, parameterized read-only tools may run.',
    };
  }

  if (WRITE_MUTATION_PATTERNS.some((re) => re.test(q))) {
    return {
      blocked: true,
      code: 'POLICY_WRITE_FORBIDDEN',
      reason: 'Refused: the company agent operates strictly read-only. Perform document deletions, governance approvals, or role updates via the authenticated REST API.',
    };
  }

  return { blocked: false };
}

/**
 * Deterministic zero-trust policy gate invoked immediately before EVERY tool execution.
 */
export function authorizeToolCall(input: {
  role: AgentRole;
  orgId: string;
  tool: AgentToolName;
  params?: Record<string, unknown>;
}): AgentPolicyDecision {
  const spec = TOOL_POLICY_MATRIX[input.tool];
  if (!input.orgId || !input.orgId.trim()) {
    return {
      allowed: false,
      code: 'POLICY_CROSS_TENANT_FORBIDDEN',
      reason: 'Refused: missing verified organization scope.',
      requiredRoles: spec.allowedRoles,
    };
  }

  if (!spec.allowedRoles.includes(input.role)) {
    return {
      allowed: false,
      code: 'POLICY_ROLE_DENIED',
      reason: `Refused: tool '${spec.name}' requires role [${spec.allowedRoles.join(', ')}], but caller holds '${input.role}'.`,
      requiredRoles: spec.allowedRoles,
    };
  }

  if (
    input.params?.includeDrafts === true &&
    !TOOL_POLICY_MATRIX.search_draft_documents.allowedRoles.includes(input.role)
  ) {
    return {
      allowed: false,
      code: 'POLICY_ROLE_DENIED',
      reason: `Refused: draft document visibility requires role [owner, admin], but caller holds '${input.role}'.`,
      requiredRoles: ADMINS,
    };
  }

  if (input.params && typeof input.params.limit === 'number') {
    input.params.limit = Math.max(1, Math.min(spec.maxRows, Math.round(input.params.limit)));
  }

  return {
    allowed: true,
    reason: `Authorized '${spec.name}' for role '${input.role}'.`,
    requiredRoles: spec.allowedRoles,
  };
}
