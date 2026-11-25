import { authorizeToolCall, detectOutOfAuthorityRequest } from './policy-guard.js';
import type { AgentPlan, AgentRole, AgentToolCall, AgentToolResult } from './types.js';

export const AGENT_MAX_STEPS = 3;

const AUDIT_LOG_SIGNALS =
  /\b(security\s+audit|audit\s+logs?|policy\s+denials?|denied\s+requests?|unauthorized\s+attempts?|agent\s+audit|log\s+keamanan)\b/i;

const MEMBER_SIGNALS =
  /\b(workspace\s+members?|library\s+members?|list\s+(the\s+)?members?|team\s+members?|who\s+(are|is)\s+(the\s+)?(members?|admins?|owners?|in\s+this\s+library)|member\s+roles?|active\s+invites?|pending\s+invites?|daftar\s+anggota)\b/i;

const INGESTION_JOB_SIGNALS =
  /\b(ingestion\s+jobs?|ingestion\s+queue|failed\s+(uploads?|jobs?|ingestions?)|processing\s+(queue|jobs?)|pending\s+jobs?|job\s+status|bullmq|status\s+antrian|gagal\s+upload)\b/i;

const GOVERNANCE_AUDIT_SIGNALS =
  /\b(who\s+approved|approval\s+(trail|history|audit)|governance\s+(history|audit|transitions?|status\s+history)|superseding\s+chain|superseded\s+documents?|audit\s+trail\s+of\s+policies|riwayat\s+persetujuan)\b/i;

const INVENTORY_SIGNALS =
  /\b(how\s+many\s+(active\s+|total\s+)?documents?|document\s+(count|inventory|stats|statistics|breakdown|statuses)|storage\s+(usage|used|consumed)|total\s+storage|list\s+(all\s+|active\s+|expiring\s+)?documents?|expiring\s+(soon|policies|documents?)|which\s+documents?\s+(are|have)\s+(failed|ready|expiring|superseded|archived|active)|berapa\s+jumlah\s+dokumen|daftar\s+dokumen)\b/i;

const DRAFT_SEARCH_SIGNALS =
  /\b(draft\s+(documents?|polic(?:y|ies)|sop|version)|unpublished\s+(draft|polic(?:y|ies)|documents?)|include\s+drafts?|in\s+drafts?|dokumen\s+draft)\b/i;

const DOC_CONTENT_SIGNALS =
  /\b(pol-[a-z0-9-]+|sop-[a-z0-9-]+|inc-[a-z0-9-]+|sla-[a-z0-9-]+|arc-[a-z0-9-]+|run-[a-z0-9-]+)\b/i;

const DOC_QUESTION_SIGNALS =
  /\b(what\s+does|according\s+to|policy\s+say|procedure|requirements?|reimbursement|mfa|rotation|escalation|compliance|retention|disaster\s+recovery|apa\s+isi|kebijakan|prosedur)\b/i;

const DIRECT_GREETING_SIGNALS =
  /^(hi|hello|hey|halo|thanks|thank you|terima kasih|what can you do|what tools do you have|help|bantuan)[\s!?.]*$/i;

function extractDaysParam(query: string, fallback = 90): number {
  const m = /(\d{1,3})\s*(days?|hari)/i.exec(query);
  if (!m?.[1]) return fallback;
  const n = parseInt(m[1], 10);
  return Number.isFinite(n) && n > 0 ? Math.min(365, n) : fallback;
}

/**
 * Deterministic, fast (<1ms), audit-logged agent tool planner.
 * Decides whether to search documents (`LightRagRetriever`), query parameterized
 * PostgreSQL metadata tools, combine both in a bounded multi-step plan (`<= AGENT_MAX_STEPS`),
 * answer directly, or deterministically refuse unauthorized/out-of-scope requests.
 */
export function planAgentToolsDeterministic(
  query: string,
  role: AgentRole,
  opts: { maxSteps?: number; orgId?: string } = {},
): AgentPlan {
  const maxSteps = Math.max(1, Math.min(AGENT_MAX_STEPS, opts.maxSteps ?? AGENT_MAX_STEPS));
  const q = query.trim();

  // 1. Fail-closed boundary check (secret exfil, cross-tenant, raw SQL, write mutations)
  const block = detectOutOfAuthorityRequest(q, role);
  if (block.blocked && block.code) {
    return {
      decision: 'refuse',
      refused: true,
      refusalCode: block.code,
      refusalReason: block.reason,
      deniedTool: 'refuse_out_of_scope',
      calls: [
        {
          tool: 'refuse_out_of_scope',
          params: { code: block.code },
          rationale: block.reason ?? 'Out of authority',
        },
      ],
      maxSteps,
    };
  }

  // 2. Capability intent matching
  const calls: AgentToolCall[] = [];
  const wantsAuditLogs = AUDIT_LOG_SIGNALS.test(q);
  const wantsMembers = MEMBER_SIGNALS.test(q);
  const wantsJobs = INGESTION_JOB_SIGNALS.test(q);
  const wantsGovernance = GOVERNANCE_AUDIT_SIGNALS.test(q);
  const wantsInventory = INVENTORY_SIGNALS.test(q);
  const wantsDrafts = DRAFT_SEARCH_SIGNALS.test(q);
  const wantsDocContent = DOC_CONTENT_SIGNALS.test(q) || DOC_QUESTION_SIGNALS.test(q);

  if (wantsAuditLogs) {
    calls.push({
      tool: 'query_security_audit_logs',
      params: { limit: 20 },
      rationale: 'Inspect tenant security & agent policy audit events',
    });
  }
  if (wantsMembers) {
    calls.push({
      tool: 'query_workspace_members',
      params: { limit: 50 },
      rationale: 'Inspect workspace membership roles and active invite count',
    });
  }
  if (wantsJobs) {
    const failedOnly = /\bfailed|gagal\b/i.test(q);
    calls.push({
      tool: 'query_ingestion_jobs',
      params: { ...(failedOnly ? { status: 'failed' } : {}), limit: 20 },
      rationale: 'Inspect structured BullMQ document ingestion job statuses',
    });
  }
  if (wantsGovernance) {
    calls.push({
      tool: 'query_governance_audit',
      params: { limit: 25 },
      rationale: 'Inspect structured approval & governance audit trail',
    });
  }
  if (wantsInventory) {
    const expiringSoon = /\bexpir/i.test(q);
    calls.push({
      tool: 'query_document_inventory',
      params: {
        includeDrafts: role === 'owner' || role === 'admin',
        ...(expiringSoon ? { expiringWithinDays: extractDaysParam(q, 90) } : {}),
        limit: 25,
      },
      rationale: 'Query live document counts, storage usage, and governance/status breakdown',
    });
  }
  if (wantsDrafts) {
    calls.push({
      tool: 'search_draft_documents',
      params: { includeDrafts: true, limit: 15 },
      rationale: 'Search unpublished draft documents alongside active policies',
    });
  } else if (wantsDocContent && (calls.length === 0 || calls.length < maxSteps)) {
    calls.push({
      tool: 'search_documents',
      params: { includeDrafts: false },
      rationale: 'Run hybrid HNSW + composite GIN + KG/PPR document retrieval',
    });
  }

  // 3. Direct answer or default fallback
  if (calls.length === 0) {
    if (DIRECT_GREETING_SIGNALS.test(q)) {
      return {
        decision: 'direct',
        refused: false,
        calls: [
          {
            tool: 'direct_answer',
            params: {},
            rationale: 'Respond directly with role-authorized agent capabilities',
          },
        ],
        maxSteps,
      };
    }

    calls.push({
      tool: 'search_documents',
      params: { includeDrafts: false },
      rationale: 'Default substantive query to verified document retrieval',
    });
  }

  const boundedCalls = calls.slice(0, maxSteps);
  const evalOrgId = opts.orgId ?? '00000000-0000-0000-0000-000000000000';

  // 4. Deterministic Zero-Trust Policy Gate over EVERY planned tool call (fail-closed if any denied)
  for (const call of boundedCalls) {
    const gate = authorizeToolCall({
      role,
      orgId: evalOrgId,
      tool: call.tool,
      params: call.params,
    });
    if (!gate.allowed) {
      return {
        decision: 'refuse',
        refused: true,
        refusalCode: gate.code ?? 'POLICY_ROLE_DENIED',
        refusalReason: gate.reason,
        deniedTool: call.tool,
        requiredRoles: gate.requiredRoles,
        calls: [
          {
            tool: 'refuse_out_of_scope',
            params: { attemptedTool: call.tool, requiredRoles: gate.requiredRoles },
            rationale: gate.reason,
          },
        ],
        maxSteps,
      };
    }
  }

  const isDirectOnly = boundedCalls.length === 1 && boundedCalls[0]?.tool === 'direct_answer';
  return {
    decision: isDirectOnly ? 'direct' : 'execute',
    refused: false,
    calls: boundedCalls,
    maxSteps,
  };
}

/**
 * Formats read-only structured database tool results into compact, non-hallucinated
 * prompt context for grounded answer generation.
 */
export function formatStructuredToolContext(results: AgentToolResult[]): string {
  const structured = results.filter(
    (r) =>
      r.tool !== 'search_documents' &&
      r.tool !== 'search_draft_documents' &&
      r.tool !== 'direct_answer' &&
      r.tool !== 'refuse_out_of_scope',
  );
  if (structured.length === 0) return '';

  const blocks = structured.map((r) => {
    const jsonSnippet = JSON.stringify(r.data, null, 2);
    return `### Tool: ${r.tool}\nSummary: ${r.summary}\n\`\`\`json\n${jsonSnippet}\n\`\`\``;
  });

  return [
    'VERIFIED DATABASE TOOL OUTPUT (Read-Only, Role-Authorized):',
    'Use these exact counts, statuses, and timestamps directly when answering structured/operational questions.',
    ...blocks,
  ].join('\n\n');
}
