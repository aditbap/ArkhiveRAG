export type AgentRole = 'owner' | 'admin' | 'member' | 'viewer';

export type AgentToolName =
  | 'search_documents'
  | 'search_draft_documents'
  | 'query_document_inventory'
  | 'query_governance_audit'
  | 'query_workspace_members'
  | 'query_ingestion_jobs'
  | 'query_security_audit_logs'
  | 'direct_answer'
  | 'refuse_out_of_scope';

export type AgentRefusalCode =
  | 'POLICY_ROLE_DENIED'
  | 'POLICY_WRITE_FORBIDDEN'
  | 'POLICY_CROSS_TENANT_FORBIDDEN'
  | 'POLICY_SECRET_EXFIL_FORBIDDEN'
  | 'POLICY_RAW_SQL_FORBIDDEN';

export interface AgentToolSpec {
  name: AgentToolName;
  description: string;
  allowedRoles: readonly AgentRole[];
  readOnly: true;
  maxRows: number;
}

export interface AgentToolCall {
  tool: AgentToolName;
  params: Record<string, unknown>;
  rationale: string;
}

export interface AgentPolicyDecision {
  allowed: boolean;
  code?: AgentRefusalCode;
  reason: string;
  requiredRoles: readonly AgentRole[];
}

export interface AgentPlan {
  decision: 'execute' | 'direct' | 'refuse';
  refused: boolean;
  refusalCode?: AgentRefusalCode;
  refusalReason?: string;
  deniedTool?: AgentToolName;
  requiredRoles?: readonly AgentRole[];
  calls: AgentToolCall[];
  maxSteps: number;
}

export interface AgentStepRecord {
  step: number;
  tool: AgentToolName;
  outcome: 'allowed' | 'denied' | 'error';
  code?: AgentRefusalCode;
  latencyMs: number;
  summary: string;
}

export interface AgentToolResult {
  tool: AgentToolName;
  summary: string;
  data: unknown;
}

export interface AgentEvalCase {
  id: string;
  category:
    | 'doc_search'
    | 'structured_db'
    | 'hybrid_multi_tool'
    | 'role_refusal'
    | 'security_attack'
    | 'direct_answer';
  role: AgentRole;
  query: string;
  expectedDecision: 'execute' | 'direct' | 'refuse';
  expectedTools: AgentToolName[];
  expectedRefusalCode?: AgentRefusalCode;
  assertStructured?: (results: AgentToolResult[]) => boolean;
}

export interface AgentEvalCaseOutcome {
  id: string;
  category: AgentEvalCase['category'];
  role: AgentRole;
  passed: boolean;
  toolMatched: boolean;
  refusalMatched: boolean;
  structuredExact: boolean;
  withinBudget: boolean;
  actualDecision: AgentPlan['decision'];
  actualTools: AgentToolName[];
  actualRefusalCode?: AgentRefusalCode;
  latencyMs: number;
  reason?: string;
}

export interface AgentEvalReport {
  totalCases: number;
  passedCases: number;
  toolSelectionAccuracy: number;
  authorityRefusalRecall: number;
  authorizedFalseRefusalRate: number;
  structuredExactness: number;
  budgetComplianceRate: number;
  gatePassed: boolean;
  outcomes: AgentEvalCaseOutcome[];
}
