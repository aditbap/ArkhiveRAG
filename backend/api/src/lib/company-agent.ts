import {
  AGENT_MAX_STEPS,
  authorizeToolCall,
  formatStructuredToolContext,
  listToolsForRole,
  normalizeAgentRole,
  planAgentToolsDeterministic,
  type AgentPlan,
  type AgentRefusalCode,
  type AgentRole,
  type AgentStepRecord,
  type AgentToolName,
  type AgentToolResult,
  type RagQueryMode,
  type RetrievalConfigPayload,
  type TokenBudgetConfig,
} from '@rag/core';
import type { AgentToolsRepository } from '@rag/database';
import type { LLMProvider } from '@rag/providers';
import type { LightRagRetrievalResult, LightRagRetriever } from './lightrag-retriever.js';

export interface CompanyAgentTelemetryHooks {
  onToolCall?: (tool: AgentToolName, outcome: 'allowed' | 'denied') => void;
  onPolicyDenial?: (role: AgentRole, code: AgentRefusalCode) => void;
  onStepsCompleted?: (steps: number) => void;
}

export interface CompanyAgentDeps {
  retriever?: LightRagRetriever | null;
  toolsRepo: Pick<
    AgentToolsRepository,
    | 'queryDocumentInventory'
    | 'queryGovernanceAudit'
    | 'queryWorkspaceMembers'
    | 'queryIngestionJobs'
    | 'querySecurityAuditLogs'
    | 'recordAgentAuditLog'
  >;
  llm?: LLMProvider | null;
  telemetry?: CompanyAgentTelemetryHooks;
}

export interface CompanyAgentRunInput {
  orgId: string;
  userId?: string | null;
  role: string | AgentRole;
  userQuery: string;
  history?: Array<{ role: string; content: string }>;
  documentIds?: string[];
  mode?: RagQueryMode;
  rerankLimit?: number;
  rerankMinScore?: number;
  retrievalConfig?: RetrievalConfigPayload;
  budget?: Partial<TokenBudgetConfig>;
  maxSteps?: number;
  requestId?: string;
  ipAddress?: string;
  synthesizeAnswer?: boolean;
}

export interface CompanyAgentRunResult {
  decision: AgentPlan['decision'];
  role: AgentRole;
  refused: boolean;
  refusalCode?: AgentRefusalCode;
  refusalReason?: string;
  requiredRoles?: readonly AgentRole[];
  steps: AgentStepRecord[];
  results: AgentToolResult[];
  structuredContextText: string;
  retrieval: LightRagRetrievalResult | null;
  answer: string;
  timingsMs: {
    planning: number;
    execution: number;
    synthesis: number;
    total: number;
  };
}

function summarizeToolData(tool: AgentToolName, data: unknown): string {
  if (!data || typeof data !== 'object') return `Executed ${tool}`;
  const obj = data as Record<string, unknown>;
  switch (tool) {
    case 'query_document_inventory':
      return `Inventory: ${obj.totalDocuments ?? 0} docs (${Number(obj.totalStorageBytes ?? 0)} bytes, draftsIncluded=${String(obj.draftsIncluded)})`;
    case 'query_governance_audit':
      return `Governance audit: ${Array.isArray(obj.recentTransitions) ? obj.recentTransitions.length : 0} transitions, ${Array.isArray(obj.supersedingDocuments) ? obj.supersedingDocuments.length : 0} superseding docs`;
    case 'query_workspace_members':
      return `Members: ${obj.totalMembers ?? 0} active members, ${obj.activeInvites ?? 0} active invites`;
    case 'query_ingestion_jobs':
      return `Ingestion jobs: ${obj.totalJobs ?? 0} total (${JSON.stringify(obj.byStatus ?? {})})`;
    case 'query_security_audit_logs':
      return `Security audit logs: ${Array.isArray(data) ? data.length : 0} recent entries`;
    case 'direct_answer':
      return 'Provided role-authorized capability overview directly';
    default:
      return `Executed ${tool}`;
  }
}

function renderDeterministicAnswer(
  role: AgentRole,
  results: AgentToolResult[],
  retrieval: LightRagRetrievalResult | null,
): string {
  const parts: string[] = [];
  for (const res of results) {
    if (res.tool === 'direct_answer') {
      const cap = listToolsForRole(role);
      parts.push(
        `You are signed in with role \`${role}\`. Authorized tools: ${cap.authorized.map((t) => t.name).join(', ')}.` +
          (cap.restricted.length > 0
            ? ` Restricted tools for higher roles: ${cap.restricted.map((t) => `${t.name} (${t.allowedRoles.join('/')})`).join(', ')}.`
            : ''),
      );
      continue;
    }
    if (res.tool !== 'search_documents' && res.tool !== 'search_draft_documents') {
      parts.push(res.summary);
    }
  }
  if (retrieval) {
    parts.push(
      `Retrieved ${retrieval.assembled.chunksUsed.length} grounded chunk(s) using mode '${retrieval.effectiveMode}' (${retrieval.assembled.estimatedTokens.total} context tokens).`,
    );
  }
  return parts.join('\n\n') || 'Completed authorized query.';
}

/**
 * V3.1 RBAC-Enforced Company Agent Orchestrator.
 * Every tool call is bounded (`<= AGENT_MAX_STEPS`) and deterministically validated
 * by `authorizeToolCall` immediately before execution and audit-logged to `audit_logs`.
 */
export class CompanyAgentOrchestrator {
  constructor(private readonly deps: CompanyAgentDeps) {}

  async run(input: CompanyAgentRunInput): Promise<CompanyAgentRunResult> {
    const totalStart = Date.now();
    const role = normalizeAgentRole(input.role);
    const maxSteps = Math.max(1, Math.min(AGENT_MAX_STEPS, input.maxSteps ?? AGENT_MAX_STEPS));

    const planStart = Date.now();
    const plan = planAgentToolsDeterministic(input.userQuery, role, {
      maxSteps,
      orgId: input.orgId,
    });
    const planningMs = Date.now() - planStart;

    // 1. Fail-closed refusal path: zero DB reads, zero retrieval, zero LLM spend
    if (plan.refused) {
      const code = plan.refusalCode ?? 'POLICY_ROLE_DENIED';
      const deniedTool = plan.deniedTool ?? 'refuse_out_of_scope';
      const reason = plan.refusalReason ?? 'Request exceeds caller authority.';

      this.deps.telemetry?.onPolicyDenial?.(role, code);
      this.deps.telemetry?.onToolCall?.(deniedTool, 'denied');
      this.deps.telemetry?.onStepsCompleted?.(0);

      await this.deps.toolsRepo.recordAgentAuditLog({
        orgId: input.orgId,
        userId: input.userId,
        action: 'agent:policy_deny',
        tool: deniedTool,
        role,
        allowed: false,
        code,
        reason,
        requestId: input.requestId,
        latencyMs: Date.now() - totalStart,
        ipAddress: input.ipAddress,
      });

      return {
        decision: 'refuse',
        role,
        refused: true,
        refusalCode: code,
        refusalReason: reason,
        requiredRoles: plan.requiredRoles,
        steps: [
          {
            step: 1,
            tool: deniedTool,
            outcome: 'denied',
            code,
            latencyMs: Date.now() - totalStart,
            summary: reason,
          },
        ],
        results: [],
        structuredContextText: '',
        retrieval: null,
        answer: reason,
        timingsMs: {
          planning: planningMs,
          execution: 0,
          synthesis: 0,
          total: Date.now() - totalStart,
        },
      };
    }

    // 2. Bounded execution of authorized tools (re-verifying PolicyGuard before every invocation)
    const execStart = Date.now();
    const steps: AgentStepRecord[] = [];
    const results: AgentToolResult[] = [];
    let retrievalResult: LightRagRetrievalResult | null = null;

    for (let i = 0; i < Math.min(plan.calls.length, maxSteps); i++) {
      const call = plan.calls[i]!;
      const stepT0 = Date.now();
      const gate = authorizeToolCall({
        role,
        orgId: input.orgId,
        tool: call.tool,
        params: call.params,
      });

      if (!gate.allowed) {
        const code = gate.code ?? 'POLICY_ROLE_DENIED';
        this.deps.telemetry?.onPolicyDenial?.(role, code);
        this.deps.telemetry?.onToolCall?.(call.tool, 'denied');
        await this.deps.toolsRepo.recordAgentAuditLog({
          orgId: input.orgId,
          userId: input.userId,
          action: 'agent:policy_deny',
          tool: call.tool,
          role,
          allowed: false,
          code,
          reason: gate.reason,
          requestId: input.requestId,
          latencyMs: Date.now() - stepT0,
          ipAddress: input.ipAddress,
        });
        return {
          decision: 'refuse',
          role,
          refused: true,
          refusalCode: code,
          refusalReason: gate.reason,
          requiredRoles: gate.requiredRoles,
          steps: [
            ...steps,
            {
              step: i + 1,
              tool: call.tool,
              outcome: 'denied',
              code,
              latencyMs: Date.now() - stepT0,
              summary: gate.reason,
            },
          ],
          results,
          structuredContextText: '',
          retrieval: null,
          answer: gate.reason,
          timingsMs: {
            planning: planningMs,
            execution: Date.now() - execStart,
            synthesis: 0,
            total: Date.now() - totalStart,
          },
        };
      }

      let toolData: unknown = null;
      let summary = '';

      if (call.tool === 'search_documents' || call.tool === 'search_draft_documents') {
        if (!this.deps.retriever) {
          throw new Error(`Retriever dependency not configured for tool '${call.tool}'`);
        }
        const includeDrafts =
          call.tool === 'search_draft_documents' && (role === 'owner' || role === 'admin');
        retrievalResult = await this.deps.retriever.retrieve({
          orgId: input.orgId,
          userQuery: input.userQuery,
          history: input.history,
          mode: input.mode,
          documentIds: input.documentIds,
          includeDrafts,
          rerankLimit: input.rerankLimit,
          rerankMinScore: input.rerankMinScore,
          retrievalConfig: input.retrievalConfig,
          budget: input.budget,
        });
        toolData = {
          effectiveMode: retrievalResult.effectiveMode,
          intent: retrievalResult.intent,
          chunksCount: retrievalResult.assembled.chunksUsed.length,
          entitiesCount: retrievalResult.assembled.entitiesUsed.length,
          relationsCount: retrievalResult.assembled.relationsUsed.length,
          estimatedTokens: retrievalResult.assembled.estimatedTokens.total,
        };
        summary = `Retrieved ${retrievalResult.assembled.chunksUsed.length} chunks (mode=${retrievalResult.effectiveMode}, drafts=${includeDrafts})`;
      } else if (call.tool === 'query_document_inventory') {
        const allowDrafts = role === 'owner' || role === 'admin';
        toolData = await this.deps.toolsRepo.queryDocumentInventory(input.orgId, {
          includeDrafts: allowDrafts && call.params.includeDrafts !== false,
          expiringWithinDays:
            typeof call.params.expiringWithinDays === 'number'
              ? call.params.expiringWithinDays
              : undefined,
          limit: typeof call.params.limit === 'number' ? call.params.limit : 25,
        });
        summary = summarizeToolData(call.tool, toolData);
      } else if (call.tool === 'query_governance_audit') {
        toolData = await this.deps.toolsRepo.queryGovernanceAudit(input.orgId, {
          limit: typeof call.params.limit === 'number' ? call.params.limit : 25,
        });
        summary = summarizeToolData(call.tool, toolData);
      } else if (call.tool === 'query_workspace_members') {
        toolData = await this.deps.toolsRepo.queryWorkspaceMembers(input.orgId, {
          limit: typeof call.params.limit === 'number' ? call.params.limit : 50,
        });
        summary = summarizeToolData(call.tool, toolData);
      } else if (call.tool === 'query_ingestion_jobs') {
        toolData = await this.deps.toolsRepo.queryIngestionJobs(input.orgId, {
          status: typeof call.params.status === 'string' ? call.params.status : undefined,
          limit: typeof call.params.limit === 'number' ? call.params.limit : 20,
        });
        summary = summarizeToolData(call.tool, toolData);
      } else if (call.tool === 'query_security_audit_logs') {
        toolData = await this.deps.toolsRepo.querySecurityAuditLogs(input.orgId, {
          limit: typeof call.params.limit === 'number' ? call.params.limit : 20,
        });
        summary = summarizeToolData(call.tool, toolData);
      } else if (call.tool === 'direct_answer') {
        toolData = listToolsForRole(role);
        summary = summarizeToolData(call.tool, toolData);
      }

      const stepLatencyMs = Date.now() - stepT0;
      this.deps.telemetry?.onToolCall?.(call.tool, 'allowed');
      await this.deps.toolsRepo.recordAgentAuditLog({
        orgId: input.orgId,
        userId: input.userId,
        action: 'agent:tool_allow',
        tool: call.tool,
        role,
        allowed: true,
        reason: summary,
        requestId: input.requestId,
        latencyMs: stepLatencyMs,
        ipAddress: input.ipAddress,
      });

      steps.push({
        step: i + 1,
        tool: call.tool,
        outcome: 'allowed',
        latencyMs: stepLatencyMs,
        summary,
      });
      results.push({
        tool: call.tool,
        summary,
        data: toolData,
      });
    }

    const executionMs = Date.now() - execStart;
    this.deps.telemetry?.onStepsCompleted?.(steps.length);

    const structuredContextText = formatStructuredToolContext(results);
    const synthStart = Date.now();
    let answer = renderDeterministicAnswer(role, results, retrievalResult);

    if (input.synthesizeAnswer && this.deps.llm && plan.decision === 'execute') {
      const contextSections = [
        structuredContextText,
        retrievalResult?.assembled.contextText
          ? `DOCUMENT KNOWLEDGE CONTEXT:\n${retrievalResult.assembled.contextText}`
          : '',
      ]
        .filter(Boolean)
        .join('\n\n');

      const sysPrompt =
        `You are an internal enterprise RBAC company assistant for role '${role}'. ` +
        `Answer accurately using ONLY the verified database tool outputs and/or cited document chunks below. ` +
        `Preserve exact numbers and cite document chunks as [Citation N] when referencing documents.\n\n${contextSections}`;

      const generated = await this.deps.llm.generate([
        { role: 'system', content: sysPrompt },
        { role: 'user', content: input.userQuery },
      ]);
      if (generated.content.trim()) {
        answer = generated.content.trim();
      }
    }
    const synthesisMs = Date.now() - synthStart;

    return {
      decision: plan.decision,
      role,
      refused: false,
      steps,
      results,
      structuredContextText,
      retrieval: retrievalResult,
      answer,
      timingsMs: {
        planning: planningMs,
        execution: executionMs,
        synthesis: synthesisMs,
        total: Date.now() - totalStart,
      },
    };
  }
}
