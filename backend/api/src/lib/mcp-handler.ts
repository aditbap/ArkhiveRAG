import {
  isRagQueryMode,
  listToolsForRole,
  type AgentRole,
  type RagQueryMode,
} from '@rag/core';

export const MCP_PROTOCOL_VERSION = '2025-03-26';

export interface McpToolDefinition {
  name: string;
  description: string;
  minRole: AgentRole;
  inputSchema: Record<string, unknown>;
}

const ROLE_ORDER: Record<AgentRole, number> = {
  viewer: 0,
  member: 1,
  admin: 2,
  owner: 3,
};

export const MCP_TOOLS: readonly McpToolDefinition[] = [
  {
    name: 'search_knowledge_base',
    description:
      'Search active workspace documents using ArkhiveRAG hybrid pgvector + BM25 + Knowledge Graph retrieval with inline citations.',
    minRole: 'viewer',
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Natural-language search question.' },
        mode: {
          type: 'string',
          enum: ['auto', 'mix', 'hybrid', 'local', 'global', 'naive'],
          description: 'Retrieval strategy (defaults to auto).',
        },
        documentIds: {
          type: 'array',
          items: { type: 'string' },
          description: 'Optional UUID list scoping search to specific documents.',
        },
      },
      required: ['query'],
    },
  },
  {
    name: 'ask_company_agent',
    description:
      'Run the V3.1 RBAC-enforced Company Agent across document retrieval and allowlisted read-only PostgreSQL tools with fail-closed policy enforcement.',
    minRole: 'viewer',
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Question or structured workspace inquiry.' },
        synthesize: { type: 'boolean', description: 'Whether to synthesize a natural-language answer (default true).' },
        documentIds: { type: 'array', items: { type: 'string' } },
      },
      required: ['query'],
    },
  },
  {
    name: 'list_documents',
    description: 'List workspace documents, statuses, governance tiers, and storage totals.',
    minRole: 'viewer',
    inputSchema: {
      type: 'object',
      properties: {
        limit: { type: 'integer', minimum: 1, maximum: 50 },
      },
    },
  },
  {
    name: 'get_governance_audit',
    description: 'Admin/owner only: inspect draft, superseding, and governance lifecycle transitions.',
    minRole: 'admin',
    inputSchema: {
      type: 'object',
      properties: {
        limit: { type: 'integer', minimum: 1, maximum: 50 },
      },
    },
  },
  {
    name: 'get_security_audit_logs',
    description: 'Owner only: inspect agent and security allow/deny audit trail entries.',
    minRole: 'owner',
    inputSchema: {
      type: 'object',
      properties: {
        limit: { type: 'integer', minimum: 1, maximum: 50 },
      },
    },
  },
];

export function listMcpToolsForRole(role: AgentRole): McpToolDefinition[] {
  const rank = ROLE_ORDER[role];
  return MCP_TOOLS.filter((tool) => ROLE_ORDER[tool.minRole] <= rank);
}

export interface JsonRpcRequest {
  jsonrpc?: unknown;
  id?: string | number | null;
  method?: unknown;
  params?: unknown;
}

export interface McpExecutionContext {
  orgId: string;
  userId: string;
  role: AgentRole;
  requestId: string;
  ipAddress?: string;
  searchKnowledgeBase: (args: {
    query: string;
    mode: RagQueryMode;
    documentIds?: string[];
  }) => Promise<unknown>;
  askCompanyAgent: (args: {
    query: string;
    synthesize: boolean;
    documentIds?: string[];
  }) => Promise<{ refused?: boolean; refusalReason?: string; [k: string]: unknown }>;
  listDocuments: (args: { limit?: number }) => Promise<unknown>;
  getGovernanceAudit: (args: { limit?: number }) => Promise<unknown>;
  getSecurityAuditLogs: (args: { limit?: number }) => Promise<unknown>;
  recordAudit?: (entry: {
    action: string;
    tool: string;
    allowed: boolean;
    reason?: string;
  }) => Promise<void>;
}

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

function rpcResult(id: string | number | null, result: unknown) {
  return { jsonrpc: '2.0', id, result };
}

function rpcError(id: string | number | null, code: number, message: string) {
  return { jsonrpc: '2.0', id, error: { code, message } };
}

function validateUuidArray(raw: unknown): string[] | undefined | null {
  if (raw === undefined) return undefined;
  if (!Array.isArray(raw) || !raw.every((x): x is string => typeof x === 'string' && UUID_RE.test(x))) {
    return null;
  }
  return raw;
}

/** Pure MCP Streamable HTTP JSON-RPC 2.0 dispatcher with strict per-tool RBAC gating. */
export async function handleMcpJsonRpc(
  body: JsonRpcRequest,
  ctx: McpExecutionContext,
): Promise<{ status: number; payload: unknown }> {
  if (body.jsonrpc !== '2.0' || typeof body.method !== 'string') {
    return { status: 400, payload: rpcError(body.id ?? null, -32600, 'Invalid JSON-RPC 2.0 request') };
  }

  const id = body.id ?? null;

  switch (body.method) {
    case 'initialize':
      return {
        status: 200,
        payload: rpcResult(id, {
          protocolVersion: MCP_PROTOCOL_VERSION,
          capabilities: { tools: {} },
          serverInfo: { name: 'arkhiverag-mcp', version: '3.2.0' },
          instructions: `Authenticated for org ${ctx.orgId} with role '${ctx.role}'.`,
        }),
      };

    case 'notifications/initialized':
      return { status: 202, payload: null };

    case 'ping':
      return { status: 200, payload: rpcResult(id, {}) };

    case 'tools/list': {
      const allowedTools = listMcpToolsForRole(ctx.role).map(({ minRole: _minRole, ...rest }) => rest);
      return {
        status: 200,
        payload: rpcResult(id, {
          tools: allowedTools,
          _meta: { role: ctx.role, agentTools: listToolsForRole(ctx.role) },
        }),
      };
    }

    case 'tools/call': {
      const params = (body.params ?? {}) as { name?: unknown; arguments?: unknown };
      if (typeof params.name !== 'string' || !params.name.trim()) {
        return { status: 200, payload: rpcError(id, -32602, 'Tool name is required') };
      }

      const toolDef = MCP_TOOLS.find((t) => t.name === params.name);
      if (!toolDef) {
        return { status: 200, payload: rpcError(id, -32601, `Unknown tool: ${params.name}`) };
      }

      if (ROLE_ORDER[ctx.role] < ROLE_ORDER[toolDef.minRole]) {
        await ctx.recordAudit?.({
          action: 'mcp:policy_deny',
          tool: toolDef.name,
          allowed: false,
          reason: `Requires role '${toolDef.minRole}', caller has '${ctx.role}'`,
        });
        return {
          status: 200,
          payload: rpcResult(id, {
            isError: true,
            content: [
              {
                type: 'text',
                text: `Permission denied: '${toolDef.name}' requires role '${toolDef.minRole}', caller holds '${ctx.role}'.`,
              },
            ],
          }),
        };
      }

      const args = (params.arguments && typeof params.arguments === 'object'
        ? params.arguments
        : {}) as Record<string, unknown>;

      if (toolDef.name === 'search_knowledge_base') {
        const query = typeof args.query === 'string' ? args.query.trim() : '';
        if (!query || query.length > 4000) {
          return { status: 200, payload: rpcError(id, -32602, 'query must be a non-empty string (<= 4000 chars)') };
        }
        const mode: RagQueryMode =
          args.mode === undefined ? 'auto' : isRagQueryMode(args.mode) ? args.mode : 'auto';
        const docIds = validateUuidArray(args.documentIds);
        if (docIds === null) {
          return { status: 200, payload: rpcError(id, -32602, 'documentIds must be an array of UUIDs') };
        }
        const data = await ctx.searchKnowledgeBase({ query, mode, documentIds: docIds });
        await ctx.recordAudit?.({ action: 'mcp:tool_allow', tool: toolDef.name, allowed: true });
        return {
          status: 200,
          payload: rpcResult(id, { content: [{ type: 'text', text: JSON.stringify(data, null, 2) }], isError: false }),
        };
      }

      if (toolDef.name === 'ask_company_agent') {
        const query = typeof args.query === 'string' ? args.query.trim() : '';
        if (!query || query.length > 4000) {
          return { status: 200, payload: rpcError(id, -32602, 'query must be a non-empty string (<= 4000 chars)') };
        }
        const docIds = validateUuidArray(args.documentIds);
        if (docIds === null) {
          return { status: 200, payload: rpcError(id, -32602, 'documentIds must be an array of UUIDs') };
        }
        const data = await ctx.askCompanyAgent({
          query,
          synthesize: args.synthesize !== false,
          documentIds: docIds,
        });
        await ctx.recordAudit?.({
          action: data.refused ? 'mcp:policy_deny' : 'mcp:tool_allow',
          tool: toolDef.name,
          allowed: !data.refused,
          reason: data.refusalReason,
        });
        return {
          status: 200,
          payload: rpcResult(id, {
            content: [{ type: 'text', text: JSON.stringify(data, null, 2) }],
            isError: Boolean(data.refused),
          }),
        };
      }

      const rawLimit =
        typeof args.limit === 'number' && Number.isFinite(args.limit)
          ? Math.min(50, Math.max(1, Math.trunc(args.limit)))
          : undefined;

      if (toolDef.name === 'list_documents') {
        const data = await ctx.listDocuments({ limit: rawLimit });
        await ctx.recordAudit?.({ action: 'mcp:tool_allow', tool: toolDef.name, allowed: true });
        return {
          status: 200,
          payload: rpcResult(id, { content: [{ type: 'text', text: JSON.stringify(data, null, 2) }], isError: false }),
        };
      }

      if (toolDef.name === 'get_governance_audit') {
        const data = await ctx.getGovernanceAudit({ limit: rawLimit });
        await ctx.recordAudit?.({ action: 'mcp:tool_allow', tool: toolDef.name, allowed: true });
        return {
          status: 200,
          payload: rpcResult(id, { content: [{ type: 'text', text: JSON.stringify(data, null, 2) }], isError: false }),
        };
      }

      if (toolDef.name === 'get_security_audit_logs') {
        const data = await ctx.getSecurityAuditLogs({ limit: rawLimit });
        await ctx.recordAudit?.({ action: 'mcp:tool_allow', tool: toolDef.name, allowed: true });
        return {
          status: 200,
          payload: rpcResult(id, { content: [{ type: 'text', text: JSON.stringify(data, null, 2) }], isError: false }),
        };
      }

      return { status: 200, payload: rpcError(id, -32601, `Unhandled tool: ${toolDef.name}`) };
    }

    default:
      return { status: 200, payload: rpcError(id, -32601, `Method not found: ${body.method}`) };
  }
}
