import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { hashApiKey, parseApiKeyPrefix } from '@rag/database';
import { ROLE_RANK, capRole } from '@rag/auth';
import { handleMcpJsonRpc, listMcpToolsForRole, MCP_PROTOCOL_VERSION } from '../../src/lib/mcp-handler.js';

describe('V3.2 Phase 2 — Scoped API Key Helpers & Privilege Capping', () => {
  it('validates and extracts prefix strictly from well-formed ak_live_ credentials', () => {
    const validKey = `ak_live_001122334455_${'ab'.repeat(24)}`;
    assert.equal(parseApiKeyPrefix(validKey), 'ak_live_001122334455');
    assert.equal(parseApiKeyPrefix(`  ${validKey}  `), 'ak_live_001122334455');

    // Malformed / wrong prefix / short secret rejected
    assert.equal(parseApiKeyPrefix('sk_live_001122334455_' + 'ab'.repeat(24)), null);
    assert.equal(parseApiKeyPrefix('ak_live_001122334455_short'), null);
    assert.equal(parseApiKeyPrefix('ak_live_ZZZZ22334455_' + 'ab'.repeat(24)), null);
  });

  it('produces deterministic 64-char SHA-256 hex hashes', () => {
    const key = `ak_live_deadbeef1234_${'cd'.repeat(24)}`;
    const digest = hashApiKey(key);
    assert.equal(digest.length, 64);
    assert.match(digest, /^[0-9a-f]{64}$/);
    assert.equal(digest, hashApiKey(key));
    assert.notEqual(digest, hashApiKey(`ak_live_deadbeef1234_${'ef'.repeat(24)}`));
  });

  it('caps effective role to the lower of API key role and live membership role', () => {
    assert.ok(ROLE_RANK.owner > ROLE_RANK.admin);
    assert.ok(ROLE_RANK.admin > ROLE_RANK.member);
    assert.ok(ROLE_RANK.member > ROLE_RANK.viewer);

    assert.equal(capRole('admin', 'viewer'), 'viewer');
    assert.equal(capRole('viewer', 'owner'), 'viewer');
    assert.equal(capRole('member', 'admin'), 'member');
    assert.equal(capRole('admin', 'admin'), 'admin');
  });
});

describe('V3.2 Phase 3 — Native RBAC-Enforced MCP Server (/mcp)', () => {
  const makeContext = (role: 'owner' | 'admin' | 'member' | 'viewer') => {
    const audits: Array<{ action: string; tool: string; allowed: boolean; reason?: string }> = [];
    const ctx = {
      orgId: '00000000-0000-4000-8000-000000000001',
      userId: '00000000-0000-4000-8000-000000000002',
      role,
      requestId: 'req-mcp-test',
      searchKnowledgeBase: async ({ query, mode }: { query: string; mode: string }) => ({
        mode,
        query,
        chunks: [{ index: 1, content: 'Sample excerpt' }],
      }),
      askCompanyAgent: async ({ query }: { query: string }) => {
        if (query.toLowerCase().includes('delete from')) {
          return {
            decision: 'refuse',
            refused: true,
            refusalCode: 'OUT_OF_AUTHORITY_WRITE',
            refusalReason: 'Read-only policy blocked mutation attempt',
          };
        }
        return { decision: 'execute', refused: false, answer: `Answered: ${query}` };
      },
      listDocuments: async () => ({ totalDocuments: 2, items: [] }),
      getGovernanceAudit: async () => ({ byGovernanceStatus: { active: 2 } }),
      getSecurityAuditLogs: async () => [{ action: 'agent:policy_deny' }],
      recordAudit: async (entry: { action: string; tool: string; allowed: boolean; reason?: string }) => {
        audits.push(entry);
      },
    };
    return { ctx, audits };
  };

  it('handles JSON-RPC initialize, ping, notifications/initialized, and invalid payloads', async () => {
    const { ctx } = makeContext('member');
    const bad = await handleMcpJsonRpc({ method: 'ping' }, ctx);
    assert.equal(bad.status, 400);

    const init = await handleMcpJsonRpc({ jsonrpc: '2.0', id: 1, method: 'initialize' }, ctx);
    assert.equal(init.status, 200);
    const initPayload = init.payload as { result: { protocolVersion: string; serverInfo: { name: string } } };
    assert.equal(initPayload.result.protocolVersion, MCP_PROTOCOL_VERSION);
    assert.equal(initPayload.result.serverInfo.name, 'arkhiverag-mcp');

    const notif = await handleMcpJsonRpc({ jsonrpc: '2.0', method: 'notifications/initialized' }, ctx);
    assert.equal(notif.status, 202);
    assert.equal(notif.payload, null);

    const ping = await handleMcpJsonRpc({ jsonrpc: '2.0', id: 2, method: 'ping' }, ctx);
    assert.equal(ping.status, 200);
  });

  it('filters exposed MCP tools dynamically by caller RBAC role', async () => {
    assert.deepEqual(
      listMcpToolsForRole('viewer').map((t) => t.name),
      ['search_knowledge_base', 'ask_company_agent', 'list_documents'],
    );
    assert.deepEqual(
      listMcpToolsForRole('admin').map((t) => t.name),
      ['search_knowledge_base', 'ask_company_agent', 'list_documents', 'get_governance_audit'],
    );
    assert.deepEqual(
      listMcpToolsForRole('owner').map((t) => t.name),
      [
        'search_knowledge_base',
        'ask_company_agent',
        'list_documents',
        'get_governance_audit',
        'get_security_audit_logs',
      ],
    );

    const { ctx } = makeContext('member');
    const listRes = await handleMcpJsonRpc({ jsonrpc: '2.0', id: 3, method: 'tools/list' }, ctx);
    const payload = listRes.payload as { result: { tools: Array<{ name: string }> } };
    assert.equal(payload.result.tools.some((t) => t.name === 'get_governance_audit'), false);
    assert.equal(payload.result.tools.some((t) => t.name === 'get_security_audit_logs'), false);
  });

  it('enforces fail-closed RBAC denial on direct unauthorized tools/call and audits allow/deny', async () => {
    const member = makeContext('member');
    const denied = await handleMcpJsonRpc(
      {
        jsonrpc: '2.0',
        id: 10,
        method: 'tools/call',
        params: { name: 'get_security_audit_logs', arguments: { limit: 5 } },
      },
      member.ctx,
    );
    const deniedResult = (denied.payload as { result: { isError: boolean; content: Array<{ text: string }> } }).result;
    assert.equal(deniedResult.isError, true);
    assert.match(deniedResult.content[0]?.text ?? '', /Permission denied/);
    assert.equal(member.audits[0]?.action, 'mcp:policy_deny');
    assert.equal(member.audits[0]?.allowed, false);

    // Owner succeeds on security audit logs; agent SQL write probe propagates isError: true + audit deny
    const owner = makeContext('owner');
    const allowed = await handleMcpJsonRpc(
      {
        jsonrpc: '2.0',
        id: 11,
        method: 'tools/call',
        params: { name: 'get_security_audit_logs', arguments: { limit: 10 } },
      },
      owner.ctx,
    );
    assert.equal((allowed.payload as { result: { isError: boolean } }).result.isError, false);
    assert.equal(owner.audits[0]?.action, 'mcp:tool_allow');

    const writeProbe = await handleMcpJsonRpc(
      {
        jsonrpc: '2.0',
        id: 12,
        method: 'tools/call',
        params: { name: 'ask_company_agent', arguments: { query: 'DELETE FROM users;' } },
      },
      owner.ctx,
    );
    assert.equal((writeProbe.payload as { result: { isError: boolean } }).result.isError, true);
    assert.equal(owner.audits[1]?.action, 'mcp:policy_deny');
  });
});
