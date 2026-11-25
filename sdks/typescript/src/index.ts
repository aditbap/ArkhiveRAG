export type RagMode = 'auto' | 'mix' | 'hybrid' | 'local' | 'global' | 'naive';
export type ApiKeyRole = 'admin' | 'member' | 'viewer';
export type GovernanceStatus = 'draft' | 'approved' | 'active' | 'superseded' | 'archived';

export interface ArkhiveRAGClientOptions {
  /** Base URL of the API server (e.g. `http://localhost:3002` or `https://api.example.com`). */
  baseUrl: string;
  /** Org-scoped API key (`ak_live_...`), sent via `X-Api-Key`. */
  apiKey?: string;
  /** Keycloak OIDC JWT access token, sent via `Authorization: Bearer <token>`. */
  token?: string;
  /** Optional shared library / organization UUID (`X-Org-Id` when using a Keycloak JWT). */
  orgId?: string;
  /** Optional custom `fetch` implementation (useful in tests or edge runtimes). */
  fetch?: typeof globalThis.fetch;
}

export class ArkhiveRAGError extends Error {
  readonly status: number;
  readonly code: string;
  readonly requestId?: string;

  constructor(status: number, code: string, message: string, requestId?: string) {
    super(message);
    this.name = 'ArkhiveRAGError';
    this.status = status;
    this.code = code;
    this.requestId = requestId;
  }
}

export interface ChatMessageInput {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface ChatRequest {
  messages: ChatMessageInput[];
  mode?: RagMode;
  agentic?: boolean;
  conversationId?: string;
  documentIds?: string[];
  includeDrafts?: boolean;
}

export interface ChatStreamEvent {
  type?: string;
  content?: string;
  conversationId?: string;
  citations?: Array<Record<string, unknown>>;
  [key: string]: unknown;
}

export interface ChatCompletionResponse {
  id?: string;
  conversationId?: string;
  choices?: Array<{
    index?: number;
    message?: { role?: string; content?: string };
  }>;
  citations?: Array<Record<string, unknown>>;
  agent?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface AgentQueryRequest {
  query: string;
  documentIds?: string[];
  synthesize?: boolean;
}

export interface AgentQueryResponse {
  data: {
    decision: 'execute' | 'direct' | 'refuse';
    role: string;
    refused: boolean;
    refusalCode?: string;
    refusalReason?: string;
    requiredRoles?: string[];
    answer: string;
    steps: Array<{ step: number; tool: string; outcome: string; summary: string }>;
    structuredData: Array<{ tool: string; summary: string; data: unknown }>;
    citations: Array<Record<string, unknown>>;
    timingsMs: Record<string, number>;
  };
}

export interface ApiKeyMetadata {
  id: string;
  orgId: string;
  createdBy: string;
  name: string;
  keyPrefix: string;
  role: ApiKeyRole;
  expiresAt: string | null;
  revokedAt: string | null;
  lastUsedAt: string | null;
  createdAt: string;
}

export class ArkhiveRAGClient {
  private readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly token?: string;
  private readonly orgId?: string;
  private readonly fetchImpl: typeof globalThis.fetch;

  constructor(options: ArkhiveRAGClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, '');
    this.apiKey = options.apiKey;
    this.token = options.token;
    this.orgId = options.orgId;
    this.fetchImpl = options.fetch ?? globalThis.fetch.bind(globalThis);
  }

  private buildHeaders(extra?: Record<string, string>, includeJsonContentType = true): Record<string, string> {
    const headers: Record<string, string> = {};
    if (includeJsonContentType) headers['Content-Type'] = 'application/json';
    if (this.apiKey) {
      headers['X-Api-Key'] = this.apiKey;
    } else if (this.token) {
      headers.Authorization = `Bearer ${this.token}`;
    }
    if (this.orgId) {
      headers['X-Org-Id'] = this.orgId;
    }
    return { ...headers, ...extra };
  }

  private async request<T>(path: string, init: RequestInit = {}, includeJsonContentType = true): Promise<T> {
    const res = await this.fetchImpl(`${this.baseUrl}${path}`, {
      ...init,
      headers: this.buildHeaders(init.headers as Record<string, string> | undefined, includeJsonContentType),
    });

    if (res.status === 204) return undefined as T;
    const body = (await res.json().catch(() => null)) as {
      error?: { code?: string; message?: string; requestId?: string };
    } | null;

    if (!res.ok) {
      throw new ArkhiveRAGError(
        res.status,
        body?.error?.code ?? 'HTTP_ERROR',
        body?.error?.message ?? `Request failed with status ${res.status}`,
        body?.error?.requestId,
      );
    }
    return body as T;
  }

  health(): Promise<{ status: string; timestamp: string }> {
    return this.request('/health', { method: 'GET' }, false);
  }

  ready(): Promise<{ status: string; checks: Record<string, string> }> {
    return this.request('/ready', { method: 'GET' }, false);
  }

  chat(params: ChatRequest): Promise<ChatCompletionResponse> {
    return this.request('/api/v1/chat/completions', {
      method: 'POST',
      body: JSON.stringify({ ...params, stream: false }),
    });
  }

  async *chatStream(params: ChatRequest): AsyncGenerator<ChatStreamEvent, void, unknown> {
    const res = await this.fetchImpl(`${this.baseUrl}/api/v1/chat/completions`, {
      method: 'POST',
      headers: this.buildHeaders(),
      body: JSON.stringify({ ...params, stream: true }),
    });

    if (!res.ok) {
      const body = (await res.json().catch(() => null)) as {
        error?: { code?: string; message?: string; requestId?: string };
      } | null;
      throw new ArkhiveRAGError(
        res.status,
        body?.error?.code ?? 'HTTP_ERROR',
        body?.error?.message ?? `Stream failed with status ${res.status}`,
        body?.error?.requestId,
      );
    }

    if (!res.body) return;
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6).trim();
        if (!payload || payload === '[DONE]') continue;
        try {
          yield JSON.parse(payload) as ChatStreamEvent;
        } catch {
          // ignore malformed SSE frame
        }
      }
    }
  }

  readonly agent = {
    tools: (): Promise<{ data: Record<string, unknown> }> =>
      this.request('/api/v1/agent/tools', { method: 'GET' }, false),
    query: (params: AgentQueryRequest): Promise<AgentQueryResponse> =>
      this.request('/api/v1/agent/query', {
        method: 'POST',
        body: JSON.stringify(params),
      }),
    audit: (limit = 25): Promise<{ data: Array<Record<string, unknown>> }> =>
      this.request(`/api/v1/agent/audit?limit=${encodeURIComponent(String(limit))}`, { method: 'GET' }, false),
  };

  readonly documents = {
    list: (params?: { limit?: number; offset?: number; folderId?: string }): Promise<{
      data: Array<Record<string, unknown>>;
    }> => {
      const qs = new URLSearchParams();
      if (params?.limit !== undefined) qs.set('limit', String(params.limit));
      if (params?.offset !== undefined) qs.set('offset', String(params.offset));
      if (params?.folderId) qs.set('folderId', params.folderId);
      const suffix = qs.toString() ? `?${qs.toString()}` : '';
      return this.request(`/api/v1/documents${suffix}`, { method: 'GET' }, false);
    },
    upload: async (
      file: Blob,
      fileName: string,
      options?: { folderId?: string; notes?: string },
    ): Promise<{ data: Record<string, unknown> }> => {
      const form = new FormData();
      form.append('file', file, fileName);
      if (options?.folderId) form.append('folderId', options.folderId);
      if (options?.notes) form.append('notes', options.notes);
      return this.request('/api/v1/documents/upload', { method: 'POST', body: form }, false);
    },
    updateGovernance: (
      documentId: string,
      patch: {
        governanceStatus: GovernanceStatus;
        authorityTier?: number;
        effectiveAt?: string;
        expiresAt?: string;
        supersedesDocumentId?: string;
      },
    ): Promise<{ data: Record<string, unknown> }> =>
      this.request(`/api/v1/documents/${encodeURIComponent(documentId)}/governance`, {
        method: 'PATCH',
        body: JSON.stringify(patch),
      }),
    delete: (documentId: string): Promise<void> =>
      this.request(`/api/v1/documents/${encodeURIComponent(documentId)}`, { method: 'DELETE' }, false),
  };

  readonly libraries = {
    list: (): Promise<{ data: Array<Record<string, unknown>>; currentUserId?: string }> =>
      this.request('/api/v1/libraries', { method: 'GET' }, false),
  };

  readonly apiKeys = {
    list: (): Promise<{ data: ApiKeyMetadata[] }> =>
      this.request('/api/v1/api-keys', { method: 'GET' }, false),
    create: (params: {
      name: string;
      role?: ApiKeyRole;
      expiresInDays?: number;
    }): Promise<{ data: ApiKeyMetadata & { key: string } }> =>
      this.request('/api/v1/api-keys', {
        method: 'POST',
        body: JSON.stringify(params),
      }),
    revoke: (id: string): Promise<{ data: { revoked: boolean; id: string } }> =>
      this.request(`/api/v1/api-keys/${encodeURIComponent(id)}`, { method: 'DELETE' }, false),
  };
}
