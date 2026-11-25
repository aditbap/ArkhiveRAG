import { FastifyPluginAsync } from 'fastify';
import { ROLE_RANK, requirePermissionMiddleware, type Role } from '@rag/auth';
import { ApiKeyRepository, auditLogs, db, type ApiKeyRole } from '@rag/database';

const apiKeyRepo = new ApiKeyRepository();
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const VALID_KEY_ROLES: readonly ApiKeyRole[] = ['viewer', 'member', 'admin'];

function isValidKeyRole(value: unknown): value is ApiKeyRole {
  return typeof value === 'string' && VALID_KEY_ROLES.includes(value as ApiKeyRole);
}

export const apiKeyRoutes: FastifyPluginAsync = async (fastify) => {
  fastify.get(
    '/',
    { preHandler: requirePermissionMiddleware('settings:read') },
    async (request, reply) => {
      const orgId = request.user?.orgId || request.user?.id;
      if (!orgId) {
        return reply.status(403).send({ error: { code: 'FORBIDDEN', message: 'Organization required', requestId: request.id } });
      }
      const rows = await apiKeyRepo.listByOrg(orgId);
      return { data: rows };
    },
  );

  fastify.post(
    '/',
    { preHandler: requirePermissionMiddleware('members:manage') },
    async (request, reply) => {
      const orgId = request.user?.orgId || request.user?.id;
      const userId = request.user?.id;
      const callerRole: Role = request.userRole ?? 'viewer';
      if (!orgId || !userId) {
        return reply.status(403).send({ error: { code: 'FORBIDDEN', message: 'Organization required', requestId: request.id } });
      }
      if (request.apiKey) {
        return reply.status(403).send({
          error: {
            code: 'FORBIDDEN',
            message: 'API keys cannot mint other API keys; authenticate with a user session instead.',
            requestId: request.id,
          },
        });
      }

      const body = (request.body ?? {}) as {
        name?: unknown;
        role?: unknown;
        expiresInDays?: unknown;
      };

      const name = typeof body.name === 'string' ? body.name.trim() : '';
      if (!name || name.length > 120) {
        return reply.status(400).send({
          error: { code: 'BAD_REQUEST', message: 'name must be 1..120 characters', requestId: request.id },
        });
      }

      const requestedRole: ApiKeyRole = body.role === undefined ? 'member' : isValidKeyRole(body.role) ? body.role : ('invalid' as ApiKeyRole);
      if (!isValidKeyRole(requestedRole)) {
        return reply.status(400).send({
          error: { code: 'BAD_REQUEST', message: 'role must be one of: viewer, member, admin', requestId: request.id },
        });
      }

      if (ROLE_RANK[requestedRole] > ROLE_RANK[callerRole]) {
        return reply.status(403).send({
          error: { code: 'FORBIDDEN', message: 'Cannot create an API key with higher privileges than your own role', requestId: request.id },
        });
      }

      let expiresAt: Date | null = null;
      if (body.expiresInDays !== undefined && body.expiresInDays !== null) {
        if (typeof body.expiresInDays !== 'number' || !Number.isFinite(body.expiresInDays) || body.expiresInDays < 1 || body.expiresInDays > 365) {
          return reply.status(400).send({
            error: { code: 'BAD_REQUEST', message: 'expiresInDays must be between 1 and 365', requestId: request.id },
          });
        }
        expiresAt = new Date(Date.now() + body.expiresInDays * 24 * 60 * 60 * 1000);
      }

      const created = await apiKeyRepo.createKey({
        orgId,
        createdBy: userId,
        name,
        role: requestedRole,
        expiresAt,
      });

      await db.insert(auditLogs).values({
        orgId,
        userId,
        action: 'api_key:create',
        resourceType: 'api_key',
        resourceId: created.record.id,
        details: { name, role: requestedRole, keyPrefix: created.record.keyPrefix, expiresAt: expiresAt?.toISOString() ?? null },
        ipAddress: request.ip,
      });

      return reply.status(201).send({
        data: {
          ...created.record,
          key: created.key,
        },
      });
    },
  );

  fastify.delete(
    '/:id',
    { preHandler: requirePermissionMiddleware('members:manage') },
    async (request, reply) => {
      const orgId = request.user?.orgId || request.user?.id;
      const userId = request.user?.id;
      const { id } = request.params as { id?: string };
      if (!orgId || !userId) {
        return reply.status(403).send({ error: { code: 'FORBIDDEN', message: 'Organization required', requestId: request.id } });
      }
      if (!id || !UUID_RE.test(id)) {
        return reply.status(400).send({ error: { code: 'BAD_REQUEST', message: 'Invalid API key id', requestId: request.id } });
      }

      const revoked = await apiKeyRepo.revokeKey(orgId, id);
      if (!revoked) {
        return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Active API key not found', requestId: request.id } });
      }

      await db.insert(auditLogs).values({
        orgId,
        userId,
        action: 'api_key:revoke',
        resourceType: 'api_key',
        resourceId: id,
        details: { revokedByApiKey: Boolean(request.apiKey) },
        ipAddress: request.ip,
      });

      return { data: { revoked: true, id } };
    },
  );
};
