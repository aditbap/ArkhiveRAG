## Summary

Describe the change, motivation, and any architectural or migration impact.

## Type of Change

- [ ] Bug fix
- [ ] Feature / Retrieval enhancement
- [ ] Database performance / Migration (forward-only, `IF NOT EXISTS`)
- [ ] Documentation / Operations / Observability

## Verification Checklist

- [ ] `pnpm typecheck && pnpm lint && pnpm test:unit && pnpm build` pass locally
- [ ] If touching queries/indexes/Docker: `pnpm ops:verify-plans` & `pnpm ops:verify-resilience` pass
- [ ] No secrets, `.env` values, or machine-specific paths committed
- [ ] Documentation / `docs/CHANGELOG.md` updated if public behavior or config changed
