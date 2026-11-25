---
name: Bug Report
about: Report a reproducible defect or performance regression
title: "[Bug] "
labels: bug
---

## Description
A clear description of the issue and expected vs. actual behavior.

## Reproduction Steps
1. Environment (OS, Docker/Node/Postgres versions, BYOK provider)
2. Steps or request payload
3. Error logs (`X-Request-Id` / SQLSTATE / worker log)

## Diagnostics (if database/performance related)
- Output of `pnpm ops:verify-plans` (if applicable)
- Relevant `GET /ready` output
