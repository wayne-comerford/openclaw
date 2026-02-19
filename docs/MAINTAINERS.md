# Maintainers and Governance

This document defines repository ownership for `wayne-comerford/openclaw`.

## Ownership

- Repository owner and primary maintainer: `@wayne-comerford`
- Default code owner: `@wayne-comerford` (see `.github/CODEOWNERS`)

## Scope of Maintainer Responsibility

- Keep `main` releasable.
- Review and merge pull requests.
- Triage issues and security reports.
- Maintain CI/CD and release workflows.
- Keep dependency and security updates current.

## Decision Policy

- Single-maintainer model: the repository owner is the final decision maker.
- For high-risk changes (security, auth, release, data migration), require a dedicated review pass before merge.
- Prefer small PRs with rollback paths.

## Branch and Merge Policy

- Protected branch: `main`
- Required status checks must pass before merge.
- Require at least one approval (self-approval should only be used for urgent hotfixes).
- Squash merge preferred for feature work.
- Revert-first policy for production regressions.

## Upstream Sync Policy

This repository is a fork of `openclaw/openclaw`.

- Add `upstream` remote and fetch regularly.
- Sync cadence: at least every 2-3 days, daily during active release periods.
- Default strategy: merge `upstream/main` into `main`.
- If conflicts are risky, create a sync PR branch and test before merging.

## Release Authority

- Only the maintainer may cut stable tags/releases.
- Release gates:
- CI green on required checks.
- No unresolved critical incidents.
- Release notes include user-impacting changes and rollback notes.

## Security Ownership

- Security contact remains `security@openclaw.ai` for upstream project vulnerabilities.
- Fork-specific secrets, tokens, and environments are owned by `@wayne-comerford`.
- Critical security fixes may bypass normal cadence but still require post-incident documentation.

## Incident Ownership

- Incident commander: `@wayne-comerford`
- Severity policy:
- Sev 1: active outage/security event, immediate mitigation.
- Sev 2: major degradation, same-day mitigation.
- Sev 3: non-critical defect, next planned cycle.

## Maintenance Cadence

- Daily: check CI failures and urgent issues.
- Weekly: dependency/security updates, stale triage.
- Per release: run release checklist from `docs/RUNBOOK.md`.
