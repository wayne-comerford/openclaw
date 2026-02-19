# Maintainer Runbook

Operational runbook for `wayne-comerford/openclaw`.

## 1) Routine Upstream Sync

Goal: keep fork current with `openclaw/openclaw` while preserving fork-specific changes.

```bash
git remote add upstream https://github.com/openclaw/openclaw.git # one-time
git fetch upstream
git checkout main
git pull origin main
git merge upstream/main
```

If conflicts occur:

1. Resolve conflicts in focused commits.
2. Run required checks locally.
3. Push to a `sync/*` branch and open a PR if risk is non-trivial.

## 2) Pre-Merge Checklist

Before merging any PR into `main`:

1. CI required checks are green.
2. Security and auth-sensitive code has an explicit review pass.
3. Changelog/release notes impact is identified.
4. Rollback path is known (revert commit or rollback release artifact).

## 3) Release Checklist

Use this for stable releases.

1. Confirm `main` is green on required CI checks.
2. Verify no open Sev 1 or unresolved Sev 2 incidents.
3. Pull latest `main` locally.
4. Run local validation:

```bash
pnpm install
pnpm build
pnpm check
pnpm test
```

5. Create and push release tag (example):

```bash
git tag vYYYY.M.D
git push origin vYYYY.M.D
```

6. Verify post-tag automation:
- GitHub Actions release workflows complete.
- Docker artifacts (if enabled) publish successfully.

7. Publish release notes:
- Breaking changes.
- Migration notes.
- Rollback instructions.

## 4) Hotfix Process

For urgent production issues:

1. Branch from latest `main` as `hotfix/<short-name>`.
2. Keep the patch minimal and scoped.
3. Run targeted tests for impacted area plus CI.
4. Merge via PR with expedited review.
5. Tag patch release (`vYYYY.M.D-<n>` style).
6. Announce impact, fix, and verification steps.

## 5) Incident Response

### Severity definitions

- Sev 1: service unavailable, data/security incident, or widespread failure.
- Sev 2: major degradation with workarounds.
- Sev 3: isolated defects or low-risk regressions.

### Incident steps

1. Declare incident and severity.
2. Stabilize first (feature flag disable, rollback, or revert).
3. Communicate status every 30-60 minutes for Sev 1/2.
4. Confirm recovery with smoke tests and key workflows.
5. Record incident summary and root cause.
6. Create follow-up action items with owners and dates.

## 6) Security Event Procedure

1. Rotate potentially exposed credentials immediately.
2. Restrict access surface (disable risky integrations temporarily).
3. Patch and deploy minimal fix.
4. Audit related logs and recent deploys.
5. Document timeline, blast radius, and corrective actions.

## 7) Repository Hygiene (Weekly)

1. Review Dependabot PRs and security alerts.
2. Triage new issues/PRs and apply labels.
3. Verify branch protection and required checks are unchanged.
4. Confirm backup access to release credentials and environments.
5. Sync with upstream and resolve drift early.

## 8) Ownership Exit/Expansion

If adding maintainers later:

1. Update `.github/CODEOWNERS`.
2. Update `docs/MAINTAINERS.md` with role boundaries.
3. Grant least-privilege GitHub permissions.
4. Require documented onboarding on this runbook.
