# Project Governance

CrowdTensor uses maintainer review and evidence-backed RFCs.

- Maintainers merge code, manage releases, and enforce security boundaries.
- Contributors own tests and documentation for behavior they change.
- Model/Provider additions, protocol changes, security-boundary changes, and
  incompatible artifact schemas require an RFC under `docs/rfcs/`.
- Small bug fixes and documentation corrections can use ordinary issues/PRs.
- Release readiness is determined by the strict checker, not maintainer intent
  or a passing subset of tests.

Decisions should record alternatives, compatibility, migration, security,
resource cost, evidence, rollback, and explicit non-goals. Maintainers may
reject claims that exceed retained evidence even when code is promising.

Security reports follow `SECURITY.md`. Conduct follows `CODE_OF_CONDUCT.md`.
Apache-2.0 applies to accepted contributions unless a file states otherwise.
