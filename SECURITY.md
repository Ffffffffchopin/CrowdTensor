# Security Policy

CrowdTensorD is currently an experimental alpha. It is designed for local and controlled remote demos, not as a hardened public-internet Coordinator.

## Reporting Security Issues

Please report suspected vulnerabilities through GitHub Security Advisories for this repository. If advisories are unavailable, open a minimal public issue that avoids exploit details and asks for a private contact path.

Do not publish working exploit steps, live tokens, private endpoints, or sensitive deployment details in public issues.

## Supported Versions

Only the current `main` branch is supported during the alpha phase. Tagged compatibility and long-term support policies will be defined after the control plane reaches beta readiness.

## Current Security Boundaries

The current implementation includes:

- Miner, observer, and admin token gates.
- Optional hashed token configuration.
- Per-Miner token registry support.
- Redaction for lease tokens and result idempotency material in public state/event views.
- Workload-scoped Miner quarantine and operator trust overrides.

The current implementation does not yet include:

- Byzantine fault-tolerant aggregation.
- End-to-end encrypted P2P transport.
- Reward, staking, slashing, or payment security.
- Production-grade identity, rate limiting, or abuse prevention.
- A hardened deployment profile for exposing Coordinator APIs directly to the public internet.

See [docs/security.md](docs/security.md) for operational guidance.
