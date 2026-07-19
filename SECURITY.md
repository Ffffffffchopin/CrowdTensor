# Security Policy

CrowdTensor is a controlled Community RC. It is not a hardened permissionless
public-internet training network or a production SLA.

## Reporting Security Issues

Please report suspected vulnerabilities through GitHub Security Advisories for this repository. If advisories are unavailable, open a minimal public issue that avoids exploit details and asks for a private contact path.

Do not publish working exploit steps, live tokens, private endpoints, or sensitive deployment details in public issues.

## Supported Versions

Only the current `main` branch and `0.2.0rc1` Community release candidate are
supported. Long-term support begins only after a future GA policy.

## Current Security Boundaries

The current implementation includes:

- Miner, observer, and admin token gates.
- Community owner/miner/observer default-deny RBAC.
- Short-lived rotatable credentials, signed task envelopes, and replay windows.
- A trusted TLS/reverse-proxy header contract.
- Optional hashed token configuration.
- Per-Miner token registry support.
- Redaction for lease tokens and result idempotency material in public state/event views.
- Workload-scoped Miner quarantine and operator trust overrides.
- Restricted worker command/module/file/network/resource policy.
- Finite/shape/norm anomaly isolation and content-addressed checkpoint repair.
- Public artifact scanning for credentials, private URLs, paths, text/token IDs,
  activations, gradients, and tensors.

The current implementation does not establish:

- Byzantine fault-tolerant aggregation.
- Sybil or semantic-poisoning resistance.
- Secure aggregation, privacy-preserving computation, differential privacy, or
  trusted execution environments.
- End-to-end encrypted permissionless P2P transport.
- Reward, staking, slashing, or payment security.
- A hardened deployment profile for exposing Coordinator APIs directly to the public internet.

See [docs/threat-model.md](docs/threat-model.md) and
[docs/security.md](docs/security.md) for boundaries and operational guidance.
