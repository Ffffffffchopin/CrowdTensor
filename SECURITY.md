# Security Policy

CrowdTensor is an alpha collaborative-training system. It is not a hardened
permissionless network or a production SLA.

## Reporting

Use GitHub Security Advisories for suspected vulnerabilities. If advisories are
unavailable, open a minimal public issue asking for a private contact path.
Do not publish exploit steps, credentials, private endpoints, model data, or
sensitive deployment details.

## Supported Versions

Security fixes target the current `main` branch and latest published release
candidate. No LTS commitment exists before GA.

## Scope

Relevant reports include authentication/scope bypass, replay or stale-work
acceptance, checkpoint/receipt tampering, unsafe archive extraction, public
data leakage, quota bypass, and remote code execution in default paths.

Known non-capabilities include Sybil and semantic-poisoning resistance,
Byzantine consensus, secure aggregation, privacy against the operator,
permissionless P2P security, billing/reward security, and a hardened public
deployment profile.

See [the operational boundary](docs/security.md) and
[threat model](docs/threat-model.md).
