# Security Boundary

CrowdTensor coordinates untrusted or unreliable contributed compute but is not
a Byzantine-complete or privacy-preserving training system.

## Implemented Controls

- schema/content-hash binding for projects, Work Units, checkpoints, plans, and
  receipts;
- lease generation, expiry, old-base, contributor-owner, and replay fencing;
- exactly-once receipt/checkpoint commit with restart recovery;
- finite, shape, byte-size, norm, and metadata validation for Volunteer deltas;
- short-lived Cell-bound credentials, scopes, revocation, nonce replay windows,
  request/upload quotas, and bounded concurrent leases;
- HTTPS/trusted-proxy contract and bounded request bodies;
- content-addressed local or S3-compatible resumable storage;
- recursive public projections that remove credentials, private paths, raw
  rows, token IDs, tensor values, and Cell identifiers;
- bounded contributor download, local-step, Work-Unit, timeout, and device
  policies.

## Trust Model

The Session owner/operator and selected model/backend plugins are trusted.
Contributors may disappear, be slow, replay old work, submit malformed values,
or run compromised machines. Upstream model/runtime packages and storage or
compute providers remain separate trust boundaries.

For remote use, bind the Python service to loopback and place a maintained TLS
proxy with rate/request-size limits in front of it. Never publish Campaign
invites, pairing codes, scoped credentials, storage keys, or private endpoints.

## Not Solved

- Sybil identities or permissionless admission;
- semantic data/update poisoning and colluding contributors;
- Byzantine consensus or secure aggregation;
- differential privacy, confidential computing, or privacy against operator;
- denial of service beyond configured local quotas;
- rewards, billing, staking, or slashing;
- a hardened GA deployment profile or SLA.

Shape/finiteness/norm checks are necessary integrity controls, not proof that
an update is useful or benign. Public Campaigns need admission, independent
evaluation, moderation, rollback, and incident owners.

See [the threat model](threat-model.md) and root
[security policy](../SECURITY.md).
