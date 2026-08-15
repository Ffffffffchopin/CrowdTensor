# Threat Model

## Assets

- model/dataset revisions and checkpoint lineage;
- Work Unit ownership and exactly-once receipts;
- private training rows, token IDs, tensors, credentials, and local paths;
- contributor and provider resource limits;
- public claims about model quality and execution evidence.

## Trust Boundaries

The user-owned Session controller is trusted. Admitted contributors may be
faulty, stale, slow, malformed, or compromised. Model/backend plugins execute
trusted Python. TLS proxies, object stores, model registries, and accelerator
providers are independently administered external systems.

## Covered Failures

The controller fences lease expiry, stale generations, old checkpoints,
duplicate submissions, owner mismatch, and replay. Volunteer validation checks
hashes, schemas, bounds, shapes, finiteness, norms, and quotas. Public views
exclude credentials, identities, raw data, and tensor payloads.

## Residual Risk

The system does not identify a malicious contributor's real-world identity or
prove semantic update quality. It does not provide Byzantine consensus, secure
aggregation, differential privacy, trusted execution, side-channel defense, or
unbounded denial-of-service protection. Operator and plugin compromise can
invalidate the Session.

Use controlled admission, pinned dependencies, independent benchmark splits,
quorum where appropriate, checkpoint rollback, bounded exposure, and a public
decision log. Do not describe the current system as permissionless-safe.
