# Community RC Threat Model

## Assets

- owner, Miner, observer, Hugging Face, Kaggle, and object-store credentials;
- private prompts/text, token IDs, activations, gradients, checkpoints, and
  adapter tensors;
- job integrity, exactly-once ledger state, compute quota, and release claims.

## Trust Boundaries

The owner and Coordinator are trusted. Admitted Miners are controlled but may
be faulty, stale, slow, malformed, or compromised. A reverse proxy is trusted
only when its identity hash is configured. Kaggle and MinIO are external
execution/storage systems under their own policies. Public artifacts and
release bundles are untrusted outputs until scanned.

## Implemented Controls

- owner/miner/observer default-deny RBAC;
- short-lived HMAC credentials with overlapping key rotation;
- signed task envelopes, expiry, nonces, and bounded replay windows;
- TLS/reverse-proxy contract with trusted forwarded-header identity;
- lease generation fencing, duplicate/late-result rejection, byte/disk quotas,
  backpressure, cancellation, cleanup retry, and quarantine;
- restricted worker executable/module/file/network/resource policy;
- finite, shape, absolute-norm, and historical-norm anomaly isolation;
- content-addressed checkpoints, mirror fallback/repair, retention, and restart;
- recursive public-safety scanning for credential material, private URLs,
  absolute paths, raw text/token IDs, activations, gradients, and tensors.

## Residual Risks

The controls do not solve semantic poisoning, Sybil identities, Byzantine
consensus, colluding Miners, privacy-preserving computation, secure
aggregation, differential privacy, trusted execution environments, side
channels, or denial of service outside configured quotas. LoRA norm checks are
not proof that an update is useful or benign.

Do not expose the Coordinator directly to an untrusted public network. Put TLS,
rate limiting, request-size limits, and operator monitoring in front of it.
Rotate credentials after temporary tunnel or hosted-notebook experiments.
