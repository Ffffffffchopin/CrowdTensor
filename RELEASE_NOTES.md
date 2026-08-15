# CrowdTensor 0.3.0a1

CrowdTensor 0.3.0a1 is the first compact Training Architecture v2 alpha. It
coordinates bounded volunteer training contributions while upstream libraries
continue to own model execution and kernels.

## What Is Included

- `elastic_delta` sessions for intermittent PEFT contributors;
- `stable_sharded` sessions for stable FSDP2-style rank groups;
- Work Units, capability-aware placement, checkpoint lineage, stale-worker
  fencing, validation, and exactly-once contribution receipts;
- user-owned Campaign operators with Campaign-scoped release downloads;
- native CPU/CUDA contributor preflight, bounded work, local status, and
  graceful stop;
- optional held-out loss and perplexity comparison for compatible Campaigns.

Operators can stage and verify the exact same-origin contributor artifacts with
`crowdtensor release prepare` and `crowdtensor release verify`. The installer
checks the wheel SHA-256 before creating a contributor environment. Interrupted
Campaign downloads can be resumed; slow PyTorch installs use bounded retries
and a longer read timeout, and operators may provide a predownloaded trusted
PyTorch wheel with `CROWDTENSOR_TORCH_WHEEL_PATH`.

## Evidence Boundary

The retained 7B showcase proves resumable Kaggle logical-worker LoRA training,
not independently administered physical multi-host operation. This alpha does
not claim permissionless admission, poisoning or Sybil resistance, arbitrary
model compatibility, full-parameter volunteer training, GA, or an SLA.

Install only from a Campaign operator you trust. Verify the wheel against the
Campaign's `SHA256SUMS` and `release.json` before joining.
