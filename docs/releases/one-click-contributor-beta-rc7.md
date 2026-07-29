# One-Click Contributor Beta RC7

CrowdTensor `0.2.0rc7` reduces controlled-beta enrollment to one pairing code
and at most three visible steps.

## Contributor Paths

- `/join` runs a bounded browser scheduler-calibration task in a dedicated Web
  Worker. It prefers WebGPU and falls back to WASM/CPU. The Coordinator
  independently recomputes the result before accepting it.
- The native Agent accepts a Coordinator URL plus a one-time Agent code. It
  auto-detects CPU/CUDA, applies local work/download limits, serves a
  loopback-only status page, and converts SIGINT/SIGTERM into a stop request
  after the current atomic work unit.
- The installer uses CPU-only PyTorch wheels on CPU hosts, pins the verified
  contributor runtime, and accepts a Provider PyTorch index override for CUDA
  hosts. S3 dependencies are not installed for the ordinary Agent path.
- JAX/TPU remains available through the managed heterogeneous-stage Miner
  workflow. It is not supported by the one-click Volunteer Agent in this RC.
- Legacy mode-0600 invite files remain compatible for existing Operators.

## Security And Scope

Pairing records contain only code hashes, expiry, mode, and aggregate counters.
Redeeming a code consumes it atomically and returns a short-lived, Cell-bound,
scoped credential. Browser credentials cannot claim LoRA work; native Agent
credentials cannot claim browser work.

Browser calibration is not browser LoRA training, a model update, WebGPU model
sharding, anonymous admission, or useful-model quality evidence. Enrollment is
still controlled. This release does not claim permissionless Sybil/poisoning
safety, Kimi K3 support, GA, or an SLA.

## Install

```bash
curl -fsSL https://crowdtensor.24.199.118.54.nip.io/downloads/install-contributor.sh | sh -s -- https://crowdtensor.24.199.118.54.nip.io CT-XXXX-XXXX-XXXX
```

The release directory also contains `install-contributor.sh`, `SHA256SUMS`,
and `release.json`. Verify the wheel before use:

```bash
curl -fsSLO https://crowdtensor.24.199.118.54.nip.io/downloads/SHA256SUMS
sha256sum -c SHA256SUMS --ignore-missing
```
