# Changelog

All notable CrowdTensorD Alpha changes are tracked here.

## 0.1.0a0 - Alpha

CrowdTensorD is currently an experimental control plane for fault-tolerant distributed AI workload Miners. This release is intended for local development, controlled remote demos, and implementation review.

### Added

- Coordinator and Miner loop with task claims, heartbeats, lease timeout recovery, checkpoint replay, and result submission.
- Deterministic CPU-only workload contracts, including dense toy training, CPU LoRA mock, micro Transformer LM, model bundle LM, and read-only multi-request model bundle inference.
- Runtime validation for finite values, tensor shape, loss/norm gates, deterministic replay audit, and low-frequency outer optimizer behavior.
- Miner admission controls with shared tokens, per-Miner token registry, hashed token verifiers, observer/admin separation, and security preflight checks.
- Operator views for `/health`, `/version`, `/ready`, `/metrics`, redacted `/state`, admin event tails, and admin result ledger.
- Remote Miner demo flow with invite generation, readiness checks, retry counters, result `idempotency_key`, and controlled remote acceptance.
- Browser experiments for WebRTC tensor transfer, browser compute probes, and browser Miner bridge smoke tests.
- Release tooling: `scripts/release_gate.py`, `scripts/runtime_acceptance_pack.py`, `scripts/browser_acceptance_pack.py`, `scripts/release_evidence_pack.py`, and `scripts/support_bundle.py`.
- User-facing local inference session demo with safe latency, throughput, read-only, redaction, and Miner hardware profile summaries.

### Known Limitations

- This is not a production DePIN network, payment system, public-internet security layer, or real LLM training platform.
- Current workloads are intentionally small and CPU-friendly so reliability behavior can be tested without GPU access.
- Browser and remote Miner paths are controlled demos; they require operator-provided transport security when used off localhost.
- P2P discovery, NAT traversal, GPU execution, WebGPU model shards, and real distributed LLM fine-tuning remain future work.

### Verification

Before publishing this alpha, maintainers should run the release flow in [docs/release.md](docs/release.md), including release gate, unit tests, runtime acceptance, release evidence, and Support Bundle generation.
