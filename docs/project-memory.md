# Project Memory

This document is the long-form durable memory for CrowdTensor. It exists so future development sessions can recover project intent, current facts, and engineering boundaries even when chat context is lost.

## Naming and Positioning

CrowdTensor is the open-source network vision: fault-tolerant AI swarms built from ordinary home compute.

CrowdTensorD is the current Alpha daemon/control plane inside this repository. It proves leasing, validation, recovery, observability, and operator control before the project claims real model-scale home GPU aggregation.

The target audience is:

- home open-model players with limited local compute
- remote Miner operators who can contribute controlled Linux/container capacity
- browser experimenters interested in WebRTC/WebGPU-style participation
- protocol contributors building reliable distributed AI workload contracts

The project should be honest about status. It is an Alpha control plane, not yet a production DePIN network or real LLM deployment platform.

## Current Completed Capabilities

The project currently includes:

- FastAPI Coordinator with task queues, task lanes, leases, heartbeat deadlines, checkpoint state, append-only event replay, result validation, replay audit, metrics, admin result ledger, and trust overrides.
- Python Miner CLI with capability advertisement, CPU `hardware_profile`, `/ready` preflight, bounded retry behavior, result `idempotency_key`, heartbeats, and bounded session controls.
- Deterministic CPU-only workload contracts: `diloco_train`, `cpu_lora_mock`, `micro_transformer_lm`, `model_bundle_lm`, `model_bundle_infer`, optional `external_llm_infer`, and `browser_probe`.
- Protocol boundary around `runtime_contract_v1`, `outer_optimizer_contract_v1`, supported workloads, supported delta formats, and workload-specific validation.
- Delta transport paths for `dense_float`, `sign_compressed`, and `sign_compressed_ef`.
- Admission and operator safety: shared Miner token, per-Miner token registry, observer token, admin token, hashed token verifiers, security preflight, redacted `/state`, aggregate `/metrics`, and safe admin ledger views.
- Controlled remote Miner onboarding through invite generation, readiness checks, remote join checks, retry counters, and Support Bundle diagnostics.
- Browser experiments for WebRTC tensor tunnel, browser Worker compute probe, and browser Miner bridge.
- Release and support tooling: First-run Doctor, user-facing inference session demo, release gate, runtime acceptance pack, browser acceptance pack, release evidence pack, Support Bundle, changelog, release process docs, roadmap, protocol docs, use-case docs, and static site.

## Explicit Non-Capabilities

Do not imply these are implemented:

- real Swarm Inference for production LLM serving
- real Swarm Training or LLM fine-tuning
- GPU pooling across home machines
- WebGPU model shard execution
- libp2p discovery or NAT traversal
- decentralized identity or public P2P routing
- reward, staking, payment, or token economics
- hardware attestation
- hardened public-internet security

The current model bundle, measurable multi-request model bundle inference, optional `external_llm_infer_v1` adapter, and micro LM workloads are dependency-free contract rehearsals, not real LLM or GPU throughput benchmarks. `external_llm_infer` is read-only, validates `external_llm_results`, records safe `completion_count`, `output_chars`, `adapter_kind`, and `model_id` summaries, supports deterministic mock, command, and OpenAI-compatible HTTP adapters, and must keep raw prompts, raw `output_text`, runtime URLs, and API keys out of public state.

## Strategic Route

The near-term goal is to make CrowdTensor credible and useful for open-source users before making large model-scale claims.

Recommended sequence:

1. Keep the Alpha control plane reliable, testable, and well documented.
2. Keep README, ROADMAP, protocol docs, use cases, static site, and project memory synchronized.
3. Expand the read-only multi-request `model_bundle_infer` probe into a useful home-compute demo that feels close to Swarm Inference, starting with small model artifacts and explicit capability matching.
4. Keep `external_llm_infer_v1` as the narrow optional runtime adapter contract: deterministic `--enable-mock-llm-runtime` for CI, explicit `--llm-runtime-cmd` / `CROWDTENSOR_LLM_RUNTIME_CMD` for operator-owned local experiments, and `--llm-runtime-url` / `CROWDTENSOR_LLM_RUNTIME_URL` for OpenAI-compatible local servers.
5. Add hardware/runtime matrices for CPU, NVIDIA, AMD, Apple Silicon, browser, and remote container paths.
6. Introduce optional GPU/runtime adapters without making the control plane depend on one framework.
7. Expand browser-native participation from WebRTC/Worker probes toward WebGPU/WebAssembly only after tensor transfer and lifecycle limits are measured.
8. Add P2P/NAT routing after useful workloads and operator safety are proven.
9. Treat reputation and incentives as later protocol layers built on result validation and trust history.

## Engineering Principles

Network orchestration and tensor computation must stay decoupled. Task leasing, heartbeat, retries, validation, and operator state belong to the control plane; workload math belongs behind explicit workload contracts.

CPU-only deterministic smoke paths are strategic. They let CI, restricted Linux environments, and users without GPU access validate behavior. Optional accelerator paths must not remove or weaken these tests.

Protocol changes must be explicit and versioned. Current protocol names like `runtime_contract_v1` and `outer_optimizer_contract_v1` are compatibility boundaries.

Operator outputs must be safe by default. Support Bundle, `/metrics`, admin result ledger, and redacted state should avoid raw tokens, lease tokens, idempotency material, tensor deltas, raw registry secrets, and full raw state dumps.

Release quality matters. If a behavior becomes user-visible, update docs, tests, release gate expectations, changelog when appropriate, roadmap if strategic direction changes, and this project memory if the long-term story changes.

## Development Checks

Baseline checks:

```bash
python3 scripts/release_gate.py --json
python3 -m unittest tests.test_release_gate -v
python3 -m py_compile coordinator.py miner_cli.py scripts/*.py crowdtensor/*.py
python3 -m unittest discover -s tests -v
```

Runtime checks for Coordinator/Miner behavior:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8910 \
  --report /tmp/crowdtensor_acceptance.json
```

Browser checks when Playwright/Chromium are available:

```bash
python3 scripts/browser_acceptance_pack.py \
  --allow-skip \
  --base-port 9310 \
  --report /tmp/crowdtensor_browser_acceptance.json
```

Support Bundle for issue reports:

```bash
python3 scripts/support_bundle.py \
  --json-out /tmp/crowdtensor_support_bundle.json
```

## Maintenance Rule

If future work changes project identity, target users, protocol boundaries, implemented capability, non-capability claims, roadmap priority, validation commands, or release workflow, update this document and `AGENTS.md` in the same change.
