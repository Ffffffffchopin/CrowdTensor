# Agent Instructions for CrowdTensor

Read this file before making changes in this repository. It is the short, durable project memory for future agents and contributors.

## Project Identity

CrowdTensor is the project and network vision: open AI infrastructure that can eventually use ordinary home compute for fault-tolerant AI workloads.

CrowdTensorD is the current Alpha daemon/control plane. It validates the reliability and protocol mechanics needed before real home GPU aggregation, Swarm Inference, Swarm Training, browser compute, and P2P routing are added.

## Current Alpha Reality

The current code supports:

- Coordinator/Miner task leasing, heartbeat recovery, timeout requeue, stale result rejection, and checkpoint replay.
- Deterministic CPU-only workloads: `diloco_train`, `cpu_lora_mock`, `micro_transformer_lm`, `model_bundle_lm`, measurable read-only `model_bundle_infer`, optional read-only `external_llm_infer`, and `browser_probe`.
- `runtime_contract_v1`, workload capability matching, CPU `hardware_profile`, delta transport negotiation, validation, replay audit, result ledger, trust quarantine, and operator controls.
- Controlled remote Miner demos with token-backed admission, hashed token config, `/ready` preflight, retries, and Support Bundle diagnostics.
- Browser experiments for WebRTC tensor transport, Worker compute probes, and a browser Miner bridge.
- Release tooling: runtime capability matrix, matrix-guided home-compute demo, user-facing inference session demo, external LLM adapter smoke, release gate, runtime acceptance pack, browser acceptance pack, release evidence, doctor diagnostics, security preflight, and Support Bundle.
- Open-source entrypoints: README, ROADMAP, protocol/use-case docs, release docs, changelog, and static site.

Do not describe the project as already providing production P2P, NAT traversal, real LLM inference/training, GPU pooling, WebGPU model shards, payments, staking, or hardened public-internet security.

## Strategic Direction

The next high-value product direction is a useful home-compute demo, likely Swarm Inference shaped before real Swarm Training. Training and P2P remain important, but open-source users need a concrete deployment story first.

Roadmap priority:

1. Preserve Alpha reliability and operator trust.
2. Make the project easy for strangers to understand and run.
3. Keep `scripts/runtime_matrix.py` and `scripts/runtime_matrix_check.py` as the first runtime capability matrix and hardware/runtime matrix for new users, including `matched_capabilities` and `missing_capabilities` route explanations.
4. Keep expanding `scripts/home_compute_demo.py` around the current read-only multi-request `model_bundle_infer` probe into a useful home-compute inference demo with explicit `hardware_targets`, `recommended_routes`, `route_decision`, capped `request_trace` summaries, and hardware/capability matching.
5. Treat `external_llm_infer_v1` as the narrow optional runtime adapter boundary: use `--enable-mock-llm-runtime` for deterministic checks, `--llm-runtime-cmd` / `CROWDTENSOR_LLM_RUNTIME_CMD` for operator-owned command wrappers, and `--llm-runtime-url` / `CROWDTENSOR_LLM_RUNTIME_URL` for OpenAI-compatible local servers.
6. Grow toward remote Miners, browser-native participation, optional GPU/runtime adapters, and then P2P/NAT routing.
7. Treat incentives and reputation as later protocol layers, not prerequisites for local demos.

## Engineering Rules

- Keep network/control-plane code physically separate from workload compute code.
- Keep CPU-only deterministic smoke paths working even when optional accelerators are added.
- Version protocol changes; preserve `runtime_contract_v1` unless a change is intentionally versioned.
- New workload contracts should not mutate task lease or heartbeat semantics.
- Do not expose raw lease tokens, idempotency material, tensor deltas, registry tokens, raw external LLM `output_text`, or raw state in operator-friendly outputs.
- Prefer narrow, testable additions over broad rewrites.
- Update public docs, changelog, roadmap, and project memory when user-visible behavior or strategy changes.

## Validation Commands

Run focused checks first, then broader checks when changing shared behavior:

```bash
python3 scripts/release_gate.py --json
python3 -m unittest tests.test_release_gate -v
python3 -m py_compile coordinator.py miner_cli.py scripts/*.py crowdtensor/*.py
python3 -m unittest discover -s tests -v
```

For runtime behavior changes, also run the acceptance pack from a normal shell that permits localhost networking:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8910 \
  --report /tmp/crowdtensor_acceptance.json
```

Browser checks are opt-in:

```bash
python3 scripts/browser_acceptance_pack.py \
  --allow-skip \
  --base-port 9310 \
  --report /tmp/crowdtensor_browser_acceptance.json
```

## Git and Release Notes

Use the external git directory when this workspace is configured that way:

```bash
git --git-dir=/tmp/crowdtensor-release-git --work-tree=/root/sync/Codes/hivemind status --short --branch
```

Do not commit local state directories, token files, browser profiles, checkpoints, generated caches, or secrets.

Before public release work, read:

- [Project Memory](docs/project-memory.md)
- [Roadmap](ROADMAP.md)
- [Protocol Boundary](docs/protocol.md)
- [Release Process](docs/release.md)
