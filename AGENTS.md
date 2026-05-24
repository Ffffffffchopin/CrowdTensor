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
- Controlled remote Miner demos with token-backed admission, hashed token config, `/ready` preflight, retries, `remote_compute_observability_v1`, `remote_demo_observability_v1`, and Support Bundle diagnostics.
- Browser experiments for WebRTC tensor transport, Worker compute probes, and a browser Miner bridge.
- Release tooling: runtime capability matrix, matrix-guided home-compute demo, user-facing inference session demo, `inference_session_client_v1`, admin-created read-only inference session API check, external LLM adapter smoke, release gate, fresh clone onboarding gate, runtime acceptance pack, browser acceptance pack, release evidence, doctor diagnostics, security preflight, and Support Bundle. Preserve safe runtime `summary_json`, route `diagnosis_codes`, `operator_action`, `diagnosis_summary`, `diagnosis_by_check`, and remote `observability_summaries` across runtime, release evidence, and Support Bundle outputs.
- Release readiness gate: `crowdtensor release-ready` in `crowdtensor/cli.py` wraps `scripts/release_readiness_pack.py` and emits `release_readiness_v1` by aggregating Git metadata, the release gate, security preflight, and `demo_manifest_v1`. Dirty worktrees block by default with `git_dirty`; `scripts/release_readiness_check.py --allow-dirty` is only for development/CI smoke validation. It is an Alpha maintainer gate, not production Swarm Inference readiness.
- Fresh clone onboarding gate: `scripts/onboarding_gate.py --quick` emits `onboarding_gate_v1` by creating a clean temporary virtualenv, running `python -m pip install -e .[dev]`, checking `crowdtensor --help`, `crowdtensord --help`, and `crowdtensor-miner --help`, then smoke-validating `crowdtensor local-proof`, `crowdtensor home-infer`, `crowdtensor llm-infer --mock`, and `crowdtensor release-ready --allow-dirty`. It is a fresh-checkout onboarding gate, not production Swarm Inference readiness.
- One-command local proof: `crowdtensor local-proof` in `crowdtensor/cli.py` emits `local_proof_summary_v1` by chaining Doctor, runtime matrix, the CPU-only read-only home-compute demo, and the Demo Manifest path. It is a local proof, not production Swarm Inference.
- Home inference proof CLI: `crowdtensor home-infer` emits `home_inference_cli_v1`, wraps `scripts/home_compute_evidence_pack.py`, and writes `home_compute_evidence_v1` artifacts with the CPU-only read-only `model_bundle_infer` route, fixed `model_bundle_inference_scenario_v1` metadata, capped `request_trace`, `diagnosis_codes`, and read-only/redaction status. Built-in scenario IDs are `route-baseline`, `gradient-safety`, and `mixed-prompts`; it is not production Swarm Inference or arbitrary prompt serving.
- External LLM proof CLI: `crowdtensor llm-infer` emits `llm_inference_cli_v1`, wraps `scripts/external_llm_evidence_pack.py`, and writes `external_llm_evidence_v1` artifacts for the read-only `external_llm_infer` route. It records adapter kind, model id, request/completion count, output chars, throughput, and redaction status while keeping raw prompts, `output_text`, runtime URL, and API key out of public artifacts. It is fixed-prompt operator-owned runtime evidence, not public arbitrary prompt serving.
- Safe artifact cleanup: `crowdtensor clean-artifacts` emits `cleanup_report_v1`, defaults to dry-run, removes generated `__pycache__` / `.pyc` caches and old CrowdTensor temp directories only with `--apply`, keeps reports unless `--include-reports` is used, and does not delete state or source files.
- Remote demo operator CLI: `crowdtensor remote-runbook` emits `remote_runbook_cli_v1` and wraps `scripts/remote_demo_runbook_pack.py`; `crowdtensor remote-acceptance` emits `remote_acceptance_cli_v1`, defaults to `--create-session`, wraps `scripts/remote_demo_acceptance_pack.py`, carries fixed `model_bundle_inference_scenario_v1` scenarios such as `route-baseline`, and applies token redaction to captured command output. It is a controlled two-machine helper, not production Swarm Inference and not P2P routing.
- Remote home-compute demo CLI: `crowdtensor remote-demo prepare`, `crowdtensor remote-demo doctor`, `crowdtensor remote-demo verify`, `crowdtensor remote-demo collect`, and `crowdtensor remote-demo clean` emit `remote_home_compute_demo_v1`, `remote_home_compute_doctor_v1`, `remote_home_compute_collect_v1`, and `remote_home_compute_cleanup_v1` through `scripts/remote_home_compute_demo_pack.py`. The prepare path creates `operator.private.env`, `miner.private.env`, the hashed registry, and the public runbook; doctor checks local files, token presence, Coordinator reachability, task lane visibility, and optional accepted-result readiness; the default verify path uses `POST /admin/inference-sessions` for read-only `model_bundle_infer`, validates `remote_python_model_bundle_infer`, and summarizes `remote_compute_evidence_v1`, `remote_demo_observability_v1`, and Support Bundle artifacts; collect gathers evidence/support from an already running demo; clean defaults to dry-run and only removes private env/registry files with `--include-private`. `--workload external-llm` queues read-only `external_llm_infer`, validates `remote_python_external_llm_infer`, and summarizes `remote_external_llm_evidence_v1` plus `remote_external_llm_observability_v1` for deterministic `--mock` or explicit operator-owned `--llm-runtime-cmd` / `--llm-runtime-url` adapters. `scripts/remote_home_compute_demo_check.py` validates both local-loopback stand-ins across prepare, doctor, verify, collect, and clean. It is not production Swarm Inference, not P2P routing, not GPU pooling, and not public arbitrary prompt serving.
- Local multi-Miner scenario sweep: `scripts/multi_miner_scenario_sweep_check.py` defaults to concurrent mode and emits `multi_miner_scenario_sweep_v1` / `multi_miner_scenario_sweep_observability_v1` by creating fixed read-only inference sessions, starting distinct registry-backed Python Miner identities together, checking one accepted ledger row per task via `lease_summary`, checking process health via `process_summary`, and emitting `multi_miner_concurrent_ready`. With `--failure-mode kill-after-claim`, it terminates one claimed Miner, observes lease timeout requeue, requires a rescue Miner to complete the same `task_id`, records `requeue_summary`, and emits `multi_miner_requeue_ready`. It is a local lease-race/requeue proof, not production throughput scaling or P2P routing.
- Demo Manifest tooling: `scripts/demo_manifest_pack.py` and `scripts/demo_manifest_check.py` produce `demo_manifest_v1`, the current latest output artifact for local-loopback handoff. It indexes `runtime_matrix.json`, `remote_compute_evidence_v1`, `external_llm_evidence_v1`, `support_bundle`, and `remote_compute_observability_v1` summaries without widening the project claim.
- Open-source entrypoints: README, ROADMAP, protocol/use-case docs, release docs, changelog, and static site.

Do not describe the project as already providing production P2P, NAT traversal, real LLM inference/training, GPU pooling, WebGPU model shards, payments, staking, or hardened public-internet security.

## Strategic Direction

The next high-value product direction is a useful home-compute demo, likely Swarm Inference shaped before real Swarm Training. Training and P2P remain important, but open-source users need a concrete deployment story first.

Roadmap priority:

1. Preserve Alpha reliability and operator trust.
2. Make the project easy for strangers to understand and run.
3. Keep `scripts/runtime_matrix.py` and `scripts/runtime_matrix_check.py` as the first runtime capability matrix and hardware/runtime matrix for new users, including `matched_capabilities`, `missing_capabilities`, target and route `diagnosis_codes`, `operator_action`, and `hardware_diagnosis_summary` explanations.
4. Keep `crowdtensor release-ready` as the maintainer-facing publish gate, preserving `release_readiness_v1`, `scripts/release_readiness_pack.py`, `scripts/release_readiness_check.py`, `--allow-dirty`, `git_dirty`, release gate aggregation, `demo_manifest_v1`, and explicit not production boundaries.
5. Keep `scripts/onboarding_gate.py --quick` as the fresh clone install-and-run proof, preserving `onboarding_gate_v1`, clean virtualenv creation, `python -m pip install -e .[dev]`, console script checks, `crowdtensor local-proof`, `crowdtensor home-infer`, `crowdtensor llm-infer --mock`, `crowdtensor release-ready --allow-dirty`, `/tmp` output defaults, and explicit non-production Swarm Inference boundaries.
6. Keep `crowdtensor local-proof` as the shortest user-facing local proof, preserving `local_proof_summary_v1`, Doctor, runtime matrix, CPU-only read-only home-compute demo, Demo Manifest output, and explicit non-production Swarm Inference boundaries.
7. Keep `crowdtensor home-infer` as the shortest shareable local read-only inference proof, preserving `home_inference_cli_v1`, `home_compute_evidence_v1`, `model_bundle_infer`, fixed `model_bundle_inference_scenario_v1` scenarios, capped `request_trace`, `diagnosis_codes`, and explicit non-production Swarm Inference boundaries.
8. Keep `crowdtensor llm-infer` as the shortest shareable external LLM runtime proof, preserving `llm_inference_cli_v1`, `external_llm_evidence_v1`, deterministic `--mock`, explicit operator-owned `--llm-runtime-cmd` / `--llm-runtime-url`, fixed claim-time prompts, read-only semantics, and explicit non-public-serving boundaries.
9. Keep `crowdtensor clean-artifacts` as the safe maintenance path for repeated agent runs, preserving `cleanup_report_v1`, dry-run default, `--apply`, `--include-reports`, and the rule that cleanup does not delete state or source files.
10. Keep `crowdtensor remote-runbook` and `crowdtensor remote-acceptance` as the operator-facing wrappers for the controlled two-machine path, preserving `remote_runbook_cli_v1`, `remote_acceptance_cli_v1`, fixed scenario propagation, token redaction, default `--create-session`, and explicit not production / not P2P boundaries.
11. Keep `crowdtensor remote-demo prepare` / `doctor` / `verify` / `collect` / `clean` as the high-level two-machine home-compute demo, preserving `remote_home_compute_demo_v1`, `remote_home_compute_doctor_v1`, `remote_home_compute_collect_v1`, `remote_home_compute_cleanup_v1`, `scripts/remote_home_compute_demo_pack.py`, `scripts/remote_home_compute_demo_check.py`, private `operator.private.env` / `miner.private.env`, dry-run cleanup defaults, `POST /admin/inference-sessions`, `model_bundle_infer`, `remote_python_model_bundle_infer`, `remote_compute_evidence_v1`, `remote_demo_observability_v1`, `--workload external-llm`, `external_llm_infer`, `remote_python_external_llm_infer`, `remote_external_llm_evidence_v1`, `remote_external_llm_observability_v1`, token/runtime redaction, and explicit not production / not P2P / not public prompt-serving boundaries.
12. Keep expanding `scripts/home_compute_demo.py` around the current read-only multi-request `model_bundle_infer` probe into a useful home-compute inference demo with explicit `hardware_targets`, `recommended_routes`, `route_decision`, capped `request_trace` summaries, stable `diagnosis_codes` such as `home_compute_ready` and `runtime_matrix_blocked`, and hardware/capability matching.
13. Keep `scripts/home_compute_evidence_pack.py` and `scripts/home_compute_evidence_check.py` as the safe, shareable `home_compute_evidence_v1` layer for public issue reports and demos, preserving `route_decision`, `matched_capabilities`, `diagnosis_codes`, and capped `request_trace` while redacting secret-shaped fields.
14. Keep `scripts/inference_session_client.py` and `scripts/inference_session_client_check.py` as the narrow user-facing client for a running Coordinator, preserving `inference_session_client_v1`, `session_client_ready`, `POST /admin/inference-sessions`, `task_id` filtering, `model_bundle_infer`, read-only semantics, and `--skip-inference-session-client` acceptance control.
15. Keep `scripts/admin_inference_session_check.py` as the narrow service-shaped API acceptance path for `POST /admin/inference-sessions`, preserving `inference_session_request_v1`, `task_id` filtering, `model_bundle_infer`, read-only semantics, and `--skip-admin-inference-session` acceptance control.
16. Keep `scripts/remote_compute_evidence_pack.py` and `scripts/remote_compute_evidence_check.py` as the safe, shareable `remote_compute_evidence_v1` layer for registry-backed remote-style Python Miner demos, preserving `remote_python_model_bundle_infer`, `remote_compute_observability_v1`, fixed `model_bundle_inference_scenario_v1` metadata and scenario match status, safe metrics, capped `request_trace`, and hashed registry status.
17. Keep `scripts/remote_demo_runbook_pack.py` and `scripts/remote_demo_runbook_check.py` as the safe two-machine `remote_demo_runbook_v1` path, preserving `operator.private.env`, `miner.private.env`, `model_bundle_infer`, `--scenario-id route-baseline`, and `remote_compute_evidence_pack.py --mode collect`.
18. Keep `scripts/remote_demo_acceptance_pack.py` and `scripts/remote_demo_acceptance_check.py` as the safe two-machine `remote_demo_acceptance_v1` layer that can use `--create-session` to call `POST /admin/inference-sessions` with `scenario_id`, wait for the returned `task_id`, verify scenario match, and collect `remote_compute_evidence_v1`, `remote_demo_observability_v1`, and `support_bundle`, with stable `diagnosis_codes` such as `coordinator_unreachable`, `observer_auth_failed`, `session_create_failed`, and `artifact_collection_failed`.
19. Keep `scripts/multi_miner_scenario_sweep.py` and `scripts/multi_miner_scenario_sweep_check.py` as the controlled local multi-Miner lease-race and failure-requeue proof, preserving concurrent mode, `multi_miner_scenario_sweep_v1`, `multi_miner_scenario_sweep_observability_v1`, three fixed scenarios, distinct Miner identities, `local_multi_miner_model_bundle_infer`, `lease_summary`, `process_summary`, `requeue_summary`, `multi_miner_concurrent_ready`, `multi_miner_requeue_ready`, and `--include-multi-miner-sweep` / `--include-multi-miner-requeue` opt-in coverage.
20. Keep `scripts/demo_manifest_pack.py` and `scripts/demo_manifest_check.py` as the latest output artifact entrypoint for local-loopback handoff, combining runtime matrix, remote-compute evidence, deterministic mock external LLM evidence, and support bundle summaries.
21. Treat `external_llm_infer_v1` / `external_llm_evidence_v1` as the narrow optional runtime adapter boundary: use `--enable-mock-llm-runtime` or `--mock` for deterministic checks, `--llm-runtime-cmd` / `CROWDTENSOR_LLM_RUNTIME_CMD` for operator-owned command wrappers, and `--llm-runtime-url` / `CROWDTENSOR_LLM_RUNTIME_URL` for OpenAI-compatible local servers.
22. Grow toward remote Miners, browser-native participation, optional GPU/runtime adapters, and then P2P/NAT routing.
23. Treat incentives and reputation as later protocol layers, not prerequisites for local demos.

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

Use the normal repository Git metadata from the project root:

```bash
git status --short --branch
```

Do not commit local state directories, token files, browser profiles, checkpoints, generated caches, or secrets.

Before public release work, read:

- [Project Memory](docs/project-memory.md)
- [Roadmap](ROADMAP.md)
- [Protocol Boundary](docs/protocol.md)
- [Release Process](docs/release.md)
