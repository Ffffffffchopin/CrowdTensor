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
- Release and support tooling: First-run Doctor, runtime capability matrix, matrix-guided home-compute demo, user-facing inference session demo, `inference_session_client_v1`, admin-created read-only inference session API check, release gate, fresh clone onboarding gate, release readiness gate, runtime acceptance pack, browser acceptance pack, release evidence pack, Support Bundle, changelog, release process docs, roadmap, protocol docs, use-case docs, and static site. Runtime acceptance emits safe per-check `summary_json` and top-level `diagnosis_summary`; release evidence and Support Bundle preserve `diagnosis_by_check` plus safe remote `observability_summaries` for operator triage.
- Release readiness gate: `crowdtensor release-ready` in `crowdtensor/cli.py` wraps `scripts/release_readiness_pack.py` and emits `release_readiness_v1` by aggregating Git metadata, the release gate, security preflight, and `demo_manifest_v1`. Dirty worktrees block by default with `git_dirty`; `scripts/release_readiness_check.py --allow-dirty` is the development/CI smoke path. It is not production Swarm Inference readiness.
- Fresh clone onboarding gate: `scripts/onboarding_gate.py --quick` emits `onboarding_gate_v1` by creating a clean temporary virtualenv, running `python -m pip install -e .[dev]`, checking `crowdtensor --help`, `crowdtensord --help`, and `crowdtensor-miner --help`, then smoke-validating `crowdtensor local-proof`, `crowdtensor home-infer`, `crowdtensor llm-infer --mock`, and `crowdtensor release-ready --allow-dirty`. It is a fresh-checkout onboarding gate, not production Swarm Inference readiness.
- One-command local proof: `crowdtensor local-proof` in `crowdtensor/cli.py` emits `local_proof_summary_v1` by chaining Doctor, runtime matrix, the CPU-only read-only home-compute demo, and the Demo Manifest path. It is not production Swarm Inference; it is a user-facing local proof artifact.
- Home inference proof CLI: `crowdtensor home-infer` emits `home_inference_cli_v1`, wraps `scripts/home_compute_evidence_pack.py`, and writes `home_compute_evidence_v1` JSON/Markdown with the CPU-only `model_bundle_infer` route, fixed `model_bundle_inference_scenario_v1` metadata, capped `request_trace`, `diagnosis_codes`, read-only status, and redaction status. Built-in scenario IDs are `route-baseline`, `gradient-safety`, and `mixed-prompts`; it is not production Swarm Inference or arbitrary prompt serving.
- External LLM proof CLI: `crowdtensor llm-infer` emits `llm_inference_cli_v1`, wraps `scripts/external_llm_evidence_pack.py`, and writes `external_llm_evidence_v1` JSON/Markdown with the read-only `external_llm_infer` route, adapter kind, model id, request/completion count, output chars, throughput, diagnosis codes, read-only status, and redaction status. The default path is deterministic mock; command and OpenAI-compatible HTTP runtimes are explicit operator-owned adapters. This is not public arbitrary prompt serving.
- Safe artifact cleanup: `crowdtensor clean-artifacts` emits `cleanup_report_v1`, defaults to dry-run, removes generated `__pycache__` / `.pyc` caches and old CrowdTensor temp directories only with `--apply`, keeps reports unless `--include-reports` is used, and does not delete state, source files, release artifacts, or private env material.
- Remote demo operator CLI: `crowdtensor remote-runbook` emits `remote_runbook_cli_v1` and wraps `scripts/remote_demo_runbook_pack.py`; `crowdtensor remote-acceptance` emits `remote_acceptance_cli_v1`, defaults to `--create-session`, wraps `scripts/remote_demo_acceptance_pack.py`, carries fixed `model_bundle_inference_scenario_v1` scenarios such as `route-baseline`, and applies token redaction to captured command output. These are controlled two-machine helpers, not production Swarm Inference and not P2P routing.
- Remote home-compute demo CLI: `crowdtensor remote-demo prepare` and `crowdtensor remote-demo verify` emit `remote_home_compute_demo_v1` through `scripts/remote_home_compute_demo_pack.py`. The prepare path creates `operator.private.env`, `miner.private.env`, the hashed registry, and the public runbook; the verify path creates a read-only `POST /admin/inference-sessions` task for `model_bundle_infer`, validates the `remote_python_model_bundle_infer` route, and summarizes `remote_compute_evidence_v1`, `remote_demo_observability_v1`, and Support Bundle artifacts. `scripts/remote_home_compute_demo_check.py` validates the local-loopback stand-in. It is a controlled two-machine CPU demo, not production Swarm Inference, not P2P routing, and not GPU pooling.
- Local multi-Miner scenario sweep: `scripts/multi_miner_scenario_sweep.py` and `scripts/multi_miner_scenario_sweep_check.py` emit `multi_miner_scenario_sweep_v1` and `multi_miner_scenario_sweep_observability_v1` by creating three read-only `POST /admin/inference-sessions` tasks for `route-baseline`, `gradient-safety`, and `mixed-prompts`, then running separate registry-backed Python Miner identities through `local_multi_miner_model_bundle_infer`. The check defaults to `--execution-mode concurrent`, starts all Miner processes together, verifies scenario matches, distinct Miner distribution, `lease_summary` one-result-per-task uniqueness, `process_summary`, read-only/redaction/hashed-registry safety, and `multi_miner_concurrent_ready`; `--failure-mode kill-after-claim` terminates one claimed Miner before upload, observes lease timeout requeue, requires a rescue Miner to complete the same `task_id`, and emits `multi_miner_requeue_ready`. Runtime acceptance can opt in with `--include-multi-miner-sweep` and `--include-multi-miner-requeue`. This is local lease-race and requeue evidence, not P2P routing, production throughput scaling, GPU pooling, or production Swarm Inference.
- Demo Manifest tooling: `scripts/demo_manifest_pack.py` and `scripts/demo_manifest_check.py` produce `demo_manifest_v1`, the current latest output artifact for local-loopback handoff. It indexes `runtime_matrix.json`, `remote_compute_evidence_v1`, `external_llm_evidence_v1`, `support_bundle`, and `remote_compute_observability_v1` summaries without claiming GPU, P2P, WebGPU, public prompt serving, or production Swarm Inference readiness.

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

The current model bundle, measurable multi-request model bundle inference, admin-created `POST /admin/inference-sessions` route with `schema=inference_session_request_v1`, `scripts/inference_session_client.py` with `schema=inference_session_client_v1`, optional `external_llm_infer_v1` adapter and `external_llm_evidence_v1` proof path, and micro LM workloads are dependency-free contract rehearsals, not real LLM or GPU throughput benchmarks. `model_bundle_infer` is read-only and exposes only Coordinator-derived capped `request_trace` summaries instead of raw `inference_results`; admin-created sessions must be inspected through `GET /admin/results?task_id=<task_id>&workload_type=model_bundle_infer` and must remain read-only. `inference_session_client.py` is only a thin user-facing client over that API; `session_client_ready` means the existing CPU read-only result was accepted, not that arbitrary prompt serving exists. `external_llm_infer` is read-only, validates `external_llm_results`, records safe `completion_count`, `output_chars`, `adapter_kind`, and `model_id` summaries, supports deterministic mock, command, and OpenAI-compatible HTTP adapters, and must keep raw prompts, raw `output_text`, runtime URLs, and API keys out of public state and public evidence.

## Strategic Route

The near-term goal is to make CrowdTensor credible and useful for open-source users before making large model-scale claims.

Recommended sequence:

1. Keep the Alpha control plane reliable, testable, and well documented.
2. Keep README, ROADMAP, protocol docs, use cases, static site, and project memory synchronized.
3. Keep `scripts/runtime_matrix.py` as the first open-source user diagnostic so contributors can see CPU-only readiness, optional browser support, external LLM adapter configuration, `matched_capabilities`, `missing_capabilities`, target and route `diagnosis_codes`, `operator_action`, top-level `diagnosis_summary`, `hardware_diagnosis_summary`, and the hardware/runtime matrix before running longer smoke tests.
4. Keep `crowdtensor release-ready` as the maintainer-facing publish gate, preserving `release_readiness_v1`, `scripts/release_readiness_pack.py`, `scripts/release_readiness_check.py`, `--allow-dirty`, `git_dirty`, release gate aggregation, `demo_manifest_v1`, and explicit not production boundaries.
5. Keep `scripts/onboarding_gate.py --quick` as the fresh clone install-and-run proof, preserving `onboarding_gate_v1`, clean virtualenv creation, `python -m pip install -e .[dev]`, console script checks, `crowdtensor local-proof`, `crowdtensor home-infer`, `crowdtensor llm-infer --mock`, `crowdtensor release-ready --allow-dirty`, `/tmp` output defaults, and explicit non-production Swarm Inference boundaries.
6. Keep `crowdtensor local-proof` as the shortest user-facing local proof, preserving `local_proof_summary_v1`, Doctor, runtime matrix, CPU-only read-only home-compute demo, Demo Manifest output, and explicit non-production Swarm Inference boundaries.
7. Keep `crowdtensor home-infer` as the shortest shareable local read-only inference proof, preserving `home_inference_cli_v1`, `home_compute_evidence_v1`, `model_bundle_infer`, fixed `model_bundle_inference_scenario_v1` scenarios, capped `request_trace`, `diagnosis_codes`, and explicit not production Swarm Inference boundaries.
8. Keep `crowdtensor llm-infer` as the shortest shareable operator-owned local LLM runtime proof, preserving `llm_inference_cli_v1`, `external_llm_evidence_v1`, fixed claim-time prompts, `external_llm_infer`, adapter summaries, read-only/redaction safety, and explicit not public arbitrary prompt serving boundaries.
9. Keep `crowdtensor clean-artifacts` as the safe maintenance path for repeated agent runs, preserving `cleanup_report_v1`, dry-run default, `--apply`, `--include-reports`, and the rule that cleanup does not delete state or source files.
10. Keep `crowdtensor remote-runbook` and `crowdtensor remote-acceptance` as the operator-facing wrappers for the controlled two-machine path, preserving `remote_runbook_cli_v1`, `remote_acceptance_cli_v1`, fixed scenario propagation, token redaction, default `--create-session`, and explicit not production / not P2P boundaries.
11. Keep `crowdtensor remote-demo prepare` and `crowdtensor remote-demo verify` as the high-level controlled two-machine home-compute demo, preserving `remote_home_compute_demo_v1`, `scripts/remote_home_compute_demo_pack.py`, `scripts/remote_home_compute_demo_check.py`, private `operator.private.env` / `miner.private.env`, `POST /admin/inference-sessions`, `model_bundle_infer`, `remote_python_model_bundle_infer`, `remote_compute_evidence_v1`, `remote_demo_observability_v1`, and explicit not production / not P2P boundaries.
12. Keep expanding `scripts/home_compute_demo.py` as the useful home-compute demo that feels close to Swarm Inference: it should pair `scripts/runtime_matrix.py` `hardware_targets` / `recommended_routes` capability matching and `route_decision` with the read-only multi-request `model_bundle_infer` session and stable `diagnosis_codes` such as `home_compute_ready` and `runtime_matrix_blocked` before larger artifacts or runtime adapters are added.
13. Keep `scripts/home_compute_evidence_pack.py` and `scripts/home_compute_evidence_check.py` as the safe, shareable `home_compute_evidence_v1` layer for public issue reports and demos, preserving `route_decision`, `matched_capabilities`, `diagnosis_codes`, and capped `request_trace` while redacting secret-shaped fields.
14. Keep `scripts/inference_session_client.py` and `scripts/inference_session_client_check.py` as the narrow user-facing client path for a running Coordinator, preserving `inference_session_client_v1`, `session_client_ready`, `POST /admin/inference-sessions`, `task_id` filtering, `model_bundle_infer`, read-only semantics, and `--skip-inference-session-client` acceptance control.
15. Keep `scripts/admin_inference_session_check.py` as the narrow service-shaped API acceptance path for `POST /admin/inference-sessions`, preserving `inference_session_request_v1`, `task_id` filtering, `model_bundle_infer`, read-only semantics, and `--skip-admin-inference-session` acceptance control.
16. Keep `scripts/remote_compute_evidence_pack.py` and `scripts/remote_compute_evidence_check.py` as the safe, shareable `remote_compute_evidence_v1` layer for registry-backed remote-style Python Miner demos, preserving `remote_python_model_bundle_infer`, `remote_compute_observability_v1`, fixed `model_bundle_inference_scenario_v1` metadata and scenario match status, safe metrics, capped `request_trace`, and hashed registry status.
17. Keep `scripts/remote_demo_runbook_pack.py` and `scripts/remote_demo_runbook_check.py` as the safe two-machine `remote_demo_runbook_v1` path for controlled remote demos, preserving `operator.private.env`, `miner.private.env`, hashed registry setup, `model_bundle_infer`, `--scenario-id route-baseline`, and `remote_compute_evidence_pack.py --mode collect`.
18. Keep `scripts/remote_demo_acceptance_pack.py` and `scripts/remote_demo_acceptance_check.py` as the safe two-machine `remote_demo_acceptance_v1` path that can use `--create-session` to call `POST /admin/inference-sessions` with `scenario_id`, wait for the returned `task_id`, verify scenario match, then collect `remote_compute_evidence_v1`, `remote_demo_observability_v1`, `support_bundle`, and stable `diagnosis_codes` such as `coordinator_unreachable`, `observer_auth_failed`, `session_create_failed`, and `artifact_collection_failed`.
19. Keep `scripts/multi_miner_scenario_sweep.py` and `scripts/multi_miner_scenario_sweep_check.py` as the controlled local multi-Miner lease-race and failure-requeue proof, preserving `multi_miner_scenario_sweep_v1`, `multi_miner_scenario_sweep_observability_v1`, three fixed scenarios, distinct Miner identities, `local_multi_miner_model_bundle_infer`, concurrent mode by default in the check, `lease_summary`, `process_summary`, `requeue_summary`, read-only/redaction/hashed-registry safety, `multi_miner_concurrent_ready`, `multi_miner_requeue_ready`, and `--include-multi-miner-sweep` / `--include-multi-miner-requeue` opt-in coverage.
20. Keep `scripts/demo_manifest_pack.py` and `scripts/demo_manifest_check.py` as the latest output artifact entrypoint for local-loopback handoff. The manifest should combine runtime matrix, remote-compute evidence, deterministic mock external LLM evidence, and support bundle summaries while staying CPU-only and safe by default.
21. Keep `external_llm_infer_v1` and `external_llm_evidence_v1` as the narrow optional runtime adapter proof: deterministic `--enable-mock-llm-runtime` / `--mock` for CI, explicit `--llm-runtime-cmd` / `CROWDTENSOR_LLM_RUNTIME_CMD` for operator-owned local experiments, and `--llm-runtime-url` / `CROWDTENSOR_LLM_RUNTIME_URL` for OpenAI-compatible local servers.
22. Add hardware/runtime matrices for CPU, NVIDIA, AMD, Apple Silicon, browser, and remote container paths.
23. Introduce optional GPU/runtime adapters without making the control plane depend on one framework.
24. Expand browser-native participation from WebRTC/Worker probes toward WebGPU/WebAssembly only after tensor transfer and lifecycle limits are measured.
25. Add P2P/NAT routing after useful workloads and operator safety are proven.
26. Treat reputation and incentives as later protocol layers built on result validation and trust history.

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

Home-compute evidence pack for a safe, shareable route/session artifact:

```bash
python3 scripts/home_compute_evidence_pack.py \
  --port 8911 \
  --request-count 4 \
  --json-out /tmp/crowdtensor_home_evidence.json \
  --markdown-out /tmp/crowdtensor_home_evidence.md
```

Remote-compute evidence pack for a safe, shareable registry-backed Miner artifact:

```bash
python3 scripts/remote_compute_evidence_pack.py \
  --port 8912 \
  --request-count 4 \
  --json-out /tmp/crowdtensor_remote_evidence.json \
  --markdown-out /tmp/crowdtensor_remote_evidence.md
```

Safe two-machine remote demo runbook:

```bash
crowdtensor remote-runbook \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --output-dir dist/remote-demo \
  --json
```

Safe two-machine remote demo acceptance pack:

```bash
crowdtensor remote-acceptance \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" \
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \
  --create-session \
  --output-dir dist/remote-demo-acceptance \
  --json
```

Recommended two-machine remote home-compute demo:

```bash
crowdtensor remote-demo prepare \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --output-dir dist/remote-home-compute \
  --json

crowdtensor remote-demo verify \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" \
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \
  --output-dir dist/remote-home-compute \
  --json
```

`crowdtensor remote-demo` emits `remote_home_compute_demo_v1`, wraps `scripts/remote_home_compute_demo_pack.py`, preserves private `operator.private.env` / `miner.private.env` handling, uses `POST /admin/inference-sessions`, and summarizes `remote_compute_evidence_v1` plus `remote_demo_observability_v1` for the `remote_python_model_bundle_infer` route. Validate this path with `scripts/remote_home_compute_demo_check.py`. It remains not production Swarm Inference, not P2P, not GPU pooling, and not public arbitrary prompt serving.

Demo Manifest latest output artifact:

```bash
python3 scripts/demo_manifest_pack.py \
  --output-dir dist/demo-manifest \
  --port 8914 \
  --request-count 4
```

## Maintenance Rule

If future work changes project identity, target users, protocol boundaries, implemented capability, non-capability claims, roadmap priority, validation commands, or release workflow, update this document and `AGENTS.md` in the same change.
