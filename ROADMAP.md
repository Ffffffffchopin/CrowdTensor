# Roadmap

CrowdTensor is aimed at open AI infrastructure for people with ordinary home hardware. The roadmap is intentionally staged so reliability, security, and operator trust land before real model-scale claims.

## Alpha: Control Plane Reliability

Status: current.

- Fault-tolerant Coordinator/Miner loop with task leases, heartbeats, timeout requeue, stale result rejection, and checkpoint recovery.
- Deterministic CPU-only workload contracts for dense, adapter, micro LM, model bundle training, read-only model bundle inference, optional external LLM adapter inference, and browser probe tasks.
- Admission controls, result validation, replay audit, trust quarantine, operator ledger, release gate, release evidence, and Support Bundle.
- Maintainer release readiness gate through `crowdtensor release-ready`, `release_readiness_v1`, `scripts/release_readiness_pack.py`, and `scripts/release_readiness_check.py --allow-dirty`, aggregating Git metadata, release gate, security preflight, and `demo_manifest_v1` while surfacing blockers such as `git_dirty`.
- Fresh clone onboarding gate through `scripts/onboarding_gate.py --quick` and `onboarding_gate_v1`, creating a clean temporary virtualenv, running `python -m pip install -e .[dev]`, checking `crowdtensor --help`, `crowdtensord --help`, `crowdtensor-miner --help`, and smoke-validating `crowdtensor local-proof`, `crowdtensor home-infer`, `crowdtensor llm-infer --mock`, and `crowdtensor release-ready --allow-dirty` without claiming production readiness.
- Runtime capability matrix so new users can inspect CPU-only readiness, optional browser support, and external LLM adapter configuration before running longer checks.
- Matrix-guided home-compute demo that combines local capability discovery with the read-only `model_bundle_infer` inference session.
- Controlled remote Miner demo and browser WebRTC/Worker experiments.

Success signal: a contributor can run the local demo, inspect the protocol boundary, and reproduce acceptance checks without GPU access.

## Beta: Useful Home-Compute Demo

Status: planned.

- Expand `scripts/home_compute_demo.py` around the current read-only multi-request `model_bundle_infer` probe into the first user-facing Swarm Inference shaped workload, starting with small model artifacts, `route_decision`, capped `request_trace` summaries, stable `diagnosis_codes` such as `home_compute_ready` and `runtime_matrix_blocked`, and explicit capability matching.
- Keep `crowdtensor local-proof` as the shortest one-command local proof for new users: the `crowdtensor/cli.py` entrypoint emits `local_proof_summary_v1` by chaining Doctor, runtime matrix, the CPU-only read-only home-compute demo, and Demo Manifest output. It is not production Swarm Inference.
- Keep `crowdtensor release-ready` as the shortest maintainer publish gate: it emits `release_readiness_v1`, runs `scripts/release_readiness_pack.py`, preserves `--allow-dirty` for development smoke only, reports `git_dirty` by default on dirty worktrees, aggregates the release gate and `demo_manifest_v1`, and remains not production readiness.
- Keep `crowdtensor home-infer` as the shortest shareable local read-only inference proof: it emits `home_inference_cli_v1`, writes `home_compute_evidence_v1`, and surfaces `model_bundle_infer`, fixed `model_bundle_inference_scenario_v1` scenarios such as `route-baseline`, capped `request_trace`, `diagnosis_codes`, and read-only/redaction status without claiming production Swarm Inference.
- Keep `crowdtensor llm-infer` as the shortest shareable operator-owned local LLM runtime proof: it emits `llm_inference_cli_v1`, writes `external_llm_evidence_v1`, uses deterministic `--mock` by default, supports explicit `--llm-runtime-cmd` / `--llm-runtime-url` adapters, and reports adapter kind, model id, request/completion count, output chars, throughput, and diagnosis codes without exposing raw prompts, `output_text`, runtime URL, or API key.
- Keep `scripts/home_compute_evidence_pack.py` as the safe, shareable `home_compute_evidence_v1` artifact for demos and issue reports, with `diagnosis_codes`, `scripts/home_compute_evidence_check.py` in acceptance, and `--skip-home-compute-evidence` available for lanes that only need lower-level checks.
- Keep `scripts/remote_compute_evidence_pack.py` as the safe, shareable `remote_compute_evidence_v1` artifact for registry-backed remote-style Miner demos, with `remote_python_model_bundle_infer`, `remote_compute_observability_v1`, fixed `model_bundle_inference_scenario_v1` scenario metadata, scenario match status, and `--include-remote-evidence` acceptance coverage.
- Keep `scripts/remote_demo_runbook_pack.py` as the safe two-machine `remote_demo_runbook_v1` artifact that prepares `operator.private.env`, `miner.private.env`, a hashed registry, a `model_bundle_infer` lane, and a `remote_compute_evidence_pack.py --mode collect --scenario-id route-baseline` command.
- Keep `scripts/remote_demo_acceptance_pack.py` as the safe two-machine `remote_demo_acceptance_v1` artifact that can use `--create-session` to call `POST /admin/inference-sessions` with `scenario_id`, wait for the returned `task_id`, verify scenario match, and collect `remote_compute_evidence_v1`, `remote_demo_observability_v1`, and `support_bundle`, with stable `diagnosis_codes` such as `coordinator_unreachable`, `observer_auth_failed`, `session_create_failed`, and `artifact_collection_failed`.
- Keep `crowdtensor remote-runbook` and `crowdtensor remote-acceptance` as the operator-facing wrappers for those controlled two-machine artifacts, preserving `remote_runbook_cli_v1`, `remote_acceptance_cli_v1`, fixed scenario propagation, token redaction, default `--create-session`, and explicit not production / not P2P boundaries.
- Keep `scripts/multi_miner_scenario_sweep.py` as the controlled local multi-Miner lease-race and failure-requeue proof: it emits `multi_miner_scenario_sweep_v1`, creates three read-only `POST /admin/inference-sessions` tasks for `route-baseline`, `gradient-safety`, and `mixed-prompts`, starts registry-backed Python Miner identities through `local_multi_miner_model_bundle_infer`, records `lease_summary`, `process_summary`, `requeue_summary`, and `multi_miner_scenario_sweep_observability_v1`, and stays opt-in through `--include-multi-miner-sweep` / `--include-multi-miner-requeue`. Concurrent mode proves local claim uniqueness with `multi_miner_concurrent_ready`; `--failure-mode kill-after-claim` terminates one claimed Miner, observes lease timeout requeue, requires a rescue Miner to complete the same task, and emits `multi_miner_requeue_ready`. This proves local lease-race and timeout rescue behavior without P2P, production throughput scaling, GPU pooling, or production Swarm Inference claims.
- Keep `scripts/demo_manifest_pack.py` as the `demo_manifest_v1` latest output artifact for local-loopback handoff, indexing `runtime_matrix.json`, `remote_compute_evidence_v1`, `external_llm_evidence_v1`, `support_bundle`, and `remote_compute_observability_v1` summaries without claiming production Swarm Inference or public prompt serving.
- Keep the runtime capability matrix visible as the first user diagnostic while `hardware_targets`, `recommended_routes`, `matched_capabilities`, and `missing_capabilities` expand the hardware/runtime matrix toward CPU, NVIDIA, AMD, Apple Silicon, browser, and remote container paths.
- Keep the measurable CPU inference session path visible with safe latency, throughput, accuracy, and Miner hardware profile summaries.
- Keep `scripts/inference_session_demo.py` as the low-friction local demo while richer runtime adapters mature.
- Keep `scripts/inference_session_client.py` as the narrow user-facing client for an already running Coordinator: it emits `inference_session_client_v1`, calls `POST /admin/inference-sessions`, waits for the returned `task_id`, reports `session_client_ready`, and stays read-only without arbitrary prompts.
- Keep `POST /admin/inference-sessions` as the narrow service-shaped read-only API boundary for CPU `model_bundle_infer`, with `inference_session_request_v1`, `task_id` result filtering, `scripts/admin_inference_session_check.py`, and `--skip-admin-inference-session` coverage before any public chat or arbitrary-prompt serving API is claimed.
- Use `external_llm_infer_v1` and `external_llm_evidence_v1` as the first narrow optional adapter proof: deterministic `--enable-mock-llm-runtime` / `--mock` for CI, operator-owned `--llm-runtime-cmd` / `CROWDTENSOR_LLM_RUNTIME_CMD`, OpenAI-compatible `--llm-runtime-url` / `CROWDTENSOR_LLM_RUNTIME_URL` for local servers, safe evidence validation through `scripts/external_llm_evidence_check.py`, and read-only validation of `external_llm_results` before any production serving claims.
- Clear hardware matrix for CPU, NVIDIA, AMD, Apple Silicon, and browser paths.
- Remote Miner onboarding that works through common home-network setups with operator-provided TLS or VPN.
- Keep extending observability from `remote_compute_observability_v1` and `remote_demo_observability_v1` toward broader throughput, latency, availability, and rejected-work dashboards.

Success signal: two ordinary machines can join a controlled swarm demo and produce a useful, verifiable model-serving result.

## P2P and NAT Traversal

Status: planned.

- Replace the central-only discovery path with a P2P daemon layer.
- Add peer identity, capability advertisements, health scoring, and failure-aware routing.
- Evaluate libp2p/WebRTC connectivity paths for home routers and browser participants.

Success signal: Miners can discover and route work without a single hard-coded Coordinator address in controlled test networks.

## Browser-Native Swarm

Status: experimental.

- Move from JavaScript Worker probes toward WebGPU/WebAssembly compute slices.
- Keep browser transport, compute, and safety limits separate from the Python Miner path.
- Measure tensor transfer cost, browser tab lifecycle behavior, and sandbox limits before promising model throughput.

Success signal: browser participants can complete a bounded tensor workload with predictable I/O and failure behavior.

## GPU and Runtime Adapters

Status: planned.

- Add optional adapters for local inference/training runtimes without making the control plane depend on one framework.
- Prefer narrow workload contracts around model artifacts, tensor deltas, and validation over framework-specific APIs.
- Keep CPU-only smoke paths as the baseline acceptance suite.

Success signal: a GPU-capable Miner can advertise backend support and run a real model-shaped workload while CPU-only CI remains deterministic.

## Incentives and Reputation

Status: research.

- Extend workload-scoped trust into longer-lived reputation.
- Explore reward, accounting, and abuse-resistance only after useful workloads and network reliability are proven.
- Treat payment/staking as separate protocol layers, not prerequisites for local and controlled remote demos.

Success signal: the project can explain contribution quality, abuse handling, and operator risk without hiding behind token mechanics.
