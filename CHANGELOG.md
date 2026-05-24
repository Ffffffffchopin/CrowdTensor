# Changelog

All notable CrowdTensorD Alpha changes are tracked here.

## 0.1.0a0 - Alpha

CrowdTensorD is currently an experimental control plane for fault-tolerant distributed AI workload Miners. This release is intended for local development, controlled remote demos, and implementation review.

### Added

- Coordinator and Miner loop with task claims, heartbeats, lease timeout recovery, checkpoint replay, and result submission.
- Deterministic CPU-only workload contracts, including dense toy training, CPU LoRA mock, micro Transformer LM, model bundle LM, read-only multi-request model bundle inference, and optional read-only `external_llm_infer_v1` adapter validation.
- Runtime validation for finite values, tensor shape, loss/norm gates, deterministic replay audit, and low-frequency outer optimizer behavior.
- Miner admission controls with shared tokens, per-Miner token registry, hashed token verifiers, observer/admin separation, and security preflight checks.
- Operator views for `/health`, `/version`, `/ready`, `/metrics`, redacted `/state`, admin event tails, and admin result ledger.
- Remote Miner demo flow with invite generation, readiness checks, retry counters, result `idempotency_key`, and controlled remote acceptance.
- Browser experiments for WebRTC tensor transfer, browser compute probes, and browser Miner bridge smoke tests.
- Release tooling: `scripts/release_gate.py`, `scripts/runtime_acceptance_pack.py`, `scripts/browser_acceptance_pack.py`, `scripts/release_evidence_pack.py`, and `scripts/support_bundle.py`.
- Release readiness tooling (`crowdtensor release-ready`, `scripts/release_readiness_pack.py`, `scripts/release_readiness_check.py`) with `release_readiness_v1`, Git metadata aggregation, release gate and security preflight summaries, `demo_manifest_v1` handoff validation, dirty-tree blocker diagnosis such as `git_dirty`, and `--allow-dirty` for development/CI smoke checks only.
- Fresh clone onboarding gate (`scripts/onboarding_gate.py --quick`) with `onboarding_gate_v1`, clean temporary virtualenv creation, `python -m pip install -e .[dev]`, console script checks for `crowdtensor --help`, `crowdtensord --help`, and `crowdtensor-miner --help`, plus smoke validation of `crowdtensor local-proof`, `crowdtensor home-infer`, `crowdtensor llm-infer --mock`, and `crowdtensor release-ready --allow-dirty` without claiming production readiness.
- Runtime acceptance diagnosis aggregation: per-check safe `summary_json`, top-level `diagnosis_summary`, and release evidence / Support Bundle `diagnosis_by_check` propagation for operator triage.
- Runtime capability matrix (`scripts/runtime_matrix.py`, `scripts/runtime_matrix_check.py`) for CPU-only readiness, optional browser support, external LLM runtime configuration checks, `hardware_targets`, `recommended_routes`, `matched_capabilities`, `missing_capabilities`, route `diagnosis_codes`, `operator_action`, top-level `diagnosis_summary`, and hardware/runtime matrix diagnostics.
- Hardware target diagnosis in `runtime_matrix.py` with target-level `diagnosis_codes`, `operator_action`, matched/missing capabilities, and `hardware_diagnosis_summary` for CPU, NVIDIA, AMD, Apple, browser, remote container, and external LLM targets.
- Matrix-guided home-compute demo (`scripts/home_compute_demo.py`, `scripts/home_compute_demo_check.py`) that pairs runtime capability discovery and `route_decision` with the read-only `model_bundle_infer` session and stable `diagnosis_codes` such as `home_compute_ready` and `runtime_matrix_blocked`.
- Safe, shareable home-compute evidence pack (`scripts/home_compute_evidence_pack.py`, `scripts/home_compute_evidence_check.py`) with `home_compute_evidence_v1`, `matched_capabilities`, `diagnosis_codes`, capped `request_trace`, and runtime acceptance skip flag `--skip-home-compute-evidence`.
- Safe, shareable remote-compute evidence pack (`scripts/remote_compute_evidence_pack.py`, `scripts/remote_compute_evidence_check.py`) with `remote_compute_evidence_v1`, `remote_compute_observability_v1`, `remote_python_model_bundle_infer`, fixed `model_bundle_inference_scenario_v1` metadata, scenario match status, hashed registry status, capped `request_trace`, and runtime acceptance opt-in flag `--include-remote-evidence`.
- Controlled local multi-Miner scenario sweep (`scripts/multi_miner_scenario_sweep.py`, `scripts/multi_miner_scenario_sweep_check.py`) with `multi_miner_scenario_sweep_v1`, `multi_miner_scenario_sweep_observability_v1`, three fixed `model_bundle_inference_scenario_v1` scenarios, registry-backed Python Miner identities, concurrent lease-race mode, `kill-after-claim` failure-requeue mode, `local_multi_miner_model_bundle_infer`, `lease_summary`, `process_summary`, `requeue_summary`, read-only/redaction/hashed-registry safety, `multi_miner_concurrent_ready`, `multi_miner_requeue_ready`, and runtime acceptance opt-in flags `--include-multi-miner-sweep` / `--include-multi-miner-requeue`.
- Safe two-machine remote demo runbook (`scripts/remote_demo_runbook_pack.py`, `scripts/remote_demo_runbook_check.py`) with `remote_demo_runbook_v1`, `operator.private.env`, `miner.private.env`, hashed registry setup, `model_bundle_infer`, and `remote_compute_evidence_pack.py --mode collect --scenario-id route-baseline`.
- Safe two-machine remote demo acceptance pack (`scripts/remote_demo_acceptance_pack.py`, `scripts/remote_demo_acceptance_check.py`) with `remote_demo_acceptance_v1`, `remote_demo_observability_v1`, `--create-session` active task creation through `POST /admin/inference-sessions` with `scenario_id`, `task_id`-bound wait behavior, scenario match verification, `remote_compute_evidence_v1`, `support_bundle` collection, and stable `diagnosis_codes` such as `coordinator_unreachable`, `observer_auth_failed`, `session_create_failed`, and `artifact_collection_failed`.
- Release Evidence and Support Bundle now preserve safe remote `observability_summaries` so issue reports and release artifacts carry `remote_compute_observability_v1` / `remote_demo_observability_v1` without raw tokens, local artifact paths, or raw state dumps.
- Demo Manifest tooling (`scripts/demo_manifest_pack.py`, `scripts/demo_manifest_check.py`) with `demo_manifest_v1`, the local-loopback latest output artifact that indexes `runtime_matrix.json`, `remote_compute_evidence_v1`, `external_llm_evidence_v1`, `support_bundle`, and `remote_compute_observability_v1` summaries.
- User-facing local inference session demo with safe latency, throughput, capped `request_trace`, read-only, redaction, and Miner hardware profile summaries.
- User-facing inference session client (`scripts/inference_session_client.py`) with `inference_session_client_v1`, `session_client_ready`, `POST /admin/inference-sessions`, `task_id`-bound admin ledger polling, safe validation/throughput summaries, `scripts/inference_session_client_check.py`, and runtime acceptance skip flag `--skip-inference-session-client`.
- One-command local proof CLI (`crowdtensor local-proof` in `crowdtensor/cli.py`) with `local_proof_summary_v1`, Doctor/runtime matrix/home-compute demo/Demo Manifest orchestration, CPU-only read-only boundaries, and explicit not production Swarm Inference positioning.
- Home inference proof CLI (`crowdtensor home-infer`) with `home_inference_cli_v1`, `home_compute_evidence_v1` artifact generation, CPU-only read-only `model_bundle_infer`, fixed `model_bundle_inference_scenario_v1` scenarios (`route-baseline`, `gradient-safety`, `mixed-prompts`), capped `request_trace`, `diagnosis_codes`, and explicit not production Swarm Inference positioning.
- External LLM proof CLI (`crowdtensor llm-infer`) with `llm_inference_cli_v1`, `external_llm_evidence_v1` artifact generation through `scripts/external_llm_evidence_pack.py`, deterministic `--mock`, explicit operator-owned `--llm-runtime-cmd` / `--llm-runtime-url` adapters, adapter/model summaries, request/completion counts, output chars, throughput, `external_llm_evidence_ready`, redaction checks, `scripts/external_llm_evidence_check.py`, and runtime acceptance skip flag `--skip-external-llm-evidence`.
- Safe artifact cleanup CLI (`crowdtensor clean-artifacts`) with `cleanup_report_v1`, dry-run default, explicit `--apply`, optional `--include-reports`, generated `__pycache__` cleanup, temporary artifact cleanup, and guardrails that do not delete state or source files.
- Remote demo operator CLI (`crowdtensor remote-runbook`, `crowdtensor remote-acceptance`) with `remote_runbook_cli_v1`, `remote_acceptance_cli_v1`, default `--create-session`, fixed `--scenario-id route-baseline`, token redaction for captured command output, and explicit not production / not P2P boundaries.
- Admin-created read-only inference session API (`POST /admin/inference-sessions`) with `inference_session_request_v1`, `task_id` result filtering, CPU `model_bundle_infer`, `scripts/admin_inference_session_check.py`, and runtime acceptance skip flag `--skip-admin-inference-session`.
- External LLM adapter smokes (`scripts/external_llm_inference_smoke.py`, `scripts/external_llm_http_adapter_smoke.py`) using `--enable-mock-llm-runtime`, optional `--llm-runtime-cmd` / `CROWDTENSOR_LLM_RUNTIME_CMD`, and OpenAI-compatible `--llm-runtime-url` / `CROWDTENSOR_LLM_RUNTIME_URL` for operator-owned local runtime wrappers.

### Known Limitations

- This is not a production DePIN network, payment system, public-internet security layer, or real LLM training platform.
- Current workloads are intentionally small and CPU-friendly so reliability behavior can be tested without GPU access.
- Browser and remote Miner paths are controlled demos; they require operator-provided transport security when used off localhost.
- P2P discovery, NAT traversal, GPU execution, WebGPU model shards, and real distributed LLM fine-tuning remain future work.

### Verification

Before publishing this alpha, maintainers should run the release flow in [docs/release.md](docs/release.md), including release gate, unit tests, runtime acceptance, release evidence, and Support Bundle generation.
