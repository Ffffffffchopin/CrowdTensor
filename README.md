# CrowdTensor

CrowdTensor is an open-source path toward fault-tolerant AI swarms built from ordinary home compute.

`CrowdTensorD` is the current Alpha daemon/control plane. It validates the V1 mechanics needed before real home GPU aggregation, Swarm Inference, browser compute, and future P2P routing are added: task leasing, heartbeat recovery, checkpoint replay, result validation, replay audit, Miner admission, and CPU-only workload contracts.

## What Works Today

- Run a local Coordinator and Miner loop on a normal CPU-only Linux machine.
- Connect controlled remote Python Miners with token-backed admission and retry behavior.
- Validate timeout recovery, stale result rejection, checkpoint replay, result ledger, and Support Bundle generation.
- Run deterministic tiny workloads shaped like future model contracts: `diloco_train`, `cpu_lora_mock`, `micro_transformer_lm`, `model_bundle_lm`, read-only `model_bundle_infer`, and optional read-only `external_llm_infer`.
- Try browser-native experiments for WebRTC tensor transport, browser Worker compute probes, and a browser Miner bridge.

## What Is Not Ready

This Alpha is not yet:

- a production DePIN network
- a real LLM training or inference platform
- a reward, staking, or payment system
- a complete P2P/NAT traversal network
- a hardened public-internet security model
- a GPU, WebGPU, PyTorch, or Transformers benchmark

The current workloads are intentionally small and deterministic so the runtime can be tested without GPU access or external model dependencies.

## Quickstart

Run a 5-minute local swarm demo with Python 3.11 or newer.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .[dev]
```

For a fresh-clone onboarding check that avoids system pip / PEP 668 issues and validates the documented console commands from a clean virtualenv, run:

```bash
python scripts/onboarding_gate.py --quick --json-out /tmp/crowdtensor_onboarding_gate.json
```

The `onboarding_gate_v1` report creates a temporary venv, runs `python -m pip install -e .[dev]`, checks `crowdtensor --help`, `crowdtensord --help`, and `crowdtensor-miner --help`, then smoke-validates `crowdtensor local-proof`, `crowdtensor home-infer`, `crowdtensor llm-infer --mock`, and `crowdtensor release-ready --allow-dirty`. It is an Alpha onboarding gate, not production Swarm Inference readiness.

The install creates these console commands:

```bash
crowdtensor --help
crowdtensord --help
crowdtensor-miner --help
```

Run the First-run Doctor before starting services:

```bash
python3 scripts/doctor.py --json
```

For maintainer release readiness before pushing or tagging:

```bash
crowdtensor release-ready --json
```

The `crowdtensor/cli.py` entrypoint wraps `scripts/release_readiness_pack.py` and emits `release_readiness_v1` under `dist/release-readiness`. It checks Git metadata, the release gate, security preflight, and `demo_manifest_v1`, then reports blocker diagnosis such as `git_dirty`, `release_gate_failed`, or `demo_manifest_failed`. Dirty worktrees block by default; use `--allow-dirty` only for development smoke checks such as `scripts/release_readiness_check.py --allow-dirty`. This is not production readiness for Swarm Inference; it is an Alpha maintainer gate for the current CPU-only public repository state.

For the shortest one-command local proof, run:

```bash
crowdtensor local-proof --json
```

The `crowdtensor/cli.py` entrypoint writes a `local_proof_summary_v1` report under `dist/local-proof` by running Doctor, the runtime matrix, the CPU-only read-only home-compute demo, and the Demo Manifest path. This is not production Swarm Inference; it is a safe local proof that the current checkout can execute the Alpha CPU control-plane path without claiming real LLM serving, GPU pooling, P2P routing, or WebGPU shards.

To produce the shortest shareable local read-only inference proof:

```bash
crowdtensor home-infer --scenario-id route-baseline --json
```

The `crowdtensor/cli.py` wrapper emits `home_inference_cli_v1` and writes `home_compute_evidence_v1` JSON/Markdown under `dist/home-infer`. It runs the CPU-only `model_bundle_infer` path, summarizes the selected route, fixed `model_bundle_inference_scenario_v1` scenario, safe `request_trace`, `diagnosis_codes`, read-only/redaction status, and artifact paths. Built-in scenarios include `route-baseline`, `gradient-safety`, and `mixed-prompts`; this is not production Swarm Inference, arbitrary prompt serving, or real LLM serving.

To produce a safe proof that CrowdTensor can route fixed prompt work to an operator-owned local LLM runtime:

```bash
crowdtensor llm-infer --mock --json
```

The `llm_inference_cli_v1` wrapper writes `external_llm_evidence_v1` JSON/Markdown under `dist/llm-infer`. It uses the read-only `external_llm_infer` contract with deterministic mock by default, or an explicit `--llm-runtime-cmd` / `--llm-runtime-url` runtime when the operator provides one. Reports include adapter kind, model id, request/completion count, output chars, throughput, and `external_llm_evidence_ready` without exposing raw prompts, `output_text`, runtime URL, API key, lease token, or idempotency material. This is fixed-prompt local runtime evidence, not public arbitrary prompt serving.

Inspect generated caches and temporary artifacts before deleting them:

```bash
crowdtensor clean-artifacts --json
```

Apply the safe cleanup only after reviewing the dry-run report:

```bash
crowdtensor clean-artifacts --apply --json
```

The `cleanup_report_v1` cleanup path removes clearly generated `__pycache__` / `.pyc` caches and old CrowdTensor temp directories. It defaults to dry-run, keeps `/tmp/crowdtensor_*.json` and Markdown reports unless `--include-reports` is passed, and does not delete state, source files, release artifacts, or private env material.

Check what this machine can run:

```bash
python3 scripts/runtime_matrix.py --json
```

The runtime capability matrix reports CPU-only workload readiness, optional browser support, optional external LLM command/HTTP runtime configuration, and a hardware/runtime matrix with `hardware_targets`, `recommended_routes`, `matched_capabilities`, `missing_capabilities`, target and route `diagnosis_codes`, `operator_action`, top-level `diagnosis_summary`, and `hardware_diagnosis_summary`. It does not print token, URL, or API key values.

Run the matrix-guided home-compute demo:

```bash
python3 scripts/home_compute_demo.py --port 8909 --request-count 4
```

This combines the runtime capability matrix with the read-only `model_bundle_infer` path and reports safe latency, throughput, `hardware_profile`, selected capability route, `route_decision`, a Coordinator-derived `request_trace`, read-only, redaction status, and stable `diagnosis_codes` such as `home_compute_ready` and `runtime_matrix_blocked`. It is a CPU-only Swarm Inference-shaped demo, not real LLM serving or GPU pooling.

Build a safe, shareable evidence pack for issue reports or demos:

```bash
python3 scripts/home_compute_evidence_pack.py \
  --port 8911 \
  --request-count 4 \
  --json-out /tmp/crowdtensor_home_evidence.json \
  --markdown-out /tmp/crowdtensor_home_evidence.md
```

The `home_compute_evidence_v1` report wraps the runtime matrix, `route_decision`, `matched_capabilities`, safe metrics, capped `request_trace`, and `diagnosis_codes` rows without exposing token, URL, API key, lease, idempotency, weight, or delta-shaped fields. CI validates this with `scripts/home_compute_evidence_check.py`, and runtime acceptance can skip it with `--skip-home-compute-evidence`.

Build a safe, shareable remote-compute evidence pack:

```bash
python3 scripts/remote_compute_evidence_pack.py \
  --port 8912 \
  --request-count 4 \
  --json-out /tmp/crowdtensor_remote_evidence.json \
  --markdown-out /tmp/crowdtensor_remote_evidence.md
```

The `remote_compute_evidence_v1` report runs a registry-backed remote-style Python Miner through the read-only `model_bundle_infer` path, records `remote_python_model_bundle_infer`, route capabilities, safe latency/throughput, capped `request_trace` rows, and `remote_compute_observability_v1`, and verifies the invite registry stores only a hashed token. CI validates this with `scripts/remote_compute_evidence_check.py`; runtime acceptance can opt in with `--include-remote-evidence`.

Run a controlled local multi-Miner scenario sweep:

```bash
python3 scripts/multi_miner_scenario_sweep_check.py \
  --port 8916 \
  --execution-mode concurrent \
  --scenario-ids route-baseline,gradient-safety,mixed-prompts
```

The `multi_miner_scenario_sweep_v1` report creates three read-only `POST /admin/inference-sessions` tasks, starts registry-backed Python Miner identities concurrently through the CPU-only `model_bundle_infer` route `local_multi_miner_model_bundle_infer`, verifies each fixed `model_bundle_inference_scenario_v1` scenario match, records `multi_miner_scenario_sweep_observability_v1`, checks read-only/redaction/hashed-registry safety, confirms one accepted ledger row per task, and emits `multi_miner_concurrent_ready` when all expected miners are seen. Add `--failure-mode kill-after-claim` to terminate one Miner after claim, wait for lease timeout requeue, and require a rescue Miner to finish the requeued task with `multi_miner_requeue_ready`. This is local controlled lease-race and requeue evidence, not P2P routing, production throughput scaling, GPU pooling, or production Swarm Inference. Runtime acceptance can opt in with `--include-multi-miner-sweep` for the happy path and `--include-multi-miner-requeue` for the failure path.

Build the local-loopback Demo Manifest as the latest output artifact:

```bash
python3 scripts/demo_manifest_pack.py \
  --output-dir dist/demo-manifest \
  --port 8914 \
  --request-count 4
```

The `demo_manifest_v1` artifact indexes `runtime_matrix.json`, `remote_compute_evidence_v1`, `external_llm_evidence_v1`, `support_bundle`, and `remote_compute_observability_v1` summaries in one safe JSON/Markdown pair. It is the recommended handoff artifact for showing what this checkout can run today. The external LLM entry uses deterministic mock evidence by default and does not expose raw prompts, `output_text`, runtime URL, or API key. CI validates the path with `scripts/demo_manifest_check.py`.

Build a safe two-machine remote demo runbook:

```bash
crowdtensor remote-runbook \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --scenario-id route-baseline \
  --output-dir dist/remote-demo \
  --json
```

The `crowdtensor/cli.py` wrapper emits `remote_runbook_cli_v1` and delegates to `scripts/remote_demo_runbook_pack.py`. The underlying `remote_demo_runbook_v1` artifact prepares a registry-backed Coordinator/Miner demo for `model_bundle_infer`: it writes `operator.private.env` and `miner.private.env` with `0600` permissions, stores only hashed Miner token verifiers in the registry, and keeps the public JSON/Markdown free of plaintext tokens. The generated commands include security preflight, `crowdtensord --task-lane python-cli:cpu:1:model_bundle_infer`, `crowdtensor-miner`, and `remote_compute_evidence_pack.py --mode collect --scenario-id route-baseline`. The remote path uses the same fixed `model_bundle_inference_scenario_v1` IDs as `crowdtensor home-infer`. CI validates this with `scripts/remote_demo_runbook_check.py`.

After the Coordinator and remote Miner are running, collect the safe two-machine acceptance pack:

```bash
crowdtensor remote-acceptance \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" \
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \
  --create-session \
  --scenario-id route-baseline \
  --output-dir dist/remote-demo-acceptance \
  --json
```

The `crowdtensor/cli.py` wrapper emits `remote_acceptance_cli_v1`, applies token redaction to stdout/stderr tails, and delegates to `scripts/remote_demo_acceptance_pack.py`. The recommended controlled path uses `--create-session` to call `POST /admin/inference-sessions`, queue a read-only `model_bundle_infer` task for the selected `model_bundle_inference_scenario_v1`, and wait for the returned `task_id` through the admin result ledger. The `remote_demo_acceptance_v1` report then writes `remote_compute_evidence_v1`, `support_bundle`, `remote_demo_observability_v1`, scenario match status, and a top-level JSON/Markdown summary. It also emits stable `diagnosis_codes` for operator triage, including `coordinator_unreachable`, `observer_auth_failed`, `admin_auth_failed`, `session_create_failed`, `miner_not_seen`, `task_lane_missing`, `workload_not_advertised`, `no_accepted_result`, `validation_failed`, `request_count_mismatch`, `artifact_collection_failed`, and `acceptance_ready`. This is not production Swarm Inference and not P2P routing. CI validates the local stand-in with `scripts/remote_demo_acceptance_check.py`.

Request one read-only session from an already running Coordinator:

```bash
python3 scripts/inference_session_client.py \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \
  --request-count 4 \
  --json
```

The `inference_session_client_v1` report is the narrow user-facing client for the existing `POST /admin/inference-sessions` API. It creates a CPU `model_bundle_infer` session, waits for the returned `task_id` in the admin result ledger, and emits safe latency, throughput, validation, and `session_client_ready` diagnostics. It does not accept arbitrary prompts, expose raw `inference_results`, or claim production LLM serving. Runtime acceptance covers it with `scripts/inference_session_client_check.py` and `--skip-inference-session-client`.

For optional remote and browser checks:

```bash
python3 scripts/doctor.py --remote-demo --browser --json
```

Start the Coordinator:

```bash
crowdtensord --host 127.0.0.1 --port 8787 --state-dir state
```

In another shell, run one Miner task:

```bash
crowdtensor-miner \
  --coordinator http://127.0.0.1:8787 \
  --miner-id local-1 \
  --once
```

The Miner claims a task, runs a dependency-free local training loop, uploads a DiLoCo-style delta, and exits with a JSON summary.

For the full walkthrough, see [docs/quickstart.md](docs/quickstart.md).
For user scenarios and hardware status, see [docs/use-cases.md](docs/use-cases.md).
For the current protocol boundary, see [docs/protocol.md](docs/protocol.md).
For endpoint-level integration details, see [docs/api.md](docs/api.md).
For controlled remote Miner setup, see [docs/remote-miner.md](docs/remote-miner.md).
For the project roadmap, see [ROADMAP.md](ROADMAP.md).
For durable project memory and future-agent context, see [AGENTS.md](AGENTS.md) and [docs/project-memory.md](docs/project-memory.md).
For release history and maintainer release flow, see [CHANGELOG.md](CHANGELOG.md) and [docs/release.md](docs/release.md).

## Docker Compose

Run the local demo stack:

```bash
docker compose up --build coordinator miner
```

Check the Coordinator:

```bash
curl http://127.0.0.1:8787/health
curl http://127.0.0.1:8787/version
curl http://127.0.0.1:8787/ready
curl -H 'x-crowdtensor-observer-token: local-observer' \
  http://127.0.0.1:8787/metrics
```

The Compose file uses local demo tokens by default. Copy `.env.example` to `.env` to override them.

## Core Capabilities

- **Coordinator / Miner loop**: task claim, heartbeat, result submission, and bounded long-running Miner sessions.
- **Fault tolerance**: lease timeout requeue, stale result rejection, checkpoint recovery, and append-only event replay.
- **Runtime contracts**: deterministic CPU-only `diloco_train`, `cpu_lora_mock`, `micro_transformer_lm`, `model_bundle_lm`, `model_bundle_infer`, and optional `external_llm_infer` workloads.
- **Runtime capability matrix**: `scripts/runtime_matrix.py` and `scripts/runtime_matrix_check.py` summarize local `hardware_profile`, CPU-only baseline readiness, optional browser support, and `CROWDTENSOR_LLM_RUNTIME_URL` adapter configuration.
- **Validation**: finite-value checks, shape checks, norm/loss gates, and optional deterministic replay audit.
- **Trust controls**: workload-scoped Miner scoring, quarantine, admin trust overrides, and redacted event tails.
- **Result traceability**: admin result ledger for accepted/rejected outcomes, validation, audit, model impact, and Miner score summaries.
- **Admission controls**: shared Miner token, observer token, admin token, per-Miner token registry, and hashed token configuration.
- **Remote Miner resilience**: startup `/ready` preflight, bounded retry for transient claim/heartbeat/result failures, result `idempotency_key`, and retry counters.
- **Browser experiments**: WebRTC tensor tunnel, browser Worker compute probe, and browser Miner bridge.
- **Acceptance pack**: repeatable smoke suite for runtime behavior and operator controls.

## Runtime Acceptance

Run the Alpha release gate first to verify package metadata, docs links, Docker/Compose shape, and CI wiring:

```bash
python3 scripts/release_gate.py --json
```

Run the offline security preflight before a controlled remote demo:

```bash
python3 scripts/security_preflight.py \
  --host 0.0.0.0 \
  --miner-token-registry state/miner_registry.json \
  --observer-token sha256:OBSERVER_DIGEST \
  --admin-token sha256:ADMIN_DIGEST \
  --json
```

Run the non-browser V1 acceptance pack from a normal Linux shell with localhost networking:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8910 \
  --report /tmp/crowdtensor_acceptance.json
```

It runs the core smoke checks sequentially:

- readiness/profile
- runtime capability matrix
- API contract
- chaos recovery
- trust quarantine
- replay audit
- operator control
- micro Transformer LM
- model bundle LM
- result idempotency
- result ledger
- Miner resilience
- Miner auth
- observer auth
- per-Miner registry auth
- hashed token auth
- outer optimizer contract
- compressed error-feedback delta transport
- delta transport negotiation
- read-only model bundle inference
- local inference session demo
- admin-created read-only inference session API
- optional external LLM runtime adapter contract

Browser-native checks are opt-in because they require Playwright and a Chromium-compatible browser:

```bash
python3 scripts/browser_acceptance_pack.py \
  --base-port 9310 \
  --report /tmp/crowdtensor_browser_acceptance.json
```

The browser acceptance pack runs the core browser checks: `webrtc_smoke.py`, `runtime_contract_check.py`, and `browser_miner_smoke.py`. CI uses `--allow-skip` so environments without Playwright or Chromium report a skipped browser pack instead of failing the whole job.

For the broader browser smoke set, use the runtime acceptance pack:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8930 \
  --include-browser \
  --report /tmp/crowdtensor_browser_acceptance.json
```

Generate the release evidence bundle after the acceptance reports exist:

```bash
python3 scripts/release_evidence_pack.py \
  --runtime-report /tmp/crowdtensor_acceptance.json \
  --browser-report /tmp/crowdtensor_browser_acceptance.json \
  --json-out dist/release-evidence.json \
  --markdown-out dist/release-evidence.md
```

The Release Evidence output records the git commit, package metadata, release gate result, security preflight result, and acceptance report summaries. Runtime acceptance summaries preserve safe per-check `summary_json` fields plus top-level `diagnosis_summary` / `diagnosis_by_check` rows, and remote reports preserve safe `observability_summaries` such as `remote_compute_observability_v1` and `remote_demo_observability_v1`, so release artifacts show stable triage and remote-demo observability without raw tokens or tensor payloads. CI uploads `release-evidence.json` and the Markdown companion as build artifacts.

Build a Support Bundle for issues or remote-demo troubleshooting:

```bash
python3 scripts/support_bundle.py \
  --json-out /tmp/crowdtensor_support_bundle.json
```

The Support Bundle includes doctor and release-gate summaries, optional acceptance report summaries, runtime `diagnosis_summary` / `diagnosis_by_check` rows, safe remote `observability_summaries`, and safe online Coordinator summaries when `--coordinator` is provided. It redacts token, lease, idempotency, weight, and delta-shaped fields before writing output.

Some sandboxes block localhost client sockets. In that case, run unit tests inside the sandbox and run the acceptance pack in an unrestricted shell or CI job.

Run only the readiness/profile smoke:

```bash
python3 scripts/readiness_check.py --port 8890
```

Run only the API contract smoke:

```bash
python3 scripts/api_contract_check.py --port 8891
```

Run only the Miner resilience smoke:

```bash
python3 scripts/miner_resilience_check.py --port 8894
```

Run only the result idempotency smoke:

```bash
python3 scripts/result_idempotency_check.py --port 8896
```

Run only the result ledger smoke:

```bash
python3 scripts/result_ledger_check.py --port 8897
```

Run only the opt-in Nesterov outer optimizer smoke:

```bash
python3 scripts/outer_optimizer_check.py --port 8899
```

Run only the sign-compressed error-feedback transport smoke:

```bash
python3 scripts/compressed_error_feedback_check.py --port 8900
```

Run only the delta transport negotiation smoke:

```bash
python3 scripts/delta_transport_negotiation_check.py --port 8901
```

Run only the model bundle LM smoke:

```bash
python3 scripts/model_bundle_smoke.py --port 8902
```

Run only the read-only model bundle inference smoke:

```bash
python3 scripts/model_bundle_inference_smoke.py --port 8903
```

The inference smoke runs a read-only multi-request session by default; use `--request-count N` to change the number of prompts in the task.
It reports safe session metrics such as `elapsed_ms`, `requests_per_second`, `request_count`, accuracy, a capped `request_trace`, and the Python Miner `hardware_profile` so users can inspect the CPU baseline without treating it as a real LLM or GPU benchmark.

Run the user-facing local inference session demo:

```bash
python3 scripts/inference_session_demo.py --port 8904 --request-count 4
```

Use `--json` when you need a machine-readable report for CI or issue reports.

Run the admin-created read-only inference session API check:

```bash
python3 scripts/admin_inference_session_check.py --port 8915 --request-count 4
```

This validates `POST /admin/inference-sessions`, which returns `schema=inference_session_request_v1`, queues a CPU-only `model_bundle_infer` task, and points operators at `GET /admin/results?task_id=<task_id>&workload_type=model_bundle_infer`. The result is read-only and safe for operator inspection: model versions do not advance, raw `inference_results`, lease tokens, and idempotency material stay out of the admin ledger. The runtime acceptance pack includes this check by default and can omit it with `--skip-admin-inference-session`.

Run the matrix-guided home-compute demo:

```bash
python3 scripts/home_compute_demo.py --port 8909 --request-count 4 --json
```

The home-compute demo first checks `scripts/runtime_matrix.py`, selects the CPU-only `model_bundle_infer` workload and `local_cpu_model_bundle_infer` route when available, runs `scripts/inference_session_demo.py`, and emits one report with runtime capability, `route_decision`, session metrics, capped `request_trace` rows, read-only status, redaction status, `hardware_targets`, `recommended_routes`, `matched_capabilities`, `missing_capabilities`, stable `diagnosis_codes` such as `home_compute_ready`, `runtime_matrix_blocked`, `workload_unavailable`, `cpu_route_unavailable`, `session_failed`, `trace_missing`, and recommended next commands. CI validates this path with `scripts/home_compute_demo_check.py`; the runtime acceptance pack includes it by default and can skip it with `--skip-home-compute-demo`.

For a safe, shareable artifact, run `scripts/home_compute_evidence_pack.py --port 8911 --request-count 4 --json-out /tmp/crowdtensor_home_evidence.json --markdown-out /tmp/crowdtensor_home_evidence.md`. The `home_compute_evidence_v1` evidence pack preserves the route, metrics, capped trace, and `diagnosis_codes` while redacting secret-shaped fields; CI validates it with `scripts/home_compute_evidence_check.py`, and the runtime acceptance pack can skip it with `--skip-home-compute-evidence`.

Run only the optional external LLM adapter smoke:

```bash
python3 scripts/external_llm_inference_smoke.py --port 8906 --request-count 3
```

Run the OpenAI-compatible HTTP adapter variant:

```bash
python3 scripts/external_llm_http_adapter_smoke.py --port 8907 --runtime-port 8908
```

The `external_llm_infer` workload uses the `external_llm_infer_v1` schema. It is read-only and validates `external_llm_results` against claim-time prompt hashes before recording safe `request_count`, `completion_count`, `output_chars`, `adapter_kind`, and `model_id` summaries. The smoke path uses `crowdtensor-miner --enable-mock-llm-runtime` for deterministic CI. Operators can opt into a local command adapter with `--llm-runtime-cmd` or `CROWDTENSOR_LLM_RUNTIME_CMD`; the command receives `prompt` and `max_tokens` arguments. Operators can also opt into an OpenAI-compatible chat completions endpoint with `--llm-runtime-url` or `CROWDTENSOR_LLM_RUNTIME_URL`, plus optional `--llm-runtime-api-key` / `CROWDTENSOR_LLM_RUNTIME_API_KEY`. Runtime URLs and API keys are never advertised in Miner capabilities. Raw prompts and `output_text` are kept out of `/state` and admin ledger summaries.

Build a safe external LLM evidence artifact:

```bash
python3 scripts/external_llm_evidence_pack.py \
  --mock \
  --port 8919 \
  --request-count 3 \
  --json-out /tmp/crowdtensor_external_llm_evidence.json \
  --markdown-out /tmp/crowdtensor_external_llm_evidence.md
```

Validate the default mock path with `scripts/external_llm_evidence_check.py`; runtime acceptance includes it by default and can skip it with `--skip-external-llm-evidence`.

Run only the remote Miner invite/join smoke:

```bash
python3 scripts/remote_miner_join_check.py --port 8898
```

Run only the remote-compute evidence smoke:

```bash
python3 scripts/remote_compute_evidence_check.py --port 8912
```

Run only the safe two-machine runbook generator check:

```bash
python3 scripts/remote_demo_runbook_check.py
```

Run only the safe two-machine acceptance check:

```bash
python3 scripts/remote_demo_acceptance_check.py --port 8913
```

Run only the security preflight:

```bash
python3 scripts/security_preflight.py --json
```

## Security Model

CrowdTensorD has local-development admission controls, not a complete public network security model.

Coordinator supports:

- `--miner-token` / `CROWDTENSOR_MINER_TOKEN`
- `--miner-token-registry` / `CROWDTENSOR_MINER_TOKEN_REGISTRY`
- `--observer-token` / `CROWDTENSOR_OBSERVER_TOKEN`
- `--admin-token` / `CROWDTENSOR_ADMIN_TOKEN`

Token config values may be plaintext for local demos or `sha256:<digest>` verifiers for remote demos.

Miner startup checks `/ready` by default. Use `--skip-preflight` only for legacy Coordinators. Transient claim, heartbeat, and idempotent result upload failures are retried with `--max-request-attempts`; summaries include `request_retries`.

See [docs/security.md](docs/security.md) before exposing a Coordinator beyond localhost.

## Documentation

- [Quickstart](docs/quickstart.md)
- [API Reference](docs/api.md)
- [Protocol Boundary](docs/protocol.md)
- [Remote Miner Onboarding](docs/remote-miner.md)
- [Use Cases](docs/use-cases.md)
- [Architecture](docs/architecture.md)
- [Security](docs/security.md)
- [Operations](docs/operations.md)
- [Release Process](docs/release.md)
- [Changelog](CHANGELOG.md)
- [Roadmap](ROADMAP.md)
- [Project Memory](docs/project-memory.md)
- [Agent Instructions](AGENTS.md)
- [Static Site](site/index.html)

## License

Apache-2.0. See [LICENSE](LICENSE).
