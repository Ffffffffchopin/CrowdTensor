# Quickstart

This guide runs a local Coordinator and a Python Miner. It validates the current CrowdTensorD Alpha control-plane loop; it does not train a real LLM.

## Python Environment

Use Python 3.11 or newer.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .[dev]
```

Avoid installing into the system Python. A virtualenv keeps the checkout compatible with distributions that enforce PEP 668 externally managed Python environments. If your environment has no network access, preinstall `setuptools` and `wheel` in the virtualenv or use a base image that already includes them. The package install uses standard Python build metadata.

The install creates two console commands:

```bash
crowdtensor --help
crowdtensord --help
crowdtensor-miner --help
```

To verify the documented fresh-clone path from a clean virtualenv, run:

```bash
python scripts/onboarding_gate.py --quick --json-out /tmp/crowdtensor_onboarding_gate.json
```

The `onboarding_gate_v1` report creates a temporary venv, runs `python -m pip install -e .[dev]`, validates the three console commands above, then runs `crowdtensor local-proof`, `crowdtensor home-infer`, `crowdtensor llm-infer --mock`, and `crowdtensor release-ready --allow-dirty` with reduced request counts. It is an Alpha onboarding gate, not production Swarm Inference, arbitrary prompt serving, GPU pooling, P2P routing, or WebGPU execution.

Run the one-command local proof first when you want the shortest open-source path from checkout to safe artifact:

```bash
crowdtensor local-proof --json
```

The `crowdtensor/cli.py` entrypoint emits `local_proof_summary_v1` and writes artifacts under `dist/local-proof` by chaining Doctor, `runtime_matrix.py`, the CPU-only read-only home-compute demo, and the Demo Manifest. It is not production Swarm Inference, arbitrary prompt serving, GPU pooling, P2P routing, or WebGPU execution.

Run the local read-only inference proof when you want the shortest shareable result trace:

```bash
crowdtensor home-infer --scenario-id route-baseline --json
```

The `crowdtensor/cli.py` wrapper emits `home_inference_cli_v1`, writes `home_compute_evidence_v1` JSON/Markdown under `dist/home-infer`, and summarizes the CPU-only `model_bundle_infer` route, fixed `model_bundle_inference_scenario_v1` scenario, capped `request_trace`, `diagnosis_codes`, and read-only/redaction status. Built-in scenarios include `route-baseline`, `gradient-safety`, and `mixed-prompts`. It is not production Swarm Inference and does not accept arbitrary prompts.

For a safe local LLM runtime proof, run:

```bash
crowdtensor llm-infer --mock --json
```

This emits `llm_inference_cli_v1` and writes `external_llm_evidence_v1` JSON/Markdown under `dist/llm-infer`. The default mock path is deterministic. Operators can pass `--llm-runtime-cmd /path/to/wrapper` or `--llm-runtime-url http://127.0.0.1:11434/v1/chat/completions` when they own the runtime. Reports keep raw prompts, `output_text`, runtime URL, and API key out of public artifacts.

When repeated demos or tests create temporary files, inspect cleanup candidates first:

```bash
crowdtensor clean-artifacts --json
```

Then apply the conservative cleanup:

```bash
crowdtensor clean-artifacts --apply --json
```

The `cleanup_report_v1` report covers generated `__pycache__` / `.pyc` caches and old CrowdTensor temp directories. It defaults to dry-run, keeps reports unless `--include-reports` is used, and does not delete state, source files, release evidence, or private env material.

## First-run Doctor

Run the lightweight diagnostics before starting services:

```bash
python3 scripts/doctor.py --json
```

The First-run Doctor checks Python version, core imports, FastAPI/Uvicorn availability, state directory writability, default port binding, and console entrypoints. It is a quick environment check, not a replacement for runtime acceptance.

For remote-demo and browser dependency probes:

```bash
python3 scripts/doctor.py --remote-demo --browser --json
```

## Run Coordinator

```bash
crowdtensord \
  --host 127.0.0.1 \
  --port 8787 \
  --state-dir state
```

The Coordinator creates and maintains the local checkpoint/event state under `state/`.

## Run One Miner

In another shell:

```bash
crowdtensor-miner \
  --coordinator http://127.0.0.1:8787 \
  --miner-id local-1 \
  --once
```

Expected behavior:

- Miner claims one task.
- Miner sends heartbeats while computing.
- Miner submits a validated result.
- Coordinator updates the tiny model state.
- Miner exits with a JSON summary.

## Token-Protected Local Run

For local demos, plaintext tokens are simplest. For remote demos, generate hashed token config values:

```bash
python3 scripts/hash_token.py local-miner
```

The Coordinator accepts either plaintext values or `sha256:` verifiers. Miners still send the original token.

Start Coordinator:

```bash
crowdtensord \
  --host 127.0.0.1 \
  --port 8787 \
  --state-dir state \
  --miner-token local-miner \
  --observer-token local-observer \
  --admin-token local-admin
```

Run Miner:

```bash
CROWDTENSOR_MINER_TOKEN=local-miner crowdtensor-miner \
  --coordinator http://127.0.0.1:8787 \
  --miner-id local-secure-1 \
  --once
```

Read metrics:

```bash
curl -H 'x-crowdtensor-observer-token: local-observer' \
  http://127.0.0.1:8787/metrics
```

## Runtime Acceptance Pack

Run the release gate to check Alpha packaging and documentation integrity:

```bash
python3 scripts/release_gate.py --json
```

This is a static open-source release check. It does not replace runtime acceptance.

Check local runtime capability readiness:

```bash
python3 scripts/runtime_matrix.py --json
```

The runtime capability matrix is the fastest way to see which CPU-only workloads are ready, whether optional browser checks can run, whether an external LLM HTTP adapter is configured through `CROWDTENSOR_LLM_RUNTIME_URL`, and which hardware/runtime matrix routes are realistic today. It reports `hardware_profile` style host facts, `hardware_targets`, `recommended_routes`, `matched_capabilities`, `missing_capabilities`, target and route `diagnosis_codes`, `operator_action`, top-level `diagnosis_summary`, and `hardware_diagnosis_summary` without printing runtime URL, token, or API key values.

Run the matrix-guided home-compute demo:

```bash
python3 scripts/home_compute_demo.py --port 8909 --request-count 4 --json
```

This runs `scripts/runtime_matrix.py`, selects the CPU-only `model_bundle_infer` workload and `local_cpu_model_bundle_infer` capability route when available, then runs the local inference session demo. The report includes `route_decision`, safe metrics, a capped Coordinator-derived `request_trace`, and stable `diagnosis_codes` such as `home_compute_ready`, `runtime_matrix_blocked`, `workload_unavailable`, `cpu_route_unavailable`, `session_failed`, and `trace_missing`, making it the shortest open-source path from local capability discovery to a measurable Swarm Inference-shaped result without requiring GPU access.

Build a safe, shareable home-compute evidence pack:

```bash
python3 scripts/home_compute_evidence_pack.py \
  --port 8911 \
  --request-count 4 \
  --json-out /tmp/crowdtensor_home_evidence.json \
  --markdown-out /tmp/crowdtensor_home_evidence.md
```

The `home_compute_evidence_v1` artifact wraps the runtime matrix, `route_decision`, `matched_capabilities`, safe metrics, capped `request_trace` rows, `diagnosis_codes`, and runtime acceptance summary if `--runtime-report` is provided. It is intended for demos and issue reports, so it redacts token, URL, API key, lease, idempotency, weight, and delta-shaped fields.

Build the local-loopback Demo Manifest when you want one latest output artifact for a demo, handoff, or issue:

```bash
python3 scripts/demo_manifest_pack.py \
  --output-dir dist/demo-manifest \
  --port 8914 \
  --request-count 4
```

The `demo_manifest_v1` output writes `demo_manifest.json` / `demo_manifest.md` and indexes `runtime_matrix.json`, `remote_compute_evidence_v1`, `external_llm_evidence_v1`, `support_bundle`, and `remote_compute_observability_v1` summaries. It stays local-loopback and CPU-only by default; the external LLM section uses deterministic mock evidence and keeps raw prompts, `output_text`, runtime URL, and API key out of the manifest. Validate the full path with `scripts/demo_manifest_check.py`.

Run the default non-browser smoke suite:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8910 \
  --report /tmp/crowdtensor_acceptance.json
```

The default suite includes `scripts/runtime_matrix_check.py`, `scripts/home_compute_demo_check.py`, `scripts/home_compute_evidence_check.py`, the CPU-only `model_bundle_lm` contract smoke (`scripts/model_bundle_smoke.py`), read-only multi-request `model_bundle_infer` smoke (`scripts/model_bundle_inference_smoke.py`), user-facing inference session demo (`scripts/inference_session_demo.py`), admin-created read-only inference session API check (`scripts/admin_inference_session_check.py`), optional external LLM mock/command adapter smoke (`scripts/external_llm_inference_smoke.py`), OpenAI-compatible HTTP adapter smoke (`scripts/external_llm_http_adapter_smoke.py`), and safe external LLM evidence check (`scripts/external_llm_evidence_check.py`) alongside dense, adapter, micro LM, auth, audit, and operator checks. Use `--skip-runtime-matrix`, `--skip-home-compute-demo`, `--skip-home-compute-evidence`, `--skip-admin-inference-session`, `--skip-external-llm-inference`, `--skip-external-llm-http-adapter`, or `--skip-external-llm-evidence` if you need to omit those adapter checks.

Run only the local inference session demo:

```bash
python3 scripts/inference_session_demo.py --port 8904 --request-count 4
```

Add `--json` for a machine-readable report with `request_count`, `accuracy`, `elapsed_ms`, `requests_per_second`, `request_trace`, read-only status, redaction status, and Miner `hardware_profile`.

Request one session from an already running Coordinator:

```bash
python3 scripts/inference_session_client.py \
  --coordinator-url http://127.0.0.1:8787 \
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \
  --request-count 4 \
  --json
```

The `inference_session_client_v1` client calls `POST /admin/inference-sessions`, waits for the returned `task_id` through `GET /admin/results`, and reports safe `model_bundle_infer` validation and throughput with `session_client_ready` when complete. It is read-only, CPU-only, and does not accept arbitrary prompts. Runtime acceptance includes `scripts/inference_session_client_check.py`; use `--skip-inference-session-client` only when omitting this user-facing client check.

Run only the admin-created read-only inference session API check:

```bash
python3 scripts/admin_inference_session_check.py --port 8915 --request-count 4
```

This exercises `POST /admin/inference-sessions`, expects `schema=inference_session_request_v1`, enqueues a CPU `model_bundle_infer` task, and verifies the accepted result through `GET /admin/results?task_id=<task_id>&workload_type=model_bundle_infer`. It is a service-shaped control-plane boundary, not a public chat API or real LLM serving endpoint.

Run only the optional external LLM adapter contract smoke:

```bash
python3 scripts/external_llm_inference_smoke.py --port 8906 --request-count 3
```

Run the OpenAI-compatible HTTP adapter variant:

```bash
python3 scripts/external_llm_http_adapter_smoke.py --port 8907 --runtime-port 8908
```

These smokes exercise `external_llm_infer_v1`, validate `external_llm_results`, and check that the read-only ledger exposes `completion_count`, `output_chars`, and `adapter_kind` without leaking raw prompts or `output_text`. To use a local runtime wrapper instead of the mock, start a Miner with `--llm-runtime-cmd /path/to/wrapper` or `CROWDTENSOR_LLM_RUNTIME_CMD=/path/to/wrapper`; the wrapper receives `prompt` and `max_tokens` arguments. To use an OpenAI-compatible local server, start a Miner with `--llm-runtime-url http://127.0.0.1:11434/v1/chat/completions` or `CROWDTENSOR_LLM_RUNTIME_URL=...`, plus optional `--llm-runtime-api-key` / `CROWDTENSOR_LLM_RUNTIME_API_KEY`.

For a shareable evidence artifact:

```bash
python3 scripts/external_llm_evidence_check.py --port 8919
```

The check drives `scripts/external_llm_evidence_pack.py` through the deterministic mock runtime and verifies `external_llm_evidence_v1`, `external_llm_evidence_ready`, read-only status, and redaction.

Run the same suite with local auth enabled inside checks that support shared auth env vars:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8950 \
  --miner-token local-miner \
  --observer-token local-observer \
  --report /tmp/crowdtensor_auth_acceptance.json
```

## Remote Miner Demo

Use the high-level remote home-compute demo first. It creates the registry, private env files, public commands, and `remote_home_compute_demo_v1` summary without making users manually stitch lower-level scripts together:

```bash
crowdtensor remote-demo prepare \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --output-dir dist/remote-home-compute \
  --json
```

After the generated Coordinator and Miner commands are running:

```bash
. dist/remote-home-compute/operator.private.env
crowdtensor remote-demo verify \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" \
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \
  --output-dir dist/remote-home-compute \
  --json
```

The wrapper uses `scripts/remote_home_compute_demo_pack.py`, validates through `scripts/remote_home_compute_demo_check.py`, creates the read-only `model_bundle_infer` session with `POST /admin/inference-sessions`, and summarizes `remote_python_model_bundle_infer`, `remote_compute_evidence_v1`, and `remote_demo_observability_v1`. It keeps `operator.private.env` and `miner.private.env` private. This is not production Swarm Inference and not P2P routing.

The lower-level safe two-machine runbook is still available:

```bash
crowdtensor remote-runbook \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --output-dir dist/remote-demo \
  --json
```

The `remote_runbook_cli_v1` summary is produced by `crowdtensor/cli.py` and wraps `scripts/remote_demo_runbook_pack.py`. After the Coordinator and remote Miner are running, validate the controlled demo:

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

The `remote_acceptance_cli_v1` summary applies token redaction to captured output, delegates to `scripts/remote_demo_acceptance_pack.py`, and keeps the path bounded to a read-only `model_bundle_infer` demo. This is not production Swarm Inference and not P2P routing.

For lower-level registry work, generate a registry-backed invite and run a Miner on another Linux host or container:

```bash
python3 scripts/create_miner_invite.py \
  --registry state/miner_registry.json \
  --miner-id remote-linux-1 \
  --coordinator-url https://YOUR_COORDINATOR_HOST
```

See [Remote Miner Onboarding](remote-miner.md) for the full controlled remote demo flow and `scripts/remote_miner_join_check.py`.

## Docker Compose

Run the local stack:

```bash
docker compose up --build coordinator miner
```

Check health:

```bash
curl http://127.0.0.1:8787/health
curl http://127.0.0.1:8787/version
curl http://127.0.0.1:8787/ready
```

Check metrics:

```bash
curl -H 'x-crowdtensor-observer-token: local-observer' \
  http://127.0.0.1:8787/metrics
```

The default Compose tokens are for local demos only. Copy `.env.example` to `.env` and change the values before sharing a machine.

## Browser Experiments

Serve the static web directory:

```bash
python3 -m http.server 8765 --directory web
```

Open the WebRTC tensor tunnel:

```text
http://127.0.0.1:8765/index.html?role=receiver&room=demo
http://127.0.0.1:8765/index.html?role=sender&room=demo
```

Run the core browser acceptance pack when Playwright and a browser are available:

```bash
python3 scripts/browser_acceptance_pack.py \
  --base-port 9310 \
  --report /tmp/crowdtensor_browser_acceptance.json
```

It runs `webrtc_smoke.py`, `runtime_contract_check.py`, and `browser_miner_smoke.py`. Use `--allow-skip` in CI-style environments where Playwright or Chromium may be unavailable.

Run the broader browser smoke set:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8930 \
  --include-browser \
  --report /tmp/crowdtensor_browser_acceptance.json
```
