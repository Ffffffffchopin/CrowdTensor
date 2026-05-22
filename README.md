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
pip install -r requirements.txt
```

Run the First-run Doctor before starting services:

```bash
python3 scripts/doctor.py --json
```

Check what this machine can run:

```bash
python3 scripts/runtime_matrix.py --json
```

The runtime capability matrix reports CPU-only workload readiness, optional browser support, optional external LLM command/HTTP runtime configuration, and a hardware/runtime matrix with `hardware_targets` and `recommended_routes`. It does not print token, URL, or API key values.

Run the matrix-guided home-compute demo:

```bash
python3 scripts/home_compute_demo.py --port 8909 --request-count 4
```

This combines the runtime capability matrix with the read-only `model_bundle_infer` path and reports safe latency, throughput, `hardware_profile`, selected capability route, read-only, and redaction status. It is a CPU-only Swarm Inference-shaped demo, not real LLM serving or GPU pooling.

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

The Release Evidence output records the git commit, package metadata, release gate result, security preflight result, and acceptance report summaries. CI uploads `release-evidence.json` and the Markdown companion as build artifacts.

Build a Support Bundle for issues or remote-demo troubleshooting:

```bash
python3 scripts/support_bundle.py \
  --json-out /tmp/crowdtensor_support_bundle.json
```

The Support Bundle includes doctor and release-gate summaries, optional acceptance report summaries, and safe online Coordinator summaries when `--coordinator` is provided. It redacts token, lease, idempotency, weight, and delta-shaped fields before writing output.

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
It reports safe session metrics such as `elapsed_ms`, `requests_per_second`, `request_count`, accuracy, and the Python Miner `hardware_profile` so users can inspect the CPU baseline without treating it as a real LLM or GPU benchmark.

Run the user-facing local inference session demo:

```bash
python3 scripts/inference_session_demo.py --port 8904 --request-count 4
```

Use `--json` when you need a machine-readable report for CI or issue reports.

Run the matrix-guided home-compute demo:

```bash
python3 scripts/home_compute_demo.py --port 8909 --request-count 4 --json
```

The home-compute demo first checks `scripts/runtime_matrix.py`, selects the CPU-only `model_bundle_infer` workload and `local_cpu_model_bundle_infer` route when available, runs `scripts/inference_session_demo.py`, and emits one report with runtime capability, session metrics, read-only status, redaction status, `hardware_targets`, `recommended_routes`, and recommended next commands. CI validates this path with `scripts/home_compute_demo_check.py`; the runtime acceptance pack includes it by default and can skip it with `--skip-home-compute-demo`.

Run only the optional external LLM adapter smoke:

```bash
python3 scripts/external_llm_inference_smoke.py --port 8906 --request-count 3
```

Run the OpenAI-compatible HTTP adapter variant:

```bash
python3 scripts/external_llm_http_adapter_smoke.py --port 8907 --runtime-port 8908
```

The `external_llm_infer` workload uses the `external_llm_infer_v1` schema. It is read-only and validates `external_llm_results` against claim-time prompt hashes before recording safe `request_count`, `completion_count`, `output_chars`, `adapter_kind`, and `model_id` summaries. The smoke path uses `crowdtensor-miner --enable-mock-llm-runtime` for deterministic CI. Operators can opt into a local command adapter with `--llm-runtime-cmd` or `CROWDTENSOR_LLM_RUNTIME_CMD`; the command receives `prompt` and `max_tokens` arguments. Operators can also opt into an OpenAI-compatible chat completions endpoint with `--llm-runtime-url` or `CROWDTENSOR_LLM_RUNTIME_URL`, plus optional `--llm-runtime-api-key` / `CROWDTENSOR_LLM_RUNTIME_API_KEY`. Runtime URLs and API keys are never advertised in Miner capabilities. Raw prompts and `output_text` are kept out of `/state` and admin ledger summaries.

Run only the remote Miner invite/join smoke:

```bash
python3 scripts/remote_miner_join_check.py --port 8898
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
