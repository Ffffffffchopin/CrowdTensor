# Operations

This document collects common commands for local Alpha operation.

## Start Coordinator

```bash
crowdtensord \
  --host 127.0.0.1 \
  --port 8787 \
  --state-dir state
```

With local tokens:

```bash
crowdtensord \
  --host 127.0.0.1 \
  --port 8787 \
  --state-dir state \
  --miner-token local-miner \
  --observer-token local-observer \
  --admin-token local-admin
```

To run the experimental OpenDiLoCo-inspired Nesterov outer update for new dense model state:

```bash
crowdtensord \
  --host 127.0.0.1 \
  --port 8787 \
  --state-dir state-nesterov \
  --outer-optimizer diloco_nesterov
```

To avoid storing a usable token directly in Coordinator config, generate a hashed token verifier:

```bash
python3 scripts/hash_token.py local-miner
```

Then pass the printed `sha256:` value to `--miner-token`, `--observer-token`, `--admin-token`, or a per-Miner registry entry. Clients still send the original token.

## Run Miner

```bash
CROWDTENSOR_MINER_TOKEN=local-miner crowdtensor-miner \
  --coordinator http://127.0.0.1:8787 \
  --miner-id local-1 \
  --max-tasks 10 \
  --compute-seconds 0.2
```

Useful flags:

- `--once`: process one task and exit
- `--max-tasks N`: stop after N accepted tasks
- `--max-runtime-seconds N`: stop after a wall-clock budget
- `--heartbeat-interval N`: tune heartbeat cadence
- `--skip-preflight`: skip the startup `/ready` compatibility check
- `--max-request-attempts N`: retry transient `/ready`, claim, heartbeat, and idempotent result upload failures
- `--enable-mock-llm-runtime`: advertise `external_llm_infer` with the deterministic mock runtime
- `--llm-runtime-cmd CMD`: advertise `external_llm_infer` with an operator-owned command wrapper; `CROWDTENSOR_LLM_RUNTIME_CMD` is also supported
- `--llm-runtime-url URL`: advertise `external_llm_infer` with an OpenAI-compatible chat completions endpoint; `CROWDTENSOR_LLM_RUNTIME_URL` is also supported
- `--llm-runtime-api-key TOKEN`: optional bearer token for `--llm-runtime-url`; `CROWDTENSOR_LLM_RUNTIME_API_KEY` is also supported and is not advertised in capabilities
- `--idle-sleep N`: sleep between failed or unavailable claims

The Miner summary includes `request_retries` and `preflight_failures` so operators can spot unstable links without parsing stderr.

Result uploads include an `idempotency_key`, so a lost response can be retried without applying the same model update twice.

## Health and Metrics

```bash
curl http://127.0.0.1:8787/health
```

```bash
curl http://127.0.0.1:8787/version
```

```bash
curl http://127.0.0.1:8787/ready
```

```bash
curl -H 'x-crowdtensor-observer-token: local-observer' \
  http://127.0.0.1:8787/state
```

```bash
curl -H 'x-crowdtensor-observer-token: local-observer' \
  http://127.0.0.1:8787/metrics
```

`/metrics` is aggregate-only and avoids lease tokens, task payloads, Miner IDs, and raw Miner metadata.

Admin result ledger:

```bash
curl -H 'x-crowdtensor-admin-token: local-admin' \
  'http://127.0.0.1:8787/admin/results?status=rejected&limit=20'
```

`GET /admin/results` is the safest operator view for result traceability. It includes validation, replay audit, model impact, and Miner workload score summaries, but avoids raw lease tokens, idempotency material, full result responses, and tensor deltas.

## Acceptance Checks

First-run Doctor:

```bash
python3 scripts/doctor.py --json
```

Remote and browser dependency probes:

```bash
python3 scripts/doctor.py --remote-demo --browser --json
```

Alpha release gate:

```bash
python3 scripts/release_gate.py --json
```

Security preflight:

```bash
python3 scripts/security_preflight.py --json
```

Remote demo preflight:

```bash
python3 scripts/security_preflight.py \
  --host 0.0.0.0 \
  --miner-token-registry state/miner_registry.json \
  --observer-token sha256:OBSERVER_DIGEST \
  --admin-token sha256:ADMIN_DIGEST \
  --json
```

Readiness/profile smoke:

```bash
python3 scripts/readiness_check.py --port 8890
```

API contract smoke:

```bash
python3 scripts/api_contract_check.py --port 8891
```

Miner resilience smoke:

```bash
python3 scripts/miner_resilience_check.py --port 8894
```

Result idempotency smoke:

```bash
python3 scripts/result_idempotency_check.py --port 8896
```

Result ledger smoke:

```bash
python3 scripts/result_ledger_check.py --port 8897
```

Remote Miner invite/join smoke:

```bash
python3 scripts/remote_miner_join_check.py --port 8898
```

Unit tests:

```bash
python3 -m unittest discover -s tests -v
```

Default runtime acceptance:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8910 \
  --report /tmp/crowdtensor_acceptance.json
```

Outer optimizer contract smoke:

```bash
python3 scripts/outer_optimizer_check.py --port 8899
```

Compressed error-feedback transport smoke:

```bash
python3 scripts/compressed_error_feedback_check.py --port 8900
```

Delta transport negotiation smoke:

```bash
python3 scripts/delta_transport_negotiation_check.py --port 8901
```

Model bundle LM smoke:

```bash
python3 scripts/model_bundle_smoke.py --port 8902
```

Model bundle inference smoke:

```bash
python3 scripts/model_bundle_inference_smoke.py --port 8903
```

Use `--request-count N` to exercise a multi-request read-only inference session in one task.

User-facing local inference session demo:

```bash
python3 scripts/inference_session_demo.py --port 8904 --request-count 4
```

Use `--json` for automation. The demo reports safe session metrics, a capped Coordinator-derived `request_trace`, read-only status, redaction status, and Miner `hardware_profile`; it is a CPU-only Swarm Inference shaped demo, not a real LLM serving benchmark.

Matrix-guided home-compute demo:

```bash
python3 scripts/home_compute_demo.py --port 8909 --request-count 4 --json
```

This combines runtime capability discovery with the read-only `model_bundle_infer` inference session. It selects the CPU-only workload and `local_cpu_model_bundle_infer` route only when `scripts/runtime_matrix.py` reports it as available, then emits one report with host capability, selected workload, selected route, `route_decision`, session metrics, capped `request_trace` rows, read-only status, redaction status, stable `diagnosis_codes` such as `home_compute_ready`, `runtime_matrix_blocked`, `workload_unavailable`, `cpu_route_unavailable`, `session_failed`, and `trace_missing`, and recommended next commands. `scripts/home_compute_demo_check.py` is included in the default acceptance pack and can be skipped with `--skip-home-compute-demo`.

Safe, shareable home-compute evidence pack:

```bash
python3 scripts/home_compute_evidence_pack.py \
  --port 8911 \
  --request-count 4 \
  --runtime-report /tmp/crowdtensor_acceptance.json \
  --json-out /tmp/crowdtensor_home_evidence.json \
  --markdown-out /tmp/crowdtensor_home_evidence.md
```

The `home_compute_evidence_v1` report is the preferred operator artifact for showing the current home-compute path. It combines the runtime matrix, selected workload, `route_decision`, `matched_capabilities`, capped `request_trace`, `diagnosis_codes`, safety flags, and optional runtime acceptance summary into a safe, shareable JSON/Markdown pair. It redacts token, URL, API key, lease, idempotency, weight, and delta-shaped fields. `scripts/home_compute_evidence_check.py` is included in the default acceptance pack and can be skipped with `--skip-home-compute-evidence`.

Runtime capability matrix:

```bash
python3 scripts/runtime_matrix.py --json
```

The runtime capability matrix reports CPU-only baseline readiness, optional browser support, optional external LLM runtime configuration, and a hardware/runtime matrix through `hardware_targets`, `recommended_routes`, `matched_capabilities`, and `missing_capabilities`. GPU, Apple, AMD, browser, and remote container targets may be detected without being usable runtime adapters. `scripts/runtime_matrix_check.py` is included in the default acceptance pack and can be skipped with `--skip-runtime-matrix`. It notes whether `CROWDTENSOR_LLM_RUNTIME_URL` is configured without printing the URL, token, or API key value.

External LLM adapter contract smoke:

```bash
python3 scripts/external_llm_inference_smoke.py --port 8906 --request-count 3
```

OpenAI-compatible HTTP adapter smoke:

```bash
python3 scripts/external_llm_http_adapter_smoke.py --port 8907 --runtime-port 8908
```

The default smoke uses `crowdtensor-miner --enable-mock-llm-runtime` so it is deterministic and CPU-only. For an operator-provided local runtime, run the Miner with `--llm-runtime-cmd /path/to/wrapper` or set `CROWDTENSOR_LLM_RUNTIME_CMD=/path/to/wrapper`; the wrapper receives `prompt` and `max_tokens` arguments and should print completion text to stdout. For OpenAI-compatible local servers, use `--llm-runtime-url http://127.0.0.1:11434/v1/chat/completions` or `CROWDTENSOR_LLM_RUNTIME_URL=...`, with optional `--llm-runtime-api-key` / `CROWDTENSOR_LLM_RUNTIME_API_KEY`. `external_llm_infer` is read-only, validates `external_llm_infer_v1` prompt hashes and `external_llm_results`, records `request_count`, `completion_count`, `output_chars`, `adapter_kind`, `model_id`, and `requests_per_second`, and keeps raw prompts and `output_text` out of `/state` and admin result ledger summaries.

Remote-style Miner readiness:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8920 \
  --include-remote-miner \
  --miner-token local-miner \
  --observer-token local-observer \
  --report /tmp/crowdtensor_remote_acceptance.json
```

The remote readiness smoke verifies `diloco_train`, `cpu_lora_mock`, `micro_transformer_lm`, and `model_bundle_lm` in one long-running Python Miner session. The default runtime acceptance pack separately verifies the read-only `model_bundle_infer` Swarm Inference shaped probe and the optional `external_llm_infer` adapter contract.

Remote-compute evidence for a read-only inference result:

```bash
python3 scripts/remote_compute_evidence_pack.py \
  --port 8912 \
  --request-count 4 \
  --json-out /tmp/crowdtensor_remote_evidence.json \
  --markdown-out /tmp/crowdtensor_remote_evidence.md
```

The `remote_compute_evidence_v1` report is a safe, shareable proof that a registry-backed remote-style Python Miner completed the read-only `model_bundle_infer` route `remote_python_model_bundle_infer`. It records route capabilities, safe metrics, capped `request_trace`, ledger summary, read-only status, redaction status, and hashed registry status. In a full acceptance run, add `--include-remote-evidence`; this remains opt-in because it starts an additional Coordinator/Miner pair.

Safe two-machine remote runbook:

```bash
python3 scripts/remote_demo_runbook_pack.py \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --output-dir dist/remote-demo
```

The `remote_demo_runbook_v1` pack generates `operator.private.env`, `miner.private.env`, a hashed Miner registry, and public JSON/Markdown commands for a controlled `model_bundle_infer` demo. It keeps plaintext tokens out of the public artifact, includes the `remote_compute_evidence_pack.py --mode collect` command, and is checked by `scripts/remote_demo_runbook_check.py`.

Safe two-machine remote acceptance:

```bash
python3 scripts/remote_demo_acceptance_pack.py \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" \
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \
  --output-dir dist/remote-demo-acceptance
```

The `remote_demo_acceptance_v1` pack waits for the selected remote Miner to complete read-only `model_bundle_infer`, then writes `remote_compute_evidence_v1`, `support_bundle`, and a top-level acceptance JSON/Markdown report. Its `diagnosis_codes` give operators stable next-step triage for `coordinator_unreachable`, `observer_auth_failed`, `admin_auth_failed`, `miner_not_seen`, `task_lane_missing`, `workload_not_advertised`, `no_accepted_result`, `validation_failed`, `request_count_mismatch`, `artifact_collection_failed`, and `acceptance_ready`. `scripts/remote_demo_acceptance_check.py` validates the local stand-in path.

`--miner-token` and `--observer-token` are passed only to checks that explicitly support shared auth env vars. Auth-specific smoke tests keep their own local tokens so they can validate rejection paths deterministically.

Core browser acceptance:

```bash
python3 scripts/browser_acceptance_pack.py \
  --base-port 9310 \
  --report /tmp/crowdtensor_browser_acceptance.json
```

This runs `webrtc_smoke.py`, `runtime_contract_check.py`, and `browser_miner_smoke.py`. Use `--allow-skip` when CI should skip cleanly if Playwright or Chromium is unavailable.

Broader browser acceptance:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8930 \
  --include-browser \
  --report /tmp/crowdtensor_browser_acceptance.json
```

## Release Evidence

After runtime acceptance has produced `/tmp/crowdtensor_acceptance.json`, build a local release evidence bundle:

```bash
python3 scripts/release_evidence_pack.py \
  --runtime-report /tmp/crowdtensor_acceptance.json \
  --browser-report /tmp/crowdtensor_browser_acceptance.json \
  --remote-report /tmp/crowdtensor_remote_acceptance.json \
  --json-out dist/release-evidence.json \
  --markdown-out dist/release-evidence.md
```

`scripts/release_evidence_pack.py` records the current git commit, package metadata, release gate summary, security preflight summary, and acceptance report summaries. The runtime report is required. Browser and remote reports are optional by default; use `--strict-optional` when a release candidate must prove both. CI writes `release-evidence.json` and uploads it as an artifact.

## Support Bundle

For issue reports or remote-demo troubleshooting, generate a Support Bundle:

```bash
python3 scripts/support_bundle.py \
  --json-out /tmp/crowdtensor_support_bundle.json \
  --markdown-out /tmp/crowdtensor_support_bundle.md
```

To include live Coordinator summaries:

```bash
python3 scripts/support_bundle.py \
  --coordinator http://127.0.0.1:8787 \
  --observer-token local-observer \
  --admin-token local-admin \
  --runtime-report /tmp/crowdtensor_acceptance.json \
  --release-evidence /tmp/crowdtensor_release_evidence.json \
  --json-out /tmp/crowdtensor_support_bundle.json
```

`scripts/support_bundle.py` redacts token, lease, idempotency, weight, and delta-shaped fields. Prefer sharing this bundle over raw `state/` files, raw `/state` output, shell history, or token registry files.

## Troubleshooting

**Address already in use**

Change `--port` or `--base-port`. The acceptance pack consumes a range of ports.

**Operation not permitted in sandbox**

Some restricted environments block localhost client sockets. Run unit tests there, then run acceptance checks on a normal shell or CI host.

**401 invalid miner token**

Confirm the Coordinator and Miner use the same `CROWDTENSOR_MINER_TOKEN`, or use the exact per-Miner token from the registry.

For registry-backed remote Miners, generate or rotate the entry with `scripts/create_miner_invite.py` and confirm the remote `--miner-id` matches the registry entry.

**401 invalid observer token**

Pass `x-crowdtensor-observer-token` when reading `/state` or `/metrics`.

**Security preflight fails**

Fix `error` findings before a remote demo. Typical causes are binding to `0.0.0.0` without Miner, observer, or admin tokens, using `local-*` demo tokens on a remote bind, or storing plaintext tokens in `--miner-token-registry`. Use `--strict` when warning-level findings should also block CI.

**503 no compatible queued task available**

The Miner capabilities do not match queued lanes. Check `--task-lane` values and the capabilities sent by the Miner/browser.

**Playwright browser not found**

Install browser dependencies or pass `--browser /path/to/chrome` to the browser smoke scripts.
