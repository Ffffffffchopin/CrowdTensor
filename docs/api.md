# API Reference

CrowdTensorD exposes a small HTTP API from the Coordinator. This Alpha API is intended for local and controlled remote demos; fields may grow over time, but the core fields below are treated as the current contract.

All request and response bodies are JSON unless noted. Miner, observer, legacy admin, and operator-registry tokens may be configured as plaintext or `sha256:<digest>` verifiers on the Coordinator. Clients always send the original plaintext token.

## Authentication Headers

- `x-crowdtensor-miner-token`: required for Miner task endpoints when `--miner-token` or `--miner-token-registry` is configured.
- `x-crowdtensor-observer-token`: required for `/state` and `/metrics` when `--observer-token` is configured.
- `x-crowdtensor-admin-token`: required for admin endpoints. A legacy `--admin-token` has owner-level access. `--operator-token-registry` can instead configure per-operator roles: `owner`, `admin`, `accounting`, and `auditor`, plus optional safe `session_policy` limits for `/admin/inference-sessions`.

## Public Endpoints

### `GET /health`

Lightweight process health check.

Response:

```json
{"ok": true, "service": "crowdtensord-coordinator", "version": "0.1.0a0"}
```

### `GET /version`

Public API and protocol profile.

Response fields:

- `service`
- `version`
- `protocol_version`
- `default_workload_type`
- `api_status`

### `GET /ready`

Public readiness and non-sensitive runtime profile. It is safe for load balancers and container health checks.

Response fields:

- `ok`
- the `/version` fields
- `event_index`
- `task_counts`
- `task_lanes`
- `auth.miner_required`
- `auth.observer_required`
- `auth.admin_configured`
- `auth.miner_registry_configured`
- `auth.operator_registry_configured`
- optional `operator_registry_summary` with operator IDs, enabled flags, labels, and roles; plaintext operator tokens are never exposed

## Observer Endpoints

### `GET /state`

Returns the Coordinator state summary. If observer auth is configured, pass `x-crowdtensor-observer-token`.

Important response fields:

- `model`
- `loss`
- `event_index`
- `task_counts`
- `accepted_results`
- `rejected_results`
- `miner_profiles`
- `miner_workload_scores`
- `miner_trust_overrides`
- `task_lanes`
- `tasks`

Lease tokens in the state summary are redacted as `<redacted>`.

### `GET /metrics`

Returns Prometheus text metrics. If observer auth is configured, pass `x-crowdtensor-observer-token`.

The metrics output is aggregate-only and avoids raw task payloads, lease tokens, and raw Miner metadata.

## Admin Endpoints

### `GET /admin/events?limit=50`

Returns the append-only event tail. Requires `x-crowdtensor-admin-token` with legacy admin, `owner`, `admin`, or `auditor` access.

`limit` is clamped by the API to the range `0..500`. Lease tokens are redacted.

Response:

```json
{"events": [], "limit": 50}
```

### `GET /admin/results?limit=50&status=any`

Returns a result-level traceability ledger. Requires `x-crowdtensor-admin-token` with legacy admin, `owner`, `admin`, or `auditor` access.

Query parameters:

- `limit`: `0..500`, default `50`
- `status`: `any`, `accepted`, or `rejected`
- `task_id`: optional exact task filter
- `miner_id`: optional exact Miner ID filter
- `workload_type`: optional exact workload filter

Each result row is newest-first by `event_index` and includes safe operator fields:

- `event_index`
- `task_id`
- `session_id`
- `created_by_subject`
- `status`
- `accepted`
- `miner_id`
- `workload_type`
- `attempt`
- `base_model_version`
- `result_model_version`
- `staleness`
- `model_updated`
- `adapter_updated`
- `micro_transformer_updated`
- `model_bundle_updated`
- `idempotent`
- `terminal_at`
- `validation`
- `audit`
- `miner_workload_score`

`created_by_subject` is a safe attribution label such as `legacy-admin` or
`operator:<operator_id>` for admin-created inference sessions. It is not a
secret and never contains the plaintext admin/operator token. The ledger
intentionally avoids raw `lease_token`, idempotency keys or hashes,
`result_response`, `local_delta`, `adapter_delta`, and `bundle_delta`.

Response:

```json
{
  "results": [],
  "limit": 50,
  "status": "any",
  "task_id": "",
  "miner_id": "",
  "workload_type": ""
}
```

### `GET /admin/accounting?limit=50&status=any`

Returns a Miner-oriented accounting summary. Requires
`x-crowdtensor-admin-token` with legacy admin, `owner`, `admin`, or `accounting`
access.

Query parameters:

- `limit`: `0..500`, default `50`
- `status`: `any`, `leased`, `accepted`, or `rejected`
- `miner_id`: optional exact Miner ID filter
- `workload_type`: optional exact workload filter
- `session_id`: optional exact session filter
- `created_by_subject`: optional exact admin/operator subject filter

The response includes `miner_accounting_summary_v1` rows, `miner_totals`
grouped by Miner/workload, and `created_by_subject_totals` grouped by
admin-created session subject/workload. Rows expose safe accounting fields such
as Miner ID, workload, accepted/rejected/leased status, stage, backend, model
ID, session ID, admin-created session subject (`created_by_subject`), elapsed
time, and workload-specific work units. The `created_by_subject` query is an
exact match against safe labels such as `legacy-admin` or
`operator:<operator_id>`; it is not a token lookup. `created_by_subject_totals`
only includes rows with a non-empty subject, so ordinary background/training
tasks are not silently charged to an anonymous subject. When the Miner came
from a registry invite, rows include redacted join policy metadata: trust tier,
quota limit, claim-rate limit, claim-rate window, reward-account presence, and
read-only workload. The endpoint does not expose raw prompts, generated text,
token ids, activations, lease material, idempotency keys, plaintext Miner
tokens, plaintext admin/operator tokens, or reward account values.

### `GET /admin/settlement?limit=50`

Returns a draft-only Miner settlement summary. Requires
`x-crowdtensor-admin-token` with legacy admin, `owner`, `admin`, or `accounting`
access.

Query parameters:

- `limit`: `0..500`, default `50`
- `miner_id`: optional exact Miner ID filter
- `workload_type`: optional exact workload filter
- `session_id`: optional exact session filter
- `created_by_subject`: optional exact admin/operator subject filter
- `unit_price_microcredits`: optional non-negative integer price per reward unit

The response schema is `miner_settlement_draft_v1`. It includes accepted-only
`miner_settlement_row_v1` rows, Miner/workload `settlement_totals`,
subject/workload `created_by_subject_totals`, reward units,
`reward_amount_microcredits`, safe admin-created session subject attribution
(`created_by_subject`), and redacted join policy metadata when a Miner came
from a registry invite. The `created_by_subject` query is an exact match
against safe labels, and subject totals only include rows with a non-empty
`created_by_subject`. It is an operator accounting draft only:
`draft_only` is true, `payment_executed` is false, and reward account values are
never exposed.

### `POST /admin/inference-sessions`

Creates one admin-controlled, read-only `model_bundle_infer` task. Requires `x-crowdtensor-admin-token` with legacy admin, `owner`, or `admin` access.

This is the current service-shaped API for asking the Coordinator to enqueue a bounded local CPU inference session and then inspect the accepted result by `task_id`. It is not an OpenAI-compatible chat API, does not accept arbitrary prompts, and does not start real LLM serving. The created task is constrained to `runtime=python-cli`, `backend=cpu`, `schema=inference_session_request_v1`, and a `request_count` between `1` and `8`.

Operators can protect this endpoint with `--inference-session-rate-limit` and
`--inference-session-rate-window-seconds`. The limit is enforced per legacy
admin or per operator-registry subject. When exceeded, the endpoint returns
`429` with `reason=inference_session_rate_limited` and records a safe
`control_plane_blocked` audit event without raw prompts or tokens.

Operator registry entries may also define `session_policy` for admin/owner
operators. The policy can limit `allowed_workloads`, `max_request_count`,
`max_decode_steps`, `max_new_tokens`, `max_active_sessions`,
`max_total_sessions`, and a per-operator `rate_limit` / `rate_window_seconds`.
Active sessions are queued or leased session tasks attributed to that operator
subject; total sessions include all session tasks ever attributed to that
subject in the current Coordinator state. Policy blocks return `403`, `422`, or
`429` with `operator_session_policy_*` reasons and append safe
`control_plane_blocked` events. Policy summaries are safe to expose through
`/ready`; plaintext tokens and raw prompts are never included.

Request:

```json
{
  "request_count": 4,
  "runtime": "python-cli",
  "backend": "cpu"
}
```

Response:

```json
{
  "schema": "inference_session_request_v1",
  "accepted": true,
  "task_id": "task-id",
  "status": "queued",
  "workload_type": "model_bundle_infer",
  "created_by_subject": "operator:owner-a",
  "request_count": 4,
  "task_requirements": {
    "runtime": "python-cli",
    "backend": "cpu",
    "protocol_version": "runtime_contract_v1"
  },
  "result_query": "/admin/results?task_id=task-id&workload_type=model_bundle_infer",
  "claim_requirements": {
    "runtime": "python-cli",
    "backend": "cpu",
    "protocol_version": "runtime_contract_v1"
  }
}
```

After a compatible Miner completes the task, query `GET /admin/results?task_id=<task_id>&workload_type=model_bundle_infer&status=accepted` to retrieve the safe session summary. The returned session, result ledger row, accounting row, settlement draft row, and subject totals carry the same `created_by_subject` label for billing or operator chargeback attribution. The ledger row remains read-only: `model_updated=false`, `model_bundle_updated=false`, raw `inference_results`, lease tokens, plaintext operator/admin tokens, and idempotency material are not exposed.

The supported user-facing client for this API is:

```bash
python3 scripts/inference_session_client.py \
  --coordinator-url http://127.0.0.1:8787 \
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \
  --request-count 4 \
  --json
```

It emits `schema=inference_session_client_v1`, waits for the exact returned `task_id`, and reports `session_client_ready` on success. It is only a thin client over the admin session API; it does not add arbitrary prompts, real LLM serving, GPU execution, P2P routing, or a new wire contract. CI validates the path with `scripts/inference_session_client_check.py`, and runtime acceptance can skip it with `--skip-inference-session-client`.

### `POST /admin/trust-overrides`

Sets or clears a workload-scoped trust override. Requires `x-crowdtensor-admin-token` with legacy admin, `owner`, or `admin` access.

Request:

```json
{
  "miner_id": "miner-1",
  "workload_type": "diloco_train",
  "mode": "block",
  "reason": "operator decision"
}
```

`mode` must be one of:

- `allow`
- `block`
- `none`

Response:

```json
{
  "accepted": true,
  "miner_id": "miner-1",
  "workload_type": "diloco_train",
  "mode": "block",
  "reason": "operator decision",
  "event_index": 1
}
```

## Miner Endpoints

### `POST /tasks/preflight`

Checks Miner admission without claiming work. If Miner auth is configured, pass
`x-crowdtensor-miner-token`. The endpoint reuses the claim-time Miner token,
per-Miner registry, join-policy, quota, and claim-rate checks, but it does not
lease a task, create a claim event, or record a blocked-claim event.

Request:

```json
{
  "miner_id": "miner-1",
  "capabilities": {
    "runtime": "python-cli",
    "backend": "cpu",
    "protocol_version": "runtime_contract_v1",
    "supported_workloads": ["real_llm_sharded_infer"]
  }
}
```

Response:

```json
{
  "schema": "crowdtensor_miner_admission_preflight_v1",
  "ok": true,
  "miner_id": "miner-1",
  "policy_configured": true,
  "reason": "",
  "would_status_code": 200,
  "claim_attempted": false,
  "task_claimed": false,
  "token_public": false
}
```

If the token is invalid, the endpoint returns `401`. If policy would reject the
Miner, `ok=false` and `reason` is the same `join_policy_*` reason claim would
return, such as `join_policy_stage_mismatch`, `join_policy_quota_exhausted`, or
`join_policy_rate_limited`.

### `POST /tasks/claim`

Claims the oldest queued task that matches the Miner capabilities. If Miner auth is configured, pass `x-crowdtensor-miner-token`.

Request:

```json
{
  "miner_id": "miner-1",
  "capabilities": {
    "runtime": "python-cli",
    "backend": "cpu",
    "hardware_profile": {
      "os": "Linux",
      "platform": "Linux",
      "machine": "x86_64",
      "processor": "x86_64",
      "cpu_count": 8,
      "python_version": "3.12.0"
    },
    "protocol_version": "runtime_contract_v1",
    "supported_workloads": ["diloco_train"]
  }
}
```

Minimal response fields:

- `task_id`
- `attempt`
- `lease_token`
- `lease_expires_at`
- `model_version`
- `weights`
- `inner_steps`
- `workload_type`
- `workload_spec`
- `audit_mode`
- `heartbeat_interval`
- `schema_version`
- `optimizer_step`
- `optimizer_spec`
- `task_requirements`
- `training_spec`

For `diloco_train`, `optimizer_spec` describes the outer update contract returned with the claim:

```json
{
  "contract_version": "outer_optimizer_contract_v1",
  "optimizer_type": "diloco_momentum",
  "delta_format": "dense_float",
  "optimizer_step": 0,
  "outer_lr": 0.5,
  "outer_momentum": 0.9,
  "weight_count": 3
}
```

Coordinator defaults to `diloco_momentum` and `dense_float`. Starting it with `--outer-optimizer diloco_nesterov` changes `optimizer_type` in claim and result summaries for new dense model state. Starting it with `--delta-format sign_compressed` or `--delta-format sign_compressed_ef` changes the claim-time `delta_format`; Miners should follow `optimizer_spec.delta_format`, and should advertise compatible `supported_delta_formats` in claim capabilities.

### `POST /tasks/{task_id}/heartbeat`

Extends a live lease. If Miner auth is configured, pass `x-crowdtensor-miner-token`.

Request:

```json
{
  "lease_token": "lease-token-from-claim",
  "attempt": 1,
  "runtime_status": {"phase": "inner_loop"}
}
```

Response:

```json
{"task_id": "task-id", "attempt": 1, "lease_expires_at": 1770000000.0}
```

### `POST /tasks/{task_id}/result`

Completes a live lease and submits workload output. If Miner auth is configured, pass `x-crowdtensor-miner-token`.

For the default `diloco_train` workload, send `local_delta` or `pseudo_gradient` plus optional metrics:

```json
{
  "lease_token": "lease-token-from-claim",
  "attempt": 1,
  "idempotency_key": "stable-random-key-for-this-result",
  "local_delta": [0.1, -0.2, 0.05],
  "metrics": {"elapsed_ms": 10.0}
}
```

`diloco_train` also accepts experimental `compressed_delta` transports when the claimed `optimizer_spec.delta_format` requests them. `sign_compressed` is a CPU-only contract check format: Coordinator decodes it to a dense delta, then reuses the normal validation, audit, optimizer, checkpoint, and idempotency path.

```json
{
  "lease_token": "lease-token-from-claim",
  "attempt": 1,
  "compressed_delta": {
    "format": "sign_compressed",
    "encoding": "ternary_signs_v1",
    "scale": 0.1,
    "signs": [1, -1, 0]
  },
  "metrics": {"delta_format": "sign_compressed"}
}
```

`sign_compressed_ef` uses the same `ternary_signs_v1` payload plus error-feedback metadata from the Miner-local residual buffer:

```json
{
  "lease_token": "lease-token-from-claim",
  "attempt": 1,
  "compressed_delta": {
    "format": "sign_compressed_ef",
    "encoding": "ternary_signs_v1",
    "scale": 0.1,
    "signs": [1, -1, 0],
    "error_feedback": {
      "residual_norm": 0.04,
      "corrected_delta_norm": 0.18
    }
  },
  "metrics": {"delta_format": "sign_compressed_ef"}
}
```

`signs` may contain only `-1`, `0`, or `1`; `scale`, `residual_norm`, and `corrected_delta_norm` must be finite and non-negative. `sign_compressed_ef` is accepted only as a transport contract. Replay audit rejects it with `audit_code=error_feedback_replay_unsupported` because the Coordinator cannot reconstruct a Miner-local residual buffer from claim-time state. If multiple delta forms are present, Coordinator prefers `local_delta`, then `pseudo_gradient`, then `compressed_delta`; if the decoded result format does not match the claim contract, Coordinator rejects it with `delta_format_mismatch`.

`idempotency_key` is optional for compatibility with older Miners, but remote Miners should send a stable unique value per claimed task. When a result with the same `task_id`, `attempt`, `lease_token`, and `idempotency_key` is retried after a lost response, Coordinator returns the original response without applying the update twice. Reusing a different key after the task is already terminal returns `409`.

For `model_bundle_lm`, submit a `bundle_delta` object. It is bound to the claim-time bundle identity:

```json
{
  "lease_token": "lease-token-from-claim",
  "attempt": 1,
  "bundle_delta": {
    "schema_version": "model_bundle_lm_v1",
    "bundle_id": "builtin-char-bundle",
    "base_bundle_version": 0,
    "artifact_hash": "sha256:...",
    "values": [0.01, -0.02]
  },
  "metrics": {"bundle_loss_start": 2.8, "bundle_loss_end": 2.7}
}
```

Coordinator rejects stale bundle versions, artifact-hash mismatches, non-finite values, shape mismatches, excessive delta norm, and excessive bundle loss spikes before applying a nested bundle update.

For `model_bundle_infer`, submit an `inference_results` array for the claim-time `requests` list. The older single `inference_result` object remains accepted for compatibility. Every row is bound to the current bundle identity and is recomputed by Coordinator before acceptance:

```json
{
  "lease_token": "lease-token-from-claim",
  "attempt": 1,
  "inference_results": [
    {
      "schema_version": "model_bundle_infer_v1",
      "request_id": "req-1",
      "bundle_id": "builtin-char-bundle",
      "base_bundle_version": 0,
      "artifact_hash": "sha256:...",
      "prompt_token_ids": [1, 2, 3, 4],
      "target_token_id": 5,
      "predicted_token_id": 5,
      "top_k": [{"token_id": 5, "probability": 0.25}],
      "correct": true
    }
  ],
  "metrics": {
    "request_count": 1,
    "prediction_correct": true,
    "elapsed_ms": 2.5,
    "requests_per_second": 400.0
  }
}
```

Accepted `model_bundle_infer` results are read-only: `model_updated=false`, `model_bundle_updated=false`, dense `global_step` does not change, and `model_bundle.version` does not advance. The admin result ledger keeps prediction summary fields plus `request_count`, `correct_count`, `accuracy`, and a capped `request_trace` with Coordinator-derived prompt text, predicted token, target token, top-k token labels, and correctness. Redacted `/state` avoids raw `inference_result` and `inference_results` payloads while retaining safe aggregate metrics, the derived `request_trace`, and Miner capability profiles.

For `external_llm_infer`, submit `external_llm_results` for the claim-time prompt request list. The single `external_llm_result` object is also accepted for a one-request result. Miners only advertise this workload when started with `--enable-mock-llm-runtime`, an operator-provided `--llm-runtime-cmd` / `CROWDTENSOR_LLM_RUNTIME_CMD`, or an OpenAI-compatible `--llm-runtime-url` / `CROWDTENSOR_LLM_RUNTIME_URL`. HTTP adapter API keys come from `--llm-runtime-api-key` or `CROWDTENSOR_LLM_RUNTIME_API_KEY` and are never part of the capability advertisement.

```json
{
  "lease_token": "lease-token-from-claim",
  "attempt": 1,
  "external_llm_results": [
    {
      "schema_version": "external_llm_infer_v1",
      "request_id": "req-1",
      "prompt_hash": "sha256:...",
      "adapter_kind": "mock",
      "model_id": "mock-external-llm",
      "output_text": "mock completion for: Explain CrowdTensor in one sentence.",
      "output_chars": 61
    }
  ],
  "metrics": {
    "request_count": 1,
    "completion_count": 1,
    "output_chars": 61,
    "adapter_kind": "mock",
    "elapsed_ms": 2.5,
    "requests_per_second": 400.0
  }
}
```

Accepted `external_llm_infer` results are read-only: `model_updated=false`, `model_bundle_updated=false`, and no model checkpoint is changed. Coordinator validates `external_llm_infer_v1`, request order, prompt hashes, non-empty output, output length, and request count. The admin result ledger keeps safe `request_count`, `completion_count`, `output_chars`, `adapter_kind`, and `model_id` summaries while redacting `output_preview`; redacted `/state` avoids raw prompts, `external_llm_result`, `external_llm_results`, and `output_text` payloads. Admin-created `POST /admin/inference-sessions` can also queue this fixed-prompt contract with `{"workload_type": "external_llm_infer", "request_count": N}`; this is the boundary used by `crowdtensor remote-demo verify --workload external-llm` and `remote_external_llm_evidence_v1`, not a public arbitrary prompt API.

Successful `diloco_train` response fields include:

- `accepted`
- `model_version`
- `global_step`
- `optimizer_step`
- `weights`
- `outer_velocity`
- `optimizer`
- `loss`
- `staleness`

`optimizer` summarizes the accepted outer update, including `contract_version`, `optimizer_type`, `delta_format`, `optimizer_step_before`, `optimizer_step_after`, `delta_norm`, `decoded_delta_norm`, `compression_ratio_estimate`, `velocity_norm`, and `outer_update_norm`.

Other current workload result fields:

- `browser_probe`: submit `probe_result`.
- `cpu_lora_mock`: submit `adapter_delta`.
- `micro_transformer_lm`: submit `local_delta`.
- `model_bundle_lm`: submit `bundle_delta`.
- `model_bundle_infer`: submit `inference_results` or legacy `inference_result`.
- `external_llm_infer`: submit `external_llm_results` or legacy one-row `external_llm_result`.

## Common Failure Statuses

- `401`: invalid Miner or observer token.
- `403`: admin token missing, invalid, or not configured.
- `409`: stale, expired, unknown, or already-completed lease.
- `422`: schema validation failure or rejected workload result.
- `503`: no queued task, no compatible task, or Miner is blocked/quarantined for the workload.

## Contract Check

Run the executable API contract smoke before exposing a Coordinator to other clients:

```bash
python3 scripts/api_contract_check.py --port 8891
```

The full acceptance pack runs this check by default.

Run the result idempotency smoke when changing Miner result upload behavior:

```bash
python3 scripts/result_idempotency_check.py --port 8896
```

Run the result ledger smoke when changing operator audit views:

```bash
python3 scripts/result_ledger_check.py --port 8897
```
