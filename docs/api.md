# API Reference

CrowdTensorD exposes a small HTTP API from the Coordinator. This Alpha API is intended for local and controlled remote demos; fields may grow over time, but the core fields below are treated as the current contract.

All request and response bodies are JSON unless noted. Miner, observer, and admin tokens may be configured as plaintext or `sha256:<digest>` verifiers on the Coordinator. Clients always send the original plaintext token.

## Authentication Headers

- `x-crowdtensor-miner-token`: required for Miner task endpoints when `--miner-token` or `--miner-token-registry` is configured.
- `x-crowdtensor-observer-token`: required for `/state` and `/metrics` when `--observer-token` is configured.
- `x-crowdtensor-admin-token`: required for admin endpoints. Admin endpoints return `403` if no admin token is configured.

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

Returns the append-only event tail. Requires `x-crowdtensor-admin-token`.

`limit` is clamped by the API to the range `0..500`. Lease tokens are redacted.

Response:

```json
{"events": [], "limit": 50}
```

### `GET /admin/results?limit=50&status=any`

Returns a result-level traceability ledger. Requires `x-crowdtensor-admin-token`.

Query parameters:

- `limit`: `0..500`, default `50`
- `status`: `any`, `accepted`, or `rejected`
- `miner_id`: optional exact Miner ID filter
- `workload_type`: optional exact workload filter

Each result row is newest-first by `event_index` and includes safe operator fields:

- `event_index`
- `task_id`
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
- `idempotent`
- `terminal_at`
- `validation`
- `audit`
- `miner_workload_score`

The ledger intentionally avoids raw `lease_token`, idempotency keys or hashes, `result_response`, `local_delta`, and `adapter_delta`.

Response:

```json
{
  "results": [],
  "limit": 50,
  "status": "any",
  "miner_id": "",
  "workload_type": ""
}
```

### `POST /admin/trust-overrides`

Sets or clears a workload-scoped trust override. Requires `x-crowdtensor-admin-token`.

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

### `POST /tasks/claim`

Claims the oldest queued task that matches the Miner capabilities. If Miner auth is configured, pass `x-crowdtensor-miner-token`.

Request:

```json
{
  "miner_id": "miner-1",
  "capabilities": {
    "runtime": "python-cli",
    "backend": "cpu",
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

`diloco_train` also accepts an experimental `compressed_delta` transport. `sign_compressed` is a CPU-only contract check format: Coordinator decodes it to a dense delta, then reuses the normal validation, audit, optimizer, checkpoint, and idempotency path.

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

`signs` may contain only `-1`, `0`, or `1`; `scale` must be finite and non-negative. If multiple delta forms are present, Coordinator prefers `local_delta`, then `pseudo_gradient`, then `compressed_delta`.

`idempotency_key` is optional for compatibility with older Miners, but remote Miners should send a stable unique value per claimed task. When a result with the same `task_id`, `attempt`, `lease_token`, and `idempotency_key` is retried after a lost response, Coordinator returns the original response without applying the update twice. Reusing a different key after the task is already terminal returns `409`.

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

`optimizer` summarizes the accepted outer update, including `contract_version`, `optimizer_type`, `delta_format`, `optimizer_step_before`, `optimizer_step_after`, `delta_norm`, `decoded_delta_norm`, `compression_ratio_estimate`, and `velocity_norm`.

Other current workload result fields:

- `browser_probe`: submit `probe_result`.
- `cpu_lora_mock`: submit `adapter_delta`.
- `micro_transformer_lm`: submit `local_delta`.

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
