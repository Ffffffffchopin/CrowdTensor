# Protocol Boundary

CrowdTensorD Alpha exposes a small, versioned runtime boundary between the Coordinator, Miners, browser experiments, and operator tools. This document summarizes the stable concepts; endpoint details live in [API Reference](api.md).

## Runtime Contract

The current protocol version is `runtime_contract_v1`.

The Coordinator owns:

- task queues and task lanes
- lease tokens and heartbeat deadlines
- checkpoint and event replay
- result validation and replay audit
- workload-specific state updates
- operator state, metrics, result ledger, and trust overrides

Miners own:

- runtime/backend capability advertisement
- local workload execution
- heartbeat delivery while work is in progress
- result payload generation
- retry behavior for transient request failures

## Miner Capability Shape

`POST /tasks/claim` accepts a `capabilities` object. Current compatibility checks use:

- `runtime`
- `backend`
- `protocol_version`
- `supported_workloads`
- `supported_delta_formats`
- optional `hardware_profile` metadata such as OS, CPU count, Python version, and platform

Miners that do not advertise newer optional fields remain compatible with default dense CPU work, but they are not eligible for workloads requiring compressed transports or specific backend features.

## Workload Contracts

Current workload types:

- `diloco_train`: deterministic dense DiLoCo-style update
- `cpu_lora_mock`: dependency-free adapter update mock
- `micro_transformer_lm`: tiny analytic character language-model workload
- `model_bundle_lm`: model-artifact-shaped contract with identity and version checks
- `model_bundle_infer`: read-only inference-shaped model bundle probe with prediction recomputation
- `external_llm_infer`: optional read-only external LLM runtime adapter contract using `external_llm_infer_v1`
- `browser_probe`: browser Worker compute probe that does not update model state

Each workload defines claim-time input, result payload shape, validation gates, and update behavior. New workloads should keep the network lease/heartbeat protocol unchanged.

`external_llm_infer` is advertised only by Miners that opt into a local runtime. `--enable-mock-llm-runtime` enables the deterministic mock path used by CI and `scripts/external_llm_inference_smoke.py`; `--llm-runtime-cmd` or `CROWDTENSOR_LLM_RUNTIME_CMD` enables an operator-provided command that receives `prompt` and `max_tokens` arguments. Coordinator validates `external_llm_results` by schema, request order, prompt hash, non-empty output, output length, and request count. The workload is read-only and records safe `completion_count`, `output_chars`, `adapter_kind`, and `model_id` summaries.

## Delta Transport

Current delta formats:

- `dense_float`: default dense numeric delta
- `sign_compressed`: ternary sign payload with norm metadata
- `sign_compressed_ef`: sign compression with Miner-local error-feedback residuals

Coordinator decodes supported compressed transports before validation and outer update. Unsupported delta formats must fail as compatibility or validation errors instead of silently falling back.

## Result Validation

Accepted results must pass:

- finite-value checks
- shape checks
- norm and loss-spike gates
- workload identity checks when model bundles are used
- optional deterministic replay audit

Rejected results do not update model state. They are recorded in the result ledger and can affect workload-scoped Miner trust.

## Observer and Admin Boundary

Public endpoints are limited to health, version, and readiness.

Observer endpoints expose redacted state and aggregate metrics. Admin endpoints expose event tails, result ledger, and trust overrides. Tokens may be plaintext for local demos or `sha256:` verifiers in Coordinator config.

Raw lease tokens, idempotency material, and tensor deltas should not be exposed through operator-friendly summaries.

## Browser Boundary

The `web/` directory is experimental. Browser code currently validates WebRTC tensor transport, Worker compute, and a browser Miner bridge. It is not yet a WebGPU model-shard runtime.

Browser protocol changes should remain additive until a dedicated browser contract version exists.
