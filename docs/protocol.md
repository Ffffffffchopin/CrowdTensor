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
- `micro_llm_sharded_infer`: read-only two-stage tiny Transformer decode proof with activation hashes, decoded-token validation, optional stage-specific Miner capabilities, and optional file-backed `micro_llm_artifact_v1` metadata
- `real_llm_sharded_infer`: optional read-only two-stage Hugging Face tiny GPT proof using `real_llm_sharded_infer_v1`, safe `real_llm_artifact_v1` metadata, stage-specific Miner capabilities, activation hashes, decoded-token validation, and explicit `hf_transformers_cpu` or `hf_transformers_cuda`
- `model_bundle_lm`: model-artifact-shaped contract with identity and version checks
- `model_bundle_infer`: read-only inference-shaped model bundle probe with prediction recomputation and capped `request_trace` summaries
- `external_llm_infer`: optional read-only external LLM runtime adapter contract using `external_llm_infer_v1`
- `browser_probe`: browser Worker compute probe that does not update model state

Each workload defines claim-time input, result payload shape, validation gates, and update behavior. New workloads should keep the network lease/heartbeat protocol unchanged.

`micro_llm_artifact_v1` is the current file-backed tiny model package for `micro_llm_sharded_infer`. It is composed of `manifest.json`, `config.json`, `tokenizer.json`, and `weights.json`, with a canonical artifact hash recorded in Coordinator model state, task metadata, session summaries, validation summaries, and evidence packs. Its tokenizer schema is `char_tokenizer_v1`; prompt text is accepted only through fixed operator-provided prompt lists such as `--prompt-texts arn,ten`, converted to exact-length token ids before task claim. The artifact boundary is dependency-free and CPU-only; it is not a Hugging Face, GGUF, llama.cpp, or large-model artifact format.

`real_llm_sharded_infer_v1` is the narrow optional real-weight split contract. It is advertised only by Miners that pass `--enable-hf-tiny-gpt-runtime` and install the optional `[hf]` dependencies. The default model id is `sshleifer/tiny-gpt2`; Coordinator and Miner flags accept `--hf-model-id`, `--hf-cache-dir`, and `--real-llm-backend`. The Coordinator records a safe `real_llm_artifact_v1` summary containing artifact hash, model id, backend, model type, layer count, hidden size, vocab size, and split index. CPU stage 0 Miners advertise `real_llm_sharded_stage0`, CPU stage 1 Miners advertise `real_llm_sharded_stage1`, and broad CPU Miners may advertise `real_llm_sharded_both` through `--real-llm-stage-role`. CUDA stage tasks require `hf_transformers_cuda` and route only to Miners advertising `real_llm_sharded_cuda_stage0`, `real_llm_sharded_cuda_stage1`, or `real_llm_sharded_cuda_both`; `hf_transformers_cuda` is explicit, the Coordinator may create metadata-only CUDA sessions without local CUDA, and Miner execution still fails closed when `torch.cuda.is_available()` is false. Stage 0 returns only validation-safe activation summaries and an activation payload for the next task; stage 1 validates the generated token against a local full-model baseline and records `baseline_match` plus `decoded_tokens_match`. Public operator artifacts must not expose raw prompt text, hidden states, logits, or raw activation payloads. This contract is read-only optional Hugging Face tiny-model evidence; CUDA support is an optional tiny GPT runtime adapter, not production Swarm Inference, not P2P, not a GPU pooling marketplace, not GGUF/llama.cpp serving, and not large-model serving.

## Session Protocol and P2P-lite Discovery

`session_protocol_v1` is the public product-facing request summary used by `crowdtensor generate` and `crowdtensor public-swarm-product-rc`. It records workload type, backend, stage mode, `prompt_hash`, bounded `max_new_tokens`, route requirements, and safety flags while keeping raw prompt text out of public artifacts. The private Coordinator request is derived separately and may include the prompt only for the local command process that calls `POST /admin/inference-sessions`.

`p2p_lite_peer_v1` is an HTTP-gossip route discovery shape for the current RC. Peers announce `swarm_id`, `peer_id`, `role`, safe URLs, capabilities, stage role, backend, TTL, and `last_seen`. The resolver can discover a Coordinator URL and stage-capable Miners for `real_llm_sharded_infer`, then the normal Coordinator API still owns session creation, task leasing, heartbeats, validation, and result ledgers. P2P-lite never gossips admin/miner/observer tokens, raw prompts, generated text, token ids, activations, registry material, leases, or idempotency keys.

P2P-lite is not libp2p, not a DHT, not NAT traversal, not decentralized security, and not production P2P. It exists to remove hard-coded runbook URLs and to prepare the boundary that a later real P2P daemon can replace.

`external_llm_infer` is advertised only by Miners that opt into a local runtime. `--enable-mock-llm-runtime` enables the deterministic mock path used by CI and `scripts/external_llm_inference_smoke.py`; `--llm-runtime-cmd` or `CROWDTENSOR_LLM_RUNTIME_CMD` enables an operator-provided command that receives `prompt` and `max_tokens` arguments; `--llm-runtime-url` or `CROWDTENSOR_LLM_RUNTIME_URL` enables an OpenAI-compatible chat completions adapter covered by `scripts/external_llm_http_adapter_smoke.py`. Optional HTTP bearer tokens come from `--llm-runtime-api-key` or `CROWDTENSOR_LLM_RUNTIME_API_KEY` and are never advertised in Miner capabilities. Coordinator validates `external_llm_results` by schema, request order, prompt hash, non-empty output, output length, and request count. The workload is read-only and records safe `completion_count`, `output_chars`, `adapter_kind`, and `model_id` summaries.

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
