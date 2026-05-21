# Architecture

CrowdTensorD is currently a hub-and-spoke Alpha runtime. It validates the control-plane mechanics needed before introducing a full P2P network or real accelerator workloads.

## Components

**Coordinator**

The Coordinator is a FastAPI process that owns task queues, leases, model/checkpoint state, validation, audit, metrics, and operator controls.

**Python Miner**

`crowdtensor-miner` is a headless worker process. It claims compatible tasks, runs the requested dependency-free workload, sends heartbeats, submits results, and can run once or as a bounded long-lived session.

**Browser Miner and WebRTC Experiments**

The `web/` directory contains browser-native experiments for a WebRTC tensor tunnel, Worker compute probe, and browser Miner bridge. These are CPU/JavaScript baselines, not WebGPU benchmarks.

**State Store**

The state store persists checkpoint and task event data under `--state-dir`. Restart recovery is driven by the persisted model state and append-only task log.

## Task Lifecycle

1. Coordinator keeps a backlog of queued tasks.
2. Miner calls `POST /tasks/claim` with capabilities and metadata.
3. Coordinator leases the oldest compatible task and returns a lease token.
4. Miner sends `POST /tasks/{task_id}/heartbeat` while working.
5. Miner submits `POST /tasks/{task_id}/result`.
6. Coordinator validates the result, applies the workload-specific update, and checkpoints state.
7. Expired leases are requeued; stale results are rejected.

The concrete HTTP contract for public, observer, admin, and Miner endpoints is documented in [API Reference](api.md). The `scripts/api_contract_check.py` smoke keeps that contract tied to the running Coordinator behavior.

## Workload Lanes

Coordinator can maintain separate task lanes with `--task-lane runtime:backend:count[:workload_type]`.

Current workload types:

- `diloco_train`: tiny deterministic DiLoCo-style dense update
- `cpu_lora_mock`: dependency-free adapter update mock
- `micro_transformer_lm`: tiny character language-model workload with analytic CPU backprop
- `model_bundle_lm`: CPU-only model bundle contract with artifact identity, versioning, and replayable deltas
- `model_bundle_infer`: read-only model bundle inference-shaped probe with Coordinator recomputation
- `browser_probe`: deterministic browser Worker compute probe that does not update model state

These workloads validate protocol contracts and recovery behavior. They do not represent real model throughput.

## Outer Optimizer Contract

`diloco_train` now exposes an explicit `outer_optimizer_contract_v1`. The default implementation is `diloco_momentum` over `dense_float` local deltas, preserving the existing CPU-only math while making the outer optimizer state visible in claims, result responses, checkpoints, and the admin result ledger. `--outer-optimizer diloco_nesterov` enables an OpenDiLoCo-inspired Nesterov outer update for new dense model state without changing the Miner result payload.

This keeps the network layer physically separate from tensor math: Miners advertise `supported_delta_formats`, receive an `optimizer_spec`, produce the requested delta format, and the Coordinator applies the contract. Legacy Miners without a delta capability can still claim default `dense_float` work, but they are incompatible with Coordinator-advertised compressed transports. The first compressed transport is `sign_compressed` with `ternary_signs_v1`; Coordinator decodes it to a dense delta before validation and outer update. `sign_compressed_ef` adds a DisTrO-style error-feedback residual loop on the Miner side: the Miner compresses `local_delta + residual`, uploads the sign payload and norm metadata, and only advances its residual buffer after Coordinator acceptance. Future OpenDiLoCo or DisTrO-style optimizers should extend this contract instead of changing task leasing or heartbeat semantics.

## Model Bundle Contract

`model_bundle_lm` is the first dependency-free boundary shaped like a real model artifact. The claim includes `bundle_id`, `bundle_version`, `artifact_hash`, weights, config, and token IDs inside `workload_spec`; the Miner returns `bundle_delta` with the same identity fields plus numeric delta values. Coordinator validates bundle identity, shape, finite values, delta norm, and loss spike before applying a nested bundle update. Dense `global_step` remains unchanged, while `model_bundle.version`, `model_bundle.optimizer_step`, and `model_bundle.artifact_hash` advance.

`model_bundle_infer` reuses the same built-in bundle identity but is intentionally read-only. The claim includes a small `requests` array of prompts, target tokens, and top-k settings plus bundle config and weights; the Miner returns `inference_results`; Coordinator recomputes logits from current bundle state and accepts only matching predictions. It records request count, correct count, accuracy, elapsed time, and requests per second in safe session summaries without changing dense `global_step` or `model_bundle.version`, making it the first measurable Swarm Inference shaped CPU session contract rather than a serving benchmark.

## Validation and Audit

Every training result passes shape, finite-value, norm, and loss-spike checks before it can update state.

With `--replay-audit`, Coordinator also recomputes expected deterministic results for supported workloads from claim-time state. For `sign_compressed`, replay audit recomputes the dense DiLoCo result, applies the same deterministic compression/decode contract, and compares the decoded delta. For `sign_compressed_ef`, replay audit rejects the result with `error_feedback_replay_unsupported` because the residual buffer is intentionally Miner-local state. Mismatches are rejected and feed the normal trust/quarantine ledger.

## Trust and Scheduling

Miner trust is workload-scoped. Accepted results improve a score, rejected results reduce it, and repeated or severe failures quarantine a Miner for that workload. Admin trust overrides can block, allow, or reset automatic behavior.

Capability-aware scheduling lets Miners advertise runtime/backend/protocol support. If no queued task is compatible, claim returns a controlled `503`.

## Current Boundaries

CrowdTensorD does not yet include libp2p, NAT traversal, decentralized identity, reward accounting, hardware attestation, GPU kernels, WebGPU kernels, or real LLM fine-tuning.
