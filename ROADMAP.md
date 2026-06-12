# Roadmap

CrowdTensor's north star is open AI infrastructure that can use ordinary
machines for fault-tolerant inference and, later, broader AI workloads.

The project is deliberately staged. Each stage must leave behind runnable
commands, redacted evidence, and clear boundaries before the next layer is
claimed.

## Development Layers

Use three non-overlapping planning layers:

- **Core technology layer:** model execution across devices. It owns large-model
  runtime adapters, layer/pipeline/tensor/expert partitioning, activation and
  KV-cache transport, prefill/decode split, batching, streaming generation,
  heterogeneous placement, correctness checks, and future training/fine-tuning
  mechanics. This is the main technical breakthrough layer.
- **Control layer:** resource governance. It owns Coordinator sessions, task
  leases, heartbeats, result ledgers, admission, role and tenant policy, quotas,
  rate limits, trust/quarantine, P2P provider records, accounting, settlement
  drafts, future incentives, and abuse controls.
- **User-facing layer:** usability and product surface. It owns CLI commands,
  bootstrap, quickstart, Miner join packs, route/tunnel helpers, dashboards,
  docs, support bundles, redacted evidence, onboarding gates, diagnostics, and
  user-visible health, answer, and cost surfaces.

Security, privacy, observability, artifact redaction, tests, and performance are
cross-cutting requirements for every layer. The project should not mistake user
experience polish or control-plane readiness for completion of the core
large-model sharding breakthrough.

## Current Milestone

**Public Real-LLM Swarm Inference Beta + Large-Model Shard Alpha**

The current beta proves a small real Hugging Face GPT model can be split across
two stage Miners behind a Coordinator-backed swarm route.

What is working today:

- Local `p2pd -> serve -> join stage0/stage1 -> generate` flow.
- Product-shaped `public-real-llm-swarm-beta` release gate.
- Stage-aware scheduling and distinct stage Miner validation.
- Read-only split inference with decoded-token baseline checks.
- KV cache evidence and 16-token generation readiness.
- Stage failure requeue proofs for controlled live-style runs.
- Redacted public evidence, support bundles, and cleanup paths.
- Optional CUDA tiny-model stage execution when explicitly enabled.

This is an engineering beta, not production Swarm Inference and not
Hivemind-level large-model serving.

The core technology layer now has Large-Model Shard Alpha plus an Inference RC:

- `crowdtensor large-model-shard` emits `large_model_shard_alpha_v1`.
- `crowdtensor large-model-shard-rc` emits `core_technology_inference_rc_v1`
  and validates with `scripts/large_model_inference_rc_check.py`.
- The runtime adapter target is GGUF / llama.cpp RPC for controlled
  LAN/VPN/local-process operation.
- The default 7B-class RC path is CI-safe fixture and diagnostic evidence and
  keeps `real_runtime_verified=false` unless a runner/supervisor real run or
  `--real-run-report` import proves a short controlled runtime execution.
- The partition planner emits `large_model_partition_manifest_v1` with
  layer-range placement, memory budget checks, controlled endpoint checks,
  latency/bandwidth metadata, and blocker diagnostics.
- The RC planner emits `large_model_partition_manifest_v2` with tensor split,
  KV-cache reservation, prefill/decode memory estimates, single-device fallback,
  multi-worker feasibility, and explicit blocker details.
- Runtime probing emits `large_model_runtime_adapter_probe_v2` with binary
  probes, version digests, local model metadata, RPC endpoint health, command
  validation, sanitized log policy, and controlled LAN/VPN boundaries.
- Device profiling emits `large_model_device_profile_v2` from local probes or
  JSON imports, including CPU/RAM, optional GPU/VRAM, usable memory, backend
  capabilities, latency/bandwidth estimates, and endpoint control checks.
- The workload contract is `large_model_sharded_generate_v1` and includes
  prefill/decode/finalize steps, KV/prefix cache metadata, streaming, bounded
  batch, cancellation, and health-aware route hooks.
- The RC adds `large_model_runner_result_v1`, `large_model_benchmark_v2`,
  `large_model_correctness_summary_v1`, and `large_model_serving_hooks_v1`.
  Benchmarks record TTFT, tokens/s, p50/p95 when available, wall time, memory,
  network bytes/token, cache hit/miss metrics, correctness status, failure
  diagnosis, and single-device fallback vs sharded adapter comparison.
- Public artifacts keep prompts, generated text, token ids, activations,
  KV-cache data, credentials, leases, and idempotency material out of reports.
- Future runtime descriptors exist for vLLM, SGLang, TensorRT-LLM, and
  Petals-like backends, but they are explicit `unsupported_runtime_backend`
  placeholders behind the same adapter interface.

This core path is still not a production serving claim, not public RPC security,
not P2P/NAT traversal, not a GPU marketplace, and not training or fine-tuning.
In environments without GGUF, llama.cpp binaries, reachable RPC workers, or
sufficient hardware, the expected RC outcome is `ok=true` with
`real_runtime_verified=false`, `real_7b_runtime_verified=false`, and concrete
blockers.

## Near Term

**Make the beta easier to run on real machines.**

- Shorten the two-machine setup path.
- Improve join packs for non-expert Miner operators.
- Add first-class operator invite flows for role-scoped generation,
  accounting, and audit users while keeping plaintext tokens out of public
  reports.
- Extend role-scoped operator tokens into tenant/project-scoped policies and
  user-facing operator management.
- Extend private Miner join policy enforcement, claim-rate limits, accounting
  rows, and draft reward summaries into settlement exports and operator-visible
  trust state.
- Make logs, diagnosis codes, and support bundles easier to read.
- Keep printed next commands copyable while surfacing tokens and peer secrets
  as environment requirements rather than report contents.
- Keep private tokens and generated runtime state out of shareable artifacts by
  default.
- Add more operator-facing examples for local, LAN, VPN, and temporary public
  demos.
- Keep remote-Miner onboarding explicit about reachability: a Coordinator needs
  a Miner-facing URL via public HTTPS, a tunnel, VPN, reverse proxy, or trusted
  LAN before Miner hosts outside the local machine can join.

**Harden the public swarm route.**

- Keep the Coordinator as the current authority for sessions and validation.
- Improve discovery reliability and endpoint health checks.
- Continue failure-mode testing for killed or delayed stage Miners.
- Preserve deterministic evidence for every user-facing release gate.

**Improve the real-model path.**

- Keep `sshleifer/tiny-gpt2` as the safe default.
- Add clearer model compatibility checks before a run starts.
- Expand small-model variants only when correctness and artifact safety remain
  easy to verify.
- Keep CPU as the default path; keep CUDA opt-in and fail-closed.
- Use `large-model-shard-rc` as the core-technology transition path for
  7B-class GGUF / llama.cpp RPC runtime probing, planner v2, runner/supervisor,
  benchmark v2, correctness, and serving-hook evidence, without claiming the
  current tiny-model Beta is large-model serving.
- Add real controlled LAN/VPN runner imports before widening the
  `real_runtime_verified` claim beyond fixture diagnostics.

## Mid Term

**Better networking.**

- Move from controlled discovery toward stronger P2P-lite behavior.
- Evaluate relay, NAT traversal, identity, and signed peer records.
- Keep production DHT/libp2p claims gated behind real external evidence.

**Better serving ergonomics.**

- Improve streaming generation output.
- Add better session inspection and cancellation commands.
- Make Miner capability selection and health easier to understand.
- Keep a clean separation between public user commands and maintainer release
  gates.

**Core large-model path.**

- Keep llama.cpp RPC / GGUF as the first concrete adapter while evaluating
  vLLM/SGLang worker integration, TensorRT-LLM, and Petals-like layer workers
  as later runtime backends rather than rebuilding kernels inside the control
  plane.
- Extend the partition planner from layer-range placement into tensor/expert,
  KV-cache placement, prefill/decode split, health-aware routing, and device
  reliability scoring.
- Produce repeatable two-to-four-device real benchmarks for quantized 7B/13B
  models before widening claims to larger open-weight models.
- Compare LAN/trusted-cluster throughput separately from wide-area Petals-style
  availability; optimize each route for its own constraints.

**Browser and edge experiments.**

- Continue WebRTC tensor transport experiments.
- Continue Web Worker and browser Miner probes.
- Evaluate WebGPU only after the native route remains stable.

## Later

These are intentionally later because they require stronger reliability,
security, and economics than the current beta provides:

- Permissionless public Miner admission.
- Large open-weight model sharding.
- Production GPU pooling.
- Training workloads across untrusted home machines.
- Incentives, staking, accounting, and reputation.
- Strong adversarial validation for unknown public Miners.

## Non-Goals For The Current Beta

CrowdTensor is not currently:

- A drop-in replacement for Hivemind, Petals, vLLM, or llama.cpp.
- A production inference API for arbitrary public prompts.
- A finished P2P network with DHT/NAT traversal.
- A large-model serving platform.
- A tokenized compute marketplace.

The current goal is narrower and more useful: prove that small real-model swarm
inference can be routed, recovered, validated, and explained in a way ordinary
contributors can run.
