# Roadmap

CrowdTensor is aimed at open AI infrastructure for people with ordinary home hardware. The roadmap is intentionally staged so reliability, security, and operator trust land before real model-scale claims.

## Alpha: Control Plane Reliability

Status: current.

- Fault-tolerant Coordinator/Miner loop with task leases, heartbeats, timeout requeue, stale result rejection, and checkpoint recovery.
- Deterministic CPU-only workload contracts for dense, adapter, micro LM, model bundle training, read-only model bundle inference, optional external LLM adapter inference, and browser probe tasks.
- Admission controls, result validation, replay audit, trust quarantine, operator ledger, release gate, release evidence, and Support Bundle.
- Runtime capability matrix so new users can inspect CPU-only readiness, optional browser support, and external LLM adapter configuration before running longer checks.
- Matrix-guided home-compute demo that combines local capability discovery with the read-only `model_bundle_infer` inference session.
- Controlled remote Miner demo and browser WebRTC/Worker experiments.

Success signal: a contributor can run the local demo, inspect the protocol boundary, and reproduce acceptance checks without GPU access.

## Beta: Useful Home-Compute Demo

Status: planned.

- Expand `scripts/home_compute_demo.py` around the current read-only multi-request `model_bundle_infer` probe into the first user-facing Swarm Inference shaped workload, starting with small model artifacts, `route_decision`, capped `request_trace` summaries, and explicit capability matching.
- Keep the runtime capability matrix visible as the first user diagnostic while `hardware_targets`, `recommended_routes`, `matched_capabilities`, and `missing_capabilities` expand the hardware/runtime matrix toward CPU, NVIDIA, AMD, Apple Silicon, browser, and remote container paths.
- Keep the measurable CPU inference session path visible with safe latency, throughput, accuracy, and Miner hardware profile summaries.
- Keep `scripts/inference_session_demo.py` as the low-friction local demo while richer runtime adapters mature.
- Use `external_llm_infer_v1` as the first narrow optional adapter contract: deterministic `--enable-mock-llm-runtime` for CI, operator-owned `--llm-runtime-cmd` / `CROWDTENSOR_LLM_RUNTIME_CMD`, OpenAI-compatible `--llm-runtime-url` / `CROWDTENSOR_LLM_RUNTIME_URL` for local servers, and read-only validation of `external_llm_results` before any production serving claims.
- Clear hardware matrix for CPU, NVIDIA, AMD, Apple Silicon, and browser paths.
- Remote Miner onboarding that works through common home-network setups with operator-provided TLS or VPN.
- Better observability for throughput, latency, availability, and rejected work.

Success signal: two ordinary machines can join a controlled swarm demo and produce a useful, verifiable model-serving result.

## P2P and NAT Traversal

Status: planned.

- Replace the central-only discovery path with a P2P daemon layer.
- Add peer identity, capability advertisements, health scoring, and failure-aware routing.
- Evaluate libp2p/WebRTC connectivity paths for home routers and browser participants.

Success signal: Miners can discover and route work without a single hard-coded Coordinator address in controlled test networks.

## Browser-Native Swarm

Status: experimental.

- Move from JavaScript Worker probes toward WebGPU/WebAssembly compute slices.
- Keep browser transport, compute, and safety limits separate from the Python Miner path.
- Measure tensor transfer cost, browser tab lifecycle behavior, and sandbox limits before promising model throughput.

Success signal: browser participants can complete a bounded tensor workload with predictable I/O and failure behavior.

## GPU and Runtime Adapters

Status: planned.

- Add optional adapters for local inference/training runtimes without making the control plane depend on one framework.
- Prefer narrow workload contracts around model artifacts, tensor deltas, and validation over framework-specific APIs.
- Keep CPU-only smoke paths as the baseline acceptance suite.

Success signal: a GPU-capable Miner can advertise backend support and run a real model-shaped workload while CPU-only CI remains deterministic.

## Incentives and Reputation

Status: research.

- Extend workload-scoped trust into longer-lived reputation.
- Explore reward, accounting, and abuse-resistance only after useful workloads and network reliability are proven.
- Treat payment/staking as separate protocol layers, not prerequisites for local and controlled remote demos.

Success signal: the project can explain contribution quality, abuse handling, and operator risk without hiding behind token mechanics.
