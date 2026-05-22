# Use Cases

This page describes who CrowdTensor is for and what each audience can do today.

## Home Open-Model Player

Goal: use spare machines for open AI workloads without owning a large datacenter GPU cluster.

Today:

- Run `scripts/runtime_matrix.py --json` to see the local runtime capability matrix before starting services.
- Run `scripts/home_compute_demo.py --json` for the shortest matrix-guided path from local capability discovery to a measurable CPU-only home-compute inference report.
- Run the 5-minute local Coordinator/Miner demo from [Quickstart](quickstart.md).
- Inspect how a Miner claims work, sends heartbeats, retries transient failures, and submits a validated result.
- Run `scripts/inference_session_demo.py` for a user-facing local inference session summary.
- Run the read-only `model_bundle_infer` smoke to see a multi-request Swarm Inference shaped result path without model mutation.
- Use CPU-only workloads to understand reliability behavior before trusting remote hardware.

Not ready yet:

- Real multi-GPU LLM inference.
- VRAM pooling across home machines.
- Production public-internet deployment.

## Remote Miner Operator

Goal: connect a remote Linux host or container to a controlled Coordinator demo.

Today:

- Generate a token-backed invite with `scripts/create_miner_invite.py`.
- Run `scripts/remote_miner_join_check.py` and `scripts/remote_miner_readiness_check.py`.
- Use hashed token config, `/ready` preflight, retry counters, and Support Bundle diagnostics.

Operator boundary:

- Use HTTPS, VPN, or private networking outside localhost.
- Do not expose admin endpoints directly to the public internet.

## Browser Experimenter

Goal: test browser-native participation without installing a Python Miner.

Today:

- Serve `web/index.html` for the WebRTC tensor tunnel.
- Run browser acceptance with `scripts/browser_acceptance_pack.py`.
- Inspect the browser Miner bridge in `web/browser_miner.html`.

Not ready yet:

- WebGPU model shards.
- Persistent browser compute identity.
- Production browser swarm scheduling.

## Protocol Contributor

Goal: extend the runtime toward useful inference and training while keeping the network layer clean.

Today:

- Start with [Protocol Boundary](protocol.md) and [API Reference](api.md).
- Preserve `runtime_contract_v1` compatibility unless a change is explicitly versioned.
- Add deterministic tests for new workload contracts and keep CPU-only acceptance available.

## Future Swarm Inference User

Goal: split useful model-serving work across ordinary machines.

Today:

- `model_bundle_infer` can verify a tiny built-in bundle prediction session through the same claim, heartbeat, result, validation, and ledger path used by other workloads.
- The Python Miner advertises a CPU `hardware_profile`, and `scripts/inference_session_demo.py` reports aggregate `elapsed_ms`, `requests_per_second`, request count, accuracy, read-only status, and redaction status for the session.
- `scripts/home_compute_demo.py` combines `scripts/runtime_matrix.py` with the local inference session so users can see whether their machine can run the CPU-only Swarm Inference-shaped path before a longer acceptance run.

Planned path:

- Model bundle identity and validation.
- Backend capability matching.
- Result traceability and trust scoring.
- Richer latency and throughput reporting.
- Runtime adapters for real inference engines.

CrowdTensorD does not currently provide real Swarm Inference. The current model bundle path is a dependency-free contract rehearsal.

## Future Swarm Training User

Goal: support low-frequency communication training or fine-tuning across unreliable participants.

Planned path:

- DiLoCo/OpenDiLoCo-inspired outer optimizer contracts.
- Compressed delta transport and error feedback.
- Replay audit for deterministic workloads.
- Longer-running remote Miner sessions and failure-aware scheduling.

CrowdTensorD does not currently train a real LLM. The current workloads validate failure handling, validation, and optimizer boundaries.
