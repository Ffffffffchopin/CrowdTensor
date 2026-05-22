# Use Cases

This page describes who CrowdTensor is for and what each audience can do today.

## Home Open-Model Player

Goal: use spare machines for open AI workloads without owning a large datacenter GPU cluster.

Today:

- Run `scripts/runtime_matrix.py --json` to see the local runtime capability matrix, hardware/runtime matrix, `hardware_targets`, `recommended_routes`, `matched_capabilities`, and `missing_capabilities` before starting services.
- Run `scripts/home_compute_demo.py --json` for the shortest matrix-guided path from local capability discovery to a measurable CPU-only home-compute inference report with `route_decision`.
- Run `scripts/home_compute_evidence_pack.py` when you need a safe, shareable `home_compute_evidence_v1` artifact with `route_decision`, `matched_capabilities`, and capped `request_trace` rows for an issue report or demo.
- Run `scripts/remote_compute_evidence_pack.py` when you want a safe, shareable `remote_compute_evidence_v1` artifact showing a registry-backed remote-style Miner completing read-only `model_bundle_infer`.
- Run `scripts/remote_demo_runbook_pack.py` for a safe two-machine `remote_demo_runbook_v1` with `operator.private.env`, `miner.private.env`, and a `remote_compute_evidence_pack.py --mode collect` command.
- Run `scripts/remote_demo_acceptance_pack.py` after the two-machine demo is running to collect `remote_demo_acceptance_v1`, `remote_compute_evidence_v1`, `support_bundle`, and `diagnosis_codes` such as `coordinator_unreachable`, `observer_auth_failed`, and `artifact_collection_failed`.
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
- Generate a safe two-machine runbook with `scripts/remote_demo_runbook_pack.py`; copy only `miner.private.env` to the remote host and keep `operator.private.env` on the Coordinator/operator side.
- Run `scripts/remote_miner_join_check.py` and `scripts/remote_miner_readiness_check.py`.
- Run `scripts/remote_compute_evidence_pack.py --mode collect` after a real remote Miner completes `model_bundle_infer` to collect the safe `remote_python_model_bundle_infer` evidence report.
- Run `scripts/remote_demo_acceptance_pack.py` to wait for the real remote Miner result and collect the safe acceptance report plus `support_bundle`.
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
- `scripts/home_compute_demo.py` combines `scripts/runtime_matrix.py` with the local inference session so users can see whether their machine can run the CPU-only `local_cpu_model_bundle_infer` Swarm Inference-shaped path before a longer acceptance run, including `route_decision`, `matched_capabilities`, and `missing_capabilities`.
- `scripts/home_compute_evidence_pack.py` turns that route and session into a safe, shareable evidence pack validated by `scripts/home_compute_evidence_check.py`; runtime acceptance can skip it with `--skip-home-compute-evidence` when a CI lane only needs lower-level checks.
- `scripts/remote_compute_evidence_pack.py` turns a registry-backed remote-style Python Miner run into `remote_compute_evidence_v1`, validated by `scripts/remote_compute_evidence_check.py`; runtime acceptance can opt in with `--include-remote-evidence`.
- `scripts/remote_demo_runbook_pack.py` prepares the safe two-machine `remote_demo_runbook_v1` path for a controlled `model_bundle_infer` result before collecting remote evidence.
- `scripts/remote_demo_acceptance_pack.py` validates the running safe two-machine path and emits `remote_demo_acceptance_v1` with `remote_compute_evidence_v1` and `support_bundle` summaries.

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
