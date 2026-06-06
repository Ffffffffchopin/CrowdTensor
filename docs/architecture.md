# Architecture

CrowdTensor is a Coordinator-backed swarm inference beta. It currently favors
clear correctness, recovery, and evidence over permissionless networking or
large-model scale.

The current public path runs a small real Hugging Face GPT model through a
two-stage inference route:

```text
generate client
    |
    v
Coordinator/API  <->  discovery daemon
    |
    +--> stage 0 Miner
    |
    +--> stage 1 Miner
```

The same control-plane mechanics also support deterministic CPU demos, remote
Miner rehearsals, browser probes, and release evidence packs.

## Main Components

### Coordinator

The Coordinator is the authority for the current beta. It owns:

- Session creation and read-only inference requests.
- Task leasing and heartbeat tracking.
- Timeout requeue and stale-result rejection.
- Stage-aware scheduling.
- Result validation and replay audit.
- Trust quarantine and operator diagnostics.
- Redacted evidence and support bundle output.

The Coordinator is intentionally centralized for now. That makes correctness
and recovery easier to inspect before the project moves more responsibility into
P2P networking.

### Discovery

`crowdtensor p2pd` provides the current swarm discovery surface used by
`serve`, `join`, and `generate` flows. It helps clients find the Coordinator and
route through a named `--swarm-id`.

Discovery is not yet a full permissionless libp2p/DHT/NAT traversal system. It
is a controlled beta mechanism for local, LAN, and temporary public proofs.

### Miners

Miners opt in to capabilities. For the real split inference route, the key
roles are:

- `stage0`
- `stage1`
- `both`

The Coordinator only schedules work to Miners that advertise matching
capabilities. Stage-aware demos require distinct stage Miners so the route
actually exercises split execution.

Optional Hugging Face and CUDA paths are also explicit opt-ins. CPU remains the
default.

### Workloads

CrowdTensor supports several workload families:

- Deterministic CPU proofs for protocol and release checks.
- Read-only model bundle inference.
- Optional external LLM adapter smoke tests.
- Micro-LLM and tiny real-LLM sharded inference.
- Browser compute and transport experiments.

The most important public-facing workload today is the real tiny GPT split path
behind `public-real-llm-swarm-beta`.

### Client Commands

User-facing commands wrap the runtime into safer flows:

- `crowdtensor local-proof`
- `crowdtensor infer`
- `crowdtensor cpu-infer`
- `crowdtensor serve`
- `crowdtensor join`
- `crowdtensor generate`
- `crowdtensor public-swarm-beta`
- `crowdtensor public-real-llm-swarm-beta`

Maintainer and release commands produce stricter evidence, validate contracts,
and keep shareable artifacts redacted.

## Request Lifecycle

1. A client calls `infer`, `generate`, or a higher-level beta command.
2. The Coordinator creates a read-only inference session.
3. Stage-specific work is leased to capable Miners.
4. Miners heartbeat while working.
5. If a Miner disappears, the lease times out and the work is requeued.
6. Results are accepted only if they match the current lease/session state.
7. Validation checks compare the split route against a local baseline when the
   workload supports it.
8. Redacted evidence records readiness, diagnostics, stage assignment, and
   recovery details.

## Failure Handling

The runtime is built around observable failure modes:

- Claimed work can be requeued after lease timeout.
- Late results from stale leases can be rejected.
- Stage assignment can require distinct Miners.
- Trust quarantine can block unhealthy workers.
- Release gates can require specific readiness booleans before passing.

This is why many scripts emit structured fields such as
`decoded_tokens_match`, `stage_assignment_valid`, `distinct_stage_miners`, and
`stage_requeue_ready`.

## Data And Artifact Safety

Public artifacts are designed to be safe to share:

- Raw API keys and private tokens are redacted.
- Raw prompts and output text are avoided in public evidence where required.
- Private runtime env files stay local.
- Cleanup commands remove generated private live artifacts by default in live
  proof wrappers.

The project still assumes controlled operators and trusted network boundaries.
It is not safe to expose as an untrusted public mining network.

## What Is Deliberately Not Decentralized Yet

Several pieces remain intentionally simple:

- The Coordinator is still authoritative.
- Discovery is controlled.
- Miner admission is token-backed.
- Validation is workload-specific.
- Model sizes remain small.
- Public prompt serving is not enabled.

Those constraints keep the beta testable. Future work can decentralize pieces
only after correctness and safety remain clear.

## Current Mental Model

CrowdTensor is best understood as:

```text
controlled swarm inference beta
+ real tiny-model split execution
+ observable recovery and validation
+ redacted release evidence
- production P2P
- large-model serving
- permissionless public mining
```
