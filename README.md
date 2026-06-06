# CrowdTensor

CrowdTensor is an open-source experiment in swarm inference: splitting small
real model workloads across ordinary machines, while keeping routing,
validation, failure recovery, and evidence auditable.

The current milestone is **Public Real-LLM Swarm Inference Beta**. Today the
project can run a real tiny Hugging Face GPT model through a Coordinator-backed
two-stage swarm:

```text
p2pd -> serve --p2p -> join stage0 + join stage1 -> generate --p2p
```

That path proves the mechanics needed for larger open AI infrastructure:
peer discovery, stage-aware scheduling, lease recovery, result validation,
redacted evidence, and operator controls.

CrowdTensor is usable as an engineering beta. It is not yet production Swarm
Inference, not a full Hivemind/Petals replacement, not large-model serving, and
not a permissionless P2P network.

## Why It Matters

Most useful AI infrastructure assumes datacenter hardware, trusted operators, or
centralized serving. CrowdTensor explores a different path: ordinary machines
joining controlled, verifiable AI workloads one small step at a time.

The project focuses on the hard parts before the hype: routing, recovery,
validation, observability, artifact safety, and operator experience.

## What You Can Do Today

- Run a local end-to-end split inference proof with a real tiny GPT model.
- Start a local discovery daemon, Coordinator, two stage Miners, and a user
  `generate` request.
- Validate stage assignment, distinct stage Miners, decoded-token correctness,
  KV cache reuse, and failure requeue evidence.
- Package controlled two-machine and Kaggle-style rehearsals for remote CPU
  Miners.
- Try optional CUDA tiny-model stage execution when the Miner host explicitly
  enables it.
- Produce redacted JSON/Markdown evidence and support bundles for debugging or
  release review.

## Quick Start

Use Python 3.11 or newer. The `[hf]` extra installs the optional Hugging Face
runtime used by the real tiny-model demos.

```bash
git clone https://github.com/Ffffffffchopin/CrowdTensor.git
cd CrowdTensor

python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,hf]'

crowdtensor --help
```

Run the fast local proof first:

```bash
crowdtensor local-proof --json
```

Run the current public swarm beta gate:

```bash
crowdtensor public-real-llm-swarm-beta release \
  --max-new-tokens 16 \
  --http-timeout 30 \
  --json
```

This starts local stand-ins for the public swarm path, runs real tiny-model
split generation, validates the evidence, and writes artifacts under `dist/`.

If you only want CPU-only deterministic demos without Hugging Face dependencies:

```bash
python -m pip install -e '.[dev]'
crowdtensor cpu-infer --mode local --json
```

## Manual Swarm Demo

The beta can also be run as separate local processes. Open five terminals from
the repository root after installing the package.

```bash
# Terminal 1: discovery
crowdtensor p2pd --swarm-id public-swarm-v2 --run

# Terminal 2: Coordinator/API
crowdtensor serve --p2p --swarm-id public-swarm-v2 --run

# Terminal 3: stage 0 Miner
crowdtensor join --stage stage0 --p2p --swarm-id public-swarm-v2 --miner-id stage0 --run

# Terminal 4: stage 1 Miner
crowdtensor join --stage stage1 --p2p --swarm-id public-swarm-v2 --miner-id stage1 --run

# Terminal 5: user request
crowdtensor generate \
  --p2p \
  --swarm-id public-swarm-v2 \
  --prompt "CrowdTensor routes small models across home compute" \
  --max-new-tokens 16 \
  --http-timeout 30
```

For real multi-machine trials, keep the Coordinator on a trusted network
boundary, use explicit tokens, and rotate temporary tokens after public demos.

## How It Works

CrowdTensor is intentionally simple at the control-plane layer:

- **Coordinator** owns sessions, leases, result validation, trust state, and
  public HTTP APIs.
- **Discovery daemon** advertises and discovers swarm endpoints for local and
  controlled remote demos.
- **Stage Miners** opt in to specific capabilities such as `stage0`, `stage1`,
  CPU tiny-model inference, or optional CUDA tiny-model inference.
- **Generate client** creates a read-only inference session and streams or
  collects the decoded result.
- **Evidence packs** record redacted readiness, diagnostics, stage assignment,
  failure recovery, and support bundle details.

The current inference work is small by design. It is meant to prove that the
distributed route is correct before the project expands model size, networking,
market incentives, and browser/GPU participation.

## Current Boundaries

CrowdTensor does not currently provide:

- Permissionless production P2P routing with DHT/NAT traversal.
- Hivemind-level distributed large-model serving.
- Open public prompt serving for arbitrary users.
- GPU pooling as a production marketplace.
- Strong economic incentives or staking.
- A security model suitable for untrusted public Miners.

The safe mental model is: **controlled, auditable swarm inference beta for small
models and protocol development**.

## Useful Commands

```bash
# Local proof bundle
crowdtensor local-proof --json

# CPU-only inference aggregate
crowdtensor cpu-infer --mode local --json

# Product-shaped public swarm beta
crowdtensor public-swarm-beta product-beta --json

# Public Real-LLM Swarm Inference Beta
crowdtensor public-real-llm-swarm-beta release --max-new-tokens 16 --json

# Package a two-machine style public real-LLM swarm run
crowdtensor public-real-llm-swarm-beta package --output-dir dist/public-real-llm-package --json

# Clean generated caches and temporary artifacts, dry-run by default
crowdtensor clean-artifacts
```

## Repository Map

- `crowdtensor/` - CLI entry points and user-facing commands.
- `crowdtensord/` - Coordinator, Miner, runtime contracts, validation, and
  workload implementations.
- `scripts/` - evidence packs, release checks, live proof wrappers, and
  acceptance gates.
- `tests/` - unit and integration-style checks for the runtime and evidence
  contracts.
- `docs/quickstart.md` - a guided first run.
- `docs/architecture.md` - control-plane and swarm architecture.
- `docs/use-cases.md` - who the project is useful for today.
- `ROADMAP.md` - what is current, next, and intentionally later.

## Who Should Try It

CrowdTensor is a good fit if you want to:

- Study practical distributed inference mechanics.
- Contribute to open AI infrastructure before it becomes a large production
  network.
- Run controlled home-compute or lab-machine experiments.
- Help harden routing, validation, observability, and operator ergonomics.

It is not the right tool yet if you need production uptime, large open-weight
model serving, untrusted public miners, or a finished token economy.

## Development

```bash
python -m pip install -e '.[dev,hf]'
python -m unittest discover -s tests
```

For documentation-only changes, at minimum run:

```bash
git diff --check
```

## License

CrowdTensor is released under the Apache License 2.0.
