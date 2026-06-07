# Use Cases

CrowdTensor is useful today for people who want to build, test, or understand
controlled swarm inference before it becomes a larger public network.

It is not yet a production inference service or a permissionless compute
marketplace.

## 1. Local Swarm Inference Experimenter

You want to see a small real model split across multiple local workers.

Run:

```bash
python -m pip install -e '.[dev,hf]'
crowdtensor infer "CrowdTensor routes small models across home compute"
```

You get:

- A fast local Coordinator and two-stage Miner loopback proof.
- Real tiny GPT split inference.
- Local terminal output for the generated text.
- A compact user-facing `infer_summary.json` and safe `infer_summary.md`.
- Decoded-token correctness checks.
- Stage assignment and distinct Miner evidence.
- Redacted artifacts under `dist/`.

Best for:

- Learning how split inference is routed.
- Checking that the project works on your machine.
- Contributing fixes to user-facing commands.

## 2. Home Compute Or Lab Miner Operator

You have one or more trusted machines and want to try them as controlled Miners.

Start with:

```bash
crowdtensor public-real-llm-swarm-beta package \
  --output-dir dist/public-real-llm-package \
  --json
```

Use the generated runbook and join files on trusted hosts only.

Best for:

- LAN or VPN experiments.
- Testing stage 0 and stage 1 on separate machines.
- Exercising failure recovery by stopping a Miner and observing requeue.

Do not expose this as an open public service without your own network controls,
tokens, monitoring, and cleanup process.

## 3. Protocol Or Runtime Contributor

You want to improve the control plane.

Good areas to work on:

- Leasing, heartbeat, timeout, and stale-result handling.
- Stage-aware scheduling.
- Runtime contracts and capability matching.
- Result validation and replay audit.
- Evidence packs and diagnosis codes.
- Cleaner operator commands.

Useful checks:

```bash
python -m unittest discover -s tests
crowdtensor local-proof --json
crowdtensor infer "CrowdTensor routes small models across home compute" --json
crowdtensor public-real-llm-swarm-beta release --max-new-tokens 16 --json
```

Best for:

- Contributors interested in distributed systems.
- People who want correctness before scale.
- Maintainers improving release gates and support bundles.

## 4. Small-Model And CUDA Tester

You want to test the optional real tiny-model backend.

CPU is the default and safest path. CUDA is explicit opt-in:

```bash
crowdtensor public-real-llm-swarm-beta release \
  --public-swarm-v2-backend cuda \
  --max-new-tokens 16 \
  --json
```

Best for:

- Verifying tiny-model stage placement.
- Checking that CUDA hosts fail closed when CUDA is unavailable.
- Comparing CPU and CUDA evidence for the same route.

This is not production GPU pooling and not large-model serving.

## 5. Browser And Edge Experimenter

You are interested in future browser participation.

CrowdTensor already contains browser-oriented experiments for:

- Worker compute probes.
- WebRTC tensor transport.
- Browser Miner bridge concepts.

Best for:

- Exploring how ordinary devices might participate later.
- Testing transport and worker constraints.
- Contributing isolated demos without changing the production control plane.

Browser paths are experiments, not the current main inference route.

## 6. Product Or Demo Evaluator

You want to understand whether the project is ready for a public demo.

Start with:

```bash
crowdtensor public-swarm-beta product-beta --json
crowdtensor public-real-llm-swarm-beta release --max-new-tokens 16 --json
```

Look for readiness fields such as:

- `public_swarm_product_beta_ready`
- `public_real_llm_swarm_beta_ready`
- `real_llm_split_route_ready`
- `decoded_tokens_match`
- `stage_assignment_valid`
- `private_artifacts_cleaned`

Best for:

- Maintainer release review.
- Investor or community demo preparation.
- Checking that shareable artifacts stay redacted.

## Not A Good Fit Yet

CrowdTensor is not ready for:

- Production arbitrary prompt serving.
- Large open-weight model inference.
- Hivemind/Petals-level distributed serving.
- Permissionless public Miners.
- Public economic incentives or staking.
- Running untrusted code from unknown machines.

The strongest current use case is controlled, auditable, small-model swarm
inference development.
