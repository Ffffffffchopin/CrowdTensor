# Community Training Quickstart

CrowdTensor Community RC has one owner workflow. It is a controlled
heterogeneous LoRA system, not permissionless training and not a production
SLA.

## Install

Use Python 3.11 or 3.12 in an isolated environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install 'crowdtensord[hf,storage]==0.2.0rc1'
```

For a source checkout, replace the last command with
`python -m pip install '.[dev,hf,storage]'`.

## Owner Workflow

```bash
crowdtensor community init training-run
crowdtensor community validate training-run
crowdtensor community plan training-run

# Terminal 1
crowdtensor community coordinator up training-run --run

# Terminal 2 or a controlled contributor host
crowdtensor community miner join training-run --run --device-policy auto

# Owner terminal
crowdtensor community train training-run
crowdtensor community status training-run
crowdtensor community pause training-run
crowdtensor community resume training-run
crowdtensor community rebalance training-run --reason performance_rebalance
crowdtensor community export training-run
crowdtensor community stop training-run
crowdtensor community cleanup training-run
```

Add `--dry-run --json` to a mutating command before allocating resources.
Commands return `0` on success, `2` for validation, `3` for state, `4` for
protocol, and `5` for runtime failures. Public JSON uses placeholders and
hashes; private invites and credentials remain below `.crowdtensor/private`.

Remote Coordinators require operator-managed HTTPS or a trusted VPN/reverse
proxy. The RC validates that transport contract but does not provision public
TLS, DNS, NAT traversal, or a durable tunnel.

## Verified Scope

- Qwen2.5-7B remains the pinned heterogeneous production family.
- SmolLM2-135M is the second adapter family and live validation model.
- CPU, CUDA, and optional JAX TPU are explicit Provider capabilities.
- Kaggle Kernels are logical nodes in retained validation. They are not proof
  of independent physical multi-machine operation.
- Only PEFT LoRA is supported. Full-parameter training, data parallelism,
  arbitrary architectures, and in-flight stage migration fail closed.
