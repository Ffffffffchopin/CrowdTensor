# Community Training Quickstart

CrowdTensor Community RC has one owner workflow. It is a controlled
heterogeneous LoRA system, not permissionless training and not a production
SLA.

## Contribute In Three Steps

1. Receive a one-time `browser` or `agent` pairing code from the Campaign
   Operator.
2. Open `https://crowdtensor.24.199.118.54.nip.io/join`, or install and join a
   native Cell with the command shown there.
3. Wait for the Coordinator to accept the bounded task or LoRA update, then
   leave safely.

Browser calibration is verified auxiliary work and never changes the model.
Use the native Agent for real CPU/CUDA LoRA training.

## Install A Native Agent

Use Python 3.11 or 3.12:

```bash
curl -fsSL https://crowdtensor.24.199.118.54.nip.io/downloads/install-contributor.sh | sh -s -- https://crowdtensor.24.199.118.54.nip.io CT-XXXX-XXXX-XXXX
```

For a source checkout, replace the install command with
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
- CPU and CUDA are supported by the one-click Volunteer Agent.
- Optional JAX TPU is an explicit Provider capability in the managed
  heterogeneous-stage Miner workflow, not the one-click Agent.
- Kaggle Kernels are logical nodes in retained validation. They are not proof
  of independent physical multi-machine operation.
- Only PEFT LoRA is supported. Full-parameter training, data parallelism,
  arbitrary architectures, and in-flight stage migration fail closed.
