# CrowdTensor

**Train a model together, one bounded contribution at a time.**

CrowdTensor is a training-first coordination layer for intermittent CPU and GPU
contributors. A participant can complete a small Work Unit, disconnect, and let
the next participant continue from the last committed checkpoint.

CrowdTensor owns the parts that upstream trainers do not:

- Work Units and capability-aware placement;
- checkpoint lineage, restart, and stale-worker fencing;
- validation and exactly-once contribution receipts;
- a small operator/contributor workflow;
- model, provider, backend, and optimization plugin boundaries.

Transformers, PEFT, Accelerate, PyTorch FSDP2, and DeepSpeed remain responsible
for model execution and kernels. CrowdTensor does not maintain a second trainer
or require a project-hosted global Coordinator.

## Two Explicit Modes

- `elastic_delta`: intermittent contributors perform bounded PEFT/delta work.
  Progress waits safely when no eligible contributor is online.
- `stable_sharded`: a stable rank group runs an upstream FSDP2/DeepSpeed-style
  trainer. Losing a rank restarts the whole group from a committed checkpoint.

The modes are never silently substituted for one another.

## Quick Start

```bash
python -m pip install -e '.[dev]'

crowdtensor train init ./training-project \
  --model Qwen/Qwen2.5-7B-Instruct \
  --model-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --dataset openai/gsm8k \
  --dataset-revision 740312add88f781978c0658806c59bc2815b9866 \
  --model-adapter qwen2_lora_v1

crowdtensor train inspect ./training-project
crowdtensor train backends
crowdtensor adapters list
```

Run a bounded local Volunteer PEFT session in two terminals:

```bash
# Operator terminal
crowdtensor volunteer campaign create-local ./campaign --target-rounds 1
crowdtensor train run ./operator --campaign-dir ./campaign

# Contributor terminal
crowdtensor train join ./contributor \
  --invite ./campaign/.private/volunteer_invite.json \
  --device cpu --max-work-units 1
```

Remote contributors use HTTPS plus a one-time code. Step, download, timeout,
and Work-Unit limits remain explicit. See the [quickstart](docs/quickstart.md).

## Status

`0.3.0a1` is the current Training Architecture v2 alpha. The v2 core and both
backend modes are implemented. Tests cover
concurrent elastic leases, heartbeat renewal, expiry and generation fencing,
restart-safe lineage, exactly-once receipts, real CPU PEFT work, and a real
two-rank CPU FSDP2 checkpoint/restart gate. CUDA execution and independently
administered physical multi-host validation remain separate milestones.

Retained Qwen2.5-7B GSM8K evidence completed 256 exactly-once LoRA steps across
successive Kaggle logical worker groups and improved a bounded 128-item holdout
from 71.875% to 74.219%. This is showcase evidence, not broad model-quality or
statistical-significance evidence. See the [7B report](docs/qwen7b-gsm8k-elastic-showcase.md).

CrowdTensor does not claim permissionless admission, Sybil or semantic-
poisoning resistance, privacy against an operator, GA, or an SLA.

## Campaign In A Box

Build one immutable contributor release and attach it to a user-owned Session:

```bash
python -m build
crowdtensor release prepare ./campaign-release

crowdtensor train run ./operator \
  --campaign-dir ./campaign \
  --release-dir ./campaign-release
```

The Campaign website then exposes a same-origin Native Agent command. The
installer verifies `SHA256SUMS`, performs a non-consuming hardware/download
preflight, and executes at most one Work Unit by default. Campaign operators
remain responsible for HTTPS, admission, model/data licensing, and backups.

## Repository Boundary

The active source tree is intentionally small. Historical inference/P2P code,
model-specific provider experiments, and repeated milestone scripts were
removed from the runtime tree and remain recoverable from Git history. See the
[archive manifest](architecture/archive-manifest.json) and
[archive guide](docs/archive.md).

New functionality belongs in one of four places:

- `crowdtensor/core`: framework-neutral contracts and controller state;
- `crowdtensor/backends`: upstream numerical runtime bridges;
- `crowdtensor/adapters`: model/provider/data compatibility;
- a separately installed plugin.

## Development

```bash
python scripts/check_repository.py --json
python -m compileall -q crowdtensor tests
python -m pytest -q
python -m build --wheel
```

Read the [architecture](docs/architecture.md),
[RFC 0002](docs/rfcs/0002-training-first-architecture-v2.md),
[stable trainer contract](docs/stable-sharded-trainer.md), and
[contribution guide](CONTRIBUTING.md) before changing protocol behavior.

## License

CrowdTensor is Apache-2.0. Model and dataset licenses remain separate and must
be reviewed by each Campaign operator.
