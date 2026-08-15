# CrowdTensor Agent Instructions

## Active Direction

CrowdTensor is training-first. The primary goal is durable collaborative
training from short, bounded CPU/GPU contributions. Inference is secondary and
may return only through an upstream-engine adapter.

Follow:

- `docs/rfcs/0002-training-first-architecture-v2.md`
- `architecture/module-map.json`
- `architecture/archive-manifest.json`

CrowdTensor owns Work Units, capability-aware placement, checkpoint/recovery,
validation, contribution receipts, and the user workflow. Transformers, PEFT,
Accelerate, PyTorch FSDP2, DeepSpeed, vLLM, and SGLang own model execution and
kernels. Core code must remain framework and provider neutral.

The modes are explicit:

- `elastic_delta`: intermittent volunteer PEFT/delta work;
- `stable_sharded`: a stable rank group using an upstream collective trainer.

Never represent intermittent workers as synchronous full-parameter data
parallelism, and never silently switch modes after a worker disappears.

Do not add a project-hosted global Coordinator as a required dependency. A
session controller belongs to the user who starts a run. Keep historical code
in Git history; do not restore archived command families into the active CLI.

## Current Boundary

The v2 controller supports concurrent elastic Work Units, durable ownership,
heartbeat renewal, generation/expiry/old-base fencing, exactly-once receipts,
and restart-safe lineage. `crowdtensor train run --campaign-dir` creates a
user-owned Volunteer PEFT Session; `train join` performs bounded Cell work.
The stable path launches one upstream rank group and commits only complete,
hash-verified checkpoints.

Real validation covers CPU Volunteer PEFT and a two-rank CPU FSDP2 restart from
the latest committed checkpoint. It does not establish CUDA production,
independent physical multi-host execution, permissionless trust, poisoning or
Sybil resistance, GA, or an SLA.

## Repository Rules

- New core behavior belongs in `crowdtensor/core`.
- Numerical execution belongs in `crowdtensor/backends` or an external plugin.
- Model/provider/data compatibility belongs in `crowdtensor/adapters` or the
  model-adapter plugin interface.
- Do not create per-goal `probe`/`pack`/`check` script families. Extend tests or
  `scripts/check_repository.py`.
- `dist/`, caches, model weights, Campaign state, credentials, and browser
  profiles are not runtime source and must stay out of Git.
- Archived implementations are recoverable from Git baseline `e332a7b`; use a
  separate branch/worktree to inspect them.

## Shared WSL Resource Safety

This WSL instance runs multiple Codex sessions. Treat another session's
process, environment, cache, virtual environment, container, and remote
notebook as user-owned state. Never terminate, replace, or mutate it.

- Inspect `codex-resource status` before GPU use or heavy dependency work.
- Run local GPU workloads through
  `codex-resource run local-gpu-0 -- <command>`.
- Run large framework/build installs through
  `codex-resource run local-heavy-install -- <command>`.
- Before Kaggle work, lease `kaggle-gpu` or `kaggle-tpu`, inspect active
  sessions/quota, keep the lease through cleanup, and stop only resources
  created by this project operation.
- Use a separate virtual environment and project-specific caches per stack.
- Under WSL use the Windows-provided `/dev/dxg` driver. Never install a Linux
  NVIDIA display driver.
- The local RTX 3060 Laptop GPU has 6 GiB VRAM. Use bounded memory and one GPU
  workload at a time.
- WSL has limited RAM. Avoid duplicate model downloads and unbounded build/test
  workers; prefer at most two native build jobs.
- Never run global Docker cleanup or reuse ports/containers owned by another
  session.

See `docs/wsl-local-operations.md` for commands and operational details.
