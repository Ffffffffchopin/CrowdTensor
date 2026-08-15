# WSL Local Operations

This WSL2 Ubuntu host is shared by multiple Codex sessions. The Windows NVIDIA
driver exposes one RTX 3060 Laptop GPU with 6 GiB VRAM through `/dev/dxg`.

## Resource Guard

Inspect ownership before any GPU work or heavy install:

```bash
codex-resource status
nvidia-smi
```

Run side effects through the appropriate guard:

```bash
codex-resource run local-heavy-install -- <install-command>
codex-resource run local-gpu-0 -- <gpu-command>
```

Before Kaggle work:

```bash
codex-resource lease kaggle-gpu <minutes> <label>
codex-resource lease kaggle-tpu <minutes> <label>
```

After leasing, inspect active sessions and quota. Keep the lease through launch,
wait, evidence retrieval, remote cleanup, and release. Stop only resources
created by the current operation.

## Environment Isolation

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'

export PIP_CACHE_DIR="$HOME/.cache/crowdtensor/pip"
export HF_HOME="$HOME/.cache/crowdtensor/huggingface"
export TORCH_HOME="$HOME/.cache/crowdtensor/torch"
```

Install `[hf]`, `[tpu]`, browser dependencies, or provider-specific CUDA
wheels only when required. Never install a Linux NVIDIA display driver inside
WSL; use framework wheels with the Windows-provided driver. Do not convert an
established CPU environment in place.

WSL has limited RAM. Avoid duplicate model downloads, unbounded worker counts,
and concurrent native compiles; use at most two build jobs. One 6 GiB GPU job
at a time is the default.

## Shared Processes

An idle `nvidia-smi` does not mean the resource lock is free. Existing
unwrapped installers, processes, containers, ports, caches, and notebooks are
authoritative user-owned state. Never terminate or mutate them. Never run
global Docker cleanup.

The former hosted global Campaign was not migrated into WSL because the active
architecture uses user-owned Sessions. Historical private Campaign backups are
operational archives, not repository or runtime dependencies.
