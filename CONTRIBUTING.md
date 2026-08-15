# Contributing to CrowdTensor

CrowdTensor is a training-first system for durable progress from intermittent
compute. Changes should preserve the small framework-neutral boundary in
[`RFC 0002`](docs/rfcs/0002-training-first-architecture-v2.md).

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

Install `.[hf]`, `.[storage]`, or `.[tpu]` only when the changed adapter needs
it. Keep accelerator environments isolated.

## Required Checks

```bash
python scripts/check_repository.py --json
python -m compileall -q crowdtensor tests
python -m pytest -q
python -m build --wheel --no-isolation
```

## Design Rules

- Keep `crowdtensor/core` independent of Torch, JAX, Transformers, cloud SDKs,
  and provider APIs.
- Keep `elastic_delta` and `stable_sharded` semantics explicit.
- Put model behavior behind `ModelAdapter`, provider discovery behind an
  adapter, and numerical execution behind `TrainingBackend`.
- Do not add a required project-hosted global Coordinator.
- Do not add per-milestone `probe`/`pack`/`check` script families. Add reusable
  tests or extend the generic repository check.
- Fail closed on unknown schemas, adapters, stale generations, replay, and
  unsupported execution modes.
- Keep credentials, model caches, checkpoints, Campaign state, browser
  profiles, and generated evidence out of Git.
- Label hosted-notebook runs as logical nodes, not independent physical hosts.
- Keep claims narrower than the retained evidence.

Protocol, security-boundary, backend, provider, and model-family changes need
an RFC under `docs/rfcs/`. User-visible changes belong in `CHANGELOG.md`.
Security reports follow [SECURITY.md](SECURITY.md), and conduct follows
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
