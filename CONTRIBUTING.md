# Contributing to CrowdTensor

CrowdTensor is a Community RC for controlled collaborative inference and
heterogeneous LoRA training. Contributions should be bounded, reviewable, and
backed by executable evidence.

## Development Setup

Use Python 3.11 or newer.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,hf,storage]'
```

Run the core checks before opening a pull request:

```bash
python3 scripts/doctor.py --json
python3 -m py_compile coordinator.py miner_cli.py scripts/*.py crowdtensor/*.py tests/*.py
python3 -m pytest -q
python3 scripts/release_gate.py --json
```

If your change touches Coordinator/Miner behavior, also run the non-browser acceptance pack from a shell that allows localhost networking:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8910 \
  --report dist/crowdtensor_acceptance.json
```

For changes to the public Campaign workflow, run the bounded preview and its
offline checks. The demo uses only a tiny local fixture and removes its private
runtime before returning:

```bash
PYTHONPATH=. python scripts/volunteer_training_public_demo.py \
  --output-dir dist/volunteer-training-public-demo --json
PYTHONPATH=. python scripts/volunteer_training_public_demo_check.py \
  --report dist/volunteer-training-public-demo/volunteer_training_public_demo.json \
  --require-verified --json
PYTHONPATH=. python scripts/community_docs_check.py --json
```

Do not use a same-host preview, a Kaggle logical node, or a queue screenshot as
evidence for the formal physical multi-host launch gate.

## Contribution Guidelines

- Read `AGENTS.md` and `docs/project-memory.md` before making broad design, protocol, roadmap, or positioning changes.
- Keep network/control-plane code physically separate from workload compute code.
- Put new model-family behavior behind `model_adapter_v1.0`; unsupported
  architectures must fail closed.
- Label hosted-notebook evidence `Kaggle logical multi-node`; never claim it is
  independent physical multi-machine validation.
- Preserve deterministic CPU-only smoke paths unless the change is explicitly about an optional accelerator path.
- Add focused tests for changes in lease handling, replay, validation, auth, or result application.
- Keep public docs in sync with API or operator behavior changes.
- Update `AGENTS.md` and `docs/project-memory.md` when long-term project identity, implemented capability, non-capability claims, or roadmap priority changes.
- Update `CHANGELOG.md` for user-visible runtime, API, packaging, security, or operator workflow changes.
- Do not commit local state directories, token files, browser profiles, checkpoints, or generated caches.

## Pull Request Checklist

- The release gate passes.
- Unit tests pass.
- Runtime acceptance is run or the PR explains why it is not applicable.
- Public docs are updated for user-visible behavior.
- Release-facing changes update `CHANGELOG.md`; maintainer releases follow `docs/release.md`.
- Secrets and local runtime artifacts are not included.
- Protocol, Provider, Model Adapter, or security-boundary changes include an
  RFC under `docs/rfcs/`.
- Contributor conduct follows `CODE_OF_CONDUCT.md`.
