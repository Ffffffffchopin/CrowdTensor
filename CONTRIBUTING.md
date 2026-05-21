# Contributing to CrowdTensorD

CrowdTensorD is an alpha-stage control plane for fault-tolerant distributed AI workload miners. Contributions are welcome, but the project deliberately favors small, reviewable changes over broad rewrites.

## Development Setup

Use Python 3.11 or newer.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .[dev]
```

Run the core checks before opening a pull request:

```bash
python3 -m py_compile coordinator.py miner_cli.py scripts/*.py crowdtensor/*.py tests/*.py
python3 -m unittest discover -s tests -v
python3 scripts/release_gate.py --json
```

If your change touches Coordinator/Miner behavior, also run the non-browser acceptance pack from a shell that allows localhost networking:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8910 \
  --report /tmp/crowdtensor_acceptance.json
```

## Contribution Guidelines

- Keep network/control-plane code physically separate from workload compute code.
- Preserve deterministic CPU-only smoke paths unless the change is explicitly about an optional accelerator path.
- Add focused tests for changes in lease handling, replay, validation, auth, or result application.
- Keep public docs in sync with API or operator behavior changes.
- Do not commit local state directories, token files, browser profiles, checkpoints, or generated caches.

## Pull Request Checklist

- The release gate passes.
- Unit tests pass.
- Runtime acceptance is run or the PR explains why it is not applicable.
- Public docs are updated for user-visible behavior.
- Secrets and local runtime artifacts are not included.
