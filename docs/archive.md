# Repository Archive

The active tree is intentionally small and training-first. The archive
boundary was cut from Git ref `e332a7b` on 2026-08-14 after the Training
Architecture v2 implementation was added.

## Retained In The Active Tree

- `crowdtensor/core`: framework-neutral Work Units, checkpoint lineage,
  controller, workspace, and execution contracts;
- `crowdtensor/backends`: Volunteer PEFT and stable Accelerate/FSDP2 bridges;
- `crowdtensor/adapters`: provider, manifest, and small data-shaping adapters;
- `crowdtensor/model_adapter.py`: versioned model-family/plugin boundary;
- `crowdtensor/volunteer_*.py`: the bounded contributor and operator workflow;
- `crowdtensor/project_site` and `volunteer_dashboard`: the optional web UI;
- one generic repository check and the one-click contributor installer.

The exact machine-readable allowlist and archive patterns are in
[`architecture/archive-manifest.json`](../architecture/archive-manifest.json).

## Archived

The removed tree consisted of the former inference/P2P command surface,
provider- and model-specific Kaggle experiments, repeated per-goal
`probe`/`pack`/`check` scripts, duplicate deployment sites, and tests/docs
that only exercised those historical surfaces. They are not silently
supported by the current CLI.

Historical source remains recoverable from Git history and release tags. The
ignored local `dist/` directory may contain public-safe evidence bundles, but
it is not part of the package, is not required to build or run CrowdTensor,
and must not be copied into a deployment image.

## Restoring History

Use a separate branch or worktree when inspecting an archived implementation:

```bash
git show e332a7b:crowdtensor/cli.py
git worktree add ../hivemind-history e332a7b
```

Do not revive an old command by adding another top-level import. New work
must enter through the v2 contracts, a backend/provider adapter, or a model
adapter plugin.

Before a release build, remove `build/` and all root `*.egg-info/`
directories. Setuptools can otherwise copy deleted historical modules from a
stale build tree. `python scripts/check_repository.py` enforces this boundary.
