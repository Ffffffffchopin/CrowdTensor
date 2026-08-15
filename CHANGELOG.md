# Changelog

## 0.3.0a1

This alpha establishes the compact Training Architecture v2 release line. It
is intended for bounded, operator-owned Campaigns and contributor testing, not
for permissionless or production use.

### Training Architecture v2

- Added framework-neutral Work Unit, checkpoint lineage, contribution receipt,
  provider snapshot, execution-plan, and user workspace contracts.
- Added concurrent elastic lease/replay/restart handling and one stable rank-
  group controller path.
- Added Volunteer PEFT and Accelerate/FSDP2 backend bridges.
- Added a real two-rank CPU FSDP2 checkpoint-recovery integration gate.
- Consolidated the public CLI under `crowdtensor train`, `volunteer`, and
  `adapters`.

### Repository Slimming

- Archived legacy inference/P2P implementations, model/provider experiments,
  old global-Coordinator entrypoints, duplicate sites, and milestone-specific
  scripts at Git baseline `e332a7b`.
- Reduced scripts to one generic repository check plus the contributor
  installer, and reduced tests to the active architecture/protocol suite.
- Added `architecture/archive-manifest.json`, compatibility import shims, and
  old-workspace inspect/export regression coverage.
- Removed the default global-Coordinator Compose deployment and made the
  container a neutral `crowdtensor` CLI image.

### Campaign In A Box

- Added a Campaign-scoped release directory so a user-owned Session can serve
  the exact contributor wheel, installer, checksums, and release metadata.
- Made the native CPU/CUDA Agent the primary contributor path, with a
  non-consuming resource preflight before a one-time pairing code is redeemed.
- Added explicit local status, graceful stop, and held-out quality-evaluation
  contracts to the v2 workflow.

## 0.2.0rc7

- Published the controlled One-Click Volunteer Contributor Beta and its native
  CPU/CUDA LoRA path.
- Added one-time pairing codes, scoped short-lived credentials, resumable
  uploads, public-safe status, and packaged project/contributor UI assets.
- Retained release evidence remains available through the corresponding Git
  tag and local ignored evidence archives; old release automation is not an
  active runtime dependency.

Earlier experimental history is preserved in Git rather than repeated here.
