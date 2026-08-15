# RFC 0002: Training-First Architecture v2

- Status: Accepted and implemented for the local/CPU boundary
- Date: 2026-08-11
- Archive phase completed: 2026-08-14

## Context

CrowdTensor accumulated separate inference, provider, model, release, and
Campaign implementations. Much of that code encoded one experiment rather than
a durable product boundary. The project's differentiating property is not a
new inference kernel; it is durable training progress from intermittent,
bounded contributors.

## Decision

CrowdTensor owns only:

1. Work Units and capability-aware placement;
2. checkpoint lineage, restart, and recovery;
3. validation and contribution receipts;
4. a user-owned operator/contributor workflow;
5. plugin contracts for model, provider, backend, and optional optimization.

Upstream libraries own model execution, kernels, optimizer semantics,
collectives, and distributed checkpoint payloads. A project-hosted global
Coordinator is not a required dependency.

## Modes

### `elastic_delta`

Intermittent contributors perform independent bounded PEFT/delta work. Multiple
leases can share a base checkpoint. Lease expiry and generation changes fence
late results; only validated aggregation advances lineage. No contributor
availability means safe waiting, not failed synchronous training.

### `stable_sharded`

A known stable rank group runs an upstream collective trainer. Only one rank-
group Work Unit is active. Rank failure restarts the whole group from the last
committed checkpoint. The controller must not silently replace this algorithm
with elastic delta work.

## Contracts

- `TrainingProject`: pinned intent and mode;
- `WorkUnit`: bounded execution against an exact base checkpoint;
- `CheckpointRef`/`CheckpointLineage`: immutable progress chain;
- `ContributionReceipt`: accepted/rejected outcome and attribution hash;
- `ProviderSnapshot`/`TrainingExecutionPlan`: public-safe resource plan;
- `TrainingBackend`, `ModelAdapter`, and `ProviderAdapter`: plugin boundaries.

All persisted contracts are canonical-JSON/content-hash-bound and reject
mutation, unknown schemas, stale generations, and invalid replay.

## Module Ownership

- `crowdtensor/core`: framework/provider-neutral state and protocols;
- `crowdtensor/backends`: Volunteer PEFT and Accelerate/FSDP2 bridges;
- `crowdtensor/adapters`: capability, provider, manifest, and data adapters;
- `crowdtensor/model_adapter.py`: model-family plugin registry;
- `crowdtensor/volunteer_*.py`: retained backend support and optional UI;
- `crowdtensor/cli.py`: thin lazy dispatcher.

The exact map is in
[`architecture/module-map.json`](../../architecture/module-map.json).

## Implemented Evidence

- concurrent elastic claims, heartbeat renewal, expiry/generation/old-base
  fencing, exactly-once receipts, and restart-safe lineage;
- real CPU Transformers/PEFT contributor work through direct and HTTP paths;
- user-owned Session composition through `train run` and `train join`;
- stable execution plan/materialization and bounded external trainer launch;
- real two-rank CPU FSDP2 full-parameter updates with distributed checkpoint:
  commit step 2, discard an uncommitted failed step, restore step 2 under the
  next generation, and commit step 4.

This does not prove CUDA production, provider-owned physical multi-host launch,
permissionless admission, semantic-poisoning safety, GA, or an SLA.

## Archive Decision

Superseded inference/P2P code, provider/model experiments, global-Coordinator
entrypoints, duplicate sites, and per-milestone scripts/tests/docs were removed
from the active tree after v2 replacement tests passed. The recoverable baseline
is Git ref `e332a7b`; the machine-readable boundary is
[`architecture/archive-manifest.json`](../../architecture/archive-manifest.json).

Only thin import forwarders remain for the previous heterogeneous manifest and
capability module names. Existing v2 workspaces retain inspect/export coverage.
Historical `dist/` evidence is ignored and never a runtime input.

## Consequences

Positive:

- one public command family and one controller vocabulary;
- framework-neutral correctness tests run without accelerator frameworks;
- upstream runtime upgrades do not require copying their kernels;
- unsupported models/providers fail through explicit plugin boundaries;
- repository and review surface are substantially smaller.

Costs:

- old experimental commands are no longer available on `main`;
- historical evidence reproduction requires a Git worktree at its source ref;
- stable multi-machine launch still needs provider-specific plugins;
- compatibility is deliberately limited to persisted v2 contracts and named
  import forwarders.

## Acceptance Gates

The local v2 architecture gate requires:

1. core imports without ML/provider frameworks;
2. mutation/replay tests for contracts and controller state;
3. both modes use the same Work Unit/lineage/receipt contracts;
4. one real PEFT path and one real stable FSDP2 path;
5. restart resumes from the latest commit exactly once;
6. one documented public CLI family;
7. archived paths stay absent through an automated repository check;
8. old v2 workspaces remain inspectable/exportable.

All eight are implemented for the local/CPU boundary. External CUDA and
physical multi-host evidence remain roadmap gates, not reasons to reopen the
architecture.
