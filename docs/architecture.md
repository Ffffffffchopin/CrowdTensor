# Architecture

CrowdTensor is a small collaboration layer around established training
runtimes. Its organizing unit is a user-owned training Session, not a hosted
global service.

```text
CLI / optional UI
        |
user-owned Session Controller
        |-- Work Units and capability placement
        |-- checkpoint lineage and restart
        |-- validation and contribution receipts
        |
TrainingBackend --- ModelAdapter --- ProviderAdapter
        |
Transformers / PEFT / Accelerate / FSDP2 / DeepSpeed
```

## Core Contracts

`crowdtensor/core` contains no Torch, JAX, Transformers, Accelerate, DeepSpeed,
or provider SDK imports.

- `contracts.py`: projects, Work Units, checkpoints, lineage, and receipts;
- `controller.py`: lease ownership, expiry/generation fencing, replay, commit,
  and public projections;
- `execution.py`: provider snapshots, execution plans, and stable launch specs;
- `workspace.py`: idempotent local lifecycle and public export;
- `plugins.py`: structural protocols for backends/providers/adapters;
- `cli.py`: the `crowdtensor train` lifecycle.

Every persisted contract is schema- and content-hash-bound. Public projections
exclude credentials, local paths, tensor values, and contributor identities.

## Execution Modes

### `elastic_delta`

Independent contributors claim bounded PEFT/delta Work Units from one base
checkpoint. Multiple leases may coexist. Expired leases can be reassigned at a
new generation; old-base or late submissions are fenced. Accepted deltas can
record receipts before quorum, while only a validated aggregate advances
lineage.

The built-in `volunteer_peft` backend bridges to the retained Volunteer
Campaign protocol and real Transformers/PEFT Cell runtime.

### `stable_sharded`

Exactly one stable rank-group Work Unit is active. An upstream trainer owns
model, optimizer, collectives, and distributed checkpointing. CrowdTensor owns
the bounded launch, result validation, checkpoint promotion, retry generation,
and receipt. A rank loss restarts the whole group from the latest commit and
never converts the run to elastic delta work.

The built-in `accelerate_fsdp2` backend produces an explicit launch contract.
See [the trainer contract](stable-sharded-trainer.md).

## Adapter Boundaries

- `crowdtensor/model_adapter.py`: model-family validation, partition/resource
  estimates, PEFT targets, export/reload, and entry-point discovery;
- `crowdtensor/adapters/capabilities.py`: legacy/public capability discovery;
- `crowdtensor/adapters/providers.py`: generic provider snapshot mapping;
- `crowdtensor/adapters/manifests.py`: the retained Qwen model manifest;
- `crowdtensor/adapters/text_data.py`: bounded model-neutral text shaping.

Provider adapters discover resources. They do not acquire accounts or weaken
training acceptance. Model adapters execute trusted Python and must be pinned
and reviewed.

## User Surface

The public command tree is intentionally narrow:

```text
crowdtensor train ...
crowdtensor volunteer ...
crowdtensor adapters ...
```

`train run --campaign-dir` and `train join` compose the ordinary elastic
workflow. The optional packaged site/dashboard is served by a user-owned
Volunteer Session; it is not a required central CrowdTensor service.

## Evidence Boundary

Current real gates cover CPU PEFT and local two-rank CPU FSDP2 recovery. A
generated plan is not execution evidence. Hosted-notebook workers are logical
nodes, not proof of independently administered physical machines. CUDA,
provider-owned multi-machine launch, semantic-poisoning defenses, and public
permissionless operation remain external milestones.

The old inference/P2P architecture and milestone automation are archived at
Git baseline `e332a7b`; see [the archive guide](archive.md). They are not
available through the current CLI and must not return as top-level imports.
