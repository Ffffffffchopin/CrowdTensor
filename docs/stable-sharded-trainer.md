# Stable-Sharded Trainer Contract

CrowdTensor does not implement a model trainer. In `stable_sharded` mode it
launches a user-selected upstream trainer, binds one bounded Work Unit to the
latest committed checkpoint, validates the trainer result, and commits the new
checkpoint and contribution receipt exactly once.

## Current Boundary

The ordinary runner supports one stable machine with at least two CUDA ranks.
The generated launch uses Accelerate with FSDP2 and delegates model, dataset,
optimizer, collective, and distributed-checkpoint behavior to the trainer.
Plans spanning more than one physical machine fail with
`stable_multimachine_launcher_required` until a provider-owned launcher can
start every rank without exposing credentials.

A test-only API path also admits a two-rank CPU group. It is not exposed by the
CLI and cannot use the production Accelerate command. The retained integration
test performs real PyTorch FSDP2 full-parameter updates and real
`torch.distributed.checkpoint` save/load on one physical host. It commits steps
0 -> 2, executes and discards one uncommitted optimizer step after a forced
rank-group failure, then restores step 2 under generation 2 and commits step 4.
This is local CPU recovery evidence, not CUDA or physical multi-host evidence.

## Trainer Arguments

The trainer entry point declared during `train plan` must accept these
arguments:

```text
--crowdtensor-project <public TrainingProject JSON>
--crowdtensor-checkpoint-dir <backend-owned checkpoint root>
--crowdtensor-work-unit <private WorkUnit JSON>
--crowdtensor-base-checkpoint <private CheckpointRef JSON>
--crowdtensor-base-payload <base distributed-checkpoint directory>
--crowdtensor-output-checkpoint <new attempt directory>
--crowdtensor-result <rank-zero result JSON>
```

At step zero the base payload directory is empty and the trainer loads the
pinned model revision from `TrainingProject`. At later steps the trainer must
restore model and optimizer state from `--crowdtensor-base-payload`. Every rank
writes its shard under `--crowdtensor-output-checkpoint`; only rank zero writes
the result after the distributed checkpoint is complete.

The result uses schema `crowdtensor_stable_sharded_trainer_result_v1` and binds:

- the exact Work Unit and base-checkpoint content hashes;
- step start, completed step count, restored step, and rank count;
- `distributed_type: fsdp2` and the trainer-reported device type;
- non-negative sample/token counts and finite metrics;
- canonical JSON content hash and public-safety flags.

Unknown fields, mutation, a stale base, a partial step range, a missing shard,
or a non-finite metric fails closed. CrowdTensor independently hashes every
checkpoint file before promoting the attempt directory. Tensor payloads,
trainer logs, credentials, and private paths are excluded from public exports.

## Ordinary Workflow

Create a stable project and record a stable CUDA capability group:

```bash
crowdtensor train init ./stable-project \
  --mode stable-sharded \
  --model <model-id> --model-revision <revision> \
  --dataset <dataset-id> --dataset-revision <revision> \
  --model-adapter <adapter-id> --target-steps 100

crowdtensor train plan ./stable-project \
  --capability ./stable-cuda-capability.json \
  --runtime-probe \
  --trainer-entrypoint train.py \
  --trainer-contract-verified \
  --transformer-layer-class <decoder-layer-class> \
  --materialize
```

Run one bounded checkpoint interval:

```bash
crowdtensor train run ./stable-project \
  --work-unit-steps 10 \
  --max-work-units 1 \
  --execution-timeout-seconds 3600
```

`--max-work-units` bounds one CLI invocation. The launch specification's
`max_restarts` bounds whole-rank-group retries for each Work Unit. Accelerate's
internal elastic restart count is zero so the v2 controller, not a hidden
launcher loop, owns every generation transition and rejected receipt.

## Recovery Rules

Only one stable rank-group Work Unit may be active. A second logical Work Unit
is rejected while it is active. If the operator process disappears:

1. a complete, hash-valid trainer result is committed without retraining;
2. an incomplete attempt receives a rejected receipt;
3. the same logical Work Unit is reissued at the next generation;
4. every rank reloads the latest committed checkpoint;
5. only a complete result advances lineage.

The controller never converts this failure into `elastic_delta` work. A CUDA
live gate and a provider-owned multi-machine launcher remain separate external
validation milestones.
