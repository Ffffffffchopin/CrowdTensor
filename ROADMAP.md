# Roadmap

CrowdTensor's target is collaborative training from bounded, interruptible
contributions. Inference is secondary and should return only as an upstream
engine adapter, not as a second project architecture.

## Complete: Architecture v2 Foundation

- framework-neutral project, Work Unit, checkpoint, receipt, provider, and
  execution contracts;
- concurrent `elastic_delta` leases with expiry, generation, old-base, and
  replay fencing;
- one stable rank-group controller path for `stable_sharded`;
- user-owned workspace lifecycle: `init`, `plan`, `run`, `join`, `status`,
  `pause`, `resume`, and `export`;
- Volunteer PEFT and Accelerate/FSDP2 backend bridges;
- model and provider adapter plugin boundaries;
- real CPU PEFT and two-rank CPU FSDP2 checkpoint-recovery gates;
- repository slimming and explicit historical archive manifest.

## Immediate: Intermittent Training Study

The active research priority is the [Work-Unit study](docs/intermittent-training-study.md).
It does not require user recruitment or a hosted global Campaign.

- complete deterministic trace replay and upstream/elastic CPU calibration;
- freeze small-model inputs, evaluation, resource budgets, and failure traces;
- compare fixed task sizes before introducing a bounded adaptive policy;
- check FedCompass, FedBalancer, and availability-aware optimization before
  making novelty claims;
- publish matched-resource and matched-token results, including failures, then
  confirm selected comparisons using the existing 3B backend.

## P0: External Reproducibility (Later)

- run the ordinary operator/contributor flow on two independently administered
  Internet hosts;
- publish WAN throughput, reconnect, checkpoint, and cleanup measurements;
- validate one stable CUDA group with the external trainer contract;
- keep all resource acquisition provider-owned and bounded.

## P1: Campaign Quality

- define one public Campaign with immutable model/data revisions, licenses,
  benchmark, rollback owner, and moderation policy;
- publish one launch-candidate community brief with separate compute, data, and
  artifact-review intake paths; external participation still requires evidence;
- add update-quality defenses beyond shape/finiteness/norm checks;
- publish checkpoint lineage and contribution receipts without private data;
- prove statistically meaningful task improvement before making quality claims.

## P2: Contributor Experience

- retain one URL-plus-code native Agent flow;
- make hardware/resource limits understandable before work starts;
- provide reconnect, cache, progress, pause, and cleanup UX;
- package browser work only where it performs real model work or clearly label
  scheduler-calibration tasks.

## P3: Stable-Sharded Providers

- add provider-owned multi-machine launch contracts;
- support upstream FSDP2 and DeepSpeed checkpoint formats through plugins;
- add CUDA and failure-restart evidence without changing training semantics;
- benchmark useful throughput rather than maximum parameter count alone.

## P4: Community Maturity

- independent security review and incident process;
- stable compatibility/deprecation policy;
- reproducible signed releases and dependency inventory;
- governance for Campaign admission, attribution, and result ownership.

Not planned as core ownership: attention kernels, tokenizer semantics, cloud
account automation, a bespoke inference engine, rewards/billing, or a required
global public Coordinator.
