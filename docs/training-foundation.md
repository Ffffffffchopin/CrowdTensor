# CrowdTensor Training Foundation

## Status

The CPU-only Training Foundation RC is complete as of 2026-07-10. Its
canonical artifact is:

`dist/training-foundation-rc-20260710/training_foundation_rc.json`

Validate it with:

```bash
PYTHONPATH=. python scripts/training_foundation_rc_check.py \
  --report dist/training-foundation-rc-20260710/training_foundation_rc.json \
  --require-ready --json
```

This RC is real local training evidence. It is not evidence of GPU training,
large-model training, WAN training, full-parameter fine-tuning, or secure
anonymous Miner participation.

## CPU/GPU/Kaggle TPU Training Production RC

The pinned `Qwen/Qwen2.5-7B` CPU/CUDA/JAX-TPU LoRA Training Production RC is
achieved as of 2026-07-17. The canonical artifact is:

`dist/training-heterogeneous-production-rc-20260717-r5-path-redacted-final-ready/training_heterogeneous_production_rc.json`

Validate it with:

```bash
PYTHONPATH=. python scripts/training_heterogeneous_production_rc_check.py \
  --report dist/training-heterogeneous-production-rc-20260717-r5-path-redacted-final-ready/training_heterogeneous_production_rc.json \
  --require-ready --json
```

The strict result is `ok=true`, `error_count=0`, and
`training_production_rc_ready=true`. The artifact file SHA-256 is
`df1f1067ed67339445b9040ca4fd37988dc54d52c505c9e7b6a9680aa8aa2ddc`;
its embedded content hash is
`sha256:2767155e863275a070c1fe22493d8b6410fdefab5684f131779af30f6b047fe5`.

The real gate used two Kaggle T4x2 Kernels, one CPU Kernel, and one JAX TPU
v5e-8 Kernel in one five-stage Qwen Job. The ledger is exactly steps 1..400,
with no gaps or duplicates. The measured training interval was 15,964.81
seconds and the full gate was 16,876.05 seconds, under the six-hour bound.
Every step produced a complete five-stage checkpoint; gradients, losses, and
updates were finite; all LoRA hashes changed; and the final 392-tensor standard
PEFT adapter reloaded in an independent CPU process for a finite forward.

Dynamic recovery was exercised on every provider. CUDA stage 1 left after
step 70 and was taken over by a different Miner in the second T4x2 Kernel at
step 71 under a higher placement generation. CPU stage 4 restored step 90 and
continued at step 91. TPU stage 2 restored step 100 and continued at step 101.
The Coordinator restarted at committed step 80 with no progress loss, and a
stale generation result was rejected. Replacement acceptance binds the old and
new identity hashes, identical stage id, contiguous committed steps, increased
generation, checkpoint downloads, and validated checkpoint archives.

The fixed-workload benchmark used five baseline and five candidate windows of
five steps each. Median throughput improved 32.85%, median p50 step latency
improved 27.05%, and p95 improved 18.95%. The optimized path uses persistent
HTTP for small messages, keeps payloads above 4 MiB on isolated connections,
uses inline/indexed tensor transport, and samples expensive resource telemetry
less often while continuing heartbeat and lease renewal.

The ordinary-user lifecycle is:

```bash
crowdtensor train validate --config production.json --json
crowdtensor train plan --config production.json --json
crowdtensor train start dist/my-production-job \
  --config production.json --dry-run --json
crowdtensor train start dist/my-production-job \
  --config production.json --json
crowdtensor train status dist/my-production-job --watch
crowdtensor train metrics dist/my-production-job --format prometheus
crowdtensor train events dist/my-production-job --limit 100 --json
crowdtensor train pause dist/my-production-job --json
crowdtensor train resume dist/my-production-job --json
crowdtensor train rebalance dist/my-production-job \
  --reason health_degraded --json
crowdtensor train stop dist/my-production-job --json
crowdtensor train cleanup dist/my-production-job --json
```

Omit `--config` to use the pinned default. `start` is idempotent, and status
keeps the credential- and path-free
`crowdtensor train resume <job-dir>` command. Health/readiness, bounded event
pages, low-cardinality Prometheus metrics, worker/stage state, leases,
throughput/latency, queue/retry/reassignment, checkpoint age, telemetry, and
transfer counters are implemented. Fault governance covers bounded timeout and
retry with deterministic jitter, idempotency, lease/generation fencing,
checkpoint integrity and mirrored fallback/repair, journal recovery,
quarantine/circuit breaking, cancellation, and cleanup retry.

The canonical live artifact is an audited replay of retained r9 evidence:
`dist/training-heterogeneous-production-live-20260717-r10-r9-evidence-replay/training_heterogeneous_production_live_probe.json`.
The old aggregator expected the process literally labeled `gpu_replacement`
to take over, although the scheduler correctly assigned stage 1 to an already
available Miner in the other GPU Kernel. Replay only re-derives replacement
and effective-Kernel attribution from all four raw reports and records
`training_measurements_changed=false`; it does not rerun or alter training.

Comprehensive regressions are 301 passed, 2 conditional skips, and 0 failed
across 45 files. Cleanup removed all remote Kernels, private packages,
credentials, tensor payloads, Coordinator, and tunnel, with
`live_resources_left_running=false`. Do not rerun the multi-hour gate merely
to reproduce the achieved artifact.

Production RC does not mean GA or an SLA. This evidence does not cover
arbitrary architecture auto-partitioning, full-parameter training,
data-parallel replicas, in-flight microbatch migration, permissionless
Byzantine/poisoning resistance, secure aggregation, billing/rewards, or
larger-model training.

## Unified CPU/GPU/Kaggle TPU Heterogeneous Training Scheduler Beta

The CPU/CUDA/JAX-TPU extension is achieved for the fixed
`Qwen/Qwen2.5-7B` LoRA topology as of 2026-07-15. The canonical public-safe
artifact is:

`dist/training-heterogeneous-tpu-beta-20260715-r15-gate6-live-achieved/training_heterogeneous_tpu_beta.json`

Validate it with the strict checker:

```bash
PYTHONPATH=. python scripts/training_heterogeneous_tpu_beta_check.py \
  --report dist/training-heterogeneous-tpu-beta-20260715-r15-gate6-live-achieved/training_heterogeneous_tpu_beta.json \
  --require-ready --json
```

That returns `ok=true`, `error_count=0`, no public-safety errors, and
`heterogeneous_training_tpu_beta_ready=true`. The artifact file SHA-256 is
`689a89089f81d3d1ac7362629c1811632647bac74a605ce91429f09dad3341b8`;
its embedded content hash is
`sha256:ee1e0d0b7a1b15909a46b67552fd4bda794eed8f09d595a2d627704a71bb1783`.

Manifest v2 retains five stages `[0,7)`, `[7,14)`, `[14,20)`, `[20,26)`, and
`[26,28)`. Stage 2 is a required `jax_tpu` resource group over all eight v5e
devices; stages 0, 1, and 3 prefer CUDA, and stage 4 is CPU-only. The scheduler
accounts for HBM reserve, mesh/dtype fit, compile latency, steady-state
forward/backward latency, network/load cost, and migration cost without
pretending TPU is CUDA.

The JAX runtime loads only the real stage-owned HF safetensors, shards weights
and LoRA output dimensions over a named mesh, runs BF16 Qwen forward/backward,
applies real Adam updates, and reports finite/checksummed public metadata.
PyTorch and JAX boundaries use the same chunked safetensors protocol. TPU
checkpoints contain adapter, Adam moments, scheduler state, JAX PRNG, manifest,
and progress as safetensors plus JSON; pickle and GradScaler state are not used
for JAX TPU. Standard PEFT export naming and CPU reload remain part of the
strict acceptance contract.

The successful gate used two T4x2 Kernels, one CPU Kernel, and one TPU v5e-8
Kernel. Six Miners joined one Coordinator. Placement generation 1 mapped
stages 0/1/3 to CUDA, stage 2 to `jax_tpu`, and stage 4 to CPU. The global
commit ledger is exactly `[1,2,3,4,5,6]`; every stage produced finite gradients,
executed a real LoRA optimizer update, changed adapter state, and contributed
to each atomic checkpoint barrier.

After step 3, the old TPU Miner left and training paused. A replacement Miner
inside the same retained TPU Kernel registered under placement generation 2,
restored the step-3 adapter, Adam moments, scheduler, JAX PRNG, manifest, and
progress state, then completed steps 4-6. A deliberately delayed generation-1
runtime result was rejected. The data-plane evidence contains at least six
GPU->TPU and TPU->GPU activations and backward gradients, plus the CPU boundary,
all as checksummed, chunked safetensors with finite retry and idempotence.

The TPU stage used JAX 0.10.2 on all eight v5e devices, BF16 compute, named-
mesh parameter sharding, explicit replicated boundary outputs, and explicit
backward gradient layouts. The old and replacement worker statuses measured
about 39,030 ms and 36,674 ms compile latency. Gate 6 omitted these values from
its top-level summary due to a builder field-source bug; the fixed builder now
reads runtime status, and the canonical pack imports the retained status only
after verifying both Miner identities, the same manifest, stage 2, mesh, and
sharding contract. The checker does not relax the positive compile-latency
gate and rejects altered imports.

The final adapter contains 392 standard PEFT tensors across all 28 layers and
was independently reloaded on CPU for a finite full stagewise forward. Final
regressions are 256 passed, 2 conditional skips, and 0 failed. The attempt
ledgers preserve all previous attempts and the unlimited-count authorization,
while each acquisition window remains capped at 12 hours and each live gate at
6 hours. All remote Kernels, temporary packages, credentials, tensor payloads,
Coordinator, and tunnel were removed; `live_resources_left_running=false`.

This Beta remains fixed to Qwen2.5-7B LoRA, sequence length 8, microbatch 1,
and the five-stage topology. It is not arbitrary-model partitioning,
full-parameter training, data parallelism, multi-slice TPU training,
permissionless Byzantine defense, billing, rewards, or a production SLA.

## Unified CPU/GPU Heterogeneous Training Scheduler Beta

The manifest-driven CPU/GPU Heterogeneous Training Scheduler Beta is achieved
as of 2026-07-13 for pinned `Qwen/Qwen2.5-7B` LoRA training. Its canonical
artifact is:

`dist/training-heterogeneous-beta-20260713-r3-r2-live-achieved/training_heterogeneous_beta.json`

Validate it with:

```bash
PYTHONPATH=. python scripts/training_heterogeneous_beta_check.py \
  --report dist/training-heterogeneous-beta-20260713-r3-r2-live-achieved/training_heterogeneous_beta.json \
  --require-ready --json
```

The strict checker returns `heterogeneous_training_beta_ready=true`, zero
errors, and no public-safety findings. The file SHA-256 is
`6c89c167d96e7548ffadba8454fdffa33cc5196b730db04c2d7a0f29e52e2884`;
its embedded content hash is
`sha256:9f2cfe6a982a8d589df172ed38cceeda28baf153055b33b75c1cfdd6d1830a2f`.

Create and operate a heterogeneous Job through the existing owner lifecycle:

```bash
export HF_TOKEN='private-value-if-required'
crowdtensor train create dist/my-heterogeneous-training \
  --heterogeneous --model Qwen/Qwen2.5-7B \
  --hf-token-env HF_TOKEN --json
crowdtensor train serve \
  --elastic-job dist/my-heterogeneous-training \
  --host 0.0.0.0 --port 8791
crowdtensor train invite dist/my-heterogeneous-training \
  --coordinator https://coordinator.example \
  --output-file state/private/heterogeneous-miner.invite.json --json
crowdtensor train status dist/my-heterogeneous-training --watch
crowdtensor train export dist/my-heterogeneous-training \
  --output-dir dist/my-heterogeneous-adapter
crowdtensor train cleanup dist/my-heterogeneous-training
```

The invite remains a private mode-0600 input. A contributor uses the same
Miner entry point as the Elastic Product Beta:

```bash
crowdtensor-miner join --training \
  --invite state/private/heterogeneous-miner.invite.json --role auto
```

The default heterogeneous manifest is pinned in
`crowdtensor/heterogeneous_training_manifest.py`; its published JSON Schema is
`schemas/heterogeneous_training_manifest_v1.schema.json`. A custom
`--manifest` can declare model/revision, LoRA parameters, dataset identity,
precision, sequence and microbatch settings, checkpoint retention, device
policy, memory reserves, straggler thresholds, and arbitrary contiguous stage
boundaries. The achieved five-stage topology is `[0,7)`, `[7,14)`, `[14,20)`,
`[20,26)`, and `[26,28)`. Stages 0-3 allow CPU or CUDA and prefer CUDA; the
final trainable stage is CPU-only. All stages still belong to one Job and one
commit ledger.

Each Miner reports CPU core/RAM/dtype data, every visible GPU's hashed model
name, compute capability, total/free memory and dtype support, current load,
network latency/bandwidth, stage limits, and measured forward/backward
profiles. A host with one visible GPU registers successfully as a one-stage
Miner. A multi-GPU host can expose one independently fenced Miner process per
device; the scheduler does not require two devices in one Miner process.

`crowdtensor/heterogeneous_training_scheduler.py` estimates resident weights,
LoRA parameters and gradients, optimizer state, activations, activation
gradients, workspace, microbatch, and sequence-length peak memory. Placement
first rejects devices that violate hard memory/RAM or dtype constraints. The
remaining candidates are scored by measured or estimated compute time,
adjacent-stage transfer cost, current load, preferred device type, safety
reserve, and model-stage migration cost. Every plan retains candidate audits,
resource estimates, capacity remaining, score components, selection reasons,
capability hashes, and a placement generation. Migration cost prevents a small
profile fluctuation from forcing a multi-GB stage reload; OOM, persistent
straggler, owner-requested rebalance, lease expiry, and Miner loss still have
bounded recovery paths.

`crowdtensor/heterogeneous_tensor_transport.py` carries activation and gradient
tensors as chunked safetensors payloads. The envelope binds the Job and
manifest, global step, microbatch, source/target stage, direction, placement
generation, assignment hash, dtype/shape metadata, chunk and payload hashes,
TTL, byte limits, and retry limits. Receivers validate every chunk and the
assembled payload before dtype/device conversion. Identical replay is
idempotent; conflicting duplicate, expired, oversized, corrupted, or stale
generation messages fail closed. Untrusted pickle deserialization is not
used.

Every stage saves adapter, optimizer, LR scheduler, GradScaler, RNG, and
manifest/progress state. Checkpoint submissions are signed and validated for
stage ownership, tensor coverage, names, shapes, dtypes, finite values, safe
state loading, hashes, step, cursor, and placement generation. A global step
commits exactly once only after all stage archives reach the same barrier. A
missing, OOM, stale, or timed-out stage aborts the whole speculative epoch;
late assignment results cannot enter a later generation. The next complete
placement restores the preceding global checkpoint. Long stage loading and
operations renew leases, while a stage operation has a bounded timeout
separate from the outer Kernel lifetime.

The live gate used two private T4x2 Kaggle Kernels as four one-GPU Miner
processes and one private CPU Kernel as the final stage. Steps 1-3 committed
under the initial placement. A trainable GPU Miner then left deliberately, the
next epoch was revoked, and training paused/rebalanced at step 3. A different
Miner restored the central checkpoint. After bounded Kernel and tunnel
recovery, step 4 committed in placement generation 7; measured profiles drove
generation 8 and steps 5-6 completed without another migration. The exact
commit ledger is `[1,2,3,4,5,6]` with no duplicates or gaps.

Committed transport evidence contains 24 forward activations and 24 backward
gradients: six per stage boundary direction, including six real
CUDA-to-CPU activations and six CPU-to-CUDA gradients. All 48 committed
messages have verified chunk and payload hashes. Four partial messages from
aborted generations remain as fencing evidence but are not counted as
committed traffic. Losses and LoRA gradients are finite, all five stages apply
real optimizer/scheduler updates, and every stage's adapter hash changes.

The completed standard PEFT adapter contains 392 tensors covering Qwen layers
0..27. A separate pure-CPU Kernel downloaded that export, reloaded each of the
five stage shards, applied the adapter, and completed a finite full stagewise
forward. The live harness deleted or verified prior deletion of eight
historical/current Kernel refs, revoked all leases, removed 52 private tensor
payloads, and stopped the Coordinator and tunnel. Temporary packages,
credentials, checkpoints, and private runtime were removed; no live resource
remains.

Final regressions pass 145 tests with zero failures: 66 heterogeneous/shared
Elastic tests and 79 legacy Qwen/CUDA training tests. Evidence implementation
is in `scripts/training_heterogeneous_beta_live_probe.py`, pack/check are
`scripts/training_heterogeneous_beta_pack.py` and
`scripts/training_heterogeneous_beta_check.py`, and the public lifecycle is in
`crowdtensor/heterogeneous_training_beta.py` plus the existing CLI and Miner
entry points.

This Beta proves one pinned Qwen2.5-7B PEFT topology, explicit Qwen stage
boundaries, sequence length 8, microbatch size 1, epoch-level replacement, and
CPU plus single-GPU/multi-GPU-host scheduling. It does not prove arbitrary
architecture auto-partitioning, full-parameter or TPU training, data-parallel
replicas, in-flight microbatch migration, permissionless Byzantine/poisoning
resistance, secure aggregation, multi-account production operation,
incentives/billing, production GA, an SLA, or larger-model training. Anonymous
Hugging Face stage downloads and free-provider tunnel/Kernel lifetimes remain
important availability and throughput constraints.

## Elastic Volunteer Training Runtime

The epoch-level Elastic Volunteer Training Runtime is achieved as of
2026-07-12 for pinned `Qwen/Qwen2.5-1.5B` four-stage LoRA training. Its
canonical artifact is:

`dist/training-qwen15b-elastic-live-20260712-r2-repacked-achieved/training_qwen15b_elastic_live_probe.json`

Validate it with:

```bash
PYTHONPATH=. python scripts/training_qwen15b_elastic_check.py \
  --report dist/training-qwen15b-elastic-live-20260712-r2-repacked-achieved/training_qwen15b_elastic_live_probe.json \
  --require-ready --json
```

The persistent runtime assigns stages through expiring Miner leases and starts
one barrier epoch from the latest committed global step. Each stage optimizer
update is speculative until all four validated checkpoint archives arrive.
The final submission atomically commits the checkpoint set and global step in
SQLite. Identical retries are idempotent; conflicting duplicates and stale
leases fail closed. Losing any assigned Miner aborts the entire uncommitted
epoch, discards its candidates, and restores all stages from the previous
global commit. Incomplete stage coverage enters
`paused_waiting_for_miners`; compatible registrations automatically wake it.

The live gate trained to step 4 on one Kaggle T4x2 pair, removed both old
Miners and Kernels, observed a ten-second zero-Miner pause, and launched an
entirely new T4x2 pair. The new processes downloaded central step-4 adapter,
optimizer, GradScaler, and RNG archives into fresh directories and first
committed step 5. Training finished at step 8 with exactly eight contiguous
global commits, standard PEFT export, CPU/CUDA reload, improved validation
loss, and complete resource cleanup.

Authenticated Coordinator routes are available under `/elastic-training` for
Miner registration, heartbeat/offline transitions, assignments, binary
checkpoint upload/download, barriers, and status. Local operators can inspect
the private persistent state without exposing session or assignment tokens:

```bash
crowdtensor train elastic-status \
  --state /path/to/elastic-training.sqlite3 \
  --run-id private-run-id --json
```

This is not permissionless training security or arbitrary elastic scaling.
The verified policy rolls back a whole uncommitted epoch; it does not migrate
an in-flight microbatch. Model/topology admission, poisoning defenses, secure
aggregation, incentives, multi-account training, and production operations
remain separate work.

## Elastic Volunteer Training Product Beta

The ordinary-user and ordinary-Miner Elastic Volunteer Training Beta is
achieved as of 2026-07-12. Its canonical artifact is:

`dist/training-elastic-beta-live-20260712-r6-repacked-achieved/training_elastic_beta_live_probe.json`

Validate the retained live evidence with:

```bash
PYTHONPATH=. python scripts/training_elastic_beta_check.py \
  --report dist/training-elastic-beta-live-20260712-r6-repacked-achieved/training_elastic_beta_live_probe.json \
  --require-ready --json
```

The strict checker returns `elastic_training_beta_ready=true` with zero errors.
The file SHA-256 is
`a6476c610201877108a8630007c38675229dc33f9bb48fef4de64dbb9ddfae74`.
The report embeds content hash
`sha256:d58c1bd0304ddc126c3c5e5cccf39b0339c7dc1a18ff2370778ef99255d0ab78`.

Create and operate the pinned job through the public owner CLI:

```bash
crowdtensor train create dist/my-elastic-training --json
crowdtensor train serve \
  --elastic-job dist/my-elastic-training \
  --host 0.0.0.0 --port 8791

crowdtensor train invite dist/my-elastic-training \
  --coordinator https://coordinator.example \
  --output-file state/private/miner.invite.json --json
crowdtensor train status dist/my-elastic-training --watch
crowdtensor train export dist/my-elastic-training \
  --output-dir dist/my-elastic-adapter
crowdtensor train cleanup dist/my-elastic-training

# Use this instead of waiting for export when a running job must be stopped.
crowdtensor train cancel dist/my-elastic-training
```

The service address supplied to `train invite` must be reachable from the
Miner. Put TLS and any tunnel or reverse proxy in front of the service when it
crosses a trusted local network. The generated invite is a private mode-0600
file. Transfer it through a private channel, then join from a two-CUDA-device
Miner host:

```bash
crowdtensor-miner join --training \
  --invite state/private/miner.invite.json \
  --role auto
```

The Miner inspects CUDA capability, fetches the pinned private bootstrap,
registers an expiring session, and automatically receives either stage group
`[0,1]` or `[2,3]`. It heartbeats while running, signs each checkpoint
submission, restores the latest committed checkpoint into a fresh private
directory, and drains only after its current barrier commit on SIGINT,
SIGTERM, or a configured drain file. A complete stage cover wakes a paused
job without an owner resume command. Loss of any assigned Miner revokes the
whole uncommitted epoch and all stages restart from the preceding atomic
commit.

Owner status exposes `committed_step`, `online_miner_count`,
`missing_stage_ids`, and a stable `pause_reason`, plus hashed Miner,
assignment, epoch, commit, and event details. A newly created zero-Miner job,
for example, reports missing stages `[0,1,2,3]` and
`pause_reason=incomplete_stage_coverage` instead of only a generic waiting
state.

The owner and Miner HTTP surfaces use separate credentials. Owner operations
are `GET /v1/training/jobs/{job_id}` and authenticated `POST` routes for
`cancel`, `export`, and `cleanup`. Miner bootstrap, capability, registration,
heartbeat, assignment, checkpoint, barrier, and Qwen rendezvous routes are
separately authenticated. Restarting `train serve --elastic-job` reloads the
same private SQLite job and rendezvous state. Public status includes only
hashed identities and aggregate progress; it excludes credential values and
paths, Coordinator URLs, raw training text, token IDs, activations, gradients,
checkpoint tensors, adapter values, and private runtime paths.

Checkpoint archives are HMAC-SHA256 signed over the run/session/assignment/
epoch/stage/archive identity. Before commit, the Coordinator validates model
and stage ownership, required file coverage and hashes, safetensors names,
dtypes, shapes and finite values, plus optimizer, GradScaler, and RNG payloads
through safe `torch.load(weights_only=True)`. Online-Miner limits,
per-session byte limits, rejection counters, stale lease fencing, and session
quarantine bound malformed or abusive submissions. Heavy archive validation
runs outside the event loop so heartbeats continue. This rejects unsigned,
stale, malformed, non-finite, over-quota, and conflicting submissions. It does
not detect a credentialed Miner that submits structurally valid but
semantically poisoned updates, and it is not permissionless Byzantine
training security.

Checkpoint storage defaults to private local content-addressed blobs with
atomic writes and a configurable committed-step retention window. Install the
optional storage dependency and select `--checkpoint-store s3` for an
S3-compatible or MinIO endpoint; bucket, endpoint, prefix, region, and private
credential environment names are configurable. The S3/MinIO implementation is
unit-tested, including content addressing, duplicate puts, reads, listings,
deletes, and public redaction. It has not been exercised against an external
S3 or MinIO deployment, so external object-store reliability is not part of
this Beta evidence.

The public cleanup command and owner HTTP cleanup route are idempotent. They
fence active Miner leases, abort candidates from an uncommitted epoch, remove
private rendezvous payloads and unretained blobs, and preserve an existing
exported adapter, public evidence, and the configured committed checkpoint
window. Volunteer hosts remain responsible for stopping their local Miner
process; the process observes the terminal job state and exits through its
graceful drain path. In the canonical live run, the probe-owned Kaggle wrapper
also deleted all four temporary experiment Kernels and a read-only account
audit found zero active Kernels for the selected account.

The live gate used public owner create/status/export commands and two
successive, distinct pairs of product T4x2 Miners. The old pair committed steps
1-4 and left. Ten observations showed zero live Miners and an unchanged step 4.
The Coordinator service restarted, then the replacement pair restored all four
central stage checkpoints and committed steps 5-8 exactly once. The final
standard PEFT adapter has 392 tensors across layers 0..27 and lowers held-out
validation loss from 2.6663523763 to 2.4811929762. The r6 artifact transparently
binds the immutable r5 runtime measurements to a successful post-cleanup audit;
it does not alter runtime measurements.

Final training regression evidence is:

`dist/training-elastic-beta-tests-20260712-r2-final/training_qwen15b_test_summary.json`

It records 380 passed tests, zero failures, 36 files, and 10 suites. The
Product Beta suite covers public CLI and HTTP lifecycle operations, restart and
replacement semantics, topology assignment, checkpoint signature/non-finite/
quota rejection, quarantine, local storage, S3/MinIO compatibility, evidence
packing, and strict negative checks.

This Beta remains pinned to `Qwen/Qwen2.5-1.5B`, eight steps, four fixed stage
ranges, and two CUDA devices per Miner. It is not evidence for arbitrary
models/topologies, single-GPU contributors, 7B+, full-parameter training,
multi-account training, secure aggregation, permissionless poisoning
resistance, rewards/billing, production GA, or an SLA.

## Qwen 1.5B Training Service Beta RC

The ordinary-user Qwen Training Service Beta RC is achieved as of 2026-07-12.
The canonical artifact is:

`dist/training-qwen15b-beta-20260712-r3-live-achieved/training_qwen15b_beta.json`

Validate it with:

```bash
PYTHONPATH=. python scripts/training_qwen15b_beta_check.py \
  --report dist/training-qwen15b-beta-20260712-r3-live-achieved/training_qwen15b_beta.json \
  --require-ready --json
```

Start the verified topology with private Kaggle credentials supplied by a file
or environment variable:

```bash
crowdtensor train lora --backend cuda \
  --model Qwen/Qwen2.5-1.5B \
  --topology kaggle-2x-t4x2 --steps 8 \
  --output-dir dist/my-qwen-beta-job \
  --kaggle-token-file /path/to/private-token

crowdtensor train status dist/my-qwen-beta-job --watch
crowdtensor train resume dist/my-qwen-beta-job --backend cuda
crowdtensor train export dist/my-qwen-beta-job \
  --output-dir dist/my-qwen-adapter
crowdtensor train cancel dist/my-qwen-beta-job
crowdtensor train cleanup dist/my-qwen-beta-job
```

The command prepares the pinned model source and private WikiText inputs inside
the job, performs a read-only account/quota/activity preflight, and launches two
same-account T4x2 Kernels. It does not require a prebuilt source or dataset
artifact. The stable phase set is `model_resolution`, `dataset`,
`account_preflight`, `allocation`, `kernel_launch`, `stage_loading`, `forward`,
`backward`, `checkpoint`, `recovery`, `evaluation`, `export`, and `cleanup`.

`TrainingBetaJobStore` uses SQLite WAL for durable public status and private
request inputs. Submit and event IDs are idempotent, global step is monotonic,
expired leases enter recovery without resetting progress, only one GPU job can
run at a time, and additional jobs use a bounded queue. CLI operations and the
authenticated HTTP application use the same `TrainingBetaController`. Start
the service with a private token environment variable:

```bash
export CROWDTENSOR_TRAINING_SERVICE_TOKEN='private-value'
crowdtensor train serve \
  --store dist/private-training-service/jobs.sqlite3 \
  --token-env CROWDTENSOR_TRAINING_SERVICE_TOKEN \
  --host 127.0.0.1 --port 8765
```

The API exposes authenticated submit, status, resume, cancel, export, cleanup,
events, and artifact routes; `/health` is unauthenticated. Public responses
contain hashes and phase summaries, not token values/paths, samples, token IDs,
activations, gradients, adapter tensors, private URLs, or Coordinator secrets.
Running cancellation writes a private mode-0600 marker. The live probe checks
it during bounded polling, fails closed, and deletes only Kernel references in
that job's private audit record. Repeated cancel and cleanup calls do not repeat
deletes or mutate revisions.

The canonical run completed in one attempt. Coordinator restart recovery after
resumed step 4 restored all four stage checkpoints and preserved unique
optimizer steps. Deployment/training took 452.652131 seconds, first optimizer
step took 153.472964 seconds, the maximum of 16 recorded step latencies was
8.625027 seconds, and maximum four-stage overlap was 35.103845 ms. The run
transported 129 private payloads / 34,468,880 bytes and recorded 2,439,433,728
peak stage allocated bytes. All temporary resources were removed. Final tests
pass 358/358, including the required service and fault-injection matrix.

This is a Beta RC for pinned Qwen2.5 1.5B PEFT LoRA. It is not production GA,
an SLA, full-parameter or 7B+ training, elastic scaling, multi-account training,
permissionless Miner security, billing, or a Web UI.

## CUDA Two-Node RC Status

The CUDA Two-Node RC is achieved as of 2026-07-11. Its canonical artifact is:

`dist/training-cuda-two-node-rc-20260711-r5-live-achieved/training_cuda_two_node_rc.json`

Run the strict checker with:

```bash
PYTHONPATH=. python scripts/training_cuda_two_node_rc_check.py \
  --report dist/training-cuda-two-node-rc-20260711-r5-live-achieved/training_cuda_two_node_rc.json \
  --require-ready --json
```

It passes with zero errors, both live gates verified, and
`goal_achieved=true`. The authoritative live report is:

`dist/training-cuda-two-node-live-20260711-a3-embedded-single-gate-gradscaler/training_cuda_two_node_live_probe.json`

Implemented CUDA engineering includes `CUDALoRATrainingRuntime`,
`CUDAStageRuntime`, FP16 autocast, GradScaler state, gradient clipping, CUDA
OOM classification, device placement and memory evidence, authenticated
in-memory activation/gradient rendezvous, remote safetensors delta upload into
the existing StateStore validation and DiLoCo path, bounded Kaggle package
lifecycle, CPU/CUDA PEFT evaluation comparison, and strict public-safety gates.

The live attempt used one Kaggle account and two concurrent private T4x2
Kernels. Before joining the cross-machine run, stage0 used both local T4s for
the single-Kernel gate. Separate `cuda:0` and `cuda:1` processes completed a
four-step baseline and a four-step controlled-resume run. Stage1 was stopped
after step 2, restarted under a different PID, restored LoRA/optimizer/
GradScaler/cursor state, and produced adapters and final loss identical to the
baseline. Loss fell from `4.2349076271` to `3.9638671875`, neither stage loaded
the full model, and base weights remained frozen.

The two Kernels then ran one stage each on `cuda:0`, completed four real
cross-machine forward/backward steps, and exchanged four private activation
and four private gradient payloads through the authenticated Coordinator. Two
GPU Miners trained distinct shards and returned 28 named safetensors LoRA
deltas each. StateStore accepted both, completed one DiLoCo outer aggregation,
and advanced adapter version and outer step to 1. Error-feedback transport
reconstructed the dense delta with its residual.

The exported standard PEFT adapter loads on CPU and CUDA, changes logits, and
reduces fixed validation loss from `4.1791639328` to `4.0241522789`. CPU/CUDA
logits agree within `4.4703483582e-8` maximum absolute difference. Downloaded
checkpoint archives preserve 25 stage0 files (embedded baseline/resume plus
pipeline/Miner) and 7 stage1 pipeline/Miner files; their host hashes match the
worker reports.

Strict acceptance remains record-level. The selected single gate is bound to
the stage0 worker's per-step records, source Kernel hash, execution order,
attempt number, checkpoint bundle, and cleanup. The two-node gate requires two
distinct Kernel hashes, authenticated CUDA registrations, complete payload
records, matching model/adapter identities on distinct shards, real delta and
memory/checkpoint evidence, evaluation, and cleanup. CPU fallback,
capability-only evidence, or co-location metadata cannot pass.

The public rejection matrix verifies duplicate, wrong-shard, stale-version,
shape/dtype, NaN/Inf, excessive-norm, and loss-spike rejection. Final
regressions pass with 64 CUDA tests, 27 CPU-training tests, and 167
StateStore/Miner/Coordinator tests. All temporary Kaggle Kernels, packages,
payloads, Coordinator/tunnel processes, and local runtime were removed; the
checkpoint archives and public evidence remain. The one-time RC attempt budget
is exhausted and the achieved run should not be repeated merely to recreate
evidence.

## Qwen 1.5B Four-GPU Training Alpha

The Qwen four-GPU Pipeline Training Alpha is achieved as of 2026-07-12. Its
canonical artifact is:

`dist/training-qwen15b-four-gpu-alpha-20260712-r5-live-achieved/training_qwen15b_four_gpu_alpha.json`

Validate it with:

```bash
PYTHONPATH=. python scripts/training_qwen15b_four_gpu_alpha_check.py \
  --report dist/training-qwen15b-four-gpu-alpha-20260712-r5-live-achieved/training_qwen15b_four_gpu_alpha.json \
  --require-ready --json
```

One Kaggle account supplied two concurrent T4x2 Kernels. Four stage processes
selectively loaded pinned `Qwen/Qwen2.5-1.5B` layers `[0,7)`, `[7,14)`,
`[14,21)`, and `[21,28)` onto four T4s. Both uninterrupted and controlled-
resume runs completed 8 real training steps, with stage 2 restarted after step
4 under a new PID. Baseline and resumed adapters/losses match exactly, training
loss fell from `3.6082336903` to `2.8676728606`, and four-stage compute overlap
reached `50.719332 ms`.

The stable T4 path uses FP32 frozen-model compute, FP32 LoRA parameters and
GradScaler, while authenticated stage-boundary activations and gradients use
FP16 transport. FP16 autocast is disabled because the preceding live attempt
produced a non-finite stage0 activation. PEFT and CUDA precision smoke checks
run before shard materialization, and non-finite runtime values fail closed.

The standard PEFT adapter contains 392 tensors for layers 0..27, loads on CPU
and CUDA, changes logits, and lowers validation loss from `2.6663523763` to
`2.4811929762`. All archives were hash-verified and all temporary cloud/local
runtime resources were cleaned. The final regression suite passes 313 tests.

The Alpha ledger permits an unbounded cumulative number of explicitly invoked
dual-Kernel attempts under the user's superseding authorization. Each attempt
still uses one same-account T4x2 pair, is capped at 30 minutes, and is one
attempt per probe invocation; the runtime must not start an automatic infinite
retry loop or erase attempt history. This is allocation retry policy, not
unbounded simultaneous GPU pooling.

## Install

Training requires the Hugging Face extra:

```bash
python -m pip install -e '.[dev,hf]'
```

The verified RC stack used PyTorch CPU, Transformers, PEFT, safetensors, and
Accelerate. No model download is required: the default job creates a small
local `LlamaForCausalLM` fixture and a standard PEFT LoRA adapter.

## User Commands

Start one bounded local training job:

```bash
crowdtensor train lora --output-dir dist/my-training-job
```

Inspect or recover it:

```bash
crowdtensor train status dist/my-training-job
crowdtensor train resume dist/my-training-job
crowdtensor train export dist/my-training-job --output-dir dist/my-adapter
crowdtensor train cleanup dist/my-training-job
```

The CUDA user path is:

```bash
crowdtensor train lora --backend cuda --output-dir dist/my-cuda-training-job \
  --kaggle-token-file /path/to/private-token
crowdtensor train status dist/my-cuda-training-job
crowdtensor train resume dist/my-cuda-training-job --backend cuda \
  --kaggle-token-file /path/to/private-token
crowdtensor train export dist/my-cuda-training-job --backend cuda
crowdtensor train cleanup dist/my-cuda-training-job --backend cuda
```

Token values and token paths are runtime-only inputs and are not written to
public status. The small-fixture CUDA RC job permits at most two single-Kernel
and two two-Kernel allocation attempts, each bounded to 30 minutes. The
separate Qwen Alpha ledger follows the superseding policy documented above.

The status surface reports these stable phases: `configuration`, `dataset`,
`worker_assignment`, `forward`, `backward`, `local_step`,
`outer_aggregation`, `checkpoint`, `evaluation`, and `cleanup`. A failed job
records a public-safe blocker and a `crowdtensor train resume ...` command.

`cleanup` removes only temporary runtime state. It preserves checkpoints,
evaluation evidence, and the exported adapter.

## Runtime Architecture

The `hf_lora_train` workload is separate from the older `diloco_train` and
`cpu_lora_mock` workloads.

1. The local fixture creates a versioned model manifest, LoRA manifest, and a
   deterministic tokenized JSONL dataset with two hashed shards.
2. The existing HTTP Coordinator creates exactly two leased tasks. Each claim
   binds a model version, adapter version, shard hash, seed, step range, and
   optimizer contract.
3. Two independent `crowdtensor-miner` processes load the same frozen base and
   initial adapter but train different shards through real PyTorch autograd,
   Transformers, and PEFT.
4. Each Miner returns a named-tensor safetensors adapter delta. The Coordinator
   validates names, shapes, dtypes, versions, finite values, norm, loss, shard,
   and duplicate result IDs.
5. The Coordinator averages both deltas and applies one named-tensor
   DiLoCo-style momentum outer step. The global adapter and outer optimizer
   advance from version/step 0 to 1.
6. A deterministic replay re-runs one trusted result and requires exact delta
   tensors. Sign transport with error feedback separately verifies that the
   decoded delta plus residual reconstructs the dense input.

The current trust mode is `permissioned_trusted_local_workers`. Replay catches
accidental or deterministic mismatches; it is not a final defense against
colluding or adaptive malicious public Miners.

## Pipeline Reference

The split-training reference runs at most two worker processes concurrently:

- stage0 owns token/position embeddings and Transformer layers 0-1;
- stage1 owns Transformer layers 2-3, final norm, LM head, and loss;
- neither stage constructs or loads the full model;
- stage0 sends a real activation to stage1;
- stage1 runs loss/backward and sends the activation gradient to stage0;
- each stage updates only its own PEFT LoRA tensors and optimizer state.

Every step records forward and gradient hashes, LoRA gradient norm, optimizer
step, and checkpoint hash. The controlled recovery run hard-stops stage1 at
the midpoint, starts a new process, restores its adapter and AdamW state, and
finishes with adapter tensors and final loss matching the uninterrupted run.

The global checkpoint references both stage adapter checkpoints, both
optimizer states, the dataset cursor, global/outer step, and content hashes.

## Export And Evaluation

The outer-step adapter is exported in the standard PEFT layout:

```text
exported_adapter/
  adapter_config.json
  adapter_model.safetensors
```

The RC loads that directory with standard `PeftModel.from_pretrained` on CPU,
runs the fixed validation set, requires lower loss than the frozen base model,
and requires changed logits. The canonical run reduced validation loss from
`4.1791639328` to `4.0513637066`.

## GPU Continuation And Live RC

Every completed job writes `gpu_training_continuation_manifest.json`. It
contains model/dataset/checkpoint identities, two-stage CUDA placement,
estimated parameter/LoRA/AdamW memory, CUDA worker arguments, a future
two-machine live command, and all conditions that remain unverified for that
specific CPU job. The manifest itself remains dry-run evidence; the separate
r5 artifact above is the live CUDA proof.

`TrainingRuntime` and `StageRuntime` remain device-neutral contracts. Both the
CPU Training Foundation RC and the bounded CUDA Two-Node RC are achieved.
CUDA preserves the CPU claim, checkpoint, and adapter-delta contracts, but the
live RC is still a small deterministic LoRA fixture, not large-model or
full-parameter training. `CUDATrainingRuntimeDryRun`, capability metadata, one
GPU, one process, or one Kernel alone can never satisfy the CUDA checker.

## Development Checks

```bash
PYTHONPATH=. pytest -q \
  tests/test_training_contract.py \
  tests/test_named_tensor_optimizer.py \
  tests/test_hf_lora_training.py \
  tests/test_pipeline_lora_training.py \
  tests/test_training_cli.py \
  tests/test_training_public_safety.py \
  tests/test_training_foundation_rc.py

PYTHONPATH=. pytest -q \
  tests/test_cuda_training_contract.py \
  tests/test_cuda_training_rendezvous.py \
  tests/test_cuda_training_worker.py \
  tests/test_cuda_training_remote_delta.py \
  tests/test_training_cuda_single_kernel_package.py \
  tests/test_training_cuda_single_kernel_probe.py \
  tests/test_training_cuda_two_node_package.py \
  tests/test_training_cuda_two_node_probe.py \
  tests/test_training_cuda_two_node_rc.py
```

The strict checker rejects mock-only training, missing backward, single-process
fake sharding, changed base weights, no loss reduction, missing real adapter
tensors, missing outer aggregation, missing checkpoint recovery, missing PEFT
export, and any CUDA dry-run represented as GPU success.
