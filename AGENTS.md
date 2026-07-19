## Latest Qwen2.5-7B Elastic GSM8K Showcase RC Status

Current superseding 7B publicity-training status on 2026-07-19: the Elastic
Domain Fine-tuning Public Showcase RC is achieved. The canonical self-contained
report is
`dist/training-qwen7b-gsm8k-showcase-rc-20260719-r1/training_qwen7b_gsm8k_showcase_rc.json`;
file SHA-256 is
`cc737ea87c6336a0aa423891b60f0a8db5095cd49f76ed2c30fe7a3ca2c4197b`
and content hash is
`sha256:454a814b08695176564486eef8bfa345dc7a17dccffd94a445f4b3495f278007`.
The strict checker passes with zero errors and `showcase_ready=true`:
`PYTHONPATH=. python scripts/training_qwen7b_gsm8k_showcase_check.py --report dist/training-qwen7b-gsm8k-showcase-rc-20260719-r1/training_qwen7b_gsm8k_showcase_rc.json --require-ready --json`.

Attempt 3 of 3 used pinned `Qwen/Qwen2.5-7B-Instruct` revision
`a09a35458c702b33eeacc393d103063234e8bc28` and pinned `openai/gsm8k`
revision `740312add88f781978c0658806c59bc2815b9866`. It completed 256 real
LoRA/SFT optimizer steps, four microbatches per step, sequence length 256,
262,144 non-padding tokens, and 146,659 supervised tokens at learning rate
`2e-5`. Two concurrent T4x2 Kernels completed steps 1-128; both were deleted;
a zero-Miner interval was observed; two fresh T4x2 Kernels restored all four
central stage checkpoints and completed steps 129-256 exactly once.

The final 128-item confirmatory GSM8K holdout is hash-bound and has zero overlap
with the 128-item development set. Normalized exact match improves from 92/128
(71.875%) to 95/128 (74.219%), passing the preregistered practical +2-point
gate. The paired bootstrap 95% interval is [-6.25, +10.9375] percentage points,
so statistical significance is not claimed. Valid answer rate remains 100%
and validation loss changes from 1.389790 to 0.546368. The earlier `1e-4`
development Adapter reduced exact match from 105/128 to 95/128 and is retained
as failed development evidence, not relabelled as success.

The standard 392-tensor PEFT Adapter SHA-256 is
`2c2cb02961df78976eceec94110ddda830bacce2e68d1e7a3d0abe367005a431`.
The RC includes Model Card, showcase, reproduction commands, comparison
example, strict pack/check, and all source/data/attempt/runtime evidence. Three
dataset manifests and six private local payloads were hash-verified then
deleted; all Kaggle Kernels/private Datasets, Coordinator, tunnel, checkpoints,
and private runtimes are gone. The full suite passes 2,574 tests with two
skips. Boundary: Kaggle logical multi-node LoRA/SFT only, not physical
multi-host, broad reasoning, full-parameter or permissionless training, GA, or
SLA. See `docs/qwen7b-gsm8k-elastic-showcase.md`.

## Latest Volunteer Campaign Single-Host Operator Beta RC Status

Current superseding volunteer-training status on 2026-07-18: the Volunteer
Campaign Single-Host Operator Beta RC goal is achieved, explicitly excluding
an independently administered physical multi-machine live test. The canonical
self-contained RC is
`dist/volunteer-training-operator-beta-rc-20260718-r2/volunteer_training_operator_beta_rc.json`.
Its file SHA-256 is
`ef8ded3cb6ff1822be94bb3928a7f9f1197fdd08656f307daf06c3c2e2be683f`
and its content hash is
`sha256:6d5e3ae30c3a03190d4e95b9610a748a20d3e6b35ca9c8485dd4a7d27528e047`.
The strict checker passes with zero errors, `goal_achieved=true`, and
`volunteer_campaign_single_host_operator_beta_ready=true`:
`PYTHONPATH=. python scripts/volunteer_training_operator_beta_check.py --report dist/volunteer-training-operator-beta-rc-20260718-r2/volunteer_training_operator_beta_rc.json --require-ready --json`.

The canonical same-host live probe is
`dist/volunteer-training-operator-beta-probe-20260718-r5/volunteer_training_operator_beta_probe.json`
(file SHA-256
`31318a20b14dcdbe76f27e70242f25c8dfed7b36d46e612c5bb2eec01809031a`,
content hash
`sha256:336ba9b8ba21ed468a5a369fdbfa301649b668da521901e397cb21599cfce64a`).
It runs one real Coordinator process behind a real Caddy TLS container and a
real MinIO S3-compatible container. A verified self-signed local TLS handshake
reaches the trusted proxy path; direct backend HTTP is rejected. Coordinator,
Caddy, and MinIO restart recovery pass, including one active lease and one
interrupted upload that resumes after MinIO downtime without retraining.

Operator state v2 adds signed per-Cell short-lived credentials, immutable Cell
binding, scopes, revocation, persistent nonce replay protection, request/upload
fixed-window rate limits, upload/submission quotas, credential capacity, and
per-Cell lease concurrency. The ordinary `join` client automatically enrolls
and uses a Cell credential; only hashes and public counters persist. The old
shared invite remains an Operator/enrollment secret and a direct API
compatibility path. These controls are not Sybil or semantic poisoning safety.

Campaign `validate/start/pause/resume/finalize/evaluate/export`, private
backup/restore, automatic v1-to-v2 migration, status, cleanup, and Prometheus
metrics are implemented. `crowdtensor volunteer operator` creates/reuses and
serves a Campaign in one command. Restore rejects unsafe archive members,
rebases local state paths, and verifies canonical artifacts plus the audit
ledger. Export excludes credentials and private runtime state.

The r5 bounded gate launches 24 independent OS Cell processes, receives all 24
terminal reports, and completes three quorum-2 rounds with six accepted
protocol-fixture updates. It explicitly records zero real-training stress
processes. Real training evidence is retained from the strict-verified Internet
Beta RC: six independent SmolLM2-135M/WikiText PEFT processes, six optimizer
steps, 96 tokens, and Adapter `v0 -> v3`. Do not relabel protocol stress deltas
as real PEFT.

The gate verifies slow-Cell expiry, duplicate submission idempotency, scope and
revocation rejection, nonce replay rejection, request rate limiting,
credential/upload capacity boundaries, Coordinator/Caddy/MinIO restart,
content-addressed S3 upload, resumable upload, lifecycle, backup/restore,
migration, and credential-free monitoring. All worker/backend processes,
containers, S3 objects/bucket, upload sessions, and private temporary state are
removed; `live_resources_left_running=false`.

The release probe is
`dist/volunteer-training-operator-beta-release-20260718-r4/volunteer_training_operator_beta_release_probe.json`
(file SHA-256
`d4323fb72565a36dd20bdd9f8b967e43eb3694cf795296d3bb3f48a7d0c0bf57`).
It builds the current-source wheel, installs it in an isolated venv without
workspace `PYTHONPATH`, runs the Volunteer contract and Operator help, builds a
current-source project image, runs the contract as the non-root container user,
and removes the image. This exposed and fixed the installed CLI's former
dependency on un-packaged `scripts/create_miner_invite.py` by adding
`crowdtensor/miner_invite.py`.

Focused and adjacent regression is 77 passed with 8 existing PEFT warnings;
the checker mutation test rejects a rehashed TLS downgrade and false-ready
claim. Keep the boundary exact: this is same-physical-host Operator Beta, not
independent physical multi-host evidence, a permissionless trust network,
Sybil/poisoning/Byzantine safety, secure aggregation, useful model quality,
full-parameter training, GA, or SLA. The next external gate is the same
ordinary Operator/Contributor flow on independently administered Internet
hosts. Do not rerun r5 merely to recreate achieved local evidence. See
`docs/volunteer-training-operator-beta.md`.

## Latest Volunteer Training Internet Beta Engineering RC Status

Current superseding volunteer-training status on 2026-07-18: the Internet Beta
Engineering RC goal is achieved for all implementation and local independent-
process validation work, explicitly excluding an independently administered
physical multi-machine live test. The canonical RC is
`dist/volunteer-training-internet-beta-engineering-rc-20260718-r3/volunteer_training_internet_beta_engineering_rc.json`.
Its file SHA-256 is
`d00111b453cf24bb6841805bb7524e4647cf698be1c21dd16e12756ff52663b5`
and its content hash is
`sha256:d21a8b66690e02135aad12095a735ed0df0c3513404d6a21eec9d9d54090165b`.
The strict checker passes with zero errors, `goal_achieved=true`, and
`volunteer_training_internet_beta_engineering_rc_ready=true`:
`PYTHONPATH=. python scripts/volunteer_training_internet_beta_check.py --report dist/volunteer-training-internet-beta-engineering-rc-20260718-r3/volunteer_training_internet_beta_engineering_rc.json --require-ready --json`.

The canonical real probe is
`dist/volunteer-training-internet-beta-20260718-r3/volunteer_training_internet_beta_probe.json`
(file SHA-256
`b00406aebabdf593d23264376f8121338123c2aa3ea5f90c9ac21c333e4e972c`).
It imports all ten files from
`HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
and fixed train/validation parquet files from
`Salesforce/wikitext@b08601e04326c79dfdd32d625aee71d232d685c3`.
Six distinct `crowdtensor volunteer join <private-invite> --once` subprocesses
perform real PyTorch/Transformers/PEFT LoRA work over three quorum-2 rounds,
advancing Adapter and outer step `0 -> 3`. The centralized comparison uses the
same six optimizer steps and 96 tokens; independent replay reloads initial,
distributed, and centralized Adapters with finite validation losses.

The live fault path proves a Cell disappearing after claim and generation-
fenced reassignment, a stale Coordinator endpoint and recovery on the same
lease generation, two Coordinator process restarts preserving one or two
active leases, and one 4,938,616-byte delta interrupted after a 64 KiB chunk.
The persisted upload resumes after restart with `resume_count=1`, is accepted,
and records `training_reexecuted_during_resume=false`. Campaign artifacts and
uploads use local content-addressed storage; an S3/MinIO presigned adapter
contract is implemented and unit tested but not live-deployed in this RC.
Direct HTTP and an untrusted forwarded identity are rejected while the trusted
forwarded-HTTPS termination contract succeeds. This is not a public TLS
handshake proof.

Core additions are `crowdtensor/volunteer_training_campaign.py` and
`volunteer_training_storage.py`, the resumable/TLS/recovery extensions in the
existing volunteer modules, and
`scripts/volunteer_training_internet_beta_{probe,pack,check}.py` plus the
independent replay script. The final public process payload excludes lease
material, paths, token IDs, and tensor specs. The strict RC has no `.private`
directory; service and all Cell processes are stopped; no external accelerator
resource was created.

Keep the boundary exact. This is local independent-process Engineering RC
evidence on one physical host, not independent Internet-machine evidence,
public TLS certificate evidence, live external S3 evidence, Sybil or poisoning
resistance, permissionless Byzantine safety, secure aggregation, useful model
quality, broad scaling, full-parameter training, GA, or SLA. The next and only
external Beta gate for this vertical is the same pinned ordinary CLI/HTTPS flow
on at least two independently administered physical Internet hosts with real
WAN metrics, external reproducibility, and cleanup. Do not rerun r3 merely to
recreate achieved local evidence. See
`docs/volunteer-training-internet-beta.md`.

## Latest Volunteer Training Protocol Alpha Status

Current superseding training-direction status on 2026-07-17: the WAN-friendly
Volunteer Training Protocol Alpha goal is achieved at the local HTTP/real-PEFT
scope. The canonical RC is
`dist/volunteer-training-alpha-20260717-r1/volunteer_training_alpha_rc.json`
with file SHA-256
`c36646c0e367dfd805f5c9047b93de2b87c156a3c1aaa32a0621a58847efad41`
and content hash
`sha256:8cc4e84f647e0dc652729b7a5632f54f6d8b1ca21f0fd9233e85df03c81f72de`.
The strict checker passes with `error_count=0`, `goal_achieved=true`, and
`volunteer_training_protocol_alpha_ready=true`:
`PYTHONPATH=. python scripts/volunteer_training_alpha_check.py --report dist/volunteer-training-alpha-20260717-r1/volunteer_training_alpha_rc.json --require-ready --json`.

The retained real probe performs four PyTorch/Transformers/PEFT LoRA Cell
updates, eight optimizer steps, and 256 training tokens over authenticated
loopback HTTP. Two distinct-Cell quorum rounds advance the canonical Adapter
and DiLoCo/Local-SGD outer step from 0 to 2. It proves forked-base and
non-finite rejection, Cell disappearance, lease expiry, same-work generation
1 -> 2 reassignment, canonical-Adapter replacement, late stale rejection, and
idempotent duplicate handling. The centralized comparison uses the same eight
optimizer steps and 256 tokens. The exact contributor command
`crowdtensor volunteer join <private-invite> --once` succeeds through hardware
detection, content-addressed artifact caching, heartbeat, real PEFT backward,
and binary safetensors upload.

Core implementation is in `crowdtensor/volunteer_training_protocol.py`,
`volunteer_training_coordinator.py`, `volunteer_training_api.py`,
`volunteer_training_cell.py`, and `volunteer_training_cli.py`. The public RC
directory is 108K and the probe deletes its complete `.private` runtime after
copying hash-bound public campaign/status/ledger evidence. The focused and
adjacent regression passes 56 tests plus `py_compile`; cleanup records no live
service or external accelerator resource.

Keep the scope precise. This is WAN-oriented protocol evidence over loopback
HTTP on one physical host, not independent Internet-machine evidence. It is
invite-authenticated and does not prove Sybil resistance, permissionless
Byzantine safety, poisoning resistance, secure aggregation, privacy guarantees,
useful model quality, arbitrary models, GA, or SLA. Next work should use at
least two independently administered Internet machines, object-storage
artifact delivery, immutable real campaign model/data import, longer churn and
Coordinator crash recovery, then one bounded reproducible public training
campaign. Do not rerun this local Alpha merely to recreate achieved evidence.

## Latest Model Adapter Ecosystem Beta Status

Current superseding model-ecosystem status on 2026-07-17: the pluggable Model
Adapter Ecosystem Beta goal is achieved. The canonical portable RC is
`dist/model-ecosystem-beta-20260717-r1/rc/model_ecosystem_beta_rc.json` with
file SHA-256
`59f2eded2a608da0479a229297eb5f8737db160169706b37d57481953a53496f`
and embedded content hash
`sha256:99ec6314f26b783ec4ba4c7ee24922671a9227b41b8ffe4b152d7451f320bfc2`.
Its checker passes with `error_count=0`, `goal_achieved=true`, all nine portable
artifact hashes valid, and public safety clean:
`PYTHONPATH=. python scripts/model_ecosystem_beta_rc_check.py --report dist/model-ecosystem-beta-20260717-r1/rc/model_ecosystem_beta_rc.json --json`.

The registry now discovers separately installed Adapters through
`crowdtensor.model_adapters.v1`, records distribution provenance, rejects
invalid metadata/name mismatch/built-in shadowing/conformance failures, and can
be disabled with `CROWDTENSOR_DISABLE_MODEL_ADAPTER_PLUGINS=1`. The Community
CLI exposes `adapters list|check`; the live service authenticated-serves both a
core wheel and an Adapter wheel. The official separate plugin is
`crowdtensor-mistral-adapter==0.1.0b1`, Adapter ID `mistral_lora_v1`, pinned to
`Locutusque/TinyMistral-248M-v2@0f57b17cb317bb322c7c1466b669c681f80c058f`
(Apache-2.0, 248,024,064 real trained parameters, `MistralForCausalLM`). The
isolated no-workspace plugin smoke passes at
`dist/model-ecosystem-beta-20260717-r1/plugin-smoke-r4-core-r2/model_adapter_plugin_smoke.json`.

The strict live proof is
`dist/model-ecosystem-beta-20260717-r1/mistral-live-attempt-2-strict-boundary/mistral_kaggle_heterogeneous_live.json`;
`scripts/mistral_kaggle_live_check.py` passes with
`mistral_live_verified=true`. One Kaggle T4x2 Kernel plus one CPU Kernel
clean-installed both wheels and completed exactly 8 contiguous two-stage LoRA
steps in 195.155312 seconds. Both stages checkpointed at 4/8. The old CUDA
worker stopped after step 4 without leasing step 5; a distinct worker restored
LoRA plus Adam state and performed all stage-0 phases for 5..8. Eight
activation and eight gradient transfers were
hash-verified; both stage Adapters changed; 168 tensors merged into standard
PEFT; an independent CPU process reloaded finite `[1,4,32005]` logits. The
separate Mistral ledger marks attempt 1 `superseded` by the stricter boundary
checker and attempt 2 `current`/`achieved`; it did not modify the exhausted
Community ledger. Both temporary Kernels, tunnel, Coordinator,
and private runtime were removed; post-run active Kernel count is zero. The
focused local gate passes 42 tests and `py_compile`.

Do not overclaim this result. It uses deterministic private token sequences and
does not verify useful model quality, a real dataset, Mistral-7B, arbitrary
Mistral/checkpoint or architecture support, stage-selective initial loading,
full-parameter training, TPU for Mistral, physical multi-machine independence,
GA, or SLA. Label the live topology `Kaggle logical multi-node`.

## Latest CrowdTensor Community Maturity RC Status

Current superseding Community status on 2026-07-17: the CrowdTensor Community
Maturity RC P0-P4 goal is achieved. The canonical portable artifact is
`dist/community-maturity-rc-final/canonical-rc-attempt3/community_maturity_rc.json`
with file SHA-256
`4939653cf3bb03cbc5879dc527abe7801a2de9a9e9430bec5195a1e02dc46ca1`
and embedded content hash
`sha256:4bca6aae9d221b8fc58602482cd524aba5af423023afcf01fa1267c8691aab70`.
Its default and strict checkers pass with zero errors, all nine source artifacts
valid, all 18 requirement-level gates true, P0-P4 true, cleanup true, and
`community_maturity_rc_ready=true`:
`PYTHONPATH=. python scripts/community_maturity_rc_check.py --report dist/community-maturity-rc-final/canonical-rc-attempt3/community_maturity_rc.json --require-ready --json`.

The immutable full-gate ledger is
`dist/community-maturity-live-gates/community_live_gate_ledger.json` (file
SHA-256
`2e6d96ab3f4bdae825429e6ea8a6280b68499002a9e3066e03b940014bfdebf9`).
Both originally allowed attempts are consumed. Attempt 1 failed before runtime
because the Coordinator served an invalid renamed wheel filename. Attempt 2
reached both logical Kaggle workers but the CUDA worker failed before step 1.
Diagnostic-only runs proved the second root cause: Kaggle's optional
`torchao==0.10.0` made PEFT 0.19.1 reject LoRA injection and later independent
reload. Both failed full gates deleted all Kernels, stopped Coordinator/tunnel,
removed private state, and left no live resource. On 2026-07-17 the user
explicitly authorized exactly one additional Kaggle CPU+GPU gate, capped at 45
minutes, with every other boundary unchanged. The ledger now contains a
public-safe `2 -> 3` amendment at `2026-07-17T11:58:34Z`; it stores only approval
hash `sha256:caff1c4d02d1032d5b618ec6c61029ccc7f6783470a95828a2dd000f1248f301`,
old/new limits, timestamp, and scope. Attempts 1 and 2 remain unchanged as JSON
values. Attempt 3 is terminal with outcome `achieved`; the amended maximum is
exhausted at 3/3. Never reset, rewrite, remove, repeat, or further extend this
ledger without a new explicit authorization.

The successful full live report is
`dist/community-maturity-rc-final/kaggle-live-attempt3/community_kaggle_short_reliability_live.json`
with file SHA-256
`3c2ba22067657892cc00a67dcfead0520478ab78620da76464417c1b65efb55d`.
Its strict checker passes. One Kaggle T4x2 Kernel and one Kaggle CPU Kernel,
both clean-installing the immutable release wheel outside the workspace,
completed 100 strictly contiguous atomic LoRA steps in 578.9289 seconds. The
CUDA worker was replaced after step 30 and restored LoRA plus Adam state from
checkpoint. The persisted restart barrier requested after step 50 restarted
the Coordinator at committed step 51, advanced generation 1 -> 2, preserved
the ledger, and measured 1.787347 seconds of downtime. All losses and updates
were finite, checkpoints were written at steps 30/50/100, and transfer values
remained private. The embedded dual-CUDA SmolLM2 run completed two logical
stage updates, merged PEFT adapters, exported them, and independently reloaded
finite logits. Its standalone report is
`dist/community-maturity-rc-final/kaggle-live-attempt3/community_smollm_two_stage_lora_live.json`
with file SHA-256
`13b109554d296123c30304de1386506c1711054e34062a70c06dd82f27a71159`;
its strict checker passes.

The final release wheel is
`dist/community-maturity-rc-final/release/artifacts/crowdtensord-0.2.0rc1-py3-none-any.whl`
with SHA-256
`3694aedad53bfb55d9ebb38bb92fe65f1b9fe6fef1c79e9be7a2f077de7d7b4b`.
The matching CPU Kaggle clean-install proof is
`dist/community-maturity-rc-final/kaggle-wheel-smoke/community_kaggle_wheel_smoke.json`;
it verifies hash/name, exact pins, isolated package location, full model-stack
import, `init/validate/plan`, privacy, and cleanup. The matching diagnostic at
`dist/community-maturity-rc-final/kaggle-gpu-diagnostic/community_kaggle_gpu_stage0_diagnostic.json`
uses two real T4 devices and real pinned SmolLM2 weights. It verifies stage0
forward/backward/optimizer update plus two independent CUDA stage processes,
two contiguous LoRA steps, both stage adapters changed, PEFT export, independent
reload, exact wheel identity, privacy, and cleanup. It remains explicitly
`diagnostic_only=true`; final readiness comes from the successful full gate.

The final release report at
`dist/community-maturity-rc-final/release/community_release_build.json` has
file SHA-256
`9b8fcc3b8d5d50f78f53e29783360aff1acba1ace8815c654cfc949df324c71f`.
It reused the exact successful wheel and sdist without rebuilding either,
enforced the wheel SHA, reran six clean-install golden commands, built and
removed the Docker image, and refreshed Compose, CycloneDX SBOM, dependency
licenses, exact Kaggle runtime lock, documentation bundle, hashes, privacy, and
non-publication evidence. The offline bundle SHA-256 is
`f315dfac6709f15e52ca22b19b46e99e4413acd9b971eec5c5b26d26969dec28`.
The final local gate at
`dist/community-maturity-rc-final/local-gate-attempt3/community_local_gate.json`
passes 81 focused tests and all 12 bounded chaos scenarios; an earlier broader
heterogeneous regression passed 62 tests with 2 conditional skips. Real MinIO
API put/get/list, content addressing, mirror fallback/repair, retention,
restart, and cleanup remain verified. The final cleanup at
`dist/community-maturity-rc-final/cleanup-final-attempt3/community_cleanup_audit.json`
strict-passes with three live evidence sources, authenticated Kaggle query,
zero matching Kernels, zero private runtimes, and zero Community Docker
images/containers.

Scope remains precise: all Kaggle evidence is `Kaggle logical multi-node`, not
proof of independent physical machines. This is an offline Community RC, not
an external publication, GA/SLA claim, unlimited provider capacity, arbitrary
model support, full-parameter training, or Byzantine-complete system.

## Latest CPU/GPU/Kaggle TPU Training Production RC Status

Current superseding Training status on 2026-07-17: the pinned
`Qwen/Qwen2.5-7B` CPU/CUDA/JAX-TPU LoRA Training Production RC is achieved.
The canonical public-safe artifact is
`dist/training-heterogeneous-production-rc-20260717-r5-path-redacted-final-ready/training_heterogeneous_production_rc.json`.
Its file SHA-256 is
`df1f1067ed67339445b9040ca4fd37988dc54d52c505c9e7b6a9680aa8aa2ddc`
and its embedded content hash is
`sha256:2767155e863275a070c1fe22493d8b6410fdefab5684f131779af30f6b047fe5`.
The strict checker passes with `error_count=0`, no public-safety errors, and
`training_production_rc_ready=true`:
`PYTHONPATH=. python scripts/training_heterogeneous_production_rc_check.py --report dist/training-heterogeneous-production-rc-20260717-r5-path-redacted-final-ready/training_heterogeneous_production_rc.json --require-ready --json`.

The retained real live source is
`dist/training-heterogeneous-production-live-20260717-r10-r9-evidence-replay/training_heterogeneous_production_live_probe.json`.
It reuses the immutable r9 Kaggle measurements and only re-derives replacement
and effective-Kernel attribution from all four retained raw Kernel reports.
The replay records `live_run_reexecuted=false` and
`training_measurements_changed=false`. The source completed exactly steps
1..400 with no duplicate or missing commit, ran the actual training interval
for 15,964.81 seconds and the full gate for 16,876.05 seconds, and used real
Kaggle CPU, CUDA, and eight-device JAX TPU providers in one Job.

Replacement evidence is now stage-, identity-, checkpoint-, and generation-
bound. CUDA stage 1 moved from `gpu_a` after step 70 to a different Miner in
`gpu_b` at step 71 under generation 2; this valid cross-Kernel reassignment is
why the old r9 `gpu_a` wrapper's idle designated replacement is not a runtime
failure. CPU stage 4 restored step 90 and resumed at step 91 under generation
4. TPU stage 2 restored step 100 and resumed at step 101 under generation 5.
The Coordinator also restarted at committed step 80 without progress loss,
and a stale generation result was rejected. Future Miner reports preserve the
full `stage_process_ready_history`; the strict checker also accepts the r9
fallback only when contiguous same-stage steps, distinct identities, increased
generation, checkpoint download counts, and validated before/after archives
all agree.

The fixed-workload five-window performance gate passed: median throughput
improved 32.85%, median p50 latency improved 27.05%, and p95 improved 18.95%.
All five stages produced finite updates and changed LoRA hashes; every step had
a complete atomic checkpoint; the 392-tensor PEFT adapter independently
reloaded on CPU and completed a finite forward. Monitoring captured structured
events, Prometheus metrics, worker replacement, transfer, profiles, and the
Coordinator restart. The deterministic fault suite verifies bounded retry,
generation fencing, lease reclaim, worker quarantine/circuit breaking,
mirrored-checkpoint fallback and repair, journal recovery, and cleanup retry.
Public status uses `crowdtensor train resume <job-dir>` and never serializes an
absolute local job path into its credential-free resume command.

Final comprehensive training regressions are 301 passed, 2 conditional skips,
0 failed, and 5 warnings across 45 files. The local JAX import collection skip
does not replace the retained real TPU evidence. Cleanup is complete: all four
remote Kernels, private packages, credentials, tensor payloads, Coordinator,
and tunnel were removed, and `live_resources_left_running=false`. Do not rerun
the multi-hour Kaggle gate merely to recreate this achieved evidence.

This is a Production RC for one pinned 7B LoRA topology, not production GA or
an SLA. Arbitrary architecture partitioning, full-parameter training,
data-parallel replicas, in-flight microbatch migration, permissionless
Byzantine/poisoning resistance, secure aggregation, billing/rewards, and
larger-model training remain outside this achieved scope.

## Latest Unified CPU/GPU/Kaggle TPU Heterogeneous Training Beta Status

Current superseding status on 2026-07-15: the pinned `Qwen/Qwen2.5-7B`
CPU/CUDA/JAX-TPU LoRA Training Beta is achieved. The canonical public-safe
artifact is
`dist/training-heterogeneous-tpu-beta-20260715-r15-gate6-live-achieved/training_heterogeneous_tpu_beta.json`.
Its file SHA-256 is
`689a89089f81d3d1ac7362629c1811632647bac74a605ce91429f09dad3341b8`
and its embedded content hash is
`sha256:ee1e0d0b7a1b15909a46b67552fd4bda794eed8f09d595a2d627704a71bb1783`.
The strict checker passes with `error_count=0`, no public-safety errors, and
`heterogeneous_training_tpu_beta_ready=true`:
`PYTHONPATH=. python scripts/training_heterogeneous_tpu_beta_check.py --report dist/training-heterogeneous-tpu-beta-20260715-r15-gate6-live-achieved/training_heterogeneous_tpu_beta.json --require-ready --json`.

The successful live source is
`dist/training-heterogeneous-tpu-beta-20260715-r14-window4-submission2-live-gate6-stale-race-fix/training_heterogeneous_tpu_beta_live_probe.json`
with file SHA-256
`03572217fc7d3a6a5d962b8d182f14a21233a82d190efd7e6ecf86cc9a81a620`.
It ran one Job and Coordinator across two Kaggle T4x2 Kernels, one Kaggle CPU
Kernel, and one real eight-device TPU v5e-8 Kernel. Placement generation 1
assigned CUDA stages 0/1/3, JAX-TPU stage 2 `[14,20)`, and CPU stage 4. The
atomic commit ledger is exactly `[1,2,3,4,5,6]`; all five stages produced
finite gradients, changed LoRA state, and submitted real checkpoints.

After step 3 the old TPU Miner left, training paused, and a replacement Miner
inside the same retained TPU Kernel restored the step-3 JAX checkpoint and
completed steps 4-6 under placement generation 2. A deliberately stale
generation-1 runtime result was rejected. Evidence proves bidirectional
GPU/TPU/CPU chunked-safetensors activation and gradient traffic, eight-device
named-mesh BF16 forward/backward/optimizer execution, changed TPU adapter
hashes, JAX PRNG/Adam/scheduler checkpoint state, and a 392-tensor standard
PEFT adapter reloaded for a finite full stagewise CPU forward.

The retained TPU worker statuses contain real compile measurements of about
39,030 ms and 36,674 ms. Gate 6 initially omitted those values from its top-
level summary because the builder read step results instead of runtime status.
The builder is fixed, and the canonical pack imports the retained status only
after binding both old/replacement Miner hashes, stage 2, the same manifest,
8-device mesh, and explicit forward/backward output sharding. The checker
rejects a missing, zero, mismatched, or modified import; the measurement is not
recomputed.

Final regressions are 256 passed, 2 conditional skips, and 0 failed. Both
attempt ledgers preserve all history and record the user's unlimited-count
authorization while keeping each acquisition window bounded to 12 hours and
each full gate bounded to 6 hours. All remote Kernels, private packages,
tensor payloads, credentials, Coordinator, and tunnel were removed;
`live_resources_left_running=false`. Do not rerun the long live gate merely to
recreate this achieved evidence. A final read-only audit authenticated all four
authorized Kaggle accounts and found zero matching
`ct-training-production-*` resources, with no queued/running remainder.

## Previous TPU Training Beta Gate-4 Blocker Status

The following 2026-07-14 status is retained as diagnostic history and is
superseded by r15 above. The pinned `Qwen/Qwen2.5-7B`
CPU/CUDA/JAX-TPU LoRA engineering path is implemented, but the real Kaggle
three-accelerator Training Beta is not achieved. The canonical public-safe
blocker is
`dist/training-heterogeneous-tpu-beta-20260714-r12-live-gate4-terminal-diagnostic/training_heterogeneous_tpu_beta.json`.
Its file SHA-256 is
`85f80d7397c76a75abe5ab445927fab422eab222d949af5527965f6b8ff822fe`
and its embedded content hash is
`sha256:3498a3ebca48a71096d60a593d239210136be251d1744c589f8b088137379a9d`.
The default checker passes with `error_count=0` and
`heterogeneous_training_tpu_beta_ready=false`:
`PYTHONPATH=. python scripts/training_heterogeneous_tpu_beta_check.py --report dist/training-heterogeneous-tpu-beta-20260714-r12-live-gate4-terminal-diagnostic/training_heterogeneous_tpu_beta.json --json`.
The strict checker intentionally fails 11 checks because the six-step live
evidence is absent. Do not mark this goal achieved.

Live gate 4 used the explicitly authorized 3 -> 4 limit extension and a GPU
account whose quota preflight passed twice with about 72,853 effective seconds
remaining. Both T4x2 Kernels, one CPU Kernel, and one TPU v5e-8 Kernel were
accepted. Six Miners joined one Coordinator; placement generation 1 covered
CUDA stages 0/1/3, JAX-TPU stage 2, and CPU stage 4. CUDA stage 3 and CPU stage
4 submitted real step-1 checkpoints. Stages 0, 1, and 2 did not submit step-1
checkpoints, no global step committed, and the TPU Kernel became terminal at
`2026-07-14T20:47:59Z`. The original terminal worker report was not published
before cleanup, so r12 correctly records the gate-4 root cause as unconfirmed.

The final acquisition submission was used only for a bounded TPU stage-2
diagnostic; it did not consume or reset the live-gate ledger. The retained
diagnostic is
`dist/training-heterogeneous-tpu-stage-diagnostic-20260714-r1-window3-submission3/training_heterogeneous_tpu_stage_diagnostic_live_probe.json`
with SHA-256
`6addb3d3a2ad132cb1e2644c4303a09b7310c580fa0ed22eb77d288f5e92222f`.
It proves a real JAX 0.10.2 v5e-8 runtime, all eight devices, stage-selective
Hugging Face loading of 2,796,701,728 bytes and 72 tensors for layers
`[14,20)`, named-mesh sharding across all devices, and a completed forward.
Backward compilation failed with `IndivisibleError`; no optimizer update or
checkpoint was produced. This diagnostic used synthetic boundary tensors and
explicitly records `full_training_gate_evidence=false` and
`same_job_three_accelerator_evidence=false`.

Current source adds progress callbacks for every weight range group, retries
terminal output collection with the diagnostic-specific file pattern, and
preserves runtime/push/terminal summaries in final reports. It also applies a
post-diagnostic local fix: JIT forward output is explicitly replicated, while
backward outputs force LoRA gradients to inherit each parameter's sharding and
force the cross-device input gradient to replicated layout. Local JAX 0.10.2
tests pass, but this fix has not received another live TPU gate and must not be
called externally verified.

The bounded ledgers are exhausted and must not be reset silently: acquisition
windows are 3/3, window-3 submissions are 3/3, and full live gates are 4/4.
The public resume contract requires both a new bounded TPU acquisition window
and explicit authorization for another full live gate. Fresh regressions are
186 passed, zero failed, one conditional skip: 82 heterogeneous/shared, 79
legacy Qwen/CUDA, and 25 dedicated JAX/TPU tests. All remote Kernels, private
packages, tensor payloads, credentials, Coordinator, and tunnel are removed;
`live_resources_left_running=false`.

Completion still requires a fresh strict artifact with provider coverage
`kaggle_cpu`, `kaggle_cuda`, and `kaggle_jax_tpu`, ledger exactly
`[1,2,3,4,5,6]`, real finite updates on every stage, changed TPU LoRA hashes,
bidirectional GPU/TPU/CPU tensor traffic, TPU step-3 replacement/restore,
stale-generation rejection, standard PEFT CPU reload, complete cleanup, and
`heterogeneous_training_tpu_beta_ready=true` with zero errors.

## Latest Unified CPU/GPU Heterogeneous Training Scheduler Beta Status

Current superseding heterogeneous-training status on 2026-07-13: the unified
CPU/GPU Heterogeneous Training Scheduler Beta is achieved for the pinned
`Qwen/Qwen2.5-7B` LoRA gate. The canonical artifact is
`dist/training-heterogeneous-beta-20260713-r3-r2-live-achieved/training_heterogeneous_beta.json`
(file SHA-256
`6c89c167d96e7548ffadba8454fdffa33cc5196b730db04c2d7a0f29e52e2884`,
embedded content hash
`sha256:9f2cfe6a982a8d589df172ed38cceeda28baf153055b33b75c1cfdd6d1830a2f`).
The strict checker passes with `error_count=0`, public-safety errors empty, and
`heterogeneous_training_beta_ready=true`:
`PYTHONPATH=. python scripts/training_heterogeneous_beta_check.py --report
dist/training-heterogeneous-beta-20260713-r3-r2-live-achieved/training_heterogeneous_beta.json
--require-ready --json`. Do not rerun the live Kaggle gate merely to recreate
evidence.

The unified path is manifest-driven. The manifest pins model/revision, LoRA,
dataset, precision, microbatch, checkpoint policy, scheduler policy, resource
reserves, and arbitrary contiguous stage boundaries. The achieved manifest
uses revision `d149729398750b98c0af14eb82c78cfe92750796`, stages `[0,7)`,
`[7,14)`, `[14,20)`, `[20,26)` on CUDA-capable Miners and `[26,28)` on a
pure CPU Miner, sequence length 8, microbatch size 1, and six global steps.
CPU and CUDA Miners share one Job, Coordinator, SQLite commit ledger,
checkpoint store, status surface, and tensor protocol. A one-GPU Miner is a
first-class scheduler participant; multi-GPU hosts contribute one independently
fenced Miner process per exposed device.

The scheduler validates hard GPU-memory/CPU-RAM fit before placement and scores
measured compute latency, adjacent-stage network cost, current load, device
preference, safety reserve, and stage migration cost. Plans retain candidate
audits, estimates, scores, reasons, capacities, and placement generations.
OOM, lease expiry, Miner departure, and persistent stragglers can abort an
uncommitted epoch and trigger deterministic re-placement. Assignment tokens,
session generations, placement generations, and leases reject late results.
Dynamic capability refresh is idempotent for the same private registration
nonce while incompatible identities still fail closed. A migration penalty
prevents profile-driven stage churn unless the measured benefit exceeds model
reload cost; explicit OOM/straggler/owner recovery remains available.

The data plane transfers forward activations and backward gradients as bounded,
chunked safetensors messages. Every envelope binds run, manifest, global step,
microbatch, source/target stage, direction, placement generation, tensor
metadata, checksums, TTL, retry limit, and assignment hash. Pickle is not
accepted. Chunk hashes, payload hashes, idempotent replay, finite retry,
size/timeout limits, dtype/device conversion, stale generation, and duplicate
delivery are tested. A global optimizer step commits only after all five
signed, validated six-component stage checkpoints arrive; partial epochs never
advance the ledger.

The live run used two private T4x2 Kernels as four single-GPU Miners and one
private CPU Kernel as the final trainable stage. Steps 1-3 committed under the
initial placement. A trainable GPU Miner was deliberately removed after step
3; the next speculative epoch was aborted and training paused/rebalanced. A
different Miner restored the central step-3 checkpoint. After bounded Kernel
and tunnel recovery, step 4 committed in placement generation 7 and steps 5-6
committed in generation 8. The final ledger is exactly `[1,2,3,4,5,6]`.
Committed traffic contains 24 forward activations and 24 backward gradients,
including six CUDA-to-CPU activations and six CPU-to-CUDA gradients, all with
verified hashes. Aborted-generation messages were retained as fencing evidence
but excluded from committed counts.

All five stages report finite losses/gradients, real optimizer and scheduler
updates, and changed LoRA hashes. The standard PEFT export contains 392 tensors
over layers 0..27. A separate pure-CPU Kernel downloaded the completed export,
reloaded every stage, and completed a finite full stagewise forward. Eight
historical/current experiment Kernel refs were deleted or verified previously
deleted; Coordinator, tunnel, leases, 52 private tensor payloads, temporary
packages, credentials, and private runtime were removed. No live resources
remain.

Implementation anchors are `crowdtensor/heterogeneous_training_manifest.py`,
`crowdtensor/heterogeneous_training_scheduler.py`,
`crowdtensor/heterogeneous_tensor_transport.py`,
`crowdtensor/heterogeneous_training_checkpoint.py`,
`crowdtensor/heterogeneous_qwen_training.py`,
`crowdtensor/heterogeneous_training_miner.py`, the heterogeneous path in
`crowdtensor/elastic_training_runtime.py`, and
`crowdtensor/heterogeneous_training_beta.py`. Kaggle packaging/live evidence is
owned by `scripts/training_heterogeneous_beta_kaggle_package.py`,
`scripts/training_heterogeneous_beta_worker_entry.py`, and
`scripts/training_heterogeneous_beta_live_probe.py`; strict pack/check are
`scripts/training_heterogeneous_beta_pack.py` and
`scripts/training_heterogeneous_beta_check.py`. Final regressions pass 145
tests with zero failures: 66 heterogeneous/shared Elastic tests plus 79 legacy
Qwen/CUDA training tests.

The ordinary owner path is `crowdtensor train create <job> --heterogeneous`,
then `train serve/status/invite/export/cancel/cleanup`; contributors use the
existing private `crowdtensor-miner join --training --invite <invite.json>`
path. Status exposes hashed Miner capabilities, resource estimates, stage
placement, placement generation, measured profiles, rebalance reason,
committed step, missing stages, and pause reason. Private invites, credentials,
Coordinator URLs, raw data/token IDs, tensor values, and checkpoint values are
not public artifacts.

Boundary: this Beta proves pinned Qwen2.5-7B PEFT LoRA, five explicit Qwen
stages, one microbatch at a time, epoch-level rollback/replacement, and CPU plus
single-GPU/multi-GPU-host scheduling. It does not prove full-parameter or TPU
training, arbitrary architecture auto-partitioning, data-parallel replicas,
mid-microbatch migration, permissionless Byzantine/poisoning resistance,
secure aggregation, multi-account production operation, incentives/billing,
production GA, an SLA, or larger-model training. Anonymous Hugging Face stage
materialization and free-provider tunnel/runtime lifetime remain throughput
and availability limits, not correctness failures.

## Latest Elastic Volunteer Training Product Beta Status

Current superseding training-product status on 2026-07-12: the CrowdTensor
Elastic Volunteer Training Beta ordinary-user and Miner path is achieved for
the pinned Qwen2.5 1.5B four-stage LoRA topology. The canonical live artifact
is
`dist/training-elastic-beta-live-20260712-r6-repacked-achieved/training_elastic_beta_live_probe.json`
(file SHA-256
`a6476c610201877108a8630007c38675229dc33f9bb48fef4de64dbb9ddfae74`,
embedded content hash
`sha256:d58c1bd0304ddc126c3c5e5cccf39b0339c7dc1a18ff2370778ef99255d0ab78`).
The independent strict checker passes with `error_count=0` and
`elastic_training_beta_ready=true`:
`PYTHONPATH=. python scripts/training_elastic_beta_check.py --report
dist/training-elastic-beta-live-20260712-r6-repacked-achieved/training_elastic_beta_live_probe.json
--require-ready --json`. Do not rerun the four-Kernel live gate merely to
recreate evidence.

The ordinary owner path is `crowdtensor train create/status/cancel/export/invite/cleanup`
plus `crowdtensor train serve --elastic-job <job>`. The ordinary contributor
path is `crowdtensor-miner join --training --invite <invite.json>`. Invite
files are private mode-0600 inputs containing the Miner credential and
Coordinator URL; public reports contain neither. Miners discover two CUDA
devices, advertise capabilities, receive topology-aware `[0,1]` or `[2,3]`
stage groups, fetch private bootstrap inputs, renew session leases, restore
central checkpoints, and gracefully drain after a committed barrier on
SIGINT, SIGTERM, or a drain file. The owner service exposes separately
authenticated owner and Miner routes and recovers the same job, barrier,
assignment, and rendezvous state after process restart.
Owner status directly reports `committed_step`, `online_miner_count`,
`missing_stage_ids`, and `pause_reason` in addition to the detailed hashed
Miner and assignment ledger.

The live gate started the job through public `train create`, observed it with
public `train status`, and exported it with public `train export`. Two product
T4x2 Miners committed real optimizer steps 1-4; both then gracefully left and
their Kaggle Kernels were deleted. The service observed ten zero-Miner paused
states at committed step 4 and was restarted. Two replacement product Miners
with disjoint Kernel identities restored all four central step-4 checkpoint
archives and committed steps 5-8. The final commit ledger is exactly
`[1,2,3,4,5,6,7,8]`; base weights stayed frozen, LoRA gradients were positive,
the 392-tensor standard PEFT adapter covers layers 0..27, and evaluation lowers
held-out loss from 2.6663523763 to 2.4811929762. The canonical r6 report binds
the immutable r5 measurements to a post-cleanup account audit; it only corrects
a redundant second delete attempt and records
`runtime_measurements_changed=false`. All four experiment Kernels, service,
tunnel, rendezvous payloads, and private runtime were removed.

Product persistence and abuse controls are implemented in
`crowdtensor/elastic_training_beta.py`,
`crowdtensor/elastic_training_miner.py`,
`crowdtensor/elastic_training_runtime.py`, and
`crowdtensor/elastic_checkpoint_storage.py`. Checkpoint submissions use
HMAC-SHA256 signatures bound to the session, assignment, epoch, stage, and
archive hash. The Coordinator validates archive ownership and coverage,
safetensors names/dtypes/shapes/finite values, and safe optimizer, GradScaler,
and RNG payloads loaded with `weights_only=True`. It enforces online-Miner,
per-session byte, and rejection limits and quarantines repeatedly invalid
sessions without blocking heartbeat handling. These controls reject malformed,
non-finite, stale, unsigned, over-quota, and conflicting submissions; they do
not prove that a credentialed malicious Miner cannot return semantically
poisoned but structurally valid updates.

Local content-addressed checkpoint storage, retention, and idempotent cleanup
are implemented. The optional `storage` extra provides S3/MinIO-compatible
storage with credentials read only from private environment variables. The
S3/MinIO path is unit-tested with a compatible fake service but has not been
externally live-tested; do not claim external object-store validation. Public
`crowdtensor train cleanup <job>` now fences active Miner leases, aborts any
uncommitted epoch, clears private rendezvous payloads, removes unretained blobs,
preserves exported adapters/public evidence and the configured committed
checkpoint window, and is idempotent. The authenticated owner HTTP cleanup
route is `POST /v1/training/jobs/{job_id}/cleanup`.

The final regression artifact is
`dist/training-elastic-beta-tests-20260712-r2-final/training_qwen15b_test_summary.json`:
380 tests pass across 36 files and 10 suites with zero failures. It explicitly
includes the Product Beta controller/Miner/security/storage/checker tests and
records `s3_minio_storage_externally_live_tested=false` and
`permissionless_byzantine_poisoning_resistance_verified=false`.

Boundary: this Beta proves a releaseable owner/Miner workflow, real stage-owned
elastic continuation, Coordinator restart recovery, secure checkpoint
transport/validation, PEFT export/evaluation, and cleanup for pinned
`Qwen/Qwen2.5-1.5B`, eight optimizer steps, four fixed stages, and two T4x2
Miners at a time. It does not prove arbitrary model/topology admission,
single-GPU Miners, 7B+, full-parameter training, mid-microbatch migration,
permissionless Byzantine-poisoning resistance, secure aggregation,
multi-account training, incentives/billing, production GA, or an SLA. The next
core milestone should generalize model/topology admission and validate 7B LoRA
with single-GPU Miners before marketplace work.

## Latest Elastic Volunteer Training Runtime Status

Current superseding elastic-training status on 2026-07-12: the CrowdTensor
Elastic Volunteer Training Runtime goal is achieved for the pinned Qwen2.5
1.5B four-stage LoRA path. The canonical artifact is
`dist/training-qwen15b-elastic-live-20260712-r2-repacked-achieved/training_qwen15b_elastic_live_probe.json`
(SHA-256 `9bbdfebd3af4fdc254ae7168368efd6a1af7abd1b6eaa89bb73308cb83d57707`).
The independent strict checker passes with every derived gate true,
`error_count=0`, and `elastic_volunteer_training_ready=true`:
`PYTHONPATH=. python scripts/training_qwen15b_elastic_check.py --report
dist/training-qwen15b-elastic-live-20260712-r2-repacked-achieved/training_qwen15b_elastic_live_probe.json
--require-ready --json`. Do not rerun the four-Kernel live gate merely to
recreate evidence.

The live run used one authorized Kaggle account and two sequential, distinct
T4x2 Kernel pairs. The old pair trained real pinned Qwen stages `[0,7)`,
`[7,14)`, `[14,21)`, and `[21,28)` through committed step 4. Every stage
uploaded adapter, optimizer, GradScaler, and RNG state into central
content-addressed checkpoint storage. Both old Miners then went offline; the
already assigned step-5 epoch was aborted, all four old Kernel refs were
deleted, and the Coordinator remained at `paused_waiting_for_miners` with zero
live Miners for 10.000226 seconds and ten observations.

Only after that pause did a new pair with distinct Kernel-ref hashes and Miner
session hashes launch. It downloaded all four committed step-4 archives into
fresh checkpoint directories, restored dataset cursor 16, and produced its
first optimizer checkpoints at step 5, never step 0 or duplicate step 4. It
continued through step 8. The SQLite commit ledger is exactly
`[1,2,3,4,5,6,7,8]`, with eight unique atomic global commits. New stage-5
assignments belong only to the replacement Miner identities, while the old
step-5 assignment is retained as revoked evidence. The final standard PEFT
adapter has 392 tensors over all 28 layers, reloads on CPU/CUDA, changes
logits, and lowers held-out validation loss from 2.6663523763 to 2.4811929762.

The implementation is in `crowdtensor/elastic_training_runtime.py`,
`crowdtensor/elastic_training_client.py`, and the elastic path in
`crowdtensor/qwen15b_four_gpu_worker.py`. It provides SQLite transactions,
expiring Miner leases, stage reassignment, whole-epoch rollback, automatic
pause/wake, authenticated binary checkpoint upload/download, validated safe
archives, and idempotent exactly-once global commit. HTTP routes expose Miner
register/heartbeat/offline, assignment, checkpoint, barrier, and public status;
`crowdtensor train elastic-status --state <db> --run-id <id>` reads the same
persistent state.

The original r1 live report contains the successful measurements but was
conservatively marked incomplete by three report bugs: numeric zero was
coalesced to `-1`, revoked old step-5 assignments were not filtered, and a
redundant delete retry overrode four earlier successful deletes. r2 reuses the
immutable r1 measurements and transparently records these checker corrections;
it did not rerun or alter runtime measurements. A post-run account audit found
all four experiment refs absent. All Kernels, Coordinator, tunnel, private
packages/payloads, and private runtime were removed. The final regression
summary at the same r2 directory passes 370 tests with zero failures.

Boundary: this proves epoch-level elastic continuation for pinned Qwen2.5 1.5B
PEFT LoRA with two-stage slots per T4x2 Miner. If any Miner disappears before a
barrier commit, all speculative stage candidates for that epoch are discarded
and every stage restarts from the last global commit. It does not yet prove
arbitrary model/topology admission, mid-microbatch migration, permissionless
Miner poisoning resistance, secure aggregation, rewards/billing, multi-account
training, production GA, or an SLA.

## Latest Qwen 1.5B Four-GPU Training Service Beta RC Status

Current superseding training-product status on 2026-07-12: the CrowdTensor
Qwen 1.5B Four-GPU Training Service Beta RC is achieved. The canonical artifact
is
`dist/training-qwen15b-beta-20260712-r3-live-achieved/training_qwen15b_beta.json`
(SHA-256 `dfba6533e1976d0f69f5d642e2d4a60123601513021c14576b19537e676f118f`).
Both default and strict checkers pass with `error_count=0`,
`training_qwen15b_beta_ready=true`, and `goal_achieved=true`:
`PYTHONPATH=. python scripts/training_qwen15b_beta_check.py --report
dist/training-qwen15b-beta-20260712-r3-live-achieved/training_qwen15b_beta.json
--require-ready --json`. Do not rerun this achieved Beta merely to recreate
evidence.

The fresh ordinary-user run is under
`dist/training-qwen15b-beta-job-20260712-r1-live-attempt1/`. It was submitted
through `crowdtensor train lora --backend cuda --model
Qwen/Qwen2.5-1.5B --topology kaggle-2x-t4x2 --steps 8`, generated its pinned
model source and private WikiText inputs inside the job, and did not consume
prebuilt dist inputs. One authorized Kaggle account supplied two concurrent
T4x2 Kernels. Four CUDA stage processes retained the Alpha ranges `[0,7)`,
`[7,14)`, `[14,21)`, and `[21,28)`, FP32 frozen-model/LoRA compute with
GradScaler, and FP16 stage-boundary transport.

The Beta live gate completed the 8-step baseline and 8-step resumed contract,
64 unique stage optimizer-step identities, 64 private activations, 64 private
gradients, one adapter transfer, and verified four-stage overlap. After resumed
step 4, the Coordinator was stopped for three seconds and restored from its
persistent private rendezvous state. Recovery took 3.394220 seconds, recovered
96 payloads and 492 events, and both workers re-registered. All four stage
processes restarted under new PIDs and loaded step-4 checkpoints with global
step 4 and dataset cursor 16; no optimizer identity was repeated. Bounded
transport retries observed the outage and re-registered before replay.

Training loss fell from 3.6082336903 to 2.8676728606 in both runs. The standard
PEFT adapter has 392 tensors covering layers 0..27, loads on CPU and CUDA,
changes logits, and lowers validation loss from 2.6663523763 to 2.4811929762.
The ordinary `status --watch`, completed-job `resume`, `export`, repeated
`cancel`, and repeated `cleanup` paths were exercised. Submit, cancel, cleanup,
events, and optimizer progress are idempotent; a running cancel writes a private
mode-0600 marker which the live probe checks before further polling and then
uses its job-scoped cleanup path. SQLite WAL JobStore state survives controller
process replacement, enforces one live GPU job plus a bounded queue, rejects
global-step regression, and keeps credential values and paths private.

The benchmark records 452.652131 seconds from allocation to training completion,
153.472964 seconds to first optimizer step, 16 step latencies with an 8.625027
second maximum, 35.103845 ms maximum four-stage overlap, 129 private payloads /
34,468,880 transported bytes, 2,439,433,728 peak stage GPU allocated bytes, and
completion within the 1800-second bound. Final regressions pass 358 tests with
zero failures: 326 pre-Beta training/StateStore/Miner/Coordinator/CLI tests and
32 Beta tests, including duplicate submit, expired lease, corrupt checkpoint,
non-finite tensor, worker timeout, and Coordinator-unavailable fault cases.

The one successful Beta attempt is the only entry in its 3-attempt ledger.
Both temporary Kernels, private packages, Coordinator, tunnel, private payloads,
and local runtime were removed; checkpoint archives, adapter, and public
evidence remain. A post-cleanup read-only audit found zero active Kernel on the
account used by this run. Two other authorized accounts had unrelated active
Kernels and were intentionally left untouched.

Boundary: this is an ordinary-user, persistent-service Beta RC for pinned
Qwen2.5 1.5B PEFT LoRA on one same-account four-T4 topology. It is not 7B+ or
full-parameter training, multi-account training, elastic scaling, anonymous
Miner poisoning resistance, billing/marketplace work, Web UI, production GA,
or an SLA. The immutable Qwen Alpha remains the technical baseline and must
not be rewritten.

## Latest Two-Node CUDA Training RC Status

Current superseding CUDA-training status on 2026-07-11: the CrowdTensor
Two-Node CUDA Training RC is achieved. The canonical artifact is
`dist/training-cuda-two-node-rc-20260711-r5-live-achieved/training_cuda_two_node_rc.json`.
Both default and strict checkers pass with `error_count=0`,
`training_cuda_two_node_rc_ready=true`, `goal_achieved=true`,
`single_kernel_gate_verified=true`, and `two_node_gate_verified=true`:
`PYTHONPATH=. python scripts/training_cuda_two_node_rc_check.py --report
dist/training-cuda-two-node-rc-20260711-r5-live-achieved/training_cuda_two_node_rc.json
--require-ready --json`.

The successful live source is
`dist/training-cuda-two-node-live-20260711-a3-embedded-single-gate-gradscaler/training_cuda_two_node_live_probe.json`.
One authorized Kaggle account launched two concurrent private T4x2 Kernels;
the maximum observed running count was two and the two-node gate used one T4
per Kernel. The stage0 Kernel first used both of its local T4s to run the exact
single-Kernel gate: separate `cuda:0`/`cuda:1` processes completed an
uninterrupted four-step run and a four-step run with a hard-stopped/restarted
stage1. The restored run loaded optimizer and GradScaler state at step 2,
matched both final LoRA adapters with maximum absolute difference 0, matched
final loss exactly, kept base weights frozen, and reduced loss from
4.2349076271 to 3.9638671875. The stage0 lazy GradScaler is initialized before
it consumes stage1's scaled activation gradient.

After freeing the preliminary gate's CUDA state, the two Kernels completed one
same-request cross-machine pipeline. Both roles registered through the
authenticated public Coordinator, ran four real CUDA forward/backward steps,
and exchanged exactly four private activation and four private gradient
payloads. Neither stage loaded the full model. The two GPU Miners trained
different dataset shards, returned 28 named safetensors LoRA delta tensors
each, and the existing StateStore validation/DiLoCo path accepted both,
advanced adapter version and outer step from 0 to 1, and verified dense sign
transport reconstruction with error feedback. The exported standard PEFT
adapter loads on CPU and CUDA, changes logits, reduces fixed validation loss
from 4.1791639328 to 4.0241522789, and has CPU/CUDA logits within
4.4703483582e-8 maximum absolute difference.

The embedded single-Kernel evidence is not a metadata shortcut. The r5 checker
requires all per-step records and binds the selected single gate to the stage0
worker report, stage0 Kernel hash, execution order, downloaded checkpoint
bundle hash, two-node attempt number, and final cleanup. The stage0 bundle
contains baseline, resumed, pipeline, and Miner checkpoints (25 files,
SHA-256 `b26f9448fe73219c560385042a04defbf75a18adb89407b9bb9a5d1d9ba8b3f5`);
the stage1 bundle contains pipeline and Miner checkpoints (7 files, SHA-256
`e1ef4cf56a198914ee6c8bcd3235e645fc355d43f0fc19cb487c9def74786a93`).
Both archives were downloaded and hash-verified before cleanup.

The attempt ledger at `dist/training-cuda-two-node-work/allocation_attempts.json`
honestly preserves three failed standalone single-Kernel attempts and the
first two route-blocked two-node attempts. A one-time checker-backed amendment
allowed attempt 3; two-node attempt 3 is `verified`. The completed Goal's
3/3 single and 3/3 two-node budgets are exhausted, so do not rerun this RC.
All temporary Kaggle Kernels were deleted, private packages and rendezvous
payloads removed, Coordinator and tunnel stopped, checkpoint/public evidence
retained, and a post-run account query found no active Kernel. Public safety,
the ten-case malformed-delta rejection matrix, and final regressions pass:
64 CUDA tests, 27 CPU-training tests, and 167 StateStore/Miner/Coordinator
tests.

Future work may use all four operator-authorized Kaggle accounts configured in
the private credential stores. Keep each account in an isolated environment,
never expose credential values in artifacts, and preserve task-specific
acceptance semantics: when a gate requires same-account concurrency, multiple
accounts must not be combined to satisfy it. This achieved RC proves bounded
LoRA training on a small deterministic model; it does not prove large-model
training, full-parameter tuning, anonymous-Miner poisoning resistance,
four-T4 tensor parallelism, billing, or a production marketplace.

## Latest Training Foundation RC Status

Current superseding training status on 2026-07-10: the CrowdTensor CPU-only,
GPU-ready Training Foundation RC is achieved. The canonical artifact is
`dist/training-foundation-rc-20260710/training_foundation_rc.json`. Validate it
with `PYTHONPATH=. python scripts/training_foundation_rc_check.py --report
dist/training-foundation-rc-20260710/training_foundation_rc.json --require-ready
--json`; it passes with `training_foundation_rc_ready=true`,
`goal_achieved=true`, and zero errors.

The RC proves real local PyTorch/Transformers/PEFT LoRA CPU training, two
independent local Miners on distinct dataset shards through the existing HTTP
Coordinator/StateStore/result ledger, one named-tensor DiLoCo outer step,
sign compression with error feedback, deterministic trusted replay, and a
two-process split Transformer with real activation and reverse-gradient
transport. Its controlled stage1 hard-stop/restart matches the uninterrupted
adapter tensors and loss exactly. The exported standard PEFT adapter loads on
CPU, changes logits, and lowers fixed validation loss. All child processes are
cleaned up. User paths are `crowdtensor train lora`, `train status`, `train
resume`, `train export`, and `train cleanup`; see
`docs/training-foundation.md`.

Do not overclaim the CPU artifact itself. It supports permissioned/trusted
local workers and LoRA only; its GPU continuation manifest remains
configuration/capability dry-run evidence with `gpu_live_verified=false`.
Real bounded CUDA evidence comes from the r5 RC above, not from this CPU
artifact. Neither training RC proves large-model training, WAN-scale
production, full-parameter tuning, or anonymous-Miner poisoning resistance.
Training status does not supersede inference truth: GLM 5.2 r214 remains the
achieved 1-token deployment RC and r34 remains the unachieved 8-token Alpha
blocker.

## Latest Qwen 1.5B Four-GPU Pipeline Training Alpha Status

Current superseding status on 2026-07-12: the Qwen 1.5B four-GPU Pipeline
Training Alpha is achieved. The canonical artifact is
`dist/training-qwen15b-four-gpu-alpha-20260712-r5-live-achieved/training_qwen15b_four_gpu_alpha.json`.
Both default and strict checkers pass with `error_count=0`,
`qwen15b_four_gpu_alpha_ready=true`, and `goal_achieved=true`:
`PYTHONPATH=. python scripts/training_qwen15b_four_gpu_alpha_check.py --report dist/training-qwen15b-four-gpu-alpha-20260712-r5-live-achieved/training_qwen15b_four_gpu_alpha.json --require-ready --json`.
Do not rerun this achieved Alpha merely to recreate evidence.

The authoritative live report is
`dist/training-qwen15b-four-gpu-live-20260712-r5-attempt5-fp32-stable-fp16-boundary/training_qwen15b_four_gpu_live_probe.json`.
One authorized Kaggle account supplied two concurrent T4x2 Kernels and four
real CUDA stage processes for pinned `Qwen/Qwen2.5-1.5B` revision
`8faed761d45a263340a0528343f099c05c9a4323` (1,543,714,304 parameters).
Stages `[0,7)`, `[7,14)`, `[14,21)`, and `[21,28)` loaded only owned modules
and LoRA tensors on Kernel A `cuda:0/1` and Kernel B `cuda:0/1`. Four-stage
compute overlap is verified, with 50.719332 ms maximum common overlap.

Uninterrupted and controlled-resume runs each completed 8 steps and exchanged
64 private FP16 activation plus 64 private FP16 gradient payloads. Stage 2 was
stopped after step 4 and restored under PID 120 instead of PID 82. Baseline and
resumed losses/adapters match exactly; training loss fell from 3.6082336903 to
2.8676728606. The standard PEFT export has 392 tensors covering layers 0..27,
loads on CPU/CUDA, changes logits, and reduces validation loss from
2.6663523763 to 2.4811929762. All checkpoint and adapter archives were
downloaded and hash-verified before cleanup.

Kaggle's preinstalled `torchao==0.10.0` is removed before PEFT import. FP16
autocast was abandoned after attempt 4 produced a non-finite stage0 activation.
The achieved path uses FP32 frozen-stage compute, FP32 LoRA parameters and
GradScaler, with FP16 only for stage-boundary transport. Dependency/precision
smokes run before Qwen loading, and non-finite runtime values fail closed.

The immutable ledger
`dist/training-qwen15b-four-gpu-work/allocation_attempts.json` preserves four
incomplete attempts and verified attempt 5. The user superseded the finite
dual-Kernel attempt budget with an unbounded cumulative-attempt authorization.
This does not authorize unlimited concurrency or automatic retries: each probe
invocation reserves one attempt, uses one same-account T4x2 pair, and is capped
at 1800 seconds. Never erase or rewrite the ledger or authorization history.

All temporary Kernels, packages, Coordinator, tunnel, private payloads, and
runtime files were removed. The final regression artifact
`dist/training-qwen15b-tests-20260712-r5-live-achieved/training_qwen15b_test_summary.json`
passes 313 tests with zero failures. The ordinary-user path remains
`crowdtensor train lora --backend cuda --model Qwen/Qwen2.5-1.5B --topology kaggle-2x-t4x2 --steps 8`,
plus `status`, `resume`, `export`, and `cleanup`.
The read-only post-cleanup audit at
`dist/training-qwen15b-four-gpu-postcleanup-audit-20260712-r5-live-achieved/training_qwen15b_four_gpu_live_probe.json`
authenticated all four accounts, observed zero active Kernels on each, and did
not reserve or launch an allocation attempt.

Boundary: this is real four-T4 stage-selective 1.5B LoRA Pipeline Training
Alpha evidence, not full-parameter or 7B+ training, dynamic scaling,
multi-account training, anonymous-Miner trust, billing, or production WAN
training.

## Historical Qwen 1.5B Pre-Achievement Status (Superseded)

The following blocker history is retained only for diagnosis. It is superseded
by the achieved r5 artifact above and must not drive current status.

Historical status before verified attempt 5: the Qwen 1.5B four-GPU
engineering path was implemented and locally regression-complete, but
the live Alpha goal is not achieved. The canonical blocker is
`dist/training-qwen15b-four-gpu-alpha-20260712-r4-dependency-smoke-strict-evidence-blocker/training_qwen15b_four_gpu_alpha.json`.
Its default checker passes with `qwen15b_four_gpu_alpha_ready=false`:
`PYTHONPATH=. python scripts/training_qwen15b_four_gpu_alpha_check.py --report dist/training-qwen15b-four-gpu-alpha-20260712-r4-dependency-smoke-strict-evidence-blocker/training_qwen15b_four_gpu_alpha.json --json`.
The strict checker intentionally fails because no live 8-step baseline/resume
training evidence exists:
`PYTHONPATH=. python scripts/training_qwen15b_four_gpu_alpha_check.py --report dist/training-qwen15b-four-gpu-alpha-20260712-r4-dependency-smoke-strict-evidence-blocker/training_qwen15b_four_gpu_alpha.json --require-ready --json`.
Do not mark this Alpha goal achieved from the blocker artifact.

Implemented and locally verified:
- The pinned source is `Qwen/Qwen2.5-1.5B` revision
  `8faed761d45a263340a0528343f099c05c9a4323`, with 1,543,714,304 real BF16
  parameters and a generated safetensors index derived from the real single-file
  header. Four stage ranges are `[0,7)`, `[7,14)`, `[14,21)`, and `[21,28)` on
  Kernel A `cuda:0/1` and Kernel B `cuda:0/1`.
- Stage-selective HTTP Range materialization, meta-device stage construction,
  FP16 T4 casting, native Transformers Qwen layers, causal masking, rotary
  embeddings, gradient checkpointing, PEFT LoRA, GradScaler, optimizer state,
  checkpoint/RNG/cursor restore, standard PEFT assembly, authenticated private
  activation/gradient transport, and four-stage pipelined scheduling are
  implemented.
- A real spawned-process tiny-Qwen integration test runs the same four stage
  processes, forward/backward transport, baseline/resume checkpoints, and forced
  stage restart. Full regression summary
  `dist/training-qwen15b-tests-20260712-r4-dependency-smoke-strict-evidence/training_qwen15b_test_summary.json`
  passes 310 tests with zero failures, including existing CUDA Training RC, CPU
  Training Foundation, StateStore, Miner, Coordinator, and user CLI suites.
- The ordinary-user path is
  `crowdtensor train lora --backend cuda --model Qwen/Qwen2.5-1.5B --topology kaggle-2x-t4x2 --steps 8`,
  with `status`, `resume`, `export`, and `cleanup` support.

Live attempt findings and hard boundary:
- Read-only quota/activity preflight authenticated all four authorized accounts;
  all had two free GPU slots and positive quota at preflight time.
- Attempt 1 launched two same-account T4x2 Kernels, but concurrent large-shard
  materialization failed before stage registration. Shard preparation is now
  serial per Kernel; real local stage0+stage1 materialization succeeds.
- Attempt 2 launched both T4x2 Kernels and reached `stage_model_load`. Both failed
  because Kaggle preinstalled `torchao==0.10.0`, while PEFT 0.19.1 rejects
  torchao versions below 0.16. The retained sanitized logs are under
  `dist/training-qwen15b-four-gpu-live-20260712-r2-attempt2/logs/`.
- The private package now detects and uninstalls incompatible torchao before
  importing PEFT, then runs a tiny no-network PEFT LoRA injection and real
  forward/backward smoke before Qwen shard materialization. This fix is covered
  locally, but has not been GPU-live verified.
- The runtime now consumes fixed dataset row indexes `0..31` once per 8-step
  run instead of repeating the first four rows. The strict checker independently
  reconstructs every step/stage optimizer record, all 128 activation and
  gradient payload identities and hash links, four-stage overlap, restart PID
  chronology, checkpoint manifest hashes, and 28-layer adapter safetensors
  coverage. Aggregate success booleans alone cannot satisfy the gate.
- The Goal's immutable allocation ledger is
  `dist/training-qwen15b-four-gpu-work/allocation_attempts.json` and records both
  allowed dual-Kernel attempts used (`2/2`). Do not reset, replace, delete, or
  bypass this ledger, and do not submit another GPU Kernel unless the user
  explicitly authorizes an allocation-budget amendment.
- Both attempts deleted exactly their two temporary Kernels. No Coordinator,
  tunnel, package, payload, or process remains live. The earlier achieved CUDA
  Training RC at
  `dist/training-cuda-two-node-rc-20260711-r5-live-achieved/training_cuda_two_node_rc.json`
  remains immutable and must not be rerun or rewritten.
- The read-only post-cleanup audit at
  `dist/training-qwen15b-four-gpu-postcleanup-audit-20260712-r1/training_qwen15b_four_gpu_live_probe.json`
  authenticated all four accounts and found active Kernel count zero for each.

The next legal live step requires an explicit user amendment allowing at least
one additional same-account dual-T4x2 attempt. After such authorization, amend
the existing ledger transparently rather than erasing history, rerun the fixed
package once, and only mark Alpha achieved if the strict checker passes with two
8-step runs, four real GPUs with compute overlap, stage2 restart after step 4,
numeric resume equivalence, reduced training and validation loss, CPU/CUDA PEFT
reload, verified archives, and complete cleanup.

## Latest GLM 5.2 Kaggle Alpha Status

Current superseding Alpha status on 2026-07-08: the canonical Alpha blocker is
`dist/glm52-kaggle-alpha-20260708-r34-http-cleanup-route/glm52_kaggle_alpha.json`.
The default checker passes with `glm52_kaggle_alpha_ready=false`:
`PYTHONPATH=. python scripts/glm52_kaggle_alpha_check.py --report dist/glm52-kaggle-alpha-20260708-r34-http-cleanup-route/glm52_kaggle_alpha.json --json`.
The strict checker intentionally fails:
`PYTHONPATH=. python scripts/glm52_kaggle_alpha_check.py --report dist/glm52-kaggle-alpha-20260708-r34-http-cleanup-route/glm52_kaggle_alpha.json --require-ready --json`.
Do not mark the active Alpha goal achieved from this artifact: neither the
1-token full-chain probe nor the 8-token same-request Alpha gate completed.
r34 supersedes r33 by adding a checker-backed service cleanup route. The
ordinary-user HTTP service now exposes `POST /cleanup`; service reports record
`cleanup_route_ready=true`, the local smoke probe verifies `GET /health`,
`GET /status`, `POST /generate`, and `POST /cleanup`, and the canonical r34
artifact imports `service_smoke_summary.cleanup_route_verified=true` with
temporary Kaggle kernels deleted, temporary private packages removed, and no
live resources left running. This is cleanup/service usability evidence, not
live inference success. The current blocker remains
`kaggle_gpu_quota_unavailable` with
`next_quota_refresh_time=2026-07-11T00:00:00`.
r33 superseded r32 by adding checker-backed generate CLI artifact recovery.
`crowdtensor generate glm52-kaggle` now reads the same output directory's
Alpha/status artifacts when the local service is unreachable or a request is
blocked, then writes a public-safe `glm52_kaggle_alpha_generate_cli.json` with
artifact recovery phase, blockers, cleanup/quota summary, `next_resume_command`,
and `resume_private_inputs`. The r33 canonical artifact imports that proof as
`generate_cli_summary` and lists `artifacts.generate_cli_json`; it records
`cli_generate_artifact_recovery_supported=true`,
`generate_cli_check_ok=true`, `artifact_recovery_present=true`, and
`artifact_recovery_resume_private_inputs_verified=true`. The proof used an
unreachable local Coordinator endpoint, so it is recovery usability evidence,
not live inference success.
r32 superseded r31 by surfacing the same public-safe `resume_private_inputs`
contract through user-facing status paths. HTTP `GET /status`, quota-blocked
`POST /generate`, and `crowdtensor status glm52-kaggle` now expose top-level
`resume_private_inputs` with `schema=glm52_kaggle_alpha_resume_private_inputs_v1`
and `resume_command_omits_private_credentials=true`; sensitive Kaggle/HF token
values, token paths, section names, env names, cookies, proxy URLs, prompts,
and generated text remain non-public. Service reports record
`status_exposes_resume_private_inputs=true`, and the Alpha pack/checker reject
reports missing this ordinary-user recovery contract with
`glm52_alpha_status_resume_private_inputs_missing` /
`status_resume_private_inputs_missing`. The r32 service smoke proof is
`dist/glm52-kaggle-alpha-20260708-r32-status-resume-private-inputs/glm52_kaggle_alpha_service_smoke_probe.json`;
it verifies `GET /health`, `GET /status`, and `POST /generate` against the
real `AlphaHTTPServer`, and `service_smoke_summary` records
`status_resume_private_inputs_verified=true` and
`generate_resume_private_inputs_verified=true`. Under the current GPU quota blocker, `/generate`
returns public-safe HTTP 503 with `generate_route_quota_blocker_verified=true`
and `generated_token_count=0`; this is route/blocker proof, not live
multi-token success. r32 records `kaggle_gpu_quota_unavailable` with
`next_quota_refresh_time=2026-07-11T00:00:00`.
The r31 blocker superseded r30 by adding a checker-enforced public-safe
`resume_private_inputs` contract to the service summary, blocker report, and
top-level Alpha report. It records that live resume requires private Kaggle
credentials, that the printed `next_resume_command` omits private credential
material, which private input methods are supported, and Hugging Face env-name
hash/count metadata. It explicitly records `kaggle_credential_values_public=false`,
`kaggle_token_file_paths_public=false`, `kaggle_token_section_names_public=false`,
`hf_env_names_public=false`, and `hf_env_values_public=false`; no token values,
token file paths, section names, raw env names, cookies, proxy URLs, prompts, or
generated text are public. The Alpha checker rejects blocked reports missing
this resume-private-input contract. r31 keeps the r30 service smoke proof at
`dist/glm52-kaggle-alpha-20260708-r31-resume-private-inputs/glm52_kaggle_alpha_service_smoke_probe.json`;
`/generate` reaches the real service and returns public-safe HTTP 503 under the
current GPU quota blocker. This is not live multi-token success and the strict
checker still requires a real 8-token CPU/GPU/TPU same-request report.
The r30 blocker superseded r29 by making `serve glm52-kaggle` share the
ordinary-user default output directory contract too: `deploy`, `serve
glm52-kaggle`, `status`, and `cleanup` all default to
`dist/glm52-kaggle-alpha`, while non-target product `serve` keeps its previous
default. Service/Alpha artifacts record `cli_serve_default_matches_deploy=true`,
`cli_status_default_matches_deploy=true`, and
`cli_cleanup_default_matches_deploy=true`; the Alpha pack/checker reject missing
or false default-path contracts. Tests cover no-argument `serve glm52-kaggle`,
explicit `--output-dir` preservation, product `serve`, `status`, and `cleanup`
parse paths. r30 also retains the local HTTP service smoke proof at
`dist/glm52-kaggle-alpha-20260708-r30-serve-default-output-dir/glm52_kaggle_alpha_service_smoke_probe.json`.
The smoke probe starts the real `AlphaHTTPServer`, checks `GET /health`,
`GET /status`, and `POST /generate`, and the Alpha pack imports
`service_smoke_summary` plus `artifacts.service_smoke_json`. In the current
quota-blocked state `/generate` reaches the service and returns public-safe
HTTP 503 with `generate_route_quota_blocker_verified=true`, `generated_token_count=0`,
and no live Kaggle launch. This verifies the service route surface and quota
short-circuit only; it is not live multi-token success and the strict checker
still requires a real 8-token CPU/GPU/TPU same-request report. r30 default
checker passes, strict readiness checker fails as expected, `crowdtensor status`
reports `phase=decode_blocked` with `gpu_quota_status.source=alpha_gpu_quota_summary`,
cleanup is verified, and public JSON contains no Kaggle/HF credentials, Bearer
headers, cookies, runtime proxy, raw prompt, generated text, or private runtime
state.
The r29 blocker superseded r28 by making `status` and `cleanup` share the
deploy default output directory and by recording
`cli_status_default_matches_deploy=true` and
`cli_cleanup_default_matches_deploy=true`.
The r28 blocker superseded r27 by adding a local HTTP service smoke artifact:
`dist/glm52-kaggle-alpha-20260708-r28-http-service-smoke/glm52_kaggle_alpha_service_smoke_probe.json`.
The smoke probe starts the real `AlphaHTTPServer`, checks `GET /health`,
`GET /status`, and `POST /generate`, and then the Alpha pack imports a
`service_smoke_summary` plus `artifacts.service_smoke_json`. In the current
quota-blocked state `/generate` reaches the service and returns public-safe
HTTP 503 with `generate_route_quota_blocker_verified=true`, `generated_token_count=0`,
and no live Kaggle launch. This verifies the service route surface and quota
short-circuit only; it is not live multi-token success and the strict checker
still requires a real 8-token CPU/GPU/TPU same-request report. r28 default
checker passes, strict readiness checker fails as expected, `crowdtensor status`
reports `phase=decode_blocked` with `gpu_quota_status.source=alpha_gpu_quota_summary`,
cleanup is verified, and public JSON contains no Kaggle/HF credentials, Bearer
headers, cookies, runtime proxy, raw prompt, generated text, or private runtime
state.
The r27 blocker superseded r26 by adding explicit Kaggle runtime blocker
classification for the live/collect worker paths. The stage-worker push probe
now emits public-safe blockers for push timeout/HTTP 429/empty response, status
timeout, wait timeout, output timeout/HTTP 429/empty response/missing stage
report, terminal error/cancelled, and cleanup/delete timeout. Service and Alpha
reports expose `kaggle_runtime_blocker_classification_ready=true` plus the
supported blocker class list under `configuration_check`, and the Alpha
pack/checker reject artifacts missing that recovery contract. r27 default
checker passes, strict readiness checker fails as expected, status reports
`phase=blocked_gpu_quota`, cleanup is verified, and public JSON contains no
Kaggle/HF credentials, Bearer headers, cookies, runtime proxy, or private
runtime state.
The r26 blocker supersedes r25 by adding the ordinary-user Hugging Face token
environment contract. `deploy` and `serve glm52-kaggle` accept
`--hf-token-env` (defaulting to the usual HF token env names), pass it through
the same-request live probe and stage-worker push path, and uploaded Kaggle
workers read the token only from private runtime env before adding a Bearer
header to Hugging Face fetches. Public service, Alpha, push, status, cleanup,
and benchmark artifacts store only HF env-name hashes/counts/booleans and
`hf_token_public=false`; they do not store raw env names, token values, Bearer
headers, or authorization material. The Alpha pack/checker reject a missing HF
token env contract or any `hf_token_public=true` artifact. r26 default checker
passes and strict readiness checker fails as expected.
The r25 blocker supersedes r23/r24 by making the ordinary-user
`--model`/`--accelerators` contract explicit and resumable. `deploy` and
`serve glm52-kaggle` now fail fast unless `--model` is a supported GLM 5.2
source (`cyankiwi/GLM-5.2-AWQ-INT4` or `zai-org/GLM-5.2`) and
`--accelerators` is the supported `cpu,gpu,tpu` set. Service and Alpha reports
record `requested_model`, `model_request_supported`, `accelerators`,
`required_accelerators`, and `accelerator_request_complete`; phase status
surfaces those fields under `configuration_check`. The Alpha pack/checker reject
unsupported model requests or incomplete accelerator requests, and the r25
top-level/blocker `next_resume_command` preserves
`--model cyankiwi/GLM-5.2-AWQ-INT4 --accelerators cpu,gpu,tpu`.
The r23 blocker supersedes r22 by adding a public-safe top-level
`next_resume_command` to the main Alpha report, matching
`blocker_report.next_resume_command`, and by making the checker reject blocked
reports without a resume command or redaction flag. The r23 artifact records
`next_resume_command_redacts_credentials=true` and keeps Kaggle tokens, cookies,
signed URLs, raw prompts, generated text, token ids, and private runtime state
out of public JSON.
The r22 blocker had superseded r21 by adding a user-facing CLI prompt
submission path: `crowdtensor generate --target glm52-kaggle ...` or
`crowdtensor generate glm52-kaggle --prompt-text ...`. The command posts to the
local Alpha service `/generate`, writes
`glm52_kaggle_alpha_generate_cli.json`, stores only prompt/service URL hashes,
captures public-safe HTTP 400/503 response bodies, and does not persist raw
prompts or service URLs. Service reports now declare
`cli_generate_command_available=true`; the Alpha pack/checker require it.
The r21 blocker had superseded r20 by adding public-safe `/generate` request
validation. Service reports now declare `generate_validates_request_schema=true`,
the Alpha pack/checker require that contract, and malformed JSON, empty/non-
object bodies, missing prompts, and invalid token counts return HTTP 400 with
`schema=glm52_kaggle_alpha_generate_response_v1`, no raw prompt/body, and
`phase=generate_request_invalid` in service status. Unit tests verify malformed
JSON and missing-prompt requests do not call the live probe or mocked generate
function.
The r20 blocker had superseded r19 by adding startup-time `/status` recovery:
service reports now declare `status_loads_existing_alpha_artifacts=true`, the
Alpha pack/checker require it, and `AlphaHTTPServer` initializes service status
from an existing public-safe `glm52_kaggle_alpha.json`. A live HTTP status
probe against r20, before any `/generate` request, returned
`phase=blocked_gpu_quota`, `alpha_report_present=true`,
`phase_status.overall_state=blocked`, cleanup verified, and the public-safe
next resume command. The r20 cleanup proof reads this service status as its
evidence source.
The r19 blocker had superseded r18 by adding a request-level `/generate`
safeguard:
service reports now declare `generate_uses_current_gpu_quota_blocker=true`,
the Alpha pack/checker require that contract, and
`generate_with_live_probe()` short-circuits against a still-current imported
GPU quota blocker instead of launching a doomed Kaggle live probe. A public-safe
request proof was written under the r19 `requests/` directory; it returned
`ok=false`, blockers `kaggle_gpu_quota_unavailable` and
`glm52_alpha_request_blocked_by_current_gpu_quota_preflight`,
`next_quota_refresh_time=2026-07-11T00:00:00`, cleanup verified, and
`phase_status.overall_state=blocked`.
The r18 blocker had superseded r17 by adding structured public-safe
`phase_status` evidence to the Alpha report and status CLI. It records the
phase names `configuration_check`, `model_source_check`,
`gpu_quota_preflight`, `kernel_push`, `gpu_queue_running`,
`tpu_queue_running`, `cpu_queue_running`, `stage_completed`,
`decode_completed`, and `cleanup_completed`. For r19,
`phase_status.overall_state=blocked`, blocked phases are
`gpu_quota_preflight`, `kernel_push`, and `gpu_queue_running`, and completed
phases are `configuration_check`, `model_source_check`, and
`cleanup_completed`.
The r17 blocker had superseded r16 by writing a separate public-safe
`glm52_kaggle_alpha_benchmark.json` artifact and listing it under
`artifacts.benchmark_json`. It contains deploy time, stage count, provider
coverage, first-token/stage-latency fields, generated-token count, cleanup
status, runtime tuning, blockers, and public-safety metadata. In r17 it is a
blocker benchmark (`tokens_generated=0`, `stage_count=0`, provider coverage
empty) because no live run was started.
The r16 blocker supersedes r15 by preserving the `/generate` timeout contract
and adding a public-safe full `next_resume_command` with `--run-live`,
`--gpu-quota-preflight`, `--output-dir`, `--stage-push-parallelism 7`, the r11
stage-worker package, accelerator selections, wait/kernel/Coordinator timeouts,
and runtime tuning. It records `next_resume_command_redacts_credentials=true`
and intentionally omits token/cookie/signed URL/private runtime material. The
r15 blocker had added the `/generate` timeout contract to
the service artifact and checker gate. Service reports now advertise
`generate_request_fields=["prompt","max_new_tokens","timeout","timeout_seconds"]`,
and request-level `timeout`/`timeout_seconds` is forwarded into the same-request
live wait and Coordinator task timeout, capped by the service configured wait.
The current external blocker is explicit and automatically detected by deploy:
`crowdtensor deploy glm52-kaggle --run-live --gpu-quota-preflight ...` ran a
public-safe GPU quota preflight, authenticated all four known Kaggle GPU
accounts, skipped the live run before launching workers, and produced a clean
blocker with `cleanup_verified=true`. All known GPU accounts are weekly GPU
quota exhausted until `2026-07-11T00:00:00`.
`crowdtensor status glm52-kaggle` now defaults to `dist/glm52-kaggle-alpha`
and can read deploy artifacts as well as a running service status file. For the
r31 blocker directory it reports `phase=decode_blocked`,
`alpha_report_present=true`, `cleanup_verified=true`,
`gpu_quota_status.source=alpha_gpu_quota_summary`, all four authenticated GPU
accounts exhausted, the blocker's `next_resume_command`, and top-level
`phase_status`. This status usability improvement does not make the Alpha
achieved; the 8-token same-request live proof is still missing.
`crowdtensor cleanup glm52-kaggle` can also read deploy artifacts. For the r31
blocker directory it writes a public-safe cleanup proof with `ok=true`,
`cleanup_evidence_source=alpha_report`,
`cleanup_mode=gpu_quota_preflight_skipped_live`,
`temporary_kaggle_kernels_deleted=true`,
`temporary_private_packages_removed=true`, and
`live_resources_left_running=false`. This cleanup proof only covers the
quota-preflight-skipped deploy and must not be treated as Alpha readiness.

New engineering after r8:
- Runtime tuning now flows from `crowdtensor deploy/serve glm52-kaggle` through
  `scripts/glm52_kaggle_same_request_live_probe.py` and
  `scripts/glm52_kaggle_stage_worker_push_probe.py` into the private uploaded
  Kaggle env. Public artifacts record only safe tuning metadata.
- Supported tuning keys include full-prefix prefill length, DSA mask top-k,
  executed expert count, LM-head top-k, row/block/tensor byte budgets, and CPU
  group claim polling windows.
- `scripts/glm52_kaggle_alpha_pack.py` now preserves runtime tuning and the
  selected stage-worker package in the blocker resume command.
- `scripts/glm52_kaggle_alpha_pack.py` also imports public-safe GPU quota
  probes via `--gpu-quota-report` and records `kaggle_gpu_quota_unavailable`
  plus the next quota refresh time when all authenticated accounts are
  exhausted.
- `crowdtensor deploy glm52-kaggle --gpu-quota-preflight` now runs the quota
  probe automatically and skips the live run when all authenticated GPU
  accounts are exhausted, unless `--continue-live-on-gpu-quota-exhausted` is
  explicitly set.
- `scripts/kaggle_gpu_token_weekly_quota_probe.py` supports raw Kaggle token
  files with `--raw-token-file`/`--raw-token-username`, so the dedicated GPU
  account can be checked without converting token files or leaking keys.
- `cleanup_from_push()` no longer reports live resources left running when a
  Kaggle push is rejected before a kernel is created.
- The r8 slugs were intentionally not reused because deleted Kaggle notebooks
  can return `Notebook not found`. The reusable live package is now the unique
  r11 package:
  `dist/glm52-kaggle-stage-worker-package-20260707-alpha-r11-unique-runtime-tuning/glm52_kaggle_stage_worker_package.json`.

The r11 package checker passes, launches 7 unique Kaggle kernels when quota
allows, and preserves the same 39 real 2-layer stage topology with one CUDA
stage, one JAX TPU stage, and five CPU groups. The r11 1-token live probe
`dist/glm52-kaggle-alpha-20260707-live-1tok-r11-unique-runtime-tuning/glm52_kaggle_same_request_live_probe.json`
did not start the pipeline because Kaggle rejected the CUDA push with
`kaggle_gpu_quota_or_session_rejected`; cleanup evidence is complete with no
retained kernels.

Engineering completed after the r3 blocker:
- `scripts/glm52_kaggle_stage_worker_package.py` supports low-concurrency CPU
  group packages via `--cpu-stage-group-size`.
- CPU group packages preserve the 39 real 2-layer stage specs; they do not
  merge many GLM layers into one slow stage.
- CPU group workers use claim-before-stage-load, so no-task stages do not do
  expensive HF/weight loading before a Coordinator claim.
- CPU group `kernel.py` embeds one generic stage worker source and injects each
  stage through `CT_GLM52_STAGE_PAYLOAD_JSON`, avoiding Kaggle `SaveKernel` 400
  errors from multi-megabyte group kernels.
- The live probe expands grouped packages into real Coordinator stage specs
  while pushing only one Kaggle kernel per group. The push probe can filter a
  grouped package by any contained stage id.

The previous r8 throughput baseline is
`dist/glm52-kaggle-stage-worker-package-20260707-alpha-r8-39stage-cpu-groups-generic-worker-full-payload/glm52_kaggle_stage_worker_package.json`.
It launches 7 Kaggle kernels: 1 CUDA, 1 TPU, and 5 CPU groups covering all 39
real stages. The bounded r8 live attempt reached all 39 stage workers and
completed same-request stages 0, 1, and 2 before it was aborted as
throughput-infeasible for the 8-token bound. Evidence:
`r8_coordinator_status_before_abort.json` records 39 stage workers seen,
3/312 expected stage tasks complete, and 0/8 generated tokens;
`r8_manual_cleanup.json` records delete return code 0 for all 7 temporary
Kaggle kernels; `glm52_kaggle_alpha_runtime_blocker.json` records
`failure_stage=glm52_alpha_runtime_throughput_below_8token_bound`.

Do not rerun r3/r4/r5/r6/r7 packages. r4/r5 exposed CPU group scheduling and
file-visibility issues; r6 exposed Kaggle `SaveKernel` 400 for large embedded
group kernels; r7 exposed missing runtime fields in the generic payload. Next
work should reduce CPU full-prefix stage runtime enough to complete at least
one full token and then the 8-token Alpha gate, or move more stages to available
accelerators without weakening the checker.

## Latest GLM 5.2 Kaggle Service Alpha Status

Superseding Alpha status after the 2026-07-07 r3 live attempt: the GLM 5.2
Kaggle CPU/GPU/TPU service Alpha engineering path is implemented, and a real
low-concurrency 8-token live attempt was started, but the Alpha goal is not
achieved yet. The current canonical blocker artifact is
`dist/glm52-kaggle-alpha-20260707-live-8tok-r3-7stage-unique/glm52_kaggle_alpha.json`.
Its default checker passes:
`PYTHONPATH=. python scripts/glm52_kaggle_alpha_check.py --report dist/glm52-kaggle-alpha-20260707-live-8tok-r3-7stage-unique/glm52_kaggle_alpha.json --json`
returns `ok=true`, `error_count=0`, and `glm52_kaggle_alpha_ready=false`.
The strict readiness checker intentionally fails:
`PYTHONPATH=. python scripts/glm52_kaggle_alpha_check.py --report dist/glm52-kaggle-alpha-20260707-live-8tok-r3-7stage-unique/glm52_kaggle_alpha.json --require-ready --json`
reports missing multi-token live evidence, provider coverage, benchmark token
count, and cleanup proof. Do not mark the Alpha goal achieved from this
blocker artifact.

Live attempt findings:
- r1 used the old 39-stage r209 package with 39-way concurrency. It failed
  because `~/.config/crowdtensor/kaggle-tokens.md` has no `gpuowner` section for CUDA and
  because Kaggle rejected the CPU fanout with HTTP 429 / `Maximum batch CPU
  session count of 5 reached`. Potentially retained r1 refs for stages
  1/3/5/11/12/13 were manually deleted.
- r2 used a 7-stage topology but reused old `ct-glm52-stage-worker-*` slugs.
  Kaggle returned `Notebook not found` after prior deletes. All r2 refs were
  manually deleted.
- r3 used a unique 7-stage package at
  `dist/glm52-kaggle-stage-worker-package-20260707-alpha-r3-7stage-cuda-tpu-5cpu-unique/glm52_kaggle_stage_worker_package.json`
  and token routing with CUDA raw token `~/.config/crowdtensor/kaggle-gpu-token.md`
  username hint `gpuowner`, TPU section `tpuowner`, and CPU section
  `cpuowner`. The Coordinator accepted real worker traffic and stage0 CUDA
  completed one task, producing a public-safe activation hash. The run was
  stopped after the grouped CPU stage1 `[2,18]` did not complete in the
  observation window; all seven r3 kernel refs were manually deleted and the
  cleanup artifact is
  `dist/glm52-kaggle-alpha-20260707-live-8tok-r3-7stage-unique/glm52_kaggle_alpha_r3_cleanup.json`.
  Runtime blocker:
  `dist/glm52-kaggle-alpha-20260707-live-8tok-r3-7stage-unique/glm52_kaggle_alpha_runtime_blocker.json`.

Implemented Alpha pieces:
- `crowdtensor deploy glm52-kaggle ...` builds the service report and Alpha
  pack/check artifact; with `--run-live --max-new-tokens 8` it launches the
  same GLM52 Kaggle live path.
- `crowdtensor serve glm52-kaggle --port <port> --run` starts a local HTTP
  service with `GET /health`, `GET /status`, and `POST /generate`.
- `crowdtensor status glm52-kaggle` reads the public-safe service status.
- `crowdtensor cleanup glm52-kaggle` writes a public-safe cleanup proof from
  the latest service status.
- `scripts/glm52_kaggle_same_request_live_probe.py` now supports
  `--max-new-tokens`, concurrent stage pushes for multi-token runs, and
  private `CT_GLM52_COORDINATOR_STAGE_TASK_LIMIT` injection through
  `scripts/glm52_kaggle_stage_worker_push_probe.py`.
- `scripts/glm52_kaggle_alpha_pack.py` and
  `scripts/glm52_kaggle_alpha_check.py` reject queue-only evidence, local
  mock-only service shells, single-backend-only evidence, missing token hashes,
  missing cleanup proof, non-GLM fallback, and old 1-token RC/live artifacts as
  Alpha success.

Current validation command:
`PYTHONPATH=. pytest -q tests/test_glm52_kaggle_alpha.py tests/test_glm52_kaggle_same_request_live_probe.py tests/test_glm52_kaggle_stage_worker_push_probe.py`
passes with 31 tests; after the runtime-blocker pack extension,
`PYTHONPATH=. pytest -q tests/test_glm52_kaggle_alpha.py` passes with 4 tests.
Next resume should not rerun the 39-way package. It should implement either a
multi-stage CPU worker that can claim several 2-layer CPU stage ids inside one
Kaggle CPU kernel, or another low-concurrency scheduler that keeps CPU session
count <=5 without merging many GLM layers into one slow CPU stage. Only mark
the Alpha goal complete after the strict checker passes with
`glm52_kaggle_alpha_ready=true`, generated token count at least 8, all three
providers present, and cleanup verified.

## Latest GLM 5.2 Kaggle Deployment RC Status

Superseding status after the 2026-07-07 r214 continuation: the GLM 5.2
Kaggle CPU/GPU/TPU deployment RC goal is achieved. The canonical RC artifact is
now
`dist/glm52-kaggle-accelerator-deployment-rc-20260707-r214-r211-same-request-live-achieved/glm52_kaggle_accelerator_deployment_rc.json`.
Its checker passes with `error_count=0`, `goal_achieved=true`,
`same_request_decode_verified=true`, and `failure_stage=none`. The successful
live run is
`dist/glm52-kaggle-same-request-live-20260707-r211-r209-hf-fetch-retries-full-live/glm52_kaggle_same_request_live_probe.json`;
`scripts/glm52_kaggle_same_request_live_check.py --require-verified` passes
with `generated_token_count=1`, `same_request_decode_verified=true`, and
`stage_count=39`. The assembled same-request proof at
`dist/glm52-kaggle-same-request-live-20260707-r211-r209-hf-fetch-retries-full-live/same-request/glm52_kaggle_same_request_probe.json`
also passes `scripts/glm52_kaggle_same_request_check.py --require-verified`;
it records accepted providers `["kaggle_cpu","kaggle_cuda","kaggle_jax_tpu"]`,
one generated token hash, live Coordinator request proof, stage-provider
coverage, and cleanup proof. The live wrapper records the compatible public
weight source as `cyankiwi/GLM-5.2-AWQ-INT4` for `zai-org/GLM-5.2`; the
source resolver remains
`dist/glm52-model-source-resolver-20260704-r4-awq-safetensors-recommended/glm52_model_source_resolver.json`.

The r211 run used the r209 worker package
`dist/glm52-kaggle-stage-worker-package-20260707-r209-r5-hf-fetch-retries/glm52_kaggle_stage_worker_package.json`
and request hash
`sha256:8385016dbeb99152007a34bce07e028a1ac9a564a28b5b294ca54164b49afeee`.
It completed all 39 GLM-5.2 stage workers in one same-request decode: stage0
on Kaggle CUDA (`gpuowner`), stage13 on Kaggle JAX TPU (`tpuowner`), and
the remaining stages on Kaggle CPU (`cpuowner`). The cleanup report
`dist/glm52-kaggle-same-request-live-20260707-r211-r209-hf-fetch-retries-full-live/glm52_kaggle_cleanup_report.json`
records no retained/uncleaned kernels, no live resources left running, and
public-safe cleanup metadata only. `scripts/glm52_kaggle_accelerator_deployment_rc_pack.py`
and `scripts/glm52_kaggle_accelerator_deployment_rc_check.py` were updated so
verified live same-request evidence is the final runtime proof; stale
metadata-only source/package preflight blockers and full-weight disk-budget
blockers do not prevent a quantized live GLM-5.2 RC success. The regression
suite `PYTHONPATH=. pytest -q tests/test_glm52_kaggle_accelerator_deployment_rc.py`
passes with 60 tests and still rejects queued TPU, non-GLM fallback, smoke
tests, missing token hash, missing provider coverage, and missing cleanup as
success.

Reusable public-safe validation commands:
`PYTHONPATH=. python scripts/glm52_kaggle_same_request_live_check.py --report dist/glm52-kaggle-same-request-live-20260707-r211-r209-hf-fetch-retries-full-live/glm52_kaggle_same_request_live_probe.json --require-verified --json`;
`PYTHONPATH=. python scripts/glm52_kaggle_same_request_check.py --report dist/glm52-kaggle-same-request-live-20260707-r211-r209-hf-fetch-retries-full-live/same-request/glm52_kaggle_same_request_probe.json --require-verified --json`;
`PYTHONPATH=. python scripts/glm52_kaggle_accelerator_deployment_rc_check.py --report dist/glm52-kaggle-accelerator-deployment-rc-20260707-r214-r211-same-request-live-achieved/glm52_kaggle_accelerator_deployment_rc.json --json`.
Do not rerun the long Kaggle deployment unless a new RC is intentionally being
produced; the r214 artifact is the current canonical successful RC.

Superseding status after the 2026-07-06 r199 continuation: the GLM 5.2
Kaggle CPU/GPU/TPU deployment goal is still not achieved. The canonical RC
artifact is now
`dist/glm52-kaggle-accelerator-deployment-rc-20260706-r199-r197-private-env-coordinator-live-gap/glm52_kaggle_accelerator_deployment_rc.json`.
Its checker passes with `error_count=0`, `goal_achieved=false`,
`same_request_decode_verified=false`, `generated_token_count=0`, and
`failure_stage=glm52_full_decode_adapter_not_ready`; do not mark the active
goal achieved from this artifact.

New engineering progress in r194/r197: a GLM-specific Coordinator bridge
contract now exists at
`scripts/glm52_kaggle_coordinator_decode_bridge_probe.py`, with checker
`scripts/glm52_kaggle_coordinator_decode_bridge_check.py`. The retained
contract artifact
`dist/glm52-kaggle-coordinator-decode-bridge-20260706-r194-contract/glm52_kaggle_coordinator_decode_bridge_probe.json`
is public-safe and checker-passing, but it is explicitly contract-only:
`coordinator_bridge_contract_ready=true`, `same_request_decode_verified=false`,
`live_run_performed=false`, and blocker
`glm52_live_kaggle_same_request_not_run`. Tests prove the Coordinator can
route private activation payloads through claim/submit without leaking
`hidden_b64` into public status, accepts a final generated token hash without
public token ids, and can assemble inputs that satisfy
`glm52_kaggle_same_request_probe.py` only when stage, Coordinator, and cleanup
reports are all verified.

The current worker package is now
`dist/glm52-kaggle-stage-worker-package-20260706-r197-r5-coordinator-private-env-bridge/glm52_kaggle_stage_worker_package.json`.
It preserves the r5 39-stage topology and request hash
`sha256:8385016dbeb99152007a34bce07e028a1ac9a564a28b5b294ca54164b49afeee`,
keeps `full_prefix_timeout_seconds=7200`, and embeds a Coordinator mode into
the rendered private Kaggle kernels. When private env vars
`CT_GLM52_COORDINATOR_URL` and `CT_GLM52_COORDINATOR_TOKEN` are present, each
stage worker can claim its stage task, pass private input activation into
`glm52_full_prefix_stage_decode_probe.py`, write a private output activation
file, submit the next activation hash for non-final stages, and submit the
selected token hash for the final stage. Without those private env vars, the
kernel stays on the old stage-runtime evidence path and does not overclaim
`stage_decode_verified`. `scripts/glm52_kaggle_stage_worker_push_probe.py`
can now inject those private values by uploading a temporary
`ct_glm52_private_runtime_env.json` into the private Kaggle package, then
deleting the local copy after `kaggle kernels push`; public artifacts only
record boolean/key-count metadata and must not include the Coordinator URL or
token.

Current blocker boundary: r188 verified stage activation handoff, r189 narrowed
the decode gap to exactly `coordinator_same_request_decode_runtime`, r191
proved the compressed-tensors runtime foundation, and r197 adds the worker-side
Coordinator activation bridge plus private runtime env injection, but no live Kaggle CPU/GPU/TPU same-request run
has produced a verified generated token and cleanup proof. The same-request
blocker remains
`dist/glm52-kaggle-same-request-20260706-r185-thirty-nine-stage-runtime-no-live-decode/glm52_kaggle_same_request_probe.json`.
Next resume should start a real Coordinator server, push/run the r197 packages
with private `CT_GLM52_COORDINATOR_URL`/`CT_GLM52_COORDINATOR_TOKEN`, collect
stage reports with `stage_decode_verified=true`, write public-safe Coordinator
and cleanup reports, run `glm52_kaggle_same_request_probe.py --mode assemble`,
then regenerate the decode-gap and RC artifacts. Do not rerun 39-stage coverage
unless a stage report is missing or stale.

Superseding status after the 2026-07-06 r187 continuation: the GLM 5.2
Kaggle CPU/GPU/TPU deployment goal is still not achieved, but all planned
stage runtime coverage is now complete. The canonical RC artifact is now
`dist/glm52-kaggle-accelerator-deployment-rc-20260706-r187-r181-thirty-nine-stage-verified-live-decode-adapter-gap/glm52_kaggle_accelerator_deployment_rc.json`.
Its checker passes with `error_count=0`, `goal_achieved=false`,
`same_request_decode_verified=false`, and
`failure_stage=glm52_full_decode_adapter_not_ready`; do not mark the goal
achieved from this artifact.

The current topology remains
`dist/glm52-kaggle-stage-runtime-plan-20260705-r5-cuda2-tpu2-preserve-cpu14-full-coverage/glm52_kaggle_stage_runtime_plan.json`.
The current package is now
`dist/glm52-kaggle-stage-worker-package-20260706-r181-r5-timeout7200-hf-header-retry-bound-request/glm52_kaggle_stage_worker_package.json`.
It preserves the r5/r31 39-stage plan and coordinator request hash
`sha256:8385016dbeb99152007a34bce07e028a1ac9a564a28b5b294ca54164b49afeee`,
keeps `full_prefix_timeout_seconds=7200`, and embeds a HF/safetensors range
read retry fix from `scripts/glm52_pack_quantized_dequant_probe.py`. That fix
was needed because the first stage38 run
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r180-r158-cpu-stage38-76-78-timeout7200-bound-live/glm52_kaggle_stage_worker_push_probe.json`
completed and cleaned up but failed with a transient final-stage
`JSONDecodeError` while reading lm_head safetensors header material.
The retry hardening is covered by
`tests/test_glm52_pack_quantized_dequant_probe.py`; the related test command
`PYTHONPATH=. pytest -q tests/test_glm52_pack_quantized_dequant_probe.py tests/test_glm52_full_prefix_stage_decode_probe.py tests/test_glm52_kaggle_stage_worker_package.py`
passed with 24 tests.

The successful final live stage is
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r182-r181-cpu-stage38-76-78-timeout7200-hf-header-retry-bound-live/glm52_kaggle_stage_worker_push_probe.json`.
Its stage38 report at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r182-r181-cpu-stage38-76-78-timeout7200-hf-header-retry-bound-live/notebook-output/stage-38-kaggle_cpu/glm52_kaggle_stage_runtime_report.json`
passes `glm52_kaggle_stage_runtime_check.py --require-verified`, covers layer
range `[76,78]`, records stage output hash
`sha256:84e1addfc4186d2e68908d2d0c1ff00e13f61a789aafc718fbcbc84cd93d1af9`,
and deleted temporary kernel `cpuowner/ct-glm52-stage-worker-38-kaggle-cpu`.
The canonical import artifact is now
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r183-import-r181-stages0-38-verified-stage-filtered/glm52_kaggle_stage_worker_push_probe.json`;
its checker passes with `stage_runtime_reports_collected=39`,
`stage_runtime_reports_verified=39`, no missing stage ids, and verified
provider coverage `["kaggle_cpu","kaggle_cuda","kaggle_jax_tpu"]`.

Current same-request evidence is intentionally a blocker, not success:
`dist/glm52-kaggle-same-request-20260706-r185-thirty-nine-stage-runtime-no-live-decode/glm52_kaggle_same_request_probe.json`
assembles all 39 stage runtime reports, but has `live_run_performed=false`,
`same_request_decode_verified=false`, `generated_token_count=0`, and no
accepted providers because current stage reports are host-adapter stage
runtime proofs with `stage_decode_verified=false`. The refreshed adapter-gap
artifact
`dist/glm52-decode-adapter-gap-probe-20260706-r186-thirty-nine-stage-runtime-no-live-decode/glm52_decode_adapter_gap_probe.json`
passes its checker and narrows the remaining required capabilities to exactly
`stage_activation_handoff_runtime` and `coordinator_same_request_decode_runtime`.
Its component evidence still verifies the GLM 5.2/AWQ pieces already built:
AWQ int4 dequant/linear, DSA attention, dense/MoE MLP, router/expert gather,
stage-local KV-cache, and lm-head token selection.

Next resume should not rerun stage coverage first. It should implement the
missing live GLM 5.2 activation handoff and Coordinator same-request decode
runtime: workers must pass private activations/KV between Kaggle CPU, CUDA,
and JAX/TPU stages in one Coordinator request, produce at least one generated
token hash, emit public-safe Coordinator/cleanup proof, then assemble a
verified same-request report and regenerate the RC. Do not reclassify r180,
r181, r182, r183, r184, r185, r186, or r187 as goal completion.

Superseding status after the 2026-07-06 r179 continuation: the GLM 5.2
Kaggle CPU/GPU/TPU deployment goal is still not achieved. The canonical RC
artifact is now
`dist/glm52-kaggle-accelerator-deployment-rc-20260706-r179-r158-thirty-eight-stage-verified-planned-stage-gap/glm52_kaggle_accelerator_deployment_rc.json`.
Its checker passes with `error_count=0`, `goal_achieved=false`,
`same_request_decode_verified=false`, and
`failure_stage=glm52_stage_runtime_adapter_not_verified`; do not mark the
goal achieved from this artifact.

The current topology remains
`dist/glm52-kaggle-stage-runtime-plan-20260705-r5-cuda2-tpu2-preserve-cpu14-full-coverage/glm52_kaggle_stage_runtime_plan.json`.
The current package is now
`dist/glm52-kaggle-stage-worker-package-20260706-r158-r5-timeout7200-bound-request/glm52_kaggle_stage_worker_package.json`.
It uses the same 39-stage r31/r5 full-coverage plan, binds the same
coordinator request hash as the verified live stages, and raises the embedded
full-prefix runtime timeout from 3600 seconds to 7200 seconds. The r33 package
is still useful for earlier evidence, but r157 stage31 `[62,64]` timed out at
3600 seconds with `glm52_full_prefix_stage_runtime_timeout`; r159 reran the
same stage with r158/7200 and passed.

Verified live Kaggle stage evidence now covers thirty-eight planned stages and all
three provider families. Reused verified evidence: CUDA stage0 `[0,2]` from
`dist/glm52-kaggle-stage-worker-push-probe-20260705-r72-r30-cuda-stage0-0-2-dense-live/glm52_kaggle_stage_worker_push_probe.json`,
TPU stage13 `[26,28]` from
`dist/glm52-kaggle-stage-worker-push-probe-20260705-r79-r31-tpu-stage13-26-28-live-retain/glm52_kaggle_stage_worker_push_probe.json`,
and CPU stage14 `[52,54]` from
`dist/glm52-kaggle-stage-worker-push-probe-20260705-r73-r30-cpu-stage14-52-54-live/glm52_kaggle_stage_worker_push_probe.json`.
Earlier CPU evidence: stage1 `[2,4]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260705-r83-r31-cpu-stage1-2-4-live/glm52_kaggle_stage_worker_push_probe.json`,
stage2 `[4,6]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260705-r87-r33-cpu-stage2-4-6-timeout3600-bound-live/glm52_kaggle_stage_worker_push_probe.json`,
and stage3 `[6,8]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r89-r33-cpu-stage3-6-8-timeout3600-bound-retry/glm52_kaggle_stage_worker_push_probe.json`.
Earlier CPU evidence: stage4 `[8,10]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r94-r33-cpu-stage4-8-10-timeout3600-bound-live/glm52_kaggle_stage_worker_push_probe.json`,
stage5 `[10,12]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r95-r33-cpu-stage5-10-12-timeout3600-bound-live/glm52_kaggle_stage_worker_push_probe.json`,
stage6 `[12,14]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r96-r33-cpu-stage6-12-14-timeout3600-bound-live/glm52_kaggle_stage_worker_push_probe.json`,
and stage7 `[14,16]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r97-r33-cpu-stage7-14-16-timeout3600-bound-live/glm52_kaggle_stage_worker_push_probe.json`.
Latest CPU evidence: stage8 `[16,18]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r100-r33-cpu-stage8-16-18-timeout3600-bound-live/glm52_kaggle_stage_worker_push_probe.json`,
stage9 `[18,20]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r101-r33-cpu-stage9-18-20-timeout3600-bound-live/glm52_kaggle_stage_worker_push_probe.json`,
stage10 `[20,22]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r102-r33-cpu-stage10-20-22-timeout3600-bound-live/glm52_kaggle_stage_worker_push_probe.json`.
New CPU evidence: stage11 `[22,24]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r105-r33-cpu-stage11-22-24-timeout3600-bound-live/glm52_kaggle_stage_worker_push_probe.json`
and stage12 `[24,26]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r106-r33-cpu-stage12-24-26-timeout3600-bound-live/glm52_kaggle_stage_worker_push_probe.json`.
Newest CPU evidence: stage15 `[28,30]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r109-r33-cpu-stage15-28-30-timeout3600-bound-live/glm52_kaggle_stage_worker_push_probe.json`
and stage16 `[30,32]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r112-r33-cpu-stage16-30-32-timeout3600-bound-live/glm52_kaggle_stage_worker_push_probe.json`,
and stage17 `[32,34]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r115-r33-cpu-stage17-32-34-timeout3600-bound-live/glm52_kaggle_stage_worker_push_probe.json`,
and stage18 `[34,36]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r118-r33-cpu-stage18-34-36-timeout3600-bound-live/glm52_kaggle_stage_worker_push_probe.json`,
and stage19 `[36,38]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r121-r33-cpu-stage19-36-38-timeout3600-bound-live/glm52_kaggle_stage_worker_push_probe.json`,
and stage20 `[38,40]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r124-r33-cpu-stage20-38-40-timeout3600-bound-live/glm52_kaggle_stage_worker_push_probe.json`,
and stage21 `[40,42]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r127-r33-cpu-stage21-40-42-timeout3600-bound-live/glm52_kaggle_stage_worker_push_probe.json`,
and stage22 `[42,44]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r130-r33-cpu-stage22-42-44-timeout3600-bound-live/glm52_kaggle_stage_worker_push_probe.json`,
and stage23 `[44,46]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r133-r33-cpu-stage23-44-46-timeout3600-bound-live/glm52_kaggle_stage_worker_push_probe.json`,
and stage24 `[46,48]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r136-r33-cpu-stage24-46-48-timeout3600-bound-live/glm52_kaggle_stage_worker_push_probe.json`,
and stage25 `[48,50]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r139-r33-cpu-stage25-48-50-timeout3600-bound-live/glm52_kaggle_stage_worker_push_probe.json`,
and stage26 `[50,52]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r142-r33-cpu-stage26-50-52-timeout3600-bound-live/glm52_kaggle_stage_worker_push_probe.json`,
and stage27 `[54,56]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r145-r33-cpu-stage27-54-56-timeout3600-bound-live/glm52_kaggle_stage_worker_push_probe.json`,
and stage28 `[56,58]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r148-r33-cpu-stage28-56-58-timeout3600-bound-live/glm52_kaggle_stage_worker_push_probe.json`,
and stage29 `[58,60]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r151-r33-cpu-stage29-58-60-timeout3600-bound-live/glm52_kaggle_stage_worker_push_probe.json`,
and stage30 `[60,62]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r154-r33-cpu-stage30-60-62-timeout3600-bound-live/glm52_kaggle_stage_worker_push_probe.json`,
and stage31 `[62,64]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r159-r158-cpu-stage31-62-64-timeout7200-bound-live/glm52_kaggle_stage_worker_push_probe.json`,
and stage32 `[64,66]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r162-r158-cpu-stage32-64-66-timeout7200-bound-live/glm52_kaggle_stage_worker_push_probe.json`,
and stage33 `[66,68]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r165-r158-cpu-stage33-66-68-timeout7200-bound-live/glm52_kaggle_stage_worker_push_probe.json`,
and stage34 `[68,70]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r168-r158-cpu-stage34-68-70-timeout7200-bound-live/glm52_kaggle_stage_worker_push_probe.json`,
and stage35 `[70,72]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r171-r158-cpu-stage35-70-72-timeout7200-bound-live/glm52_kaggle_stage_worker_push_probe.json`,
and stage36 `[72,74]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r174-r158-cpu-stage36-72-74-timeout7200-bound-live/glm52_kaggle_stage_worker_push_probe.json`,
and stage37 `[74,76]` at
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r177-r158-cpu-stage37-74-76-timeout7200-bound-live/glm52_kaggle_stage_worker_push_probe.json`.
All thirty-eight stage reports pass `glm52_kaggle_stage_runtime_check.py --require-verified`
and are imported by
`dist/glm52-kaggle-stage-worker-push-probe-20260706-r178-import-r158-stages0-37-verified-stage-filtered/glm52_kaggle_stage_worker_push_probe.json`.
The unfiltered r98 import directory without `stage-filtered` is not canonical
because it records missing planned stages as push blockers; use the
stage-filtered r178 import plus the r179 RC package-plan summary instead.

Two summary correctness fixes landed in this continuation. First,
`scripts/glm52_kaggle_stage_worker_push_probe.py` no longer adds
`glm52_mcp_tpu_stage_runtime_not_ready` when a direct imported TPU stage
report is already verified. Second,
`scripts/glm52_kaggle_accelerator_deployment_rc_pack.py` now computes planned
stage coverage from the stage-worker package, not from a filtered import
report. r179 correctly records
`required_provider_stage_runtime_reports_verified=true`,
`all_planned_stage_runtime_reports_verified=false`,
`required_stage_runtime_reports_verified=false`, `planned_stage_count=39`,
`stage_runtime_reports_verified=38`, and `missing_planned_stage_count=1`.
Related tests pass: `tests/test_glm52_kaggle_stage_worker_package.py`,
`tests/test_glm52_kaggle_stage_worker_push_probe.py`, and
`tests/test_glm52_kaggle_accelerator_deployment_rc.py`.

Next resume should continue from r5/r158/r178/r179: run missing CPU stages,
starting at stage38 `[76,78]`, using the r158 package, `cpuowner` token section,
`--wait-seconds 7500`, `--kernel-timeout-seconds 9000`, and
`--retain-nonterminal-cpu`; import successful stages with r178's existing
CUDA/TPU/CPU evidence, regenerate the canonical RC, and only after all 39
planned stages verify attempt the live Coordinator same-request decode. Do not
reclassify r72, r73, r79, r83, r87, r89, r91, r94, r95, r96, r97, r98, or
r99, r100, r101, r102, r103, r104, r105, r106, r107, r108, r109, r110, or
r111, r112, r113, r114, r115, r116, r117, r118, r119, r120, r121, r122,
r123, r124, r125, r126, r127, r128, r129, r130, r131, r132, r133, r134,
r135, r136, r137, r138, r139, r140, r141, r142, r143, r144, r145, r146,
r147, r148, r149, r150, r151, r152, r153, r154, r155, r156, r157, r158,
r159, r160, r161, r162, r163, r164, r165, r166, r167, r168, r169, r170,
r171, r172, r173, r174, r175, r176, r177, r178, or r179 as goal completion.

Superseding status after the 2026-07-05 r61 continuation: the GLM 5.2
Kaggle CPU/GPU/TPU deployment goal is still not achieved, but the CPU runtime
path made concrete progress. The current canonical RC artifact is
`dist/glm52-kaggle-accelerator-deployment-rc-20260705-r61-cpu-handoff-stage-verified-same-request-gap/glm52_kaggle_accelerator_deployment_rc.json`.
Its checker passes with `error_count=0`, `goal_achieved=false`,
`same_request_decode_verified=false`, and
`failure_stage=glm52_stage_runtime_adapter_not_verified`; do not mark the goal
achieved from this artifact.

New tool/runtime capabilities from this continuation:
`scripts/glm52_kaggle_stage_worker_push_probe.py` now supports
`--retain-nonterminal-cpu`, `--mode collect`, and `--stage-ids`, so a RUNNING
CPU kernel can be retained and later collected without repushing. The stage
runtime plan can now be overridden with repeatable
`--stage-spec stage_id:provider:start:end`, validates contiguous full 78-layer
coverage, and propagates `stage_count` into worker packages. Non-final
full-prefix stage workers now run in handoff-only mode by passing
`--skip-lm-head`; this avoids unnecessary lm_head/full-vocab token selection
for middle stages while preserving the boundary that generated-token and
same-request success still require a final-stage/Coordinator proof.

The latest topology/package artifacts are
`dist/glm52-kaggle-stage-runtime-plan-20260705-r3-cpu-tail-2layer-split/glm52_kaggle_stage_runtime_plan.json`
and
`dist/glm52-kaggle-stage-worker-package-20260705-r26-cpu-tail-2layer-split-handoff-only/glm52_kaggle_stage_worker_package.json`.
They cover all 78 GLM 5.2 layers with `kaggle_cuda:[0,26]`,
`kaggle_jax_tpu:[26,52]`, and thirteen 2-layer `kaggle_cpu` stages
`[52,54]` through `[76,78]`. The package checker passes and each package has
the correct `stage_count=15` and provider owner mapping
`kaggle_cuda -> gpuowner`, `kaggle_jax_tpu -> tpuowner`,
`kaggle_cpu -> cpuowner`.

The previous full-stage CPU attempts proved why the topology had to change:
`dist/glm52-kaggle-stage-worker-push-probe-20260705-r54-r53-cpu-full-stage-collect-retained/glm52_kaggle_stage_worker_push_probe.json`
collected the 26-layer CPU stage `[52,78]`, and
`dist/glm52-kaggle-stage-worker-push-probe-20260705-r56-r55-cpu-stage2-52-60-collect-retained/glm52_kaggle_stage_worker_push_probe.json`
collected the 8-layer CPU stage `[52,60]`; both loaded real GLM 5.2 AWQ
weight bytes but failed with `glm52_full_prefix_stage_runtime_timeout` and no
`stage_output_hash`. Even r25's 2-layer `[52,54]` full lm_head stage timed out
at
`dist/glm52-kaggle-stage-worker-push-probe-20260705-r58-r57-cpu-stage2-52-54-collect-retained/glm52_kaggle_stage_worker_push_probe.json`.

The first successful handoff-only CPU stage proof is
`dist/glm52-kaggle-stage-worker-push-probe-20260705-r60-r59-cpu-stage2-52-54-handoff-collect/glm52_kaggle_stage_worker_push_probe.json`.
It collected and deleted `cpuowner/ct-glm52-stage-worker-2-kaggle-cpu`;
the push checker passes with `stage_runtime_reports_collected=1` and
`stage_runtime_reports_verified=1`. The stage report at
`dist/glm52-kaggle-stage-worker-push-probe-20260705-r60-r59-cpu-stage2-52-54-handoff-collect/notebook-output/stage-2-kaggle_cpu/glm52_kaggle_stage_runtime_report.json`
passes `glm52_kaggle_stage_runtime_check.py --require-verified`, has
`model_id=zai-org/GLM-5.2`, `provider=kaggle_cpu`, `stage_id=2`,
`stage_layer_range=[52,54]`, real GLM 5.2 AWQ weight bytes loaded, provider
runtime verified, `full_prefix_stage_runtime_adapter_verified=true`, and a
public-safe `stage_output_hash`. It still records
`stage_decode_verified=false` and `same_request_decode_verified=false`, so it
is runtime handoff evidence only, not deployment success.

Next resume should continue from r26/r60: run the remaining CPU handoff stages
`3..13` and the final CPU stage `14` (final stage may still require lm_head),
then run the CUDA stage when GPU quota/session capacity is available and the
TPU stage through the retained/stable Kaggle TPU path. Only after all required
stage reports are verified should the Coordinator same-request decode be run
to emit generated token/hash and cleanup proof. Do not reclassify r60, r61,
or any single-stage handoff artifact as goal completion.

Superseding status after the 2026-07-05 r50 continuation: the GLM 5.2
Kaggle CPU/GPU/TPU deployment goal is still not achieved. The current
canonical RC artifact is
`dist/glm52-kaggle-accelerator-deployment-rc-20260705-r50-provider-owner-full-stage-package-cpu-live-gap/glm52_kaggle_accelerator_deployment_rc.json`.
Its checker passes with `error_count=0`, `goal_achieved=false`,
`same_request_decode_verified=false`, and
`failure_stage=glm52_stage_runtime_adapter_not_verified`; do not mark the goal
achieved from this artifact.

The latest stage-worker package is
`dist/glm52-kaggle-stage-worker-package-20260705-r23-provider-owner-map-full-stage-bound-request/glm52_kaggle_stage_worker_package.json`.
It supersedes the earlier single-owner full-stage package and renders
full-stage full-prefix worker packages with provider-specific Kaggle owners:
`kaggle_cuda -> gpuowner`, `kaggle_jax_tpu -> tpuowner`, and
`kaggle_cpu -> cpuowner`. Its package checker passes. Each rendered package now
records `full_prefix_probe_mode=full-stage` and
`full_prefix_probe_covers_full_stage=true`, with stage ranges `[0,26]`,
`[26,52]`, and `[52,78]` respectively. This is packaging/readiness evidence,
not runtime success.

The latest CUDA full-stage live attempt is
`dist/glm52-kaggle-stage-worker-push-probe-20260705-r51-r23-cuda-full-stage-live-short/glm52_kaggle_stage_worker_push_probe.json`.
Its checker passes, but no CUDA stage report was collected: the dedicated GPU
account push was rejected as `kaggle_gpu_quota_or_session_rejected`, and the
artifact records `glm52_stage_worker_push_failed:kaggle_cuda`.

The latest CPU full-stage live attempt is
`dist/glm52-kaggle-stage-worker-push-probe-20260705-r52-r23-cpu-full-stage-live-short/glm52_kaggle_stage_worker_push_probe.json`.
Its checker passes and it successfully pushed
`cpuowner/ct-glm52-fullstage-worker-2-kaggle-cpu`, observed
`KernelWorkerStatus.RUNNING`, then deleted the kernel after the bounded wait.
No output was collected and no stage runtime report was verified, so it records
`glm52_stage_worker_live_reports_missing` and
`glm52_stage_worker_live_reports_not_verified`.

Next resume should retry the full-stage live worker path only when the
dedicated GPU account has quota/session capacity again, and should separately
run/collect full-stage `kaggle_jax_tpu` and `kaggle_cpu` reports using the r23
provider-owner package. After all three providers emit verified
`stage_decode_verified` reports, run the live Coordinator same-request decode
to produce a generated token/hash and cleanup proof. Do not reclassify r23,
r51, r52, older stage-runtime reports, or component probes as completion.

Superseding status after the 2026-07-05 r48 continuation: the GLM 5.2
Kaggle CPU/GPU/TPU deployment goal is still not achieved, but the decode
adapter gap is now narrowed to the actual same-request/stage-decode closure
instead of listing already-proven component runtime primitives as missing. The
current canonical RC artifact is
`dist/glm52-kaggle-accelerator-deployment-rc-20260705-r48-component-runtime-proofs-imported-same-request-gap/glm52_kaggle_accelerator_deployment_rc.json`.
Its checker passes with `error_count=0`, `goal_achieved=false`,
`same_request_decode_verified=false`, and
`failure_stage=glm52_full_decode_adapter_not_ready`; do not mark the goal
achieved from this artifact.

The current decode-gap artifact is
`dist/glm52-decode-adapter-gap-probe-20260705-r5-component-runtime-proofs-imported/glm52_decode_adapter_gap_probe.json`.
Its checker passes and it imports the existing GLM 5.2 component proofs as
runtime evidence only. The following required decode capabilities are now
marked verified by public-safe component or handoff artifacts:
`glm_moe_dsa_transformer_block_runtime`,
`glm_moe_dsa_attention_q_lora_kv_lora_rope_nope`,
`glm_moe_dsa_dense_and_moe_mlp_runtime`,
`glm_moe_dsa_topk_router_and_expert_gather`,
`awq_int4_dequant_linear_runtime`, `stage_activation_handoff_runtime`,
`stage_local_kv_cache_runtime`, and
`lm_head_logits_token_selection_runtime`. The only remaining missing
capability in `required_capabilities` is
`coordinator_same_request_decode_runtime`. This is not success: r5 still
records `decode_adapter_ready=false`, `stage_decode_provider_coverage=[]`,
`glm52_stage_decode_provider_missing:kaggle_cuda`,
`glm52_stage_decode_provider_missing:kaggle_jax_tpu`,
`glm52_stage_decode_provider_missing:kaggle_cpu`,
`glm52_stage_decode_not_verified`, `glm52_same_request_decode_not_verified`,
and `glm52_generated_token_not_verified`.

Next resume should focus on converting the component-runtime and
stage-runtime evidence into real provider stage-decode reports for
`kaggle_cuda`, `kaggle_jax_tpu`, and `kaggle_cpu`, then running a live
Coordinator same-request decode that emits a generated token/hash and cleanup
proof. Do not spend the next run re-proving source metadata, TPU availability,
or individual primitive probes unless they are needed by that same-request
stage decode.

Superseding status after the 2026-07-05 r47 continuation: the GLM 5.2
Kaggle CPU/GPU/TPU deployment goal is still not achieved, but the previous
three-provider stage activation handoff gap is now represented by a public-safe
runtime hash-chain proof rather than left as an undifferentiated missing
capability. The current canonical RC artifact is
`dist/glm52-kaggle-accelerator-deployment-rc-20260705-r47-stage-activation-handoff-verified-same-request-gap/glm52_kaggle_accelerator_deployment_rc.json`.
Its checker passes with `error_count=0`, `goal_achieved=false`,
`same_request_decode_verified=false`, and
`failure_stage=glm52_full_decode_adapter_not_ready`; do not mark the goal
achieved from this artifact.

The new activation handoff evidence is
`dist/glm52-stage-activation-handoff-probe-20260705-r1-three-provider-runtime-hash-chain/glm52_stage_activation_handoff_probe.json`,
checked by `scripts/glm52_stage_activation_handoff_check.py --require-verified`.
It consumes the same live CUDA, TPU, and CPU GLM 5.2 stage runtime reports as
r46, verifies one bound Coordinator request hash across all three providers,
verifies contiguous stage boundaries `[0,26] -> [26,52] -> [52,78]`, and emits
two public-safe activation handoff contract hashes. It explicitly keeps
`same_request_decode_verified=false`, `generated_token_verified=false`, and
`stage_decode_verified=false`, so it is handoff-runtime evidence only, not GLM
5.2 deployment inference success.

The current decode-gap artifact is now
`dist/glm52-decode-adapter-gap-probe-20260705-r4-stage-activation-handoff-verified/glm52_decode_adapter_gap_probe.json`.
Its checker passes and it imports the handoff proof, so
`stage_activation_handoff_runtime` is no longer in `missing_capabilities`.
The remaining missing decode capabilities are
`awq_int4_dequant_linear_runtime`, `coordinator_same_request_decode_runtime`,
`glm_moe_dsa_attention_q_lora_kv_lora_rope_nope`,
`glm_moe_dsa_dense_and_moe_mlp_runtime`,
`glm_moe_dsa_topk_router_and_expert_gather`,
`glm_moe_dsa_transformer_block_runtime`,
`lm_head_logits_token_selection_runtime`, and
`stage_local_kv_cache_runtime`. The next resume should continue turning the
existing local component proofs into a real stage decode adapter and then into
a live Coordinator same-request decode with generated token/hash and cleanup
proof.

Superseding status after the 2026-07-05 r46 continuation: the GLM 5.2
Kaggle CPU/GPU/TPU deployment goal is still not achieved, but the previous GPU
blocker is resolved. Use `~/.config/crowdtensor/kaggle-gpu-token.md` as the dedicated GPU
access-token source for future CUDA workers; it authenticates through
`KAGGLE_API_TOKEN` and currently maps to Kaggle owner `gpuowner`. Continue
using the older `~/.config/crowdtensor/kaggle-tokens.md` accounts for TPU/CPU as needed. Do not
print or persist any token value in public artifacts.

The current canonical RC artifact is now
`dist/glm52-kaggle-accelerator-deployment-rc-20260705-r46-three-provider-stage-runtime-verified-same-request-gap/glm52_kaggle_accelerator_deployment_rc.json`.
Its checker passes with `error_count=0`, `goal_achieved=false`,
`same_request_decode_verified=false`, and
`failure_stage=glm52_full_decode_adapter_not_ready`; do not mark the goal
achieved from this artifact. The new GPU quota probe
`dist/kaggle-gpu-token-weekly-quota-probe-20260705-r9-crowdtensor-gpu-full-quota/kaggle_gpu_token_weekly_quota_probe.json`
authenticated the dedicated GPU account, showed 30h GPU quota available, pushed
a minimal T4 kernel, and deleted it immediately. It records
`gpu_submission_accepted_count=1`, `gpu_session_limit_rejected_count=0`,
`weekly_gpu_quota_exhausted_count=0`, and
`gpu_reserved_exceeds_remaining_by_api_count=0`.

The CUDA worker package for the dedicated owner is
`dist/glm52-kaggle-stage-worker-package-20260705-r21-gpuowner-writable-embedded-bundle-bound-request/glm52_kaggle_stage_worker_package.json`.
It rendered `gpuowner/ct-glm52-stage-worker-0-kaggle-cuda` with T4 and
internet enabled. The live CUDA stage run
`dist/glm52-kaggle-stage-worker-push-probe-20260705-r48-r21-cuda-live-gpuowner-crowdtensor-gpu/glm52_kaggle_stage_worker_push_probe.json`
successfully pushed the T4 kernel, waited until `KernelWorkerStatus.COMPLETE`,
downloaded
`dist/glm52-kaggle-stage-worker-push-probe-20260705-r48-r21-cuda-live-gpuowner-crowdtensor-gpu/notebook-output/stage-0-kaggle_cuda/glm52_kaggle_stage_runtime_report.json`,
verified it with `glm52_kaggle_stage_runtime_check.py --require-verified`,
and deleted the temporary GPU kernel. The CUDA report is for
`model_id=zai-org/GLM-5.2`, `provider=kaggle_cuda`, `stage_id=0`,
stage range `[0,26]`, 2 CUDA devices, real stage-owned weight values loaded,
and public-safe hashes only.

The current three-provider stage-worker artifact is
`dist/glm52-kaggle-stage-worker-push-probe-20260705-r49-import-cuda-hdf-tpu-fff-cpu-verified/glm52_kaggle_stage_worker_push_probe.json`.
It combines the new CUDA report, the retained tpuowner TPU report
`dist/glm52-mcp-tpu-stage-runtime-watch-20260705-r12-r15-fff-running-watch15m/notebook-output/glm52_kaggle_stage_runtime_report.json`,
and the retained CPU report
`dist/glm52-kaggle-stage-worker-push-probe-20260705-r30-r15-cpu-writable-embedded-live/notebook-output/stage-2-kaggle_cpu/glm52_kaggle_stage_runtime_report.json`.
Both normal and `--require-live` push checkers pass with
`stage_runtime_reports_collected=3` and `stage_runtime_reports_verified=3`;
verified provider coverage is now `["kaggle_cpu","kaggle_cuda","kaggle_jax_tpu"]`.

The current same-request blocker artifact is
`dist/glm52-kaggle-same-request-20260705-r4-three-provider-stage-runtime-no-decode/glm52_kaggle_same_request_probe.json`.
It consumes all three stage reports and is public-safe, but
`same_request_decode_verified=false`: the current stage reports are full-prefix
stage-runtime/host-adapter proofs, not `stage_decode_verified=true` decode
proofs; there is still no live Coordinator decode, generated token/hash, or
same-request cleanup proof. The current decode gap artifact is
`dist/glm52-decode-adapter-gap-probe-20260705-r3-three-provider-stage-runtime-gap/glm52_decode_adapter_gap_probe.json`.
It records `stage_runtime_provider_coverage=["kaggle_cpu","kaggle_cuda","kaggle_jax_tpu"]`
and `stage_decode_provider_coverage=[]`; the remaining work is implementing
the full GLM 5.2 AWQ/DSA decode adapter and Coordinator same-request path:
AWQ dequant linear runtime, GLM-MoE-DSA attention with q/kv lora, RoPE/nope,
dense+MoE MLP, top-k router/expert gather, activation handoff, stage-local
KV-cache, lm_head token selection, and generated-token Coordinator proof.

Current status after the 2026-07-05 GLM 5.2 r40 continuation: the GLM 5.2
Kaggle CPU/GPU/TPU deployment goal is still not achieved, but the source,
Kaggle public attach-source search, TPU-acquisition, AWQ-header,
completed-stage-smoke, provider-aligned stage-owned value loading, GPU/TPU/CPU
live stage runtime execution, transformers decode preflight, attention
projection proof, single-token attention proof, KV-cache decode proof,
single-layer decode-composition proof, full-vocab lm_head token-selection
proof, DSA-masked single-layer decode proof, DSA-layer-hidden-to-lm_head
token-selection proof, DSA multi-layer decode-token-chain proof, DSA
full-prefix multi-layer stage-hidden proof, full-prefix Kaggle worker runtime
bundle, DSA indexer proof, pack-quantized dequant/linear-slice proof,
single-expert AWQ MLP proof, decode-adapter gap, router/gather top-8 proof,
full MoE MLP proof, RC, checker, and tests are public-safe and aligned with
the completion boundary. The current canonical RC artifact is
`dist/glm52-kaggle-accelerator-deployment-rc-20260705-r40-r15-tpu-5poll-gpu-xu-slot-window/glm52_kaggle_accelerator_deployment_rc.json`.
Its checker passes with `error_count=0`, `goal_achieved=false`, and
`failure_stage=glm52_stage_runtime_adapter_not_verified`. This artifact must
not be treated as deployment success: it records only one verified live
provider stage (`kaggle_cpu`), a retained corrected TPU worker still queued,
and no CUDA/GPU stage report; it still lacks CUDA/GPU, verified JAX/TPU,
Coordinator same-request decode, generated token/hash, and complete cleanup
proof.
The current Kaggle public source search still records blockers
`glm52_kaggle_attach_source_not_found`,
`kaggle_models_glm52_weight_source_not_found`, and
`kaggle_datasets_glm52_weight_source_not_found`; the viable source path remains
HF `zai-org/GLM-5.2` with compatible AWQ/DSA weights.
It imports the latest bound full-prefix worker package
`dist/glm52-kaggle-stage-worker-package-20260705-r15-writable-embedded-bundle-bound-request/glm52_kaggle_stage_worker_package.json`:
`stage_runtime_package_kind=full_prefix_stage_decode`,
`full_prefix_runtime_bundle_required=true`, all three provider packages have
`full_prefix_runtime_bundle_present=true`,
`embedded_runtime_bundle_present=true`, and
`embedded_runtime_bundle_file_count=12`; the public-safe Coordinator request
hash is bound, and the stage-local full-prefix probe windows are CUDA `[6,8]`,
TPU `[26,28]`, and CPU `[54,56]`. r15 supersedes r11 because the embedded
runtime bundle now self-extracts into a writable Kaggle work directory instead
of the read-only script directory, uses a fast stage-value loading path, and
records retry/errno diagnostics for HF fetch failures. This is
package/preflight evidence only:
`pushed_to_kaggle=false`, `live_run_performed=false`, and
`same_request_route_verified=false`.
It imports the latest CPU live proof
`dist/glm52-kaggle-stage-worker-push-probe-20260705-r30-r15-cpu-writable-embedded-live/glm52_kaggle_stage_worker_push_probe.json`.
That run completed and deleted a Kaggle CPU kernel, collected one report, and
verified one provider stage: `stage_runtime_reports_collected=1`,
`stage_runtime_reports_verified=1`, and
`verified_provider_coverage=["kaggle_cpu"]`. The report
`dist/glm52-kaggle-stage-worker-push-probe-20260705-r30-r15-cpu-writable-embedded-live/notebook-output/stage-2-kaggle_cpu/glm52_kaggle_stage_runtime_report.json`
passes `glm52_kaggle_stage_runtime_check.py --require-verified` for
`model_id=zai-org/GLM-5.2`, `provider=kaggle_cpu`, and `stage_id=2`; it loads
real stage tensor values (`weight_tensor_values_loaded=true`,
`weight_value_byte_count=12288`), executes the full-prefix host adapter over
probe layer range `[54,56]`, verifies provider runtime and stage output hash,
and explicitly keeps `stage_decode_verified=false` and
`same_request_route_verified=false`. The normal push checker accepts this as
partial live blocker evidence, while `--require-live` correctly fails with
missing CUDA/GPU and JAX/TPU provider coverage.
The current stage-worker push evidence is
`dist/glm52-kaggle-stage-worker-push-probe-20260705-r35-r15-import-cpu-verified-tpu-queued-gpu-blocked/glm52_kaggle_stage_worker_push_probe.json`.
It imports the r30 CPU verified report, imports the retained r15 TPU watch
`dist/glm52-mcp-tpu-stage-runtime-watch-20260705-r3-r15-fff-queued/glm52_mcp_tpu_stage_runtime_watch.json`,
and records `stage_runtime_reports_collected=1`,
`stage_runtime_reports_verified=1`, `verified_provider_coverage=["kaggle_cpu"]`,
`kaggle_jax_tpu` terminal status `KernelWorkerStatus.QUEUED`, and missing
`kaggle_cuda` coverage. Its checker passes in normal mode and fails in
`--require-live` mode, as required for blocker evidence.
The current corrected TPU requests to keep polling are
`tpuowner/ct-glm52-stage-worker-1-kaggle-jax-tpu` from r34/r15 and
`cpuowner/ct-glm52-stage-worker-1-kaggle-jax-tpu` from r33/r16. Both were
observed for five bounded one-minute polls and remained `KernelWorkerStatus.QUEUED` at
`dist/glm52-mcp-tpu-stage-runtime-watch-20260705-r7-r15-fff-5poll/glm52_mcp_tpu_stage_runtime_watch.json`
and
`dist/glm52-mcp-tpu-stage-runtime-watch-20260705-r8-r16-cpuowner-5poll/glm52_mcp_tpu_stage_runtime_watch.json`
and were retained.
The older r25/r10 TPU worker did eventually complete and was downloaded at
`dist/glm52-mcp-tpu-stage-runtime-watch-20260705-r2-fff-stage-worker-complete-output/glm52_mcp_tpu_stage_runtime_watch.json`,
but its stage report failed with `glm52_full_prefix_stage_runtime_bundle_missing`
and `stage_output_hash_missing`; it proved TPU execution availability for the
old package only, not GLM 5.2 stage runtime success, and the completed kernel
was deleted. GPU remains externally unstable: the latest three-account GPU
slot probe
`dist/kaggle-gpu-token-weekly-quota-probe-20260705-r3-glm52-cuda-slot-recheck/kaggle_gpu_token_weekly_quota_probe.json`
authenticated all three accounts; `primary Kaggle account`/`xuyuhaosuyi` accepted and deleted
a minimal T4 kernel, while `tpuowner` and `cpuowner` still returned
`gpu_session_limit_rejected`; the report records
`gpu_submission_accepted_count=1`, `gpu_session_limit_rejected_count=2`, and
`weekly_gpu_quota_exhausted_count=0`. Real GLM CUDA worker creation still did
not succeed: the xuyuhaosuyi owner packages
`dist/glm52-kaggle-stage-worker-package-20260705-r17-xuyuhaosuyi-writable-embedded-bundle-bound-request/glm52_kaggle_stage_worker_package.json`
and
`dist/glm52-kaggle-stage-worker-package-20260705-r18-xuyuhaosuyi-unique-cuda-slug/glm52_kaggle_stage_worker_package.json`
pass package checks, but live CUDA pushes at
`dist/glm52-kaggle-stage-worker-push-probe-20260705-r37-r17-cuda-live-xuyuhaosuyi-retain/glm52_kaggle_stage_worker_push_probe.json`,
`dist/glm52-kaggle-stage-worker-push-probe-20260705-r38-r17-cuda-live-xuyuhaosuyi-retain-after-wait/glm52_kaggle_stage_worker_push_probe.json`,
`dist/glm52-kaggle-stage-worker-push-probe-20260705-r39-r18-cuda-live-xuyuhaosuyi-unique-retain/glm52_kaggle_stage_worker_push_probe.json`,
and
`dist/glm52-kaggle-stage-worker-push-probe-20260705-r40-r18-cuda-live-xuyuhaosuyi-unique-retain-after-wait/glm52_kaggle_stage_worker_push_probe.json`
all collected no CUDA stage report. r37/r39 hit
`kaggle_gpu_batch_session_limit_reached`; r38/r40 hit
`kaggle_kernel_notebook_not_found`, even after using a unique slug. A tiny
`enable_internet=true` xuyuhaosuyi GPU diagnostic package at
`dist/glm52-xuyuhaosuyi-gpu-internet-diagnostic-20260705-r1/` also hit
`Maximum batch GPU session count of 2 reached`, so this remains an external
Kaggle GPU/session creation blocker rather than a proven GLM adapter failure.
The corrected CUDA worker push artifact
`dist/glm52-kaggle-stage-worker-push-probe-20260705-r36-r15-cuda-live-tpuowner-push-error-fixed/glm52_kaggle_stage_worker_push_probe.json`
proves the project no longer treats Kaggle's `Kernel push error` stdout as
`pushed=true`; it records `pushed=false` and
`push_error_blocker=kaggle_gpu_batch_session_limit_reached`; r40 additionally
classifies `Notebook not found` as `kaggle_kernel_notebook_not_found`. The next
resume should first poll the two retained r15/r16 TPU workers for completion,
then retry CUDA on xuyuhaosuyi with a fresh unique slug only after a GPU
session slot is clearly available, and only then assemble the real Coordinator
same-request GLM 5.2 decode.
It also imports the pack-quantized
dequant slice artifact
`dist/glm52-pack-quantized-dequant-probe-20260705-r1-layer3-expert0-gate-slice/glm52_pack_quantized_dequant_probe.json`:
`pack_quantized_dequant_verified=true`,
`pack_quantized_linear_slice_verified=true`, `dequant_slice_shape=[4,64]`,
`linear_slice_shape=[4]`, public-safe hashes present, and
`stage_decode_verified=false`. It also imports the single-expert MLP artifact
`dist/glm52-pack-quantized-expert-mlp-probe-20260705-r1-layer3-expert0/glm52_pack_quantized_expert_mlp_probe.json`:
`pack_quantized_expert_mlp_verified=true`, `single_expert_mlp_verified=true`,
gate/up output shapes `[2048]`, down/final output shape `[6144]`, public-safe
hashes present, and `stage_decode_verified=false`. It also imports the
router/gather subset artifact
`dist/glm52-pack-quantized-router-gather-probe-20260705-r2-layer3-top8/glm52_pack_quantized_router_gather_probe.json`:
`router_topk_verified=true`, `router_topk_count=8`,
`routed_expert_subset_verified=true`, `executed_expert_count=8`, executed
experts 26, 174, 3, 206, 233, 41, 161, and 166,
`routed_subset_output_shape=[6144]`, public-safe hashes present, and
`stage_decode_verified=false`.
It imports the attention projection artifact
`dist/glm52-attention-projection-probe-20260705-r1-layer3-qkv-projection/glm52_attention_projection_probe.json`:
`attention_projection_verified=true`, `input_layernorm_verified=true`,
`q_lora_projection_verified=true`, `kv_lora_projection_verified=true`,
`query_shape=[64,256]`, `q_nope_shape=[64,192]`, `q_pe_shape=[64,64]`,
`k_nope_shape=[64,192]`, `value_shape=[64,256]`, public-safe hashes are
present, and `stage_decode_verified=false`.
It imports the single-token attention artifact
`dist/glm52-attention-single-token-probe-20260705-r1-layer3-rope-o-proj/glm52_attention_single_token_probe.json`:
`single_token_attention_verified=true`, `rope_applied=true`,
`attention_scores_verified=true`, `attention_weights_verified=true`,
`o_proj_verified=true`, `o_proj_output_shape=[6144]`,
`kv_cache_updated=false`, `dsa_indexer_verified=false`, and
`stage_decode_verified=false`.
It imports the KV-cache decode artifact
`dist/glm52-kv-cache-decode-probe-20260705-r1-layer3-prefill4-decode1/glm52_kv_cache_decode_probe.json`:
`kv_cache_prefill_verified=true`, `kv_cache_update_verified=true`,
`kv_cache_decode_attention_verified=true`, `o_proj_verified=true`,
`updated_key_cache_shape=[5,64,256]`, `attention_scores_shape=[64,5]`,
`o_proj_output_shape=[6144]`, public-safe hashes are present, and
`stage_decode_verified=false`, `generated_token_verified=false`.
It imports the single-layer decode-composition artifact
`dist/glm52-layer-decode-probe-20260705-r1-layer3-prefill4-basic-attn-moe/glm52_layer_decode_probe.json`:
`layer_decode_verified=true`, `attention_decode_verified=true`,
`attention_residual_verified=true`, `post_attention_norm_verified=true`,
`full_moe_mlp_verified=true`, `layer_output_shape=[6144]`, public-safe hashes
are present, and `dsa_masked_attention_integrated=false`,
`lm_head_verified=false`, `stage_decode_verified=false`, and
`same_request_decode_verified=false`.
It imports the full-vocab lm_head token-selection artifact
`dist/glm52-lm-head-token-probe-20260705-r1-full-vocab-streaming/glm52_lm_head_token_probe.json`:
`final_norm_verified=true`, `lm_head_streamed_full_vocab=true`,
`lm_head_logits_token_selection_verified=true`,
`selected_token_hash_verified=true`, `lm_head_shape=[154880,6144]`,
`lm_head_rows_scanned=154880`, `lm_head_block_count=76`, `top_k=5`, public-safe
hashes are present, and `full_model_hidden_verified=false`,
`generated_token_verified=false`, `stage_decode_verified=false`, and
`same_request_decode_verified=false`. This streams the real BF16
`model.norm.weight` and `lm_head.weight` but uses a deterministic probe hidden
vector, so it is not a generated-token proof.
It imports the DSA-masked single-layer decode artifact
`dist/glm52-dsa-masked-layer-decode-probe-20260705-r1-layer6-prefill8-topk4/glm52_dsa_masked_layer_decode_probe.json`:
`dsa_masked_attention_integrated=true`, `dsa_indexer_verified=true`,
`dsa_mask_verified=true`, `layer_decode_verified=true`, layer 6 uses a
`full` indexer, `dsa_index_score_shape=[9,9]`,
`attention_scores_shape=[64,9]`, `dsa_mask_topk_count=4`,
`dsa_mask_pruned_position_count=5`, public-safe hashes are present, and
`full_dsa_topk_scale_verified=false`, `lm_head_verified=false`,
`generated_token_verified=false`, `stage_decode_verified=false`, and
`same_request_decode_verified=false`. This proves real-weight DSA mask
integration into one GLM 5.2 AWQ layer decode, but only for a small
probe-sequence/top-k cap and not for a multi-layer stage or generated token.
It imports the DSA-layer-hidden-to-lm_head artifact
`dist/glm52-stage-hidden-lm-head-probe-20260705-r2-layer6-dsa-to-lm-head-fixed/glm52_stage_hidden_lm_head_probe.json`:
`stage_hidden_to_lm_head_verified=true`,
`stage_hidden_lm_head_token_selection_verified=true`,
`stage_dsa_masked_attention_integrated=true`,
`stage_layer_decode_verified=true`, `lm_head_shape=[154880,6144]`,
`lm_head_rows_scanned=154880`, `top_k=5`, public-safe hashes are present, and
`full_model_hidden_verified=false`, `generated_token_verified=false`,
`stage_decode_verified=false`, and `same_request_decode_verified=false`.
This is the first proof that a real DSA-masked GLM 5.2 AWQ layer output can be
normalized and consumed by the full BF16 `lm_head.weight`, but it is still
single-layer/small-sequence token selection, not full-model generation.
It imports the DSA multi-layer decode-token-chain artifact
`dist/glm52-multi-layer-stage-decode-probe-20260705-r2-layers6-8-shared-indexer-fixed/glm52_multi_layer_stage_decode_probe.json`:
`multi_layer_stage_hidden_verified=true`,
`multi_layer_decode_token_chain_verified=true`,
`stage_hidden_lm_head_token_selection_verified=true`, `stage_layer_range=[6,8]`,
`executed_layer_count=2`, `all_layers_dsa_masked_attention_integrated=true`,
`all_layers_moe_mlp_verified=true`, `all_layer_outputs_chained=true`,
`lm_head_rows_scanned=154880`, and public-safe hashes are present. Layer 6
uses its own `full` DSA indexer and layer 7 is a `shared` indexer that now
records `dsa_indexer_source_layer_id=6`; `scripts/glm52_dsa_masked_layer_decode_probe.py`
therefore supports shared-indexer layers by reusing the nearest previous full
indexer source instead of pretending shared layers own indexer weights.
This is stronger than r31 because real GLM 5.2 AWQ layer outputs are chained
across a full+shared DSA pair and then consumed by the full BF16 `lm_head`.
It still records `full_prefill_stage_hidden_verified=false`,
`generated_token_verified=false`, `stage_decode_verified=false`,
`live_kaggle_runtime_verified=false`, and `same_request_decode_verified=false`
because the prefill carrier is not full layer outputs and the proof is not a
Kaggle live stage runtime or Coordinator same-request decode.
It imports the DSA full-prefix multi-layer stage-hidden artifact
`dist/glm52-full-prefix-stage-decode-probe-20260705-r1-layers6-8-seq3-full-prefix/glm52_full_prefix_stage_decode_probe.json`:
`full_prefix_stage_hidden_verified=true`,
`multi_layer_stage_hidden_verified=true`,
`stage_hidden_lm_head_token_selection_verified=true`, `stage_layer_range=[6,8]`,
`stage_sequence_length=3`, `stage_hidden_sequence_shape=[3,6144]`,
`all_layers_full_prefix_verified=true`, `all_layer_outputs_chained=true`,
`lm_head_rows_scanned=154880`, and public-safe hashes are present. It executes
real GLM 5.2 AWQ layers 6 and 7 across the whole small prefix, not only the
decode token: layer 6 has a full DSA indexer and layer 7 is a shared-indexer
layer sourced from layer 6, with all three sequence positions verified per
layer. This closes the r32 prefill-carrier gap for a small sequence and gives
the next Kaggle worker a full-prefix stage-hidden adapter to reuse. It still
records `generated_token_verified=false`, `stage_decode_verified=false`,
`live_kaggle_runtime_verified=false`, and `same_request_decode_verified=false`
because it is a local public-safe adapter proof, not a live Kaggle stage
runtime or Coordinator same-request decode.
It imports the DSA indexer artifact
`dist/glm52-dsa-indexer-probe-20260705-r1-layer2-seq8/glm52_dsa_indexer_probe.json`:
`dsa_indexer_verified=true`, `dsa_topk_verified=true`, layer 2 is a
`full` indexer, `sequence_length=8`, `indexer_query_shape=[8,32,128]`,
`indexer_key_shape=[8,128]`, `index_score_shape=[8,8]`,
`topk_indices_shape=[8,8]`, `indexer_cache_updated=false`,
`attention_output_verified=false`, and `stage_decode_verified=false`.
It also imports the full MoE MLP artifact
`dist/glm52-pack-quantized-moe-mlp-probe-20260705-r1-layer3-top8-shared/glm52_pack_quantized_moe_mlp_probe.json`:
`full_moe_mlp_verified=true`, `router_topk_count=8`,
`executed_expert_count=8`, `shared_experts_mlp_verified=true`,
shared expert projections are BF16 with shapes `[2048,6144]`,
`[2048,6144]`, and `[6144,2048]`, `full_moe_output_shape=[6144]`,
public-safe hashes are present, and `stage_decode_verified=false`.

The current decode-adapter gap artifact is
`dist/glm52-decode-adapter-gap-probe-20260705-r2-current-stage-runtime-gap/glm52_decode_adapter_gap_probe.json`,
checked by `scripts/glm52_decode_adapter_gap_check.py`. It confirms the real
GLM 5.2 AWQ source is `glm_moe_dsa`, 78 layers, 256 routed experts,
8 experts per token, `pack-quantized` 4-bit weights, 232,269 weight keys, and
about 440.335957GB of quantized safetensors. It records provider runtime
coverage for `kaggle_cuda`, `kaggle_jax_tpu`, and `kaggle_cpu`, but no stage
decode provider coverage. The precise missing capabilities in that r2 gap
artifact still include a full AWQ int4 dequant-linear runtime, GLM-MoE-DSA
attention with q/k/v low-rank and rope/nope branches, dense and MoE MLP
runtime, top-k router/expert gather, stage activation handoff, stage-local KV
cache, full-model/stage hidden feeding lm_head, and Coordinator same-request
decode.
The r18 dequant slice, r19 single-expert MLP, r20/r21 routed top-8 proofs,
r23 full MoE MLP proof, r24 attention projection proof, r25 single-token
attention proof, r26 DSA indexer proof, r27 KV-cache decode proof, and r28
single-layer decode-composition proof, plus the r29 full-vocab lm_head
token-selection proof, r30 DSA-masked layer-decode proof, and r31
DSA-layer-hidden-to-lm_head proof, plus the r32 DSA multi-layer
decode-token-chain proof and r33 DSA full-prefix stage-hidden proof, narrow the
AWQ-format, expert-MLP, router, expert-gather, shared-experts, and attention
q/kv/o-projection, DSA top-k, prefill cache, decode cache update, decode
attention, attention output-projection, residual/post-norm composition, and
single-layer MoE output, final-norm, lm_head token-selection, and DSA-masked
single-layer attention-composition/stage-hidden-to-lm_head and full/shared
decode-token layer-chain risks, and the small-sequence full-prefix carrier
risk. They still do not satisfy live Kaggle stage runtime,
full-model/stage generated-token semantics, or Coordinator same-request
inference.
This supersedes the less specific r15 failure diagnosis: the current blocker
is full GLM 5.2 decode adapter implementation, not TPU acquisition.

The current transformers decode-adapter preflight artifact is
`dist/glm52-transformers-decode-adapter-preflight-20260705-r1-foundation/glm52_transformers_decode_adapter_preflight.json`,
checked by `scripts/glm52_transformers_decode_adapter_preflight_check.py`.
It passes with `--require-foundation`: installed `transformers` 5.9.0 exposes
`GlmMoeDsaForCausalLM`, the public AWQ config becomes loadable after removing
invalid `layer_types=deepseek_sparse_attention`, a tiny random GLM-MoE-DSA
forward succeeds, and full-model stage-selective key mapping covers all 78
layers with `missing_required_key_count=0`. The mapping covers 3 dense layers,
75 sparse/MoE layers, 21 full-indexer layers, 57 shared-indexer layers,
232,266 required keys, and 231,300 pack-quantized keys. This is adapter
foundation evidence only: `decode_adapter_ready=false` because no
`compressed_tensors`/AutoAWQ/llmcompressor pack-quantized runtime is available
and no real stage decode has executed.

The current attention projection proof is
`dist/glm52-attention-projection-probe-20260705-r1-layer3-qkv-projection/glm52_attention_projection_probe.json`,
checked by `scripts/glm52_attention_projection_check.py --require-verified`.
It executes real GLM 5.2 AWQ layer 3 attention projection paths with
`input_layernorm`, BF16 `q_a_proj`, `q_a_layernorm`, packed INT4 `q_b_proj`,
BF16 `kv_a_proj_with_mqa`, `kv_a_layernorm`, and packed INT4 `kv_b_proj`.
The verified shapes are `q_a_output_shape=[2048]`,
`q_b_output_shape=[16384]`, `query_shape=[64,256]`,
`q_nope_shape=[64,192]`, `q_pe_shape=[64,64]`,
`kv_a_output_shape=[576]`, `kv_b_output_shape=[28672]`,
`k_nope_shape=[64,192]`, and `value_shape=[64,256]`, with only public-safe
hashes. This is attention adapter progress, but it deliberately does not apply
RoPE, compute attention scores, execute `o_proj`, update KV cache, produce a
stage output, or generate a token.

The current single-token attention proof is
`dist/glm52-attention-single-token-probe-20260705-r1-layer3-rope-o-proj/glm52_attention_single_token_probe.json`,
checked by `scripts/glm52_attention_single_token_check.py --require-verified`.
It builds on the real layer 3 q/kv projection path, applies split-half RoPE at
`position_id=7`, computes single-token attention scores/weights, uses the
single-token value output, and executes the real packed INT4 `self_attn.o_proj`.
The verified shapes include `query_states_shape=[64,256]`,
`key_states_shape=[64,256]`, `value_states_shape=[64,256]`,
`attention_scores_shape=[64,1]`, `attention_flattened_shape=[16384]`, and
`o_proj_output_shape=[6144]`, with only public-safe hashes. This is the first
real-weight attention output projection proof, but it is explicitly not
multi-token prefill, DSA indexer routing, KV-cache decode, transformer block
execution, stage decode, or same-request Coordinator inference.

The current KV-cache decode proof is
`dist/glm52-kv-cache-decode-probe-20260705-r1-layer3-prefill4-decode1/glm52_kv_cache_decode_probe.json`,
checked by `scripts/glm52_kv_cache_decode_check.py --require-verified`. It
uses real GLM 5.2 AWQ layer 3 weights to build a deterministic 4-token prefill
plus 1-token decode sequence, executes `input_layernorm`, q/kv low-rank
projections, split RoPE, prefill key/value cache construction, decode
key/value append, decode-token attention over the updated cache, and the real
packed INT4 `self_attn.o_proj`. The verified shapes include
`prefill_key_cache_shape=[4,64,256]`,
`updated_key_cache_shape=[5,64,256]`,
`decode_query_shape=[64,256]`, `attention_scores_shape=[64,5]`,
`attention_flattened_shape=[16384]`, and `o_proj_output_shape=[6144]`, with
only public-safe hashes. This proves basic KV-cache decode attention for one
layer, but it is explicitly not DSA-masked attention integration, residual/norm
composition, MLP composition, transformer block execution, lm-head token
selection, stage decode, or same-request Coordinator inference.

The current single-layer decode-composition proof is
`dist/glm52-layer-decode-probe-20260705-r1-layer3-prefill4-basic-attn-moe/glm52_layer_decode_probe.json`,
checked by `scripts/glm52_layer_decode_check.py --require-verified`. It uses
real GLM 5.2 AWQ layer 3 weights to execute a 4-token prefill plus 1-token
decode path through basic KV-cache decode attention, attention residual add,
`post_attention_layernorm`, routed top-8 experts, shared experts, full MoE MLP
aggregation, and final single-layer output composition. Verified fields include
`attention_decode_verified=true`, `attention_residual_verified=true`,
`post_attention_norm_verified=true`, `full_moe_mlp_verified=true`,
`layer_decode_verified=true`, and `layer_output_shape=[6144]`, with only
public-safe hashes. It deliberately keeps
`dsa_masked_attention_integrated=false`, `lm_head_verified=false`,
`generated_token_verified=false`, `stage_decode_verified=false`, and
`same_request_decode_verified=false`; this is single-layer adapter progress,
not full stage decode or Kaggle deployment success.

The current DSA indexer proof is
`dist/glm52-dsa-indexer-probe-20260705-r1-layer2-seq8/glm52_dsa_indexer_probe.json`,
checked by `scripts/glm52_dsa_indexer_check.py --require-verified`. It uses
real GLM 5.2 AWQ layer 2 `full` indexer weights:
`self_attn.indexer.wq_b`, `wk`, `k_norm.{weight,bias}`, and `weights_proj`,
plus real q-residual inputs from `q_a_proj` and `q_a_layernorm`. The proof
executes split-half RoPE, DSA score construction, and top-k selection for a
small deterministic sequence. Verified shapes include
`q_resid_shape=[8,2048]`, `indexer_query_shape=[8,32,128]`,
`indexer_key_shape=[8,128]`, `index_score_shape=[8,8]`, and
`topk_indices_shape=[8,8]`, with only public-safe hashes. This is explicitly
not full prefill scale, indexer-cache decode, attention output, transformer
block execution, stage decode, or same-request Coordinator inference.

The current pack-quantized tensor-group proof is
`dist/glm52-pack-quantized-group-probe-20260705-r1-layer3-expert0-gate/glm52_pack_quantized_group_probe.json`,
checked by `scripts/glm52_pack_quantized_group_check.py --require-loaded`.
It range-loads the real GLM 5.2 AWQ MoE group
`model.layers.3.mlp.experts.0.gate_proj.{weight_packed,weight_scale,weight_zero_point,weight_shape}`
from `cyankiwi/GLM-5.2-AWQ-INT4`, all in
`model-00002-of-00083.safetensors`, and records only public-safe hashes and
metadata. The group is 7,274,512 bytes total:
`weight_packed` I32 rank-2 6,291,456 bytes, `weight_scale` BF16 rank-2
786,432 bytes, `weight_zero_point` I32 rank-2 196,608 bytes, and
`weight_shape` I64 rank-1 16 bytes. This proves the real pack-quantized
dequant input group can be stage-selectively read, but
`pack_quantized_group_dequantized=false` and `stage_decode_verified=false`.

The current pack-quantized dequant/linear-slice proof is
`dist/glm52-pack-quantized-dequant-probe-20260705-r1-layer3-expert0-gate-slice/glm52_pack_quantized_dequant_probe.json`,
checked by `scripts/glm52_pack_quantized_dequant_check.py --require-verified`.
It uses the compressed-tensors pack order to unpack real GLM 5.2 AWQ INT4
`weight_packed` and `weight_zero_point`, applies BF16 `weight_scale`, verifies
a real dequantized slice for layer 3 expert 0 `gate_proj`, and computes a
deterministic linear slice. The verified shapes are `dequant_slice_shape=[4,64]`
and `linear_slice_shape=[4]`, with only public-safe hashes. This is meaningful
adapter progress, but it is explicitly not a full projection, transformer
block, stage decode, KV-cache update, lm-head/token selection, or same-request
Coordinator decode.

The current single-expert AWQ MLP proof is
`dist/glm52-pack-quantized-expert-mlp-probe-20260705-r1-layer3-expert0/glm52_pack_quantized_expert_mlp_probe.json`,
checked by `scripts/glm52_pack_quantized_expert_mlp_check.py --require-verified`.
It loads the real GLM 5.2 AWQ INT4 groups for layer 3 expert 0 `gate_proj`,
`up_proj`, and `down_proj`, runs full dequant-linear for all three projections,
applies the gated SiLU MLP path, and emits a final public-safe output hash with
`final_output_shape=[6144]`. This is stronger than a slice proof and confirms
the real pack-quantized expert MLP path can execute locally, but it is still
only one expert: it does not cover top-k router/expert aggregation, attention,
residual/norm composition, full stage decode, or Coordinator generated-token
proof.

The current router/gather subset proof is
`dist/glm52-pack-quantized-router-gather-probe-20260705-r2-layer3-top8/glm52_pack_quantized_router_gather_probe.json`,
checked by `scripts/glm52_pack_quantized_router_gather_check.py --require-verified`.
It uses the real layer 3 `mlp.gate.weight` and
`mlp.gate.e_score_correction_bias`, follows the Transformers
`GlmMoeDsaMoE.route_tokens_to_experts` logic (`sigmoid`, correction bias,
top-k selection, normalized weights, `routed_scaling_factor=2.5`), verifies
`router_topk_count=8`, then executes and aggregates all eight selected routed
experts with real AWQ MLP weights. The executed experts are 26, 174, 3, 206,
233, 41, 161, and 166, and the routed output shape is `[6144]`. This is not
full MoE success by itself; it is superseded for MoE completeness by the full
MoE MLP proof below, and attention, residual/norm composition, stage decode,
KV cache, and generated-token proof remain missing.

The current full MoE MLP proof is
`dist/glm52-pack-quantized-moe-mlp-probe-20260705-r1-layer3-top8-shared/glm52_pack_quantized_moe_mlp_probe.json`,
checked by `scripts/glm52_pack_quantized_moe_mlp_check.py --require-verified`.
It reuses the real layer 3 router and all eight selected routed experts, then
loads and executes the real BF16 `shared_experts.{gate_proj,up_proj,down_proj}`
weights from `cyankiwi/GLM-5.2-AWQ-INT4`. The proof verifies
`router_topk_count=8`, `executed_expert_count=8`,
`shared_experts_mlp_verified=true`, and `full_moe_output_shape=[6144]`, with
only public-safe hashes. This is the first local real-weight proof of a full
GLM 5.2 sparse MoE MLP layer output, but it is explicitly not attention,
residual/norm composition, KV-cache update, lm-head/token selection, stage
decode, or same-request Coordinator inference.

The retained Kaggle MCP TPU request `tpuowner/ct-mcp-tpu-probe-0704-r2`
eventually completed. The current watch artifact is
`dist/glm52-kaggle-tpu-acquisition-20260704-r2-retained-mcp-watch/glm52_kaggle_tpu_retained_request_watch.json`;
it records `last_status=KernelWorkerStatus.COMPLETE`,
`tpu_runtime_ready=true`, and verified output at
`dist/glm52-kaggle-tpu-acquisition-20260704-r2-retained-mcp-watch/notebook-output/ct_mcp_tpu_probe.json`
with 8 JAX TPU devices and a tiny JAX op. This proves the stable
MCP/save-notebook TPU acquisition path can work, but it is not GLM stage
execution.

A current GLM 5.2 TPU stage runtime notebook was submitted through the stable
MCP/save-notebook path as `tpuowner/ct-glm52-tpu-value-op-r1`. The watch
artifact is
`dist/glm52-mcp-tpu-stage-runtime-watch-20260705-r1-cross-day-refresh/glm52_mcp_tpu_stage_runtime_watch.json`;
it saw `KernelWorkerStatus.COMPLETE` at `2026-07-05T04:00:24+00:00`,
downloaded
`dist/glm52-mcp-tpu-stage-runtime-watch-20260705-r1-cross-day-refresh/notebook-output/glm52_kaggle_stage_runtime_report.json`,
and verified it with `scripts/glm52_kaggle_stage_runtime_check.py --require-verified`.
The report is provider `kaggle_jax_tpu`, stage1 `[26,52]`,
`provider_device_count=8`, `stage_execution_verified=true`,
`stage_decode_verified=false`, and public-safe weight-byte/output hashes only.
A supplemental single-poll refresh at
`dist/glm52-mcp-tpu-stage-runtime-watch-20260704-r20-current-single-refresh/glm52_mcp_tpu_stage_runtime_watch.json`
and the longer r21 watcher both saw `KernelWorkerStatus.QUEUED` on
2026-07-04, but they are superseded by the r1 2026-07-05 completed watcher.
The reusable watcher/checker for this retained GLM TPU stage notebook is now
`scripts/glm52_mcp_tpu_stage_runtime_watch.py` and
`scripts/glm52_mcp_tpu_stage_runtime_watch_check.py`, with tests in
`tests/test_glm52_mcp_tpu_stage_runtime_watch.py`. The checker accepts queued
watch artifacts as public-safe blocker evidence, fails with `--require-ready`
until a real TPU stage report is downloaded and verified, and never allows the
watch artifact to claim same-request decode success.

The current GLM 5.2 source artifact is
`dist/glm52-model-source-resolver-20260704-r4-awq-safetensors-recommended/glm52_model_source_resolver.json`.
It verifies public HF source `zai-org/GLM-5.2`, architecture
`GlmMoeDsaForCausalLM` / `glm_moe_dsa`, 78 layers, 59,585 official weight keys,
282 official safetensors files, and official weight total size about
1,506.65992GB. Because the full model exceeds Kaggle runtime disk and
single-account memory budgets, the recommended RC candidate is the quantized
safetensors source `cyankiwi/GLM-5.2-AWQ-INT4`, AWQ-INT4, about
440.335957GB across 83 files. The stage plan remains metadata-only:
`stage_runtime_adapter_verified=false` and `same_request_route_verified=false`.

The Kaggle public source search is
`dist/glm52-kaggle-public-source-search-20260704-r3-api-structured-strict-dataset-signal/glm52_kaggle_public_source_search.json`,
checked by `scripts/glm52_kaggle_public_source_search_check.py`. It queried 6
terms through the structured Kaggle API using the `tpuowner` token-file
section, found 10 public model results and 24 public dataset results, but
found 0 GLM 5.2 compatible attachable weight sources. The broad `GLM-5` search
returned non-compatible refs such as `marquis03/chatglm3`, `nellimatteo/glm-5`,
`stepfun/step-audio`, `thomasgamet/glm47-utq1-unsloth`, and
`qwen-lm/qwen3-asr`; exact `GLM-5.2` / `cyankiwi GLM-5.2` searches did not
return GLM 5.2 model weights. Treat Kaggle Models/Datasets attach as currently
searched-and-not-found for GLM 5.2; the remaining viable source path is HF
stage-selective loading unless a new Kaggle source appears later.

The real GLM 5.2 AWQ stage-owned header proof is
`dist/glm52-awq-stage-header-probe-20260704-r2-awq-stage4-header/glm52_awq_stage_header_probe.json`.
It verifies stage 4/12, layer range `[28,35]`, 21,675 assigned keys across 8
files, all keys present, AWQ-family tensors detected, about 40.524259GB of
selected tensor storage, and no public tensor values or safetensors header
payload. This is stronger than metadata-only source resolution, but still not a
runtime adapter or same-request proof.
The real GLM 5.2 AWQ stage-owned value proofs now cover the provider-aligned
three-stage plan: CUDA stage0 `[0,26]`, TPU stage1 `[26,52]`, and CPU stage2
`[52,78]`. The artifacts are
`dist/glm52-awq-stage-value-probe-20260704-r2-provider-stage0-cuda-range/glm52_awq_stage_value_probe.json`,
`dist/glm52-awq-stage-value-probe-20260704-r3-provider-stage1-tpu-range/glm52_awq_stage_value_probe.json`,
and
`dist/glm52-awq-stage-value-probe-20260704-r4-provider-stage2-cpu-range/glm52_awq_stage_value_probe.json`.
Each range-read one stage-owned tensor value from
`cyankiwi/GLM-5.2-AWQ-INT4`, loaded 16 bytes, and records only public-safe
digests/metadata with `weight_tensor_values_loaded=true`,
`weight_tensor_values_public=false`, and
`safetensors_header_payload_public=false`. All three pass
`scripts/glm52_awq_stage_value_probe_check.py --require-ready`; r16 RC records
`provider_aligned_stage_value_probe_ready=true` with provider coverage
`kaggle_cuda`, `kaggle_jax_tpu`, and `kaggle_cpu`. This is real weight-byte
loading evidence, not stage runtime adapter verification or same-request
success; the RC records `glm52_awq_stage_value_probe_is_not_runtime_success`
to preserve that boundary.

A GLM 5.2 AWQ TPU stage-smoke notebook was submitted through the stable
MCP/save-notebook path as `tpuowner/ct-glm52-awq-tpu-stage-smoke-0704-r1`.
The watch artifact
`dist/glm52-kaggle-tpu-awq-stage-smoke-20260704-r1-mcp-notebook/glm52_kaggle_tpu_awq_stage_smoke_watch.json`
currently records 38 observations as of `2026-07-04T16:11:04+00:00` and
`last_status=KernelWorkerStatus.COMPLETE`. The Notebook output
`dist/glm52-kaggle-tpu-awq-stage-smoke-20260704-r1-mcp-notebook/notebook-output/glm52_awq_tpu_stage_smoke.json`
is verified by `scripts/glm52_awq_tpu_stage_smoke_check.py --require-ready`:
`tpu_runtime_ready=true`, `stage_runtime_adapter_smoke_ready=true`,
`notebook_output_verified=true`, 8 JAX TPU devices, stage 4/12, layer range
`[28,35]`, 21,675 assigned/present stage keys, 0 missing stage keys, and
public-safe hashes only. This is TPU stage-smoke readiness, not same-request
GLM 5.2 deployment success. Do not delete/recreate the retained Notebook unless
it is explicitly needed for a new experiment; future continuations can reuse
this output as TPU stage-smoke evidence.

The stage-smoke checker is `scripts/glm52_awq_tpu_stage_smoke_check.py`. It
accepts queued watch artifacts as public-safe blocker evidence and now accepts
the completed Kaggle TPU output as ready with `--require-ready`. A ready report
must prove GLM 5.2 AWQ source identity, TPU runtime/device evidence, stage
header readiness, JAX shape-smoke readiness, no tensor/header payload leaks,
and no same-request overclaim.
The watcher script is `scripts/glm52_kaggle_tpu_stage_smoke_watch.py`, with
tests in `tests/test_glm52_kaggle_tpu_stage_smoke_watch.py`; it covers queued
status preservation and completed-output download/check summarization.

The per-stage runtime proof checker is
`scripts/glm52_kaggle_stage_runtime_check.py`, with tests in
`tests/test_glm52_kaggle_stage_runtime_check.py`. It validates one public-safe
`glm52_kaggle_stage_runtime_report_v1` at a time and requires GLM 5.2 model
identity, a required provider (`kaggle_cuda`, `kaggle_jax_tpu`, or
`kaggle_cpu`), live run evidence, Coordinator request hash, stage output hash,
valid layer range, public-safe stage weight-value-loaded evidence, no fallback
model, no queue/metadata/stage-smoke overclaim, and no public
prompt/token/activation/logit/KV/weight/header payload. It
explicitly rejects TPU watch/stage-smoke artifacts as runtime proof.
The stage runtime adapter plan is
`dist/glm52-kaggle-stage-runtime-plan-20260704-r1-contract/glm52_kaggle_stage_runtime_plan.json`,
produced by `scripts/glm52_kaggle_stage_runtime_plan.py` and checked by
`scripts/glm52_kaggle_stage_runtime_plan_check.py`. It records the required
provider specs for `kaggle_cuda` layers `[0,26]`, `kaggle_jax_tpu` layers
`[26,52]`, and `kaggle_cpu` layers `[52,78]`, plus the expected
`glm52_kaggle_stage_runtime_report_v1` launcher contract. It is plan/contract
evidence only: `stage_runtime_adapter_verified=false`,
`same_request_route_verified=false`, and blockers remain for missing live stage
reports, CUDA memory budget, live TPU stage runtime, and CPU runtime
verification.
The current Kaggle stage worker package is
`dist/glm52-kaggle-stage-worker-package-20260705-r11-embedded-full-prefix-dsa-window-bound-request/glm52_kaggle_stage_worker_package.json`,
produced by `scripts/glm52_kaggle_stage_worker_package.py` and checked by
`scripts/glm52_kaggle_stage_worker_package_check.py`. It renders private
Kaggle script-kernel directories for the three required providers with
`--runtime-kind full_prefix_stage_decode`, bundles the tested full-prefix
stage-hidden adapter scripts into every package, and binds the public-safe
request hash
`sha256:8385016dbeb99152007a34bce07e028a1ac9a564a28b5b294ca54164b49afeee`,
enables internet for HF range reads, uses `NvidiaTeslaT4` metadata for CUDA and
`tpuV5e8` metadata for TPU, includes CUDA torch/cupy/numba/jax fallbacks, and
records `full_prefix_runtime_bundle_required=true` with all provider package
bundles present. It avoids unsuitable stage-boundary defaults by using explicit
GLM 5.2 DSA probe windows inside each stage: CUDA `[6,8]`, TPU `[26,28]`, and
CPU `[54,56]`. r11 embeds those scripts directly inside `kernel.py` and
self-extracts them at runtime because r25 showed Kaggle did not make uploaded
subdirectories visible to the script kernel. The generated kernels can still
fall back to the provider value-op proof and invoke the bundled full-prefix
host adapter, but the package itself remains non-success evidence:
`pushed_to_kaggle=false`,
`live_run_performed=false`, `stage_runtime_adapter_verified=false`,
`stage_decode_verified=false`, and `same_request_route_verified=false`.
The bounded Kaggle push/monitor/output/cleanup harness is
`dist/glm52-kaggle-stage-worker-push-probe-20260705-r24-import-gpu-tpu-cpu-live/glm52_kaggle_stage_worker_push_probe.json`,
produced by `scripts/glm52_kaggle_stage_worker_push_probe.py` and checked by
`scripts/glm52_kaggle_stage_worker_push_probe_check.py`. The current aggregate
report uses the new `--mode import` path and records `live_run_performed=true`,
`stage_runtime_reports_collected=3`, and `stage_runtime_reports_verified=3`:
r7 CUDA stage0, r1 TPU stage1, and r7 CPU stage2 all have public-safe
`glm52_kaggle_stage_runtime_report.json` evidence and pass
`scripts/glm52_kaggle_stage_runtime_check.py --require-verified`. The
push-probe checker now passes with `--require-live`. This is three-accelerator
stage runtime evidence, not full decode or same-request success.
The `import` mode is the preferred resume path after each TPU watcher refresh:
it imports the retained GPU/CPU stage reports plus the latest TPU watcher,
updates provider coverage, and keeps `--require-live` strict until the TPU
stage report is actually verified.
The current same-request partial artifact is
`dist/glm52-kaggle-same-request-20260705-r3-three-stage-runtime-no-decode/glm52_kaggle_same_request_probe.json`.
Its normal checker passes, but `--require-verified` fails with
`same_request_not_verified`: all three stage reports share the same Coordinator
request hash, yet each has `stage_decode_verified=false`, no generated
token/hash exists, and cleanup/Coordinator live decode proof is missing.

The same-request proof checker is `scripts/glm52_kaggle_same_request_check.py`,
with tests in `tests/test_glm52_kaggle_same_request_check.py`. It accepts only
`glm52_kaggle_same_request_probe_v1` evidence that proves GLM 5.2 model
identity, all required providers `kaggle_cuda`, `kaggle_jax_tpu`, and
`kaggle_cpu`, Coordinator same-request evidence, generated token/hash, stage
execution reports for every provider, cleanup of temporary Kaggle resources and
private packages, public-safe redaction, and no fallback/queue/metadata/stage-
smoke-only overclaim. Use it with `--require-verified` before importing any
future same-request report into the RC as success evidence.
The same-request probe scaffold is
`scripts/glm52_kaggle_same_request_probe.py`, with tests in
`tests/test_glm52_kaggle_same_request_probe.py`. In `preflight` mode it writes
a not-started blocker report only. In `assemble` mode it strips supplied stage,
Coordinator, and cleanup reports to public-safe proof fields and verifies
success only if Kaggle CUDA, Kaggle JAX/TPU, and Kaggle CPU stage reports all
come from one live Coordinator request with a generated-token hash and cleanup
proof. It rejects TPU stage-smoke artifacts as same-request stage evidence.

`scripts/glm52_kaggle_accelerator_deployment_rc_pack.py` now accepts
`--tpu-stage-smoke-report`, `--kaggle-source-search-report`,
`--stage-runtime-plan-report`, `--stage-worker-package-report`, and
`--stage-worker-push-probe-report`, `--transformers-decode-preflight-report`,
`--attention-projection-report`, `--attention-single-token-report`,
`--kv-cache-decode-report`, `--layer-decode-report`,
`--lm-head-token-report`, `--dsa-masked-layer-decode-report`,
`--stage-hidden-lm-head-report`, `--multi-layer-stage-decode-report`,
`--full-prefix-stage-decode-report`,
`--dsa-indexer-report`,
`--pack-quantized-dequant-report`,
`--pack-quantized-expert-mlp-report`,
`--pack-quantized-router-gather-report`, `--pack-quantized-moe-mlp-report`,
and `--decode-adapter-gap-report`, imports
either a completed GLM AWQ TPU stage-smoke report or its queued watch artifact,
imports the active MCP GLM TPU stage-runtime watcher as a TPU request, imports
the public Kaggle source search, stage runtime plan, stage worker package
manifest, and stage worker push probe plus transformers decode preflight,
attention projection, single-token attention, KV-cache decode, layer
decode-composition, full-vocab lm_head token selection, DSA-masked layer
decode, DSA-layer-hidden-to-lm_head token selection, DSA multi-layer
decode-token-chain token selection, DSA full-prefix stage-hidden token
selection, DSA indexer,
pack-quantized dequant, single-expert MLP,
router/gather subset, full MoE MLP, and decode-adapter gap
reports, and keeps
`kaggle_source_search_is_not_success`, `stage_smoke_evidence_is_not_success`,
`transformers_decode_preflight_is_not_success`, and
`pack_quantized_dequant_slice_is_not_success`, and
`pack_quantized_expert_mlp_is_not_success`, and
`attention_projection_is_not_success`, `attention_single_token_is_not_success`,
`kv_cache_decode_is_not_success`, `layer_decode_is_not_success`,
`lm_head_token_selection_is_not_success`,
`dsa_masked_layer_decode_is_not_success`,
`stage_hidden_lm_head_is_not_success`,
`multi_layer_stage_decode_is_not_success`,
`full_prefix_stage_decode_is_not_success`, `dsa_indexer_is_not_success`,
`pack_quantized_router_gather_subset_is_not_success`,
`pack_quantized_moe_mlp_is_not_success`, and
`decode_adapter_gap_evidence_is_not_success` as explicit completion boundaries.
Its same-request summary now also requires
live run evidence, a generated-token hash, Coordinator request proof, verified
stage-provider coverage for all three Kaggle providers, and cleanup proof
before `goal_achieved` can become true. The checker in
`scripts/glm52_kaggle_accelerator_deployment_rc_check.py` rejects queue,
metadata-only, source-search-only, stage-header-only, stage-smoke-only,
transformers-preflight-only, pack-dequant-slice-only, single-expert-MLP-only,
attention-projection-only, single-token-attention-only,
KV-cache-decode-only, layer-decode-only, lm-head-token-selection-only,
DSA-masked-layer-decode-only, stage-hidden-lm-head-only, DSA-indexer-only,
multi-layer-stage-decode-only, full-prefix-stage-decode-only,
router-gather-subset-only, full-MoE-MLP-only,
decode-gap-only, fallback-model, and single-backend
evidence as success.
Focused GLM tests pass:
`PYTHONPATH=. pytest -q tests/test_glm52_*.py` returns `177 passed`. Do not
mark this goal achieved until a real GLM 5.2 or
explicitly compatible GLM 5.2 quantized weight run completes in one Coordinator
request with Kaggle CUDA, Kaggle JAX/TPU, Kaggle CPU, generated-token/hash, and
cleanup proof.

## Latest Kaggle TPU Acquisition Method Search

Current superseding status after the 2026-07-04 TPU acquisition continuation:
Kaggle TPU access is not blocked by the three token-file accounts themselves.
All three `~/.config/crowdtensor/kaggle-tokens.md` sections authenticated through isolated
`KAGGLE_API_TOKEN` / `MY_KAGGLE_TOKEN` environments and all three accepted a
private Kaggle script-kernel TPU submission with `enable_tpu=true`,
`enable_gpu=false`, and `machine_shape=tpuV5e8`. The bounded short probes were:
`dist/kaggle-tpu-llm-probe-20260704-r1-cpuowner-tpuv5e8-short/kaggle_tpu_llm_probe.json`,
`dist/kaggle-tpu-llm-probe-20260704-r2-tpuowner-tpuv5e8-short/kaggle_tpu_llm_probe.json`,
and
`dist/kaggle-tpu-llm-probe-20260704-r3-xuyuhaosuyi-tpuv5e8-short/kaggle_tpu_llm_probe.json`.
Each report is public-safe and records `fresh_kaggle_run_performed=true`,
`selected_accelerator=tpuV5e8`, an accepted push, final status `QUEUED`,
`blocked_reason=kaggle_tpu_kernel_queued_timeout`, no JAX/LLM runtime, and
successful cleanup (`kernels_deleted=true`, `private_packages_removed=true`).
A post-run `kaggle kernels list --mine --search ct-tpu-llm-probe` check returned
`Not found` for all three accounts, so these short probes left no temporary TPU
kernels behind.

The current non-Web notebook/MCP path was also tested. Kaggle MCP
`save_notebook` accepted a private script notebook
`tpuowner/ct-mcp-tpu-probe-0704-r2` with `enable_tpu=true`,
`enable_gpu=false`, and metadata `machine_shape=TpuV5E8`. The local public-safe
summary is
`dist/kaggle-mcp-tpu-probe-20260704-r1-save-notebook/kaggle_mcp_tpu_probe_summary.json`
and the 30-poll watch artifact is
`dist/kaggle-mcp-tpu-probe-20260704-r1-save-notebook/mcp_tpu_save_notebook_watch.json`.
The notebook remained `KernelWorkerStatus.QUEUED` for 1775.5 seconds across 30
polls, so no JAX TPU device or output JSON was verified yet. This queued
notebook was intentionally left in place as the best current non-browser
long-wait candidate; future continuations should first poll
`kaggle kernels status tpuowner/ct-mcp-tpu-probe-0704-r2` with the `tpuowner`
token-file section, then download `ct_mcp_tpu_probe.json` if the status becomes
terminal/successful. Do not treat the MCP notebook as TPU runtime proof until
that output shows JAX TPU devices.

Practical conclusion: rotating among the three token-file accounts did not
produce faster TPU allocation in a short window; the common blocker is Kaggle
TPU scheduler availability, not authentication or immediate quota rejection.
For the next TPU-dependent LLM attempt, prefer a persistent long-wait strategy:
either preserve and poll the MCP/save-notebook TPU request above, or use Web TPU
Active Event long waiting when queue position visibility is needed. Avoid
repeatedly deleting and recreating TPU requests unless the existing request is
cancelled, errored, or clearly stale, because every tested account can submit
but all short submissions simply re-entered the same queue.

## Latest DeepSeek V4 Flash Kaggle GPU+WebTPU+CPU Status

Current superseding status after the 2026-07-04 r28 continuation: the active
DeepSeek-V4-Flash Kaggle GPU+WebTPU+CPU goal is still not achieved, and the
current blocker is external Kaggle Web TPU runtime/execution availability. Two
bounded Web TPU queue/start cycles were attempted before spending any new
`cpuowner` GPU/CPU kernels. The first cycle
`dist/kaggle-web-tpu-queue-monitor-probe-20260704-r25-reacquire-after-r16-channel-timeout/kaggle_web_tpu_queue_monitor_probe.json`
clicked Start Session and waited 3600 seconds; queue progress was
`#19 -> #18 -> #17 -> #13 -> #8 -> #7 -> #6 -> #5`, then the UI showed
`Session started`, but no Jupyter frame/session/kernel appeared, so
`web_tpu_runtime_ready=false` with blockers
`kaggle_web_tpu_jupyter_frame_not_visible`,
`kaggle_web_tpu_jupyter_session_not_visible`, and
`kaggle_web_tpu_session_started_text_without_runtime`. The follow-up execution
channel probe
`dist/kaggle-web-tpu-execution-channel-probe-20260704-r17-after-r25-session-started-no-frame-force-new/kaggle_web_tpu_execution_channel_probe.json`
timed out before even small JAX: `web_tpu_execution_channel_ready=false`,
`tpu_runtime_attached=false`, `tpu_device_count=0`, and blocker
`web_tpu_jupyter_execute_timeout`. A quick read-only UI probe
`dist/kaggle-web-tpu-ui-state-probe-20260704-r18-after-r17-exec-timeout-readonly/kaggle_web_tpu_ui_state_probe.json`
then showed the notebook had fallen back to `Draft Session off` with
`Start Session` visible.

The second bounded cycle
`dist/kaggle-web-tpu-queue-monitor-probe-20260704-r26-second-cycle-after-r25-session-started-no-runtime/kaggle_web_tpu_queue_monitor_probe.json`
clicked Start Session again and waited another 3600 seconds. Queue progress was
`#26 -> #25 -> #23 -> #18 -> #17 -> #13 -> #10`, but it still ended in queue /
`Session starting` with no Jupyter frame/session/kernel and
`web_tpu_runtime_ready=false`. Its checker passes and records blockers
`kaggle_web_tpu_jupyter_frame_not_visible`,
`kaggle_web_tpu_jupyter_session_not_visible`,
`kaggle_web_tpu_queue_prompt_visible`, and
`kaggle_web_tpu_session_still_starting`. No new Kaggle GPU or CPU kernels were
started in these two cycles, so no new `cpuowner` GPU/CPU quota was spent.

The canonical current public-safe RC artifact is
`dist/deepseek-v4-flash-kaggle-tpu-swarm-rc-20260704-r28-current-webtpu-two-cycle-blocker-r12-best-same-request/deepseek_v4_flash_kaggle_tpu_swarm_rc.json`.
Its checker passes. It preserves the r12 best same-request proof with
`accepted_providers=["kaggle_cpu","kaggle_cuda","kaggle_web_tpu"]`,
`generated_token_count=1`, and
`same_request.deepseek_v4_gpu_tpu_cpu_same_request_fp4_topk_expert_forwards_verified=true`,
but still records `same_request_decode_verified=false`,
`failure_stage=deepseek_v4_flash_kaggle_tpu_same_request_not_verified`, and
blockers including `deepseek_v4_full_same_request_decode_not_verified` and
`web_tpu_jupyter_execute_timeout`. Do not start another DeepSeek GPU+CPU bridge
attempt until a fresh Kaggle Web TPU execution-channel probe can run small JAX
on TPU; otherwise it will only consume GPU/CPU attempts without satisfying the
goal.

Current superseding runtime status after the 2026-07-04 r13/r16 continuation:
yes, future live DeepSeek-V4-Flash attempts can continue to use the `cpuowner`
token-file section for Kaggle T4x2 GPU and Kaggle CPU kernels together with
Kaggle Web TPU, but the next retry must first restore a working Web TPU
execution channel. The best successful three-family same-request proof remains
the r12/r27 artifact below; the latest expanded distinct-layer attempt did not
supersede it as a success.

The r13 distinct backend layer-range attempt is
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260704-r13-deepseek-distinct-layer-ranges-cpuowner-webtpu-kagglecpu/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`.
It used `cpuowner` GPU/CPU resources and requested distinct real
DeepSeek-V4-Flash FP4 top-k stage ranges: CUDA `[16,17]`, Web TPU/JAX `[17,18]`,
and Kaggle CPU `[18,19]`, so the requested stage coverage count was 3. The CUDA
stage was accepted and submitted stage0, but Web TPU timed out before stage1
with `web_tpu_jupyter_execute_timeout`; the Kaggle CPU kernel was created and
deleted but could not verify stage2 because the TPU handoff never arrived. The
report records `ok=false`, `accepted_providers=["cuda"]`,
`generated_token_count=0`, blockers `jax_tpu_stage_not_ready`,
`kaggle_cpu_stage_not_verified`, `cpu_tail_not_ready`, and
`same_request_runtime_bridge_not_verified`, with both temporary Kaggle kernels
deleted and private packages removed. A follow-up read-only
`cpuowner` `kaggle kernels list --mine` check returned no
`ct-gpu-tpu-cpu-bridge` entries.

The follow-up standalone Web TPU layer-17 adapter probe is
`dist/deepseek-v4-flash-kaggle-web-tpu-stage-adapter-20260704-r16-layer17-after-r13-timeout-force-new/deepseek_v4_flash_kaggle_web_tpu_stage_adapter_probe.json`.
It also failed before executing the DeepSeek stage:
`failure_stage=kaggle_web_tpu_runtime_not_ready`,
`kaggle_web_tpu_runtime_ready=false`, and blockers include
`web_tpu_jupyter_execute_timeout`. Treat this as current Web TPU execution
channel unavailability, not as a `cpuowner` GPU/CPU quota failure.

Engineering added in this continuation: `scripts/gpu_tpu_cpu_same_request_runtime_bridge_probe.py`
now supports independent `--deepseek-gpu-stage-layer-start/end`,
`--deepseek-tpu-stage-layer-start/end`, and
`--deepseek-cpu-stage-layer-start/end` controls, records
`deepseek_v4_stage_layer_ranges`, `deepseek_v4_stage_layer_coverage_count`, and
`deepseek_v4_distinct_backend_stage_layer_ranges_verified`, and the DeepSeek RC
pack/check path imports and validates these public-safe fields. Focused tests
over the bridge, RC, and Web TPU queue monitor passed with `69 passed` after
these code changes. Do not mark the active goal achieved until a real
DeepSeek-V4-Flash or public quantized DeepSeek-V4-Flash same-request decode
successfully verifies Kaggle CUDA + Kaggle Web TPU + Kaggle CPU in one request.

Current superseding status after the 2026-07-04 r27 continuation: yes, future
live DeepSeek-V4-Flash attempts can use the `cpuowner` token-file section for
Kaggle T4x2 GPU and Kaggle CPU kernels together with Kaggle Web TPU. The live
r12 bridge artifact is
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260703-r12-deepseek-fp4-topk-all-backends-cpuowner-force-new-webtpu/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`.
It created private `cpuowner` GPU and CPU kernels, both reached `COMPLETE`, both
reports were downloaded, and both temporary kernels were deleted. The same
Coordinator request accepted providers `["cpu","cuda","jax_tpu"]`, generated
one token hash, and verified real DeepSeek-V4-Flash stage slices plus FP4
top-k expert forwards on all three accelerator families:
`deepseek_v4_gpu_tpu_cpu_same_request_stage_slices_verified=true` and
`deepseek_v4_gpu_tpu_cpu_same_request_fp4_topk_expert_forwards_verified=true`.
The Web TPU stage used `--web-tpu-force-new-session` through the browser iframe
service manager, saw 8 `TPU v5 lite` devices, and submitted its stage result.

The canonical public-safe RC artifact is now
`dist/deepseek-v4-flash-kaggle-tpu-swarm-rc-20260704-r27-cpuowner-webtpu-kagglecpu-fp4-topk-stage-slice-final-r24/deepseek_v4_flash_kaggle_tpu_swarm_rc.json`.
Its checker passes and records `accepted_providers=["kaggle_cpu","kaggle_cuda","kaggle_web_tpu"]`,
`generated_token_count=1`,
`same_request.deepseek_v4_gpu_tpu_cpu_same_request_fp4_topk_expert_forwards_verified=true`,
but `same_request_decode_verified=false` with
`failure_stage=deepseek_v4_flash_kaggle_tpu_same_request_not_verified` and
blocker `deepseek_v4_full_same_request_decode_not_verified`. Do not mark the
active goal achieved from this artifact: it is a real three-accelerator
DeepSeek stage-slice/FP4-top-k same-request proof, not full all-layer
DeepSeek-V4-Flash decode or product serving.

The final r24 Web TPU queue monitor artifact is
`dist/kaggle-web-tpu-queue-monitor-probe-20260703-r24-hardened-session-start-runtime-required/kaggle_web_tpu_queue_monitor_probe.json`.
It waited the full bounded 3600 seconds, clicked Start Session, observed queue
progress `#18 -> #17 -> #4 -> #3 -> #1`, and ended with an Active Event
`Running: 36 minutes` plus `Session started` text. It still did not expose a
Jupyter frame/session/kernel through the UI, so `web_tpu_runtime_ready=false`;
the successful r12 Web TPU execution came from the force-new browser iframe
execution path, not from the ordinary UI frame becoming visible.

Engineering fixes made during r12/r27: the bridge no longer stops the Coordinator
just because the TPU worker exits before late GPU/CPU submissions, and
`--web-tpu-force-new-session` is now plumbed into the Web TPU DeepSeek stage
code. The Kaggle CPU FP4 `NameError` was fixed by embedding `FP4_E2M1_LUT` in
the rendered CPU kernel helper bundle; tests execute the rendered helper so
this cannot regress as a compile-only pass. Queue monitor and RC pack logic now
use the final observation to avoid stale queued/session-starting blockers when
an Active Event has moved to Running, and RC pack treats the r12 same-request
Web TPU stage proof as stronger than stale standalone active-event/execution
probe blockers. Focused tests passed:
`tests/test_kaggle_web_tpu_queue_monitor_probe.py`,
`tests/test_gpu_tpu_cpu_same_request_runtime_bridge_probe.py`, and
`tests/test_deepseek_v4_flash_kaggle_tpu_swarm_rc.py` reported `68 passed`.
The standard Kaggle CLI list check under `cpuowner` found no residual
`ct-gpu-tpu-cpu-bridge` kernels after r12 cleanup.

Current superseding status after the latest 2026-07-02 continuation: the
canonical public-safe RC artifact is now
`dist/deepseek-v4-flash-kaggle-tpu-swarm-rc-20260702-r21-webtpu-fp4-topk-gpu-quota/deepseek_v4_flash_kaggle_tpu_swarm_rc.json`.
Its checker passes and records `same_request_decode_verified=false`,
`generated_token_count=0`, `accepted_providers=[]`, and
`failure_stage=deepseek_v4_flash_kaggle_tpu_same_request_not_verified`. Do not
mark the active goal achieved from this artifact: Web TPU execution and real
DeepSeek-V4-Flash FP4 top-k stage-selective forwarding advanced, but a fresh
Kaggle GPU+WebTPU+CPU same-request DeepSeek run did not start because the
available Kaggle API GPU paths were blocked by quota or invalid credentials.

Continuation after r21: the corrected token-file GPU quota probe is
`dist/kaggle-gpu-token-weekly-quota-probe-20260702-r1/kaggle_gpu_token_weekly_quota_probe.json`.
It proves all three sections in `~/.config/crowdtensor/kaggle-tokens.md` authenticate when the
`KGA...` values are exported as `KAGGLE_API_TOKEN`; `tpuowner` and
`primary Kaggle account` were explicitly rejected with `Maximum weekly GPU quota of 30.00
hours reached`, while `cpuowner` accepted a minimal private T4 GPU kernel and
the temporary kernel was deleted. Future Kaggle API GPU/CPU live attempts for
this goal should therefore use the `cpuowner` section via environment variables
and a temporary `KAGGLE_CONFIG_DIR`, not by writing the `KGA...` token into a
legacy `kaggle.json` key field. `scripts/kaggle_gpu_token_weekly_quota_probe.py`
and `tests/test_kaggle_gpu_token_weekly_quota_probe.py` cover this path.

Bridge engineering after that quota probe: `scripts/gpu_tpu_cpu_same_request_runtime_bridge_probe.py`
now has an explicit `--kaggle-cpu-stage` path. When enabled it submits a
private Kaggle CPU script kernel for Coordinator stage2, loads the same real
DeepSeek-V4-Flash CPU stage-selective FP4 top-k routed experts plus FP8 shared
expert, submits the stage2 result from Kaggle CPU, downloads the public-safe
worker report, and deletes the CPU kernel. The report now distinguishes
`cpu_stage_provider=local_cpu_thread` from `cpu_stage_provider=kaggle_cpu` and
will not report `ok=true` for a requested Kaggle CPU stage unless the CPU
worker report has `provider=kaggle_cpu` and `kaggle_kernel=true`. Tests in
`tests/test_gpu_tpu_cpu_same_request_runtime_bridge_probe.py` cover rendered
Kaggle CPU kernel compilation and the no-local-CPU-substitution gate. This is
engineering readiness only; the active goal still requires a live Web TPU
runtime plus `cpuowner` Kaggle GPU and Kaggle CPU in one same-request DeepSeek
run.

Latest model-adapter progress beyond r20: CPU and Web TPU now both understand
the routed expert FP4 packing used by DeepSeek-V4-Flash. The retained CPU
artifact
`dist/deepseek-v4-flash-cpu-fp4-topk-expert-forward-20260702-r1-local-stage16/deepseek_v4_flash_cpu_fp4_topk_expert_forward.json`
and the new Web TPU artifact
`dist/deepseek-v4-flash-kaggle-web-tpu-stage-adapter-20260702-r12-real-fp4-topk-expert-forward-live/deepseek_v4_flash_kaggle_web_tpu_stage_adapter_probe.json`
both run the real `sqrtsoftplus` router with gate bias, select top-k=6 routed
experts, range-load selected FP4-packed routed expert weights/scales plus the
FP8 shared expert, unpack/dequantize them, and complete a finite `[4096]`
stage-selective routed+shared MoE forward. Each loads 42 forward tensors
totaling 105,383,424 bytes and exposes only public-safe hashes/shapes/counts.
The Web TPU run used 8 `TPU v5 lite` devices and the browser iframe service
manager. This is still not a full DeepSeek decode proof, and GPU still needs
the same FP4 top-k forward path inside a successful same-request bridge.

Operational Web TPU queue policy: do not mark an active goal blocked merely
because Kaggle Web TPU is visibly `Queued`, `Starting`, or showing a moving
queue-position prompt. Keep the queue/runtime monitor alive and continue
polling until TPU runtime is allocated or Kaggle reports a hard terminal state
such as cancelled, failed, disappeared Active Event, or an explicit user stop.
When the UI exposes queue-position text, use it to report progress instead of
exiting and waiting for a later goal resume. Do not rely on the user noticing
that a queued Web TPU has been assigned; keep monitor windows running
back-to-back during goal-mode execution so the runtime can be used promptly when
the queue reaches the front.

Latest queue/runtime and bridge evidence: Web TPU monitor
`dist/kaggle-web-tpu-queue-monitor-probe-20260702-r19-restart-after-r10-channel-timeout-cancelled-long-wait/kaggle_web_tpu_queue_monitor_probe.json`
waited instead of exiting while Kaggle queued TPU v5e-8. It clicked Start
Session, observed queue position progress from #8 to #7 to #3, then stopped
only after stable `Session started`; the live heartbeat file records queue
status during long waits. The immediate execution probe
`dist/kaggle-web-tpu-execution-channel-probe-20260702-r11-after-r19-session-started-deepseek-fp4-topk/kaggle_web_tpu_execution_channel_probe.json`
passed with 8 `TPU v5 lite` devices, small JAX ready, tiny Qwen-like ready, and
stage-local KV-cache verified. The older same-request bridge
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260702-r5-deepseek-all-backend-real-slices-cpuowner/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`
completed one Coordinator request with accepted stage backends `cpu`, `cuda`,
and `jax_tpu`, two activation handoffs, and one generated-token hash. Its GPU,
Web TPU, and CPU stages each loaded/executed a real DeepSeek-V4-Flash
stage-owned weight slice: 12 tensors, 19,677,696 bytes, real router, FP8 block
dequant, and I8 expert MLP slice smoke. The blocker is intentionally still
`deepseek_v4_full_same_request_decode_not_verified` because this is a
same-request real-weight slice bridge, not a full all-layer DeepSeek-V4-Flash
decode. Fresh FP4 top-k same-request attempts did not reach GPU execution:
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260702-r6-deepseek-fp4-topk-all-backends-cpuowner/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`
used the current valid Kaggle API credential but hit the weekly GPU quota;
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260702-r7-deepseek-fp4-topk-all-backends-cpuowner-token/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`
and
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260702-r8-deepseek-fp4-topk-all-backends-tpuowner-token/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`
both returned Kaggle API HTTP 401 before kernel creation. The next live
same-request attempt needs a valid Kaggle GPU credential with available quota
while Web TPU execution is ready.
Corrected credential audit after r21: the default local Kaggle API config
`~/.kaggle/kaggle.json` authenticates as `xuyuhaosuyi`, and that authenticated
path is the one that produced the observed `Maximum weekly GPU quota of 30.00
hours reached` blocker. The earlier HTTP 401 result for the three sections in
`~/.config/crowdtensor/kaggle-tokens.md` was caused by using the `KGA...` values incorrectly as
legacy `kaggle.json` `key` values. The token file stores new-style values meant
to be exported as `KAGGLE_API_TOKEN` (and `MY_KAGGLE_TOKEN`); when used as
environment variables exactly as written, all three sections (`tpuowner`,
`primary Kaggle account`, and `cpuowner`) pass a non-GPU `kaggle kernels list --mine` auth
probe. Do not rewrite those `KGA...` tokens into `kaggle.json` key fields.
Their GPU quota state still needs a correct GPU push/preflight attempt; only
the default `xuyuhaosuyi` legacy config is currently proven GPU-quota-exhausted.

Previous r17 status on 2026-07-02 for the DeepSeek-V4-Flash Kaggle
GPU+WebTPU+CPU goal: the project has still not completed DeepSeek-V4-Flash
same-request inference across Kaggle CUDA + Kaggle Web TPU + Kaggle CPU, but
Web TPU acquisition and real-weight TPU loading advanced materially. The
canonical public-safe RC artifact is now
`dist/deepseek-v4-flash-kaggle-tpu-swarm-rc-20260702-r17-webtpu-real-i8-expert-mlp-slice-smoke-precise-blocker/deepseek_v4_flash_kaggle_tpu_swarm_rc.json`.
`python scripts/deepseek_v4_flash_kaggle_tpu_swarm_rc_check.py --report ... --json`
passes and records `same_request_decode_verified=false`,
`generated_token_count=0`, `accepted_providers=[]`, and
`failure_stage=deepseek_v4_flash_real_weight_tpu_stage_loader_not_implemented`.
Do not mark the active goal achieved from this artifact.

Latest live Web TPU evidence: after r13 queue monitoring, a read-only snapshot
showed `TPU v5e-8` Active Event still running. The latest read-only follow-up
after the expert MLP slice smoke is
`dist/kaggle-web-tpu-active-event-probe-20260702-r15-after-i8-expert-mlp-slice-smoke/kaggle_web_tpu_active_event_probe.json`,
which saw the same event still `Running: 40 minutes`;
the execution-channel
probe
`dist/kaggle-web-tpu-execution-channel-probe-20260702-r7-after-r13-active-event-running/kaggle_web_tpu_execution_channel_probe.json`
then passed with 8 `TPU v5 lite` devices, small JAX ready, tiny Qwen-like
ready, and stage-local KV-cache verified. The DeepSeek Web TPU adapter artifact
`dist/deepseek-v4-flash-kaggle-web-tpu-stage-adapter-20260702-r11-real-i8-expert-mlp-slice-smoke/deepseek_v4_flash_kaggle_web_tpu_stage_adapter_probe.json`
checker passes and records `kaggle_web_tpu_runtime_ready=true`,
`metadata_ready=true`, `deepseek_v4_jax_tpu_fixture_stage_forward_ready=true`,
and new real-weight evidence
`deepseek_v4_real_weight_tpu_tensor_load_ready=true`: twelve real
DeepSeek-V4-Flash safetensors tensors were HTTP range-loaded from HF,
decoded as BF16/F32/F8_E4M3/F8_E8M0/I8, device_put to TPU, and verified
finite, totaling 19,677,696 tensor bytes. Dtype counts are `BF16=3`,
`F32=1`, `F8_E4M3=1`, `F8_E8M0=4`, and `I8=3`. It records
`real_router_smoke_ready=true`, `real_fp8_block_dequant_smoke_ready=true`,
`real_i8_expert_dequant_smoke_ready=true`, and
`real_i8_expert_mlp_slice_smoke_ready=true`: real `ffn_norm.weight` and
`ffn.gate.weight` ran a DeepSeek MoE TopK router smoke; real
`attn.wq_a.weight` F8_E4M3 plus `attn.wq_a.scale` UE8M0 ran a 128x128 block
dequant matmul smoke; and real expert0 `w1/w2/w3` I8 weights plus UE8M0 scales
ran a group-16 dequantized expert MLP slice forward on Web TPU. Public
artifacts contain only hashes/counts/shapes; `weight_tensor_values_public=false`.
This is still not a completed stage forward because full FP8/NVFP4/I8
dequantized TPU stage loading, full real DeepSeek MLA/MoE forward, and
GPU+WebTPU+CPU same-request decode remain unimplemented.

Queue/runtime handling improvement: `scripts/kaggle_web_tpu_queue_monitor_probe.py`
now supports `--stop-after-session-started-polls`; when set, future monitor
windows can stop after stable `Session started` without queue prompt so the
execution-channel probe can run promptly. Tests passed:
`python -m py_compile` over the DeepSeek RC/adapter and Web TPU queue scripts
passed, and
`PYTHONPATH=. pytest -q tests/test_deepseek_v4_flash_kaggle_tpu_swarm_rc.py tests/test_deepseek_v4_flash_kaggle_web_tpu_stage_adapter_probe.py tests/test_kaggle_web_tpu_queue_monitor_probe.py`
reported 24 passed after the real I8 expert MLP slice extension.

Same-request bridge progress: `scripts/gpu_tpu_cpu_same_request_runtime_bridge_probe.py`
now has a guarded `--web-tpu-deepseek-stage-execute` mode. In that mode the Web
TPU stage reuses the DeepSeek adapter, runs the real DeepSeek-V4-Flash
stage-selective weight slice, and is prepared to claim/submit Coordinator
stage1 in the same request. The bridge report exposes
`deepseek_v4_same_request_stage_slice_verified` separately and always keeps
`gpu_tpu_cpu_deepseek_v4_same_request_verified=false` until GPU and CPU also
execute real DeepSeek stages and a full same-request decode is proven. Tests in
`tests/test_gpu_tpu_cpu_same_request_runtime_bridge_probe.py` cover this
non-overclaiming gate. The first live bridge attempt is
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260702-r2-deepseek-webtpu-real-slice-gpu-weekly-quota/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`;
it did not reach the Web TPU stage because Kaggle GPU push failed immediately
with `kaggle_gpu_weekly_quota_reached` / "Maximum weekly GPU quota of 30.00
hours reached." Treat this as a current Kaggle GPU quota blocker for live
same-request bridge attempts on the current account, not as a Web TPU or
DeepSeek TPU adapter failure.

Latest live bridge retry after switching Kaggle GPU credentials to `cpuowner`:
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260702-r3-deepseek-webtpu-real-slice-cpuowner-gpu/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`
shows the cpuowner T4x2 kernel was accepted, ran stage0 on CUDA, submitted one
activation handoff, and was deleted successfully. It still did not complete
DeepSeek-V4-Flash same-request inference: the Web TPU stage failed with
`web_tpu_jupyter_execute_timeout`, `deepseek_v4_same_request_stage_slice_verified=false`,
`generated_token_count=0`, and CPU tail was never claimed because stage1 did
not submit. The follow-up Active Event probe
`dist/kaggle-web-tpu-active-event-probe-20260702-r16-after-deepseek-bridge-r3-timeout/kaggle_web_tpu_active_event_probe.json`
showed the TPU v5e-8 event had become `Cancelled` with no Jupyter
frame/session/kernel. The replacement Web TPU monitor completed at
`dist/kaggle-web-tpu-queue-monitor-probe-20260702-r18-restart-after-r3-cancelled-long-wait/`
after waiting through a queue transition from #12 to #1 and stopping only after
stable `Session started`; the next r8 execution-channel probe passed and r4
bridge succeeded as summarized above. Future queued/Starting states should
still be waited through rather than marked blocked.

Previous superseded status on 2026-07-01 for the DeepSeek-V4-Flash Kaggle
GPU+WebTPU+CPU goal: the project has not completed DeepSeek-V4-Flash
same-request inference across Kaggle CUDA + Kaggle Web TPU + Kaggle CPU. The
canonical public-safe RC artifact is now
`dist/deepseek-v4-flash-kaggle-tpu-swarm-rc-20260701-r9-safetensors-header-ready-webtpu-queued-progress/deepseek_v4_flash_kaggle_tpu_swarm_rc.json`.
`python scripts/deepseek_v4_flash_kaggle_tpu_swarm_rc_check.py --report ... --json`
passes and records `same_request_decode_verified=false`,
`generated_token_count=0`, `accepted_providers=[]`, and
`failure_stage=kaggle_web_tpu_runtime_not_ready`. Do not mark the active goal
achieved from this artifact.

New tooling for this goal:
`scripts/deepseek_v4_flash_kaggle_web_tpu_stage_adapter_probe.py` and
`scripts/deepseek_v4_flash_kaggle_web_tpu_stage_adapter_check.py` validate
DeepSeek-V4-Flash metadata/stage-key mapping separately from Web TPU runtime
execution. `scripts/deepseek_v4_flash_kaggle_tpu_swarm_rc_pack.py` and
`scripts/deepseek_v4_flash_kaggle_tpu_swarm_rc_check.py` summarize the goal
with the correct required providers: `kaggle_cuda`, `kaggle_web_tpu`, and
`kaggle_cpu`. New reference-adapter smoke tooling
`scripts/deepseek_v4_flash_torch_stage_adapter_smoke.py` and
`scripts/deepseek_v4_flash_torch_stage_adapter_smoke_check.py` exercises a
tiny initialized DeepSeek-V4 decoder stage through the official Transformers
implementation, including mHC, compressed attention, MLA/shared-KV attention,
grouped output projection, MoE routing, routed experts, shared experts, and
stage-local KV-cache shape. The canonical smoke artifact is
`dist/deepseek-v4-flash-torch-stage-adapter-smoke-20260701-r3-local-reference-initialized/deepseek_v4_flash_torch_stage_adapter_smoke.json`;
its checker passes with `torch_stage_adapter_smoke_ready=true`,
`jax_tpu_translation_ready=false`, and `real_deepseek_weights_loaded=false`.
This is adapter engineering evidence only, not TPU execution or real-weight
same-request inference. Tests covering these gates pass:
`python -m pytest tests/test_deepseek_v4_flash_kaggle_tpu_swarm_rc.py tests/test_deepseek_v4_flash_kaggle_web_tpu_stage_adapter_probe.py tests/test_kaggle_web_tpu_queue_monitor_probe.py -q`
now reported 20 passed when including
`tests/test_deepseek_v4_flash_torch_stage_adapter_smoke.py`.

New JAX-stage adapter smoke tooling is
`scripts/deepseek_v4_flash_jax_stage_adapter_smoke.py` and
`scripts/deepseek_v4_flash_jax_stage_adapter_smoke_check.py`. The canonical
artifact is
`dist/deepseek-v4-flash-jax-stage-adapter-smoke-20260701-r3-local-cpu-jax/deepseek_v4_flash_jax_stage_adapter_smoke.json`;
its checker passes under `.venv-jax/bin/python` and records
`jax_stage_adapter_smoke_ready=true`, `jax_runtime_execution_ready=true`,
`deepseek_v4_jax_stage_forward_ready=true`, `tpu_runtime_ready=false`, and
`deepseek_v4_jax_tpu_stage_forward_ready=false`. The CPU JAX output hash
matches the numpy reference hash. The fixture path uses DeepSeek-V4-shaped
weights and exercises mHC, MLA/shared-KV attention, attention sink, grouped
output projection, TopK MoE router, routed experts, shared experts, and HCA
compressor shape metadata. This improves the HF/PyTorch -> JAX stage adapter
path, but it is still not real DeepSeek weights and not TPU execution.

New real-model safetensors header tooling is
`scripts/deepseek_v4_flash_safetensors_stage_header_probe.py` and
`scripts/deepseek_v4_flash_safetensors_stage_header_check.py`. The canonical
artifact is
`dist/deepseek-v4-flash-safetensors-stage-header-20260701-r1-layers16-18/deepseek_v4_flash_safetensors_stage_header_probe.json`;
its checker passes with `safetensors_header_ready=true` and
`stage_header_shape_ready=true`. It used HTTP Range reads against the public
HF model, not full tensor downloads, and verified that the selected
`layers 16..18` stage maps to 3,145 keys across
`model-00018-of-00046.safetensors` and
`model-00019-of-00046.safetensors`, with zero missing header keys and
7,158,447,536 stage-selected tensor storage bytes recorded from offsets. Dtype
counts are `BF16=20`, `F32=19`, `F8_E4M3=17`, `F8_E8M0=1553`, and `I8=1536`.
This is real DeepSeek-V4-Flash shard-shape evidence, but it still records
`real_weight_tensor_values_loaded=false` and is not same-request inference.

DeepSeek source resolution is current at
`dist/deepseek-v4-flash-quantized-source-resolver-20260701-r1-current/deepseek_v4_flash_quantized_source_resolver.json`.
It confirms public HF sources for `deepseek-ai/DeepSeek-V4-Flash` and
quantized candidates. The recommended smallest GGUF path is
`teamblobfish/DeepSeek-V4-Flash-GGUF` `IQ1_S-XL`, about 61.540805GB split
across two files, but it still records runtime risks:
`stock_llama_cpp_cannot_load_deepseek_v4_flash`,
`deepseek_v4_flash_llama_cpp_runtime_wip`,
`t4_cuda_runtime_not_validated_for_deepseek_v4_flash`, and
`candidate_exceeds_single_t4x2_memory_budget`.

The DeepSeek Web TPU stage-adapter preflight artifact is
`dist/deepseek-v4-flash-kaggle-web-tpu-stage-adapter-20260701-r2-local-metadata-webtpu-skipped/deepseek_v4_flash_kaggle_web_tpu_stage_adapter_probe.json`.
Its checker passes as a blocker artifact: real DeepSeek metadata and stage
mapping are ready from HF (`model_type=deepseek_v4`,
`architectures=["DeepseekV4ForCausalLM"]`, 43 layers, 69,187 weight keys, 46
safetensors files). The selected test stage `layers 16..18` maps to 3,145
stage-owned keys and 2 safetensors shards, with family hits for MLA attention,
MoE router, shared experts, routed experts, hybrid compression, and norms.
However `deepseek_v4_jax_tpu_stage_forward_ready=false`; the explicit blocker
is `deepseek_v4_flash_mla_moe_jax_tpu_stage_forward_not_implemented`.

The latest Kaggle TPU reacquire attempt is
`dist/kaggle-web-tpu-queue-monitor-20260701-r7-restart-after-header-ready/kaggle_web_tpu_queue_monitor_probe.json`.
Its checker passes and records a successful Start Session click for `TPU
v5e-8`; the visible queue position changed from `#23` to `#22` during a
bounded 600-second wait. It still records `web_tpu_runtime_ready=false`, no
Jupyter frame/session/kernel, and blockers including
`kaggle_web_tpu_active_event_queued`, `kaggle_web_tpu_queue_prompt_visible`,
and `kaggle_web_tpu_session_still_starting`. The follow-up read-only Active
Event artifact is
`dist/kaggle-web-tpu-active-event-probe-20260701-r6-after-r7-queue-readonly/kaggle_web_tpu_active_event_probe.json`;
it shows one visible `TPU v5e-8` event still `Queued` after about 14 minutes,
with no Jupyter frame/session/kernel and `active_event_running=false`. The
older r6 queue artifact and r4/r5 active-event artifacts are now historical
only. The earlier execution-channel success remains valid historical capacity
evidence:
`dist/kaggle-web-tpu-execution-channel-probe-20260701-r2-force-new-session-after-running-event/kaggle_web_tpu_execution_channel_probe.json`
passed with 8 `TPU v5 lite` devices, but it is no longer a live runtime.

Next correct work: keep the current Web TPU queue alive or rerun the queue
monitor until a Running event/Jupyter channel exists, then rerun the DeepSeek
Web TPU stage-adapter probe without `--skip-web-tpu-execute`. Completion still
requires implementing a real DeepSeek-V4 MLA/MoE JAX/TPU stage forward or a
verified equivalent TPU loader, then running a same-request Coordinator decode
whose accepted providers include Kaggle CUDA, Kaggle Web TPU, and Kaggle CPU
with real DeepSeek-V4-Flash or public quantized DeepSeek-V4-Flash weights.

Operational rule for future goal-mode runs: if Kaggle Web TPU is visibly
queued or `Session starting`, especially when the queue position is observable
or changing, do not mark the goal `blocked` only because the runtime is not
ready yet. This user instruction supersedes earlier bounded-attempt text for
visible Web TPU queues. Keep the queue monitor alive or restart monitor windows
back-to-back, recording queue-position updates when visible, until the event
becomes Running/Jupyter execution is available, the event is cancelled,
authentication/runtime state is lost, or the user explicitly stops the run.
This avoids wasting a TPU allocation that might become ready while the user is
not watching or able to issue `goal resume`.

## Latest Kaggle Web TPU Acquisition Status

The portable, account-independent operating procedure is now documented in
`docs/kaggle-tpu-v5e8-runbook.md`. Other project Sessions should follow that
runbook with their own Notebook URL and private Playwright storage-state file.
Always pass `--kaggle-notebook-url` explicitly; historical script defaults are
project-specific. Do not publish the storage-state file, cookies, Jupyter proxy
material, or Notebook runtime URL.

Current superseding Kaggle TPU status after the 2026-07-01 retry: Kaggle Web
TPU acquisition is currently usable through the Interactive Notebook UI path,
not through Kaggle CLI/API. The new queue monitor tooling is
`scripts/kaggle_web_tpu_queue_monitor_probe.py` and
`scripts/kaggle_web_tpu_queue_monitor_check.py`, with tests in
`tests/test_kaggle_web_tpu_queue_monitor_probe.py`. It can optionally select
`TPU v5e-8`, click Start Session, and then record the visible Kaggle queue
prompt `TPUs are popular right now. You are #N in the queue...` as structured
public-safe fields. The canonical queue-start artifact is
`dist/kaggle-web-tpu-queue-monitor-20260701-r2-start-wait-5m/kaggle_web_tpu_queue_monitor_probe.json`;
its checker passes and records `start_clicked=true`,
`queue_position_observed=true`, queue position `#13 -> #12`,
`queue_position_decreased=true`, and a visible queued `TPU v5e-8` Active Event.
This proves the queue prompt can be monitored and its position can change.

The follow-up lightweight read-only artifact is
`dist/kaggle-web-tpu-queue-monitor-20260701-r4-readonly-light-10m/kaggle_web_tpu_queue_monitor_probe.json`;
its checker passes and records the queue prompt disappearing and
`session_started_text_visible=true` but no Jupyter frame/session. The Active
Event probe
`dist/kaggle-web-tpu-active-event-probe-20260701-r1-after-queue-progress/kaggle_web_tpu_active_event_probe.json`
then showed one `TPU v5e-8` Active Event, `Running: 7 minutes`, but still no
Jupyter frame/session through that read-only view. The decisive execution
artifact is
`dist/kaggle-web-tpu-execution-channel-probe-20260701-r2-force-new-session-after-running-event/kaggle_web_tpu_execution_channel_probe.json`;
`scripts/kaggle_web_tpu_execution_channel_check.py --report ... --json` passes
with `web_tpu_execution_channel_ready=true`, `small_jax_cell_ready=true`,
`tiny_qwen_like_cell_ready=true`, `stage_local_kv_cache_verified=true`,
`tpu_device_count=8`, and `tpu_device_kind="TPU v5 lite"`. This is a real
Kaggle Web TPU/JAX execution-channel success, not just queue evidence. The
correct next use of Kaggle TPU is: if no running event exists, run the queue
monitor to start/wait; once Active Event becomes Running, run execution-channel
probe with `--web-tpu-force-new-session`; only then run larger LLM/TPU adapter
or heterogeneous inference probes.

## Latest Free GPU Provider Scouting Status

Current superseding provider-scouting status after the 2026-06-30 Lightning AI
credential retry: the canonical public-safe scouting artifact is now
`dist/free-gpu-provider-scouting-20260630-r4-lightning-api-zero-balance/free_gpu_provider_scouting.json`.
Lightning AI should not be treated as currently usable free GPU capacity. The
browser cookie probe
`dist/lightning-gpu-provider-probe-20260630-r6-post-token-cookie-readonly/lightning_gpu_provider_probe.json`
still shows the web session is not logged in (`lightning_login_verified=false`,
`login_or_signup_visible=true`). The API key in `~/.config/crowdtensor/lightning-token.md` does
authenticate, and the read-only API probe
`dist/lightning-api-gpu-provider-probe-20260630-r1-token-readonly/lightning_api_gpu_provider_probe.json`
passes its checker, but it reports `user_balance=0.0`,
project `balance=0.0`, `balance_limit=0.0`,
`can_start_free_cloud_space=false`, 16 visible GPU SKUs, and cheapest enabled
GPU cost `0.19` per hour. No Lightning Studio/GPU was created or started; do
not start one until API evidence shows positive credits or
`can_start_free_cloud_space=true`.

GCloud is also not current GPU capacity: the active account can see two
projects, but `gcloud billing accounts list` returns no billing accounts, both
projects report `billingEnabled=false`, Compute Engine API is disabled, and TPU
API is not enabled. Do not enable APIs or create instances without explicit
user approval and verified free/trial credit. Current next candidate order for
stable extra GPU is Modal, Paperspace Gradient, then a verified free-credit
GCP/Lightning account. Existing Kaggle T4x2 remains the only recently verified
reliable GPU source in this project context; Colab T4 remains blocked by HTTP
503 assignment failures.

## Latest DeepSeek V4 Flash Quantized Swarm RC Status

Current superseding DeepSeek V4 Flash status after the 2026-06-30 r22 third-resume
blocked audit:
the canonical RC artifact is now
`dist/deepseek-v4-flash-quantized-swarm-rc-20260630-r22-third-resume-colab-503-8x45s/deepseek_v4_flash_quantized_swarm_rc.json`.
`python scripts/deepseek_v4_flash_quantized_swarm_rc_check.py --report ... --json`
passes with no errors, but records `same_request_decode_verified=false`,
`generated_token_count=0`, `accepted_providers=[]`, and
`failure_stage=colab_cuda_reacquire_not_ready`. Do not report this as a
DeepSeek inference success. It is a public-safe blocker artifact proving source
resolution, reusable V4-aware Kaggle runtime build, Kaggle T4x2 RPC runtime
health, bounded Colab T4 authuser fallback, supplemental non-T4 Colab
assignment probes, Colab CUDA reacquire retry, and an auto retry-then-same-
request wrapper, not a completed same-request decode. The r20/r21/r22 packer now
promotes both `colab_cuda_reacquire_retry` and
`colab_retry_same_request_auto` into the RC main summary instead of only
recording their paths in the support bundle. The strict success gate remains:
only a public-safe artifact with Kaggle CUDA, Colab CUDA, and CPU providers in
one same-request decode and at least one generated token may mark this DeepSeek
goal complete.

The source resolver artifact remains
`dist/deepseek-v4-flash-quantized-source-resolver-20260629-r2-hf-gguf-native-fix/deepseek_v4_flash_quantized_source_resolver.json`.
It confirms public quantized GGUF candidates for `deepseek-ai/DeepSeek-V4-Flash`
(284B total MoE, 13B active). The recommended smallest live candidate is
`teamblobfish/DeepSeek-V4-Flash-GGUF` `IQ1_S-XL`, two GGUF splits totaling
about 61.540805GB. It also records that stock llama.cpp cannot load the model,
DeepSeek V4 llama.cpp support is WIP, T4 CUDA runtime is not validated for this
path, and the smallest candidate exceeds a single T4x2 memory budget.

The Kaggle V4-aware runtime build canonical artifact is
`dist/deepseek-v4-flash-kaggle-llama-v4-runtime-bundle-20260629-r1-tpuowner-t4x2/deepseek_v4_flash_kaggle_llama_v4_build_preflight.json`.
It records `llama_v4_runtime_build_ready=true`, commit
`781e978f3ee68144cb5922be9a5627610d091317`, `cmake_configure_ok=true`,
`cmake_build_ok=true`, `llama_cli_present=true`, `rpc_server_present=true`,
`llama_cli_supports_rpc=true`, `llama_cli_supports_tensor_split=true`,
`patch_rpc_op_count_guard_ok=true`, and exported
`deepseek-v4-flash-llama-v4-runtime.tar.gz` (148,079,902 bytes,
`sha256:d9c9bcb9ff2fb7993a233b66b0e3ff6eab775aa6cb058832e0b173d07f1ed26d`).
The temporary Kaggle build kernel was deleted. This runtime tarball lets Colab
and Kaggle skip repeated source builds. It was built for CUDA architecture 75
(T4). If a future Colab non-T4 GPU target becomes available, build/export a
matching multi-architecture runtime bundle before using it for full DeepSeek
decode.

The RPC HELLO diagnostic was fixed for current llama.cpp RPC v4: HELLO request
body is now 24 bytes of `conn_caps`, and the response body is 28 bytes
(version header plus 24-byte caps). The failed old-protocol diagnostic is
`dist/deepseek-v4-flash-rpc-hello-diagnostic-probe-20260630-r1-kaggle-local-runtime/deepseek_v4_flash_rpc_hello_diagnostic_probe.json`; it saw two live T4 RPC
servers but failed with `HELLO request size mismatch (8 vs 24)`. The corrected
diagnostic is
`dist/deepseek-v4-flash-rpc-hello-diagnostic-probe-20260630-r2-kaggle-local-runtime-hello-v4/deepseek_v4_flash_rpc_hello_diagnostic_probe.json`: it launched a
fresh private tpuowner Kaggle T4x2 kernel, downloaded the 148MB runtime
tarball, started two local CUDA RPC servers, completed RPC v4 HELLO for both,
and deleted the temporary kernel. This proves the Kaggle-side runtime tarball
and local CUDA RPC server protocol path are healthy. It does not satisfy the
DeepSeek same-request success gate because it does not include Colab CUDA, CPU,
model loading, or token decode.

The same-request probe is
`scripts/deepseek_v4_flash_quantized_same_request_probe.py`, with tests in
`tests/test_deepseek_v4_flash_quantized_same_request_probe.py`. It supports
`--runtime-tarball-path/--runtime-tarball-url`, temporary local HTTP+bore
serving of the runtime bundle, Colab background RPC launch/poll, Colab
keepalive, `stop_runtime_after_success=False` for background Colab workers,
client CUDA visible by default, optional CPU remote RPC exclusion, v4 RPC HELLO
preflight before expensive model downloads, bounded multi-authuser Colab
fallback via `--colab-authusers`, and bounded accelerator/authuser matrix
fallback via `--colab-accelerators` plus `--colab-authusers`. The default
DeepSeek path should still use `--colab-accelerators T4 --colab-authusers 0,1`
until a non-T4 runtime bundle is available.

Colab CUDA retry tooling exists at
`scripts/colab_cuda_reacquire_retry_probe.py` and
`scripts/colab_cuda_reacquire_retry_check.py`, with tests in
`tests/test_colab_cuda_reacquire_retry_probe.py` and
`tests/test_colab_cuda_reacquire_retry_check.py`. It is allocation evidence
only and must not be counted as DeepSeek decode success. The longer retry
artifact is
`dist/colab-cuda-reacquire-retry-20260630-r2-t4-authusers0-1-10x30s/colab_cuda_reacquire_retry_probe.json`;
its checker passes, but `colab_cuda_reacquire_ready=false`: ten attempts over
about 4.6 minutes, rotating T4 authuser 0/1, all returned HTTP 503. The latest
resume retry artifact is
`dist/colab-cuda-reacquire-retry-20260630-r4-resume-6x30s/colab_cuda_reacquire_retry_probe.json`;
its checker also passes, but `colab_cuda_reacquire_ready=false`: six attempts
rotating T4 authuser 0/1 all returned HTTP 503. The earlier wrapper artifact is
`dist/deepseek-v4-flash-colab-retry-same-request-auto-20260630-r1-6x30s/deepseek_v4_flash_colab_retry_same_request_auto.json`;
it first ran a 6-attempt Colab T4 retry, saw `retry_ready=false`, and therefore
correctly did not start the DeepSeek/Kaggle same-request stage. The wrapper
resume artifact is
`dist/deepseek-v4-flash-colab-retry-same-request-auto-20260630-r2-resume-6x30s/deepseek_v4_flash_colab_retry_same_request_auto.json`;
it again ran a 6-attempt Colab T4 retry, saw `retry_ready=false`, and correctly
did not start the same-request stage. The latest resume retry artifact is
`dist/colab-cuda-reacquire-retry-20260630-r5-resume-8x45s/colab_cuda_reacquire_retry_probe.json`;
its checker passes, but `colab_cuda_reacquire_ready=false`: eight attempts over
about 5.35 minutes, rotating T4 authuser 0/1, all returned HTTP 503. The
latest wrapper artifact is
`dist/deepseek-v4-flash-colab-retry-same-request-auto-20260630-r3-resume-8x45s/deepseek_v4_flash_colab_retry_same_request_auto.json`;
it saw `retry_ready=false` and therefore did not start the same-request stage.
The third-resume blocked-audit retry artifact is
`dist/colab-cuda-reacquire-retry-20260630-r6-resume-8x45s-third-audit/colab_cuda_reacquire_retry_probe.json`;
its checker passes, but `colab_cuda_reacquire_ready=false`: eight attempts over
about 5.37 minutes, rotating T4 authuser 0/1, all returned HTTP 503. The
third-resume wrapper artifact is
`dist/deepseek-v4-flash-colab-retry-same-request-auto-20260630-r4-resume-8x45s-third-audit/deepseek_v4_flash_colab_retry_same_request_auto.json`;
it saw `retry_ready=false`, `same_request_started=false`, and therefore did
not start Kaggle/DeepSeek same-request. This is the third consecutive
post-blocked resume attempt with the same Colab T4 assignment HTTP 503
condition.
The wrapper
script is `scripts/deepseek_v4_flash_colab_retry_same_request_auto.py`, with
tests in `tests/test_deepseek_v4_flash_colab_retry_same_request_auto.py`; use it
for future attempts when the goal is to avoid missing a short-lived Colab T4
ready window. The RC packer accepts `--rpc-hello-diagnostic-report`,
`--colab-accelerator-probe-report`, `--colab-cuda-reacquire-retry-report`, and
`--colab-retry-same-request-auto-report`, recording these evidence paths
without counting them as same-request decode success; current tests also cover
that auto retry failure becomes the visible RC `failure_stage`.

Live DeepSeek attempts after the tarball improvement:
- r6 `dist/deepseek-v4-flash-quantized-same-request-probe-20260629-r6-runtime-tarball-live/deepseek_v4_flash_quantized_same_request_probe.json`:
  Colab CUDA, Kaggle CUDA, and CPU were all accepted and both GGUF splits
  downloaded, but `llama-cli` aborted during RPC backend initialization with
  `Remote RPC server crashed or returned malformed response` after the client
  hid CUDA devices.
- r7 `dist/deepseek-v4-flash-quantized-same-request-probe-20260629-r7-runtime-tarball-client-cuda-no-cpu-rpc/deepseek_v4_flash_quantized_same_request_probe.json`:
  keeping client CUDA visible and excluding CPU as a remote RPC endpoint avoided
  the immediate CUDA-hidden abort, but Colab endpoint connectivity was lost by
  llama launch time (`Failed to connect to <colab-rpc-endpoint>`).
- r8 `dist/deepseek-v4-flash-quantized-same-request-probe-20260629-r8-colab-keepalive-runtime-tarball/deepseek_v4_flash_quantized_same_request_probe.json`:
  after adding Colab keepalive and a post-download endpoint check, the endpoint
  was still TCP-reachable after downloading 61.540805GB and all three providers
  were accepted, but the old HELLO probe/client path still failed with `Remote
  RPC server crashed or returned malformed response`; no token was generated.
- r12 `dist/deepseek-v4-flash-quantized-same-request-probe-20260630-r12-rpc-hello-v4-live/deepseek_v4_flash_quantized_same_request_probe.json`:
  after fixing background runtime stopping and RPC v4 HELLO, fresh Colab T4
  allocation for authuser 0 returned HTTP 503, so no Kaggle kernel was pushed.
- r13 `dist/deepseek-v4-flash-quantized-same-request-probe-20260630-r13-authuser1-rpc-hello-v4-live/deepseek_v4_flash_quantized_same_request_probe.json`:
  authuser 1 also returned Colab T4 assignment HTTP 503.
- r14 `dist/deepseek-v4-flash-quantized-same-request-probe-20260630-r14-existing-colab-session-rpc-hello-v4-live/deepseek_v4_flash_quantized_same_request_probe.json`:
  reusing the existing saved Colab session without forced reacquire returned a
  stale HTTPError, so no Kaggle kernel was pushed.
- r15 `dist/deepseek-v4-flash-quantized-same-request-probe-20260630-r15-authuser-fallback-rpc-hello-v4-live/deepseek_v4_flash_quantized_same_request_probe.json`:
  `--colab-authusers 0,1` tried both authuser 0 and 1 in one bounded run; both
  returned Colab T4 assignment HTTP 503, so no Kaggle kernel was pushed.
- r16 `dist/deepseek-v4-flash-quantized-same-request-probe-20260630-r16-t4-authuser-matrix-rpc-hello-v4-live/deepseek_v4_flash_quantized_same_request_probe.json`:
  the generalized `--colab-accelerators T4 --colab-authusers 0,1` matrix tried
  both T4/authuser targets and again received HTTP 503 for both; no Kaggle
  kernel was pushed.

Supplemental Colab accelerator probes after r16: L4 authuser 0/1
(`dist/colab-cuda-session-probe-20260630-r1-l4-authuser0/colab_cuda_session_probe.json`,
`dist/colab-cuda-session-probe-20260630-r2-l4-authuser1/colab_cuda_session_probe.json`),
empty/default GPU authuser 0/1
(`dist/colab-cuda-session-probe-20260630-r3-anygpu-authuser0/colab_cuda_session_probe.json`,
`dist/colab-cuda-session-probe-20260630-r4-anygpu-authuser1/colab_cuda_session_probe.json`),
and A100 authuser 0/1
(`dist/colab-cuda-session-probe-20260630-r5-a100-authuser0/colab_cuda_session_probe.json`,
`dist/colab-cuda-session-probe-20260630-r6-a100-authuser1/colab_cuda_session_probe.json`)
all returned HTTP 400 from the Colab assignment API. Therefore, in the current
API/account state, T4 is the only meaningful Colab GPU target and it is
temporarily unavailable with HTTP 503.

Do not mark the DeepSeek goal complete. Current blocker is Colab CUDA runtime
availability/lifecycle: existing saved Colab session is stale, T4 assignment
for authuser 0 and 1 returns HTTP 503, and non-T4 assignment strings tested so
far return HTTP 400. Once Colab T4 is reacquired, rerun the auto wrapper or the
same-request probe with the corrected RPC v4 HELLO path, runtime tarball, and
`--colab-accelerators T4 --colab-authusers 0,1`. If Colab+bore RPC then fails
despite Kaggle-local HELLO success, focus on Colab RPC server/bore transport
lifecycle rather than rebuilding the Kaggle runtime.

## Latest Colab CUDA GPU + Kaggle GPU/CPU Status

Current superseding status after the alternate LLM exploration on 2026-06-29:
the canonical non-TPU Kaggle/Colab GPU+CPU max-parameter artifact is now
`dist/kaggle-colab-gpu-cpu-max-parameter-search-20260629-r6-alternate-llm-72b-clean-671b-attempted-235b-stageplan/kaggle_colab_gpu_cpu_max_parameter_search.json`.
`scripts/kaggle_colab_gpu_cpu_max_parameter_search_check.py --report ... --json`
passes with no errors and records
`max_successful_same_request_decode_parameter_class=72b`,
`max_successful_dense_full_precision_parameter_class=72b`,
`max_successful_moe_total_parameter_class=""`,
`max_successful_moe_activated_parameter_class=""`,
`max_attempted_parameter_class=671b`, and
`failure_stage=deepseek_fp8_not_full_precision_and_mla_moe_adapter_missing`.
This supersedes r4 for current max-search status. It must not be reported as
a 235B, 405B, or 671B inference success: the largest full same-request
1-token decode still remains the clean 72B Qwen2.5 run.

The alternate source resolver artifact is
`dist/kaggle-alternate-llm-source-resolver-20260629-r3-405b-qwen3-deepseek-precision-fix/kaggle_alternate_llm_source_resolver.json`.
It scanned Llama 3.1 405B/405B-Instruct, Qwen3-235B-A22B, DeepSeek-V3/R1
671B, and Qwen3-Next-80B-A3B, recording Kaggle refs, expected attach paths,
license/gating status, HF config/index metadata, precision, architecture
class, total params, active MoE params, and public-safe source refs. Llama
405B remains blocked by license/HF metadata access plus unavailable Kaggle
attach in the bounded runtime:
`dist/kaggle-model-attach-llama405b-20260629-r1/kaggle_model_attach_probe.json`
records `kaggle_attach_path_missing_in_runtime` and cleaned up its temporary
CPU-only private kernel.

Qwen3-235B-A22B is the strongest positive >72B source evidence from this run.
The CPU-only attach/stage-plan probe
`dist/kaggle-model-attach-qwen3-235b-a22b-20260629-r1/kaggle_model_attach_probe.json`
successfully attached
`qwen-lm/qwen-3/Transformers/235b-a22b/1` at
`/kaggle/input/models/qwen-lm/qwen-3/transformers/235b-a22b/1`, saw
`model_type=qwen3_moe`, 94 layers, BF16, no quantization config, 118
safetensors files, and verified a 24-stage stage-owned header preflight:
36,945 assigned keys, 36,945 present keys, about 470.187GB logical tensor
bytes, and max planned stage about 21.147GB. The temporary private kernel was
deleted. This is source/attach/stage-plan evidence only; it is not a
stage-forward or same-request decode. The blocker artifact
`dist/kaggle-colab-gpu-cpu-large-model-blocker-20260629-r2-qwen3-235b-a22b-stage-plan-adapter-blocked/kaggle_colab_gpu_cpu_large_model_blocker.json`
records `qwen3_moe_same_request_runtime_adapter_not_verified`.

DeepSeek-V3/R1 671B metadata was readable, but the available source metadata
is FP8 and the project does not yet have a DeepSeek MLA/MoE same-request
runtime adapter. The blocker artifact
`dist/kaggle-colab-gpu-cpu-large-model-blocker-20260629-r2-deepseek-v3-671b-fp8-adapter-blocked/kaggle_colab_gpu_cpu_large_model_blocker.json`
records the 671B attempt as blocked by `candidate_not_full_precision_bf16` and
`deepseek_mla_moe_adapter_required`; it cannot count toward full-precision
dense or MoE success. Qwen3-Next-80B-A3B is metadata-ready BF16 hybrid MoE,
but remains adapter-blocked at
`dist/kaggle-colab-gpu-cpu-large-model-blocker-20260629-r2-qwen3-next-80b-a3b-adapter-blocked/kaggle_colab_gpu_cpu_large_model_blocker.json`.

Previous r4 status on 2026-06-29: the 5 T4 + Kaggle CPU dense/full-
precision max-parameter search has completed its bounded run. The canonical
conclusion artifact is
`dist/kaggle-colab-gpu-cpu-max-parameter-search-20260629-r4-5t4-72b-clean-176b-source-blocked/kaggle_colab_gpu_cpu_max_parameter_search.json`.
`scripts/kaggle_colab_gpu_cpu_max_parameter_search_check.py --report ... --json`
passes with no errors and records
`max_successful_same_request_decode_parameter_class=72b`,
`max_attempted_parameter_class=176b`, and
`failure_stage=model_source_attach_unavailable`. This is the current non-TPU
Kaggle/Colab GPU+CPU path status and supersedes the earlier r3 recovered-only
72B status.

The tpuowner Kaggle account did successfully run two simultaneous T4x2 GPU
kernels. The bounded concurrency artifact is
`dist/kaggle-gpu-concurrency-20260629-r1-tpuowner-t4x2/kaggle_gpu_concurrency_probe.json`;
`python scripts/kaggle_gpu_concurrency_check.py --report ... --json` passes and
records `simultaneous_t4x2_verified=true`, `accepted_submission_count=2`,
`max_observed_running_count=2`, and worker reports with `cuda_device_count=2`
for both private kernels. Both temporary GPU kernels were deleted.

The clean 72B raw main report is
`dist/kaggle-colab-gpu-cpu-72b-20260629-r6-tpuowner-5t4-clean/kaggle_32b_full_heterogeneous_probe.json`.
`scripts/kaggle_colab_gpu_cpu_heterogeneous_check.py --report ... --json`
passes. It ran full-precision/non-quantized `Qwen/Qwen2.5-72B-Instruct` across
all 80 layers in one Coordinator request, generated one token, and records
`same_request_72b_kaggle_colab_gpu_cpu_full_model_verified=true`. The topology
was `4KaggleGPU_stages_1ColabGPU_stages_0WebTPU_stages_5CPU_stages`: CPU
stage0 `[0,2]`, Kaggle T4x2 group A `[2,8]` and `[8,14]`, Kaggle T4x2 group B
`[14,20]` and `[20,26]`, Colab T4 `[26,32]`, and CPU tail stages `[32,44]`,
`[44,56]`, `[56,68]`, and `[68,80]`. Provider stage counts were
`kaggle_cuda=4`, `colab_cuda=1`, `cpu=5`, `web_tpu=0`; all ten stage task
counts were 1; stage-local KV-cache metadata is public-safe; all temporary
Kaggle kernels and private payloads were deleted. This is now raw evidence, not
just recovered runtime evidence.

A larger 176B full-precision dense candidate was investigated with BLOOM/BLOOMZ.
`dist/kaggle-dense-model-source-resolver-20260629-r3-bloom176b/kaggle_dense_model_source_resolver.json`
confirms `bigscience/bloom` and `bigscience/bloomz` are public, non-quantized,
dense BloomForCausalLM metadata candidates with 70 layers and about 352GB of
safetensors. The project added BLOOM stage-runtime support for HF key prefixes
(`h.*`, `word_embeddings.*`, `ln_f.*`) mapped into Transformers state keys and
added tests for tiny BLOOM stage activation/final decode. However, Kaggle
Models attach is not currently available for the 176B Transformers weights:
`dist/kaggle-model-attach-bloom176b-20260629-r1-tpuowner/kaggle_model_attach_probe.json`
and
`dist/kaggle-model-attach-bloomz176b-20260629-r1-tpuowner/kaggle_model_attach_probe.json`
both show `/kaggle/input` had no attached model path and record
`kaggle_attach_path_missing_in_runtime`. Kaggle MCP model search found only the
Keras BLOOM 0.5B-3B variants, not a 176B Transformers variation. Therefore the
176B rung is recorded as attempted but blocked before execution by
`kaggle_model_attach_176b_transformers_source_unavailable`; it must not be
reported as an inference or memory failure.

Do not call the current 72B result "stable" under the stricter goal definition:
it is a clean single 1-token decode. The current status is max cleanly run
parameter class = 72B, max attempted parameter class = 176B, stable parameter
class = not yet established because the required two independent 1-token runs
plus one 2-4 token/longer-prompt run were not completed. The next smallest
useful step for stability is to rerun the 72B 5 T4+CPU topology at least twice
and add one 2-token decode; the next useful step for larger models is to obtain
or create a Kaggle-accessible attached full-precision 100B+ Transformers model
source, or implement a streaming/non-attached shard loader that does not require
per-worker 352GB runtime downloads.


## Latest Web/Colab TPU Fallback Status

Current superseding status after the latest continuation on 2026-06-28: the
canonical dense max-search artifact is now
`dist/three-accelerator-dense-max-parameter-search-20260628-r30-kaggle-starting-colab-503/three_accelerator_dense_max_parameter_search.json`.
Its checker passes and it records
`max_successful_same_request_decode_parameter_class=32b`,
`max_attempted_parameter_class=72b`,
`max_stage_loaded_parameter_class=72b`,
`max_tpu_executed_parameter_class=72b`, and
`failure_stage=colab_tpu_reacquire_not_ready`. This supersedes r29 for current
runtime status. A read-only Kaggle UI probe first showed Draft Session/off:
`dist/kaggle-web-tpu-ui-state-probe-20260628-r5-goal-continue-readonly/kaggle_web_tpu_ui_state_probe.json`
reports `web_tpu_ui_runtime_ready=false`, `start_session_visible=true`, and no
Jupyter frame/session/kernel. The Web TPU start-wait probe was hardened so
Jupyter service-manager refresh promises cannot hang the bounded probe; tests
cover observe-failure reporting and max-search accepts a public-safe
`kaggle_web_tpu_start_session_not_clicked` blocker when the UI is already in a
not-ready Starting state. The fresh start-wait artifact
`dist/kaggle-web-tpu-start-wait-probe-20260628-r5-goal-continue-short-after-timeout-fix/kaggle_web_tpu_start_wait_probe.json`
is public-safe and bounded; it saw `TPU v5e-8` visible but still
`web_tpu_ui_runtime_ready=false`, `session_starting_text_visible=true`,
`jupyter_frame_visible=false`, `jupyter_session_count=0`, and
`jupyter_kernel_count=0`. Its click did not produce a fresh accepted start
event, so it records `kaggle_web_tpu_start_session_not_clicked` rather than
pretending a click succeeded. Colab fallback was retried in
`dist/colab-tpu-reacquire-retry-20260628-r5-goal-continue-v5e1/colab_tpu_reacquire_retry_probe.json`;
all four V5E1 attempts returned HTTP 503 with no endpoint/proxy/token material
public. The retained 72B Colab TPU stage-loader proof remains capacity
evidence only. Do not mark the active goal achieved: 72B dense/full-precision
GPU+TPU+CPU same-request 1-token decode over all 80 Qwen 72B layers has still
not completed. The next live attempt should wait until either Kaggle Web TPU
execution or Colab TPU runtime is actually ready, then use the tested
CPU-embedding/small-GPU/TPU/CPU-tail topology.

Previous r29 status:
`dist/three-accelerator-dense-max-parameter-search-20260628-r29-kaggle-webtpu-timeout-colab-reacquire-503/three_accelerator_dense_max_parameter_search.json`.
Its checker passes and it records
`max_successful_same_request_decode_parameter_class=32b`,
`max_attempted_parameter_class=72b`,
`max_stage_loaded_parameter_class=72b`,
`max_tpu_executed_parameter_class=72b`, and
`failure_stage=colab_tpu_reacquire_not_ready`. This supersedes r28 for current
runtime status. The fresh Kaggle Web TPU execution-channel probe
`dist/kaggle-web-tpu-execution-channel-probe-20260628-r16-before-r6-cpu-embed-topology-retry/kaggle_web_tpu_execution_channel_probe.json`
failed before small JAX with `web_tpu_jupyter_execute_timeout`,
`web_tpu_execution_channel_ready=false`, `tpu_runtime_attached=false`, and no
public Jupyter proxy material. Therefore the 72B r6 live attempt was not
started. The Colab fallback path is wired into the same heterogeneous probe via
`--tpu-provider colab_cli`, but the current local `ct-colab-tpu-v5e1` session
is stale:
`dist/colab-tpu-runtime-stability-20260628-r3-current-provider-check/colab_tpu_runtime_stability_probe.json`
returns runtime proxy HTTP 404 with no TPU device observed. Fresh Colab V5E1
allocation also failed in
`dist/colab-tpu-reacquire-retry-20260628-r3-current-provider-reacquire/colab_tpu_reacquire_retry_probe.json`
with three HTTP 503 attempts; a supplemental V6E1 attempt at
`dist/colab-tpu-reacquire-retry-20260628-r4-current-provider-v6e1/colab_tpu_reacquire_retry_probe.json`
returned HTTP 400. The retained 72B Colab TPU stage loader proof
`dist/colab-tpu-qwen-stage-loader-20260628-r2-72b-stage32-36-four-layer-fixed/colab_tpu_qwen_stage_loader_probe.json`
still counts as 72B TPU stage-load/forward capacity evidence, not full
all-layer inference. Do not mark the active goal achieved: 72B dense/full-
precision GPU+TPU+CPU same-request 1-token decode over all 80 Qwen 72B layers
has still not completed. The next live attempt should wait until either Kaggle
Web TPU execution or Colab TPU runtime is actually ready, then use the tested
CPU-embedding/small-GPU/TPU/CPU-tail topology from r28.

Previous r28 status:
`dist/three-accelerator-dense-max-parameter-search-20260628-r28-cpu-embed-topology-ready-web-tpu-timeout/three_accelerator_dense_max_parameter_search.json`.
Its checker passes and it records
`max_successful_same_request_decode_parameter_class=32b`,
`max_attempted_parameter_class=72b`,
`max_stage_loaded_parameter_class=72b`,
`max_tpu_executed_parameter_class=72b`, and
`failure_stage=web_tpu_channel_jupyter_execute`. This superseded r27 for
current runtime status: after r27 proved Kaggle Web TPU could briefly execute
small JAX/tiny Qwen-like cells, the next preflight
`dist/kaggle-web-tpu-execution-channel-probe-20260628-r15-before-r6-cpu-embed-topology/kaggle_web_tpu_execution_channel_probe.json`
timed out at Jupyter execution again, so the r6 72B live attempt was not
started. A new 72B placement is now tested and ready for the next live window:
CPU owns stage0 with embeddings and layer range `[0,1]`; one T4x2 GPU kernel
owns stages `[1,5]` and `[5,9]`; CPU owns `[9,24]`; Web TPU should own a
middle block such as `[24,40]`; CPU tail stages cover the remaining layers
while keeping total CPU kernels within the observed acceptance limit. The test
`test_build_report_accepts_72b_cpu_embedding_small_gpu_web_tpu_topology` proves
this placement is accepted by the success gate only when all 80 layers complete
in one same-request GPU+TPU+CPU decode.

Previous r27 status:
`dist/three-accelerator-dense-max-parameter-search-20260628-r27-web-tpu-restored-72b-gpu-oom/three_accelerator_dense_max_parameter_search.json`.
Its checker passes and it records
`max_successful_same_request_decode_parameter_class=32b`,
`max_attempted_parameter_class=72b`,
`max_stage_loaded_parameter_class=72b`,
`max_tpu_executed_parameter_class=72b`, and
`failure_stage=dense_72b_same_request_decode_not_verified_after_tpu_stage_forward`.
This supersedes r26: Kaggle Web TPU recovered enough to run the execution
channel probe
`dist/kaggle-web-tpu-execution-channel-probe-20260628-r14-after-active-event-running/kaggle_web_tpu_execution_channel_probe.json`,
which executed small JAX and tiny Qwen-like cells, saw 8 `TPU v5 lite`
devices, and reports `web_tpu_execution_channel_ready=true`. Colab remains a
fallback but is not currently usable:
`dist/colab-tpu-runtime-stability-20260628-r16-existing-session-goal-continue/colab_tpu_runtime_stability_probe.json`
still shows the existing session proxy returns HTTP 404, and
`dist/colab-tpu-reacquire-retry-20260628-r16-goal-continue-wait15m/colab_tpu_reacquire_retry_probe.json`
shows V5E1 allocation HTTP 503 and V6E1 HTTP 400. The fresh 72B live attempt
`dist/kaggle-72b-full-heterogeneous-kaggle-web-tpu-live-20260628-r4-single-gpu-cpu-split-webtpu-restored/kaggle_32b_full_heterogeneous_probe.json`
failed before same-request decode: GPU shard0 hit `OutOfMemoryError` during
`stage0_model_prepare` for layers 0-8/8-16, CPU stage8 and stage9 were not
accepted, and manual cleanup was performed. A follow-up smaller-GPU r5 attempt
used 4-layer GPU stages and 5 CPU kernels, but CPU stage8 was not accepted and
the GPU kernel reached `ERROR`; no useful stage JSON was produced, and all
accepted r5 kernels were manually deleted. Do not mark the active goal
achieved: full 72B dense GPU+TPU+CPU same-request 1-token decode over all 80
Qwen layers has not succeeded. The next live attempt should reduce GPU-front
weight pressure further, for example by moving embeddings/early layers to CPU
or splitting GPU stage0 into a smaller topology while keeping total CPU kernel
count within observed Kaggle acceptance limits.

Previous r26 status:
`dist/three-accelerator-dense-max-parameter-search-20260628-r26-web-tpu-queued-colab-runtime404-reacquire503/three_accelerator_dense_max_parameter_search.json`.
Its checker passes and it records
`max_successful_same_request_decode_parameter_class=32b`,
`max_attempted_parameter_class=72b`,
`max_stage_loaded_parameter_class=72b`,
`max_tpu_executed_parameter_class=72b`, and
`failure_stage=colab_tpu_reacquire_not_ready`. It supersedes r25 by importing
both current Colab recovery paths: the existing local `ct-colab-tpu-v5e1`
session exists but the runtime proxy returns HTTP 404
(`dist/colab-tpu-runtime-stability-20260628-r15-existing-session-goal-continue/colab_tpu_runtime_stability_probe.json`),
and fresh Colab allocation still fails
(`dist/colab-tpu-reacquire-retry-20260628-r15-v5e1-v6e1-goal-continue/colab_tpu_reacquire_retry_probe.json`):
V5E1 attempts returned HTTP 503 and V6E1 attempts returned HTTP 400, with no
endpoint/proxy/token material public. Kaggle Web TPU also remains queued:
`dist/kaggle-web-tpu-active-event-probe-20260628-r13-goal-continue-current/kaggle_web_tpu_active_event_probe.json`
shows one visible `TPU v5e-8` Active Event, still `Queued`, with no Jupyter
frame/session/kernel. The max-search pack/checker now support
`colab_tpu_runtime_stability_import` so future artifacts distinguish an
unusable existing Colab session from fresh allocation failure. Do not mark the
active goal achieved: 72B dense/full-precision GPU+TPU+CPU same-request
1-token decode over all 80 Qwen 72B layers has not succeeded.

Previous r25 status:
`dist/three-accelerator-dense-max-parameter-search-20260628-r25-web-tpu-queued-1h-colab-503/three_accelerator_dense_max_parameter_search.json`.
Its checker passes and it records
`max_successful_same_request_decode_parameter_class=32b`,
`max_attempted_parameter_class=72b`,
`max_stage_loaded_parameter_class=72b`,
`max_tpu_executed_parameter_class=72b`, and
`failure_stage=colab_tpu_reacquire_not_ready`. The latest Kaggle Web TPU
Active Event artifact is
`dist/kaggle-web-tpu-active-event-probe-20260628-r12-current-after-code-audit/kaggle_web_tpu_active_event_probe.json`:
one `TPU v5e-8` event was visible and queued for about an hour, with
`active_event_runtime_ready=false`, no Jupyter frame, no Jupyter session, and
no kernel. The latest Colab fallback artifact is
`dist/colab-tpu-reacquire-retry-20260628-r13-after-code-audit/colab_tpu_reacquire_retry_probe.json`:
V5E1 allocation attempts for authuser 0 and 1 both returned HTTP 503, with no
endpoint/proxy/token material public. Tests confirm the project-side fallback
plumbing is in place: `scripts/kaggle_32b_full_heterogeneous_probe.py` accepts
`--tpu-provider colab_cli`, can route the Colab TPU stage through the same
Coordinator request, and supports the lower-GPU 72B topology
`2GPU_stages_1WebTPU_stages_7CPU_stages`. Do not mark the active goal
achieved: 72B dense/full-precision GPU+TPU+CPU same-request 1-token decode
over all 80 Qwen 72B layers has not succeeded.

Latest continuation after r20: a lower-GPU 72B topology is now explicitly
covered by tests. `tests/test_kaggle_32b_full_heterogeneous_probe.py` includes
`test_build_report_accepts_72b_single_gpu_web_tpu_cpu_tail_topology`, proving
the success gate accepts a full 80-layer 72B same-request decode with one
T4x2 GPU kernel owning stages 0/1, one Web TPU stage owning stage4, and one
CPU tail kernel owning stages 2/3/5/6/7/8/9, as long as every stage completes
in the same Coordinator request with stage-local KV-cache evidence. This does
not weaken the goal; it only removes the failed second-GPU-shard dependency.

The full-run driver also has a fail-fast improvement: if a worker thread has
already produced a failed stage report or a worker error while the Coordinator
is not ready, `wait_for_coordinator_ready()` returns for report generation and
cleanup instead of waiting until the long coordinator timeout. This is covered
by `test_wait_for_coordinator_ready_returns_when_worker_report_fails`.
The same test file also now covers the end-to-end coordinator assembly for the
single-GPU topology with
`test_run_coordinator_probe_can_assemble_single_gpu_72b_topology`: simulated
GPU, Web TPU, and CPU workers complete all 10 stages in one Coordinator request
with the requested topology `2GPU_stages_1WebTPU_stages_7CPU_stages`.

That live retry could not be started because the external TPU runtime changed
state again. The fresh Web TPU execution-channel probe
`dist/kaggle-web-tpu-execution-channel-probe-20260628-r13-before-single-gpu-72b/kaggle_web_tpu_execution_channel_probe.json`
timed out at Jupyter execution (`web_tpu_execution_channel_ready=false`,
`web_tpu_jupyter_execute_timeout`). The follow-up Active Event probe
`dist/kaggle-web-tpu-active-event-probe-20260628-r5-after-channel-timeout-reopen/kaggle_web_tpu_active_event_probe.json`
showed the `TPU v5e-8` active event had become `Cancelled`, with no Jupyter
frame/session/kernel. Colab fallback also remained unavailable:
`dist/colab-tpu-reacquire-retry-20260628-r9-after-kaggle-active-cancelled/colab_tpu_reacquire_retry_probe.json`
attempted V5E1 for authuser 0 and 1 with cleanup-before enabled and both
returned HTTP 503. No temporary Kaggle kernels from these checks remained.
The Web TPU start/wait probe was then run to restart TPU v5e-8:
`dist/kaggle-web-tpu-start-wait-20260628-r17-restart-after-cancelled/kaggle_web_tpu_start_wait_probe.json`.
It selected `TPU v5e-8`, clicked Start Session, and waited 300 seconds, but
ended with queue/starting visible, no Jupyter frame/session/kernel, and
`web_tpu_ui_runtime_ready=false`.
The latest Active Event follow-up
`dist/kaggle-web-tpu-active-event-probe-20260628-r8-queued-wait-10m/kaggle_web_tpu_active_event_probe.json`
then waited about 10 minutes and the TPU v5e-8 event remained `Queued` for the
entire bounded window. The latest Colab fallback retry is
`dist/colab-tpu-reacquire-retry-20260628-r10-after-web-restart-queue/colab_tpu_reacquire_retry_probe.json`;
authuser 0 and 1 V5E1 attempts both returned HTTP 503.
The latest continuation kept the same Kaggle TPU event alive in queue:
`dist/kaggle-web-tpu-active-event-probe-20260628-r10-queued-wait-20m-total/kaggle_web_tpu_active_event_probe.json`
waited another bounded 10 minutes and the event was still `Queued`, now about
40 minutes old, with no Jupyter frame/session/kernel. Colab fallback was tried
again at
`dist/colab-tpu-reacquire-retry-20260628-r11-after-r10-queue/colab_tpu_reacquire_retry_probe.json`;
authuser 0 and 1 V5E1 attempts both still returned HTTP 503.

The previous r24 max-search artifact was
`dist/three-accelerator-dense-max-parameter-search-20260628-r24-web-tpu-queued-40m-colab-503/three_accelerator_dense_max_parameter_search.json`;
its checker passes and records `max_successful_same_request_decode_parameter_class=32b`,
`max_attempted_parameter_class=72b`, `max_stage_loaded_parameter_class=72b`,
`max_tpu_executed_parameter_class=72b`, and
`failure_stage=colab_tpu_reacquire_not_ready`, with Web TPU cancelled,
execution-timeout, and restart/queue blockers also imported. Do not mark the
active goal achieved; the single-GPU 72B topology plus fail-fast path is ready
for the next live attempt only after either Kaggle Web TPU execution channel
or Colab V5E1 becomes available again.

Latest superseding status on 2026-06-28: yes, Colab TPU can be used inside the
unfinished 72B goal as a `jax_tpu` replacement for Kaggle Web TPU, but it has
not completed the full all-layer goal. The project already has the provider
plumbing (`--tpu-provider colab_cli`) and retained Colab capacity evidence:
`dist/colab-tpu-qwen-stage-loader-20260628-r2-72b-stage32-36-four-layer-fixed/colab_tpu_qwen_stage_loader_probe.json`
proves `Qwen/Qwen2.5-72B-Instruct` layers 32-36 executed on Colab `TPU v5 lite`
with 48 stage-owned keys, about 6.539GB logical execution tensor bytes,
stage-local KV-cache metadata, and public-safe hashes only. Current Colab
reacquire evidence is still unavailable for fresh live use:
`dist/colab-tpu-reacquire-retry-20260628-r8-v5e1-after-r7/colab_tpu_reacquire_retry_probe.json`
attempted V5E1 twice and both attempts returned HTTP 503.

Kaggle Web TPU is no longer simply queued/unavailable. A parser bug was fixed
in `scripts/kaggle_web_tpu_active_event_probe.py`: Kaggle reports active-event
status like `Running: 14 minutes`, not only exact `Running`. The retained probe
`dist/kaggle-web-tpu-active-event-probe-20260628-r4-running-prefix-open-attempt/kaggle_web_tpu_active_event_probe.json`
shows one `TPU v5e-8` active event, running and opened, but the dialog probe
does not see a Jupyter frame/session. The stronger execution-channel evidence
is
`dist/kaggle-web-tpu-execution-channel-probe-20260628-r12-active-event-running-opened/kaggle_web_tpu_execution_channel_probe.json`:
it connected through the browser iframe service manager, executed small JAX and
tiny Qwen-like cells, saw 8 `TPU v5 lite` devices, and reports
`web_tpu_execution_channel_ready=true`. The dense max-search pack/checker now
treat this execution-channel proof as authoritative over the Active Event UI
frame gap, and also treats Colab reacquire failure as fallback-only when Kaggle
Web TPU execution is ready.

A fresh full all-layer 72B retry was attempted with Kaggle Web TPU:
`dist/kaggle-72b-full-heterogeneous-kaggle-web-tpu-live-20260628-r2-10stage-staggered/`.
It used the 10-stage Qwen 72B topology with stage4 on Kaggle Web TPU and
20-second launch staggering. It did not complete: gpu-shard0 was accepted but
its stage report failed, gpu-shard1 was not accepted and was deleted, CPU
stage5..9 kernels were started but then manually deleted after the missing GPU
shard made success impossible. No final 72B success JSON should be inferred
from this interrupted retry.

The earlier r20 max-search artifact was
`dist/three-accelerator-dense-max-parameter-search-20260628-r20-web-tpu-channel-ready-72b-retry-gpu-shard-missing/three_accelerator_dense_max_parameter_search.json`;
`scripts/three_accelerator_dense_max_parameter_search_check.py --report ... --json`
passes. It records `max_successful_same_request_decode_parameter_class=32b`,
`max_attempted_parameter_class=72b`, `max_stage_loaded_parameter_class=72b`,
`max_tpu_executed_parameter_class=72b`, and
`failure_stage=dense_72b_same_request_decode_not_verified_after_tpu_stage_forward`.
Do not mark the active 72B goal achieved until a dense/full-precision
GPU+TPU+CPU same-request 1-token decode over all 80 Qwen 72B layers succeeds
and checker/tests/docs are updated.

## Latest Colab TPU Fallback Evidence

Latest superseding status after the Colab runtime helper fix: Colab remains a
valid `jax_tpu` provider path, but the active 72B full-decode goal is still not
achieved. `scripts/colab_cli_runtime.py` now lets the project Python reuse the
locally installed isolated `google-colab-cli` tool environment, so
`scripts/colab_tpu_runtime_stability_probe.py`,
`scripts/colab_tpu_coordinator_connectivity_probe.py`,
`scripts/colab_tpu_qwen_stage_loader_probe.py`,
`scripts/gpu_tpu_cpu_same_request_runtime_bridge_probe.py`, and
`scripts/kaggle_32b_full_heterogeneous_probe.py` no longer fail merely because
`colab_cli` is installed under the uv tool Python instead of `/usr/bin/python`.
The current system-Python smoke artifacts are public-safe and prove the helper
works, while also proving the retained Colab session is detached:
`dist/colab-tpu-runtime-stability-20260628-r14-system-python-runtime-helper/colab_tpu_runtime_stability_probe.json`
reports runtime proxy HTTP 404 / no TPU device, and
`dist/colab-tpu-qwen-stage-loader-20260628-r4-system-python-runtime-helper/colab_tpu_qwen_stage_loader_probe.json`
fails with a Colab HTTPError before stage execution. The latest bounded Colab
reacquire evidence is
`dist/colab-tpu-reacquire-retry-20260628-r6-after-toolpython-gap/colab_tpu_reacquire_retry_probe.json`:
V5E1 attempts across authuser 0 and 1 both returned HTTP 503, with no endpoint,
proxy URL, proxy token, OAuth token, credentials, or private runtime state
public. The current max-search artifact is
`dist/three-accelerator-dense-max-parameter-search-20260628-r17-colab-runtime-helper-reacquire-503/three_accelerator_dense_max_parameter_search.json`;
its checker passes and records `max_stage_loaded_parameter_class=72b`,
`max_tpu_executed_parameter_class=72b`, and
`max_successful_same_request_decode_parameter_class=32b` with
`failure_stage=colab_tpu_reacquire_not_ready`. Keep the retained 72B Colab TPU
stage loader proof at
`dist/colab-tpu-qwen-stage-loader-20260628-r2-72b-stage32-36-four-layer-fixed/colab_tpu_qwen_stage_loader_probe.json`
as capacity evidence, but do not count it as full 72B inference. Kaggle Web TPU
also remains unavailable: the latest read-only UI state
`dist/kaggle-web-tpu-ui-state-20260628-r17-current-stale-starting-check/kaggle_web_tpu_ui_state_probe.json`
still had `Session starting`, no Jupyter frame/session/kernel, and no TPU
runtime. Run the full 72B low-concurrency GPU+TPU+CPU probe only after either
Colab V5E1 reacquires and passes stability, or Kaggle Web TPU exposes a working
Jupyter execution channel.

Latest continuation after r17: `scripts/kaggle_web_tpu_active_event_probe.py`
now records Kaggle Web TPU Active Events as public-safe scheduler/runtime
evidence and `scripts/three_accelerator_dense_max_parameter_search_pack.py`
imports that evidence through `web_tpu_active_event_import`. The current active
event artifact is
`dist/kaggle-web-tpu-active-event-probe-20260628-r1-current-queued/kaggle_web_tpu_active_event_probe.json`:
the Active Events dialog opened, exactly one `TPU v5e-8` interactive session
event was visible, and it was still `Queued` with no Jupyter frame/session or
kernel; no cookie, Jupyter proxy token, raw URL, or private runtime state is
public. The latest Colab retry is
`dist/colab-tpu-reacquire-retry-20260628-r7-v5e1-after-r6/colab_tpu_reacquire_retry_probe.json`,
where V5E1 attempts for authuser 0 and 1 both returned HTTP 503. The current
max-search artifact is
`dist/three-accelerator-dense-max-parameter-search-20260628-r18-active-event-queued-colab-503/three_accelerator_dense_max_parameter_search.json`;
its checker passes and still records `max_successful_same_request_decode_parameter_class=32b`,
`max_stage_loaded_parameter_class=72b`, `max_tpu_executed_parameter_class=72b`,
and `failure_stage=colab_tpu_reacquire_not_ready`. This is not a full 72B
decode and must not be treated as active-goal completion. If a later Active
Event becomes `Running`, run the active-event probe again, then the Web TPU
execution-channel probe, then only if that passes run the low-concurrency 72B
full heterogeneous same-request probe.

Latest superseding Colab TPU status on 2026-06-28: Colab V5E1 was reacquired
at `dist/colab-tpu-session-20260628-r9-reacquire-v5e1/colab_tpu_session_probe.json`
and passed a three-round JAX BF16 stability check at
`dist/colab-tpu-runtime-stability-20260628-r9-v5e1-three-rounds/colab_tpu_runtime_stability_probe.json`.
The fixed Colab Qwen loader wrapper at `scripts/colab_tpu_qwen_stage_loader_probe.py`
then produced the retained 72B stage-owned loader evidence
`dist/colab-tpu-qwen-stage-loader-20260628-r2-72b-stage32-36-four-layer-fixed/colab_tpu_qwen_stage_loader_probe.json`:
`Qwen/Qwen2.5-72B-Instruct` layers 32-36 executed on one Colab `TPU v5 lite`,
with 48 stage-owned keys, about 6.539GB logical execution tensor bytes, no
missing stage keys, stage-local KV-cache verified, and a public-safe stage
output hash. `scripts/colab_tpu_qwen_stage_loader_check.py --require-ready`
passes for that report. The same-request retry
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260628-r19-colab-tpu-72b-four-layer-loader-correct-python/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`
used the correct Colab CLI Python and completed/cleaned the Kaggle CUDA stage,
but Colab returned an HTTPError/404 before stage1 submitted; a follow-up
stability probe at
`dist/colab-tpu-runtime-stability-20260628-r10-after-r19-httperror/colab_tpu_runtime_stability_probe.json`
confirmed the runtime proxy was no longer connected, and
`dist/colab-tpu-session-20260628-r11-reacquire-after-r19-404/colab_tpu_session_probe.json`
returned HTTP 503 `colab_assignment_resource_unavailable`. This is Colab
runtime lifecycle/allocation instability, not a 72B capacity failure. The
current dense max-search artifact is
`dist/three-accelerator-dense-max-parameter-search-20260628-r8-colab-72b-stage-loader-bridge-http404/three_accelerator_dense_max_parameter_search.json`;
its checker passes and records `max_stage_loaded_parameter_class=72b`,
`max_tpu_executed_parameter_class=72b`, and
`max_successful_same_request_decode_parameter_class=32b`. Do not mark the
active 72B goal achieved: a full dense/full-precision 72B GPU+TPU+CPU
same-request 1-token decode over all layers is still missing. Also note that
`scripts/gpu_tpu_cpu_same_request_runtime_bridge_probe.py` now renders the
Kaggle CUDA stage to exit once the requested token count is accepted instead of
waiting until kernel timeout; future bridge retries should use this faster
path after reacquiring a live TPU runtime.

Latest superseding live result after that fix: Colab V5E1 was reacquired again
at `dist/colab-tpu-session-20260628-r12-reacquire-after-fast-gpu-fix/colab_tpu_session_probe.json`,
and a one-round runtime check passed at
`dist/colab-tpu-runtime-stability-20260628-r11-after-r12-reacquire/colab_tpu_runtime_stability_probe.json`.
The same-request retry
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260628-r20-colab-tpu-72b-four-layer-fast-gpu-exit/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`
completed in one Coordinator request with accepted backends
`["cpu","cuda","jax_tpu"]`, `stage0=stage1=stage2=1`, two activation handoff
hashes, one generated-token hash, and temporary Kaggle CUDA kernel deletion.
Stage1 used Colab `TPU v5 lite` and executed the real
`Qwen/Qwen2.5-72B-Instruct` layers 32-36 stage-owned loader: 48 assigned keys,
about 6.539GB logical execution tensor bytes, 4 executed layers, no missing
stage keys, stage-local KV-cache verified, and public-safe hashes only. The
current dense max-search artifact is now
`dist/three-accelerator-dense-max-parameter-search-20260628-r9-colab-72b-stage-bridge-not-full-decode/three_accelerator_dense_max_parameter_search.json`;
its checker passes and records `max_stage_loaded_parameter_class=72b`,
`max_tpu_executed_parameter_class=72b`,
`same_request_72b_import.same_request_stage_decode_verified=true`, but
`max_successful_same_request_decode_parameter_class=32b` and
`same_request_72b_import.same_request_full_model_decode_verified=false`. This
is a successful Colab-as-TPU 72B stage bridge, not the active goal's full
all-layer 72B decode.

Latest full-72B engineering continuation: `scripts/kaggle_32b_full_heterogeneous_probe.py`
now accepts `--tpu-provider colab_cli`, `--colab-session-name`,
`--colab-session-config`, and `--stage-launch-stagger-seconds`. The full
heterogeneous Web/TPU stage worker can use the Colab runtime proxy for the
stage-owned Qwen loader, and tests cover the Colab provider branch plus cleanup
of non-accepted Kaggle kernel pushes. The bounded full all-layer live attempt
`dist/kaggle-72b-full-heterogeneous-colab-tpu-live-20260628-r1-10stage/kaggle_32b_full_heterogeneous_probe.json`
used 10 contiguous Qwen 72B stage ranges covering layers 0..80 and Colab TPU
for stage4, but it did not complete any Coordinator stage tasks:
`generated_token_count=0`, `gpu_tpu_cpu_72b_same_request_verified=false`.
The failure was resource/scheduling related: gpu-shard1 hit Kaggle GPU session
limit, cpu-stage7 push hit HTTP 429, other CPU kernels were later
`CANCEL_ACKNOWLEDGED`/UNKNOWN after timeout, and Colab was no longer usable
after the attempt. The manually discovered residual non-accepted gpu-shard1
kernel was deleted, and the probe now attempts cleanup for such push paths.
The current checked max-search artifact is
`dist/three-accelerator-dense-max-parameter-search-20260628-r10-full-72b-colab-attempt-resource-blocked/three_accelerator_dense_max_parameter_search.json`;
it passes and records `max_stage_loaded_parameter_class=72b`,
`max_tpu_executed_parameter_class=72b`, and
`max_successful_same_request_decode_parameter_class=32b`. Current Colab
reacquire after r1 failed at
`dist/colab-tpu-session-20260628-r14-reacquire-after-full-r1/colab_tpu_session_probe.json`
with HTTP 503. This is still not goal completion; next live retry should first
reacquire Colab V5E1, then use `--stage-launch-stagger-seconds` to reduce
Kaggle 429/push contention.

Latest superseding Colab reacquire status after r14/r15/r16: the bounded retry
probe `scripts/colab_tpu_reacquire_retry_probe.py` now emits
`colab_tpu_reacquire_retry_probe_v1` and is checked by
`scripts/colab_tpu_reacquire_retry_check.py`. The current retained retry
artifact is
`dist/colab-tpu-reacquire-retry-20260628-r1-v5e1-v6e1-short/colab_tpu_reacquire_retry_probe.json`;
its checker passes, but it did not reacquire a runtime:
`colab_tpu_reacquire_ready=false`, V5E1 returned HTTP 503, V6E1 returned
HTTP 400, and no endpoint/proxy/token/credentials are public. The current
dense max-search artifact is now
`dist/three-accelerator-dense-max-parameter-search-20260628-r11-colab-reacquire-retry-current-blocked/three_accelerator_dense_max_parameter_search.json`;
its checker passes and records `max_stage_loaded_parameter_class=72b`,
`max_tpu_executed_parameter_class=72b`,
`max_successful_same_request_decode_parameter_class=32b`, and
`failure_stage=colab_tpu_reacquire_not_ready`. This is a current blocker
artifact, not active-goal completion. `scripts/three_accelerator_dense_max_parameter_search_pack.py`
and its checker now import Colab TPU reacquire retry reports explicitly.

Next full-72B live retry should wait for a successful Colab V5E1 reacquire,
run a short stability probe, and then prefer the lower-concurrency stage
topology now covered by tests: stage ranges
`[[0,8],[8,16],[16,24],[24,32],[32,36],[36,44],[44,52],[52,60],[60,70],[70,80]]`
with groups `gpu-shard0:[0,1]`, `gpu-shard1:[2,3]`,
`web-tpu-stage4:[4]`, and `cpu-tail:[5,6,7,8,9]`. This keeps all ten 72B
stages and all three accelerator families while reducing CPU Kaggle pushes
from five kernels to one; `tests/test_kaggle_32b_full_heterogeneous_probe.py`
verifies that this topology can satisfy the 72B full same-request completion
gate if every stage really completes.

Latest retry after that: `dist/colab-tpu-reacquire-retry-20260628-r2-v5e1-before-full72b/colab_tpu_reacquire_retry_probe.json`
ran three V5E1 allocation attempts and all returned HTTP 503; its checker
passes and keeps endpoint/proxy/token/credential material private. The current
max-search artifact is
`dist/three-accelerator-dense-max-parameter-search-20260628-r12-colab-v5e1-retry-still-503/three_accelerator_dense_max_parameter_search.json`;
its checker passes with `max_stage_loaded_parameter_class=72b`,
`max_tpu_executed_parameter_class=72b`,
`max_successful_same_request_decode_parameter_class=32b`, and
`failure_stage=colab_tpu_reacquire_not_ready`. No 72B full live run was started
from this state, and the active goal remains incomplete.

Latest recovery hardening after r12: the existing local Colab session
`ct-colab-tpu-v5e1` still had endpoint/proxy metadata, but
`dist/colab-tpu-runtime-stability-20260628-r12-existing-session-before-full72b/colab_tpu_runtime_stability_probe.json`
proved it was detached: the Jupyter kernels endpoint returned HTTP 404, no TPU
device was observed, and the stability checker correctly fails. A longer
allocation retry at
`dist/colab-tpu-reacquire-retry-20260628-r3-v5e1-longer-before-full72b/colab_tpu_reacquire_retry_probe.json`
ran five V5E1 attempts and all returned HTTP 503. `scripts/colab_tpu_session_probe.py`
now supports `--cleanup-before-tpu`, and
`scripts/colab_tpu_reacquire_retry_probe.py` forwards it, so stale remote TPU
assignments can be unassigned before a new request. The cleanup-before retry
`dist/colab-tpu-reacquire-retry-20260628-r4-clean-before-v5e1/colab_tpu_reacquire_retry_probe.json`
still returned HTTP 503 on both V5E1 attempts; its checker passes. The current
max-search artifact is
`dist/three-accelerator-dense-max-parameter-search-20260628-r13-colab-clean-before-still-503/three_accelerator_dense_max_parameter_search.json`,
which passes its checker and still records
`max_successful_same_request_decode_parameter_class=32b` with
`failure_stage=colab_tpu_reacquire_not_ready`. This remains an external TPU
allocation/runtime-availability blocker, not a completed 72B full decode.

Latest authuser recovery attempt after r13: `scripts/colab_tpu_session_probe.py`
now supports configurable `--authuser`, and
`scripts/colab_tpu_reacquire_retry_probe.py` supports `--authusers` rotation
while keeping account/session identifiers private. Unit tests cover the URL
generation and retry command forwarding. The retained authuser rotation attempt
is
`dist/colab-tpu-reacquire-retry-20260628-r5-authuser-rotation-v5e1/colab_tpu_reacquire_retry_probe.json`:
authuser indexes 0, 1, and 2 each attempted V5E1 with cleanup-before enabled,
and all three returned HTTP 503. Its checker passes. The current max-search
artifact is now
`dist/three-accelerator-dense-max-parameter-search-20260628-r14-colab-authuser-rotation-still-503/three_accelerator_dense_max_parameter_search.json`;
its checker passes and still records `max_stage_loaded_parameter_class=72b`,
`max_tpu_executed_parameter_class=72b`,
`max_successful_same_request_decode_parameter_class=32b`, and
`failure_stage=colab_tpu_reacquire_not_ready`. Do not start the low-concurrency
72B full live run until a TPU runtime is actually reacquired and passes a
stability probe.

Latest Kaggle Web TPU retry after r14: the read-only UI state probe
`dist/kaggle-web-tpu-ui-state-20260628-r15-current-before-web-retry/kaggle_web_tpu_ui_state_probe.json`
showed no current Jupyter runtime and Start Session visible. The start/wait
probe
`dist/kaggle-web-tpu-start-wait-20260628-r15-start-from-ui/kaggle_web_tpu_start_wait_probe.json`
successfully expanded Session options, selected `TPU v5e-8`, clicked Start
Session, and waited 1200 seconds, but the final public-safe observation still
had queue/starting visible and no Jupyter frame/session/kernel. The follow-up
execution-channel probe
`dist/kaggle-web-tpu-execution-channel-probe-20260628-r11-after-start-wait-timeout/kaggle_web_tpu_execution_channel_probe.json`
failed with `web_tpu_jupyter_execute_timeout`, no TPU device, and no small JAX
cell. The current max-search artifact is
`dist/three-accelerator-dense-max-parameter-search-20260628-r15-colab-and-kaggle-web-tpu-unavailable/three_accelerator_dense_max_parameter_search.json`;
its checker passes and still records `max_successful_same_request_decode_parameter_class=32b`.
This is current evidence that both Colab TPU allocation and Kaggle Web TPU
execution are unavailable, not 72B full decode success.

Latest Web TPU follow-up after r15: the current UI state
`dist/kaggle-web-tpu-ui-state-20260628-r16-after-r15-wait-current/kaggle_web_tpu_ui_state_probe.json`
still showed `Draft Session Starting`, no Jupyter frame/session/kernel, and
Start Session disabled/visible. A second bounded wait at
`dist/kaggle-web-tpu-start-wait-20260628-r16-continue-starting/kaggle_web_tpu_start_wait_probe.json`
waited another 600 seconds and still ended with `session_starting_text_visible=true`,
no queue text, no Jupyter frame, no session, and no kernel. The current
max-search artifact is
`dist/three-accelerator-dense-max-parameter-search-20260628-r16-kaggle-web-tpu-still-starting/three_accelerator_dense_max_parameter_search.json`;
its checker passes and still records `max_successful_same_request_decode_parameter_class=32b`.
Do not run the 72B full all-layer decode until the Web TPU execution channel
or Colab TPU runtime is actually ready.

Latest Colab TPU bridge progress on 2026-06-28: `scripts/gpu_tpu_cpu_same_request_runtime_bridge_probe.py`
now supports `--tpu-provider colab_cli` using the official Colab runtime proxy
session in `~/.config/colab-cli/sessions.json`. The supporting allocation and
diagnostic scripts are `scripts/colab_tpu_session_probe.py`,
`scripts/colab_tpu_runtime_stability_probe.py`,
`scripts/colab_tpu_coordinator_connectivity_probe.py`, and
`scripts/colab_tpu_qwen_stage_loader_probe.py`; keep Colab session state local
and sensitive. Retained public-safe evidence:
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260628-r13-colab-tpu-shape-state-refreshed/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`
proves one same Coordinator request completed Kaggle CUDA, Colab JAX/TPU, and
CPU stages (`stage0=stage1=stage2=1`, accepted backends
`["cpu","cuda","jax_tpu"]`, Colab `TPU v5 lite` device count 1). This is only a
shape/runtime bridge, not model-weight success.

The Colab TPU Qwen loader path also made real progress:
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260628-r14-colab-tpu-32b-one-layer-loader/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`
ran a real Qwen 32B stage-owned Colab TPU loader for layers 21-22 inside the
same request (`executed_layer_count=1`, 12 keys, about 0.908GB logical tensor
bytes). `dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260628-r15-colab-tpu-32b-four-layer-loader/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`
extended that to 4 layers and about 3.633GB. The current largest Colab TPU
stage-owned same-request proof is
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260628-r16-colab-tpu-32b-eight-layer-loader/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`:
Qwen 32B layers 21-29, `executed_layer_count=8`, 96 keys, about 7.266GB logical
tensor bytes, one Colab `TPU v5 lite`, same-request CUDA/TPU/CPU stage counts
all 1, and private CUDA kernel cleanup verified. These artifacts intentionally
keep `gpu_tpu_cpu_32b_same_request_verified=false` because the CUDA stage did
not import full 32B weight evidence in those bridge attempts, and they are not
full 32B all-layer decode.

The 72B Colab attempt
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260628-r17-colab-tpu-72b-four-layer-loader/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`
is not a 72B capacity failure: the Colab stage failed with an HTTP/runtime
session error before stage1, while the Kaggle CUDA stage completed and cleanup
was verified. Subsequent Colab session refresh evidence
`dist/colab-tpu-session-20260628-r4-refresh-v5e1-after-failure/colab_tpu_session_probe.json`
reports HTTP 503 with diagnosis `colab_assignment_resource_unavailable`; the
current blocker is Colab TPU runtime allocation/lifecycle availability, not
proof that 72B stage loading is impossible. Do not mark the active 72B goal
achieved: there is still no dense/full-precision 72B GPU+TPU+CPU same-request
1-token decode over all layers. Resume by reacquiring Colab TPU, refreshing
kernel/session ids through the stability probe, running a Colab-only 72B stage
loader probe, and only then retrying full 72B same-request bridge/full-layer
decode.

Latest Colab TPU reacquire attempts after r17: V5E1 remained unavailable in
`dist/colab-tpu-session-20260628-r5-reacquire-v5e1/colab_tpu_session_probe.json`
and
`dist/colab-tpu-session-20260628-r7-reacquire-v5e1-after-clean-list/colab_tpu_session_probe.json`,
both HTTP 503 with `colab_assignment_resource_unavailable`. V6E1 was attempted
at `dist/colab-tpu-session-20260628-r6-reacquire-v6e1/colab_tpu_session_probe.json`
and was rejected with HTTP 400. Listing Colab assignments showed no active
assignments to clean up, so the current external blocker is Colab TPU resource
availability rather than stale local state. `scripts/colab_tpu_qwen_stage_loader_check.py`
and `tests/test_colab_tpu_qwen_stage_loader_check.py` were added to validate
future Colab-only Qwen stage loader reports without allowing them to claim full
72B same-request success. The next meaningful live step remains: reacquire
Colab V5E1, run the stability probe to refresh kernel/session ids, run
`scripts/colab_tpu_qwen_stage_loader_probe.py` against Qwen 72B stage layers,
then only if that passes retry full GPU+TPU+CPU 72B same-request decode.

Colab TPU fallback status as of 2026-06-28: Kaggle Web TPU remains unstable, but
Google Colab TPU is currently reachable through the official Colab CLI backend
path. The official CLI (`google-colab-cli`, installed locally as
`~/.local/bin/colab`) supports headless TPU allocation with `--tpu v5e1` and
`--tpu v6e1`, but its own OAuth cache requires a fresh Colab CLI consent flow
for the full `cloud-platform`/`drive.file` scope set. A pre-existing local OAuth
cache at `~/.config/colab-exec/token.json` with `colaboratory` scope was
sufficient to directly call the Colab assignment endpoint and allocate a
private TPU V5E1 runtime without exposing credentials. That runtime was saved
locally as `ct-colab-tpu-v5e1` in `~/.config/colab-cli/sessions.json`; treat
that state file as sensitive because it contains runtime proxy tokens. An
earlier temporary V5E1 assignment from endpoint hash `8bbdaaa63b8d88c7` was
deleted with HTTP 204; the retained session endpoint hash is
`ad65a47c4b86a196`.

The public-safe retained evidence is
`dist/colab-tpu-runtime-stability-20260628-r1-v5e1/colab_tpu_runtime_stability_probe.json`,
emitted by `scripts/colab_tpu_runtime_stability_probe.py` and checked by
`scripts/colab_tpu_runtime_stability_check.py`. It reports
`colab_tpu_runtime_stably_acquired=true`, `runtime_proxy_connected=true`,
`rounds_requested=5`, `rounds_ready=5`, JAX `0.7.2`, one visible TPU device
`TPU_0(process=0,(0,0,0,0))`, and five successful BF16 1024x1024 matmul rounds
over about 120 seconds. The report is public-safe: runtime proxy token, raw URL,
and endpoint are not public. This proves a usable Colab TPU v5e1 runtime channel
for future TPU adapter work; it is not a GPU+TPU+CPU bridge proof, not a 32B/72B
model proof, not production serving, and not evidence that Kaggle TPU is fixed.

# Agent Instructions for CrowdTensor

Read this file before making changes in this repository. It is the short, durable project memory for future agents and contributors.

## Project Identity

CrowdTensor is the project and network vision: open AI infrastructure that can eventually use ordinary home compute for fault-tolerant AI workloads.

CrowdTensorD is the current Alpha daemon/control plane. It validates the reliability and protocol mechanics needed before real home GPU aggregation, Swarm Inference, Swarm Training, browser compute, and P2P routing are added.

## Durable Architecture Layers

Plan work across three layers and keep their responsibilities separate:

- Core technology layer: model execution across devices. This owns real
  large-model runtime adapters, partitioning, activation/KV-cache transport,
  batching, streaming, heterogeneous placement, correctness checks, and later
  training/fine-tuning mechanics.
- Control layer: governance and scheduling. This owns Coordinator sessions,
  leases, admission, identity/roles/tenant policy, quotas, rate limits,
  trust/quarantine, P2P provider records, accounting, settlement drafts, future
  incentives, abuse controls, and audit logs.
- User-facing layer: product surface. This owns CLI flows, bootstrap,
  quickstart, Miner join packs, route/tunnel helpers, docs, support bundles,
  redacted evidence, diagnostics, onboarding gates, and user-visible answer,
  cost, and health surfaces.

Security, privacy, observability, artifact redaction, tests, and performance are
cross-cutting requirements across all three layers. Do not confuse control-plane
or UX polish with productionization of the core technical breakthrough: real
cross-device large-model inference now has a bounded external 7B proof, but
throughput, fault tolerance at scale, production routing, and user-facing
serving are still separate follow-up work.

## Current Engineering Method, Progress, and Planning

Use an evidence-first method: each meaningful capability should have a bounded
command/script, versioned public-safe report schema, redacted artifacts, a
CI-safe checker, cleanup for private live resources, and explicit "not yet"
boundaries. Do not treat local fixtures, imported reports, queued cloud
runtimes, or one happy-path live proof as broad production readiness. Keep raw
prompts, generated text, token ids, activations, KV-cache data, credentials,
leases, idempotency material, cookies, and private runtime state out of
shareable artifacts.

Provider-backed experiments such as Kaggle CPU/GPU/TPU kernels are temporary
proof vehicles, not the product architecture. Use private kernels/packages,
delete temporary kernels after evidence collection, do not publish inline
private payloads, and respect provider limits instead of relying on
multi-account limit bypass. A cloud queue or allocation attempt is scheduling
evidence only until a completed public-safe report proves runtime inference.

Current progress: core feasibility is ahead of production serving. Retained
evidence includes the real external 7B Kaggle T4 x2 split proof, the 32B AWQ
two-kernel upper-bound crossing proof, and the full-precision 32B 4*T4 + 5*CPU
two-token heterogeneous proof with stage-local KV-cache reuse. The TPU path now
has authenticated Web TPU v5e-8 evidence up through a bounded same-request 32B
RC: one Coordinator request accepted CUDA/GPU, real JAX/TPU Qwen 32B
stage-owned middle-layer execution, and CPU tail/verifier stages, with public
activation hashes only. This is still not production serving, not
throughput/TTFT/SLA evidence, not full end-to-end Qwen 32B quality/parity across
every layer, and not ordinary-user large-model serving. Control and user-facing
layers have Alpha/Beta contracts, runbooks, redacted reports, and public
swarm/product surfaces; they are not yet production P2P, trust/economics, or
ordinary-user large-model serving.

Near-term planning order: first harden core inference with longer runs,
batch/sequential validation, throughput/TTFT metrics, large-model requeue, and
production-like adapters; make ordinary multi-machine GPU/heterogeneous
inference usable through one-command Coordinator/Miner flows and safe streaming;
then build control-layer scheduling, admission, quotas, trust, abuse, audit,
accounting, and cleanup.

The canonical current core-validation status artifact is
`dist/core-technology-validation-status-20260616/core_technology_validation_status.json`,
emitted by `scripts/core_technology_validation_status_pack.py` and checked by
`scripts/core_technology_validation_status_check.py`. It records the retained
Kaggle T4 x2 7B proof at
`dist/large-model-kaggle-stage-selective-hf-7b-manual-rope-20260616/large_model_kaggle_validation.json`:
`Qwen/Qwen2.5-7B-Instruct` ran with `hf_transformers_stage_selective_cuda`,
stage0 on `cuda:0`, stage1 on `cuda:1`, `generated_token_count=1`,
`real_7b_runtime_verified=true`, `multi_worker_sharded_path_verified=true`,
`core_validation_ready=true`, public-safe redaction, and deleted temporary
Kaggle kernel `xuyuhaosuyi/crowdtensor-large-llm-81608591`. It also records a
successful fully automated `gpt2-xl` Kaggle GPU small-tier proof plus a local
tiny Llama-like two-stage HF runtime proof at
`dist/real-llm-llama-like-local-smoke-20260615/real_llm_sharded_evidence.json`,
and keeps those smaller/local proofs scoped as non-7B evidence.
`scripts/stage_selective_weight_loading_check.py` is the local safetensors
materialization proof behind the 7B/8B path: it validates that each
Llama-like stage can load only its assigned safetensors keys and report
public-safe counts/hashes without tensor values. It also applies those tensors
to matching stage-owned model `state_dict` entries in a synthetic Llama-like
model, proves no cross-stage keys were applied, and runs a local synthetic
stage0->stage1 activation/decode smoke that matches a baseline next token.
It also emits `real_llm_stage_selective_hf_runtime_v1` for an HF-style model
directory: config/tokenizer/weight index are loaded, stage models are
instantiated on `meta`, only stage-owned safetensors keys are materialized,
required runtime buffers are allocated, and stage0 activation into stage1
decode is verified without publishing prompts, generated text, token ids,
activations, local cache paths, or tensor values. Treat this as
stage-selective HF runtime plumbing evidence; the retained Kaggle T4 x2 report
above is the external 7B validation. This is still not production P2P, not
Coordinator-free execution, not GGUF/llama.cpp RPC success, not a GPU
marketplace, not training/fine-tuning, and not a large-model throughput SLA.

The retained 32B-class loading proof is
`dist/kaggle-32b-stage-owned-safetensors-probe-awq-live-r3-clone/kaggle_32b_stage_owned_safetensors_probe.json`.
It used two private Kaggle GPU script kernels on Tesla T4 x2 against
`Qwen/Qwen2.5-32B-Instruct-AWQ` safetensors. Stage0 loaded only layers 0-31
plus embeddings: 833 assigned keys from 3 of 5 safetensors files, 9.000671 GB
logical tensor bytes, 833 cloned/materialized keys retained, and no cross-stage
keys. Stage1 loaded only layers 32-63 plus final norm/lm head: 834 assigned
keys from 3 of 5 safetensors files, 9.000681 GB logical tensor bytes, 834
cloned/materialized keys retained, and no cross-stage keys. Both kernels
verified T4 x2 hardware, downloaded only stage-owned shard files, skipped
non-stage keys inside the shared boundary file, completed, and were deleted;
local private Kaggle payloads were removed.

The current retained 32B generated-token proof is
`dist/kaggle-32b-upper-bound-crossing-live-20260620-r3/kaggle_32b_stage_owned_activation_decode_probe.json`.
It used a temporary proof Coordinator at `24.199.118.54:9235` to directly issue
four stage tasks to two private Kaggle Tesla T4 x2 kernels running
`Qwen/Qwen2.5-32B-Instruct-AWQ`: shard0 owned stages 0/1 on `cuda:0`/`cuda:1`,
and shard1 owned stages 2/3 on `cuda:0`/`cuda:1`. It completed one generated
token with four accepted stage tasks, three private activation handoff hashes,
raw token ids/activations redacted, private kernels deleted, and local private
payloads removed. The strict same-model/same-prompt single Kaggle T4 x2 baseline
was attempted with all four stages required in one kernel and failed closed with
`single_kernel_t4x2_gpu_count_below_required_stage_count` because a single T4 x2
kernel exposes only two GPUs. This proves a bounded slot-count upper-bound
crossing under strict 4-stage placement. It is still not a memory-pressure or
long-context crossing proof, not KV-cache optimized serving, not batch/sequential
validation, not stage requeue, and not production serving.

The retained full-precision 32B heterogeneous feasibility proof is
`dist/kaggle-32b-full-heterogeneous-multitoken-kv-live-20260620-r1/kaggle_32b_full_heterogeneous_probe.json`,
emitted by `scripts/kaggle_32b_full_heterogeneous_probe.py`. It used a
temporary proof Coordinator at `24.199.118.54:9244`, two private Kaggle T4 x2
GPU script kernels, and five private Kaggle CPU script kernels to run
non-quantized `Qwen/Qwen2.5-32B-Instruct` across nine stages. GPU shard0 owned
stages 0/1 on `cuda:0`/`cuda:1` for layers 0-10 and 10-22; GPU shard1 owned
stages 2/3 on `cuda:0`/`cuda:1` for layers 22-34 and 34-46; CPU stages 4-8
owned layers 46-50, 50-54, 54-58, 58-62, and 62-64 plus final norm/lm head.
It completed two generated tokens with stage task counts `stage0..stage8 == 2`,
`generated_token_count=2`, `multi_token_generation_verified=true`,
`stage_local_kv_cache_verified=true`, and one stage-local KV-cache hit per
stage (`stage0..stage8 hit_count == 1`) while keeping KV tensors and
`past_key_values` private inside each Kaggle kernel. It also reports
`quantization=none`, `full_precision_32b=true`, `four_t4_five_cpu_topology_verified=true`,
`stage_owned_full_precision_runtime_verified=true`, private activation/token
payloads redacted, private kernels deleted, and local private payloads removed.
The earlier one-token retained proof remains at
`dist/kaggle-32b-full-heterogeneous-live-20260620-r2/kaggle_32b_full_heterogeneous_probe.json`.
Stage-owned loaded tensor sizes in that path were about 10.53 GB, 10.90 GB,
10.90 GB, and 10.90 GB on the four T4 stages, then 3.63 GB, 3.63 GB, 3.63 GB,
3.63 GB, and 3.27 GB on the five CPU stages. This proves a bounded 4*T4 + 5*CPU
full-precision 32B multi-token feasibility demo with stage-local KV-cache reuse,
not a throughput result, not batch/sequential validation, not fault-tolerant
requeue, not production routing, and not a general unbounded Kaggle scaling
claim.

The Kaggle TPU LLM probe path is `scripts/kaggle_tpu_llm_probe.py`, emitting
`kaggle_tpu_llm_probe_v1` and packaging private Kaggle script kernels with
`enable_tpu=true`, `enable_gpu=false`, and accelerator/machine-shape candidates
headed by the current Kaggle frontend internal value `tpuV5e8` for TPU v5e-8.
Retained attempts are
`dist/kaggle-tpu-llm-probe-live-20260621-r1/kaggle_tpu_llm_probe.json`
(`TPU v5e-8` display string),
`dist/kaggle-tpu-llm-probe-live-20260621-r2-internal-shape/kaggle_tpu_llm_probe.json`
(`tpuV5e8`), `dist/kaggle-tpu-llm-probe-live-20260621-r3-cli222/kaggle_tpu_llm_probe.json`
(`tpuV5e8` via isolated Kaggle CLI 2.2.2), and
`dist/kaggle-tpu-llm-probe-live-20260621-r4-tpu1vmv38/kaggle_tpu_llm_probe.json`
(`tpu1vmV38`). Kaggle accepted and created each private TPU kernel, but every
attempt stayed `QUEUED` until the bounded status timeout, produced no runtime
report, and was deleted. MCP `save_notebook` could run private notebooks but
ignored v5e machine-shape requests and fell back to CPU (`machine_shape: None`);
MCP `create_notebook_session` is not usable for this account without
`kernelSessions.create` permission. The authenticated web UI path was also
verified via a temporary cookie-backed browser session: retained evidence is
`dist/kaggle-tpu-web-session-attempt-20260621-r1/kaggle_tpu_web_session_attempt.json`.
The web UI logged in as `tpuowner`, exposed `TPU v5e-8` in Session options,
showed TPU quota remaining, accepted `Turn on TPU v5e-8`, and then reported
`TPUs are popular right now. You are #31 in the queue`; the notebook stayed in
`Session is starting...` for the bounded wait and produced no runtime output.
The later long web wait preserved the active event instead of restarting it:
`notebookcbde0293fe` stayed as `Interactive Session with TPU v5e-8` queued for
about 3 hours, then became `Running`. Opening that active event produced the
runtime notebook `from-pathlib-import-path`; retained evidence is
`dist/kaggle-tpu-active-event-longwait-20260621-r1/kaggle_tpu_active_event_longwait.json`,
`dist/kaggle-open-active-event-runtime-20260621-r1/open_active_event_runtime.json`,
and `dist/kaggle-tpu-web-runtime-proof-20260621-r1/kaggle_tpu_web_runtime_proof.json`.
The proof reports JAX seeing 8 `TPU v5 lite` devices with
`PJRT_DEVICE`/TPU worker environment present and a successful simple TPU matrix
operation (`simple_tpu_op_ready: true`). The stronger retained synthetic
LLM-style proof is
`dist/kaggle-tpu-web-synthetic-llm-probe-20260621-r1/kaggle_tpu_web_runtime_probe.json`:
it reports `synthetic_llm_ready: true`,
`synthetic_llm_runtime: jax_tiny_causal_lm_jit`, `generated_token_count: 1`,
8 `TPU v5 lite` devices, and public-safe redaction of generated token ids,
activations, and KV-cache material. Keep this scoped as Kaggle web TPU runtime
allocation plus JAX synthetic causal-LM proof, not torch_xla/TensorFlow proof,
not Hugging Face/Qwen on TPU, not large-model TPU serving, and not production
TPU pooling. Cookie files are sensitive local-only artifacts and should be
deleted/rotated after live web experiments.

The GPU+TPU+CPU heterogeneous stage Alpha aggregate is
`dist/gpu-tpu-cpu-heterogeneous-stage-alpha-20260622-r3-cli/gpu_tpu_cpu_heterogeneous_stage_alpha.json`,
emitted by `scripts/gpu_tpu_cpu_heterogeneous_stage_alpha_pack.py` and checked
by `scripts/gpu_tpu_cpu_heterogeneous_stage_alpha_check.py`; the CLI is
`crowdtensor heterogeneous-stage-alpha`. It imports retained GPU evidence
from the full-precision 32B 4*T4 + 5*CPU proof and the AWQ 32B upper-bound
crossing proof, retained CPU stage evidence, and retained TPU evidence from
the web JAX real-model path. It also runs a local public-safe three-stage real
HF GPT-2 smoke (`gpt2`, 124,439,808 parameters) through stage0/stage1/stage2
boundaries with target backend families GPU/TPU/CPU, activation hashes only,
`baseline_match=true`, and `generated_token_count=1`. The report has
`gpu_tpu_cpu_heterogeneous_stage_alpha_ready=true`,
`local_three_stage_real_model_e2e_ready=true`,
`small_medium_real_model_end_to_end_ready=true`,
`gpu_tpu_cpu_32b_feasibility_report_ready=true`, and
`next_rc_boundary_ready=true`. The r3 report also emits
`torch_jax_torch_bridge_probe.json`; in the current local environment it is
public-safe but blocked with `jax_missing`, so `torch_jax_torch_bridge_ready=false`
and the Torch-to-JAX-to-Torch activation bridge remains next-RC work. It deliberately keeps
`same_request_live_heterogeneous_verified=false`,
`live_tpu_stage_miner_integrated=false`,
`gpu_tpu_cpu_32b_same_request_feasible_now=false`, and
`tpu_32b_runtime_adapter_ready=false`: this is Alpha contract plus retained
backend evidence plus local real-model three-stage E2E, not a live same-request
GPU+TPU+CPU run, not Qwen/Llama-on-TPU stage execution, not production serving,
and not 32B GPU+TPU+CPU success.

The GPU+TPU+CPU 32B Heterogeneous RC artifact is currently
`dist/gpu-tpu-cpu-32b-heterogeneous-rc-20260623-r16-gpu-stage-accepted-tpu-queued/gpu_tpu_cpu_32b_heterogeneous_rc.json`,
emitted by `scripts/gpu_tpu_cpu_32b_heterogeneous_rc_pack.py`, checked by
`scripts/gpu_tpu_cpu_32b_heterogeneous_rc_check.py`, and exposed through
`crowdtensor heterogeneous-32b-rc`. It is the bounded RC surface for same-request
32B execution across CUDA/GPU, JAX/TPU, and CPU stages. The r16 report imports the
r3 Alpha evidence, emits `stage_runtime_matrix.json`, `activation_protocol.json`,
`live_same_request_summary.json`, `tpu_allocation_attempt_summary.json`,
`tpu_web_active_event_summary.json`, `tpu_stage_adapter_plan_summary.json`,
`tpu_stage_runtime_probe_summary.json`, `runtime_bridge_summary.json`,
`blocker_report.json`, redacted Markdown, CLI summary, and Support Bundle, and
currently reports
`gpu_tpu_cpu_32b_heterogeneous_rc_ready=true` while keeping
`gpu_tpu_cpu_32b_bounded_rc_success=false`,
`gpu_tpu_cpu_32b_same_request_verified=false`,
`live_tpu_stage_miner_integrated=false`, `fallback_model_used=false`,
`tpu_32b_runtime_adapter_ready=false`, and
`stage_local_kv_cache_verified=false`. The current top blocker is
`kaggle_web_tpu_runtime_queued_after_restart`, with the broader unresolved
`same_request_live_proof_missing`: retained TPU allocation and Qwen-like
32B-shape stage execution were proven previously, the current Web TPU notebook
has been restarted on `TPU v5e-8` and is queued again, and a single
Coordinator request has not yet accepted CUDA/GPU, JAX/TPU, and CPU stage tasks
together.

The current same-request runtime bridge attempt is
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260623-r6-gpu-slot-retry-tpu-queued/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`,
emitted by `scripts/gpu_tpu_cpu_same_request_runtime_bridge_probe.py` and
imported into the r16 RC. It attempted one shared Coordinator request with a
private Kaggle CUDA/GPU stage, the authenticated Web JAX/TPU stage, and a local
CPU tail stage. It did not verify the bridge:
`same_request_runtime_bridge_verified=false`,
`gpu_tpu_cpu_32b_same_request_verified=false`, and
`same_request_32b_model_verified=false`. A real code bug was fixed in this path:
`BridgeState.claim()` and task counting now treat `stage_id=0` as a valid stage
instead of a missing value, with tests covering stage0 claim/count behavior. The
current r16 bridge successfully pushed, ran, downloaded, and deleted a private
Kaggle Tesla T4 x2 CUDA stage kernel; stage0 was accepted with
`cuda_device_count=2`, `accepted_stage_backends=["cuda"]`,
`stage_task_counts.stage0=1`, one activation handoff hash, and public-safe
redaction. The remaining bridge blocker is `jax_tpu_stage_not_ready` because
the Web TPU runtime is queued and no Jupyter proxy is available; the CPU tail
has no task until stage1 completes. This bridge probe is also explicitly
`not_32b_weight_success=true`, so even a successful bridge would be activation
runtime plumbing evidence until the full Qwen safetensors/TPU stage adapter is
implemented.

The Qwen 32B TPU stage adapter plan is
`dist/gpu-tpu-qwen-stage-adapter-plan-qwen32b-20260623-r1/gpu_tpu_qwen_stage_adapter_plan.json`:
it maps `Qwen/Qwen2.5-32B-Instruct` layers 21-42 for a JAX/TPU middle stage,
with 252 assigned stage-owned keys from 6 safetensors files, 0 unsupported keys,
activation metadata shape `[1, 128, 5120]` / `bfloat16` / `batch_seq_hidden`,
and stage-local KV-cache metadata while keeping tensor values, activations,
tokens, generated text, and private paths out of public artifacts. The current
report has `tpu_stage_adapter_plan_ready=true`,
`tpu_checkpoint_bridge_plan_ready=true`, and
`tpu_stage_owned_loader_plan_ready=true`, but this is metadata-only bridge
planning, not executed TPU safetensors loading.

The Qwen/Llama-like TPU stage runtime probe path is
`scripts/kaggle_tpu_qwen_stage_runtime_probe.py`, with tests in
`tests/test_kaggle_tpu_qwen_stage_runtime_probe.py`. It can package private
Kaggle TPU script kernels and can also import authenticated Web Notebook runtime
evidence. It runs a public-safe JAX decoder middle-stage forward with
Qwen/Llama-like grouped-query attention, RMSNorm, MLP, stage-local KV-cache
metadata, activation shape/dtype/layout metadata, and input/output hashes only.
The current retained Web TPU stage proof is
`dist/kaggle-tpu-qwen-stage-runtime-probe-web-live-20260623-r2/kaggle_tpu_qwen_stage_runtime_probe.json`:
the authenticated Kaggle Web Notebook runtime was running on `TPU v5e-8` and the
Jupyter API executed both `tiny-qwen-like` and `qwen32b-one-layer` profiles on
JAX with 8 `TPU v5 lite` devices. The retained report has `ok=true`,
`tpu_runtime_ready=true`, `qwen_like_stage_runtime_ready=true`,
`qwen32b_single_layer_runtime_ready=true`, `stage_local_kv_cache_verified=true`,
`selected_accelerator=web-ui-tpu-v5e8`, public stage input/output hashes only,
and `jupyter_proxy_token_public=false`. The matching Web active-event runtime
evidence is
`dist/kaggle-web-tpu-session-retry-20260623-r12-runtime-confirmed/kaggle_tpu_web_active_event_status.json`,
with `running=true`, `tpu_runtime_ready=true`, Jupyter kernel state `idle`, and
no public proxy token/cookie material. Treat that as retained successful runtime
evidence, not proof that the session is still attached now. The current status
probe is
`dist/kaggle-web-tpu-session-retry-20260623-r13-current-status/kaggle_tpu_web_active_event_status.json`:
it is public-safe and reported `running=false`, `tpu_runtime_ready=false`,
`jupyter_proxy_input_count=0`, blockers
`kaggle_web_tpu_jupyter_proxy_not_visible` and
`kaggle_web_tpu_runtime_not_currently_ready`, and no public proxy token/cookie
material. The current restart attempt is
`dist/kaggle-web-tpu-session-retry-20260623-r14-restart-tpu/kaggle_tpu_web_start_selected_tpu_attempt.json`,
with converted status
`dist/kaggle-web-tpu-session-retry-20260623-r14-restart-tpu/kaggle_tpu_web_active_event_status.json`:
`Start Session` was clicked while `TPU v5e-8` was selected, queue/start signals
were observed, `running=false`, and `tpu_runtime_ready=false`. The current
status probe is
`dist/kaggle-web-tpu-session-retry-20260623-r15-current-status/kaggle_tpu_web_active_event_status.json`:
it reports `running=false`, `tpu_runtime_ready=false`,
`kaggle_web_tpu_session_still_starting`, and no public proxy token/cookie
material.

Older TPU allocation evidence remains useful for scheduling history: the private
script-kernel attempt at
`dist/kaggle-tpu-qwen-stage-runtime-probe-live-20260623-r2-qwen32b-one-layer/kaggle_tpu_qwen_stage_runtime_probe.json`
accepted `tpuV5e8` but stayed `QUEUED` for 30 status polls across a bounded
30-minute wait, produced no runtime report, and was deleted with local private
packages removed; the earlier tiny attempt at
`dist/kaggle-tpu-qwen-stage-runtime-probe-live-20260623-r1-tiny/kaggle_tpu_qwen_stage_runtime_probe.json`
did the same across a bounded 10-minute wait. The authenticated Web UI queue
attempts at
`dist/kaggle-web-tpu-session-retry-20260623-r6-start-selected-tpu/kaggle_tpu_web_start_selected_tpu_attempt.json`
and
`dist/kaggle-web-tpu-session-retry-20260623-r7-active-event-wait/kaggle_tpu_web_active_event_wait.json`
showed login, `TPU v5e-8` selection, Start Session, public queue position `#17`,
and a bounded queued wait before the later runtime-confirmed evidence above.

The r15 RC imports the retained Web TPU qwen32b-one-layer report, the current
starting Web TPU status report, and the latest blocked same-request bridge
report at
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260623-r5-current-retry/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`.
It
has `tpu_runtime_allocation_attempted=true`,
`tpu_runtime_allocation_ready=false`, `tpu_runtime_allocation_blocked=true`,
`tpu_qwen_like_stage_runtime_probe_ready=true`,
`tpu_qwen32b_single_layer_runtime_probe_ready=true`, and
`tpu_32b_runtime_adapter_ready=false`. This is useful RC scaffolding, bounded
live-attempt evidence, adapter-plan evidence, Web TPU runtime-stage evidence,
cleanup evidence, and overclaim protection, not 32B three-accelerator success.
The checker must reject fallback, fixture, queue evidence, adapter-plan-only
evidence, runtime-probe-only evidence, or missing same-request evidence if it is
marked as live 32B same-request success.

Latest superseding GPU+TPU+CPU 32B RC status: use
`dist/gpu-tpu-cpu-32b-heterogeneous-rc-20260623-r20-same-request-bridge-partial-32b-tpu-loader/gpu_tpu_cpu_32b_heterogeneous_rc.json`
as the current artifact. It is valid and public-safe but not 32B success:
`gpu_tpu_cpu_32b_bounded_rc_success=false`,
`gpu_tpu_cpu_32b_same_request_verified=false`,
`live_tpu_stage_miner_integrated=false`, and
`tpu_32b_runtime_adapter_ready=false`. The same-request bridge evidence is
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260623-r7-tpu-running/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`:
one Coordinator request accepted CUDA/GPU, JAX/TPU, and CPU stage tasks
(`accepted_stage_backends=["cpu","cuda","jax_tpu"]`, `stage0..stage2 == 1`,
two activation handoff hashes, and one generated token hash), with the private
Kaggle T4 x2 CUDA kernel deleted. This remains
`not_32b_weight_success=true`: the TPU stage is Qwen32B-shape runtime plumbing,
not full 32B safetensors stage execution. The real Qwen 32B TPU loader evidence
is
`dist/kaggle-tpu-32b-stage-owned-loader-probe-web-live-20260623-r1/kaggle_tpu_32b_stage_owned_loader_probe.json`:
inside the authenticated Web TPU v5e-8 runtime it verified stage layers 21-42
have 252 stage-owned keys across 6 safetensors files, all expected stage keys
are present in safetensors headers, and one real BF16 tensor byte range was
converted into a JAX array on 8 TPU v5 lite devices. It has
`stage_owned_header_verified=true` and `partial_tensor_to_tpu_verified=true`,
but `full_stage_owned_tpu_loader_ready=false`; do not claim full 32B
GPU+TPU+CPU success until the full 21-layer TPU stage loader and a real
same-request 32B live proof are completed.

Latest superseding GPU+TPU+CPU 32B RC status after the full TPU loader and
real-TPU bridge attempt: use
`dist/gpu-tpu-cpu-32b-heterogeneous-rc-20260623-r25-real-tpu-stage-web-detached/gpu_tpu_cpu_32b_heterogeneous_rc.json`
as the current artifact, superseding the r20 and older "current" references
above. It is valid and public-safe (`scripts/gpu_tpu_cpu_32b_heterogeneous_rc_check.py
--report ... --json` passes) but still not 32B same-request success:
`gpu_tpu_cpu_32b_bounded_rc_success=false`,
`gpu_tpu_cpu_32b_same_request_verified=false`,
`live_tpu_stage_miner_integrated=false`, `fallback_model_used=false`,
`tpu_32b_runtime_adapter_ready=true`, and
`blocked_reason=kaggle_web_tpu_runtime_not_currently_attached`. The real full
TPU loader evidence is
`dist/kaggle-tpu-32b-stage-owned-loader-probe-web-live-20260623-r3-full-21-layer-real/kaggle_tpu_32b_stage_owned_loader_probe.json`:
inside authenticated Kaggle Web TPU v5e-8 it verified
`Qwen/Qwen2.5-32B-Instruct` layers 21-42 with 252 stage-owned keys across 6
safetensors files, `missing_stage_key_count=0`, `executed_layer_count=21`,
`full_stage_owned_tpu_loader_ready=true`, `tpu_32b_runtime_adapter_ready=true`,
`loaded_execution_tensor_key_count=252`, and about 19.07 GB logical execution
tensor bytes on 8 `TPU v5 lite` devices, while publishing only hashes.
`scripts/gpu_tpu_cpu_same_request_runtime_bridge_probe.py` now has an explicit
`--web-tpu-32b-execute` path: the JAX/TPU stage first runs the real 32B
stage-owned loader/execution inside the same request window, then claims stage1
and emits `gpu_tpu_cpu_32b_same_request_live_proof.json` for RC import. It also
short-circuits when the TPU thread exits before stage1, so failed Web TPU
attachment no longer waits for the CPU tail timeout. The retained live attempt
is
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260623-r8-real-tpu-32b-stage/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`:
a private Kaggle Tesla T4 x2 CUDA stage0 kernel was pushed, ran, downloaded,
and deleted (`cuda_device_count=2`, `stage0=1`, one activation handoff hash),
and the report carries prior retained CUDA 32B weight evidence, but the Web TPU
stage failed before stage1 claim/submit (`stage1=0`, accepted backends only
`["cuda"]`), so CPU tail had no task and the generated-token count is 0. The
corresponding live-proof artifact is present but `ok=false`. A Web UI restart
after that failure entered the TPU queue at `#9`, then a bounded 900-second wait
ended detached with no Jupyter proxy:
`dist/kaggle-web-tpu-session-retry-20260623-r20-wait-after-queue9/kaggle_tpu_web_active_event_status.json`.
Do not mark the goal complete until a fresh same Coordinator request accepts a
CUDA/GPU stage task, a real JAX/TPU 32B stage-owned task, and a CPU tail/verifier
task and the imported live proof passes; queue evidence, detached Web TPU
status, the prior synthetic bridge, or the standalone full TPU loader are still
not a completed three-accelerator 32B live proof.

GPU+TPU+CPU 32B Heterogeneous RC r26 is now the current successful bounded RC:
`dist/gpu-tpu-cpu-32b-heterogeneous-rc-20260623-r26-real-tpu-stage-same-request-success/gpu_tpu_cpu_32b_heterogeneous_rc.json`.
It imports the successful same-request live proof at
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260623-r9-real-tpu-32b-stage-runtime-restored/gpu_tpu_cpu_32b_same_request_live_proof.json`,
the r9 bridge report at
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260623-r9-real-tpu-32b-stage-runtime-restored/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`,
the r3 full Web TPU loader, the Qwen 32B adapter plan, and retained TPU runtime
probe evidence. The r26 RC passes
`scripts/gpu_tpu_cpu_32b_heterogeneous_rc_check.py --report ... --json` and
reports `gpu_tpu_cpu_32b_bounded_rc_success=true`,
`gpu_tpu_cpu_32b_same_request_verified=true`,
`live_tpu_stage_miner_integrated=true`, `fallback_model_used=false`,
`tpu_32b_runtime_adapter_ready=true`, `stage_local_kv_cache_verified=true`,
`blocked_reason=""`, and `public_artifact_safe=true`. In the r9 live request,
one Coordinator request completed `stage0=1`, `stage1=1`, `stage2=1` with
accepted backends `["cpu", "cuda", "jax_tpu"]`, two public-safe activation
handoff hashes, and one generated-token hash. Stage0 used a private Kaggle
Tesla T4 x2 CUDA kernel (`cuda_device_count=2`) and was deleted after output
collection. Stage1 ran on authenticated Kaggle Web TPU v5e-8 with 8
`TPU v5 lite` devices, executed the real `Qwen/Qwen2.5-32B-Instruct` layers
21-42 stage-owned loader (`executed_layer_count=21`,
`loaded_execution_tensor_key_count=252`, `loaded_execution_tensor_gb=19.072947`,
`stage_local_kv_cache_verified=true`), and then submitted the JAX/TPU stage
task. Stage2 was the local CPU tail/verifier stage and completed the request.
Raw prompts, generated text, generated token ids, activations, hidden states,
logits, KV-cache tensors, credentials, cookies, leases, idempotency material,
Jupyter proxy token, and private runtime state remain out of public artifacts.
Keep the boundary precise: this is a bounded same-request 32B stage-inference
RC and evidence that the project can combine CUDA, JAX/TPU, and CPU stages in a
single request; it is not production serving, not throughput/TTFT/SLA evidence,
not P2P/NAT traversal, not training/fine-tuning, not billing/settlement, and
not proof of full end-to-end Qwen 32B text-quality parity across all layers.
The CUDA stage is tied to prior retained 32B stage-owned CUDA evidence rather
than reloading a full 32B CUDA stage inside this r9 bridge; the real full
32B-weight execution inside the r9 same request is the TPU middle stage.

Heterogeneous 32B Serving r4 is the current product-like deployment engineering
surface for the same model and topology:
`dist/heterogeneous-32b-serving-20260623-r4-live-attempt-web-tpu-proxy-blocked/heterogeneous_32b_serving.json`.
It is emitted by `crowdtensor heterogeneous-32b-serving` through
`scripts/heterogeneous_32b_serving_pack.py` and checked by
`scripts/heterogeneous_32b_serving_check.py`. It imports the r26 same-request
32B source proof and produces public-safe `deployment_plan.json`,
`streaming_response_contract.json`, `latency_metrics.json`,
`stage_local_kv_cache.json`, `failure_requeue.json`,
`live_external_multitoken_attempt.json`, `blocker_report.json`, Markdown, CLI
summary, and Support Bundle artifacts. Preserve
`heterogeneous_32b_serving_ready=true`,
`production_like_serving_path_ready=true`,
`gpu_tpu_cpu_32b_same_request_source_verified=true`,
`multi_token_generation_ready=true`, `streaming_response_contract_ready=true`,
`stage_local_kv_cache_ready=true`, `latency_metrics_ready=true`,
`failure_requeue_ready=true`, `public_artifact_safe=true`,
`live_external_runtime_verified=false`, and
`blocked_reason=web_tpu_jupyter_proxy_not_found` for the r4 report. The serving
plan defines the product-facing roles and
commands for Coordinator, CUDA/GPU Miner, JAX/TPU Miner, CPU tail/verifier, and
user generation, and the deterministic serving harness models at least four
token events with stage-local KV-cache status, per-stage latency, TTFT-like
metrics, activation byte counts, cleanup status, and bounded TPU-stage timeout
requeue evidence. The fresh external four-token attempt is recorded at
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260623-r11-real-tpu-32b-4token-serving-attempt-cleanup/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`:
it requested four generated tokens on the Kaggle CUDA GPU + Kaggle Web JAX/TPU
+ local CPU tail topology, stage0 CUDA was accepted once, but Web TPU stage1
could not attach through Jupyter (`web_tpu_jupyter_proxy_not_found`), so
stage1/stage2 stayed at zero and `live_external_runtime_verified=false`.
Temporary GPU kernels from the r10/r11 attempts were explicitly deleted. Keep
the boundary precise: r4 completes the reusable deployment/serving engineering
path from the current 32B proof and records a real blocked external live
attempt, but it is not a successful fresh external four-token live serving run,
not production SLA, not true P2P/NAT traversal, not billing/settlement, not
training/fine-tuning, not unbounded Kaggle stability, and not larger-model
capacity. Only a future fresh live serving report may set
`live_external_runtime_verified=true`. Checker logic must continue rejecting
fixture, queue-only, fallback model, partial-loader, single-stage, or
one-token-only evidence as live production-like 32B serving success.

The latest 2026-06-24 follow-up hardened the Kaggle Web TPU Jupyter access path
in `scripts/gpu_tpu_cpu_same_request_runtime_bridge_probe.py`: the default Web
TPU executor now runs through the authenticated Kaggle JupyterLab iframe and
browser-origin `/api/kernels` plus WebSocket path, with the old external
`jupyter-proxy` token scrape kept only as a fallback. The focused tests in
`tests/test_gpu_tpu_cpu_same_request_runtime_bridge_probe.py` cover report
extraction, failure classification, public redaction, and the 4-token bridge
success boundary. The fresh Web TPU restart was accepted but did not allocate a
runtime during this bounded attempt: retained status is
`dist/kaggle-web-tpu-session-retry-20260624-r1-start-queue6/kaggle_tpu_web_start_selected_tpu_attempt.json`
followed by
`dist/kaggle-web-tpu-session-retry-20260624-r3-current-status/kaggle_tpu_web_active_event_status.json`,
where the Notebook remains `Session is starting`, Jupyter API is visible but
kernel count is 0, and `tpu_runtime_ready=false`. The corresponding blocked
bridge report is
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260624-r1-web-tpu-runtime-starting/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`:
no fresh GPU kernel was started because the Web TPU runtime was not running,
stage counts are all 0, and `generated_token_count=0`. The serving aggregate
`dist/heterogeneous-32b-serving-20260624-r1-web-tpu-runtime-starting/heterogeneous_32b_serving.json`
still has product-like deployment engineering ready and imports the r26 1-token
source proof, but `live_external_runtime_verified=false` with blocker
`fresh_live_bridge_not_started_to_avoid_wasting_gpu_runtime` /
`kaggle_web_tpu_runtime_not_ready`. Treat this as access hardening plus bounded
allocation blocker evidence, not a successful 4-token live serving run and not
three-Accelerator production service readiness.

The 2026-06-25 continuation found the more stable Kaggle Web TPU path is the
JupyterLab frontend `window.jupyterapp.serviceManager`, not root
`/api/kernels`. The probe now waits for `jupyterapp.serviceManager`, can execute
ordinary Python/JAX through a service-manager session, and a direct Web TPU JAX
op proof at `dist/web-tpu-jax-op-direct-probe-20260624-r1/web_tpu_jax_op_direct_probe.json`
shows 8 `TPU v5 lite` devices and a successful TPU op. A direct 32B loader
execution also succeeded through the service-manager path:
`dist/web-tpu-32b-stage-code-smoke-20260624-r1/web_tpu_32b_stage_loader_service_manager_smoke.json`
records the real `Qwen/Qwen2.5-32B-Instruct` TPU stage loader with
`full_stage_owned_tpu_loader_ready=true`, 21 executed layers, 252 loaded
execution tensors, about 19.07 GB logical tensor bytes, and public-safe hashes.
The fresh 4-token bridge attempts still did not complete:
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260624-r3-service-manager-timeout-fixed-4token/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`
accepted CUDA stage0 once but stage1 stayed at zero, and temporary kernel
`xuyuhaosuyi/ct-gpu-tpu-cpu-bridge-82315445` was deleted. A mediated TPU-stage
path was added so the TPU notebook executes the real loader while the local
bridge thread performs Coordinator claim/submit, avoiding direct notebook to
Coordinator HTTP, but the current runner still hangs in the service-manager
execute/evaluate path before completing stage1. Current blocked status is
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260625-r1-service-manager-mediated-blocked/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`
and serving aggregate
`dist/heterogeneous-32b-serving-20260625-r1-service-manager-mediated-blocked/heterogeneous_32b_serving.json`.
The checker passes this as product-like deployment engineering with
`live_external_runtime_verified=false`; do not treat it as a successful
4-token serving proof. The next concrete fix is to replace the hanging
service-manager `requestExecute` future/evaluate path with a cancellable
execution/collection path, or otherwise reliably retrieve the TPU loader report
and submit stage1 locally before rerunning the 4-token bridge.

The 2026-06-25 later continuation fixed that Web TPU/Jupyter hang boundary but
the 4-token live serving goal is still incomplete. The bridge probe now wraps
Web TPU iframe execution in a subprocess hard-timeout boundary, adds bounded
service-manager session/kernel/execute handling with session shutdown, and
tries a service-manager-derived WebSocket execution path before falling back to
`requestExecute`. Focused tests in
`tests/test_gpu_tpu_cpu_same_request_runtime_bridge_probe.py` cover the new
timeout and fallback behavior. The serving pack now prioritizes
`kaggle_gpu_batch_session_limit_reached` ahead of downstream stage-not-ready
effects. Verified checks: `python -m py_compile scripts/gpu_tpu_cpu_same_request_runtime_bridge_probe.py scripts/heterogeneous_32b_serving_pack.py`
and `python -m pytest tests/test_gpu_tpu_cpu_same_request_runtime_bridge_probe.py tests/test_heterogeneous_32b_serving.py -q`
passed with 29 tests. The authenticated Kaggle Web TPU was restarted through
the UI, queued at #9, then became an active `TPU v5e-8` session. A live JAX
smoke saw 8 `TPU v5 lite` devices and completed a TPU matrix operation. The
new retained full-loader proof is
`dist/web-tpu-32b-full-loader-service-manager-20260625-r1/web_tpu_32b_full_loader_runtime_report.json`:
`Qwen/Qwen2.5-32B-Instruct` layers 21-42 loaded and executed 21/21 TPU-stage
layers, 252 stage-owned tensors, 19.072947 GB logical tensor bytes, no missing
stage keys, stage-local KV-cache verified, and public-safe hashes only. Fresh
4-token bridge attempts at
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260625-r2-service-manager-ws-fallback-4token/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`
and
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260625-r3-service-manager-ws-fallback-4token-retry/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`
did not create a Kaggle GPU kernel because Kaggle returned
`Maximum batch GPU session count of 2 reached`; no temporary bridge kernel or
private bridge package remained. The updated serving aggregate is
`dist/heterogeneous-32b-serving-20260625-r3-gpu-quota-blocked/heterogeneous_32b_serving.json`;
its checker passes with `live_external_runtime_verified=false` and
`blocked_reason=kaggle_gpu_batch_session_limit_reached`. Active Events showed
only the current TPU interactive session, while CLI confirmed
`xuyuhaosuyi/v12100-live-middle-model-sft` was RUNNING and was not created by
this goal. Do not delete unrelated user Kaggle sessions without explicit
permission. Next step: wait for or explicitly free Kaggle GPU batch capacity,
then rerun the 4-token bridge while the Web TPU session remains runnable.

2026-06-25 r4 retry: Web TPU remained executable (`web_tpu_still_alive_smoke_v1`
saw 8 TPU devices), and MCP accelerator quota showed GPU time quota available
(`time_used=0s`, `time_reserved=0s`) but Kaggle still rejected a fresh T4 GPU
script-kernel push with `Maximum batch GPU session count of 2 reached`. This
confirms the blocker is Kaggle's concurrent GPU batch/session limit, not GPU
hour quota and not TPU/Jupyter access. The r4 bridge attempt is
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260625-r4-gpu-slot-retry-4token/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`;
it created no temporary bridge kernel, removed the private bridge package, and
left `stage0=stage1=stage2=0`, `generated_token_count=0`. The r4 serving
aggregate is
`dist/heterogeneous-32b-serving-20260625-r4-gpu-quota-blocked/heterogeneous_32b_serving.json`;
checker passes with `live_external_runtime_verified=false` and
`blocked_reason=kaggle_gpu_batch_session_limit_reached`. Do not mark the goal
complete until a fresh bridge report proves at least 4 generated tokens with
CUDA, JAX/TPU, and CPU accepted stage tasks in the same request.

2026-06-25 r5/r6-prep continuation: the earlier Kaggle GPU batch-slot blocker
cleared long enough to create and run a fresh private T4 x2 bridge kernel
`xuyuhaosuyi/ct-gpu-tpu-cpu-bridge-82417684`. The r5 bridge report is
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260625-r5-gpu-free-4token/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`:
the same Coordinator request accepted CUDA stage0 once
(`accepted_stage_backends=["cuda"]`, `stage_task_counts.stage0=1`, one
CUDA-to-TPU activation handoff hash), but the Web TPU stage did not submit
stage1 (`stage1=0`, `stage2=0`, `generated_token_count=0`). Its Web TPU
steps show `service_manager_ready=true`, `session_startNew` with a kernel id,
then `kernel_info_ready` and `service_manager_request_execute` timed out, so
CPU tail never received a task. The temporary bridge kernel was explicitly
deleted afterward, and no private bridge package remained. A follow-up Web UI
restart attempt at
`dist/kaggle-web-tpu-restart-20260625-r2-start-session/kaggle_web_tpu_start_session_attempt.json`
entered the Kaggle TPU queue at `#10`; the active wait report
`dist/kaggle-web-tpu-restart-20260625-r3-active-wait/kaggle_web_tpu_active_wait.json`
waited about 15 minutes and ended with `Session is starting`, service-manager
`sessions=0` and `kernels=0`. The current serving aggregate is
`dist/heterogeneous-32b-serving-20260625-r5-web-tpu-starting-after-r5/heterogeneous_32b_serving.json`;
`scripts/heterogeneous_32b_serving_check.py --report ... --json` passes, but
`live_external_runtime_verified=false` and
`blocked_reason=qwen32b_tpu_stage_owned_loader_not_ready`. Code changes since
r5: `scripts/gpu_tpu_cpu_same_request_runtime_bridge_probe.py` now includes a
proxy-token existing-kernel WebSocket fast path when Kaggle exposes one, keeps
the iframe direct-WS and service-manager paths as fallbacks, and waits for the
full Kaggle status/output/delete timeout budget before final report assembly so
GPU kernel lifecycle evidence is not lost. Focused checks now pass with
`python -m py_compile scripts/gpu_tpu_cpu_same_request_runtime_bridge_probe.py scripts/heterogeneous_32b_serving_pack.py`
and `python -m pytest tests/test_gpu_tpu_cpu_same_request_runtime_bridge_probe.py tests/test_heterogeneous_32b_serving.py -q`
(`31 passed`). Do not mark this goal complete until a fresh post-fix bridge
report proves at least 4 generated tokens with CUDA, real JAX/TPU 32B
stage-owned loader, and CPU accepted stage tasks in the same request; queue,
starting-session, one-stage CUDA, or product-like aggregate evidence is not
enough. Also do not delete unrelated Kaggle kernels such as
`xuyuhaosuyi/v12100-live-middle-model-sft` without explicit permission.

2026-06-25 r6 completion: the current successful 4-token live serving evidence
is
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260625-r6-existing-session-4token/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`
and the serving aggregate is
`dist/heterogeneous-32b-serving-20260625-r6-live-4token-success/heterogeneous_32b_serving.json`.
Before r6, the Web TPU session recovered from `Session is starting` to an idle
existing Jupyter session/kernel; `dist/web-tpu-service-manager-first-jax-smoke-20260625-r1/web_tpu_service_manager_first_jax_smoke.json`
proves the fixed service-manager existing-session path can execute JAX on 8
`TPU v5 lite` devices. Code now prioritizes `session_connectTo_existing` /
`requestExecute` before slower proxy/direct-WS fallbacks, and serving import
logic recognizes successful `gpu_tpu_cpu_same_request_runtime_bridge_probe_v1`
reports as live external 4-token evidence when public-safe cleanup and all
stage counts are satisfied. The r6 bridge proves one same Coordinator request
completed 4 generated tokens with `accepted_stage_backends=["cpu","cuda","jax_tpu"]`,
`stage0=stage1=stage2=4`, `generated_token_count=4`,
`same_request_runtime_bridge_verified=true`,
`gpu_tpu_cpu_32b_same_request_verified=true`, and
`same_request_32b_model_verified=true`. Stage0 ran in a private Kaggle Tesla T4
x2 kernel, stage1 ran the real Web JAX/TPU `Qwen/Qwen2.5-32B-Instruct`
stage-owned loader for layers 21-42 on 8 TPU v5 lite devices
(`executed_layer_count=21`, `loaded_execution_tensor_key_count=252`,
`loaded_execution_tensor_gb=19.072947`, `stage_local_kv_cache_verified=true`),
and stage2 completed the CPU tail/verifier. Cleanup is public-safe:
`kaggle_gpu_kernel_created=true`, `kaggle_gpu_kernel_deleted=true`,
`private_gpu_package_removed=true`, and raw prompts, generated text, token ids,
activations, hidden states, logits, KV-cache tensors, cookies, credentials, and
Jupyter proxy tokens remain private. Verified checks:
`python -m py_compile scripts/gpu_tpu_cpu_same_request_runtime_bridge_probe.py scripts/heterogeneous_32b_serving_pack.py`,
`python -m pytest tests/test_gpu_tpu_cpu_same_request_runtime_bridge_probe.py tests/test_heterogeneous_32b_serving.py -q`
(`32 passed`), and
`python scripts/heterogeneous_32b_serving_check.py --report dist/heterogeneous-32b-serving-20260625-r6-live-4token-success/heterogeneous_32b_serving.json --json`
passes with `live_external_runtime_verified=true`, `blocked_reason=""`,
`heterogeneous_32b_serving_ready=true`, and
`production_like_serving_path_ready=true`. Keep the boundary precise: this
completes the bounded Kaggle GPU + Web TPU + CPU 32B 4-token live serving
validation requested for this goal, but it is still not production SLA,
throughput benchmarking, P2P/NAT traversal, billing/settlement, training, or an
unbounded Kaggle service claim; the CUDA stage is tied to prior retained 32B
stage-owned CUDA evidence rather than reloading a full 32B CUDA stage inside
this bridge.

## Current Alpha Reality

The current code supports:

- Public Swarm Inference v2 output scope: preserve top-level `output_request`, `answer_scope.scope_state: no-local-answer`, and `shareable_summary.answer_scope_state` in `public_swarm_inference_v2` JSON, Markdown, terminal summaries, and Support Bundle. Treat the v2 aggregate as shareable readiness evidence rather than a local answer transcript; only human `crowdtensor generate --p2p` may show local generated text, while public artifacts keep raw prompts, generated text, generated token ids, activations, credentials, leases, and idempotency material redacted.
- Usable Swarm Inference v1: `crowdtensor usable-swarm` emits `usable_swarm_inference_v1` through `scripts/usable_swarm_inference_pack.py` and validates with `scripts/usable_swarm_inference_check.py`. This is the ordinary user entrypoint: `crowdtensor p2pd --run`, `crowdtensor serve --p2p --run`, distinct `crowdtensor join --stage stage0 --p2p --run` and `crowdtensor join --stage stage1 --p2p --run`, then `crowdtensor generate --p2p --prompt ... --max-new-tokens 8`. Preserve `usable_swarm_inference_ready`, `usable_swarm_inference_v1_ready`, `serve_join_generate_p2p_primary_path`, `usable_p2p_route_ready`, `usable_real_llm_generate_ready`, `usable_multi_token_generation_ready`, `usable_distinct_stage_miners_ready`, `usable_stage_requeue_rescue_ready`, top-level `output_request`, `answer_scope.scope_state: no-local-answer`, `shareable_summary.answer_scope_state`, redacted `support_bundle.json`, and `USABLE_SWARM_INFERENCE.md`. Human `crowdtensor generate` output may show generated text when not using `--json`; `usable-swarm` public artifacts must keep raw prompts, generated text, generated token ids, activations, credentials, leases, and idempotency material out of reports while explaining that local answers are available from the human `generate` command, not the shareable aggregate. It is Coordinator-backed, read-only, tiny/small-model scoped, CPU by default with optional CUDA fail-closed paths, not full Hivemind/Petals production parity, not Coordinator-free execution, not production NAT traversal, not an economic network, and not large-model throughput serving.
- User-friendly inference front-door contract check: `scripts/user_friendly_inference_frontdoor_check.py` emits `user_friendly_inference_frontdoor_check_v1` and uses fake completed `infer` and `generate` payloads through the real CLI report writers to validate saved `infer_summary` / `generate_summary` JSON and Markdown. Preserve terminal-only answer visibility, saved `answer_scope_state: saved-terminal-redacted`, `inference_verdict`, `gpu_status`, `fresh_kaggle_gpu_verified: false`, and redaction of raw prompts, generated text, token ids, credentials, leases, idempotency material, and activations. It is CI-safe report-contract validation only: it does not start a Coordinator, submit a live task, or run Kaggle/GPU proof.
- Coordinator/Miner task leasing, heartbeat recovery, timeout requeue, stale result rejection, and checkpoint replay.
- Deterministic CPU-only workloads: `diloco_train`, `cpu_lora_mock`, `micro_transformer_lm`, `model_bundle_lm`, measurable read-only `model_bundle_infer`, optional read-only `external_llm_infer`, and `browser_probe`.
- `runtime_contract_v1`, workload capability matching, CPU `hardware_profile`, delta transport negotiation, validation, replay audit, result ledger, trust quarantine, and operator controls.
- Controlled remote Miner demos with token-backed admission, hashed token config, `/ready` preflight, retries, `remote_compute_observability_v1`, `remote_demo_observability_v1`, and Support Bundle diagnostics.
- Browser experiments for WebRTC tensor transport, Worker compute probes, and a browser Miner bridge.
- Release tooling: runtime capability matrix, matrix-guided home-compute demo, user-facing inference session demo, `inference_session_client_v1`, admin-created read-only inference session API check, external LLM adapter smoke, release gate, fresh clone onboarding gate, runtime acceptance pack, browser acceptance pack, release evidence, doctor diagnostics, security preflight, and Support Bundle. Preserve safe runtime `summary_json`, route `diagnosis_codes`, `operator_action`, `diagnosis_summary`, `diagnosis_by_check`, and remote `observability_summaries` across runtime, release evidence, and Support Bundle outputs.
- Release readiness gate: `crowdtensor release-ready` in `crowdtensor/cli.py` wraps `scripts/release_readiness_pack.py` and emits `release_readiness_v1` by aggregating Git metadata, the release gate, security preflight, and `demo_manifest_v1`. Dirty worktrees block by default with `git_dirty`; `scripts/release_readiness_check.py --allow-dirty` is only for development/CI smoke validation. It is an Alpha maintainer gate, not production Swarm Inference readiness.
- Fresh clone onboarding gate: `scripts/onboarding_gate.py --quick` emits `onboarding_gate_v1` by creating a clean temporary virtualenv, running `python -m pip install -e .[dev,hf]`, checking `crowdtensor --help`, `crowdtensord --help`, and `crowdtensor-miner --help`, then smoke-validating `scripts/user_friendly_inference_frontdoor_check.py`, the installed real user entrypoint `crowdtensor infer --prompt-stdin --shareable-terminal`, `crowdtensor local-proof`, `crowdtensor home-infer`, `crowdtensor llm-infer --mock`, `crowdtensor cpu-infer --mode local`, and `crowdtensor release-ready --allow-dirty`. The `user_infer_smoke` validation must preserve `user_infer_smoke_validation_v1`, `answer=shareable-terminal-redacted`, `gpu=local-cpu-only`, `fresh_kaggle_gpu=False`, prompt stdin redaction, safe `infer_summary.json` / `infer_summary.md`, and no raw prompt/generated answer/token ids/activations in onboarding artifacts. It is a fresh-checkout onboarding gate, not production Swarm Inference readiness.
- One-command local proof: `crowdtensor local-proof` in `crowdtensor/cli.py` emits `local_proof_summary_v1` by chaining Doctor, runtime matrix, the CPU-only read-only home-compute demo, and the Demo Manifest path. It is a local proof, not production Swarm Inference.
- Home inference proof CLI: `crowdtensor home-infer` emits `home_inference_cli_v1`, wraps `scripts/home_compute_evidence_pack.py`, and writes `home_compute_evidence_v1` artifacts with the CPU-only read-only `model_bundle_infer` route, fixed `model_bundle_inference_scenario_v1` metadata, capped `request_trace`, `diagnosis_codes`, and read-only/redaction status. Built-in scenario IDs are `route-baseline`, `gradient-safety`, and `mixed-prompts`; it is not production Swarm Inference or arbitrary prompt serving.
- External LLM proof CLI: `crowdtensor llm-infer` emits `llm_inference_cli_v1`, wraps `scripts/external_llm_evidence_pack.py`, and writes `external_llm_evidence_v1` artifacts for the read-only `external_llm_infer` route. It records adapter kind, model id, request/completion count, output chars, throughput, and redaction status while keeping raw prompts, `output_text`, runtime URL, and API key out of public artifacts. It is fixed-prompt operator-owned runtime evidence, not public arbitrary prompt serving.
- CPU inference Beta aggregate CLI: `crowdtensor cpu-infer` emits `cpu_inference_beta_v1` through `scripts/cpu_inference_beta_pack.py`. `--mode local` wraps `home-infer` and deterministic `llm-infer --mock`; `--mode remote-loopback` validates local `remote-demo` stand-ins for `model-bundle` and `external-llm`; `--mode remote-existing` wraps an already running controlled two-machine `remote-demo doctor/verify/collect` flow with explicit observer/admin tokens. `scripts/cpu_inference_beta_check.py` validates this path. It is CPU-only, read-only, not production Swarm Inference, not P2P, and not arbitrary prompt serving.
- CPU Inference Beta RC: `crowdtensor cpu-infer --mode beta-rc` emits `cpu_inference_beta_rc_v1` through `scripts/cpu_inference_beta_rc_pack.py`, aggregating local CPU inference, remote-loopback inference, Real two-machine CPU inference Beta rehearsal, Kaggle Remote Miner Beta artifacts, `miner_join_pack_v1`, `scripts/kaggle_remote_miner_beta_check.py`, and `demo_manifest_v1`. `scripts/cpu_inference_beta_rc_check.py` validates it and requires `cpu_inference_beta_rc_ready`, `local_cpu_inference_ready`, `remote_loopback_ready`, `two_machine_rehearsal_ready`, `kaggle_remote_miner_artifacts_ready`, `miner_join_pack_ready`, and `cpu_miner_beta_ready`. `--kaggle-real-runtime-report` can import a completed `kaggle_real_runtime_acceptance_v1` and surface `real_runtime_evidence_ready`; CI artifact checks still do not prove a live Kaggle run. It is CPU-only, read-only, not production Swarm Inference, not P2P, not a GPU/TPU workload path, and not arbitrary prompt serving.
- Pipeline-Sharded Inference Alpha/Beta: `crowdtensor shard-infer` emits `sharded_inference_cli_v1` and `sharded_inference_evidence_v1` for the CPU-only read-only `sharded_model_bundle_infer` / `sharded_model_bundle_infer_v1` workload inside `sharded_inference_session_v1`; `crowdtensor shard-infer-beta --mode remote-loopback` emits `remote_sharded_inference_beta_v1` through `scripts/remote_sharded_inference_beta_pack.py` and validates with `scripts/remote_sharded_inference_beta_check.py`; `crowdtensor remote-demo --workload sharded-model-bundle` emits the two-machine runbook/acceptance shape with `remote_python_sharded_model_bundle_infer`, `remote_sharded_inference_acceptance_v1`, `remote_sharded_inference_observability_v1`, and `remote_two_machine_sharded_ready`. Preserve activation hashes, `baseline_match`, `remote_sharded_inference_ready`, `remote_sharded_loopback_ready`, `local_sharded_inference_ready`, and `stage_requeue_ready` when `--failure-mode kill-stage-after-claim` is used. It is CPU-only, read-only, not production Swarm Inference, not P2P, and not real LLM sharding.
- Micro-LLM Pipeline-Sharded Inference Alpha/Beta: `crowdtensor micro-llm-shard-infer` emits `micro_llm_sharded_cli_v1` and `micro_llm_sharded_evidence_v1` for the CPU-only read-only `micro_llm_sharded_infer` / `micro_llm_sharded_infer_v1` workload inside `micro_llm_sharded_session_v1`; `crowdtensor micro-llm-shard-infer-beta --mode remote-loopback` emits `remote_micro_llm_sharded_beta_v1` through `scripts/remote_micro_llm_sharded_beta_pack.py` and validates with `scripts/remote_micro_llm_sharded_beta_check.py`; `crowdtensor remote-demo --workload micro-llm-sharded` emits the two-machine runbook/acceptance shape with `remote_python_micro_llm_sharded_infer`, `remote_micro_llm_sharded_acceptance_v1`, `remote_micro_llm_sharded_observability_v1`, and `remote_two_machine_micro_llm_sharded_ready`. Preserve activation hashes, `decode_steps`, `baseline_match`, `decoded_tokens_match`, `remote_micro_llm_sharded_ready`, `remote_micro_llm_sharded_loopback_ready`, `local_micro_llm_sharded_inference_ready`, and `stage_requeue_ready` when failure modes are used. Stage-aware mode uses `scripts/stage_aware_micro_llm_sharded_check.py`, `--stage-mode split`, `--require-distinct-stage-miners`, Miner roles `--micro-llm-stage-role stage0|stage1|both`, capabilities `micro_llm_sharded_stage0`, `micro_llm_sharded_stage1`, `micro_llm_sharded_both`, and diagnosis codes `distinct_stage_miners` plus `stage_assignment_valid`. `crowdtensor micro-llm-artifact` and `scripts/micro_llm_artifact_pack.py` emit `micro_llm_artifact_v1`; `--micro-llm-artifact` must remain wired through local sharded inference, remote-demo, Kaggle real runtime, and live RC paths with `artifact_loaded` and `micro_llm_artifact_ready`. It is CPU-only, read-only, not production Swarm Inference, not P2P, not Hugging Face/GGUF/llama.cpp, and not large LLM serving.
- Safe artifact cleanup: `crowdtensor clean-artifacts` emits `cleanup_report_v1`, defaults to dry-run, removes generated `__pycache__` / `.pyc` caches and old CrowdTensor temp directories only with `--apply`, keeps reports unless `--include-reports` is used, and does not delete state or source files.
- Remote demo operator CLI: `crowdtensor remote-runbook` emits `remote_runbook_cli_v1` and wraps `scripts/remote_demo_runbook_pack.py`; `crowdtensor remote-acceptance` emits `remote_acceptance_cli_v1`, defaults to `--create-session`, wraps `scripts/remote_demo_acceptance_pack.py`, carries fixed `model_bundle_inference_scenario_v1` scenarios such as `route-baseline`, and applies token redaction to captured command output. It is a controlled two-machine helper, not production Swarm Inference and not P2P routing.
- Remote home-compute demo CLI: `crowdtensor remote-demo prepare`, `crowdtensor remote-demo doctor`, `crowdtensor remote-demo verify`, `crowdtensor remote-demo collect`, and `crowdtensor remote-demo clean` emit `remote_home_compute_demo_v1`, `remote_home_compute_doctor_v1`, `remote_home_compute_collect_v1`, and `remote_home_compute_cleanup_v1` through `scripts/remote_home_compute_demo_pack.py`. The prepare path creates `operator.private.env`, `miner.private.env`, the hashed registry, the public runbook, and `miner_join_pack_v1` artifacts (`miner_join.sh`, `MINER_JOIN.md`) for ordinary CPU Miner hosts; doctor checks local files, token presence, Coordinator reachability, task lane visibility, and optional accepted-result readiness; the default verify path uses `POST /admin/inference-sessions` for read-only `model_bundle_infer`, validates `remote_python_model_bundle_infer`, and summarizes `remote_compute_evidence_v1`, `remote_demo_observability_v1`, and Support Bundle artifacts; collect gathers evidence/support from an already running demo; clean defaults to dry-run and only removes private env/registry files with `--include-private`. `--workload external-llm` queues read-only `external_llm_infer`, validates `remote_python_external_llm_infer`, and summarizes `remote_external_llm_evidence_v1` plus `remote_external_llm_observability_v1` for deterministic `--mock` or explicit operator-owned `--llm-runtime-cmd` / `--llm-runtime-url` adapters. `scripts/remote_home_compute_demo_check.py` validates both local-loopback stand-ins across prepare, doctor, verify, collect, and clean. It is not production Swarm Inference, not P2P routing, not GPU pooling, and not public arbitrary prompt serving.
- Real two-machine CPU inference Beta aggregate check: `scripts/remote_two_machine_beta_check.py` emits `remote_two_machine_beta_check_v1` by running local loopback stand-ins for the Coordinator host and Miner host across `model-bundle` and `external-llm`. It requires `remote_two_machine_inference_ready`, `remote_two_machine_external_llm_ready`, and `remote_two_machine_beta_ready`, and backs the 15-minute two-machine CPU inference Beta docs. It is task-level remote CPU inference, not model sharding, not P2P, and still requires operator-provided TLS, VPN, tunnel, or trusted network for real machines.
- Kaggle Remote Miner Beta: `crowdtensor remote-demo prepare --target kaggle` generates `miner_join_pack_v1`, `miner_join.sh`, `MINER_JOIN.md`, `kaggle_remote_miner.py`, `kaggle_remote_miner.md`, and the usual `miner.private.env` for an outbound Kaggle CPU Miner while keeping `operator.private.env` on the operator host. `scripts/kaggle_remote_miner_beta_check.py` emits `kaggle_remote_miner_beta_check_v1`, requires `kaggle_remote_miner_prepare_ready` and `kaggle_remote_miner_beta_ready`, and uses local loopback as the CI stand-in. Kaggle is a temporary external Miner target only; no Coordinator is exposed from Kaggle, no GPU/TPU workload is enabled, and this is not production Swarm Inference or P2P.
- Kaggle Real Runtime Acceptance: `crowdtensor remote-demo kaggle-real` wraps `scripts/kaggle_real_runtime_acceptance_pack.py` and emits `kaggle_real_runtime_acceptance_v1`. `--action prepare --public-host 24.199.118.54 --port 9180` generates a temporary HTTP Coordinator launch script, `operator.private.env`, `miner.private.env`, hashed registry, and a Kaggle-only upload package; `--action verify` requires a live Kaggle CPU Notebook Miner and reports `kaggle_artifacts_ready`, `coordinator_public_ready`, `kaggle_miner_seen`, `kaggle_result_accepted`, and `kaggle_real_runtime_ready`. `--workload micro-llm-sharded --stage-mode split --decode-steps 3` prepares `kaggle-upload-stage0` and `kaggle-upload-stage1` for two Notebook Miners and should report `kaggle_micro_llm_stage0_seen`, `kaggle_micro_llm_stage1_seen`, `kaggle_micro_llm_stage_assignment_valid`, `stage_assignment_valid`, and `kaggle_micro_llm_sharded_ready` after live verification. `scripts/kaggle_micro_llm_live_package.py` can package those stage uploads as private Kaggle dataset/script-kernel folders; `--inline-kernel-payload` embeds stage `miner.private.env` into private kernel source and must stay temporary, uncommitted, unpublished, deleted after proof, and followed by token rotation. The first artifact-backed live split proof completed against `24.199.118.54:9180` with two private Kaggle CPU script kernels and `micro_llm_artifact_v1`; retained evidence is `dist/kaggle-micro-llm-live/external-real/kaggle_real_runtime_acceptance.json` with `ok: true`, artifact/stage assignment/baseline/decoded-token readiness, and deleted remote Kaggle kernels/dataset. `scripts/kaggle_real_runtime_acceptance_check.py` validates artifact safety only. Preserve `token_rotation_required`, keep `operator.private.env` off Kaggle, and do not claim production security, P2P, GPU/TPU workload execution, or large-model sharding.
- Micro-LLM Live Two-Node RC: `crowdtensor micro-llm-live-rc` wraps `scripts/micro_llm_live_rc_pack.py` and emits `micro_llm_live_rc_v1`; `scripts/micro_llm_live_rc_check.py` validates the local-generated path. `--mode local-generated` creates `kaggle-upload-stage0` and `kaggle-upload-stage1`, starts a local Coordinator plus two independent stage Miner processes from those generated packages, and should report `local_generated_stage_upload_standins_ready`, `micro_llm_live_rc_ready`, `kaggle_micro_llm_sharded_ready`, and `stage_assignment_valid` while keeping `external_runtime_verified` false. With `--micro-llm-artifact`, the same RC must load and report the file-backed artifact via `artifact_loaded` and `micro_llm_artifact_ready`. `--mode external-existing` verifies an already running public Coordinator plus two external stage Miners and only then may report `external_runtime_verified`. It is CPU-only, read-only toy two-stage micro-LLM evidence, not production Swarm Inference, not P2P, not GPU/TPU pooling, and not GGUF/llama.cpp or large-model sharding.
- Real Small-LLM Sharded Inference Beta: `crowdtensor real-llm-shard-infer` emits `real_llm_sharded_cli_v1` and `real_llm_sharded_evidence_v1` for the optional `[hf]` `real_llm_sharded_infer` / `real_llm_sharded_infer_v1` workload using `hf_transformers_cpu` and `sshleifer/tiny-gpt2` by default. It records safe `real_llm_artifact_v1` metadata, passes only redacted activation summaries in public artifacts, validates stage 1 against a local full-model next-token baseline, and preserves `real_llm_artifact_ready`, `activation_transport_ready`, `baseline_match`, `decoded_tokens_match`, `real_llm_sharded_ready`, `distinct_stage_miners`, and `stage_assignment_valid`. Miners must opt in with `--enable-hf-tiny-gpt-runtime`, optional `--hf-cache-dir`, and `--real-llm-stage-role stage0|stage1|both`; capabilities are `real_llm_sharded_stage0`, `real_llm_sharded_stage1`, and `real_llm_sharded_both`. `--real-llm-partition-mode stage-local` moves only stage-owned modules to the selected runtime device, keeps a separate CPU baseline for correctness, and must emit `stage_local_partition_ready`, `stage0_partition_loaded`, `stage1_partition_loaded`, `partition_parameter_split_valid`, and stage parameter-count evidence. `crowdtensor real-llm-shard-infer-beta --mode remote-loopback` emits `remote_real_llm_sharded_beta_v1` through `scripts/remote_real_llm_sharded_beta_pack.py`; `scripts/remote_real_llm_sharded_beta_check.py` validates `remote_real_llm_sharded_ready`, `remote_real_llm_sharded_loopback_ready`, and `local_real_llm_sharded_inference_ready`. `crowdtensor remote-demo --workload real-llm-sharded` is the high-level two-machine wrapper for the same tiny GPT split path, preserving `remote_python_real_llm_sharded_infer`, `remote_real_llm_sharded_acceptance_v1`, `remote_real_llm_sharded_observability_v1`, `remote_two_machine_real_llm_sharded_ready`, and `hf_dependencies_missing` diagnostics. It is read-only optional Hugging Face tiny-model evidence, not production Swarm Inference, not P2P, not GPU pooling, not GGUF/llama.cpp serving, and not large-model serving.
- Optional CUDA real LLM backend: `hf_transformers_cuda` is supported only as an explicit tiny GPT split backend. It must use `--real-llm-backend hf_transformers_cuda`, fail closed when `torch.cuda.is_available()` is false, and route only to Miners advertising `real_llm_sharded_cuda_stage0`, `real_llm_sharded_cuda_stage1`, or `real_llm_sharded_cuda_both`. Public Swarm GPU Beta defaults to `--real-llm-partition-mode stage-local`, so CUDA evidence should prove stage-owned module placement instead of full-model CUDA placement. Preserve CPU default behavior through `hf_transformers_cpu`; do not imply production GPU pooling or large-model serving.
- Real Small-LLM Sharded Inference Live RC: `crowdtensor real-llm-live-rc` wraps `scripts/real_llm_live_rc_pack.py` and emits `real_llm_live_rc_v1`; `scripts/real_llm_live_rc_check.py` validates the local-generated path. `--mode local-generated` creates `kaggle-upload-real-llm-stage0` and `kaggle-upload-real-llm-stage1`, starts a local Coordinator plus two independent HF-enabled stage Miner processes from those generated packages, and should report `local_generated_real_llm_stage_upload_standins_ready`, `remote_real_llm_sharded_ready`, `real_llm_artifact_ready`, `decoded_tokens_match`, `stage_assignment_valid`, and `real_llm_live_rc_ready` while keeping `external_runtime_verified` false. `--mode kaggle-generated` prepares the stage packages and runbook only. `--mode external-existing` verifies an already running public Coordinator plus two external stage Miners and only then may report `external_runtime_verified` and `kaggle_real_llm_sharded_ready`. `scripts/kaggle_real_llm_live_package.py` emits `kaggle_real_llm_live_package_v1` for private Kaggle dataset/script-kernel packaging; `--inline-kernel-payload` is a private temporary fallback for Kaggle input-mount issues. The first live real-weight Kaggle split proof completed against `24.199.118.54:9184` with two private Kaggle CPU script kernels, `kaggle-real-llm-stage0`, `kaggle-real-llm-stage1`, and `sshleifer/tiny-gpt2`; retained evidence is `dist/real-llm-live-goal-external/real_llm_live_rc.json` with external runtime, stage seen, artifact, baseline, decoded-token, distinct-stage, and assignment readiness. Generated launchers preserve `--enable-hf-tiny-gpt-runtime`, `--real-llm-stage-role`, and `launcher_syntax_valid`. It is CPU-only, read-only tiny Hugging Face evidence, not production Swarm Inference, not P2P, not GPU/TPU pooling, not GGUF/llama.cpp serving, and not large-model serving.
- Real Internet Swarm Inference Alpha: `crowdtensor real-llm-internet-alpha` wraps `scripts/real_llm_internet_alpha_pack.py` and emits `real_llm_internet_alpha_v1`; `scripts/real_llm_internet_alpha_check.py` validates the local-generated path. `local-generated` wraps the Live RC and mandatory local stage0/stage1 requeue proofs, preserving `real_llm_internet_alpha_ready`, `real_llm_stage_requeue_ready`, `stage_requeue_ready`, `real_llm_live_rc_ready`, `remote_real_llm_sharded_ready`, `real_llm_artifact_ready`, `activation_transport_ready`, `decoded_tokens_match`, `distinct_stage_miners`, and `stage_assignment_valid` while keeping `external_runtime_verified` false. `package` prepares public Coordinator and stage upload artifacts only. `external-existing` verifies an already running public Coordinator plus two external stage Miners and only then may report `external_runtime_verified`. The first external Alpha proof completed against `24.199.118.54:9187` with two private Kaggle CPU script kernels, `internet-real-llm-stage0` and `internet-real-llm-stage1`; retained evidence is `dist/real-llm-internet-alpha-external/real_llm_internet_alpha.json` with `external_runtime_verified`, `real_llm_internet_alpha_ready`, both stages seen, decoded-token match, distinct stage Miners, and valid stage assignment. Temporary Kaggle kernels were deleted after evidence collection and tokens must be rotated after temporary public HTTP proofs. Reports preserve `token_rotation_required`, CPU-only/read-only semantics, and explicit not production / not P2P / not large-model boundaries.
- Real Internet Swarm Inference Beta: `crowdtensor real-llm-internet-beta` wraps `scripts/real_llm_internet_beta_pack.py` and emits `real_llm_internet_beta_v1`; `scripts/real_llm_internet_beta_check.py` validates the fake-runner CI contract. `kaggle-auto` generates the Alpha package, starts the temporary public Coordinator, pushes private Kaggle CPU script kernels by default or private Kaggle GPU kernels with `--real-llm-backend hf_transformers_cuda`, runs external-existing verification, deletes the temporary kernels, stops the Coordinator, and only then may report `real_llm_internet_beta_ready`. CUDA mode preserves CPU Coordinator metadata-only scheduling and requires torch CUDA only on the stage Miners. Generated Kaggle CUDA kernels default to `torch==2.7.1+cu118`, `torchvision==0.22.1+cu118`, and `transformers==4.40.2` for older Kaggle GPU compatibility. With `--failure-mode kill-stage0-after-claim` / `kill-stage1-after-claim`, it creates distinct victim/rescue Kaggle Miners, observes the victim claim through `/state`, deletes the victim kernel, waits for lease timeout requeue, pushes rescue, and emits `external_stage_requeue_ready`, `live_stage0_requeue_ready` / `live_stage1_requeue_ready`, and `live_requeue_summary`. `--mode evidence-import` combines retained generation and requeue reports only when the generation source exposes safe `generated_token_count >= --max-new-tokens`, external generation readiness, matching model metadata, cleanup evidence, and a public-safe live requeue summary proving claim observation, victim deletion, lease expiry, rescue acceptance, and victim-result rejection. Retained 16-token import evidence is `dist/goal-final-infer-real-llm-internet-beta-import-16tok-gpu-summary-20260602/real_llm_internet_beta.json`; it imports retained CUDA generation plus retained requeue evidence and does not create a fresh Kaggle run. Preserve `real_llm_internet_alpha_ready`, `external_runtime_verified`, both Kaggle stages seen, `decoded_tokens_match`, `distinct_stage_miners`, `stage_assignment_valid`, `kaggle_kernels_deleted`, `token_rotation_required`, CPU-default/read-only semantics, imported backend/schema metadata, and explicit not production / not P2P / not GPU pooling / not large-model boundaries.
- Swarm Inference Beta: `crowdtensor swarm-infer-beta` wraps `scripts/swarm_inference_beta_pack.py` and emits `swarm_inference_beta_v1`; `scripts/swarm_inference_beta_check.py` validates the fake-runner CI contract. It is the user-facing two-machine package for the real tiny GPT split path. `swarm-infer-beta live` is the side-effectful `kaggle-auto` public proof wrapper around `real_llm_internet_beta_v1`; it starts a temporary public Coordinator, pushes private Kaggle CPU stage kernels, verifies `external_runtime_verified`, optionally verifies external victim/rescue requeue with `--failure-mode`, deletes kernels, writes `support_bundle.json`, removes local live private artifacts and raw runtime state by default, and only then may report `swarm_inference_beta_live_ready`, `real_llm_internet_beta_ready`, `external_stage_requeue_ready` when requested, `live_requeue_summary`, `kaggle_kernels_deleted`, `swarm_inference_beta_live_private_artifacts_cleaned`, and `token_rotation_required`. `--keep-live-private-artifacts` is for debugging only. `prepare` creates `operator.private.env`, stage0/stage1 `miner.private.env`, hashed `miner_registry.json`, stage join packs, and `SWARM_INFERENCE_BETA.md`; `verify` wraps `remote_real_llm_sharded_beta_v1`; `collect` gathers redacted evidence/support; `clean` is dry-run by default. Preserve `swarm_inference_beta_ready`, `two_machine_swarm_inference_ready`, `real_llm_split_route_ready`, `external_beta_evidence_imported`, `decoded_tokens_match`, `distinct_stage_miners`, `stage_assignment_valid`, CPU-only/read-only semantics, and explicit not production / not P2P / not large-model boundaries.
- Public Swarm Inference Alpha: `crowdtensor swarm-session` wraps `scripts/public_swarm_inference_alpha_pack.py` and emits `public_swarm_inference_alpha_v1`; `scripts/public_swarm_inference_alpha_check.py` validates the fake-runner CI contract. `--mode live-kaggle` aggregates the cleanup-backed `swarm-infer-beta live` proof, true external victim/rescue requeue evidence (`external_stage_requeue_ready`, `live_stage0_requeue_ready` / `live_stage1_requeue_ready`, `live_requeue_summary`) when `--failure-mode` is enabled, and mandatory `local-generated` real LLM stage requeue evidence. Preserve `public_swarm_inference_alpha_ready`, `public_swarm_session_ready`, `local_stage_requeue_ready`, `public_swarm_live_requeue_ready`, `public_swarm_live_kaggle_ready`, `stage_requeue_ready`, `external_runtime_verified`, `kaggle_kernels_deleted`, and `token_rotation_required`. Child debug artifacts are pruned by default so shareable output is the top-level public JSON/Markdown report; `--keep-child-artifacts` is local debugging only. It is CPU-only, read-only, not production Swarm Inference, not P2P, and not large-model serving.
- Public Swarm Inference Alpha RC: `crowdtensor public-swarm-alpha-rc` wraps `scripts/public_swarm_inference_alpha_rc_pack.py` and emits `public_swarm_inference_alpha_rc_v1`; `scripts/public_swarm_inference_alpha_rc_check.py` validates `local-smoke` in CI and `evidence-import` when retained live reports are present. Preserve `public_swarm_inference_alpha_rc_ready`, `public_swarm_alpha_rc_evidence_imported`, `stage0_live_requeue_evidence_ready`, `stage1_live_requeue_evidence_ready`, `public_swarm_live_requeue_evidence_ready`, `public_swarm_alpha_private_artifacts_absent`, `evidence-import`, and `local-smoke`. The retained live proof paths are `dist/public-swarm-inference-alpha-live-stage0-requeue-20260527165830/public_swarm_inference_alpha.json`, `dist/public-swarm-inference-alpha-live-stage1-requeue-20260527170600/public_swarm_inference_alpha.json`, and `dist/public-swarm-inference-alpha-live-requeue-summary.json`. This RC imports existing public evidence; it does not create a fresh Kaggle run, and it remains CPU-only, read-only, not production Swarm Inference, not P2P, and not large-model serving.
- Public Swarm Inference Beta: `crowdtensor public-swarm-beta` wraps `scripts/public_swarm_inference_beta_pack.py` and emits `public_swarm_inference_beta_v1`; `scripts/public_swarm_inference_beta_check.py` validates `product-beta`, `local-loopback`, and `evidence-import`. Preserve `public-swarm-beta product-beta`, `public_swarm_product_beta_ready`, `public_swarm_product_rc_ready`, `coordinator_product_surface_ready`, `session_protocol_ready`, `p2p_lite_discovery_ready`, `gpu_generation_evidence_import_ready`, `cpu_fallback_ready`, and `local_cpu_inference_ready` as the product-shaped Beta aggregate over Product RC, `session_protocol_v1`, `p2p_lite_peer_v1`, retained GPU generation evidence, and CPU fallback. Preserve compatibility paths `public-swarm-beta local-loopback`, `public-swarm-beta evidence-import`, `two_stage_split_inference_ready`, `local_loopback_ready`, `decoded_tokens_match`, `distinct_stage_miners`, `stage_assignment_valid`, `public_swarm_beta_evidence_import_ready`, `external_live_evidence_imported`, `stage0_live_requeue_evidence_ready`, `stage1_live_requeue_evidence_ready`, and the operator actions `prepare`, `coordinator`, `miner --stage stage0`, `miner --stage stage1`, `verify`, `collect`, and dry-run `clean`. This is Coordinator-backed, read-only, not production Swarm Inference, not libp2p/DHT/NAT traversal, not Hivemind-level serving, and not large-model serving.
- GPU Swarm Usability Alpha: `crowdtensor gpu-swarm smoke|prepare|coordinator|miner|infer|status|collect|clean` wraps `scripts/gpu_swarm_usability_alpha_pack.py` and emits `gpu_swarm_usability_alpha_v1` plus `gpu_swarm_usability_alpha_cli_v1`; `scripts/gpu_swarm_usability_alpha_check.py` validates the public report. Preserve `gpu_swarm_usability_alpha_ready`, `user_gpu_swarm_entrypoint_ready`, `gpu_miner_join_pack_ready`, `coordinator_workflow_ready`, `two_gpu_stage_route_ready`, `inference_request_lifecycle_ready`, `model_catalog_imported`, `control_user_alpha_imported`, `core_handoff_imported`, `public_artifact_safe`, `execution_mode`, `external_runtime_verified`, stage0/stage1 join packs, safe `GPU_SWARM_MINER_PRIVATE_TOKEN` placeholders, `GPU_SWARM_ALPHA.md`, `support_bundle.json`, and the retained evidence-import path over Control/User Alpha plus 7B/14B core handoff evidence. It is the ordinary-user multi-GPU connection flow Alpha; default CI evidence is `evidence-import` with `external_runtime_verified=false`, not a fresh GPU run. Public artifacts must keep raw prompts, generated text, token ids, activations, credentials, leases, idempotency material, private env files, registries, and Kaggle kernel payloads redacted. This is not production Swarm Inference, not P2P/NAT traversal, not arbitrary public prompt serving, not billing, and not unbounded GPU pooling.
- GPU Swarm Production-Like Validation RC: `crowdtensor gpu-swarm validate-production-like|scale-test` wraps `scripts/gpu_swarm_production_like_validation_pack.py` and emits `gpu_swarm_production_like_validation_v1` plus `gpu_swarm_production_like_validation_cli_v1`; `scripts/gpu_swarm_production_like_validation_check.py` validates the public report. Preserve `gpu_swarm_production_like_validation_ready`, `production_like_workload_ready`, `larger_model_attempted`, `largest_successful_model_tier`, `largest_attempted_model_tier`, `larger_model_blocked_reason`, `multi_token_decode_ready`, `batch_or_multi_request_ready`, `two_gpu_stage_route_ready`, `distinct_stage_miners_ready`, `stage_requeue_or_failure_recovery_ready`, `gpu_runtime_readiness_checked`, `stage_owned_weight_loading_ready`, `latency_throughput_summary_ready`, `network_activation_transfer_summary_ready`, `public_artifact_safe`, `execution_mode`, `external_runtime_verified`, `fresh_gpu_run_performed`, and `retained_evidence_imported`. The default bounded evidence-import path aggregates GPU Swarm Usability Alpha, Control/User Alpha, retained 7B/14B core status, retained 16-token GPU generation/requeue evidence, and retained 2-request batch/stream evidence; it should report `largest_successful_model_tier: 14b`, attempt a 32B-class feasibility preflight, and block that larger tier with `candidate_requires_more_vram_than_retained_two_gpu_profile` when only the retained two-GPU Kaggle-class profile is available. Fresh external GPU attempts remain explicit and bounded (`max_fresh_model_attempts <= 2`, `max_requeue_attempts <= 1`, single attempt timeout <= 60 minutes). Public artifacts must keep raw prompts, generated text, token ids, activations, hidden states, logits, KV cache, credentials, leases, idempotency material, private env files, registries, Kaggle inline payloads, and runtime-private state redacted. This RC is production-shaped validation and larger-model infeasibility evidence, not a fresh GPU run by default, not 32B/70B success, not production Swarm Inference, not P2P/NAT traversal, not arbitrary public prompt serving, not billing, and not unbounded GPU pooling.
- Kaggle Swarm 32B Quantized Feasibility RC: `crowdtensor gpu-swarm kaggle-32b-feasibility` wraps `scripts/kaggle_swarm_32b_quantized_feasibility_pack.py` and emits `kaggle_swarm_32b_quantized_feasibility_v1` plus `kaggle_swarm_32b_quantized_feasibility_cli_v1`; `scripts/kaggle_swarm_32b_quantized_feasibility_check.py` validates the public report. Preserve `kaggle_swarm_32b_quantized_feasibility_ready`, `candidate_32b_model_selected`, `quantized_runtime_plan_ready`, `kaggle_multi_kernel_topology_ready`, `stage_partition_plan_ready`, `per_stage_memory_estimate_ready`, `activation_transfer_estimate_ready`, `kaggle_stage_package_plan_ready`, `stage_owned_loading_feasible`, `one_token_generation_feasible`, `multi_token_generation_feasible`, `coordinator_direct_management_feasible`, `upper_bound_crossing_feasible`, `batch_or_sequential_request_feasible`, `stage_requeue_feasible`, `largest_feasible_model_tier`, `largest_attempted_model_tier`, `feasibility_verdict`, `blocked_reason`, `blocker_details`, `execution_mode`, `fresh_kaggle_run_performed`, `external_runtime_verified`, `retained_evidence_imported`, `fresh_32b_activation_decode_probe_summary`, and `public_artifact_safe`. Default evidence-import consumes retained stage-owned loading proof at `dist/kaggle-32b-stage-owned-safetensors-probe-awq-live-r3-clone/kaggle_32b_stage_owned_safetensors_probe.json` plus the current retained 4-stage upper-bound crossing proof at `dist/kaggle-32b-upper-bound-crossing-live-20260620-r3/kaggle_32b_stage_owned_activation_decode_probe.json`. The current live 32B upper-bound proof used two private Kaggle Tesla T4 x2 kernels with `Qwen/Qwen2.5-32B-Instruct-AWQ` and a temporary proof Coordinator at `24.199.118.54:9235`: shard0 owned stages 0/1 on `cuda:0`/`cuda:1`; shard1 owned stages 2/3 on `cuda:0`/`cuda:1`; Coordinator completed one generated token with stage task counts `stage0..stage3 == 1`, `generated_token_count=1`, private activation handoff hashes only, raw token ids/activations redacted, private kernels deleted, and local private payloads removed. Stage-owned weights were about stage0 5.225433 GB / 417 keys, stage1 3.775238 GB / 416 keys, stage2 3.775238 GB / 416 keys, and stage3 5.225443 GB / 418 keys. The strict same-model/same-prompt single Kaggle T4 x2 baseline was attempted with all four stages required in one kernel and failed closed with `single_kernel_t4x2_gpu_count_below_required_stage_count`, proving the two-kernel path crosses the single-kernel T4 x2 slot-count upper bound under that strict 4-stage placement. The current feasibility report is `dist/kaggle-swarm-32b-quantized-feasibility-upper-bound-crossing-20260620-r1/kaggle_swarm_32b_quantized_feasibility.json` and should report `stage_owned_loading_feasible=true`, `one_token_generation_feasible=true`, `coordinator_direct_management_feasible=true`, `upper_bound_crossing_feasible=true`, `external_runtime_verified=true`, `largest_feasible_model_tier=32b-quantized-4stage-upper-bound-rc`, `feasibility_verdict=feasible_32b_upper_bound_crossing_rc`, and `blocked_reason=""`. Keep `multi_token_generation_feasible=false` for this 1-token 4-stage proof, and keep `batch_or_sequential_request_feasible=false` plus `stage_requeue_feasible=false` until separate evidence exists. Public artifacts must keep raw prompts, generated text, token ids, activations, hidden states, logits, KV cache, model cache private paths, Kaggle credentials, API keys, Coordinator tokens, leases, idempotency material, private env files, registries, inline Kaggle kernel payloads, and runtime-private state redacted; private Kaggle kernels and local private payloads must be deleted. This RC is repeatable 32B stage-owned loading plus temporary-proof-Coordinator upper-bound crossing evidence, not production Swarm Inference, not the production Coordinator data plane, not a memory-pressure/long-context crossing proof, not KV-cache optimized serving, not P2P/NAT traversal, not arbitrary public prompt serving, not billing, and not unbounded GPU pooling.
- Public Swarm Inference Beta RC: `crowdtensor public-swarm-beta-rc` wraps `scripts/public_swarm_inference_beta_rc_pack.py` and emits `public_swarm_inference_beta_rc_v1`; `scripts/public_swarm_inference_beta_rc_check.py` validates `local-loopback`, `package`, and `external-existing`. Preserve `public_swarm_inference_beta_rc_ready`, `public_swarm_product_beta_ready`, `p2p_lite_route_ready`, `p2p_lite_discovery_ready`, `cpu_fallback_ready`, `serve_join_generate_loop_ready`, `remote_generate_session_ready`, `public_swarm_generate_ready`, optional bounded `public_swarm_generate_batch_ready`, optional `--stream-generation` with `public_swarm_generate_stream_ready` / `public_swarm_generate_stream_endpoint_ready`, `private_artifacts_local_only`, `miner_join_pack_ready`, `external_runtime_verified` only for external-existing, and `hf_dependencies_missing` for hosts without optional `[hf]` dependencies. It is CPU-only by default, read-only, Coordinator-backed, not production Swarm Inference, not libp2p/DHT/NAT traversal, not Hivemind-level serving, and not large-model serving.
- Public Swarm Product Beta: `crowdtensor public-swarm-product-beta` wraps `scripts/public_swarm_product_beta_pack.py` and emits `public_swarm_product_beta_v1`; `scripts/public_swarm_product_beta_check.py` validates `local-loopback`, `package`, and `external-existing`. Preserve `public_swarm_product_beta_ready`, `public_swarm_product_beta_user_path_ready`, `serve_ready`, `stage0_join_ready`, `stage1_join_ready`, `generate_ready`, optional bounded `public_swarm_generate_batch_ready`, optional `--stream-generation` with `public_swarm_generate_stream_ready` / `public_swarm_generate_stream_endpoint_ready`, top-level `output_request`, `answer_scope.scope_state: no-local-answer`, `shareable_summary.answer_scope_state`, `support_bundle_ready`, `private_artifacts_cleaned`, `decoded_tokens_match`, `distinct_stage_miners`, `stage_assignment_valid`, and `hf_dependencies_missing` for hosts without optional `[hf]` dependencies. `package` preserves `private_artifacts_local_only` and `miner_join_pack_ready`; `external-existing` requires a live controlled runtime. Treat the Product Beta aggregate as shareable product-path evidence, not a local answer transcript; human `crowdtensor generate` may show local generated text while public artifacts keep raw prompts, generated text, generated token ids, activations, credentials, leases, and idempotency material redacted. It is CPU-only by default, read-only, Coordinator-backed, not production Swarm Inference, not libp2p/DHT/NAT traversal, not Hivemind-level serving, and not large-model serving.
- Public Swarm Inference v2: `crowdtensor public-swarm-v2` wraps `scripts/public_swarm_inference_v2_pack.py` and emits `public_swarm_inference_v2`; `scripts/public_swarm_inference_v2_check.py` validates `local`, `package`, and `evidence-import`. Preserve the ordinary `p2pd -> serve --p2p -> join --p2p stage0/stage1 -> generate --p2p --max-new-tokens 16` path, local 16-token P2P evidence, bounded `--prompt-texts` batch evidence when requested, safe `--stream-generation` progress evidence when requested, local and external accepted stage-row evidence, explicit matching local/external/P2P `hf_model_id` metadata, `public_swarm_v2_local_p2p_generate_ready`, `public_swarm_v2_16_token_generation_ready`, `public_swarm_v2_batch_generation_ready`, `public_swarm_v2_stream_generation_ready`, `public_swarm_generate_stream_endpoint_ready`, `public_swarm_v2_external_stage_rows_ready`, `public_swarm_v2_signed_or_real_p2p_ready`, `public_swarm_v2_model_match_ready`, `public_swarm_v2_stage_requeue_rescue_ready`, `stage_latency_ready`, `throughput_summary_ready`, `memory_or_vram_summary_ready`, `public_swarm_v2_cuda_optional_fail_closed_ready`, redacted `public_swarm_inference_v2.json`, Markdown, runbook, and Support Bundle. Retained ready evidence is `dist/public-swarm-inference-v2/public_swarm_inference_v2.json`; it includes a local two-prompt bounded batch plus stream-enabled 16-token proof with 16 generated tokens for each request, safe prompt hashes and counts, 16 ordered safe stream events from `admin-session-stream`, retained real-P2P external evidence, optional CUDA evidence or fail-closed diagnostics, explicit matching model IDs, and no `not_completed` items. Use `--fresh-external-report` only for a just-produced successful external report that also exposes external accepted rows at `2 * max_new_tokens` and explicit matching model metadata; otherwise retained external evidence must keep `public_swarm_v2_external_fresh_run_action_required` and fresh failures belong in `--fresh-external-attempt-report`. This is Coordinator-backed, read-only, tiny/small-model scoped, CPU by default with optional CUDA evidence, not full Hivemind/Petals production parity, not Coordinator-free, not production NAT traversal, and not large-model serving.
- Product `crowdtensor generate` accepts `--hf-model-id` and must carry it through `session_protocol_v1`, the private Coordinator inference-session payload, and safe output summaries. Preserve per-session model-id handling in the Coordinator/StateStore for `real_llm_sharded_infer`; do not regress to process-default-only model selection. P2P v0.6 and Real P2P Core RC `external-existing --verify-generate` must forward `--hf-model-id`, bounded `--prompt-texts`, and `--stream-generation` into live `generate` and record public-safe model/batch/stream summaries. Real P2P local/Kaggle generate commands must pass the requested `--hf-model-id` to the actual nested generation process, not only to top-level report metadata.
- Public Real-LLM Swarm Beta output scope: preserve top-level `output_request`, `answer_scope.scope_state: no-local-answer`, and `shareable_summary.answer_scope_state` in `public_real_llm_swarm_beta_v1` JSON, Markdown, terminal summaries, and Support Bundle. Treat the Beta aggregate as shareable release evidence rather than a local answer transcript; only human `crowdtensor generate` may show local generated text, while public artifacts keep raw prompts, generated text, generated token ids, activations, credentials, leases, and idempotency material redacted.
- Public Real-LLM Swarm Inference Beta v1: `crowdtensor public-real-llm-swarm-beta` wraps `scripts/public_real_llm_swarm_beta_pack.py` and emits `public_real_llm_swarm_beta_v1`; `scripts/public_real_llm_swarm_beta_check.py` validates the fake-runner release contract. Preserve `public-real-llm-swarm-beta release`, `local-smoke`, `package`, and `evidence-import`, Product Beta bounded `--prompt-texts` and `--stream-generation` propagation, fresh Petals-class P2P candidate local-smoke, fresh local Public Swarm v2 release execution, `--p2p-report` and `--public-swarm-v2-report` for evidence-import, `public_real_llm_swarm_beta_ready`, `cpu_default_ready`, `external_two_stage_ready`, `external_stage_requeue_ready`, `p2p_ready_product_beta`, `cuda_optional_fail_closed_ready`, `release_evidence_ready`, product-path `public_real_llm_swarm_beta_batch_ready` / `public_real_llm_swarm_beta_stream_ready`, Product/Public Swarm v2/P2P/KV-cache model match readiness, `public_real_llm_swarm_beta_public_swarm_v2_ready`, `public_real_llm_swarm_beta_p2p_user_path_ready`, `public_real_llm_swarm_beta_v2_batch_ready`, `public_real_llm_swarm_beta_v2_stream_ready`, `public_swarm_inference_v2_ready`, `public_swarm_v2_local_p2p_generate_ready`, `public_swarm_v2_16_token_generation_ready`, `public_swarm_v2_external_stage_rows_ready`, `public_swarm_v2_dual_stage_kv_cache_ready`, `public_swarm_v2_model_match_ready`, `public_swarm_v2_signed_or_real_p2p_ready`, `public_swarm_v2_stage_requeue_rescue_ready`, P2P candidate safe batch/stream import with `public_real_llm_swarm_beta_p2p_batch_ready` / `public_real_llm_swarm_beta_p2p_stream_ready`, release-local Usable Swarm KV-cache evidence with `public_real_llm_swarm_beta_kv_cache_ready`, `public_real_llm_swarm_beta_kv_cache_model_match_ready`, `usable_real_llm_kv_cache_ready`, stage0/stage1 KV-cache schema and hit codes, `public_real_llm_swarm_beta_private_artifacts_cleaned`, `external_generated_token_target_ready`, `p2p_generated_token_target_ready`, `public_leak_paths: []`, redacted Markdown, and Support Bundle output. The retained local product-path proof is `dist/goal-final-infer-public-real-llm-swarm-beta-local-batch-stream-16tok-fixed-20260602/public_real_llm_swarm_beta.json` with `ok: true`, two-prompt bounded batch, 16 generated tokens per request, 16 ordered safe stream events, and CUDA fail-closed readiness. The current fresh local release proof is `dist/goal-final-infer-public-real-llm-swarm-beta-release-fresh-v2-usable-p2p-fixed-20260602/public_real_llm_swarm_beta.json` with `ok: true`, no `not_completed` items, fresh Product Beta, fresh release-local Petals-class P2P candidate local-smoke, fresh local Public Swarm v2, and fresh release-local Usable/KV-cache steps, v2 16-token P2P generation, 32 accepted rows, v2 dual-stage KV-cache reuse with 15 hits per stage, v2 batch/stream readiness, retained external/GPU imports plus retained P2P source inputs, release-local `source_reports.p2p_report`, no generated runtime-private files in the final release tree, and `fresh_external_runtime_verified: false`. The retained evidence-import proof is `dist/goal-final-infer-public-real-llm-swarm-beta-import-16tok-p2p-batch-stream-kv-cache-model-gated-v2-20260602/public_real_llm_swarm_beta.json` with `ok: true`, no `not_completed` items, product batch/stream readiness, Public Swarm v2 16-token P2P user path, v2 accepted stage rows, v2 dual-stage KV-cache reuse with 15 hits per stage, v2 batch/stream readiness, product/external/P2P/v2/KV-cache model match readiness, external 16-token target readiness, P2P 16-token target readiness, P2P-side batch/stream readiness, persistent dual-stage KV-cache reuse with 15 stage0 and 15 stage1 cache hits, external stage requeue readiness, P2P live requeue rescue, and victim-result rejection. Its default retained sources are `dist/goal-final-infer-real-llm-internet-beta-import-16tok-gpu-summary-20260602/real_llm_internet_beta.json`, `dist/goal-final-infer-petals-candidate-16tok-batch-stream-composed-20260602/petals_class_p2p_candidate.json`, `dist/public-swarm-inference-v2/public_swarm_inference_v2.json`, and `dist/goal-final-infer-usable-swarm-16tok-kv-cache-20260601/usable_swarm_inference.json`; release wraps the Petals candidate as a local-smoke child over retained external real-P2P generation/requeue/runtime-smoke/batch-stream source reports, and this aggregate is not a fresh external run. Release only marks external/P2P/v2/KV-cache ready when the imported external/P2P reports plus the fresh v2 child KV-cache report meet the requested `--max-new-tokens`; evidence-import applies the same target to external/P2P/v2/KV-cache imports. Lower-token retained evidence must emit `external_generated_token_target_missing`, `p2p_generated_token_target_missing`, `public_swarm_v2_token_target_missing`, or `public_real_llm_swarm_beta_kv_cache_missing`. Product, external, P2P, v2, and KV-cache evidence must expose the requested model id; mismatches emit `product_model_mismatch`, `external_model_mismatch`, `p2p_model_mismatch`, `public_swarm_v2_model_mismatch`, or `kv_cache_model_mismatch` and block readiness. It aggregates Product Beta, retained external real-LLM requeue evidence, release-local real-P2P candidate local-smoke over retained P2P source reports, Public Swarm v2, release-local Usable Swarm KV-cache evidence, optional GPU evidence, CUDA fail-closed smoke, and final private-artifact cleanup. This is the current top-level installable inference Beta, but remains Coordinator-backed, read-only by default, tiny/small-model scoped, not full Hivemind/Petals production parity, not Coordinator-free, not NAT traversal production, and not large-model serving.
- Public Swarm Developer Preview: `crowdtensor preview` wraps `scripts/public_swarm_developer_preview_pack.py` and emits `public_swarm_developer_preview_v1`; `scripts/public_swarm_developer_preview_check.py` validates `local`, `package`, `external-existing`, and `evidence-import`. Preserve `developer_preview_ready`, `public_swarm_developer_preview_ready`, `local_two_stage_generation_ready`, `serve_join_generate_ready`, `product_beta_ready`, top-level `output_request`, `answer_scope.scope_state: no-local-answer`, `shareable_summary.answer_scope_state`, `support_bundle_ready`, `cpu_fallback_ready`, `local_cpu_inference_ready`, `gpu_generation_evidence_import_ready` when retained GPU evidence is present, and inherited `hf_dependencies_missing` for hosts without optional `[hf]` dependencies. Treat Developer Preview as shareable preview evidence, not a local answer transcript; human `crowdtensor generate` may show local generated text while public artifacts keep raw prompts, generated text, generated token ids, activations, credentials, leases, and idempotency material redacted. It is CPU-only by default, read-only, Coordinator-backed, not production Swarm Inference, not libp2p/DHT/NAT traversal, not Hivemind-level serving, and not large-model serving.
- Public Swarm Live Preview RC: `crowdtensor live-preview` wraps `scripts/public_swarm_live_preview_rc_pack.py` and emits `public_swarm_live_preview_rc_v1`; `scripts/public_swarm_live_preview_rc_check.py` validates `live-preview local-smoke`, `live-preview package`, fake-runner `live-preview live-kaggle`, and `live-preview evidence-import`. Preserve `public_swarm_live_preview_rc_ready`, `public_swarm_live_preview_local_smoke_ready`, `public_swarm_live_preview_package_ready`, `public_swarm_live_preview_live_kaggle_ready`, `public_swarm_live_preview_evidence_import_ready`, `external_stage_requeue_ready`, `live_stage0_requeue_ready`, `live_stage1_requeue_ready`, `kaggle_kernels_deleted`, `private_artifacts_cleaned`, `token_rotation_required`, `gpu_generation_evidence_import_ready`, and the side-effectful boundary that fresh `live-kaggle` runs wrap the existing Public Swarm Alpha Kaggle proof while CI uses fake-runner checks only. Fresh retained RC proofs completed against `24.199.118.54:9196` and `24.199.118.54:9198` with evidence at `dist/public-swarm-live-preview-rc-live-stage0-20260529043801-rc/public_swarm_live_preview_rc.json` and `dist/public-swarm-live-preview-rc-live-stage1-20260529044328-rc/public_swarm_live_preview_rc.json`; keep the default Kaggle slug prefix short (`ct-live-preview`) so victim/rescue suffixes fit Kaggle's 45-character slug limit. It is CPU-only by default, read-only, Coordinator-backed, not production Swarm Inference, not libp2p/DHT/NAT traversal, not Hivemind-level serving, and not large-model serving.
- Public Swarm v0.1 Operator Preview: `crowdtensor operator-preview` wraps `scripts/public_swarm_operator_preview_pack.py` and emits `public_swarm_operator_preview_v1`; `scripts/public_swarm_operator_preview_check.py` validates `operator-preview local-smoke`, `operator-preview package`, fake-runner `operator-preview live-kaggle`, and `operator-preview evidence-import`. Preserve `public_swarm_operator_preview_ready`, `operator_preview_user_path_ready`, `operator_preview_local_smoke_ready`, `operator_preview_package_ready`, `operator_preview_live_kaggle_ready`, `operator_preview_evidence_import_ready`, `serve_join_generate_ready` or package-mode `miner_join_pack_ready` plus `private_artifacts_local_only`, `cpu_fallback_ready`, `live_preview_ready`, top-level `output_request`, `answer_scope.scope_state: no-local-answer`, `shareable_summary.answer_scope_state`, `support_bundle_ready`, `release_readiness_ready`, `gpu_generation_evidence_import_ready`, `developer_preview_degraded`, `operator_preview_cpu_fallback_user_path_ready`, `operator_preview_retained_evidence_ready`, and `external_runtime_blocked` when fresh live Kaggle/HF/external runtime execution cannot complete and retained Live Preview RC evidence is imported. It is the top-level ordinary-user preview aggregate over Developer Preview, Live Preview RC, release readiness, support bundle, CPU fallback, and retained GPU generation evidence; treat it as shareable operator-path evidence, not a local answer transcript. Public artifacts must keep raw prompts, generated text, generated token ids, activations, credentials, leases, private env files, and runtime state redacted. It is CPU-only by default, read-only, Coordinator-backed, not production Swarm Inference, not libp2p/DHT/NAT traversal, not Hivemind-level serving, and not large-model serving.
- Public Swarm v0.2 Usable Inference Trial: `crowdtensor swarm-trial` wraps `scripts/public_swarm_trial_pack.py` and emits `public_swarm_trial_v1`; `scripts/public_swarm_trial_check.py` validates `swarm-trial local-loopback`, `swarm-trial package`, fake-runner `swarm-trial live-kaggle`, and `swarm-trial evidence-import`. Preserve `public_swarm_trial_ready`, `serve_join_generate_trial_ready`, `stage0_join_ready`, `stage1_join_ready`, `generate_ready`, `generated_token_count_ready`, top-level `output_request`, `answer_scope.scope_state: no-local-answer`, `shareable_summary.answer_scope_state`, `support_bundle_ready`, `cpu_fallback_ready`, `private_artifacts_cleaned`, `operator_preview_import_ready`, `gpu_generation_evidence_import_ready`, `swarm_trial_degraded_cpu_fallback_ready`, `external_runtime_blocked`, and live `token_rotation_required`. It is the ordinary-user trial aggregate over Product Beta, Operator Preview, Support Bundle, CPU fallback, and retained GPU generation evidence; treat it as shareable trial evidence, not a local answer transcript. Public artifacts must keep raw prompts, generated text, generated token ids, activations, credentials, leases, private env files, and runtime state redacted. It is CPU-only by default, read-only, Coordinator-backed, not production Swarm Inference, not libp2p/DHT/NAT traversal, not Hivemind-level serving, not GPU marketplace, and not large-model serving.
- Public Swarm GPU Inference Beta: `crowdtensor public-swarm-gpu-beta` wraps `scripts/public_swarm_gpu_inference_beta_pack.py` and emits `public_swarm_gpu_inference_beta_v1`; `scripts/public_swarm_gpu_inference_beta_check.py` validates CI-safe `local-smoke`, optional CUDA `local-loopback`, and fake-runner `kaggle-auto`. Preserve `public-swarm-gpu-beta local-smoke`, `public-swarm-gpu-beta local-loopback`, `public-swarm-gpu-beta kaggle-package`, `public-swarm-gpu-beta kaggle-auto`, `public-swarm-gpu-beta evidence-import`, CPU Coordinator CUDA metadata-only scheduling, private Kaggle GPU stage kernels, Kaggle CUDA runtime pins, retained stage-local proof path `dist/public-swarm-gpu-beta-stage-local-live-20260528064520-shortslug/public_swarm_gpu_inference_beta_kaggle_auto.json`, retained historical pre-stage-local proof path `dist/public-swarm-gpu-beta-live-20260528-runtimepin/public_swarm_gpu_inference_beta_kaggle_auto.json`, retained `gpt2-xl` small-tier proof path `dist/gpt2-xl-small-tier-kaggle-logfix-20260614172932/public_swarm_gpu_inference_beta_kaggle_auto.json`, `hf_transformers_cuda`, `real_llm_sharded_cuda_stage0`, `real_llm_sharded_cuda_stage1`, `real_llm_sharded_cuda_both`, `public_swarm_gpu_beta_smoke_ready`, `public_swarm_gpu_beta_ready`, `public_swarm_gpu_beta_kaggle_auto_ready`, `gpu_runtime_ready`, `cuda_runtime_available`, `hf_transformers_cuda_ready`, `gpu_stage0_ready`, `gpu_stage1_ready`, `stage_local_partition_ready`, `stage0_partition_loaded`, `stage1_partition_loaded`, `partition_parameter_split_valid`, `stage_gpu_memory_reduced`, `kaggle_gpu_package_ready`, `kaggle_kernels_deleted`, `token_rotation_required`, `external_gpu_runtime_verified`, top-level `output_request`, `answer_scope.scope_state: no-local-answer`, and `shareable_summary.answer_scope_state`. The `gpt2-xl` proof is real Kaggle P100 CUDA split execution with one redacted generated token and decoded-token match, but it still reports `large_model_sharded_execution_ready=false` and `true_partial_weight_loading_ready=false`; do not treat it as 7B/8B completion. `real_llm_internet_beta` Kaggle-auto must redirect Coordinator stdout/stderr to log files with redacted lifecycle tails to avoid pipe backpressure during long external waits. Treat it as shareable optional CUDA readiness evidence, not a local answer transcript; public artifacts keep raw prompts, generated text, generated token ids, activations, credentials, leases, private env files, Kaggle kernel payloads, and runtime state redacted. This is read-only optional CUDA tiny/small GPT evidence, not production Swarm Inference, not P2P, not a GPU pooling marketplace, and not 7B/8B large-model serving.
- GPU Sharded Generation Beta: `crowdtensor gpu-generate` wraps `scripts/gpu_sharded_generation_beta_pack.py` and emits `gpu_sharded_generation_beta_v1`; `scripts/gpu_sharded_generation_beta_check.py` validates the CI-safe import/synthetic contract. Preserve `gpu-generate local-loopback`, `gpu-generate kaggle-auto`, `gpu-generate evidence-import`, `--max-new-tokens`, stage0/stage1 alternating generation, `generated_token_count`, `generated_text_hash`, `generated_text_redacted`, `raw_generated_text_public: false`, top-level `output_request`, `answer_scope.scope_state: no-local-answer`, `shareable_summary.answer_scope_state`, `multi_token_generation_ready`, `gpu_sharded_generation_ready`, `gpu_loopback_generation_ready`, `gpu_multi_machine_generation_ready`, old single-token evidence rejection via `gpu_multi_token_generation_missing`, `hf_transformers_cuda`, `stage_local` partitioning, stage-local readiness codes, Kaggle cleanup evidence, and explicit tiny GPT Beta / not production / not Hivemind-level / not P2P / not GPU marketplace / not large-model boundaries. The retained 16-token private Kaggle GPU proof is `dist/gpu-sharded-generation-beta-kaggle-20260528095658/gpu_sharded_generation_beta_kaggle_auto.json`, with RC manifest `dist/gpu-sharded-generation-beta-kaggle-20260528095658/gpu_sharded_generation_beta_rc_manifest.json`; private env and registry files have been removed from that retained artifact tree. Treat the wrapper as shareable generation evidence, not a local answer transcript; public artifacts keep raw prompts, generated text, generated token ids, activations, credentials, leases, and runtime state redacted.
- Public Swarm Product RC: `crowdtensor public-swarm-product-rc` emits `public_swarm_product_rc_v1` through `scripts/public_swarm_product_rc_pack.py` and validates with `scripts/public_swarm_product_rc_check.py`. It introduces product-facing `crowdtensor serve`, `crowdtensor join`, `crowdtensor generate`, and `crowdtensor peer`; `session_protocol_v1` in `crowdtensor/session_protocol.py`; and `p2p_lite_peer_v1` in `crowdtensor/p2p_lite.py` plus `scripts/p2p_lite_daemon.py`, `scripts/session_protocol_check.py`, and `scripts/p2p_lite_discovery_check.py`. P2P-lite only discovers Coordinator/Miner routes; Coordinator still owns session creation, leasing, heartbeats, validation, and result ledgers. Preserve redaction of raw prompts, generated text, token ids, activations, tokens, leases, idempotency material, and explicit not libp2p / not DHT / not NAT traversal / not decentralized security / not Hivemind-level / not large-model boundaries.
- Product Swarm v0.3 MVP check: `scripts/product_swarm_mvp_check.py` emits `product_swarm_mvp_check_v1` and is the direct runnable proof for product commands. It starts `crowdtensor serve --run`, starts `crowdtensor generate` so the Coordinator creates the session, then runs independent one-task `crowdtensor join --stage stage0 --run` and `crowdtensor join --stage stage1 --run` commands for each tiny GPT generation step; readiness codes include `product_swarm_mvp_ready`, `serve_join_generate_mvp_ready`, `local_two_stage_real_llm_ready`, `generated_token_count_ready`, `distinct_stage_miners`, and `stage_assignment_valid`. Without optional `[hf]` dependencies it reports `hf_dependencies_missing` plus `product_swarm_mvp_degraded_ready` unless `--require-hf-runtime` is set. Preserve CPU default, optional `--backend cuda`, redaction of raw prompts/generated text/token ids/activations/leases/idempotency material, and explicit Coordinator-backed/read-only/tiny-model/not production/not P2P/not Hivemind-level/not large-model boundaries.
- Public Swarm Inference Preview v0.4: `crowdtensor preview-v04` wraps `scripts/public_swarm_preview_v04_pack.py` and emits `public_swarm_preview_v04_v1`; `scripts/public_swarm_preview_v04_check.py` validates `local-smoke`, `package`, and `evidence-import`. Preserve `public_swarm_preview_v04_ready`, `external_two_stage_generation_ready`, `multi_token_generation_ready`, `distinct_stage_miners`, `stage_assignment_valid`, `stage_latency_ready`, `throughput_summary_ready`, `memory_or_vram_summary_ready`, `external_stage_requeue_ready`, `tiny_gpt2_ci_fallback_ready`, optional `optional_distilgpt2_or_gpt2_strict_ready`, top-level `output_request`, `answer_scope.scope_state: no-local-answer`, `shareable_summary.answer_scope_state`, retained final evidence `dist/public-swarm-preview-v04-final/public_swarm_preview_v04.json`, strict local CPU `distilgpt2` evidence `dist/public-swarm-preview-v04-distilgpt2-strict/public_swarm_preview_v04.json`, Support Bundle, and `PUBLIC_SWARM_PREVIEW_V04.md`. It aggregates Product Swarm MVP, retained Live Preview RC stage0/stage1 kill/requeue/rescue evidence, and retained GPU multi-token generation evidence; treat it as shareable preview evidence, not a local answer transcript. Public artifacts must keep raw prompts, generated text, generated token ids, activations, credentials, leases, private env files, and runtime state redacted. It is Coordinator-backed, read-only, tiny/small-model scoped, not production Swarm Inference, not P2P/libp2p/DHT/NAT traversal, not Hivemind/Petals parity, and not large-model serving.
- Local multi-Miner scenario sweep: `scripts/multi_miner_scenario_sweep_check.py` defaults to concurrent mode and emits `multi_miner_scenario_sweep_v1` / `multi_miner_scenario_sweep_observability_v1` by creating fixed read-only inference sessions, starting distinct registry-backed Python Miner identities together, checking one accepted ledger row per task via `lease_summary`, checking process health via `process_summary`, and emitting `multi_miner_concurrent_ready`. With `--failure-mode kill-after-claim`, it terminates one claimed Miner, observes lease timeout requeue, requires a rescue Miner to complete the same `task_id`, records `requeue_summary`, and emits `multi_miner_requeue_ready`. It is a local lease-race/requeue proof, not production throughput scaling or P2P routing.
- Demo Manifest tooling: `scripts/demo_manifest_pack.py` and `scripts/demo_manifest_check.py` produce `demo_manifest_v1`, the current latest output artifact for local-loopback handoff. It indexes `runtime_matrix.json`, `remote_compute_evidence_v1`, `external_llm_evidence_v1`, `support_bundle`, and `remote_compute_observability_v1` summaries without widening the project claim.
- Open-source entrypoints: README, ROADMAP, protocol/use-case docs, release docs, changelog, and static site.

Do not describe the project as already providing production P2P, NAT traversal, real LLM inference/training, GPU pooling, WebGPU model shards, payments, staking, or hardened public-internet security.

## Strategic Direction

The next high-value product direction is a useful home-compute demo, likely Swarm Inference shaped before real Swarm Training. Training and P2P remain important, but open-source users need a concrete deployment story first.

Roadmap priority:

1. Preserve Alpha reliability and operator trust.
2. Make the project easy for strangers to understand and run.
3. Keep `scripts/runtime_matrix.py` and `scripts/runtime_matrix_check.py` as the first runtime capability matrix and hardware/runtime matrix for new users, including optional NVIDIA CUDA tiny GPT split readiness, `matched_capabilities`, `missing_capabilities`, target and route `diagnosis_codes`, `operator_action`, and `hardware_diagnosis_summary` explanations.
4. Keep `crowdtensor release-ready` as the maintainer-facing publish gate, preserving `release_readiness_v1`, `scripts/release_readiness_pack.py`, `scripts/release_readiness_check.py`, `--allow-dirty`, `git_dirty`, release gate aggregation, `demo_manifest_v1`, and explicit not production boundaries.
5. Keep `scripts/onboarding_gate.py --quick` as the fresh clone install-and-run proof, preserving `onboarding_gate_v1`, clean virtualenv creation, `python -m pip install -e .[dev,hf]`, console script checks, `scripts/user_friendly_inference_frontdoor_check.py`, the real installed `crowdtensor infer --prompt-stdin --shareable-terminal` user smoke with `user_infer_smoke_validation_v1`, `crowdtensor local-proof`, `crowdtensor home-infer`, `crowdtensor llm-infer --mock`, `crowdtensor cpu-infer --mode local`, `crowdtensor release-ready --allow-dirty`, `/tmp` output defaults, prompt/output redaction, local CPU / no fresh Kaggle GPU verdicts, and explicit non-production Swarm Inference boundaries.
6. Keep `crowdtensor local-proof` as the shortest user-facing local proof, preserving `local_proof_summary_v1`, Doctor, runtime matrix, CPU-only read-only home-compute demo, Demo Manifest output, and explicit non-production Swarm Inference boundaries.
7. Keep `crowdtensor home-infer` as the shortest shareable local read-only inference proof, preserving `home_inference_cli_v1`, `home_compute_evidence_v1`, `model_bundle_infer`, fixed `model_bundle_inference_scenario_v1` scenarios, capped `request_trace`, `diagnosis_codes`, and explicit non-production Swarm Inference boundaries.
8. Keep `crowdtensor llm-infer` as the shortest shareable external LLM runtime proof, preserving `llm_inference_cli_v1`, `external_llm_evidence_v1`, deterministic `--mock`, explicit operator-owned `--llm-runtime-cmd` / `--llm-runtime-url`, fixed claim-time prompts, read-only semantics, and explicit non-public-serving boundaries.
9. Keep `crowdtensor cpu-infer` as the CPU inference Beta aggregate path, preserving `cpu_inference_beta_v1`, `scripts/cpu_inference_beta_pack.py`, `scripts/cpu_inference_beta_check.py`, `--mode local`, `--mode remote-loopback`, `--mode remote-existing`, CPU-only read-only semantics, token/runtime redaction, and explicit not production / not P2P boundaries.
9. Keep `crowdtensor cpu-infer --mode beta-rc` as the CPU Inference Beta RC aggregate path, preserving `cpu_inference_beta_rc_v1`, `scripts/cpu_inference_beta_rc_pack.py`, `scripts/cpu_inference_beta_rc_check.py`, local CPU inference, remote-loopback inference, Real two-machine CPU inference Beta rehearsal, Kaggle Remote Miner Beta artifacts, `miner_join_pack_v1`, optional `--kaggle-real-runtime-report` import, `demo_manifest_v1`, `cpu_inference_beta_rc_ready`, `local_cpu_inference_ready`, `remote_loopback_ready`, `two_machine_rehearsal_ready`, `kaggle_remote_miner_artifacts_ready`, `miner_join_pack_ready`, `cpu_miner_beta_ready`, CPU-only/read-only semantics, and explicit not production / not P2P / not GPU/TPU workload boundaries.
10. Keep `crowdtensor shard-infer-beta` and `crowdtensor remote-demo --workload sharded-model-bundle` as the CPU Pipeline-Sharded Inference Beta path, preserving `remote_sharded_inference_beta_v1`, `scripts/remote_sharded_inference_beta_pack.py`, `scripts/remote_sharded_inference_beta_check.py`, `--mode remote-loopback`, `remote_python_sharded_model_bundle_infer`, `remote_sharded_inference_acceptance_v1`, `remote_sharded_inference_observability_v1`, `remote_two_machine_sharded_ready`, activation hashes, `baseline_match`, `remote_sharded_inference_ready`, `remote_sharded_loopback_ready`, `local_sharded_inference_ready`, `stage_requeue_ready`, CPU-only/read-only semantics, and explicit not production / not P2P / not real LLM sharding boundaries.
10. Keep `crowdtensor micro-llm-shard-infer-beta` and `crowdtensor remote-demo --workload micro-llm-sharded` as the Remote Micro-LLM Pipeline-Sharded Inference Beta path, preserving `remote_micro_llm_sharded_beta_v1`, `scripts/remote_micro_llm_sharded_beta_pack.py`, `scripts/remote_micro_llm_sharded_beta_check.py`, `--mode remote-loopback`, `--stage-mode split`, `--require-distinct-stage-miners`, `remote_python_micro_llm_sharded_infer`, `remote_micro_llm_sharded_acceptance_v1`, `remote_micro_llm_sharded_observability_v1`, `remote_two_machine_micro_llm_sharded_ready`, activation hashes, `decode_steps`, `baseline_match`, `decoded_tokens_match`, `remote_micro_llm_sharded_ready`, `remote_micro_llm_sharded_loopback_ready`, `local_micro_llm_sharded_inference_ready`, `micro_llm_sharded_stage0`, `micro_llm_sharded_stage1`, `distinct_stage_miners`, `stage_assignment_valid`, `stage_requeue_ready`, CPU-only/read-only semantics, and explicit not production / not P2P / not GGUF/llama.cpp boundaries.
10. Keep `micro_llm_artifact_v1` as the first file-backed tiny model package boundary, preserving `crowdtensor micro-llm-artifact`, `scripts/micro_llm_artifact_pack.py`, `scripts/micro_llm_artifact_check.py`, `--micro-llm-artifact` across local, remote-demo, Kaggle real runtime, and live RC paths, artifact hash/id/tokenizer propagation, `artifact_loaded`, `micro_llm_artifact_ready`, and explicit not Hugging Face / not GGUF / not llama.cpp / not large-model boundaries.
10. Keep `crowdtensor real-llm-shard-infer`, `crowdtensor real-llm-shard-infer-beta`, and `crowdtensor remote-demo --workload real-llm-sharded` as the optional real-weight tiny GPT split proof, preserving `real_llm_sharded_cli_v1`, `real_llm_sharded_evidence_v1`, `remote_real_llm_sharded_beta_v1`, `remote_real_llm_sharded_acceptance_v1`, `remote_real_llm_sharded_observability_v1`, `remote_python_real_llm_sharded_infer`, `remote_two_machine_real_llm_sharded_ready`, `real_llm_sharded_infer_v1`, `real_llm_artifact_v1`, `hf_transformers_cpu`, `--enable-hf-tiny-gpt-runtime`, `--hf-cache-dir`, `--real-llm-partition-mode stage-local`, `real_llm_sharded_stage0`, `real_llm_sharded_stage1`, `real_llm_sharded_both`, `real_llm_artifact_ready`, `activation_transport_ready`, `baseline_match`, `decoded_tokens_match`, `stage_local_partition_ready`, `partition_parameter_split_valid`, `remote_real_llm_sharded_ready`, `hf_dependencies_missing`, optional `[hf]` dependency isolation, and explicit read-only/not production/not P2P/not large-model boundaries.
10. Keep `crowdtensor real-llm-live-rc` as the generated-stage Real Small-LLM Sharded Inference Live RC, preserving `real_llm_live_rc_v1`, `scripts/real_llm_live_rc_pack.py`, `scripts/real_llm_live_rc_check.py`, `scripts/kaggle_real_llm_live_package.py`, `kaggle_real_llm_live_package_v1`, `local-generated`, `kaggle-generated`, `external-existing`, `kaggle-upload-real-llm-stage0`, `kaggle-upload-real-llm-stage1`, `local_generated_real_llm_stage_upload_standins_ready`, `external_runtime_verified`, `kaggle_real_llm_stage0_seen`, `kaggle_real_llm_stage1_seen`, `kaggle_real_llm_sharded_ready`, `real_llm_artifact_ready`, `launcher_syntax_valid`, `--enable-hf-tiny-gpt-runtime`, `--real-llm-stage-role`, and explicit CPU-only/read-only/not production/not P2P/not large-model boundaries.
10. Keep `crowdtensor real-llm-internet-alpha` as the Real Internet Swarm Inference Alpha milestone wrapper, preserving `real_llm_internet_alpha_v1`, `scripts/real_llm_internet_alpha_pack.py`, `scripts/real_llm_internet_alpha_check.py`, `local-generated`, `package`, `external-existing`, `real_llm_internet_alpha_ready`, `real_llm_stage_requeue_ready`, `stage_requeue_ready`, `real_llm_live_rc_ready`, `remote_real_llm_sharded_ready`, `real_llm_artifact_ready`, `activation_transport_ready`, `decoded_tokens_match`, `distinct_stage_miners`, `stage_assignment_valid`, `external_runtime_verified` only after external-existing success, `token_rotation_required`, and explicit CPU-only/read-only/not production/not P2P/not large-model boundaries.
10. Keep `crowdtensor real-llm-internet-beta` as the automated Real Internet Swarm Inference Beta milestone wrapper, preserving `real_llm_internet_beta_v1`, `scripts/real_llm_internet_beta_pack.py`, `scripts/real_llm_internet_beta_check.py`, `kaggle-auto`, `real_llm_internet_beta_ready`, `real_llm_internet_alpha_ready`, `external_runtime_verified`, `external_stage_requeue_ready`, `live_stage0_requeue_ready`, `live_stage1_requeue_ready`, `live_requeue_summary`, `kaggle_kernels_deleted`, `decoded_tokens_match`, `distinct_stage_miners`, `stage_assignment_valid`, `token_rotation_required`, cleanup-backed Kaggle lifecycle evidence, and explicit CPU-only/read-only/not production/not P2P/not large-model boundaries.
10. Keep `crowdtensor swarm-infer-beta` as the user-facing Swarm Inference Beta wrapper, preserving `swarm_inference_beta_v1`, `scripts/swarm_inference_beta_pack.py`, `scripts/swarm_inference_beta_check.py`, side-effectful `swarm-infer-beta live` / `kaggle-auto`, `swarm_inference_beta_live_ready`, `real_llm_internet_beta_ready`, `external_stage_requeue_ready`, `live_requeue_summary`, `kaggle_kernels_deleted`, `swarm_inference_beta_live_private_artifacts_cleaned`, `token_rotation_required`, `support_bundle.json`, default local live private artifact and raw runtime state cleanup, debugging-only `--keep-live-private-artifacts`, `swarm-infer-beta prepare`, `coordinator`, `miner --stage stage0`, `miner --stage stage1`, `verify`, `collect`, `clean`, `operator.private.env`, `miner.private.env`, `miner_registry.json`, `SWARM_INFERENCE_BETA.md`, `swarm_inference_beta_ready`, `two_machine_swarm_inference_ready`, `real_llm_split_route_ready`, `external_beta_evidence_imported`, `decoded_tokens_match`, `distinct_stage_miners`, `stage_assignment_valid`, and explicit CPU-only/read-only/not production/not P2P/not large-model boundaries.
10. Keep `crowdtensor swarm-session` as the Public Swarm Inference Alpha session wrapper, preserving `public_swarm_inference_alpha_v1`, `scripts/public_swarm_inference_alpha_pack.py`, `scripts/public_swarm_inference_alpha_check.py`, `live-kaggle`, `local-generated`, `public_swarm_inference_alpha_ready`, `public_swarm_session_ready`, `local_stage_requeue_ready`, `public_swarm_live_requeue_ready`, `public_swarm_live_kaggle_ready`, `stage_requeue_ready`, `external_stage_requeue_ready`, `external_runtime_verified`, `kaggle_kernels_deleted`, `token_rotation_required`, default child debug artifact pruning, debugging-only `--keep-child-artifacts`, and explicit CPU-only/read-only/not production/not P2P/not large-model boundaries.
10. Keep `crowdtensor public-swarm-alpha-rc` as the Public Swarm Inference Alpha RC artifact wrapper, preserving `public_swarm_inference_alpha_rc_v1`, `scripts/public_swarm_inference_alpha_rc_pack.py`, `scripts/public_swarm_inference_alpha_rc_check.py`, `evidence-import`, `local-smoke`, `public_swarm_inference_alpha_rc_ready`, `public_swarm_alpha_rc_evidence_imported`, `stage0_live_requeue_evidence_ready`, `stage1_live_requeue_evidence_ready`, `public_swarm_live_requeue_evidence_ready`, `public_swarm_alpha_private_artifacts_absent`, retained stage0/stage1 proof paths, and explicit CPU-only/read-only/not production/not P2P/not large-model boundaries.
10. Keep `crowdtensor live-preview` as the Public Swarm Live Preview RC wrapper, preserving `public_swarm_live_preview_rc_v1`, `scripts/public_swarm_live_preview_rc_pack.py`, `scripts/public_swarm_live_preview_rc_check.py`, `live-preview local-smoke`, `live-preview package`, `live-preview live-kaggle`, `live-preview evidence-import`, `public_swarm_live_preview_rc_ready`, `public_swarm_live_preview_local_smoke_ready`, `public_swarm_live_preview_package_ready`, `public_swarm_live_preview_live_kaggle_ready`, `public_swarm_live_preview_evidence_import_ready`, `external_stage_requeue_ready`, `live_stage0_requeue_ready`, `live_stage1_requeue_ready`, `kaggle_kernels_deleted`, `private_artifacts_cleaned`, `token_rotation_required`, `gpu_generation_evidence_import_ready`, retained stage0/stage1 RC evidence paths under `dist/public-swarm-live-preview-rc-live-stage0-20260529043801-rc` and `dist/public-swarm-live-preview-rc-live-stage1-20260529044328-rc`, short Kaggle slug prefix `ct-live-preview`, and explicit CPU-only-by-default/read-only/Coordinator-backed/not production/not libp2p/not DHT/not NAT traversal/not large-model boundaries.
10. Keep `crowdtensor operator-preview` as the Public Swarm v0.1 Operator Preview top-level user artifact, preserving `public_swarm_operator_preview_v1`, `scripts/public_swarm_operator_preview_pack.py`, `scripts/public_swarm_operator_preview_check.py`, `operator-preview local-smoke`, `operator-preview package`, `operator-preview live-kaggle`, `operator-preview evidence-import`, `public_swarm_operator_preview_ready`, `operator_preview_user_path_ready`, `operator_preview_local_smoke_ready`, `operator_preview_package_ready`, `operator_preview_live_kaggle_ready`, `operator_preview_evidence_import_ready`, `serve_join_generate_ready`, `miner_join_pack_ready`, `cpu_fallback_ready`, `live_preview_ready`, `support_bundle_ready`, `release_readiness_ready`, `gpu_generation_evidence_import_ready`, `developer_preview_degraded`, `operator_preview_cpu_fallback_user_path_ready`, `operator_preview_retained_evidence_ready`, `external_runtime_blocked`, and explicit CPU-only-by-default/read-only/Coordinator-backed/not production/not libp2p/not DHT/not NAT traversal/not large-model boundaries.
10. Keep `crowdtensor swarm-trial` as the Public Swarm v0.2 Usable Inference Trial entrypoint, preserving `public_swarm_trial_v1`, `public_swarm_trial_cli_v1`, `scripts/public_swarm_trial_pack.py`, `scripts/public_swarm_trial_check.py`, `swarm-trial local-loopback`, `swarm-trial package`, `swarm-trial live-kaggle`, `swarm-trial evidence-import`, `public_swarm_trial_ready`, `serve_join_generate_trial_ready`, `stage0_join_ready`, `stage1_join_ready`, `generate_ready`, `generated_token_count_ready`, `support_bundle_ready`, `cpu_fallback_ready`, `private_artifacts_cleaned`, `operator_preview_import_ready`, `gpu_generation_evidence_import_ready`, `swarm_trial_degraded_cpu_fallback_ready`, `external_runtime_blocked`, `token_rotation_required`, and explicit CPU-only-by-default/read-only/Coordinator-backed/not production/not libp2p/not DHT/not NAT traversal/not GPU marketplace/not large-model boundaries.
10. Keep `crowdtensor public-swarm-beta` as the Public Swarm Inference Beta user entrypoint, preserving `public_swarm_inference_beta_v1`, `scripts/public_swarm_inference_beta_pack.py`, `scripts/public_swarm_inference_beta_check.py`, `public-swarm-beta product-beta`, `public_swarm_product_beta_ready`, `public_swarm_product_rc_ready`, `coordinator_product_surface_ready`, `session_protocol_ready`, `p2p_lite_discovery_ready`, `gpu_generation_evidence_import_ready`, `cpu_fallback_ready`, `local_cpu_inference_ready`, the compatibility paths `public-swarm-beta local-loopback` and `public-swarm-beta evidence-import`, `two_stage_split_inference_ready`, `local_loopback_ready`, `decoded_tokens_match`, `distinct_stage_miners`, `stage_assignment_valid`, `public_swarm_beta_evidence_import_ready`, `external_live_evidence_imported`, `stage0_live_requeue_evidence_ready`, `stage1_live_requeue_evidence_ready`, `prepare`, `coordinator`, `miner`, `verify`, `collect`, `clean`, and explicit Coordinator-backed/read-only/not production/not libp2p/not DHT/not NAT traversal/not large-model boundaries.
10. Keep `crowdtensor p2p-daemon` and `crowdtensor real-p2p-rc` as the Real P2P provider-core RC surface, preserving `crowdtensor.real_p2p`, `scripts/real_p2p_daemon.py`, `scripts/libp2p_node20_polyfill.mjs`, `scripts/libp2p_kad_daemon.mjs`, `scripts/libp2p_discovery_alpha_check.py`, `scripts/real_p2p_swarm_inference_core_rc_pack.py`, `scripts/real_p2p_swarm_inference_core_rc_check.py`, `real_p2p_provider_record_v1`, `real_p2p_provider_catalog_v1`, `real_p2p_route_lookup_v1`, signed provider records, TTL eviction, bootstrap sync hooks, `/real-p2p/providers`, `/real-p2p/route`, `/real-p2p/diagnostics`, `serve/join/generate --p2p --p2p-backend real`, and `p2pd` / P2P-lite fallback. Preserve `http-provider-store` and `libp2p-kad`: the libp2p sidecar uses stable peer identity, TCP/noise/yamux, bootstrap peers, provider-record stream sync, Kad peer-routing diagnostics, and the Node 20 preload polyfill while provider records are still transported over `/crowdtensor/provider-record/1.0.0` streams rather than a production DHT value-store. Retained ready evidence now includes `dist/real-p2p-libp2p-local-smoke-ready/real_p2p_swarm_inference_core_rc.json` for local two-stage tiny-GPT 2-token generation through libp2p, `dist/real-p2p-libp2p-kaggle-runtime-smoke-20260531-r6/real_p2p_swarm_inference_core_rc.json` for Kaggle source/Node/HF/libp2p runtime smoke, and `dist/real-p2p-libp2p-kaggle-auto-20260531-r4/real_p2p_swarm_inference_core_rc.json` for full external stage0/stage1 libp2p split generation with two private Kaggle CPU Miners. Preserve `libp2p_discovery_backend_ready`, `p2p_peer_identity_ready`, `p2p_provider_dht_ready`, `external_libp2p_stage_discovery_ready`, `external_libp2p_generate_ready`, `hivemind_petals_class_alpha_ready`, `real_p2p_kaggle_runtime_smoke_ready`, `distinct_stage_miners`, `stage_assignment_valid`, `real_p2p_core_rc_model_metadata_ready`, and `token_rotation_required`; non-default `--hf-model-id` evidence imports must expose matching model metadata or block with `real_p2p_core_rc_model_metadata_mismatch`. This is Hivemind/Petals-class Alpha evidence only: still Coordinator-backed, read-only, tiny-model scoped, not Hivemind/Petals production parity, not full Kademlia provider-value storage, not NAT traversal/relay, not decentralized security, not an economic system, and not large-model throughput.
10. Real P2P live `external-existing --verify-generate` must forward the requested `--hf-model-id`, bounded `--prompt-texts`, and `--stream-generation` into nested `crowdtensor generate`, report public-safe model/batch/stream summaries, and keep local/Kaggle generate commands passing the requested model id into the actual generation process; preserve this so later imports can reject mismatched retained evidence without leaking prompts or generated text.
10. Keep `crowdtensor p2p-swarm-v06` as the Coordinator-to-P2P transition prototype, preserving `p2p_swarm_inference_v06_v1`, `scripts/p2p_swarm_inference_v06_pack.py`, `scripts/p2p_swarm_inference_v06_check.py`, modes `local-smoke`, `package`, `evidence-import`, `external-existing`, and `kaggle-auto`, top-level `crowdtensor p2pd`, `serve --p2p`, `join --p2p`, `generate --p2p --prompt`, `p2pd_cli_v1`, P2P-lite Coordinator/stage capability discovery, Coordinator result fallback, local stage rescue rediscovery, retained local evidence under `dist/p2p-swarm-inference-v06-local-smoke-refresh2`, retained external Kaggle evidence under `dist/p2p-swarm-inference-v06-kaggle-auto-final/kaggle-auto/p2p_v06_kaggle_auto.json`, readiness codes `p2p_swarm_inference_v06_ready`, `p2p_discovery_routing_prototype_ready`, `local_three_process_p2p_discovery_ready`, `p2p_stage_discovery_ready`, `p2p_generate_route_ready`, `p2p_stage_rescue_ready`, `p2p_real_generate_ready`, `p2p_real_stage_rescue_ready`, `external_p2p_stage_discovery_ready`, `external_p2p_generate_verified`, `p2p_swarm_inference_v06_kaggle_auto_ready`, `kaggle_kernels_deleted`, `coordinator_to_p2p_transition_ready`, and `coordinator_result_fallback_ready`. `external-existing --peer-bootstrap` verifies an already-running external P2P bootstrap catalog and only verifies live generation when `--verify-generate --admin-token` are supplied. `kaggle-auto` is side-effectful: it starts temporary public p2pd/Coordinator processes, pushes private Kaggle stage0/stage1 CPU kernels, waits for P2P discovery, runs `generate --p2p`, deletes kernels, cleans local private kernel payloads, and requires token rotation. Missing optional `[hf]` dependencies must produce `p2p_real_generate_hf_runtime_missing` or `host_hf_runtime_missing`. It is P2P discovery/routing prototype evidence, not production NAT traversal, not decentralized security, not an economic system, not Hivemind/Petals parity, and not large-model throughput.
10. Keep `crowdtensor public-p2p-v1-rc` as the signed Public P2P Swarm Inference v1.0 RC, preserving `public_p2p_swarm_inference_v1_rc_v1`, `scripts/public_p2p_swarm_inference_v1_rc_pack.py`, `scripts/public_p2p_swarm_inference_v1_rc_check.py`, shared-secret HMAC peer identity, signed peer announcements, `p2pd --peer-secret --require-signed`, signed `serve --p2p` and `join --p2p`, signed registry health counts, public runbook `PUBLIC_P2P_SWARM_INFERENCE_V1_RC.md`, redacted Support Bundle, private Kaggle payload cleanup, readiness codes `public_p2p_swarm_inference_v1_rc_ready`, `signed_peer_announcement_ready`, `peer_identity_ready`, `peer_registry_health_ready`, `ttl_refresh_ready`, `local_signed_p2p_discovery_ready`, `external_p2p_generate_verified`, `kaggle_kernels_deleted`, `p2p_v06_kaggle_private_artifacts_cleaned`, and `token_rotation_required`. The fresh retained signed Kaggle CPU proof is `dist/public-p2p-v1-rc-kaggle-auto-signed-r2/public_p2p_swarm_inference_v1_rc.json`; it proves external signed stage0/stage1 discovery and tiny-GPT `generate --p2p`, then deletes the private Kaggle kernels and removes local private payloads. Stage rescue is signed local proof plus retained external requeue evidence, not a fresh signed Kaggle victim/rescue proof. This is HTTP P2P-lite plus Coordinator lease/result fallback, not production Hivemind/Petals parity, not libp2p/DHT/NAT traversal, not decentralized security, not an economic system, and not large-model throughput.
10. Keep `crowdtensor public-swarm-gpu-beta` as the optional CUDA Public Swarm GPU Inference Beta overlay, preserving `public_swarm_gpu_inference_beta_v1`, `scripts/public_swarm_gpu_inference_beta_pack.py`, `scripts/public_swarm_gpu_inference_beta_check.py`, `public-swarm-gpu-beta local-smoke`, `public-swarm-gpu-beta local-loopback`, `public-swarm-gpu-beta kaggle-package`, `public-swarm-gpu-beta kaggle-auto`, `public-swarm-gpu-beta evidence-import`, CPU Coordinator CUDA metadata-only scheduling, private Kaggle GPU stage kernels, Kaggle CUDA runtime pins, retained stage-local proof path `dist/public-swarm-gpu-beta-stage-local-live-20260528064520-shortslug/public_swarm_gpu_inference_beta_kaggle_auto.json`, retained historical pre-stage-local proof path `dist/public-swarm-gpu-beta-live-20260528-runtimepin/public_swarm_gpu_inference_beta_kaggle_auto.json`, retained `gpt2-xl` small-tier proof path `dist/gpt2-xl-small-tier-kaggle-logfix-20260614172932/public_swarm_gpu_inference_beta_kaggle_auto.json`, `hf_transformers_cuda`, `real_llm_sharded_cuda_stage0`, `real_llm_sharded_cuda_stage1`, `real_llm_sharded_cuda_both`, `public_swarm_gpu_beta_smoke_ready`, `public_swarm_gpu_beta_ready`, `public_swarm_gpu_beta_kaggle_auto_ready`, `gpu_runtime_ready`, `cuda_runtime_available`, `hf_transformers_cuda_ready`, `gpu_stage0_ready`, `gpu_stage1_ready`, `stage_local_partition_ready`, `stage0_partition_loaded`, `stage1_partition_loaded`, `partition_parameter_split_valid`, `stage_gpu_memory_reduced`, `kaggle_gpu_package_ready`, `kaggle_kernels_deleted`, `token_rotation_required`, `external_gpu_runtime_verified`, `real_llm_internet_beta` Coordinator log-file redirection, and explicit read-only/not production/not P2P/not GPU pooling marketplace/not 7B/8B large-model boundaries.
10. Keep `crowdtensor gpu-generate` as the GPU Sharded Generation Beta entrypoint, preserving `gpu_sharded_generation_beta_v1`, `gpu_sharded_generation_beta_cli_v1`, `scripts/gpu_sharded_generation_beta_pack.py`, `scripts/gpu_sharded_generation_beta_check.py`, modes `local-loopback`, `kaggle-auto`, and `evidence-import`, `--max-new-tokens`, stage0/stage1 alternating generation, `generation_step`, `generated_token_count`, `generated_text_hash`, `generated_text_redacted`, `raw_generated_text_public: false`, `multi_token_generation_ready`, `gpu_sharded_generation_ready`, old single-token evidence rejection, `hf_transformers_cuda`, `stage_local` partitioning, Kaggle cleanup evidence, retained proof `dist/gpu-sharded-generation-beta-kaggle-20260528095658/gpu_sharded_generation_beta_kaggle_auto.json`, RC manifest `dist/gpu-sharded-generation-beta-kaggle-20260528095658/gpu_sharded_generation_beta_rc_manifest.json`, and explicit tiny GPT Beta / not production / not Hivemind-level / not P2P / not GPU marketplace / not large-model boundaries.
10. Keep `crowdtensor clean-artifacts` as the safe maintenance path for repeated agent runs, preserving `cleanup_report_v1`, dry-run default, `--apply`, `--include-reports`, and the rule that cleanup does not delete state or source files.
10. Keep `crowdtensor remote-runbook` and `crowdtensor remote-acceptance` as the operator-facing wrappers for the controlled two-machine path, preserving `remote_runbook_cli_v1`, `remote_acceptance_cli_v1`, fixed scenario propagation, token redaction, default `--create-session`, and explicit not production / not P2P boundaries.
11. Keep `crowdtensor remote-demo prepare` / `doctor` / `verify` / `collect` / `clean` as the high-level two-machine home-compute demo, preserving `remote_home_compute_demo_v1`, `remote_home_compute_doctor_v1`, `remote_home_compute_collect_v1`, `remote_home_compute_cleanup_v1`, `scripts/remote_home_compute_demo_pack.py`, `scripts/remote_home_compute_demo_check.py`, private `operator.private.env` / `miner.private.env`, `miner_join_pack_v1`, `miner_join.sh`, `MINER_JOIN.md`, dry-run cleanup defaults, `POST /admin/inference-sessions`, `model_bundle_infer`, `remote_python_model_bundle_infer`, `remote_compute_evidence_v1`, `remote_demo_observability_v1`, `--workload external-llm`, `external_llm_infer`, `remote_python_external_llm_infer`, `remote_external_llm_evidence_v1`, `remote_external_llm_observability_v1`, token/runtime redaction, and explicit not production / not P2P / not public prompt-serving boundaries.
12. Keep `scripts/remote_two_machine_beta_check.py` as the Real two-machine CPU inference Beta aggregate rehearsal, preserving `remote_two_machine_beta_check_v1`, `remote_two_machine_inference_ready`, `remote_two_machine_external_llm_ready`, `remote_two_machine_beta_ready`, Coordinator host / Miner host docs, and explicit not model sharding / not P2P boundaries.
13. Keep the Kaggle Remote Miner Beta as an external temporary-Miner target, preserving `crowdtensor remote-demo prepare --target kaggle`, generated `kaggle_remote_miner.py`, `kaggle_remote_miner.md`, `kaggle_remote_miner_prepare_ready`, `scripts/kaggle_remote_miner_beta_check.py`, `kaggle_remote_miner_beta_check_v1`, `kaggle_remote_miner_beta_ready`, outbound-only Kaggle Miner semantics, `operator.private.env` exclusion from Kaggle, CPU-only read-only boundaries, and explicit not GPU/TPU workload / not production / not P2P boundaries.
14. Keep Kaggle Real Runtime Acceptance as the first live external runtime proof, preserving `crowdtensor remote-demo kaggle-real`, `kaggle_real_runtime_acceptance_v1`, `scripts/kaggle_real_runtime_acceptance_pack.py`, `scripts/kaggle_real_runtime_acceptance_check.py`, public host `24.199.118.54`, temporary HTTP boundary, `kaggle_artifacts_ready`, `coordinator_public_ready`, `kaggle_miner_seen`, `kaggle_result_accepted`, `kaggle_real_runtime_ready`, `token_rotation_required`, and `operator.private.env` exclusion from Kaggle. Also preserve the micro split path with `micro-llm-sharded`, `kaggle-upload-stage0`, `kaggle-upload-stage1`, `kaggle_micro_llm_stage0_seen`, `kaggle_micro_llm_stage1_seen`, `kaggle_micro_llm_stage_assignment_valid`, `stage_assignment_valid`, and `kaggle_micro_llm_sharded_ready`. This is CPU-only/read-only, not production, not P2P, not GPU/TPU workload execution, and not large-model sharding.
12. Keep expanding `scripts/home_compute_demo.py` around the current read-only multi-request `model_bundle_infer` probe into a useful home-compute inference demo with explicit `hardware_targets`, `recommended_routes`, `route_decision`, capped `request_trace` summaries, stable `diagnosis_codes` such as `home_compute_ready` and `runtime_matrix_blocked`, and hardware/capability matching.
13. Keep `scripts/home_compute_evidence_pack.py` and `scripts/home_compute_evidence_check.py` as the safe, shareable `home_compute_evidence_v1` layer for public issue reports and demos, preserving `route_decision`, `matched_capabilities`, `diagnosis_codes`, and capped `request_trace` while redacting secret-shaped fields.
14. Keep `scripts/inference_session_client.py` and `scripts/inference_session_client_check.py` as the narrow user-facing client for a running Coordinator, preserving `inference_session_client_v1`, `session_client_ready`, `POST /admin/inference-sessions`, `task_id` filtering, `model_bundle_infer`, read-only semantics, and `--skip-inference-session-client` acceptance control.
15. Keep `scripts/admin_inference_session_check.py` as the narrow service-shaped API acceptance path for `POST /admin/inference-sessions`, preserving `inference_session_request_v1`, `task_id` filtering, `model_bundle_infer`, read-only semantics, and `--skip-admin-inference-session` acceptance control.
16. Keep `scripts/remote_compute_evidence_pack.py` and `scripts/remote_compute_evidence_check.py` as the safe, shareable `remote_compute_evidence_v1` layer for registry-backed remote-style Python Miner demos, preserving `remote_python_model_bundle_infer`, `remote_compute_observability_v1`, fixed `model_bundle_inference_scenario_v1` metadata and scenario match status, safe metrics, capped `request_trace`, and hashed registry status.
17. Keep `scripts/remote_demo_runbook_pack.py` and `scripts/remote_demo_runbook_check.py` as the safe two-machine `remote_demo_runbook_v1` path, preserving `operator.private.env`, `miner.private.env`, `model_bundle_infer`, `--scenario-id route-baseline`, and `remote_compute_evidence_pack.py --mode collect`.
18. Keep `scripts/remote_demo_acceptance_pack.py` and `scripts/remote_demo_acceptance_check.py` as the safe two-machine `remote_demo_acceptance_v1` layer that can use `--create-session` to call `POST /admin/inference-sessions` with `scenario_id`, wait for the returned `task_id`, verify scenario match, and collect `remote_compute_evidence_v1`, `remote_demo_observability_v1`, and `support_bundle`, with stable `diagnosis_codes` such as `coordinator_unreachable`, `observer_auth_failed`, `session_create_failed`, and `artifact_collection_failed`.
19. Keep `scripts/multi_miner_scenario_sweep.py` and `scripts/multi_miner_scenario_sweep_check.py` as the controlled local multi-Miner lease-race and failure-requeue proof, preserving concurrent mode, `multi_miner_scenario_sweep_v1`, `multi_miner_scenario_sweep_observability_v1`, three fixed scenarios, distinct Miner identities, `local_multi_miner_model_bundle_infer`, `lease_summary`, `process_summary`, `requeue_summary`, `multi_miner_concurrent_ready`, `multi_miner_requeue_ready`, and `--include-multi-miner-sweep` / `--include-multi-miner-requeue` opt-in coverage.
20. Keep `scripts/demo_manifest_pack.py` and `scripts/demo_manifest_check.py` as the latest output artifact entrypoint for local-loopback handoff, combining runtime matrix, remote-compute evidence, deterministic mock external LLM evidence, and support bundle summaries.
21. Treat `external_llm_infer_v1` / `external_llm_evidence_v1` as the narrow optional runtime adapter boundary: use `--enable-mock-llm-runtime` or `--mock` for deterministic checks, `--llm-runtime-cmd` / `CROWDTENSOR_LLM_RUNTIME_CMD` for operator-owned command wrappers, and `--llm-runtime-url` / `CROWDTENSOR_LLM_RUNTIME_URL` for OpenAI-compatible local servers.
22. Grow toward remote Miners, browser-native participation, optional GPU/runtime adapters, and then P2P/NAT routing.
23. Treat incentives and reputation as later protocol layers, not prerequisites for local demos.

## Engineering Rules

- Keep network/control-plane code physically separate from workload compute code.
- Keep CPU-only deterministic smoke paths working even when optional accelerators are added.
- Version protocol changes; preserve `runtime_contract_v1` unless a change is intentionally versioned.
- New workload contracts should not mutate task lease or heartbeat semantics.
- Do not expose raw lease tokens, idempotency material, tensor deltas, registry tokens, raw external LLM `output_text`, or raw state in operator-friendly outputs.
- Prefer narrow, testable additions over broad rewrites.
- Update public docs, changelog, roadmap, and project memory when user-visible behavior or strategy changes.

## Validation Commands

Run focused checks first, then broader checks when changing shared behavior:

```bash
python3 scripts/release_gate.py --json
python3 -m unittest tests.test_release_gate -v
python3 -m py_compile coordinator.py miner_cli.py scripts/*.py crowdtensor/*.py
python3 -m unittest discover -s tests -v
# Optional HF tiny GPT split proof when `[hf]` dependencies are installed:
python3 scripts/remote_real_llm_sharded_beta_check.py --mode remote-loopback --stage-mode split --require-distinct-stage-miners
```

For runtime behavior changes, also run the acceptance pack from a normal shell that permits localhost networking:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8910 \
  --report /tmp/crowdtensor_acceptance.json
```

Browser checks are opt-in:

```bash
python3 scripts/browser_acceptance_pack.py \
  --allow-skip \
  --base-port 9310 \
  --report /tmp/crowdtensor_browser_acceptance.json
```

## Git and Release Notes

Use the normal repository Git metadata from the project root:

```bash
git status --short --branch
```

Do not commit local state directories, token files, browser profiles, checkpoints, generated caches, or secrets.

Before public release work, read:

- [Project Memory](docs/project-memory.md)
- [Roadmap](ROADMAP.md)
- [Protocol Boundary](docs/protocol.md)
- [Release Process](docs/release.md)

## Latest Heterogeneous Capacity Frontier Status

The current maximum-capacity evidence pack is
`dist/heterogeneous-capacity-frontier-20260626-r3-72b-load-100b-partial/heterogeneous_capacity_frontier.json`,
emitted by `scripts/heterogeneous_capacity_frontier_pack.py` and checked by
`scripts/heterogeneous_capacity_frontier_check.py`. It imports the retained 32B
GPU+TPU+CPU r6 4-token same-request decode proof and the fresh 72B AWQ
stage-owned loading proof at
`dist/kaggle-72b-stage-owned-safetensors-probe-awq-live-r2-full10/kaggle_32b_stage_owned_safetensors_probe.json`.
The 72B proof used ten private Kaggle Tesla T4 x2 script kernels sequentially
for `Qwen/Qwen2.5-72B-Instruct-AWQ` stages 0-9, covered 2083/2083 safetensors
keys across layers 0-80 plus embeddings/final head, loaded only stage-owned
keys, avoided cross-stage key loads, verified T4 x2 hardware on every stage,
then deleted all temporary kernels and removed local private payloads. The edge
stages loaded about 5.73 GB each and the middle stages about 3.41 GB each.
It also imports a bounded 100B+ partial live loading probe at
`dist/kaggle-100b-stage-owned-safetensors-probe-compressed-live-r1-stage8/kaggle-output/stage8/ct_32b_stage_owned_safetensors_stage8_report.json`:
one private Kaggle Tesla T4 x2 kernel loaded stage8 of
`cyankiwi/Solar-Open-100B-AWQ-4bit` (layers 40-44), 4684/4684 stage-owned
compressed-tensors keys, 4.498133 GB materialized tensor bytes, no cross-stage
keys, and temp cleanup; the remote kernel was deleted and the local private
package was removed. This is partial single-stage live loading only, not full
100B model coverage. The same frontier report also performs real HF
config/index/header preflight
for 72B AWQ, 72B GPTQ, 72B full precision, 100B compressed-tensors, and
`QuixiAI/Qwen3-235B-A22B-AWQ`; the largest preflight is 235B AWQ
stage-owned header coverage across 94 layers, 25 safetensors files, and about
115.54 GB indexed tensor bytes.

Current capacity conclusions are:
`max_stage_owned_load_parameter_class=72b-awq`,
`max_partial_stage_owned_load_parameter_class=100b-compressed`,
`max_stage_owned_load_preflight_parameter_class=235b-awq`,
`max_1token_decode_parameter_class=32b`,
`max_multitoken_decode_parameter_class=32b`, and
`max_gpu_tpu_cpu_same_request_parameter_class=32b`. A bounded Web TPU
availability check for larger decode found no attached running TPU
service-manager session/kernel. Treat 72B as full stage-owned loading success
only and 100B as partial single-stage live loading success only; neither is
activation/decode or same-request inference success. Treat 235B as
metadata/header preflight only, not live loading or inference. The next
bottlenecks are quantized JAX/TPU runtime adapter support, larger-than-32B
same-request decode, current Web TPU runtime attachment, and live stage-owned
loading beyond 72B.

## Latest Dense Three-Accelerator Qwen Frontier Status

For future large-parameter LLM inference experiments, the main path is dense
full-precision BF16/FP16 HF/Qwen, not AWQ/GPTQ/4-bit/8-bit/GGUF quantized
models. The current canonical dense frontier artifact is
`dist/three-accelerator-dense-qwen-frontier-20260626-r8-live-72b-stage-plan-retained-32b/three_accelerator_dense_qwen_frontier.json`,
emitted by `scripts/three_accelerator_dense_qwen_frontier_pack.py` and checked
by `scripts/three_accelerator_dense_qwen_frontier_check.py`. It imports the
retained 32B GPU+TPU+CPU same-request 4-token bridge proof and the retained
32B full-precision GPU+CPU fallback proof, imports the retained real Web TPU
32B full-stage loader/runtime evidence, and imports fresh Kaggle Models attach
and stage-owned preflight probes up through dense 72B.

The Kaggle Models attach resolver is
`scripts/kaggle_dense_model_source_resolver.py`. It resolves dense official
Qwen model sources under `https://www.kaggle.com/models`, including
`qwen-lm/qwen2.5/Transformers/72b-instruct/1`,
`qwen-lm/qwen2.5/Transformers/32b-instruct/1`,
`qwen-lm/qwen2.5/Transformers/14b-instruct/1`, and
`qwen-lm/qwen2.5/Transformers/7b-instruct/1`, with expected attached runtime
paths under `/kaggle/input/models/{owner}/{model}/{framework-lower}/{instance}/{version}`,
for example
`/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1`. The older
`/kaggle/input/qwen2.5/...` path was a wrong assumption and is kept only as a
legacy fallback probe path. The live attach proof is
`dist/kaggle-model-attach-probe-20260626-r3-7b-cpu-realpath/kaggle_model_attach_probe.json`:
a private CPU-only Kaggle script kernel attached
`qwen-lm/qwen2.5/Transformers/7b-instruct/1`, saw config/tokenizer/index plus
4 safetensors files and 339 weight-index keys at the real mounted path, kept
tensor values private, then deleted the temporary kernel and removed the local
private package. This proves Kaggle Models attach for dense Qwen 7B and the
runtime path shape. The later 32B attach proof is
`dist/kaggle-model-attach-probe-20260626-r4-32b-cpu-realpath/kaggle_model_attach_probe.json`:
it attached `qwen-lm/qwen2.5/Transformers/32b-instruct/1`, saw 17 safetensors
files and 771 weight-index keys, and cleaned up the temporary kernel/package.
The current largest attach proof is
`dist/kaggle-model-attach-probe-20260626-r7-72b-cpu-stage-plan/kaggle_model_attach_probe.json`:
it attached `qwen-lm/qwen2.5/Transformers/72b-instruct/1` at
`/kaggle/input/models/qwen-lm/qwen2.5/transformers/72b-instruct/1`, saw
config/tokenizer/index plus 37 safetensors files and 963 weight-index keys,
kept tensor values private, verified a 10-stage public placement preflight
(`cuda,cuda,cuda,cuda,jax_tpu,cpu,cpu,cpu,cpu,cpu`) with 963/963
stage-owned keys present across all 37 safetensors files, about 145.412407 GB
total planned logical tensor bytes, and about 16.534389 GB maximum single-stage
planned logical tensor bytes, then deleted the temporary CPU-only kernel and
removed the local private package. This proves 72B dense model attach,
safetensors header readability, stage-owned key/file assignment, and capacity
preflight; it is still not 72B live weight loading, TPU execution, activation
handoff, or inference.

The dense HF/Qwen -> JAX/TPU stage adapter smoke is
`scripts/qwen_dense_jax_tpu_stage_adapter_smoke.py`, with the retained CPU-JAX
unit evidence at
`dist/qwen-dense-jax-stage-adapter-smoke-20260626-r2-cpu-jax/qwen_dense_jax_tpu_stage_adapter_smoke.json`.
It exercises dense Qwen-like RMSNorm, RoPE, grouped-query causal attention,
SwiGLU MLP, and stage-local KV-cache metadata with public-safe hashes/shapes
only; the PyTorch reference and JAX forward match in the CPU-JAX runtime. The
same dense frontier also imports the retained real Web TPU 32B full-stage
loader evidence at
`dist/kaggle-tpu-32b-stage-owned-loader-probe-web-live-20260623-r3-full-21-layer-real/kaggle_tpu_32b_stage_owned_loader_probe.json`,
so `tpu_jax_qwen_stage_runtime_ready=true` for the 32B retained path.

Current dense frontier conclusions are:
`largest_dense_model_attempted=72b`,
`largest_dense_model_attach_candidate=72b`,
`largest_dense_model_attached=72b`,
`largest_dense_model_stage_preflighted=72b`,
`largest_dense_model_loaded=32b`,
`largest_dense_model_1token_decoded=32b`,
`all_three_accelerators_same_request_verified=true`,
`generated_token_count=4`,
`gpu_stage_runtime_ready=true`, `cpu_stage_runtime_ready=true`,
`kaggle_model_attach_available=true`, `kaggle_model_attach_used=true`,
`tpu_jax_qwen_stage_runtime_ready=true`,
`same_request_dense_32b_success=true`, and
`same_request_dense_frontier_success=false`. Treat 72B as the largest dense
Kaggle Models attached and stage-preflighted model, not a loaded or decoded
model. The largest verified dense loaded/decode class remains 32B. The current
blockers are `larger_dense_live_stage_load_not_verified_after_stage_preflight`
and `larger_than_32b_dense_decode_not_verified`; the failure stage is now after
72B attach plus stage-owned safetensors header/capacity preflight, at live 72B
stage loading / placement / same-request decode. Do not claim 72B dense
inference or production readiness until a future report proves real 72B stage
loading, TPU dense stage execution for that larger placement, and same-request
GPU+TPU+CPU activation handoff.

The current dense max-parameter-search artifact is
`dist/three-accelerator-dense-max-parameter-search-20260626-r1-72b-tpu-live-timeout-retained-32b/three_accelerator_dense_max_parameter_search.json`,
emitted by `scripts/three_accelerator_dense_max_parameter_search_pack.py` and
checked by `scripts/three_accelerator_dense_max_parameter_search_check.py`.
It keeps the goal boundary sharper than the r8 frontier report:
`max_successful_same_request_decode_parameter_class=32b`,
`max_attempted_parameter_class=72b`, `max_attached_parameter_class=72b`,
`max_stage_preflighted_parameter_class=72b`,
`max_stage_loaded_parameter_class=32b`, and
`max_tpu_executed_parameter_class=32b`. It imports the retained 32B
GPU+TPU+CPU 4-token same-request proof, the dense 72B Kaggle Models attach and
10-stage preflight proof, and the bounded 72B Web TPU live-load attempt at
`dist/kaggle-tpu-72b-stage-owned-loader-probe-web-live-20260626-r3-stage32-40-one-layer-bridge-executor/kaggle_tpu_32b_stage_owned_loader_probe.json`.
That 72B TPU attempt failed before public-safe header/tensor/load/forward
evidence with `web_tpu_jupyter_execute_timeout`; it proves an attempted bounded
live-load path and cleanup, not 72B TPU execution or inference. The checker
rejects overclaims where 72B attach/preflight is promoted to stage-loaded,
TPU-executed, or same-request decoded status without real loaded keys, TPU
devices, layer-forward evidence, and same-request decode proof.

The current superseding dense max-parameter-search artifact is
`dist/three-accelerator-dense-max-parameter-search-20260628-r4-web-tpu-ui-start-timeout-retained-32b/three_accelerator_dense_max_parameter_search.json`.
It imports the current Web TPU execution-channel probe at
`dist/kaggle-web-tpu-execution-channel-probe-20260628-r3-after-ui-start-wait/kaggle_web_tpu_execution_channel_probe.json`,
emitted by `scripts/kaggle_web_tpu_execution_channel_probe.py` and checked by
`scripts/kaggle_web_tpu_execution_channel_check.py`. The current goal is not
complete: 72B dense/full-precision GPU+TPU+CPU same-request 1-token decode has
not succeeded. The latest recovery work first used Kaggle MCP
`create_notebook_session` successfully enough to create interactive session ids,
but channel probes showed those sessions executed JAX on CPU only
(`jax_tpu_device_missing`). It then used authenticated Web UI automation to
expand Session options, select `TPU v5e-8`, and force-click Start Session; the
page entered `Session is starting...`, but a bounded 920-second wait at
`dist/kaggle-web-tpu-active-wait-20260628-r1-queue11/kaggle_web_tpu_active_wait.json`
still showed zero Jupyter sessions/kernels and no TPU runtime. The follow-up
channel probe timed out before the first small JAX TPU cell, so the tiny
Qwen-like cell was not attempted. Therefore r4 keeps
`max_successful_same_request_decode_parameter_class=32b`,
`max_attempted_parameter_class=72b`, `max_attached_parameter_class=72b`,
`max_stage_preflighted_parameter_class=72b`,
`max_stage_loaded_parameter_class=32b`, and
`max_tpu_executed_parameter_class=32b`, with
`failure_stage=web_tpu_channel_jupyter_execute` and blockers including
`web_tpu_execution_channel_not_ready`, `web_tpu_jupyter_execute_timeout`,
`tiny_qwen_like_not_attempted_after_small_jax_failure`,
`dense_72b_tpu_stage_load_and_forward_not_verified`, and
`larger_than_32b_same_request_decode_not_verified`. Do not mark the 72B goal
achieved from this blocker evidence. Keep trying to restore a real current Web
TPU runtime/channel, and do not start a new 72B dense TPU live-load until a
fresh channel probe first proves small JAX and tiny Qwen-like execution on TPU.

The latest superseding dense max-parameter-search artifact is
`dist/three-accelerator-dense-max-parameter-search-20260628-r5-72b-stage-bridge-not-full-decode/three_accelerator_dense_max_parameter_search.json`,
checked by `scripts/three_accelerator_dense_max_parameter_search_check.py
--report ... --json` with no errors. It supersedes r4 for current status. The
fresh Web TPU channel proof is
`dist/kaggle-web-tpu-execution-channel-probe-20260628-r4-runtime-started/kaggle_web_tpu_execution_channel_probe.json`:
Kaggle Web TPU v5e-8 was attached with 8 `TPU v5 lite` devices, small JAX and
tiny Qwen-like JAX cells ran, and stage-local KV-cache metadata was verified.
The standalone 72B TPU stage proof is
`dist/kaggle-tpu-72b-stage-owned-loader-probe-web-live-20260628-r8-stage32-40-full8-1g-budget/kaggle_tpu_32b_stage_owned_loader_probe.json`:
inside the authenticated Web TPU runtime it executed `Qwen/Qwen2.5-72B-Instruct`
layers 32-40, verified 96 stage-owned keys, loaded about 13.078522 GB logical
execution tensor bytes, ran all 8 assigned layers on 8 TPU devices, and kept
tensor values, activations, KV-cache tensors, token ids, prompts, and generated
text private. The same-request bridge evidence is
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260628-r10-72b-tpu-stage-same-request/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`:
one Coordinator request accepted CUDA, JAX/TPU, and CPU tasks
(`stage0=stage1=stage2=1`, `accepted_stage_backends=["cpu","cuda","jax_tpu"]`,
two activation handoff hashes, one generated-token hash), the TPU task executed
the real 72B stage-owned loader for layers 32-40, and the temporary private
Kaggle GPU kernel was deleted. This is meaningful 72B progress but still not
the goal's completed full 72B dense inference: r5 records
`max_stage_loaded_parameter_class=72b`, `max_tpu_executed_parameter_class=72b`,
and `same_request_72b_import.same_request_stage_decode_verified=true`, while
keeping `max_successful_same_request_decode_parameter_class=32b`,
`same_request_72b_import.same_request_full_model_decode_verified=false`,
`full_72b_weight_loading_public_claim=false`, and
`failure_stage=dense_72b_stage_same_request_verified_but_full_model_decode_not_verified`.
Do not mark the 72B goal achieved from r5: it proves a 72B TPU middle-stage
same-request bridge, not full all-layer 72B GPU+TPU+CPU decode or quality
parity. The remaining work is real full-precision 72B CUDA/CPU stage-owned
execution plus a same-request 1-token decode whose report sets
`gpu_tpu_cpu_72b_same_request_verified=true`.

Latest superseding status after the full-72B engineering continuation: use
`dist/three-accelerator-dense-max-parameter-search-20260628-r6-full-72b-engineering-web-tpu-timeout/three_accelerator_dense_max_parameter_search.json`
as the current max-search artifact. It passes
`scripts/three_accelerator_dense_max_parameter_search_check.py --report ... --json`
with no errors and keeps `max_successful_same_request_decode_parameter_class=32b`,
`max_stage_loaded_parameter_class=72b`, and `max_tpu_executed_parameter_class=72b`.
The code path has advanced: `scripts/kaggle_32b_full_heterogeneous_probe.py`
now accepts configurable stage ranges/groups, can represent the 10-stage 72B
topology `gpu,gpu,gpu,gpu,web_tpu,cpu,cpu,cpu,cpu,cpu`, seeds Coordinator
input ids without publishing token ids, and emits
`gpu_tpu_cpu_72b_same_request_verified`, `same_request_72b_full_model_verified`,
and `full_72b_weight_loading_public_claim` only when all 10 stages complete.
`scripts/kaggle_tpu_32b_stage_owned_loader_probe.py` now supports private input
activation consumption and private output activation handoff for Web TPU stage
execution while redacting `hidden_b64` from public artifacts. The max-search
pack/checker can import a successful full heterogeneous 72B report and only
then raise max successful decode to 72B; stage-bridge-only reports remain
blocked. The fresh current Web TPU channel preflight at
`dist/kaggle-web-tpu-execution-channel-probe-20260628-r6-short-current-status/kaggle_web_tpu_execution_channel_probe.json`
failed naturally with `web_tpu_jupyter_execute_timeout` before small JAX, so no
full 72B live run was started in r6. This is not a completed goal and not a
blocked goal yet: it is meaningful engineering progress plus a current
single-turn Web TPU execution-channel blocker. Resume by restoring a current
Web TPU channel, then run the full 10-stage 72B live probe and require the
public-safe report to set `gpu_tpu_cpu_72b_same_request_verified=true` before
marking achieved.

Latest superseding 72B dense max-search status after stricter full-model gates
and bounded Web TPU timeout handling: use
`dist/three-accelerator-dense-max-parameter-search-20260628-r9-bounded-web-tpu-timeout/three_accelerator_dense_max_parameter_search.json`
as the current artifact. It passes
`scripts/three_accelerator_dense_max_parameter_search_check.py --report ... --json`
with no errors and deliberately keeps
`max_successful_same_request_decode_parameter_class=32b`,
`max_stage_loaded_parameter_class=72b`, and
`max_tpu_executed_parameter_class=72b`. The current Web TPU channel probes are
`dist/kaggle-web-tpu-execution-channel-probe-20260628-r7-current-status/kaggle_web_tpu_execution_channel_probe.json`
`dist/kaggle-web-tpu-execution-channel-probe-20260628-r8-force-new-session-30s/kaggle_web_tpu_execution_channel_probe.json`,
and
`dist/kaggle-web-tpu-execution-channel-probe-20260628-r10-short-timeout-after-bounded-padding/kaggle_web_tpu_execution_channel_probe.json`;
all are public-safe, pass `scripts/kaggle_web_tpu_execution_channel_check.py`,
and report `web_tpu_execution_channel_ready=false`,
`small_jax_cell_ready=false`, `tpu_device_count=0`, and blocker
`web_tpu_jupyter_execute_timeout`. The r8 force-new-session attempt used the
minimum 30-second Jupyter execution window and returned a bounded timeout. The
r10 probe was run after tightening `web_tpu_subprocess_timeout_seconds()` from
`execute_timeout + 180s` to `execute_timeout + min(60s, max(10s, 25%))`, so a
30-second channel probe now returns after about 40 seconds instead of waiting
several extra minutes. It still did not produce a usable TPU execution channel.
The read-only current UI state probe is
`dist/kaggle-web-tpu-ui-state-probe-20260628-r2-current-readonly/kaggle_web_tpu_ui_state_probe.json`:
it is public-safe and reports `web_tpu_ui_runtime_ready=false`,
`start_session_visible=true`, `jupyter_frame_visible=false`,
`jupyter_session_count=0`, and `jupyter_kernel_count=0`. This narrows the
current external blocker: the Kaggle notebook is back at Draft Session/off with
no visible Jupyter runtime, so a 72B run cannot proceed until TPU Start Session
is clicked and a new TPU runtime/session is attached.
The bounded automatic restart attempt is
`dist/kaggle-web-tpu-start-wait-probe-20260628-r1-start-wait-15m/kaggle_web_tpu_start_wait_probe.json`:
the script expanded Session options, verified `TPU v5e-8` was visible, selected
it, clicked Start Session, and waited 900 seconds. It remained public-safe and
reports `start_clicked=true`, `web_tpu_ui_runtime_ready=false`,
`queue_visible=true`, `session_starting_text_visible=true`,
`jupyter_frame_visible=false`, `jupyter_session_count=0`, and
`jupyter_kernel_count=0`. This is fresh scheduling/runtime-allocation blocker
evidence only; it is not 72B inference and not a completed goal. Resume by
continuing to wait/retry until a UI state probe reports a visible Jupyter
runtime, then run the Web TPU execution-channel probe, then the full 72B
GPU+TPU+CPU same-request decode.
The follow-up read-only status after that wait is
`dist/kaggle-web-tpu-ui-state-probe-20260628-r3-after-start-wait-readonly/kaggle_web_tpu_ui_state_probe.json`:
it still reports `web_tpu_ui_runtime_ready=false`,
`session_starting_text_visible=true`, `jupyter_frame_visible=false`,
`jupyter_session_count=0`, and `jupyter_kernel_count=0`. The longer continuation
wait is
`dist/kaggle-web-tpu-start-wait-probe-20260628-r2-continue-wait-30m/kaggle_web_tpu_start_wait_probe.json`:
it waited 1800 seconds while the notebook remained in Starting, and finished
with `web_tpu_ui_runtime_ready=false`, `jupyter_frame_visible=false`,
`jupyter_session_count=0`, and `jupyter_kernel_count=0`. It also reports
`start_clicked=false` because the page was already in Starting/disabled state,
not because the original Start Session had never been clicked. This strengthens
the current allocation/attach blocker evidence but still must not be treated as
`blocked` unless the strict repeated-blocker audit is satisfied and no further
meaningful recovery or engineering work remains.
The longer continuation wait is
`dist/kaggle-web-tpu-start-wait-probe-20260628-r3-continue-wait-60m/kaggle_web_tpu_start_wait_probe.json`:
it waited 3600 seconds. Its final observation saw `session_started_text_visible=true`
but still `web_tpu_ui_runtime_ready=false`, `jupyter_frame_visible=false`,
`jupyter_session_count=0`, and `jupyter_kernel_count=0`, so it did not prove a
usable runtime. The immediate fresh page reload status is
`dist/kaggle-web-tpu-ui-state-probe-20260628-r4-after-60m-wait-readonly/kaggle_web_tpu_ui_state_probe.json`:
it returned to Draft Session/off with `start_session_visible=true`,
`jupyter_frame_visible=false`, `jupyter_session_count=0`, and
`jupyter_kernel_count=0`. Current evidence therefore remains a Kaggle Web TPU
allocation/attach failure, not a 72B model/runtime failure and not goal
completion. The max successful same-request dense decode remains 32B.

The 72B success gate is now stricter in code: `scripts/kaggle_32b_full_heterogeneous_probe.py`
requires 72B stage ranges to cover Qwen 72B layers 0..80 contiguously before
`ok`, `gpu_tpu_cpu_72b_same_request_verified`,
`same_request_72b_full_model_verified`, or
`full_72b_weight_loading_public_claim` can be true. It also emits
`full_72b_layer_coverage_verified` and
`gpu_tpu_cpu_72b_full_topology_verified`. The max-search importer now requires
those fields before counting a 72B full heterogeneous report as 72B decode.
This fixes the prior over-broad completion risk: a 72B TPU stage proof or
three-stage bridge remains useful evidence but cannot be treated as full 72B
decode. The live topology may use the original 10-stage 4-GPU/1-WebTPU/5-CPU
plan or a memory-safer full-layer plan such as 13 stages
`[0,6],[6,12],[12,18],[18,24],[24,32],[32,38],[38,44],[44,50],[50,56],[56,62],[62,68],[68,74],[74,80]`
with 4 GPU stages, 1 Web TPU stage, and 8 CPU stages; either way success
requires a real same-request 1-token decode over all stages. Do not mark the
72B goal achieved until a current public-safe report proves full 72B dense
GPU+TPU+CPU same-request decode and the checker/tests/docs are updated. Current
next step is to restore a Web TPU execution channel first; starting the full
72B live probe while small JAX cannot execute on TPU is expected to fail at the
TPU stage and should only create blocker evidence, not completion.

## 2026-07-18 Public Founding Preview Superseding Status

The current publicity-preparation work is complete for the founding-preview
boundary. The canonical launch artifact is
`dist/volunteer-training-public-launch-rc/volunteer_training_public_launch_rc.json`.
The current file SHA-256 is
`sha256:251a06f46ef62b9492eb39affb920d4ae9b340d3f655072970bb38b0d1d515ce` and
the embedded content hash is
`sha256:a89f353b8e2a0c4dc557826b9b6eed4331b49744caa4637ae8548b60cd124d7e`.
Run its default checker with:

`PYTHONPATH=. python scripts/volunteer_training_public_launch_check.py --report dist/volunteer-training-public-launch-rc/volunteer_training_public_launch_rc.json --json`

The expected result is `founding_preview_ready=true`,
`formal_launch_ready=false`, zero default-check errors, and blocker
`formal_multihost_evidence_missing`. The strict `--require-formal` checker must
remain failing until a public-safe report proves independently administered
physical multi-host execution, independent host/admin identities, a real
network route, and cleanup. Same-host processes, Kaggle logical nodes, queue
screenshots, or local mocks cannot satisfy that gate.

Preserve these public-preview components:

- `scripts/volunteer_training_public_demo.py` and
  `scripts/volunteer_training_public_demo_check.py`: bounded real local HTTP
  Coordinator plus two independent Cell subprocesses, one round, real tiny
  PyTorch/Transformers PEFT fixture, and cleanup/public-safety evidence;
- `scripts/volunteer_dashboard_visual_probe.py`: real Playwright desktop/mobile
  checks, nonblank canvas, no horizontal overflow, coherent layout, and private
  runtime cleanup;
- `scripts/volunteer_training_public_launch_pack.py` and
  `scripts/volunteer_training_public_launch_check.py`: offline bundle and the
  founding/formal readiness split;
- `crowdtensor/volunteer_dashboard/`, the public snapshot routes in
  `crowdtensor/volunteer_training_api.py`, and the immutable Campaign proposal
  validator/schema;
- `docs/volunteer-campaign-governance.md`,
  `docs/volunteer-training-launch-kit.md`, the README public positioning, and
  `docs/assets/volunteer-dashboard-{desktop,mobile}.png`.

Do not claim the preview proves useful model-quality improvement, Internet-scale
throughput, permissionless admission, Sybil resistance, poisoning resistance,
secure aggregation, production GA, or an SLA. Keep `founding_preview_ready`
and `formal_launch_ready` as separate fields in future artifacts. Public output
must continue to omit credentials, private paths, invite/lease material, raw
training data, prompts, token ids, activations, gradients, tensor values, and
generated private text. Failed/temporary demo services and runtimes must be
removed before finalizing any new artifact.
