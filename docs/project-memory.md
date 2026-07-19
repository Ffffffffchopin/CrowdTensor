# Project Memory

This document is the long-form durable memory for CrowdTensor. It exists so future development sessions can recover project intent, current facts, and engineering boundaries even when chat context is lost.

## 2026-07-19 Qwen2.5-7B Elastic GSM8K Showcase RC Achieved

The 7B Elastic Domain Fine-tuning Public Showcase RC is achieved. The canonical
self-contained report is
`dist/training-qwen7b-gsm8k-showcase-rc-20260719-r1/training_qwen7b_gsm8k_showcase_rc.json`.
Its file SHA-256 is
`cc737ea87c6336a0aa423891b60f0a8db5095cd49f76ed2c30fe7a3ca2c4197b`
and content hash is
`sha256:454a814b08695176564486eef8bfa345dc7a17dccffd94a445f4b3495f278007`.
The strict checker returns zero errors, `showcase_ready=true`, and
`public_artifact_safe=true`:

`PYTHONPATH=. python scripts/training_qwen7b_gsm8k_showcase_check.py --report dist/training-qwen7b-gsm8k-showcase-rc-20260719-r1/training_qwen7b_gsm8k_showcase_rc.json --require-ready --json`.

The run uses pinned `Qwen/Qwen2.5-7B-Instruct` revision
`a09a35458c702b33eeacc393d103063234e8bc28` (7,615,616,512 parameters,
Apache-2.0) and pinned `openai/gsm8k` revision
`740312add88f781978c0658806c59bc2815b9866` (`main`, MIT). Attempt 3 of 3
completed 256 real LoRA/SFT optimizer steps, four microbatches per step,
sequence length 256, 262,144 non-padding tokens, and 146,659 supervised tokens
at learning rate `2e-5`, rank 4, alpha 8.

Two concurrent Kaggle T4x2 Kernels completed steps 1-128. Both were deleted;
the Coordinator observed a zero-Miner pause; two fresh T4x2 Kernels restored
four stage-owned Adapter/optimizer/GradScaler/RNG checkpoints from central
storage and completed steps 129-256. Exactly-once commits, four distinct Kernel
references and Miner sessions, base-weight freezing, positive gradients, and
complete cleanup are verified. This is Kaggle logical multi-node evidence, not
independently administered physical multi-host evidence.

Attempt integrity is explicit. A disjoint development benchmark showed that
the prior `1e-4` Adapter reduced normalized exact match from 105/128 to 95/128
despite a lower validation loss, so it was rejected. Before attempt 3, a new
128-item confirmatory holdout excluding every development example and the
`2e-5` training contract were preregistered. The checker recomputes both
example-index set hashes and verifies zero overlap.

On the confirmatory holdout, normalized exact match changes from 92/128
(71.875%) to 95/128 (74.219%), an absolute improvement of 2.34375 percentage
points. This passes the preregistered practical +2-point rule. The paired
bootstrap 95% interval is [-6.25, +10.9375] percentage points and includes
zero; do not claim statistical significance. Valid answer rate stays 100%,
strict exact match changes from 88/128 to 95/128, and reserved-train loss
changes from 1.389790 to 0.546368.

The standard PEFT Adapter has 392 tensors over all 28 layers and SHA-256
`2c2cb02961df78976eceec94110ddda830bacce2e68d1e7a3d0abe367005a431`.
It was independently reloaded into an NF4 7B runtime for same-Kernel base and
Adapter evaluation. The RC contains a Model Card, showcase, complete
reproduction commands, minimal ZIP/directory comparison example, and
hash-bound source/dataset/preregistration/development/baseline/training/post/
cleanup evidence.

The cleanup audit verifies three manifests and deletes six hash-matched local
private payloads. All temporary Kaggle Kernels and private Datasets,
Coordinator, tunnel, checkpoint payloads, and runtime-private directories are
removed; `live_resources_left_running=false`. The complete repository suite
passes 2,574 tests, skips two, and fails none. This RC proves a bounded GSM8K
LoRA result only, not broad reasoning, out-of-domain quality, full-parameter or
permissionless training, GA, uptime, or SLA. See
`docs/qwen7b-gsm8k-elastic-showcase.md`.

## 2026-07-18 Volunteer Campaign Single-Host Operator Beta RC Achieved

The Volunteer Campaign Single-Host Operator Beta RC is achieved for its exact
same-physical-host scope. The canonical self-contained report is
`dist/volunteer-training-operator-beta-rc-20260718-r2/volunteer_training_operator_beta_rc.json`;
file SHA-256 is
`ef8ded3cb6ff1822be94bb3928a7f9f1197fdd08656f307daf06c3c2e2be683f`
and content hash is
`sha256:6d5e3ae30c3a03190d4e95b9610a748a20d3e6b35ca9c8485dd4a7d27528e047`.
The strict checker returns zero errors, `goal_achieved=true`, and
`volunteer_campaign_single_host_operator_beta_ready=true`.

Coordinator state automatically migrates from v1 to v2 and durably stores an
Operator policy, signed per-Cell short-lived credential hashes, scope and
revocation state, fixed-window request/upload accounting, total quotas,
credential and active-lease capacity, and replay-nonce hashes. The ordinary
HTTP Cell client automatically exchanges the private enrollment invite for a
Cell-specific credential and sends a fresh nonce on work, artifact, heartbeat,
submission, and resumable-upload requests. Raw credential and nonce values do
not enter public artifacts. The shared invite remains the private Operator
enrollment secret and a compatibility path for earlier direct API callers.

Campaign lifecycle now includes validate, start, pause, resume, finalize,
evaluate, export, private backup, restore, and cleanup. Evaluation is limited
to aggregate training metrics and checkpoint integrity. Export contains the
public Campaign, status, evaluation, ledger, and canonical PEFT Adapter but no
private state. Restore rejects unsafe tar members, rebases durable paths,
reloads the Coordinator, and verifies the content-addressed artifact store and
hash-chained ledger. Prometheus metrics expose progress and bounded fault
counters without per-Cell labels. `crowdtensor volunteer operator` is the
one-command create/reuse/serve path; `crowdtensor volunteer join` remains the
Contributor path.

The canonical r5 live probe is
`dist/volunteer-training-operator-beta-probe-20260718-r5/volunteer_training_operator_beta_probe.json`
(file SHA-256
`31318a20b14dcdbe76f27e70242f25c8dfed7b36d46e612c5bb2eec01809031a`).
One host runs the real Coordinator, a Caddy TLS reverse-proxy container, and a
MinIO S3-compatible container. The probe verifies the TLS certificate
handshake, rejects direct backend HTTP, restarts Coordinator and Caddy while
preserving an active lease, interrupts MinIO during a resumable upload, and
finishes the same upload after restart without retraining. S3 content
addressing and duplicate submission idempotency pass.

The bounded stress gate launches 24 independent OS Cell processes and receives
all 24 terminal reports. Five stress Cells plus the upload-recovery Cell provide
six protocol-fixture updates over three quorum-2 rounds. The stress updates are
not real training and are labelled as such. The RC self-contains the previous
strict Internet Beta evidence for the real PEFT boundary: six independent
SmolLM2-135M/WikiText processes, six optimizer steps, 96 tokens, and canonical
Adapter `v0 -> v3`. This avoids rerunning a proven real workload while keeping
the new stress claim honest.

Fault evidence covers slow-Cell lease expiry, replay, scope, revocation, rate
and capacity rejection, Coordinator/Caddy/MinIO restart, upload resume,
duplicate submit, lifecycle, migration, backup/restore, and monitoring. Cleanup
deletes all temporary processes, containers, S3 objects and bucket, upload
sessions, credentials, and private workspaces. No external accelerator was
used.

The r4 release probe builds a current-source wheel, performs an isolated venv
install with no workspace `PYTHONPATH`, verifies the Volunteer CLI, builds a
current-source project container, verifies its non-root CLI, and removes the
image. It found and fixed an existing packaging defect by moving Miner invite
creation into packaged `crowdtensor/miner_invite.py`. The focused and adjacent
suite passes 77 tests; strict checker mutation
coverage rejects rehashed false TLS evidence.

This is not independent physical multi-host evidence, permissionless trust,
Sybil resistance, semantic poisoning safety, Byzantine consensus, secure
aggregation, useful model quality, full-parameter training, GA, or SLA. The
next external milestone is the same unchanged Operator/Contributor workflow on
independently administered Internet hosts with WAN measurements and independent
reproduction. Do not repeat r5 merely to regenerate local evidence. Operational
usage and exact boundaries are in
`docs/volunteer-training-operator-beta.md`.

## 2026-07-18 Volunteer Training Internet Beta Engineering RC Achieved

The Volunteer Training Internet Beta Engineering RC is achieved for every
planned implementation and local independent-process validation item. The
explicitly excluded physical multi-machine live run remains the sole external
Beta gate. The canonical RC is
`dist/volunteer-training-internet-beta-engineering-rc-20260718-r3/volunteer_training_internet_beta_engineering_rc.json`;
file SHA-256 is
`d00111b453cf24bb6841805bb7524e4647cf698be1c21dd16e12756ff52663b5`
and content hash is
`sha256:d21a8b66690e02135aad12095a735ed0df0c3513404d6a21eec9d9d54090165b`.
The `--require-ready` checker returns zero errors and `goal_achieved=true`.

The new Campaign importer pins the Apache-2.0 SmolLM2-135M revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, hashes all ten imported files,
and uses `smollm2_lora_v1`. It pins WikiText revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, config
`wikitext-2-raw-v1`, licenses CC-BY-SA-3.0/GFDL, and exact train/validation
parquet hashes. Deterministic token rows, raw source text, model weights,
invites, leases, and training tensors stay private.

The r3 probe ran three quorum-2 rounds using six distinct one-command Cell CLI
subprocesses. All six updates used real PyTorch autograd and Transformers/PEFT
LoRA, base weights remained frozen, and canonical Adapter/outer step advanced
`0 -> 3`. A Cell disappeared after claim and its work moved from generation 1
to 2. A network-unavailable CLI attempt recovered against a restarted
Coordinator without changing the lease generation. A 4,938,616-byte upload
stopped after one 64 KiB chunk, survived Coordinator/API recreation, resumed
the same content-addressed session, and submitted without rerunning training.
Two Coordinator restarts verified canonical Adapter, hash-chained ledger,
artifact blobs, and active leases.

The distributed and centralized runs both consumed six optimizer steps and 96
tokens. A separate process reloaded initial, distributed, and centralized PEFT
Adapters and found finite validation losses. The result is a mechanics and
recovery comparison, not a quality equivalence or superiority claim. Public
communication evidence records six 4,938,616-byte accepted deltas (29,631,696
bytes), six upload sessions, one resumed session, shared-cache savings, and
sub-second local Coordinator recovery. It is not a WAN throughput benchmark.

`LocalVolunteerBlobStore` provides atomic SHA-256 addressed artifacts;
`ResumableUploadManager` persists chunk hashes and private submission metadata;
`S3VolunteerBlobStore` supplies an S3/MinIO presign contract. The RC exercises
the local store and unit-tests S3 compatibility, but does not claim a live
external object service. The HTTP service rejects direct HTTP and untrusted
forwarded identities under the HTTPS policy and accepts a trusted reverse-
proxy header contract. This is not certificate or public TLS-handshake proof.

Cleanup stopped the HTTP service, reaped all Cell/replay processes, removed all
upload sessions and the entire private proof tree, and left hash-bound public
artifacts only. The strict checker also rejects tensor specs, lease/token/path
material, source/hash mutations, broken lineage, upload-resume retraining,
baseline budget mismatch, and physical multi-host overclaims.

Next work should not repeat this local proof. Run the same pinned ordinary
`crowdtensor volunteer campaign import-smollm-wikitext`, `serve`, and `join`
workflow on at least two independently administered physical Internet hosts,
measure real WAN latency/bandwidth and churn, independently reproduce the
result, and clean all remote resources. Sybil/poisoning resistance,
permissionless Byzantine safety, secure aggregation, useful model quality,
full-parameter training, GA, and SLA remain later milestones. Detailed usage
and scope are in `docs/volunteer-training-internet-beta.md`.

## 2026-07-17 Model Adapter Ecosystem Beta RC Achieved

The pluggable Model Adapter Ecosystem Beta goal is achieved. The canonical
portable artifact is
`dist/model-ecosystem-beta-20260717-r1/rc/model_ecosystem_beta_rc.json`.
Its file SHA-256 is
`59f2eded2a608da0479a229297eb5f8737db160169706b37d57481953a53496f`
and its embedded content hash is
`sha256:99ec6314f26b783ec4ba4c7ee24922671a9227b41b8ffe4b152d7451f320bfc2`.
The strict checker validates all nine portable evidence artifacts and returns
`goal_achieved=true` with zero errors:

`PYTHONPATH=. python scripts/model_ecosystem_beta_rc_check.py --report dist/model-ecosystem-beta-20260717-r1/rc/model_ecosystem_beta_rc.json --json`.

`model_adapter_v1.0` now supports separately installed plugins through the
`crowdtensor.model_adapters.v1` entry-point group. Discovery is fail-closed:
entry-point names must equal Adapter IDs, distribution name/version metadata
is mandatory, plugins cannot shadow built-ins, and every loaded object must
pass conformance. `CROWDTENSOR_DISABLE_MODEL_ADAPTER_PLUGINS=1` disables all
external plugins without affecting built-in Qwen2/SmolLM2. The Community CLI
provides `crowdtensor community adapters list|check`. Coordinator and Kaggle
workers can authenticated-download and hash-verify both the core wheel and a
plugin wheel. The isolated no-workspace smoke report is
`dist/model-ecosystem-beta-20260717-r1/plugin-smoke-r4-core-r2/model_adapter_plugin_smoke.json`;
its file SHA-256 is
`45655494d22c1c52b5e5b05adfa7f2f888391d2ac361f2e328185b98283a3b4d`.

The first official plugin is `crowdtensor-mistral-adapter==0.1.0b1`, Adapter ID
`mistral_lora_v1`. It pins the non-gated Apache-2.0 real trained checkpoint
`Locutusque/TinyMistral-248M-v2` at revision
`0f57b17cb317bb322c7c1466b669c681f80c058f` (248,024,064 parameters,
`MistralForCausalLM`, 12 layers, hidden size 1024, 32 attention heads, 8 KV
heads, sliding window 32). The core wheel SHA-256 is
`0e1c80070b470c511586e1e4830ca6e4507c03febd1d4294399057943d008232`;
the plugin wheel SHA-256 is
`e6fb1d9b817c5bb97b23442d257159db2a8931e776257028c2d85f32abb7ff77`.

The strict real live source is
`dist/model-ecosystem-beta-20260717-r1/mistral-live-attempt-2-strict-boundary/mistral_kaggle_heterogeneous_live.json`
with file SHA-256
`af9e32c7c4b3dcffc335d3e268fca00baffb2b04d321abb7316f2e4007d7d15e`.
One Kaggle T4x2 Kernel ran stage 0 and one Kaggle CPU Kernel ran stage 1.
Both clean-installed the core and plugin wheels outside the workspace, loaded
the pinned real weights, and committed exactly steps 1..8 in 195.155312
seconds. Both stages checkpointed LoRA plus Adam state at steps 4 and 8. The
first CUDA worker exited after step 4 without leasing any step-5 work; a
distinct generation-2 CUDA worker restored the step-4 checkpoint and completed
all stage-0 phases for steps 5..8. Eight forward
activations and eight backward gradients crossed the CPU/CUDA boundary through
hash-verified safetensors payloads. Both stage Adapters changed, 168 disjoint
tensors merged into standard PEFT format, and a separate process reloaded the
Adapter on CPU with finite logits of shape `[1,4,32005]`.

The Mistral gate has its own two-attempt ledger at
`dist/model-ecosystem-beta-20260717-r1/mistral-live-gate-ledger.json`; attempt
1 is retained but has `strict_status=superseded` because it allowed the old
worker to send step-5 forward before replacement. Attempt 2 is terminal
`achieved` with `strict_status=current` and the exact 1..4/5..8 worker boundary.
The ledger did not modify the exhausted immutable Community Maturity ledger.
Both temporary Kaggle Kernels were deleted, Coordinator and
tunnel stopped, private runtime removed, and an authenticated post-run query
found zero active Kernels. The focused local gate passed 42 tests plus
`py_compile` with zero failures.

Boundary: this is fixed-checkpoint, two-stage PEFT LoRA engineering evidence
using deterministic private token sequences. It does not establish useful task
quality, a real dataset result, Mistral-7B support, arbitrary Mistral or
arbitrary architecture support, stage-selective initial weight loading,
full-parameter training, TPU support for this Adapter, independent physical
multi-machine execution, GA, or an SLA. Hosted evidence remains `Kaggle logical
multi-node`.

## 2026-07-17 Community Maturity RC Achieved

The Community Maturity RC P0-P4 goal is achieved. The canonical portable
artifact is
`dist/community-maturity-rc-final/canonical-rc-attempt3/community_maturity_rc.json`
with file SHA-256
`4939653cf3bb03cbc5879dc527abe7801a2de9a9e9430bec5195a1e02dc46ca1`
and content hash
`sha256:4bca6aae9d221b8fc58602482cd524aba5af423023afcf01fa1267c8691aab70`.
Default and `--require-ready` checks pass with zero errors, nine valid source
artifacts, all 18 numbered requirements true, P0-P4 true, cleanup true, wheel
identity true, and `community_maturity_rc_ready=true`:

`PYTHONPATH=. python scripts/community_maturity_rc_check.py --report dist/community-maturity-rc-final/canonical-rc-attempt3/community_maturity_rc.json --require-ready --json`.

P0-P4 implementation now includes the versioned Community workflow and
protocol, Qwen2/SmolLM2 Model Adapters, RBAC/signing/replay/TLS/restricted
execution/quarantine/privacy controls, 12 bounded chaos scenarios, real MinIO
integration, clean wheel/sdist/container release infrastructure, SBOM and
license inventory, governance/community docs and templates, exact dependency
and protocol refusal contracts, portable RC pack/check, and final cleanup
audit. The final local-equivalent gate passed 81 focused tests and 12/12 chaos
scenarios. A separate earlier
heterogeneous scheduler/checkpoint/transport/JAX/product regression passed 62
tests with 2 conditional skips and no failures.

The first two full Kaggle gates are immutable failures in
`dist/community-maturity-live-gates/community_live_gate_ledger.json`. Attempt
1 served a wheel renamed to an invalid filename; attempt 2 reached the CPU and
CUDA logical workers but failed before the first step. Structured GPU
diagnostics found that Kaggle's optional `torchao==0.10.0` caused PEFT 0.19.1
to throw during ordinary dense LoRA dispatch and again during independent
reload. Both full attempts cleaned all remote and local resources. The
original maximum was exhausted at 2/2. On 2026-07-17 the user explicitly
authorized one and only one additional Kaggle CPU+GPU gate, capped at 45
minutes, with all other boundaries unchanged. The ledger was amended in place
from maximum 2 to 3 at `2026-07-17T11:58:34Z`; the first two attempt values are
unchanged. Its public amendment stores approval hash
`sha256:caff1c4d02d1032d5b618ec6c61029ccc7f6783470a95828a2dd000f1248f301`
instead of approval text. After successful attempt 3, the ledger file SHA-256
is `2e6d96ab3f4bdae825429e6ea8a6280b68499002a9e3066e03b940014bfdebf9`.
Attempt 3 is terminal with outcome `achieved`; the amended maximum is exhausted
at 3/3 and cannot be reset or repeated.

The successful full report is
`dist/community-maturity-rc-final/kaggle-live-attempt3/community_kaggle_short_reliability_live.json`
(file SHA-256
`3c2ba22067657892cc00a67dcfead0520478ab78620da76464417c1b65efb55d`).
One T4x2 Kernel and one CPU Kernel clean-installed the immutable wheel and
completed 100 contiguous atomic SmolLM2 LoRA steps in 578.9289 seconds. The
CUDA worker was replaced after step 30 and restored checkpoint plus Adam state.
The persisted restart barrier requested after step 50 restarted generation
1 -> 2 at committed step 51 with 1.787347 seconds downtime. All losses and
updates were finite; stage checkpoints were written at steps 30, 50, and 100.
The embedded dual-CUDA second-model run updated two logical stages, merged and
exported PEFT adapters, and independently reloaded finite logits. Its standalone
strict-passing report is
`dist/community-maturity-rc-final/kaggle-live-attempt3/community_smollm_two_stage_lora_live.json`
with file SHA-256
`13b109554d296123c30304de1386506c1711054e34062a70c06dd82f27a71159`.

The release wheel SHA-256 is
`3694aedad53bfb55d9ebb38bb92fe65f1b9fe6fef1c79e9be7a2f077de7d7b4b`.
The matching CPU smoke at
`dist/community-maturity-rc-final/kaggle-wheel-smoke/community_kaggle_wheel_smoke.json`
passes clean install, exact pins, full model-stack import, three golden commands,
privacy, and cleanup. The matching dual-T4 diagnostic at
`dist/community-maturity-rc-final/kaggle-gpu-diagnostic/community_kaggle_gpu_stage0_diagnostic.json`
loads real pinned SmolLM2 weights, performs stage0 forward/backward and an
optimizer update, then runs two independent CUDA stage processes for two
contiguous LoRA steps; both adapters change, merge/export succeeds, and
independent reload is finite. This remains diagnostic-only; final readiness is
provided by the successful CPU+GPU reliability gate above.

The final offline release report is
`dist/community-maturity-rc-final/release/community_release_build.json` (file
SHA-256
`9b8fcc3b8d5d50f78f53e29783360aff1acba1ace8815c654cfc949df324c71f`).
It reused the successful wheel and sdist without rebuilding either and enforced
wheel SHA-256
`3694aedad53bfb55d9ebb38bb92fe65f1b9fe6fef1c79e9be7a2f077de7d7b4b`.
Clean install, Docker build/removal, SBOM/license/runtime lock, documentation,
non-publication, and privacy checks pass. The offline bundle SHA-256 is
`f315dfac6709f15e52ca22b19b46e99e4413acd9b971eec5c5b26d26969dec28`.
The final cleanup is
`dist/community-maturity-rc-final/cleanup-final-attempt3/community_cleanup_audit.json`
with file SHA-256
`85674cfb4429cdfa810cf401c5881558bd09bff34e1e77a5ccfb79ca9e724c3f`.
It strict-passes after an authenticated Kaggle query and reports zero matching
Community Kernels, local private runtimes, Docker containers, and images. No
Goal resource is running.

Boundary: this RC validates Kaggle logical multi-node execution, not physical
machine independence. It is an offline Community RC, not PyPI/GitHub/registry
publication, GA/SLA, permanent provider capacity, arbitrary architecture
partitioning, full-parameter training, or complete Byzantine defense.

## 2026-07-19 Qwen 1.5B Elastic Training Showcase Achieved

The larger real-training showcase is achieved. The canonical public-safe
artifact is
`dist/training-showcase-20260718-final-r1/training_qwen15b_showcase.json`.
Its strict checker passes with `showcase_ready=true`, `goal_achieved=true`, and
zero errors:

`PYTHONPATH=. python scripts/training_qwen15b_showcase_check.py --report dist/training-showcase-20260718-final-r1/training_qwen15b_showcase.json --require-ready --json`.

The live run used pinned `Qwen/Qwen2.5-1.5B` revision
`8faed761d45a263340a0528343f099c05c9a4323` and pinned WikiText-2 revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. It completed 256 real LoRA
optimizer steps, four microbatches per step, 128-token sequences, and 131,072
training tokens. Two concurrent T4x2 Kernels completed steps 1-128; both were
deleted; the Coordinator observed a zero-Miner pause; two new T4x2 Kernels
restored all four central stage checkpoints and completed steps 129-256.
Exactly-once contiguous commits, distinct Kernel/Miner sessions, base-weight
freezing, positive LoRA gradients, and full cleanup are verified.

The 392-tensor standard PEFT adapter reloads on CPU and CUDA. On 64 pinned
held-out sequences, validation loss changed from 2.7319372669 to 2.3505238779
(-13.9613%) and perplexity from 15.3626 to 10.4911. The adapter archive is
`dist/training-showcase-20260718-live-256step-r1/training_qwen15b_standard_peft_adapter.zip`
with SHA-256
`f244519109fc22c8a6c9d61e9018273d12903700710f59baca0b3066fc83d075`.
The report SHA-256 is
`c51bdfa3fede4f5c8226835e3679795e5daf75befb4baea616928cefa1516c36`.
The post-change full regression result is 2,557 passed, two skipped, zero
failed.

Boundary: this is WikiText causal-LM LoRA adaptation over Kaggle logical
multi-node runtimes. It is not instruction tuning, general capability proof,
full-parameter training, independent physical multi-host evidence, GA, or an
SLA. Dataset licensing must be reviewed before publishing adapter weights. See
`docs/qwen15b-elastic-training-showcase.md`.

## Latest Qwen 1.5B Training Service Beta RC Status

Current superseding training-product status on 2026-07-12: the Qwen 1.5B
Four-GPU Training Service Beta RC is achieved. The canonical artifact is
`dist/training-qwen15b-beta-20260712-r3-live-achieved/training_qwen15b_beta.json`.
Its default and strict checkers return zero errors,
`training_qwen15b_beta_ready=true`, and `goal_achieved=true`:

`PYTHONPATH=. python scripts/training_qwen15b_beta_check.py --report dist/training-qwen15b-beta-20260712-r3-live-achieved/training_qwen15b_beta.json --require-ready --json`.

The fresh job at
`dist/training-qwen15b-beta-job-20260712-r1-live-attempt1/` used the real
ordinary-user CLI. It generated its own pinned Qwen source manifest and private
WikiText token payload, selected one authorized account, launched two concurrent
T4x2 Kernels, and placed ranges `[0,7)`, `[7,14)`, `[14,21)`, and `[21,28)` on
four CUDA processes. It retained FP32 frozen-stage and LoRA compute, GradScaler,
and FP16 activation/gradient boundary transport from the achieved Alpha.

Both 8-step runs completed. The resumed run forced a real Coordinator process
restart after step 4. Persistent rendezvous recovery took 3.394220 seconds,
restored 96 payloads and 492 events, observed both workers re-register, and
forced all four stage processes to restart from step-4 checkpoint state under
new PIDs. The final proof contains exactly 64 unique optimizer identities, 64
activations, 64 gradients, one stage-adapter payload, and verified four-stage
overlap. Loss dropped from 3.6082336903 to 2.8676728606. The 392-tensor PEFT
adapter covers all 28 layers, loads on CPU and CUDA, changes logits, and reduces
validation loss from 2.6663523763 to 2.4811929762.

The persistent service uses SQLite WAL plus an append-only idempotent event
ledger, monotonic global-step checks, expiring leases, one active GPU job, and a
bounded queue. CLI and authenticated HTTP routes share one controller for
submit, status/watch, resume, cancel, export, cleanup, events, and artifacts.
Credentials remain in private request state. Running cancellation uses a
private mode-0600 marker checked by the live probe; repeated submit, cancel, and
cleanup do not repeat work or advance revisions. The completed job was resumed
idempotently, exported through the user command, and cleaned twice with stable
`cleaned` status and no live resources.

Benchmark facts: 452.652131 seconds deployment/training, 153.472964 seconds to
first optimizer step, maximum step latency 8.625027 seconds, 35.103845 ms
four-stage overlap, 34,468,880 bytes over 129 private payloads, and
2,439,433,728 peak stage allocated bytes. The final suite records 358 passed,
zero failed tests: 326 existing regressions and 32 Beta tests. Default and
strict artifact checks, credential-value scanning, archive/export hashes, and
local process cleanup all pass.

The Beta ledger contains one verified attempt out of the goal-local limit of
three. Do not rerun it to recreate evidence. All resources created by the run
were removed. The post-cleanup audit saw zero active Kernel on the selected
account; unrelated active Kernels on two other accounts were not touched.

Boundary: this proves a usable persistent Beta service for pinned Qwen2.5 1.5B
PEFT LoRA on four same-account T4 GPUs. It does not prove 7B+, full-parameter or
pretraining, multi-account aggregation, elastic workers, permissionless trust,
billing, Web UI, production GA, or an SLA.

## Latest Qwen 1.5B Four-GPU Training Alpha Status

Current superseding training status on 2026-07-12: the Qwen 1.5B four-GPU
Pipeline Training Alpha is achieved. The canonical artifact is
`dist/training-qwen15b-four-gpu-alpha-20260712-r5-live-achieved/training_qwen15b_four_gpu_alpha.json`.
Default and strict checkers both pass with zero errors,
`qwen15b_four_gpu_alpha_ready=true`, and `goal_achieved=true`:

`PYTHONPATH=. python scripts/training_qwen15b_four_gpu_alpha_check.py --report dist/training-qwen15b-four-gpu-alpha-20260712-r5-live-achieved/training_qwen15b_four_gpu_alpha.json --require-ready --json`.

The live proof used one authorized Kaggle account and two concurrent T4x2
Kernels. Four CUDA processes selectively loaded pinned
`Qwen/Qwen2.5-1.5B` revision
`8faed761d45a263340a0528343f099c05c9a4323` ranges `[0,7)`, `[7,14)`,
`[14,21)`, and `[21,28)`. All stages performed real forward/backward and
optimizer steps with 50.719332 ms maximum four-stage overlap. Two 8-step runs
exchanged 64 private activations and 64 private gradients. The controlled run
stopped stage 2 after step 4 and restored it under PID 120 instead of PID 82;
losses and final adapters exactly match the uninterrupted run. Training loss
fell from 3.6082336903 to 2.8676728606.

The stable numerical path uses FP32 frozen-stage compute, FP32 LoRA parameters
and GradScaler, with FP16 only at activation/gradient stage boundaries. FP16
autocast was abandoned after attempt 4 produced a non-finite stage0 activation.
The private package removes Kaggle's incompatible `torchao==0.10.0`, then runs
PEFT and CUDA precision smokes before loading Qwen shards. Non-finite
activation, logits, loss, and gradient checks fail closed.

The merged standard PEFT adapter has 392 tensors covering layers 0..27, loads
on CPU/CUDA, changes logits, and reduces validation loss from 2.6663523763 to
2.4811929762. Archives were hash-verified before all temporary Kernels,
packages, payloads, Coordinator, tunnel, and local runtime state were removed.
The full regression artifact
`dist/training-qwen15b-tests-20260712-r5-live-achieved/training_qwen15b_test_summary.json`
records 313 passed and zero failed tests.
The read-only audit
`dist/training-qwen15b-four-gpu-postcleanup-audit-20260712-r5-live-achieved/training_qwen15b_four_gpu_live_probe.json`
then authenticated all four accounts, observed zero active Kernels on each,
and launched no allocation.

The immutable allocation ledger is
`dist/training-qwen15b-four-gpu-work/allocation_attempts.json`; it preserves four
incomplete attempts and verified attempt 5. The user authorizes an unbounded
cumulative number of same-account dual-Kernel attempts for this path. This is
not unlimited simultaneous allocation: one probe invocation reserves one
attempt, each attempt uses exactly two same-account T4x2 Kernels and is capped
at 1800 seconds, and no automatic infinite retry loop is allowed. Never erase
or rewrite prior ledger entries.

This Alpha proves real stage-selective 1.5B LoRA training over four T4s. It
does not prove full-parameter or 7B+ training, dynamic scaling, multi-account
training, anonymous-Miner trust, billing, or production WAN training. Do not
rerun it merely to recreate evidence.

## Latest CUDA Training RC Status

Current superseding CUDA-training status on 2026-07-11: the Two-Node CUDA
Training RC is achieved. The canonical artifact is
`dist/training-cuda-two-node-rc-20260711-r5-live-achieved/training_cuda_two_node_rc.json`.
Its default and strict checkers both return `ok=true`, `error_count=0`,
`training_cuda_two_node_rc_ready=true`, `goal_achieved=true`, and both live
gates verified:

`PYTHONPATH=. python scripts/training_cuda_two_node_rc_check.py --report dist/training-cuda-two-node-rc-20260711-r5-live-achieved/training_cuda_two_node_rc.json --require-ready --json`.

The authoritative live report is
`dist/training-cuda-two-node-live-20260711-a3-embedded-single-gate-gradscaler/training_cuda_two_node_live_probe.json`.
It used one authorized Kaggle account for both private T4x2 Kernels and
observed both running concurrently. The first Kernel's stage0 role ran the
single-Kernel acceptance gate before joining the cross-machine request. Two
independent processes on `cuda:0` and `cuda:1` completed a four-step baseline
and a four-step controlled-resume run. Stage1 was terminated after step 2,
restarted under a different PID, and restored LoRA, AdamW, GradScaler, cursor,
and optimizer-step state. Baseline and resumed adapters had maximum absolute
difference 0 and identical final loss; both reduced loss from 4.2349076271 to
3.9638671875. The stage0 GradScaler fix primes its lazy scale state before
applying stage1's already-scaled activation gradient.

The same two Kernels then completed four cross-machine CUDA training steps.
Authenticated Coordinator status records two distinct role registrations,
four hashed activation payloads, four hashed gradient payloads, and both role
completions. Each stage owns only half of the four-layer fixture and uses one
T4 (`cuda:0`) in its Kernel. Both Miners performed real Transformers/PEFT LoRA
backward on different dataset shards, kept the base frozen, and returned 28
named safetensors delta tensors. The existing StateStore rejection and DiLoCo
logic accepted both results, advanced global adapter version and outer step to
1, and verified sign compression plus error-feedback reconstruction.

The standard PEFT export loads on CPU and both CUDA workers, changes logits,
and reduces fixed validation loss from 4.1791639328 to 4.0241522789. CPU/CUDA
logits agree within 4.4703483582e-8 maximum absolute difference. The retained
stage0 checkpoint bundle contains embedded baseline/resume plus cross-node
pipeline/Miner checkpoints (25 files; SHA-256
`b26f9448fe73219c560385042a04defbf75a18adb89407b9bb9a5d1d9ba8b3f5`).
The stage1 bundle contains its pipeline/Miner checkpoints (7 files; SHA-256
`e1ef4cf56a198914ee6c8bcd3235e645fc355d43f0fc19cb487c9def74786a93`).
Host-side hashes match both worker reports.

The embedded single gate is accepted only because r5 binds its full record-level
evidence to the stage0 worker object, source Kernel hash, two-node attempt 3,
`before_cross_node_stage0` execution order, checkpoint archive, and inherited
Kernel deletion/private cleanup. Mutating any binding or any underlying
per-step record makes the strict checker fail. It is not co-location metadata
or a CPU/dry-run substitute.

The attempt ledger at `dist/training-cuda-two-node-work/allocation_attempts.json`
preserves all history: three standalone single-Kernel failures, two route-blocked
two-node attempts, and verified two-node attempt 3 under the one-time amended
3/3 budgets. Do not rerun this completed RC. All temporary Kernels were
deleted, checkpoint bundles and public evidence retained, private packages and
rendezvous payloads removed, Coordinator/tunnel stopped, and post-run checks
found no live Kaggle Kernel or local process. Public safety and all ten
malformed-delta rejection cases pass. The final code snapshot passes 64 CUDA,
27 CPU-training, and 167 StateStore/Miner/Coordinator tests.

User commands remain `crowdtensor train lora --backend cuda`, `train status`,
`train resume`, `train export`, and `train cleanup`. Credential values and
private tensor/sample payloads remain runtime-only. For future work, the
operator authorizes all four Kaggle accounts configured across the private
credential stores and all resources available to them. Isolate credentials
per process and do not use multiple accounts to satisfy a same-account gate.
This RC is bounded small-model LoRA evidence, not large-model/full-parameter
training, four-T4 tensor parallelism, anonymous-Miner trust, billing, or a
production marketplace.

## Latest Training Foundation RC Status

Current superseding training status on 2026-07-10: the CPU-only, GPU-ready
Training Foundation RC is achieved. The canonical artifact is
`dist/training-foundation-rc-20260710/training_foundation_rc.json`; the strict
checker passes with `training_foundation_rc_ready=true`, `goal_achieved=true`,
and no errors:

`PYTHONPATH=. python scripts/training_foundation_rc_check.py --report dist/training-foundation-rc-20260710/training_foundation_rc.json --require-ready --json`.

The achieved evidence uses a local deterministic 22,688-parameter
`LlamaForCausalLM`, real PyTorch autograd, Transformers, PEFT LoRA, and
safetensors. Two independent local `crowdtensor-miner` processes claimed two
distinct dataset shards through the existing HTTP Coordinator, task lease,
StateStore, and result ledger. Both returned 28 named adapter delta tensors;
the Coordinator completed one DiLoCo-style outer aggregation and advanced the
global adapter and outer optimizer from version/step 0 to 1. A trusted replay
reproduced one delta exactly, and named sign compression with error feedback
reconstructed the dense delta with its residual.

The separate two-process pipeline reference gives stage0 only embeddings and
layers 0-1 and stage1 only layers 2-3, norm, LM head, and loss. It transports a
real activation forward and activation gradient backward. A controlled run
hard-stopped stage1 midway, restored its LoRA and AdamW checkpoint in a new
process, and matched the uninterrupted final adapters with maximum absolute
difference 0 and final-loss difference 0. The pipeline loss fell from
4.2352905273 to 3.4311323166; all worker processes were cleaned up.

The aggregated adapter uses the standard PEFT layout, loads through
`PeftModel.from_pretrained`, changes logits, and reduces fixed validation loss
from 4.1791639328 to 4.0513637066. User paths are `crowdtensor train lora`,
`train status`, `train resume`, `train export`, and `train cleanup`. Detailed
architecture and operations are in `docs/training-foundation.md`.

Hard boundary: this CPU artifact is local LoRA evidence for
permissioned/trusted workers. Its `gpu_training_continuation_manifest.json` is
still a device-neutral dry-run handoff with `gpu_live_verified=false`; the
real bounded CUDA proof comes only from the r5 RC above. Neither RC proves
large-model training, WAN-scale production, full-parameter tuning,
anonymous-Miner poisoning resistance, billing, or a marketplace.

This training RC does not supersede inference truth: GLM 5.2 Kaggle deployment
r214 remains the achieved 1-token inference RC, while r34 remains the canonical
unachieved 8-token service Alpha blocker.

## Latest GLM 5.2 Kaggle Alpha Status

Current superseding Alpha status on 2026-07-08: the canonical Alpha blocker is
`dist/glm52-kaggle-alpha-20260708-r34-http-cleanup-route/glm52_kaggle_alpha.json`.
The default checker passes and keeps `glm52_kaggle_alpha_ready=false`; the
strict `--require-ready` checker fails because no token was generated and no
provider completed the live pipeline. This is not an achieved Alpha.

Latest r34 improvement: the ordinary-user HTTP service now exposes a
checker-backed `POST /cleanup` route. Service reports record
`cleanup_route_ready=true`; the smoke probe starts the real `AlphaHTTPServer`
and verifies `GET /health`, `GET /status`, quota-blocked `POST /generate`, and
`POST /cleanup`; the canonical r34 blocker imports
`service_smoke_summary.cleanup_route_verified=true` with temporary Kaggle
kernels deleted, temporary private packages removed, and no live resources
left running. This proves cleanup/service usability under the current quota
blocker, not live inference. The current blocker remains
`kaggle_gpu_quota_unavailable` with
`next_quota_refresh_time=2026-07-11T00:00:00`.

Latest r33 improvement: the user-facing generate CLI now has artifact-backed
recovery when the local service is unreachable. `crowdtensor generate
glm52-kaggle` reads the same output directory's Alpha/status artifacts after a
connection failure or blocked request and writes
`glm52_kaggle_alpha_generate_cli.json` with public-safe artifact recovery phase,
blockers, cleanup/quota summary, `next_resume_command`, and
`resume_private_inputs`. The r33 canonical blocker imports this proof under
`generate_cli_summary` and `artifacts.generate_cli_json`; it records
`cli_generate_artifact_recovery_supported=true`,
`generate_cli_check_ok=true`, `artifact_recovery_present=true`,
`artifact_recovery_resume_command_present=true`, and
`artifact_recovery_resume_private_inputs_verified=true`. The proof used an
unreachable local Coordinator endpoint and therefore proves recovery usability,
not live inference. The current blocker remains `kaggle_gpu_quota_unavailable`
with `next_quota_refresh_time=2026-07-11T00:00:00`.

Latest r32 improvement: the r31 public-safe `resume_private_inputs` recovery
contract is now exposed through ordinary-user status surfaces. HTTP
`GET /status`, quota-blocked `POST /generate`, and
`crowdtensor status glm52-kaggle` all include top-level
`resume_private_inputs` with
`schema=glm52_kaggle_alpha_resume_private_inputs_v1` and
`resume_command_omits_private_credentials=true`; Kaggle/HF token values, token
paths, section names, raw env names, cookies, proxy URLs, prompts, and
generated text remain non-public. Service reports record
`status_exposes_resume_private_inputs=true`, and the Alpha pack/checker reject
artifacts missing this recovery surface. The r32 canonical blocker retains the
local HTTP service smoke proof at
`dist/glm52-kaggle-alpha-20260708-r32-status-resume-private-inputs/glm52_kaggle_alpha_service_smoke_probe.json`.
It starts the real `AlphaHTTPServer` and verifies `GET /health`, `GET /status`,
and `POST /generate` against the r32 output directory. Its imported
`service_smoke_summary` records `status_resume_private_inputs_verified=true`
and `generate_resume_private_inputs_verified=true`. Because the current state is
still quota-blocked, `/generate` reaches the service but returns
public-safe HTTP 503 with `generate_route_quota_blocker_verified=true` and
`generated_token_count=0` instead of launching Kaggle live work. The r32 default
checker passes, strict readiness checker fails as expected, `crowdtensor status
glm52-kaggle --output-dir dist/glm52-kaggle-alpha-20260708-r32-status-resume-private-inputs
--json` reports the stored blocker with top-level `resume_private_inputs`, and
`crowdtensor cleanup glm52-kaggle --output-dir
dist/glm52-kaggle-alpha-20260708-r32-status-resume-private-inputs --json`
reports no live resources left running. This remains CLI/service-route and
quota-short-circuit proof only; the active Alpha goal remains incomplete until
a real 8-token CPU/GPU/TPU same-request GLM 5.2 request succeeds.

Latest r31 improvement: blocked Alpha reports include a checker-enforced
public-safe `resume_private_inputs` contract in the service summary, blocker
report, and top-level Alpha report. It records that live resume requires
private Kaggle credentials, that `next_resume_command` omits private credential
material, supported private credential input methods, and Hugging Face env-name
hash/count metadata. It explicitly records
`kaggle_credential_values_public=false`, `kaggle_token_file_paths_public=false`,
`kaggle_token_section_names_public=false`, `hf_env_names_public=false`, and
`hf_env_values_public=false`; no token values, token file paths, section names,
raw env names, cookies, proxy URLs, prompts, or generated text are public. The
r31 canonical blocker retains the local HTTP service smoke proof at
`dist/glm52-kaggle-alpha-20260708-r31-resume-private-inputs/glm52_kaggle_alpha_service_smoke_probe.json`.
It starts the real `AlphaHTTPServer` and verifies `GET /health`, `GET /status`,
and `POST /generate` against the r31 output directory. Because the current state
is still quota-blocked, `/generate` reaches the service but returns public-safe
HTTP 503 with `generate_route_quota_blocker_verified=true` and
`generated_token_count=0` instead of launching Kaggle live work. The r31 default
checker passes, strict readiness checker fails as expected, `crowdtensor status
glm52-kaggle --output-dir dist/glm52-kaggle-alpha-20260708-r31-resume-private-inputs
--json` reports `phase=decode_blocked`, and `crowdtensor cleanup glm52-kaggle
--output-dir dist/glm52-kaggle-alpha-20260708-r31-resume-private-inputs --json`
reports no live resources left running. This remains CLI/service-route and
quota-short-circuit proof only; the active Alpha goal remains incomplete until
a real 8-token CPU/GPU/TPU same-request GLM 5.2 request succeeds.

Latest r30 improvement: the ordinary-user default output directory contract now
covers the service command too. `deploy`, `serve glm52-kaggle`, `status`, and
`cleanup` share `dist/glm52-kaggle-alpha` by default, while non-target product
`serve` keeps its prior default. Service/Alpha artifacts record
`cli_serve_default_matches_deploy=true`,
`cli_status_default_matches_deploy=true` and
`cli_cleanup_default_matches_deploy=true`, and the pack/checker reject missing
or false default-path contracts. Tests cover `serve glm52-kaggle`, explicit
`--output-dir` preservation, product `serve`, `status`, and `cleanup` parse
paths. The r30 canonical blocker also retains the local HTTP service smoke proof
at
`dist/glm52-kaggle-alpha-20260708-r30-serve-default-output-dir/glm52_kaggle_alpha_service_smoke_probe.json`.
It starts the real `AlphaHTTPServer` and verifies `GET /health`,
`GET /status`, and `POST /generate` against the r30 output directory. Because
the current state is still quota-blocked, `/generate` reaches the service but
returns public-safe HTTP 503 with
`generate_route_quota_blocker_verified=true` and `generated_token_count=0`
instead of launching Kaggle live work. The r30 default checker passes, strict
readiness checker fails as expected, `crowdtensor status glm52-kaggle
--output-dir dist/glm52-kaggle-alpha-20260708-r30-serve-default-output-dir
--json` reports `phase=decode_blocked` with
`gpu_quota_status.source=alpha_gpu_quota_summary`, and
`crowdtensor cleanup glm52-kaggle --output-dir
dist/glm52-kaggle-alpha-20260708-r30-serve-default-output-dir --json`
reports no live resources left running. Public JSON contains no Kaggle/HF
credentials, Bearer headers, cookies, runtime proxy, raw prompt, generated
text, token ids, or private runtime state. This is CLI/service-route and
quota-short-circuit proof only; the active Alpha goal remains incomplete until
a real 8-token CPU/GPU/TPU same-request GLM 5.2 request succeeds.

Latest r29 improvement: the ordinary-user default output directory contract
became explicit and checker-enforced for `deploy`, `status`, and `cleanup`.
Those commands share `dist/glm52-kaggle-alpha` by default, service/Alpha
artifacts record `cli_status_default_matches_deploy=true` and
`cli_cleanup_default_matches_deploy=true`, and the pack/checker reject missing
or false default-path contracts.

Latest r28 improvement: a local HTTP service smoke proof is now imported into
the canonical Alpha blocker. The retained smoke artifact is
`dist/glm52-kaggle-alpha-20260708-r28-http-service-smoke/glm52_kaggle_alpha_service_smoke_probe.json`.
It starts the real `AlphaHTTPServer` and verifies `GET /health`,
`GET /status`, and `POST /generate` against the r28 output directory. Because
the current state is still quota-blocked, `/generate` reaches the service but
returns public-safe HTTP 503 with
`generate_route_quota_blocker_verified=true` and `generated_token_count=0`
instead of launching Kaggle live work. `scripts/glm52_kaggle_alpha_pack.py`
imports this as `service_smoke_summary`, records
`artifacts.service_smoke_json`, and `scripts/glm52_kaggle_alpha_check.py`
validates the summary when present. The r28 default checker passes, strict
readiness checker fails as expected, `crowdtensor status glm52-kaggle
--output-dir dist/glm52-kaggle-alpha-20260708-r28-http-service-smoke --json`
reports `phase=decode_blocked` with
`gpu_quota_status.source=alpha_gpu_quota_summary`, and
`crowdtensor cleanup glm52-kaggle --output-dir
dist/glm52-kaggle-alpha-20260708-r28-http-service-smoke --json` reports no
live resources left running. Public JSON contains no Kaggle/HF credentials,
Bearer headers, cookies, runtime proxy, raw prompt, generated text, token ids,
or private runtime state. This is service-route and quota-short-circuit proof
only; the active Alpha goal remains incomplete until a real 8-token
CPU/GPU/TPU same-request GLM 5.2 request succeeds.

Latest r27 improvement: Kaggle runtime failure classification is now an
explicit recovery contract. `scripts/glm52_kaggle_stage_worker_push_probe.py`
classifies push timeout/HTTP 429/empty response, status timeout, wait timeout,
output timeout/HTTP 429/empty response/missing stage report, terminal
error/cancelled, and cleanup/delete timeout into public-safe blockers for both
live and collect modes. Service and Alpha artifacts expose
`kaggle_runtime_blocker_classification_ready=true` and the supported class list
under `phase_status.configuration_check.evidence`. The Alpha pack/checker
reject artifacts missing that recovery contract. The r27 default checker
passes, strict readiness checker fails as expected, `crowdtensor status
glm52-kaggle --output-dir
dist/glm52-kaggle-alpha-20260708-r27-runtime-blocker-classification --json`
reports `phase=blocked_gpu_quota`, and `crowdtensor cleanup glm52-kaggle
--output-dir dist/glm52-kaggle-alpha-20260708-r27-runtime-blocker-classification
--json` writes a public-safe cleanup proof with no live resources left running.

Latest r26 improvement: the ordinary-user Hugging Face token environment
contract is now explicit and public-safe. `crowdtensor deploy glm52-kaggle` and
`crowdtensor serve glm52-kaggle` accept `--hf-token-env`, forward it into
`scripts/glm52_kaggle_same_request_live_probe.py` and
`scripts/glm52_kaggle_stage_worker_push_probe.py`, and uploaded Kaggle workers
receive the token only through private runtime env. The worker package and GLM
HF range fetch helpers add an Authorization Bearer header from private
`HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` values when present. Public service, Alpha,
push, status, cleanup, and benchmark artifacts store only HF env-name hashes,
counts, configured booleans, and `hf_token_public=false`; raw env names, token
values, Bearer headers, and authorization material are not persisted. The
Alpha pack/checker reject a missing HF token env contract or
`hf_token_public=true`. The r26 default checker passed and strict readiness
checker failed as expected.

Latest r25 improvement: the ordinary-user `--model`/`--accelerators` contract
is now explicit and preserved in resume commands. `crowdtensor deploy
glm52-kaggle` and `crowdtensor serve glm52-kaggle` fail fast unless the model is
a supported GLM 5.2 source (`cyankiwi/GLM-5.2-AWQ-INT4` or `zai-org/GLM-5.2`)
and the accelerator request is the supported `cpu,gpu,tpu` set. Service and
Alpha artifacts record `requested_model`, `model_request_supported`,
`accelerators`, `required_accelerators`, and
`accelerator_request_complete`; `phase_status.configuration_check.evidence`
surfaces these fields. The Alpha pack/checker reject unsupported model requests
or incomplete accelerator requests. The r25 `next_resume_command` includes
`--model cyankiwi/GLM-5.2-AWQ-INT4 --accelerators cpu,gpu,tpu` while keeping
credentials, raw prompts, generated text, token ids, cookies, and private
runtime state out of public artifacts.

Latest r23 improvement: the main Alpha report now exposes the same public-safe
`next_resume_command` that is stored in `blocker_report.next_resume_command`.
The checker rejects blocked reports without a resume command, without the
redaction flag, or with mismatched top-level/blocker resume commands. r23 keeps
`next_resume_command_redacts_credentials=true` and does not write Kaggle tokens,
cookies, signed URLs, raw prompts, generated text, token ids, or private runtime
state into public artifacts.

Latest r22 improvement: ordinary users can submit prompts through
`crowdtensor generate --target glm52-kaggle ...` or
`crowdtensor generate glm52-kaggle --prompt-text ...`. The command posts to the
local Alpha service `/generate`, writes
`glm52_kaggle_alpha_generate_cli.json`, stores prompt and service URL hashes
instead of raw values, captures public-safe HTTP 400/503 response bodies, and
does not persist raw prompts or raw service URLs. Service reports declare
`cli_generate_command_available=true`; `scripts/glm52_kaggle_alpha_pack.py` and
`scripts/glm52_kaggle_alpha_check.py` require that contract.

Latest r21 improvement: `/generate` now has public-safe request validation.
Service reports declare `generate_validates_request_schema=true`, and
`scripts/glm52_kaggle_alpha_pack.py` plus
`scripts/glm52_kaggle_alpha_check.py` require that contract. Malformed JSON,
empty/non-object bodies, missing prompts, and invalid token counts return HTTP
400 with schema `glm52_kaggle_alpha_generate_response_v1`, no raw prompt/body,
and `phase=generate_request_invalid` in service status. Unit tests verify
malformed JSON and missing-prompt requests do not call the live probe or mocked
generate function.

Latest r20 improvement: service `/status` now loads existing public-safe Alpha
artifacts on startup. Service reports declare
`status_loads_existing_alpha_artifacts=true`; `scripts/glm52_kaggle_alpha_pack.py`
and `scripts/glm52_kaggle_alpha_check.py` require that contract. A live HTTP
status probe against the r20 blocker, before any `/generate`, returned
`phase=blocked_gpu_quota`, `alpha_report_present=true`,
`phase_status.overall_state=blocked`, cleanup verified, and the public-safe
next resume command. `crowdtensor cleanup glm52-kaggle` against r20 reads this
service status as `cleanup_evidence_source=service_status`.

Latest r19 improvement: `/generate` now uses still-current GPU quota blocker
evidence before launching Kaggle. Service reports declare
`generate_uses_current_gpu_quota_blocker=true`, and
`scripts/glm52_kaggle_alpha_pack.py` plus
`scripts/glm52_kaggle_alpha_check.py` require that contract. Against the r19
blocker, `generate_with_live_probe()` short-circuits with public-safe blockers
`kaggle_gpu_quota_unavailable` and
`glm52_alpha_request_blocked_by_current_gpu_quota_preflight`, includes
`next_quota_refresh_time=2026-07-11T00:00:00`, preserves cleanup proof, and
does not launch the live Kaggle probe. This improves ordinary-user request
behavior while keeping the Alpha unachieved until a real 8-token live run
passes.

Latest r18 improvement: `scripts/glm52_kaggle_alpha_pack.py` now writes
structured public-safe `phase_status` evidence into the Alpha report, and
`crowdtensor status glm52-kaggle` exposes it at top level. The phase schema is
`glm52_kaggle_alpha_phase_status_v1` and records
`configuration_check`, `model_source_check`, `gpu_quota_preflight`,
`kernel_push`, `gpu_queue_running`, `tpu_queue_running`, `cpu_queue_running`,
`stage_completed`, `decode_completed`, and `cleanup_completed`. In the r19
quota blocker, `phase_status.overall_state=blocked`, blocked phases are
`gpu_quota_preflight`, `kernel_push`, and `gpu_queue_running`, and completed
phases are `configuration_check`, `model_source_check`, and
`cleanup_completed`. This satisfies the user-facing stage observability
requirement without claiming live inference success.

Latest r17 improvement: `scripts/glm52_kaggle_alpha_pack.py` now writes a
separate public-safe benchmark artifact at
`glm52_kaggle_alpha_benchmark.json`, and the Alpha summary lists it under
`artifacts.benchmark_json`. The artifact schema is
`glm52_kaggle_alpha_benchmark_v1` and includes deploy time, stage count,
provider coverage, first-token/stage-latency fields, generated-token count,
cleanup status, runtime tuning, blockers, and public-safety metadata. In the
r17 quota blocker it is a blocker benchmark with `tokens_generated=0`,
`stage_count=0`, and empty provider coverage because no live run was started.
This satisfies the artifact-shape requirement without claiming inference
success.

Latest r16 improvement: the blocker `next_resume_command` is now a public-safe
full deploy command, not a partial hint. It includes `--run-live`,
`--gpu-quota-preflight`, `--output-dir`, `--stage-push-parallelism 7`, the r11
stage-worker package, accelerator selections, wait/kernel/Coordinator timeout
settings, and runtime tuning. The blocker records
`next_resume_command_redacts_credentials=true`; it intentionally does not write
Kaggle tokens, cookies, signed URLs, raw prompts, generated text, token ids, or
private runtime material into public artifacts. r16 still imports the same r14
GPU quota evidence and remains blocked until a GPU quota refresh or a different
available GPU account is used.

Latest r15 improvement: `/generate` now has an explicit timeout contract.
Service reports advertise `generate_request_fields=["prompt","max_new_tokens","timeout","timeout_seconds"]`;
request-level `timeout`/`timeout_seconds` is forwarded into the same-request
live probe's wait timeout and Coordinator task timeout, capped by the service
configured wait. `scripts/glm52_kaggle_alpha_pack.py` and
`scripts/glm52_kaggle_alpha_check.py` now require this timeout contract before
Alpha readiness. The r15 blocker imports the same public-safe r14 GPU quota
evidence, so it records all four authenticated GPU accounts exhausted until
`2026-07-11T00:00:00`, but does not rerun Kaggle GPU work.

New after r8: runtime tuning now flows from `crowdtensor deploy/serve
glm52-kaggle` into the private uploaded Kaggle worker env and is preserved in
public-safe service, live, benchmark, and blocker artifacts. The r11 blocker
records the tested low-cost settings:
`full_prefix_prefill_length=1`, `full_prefix_dsa_mask_topk=1`,
`full_prefix_executed_expert_count=2`, `full_prefix_top_k=1`,
`full_prefix_row_block_size=512`, `full_prefix_max_tensor_bytes=33554432`,
`full_prefix_max_block_bytes=16777216`,
`cpu_group_stage_attempt_seconds=2.5`, and
`cpu_group_stage_poll_seconds=0.5`.

The current reusable live package is
`dist/glm52-kaggle-stage-worker-package-20260707-alpha-r11-unique-runtime-tuning/glm52_kaggle_stage_worker_package.json`.
It was regenerated from the r8 topology with unique `ct-glm52-alpha-r11-*`
Kaggle slugs because deleted r8 slugs can return `Notebook not found`.
Its package checker passes and it still launches 7 Kaggle kernels when quota
allows: one CUDA kernel, one TPU kernel, and five CPU group kernels covering
all 39 real 2-layer stages.

The r11 1-token live probe is
`dist/glm52-kaggle-alpha-20260707-live-1tok-r11-unique-runtime-tuning/glm52_kaggle_same_request_live_probe.json`.
It did not enter the stage pipeline because Kaggle rejected the CUDA push with
`kaggle_gpu_quota_or_session_rejected`. Cleanup is verified:
`temporary_kaggle_kernels_deleted=true`, `temporary_private_packages_removed=true`,
`live_resources_left_running=false`, and no retained kernels.

The r12 quota probes imported into the Alpha blocker prove this is currently an
external GPU availability blocker, not a slug/package/cleanup issue. Public-safe
evidence:
`dist/kaggle-gpu-token-weekly-quota-probe-20260707-alpha-r12-section-accounts/kaggle_gpu_token_weekly_quota_probe.json`
authenticated and probed `tpuowner`, `primary Kaggle account`, and `cpuowner`; all returned
`weekly_gpu_quota_exhausted`.
`dist/kaggle-gpu-token-weekly-quota-probe-20260707-alpha-r12-crowdtensor-raw/kaggle_gpu_token_weekly_quota_probe.json`
authenticated and probed the dedicated raw-token GPU account `gpuowner`;
it also returned `weekly_gpu_quota_exhausted`. The shared quota refresh time is
`2026-07-11T00:00:00`. The next resume should use a Kaggle account with
available GPU quota or wait for that refresh, then run the r12 blocker report's
`next_resume_command`.

Latest r14 improvement: GPU quota preflight is now part of the deploy path.
`crowdtensor deploy glm52-kaggle --run-live --gpu-quota-preflight ...`
automatically ran `scripts/kaggle_gpu_token_weekly_quota_probe.py`, imported
the quota report into Alpha pack, and skipped the live run before launching any
GLM workers because all four authenticated GPU accounts were exhausted. The r14
artifact records `gpu_quota_preflight_performed=true`,
`live_skipped_by_gpu_quota_preflight=true`, `cleanup_verified=true`,
`kaggle_gpu_quota_unavailable`, and the same next quota refresh time
`2026-07-11T00:00:00`. This is the recommended resume path because it avoids a
doomed CUDA push while still producing a checker-passing blocker.

Current status usability behavior after r31: `crowdtensor status
glm52-kaggle` now defaults to the deploy output directory
`dist/glm52-kaggle-alpha` and can aggregate a service status file, deploy CLI
summary, canonical Alpha artifact, live report, and GPU quota preflight report
from one output directory. Against the r31 blocker directory it reports
`phase=decode_blocked`, `alpha_report_present=true`, `cleanup_verified=true`,
`gpu_quota_status.source=alpha_gpu_quota_summary`, all four authenticated GPU
accounts exhausted, and the blocker's `next_resume_command`, plus top-level
`phase_status` with the blocked/completed phase names above.
This is a user-facing status improvement only; it does not make the Alpha goal
achieved and does not replace the missing 8-token same-request live proof.

Current cleanup usability behavior after r31: `crowdtensor cleanup
glm52-kaggle` can now read deploy artifacts as well as an HTTP service status
file. Against the r31 blocker directory it writes
`glm52_kaggle_alpha_cleanup.json` with `ok=true`,
`cleanup_evidence_source=alpha_report`,
`cleanup_mode=gpu_quota_preflight_skipped_live`,
`temporary_kaggle_kernels_deleted=true`,
`temporary_private_packages_removed=true`, and
`live_resources_left_running=false`. This only proves cleanup/no retained live
resources for the quota-preflight-skipped deploy; it does not make the Alpha
ready and must not be counted as multi-token GLM 5.2 inference success.

The r8 engineering baseline is
`dist/glm52-kaggle-stage-worker-package-20260707-alpha-r8-39stage-cpu-groups-generic-worker-full-payload/glm52_kaggle_stage_worker_package.json`.
It preserves the 39 real 2-layer stage specs but pushes only 7 Kaggle kernels:
one CUDA kernel, one TPU kernel, and five CPU group kernels. CPU groups now use
claim-before-stage-load and one embedded generic stage worker source injected
with `CT_GLM52_STAGE_PAYLOAD_JSON`, so large CPU groups avoid Kaggle
`SaveKernel` 400 and no-task stages avoid expensive HF/weight loading.

The bounded r8 live attempt reached all 39 stage workers and completed
same-request stages 0, 1, and 2. It was aborted after about 2220 seconds because
only 3/312 expected stage tasks were complete and 0/8 tokens had been generated,
making the 8-token Alpha bound throughput-infeasible on the current CPU
full-prefix runtime. Evidence is public-safe:
`r8_coordinator_status_before_abort.json`,
`r8_manual_cleanup.json`, and
`glm52_kaggle_alpha_runtime_blocker.json` with
`failure_stage=glm52_alpha_runtime_throughput_below_8token_bound`.

Do not rerun r3/r4/r5/r6/r7 packages. Next work should optimize CPU
full-prefix runtime, reduce per-stage CPU cost, or move more stages to
available accelerators while preserving the checker requirement that a real
same-request multi-token GLM 5.2 path completes across required providers.

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
because multi-token live evidence, three-provider coverage, benchmark token
count, and cleanup proof are still missing. Do not mark the Alpha goal
achieved from this blocker artifact.

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
`same_request_decode_verified=true`, and `failure_stage=none`.

The successful live run is
`dist/glm52-kaggle-same-request-live-20260707-r211-r209-hf-fetch-retries-full-live/glm52_kaggle_same_request_live_probe.json`.
`scripts/glm52_kaggle_same_request_live_check.py --require-verified` passes
with `generated_token_count=1`, `same_request_decode_verified=true`, and
`stage_count=39`. The assembled same-request proof at
`dist/glm52-kaggle-same-request-live-20260707-r211-r209-hf-fetch-retries-full-live/same-request/glm52_kaggle_same_request_probe.json`
also passes `scripts/glm52_kaggle_same_request_check.py --require-verified`;
it records accepted providers `["kaggle_cpu","kaggle_cuda","kaggle_jax_tpu"]`,
one generated token hash, live Coordinator request proof, stage-provider
coverage, and cleanup proof. The model source remains GLM 5.2 with compatible
public quantized weights: `zai-org/GLM-5.2` via
`cyankiwi/GLM-5.2-AWQ-INT4`, as resolved by
`dist/glm52-model-source-resolver-20260704-r4-awq-safetensors-recommended/glm52_model_source_resolver.json`.

The r211 live run used the r209 worker package
`dist/glm52-kaggle-stage-worker-package-20260707-r209-r5-hf-fetch-retries/glm52_kaggle_stage_worker_package.json`
and request hash
`sha256:8385016dbeb99152007a34bce07e028a1ac9a564a28b5b294ca54164b49afeee`.
It completed all 39 GLM-5.2 stage workers in one same-request decode: stage0
on Kaggle CUDA (`gpuowner`), stage13 on Kaggle JAX TPU (`tpuowner`), and
the remaining stages on Kaggle CPU (`cpuowner`). The cleanup report
`dist/glm52-kaggle-same-request-live-20260707-r211-r209-hf-fetch-retries-full-live/glm52_kaggle_cleanup_report.json`
records no retained/uncleaned kernels, no live resources left running, and
public-safe cleanup metadata only.

The RC pack/checker were updated so verified live same-request evidence is the
final runtime proof. This means stale metadata-only source/package preflight
blockers and full-weight disk-budget blockers do not prevent a quantized live
GLM-5.2 RC success, while queue evidence, smoke tests, non-GLM fallback,
missing token hash, missing provider coverage, and missing cleanup are still
rejected. The regression command
`PYTHONPATH=. pytest -q tests/test_glm52_kaggle_accelerator_deployment_rc.py`
passes with 60 tests.

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
`failure_stage=glm52_full_decode_adapter_not_ready`; this is not goal
completion.

New r194/r197 progress: the project now has a GLM-specific Coordinator bridge
contract in `scripts/glm52_kaggle_coordinator_decode_bridge_probe.py` plus
checker `scripts/glm52_kaggle_coordinator_decode_bridge_check.py`. The retained
contract artifact is
`dist/glm52-kaggle-coordinator-decode-bridge-20260706-r194-contract/glm52_kaggle_coordinator_decode_bridge_probe.json`.
It is checker-passing and public-safe, but explicitly contract-only:
`coordinator_bridge_contract_ready=true`, `same_request_decode_verified=false`,
`live_run_performed=false`, with blocker
`glm52_live_kaggle_same_request_not_run`.

The current worker package is
`dist/glm52-kaggle-stage-worker-package-20260706-r197-r5-coordinator-private-env-bridge/glm52_kaggle_stage_worker_package.json`.
It keeps the r5 39-stage full-coverage topology and request hash
`sha256:8385016dbeb99152007a34bce07e028a1ac9a564a28b5b294ca54164b49afeee`.
Rendered private Kaggle kernels now support Coordinator mode: with private
`CT_GLM52_COORDINATOR_URL` and `CT_GLM52_COORDINATOR_TOKEN`, a worker claims
its stage task, consumes private upstream activation, writes private downstream
activation, submits activation hashes for non-final stages, and submits the
selected token hash for the final stage. Without those env vars, the worker
remains a stage-runtime proof and does not set `stage_decode_verified=true`.
`scripts/glm52_kaggle_stage_worker_push_probe.py` can inject those private
values by uploading a temporary `ct_glm52_private_runtime_env.json` into the
private Kaggle package, then deleting the local copy after `kaggle kernels
push`; public artifacts only record boolean/key-count metadata and must not
include the Coordinator URL or token.

The precise remaining live gap is still
`coordinator_same_request_decode_runtime`. r188 verified activation handoff,
r189 records only that missing capability, r191 made the compressed-tensors
runtime foundation available, and r197 adds the worker-side Coordinator bridge
plus private runtime env injection.
The missing proof is a real Kaggle CPU/GPU/TPU same-request run that produces
stage reports with `stage_decode_verified=true`, a public-safe Coordinator
generated-token hash report, cleanup proof, and a passing
`glm52_kaggle_same_request_check.py --require-verified`. Next resume should
run that live path with r197; do not rerun all 39 stage-runtime coverage first.

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
`failure_stage=web_tpu_channel_jupyter_execute`. This supersedes r27 for
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

Latest continuation after r20: the code now has explicit test coverage for a
lower-GPU 72B topology. `tests/test_kaggle_32b_full_heterogeneous_probe.py`
includes `test_build_report_accepts_72b_single_gpu_web_tpu_cpu_tail_topology`,
which keeps the full success gate intact while avoiding the failed second GPU
shard: one T4x2 GPU kernel owns stages 0/1, one Web TPU stage owns stage4, and
one CPU tail kernel owns stages 2/3/5/6/7/8/9. A report only counts as 72B
success if all 80 Qwen 72B layers complete in the same Coordinator request
with GPU, TPU, and CPU stage evidence.

The full-run driver now also fails fast when a worker has already reported a
failed stage while the Coordinator is not ready, so future live attempts should
summarize and clean up instead of waiting through the full coordinator timeout.
This is covered by
`test_wait_for_coordinator_ready_returns_when_worker_report_fails`.
The same test file now also includes
`test_run_coordinator_probe_can_assemble_single_gpu_72b_topology`, which
simulates GPU, Web TPU, and CPU workers completing all 10 stages in one
Coordinator request for the topology `2GPU_stages_1WebTPU_stages_7CPU_stages`.

The live attempt was deferred because both TPU provider paths were unavailable
again. The fresh Web TPU execution-channel probe
`dist/kaggle-web-tpu-execution-channel-probe-20260628-r13-before-single-gpu-72b/kaggle_web_tpu_execution_channel_probe.json`
timed out at Jupyter execution, and the follow-up Active Event probe
`dist/kaggle-web-tpu-active-event-probe-20260628-r5-after-channel-timeout-reopen/kaggle_web_tpu_active_event_probe.json`
showed the `TPU v5e-8` event had become `Cancelled`. Colab fallback also did
not reacquire: `dist/colab-tpu-reacquire-retry-20260628-r9-after-kaggle-active-cancelled/colab_tpu_reacquire_retry_probe.json`
attempted V5E1 for authuser 0 and 1 with cleanup-before enabled and both
returned HTTP 503. No temporary Kaggle kernels from these checks remained.
The subsequent Web TPU start/wait probe
`dist/kaggle-web-tpu-start-wait-20260628-r17-restart-after-cancelled/kaggle_web_tpu_start_wait_probe.json`
selected TPU v5e-8, clicked Start Session, and waited 300 seconds, but ended
with queue/starting visible and no Jupyter frame/session/kernel.
The latest bounded Active Event follow-up is
`dist/kaggle-web-tpu-active-event-probe-20260628-r8-queued-wait-10m/kaggle_web_tpu_active_event_probe.json`:
the TPU v5e-8 event stayed `Queued` for the full 10-minute window. Colab
fallback remains unavailable in
`dist/colab-tpu-reacquire-retry-20260628-r10-after-web-restart-queue/colab_tpu_reacquire_retry_probe.json`,
where V5E1 attempts for authuser 0 and 1 both returned HTTP 503.
The latest continuation kept waiting on the same Kaggle TPU event:
`dist/kaggle-web-tpu-active-event-probe-20260628-r10-queued-wait-20m-total/kaggle_web_tpu_active_event_probe.json`
waited another bounded 10 minutes and the event was still `Queued`, about
40 minutes old, with no Jupyter frame/session/kernel. Colab fallback was tried
again in
`dist/colab-tpu-reacquire-retry-20260628-r11-after-r10-queue/colab_tpu_reacquire_retry_probe.json`;
both authuser 0 and 1 V5E1 attempts returned HTTP 503.

The previous r24 max-search artifact was
`dist/three-accelerator-dense-max-parameter-search-20260628-r24-web-tpu-queued-40m-colab-503/three_accelerator_dense_max_parameter_search.json`.
Its checker passes and records `max_successful_same_request_decode_parameter_class=32b`,
`max_attempted_parameter_class=72b`, `max_stage_loaded_parameter_class=72b`,
`max_tpu_executed_parameter_class=72b`, and
`failure_stage=colab_tpu_reacquire_not_ready`, with Web TPU cancelled/timeout
and restart/queue blockers imported separately. This is still not 72B full
all-layer same-request decode success.

Latest superseding status on 2026-06-28: Colab TPU can be used as the `jax_tpu`
fallback path inside the unfinished 72B goal, but it has not completed the
full all-layer objective. The retained Colab capacity evidence is
`dist/colab-tpu-qwen-stage-loader-20260628-r2-72b-stage32-36-four-layer-fixed/colab_tpu_qwen_stage_loader_probe.json`:
`Qwen/Qwen2.5-72B-Instruct` layers 32-36 executed on Colab `TPU v5 lite` with
48 stage-owned keys, about 6.539GB logical execution tensor bytes, stage-local
KV-cache metadata, and public-safe hashes. Fresh Colab V5E1 reacquire is not
currently available: `dist/colab-tpu-reacquire-retry-20260628-r8-v5e1-after-r7/colab_tpu_reacquire_retry_probe.json`
attempted V5E1 twice and both attempts returned HTTP 503.

Kaggle Web TPU execution recovered after fixing Active Event parsing. Kaggle
shows status strings like `Running: 14 minutes`, so
`scripts/kaggle_web_tpu_active_event_probe.py` now treats `Running*` as
running. The active-event probe
`dist/kaggle-web-tpu-active-event-probe-20260628-r4-running-prefix-open-attempt/kaggle_web_tpu_active_event_probe.json`
shows one running/opened `TPU v5e-8` event, though that UI probe still does not
see a Jupyter frame/session. The authoritative runtime evidence is
`dist/kaggle-web-tpu-execution-channel-probe-20260628-r12-active-event-running-opened/kaggle_web_tpu_execution_channel_probe.json`:
the browser iframe service manager connected to an existing Jupyter session,
small JAX and tiny Qwen-like cells executed, and JAX saw 8 `TPU v5 lite`
devices. The max-search pack/checker now lets this execution-channel proof
override the weaker Active Event UI frame gap, and treats Colab reacquire
failure as fallback-only when Kaggle Web TPU is ready.

A fresh 10-stage all-layer 72B retry was attempted at
`dist/kaggle-72b-full-heterogeneous-kaggle-web-tpu-live-20260628-r2-10stage-staggered/`.
It used Kaggle Web TPU for stage4 and 20-second launch staggering. It did not
complete: gpu-shard0 was accepted but its stage report failed, gpu-shard1 was
not accepted and was deleted, and CPU stage5..9 kernels were manually deleted
after the missing GPU shard made success impossible. Do not infer 72B success
from this interrupted retry.

The earlier r20 max-search artifact was
`dist/three-accelerator-dense-max-parameter-search-20260628-r20-web-tpu-channel-ready-72b-retry-gpu-shard-missing/three_accelerator_dense_max_parameter_search.json`.
Its checker passes and it records `max_successful_same_request_decode_parameter_class=32b`,
`max_attempted_parameter_class=72b`, `max_stage_loaded_parameter_class=72b`,
`max_tpu_executed_parameter_class=72b`, and
`failure_stage=dense_72b_same_request_decode_not_verified_after_tpu_stage_forward`.
The active goal remains incomplete until dense/full-precision GPU+TPU+CPU
same-request 1-token decode over all 80 Qwen 72B layers succeeds and
checker/tests/docs are updated.

## Latest Colab TPU Fallback Status

Latest superseding status after the Colab runtime helper fix: Colab remains a
valid `jax_tpu` provider path, but the active 72B full-decode goal is still not
achieved. `scripts/colab_cli_runtime.py` now lets the project Python reuse the
locally installed isolated `google-colab-cli` tool environment, so Colab live
probes no longer fail merely because `colab_cli` is installed under the uv tool
Python instead of `/usr/bin/python`. The current system-Python smoke artifacts
prove that helper works and that the retained Colab session is detached:
`dist/colab-tpu-runtime-stability-20260628-r14-system-python-runtime-helper/colab_tpu_runtime_stability_probe.json`
reports runtime proxy HTTP 404 / no TPU device, and
`dist/colab-tpu-qwen-stage-loader-20260628-r4-system-python-runtime-helper/colab_tpu_qwen_stage_loader_probe.json`
fails with a Colab HTTPError before stage execution.

The latest bounded Colab reacquire evidence is
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
evidence, and the dense max-search pack/checker imports it as
`web_tpu_active_event_import`. The current active event artifact is
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

As of 2026-06-28, Colab can be used as a TPU provider fallback, but it has not
completed the active 72B goal. Colab V5E1 was reacquired in
`dist/colab-tpu-session-20260628-r9-reacquire-v5e1/colab_tpu_session_probe.json`
and passed a three-round JAX BF16 stability check in
`dist/colab-tpu-runtime-stability-20260628-r9-v5e1-three-rounds/colab_tpu_runtime_stability_probe.json`.
The retained Colab 72B stage-owned loader proof is
`dist/colab-tpu-qwen-stage-loader-20260628-r2-72b-stage32-36-four-layer-fixed/colab_tpu_qwen_stage_loader_probe.json`:
`Qwen/Qwen2.5-72B-Instruct` layers 32-36 executed on one Colab `TPU v5 lite`,
loading 48 stage-owned keys and about 6.539GB logical execution tensor bytes,
with no missing stage keys, stage-local KV-cache verified, and public-safe
hashes only. Its checker passes with `--require-ready`.

The corresponding same-request retry is
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260628-r19-colab-tpu-72b-four-layer-loader-correct-python/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`.
It used the correct Colab CLI Python and cleaned up the temporary Kaggle CUDA
kernel, but only stage0 submitted; the Colab TPU stage hit an HTTPError/404
before stage1. The follow-up runtime probe
`dist/colab-tpu-runtime-stability-20260628-r10-after-r19-httperror/colab_tpu_runtime_stability_probe.json`
confirmed the Colab runtime proxy was disconnected, and the reacquire attempt
`dist/colab-tpu-session-20260628-r11-reacquire-after-r19-404/colab_tpu_session_probe.json`
returned HTTP 503 `colab_assignment_resource_unavailable`. Treat this as Colab
runtime lifecycle/allocation instability, not as a 72B capacity failure.

The current dense max-search status is
`dist/three-accelerator-dense-max-parameter-search-20260628-r8-colab-72b-stage-loader-bridge-http404/three_accelerator_dense_max_parameter_search.json`,
which passes `scripts/three_accelerator_dense_max_parameter_search_check.py`.
It records `max_stage_loaded_parameter_class=72b`,
`max_tpu_executed_parameter_class=72b`, and
`max_successful_same_request_decode_parameter_class=32b`. The active goal is
still incomplete until a public-safe report proves dense/full-precision 72B
GPU+TPU+CPU same-request 1-token decode over all layers. Future retries should
use the updated `scripts/gpu_tpu_cpu_same_request_runtime_bridge_probe.py`,
whose Kaggle CUDA stage now exits after the requested token count is accepted
instead of waiting until kernel timeout.

Latest superseding live result: after the GPU fast-exit fix, Colab V5E1 was
reacquired again in
`dist/colab-tpu-session-20260628-r12-reacquire-after-fast-gpu-fix/colab_tpu_session_probe.json`
and passed a one-round runtime check in
`dist/colab-tpu-runtime-stability-20260628-r11-after-r12-reacquire/colab_tpu_runtime_stability_probe.json`.
The same-request retry
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260628-r20-colab-tpu-72b-four-layer-fast-gpu-exit/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`
completed with accepted backends `["cpu","cuda","jax_tpu"]`,
`stage0=stage1=stage2=1`, two activation handoff hashes, one generated-token
hash, and verified deletion of the temporary Kaggle CUDA kernel. Stage1 used
Colab `TPU v5 lite` and executed the real `Qwen/Qwen2.5-72B-Instruct` layers
32-36 stage-owned loader: 48 assigned keys, about 6.539GB logical execution
tensor bytes, 4 executed layers, no missing stage keys, stage-local KV-cache
verified, and public-safe hashes only.

The current dense max-search artifact is now
`dist/three-accelerator-dense-max-parameter-search-20260628-r9-colab-72b-stage-bridge-not-full-decode/three_accelerator_dense_max_parameter_search.json`;
its checker passes and records `max_stage_loaded_parameter_class=72b`,
`max_tpu_executed_parameter_class=72b`,
`same_request_72b_import.same_request_stage_decode_verified=true`,
`same_request_72b_import.same_request_full_model_decode_verified=false`, and
`max_successful_same_request_decode_parameter_class=32b`. This is a successful
Colab-as-TPU 72B stage bridge, not a full all-layer 72B decode and not active
goal completion.

Full-72B engineering continuation: `scripts/kaggle_32b_full_heterogeneous_probe.py`
now accepts `--tpu-provider colab_cli`, `--colab-session-name`,
`--colab-session-config`, and `--stage-launch-stagger-seconds`. The full
heterogeneous TPU stage worker can use the Colab runtime proxy for the Qwen
stage-owned loader, while the completion gate still requires full Qwen 72B
layer coverage and all stages to finish before setting
`gpu_tpu_cpu_72b_same_request_verified=true`. Tests cover the Colab provider
branch and cleanup for non-accepted Kaggle kernel pushes.

The first bounded full all-layer Colab-TPU live attempt is
`dist/kaggle-72b-full-heterogeneous-colab-tpu-live-20260628-r1-10stage/kaggle_32b_full_heterogeneous_probe.json`.
It used 10 contiguous Qwen 72B ranges covering layers 0..80 with 4 GPU stages,
1 Colab TPU stage, and 5 CPU stages, but it did not complete any Coordinator
stage task: `generated_token_count=0` and
`gpu_tpu_cpu_72b_same_request_verified=false`. The failure was resource and
scheduling related, not a 72B math/parity success or failure: gpu-shard1 hit
the Kaggle GPU session limit, cpu-stage7 push hit HTTP 429, several CPU kernels
later became `CANCEL_ACKNOWLEDGED`/UNKNOWN after timeout, and the Colab runtime
was not reusable afterward. A residual non-accepted gpu-shard1 kernel was
manually deleted, and the probe now attempts deletion for that path.

The current checked max-search artifact is
`dist/three-accelerator-dense-max-parameter-search-20260628-r10-full-72b-colab-attempt-resource-blocked/three_accelerator_dense_max_parameter_search.json`.
It records `max_stage_loaded_parameter_class=72b`,
`max_tpu_executed_parameter_class=72b`, and
`max_successful_same_request_decode_parameter_class=32b`; therefore the active
goal is still incomplete. The latest Colab reacquire attempt after r1 is
`dist/colab-tpu-session-20260628-r14-reacquire-after-full-r1/colab_tpu_session_probe.json`,
which returned HTTP 503. Next live retry should first reacquire Colab V5E1 and
then use `--stage-launch-stagger-seconds` to reduce Kaggle 429/push contention.

Latest superseding Colab reacquire status: the bounded retry probe
`scripts/colab_tpu_reacquire_retry_probe.py` and checker
`scripts/colab_tpu_reacquire_retry_check.py` were added so Colab TPU recovery
is recorded as a public-safe artifact rather than ad hoc single attempts. The
current retry report is
`dist/colab-tpu-reacquire-retry-20260628-r1-v5e1-v6e1-short/colab_tpu_reacquire_retry_probe.json`;
its checker passes, but it reports `colab_tpu_reacquire_ready=false`, with V5E1
returning HTTP 503 and V6E1 returning HTTP 400. No endpoint, proxy URL, proxy
token, OAuth token, credentials, or private runtime state are public.

The current max-search status is now
`dist/three-accelerator-dense-max-parameter-search-20260628-r11-colab-reacquire-retry-current-blocked/three_accelerator_dense_max_parameter_search.json`,
checked by `scripts/three_accelerator_dense_max_parameter_search_check.py`.
It imports the Colab reacquire retry artifact explicitly and records
`max_stage_loaded_parameter_class=72b`, `max_tpu_executed_parameter_class=72b`,
`max_successful_same_request_decode_parameter_class=32b`, and
`failure_stage=colab_tpu_reacquire_not_ready`. This is current blocker
evidence only; the active 72B goal remains incomplete until a dense/full-
precision GPU+TPU+CPU all-layer 72B same-request 1-token decode succeeds.

The next full-72B live retry should use a lower-concurrency topology after
Colab V5E1 is reacquired and passes a short stability probe: stage ranges
`[[0,8],[8,16],[16,24],[24,32],[32,36],[36,44],[44,52],[52,60],[60,70],[70,80]]`
and groups `gpu-shard0:[0,1]`, `gpu-shard1:[2,3]`,
`web-tpu-stage4:[4]`, `cpu-tail:[5,6,7,8,9]`. This still covers all 80 Qwen
72B layers and all three accelerator families, but reduces CPU Kaggle pushes
from five kernels to one. The completion gate for this topology is covered by
`tests/test_kaggle_32b_full_heterogeneous_probe.py`.

Latest retry after that plan: the bounded V5E1 retry
`dist/colab-tpu-reacquire-retry-20260628-r2-v5e1-before-full72b/colab_tpu_reacquire_retry_probe.json`
ran three Colab V5E1 assignment attempts and all returned HTTP 503. Its
checker passes and confirms public-safe redaction. The current max-search
artifact is now
`dist/three-accelerator-dense-max-parameter-search-20260628-r12-colab-v5e1-retry-still-503/three_accelerator_dense_max_parameter_search.json`;
its checker passes and records `max_stage_loaded_parameter_class=72b`,
`max_tpu_executed_parameter_class=72b`,
`max_successful_same_request_decode_parameter_class=32b`, and
`failure_stage=colab_tpu_reacquire_not_ready`. No 72B full all-layer live run
was started from this unavailable Colab state.

Latest recovery hardening after r12: the local `ct-colab-tpu-v5e1` Colab
session still had endpoint/proxy metadata, but
`dist/colab-tpu-runtime-stability-20260628-r12-existing-session-before-full72b/colab_tpu_runtime_stability_probe.json`
showed it was no longer usable: the Jupyter kernels endpoint returned HTTP 404,
no observations completed, and no TPU device was seen. A longer allocation
retry at
`dist/colab-tpu-reacquire-retry-20260628-r3-v5e1-longer-before-full72b/colab_tpu_reacquire_retry_probe.json`
ran five V5E1 attempts, all HTTP 503. The Colab session probe now supports
`--cleanup-before-tpu`, and the retry probe forwards that option; this lets the
recovery path unassign stale remote TPU assignments before requesting a fresh
V5E1 runtime. The cleanup-before retry
`dist/colab-tpu-reacquire-retry-20260628-r4-clean-before-v5e1/colab_tpu_reacquire_retry_probe.json`
still returned HTTP 503 on both V5E1 attempts, with checker passing.

The current max-search artifact is now
`dist/three-accelerator-dense-max-parameter-search-20260628-r13-colab-clean-before-still-503/three_accelerator_dense_max_parameter_search.json`.
Its checker passes and still records `max_stage_loaded_parameter_class=72b`,
`max_tpu_executed_parameter_class=72b`,
`max_successful_same_request_decode_parameter_class=32b`, and
`failure_stage=colab_tpu_reacquire_not_ready`. This remains an external Colab
TPU availability blocker, not a 72B full all-layer same-request decode.

Latest authuser recovery attempt after r13: the Colab session probe now accepts
`--authuser`, and the reacquire retry probe accepts `--authusers` for public-
safe authuser index rotation. Tests cover URL generation and retry command
forwarding. The retained authuser rotation artifact is
`dist/colab-tpu-reacquire-retry-20260628-r5-authuser-rotation-v5e1/colab_tpu_reacquire_retry_probe.json`:
authuser indexes 0, 1, and 2 each attempted V5E1 with cleanup-before enabled,
and all three returned HTTP 503. Its checker passes. The current max-search
artifact is
`dist/three-accelerator-dense-max-parameter-search-20260628-r14-colab-authuser-rotation-still-503/three_accelerator_dense_max_parameter_search.json`;
its checker passes and still records `max_stage_loaded_parameter_class=72b`,
`max_tpu_executed_parameter_class=72b`,
`max_successful_same_request_decode_parameter_class=32b`, and
`failure_stage=colab_tpu_reacquire_not_ready`. Do not run the low-concurrency
72B full all-layer decode until a TPU runtime is reacquired and passes the
stability probe.

Latest Kaggle Web TPU retry after r14: the UI state probe
`dist/kaggle-web-tpu-ui-state-20260628-r15-current-before-web-retry/kaggle_web_tpu_ui_state_probe.json`
showed Start Session visible and no current Jupyter runtime. The bounded
start/wait probe
`dist/kaggle-web-tpu-start-wait-20260628-r15-start-from-ui/kaggle_web_tpu_start_wait_probe.json`
expanded Session options, selected `TPU v5e-8`, clicked Start Session, and
waited 1200 seconds. It remained public-safe but did not become runtime-ready:
queue/starting stayed visible and no Jupyter frame/session/kernel appeared.
The follow-up execution-channel probe
`dist/kaggle-web-tpu-execution-channel-probe-20260628-r11-after-start-wait-timeout/kaggle_web_tpu_execution_channel_probe.json`
failed with `web_tpu_jupyter_execute_timeout`, no TPU device, and no small JAX
cell.

The current max-search artifact is now
`dist/three-accelerator-dense-max-parameter-search-20260628-r15-colab-and-kaggle-web-tpu-unavailable/three_accelerator_dense_max_parameter_search.json`.
Its checker passes and still records `max_stage_loaded_parameter_class=72b`,
`max_tpu_executed_parameter_class=72b`, and
`max_successful_same_request_decode_parameter_class=32b`. This is current
evidence that both Colab TPU allocation and Kaggle Web TPU execution are
unavailable; it is not 72B full all-layer same-request decode success.

Latest Web TPU follow-up after r15: the current UI state probe
`dist/kaggle-web-tpu-ui-state-20260628-r16-after-r15-wait-current/kaggle_web_tpu_ui_state_probe.json`
still showed the Kaggle notebook in `Draft Session Starting` state with no
Jupyter frame/session/kernel. A second bounded wait at
`dist/kaggle-web-tpu-start-wait-20260628-r16-continue-starting/kaggle_web_tpu_start_wait_probe.json`
waited 600 more seconds and still ended with `session_starting_text_visible=true`,
no queue text, no Jupyter runtime, no session, and no kernel. The current
max-search artifact is
`dist/three-accelerator-dense-max-parameter-search-20260628-r16-kaggle-web-tpu-still-starting/three_accelerator_dense_max_parameter_search.json`;
its checker passes and still records `max_successful_same_request_decode_parameter_class=32b`.
The 72B full all-layer run must wait until either Kaggle Web TPU execution or
Colab TPU runtime is actually ready.

## Naming and Positioning

CrowdTensor is the open-source network vision: fault-tolerant AI swarms built from ordinary home compute.

CrowdTensorD is the current Alpha daemon/control plane inside this repository. It proves leasing, validation, recovery, observability, and operator control before the project claims real model-scale home GPU aggregation.

The target audience is:

- home open-model players with limited local compute
- remote Miner operators who can contribute controlled Linux/container capacity
- browser experimenters interested in WebRTC/WebGPU-style participation
- protocol contributors building reliable distributed AI workload contracts

The project should be honest about status. It is an Alpha control plane with a
bounded external 7B sharded-inference proof, a temporary-Coordinator 32B AWQ
upper-bound crossing proof, and a temporary-Coordinator full-precision 32B
4*T4 + 5*CPU two-token heterogeneous feasibility proof with stage-local
KV-cache reuse, not yet a production DePIN network or real LLM deployment
platform.

## Durable Architecture Layering

Use this three-layer model when planning future work. It is meant to keep the
project focused on the real technical north star: cross-device large-model
inference for ordinary machines, with later training and fine-tuning support.

1. Core technology layer: solves how the model actually runs across devices.
   This layer owns large-model runtime adapters such as vLLM, SGLang,
   TensorRT-LLM, llama.cpp/GGUF/RPC, or Petals-like workers; layer, pipeline,
   tensor, expert, and prefill/decode partitioning; activation and KV-cache
   transport; streaming token generation; batching; heterogeneous CPU/GPU
   placement; correctness checks; redundancy; and future LoRA/DiLoCo-style
   training or fine-tuning. If changing a component changes whether a model can
   run, how large it can be, or how fast it runs, treat it as core technology.
2. Control layer: decides who can participate, which work they receive, how
   failures recover, how usage is metered, and how trust or economics are
   enforced. This layer owns the Coordinator, sessions, task leases, heartbeats,
   result ledgers, Miner admission, operator and user identities, tenant/project
   policy, quotas, rate limits, capability routing, P2P provider records, trust
   tiers, quarantine, overrides, accounting summaries, settlement drafts, future
   rewards, incentives, staking/slashing, abuse detection, and audit logs. It
   must not embed model math; it schedules and governs explicit workload
   contracts.
3. User-facing layer: makes the system usable by ordinary operators, Miners, and
   request users. This layer owns CLI flows such as `infer`, `generate`, `serve`,
   `join`, and `swarm-bootstrap`; one-command bootstrap, quickstart, install,
   route-prep, and runner scripts; dashboards and APIs; docs and runbooks;
   support bundles; redacted reports; onboarding and release gates; diagnostics;
   next-command guidance; and user-visible cost, health, and answer surfaces. It
   should wrap lower layers without inventing policy or runtime semantics.

Cross-cutting requirements such as security, privacy, observability, artifact
redaction, testing, and performance apply to all three layers. Do not treat them
as a fourth product layer. For example, core technology must avoid leaking
activations/KV-cache data, the control layer must protect tokens and enforce
policy, and the user-facing layer must keep prompts, generated text, credentials,
lease material, and private runtime state out of public artifacts.

This separation should guide task selection. Near-term user-facing and control
work is useful, but it should build on the core technical facts that a real 7B
cross-device GPU proof, a bounded 32B AWQ upper-bound crossing proof, and a
full-precision 32B 4*T4 + 5*CPU multi-token heterogeneous proof with
stage-local KV-cache reuse now exist while production Swarm Inference still
needs throughput work, batch/sequential validation, robust scheduling,
production routing, and operator UX.

## Current Engineering Method, Progress, and Planning

Use an evidence-first development method. Every meaningful capability should
land with a bounded command or script, a versioned public-safe report schema,
redacted JSON/Markdown/Support Bundle artifacts, a CI-safe checker, explicit
cleanup behavior for private live resources, and a clear "not yet" boundary.
Do not count a design, local fixture, imported report, queued cloud runtime, or
single happy-path live run as broader production readiness. Preserve prompt,
generated text, token id, activation, KV-cache, credential, lease,
idempotency, cookie, and private runtime redaction in all shareable artifacts.

Use provider-backed live experiments as temporary proof vehicles, not as the
product architecture. Kaggle GPU/CPU/TPU kernels may be used to create bounded
external evidence when the account quota and provider UI/API allow it, but
proofs must use private kernels/packages, delete temporary kernels after
collection, avoid publishing inline private payloads, and respect provider
limits instead of depending on multi-account limit bypass. A live queue or
runtime allocation attempt is useful scheduling evidence only when recorded as
such; it is not runtime inference success until a completed public-safe report
proves the model path ran.

Current core progress is strongest on feasibility proof, not production
serving. The retained evidence includes a real external 7B two-stage Kaggle T4
x2 proof, a 32B AWQ four-stage upper-bound crossing across two T4 x2 kernels,
and a full-precision 32B heterogeneous 4*T4 + 5*CPU two-token proof with
stage-local KV-cache reuse. The TPU path now has a reusable private-kernel
probe plus authenticated web-UI evidence that TPU v5e-8 can be queued,
allocated, and used by JAX for a tiny synthetic causal-LM decode across 8 TPU
v5 lite devices; it is still not Hugging Face/Qwen or large-model TPU
inference. Control and
user-facing layers have many Alpha/Beta contracts, runbooks, redacted reports,
release/onboarding gates, and public swarm/product surfaces; they are still
Coordinator-backed Alpha/Beta surfaces, not production P2P, economic
settlement, abuse-resistant trust, or ordinary-user large-model serving.

Plan future work in this order unless a task explicitly says otherwise:

1. Harden the core inference path from feasibility to repeatable serving:
   longer multi-token runs, batch and sequential request validation,
   throughput/TTFT measurement, stage requeue for large-model paths,
   memory-pressure crossing evidence beyond slot-count crossing, and
   production-like adapter choices for GGUF/llama.cpp, HF stage-selective
   CUDA, vLLM/SGLang/TensorRT-LLM, or Petals-like workers behind explicit
   interfaces.
2. Make ordinary multi-machine GPU inference usable: one-command Coordinator,
   Miner join, model partition preparation, health checks, safe streaming
   answer surface, support bundles, and recovery guidance that wrap the core
   path without overstating production readiness.
3. Build the control layer needed for real users: capability discovery,
   scheduling, admission, quotas, rate limits, identity, trust/quarantine,
   abuse controls, audit logs, accounting summaries, cleanup, and future
   incentive hooks. Keep policy out of model math and keep model execution
   contracts explicit.
4. Continue hardware expansion only with bounded evidence: retry Kaggle TPU
   when runtime allocation is available, first prove the JAX synthetic causal
   LM on TPU, then add a real TPU LLM adapter if feasible. Treat TPU support
   as optional acceleration until runtime proof exists.
5. After the above is stable, move toward production-facing surfaces:
   dashboards, reliable public routing, NAT/tunnel strategy, stronger security
   model, operator economics, and broader community demos.

## Current Completed Capabilities

The project currently includes:

- Core Technology Validation Status: `scripts/core_technology_validation_status_pack.py`
  emits `core_technology_validation_status_v1` as the canonical retained
  status over current core validation evidence. The retained status artifact is
  `dist/core-technology-validation-status-20260616/core_technology_validation_status.json`;
  it imports the fully automated `gpt2-xl` Kaggle GPU small-tier proof, the
  local tiny Llama-like HF two-stage runtime proof at
  `dist/real-llm-llama-like-local-smoke-20260615/real_llm_sharded_evidence.json`,
  the local stage-selective safetensors materialization/runtime proof, and the
  retained Kaggle T4 x2 7B validation report at
  `dist/large-model-kaggle-stage-selective-hf-7b-manual-rope-20260616/large_model_kaggle_validation.json`.
  The 7B report ran `Qwen/Qwen2.5-7B-Instruct` through
  `hf_transformers_stage_selective_cuda`, with stage0 on `cuda:0`, stage1 on
  `cuda:1`, `generated_token_count=1`, `memory_peak_mb=14608`,
  `real_7b_runtime_verified=true`, `sharded_path_verified=true`,
  `multi_worker_sharded_path_verified=true`, `core_validation_ready=true`,
  public-safe redaction, and cleanup of temporary Kaggle kernel
  `xuyuhaosuyi/crowdtensor-large-llm-81608591`. The check still validates that
  the small-tier Kaggle proof, local tiny Llama-like smoke, and local
  stage-selective loading proof cannot by themselves be treated as 7B/8B
  completion. This is a bounded external core validation proof, not production
  P2P/NAT traversal, not GGUF/llama.cpp RPC success, not a GPU marketplace, not
  training/fine-tuning, and not large-model throughput serving.
- Large-Model Shard Alpha: `crowdtensor large-model-shard` emits
  `large_model_shard_alpha_v1` through
  `scripts/large_model_shard_alpha_pack.py` and is validated by
  `scripts/large_model_shard_alpha_check.py` as
  `large_model_shard_alpha_check_v1`. This is the core technology layer
  Alpha/MVP for moving beyond tiny/small-model stage proofs toward real
  cross-device large-model inference. It targets GGUF / llama.cpp RPC first,
  because that is a practical consumer-device runtime adapter, and preserves
  versioned artifacts for `large_model_runtime_adapter_v1`,
  `large_model_partition_manifest_v1`, `large_model_sharded_generate_v1`,
  `large_model_shard_benchmark_v1`, `large_model_manifest_v1`, and
  `large_model_device_profile_v1`. The default model is a 7B-class
  `Q4_K_M` planning fixture with 32 layers, two controlled local/LAN-style RPC
  workers, layer-range placement, memory budget checks, latency/bandwidth
  metadata, controlled endpoint checks, serving hooks for stream events,
  bounded batch requests, cancellation, KV/prefix cache metadata, and
  health-aware route metadata. The benchmark harness records TTFT, tokens/s,
  p50/p95 when provided, memory, network bytes/token, cache hit/miss metrics,
  correctness status, failure diagnosis, and single-device fallback vs sharded
  adapter comparison. CI and ordinary development runs must keep
  `real_runtime_verified: false`, `evidence_scope: fixture-contract-plan`, and
  `large_model_7b_real_runtime_deferred` unless `--real-benchmark-report`
  imports actual controlled LAN/VPN/runtime metrics. Public JSON, Markdown,
  terminal summaries, and Support Bundle outputs must keep raw prompts,
  generated text, generated token ids, activations, KV-cache material,
  credentials, leases, idempotency material, private env files, and registries
  out of reports. Preserve the boundary that this is controlled LAN/VPN/local
  process evidence, not public RPC security, not production Petals/Hivemind
  parity, not Coordinator-free public P2P, not NAT traversal, not GPU
  marketplace settlement, not training/fine-tuning, and not a large-model
  serving SLA.
- Core Technology Inference RC: `crowdtensor large-model-shard-rc` emits
  `core_technology_inference_rc_v1` through
  `scripts/large_model_inference_rc_pack.py` and is validated by
  `scripts/large_model_inference_rc_check.py` as
  `core_technology_inference_rc_check_v1`. It preserves the Large-Model Shard
  Alpha child report and advances the core technology layer with
  `large_model_runtime_adapter_interface_v1`,
  `large_model_runtime_adapter_probe_v2`, `large_model_device_profile_v2`,
  `large_model_partition_manifest_v2`, `large_model_runner_result_v1`,
  `large_model_benchmark_v2`, `large_model_correctness_summary_v1`, and
  `large_model_serving_hooks_v1`. The first supported adapter remains
  GGUF / llama.cpp RPC for controlled LAN/VPN/local-process operation. The
  runtime probe checks llama.cpp client/server binaries, version digests, local
  model metadata, RPC endpoint health, command templates, sanitized log policy,
  and controlled-network boundaries. Device profiling uses local CPU/RAM and
  optional GPU/VRAM probes or JSON imports, with usable memory,
  latency/bandwidth, backend capabilities, and endpoint control checks.
  Partition v2 adds tensor split planning, KV-cache reservation,
  prefill/decode memory estimates, single-device fallback feasibility,
  multi-worker feasibility, and blocker details. The runner supports `plan`,
  `fixture`, and `real`; real mode is bounded to short prompts,
  `--max-new-tokens <= 8`, and a maximum 20 minute timeout, with process
  cleanup and redacted logs. `--real-run-report` may import a completed
  public-safe real run with TTFT, tokens/s, wall time, generated token count,
  and output digest; `--real-benchmark-report` can supplement metrics but must
  not by itself set the RC real-runtime claim. Benchmark v2 records TTFT,
  tokens/s, p50/p95 when available, wall time, memory, network bytes/token,
  cache hit/miss metadata, single-device fallback vs sharded adapter comparison,
  correctness, and failure diagnosis. Correctness reports token count, output
  digest, optional baseline digest match, and model/adapter/partition hash
  consistency. Serving hooks preserve streaming event samples, bounded batch
  request shape, cancellation/timeout fields, KV/prefix metadata, and
  health-aware route metadata. Future vLLM, SGLang, TensorRT-LLM, and
  Petals-like adapters are explicit `unsupported_runtime_backend` descriptors
  behind the same interface. CI-safe runs without GGUF, llama.cpp binaries,
  reachable RPC workers, or sufficient hardware should still pass with
  `ok: true`, `real_runtime_verified: false`,
  `real_7b_runtime_verified: false`, `core_technology_real_7b_runtime_not_verified`,
  and concrete blockers. Public JSON, Markdown, terminal summaries, and Support
  Bundle outputs must continue to keep raw prompts, generated text, generated
  token ids, activations, KV-cache material, credentials, leases, idempotency
  material, private env files, and registries out of reports. Preserve the
  boundary that this is inference core technology only, controlled
  LAN/VPN/local-process evidence, not public RPC security, not production
  Petals/Hivemind parity, not Coordinator-free public P2P, not NAT traversal,
  not GPU marketplace settlement, not training/fine-tuning, and not a
  large-model serving SLA.
- Core Technology Handoff RC: `crowdtensor core-tech-handoff` emits
  `core_technology_handoff_rc_v1` through
  `scripts/core_technology_handoff_pack.py` and is validated by
  `scripts/core_technology_handoff_check.py` as
  `core_technology_handoff_rc_check_v1`. This is the handoff package that lets
  control-layer, user-facing-layer, and future permissions/trust/billing work
  proceed without repeatedly reworking the core inference foundation. It embeds
  the Inference RC report and adds `core_technology_deployment_runbook_v1`,
  `core_technology_next_layer_contract_v1`,
  `core_technology_adapter_conformance_v1`,
  `core_technology_test_gate_summary_v1`, and
  `core_technology_handoff_rc_support_bundle_v1`. Preserve stable entrypoints:
  `crowdtensor large-model-shard-rc`, `crowdtensor core-tech-handoff`,
  `scripts/core_technology_handoff_pack.py`, and
  `scripts/core_technology_handoff_check.py`. The deployment runbook must cover
  local fixture, local real runtime, LAN/VPN two-worker runtime, retained
  real-run import, troubleshooting, and cleanup including a process leak check.
  The next-layer contract must expose control-layer scheduling inputs,
  user-layer safe status/stream/batch schemas, and core signals for later
  permissions/trust/billing work such as runtime backend, model id, partition
  hash, real-runtime status, benchmark metrics, correctness digest, route
  health, and process cleanup status. Adapter conformance must keep
  llama.cpp/GGUF/RPC as the first supported adapter and vLLM, SGLang,
  TensorRT-LLM, and Petals-like backends as explicit
  `unsupported_runtime_backend` descriptors behind the same interface. CI-safe
  handoff reports without GGUF, llama.cpp binaries, reachable RPC workers, or
  sufficient hardware should still pass with `ok: true`,
  `real_runtime_verified: false`, `real_7b_runtime_verified: false`,
  `core_technology_real_runtime_not_verified`,
  `core_technology_handoff_fixture_or_import_ready`, and blockers including
  `external_real_runtime_resources_required`. `--real-run-report` may mark the
  handoff real-verified only when the embedded Inference RC runner/import
  evidence is real. Public Handoff artifacts must continue to keep raw prompts,
  generated text, generated token ids, activations, KV-cache material,
  credentials, leases, idempotency material, private env files, and registries
  out of reports. Preserve the boundary that this is a core technology handoff
  artifact, not accounts, permissions, billing, trust scoring, incentives,
  staking/slashing, production public P2P/NAT traversal, production
  Petals/Hivemind parity, training/fine-tuning, or a large-model serving SLA.
- Large-Model Kaggle Validation: `crowdtensor large-model-kaggle-validate`
  emits `large_model_kaggle_validation_v1` through
  `scripts/large_model_kaggle_validation_pack.py` and validates public-safe
  reports with `scripts/large_model_kaggle_validation_check.py`. This is the
  bounded fresh Kaggle GPU proof path for the core technology layer. It creates
  a private GPU script kernel, probes Kaggle hardware, runs selected model
  tiers, first tries to download `large_model_kaggle_validation_run.json`, uses
  a full output fallback when that report is missing, cleans up the temporary
  kernel by default, and preserves redacted JSON/Markdown/support evidence. The
  Kaggle script must keep writing partial run reports after hardware probe,
  runtime preparation, RPC startup, and each tier attempt so killed kernels
  still leave diagnostics when Kaggle retains outputs. The strict completion
  signal is `core_validation_ready`, which may be true only when the same report
  proves real token generation, a 7B/8B-class model, Kaggle CUDA runtime
  execution, and the intended sharded/RPC path. Keep `real_runtime_verified`,
  `real_7b_runtime_verified`,
  `gpu_runtime_verified`, `sharded_path_verified`,
  `multi_worker_sharded_path_verified`, and `core_validation_ready` separate so
  partial evidence cannot overclaim. Prefer `--accelerator NvidiaTeslaT4` for
  the main Kaggle validation; a 2026-06-13 hardware probe at
  `dist/kaggle-gpu-shape-probe-20260613073043/attempt_summary.json` verified two
  `Tesla T4` devices and Torch CUDA visibility, while the generic `GPU` request
  previously assigned single P100 runs. The 2026-06-13 T4 x2 source-CUDA/RPC
  attempts verified `CMAKE_CUDA_ARCHITECTURES=75`, `GGML_CUDA_NO_VMM=ON`,
  `GGML_RPC=ON`, successful CUDA llama.cpp build with
  `--cuda-build-jobs 2 --cuda-build-timeout-seconds 5400`, and live RPC workers.
  The strongest retained RPC blocked report is
  `dist/large-model-kaggle-validation-t4x2-rpc-small-telemetry-inplace-20260613/large_model_kaggle_validation.json`:
  it hides CUDA from the local llama.cpp client in RPC mode so only RPC workers
  own GPU placement, verifies one live CUDA0 RPC worker, downloads the 1.5B GGUF,
  and monitors `llama-cli --rpc 127.0.0.1:50052` for about 451 seconds before
  Kaggle terminates the run without generated tokens. Re-importing the raw report
  now emits `resource_pressure_summary` and diagnosis codes showing
  `cgroup_memory_peak_ratio=0.9345`, `gpu_memory_used_peak_ratio=0.0702`,
  `large_model_kaggle_cgroup_memory_pressure`, and
  `large_model_kaggle_container_memory_pressure_not_vram`. The blocker is Kaggle
  container memory pressure during single-Notebook llama.cpp RPC execution, not
  T4 assignment, CUDA build, model download, 7B size alone, or two-worker tensor
  split startup. The 7B CLI fallback evidence at
  `dist/large-model-kaggle-validation-t4x2-cli-7b-20260613-r1/large_model_kaggle_validation.json`
  verified T4 x2 hardware, CUDA llama.cpp build, Qwen2.5 7B Q2_K GGUF download,
  and `llama_cpp_cli` run start, but failed before token generation with
  `cgroup_memory_peak_ratio=0.9335`, `disk_min_free_bytes=335552512`, and
  `large_model_kaggle_disk_pressure`; GPU memory stayed low
  (`gpu_memory_used_peak_ratio=0.1075`). The bounded r2 retry at
  `dist/large-model-kaggle-validation-t4x2-cli-7b-20260613-r2/large_model_kaggle_validation.json`
  added real Kaggle slug resolution, minimal `llama-runtime` compaction,
  `CUDA_CACHE_DISABLE=1`, `CUDA_MODULE_LOADING=LAZY`, small `-b/-ub 32`
  llama.cpp batches, and report-write fallback. It still ended with Kaggle
  `Killed` plus `No space left on device`, and the temporary kernel was deleted.
  Current conclusion: a single Kaggle Notebook cgroup is not a reliable place to
  validate llama.cpp RPC or 7B GGUF CLI execution for this goal. The next aligned
  route is multi-kernel or true partial-weight stage-local loading, not more
  single-container retries. `real_llm_artifact_v1` and real-LLM workload specs
  now include `execution_support` / `execution_family`: the current HF splitter
  supports only GPT-2/tiny-GPT module structure, records
  `real_llm_true_partial_weight_loading_missing`, and classifies
  Llama/Qwen/Mistral/Gemma/Phi-style 7B/8B candidates as `llama_like` with
  `real_llm_llama_like_stage_adapter_missing`. Unsupported large-model
  candidates fail before runtime load so Kaggle runs do not waste quota
  downloading/loading models the current adapter cannot partition. The core
  technical gap is therefore two-layered: Kaggle single-container resource
  pressure for llama.cpp RPC/CLI, plus missing true partial-weight stage
  adapters for large HF causal LM families. The current bounded success path is
  to use the existing Public Swarm GPU two-stage Kaggle wrapper with
  `--hf-model-id gpt2-xl --real-llm-partition-mode stage-local`; `gpt2-xl` is
  reported as a GPT-2-family 1.5B small-tier candidate via `execution_support`.
  If that succeeds, it is real Kaggle GPU sharded small-tier evidence only, not
  7B/8B completion. Keep `core_validation_ready=false` for these reports.
  The first three `gpt2-xl` Kaggle GPU small-tier attempts on 2026-06-13 are not
  successful inference evidence. The retained blocked reports are
  `dist/gpt2-xl-small-tier-kaggle-20260613201430/public_swarm_gpu_inference_beta_kaggle_auto.json`
  `dist/gpt2-xl-small-tier-kaggle-timeoutfix-20260613202418/public_swarm_gpu_inference_beta_kaggle_auto.json`,
  and
  `dist/gpt2-xl-small-tier-kaggle-pollfix-20260613204918/public_swarm_gpu_inference_beta_kaggle_auto.json`;
  all packaged private CUDA stage kernels, deleted the temporary kernels, and
  ended before any generated token evidence. The first attempt exposed a
  timeout passthrough gap in `remote_real_llm_sharded_beta_pack.py`. The second
  showed that a single `/state` HTTP timeout during external stage polling could
  still abort the wait around 356 seconds despite larger outer timeouts. The
  remote sharded wrappers now preserve timeout arguments in recommended rerun
  commands, retry transient `/state` poll failures until
  `remote_timeout_seconds`, and surface `remote_state_poll_retry` without
  weakening readiness. The post-fix third run reached live Kaggle P100 CUDA
  stage kernels but the Coordinator repeatedly recorded
  `join_policy_backend_mismatch`: the generated stage invite registry defaulted
  to CPU policy while the CUDA miners advertised `backend=cuda`. The generated
  real-LLM Live RC registry policy now writes per-stage `stage`, backend
  `cuda`/`cpu`, and `hf_model_id` into `create_invite()`. A 2026-06-14
  policy-fix run then completed a real Kaggle P100 CUDA small-tier split proof
  for `gpt2-xl`, but required manually draining Coordinator stdout/stderr pipes
  after the Kaggle stage result had already been posted. The durable retained
  proof is the follow-up fully automated log-fix run at
  `dist/gpt2-xl-small-tier-kaggle-logfix-20260614172932/public_swarm_gpu_inference_beta_kaggle_auto.json`
  (`sha256:83306de8c1f36e323a7ae554a76190ee8776e928ca1c0830f90142da5f6571d7`).
  It reported `ok: true`, `public_swarm_gpu_beta_kaggle_auto_ready`,
  `external_runtime_verified`, `cuda_runtime_available`,
  `decoded_tokens_match`, distinct stage Miners, valid stage assignment,
  `stage_local_partition_ready`, `stage0_partition_loaded`,
  `stage1_partition_loaded`, one generated token, redacted generated text,
  `kaggle_kernels_deleted`, public leak check hits `[]`, and no matching
  temporary Kaggle kernels after cleanup. The `real_llm_internet_beta`
  Kaggle-auto Coordinator now redirects stdout/stderr to log files and stores
  only redacted tails in public lifecycle output; the log-fix run shows
  `stdout_stderr_to_files: true` and a stage result `POST ... /result` 200 in
  the redacted tail. This is successful small-tier real GPU evidence only; it
  still does not validate the 7B/8B target because current reports keep
  `large_model_sharded_execution_ready=false`,
  `true_partial_weight_loading_ready=false`, and
  `real_llm_true_partial_weight_loading_missing`.
  The 2026-06-12 retained P100 attempts
  verified GPU hardware and kernel cleanup but did not complete the required
  7B/8B sharded/RPC proof: the initial GGUF release path exposed
  missing/shared-library and non-CUDA-runtime issues, the `source-cuda`
  llama.cpp path failed during CMake/CUDA build, and a Hugging Face CUDA
  compatibility smoke generated 4 tokens from `sshleifer/tiny-gpt2` on P100
  after installing CUDA 11.8-compatible PyTorch/Transformers. Later P100
  source-CUDA/RPC attempts showed the required fixes:
  `CMAKE_CUDA_ARCHITECTURES=60` for P100, `GGML_CUDA_NO_VMM=ON` to avoid the
  missing `CUDA::cuda_driver` target on Kaggle, and `GGML_RPC=ON` so the
  `rpc-server` target exists. With those fixes, Kaggle killed the run before a
  public-safe run report was retained, so it remains a runtime-resource blocker,
  not a 7B/RPC success. Preserve the tiny-model GPU smoke as partial
  real-runtime evidence only; it is not 7B/8B-class and not the intended
  sharded/RPC path. Do not claim the core technology layer is fully validated
  until a fresh report sets
  `core_validation_ready=true` for a 7B/8B sharded/RPC Kaggle run. Public
  artifacts must continue to redact raw prompts, generated text, generated
  token ids, activations, KV cache, credentials, leases, idempotency material,
  private Kaggle files, and model secrets.
- Product swarm bootstrap: `crowdtensor swarm-bootstrap` emits `crowdtensor_swarm_bootstrap_v1` and creates a local private setup directory with `operator_registry.json`, `miner_registry.json`, separate `coordinator.private.env` / admin `operator.private.env` / auditor `auditor.private.env` / accounting `accounting.private.env` / optional `tunnel.private.env`, role-scoped private `crowdtensor_operator_invite_v1` files for admin/auditor/accounting operators, private stage0/stage1 `crowdtensor_miner_join_invite_v1` files, stage packages with private `stage0.miner-package.tar.gz` / `stage1.miner-package.tar.gz` archives, matching `stage0.run-miner.sh` / `stage1.run-miner.sh` helpers, `stage0.handoff.sha256` / `stage1.handoff.sha256`, `stage_handoff_manifest.json`, plus `miner.join-code.txt` / `miner.invite.json` / `install.sh` / `doctor.sh` / `check_join.sh` / `support_bundle.sh` / `join.sh` / `MINER_JOIN.md`, executable `install_operator.sh`, `operator_quickstart.sh`, `start_control_plane.sh`, optional `start_tunnel.sh`, `tunnel_doctor.sh`, `start_discovery.sh`, `start_coordinator.sh`, `check_route.sh`, `verify_bootstrap.sh`, `handoff_doctor.sh`, `ready_for_handoff.sh`, `operator_status.sh`, `auditor_status.sh`, `accounting_status.sh`, `trust_review.sh`, `settlement_review.sh`, `operator_review.sh`, `check_generation.sh`, `submit_generation.sh`, and `SWARM_BOOTSTRAP.md`. Optional `--stage0-reward-account` and `--stage1-reward-account` values are private Beta accounting metadata stored in the matching stage invite and private Miner registry entry; public bootstrap/check/support/handoff reports expose only reward-account presence and must not expose account values. `install_operator.sh` creates `.crowdtensor-operator-venv` by default, can be relocated with `CROWDTENSOR_OPERATOR_VENV`, checks `crowdtensor`, `crowdtensord`, and `crowdtensor-miner`, supports `--dry-run`, `CROWDTENSOR_INSTALL_SPEC`, and `CROWDTENSOR_INSTALL_SOURCE`, and does not read `operator.private.env`, stage invites, join-code files, or start services; generated Coordinator/Operator scripts prefer that venv when present. `operator_quickstart.sh` is the recommended Operator host path: it installs the runtime when needed, starts `start_control_plane.sh` in the background, writes `run/control_plane.pid` and `logs/control_plane.log`, waits through `check_route.sh --check-ready`, and then runs `ready_for_handoff.sh`; `CROWDTENSOR_QUICKSTART_WAIT_SECONDS` adjusts the route wait and `CROWDTENSOR_QUICKSTART_SKIP_INSTALL=1` skips installation for externally managed runtimes. When `--peer-bootstrap` is supplied, private Miner invites include `crowdtensor_miner_join_discovery_v1`, `start_control_plane.sh` starts an optional operator-supplied tunnel command, discovery, and the Coordinator together, `start_tunnel.sh` sources private `tunnel.private.env` without echoing the raw command in public artifacts, `tunnel_doctor.sh` wraps `crowdtensor swarm-tunnel-doctor`, emits `crowdtensor_swarm_tunnel_doctor_v1` plus `tunnel_doctor.json`, checks the private tunnel env, provider binary, control-plane launcher, and Miner-facing URL without starting a tunnel, `ready_for_handoff.sh` chains `tunnel_doctor.sh`, `check_route.sh --check-ready`, `verify_bootstrap.sh`, and `handoff_doctor.sh` as the one-command operator handoff gate after the control plane is running, `start_discovery.sh` starts the matching P2P-lite/real-P2P discovery daemon for manual runs, stage `install.sh` creates `.crowdtensor-venv` with the default `[hf]` runtime when `crowdtensor` is not already on PATH, stage `doctor.sh` writes safe `miner_support_bundle.json` diagnostics and checks admission without starting the Miner, stage `check_join.sh` verifies the invite-code admission path without `--run`, stage `support_bundle.sh` writes safe diagnostics without raw join codes or Miner tokens and includes `crowdtensor_miner_local_environment_v1` with `local_environment_ready`, CLI, checksum, Python, and optional torch/CUDA probes, each stage archive plus its runner and checksum is the preferred copy unit for remote Miner handoff, the runner verifies `stageX.handoff.sha256`, supports `./stageX.run-miner.sh --quickstart` as the recommended install/diagnose/preflight/start path, keeps `--setup` followed by `--start`, `CROWDTENSOR_MINER_QUICKSTART_SKIP_INSTALL=1`, `--install --dry-run`, `--doctor`, `--check-only`, and `--support-bundle` as troubleshooting paths, safely extracts, preflights, and starts the Miner, and stage `join.sh` still defaults to `crowdtensor join --invite-code-file miner.join-code.txt --check-admission --expect-remote-coordinator --run`; `miner.invite.json` remains private compatibility material. `verify_bootstrap.sh` wraps `crowdtensor swarm-bootstrap-check --check-admission` for the operator's live no-claim gate after Coordinator start and before stage package handoff; `check_route.sh` wraps `crowdtensor coordinator-route` for no-token URL classification plus optional `/ready` checks and writes `coordinator_route.json` / `coordinator_route.md` with `join_options`, `recommended_join_option`, and `recommended_setup_command` templates for public HTTPS/reverse-proxy, tunnel, VPN/LAN, or explicit port-forwarding; it is not NAT traversal and does not join Miners or claim tasks; `operator_status.sh` sources only `private/operator.private.env`, runs `crowdtensor operator-status --include-admin-summaries --require-state --require-admin-summaries`, and writes `operator_status.json` / `operator_status.md` for read-only Coordinator, trust, accounting, and settlement triage; `auditor_status.sh`, `accounting_status.sh`, `trust_review.sh`, and `settlement_review.sh` split event/accounting/trust/settlement review across least-privilege env files, while `operator_review.sh` chains all five status/review scripts for a one-command operational review after the control plane is running; both bootstrap reports expose `bootstrap_handoff` with `remote_miners_ready`, `recommended_launcher`, `verify_before_handoff`, `one_command_handoff_check`, `manual_launchers.operator_quickstart`, `manual_launchers.tunnel_doctor`, `manual_launchers.ready_for_handoff`, `manual_launchers.operator_status`, `manual_launchers.trust_review`, `manual_launchers.settlement_review`, `manual_launchers.operator_review`, and `ready_to_copy_stage_packages` so operators can tell whether stage directories are ready to copy; `crowdtensor swarm-handoff-doctor` / `handoff_doctor.sh` emits `crowdtensor_swarm_handoff_doctor_v1`, `handoff_doctor.json`, and `handoff_doctor.md` with blockers and exact stage files to copy. `crowdtensor swarm-bootstrap-check` emits `crowdtensor_swarm_bootstrap_check_v1` and validates required files, `0600` private env/invite/join-code permissions, `0700` scripts, hashed registries, Coordinator/operator env separation, stage invite Coordinator URL consistency, stage reward-account metadata consistency, stage discovery metadata consistency, tunnel private-env and launcher readiness, `install_operator.sh`, `operator_quickstart.sh`, `operator_install_script_ready`, `operator_quickstart_script_ready`, `operator_scripts_use_operator_venv`, `start_control_plane.sh`, `start_tunnel.sh`, `tunnel_doctor.sh`, `ready_for_handoff.sh`, `start_discovery.sh`, `check_route.sh`, `operator_status.sh`, `auditor_status.sh`, `accounting_status.sh`, `trust_review.sh`, `settlement_review.sh`, `operator_review.sh`, `trust_review_script_ready`, `settlement_review_script_ready`, `operator_review_script_ready`, `install.sh`, `doctor.sh`, `check_join.sh`, `support_bundle.sh`, stage package archives, stage archive runners, stage handoff checksums, stage join-code consistency, optional `--expect-remote-miners` remote route readiness, optional live `/ready` checks through `--check-coordinator`, optional token-backed no-claim `/tasks/preflight` checks through `--check-admission`, and plaintext token, reward-account, tunnel command, or join-code leakage in scripts or public Markdown before stage package handoff. Preserve `swarm_bootstrap_ready`, `swarm_bootstrap_package_ready`, `crowdtensor_bootstrap_handoff_v1`, `coordinator_url_remote_route_ready`, `swarm_bootstrap_live_preflight_ready`, `private_invites_local_only`, `private_join_code_files_created`, `stage_join_code_files_match_invites`, `stage_reward_account_metadata_ready`, `stage_invite_discovery_metadata_ready`, `tunnel_private_env_ready`, `start_control_plane_script_ready`, `start_tunnel_script_ready`, `tunnel_doctor_script_ready`, `ready_for_handoff_script_ready`, `start_discovery_script_ready`, `check_route_script_ready`, `operator_status_script_ready`, `auditor_status_script_ready`, `accounting_status_script_ready`, `trust_review_script_ready`, `settlement_review_script_ready`, `operator_review_script_ready`, `operator_install_script_ready`, `operator_quickstart_script_ready`, `operator_scripts_use_operator_venv`, `stage_install_scripts_ready`, `stage_doctor_scripts_ready`, `stage_check_join_scripts_ready`, `stage_support_bundle_scripts_ready`, `stage_package_archives_ready`, `stage_archive_runner_scripts_ready`, `stage_setup_start_runner_ready`, `stage_handoff_checksums_ready`, `stage_join_scripts_use_invite_code_file`, `operator_env_local_only`, `coordinator_env_excludes_operator_credentials`, `registries_store_hashed_tokens`, `scripts_created`, `scripts_embed_plaintext_tokens: false`, `operator_plaintext_token_public: false`, and `miner_plaintext_tokens_public: false`. It is a private setup helper and offline/live no-claim package gate for controlled local/LAN/VPN/tunnel swarms, not proof that generation completed, not production NAT traversal, not billing/payment execution, and not large-model serving.
- Cloudflare quick route prep: `crowdtensor swarm-bootstrap --tunnel-provider cloudflare-quick --expect-remote-miners` emits a route-prep `crowdtensor_swarm_bootstrap_v1` report with `bootstrap_stage: route-prep`, `route_prep_ready`, `final_bootstrap_ready: false`, `cloudflare_quick_tunnel_route_prep_ready`, `miner_invites_deferred_until_tunnel_url_known`, `discover_cloudflare_tunnel.sh`, `create_bootstrap_from_tunnel.sh`, and `SWARM_ROUTE_PREP.md`. It starts a temporary `cloudflared tunnel --url http://127.0.0.1:<port>`, extracts the `trycloudflare.com` URL from logs, and then creates the final real bootstrap package with that URL. It intentionally does not create registries, operator env files, Miner invites, stage packages, or final handoff artifacts until the temporary URL is known, and the final package does not restart the quick tunnel because that can produce a different random URL. Preserve `route_prep_scripts_created`, `miner_invites_deferred_until_tunnel_url_known`, `cloudflare_quick_ephemeral_url`, and the boundary that this is a temporary route helper, not stable production tunnel management, NAT traversal, third-party account provisioning, billing/payment execution, or large-model serving.
- Product operator invites: `crowdtensor operator-invite` and `scripts/create_operator_invite.py` can create role-scoped operator entries for `--operator-token-registry`. The product CLI writes a hashed registry verifier plus a private `crowdtensor_operator_invite_v1` file, reports `crowdtensor_operator_invite_cli_v1`, and keeps plaintext operator tokens and invite codes out of public-safe output. Optional `session_policy` limits cover `/admin/inference-sessions` workload allowlists, request/decode/token caps, active session count, cumulative session count, and create rate windows. `scripts/create_operator_invite.py --json` may print the plaintext private invite for controlled handoff and must not be treated as public evidence. This advances multi-user role separation, abuse limits, and chargeback attribution, but it is not production account management, billing, staking, permissionless admission, or decentralized trust.
- Product Miner join invites: `scripts/create_miner_invite.py --invite-file` can create a private `crowdtensor_miner_join_invite_v1` JSON file plus base64url invite code for the ordinary `crowdtensor join` path. `crowdtensor join --invite-file/--invite` applies the Coordinator URL, Miner ID, stage, backend, Hugging Face model id, Miner token, and policy limits, then redacts the plaintext token from reports and terminal summaries. `crowdtensor join --check-coordinator` verifies `/ready` reachability and invite-vs-redacted-registry policy metadata before sending a Miner token; `--check-admission` then calls token-backed `POST /tasks/preflight` to verify Miner auth plus claim-time join policy, quota, rate, stage, backend, and model constraints without leasing a task. The Coordinator registry keeps only token verifiers plus `join_policy`; `/ready` exposes safe `crowdtensor_miner_registry_policy_summary_v1` metadata with stage/backend/trust/quota/claim-rate/reward-account-presence and no plaintext token or reward account. Claim-time enforcement restricts registered Miners to the invited workload, real-LLM stage, backend, model, positive claim-count quota, and positive claim-rate windows before leasing work, and records policy rejects in `blocked_claims`; preflight reports do not record blocked-claim events or claim tasks. `GET /admin/accounting` emits `miner_accounting_summary_v1` with safe Miner/workload rows, grouped totals, work units, `created_by_subject` plus `created_by_subject_totals` for admin-created inference session chargeback attribution, exact `created_by_subject` filtering for subject-level usage exports, and redacted join-policy metadata for trust tier, quota, claim-rate limits, reward-account presence, and read-only workload. `GET /admin/settlement` emits `miner_settlement_draft_v1` with accepted-only settlement rows, reward units, optional microcredit estimates, `created_by_subject`, grouped totals, `created_by_subject_totals`, exact `created_by_subject` filtering, redacted policy metadata, and no reward account values or payment execution. `crowdtensor operator-status` emits `crowdtensor_operator_status_cli_v1`, reads `/ready` plus `/state`, optionally reads `/admin/accounting` and `/admin/settlement` with `--include-admin-summaries`, writes `operator_status.json` / `operator_status.md`, summarizes operator registry roles/session policies, Miner registry stage/backend/trust/reward-account-presence metadata, trust/quarantine/blocked-claim counters, accounting/settlement readiness, and next actions, and redacts admin credentials, observer credentials, prompts, outputs, reward account values, and claim private material. `crowdtensor settlement` wraps accounting endpoints for accounting operators, emits `crowdtensor_settlement_cli_v1`, writes `settlement_summary.json` / `settlement_summary.md`, supports the same exact filters plus `--include-accounting`, and redacts admin credentials, reward account values, prompts, outputs, and lease material from saved artifacts. `crowdtensor trust` emits `crowdtensor_trust_cli_v1`, reads `/state` into safe `trust_summary.json` / `trust_summary.md` artifacts with automatic quarantine counts, manual allow/block overrides, effective blocked Miner/workload pairs, `blocked_claims`, and redacted row summaries, and can call `/admin/trust-overrides` with `--mode block|allow|reset` for owner/admin operators while keeping admin credentials, observer credentials, raw override reason text, prompts, outputs, and claim private material out of saved artifacts. The Coordinator still must be reachable from the Miner by public HTTPS, VPN, trusted private network, or tunnel. Reward fields are Beta accounting metadata, operator-status is read-only, and trust overrides are manual operator controls, not production billing, staking, automatic incentives, Sybil resistance, slashing, permissionless admission, NAT traversal, or large-model serving.
- Public Swarm Inference v2 output scope: top-level `public_swarm_inference_v2` JSON, Markdown, terminal summaries, and `support_bundle.json` preserve `output_request`, `answer_scope.scope_state: no-local-answer`, and `shareable_summary.answer_scope_state`. The v2 aggregate is shareable readiness evidence, not a local answer transcript; run `crowdtensor generate --p2p` in human mode to see local generated text while public artifacts keep raw prompts, generated text, generated token ids, activations, credentials, leases, and idempotency material out of reports.
- Public Swarm Inference v2: `crowdtensor public-swarm-v2` emits `public_swarm_inference_v2` through `scripts/public_swarm_inference_v2_pack.py` and validates with `scripts/public_swarm_inference_v2_check.py`. This is the current strongest public-preview gate over the ordinary path: `crowdtensor p2pd --run`, `crowdtensor serve --p2p --run`, distinct `crowdtensor join --stage stage0 --p2p --run` and `crowdtensor join --stage stage1 --p2p --run`, then `crowdtensor generate --p2p --prompt ... --max-new-tokens 16`, the bounded `--prompt-texts` batch path, or `--stream-generation` evidence. Ready reports preserve `public_swarm_inference_v2_ready`, `public_swarm_v2_local_p2p_generate_ready`, `public_swarm_v2_16_token_generation_ready`, `public_swarm_v2_dual_stage_kv_cache_ready`, `public_swarm_v2_external_stage_rows_ready`, `public_swarm_v2_signed_or_real_p2p_ready`, `public_swarm_v2_model_match_ready`, `public_swarm_v2_stage_requeue_rescue_ready`, `public_swarm_v2_batch_generation_ready` when bounded batch evidence is present, `public_swarm_v2_stream_generation_ready`, `public_swarm_generate_stream_ready`, and `public_swarm_generate_stream_endpoint_ready` when stream evidence is requested, `stage_latency_ready`, `throughput_summary_ready`, `memory_or_vram_summary_ready`, `public_swarm_v2_cuda_optional_fail_closed_ready`, redacted `public_swarm_inference_v2.md`, `PUBLIC_SWARM_INFERENCE_V2.md`, and `support_bundle.json`. A v2 local proof should show route source `p2p-discovery`, 16 generated tokens, 32 accepted rows, distinct stage0/stage1 Miners, stage rescue evidence, and `p2p_real_generate_dual_stage_kv_cache_v1` with cache-ready rows and hits for both stages; batch proofs must preserve prompt hashes/counts, request counts, per-request generated token counts, and `raw_prompts_public: false`; stream-enabled proofs must preserve public-safe progress milestones with `observed_token_counts` from 1 through `max_new_tokens`, `monotonic_progress: true`, endpoint readiness, and `raw_generated_text_public: false`. External validation imports real/signed P2P evidence and must meet both the requested v2 token target and the corresponding external accepted stage rows (`2 * max_new_tokens`) before `public_swarm_v2_fresh_external_runtime_verified` or `public_swarm_v2_external_evidence_ready` are emitted; stale nested `public_swarm_inference_v2_ready` codes from imported reports are filtered so blocked reports cannot inherit a false ready code. All v2 imports, including the default `sshleifer/tiny-gpt2` path, must expose explicit matching local/external/P2P `hf_model_id` metadata; otherwise v2 emits `public_swarm_v2_local_model_mismatch`, `public_swarm_v2_external_model_mismatch`, or `public_swarm_v2_p2p_model_mismatch` and blocks readiness. Current retained evidence is `dist/public-swarm-inference-v2/public_swarm_inference_v2.json` with `ok: true`, local two-prompt bounded batch generation with 16 generated tokens per request, safe prompt hashes and prompt char counts, local stream-enabled 16-token generation with 16 ordered safe stream events from `admin-session-stream`, 32 accepted local rows, 15 stage0 and 15 stage1 KV-cache hits, retained 16-token real-P2P external evidence with 32 accepted external rows, explicit matching model IDs, and no `not_completed` items. Use `--fresh-external-report` only when the referenced external report was produced in the current run and carries accepted stage rows plus matching model metadata; otherwise keep it as retained evidence with `public_swarm_v2_external_fresh_run_action_required`. Human `crowdtensor generate` may show generated text locally; public artifacts must redact raw prompts, generated text, generated token ids, activations, credentials, leases, and idempotency material. This remains Coordinator-backed, read-only, tiny/small-model scoped, CPU by default with optional CUDA evidence, not full Hivemind/Petals production parity, not Coordinator-free execution, not production NAT traversal, not an economic network, and not large-model throughput serving.
- Public Swarm Inference v2 local model variant: `crowdtensor public-swarm-v2 local-model-variant --hf-model-id <small-hf-model>` proves a non-default local small Hugging Face model through local v2 P2P generation, v2 child Usable/KV-cache evidence, and v2 local real-P2P route-hardening/requeue without claiming retained external/Kaggle validation. Ready reports preserve `public_swarm_inference_v2_local_model_variant_ready`, `public_swarm_v2_local_model_variant_ready`, `public_swarm_v2_local_model_variant_model_match_ready`, and `public_swarm_v2_external_validation_not_claimed`; they must not emit `public_swarm_inference_v2_ready`, `public_swarm_v2_external_stage_rows_ready`, `external_runtime_verified`, or retained GPU success codes for this mode. The CI-safe check entry is `python scripts/public_swarm_inference_v2_check.py --mode local-model-variant --hf-model-id distilgpt2 --json`.
- Product `generate --stream`: `crowdtensor generate --stream` now exposes a product-facing progress path for the Coordinator-backed real LLM split route. The Coordinator provides `GET /admin/session-stream` with `admin_session_stream_v1`, built from accepted stage-1 `real_llm_sharded_infer` rows through `safe_stream_event`, so streaming progress can be polled without exposing the full admin result ledger. While waiting for the final result, CLI output emits `session_stream_event_v1` summaries with session/task/miner/stage, `generated_token_count`, `generation_step`, `max_new_tokens`, `generated_text_hash`, optional `observed_at`, and a progress summary with monotonic/all-token completion flags; `public_swarm_generate_stream_ready` is emitted only when progress includes every requested token count, and `public_swarm_generate_stream_endpoint_ready` records the safe endpoint path. `scripts/product_swarm_mvp_check.py --stream-generation`, `scripts/p2p_swarm_inference_v06_pack.py --stream-generation`, top-level `crowdtensor p2p-swarm-v06 --stream-generation`, `crowdtensor usable-swarm --stream-generation`, and `crowdtensor public-swarm-v2 --stream-generation` now propagate this safe stream proof and block when requested stream progress is incomplete. Preserve `product_swarm_mvp_stream_ready`, `p2p_real_generate_stream_ready`, `usable_real_llm_stream_ready`, and `public_swarm_v2_stream_generation_ready` in ready reports. The CLI falls back to `/admin/results` only for older Coordinators. JSON/public output must still omit raw generated text, generated token ids, activations, credentials, leases, and idempotency material. Human mode can print progress lines as each token-count milestone appears. This is an interaction/observability improvement for the existing tiny/small-model sharded generation path; it is not direct Miner-to-client token streaming, not KV-cache reuse by itself, not large-model serving, and not production P2P.
- Product `generate --prompt-texts`: `crowdtensor generate` now supports a bounded batch of up to four prompts in one real-LLM split session through `--prompt-texts`, while preserving the single-prompt `--prompt` / `--prompt-text` path. Public `session_protocol_v1` requests carry only prompt hashes, prompt char counts, request count, and batch metadata; the private Coordinator payload carries the raw prompt list for task execution. Public reports use `safe_generation_summary` to emit per-request generated counts and generated text hashes, `batch_generation_ready`, and `public_swarm_generate_batch_ready` only when every requested prompt completes. `scripts/product_swarm_mvp_check.py` also accepts `--prompt-texts` and emits safe batch evidence with `product_swarm_mvp_batch_ready`; a live local tiny-GPT MVP smoke verified two prompts with two generated tokens each, four accepted stage rows, and safe public output. `scripts/p2p_swarm_inference_v06_pack.py` forwards the same bounded batch into the persistent P2P real-generate probe and emits `p2p_real_generate_batch_ready` plus `public_swarm_generate_batch_ready` in safe public evidence. Top-level `crowdtensor usable-swarm --prompt-texts` and `crowdtensor public-swarm-v2 --prompt-texts` validate the same bounded list, forward it into their pack scripts, preserve imported batch summaries, and surface `usable_real_llm_batch_ready` / `public_swarm_v2_batch_generation_ready` when present. This is bounded operator/user batch generation for the existing Coordinator-backed tiny/small-model path; it is not arbitrary public prompt serving, not a throughput batching scheduler, not speculative decoding, not direct Miner-to-client token streaming, and not large-model serving.
- Product `generate --hf-model-id`: `crowdtensor generate` now accepts `--hf-model-id` and carries it through `session_protocol_v1`, the private Coordinator `/admin/inference-sessions` payload, and the returned safe session summary. The Coordinator honors per-session model ids for `real_llm_sharded_infer` by inspecting the requested small Hugging Face artifact instead of always using the process default. `scripts/p2p_swarm_inference_v06_pack.py external-existing --verify-generate` and `scripts/real_p2p_swarm_inference_core_rc_pack.py external-existing --verify-generate` forward `--hf-model-id`, bounded `--prompt-texts`, and `--stream-generation` into live external `generate` verification, then record safe batch/stream summaries plus observed model metadata for later gates. Real P2P local and Kaggle generate commands also pass the requested `--hf-model-id` into the nested generation process instead of only reporting top-level metadata. Public artifacts still redact raw prompts, generated text, generated token ids, activations, leases, and credentials. This is model-selection plumbing for tiny/small HF split inference, not large-model serving or arbitrary model hosting.
- Product inference evidence scope: `crowdtensor infer` and `crowdtensor generate` reports include top-level `evidence_scope` plus terminal/Markdown `evidence_scope_note` so ordinary users can tell what the current command actually ran without interpreting nested evidence. They also print/save `gpu_status` as the direct verdict: `local-cpu-only`, `local-gpu-smoke-only`, `retained-gpu-evidence`, `fresh-kaggle-gpu-attempted-unverified`, `fresh-kaggle-gpu-verified`, or `no-gpu-evidence`, and save `gpu_proof_next_step` with explicit optional CUDA smoke, Kaggle package, and side-effectful fresh Kaggle GPU proof commands. Preserve `local-cpu-loopback`, `local-full-evidence`, `existing-runtime-preflight`, `existing-runtime-submit`, `p2p-runtime-preflight`, `p2p-runtime-submit`, `retained_gpu_evidence_imported`, `fresh_kaggle_gpu_attempted`, `fresh_kaggle_gpu_verified`, `gpu_proof_next_step.requires_kaggle`, `cleanup_required`, and `token_rotation_required` semantics. The default quick-start `infer` path is local CPU / local loopback; `generate --dry-run` is preflight unless it submits; retained GPU imports are historical evidence, `fresh_kaggle_gpu_attempted: true` without verification is only an attempted GPU path, and only `fresh_kaggle_gpu_verified: true` supports a fresh Kaggle GPU claim for that report.
- Product fallback guidance: `crowdtensor infer --mode existing`, `crowdtensor generate`, and `crowdtensor join` keep token and peer-secret values out of safe reports by surfacing them as `requires_env` / `# requires CROWDTENSOR_...` next-command hints. When P2P discovery is unreachable, P2P-lite paths should recommend `crowdtensor p2pd`, while `--p2p-backend real` paths should recommend `crowdtensor p2p-daemon`; preserve this backend-aware behavior for `infer`, `generate`, and `join`.
- Real LLM token continuation: multi-token `real_llm_sharded_infer` now keeps the original prompt stable across generation steps and carries the generated prefix as token ids into stage 0. Stage 0 appends `generated_token_ids` after tokenizing the original prompt, records `prompt_token_count`, `generated_prefix_token_count`, `input_token_count`, and `token_continuation_ready`, and no longer depends on re-tokenizing `prompt + next_token_text` or character truncation for continuation. This is a correctness/runtime stepping improvement that moves toward real autoregressive serving, but it is still not speculative decoding, batched streaming, large-model sharding, or production throughput optimization.
- Real LLM session-stage affinity: the Coordinator now records `real_llm_stage_affinity_v1` / `session_stage_sticky_v1` for `real_llm_sharded_infer` tasks. Within a generation session, stage 0 stays bound to the first stage0 Miner and stage 1 stays bound to the first stage1 Miner across subsequent token steps; this preserves the stage-local runtime continuity needed for KV-cache reuse. Lease-timeout requeue remains allowed to rescue and rebind a stage when the original Miner disappears. Local product evidence `dist/goal-final-infer-product-mvp-stage-affinity-20260601/product_swarm_mvp_check.json` shows 3 generated tokens, 6 accepted stage rows, distinct stage0/stage1 Miners, and internal state confirms all stage0 steps used `product-mvp-stage0` while all stage1 steps used `product-mvp-stage1`. This is still Coordinator-backed and not production failover; it is the scheduling invariant required before broader cache and streaming layers.
- Real LLM dual-stage KV cache prototype: stage0 now maintains an in-process `real_llm_stage0_kv_cache_v1` prefix cache and stage1 maintains an in-process `real_llm_stage1_kv_cache_v1` suffix cache, both keyed by session, request, artifact, split index, and Miner id. For persistent stage Miners, the first token runs the full stage forward and later token steps reuse GPT-2 `DynamicCache` past state plus cached hidden prefixes/suffixes, while preserving the existing activation/result shapes for validation. Cache misses safely fall back to the full stage forward path, including Miner changes and rescue/requeue cases. Unit coverage verifies stage0 and stage1 cache-hit correctness against uncached outputs plus cache miss on Miner change; live local evidence `dist/goal-final-infer-persistent-dual-kv-cache-20260601/persistent_real_llm_kv_cache_check.json` shows 3-token generation with long-lived stage0/stage1 Miners, 3 cache-ready rows per stage, and 2 cache hits per stage. `dist/goal-final-infer-product-mvp-kv-cache-20260601/product_swarm_mvp_check.json` verifies compatibility with the existing one-task-per-Miner Product MVP, where cache readiness is true but hits are not expected because each step starts a fresh Miner process. This is in-process-only, tiny GPT/HF-only, not cross-process cache persistence, not a production KV memory manager, not large-model serving, and not true low-latency streaming.
- P2P/product KV cache evidence: `scripts/p2p_swarm_inference_v06_pack.py` now extracts a public-safe `p2p_real_generate_dual_stage_kv_cache_v1` summary from the private real-generate state log and reports `p2p_real_generate_kv_cache_ready`, `real_llm_stage0_kv_cache_v1_ready`, `real_llm_stage1_kv_cache_v1_ready`, `stage0_kv_cache_hits_ready`, and `stage1_kv_cache_hits_ready` when the persistent P2P stage Miners reuse both caches. Its real-generate probe also accepts bounded `--prompt-texts` batches and preserves safe batch summaries with `p2p_real_generate_batch_ready` when every request completes; `--stream-generation` adds safe stream summaries and `p2p_real_generate_stream_ready` only when progress covers every requested token count. `usable_swarm_inference_v1` requires KV cache as `usable_real_llm_kv_cache_ready`, and `public_swarm_inference_v2` requires it as `public_swarm_v2_dual_stage_kv_cache_ready`. Public P2P summaries deliberately use public-safe field names and omit raw prompts, activations, token ids, raw generated text, output hashes, leases, and credentials. A completed local 16-token v0.6 proof exists at `dist/goal-final-infer-p2p-v06-16tok-kv-cache-20260601/p2p_swarm_inference_v06.json` with `ok: true`, `p2p_swarm_inference_v06_ready`, `p2p_real_generate_ready`, `p2p_real_generate_kv_cache_ready`, 16 generated tokens, 32 accepted rows, both stage KV-cache schema ready codes, 16 cache-ready rows per stage, 15 stage0 cache hits, 15 stage1 cache hits, `p2p_real_stage_rescue_ready`, `stage0_rescue_generation_completed`, and `stage1_rescue_generation_completed`; no related local services were left running after the proof. The earlier completed two-token proof remains at `dist/goal-final-infer-p2p-v06-ready-kv-cache-20260601/p2p_swarm_inference_v06.json`; the even earlier attempt at `dist/goal-final-infer-p2p-v06-kv-cache-20260601/p2p-real-generate/real-generate-state/tasks.jsonl` showed cache hits but was manually cleaned up after wrapper teardown hung, so do not count that earlier directory as a completed v0.6 gate.
- Real-P2P/libp2p local recovery note: a 2026-06-01 regression in `scripts/libp2p_kad_daemon.mjs` made `/real-p2p/health` and `/real-p2p/providers` return 422 because `healthScore()` was missing. The fix adds the daemon-side health score helper plus `tests/test_libp2p_kad_daemon.py`, and fresh local evidence now exists at `dist/goal-final-infer-libp2p-discovery-check-20260601/libp2p_discovery_alpha_check.json` and `dist/goal-final-infer-local-real-p2p-fixed-20260601/real_p2p_swarm_inference_core_rc.json`. The latter proves `libp2p-kad`, signed provider records, `real-p2p-discovery` route lookup, distinct stage0/stage1 Miners, 2-token tiny-GPT split generation, 4 accepted rows, stage latency, throughput, and memory summaries. This repairs local real-P2P routing but does not by itself prove fresh external Kaggle/P2P readiness or production NAT traversal.
- Real-P2P local stage requeue: `scripts/real_p2p_swarm_inference_core_rc_pack.py local-smoke --failure-mode kill-stage1-after-claim` now exercises a live local victim/rescue path. The retained proof is `dist/goal-final-infer-real-p2p-local-stage1-requeue-ready-20260602/real_p2p_swarm_inference_core_rc.json`; it has `ok: true`, observes the victim claim through public state, terminates the victim process, waits for lease timeout requeue, starts a rescue Miner, accepts the rescue result at attempt 2, and records `victim_result_accepted: false`. Preserve `real_p2p_local_stage_requeue_ready`, `local_stage_requeue_ready`, `stage_requeue_ready`, `live_stage1_requeue_ready`, `accepted_result_after_requeue`, `rescue_miner_used`, and `local_requeue_victim_process_terminated`. This is local route-hardening evidence, not external/Kaggle readiness, not production NAT traversal, and not Coordinator-free execution.
- Fresh external real-P2P recovery proof: after the same 2026-06-01 `healthScore()` fix, `scripts/real_p2p_swarm_inference_core_rc_pack.py kaggle-auto` completed against public host `24.199.118.54` on temporary ports `9950/9951/10950` with private Kaggle CPU kernels `xuyuhaosuyi/ct-fi-0601-stage0` and `xuyuhaosuyi/ct-fi-0601-stage1`. Retained two-token evidence is `dist/goal-final-infer-fresh-real-p2p-kaggle-20260601/real_p2p_swarm_inference_core_rc.json` plus the import at `dist/goal-final-infer-real-p2p-core-fresh-import-20260601/real_p2p_swarm_inference_core_rc.json`. Current 16-token external evidence is `dist/goal-final-infer-fresh-real-p2p-kaggle-16tok-20260601/real_p2p_swarm_inference_core_rc.json` plus the strict import at `dist/goal-final-infer-real-p2p-core-fresh-16tok-import-strict-20260601/real_p2p_swarm_inference_core_rc.json`; it preserves the imported `external.external_runtime_verified` and `external.external_generate_verified` fields. It reports `ok: true`, `external_libp2p_stage_discovery_ready`, `external_libp2p_generate_ready`, `hivemind_petals_class_alpha_ready`, 16 generated tokens, 32 accepted rows, distinct Kaggle stage Miners, deleted Kaggle kernels, cleaned local private kernel payloads, and `token_rotation_required`. This is fresh external tiny-GPT real-P2P/libp2p proof for the Public v2 token target, but still not production NAT traversal, not Coordinator-free execution, and not large-model serving.
- Usable Swarm Inference v1: `crowdtensor usable-swarm` emits `usable_swarm_inference_v1` through `scripts/usable_swarm_inference_pack.py` and validates with `scripts/usable_swarm_inference_check.py`. This is now the ordinary user entrypoint: `crowdtensor p2pd --run`, `crowdtensor serve --p2p --run`, distinct `crowdtensor join --stage stage0 --p2p --run` and `crowdtensor join --stage stage1 --p2p --run`, then `crowdtensor generate --p2p --prompt ... --max-new-tokens 8`, optional bounded `--prompt-texts` batches, optional `--stream-generation`, or the stricter 16-token maintainer gate. Ready reports preserve `usable_swarm_inference_ready`, `usable_swarm_inference_v1_ready`, `serve_join_generate_p2p_primary_path`, `usable_p2p_route_ready`, `usable_real_llm_generate_ready`, `usable_real_llm_kv_cache_ready`, `usable_real_llm_batch_ready` when bounded batch evidence is present, `usable_real_llm_stream_ready` when requested stream progress is complete, `usable_multi_token_generation_ready`, `usable_distinct_stage_miners_ready`, `usable_stage_requeue_rescue_ready`, `usable_swarm_model_match_ready`, top-level `output_request`, `answer_scope.scope_state: no-local-answer`, `shareable_summary.answer_scope_state`, redacted `support_bundle.json`, and `USABLE_SWARM_INFERENCE.md`. Non-default `--hf-model-id` evidence imports must expose a matching `hf_model_id` in the P2P v0.6 report or emit `usable_swarm_model_mismatch` and block readiness; the CI-safe non-default model check is `python scripts/usable_swarm_inference_check.py --mode local --hf-model-id distilgpt2 --json`. Current strict evidence is `dist/goal-final-infer-usable-swarm-16tok-kv-cache-20260601/usable_swarm_inference.json` with 16 generated tokens, 32 accepted rows, 15 KV-cache hits per stage, and no `not_completed` items. Human `crowdtensor generate` output may show generated text when not using `--json`; the shareable Usable Swarm aggregate now explicitly records that it has no local answer transcript and keeps raw prompts, generated text, generated token ids, activations, credentials, leases, and idempotency material out of public JSON/Markdown/support artifacts. It is Coordinator-backed, read-only, tiny/small-model scoped, CPU by default with optional CUDA fail-closed paths, not full Hivemind/Petals production parity, not Coordinator-free execution, not production NAT traversal, not an economic network, and not large-model throughput serving.
- FastAPI Coordinator with task queues, task lanes, leases, heartbeat deadlines, checkpoint state, append-only event replay, result validation, replay audit, metrics, admin result ledger, and trust overrides.
- Python Miner CLI with capability advertisement, CPU `hardware_profile`, `/ready` preflight, bounded retry behavior, result `idempotency_key`, heartbeats, and bounded session controls.
- Deterministic CPU-only workload contracts: `diloco_train`, `cpu_lora_mock`, `micro_transformer_lm`, `model_bundle_lm`, `model_bundle_infer`, optional `external_llm_infer`, and `browser_probe`.
- Protocol boundary around `runtime_contract_v1`, `outer_optimizer_contract_v1`, supported workloads, supported delta formats, and workload-specific validation.
- Delta transport paths for `dense_float`, `sign_compressed`, and `sign_compressed_ef`.
- Admission and operator safety: shared Miner token, per-Miner token registry, observer token, legacy owner-level admin token, role-scoped operator token registry (`owner`, `admin`, `accounting`, `auditor`) with optional safe `session_policy` limits for `/admin/inference-sessions` including active queued/leased session caps and cumulative session quotas, hashed token verifiers, security preflight, redacted `/state`, aggregate `/metrics`, and safe admin ledger/accounting/settlement views.
- Controlled remote Miner onboarding through invite generation, readiness checks, remote join checks, retry counters, and Support Bundle diagnostics.
- Browser experiments for WebRTC tensor tunnel, browser Worker compute probe, and browser Miner bridge.
- Release and support tooling: First-run Doctor, runtime capability matrix, matrix-guided home-compute demo, user-facing inference session demo, `inference_session_client_v1`, admin-created read-only inference session API check, release gate, fresh clone onboarding gate, release readiness gate, runtime acceptance pack, browser acceptance pack, release evidence pack, Support Bundle, changelog, release process docs, roadmap, protocol docs, use-case docs, and static site. Runtime acceptance emits safe per-check `summary_json` and top-level `diagnosis_summary`; release evidence and Support Bundle preserve `diagnosis_by_check` plus safe remote `observability_summaries` for operator triage.
- Release readiness gate: `crowdtensor release-ready` in `crowdtensor/cli.py` wraps `scripts/release_readiness_pack.py` and emits `release_readiness_v1` by aggregating Git metadata, the release gate, security preflight, and `demo_manifest_v1`. Dirty worktrees block by default with `git_dirty`; `scripts/release_readiness_check.py --allow-dirty` is the development/CI smoke path. It is not production Swarm Inference readiness.
- Fresh clone onboarding gate: `scripts/onboarding_gate.py --quick` emits `onboarding_gate_v1` by creating a clean temporary virtualenv, running `python -m pip install -e .[dev,hf]`, checking `crowdtensor --help`, `crowdtensord --help`, and `crowdtensor-miner --help`, then smoke-validating `scripts/user_friendly_inference_frontdoor_check.py`, the installed real user entrypoint `crowdtensor infer --prompt-stdin --shareable-terminal`, `crowdtensor local-proof`, `crowdtensor home-infer`, `crowdtensor llm-infer --mock`, `crowdtensor cpu-infer --mode local`, and `crowdtensor release-ready --allow-dirty`. The `user_infer_smoke` validation must preserve `user_infer_smoke_validation_v1`, `answer=shareable-terminal-redacted`, `gpu=local-cpu-only`, `fresh_kaggle_gpu=False`, prompt stdin redaction, safe `infer_summary.json` / `infer_summary.md`, and no raw prompt/generated answer/token ids/activations in onboarding artifacts. It is a fresh-checkout onboarding gate, not production Swarm Inference readiness.
- One-command local proof: `crowdtensor local-proof` in `crowdtensor/cli.py` emits `local_proof_summary_v1` by chaining Doctor, runtime matrix, the CPU-only read-only home-compute demo, and the Demo Manifest path. It is not production Swarm Inference; it is a user-facing local proof artifact.
- Home inference proof CLI: `crowdtensor home-infer` emits `home_inference_cli_v1`, wraps `scripts/home_compute_evidence_pack.py`, and writes `home_compute_evidence_v1` JSON/Markdown with the CPU-only `model_bundle_infer` route, fixed `model_bundle_inference_scenario_v1` metadata, capped `request_trace`, `diagnosis_codes`, read-only status, and redaction status. Built-in scenario IDs are `route-baseline`, `gradient-safety`, and `mixed-prompts`; it is not production Swarm Inference or arbitrary prompt serving.
- External LLM proof CLI: `crowdtensor llm-infer` emits `llm_inference_cli_v1`, wraps `scripts/external_llm_evidence_pack.py`, and writes `external_llm_evidence_v1` JSON/Markdown with the read-only `external_llm_infer` route, adapter kind, model id, request/completion count, output chars, throughput, diagnosis codes, read-only status, and redaction status. The default path is deterministic mock; command and OpenAI-compatible HTTP runtimes are explicit operator-owned adapters. This is not public arbitrary prompt serving.
- CPU inference Beta aggregate CLI: `crowdtensor cpu-infer` emits `cpu_inference_beta_v1` through `scripts/cpu_inference_beta_pack.py`. `--mode local` wraps `home-infer` and deterministic `llm-infer --mock`; `--mode remote-loopback` validates local `remote-demo` stand-ins for `model-bundle` and `external-llm`; `--mode remote-existing` wraps an already running controlled two-machine `remote-demo doctor/verify/collect` flow with explicit observer/admin tokens. `scripts/cpu_inference_beta_check.py` validates the path. It is CPU-only, read-only, not production Swarm Inference, not P2P, and not arbitrary prompt serving.
- CPU Inference Beta RC: `crowdtensor cpu-infer --mode beta-rc` emits `cpu_inference_beta_rc_v1` through `scripts/cpu_inference_beta_rc_pack.py`. It aggregates local CPU inference, remote-loopback inference, Real two-machine CPU inference Beta rehearsal, Kaggle Remote Miner Beta artifacts, `miner_join_pack_v1`, `scripts/kaggle_remote_miner_beta_check.py`, and `demo_manifest_v1`; `scripts/cpu_inference_beta_rc_check.py` validates the path with `cpu_inference_beta_rc_ready`, `local_cpu_inference_ready`, `remote_loopback_ready`, `two_machine_rehearsal_ready`, `kaggle_remote_miner_artifacts_ready`, `miner_join_pack_ready`, and `cpu_miner_beta_ready`. `--kaggle-real-runtime-report` can import a completed `kaggle_real_runtime_acceptance_v1` report and surface `real_runtime_evidence_ready`; it must not be treated as CI live Kaggle proof. It is CPU-only, read-only, not production Swarm Inference, not P2P, not a GPU/TPU workload path, and not arbitrary prompt serving.
- Pipeline-Sharded Inference Alpha/Beta: `crowdtensor shard-infer` emits `sharded_inference_cli_v1` and `sharded_inference_evidence_v1` for the CPU-only read-only `sharded_model_bundle_infer` / `sharded_model_bundle_infer_v1` workload inside `sharded_inference_session_v1`; `crowdtensor shard-infer-beta --mode remote-loopback` emits `remote_sharded_inference_beta_v1` through `scripts/remote_sharded_inference_beta_pack.py` and validates with `scripts/remote_sharded_inference_beta_check.py`; `crowdtensor remote-demo --workload sharded-model-bundle` emits the two-machine runbook/acceptance shape with `remote_python_sharded_model_bundle_infer`, `remote_sharded_inference_acceptance_v1`, `remote_sharded_inference_observability_v1`, and `remote_two_machine_sharded_ready`. The path preserves activation hashes, `baseline_match`, `remote_sharded_inference_ready`, `remote_sharded_loopback_ready`, `local_sharded_inference_ready`, and `stage_requeue_ready` when `--failure-mode kill-stage-after-claim` is used. It is CPU-only, read-only, not production Swarm Inference, not P2P, and not real LLM sharding.
- Micro-LLM Pipeline-Sharded Inference Alpha/Beta: `crowdtensor micro-llm-shard-infer` emits `micro_llm_sharded_cli_v1` and `micro_llm_sharded_evidence_v1` for the CPU-only read-only `micro_llm_sharded_infer` / `micro_llm_sharded_infer_v1` workload inside `micro_llm_sharded_session_v1`; `crowdtensor micro-llm-shard-infer-beta --mode remote-loopback` emits `remote_micro_llm_sharded_beta_v1` through `scripts/remote_micro_llm_sharded_beta_pack.py` and validates with `scripts/remote_micro_llm_sharded_beta_check.py`; `crowdtensor remote-demo --workload micro-llm-sharded` emits the two-machine runbook/acceptance shape with `remote_python_micro_llm_sharded_infer`, `remote_micro_llm_sharded_acceptance_v1`, `remote_micro_llm_sharded_observability_v1`, and `remote_two_machine_micro_llm_sharded_ready`. The path preserves activation hashes, `decode_steps`, `baseline_match`, `decoded_tokens_match`, `remote_micro_llm_sharded_ready`, `remote_micro_llm_sharded_loopback_ready`, `local_micro_llm_sharded_inference_ready`, and `stage_requeue_ready` when `--failure-mode kill-stage-after-claim` is used. It is CPU-only, read-only, not production Swarm Inference, not P2P, and not GGUF/llama.cpp or large LLM serving.
- Micro-LLM Artifact Alpha: `crowdtensor micro-llm-artifact` and `scripts/micro_llm_artifact_pack.py` emit `micro_llm_artifact_v1`, a dependency-free tiny JSON model package with `manifest.json`, `config.json`, `tokenizer.json`, and `weights.json`. `crowdtensor micro-llm-shard-infer --micro-llm-artifact`, `scripts/micro_llm_artifact_check.py`, `crowdtensor micro-llm-live-rc --micro-llm-artifact`, `crowdtensor remote-demo --workload micro-llm-sharded --micro-llm-artifact`, and `crowdtensor remote-demo kaggle-real --workload micro-llm-sharded --micro-llm-artifact` preserve artifact id/hash/tokenizer metadata through session creation, stage validation, remote evidence, and local-generated RC artifacts. Ready evidence should include `artifact_loaded` and `micro_llm_artifact_ready`. This remains CPU-only/read-only toy micro-LLM proof; it is not Hugging Face, GGUF, llama.cpp, large-model sharding, or production Swarm Inference.
- Stage-Aware Micro-LLM Pipeline-Sharded Inference: `scripts/stage_aware_micro_llm_sharded_check.py`, `crowdtensor micro-llm-shard-infer --stage-mode split --require-distinct-stage-miners`, and `crowdtensor micro-llm-shard-infer-beta --mode remote-loopback --stage-mode split --require-distinct-stage-miners` prove explicit CPU-only stage routing. Miners advertise `micro_llm_sharded_stage0`, `micro_llm_sharded_stage1`, or `micro_llm_sharded_both` through `--micro-llm-stage-role`; evidence preserves `distinct_stage_miners`, `stage_assignment_valid`, and stage-specific `stage_requeue_ready` for `--failure-mode kill-stage0-after-claim` / `--failure-mode kill-stage1-after-claim`. This is still a controlled task-level proof, not production model sharding.
- Real Small-LLM Sharded Inference Beta: `crowdtensor real-llm-shard-infer` emits `real_llm_sharded_cli_v1` and `real_llm_sharded_evidence_v1` for the optional `[hf]` `real_llm_sharded_infer` / `real_llm_sharded_infer_v1` workload. The default model is `sshleifer/tiny-gpt2` running on `hf_transformers_cpu`; Coordinator session creation records safe `real_llm_artifact_v1` metadata and then schedules stage 0 and stage 1 tasks. Stage 0 tokenizes the fixed prompt and runs embeddings plus the first transformer blocks, while stage 1 consumes the activation, runs remaining blocks plus the lm head, and validates against a local full-model next-token baseline. Artifacts and workload specs include `execution_support` / `execution_family`; current executable support is `gpt2` only, `true_partial_weight_loading_ready` is false, and Llama/Qwen/Mistral/Gemma/Phi-style candidates fail closed with `real_llm_llama_like_stage_adapter_missing` before runtime load. Miners must opt in with `--enable-hf-tiny-gpt-runtime`, optional `--hf-cache-dir`, and `--real-llm-stage-role stage0|stage1|both`; capability routing uses `real_llm_sharded_stage0`, `real_llm_sharded_stage1`, and `real_llm_sharded_both`. `--real-llm-partition-mode stage-local` is the explicit module-placement proof: the Miner loads the model object on CPU, moves only stage-owned modules to the selected runtime device, keeps a separate CPU baseline for correctness, and emits `stage_local_partition_ready`, `stage0_partition_loaded`, `stage1_partition_loaded`, `partition_parameter_split_valid`, stage parameter counts, and `stage_gpu_memory_reduced` when CUDA is used. Evidence must preserve `real_llm_artifact_ready`, `activation_transport_ready`, `baseline_match`, `decoded_tokens_match`, `real_llm_sharded_ready`, `distinct_stage_miners`, and `stage_assignment_valid` while redacting prompts, hidden states, logits, and raw activation payloads from public summaries. `crowdtensor real-llm-shard-infer-beta --mode remote-loopback` emits `remote_real_llm_sharded_beta_v1` through `scripts/remote_real_llm_sharded_beta_pack.py`, and `scripts/remote_real_llm_sharded_beta_check.py` validates `remote_real_llm_sharded_ready`, `remote_real_llm_sharded_loopback_ready`, and `local_real_llm_sharded_inference_ready`. The high-level operator path is `crowdtensor remote-demo --workload real-llm-sharded`; it prepares stage0/stage1 join packs, verifies `remote_python_real_llm_sharded_infer`, emits `remote_real_llm_sharded_runbook_v1`, `remote_real_llm_sharded_acceptance_v1`, `remote_real_llm_sharded_observability_v1`, and `remote_real_llm_sharded_beta_v1`, and reports `remote_two_machine_real_llm_sharded_ready` when distinct stage Miners complete. Missing optional runtime dependencies must produce `hf_dependencies_missing` with the operator action `python -m pip install -e '.[hf]'`. This is read-only optional Hugging Face tiny-model evidence, not production Swarm Inference, not P2P, not GPU/TPU pooling, not GGUF/llama.cpp serving, and not large-model serving.
- Real Small-LLM Sharded Inference Live RC: `crowdtensor real-llm-live-rc` emits `real_llm_live_rc_v1` through `scripts/real_llm_live_rc_pack.py` and is checked by `scripts/real_llm_live_rc_check.py`. `--mode local-generated` creates `kaggle-upload-real-llm-stage0` and `kaggle-upload-real-llm-stage1`, starts a local Coordinator plus two independent HF-enabled stage Miner processes from those generated packages, and should report `local_generated_real_llm_stage_upload_standins_ready`, `remote_real_llm_sharded_ready`, `real_llm_artifact_ready`, `decoded_tokens_match`, `stage_assignment_valid`, and `real_llm_live_rc_ready` while keeping `external_runtime_verified` false. `--mode kaggle-generated` prepares those upload packages and operator runbook only. `--mode external-existing` verifies an already running public Coordinator plus two external stage Miners and only then may report `external_runtime_verified` and `kaggle_real_llm_sharded_ready`. `scripts/kaggle_real_llm_live_package.py` emits `kaggle_real_llm_live_package_v1` and packages those stage uploads into private Kaggle dataset/script-kernel folders; `--inline-kernel-payload` is a temporary private fallback when Kaggle input mounts are unreliable. The first live real-weight Kaggle split proof completed against `24.199.118.54:9184` with two private Kaggle CPU script kernels, `kaggle-real-llm-stage0`, `kaggle-real-llm-stage1`, and `sshleifer/tiny-gpt2`; retained evidence is `dist/real-llm-live-goal-external/real_llm_live_rc.json` with `ok: true`, `external_runtime_verified`, `kaggle_real_llm_stage0_seen`, `kaggle_real_llm_stage1_seen`, `kaggle_real_llm_sharded_ready`, `real_llm_artifact_ready`, baseline match, decoded-token match, distinct stage Miners, and valid stage assignment. Temporary Kaggle kernels/dataset were deleted after evidence collection. Generated launchers must preserve `--enable-hf-tiny-gpt-runtime`, `--real-llm-stage-role stage0|stage1`, and `launcher_syntax_valid` so packaged `kaggle_remote_miner.py` files fail closed before upload if the launcher is invalid. This is CPU-only, read-only tiny Hugging Face evidence, not production Swarm Inference, not P2P, not GPU/TPU pooling, not GGUF/llama.cpp serving, and not large-model serving.
- Real Internet Swarm Inference Alpha: `crowdtensor real-llm-internet-alpha` emits `real_llm_internet_alpha_v1` through `scripts/real_llm_internet_alpha_pack.py` and is checked by `scripts/real_llm_internet_alpha_check.py`. `--mode local-generated` wraps the Live RC and adds mandatory local stage0/stage1 timeout rescue checks, so ready evidence must include `real_llm_internet_alpha_ready`, `real_llm_stage_requeue_ready`, `stage_requeue_ready`, `real_llm_live_rc_ready`, `remote_real_llm_sharded_ready`, `real_llm_artifact_ready`, `activation_transport_ready`, `decoded_tokens_match`, `distinct_stage_miners`, and `stage_assignment_valid` while keeping `external_runtime_verified` false. `--mode package` prepares public Coordinator and stage upload artifacts only. `--mode external-existing` verifies an already running public Coordinator plus two external stage Miners and only then may report `external_runtime_verified`. The first external Alpha proof completed against `24.199.118.54:9187` with two private Kaggle CPU script kernels, `internet-real-llm-stage0` and `internet-real-llm-stage1`, and `sshleifer/tiny-gpt2`; retained evidence is `dist/real-llm-internet-alpha-external/real_llm_internet_alpha.json` with `ok: true`, `external_runtime_verified`, `real_llm_internet_alpha_ready`, `kaggle_real_llm_stage0_seen`, `kaggle_real_llm_stage1_seen`, `kaggle_real_llm_sharded_ready`, baseline/decoded-token match, distinct stage Miners, and valid stage assignment. Temporary Kaggle kernels were deleted after evidence collection and tokens must be rotated after temporary public HTTP proofs. Reports preserve `token_rotation_required`, redact raw prompts, hidden states, logits, activations, tokens, and lease material, and remain CPU-only/read-only. This is not production Swarm Inference, not P2P, not GPU/TPU pooling, not GGUF/llama.cpp serving, and not large-model serving.
- Real Internet Swarm Inference Beta: `crowdtensor real-llm-internet-beta` emits `real_llm_internet_beta_v1` through `scripts/real_llm_internet_beta_pack.py` and is checked by `scripts/real_llm_internet_beta_check.py`. `--mode kaggle-auto` generates the Alpha package, starts the temporary public Coordinator, pushes private Kaggle CPU script kernels by default or private Kaggle GPU kernels with `--real-llm-backend hf_transformers_cuda`, runs external-existing verification, deletes the temporary kernels, stops the Coordinator, and only then may report `real_llm_internet_beta_ready`. CUDA mode preserves CPU Coordinator metadata-only scheduling, `coordinator_cuda_runtime_required: false`, and Miner-side torch CUDA requirements. Kaggle CUDA kernels default to `torch==2.7.1+cu118`, `torchvision==0.22.1+cu118`, and `transformers==4.40.2` for older Kaggle GPU compatibility. With `--failure-mode kill-stage0-after-claim` or `kill-stage1-after-claim`, it creates distinct target-stage victim/rescue Kaggle Miners, observes the victim claim through `/state`, deletes the victim kernel, waits for lease timeout requeue, pushes the rescue kernel, and emits `external_stage_requeue_ready`, `live_stage0_requeue_ready` / `live_stage1_requeue_ready`, and `live_requeue_summary`. Ready evidence preserves `real_llm_internet_alpha_ready`, `external_runtime_verified`, both Kaggle stages seen, `decoded_tokens_match`, `distinct_stage_miners`, `stage_assignment_valid`, `kaggle_kernels_deleted`, `token_rotation_required`, CPU-default/read-only semantics, and explicit not production Swarm Inference / not P2P / not GPU pooling / not large-model boundaries. The check script is CI-safe and fake-runner based; it does not create Kaggle resources.
- Swarm Inference Beta: `crowdtensor swarm-infer-beta` emits `swarm_inference_beta_v1` through `scripts/swarm_inference_beta_pack.py` and is checked by `scripts/swarm_inference_beta_check.py`. It is the user-facing two-machine wrapper around `real_llm_sharded_infer`. `swarm-infer-beta live` is the side-effectful `kaggle-auto` public proof wrapper around `real_llm_internet_beta_v1`; it starts a temporary public Coordinator, pushes private Kaggle CPU stage kernels, verifies `external_runtime_verified`, optionally verifies external victim/rescue requeue with `--failure-mode`, deletes kernels, writes `support_bundle.json`, removes local live private artifacts and raw runtime state by default, and only then may report `swarm_inference_beta_live_ready`, `real_llm_internet_beta_ready`, `external_stage_requeue_ready` when requested, `live_requeue_summary`, `kaggle_kernels_deleted`, `swarm_inference_beta_live_private_artifacts_cleaned`, and `token_rotation_required`. `--keep-live-private-artifacts` is for debugging only. `swarm-infer-beta prepare` writes `operator.private.env`, stage0/stage1 `miner.private.env`, a hashed `miner_registry.json`, stage join packs, and `SWARM_INFERENCE_BETA.md`; `coordinator` and `miner --stage stage0|stage1` print or run the generated commands; `verify` wraps `remote_real_llm_sharded_beta_v1`; `collect` gathers redacted evidence/support; `clean` is dry-run by default. Ready evidence preserves `swarm_inference_beta_ready`, `two_machine_swarm_inference_ready`, `real_llm_split_route_ready`, `decoded_tokens_match`, `distinct_stage_miners`, `stage_assignment_valid`, and `external_beta_evidence_imported` only when retained `real_llm_internet_beta_v1` evidence is imported. It is CPU-only, read-only, not production Swarm Inference, not P2P, not GPU/TPU pooling, not GGUF/llama.cpp, and not large-model serving.
- Public Swarm Inference Alpha: `crowdtensor swarm-session` emits `public_swarm_inference_alpha_v1` through `scripts/public_swarm_inference_alpha_pack.py` and is checked by `scripts/public_swarm_inference_alpha_check.py`. `--mode live-kaggle` aggregates cleanup-backed `swarm-infer-beta live` evidence, true external victim/rescue requeue evidence (`external_stage_requeue_ready`, `live_stage0_requeue_ready` / `live_stage1_requeue_ready`, `live_requeue_summary`) when `--failure-mode` is enabled, and mandatory `local-generated` real LLM stage requeue evidence. It should report `public_swarm_inference_alpha_ready`, `public_swarm_session_ready`, `local_stage_requeue_ready`, `public_swarm_live_requeue_ready`, `public_swarm_live_kaggle_ready`, `stage_requeue_ready`, `external_runtime_verified`, `kaggle_kernels_deleted`, and `token_rotation_required`. Child debug artifacts are pruned by default so shareable output is the top-level public JSON/Markdown report; `--keep-child-artifacts` is local debugging only. It is CPU-only, read-only, not production Swarm Inference, not P2P, and not large-model serving.
- Public Swarm Inference Alpha RC: `crowdtensor public-swarm-alpha-rc` emits `public_swarm_inference_alpha_rc_v1` through `scripts/public_swarm_inference_alpha_rc_pack.py` and is checked by `scripts/public_swarm_inference_alpha_rc_check.py`. `evidence-import` audits retained public live reports and should emit `public_swarm_inference_alpha_rc_ready`, `public_swarm_alpha_rc_evidence_imported`, `stage0_live_requeue_evidence_ready`, `stage1_live_requeue_evidence_ready`, `public_swarm_live_requeue_evidence_ready`, `public_swarm_alpha_private_artifacts_absent`, and `public_swarm_live_requeue_summary_ready`; `local-smoke` is CI-safe and does not create Kaggle resources. The retained proof paths are `dist/public-swarm-inference-alpha-live-stage0-requeue-20260527165830/public_swarm_inference_alpha.json`, `dist/public-swarm-inference-alpha-live-stage1-requeue-20260527170600/public_swarm_inference_alpha.json`, and `dist/public-swarm-inference-alpha-live-requeue-summary.json`. It is CPU-only, read-only, not production Swarm Inference, not P2P, and not large-model serving.
- Public Swarm Live Preview RC: `crowdtensor live-preview` emits `public_swarm_live_preview_rc_v1` through `scripts/public_swarm_live_preview_rc_pack.py` and is checked by `scripts/public_swarm_live_preview_rc_check.py`. Modes are `live-preview local-smoke`, `live-preview package`, `live-preview live-kaggle`, and `live-preview evidence-import`. Preserve `public_swarm_live_preview_rc_ready`, `public_swarm_live_preview_local_smoke_ready`, `public_swarm_live_preview_package_ready`, `public_swarm_live_preview_live_kaggle_ready`, `public_swarm_live_preview_evidence_import_ready`, `external_stage_requeue_ready`, `kaggle_kernels_deleted`, `private_artifacts_cleaned`, `token_rotation_required`, and `gpu_generation_evidence_import_ready` when retained GPU evidence is present. Fresh retained stage0/stage1 RC proofs completed against `24.199.118.54:9196` and `24.199.118.54:9198` with evidence at `dist/public-swarm-live-preview-rc-live-stage0-20260529043801-rc/public_swarm_live_preview_rc.json` and `dist/public-swarm-live-preview-rc-live-stage1-20260529044328-rc/public_swarm_live_preview_rc.json`; they preserve `external_runtime_verified`, `external_stage_requeue_ready`, `live_stage0_requeue_ready` or `live_stage1_requeue_ready`, `kaggle_kernels_deleted`, `private_artifacts_cleaned`, and `token_rotation_required`. The default Kaggle slug prefix is `ct-live-preview` to fit victim/rescue suffixes under Kaggle's 45-character slug limit. `live-kaggle` is side-effectful and wraps the existing Public Swarm Alpha Kaggle proof; CI uses fake-runner checks only. It is CPU-only by default, read-only, Coordinator-backed, not production Swarm Inference, not libp2p, not DHT, not NAT traversal, and not large-model serving.
- Public Swarm v0.1 Operator Preview: `crowdtensor operator-preview` emits `public_swarm_operator_preview_v1` through `scripts/public_swarm_operator_preview_pack.py` and is checked by `scripts/public_swarm_operator_preview_check.py`. Modes are `operator-preview local-smoke`, `operator-preview package`, `operator-preview live-kaggle`, and `operator-preview evidence-import`. Preserve `public_swarm_operator_preview_ready`, `operator_preview_user_path_ready`, `operator_preview_local_smoke_ready`, `operator_preview_package_ready`, `operator_preview_live_kaggle_ready`, `operator_preview_evidence_import_ready`, `serve_join_generate_ready` or package-mode `miner_join_pack_ready` plus `private_artifacts_local_only`, `cpu_fallback_ready`, `live_preview_ready`, `support_bundle_ready`, `release_readiness_ready`, `gpu_generation_evidence_import_ready`, `developer_preview_degraded`, `operator_preview_cpu_fallback_user_path_ready`, `operator_preview_retained_evidence_ready`, and `external_runtime_blocked` when fresh external execution is unavailable and retained Live Preview RC evidence is imported. It is the top-level ordinary-user preview aggregate over Developer Preview, Live Preview RC, release readiness, support bundle, CPU fallback, and retained GPU generation evidence. It is CPU-only by default, read-only, Coordinator-backed, not production Swarm Inference, not libp2p, not DHT, not NAT traversal, and not large-model serving.
- Public Swarm v0.2 Usable Inference Trial: `crowdtensor swarm-trial` emits `public_swarm_trial_v1` / `public_swarm_trial_cli_v1` through `scripts/public_swarm_trial_pack.py` and is checked by `scripts/public_swarm_trial_check.py`. Modes are `swarm-trial local-loopback`, `swarm-trial package`, `swarm-trial live-kaggle`, and `swarm-trial evidence-import`. Preserve `public_swarm_trial_ready`, `serve_join_generate_trial_ready`, `stage0_join_ready`, `stage1_join_ready`, `generate_ready`, `generated_token_count_ready`, top-level `output_request`, `answer_scope.scope_state: no-local-answer`, `shareable_summary.answer_scope_state`, `support_bundle_ready`, `cpu_fallback_ready`, `private_artifacts_cleaned`, `operator_preview_import_ready`, `gpu_generation_evidence_import_ready`, `swarm_trial_degraded_cpu_fallback_ready`, `external_runtime_blocked`, and `token_rotation_required`. It aggregates Product Beta, Operator Preview, Support Bundle diagnostics, CPU fallback, and retained GPU generation evidence into an ordinary-user trial report. Treat it as shareable trial evidence, not a local answer transcript; run `crowdtensor generate` in human mode to see local generated text while public artifacts keep raw prompts, generated text, generated token ids, activations, credentials, leases, private env files, and runtime state redacted. It is CPU-only by default, read-only, Coordinator-backed, not production Swarm Inference, not libp2p, not DHT, not NAT traversal, not GPU marketplace, and not large-model serving.
- Public Swarm Inference Beta: `crowdtensor public-swarm-beta` emits `public_swarm_inference_beta_v1` through `scripts/public_swarm_inference_beta_pack.py` and is checked by `scripts/public_swarm_inference_beta_check.py`. `public-swarm-beta product-beta` is now the product-shaped aggregate over Product RC, `session_protocol_v1`, `p2p_lite_peer_v1`, retained GPU sharded generation evidence, and CPU fallback; it should emit `public_swarm_product_beta_ready`, `public_swarm_product_rc_ready`, `coordinator_product_surface_ready`, `session_protocol_ready`, `p2p_lite_discovery_ready`, `gpu_generation_evidence_import_ready`, `cpu_fallback_ready`, and `local_cpu_inference_ready`. `public-swarm-beta local-loopback` still wraps `remote_real_llm_sharded_beta_v1` in split mode and should emit `two_stage_split_inference_ready`, `local_loopback_ready`, `decoded_tokens_match`, `distinct_stage_miners`, and `stage_assignment_valid`. `public-swarm-beta evidence-import` still imports the retained Alpha RC report and should emit `public_swarm_beta_evidence_import_ready`, `external_live_evidence_imported`, `stage0_live_requeue_evidence_ready`, and `stage1_live_requeue_evidence_ready`. Preserve `prepare`, `coordinator`, `miner --stage stage0`, `miner --stage stage1`, `verify`, `collect`, and dry-run `clean`. It is Coordinator-backed, read-only, not production Swarm Inference, not libp2p/DHT/NAT traversal, not Hivemind-level serving, and not large-model serving.
- GPU Swarm Usability Alpha: `crowdtensor gpu-swarm` is the ordinary-user multi-GPU connection flow over the retained large-model core evidence. It emits `gpu_swarm_usability_alpha_v1` / `gpu_swarm_usability_alpha_cli_v1` through `scripts/gpu_swarm_usability_alpha_pack.py` and is checked by `scripts/gpu_swarm_usability_alpha_check.py`. Preserve subcommands `smoke`, `prepare`, `coordinator`, `miner --stage stage0|stage1`, `infer`, `status`, `collect`, and `clean`; the report fields `gpu_swarm_usability_alpha_ready`, `user_gpu_swarm_entrypoint_ready`, `gpu_miner_join_pack_ready`, `coordinator_workflow_ready`, `two_gpu_stage_route_ready`, `inference_request_lifecycle_ready`, `model_catalog_imported`, `control_user_alpha_imported`, `core_handoff_imported`, `public_artifact_safe`, `execution_mode`, and `external_runtime_verified`; stage join packs with `real_llm_sharded_cuda_stage0` and `real_llm_sharded_cuda_stage1`; safe `GPU_SWARM_MINER_PRIVATE_TOKEN` placeholders; `GPU_SWARM_ALPHA.md`; and redacted `support_bundle.json`. Default CI/dev proof is `evidence-import` and consumes Control/User Alpha plus retained 7B/14B core handoff evidence without claiming a fresh live GPU run (`external_runtime_verified=false`). Public artifacts must not include raw prompts, generated text, token ids, activations, credentials, leases, idempotency material, private env files, registries, or Kaggle kernel payloads. This is not production Swarm Inference, not P2P/NAT traversal, not arbitrary public prompt serving, not billing, and not unbounded GPU pooling.
- GPU Swarm Production-Like Validation RC: `crowdtensor gpu-swarm validate-production-like` and `crowdtensor gpu-swarm scale-test` are the bounded production-shaped validation entrypoints over the retained GPU Swarm evidence. They emit `gpu_swarm_production_like_validation_v1` / `gpu_swarm_production_like_validation_cli_v1` through `scripts/gpu_swarm_production_like_validation_pack.py` and are checked by `scripts/gpu_swarm_production_like_validation_check.py`. Preserve `gpu_swarm_production_like_validation_ready`, `production_like_workload_ready`, `larger_model_attempted`, `largest_successful_model_tier`, `largest_attempted_model_tier`, `larger_model_blocked_reason`, `multi_token_decode_ready`, `batch_or_multi_request_ready`, `two_gpu_stage_route_ready`, `distinct_stage_miners_ready`, `stage_requeue_or_failure_recovery_ready`, `gpu_runtime_readiness_checked`, `stage_owned_weight_loading_ready`, `latency_throughput_summary_ready`, `network_activation_transfer_summary_ready`, `public_artifact_safe`, `execution_mode`, `external_runtime_verified`, `fresh_gpu_run_performed`, `retained_evidence_imported`, redacted Markdown, and redacted `support_bundle.json`. Default CI/dev proof is `evidence-import`: it imports GPU Swarm Usability Alpha, Control/User Alpha, retained 7B/14B core status, retained 16-token GPU generation/requeue evidence, and retained 2-request batch/stream evidence, while keeping `fresh_gpu_run_performed=false` and `external_runtime_verified=false`. The retained largest successful tier is 14B; the larger-model ladder attempts a 32B-class preflight and records `candidate_requires_more_vram_than_retained_two_gpu_profile` when the two-GPU Kaggle-class retained profile is insufficient. Fresh GPU validation remains explicit and bounded to at most two model-tier attempts plus one requeue attempt, with a single-attempt timeout cap of 60 minutes. Public artifacts must not include raw prompts, generated text, token ids, activations, hidden states, logits, KV cache, credentials, leases, idempotency material, private env files, registries, Kaggle inline payloads, or runtime-private state. This is production-like validation and larger-model blocker evidence, not a fresh GPU run by default, not 32B/70B success, not production Swarm Inference, not P2P/NAT traversal, not arbitrary public prompt serving, not billing, and not unbounded GPU pooling.
- Kaggle Swarm 32B Quantized Feasibility RC: `crowdtensor gpu-swarm kaggle-32b-feasibility` is the bounded feasibility/live-demo path for the community-facing "multiple Kaggle GPU kernels as stage Miners for a 32B-class quantized model" story. It emits `kaggle_swarm_32b_quantized_feasibility_v1` / `kaggle_swarm_32b_quantized_feasibility_cli_v1` through `scripts/kaggle_swarm_32b_quantized_feasibility_pack.py` and is checked by `scripts/kaggle_swarm_32b_quantized_feasibility_check.py`. Preserve `kaggle_swarm_32b_quantized_feasibility_ready`, `candidate_32b_model_selected`, `quantized_runtime_plan_ready`, `kaggle_multi_kernel_topology_ready`, `stage_partition_plan_ready`, `per_stage_memory_estimate_ready`, `activation_transfer_estimate_ready`, `kaggle_stage_package_plan_ready`, `stage_owned_loading_feasible`, `one_token_generation_feasible`, `multi_token_generation_feasible`, `coordinator_direct_management_feasible`, `upper_bound_crossing_feasible`, `batch_or_sequential_request_feasible`, `stage_requeue_feasible`, `largest_feasible_model_tier`, `largest_attempted_model_tier`, `feasibility_verdict`, `blocked_reason`, `blocker_details`, `execution_mode`, `fresh_kaggle_run_performed`, `external_runtime_verified`, `retained_evidence_imported`, `fresh_32b_activation_decode_probe_summary`, `fresh_32b_cross_kernel_activation_decode_verified`, `fresh_32b_one_token_generation_verified`, `fresh_32b_multi_token_decode_verified`, `fresh_32b_coordinator_direct_management_verified`, `fresh_32b_single_kernel_baseline_attempted`, `fresh_32b_single_kernel_baseline_ok`, `fresh_32b_stage_owned_awq_runtime_verified`, `fresh_32b_activation_handoff_verified`, `fresh_32b_private_activation_removed`, redacted Markdown, public-safe `kaggle_stage_package_plan.json`, and redacted `support_bundle.json`. Default evidence-import consumes retained stage-owned loading proof at `dist/kaggle-32b-stage-owned-safetensors-probe-awq-live-r3-clone/kaggle_32b_stage_owned_safetensors_probe.json` plus the current retained 4-stage upper-bound crossing proof at `dist/kaggle-32b-upper-bound-crossing-live-20260620-r3/kaggle_32b_stage_owned_activation_decode_probe.json`. The current live 32B upper-bound proof used two private Kaggle Tesla T4 x2 kernels with `Qwen/Qwen2.5-32B-Instruct-AWQ` and a temporary proof Coordinator at `24.199.118.54:9235`: shard0 owned stages 0/1 on `cuda:0`/`cuda:1`; shard1 owned stages 2/3 on `cuda:0`/`cuda:1`; Coordinator completed one generated token with stage task counts `stage0..stage3 == 1`, `generated_token_count=1`, private activation handoff hashes only, raw token ids/activations redacted, private kernels deleted, and local private payloads removed. Stage-owned weights were about stage0 5.225433 GB / 417 keys, stage1 3.775238 GB / 416 keys, stage2 3.775238 GB / 416 keys, and stage3 5.225443 GB / 418 keys. The strict same-model/same-prompt single Kaggle T4 x2 baseline was attempted with all four stages required in one kernel and failed closed with `single_kernel_t4x2_gpu_count_below_required_stage_count`, proving the two-kernel path crosses the single-kernel T4 x2 slot-count upper bound under that strict 4-stage placement. The current feasibility report is `dist/kaggle-swarm-32b-quantized-feasibility-upper-bound-crossing-20260620-r1/kaggle_swarm_32b_quantized_feasibility.json` and should report `stage_owned_loading_feasible=true`, `one_token_generation_feasible=true`, `coordinator_direct_management_feasible=true`, `upper_bound_crossing_feasible=true`, `external_runtime_verified=true`, `largest_feasible_model_tier=32b-quantized-4stage-upper-bound-rc`, `feasibility_verdict=feasible_32b_upper_bound_crossing_rc`, and `blocked_reason=""`. Keep `multi_token_generation_feasible=false` for this 1-token 4-stage proof, and keep `batch_or_sequential_request_feasible=false` plus `stage_requeue_feasible=false` until separate evidence exists. Public artifacts must keep raw prompts, generated text, token ids, activations, hidden states, logits, KV cache, model cache private paths, Kaggle credentials, API keys, Coordinator tokens, leases, idempotency material, private env files, registries, inline Kaggle kernel payloads, and runtime-private state redacted. This RC is repeatable 32B stage-owned loading plus temporary-proof-Coordinator upper-bound crossing evidence, not production Swarm Inference, not the production Coordinator data plane, not a memory-pressure/long-context crossing proof, not KV-cache optimized serving, not batch/sequential validation, not stage requeue, not P2P/NAT traversal, not arbitrary public prompt serving, not billing, and not unbounded GPU pooling.
- GPU+TPU+CPU Heterogeneous Stage Alpha: `crowdtensor heterogeneous-stage-alpha` emits `gpu_tpu_cpu_heterogeneous_stage_alpha_v1` / `gpu_tpu_cpu_heterogeneous_stage_alpha_cli_v1` through `scripts/gpu_tpu_cpu_heterogeneous_stage_alpha_pack.py` and is checked by `scripts/gpu_tpu_cpu_heterogeneous_stage_alpha_check.py`. The current retained report is `dist/gpu-tpu-cpu-heterogeneous-stage-alpha-20260622-r3-cli/gpu_tpu_cpu_heterogeneous_stage_alpha.json`; it imports retained GPU evidence from the full-precision 32B 4*T4 + 5*CPU proof and the AWQ 32B upper-bound crossing proof, imports retained CPU stage evidence, imports retained TPU web JAX real-model evidence, and runs a local public-safe three-stage real HF GPT-2 smoke (`gpt2`, 124,439,808 parameters) through target backend families GPU/TPU/CPU with activation hashes only, `baseline_match=true`, and `generated_token_count=1`. Preserve `gpu_tpu_cpu_heterogeneous_stage_alpha_ready`, `backend_evidence_imported`, `gpu_backend_evidence_ready`, `tpu_backend_evidence_ready`, `cpu_backend_evidence_ready`, `logical_stage_contract_ready`, `local_three_stage_real_model_e2e_ready`, `small_medium_real_model_end_to_end_ready`, `gpu_tpu_cpu_32b_feasibility_report_ready`, `next_rc_boundary_ready`, redacted `stage_contract_smoke.json`, redacted `local_three_stage_real_model_e2e.json`, redacted `heterogeneous_32b_feasibility_report.json`, redacted `torch_jax_torch_bridge_probe.json`, and redacted `support_bundle.json`. In r3 the optional Torch-to-JAX-to-Torch bridge probe is public-safe but blocked locally with `jax_missing`, so `torch_jax_torch_bridge_ready=false` and the activation bridge remains next-RC work. Also preserve the explicit Alpha boundaries: `same_request_live_heterogeneous_verified=false`, `live_tpu_stage_miner_integrated=false`, `gpu_tpu_cpu_32b_same_request_feasible_now=false`, and `tpu_32b_runtime_adapter_ready=false`. This is stronger than a pure evidence aggregate because it executes a real local three-stage model path, but it is still not a live same-request GPU+TPU+CPU run, not Qwen/Llama-on-TPU stage execution, not production serving, not P2P/NAT traversal, not billing, and not 32B GPU+TPU+CPU success. The next RC must implement a JAX/TPU Qwen-or-Llama-like stage runtime, a safetensors/MaxText checkpoint bridge, CUDA-to-JAX activation wire format, TPU stage-local KV-cache handling, Coordinator backend capability routing, and bounded live cleanup/requeue before it can claim live same-request heterogeneous inference.
- GPU+TPU+CPU 32B Heterogeneous RC: `crowdtensor heterogeneous-32b-rc` emits `gpu_tpu_cpu_32b_heterogeneous_rc_v1` / `gpu_tpu_cpu_32b_heterogeneous_rc_cli_v1` through `scripts/gpu_tpu_cpu_32b_heterogeneous_rc_pack.py` and is checked by `scripts/gpu_tpu_cpu_32b_heterogeneous_rc_check.py`. The current retained report is `dist/gpu-tpu-cpu-32b-heterogeneous-rc-20260623-r8-web-tpu-qwen32b-stage-runtime-ready/gpu_tpu_cpu_32b_heterogeneous_rc.json`; it imports the r3 Alpha evidence and emits public-safe `stage_runtime_matrix.json`, `activation_protocol.json`, `live_same_request_summary.json`, `tpu_allocation_attempt_summary.json`, `tpu_web_active_event_summary.json`, `tpu_stage_adapter_plan_summary.json`, `tpu_stage_runtime_probe_summary.json`, `blocker_report.json`, Markdown, CLI summary, and Support Bundle. Preserve `gpu_tpu_cpu_32b_heterogeneous_rc_ready`, `gpu_tpu_cpu_32b_bounded_rc_success`, `gpu_tpu_cpu_32b_same_request_verified`, `live_tpu_stage_miner_integrated`, `fallback_model_used`, `tpu_32b_runtime_adapter_ready`, `stage_local_kv_cache_verified`, `tpu_runtime_allocation_attempted`, `tpu_runtime_allocation_ready`, `tpu_runtime_allocation_blocked`, `tpu_stage_adapter_plan_ready`, `tpu_checkpoint_bridge_plan_ready`, `tpu_stage_owned_loader_plan_ready`, `tpu_qwen_like_stage_runtime_probe_ready`, `tpu_qwen32b_single_layer_runtime_probe_ready`, `blocked_reason`, `public_artifact_safe`, and the overclaim-rejection checks. Current r8 status is report-ready but not successful: `gpu_tpu_cpu_32b_bounded_rc_success=false`, `gpu_tpu_cpu_32b_same_request_verified=false`, `live_tpu_stage_miner_integrated=false`, `fallback_model_used=false`, `tpu_32b_runtime_adapter_ready=false`, `stage_local_kv_cache_verified=false`, `tpu_runtime_allocation_attempted=true`, `tpu_runtime_allocation_ready=true`, `tpu_runtime_allocation_blocked=false`, `tpu_stage_adapter_plan_ready=true`, `tpu_checkpoint_bridge_plan_ready=true`, `tpu_stage_owned_loader_plan_ready=true`, `tpu_qwen_like_stage_runtime_probe_ready=true`, `tpu_qwen32b_single_layer_runtime_probe_ready=true`, and `blocked_reason=same_request_live_proof_missing`. The Qwen 32B TPU stage adapter plan is `dist/gpu-tpu-qwen-stage-adapter-plan-qwen32b-20260623-r1/gpu_tpu_qwen_stage_adapter_plan.json`: it maps `Qwen/Qwen2.5-32B-Instruct` layers 21-42 for a JAX/TPU middle stage, with 252 assigned stage-owned keys from 6 safetensors files, 0 unsupported keys, activation metadata shape `[1, 128, 5120]` / `bfloat16` / `batch_seq_hidden`, and stage-local KV-cache metadata while keeping tensor values, activations, tokens, generated text, and private paths out of public artifacts. This proves a metadata-only checkpoint/shape/KV bridge plan, not executed TPU safetensors loading. The Qwen/Llama-like TPU stage runtime probe path is `scripts/kaggle_tpu_qwen_stage_runtime_probe.py`, with tests in `tests/test_kaggle_tpu_qwen_stage_runtime_probe.py`; it can package private Kaggle TPU script kernels and can also import authenticated Web Notebook runtime evidence. It runs a public-safe JAX decoder middle-stage forward with grouped-query attention, RMSNorm, MLP, stage-local KV-cache metadata, activation shape/dtype/layout metadata, and input/output hashes only. The current retained Web TPU stage proof is `dist/kaggle-tpu-qwen-stage-runtime-probe-web-live-20260623-r2/kaggle_tpu_qwen_stage_runtime_probe.json`: the authenticated Kaggle Web Notebook runtime was running on `TPU v5e-8` and the Jupyter API executed both `tiny-qwen-like` and `qwen32b-one-layer` profiles on JAX with 8 `TPU v5 lite` devices. The report has `ok=true`, `tpu_runtime_ready=true`, `qwen_like_stage_runtime_ready=true`, `qwen32b_single_layer_runtime_ready=true`, `stage_local_kv_cache_verified=true`, `selected_accelerator=web-ui-tpu-v5e8`, public stage input/output hashes only, and `jupyter_proxy_token_public=false`. The matching Web active-event runtime evidence is `dist/kaggle-web-tpu-session-retry-20260623-r12-runtime-confirmed/kaggle_tpu_web_active_event_status.json`, with `running=true`, `tpu_runtime_ready=true`, Jupyter kernel state `idle`, and no public proxy token/cookie material. Older TPU scheduling evidence remains useful but is no longer the current blocker: `dist/kaggle-tpu-qwen-stage-runtime-probe-live-20260623-r2-qwen32b-one-layer/kaggle_tpu_qwen_stage_runtime_probe.json` accepted `tpuV5e8` for the private script-kernel `qwen32b-one-layer` profile but stayed `QUEUED` for 30 status polls across the bounded 30-minute wait, produced no runtime report, and was deleted with local private packages removed; `dist/kaggle-tpu-qwen-stage-runtime-probe-live-20260623-r1-tiny/kaggle_tpu_qwen_stage_runtime_probe.json` did the same for `tiny-qwen-like`; `dist/kaggle-web-tpu-session-retry-20260623-r6-start-selected-tpu/kaggle_tpu_web_start_selected_tpu_attempt.json` and `dist/kaggle-web-tpu-session-retry-20260623-r7-active-event-wait/kaggle_tpu_web_active_event_wait.json` show Web UI login, TPU v5e-8 selection, Start Session, public queue position `#17`, and a bounded queued wait before the later runtime-confirmed evidence. The remaining required work is a single Coordinator same-request proof with at least one accepted CUDA/GPU stage task, one accepted JAX/TPU stage task, and one accepted CPU stage task for a 32B-class model request. This is RC scaffolding plus bounded live-attempt evidence, adapter-plan evidence, Web TPU runtime-stage evidence, cleanup evidence, and public artifact safety, not 32B three-accelerator live success; fallback, fixture, queue evidence, adapter-plan-only evidence, runtime-probe-only evidence, or missing same-request evidence must never be marked as 32B same-request success.
- GPU+TPU+CPU 32B Heterogeneous RC current superseding status: use `dist/gpu-tpu-cpu-32b-heterogeneous-rc-20260623-r10-web-tpu-ready-gpu-quota-bridge-blocked/gpu_tpu_cpu_32b_heterogeneous_rc.json` as the current RC artifact instead of the older r8 path above. It imports the r3 Alpha evidence, the Web TPU active-event runtime report, the Qwen/Llama-like Web TPU `qwen32b-one-layer` stage runtime proof, the Qwen 32B TPU adapter plan, and the same-request runtime bridge attempt at `dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260623-r2/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`. The r10 report is valid and public-safe, with `gpu_tpu_cpu_32b_heterogeneous_rc_ready=true`, `tpu_runtime_allocation_blocked=false`, `tpu_qwen_like_stage_runtime_probe_ready=true`, but `gpu_tpu_cpu_32b_bounded_rc_success=false`, `gpu_tpu_cpu_32b_same_request_verified=false`, `live_tpu_stage_miner_integrated=false`, `tpu_32b_runtime_adapter_ready=false`, and `blocked_reason=same_request_live_proof_missing`. The bridge attempt tried one shared Coordinator request with a private Kaggle CUDA/GPU stage, authenticated Web JAX/TPU stage, and local CPU tail stage, but Kaggle rejected the CUDA stage kernel push with `Maximum batch GPU session count of 2 reached`, recorded as `kaggle_gpu_batch_session_limit_reached`; no unrelated existing Kaggle GPU kernels should be deleted to clear this without explicit user permission. This bridge is intentionally `not_32b_weight_success=true`, so even a future bridge success is only same-request runtime plumbing evidence until full Qwen safetensors/TPU stage loading is implemented. The next live attempt should first wait for or free a Kaggle GPU batch session, then rerun the same-request bridge/full proof, then separately upgrade from Qwen-like synthetic TPU stage execution to full 32B stage-owned TPU weight loading before claiming true GPU+TPU+CPU 32B success.
- GPU+TPU+CPU 32B Heterogeneous RC latest current status: use `dist/gpu-tpu-cpu-32b-heterogeneous-rc-20260623-r15-current-gpu-quota-tpu-starting/gpu_tpu_cpu_32b_heterogeneous_rc.json` as the current artifact. It supersedes r10/r12/r13/r14 for current external runtime status while preserving those paths as historical evidence. Relevant code fixes since r10: `scripts/gpu_tpu_cpu_same_request_runtime_bridge_probe.py` now treats `stage_id=0` as a valid claim/count value, with tests in `tests/test_gpu_tpu_cpu_same_request_runtime_bridge_probe.py`, and no longer reports missing Kaggle cleanup when a GPU kernel was never created because push was rejected. The r15 report is valid and public-safe, with `gpu_tpu_cpu_32b_heterogeneous_rc_ready=true`, `gpu_tpu_cpu_32b_bounded_rc_success=false`, `gpu_tpu_cpu_32b_same_request_verified=false`, `live_tpu_stage_miner_integrated=false`, `fallback_model_used=false`, `tpu_32b_runtime_adapter_ready=false`, `stage_local_kv_cache_verified=false`, `tpu_qwen_like_stage_runtime_probe_ready=true`, `tpu_runtime_allocation_blocked=true`, and `blocked_reason=kaggle_web_tpu_session_still_starting`. The retained Web TPU stage proof `dist/kaggle-tpu-qwen-stage-runtime-probe-web-live-20260623-r2/kaggle_tpu_qwen_stage_runtime_probe.json` still proves a previous Web TPU v5e-8 Qwen-like `qwen32b-one-layer` runtime. Current Web status progressed from detached in `dist/kaggle-web-tpu-session-retry-20260623-r13-current-status/kaggle_tpu_web_active_event_status.json` to restarted/requeued in `dist/kaggle-web-tpu-session-retry-20260623-r14-restart-tpu/kaggle_tpu_web_start_selected_tpu_attempt.json` and then still-starting in `dist/kaggle-web-tpu-session-retry-20260623-r15-current-status/kaggle_tpu_web_active_event_status.json`: `TPU v5e-8` was selected and started, but `running=false`, `tpu_runtime_ready=false`, and no Jupyter proxy is public. The current same-request bridge report is `dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260623-r5-current-retry/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`; after the stage0 claim fix, Kaggle still rejected CUDA stage kernel creation with `Maximum batch GPU session count of 2 reached`, recorded as `kaggle_gpu_batch_session_limit_reached` and `kaggle_gpu_kernel_not_created`, so no CUDA/JAX-TPU/CPU accepted stage chain exists. Current blockers are therefore external runtime availability first (GPU batch session slot plus TPU runtime allocation), then full 32B TPU safetensors/stage-owned loading. Do not mark the goal complete until the same Coordinator request has accepted CUDA/GPU, JAX/TPU, and CPU stage tasks and the 32B success boundary is met or an explicitly scoped fallback/blocker report is requested.
- GPU+TPU+CPU 32B Heterogeneous RC r16 superseding status: use `dist/gpu-tpu-cpu-32b-heterogeneous-rc-20260623-r16-gpu-stage-accepted-tpu-queued/gpu_tpu_cpu_32b_heterogeneous_rc.json` as the latest current artifact. It supersedes r15 for current bridge evidence. The r16 report is valid and public-safe, with `gpu_tpu_cpu_32b_heterogeneous_rc_ready=true`, `gpu_tpu_cpu_32b_bounded_rc_success=false`, `gpu_tpu_cpu_32b_same_request_verified=false`, `live_tpu_stage_miner_integrated=false`, `tpu_32b_runtime_adapter_ready=false`, `stage_local_kv_cache_verified=false`, `tpu_qwen_like_stage_runtime_probe_ready=true`, `tpu_runtime_allocation_blocked=true`, and `blocked_reason=kaggle_web_tpu_runtime_queued_after_restart`. The current same-request bridge report is `dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260623-r6-gpu-slot-retry-tpu-queued/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`: after the stage0 claim fix and after the earlier GPU batch slot cleared, a private Kaggle Tesla T4 x2 CUDA stage kernel was created, ran, produced a public-safe report, and was deleted. It has `accepted_stage_backends=["cuda"]`, `runtime_device_summary.cuda_stage_ready=true`, `cuda_device_count=2`, `stage_task_counts.stage0=1`, and one activation handoff hash, but `same_request_runtime_bridge_verified=false` because the Web TPU runtime is still queued and stage1 was not accepted; CPU tail has no task until stage1 completes. Current Web TPU status is `dist/kaggle-web-tpu-session-retry-20260623-r16-restart-after-gpu-free/kaggle_tpu_web_active_event_status.json`: `TPU v5e-8` was selected and restarted, queue signals were observed for a bounded 120-second wait, `running=false`, and `tpu_runtime_ready=false`. The next live attempt should wait for Web TPU runtime/Jupyter proxy availability, then rerun the same-request bridge while GPU slot is free. Do not claim full 32B success until JAX/TPU stage and CPU tail are also accepted in the same Coordinator request and the full 32B TPU stage-owned loading boundary is satisfied or explicitly reported as a blocker/fallback.
- GPU+TPU+CPU 32B Heterogeneous RC r20 superseding status: use `dist/gpu-tpu-cpu-32b-heterogeneous-rc-20260623-r20-same-request-bridge-partial-32b-tpu-loader/gpu_tpu_cpu_32b_heterogeneous_rc.json` as the current artifact. It is valid and public-safe, imports the r3 Alpha evidence, the running Web TPU status at `dist/kaggle-web-tpu-session-retry-20260623-r18-active-event-running/kaggle_tpu_web_active_event_status.json`, the same-request runtime bridge at `dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260623-r7-tpu-running/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`, the Qwen 32B adapter plan, the Web TPU qwen32b-one-layer runtime probe, and the new real safetensors loader probe at `dist/kaggle-tpu-32b-stage-owned-loader-probe-web-live-20260623-r1/kaggle_tpu_32b_stage_owned_loader_probe.json`. The r7 bridge proves one Coordinator request accepted CUDA/GPU, JAX/TPU, and CPU stage tasks (`accepted_stage_backends=["cpu","cuda","jax_tpu"]`, `stage0..stage2 == 1`, two activation handoff hashes, and one generated token hash) and deleted the private Kaggle T4 x2 CUDA kernel, but it remains `not_32b_weight_success=true`. The r1 loader probe ran inside authenticated Kaggle Web TPU v5e-8, verified Qwen/Qwen2.5-32B-Instruct stage layers 21-42 have 252 stage-owned keys across 6 safetensors files with all expected stage keys present in headers, and converted one real BF16 tensor byte range (`model.layers.21.input_layernorm.weight`, 10,240 bytes) into a JAX array on 8 TPU v5 lite devices, exposing only public-safe hashes. Current r20 status is `gpu_tpu_cpu_32b_bounded_rc_success=false`, `gpu_tpu_cpu_32b_same_request_verified=false`, `live_tpu_stage_miner_integrated=false`, `tpu_32b_runtime_adapter_ready=false`, `stage_owned_header_verified=true`, `partial_tensor_to_tpu_verified=true`, and `full_stage_owned_tpu_loader_ready=false`. The remaining blocker is full 21-layer Qwen 32B TPU stage-owned loading/execution plus a real same-request 32B live proof; do not treat bridge success, qwen32b-shape synthetic runtime, or partial tensor-to-TPU evidence as full 32B GPU+TPU+CPU success.
- GPU+TPU+CPU 32B Heterogeneous RC r25 superseding status: use `dist/gpu-tpu-cpu-32b-heterogeneous-rc-20260623-r25-real-tpu-stage-web-detached/gpu_tpu_cpu_32b_heterogeneous_rc.json` as the current artifact instead of r20/r16/r15. It is valid and public-safe, and `scripts/gpu_tpu_cpu_32b_heterogeneous_rc_check.py --report dist/gpu-tpu-cpu-32b-heterogeneous-rc-20260623-r25-real-tpu-stage-web-detached/gpu_tpu_cpu_32b_heterogeneous_rc.json --json` passes with no errors. Current r25 status is `gpu_tpu_cpu_32b_bounded_rc_success=false`, `gpu_tpu_cpu_32b_same_request_verified=false`, `live_tpu_stage_miner_integrated=false`, `fallback_model_used=false`, `tpu_32b_runtime_adapter_ready=true`, `tpu_runtime_allocation_blocked=true`, `tpu_runtime_allocation_ready=false`, and `blocked_reason=kaggle_web_tpu_runtime_not_currently_attached`. The full TPU loader proof is now `dist/kaggle-tpu-32b-stage-owned-loader-probe-web-live-20260623-r3-full-21-layer-real/kaggle_tpu_32b_stage_owned_loader_probe.json`: authenticated Kaggle Web TPU v5e-8 executed the full `Qwen/Qwen2.5-32B-Instruct` middle TPU stage for layers 21-42 with 252 stage-owned safetensors keys, `missing_stage_key_count=0`, `executed_layer_count=21`, `full_stage_owned_tpu_loader_ready=true`, `tpu_32b_runtime_adapter_ready=true`, `loaded_execution_tensor_key_count=252`, about 19.07 GB logical execution tensor bytes, 8 `TPU v5 lite` devices, and public-safe hashes only. `scripts/kaggle_tpu_32b_stage_owned_loader_probe.py` owns that Web TPU loader path, and `scripts/gpu_tpu_cpu_same_request_runtime_bridge_probe.py` now has an explicit `--web-tpu-32b-execute` path that embeds the real loader into the same-request JAX/TPU stage and emits `gpu_tpu_cpu_32b_same_request_live_proof.json` for RC import. The retained real-TPU bridge attempt is `dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260623-r8-real-tpu-32b-stage/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`: a private Kaggle Tesla T4 x2 CUDA stage0 kernel was pushed, ran, downloaded, and deleted; `cuda_device_count=2`, `stage0=1`, one CUDA-to-TPU activation handoff hash, and prior retained CUDA 32B weight evidence were recorded. It did not complete the goal because the Web TPU stage failed before claiming/submitting stage1 (`stage1=0`, accepted backends only `["cuda"]`), so the CPU tail had no stage2 task and generated-token count stayed 0. The corresponding `gpu_tpu_cpu_32b_same_request_live_proof.json` is present but `ok=false`. After the bridge failure, a Web UI restart selected TPU v5e-8 and entered the queue at `#9`, then `dist/kaggle-web-tpu-session-retry-20260623-r20-wait-after-queue9/kaggle_tpu_web_active_event_status.json` recorded a bounded 900-second wait ending with `running=false`, `tpu_runtime_ready=false`, no Jupyter proxy, and `kaggle_web_tpu_runtime_not_currently_attached`. This means the code path has advanced from partial-loader/shape-bridge to real full TPU-stage loader plus same-request live-proof export, but the goal is still incomplete until a fresh same Coordinator request accepts CUDA/GPU, real JAX/TPU 32B stage-owned, and CPU tail/verifier tasks together. Queue evidence, detached Web TPU status, the older synthetic bridge, or the standalone full TPU loader must not be marked as completed three-accelerator 32B live inference.
- GPU+TPU+CPU 32B Heterogeneous RC r26 successful bounded status: use `dist/gpu-tpu-cpu-32b-heterogeneous-rc-20260623-r26-real-tpu-stage-same-request-success/gpu_tpu_cpu_32b_heterogeneous_rc.json` as the current artifact instead of r25 and older blocked reports. It imports the successful same-request live proof at `dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260623-r9-real-tpu-32b-stage-runtime-restored/gpu_tpu_cpu_32b_same_request_live_proof.json`, the r9 bridge report at `dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260623-r9-real-tpu-32b-stage-runtime-restored/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`, the r3 full Web TPU loader, the Qwen 32B adapter plan, and retained TPU runtime evidence. The RC checker passes with no errors and the r26 report has `gpu_tpu_cpu_32b_heterogeneous_rc_ready=true`, `gpu_tpu_cpu_32b_bounded_rc_success=true`, `gpu_tpu_cpu_32b_same_request_verified=true`, `live_tpu_stage_miner_integrated=true`, `fallback_model_used=false`, `tpu_32b_runtime_adapter_ready=true`, `stage_local_kv_cache_verified=true`, `blocked_reason=""`, and `public_artifact_safe=true`. In r9, one Coordinator request accepted all three required stages: `stage0=1` CUDA/GPU, `stage1=1` JAX/TPU, and `stage2=1` CPU tail/verifier, with `accepted_stage_backends=["cpu","cuda","jax_tpu"]`, two activation handoff hashes, and one generated-token hash. Stage0 used a private Kaggle Tesla T4 x2 CUDA kernel and was cleaned up after output collection. Stage1 ran inside authenticated Kaggle Web TPU v5e-8 on 8 `TPU v5 lite` devices and executed the real `Qwen/Qwen2.5-32B-Instruct` layers 21-42 stage-owned loader with `executed_layer_count=21`, `loaded_execution_tensor_key_count=252`, `loaded_execution_tensor_gb=19.072947`, `stage_owned_model_loaded=true`, and `stage_local_kv_cache_verified=true` before submitting the TPU stage task. Stage2 was the local CPU tail/verifier and completed the request. Public artifacts keep raw prompts, generated text, generated token ids, activations, hidden states, logits, KV-cache tensors, credentials, cookies, Jupyter proxy token, leases, idempotency material, private runtime state, and private Kaggle payloads out of reports. Keep the boundary precise: this is a bounded same-request 32B stage-inference RC proving CUDA + real JAX/TPU 32B middle-stage + CPU tail coordination, not production serving, not throughput/TTFT/SLA evidence, not P2P/NAT traversal, not training/fine-tuning, not billing/settlement, and not a full end-to-end Qwen 32B quality/parity benchmark across every layer. The r9 bridge ties the CUDA stage to prior retained 32B stage-owned CUDA evidence rather than reloading a full 32B CUDA stage in that bridge; the real full 32B-weight execution inside the r9 same request is the TPU middle stage.
- Heterogeneous 32B Serving r4: `crowdtensor heterogeneous-32b-serving` emits `heterogeneous_32b_serving_v1` / `heterogeneous_32b_serving_cli_v1` through `scripts/heterogeneous_32b_serving_pack.py` and validates with `scripts/heterogeneous_32b_serving_check.py`. The current retained product-like serving engineering report is `dist/heterogeneous-32b-serving-20260623-r4-live-attempt-web-tpu-proxy-blocked/heterogeneous_32b_serving.json`; it imports the r26 same-request 32B source proof and emits public-safe `deployment_plan.json`, `streaming_response_contract.json`, `latency_metrics.json`, `stage_local_kv_cache.json`, `failure_requeue.json`, `live_external_multitoken_attempt.json`, `blocker_report.json`, Markdown, CLI summary, and Support Bundle. Preserve top-level `heterogeneous_32b_serving_ready=true`, `production_like_serving_path_ready=true`, `gpu_tpu_cpu_32b_same_request_source_verified=true`, `multi_token_generation_ready=true`, `streaming_response_contract_ready=true`, `stage_local_kv_cache_ready=true`, `latency_metrics_ready=true`, `failure_requeue_ready=true`, `public_artifact_safe=true`, `live_external_runtime_verified=false`, and `blocked_reason=web_tpu_jupyter_proxy_not_found`. The deployment plan defines product-facing Coordinator, CUDA/GPU Miner, JAX/TPU Miner, CPU tail/verifier, and user generation commands; the serving contract models at least four token events with token hashes only, stage-local KV-cache reuse/status, per-stage latency, TTFT-like and throughput metrics, activation byte counts, cleanup summary, and a bounded TPU-timeout requeue proof. `live_external_multitoken_attempt.json` records `fresh_live_run_attempted=true`, `attempt_status=attempted_external_runtime_blocked`, `bridge_attempt_report=true`, requested four generated tokens for `Qwen/Qwen2.5-32B-Instruct`, accepted backends `["cuda"]`, stage counts `stage0=1`, `stage1=0`, `stage2=0`, and blockers headed by `web_tpu_jupyter_proxy_not_found`. The source live attempt is `dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260623-r11-real-tpu-32b-4token-serving-attempt-cleanup/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`; it used the Kaggle CUDA GPU + Kaggle Web JAX/TPU + local CPU tail topology, accepted one CUDA stage task, but Web TPU stage1 could not attach through Jupyter, so no stage1/stage2 tasks or live external generated tokens completed. Temporary private GPU kernels from the r10/r11 attempts were explicitly deleted after the attempt. This completes the reusable deployment/serving engineering path for the current 32B three-accelerator proof and records a real blocked external four-token attempt, but it is not a successful fresh external four-token live serving run and must not claim production SLA, true P2P/NAT traversal, billing/settlement, training/fine-tuning, unbounded Kaggle stability, or larger-model capacity. Only a future fresh live serving report may set `live_external_runtime_verified=true`; checker logic rejects fixture, queue-only, fallback model, partial-loader, single-stage, or one-token-only evidence as live production-like 32B serving success.
- Heterogeneous 32B Serving 2026-06-24 follow-up: `scripts/gpu_tpu_cpu_same_request_runtime_bridge_probe.py` now has a hardened default Kaggle Web TPU executor that runs through the authenticated JupyterLab iframe using browser-origin `/api/kernels` and WebSocket execution, with the old external `jupyter-proxy` token scrape retained only as fallback. `tests/test_gpu_tpu_cpu_same_request_runtime_bridge_probe.py` covers stdout report extraction, iframe/Jupyter failure classification, public-safe failure redaction, and the existing 4-token bridge boundary; the focused bridge tests and heterogeneous serving tests pass. The fresh Web TPU restart was accepted, but the bounded wait did not allocate a running TPU kernel: retained scheduling/status evidence is `dist/kaggle-web-tpu-session-retry-20260624-r1-start-queue6/kaggle_tpu_web_start_selected_tpu_attempt.json` and `dist/kaggle-web-tpu-session-retry-20260624-r3-current-status/kaggle_tpu_web_active_event_status.json`, where the Notebook remains `Session is starting`, Jupyter API is visible but kernel count is 0, and `tpu_runtime_ready=false`. The blocked bridge status is `dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260624-r1-web-tpu-runtime-starting/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`: no fresh GPU kernel was started, stage counts are all 0, and `generated_token_count=0` because starting the GPU path without a running TPU would waste the Kaggle GPU slot. The serving aggregate `dist/heterogeneous-32b-serving-20260624-r1-web-tpu-runtime-starting/heterogeneous_32b_serving.json` still reports product-like deployment engineering ready from the r26 1-token source proof, but `live_external_runtime_verified=false` with blockers headed by `fresh_live_bridge_not_started_to_avoid_wasting_gpu_runtime` and `kaggle_web_tpu_runtime_not_ready`. Treat this as Web TPU access hardening plus bounded allocation blocker evidence, not a successful 4-token live serving proof and not product-level three-Accelerator 32B service readiness.
- Heterogeneous 32B Serving 2026-06-25 follow-up: the more reliable Kaggle Web TPU control path is the JupyterLab frontend `window.jupyterapp.serviceManager`, not root `/api/kernels`. `scripts/gpu_tpu_cpu_same_request_runtime_bridge_probe.py` now waits for `jupyterapp.serviceManager`, can execute Python/JAX through a service-manager session, and includes a mediated TPU-stage path where the TPU notebook executes the real 32B loader while the local bridge thread performs Coordinator claim/submit. Direct Web TPU evidence improved: `dist/web-tpu-jax-op-direct-probe-20260624-r1/web_tpu_jax_op_direct_probe.json` shows 8 `TPU v5 lite` devices and a successful TPU op, and `dist/web-tpu-32b-stage-code-smoke-20260624-r1/web_tpu_32b_stage_loader_service_manager_smoke.json` records a successful service-manager execution of the real `Qwen/Qwen2.5-32B-Instruct` TPU stage loader with `full_stage_owned_tpu_loader_ready=true`, 21 executed layers, 252 loaded execution tensors, about 19.07 GB logical tensor bytes, and public-safe hashes. Fresh 4-token bridge attempts still did not complete: `dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260624-r3-service-manager-timeout-fixed-4token/gpu_tpu_cpu_same_request_runtime_bridge_probe.json` accepted CUDA stage0 once but stage1 stayed at zero, and temporary kernel `xuyuhaosuyi/ct-gpu-tpu-cpu-bridge-82315445` was deleted. Current blocked status is `dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260625-r1-service-manager-mediated-blocked/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`, and the serving aggregate is `dist/heterogeneous-32b-serving-20260625-r1-service-manager-mediated-blocked/heterogeneous_32b_serving.json`; checker passes it as product-like deployment engineering with `live_external_runtime_verified=false`, blocker `same_request_runtime_bridge_not_verified`, and stage counts `stage0=1`, `stage1=0`, `stage2=0`. Treat this as stronger Web TPU runtime/Jupyter access evidence plus a still-blocked 4-token live serving attempt, not a successful 4-token proof. The next concrete fix is to replace the hanging service-manager `requestExecute`/Playwright evaluate path with a reliably cancellable execution/collection path, or otherwise retrieve the TPU loader report and submit stage1 locally before rerunning the 4-token bridge.
- Heterogeneous 32B Serving 2026-06-25 late continuation: Web TPU/Jupyter execution is now bounded and cleaner, but 4-token live serving is still not complete. `scripts/gpu_tpu_cpu_same_request_runtime_bridge_probe.py` wraps Web TPU iframe execution in a subprocess hard-timeout boundary, adds bounded service-manager session/kernel/execute handling with session shutdown, and tries a service-manager-derived WebSocket execution path before falling back to `requestExecute`; `tests/test_gpu_tpu_cpu_same_request_runtime_bridge_probe.py` covers those boundaries. `scripts/heterogeneous_32b_serving_pack.py` now prioritizes `kaggle_gpu_batch_session_limit_reached` before downstream stage-not-ready effects. Verified commands: `python -m py_compile scripts/gpu_tpu_cpu_same_request_runtime_bridge_probe.py scripts/heterogeneous_32b_serving_pack.py` and `python -m pytest tests/test_gpu_tpu_cpu_same_request_runtime_bridge_probe.py tests/test_heterogeneous_32b_serving.py -q` passed with 29 tests. The authenticated Kaggle Web TPU was restarted through the UI, queued at #9, then became active; a live JAX smoke saw 8 `TPU v5 lite` devices and completed a TPU matrix operation. The new retained full-loader proof is `dist/web-tpu-32b-full-loader-service-manager-20260625-r1/web_tpu_32b_full_loader_runtime_report.json`: `Qwen/Qwen2.5-32B-Instruct` layers 21-42 loaded and executed 21/21 TPU-stage layers, 252 stage-owned tensors, 19.072947 GB logical tensor bytes, no missing stage keys, stage-local KV-cache verified, and public-safe hashes only. Fresh 4-token bridge attempts at `dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260625-r2-service-manager-ws-fallback-4token/gpu_tpu_cpu_same_request_runtime_bridge_probe.json` and `dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260625-r3-service-manager-ws-fallback-4token-retry/gpu_tpu_cpu_same_request_runtime_bridge_probe.json` did not create a Kaggle GPU kernel because Kaggle returned `Maximum batch GPU session count of 2 reached`; no temporary bridge kernel or private bridge package remained. The updated serving aggregate is `dist/heterogeneous-32b-serving-20260625-r3-gpu-quota-blocked/heterogeneous_32b_serving.json`; its checker passes with `live_external_runtime_verified=false` and `blocked_reason=kaggle_gpu_batch_session_limit_reached`. Active Events showed only the current TPU interactive session, while CLI confirmed `xuyuhaosuyi/v12100-live-middle-model-sft` was RUNNING and was not created by this goal. Do not delete unrelated user Kaggle sessions without explicit permission. Next step: wait for or explicitly free Kaggle GPU batch capacity, then rerun the 4-token bridge while the Web TPU session remains runnable.
- Heterogeneous 32B Serving 2026-06-25 r4 retry: Web TPU remained executable (`web_tpu_still_alive_smoke_v1` saw 8 TPU devices), and MCP accelerator quota showed GPU time quota available (`time_used=0s`, `time_reserved=0s`) but Kaggle still rejected a fresh T4 GPU script-kernel push with `Maximum batch GPU session count of 2 reached`. This confirms the blocker is Kaggle's concurrent GPU batch/session limit, not GPU hour quota and not TPU/Jupyter access. The r4 bridge attempt is `dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260625-r4-gpu-slot-retry-4token/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`; it created no temporary bridge kernel, removed the private bridge package, and left `stage0=stage1=stage2=0`, `generated_token_count=0`. The r4 serving aggregate is `dist/heterogeneous-32b-serving-20260625-r4-gpu-quota-blocked/heterogeneous_32b_serving.json`; checker passes with `live_external_runtime_verified=false` and `blocked_reason=kaggle_gpu_batch_session_limit_reached`. Do not mark the goal complete until a fresh bridge report proves at least 4 generated tokens with CUDA, JAX/TPU, and CPU accepted stage tasks in the same request.
- Kaggle TPU LLM Probe: `scripts/kaggle_tpu_llm_probe.py` emits `kaggle_tpu_llm_probe_v1` for a bounded private Kaggle TPU scheduling/runtime probe. It packages a script kernel with `enable_tpu=true`, `enable_gpu=false`, and accelerator/machine-shape candidates headed by Kaggle's frontend internal TPU v5e-8 value `tpuV5e8`; the kernel attempts public-safe TPU discovery through JAX, torch_xla, and TensorFlow, with the LLM success condition currently defined as a JAX `jax_tiny_causal_lm_jit` one-token synthetic causal-LM forward/generate on a TPU device. Retained live attempts are `dist/kaggle-tpu-llm-probe-live-20260621-r1/kaggle_tpu_llm_probe.json` for display string `TPU v5e-8`, `dist/kaggle-tpu-llm-probe-live-20260621-r2-internal-shape/kaggle_tpu_llm_probe.json` for `tpuV5e8`, `dist/kaggle-tpu-llm-probe-live-20260621-r3-cli222/kaggle_tpu_llm_probe.json` for `tpuV5e8` with isolated Kaggle CLI 2.2.2, and `dist/kaggle-tpu-llm-probe-live-20260621-r4-tpu1vmv38/kaggle_tpu_llm_probe.json` for `tpu1vmV38`. Kaggle accepted and created each private TPU kernel, but every CLI/API attempt remained `QUEUED` until the bounded status timeout, produced no runtime report, and was deleted. MCP `save_notebook` could run private notebooks but ignored v5e machine-shape requests and fell back to CPU (`machine_shape: None`, JAX devices CPU only); MCP `create_notebook_session` is unavailable for this account without `kernelSessions.create`. The authenticated web UI path was verified with a temporary cookie-backed browser session; retained initial queue evidence is `dist/kaggle-tpu-web-session-attempt-20260621-r1/kaggle_tpu_web_session_attempt.json`. The web UI logged in as `tpuowner`, exposed `TPU v5e-8` in Session options, showed TPU quota remaining, accepted `Turn on TPU v5e-8`, and entered the TPU queue. A later long web wait preserved the active event instead of restarting it: `notebookcbde0293fe` remained `Interactive Session with TPU v5e-8` queued for about 3 hours, then became `Running`; retained evidence is `dist/kaggle-tpu-active-event-longwait-20260621-r1/kaggle_tpu_active_event_longwait.json`. Opening that active event produced the runtime notebook `from-pathlib-import-path`, with retained evidence at `dist/kaggle-open-active-event-runtime-20260621-r1/open_active_event_runtime.json`. The runtime proof at `dist/kaggle-tpu-web-runtime-proof-20260621-r1/kaggle_tpu_web_runtime_proof.json` reports JAX seeing 8 `TPU v5 lite` devices, `PJRT_DEVICE` plus TPU worker environment present, and a successful simple TPU matrix operation (`simple_tpu_op_ready: true`). The stronger retained synthetic LLM-style proof is `dist/kaggle-tpu-web-synthetic-llm-probe-20260621-r1/kaggle_tpu_web_runtime_probe.json`: it reports `ok: true`, `synthetic_llm_ready: true`, `synthetic_llm_runtime: jax_tiny_causal_lm_jit`, `generated_token_count: 1`, 8 `TPU v5 lite` devices, a redacted next-token hash, and no public generated token ids, activations, or KV-cache material. Preserve `kaggle_web_tpu_v5e8_option_visible`, `kaggle_web_tpu_queue_entered`, `kaggle_tpu_active_event_runtime_ready`, `web_tpu_synthetic_llm_probe_ready`, cleanup evidence, public-safe redaction, and the boundary that this is Kaggle web TPU runtime allocation plus JAX synthetic causal-LM proof, not torch_xla/TensorFlow proof, not Hugging Face/Qwen on TPU, not large-model TPU serving, and not production TPU pooling. Cookie files are sensitive local-only artifacts and should be deleted/rotated after live web experiments.
- Public Swarm GPU Inference Beta: `crowdtensor public-swarm-gpu-beta` emits `public_swarm_gpu_inference_beta_v1` through `scripts/public_swarm_gpu_inference_beta_pack.py` and is checked by `scripts/public_swarm_gpu_inference_beta_check.py`. It is the optional CUDA overlay for the real tiny GPT split proof. `public-swarm-gpu-beta local-smoke` is CI-safe on CPU-only hosts and should emit `public_swarm_gpu_beta_smoke_ready` without claiming `public_swarm_gpu_beta_ready`. `public-swarm-gpu-beta local-loopback` selects `hf_transformers_cuda`, requires `cuda_runtime_available`, `hf_transformers_cuda_ready`, `gpu_runtime_ready`, stage capabilities `real_llm_sharded_cuda_stage0`, `real_llm_sharded_cuda_stage1`, or `real_llm_sharded_cuda_both`, and should emit `public_swarm_gpu_beta_ready`, `gpu_stage0_ready`, `gpu_stage1_ready`, `stage_local_partition_ready`, `stage0_partition_loaded`, `stage1_partition_loaded`, and `partition_parameter_split_valid` after a real CUDA loopback. `public-swarm-gpu-beta kaggle-package` prepares private Kaggle GPU stage templates and should emit `kaggle_gpu_package_ready`. `public-swarm-gpu-beta kaggle-auto` is the side-effectful private Kaggle GPU proof: a CPU-capable public Coordinator may create CUDA metadata-only sessions, CUDA execution is deferred to private Kaggle GPU stage Miners, and a ready report must include `public_swarm_gpu_beta_kaggle_auto_ready`, `external_gpu_runtime_verified`, `kaggle_kernels_deleted`, `token_rotation_required`, the stage-local partition codes, top-level `output_request`, `answer_scope.scope_state: no-local-answer`, and `shareable_summary.answer_scope_state`. The GPU Beta overlay defaults to `--real-llm-partition-mode stage-local`, so private Kaggle GPU stage Miners prove stage-owned CUDA module placement instead of moving the full model to CUDA. The retained successful stage-local Kaggle GPU proof is `dist/public-swarm-gpu-beta-stage-local-live-20260528064520-shortslug/public_swarm_gpu_inference_beta_kaggle_auto.json`; it reports `stage_local_partition_ready`, `stage0_partition_loaded`, `stage1_partition_loaded`, `partition_parameter_split_valid`, `stage_gpu_memory_reduced`, and `kaggle_kernels_deleted`. The older `dist/public-swarm-gpu-beta-live-20260528-runtimepin/public_swarm_gpu_inference_beta_kaggle_auto.json` proof is retained historical pre-stage-local CUDA runtime-pin evidence. `public-swarm-gpu-beta evidence-import` imports a completed GPU report with `external_gpu_runtime_verified`. Treat it as shareable optional CUDA readiness evidence, not a local answer transcript; public artifacts keep raw prompts, generated text, generated token ids, activations, credentials, leases, private env files, Kaggle kernel payloads, and runtime state redacted. This is read-only optional CUDA tiny GPT evidence, not production Swarm Inference, not P2P, not a GPU pooling marketplace, and not large-model serving.
- GPU Sharded Generation Beta: `crowdtensor gpu-generate` emits `gpu_sharded_generation_beta_v1` / `gpu_sharded_generation_beta_cli_v1` through `scripts/gpu_sharded_generation_beta_pack.py` and is checked by `scripts/gpu_sharded_generation_beta_check.py`. It wraps the optional CUDA Public Swarm GPU path and threads `--max-new-tokens` through the real LLM split stack so stage0/stage1 alternate until the requested generation count completes. Modes are `local-loopback`, `kaggle-auto`, and `evidence-import`; old single-token GPU evidence must surface `gpu_multi_token_generation_missing` instead of passing this Beta. Ready reports preserve `gpu_sharded_generation_ready`, `multi_token_generation_ready`, `gpu_loopback_generation_ready` or `gpu_multi_machine_generation_ready`, `generated_token_count`, `generated_text_hash`, `generated_text_redacted`, `raw_generated_text_public: false`, top-level `output_request`, `answer_scope.scope_state: no-local-answer`, `shareable_summary.answer_scope_state`, `hf_transformers_cuda`, stage-local partitioning, distinct stage Miners, and Kaggle cleanup evidence when applicable. The retained 16-token private Kaggle GPU proof is `dist/gpu-sharded-generation-beta-kaggle-20260528095658/gpu_sharded_generation_beta_kaggle_auto.json`, with RC manifest `dist/gpu-sharded-generation-beta-kaggle-20260528095658/gpu_sharded_generation_beta_rc_manifest.json`; private env and registry files have been removed from that retained artifact tree. Treat it as shareable generation evidence, not a local answer transcript; public artifacts keep raw prompts, generated text, generated token ids, activations, credentials, leases, and runtime state redacted. This is a tiny GPT CUDA multi-token Beta proof, not production Swarm Inference, not Hivemind-level serving, not P2P, not a GPU marketplace, not arbitrary prompt serving, and not large-model serving.
- Public Swarm Product RC: `crowdtensor public-swarm-product-rc` emits `public_swarm_product_rc_v1` through `scripts/public_swarm_product_rc_pack.py` and validates with `scripts/public_swarm_product_rc_check.py`. It is the current product-surface bridge from Coordinator-backed GPU generation evidence toward a later P2P daemon: `crowdtensor serve` prints/runs Coordinator commands, `crowdtensor join` prints/runs stage Miner commands, `crowdtensor generate` builds bounded `session_protocol_v1` requests with prompt hashes and optional Coordinator session creation, and `crowdtensor peer` exposes `p2p_lite_peer_v1` HTTP-gossip discovery through `scripts/p2p_lite_daemon.py`. Supporting checks are `scripts/session_protocol_check.py` and `scripts/p2p_lite_discovery_check.py`. P2P-lite can resolve Coordinator and stage-capable Miner routes, but Coordinator still owns leases, heartbeats, validation, and result ledgers. Public RC artifacts must not include raw prompts, generated text, token ids, activations, admin/miner/observer tokens, leases, or idempotency material. This is not libp2p, not DHT, not NAT traversal, not decentralized security, not Hivemind/Petals-level serving, and not large-model serving.
- Product Swarm v0.3 MVP check: `scripts/product_swarm_mvp_check.py` emits `product_swarm_mvp_check_v1` as the direct runnable product-command proof. It starts `crowdtensor serve --run`, starts `crowdtensor generate` against the live Coordinator so the session exists, then runs independent one-task `crowdtensor join --stage stage0 --run` and `crowdtensor join --stage stage1 --run` commands for every real tiny-GPT generation step. It can also run bounded batch evidence through `--prompt-texts`; each stage task processes the batch for that generation step, and reports should preserve safe prompt hashes, per-request generated text hashes, `batch_generation_ready`, `public_swarm_generate_batch_ready`, and `product_swarm_mvp_batch_ready`. `--stream-generation` runs `crowdtensor generate --stream` and requires safe progress events before emitting `product_swarm_mvp_stream_ready`, `public_swarm_generate_stream_ready`, and `public_swarm_generate_stream_endpoint_ready`. `--shareable-generate-terminal` runs the same local product loop through real human `crowdtensor generate --prompt-stdin --shareable-terminal` output and requires `shareable_generate_terminal_ready`, `product_swarm_mvp_shareable_generate_terminal_v1`, `answer_scope_state=shareable-terminal-redacted`, `gpu_state=local-cpu-only`, `fresh_kaggle_gpu_verified=false`, and `terminal_answer_text_hidden=true` without persisting raw terminal output. A ready report should include `product_swarm_mvp_ready`, `serve_join_generate_mvp_ready`, `local_two_stage_real_llm_ready`, `generated_token_count_ready`, `distinct_stage_miners`, and `stage_assignment_valid`. If optional `[hf]` dependencies are unavailable, default output is degraded with `hf_dependencies_missing` and `product_swarm_mvp_degraded_ready`; `--require-hf-runtime` makes that a hard failure. Preserve CPU default, optional `--backend cuda`, safe generation counts and hashes, redaction of raw prompts/generated text/terminal output/token ids/activations/leases/idempotency material, and the boundary that this is Coordinator-backed, read-only, tiny-model scoped, not production, not P2P, not Hivemind/Petals-level, and not large-model serving.
- P2P Swarm Inference v0.6: `crowdtensor p2p-swarm-v06` emits `p2p_swarm_inference_v06_v1` through `scripts/p2p_swarm_inference_v06_pack.py` and validates with `scripts/p2p_swarm_inference_v06_check.py`. Preserve the top-level `crowdtensor p2pd` command, `serve --p2p`, `join --p2p`, `generate --p2p --prompt`, optional `--prompt-texts`, optional `--stream-generation`, `p2pd_cli_v1`, `p2p_lite_peer_v1`, Coordinator result fallback, modes `local-smoke`, `package`, `evidence-import`, `external-existing`, and `kaggle-auto`, and the report codes `p2p_swarm_inference_v06_ready`, `p2p_discovery_routing_prototype_ready`, `local_three_process_p2p_discovery_ready`, `p2p_stage_discovery_ready`, `p2p_generate_route_ready`, `p2p_stage_rescue_ready`, `p2p_real_generate_ready`, `p2p_real_generate_stream_ready` when requested stream progress is complete, `p2p_real_stage_rescue_ready`, `p2p_v06_model_metadata_ready`, `external_p2p_stage_discovery_ready`, `external_p2p_generate_verified`, `p2p_swarm_inference_v06_kaggle_auto_ready`, `kaggle_kernels_deleted`, `coordinator_to_p2p_transition_ready`, and `coordinator_result_fallback_ready`. Reports now expose `p2p.hf_model_id`, `p2p.observed_hf_model_id`, and `p2p.model_id_match`; non-default `--hf-model-id` imports must expose matching P2P evidence metadata or emit `p2p_v06_model_metadata_mismatch` and block readiness. The CI-safe non-default model check is `python scripts/p2p_swarm_inference_v06_check.py --mode evidence-import --hf-model-id distilgpt2 --json`. `local-smoke` starts a real local P2P-lite daemon, announces a Coordinator plus stage0/stage1 Miner capabilities, verifies P2P route selection, runs real tiny-GPT generation when optional `[hf]` dependencies are installed, and runs local rescue rediscovery with short-lived victim peers followed by rescue peers. `external-existing --peer-bootstrap` verifies an already-running external P2P bootstrap catalog and only verifies live generation when `--verify-generate --admin-token` are provided. `kaggle-auto` is side-effectful: it starts temporary public p2pd/Coordinator processes, pushes private Kaggle stage0/stage1 CPU kernels, waits for P2P stage discovery, runs `generate --p2p`, deletes the kernels, cleans local private kernel payloads, and requires token rotation. Retained local evidence is `dist/p2p-swarm-inference-v06-local-smoke-refresh2/p2p_swarm_inference_v06.json`; retained external Kaggle proof is `dist/p2p-swarm-inference-v06-kaggle-auto-final/kaggle-auto/p2p_v06_kaggle_auto.json` with 2 generated tokens, 4 accepted ledger rows, distinct external stage Miners, deleted kernels, and `token_rotation_required`. Missing optional `[hf]` dependencies must produce `p2p_real_generate_hf_runtime_missing` or `host_hf_runtime_missing` rather than claiming tiny-GPT execution. This is P2P discovery/routing prototype evidence, not production NAT traversal, not decentralized security, not an economic system, not Hivemind/Petals parity, and not large-model throughput.
- Public P2P Swarm Inference v1.0 RC: `crowdtensor public-p2p-v1-rc` emits `public_p2p_swarm_inference_v1_rc_v1` through `scripts/public_p2p_swarm_inference_v1_rc_pack.py` and validates with `scripts/public_p2p_swarm_inference_v1_rc_check.py`. Preserve shared-secret HMAC peer identity, signed peer announcements, `p2pd --peer-secret --require-signed`, signed `serve --p2p` and `join --p2p` announcements, signed registry health counts, `public_p2p_swarm_inference_v1_rc_ready`, `signed_peer_announcement_ready`, `peer_identity_ready`, `peer_registry_health_ready`, `ttl_refresh_ready`, `local_signed_p2p_discovery_ready`, `serve_join_generate_p2p_commands_ready`, `public_p2p_v1_rc_model_metadata_ready`, `external_p2p_generate_verified`, `kaggle_kernels_deleted`, `p2p_v06_kaggle_private_artifacts_cleaned`, redacted `support_bundle.json`, and `PUBLIC_P2P_SWARM_INFERENCE_V1_RC.md`. The report exposes `p2p.hf_model_id` plus local/external/Kaggle model compatibility summaries; non-default `--hf-model-id` imports must carry matching signed local/external/Kaggle v0.6 `hf_model_id` metadata or emit `public_p2p_v1_rc_local_model_mismatch`, `public_p2p_v1_rc_external_model_mismatch`, or `public_p2p_v1_rc_kaggle_model_mismatch` and block readiness. The fresh retained signed Kaggle CPU proof is `dist/public-p2p-v1-rc-kaggle-auto-signed-r2/public_p2p_swarm_inference_v1_rc.json`; it verified external signed stage0/stage1 P2P discovery, tiny-GPT `generate --p2p` with 2 generated tokens, private Kaggle kernel deletion, local private payload cleanup, and `token_rotation_required`. Stage rescue remains signed local proof plus retained external requeue evidence, not a fresh signed Kaggle victim/rescue proof. This is a Petals-style public preview shape using HTTP P2P-lite and Coordinator lease/result fallback; it is not production Hivemind/Petals parity, libp2p, DHT, NAT traversal, decentralized security, an economic system, arbitrary prompt serving, or large-model throughput.
- Public Swarm Inference Preview v0.4: `crowdtensor preview-v04` emits `public_swarm_preview_v04_v1` through `scripts/public_swarm_preview_v04_pack.py` and is checked by `scripts/public_swarm_preview_v04_check.py`. Modes are `local-smoke`, `package`, and `evidence-import`. Preserve `public_swarm_preview_v04_ready`, `external_two_stage_generation_ready`, `multi_token_generation_ready`, `distinct_stage_miners`, `stage_assignment_valid`, `stage_latency_ready`, `throughput_summary_ready`, `memory_or_vram_summary_ready`, `external_stage_requeue_ready`, `tiny_gpt2_ci_fallback_ready`, `optional_distilgpt2_or_gpt2_strict_ready` when strict optional evidence is supplied, top-level `output_request`, `answer_scope.scope_state: no-local-answer`, `shareable_summary.answer_scope_state`, `redacted_evidence_ready`, `support_bundle.json`, and `PUBLIC_SWARM_PREVIEW_V04.md`. Retained ready evidence is `dist/public-swarm-preview-v04-final/public_swarm_preview_v04.json`; the strict local CPU `distilgpt2` proof is `dist/public-swarm-preview-v04-distilgpt2-strict/public_swarm_preview_v04.json`. It imports retained external Live Preview RC stage0/stage1 kill/requeue/rescue evidence and retained GPU multi-token generation evidence. Treat it as shareable preview evidence, not a local answer transcript; public artifacts keep raw prompts, generated text, generated token ids, activations, credentials, leases, private env files, and runtime state redacted. It is Coordinator-backed, read-only, tiny/small-model scoped, not production Swarm Inference, not P2P/libp2p/DHT/NAT traversal, not Hivemind/Petals parity, and not large-model serving.
- Public Real-LLM Swarm Beta output scope: top-level `public_real_llm_swarm_beta_v1` JSON, Markdown, terminal summaries, and `support_bundle.json` preserve `output_request`, `answer_scope.scope_state: no-local-answer`, `shareable_summary.answer_scope_state`, `evidence_scope`, `gpu_status`, and `gpu_proof_next_step`; check artifacts and terminal summaries preserve `checked_runtime_provenance`, `checked_evidence_scope`, `checked_gpu_status`, and `checked_gpu_proof_next_step`. The Beta aggregate is shareable release evidence, not a local answer transcript; run `crowdtensor generate` in human mode to see local generated text while public artifacts keep raw prompts, generated text, generated token ids, activations, credentials, leases, and idempotency material out of reports. `evidence_scope` / `checked_evidence_scope` are the shortest scoped evidence answer to whether the proof is local CPU, retained evidence, or a fresh Kaggle GPU proof; `gpu_status` / `checked_gpu_status` are the direct CPU/GPU verdict lines; `gpu_proof_next_step` / `checked_gpu_proof_next_step` record the explicit optional CUDA smoke, Kaggle package, fresh Kaggle GPU proof, and cleanup/token-rotation requirements when fresh GPU proof is not verified; `checked_runtime_provenance` is the detailed check-side source/proof summary behind that scope. Only `fresh_kaggle_gpu_verified: true` / `fresh_kaggle_gpu=True` supports a fresh Kaggle GPU claim.
- Public Real-LLM Swarm Inference Beta v1: `crowdtensor public-real-llm-swarm-beta` emits `public_real_llm_swarm_beta_v1` through `scripts/public_real_llm_swarm_beta_pack.py`; `crowdtensor public-real-llm-swarm-beta check` is the official user-facing validation entry over `scripts/public_real_llm_swarm_beta_check.py` and emits `public_real_llm_swarm_beta_check_v1` with `cli_mode: check`, `review_summary`, `artifact_summary`, `operator_action`, `checked_runtime_provenance`, `checked_evidence_scope`, and safe output-scope fields. `check --beta-report <public_real_llm_swarm_beta.json>` validates an existing release/local-model-variant artifact and records `check_source: beta-report`; omitting `--beta-report` keeps the CI-safe fixture check path. Preserve modes `release`, `local-smoke`, `local-model-variant`, `package`, `evidence-import`, and `check`; the `release` path runs Product Beta, imports retained external two-stage real-LLM requeue evidence, fresh-runs real-P2P candidate local-smoke over retained external generation/requeue/runtime-smoke/batch-stream source reports, fresh-runs the Public Swarm v2 ordinary P2P user-path report plus a v2 real-P2P local route-hardening child, and uses its fresh `usable-v1-local/usable_swarm_inference.json` child report for top-level KV-cache readiness, runs optional CUDA fail-closed smoke, and writes redacted JSON/Markdown/support artifacts. `--prompt-texts` and `--stream-generation` now propagate into the Product Beta path; top-level reports preserve safe batch metadata with `public_real_llm_swarm_beta_batch_ready` / `public_swarm_generate_batch_ready`, safe stream progress with `public_real_llm_swarm_beta_stream_ready` / `public_swarm_generate_stream_ready`, Public Swarm v2 readiness with `public_real_llm_swarm_beta_public_swarm_v2_ready`, `public_real_llm_swarm_beta_p2p_user_path_ready`, `public_real_llm_swarm_beta_v2_real_p2p_local_ready`, `public_swarm_v2_real_p2p_local_ready`, `public_real_llm_swarm_beta_v2_batch_ready`, and `public_real_llm_swarm_beta_v2_stream_ready`, product model match readiness with `public_real_llm_swarm_beta_product_model_match_ready`, and persistent cache reuse with `public_real_llm_swarm_beta_kv_cache_ready` plus `public_real_llm_swarm_beta_kv_cache_model_match_ready` only when the requested evidence completes. Current local product-path evidence is `dist/goal-final-infer-public-real-llm-swarm-beta-local-batch-stream-16tok-fixed-20260602/public_real_llm_swarm_beta.json` with `ok: true`, `public_real_llm_swarm_beta_local_smoke_ready`, two prompt hashes/char counts, 16 generated tokens per request, 16 ordered safe stream events from `admin-session-stream`, monotonic complete stream progress, and CUDA fail-closed readiness. Current fresh local release evidence is `dist/goal-final-infer-public-real-llm-swarm-beta-release-fresh-v2-local-requeue-20260602/public_real_llm_swarm_beta.json`; it has `ok: true`, no `not_completed`, fresh Product Beta, release-local Petals-class P2P candidate local-smoke, local Public Swarm v2, fresh v2 `real-p2p-local/real_p2p_swarm_inference_core_rc.json` route-hardening with 16 generated tokens, and release-local Usable/KV-cache steps, v2 `generated_token_count: 16`, v2 local/external accepted rows of 32, v2 dual-stage KV-cache reuse with 15 hits per stage, v2 batch/stream readiness, retained external/GPU imports plus retained P2P source inputs, release-local `source_reports.p2p_report`, no generated runtime-private files in the final release tree, and `fresh_external_runtime_verified: false`. Current release-grade evidence-import proof is `dist/goal-final-infer-public-real-llm-swarm-beta-import-16tok-p2p-batch-stream-kv-cache-model-gated-v2-20260602/public_real_llm_swarm_beta.json`; it has `ok: true`, no `not_completed`, product batch/stream readiness, Public Swarm v2 16-token P2P user-path readiness, v2 local/external accepted rows of 32, v2 dual-stage KV-cache reuse with 15 hits per stage, v2 batch/stream readiness, product/external/P2P/v2/KV-cache model match readiness, external and P2P `generated_token_count: 16`, `external_generated_token_target_ready`, `p2p_generated_token_target_ready`, external stage requeue, P2P live requeue rescue, P2P-side batch/stream readiness, persistent dual-stage KV-cache reuse with 15 stage0 and 15 stage1 hits, and `victim_result_accepted: false`. Its retained external source is `dist/goal-final-infer-real-llm-internet-beta-import-16tok-gpu-summary-20260602/real_llm_internet_beta.json`, produced by `real_llm_internet_beta --mode evidence-import` from a retained CUDA generation summary plus retained requeue evidence. Its retained P2P source is `dist/goal-final-infer-petals-candidate-16tok-batch-stream-composed-20260602/petals_class_p2p_candidate.json`, composed from a 16-token external real-P2P generation proof, a live stage0 victim/rescue requeue proof, and a safe supplemental batch/stream report. Its retained v2 source is `dist/public-swarm-inference-v2/public_swarm_inference_v2.json`; its retained KV-cache source is `dist/goal-final-infer-usable-swarm-16tok-kv-cache-20260601/usable_swarm_inference.json`. `release` fresh-runs the local Product Beta, Petals-class P2P candidate local-smoke, local Public Swarm v2, and top-level Usable/KV-cache evidence; external/GPU sources and P2P candidate source reports remain retained imports unless separately refreshed. Petals-class P2P candidate imports must preserve safe Real P2P batch/stream summaries when present and surface `public_real_llm_swarm_beta_p2p_batch_ready`, `public_real_llm_swarm_beta_p2p_stream_ready`, and shared `public_swarm_generate_batch_ready` / `public_swarm_generate_stream_ready` readiness without exposing raw prompts, generated text, or token ids. Release requires imported external/P2P reports plus the fresh v2 child KV-cache report to meet the requested token target before `external_generated_token_target_ready`, `p2p_generated_token_target_ready`, `public_swarm_v2_16_token_generation_ready`, or `public_real_llm_swarm_beta_kv_cache_ready` can be counted; evidence-import applies the same target to imported external, P2P, v2, and KV-cache reports. Lower-token retained evidence should emit token-target, v2, or KV-cache diagnostics and block readiness. Ready reports must preserve `public_real_llm_swarm_beta_ready`, `cpu_default_ready`, `external_two_stage_ready`, `external_stage_requeue_ready`, `p2p_ready_product_beta`, `p2p_live_requeue_rescue_ready`, `p2p_victim_result_not_accepted`, `public_real_llm_swarm_beta_public_swarm_v2_ready`, `public_real_llm_swarm_beta_p2p_user_path_ready`, `public_real_llm_swarm_beta_product_model_match_ready`, `public_real_llm_swarm_beta_kv_cache_ready`, `public_real_llm_swarm_beta_kv_cache_model_match_ready`, `cuda_optional_fail_closed_ready`, `release_evidence_ready`, `public_real_llm_swarm_beta_private_artifacts_cleaned`, `public_leak_paths: []`, and the boundary that this is the current top-level installable inference Beta, not full Hivemind/Petals production parity, not Coordinator-free, not NAT traversal production, not arbitrary prompt serving, and not large-model throughput. Real-P2P candidate evidence is not release-grade unless its public-safe `live_requeue_summary` proves claim observation, victim kernel deletion, lease expiry, rescue acceptance, and `victim_result_accepted: false`. Non-default `--hf-model-id` releases must have product, external, P2P, v2, and KV-cache evidence with matching `hf_model_id`; mismatched retained tiny-GPT evidence should emit `product_model_mismatch`, `external_model_mismatch`, `p2p_model_mismatch`, `public_swarm_v2_model_mismatch`, or `kv_cache_model_mismatch` and block readiness.
- Public Real-LLM Swarm Beta local real-P2P requeue import: `dist/goal-final-infer-public-real-llm-swarm-beta-import-v2-local-requeue-batch-stream-20260602/public_real_llm_swarm_beta.json` is now the retained top-level import that includes the fresh local Public Swarm v2 real-P2P stage1 victim/rescue proof at `dist/goal-final-infer-public-swarm-v2-local-real-p2p-requeue-batch-stream-20260602/public_swarm_inference_v2.json`. It has `ok: true`, no `not_completed`, `public_real_llm_swarm_beta_v2_real_p2p_local_requeue_ready`, `public_swarm_v2_real_p2p_local_requeue_ready`, `real_p2p_local_stage_requeue_ready`, `stage_requeue_ready`, `accepted_result_after_requeue`, and `victim_result_accepted: false`. Future top-level release/import gates must keep this local requeue requirement separate from external/Kaggle requeue evidence and must not claim production P2P, NAT traversal, Coordinator-free execution, or large-model serving from it.
- Public Real-LLM Swarm Beta fresh local requeue release: `dist/goal-final-infer-public-real-llm-swarm-beta-release-fresh-v2-local-requeue-20260602/public_real_llm_swarm_beta.json` is the stronger retained local release proof. It fresh-runs Product Beta, Petals-class P2P candidate local-smoke, Public Swarm v2, the v2 local real-P2P stage1 victim/rescue child, release-local Usable/KV-cache evidence, and optional CUDA fail-closed smoke. It has `ok: true`, no `not_completed`, `max_new_tokens: 16`, `public_swarm_v2_real_p2p_local_requeue_ready`, `real_p2p_local_stage_requeue_ready`, 32 accepted v2 stage rows, v2 batch/stream readiness, 15 KV-cache hits per stage, `private_artifacts_cleaned: true`, and release-local `source_reports.public_swarm_v2_report`. This supersedes older local release evidence that proved route hardening without the fresh v2 local requeue child.
- Public Real-LLM Swarm Beta local model variant: `crowdtensor public-real-llm-swarm-beta local-model-variant --hf-model-id distilgpt2 --max-new-tokens 16 --stream-generation` is the current proof for a non-default local small Hugging Face model without release-grade external claims. Fresh retained evidence is `dist/goal-final-infer-local-model-variant-distilgpt2-clean-codes-v2-20260602/public_real_llm_swarm_beta.json`; it has `ok: true`, no `not_completed`, `public_real_llm_swarm_beta_local_model_variant_ready`, `public_real_llm_swarm_beta_local_model_variant_only`, `public_swarm_inference_v2_local_model_variant_ready`, `public_swarm_v2_external_validation_not_claimed`, `external_validation_not_claimed`, Product Beta readiness, v2 batch/stream readiness, v2 local real-P2P stage1 requeue, 15 KV-cache hits per stage, CUDA fail-closed readiness, and private artifact cleanup. It intentionally keeps `release_evidence_ready`, `external_two_stage_ready`, `external_stage_requeue_ready`, and P2P candidate claims false; use `release` or `evidence-import` for retained external/Kaggle/P2P release claims. The user-facing check entry is `crowdtensor public-real-llm-swarm-beta check --hf-model-id distilgpt2 --beta-report <public_real_llm_swarm_beta.json> --json`, which maps to the script-level local-model-variant check and must keep release/external claims false.
- Safe artifact cleanup: `crowdtensor clean-artifacts` emits `cleanup_report_v1`, defaults to dry-run, removes generated `__pycache__` / `.pyc` caches and old CrowdTensor temp directories only with `--apply`, keeps reports unless `--include-reports` is used, and does not delete state, source files, release artifacts, or private env material.
- Remote demo operator CLI: `crowdtensor remote-runbook` emits `remote_runbook_cli_v1` and wraps `scripts/remote_demo_runbook_pack.py`; `crowdtensor remote-acceptance` emits `remote_acceptance_cli_v1`, defaults to `--create-session`, wraps `scripts/remote_demo_acceptance_pack.py`, carries fixed `model_bundle_inference_scenario_v1` scenarios such as `route-baseline`, and applies token redaction to captured command output. These are controlled two-machine helpers, not production Swarm Inference and not P2P routing.
- Remote home-compute demo CLI: `crowdtensor remote-demo prepare`, `crowdtensor remote-demo doctor`, `crowdtensor remote-demo verify`, `crowdtensor remote-demo collect`, and `crowdtensor remote-demo clean` emit `remote_home_compute_demo_v1`, `remote_home_compute_doctor_v1`, `remote_home_compute_collect_v1`, and `remote_home_compute_cleanup_v1` through `scripts/remote_home_compute_demo_pack.py`. The prepare path creates `operator.private.env`, `miner.private.env`, the hashed registry, the public runbook, and `miner_join_pack_v1` artifacts (`miner_join.sh` and `MINER_JOIN.md`) for ordinary CPU Miner hosts; doctor checks local files, token presence, Coordinator reachability, task lane visibility, and optional accepted-result readiness; the default verify path creates a read-only `POST /admin/inference-sessions` task for `model_bundle_infer`, validates the `remote_python_model_bundle_infer` route, and summarizes `remote_compute_evidence_v1`, `remote_demo_observability_v1`, and Support Bundle artifacts; collect gathers evidence/support from an already running demo; clean defaults to dry-run and only removes private env/registry files with `--include-private`. `--workload external-llm` creates a read-only `external_llm_infer` session, validates `remote_python_external_llm_infer`, and summarizes `remote_external_llm_evidence_v1` plus `remote_external_llm_observability_v1` for deterministic `--mock` or explicit operator-owned `--llm-runtime-cmd` / `--llm-runtime-url` adapters. `scripts/remote_home_compute_demo_check.py` validates both local-loopback stand-ins across prepare, doctor, verify, collect, and clean. It is a controlled two-machine CPU demo, not production Swarm Inference, not P2P routing, not GPU pooling, and not public arbitrary prompt serving.
- Real two-machine CPU inference Beta aggregate check: `scripts/remote_two_machine_beta_check.py` emits `remote_two_machine_beta_check_v1` by running local loopback stand-ins for the Coordinator host and Miner host across `model-bundle` and `external-llm`. It requires `remote_two_machine_inference_ready`, `remote_two_machine_external_llm_ready`, and `remote_two_machine_beta_ready`, and supports the 15-minute two-machine CPU inference Beta docs. This is task-level remote CPU inference, not model sharding, not P2P, and real machines still need operator-provided TLS, VPN, tunnel, or trusted network.
- Kaggle Remote Miner Beta: `crowdtensor remote-demo prepare --target kaggle` emits normal `remote_home_compute_demo_v1` prepare evidence plus generated `miner_join_pack_v1`, `miner_join.sh`, `MINER_JOIN.md`, `kaggle_remote_miner.py`, and `kaggle_remote_miner.md`. Operators upload only `miner.private.env` and `kaggle_remote_miner.py` to Kaggle, keep `operator.private.env` on the operator host, and run Kaggle as an outbound CPU Miner. `scripts/kaggle_remote_miner_beta_check.py` emits `kaggle_remote_miner_beta_check_v1`, requires `kaggle_remote_miner_prepare_ready` and `kaggle_remote_miner_beta_ready`, and validates token redaction plus the same local loopback remote-demo protocol. Kaggle is a temporary external Miner target, not production infrastructure, not P2P, and not a GPU/TPU workload path.
- Kaggle Real Runtime Acceptance: `crowdtensor remote-demo kaggle-real` emits `kaggle_real_runtime_acceptance_v1` through `scripts/kaggle_real_runtime_acceptance_pack.py`. Prepare uses public host `24.199.118.54` and temporary HTTP by default to create a public Coordinator launch script, `operator.private.env`, `miner.private.env`, hashed registry, and a Kaggle-only upload package. Verify requires a live Kaggle CPU Notebook Miner and should report `kaggle_artifacts_ready`, `coordinator_public_ready`, `kaggle_miner_seen`, `kaggle_result_accepted`, and `kaggle_real_runtime_ready`; `scripts/kaggle_real_runtime_acceptance_check.py` only validates artifacts in CI. `--workload micro-llm-sharded --stage-mode split --decode-steps 3` prepares `kaggle-upload-stage0` and `kaggle-upload-stage1` for two Notebook Miners and should report `kaggle_micro_llm_stage0_seen`, `kaggle_micro_llm_stage1_seen`, `kaggle_micro_llm_stage_assignment_valid`, `stage_assignment_valid`, and `kaggle_micro_llm_sharded_ready` after live verification. `scripts/kaggle_micro_llm_live_package.py` builds private Kaggle dataset/script-kernel upload folders for this live path; default mode uses a private dataset, while `--inline-kernel-payload` embeds source and stage `miner.private.env` into private kernel source as a temporary fallback. The first artifact-backed live split proof completed against `24.199.118.54:9180` with two private Kaggle CPU script kernels and `micro_llm_artifact_v1`; retained evidence is `dist/kaggle-micro-llm-live/external-real/kaggle_real_runtime_acceptance.json` with `ok: true`, `artifact_loaded`, `micro_llm_artifact_ready`, stage0/stage1 seen, valid stage assignment, baseline match, decoded-token match, and `kaggle_micro_llm_sharded_ready`. Temporary Kaggle kernels/dataset were deleted after evidence collection. Preserve `token_rotation_required`, keep `operator.private.env` off Kaggle, never publish inline private kernel payloads, and keep the boundary CPU-only/read-only, not production, not P2P, not GPU/TPU workload execution, and not large-model sharding.
- Micro-LLM Live Two-Node RC: `crowdtensor micro-llm-live-rc` emits `micro_llm_live_rc_v1` through `scripts/micro_llm_live_rc_pack.py` and is checked by `scripts/micro_llm_live_rc_check.py`. `--mode local-generated` creates `kaggle-upload-stage0` and `kaggle-upload-stage1`, starts a local Coordinator plus two independent stage Miner processes from those generated packages, and should report `local_generated_stage_upload_standins_ready`, `micro_llm_live_rc_ready`, `kaggle_micro_llm_sharded_ready`, and `stage_assignment_valid` without claiming external runtime. With `--micro-llm-artifact`, the same RC must load the file-backed `micro_llm_artifact_v1` package and preserve `artifact_loaded` / `micro_llm_artifact_ready`. `--mode external-existing` verifies an already running public Coordinator plus two external stage Miners and only then may report `external_runtime_verified`. It is CPU-only, read-only toy two-stage micro-LLM evidence, not production Swarm Inference, not P2P, not GPU/TPU pooling, and not GGUF/llama.cpp or large-model sharding.
- Local multi-Miner scenario sweep: `scripts/multi_miner_scenario_sweep.py` and `scripts/multi_miner_scenario_sweep_check.py` emit `multi_miner_scenario_sweep_v1` and `multi_miner_scenario_sweep_observability_v1` by creating three read-only `POST /admin/inference-sessions` tasks for `route-baseline`, `gradient-safety`, and `mixed-prompts`, then running separate registry-backed Python Miner identities through `local_multi_miner_model_bundle_infer`. The check defaults to `--execution-mode concurrent`, starts all Miner processes together, verifies scenario matches, distinct Miner distribution, `lease_summary` one-result-per-task uniqueness, `process_summary`, read-only/redaction/hashed-registry safety, and `multi_miner_concurrent_ready`; `--failure-mode kill-after-claim` terminates one claimed Miner before upload, observes lease timeout requeue, requires a rescue Miner to complete the same `task_id`, and emits `multi_miner_requeue_ready`. Runtime acceptance can opt in with `--include-multi-miner-sweep` and `--include-multi-miner-requeue`. This is local lease-race and requeue evidence, not P2P routing, production throughput scaling, GPU pooling, or production Swarm Inference.
- Demo Manifest tooling: `scripts/demo_manifest_pack.py` and `scripts/demo_manifest_check.py` produce `demo_manifest_v1`, the current latest output artifact for local-loopback handoff. It indexes `runtime_matrix.json`, `remote_compute_evidence_v1`, `external_llm_evidence_v1`, `support_bundle`, and `remote_compute_observability_v1` summaries without claiming GPU, P2P, WebGPU, public prompt serving, or production Swarm Inference readiness.

## Explicit Non-Capabilities

Do not imply these are implemented:

- real Swarm Inference for production LLM serving
- real Swarm Training or LLM fine-tuning
- GPU pooling across home machines
- WebGPU model shard execution
- libp2p discovery or NAT traversal
- decentralized identity or public P2P routing
- reward, staking, payment, or token economics
- hardware attestation
- hardened public-internet security

The current model bundle, measurable multi-request model bundle inference, admin-created `POST /admin/inference-sessions` route with `schema=inference_session_request_v1`, `scripts/inference_session_client.py` with `schema=inference_session_client_v1`, optional `external_llm_infer_v1` adapter and `external_llm_evidence_v1` proof path, and micro LM workloads are dependency-free contract rehearsals, not real LLM or GPU throughput benchmarks. `real_llm_sharded_infer_v1` is the narrow exception that can load a real Hugging Face tiny GPT model when `[hf]` dependencies are installed; it defaults to `hf_transformers_cpu` and has an explicit `hf_transformers_cuda` path for CUDA tiny GPT evidence only. It is still fixed-prompt read-only evidence, not arbitrary prompt serving, not large-model serving, and not production Swarm Inference. `model_bundle_infer` is read-only and exposes only Coordinator-derived capped `request_trace` summaries instead of raw `inference_results`; admin-created sessions must be inspected through `GET /admin/results?task_id=<task_id>&workload_type=model_bundle_infer` and must remain read-only. `inference_session_client.py` is only a thin user-facing client over that API; `session_client_ready` means the existing CPU read-only result was accepted, not that arbitrary prompt serving exists. `external_llm_infer` is read-only, validates `external_llm_results`, records safe `completion_count`, `output_chars`, `adapter_kind`, and `model_id` summaries, supports deterministic mock, command, and OpenAI-compatible HTTP adapters, and must keep raw prompts, raw `output_text`, runtime URLs, and API keys out of public state and public evidence.

## Strategic Route

The near-term goal is to make CrowdTensor credible and useful for open-source users before making large model-scale claims.

Recommended sequence:

1. Keep the Alpha control plane reliable, testable, and well documented.
2. Keep README, ROADMAP, protocol docs, use cases, static site, and project memory synchronized.
3. Keep `scripts/runtime_matrix.py` as the first open-source user diagnostic so contributors can see CPU-only readiness, optional browser support, external LLM adapter configuration, optional NVIDIA CUDA tiny GPT split readiness, `matched_capabilities`, `missing_capabilities`, target and route `diagnosis_codes`, `operator_action`, top-level `diagnosis_summary`, `hardware_diagnosis_summary`, and the hardware/runtime matrix before running longer smoke tests.
4. Keep `crowdtensor release-ready` as the maintainer-facing publish gate, preserving `release_readiness_v1`, `scripts/release_readiness_pack.py`, `scripts/release_readiness_check.py`, `--allow-dirty`, `git_dirty`, release gate aggregation, `demo_manifest_v1`, and explicit not production boundaries.
5. Keep `scripts/onboarding_gate.py --quick` as the fresh clone install-and-run proof, preserving `onboarding_gate_v1`, clean virtualenv creation, `python -m pip install -e .[dev,hf]`, console script checks, `scripts/user_friendly_inference_frontdoor_check.py`, the real installed `crowdtensor infer --prompt-stdin --shareable-terminal` user smoke with `user_infer_smoke_validation_v1`, `crowdtensor local-proof`, `crowdtensor home-infer`, `crowdtensor llm-infer --mock`, `crowdtensor cpu-infer --mode local`, `crowdtensor release-ready --allow-dirty`, `/tmp` output defaults, prompt/output redaction, local CPU / no fresh Kaggle GPU verdicts, and explicit non-production Swarm Inference boundaries.
6. Keep `crowdtensor local-proof` as the shortest user-facing local proof, preserving `local_proof_summary_v1`, Doctor, runtime matrix, CPU-only read-only home-compute demo, Demo Manifest output, and explicit non-production Swarm Inference boundaries.
7. Keep `crowdtensor home-infer` as the shortest shareable local read-only inference proof, preserving `home_inference_cli_v1`, `home_compute_evidence_v1`, `model_bundle_infer`, fixed `model_bundle_inference_scenario_v1` scenarios, capped `request_trace`, `diagnosis_codes`, and explicit not production Swarm Inference boundaries.
8. Keep `crowdtensor llm-infer` as the shortest shareable operator-owned local LLM runtime proof, preserving `llm_inference_cli_v1`, `external_llm_evidence_v1`, fixed claim-time prompts, `external_llm_infer`, adapter summaries, read-only/redaction safety, and explicit not public arbitrary prompt serving boundaries.
9. Keep `crowdtensor cpu-infer` as the CPU inference Beta aggregate path, preserving `cpu_inference_beta_v1`, `scripts/cpu_inference_beta_pack.py`, `scripts/cpu_inference_beta_check.py`, `--mode local`, `--mode remote-loopback`, `--mode remote-existing`, CPU-only read-only semantics, token/runtime redaction, and explicit not production / not P2P boundaries.
9. Keep `crowdtensor cpu-infer --mode beta-rc` as the CPU Inference Beta RC aggregate path, preserving `cpu_inference_beta_rc_v1`, `scripts/cpu_inference_beta_rc_pack.py`, `scripts/cpu_inference_beta_rc_check.py`, local CPU inference, remote-loopback inference, Real two-machine CPU inference Beta rehearsal, Kaggle Remote Miner Beta artifacts, `miner_join_pack_v1`, optional `--kaggle-real-runtime-report` import of `kaggle_real_runtime_acceptance_v1`, `demo_manifest_v1`, `cpu_inference_beta_rc_ready`, `local_cpu_inference_ready`, `remote_loopback_ready`, `two_machine_rehearsal_ready`, `kaggle_remote_miner_artifacts_ready`, `miner_join_pack_ready`, `cpu_miner_beta_ready`, `real_runtime_evidence_ready` when imported, CPU-only/read-only semantics, and explicit not production / not P2P / not GPU/TPU workload boundaries.
10. Keep `crowdtensor shard-infer-beta` and `crowdtensor remote-demo --workload sharded-model-bundle` as the CPU Pipeline-Sharded Inference Beta path, preserving `remote_sharded_inference_beta_v1`, `scripts/remote_sharded_inference_beta_pack.py`, `scripts/remote_sharded_inference_beta_check.py`, `--mode remote-loopback`, `remote_python_sharded_model_bundle_infer`, `remote_sharded_inference_acceptance_v1`, `remote_sharded_inference_observability_v1`, `remote_two_machine_sharded_ready`, activation hashes, `baseline_match`, `remote_sharded_inference_ready`, `remote_sharded_loopback_ready`, `local_sharded_inference_ready`, `stage_requeue_ready`, CPU-only/read-only semantics, and explicit not production / not P2P / not real LLM sharding boundaries.
10. Keep `crowdtensor micro-llm-shard-infer-beta` and `crowdtensor remote-demo --workload micro-llm-sharded` as the Remote Micro-LLM Pipeline-Sharded Inference Beta path, preserving `remote_micro_llm_sharded_beta_v1`, `scripts/remote_micro_llm_sharded_beta_pack.py`, `scripts/remote_micro_llm_sharded_beta_check.py`, `--mode remote-loopback`, `--stage-mode split`, `--require-distinct-stage-miners`, `remote_python_micro_llm_sharded_infer`, `remote_micro_llm_sharded_acceptance_v1`, `remote_micro_llm_sharded_observability_v1`, `remote_two_machine_micro_llm_sharded_ready`, activation hashes, `decode_steps`, `baseline_match`, `decoded_tokens_match`, `remote_micro_llm_sharded_ready`, `remote_micro_llm_sharded_loopback_ready`, `local_micro_llm_sharded_inference_ready`, `micro_llm_sharded_stage0`, `micro_llm_sharded_stage1`, `distinct_stage_miners`, `stage_assignment_valid`, `stage_requeue_ready`, CPU-only/read-only semantics, and explicit not production / not P2P / not GGUF/llama.cpp boundaries.
10. Keep `micro_llm_artifact_v1` as the first file-backed tiny model package boundary, preserving `crowdtensor micro-llm-artifact`, `scripts/micro_llm_artifact_pack.py`, `scripts/micro_llm_artifact_check.py`, `--micro-llm-artifact` across local, remote-demo, Kaggle prepare/verify, and live RC paths, artifact hash/id/tokenizer propagation, `artifact_loaded`, `micro_llm_artifact_ready`, and explicit not Hugging Face / not GGUF / not llama.cpp / not large-model boundaries.
10. Keep Real Small-LLM Sharded Inference Beta as the first optional real-weight tiny GPT split proof, preserving `crowdtensor real-llm-shard-infer`, `crowdtensor real-llm-shard-infer-beta`, `crowdtensor remote-demo --workload real-llm-sharded`, `real_llm_sharded_cli_v1`, `real_llm_sharded_evidence_v1`, `remote_real_llm_sharded_beta_v1`, `remote_real_llm_sharded_acceptance_v1`, `remote_real_llm_sharded_observability_v1`, `remote_python_real_llm_sharded_infer`, `remote_two_machine_real_llm_sharded_ready`, `real_llm_sharded_infer_v1`, `real_llm_artifact_v1`, `hf_transformers_cpu`, `scripts/real_llm_sharded_inference_evidence_pack.py`, `scripts/remote_real_llm_sharded_beta_pack.py`, `scripts/remote_real_llm_sharded_beta_check.py`, optional `[hf]` dependency isolation, `hf_dependencies_missing`, `--enable-hf-tiny-gpt-runtime`, `--hf-cache-dir`, `--real-llm-stage-role`, `real_llm_sharded_stage0`, `real_llm_sharded_stage1`, `real_llm_sharded_both`, `real_llm_artifact_ready`, `activation_transport_ready`, `baseline_match`, `decoded_tokens_match`, `remote_real_llm_sharded_ready`, CPU-only/read-only semantics, redaction of raw prompts/hidden states/logits/activation payloads, and explicit not production / not P2P / not GPU/TPU / not GGUF/llama.cpp / not large-model boundaries. For the large-model path, `scripts/stage_selective_weight_loading_check.py` emits `stage_selective_weight_loading_check_v1` and proves public-safe Llama-like safetensors stage materialization plus stage-owned `state_dict` application, local synthetic stage-selective runtime execution, and `real_llm_stage_selective_hf_runtime_v1` over an HF-style model directory: each stage opens only its assigned shard files, loads/applies only assigned weight keys into `meta`-instantiated stage models, materializes required runtime buffers, runs a stage0 activation into stage1 decode, matches a baseline next token for local fixtures, and reports counts/byte totals/digests/hashes. The local stage-selective check is still not by itself a Kaggle/7B claim; the retained Kaggle T4 x2 `large_model_kaggle_validation_v1` report is the external 7B runtime proof.
10. Keep Real Small-LLM Sharded Inference Live RC as the generated two-stage acceptance wrapper, preserving `crowdtensor real-llm-live-rc`, `real_llm_live_rc_v1`, `scripts/real_llm_live_rc_pack.py`, `scripts/real_llm_live_rc_check.py`, `scripts/kaggle_real_llm_live_package.py`, `kaggle_real_llm_live_package_v1`, `local-generated`, `kaggle-generated`, `external-existing`, `kaggle-upload-real-llm-stage0`, `kaggle-upload-real-llm-stage1`, `local_generated_real_llm_stage_upload_standins_ready`, `external_runtime_verified`, `kaggle_real_llm_stage0_seen`, `kaggle_real_llm_stage1_seen`, `kaggle_real_llm_sharded_ready`, `real_llm_artifact_ready`, `launcher_syntax_valid`, `--enable-hf-tiny-gpt-runtime`, `--real-llm-stage-role`, CPU-only/read-only semantics, and explicit not production / not P2P / not large-model boundaries. The retained live evidence is `dist/real-llm-live-goal-external/real_llm_live_rc.json`.
11. Keep Real Internet Swarm Inference Alpha as the larger real-weight milestone wrapper, preserving `crowdtensor real-llm-internet-alpha`, `real_llm_internet_alpha_v1`, `scripts/real_llm_internet_alpha_pack.py`, `scripts/real_llm_internet_alpha_check.py`, modes `local-generated`, `package`, and `external-existing`, `real_llm_internet_alpha_ready`, `real_llm_stage_requeue_ready`, `stage_requeue_ready`, `real_llm_live_rc_ready`, `remote_real_llm_sharded_ready`, `real_llm_artifact_ready`, `activation_transport_ready`, `decoded_tokens_match`, `distinct_stage_miners`, `stage_assignment_valid`, `external_runtime_verified` only for external-existing success, `token_rotation_required`, CPU-only/read-only semantics, and explicit not production / not P2P / not large-model boundaries. The retained external proof is `dist/real-llm-internet-alpha-external/real_llm_internet_alpha.json`.
11. Keep Real Internet Swarm Inference Beta as the automated external milestone wrapper, preserving `crowdtensor real-llm-internet-beta`, `real_llm_internet_beta_v1`, `scripts/real_llm_internet_beta_pack.py`, `scripts/real_llm_internet_beta_check.py`, mode `kaggle-auto`, `evidence-import`, CPU-default Kaggle kernels, optional `--real-llm-backend hf_transformers_cuda` private Kaggle GPU kernels, Kaggle CUDA runtime pins, CPU Coordinator CUDA metadata-only scheduling, `coordinator_cuda_runtime_required: false`, `real_llm_internet_beta_ready`, `real_llm_internet_alpha_ready`, `external_runtime_verified`, `kaggle_kernels_deleted`, both Kaggle stages seen, decoded-token match, distinct stage Miners, valid stage assignment, `token_rotation_required`, read-only semantics, cleanup-backed lifecycle evidence, and explicit not production / not P2P / not GPU pooling / not large-model boundaries. `evidence-import` must require safe generated-token target evidence, imported backend/schema metadata, matching model metadata, cleanup evidence, and public-safe requeue details before it can emit `real_llm_internet_beta_evidence_import_ready`; retained import evidence is `dist/goal-final-infer-real-llm-internet-beta-import-16tok-gpu-summary-20260602/real_llm_internet_beta.json`.
11. Keep Swarm Inference Beta as the user-facing real tiny GPT two-machine operator wrapper, preserving `crowdtensor swarm-infer-beta`, `swarm_inference_beta_v1`, `scripts/swarm_inference_beta_pack.py`, `scripts/swarm_inference_beta_check.py`, side-effectful `swarm-infer-beta live` / `kaggle-auto`, `swarm_inference_beta_live_ready`, `real_llm_internet_beta_ready`, `kaggle_kernels_deleted`, `swarm_inference_beta_live_private_artifacts_cleaned`, `token_rotation_required`, `support_bundle.json`, default local live private artifact and raw runtime state cleanup, debugging-only `--keep-live-private-artifacts`, `swarm-infer-beta prepare`, `coordinator`, `miner --stage stage0`, `miner --stage stage1`, `verify`, `collect`, `clean`, `operator.private.env`, `miner.private.env`, `miner_registry.json`, stage join packs, `SWARM_INFERENCE_BETA.md`, `remote_real_llm_sharded_beta_v1` verification, optional import of `real_llm_internet_beta_v1` as `external_beta_evidence_imported`, `swarm_inference_beta_ready`, `two_machine_swarm_inference_ready`, `real_llm_split_route_ready`, `decoded_tokens_match`, `distinct_stage_miners`, `stage_assignment_valid`, CPU-only/read-only semantics, and explicit not production / not P2P / not large-model boundaries.
11. Keep Public Swarm Inference Alpha as the user-facing session wrapper, preserving `crowdtensor swarm-session`, `public_swarm_inference_alpha_v1`, `scripts/public_swarm_inference_alpha_pack.py`, `scripts/public_swarm_inference_alpha_check.py`, `live-kaggle`, `local-generated`, `public_swarm_inference_alpha_ready`, `public_swarm_session_ready`, `local_stage_requeue_ready`, `public_swarm_live_kaggle_ready`, `stage_requeue_ready`, `external_runtime_verified`, `kaggle_kernels_deleted`, `token_rotation_required`, default child debug artifact pruning, debugging-only `--keep-child-artifacts`, CPU-only/read-only semantics, and explicit not production / not P2P / not large-model boundaries.
11. Keep Public Swarm Inference Alpha RC as the release-candidate evidence-import layer, preserving `crowdtensor public-swarm-alpha-rc`, `public_swarm_inference_alpha_rc_v1`, `scripts/public_swarm_inference_alpha_rc_pack.py`, `scripts/public_swarm_inference_alpha_rc_check.py`, `evidence-import`, `local-smoke`, `public_swarm_inference_alpha_rc_ready`, `public_swarm_alpha_rc_evidence_imported`, `stage0_live_requeue_evidence_ready`, `stage1_live_requeue_evidence_ready`, `public_swarm_live_requeue_evidence_ready`, `public_swarm_alpha_private_artifacts_absent`, retained stage report paths, CPU-only/read-only semantics, and explicit not production / not P2P / not large-model boundaries.
11. Keep Public Swarm Live Preview RC as the broadest shareable live-preview wrapper, preserving `crowdtensor live-preview`, `public_swarm_live_preview_rc_v1`, `scripts/public_swarm_live_preview_rc_pack.py`, `scripts/public_swarm_live_preview_rc_check.py`, `live-preview local-smoke`, `live-preview package`, `live-preview live-kaggle`, `live-preview evidence-import`, `public_swarm_live_preview_rc_ready`, `public_swarm_live_preview_local_smoke_ready`, `public_swarm_live_preview_package_ready`, `public_swarm_live_preview_live_kaggle_ready`, `public_swarm_live_preview_evidence_import_ready`, `external_stage_requeue_ready`, `live_stage0_requeue_ready`, `live_stage1_requeue_ready`, `kaggle_kernels_deleted`, `private_artifacts_cleaned`, `token_rotation_required`, `gpu_generation_evidence_import_ready`, the retained stage0/stage1 RC proof paths under `dist/public-swarm-live-preview-rc-live-stage0-20260529043801-rc` and `dist/public-swarm-live-preview-rc-live-stage1-20260529044328-rc`, short Kaggle slug prefix `ct-live-preview`, CPU-only-by-default/read-only semantics, and explicit Coordinator-backed/not production/not libp2p/not DHT/not NAT traversal/not large-model boundaries.
11. Keep Public Swarm v0.1 Operator Preview as the top-level ordinary-user preview artifact, preserving `crowdtensor operator-preview`, `public_swarm_operator_preview_v1`, `scripts/public_swarm_operator_preview_pack.py`, `scripts/public_swarm_operator_preview_check.py`, `operator-preview local-smoke`, `operator-preview package`, `operator-preview live-kaggle`, `operator-preview evidence-import`, `public_swarm_operator_preview_ready`, `operator_preview_user_path_ready`, `operator_preview_local_smoke_ready`, `operator_preview_package_ready`, `operator_preview_live_kaggle_ready`, `operator_preview_evidence_import_ready`, `serve_join_generate_ready`, `miner_join_pack_ready`, `cpu_fallback_ready`, `live_preview_ready`, top-level `output_request`, `answer_scope.scope_state: no-local-answer`, `shareable_summary.answer_scope_state`, `support_bundle_ready`, `release_readiness_ready`, `gpu_generation_evidence_import_ready`, `developer_preview_degraded`, `operator_preview_cpu_fallback_user_path_ready`, `operator_preview_retained_evidence_ready`, `external_runtime_blocked`, CPU-only-by-default/read-only semantics, and explicit Coordinator-backed/not production/not libp2p/not DHT/not NAT traversal/not large-model boundaries. Treat the aggregate as shareable operator-path evidence, not a local answer transcript; public artifacts keep raw prompts, generated text, generated token ids, activations, credentials, leases, private env files, and runtime state redacted.
11. Keep Public Swarm Inference Beta as the ordinary user entrypoint, preserving `crowdtensor public-swarm-beta`, `public_swarm_inference_beta_v1`, `scripts/public_swarm_inference_beta_pack.py`, `scripts/public_swarm_inference_beta_check.py`, `public-swarm-beta product-beta`, `public_swarm_product_beta_ready`, `public_swarm_product_rc_ready`, `coordinator_product_surface_ready`, `session_protocol_ready`, `p2p_lite_discovery_ready`, `gpu_generation_evidence_import_ready`, `cpu_fallback_ready`, `local_cpu_inference_ready`, compatibility `public-swarm-beta local-loopback`, compatibility `public-swarm-beta evidence-import`, `two_stage_split_inference_ready`, `local_loopback_ready`, `decoded_tokens_match`, `distinct_stage_miners`, `stage_assignment_valid`, `public_swarm_beta_evidence_import_ready`, `external_live_evidence_imported`, `stage0_live_requeue_evidence_ready`, `stage1_live_requeue_evidence_ready`, `prepare`, `coordinator`, `miner`, `verify`, `collect`, `clean`, Coordinator-backed/read-only semantics, and explicit not production / not libp2p / not DHT / not NAT traversal / not large-model boundaries.
11. Keep the Real P2P Swarm Inference Core RC as the current networking layer, preserving `crowdtensor p2p-daemon`, `crowdtensor real-p2p-rc`, `crowdtensor.real_p2p`, `scripts/real_p2p_daemon.py`, `scripts/libp2p_node20_polyfill.mjs`, `scripts/libp2p_kad_daemon.mjs`, `scripts/libp2p_discovery_alpha_check.py`, `scripts/real_p2p_swarm_inference_core_rc_pack.py`, `scripts/real_p2p_swarm_inference_core_rc_check.py`, `real_p2p_provider_record_v1`, `real_p2p_provider_catalog_v1`, `real_p2p_route_lookup_v1`, signed provider records, TTL eviction, bootstrap sync hooks, NAT/relay diagnostics, `serve/join/generate --p2p --p2p-backend real`, and the `p2pd` / P2P-lite fallback. The `http-provider-store` backend remains the fallback; the `libp2p-kad` backend adds a Node libp2p sidecar with persistent peer identity, TCP/noise/yamux, bootstrap peers, provider-record stream sync, Kad peer-routing diagnostics, and Node 20 preload compatibility. Current retained libp2p evidence is `dist/real-p2p-libp2p-local-smoke-ready/real_p2p_swarm_inference_core_rc.json`, proving local two-stage tiny-GPT 2-token generation with distinct stage Miners and accepted ledger rows; `dist/real-p2p-libp2p-kaggle-runtime-smoke-20260531-r6/real_p2p_swarm_inference_core_rc.json`, proving Kaggle can install source, Node/libp2p, HF dependencies, import the CLI, and start a libp2p sidecar; and `dist/real-p2p-libp2p-kaggle-auto-20260531-r4/real_p2p_swarm_inference_core_rc.json`, proving external stage0/stage1 libp2p discovery plus 2-token tiny-GPT split generation with two private Kaggle CPU Miners. Preserve `external_libp2p_stage_discovery_ready`, `external_libp2p_generate_ready`, `hivemind_petals_class_alpha_ready`, `distinct_stage_miners`, `stage_assignment_valid`, `real_p2p_core_rc_model_metadata_ready`, and `token_rotation_required`; non-default `--hf-model-id` evidence imports must expose matching `hf_model_id` metadata or emit `real_p2p_core_rc_model_metadata_mismatch` and block readiness. Provider records are still transported over libp2p streams rather than a production DHT value-store. This is Hivemind/Petals-class Alpha evidence only: Coordinator-backed, read-only, tiny-model scoped, not Hivemind/Petals production parity, not NAT traversal/relay, not decentralized security, not an economic system, and not large-model throughput.
11. Real P2P live external verification note: `external-existing --verify-generate` must forward the requested `--hf-model-id`, bounded `--prompt-texts`, and `--stream-generation` into nested `crowdtensor generate`, report public-safe model/batch/stream summaries, and keep local/Kaggle generate commands passing the requested model id into the actual generation process. This keeps live checks and later retained imports model-consistent without exposing prompts or generated text.
11. Keep Public Real-LLM Swarm Inference Beta v1 as the top-level release aggregate for the current installable inference product, preserving `crowdtensor public-real-llm-swarm-beta`, `crowdtensor public-real-llm-swarm-beta check`, check `--beta-report` validation of existing `public_real_llm_swarm_beta.json` with `check_source: beta-report`, `public_real_llm_swarm_beta_v1`, `public_real_llm_swarm_beta_check_v1`, `scripts/public_real_llm_swarm_beta_pack.py`, `scripts/public_real_llm_swarm_beta_check.py`, modes `release`, `local-smoke`, `local-model-variant`, `package`, `evidence-import`, and `check`, bounded `--prompt-texts` and `--stream-generation` propagation into Product Beta, P2P candidate safe batch/stream import, fresh Petals-class P2P candidate local-smoke plus `--p2p-report` for evidence-import, fresh Public Swarm v2 release step plus `--public-swarm-v2-report` for evidence-import, release-local Usable/KV-cache evidence plus retained Usable import for evidence-import, `public_real_llm_swarm_beta_ready`, `public_real_llm_swarm_beta_local_model_variant_ready`, `public_real_llm_swarm_beta_local_model_variant_only`, `external_validation_not_claimed`, `cpu_default_ready`, `external_two_stage_ready`, `external_stage_requeue_ready`, `p2p_ready_product_beta`, `p2p_live_requeue_rescue_ready`, `p2p_victim_result_not_accepted`, `public_real_llm_swarm_beta_public_swarm_v2_ready`, `public_real_llm_swarm_beta_p2p_user_path_ready`, `public_real_llm_swarm_beta_v2_batch_ready`, `public_real_llm_swarm_beta_v2_stream_ready`, `public_swarm_inference_v2_ready`, `public_swarm_inference_v2_local_model_variant_ready`, `public_swarm_v2_external_validation_not_claimed`, `public_swarm_v2_16_token_generation_ready`, `public_swarm_v2_dual_stage_kv_cache_ready`, `public_swarm_v2_stage_requeue_rescue_ready`, `public_real_llm_swarm_beta_product_model_match_ready`, `public_real_llm_swarm_beta_kv_cache_ready`, `public_real_llm_swarm_beta_kv_cache_model_match_ready`, `cuda_optional_fail_closed_ready`, `release_evidence_ready`, `public_real_llm_swarm_beta_batch_ready`, `public_real_llm_swarm_beta_stream_ready`, `public_real_llm_swarm_beta_p2p_batch_ready`, `public_real_llm_swarm_beta_p2p_stream_ready`, `external_generated_token_target_ready`, `p2p_generated_token_target_ready`, `public_swarm_generate_batch_ready`, `public_swarm_generate_stream_ready`, redacted public reports, Support Bundle output, and final private-artifact cleanup. The current fresh local release evidence is `dist/goal-final-infer-public-real-llm-swarm-beta-release-fresh-v2-local-requeue-20260602/public_real_llm_swarm_beta.json`; the current local model variant proof is `dist/goal-final-infer-local-model-variant-distilgpt2-clean-codes-v2-20260602/public_real_llm_swarm_beta.json`; the retained complete 16-token evidence-import proof is `dist/goal-final-infer-public-real-llm-swarm-beta-import-16tok-p2p-batch-stream-kv-cache-model-gated-v2-20260602/public_real_llm_swarm_beta.json`. Keep the default external/P2P/v2/Usable retained paths aligned with the imported 16-token sources. The P2P import must preserve P2P-side batch/stream readiness and `live_requeue_summary` with victim claim observation, victim kernel deletion, lease expiry, rescue acceptance, and victim-result rejection; external, P2P, and Public Swarm v2 imports must meet the requested token target; the Usable import must preserve dual-stage KV-cache readiness and per-stage cache-hit counts. Non-default model runs must not reuse default tiny-GPT imports; preserve product/external/P2P/v2/KV-cache model compatibility checks and `product_model_mismatch` / `external_model_mismatch` / `p2p_model_mismatch` / `public_swarm_v2_model_mismatch` / `kv_cache_model_mismatch` diagnostics. `local-model-variant` is the explicit exception for local-only non-default small-model proof and must not emit `public_real_llm_swarm_beta_ready` or `release_evidence_ready`; `check --hf-model-id <non-default> --beta-report <public_real_llm_swarm_beta.json>` must validate that exception through the local-model-variant check path. This aggregate may import retained real external, P2P, Public Swarm v2, Usable KV-cache, and GPU evidence in evidence-import mode, but release must fresh-run local Product Beta, Petals-class P2P candidate local-smoke, local Public Swarm v2, and top-level Usable/KV-cache evidence and must not erase the boundary: Coordinator-backed, read-only by default, tiny/small-model scoped, not full Hivemind/Petals production parity, not Coordinator-free, not NAT traversal production, and not large-model serving.
11. Validate the non-default local small-model exception with `scripts/public_swarm_inference_v2_check.py --mode local-model-variant --hf-model-id distilgpt2` and the user-facing `crowdtensor public-real-llm-swarm-beta check --hf-model-id distilgpt2 --beta-report <public_real_llm_swarm_beta.json>`; these checks must keep external, Kaggle, GPU-success, P2P-candidate, and release-ready claims false.
11. Keep Public Swarm Product Beta as the ordinary user-facing path, preserving `crowdtensor public-swarm-product-beta`, `public_swarm_product_beta_v1`, `scripts/public_swarm_product_beta_pack.py`, `scripts/public_swarm_product_beta_check.py`, modes `local-loopback`, `package`, and `external-existing`, bounded `--prompt-texts` forwarding into Beta RC, optional `--stream-generation` forwarding into Beta RC, `public_swarm_product_beta_ready`, `public_swarm_product_beta_user_path_ready`, `serve_ready`, `stage0_join_ready`, `stage1_join_ready`, `generate_ready`, `public_swarm_generate_batch_ready`, `public_swarm_generate_stream_ready`, `public_swarm_generate_stream_endpoint_ready`, top-level `output_request`, `answer_scope.scope_state: no-local-answer`, `shareable_summary.answer_scope_state`, `support_bundle_ready`, `private_artifacts_cleaned`, `decoded_tokens_match`, `distinct_stage_miners`, `stage_assignment_valid`, `private_artifacts_local_only`, `miner_join_pack_ready`, `hf_dependencies_missing` when optional `[hf]` dependencies are absent, CPU-only-by-default/read-only semantics, and explicit Coordinator-backed/not production/not libp2p/not DHT/not NAT traversal/not large-model boundaries. Treat the aggregate as shareable product-path evidence, not a local answer transcript; human `crowdtensor generate` may show local generated text while public artifacts keep raw prompts, generated text, generated token ids, activations, credentials, leases, and idempotency material redacted.
11. Keep Product Swarm v0.3 MVP check as the direct product-command runtime proof, preserving `scripts/product_swarm_mvp_check.py`, `product_swarm_mvp_check_v1`, `crowdtensor serve --run`, `crowdtensor generate`, bounded `--prompt-texts` batch evidence, optional `--shareable-generate-terminal` real human `generate --prompt-stdin --shareable-terminal` validation, `crowdtensor join --stage stage0 --run`, `crowdtensor join --stage stage1 --run`, `product_swarm_mvp_ready`, `serve_join_generate_mvp_ready`, `local_two_stage_real_llm_ready`, `generated_token_count_ready`, `public_swarm_generate_batch_ready`, `product_swarm_mvp_batch_ready`, `shareable_generate_terminal_ready`, `product_swarm_mvp_shareable_generate_terminal_v1`, `answer_scope_state=shareable-terminal-redacted`, `terminal_answer_text_hidden`, `gpu_state=local-cpu-only`, `fresh_kaggle_gpu_verified=false`, `distinct_stage_miners`, `stage_assignment_valid`, `product_swarm_mvp_degraded_ready` when optional `[hf]` dependencies are absent, `--require-hf-runtime`, CPU default plus optional `--backend cuda`, read-only semantics, and explicit Coordinator-backed/not production/not P2P/not Hivemind-level/not large-model boundaries.
11. Keep Public Swarm Inference Preview v0.4 as the broadest current preview aggregate, preserving `crowdtensor preview-v04`, `public_swarm_preview_v04_v1`, `scripts/public_swarm_preview_v04_pack.py`, `scripts/public_swarm_preview_v04_check.py`, `local-smoke`, `package`, `evidence-import`, Product Swarm MVP import, retained Live Preview RC stage0/stage1 requeue import, retained GPU multi-token generation import, `public_swarm_preview_v04_ready`, `external_two_stage_generation_ready`, `multi_token_generation_ready`, `distinct_stage_miners`, `stage_assignment_valid`, `stage_latency_ready`, `throughput_summary_ready`, `memory_or_vram_summary_ready`, `external_stage_requeue_ready`, `tiny_gpt2_ci_fallback_ready`, optional `optional_distilgpt2_or_gpt2_strict_ready`, top-level `output_request`, `answer_scope.scope_state: no-local-answer`, `shareable_summary.answer_scope_state`, retained final evidence under `dist/public-swarm-preview-v04-final`, strict local `distilgpt2` evidence under `dist/public-swarm-preview-v04-distilgpt2-strict`, redacted evidence, Support Bundle, and explicit Coordinator-backed/not production/not P2P/not Hivemind-level/not large-model boundaries. Treat the aggregate as shareable preview evidence, not a local answer transcript; human `crowdtensor generate` may show local generated text while public artifacts keep raw prompts, generated text, generated token ids, activations, credentials, leases, and runtime state redacted.
11. Keep Public Swarm Developer Preview as the largest ordinary-user preview entrypoint, preserving `crowdtensor preview`, `public_swarm_developer_preview_v1`, `scripts/public_swarm_developer_preview_pack.py`, `scripts/public_swarm_developer_preview_check.py`, modes `local`, `package`, `external-existing`, and `evidence-import`, `developer_preview_ready`, `public_swarm_developer_preview_ready`, `local_two_stage_generation_ready`, `serve_join_generate_ready`, `product_beta_ready`, top-level `output_request`, `answer_scope.scope_state: no-local-answer`, `shareable_summary.answer_scope_state`, `support_bundle_ready`, `cpu_fallback_ready`, `local_cpu_inference_ready`, `gpu_generation_evidence_import_ready` when retained GPU evidence is present, and inherited `hf_dependencies_missing` when optional `[hf]` dependencies are absent. Treat the aggregate as shareable preview evidence, not a local answer transcript; human `crowdtensor generate` may show local generated text while public artifacts keep raw prompts, generated text, generated token ids, activations, credentials, leases, and idempotency material redacted. It is CPU-only by default, read-only, Coordinator-backed, not production Swarm Inference, not libp2p/DHT/NAT traversal, not Hivemind-level serving, and not large-model serving.
12. Keep Public Swarm Inference Beta RC as the release-candidate product aggregate, preserving `crowdtensor public-swarm-beta-rc`, `public_swarm_inference_beta_rc_v1`, `scripts/public_swarm_inference_beta_rc_pack.py`, `scripts/public_swarm_inference_beta_rc_check.py`, modes `local-loopback`, `package`, and `external-existing`, bounded `--prompt-texts` product generation, optional `--stream-generation` safe stream progress, `public_swarm_inference_beta_rc_ready`, `public_swarm_product_beta_ready`, `p2p_lite_route_ready`, `p2p_lite_discovery_ready`, `cpu_fallback_ready`, `serve_join_generate_loop_ready`, `remote_generate_session_ready`, `public_swarm_generate_ready`, `public_swarm_generate_batch_ready`, `public_swarm_generate_stream_ready`, `public_swarm_generate_stream_endpoint_ready`, `private_artifacts_local_only`, `miner_join_pack_ready`, `external_runtime_verified` only for external-existing, `hf_dependencies_missing` when optional `[hf]` dependencies are absent, CPU-only-by-default/read-only semantics, and explicit Coordinator-backed/not production/not libp2p/not DHT/not NAT traversal/not large-model boundaries.
11. Keep Public Swarm GPU Inference Beta as the optional CUDA overlay, preserving `crowdtensor public-swarm-gpu-beta`, `public_swarm_gpu_inference_beta_v1`, `scripts/public_swarm_gpu_inference_beta_pack.py`, `scripts/public_swarm_gpu_inference_beta_check.py`, `public-swarm-gpu-beta local-smoke`, `public-swarm-gpu-beta local-loopback`, `public-swarm-gpu-beta kaggle-package`, `public-swarm-gpu-beta kaggle-auto`, `public-swarm-gpu-beta evidence-import`, CPU Coordinator CUDA metadata-only scheduling, private Kaggle GPU stage kernels, Kaggle CUDA runtime pins, retained stage-local proof path `dist/public-swarm-gpu-beta-stage-local-live-20260528064520-shortslug/public_swarm_gpu_inference_beta_kaggle_auto.json`, retained historical pre-stage-local proof path `dist/public-swarm-gpu-beta-live-20260528-runtimepin/public_swarm_gpu_inference_beta_kaggle_auto.json`, `hf_transformers_cuda`, `real_llm_sharded_cuda_stage0`, `real_llm_sharded_cuda_stage1`, `real_llm_sharded_cuda_both`, `public_swarm_gpu_beta_smoke_ready`, `public_swarm_gpu_beta_ready`, `public_swarm_gpu_beta_kaggle_auto_ready`, `gpu_runtime_ready`, `cuda_runtime_available`, `hf_transformers_cuda_ready`, `gpu_stage0_ready`, `gpu_stage1_ready`, `stage_local_partition_ready`, `stage0_partition_loaded`, `stage1_partition_loaded`, `partition_parameter_split_valid`, `stage_gpu_memory_reduced`, `kaggle_gpu_package_ready`, `kaggle_kernels_deleted`, `token_rotation_required`, `external_gpu_runtime_verified`, top-level `output_request`, `answer_scope.scope_state: no-local-answer`, `shareable_summary.answer_scope_state`, read-only semantics, and explicit not production / not P2P / not GPU pooling marketplace / not large-model boundaries. Treat the aggregate as shareable CUDA readiness evidence, not a local answer transcript; public artifacts keep raw prompts, generated text, generated token ids, activations, credentials, leases, private env files, and runtime state redacted.
11. Keep `crowdtensor gpu-generate` as the GPU Sharded Generation Beta entrypoint, preserving `gpu_sharded_generation_beta_v1`, `gpu_sharded_generation_beta_cli_v1`, `scripts/gpu_sharded_generation_beta_pack.py`, `scripts/gpu_sharded_generation_beta_check.py`, `local-loopback`, `kaggle-auto`, `evidence-import`, `--max-new-tokens`, multi-step stage0/stage1 generation chaining, `generation_step`, `generated_token_count`, `generated_text_hash`, `generated_text_redacted`, `raw_generated_text_public: false`, top-level `output_request`, `answer_scope.scope_state: no-local-answer`, `shareable_summary.answer_scope_state`, `multi_token_generation_ready`, `gpu_sharded_generation_ready`, `gpu_loopback_generation_ready`, `gpu_multi_machine_generation_ready`, old single-token evidence rejection through `gpu_multi_token_generation_missing`, `hf_transformers_cuda`, `stage_local` partitioning, Kaggle cleanup evidence, retained proof `dist/gpu-sharded-generation-beta-kaggle-20260528095658/gpu_sharded_generation_beta_kaggle_auto.json`, RC manifest `dist/gpu-sharded-generation-beta-kaggle-20260528095658/gpu_sharded_generation_beta_rc_manifest.json`, and explicit tiny GPT Beta / not production / not Hivemind-level / not P2P / not GPU marketplace / not large-model boundaries. Treat the wrapper as shareable generation evidence, not a local answer transcript; public artifacts keep raw prompts, generated text, generated token ids, activations, credentials, leases, and runtime state redacted.
10. Keep `crowdtensor clean-artifacts` as the safe maintenance path for repeated agent runs, preserving `cleanup_report_v1`, dry-run default, `--apply`, `--include-reports`, and the rule that cleanup does not delete state or source files.
10. Keep `crowdtensor remote-runbook` and `crowdtensor remote-acceptance` as the operator-facing wrappers for the controlled two-machine path, preserving `remote_runbook_cli_v1`, `remote_acceptance_cli_v1`, fixed scenario propagation, token redaction, default `--create-session`, and explicit not production / not P2P boundaries.
11. Keep `crowdtensor remote-demo prepare`, `doctor`, `verify`, `collect`, and `clean` as the high-level controlled two-machine home-compute demo, preserving `remote_home_compute_demo_v1`, `remote_home_compute_doctor_v1`, `remote_home_compute_collect_v1`, `remote_home_compute_cleanup_v1`, `scripts/remote_home_compute_demo_pack.py`, `scripts/remote_home_compute_demo_check.py`, private `operator.private.env` / `miner.private.env`, `miner_join_pack_v1`, `miner_join.sh`, `MINER_JOIN.md`, dry-run cleanup defaults, `POST /admin/inference-sessions`, `model_bundle_infer`, `remote_python_model_bundle_infer`, `remote_compute_evidence_v1`, `remote_demo_observability_v1`, `--workload external-llm`, `external_llm_infer`, `remote_python_external_llm_infer`, `remote_external_llm_evidence_v1`, `remote_external_llm_observability_v1`, and explicit not production / not P2P / not public prompt-serving boundaries.
12. Keep `scripts/remote_two_machine_beta_check.py` as the Real two-machine CPU inference Beta aggregate rehearsal, preserving `remote_two_machine_beta_check_v1`, `remote_two_machine_inference_ready`, `remote_two_machine_external_llm_ready`, `remote_two_machine_beta_ready`, Coordinator host / Miner host documentation, and explicit not model sharding / not P2P boundaries.
13. Keep the Kaggle Remote Miner Beta as a controlled external temporary-Miner target, preserving `crowdtensor remote-demo prepare --target kaggle`, generated `kaggle_remote_miner.py`, `kaggle_remote_miner.md`, `kaggle_remote_miner_prepare_ready`, `scripts/kaggle_remote_miner_beta_check.py`, `kaggle_remote_miner_beta_check_v1`, `kaggle_remote_miner_beta_ready`, outbound-only Kaggle Miner semantics, and the rule that `operator.private.env` must not be uploaded to Kaggle. This remains CPU-only/read-only, not GPU/TPU workload execution, not production Swarm Inference, and not P2P.
14. Keep Kaggle Real Runtime Acceptance as the first live external runtime proof, preserving `crowdtensor remote-demo kaggle-real`, `kaggle_real_runtime_acceptance_v1`, `scripts/kaggle_real_runtime_acceptance_pack.py`, `scripts/kaggle_real_runtime_acceptance_check.py`, public host `24.199.118.54`, temporary HTTP boundary, `kaggle_artifacts_ready`, `coordinator_public_ready`, `kaggle_miner_seen`, `kaggle_result_accepted`, `kaggle_real_runtime_ready`, `token_rotation_required`, and the rule that `operator.private.env` must not be uploaded to Kaggle. Also preserve the micro split artifact path: `micro-llm-sharded`, `kaggle-upload-stage0`, `kaggle-upload-stage1`, `kaggle_micro_llm_stage0_seen`, `kaggle_micro_llm_stage1_seen`, `kaggle_micro_llm_stage_assignment_valid`, `stage_assignment_valid`, and `kaggle_micro_llm_sharded_ready`. CI artifact checks are not live Kaggle proof. This remains CPU-only/read-only, not production, not P2P, not GPU/TPU workload execution, and not large-model sharding.
12. Keep expanding `scripts/home_compute_demo.py` as the useful home-compute demo that feels close to Swarm Inference: it should pair `scripts/runtime_matrix.py` `hardware_targets` / `recommended_routes` capability matching and `route_decision` with the read-only multi-request `model_bundle_infer` session and stable `diagnosis_codes` such as `home_compute_ready` and `runtime_matrix_blocked` before larger artifacts or runtime adapters are added.
13. Keep `scripts/home_compute_evidence_pack.py` and `scripts/home_compute_evidence_check.py` as the safe, shareable `home_compute_evidence_v1` layer for public issue reports and demos, preserving `route_decision`, `matched_capabilities`, `diagnosis_codes`, and capped `request_trace` while redacting secret-shaped fields.
14. Keep `scripts/inference_session_client.py` and `scripts/inference_session_client_check.py` as the narrow user-facing client path for a running Coordinator, preserving `inference_session_client_v1`, `session_client_ready`, `POST /admin/inference-sessions`, `task_id` filtering, `model_bundle_infer`, read-only semantics, and `--skip-inference-session-client` acceptance control.
15. Keep `scripts/admin_inference_session_check.py` as the narrow service-shaped API acceptance path for `POST /admin/inference-sessions`, preserving `inference_session_request_v1`, `task_id` filtering, `model_bundle_infer`, read-only semantics, and `--skip-admin-inference-session` acceptance control.
16. Keep `scripts/remote_compute_evidence_pack.py` and `scripts/remote_compute_evidence_check.py` as the safe, shareable `remote_compute_evidence_v1` layer for registry-backed remote-style Python Miner demos, preserving `remote_python_model_bundle_infer`, `remote_compute_observability_v1`, fixed `model_bundle_inference_scenario_v1` metadata and scenario match status, safe metrics, capped `request_trace`, and hashed registry status.
17. Keep `scripts/remote_demo_runbook_pack.py` and `scripts/remote_demo_runbook_check.py` as the safe two-machine `remote_demo_runbook_v1` path for controlled remote demos, preserving `operator.private.env`, `miner.private.env`, hashed registry setup, `model_bundle_infer`, `--scenario-id route-baseline`, and `remote_compute_evidence_pack.py --mode collect`.
18. Keep `scripts/remote_demo_acceptance_pack.py` and `scripts/remote_demo_acceptance_check.py` as the safe two-machine `remote_demo_acceptance_v1` path that can use `--create-session` to call `POST /admin/inference-sessions` with `scenario_id`, wait for the returned `task_id`, verify scenario match, then collect `remote_compute_evidence_v1`, `remote_demo_observability_v1`, `support_bundle`, and stable `diagnosis_codes` such as `coordinator_unreachable`, `observer_auth_failed`, `session_create_failed`, and `artifact_collection_failed`.
19. Keep `scripts/multi_miner_scenario_sweep.py` and `scripts/multi_miner_scenario_sweep_check.py` as the controlled local multi-Miner lease-race and failure-requeue proof, preserving `multi_miner_scenario_sweep_v1`, `multi_miner_scenario_sweep_observability_v1`, three fixed scenarios, distinct Miner identities, `local_multi_miner_model_bundle_infer`, concurrent mode by default in the check, `lease_summary`, `process_summary`, `requeue_summary`, read-only/redaction/hashed-registry safety, `multi_miner_concurrent_ready`, `multi_miner_requeue_ready`, and `--include-multi-miner-sweep` / `--include-multi-miner-requeue` opt-in coverage.
20. Keep `scripts/demo_manifest_pack.py` and `scripts/demo_manifest_check.py` as the latest output artifact entrypoint for local-loopback handoff. The manifest should combine runtime matrix, remote-compute evidence, deterministic mock external LLM evidence, and support bundle summaries while staying CPU-only and safe by default.
21. Keep `external_llm_infer_v1` and `external_llm_evidence_v1` as the narrow optional runtime adapter proof: deterministic `--enable-mock-llm-runtime` / `--mock` for CI, explicit `--llm-runtime-cmd` / `CROWDTENSOR_LLM_RUNTIME_CMD` for operator-owned local experiments, and `--llm-runtime-url` / `CROWDTENSOR_LLM_RUNTIME_URL` for OpenAI-compatible local servers.
22. Add hardware/runtime matrices for CPU, NVIDIA, AMD, Apple Silicon, browser, and remote container paths.
23. Introduce optional GPU/runtime adapters without making the control plane depend on one framework.
24. Expand browser-native participation from WebRTC/Worker probes toward WebGPU/WebAssembly only after tensor transfer and lifecycle limits are measured.
25. Add P2P/NAT routing after useful workloads and operator safety are proven.
26. Treat reputation and incentives as later protocol layers built on result validation and trust history.

## Engineering Principles

Network orchestration and tensor computation must stay decoupled. Task leasing, heartbeat, retries, validation, and operator state belong to the control plane; workload math belongs behind explicit workload contracts.

CPU-only deterministic smoke paths are strategic. They let CI, restricted Linux environments, and users without GPU access validate behavior. Optional accelerator paths must not remove or weaken these tests.

Protocol changes must be explicit and versioned. Current protocol names like `runtime_contract_v1` and `outer_optimizer_contract_v1` are compatibility boundaries.

Operator outputs must be safe by default. Support Bundle, `/metrics`, admin result ledger, and redacted state should avoid raw tokens, lease tokens, idempotency material, tensor deltas, raw registry secrets, and full raw state dumps.

Release quality matters. If a behavior becomes user-visible, update docs, tests, release gate expectations, changelog when appropriate, roadmap if strategic direction changes, and this project memory if the long-term story changes.

## Development Checks

Baseline checks:

```bash
python3 scripts/release_gate.py --json
python3 -m unittest tests.test_release_gate -v
python3 -m py_compile coordinator.py miner_cli.py scripts/*.py crowdtensor/*.py
python3 -m unittest discover -s tests -v
```

Runtime checks for Coordinator/Miner behavior:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8910 \
  --report /tmp/crowdtensor_acceptance.json
```

Browser checks when Playwright/Chromium are available:

```bash
python3 scripts/browser_acceptance_pack.py \
  --allow-skip \
  --base-port 9310 \
  --report /tmp/crowdtensor_browser_acceptance.json
```

Support Bundle for issue reports:

```bash
python3 scripts/support_bundle.py \
  --json-out /tmp/crowdtensor_support_bundle.json
```

Home-compute evidence pack for a safe, shareable route/session artifact:

```bash
python3 scripts/home_compute_evidence_pack.py \
  --port 8911 \
  --request-count 4 \
  --json-out /tmp/crowdtensor_home_evidence.json \
  --markdown-out /tmp/crowdtensor_home_evidence.md
```

Remote-compute evidence pack for a safe, shareable registry-backed Miner artifact:

```bash
python3 scripts/remote_compute_evidence_pack.py \
  --port 8912 \
  --request-count 4 \
  --json-out /tmp/crowdtensor_remote_evidence.json \
  --markdown-out /tmp/crowdtensor_remote_evidence.md
```

Safe two-machine remote demo runbook:

```bash
crowdtensor remote-runbook \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --output-dir dist/remote-demo \
  --json
```

Safe two-machine remote demo acceptance pack:

```bash
crowdtensor remote-acceptance \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" \
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \
  --create-session \
  --output-dir dist/remote-demo-acceptance \
  --json
```

Recommended two-machine remote home-compute demo:

```bash
crowdtensor remote-demo prepare \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --output-dir dist/remote-home-compute \
  --json

crowdtensor remote-demo doctor \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" \
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \
  --output-dir dist/remote-home-compute \
  --json

crowdtensor remote-demo verify \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" \
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \
  --output-dir dist/remote-home-compute \
  --json

crowdtensor remote-demo collect \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" \
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \
  --output-dir dist/remote-home-compute \
  --json
```

`crowdtensor remote-demo` emits `remote_home_compute_demo_v1`, `remote_home_compute_doctor_v1`, `remote_home_compute_collect_v1`, and `remote_home_compute_cleanup_v1`, wraps `scripts/remote_home_compute_demo_pack.py`, preserves private `operator.private.env` / `miner.private.env` handling, uses `POST /admin/inference-sessions`, and summarizes `remote_compute_evidence_v1` plus `remote_demo_observability_v1` for the `remote_python_model_bundle_infer` route. Validate this path with `scripts/remote_home_compute_demo_check.py`; it covers prepare, doctor, verify, collect, and clean. It remains not production Swarm Inference, not P2P, not GPU pooling, and not public arbitrary prompt serving.

Demo Manifest latest output artifact:

```bash
python3 scripts/demo_manifest_pack.py \
  --output-dir dist/demo-manifest \
  --port 8914 \
  --request-count 4
```

## Maintenance Rule

If future work changes project identity, target users, protocol boundaries, implemented capability, non-capability claims, roadmap priority, validation commands, or release workflow, update this document and `AGENTS.md` in the same change.

## Latest GPU+TPU+CPU 32B Serving Goal Status

- 2026-06-25 r5/r6-prep status: the Kaggle GPU batch-slot blocker cleared long
  enough to run a fresh private T4 x2 bridge kernel
  `xuyuhaosuyi/ct-gpu-tpu-cpu-bridge-82417684`, but the 4-token live serving
  goal remains incomplete. The r5 report is
  `dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260625-r5-gpu-free-4token/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`:
  CUDA stage0 was accepted once in the same Coordinator request
  (`accepted_stage_backends=["cuda"]`, `stage0=1`, one CUDA-to-TPU activation
  hash), while Web TPU stage1 did not submit (`stage1=0`, `stage2=0`,
  `generated_token_count=0`). The Web TPU service-manager path reached
  `service_manager_ready` and `session_startNew`, then timed out at
  `kernel_info_ready` / `service_manager_request_execute`. The temporary bridge
  kernel was deleted afterward. A follow-up Web UI restart attempt entered the
  TPU queue at `#10`
  (`dist/kaggle-web-tpu-restart-20260625-r2-start-session/kaggle_web_tpu_start_session_attempt.json`);
  a bounded active wait of about 15 minutes ended with `Session is starting`,
  service-manager `sessions=0`, and `kernels=0`
  (`dist/kaggle-web-tpu-restart-20260625-r3-active-wait/kaggle_web_tpu_active_wait.json`).
  The current serving aggregate is
  `dist/heterogeneous-32b-serving-20260625-r5-web-tpu-starting-after-r5/heterogeneous_32b_serving.json`;
  its checker passes but `live_external_runtime_verified=false` and
  `blocked_reason=qwen32b_tpu_stage_owned_loader_not_ready`. Code now includes
  a Web TPU proxy-token existing-kernel WebSocket fast path when Kaggle exposes
  one, keeps iframe direct-WS and service-manager fallbacks, and waits for the
  full Kaggle status/output/delete timeout budget before final report assembly.
  Focused validation passed: `python -m py_compile
  scripts/gpu_tpu_cpu_same_request_runtime_bridge_probe.py
  scripts/heterogeneous_32b_serving_pack.py` and `python -m pytest
  tests/test_gpu_tpu_cpu_same_request_runtime_bridge_probe.py
  tests/test_heterogeneous_32b_serving.py -q` (`31 passed`). Completion still
  requires a fresh post-fix bridge report with at least 4 generated tokens and
  CUDA, real JAX/TPU 32B stage-owned loader, and CPU accepted stage tasks in the
  same request. Do not delete unrelated Kaggle kernels such as
  `xuyuhaosuyi/v12100-live-middle-model-sft` without explicit permission.
- 2026-06-25 r6 successful 4-token live serving status: the current successful
  bridge report is
  `dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260625-r6-existing-session-4token/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`,
  with matching live proof
  `dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260625-r6-existing-session-4token/gpu_tpu_cpu_32b_same_request_live_proof.json`
  and serving aggregate
  `dist/heterogeneous-32b-serving-20260625-r6-live-4token-success/heterogeneous_32b_serving.json`.
  The Web TPU fix was verified first by
  `dist/web-tpu-service-manager-first-jax-smoke-20260625-r1/web_tpu_service_manager_first_jax_smoke.json`,
  which uses service-manager `session_connectTo_existing` and executes JAX on
  8 `TPU v5 lite` devices. The r6 bridge proves a same Coordinator request
  completed 4 generated tokens with `accepted_stage_backends=["cpu","cuda","jax_tpu"]`,
  `stage0=stage1=stage2=4`, `generated_token_count=4`,
  `same_request_runtime_bridge_verified=true`,
  `gpu_tpu_cpu_32b_same_request_verified=true`, and
  `same_request_32b_model_verified=true`. Stage0 used a temporary private
  Kaggle Tesla T4 x2 CUDA kernel; stage1 ran the real Web JAX/TPU
  `Qwen/Qwen2.5-32B-Instruct` stage-owned loader for layers 21-42 on 8 TPU v5
  lite devices (`executed_layer_count=21`, `loaded_execution_tensor_key_count=252`,
  `loaded_execution_tensor_gb=19.072947`, `stage_local_kv_cache_verified=true`);
  stage2 completed the CPU tail/verifier. Cleanup is verified:
  `kaggle_gpu_kernel_created=true`, `kaggle_gpu_kernel_deleted=true`,
  `private_gpu_package_removed=true`, no private bridge package remains, and
  the deleted temporary Kaggle kernel is no longer accessible. Current checks:
  `python -m py_compile scripts/gpu_tpu_cpu_same_request_runtime_bridge_probe.py
  scripts/heterogeneous_32b_serving_pack.py`, `python -m pytest
  tests/test_gpu_tpu_cpu_same_request_runtime_bridge_probe.py
  tests/test_heterogeneous_32b_serving.py -q` (`32 passed`), and
  `python scripts/heterogeneous_32b_serving_check.py --report
  dist/heterogeneous-32b-serving-20260625-r6-live-4token-success/heterogeneous_32b_serving.json
  --json` all pass. The serving aggregate now reports
  `live_external_runtime_verified=true`, `blocked_reason=""`,
  `heterogeneous_32b_serving_ready=true`, and
  `production_like_serving_path_ready=true`. Preserve the boundary: this
  completes the bounded Kaggle GPU + Web TPU + CPU 32B 4-token live serving
  validation, not production SLA/throughput, P2P/NAT traversal,
  billing/settlement, training, or an unbounded Kaggle service. The CUDA stage
  remains tied to prior retained 32B stage-owned CUDA evidence rather than
  reloading a full 32B CUDA stage inside this bridge.

## Latest Heterogeneous Capacity Frontier Status

- 2026-06-26 r3 capacity-frontier status: the current maximum-capacity evidence
  pack is
  `dist/heterogeneous-capacity-frontier-20260626-r3-72b-load-100b-partial/heterogeneous_capacity_frontier.json`,
  emitted by `scripts/heterogeneous_capacity_frontier_pack.py` and checked by
  `scripts/heterogeneous_capacity_frontier_check.py`. It imports the retained
  32B GPU+TPU+CPU r6 4-token same-request decode proof and a fresh full
  stage-owned loading proof for `Qwen/Qwen2.5-72B-Instruct-AWQ` at
  `dist/kaggle-72b-stage-owned-safetensors-probe-awq-live-r2-full10/kaggle_32b_stage_owned_safetensors_probe.json`.
  The 72B live loading proof ran 10 private Kaggle Tesla T4 x2 script kernels
  sequentially for stages 0-9, covering 2083/2083 safetensors keys across
  layers 0-80 plus embeddings/final head, with every stage reporting
  `stage_owned_quantized_32b_loading_ready=true`, `loads_only_stage_weight_keys=true`,
  no cross-stage keys, and loaded tensor sizes of about 5.73 GB for edge
  stages and 3.41 GB for middle stages. All 10 temporary kernels were deleted
  and the local private package directory was removed. It also imports a
  bounded 100B+ partial live loading probe at
  `dist/kaggle-100b-stage-owned-safetensors-probe-compressed-live-r1-stage8/kaggle-output/stage8/ct_32b_stage_owned_safetensors_stage8_report.json`:
  one private Kaggle Tesla T4 x2 kernel loaded stage8 of
  `cyankiwi/Solar-Open-100B-AWQ-4bit` (layers 40-44), 4684/4684 stage-owned
  compressed-tensors keys, 4.498133 GB materialized tensor bytes, no cross-stage
  keys, and temp cleanup; the remote kernel was deleted and the local private
  package was removed. This is partial single-stage live loading only, not full
  100B model coverage. The frontier report also performs real Hugging Face
  config/index/header preflight for 72B AWQ, 72B
  GPTQ, 72B full precision, 100B compressed-tensors, and 235B AWQ candidates;
  the largest preflight candidate is `QuixiAI/Qwen3-235B-A22B-AWQ` with
  `stage_owned_load_preflight_verified=true`, 94 layers, 25 safetensors files,
  and about 115.54 GB total indexed tensor bytes. Current conclusions are:
  `max_stage_owned_load_parameter_class=72b-awq`,
  `max_partial_stage_owned_load_parameter_class=100b-compressed`,
  `max_stage_owned_load_preflight_parameter_class=235b-awq`,
  `max_1token_decode_parameter_class=32b`,
  `max_multitoken_decode_parameter_class=32b`, and
  `max_gpu_tpu_cpu_same_request_parameter_class=32b`. A bounded Web TPU
  availability check for larger decode found no attached running TPU
  service-manager session/kernel, so the next bottlenecks are quantized
  JAX/TPU runtime adapter support, larger-than-32B same-request decode, current
  Web TPU runtime attachment, and live stage-owned loading beyond 72B. Preserve
  the boundary: 72B is currently a full stage-owned loading success, 100B is
  currently a partial single-stage live loading success, and neither is an
  activation/decode or same-request inference success; 235B is currently a
  metadata/header preflight success, not a live load or decode success.

## Latest Dense Three-Accelerator Qwen Frontier Status

- 2026-06-26 r8 dense/full-precision frontier status: future large-parameter
  LLM inference experiments should use dense BF16/FP16 HF/Qwen as the main
  path, not AWQ/GPTQ/4-bit/8-bit/GGUF quantized variants. The canonical current
  dense frontier artifact is
  `dist/three-accelerator-dense-qwen-frontier-20260626-r8-live-72b-stage-plan-retained-32b/three_accelerator_dense_qwen_frontier.json`,
  emitted by `scripts/three_accelerator_dense_qwen_frontier_pack.py` and
  checked by `scripts/three_accelerator_dense_qwen_frontier_check.py`. It
  imports the retained 32B GPU+TPU+CPU same-request 4-token bridge proof plus
  the retained 32B full-precision GPU+CPU fallback proof, imports the retained
  real Web TPU 32B full-stage loader/runtime evidence, and imports fresh
  Kaggle Models attach and stage-owned preflight probes up through dense 72B.
- `scripts/kaggle_dense_model_source_resolver.py` resolves dense official Qwen
  Kaggle Models attach sources under `https://www.kaggle.com/models`, including
  `qwen-lm/qwen2.5/Transformers/72b-instruct/1`,
  `qwen-lm/qwen2.5/Transformers/32b-instruct/1`,
  `qwen-lm/qwen2.5/Transformers/14b-instruct/1`, and
  `qwen-lm/qwen2.5/Transformers/7b-instruct/1`, with expected mounted paths
  under `/kaggle/input/models/{owner}/{model}/{framework-lower}/{instance}/{version}`,
  for example
  `/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1`. The older
  `/kaggle/input/qwen2.5/...` path was a wrong assumption and is now only a
  legacy fallback probe path. The live attach proof is
  `dist/kaggle-model-attach-probe-20260626-r3-7b-cpu-realpath/kaggle_model_attach_probe.json`:
  a private CPU-only Kaggle script kernel attached
  `qwen-lm/qwen2.5/Transformers/7b-instruct/1`, saw config/tokenizer/index plus
  4 safetensors files and 339 weight-index keys at the real mounted path, kept
  tensor values private, and then deleted the temporary kernel plus removed the
  local private package. This proves Kaggle Models attach for dense Qwen 7B and
  the runtime path shape. The later 32B attach proof is
  `dist/kaggle-model-attach-probe-20260626-r4-32b-cpu-realpath/kaggle_model_attach_probe.json`:
  it attached `qwen-lm/qwen2.5/Transformers/32b-instruct/1`, saw 17
  safetensors files and 771 weight-index keys, and cleaned up the temporary
  kernel/package. The current largest attach and stage-owned preflight proof is
  `dist/kaggle-model-attach-probe-20260626-r7-72b-cpu-stage-plan/kaggle_model_attach_probe.json`:
  it attached `qwen-lm/qwen2.5/Transformers/72b-instruct/1` at
  `/kaggle/input/models/qwen-lm/qwen2.5/transformers/72b-instruct/1`, saw
  config/tokenizer/index plus 37 safetensors files and 963 weight-index keys,
  kept tensor values private, verified a 10-stage public placement preflight
  (`cuda,cuda,cuda,cuda,jax_tpu,cpu,cpu,cpu,cpu,cpu`) with 963/963
  stage-owned keys present across all 37 safetensors files, about 145.412407 GB
  total planned logical tensor bytes, and about 16.534389 GB maximum single
  stage planned logical tensor bytes, then deleted the temporary CPU-only kernel
  and removed the local private package. This proves 72B dense model attach,
  safetensors header readability, stage-owned key/file assignment, and capacity
  preflight; it is still not 72B live weight loading, TPU execution, activation
  handoff, or inference.
- `scripts/qwen_dense_jax_tpu_stage_adapter_smoke.py` is the current
  HF/Qwen -> JAX/TPU stage adapter engineering path. The retained CPU-JAX unit
  evidence is
  `dist/qwen-dense-jax-stage-adapter-smoke-20260626-r2-cpu-jax/qwen_dense_jax_tpu_stage_adapter_smoke.json`:
  it exercises dense Qwen-like RMSNorm, RoPE, grouped-query causal attention,
  SwiGLU MLP, and stage-local KV-cache metadata with public-safe hashes/shapes
  only, and the PyTorch reference and JAX forward match. The dense frontier
  also imports the retained real Web TPU 32B full-stage loader evidence at
  `dist/kaggle-tpu-32b-stage-owned-loader-probe-web-live-20260623-r3-full-21-layer-real/kaggle_tpu_32b_stage_owned_loader_probe.json`,
  so `tpu_jax_qwen_stage_runtime_ready=true` for the retained 32B path.
- Current dense frontier conclusions are:
  `largest_dense_model_attempted=72b`,
  `largest_dense_model_attach_candidate=72b`,
  `largest_dense_model_attached=72b`,
  `largest_dense_model_stage_preflighted=72b`,
  `largest_dense_model_loaded=32b`,
  `largest_dense_model_1token_decoded=32b`,
  `all_three_accelerators_same_request_verified=true`,
  `generated_token_count=4`, `gpu_stage_runtime_ready=true`,
  `cpu_stage_runtime_ready=true`, `kaggle_model_attach_available=true`,
  `kaggle_model_attach_used=true`, `tpu_jax_qwen_stage_runtime_ready=true`,
  `same_request_dense_32b_success=true`, and
  `same_request_dense_frontier_success=false`. Treat 72B as the largest dense
  Kaggle Models attached and stage-preflighted model, not a loaded or decoded
  model. The largest verified dense loaded/decode class remains 32B. The
  unresolved blockers are
  `larger_dense_live_stage_load_not_verified_after_stage_preflight` and
  `larger_than_32b_dense_decode_not_verified`; the failure stage is now after
  72B attach plus stage-owned safetensors header/capacity preflight, at live 72B
  stage loading / placement / same-request decode. Do not claim 72B dense
  inference or production readiness until a future report proves real 72B stage
  loading, TPU dense stage execution for that larger placement, and same-request
  GPU+TPU+CPU activation handoff.
- 2026-06-26 r1 dense max-parameter-search status: use
  `dist/three-accelerator-dense-max-parameter-search-20260626-r1-72b-tpu-live-timeout-retained-32b/three_accelerator_dense_max_parameter_search.json`
  as the canonical current answer for "how large can the single-account
  dense GPU+TPU+CPU path actually infer today". It is emitted by
  `scripts/three_accelerator_dense_max_parameter_search_pack.py` and checked
  by `scripts/three_accelerator_dense_max_parameter_search_check.py`. The
  checked result is `max_successful_same_request_decode_parameter_class=32b`,
  `max_attempted_parameter_class=72b`, `max_attached_parameter_class=72b`,
  `max_stage_preflighted_parameter_class=72b`,
  `max_stage_loaded_parameter_class=32b`, and
  `max_tpu_executed_parameter_class=32b`, with
  `accepted_stage_backends=["cpu","cuda","jax_tpu"]`,
  `generated_token_count=4`, and public-safe artifacts only. It imports the
  retained 32B same-request bridge, the dense 72B Kaggle Models attach plus
  10-stage preflight proof, and the bounded 72B Web TPU live-load attempt at
  `dist/kaggle-tpu-72b-stage-owned-loader-probe-web-live-20260626-r3-stage32-40-one-layer-bridge-executor/kaggle_tpu_32b_stage_owned_loader_probe.json`.
  That 72B TPU attempt failed with `web_tpu_jupyter_execute_timeout` before
  proving 72B header read, tensor materialization, TPU device execution, or a
  real Qwen layer forward. This is a structured blocker and cleanup record, not
  evidence that 72B is impossible and not evidence that 72B inference works.
  The max-search checker intentionally rejects overclaims where 72B attach or
  stage preflight is promoted to stage-loaded, TPU-executed, or same-request
  decoded status without real loaded-key counts, TPU device evidence,
  layer-forward hashes, and same-request decode evidence.
- 2026-06-28 r4 dense max-parameter-search superseding status: use
  `dist/three-accelerator-dense-max-parameter-search-20260628-r4-web-tpu-ui-start-timeout-retained-32b/three_accelerator_dense_max_parameter_search.json`
  as the current canonical max-search artifact. It imports the fresh Web TPU
  execution-channel probe at
  `dist/kaggle-web-tpu-execution-channel-probe-20260628-r3-after-ui-start-wait/kaggle_web_tpu_execution_channel_probe.json`,
  emitted by `scripts/kaggle_web_tpu_execution_channel_probe.py` and checked by
  `scripts/kaggle_web_tpu_execution_channel_check.py`. This goal is still
  incomplete because 72B dense/full-precision GPU+TPU+CPU same-request 1-token
  decode has not succeeded. The latest recovery work tried Kaggle MCP
  `create_notebook_session`, which created interactive session ids but channel
  probes showed JAX CPU only (`jax_tpu_device_missing`). It then used
  authenticated Web UI automation to expand Session options, select
  `TPU v5e-8`, and force-click Start Session; the page entered
  `Session is starting...`, but a bounded 920-second wait at
  `dist/kaggle-web-tpu-active-wait-20260628-r1-queue11/kaggle_web_tpu_active_wait.json`
  still showed zero Jupyter sessions/kernels and no TPU runtime. The follow-up
  channel probe timed out before the first small JAX TPU cell, so the tiny
  Qwen-like cell was not attempted. The r4 result is valid and public-safe with
  `max_successful_same_request_decode_parameter_class=32b`,
  `max_attempted_parameter_class=72b`, `max_attached_parameter_class=72b`,
  `max_stage_preflighted_parameter_class=72b`,
  `max_stage_loaded_parameter_class=32b`,
  `max_tpu_executed_parameter_class=32b`,
  `failure_stage=web_tpu_channel_jupyter_execute`, and blockers including
  `web_tpu_execution_channel_not_ready`, `web_tpu_jupyter_execute_timeout`,
  `tiny_qwen_like_not_attempted_after_small_jax_failure`,
  `dense_72b_tpu_stage_load_and_forward_not_verified`, and
  `larger_than_32b_same_request_decode_not_verified`. This does not prove 72B
  is impossible; it proves the current Web TPU execution channel is not healthy
  enough to run a meaningful new 72B live-load attempt. Do not mark the 72B
  goal achieved from this blocker evidence. The next attempt should first get a
  current Web TPU runtime/channel probe passing both small JAX and tiny
  Qwen-like cells on TPU, then rerun 72B dense TPU stage-owned load/forward.
- 2026-06-28 r5 dense max-parameter-search superseding status: use
  `dist/three-accelerator-dense-max-parameter-search-20260628-r5-72b-stage-bridge-not-full-decode/three_accelerator_dense_max_parameter_search.json`
  as the current canonical max-search artifact. It is checked by
  `scripts/three_accelerator_dense_max_parameter_search_check.py --report ...
  --json` with no errors. r5 supersedes r4 because the Web TPU channel was
  restored and a bounded 72B stage path ran. The fresh channel proof is
  `dist/kaggle-web-tpu-execution-channel-probe-20260628-r4-runtime-started/kaggle_web_tpu_execution_channel_probe.json`:
  Web TPU v5e-8 attached with 8 `TPU v5 lite` devices, small JAX and tiny
  Qwen-like cells executed on TPU, and stage-local KV-cache metadata was
  verified. The standalone 72B TPU stage proof is
  `dist/kaggle-tpu-72b-stage-owned-loader-probe-web-live-20260628-r8-stage32-40-full8-1g-budget/kaggle_tpu_32b_stage_owned_loader_probe.json`:
  it executed `Qwen/Qwen2.5-72B-Instruct` layers 32-40 in the authenticated Web
  TPU runtime, verified 96 stage-owned keys, loaded about 13.078522 GB logical
  execution tensor bytes, ran all 8 assigned layers on 8 TPU devices, and kept
  raw tensors, activations, prompts, generated text, token ids, logits, and
  KV-cache tensors private. The same-request 72B stage bridge is
  `dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260628-r10-72b-tpu-stage-same-request/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`:
  one Coordinator request accepted CUDA, JAX/TPU, and CPU tasks
  (`stage0=stage1=stage2=1`, `accepted_stage_backends=["cpu","cuda","jax_tpu"]`,
  two activation handoff hashes, one generated-token hash), the TPU task ran
  the real 72B stage-owned loader for layers 32-40, and the temporary private
  Kaggle GPU kernel was deleted. This is not goal completion: r5 records
  `max_stage_loaded_parameter_class=72b`, `max_tpu_executed_parameter_class=72b`,
  and `same_request_72b_import.same_request_stage_decode_verified=true`, but it
  intentionally keeps `max_successful_same_request_decode_parameter_class=32b`,
  `same_request_72b_import.same_request_full_model_decode_verified=false`,
  `full_72b_weight_loading_public_claim=false`, and
  `failure_stage=dense_72b_stage_same_request_verified_but_full_model_decode_not_verified`.
  Do not mark a 72B goal achieved from r5. It proves a real 72B TPU
  middle-stage same-request bridge, not full all-layer 72B GPU+TPU+CPU decode,
  not 72B text-quality/parity, and not production serving. The remaining
  completion condition is a fresh public-safe report where real full-precision
  72B CUDA/CPU stage-owned execution plus the TPU stage complete a same-request
  1-token decode and set `gpu_tpu_cpu_72b_same_request_verified=true`.
- 2026-06-28 r6 dense max-parameter-search superseding status: use
  `dist/three-accelerator-dense-max-parameter-search-20260628-r6-full-72b-engineering-web-tpu-timeout/three_accelerator_dense_max_parameter_search.json`
  as the current canonical max-search artifact. It passes
  `scripts/three_accelerator_dense_max_parameter_search_check.py --report ...
  --json` with no errors and keeps
  `max_successful_same_request_decode_parameter_class=32b`,
  `max_stage_loaded_parameter_class=72b`, and
  `max_tpu_executed_parameter_class=72b`. The implementation moved forward:
  `scripts/kaggle_32b_full_heterogeneous_probe.py` now accepts configurable
  stage ranges/groups, can represent the full 10-stage 72B topology
  `gpu,gpu,gpu,gpu,web_tpu,cpu,cpu,cpu,cpu,cpu`, seeds Coordinator input ids
  without publishing token ids, and only emits
  `gpu_tpu_cpu_72b_same_request_verified`, `same_request_72b_full_model_verified`,
  and `full_72b_weight_loading_public_claim` when every stage completes.
  `scripts/kaggle_tpu_32b_stage_owned_loader_probe.py` now supports private
  input activation consumption and private output activation handoff for Web
  TPU stage execution while keeping `hidden_b64` out of public artifacts. The
  max-search pack/checker can import a successful full heterogeneous 72B report
  and only then raise max successful decode to 72B; stage-bridge-only reports
  remain blocked. A current Web TPU channel preflight at
  `dist/kaggle-web-tpu-execution-channel-probe-20260628-r6-short-current-status/kaggle_web_tpu_execution_channel_probe.json`
  failed with `web_tpu_jupyter_execute_timeout` before small JAX, so the full
  72B live run was not started in r6. This is not goal completion and not a
  blocked goal yet; it is meaningful engineering progress plus a current
  single-turn Web TPU execution-channel blocker. Resume by restoring the Web TPU
  channel, then run the full 10-stage 72B live probe and require
  `gpu_tpu_cpu_72b_same_request_verified=true` before marking achieved.
- 2026-06-28 r9 dense max-parameter-search superseding status: use
  `dist/three-accelerator-dense-max-parameter-search-20260628-r9-bounded-web-tpu-timeout/three_accelerator_dense_max_parameter_search.json`
  as the current canonical artifact. It passes
  `scripts/three_accelerator_dense_max_parameter_search_check.py --report ...
  --json` with no errors. It still records
  `max_successful_same_request_decode_parameter_class=32b`,
  `max_stage_loaded_parameter_class=72b`, and
  `max_tpu_executed_parameter_class=72b`; therefore the 72B goal is not
  achieved. Current Web TPU channel probes
  `dist/kaggle-web-tpu-execution-channel-probe-20260628-r7-current-status/kaggle_web_tpu_execution_channel_probe.json`
  and
  `dist/kaggle-web-tpu-execution-channel-probe-20260628-r8-force-new-session-30s/kaggle_web_tpu_execution_channel_probe.json`
  plus the latest
  `dist/kaggle-web-tpu-execution-channel-probe-20260628-r10-short-timeout-after-bounded-padding/kaggle_web_tpu_execution_channel_probe.json`
  are public-safe and checker-valid but all report
  `web_tpu_execution_channel_ready=false`, `small_jax_cell_ready=false`,
  `tpu_device_count=0`, and blocker `web_tpu_jupyter_execute_timeout`. The r8
  force-new-session attempt used the minimum 30-second Jupyter execute window.
  After r8, `scripts/gpu_tpu_cpu_same_request_runtime_bridge_probe.py` tightened
  `web_tpu_subprocess_timeout_seconds()` from `execute_timeout + 180s` to
  `execute_timeout + min(60s, max(10s, 25%))`; the r10 probe verified that a
  30-second channel probe now fails in a bounded short window while still
  showing the current Web TPU channel is not usable.
  The read-only UI state artifact
  `dist/kaggle-web-tpu-ui-state-probe-20260628-r2-current-readonly/kaggle_web_tpu_ui_state_probe.json`
  is the latest direct page-state evidence: `web_tpu_ui_runtime_ready=false`,
  `start_session_visible=true`, `jupyter_frame_visible=false`,
  `jupyter_session_count=0`, and `jupyter_kernel_count=0`. Current blocker is
  therefore not 72B weight loading; the Kaggle notebook is back at Draft
  Session/off and needs TPU Start Session plus runtime attachment before another
  meaningful 72B live run.
  The bounded restart attempt
  `dist/kaggle-web-tpu-start-wait-probe-20260628-r1-start-wait-15m/kaggle_web_tpu_start_wait_probe.json`
  selected visible `TPU v5e-8`, clicked Start Session, and waited 900 seconds.
  It reports `start_clicked=true` but `web_tpu_ui_runtime_ready=false`,
  `queue_visible=true`, `session_starting_text_visible=true`,
  `jupyter_frame_visible=false`, `jupyter_session_count=0`, and
  `jupyter_kernel_count=0`. This is only fresh Kaggle TPU allocation blocker
  evidence. Continue by waiting/retrying until a Jupyter runtime is visible,
  then rerun the Web TPU execution-channel probe before any full 72B same-request
  decode attempt.
  The follow-up read-only status
  `dist/kaggle-web-tpu-ui-state-probe-20260628-r3-after-start-wait-readonly/kaggle_web_tpu_ui_state_probe.json`
  still shows `web_tpu_ui_runtime_ready=false`,
  `session_starting_text_visible=true`, `jupyter_frame_visible=false`,
  `jupyter_session_count=0`, and `jupyter_kernel_count=0`. The longer continuation
  wait
  `dist/kaggle-web-tpu-start-wait-probe-20260628-r2-continue-wait-30m/kaggle_web_tpu_start_wait_probe.json`
  waited 1800 seconds and ended with the notebook still in Starting, no Jupyter
  frame/session/kernel, and no usable TPU runtime. It reports `start_clicked=false`
  because the page was already in Starting/disabled state; the earlier r1 wait
  is the evidence that Start Session had been clicked. This strengthens the
  current Kaggle TPU allocation blocker but still is not 72B inference evidence.
  The longer continuation wait
  `dist/kaggle-web-tpu-start-wait-probe-20260628-r3-continue-wait-60m/kaggle_web_tpu_start_wait_probe.json`
  waited 3600 seconds. Its final observation saw
  `session_started_text_visible=true`, but still
  `web_tpu_ui_runtime_ready=false`, `jupyter_frame_visible=false`,
  `jupyter_session_count=0`, and `jupyter_kernel_count=0`, so no usable runtime
  was proven. The immediate fresh page reload
  `dist/kaggle-web-tpu-ui-state-probe-20260628-r4-after-60m-wait-readonly/kaggle_web_tpu_ui_state_probe.json`
  returned to Draft Session/off with Start Session visible and no Jupyter
  frame/session/kernel. Current evidence is still a Kaggle Web TPU
  allocation/attach failure, not a 72B model failure and not completion.
- The 72B full-model gate is now stricter. `scripts/kaggle_32b_full_heterogeneous_probe.py`
  requires 72B stage ranges to cover Qwen 72B layers 0..80 contiguously before
  `ok`, `gpu_tpu_cpu_72b_same_request_verified`,
  `same_request_72b_full_model_verified`, or
  `full_72b_weight_loading_public_claim` can become true. It emits
  `full_72b_layer_coverage_verified` and
  `gpu_tpu_cpu_72b_full_topology_verified`; the max-search importer requires
  these before counting 72B as the max successful same-request decode. This
  prevents 72B TPU stage proof or 72B three-stage bridge proof from being
  mistaken for complete 72B inference. A memory-safer live plan can use 13
  contiguous stages
  `[0,6],[6,12],[12,18],[18,24],[24,32],[32,38],[38,44],[44,50],[50,56],[56,62],[62,68],[68,74],[74,80]`
  with 4 GPU stages, 1 Web TPU stage, and 8 CPU stages, but success still
  requires a real public-safe all-stage same-request 1-token decode. Next work:
  restore a Web TPU execution channel first, then run the full 72B live probe.
## Latest Colab TPU Fallback Evidence

Latest Colab TPU bridge progress on 2026-06-28: `scripts/gpu_tpu_cpu_same_request_runtime_bridge_probe.py`
now supports `--tpu-provider colab_cli` using the official Colab runtime proxy
session in `~/.config/colab-cli/sessions.json`. Supporting scripts:
`scripts/colab_tpu_session_probe.py`, `scripts/colab_tpu_runtime_stability_probe.py`,
`scripts/colab_tpu_coordinator_connectivity_probe.py`, and
`scripts/colab_tpu_qwen_stage_loader_probe.py`. The public-safe same-request
shape bridge proof is
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260628-r13-colab-tpu-shape-state-refreshed/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`:
one Coordinator request completed Kaggle CUDA, Colab JAX/TPU, and CPU stages
with `stage0=stage1=stage2=1`, accepted backends `["cpu","cuda","jax_tpu"]`,
and one Colab `TPU v5 lite` device. This is runtime plumbing, not model-weight
success.

The Colab TPU Qwen loader path progressed to real 32B stage-owned execution:
r14 ran Qwen 32B layers 21-22 for 1 layer and about 0.908GB logical tensor
bytes; r15 ran 4 layers and about 3.633GB; r16 is the current largest Colab TPU
same-request stage proof at
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260628-r16-colab-tpu-32b-eight-layer-loader/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`,
with Qwen 32B layers 21-29, `executed_layer_count=8`, 96 keys, about 7.266GB
logical tensor bytes, same-request CUDA/TPU/CPU completion, and private Kaggle
kernel cleanup. These do not claim full 32B or 72B success.

The 72B Colab attempt at
`dist/gpu-tpu-cpu-same-request-runtime-bridge-live-20260628-r17-colab-tpu-72b-four-layer-loader/gpu_tpu_cpu_same_request_runtime_bridge_probe.json`
failed before stage1 with a Colab HTTP/runtime session error; it is not a 72B
capacity failure. Current Colab session refresh evidence
`dist/colab-tpu-session-20260628-r4-refresh-v5e1-after-failure/colab_tpu_session_probe.json`
reports HTTP 503 / `colab_assignment_resource_unavailable`. Current blocker is
Colab TPU allocation/lifecycle availability. The active 72B goal is not
achieved: no dense/full-precision 72B GPU+TPU+CPU same-request 1-token decode
over all layers exists.

Latest reacquire attempts after r17: V5E1 stayed unavailable in
`dist/colab-tpu-session-20260628-r5-reacquire-v5e1/colab_tpu_session_probe.json`
and
`dist/colab-tpu-session-20260628-r7-reacquire-v5e1-after-clean-list/colab_tpu_session_probe.json`,
both HTTP 503 / `colab_assignment_resource_unavailable`. V6E1 was attempted in
`dist/colab-tpu-session-20260628-r6-reacquire-v6e1/colab_tpu_session_probe.json`
and returned HTTP 400. There were no active Colab assignments to clean up. A
Colab-only Qwen stage loader checker was added:
`scripts/colab_tpu_qwen_stage_loader_check.py` with tests in
`tests/test_colab_tpu_qwen_stage_loader_check.py`; it explicitly rejects any
stage-loader artifact that claims full 72B same-request success. Resume by
reacquiring Colab V5E1, refreshing kernel/session ids with the stability probe,
running `scripts/colab_tpu_qwen_stage_loader_probe.py` for Qwen 72B stage
layers, and only then retrying full GPU+TPU+CPU 72B same-request decode.

As of 2026-06-28, Kaggle Web TPU remains unstable, but Google Colab TPU is
currently reachable through the official Colab CLI backend path. The official
`google-colab-cli` is installed locally as `~/.local/bin/colab` and supports
`--tpu v5e1` / `--tpu v6e1`. Its own OAuth cache still needs a fresh Colab CLI
consent flow for the full `cloud-platform` / `drive.file` scope set, but the
existing local `~/.config/colab-exec/token.json` `colaboratory` OAuth cache was
enough to call Colab assignment endpoints directly and allocate a private TPU
V5E1 runtime. The retained local session is `ct-colab-tpu-v5e1` in
`~/.config/colab-cli/sessions.json`; that file is sensitive because it contains
runtime proxy tokens. A temporary V5E1 assignment from endpoint hash
`8bbdaaa63b8d88c7` was deleted with HTTP 204, leaving retained endpoint hash
`ad65a47c4b86a196`.

The public-safe retained evidence is
`dist/colab-tpu-runtime-stability-20260628-r1-v5e1/colab_tpu_runtime_stability_probe.json`,
emitted by `scripts/colab_tpu_runtime_stability_probe.py` and checked by
`scripts/colab_tpu_runtime_stability_check.py`. It reports
`colab_tpu_runtime_stably_acquired=true`, `runtime_proxy_connected=true`,
`rounds_requested=5`, `rounds_ready=5`, JAX `0.7.2`, one visible TPU device
`TPU_0(process=0,(0,0,0,0))`, and five successful BF16 1024x1024 matmul rounds
over about 120 seconds, without publishing proxy token, raw URL, or endpoint.
This proves a usable Colab TPU v5e1 runtime channel for future TPU adapter work;
it is not a GPU+TPU+CPU bridge proof, not a 32B/72B model proof, not production
serving, and not evidence that Kaggle TPU is fixed.
## 2026-07-12 Elastic Volunteer Training Runtime

- Canonical achieved artifact:
  `dist/training-qwen15b-elastic-live-20260712-r2-repacked-achieved/training_qwen15b_elastic_live_probe.json`.
- Strict validation:
  `PYTHONPATH=. python scripts/training_qwen15b_elastic_check.py --report dist/training-qwen15b-elastic-live-20260712-r2-repacked-achieved/training_qwen15b_elastic_live_probe.json --require-ready --json`.
- Durable semantics: a global optimizer step commits exactly once only after
  all four stage checkpoint archives reach the same epoch barrier. A Miner
  loss aborts the full uncommitted epoch; all stages restore the prior commit.
- Scheduler semantics: insufficient stage coverage automatically enters
  `paused_waiting_for_miners`; compatible Miner registration automatically
  wakes training. Session and assignment leases fence old workers.
- Live proof: old same-account T4x2 pair committed steps 1-4, both old Kernels
  were deleted, zero Miners were observed for 10 seconds, and a distinct new
  pair restored central step-4 state and committed steps 5-8. Commit history is
  exactly `[1,2,3,4,5,6,7,8]`.
- Final PEFT evaluation lowered validation loss from 2.6663523763 to
  2.4811929762. All four Kernels and private runtime resources were removed.
- Regression summary: 370 passed, 0 failed.
- Boundary: pinned Qwen2.5 1.5B LoRA and epoch-level rollback are proven;
  permissionless security, incentives/billing, arbitrary models/topologies,
  multi-account training, and in-flight microbatch migration are not.

## 2026-07-12 Elastic Volunteer Training Product Beta

- The ordinary-user and ordinary-Miner Elastic Volunteer Training Beta is
  achieved for pinned `Qwen/Qwen2.5-1.5B`, eight optimizer steps, four fixed
  stages, and two T4x2 product Miners at a time.
- Canonical live artifact:
  `dist/training-elastic-beta-live-20260712-r6-repacked-achieved/training_elastic_beta_live_probe.json`.
  File SHA-256:
  `a6476c610201877108a8630007c38675229dc33f9bb48fef4de64dbb9ddfae74`.
  Embedded content hash:
  `sha256:d58c1bd0304ddc126c3c5e5cccf39b0339c7dc1a18ff2370778ef99255d0ab78`.
- Strict validation command:
  `PYTHONPATH=. python scripts/training_elastic_beta_check.py --report dist/training-elastic-beta-live-20260712-r6-repacked-achieved/training_elastic_beta_live_probe.json --require-ready --json`.
  It passes with `elastic_training_beta_ready=true`, zero errors, ten required
  acceptance gates true, no blockers, and public-safe cleanup evidence.
- Public owner commands are `crowdtensor train create/status/cancel/export`,
  `crowdtensor train invite`, idempotent `crowdtensor train cleanup`, and
  `crowdtensor train serve --elastic-job <job>`. The authenticated owner HTTP
  surface includes status, cancel, export, and cleanup. The private contributor
  command is `crowdtensor-miner join --training --invite <invite.json>`.
- Owner status directly exposes `committed_step`, `online_miner_count`,
  `missing_stage_ids`, and stable `pause_reason` fields; detailed Miner,
  assignment, epoch, commit, and event identities remain hashed/public-safe.
- Private invite files carry the Miner token and Coordinator URL with mode
  0600; public reports carry neither. The Miner discovers two CUDA devices,
  advertises capability, automatically receives contiguous stage group
  `[0,1]` or `[2,3]`, restores central checkpoints, heartbeats its lease, and
  gracefully exits after a committed barrier when signaled or drained.
- The canonical live sequence used public owner create/status/export. Two old
  product Miners committed steps 1-4, gracefully left, and their two Kernels
  were deleted. The service observed ten zero-Miner paused states at step 4,
  then restarted from durable SQLite/rendezvous state. Two replacement product
  Miners with disjoint Kernel identities restored all four central step-4
  checkpoints and committed steps 5-8. The ledger is exactly
  `[1,2,3,4,5,6,7,8]` with no duplicate optimizer commit.
- Real training semantics are retained: all stages load only their assigned
  Qwen ranges, base weights remain frozen, LoRA gradient norms are positive,
  and the standard 392-tensor PEFT adapter covers layers 0..27. CPU/CUDA reload
  and evaluation lower held-out loss from 2.6663523763 to 2.4811929762.
- r6 repacks immutable r5 runtime evidence only to resolve the final redundant
  delete retry. It binds generation-level successful deletes to a post-cleanup
  account audit with zero active Kernels for the selected account and records
  `runtime_measurements_changed=false`. All four experiment Kernels, service,
  tunnel, private packages, rendezvous payloads, and private runtime were
  removed. Do not rerun this live gate merely to recreate evidence.
- Checkpoint storage is content-addressed and uses configurable committed-step
  retention. The default local path is live-verified. S3/MinIO support is
  implemented through the optional `storage` dependency with private
  environment credentials and is unit-tested for put/get/list/delete,
  deduplication, hash validation, and redaction. It has not been tested against
  an external S3 or MinIO service.
- Checkpoint submissions are HMAC-SHA256 bound to run, session, assignment,
  epoch, stage, and archive hash. Validation checks ownership, file/hash
  coverage, safetensors tensor names/dtypes/shapes/finite values and safe
  optimizer/GradScaler/RNG payloads. Online-Miner, per-session byte, and
  rejection quotas plus quarantine and stale fencing resist malformed,
  non-finite, replayed, unsigned, over-quota, and conflicting submissions.
  Heavy validation uses a thread pool so heartbeat processing remains live.
- These controls establish authenticated integrity and bounded abuse handling,
  not semantic correctness of a credentialed Miner's update. Permissionless
  Byzantine poisoning resistance, robust aggregation, secure aggregation, and
  Sybil resistance remain unimplemented and must not be claimed.
- Public cleanup fences active leases, aborts an uncommitted epoch, clears
  private rendezvous payloads, prunes unretained blobs, preserves existing
  exported adapters/public evidence and the committed retention window, and is
  idempotent. The controller does not own a volunteer host process or arbitrary
  external provider resource; provider-specific live probes remain responsible
  for deleting the exact temporary refs they create.
- Final regression artifact:
  `dist/training-elastic-beta-tests-20260712-r2-final/training_qwen15b_test_summary.json`.
  It records 380 passed, zero failed, 36 test files and 10 suites, including the
  9-test Product Beta controller/Miner/storage/security/checker suite. It
  explicitly records `s3_minio_storage_externally_live_tested=false` and
  `permissionless_byzantine_poisoning_resistance_verified=false`.
- Durable implementation anchors:
  `crowdtensor/elastic_training_beta.py`,
  `crowdtensor/elastic_training_miner.py`,
  `crowdtensor/elastic_training_runtime.py`,
  `crowdtensor/elastic_training_client.py`,
  `crowdtensor/elastic_checkpoint_storage.py`, public CLI routing in
  `crowdtensor/cli.py` and `miner_cli.py`, and evidence tools
  `scripts/training_elastic_beta_live_probe.py`,
  `scripts/training_elastic_beta_pack.py`, and
  `scripts/training_elastic_beta_check.py`.
- Boundary: this is a product Beta, not production GA or an SLA. It does not
  prove arbitrary model/topology admission, one-GPU Miners, 7B+ or
  full-parameter training, in-flight microbatch migration, multi-account
  training, permissionless security, incentives, rewards, or billing. The next
  core milestone should generalize model/topology admission and validate a 7B
  LoRA run with ordinary single-GPU Miners before marketplace work.

## 2026-07-13 Portable Kaggle TPU v5e-8 Operations

- The reusable, account-independent operating procedure is
  `docs/kaggle-tpu-v5e8-runbook.md`. It applies to any authorized Kaggle account
  with Interactive Notebook TPU access; each account supplies its own Notebook
  edit URL and private Playwright storage-state file.
- The preferred acquisition path is the authenticated Web Interactive Notebook
  UI, not Kaggle CLI/API/MCP accelerator allocation. Select `TPU v5e-8`, start
  the session, and keep the queue monitor alive while the visible queue or
  Starting Active Event exists.
- `scripts/kaggle_web_tpu_queue_monitor_probe.py` records queue position and
  Active Events. A valid queued report can exit 0 and is not readiness proof.
  Continue back-to-back bounded monitor windows until Running/runtime handoff,
  cancellation, authentication loss, or an explicit user stop.
- When the Active Event becomes Running, use
  `scripts/kaggle_web_tpu_active_event_probe.py` to reopen a detached event and
  `scripts/kaggle_web_tpu_execution_channel_probe.py
  --web-tpu-force-new-session` to attach through the Notebook iframe's
  `window.jupyterapp.serviceManager`.
- TPU readiness requires successful small-JAX and tiny-Qwen-like cells,
  `web_tpu_execution_channel_ready=true`, and eight JAX TPU devices. Queue
  position, a session id, `Session started`, or a Running event alone is not
  sufficient.
- The repeated successful JAX view of Kaggle `TPU v5e-8` is eight devices with
  `device_kind="TPU v5 lite"`. Retained successful channel evidence includes
  `dist/kaggle-web-tpu-execution-channel-probe-20260701-r2-force-new-session-after-running-event/kaggle_web_tpu_execution_channel_probe.json`
  and
  `dist/kaggle-web-tpu-execution-channel-probe-20260702-r13-after-r20-session-started-before-cpuowner-kagglecpu-fp4-bridge/kaggle_web_tpu_execution_channel_probe.json`.
- Model execution remains project-specific. Use JAX/Flax/Optax, MaxText, a
  safetensors-to-JAX adapter, or a separately verified torch-xla path. Do not
  assume PyTorch/CUDA code runs on TPU. Split long training into resumable
  segments and persist model, optimizer, RNG, data cursor, and step state.
- Browser storage state is a private account credential. Keep it mode 0600,
  never print/upload/commit it, and isolate accounts in separate browser
  contexts. Stop the underlying TPU Active Event after outputs and checkpoints
  are durable; shutting down a temporary Jupyter session does not release the
  TPU allocation.

## 2026-07-13 Unified CPU/GPU Heterogeneous Training Scheduler Beta

- The unified CPU/GPU Heterogeneous Training Scheduler Beta is achieved for
  pinned `Qwen/Qwen2.5-7B` revision
  `d149729398750b98c0af14eb82c78cfe92750796` LoRA. The canonical artifact is
  `dist/training-heterogeneous-beta-20260713-r3-r2-live-achieved/training_heterogeneous_beta.json`.
  File SHA-256 is
  `6c89c167d96e7548ffadba8454fdffa33cc5196b730db04c2d7a0f29e52e2884`;
  embedded content hash is
  `sha256:9f2cfe6a982a8d589df172ed38cceeda28baf153055b33b75c1cfdd6d1830a2f`.
- Strict validation command:
  `PYTHONPATH=. python scripts/training_heterogeneous_beta_check.py --report dist/training-heterogeneous-beta-20260713-r3-r2-live-achieved/training_heterogeneous_beta.json --require-ready --json`.
  It returns `ok=true`, `heterogeneous_training_beta_ready=true`, zero errors,
  and zero public-safety errors. All 12 acceptance gates are true. Do not rerun
  the live Kaggle gate merely to recreate evidence.
- The product path is manifest-driven and uses one Job, Coordinator, SQLite
  ledger, checkpoint store, and status surface for CPU and CUDA Miners. The
  default five stages are Qwen layers `[0,7)`, `[7,14)`, `[14,20)`, `[20,26)`,
  and `[26,28)`; stages 0-3 prefer CUDA and stage 4 is CPU-only. The manifest
  schema is `schemas/heterogeneous_training_manifest_v1.schema.json`.
- One visible GPU is sufficient for a Miner. Multi-GPU hosts can launch one
  independently fenced Miner process per device. Capability reports include
  CPU/RAM, per-GPU memory/compute/dtypes, measured stage latency, network
  metrics, current load, and stage capacity without publishing raw device
  names or private paths.
- Placement rejects memory/RAM/dtype violations before scoring. The score uses
  measured/estimated compute, adjacent-stage transfer, current load, device
  preference, memory reserve, and stage migration cost. Candidate audits,
  estimates, capacities, score components, reasons, and placement generations
  remain public-safe. Migration penalty prevents profile-only stage churn;
  OOM, persistent straggler, owner request, lease expiry, and Miner departure
  retain bounded rebalance paths.
- Tensor transport is chunked safetensors only. Envelopes bind manifest, step,
  microbatch, source/target stage, direction, generation, assignment hash,
  tensor metadata, checksums, TTL, limits, and retries. Chunk/payload integrity,
  dtype/device conversion, finite values, deduplication, idempotent replay,
  stale generation, timeout, and size limits are implemented and tested.
- Atomic checkpoints contain adapter, optimizer, LR scheduler, GradScaler, RNG,
  and manifest/progress state. Any stage failure aborts the whole speculative
  epoch. Generation/lease/assignment fencing rejects late commits. Same-nonce
  recovery can refresh dynamic capability without changing logical identity;
  incompatible re-registration remains rejected. Stage startup and operation
  waits renew leases, and stage operation timeout is separate from Kernel
  lifetime.
- Live topology: two private T4x2 Kernels supplied four one-GPU CUDA Miners and
  one private CPU Kernel supplied the final trainable stage. Steps 1-3
  committed, then a trainable GPU Miner was deliberately removed. The next
  epoch aborted and the Job paused/rebalanced at step 3. A different Miner
  restored the central checkpoint. Step 4 committed in placement generation 7
  and steps 5-6 in generation 8. The ledger is exactly `[1,2,3,4,5,6]`.
- Committed transport has 24 forward activation and 24 backward gradient
  messages, including six CUDA-to-CPU activations and six CPU-to-CUDA
  gradients. All 48 committed messages pass chunk/payload checksums. Four
  partial aborted-generation messages are retained only as fencing evidence.
- Every stage has finite gradient/loss evidence, real optimizer and scheduler
  updates, and changed LoRA hashes. The standard PEFT adapter has 392 tensors
  over layers 0..27. A separate CPU Kernel reloaded all five exported stages
  and completed a finite stagewise forward.
- Cleanup is complete: eight historical/current Kernel refs were deleted or
  verified previously deleted; 52 private tensor payloads, leases, temporary
  packages, credentials, checkpoints, Coordinator, tunnel, and private runtime
  were removed. `live_resources_left_running=false`.
- Final local regressions: 145 passed, 0 failed. This combines 66
  heterogeneous/shared Elastic tests and 79 legacy Qwen/CUDA training tests.
- Owner commands are `crowdtensor train create <job> --heterogeneous`, then
  `train serve/status/invite/export/cancel/cleanup`. Contributors use
  `crowdtensor-miner join --training --invite <invite.json>`. Public status
  exposes hashed capability, placement, resource, profile, generation,
  rebalance, commit, missing-stage, and pause evidence; credentials, URLs, raw
  data/token IDs, tensor values, and checkpoint values remain private.
- Boundary: achieved means pinned Qwen2.5-7B PEFT with five explicit stages,
  microbatch 1, sequence length 8, and epoch-level recovery. It does not mean
  full-parameter/TPU training, arbitrary architecture auto-partitioning,
  data-parallel replicas, mid-microbatch migration, permissionless poisoning
  resistance, secure aggregation, multi-account production operation,
  incentives/billing, production GA, SLA, or larger-model training. HF stage
  download throttling and free-provider tunnel/Kernel lifetimes remain
  availability/throughput limits.

## 2026-07-14 CPU/GPU/Kaggle TPU Heterogeneous Training Beta Blocker

- The pinned `Qwen/Qwen2.5-7B` CPU/CUDA/JAX-TPU LoRA engineering path is
  implemented, but the real three-accelerator Training Beta is not achieved.
  The canonical blocker is
  `dist/training-heterogeneous-tpu-beta-20260714-r10-window3-gpu-quota-blocker/training_heterogeneous_tpu_beta.json`
  with embedded content hash
  `sha256:d67387461bd7e2366d0158ac0275734ea190dcc837862e2c9ca2d61505fcfde2`.
  The default checker passes with zero errors and `ready=false`; the strict
  checker fails on 11 missing live gates. Do not overclaim this artifact.
- Manifest/schema v2 fixes stage 2 to Qwen layers `[14,20)` on one eight-device
  `jax_tpu` group. TPU placement includes HBM reserve, mesh/dtype fit, compile
  and steady-state cost, network/load cost, and migration cost. Existing v1
  CPU/GPU behavior and its achieved canonical checker remain unchanged.
- `crowdtensor/heterogeneous_jax_qwen_training.py` implements stage-selective
  real safetensors loading, named-mesh sharding, BF16 forward/backward, LoRA
  gradients, Adam updates, checkpoint/restart, and the shared stage subprocess
  protocol. Tensor transport accepts JAX arrays and remains chunked,
  checksummed safetensors. TPU checkpoint state is pickle-free and includes
  adapter, optimizer moments, scheduler JSON, JAX PRNG, manifest, and progress.
- Live work exposed TPU process ownership as a correctness requirement. The
  Kaggle wrapper originally called `jax.devices()` and held the runtime before
  launching the worker. It now probes in a disposable subprocess. The Miner
  capability path had the same parent/child issue and now discovers TPU in an
  isolated process that exits before the actual JAX stage process starts.
  Local scheduler/product/package regressions cover these paths. The fully
  fixed package subsequently received a real TPU allocation in window 3, but
  no training step ran because the selected GPU account had no weekly quota.
- The live orchestrator now fails fast when any Kernel becomes terminal before
  Coordinator completion, collects the terminal Kernel report before deleting
  peers, and retains public-safe queue/runtime observation summaries on early
  failure or timeout. This prevents another TPU failure from burning the full
  GPU/CPU Kernel lifetime merely to obtain diagnostics.
- Resource evidence is explicit: the acquisition limit was auditably extended
  from 2 to 3 without resetting old entries. Window 3 submission 1 queued for
  about 81 minutes and reached TPU `RUNNING`; live gate 3 then failed at the
  first CUDA push. Read-only quota evidence proves zero effective GPU seconds
  on that account, with refresh at `2026-07-18T00:00:00`, while three other
  authorized accounts had positive quota. Window 3 had two submissions left
  until `2026-07-15T00:38:31Z` at pack time, but the live-gate ledger is 3/3.
  The canonical pack imports only redacted ledger and quota summaries; account
  labels, credentials, private URLs, prompts, tensors, and checkpoints are not
  public.
- The live orchestrator now performs read-only GPU quota preflight before TPU
  acquisition and after TPU reaches `RUNNING`, and records actionable GPU/CPU
  push classifications. This prevents the same ordering failure from consuming
  a future authorized gate.
- Final regressions are 174 passed and zero failed with one conditional skip:
  76 passed plus one skip in the heterogeneous/shared suite, 79 legacy
  Qwen/CUDA tests, and 19 dedicated JAX/TPU tests. Cleanup is verified for all
  temporary Kernels, packages, tensor payloads, credentials, Coordinator, and
  tunnel; `live_resources_left_running=false`.
- Resume boundary: do not silently reset either ledger. Before window 3 expires,
  reuse requires explicit authorization for an additional live gate and a
  quota-positive GPU account. After expiration, it also requires a new bounded
  TPU acquisition window. Goal
  completion still requires strict evidence for providers `kaggle_cpu`,
  `kaggle_cuda`, and `kaggle_jax_tpu`, ledger `[1,2,3,4,5,6]`, eight TPU
  devices, changed TPU LoRA hashes, step-3 TPU replacement/restore,
  stale-generation rejection, cross-framework activation/gradient transfer,
  standard PEFT CPU reload, full cleanup, and strict checker `ready=true`.

## 2026-07-14 TPU Training Beta Gate 4 And Stage-2 Diagnostic

This section supersedes the preceding r10 quota-blocker status.

- The current canonical blocker is
  `dist/training-heterogeneous-tpu-beta-20260714-r12-live-gate4-terminal-diagnostic/training_heterogeneous_tpu_beta.json`.
  File SHA-256 is
  `85f80d7397c76a75abe5ab445927fab422eab222d949af5527965f6b8ff822fe`;
  embedded content hash is
  `sha256:3498a3ebca48a71096d60a593d239210136be251d1744c589f8b088137379a9d`.
  The default checker passes with zero errors and `ready=false`; strict fails
  11 checks. The TPU Training Beta is not achieved.
- Live gate 4 was explicitly authorized by extending the gate ledger 3 -> 4.
  GPU quota preflight passed before TPU acquisition and after TPU reached
  running. Both T4x2 Kernels, one CPU Kernel, and one TPU v5e-8 Kernel were
  accepted. Six Miners joined the same Coordinator and placement generation 1
  covered CUDA stages 0/1/3, JAX-TPU stage 2, and CPU stage 4.
- CUDA stage 3 and CPU stage 4 submitted real step-1 checkpoints at
  `18:57:35Z` and `18:57:33Z`. Stages 0, 1, and 2 did not submit step-1
  checkpoints, the commit ledger remained empty, and the TPU Kernel failed at
  `20:47:59Z`. The terminal TPU worker report was not available before gate-4
  cleanup, so no HF/JAX/OOM/timeout root cause may be claimed for that run.
- Window-3 submission 3 was used for a bounded stage-2 TPU diagnostic only;
  it did not consume a fifth live gate. The retained artifact is
  `dist/training-heterogeneous-tpu-stage-diagnostic-20260714-r1-window3-submission3/training_heterogeneous_tpu_stage_diagnostic_live_probe.json`
  with SHA-256
  `6addb3d3a2ad132cb1e2644c4303a09b7310c580fa0ed22eb77d288f5e92222f`.
- The diagnostic obtained JAX 0.10.2 on a real eight-device v5e-8 runtime. It
  range-loaded the pinned Qwen2.5-7B layers `[14,20)` from two source files in
  26 groups: 2,796,701,728 shard bytes and 72 real tensors. All eight devices
  participated in named-mesh parameter sharding. Forward compiled and ran;
  backward compilation failed with `IndivisibleError`, before optimizer update
  or checkpoint. Synthetic diagnostic boundary tensors mean this is loader and
  stage-runtime evidence, not full training acceptance.
- Source now supports a public-safe progress callback for every source file and
  range group. Terminal output collection accepts a probe-specific file
  pattern and retries before cleanup. Final reports preserve push, queue,
  runtime-progress, early-terminal, and collection summaries. This fixed the
  local collection bug that initially omitted the new diagnostic filename.
- A post-diagnostic local fix gives JIT outputs explicit sharding: forward and
  cross-device input-gradient boundaries are replicated; each LoRA gradient
  inherits its parameter layout, keeping rank-4 A gradients replicated and B
  gradients row-sharded. Local JAX 0.10.2 tests pass. This fix has not been
  rerun on TPU and must not be called live verified.
- Bounded resources are exhausted: acquisition windows 3/3, window-3
  submissions 3/3, and full live gates 4/4. The resume contract requires both
  a newly authorized bounded TPU acquisition window and explicit authorization
  for another full live gate. Never reset or bypass either ledger silently.
- Final regressions are 186 passed, zero failed, one conditional skip: 82
  heterogeneous/shared, 79 legacy Qwen/CUDA, and 25 dedicated JAX/TPU tests.
  Cleanup is complete for all remote Kernels, temporary private packages,
  tensor payloads, credentials, Coordinator, and tunnel;
  `live_resources_left_running=false`.
- Completion still requires one strict same-job artifact with real provider
  coverage `kaggle_cpu`, `kaggle_cuda`, and `kaggle_jax_tpu`, ledger exactly
  `[1,2,3,4,5,6]`, finite updates on all stages, changed TPU LoRA hashes,
  bidirectional GPU/TPU/CPU activation and gradient transport, TPU step-3
  replacement/restore, stale-generation rejection, standard PEFT CPU reload,
  complete cleanup, and checker `ready=true` with zero errors.

## 2026-07-15 TPU Training Beta Gate 6 Achieved

This section supersedes the r12 blocker above.

- The canonical achieved artifact is
  `dist/training-heterogeneous-tpu-beta-20260715-r15-gate6-live-achieved/training_heterogeneous_tpu_beta.json`.
  File SHA-256 is
  `689a89089f81d3d1ac7362629c1811632647bac74a605ce91429f09dad3341b8`;
  embedded content hash is
  `sha256:ee1e0d0b7a1b15909a46b67552fd4bda794eed8f09d595a2d627704a71bb1783`.
  Both default and `--require-ready` checker modes return `ok=true`,
  `error_count=0`, and `heterogeneous_training_tpu_beta_ready=true`.
- The successful source live report is
  `dist/training-heterogeneous-tpu-beta-20260715-r14-window4-submission2-live-gate6-stale-race-fix/training_heterogeneous_tpu_beta_live_probe.json`
  with SHA-256
  `03572217fc7d3a6a5d962b8d182f14a21233a82d190efd7e6ecf86cc9a81a620`.
  It used two T4x2 Kernels, one CPU Kernel, one eight-device TPU v5e-8 Kernel,
  six Miners, one Job, one Coordinator, and one manifest.
- Placement generation 1 assigned CUDA stages 0/1/3, JAX-TPU stage 2
  `[14,20)`, and CPU stage 4. The ledger committed exactly steps 1-6. Every
  stage produced finite gradients, real LoRA optimizer updates, changed
  adapter hashes, and valid checkpoint components.
- The old TPU Miner completed steps 1-3, left after step 3, and caused the
  expected pause. A replacement in the same retained TPU Kernel restored the
  step-3 JAX checkpoint and completed steps 4-6 under generation 2. An injected
  generation-1 result was rejected. GPU/TPU/CPU activation and backward-
  gradient routes are all present with checksummed chunked safetensors.
- JAX 0.10.2 executed real BF16 forward, backward, and Adam updates on all
  eight v5e devices. Named-mesh parameter sharding and explicit replicated
  boundary/gradient output layouts are verified. The TPU adapter hash changed,
  and checkpoint evidence includes Adam moments, scheduler state, JAX PRNG,
  manifest, and progress without pickle or GradScaler.
- Gate 6's worker status retained real compile measurements of about 39,030 ms
  and 36,674 ms, but its original top-level live summary read the absent step-
  result field and emitted zero. The builder now reads TPU runtime status. The
  canonical pack recovers the retained values only after binding both worker
  identities, stage 2, the manifest, eight-device mesh, and explicit sharding;
  the checker rejects zero, missing, mismatched, or modified imports. No
  runtime measurement was recomputed.
- The exported adapter has 392 standard PEFT tensors for layers 0-27 and was
  independently reloaded on CPU for a finite full stagewise forward. Final
  regressions are 256 passed, 2 conditional skips, 0 failed, and 4 existing
  warnings.
- Unlimited attempt count was explicitly authorized and persisted with hashed
  authorization records. Every acquisition attempt remains individually
  bounded to 12 hours and every full live gate to 6 hours; old attempts and
  finite-limit extensions remain in the ledgers. All remote Kernels, private
  packages, tensors, credentials, Coordinator, and tunnel were removed, with
  `live_resources_left_running=false`.
- This completes the fixed Qwen2.5-7B LoRA CPU/CUDA/Kaggle TPU Training Beta.
  Do not rerun the long Kaggle gate merely to regenerate evidence. Arbitrary
  models, full-parameter training, data parallelism, multiple TPU slices,
  permissionless trust, billing/rewards, and production SLA remain out of
  scope.

## 2026-07-17 CPU/GPU/Kaggle TPU Training Production RC Achieved

- The canonical Production RC is
  `dist/training-heterogeneous-production-rc-20260717-r5-path-redacted-final-ready/training_heterogeneous_production_rc.json`.
  File SHA-256 is
  `df1f1067ed67339445b9040ca4fd37988dc54d52c505c9e7b6a9680aa8aa2ddc`;
  embedded content hash is
  `sha256:2767155e863275a070c1fe22493d8b6410fdefab5684f131779af30f6b047fe5`.
  The strict checker returns `ok=true`, `error_count=0`, and
  `training_production_rc_ready=true`.
- The real r9 run committed exactly 400 contiguous steps and used Kaggle CPU,
  CUDA, and JAX TPU v5e-8 in one Job. Soak duration is 15,964.81 seconds and
  full-gate duration is 16,876.05 seconds. Checkpoints exist at every step,
  all updates are finite, all five LoRA stage hashes changed, and the
  392-tensor adapter independently reloads on CPU for a finite forward.
- The r9 replacement blocker was an evidence-attribution defect, not a
  training failure. CUDA stage 1 moved from `gpu_a` after step 70 to a
  different Miner in `gpu_b` at step 71 under generation 2. The process named
  `gpu_replacement` remained idle because the scheduler had already restored
  complete coverage. CPU restored step 90 and continued at 91; TPU restored
  step 100 and continued at 101. Coordinator restart at step 80 and stale-
  generation rejection also passed.
- The corrected public live artifact is
  `dist/training-heterogeneous-production-live-20260717-r10-r9-evidence-replay/training_heterogeneous_production_live_probe.json`.
  It hashes all four raw Kernel reports, records
  `live_run_reexecuted=false` and `training_measurements_changed=false`, and
  only re-derives replacement/effective-Kernel fields. Future Miner reports
  retain every `stage_process_ready_history` entry so later restarts cannot
  overwrite an earlier restore event.
- Replacement validation now requires distinct hashed identities, the same
  stage id, adjacent committed steps, a higher placement generation,
  checkpoint download evidence, and valid old/new checkpoint archives. A raw
  Kernel wrapper failure can be reclassified only when its workers exited
  cleanly and another Kernel proves the exact automatic takeover.
- Five baseline and five candidate windows pass the fixed-workload performance
  gate: throughput +32.85%, p50 latency +27.05%, and p95 improves 18.95%.
  Large checkpoint bodies use isolated HTTP connections while small messages
  use the persistent pool; inline/indexed transport and bounded telemetry
  sampling remain enabled.
- The production workflow probe verifies validate/plan/start/status/pause/
  resume/rebalance/stop/cleanup, health/readiness, authenticated metrics,
  bounded events, dry-run, idempotency, safe resume commands, and cleanup. The
  public resume command is `crowdtensor train resume <job-dir>`; the CLI and
  persisted status no longer serialize an absolute local job path. The
  fault probe verifies bounded retry, generation fencing, lease reclaim,
  worker quarantine/circuit breaking, mirrored checkpoint fallback/repair,
  Coordinator journal recovery, and cleanup retry.
- Final comprehensive regression is 301 passed, 2 conditional skips, 0 failed,
  and 5 warnings across 45 files. All remote Kernels, local Coordinator,
  tunnel, private packages, credentials, and tensor payloads are removed;
  `live_resources_left_running=false`. A post-RC read-only audit authenticated
  all four authorized Kaggle accounts and found zero matching
  `ct-training-production-*` resources and zero queued/running remainder.
- Scope remains pinned Qwen2.5-7B LoRA. Production RC is not GA or an SLA and
  does not prove arbitrary models, full-parameter or data-parallel training,
  in-flight microbatch migration, permissionless poisoning resistance, secure
  aggregation, incentives/billing, or larger-model training.
## Latest Volunteer Training Protocol Alpha Status

Current superseding training-direction status on 2026-07-17: the WAN-friendly
Volunteer Training Protocol Alpha goal is achieved at the local HTTP/real-PEFT
scope. The canonical RC is
`dist/volunteer-training-alpha-20260717-r1/volunteer_training_alpha_rc.json`.
Its file SHA-256 is
`c36646c0e367dfd805f5c9047b93de2b87c156a3c1aaa32a0621a58847efad41`
and its content hash is
`sha256:8cc4e84f647e0dc652729b7a5632f54f6d8b1ca21f0fd9233e85df03c81f72de`.
Its strict checker passes with zero errors and
`volunteer_training_protocol_alpha_ready=true`:
`PYTHONPATH=. python scripts/volunteer_training_alpha_check.py --report dist/volunteer-training-alpha-20260717-r1/volunteer_training_alpha_rc.json --require-ready --json`.

The retained probe uses four real PyTorch/Transformers/PEFT LoRA Cell updates,
eight optimizer steps and 256 tokens. Two distinct-Cell quorum rounds atomically
advance the canonical Adapter and DiLoCo/Local-SGD outer step from 0 to 2. One
Cell submits forked and non-finite updates that are rejected, disappears, has
its lease expire, and has the same work reassigned under generation 2. Its late
generation-1 update is stale-rejected; a duplicate accepted result is handled
idempotently. The public audit ledger hash chain verifies, and cleanup leaves no
service or external resource running.
The canonical public directory is 108K and contains no `.private` runtime.

The exact contributor path
`crowdtensor volunteer join <private-invite> --once` succeeds through the real
HTTP service with authenticated content-addressed downloads, lease heartbeat,
real local PEFT backward, and binary safetensors submission. Campaign creation,
serve, status, pause, resume, and cleanup commands are implemented. The same
eight-step/256-token centralized baseline is recorded; fixture results are a
comparison only and do not establish useful model quality.

Scope must remain precise: this proves a WAN-oriented protocol over loopback
HTTP on one physical host, not independent Internet machines. It assumes
invite-authenticated mostly cooperative Cells and does not provide Sybil or
permissionless Byzantine resistance, secure aggregation, poisoning defense,
privacy guarantees, arbitrary models, GA, or SLA. The next gate is at least two
independently administered Internet machines, object-storage artifact delivery,
immutable real campaign data/model import, longer churn/Coordinator recovery,
and one bounded reproducible public training campaign. See
`docs/volunteer-training-alpha.md`.
## 2026-07-18 Volunteer Training Public Founding Preview

The training-first public positioning and pre-publicity engineering work is
implemented. The public unit is a Campaign: immutable model/data revisions,
bounded PEFT/LoRA work, explicit evaluation and licensing, named maintainers,
moderation/rollback ownership, and a public-safe evidence contract. The
Dashboard is served by the Volunteer Coordinator at
`GET /v1/volunteer/dashboard` and reads aggregate data from
`GET /v1/volunteer/public-snapshot`; it never exposes Cell identifiers, work
or lease material, credentials, raw training rows, or tensor values.

The current canonical public launch artifact is
`dist/volunteer-training-public-launch-rc/volunteer_training_public_launch_rc.json`.
Its file SHA-256 is
`sha256:251a06f46ef62b9492eb39affb920d4ae9b340d3f655072970bb38b0d1d515ce` and
its embedded content hash is
`sha256:a89f353b8e2a0c4dc557826b9b6eed4331b49744caa4637ae8548b60cd124d7e`.
Its default checker is:

```bash
PYTHONPATH=. python scripts/volunteer_training_public_launch_check.py \
  --report dist/volunteer-training-public-launch-rc/volunteer_training_public_launch_rc.json \
  --json
```

It reports `founding_preview_ready=true` and
`formal_launch_ready=false`. The strict form uses `--require-formal` and must
remain failing until an independently administered physical multi-host report
proves at least two hosts, independent host/admin identities, a real network
route, and cleanup. Do not substitute same-host processes, Kaggle logical
nodes, queue screenshots, or local mocks for that gate.

The founding preview includes:

- `scripts/volunteer_training_public_demo.py`, which starts a real local HTTP
  Coordinator and two independent Cell subprocesses for one bounded round;
- `scripts/volunteer_training_public_demo_check.py`, which verifies the demo's
  hash, two completed Cells, public safety, progress, and cleanup;
- `scripts/volunteer_dashboard_visual_probe.py`, which uses Playwright against
  a real local service, captures desktop/mobile screenshots, checks canvas
  pixels, overflow, and layout order, then removes private runtime state;
- `scripts/volunteer_training_public_launch_pack.py` and
  `scripts/volunteer_training_public_launch_check.py`, which combine the
  proposal, demo, Dashboard visual evidence, and strict Operator Beta RC while
  preserving the formal-launch blocker;
- `docs/volunteer-campaign-governance.md` and
  `docs/volunteer-training-launch-kit.md`, which define the proposal gate,
  claim matrix, Reddit/LocalLLaMA wording, onboarding, and 60-90 second demo;
- `examples/volunteer-campaign/campaign-proposal.json` and
  `schemas/volunteer_campaign_proposal_v1.schema.json`, checked by
  `crowdtensor volunteer campaign validate-proposal`.

The demo uses a tiny local fixture and real PyTorch/Transformers PEFT math. It
proves the ordinary workflow and protocol mechanics, not useful model-quality
improvement, Internet-scale throughput, permissionless admission, Sybil
resistance, poisoning resistance, secure aggregation, or a production SLA.
The launch kit must use the terms `founding preview` and `same-host` and must
not advertise formal launch readiness.
