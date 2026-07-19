# CrowdTensor

CrowdTensor is an open-source project for **open campaigns for volunteer model
training**. A Campaign pins a model and dataset revision, admits ordinary
machines as bounded Cells, and advances a shared LoRA checkpoint one validated
round at a time. Contributors can give a short local update and leave; the
Coordinator keeps the accepted delta and the public evidence.

The current **founding preview** includes a real HTTP Coordinator, short-lived
per-Cell credentials, lease recovery, content-addressed update validation, a
public Campaign Dashboard, and a reproducible two-Cell demo. Start with the
[`governance guide`](docs/volunteer-campaign-governance.md), the
[`launch kit`](docs/volunteer-training-launch-kit.md), or the checked
[`Campaign proposal`](examples/volunteer-campaign/campaign-proposal.json).

The preview is deliberately honest about its boundary: retained evidence is
same-host or Kaggle logical multi-node validation. It is not yet proof of independent
physical multi-host execution, permissionless admission, Sybil resistance,
poisoning resistance, useful model-quality improvement, or a production SLA.
The launch checker reports `founding_preview_ready` separately from
`formal_launch_ready`; the latter requires a real external multi-host report.

Earlier inference and heterogeneous training milestones remain available below
for compatibility and engineering reference. CrowdTensor is an engineering
beta, not a full Hivemind/Petals replacement or a permissionless P2P network.

## Public Campaign Preview

Run the bounded founding demo locally. It starts one Coordinator, serves the
Dashboard, and runs two independent Cell processes for one round. It creates
no external GPU/TPU resources and removes its private runtime before returning:

```bash
PYTHONPATH=. python scripts/volunteer_training_public_demo.py \
  --output-dir dist/volunteer-training-public-demo --json
PYTHONPATH=. python scripts/volunteer_training_public_demo_check.py \
  --report dist/volunteer-training-public-demo/volunteer_training_public_demo.json \
  --require-verified --json
```

The live Dashboard is served at `/v1/volunteer/dashboard` by the volunteer
Coordinator. Its public snapshot contains aggregate progress and hashes, not
Cell identities, leases, credentials, raw data, or tensor values. The checked
visual references are [`desktop`](docs/assets/volunteer-dashboard-desktop.png)
and [`mobile`](docs/assets/volunteer-dashboard-mobile.png).

Before recruiting contributors, validate a Campaign proposal and read the
[`governance contract`](docs/volunteer-campaign-governance.md). The complete
Reddit/LocalLLaMA framing, claim matrix, and short demo shot list are in the
[`launch kit`](docs/volunteer-training-launch-kit.md).

The CPU/CUDA/JAX-TPU path now has a Training Production RC for the fixed
Qwen2.5-7B LoRA topology. One real Kaggle Job committed exactly 400 atomic
steps over about 4.43 hours using two T4x2 Kernels, one CPU Kernel, and one
eight-device TPU v5e-8 Kernel. It recovered CUDA, CPU, and TPU worker exits,
survived a Coordinator restart, rejected stale-generation work, passed a
five-window performance gate, exported and independently reloaded a 392-tensor
PEFT adapter, and cleaned all resources. The canonical artifact is
`dist/training-heterogeneous-production-rc-20260717-r5-path-redacted-final-ready/training_heterogeneous_production_rc.json`;
its strict checker returns `training_production_rc_ready=true` with zero
errors. See [`docs/training-foundation.md`](docs/training-foundation.md) for
the exact verified scope and non-goals.

For an account-independent procedure to acquire and verify a Kaggle Interactive
TPU v5e-8 runtime for JAX inference or checkpointed training experiments, see
[`docs/kaggle-tpu-v5e8-runbook.md`](docs/kaggle-tpu-v5e8-runbook.md). The
runbook uses the Web Notebook queue, Active Events, and JupyterLab
`serviceManager`; it does not treat Kaggle CLI/API session creation as TPU
readiness proof.

## Why It Matters

Most useful AI infrastructure assumes datacenter hardware, trusted operators, or
centralized serving. CrowdTensor explores a different path: ordinary machines
joining controlled, verifiable AI workloads one small step at a time.

The project focuses on the hard parts before the hype: routing, recovery,
validation, observability, artifact safety, and operator experience.

## What You Can Do Today

- Run a local end-to-end split inference proof with a real tiny GPT model.
- Use `crowdtensor infer "your prompt"` as the shortest user-facing inference
  path.
- Start a local discovery daemon, Coordinator, two stage Miners, and a user
  `generate` request.
- Validate stage assignment, distinct stage Miners, decoded-token correctness,
  KV cache reuse, and failure requeue evidence.
- Package controlled two-machine and Kaggle-style rehearsals for remote CPU
  Miners.
- Try optional CUDA tiny-model stage execution when the Miner host explicitly
  enables it.
- Produce redacted JSON/Markdown evidence and support bundles for debugging or
  release review.
- Submit, monitor, recover, export, cancel, and clean a pinned Qwen2.5 1.5B
  four-T4 Pipeline LoRA job through one persistent CLI/HTTP service.
- Pause Qwen stage training when complete Miner coverage disappears, then
  restore central checkpoints and continue on entirely new Miner sessions.
- Create, serve, inspect, export, cancel, and clean a durable elastic job with
  public owner commands, while contributors join from a private invite with
  `crowdtensor-miner join --training`.
- Run one manifest-driven training Job across pure CPU Miners and one-GPU or
  multi-GPU hosts, with memory-aware placement, cross-device
  activation/gradient transport, atomic checkpoints, and replacement recovery.
- Validate, plan, start, pause, resume, rebalance, monitor, stop, and clean the
  pinned CPU/GPU/TPU Training Production workflow through one durable CLI.
- Install model-family plugins, list their public provenance, and fail-closed
  conformance-check them through `crowdtensor community adapters`.
- Run the pinned Mistral 248M two-stage LoRA path across CPU and CUDA with
  checkpointed GPU worker replacement and standard PEFT export/reload.
- Operate or join a versioned Volunteer Training Protocol campaign in which
  Cells perform several local PEFT steps and upload one validated LoRA delta at
  a round boundary instead of streaming activations over the WAN.

## Quick Start

Use Python 3.11 or newer. The `[hf]` extra installs the optional Hugging Face
runtime used by the real tiny-model demos.

```bash
git clone https://github.com/Ffffffffchopin/CrowdTensor.git
cd CrowdTensor

python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,hf]'

crowdtensor --help
```

Preview the training flagship without allocating resources:

```bash
crowdtensor community init training-run
crowdtensor community validate training-run --json
crowdtensor community plan training-run --json
crowdtensor community coordinator up training-run --dry-run --json
crowdtensor community miner join training-run --dry-run --json
crowdtensor community train training-run --dry-run --json
```

The complete owner/Contributor flow is in
[`docs/community-quickstart.md`](docs/community-quickstart.md). Supported
models and refusal boundaries are in
[`docs/model-adapters.md`](docs/model-adapters.md).

Create the pinned SmolLM2-135M/WikiText Campaign, then contribute one work unit
through the ordinary HTTP/CLI boundary:

```bash
crowdtensor volunteer campaign import-smollm-wikitext campaign-dir \
  --target-rounds 3 --local-steps 1
crowdtensor volunteer serve campaign-dir --prepare-only

# After an operator serves the campaign and privately sends the mode-0600 invite:
crowdtensor volunteer join campaign-invite.json --once --device auto
```

See
[`docs/volunteer-training-internet-beta.md`](docs/volunteer-training-internet-beta.md)
for pinned provenance, strict evidence, deployment flow, recovery behavior,
and the external gate. The current RC uses real PEFT math in independent local
processes; it is not evidence of independent Internet machines, poisoning
resistance, or useful model quality.

For the current Operator deployment, credential policy, lifecycle, MinIO,
backup, monitoring, and exact claim boundaries, see
[`docs/volunteer-training-operator-beta.md`](docs/volunteer-training-operator-beta.md).
The shortest Operator path is:

```bash
crowdtensor volunteer operator campaign-dir --profile local --target-rounds 3
```

Inspect the built-in and installed Adapter registry:

```bash
crowdtensor community adapters list --json
crowdtensor community adapters check qwen2_lora_v1 --json
```

The official Mistral Adapter is separately packaged under
`plugins/mistral_adapter`. Its current live claim is the pinned 248M checkpoint,
not Mistral-7B or arbitrary Mistral models.

Run the fast local proof first:

```bash
crowdtensor local-proof --json
```

Run the user-friendly local swarm inference entry point:

```bash
crowdtensor infer "CrowdTensor routes small models across home compute"
```

Run the real CPU-only collaborative LoRA training foundation:

```bash
crowdtensor train lora --output-dir dist/my-training-job
crowdtensor train status dist/my-training-job
crowdtensor train export dist/my-training-job --output-dir dist/my-adapter
```

This path starts the existing HTTP Coordinator plus two local CPU Miner
processes on distinct dataset shards, performs one named-tensor DiLoCo outer
step, runs a separate two-process activation/gradient pipeline with checkpoint
recovery, and exports a standard PEFT adapter. See
[`docs/training-foundation.md`](docs/training-foundation.md) for the verified
scope and GPU handoff boundary.

The pinned heterogeneous Training Production workflow is configuration-driven
and idempotent:

```bash
crowdtensor train validate --json
crowdtensor train plan --json
crowdtensor train start dist/my-production-training --dry-run --json
crowdtensor train start dist/my-production-training --json
crowdtensor train status dist/my-production-training --watch
crowdtensor train metrics dist/my-production-training --format prometheus
crowdtensor train events dist/my-production-training --limit 100 --json

crowdtensor train pause dist/my-production-training --json
crowdtensor train resume dist/my-production-training --json
crowdtensor train rebalance dist/my-production-training \
  --reason performance_rebalance --json
crowdtensor train stop dist/my-production-training --json
crowdtensor train cleanup dist/my-production-training --json
```

`train serve`, `train invite`, and `crowdtensor-miner join --training` expose
the same durable Job to contributor machines. Keep Hugging Face and Kaggle
credentials in private environment variables or token files; public status,
events, metrics, evidence, and resume commands contain only hashes and safe
metadata. This RC is pinned to one Qwen2.5-7B PEFT topology. It is not
full-parameter training, arbitrary model partitioning, production GA, or an
SLA.

The Qwen Training Service Beta path is:

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

One verified run used two same-account T4x2 Kaggle Kernels, four stage-owned
CUDA processes, persistent Coordinator restart recovery after step 4, and a
standard PEFT export that reloads on CPU/CUDA and lowers validation loss. The
canonical strict artifact is
`dist/training-qwen15b-beta-20260712-r3-live-achieved/training_qwen15b_beta.json`.
This is a bounded 1.5B LoRA Beta RC, not 7B+, full-parameter training,
permissionless training, or a production SLA. See
[`docs/training-foundation.md`](docs/training-foundation.md) for the exact
evidence and service API boundary.

The elastic live gate goes beyond restarting processes in one allocation. It
commits all four stage checkpoints through an atomic barrier, deletes the old
T4x2 pair at step 4, waits with zero Miners, and resumes at step 5 on a new
pair whose local disks never contained the old checkpoints. Validate the
retained artifact with:

```bash
python scripts/training_qwen15b_elastic_check.py \
  --report dist/training-qwen15b-elastic-live-20260712-r2-repacked-achieved/training_qwen15b_elastic_live_probe.json \
  --require-ready --json
```

A larger retained showcase uses the same path for a real 256-step
`Qwen/Qwen2.5-1.5B` LoRA adaptation. It trains on 131,072 pinned WikiText-2
tokens, deletes the first T4x2 pair at step 128, resumes on two new T4x2
Kernels, and lowers 64-sequence held-out loss from `2.731937` to `2.350524`
(-13.96%). The standard PEFT export reloads on CPU/CUDA and all four temporary
Kernels are deleted. See
[`docs/qwen15b-elastic-training-showcase.md`](docs/qwen15b-elastic-training-showcase.md)
for artifacts, hashes, verification, licensing, and claim boundaries. This is
Kaggle logical multi-node causal-LM adaptation, not instruction tuning or an
independent physical multi-host benchmark.

The stronger instruction-tuning showcase uses pinned
`Qwen/Qwen2.5-7B-Instruct` and pinned GSM8K. Two concurrent T4x2 Kernels train
steps 1-128, both are deleted, and two fresh T4x2 Kernels restore four central
stage checkpoints and finish steps 129-256. The final attempt processes 262,144
non-padding tokens and exports a 392-tensor standard PEFT Adapter. On a
preregistered 128-item confirmatory holdout that is disjoint from development,
normalized exact match changes from `92/128` (71.875%) to `95/128` (74.219%),
an absolute improvement of 2.344 percentage points; valid answer rate remains
100%. The practical preregistered gate passes, while the paired bootstrap
interval includes zero, so no statistical-significance claim is made. See
[`docs/qwen7b-gsm8k-elastic-showcase.md`](docs/qwen7b-gsm8k-elastic-showcase.md)
for the failed development attempt, fixed revisions, Model Card, reproducible
commands, strict checker, hashes, cleanup, and claim boundaries.

The productized elastic path exposes the same semantics to an ordinary owner
and Miner:

```bash
crowdtensor train create dist/my-elastic-training --json
crowdtensor train serve \
  --elastic-job dist/my-elastic-training \
  --host 0.0.0.0 --port 8791

crowdtensor train invite dist/my-elastic-training \
  --coordinator https://coordinator.example \
  --output-file state/private/miner.invite.json --json
crowdtensor-miner join --training \
  --invite state/private/miner.invite.json --role auto

crowdtensor train status dist/my-elastic-training --watch
crowdtensor train export dist/my-elastic-training \
  --output-dir dist/my-elastic-adapter
crowdtensor train cleanup dist/my-elastic-training
```

The canonical Product Beta evidence is
`dist/training-elastic-beta-live-20260712-r6-repacked-achieved/training_elastic_beta_live_probe.json`.
It proves public owner create/status/export, product Miner join and graceful
drain, a full zero-Miner pause, Coordinator restart, replacement Miner central
restore, exactly-once steps 1-8, PEFT evaluation, and complete experiment
cleanup. The final regression report records 380 passed tests in
`dist/training-elastic-beta-tests-20260712-r2-final/training_qwen15b_test_summary.json`.

Checkpoint signatures, tensor/archive validation, stale lease fencing, upload
quotas, rejection counters, and quarantine reject malformed, non-finite,
unsigned, stale, and over-quota submissions. They do not establish
permissionless Byzantine-poisoning resistance. Local checkpoint storage is
live-verified; the optional S3/MinIO backend is implemented and unit-tested but
has not been externally live-tested. This Beta is pinned to Qwen2.5 1.5B,
eight steps, four stages, and two CUDA devices per Miner. See
[`docs/training-foundation.md`](docs/training-foundation.md) for the exact
security, storage, lifecycle, and non-capability boundaries.

The unified heterogeneous path uses the same owner and contributor lifecycle,
but the Job topology comes from a validated manifest and each CUDA device can
join as an independent one-GPU Miner:

```bash
export HF_TOKEN='private-value-if-the-model-source-requires-it'
crowdtensor train create dist/my-heterogeneous-training \
  --heterogeneous --model Qwen/Qwen2.5-7B \
  --hf-token-env HF_TOKEN --json
crowdtensor train serve \
  --elastic-job dist/my-heterogeneous-training \
  --host 0.0.0.0 --port 8791
crowdtensor train invite dist/my-heterogeneous-training \
  --coordinator https://coordinator.example \
  --output-file state/private/heterogeneous-miner.invite.json --json
crowdtensor-miner join --training \
  --invite state/private/heterogeneous-miner.invite.json --role auto
crowdtensor train status dist/my-heterogeneous-training --watch
crowdtensor train export dist/my-heterogeneous-training \
  --output-dir dist/my-heterogeneous-adapter
crowdtensor train cleanup dist/my-heterogeneous-training
```

The achieved live gate used four single-GPU T4 Miners plus one pure CPU Miner
in the same Qwen2.5-7B LoRA Job. It committed steps 1-3, removed a trainable
GPU Miner, restored the stage on a different Miner, and committed steps 4-6
with 24 forward activations and 24 backward gradients. The 392-tensor PEFT
adapter reloads on CPU and completes a finite full stagewise forward. Validate
the retained public-safe artifact with:

```bash
PYTHONPATH=. python scripts/training_heterogeneous_beta_check.py \
  --report dist/training-heterogeneous-beta-20260713-r3-r2-live-achieved/training_heterogeneous_beta.json \
  --require-ready --json
```

This is a bounded Qwen2.5-7B PEFT Beta with explicit stage boundaries and
epoch-level recovery. It is not arbitrary-model auto-partitioning,
full-parameter or TPU training, permissionless poisoning resistance, billing,
production GA, or an SLA. See
[`docs/training-foundation.md`](docs/training-foundation.md) for the manifest,
scheduler, transport, checkpoint, evidence, and cleanup contracts.

It starts the fast local product loopback path, runs split tiny GPT inference,
prints the local display-only generated text, and writes a compact
`infer_summary.json` plus a safe `infer_summary.md` under `dist/infer`. JSON,
Markdown, and public artifacts keep raw prompts, generated text, token ids,
credentials, and activations out of shareable files. Use `--full-evidence`
when you want the broader Public Swarm v2 gate instead of the faster user path.
Default local `infer` auto-selects an available loopback Coordinator port so
nearby local smoke runs do not collide; pass `--coordinator-port` only when you
need a fixed reproducible local port.
In human mode, `infer` and `generate` print a short safe stderr start hint before
long-running checks so the terminal does not look idle; `--json` keeps stdout
machine-readable.
Start with the `verdict` line when you only need the user-facing conclusion.
It condenses completion state, answer scope, terminal answer visibility,
shareable artifact safety, evidence level, GPU state, `fresh_kaggle_gpu`, and
next step into one stable line. Saved JSON/Markdown recompute
`inference_verdict` after local answer redaction, so a shareable artifact says
`answer_visible=False` and `answer=saved-terminal-redacted` when terminal text
was shown locally but removed from the saved report.
Maintainers can validate that final user-facing report contract without
starting a Coordinator or Kaggle resource:

```bash
python scripts/user_friendly_inference_frontdoor_check.py --json
```

It emits `user_friendly_inference_frontdoor_check_v1`, builds CI-safe fake
completed `infer` and `generate` reports through the real CLI report writers,
and verifies saved JSON/Markdown keep raw prompts, generated text, token ids,
credentials, and fresh Kaggle GPU claims out of shareable artifacts.
The fresh-clone onboarding gate also runs the installed real user entrypoint as
`crowdtensor infer --prompt-stdin --shareable-terminal`; that smoke must save an
`infer_summary` verdict with `answer=shareable-terminal-redacted`,
`gpu=local-cpu-only`, and `fresh_kaggle_gpu=False` without persisting the prompt
or generated answer.
In human mode, the terminal prints `answer_scope` so the answer display state is
explicit: whether any answer text is visible in the terminal and whether saved
JSON/Markdown stay hash-only. When generated text is available, the terminal
prints it as `answer:` or `answer[n]:` before `answer_scope` and `local_output`
safety metadata; when no local answer text is available, the terminal still
prints `answer_scope=no-local-answer`. `answer_scope.scope_state` uses stable values such as
`terminal-visible`, `saved-terminal-redacted`, `shareable-terminal-redacted`,
`json-suppressed`, and `no-local-answer`; the
Markdown `What To Do Next` and `Details` sections repeat that saved JSON and
Markdown contain no generated text. The adjacent `answer_scope_note` and
`output_display_note` terminal lines spell out the same answer-display and
artifact-redaction policy in plain text. Public inference evidence Markdown
also includes `output request note`, `prompt scope note`, and
`answer scope note` lines in `Output Scope`, so shared reports explain why
artifacts contain evidence, hashes, counts, and diagnostics instead of raw
prompts or answer transcripts. `local_output` includes
safe output `count` and `source` fields such as
`local-private-task-state` or `coordinator-validation`. JSON mode can still
report completed generation through `json-suppressed` plus redacted
`local_output` metadata such as `saved_redacted=True count=N`; that means
output exists, but the raw answer is intentionally hidden from machine-readable
stdout and saved artifacts. Use non-JSON human mode when you need a local
terminal answer.
Pick one prompt source per command: use the positional prompt,
`--prompt-text`/`--prompt`, `--prompt-file prompt.txt` for a UTF-8 single
prompt file, `--prompt-stdin` for an explicit stdin single prompt, or
`--prompt-texts` for a bounded comma-separated batch. Use
`--prompt-texts-file prompts.txt` for a UTF-8 batch file with one prompt per
non-empty line. Single prompts are capped at 256 characters; batch files accept
up to 4 non-empty prompt lines. The CLI rejects mixed prompt sources instead of
guessing. Reports expose `output_request.include_output` while keeping
`output_request.raw_generated_text_public` false in JSON and saved artifacts;
read the Markdown `Output Scope` section first when deciding whether a report is
shareable. Its `output request note`, `prompt scope note`, and
`answer scope note` explain why the artifact contains evidence, hashes, counts,
and diagnostics instead of raw prompts or answer transcripts.
Reports also include `prompt_scope`: a machine-readable summary of the prompt
source (`prompt-text`, `prompt-file`, `prompt-stdin`, `prompt-texts`, or
`prompt-texts-file`), prompt count, whether terminal next commands are
local-private, whether terminal next commands contain local prompt file paths,
and whether saved artifacts use placeholders. `prompt_scope` does not contain
raw prompt text.
Read `evidence_scope` when you need the shortest answer to what actually ran.
For `infer`, `local-cpu-loopback` means the fast local CPU product path ran,
`local-full-evidence` means the broader local evidence gate ran, and
`existing-runtime-preflight` / `existing-runtime-submit` means the command
checked or used an already running Coordinator or P2P-discovered route. For
`generate`, `existing-runtime-preflight` is a request-shape/readiness check
without submitting work, `existing-runtime-submit` submitted to an existing
Coordinator, and `p2p-runtime-*` came through discovery. `retained_gpu=True`
means imported historical GPU evidence was referenced; only
`fresh_kaggle_gpu=True` means this run verified a fresh Kaggle GPU proof;
`fresh_kaggle_gpu_attempted=True` without `fresh_kaggle_gpu=True` means an
attempted GPU path did not verify.
The adjacent `evidence_scope_note` terminal line and Markdown note spell out the
same scope in plain text, for example that a `generate --dry-run` was only a
preflight and submitted no generation task.
The `gpu_status` terminal/Markdown line is the fastest direct answer to the GPU
question: `local-cpu-only` means local CPU inference, `local-gpu-smoke-only`
means only local/CI GPU smoke evidence, `retained-gpu-evidence` means imported
historical GPU evidence, and only `fresh-kaggle-gpu-verified` means a fresh
Kaggle GPU proof was verified.
`gpu_proof_next_step` in the saved JSON/Markdown gives the explicit next
commands for optional CUDA smoke, Kaggle GPU packaging, and the side-effectful
fresh Kaggle GPU proof; it marks Kaggle commands as requiring explicit user
action, cleanup, and token rotation.
The current default quick-start inference path is local CPU / local loopback,
not a fresh Kaggle GPU run.

```bash
crowdtensor infer --prompt-file prompt.txt --max-new-tokens 8
echo "your prompt" | crowdtensor infer --prompt-stdin --max-new-tokens 8
crowdtensor infer --prompt-texts-file prompts.txt --max-new-tokens 8 --stream
crowdtensor generate --prompt-file prompt.txt --coordinator-url http://127.0.0.1:8787 --dry-run
echo "your prompt" | crowdtensor generate --prompt-stdin --coordinator-url http://127.0.0.1:8787 --dry-run
crowdtensor generate --prompt-texts-file prompts.txt --coordinator-url http://127.0.0.1:8787 --dry-run
```

Existing-swarm runs also include a safe `wait_progress` summary with
poll count, accepted rows, endpoint readiness, observed token progress, batch
request progress, and safe last-error type so timeouts are actionable without
exposing raw text; both `infer` and `generate` include `operator_action`
suggestions for checking tokens, Miner health, admin API access, or timeout
limits. Live and summary stream progress use safe request ids or hash prefixes,
print per-request token/target progress for bounded batch streams, mark missing
stream slots, print `stream_issue` when a request is missing or incomplete, and
print `recommended_next` plus `next[...]` lines with safe follow-up commands.
The adjacent `runtime_options` line records safe wait/retry controls:
`timeout_seconds`, `poll_interval`, `http_timeout`, and
`admin_results_limit`. Timeout retry commands preserve non-default
poll/http/result-limit values while only extending `--timeout-seconds`, so slow
remote swarms stay debuggable without exposing prompts, generated text,
credentials, or tokens.
The `trace` line in human output and the `trace` object in JSON/Markdown give a
safe troubleshooting summary: session id, request count, accepted ledger rows,
stream event count, and per-request ids or prompt hashes. It never includes raw
prompt text, generated text, generated token ids, credentials, or activations.
When you pass an inline positional prompt, `--prompt-text`, or `--prompt-texts`,
human terminal `review_next`, `recommended_next`, and `next[...]` commands may
render those prompt values so the command is directly copyable. Treat terminal
logs from those runs as local-private. Saved JSON/Markdown keep prompt
placeholders; `prompt_scope` records that distinction without storing raw text.
Use `--prompt-file` or `--prompt-texts-file` to keep raw prompt text out of
terminal commands, but ordinary terminal output still shows the local file path
for copying and marks `terminal_local_paths=True`. Use `--prompt-stdin`, or add
`--shareable-terminal`, when the terminal log itself needs to be shareable.
If you still want human-readable terminal output while keeping terminal logs
shareable, add `--shareable-terminal`; it keeps status, diagnostics, hashes,
artifact paths, and safe next commands, but hides inline prompts, local prompt
file paths, and local answer text from stdout. Saved JSON/Markdown then record
`shareable_terminal.enabled=True` and, when answer text was hidden,
`answer_scope.scope_state=shareable-terminal-redacted`.
With `--prompt-stdin`, shareable terminal output keeps a copyable `printf`
pipe placeholder for reruns without expanding the real stdin prompt.
For local `infer` runs, child proof commands receive prompt inputs through
temporary `.private` prompt files that are cleaned after the child command
returns, so child process arguments do not carry raw prompt text or local prompt
paths. If your environment treats process lists as shareable too, start the top
level command with `--prompt-file` or `--prompt-stdin` instead of an inline
positional prompt, and add `--shareable-terminal` when local prompt file paths
should stay out of terminal logs.
The `result` line and JSON/Markdown `result` object summarize completion state,
token count, output count, generated-text hash, and display safety:
`local-private` for terminal-only generated text, `hash-only` for redacted
summaries, `hash-only-json` for JSON stdout, and `saved-terminal-redacted` when
a saved JSON/Markdown file records that local terminal text existed but has
already been removed from the saved artifact. `shareable-terminal-redacted`
means `--shareable-terminal` also hid the answer from the human terminal. These
states do not expose generated text in shareable artifacts.
The `issue` line and JSON/Markdown `issue_summary` object condense the current
state, primary diagnosis code, next step, safe progress text, and whether a
redacted detail is available, so blocked or timeout runs have one place to read
first.
The `artifacts` line and JSON/Markdown `artifact_summary` object point to the
first Markdown summary to inspect, list the redacted JSON/Markdown paths, and
keep prompts, generated text, token ids, credentials, and activations out of
shareable files.
Start by reading the `review` line, or JSON/Markdown `review_summary`: it
combines the current state, next step, first artifact to inspect, recommended
command label, primary diagnosis code, and an `attention` value for warnings
such as incomplete stream evidence or skipped preflights; Markdown explains
those warnings in `What To Do Next`. The adjacent `inspect_first` line points to
the Markdown summary to open first. The adjacent `review_next` line repeats the
safe recommended command near that summary; human terminal output renders it
with local prompt sources for copying, using a `printf` pipe placeholder for
`--prompt-stdin`. Saved Markdown command lines also use that stdin pipe
placeholder, while JSON fields and saved Markdown prompt values keep prompt
placeholders and prefer `--prompt-file`, `--prompt-stdin`, or
`--prompt-texts-file` when rerunning saved commands. Inline prompt terminal next
commands are local-private. Then use the `status` line or
`user_status` for detail: `completed` means the request finished,
`preflight-ready` means submit next, `preflight-partial` means run the
recommended check first, and `blocked` means follow `action` /
`recommended_next`. Human `infer` and `generate` output use local prompt
sources in next commands so they are directly copyable; JSON reports and saved
artifacts keep raw prompts and token values replaced with placeholders.
Coordinator/session failure `detail` fields are redacted the same way, even if
a remote endpoint echoes prompt text or tokens.
When `ready_to_submit` is present, read
`readiness_label` and `next_step` first:

- `verified` means the route, Coordinator, and distinct stage Miners were
  checked.
- `partial` means the request can be submitted, but at least one live check
  still needs the printed follow-up command first. Machine-readable
  `generate_dry_run_partial` has the same meaning for `generate --dry-run`.
- `blocked` means follow `operator_action` before submitting.
- `skipped` means only the request shape was checked, usually because live
  preflight was intentionally skipped.
  Machine-readable `generate_request_shape_ready` has the same meaning and is
  intentionally separate from `generate_dry_run_ready`.
Submit commands mirror this state: labels such as `after stage preflight`,
`after live preflight`, or `after checks pass` mean run the preceding check
command first; `with caution` means the request can run but not every live
check was proven. Machine-readable `next_step` uses stable values such as
`submit`, `run_stage_preflight`, `run_live_preflight`, `submit_with_caution`,
and `fix_blockers`. `stage_preflight_unknown` means the stage check was
required but did not return a true/false result. `stage_preflight_not_checked`
means a route or Coordinator prerequisite failed first, so fix the printed
blocker, then rerun the dry-run with `CROWDTENSOR_OBSERVER_TOKEN` before
submitting.

If `coordinator_ready` is not true, its line now includes `error=...` for a
failed live probe or `reason=...` for an intentionally skipped check, so the
next step is visible without opening the JSON report.

Machine-readable reports use the same distinction: partial existing-swarm
preflight emits `crowdtensor_infer_preflight_partial`, while a fully verified
dry run emits `crowdtensor_infer_preflight_ready`. Treat partial as runnable
but not fully checked.

The manual `serve` and `join` commands also print `operator_action` and
`next[...]`, so the five-process flow tells you whether to rerun with `--run`,
start the missing stage Miner, or preflight with `generate --dry-run`.
When a printed next command includes `# requires CROWDTENSOR_...`, export those
environment variables before copying the command. Token and peer-secret values
are intentionally shown as environment requirements instead of embedded in
shareable JSON/Markdown reports. The default P2P-lite path starts
`crowdtensor p2pd`; if you choose `--p2p-backend real`, blocked discovery
reports will point you at the matching `crowdtensor p2p-daemon` command
instead.
`generate` writes safe `generate_summary.json` and `generate_summary.md` files
under `dist/generate` by default; the `artifacts` line tells you which Markdown
file to open first, and raw prompts, generated text, token ids, and tokens stay
out of those shareable artifacts.

To check an already running Coordinator or P2P-discovered swarm before
submitting a request, use `crowdtensor infer --mode existing --dry-run` or
`crowdtensor generate --dry-run` with `--coordinator-url` or
`--peer-bootstrap`. The dry run validates the session request, route metadata,
Coordinator `/ready` when live preflight is enabled, and visible stage0/stage1
Miner capability coverage when discovery or `--observer-token` makes that
safe. CI/package checks can add `--skip-live-preflight` to keep `generate
--dry-run` or `infer --mode existing --dry-run` as an offline request-shape
check.

For maintainer-grade release evidence, run the full public swarm beta gate:

```bash
crowdtensor public-real-llm-swarm-beta release \
  --max-new-tokens 16 \
  --http-timeout 30 \
  --json

crowdtensor public-real-llm-swarm-beta check \
  --beta-report dist/public-real-llm-swarm-beta/public_real_llm_swarm_beta.json \
  --output-dir dist/public-real-llm-swarm-beta-check \
  --max-new-tokens 16 \
  --json
```

This runs the stricter release aggregate and checks retained external evidence,
route hardening, failure requeue, KV-cache readiness, and artifact safety.
The `check` command is the official user-facing validation entry for the final
Beta contract; it writes `public_real_llm_swarm_beta_check.json` plus checked
Markdown, machine-readable, and support-bundle artifact paths. Pass
`--beta-report` to validate the release artifact you just generated; omitting
it keeps the CI-safe fixture check path.
Read `evidence_scope` in `public_real_llm_swarm_beta.json` and
`checked_evidence_scope` in `public_real_llm_swarm_beta_check.json` for the
shortest answer to what was verified: local CPU, retained evidence, or fresh
Kaggle GPU. The check terminal output also prints `checked_runtime_provenance`;
read that line when you need the detailed source/proof summary behind the
checked scope, and read `checked_gpu_status` for the direct local CPU /
retained GPU / fresh Kaggle GPU verdict. Beta reports also include
`gpu_proof_next_step`, and check reports mirror it as
`checked_gpu_proof_next_step`, so the optional fresh Kaggle GPU proof remains an
explicit side-effectful action. `fresh_kaggle_gpu=True` is the only fresh Kaggle GPU claim;
`fresh_kaggle_gpu_attempted=True` without that verified flag is not a completed
GPU proof. Retained external/GPU evidence is not a new Kaggle run.
When it completes, open `dist/public-real-llm-swarm-beta/public_real_llm_swarm_beta.md`
first, then `dist/public-real-llm-swarm-beta/support_bundle.json` if you need
diagnostics. The terminal also prints the final inference status: model and
token target, external/P2P/Public Swarm v2 token counts, accepted stage rows,
batch/stream readiness, KV-cache hit counts, and any `not_completed` blockers.
Safe shareable files are `public_real_llm_swarm_beta.json`,
`public_real_llm_swarm_beta.md`, and `support_bundle.json`; do not share
private env files, registries, runtime state, raw task logs, prompts,
generated text, generated token ids, credentials, activations, leases, or
idempotency material. If `ok` is false, start with the Markdown
`Not Completed` section and the printed `not_completed` lines; they map to the
missing token target, KV-cache, route hardening, batch/stream, external
runtime, or requeue evidence that must be rerun or imported.

To work on the core technology layer instead of the current tiny/small-model
product beta, build the Large-Model Shard Alpha evidence:

```bash
crowdtensor large-model-shard --output-dir dist/large-model-shard-alpha --json
python scripts/large_model_shard_alpha_check.py \
  --report dist/large-model-shard-alpha/large_model_shard_alpha.json \
  --json
```

This emits `large_model_shard_alpha_v1` plus a CLI summary. It creates a
7B-class GGUF / llama.cpp RPC runtime adapter, layer-range partition manifest,
`large_model_sharded_generate_v1` workload contract, serving-readiness hooks,
and benchmark harness artifacts. The default path is CI-safe planning evidence:
it does not require a GGUF file or real 7B hardware and keeps
`real_runtime_verified=false`. A real controlled LAN/VPN run can pass
`--model-path`, `--model-metadata`, `--device-profile`, and
`--real-benchmark-report` to import actual TTFT, tokens/s, memory, network, and
cache metrics. Public artifacts still redact raw prompts, generated text,
generated token ids, activations, KV cache, credentials, leases, and
idempotency material. This is the core large-model sharding Alpha/MVP, not
production Petals/Hivemind parity, not public RPC security, not NAT traversal,
not training/fine-tuning, and not a large-model serving SLA.

The next core technology gate is the Inference RC:

```bash
crowdtensor large-model-shard-rc --output-dir dist/core-technology-inference-rc --json
python scripts/large_model_inference_rc_check.py \
  --report dist/core-technology-inference-rc/core_technology_inference_rc.json \
  --json
```

This emits `core_technology_inference_rc_v1` and preserves the Alpha artifacts
while adding `large_model_runtime_adapter_interface_v1`,
`large_model_runtime_adapter_probe_v2`, `large_model_device_profile_v2`,
`large_model_partition_manifest_v2`, `large_model_runner_result_v1`,
`large_model_benchmark_v2`, `large_model_correctness_summary_v1`, and
`large_model_serving_hooks_v1`. The default mode is a CI-safe
fixture/diagnostic path: it probes missing llama.cpp binaries, local model
files, endpoint health, device memory, planner feasibility, runner/supervisor
contracts, benchmark metrics, correctness digests, serving hooks, redaction,
and blockers without claiming real 7B execution. Use `--mode real` only for a
controlled local/LAN/VPN runtime; it enforces `--max-new-tokens <= 8` and a
20 minute timeout ceiling. Use `--real-run-report` to import a completed real
short run with TTFT, tokens/s, wall time, generated token count, and output
digest. A benchmark import can supplement metrics, but the RC real claim comes
from the runner or real-run import. If no GGUF, llama.cpp binary, RPC worker, or
hardware is available, the correct ready state is still `ok=true`,
`real_runtime_verified=false`, `real_7b_runtime_verified=false`, and explicit
blockers. The RC remains inference-only, controlled-network-only, not public
P2P/NAT traversal, not production Petals/Hivemind parity, not training or
fine-tuning, and not an economic network.

The core technology handoff gate packages the RC for next-layer development:

```bash
crowdtensor core-tech-handoff --output-dir dist/core-technology-handoff-rc --json
python scripts/core_technology_handoff_check.py \
  --report dist/core-technology-handoff-rc/core_technology_handoff_rc.json \
  --json
```

This emits `core_technology_handoff_rc_v1`. It embeds the Inference RC, adds a
deployment runbook, adapter conformance summary, next-layer integration
contract, test-gate summary, final Support Bundle, and answers for how the
control layer, user layer, and future permissions/trust/billing layer should
consume core signals. In CI-safe environments it remains ready with
`real_runtime_verified=false`, `real_7b_runtime_verified=false`, and blockers
when GGUF/llama.cpp/RPC/hardware are absent. A retained external Kaggle T4 x2
7B proof now exists through the HF stage-selective CUDA path; the handoff still
does not claim production Petals/Hivemind parity, public P2P/NAT traversal,
training/fine-tuning, GPU marketplace economics, or a large-model serving SLA.

For fresh Kaggle GPU validation of the core layer, use the bounded Kaggle
runner:

```bash
crowdtensor large-model-kaggle-validate \
  --mode kaggle-auto \
  --tiers 7b \
  --accelerator NvidiaTeslaT4 \
  --runtime-path hf-cuda \
  --hf-cuda-install-compat \
  --max-new-tokens 1 \
  --context-length 128 \
  --output-dir dist/large-model-kaggle-validation \
  --json
python scripts/large_model_kaggle_validation_check.py \
  --report dist/large-model-kaggle-validation/large_model_kaggle_validation.json \
  --require-real-7b \
  --require-core-ready \
  --json
```

This emits `large_model_kaggle_validation_v1` and only marks
`core_validation_ready=true` when a real 7B/8B-class run succeeds on Kaggle GPU
through the sharded runtime path. Failed or partial runs are still useful
evidence but must keep `core_validation_ready=false`. A 2026-06-16 retained
proof at
`dist/large-model-kaggle-stage-selective-hf-7b-manual-rope-20260616/large_model_kaggle_validation.json`
ran `Qwen/Qwen2.5-7B-Instruct` with `hf_transformers_stage_selective_cuda` on
two Kaggle `Tesla T4` GPUs, stage0 on `cuda:0`, stage1 on `cuda:1`,
`generated_token_count=1`, `real_7b_runtime_verified=true`,
`sharded_path_verified=true`, `multi_worker_sharded_path_verified=true`,
`core_validation_ready=true`, public-safe redaction, and deleted the temporary
Kaggle kernel. This is the current external core technology validation proof.
It is not GGUF/llama.cpp RPC success, not production P2P, not a GPU
marketplace, not training/fine-tuning, and not a throughput SLA.

A 2026-06-13 hardware probe found that Kaggle CLI accepts
`--accelerator NvidiaTeslaT4` and returned two `Tesla T4` devices with Torch
CUDA visible. Use that accelerator for the main large-model validation instead
of the generic `GPU` request, which previously assigned single P100 runs. The
retained T4 x2 source-CUDA/RPC attempts proved T4 x2 hardware,
`CMAKE_CUDA_ARCHITECTURES=75`, `GGML_CUDA_NO_VMM=ON`, `GGML_RPC=ON`, CUDA
llama.cpp build, and live RPC workers, but no GGUF tier has successfully
generated tokens through llama.cpp RPC on Kaggle. The strongest RPC blocked
artifact is
`dist/large-model-kaggle-validation-t4x2-rpc-small-telemetry-inplace-20260613/large_model_kaggle_validation.json`:
it verifies one CUDA0 RPC worker, successful 1.5B GGUF download, and monitored
`llama-cli --rpc` execution for about 451 seconds. Re-importing its raw report
produces `resource_pressure_summary` with `cgroup_memory_peak_ratio=0.9345` and
`gpu_memory_used_peak_ratio=0.0702`, so the current blocker is Kaggle container
memory pressure while executing llama.cpp RPC in one Notebook cgroup, not T4
assignment, CUDA build, model download, 7B size alone, or two-worker tensor
split startup. The runner now records cgroup/GPU/process/disk telemetry, writes
run reports atomically, limits telemetry sample growth, handles invalid raw JSON
as a blocked report, and cleans source/archive build inputs in place after a
successful source-CUDA build. The 7B CLI fallback evidence at
`dist/large-model-kaggle-validation-t4x2-cli-7b-20260613-r1/large_model_kaggle_validation.json`
verified T4 x2 hardware, CUDA llama.cpp build, Qwen2.5 7B Q2_K GGUF download,
and `llama_cpp_cli` run start, but no generated tokens were retained:
`cgroup_memory_peak_ratio=0.9335`, `disk_min_free_bytes=335552512`,
`large_model_kaggle_disk_pressure`, and low GPU memory
(`gpu_memory_used_peak_ratio=0.1075`) show Kaggle container pressure rather than
VRAM exhaustion. The bounded r2 retry at
`dist/large-model-kaggle-validation-t4x2-cli-7b-20260613-r2/large_model_kaggle_validation.json`
used the repaired actual-slug lifecycle, minimal `llama-runtime` compaction,
disabled CUDA cache, small `-b/-ub 32` batches, and report-write fallback; it
still ended with Kaggle `Killed` plus `No space left on device`, and cleanup
deleted the temporary kernel. The retained 2026-06-12 P100 attempts verified GPU
hardware and cleanup but did not produce generated tokens through the
sharded/RPC path; the successful Hugging Face CUDA compatibility smoke at
`dist/large-model-kaggle-validation-small-hf-cuda-compat-import-20260612/large_model_kaggle_validation.json`
proved only tiny-model GPU generation, not 7B/8B and not the sharded/RPC path.
Treat the historical RPC/P100 reports as partial evidence with blockers, not
completion of the core large-model validation goal. They explain why the
completed 7B Kaggle proof currently uses the HF stage-selective CUDA adapter
rather than llama.cpp RPC. Future work may return to GGUF/RPC, but that is a
separate adapter milestone. The HF `real_llm_sharded_infer` path now records an
`execution_support` summary on `real_llm_artifact_v1` and workload specs, and
the stage-selective runtime can load/apply only stage-owned safetensors keys for
Llama-like Qwen models without publishing prompts, generated text, token ids,
activations, cache paths, or tensor values.

If you only want CPU-only deterministic demos without Hugging Face dependencies:

```bash
python -m pip install -e '.[dev]'
crowdtensor cpu-infer --mode local --json
```

## Manual Swarm Demo

The beta can also be run as separate local processes. Open five terminals from
the repository root after installing the package. Use the same local tokens in
terminals 2-5:

```bash
export CROWDTENSOR_ADMIN_TOKEN=local-admin
export CROWDTENSOR_MINER_TOKEN=local-miner
export CROWDTENSOR_OBSERVER_TOKEN=local-observer
```

```bash
# Terminal 1: discovery
crowdtensor p2pd --swarm-id public-swarm-v2 --run

# Terminal 2: Coordinator/API
crowdtensor serve --p2p --swarm-id public-swarm-v2 --run

# Terminal 3: stage 0 Miner
crowdtensor join --stage stage0 --p2p --swarm-id public-swarm-v2 --miner-id stage0 --run

# Terminal 4: stage 1 Miner
crowdtensor join --stage stage1 --p2p --swarm-id public-swarm-v2 --miner-id stage1 --run

# Terminal 5: user request
crowdtensor generate \
  --p2p \
  --swarm-id public-swarm-v2 \
  --prompt "CrowdTensor routes small models across home compute" \
  --max-new-tokens 16 \
  --http-timeout 30 \
  --dry-run \
  --observer-token "$CROWDTENSOR_OBSERVER_TOKEN"

crowdtensor generate \
  --p2p \
  --swarm-id public-swarm-v2 \
  --prompt "CrowdTensor routes small models across home compute" \
  --max-new-tokens 16 \
  --http-timeout 30
```

For real multi-machine trials, keep the Coordinator on a trusted network
boundary, use explicit tokens, and rotate temporary tokens after public demos.
The Coordinator does not have to bind directly to a public interface if a tunnel,
VPN, or reverse proxy provides the Miner-facing URL. Start the Coordinator with
`--coordinator-public-url https://YOUR-TUNNEL.example --expect-remote-miners`
to keep local binding separate from the URL that remote Miners should join, and
use `crowdtensor join --expect-remote-coordinator` on Miner hosts to catch
accidental `127.0.0.1` or `localhost` invites before running.
For the shortest private setup package, generate the Coordinator registries and
stage invites in one local directory:

```bash
crowdtensor swarm-bootstrap \
  --output-dir state/swarm-bootstrap \
  --coordinator-url https://YOUR-TUNNEL.example \
  --tunnel-command 'cloudflared tunnel --url http://127.0.0.1:8787' \
  --expect-remote-miners
state/swarm-bootstrap/tunnel_doctor.sh
crowdtensor swarm-bootstrap-check --output-dir state/swarm-bootstrap --expect-remote-miners
```

If you do not have a stable public URL yet, `--tunnel-provider cloudflare-quick`
creates a route-prep package instead of Miner invites:

```bash
crowdtensor swarm-bootstrap --output-dir state/swarm-route-prep --tunnel-provider cloudflare-quick --expect-remote-miners
state/swarm-route-prep/discover_cloudflare_tunnel.sh
```

The discovery script starts a temporary Cloudflare quick tunnel, extracts the
`trycloudflare.com` URL, and then creates the final bootstrap package with that
URL. Keep the quick tunnel process running while those temporary Miner packages
are in use; for stable operation use a named tunnel, reverse proxy, VPN, or
port forwarding.

The report lists the local private operator invite, stage0/stage1 Miner invites,
private `miner.join-code.txt` files, operator/coordinator private env files,
hashed registries, and copyable discovery / `serve` / `join` / `generate`
scripts plus `start_control_plane.sh`, optional `start_tunnel.sh`,
`tunnel_doctor.sh`, `start_discovery.sh`, `check_route.sh`, `verify_bootstrap.sh`, private
`stage0.miner-package.tar.gz` / `stage1.miner-package.tar.gz`, matching
`stage0.run-miner.sh` / `stage1.run-miner.sh`,
`stage0.handoff.sha256` / `stage1.handoff.sha256`,
`stage_handoff_manifest.json`, `handoff_doctor.sh`, `operator_status.sh`,
`auditor_status.sh`, `accounting_status.sh`, `trust_review.sh`,
`settlement_review.sh`, `operator_review.sh`, stage `install.sh`,
`doctor.sh`, `check_join.sh`, `support_bundle.sh`, and `SWARM_BOOTSTRAP.md`.
When `--tunnel-command` is supplied, the command is written only to
`private/tunnel.private.env`; public reports and Markdown show the tunnel
launcher without echoing tunnel tokens or provider command lines. Keep the operator invite and
operator env on the operator host, use the coordinator env only for
`start_coordinator.sh`, run `tunnel_doctor.sh` or
`crowdtensor swarm-tunnel-doctor --output-dir state/swarm-bootstrap --expect-remote-miners`
to write `crowdtensor_swarm_tunnel_doctor_v1`
diagnostics in `tunnel_doctor.json` before startup, run `start_control_plane.sh`
to start the tunnel, discovery, and the Coordinator together, run `verify_bootstrap.sh` after the Coordinator starts,
run `check_route.sh` to confirm the advertised Coordinator URL is suitable for
the intended local or remote Miner path,
run `operator_status.sh` for a read-only `/ready` / `/state` / accounting /
settlement summary from the admin operator env, or run `operator_review.sh` to
chain admin status, auditor event status, accounting status, trust review, and
settlement draft review without embedding tokens in public scripts,
and send each private stage archive plus matching `stageX.run-miner.sh` and
`stageX.handoff.sha256` only to the matching Miner host. The runner verifies
the checksum before it safely extracts the archive; the recommended Miner-side
flow is `./stageX.run-miner.sh --setup`, then `./stageX.run-miner.sh --start`.
Use `./stageX.run-miner.sh --install --dry-run` to preview the install step,
and `--doctor`, `--check-only`, or `--support-bundle` for troubleshooting.
The runner extracts the package and delegates `--setup` to stage `install.sh`
plus `doctor.sh`.
Stage `install.sh` creates `.crowdtensor-venv` with the default `[hf]` runtime when `crowdtensor` is not already on PATH. Stage `doctor.sh` writes `miner_support_bundle.json` and then checks Coordinator
reachability plus token-backed admission without starting the Miner; stage
`check_join.sh` exposes that no-run admission check directly, and stage
`join.sh` then runs the same invite-code path
with `--run`, so the Miner host does not need to edit JSON invites. If the
preflight fails, stage `support_bundle.sh` writes public-safe
`miner_support_bundle.json` diagnostics without raw `miner.join-code.txt` or
`miner_token` values; the bundle includes
`crowdtensor_miner_local_environment_v1` with `local_environment_ready`,
`crowdtensor` CLI, `sha256sum`, Python, and optional torch/CUDA probes so a
Miner host can report local setup failures without sharing secrets. When bootstrap is run with
`--peer-bootstrap`, the private invite also carries
`crowdtensor_miner_join_discovery_v1`, so `join --invite-code-file` can enable
P2P-lite discovery and resolve the Coordinator without the Miner user
hand-writing `--peer-bootstrap`.
Read the `bootstrap_handoff` summary in the JSON or terminal output before
copying stage directories. `remote_miners_ready` means the advertised URL is a
remote-capable route; `ready_to_copy_stage_packages` becomes true only after a
live `verify_bootstrap.sh` / `--check-admission` preflight passes.
Run `handoff_doctor.sh` or `crowdtensor swarm-handoff-doctor` to write
`crowdtensor_swarm_handoff_doctor_v1` reports (`handoff_doctor.json` and
`handoff_doctor.md`) with the current blockers and exact stage files to copy.
`crowdtensor swarm-bootstrap-check` verifies required
files, `0600` private invite/env permissions, `0700` scripts, hashed registries,
Coordinator/operator env separation, and that scripts/Markdown do not embed
plaintext tokens before handoff, including `check_route_script_ready`,
`tunnel_doctor_script_ready`, `operator_status_script_ready`,
`auditor_status_script_ready`, `accounting_status_script_ready`,
`trust_review_script_ready`, `settlement_review_script_ready`,
`operator_review_script_ready`, `stage_install_scripts_ready`, `stage_doctor_scripts_ready`, and
`stage_support_bundle_scripts_ready`.
It also verifies `stage_package_archives_ready` so the operator can copy one
private tarball per Miner instead of hand-picking files, plus
`stage_archive_runner_scripts_ready` for the matching one-command Miner runner
and `stage_handoff_checksums_ready` for the copied checksum/manifest handoff.
With `--expect-remote-miners`, it also checks
that both stage invites share a Miner-facing Coordinator URL that is not
`127.0.0.1` / `localhost`. After the Coordinator is running, add
`--check-coordinator` or `--check-admission` to call `/ready` and token-backed
`/tasks/preflight` for both stage invites without claiming tasks; this is a
setup helper, not a production NAT traversal or billing system.
Multi-operator deployments can start the same product Coordinator with
`crowdtensor serve --operator-token-registry state/operator_registry.json --run`
so audit/accounting operators do not need the legacy owner-level admin token.
Create role-scoped operator entries with `crowdtensor operator-invite`; it writes
only a hashed verifier to the registry and a private invite file containing the
plaintext operator token:

```bash
crowdtensor operator-invite \
  --registry state/operator_registry.json \
  --operator-id generate-desk \
  --role admin \
  --allowed-workload real-llm-sharded \
  --max-new-tokens 8 \
  --max-total-sessions 100 \
  --invite-file state/private/generate-desk.operator.invite.json
```

Add `--inference-session-rate-limit N --inference-session-rate-window-seconds S`
to rate-limit generation session creation per admin/operator subject.
Use `crowdtensor coordinator-route --coordinator-url ... --expect-remote-miners`
as the no-token first check for whether the advertised Coordinator URL is
local-only, private-network, or public/tunnel, and add `--check-ready` after the
Coordinator is running to verify `/ready`. It writes
`crowdtensor_coordinator_route_cli_v1` artifacts with `join_options`,
`recommended_join_option`, and `recommended_setup_command` so an operator can
choose public HTTPS/reverse-proxy, tunnel, VPN/LAN, or explicit port-forwarding
before creating Miner packages. It does not join Miners, claim tasks, or provide
NAT traversal.
Use `crowdtensor operator-status --coordinator-url ...` as a read-only daily
operator check over `/ready`, `/state`, trust/quarantine counters, and optional
accounting/settlement summaries. It writes public-safe
`crowdtensor_operator_status_cli_v1` artifacts and does not create sessions,
override trust, or execute payments.
Use `crowdtensor trust --coordinator-url ...` to write a public-safe
`crowdtensor_trust_cli_v1` trust/quarantine report from `/state`, and add
`--mode block|allow|reset --miner-id ... --workload-type ...` with an
owner/admin token to set workload-scoped trust overrides. The report redacts
operator credentials and override reason text; it is an operator safety helper,
not Sybil resistance, staking, slashing, or automatic payment enforcement.

## How It Works

CrowdTensor is intentionally simple at the control-plane layer:

- **Coordinator** owns sessions, leases, result validation, trust state, and
  public HTTP APIs.
- **Discovery daemon** advertises and discovers swarm endpoints for local and
  controlled remote demos.
- **Stage Miners** opt in to specific capabilities such as `stage0`, `stage1`,
  CPU tiny-model inference, or optional CUDA tiny-model inference.
- **Infer/generate clients** create read-only inference sessions and stream or
  collect decoded results.
- **Evidence packs** record redacted readiness, diagnostics, stage assignment,
  failure recovery, and support bundle details.

The current inference work is small by design. It is meant to prove that the
distributed route is correct before the project expands model size, networking,
market incentives, and browser/GPU participation.

## Current Boundaries

CrowdTensor does not currently provide:

- Permissionless production P2P routing with DHT/NAT traversal.
- Hivemind-level distributed large-model serving.
- Open public prompt serving for arbitrary users.
- GPU pooling as a production marketplace.
- Strong economic incentives or staking.
- A security model suitable for untrusted public Miners.

The safe mental model is: **controlled, auditable swarm inference beta for small
models and protocol development**.

## Useful Commands

```bash
# Local proof bundle
crowdtensor local-proof --json

# User-friendly local swarm inference
crowdtensor infer "CrowdTensor routes small models across home compute"

# CPU-only inference aggregate
crowdtensor cpu-infer --mode local --json

# Product-shaped public swarm beta
crowdtensor public-swarm-beta product-beta --json

# Public Real-LLM Swarm Inference Beta
crowdtensor public-real-llm-swarm-beta release --max-new-tokens 16 --json
crowdtensor public-real-llm-swarm-beta check --beta-report dist/public-real-llm-swarm-beta/public_real_llm_swarm_beta.json --output-dir dist/public-real-llm-swarm-beta-check --json

# Package a two-machine style public real-LLM swarm run
crowdtensor public-real-llm-swarm-beta package --output-dir dist/public-real-llm-package --json

# Clean generated caches and temporary artifacts, dry-run by default
crowdtensor clean-artifacts
```

## Repository Map

- `crowdtensor/` - CLI entry points and user-facing commands.
- `crowdtensord/` - Coordinator, Miner, runtime contracts, validation, and
  workload implementations.
- `scripts/` - evidence packs, release checks, live proof wrappers, and
  acceptance gates.
- `tests/` - unit and integration-style checks for the runtime and evidence
  contracts.
- `docs/quickstart.md` - a guided first run.
- `docs/architecture.md` - control-plane and swarm architecture.
- `docs/use-cases.md` - who the project is useful for today.
- `ROADMAP.md` - what is current, next, and intentionally later.

## Maintainer Anchors

The short README keeps the public surface readable. Maintainer gates still
track deeper artifacts and docs such as `docs/api.md`,
`scripts/api_contract_check.py`, `api_contract`, `site/index.html`, and the
5-minute local swarm demo. See `ROADMAP.md`, `docs/protocol.md`,
`docs/use-cases.md`, and `docs/architecture.md` for the protocol boundary,
`runtime_contract_v1`, Support Bundle, and "Protocol boundary changed" context.

Compatibility anchors preserved for release checks: CrowdTensorD, What Works
Today, What Is Not Ready, Public Swarm Inference Beta,
`public_swarm_inference_beta_v1`, `public_swarm_inference_beta_ready`,
`public_swarm_product_beta_ready`, `public_swarm_product_rc_ready`,
`coordinator_product_surface_ready`, `session_protocol_ready`,
`p2p_lite_discovery_ready`, `gpu_generation_evidence_import_ready`,
`cpu_fallback_ready`, `public_swarm_beta_evidence_import_ready`,
`two_stage_split_inference_ready`, `local_loopback_ready`,
`external_live_evidence_imported`, `stage0_live_requeue_evidence_ready`,
`stage1_live_requeue_evidence_ready`, `decoded_tokens_match`,
`distinct_stage_miners`, `stage_assignment_valid`,
`public_swarm_inference_beta_pack.py`,
`public_swarm_inference_beta_check.py`, `crowdtensor public-swarm-beta`,
`public-swarm-beta product-beta`, `public-swarm-beta local-loopback`,
`public-swarm-beta evidence-import`, `prepare`, `coordinator`, `miner`,
`verify`, `collect`, `clean`, CPU-only, read-only, not libp2p, not DHT, not NAT
traversal, not production Swarm Inference, and not large-model serving.

Real small-model anchors: Real Small-LLM Sharded Inference Beta,
`real_llm_sharded_infer`, `real_llm_sharded_infer_v1`,
`real_llm_artifact_v1`, `real_llm_sharded_evidence_v1`,
`remote_real_llm_sharded_beta_v1`,
`real_llm_sharded_inference_evidence_pack.py`,
`remote_real_llm_sharded_beta_pack.py`,
`remote_real_llm_sharded_beta_check.py`,
`crowdtensor real-llm-shard-infer`,
`crowdtensor real-llm-shard-infer-beta`,
`crowdtensor remote-demo --workload real-llm-sharded`,
`--enable-hf-tiny-gpt-runtime`, `--hf-cache-dir`, `--real-llm-stage-role`,
`real_llm_sharded_stage0`, `real_llm_sharded_stage1`,
`real_llm_sharded_both`, `real_llm_artifact_ready`,
`activation_transport_ready`, `baseline_match`, `decoded_tokens_match`,
`stage_assignment_valid`, `remote_real_llm_sharded_ready`,
`remote_two_machine_real_llm_sharded_ready`,
`remote_real_llm_sharded_acceptance_v1`,
`remote_real_llm_sharded_observability_v1`,
`remote_python_real_llm_sharded_infer`, `hf_dependencies_missing`,
`hf_transformers_cpu`, optional [hf], CPU-only, read-only, not P2P, not
GGUF/llama.cpp, and not large-model.

Core large-model Alpha anchors: `crowdtensor large-model-shard`,
`large_model_shard_alpha_v1`, `large_model_runtime_adapter_v1`,
`large_model_partition_manifest_v1`, `large_model_sharded_generate_v1`,
`large_model_shard_benchmark_v1`, `large_model_shard_alpha_check_v1`,
`scripts/large_model_shard_alpha_pack.py`,
`scripts/large_model_shard_alpha_check.py`, llama.cpp RPC / GGUF,
layer-range placement, controlled LAN/VPN/local process only,
`real_runtime_verified=false` by default, fixture planning evidence unless a
real benchmark report is imported, public artifact redaction, not public RPC
safe, not production Petals/Hivemind parity, not P2P/NAT traversal, not
training/fine-tuning, and not large-model serving SLA.

Live RC anchors: Real Small-LLM Sharded Inference Live RC,
`real_llm_live_rc_v1`, `real_llm_live_rc_check.py`,
`real_llm_live_rc_pack.py`, `kaggle_real_llm_live_package.py`,
`kaggle_real_llm_live_package_v1`, `crowdtensor real-llm-live-rc`,
`local-generated`, `kaggle-generated`, `external-existing`,
`kaggle_real_llm_live_package_ready`, `kaggle-upload-real-llm-stage0`,
`kaggle-upload-real-llm-stage1`,
`local_generated_real_llm_stage_upload_standins_ready`,
`external_runtime_verified`, `kaggle_real_llm_stage0_seen`,
`kaggle_real_llm_stage1_seen`, `kaggle_real_llm_sharded_ready`,
`real_llm_artifact_ready`, `--enable-hf-tiny-gpt-runtime`,
`--real-llm-stage-role`, CPU-only, read-only, not P2P, not production Swarm
Inference, and not large-model.

## Who Should Try It

CrowdTensor is a good fit if you want to:

- Study practical distributed inference mechanics.
- Contribute to open AI infrastructure before it becomes a large production
  network.
- Run controlled home-compute or lab-machine experiments.
- Help harden routing, validation, observability, and operator ergonomics.

It is not the right tool yet if you need production uptime, large open-weight
model serving, untrusted public miners, or a finished token economy.

## Development

```bash
python -m pip install -e '.[dev,hf]'
python -m unittest discover -s tests
```

For documentation-only changes, at minimum run:

```bash
git diff --check
```

## License

CrowdTensor is released under the Apache License 2.0.
