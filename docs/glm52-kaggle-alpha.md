# GLM 5.2 Kaggle Alpha

This Alpha turns the verified GLM 5.2 Kaggle CPU/GPU/TPU same-request path into
a local service wrapper. It uses compatible public quantized weights
`cyankiwi/GLM-5.2-AWQ-INT4` for `zai-org/GLM-5.2`.

## Credentials

Provide your own Kaggle credentials and optional Hugging Face token through
private environment or token files. Public artifacts must not contain tokens,
cookies, signed URLs, raw prompts, generated text, token ids, activations,
logits, or KV-cache payloads.

The existing section-token format is supported:

```bash
crowdtensor deploy glm52-kaggle \
  --token-file ~/.config/crowdtensor/kaggle-tokens.md \
  --token-section cpuowner \
  --provider-token-section-map kaggle_cuda=gpuowner,kaggle_jax_tpu=tpuowner,kaggle_cpu=cpuowner
```

Raw token files are also supported with `--raw-token-file` or the per-provider
`--provider-raw-token-file-map`.

Hugging Face tokens are read from environment variables named by
`--hf-token-env`:

```bash
export HF_TOKEN='hf_...'

PYTHONPATH=. python -m crowdtensor.cli deploy glm52-kaggle \
  --model cyankiwi/GLM-5.2-AWQ-INT4 \
  --accelerators cpu,gpu,tpu \
  --hf-token-env HF_TOKEN,HUGGING_FACE_HUB_TOKEN \
  --output-dir dist/glm52-kaggle-alpha-local \
  --json
```

The token value is uploaded only inside the private Kaggle runtime env. Public
artifacts store only env-name hashes, counts, configured booleans, and
`hf_token_public=false`.

## Deploy

Build a public-safe service/Alpha artifact without launching Kaggle workers:

```bash
PYTHONPATH=. python -m crowdtensor.cli deploy glm52-kaggle \
  --model cyankiwi/GLM-5.2-AWQ-INT4 \
  --accelerators cpu,gpu,tpu \
  --hf-token-env HF_TOKEN,HUGGING_FACE_HUB_TOKEN \
  --output-dir dist/glm52-kaggle-alpha-local \
  --json
```

Run the side-effectful multi-token Kaggle Alpha attempt:

```bash
PYTHONPATH=. python -m crowdtensor.cli deploy glm52-kaggle \
  --model cyankiwi/GLM-5.2-AWQ-INT4 \
  --accelerators cpu,gpu,tpu \
  --hf-token-env HF_TOKEN,HUGGING_FACE_HUB_TOKEN \
  --run-live \
  --gpu-quota-preflight \
  --max-new-tokens 8 \
  --stage-worker-package-report dist/glm52-kaggle-stage-worker-package-20260707-alpha-r11-unique-runtime-tuning/glm52_kaggle_stage_worker_package.json \
  --stage-push-parallelism 7 \
  --full-prefix-prefill-length 1 \
  --full-prefix-dsa-mask-topk 1 \
  --full-prefix-executed-expert-count 2 \
  --full-prefix-top-k 1 \
  --full-prefix-row-block-size 512 \
  --full-prefix-max-tensor-bytes 33554432 \
  --full-prefix-max-block-bytes 16777216 \
  --cpu-group-stage-attempt-seconds 2.5 \
  --cpu-group-stage-poll-seconds 0.5 \
  --output-dir dist/glm52-kaggle-alpha-live \
  --json
```

The Alpha is successful only when `scripts/glm52_kaggle_alpha_check.py
--require-ready` passes for the generated `glm52_kaggle_alpha.json`.
`deploy` and `serve glm52-kaggle` currently support only the GLM 5.2 compatible
model sources `cyankiwi/GLM-5.2-AWQ-INT4` and `zai-org/GLM-5.2`, and require
the supported three-accelerator request `cpu,gpu,tpu`. Other models or missing
accelerator families fail fast instead of being packaged as non-GLM fallback
evidence.

## Serve

Start the local HTTP service:

```bash
PYTHONPATH=. python -m crowdtensor.cli serve glm52-kaggle \
  --model cyankiwi/GLM-5.2-AWQ-INT4 \
  --accelerators cpu,gpu,tpu \
  --hf-token-env HF_TOKEN,HUGGING_FACE_HUB_TOKEN \
  --port 8789 \
  --output-dir dist/glm52-kaggle-alpha \
  --run
```

Endpoints:

- `GET /health`
- `GET /status`
- `POST /generate`
- `POST /cleanup`

Example request:

```bash
curl -sS http://127.0.0.1:8789/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain CrowdTensor in one sentence.","max_new_tokens":8,"timeout":7200}'
```

The response is public-safe by default: it returns prompt hashes, token hashes,
provider coverage, cleanup status, and artifact paths. Raw generated text is
not public in the Alpha artifact boundary.
`timeout` or `timeout_seconds` is a per-request upper bound for the same-request
live wait and Coordinator task timeout, capped by the service's configured
maximum wait.
Malformed JSON, empty/non-object request bodies, missing prompts, and invalid
token counts return public-safe HTTP 400 responses. These validation failures
do not launch Kaggle workers and do not write raw prompts or request bodies to
public artifacts.
If the output directory already contains a current GPU quota blocker, `/generate`
returns HTTP 503 with public-safe blockers, cleanup proof, phase status, and the
next resume command instead of starting another doomed Kaggle push.
`POST /cleanup` uses the same cleanup proof path as the CLI cleanup command and
returns public-safe deletion evidence for temporary Kaggle kernels and private
packages.

The same request path is available through the CLI:

```bash
PYTHONPATH=. python -m crowdtensor.cli generate --target glm52-kaggle \
  --prompt-text "Explain CrowdTensor in one sentence." \
  --coordinator-url http://127.0.0.1:8789 \
  --max-new-tokens 8 \
  --output-dir dist/glm52-kaggle-alpha-request \
  --json
```

Equivalent positional target form:

```bash
PYTHONPATH=. python -m crowdtensor.cli generate glm52-kaggle \
  --prompt-text "Explain CrowdTensor in one sentence." \
  --coordinator-url http://127.0.0.1:8789 \
  --max-new-tokens 8 \
  --output-dir dist/glm52-kaggle-alpha-request \
  --json
```

The CLI writes `glm52_kaggle_alpha_generate_cli.json` with prompt and service
URL hashes, HTTP status, public-safe response data, and diagnosis codes. It
does not persist the raw prompt or raw service URL. If the local service is not
reachable, it reads the same output directory's Alpha/status artifacts and
includes public-safe artifact recovery details such as phase, blockers,
`next_resume_command`, and `resume_private_inputs`.

## Status And Cleanup

`status` defaults to the deploy output directory (`dist/glm52-kaggle-alpha`).
It can read a running service status file, a deploy CLI summary, a canonical
Alpha artifact, the live report, and the GPU quota preflight report from the
same output directory. For quota-blocked deploys it reports `phase=blocked_gpu_quota`,
the next quota refresh time, cleanup status, blockers, and the resume command.
The HTTP service mirrors this behavior on startup: `GET /status` loads an
existing public-safe Alpha artifact from the same output directory before any
new request is submitted.
`cleanup` can read the same deploy artifacts; when a live run was skipped by
GPU quota preflight, it writes a public-safe cleanup proof from the Alpha/quota
evidence instead of requiring an HTTP service status file.
`cleanup` also defaults to `dist/glm52-kaggle-alpha`, matching `deploy` and
`status`, so a user can run `crowdtensor cleanup` after the default deploy path
without passing `--output-dir`.

```bash
PYTHONPATH=. python -m crowdtensor.cli status glm52-kaggle \
  --output-dir dist/glm52-kaggle-alpha-live \
  --json

PYTHONPATH=. python -m crowdtensor.cli cleanup glm52-kaggle \
  --output-dir dist/glm52-kaggle-alpha-live \
  --json
```

Cleanup success requires temporary Kaggle kernels and private packages to be
deleted, or a retained queued TPU request to be explicitly recorded as a
blocker. Unknown live resources cannot be marked successful.

## Current Boundary

The current canonical Alpha blocker is:

`dist/glm52-kaggle-alpha-20260708-r34-http-cleanup-route/glm52_kaggle_alpha.json`

r34 supersedes r33 by adding a checker-backed HTTP cleanup route to the
ordinary-user service. Service reports now record `cleanup_route_ready=true`,
the smoke probe verifies `GET /health`, `GET /status`, `POST /generate`, and
`POST /cleanup`, and the canonical r34 report imports
`service_smoke_summary.cleanup_route_verified=true` with temporary Kaggle
kernels deleted, temporary private packages removed, and no live resources left
running. This is service/cleanup usability evidence under the quota blocker,
not live inference success.

r33 superseded r32 by adding artifact-backed recovery to the user-facing
generate CLI. If `crowdtensor generate glm52-kaggle` cannot reach the local
service, it reads the same output directory's Alpha/status artifacts and writes
`glm52_kaggle_alpha_generate_cli.json` with public-safe artifact recovery phase,
blockers, cleanup/quota summary, `next_resume_command`, and
`resume_private_inputs`. The canonical r33 report imports that proof under
`generate_cli_summary` and `artifacts.generate_cli_json`; it records
`cli_generate_artifact_recovery_supported=true`,
`generate_cli_check_ok=true`, `artifact_recovery_present=true`,
`artifact_recovery_resume_command_present=true`, and
`artifact_recovery_resume_private_inputs_verified=true`. This is recovery
usability evidence, not live inference success.

r32 superseded r31 by exposing the public-safe `resume_private_inputs` recovery
contract through the ordinary-user status surfaces. The top-level Alpha report,
service summary, blocker report, HTTP `GET /status`, quota-blocked
`POST /generate`, and `crowdtensor status glm52-kaggle` now report that live
resume requires private Kaggle credentials and that the printed
`next_resume_command` omits private credential material. They expose only
public-safe metadata such as the contract schema, supported private input
methods, and Hugging Face env-name hash/count metadata. Credential values,
token file paths, token section names, raw env names, cookies, proxy URLs,
prompts, generated text, and private runtime state are not public. The checker
rejects artifacts missing this status recovery surface.

r30 made the ordinary-user default output directory
contract cover `serve glm52-kaggle` as well: `deploy`, `serve glm52-kaggle`,
`status`, and `cleanup` share `dist/glm52-kaggle-alpha` by default, while the
non-target product `serve` keeps its previous default. Service and Alpha
artifacts record `cli_serve_default_matches_deploy=true`,
`cli_status_default_matches_deploy=true` and
`cli_cleanup_default_matches_deploy=true`; the checker rejects artifacts that
drop those fields. r34 also retains the local HTTP service smoke proof:
`dist/glm52-kaggle-alpha-20260708-r34-http-cleanup-route/glm52_kaggle_alpha_service_smoke_probe.json`.
The smoke probe starts the real `AlphaHTTPServer`, verifies `GET /health`,
`GET /status`, `POST /generate`, and `POST /cleanup`, and the Alpha pack
imports the result as `service_smoke_summary` plus
`artifacts.service_smoke_json`. The imported summary records
`status_resume_private_inputs_verified=true`,
`generate_resume_private_inputs_verified=true`, and
`cleanup_route_verified=true`. In the current quota-blocked r34 state,
`/generate` reaches the service but returns
public-safe HTTP 503 with `generate_route_quota_blocker_verified=true` and
`generated_token_count=0`; no Kaggle live workers are launched by the smoke
path. This proves the service route surface and quota short-circuit, not live
multi-token inference.

r30 made the `serve` default output directory contract explicit, r29 made the
`status`/`cleanup` default output directory contract explicit,
r28 added the local HTTP service smoke proof, and r27 made Kaggle runtime
blocker classification an explicit recovery contract.
Live/collect worker paths classify push timeout/HTTP 429/empty response, status
timeout, wait timeout, output timeout/HTTP 429/empty response/missing stage
report, terminal error/cancelled, and cleanup/delete timeout into public-safe
blockers. Service and Alpha artifacts record
`kaggle_runtime_blocker_classification_ready=true` and the supported blocker
class list; the checker rejects artifacts missing this contract. r26 made the
HF token env contract explicit and public-safe. `--hf-token-env` is forwarded
from deploy/serve into the live probe and private Kaggle worker env; workers
add the Hugging Face Bearer header only at runtime. Service and Alpha artifacts
record HF env-name hashes/counts, configured booleans, and
`hf_token_public=false`, never the raw env names or token values. r31 added the
blocked-report `resume_private_inputs` contract itself, while r25 made the
user-requested model and accelerator contract explicit. Service and Alpha
artifacts record `requested_model`, `model_request_supported`, `accelerators`,
`required_accelerators`, and `accelerator_request_complete`; the checker
rejects unsupported model requests or incomplete accelerator requests. The r32
resume command preserves
`--model cyankiwi/GLM-5.2-AWQ-INT4 --accelerators cpu,gpu,tpu`. r23 exposed the
public-safe `next_resume_command` at the
top level of the main Alpha report as well as inside `blocker_report`; the
checker rejects blocked artifacts without that resume command or redaction
flag. r22 added the user-facing CLI generate path:
`crowdtensor generate --target glm52-kaggle ...` and
`crowdtensor generate glm52-kaggle --prompt-text ...`. r21 added public-safe
`/generate` request validation: `generate_validates_request_schema=true`.
Malformed JSON, empty/non-object bodies, missing prompts, and invalid token
counts return HTTP 400 and update service status to
`phase=generate_request_invalid` without calling the live probe. r20 added
startup-time `/status` recovery:
`status_loads_existing_alpha_artifacts=true`. When a service starts against an
output directory containing an existing public-safe `glm52_kaggle_alpha.json`,
`GET /status` immediately reports the stored blocker, phase status, cleanup
proof, and next resume command before any `/generate` request is made. r19
added the request-level `/generate` safeguard:
`generate_uses_current_gpu_quota_blocker=true`. If the output directory already
contains a still-current imported GPU quota blocker, `/generate` returns a
public-safe 503 response with blockers, cleanup proof, phase status, and the
next resume command instead of launching a doomed Kaggle live probe. r18 remains
the phase-status baseline, and r17 remains the benchmark-artifact baseline: it
wrote a separate
`glm52_kaggle_alpha_benchmark.json` artifact in addition to the embedded
benchmark summary. r31 imports the same public-safe r14 GPU quota evidence. It
is still a blocker, not an achieved Alpha.

Current service smoke check:

```bash
PYTHONPATH=. python scripts/glm52_kaggle_alpha_service_smoke_check.py \
  --report dist/glm52-kaggle-alpha-20260708-r34-http-cleanup-route/glm52_kaggle_alpha_service_smoke_probe.json \
  --require-verified \
  --json
```

Current status command for that blocker:

```bash
PYTHONPATH=. python -m crowdtensor.cli status glm52-kaggle \
  --output-dir dist/glm52-kaggle-alpha-20260708-r34-http-cleanup-route \
  --json
```

It reports `phase=decode_blocked`, cleanup verified, blockers including
`kaggle_gpu_quota_unavailable` and `live_report_missing`, and
`gpu_quota_status.source=alpha_gpu_quota_summary` with four authenticated GPU
accounts, zero accepted GPU submissions, and
`next_quota_refresh_time=2026-07-11T00:00:00`. The phase table reports
`phase_status.overall_state=blocked`, blocked phases
`gpu_quota_preflight`, `kernel_push`, and `gpu_queue_running`, and completed
phases `configuration_check`, `model_source_check`, and `cleanup_completed`.

Current cleanup command for that blocker:

```bash
PYTHONPATH=. python -m crowdtensor.cli cleanup glm52-kaggle \
  --output-dir dist/glm52-kaggle-alpha-20260708-r34-http-cleanup-route \
  --json
```

It reports `ok=true`, `cleanup_evidence_source=service_status`,
`cleanup_mode=gpu_quota_preflight_skipped_live`, no live resources left running,
and no public credentials or private runtime data.

Default checker:

```bash
PYTHONPATH=. python scripts/glm52_kaggle_alpha_check.py \
  --report dist/glm52-kaggle-alpha-20260708-r34-http-cleanup-route/glm52_kaggle_alpha.json \
  --json
```

Strict readiness checker:

```bash
PYTHONPATH=. python scripts/glm52_kaggle_alpha_check.py \
  --report dist/glm52-kaggle-alpha-20260708-r34-http-cleanup-route/glm52_kaggle_alpha.json \
  --require-ready \
  --json
```

As of r14, service/CLI/checker engineering is present and the 39-stage
same-request topology has a unique-slug package that runs with 7 Kaggle
kernels: 1 CUDA, 1 TPU, and 5 CPU group kernels. Runtime tuning is passed
through private Kaggle env and is recorded public-safely. Deploy now supports
`--gpu-quota-preflight`; when all authenticated GPU accounts are exhausted it
skips the live run before launching GLM workers and writes a checker-passing
blocker. The current preflight authenticated all four known GPU-capable
accounts (`tpuowner`, `primary Kaggle account`, `cpuowner`, and `gpuowner`) and all are
weekly GPU quota exhausted until `2026-07-11T00:00:00`. The previous r8 attempt
remains throughput evidence: it reached all 39 stage workers and completed
stages 0, 1, and 2, but generated 0/8 tokens.
