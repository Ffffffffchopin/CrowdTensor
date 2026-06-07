# CrowdTensor

CrowdTensor is an open-source experiment in swarm inference: splitting small
real model workloads across ordinary machines, while keeping routing,
validation, failure recovery, and evidence auditable.

The current milestone is **Public Real-LLM Swarm Inference Beta**. Today the
project can run a real tiny Hugging Face GPT model through a Coordinator-backed
two-stage swarm:

```text
p2pd -> serve --p2p -> join stage0 + join stage1 -> generate --p2p
```

That path proves the mechanics needed for larger open AI infrastructure:
peer discovery, stage-aware scheduling, lease recovery, result validation,
redacted evidence, and operator controls.

CrowdTensor is usable as an engineering beta. It is not yet production Swarm
Inference, not a full Hivemind/Petals replacement, not large-model serving, and
not a permissionless P2P network.

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

Run the fast local proof first:

```bash
crowdtensor local-proof --json
```

Run the user-friendly local swarm inference entry point:

```bash
crowdtensor infer "CrowdTensor routes small models across home compute"
```

It starts the fast local product loopback path, runs split tiny GPT inference,
prints the local display-only generated text, and writes a compact
`infer_summary.json` plus a safe `infer_summary.md` under `dist/infer`. JSON,
Markdown, and public artifacts keep raw prompts, generated text, token ids,
credentials, and activations out of shareable files. Use `--full-evidence`
when you want the broader Public Swarm v2 gate instead of the faster user path.
In human mode, `infer` and `generate` print a short safe stderr start hint before
long-running checks so the terminal does not look idle; `--json` keeps stdout
machine-readable.
When generated text is available in human mode, the terminal prints it as
`answer:` or `answer[n]:` before `answer_scope` and `local_output` safety
metadata. `answer_scope.scope_state` uses stable values such as
`terminal-visible`, `saved-terminal-redacted`, `json-suppressed`, and
`no-local-answer`; the Markdown `What To Do Next` and `Details` sections repeat
that saved JSON and Markdown contain no generated text. `local_output` includes
safe output `count` and `source` fields such as
`local-private-task-state` or `coordinator-validation`.
Pick one prompt source per command: use the positional prompt,
`--prompt-text`/`--prompt`, or `--prompt-texts` for a bounded batch. The CLI
rejects mixed prompt sources instead of guessing. Reports expose
`output_request.include_output` while keeping
`output_request.raw_generated_text_public` false in JSON and saved artifacts.
Existing-swarm runs also include a safe `wait_progress` summary with
poll count, accepted rows, endpoint readiness, observed token progress, batch
request progress, and safe last-error type so timeouts are actionable without
exposing raw text; both `infer` and `generate` include `operator_action`
suggestions for checking tokens, Miner health, admin API access, or timeout
limits. Live and summary stream progress use safe request ids or hash prefixes,
print per-request token/target progress for bounded batch streams, mark missing
stream slots, print `stream_issue` when a request is missing or incomplete, and
print `recommended_next` plus `next[...]` lines with safe follow-up commands.
The `trace` line in human output and the `trace` object in JSON/Markdown give a
safe troubleshooting summary: session id, request count, accepted ledger rows,
stream event count, and per-request ids or prompt hashes. It never includes raw
prompt text, generated text, generated token ids, credentials, or activations.
The `result` line and JSON/Markdown `result` object summarize completion state,
token count, output count, generated-text hash, and display safety:
`local-private` for terminal-only generated text, `hash-only` for redacted
summaries, and `hash-only-json` for JSON stdout, without exposing generated text
in shareable artifacts.
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
with your current local prompt for copying, while JSON/Markdown artifacts keep
prompt placeholders. Then use the `status` line or
`user_status` for detail: `completed` means the request finished,
`preflight-ready` means submit next, `preflight-partial` means run the
recommended check first, and `blocked` means follow `action` /
`recommended_next`. Human `infer` and `generate` output use your current local
prompt so the next command is directly copyable; JSON reports and saved
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
--dry-run` as an offline request-shape check.

For maintainer-grade release evidence, run the full public swarm beta gate:

```bash
crowdtensor public-real-llm-swarm-beta release \
  --max-new-tokens 16 \
  --http-timeout 30 \
  --json
```

This runs the stricter release aggregate and checks retained external evidence,
route hardening, failure requeue, KV-cache readiness, and artifact safety.
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
