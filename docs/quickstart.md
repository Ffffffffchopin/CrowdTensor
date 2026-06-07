# Quickstart

This guide gets you from a fresh checkout to a local CrowdTensor swarm proof.
It uses the current Public Real-LLM Swarm Inference Beta: a small real Hugging
Face GPT model split across two stage Miners behind a Coordinator-backed route.

CrowdTensor is still an engineering beta. The commands below are for controlled
local or trusted-network experiments, not production public serving.

## 1. Install

Use Python 3.11 or newer.

```bash
git clone https://github.com/Ffffffffchopin/CrowdTensor.git
cd CrowdTensor

python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,hf]'
```

Check that the CLI is available:

```bash
crowdtensor --help
crowdtensord --help
crowdtensor-miner --help
```

If you do not want Hugging Face dependencies yet, install only the local CPU
path:

```bash
python -m pip install -e '.[dev]'
```

## 2. Run A Fast Local Proof

The fastest confidence check is the one-command local proof:

```bash
crowdtensor local-proof --json
```

It chains local diagnostics, runtime capability checks, CPU-only inference, and
a demo manifest. This does not require public networking.

For the CPU-only inference aggregate:

```bash
crowdtensor cpu-infer --mode local --json
```

## 3. Run User-Friendly Swarm Inference

With `[hf]` installed, the shortest user-facing inference path is:

```bash
crowdtensor infer "CrowdTensor routes small models across home compute"
```

It starts the fast local product loopback route, runs tiny GPT split inference,
prints the local display-only generated text, and writes a compact
`infer_summary.json` plus a safe `infer_summary.md` under `dist/infer`. JSON
mode, Markdown, and saved reports keep raw prompts and generated text redacted.
Use `--full-evidence` when you want the broader Public Swarm v2 gate instead
of the faster user path.

For machine-readable output:

```bash
crowdtensor infer \
  "CrowdTensor routes small models across home compute" \
  --max-new-tokens 8 \
  --json
```

`crowdtensor infer --mode existing` can target an already running Coordinator or
P2P-discovered swarm with `--coordinator-url` or `--peer-bootstrap`.
Add `--dry-run` to check the route and session metadata before submitting a
real inference request; it also checks the Coordinator `/ready` endpoint. Pass
`--observer-token "$CROWDTENSOR_OBSERVER_TOKEN"` when you want the dry run to
read `/state` and verify visible stage0/stage1 Miner capabilities.
Use `--prompt-texts "first prompt,second prompt"` for a bounded local batch;
human output prints each result separately while JSON reports keep raw text
redacted. When human output shows generated text, it prints `answer:` or
`answer[n]:` before `local_output` safety metadata with
safe output `count` and `source` fields such as `local-private-task-state` or
`coordinator-validation`.
Pick one prompt source per command: positional prompt,
`--prompt-text`/`--prompt`, or `--prompt-texts` for a bounded
batch. The CLI rejects mixed prompt sources instead of guessing. Reports expose
`output_request.include_output` while keeping
`output_request.raw_generated_text_public` false in JSON and saved artifacts.
Existing-swarm reports include `wait_progress` with poll count, accepted rows,
endpoint readiness, observed token progress, batch request progress, and safe
last-error type for timeout debugging; `infer` and `generate` turn that progress
into a concrete `operator_action`.
Live and summary stream progress use safe request ids or hash prefixes, include
per-request token/target progress for bounded batch streams, mark missing stream
slots, print `stream_issue` when a request is missing or incomplete, and print
`recommended_next` plus `next[...]` lines with safe, copyable follow-up commands.
The `trace` line and JSON/Markdown `trace` object summarize session id,
requests, accepted ledger rows, stream event count, and safe
per-request ids or prompt hashes. It never exposes raw prompts, generated text,
generated token ids, credentials, or activations.
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
such as incomplete stream evidence. The adjacent `review_next` line repeats the
safe recommended command near that summary; human terminal output renders it
with your current local prompt for copying, while JSON/Markdown artifacts keep
prompt placeholders. Then use the `status` line or
`user_status` for detail: `completed` means the request finished,
`preflight-ready` means submit next, `preflight-partial` means run the
recommended check first, and `blocked` means follow `action` /
`recommended_next`. Human `infer` and `generate` output use your current local
prompt in those next commands; JSON reports and saved artifacts keep raw prompts
and token values represented as placeholders.
If `ready_to_submit` is printed, use its `label` and `next_step` before submitting:
`verified` means route, Coordinator, and distinct stage Miners were checked;
`partial` means the request shape can submit but stage Miners still need
observer-token verification; `blocked` means follow `operator_action`; and
`skipped` means live checks were intentionally bypassed. JSON reports mirror
that status with `crowdtensor_infer_preflight_partial` for partial existing
swarm checks and `crowdtensor_infer_preflight_ready` only after full route,
Coordinator, and stage Miner verification. `stage_preflight_not_checked` means
the route or Coordinator check failed before stage Miners could be inspected;
fix the printed blocker, then rerun with `CROWDTENSOR_OBSERVER_TOKEN`.
Add `--stream` when you want safe token-progress evidence in the CLI summary.

## 4. Run The Real-LLM Swarm Beta Gate

For maintainer-grade release evidence, run the stricter real-model beta gate:

```bash
crowdtensor public-real-llm-swarm-beta release \
  --max-new-tokens 16 \
  --http-timeout 30 \
  --json
```

This command starts local stand-ins for the public swarm route, runs a tiny real
GPT split across stage 0 and stage 1, validates decoded tokens, checks evidence,
and writes artifacts under `dist/`.

Useful readiness fields in the JSON output include:

- `public_real_llm_swarm_beta_ready`
- `public_swarm_v2_ready`
- `real_llm_split_route_ready`
- `decoded_tokens_match`
- `distinct_stage_miners`
- `stage_assignment_valid`

## 5. Run The Manual Five-Process Demo

The release gate is convenient, but the manual flow shows the moving pieces.
Open five terminals from the repository root. Use the same local tokens in
terminals 2-5:

```bash
export CROWDTENSOR_ADMIN_TOKEN=local-admin
export CROWDTENSOR_MINER_TOKEN=local-miner
export CROWDTENSOR_OBSERVER_TOKEN=local-observer
```

The `serve`, `join`, and `generate` commands print an `action` line in human
mode; follow it when a step is only printing a command, missing a route, or
waiting for the other stage Miner. If a printed `next[...]` command ends with
`# requires CROWDTENSOR_...`, export those environment variables before copying
the command; token and peer-secret values are deliberately kept out of
shareable reports. The default quickstart uses P2P-lite with
`crowdtensor p2pd`. If you run the real provider-discovery preview with
`--p2p-backend real`, follow the printed `crowdtensor p2p-daemon` fallback
command instead.
`generate` also writes safe `generate_summary.json` and
`generate_summary.md` files under `dist/generate` by default; raw prompts,
generated text, token ids, and tokens stay out of those shareable artifacts.

```bash
# Terminal 1
crowdtensor p2pd --swarm-id public-swarm-v2 --run
```

```bash
# Terminal 2
crowdtensor serve --p2p --swarm-id public-swarm-v2 --run
```

```bash
# Terminal 3
crowdtensor join --stage stage0 --p2p --swarm-id public-swarm-v2 --miner-id stage0 --run
```

```bash
# Terminal 4
crowdtensor join --stage stage1 --p2p --swarm-id public-swarm-v2 --miner-id stage1 --run
```

```bash
# Optional preflight before submitting the real request.
crowdtensor generate \
  --p2p \
  --swarm-id public-swarm-v2 \
  --prompt "CrowdTensor routes small models across home compute" \
  --max-new-tokens 16 \
  --http-timeout 30 \
  --dry-run \
  --observer-token "$CROWDTENSOR_OBSERVER_TOKEN"
```

```bash
# Terminal 5
crowdtensor generate \
  --p2p \
  --swarm-id public-swarm-v2 \
  --prompt "CrowdTensor routes small models across home compute" \
  --max-new-tokens 16 \
  --http-timeout 30
```

Expected behavior:

- The Coordinator creates a read-only inference session.
- Stage 0 and stage 1 claim their stage-specific work.
- The client receives the generated tokens.
- The route records evidence for assignment and validation.

`generate --dry-run` prints `coordinator_ready`, `stage_preflight`, and
`ready_to_submit` when it can check live endpoints or P2P-discovered stage
capabilities. Treat `label=verified` as the normal submit-ready state;
`label=partial` needs an observer-token preflight; `label=blocked` needs the
printed action first. A false `coordinator_ready` line includes `error=...`
for failed probes or `reason=...` for skipped checks. For
`generate --dry-run`, JSON uses `generate_dry_run_partial` for any partial
readiness state. For `infer --mode existing --dry-run`, JSON uses
`crowdtensor_infer_preflight_partial` until the stage Miner check is fully
verified. If the command is being used only for CI-safe packaging or offline
request-shape checks, add `--skip-live-preflight` and expect `label=skipped`.
Those skipped checks emit `generate_request_shape_ready`, not
`generate_dry_run_ready`, because the route has not been proven submit-ready.
Submit command labels also reflect this state: `after live preflight`,
`after stage preflight`, or `after checks pass` means run the printed check
command before submitting; `with caution` means the request can run but not
every live check was proven. The same decision is available as
`ready_to_submit.next_step` for scripts and support tools, with stable values
such as `submit`, `run_stage_preflight`, `run_live_preflight`,
`submit_with_caution`, and `fix_blockers`. `stage_preflight_unknown` means a
required stage check did not return a true/false result. If
`stage_preflight_not_checked` appears, fix the printed route or Coordinator
blocker, then rerun the dry-run with `CROWDTENSOR_OBSERVER_TOKEN` before
submitting. Session-create failure details are redacted before they are printed
or written to JSON, including prompt text and token values echoed by a remote
endpoint.

## 6. Package A Controlled Remote Trial

For a two-machine style rehearsal, generate the package first:

```bash
crowdtensor public-real-llm-swarm-beta package \
  --output-dir dist/public-real-llm-package \
  --json
```

Use the generated runbook and private env files only on trusted hosts. For real
machines, put the Coordinator behind a trusted network boundary such as LAN,
VPN, tunnel, or explicit firewall rules. Rotate temporary tokens after demos.

## 7. Optional CUDA Tiny-Model Path

CUDA is opt-in and only applies to the tiny real-model stage runtime. It should
fail closed when CUDA is unavailable.

```bash
crowdtensor public-real-llm-swarm-beta release \
  --public-swarm-v2-backend cuda \
  --max-new-tokens 16 \
  --http-timeout 30 \
  --json
```

CPU remains the default path.

## 8. Clean Generated Artifacts

Dry-run cleanup:

```bash
crowdtensor clean-artifacts
```

Apply cleanup:

```bash
crowdtensor clean-artifacts --apply
```

Reports are kept by default. Add `--include-reports` only when you explicitly
want generated report files removed.

## Troubleshooting

If a command fails before starting:

- Confirm the virtualenv is active.
- Run `python -m pip install -e '.[dev,hf]'` again.
- Check `crowdtensor --help` to confirm the CLI points at this checkout.

If Hugging Face or torch dependencies are unavailable:

- Use `crowdtensor local-proof --json`.
- Use `crowdtensor cpu-infer --mode local --json`.
- Reinstall with `[hf]` before running real-model split demos.

If a multi-process demo hangs:

- Start `p2pd` first.
- If you selected `--p2p-backend real`, start `p2p-daemon` first instead.
- Start `serve` before Miners.
- Use the same `--swarm-id` in every terminal.
- Check that stage 0 and stage 1 use different `--miner-id` values.
- Set any `CROWDTENSOR_...` variables named in printed `# requires` hints.

## Boundaries

The quickstart proves a controlled small-model swarm route. It does not prove
production uptime, permissionless public mining, large-model sharding,
Hivemind-level serving, or a tokenized compute marketplace.
