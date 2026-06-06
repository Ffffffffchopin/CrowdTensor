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
`infer_summary.json` under `dist/infer`. JSON mode and saved reports keep raw
prompts and generated text redacted. Use `--full-evidence` when you want the
broader Public Swarm v2 gate instead of the faster user path.

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
redacted. Existing-swarm reports include `wait_progress` with poll count,
accepted rows, endpoint readiness, and observed token progress for safe timeout
debugging; `infer` and `generate` turn that progress into a concrete
`operator_action`. Human output also includes `next[...]` lines with safe,
copyable follow-up commands. Human `infer` and `generate` output use your
current local prompt in those next commands; JSON reports and saved artifacts
keep raw prompts and token values represented as placeholders.
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
Open five terminals from the repository root.
The `serve`, `join`, and `generate` commands print an `action` line in human
mode; follow it when a step is only printing a command, missing a route, or
waiting for the other stage Miner.

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
  --dry-run
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
capabilities. If the command is being used only for CI-safe packaging or
offline request-shape checks, add `--skip-live-preflight`.

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
- Start `serve` before Miners.
- Use the same `--swarm-id` in every terminal.
- Check that stage 0 and stage 1 use different `--miner-id` values.

## Boundaries

The quickstart proves a controlled small-model swarm route. It does not prove
production uptime, permissionless public mining, large-model sharding,
Hivemind-level serving, or a tokenized compute marketplace.
