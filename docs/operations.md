# Operations

This document collects common commands for local Alpha operation.

## Public Swarm Inference v2

Use this first when validating the current public-preview inference path. It uses P2P discovery for route lookup, keeps the Coordinator as the execution authority, and validates real small Hugging Face split inference with distinct stage Miners.

```bash
python -m pip install -e '.[hf]'
read -r -s -p 'Admin token: ' CROWDTENSOR_ADMIN_TOKEN; echo
read -r -s -p 'Miner token: ' CROWDTENSOR_MINER_TOKEN; echo
export CROWDTENSOR_ADMIN_TOKEN CROWDTENSOR_MINER_TOKEN
```

Run the local v2 gate:

```bash
crowdtensor public-swarm-v2 local --max-new-tokens 16 --http-timeout 30 --json
crowdtensor public-swarm-v2 local-model-variant --hf-model-id distilgpt2 --max-new-tokens 16 --stream-generation --json
python scripts/public_swarm_inference_v2_check.py --mode local --json
python scripts/public_swarm_inference_v2_check.py --mode local-model-variant --hf-model-id distilgpt2 --json
```

The v2 report is `public_swarm_inference_v2`. A ready local report requires 16 generated tokens, route source `p2p-discovery`, distinct stage0/stage1 Miners, 32 accepted stage rows, stage rescue/requeue evidence, fresh local real-P2P stage requeue evidence, latency/throughput/memory summaries, model-consistent local/external/P2P evidence, optional CUDA evidence or fail-closed diagnostics, and redacted JSON/Markdown/support outputs under `dist/public-swarm-inference-v2`. Non-default `--hf-model-id` imports must expose the same `hf_model_id` across the Usable v1 and signed/real P2P reports; otherwise v2 emits model mismatch diagnostics and blocks readiness.

The top-level JSON, Markdown, terminal output, and support bundle include `output_request`, `answer_scope.scope_state: no-local-answer`, and `shareable_summary.answer_scope_state`. Treat v2 as shareable readiness evidence, not a local answer transcript; run `crowdtensor generate --p2p` in human mode to view local generated text.

Use `local-model-variant` for a non-default local small-model proof without claiming retained external/Kaggle validation. A ready report emits `public_swarm_inference_v2_local_model_variant_ready` plus `public_swarm_v2_external_validation_not_claimed`, preserves local model compatibility, and keeps external evidence as an unclaimed comparison.

The current retained local requeue proof is `dist/goal-final-infer-public-swarm-v2-local-real-p2p-requeue-batch-stream-20260602/public_swarm_inference_v2.json`. It proves a local real-P2P stage1 victim claim, victim process termination, lease-timeout requeue, rescue Miner acceptance, and `victim_result_accepted: false`, then surfaces `public_swarm_v2_real_p2p_local_requeue_ready`, `real_p2p_local_stage_requeue_ready`, `local_stage_requeue_ready`, and `stage_requeue_ready`.

Manual five-process operation:

```bash
crowdtensor p2pd --swarm-id public-swarm-v2 --run
crowdtensor serve --p2p --swarm-id public-swarm-v2 --run
crowdtensor join --stage stage0 --p2p --swarm-id public-swarm-v2 --miner-id stage0 --run
crowdtensor join --stage stage1 --p2p --swarm-id public-swarm-v2 --miner-id stage1 --run
crowdtensor generate --p2p --swarm-id public-swarm-v2 --prompt "CrowdTensor routes small models across home compute" --max-new-tokens 16 --http-timeout 30
```

The CLI keeps token and peer-secret values out of reports. When a printed
`next[...]` command includes `# requires CROWDTENSOR_ADMIN_TOKEN`,
`CROWDTENSOR_MINER_TOKEN`, `CROWDTENSOR_OBSERVER_TOKEN`, or
`CROWDTENSOR_P2P_PEER_SECRET`, export the named variables before copying the
command. The default route uses P2P-lite and `crowdtensor p2pd`; if you run
`serve` / `join` / `generate` with `--p2p-backend real`, blocked discovery
reports should point at the matching `crowdtensor p2p-daemon` fallback.

For a two-machine or Kaggle rehearsal, keep `p2pd` and `serve` on the Coordinator host and point stage Miners at the same bootstrap:

```bash
export COORDINATOR_PUBLIC_HOST='<public-host-or-vpn-hostname>'
crowdtensor p2pd --host 0.0.0.0 --port 9888 --swarm-id public-swarm-v2 --run
crowdtensor serve --p2p --peer-bootstrap "http://$COORDINATOR_PUBLIC_HOST:9888" --swarm-id public-swarm-v2 --bind-host 0.0.0.0 --public-host "$COORDINATOR_PUBLIC_HOST" --port 9889 --i-understand-public-bind --run
```

On distinct Miner hosts or private Kaggle notebooks:

```bash
crowdtensor join --stage stage0 --p2p --peer-bootstrap "http://$COORDINATOR_PUBLIC_HOST:9888" --swarm-id public-swarm-v2 --miner-id "$(hostname)-stage0" --run
crowdtensor join --stage stage1 --p2p --peer-bootstrap "http://$COORDINATOR_PUBLIC_HOST:9888" --swarm-id public-swarm-v2 --miner-id "$(hostname)-stage1" --run
crowdtensor generate --p2p --peer-bootstrap "http://$COORDINATOR_PUBLIC_HOST:9888" --prompt "CrowdTensor routes small models across home compute" --max-new-tokens 16 --http-timeout 30
```

Use `--hf-model-id <model>` on `generate`, `serve`, `join`, and the relevant packer gates when running a non-default small Hugging Face model. The public request and evidence layers carry the model id while still redacting raw prompts, generated text, token ids, activations, and credentials.

For maintainer automation of a fresh external real-P2P proof, run the real-P2P Kaggle pack, then import it into v2:

```bash
python scripts/real_p2p_swarm_inference_core_rc_pack.py kaggle-auto \
  --discovery-backend libp2p-kad \
  --public-host 24.199.118.54 \
  --p2p-port 9888 \
  --coordinator-port 9889 \
  --max-new-tokens 16 \
  --timeout-seconds 900 \
  --generate-timeout 900 \
  --http-timeout 30 \
  --json

crowdtensor public-swarm-v2 evidence-import \
  --fresh-external-report \
  --real-p2p-report dist/<fresh-real-p2p-run>/real_p2p_swarm_inference_core_rc.json \
  --max-new-tokens 16 \
  --json
```

When using a non-default model, pass the same `--hf-model-id` to the Real P2P proof and the v2 import. Real P2P `external-existing --verify-generate` forwards that model id, optional bounded `--prompt-texts`, and optional `--stream-generation` into nested `generate`, then reports public-safe model/batch/stream summaries so later imports can enforce model consistency without exposing prompts or generated text.

If the fresh external run fails before generation completes, delete temporary Kaggle kernels, rotate tokens, write a redacted `public_swarm_v2_fresh_external_attempt_v1` summary, and import it without claiming fresh success:

```bash
crowdtensor public-swarm-v2 evidence-import \
  --fresh-external-attempt-report dist/<fresh-run>/fresh_external_attempt.json \
  --max-new-tokens 16 \
  --json
```

Use a trusted network, VPN, SSH tunnel, or temporary firewall allowlist. Put only Miner tokens in external notebooks. Keep admin tokens on the operator host, delete temporary external resources, and rotate tokens after public HTTP tests.

This remains Coordinator-backed, read-only, tiny/small-model scoped, CPU by default, optional CUDA only, not production Hivemind/Petals parity, not Coordinator-free execution, not production NAT traversal, and not large-model serving.

## Usable Swarm Inference v1

This is the previous user-facing inference path. It uses P2P discovery for route lookup, while the Coordinator still owns sessions, task leases, validation, and result ledgers.

```bash
python -m pip install -e '.[hf]'
read -r -s -p 'Admin token: ' CROWDTENSOR_ADMIN_TOKEN; echo
read -r -s -p 'Miner token: ' CROWDTENSOR_MINER_TOKEN; echo
export CROWDTENSOR_ADMIN_TOKEN CROWDTENSOR_MINER_TOKEN
```

Run the local five-process path:

```bash
crowdtensor p2pd --run
crowdtensor serve --p2p --run
crowdtensor join --stage stage0 --p2p --miner-id stage0 --run
crowdtensor join --stage stage1 --p2p --miner-id stage1 --run
crowdtensor generate --p2p --prompt "CrowdTensor routes small models across home compute" --max-new-tokens 8
```

Follow the printed `action` and `next[...]` lines when a step blocks. Required
token or peer-secret values are surfaced as `# requires CROWDTENSOR_...`
environment hints rather than embedded in shareable output. P2P-lite uses
`crowdtensor p2pd`; the real provider-discovery preview uses
`crowdtensor p2p-daemon` when commands are run with `--p2p-backend real`.

For a two-machine or public-host rehearsal, keep `p2pd` and `serve` on the Coordinator host and point stage Miners at the same bootstrap:

```bash
export COORDINATOR_PUBLIC_HOST='<public-host-or-vpn-hostname>'
crowdtensor p2pd --host 0.0.0.0 --port 9788 --swarm-id usable-swarm-v1 --run
crowdtensor serve --p2p --peer-bootstrap "http://$COORDINATOR_PUBLIC_HOST:9788" --swarm-id usable-swarm-v1 --bind-host 0.0.0.0 --public-host "$COORDINATOR_PUBLIC_HOST" --port 9789 --i-understand-public-bind --run
```

On distinct Miner hosts, or at minimum distinct Miner identities:

```bash
crowdtensor join --stage stage0 --p2p --peer-bootstrap "http://$COORDINATOR_PUBLIC_HOST:9788" --swarm-id usable-swarm-v1 --miner-id "$(hostname)-stage0" --run
crowdtensor join --stage stage1 --p2p --peer-bootstrap "http://$COORDINATOR_PUBLIC_HOST:9788" --swarm-id usable-swarm-v1 --miner-id "$(hostname)-stage1" --run
crowdtensor generate --p2p --peer-bootstrap "http://$COORDINATOR_PUBLIC_HOST:9788" --prompt "CrowdTensor routes small models across home compute" --max-new-tokens 8
```

Use a trusted network, VPN, SSH tunnel, or temporary firewall allowlist. Rotate `CROWDTENSOR_ADMIN_TOKEN` and `CROWDTENSOR_MINER_TOKEN` after public HTTP tests.

For a Kaggle CPU Miner rehearsal, treat Kaggle as the external Miner host rather than a Coordinator host. Start `p2pd` and `serve` on your public or VPN-reachable Coordinator host as above, create a private Kaggle notebook with this repository installed, set the miner token as a private Kaggle secret or notebook environment value, then run exactly one stage command per notebook:

```bash
export COORDINATOR_PUBLIC_HOST='<public-host-or-vpn-hostname>'
read -r -s -p 'Miner token: ' CROWDTENSOR_MINER_TOKEN; echo
export CROWDTENSOR_MINER_TOKEN
crowdtensor join --stage stage0 --p2p --peer-bootstrap "http://$COORDINATOR_PUBLIC_HOST:9788" --swarm-id usable-swarm-v1 --miner-id "kaggle-stage0" --run
```

Use a second private Kaggle notebook for `--stage stage1 --miner-id "kaggle-stage1"`. Then run `crowdtensor generate --p2p` from the operator host. Do not put `CROWDTENSOR_ADMIN_TOKEN` in Kaggle Miner notebooks; only the Miner token is needed there.

Maintainer acceptance:

```bash
python scripts/user_friendly_inference_frontdoor_check.py --json
crowdtensor usable-swarm local --max-new-tokens 8 --json
python scripts/usable_swarm_inference_check.py --mode local --json
python scripts/usable_swarm_inference_check.py --mode local --hf-model-id distilgpt2 --json
```

`scripts/user_friendly_inference_frontdoor_check.py` is CI-safe and does not
start a Coordinator, submit a live task, or create Kaggle resources. It builds
fake completed `infer` and `generate` front-door reports through the real CLI
report writers, then validates the saved `infer_summary` and `generate_summary`
JSON/Markdown contracts: terminal answers may be visible locally, saved
artifacts must say `saved-terminal-redacted`, raw prompts/generated text/token
ids must stay out, and `fresh_kaggle_gpu_verified` must remain false.

The `usable_swarm_inference_v1` report requires `p2pd`, `serve --p2p`, distinct `join --p2p` stage0/stage1 Miners, real small HF split generation, at least 8 generated tokens, accepted ledger rows for both stages, local stage rescue evidence, and `usable_swarm_model_match_ready`. Non-default `--hf-model-id` evidence imports must expose the same `hf_model_id` in the P2P v0.6 report; otherwise `usable_swarm_model_mismatch` blocks readiness. Public artifacts redact raw prompts, generated text, generated token ids, activations, credentials, leases, and idempotency material. The top-level JSON, Markdown, terminal output, and support bundle include `output_request`, `answer_scope.scope_state: no-local-answer`, and `shareable_summary.answer_scope_state` so operators know the aggregate is shareable evidence rather than a local answer transcript; run `crowdtensor generate --p2p` in human mode to see local generated text. This path is Coordinator-backed, read-only, tiny/small-model scoped, CPU by default, and not full Hivemind/Petals production parity, Coordinator-free execution, production NAT traversal, or large-model throughput serving.

## Fresh Clone Onboarding Gate

```bash
python scripts/onboarding_gate.py --quick --json-out /tmp/crowdtensor_onboarding_gate.json
```

The onboarding gate emits `onboarding_gate_v1`. It creates a clean temporary virtualenv, runs `python -m pip install -e .[dev,hf]`, checks `crowdtensor --help`, `crowdtensord --help`, and `crowdtensor-miner --help`, then smoke-validates `scripts/user_friendly_inference_frontdoor_check.py`, the real user entrypoint `crowdtensor infer --prompt-stdin --shareable-terminal`, `crowdtensor local-proof`, `crowdtensor home-infer`, `crowdtensor llm-infer --mock`, `crowdtensor cpu-infer --mode local`, and `crowdtensor release-ready --allow-dirty` with reduced request counts. The `user_infer_smoke` step reads the prompt from stdin, writes `infer_summary.json` / `infer_summary.md`, and validates `answer=shareable-terminal-redacted`, `gpu=local-cpu-only`, and `fresh_kaggle_gpu=False` without saving the raw prompt or generated answer.

Use this when validating a fresh checkout or CI image before longer runtime acceptance. It writes reports under `/tmp` by default, removes the temporary venv unless `--keep-venv` is passed, and is not production Swarm Inference readiness.

## Release Readiness

```bash
crowdtensor release-ready --json
```

The `crowdtensor/cli.py` release readiness entrypoint wraps `scripts/release_readiness_pack.py`, emits `release_readiness_v1`, and writes JSON/Markdown under `dist/release-readiness`. It aggregates Git metadata, the release gate, security preflight, and `demo_manifest_v1`, then reports blocker diagnosis codes such as `git_dirty`, `release_gate_failed`, and `demo_manifest_failed`.

Dirty worktrees block by default so maintainers do not accidentally tag a mixed checkout. Use `--allow-dirty` only for development smoke checks, including `scripts/release_readiness_check.py --allow-dirty` in CI. Missing runtime, browser, or remote acceptance reports are warnings unless provided and failing. This is not production Swarm Inference readiness; it is an Alpha maintainer gate for the current CPU-only repository state.

## One-Command Local Proof

```bash
crowdtensor local-proof --json
```

The `crowdtensor/cli.py` user entrypoint emits `local_proof_summary_v1` and writes artifacts under `dist/local-proof`. It runs Doctor, the runtime matrix, the CPU-only read-only home-compute demo, and the Demo Manifest path as a single local proof for a checkout. Use it before longer acceptance runs when a new operator needs one concise artifact.

This command is not production Swarm Inference, not arbitrary prompt serving, and not a GPU/P2P/WebGPU path. It only proves the current local Alpha control-plane route.

## Home Inference Proof

For the aggregate CPU inference Beta proof:

```bash
crowdtensor cpu-infer --mode local --json
```

The CPU inference Beta command emits `cpu_inference_beta_v1` through `scripts/cpu_inference_beta_pack.py`. `--mode local` wraps `home-infer` and deterministic `llm-infer --mock`; `--mode remote-loopback` validates the local remote-demo stand-in for `model-bundle` or `external-llm`; `--mode remote-existing` wraps a running two-machine `remote-demo doctor/verify/collect` flow with explicit observer/admin tokens. CI validates this with `scripts/cpu_inference_beta_check.py`. It is CPU-only, read-only, not production Swarm Inference, not P2P, and not arbitrary prompt serving.

For the CPU Inference Beta RC release-candidate aggregate:

```bash
crowdtensor cpu-infer --mode beta-rc --json
```

This emits `cpu_inference_beta_rc_v1` through `scripts/cpu_inference_beta_rc_pack.py` and writes JSON/Markdown under `dist/cpu-infer`. It aggregates the local CPU inference Beta, remote-loopback CPU inference Beta, Real two-machine CPU inference Beta rehearsal, Kaggle Remote Miner Beta artifacts, the generated `miner_join_pack_v1`, `scripts/kaggle_remote_miner_beta_check.py`, and `demo_manifest_v1`. CI validates the path with `scripts/cpu_inference_beta_rc_check.py`; a ready report includes `cpu_inference_beta_rc_ready`, `local_cpu_inference_ready`, `remote_loopback_ready`, `two_machine_rehearsal_ready`, `kaggle_remote_miner_artifacts_ready`, `miner_join_pack_ready`, and `cpu_miner_beta_ready`. Pass `--kaggle-real-runtime-report dist/kaggle-real-runtime/kaggle_real_runtime_acceptance.json` to import a completed live `kaggle_real_runtime_acceptance_v1` report and surface `real_runtime_evidence_ready`. It is CPU-only, read-only, not production Swarm Inference, not P2P, not a GPU/TPU workload path, and not arbitrary prompt serving.

For the Real two-machine CPU inference Beta aggregate rehearsal:

```bash
python scripts/remote_two_machine_beta_check.py --workload all --base-port 9050
```

This emits `remote_two_machine_beta_check_v1` and verifies the 15-minute two-machine CPU inference Beta loopback stand-in for the Coordinator host and Miner host. It requires `remote_two_machine_inference_ready`, `remote_two_machine_external_llm_ready`, and `remote_two_machine_beta_ready`, and it checks redaction boundaries. Real two-machine operation still needs operator-provided TLS, VPN, tunnel, or trusted network. It is task-level remote CPU inference, not model sharding, not P2P, and not production Swarm Inference.

For the CPU-only Pipeline-Sharded Inference Alpha:

```bash
crowdtensor shard-infer --json
```

This emits `sharded_inference_cli_v1`, wraps `scripts/sharded_inference_evidence_pack.py`, and writes `sharded_inference_evidence_v1` artifacts under `dist/shard-infer`. The `sharded_model_bundle_infer` / `sharded_model_bundle_infer_v1` workload creates a `sharded_inference_session_v1` with two fixed stages: stage 0 produces activation hashes/byte counts, stage 1 consumes the accepted activation and must match the single-task baseline. CI validates this with `scripts/sharded_inference_check.py` and diagnosis codes `stage_0_accepted`, `stage_1_accepted`, `activation_transport_ready`, `baseline_match`, `sharded_inference_ready`, and `stage_requeue_ready` when `--failure-mode kill-stage-after-claim` is used. It is CPU-only, read-only, not production Swarm Inference, not P2P, and not real LLM/GPU sharding.

For the CPU Pipeline-Sharded Inference Beta loopback:

```bash
crowdtensor shard-infer-beta --mode remote-loopback --json
```

This emits `remote_sharded_inference_beta_v1`, wraps `scripts/remote_sharded_inference_beta_pack.py`, and is validated by `scripts/remote_sharded_inference_beta_check.py`. The report keeps the Alpha activation hashes and `baseline_match` validation while adding mode-level readiness codes `remote_sharded_inference_ready`, `remote_sharded_loopback_ready`, and `local_sharded_inference_ready`; `--failure-mode kill-stage-after-claim` also requires `stage_requeue_ready`. It is CPU-only, read-only, not production Swarm Inference, not P2P, and not real LLM sharding.

The controlled two-machine helper also accepts:

```bash
crowdtensor remote-demo prepare --workload sharded-model-bundle --coordinator-url https://YOUR_COORDINATOR_HOST --miner-id remote-linux-1 --json
```

The loopback validation is `python scripts/remote_home_compute_demo_check.py --workload sharded-model-bundle`. A ready run emits `remote_python_sharded_model_bundle_infer`, `remote_sharded_inference_acceptance_v1`, `remote_sharded_inference_observability_v1`, and `remote_two_machine_sharded_ready` without widening the claim beyond CPU-only, read-only, not production Swarm Inference, not P2P, and not real LLM sharding.

For the CPU-only Micro-LLM Pipeline-Sharded Inference Alpha:

```bash
crowdtensor micro-llm-shard-infer --decode-steps 3 --json
```

This emits `micro_llm_sharded_cli_v1`, wraps `scripts/micro_llm_sharded_inference_evidence_pack.py`, and writes `micro_llm_sharded_evidence_v1` artifacts under `dist/micro-llm-shard-infer`. The `micro_llm_sharded_infer` / `micro_llm_sharded_infer_v1` workload creates a `micro_llm_sharded_session_v1` with two fixed stages: stage 0 produces hidden-state activation hashes/byte counts, stage 1 consumes the accepted activation payload and must match the full tiny Transformer baseline, including `decoded_tokens_match` for `decode_steps`. CI validates this with `scripts/micro_llm_sharded_inference_check.py` and diagnosis codes `stage_0_accepted`, `stage_1_accepted`, `activation_transport_ready`, `baseline_match`, `decoded_tokens_match`, `micro_llm_sharded_ready`, and `stage_requeue_ready` when `--failure-mode kill-stage-after-claim` is used. It is CPU-only, read-only, not production Swarm Inference, not P2P, and not GGUF/llama.cpp or large LLM serving.

Stage-aware operation uses `--stage-mode split --require-distinct-stage-miners` and is checked by `scripts/stage_aware_micro_llm_sharded_check.py`. The Miner advertises `micro_llm_sharded_stage0`, `micro_llm_sharded_stage1`, or `micro_llm_sharded_both`; evidence reports `distinct_stage_miners` and `stage_assignment_valid` when the two stages are accepted by distinct compatible Miners. Use `--failure-mode kill-stage0-after-claim` or `--failure-mode kill-stage1-after-claim` to exercise stage-specific lease timeout requeue.

For the Remote Micro-LLM Pipeline-Sharded Inference Beta loopback:

```bash
crowdtensor micro-llm-shard-infer-beta --mode remote-loopback --decode-steps 3 --json
```

This emits `remote_micro_llm_sharded_beta_v1`, wraps `scripts/remote_micro_llm_sharded_beta_pack.py`, and is validated by `scripts/remote_micro_llm_sharded_beta_check.py`. The report keeps activation hashes, `baseline_match`, and `decoded_tokens_match` validation while adding mode-level readiness codes `remote_micro_llm_sharded_ready`, `remote_micro_llm_sharded_loopback_ready`, and `local_micro_llm_sharded_inference_ready`; `--failure-mode kill-stage-after-claim` also requires `stage_requeue_ready`.

The controlled two-machine helper also accepts:

```bash
crowdtensor remote-demo prepare --workload micro-llm-sharded --coordinator-url https://YOUR_COORDINATOR_HOST --miner-id remote-linux-1 --decode-steps 3 --json
```

The loopback validation is `python scripts/remote_home_compute_demo_check.py --workload micro-llm-sharded --decode-steps 3`. A ready run emits `remote_python_micro_llm_sharded_infer`, `remote_micro_llm_sharded_acceptance_v1`, `remote_micro_llm_sharded_observability_v1`, and `remote_two_machine_micro_llm_sharded_ready` without widening the claim beyond CPU-only, read-only, not production Swarm Inference, not P2P, and not GGUF/llama.cpp serving.

For a stage-aware two-host proof, create one Miner join pack with `crowdtensor remote-demo prepare --workload micro-llm-sharded --stage-role stage0 ...` and one with `--stage-role stage1`, then run `crowdtensor remote-demo verify --workload micro-llm-sharded --stage-mode split --require-distinct-stage-miners ...`. The CI-safe loopback equivalent is `python scripts/remote_home_compute_demo_check.py --workload micro-llm-sharded --stage-mode split --require-distinct-stage-miners --decode-steps 3`.

The Micro-LLM Live Two-Node RC wraps the stage-aware path into one acceptance report:

```bash
crowdtensor micro-llm-live-rc --mode local-generated --port 9182 --request-count 2 --decode-steps 3 --json
python scripts/micro_llm_live_rc_check.py --base-port 9182 --request-count 2 --decode-steps 3
```

This emits `micro_llm_live_rc_v1` through `scripts/micro_llm_live_rc_pack.py`. `local-generated` creates `kaggle-upload-stage0` and `kaggle-upload-stage1`, starts a local Coordinator and two independent stage Miner processes from those generated packages, and should report `local_generated_stage_upload_standins_ready`, `kaggle_micro_llm_sharded_ready`, `stage_assignment_valid`, and `micro_llm_live_rc_ready`. `external-existing` verifies an already running public Coordinator plus two external stage Miners and only then may report `external_runtime_verified`. The RC remains CPU-only, read-only, toy two-stage micro-LLM, not production Swarm Inference, not P2P, and not GGUF/llama.cpp serving.

The Real Small-LLM Sharded Inference Live RC applies the same wrapper shape to the optional HF tiny GPT split path:

```bash
python -m pip install -e '.[hf]'
crowdtensor real-llm-live-rc --mode local-generated --port 9184 --request-count 1 --json
python scripts/real_llm_live_rc_check.py --base-port 9184 --request-count 1
```

This emits `real_llm_live_rc_v1` through `scripts/real_llm_live_rc_pack.py`. `local-generated` creates `kaggle-upload-real-llm-stage0` and `kaggle-upload-real-llm-stage1`, starts a local Coordinator and two independent HF-enabled stage Miner processes from those generated packages, and should report `local_generated_real_llm_stage_upload_standins_ready`, `remote_real_llm_sharded_ready`, `real_llm_artifact_ready`, `stage_assignment_valid`, and `real_llm_live_rc_ready`. `kaggle-generated` prepares packages only. `external-existing` verifies an already running public Coordinator plus two external stage Miners and only then may report `external_runtime_verified` and `kaggle_real_llm_sharded_ready`. `scripts/kaggle_real_llm_live_package.py` emits `kaggle_real_llm_live_package_v1` and can build private Kaggle dataset/script-kernel upload folders from the generated real LLM stage packages; `--inline-kernel-payload` is a temporary private fallback that embeds stage `miner.private.env` into kernel source and requires deleting the temporary kernels/dataset plus token rotation after proof. A completed live proof is retained at `dist/real-llm-live-goal-external/real_llm_live_rc.json` with `kaggle_real_llm_stage0_seen`, `kaggle_real_llm_stage1_seen`, `kaggle_real_llm_sharded_ready`, baseline/decoded-token match, distinct stage Miners, and valid stage assignment. Generated launchers preserve `--enable-hf-tiny-gpt-runtime` and `--real-llm-stage-role`. The RC remains CPU-only, read-only, not production Swarm Inference, not P2P, not GGUF/llama.cpp serving, and not large-model serving.

The Real Internet Swarm Inference Alpha is the broader operator milestone for that same real-weight path:

```bash
crowdtensor real-llm-internet-alpha --mode local-generated --port 9187 --base-port 9188 --request-count 1 --json
python scripts/real_llm_internet_alpha_check.py --port 9187 --base-port 9188 --request-count 1
```

This emits `real_llm_internet_alpha_v1` through `scripts/real_llm_internet_alpha_pack.py`. `local-generated` aggregates the Live RC with stage0 and stage1 timeout rescue checks, so a ready report must include `real_llm_internet_alpha_ready`, `real_llm_stage_requeue_ready`, `stage_requeue_ready`, `real_llm_live_rc_ready`, `remote_real_llm_sharded_ready`, `real_llm_artifact_ready`, `activation_transport_ready`, `decoded_tokens_match`, `distinct_stage_miners`, and `stage_assignment_valid` while keeping `external_runtime_verified` false. Use `package` to generate public Coordinator and stage upload artifacts without a live claim. Use `external-existing` only after an already running public Coordinator plus two external stage Miners are online; only then may the report include `external_runtime_verified`. Temporary public HTTP operation reports `token_rotation_required`. This is CPU-only, read-only, not production Swarm Inference, not P2P, not GPU/TPU pooling, not GGUF/llama.cpp serving, and not large-model serving.

The Real Internet Swarm Inference Beta automates that external path end to end:

```bash
crowdtensor real-llm-internet-beta --mode kaggle-auto --public-host 24.199.118.54 --port 9190 --base-port 9191 --request-count 2 --json
python scripts/real_llm_internet_beta_check.py --port 9190 --base-port 9191 --request-count 2
```

`crowdtensor real-llm-internet-beta` emits `real_llm_internet_beta_v1` through `scripts/real_llm_internet_beta_pack.py`. `kaggle-auto` creates the Alpha package, starts the temporary public Coordinator, pushes private Kaggle CPU script kernels, runs external-existing verification, deletes the temporary kernels, stops the Coordinator, and records the lifecycle. With `--failure-mode kill-stage0-after-claim` or `kill-stage1-after-claim`, it creates separate victim/rescue stage kernels, observes the victim claim through `/state`, deletes the victim kernel, waits for lease timeout requeue, pushes the rescue kernel, and records `external_stage_requeue_ready`, `live_stage0_requeue_ready` / `live_stage1_requeue_ready`, and `live_requeue_summary`. `--mode evidence-import` audits retained generation and requeue reports without a fresh Kaggle run; it requires safe `generated_token_count` evidence that meets `--max-new-tokens`, matching model metadata, cleanup evidence, imported backend/schema metadata, and public-safe live requeue details. Retained 16-token import evidence is `dist/goal-final-infer-real-llm-internet-beta-import-16tok-gpu-summary-20260602/real_llm_internet_beta.json`. A ready report must include `real_llm_internet_beta_ready`, `real_llm_internet_alpha_ready`, `external_runtime_verified`, both Kaggle stages seen, `decoded_tokens_match`, `distinct_stage_miners`, `stage_assignment_valid`, `kaggle_kernels_deleted`, and `token_rotation_required`. `scripts/real_llm_internet_beta_check.py` is a fake-runner contract check for CI and does not create Kaggle resources. This is CPU-only by default, read-only, not production Swarm Inference, not P2P, and not large-model serving.

The user-facing Swarm Inference Beta is the operator wrapper for the real tiny GPT split path:

```bash
crowdtensor swarm-infer-beta live \
  --public-host 24.199.118.54 \
  --port 9210 \
  --base-port 9211 \
  --request-count 2 \
  --output-dir dist/swarm-inference-beta-live \
  --json

crowdtensor swarm-infer-beta prepare --coordinator-url https://YOUR_COORDINATOR_HOST --json
crowdtensor swarm-infer-beta verify --coordinator-url https://YOUR_COORDINATOR_HOST --json
crowdtensor swarm-infer-beta collect --coordinator-url https://YOUR_COORDINATOR_HOST --json
python scripts/swarm_inference_beta_check.py --json
```

`crowdtensor swarm-infer-beta` emits `swarm_inference_beta_v1` through `scripts/swarm_inference_beta_pack.py`. The live action is the side-effectful `kaggle-auto` proof: it wraps `real_llm_internet_beta_v1`, starts a temporary public Coordinator, pushes private Kaggle CPU stage kernels, verifies `external_runtime_verified`, optionally verifies external victim/rescue requeue with `--failure-mode`, deletes the kernels, writes `support_bundle.json`, removes local live private artifacts and raw runtime state by default, and only then may report `swarm_inference_beta_live_ready`, `swarm_inference_beta_ready`, `two_machine_swarm_inference_ready`, `real_llm_internet_beta_ready`, `external_stage_requeue_ready` when requested, `live_requeue_summary`, `kaggle_kernels_deleted`, `swarm_inference_beta_live_private_artifacts_cleaned`, and `token_rotation_required`. Use `--keep-live-private-artifacts` only for debugging failed live runs. The prepare action writes `operator.private.env`, stage0/stage1 `miner.private.env`, a hashed `miner_registry.json`, stage join packs, and `SWARM_INFERENCE_BETA.md`. The verify action wraps `remote_real_llm_sharded_beta_v1` and requires `real_llm_split_route_ready`, `decoded_tokens_match`, `distinct_stage_miners`, and `stage_assignment_valid`. The collect action gathers redacted evidence/support, and clean is dry-run by default. Existing `real_llm_internet_beta_v1` can be imported as `external_beta_evidence_imported` without claiming a fresh live run. This remains CPU-only, read-only, not production Swarm Inference, not P2P, and not large-model serving.

The next product-shaped wrapper is Public Swarm Inference Alpha:

```bash
crowdtensor swarm-session --mode live-kaggle --public-host 24.199.118.54 --port 9220 --base-port 9221 --request-count 2 --json
python scripts/public_swarm_inference_alpha_check.py --json
```

`crowdtensor swarm-session` emits `public_swarm_inference_alpha_v1` through `scripts/public_swarm_inference_alpha_pack.py`. The `live-kaggle` mode aggregates the cleanup-backed `swarm-infer-beta live` proof, true external victim/rescue requeue evidence (`external_stage_requeue_ready`, `live_stage0_requeue_ready` / `live_stage1_requeue_ready`, `live_requeue_summary`) when `--failure-mode` is enabled, and the mandatory `local-generated` real LLM stage requeue proof. It then reports `public_swarm_inference_alpha_ready`, `public_swarm_session_ready`, `local_stage_requeue_ready`, `public_swarm_live_requeue_ready`, `public_swarm_live_kaggle_ready`, `stage_requeue_ready`, `external_runtime_verified`, `kaggle_kernels_deleted`, and `token_rotation_required`. `local-generated` proves the stage requeue side without creating Kaggle resources. Child debug artifacts are pruned by default so the retained evidence is the top-level public JSON/Markdown report; use `--keep-child-artifacts` only for local debugging. This is CPU-only, read-only, not production Swarm Inference, not P2P, and not large-model serving.

Use Public Swarm Inference Alpha RC when turning retained live evidence into one release-candidate artifact:

```bash
crowdtensor public-swarm-alpha-rc --mode evidence-import --json
python scripts/public_swarm_inference_alpha_rc_check.py --mode local-smoke
```

This emits `public_swarm_inference_alpha_rc_v1` through `scripts/public_swarm_inference_alpha_rc_pack.py`. The `evidence-import` mode audits retained reports for `stage0_live_requeue_evidence_ready`, `stage1_live_requeue_evidence_ready`, `public_swarm_live_requeue_evidence_ready`, `public_swarm_alpha_rc_evidence_imported`, `public_swarm_alpha_private_artifacts_absent`, and `public_swarm_inference_alpha_rc_ready`; the current retained stage reports are under `dist/public-swarm-inference-alpha-live-stage0-requeue-20260527165830` and `dist/public-swarm-inference-alpha-live-stage1-requeue-20260527170600`. `local-smoke` is CI-safe and does not create Kaggle resources. It remains CPU-only, read-only, not production Swarm Inference, not P2P, and not large-model serving.

Use Public Swarm Live Preview RC when you want the broadest public-preview evidence bundle over the current Coordinator-backed stack:

```bash
crowdtensor live-preview local-smoke --json
crowdtensor live-preview package --public-host 24.199.118.54 --json
crowdtensor live-preview live-kaggle --public-host 24.199.118.54 --failure-mode kill-stage0-after-claim --json
crowdtensor live-preview evidence-import --json
python scripts/public_swarm_live_preview_rc_check.py --mode local-smoke --json
python scripts/public_swarm_live_preview_rc_check.py --mode package --json
python scripts/public_swarm_live_preview_rc_check.py --mode live-kaggle --json
python scripts/public_swarm_live_preview_rc_check.py --mode evidence-import --json
```

`crowdtensor live-preview` emits `public_swarm_live_preview_rc_v1` through `scripts/public_swarm_live_preview_rc_pack.py`. The `live-preview local-smoke` path validates the Developer Preview and Public Swarm Alpha contracts without external side effects; `live-preview package` creates the public runbook; `live-preview live-kaggle` wraps the side-effectful public Kaggle proof and must report `public_swarm_live_preview_live_kaggle_ready`, `external_stage_requeue_ready`, `kaggle_kernels_deleted`, `private_artifacts_cleaned`, and `token_rotation_required`; `live-preview evidence-import` promotes retained Developer Preview and Alpha RC reports as `public_swarm_live_preview_evidence_import_ready`. Fresh retained stage0/stage1 RC reports are `dist/public-swarm-live-preview-rc-live-stage0-20260529043801-rc/public_swarm_live_preview_rc.json` and `dist/public-swarm-live-preview-rc-live-stage1-20260529044328-rc/public_swarm_live_preview_rc.json`; use a short Kaggle slug prefix such as `ct-live-preview` so victim/rescue suffixes fit Kaggle's 45-character kernel slug limit. Optional retained GPU generation evidence is surfaced through `gpu_generation_evidence_import_ready`. This is CPU-only by default, read-only, Coordinator-backed, not production Swarm Inference, not libp2p, not DHT, not NAT traversal, and not large-model serving.

Use Public Swarm v0.1 Operator Preview when you want one top-level ordinary-user preview artifact over product, live, release, support, CPU fallback, and retained GPU evidence:

```bash
crowdtensor operator-preview local-smoke --json
crowdtensor operator-preview package --public-host 24.199.118.54 --json
crowdtensor operator-preview live-kaggle --public-host 24.199.118.54 --failure-mode kill-stage0-after-claim --json
crowdtensor operator-preview evidence-import --json
python scripts/public_swarm_operator_preview_check.py --mode local-smoke --json
python scripts/public_swarm_operator_preview_check.py --mode package --json
python scripts/public_swarm_operator_preview_check.py --mode live-kaggle --json
python scripts/public_swarm_operator_preview_check.py --mode evidence-import --json
```

`crowdtensor operator-preview` emits `public_swarm_operator_preview_v1` through `scripts/public_swarm_operator_preview_pack.py` and is checked by `scripts/public_swarm_operator_preview_check.py`. `operator-preview local-smoke` validates the local public-preview contract, `operator-preview package` creates `OPERATOR_PREVIEW.md` plus join material, `operator-preview live-kaggle` attempts a fresh public Kaggle proof, and `operator-preview evidence-import` imports retained redacted evidence. A ready report preserves `public_swarm_operator_preview_ready`, `operator_preview_user_path_ready`, `operator_preview_local_smoke_ready`, `operator_preview_package_ready`, `operator_preview_live_kaggle_ready`, `operator_preview_evidence_import_ready`, `serve_join_generate_ready` or package-mode `miner_join_pack_ready`, `cpu_fallback_ready`, `live_preview_ready`, `support_bundle_ready`, `release_readiness_ready`, and optional `gpu_generation_evidence_import_ready`. CPU-only hosts that lack optional HF dependencies report `developer_preview_degraded` plus `operator_preview_cpu_fallback_user_path_ready`; retained evidence imports may report `operator_preview_retained_evidence_ready`. When the fresh external run cannot complete, it records `external_runtime_blocked` and uses retained stage0/stage1 Live Preview RC reports instead of claiming fresh external runtime evidence. This is CPU-only by default, read-only, Coordinator-backed, not production Swarm Inference, not libp2p, not DHT, not NAT traversal, and not large-model serving.

The top-level Operator Preview JSON, Markdown, terminal output, and support bundle include `output_request`, `answer_scope.scope_state: no-local-answer`, and `shareable_summary.answer_scope_state`. Treat `crowdtensor operator-preview` output as shareable operator-path evidence, not a local answer transcript; raw prompts, generated text, generated token ids, activations, credentials, leases, private env files, and runtime state stay out of public artifacts.

Use Public Swarm v0.2 Usable Inference Trial as the ordinary-user trial wrapper over the current product surface:

```bash
crowdtensor swarm-trial local-loopback --json
crowdtensor swarm-trial package --public-host 24.199.118.54 --json
crowdtensor swarm-trial live-kaggle --public-host 24.199.118.54 --kaggle-owner YOUR_KAGGLE_USERNAME --json
crowdtensor swarm-trial evidence-import --json
python scripts/public_swarm_trial_check.py --mode local-loopback --json
python scripts/public_swarm_trial_check.py --mode package --json
python scripts/public_swarm_trial_check.py --mode live-kaggle --json
python scripts/public_swarm_trial_check.py --mode evidence-import --json
```

`crowdtensor swarm-trial` emits `public_swarm_trial_v1` through `scripts/public_swarm_trial_pack.py` and is checked by `scripts/public_swarm_trial_check.py`. `swarm-trial local-loopback` validates `serve` / `join stage0` / `join stage1` / `generate` when optional `[hf]` dependencies exist; `swarm-trial package` creates `SWARM_TRIAL.md` plus join material; `swarm-trial live-kaggle` wraps the controlled external Operator Preview path; `swarm-trial evidence-import` imports retained Operator Preview and GPU generation evidence. A ready report preserves `public_swarm_trial_ready`, `serve_join_generate_trial_ready`, `stage0_join_ready`, `stage1_join_ready`, `generate_ready`, `generated_token_count_ready`, `support_bundle_ready`, `cpu_fallback_ready`, `private_artifacts_cleaned`, `operator_preview_import_ready`, `gpu_generation_evidence_import_ready`, and live `token_rotation_required`. If optional HF/Kaggle/external runtime execution is unavailable, it records `swarm_trial_degraded_cpu_fallback_ready` or `external_runtime_blocked` instead of claiming fresh real-weight generation. This is CPU-only by default, read-only, Coordinator-backed, not production Swarm Inference, not libp2p, not DHT, not NAT traversal, not GPU marketplace, and not large-model serving.

The top-level Swarm Trial JSON, Markdown, terminal output, and support bundle include `output_request`, `answer_scope.scope_state: no-local-answer`, and `shareable_summary.answer_scope_state`. Treat `crowdtensor swarm-trial` output as shareable ordinary-user trial evidence, not a local answer transcript; run `crowdtensor generate` in human mode to view local generated text.

Use Public Swarm Inference Beta as the ordinary user entrypoint for the current Coordinator-backed product surface:

```bash
crowdtensor public-swarm-beta product-beta --json
python scripts/public_swarm_inference_beta_check.py --mode product-beta --json

# Compatibility paths for the legacy CPU split proof and retained evidence import:
crowdtensor public-swarm-beta local-loopback --base-port 9290 --request-count 1 --json
crowdtensor public-swarm-beta evidence-import --json
python scripts/public_swarm_inference_beta_check.py --mode local-loopback --base-port 9290 --request-count 1
```

`crowdtensor public-swarm-beta` emits `public_swarm_inference_beta_v1` through `scripts/public_swarm_inference_beta_pack.py` and is checked by `scripts/public_swarm_inference_beta_check.py`. `product-beta` should report `public_swarm_product_beta_ready`, `public_swarm_product_rc_ready`, `coordinator_product_surface_ready`, `session_protocol_ready`, `p2p_lite_discovery_ready`, `gpu_generation_evidence_import_ready`, `cpu_fallback_ready`, and `local_cpu_inference_ready`. It aggregates `crowdtensor serve`, `crowdtensor join`, `crowdtensor generate`, `crowdtensor peer`, `session_protocol_v1`, `p2p_lite_peer_v1`, retained GPU sharded generation evidence, and a CPU inference fallback into one shareable Beta artifact. `public-swarm-beta local-loopback` still reports `two_stage_split_inference_ready`, `local_loopback_ready`, `decoded_tokens_match`, `distinct_stage_miners`, and `stage_assignment_valid`; `public-swarm-beta evidence-import` still reports `public_swarm_beta_evidence_import_ready`, `external_live_evidence_imported`, `stage0_live_requeue_evidence_ready`, and `stage1_live_requeue_evidence_ready`. Operator runs use `prepare`, `coordinator`, `miner --stage stage0`, `miner --stage stage1`, `verify`, `collect`, and dry-run `clean`. This remains Coordinator-backed, read-only, not production Swarm Inference, not libp2p/DHT/NAT traversal, not Hivemind-level serving, and not large-model serving.

Use Public Real-LLM Swarm Inference Beta v1 when you need the current top-level release artifact over product, external, P2P, and optional GPU evidence:

```bash
python -m pip install -e '.[hf]'
crowdtensor public-real-llm-swarm-beta release --json
crowdtensor public-real-llm-swarm-beta local-model-variant --hf-model-id distilgpt2 --max-new-tokens 16 --stream-generation --json
crowdtensor public-real-llm-swarm-beta evidence-import \
  --product-report dist/goal-final-infer-public-real-llm-swarm-beta-local-batch-stream-16tok-fixed-20260602/product-beta/public_swarm_product_beta.json \
  --max-new-tokens 16 \
  --json
crowdtensor public-real-llm-swarm-beta package --json
crowdtensor public-real-llm-swarm-beta check --beta-report dist/public-real-llm-swarm-beta/public_real_llm_swarm_beta.json --output-dir dist/public-real-llm-swarm-beta-check --json
crowdtensor public-real-llm-swarm-beta check --hf-model-id distilgpt2 --beta-report dist/public-real-llm-swarm-beta/public_real_llm_swarm_beta.json --output-dir dist/public-real-llm-swarm-beta-check-distilgpt2 --json
```

`crowdtensor public-real-llm-swarm-beta release` emits `public_real_llm_swarm_beta_v1` through `scripts/public_real_llm_swarm_beta_pack.py`. `crowdtensor public-real-llm-swarm-beta check` is the official user-facing validation wrapper around `scripts/public_real_llm_swarm_beta_check.py`; with `--beta-report` it validates an existing `public_real_llm_swarm_beta.json` and records `check_source: beta-report`, while omitting `--beta-report` keeps the CI-safe fixture check path. It writes `public_real_llm_swarm_beta_check_v1`, preserves `cli_mode: check`, records `review_summary`, `artifact_summary`, `operator_action`, `checked_runtime_provenance`, `checked_evidence_scope`, `checked_gpu_status`, `checked_gpu_proof_next_step`, and safe output-scope fields, and automatically validates non-default `--hf-model-id` runs through the local-model-variant check path. Read `evidence_scope` in the Beta report and `checked_evidence_scope` in the check report for the shortest answer to whether the verified evidence was local CPU, retained evidence, or fresh Kaggle GPU; read `gpu_status` / `checked_gpu_status` for the direct verdict (`local-cpu-only`, `local-gpu-smoke-only`, `retained-gpu-evidence`, or `fresh-kaggle-gpu-verified`), and read the check terminal `checked_runtime_provenance` line for the detailed source/proof summary behind that checked scope. Ordinary `infer` / `generate` JSON and Markdown include `gpu_proof_next_step`; Beta JSON/Markdown/terminal output also includes `gpu_proof_next_step`, and check output mirrors it as `checked_gpu_proof_next_step`. These fields name the explicit optional CUDA smoke, Kaggle package, and side-effectful fresh Kaggle GPU proof commands and mark Kaggle commands as requiring cleanup and token rotation. Only `fresh_kaggle_gpu=True` is a fresh Kaggle GPU claim. Release mode runs the Product Beta `serve` / `join` / `generate` loop, imports retained external real-LLM two-stage requeue evidence, fresh-runs the Petals-class real-P2P candidate local-smoke under `p2p-candidate` using retained external generation/requeue/runtime-smoke/batch-stream source reports, fresh-runs the Public Swarm v2 ordinary P2P user-path report under `public-swarm-v2`, fresh-runs a v2 `real-p2p-local/real_p2p_swarm_inference_core_rc.json` route-hardening and stage-requeue child, and uses `public-swarm-v2/usable-v1-local/usable_swarm_inference.json` for top-level KV-cache readiness, runs optional CUDA fail-closed smoke, and writes `public_real_llm_swarm_beta.json`, `public_real_llm_swarm_beta.md`, `PUBLIC_REAL_LLM_SWARM_BETA.md`, and `support_bundle.json`. `evidence-import` uses `--p2p-report` and `--public-swarm-v2-report` instead of fresh-running those children. A ready report should include `public_real_llm_swarm_beta_ready`, `public_real_llm_swarm_beta_public_swarm_v2_ready`, `public_real_llm_swarm_beta_p2p_user_path_ready`, `public_real_llm_swarm_beta_v2_real_p2p_local_ready`, `public_swarm_v2_real_p2p_local_ready`, `public_real_llm_swarm_beta_v2_real_p2p_local_requeue_ready`, `public_swarm_v2_real_p2p_local_requeue_ready`, `public_real_llm_swarm_beta_v2_batch_ready`, `public_real_llm_swarm_beta_v2_stream_ready`, `cpu_default_ready`, `external_two_stage_ready`, `external_stage_requeue_ready`, `p2p_ready_product_beta`, `p2p_live_requeue_rescue_ready`, `p2p_victim_result_not_accepted`, `public_real_llm_swarm_beta_product_model_match_ready`, `public_real_llm_swarm_beta_kv_cache_ready`, `public_real_llm_swarm_beta_kv_cache_model_match_ready`, `public_real_llm_swarm_beta_private_artifacts_cleaned`, `cuda_optional_fail_closed_ready`, and `release_evidence_ready`; local product-path proofs can also preserve `public_real_llm_swarm_beta_batch_ready` and `public_real_llm_swarm_beta_stream_ready`. Retained local-smoke evidence at `dist/goal-final-infer-public-real-llm-swarm-beta-local-batch-stream-16tok-fixed-20260602/public_real_llm_swarm_beta.json` proves the two-prompt 16-token batch plus safe stream path. Current fresh local release evidence at `dist/goal-final-infer-public-real-llm-swarm-beta-release-fresh-v2-local-requeue-20260602/public_real_llm_swarm_beta.json` proves fresh Product Beta, fresh release-local Petals-class P2P candidate local-smoke, fresh local Public Swarm v2 P2P execution, fresh v2 real-P2P local 16-token route-hardening plus stage1 victim/rescue requeue, and release-local Usable/KV-cache evidence with 16 tokens, 32 accepted rows, v2 batch/stream readiness, 15 KV-cache hits per stage, retained external/GPU imports plus retained P2P source inputs, release-local `source_reports.p2p_report`, private artifact cleanup, and no `not_completed` items. Retained complete import evidence at `dist/goal-final-infer-public-real-llm-swarm-beta-import-v2-local-requeue-batch-stream-20260602/public_real_llm_swarm_beta.json` combines that product path with Public Swarm v2 local real-P2P stage requeue; earlier retained 16-token import evidence remains at `dist/goal-final-infer-public-real-llm-swarm-beta-import-16tok-p2p-batch-stream-kv-cache-model-gated-v2-20260602/public_real_llm_swarm_beta.json`. Imports do not prove a fresh external run. The real-P2P candidate source reports must include public-safe `live_requeue_summary` evidence for victim claim observation, victim kernel deletion, lease expiry, rescue acceptance, and `victim_result_accepted: false`; code-only requeue reports do not satisfy the top-level release gate. Release requires retained external/P2P source reports plus the fresh Public Swarm v2 child KV-cache report and fresh v2 real-P2P route-hardening/requeue child to meet the requested token target; evidence-import requires external, P2P, Public Swarm v2, and Usable Swarm KV-cache reports to meet that target. Otherwise token-target, v2, or KV-cache diagnostics block readiness. Non-default `--hf-model-id` runs must also have product, external, P2P, v2, and KV-cache evidence with matching `hf_model_id`; otherwise `product_model_mismatch`, `external_model_mismatch`, `p2p_model_mismatch`, `public_swarm_v2_model_mismatch`, or `kv_cache_model_mismatch` blocks the aggregate. Use `evidence-import` only when a product report already exists; otherwise use `release`. Treat the report as the strongest current Beta evidence, still Coordinator-backed, read-only by default, tiny/small-model scoped, not full Hivemind/Petals production parity, not Coordinator-free, not NAT traversal production, and not large-model serving.

The top-level JSON, Markdown, terminal output, and support bundle include `output_request`, `answer_scope.scope_state: no-local-answer`, `shareable_summary.answer_scope_state`, `evidence_scope`, and check-side `checked_runtime_provenance` plus `checked_evidence_scope`. Treat Public Real-LLM Swarm Beta as shareable release evidence, not a local answer transcript; run `crowdtensor generate` in human mode to view local generated text.

Current retained evidence-import with the new local real-P2P stage requeue gate is `dist/goal-final-infer-public-real-llm-swarm-beta-import-v2-local-requeue-batch-stream-20260602/public_real_llm_swarm_beta.json`. It imports `dist/goal-final-infer-public-swarm-v2-local-real-p2p-requeue-batch-stream-20260602/public_swarm_inference_v2.json` and should preserve `public_real_llm_swarm_beta_v2_real_p2p_local_requeue_ready`, `public_swarm_v2_real_p2p_local_requeue_ready`, `real_p2p_local_stage_requeue_ready`, `stage_requeue_ready`, `accepted_result_after_requeue`, and `victim_result_accepted: false`. This is a local route-hardening proof inside the Beta aggregate, not a fresh external/Kaggle proof.

The stronger fresh release proof is `dist/goal-final-infer-public-real-llm-swarm-beta-release-fresh-v2-local-requeue-20260602/public_real_llm_swarm_beta.json`. It fresh-runs Product Beta, Petals-class P2P candidate local-smoke, Public Swarm v2, the v2 local real-P2P stage1 requeue child, and release-local Usable/KV-cache evidence with `max_new_tokens: 16`, batch/stream readiness, 15 KV-cache hits per stage, private artifact cleanup, and `not_completed: []`.

The local model variant proof is `dist/goal-final-infer-local-model-variant-distilgpt2-clean-codes-v2-20260602/public_real_llm_swarm_beta.json`. It uses `local-model-variant` with `--hf-model-id distilgpt2`, proves Product Beta, Public Swarm v2 local model compatibility, v2 local real-P2P stage1 requeue, dual-stage KV-cache, batch/stream readiness, and CUDA fail-closed behavior, and reports `public_real_llm_swarm_beta_local_model_variant_ready` while keeping `release_evidence_ready`, `external_two_stage_ready`, and `external_stage_requeue_ready` false.

Use Public Swarm Product Beta when you need the ordinary user-facing path rather than a release-candidate aggregate:

```bash
python -m pip install -e '.[hf]'
crowdtensor public-swarm-product-beta local-loopback --base-port 9320 --max-new-tokens 2 --json
crowdtensor public-swarm-product-beta package --target kaggle --json
crowdtensor public-swarm-product-beta external-existing --coordinator-url https://YOUR_COORDINATOR_HOST --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" --admin-token "$CROWDTENSOR_ADMIN_TOKEN" --json
python scripts/public_swarm_product_beta_check.py --mode local-loopback --json
python scripts/public_swarm_product_beta_check.py --mode package --target kaggle --json
python scripts/public_swarm_product_beta_check.py --mode external-existing --json
```

`crowdtensor public-swarm-product-beta` emits `public_swarm_product_beta_v1` through `scripts/public_swarm_product_beta_pack.py` and is checked by `scripts/public_swarm_product_beta_check.py`. A ready local user path reports `public_swarm_product_beta_ready`, `public_swarm_product_beta_user_path_ready`, `serve_ready`, `stage0_join_ready`, `stage1_join_ready`, `generate_ready`, `support_bundle_ready`, `private_artifacts_cleaned`, `decoded_tokens_match`, `distinct_stage_miners`, and `stage_assignment_valid`. Use `package` for two-machine or Kaggle join material with `private_artifacts_local_only` and `miner_join_pack_ready`; use `external-existing` only for an already running controlled runtime. Missing optional `[hf]` dependencies should report `hf_dependencies_missing`. It is CPU-only by default, read-only, Coordinator-backed, not production Swarm Inference, not libp2p, not DHT, not NAT traversal, and not large-model serving.

The top-level JSON, Markdown, terminal output, and support bundle include `output_request`, `answer_scope.scope_state: no-local-answer`, and `shareable_summary.answer_scope_state`. Treat Public Swarm Product Beta as shareable product-path evidence, not a local answer transcript; run `crowdtensor generate` in human mode to view local generated text.

`crowdtensor preview` is the Public Swarm Developer Preview wrapper for ordinary users who want the largest single artifact over the current product surface. It emits `public_swarm_developer_preview_v1` through `scripts/public_swarm_developer_preview_pack.py` and is checked by `scripts/public_swarm_developer_preview_check.py`. Use `preview local` to run the Product Beta `serve` / `join stage0` / `join stage1` / `generate` path and require `developer_preview_ready`, `public_swarm_developer_preview_ready`, `local_two_stage_generation_ready`, `serve_join_generate_ready`, `product_beta_ready`, `support_bundle_ready`, `cpu_fallback_ready`, and `local_cpu_inference_ready`; retained GPU generation evidence adds `gpu_generation_evidence_import_ready`. Use `preview package` for two-machine or Kaggle join material, `preview external-existing` for an already running controlled runtime, and `preview evidence-import` for retained redacted Product Beta and GPU reports. Missing optional `[hf]` dependencies should surface `hf_dependencies_missing`. This is CPU-only by default, read-only, Coordinator-backed, not production Swarm Inference, not libp2p, not DHT, not NAT traversal, and not large-model serving.

The top-level Developer Preview JSON, Markdown, terminal output, and support bundle include `output_request`, `answer_scope.scope_state: no-local-answer`, and `shareable_summary.answer_scope_state`. Treat `crowdtensor preview` output as shareable preview evidence, not a local answer transcript; run `crowdtensor generate` in human mode to view local generated text.

Use Public Swarm Inference Beta RC when you need one release-candidate artifact over the product path:

```bash
crowdtensor public-swarm-beta-rc local-loopback --base-port 9310 --max-new-tokens 2 --json
crowdtensor public-swarm-beta-rc package --target kaggle --json
crowdtensor public-swarm-beta-rc external-existing --coordinator-url https://YOUR_COORDINATOR_HOST --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" --admin-token "$CROWDTENSOR_ADMIN_TOKEN" --json
python scripts/public_swarm_inference_beta_rc_check.py --mode local-loopback --json
python scripts/public_swarm_inference_beta_rc_check.py --mode package --target kaggle --json
python scripts/public_swarm_inference_beta_rc_check.py --mode external-existing --json
```

`crowdtensor public-swarm-beta-rc` emits `public_swarm_inference_beta_rc_v1` through `scripts/public_swarm_inference_beta_rc_pack.py` and is checked by `scripts/public_swarm_inference_beta_rc_check.py`. A ready RC preserves `public_swarm_inference_beta_rc_ready`, `public_swarm_product_beta_ready`, `p2p_lite_route_ready`, `p2p_lite_discovery_ready`, `cpu_fallback_ready`, `serve_join_generate_loop_ready`, `remote_generate_session_ready`, and `public_swarm_generate_ready`. `package` keeps `private_artifacts_local_only` and `miner_join_pack_ready`; `external-existing` adds `external_runtime_verified` only after an already running controlled runtime verifies. If this host lacks `transformers`, local runtime evidence should surface `hf_dependencies_missing`. It is CPU-only by default, read-only, Coordinator-backed, not production Swarm Inference, not libp2p, not DHT, not NAT traversal, and not large-model serving.

Use Public Swarm GPU Inference Beta only when you want the optional CUDA overlay for that tiny GPT split proof:

```bash
python -m pip install -e '.[hf]'
crowdtensor public-swarm-gpu-beta local-smoke --json
crowdtensor public-swarm-gpu-beta local-loopback --base-port 9321 --request-count 1 --json
crowdtensor public-swarm-gpu-beta kaggle-package --output-dir dist/public-swarm-gpu-beta-kaggle --json
crowdtensor public-swarm-gpu-beta kaggle-auto --public-host 24.199.118.54 --port 9320 --base-port 9321 --kaggle-owner YOUR_KAGGLE_USERNAME --request-count 1 --json
crowdtensor public-swarm-gpu-beta evidence-import --gpu-report dist/public-swarm-gpu-beta/public_swarm_gpu_inference_beta.json --json
python scripts/public_swarm_gpu_inference_beta_check.py --mode local-smoke
python scripts/public_swarm_gpu_inference_beta_check.py --mode kaggle-auto
```

`crowdtensor public-swarm-gpu-beta` emits `public_swarm_gpu_inference_beta_v1` through `scripts/public_swarm_gpu_inference_beta_pack.py` and is checked by `scripts/public_swarm_gpu_inference_beta_check.py`. `public-swarm-gpu-beta local-smoke` is CI-safe on CPU-only hosts and reports `public_swarm_gpu_beta_smoke_ready` without claiming a usable GPU route. `public-swarm-gpu-beta local-loopback` selects `hf_transformers_cuda`, requires `cuda_runtime_available`, `hf_transformers_cuda_ready`, and `gpu_runtime_ready`, and routes stage work only to Miners advertising `real_llm_sharded_cuda_stage0`, `real_llm_sharded_cuda_stage1`, or `real_llm_sharded_cuda_both`; ready reports include `public_swarm_gpu_beta_ready`, `gpu_stage0_ready`, and `gpu_stage1_ready`. `public-swarm-gpu-beta kaggle-package` prepares private Kaggle GPU stage templates with `kaggle_gpu_package_ready`. The side-effectful `public-swarm-gpu-beta kaggle-auto` starts a temporary CPU-capable public Coordinator, defers CUDA runtime checks to private Kaggle GPU stage Miners, pushes private Kaggle GPU kernels, verifies `external_gpu_runtime_verified`, deletes kernels, and only then may report `public_swarm_gpu_beta_kaggle_auto_ready`, `kaggle_kernels_deleted`, and `token_rotation_required`; the check mode is a CI fake runner. Generated Kaggle CUDA kernels default to `torch==2.7.1+cu118`, `torchvision==0.22.1+cu118`, and `transformers==4.40.2`, and the retained successful proof is `dist/public-swarm-gpu-beta-live-20260528-runtimepin/public_swarm_gpu_inference_beta_kaggle_auto.json`. `public-swarm-gpu-beta evidence-import` imports a completed GPU report with `external_gpu_runtime_verified`. Ready reports include top-level `output_request`, `answer_scope.scope_state: no-local-answer`, and `shareable_summary.answer_scope_state`; treat them as shareable CUDA readiness evidence, not local answer transcripts. Public artifacts keep raw prompts, generated text, generated token ids, activations, credentials, leases, private env files, Kaggle kernel payloads, and runtime state redacted. This is read-only optional CUDA tiny GPT evidence, not production Swarm Inference, not P2P, not a GPU pooling marketplace, and not large-model serving.

`crowdtensor gpu-generate` is the multi-token generation wrapper over that CUDA split route. It emits `gpu_sharded_generation_beta_v1` through `scripts/gpu_sharded_generation_beta_pack.py` and is checked by `scripts/gpu_sharded_generation_beta_check.py`. `gpu-generate evidence-import --gpu-report dist/gpu-sharded-generation-beta-kaggle-20260528095658/gpu_sharded_generation_beta_kaggle_auto.json --max-new-tokens 16 --json` imports the retained successful Kaggle GPU proof; the report includes `generated_token_count: 16`, a safe `generated_text_hash`, `multi_token_generation_ready`, `gpu_multi_machine_generation_ready`, `external_gpu_runtime_verified`, stage-local partition readiness, distinct stage Miners, `kaggle_kernels_deleted`, top-level `output_request`, `answer_scope.scope_state: no-local-answer`, and `shareable_summary.answer_scope_state`. Treat `gpu-generate` reports as shareable generation evidence, not local answer transcripts: public artifacts keep raw prompts, generated text, generated token ids, activations, credentials, leases, and runtime state redacted. Use `python scripts/gpu_sharded_generation_beta_check.py --include-wrapper-check --max-new-tokens 4 --json` for CI-safe validation without creating Kaggle resources. The side-effectful `gpu-generate kaggle-auto` path is documented in `docs/gpu-sharded-generation-beta.md`; rotate temporary public HTTP tokens after every run and do not retain private env or registry files in public artifacts.

The Public Swarm Product RC is the current product-surface milestone:

```bash
crowdtensor swarm-bootstrap --coordinator-url https://YOUR-TUNNEL.example --tunnel-command 'cloudflared tunnel --url http://127.0.0.1:8787' --expect-remote-miners --json
crowdtensor swarm-bootstrap-check --output-dir state/swarm-bootstrap --expect-remote-miners --json
crowdtensor serve --profile gpu-generation --json
crowdtensor join --coordinator-url http://127.0.0.1:8787 --stage stage0 --backend cuda --json
crowdtensor join --coordinator-url http://127.0.0.1:8787 --stage stage1 --backend cuda --json
crowdtensor generate --coordinator-url http://127.0.0.1:8787 --prompt-text "CrowdTensor product RC" --backend cuda --dry-run --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" --json
crowdtensor peer check --json
crowdtensor public-swarm-product-rc --json
python scripts/public_swarm_product_rc_check.py --json
```

When `crowdtensor swarm-bootstrap` is given `--peer-bootstrap`, each private
stage invite includes `crowdtensor_miner_join_discovery_v1`; the generated
`join --invite-code-file` path can then resolve the Coordinator through
P2P-lite discovery without a Miner operator manually passing `--peer-bootstrap`.
The package also includes `start_discovery.sh`; run it before
`start_coordinator.sh` for discovery-backed packages, or use
`start_control_plane.sh` as the single Coordinator-host launcher. If
`--tunnel-command` is supplied, `start_tunnel.sh` sources
`private/tunnel.private.env` and `start_control_plane.sh` starts that
operator-supplied tunnel/overlay command before discovery and the Coordinator.

`crowdtensor swarm-bootstrap` creates a local private bootstrap directory with `operator_registry.json`, `miner_registry.json`, coordinator/operator/tunnel private env files, one private operator invite, stage0/stage1 Miner packages, private `stage0.miner-package.tar.gz` / `stage1.miner-package.tar.gz` archives, matching `stage0.run-miner.sh` / `stage1.run-miner.sh` helpers, `stage0.handoff.sha256` / `stage1.handoff.sha256`, `stage_handoff_manifest.json`, private `miner.join-code.txt` files, executable `start_control_plane.sh`, optional `start_tunnel.sh`, `start_discovery.sh`, `start_coordinator.sh`, `verify_bootstrap.sh`, `handoff_doctor.sh`, stage `check_join.sh` / `support_bundle.sh` / `join.sh` files, generation scripts, and `SWARM_BOOTSTRAP.md`. Optional `--stage0-reward-account` and `--stage1-reward-account` values are private Beta accounting metadata stored in the matching stage invite and registry entry; public bootstrap, check, support, and handoff reports expose only reward-account presence. Stage `check_join.sh` verifies the invite-code admission path without `--run`; stage `support_bundle.sh` writes safe `miner_support_bundle.json` diagnostics without raw join codes or Miner tokens and includes `crowdtensor_miner_local_environment_v1` with `local_environment_ready`, CLI, checksum, Python, and optional torch/CUDA probes; each stage archive plus its runner and checksum is the preferred copy unit for remote Miner handoff; the runner verifies `stageX.handoff.sha256`, supports `./stageX.run-miner.sh --doctor`, safely extracts, preflights, and starts the Miner; stage `join.sh` defaults to `crowdtensor join --invite-code-file miner.join-code.txt --check-admission --expect-remote-coordinator --run`, while `miner.invite.json` remains private compatibility material. `verify_bootstrap.sh` wraps `crowdtensor swarm-bootstrap-check --check-admission` so the operator can run a live no-claim gate after starting the Coordinator and before copying stage packages. `crowdtensor swarm-bootstrap-check` emits `crowdtensor_swarm_bootstrap_check_v1` and verifies required files, `0600` private env/invite/join-code permissions, `0700` helper scripts, hashed registries, Coordinator/operator env separation, tunnel private-env and launcher readiness, stage invite Coordinator URL consistency, stage reward-account metadata consistency, stage join-code consistency, `stage_check_join_scripts_ready`, `stage_support_bundle_scripts_ready`, `stage_package_archives_ready`, `stage_archive_runner_scripts_ready`, `stage_handoff_checksums_ready`, optional remote route readiness through `--expect-remote-miners`, optional live `/ready` checks through `--check-coordinator`, optional token-backed no-claim `/tasks/preflight` checks through `--check-admission`, and plaintext token, tunnel command, reward-account, or join-code leakage in scripts or public Markdown before stage package handoff. Bootstrap and bootstrap-check both include `bootstrap_handoff`: `remote_miners_ready` summarizes the advertised Miner route, `recommended_launcher` points to `start_control_plane.sh`, `verify_before_handoff` points to the live preflight script, and `ready_to_copy_stage_packages` only becomes true after live admission preflight passes; `crowdtensor swarm-handoff-doctor` / `handoff_doctor.sh` emits `crowdtensor_swarm_handoff_doctor_v1`, `handoff_doctor.json`, and `handoff_doctor.md` with blockers and exact stage files to copy. Bootstrap is the recommended first step for a controlled two-machine setup because it wires env-sourced `crowdtensor serve --operator-token-registry --miner-token-registry`, code-file-based `crowdtensor join --invite-code-file --check-admission`, and operator-env-sourced `crowdtensor generate --dry-run` without printing plaintext tokens, tunnel commands, reward-account values, or join codes in the public report. The Coordinator env contains only the observer verifier; operator admin credentials stay in the separate operator env. `crowdtensor serve` prints or runs the Coordinator command for `cpu-real-llm` or `gpu-generation`; public bind requires explicit acknowledgement. `--coordinator-public-url` can advertise a Miner-facing HTTPS/tunnel/VPN/reverse-proxy URL while the Coordinator still binds locally, and `--expect-remote-miners` fails early when the advertised URL is only `127.0.0.1`/`localhost` or otherwise unsuitable for remote Miner hosts. `--miner-token-registry` loads the private per-Miner registry generated by bootstrap or `scripts/create_miner_invite.py`; `--operator-token-registry` loads role-scoped operator tokens. `crowdtensor join` prints or runs a Miner command for stage0/stage1/both, can read `--invite-code-file`, `--invite-file`, or `--invite`, can resolve a Coordinator through `--peer-bootstrap`, and supports `--check-coordinator` to call the public `/ready` endpoint before running a Miner from a direct URL or invite. `--expect-remote-coordinator` fails before run when a remote Miner would use a local-only Coordinator URL. That preflight verifies the Coordinator entry point from the Miner host without sending the Miner token, claiming work, or submitting generation; for invite joins it also matches the invite against the Coordinator's redacted `miner_policy_summary`, returning `join_invite_policy_miner_missing` or `join_invite_policy_mismatch` when the Coordinator is reachable but did not load the matching registry. `--check-admission` implies that check and then calls token-backed `/tasks/preflight` to verify Miner auth plus claim-time join policy, quota, claim-rate, stage, backend, and model constraints without leasing a task. `join_coordinator_unreachable` means the public URL, VPN, tunnel, firewall, or reverse proxy path must be fixed first; `coordinator_remote_route_required` means the URL is local-only for a requested remote path; `join_admission_token_rejected` and `join_policy_*` mean the Miner token or advertised capability would be rejected before claim. `crowdtensor generate` creates a bounded `session_protocol_v1` request, hashes the prompt in public output, and uses `POST /admin/inference-sessions` when not in `--dry-run`. `generate --dry-run` can also print live `coordinator_ready`, `stage_preflight`, and `ready_to_submit` checks; package-only CI paths use `--skip-live-preflight` to keep this as an offline request-shape check. Read `evidence_scope` in `infer` or `generate` reports to tell whether the current command ran `local-cpu-loopback`, `existing-runtime-preflight`, `existing-runtime-submit`, or a `p2p-runtime-*` path; read the adjacent `evidence_scope_note` for the plain-language explanation such as "preflight only, no generation task submitted", and read `gpu_status` for the direct CPU/GPU verdict. `fresh_kaggle_gpu=True` is the only claim that the current report verified a fresh Kaggle GPU proof, while `fresh_kaggle_gpu_attempted=True` without that verified flag is only an attempted GPU path and `retained_gpu=True` means historical GPU evidence was imported. `crowdtensor peer daemon`, `peer announce`, `peer resolve`, and `peer check` expose `p2p_lite_peer_v1` HTTP-gossip discovery. The RC artifact imports the retained `gpu_sharded_generation_beta_v1` Kaggle evidence and requires `coordinator_product_surface_ready`, `session_protocol_ready`, `p2p_lite_discovery_ready`, and `gpu_generation_evidence_import_ready`; that retained import is not a fresh Kaggle GPU attempt by `crowdtensor infer` or `crowdtensor generate`. P2P-lite does not replace Coordinator leases, heartbeats, validation, or result ledgers; it is not libp2p, DHT, NAT traversal, decentralized security, Hivemind/Petals-level serving, or large-model serving.

The v0.3 Product Swarm MVP check is the shortest strict proof that the product commands themselves can run a real tiny-GPT two-stage generation loop:

```bash
python -m pip install -e '.[hf]'
python scripts/product_swarm_mvp_check.py --max-new-tokens 2 --json
python scripts/product_swarm_mvp_check.py --max-new-tokens 2 --shareable-generate-terminal --json
```

The script starts `crowdtensor serve --run` on localhost, starts `crowdtensor generate` so the Coordinator creates the session, then launches separate one-task `crowdtensor join --stage stage0 --run` and `crowdtensor join --stage stage1 --run` commands for every generation step. Successful output is `product_swarm_mvp_check_v1` in `dist/product-swarm-mvp` with `product_swarm_mvp_ready`, `serve_join_generate_mvp_ready`, `local_two_stage_real_llm_ready`, `generated_token_count_ready`, `distinct_stage_miners`, and `stage_assignment_valid`. Add `--shareable-generate-terminal` to run the same local product loop through real human `crowdtensor generate --prompt-stdin --shareable-terminal` output and require `shareable_generate_terminal_ready`, `answer_scope_state=shareable-terminal-redacted`, `gpu_state=local-cpu-only`, `fresh_kaggle_gpu_verified=false`, and `terminal_answer_text_hidden=true`. CPU is default; add `--backend cuda` only on hosts where stage Miners have a working torch CUDA runtime. If optional Hugging Face dependencies are absent, the default report is degraded but non-fatal with `hf_dependencies_missing`; add `--require-hf-runtime` for release or maintainer verification. The report contains only safe hashes, counts, verdicts, and diagnostics, not raw prompts, generated text, terminal output, token ids, hidden states, activations, leases, or idempotency material.

For a real two-machine or Kaggle rehearsal, first prove the local MVP above, then generate remote join material and verify an already-running external Coordinator/Miner set through Product Beta:

```bash
crowdtensor public-swarm-product-beta package --target kaggle --public-host YOUR_COORDINATOR_HOST --json
crowdtensor public-swarm-product-beta external-existing --coordinator-url https://YOUR_COORDINATOR_HOST --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" --admin-token "$CROWDTENSOR_ADMIN_TOKEN" --json
```

This is intentionally separate from the local MVP check. `package` prepares controlled stage Miner material; `external-existing` verifies live external Miners without widening the local smoke into an automatic public-network run.

Public Swarm Inference Preview v0.4 is the broadest current preview aggregate over the same product path:

```bash
python -m pip install -e '.[hf]'
crowdtensor preview-v04 local-smoke --max-new-tokens 2 --json
crowdtensor preview-v04 local-smoke --run-optional-model --require-optional-model-ready --optional-model-id distilgpt2 --max-new-tokens 2 --json
crowdtensor preview-v04 package --target kaggle --public-host 24.199.118.54 --json
crowdtensor preview-v04 evidence-import \
  --product-mvp-report dist/public-swarm-preview-v04-distilgpt2-strict/product-mvp/product_swarm_mvp_check.json \
  --optional-model-report dist/public-swarm-preview-v04-distilgpt2-strict/optional-model-mvp/product_swarm_mvp_check.json \
  --require-optional-model-ready \
  --json
python scripts/public_swarm_preview_v04_check.py --mode evidence-import --require-optional-model-ready --json
```

`crowdtensor preview-v04` emits `public_swarm_preview_v04_v1` through `scripts/public_swarm_preview_v04_pack.py`. It imports retained Live Preview RC stage0/stage1 external requeue reports, retained GPU multi-token generation evidence, local Product Swarm MVP evidence, and optional `distilgpt2` / `gpt2` strict evidence into one redacted JSON/Markdown/support-bundle set. A ready report should include `public_swarm_preview_v04_ready`, `external_two_stage_generation_ready`, `multi_token_generation_ready`, `distinct_stage_miners`, `stage_assignment_valid`, `stage_latency_ready`, `throughput_summary_ready`, `memory_or_vram_summary_ready`, `external_stage_requeue_ready`, `tiny_gpt2_ci_fallback_ready`, top-level `output_request`, `answer_scope.scope_state: no-local-answer`, `shareable_summary.answer_scope_state`, and optional `optional_distilgpt2_or_gpt2_strict_ready`. The retained ready artifact is `dist/public-swarm-preview-v04-final/public_swarm_preview_v04.json`; the fresh strict CPU `distilgpt2` proof is `dist/public-swarm-preview-v04-distilgpt2-strict/public_swarm_preview_v04.json`. Treat this as shareable Coordinator-backed preview evidence, not a local answer transcript: run `crowdtensor generate` in human mode to see local generated text, while `preview-v04` public artifacts keep raw prompts, generated text, generated token ids, activations, credentials, leases, private env files, and runtime state redacted. Rotate tokens after live public HTTP/Kaggle proofs and do not claim production Swarm Inference, libp2p/DHT/NAT traversal, Hivemind/Petals parity, arbitrary prompt serving, or large-model serving.

P2P Swarm Inference v0.6 is the first Coordinator-to-P2P transition prototype:

```bash
crowdtensor p2pd --host 0.0.0.0 --port 9560 --run
crowdtensor serve --p2p --peer-bootstrap http://YOUR_HOST:9560 --public-host YOUR_HOST --port 9561 --run
crowdtensor join --p2p --peer-bootstrap http://YOUR_HOST:9560 --stage stage0 --miner-id p2p-stage0 --run
crowdtensor join --p2p --peer-bootstrap http://YOUR_HOST:9560 --stage stage1 --miner-id p2p-stage1 --run
crowdtensor generate --p2p --peer-bootstrap http://YOUR_HOST:9560 --prompt "CrowdTensor P2P v0.6" --max-new-tokens 2 --json
crowdtensor p2p-swarm-v06 local-smoke --json
crowdtensor p2p-swarm-v06 package --public-host YOUR_HOST --json
crowdtensor p2p-swarm-v06 external-existing --peer-bootstrap http://YOUR_HOST:9560 --json
crowdtensor p2p-swarm-v06 external-existing --peer-bootstrap http://YOUR_HOST:9560 --verify-generate --admin-token "$CROWDTENSOR_ADMIN_TOKEN" --json
crowdtensor p2p-swarm-v06 kaggle-auto --public-host YOUR_HOST --json
crowdtensor p2p-swarm-v06 evidence-import --p2p-discovery-report dist/p2p-swarm-inference-v06-local-smoke/p2p_swarm_inference_v06.json --json
python scripts/p2p_swarm_inference_v06_check.py --mode local-smoke --json
python scripts/p2p_swarm_inference_v06_check.py --mode evidence-import --hf-model-id distilgpt2 --json
```

`crowdtensor p2p-swarm-v06` emits `p2p_swarm_inference_v06_v1` through `scripts/p2p_swarm_inference_v06_pack.py`. `local-smoke` starts a real local `p2pd`, announces a Coordinator plus stage0/stage1 Miner capabilities through `serve --p2p` and `join --p2p`, verifies P2P route selection, runs real tiny-GPT generation when optional `[hf]` dependencies are installed, and runs local stage rescue rediscovery with short-lived victim peers followed by rescue peers. `external-existing` verifies an already-running external P2P bootstrap catalog and may run a live `generate --p2p` when `--verify-generate --admin-token` are supplied. `kaggle-auto` is side-effectful: it starts temporary public p2pd/Coordinator processes, pushes private Kaggle stage0/stage1 CPU kernels, waits for P2P stage discovery, runs `generate --p2p`, deletes the kernels, cleans local private kernel payloads, and emits `p2p_swarm_inference_v06_kaggle_auto_ready` on success. The report should preserve `p2p_swarm_inference_v06_ready`, `p2p_discovery_routing_prototype_ready`, `local_three_process_p2p_discovery_ready`, `p2p_stage_discovery_ready`, `p2p_generate_route_ready`, `p2p_stage_rescue_ready`, `p2p_real_generate_ready`, `p2p_real_stage_rescue_ready`, `p2p_v06_model_metadata_ready`, `external_p2p_stage_discovery_ready`, `external_p2p_generate_verified`, `kaggle_kernels_deleted`, `coordinator_to_p2p_transition_ready`, and `coordinator_result_fallback_ready` as applicable. Retained local evidence is `dist/p2p-swarm-inference-v06-local-smoke-refresh2/p2p_swarm_inference_v06.json`; retained external Kaggle proof is `dist/p2p-swarm-inference-v06-kaggle-auto-final/kaggle-auto/p2p_v06_kaggle_auto.json` with 2 generated tokens, 4 accepted ledger rows, distinct external stage Miners, and deleted kernels. Non-default `--hf-model-id` evidence imports must expose matching P2P model metadata or block with `p2p_v06_model_metadata_mismatch`. If optional `[hf]` dependencies are absent, real generation reports should emit `p2p_real_generate_hf_runtime_missing` or `host_hf_runtime_missing` rather than claiming tiny-GPT execution. This is P2P discovery/routing prototype evidence; it is not production NAT traversal, decentralized security, an economic system, Hivemind/Petals parity, or large-model throughput.

```bash
read -r CROWDTENSOR_P2P_PEER_SECRET < <(python -c 'import secrets; print(secrets.token_urlsafe(32))')
export CROWDTENSOR_P2P_PEER_SECRET
crowdtensor p2pd --host 0.0.0.0 --port 9660 --peer-secret "$CROWDTENSOR_P2P_PEER_SECRET" --require-signed --run
crowdtensor serve --p2p --peer-bootstrap http://YOUR_HOST:9660 --peer-secret "$CROWDTENSOR_P2P_PEER_SECRET" --public-host YOUR_HOST --port 9661 --run
crowdtensor join --p2p --peer-bootstrap http://YOUR_HOST:9660 --peer-secret "$CROWDTENSOR_P2P_PEER_SECRET" --stage stage0 --miner-id p2p-stage0 --run
crowdtensor join --p2p --peer-bootstrap http://YOUR_HOST:9660 --peer-secret "$CROWDTENSOR_P2P_PEER_SECRET" --stage stage1 --miner-id p2p-stage1 --run
crowdtensor generate --p2p --peer-bootstrap http://YOUR_HOST:9660 --prompt "CrowdTensor Public P2P v1 RC" --max-new-tokens 2 --json
crowdtensor public-p2p-v1-rc local-smoke --json
crowdtensor public-p2p-v1-rc kaggle-auto --public-host YOUR_HOST --kaggle-owner YOUR_KAGGLE_USER --json
python scripts/public_p2p_swarm_inference_v1_rc_check.py --mode kaggle-auto --json
```

`crowdtensor public-p2p-v1-rc` emits `public_p2p_swarm_inference_v1_rc_v1` through `scripts/public_p2p_swarm_inference_v1_rc_pack.py`. It wraps the v0.6 route with shared-secret HMAC peer identity, signed peer announcements, `p2pd --require-signed`, signed `serve --p2p` and `join --p2p` announcements, registry health counts, model-compatible v0.6 summaries, a public runbook, redacted Support Bundle, and private Kaggle payload cleanup. The fresh retained signed external proof is `dist/public-p2p-v1-rc-kaggle-auto-signed-r2/public_p2p_swarm_inference_v1_rc.json`; it verified external signed stage0/stage1 discovery, `generate --p2p`, 2 generated tiny-GPT tokens, private Kaggle kernel deletion, local private payload cleanup, and `token_rotation_required`. Stage rescue is verified by the signed local P2P proof and retained external requeue evidence; a fresh signed Kaggle victim/rescue proof is not yet part of this RC. Preserve `public_p2p_swarm_inference_v1_rc_ready`, `signed_peer_announcement_ready`, `peer_identity_ready`, `peer_registry_health_ready`, `local_signed_p2p_discovery_ready`, `public_p2p_v1_rc_model_metadata_ready`, `external_p2p_generate_verified`, `kaggle_kernels_deleted`, `p2p_v06_kaggle_private_artifacts_cleaned`, and `token_rotation_required`. Non-default `--hf-model-id` imports must expose matching signed local/external/Kaggle v0.6 `hf_model_id` metadata or block with the corresponding Public P2P v1 RC model mismatch diagnostics. This is a Petals-style public preview shape with HTTP P2P-lite discovery and Coordinator lease/result fallback; it is not production Hivemind/Petals parity, libp2p, DHT, NAT traversal, decentralized security, an economic system, arbitrary prompt serving, or large-model throughput.

```bash
crowdtensor home-infer --scenario-id route-baseline --json
```

The home inference command emits `home_inference_cli_v1` and writes `home_compute_evidence_v1` artifacts under `dist/home-infer`. It wraps `scripts/home_compute_evidence_pack.py`, runs the CPU-only read-only `model_bundle_infer` route, and summarizes `route_decision`, fixed `model_bundle_inference_scenario_v1` metadata, `diagnosis_codes`, safe `request_trace` count, throughput, read-only status, redaction status, and generated artifact paths.

```bash
crowdtensor llm-infer --mock --json
```

The LLM inference command emits `llm_inference_cli_v1` and writes `external_llm_evidence_v1` artifacts under `dist/llm-infer`. It runs the read-only `external_llm_infer` contract against the deterministic mock runtime by default, or against an explicit operator-owned `--llm-runtime-cmd` / `--llm-runtime-url`. It reports adapter kind, model id, request count, completion count, output chars, throughput, and diagnosis codes while keeping raw prompts, `output_text`, runtime URL, API key, lease token, and idempotency material out of public artifacts.

Built-in scenario IDs are `route-baseline`, `gradient-safety`, and `mixed-prompts`. Use this after `crowdtensor local-proof` when a new user needs a compact, shareable proof that their machine can run the current Swarm Inference-shaped contract. It is not production Swarm Inference, not arbitrary prompt serving, and not a real LLM/GPU/P2P path.

## Artifact Cleanup

```bash
crowdtensor clean-artifacts --json
```

The cleanup command emits `cleanup_report_v1` and defaults to dry-run. It reports generated `__pycache__` / `.pyc` caches and old CrowdTensor temporary directories such as local proof or demo manifest scratch outputs.

After reviewing the report:

```bash
crowdtensor clean-artifacts --apply --json
```

Reports are protected by default: `/tmp/crowdtensor_*.json` and `/tmp/crowdtensor_*.md` are only eligible when `--include-reports` is passed. The cleanup path does not delete state, source files, release artifacts, private env files, symlinks, or paths outside the repo and `/tmp`.

## Start Coordinator

```bash
crowdtensord \
  --host 127.0.0.1 \
  --port 8787 \
  --state-dir state
```

With local tokens:

```bash
crowdtensord \
  --host 127.0.0.1 \
  --port 8787 \
  --state-dir state \
  --miner-token local-miner \
  --observer-token local-observer \
  --admin-token local-admin
```

To run the experimental OpenDiLoCo-inspired Nesterov outer update for new dense model state:

```bash
crowdtensord \
  --host 127.0.0.1 \
  --port 8787 \
  --state-dir state-nesterov \
  --outer-optimizer diloco_nesterov
```

To avoid storing a usable token directly in Coordinator config, generate a hashed token verifier:

```bash
python3 scripts/hash_token.py local-miner
```

Then pass the printed `sha256:` value to `--miner-token`, `--observer-token`,
`--admin-token`, a per-Miner registry entry, or a per-operator registry entry.
Clients still send the original token.

For role-scoped operator access, prefer the product invite helper instead of
hand-writing token hashes:

```bash
crowdtensor operator-invite \
  --registry state/operator_registry.json \
  --operator-id generate-desk \
  --role admin \
  --label "bounded generation desk" \
  --allowed-workload real-llm-sharded \
  --max-request-count 2 \
  --max-new-tokens 8 \
  --max-active-sessions 4 \
  --max-total-sessions 100 \
  --rate-limit 30 \
  --rate-window-seconds 60 \
  --invite-file state/private/generate-desk.operator.invite.json \
  --json
```

The CLI report is safe to paste into operator notes because it omits plaintext
operator tokens and invite codes. The private invite file contains the
operator's plaintext token and should stay local/private. For private automation
only, `scripts/create_operator_invite.py --json` can print the full invite,
including the plaintext token, while still writing only the `sha256:` verifier
to the registry.

The registry format is:

```json
{
  "operators": [
    {
      "operator_id": "accounting-desk",
      "token": "sha256:ACCOUNTING_DIGEST",
      "roles": ["accounting"]
    },
    {
      "operator_id": "generate-desk",
      "token": "sha256:GENERATE_DIGEST",
      "roles": ["admin"],
      "session_policy": {
        "allowed_workloads": ["real_llm_sharded_infer"],
        "max_request_count": 2,
        "max_decode_steps": 1,
        "max_new_tokens": 8,
        "max_active_sessions": 4,
        "max_total_sessions": 100,
        "rate_limit": 30,
        "rate_window_seconds": 60
      }
    }
  ]
}
```

Start the product Coordinator with:

```bash
crowdtensor serve \
  --operator-token-registry state/operator_registry.json \
  --miner-token-registry state/miner_registry.json \
  --inference-session-rate-limit 30 \
  --inference-session-rate-window-seconds 60 \
  --observer-token sha256:OBSERVER_DIGEST \
  --run
```

The lower-level Coordinator entrypoint accepts the same registry:

```bash
python3 coordinator.py \
  --operator-token-registry state/operator_registry.json \
  --miner-token sha256:MINER_DIGEST \
  --observer-token sha256:OBSERVER_DIGEST
```

The role split is intentionally small: `accounting` can read
`/admin/accounting` and `/admin/settlement`; `auditor` can read
`/admin/events`, `/admin/results`, and `/admin/session-stream`; `admin` and
`owner` keep full admin access. The legacy `--admin-token` remains supported as
owner-level access for existing scripts. When `crowdtensor serve` is started
with an operator registry and no explicit `--admin-token` or
`CROWDTENSOR_ADMIN_TOKEN`, it does not add the local default admin token.
Operators with `admin` or `owner` roles can also carry an optional
`session_policy`. It limits `/admin/inference-sessions` creates for that
operator by workload allowlist, `request_count`, `decode_steps`,
`max_new_tokens`, active queued/leased session count, cumulative session count,
and a per-operator create rate window. Policy blocks append a safe
`control_plane_blocked` audit event and never expose plaintext tokens.
The global session create rate limit remains available as a Coordinator-wide
fallback by returning `429` after too many creates by the same legacy admin or
operator subject in the configured window.

## Run Miner

```bash
read -r -s -p 'Miner token: ' CROWDTENSOR_MINER_TOKEN; echo
export CROWDTENSOR_MINER_TOKEN
crowdtensor-miner \
  --coordinator http://127.0.0.1:8787 \
  --miner-id local-1 \
  --max-tasks 10 \
  --compute-seconds 0.2
```

Useful flags:

- `--once`: process one task and exit
- `--max-tasks N`: stop after N accepted tasks
- `--max-runtime-seconds N`: stop after a wall-clock budget
- `--heartbeat-interval N`: tune heartbeat cadence
- `--skip-preflight`: skip the startup `/ready` compatibility check
- `--max-request-attempts N`: retry transient `/ready`, claim, heartbeat, and idempotent result upload failures
- `--enable-mock-llm-runtime`: advertise `external_llm_infer` with the deterministic mock runtime
- `--llm-runtime-cmd CMD`: advertise `external_llm_infer` with an operator-owned command wrapper; `CROWDTENSOR_LLM_RUNTIME_CMD` is also supported
- `--llm-runtime-url URL`: advertise `external_llm_infer` with an OpenAI-compatible chat completions endpoint; `CROWDTENSOR_LLM_RUNTIME_URL` is also supported
- `--llm-runtime-api-key TOKEN`: optional bearer token for `--llm-runtime-url`; `CROWDTENSOR_LLM_RUNTIME_API_KEY` is also supported and is not advertised in capabilities
- `--idle-sleep N`: sleep between failed or unavailable claims

The Miner summary includes `request_retries` and `preflight_failures` so operators can spot unstable links without parsing stderr.

Result uploads include an `idempotency_key`, so a lost response can be retried without applying the same model update twice.

## Health and Metrics

```bash
curl http://127.0.0.1:8787/health
```

```bash
curl http://127.0.0.1:8787/version
```

```bash
curl http://127.0.0.1:8787/ready
```

```bash
curl -H 'x-crowdtensor-observer-token: local-observer' \
  http://127.0.0.1:8787/state
```

```bash
curl -H 'x-crowdtensor-observer-token: local-observer' \
  http://127.0.0.1:8787/metrics
```

`/metrics` is aggregate-only and avoids lease tokens, task payloads, Miner IDs, and raw Miner metadata.

Admin result ledger:

```bash
curl -H 'x-crowdtensor-admin-token: local-admin' \
  'http://127.0.0.1:8787/admin/results?status=rejected&limit=20'
```

`GET /admin/results` is the safest operator view for result traceability. It includes validation, replay audit, model impact, and Miner workload score summaries, but avoids raw lease tokens, idempotency material, full result responses, and tensor deltas.
For sessions created through `/admin/inference-sessions`, rows include a safe
`created_by_subject` label such as `legacy-admin` or `operator:<operator_id>`
so operators can trace usage back to the creating admin subject without
publishing plaintext operator tokens.

Admin accounting summary:

```bash
curl -H 'x-crowdtensor-admin-token: local-admin' \
  'http://127.0.0.1:8787/admin/accounting?status=accepted&limit=20'
```

`GET /admin/accounting` is the safest operator view for Miner-level usage and
Beta reward/accounting preparation. It groups safe work units by Miner and
workload, joins redacted invite policy metadata such as trust tier, quota, and
claim-rate limits when present, carries `created_by_subject` on individual
session-created rows plus `created_by_subject_totals` for chargeback
attribution, supports exact `created_by_subject` filtering for one subject's
usage export, and avoids raw prompts, outputs, token ids, activations, lease
material, plaintext tokens, and reward account values.

Admin settlement draft:

```bash
curl -H 'x-crowdtensor-admin-token: local-admin' \
  'http://127.0.0.1:8787/admin/settlement?unit_price_microcredits=1&limit=20'
```

`GET /admin/settlement` is an accepted-result-only settlement draft for Beta
operator accounting. It converts safe work units into reward units and optional
microcredit amounts, joins only redacted invite policy metadata, and always
reports `draft_only=true` plus `payment_executed=false`. It does not expose
reward account values and does not perform billing, staking, or payouts. Rows
retain `created_by_subject` when the accepted work came from an admin-created
inference session, and `created_by_subject_totals` aggregates those accepted
rows by subject/workload without including anonymous background tasks. Use the
exact `created_by_subject` query parameter to draft settlement rows for one
operator/admin subject.

Operator status CLI:

```bash
crowdtensor operator-status \
  --coordinator-url http://127.0.0.1:8787 \
  --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" \
  --output-dir dist/operator-status
```

`crowdtensor operator-status` is the read-only first check for a running
Coordinator. It emits `crowdtensor_operator_status_cli_v1`, writes
`operator_status.json` and `operator_status.md`, and summarizes `/ready`
operator registry state, Miner registry policy state, `/state` trust/quarantine
and `blocked_claims`, and next actions. Add `--include-admin-summaries` with an
owner/admin/accounting token to include safe `/admin/accounting` and
`/admin/settlement` status without exposing credentials, prompts, outputs,
lease material, or reward account values:

```bash
CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} \
  crowdtensor operator-status \
    --coordinator-url http://127.0.0.1:8787 \
    --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" \
    --include-admin-summaries \
    --unit-price-microcredits 1
```

Use `--require-state` or `--require-admin-summaries` when automation should fail
unless those protected views are reachable. The command is read-only: it does
not create inference sessions, set trust overrides, execute billing, stake,
slash, or pay Miners.

Operator settlement CLI:

```bash
CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} \
  crowdtensor settlement \
    --coordinator-url http://127.0.0.1:8787 \
    --include-accounting \
    --unit-price-microcredits 1 \
    --output-dir dist/settlement
```

`crowdtensor settlement` wraps those two admin endpoints for ordinary
accounting operators. It emits `crowdtensor_settlement_cli_v1`, writes
`settlement_summary.json` and `settlement_summary.md`, supports the same exact
Miner/workload/session/`created_by_subject` filters, optionally includes the
accounting summary with `--include-accounting`, and keeps the admin token,
reward account values, raw prompts, outputs, and lease material out of saved
artifacts. The report is still draft-only: no billing, staking, payout, or
automatic settlement is executed.

Operator trust review:

```bash
crowdtensor trust \
  --coordinator-url http://127.0.0.1:8787 \
  --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" \
  --miner-id stage0-miner \
  --workload-type real_llm_sharded_infer \
  --output-dir dist/trust
```

`crowdtensor trust` emits `crowdtensor_trust_cli_v1`, reads `/state`, and writes
`trust_summary.json` plus `trust_summary.md` with automatic quarantine counts,
manual allow/block overrides, effective blocked Miner/workload pairs,
`blocked_claims`, and a safe row view for the selected Miner/workload. It does
not require an admin token for report-only mode unless the Coordinator protects
`/state` with an observer token.

Operator trust override:

```bash
CROWDTENSOR_ADMIN_TOKEN=${CROWDTENSOR_ADMIN_TOKEN:?set CROWDTENSOR_ADMIN_TOKEN} \
  crowdtensor trust \
    --coordinator-url http://127.0.0.1:8787 \
    --miner-id stage0-miner \
    --workload-type real_llm_sharded_infer \
    --mode block \
    --reason "operator review"
```

Use `--mode allow` to let a named Miner/workload bypass automatic quarantine,
and `--mode reset` to clear the manual override and return to automatic scoring.
The override path requires a legacy admin token or an operator token with
`owner`/`admin` access. Saved artifacts do not include the admin token, observer
token, raw override reason, lease material, prompts, or outputs. This is a
manual operator safety control; it is not Sybil resistance, staking, slashing,
automatic economic penalties, or permissionless trust.

## Acceptance Checks

First-run Doctor:

```bash
python3 scripts/doctor.py --json
```

Remote and browser dependency probes:

```bash
python3 scripts/doctor.py --remote-demo --browser --json
```

Alpha release gate:

```bash
python3 scripts/release_gate.py --json
```

Security preflight:

```bash
python3 scripts/security_preflight.py --json
```

Remote demo preflight:

```bash
python3 scripts/security_preflight.py \
  --host 0.0.0.0 \
  --miner-token-registry state/miner_registry.json \
  --observer-token sha256:OBSERVER_DIGEST \
  --admin-token sha256:ADMIN_DIGEST \
  --json
```

Readiness/profile smoke:

```bash
python3 scripts/readiness_check.py --port 8890
```

API contract smoke:

```bash
python3 scripts/api_contract_check.py --port 8891
```

Miner resilience smoke:

```bash
python3 scripts/miner_resilience_check.py --port 8894
```

Result idempotency smoke:

```bash
python3 scripts/result_idempotency_check.py --port 8896
```

Result ledger smoke:

```bash
python3 scripts/result_ledger_check.py --port 8897
```

Remote Miner invite/join smoke:

```bash
python3 scripts/remote_miner_join_check.py --port 8898
```

Unit tests:

```bash
python3 -m unittest discover -s tests -v
```

Default runtime acceptance:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8910 \
  --report /tmp/crowdtensor_acceptance.json
```

Outer optimizer contract smoke:

```bash
python3 scripts/outer_optimizer_check.py --port 8899
```

Compressed error-feedback transport smoke:

```bash
python3 scripts/compressed_error_feedback_check.py --port 8900
```

Delta transport negotiation smoke:

```bash
python3 scripts/delta_transport_negotiation_check.py --port 8901
```

Model bundle LM smoke:

```bash
python3 scripts/model_bundle_smoke.py --port 8902
```

Model bundle inference smoke:

```bash
python3 scripts/model_bundle_inference_smoke.py --port 8903
```

Use `--request-count N` to exercise a multi-request read-only inference session in one task.

User-facing local inference session demo:

```bash
python3 scripts/inference_session_demo.py --port 8904 --request-count 4
```

Use `--json` for automation. The demo reports safe session metrics, a capped Coordinator-derived `request_trace`, read-only status, redaction status, and Miner `hardware_profile`; it is a CPU-only Swarm Inference shaped demo, not a real LLM serving benchmark.

Admin-created read-only inference session API:

```bash
python3 scripts/admin_inference_session_check.py --port 8915 --request-count 4
```

This smoke validates `POST /admin/inference-sessions` with `schema=inference_session_request_v1`. The endpoint creates a bounded CPU `model_bundle_infer` task and returns a `task_id` plus a `GET /admin/results?task_id=<task_id>&workload_type=model_bundle_infer` query path for safe result inspection. It is read-only: dense model and model bundle versions do not advance, and raw `inference_results`, lease tokens, and idempotency material remain out of operator JSON. The default runtime acceptance pack includes `scripts/admin_inference_session_check.py`; use `--skip-admin-inference-session` only when this service-shaped API boundary is intentionally out of scope for a local run.

Matrix-guided home-compute demo:

```bash
python3 scripts/home_compute_demo.py --port 8909 --request-count 4 --json
```

This combines runtime capability discovery with the read-only `model_bundle_infer` inference session. It selects the CPU-only workload and `local_cpu_model_bundle_infer` route only when `scripts/runtime_matrix.py` reports it as available, then emits one report with host capability, selected workload, selected route, `route_decision`, session metrics, capped `request_trace` rows, read-only status, redaction status, stable `diagnosis_codes` such as `home_compute_ready`, `runtime_matrix_blocked`, `workload_unavailable`, `cpu_route_unavailable`, `session_failed`, and `trace_missing`, and recommended next commands. `scripts/home_compute_demo_check.py` is included in the default acceptance pack and can be skipped with `--skip-home-compute-demo`.

Safe, shareable home-compute evidence pack:

```bash
python3 scripts/home_compute_evidence_pack.py \
  --port 8911 \
  --request-count 4 \
  --runtime-report /tmp/crowdtensor_acceptance.json \
  --json-out /tmp/crowdtensor_home_evidence.json \
  --markdown-out /tmp/crowdtensor_home_evidence.md
```

The `home_compute_evidence_v1` report is the preferred operator artifact for showing the current home-compute path. It combines the runtime matrix, selected workload, `route_decision`, `matched_capabilities`, capped `request_trace`, `diagnosis_codes`, safety flags, and optional runtime acceptance summary into a safe, shareable JSON/Markdown pair. It redacts token, URL, API key, lease, idempotency, weight, and delta-shaped fields. `scripts/home_compute_evidence_check.py` is included in the default acceptance pack and can be skipped with `--skip-home-compute-evidence`.

Runtime capability matrix:

```bash
python3 scripts/runtime_matrix.py --json
```

The runtime capability matrix reports CPU-only baseline readiness, optional browser support, optional external LLM runtime configuration, and a hardware/runtime matrix through `hardware_targets`, `recommended_routes`, `matched_capabilities`, `missing_capabilities`, target and route `diagnosis_codes`, `operator_action`, top-level `diagnosis_summary`, and `hardware_diagnosis_summary`. NVIDIA CUDA is a narrow optional `hf_transformers_cuda` tiny GPT split route: CPU-only hosts report `nvidia_cuda_optional_missing`, partial GPU hosts report `nvidia_cuda_detected_adapter_unavailable`, and only a complete CUDA + optional HF runtime can expose the `local_cuda_real_llm_sharded_infer` recommended route to `crowdtensor public-swarm-gpu-beta local-loopback`. AMD, Apple, browser, and remote container targets may still be detected without being usable runtime adapters. `scripts/runtime_matrix_check.py` is included in the default acceptance pack and can be skipped with `--skip-runtime-matrix`. It notes whether `CROWDTENSOR_LLM_RUNTIME_URL` is configured without printing the URL, token, or API key value.

External LLM adapter contract smoke:

```bash
python3 scripts/external_llm_inference_smoke.py --port 8906 --request-count 3
```

OpenAI-compatible HTTP adapter smoke:

```bash
python3 scripts/external_llm_http_adapter_smoke.py --port 8907 --runtime-port 8908
```

The default smoke uses `crowdtensor-miner --enable-mock-llm-runtime` so it is deterministic and CPU-only. For an operator-provided local runtime, run the Miner with `--llm-runtime-cmd /path/to/wrapper` or set `CROWDTENSOR_LLM_RUNTIME_CMD=/path/to/wrapper`; the wrapper receives `prompt` and `max_tokens` arguments and should print completion text to stdout. For OpenAI-compatible local servers, use `--llm-runtime-url http://127.0.0.1:11434/v1/chat/completions` or `CROWDTENSOR_LLM_RUNTIME_URL=...`, with optional `--llm-runtime-api-key` / `CROWDTENSOR_LLM_RUNTIME_API_KEY`. `external_llm_infer` is read-only, validates `external_llm_infer_v1` prompt hashes and `external_llm_results`, records `request_count`, `completion_count`, `output_chars`, `adapter_kind`, `model_id`, and `requests_per_second`, and keeps raw prompts and `output_text` out of `/state` and admin result ledger summaries.

Safe external LLM evidence artifact:

```bash
python3 scripts/external_llm_evidence_pack.py \
  --mock \
  --port 8919 \
  --request-count 3 \
  --json-out /tmp/crowdtensor_external_llm_evidence.json \
  --markdown-out /tmp/crowdtensor_external_llm_evidence.md
```

`external_llm_evidence_v1` is the shareable proof layer for `external_llm_infer`: it records the local route, adapter summary, ledger row count, read-only status, redaction status, and `external_llm_evidence_ready`. Validate it with `scripts/external_llm_evidence_check.py --port 8919`; skip the runtime acceptance check with `--skip-external-llm-evidence` only when the lane intentionally omits local external LLM evidence.

Remote-style Miner readiness:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8920 \
  --include-remote-miner \
  --miner-token local-miner \
  --observer-token local-observer \
  --report /tmp/crowdtensor_remote_acceptance.json
```

The remote readiness smoke verifies `diloco_train`, `cpu_lora_mock`, `micro_transformer_lm`, and `model_bundle_lm` in one long-running Python Miner session. The default runtime acceptance pack separately verifies the read-only `model_bundle_infer` Swarm Inference shaped probe and the optional `external_llm_infer` adapter contract.

Remote-compute evidence for a read-only inference result:

```bash
python3 scripts/remote_compute_evidence_pack.py \
  --port 8912 \
  --request-count 4 \
  --json-out /tmp/crowdtensor_remote_evidence.json \
  --markdown-out /tmp/crowdtensor_remote_evidence.md
```

The `remote_compute_evidence_v1` report is a safe, shareable proof that a registry-backed remote-style Python Miner completed the read-only `model_bundle_infer` route `remote_python_model_bundle_infer`. It records route capabilities, safe metrics, capped `request_trace`, ledger summary, read-only status, redaction status, hashed registry status, and `remote_compute_observability_v1` for route, queue, inference, and safety visibility. In a full acceptance run, add `--include-remote-evidence`; this remains opt-in because it starts an additional Coordinator/Miner pair.

Controlled local multi-Miner scenario sweep:

```bash
python3 scripts/multi_miner_scenario_sweep_check.py \
  --port 8916 \
  --execution-mode concurrent \
  --scenario-ids route-baseline,gradient-safety,mixed-prompts
```

`scripts/multi_miner_scenario_sweep.py` starts a local Coordinator with no preload backlog, creates one read-only `POST /admin/inference-sessions` task per fixed `model_bundle_inference_scenario_v1`, and can start one registry-backed Python Miner identity per task concurrently. The resulting `multi_miner_scenario_sweep_v1` artifact records `local_multi_miner_model_bundle_infer`, scenario match status, per-session safe latency/throughput, Miner distribution, `lease_summary`, `process_summary`, `multi_miner_scenario_sweep_observability_v1`, read-only status, redaction status, hashed registry status, and `multi_miner_concurrent_ready` diagnostics in concurrent mode. Add `--failure-mode kill-after-claim` to terminate one claimed task before upload, observe lease timeout requeue, and require a rescue Miner to complete the same `task_id` with `multi_miner_requeue_ready`. Add `--include-multi-miner-sweep` to runtime acceptance for the happy path and `--include-multi-miner-requeue` for the failure path. This remains local loopback evidence, not P2P, NAT traversal, production throughput scaling, GPU pooling, or production Swarm Inference.

Local-loopback Demo Manifest:

```bash
python3 scripts/demo_manifest_pack.py \
  --output-dir dist/demo-manifest \
  --port 8914 \
  --request-count 4
```

`scripts/demo_manifest_pack.py` produces the `demo_manifest_v1` latest output artifact for operator handoff. It writes `runtime_matrix.json`, `remote_compute_evidence_v1`, `external_llm_evidence_v1`, `support_bundle`, `demo_manifest.json`, and `demo_manifest.md` under one output directory, then summarizes `remote_compute_observability_v1` and deterministic mock external LLM evidence without embedding raw state, raw prompts, `output_text`, runtime URL, API key, tokens, leases, idempotency material, or tensor payloads. Validate the path with `scripts/demo_manifest_check.py --base-port 8914`.

Recommended two-machine remote home-compute demo:

```bash
crowdtensor remote-demo prepare \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --scenario-id route-baseline \
  --output-dir dist/remote-home-compute \
  --json
```

Then start the generated Coordinator command on the Coordinator host and the generated Miner command on the Miner host, source `operator.private.env`, run doctor, verify, and collect:

```bash
crowdtensor remote-demo doctor \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" \
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \
  --scenario-id route-baseline \
  --output-dir dist/remote-home-compute \
  --json

crowdtensor remote-demo verify \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" \
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \
  --scenario-id route-baseline \
  --output-dir dist/remote-home-compute \
  --json

crowdtensor remote-demo collect \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" \
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \
  --scenario-id route-baseline \
  --output-dir dist/remote-home-compute \
  --json
```

`crowdtensor remote-demo` is the high-level operator path for the controlled home-compute remote Miner demo. It emits `remote_home_compute_demo_v1`, delegates to `scripts/remote_home_compute_demo_pack.py`, keeps `operator.private.env` and `miner.private.env` private, creates a read-only `model_bundle_infer` session through `POST /admin/inference-sessions`, verifies the `remote_python_model_bundle_infer` route, and summarizes `remote_compute_evidence_v1` plus `remote_demo_observability_v1`. Prepare now also writes a user-facing `miner_join_pack_v1` as `miner_join.sh` and `MINER_JOIN.md`; copy only those files plus `miner.private.env` to the Miner host. `remote-demo doctor`, `remote-demo collect`, and `remote-demo clean` emit `remote_home_compute_doctor_v1`, `remote_home_compute_collect_v1`, and `remote_home_compute_cleanup_v1`; cleanup defaults to dry-run and only removes known generated files, with private env/registry files gated behind `--include-private`. `scripts/remote_home_compute_demo_check.py` validates the local-loopback stand-in in CI. This remains not production Swarm Inference, not P2P routing, and not GPU pooling.

Kaggle Remote Miner Beta uses the same operator path with a Kaggle target:

```bash
crowdtensor remote-demo prepare \
  --target kaggle \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id kaggle-cpu-1 \
  --output-dir dist/remote-home-compute-kaggle \
  --json
```

The prepare report creates `kaggle_remote_miner.py`, `kaggle_remote_miner.md`, `miner_join.sh`, `MINER_JOIN.md`, and `miner.private.env`, emits `kaggle_remote_miner_prepare_ready` plus `miner_join_pack_ready`, and keeps `operator.private.env` off Kaggle. Run `python scripts/kaggle_remote_miner_beta_check.py --port 9060` for the CI-safe aggregate check; it emits `kaggle_remote_miner_beta_check_v1` and `kaggle_remote_miner_beta_ready` after validating `--target kaggle`, token redaction, generated Kaggle artifacts, and the local loopback remote-demo protocol. This path treats Kaggle as an outbound temporary CPU Miner, not a Coordinator host, not production Swarm Inference, not P2P, and not a GPU/TPU workload path.

Kaggle Real Runtime Acceptance is the operator path for proving that a live Kaggle CPU Notebook can reach a public Coordinator. The current temporary HTTP public host is `24.199.118.54`:

```bash
crowdtensor remote-demo kaggle-real \
  --action prepare \
  --public-host 24.199.118.54 \
  --port 9180 \
  --miner-id kaggle-cpu-1 \
  --output-dir dist/kaggle-real-runtime \
  --json

bash dist/kaggle-real-runtime/start_coordinator.sh
```

Upload only `dist/kaggle-real-runtime/kaggle-upload/miner.private.env` and `dist/kaggle-real-runtime/kaggle-upload/kaggle_remote_miner.py` to Kaggle. Do not upload `operator.private.env`. After the Notebook runs `python kaggle_remote_miner.py --env-file miner.private.env`, verify from the operator host:

```bash
crowdtensor remote-demo kaggle-real \
  --action verify \
  --public-host 24.199.118.54 \
  --port 9180 \
  --output-dir dist/kaggle-real-runtime \
  --json
```

The workflow emits `kaggle_real_runtime_acceptance_v1` through `scripts/kaggle_real_runtime_acceptance_pack.py`. `scripts/kaggle_real_runtime_acceptance_check.py` is CI-safe and checks generated artifacts only. A true live acceptance requires `kaggle_artifacts_ready`, `coordinator_public_ready`, `kaggle_miner_seen`, `kaggle_result_accepted`, and `kaggle_real_runtime_ready`. The report preserves `token_rotation_required`, uses temporary HTTP only for this controlled proof, and remains CPU-only, read-only, not production, not P2P, and not GPU/TPU workload execution.

For the stage-aware micro-LLM proof, run the same wrapper with `--workload micro-llm-sharded --stage-mode split --decode-steps 3`. Prepare creates `kaggle-upload-stage0` and `kaggle-upload-stage1`, each with its own `miner.private.env` and stage-specific `kaggle_remote_miner.py`; upload those directories to two separate private Kaggle Notebooks. Verify with the same workload flags. A live split success adds `kaggle_micro_llm_stage0_seen`, `kaggle_micro_llm_stage1_seen`, `kaggle_micro_llm_stage_assignment_valid`, `stage_assignment_valid`, and `kaggle_micro_llm_sharded_ready`.

When using the Kaggle CLI instead of manual Notebook uploads, `scripts/kaggle_micro_llm_live_package.py` builds private dataset and script-kernel upload folders from an existing `kaggle-real` prepare directory. The normal mode stores `crowdtensor_source.tar.gz` and stage env files in the private dataset and points two private kernels at it. If Kaggle input mounting is unreliable, `--inline-kernel-payload` embeds the source tarball and each stage `miner.private.env` directly into the corresponding private kernel source. Inline mode is operator-only and temporary: generated kernel code contains a usable Miner token, so keep it out of Git and public artifacts, delete the remote Kaggle kernels/dataset after the run, and rotate the tokens. This remains a toy two-stage pipeline, not large-model sharding, not GGUF/llama.cpp serving, and not production Swarm Inference.

The first artifact-backed live Kaggle split run completed against `http://24.199.118.54:9180` with two private Kaggle CPU script kernels, a shared `micro_llm_artifact_v1`, and final evidence at `dist/kaggle-micro-llm-live/external-real/kaggle_real_runtime_acceptance.json`. The report is `kaggle_real_runtime_acceptance_v1` with `ok: true`, `artifact_loaded`, `micro_llm_artifact_ready`, `kaggle_micro_llm_stage0_seen`, `kaggle_micro_llm_stage1_seen`, `kaggle_micro_llm_stage_assignment_valid`, `stage_assignment_valid`, `baseline_match`, `decoded_tokens_match`, and `kaggle_micro_llm_sharded_ready`. The temporary Kaggle kernels/dataset were deleted after evidence collection.

`crowdtensor micro-llm-live-rc` is the release-candidate wrapper for this proof. Use `--mode local-generated` for the required local-generated stage upload stand-ins and `--mode external-existing` only after two real Kaggle Notebooks or two real machines are already running. The local mode must report `local_generated_stage_upload_standins_ready`; the external mode must report `external_runtime_verified` before it can be treated as live external runtime evidence.

The high-level wrapper also supports the controlled remote external LLM runtime path:

```bash
crowdtensor remote-demo prepare \
  --workload external-llm \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --mock \
  --output-dir dist/remote-home-compute-llm \
  --json

crowdtensor remote-demo verify \
  --workload external-llm \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" \
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \
  --mock \
  --output-dir dist/remote-home-compute-llm \
  --json

crowdtensor remote-demo collect \
  --workload external-llm \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" \
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \
  --mock \
  --output-dir dist/remote-home-compute-llm \
  --json
```

This creates a read-only `external_llm_infer` task through `POST /admin/inference-sessions`, verifies `remote_python_external_llm_infer`, and collects `remote_external_llm_evidence_v1` plus `remote_external_llm_observability_v1` through `scripts/remote_external_llm_evidence_pack.py`. `--mock` is deterministic for CI; explicit `--llm-runtime-cmd` or `--llm-runtime-url` remains operator-owned. The report keeps raw prompts, `output_text`, runtime URL, API key, leases, and idempotency material out of public artifacts. It is fixed-prompt runtime evidence, not public arbitrary prompt serving.

Safe two-machine remote runbook:

```bash
crowdtensor remote-runbook \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --scenario-id route-baseline \
  --output-dir dist/remote-demo \
  --json
```

The `crowdtensor/cli.py` operator wrapper emits `remote_runbook_cli_v1`, then delegates to `scripts/remote_demo_runbook_pack.py`. The underlying `remote_demo_runbook_v1` pack generates `operator.private.env`, `miner.private.env`, a hashed Miner registry, and public JSON/Markdown commands for a controlled `model_bundle_infer` demo. It keeps plaintext tokens out of the public artifact, includes the `remote_compute_evidence_pack.py --mode collect --scenario-id route-baseline` command, and records the fixed `model_bundle_inference_scenario_v1` scenario used by acceptance. It is checked by `scripts/remote_demo_runbook_check.py`.

If you intentionally reuse the same output directory and `miner_id`, pass `--replace` to rotate the generated Miner entry in that runbook registry.

Safe two-machine remote acceptance:

```bash
crowdtensor remote-acceptance \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" \
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \
  --create-session \
  --scenario-id route-baseline \
  --output-dir dist/remote-demo-acceptance \
  --json
```

The `crowdtensor/cli.py` operator wrapper emits `remote_acceptance_cli_v1`, applies token redaction to captured stdout/stderr tails, then delegates to `scripts/remote_demo_acceptance_pack.py`. The recommended mode uses `--create-session`: the pack calls `POST /admin/inference-sessions` with `request_count` and `scenario_id`, receives a `task_id`, and waits for that specific read-only `model_bundle_infer` result before collecting artifacts. Use `--no-create-session` only for the older wait-only flow against an already completed result. The `remote_demo_acceptance_v1` pack writes `remote_compute_evidence_v1`, `support_bundle`, `remote_demo_observability_v1`, selected `model_bundle_inference_scenario_v1` metadata, scenario match status, and a top-level acceptance JSON/Markdown report. Its `diagnosis_codes` give operators stable next-step triage for `coordinator_unreachable`, `observer_auth_failed`, `admin_auth_failed`, `session_create_failed`, `miner_not_seen`, `task_lane_missing`, `workload_not_advertised`, `no_accepted_result`, `validation_failed`, `request_count_mismatch`, `artifact_collection_failed`, and `acceptance_ready`. This controlled path is not production Swarm Inference and not P2P routing. `scripts/remote_demo_acceptance_check.py` validates the local stand-in path.

Lightweight inference session client for an already running Coordinator:

```bash
python3 scripts/inference_session_client.py \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \
  --request-count 4 \
  --json
```

`scripts/inference_session_client.py` is the direct operator/user entrypoint for the same read-only `POST /admin/inference-sessions` API. Its `inference_session_client_v1` output creates one CPU `model_bundle_infer` task, waits for the exact returned `task_id` in `/admin/results`, and returns safe validation, request count, throughput, and `session_client_ready` diagnostics. It reports `coordinator_unreachable`, `admin_auth_failed`, `session_create_failed`, `session_timeout`, `validation_failed`, and `request_count_mismatch` without printing raw tokens, leases, idempotency material, raw state, or raw `inference_results`. The local acceptance check is `scripts/inference_session_client_check.py`; skip it with `--skip-inference-session-client` only when a lane intentionally omits the user-facing client path.

`--miner-token` and `--observer-token` are passed only to checks that explicitly support shared auth env vars. Auth-specific smoke tests keep their own local tokens so they can validate rejection paths deterministically.

Core browser acceptance:

```bash
python3 scripts/browser_acceptance_pack.py \
  --base-port 9310 \
  --report /tmp/crowdtensor_browser_acceptance.json
```

This runs `webrtc_smoke.py`, `runtime_contract_check.py`, and `browser_miner_smoke.py`. Use `--allow-skip` when CI should skip cleanly if Playwright or Chromium is unavailable.

Broader browser acceptance:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8930 \
  --include-browser \
  --report /tmp/crowdtensor_browser_acceptance.json
```

## Release Evidence

After runtime acceptance has produced `/tmp/crowdtensor_acceptance.json`, build a local release evidence bundle:

```bash
python3 scripts/release_evidence_pack.py \
  --runtime-report /tmp/crowdtensor_acceptance.json \
  --browser-report /tmp/crowdtensor_browser_acceptance.json \
  --remote-report /tmp/crowdtensor_remote_acceptance.json \
  --json-out dist/release-evidence.json \
  --markdown-out dist/release-evidence.md
```

`scripts/release_evidence_pack.py` records the current git commit, package metadata, release gate summary, security preflight summary, and acceptance report summaries. It preserves safe runtime `summary_json` fields, top-level `diagnosis_summary` and `diagnosis_by_check` rows, and remote `observability_summaries` such as `remote_compute_observability_v1` and `remote_demo_observability_v1`, so release artifacts carry stable operator triage and remote-demo observability without raw tokens or tensor payloads. The runtime report is required. Browser and remote reports are optional by default; use `--strict-optional` when a release candidate must prove both. CI writes `release-evidence.json` and uploads it as an artifact.

## Support Bundle

For issue reports or remote-demo troubleshooting, generate a Support Bundle:

```bash
python3 scripts/support_bundle.py \
  --json-out /tmp/crowdtensor_support_bundle.json \
  --markdown-out /tmp/crowdtensor_support_bundle.md
```

To include live Coordinator summaries:

```bash
python3 scripts/support_bundle.py \
  --coordinator http://127.0.0.1:8787 \
  --observer-token local-observer \
  --admin-token local-admin \
  --runtime-report /tmp/crowdtensor_acceptance.json \
  --release-evidence /tmp/crowdtensor_release_evidence.json \
  --json-out /tmp/crowdtensor_support_bundle.json
```

`scripts/support_bundle.py` carries optional acceptance `diagnosis_summary` / `diagnosis_by_check` rows and safe remote `observability_summaries` into the issue bundle, then redacts token, lease, idempotency, weight, and delta-shaped fields. Prefer sharing this bundle over raw `state/` files, raw `/state` output, shell history, or token registry files.

For a single local-loopback handoff artifact, generate the Demo Manifest before or after release evidence:

```bash
python3 scripts/demo_manifest_pack.py \
  --output-dir dist/demo-manifest \
  --port 8914 \
  --request-count 4
```

The Demo Manifest is not a production release claim; it is a compact, safe index of the current CPU-only runtime matrix, remote-compute evidence, external LLM evidence, and support bundle outputs.

## Troubleshooting

**Address already in use**

Change `--port` or `--base-port`. The acceptance pack consumes a range of ports.

**Operation not permitted in sandbox**

Some restricted environments block localhost client sockets. Run unit tests there, then run acceptance checks on a normal shell or CI host.

**401 invalid miner token**

Confirm the Coordinator and Miner use the same `CROWDTENSOR_MINER_TOKEN`, or use the exact per-Miner token from the registry.

For registry-backed remote Miners, generate or rotate the entry with `scripts/create_miner_invite.py` and confirm the remote `--miner-id` matches the registry entry.

**401 invalid observer token**

Pass `x-crowdtensor-observer-token` when reading `/state` or `/metrics`.

**Security preflight fails**

Fix `error` findings before a remote demo. Typical causes are binding to `0.0.0.0` without Miner, observer, or admin tokens, using `local-*` demo tokens on a remote bind, or storing plaintext tokens in `--miner-token-registry`. Use `--strict` when warning-level findings should also block CI.

**503 no compatible queued task available**

The Miner capabilities do not match queued lanes. Check `--task-lane` values and the capabilities sent by the Miner/browser.

**Playwright browser not found**

Install browser dependencies or pass `--browser /path/to/chrome` to the browser smoke scripts.
