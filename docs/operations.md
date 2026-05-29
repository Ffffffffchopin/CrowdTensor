# Operations

This document collects common commands for local Alpha operation.

## Fresh Clone Onboarding Gate

```bash
python scripts/onboarding_gate.py --quick --json-out /tmp/crowdtensor_onboarding_gate.json
```

The onboarding gate emits `onboarding_gate_v1`. It creates a clean temporary virtualenv, runs `python -m pip install -e .[dev]`, checks `crowdtensor --help`, `crowdtensord --help`, and `crowdtensor-miner --help`, then smoke-validates `crowdtensor local-proof`, `crowdtensor home-infer`, `crowdtensor llm-infer --mock`, `crowdtensor cpu-infer --mode local`, and `crowdtensor release-ready --allow-dirty` with reduced request counts.

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

`crowdtensor real-llm-internet-beta` emits `real_llm_internet_beta_v1` through `scripts/real_llm_internet_beta_pack.py`. `kaggle-auto` creates the Alpha package, starts the temporary public Coordinator, pushes private Kaggle CPU script kernels, runs external-existing verification, deletes the temporary kernels, stops the Coordinator, and records the lifecycle. With `--failure-mode kill-stage0-after-claim` or `kill-stage1-after-claim`, it creates separate victim/rescue stage kernels, observes the victim claim through `/state`, deletes the victim kernel, waits for lease timeout requeue, pushes the rescue kernel, and records `external_stage_requeue_ready`, `live_stage0_requeue_ready` / `live_stage1_requeue_ready`, and `live_requeue_summary`. A ready report must include `real_llm_internet_beta_ready`, `real_llm_internet_alpha_ready`, `external_runtime_verified`, both Kaggle stages seen, `decoded_tokens_match`, `distinct_stage_miners`, `stage_assignment_valid`, `kaggle_kernels_deleted`, and `token_rotation_required`. `scripts/real_llm_internet_beta_check.py` is a fake-runner contract check for CI and does not create Kaggle resources. This is CPU-only, read-only, not production Swarm Inference, not P2P, and not large-model serving.

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

`crowdtensor preview` is the Public Swarm Developer Preview wrapper for ordinary users who want the largest single artifact over the current product surface. It emits `public_swarm_developer_preview_v1` through `scripts/public_swarm_developer_preview_pack.py` and is checked by `scripts/public_swarm_developer_preview_check.py`. Use `preview local` to run the Product Beta `serve` / `join stage0` / `join stage1` / `generate` path and require `developer_preview_ready`, `public_swarm_developer_preview_ready`, `local_two_stage_generation_ready`, `serve_join_generate_ready`, `product_beta_ready`, `support_bundle_ready`, `cpu_fallback_ready`, and `local_cpu_inference_ready`; retained GPU generation evidence adds `gpu_generation_evidence_import_ready`. Use `preview package` for two-machine or Kaggle join material, `preview external-existing` for an already running controlled runtime, and `preview evidence-import` for retained redacted Product Beta and GPU reports. Missing optional `[hf]` dependencies should surface `hf_dependencies_missing`. This is CPU-only by default, read-only, Coordinator-backed, not production Swarm Inference, not libp2p, not DHT, not NAT traversal, and not large-model serving.

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

`crowdtensor public-swarm-gpu-beta` emits `public_swarm_gpu_inference_beta_v1` through `scripts/public_swarm_gpu_inference_beta_pack.py` and is checked by `scripts/public_swarm_gpu_inference_beta_check.py`. `public-swarm-gpu-beta local-smoke` is CI-safe on CPU-only hosts and reports `public_swarm_gpu_beta_smoke_ready` without claiming a usable GPU route. `public-swarm-gpu-beta local-loopback` selects `hf_transformers_cuda`, requires `cuda_runtime_available`, `hf_transformers_cuda_ready`, and `gpu_runtime_ready`, and routes stage work only to Miners advertising `real_llm_sharded_cuda_stage0`, `real_llm_sharded_cuda_stage1`, or `real_llm_sharded_cuda_both`; ready reports include `public_swarm_gpu_beta_ready`, `gpu_stage0_ready`, and `gpu_stage1_ready`. `public-swarm-gpu-beta kaggle-package` prepares private Kaggle GPU stage templates with `kaggle_gpu_package_ready`. The side-effectful `public-swarm-gpu-beta kaggle-auto` starts a temporary CPU-capable public Coordinator, defers CUDA runtime checks to private Kaggle GPU stage Miners, pushes private Kaggle GPU kernels, verifies `external_gpu_runtime_verified`, deletes kernels, and only then may report `public_swarm_gpu_beta_kaggle_auto_ready`, `kaggle_kernels_deleted`, and `token_rotation_required`; the check mode is a CI fake runner. Generated Kaggle CUDA kernels default to `torch==2.7.1+cu118`, `torchvision==0.22.1+cu118`, and `transformers==4.40.2`, and the retained successful proof is `dist/public-swarm-gpu-beta-live-20260528-runtimepin/public_swarm_gpu_inference_beta_kaggle_auto.json`. `public-swarm-gpu-beta evidence-import` imports a completed GPU report with `external_gpu_runtime_verified`. This is read-only optional CUDA tiny GPT evidence, not production Swarm Inference, not P2P, not a GPU pooling marketplace, and not large-model serving.

`crowdtensor gpu-generate` is the multi-token generation wrapper over that CUDA split route. It emits `gpu_sharded_generation_beta_v1` through `scripts/gpu_sharded_generation_beta_pack.py` and is checked by `scripts/gpu_sharded_generation_beta_check.py`. `gpu-generate evidence-import --gpu-report dist/gpu-sharded-generation-beta-kaggle-20260528095658/gpu_sharded_generation_beta_kaggle_auto.json --max-new-tokens 16 --json` imports the retained successful Kaggle GPU proof; the report includes `generated_token_count: 16`, a safe `generated_text_hash`, `multi_token_generation_ready`, `gpu_multi_machine_generation_ready`, `external_gpu_runtime_verified`, stage-local partition readiness, distinct stage Miners, and `kaggle_kernels_deleted`. Use `python scripts/gpu_sharded_generation_beta_check.py --include-wrapper-check --max-new-tokens 4 --json` for CI-safe validation without creating Kaggle resources. The side-effectful `gpu-generate kaggle-auto` path is documented in `docs/gpu-sharded-generation-beta.md`; rotate temporary public HTTP tokens after every run and do not retain private env or registry files in public artifacts.

The Public Swarm Product RC is the current product-surface milestone:

```bash
crowdtensor serve --profile gpu-generation --json
crowdtensor join --coordinator-url http://127.0.0.1:8787 --stage stage0 --backend cuda --json
crowdtensor join --coordinator-url http://127.0.0.1:8787 --stage stage1 --backend cuda --json
crowdtensor generate --coordinator-url http://127.0.0.1:8787 --prompt-text "CrowdTensor product RC" --backend cuda --dry-run --json
crowdtensor peer check --json
crowdtensor public-swarm-product-rc --json
python scripts/public_swarm_product_rc_check.py --json
```

`crowdtensor serve` prints or runs the Coordinator command for `cpu-real-llm` or `gpu-generation`; public bind requires explicit acknowledgement. `crowdtensor join` prints or runs a Miner command for stage0/stage1/both and can resolve a Coordinator through `--peer-bootstrap`. `crowdtensor generate` creates a bounded `session_protocol_v1` request, hashes the prompt in public output, and uses `POST /admin/inference-sessions` when not in `--dry-run`. `crowdtensor peer daemon`, `peer announce`, `peer resolve`, and `peer check` expose `p2p_lite_peer_v1` HTTP-gossip discovery. The RC artifact imports the retained `gpu_sharded_generation_beta_v1` Kaggle evidence and requires `coordinator_product_surface_ready`, `session_protocol_ready`, `p2p_lite_discovery_ready`, and `gpu_generation_evidence_import_ready`. P2P-lite does not replace Coordinator leases, heartbeats, validation, or result ledgers; it is not libp2p, DHT, NAT traversal, decentralized security, Hivemind/Petals-level serving, or large-model serving.

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

Then pass the printed `sha256:` value to `--miner-token`, `--observer-token`, `--admin-token`, or a per-Miner registry entry. Clients still send the original token.

## Run Miner

```bash
CROWDTENSOR_MINER_TOKEN=local-miner crowdtensor-miner \
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
