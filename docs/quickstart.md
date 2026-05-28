# Quickstart

This guide runs a local Coordinator and a Python Miner. It validates the current CrowdTensorD Alpha control-plane loop; it does not train a real LLM.

## Python Environment

Use Python 3.11 or newer.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .[dev]
```

Avoid installing into the system Python. A virtualenv keeps the checkout compatible with distributions that enforce PEP 668 externally managed Python environments. If your environment has no network access, preinstall `setuptools` and `wheel` in the virtualenv or use a base image that already includes them. The package install uses standard Python build metadata.

The install creates two console commands:

```bash
crowdtensor --help
crowdtensord --help
crowdtensor-miner --help
```

To verify the documented fresh-clone path from a clean virtualenv, run:

```bash
python scripts/onboarding_gate.py --quick --json-out /tmp/crowdtensor_onboarding_gate.json
```

The `onboarding_gate_v1` report creates a temporary venv, runs `python -m pip install -e .[dev]`, validates the three console commands above, then runs `crowdtensor local-proof`, `crowdtensor home-infer`, `crowdtensor llm-infer --mock`, `crowdtensor cpu-infer --mode local`, and `crowdtensor release-ready --allow-dirty` with reduced request counts. It is an Alpha onboarding gate, not production Swarm Inference, arbitrary prompt serving, GPU pooling, P2P routing, or WebGPU execution.

Run the one-command local proof first when you want the shortest open-source path from checkout to safe artifact:

```bash
crowdtensor local-proof --json
```

The `crowdtensor/cli.py` entrypoint emits `local_proof_summary_v1` and writes artifacts under `dist/local-proof` by chaining Doctor, `runtime_matrix.py`, the CPU-only read-only home-compute demo, and the Demo Manifest. It is not production Swarm Inference, arbitrary prompt serving, GPU pooling, P2P routing, or WebGPU execution.

## 15-Minute CPU Inference Beta Path

Run the aggregate CPU-only inference Beta proof after the local proof:

```bash
crowdtensor cpu-infer --mode local --json
```

For the larger CPU Inference Beta RC evidence path:

```bash
crowdtensor cpu-infer --mode beta-rc --json
```

The RC path emits `cpu_inference_beta_rc_v1` through `scripts/cpu_inference_beta_rc_pack.py`. It combines the local CPU proof, remote-loopback proof, Real two-machine CPU inference Beta rehearsal, Kaggle Remote Miner Beta artifact preparation, `miner_join_pack_v1`, `scripts/cpu_inference_beta_rc_check.py`, and `demo_manifest_v1`. A ready report includes `cpu_inference_beta_rc_ready`, `local_cpu_inference_ready`, `remote_loopback_ready`, `two_machine_rehearsal_ready`, `kaggle_remote_miner_artifacts_ready`, `miner_join_pack_ready`, and `cpu_miner_beta_ready`. If you have a completed live Kaggle report, add `--kaggle-real-runtime-report dist/kaggle-real-runtime/kaggle_real_runtime_acceptance.json` to import `kaggle_real_runtime_acceptance_v1` and surface `real_runtime_evidence_ready`. This is CPU-only and read-only; it is not production Swarm Inference, not P2P, not a GPU/TPU workload path, and not arbitrary prompt serving.

This emits `cpu_inference_beta_v1` through `scripts/cpu_inference_beta_pack.py`. The local mode wraps `home-infer` and deterministic `llm-infer --mock`, so a new user gets one safe report for the read-only `model_bundle_infer` route and fixed-prompt `external_llm_infer` mock runtime. CI validates it with `scripts/cpu_inference_beta_check.py`. Maintainers can also run `--mode remote-loopback` for the local remote-demo stand-in, or `--mode remote-existing` against an already running two-machine `remote-demo` with explicit observer/admin tokens.

This path is CPU-only and read-only. It is not production Swarm Inference, not P2P, not GPU pooling, and not arbitrary prompt serving.

To try the first model-execution split proof, run:

```bash
crowdtensor shard-infer --json
```

The Pipeline-Sharded Inference Alpha emits `sharded_inference_cli_v1` and `sharded_inference_evidence_v1`. It uses `sharded_model_bundle_infer` / `sharded_model_bundle_infer_v1` inside a `sharded_inference_session_v1`: stage 0 Miner produces activation hashes, stage 1 Miner consumes the accepted activation and must report `baseline_match` against the single-task model bundle inference baseline. The acceptance check is `python scripts/sharded_inference_check.py --base-port 9080`; use `--failure-mode kill-stage-after-claim` to also require `stage_requeue_ready`. This is CPU-only, read-only, not production Swarm Inference, not P2P, and not real LLM or GPU sharding.

For the CPU Pipeline-Sharded Inference Beta loopback:

```bash
crowdtensor shard-infer-beta --mode remote-loopback --json
```

This emits `remote_sharded_inference_beta_v1` through `scripts/remote_sharded_inference_beta_pack.py` and validates with `scripts/remote_sharded_inference_beta_check.py`. A ready report includes `remote_sharded_inference_ready`, `remote_sharded_loopback_ready`, `local_sharded_inference_ready`, activation hashes, `baseline_match`, and `stage_requeue_ready` when `--failure-mode kill-stage-after-claim` is used. It is CPU-only, read-only, not production Swarm Inference, not P2P, and not real LLM sharding.

For the controlled two-machine helper, use `crowdtensor remote-demo --workload sharded-model-bundle`. The loopback check `python scripts/remote_home_compute_demo_check.py --workload sharded-model-bundle` should report `remote_python_sharded_model_bundle_infer`, `remote_sharded_inference_acceptance_v1`, `remote_sharded_inference_observability_v1`, and `remote_two_machine_sharded_ready`.

For the CPU-only Micro-LLM Pipeline-Sharded Inference Alpha:

```bash
crowdtensor micro-llm-shard-infer --decode-steps 3 --json
```

This emits `micro_llm_sharded_cli_v1` and `micro_llm_sharded_evidence_v1`. It uses `micro_llm_sharded_infer` / `micro_llm_sharded_infer_v1` inside a `micro_llm_sharded_session_v1`: stage 0 produces hidden-state activation hashes, stage 1 consumes the accepted activations, runs deterministic tiny Transformer decode for `decode_steps`, and must report `baseline_match` plus `decoded_tokens_match`. The acceptance check is `python scripts/micro_llm_sharded_inference_check.py --base-port 9084 --request-count 2 --decode-steps 3`; use `--failure-mode kill-stage-after-claim` to also require `stage_requeue_ready`. This is CPU-only, read-only, not production Swarm Inference, not P2P, and not GGUF/llama.cpp or large LLM serving.

To prove explicit stage routing, run:

```bash
crowdtensor micro-llm-shard-infer --stage-mode split --require-distinct-stage-miners --decode-steps 3 --json
python scripts/stage_aware_micro_llm_sharded_check.py --base-port 9085 --request-count 2 --decode-steps 3 --require-distinct-stage-miners
```

This requires Miners advertising `micro_llm_sharded_stage0` and `micro_llm_sharded_stage1` capabilities and emits `distinct_stage_miners` plus `stage_assignment_valid` when stage 0 and stage 1 are accepted by distinct stage-capable Miners.

For the Remote Micro-LLM Pipeline-Sharded Inference Beta loopback:

```bash
crowdtensor micro-llm-shard-infer-beta --mode remote-loopback --decode-steps 3 --json
```

This emits `remote_micro_llm_sharded_beta_v1` through `scripts/remote_micro_llm_sharded_beta_pack.py` and validates with `scripts/remote_micro_llm_sharded_beta_check.py`. A ready report includes `remote_micro_llm_sharded_ready`, `remote_micro_llm_sharded_loopback_ready`, `local_micro_llm_sharded_inference_ready`, activation hashes, `baseline_match`, `decoded_tokens_match`, and `stage_requeue_ready` when `--failure-mode kill-stage-after-claim` is used. The controlled two-machine helper also accepts `crowdtensor remote-demo --workload micro-llm-sharded`; its loopback check reports `remote_python_micro_llm_sharded_infer`, `remote_micro_llm_sharded_acceptance_v1`, `remote_micro_llm_sharded_observability_v1`, and `remote_two_machine_micro_llm_sharded_ready`. It is CPU-only, read-only, not production Swarm Inference, not P2P, and not GGUF/llama.cpp serving.

For the stage-aware remote loopback proof:

```bash
crowdtensor micro-llm-shard-infer-beta --mode remote-loopback --stage-mode split --require-distinct-stage-miners --decode-steps 3 --json
python scripts/remote_micro_llm_sharded_beta_check.py --mode remote-loopback --stage-mode split --require-distinct-stage-miners --request-count 2 --decode-steps 3
```

For real controlled hosts, prepare one `crowdtensor remote-demo prepare --workload micro-llm-sharded --stage-role stage0 ...` join pack and one `--stage-role stage1` join pack, then verify with `crowdtensor remote-demo verify --workload micro-llm-sharded --stage-mode split --require-distinct-stage-miners ...`.

For the Micro-LLM Live Two-Node RC:

```bash
crowdtensor micro-llm-live-rc --mode local-generated --port 9182 --request-count 2 --decode-steps 3 --json
python scripts/micro_llm_live_rc_check.py --base-port 9182 --request-count 2 --decode-steps 3
```

This emits `micro_llm_live_rc_v1` through `scripts/micro_llm_live_rc_pack.py`, generates `kaggle-upload-stage0` and `kaggle-upload-stage1`, and should report `local_generated_stage_upload_standins_ready`, `micro_llm_live_rc_ready`, `kaggle_micro_llm_sharded_ready`, and `stage_assignment_valid`. `--mode external-existing` verifies an already running public Coordinator plus two external stage Miners and only then reports `external_runtime_verified`. It is CPU-only, read-only toy two-stage micro-LLM, not production Swarm Inference, not P2P, and not GGUF/llama.cpp serving.

For the Real Small-LLM Sharded Inference Live RC, install the optional HF runtime and run:

```bash
python -m pip install -e '.[hf]'
crowdtensor real-llm-live-rc --mode local-generated --port 9184 --request-count 1 --json
python scripts/real_llm_live_rc_check.py --base-port 9184 --request-count 1
```

This emits `real_llm_live_rc_v1` through `scripts/real_llm_live_rc_pack.py`, generates `kaggle-upload-real-llm-stage0` and `kaggle-upload-real-llm-stage1`, and should report `local_generated_real_llm_stage_upload_standins_ready`, `remote_real_llm_sharded_ready`, `real_llm_artifact_ready`, and `stage_assignment_valid`. Generated Miners use `--enable-hf-tiny-gpt-runtime` and `--real-llm-stage-role`. `--mode kaggle-generated` prepares packages only; `--mode external-existing` verifies an already running public Coordinator plus two external stage Miners and only then reports `external_runtime_verified` and `kaggle_real_llm_sharded_ready`. For Kaggle CLI-driven runs, `scripts/kaggle_real_llm_live_package.py` emits `kaggle_real_llm_live_package_v1` private dataset/script-kernel packages and can use `--inline-kernel-payload` as a temporary fallback. The first live real-weight Kaggle proof is retained at `dist/real-llm-live-goal-external/real_llm_live_rc.json` and includes `kaggle_real_llm_stage0_seen`, `kaggle_real_llm_stage1_seen`, `distinct_stage_miners`, and valid `stage_assignment_valid`. It is CPU-only, read-only, not production Swarm Inference, not P2P, and not large-model serving.

For the larger Real Internet Swarm Inference Alpha milestone:

```bash
crowdtensor real-llm-internet-alpha --mode local-generated --port 9187 --base-port 9188 --request-count 1 --json
python scripts/real_llm_internet_alpha_check.py --port 9187 --base-port 9188 --request-count 1
```

This emits `real_llm_internet_alpha_v1` through `scripts/real_llm_internet_alpha_pack.py`. `local-generated` chains the Live RC plus local stage0/stage1 failure rescue checks and must report `real_llm_internet_alpha_ready`, `real_llm_stage_requeue_ready`, `stage_requeue_ready`, `real_llm_live_rc_ready`, `remote_real_llm_sharded_ready`, `real_llm_artifact_ready`, `activation_transport_ready`, `decoded_tokens_match`, `distinct_stage_miners`, and `stage_assignment_valid` while keeping `external_runtime_verified` false. `package` prepares public Coordinator and stage upload artifacts only. `external-existing` verifies an already running public Coordinator plus two external stage Miners and only then may report `external_runtime_verified`. Reports include `token_rotation_required` for temporary public HTTP runs and remain CPU-only, read-only, not production Swarm Inference, not P2P, and not large-model serving.

The retained first external Alpha proof is `dist/real-llm-internet-alpha-external/real_llm_internet_alpha.json`. It completed against public Coordinator `24.199.118.54:9187` with two private Kaggle CPU script kernels, `internet-real-llm-stage0` and `internet-real-llm-stage1`, and reports `external_runtime_verified`, `real_llm_internet_alpha_ready`, both Kaggle stages seen, `decoded_tokens_match`, distinct stage Miners, and valid stage assignment. Temporary Kaggle kernels were deleted after the run and tokens must be rotated after temporary public HTTP proofs.

For the larger automated external Beta path:

```bash
crowdtensor real-llm-internet-beta --mode kaggle-auto --public-host 24.199.118.54 --port 9190 --base-port 9191 --request-count 2 --json
python scripts/real_llm_internet_beta_check.py --port 9190 --base-port 9191 --request-count 2
```

This emits `real_llm_internet_beta_v1` through `scripts/real_llm_internet_beta_pack.py`. `kaggle-auto` generates the Alpha package, launches the temporary public Coordinator, pushes two private Kaggle CPU script kernels by default or two private Kaggle GPU kernels when `--real-llm-backend hf_transformers_cuda` is selected, verifies `external-existing`, deletes the temporary kernels, stops the Coordinator, and only then may report `real_llm_internet_beta_ready`. A ready report includes `real_llm_internet_alpha_ready`, `external_runtime_verified`, both Kaggle stages seen, `decoded_tokens_match`, `distinct_stage_miners`, `stage_assignment_valid`, `kaggle_kernels_deleted`, and `token_rotation_required`. With CUDA selected, the Coordinator may run on CPU using metadata-only CUDA artifact inspection; the Kaggle Miners must have torch CUDA. `scripts/real_llm_internet_beta_check.py` is the CI-safe fake-runner contract check; it does not create Kaggle resources. The default path remains CPU-only; the CUDA path is optional read-only tiny GPT evidence, not production Swarm Inference, not P2P, not GPU pooling, and not large-model serving.

For the user-facing Swarm Inference Beta around the same tiny GPT split path:

```bash
python -m pip install -e '.[hf]'
crowdtensor swarm-infer-beta live \
  --public-host 24.199.118.54 \
  --port 9210 \
  --base-port 9211 \
  --request-count 2 \
  --output-dir dist/swarm-inference-beta-live \
  --json

crowdtensor swarm-infer-beta prepare --coordinator-url https://YOUR_COORDINATOR_HOST --json
crowdtensor swarm-infer-beta coordinator --json
crowdtensor swarm-infer-beta miner --stage stage0 --json
crowdtensor swarm-infer-beta miner --stage stage1 --json
crowdtensor swarm-infer-beta verify --coordinator-url https://YOUR_COORDINATOR_HOST --json
crowdtensor swarm-infer-beta collect --coordinator-url https://YOUR_COORDINATOR_HOST --json
python scripts/swarm_inference_beta_check.py --json
```

This emits `swarm_inference_beta_v1` through `scripts/swarm_inference_beta_pack.py`. `swarm-infer-beta live` is the side-effectful `kaggle-auto` path: it wraps `real_llm_internet_beta_v1`, starts a temporary public Coordinator, pushes private Kaggle CPU stage kernels, verifies `external_runtime_verified`, optionally verifies external victim/rescue requeue with `--failure-mode`, deletes the kernels, writes `support_bundle.json`, removes local live private artifacts and raw runtime state by default, and only then may report `swarm_inference_beta_live_ready`, `swarm_inference_beta_ready`, `two_machine_swarm_inference_ready`, `real_llm_internet_beta_ready`, `external_stage_requeue_ready` when requested, `live_requeue_summary`, `kaggle_kernels_deleted`, `swarm_inference_beta_live_private_artifacts_cleaned`, and `token_rotation_required`. Use `--keep-live-private-artifacts` only for debugging failed live runs, then delete those files before sharing artifacts. `swarm-infer-beta prepare` creates `operator.private.env`, stage0/stage1 `miner.private.env` files, a hashed `miner_registry.json`, stage-specific join packs, and `SWARM_INFERENCE_BETA.md`. `swarm-infer-beta verify` wraps `remote_real_llm_sharded_beta_v1`, requires `real_llm_split_route_ready`, `decoded_tokens_match`, `distinct_stage_miners`, and `stage_assignment_valid`, then reports `swarm_inference_beta_ready` plus `two_machine_swarm_inference_ready`. Passing `--real-internet-beta-report dist/real-llm-internet-beta-kaggle-auto-final/real_llm_internet_beta.json` imports existing `real_llm_internet_beta_v1` evidence as `external_beta_evidence_imported`; it does not claim a fresh live run. `swarm-infer-beta collect` gathers redacted evidence/support, and `swarm-infer-beta clean` is dry-run by default. This is CPU-only, read-only, not production Swarm Inference, not P2P, and not large-model serving.

For the product-shaped Public Swarm Inference Alpha session wrapper:

```bash
python -m pip install -e '.[hf]'
crowdtensor swarm-session \
  --mode live-kaggle \
  --public-host 24.199.118.54 \
  --port 9220 \
  --base-port 9221 \
  --request-count 2 \
  --failure-mode kill-stage0-after-claim \
  --output-dir dist/public-swarm-inference-alpha \
  --json

python scripts/public_swarm_inference_alpha_check.py --json
```

This emits `public_swarm_inference_alpha_v1` through `scripts/public_swarm_inference_alpha_pack.py`. `crowdtensor swarm-session --mode live-kaggle --failure-mode kill-stage0-after-claim` aggregates the cleanup-backed `swarm-infer-beta live` external Kaggle proof, true external victim/rescue requeue evidence (`external_stage_requeue_ready`, `live_stage0_requeue_ready` or `live_stage1_requeue_ready`, `live_requeue_summary`), and the mandatory `local-generated` real LLM stage requeue proof. It then reports a user-facing session summary with `public_swarm_inference_alpha_ready`, `public_swarm_session_ready`, `local_stage_requeue_ready`, `public_swarm_live_requeue_ready`, `public_swarm_live_kaggle_ready`, `stage_requeue_ready`, `external_runtime_verified`, `decoded_tokens_match`, `distinct_stage_miners`, `stage_assignment_valid`, `kaggle_kernels_deleted`, and `token_rotation_required`. `--mode local-generated` runs only the localhost stage requeue proof. The wrapper prunes child debug artifacts by default and retains the top-level public JSON/Markdown evidence; use `--keep-child-artifacts` only for local debugging. It is CPU-only, read-only, not production Swarm Inference, not P2P, and not large-model serving.

For a release-candidate audit of retained live evidence:

```bash
crowdtensor public-swarm-alpha-rc --mode evidence-import --json
python scripts/public_swarm_inference_alpha_rc_check.py --mode local-smoke
```

This emits `public_swarm_inference_alpha_rc_v1` through `scripts/public_swarm_inference_alpha_rc_pack.py`. `evidence-import` requires `stage0_live_requeue_evidence_ready`, `stage1_live_requeue_evidence_ready`, `public_swarm_live_requeue_evidence_ready`, `public_swarm_alpha_rc_evidence_imported`, `public_swarm_alpha_private_artifacts_absent`, and `public_swarm_inference_alpha_rc_ready` from the retained public reports and `dist/public-swarm-inference-alpha-live-requeue-summary.json`. `local-smoke` is CI-safe and does not create Kaggle resources. It is CPU-only, read-only, not production Swarm Inference, not P2P, and not large-model serving.

For the Public Swarm Inference Beta user entrypoint:

```bash
crowdtensor public-swarm-beta product-beta --json
python scripts/public_swarm_inference_beta_check.py --mode product-beta --json

# Compatibility paths:
crowdtensor public-swarm-beta local-loopback --base-port 9290 --request-count 1 --json
crowdtensor public-swarm-beta evidence-import --json
python scripts/public_swarm_inference_beta_check.py --mode local-loopback --base-port 9290 --request-count 1
```

This emits `public_swarm_inference_beta_v1` through `scripts/public_swarm_inference_beta_pack.py` and is checked by `scripts/public_swarm_inference_beta_check.py`. `product-beta` aggregates the Product RC, `session_protocol_v1`, `p2p_lite_peer_v1`, retained GPU sharded generation evidence, and CPU fallback, and should report `public_swarm_product_beta_ready`, `public_swarm_product_rc_ready`, `coordinator_product_surface_ready`, `session_protocol_ready`, `p2p_lite_discovery_ready`, `gpu_generation_evidence_import_ready`, and `cpu_fallback_ready`. `crowdtensor public-swarm-beta local-loopback` still runs the legacy two-stage split path and should report `two_stage_split_inference_ready`, `local_loopback_ready`, `decoded_tokens_match`, `distinct_stage_miners`, and `stage_assignment_valid`. `public-swarm-beta evidence-import` still imports retained live evidence and should report `public_swarm_beta_evidence_import_ready`, `external_live_evidence_imported`, `stage0_live_requeue_evidence_ready`, and `stage1_live_requeue_evidence_ready`. For a controlled two-machine run, use `prepare`, `coordinator`, `miner --stage stage0`, `miner --stage stage1`, `verify`, `collect`, and dry-run `clean`. It is Coordinator-backed, read-only, not production Swarm Inference, not libp2p/DHT/NAT traversal, not Hivemind-level serving, and not large-model serving.

For the optional Public Swarm GPU Inference Beta overlay:

```bash
python -m pip install -e '.[hf]'
crowdtensor public-swarm-gpu-beta local-smoke --json
crowdtensor public-swarm-gpu-beta local-loopback --base-port 9300 --request-count 1 --json
crowdtensor public-swarm-gpu-beta kaggle-package --output-dir dist/public-swarm-gpu-beta-kaggle --json
crowdtensor public-swarm-gpu-beta evidence-import --gpu-report dist/public-swarm-gpu-beta/public_swarm_gpu_inference_beta.json --json
python scripts/public_swarm_gpu_inference_beta_check.py --mode local-smoke
```

This emits `public_swarm_gpu_inference_beta_v1` through `scripts/public_swarm_gpu_inference_beta_pack.py` and is checked by `scripts/public_swarm_gpu_inference_beta_check.py`. `public-swarm-gpu-beta local-smoke` is safe on CPU-only machines and should report `public_swarm_gpu_beta_smoke_ready` without claiming `public_swarm_gpu_beta_ready`. `public-swarm-gpu-beta local-loopback` requires the explicit `hf_transformers_cuda` backend, CUDA runtime evidence such as `cuda_runtime_available`, `hf_transformers_cuda_ready`, and `gpu_runtime_ready`, plus stage capabilities `real_llm_sharded_cuda_stage0`, `real_llm_sharded_cuda_stage1`, or `real_llm_sharded_cuda_both`; ready reports include `gpu_stage0_ready`, `gpu_stage1_ready`, `stage_local_partition_ready`, `stage0_partition_loaded`, `stage1_partition_loaded`, and `partition_parameter_split_valid`. `public-swarm-gpu-beta kaggle-package` prepares private Kaggle GPU stage templates with `kaggle_gpu_package_ready`, and `public-swarm-gpu-beta evidence-import` promotes a completed GPU report with `external_gpu_runtime_verified`. The side-effectful `kaggle-auto` path has a retained stage-local proof at `dist/public-swarm-gpu-beta-stage-local-live-20260528064520-shortslug/public_swarm_gpu_inference_beta_kaggle_auto.json`; the older `dist/public-swarm-gpu-beta-live-20260528-runtimepin/public_swarm_gpu_inference_beta_kaggle_auto.json` path is historical pre-stage-local evidence. Kaggle CUDA kernels default to `torch==2.7.1+cu118`, `torchvision==0.22.1+cu118`, and `transformers==4.40.2`. The GPU Beta overlay defaults to `--real-llm-partition-mode stage-local`, which proves stage-owned module placement and parameter-count evidence for the tiny GPT path; it is not production memory scheduling, not P2P, not a GPU pooling marketplace, and not large-model serving.

For the multi-token GPU generation milestone, import the retained proof or run the CI-safe check:

```bash
crowdtensor gpu-generate evidence-import \
  --gpu-report dist/gpu-sharded-generation-beta-kaggle-20260528095658/gpu_sharded_generation_beta_kaggle_auto.json \
  --max-new-tokens 16 \
  --json
python scripts/gpu_sharded_generation_beta_check.py --include-wrapper-check --max-new-tokens 4 --json
```

The side-effectful `gpu-generate kaggle-auto` path is documented in `docs/gpu-sharded-generation-beta.md`. The retained Kaggle GPU proof reports 16 generated tokens, a safe `generated_text_hash`, `multi_token_generation_ready`, two distinct stage Miners, stage-local partition readiness, and `kaggle_kernels_deleted`; private env and registry files are not retained in the public evidence tree. This is still a tiny GPT CUDA Beta proof, not production Swarm Inference, not Hivemind-level serving, not P2P, not a GPU marketplace, and not large-model serving.

For the recommended two-machine operator wrapper around the same real-weight split path, use `crowdtensor remote-demo --workload real-llm-sharded` after installing the optional HF runtime on the Coordinator and both stage Miner hosts:

```bash
python -m pip install -e '.[hf]'

crowdtensor remote-demo prepare \
  --workload real-llm-sharded \
  --stage-role stage0 \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id real-stage0 \
  --output-dir dist/remote-real-llm-stage0 \
  --json

crowdtensor remote-demo prepare \
  --workload real-llm-sharded \
  --stage-role stage1 \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id real-stage1 \
  --output-dir dist/remote-real-llm-stage1 \
  --json
```

Run the generated stage launchers on distinct hosts, then verify with `crowdtensor remote-demo verify --workload real-llm-sharded --stage-mode split --require-distinct-stage-miners ...`. The wrapper emits `remote_real_llm_sharded_runbook_v1`, `remote_real_llm_sharded_acceptance_v1`, `remote_real_llm_sharded_observability_v1`, and `remote_real_llm_sharded_beta_v1` for `remote_python_real_llm_sharded_infer`; a ready report includes `remote_two_machine_real_llm_sharded_ready`. Missing optional HF dependencies surface `hf_dependencies_missing` with the install action instead of a vague runtime failure. This remains CPU-only, read-only tiny GPT evidence, not production Swarm Inference, not P2P, and not large-model serving.

Run the local read-only inference proof when you want the shortest shareable result trace:

```bash
crowdtensor home-infer --scenario-id route-baseline --json
```

The `crowdtensor/cli.py` wrapper emits `home_inference_cli_v1`, writes `home_compute_evidence_v1` JSON/Markdown under `dist/home-infer`, and summarizes the CPU-only `model_bundle_infer` route, fixed `model_bundle_inference_scenario_v1` scenario, capped `request_trace`, `diagnosis_codes`, and read-only/redaction status. Built-in scenarios include `route-baseline`, `gradient-safety`, and `mixed-prompts`. It is not production Swarm Inference and does not accept arbitrary prompts.

For a safe local LLM runtime proof, run:

```bash
crowdtensor llm-infer --mock --json
```

This emits `llm_inference_cli_v1` and writes `external_llm_evidence_v1` JSON/Markdown under `dist/llm-infer`. The default mock path is deterministic. Operators can pass `--llm-runtime-cmd /path/to/wrapper` or `--llm-runtime-url http://127.0.0.1:11434/v1/chat/completions` when they own the runtime. Reports keep raw prompts, `output_text`, runtime URL, and API key out of public artifacts.

When repeated demos or tests create temporary files, inspect cleanup candidates first:

```bash
crowdtensor clean-artifacts --json
```

Then apply the conservative cleanup:

```bash
crowdtensor clean-artifacts --apply --json
```

The `cleanup_report_v1` report covers generated `__pycache__` / `.pyc` caches and old CrowdTensor temp directories. It defaults to dry-run, keeps reports unless `--include-reports` is used, and does not delete state, source files, release evidence, or private env material.

## First-run Doctor

Run the lightweight diagnostics before starting services:

```bash
python3 scripts/doctor.py --json
```

The First-run Doctor checks Python version, core imports, FastAPI/Uvicorn availability, state directory writability, default port binding, and console entrypoints. It is a quick environment check, not a replacement for runtime acceptance.

For remote-demo and browser dependency probes:

```bash
python3 scripts/doctor.py --remote-demo --browser --json
```

## Run Coordinator

```bash
crowdtensord \
  --host 127.0.0.1 \
  --port 8787 \
  --state-dir state
```

The Coordinator creates and maintains the local checkpoint/event state under `state/`.

## Run One Miner

In another shell:

```bash
crowdtensor-miner \
  --coordinator http://127.0.0.1:8787 \
  --miner-id local-1 \
  --once
```

Expected behavior:

- Miner claims one task.
- Miner sends heartbeats while computing.
- Miner submits a validated result.
- Coordinator updates the tiny model state.
- Miner exits with a JSON summary.

## Token-Protected Local Run

For local demos, plaintext tokens are simplest. For remote demos, generate hashed token config values:

```bash
python3 scripts/hash_token.py local-miner
```

The Coordinator accepts either plaintext values or `sha256:` verifiers. Miners still send the original token.

Start Coordinator:

```bash
crowdtensord \
  --host 127.0.0.1 \
  --port 8787 \
  --state-dir state \
  --miner-token local-miner \
  --observer-token local-observer \
  --admin-token local-admin
```

Run Miner:

```bash
CROWDTENSOR_MINER_TOKEN=local-miner crowdtensor-miner \
  --coordinator http://127.0.0.1:8787 \
  --miner-id local-secure-1 \
  --once
```

Read metrics:

```bash
curl -H 'x-crowdtensor-observer-token: local-observer' \
  http://127.0.0.1:8787/metrics
```

## Runtime Acceptance Pack

Run the release gate to check Alpha packaging and documentation integrity:

```bash
python3 scripts/release_gate.py --json
```

This is a static open-source release check. It does not replace runtime acceptance.

Check local runtime capability readiness:

```bash
python3 scripts/runtime_matrix.py --json
```

The runtime capability matrix is the fastest way to see which CPU-only workloads are ready, whether optional browser checks can run, whether an external LLM HTTP adapter is configured through `CROWDTENSOR_LLM_RUNTIME_URL`, and which hardware/runtime matrix routes are realistic today. It reports `hardware_profile` style host facts, `hardware_targets`, `recommended_routes`, `matched_capabilities`, `missing_capabilities`, target and route `diagnosis_codes`, `operator_action`, top-level `diagnosis_summary`, and `hardware_diagnosis_summary` without printing runtime URL, token, or API key values. The `nvidia_cuda` target is no longer a pure future placeholder: it can recommend `crowdtensor public-swarm-gpu-beta local-loopback` only when CUDA plus optional HF runtime are ready; otherwise it reports `nvidia_cuda_optional_missing` or `nvidia_cuda_detected_adapter_unavailable`.

Run the matrix-guided home-compute demo:

```bash
python3 scripts/home_compute_demo.py --port 8909 --request-count 4 --json
```

This runs `scripts/runtime_matrix.py`, selects the CPU-only `model_bundle_infer` workload and `local_cpu_model_bundle_infer` capability route when available, then runs the local inference session demo. The report includes `route_decision`, safe metrics, a capped Coordinator-derived `request_trace`, and stable `diagnosis_codes` such as `home_compute_ready`, `runtime_matrix_blocked`, `workload_unavailable`, `cpu_route_unavailable`, `session_failed`, and `trace_missing`, making it the shortest open-source path from local capability discovery to a measurable Swarm Inference-shaped result without requiring GPU access.

Build a safe, shareable home-compute evidence pack:

```bash
python3 scripts/home_compute_evidence_pack.py \
  --port 8911 \
  --request-count 4 \
  --json-out /tmp/crowdtensor_home_evidence.json \
  --markdown-out /tmp/crowdtensor_home_evidence.md
```

The `home_compute_evidence_v1` artifact wraps the runtime matrix, `route_decision`, `matched_capabilities`, safe metrics, capped `request_trace` rows, `diagnosis_codes`, and runtime acceptance summary if `--runtime-report` is provided. It is intended for demos and issue reports, so it redacts token, URL, API key, lease, idempotency, weight, and delta-shaped fields.

Build the local-loopback Demo Manifest when you want one latest output artifact for a demo, handoff, or issue:

```bash
python3 scripts/demo_manifest_pack.py \
  --output-dir dist/demo-manifest \
  --port 8914 \
  --request-count 4
```

The `demo_manifest_v1` output writes `demo_manifest.json` / `demo_manifest.md` and indexes `runtime_matrix.json`, `remote_compute_evidence_v1`, `external_llm_evidence_v1`, `support_bundle`, and `remote_compute_observability_v1` summaries. It stays local-loopback and CPU-only by default; the external LLM section uses deterministic mock evidence and keeps raw prompts, `output_text`, runtime URL, and API key out of the manifest. Validate the full path with `scripts/demo_manifest_check.py`.

Run the default non-browser smoke suite:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8910 \
  --report /tmp/crowdtensor_acceptance.json
```

The default suite includes `scripts/runtime_matrix_check.py`, `scripts/home_compute_demo_check.py`, `scripts/home_compute_evidence_check.py`, the CPU-only `model_bundle_lm` contract smoke (`scripts/model_bundle_smoke.py`), read-only multi-request `model_bundle_infer` smoke (`scripts/model_bundle_inference_smoke.py`), user-facing inference session demo (`scripts/inference_session_demo.py`), admin-created read-only inference session API check (`scripts/admin_inference_session_check.py`), optional external LLM mock/command adapter smoke (`scripts/external_llm_inference_smoke.py`), OpenAI-compatible HTTP adapter smoke (`scripts/external_llm_http_adapter_smoke.py`), and safe external LLM evidence check (`scripts/external_llm_evidence_check.py`) alongside dense, adapter, micro LM, auth, audit, and operator checks. Use `--skip-runtime-matrix`, `--skip-home-compute-demo`, `--skip-home-compute-evidence`, `--skip-admin-inference-session`, `--skip-external-llm-inference`, `--skip-external-llm-http-adapter`, or `--skip-external-llm-evidence` if you need to omit those adapter checks.

Run only the local inference session demo:

```bash
python3 scripts/inference_session_demo.py --port 8904 --request-count 4
```

Add `--json` for a machine-readable report with `request_count`, `accuracy`, `elapsed_ms`, `requests_per_second`, `request_trace`, read-only status, redaction status, and Miner `hardware_profile`.

Request one session from an already running Coordinator:

```bash
python3 scripts/inference_session_client.py \
  --coordinator-url http://127.0.0.1:8787 \
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \
  --request-count 4 \
  --json
```

The `inference_session_client_v1` client calls `POST /admin/inference-sessions`, waits for the returned `task_id` through `GET /admin/results`, and reports safe `model_bundle_infer` validation and throughput with `session_client_ready` when complete. It is read-only, CPU-only, and does not accept arbitrary prompts. Runtime acceptance includes `scripts/inference_session_client_check.py`; use `--skip-inference-session-client` only when omitting this user-facing client check.

Run only the admin-created read-only inference session API check:

```bash
python3 scripts/admin_inference_session_check.py --port 8915 --request-count 4
```

This exercises `POST /admin/inference-sessions`, expects `schema=inference_session_request_v1`, enqueues a CPU `model_bundle_infer` task, and verifies the accepted result through `GET /admin/results?task_id=<task_id>&workload_type=model_bundle_infer`. It is a service-shaped control-plane boundary, not a public chat API or real LLM serving endpoint.

Run only the optional external LLM adapter contract smoke:

```bash
python3 scripts/external_llm_inference_smoke.py --port 8906 --request-count 3
```

Run the OpenAI-compatible HTTP adapter variant:

```bash
python3 scripts/external_llm_http_adapter_smoke.py --port 8907 --runtime-port 8908
```

These smokes exercise `external_llm_infer_v1`, validate `external_llm_results`, and check that the read-only ledger exposes `completion_count`, `output_chars`, and `adapter_kind` without leaking raw prompts or `output_text`. To use a local runtime wrapper instead of the mock, start a Miner with `--llm-runtime-cmd /path/to/wrapper` or `CROWDTENSOR_LLM_RUNTIME_CMD=/path/to/wrapper`; the wrapper receives `prompt` and `max_tokens` arguments. To use an OpenAI-compatible local server, start a Miner with `--llm-runtime-url http://127.0.0.1:11434/v1/chat/completions` or `CROWDTENSOR_LLM_RUNTIME_URL=...`, plus optional `--llm-runtime-api-key` / `CROWDTENSOR_LLM_RUNTIME_API_KEY`.

For a shareable evidence artifact:

```bash
python3 scripts/external_llm_evidence_check.py --port 8919
```

The check drives `scripts/external_llm_evidence_pack.py` through the deterministic mock runtime and verifies `external_llm_evidence_v1`, `external_llm_evidence_ready`, read-only status, and redaction.

Run the same suite with local auth enabled inside checks that support shared auth env vars:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8950 \
  --miner-token local-miner \
  --observer-token local-observer \
  --report /tmp/crowdtensor_auth_acceptance.json
```

## Remote Miner Demo

Use the Real two-machine CPU inference Beta first. It is the 15-minute two-machine CPU inference Beta path for a Coordinator host and a Miner host. It creates the registry, private env files, public commands, and `remote_home_compute_demo_v1` summary without making users manually stitch lower-level scripts together:

```bash
crowdtensor remote-demo prepare \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --output-dir dist/remote-home-compute \
  --json
```

After the generated Coordinator command is running on the Coordinator host and the generated Miner command is running on the Miner host:

```bash
. dist/remote-home-compute/operator.private.env
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

The wrapper uses `scripts/remote_home_compute_demo_pack.py`, validates through `scripts/remote_home_compute_demo_check.py`, creates the read-only `model_bundle_infer` session with `POST /admin/inference-sessions`, and summarizes `remote_python_model_bundle_infer`, `remote_compute_evidence_v1`, and `remote_demo_observability_v1`. `remote-demo doctor`, `remote-demo collect`, and `remote-demo clean` emit `remote_home_compute_doctor_v1`, `remote_home_compute_collect_v1`, and `remote_home_compute_cleanup_v1`; `remote-demo clean` defaults to dry-run and only deletes private env/registry files with `--include-private`. `scripts/remote_two_machine_beta_check.py` emits `remote_two_machine_beta_check_v1` in CI and requires `remote_two_machine_inference_ready`, `remote_two_machine_external_llm_ready`, and `remote_two_machine_beta_ready`. It keeps `operator.private.env` and `miner.private.env` private. This is task-level remote CPU inference, not model sharding, not production Swarm Inference, and not P2P routing. Real two-machine use requires operator-provided TLS, VPN, tunnel, or trusted network.

For a real external temporary Miner, use the Kaggle Remote Miner Beta target:

```bash
crowdtensor remote-demo prepare \
  --target kaggle \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id kaggle-cpu-1 \
  --output-dir dist/remote-home-compute-kaggle \
  --json
```

The generated `kaggle_remote_miner.py` is the Kaggle-side launcher. Upload only that file and `miner.private.env` to the Kaggle Notebook, install the checkout with `python -m pip install -e .`, then run `python kaggle_remote_miner.py`. Keep `operator.private.env` off Kaggle and run `remote-demo doctor`, `verify`, and `collect` from the operator host. `scripts/kaggle_remote_miner_beta_check.py` emits `kaggle_remote_miner_beta_check_v1` and validates `--target kaggle`, `kaggle_remote_miner_prepare_ready`, `kaggle_remote_miner_beta_ready`, token redaction, outbound Miner assumptions, and the same local loopback remote-demo protocol. Kaggle GPU/TPU visibility is only a runtime hint here; no GPU/TPU workload is enabled, and this is not production Swarm Inference or P2P.

For a real Kaggle Notebook runtime proof against the current public server, use Kaggle Real Runtime Acceptance:

```bash
crowdtensor remote-demo kaggle-real \
  --action prepare \
  --public-host 24.199.118.54 \
  --port 9180 \
  --miner-id kaggle-cpu-1 \
  --output-dir dist/kaggle-real-runtime \
  --json
```

Start `dist/kaggle-real-runtime/start_coordinator.sh`, upload only the generated `kaggle-upload/miner.private.env` and `kaggle-upload/kaggle_remote_miner.py` to Kaggle, run `python kaggle_remote_miner.py --env-file miner.private.env`, then run:

```bash
crowdtensor remote-demo kaggle-real \
  --action verify \
  --public-host 24.199.118.54 \
  --port 9180 \
  --output-dir dist/kaggle-real-runtime \
  --json
```

The report schema is `kaggle_real_runtime_acceptance_v1` from `scripts/kaggle_real_runtime_acceptance_pack.py`. CI uses `scripts/kaggle_real_runtime_acceptance_check.py` to validate artifact generation, temporary HTTP boundary text, `operator.private.env` exclusion, and redaction only; it does not claim a live Notebook connected. A live success includes `kaggle_artifacts_ready`, `coordinator_public_ready`, `kaggle_miner_seen`, `kaggle_result_accepted`, and `kaggle_real_runtime_ready`, plus `token_rotation_required`. This is not production public-internet security, not P2P, and not GPU/TPU workload execution.

To rehearse stage-aware micro-LLM split participation, add `--workload micro-llm-sharded --stage-mode split --decode-steps 3` to prepare and verify. Prepare writes two Kaggle upload directories, `kaggle-upload-stage0` and `kaggle-upload-stage1`, for two separate private Notebooks. A live split acceptance adds `kaggle_micro_llm_stage0_seen`, `kaggle_micro_llm_stage1_seen`, `kaggle_micro_llm_stage_assignment_valid`, `stage_assignment_valid`, and `kaggle_micro_llm_sharded_ready`.

For Kaggle CLI operation, `scripts/kaggle_micro_llm_live_package.py` can build private dataset/script-kernel upload folders from the prepared stage directories. Use default dataset-backed mode first. Use `--inline-kernel-payload` only as a temporary private fallback when Kaggle dataset inputs do not mount correctly; it embeds `miner.private.env` into the private kernel source, so delete the remote kernels/dataset and rotate tokens after the proof. A completed artifact-backed live proof is retained at `dist/kaggle-micro-llm-live/external-real/kaggle_real_runtime_acceptance.json` with `ok: true`, both Kaggle stages seen, valid stage assignment, artifact loaded, and matching decoded tokens. This is a toy two-stage pipeline proof, not large-model sharding, not GGUF/llama.cpp serving, and not production Swarm Inference.

The release-candidate wrapper is `crowdtensor micro-llm-live-rc`. Its `local-generated` mode is the CI-safe stand-in using the generated stage upload packages; `external-existing` is the mode for already running Kaggle Notebooks or real machines and must report `external_runtime_verified` before it is treated as external runtime proof.

To validate an operator-owned external LLM runtime through the same remote-demo shell, use the fixed-prompt `external_llm_infer` path:

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

The verify path emits `remote_external_llm_evidence_v1` and `remote_external_llm_observability_v1` for `remote_python_external_llm_infer`. It is deterministic with `--mock`; `--llm-runtime-cmd` or `--llm-runtime-url` can be used only when the operator owns that runtime. This is not public arbitrary prompt serving.

The lower-level safe two-machine runbook is still available:

```bash
crowdtensor remote-runbook \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --output-dir dist/remote-demo \
  --json
```

The `remote_runbook_cli_v1` summary is produced by `crowdtensor/cli.py` and wraps `scripts/remote_demo_runbook_pack.py`. After the Coordinator and remote Miner are running, validate the controlled demo:

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

The `remote_acceptance_cli_v1` summary applies token redaction to captured output, delegates to `scripts/remote_demo_acceptance_pack.py`, and keeps the path bounded to a read-only `model_bundle_infer` demo. This is not production Swarm Inference and not P2P routing.

For lower-level registry work, generate a registry-backed invite and run a Miner on another Linux host or container:

```bash
python3 scripts/create_miner_invite.py \
  --registry state/miner_registry.json \
  --miner-id remote-linux-1 \
  --coordinator-url https://YOUR_COORDINATOR_HOST
```

See [Remote Miner Onboarding](remote-miner.md) for the full controlled remote demo flow and `scripts/remote_miner_join_check.py`.

## Docker Compose

Run the local stack:

```bash
docker compose up --build coordinator miner
```

Check health:

```bash
curl http://127.0.0.1:8787/health
curl http://127.0.0.1:8787/version
curl http://127.0.0.1:8787/ready
```

Check metrics:

```bash
curl -H 'x-crowdtensor-observer-token: local-observer' \
  http://127.0.0.1:8787/metrics
```

The default Compose tokens are for local demos only. Copy `.env.example` to `.env` and change the values before sharing a machine.

## Browser Experiments

Serve the static web directory:

```bash
python3 -m http.server 8765 --directory web
```

Open the WebRTC tensor tunnel:

```text
http://127.0.0.1:8765/index.html?role=receiver&room=demo
http://127.0.0.1:8765/index.html?role=sender&room=demo
```

Run the core browser acceptance pack when Playwright and a browser are available:

```bash
python3 scripts/browser_acceptance_pack.py \
  --base-port 9310 \
  --report /tmp/crowdtensor_browser_acceptance.json
```

It runs `webrtc_smoke.py`, `runtime_contract_check.py`, and `browser_miner_smoke.py`. Use `--allow-skip` in CI-style environments where Playwright or Chromium may be unavailable.

Run the broader browser smoke set:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8930 \
  --include-browser \
  --report /tmp/crowdtensor_browser_acceptance.json
```
