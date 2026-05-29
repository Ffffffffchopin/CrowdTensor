# CrowdTensor

CrowdTensor is an open-source path toward fault-tolerant AI swarms built from ordinary home compute.

`CrowdTensorD` is the current Alpha daemon/control plane. It validates the V1 mechanics needed before real home GPU aggregation, Swarm Inference, browser compute, and future P2P routing are added: task leasing, heartbeat recovery, checkpoint replay, result validation, replay audit, Miner admission, CPU-first workload contracts, and a narrow optional CUDA tiny GPT split proof.

## What Works Today

- Run a local Coordinator and Miner loop on a normal CPU-only Linux machine.
- Connect controlled remote Python Miners with token-backed admission and retry behavior.
- Validate timeout recovery, stale result rejection, checkpoint replay, result ledger, and Support Bundle generation.
- Run deterministic tiny workloads shaped like future model contracts: `diloco_train`, `cpu_lora_mock`, `micro_transformer_lm`, `model_bundle_lm`, read-only `model_bundle_infer`, and optional read-only `external_llm_infer`.
- Run a CPU-only Pipeline-Sharded Inference Alpha with `sharded_model_bundle_infer`, where one Miner produces activation hashes and a second Miner produces the final read-only result.
- Run the CPU Pipeline-Sharded Inference Beta loopback proof, where `crowdtensor shard-infer-beta --mode remote-loopback` packages the same two-stage route as `remote_sharded_inference_beta_v1`.
- Run a CPU-only Micro-LLM Pipeline-Sharded Inference Alpha/Beta with `micro_llm_sharded_infer`, where stage 0 emits hidden-state activation hashes and stage 1 performs deterministic tiny Transformer decode validation, optionally loaded from a dependency-free `micro_llm_artifact_v1` JSON package.
- Run an optional Hugging Face tiny GPT split proof with `real_llm_sharded_infer`, where distinct stage Miners execute a real `sshleifer/tiny-gpt2` artifact through `hf_transformers_cpu` or explicit `hf_transformers_cuda`, pass hashed activation summaries, and validate the next-token result against a local full-model baseline. The optional `--real-llm-partition-mode stage-local` path moves only stage-owned modules to the selected runtime device and emits partition evidence such as `stage_local_partition_ready`, `stage0_partition_loaded`, `stage1_partition_loaded`, and `partition_parameter_split_valid`.
- Run the Public Swarm Inference Alpha wrapper with `crowdtensor swarm-session`, which emits `public_swarm_inference_alpha_v1` through `scripts/public_swarm_inference_alpha_pack.py` and validates with `scripts/public_swarm_inference_alpha_check.py`. `--mode live-kaggle --failure-mode kill-stage0-after-claim` aggregates the cleanup-backed `swarm-infer-beta live` Kaggle proof, true external victim/rescue lease requeue evidence (`external_stage_requeue_ready`, `live_stage0_requeue_ready` or `live_stage1_requeue_ready`, `live_requeue_summary`), and mandatory `local-generated` stage requeue evidence. A ready report should include `public_swarm_inference_alpha_ready`, `public_swarm_session_ready`, `local_stage_requeue_ready`, `public_swarm_live_requeue_ready`, `public_swarm_live_kaggle_ready`, `stage_requeue_ready`, `external_runtime_verified`, `kaggle_kernels_deleted`, and `token_rotation_required`. The wrapper prunes child debug artifacts by default and retains the top-level public JSON/Markdown evidence; use `--keep-child-artifacts` only for local debugging. It is CPU-only and read-only, not production Swarm Inference, not P2P, and not large-model serving.
- Build the Public Swarm Inference Alpha RC artifact with `crowdtensor public-swarm-alpha-rc`, which emits `public_swarm_inference_alpha_rc_v1` through `scripts/public_swarm_inference_alpha_rc_pack.py` and validates with `scripts/public_swarm_inference_alpha_rc_check.py`. The default `evidence-import` mode audits retained live reports for both `stage0_live_requeue_evidence_ready` and `stage1_live_requeue_evidence_ready`, plus `public_swarm_live_requeue_evidence_ready`, `public_swarm_alpha_rc_evidence_imported`, `public_swarm_alpha_private_artifacts_absent`, and `public_swarm_inference_alpha_rc_ready`. The retained evidence paths are `dist/public-swarm-inference-alpha-live-stage0-requeue-20260527165830/public_swarm_inference_alpha.json`, `dist/public-swarm-inference-alpha-live-stage1-requeue-20260527170600/public_swarm_inference_alpha.json`, and `dist/public-swarm-inference-alpha-live-requeue-summary.json`. `local-smoke` is CI-safe and does not create Kaggle resources. This RC is CPU-only and read-only, not production Swarm Inference, not P2P, and not large-model serving.
- Build the Public Swarm v0.1 Operator Preview with `crowdtensor operator-preview`, which emits `public_swarm_operator_preview_v1` through `scripts/public_swarm_operator_preview_pack.py` and validates with `scripts/public_swarm_operator_preview_check.py`. `operator-preview local-smoke` runs the local public-preview contract, `operator-preview package` creates the operator runbook and join material, `operator-preview live-kaggle` attempts a fresh public Kaggle proof, and `operator-preview evidence-import` promotes retained redacted evidence. Ready reports preserve `public_swarm_operator_preview_ready`, `operator_preview_user_path_ready`, `operator_preview_local_smoke_ready`, `operator_preview_package_ready`, `operator_preview_live_kaggle_ready`, `operator_preview_evidence_import_ready`, `serve_join_generate_ready` or `miner_join_pack_ready`, `cpu_fallback_ready`, `live_preview_ready`, `support_bundle_ready`, `release_readiness_ready`, and optional `gpu_generation_evidence_import_ready`. CPU-only hosts that lack optional HF dependencies report `developer_preview_degraded` plus `operator_preview_cpu_fallback_user_path_ready`; retained evidence imports may report `operator_preview_retained_evidence_ready`. If optional HF/Kaggle/external runtime execution is unavailable, the live path records `external_runtime_blocked` and falls back to retained stage0/stage1 Live Preview RC evidence instead of claiming a fresh run. This is CPU-only by default and read-only, Coordinator-backed, not production Swarm Inference, not libp2p, not DHT, not NAT traversal, and not large-model serving.
- Run the Public Swarm v0.2 Usable Inference Trial with `crowdtensor swarm-trial`, which emits `public_swarm_trial_v1` through `scripts/public_swarm_trial_pack.py` and validates with `scripts/public_swarm_trial_check.py`. `swarm-trial local-loopback` exercises the ordinary `serve` / `join stage0` / `join stage1` / `generate` path when optional HF dependencies are available, `swarm-trial package` creates the shareable runbook and join material, `swarm-trial live-kaggle` wraps the controlled external proof path, and `swarm-trial evidence-import` promotes retained Operator Preview plus GPU generation evidence. Ready reports preserve `public_swarm_trial_ready`, `serve_join_generate_trial_ready`, `stage0_join_ready`, `stage1_join_ready`, `generate_ready`, `generated_token_count_ready`, `support_bundle_ready`, `cpu_fallback_ready`, `private_artifacts_cleaned`, `operator_preview_import_ready`, `gpu_generation_evidence_import_ready`, and `token_rotation_required` for live evidence. CPU-only hosts without optional `[hf]` dependencies may report `swarm_trial_degraded_cpu_fallback_ready` and `external_runtime_blocked` instead of claiming a fresh real-weight generation loop. This is CPU-only by default and read-only, Coordinator-backed, not production Swarm Inference, not libp2p, not DHT, not NAT traversal, not GPU marketplace, and not large-model serving.
- Build the Public Swarm Live Preview RC with `crowdtensor live-preview`, which emits `public_swarm_live_preview_rc_v1` through `scripts/public_swarm_live_preview_rc_pack.py` and validates with `scripts/public_swarm_live_preview_rc_check.py`. `live-preview local-smoke` is CI-safe, `live-preview package` creates a controlled public runbook, `live-preview live-kaggle` wraps the public Kaggle proof with `external_stage_requeue_ready`, `kaggle_kernels_deleted`, `private_artifacts_cleaned`, and `token_rotation_required`, and `live-preview evidence-import` promotes retained Alpha RC plus Developer Preview evidence. Ready reports preserve `public_swarm_live_preview_rc_ready`, `public_swarm_live_preview_local_smoke_ready`, `public_swarm_live_preview_package_ready`, `public_swarm_live_preview_live_kaggle_ready`, `public_swarm_live_preview_evidence_import_ready`, and optional `gpu_generation_evidence_import_ready`. Fresh stage0/stage1 live RC proofs are retained at `dist/public-swarm-live-preview-rc-live-stage0-20260529043801-rc/public_swarm_live_preview_rc.json` and `dist/public-swarm-live-preview-rc-live-stage1-20260529044328-rc/public_swarm_live_preview_rc.json`. This is CPU-only by default and read-only, not production Swarm Inference, not libp2p, not DHT, not NAT traversal, and not large-model serving.
- Run Public Swarm Inference Beta with `crowdtensor public-swarm-beta`, which emits `public_swarm_inference_beta_v1` through `scripts/public_swarm_inference_beta_pack.py` and validates with `scripts/public_swarm_inference_beta_check.py`. `public-swarm-beta product-beta` is the product-shaped aggregate: it requires `public_swarm_product_beta_ready`, `public_swarm_product_rc_ready`, `coordinator_product_surface_ready`, `session_protocol_ready`, `p2p_lite_discovery_ready`, `gpu_generation_evidence_import_ready`, and `cpu_fallback_ready` by combining `crowdtensor serve` / `join` / `generate` / `peer`, `session_protocol_v1`, P2P-lite discovery, retained GPU sharded generation evidence, and the CPU inference fallback. `public-swarm-beta local-loopback` and `public-swarm-beta evidence-import` remain available for the legacy CPU-only split proof and retained Alpha RC import. This is Coordinator-backed, read-only Beta evidence, not production Swarm Inference, not libp2p/DHT/NAT traversal, not Hivemind-level serving, and not large-model serving.
- Run Public Swarm GPU Inference Beta with `crowdtensor public-swarm-gpu-beta`, which emits `public_swarm_gpu_inference_beta_v1` through `scripts/public_swarm_gpu_inference_beta_pack.py` and validates with `scripts/public_swarm_gpu_inference_beta_check.py`. `public-swarm-gpu-beta local-smoke` is CI-safe on CPU-only hosts and reports `public_swarm_gpu_beta_smoke_ready` without claiming usable GPU inference. `public-swarm-gpu-beta local-loopback` requires explicit `hf_transformers_cuda`, CUDA-capable stage Miners advertising `real_llm_sharded_cuda_stage0`, `real_llm_sharded_cuda_stage1`, or `real_llm_sharded_cuda_both`, and should report `public_swarm_gpu_beta_ready`, `gpu_runtime_ready`, `cuda_runtime_available`, `hf_transformers_cuda_ready`, `gpu_stage0_ready`, `gpu_stage1_ready`, plus stage-local partition readiness codes. `public-swarm-gpu-beta kaggle-package` writes private Kaggle GPU stage templates and should report `kaggle_gpu_package_ready`; `public-swarm-gpu-beta evidence-import` promotes a completed GPU report with `external_gpu_runtime_verified`. This is read-only optional CUDA tiny GPT split evidence, not production Swarm Inference, not P2P, not a GPU pooling marketplace, and not large-model serving.
- Run the GPU sharded generation Beta with `crowdtensor gpu-generate`, which emits `gpu_sharded_generation_beta_v1` through `scripts/gpu_sharded_generation_beta_pack.py` and validates with `scripts/gpu_sharded_generation_beta_check.py`. It wraps the CUDA Public Swarm GPU path with `--max-new-tokens` generation chaining, defaults to `sshleifer/tiny-gpt2`, `hf_transformers_cuda`, and stage-local two-stage partitioning, and provides `local-loopback`, `kaggle-auto`, and `evidence-import` modes. A ready report must include `gpu_sharded_generation_ready`, `multi_token_generation_ready`, `generated_token_count >= max_new_tokens`, `distinct_stage_miners`, stage-local partition readiness, and redacted generation summaries with `generated_text_hash` but no raw generated text or generated token ids. This is a tiny GPT CUDA multi-token Beta proof, not production Swarm Inference, not Hivemind-level serving, not P2P, not a GPU marketplace, and not large-model serving.
- Build the Public Swarm Product RC with `crowdtensor public-swarm-product-rc`, which emits `public_swarm_product_rc_v1` through `scripts/public_swarm_product_rc_pack.py` and validates with `scripts/public_swarm_product_rc_check.py`. It adds the user-facing `crowdtensor serve`, `crowdtensor join`, `crowdtensor generate`, and `crowdtensor peer` surfaces, extracts `session_protocol_v1`, and adds `p2p_lite_peer_v1` HTTP-gossip route discovery. This remains Coordinator-backed task execution: P2P-lite only discovers Coordinator/Miner routes and is not libp2p, DHT, NAT traversal, decentralized security, Hivemind-level serving, or large-model serving.
- Try browser-native experiments for WebRTC tensor transport, browser Worker compute probes, and a browser Miner bridge.

## What Is Not Ready

This Alpha is not yet:

- a production DePIN network
- a real LLM training or production inference platform
- a reward, staking, or payment system
- a complete P2P/NAT traversal network
- libp2p/DHT routing; current `crowdtensor peer` is P2P-lite HTTP-gossip discovery only
- a hardened public-internet security model
- a GPU, WebGPU, PyTorch, or Transformers throughput benchmark
- production GPU multi-machine LLM serving; `crowdtensor gpu-generate` is a tiny GPT CUDA Beta proof only

The current workloads are intentionally small and deterministic so the runtime can be tested without GPU access or external model dependencies.

## Quickstart

Run a 5-minute local swarm demo with Python 3.11 or newer.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .[dev]
```

For a fresh-clone onboarding check that avoids system pip / PEP 668 issues and validates the documented console commands from a clean virtualenv, run:

```bash
python scripts/onboarding_gate.py --quick --json-out /tmp/crowdtensor_onboarding_gate.json
```

The `onboarding_gate_v1` report creates a temporary venv, runs `python -m pip install -e .[dev]`, checks `crowdtensor --help`, `crowdtensord --help`, and `crowdtensor-miner --help`, then smoke-validates `crowdtensor local-proof`, `crowdtensor home-infer`, `crowdtensor llm-infer --mock`, `crowdtensor cpu-infer --mode local`, and `crowdtensor release-ready --allow-dirty`. It is an Alpha onboarding gate, not production Swarm Inference readiness.

The install creates these console commands:

```bash
crowdtensor --help
crowdtensord --help
crowdtensor-miner --help
```

Run the First-run Doctor before starting services:

```bash
python3 scripts/doctor.py --json
```

For maintainer release readiness before pushing or tagging:

```bash
crowdtensor release-ready --json
```

The `crowdtensor/cli.py` entrypoint wraps `scripts/release_readiness_pack.py` and emits `release_readiness_v1` under `dist/release-readiness`. It checks Git metadata, the release gate, security preflight, and `demo_manifest_v1`, then reports blocker diagnosis such as `git_dirty`, `release_gate_failed`, or `demo_manifest_failed`. Dirty worktrees block by default; use `--allow-dirty` only for development smoke checks such as `scripts/release_readiness_check.py --allow-dirty`. This is not production readiness for Swarm Inference; it is an Alpha maintainer gate for the current CPU-only public repository state.

For the shortest one-command local proof, run:

```bash
crowdtensor local-proof --json
```

The `crowdtensor/cli.py` entrypoint writes a `local_proof_summary_v1` report under `dist/local-proof` by running Doctor, the runtime matrix, the CPU-only read-only home-compute demo, and the Demo Manifest path. This is not production Swarm Inference; it is a safe local proof that the current checkout can execute the Alpha CPU control-plane path without claiming real LLM serving, GPU pooling, P2P routing, or WebGPU shards.

For the CPU-only inference Beta aggregate path, run:

```bash
crowdtensor cpu-infer --mode local --json
```

The `cpu_inference_beta_v1` report is the recommended CPU inference Beta proof for new users. It wraps `crowdtensor home-infer` and `crowdtensor llm-infer --mock`, writes JSON/Markdown under `dist/cpu-infer`, and summarizes safe read-only `model_bundle_infer` plus fixed-prompt `external_llm_infer` evidence without raw prompts, `output_text`, token values, lease material, or raw inference payloads. Maintainers can also run `python scripts/cpu_inference_beta_check.py --mode local`, `python scripts/cpu_inference_beta_check.py --mode remote-loopback --workload model-bundle`, and `python scripts/cpu_inference_beta_check.py --mode remote-loopback --workload external-llm` to validate the local and loopback remote-demo proof. `--mode remote-existing` wraps an already running `remote-demo doctor/verify/collect` flow with operator-provided tokens. This is CPU-only, read-only, not production Swarm Inference, not P2P, and not arbitrary public prompt serving.

For the current release-candidate CPU inference proof, run:

```bash
crowdtensor cpu-infer --mode beta-rc --json
```

The `cpu_inference_beta_rc_v1` report is the top-level CPU Inference Beta RC artifact. It uses `scripts/cpu_inference_beta_rc_pack.py` to aggregate the local CPU inference Beta, remote-loopback CPU inference Beta, Real two-machine CPU inference Beta rehearsal, Kaggle Remote Miner Beta artifact preparation, `miner_join_pack_v1`, `scripts/kaggle_remote_miner_beta_check.py`, and `demo_manifest_v1`. CI validates it with `scripts/cpu_inference_beta_rc_check.py` and requires `cpu_inference_beta_rc_ready`, `local_cpu_inference_ready`, `remote_loopback_ready`, `two_machine_rehearsal_ready`, `kaggle_remote_miner_artifacts_ready`, `miner_join_pack_ready`, and `cpu_miner_beta_ready`. If a live Kaggle report exists, pass `--kaggle-real-runtime-report dist/kaggle-real-runtime/kaggle_real_runtime_acceptance.json` to import `kaggle_real_runtime_acceptance_v1` as real external runtime evidence and surface `real_runtime_evidence_ready`. This remains CPU-only, read-only, not production Swarm Inference, not P2P, not a GPU/TPU workload path, and not arbitrary prompt serving.

For the first model-execution-graph split proof, run:

```bash
crowdtensor shard-infer --json
```

The Pipeline-Sharded Inference Alpha emits `sharded_inference_cli_v1` and writes `sharded_inference_evidence_v1` under `dist/shard-infer`. It uses the CPU-only read-only `sharded_model_bundle_infer` / `sharded_model_bundle_infer_v1` workload and `sharded_inference_session_v1`: stage 0 produces activation hashes and byte counts, stage 1 consumes the accepted activation payload and must match the single-task `model_bundle_infer` baseline. CI validates this with `scripts/sharded_inference_check.py` and readiness codes such as `stage_0_accepted`, `stage_1_accepted`, `activation_transport_ready`, `baseline_match`, and `sharded_inference_ready`; add `--failure-mode kill-stage-after-claim` to require `stage_requeue_ready`. This is CPU-only and read-only; it is not production Swarm Inference, not P2P, not GPU/TPU pooling, and not real LLM sharding.

For the larger CPU Pipeline-Sharded Inference Beta proof, run:

```bash
crowdtensor shard-infer-beta --mode remote-loopback --json
```

The Beta wrapper emits `remote_sharded_inference_beta_v1` through `scripts/remote_sharded_inference_beta_pack.py`, validates it with `scripts/remote_sharded_inference_beta_check.py`, and adds mode-level readiness codes such as `remote_sharded_inference_ready`, `remote_sharded_loopback_ready`, and `local_sharded_inference_ready`. Use `--failure-mode kill-stage-after-claim` to include the same `stage_requeue_ready` lease-timeout rescue proof while preserving activation hashes, `baseline_match`, CPU-only read-only semantics, and redaction of raw activation/logit payloads. This is still not production Swarm Inference, not P2P, not GPU/TPU pooling, and not real LLM sharding.

The same workload is available in the controlled two-machine helper as `crowdtensor remote-demo --workload sharded-model-bundle`. Its loopback check emits `remote_sharded_inference_acceptance_v1`, `remote_sharded_inference_observability_v1`, `remote_python_sharded_model_bundle_infer`, and `remote_two_machine_sharded_ready` while keeping the same CPU-only/read-only/not production boundaries.

For the deterministic tiny Transformer split proof, run:

```bash
crowdtensor micro-llm-shard-infer --decode-steps 3 --json
```

The Micro-LLM Pipeline-Sharded Inference Alpha emits `micro_llm_sharded_cli_v1` and writes `micro_llm_sharded_evidence_v1` under `dist/micro-llm-shard-infer`. It uses the CPU-only read-only `micro_llm_sharded_infer` / `micro_llm_sharded_infer_v1` workload and `micro_llm_sharded_session_v1`: stage 0 produces hidden-state activation hashes for fixed tiny Transformer contexts, stage 1 performs the lm-head decode and must report `baseline_match` plus `decoded_tokens_match` for the configured `decode_steps`. CI validates this with `scripts/micro_llm_sharded_inference_check.py` and readiness codes `stage_0_accepted`, `stage_1_accepted`, `activation_transport_ready`, `baseline_match`, `decoded_tokens_match`, and `micro_llm_sharded_ready`; add `--failure-mode kill-stage-after-claim` to require `stage_requeue_ready`. This is CPU-only and read-only; it is not production Swarm Inference, not P2P, not GPU/TPU pooling, and not GGUF/llama.cpp or large LLM serving.

To run the same tiny Transformer from an explicit dependency-free file artifact, first build the artifact:

```bash
crowdtensor micro-llm-artifact --output-dir dist/micro-llm-artifact --json
```

Then pass it into the split proof:

```bash
crowdtensor micro-llm-shard-infer \
  --micro-llm-artifact dist/micro-llm-artifact \
  --prompt-texts arn,ten \
  --stage-mode split \
  --require-distinct-stage-miners \
  --decode-steps 3 \
  --json
```

This emits `micro_llm_artifact_v1` and records the artifact id, tokenizer schema, and artifact hash in session/evidence summaries with readiness codes such as `artifact_loaded` and `micro_llm_artifact_ready`. The artifact is a tiny JSON package (`manifest.json`, `config.json`, `tokenizer.json`, `weights.json`) for deterministic CPU validation; it is not a Hugging Face, GGUF, llama.cpp, or large-model artifact.

For the stage-aware split proof, run `crowdtensor micro-llm-shard-infer --stage-mode split --require-distinct-stage-miners --decode-steps 3 --json` or `python scripts/stage_aware_micro_llm_sharded_check.py --base-port 9085 --request-count 2 --decode-steps 3 --require-distinct-stage-miners`. Miners advertise stage capabilities such as `micro_llm_sharded_stage0`, `micro_llm_sharded_stage1`, or `micro_llm_sharded_both`; the Coordinator only leases stage 0 to a stage-0-capable Miner and stage 1 to a stage-1-capable Miner. Successful evidence includes `distinct_stage_miners` and `stage_assignment_valid`, and stage-specific failure modes `--failure-mode kill-stage0-after-claim` / `--failure-mode kill-stage1-after-claim` preserve `stage_requeue_ready`.

For the Remote Micro-LLM Pipeline-Sharded Inference Beta proof, run:

```bash
crowdtensor micro-llm-shard-infer-beta --mode remote-loopback --decode-steps 3 --json
```

The Beta wrapper emits `remote_micro_llm_sharded_beta_v1` through `scripts/remote_micro_llm_sharded_beta_pack.py`, validates it with `scripts/remote_micro_llm_sharded_beta_check.py`, and adds readiness codes such as `remote_micro_llm_sharded_ready`, `remote_micro_llm_sharded_loopback_ready`, and `local_micro_llm_sharded_inference_ready`. The controlled two-machine helper also accepts `crowdtensor remote-demo --workload micro-llm-sharded`; its loopback check emits `remote_micro_llm_sharded_acceptance_v1`, `remote_micro_llm_sharded_observability_v1`, `remote_python_micro_llm_sharded_infer`, and `remote_two_machine_micro_llm_sharded_ready` while preserving activation hashes, `baseline_match`, `decoded_tokens_match`, CPU-only read-only semantics, and redaction of raw activation/logit payloads. This remains not production Swarm Inference, not P2P, not GPU/TPU pooling, and not GGUF/llama.cpp or arbitrary prompt serving.

The Remote Micro-LLM stage-aware Beta path supports `crowdtensor micro-llm-shard-infer-beta --mode remote-loopback --stage-mode split --require-distinct-stage-miners --decode-steps 3 --json` and the two-machine helper supports `crowdtensor remote-demo verify --workload micro-llm-sharded --stage-mode split --require-distinct-stage-miners`. Prepare separate Miner join packs with `--stage-role stage0` and `--stage-role stage1` when proving distinct hosts; this is still a controlled two-machine task-level CPU proof, not model-scale production serving.

The Micro-LLM Live Two-Node RC is the higher-level acceptance wrapper for that stage-aware path:

```bash
crowdtensor micro-llm-live-rc --mode local-generated --port 9182 --request-count 2 --decode-steps 3 --json
python scripts/micro_llm_live_rc_check.py --base-port 9182 --request-count 2 --decode-steps 3
```

It emits `micro_llm_live_rc_v1` through `scripts/micro_llm_live_rc_pack.py`, generates `kaggle-upload-stage0` and `kaggle-upload-stage1`, starts a local Coordinator plus two independent stage Miner processes as `local-generated` stage-upload stand-ins, then imports the existing Kaggle/remote evidence chain. A ready report includes `micro_llm_live_rc_ready`, `local_generated_stage_upload_standins_ready`, `kaggle_micro_llm_sharded_ready`, and `stage_assignment_valid`. `--mode external-existing` verifies an already running public Coordinator plus two external stage Miners and only then may report `external_runtime_verified`. This is CPU-only, read-only toy two-stage micro-LLM evidence; it is not production Swarm Inference, not P2P, not GPU/TPU pooling, and not GGUF/llama.cpp or large-model sharding.

For the optional Hugging Face tiny GPT split proof, install the HF runtime extra and run:

```bash
python -m pip install -e '.[dev,hf]'
crowdtensor real-llm-shard-infer \
  --stage-mode split \
  --require-distinct-stage-miners \
  --hf-model-id sshleifer/tiny-gpt2 \
  --json
```

This emits `real_llm_sharded_cli_v1` and `real_llm_sharded_evidence_v1` for `real_llm_sharded_infer` / `real_llm_sharded_infer_v1`. The Coordinator creates a two-stage `real_llm_sharded_session_v1`, records a safe `real_llm_artifact_v1` summary, leases CPU stage 0 only to Miners advertising `real_llm_sharded_stage0` and CPU stage 1 only to Miners advertising `real_llm_sharded_stage1`, and expects `activation_transport_ready`, `baseline_match`, `decoded_tokens_match`, `real_llm_artifact_ready`, `stage_assignment_valid`, and `real_llm_sharded_ready`. Miner hosts must opt in with `--enable-hf-tiny-gpt-runtime`, optional `--hf-cache-dir`, and `--real-llm-stage-role stage0|stage1|both`. When `--real-llm-backend hf_transformers_cuda` is selected, routing uses `real_llm_sharded_cuda_stage0`, `real_llm_sharded_cuda_stage1`, and `real_llm_sharded_cuda_both`; CUDA is never an implicit fallback.

The remote-shaped wrapper is:

```bash
crowdtensor real-llm-shard-infer-beta \
  --mode remote-loopback \
  --stage-mode split \
  --require-distinct-stage-miners \
  --hf-model-id sshleifer/tiny-gpt2 \
  --json
```

It emits `remote_real_llm_sharded_beta_v1` through `scripts/remote_real_llm_sharded_beta_pack.py` and is checked by `scripts/remote_real_llm_sharded_beta_check.py` with readiness codes such as `remote_real_llm_sharded_ready`, `remote_real_llm_sharded_loopback_ready`, and `local_real_llm_sharded_inference_ready`. This is a CPU-only, read-only, optional [hf] / Transformers tiny-model evidence path; it is not production Swarm Inference, not P2P, not GPU/TPU pooling, not GGUF/llama.cpp serving, and not large-model serving.

For the same real-weight path through the high-level two-machine operator wrapper, use `crowdtensor remote-demo --workload real-llm-sharded`. Install the optional HF runtime on the Coordinator and both stage Miner hosts first:

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

Run the generated stage0 and stage1 Miner launchers on two distinct hosts, then verify from the operator host:

```bash
crowdtensor remote-demo verify \
  --workload real-llm-sharded \
  --stage-mode split \
  --require-distinct-stage-miners \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id real-stage0 \
  --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" \
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \
  --output-dir dist/remote-real-llm-stage0 \
  --json
```

This route emits `remote_real_llm_sharded_runbook_v1`, `remote_real_llm_sharded_acceptance_v1`, `remote_real_llm_sharded_observability_v1`, and `remote_real_llm_sharded_beta_v1` for `remote_python_real_llm_sharded_infer`. A ready run includes `remote_two_machine_real_llm_sharded_ready`, `real_llm_artifact_ready`, `activation_transport_ready`, `baseline_match`, `decoded_tokens_match`, `distinct_stage_miners`, and `stage_assignment_valid`. If the optional runtime is missing, reports surface `hf_dependencies_missing` with the operator action `python -m pip install -e '.[hf]'`. Public artifacts keep raw prompts, hidden states, logits, activations, tokens, and runtime secrets out of summaries. This is still a controlled tiny GPT CPU split proof, not production Swarm Inference, not P2P, not GPU/TPU pooling, not GGUF/llama.cpp serving, and not large-model serving.

The Real Small-LLM Sharded Inference Live RC is the higher-level generated-stage wrapper for the same real-weight path:

```bash
crowdtensor real-llm-live-rc --mode local-generated --port 9184 --request-count 1 --json
python scripts/real_llm_live_rc_check.py --base-port 9184 --request-count 1
```

It emits `real_llm_live_rc_v1` through `scripts/real_llm_live_rc_pack.py`, generates `kaggle-upload-real-llm-stage0` and `kaggle-upload-real-llm-stage1`, starts a local Coordinator plus two independent stage Miner processes as `local-generated` stage-upload stand-ins, and verifies through `remote_real_llm_sharded_beta_v1`. A ready report includes `real_llm_live_rc_ready`, `local_generated_real_llm_stage_upload_standins_ready`, `remote_real_llm_sharded_ready`, `real_llm_artifact_ready`, `stage_assignment_valid`, and `decoded_tokens_match` while preserving `--enable-hf-tiny-gpt-runtime` and `--real-llm-stage-role` in generated Miner launchers. `--mode kaggle-generated` only prepares the two private upload packages and runbook; `--mode external-existing` verifies an already running public Coordinator plus two external stage Miners and only then may report `external_runtime_verified` and `kaggle_real_llm_sharded_ready`. This remains CPU-only and read-only; it is not production Swarm Inference, not P2P, not GPU/TPU pooling, not GGUF/llama.cpp serving, and not large-model serving.

The Real Internet Swarm Inference Alpha is the wider milestone wrapper around the same tiny GPT split path:

```bash
crowdtensor real-llm-internet-alpha --mode local-generated --port 9187 --base-port 9188 --request-count 1 --json
python scripts/real_llm_internet_alpha_check.py --port 9187 --base-port 9188 --request-count 1
```

It emits `real_llm_internet_alpha_v1` through `scripts/real_llm_internet_alpha_pack.py`. `local-generated` runs the real Live RC and also requires local stage-specific failure recovery through `stage_requeue_ready` and `real_llm_stage_requeue_ready`; a ready report includes `real_llm_internet_alpha_ready`, `real_llm_live_rc_ready`, `remote_real_llm_sharded_ready`, `real_llm_artifact_ready`, `decoded_tokens_match`, `activation_transport_ready`, `distinct_stage_miners`, and `stage_assignment_valid` while keeping `external_runtime_verified` false. `package` prepares the public Coordinator and two stage upload packages only. `external-existing` verifies an already running public Coordinator plus two external stage Miners and only then may report `external_runtime_verified`. Reports carry `token_rotation_required`, redact raw prompts, hidden states, logits, activations, tokens, and lease material, and remain CPU-only/read-only. This is not production Swarm Inference, not P2P, not GPU/TPU pooling, not GGUF/llama.cpp serving, and not large-model serving.

The Real Internet Swarm Inference Beta is the side-effectful automation wrapper for that same external tiny GPT path:

```bash
crowdtensor real-llm-internet-beta \
  --mode kaggle-auto \
  --public-host 24.199.118.54 \
  --port 9190 \
  --base-port 9191 \
  --request-count 2 \
  --json

python scripts/real_llm_internet_beta_check.py --port 9190 --base-port 9191 --request-count 2
```

It emits `real_llm_internet_beta_v1` through `scripts/real_llm_internet_beta_pack.py`. `kaggle-auto` generates the Alpha package, starts the temporary public Coordinator, creates two private Kaggle CPU script kernels by default or private Kaggle GPU kernels when `--real-llm-backend hf_transformers_cuda` is selected, waits for `external-existing` verification, deletes the temporary kernels, stops the Coordinator, and writes cleanup-backed evidence. A ready report must include `real_llm_internet_beta_ready`, `real_llm_internet_alpha_ready`, `external_runtime_verified`, both Kaggle stages seen, `decoded_tokens_match`, `distinct_stage_miners`, `stage_assignment_valid`, `kaggle_kernels_deleted`, and `token_rotation_required`. With CUDA selected, the CPU Coordinator uses metadata-only artifact inspection and the Kaggle Miners must provide torch CUDA. `scripts/real_llm_internet_beta_check.py` is CI-safe and validates the contract with a fake runner; it does not create Kaggle resources. The default path remains CPU-only; the CUDA path is optional read-only tiny GPT evidence, not production Swarm Inference, not P2P, not GPU pooling, and not large-model serving.

The user-facing Swarm Inference Beta packages the same real tiny GPT split path as a two-stage operator workflow:

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

It emits `swarm_inference_beta_v1` through `scripts/swarm_inference_beta_pack.py`. `swarm-infer-beta live` is the side-effectful `kaggle-auto` path: it wraps `real_llm_internet_beta_v1`, starts a temporary public Coordinator, pushes two private Kaggle CPU stage kernels, verifies `external_runtime_verified`, deletes the kernels, writes `support_bundle.json`, removes local live private artifacts and raw runtime state by default, and only then may report `swarm_inference_beta_live_ready`, `swarm_inference_beta_ready`, `two_machine_swarm_inference_ready`, `real_llm_internet_beta_ready`, `kaggle_kernels_deleted`, `swarm_inference_beta_live_private_artifacts_cleaned`, and `token_rotation_required`. Pass `--keep-live-private-artifacts` only when debugging failed live runs; do not publish or commit those files. `swarm-infer-beta prepare` creates `operator.private.env`, stage-specific `miner.private.env` files, a hashed `miner_registry.json`, stage0/stage1 join packs, `SWARM_INFERENCE_BETA.md`, and command wrappers. `swarm-infer-beta verify` wraps the existing `remote_real_llm_sharded_beta_v1` route, requires `real_llm_split_route_ready`, `decoded_tokens_match`, `distinct_stage_miners`, `stage_assignment_valid`, and reports `swarm_inference_beta_ready` plus `two_machine_swarm_inference_ready` when the controlled split proof completes. It can import retained `real_llm_internet_beta_v1` evidence as `external_beta_evidence_imported` without pretending it was a fresh live run. `swarm-infer-beta collect` gathers redacted evidence/support, and `swarm-infer-beta clean` is dry-run by default. This is CPU-only, read-only, not production Swarm Inference, not P2P, and not large-model serving.

Public Swarm Inference Alpha RC is the release-candidate evidence layer for the same proof:

```bash
crowdtensor public-swarm-alpha-rc --mode evidence-import --json
python scripts/public_swarm_inference_alpha_rc_check.py --mode local-smoke
```

It emits `public_swarm_inference_alpha_rc_v1` through `scripts/public_swarm_inference_alpha_rc_pack.py`. `evidence-import` reads retained public reports for the completed live stage0/stage1 victim-rescue runs, verifies `stage0_live_requeue_evidence_ready`, `stage1_live_requeue_evidence_ready`, `public_swarm_live_requeue_evidence_ready`, `public_swarm_live_requeue_summary_ready`, `public_swarm_alpha_private_artifacts_absent`, `public_swarm_alpha_rc_evidence_imported`, and `public_swarm_inference_alpha_rc_ready`, then writes JSON/Markdown under `dist/public-swarm-inference-alpha-rc`. The current retained proof set is `dist/public-swarm-inference-alpha-live-stage0-requeue-20260527165830/public_swarm_inference_alpha.json`, `dist/public-swarm-inference-alpha-live-stage1-requeue-20260527170600/public_swarm_inference_alpha.json`, and `dist/public-swarm-inference-alpha-live-requeue-summary.json`. `local-smoke` only runs the CI-safe contract path and does not create Kaggle resources. This remains CPU-only, read-only, not production Swarm Inference, not P2P, and not large-model serving.

Public Swarm Inference Beta is the ordinary user entrypoint for the current Coordinator-backed product surface:

```bash
crowdtensor public-swarm-beta product-beta --json
python scripts/public_swarm_inference_beta_check.py --mode product-beta --json

# Legacy CPU-only split proof and retained Alpha RC import remain available:
crowdtensor public-swarm-beta local-loopback --base-port 9290 --request-count 1 --json
crowdtensor public-swarm-beta evidence-import --json
crowdtensor public-swarm-beta prepare --coordinator-url https://YOUR_COORDINATOR_HOST --json
crowdtensor public-swarm-beta coordinator --json
crowdtensor public-swarm-beta miner --stage stage0 --json
crowdtensor public-swarm-beta miner --stage stage1 --json
crowdtensor public-swarm-beta verify --coordinator-url https://YOUR_COORDINATOR_HOST --json
crowdtensor public-swarm-beta collect --coordinator-url https://YOUR_COORDINATOR_HOST --json
crowdtensor public-swarm-beta clean --json
```

It emits `public_swarm_inference_beta_v1` through `scripts/public_swarm_inference_beta_pack.py`. `product-beta` aggregates the Product RC (`serve`, `join`, `generate`, `peer`), `session_protocol_v1`, `p2p_lite_peer_v1`, retained `gpu_sharded_generation_beta_v1` evidence, and the local CPU inference fallback. A ready report requires `public_swarm_inference_beta_ready`, `public_swarm_product_beta_ready`, `public_swarm_product_rc_ready`, `coordinator_product_surface_ready`, `session_protocol_ready`, `p2p_lite_discovery_ready`, `gpu_generation_evidence_import_ready`, `cpu_fallback_ready`, and `local_cpu_inference_ready`. `public-swarm-beta local-loopback` still wraps `remote_real_llm_sharded_beta_v1` in split mode and requires `two_stage_split_inference_ready`, `local_loopback_ready`, `decoded_tokens_match`, `distinct_stage_miners`, and `stage_assignment_valid`; `public-swarm-beta evidence-import` still imports retained Alpha RC evidence with `public_swarm_beta_evidence_import_ready`, `external_live_evidence_imported`, `stage0_live_requeue_evidence_ready`, and `stage1_live_requeue_evidence_ready`. This is Coordinator-backed, read-only Beta evidence, not production Swarm Inference, not libp2p/DHT/NAT traversal, not Hivemind-level serving, and not large-model serving.

Public Swarm Product Beta is the ordinary open-source user path over the current product surface:

```bash
python -m pip install -e '.[hf]'
crowdtensor public-swarm-product-beta local-loopback --base-port 9320 --max-new-tokens 2 --json
crowdtensor public-swarm-product-beta package --target kaggle --json
crowdtensor public-swarm-product-beta external-existing --coordinator-url https://YOUR_COORDINATOR_HOST --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" --admin-token "$CROWDTENSOR_ADMIN_TOKEN" --json
python scripts/public_swarm_product_beta_check.py --mode local-loopback --json
python scripts/public_swarm_product_beta_check.py --mode package --target kaggle --json
python scripts/public_swarm_product_beta_check.py --mode external-existing --json
```

It emits `public_swarm_product_beta_v1` through `scripts/public_swarm_product_beta_pack.py` and is checked by `scripts/public_swarm_product_beta_check.py`. `local-loopback` proves the user-facing `serve` / `join stage0` / `join stage1` / `generate` path and should preserve `public_swarm_product_beta_ready`, `public_swarm_product_beta_user_path_ready`, `serve_ready`, `stage0_join_ready`, `stage1_join_ready`, `generate_ready`, `serve_join_generate_loop_ready`, `remote_generate_session_ready`, `public_swarm_generate_ready`, `decoded_tokens_match`, `distinct_stage_miners`, `stage_assignment_valid`, `support_bundle_ready`, and `private_artifacts_cleaned`. `package` creates the two-machine/Kaggle join material while keeping `private_artifacts_local_only` and `miner_join_pack_ready`; `external-existing` verifies an already running Coordinator plus external stage Miners. Missing optional HF dependencies surface `hf_dependencies_missing`. This Product Beta is CPU-only by default, read-only, Coordinator-backed, not production Swarm Inference, not libp2p, not DHT, not NAT traversal, and not large-model serving.

Public Swarm Developer Preview is the larger ordinary-user preview path over Product Beta:

```bash
python -m pip install -e '.[hf]'
crowdtensor preview local --base-port 9330 --max-new-tokens 2 --json
crowdtensor preview package --target kaggle --json
crowdtensor preview external-existing --coordinator-url https://YOUR_COORDINATOR_HOST --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" --admin-token "$CROWDTENSOR_ADMIN_TOKEN" --json
crowdtensor preview evidence-import --product-beta-report dist/public-swarm-product-beta/public_swarm_product_beta.json --json
python scripts/public_swarm_developer_preview_check.py --mode local --json
python scripts/public_swarm_developer_preview_check.py --mode package --target kaggle --json
python scripts/public_swarm_developer_preview_check.py --mode external-existing --json
python scripts/public_swarm_developer_preview_check.py --mode evidence-import --json
```

It emits `public_swarm_developer_preview_v1` through `scripts/public_swarm_developer_preview_pack.py` and is checked by `scripts/public_swarm_developer_preview_check.py`. `preview local` wraps the Product Beta `serve` / `join stage0` / `join stage1` / `generate` path and should preserve `developer_preview_ready`, `public_swarm_developer_preview_ready`, `local_two_stage_generation_ready`, `serve_join_generate_ready`, `product_beta_ready`, `support_bundle_ready`, `cpu_fallback_ready`, `local_cpu_inference_ready`, `gpu_generation_evidence_import_ready` when retained GPU evidence is present, and the Product Beta `hf_dependencies_missing` behavior when optional `[hf]` dependencies are absent. `preview package` creates two-machine or Kaggle join material; `preview external-existing` verifies an already running controlled runtime; `preview evidence-import` imports retained redacted Product Beta and optional GPU generation reports. This Developer Preview is CPU-only by default, read-only, Coordinator-backed, not production Swarm Inference, not libp2p, not DHT, not NAT traversal, and not large-model serving.

Public Swarm v0.1 Operator Preview is the recommended top-level operator artifact over the current Coordinator-backed stack:

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

It emits `public_swarm_operator_preview_v1` through `scripts/public_swarm_operator_preview_pack.py` and is checked by `scripts/public_swarm_operator_preview_check.py`. The preview aggregates Developer Preview, Live Preview RC, release readiness, Support Bundle diagnostics, CPU fallback, and retained GPU generation evidence into one user-facing report. A ready report preserves `public_swarm_operator_preview_ready`, `operator_preview_user_path_ready`, `operator_preview_local_smoke_ready`, `operator_preview_package_ready`, `operator_preview_live_kaggle_ready`, `operator_preview_evidence_import_ready`, `serve_join_generate_ready` for executable local paths or `miner_join_pack_ready` / `private_artifacts_local_only` for package paths, `cpu_fallback_ready`, `live_preview_ready`, `support_bundle_ready`, `release_readiness_ready`, and `gpu_generation_evidence_import_ready` when retained GPU evidence is available. CPU-only hosts that lack optional HF dependencies report `developer_preview_degraded` plus `operator_preview_cpu_fallback_user_path_ready`; retained evidence imports may report `operator_preview_retained_evidence_ready`. If `operator-preview live-kaggle` cannot complete because optional HF dependencies, Kaggle, or external runtime execution is unavailable, it records `external_runtime_blocked` and imports retained stage0/stage1 Live Preview RC evidence instead of claiming a fresh external run. This Operator Preview is CPU-only by default, read-only, Coordinator-backed, not production Swarm Inference, not libp2p, not DHT, not NAT traversal, and not large-model serving.

Public Swarm v0.2 Usable Inference Trial is the recommended ordinary-user trial entrypoint over the current Coordinator-backed product surface:

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

It emits `public_swarm_trial_v1` through `scripts/public_swarm_trial_pack.py` and is checked by `scripts/public_swarm_trial_check.py`. The trial aggregates Product Beta, Operator Preview, Support Bundle diagnostics, CPU fallback, and retained `gpu_sharded_generation_beta_v1` evidence into one user-facing report. A ready local-loopback report preserves `public_swarm_trial_ready`, `serve_join_generate_trial_ready`, `stage0_join_ready`, `stage1_join_ready`, `generate_ready`, `generated_token_count_ready`, `support_bundle_ready`, `cpu_fallback_ready`, `private_artifacts_cleaned`, and `operator_preview_import_ready`; evidence-import also preserves `gpu_generation_evidence_import_ready`, and live paths preserve `token_rotation_required`. CPU-only hosts without optional `[hf]` dependencies may report `swarm_trial_degraded_cpu_fallback_ready` and `external_runtime_blocked` rather than claiming a fresh real-weight generation loop. The trial is CPU-only by default, read-only, Coordinator-backed, not production Swarm Inference, not libp2p, not DHT, not NAT traversal, not GPU marketplace, and not large-model serving.

Public Swarm Inference Beta RC is the release-candidate layer for the current product path:

```bash
crowdtensor public-swarm-beta-rc local-loopback --base-port 9310 --max-new-tokens 2 --json
crowdtensor public-swarm-beta-rc package --target kaggle --json
crowdtensor public-swarm-beta-rc external-existing --coordinator-url https://YOUR_COORDINATOR_HOST --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" --admin-token "$CROWDTENSOR_ADMIN_TOKEN" --json
python scripts/public_swarm_inference_beta_rc_check.py --mode local-loopback --json
python scripts/public_swarm_inference_beta_rc_check.py --mode package --target kaggle --json
python scripts/public_swarm_inference_beta_rc_check.py --mode external-existing --json
```

It emits `public_swarm_inference_beta_rc_v1` through `scripts/public_swarm_inference_beta_rc_pack.py` and is checked by `scripts/public_swarm_inference_beta_rc_check.py`. The RC aggregates `public_swarm_product_beta_ready`, `p2p_lite_route_ready`, `p2p_lite_discovery_ready`, `cpu_fallback_ready`, and a product `serve` / `join` / `generate` loop that should report `serve_join_generate_loop_ready`, `remote_generate_session_ready`, and `public_swarm_generate_ready` when optional `[hf]` dependencies are installed. `package` preserves `private_artifacts_local_only` and `miner_join_pack_ready`; `external-existing` may report `external_runtime_verified` only against an already running Coordinator plus stage Miners. On hosts missing `transformers`, the real local loop reports `hf_dependencies_missing` with the operator action `python -m pip install -e '.[hf]'`. This RC is CPU-only by default, read-only, Coordinator-backed, not production Swarm Inference, not libp2p/DHT/NAT traversal, not Hivemind-level serving, and not large-model serving.

Public Swarm GPU Inference Beta is the optional CUDA overlay for the same tiny GPT split proof:

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

It emits `public_swarm_gpu_inference_beta_v1` through `scripts/public_swarm_gpu_inference_beta_pack.py`. The CI-safe `public-swarm-gpu-beta local-smoke` path records CUDA availability and should include `public_swarm_gpu_beta_smoke_ready`, but it must not claim `public_swarm_gpu_beta_ready` on CPU-only hosts. The real `public-swarm-gpu-beta local-loopback` path selects `hf_transformers_cuda`, requires `cuda_runtime_available`, `hf_transformers_cuda_ready`, `gpu_runtime_ready`, `gpu_stage0_ready`, `gpu_stage1_ready`, and the stage-local partition codes `stage_local_partition_ready`, `stage0_partition_loaded`, `stage1_partition_loaded`, and `partition_parameter_split_valid`; it schedules only Miners advertising `real_llm_sharded_cuda_stage0`, `real_llm_sharded_cuda_stage1`, or `real_llm_sharded_cuda_both`. `public-swarm-gpu-beta kaggle-package` prepares private Kaggle GPU stage templates and should report `kaggle_gpu_package_ready`. The side-effectful `public-swarm-gpu-beta kaggle-auto` path starts a temporary CPU-capable public Coordinator, defers CUDA runtime checks to private Kaggle GPU stage Miners, pushes private Kaggle GPU kernels, verifies `external_gpu_runtime_verified`, and deletes those kernels before it may report `public_swarm_gpu_beta_kaggle_auto_ready`, `kaggle_kernels_deleted`, and `token_rotation_required`; `scripts/public_swarm_gpu_inference_beta_check.py --mode kaggle-auto` is fake-runner only and does not create Kaggle resources. Kaggle CUDA kernels default to a compatibility-pinned tiny runtime (`torch==2.7.1+cu118`, `torchvision==0.22.1+cu118`, `transformers==4.40.2`) so older Kaggle GPUs such as Tesla P100 can execute the proof. `--real-llm-partition-mode stage-local` is the default for the GPU Beta overlay; it proves stage-owned module placement and parameter-count evidence, not production memory scheduling or model-scale serving. The retained stage-local Kaggle GPU proof is `dist/public-swarm-gpu-beta-stage-local-live-20260528064520-shortslug/public_swarm_gpu_inference_beta_kaggle_auto.json`; the older runtime-pin proof at `dist/public-swarm-gpu-beta-live-20260528-runtimepin/public_swarm_gpu_inference_beta_kaggle_auto.json` is historical pre-stage-local evidence. `public-swarm-gpu-beta evidence-import` imports a completed GPU report and should report `external_gpu_runtime_verified`. This is read-only optional CUDA tiny GPT evidence, not production Swarm Inference, not P2P, not a GPU pooling marketplace, and not large-model serving.

For the multi-token generation wrapper over the same CUDA split path:

```bash
crowdtensor gpu-generate evidence-import \
  --gpu-report dist/gpu-sharded-generation-beta-kaggle-20260528095658/gpu_sharded_generation_beta_kaggle_auto.json \
  --max-new-tokens 16 \
  --json
python scripts/gpu_sharded_generation_beta_check.py \
  --gpu-report dist/gpu-sharded-generation-beta-kaggle-20260528095658/gpu_sharded_generation_beta_kaggle_auto.json \
  --max-new-tokens 16 \
  --json
```

`crowdtensor gpu-generate` emits `gpu_sharded_generation_beta_v1` and is documented in `docs/gpu-sharded-generation-beta.md`. The retained Kaggle GPU proof at `dist/gpu-sharded-generation-beta-kaggle-20260528095658/gpu_sharded_generation_beta_kaggle_auto.json` reports 16 generated tokens, `generated_text_hash`, `multi_token_generation_ready`, `external_gpu_runtime_verified`, `kaggle_kernels_deleted`, distinct stage Miners, and stage-local partition evidence without retaining private env files or raw generated payloads.

To produce the shortest shareable local read-only inference proof:

```bash
crowdtensor home-infer --scenario-id route-baseline --json
```

The `crowdtensor/cli.py` wrapper emits `home_inference_cli_v1` and writes `home_compute_evidence_v1` JSON/Markdown under `dist/home-infer`. It runs the CPU-only `model_bundle_infer` path, summarizes the selected route, fixed `model_bundle_inference_scenario_v1` scenario, safe `request_trace`, `diagnosis_codes`, read-only/redaction status, and artifact paths. Built-in scenarios include `route-baseline`, `gradient-safety`, and `mixed-prompts`; this is not production Swarm Inference, arbitrary prompt serving, or real LLM serving.

To produce a safe proof that CrowdTensor can route fixed prompt work to an operator-owned local LLM runtime:

```bash
crowdtensor llm-infer --mock --json
```

The `llm_inference_cli_v1` wrapper writes `external_llm_evidence_v1` JSON/Markdown under `dist/llm-infer`. It uses the read-only `external_llm_infer` contract with deterministic mock by default, or an explicit `--llm-runtime-cmd` / `--llm-runtime-url` runtime when the operator provides one. Reports include adapter kind, model id, request/completion count, output chars, throughput, and `external_llm_evidence_ready` without exposing raw prompts, `output_text`, runtime URL, API key, lease token, or idempotency material. This is fixed-prompt local runtime evidence, not public arbitrary prompt serving.

Inspect generated caches and temporary artifacts before deleting them:

```bash
crowdtensor clean-artifacts --json
```

Apply the safe cleanup only after reviewing the dry-run report:

```bash
crowdtensor clean-artifacts --apply --json
```

The `cleanup_report_v1` cleanup path removes clearly generated `__pycache__` / `.pyc` caches and old CrowdTensor temp directories. It defaults to dry-run, keeps `/tmp/crowdtensor_*.json` and Markdown reports unless `--include-reports` is passed, and does not delete state, source files, release artifacts, or private env material.

Check what this machine can run:

```bash
python3 scripts/runtime_matrix.py --json
```

The runtime capability matrix reports CPU-only workload readiness, optional browser support, optional external LLM command/HTTP runtime configuration, and a hardware/runtime matrix with `hardware_targets`, `recommended_routes`, `matched_capabilities`, `missing_capabilities`, target and route `diagnosis_codes`, `operator_action`, top-level `diagnosis_summary`, and `hardware_diagnosis_summary`. The `nvidia_cuda` target now describes the optional `hf_transformers_cuda` tiny GPT split route: without CUDA it reports `nvidia_cuda_optional_missing`, with partial runtime it reports `nvidia_cuda_detected_adapter_unavailable`, and only a complete CUDA + optional HF runtime can recommend `crowdtensor public-swarm-gpu-beta local-loopback`. It does not print token, URL, or API key values.

Run the matrix-guided home-compute demo:

```bash
python3 scripts/home_compute_demo.py --port 8909 --request-count 4
```

This combines the runtime capability matrix with the read-only `model_bundle_infer` path and reports safe latency, throughput, `hardware_profile`, selected capability route, `route_decision`, a Coordinator-derived `request_trace`, read-only, redaction status, and stable `diagnosis_codes` such as `home_compute_ready` and `runtime_matrix_blocked`. It is a CPU-only Swarm Inference-shaped demo, not real LLM serving or GPU pooling.

Build a safe, shareable evidence pack for issue reports or demos:

```bash
python3 scripts/home_compute_evidence_pack.py \
  --port 8911 \
  --request-count 4 \
  --json-out /tmp/crowdtensor_home_evidence.json \
  --markdown-out /tmp/crowdtensor_home_evidence.md
```

The `home_compute_evidence_v1` report wraps the runtime matrix, `route_decision`, `matched_capabilities`, safe metrics, capped `request_trace`, and `diagnosis_codes` rows without exposing token, URL, API key, lease, idempotency, weight, or delta-shaped fields. CI validates this with `scripts/home_compute_evidence_check.py`, and runtime acceptance can skip it with `--skip-home-compute-evidence`.

Build a safe, shareable remote-compute evidence pack:

```bash
python3 scripts/remote_compute_evidence_pack.py \
  --port 8912 \
  --request-count 4 \
  --json-out /tmp/crowdtensor_remote_evidence.json \
  --markdown-out /tmp/crowdtensor_remote_evidence.md
```

The `remote_compute_evidence_v1` report runs a registry-backed remote-style Python Miner through the read-only `model_bundle_infer` path, records `remote_python_model_bundle_infer`, route capabilities, safe latency/throughput, capped `request_trace` rows, and `remote_compute_observability_v1`, and verifies the invite registry stores only a hashed token. CI validates this with `scripts/remote_compute_evidence_check.py`; runtime acceptance can opt in with `--include-remote-evidence`.

Run a controlled local multi-Miner scenario sweep:

```bash
python3 scripts/multi_miner_scenario_sweep_check.py \
  --port 8916 \
  --execution-mode concurrent \
  --scenario-ids route-baseline,gradient-safety,mixed-prompts
```

The `multi_miner_scenario_sweep_v1` report creates three read-only `POST /admin/inference-sessions` tasks, starts registry-backed Python Miner identities concurrently through the CPU-only `model_bundle_infer` route `local_multi_miner_model_bundle_infer`, verifies each fixed `model_bundle_inference_scenario_v1` scenario match, records `multi_miner_scenario_sweep_observability_v1`, checks read-only/redaction/hashed-registry safety, confirms one accepted ledger row per task, and emits `multi_miner_concurrent_ready` when all expected miners are seen. Add `--failure-mode kill-after-claim` to terminate one Miner after claim, wait for lease timeout requeue, and require a rescue Miner to finish the requeued task with `multi_miner_requeue_ready`. This is local controlled lease-race and requeue evidence, not P2P routing, production throughput scaling, GPU pooling, or production Swarm Inference. Runtime acceptance can opt in with `--include-multi-miner-sweep` for the happy path and `--include-multi-miner-requeue` for the failure path.

Build the local-loopback Demo Manifest as the latest output artifact:

```bash
python3 scripts/demo_manifest_pack.py \
  --output-dir dist/demo-manifest \
  --port 8914 \
  --request-count 4
```

The `demo_manifest_v1` artifact indexes `runtime_matrix.json`, `remote_compute_evidence_v1`, `external_llm_evidence_v1`, `support_bundle`, and `remote_compute_observability_v1` summaries in one safe JSON/Markdown pair. It is the recommended handoff artifact for showing what this checkout can run today. The external LLM entry uses deterministic mock evidence by default and does not expose raw prompts, `output_text`, runtime URL, or API key. CI validates the path with `scripts/demo_manifest_check.py`.

Run the Real two-machine CPU inference Beta. This is the 15-minute two-machine CPU inference Beta path for a Coordinator host and a Miner host:

```bash
crowdtensor remote-demo prepare \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --scenario-id route-baseline \
  --output-dir dist/remote-home-compute \
  --json
```

After starting the generated Coordinator command on the Coordinator host and the generated `crowdtensor-miner` command on the Miner host, verify the same read-only session:

```bash
. dist/remote-home-compute/operator.private.env
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

Use `crowdtensor remote-demo clean --output-dir dist/remote-home-compute --json` to dry-run cleanup of known generated artifacts; add `--apply` to delete public generated files and `--include-private` only when you intentionally want to remove `operator.private.env`, `miner.private.env`, and `miner_registry.json`.

The `crowdtensor remote-demo` path emits `remote_home_compute_demo_v1` and is the preferred operator wrapper for the controlled Beta-shaped home-compute demo. It reuses `scripts/remote_home_compute_demo_pack.py`, `operator.private.env`, `miner.private.env`, `POST /admin/inference-sessions`, `model_bundle_infer`, `remote_python_model_bundle_infer`, `remote_compute_evidence_v1`, and `remote_demo_observability_v1`. `remote-demo doctor`, `remote-demo collect`, and `remote-demo clean` emit `remote_home_compute_doctor_v1`, `remote_home_compute_collect_v1`, and `remote_home_compute_cleanup_v1`; `scripts/remote_home_compute_demo_check.py` validates prepare, doctor, verify, collect, and clean for the local stand-ins. `scripts/remote_two_machine_beta_check.py` emits `remote_two_machine_beta_check_v1` in CI and requires `remote_two_machine_inference_ready`, `remote_two_machine_external_llm_ready`, and `remote_two_machine_beta_ready` from the loopback stand-in before this path is considered healthy. This is task-level remote CPU inference, not model sharding, not production Swarm Inference, not P2P routing, and not GPU pooling; real two-machine use still requires operator-provided TLS, VPN, tunnel, or trusted network.

Run the Kaggle Remote Miner Beta when you want a real external temporary CPU Miner without exposing a Coordinator from Kaggle:

```bash
crowdtensor remote-demo prepare \
  --target kaggle \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id kaggle-cpu-1 \
  --output-dir dist/remote-home-compute-kaggle \
  --json
```

Upload only `miner.private.env` and the generated `kaggle_remote_miner.py` to the Kaggle Notebook, then run the script from a checkout installed with `python -m pip install -e .`. Keep `operator.private.env` on the operator side, then use the normal `remote-demo doctor`, `remote-demo verify`, and `remote-demo collect` commands against the same Coordinator URL. `scripts/kaggle_remote_miner_beta_check.py` emits `kaggle_remote_miner_beta_check_v1`, validates `--target kaggle`, requires `kaggle_remote_miner_prepare_ready` and `kaggle_remote_miner_beta_ready`, and confirms the protocol through a local loopback stand-in. Kaggle is treated as an outbound Miner environment; GPU/TPU workload adapters are not enabled by this path, and it remains not production Swarm Inference and not P2P.

Use Kaggle Real Runtime Acceptance when you want to prove an actual Kaggle CPU Notebook can connect outbound to a public operator-owned Coordinator. The current temporary HTTP target is `24.199.118.54`, and generated tokens must be rotated after the run:

```bash
crowdtensor remote-demo kaggle-real \
  --action prepare \
  --public-host 24.199.118.54 \
  --port 9180 \
  --miner-id kaggle-cpu-1 \
  --output-dir dist/kaggle-real-runtime \
  --json
```

Start the generated `start_coordinator.sh`, upload only `dist/kaggle-real-runtime/kaggle-upload/miner.private.env` and `dist/kaggle-real-runtime/kaggle-upload/kaggle_remote_miner.py` to Kaggle, run `python kaggle_remote_miner.py --env-file miner.private.env`, then verify:

```bash
crowdtensor remote-demo kaggle-real \
  --action verify \
  --public-host 24.199.118.54 \
  --port 9180 \
  --output-dir dist/kaggle-real-runtime \
  --json
```

This emits `kaggle_real_runtime_acceptance_v1` through `scripts/kaggle_real_runtime_acceptance_pack.py`; CI runs `scripts/kaggle_real_runtime_acceptance_check.py` for artifact safety only. A real ready report carries `kaggle_artifacts_ready`, `coordinator_public_ready`, `kaggle_miner_seen`, `kaggle_result_accepted`, and `kaggle_real_runtime_ready`. `operator.private.env` stays off Kaggle, `token_rotation_required` is reported, and temporary HTTP is not production security. This path is CPU-only `model_bundle_infer`, read-only, not production Swarm Inference, not P2P, and not GPU/TPU workload execution.

The same real-runtime wrapper can prepare the stage-aware micro-LLM split proof. It creates two Kaggle-only upload packages, `kaggle-upload-stage0` and `kaggle-upload-stage1`, backed by distinct hashed Miner identities:

```bash
crowdtensor remote-demo kaggle-real \
  --action prepare \
  --workload micro-llm-sharded \
  --stage-mode split \
  --decode-steps 3 \
  --public-host 24.199.118.54 \
  --port 9180 \
  --miner-id kaggle-cpu-1 \
  --output-dir dist/kaggle-real-runtime \
  --json
```

Upload `kaggle-upload-stage0` to one private Kaggle Notebook and `kaggle-upload-stage1` to a second private Kaggle Notebook. After both Notebooks run `python kaggle_remote_miner.py --env-file miner.private.env`, verify with the same `--workload micro-llm-sharded --stage-mode split --decode-steps 3` flags. A live success adds `kaggle_micro_llm_stage0_seen`, `kaggle_micro_llm_stage1_seen`, `kaggle_micro_llm_stage_assignment_valid`, `stage_assignment_valid`, and `kaggle_micro_llm_sharded_ready`.

For Kaggle CLI-driven operator runs, `scripts/kaggle_micro_llm_live_package.py` can turn the prepared `kaggle-upload-stage0` / `kaggle-upload-stage1` directories into private Kaggle dataset and script-kernel upload folders. Default mode keeps source and stage env files in a private dataset; `--inline-kernel-payload` embeds the source tarball and stage `miner.private.env` in each private kernel source for cases where Kaggle dataset mounting is unreliable. Inline mode is a controlled fallback only: do not publish those kernels, do not commit generated kernel code, delete the temporary Kaggle kernels/dataset after the run, and rotate the generated Miner tokens. This remains a deterministic toy two-stage pipeline, not large-model sharding, not GGUF/llama.cpp serving, and not production Swarm Inference.

The first artifact-backed live Kaggle split proof was completed with public Coordinator `24.199.118.54:9180`, two private Kaggle CPU script kernels acting as stage 0 and stage 1 Miners, and `micro_llm_artifact_v1`. The retained local evidence is `dist/kaggle-micro-llm-live/external-real/kaggle_real_runtime_acceptance.json`; it reports `ok: true`, `artifact_loaded`, `micro_llm_artifact_ready`, `kaggle_micro_llm_stage0_seen`, `kaggle_micro_llm_stage1_seen`, `kaggle_micro_llm_stage_assignment_valid`, `stage_assignment_valid`, `baseline_match`, `decoded_tokens_match`, and `kaggle_micro_llm_sharded_ready`. The temporary Kaggle kernels/dataset used for that proof were deleted after evidence collection.

For the same proof as a release-candidate acceptance pack, use `crowdtensor micro-llm-live-rc`. The default `local-generated` mode is CI-safe and starts local stand-ins from the generated stage upload packages; it must report `local_generated_stage_upload_standins_ready` but must not claim `external_runtime_verified`. Pass `--micro-llm-artifact dist/micro-llm-artifact` to make the RC use the same file-backed model package. A ready artifact-backed report should carry `micro_llm_live_rc_ready`, `kaggle_micro_llm_sharded_ready`, `artifact_loaded`, `micro_llm_artifact_ready`, `distinct_stage_miners`, and `stage_assignment_valid` while keeping raw activations and token material out of public artifacts. After two real Kaggle Notebooks or two real machines are already running, rerun with `--mode external-existing --coordinator-url http://24.199.118.54:9180 --kaggle-output-dir dist/kaggle-real-runtime ...` to verify external runtime evidence.

For the real-weight tiny Hugging Face split path, use `crowdtensor real-llm-live-rc`. `local-generated` validates the generated `kaggle-upload-real-llm-stage0` and `kaggle-upload-real-llm-stage1` packages through local stand-ins and should report `local_generated_real_llm_stage_upload_standins_ready`; `kaggle-generated` prepares those two packages and the operator runbook only; `external-existing` verifies an already running public Coordinator plus two external stage Miners and only then may report `external_runtime_verified` and `kaggle_real_llm_sharded_ready`. Generated Miner launchers use `--enable-hf-tiny-gpt-runtime` and `--real-llm-stage-role stage0|stage1`. `scripts/kaggle_real_llm_live_package.py` emits `kaggle_real_llm_live_package_v1` and can package those real LLM stage uploads as private Kaggle dataset/script-kernel folders; a ready package reports `kaggle_real_llm_live_package_ready`. `--inline-kernel-payload` is the temporary fallback for Kaggle input-mount issues and embeds stage `miner.private.env` into private kernel source, so delete the temporary kernels/dataset and rotate tokens after proof.

For a single milestone artifact around the real-weight split path, use `crowdtensor real-llm-internet-alpha`. `local-generated` aggregates `real_llm_live_rc_v1` with local stage0/stage1 requeue checks and must include `real_llm_internet_alpha_ready`, `real_llm_stage_requeue_ready`, and `stage_requeue_ready` without claiming `external_runtime_verified`. `package` generates the public runbook and stage uploads without a live claim. `external-existing` is the only mode that can report `external_runtime_verified` after two already running external stage Miners complete. The same boundary applies: CPU-only, read-only, not production Swarm Inference, not P2P, not large-model serving, and token rotation is required after temporary public HTTP runs.

The first Real Internet Swarm Inference Alpha external proof completed with public Coordinator `24.199.118.54:9187`, two private Kaggle CPU script kernels acting as `internet-real-llm-stage0` and `internet-real-llm-stage1`, and `sshleifer/tiny-gpt2`. The retained local evidence is `dist/real-llm-internet-alpha-external/real_llm_internet_alpha.json`; it reports `ok: true`, `external_runtime_verified`, `real_llm_internet_alpha_ready`, `kaggle_real_llm_stage0_seen`, `kaggle_real_llm_stage1_seen`, `kaggle_real_llm_sharded_ready`, `real_llm_artifact_ready`, `baseline_match`, `decoded_tokens_match`, `distinct_stage_miners`, and `stage_assignment_valid`. Temporary Kaggle kernels were deleted after evidence collection and the report marks token rotation required. This is CPU-only, read-only `real_llm_sharded_infer` evidence, not production Swarm Inference, not P2P, not GPU/TPU pooling, not GGUF/llama.cpp serving, and not large-model serving.

For the automated version of that external proof, use `crowdtensor real-llm-internet-beta`. Its `kaggle-auto` mode wraps `real_llm_internet_alpha_v1`, `kaggle_real_llm_live_package_v1`, and external `real_llm_live_rc_v1` evidence into one `real_llm_internet_beta_v1` report. It should only claim `real_llm_internet_beta_ready` after `external_runtime_verified` is present and both temporary private Kaggle kernels are deleted, producing `kaggle_kernels_deleted`. The default path is CPU-only; `--real-llm-backend hf_transformers_cuda` is the optional CUDA tiny GPT path where a CPU Coordinator schedules metadata-only CUDA sessions and Kaggle GPU Miners perform runtime execution. Temporary public HTTP tokens must be rotated after the run, and this remains not production Swarm Inference, not P2P, not GPU pooling, and not large-model serving.

The first real-weight live Kaggle split proof was completed with public Coordinator `24.199.118.54:9184`, two private Kaggle CPU script kernels acting as `kaggle-real-llm-stage0` and `kaggle-real-llm-stage1`, and `sshleifer/tiny-gpt2` through `hf_transformers_cpu`. The retained local evidence is `dist/real-llm-live-goal-external/real_llm_live_rc.json`; it reports `ok: true`, `external_runtime_verified`, `kaggle_real_llm_stage0_seen`, `kaggle_real_llm_stage1_seen`, `kaggle_real_llm_sharded_ready`, `real_llm_artifact_ready`, `baseline_match`, `decoded_tokens_match`, `distinct_stage_miners`, and `stage_assignment_valid`. Temporary Kaggle kernels/dataset were deleted after evidence collection. This is CPU-only, read-only `real_llm_sharded_infer` evidence for `sshleifer/tiny-gpt2`, not production Swarm Inference, not P2P, not GPU/TPU pooling, not GGUF/llama.cpp serving, and not large-model serving.

The Public Swarm Live Preview RC now has fresh stage0 and stage1 failure-requeue proofs over the public Coordinator path. The stage0 proof completed against `24.199.118.54:9196` and is retained at `dist/public-swarm-live-preview-rc-live-stage0-20260529043801-rc/public_swarm_live_preview_rc.json`; the stage1 proof completed against `24.199.118.54:9198` and is retained at `dist/public-swarm-live-preview-rc-live-stage1-20260529044328-rc/public_swarm_live_preview_rc.json`. Both reports include `ok: true`, `public_swarm_live_preview_rc_ready`, `public_swarm_live_preview_live_kaggle_ready`, `external_runtime_verified`, `external_stage_requeue_ready`, `kaggle_kernels_deleted`, `private_artifacts_cleaned`, and `token_rotation_required`; the respective reports also preserve `live_stage0_requeue_ready` or `live_stage1_requeue_ready`. The default live-preview Kaggle slug prefix is intentionally short (`ct-live-preview`) so victim/rescue suffixes fit Kaggle's 45-character kernel slug limit.

The current Public Swarm GPU Inference Beta Kaggle proof completed with public Coordinator `24.199.118.54:9320`, two private Kaggle GPU script kernels, and `sshleifer/tiny-gpt2` through `hf_transformers_cuda` with `--real-llm-partition-mode stage-local`. The retained local evidence is `dist/public-swarm-gpu-beta-stage-local-live-20260528064520-shortslug/public_swarm_gpu_inference_beta_kaggle_auto.json`; it reports `ok: true`, `public_swarm_gpu_beta_kaggle_auto_ready`, `external_gpu_runtime_verified`, `cuda_runtime_available`, `hf_transformers_cuda_ready`, `gpu_stage0_ready`, `gpu_stage1_ready`, `stage_local_partition_ready`, `stage0_partition_loaded`, `stage1_partition_loaded`, `partition_parameter_split_valid`, `stage_gpu_memory_reduced`, `activation_transport_ready`, `baseline_match`, `decoded_tokens_match`, `distinct_stage_miners`, `stage_assignment_valid`, and `kaggle_kernels_deleted`. The GPU Sharded Generation Beta proof completed with public Coordinator `24.199.118.54:9360`, two private Kaggle GPU stage kernels, `--max-new-tokens 16`, and retained evidence at `dist/gpu-sharded-generation-beta-kaggle-20260528095658/gpu_sharded_generation_beta_kaggle_auto.json`; it reports `gpu_multi_machine_generation_ready`, `multi_token_generation_ready`, `generated_token_count: 16`, `generated_text_hash`, `external_gpu_runtime_verified`, and `kaggle_kernels_deleted`. The older `dist/public-swarm-gpu-beta-live-20260528-runtimepin/public_swarm_gpu_inference_beta_kaggle_auto.json` proof remains retained as historical pre-stage-local CUDA runtime-pin evidence. The generated Kaggle runtime pins `torch==2.7.1+cu118`, `torchvision==0.22.1+cu118`, and `transformers==4.40.2`; tokens must still be rotated after every temporary public HTTP proof. These are tiny read-only CUDA split proofs, not production Swarm Inference, not P2P, not a GPU pooling marketplace, and not large-model serving.

The same high-level wrapper can run the first remote external LLM runtime proof. This creates a read-only `external_llm_infer` session, expects the remote Miner to advertise a mock, command, or OpenAI-compatible operator-owned runtime, and emits `remote_external_llm_evidence_v1` plus `remote_external_llm_observability_v1` without raw prompts, `output_text`, runtime URL, API key, lease token, or idempotency material:

```bash
crowdtensor remote-demo prepare \
  --workload external-llm \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --mock \
  --output-dir dist/remote-home-compute-llm \
  --json

. dist/remote-home-compute-llm/operator.private.env
crowdtensor remote-demo doctor \
  --workload external-llm \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" \
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \
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

`--mock` is deterministic and CI-safe. Operators can replace it with explicit `--llm-runtime-cmd` or `--llm-runtime-url` / `CROWDTENSOR_LLM_RUNTIME_URL` when they own the runtime. This remains fixed-prompt runtime evidence, not public arbitrary prompt serving.

Build a safe two-machine remote demo runbook:

```bash
crowdtensor remote-runbook \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --scenario-id route-baseline \
  --output-dir dist/remote-demo \
  --json
```

The `crowdtensor/cli.py` wrapper emits `remote_runbook_cli_v1` and delegates to `scripts/remote_demo_runbook_pack.py`. The underlying `remote_demo_runbook_v1` artifact prepares a registry-backed Coordinator/Miner demo for `model_bundle_infer`: it writes `operator.private.env` and `miner.private.env` with `0600` permissions, stores only hashed Miner token verifiers in the registry, and keeps the public JSON/Markdown free of plaintext tokens. The generated commands include security preflight, `crowdtensord --task-lane python-cli:cpu:1:model_bundle_infer`, `crowdtensor-miner`, and `remote_compute_evidence_pack.py --mode collect --scenario-id route-baseline`. The remote path uses the same fixed `model_bundle_inference_scenario_v1` IDs as `crowdtensor home-infer`. CI validates this with `scripts/remote_demo_runbook_check.py`.

After the Coordinator and remote Miner are running, collect the safe two-machine acceptance pack:

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

The `crowdtensor/cli.py` wrapper emits `remote_acceptance_cli_v1`, applies token redaction to stdout/stderr tails, and delegates to `scripts/remote_demo_acceptance_pack.py`. The recommended controlled path uses `--create-session` to call `POST /admin/inference-sessions`, queue a read-only `model_bundle_infer` task for the selected `model_bundle_inference_scenario_v1`, and wait for the returned `task_id` through the admin result ledger. The `remote_demo_acceptance_v1` report then writes `remote_compute_evidence_v1`, `support_bundle`, `remote_demo_observability_v1`, scenario match status, and a top-level JSON/Markdown summary. It also emits stable `diagnosis_codes` for operator triage, including `coordinator_unreachable`, `observer_auth_failed`, `admin_auth_failed`, `session_create_failed`, `miner_not_seen`, `task_lane_missing`, `workload_not_advertised`, `no_accepted_result`, `validation_failed`, `request_count_mismatch`, `artifact_collection_failed`, and `acceptance_ready`. This is not production Swarm Inference and not P2P routing. CI validates the local stand-in with `scripts/remote_demo_acceptance_check.py`.

Request one read-only session from an already running Coordinator:

```bash
python3 scripts/inference_session_client.py \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \
  --request-count 4 \
  --json
```

The `inference_session_client_v1` report is the narrow user-facing client for the existing `POST /admin/inference-sessions` API. It creates a CPU `model_bundle_infer` session, waits for the returned `task_id` in the admin result ledger, and emits safe latency, throughput, validation, and `session_client_ready` diagnostics. It does not accept arbitrary prompts, expose raw `inference_results`, or claim production LLM serving. Runtime acceptance covers it with `scripts/inference_session_client_check.py` and `--skip-inference-session-client`.

For optional remote and browser checks:

```bash
python3 scripts/doctor.py --remote-demo --browser --json
```

Start the Coordinator:

```bash
crowdtensord --host 127.0.0.1 --port 8787 --state-dir state
```

In another shell, run one Miner task:

```bash
crowdtensor-miner \
  --coordinator http://127.0.0.1:8787 \
  --miner-id local-1 \
  --once
```

The Miner claims a task, runs a dependency-free local training loop, uploads a DiLoCo-style delta, and exits with a JSON summary.

For the full walkthrough, see [docs/quickstart.md](docs/quickstart.md).
For user scenarios and hardware status, see [docs/use-cases.md](docs/use-cases.md).
For the current protocol boundary, see [docs/protocol.md](docs/protocol.md).
For endpoint-level integration details, see [docs/api.md](docs/api.md).
For controlled remote Miner setup, see [docs/remote-miner.md](docs/remote-miner.md).
For the project roadmap, see [ROADMAP.md](ROADMAP.md).
For durable project memory and future-agent context, see [AGENTS.md](AGENTS.md) and [docs/project-memory.md](docs/project-memory.md).
For release history and maintainer release flow, see [CHANGELOG.md](CHANGELOG.md) and [docs/release.md](docs/release.md).

## Docker Compose

Run the local demo stack:

```bash
docker compose up --build coordinator miner
```

Check the Coordinator:

```bash
curl http://127.0.0.1:8787/health
curl http://127.0.0.1:8787/version
curl http://127.0.0.1:8787/ready
curl -H 'x-crowdtensor-observer-token: local-observer' \
  http://127.0.0.1:8787/metrics
```

The Compose file uses local demo tokens by default. Copy `.env.example` to `.env` to override them.

## Core Capabilities

- **Coordinator / Miner loop**: task claim, heartbeat, result submission, and bounded long-running Miner sessions.
- **Fault tolerance**: lease timeout requeue, stale result rejection, checkpoint recovery, and append-only event replay.
- **Runtime contracts**: deterministic CPU-only `diloco_train`, `cpu_lora_mock`, `micro_transformer_lm`, `model_bundle_lm`, `model_bundle_infer`, and optional `external_llm_infer` workloads.
- **Runtime capability matrix**: `scripts/runtime_matrix.py` and `scripts/runtime_matrix_check.py` summarize local `hardware_profile`, CPU-only baseline readiness, optional browser support, and `CROWDTENSOR_LLM_RUNTIME_URL` adapter configuration.
- **Validation**: finite-value checks, shape checks, norm/loss gates, and optional deterministic replay audit.
- **Trust controls**: workload-scoped Miner scoring, quarantine, admin trust overrides, and redacted event tails.
- **Result traceability**: admin result ledger for accepted/rejected outcomes, validation, audit, model impact, and Miner score summaries.
- **Admission controls**: shared Miner token, observer token, admin token, per-Miner token registry, and hashed token configuration.
- **Remote Miner resilience**: startup `/ready` preflight, bounded retry for transient claim/heartbeat/result failures, result `idempotency_key`, and retry counters.
- **Browser experiments**: WebRTC tensor tunnel, browser Worker compute probe, and browser Miner bridge.
- **Acceptance pack**: repeatable smoke suite for runtime behavior and operator controls.

## Runtime Acceptance

Run the Alpha release gate first to verify package metadata, docs links, Docker/Compose shape, and CI wiring:

```bash
python3 scripts/release_gate.py --json
```

Run the offline security preflight before a controlled remote demo:

```bash
python3 scripts/security_preflight.py \
  --host 0.0.0.0 \
  --miner-token-registry state/miner_registry.json \
  --observer-token sha256:OBSERVER_DIGEST \
  --admin-token sha256:ADMIN_DIGEST \
  --json
```

Run the non-browser V1 acceptance pack from a normal Linux shell with localhost networking:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8910 \
  --report /tmp/crowdtensor_acceptance.json
```

It runs the core smoke checks sequentially:

- readiness/profile
- runtime capability matrix
- API contract
- chaos recovery
- trust quarantine
- replay audit
- operator control
- micro Transformer LM
- model bundle LM
- result idempotency
- result ledger
- Miner resilience
- Miner auth
- observer auth
- per-Miner registry auth
- hashed token auth
- outer optimizer contract
- compressed error-feedback delta transport
- delta transport negotiation
- read-only model bundle inference
- local inference session demo
- admin-created read-only inference session API
- optional external LLM runtime adapter contract

Browser-native checks are opt-in because they require Playwright and a Chromium-compatible browser:

```bash
python3 scripts/browser_acceptance_pack.py \
  --base-port 9310 \
  --report /tmp/crowdtensor_browser_acceptance.json
```

The browser acceptance pack runs the core browser checks: `webrtc_smoke.py`, `runtime_contract_check.py`, and `browser_miner_smoke.py`. CI uses `--allow-skip` so environments without Playwright or Chromium report a skipped browser pack instead of failing the whole job.

For the broader browser smoke set, use the runtime acceptance pack:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8930 \
  --include-browser \
  --report /tmp/crowdtensor_browser_acceptance.json
```

Generate the release evidence bundle after the acceptance reports exist:

```bash
python3 scripts/release_evidence_pack.py \
  --runtime-report /tmp/crowdtensor_acceptance.json \
  --browser-report /tmp/crowdtensor_browser_acceptance.json \
  --json-out dist/release-evidence.json \
  --markdown-out dist/release-evidence.md
```

The Release Evidence output records the git commit, package metadata, release gate result, security preflight result, and acceptance report summaries. Runtime acceptance summaries preserve safe per-check `summary_json` fields plus top-level `diagnosis_summary` / `diagnosis_by_check` rows, and remote reports preserve safe `observability_summaries` such as `remote_compute_observability_v1` and `remote_demo_observability_v1`, so release artifacts show stable triage and remote-demo observability without raw tokens or tensor payloads. CI uploads `release-evidence.json` and the Markdown companion as build artifacts.

Build a Support Bundle for issues or remote-demo troubleshooting:

```bash
python3 scripts/support_bundle.py \
  --json-out /tmp/crowdtensor_support_bundle.json
```

The Support Bundle includes doctor and release-gate summaries, optional acceptance report summaries, runtime `diagnosis_summary` / `diagnosis_by_check` rows, safe remote `observability_summaries`, and safe online Coordinator summaries when `--coordinator` is provided. It redacts token, lease, idempotency, weight, and delta-shaped fields before writing output.

Some sandboxes block localhost client sockets. In that case, run unit tests inside the sandbox and run the acceptance pack in an unrestricted shell or CI job.

Run only the readiness/profile smoke:

```bash
python3 scripts/readiness_check.py --port 8890
```

Run only the API contract smoke:

```bash
python3 scripts/api_contract_check.py --port 8891
```

Run only the Miner resilience smoke:

```bash
python3 scripts/miner_resilience_check.py --port 8894
```

Run only the result idempotency smoke:

```bash
python3 scripts/result_idempotency_check.py --port 8896
```

Run only the result ledger smoke:

```bash
python3 scripts/result_ledger_check.py --port 8897
```

Run only the opt-in Nesterov outer optimizer smoke:

```bash
python3 scripts/outer_optimizer_check.py --port 8899
```

Run only the sign-compressed error-feedback transport smoke:

```bash
python3 scripts/compressed_error_feedback_check.py --port 8900
```

Run only the delta transport negotiation smoke:

```bash
python3 scripts/delta_transport_negotiation_check.py --port 8901
```

Run only the model bundle LM smoke:

```bash
python3 scripts/model_bundle_smoke.py --port 8902
```

Run only the read-only model bundle inference smoke:

```bash
python3 scripts/model_bundle_inference_smoke.py --port 8903
```

The inference smoke runs a read-only multi-request session by default; use `--request-count N` to change the number of prompts in the task.
It reports safe session metrics such as `elapsed_ms`, `requests_per_second`, `request_count`, accuracy, a capped `request_trace`, and the Python Miner `hardware_profile` so users can inspect the CPU baseline without treating it as a real LLM or GPU benchmark.

Run the user-facing local inference session demo:

```bash
python3 scripts/inference_session_demo.py --port 8904 --request-count 4
```

Use `--json` when you need a machine-readable report for CI or issue reports.

Run the admin-created read-only inference session API check:

```bash
python3 scripts/admin_inference_session_check.py --port 8915 --request-count 4
```

This validates `POST /admin/inference-sessions`, which returns `schema=inference_session_request_v1`, queues a CPU-only `model_bundle_infer` task, and points operators at `GET /admin/results?task_id=<task_id>&workload_type=model_bundle_infer`. The result is read-only and safe for operator inspection: model versions do not advance, raw `inference_results`, lease tokens, and idempotency material stay out of the admin ledger. The runtime acceptance pack includes this check by default and can omit it with `--skip-admin-inference-session`.

Run the matrix-guided home-compute demo:

```bash
python3 scripts/home_compute_demo.py --port 8909 --request-count 4 --json
```

The home-compute demo first checks `scripts/runtime_matrix.py`, selects the CPU-only `model_bundle_infer` workload and `local_cpu_model_bundle_infer` route when available, runs `scripts/inference_session_demo.py`, and emits one report with runtime capability, `route_decision`, session metrics, capped `request_trace` rows, read-only status, redaction status, `hardware_targets`, `recommended_routes`, `matched_capabilities`, `missing_capabilities`, stable `diagnosis_codes` such as `home_compute_ready`, `runtime_matrix_blocked`, `workload_unavailable`, `cpu_route_unavailable`, `session_failed`, `trace_missing`, and recommended next commands. CI validates this path with `scripts/home_compute_demo_check.py`; the runtime acceptance pack includes it by default and can skip it with `--skip-home-compute-demo`.

For a safe, shareable artifact, run `scripts/home_compute_evidence_pack.py --port 8911 --request-count 4 --json-out /tmp/crowdtensor_home_evidence.json --markdown-out /tmp/crowdtensor_home_evidence.md`. The `home_compute_evidence_v1` evidence pack preserves the route, metrics, capped trace, and `diagnosis_codes` while redacting secret-shaped fields; CI validates it with `scripts/home_compute_evidence_check.py`, and the runtime acceptance pack can skip it with `--skip-home-compute-evidence`.

Run only the optional external LLM adapter smoke:

```bash
python3 scripts/external_llm_inference_smoke.py --port 8906 --request-count 3
```

Run the OpenAI-compatible HTTP adapter variant:

```bash
python3 scripts/external_llm_http_adapter_smoke.py --port 8907 --runtime-port 8908
```

The `external_llm_infer` workload uses the `external_llm_infer_v1` schema. It is read-only and validates `external_llm_results` against claim-time prompt hashes before recording safe `request_count`, `completion_count`, `output_chars`, `adapter_kind`, and `model_id` summaries. The smoke path uses `crowdtensor-miner --enable-mock-llm-runtime` for deterministic CI. Operators can opt into a local command adapter with `--llm-runtime-cmd` or `CROWDTENSOR_LLM_RUNTIME_CMD`; the command receives `prompt` and `max_tokens` arguments. Operators can also opt into an OpenAI-compatible chat completions endpoint with `--llm-runtime-url` or `CROWDTENSOR_LLM_RUNTIME_URL`, plus optional `--llm-runtime-api-key` / `CROWDTENSOR_LLM_RUNTIME_API_KEY`. Runtime URLs and API keys are never advertised in Miner capabilities. Raw prompts and `output_text` are kept out of `/state` and admin ledger summaries.

Build a safe external LLM evidence artifact:

```bash
python3 scripts/external_llm_evidence_pack.py \
  --mock \
  --port 8919 \
  --request-count 3 \
  --json-out /tmp/crowdtensor_external_llm_evidence.json \
  --markdown-out /tmp/crowdtensor_external_llm_evidence.md
```

Validate the default mock path with `scripts/external_llm_evidence_check.py`; runtime acceptance includes it by default and can skip it with `--skip-external-llm-evidence`.

For remote Miners, `scripts/remote_external_llm_evidence_pack.py` collects the same safe proof layer from a running Coordinator. `crowdtensor remote-demo verify --workload external-llm --mock` wraps that path and records `remote_python_external_llm_infer`, `remote_external_llm_evidence_v1`, and `remote_external_llm_observability_v1` while preserving the non-production and non-public-serving boundary.

Run only the remote Miner invite/join smoke:

```bash
python3 scripts/remote_miner_join_check.py --port 8898
```

Run only the remote-compute evidence smoke:

```bash
python3 scripts/remote_compute_evidence_check.py --port 8912
```

Run only the safe two-machine runbook generator check:

```bash
python3 scripts/remote_demo_runbook_check.py
```

Run only the safe two-machine acceptance check:

```bash
python3 scripts/remote_demo_acceptance_check.py --port 8913
```

Run only the security preflight:

```bash
python3 scripts/security_preflight.py --json
```

## Security Model

CrowdTensorD has local-development admission controls, not a complete public network security model.

Coordinator supports:

- `--miner-token` / `CROWDTENSOR_MINER_TOKEN`
- `--miner-token-registry` / `CROWDTENSOR_MINER_TOKEN_REGISTRY`
- `--observer-token` / `CROWDTENSOR_OBSERVER_TOKEN`
- `--admin-token` / `CROWDTENSOR_ADMIN_TOKEN`

Token config values may be plaintext for local demos or `sha256:<digest>` verifiers for remote demos.

Miner startup checks `/ready` by default. Use `--skip-preflight` only for legacy Coordinators. Transient claim, heartbeat, and idempotent result upload failures are retried with `--max-request-attempts`; summaries include `request_retries`.

See [docs/security.md](docs/security.md) before exposing a Coordinator beyond localhost.

## Documentation

- [Quickstart](docs/quickstart.md)
- [API Reference](docs/api.md)
- [Protocol Boundary](docs/protocol.md)
- [Remote Miner Onboarding](docs/remote-miner.md)
- [Use Cases](docs/use-cases.md)
- [Architecture](docs/architecture.md)
- [Security](docs/security.md)
- [Operations](docs/operations.md)
- [Release Process](docs/release.md)
- [Changelog](CHANGELOG.md)
- [Roadmap](ROADMAP.md)
- [Project Memory](docs/project-memory.md)
- [Agent Instructions](AGENTS.md)
- [Static Site](site/index.html)

## License

Apache-2.0. See [LICENSE](LICENSE).
