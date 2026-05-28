# Release Process

This document defines the Alpha release candidate checklist. It is for maintainers preparing a public GitHub release or tag.

## Preflight

Start from a clean worktree on `main`:

```bash
git status --short --branch
```

Confirm package metadata and public docs:

```bash
python3 scripts/release_gate.py --json
python3 scripts/doctor.py --json
python3 scripts/security_preflight.py --json
```

Verify the documented fresh-clone onboarding path from a clean virtualenv:

```bash
python3 scripts/onboarding_gate.py --quick --json-out /tmp/crowdtensor_onboarding_gate.json
```

The `onboarding_gate_v1` report runs `python3 -m venv`, installs with `python -m pip install -e .[dev]`, checks `crowdtensor --help`, `crowdtensord --help`, and `crowdtensor-miner --help`, then smoke-validates `crowdtensor local-proof`, `crowdtensor home-infer`, `crowdtensor llm-infer --mock`, `crowdtensor cpu-infer --mode local`, and `crowdtensor release-ready --allow-dirty`. This is an Alpha repository onboarding gate, not production Swarm Inference readiness.

Build the maintainer readiness report before running longer acceptance:

```bash
crowdtensor release-ready --json
```

The command emits `release_readiness_v1` by wrapping `scripts/release_readiness_pack.py`. It combines Git metadata, the release gate, security preflight, and `demo_manifest_v1`, then blocks on diagnosis such as `git_dirty`, `release_gate_failed`, or `demo_manifest_failed`. Dirty worktrees block by default; `scripts/release_readiness_check.py --allow-dirty` is reserved for CI/development smoke validation and does not mean a dirty tree is tag-ready. The report is not production Swarm Inference readiness; it is an Alpha repository release gate for maintainers.

## Verification

Run compile and unit checks:

```bash
python3 -m py_compile coordinator.py miner_cli.py scripts/*.py crowdtensor/*.py
python3 -m unittest discover -s tests -v
```

Validate the Kaggle Real Runtime Acceptance artifact path before any live Notebook run:

```bash
python3 scripts/kaggle_real_runtime_acceptance_check.py --port 9180
python3 scripts/kaggle_real_runtime_acceptance_check.py --port 9181 --workload micro-llm-sharded --stage-mode split --request-count 2 --decode-steps 3
```

This checks `kaggle_real_runtime_acceptance_v1` preparation, temporary HTTP boundary text for `24.199.118.54`, `operator.private.env` exclusion from Kaggle, `miner.private.env` upload packaging, micro split `kaggle-upload-stage0` / `kaggle-upload-stage1` artifacts, `token_rotation_required`, and redaction. It is not live Kaggle proof; a real Notebook acceptance must be verified separately with `crowdtensor remote-demo kaggle-real --action verify` and `kaggle_real_runtime_ready`. The micro path should additionally report `kaggle_micro_llm_stage0_seen`, `kaggle_micro_llm_stage1_seen`, `kaggle_micro_llm_stage_assignment_valid`, `stage_assignment_valid`, and `kaggle_micro_llm_sharded_ready` after two live Notebooks finish. For Kaggle CLI-driven live proofs, `scripts/kaggle_micro_llm_live_package.py` builds private dataset/script-kernel upload folders; `--inline-kernel-payload` is a private fallback only and embeds Miner env material into kernel source, so generated kernels/datasets must be deleted and tokens rotated after evidence collection. The retained artifact-backed live proof is `dist/kaggle-micro-llm-live/external-real/kaggle_real_runtime_acceptance.json`. This is a toy two-stage pipeline, not large-model sharding.

Validate the file-backed micro-LLM artifact path before using it in a release candidate:

```bash
python3 scripts/micro_llm_artifact_check.py \
  --base-port 9182 \
  --request-count 2 \
  --decode-steps 3 \
  --json-out dist/micro-llm-artifact-local/micro_llm_artifact_check.json
```

The `micro_llm_artifact_check_v1` report builds `micro_llm_artifact_v1`, runs artifact-backed `micro_llm_sharded_infer`, and should report `micro_llm_artifact_ready`, `artifact_loaded`, `micro_llm_sharded_ready`, `stage_0_accepted`, `stage_1_accepted`, `baseline_match`, `decoded_tokens_match`, `activation_transport_ready`, `distinct_stage_miners`, and `stage_assignment_valid`. For the higher-level local-generated RC stand-in, run `crowdtensor micro-llm-live-rc --mode local-generated --micro-llm-artifact dist/micro-llm-artifact-local/artifact --request-count 2 --decode-steps 3 --json`; a ready report should include `micro_llm_live_rc_ready` without claiming live Kaggle unless `--mode external-existing` verifies already running external Miners.

Validate the Real Small-LLM Sharded Inference Live RC after installing optional HF dependencies:

```bash
python -m pip install -e '.[hf]'
python3 scripts/real_llm_live_rc_check.py \
  --base-port 9184 \
  --request-count 1 \
  --timeout-seconds 300
```

The `real_llm_live_rc_v1` report comes from `scripts/real_llm_live_rc_pack.py` or `crowdtensor real-llm-live-rc`. `local-generated` must create `kaggle-upload-real-llm-stage0` and `kaggle-upload-real-llm-stage1`, run two HF-enabled local stand-ins, and report `local_generated_real_llm_stage_upload_standins_ready`, `remote_real_llm_sharded_ready`, `real_llm_artifact_ready`, `stage_assignment_valid`, and `real_llm_live_rc_ready` without claiming `external_runtime_verified`. `kaggle-generated` is preparation only. `external-existing` may report `external_runtime_verified` and `kaggle_real_llm_sharded_ready` only after an already running public Coordinator plus two external stage Miners complete. `scripts/kaggle_real_llm_live_package.py` emits `kaggle_real_llm_live_package_v1` for private Kaggle dataset/script-kernel packaging; use `--inline-kernel-payload` only as a temporary private fallback when Kaggle input mounting fails, then delete kernels/dataset and rotate tokens. The first live real-weight Kaggle proof is retained at `dist/real-llm-live-goal-external/real_llm_live_rc.json` and reports `external_runtime_verified`, `kaggle_real_llm_stage0_seen`, `kaggle_real_llm_stage1_seen`, `kaggle_real_llm_sharded_ready`, `real_llm_artifact_ready`, baseline match, decoded-token match, distinct stage Miners, and valid stage assignment. Generated launchers use `--enable-hf-tiny-gpt-runtime` and `--real-llm-stage-role`; this is CPU-only, read-only, not production Swarm Inference, not P2P, and not large-model serving.

Validate the Real Internet Swarm Inference Alpha after the Live RC check:

```bash
python3 scripts/real_llm_internet_alpha_check.py \
  --port 9187 \
  --base-port 9188 \
  --request-count 1 \
  --timeout-seconds 300
```

The `real_llm_internet_alpha_v1` report comes from `scripts/real_llm_internet_alpha_pack.py` or `crowdtensor real-llm-internet-alpha`. `local-generated` must aggregate `real_llm_live_rc_v1` plus stage0/stage1 requeue checks and report `real_llm_internet_alpha_ready`, `real_llm_stage_requeue_ready`, `stage_requeue_ready`, `real_llm_live_rc_ready`, `remote_real_llm_sharded_ready`, `real_llm_artifact_ready`, `activation_transport_ready`, `decoded_tokens_match`, `distinct_stage_miners`, and `stage_assignment_valid` while keeping `external_runtime_verified` false. `package` prepares the public Coordinator and stage upload artifacts only. `external-existing` may report `external_runtime_verified` only after an already running public Coordinator plus two external stage Miners complete. Reports must preserve `token_rotation_required`, CPU-only/read-only semantics, and the not production Swarm Inference, not P2P, and not large-model boundaries.

Validate the Real Internet Swarm Inference Beta contract after the Alpha check:

```bash
python3 scripts/real_llm_internet_beta_check.py \
  --port 9190 \
  --base-port 9191 \
  --request-count 2 \
  --timeout-seconds 300
```

The `real_llm_internet_beta_v1` report comes from `scripts/real_llm_internet_beta_pack.py` or `crowdtensor real-llm-internet-beta`. The side-effectful `kaggle-auto` mode generates the Alpha package, starts the temporary public Coordinator, pushes private Kaggle CPU script kernels, verifies `external_runtime_verified`, deletes the temporary kernels, stops the Coordinator, and only then may report `real_llm_internet_beta_ready` with `kaggle_kernels_deleted`. With `--failure-mode kill-stage0-after-claim` or `kill-stage1-after-claim`, it must report true external victim/rescue requeue evidence: `external_stage_requeue_ready`, `live_stage0_requeue_ready` / `live_stage1_requeue_ready`, and `live_requeue_summary`. `scripts/real_llm_internet_beta_check.py` is CI-safe and uses a fake runner; it must not create Kaggle resources during release checks. Preserve `real_llm_internet_alpha_ready`, `token_rotation_required`, CPU-only/read-only semantics, and the not production Swarm Inference, not P2P, and not large-model boundaries.

Validate the user-facing Swarm Inference Beta contract:

```bash
python scripts/swarm_inference_beta_check.py --json
```

The `swarm_inference_beta_v1` report comes from `scripts/swarm_inference_beta_pack.py` or `crowdtensor swarm-infer-beta`. The side-effectful `swarm-infer-beta live` path wraps `real_llm_internet_beta_v1` in `kaggle-auto` mode, starts a temporary public Coordinator, pushes private Kaggle CPU stage kernels, verifies `external_runtime_verified`, optionally verifies external victim/rescue requeue with `--failure-mode`, deletes the kernels, writes `support_bundle.json`, removes local live private artifacts and raw runtime state by default, and only then may report `swarm_inference_beta_live_ready`, `swarm_inference_beta_ready`, `two_machine_swarm_inference_ready`, `real_llm_internet_beta_ready`, `external_stage_requeue_ready` when requested, `live_requeue_summary`, `kaggle_kernels_deleted`, `swarm_inference_beta_live_private_artifacts_cleaned`, and `token_rotation_required`. `--keep-live-private-artifacts` is for debugging only and must not be used for shareable release evidence. The manual operator path packages `stage0` and `stage1` real tiny GPT join packs, `operator.private.env`, `miner.private.env`, a hashed `miner_registry.json`, and verifies through `remote_real_llm_sharded_beta_v1`. A ready report must preserve `real_llm_split_route_ready`, `external_beta_evidence_imported` when a retained `real_llm_internet_beta_v1` report is imported, `decoded_tokens_match`, `distinct_stage_miners`, and `stage_assignment_valid`. It remains CPU-only, read-only, not production Swarm Inference, not P2P, and not large-model serving.

Public Swarm Inference Alpha is the user-facing session wrapper for the same controlled tiny GPT split evidence:

```bash
python scripts/public_swarm_inference_alpha_check.py --json
```

The `public_swarm_inference_alpha_v1` report comes from `scripts/public_swarm_inference_alpha_pack.py` or `crowdtensor swarm-session`. `live-kaggle` aggregates the cleanup-backed `swarm-infer-beta live` proof, true external victim/rescue requeue evidence (`external_stage_requeue_ready`, `live_stage0_requeue_ready` / `live_stage1_requeue_ready`, `live_requeue_summary`) when `--failure-mode` is enabled, and mandatory `local-generated` stage requeue evidence. It must report `public_swarm_inference_alpha_ready`, `public_swarm_session_ready`, `local_stage_requeue_ready`, `public_swarm_live_requeue_ready`, `public_swarm_live_kaggle_ready`, `stage_requeue_ready`, `external_runtime_verified`, `kaggle_kernels_deleted`, and `token_rotation_required`. Child debug artifacts are pruned by default so shareable evidence should retain only the top-level public JSON/Markdown report; `--keep-child-artifacts` is for local debugging only. This release check is CI-safe and does not create Kaggle resources. It remains CPU-only, read-only, not production Swarm Inference, not P2P, and not large-model serving.

Validate the Public Swarm Inference Alpha RC layer before publishing Alpha evidence:

```bash
crowdtensor public-swarm-alpha-rc --mode evidence-import --json
python scripts/public_swarm_inference_alpha_rc_check.py --mode local-smoke
```

The `public_swarm_inference_alpha_rc_v1` report comes from `scripts/public_swarm_inference_alpha_rc_pack.py` or `crowdtensor public-swarm-alpha-rc`. `evidence-import` imports the retained live stage0 and stage1 public reports plus `dist/public-swarm-inference-alpha-live-requeue-summary.json`, then requires `stage0_live_requeue_evidence_ready`, `stage1_live_requeue_evidence_ready`, `public_swarm_live_requeue_evidence_ready`, `public_swarm_alpha_rc_evidence_imported`, `public_swarm_alpha_private_artifacts_absent`, and `public_swarm_inference_alpha_rc_ready`. The retained report paths are `dist/public-swarm-inference-alpha-live-stage0-requeue-20260527165830/public_swarm_inference_alpha.json` and `dist/public-swarm-inference-alpha-live-stage1-requeue-20260527170600/public_swarm_inference_alpha.json`. `local-smoke` is CI-safe and does not create Kaggle resources. The RC remains CPU-only, read-only, not production Swarm Inference, not P2P, and not large-model serving.

Validate the Public Swarm Inference Beta user entrypoint:

```bash
crowdtensor public-swarm-beta product-beta --json
python scripts/public_swarm_inference_beta_check.py --mode product-beta --json

# Compatibility checks for the legacy CPU split route:
crowdtensor public-swarm-beta local-loopback --base-port 9290 --request-count 1 --json
crowdtensor public-swarm-beta evidence-import --json
python scripts/public_swarm_inference_beta_check.py --mode local-loopback --base-port 9290 --request-count 1
```

The `public_swarm_inference_beta_v1` report comes from `scripts/public_swarm_inference_beta_pack.py` or `crowdtensor public-swarm-beta`. `public-swarm-beta product-beta` is the product-shaped aggregate and must preserve `public_swarm_product_beta_ready`, `public_swarm_product_rc_ready`, `coordinator_product_surface_ready`, `session_protocol_ready`, `p2p_lite_discovery_ready`, `gpu_generation_evidence_import_ready`, `cpu_fallback_ready`, and `local_cpu_inference_ready`. It wraps the Product RC (`serve`, `join`, `generate`, `peer`), `session_protocol_v1`, `p2p_lite_peer_v1`, retained `gpu_sharded_generation_beta_v1` evidence, and the CPU inference fallback. `public-swarm-beta local-loopback` still wraps the two-stage real tiny GPT split path and must preserve `two_stage_split_inference_ready`, `local_loopback_ready`, `decoded_tokens_match`, `distinct_stage_miners`, and `stage_assignment_valid`. `public-swarm-beta evidence-import` still imports retained Alpha RC evidence and must preserve `public_swarm_beta_evidence_import_ready`, `external_live_evidence_imported`, `stage0_live_requeue_evidence_ready`, and `stage1_live_requeue_evidence_ready`. The operator path covers `prepare`, `coordinator`, `miner`, `verify`, `collect`, and dry-run `clean`. It remains Coordinator-backed and read-only, not production Swarm Inference, not libp2p/DHT/NAT traversal, not Hivemind-level serving, and not large-model serving.

Validate the Public Swarm Inference Beta RC before publishing the product path:

```bash
crowdtensor public-swarm-beta-rc local-loopback --base-port 9310 --max-new-tokens 2 --json
crowdtensor public-swarm-beta-rc package --target kaggle --json
crowdtensor public-swarm-beta-rc external-existing --coordinator-url https://YOUR_COORDINATOR_HOST --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" --admin-token "$CROWDTENSOR_ADMIN_TOKEN" --json
python scripts/public_swarm_inference_beta_rc_check.py --mode local-loopback --json
python scripts/public_swarm_inference_beta_rc_check.py --mode package --target kaggle --json
python scripts/public_swarm_inference_beta_rc_check.py --mode external-existing --json
```

The `public_swarm_inference_beta_rc_v1` report comes from `scripts/public_swarm_inference_beta_rc_pack.py` or `crowdtensor public-swarm-beta-rc`; CI-safe validation is in `scripts/public_swarm_inference_beta_rc_check.py`. It must preserve `public_swarm_inference_beta_rc_ready`, `public_swarm_product_beta_ready`, `p2p_lite_route_ready`, `p2p_lite_discovery_ready`, `cpu_fallback_ready`, `serve_join_generate_loop_ready`, `remote_generate_session_ready`, and `public_swarm_generate_ready`. `public-swarm-beta-rc package` must preserve `private_artifacts_local_only` and `miner_join_pack_ready`; `public-swarm-beta-rc external-existing` may report `external_runtime_verified` only for a live controlled runtime. A release host without optional `[hf]` dependencies should report `hf_dependencies_missing` instead of claiming local runtime readiness. The RC is CPU-only by default, read-only, not production Swarm Inference, not libp2p, not DHT, not NAT traversal, and not large-model serving.

Validate the Public Swarm GPU Inference Beta smoke path on every release host, and run the CUDA loopback only when CUDA is actually available:

```bash
crowdtensor public-swarm-gpu-beta local-smoke --json
python scripts/public_swarm_gpu_inference_beta_check.py --mode local-smoke
python scripts/public_swarm_gpu_inference_beta_check.py --mode kaggle-auto

# Optional on a CUDA-capable host with [hf] dependencies installed:
crowdtensor public-swarm-gpu-beta local-loopback --base-port 9321 --request-count 1 --json
crowdtensor public-swarm-gpu-beta kaggle-package --output-dir dist/public-swarm-gpu-beta-kaggle --json
crowdtensor public-swarm-gpu-beta kaggle-auto --public-host 24.199.118.54 --port 9320 --base-port 9321 --kaggle-owner YOUR_KAGGLE_USERNAME --request-count 1 --json
crowdtensor public-swarm-gpu-beta evidence-import --gpu-report dist/public-swarm-gpu-beta/public_swarm_gpu_inference_beta.json --json
```

The `public_swarm_gpu_inference_beta_v1` report comes from `scripts/public_swarm_gpu_inference_beta_pack.py` or `crowdtensor public-swarm-gpu-beta`. `public-swarm-gpu-beta local-smoke` must preserve `public_swarm_gpu_beta_smoke_ready` without claiming a usable GPU route on CPU-only hosts. `public-swarm-gpu-beta local-loopback` selects `hf_transformers_cuda`, requires `cuda_runtime_available`, `hf_transformers_cuda_ready`, `gpu_runtime_ready`, `gpu_stage0_ready`, `gpu_stage1_ready`, `stage_local_partition_ready`, `stage0_partition_loaded`, `stage1_partition_loaded`, `partition_parameter_split_valid`, and only accepts Miners advertising `real_llm_sharded_cuda_stage0`, `real_llm_sharded_cuda_stage1`, or `real_llm_sharded_cuda_both`. A completed GPU proof may be imported with `public-swarm-gpu-beta evidence-import`, which must report `external_gpu_runtime_verified`; private Kaggle GPU templates are generated by `public-swarm-gpu-beta kaggle-package` and should report `kaggle_gpu_package_ready`. `public-swarm-gpu-beta kaggle-auto` is side-effectful: it uses a CPU-capable public Coordinator with CUDA metadata-only session creation, pushes private Kaggle GPU stage kernels, verifies external CUDA stage Miners, deletes kernels, and only then may report `public_swarm_gpu_beta_kaggle_auto_ready`, `kaggle_kernels_deleted`, and `token_rotation_required`; `scripts/public_swarm_gpu_inference_beta_check.py --mode kaggle-auto` is fake-runner only. The retained successful stage-local Kaggle GPU proof is `dist/public-swarm-gpu-beta-stage-local-live-20260528064520-shortslug/public_swarm_gpu_inference_beta_kaggle_auto.json`; the older `dist/public-swarm-gpu-beta-live-20260528-runtimepin/public_swarm_gpu_inference_beta_kaggle_auto.json` proof is retained historical pre-stage-local evidence. Generated Kaggle CUDA kernels pin `torch==2.7.1+cu118`, `torchvision==0.22.1+cu118`, and `transformers==4.40.2`. GPU Beta defaults to `--real-llm-partition-mode stage-local`, which proves stage-owned module placement and parameter-count evidence for the tiny GPT route; it is read-only optional CUDA tiny GPT evidence, not production Swarm Inference, not P2P, not a GPU pooling marketplace, and not large-model serving.

Run the non-browser runtime acceptance pack from a normal Linux shell with localhost networking:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8910 \
  --report /tmp/crowdtensor_acceptance.json
```

Run browser acceptance when Playwright and Chromium are available:

```bash
python3 scripts/browser_acceptance_pack.py \
  --allow-skip \
  --base-port 9310 \
  --report /tmp/crowdtensor_browser_acceptance.json
```

## Evidence

Generate release evidence after acceptance reports exist:

```bash
python3 scripts/release_evidence_pack.py \
  --runtime-report /tmp/crowdtensor_acceptance.json \
  --browser-report /tmp/crowdtensor_browser_acceptance.json \
  --json-out dist/release-evidence.json \
  --markdown-out dist/release-evidence.md
```

The release evidence JSON preserves safe runtime acceptance `summary_json` rows, aggregates `diagnosis_summary` / `diagnosis_by_check`, and carries remote `observability_summaries` such as `remote_compute_observability_v1` and `remote_demo_observability_v1` so reviewers can see stable operator triage and remote-demo observability beside the pass/fail state.

Generate a Support Bundle for troubleshooting the candidate:

```bash
python3 scripts/support_bundle.py \
  --runtime-report /tmp/crowdtensor_acceptance.json \
  --browser-report /tmp/crowdtensor_browser_acceptance.json \
  --release-evidence dist/release-evidence.json \
  --json-out /tmp/crowdtensor_support_bundle.json \
  --markdown-out /tmp/crowdtensor_support_bundle.md
```

The Support Bundle includes acceptance `diagnosis_summary` / `diagnosis_by_check` and safe remote `observability_summaries` when reports are provided, while still redacting token, lease, idempotency, weight, and delta-shaped fields.

Generate a local-loopback Demo Manifest when reviewers need one latest output artifact for this checkout:

```bash
python3 scripts/demo_manifest_pack.py \
  --output-dir dist/demo-manifest \
  --port 8914 \
  --request-count 4
```

The `demo_manifest_v1` JSON/Markdown indexes `runtime_matrix.json`, `remote_compute_evidence_v1`, `external_llm_evidence_v1`, `support_bundle`, and `remote_compute_observability_v1` summaries without expanding the project scope beyond the CPU-only local-loopback demo. The external LLM evidence section uses deterministic mock evidence by default and does not claim public prompt serving.

## Publish

Before tagging, update [CHANGELOG.md](../CHANGELOG.md) with the release version, verification status, known limitations, and user-facing changes.

Create the tag only after checks pass:

```bash
git tag -a v0.1.0a0 -m "CrowdTensorD v0.1.0a0"
git push origin main v0.1.0a0
```

Use `.github/release.yml` to categorize GitHub release notes. Attach `dist/release-evidence.json` and `dist/release-evidence.md` to the release when available.
