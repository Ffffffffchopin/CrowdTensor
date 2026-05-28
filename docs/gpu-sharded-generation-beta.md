# GPU Sharded Generation Beta

`crowdtensor gpu-generate` is the current GPU multi-machine generation milestone. It wraps the CUDA Public Swarm GPU path and emits `gpu_sharded_generation_beta_v1` through `scripts/gpu_sharded_generation_beta_pack.py`.

This path proves that two distinct stage Miners can alternate stage 0 and stage 1 work for multiple generated tokens with `sshleifer/tiny-gpt2`, `hf_transformers_cuda`, and `--real-llm-partition-mode stage-local`. Public evidence records the generated text hash and generated token count, but never raw generated text, generated token ids, prompts, activations, logits, lease material, registry tokens, or private env files.

## Commands

CI-safe contract check:

```bash
python3 scripts/gpu_sharded_generation_beta_check.py \
  --include-wrapper-check \
  --max-new-tokens 4 \
  --json
```

Import retained evidence:

```bash
crowdtensor gpu-generate evidence-import \
  --gpu-report dist/gpu-sharded-generation-beta-kaggle-20260528095658/gpu_sharded_generation_beta_kaggle_auto.json \
  --max-new-tokens 16 \
  --json
```

Side-effectful Kaggle GPU run:

```bash
crowdtensor gpu-generate kaggle-auto \
  --public-host 24.199.118.54 \
  --port 9360 \
  --base-port 9361 \
  --max-new-tokens 16 \
  --request-count 1 \
  --kaggle-owner YOUR_KAGGLE_USERNAME \
  --kernel-slug-prefix ctgg95658 \
  --timeout-seconds 1500 \
  --remote-timeout-seconds 1200 \
  --kaggle-status-timeout-seconds 1800 \
  --json
```

Kaggle mode starts a temporary public Coordinator, creates private Kaggle GPU stage kernels, waits for both stages to complete the requested generation chain, deletes the temporary kernels, and marks `token_rotation_required`. Rotate generated tokens after every temporary public HTTP proof.

## Ready Evidence

A ready report should include:

- `gpu_sharded_generation_ready`
- `gpu_multi_machine_generation_ready` for `kaggle-auto` or `gpu_loopback_generation_ready` for `local-loopback`
- `multi_token_generation_ready`
- `generated_token_count >= max_new_tokens`
- non-empty `generated_text_hash`
- `raw_generated_text_public: false`
- `hf_transformers_cuda`
- `stage_local_partition_ready`
- `stage0_partition_loaded`
- `stage1_partition_loaded`
- `partition_parameter_split_valid`
- `decoded_tokens_match`
- `distinct_stage_miners`
- `stage_assignment_valid`
- `kaggle_kernels_deleted` for `kaggle-auto`

The retained successful Kaggle GPU generation proof is:

```text
dist/gpu-sharded-generation-beta-kaggle-20260528095658/gpu_sharded_generation_beta_kaggle_auto.json
```

It records 16 generated tokens and `generated_text_hash=sha256:10a23cc1ba2187388dfdc48a038d1bb5cb21f6d7c6dd0674fe79cd3d912723b2` with private env and registry files removed from the retained artifact tree.

## Boundaries

This is a tiny GPT CUDA multi-token Beta proof. It is not production Swarm Inference, not Hivemind-level serving, not arbitrary prompt serving, not P2P routing, not NAT traversal, not a GPU marketplace, and not large-model serving.
