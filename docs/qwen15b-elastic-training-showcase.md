# Qwen2.5-1.5B Elastic Training Showcase

CrowdTensor completed a real LoRA causal-language-model adaptation run on
2026-07-19. The run used Kaggle logical multi-node execution: two concurrent
T4x2 Kernels trained the first segment, both Kernels were deleted, and two new
T4x2 Kernels restored central checkpoints and completed the second segment.

This is a continued-pretraining-style WikiText experiment. It is not
instruction tuning and it does not establish improvement on general reasoning,
chat, safety, or downstream task benchmarks.

## Fixed Inputs

- Base model: `Qwen/Qwen2.5-1.5B`
- Model revision: `8faed761d45a263340a0528343f099c05c9a4323`
- Parameter count: `1,543,714,304`
- Model license: Apache-2.0
- Dataset: `Salesforce/wikitext`, config `wikitext-2-raw-v1`
- Dataset revision: `b08601e04326c79dfdd32d625aee71d232d685c3`
- Dataset licenses reported by the source: CC-BY-SA-3.0 and GFDL
- Training: 256 optimizer steps, four microbatches per step, 128 tokens per
  sequence, 131,072 training tokens
- Adapter: LoRA rank 4, alpha 8, learning rate `5e-4`
- Held-out evaluation: 64 pinned validation sequences

## Result

| Metric | Frozen base | Exported adapter | Change |
| --- | ---: | ---: | ---: |
| Validation loss | 2.731937 | 2.350524 | -13.96% |
| Validation perplexity | 15.3626 | 10.4911 | -31.71% |

The standard PEFT adapter contains 392 tensors covering all 28 transformer
layers. A fresh evaluation path loaded it on CPU and CUDA, changed logits, and
reproduced the held-out improvement before accepting the run.

## Elastic Evidence

- Steps 1-128 completed exactly once on the first Kernel pair.
- Both first-generation Kernels were deleted before replacement launch.
- The Coordinator observed a bounded zero-Miner pause.
- Two new Kernel references and Miner sessions restored all four stage
  checkpoints from step 128 without old local disks.
- Steps 129-256 completed exactly once on the replacement pair.
- All four temporary Kernels, the Coordinator, tunnel, private payloads, and
  temporary packages were removed after export.

The topology is accurately described as Kaggle logical multi-node. It is not
evidence of independent physical hosts or production WAN performance.

## Artifacts

- Public-safe report:
  `dist/training-showcase-20260718-final-r1/training_qwen15b_showcase.json`
- Report SHA-256:
  `c51bdfa3fede4f5c8226835e3679795e5daf75befb4baea616928cefa1516c36`
- Report content hash:
  `sha256:8b70b346b96ff3177e489e5357ef78be6bfbb245c5f1ee7d8152fbb9a7b3abc5`
- Learned PEFT archive:
  `dist/training-showcase-20260718-live-256step-r1/training_qwen15b_standard_peft_adapter.zip`
- PEFT archive SHA-256:
  `f244519109fc22c8a6c9d61e9018273d12903700710f59baca0b3066fc83d075`
- Adapter safetensors SHA-256:
  `ea9b7c0c8b73c692e99c965d392c2aadab325a21e704d8dad21fe05796d7c13e`

The public-safe report contains hashes and metrics, not adapter tensor values,
raw text, token IDs, credentials, Coordinator URLs, or runtime-private state.
The PEFT archive is the actual learned model output and must be reviewed under
the base-model and dataset licenses before external publication.

## Verify

```bash
PYTHONPATH=. python scripts/training_qwen15b_showcase_check.py \
  --report dist/training-showcase-20260718-final-r1/training_qwen15b_showcase.json \
  --require-ready --json
```

The strict checker returns `ok=true`, `showcase_ready=true`, and zero errors.
The repository regression run after implementation completed with 2,557
passed, two skipped, and zero failed tests.
