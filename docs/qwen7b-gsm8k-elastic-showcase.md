# Qwen2.5-7B Elastic GSM8K SFT Showcase

This is retained historical evidence. Its model-specific runtime and checker
are archived at Git ref `e332a7b`; they are not part of the active v2 CLI.

CrowdTensor completed a real Qwen2.5-7B-Instruct LoRA/SFT run on Kaggle
logical multi-node infrastructure on 2026-07-19. Two concurrent T4x2 Kernels
trained steps 1-128, both were deleted, and two fresh T4x2 Kernels restored
four central stage checkpoints and completed steps 129-256 exactly once.

## Fixed Inputs

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Model revision: `a09a35458c702b33eeacc393d103063234e8bc28`
- Parameter count: `7,615,616,512`
- Model license: Apache-2.0
- Dataset: `openai/gsm8k`, config `main`
- Dataset revision: `740312add88f781978c0658806c59bc2815b9866`
- Dataset license: MIT
- Training: 256 optimizer steps, four microbatches per step, sequence length
  256, and 262,144 non-padding tokens
- Supervised tokens: 146,659
- Adapter: LoRA rank 4, alpha 8, learning rate `2e-5`
- Evaluation: 128 fixed GSM8K test examples, greedy decode, up to 256 tokens

## Confirmatory Result

| Metric | Frozen base | Adapter | Change |
| --- | ---: | ---: | ---: |
| Normalized GSM8K exact match | 92/128 (71.875%) | 95/128 (74.219%) | +2.344 pp |
| Strict `####` exact match | 88/128 (68.750%) | 95/128 (74.219%) | +5.469 pp |
| Valid answer rate | 100% | 100% | 0 pp |
| Reserved-train validation loss | 1.389790 | 0.546368 | -0.843422 |
| Reserved-train perplexity | 4.014008 | 1.726970 | -2.287038 |

The preregistered primary metric was normalized exact match. The practical
success rule was an improvement of at least two percentage points, or a paired
bootstrap 95% improvement interval whose lower bound was above zero. The
observed `+2.34375` percentage-point change passes the practical rule. The
paired bootstrap interval is `[-6.25, +10.9375]` percentage points, so this RC
does not claim statistical significance. Wilson 95% intervals are
`[63.537%, 78.939%]` before and `[66.013%, 81.013%]` after.

## Attempt Integrity

This was the final full-training attempt, 3 of 3. A disjoint 128-item
development benchmark showed that the earlier `1e-4` Adapter was over-strong:
normalized exact match fell from 105/128 to 95/128 even though validation loss
improved. That result was not accepted as success.

Before attempt 3, the learning rate `2e-5`, training budget, generation config,
metric, threshold, and a new confirmatory holdout were preregistered. The
confirmatory holdout excludes all 128 development examples. The final checker
recomputes both public example-index set hashes and verifies that each set has
128 unique items and their intersection is empty.

## Elastic Evidence

- All four stages committed every optimizer step contiguously and exactly once.
- Both first-generation Kernels were deleted after step 128 and before the
  replacement generation launched.
- The Coordinator observed a bounded zero-Miner interval.
- Four fresh stage sessions restored Adapter, optimizer, GradScaler, and RNG
  state from central checkpoints, independent of old Kernel disks.
- Base weights remained frozen and all stages reported positive LoRA gradients.
- The 392-tensor standard PEFT Adapter covers all 28 transformer layers.
- A fresh NF4 Qwen2.5-7B runtime reloaded the PEFT ZIP and ran base and Adapter
  evaluation in the same T4x2 Kernel.

## Artifacts

- Canonical report:
  `dist/training-qwen7b-gsm8k-showcase-rc-20260719-r1/training_qwen7b_gsm8k_showcase_rc.json`
- Report SHA-256:
  `cc737ea87c6336a0aa423891b60f0a8db5095cd49f76ed2c30fe7a3ca2c4197b`
- Report content hash:
  `sha256:454a814b08695176564486eef8bfa345dc7a17dccffd94a445f4b3495f278007`
- Standard PEFT Adapter:
  `dist/training-qwen7b-gsm8k-showcase-rc-20260719-r1/artifacts/training_qwen7b_standard_peft_adapter.zip`
- Adapter SHA-256:
  `2c2cb02961df78976eceec94110ddda830bacce2e68d1e7a3d0abe367005a431`
- Cleanup audit SHA-256:
  `4134e6f68b5d5a825c66ad8ad03c3fa4313fde6e3879b32b6d5021db59630b35`

The RC also contains hash-bound source, dataset, preregistration, development,
baseline, training, post-benchmark, cleanup, Model Card, showcase, and
reproduction artifacts. It contains no raw questions, answers, generations,
token IDs, credentials, account names, Kernel references, or private paths.

## Reproduce Historical Verification

Create a separate worktree at `e332a7b` and run the checker recorded there
against the retained `dist/` bundle. Do not copy the historical checker or
Qwen stage runtime back into the active source tree.

The archived strict checker returned zero errors and
`showcase_ready=true`. The then-current repository regression passed 2,574
tests with two conditional skips.

## Cleanup And Boundary

All training and benchmark Kernels, private Kaggle Datasets, Coordinator,
tunnel, checkpoint payloads, runtime-private directories, and six local private
GSM8K payload files were removed. The cleanup checker returns zero errors and
`live_resources_left_running=false`.

This is a bounded GSM8K result on Kaggle logical multi-node execution. It is
not evidence of independently administered physical hosts, broad reasoning or
out-of-domain improvement, full-parameter training, permissionless training,
GA, uptime, or a service-level agreement.
