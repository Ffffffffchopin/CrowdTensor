# Benchmark Policy

Community benchmark artifacts bind measurements to a workload hash, topology,
model revision, adapter version, Provider set, and gate type.

Required metrics are:

- contiguous committed steps and stability/failure counts;
- step throughput and p50/p95 latency;
- worker replacement and Coordinator restart recovery time;
- activation/gradient transfer counts and byte summaries;
- checkpoint count, bytes, write/read/repair latency, and retention;
- CPU/GPU/TPU allocation and peak memory summaries where available.

The Community RC does not impose another 15% improvement target. A throughput
drop above 20% or p95 regression above 25% must have an explicit, reviewable
explanation. Smaller workloads or reduced topology cannot be substituted under
the same benchmark identity.

The retained Qwen Production RC is historical long-soak evidence. The new
Community gate independently requires either 30 minutes plus 50 contiguous
steps, or 100 contiguous steps before the 45-minute deadline. Shared Kaggle
host variance is reported; it is not an external SLA.

## Qwen2.5-1.5B Elastic Training Showcase

The retained 2026-07-19 showcase completed 256 real LoRA optimizer steps and
131,072 training tokens across two sequential T4x2 Kernel generations. The
first pair was deleted at step 128 before two new Kernels restored central
checkpoints. On 64 pinned WikiText-2 validation sequences, loss changed from
2.731937 to 2.350524 and perplexity from 15.3626 to 10.4911. The strict
showcase checker passes with zero errors and complete cleanup.

These values are workload-specific quality evidence, not a general model
benchmark or an SLA. Full details and artifact hashes are in
[`qwen15b-elastic-training-showcase.md`](qwen15b-elastic-training-showcase.md).

## Qwen2.5-7B Elastic GSM8K SFT Showcase

The retained 2026-07-19 RC completed 256 real LoRA/SFT optimizer steps and
262,144 non-padding tokens across two sequential pairs of concurrent T4x2
Kernels. Both first-generation Kernels were deleted at step 128 before fresh
Kernels restored four central stage checkpoints and continued exactly once.

On 128 preregistered GSM8K test examples disjoint from the development set,
normalized exact match changed from 92/128 (71.875%) to 95/128 (74.219%), or
+2.344 percentage points. Valid answer rate stayed at 100%; reserved-train loss
changed from 1.389790 to 0.546368. This passes the preregistered practical
+2-point gate. The paired bootstrap 95% interval is [-6.25, +10.9375]
percentage points and includes zero, so the result is not described as
statistically significant.

The canonical checker passes with zero errors and all live/private resources
are cleaned. These values are GSM8K-specific quality evidence, not a general
reasoning benchmark, physical multi-host proof, or SLA. Full attempt history,
artifacts, hashes, and boundaries are in
[`qwen7b-gsm8k-elastic-showcase.md`](qwen7b-gsm8k-elastic-showcase.md).
