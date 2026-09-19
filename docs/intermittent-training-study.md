# Intermittent Work-Unit Study

Status: baseline infrastructure and offline calibration. Adaptive scheduling,
natural-language quality experiments, and concurrent accelerator measurements
are later stages. No user recruitment is required.

## Research Question

How does Work-Unit length change the tradeoff between interrupted computation,
initialization/communication overhead, and PEFT learning quality when resources
arrive for short, unreliable periods?

Hypothesis to test: a policy using only observed speed and past completion
history can improve useful progress over fixed task sizes at a matched resource
budget without an unacceptable change in held-out quality or shard coverage.
Neither improvement nor algorithmic novelty is assumed. Negative results are
valid study outcomes. An accepted receipt is not proof of improved quality.

## Related Work And Overlap

This is a starting comparison based on the linked primary sources, not a claim
of an exhaustive literature review. Reviewed on 2026-09-19.

| Work | Relevant contribution | Consequence for this study |
| --- | --- | --- |
| [DiLoCo](https://arxiv.org/abs/2311.08105) | Infrequent synchronization with local and outer optimization. | Low communication and variable availability are existing ideas. Our retained outer optimizer is not a new algorithm. |
| [Decoupled DiLoCo](https://deepmind.google/blog/decoupled-diloco/) | Decoupled training islands isolate disruptions. | Compare failure assumptions; a CPU fixture cannot establish an advantage over that system. |
| [FedCompass](https://arxiv.org/abs/2309.14675) | Computes variable local workloads from client computing power in a semi-asynchronous setting. | A speed-aware task-length rule already has close prior art and must be compared before claiming novelty. |
| [FedBalancer](https://arxiv.org/abs/2201.01601) | Controls selected samples and adaptive round deadlines. | Deadline and data-selection effects must be separated from task sizing. |
| [Oort](https://www.usenix.org/conference/osdi21/presentation/lai) | Guides participant selection using system and statistical utility. | Fast-client selection can change statistical coverage; measure both. |
| [FedEx-LoRA](https://aclanthology.org/2025.acl-long.67/) | Addresses inexact aggregation of LoRA factors. | Factor averaging is not dense-update averaging; scheduling gains cannot establish equivalence or convergence. |

Before choosing the adaptive policy, review the full methods, code, and newer
availability-aware work. Pin any reproduced implementation and identify
differences in base-weight freezing, aggregation, optimizer state, data
selection, and client lifetimes. A useful outcome may be a measurement study
with clear operating limits rather than a new optimizer.

## Executable Baseline Protocol

The [versioned protocol](../research/intermittent-training/protocol.json) is
content-hash-bound. Both commands refuse an existing output directory. A new
configuration requires a new protocol hash; never edit a completed run to
match a revised hypothesis. The CLI rejects unknown fields, duplicate seeds,
invalid budgets, and step counts incompatible with the two-shard quorum.

```bash
python -m crowdtensor.cli train benchmark \
  --protocol research/intermittent-training/protocol.json \
  --backend trace --output-dir dist/research-trace-v1

CUDA_VISIBLE_DEVICES='' CROWDTENSOR_CPU_THREADS=1 \
  python -m crowdtensor.cli train benchmark \
  --protocol research/intermittent-training/protocol.json \
  --backend cpu-fixture --output-dir dist/research-cpu-v1
```

Trace replay needs only the base package. CPU calibration requires the existing
`[hf]` environment; it makes a tiny local Llama/LoRA fixture and downloads no
model or dataset. It uses at most two Torch CPU threads and no GPU. Use a
project-owned environment and the shared resource rules for dependency work.

Outputs are `protocol.json`, `report.json`, `metrics.csv`, and `complete.json`.
The completion marker binds the final report hash. Individual CPU trials are
also saved as they finish. A failed trial preserves earlier results and writes
a public error class; absence of `complete.json` means the run is incomplete.
The immutable `run-state.json` records that an attempt started, not completion.
The report binds source file hashes, and CPU trials record framework versions,
model/data hashes, baseline evaluation, and final Adapter hashes.

Private Campaigns, checkpoints, tokens, model files, and synthetic token rows
remain inside the run's `.private/` tree. Share the report/CSV, not the entire
run directory. No trial identity represents an external contributor.

### Recorded Calibration: 2026-09-19

The [public evidence summary](evidence/intermittent-training-baselines.json)
binds the protocol, source, environment, and complete local report hashes.
Twelve trace trials and eighteen real CPU trials completed. All nine
policy/seed recovery pairs produced identical final Adapter tensor hashes.
Each CPU trial accepted eight local optimizer steps and 256 nonpadding input
tokens using a 22,688-parameter randomly initialized fixture.

Mean delivered steps across the three synthetic traces were:

| Availability | Fixed short | Fixed long |
| --- | ---: | ---: |
| Stable | 52.00 | 104.00 |
| Intermittent | 29.67 | 48.00 |

Long tasks also lost more computation: in intermittent traces, mean computed
but undelivered steps were 19.83 for long tasks versus 1.50 for short tasks.
Under these assumed costs, reducing loss alone does not maximize delivery.
This is a cost-model observation, not a finding about trained model quality,
committed quorum progress, or a demonstrated adaptive-policy advantage.

Real CPU faults discarded one step in the upstream and short-task cases,
and four in the long-task case. The recovered final model matches each
policy's own uninterrupted reference, not the other policies' models. Recorded
losses are synthetic-fixture diagnostics; their ordering is not an NLP result.

### Synthetic Delivery Cost Model

Two fixed policies (one and four steps) x two availability conditions x three
trace seeds produce 12 runs. Every policy receives the same trace for a given
condition/seed. Each trace models two logical devices taking one and two seconds
per step. Stable windows last 120 seconds. Intermittent windows last 8-24 seconds
with 2-8 second gaps. These numbers and the byte costs are assumptions, not
Kaggle observations or measured WAN rates.

Every window is an ephemeral runtime with a cold cache. Cold start is charged
once per window; setup and upload are charged per attempt. Interrupted setup,
partial computation, and partial upload are accounted separately. A completion
exactly at departure counts as delivered before departure. An entirely offline
trace produces zero delivery and a null rate rather than dividing by zero.

The replay never trains a model, validates a delta, or models a quorum barrier.
It reports **delivered steps**, not committed optimizer progress or token
goodput. It is useful for validating accounting and selecting experimental
conditions; it cannot predict model quality or prove production throughput.
Fixed policies use no future availability information. A later adaptive policy
must receive observations through an interface that excludes future events.

### Real CPU Calibration

Three policies x two fault conditions x three model seeds produce 18 runs:

- `upstream_checkpoint`: Transformers Trainer with AdamW and a constant learning
  rate. The interrupted case commits halfway, computes one extra uncommitted
  step, discards the process state, and restores the saved Adapter, optimizer,
  scheduler, RNG, and data position in a new Trainer.
- `fixed_short` / `fixed_long`: existing Volunteer Cells use one/four local
  steps and quorum two through the v2 controller bridge. Each condition commits
  eight total local optimizer steps (256 nonpadding input tokens).
- In an interrupted elastic run, the first computed delta is discarded before
  submission. The injected coordinator clock advances past lease expiry,
  the coordinator and transport are reconstructed, the old result is rejected,
  and a new Cell retrains the reassigned shard. One accepted submission is also
  replayed to check receipt idempotency.

Every policy must finish with the same Adapter tensor hash in its uninterrupted
and recovered cases. Failure aborts completion; it is never converted into a
successful run. The calibration reports actual local elapsed time, discarded
steps, bytes copied/uploaded through the local protocol, and synthetic held-out
loss. It shuffles execution order deterministically to reduce ordering bias.

These are calibration faults, not a common wall-clock availability trace. The
upstream case loses one step; elastic cases lose an entire local Work Unit.
Recovery timers have different explicit scopes. Do not rank these policies by
their calibration recovery times. Only one Cell executes at a time; no physical
process kill, real network failure, or concurrent multi-device speedup is tested.

Equal token counts do not imply equal optimization: Trainer samples the full
dataset with persistent Adam state; elastic Cells use separate shards, reset
their local Adam state, and aggregate LoRA factors at different frequencies.
The tiny randomly initialized fixture and its synthetic validation rows are
not a pretrained language model or a natural-language benchmark.

## Next Experimental Gate

1. Pin one supported 0.1-0.5B pretrained model, dataset, licenses, train/dev/test
   splits, evaluation command, and metric. Keep the existing 3B Campaign for
   selected later confirmations. No new model family is required by this study.
2. Use development-only seeds/data to calibrate step sizes, cold/warm setup,
   bandwidth, and lifetime ranges. Freeze them before confirmatory runs.
3. Bind the same exogenous fault trace to real training events. Specify restart,
   upload interruption, slowdown, and all-offline behavior, including the point
   at which a step is durable. Never provide future departure times to a policy.
4. Reproduce the relevant resource-aware baseline before adding one candidate
   policy. Address variable local-step weighting and shard coverage explicitly.
5. Freeze a four-policy x two-condition x three-seed main matrix (24 runs) with
   finite token/time/retry limits. This matrix is **planned, not executed** by
   either current benchmark backend. Register failed attempts; do not tune on
   confirmatory seeds or continue until a favorable result appears.
6. Analyze equal resource timelines and equal committed-token budgets
   separately. Include initialization, upload, rejected/recomputed work, and
   idle time. Report task quality, shard coverage, and uncertainty; three seeds
   alone are not a statistical-significance guarantee.
7. Confirm selected findings with 3B and bounded, project-owned Kaggle runtimes.
   Serialize local WSL GPU use through `codex-resource`; acquire Kaggle leases
   and inspect active sessions/quota before any remote launch. A single GPU
   replay cannot supply measured multi-GPU throughput.

## Research Deliverables

- A technical report containing the question, related work, algorithmic
  assumptions, baseline definitions, figures, failed cases, and limitations.
- Pinned source/configuration and public-safe machine-readable results.
- A personal contribution record linking design decisions, implementation,
  experiment analysis, and revisions. Attribute upstream and assisted work.

Recruitment, two-maintainer Campaign governance, a hosted public service,
community popularity, and positive experimental results are not completion
requirements for this research phase. Publication suitability is assessed from
the eventual evidence and novelty, not from passing infrastructure tests.
