# RFC 0001: Community Qwen2.5-7B GSM8K Campaign

- Status: **Draft for community review**
- Published: 2026-07-25
- Decision: whether to open a controlled 7B volunteer-training Campaign
- Current live Campaign: [Founding SmolLM2 systems Campaign](https://crowdtensor.24.199.118.54.nip.io/v1/volunteer/dashboard)
- Feasibility evidence: [Qwen2.5-7B elastic GSM8K showcase](../qwen7b-gsm8k-elastic-showcase.md)

## Summary

This RFC proposes a controlled community LoRA/SFT Campaign for a pinned
Qwen2.5-7B-Instruct model and pinned GSM8K data. Contributors would complete
small, bounded updates and may leave after one work unit. The Coordinator would
validate and aggregate accepted deltas, commit checkpoint lineage, pause when
eligible compute disappears, and resume when new contributors arrive.

This document requests review. It does not open enrollment, allocate public
work, or claim that the ordinary Volunteer Campaign path can train this 7B
topology today.

## Candidate Inputs

| Item | Pinned candidate |
| --- | --- |
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Model revision | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Model license | Apache-2.0 |
| Dataset | `openai/gsm8k`, config `main` |
| Dataset revision | `740312add88f781978c0658806c59bc2815b9866` |
| Dataset license | MIT |
| Method | Frozen-base LoRA/SFT, rank 4, alpha 8 |
| Sequence length | 256 tokens |
| Candidate learning rate | `2e-5` |

The prior 256-step showcase used these identities and provides feasibility
evidence. It does not pre-approve a new Campaign or make its previously used
confirmatory holdout fresh again.

## Objective And Milestones

The literal objective is to improve bounded GSM8K-style mathematical answer
accuracy without reducing valid-answer rate. It is not a claim of broad
reasoning improvement.

1. **Systems pilot:** 16 accepted updates from at least four admitted Cells,
   with contributor replacement and checkpoint recovery.
2. **Training pilot:** 256 exactly-once optimizer steps, pausing for an interim
   evaluation every 128 steps.
3. **Extension decision:** continue toward at most 1,024 steps only after a
   public evaluation and explicit RFC amendment.

The Campaign pauses automatically when eligible compute is absent. A target is
not permission to continue through a failed quality or safety gate.

## Evaluation

- Primary metric: normalized exact match on a new hash-bound holdout that was
  not used to select the previous showcase Adapter or this Campaign's settings.
- Secondary metrics: strict `####` exact match, valid-answer rate, validation
  loss, and perplexity.
- Publish the frozen-base baseline before accepting training updates.
- Publish the candidate Adapter, exact revisions, evaluation code, checkpoint
  hashes, and both positive and negative results.
- Do not claim statistical significance unless the preregistered interval gate
  passes. A practical improvement must be labelled separately.

## Contribution Contract

The first 7B Campaign would use controlled admission and bounded work units.
The initial provider target is a CUDA device capable of the reviewed quantized
or stage-selective runtime. CPU-only and TPU participation require separate
runtime evidence before being advertised for this Campaign.

Public status may include aggregate progress, provider classes, hashes,
latency, accepted-update counts, and checkpoint lineage. It must not include
Cell identities, credentials, raw examples, token IDs, tensor values, private
paths, or account names.

## Governance And Safety

- One Operator controls admission, emergency pause, and rollback during Beta.
- At least two named maintainers, including an evaluation owner, must accept
  responsibility before the RFC can move to Accepted.
- Dataset and model redistribution rights must be reviewed again at launch.
- Norm checks, leases, replay protection, quorum, and short-lived credentials
  reduce accidental abuse; they do not prove semantic poisoning resistance,
  Sybil resistance, Byzantine safety, or secure aggregation.
- Rejected and failed attempts remain in the public decision record.

## Launch Blockers

This RFC cannot move to **Accepted** until all of the following are complete:

- a second named maintainer and an evaluation owner accept the governance role;
- the ordinary Volunteer Campaign path supports the pinned 7B quantized or
  stage-selective runtime without a private one-off launcher;
- a fresh holdout and preregistration artifact are committed and validated;
- at least two independently administered Internet hosts pass the ordinary
  invite, HTTPS, training, submission, recovery, and cleanup flow;
- an incident, pause, rollback, and contributor-removal procedure is reviewed;
- resource limits and expected download, memory, runtime, and energy costs are
  published before contributor recruitment.

## Prior Evidence And Non-Claims

The retained showcase completed 256 real LoRA/SFT steps across two successive
pairs of Kaggle T4x2 Kernels. All first-generation workers were deleted at step
128, a zero-Miner interval was observed, and fresh workers restored central
checkpoints. Normalized exact match changed from 71.875% to 74.219%; the paired
bootstrap interval included zero, so statistical significance was not claimed.

That result is Kaggle logical multi-node evidence. It is not proof of unrelated
physical contributors, a permissionless network, broad model improvement,
full-parameter training, production availability, or an SLA.

## Requested Feedback

Reviewers should focus on the new holdout design, the 256-to-1,024-step stop
rule, minimum useful work-unit size, 16 GB GPU feasibility, update poisoning,
maintainer accountability, and whether GSM8K is a sufficiently useful first
public 7B objective.

Use the [Campaign proposal issue form](https://github.com/Ffffffffchopin/CrowdTensor/issues/new?template=campaign_proposal.yml)
for a structured alternative, or open an RFC issue for a specific amendment.
