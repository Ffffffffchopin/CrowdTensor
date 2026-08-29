# Commons 3B Public Pilot

Status: **launch candidate**. This brief defines the next community-facing
Campaign; it does not claim that external contributors have joined it yet.
The completed controlled reference run remains in
[`docs/evidence/commons-3b-kaggle-live.json`](evidence/commons-3b-kaggle-live.json).

## The One-Sentence Objective

Build a small, openly documented instruction adapter by combining short
contributions from ordinary machines, reviewed community data, and independent
artifact review.

The pilot is deliberately narrow. It is a 3B PEFT adaptation Campaign, not
pretraining from scratch, full-parameter training, or a promise of a hosted
model service.

The machine-readable brief is
[`campaigns/commons-3b-public-pilot.json`](../campaigns/commons-3b-public-pilot.json).
It is content-hash-bound and contains no credentials, raw records, or private
runtime paths.

## Fixed Starting Point

- Base model: `HuggingFaceTB/SmolLM3-3B-Base`, revision
  `d78a42f79198603e614095753484a04c10c2b940`, Apache-2.0.
- Adapter: `smollm3_lora_v1`, `elastic_delta`, response-supervised PEFT LoRA.
- Reproducibility seed: `openai/gsm8k`, revision
  `740312add88f781978c0658806c59bc2815b9866`, MIT.
- Community data: only `crowdtensor_data_pack_v1` manifests that pass the
  named license, personal-data, copyright, contamination, moderation, and
  redistribution reviews.
- Work Unit: 5-15 minutes as an operator estimate, with eight local optimizer
  steps by default. The actual release preflight is authoritative for a given
  Campaign.
- Target: three rounds, minimum quorum four, frozen base weights, and a held-out
  evaluation fixed before public training admission.

The seed data is a reproducibility baseline, not a claim that it is the final
community corpus. New training and evaluation packs must remain disjoint.

## Three Contribution Lanes

### Compute

The primary lane is the Native Agent at [`/join`](project-site.md). An approved
contributor receives a one-time code, passes a local resource check, runs one
bounded CPU or CUDA Work Unit, and may leave. The user-owned Session validates
the delta and preserves the checkpoint. Enrollment remains controlled until a
named operator publishes the active Campaign's admission policy.

### Data

Open a [Data Pack submission](https://github.com/Ffffffffchopin/CrowdTensor/issues/new?template=data_pack.yml)
with a public artifact link, records hash, license, provenance, and review
record. Do not paste raw rows into an issue. A submission is a review request;
it is not admitted until an operator imports and validates the pack against the
Campaign's immutable train/evaluation boundary.

### Review

Use the [Campaign artifact review form](https://github.com/Ffffffffchopin/CrowdTensor/issues/new?template=campaign_review.yml)
to check a manifest, checkpoint lineage, evaluation report, export, or release.
Reviewers should record exact files, commands, and hashes. A review can block
publication, but it cannot grant admission or convert hosted logical workers
into independent physical contributors.

## Success Targets

These are launch targets, not current metrics:

| Target | Definition |
| --- | --- |
| 10 external contributors | Distinct, independently administered contributors accepted by the operator; logical Kaggle Cells do not count. |
| 100 accepted Work Units | Exactly-once, hash-verified native updates recorded by the Campaign. |
| 3 completed rounds | Each round meets its declared quorum and advances the canonical Adapter. |
| 2 reviewed community packs | At least two admitted training or evaluation contributions with public-safe manifests. |
| 1 independent evaluation | A reviewer can reproduce the before/after result from pinned inputs and hashes. |

The public dashboard may show active logical Cells and accepted updates. It
must not report an external-contributor number until the operator has a separate
admission ledger for that metric.

## Launch Gate

Before posting an open compute call, the operator must:

1. Replace placeholder maintainer roles with two accountable maintainers and
   publish the decision and rollback policy.
2. Validate a concrete Campaign proposal with immutable model and Data Pack
   revisions, licenses, a held-out split, and a baseline result.
3. Serve a stable HTTPS user-owned Session with a public-safe snapshot and an
   exact Campaign release. Never publish the invite or pairing code.
4. Test one complete Work Unit from the same release and verify the exported
   receipt, checkpoint lineage, and cleanup path.
5. Publish the first status snapshot before accepting community updates.

No GPU or Kaggle resource is required to prepare this gate. Kaggle results can
bootstrap engineering evidence, but they must be labeled as hosted logical
workers.

## Public Claims

The pilot may say that CrowdTensor supports bounded, interruptible PEFT work,
checkpoint recovery, validation, and public-safe accounting when those artifacts
are present. It may not say that the pilot is permissionless, Sybil-resistant,
poisoning-resistant, statistically significant, independently multi-host
validated, production-grade, or community-built before the corresponding
evidence exists.

The current controlled reference proves a useful protocol path: eight accepted
3B updates across successive hosted logical CUDA workers, recovery and
reassignment, and a held-out token-loss improvement. It is a starting proof for
the pilot, not evidence of the success targets above.

## Launch Copy

> **Help train a 3B model in short sessions.** CrowdTensor's Commons 3B pilot
> accepts one bounded CPU/GPU update at a time. Leave after one Work Unit; the
> checkpoint stays. You can also contribute reviewed data or check the public
> hashes and evaluation. Enrollment is controlled while the first maintainers
> validate the campaign.

Use the status page and the machine-readable brief as the source of truth. Keep
the call to action to one link: the active Campaign's `/join` page.
