# Volunteer Training Launch Kit

This kit is for a careful founding preview of CrowdTensor's community training
vision. It is written for a technical audience that will inspect the evidence,
not for a claim that the project already operates an open global training
network.

The current LocalLLaMA, Hugging Face, and Chinese soft-launch drafts are kept in
[`community-soft-launch-posts.md`](community-soft-launch-posts.md). The older
template below remains as the conservative same-host preview wording.

## Positioning

Use this one-sentence description:

> CrowdTensor is an open Campaign protocol for volunteer model training: each
> contributor runs a small, bounded PEFT update when their machine is available,
> and the Coordinator advances a checkable checkpoint one round at a time.

The useful contrast is **contribution granularity**, not magic scaling. A Cell
can contribute a short local job and leave; the Campaign retains a validated
delta and can resume later. Say “controlled volunteer training” and “founding
preview.” Do not say “permissionless,” “trustless,” “poisoning-proof,”
“decentralized pretraining,” or “production-ready.”

## Claim Matrix

| Statement | Evidence | Public wording |
| --- | --- | --- |
| Two Cells can contribute through the ordinary path | Reproducible local HTTP demo with two independent processes | “The founding demo runs two independent Cells and aggregates one round.” |
| Updates are real PEFT work | Cell reports and retained Operator/Internet Beta artifacts | “The preview uses real PyTorch/Transformers PEFT LoRA on a tiny fixture.” |
| A Coordinator can recover bounded work | Operator Beta checker and lease/reassignment counters | “Leases and stale updates are checked and recoverable.” |
| Inputs and results are auditable | Proposal hash, manifest, checkpoint lineage, ledger, public snapshot | “Campaign inputs and evidence are content-addressed.” |
| Internet-scale quality or speed | No qualifying evidence yet | Do not claim it. |
| Physical multi-host operation | No qualifying evidence in this repository | Say “same-host preview; external gate remains.” |
| Permissionless or adversary-resistant training | Explicitly false in the contract | Do not imply it. |

## 60-90 Second Demo

1. **0-10s:** Show the repository and the sentence “Open campaigns for
   volunteer model training.”
2. **10-20s:** Run the proposal validator and show `campaign_proposal_ready=true`.
3. **20-35s:** Start the bounded public demo; show two Cell processes and the
   Coordinator HTTP endpoint, keeping the private invite off screen.
4. **35-50s:** Open `/v1/volunteer/dashboard`; show round progress, accepted
   updates, and the audit stream.
5. **50-65s:** Open Provenance; show immutable revision hashes and the explicit
   “not claimed” trust boundaries.
6. **65-78s:** Run the public demo checker and show cleanup verified.
7. **78-90s:** Run the launch checker; show `founding_preview_ready=true` and
   `formal_launch_ready=false` until independently administered multi-host
   evidence exists.

Do not show invite files, credentials, local absolute paths, raw dataset rows,
prompts, token ids, tensor values, or generated private text. Record the
terminal with `--json` output and redact process logs before publishing.

## Founding Contributor Onboarding

The first public Campaign should be small enough for a laptop and explicit
enough to audit:

```bash
python -m pip install -e '.[dev,hf]'
crowdtensor volunteer campaign validate-proposal examples/volunteer-campaign/campaign-proposal.json --json
crowdtensor volunteer campaign import-smollm-wikitext campaign-dir --target-rounds 3 --local-steps 1
crowdtensor volunteer serve campaign-dir --prepare-only --json
crowdtensor volunteer join campaign-invite.json --once --device auto --json
```

The Operator privately sends the mode-0600 invite after checking the Cell's
hardware and resource limits. A contributor may stop after one work unit. The
public status route is the appropriate progress link; private Cell status is
not a public leaderboard.

## Suggested Reddit Post

**Title:** CrowdTensor: a small, checkable volunteer-training Campaign for
ordinary machines

**Body:**

> I am building CrowdTensor around a training-first idea: a model Campaign can
> move forward through many small volunteer LoRA updates, without requiring each
> contributor to stay online for the whole run. The current founding preview has
> a real Coordinator, short-lived per-Cell credentials, bounded leases,
> content-addressed deltas, a public-safe Dashboard, and a reproducible demo
> with two independent local Cell processes.
>
> The important caveat is also part of the project: this is same-host evidence,
> not proof of physical multi-host operation, permissionless trust, poisoning
> resistance, or useful model-quality improvement. I am looking for reviewers
> interested in Campaign governance, evaluation design, and the first
> independently administered run.

Link the repository, the launch-readiness JSON, the governance document, and
the Dashboard screenshot. Link the exact proposal and evidence hashes rather
than a hand-written benchmark claim.

For `r/LocalLLaMA`, use the same wording and link the checked artifact rather
than presenting a screenshot as a benchmark. The relevant audience is
interested in reproducible model work, so keep the LocalLLaMA post explicit
about the tiny fixture, same-host scope, and missing external gate.

## Maintainer Response Guide

- **“Is this decentralized?”** Say: “The current preview is a controlled
  Coordinator with volunteer Cells; physical multi-host and permissionless
  trust are future gates.”
- **“Does it improve the model?”** Say: “The protocol reports training and
  checkpoint evidence; the preview does not claim held-out quality improvement.”
- **“Can I contribute?”** Send the pinned proposal, resource limits, private
  invite procedure, and Dashboard URL. Never ask a contributor to post a token.
- **“Why not just use a central GPU?”** Say: “The experiment is about bounded
  participation and auditable accumulation; we will publish latency and quality
  comparisons when the evaluation gate exists.”
- **“What failed?”** Keep failed artifacts and explain the blocker. Do not
  replace a missing external gate with a local mock or a queue screenshot.

## Launch Checklist

- Proposal validates with immutable inputs, licenses, evaluation, and owners.
- Dashboard desktop and mobile screenshots pass canvas and overflow checks.
- Two-Cell demo passes with cleanup and public-safety scan.
- Operator Beta RC strict checker passes.
- README, governance, and launch kit agree on claim boundaries.
- Launch checker reports founding preview separately from formal launch.
- No private invite, credential, raw data, tensor value, or live resource remains.
