# Volunteer Campaign Governance

CrowdTensor's public unit of work is a **Campaign**: a named model and data
revision, a bounded training method, an evaluation plan, and an accountable
operating policy. A Campaign is not an unreviewed queue of arbitrary updates.
The Operator admits Cells, assigns work, validates low-frequency LoRA deltas,
and publishes aggregate evidence.

## Proposal Gate

Every public Campaign starts with
[`schemas/volunteer_campaign_proposal_v1.schema.json`](../schemas/volunteer_campaign_proposal_v1.schema.json)
and a checked proposal. The proposal must identify:

- immutable model and dataset revisions, SPDX licenses, and redistribution rights;
- the personal-data review and prohibited data categories;
- the PEFT method, local step budget, quorum, and frozen base weights;
- a baseline, a held-out split hash, named metrics, and a before/after publication plan;
- supported devices, download limits, and intermittent-contributor behavior;
- at least two named maintainers, a public decision log, and a conflict policy; and
- moderation, rollback, manifest, checkpoint, ledger, attribution, and result-license ownership.

Generate and validate the starting point with:

```bash
crowdtensor volunteer campaign proposal-template campaign-proposal.json --json
crowdtensor volunteer campaign validate-proposal campaign-proposal.json --json
```

The checked example is intentionally conservative. It sets
`permissionless_admission`, `sybil_resistance_claimed`, and
`poisoning_safety_claimed` to false. A proposal cannot turn those boundaries
into marketing claims by changing its title or summary.

## Roles

**Maintainers** own the Campaign proposal, admission policy, public decision
log, license review, release artifacts, and conflict disclosures. At least two
maintainers are named so that a pause or rollback does not depend on one
person.

The **Operator** runs the Coordinator, keeps private invites and credentials,
controls pause/resume/finalize, verifies checkpoints, and publishes redacted
status. The Operator does not receive a private Cell's raw dataset or tensor
values through public status routes.

A **Cell** contributes within a private invite and a short-lived credential.
The Cell receives a bounded shard and base adapter, performs the declared local
PEFT work, and submits one validated delta at a round boundary. It may stop
between rounds; a lease can expire and be reassigned without rewriting the
canonical checkpoint.

**Observers** may inspect public snapshots, evaluation reports, manifests, and
the append-only ledger. They do not receive lease material, credentials, raw
training rows, activations, or tensor values.

## Lifecycle

1. A maintainer opens a proposal and pins its inputs.
2. The Operator validates the proposal and creates the Campaign manifest.
3. The Operator publishes the invite policy and the public Dashboard.
4. Admitted Cells claim bounded work, send heartbeats, and submit deltas.
5. The Coordinator checks hashes, tensor contracts, finiteness, norms, lease
   generation, and distinct-Cell quorum before aggregation.
6. The Operator publishes round summaries, checkpoint lineage, evaluation
   results, and the ledger head.
7. The Campaign is paused, resumed, evaluated, finalized, or rolled back by a
   named owner. A rollback never silently replaces the public manifest.

The Dashboard is an operational view, not a trust oracle:
`GET /v1/volunteer/dashboard` and
`GET /v1/volunteer/public-snapshot` expose only aggregate and hashed metadata.

## Evaluation And Claims

The proposal's baseline and held-out split are immutable inputs to evaluation.
Training loss or a lower adapter norm is not evidence of useful model quality.
Before claiming improvement, publish the exact evaluation revision, metric
definitions, sample count, baseline result, candidate result, and checkpoint
hashes. If the held-out benchmark was not run, the Dashboard and release kit
must say so.

The current public preview proves a real PEFT protocol and a same-host
two-Cell workflow. It does not prove physical multi-host execution,
permissionless admission, Sybil resistance, Byzantine consensus, secure
aggregation, poisoning resistance, privacy against the Operator, or a service
level agreement.

## Safety And Moderation

Campaigns must reject private credentials, non-consensual personal data, and
copyrighted data without permission. Maintainers record conflicts, investigate
reports in the public decision log, and may pause admission while reviewing a
dataset or checkpoint. The moderation owner and rollback owner are explicit in
the proposal; emergency pauses are documented after the fact with a reason
code and the affected manifest hash.

The protocol's rate limits, short-lived credentials, replay protection, norm
checks, and quorum rules reduce accidental abuse. They are not a claim that an
open Internet Campaign is adversary-proof. Public artifacts are scanned for
credentials, private paths, raw data, and tensor values before release.

## Release Levels

- **Founding preview:** checked proposal, public Dashboard, reproducible
  same-host two-Cell demo, public-safe evidence, and the achieved Operator Beta.
- **Formal launch:** all preview requirements plus independently administered
  physical multi-host evidence, a real network route, a completed evaluation
  report, and a reviewed incident/rollback procedure.
- **Mature community Campaign:** repeated external evidence, maintainer
  rotation, documented dataset governance, and an explicit security review.

The repository's launch checker reports these states separately. A founding
preview can be shared without implying that the formal launch gate has passed.
