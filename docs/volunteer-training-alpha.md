# Volunteer Training Protocol Alpha

> This local protocol Alpha remains historical evidence. It is superseded by
> the [Volunteer Training Internet Beta Engineering RC](volunteer-training-internet-beta.md),
> which adds pinned public Campaign import, HTTPS proxy policy, content-addressed
> resumable uploads, restart recovery, three real PEFT rounds, and independent
> replay. Neither artifact proves independently administered physical Internet
> machines.

CrowdTensor's Volunteer Training Protocol is the first vertical slice of the
project's "Folding@home for open LLM training" direction. It lets independent
Training Cells perform several local PEFT LoRA optimizer steps and exchange one
named-tensor delta at a round boundary. The Coordinator aggregates a quorum
with a DiLoCo/Local-SGD outer step and atomically advances the canonical
Adapter.

This design deliberately does not send per-layer activations or per-step
gradients over the public Internet. A future large campaign should use fast
local links inside a Training Cell and this low-frequency protocol between
Cells.

## Verified Alpha

The canonical local HTTP proof is:

```text
dist/volunteer-training-alpha-20260717-r1/volunteer_training_alpha_rc.json
```

Its file SHA-256 is
`c36646c0e367dfd805f5c9047b93de2b87c156a3c1aaa32a0621a58847efad41`
and its embedded content hash is
`sha256:8cc4e84f647e0dc652729b7a5632f54f6d8b1ca21f0fd9233e85df03c81f72de`.

Validate it independently:

```bash
PYTHONPATH=. python scripts/volunteer_training_alpha_check.py \
  --report dist/volunteer-training-alpha-20260717-r1/volunteer_training_alpha_rc.json \
  --require-ready --json
```

The strict checker requires all of the following:

- Four real PyTorch/Transformers/PEFT LoRA Cell updates and eight optimizer
  steps, with base weights frozen.
- Two versioned rounds, a distinct two-Cell quorum in each round, and canonical
  Adapter/outer-step advancement from version 0 to version 2.
- A Cell disappearing before submit, lease expiry, generation-fenced work
  reassignment, replacement from the canonical Adapter, late stale rejection,
  and idempotent duplicate handling.
- Tensor name/shape/dtype/content validation, finite-value rejection, norm
  clipping and hard-norm policies, and forked base-Adapter rejection.
- A real centralized PEFT baseline using the same eight optimizer steps and 256
  training tokens. The fixture records both validation losses but does not use
  them to claim useful model quality.
- Authenticated artifact download, binary safetensors submission, heartbeat,
  a hash-chained public audit ledger, public-artifact scanning, and cleanup.
- Removal of the complete private proof runtime after preserving the hashed
  public campaign, status, ledger, benchmark, and checker evidence.
- A successful invocation of the exact contributor path
  `crowdtensor volunteer join <private-invite> --once`.

The retained proof is loopback HTTP on one physical host. It validates the
transport and process boundary, not independent Internet machines.

## Responsibilities

### Public Campaign

The immutable, versioned Campaign manifest binds:

- Model revision and model-manifest hash.
- Initial Adapter content hash and named-tensor contract hash.
- Dataset snapshot and per-shard hashes.
- Local steps, optimizer, batch, sequence, and resource ceilings.
- Quorum, lease, target-round, DiLoCo/Local-SGD, clipping, and admission policy.
- Explicit trust and transport limitations.

Changing any of these fields changes the manifest hash. A work unit bound to an
older manifest cannot silently join the new campaign.

### Coordinator

The Coordinator is the sole writer of the canonical Adapter. It:

- Issues one shard/version/round-specific leased work unit.
- Fences work by Adapter version, base hash, lease generation, Cell identity,
  lease token, and result idempotency key.
- Validates safetensors before copying accepted data into private storage.
- Requires a quorum of distinct Cell IDs and then stages and atomically renames
  the next canonical Adapter version.
- Persists private state and a public-safe append-only hash-chain ledger.
- Exposes public campaign/status routes and authenticated claim, heartbeat,
  artifact, and submit routes.

Distinct Cell IDs are only an Alpha quorum mechanism. They are not proof of
distinct humans or hardware and do not provide Sybil resistance.

### Training Cell

A Cell:

- Detects CPU/CUDA hardware and enforces local step/download limits.
- Downloads content-addressed model, Adapter, config, and dataset artifacts into
  a private cache and verifies every byte hash.
- Runs real PEFT LoRA locally while renewing its lease.
- Uploads only the round delta and public-safe metrics.
- Supports bounded work, foreground continuous contribution, pause/resume, and
  cleanup without publishing its ID, invite, paths, data, or tensor values.

## Operator Flow

The Alpha includes a deterministic local fixture campaign for protocol testing
and private evaluation:

```bash
python -m pip install -e '.[hf]'

crowdtensor volunteer campaign create-local campaign-dir \
  --target-rounds 2 --local-steps 2

crowdtensor volunteer serve campaign-dir \
  --host 0.0.0.0 --port 8789 \
  --public-url https://training.example.org
```

Terminate TLS in a reviewed reverse proxy. The private invite is written under
the campaign's private directory. Distribute that file through a private
channel and keep it mode `0600`; never put it in a public artifact or issue.

The current `create-local` command creates a tiny deterministic fixture. It is
not a useful public model campaign importer.

## Contributor Flow

After receiving the private invite:

```bash
chmod 600 campaign-invite.json
crowdtensor volunteer join campaign-invite.json --device auto
```

For one bounded work unit:

```bash
crowdtensor volunteer join campaign-invite.json --once --device auto
```

Optional controls:

```bash
crowdtensor volunteer pause ~/.cache/crowdtensor/volunteer/CAMPAIGN_HASH
crowdtensor volunteer resume ~/.cache/crowdtensor/volunteer/CAMPAIGN_HASH
crowdtensor volunteer status ~/.cache/crowdtensor/volunteer/CAMPAIGN_HASH --json
crowdtensor volunteer cleanup ~/.cache/crowdtensor/volunteer/CAMPAIGN_HASH
```

Use `--workspace` when a stable explicit Cell directory is preferable. Use
`--max-local-steps` and `--max-download-gib` to place hard local limits.

## HTTP Contract

```text
GET  /v1/volunteer/health
GET  /v1/volunteer/campaign
GET  /v1/volunteer/status
POST /v1/volunteer/work/claim
POST /v1/volunteer/work/heartbeat
GET  /v1/volunteer/artifacts/{artifact_id}
POST /v1/volunteer/work/submit
```

Private routes require the invite Bearer credential. Delta submission uses a
bounded length-prefixed JSON metadata header followed by raw safetensors bytes;
tensor values are not serialized into JSON. Production-scale campaigns should
replace Coordinator-proxied large artifact transfer with authenticated object
storage or presigned URLs while preserving the same content hashes.

## Security Boundary

The Alpha is invite-authenticated and assumes admitted Cells are mostly
cooperative. Shape, hash, finite-value, loss-spike, clipping, stale/fork, and
duplicate checks limit accidental and simple invalid updates. They do not solve
model poisoning, data poisoning, collusion, Sybil attacks, privacy leakage from
updates, or malicious-but-valid gradients.

Do not claim permissionless Byzantine safety, secure aggregation, differential
privacy, useful model quality, arbitrary model support, GA, or an SLA from this
Alpha.

## Next Milestones

1. Run the same strict gate across at least two independently administered
   physical Internet machines and record latency, interruption, and bandwidth.
2. Add campaign import for a supported Model Adapter and immutable public
   dataset snapshot, plus object-storage artifact transport.
3. Add signed Cell identities, admission cohorts, held-out evaluation,
   reproducible replay sampling, robust aggregation experiments, and governance
   for accepted/rejected rounds.
4. Run multi-hour churn and Coordinator crash-recovery tests with resumable
   uploads and version reconciliation.
5. Publish a bounded real continued-pretraining or fine-tuning campaign with a
   model card, dataset ledger, checkpoint lineage, and independently reproduced
   result before expanding model size or adding incentives.
