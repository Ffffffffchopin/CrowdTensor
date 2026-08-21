# Commons 3B Campaign

Status: **reference accelerator launch gate completed on 2026-08-21**.

Commons 3B is the public reference Campaign for CrowdTensor's intermittent
training mode. It is deliberately a PEFT instruction-training Campaign, not a
claim that volunteers have pretrained a 3B model from scratch.

## Fixed Boundary

- Base model: `HuggingFaceTB/SmolLM3-3B-Base`.
- Model revision: `d78a42f79198603e614095753484a04c10c2b940`.
- Model license: Apache-2.0.
- Adapter: `smollm3_lora_v1`, LoRA rank 8 and alpha 16 by default.
- Mode: `elastic_delta`; bounded Work Units may run at different times.
- Default round: four data shards and quorum four; expired work is reassigned.
- Supervision: response tokens only; prompt and padding labels are masked.
- Evaluation: a separate immutable Data Pack using mean held-out token loss and
  perplexity. It is not a statistical-significance claim.
- Trust: controlled enrollment. Permissionless poisoning and Sybil resistance
  are not claimed.

The launch gate requires a completed accelerator run that advances the
canonical Adapter through at least two rounds, survives replacement of every
original Cell, and publishes a current held-out evaluation and checkpoint
lineage. The reference run below passed that gate.

## Completed Reference Gate

The finalized `commons-3b-gsm8k-live-20260817` Campaign used the pinned
3,352,881,152-parameter model and two reviewed MIT-licensed GSM8K Data Packs:
128 training records and 16 held-out records. Real Transformers/PEFT autograd
ran in private Kaggle T4x2 hosted runtimes as bounded CUDA Cells.

| Result | Value |
| --- | --- |
| Elastic rounds | 2 of 2 |
| Accepted updates | 8 exactly once, from 8 distinct logical Cells |
| Accepted tokens | 11,052 |
| Local optimizer steps | 64 total |
| Uploaded deltas | 484,187,456 bytes |
| Recovery exercised | 3 coordinator recoveries, 3 expired leases, 3 reassignments |
| Final Adapter | v2, `sha256:06ff0db17f7ae1f229237ca22241994d1c960c0835728789b5495273b105117f` |
| Held-out token loss | 0.9121 at v0 to 0.5856 at v2, a 35.8% reduction |
| Held-out perplexity | 2.4896 at v0 to 1.7961 at v2 |
| Public export | 56,007,006 bytes, seven public-safe files |

All four accepted round-one updates used logical Cell identities distinct from
round zero. A separately launched trusted CUDA evaluator verified the pinned
model, held-out artifact, Adapter hashes, logits, loss, and lineage before the
Campaign was finalized. Two of the eight short Work Units reported a higher
ending shard loss, so the evidence preserves both per-unit variance and the
aggregate held-out result.

The [machine-readable report](evidence/commons-3b-kaggle-live.json) binds the
model, data, Adapter lineage, evaluation, and export hashes. This run used
controlled enrollment and hosted logical workers. It does not establish
independently administered physical multi-host operation, permissionless
Byzantine safety, poisoning or Sybil resistance, statistical significance, GA,
or an SLA.

## Data Pack v1

Training data enters the Campaign as canonical instruction JSONL:

```json
{"record_id":"example-001","prompt":"A bounded instruction","response":"A reviewed response","language":"en"}
```

Each Data Pack binds the canonical record hash, SPDX license, provenance,
languages, domains, a hashed contributor identity, and review decisions. The
Commons importer rejects packs unless redistribution, training, personal-data,
copyright, benchmark-contamination, moderation, and public-record gates all
pass. It also rejects duplicate training records and any train/evaluation
content overlap.

Create one reviewed pack only after the named reviews have actually occurred:

```bash
crowdtensor train data-pack create records.jsonl ./packs/reasoning-001 \
  --pack-id reasoning-001 \
  --license CC-BY-4.0 \
  --source-kind contributor_authored \
  --language en --domain reasoning \
  --contributor-id maintainer-local-id \
  --redistribution-allowed --training-allowed \
  --personal-data-reviewed --copyright-reviewed \
  --benchmark-contamination-reviewed \
  --moderation-status approved --public-records

crowdtensor train data-pack validate ./packs/reasoning-001
```

The manifest is public-safe, but `records.jsonl` remains a separate file. A
public flag records redistribution approval; it does not cause the Session API
to publish raw rows.

## Build The Campaign

Obtain the exact public model revision into a dedicated model directory, then
create a user-owned Campaign:

```bash
crowdtensor train campaign import-commons ./commons-3b \
  --model-dir ./models/smollm3-3b-base \
  --train-data-pack ./packs/reasoning-001 \
  --evaluation-data-pack ./packs/reasoning-heldout-001 \
  --attest-model-source \
  --campaign-id commons-3b-preview \
  --target-rounds 100 \
  --work-shards 4 --minimum-quorum 4

crowdtensor train campaign validate ./commons-3b
crowdtensor train run ./operator --campaign-dir ./commons-3b
```

`--attest-model-source` means the operator has verified that the local files
came from the declared public, non-gated immutable revision. CrowdTensor hashes
every local model file but cannot infer its remote identity from bytes alone.

After contributions arrive, publish evidence for the current Adapter and
export the public result:

```bash
crowdtensor train campaign evaluate ./commons-3b --heldout-quality
crowdtensor train campaign export ./commons-3b ./commons-3b-public.zip
```

The public snapshot and export include aggregate Data Pack metadata, current
evaluation, checkpoint lineage, Adapter files, and the append-only ledger. They
exclude credentials, Cell identities, raw records, token IDs, and private
runtime state.

## Contribution Accounting

Compute contributions are represented by exactly-once receipts and accepted
token counts. Data contributions are represented by content-addressed Data Pack
manifests. These records provide auditable attribution; they do not create a
token, payment promise, ownership share, or proof that an update improved model
quality.
