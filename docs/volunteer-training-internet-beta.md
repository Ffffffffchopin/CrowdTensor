# Volunteer Training Internet Beta Engineering RC

The Volunteer Training Internet Beta Engineering RC is the current strongest
local engineering proof for CrowdTensor's low-frequency volunteer PEFT path.
It closes the implementation and local-process validation work that can be
done without claiming an independently administered physical multi-machine
run.

## Canonical Evidence

The canonical RC is:

```text
dist/volunteer-training-internet-beta-engineering-rc-20260718-r3/
  volunteer_training_internet_beta_engineering_rc.json
```

File SHA-256:

```text
d00111b453cf24bb6841805bb7524e4647cf698be1c21dd16e12756ff52663b5
```

Embedded content hash:

```text
sha256:d21a8b66690e02135aad12095a735ed0df0c3513404d6a21eec9d9d54090165b
```

Strict verification:

```bash
PYTHONPATH=. python scripts/volunteer_training_internet_beta_check.py \
  --report dist/volunteer-training-internet-beta-engineering-rc-20260718-r3/volunteer_training_internet_beta_engineering_rc.json \
  --require-ready --json
```

The checker returns zero errors, `goal_achieved=true`, and
`volunteer_training_internet_beta_engineering_rc_ready=true`.

## Verified Scope

- Imports the public `HuggingFaceTB/SmolLM2-135M` model at exact revision
  `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` through
  `smollm2_lora_v1`. All ten imported snapshot files are hash-bound; the model
  declares Apache-2.0.
- Imports `Salesforce/wikitext`, config `wikitext-2-raw-v1`, at exact revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`. The train and validation parquet
  SHA-256 values are fixed in the Campaign provenance; raw text and token IDs
  remain private.
- Runs three quorum-2 rounds. Six distinct CLI subprocesses perform real
  PyTorch autograd and Transformers/PEFT LoRA updates, advancing the canonical
  Adapter from `v0` to `v3` through a verified three-link checkpoint lineage.
- Rejects direct HTTP and an untrusted forwarded identity while accepting the
  trusted forwarded-HTTPS reverse-proxy contract. This is a TLS-termination
  contract test, not a public TLS handshake or certificate test.
- Uses content-addressed local object storage for campaign artifacts and
  completed upload blobs. The storage module also provides an S3/MinIO
  presigned-download adapter contract; this RC does not claim a live external
  S3 deployment.
- Uses persistent 64 KiB chunk sessions. One 4,938,616-byte delta is interrupted
  after its first chunk, survives an API/Coordinator process restart, resumes
  the same upload, and is accepted without rerunning training.
- Recovers one Cell that disappears after claiming work, one unavailable
  Coordinator endpoint, and two Coordinator process restarts. Active lease
  generations and canonical artifact hashes are preserved.
- Compares the distributed result with a centralized run using exactly six
  optimizer steps and 96 tokens on the same model/data contract. Initial,
  distributed, and centralized validation losses are finite. This comparison
  does not assert quality equivalence or superiority.
- Launches a separate replay process that reloads all three Adapter states,
  checks the lineage head, and reevaluates the immutable validation set.
- Stops the HTTP service, reaps every Cell subprocess, removes resumable
  sessions and the entire private proof tree, and leaves only hash-bound public
  evidence.

## Operator Workflow

Create the pinned campaign:

```bash
crowdtensor volunteer campaign import-smollm-wikitext campaign-dir \
  --target-rounds 3 --local-steps 1
```

For local evaluation, prepare or serve on loopback:

```bash
crowdtensor volunteer serve campaign-dir --prepare-only
crowdtensor volunteer serve campaign-dir --host 127.0.0.1 --port 8789
```

For an Internet-facing deployment, put the service behind a configured HTTPS
reverse proxy. The proxy must inject the trusted forwarded protocol and proxy
identity expected by the Coordinator; the identity is private configuration.

```bash
crowdtensor volunteer serve campaign-dir \
  --host 127.0.0.1 --port 8789 \
  --public-url https://training.example.org \
  --require-https --trust-forwarded-headers \
  --trusted-proxy-id "$PRIVATE_PROXY_ID"
```

The operator privately sends the mode-0600 invite. A contributor needs one
command:

```bash
crowdtensor volunteer join campaign-invite.json --once --device auto
```

The Cell detects hardware, enforces download/local-step limits, uses a shared
content cache when configured, heartbeats its lease, performs local PEFT, and
uploads only a low-frequency safetensors delta. Its workspace can be paused,
resumed, inspected, and cleaned with the matching `volunteer` subcommands.

## Remaining Gate

This RC deliberately does not prove:

- two independently administered physical Internet machines;
- a public TLS certificate/handshake or live external S3/MinIO service;
- Sybil resistance, poisoning resistance, permissionless Byzantine safety,
  secure aggregation, or confidential training;
- useful model quality, broad scaling, full-parameter training, GA, or SLA.

The next and only external Beta acceptance gate for this vertical is to run the
same pinned campaign through its ordinary CLI/HTTPS path on at least two
independently administered Internet hosts, retain latency/bandwidth and churn
evidence, independently reproduce the RC, and clean all remote resources. Do
not rerun the local r3 proof merely to recreate achieved evidence.
