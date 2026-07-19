# Volunteer Campaign Single-Host Operator Beta

The Operator Beta is the current reproducible deployment boundary for an open
volunteer LoRA campaign. One physical host runs the Coordinator behind a real
HTTPS reverse proxy and uses MinIO or another S3-compatible service for
content-addressed resumable update uploads. Contributors use the ordinary
`crowdtensor volunteer join` command.

This release is intentionally not evidence of independently administered
physical machines. The stress gate uses independent operating-system processes
on one host. The retained SmolLM2/WikiText evidence contains real PEFT work;
the 24-Cell stress workload uses explicitly labelled protocol fixture deltas.

## Install

Use Python 3.11 or newer:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install 'crowdtensord[hf,storage]'
```

The base package exposes the protocol and CLI. The `hf` extra is required to
create or train the pinned SmolLM2 campaign. The `storage` extra is required
for S3/MinIO uploads.

## One-command Operator

For a bounded private/local campaign, create and start the Coordinator in one
command:

```bash
crowdtensor volunteer operator ./campaign \
  --profile local \
  --target-rounds 3 \
  --host 127.0.0.1 \
  --port 8789
```

Use `--profile smollm-wikitext` for the pinned public
`HuggingFaceTB/SmolLM2-135M` and `Salesforce/wikitext` Campaign. The command
reuses an existing durable Campaign directory after a restart. Use
`--prepare-only --json` to validate command/configuration handling without
starting a service.

The Operator privately sends the mode-0600 file
`campaign/.private/volunteer_invite.json` to an admitted contributor. A
contributor runs:

```bash
crowdtensor volunteer join volunteer_invite.json --once --device auto
```

The client automatically exchanges the enrollment invite for a unique,
short-lived per-Cell credential. It does not persist the credential value in a
public report.

## HTTPS and MinIO

Internet-facing use must terminate TLS at a reverse proxy. Start the
Coordinator on a private bind address and configure the public URL, trusted
forwarded headers, and one private proxy identity:

```bash
export AWS_ACCESS_KEY_ID='<private MinIO access key>'
export AWS_SECRET_ACCESS_KEY='<private MinIO secret key>'

crowdtensor volunteer serve ./campaign \
  --host 127.0.0.1 \
  --port 8789 \
  --public-url https://training.example.org \
  --require-https \
  --trust-forwarded-headers \
  --trusted-proxy-id '<private proxy identity>' \
  --upload-storage s3 \
  --s3-endpoint http://127.0.0.1:9000 \
  --s3-bucket crowdtensor-volunteer \
  --s3-prefix campaigns/current
```

Create the private bucket before starting the service. Keep MinIO and the
Coordinator state on persistent volumes. The proxy must overwrite, rather than
forward from the client, `X-Forwarded-Proto` and
`X-CrowdTensor-Proxy-Id`. The backend rejects direct HTTP when the HTTPS policy
is enabled.

The RC gate uses a self-signed local CA only to verify an actual TLS handshake
and trusted reverse-proxy path. A public deployment should use an ordinary
public certificate and must not expose the backend or MinIO administrative
port.

## Credential Policy

The private Campaign state stores only credential hashes and signed claim
metadata. Each Cell credential has:

- a bounded expiration time;
- an immutable Cell identity binding;
- explicit claim, heartbeat, artifact, submit, and upload scopes;
- immediate Operator revocation;
- persistent nonce replay protection;
- fixed-window request and upload-byte limits;
- total upload and accepted-submission quotas; and
- a maximum active-lease count per Cell.

The Campaign invite remains an Operator/enrollment secret. It is accepted by
the direct Python API for compatibility with the earlier Internet Beta, but the
ordinary HTTP Cell workflow uses independent credentials.

These controls limit accidental abuse. They do not provide Sybil resistance,
semantic poisoning detection, Byzantine consensus, secure aggregation, or
confidential training.

## Lifecycle

Operator lifecycle commands are durable and idempotent where appropriate:

```bash
crowdtensor volunteer campaign validate ./campaign --json
crowdtensor volunteer campaign start ./campaign --json
crowdtensor volunteer campaign pause ./campaign --json
crowdtensor volunteer campaign resume ./campaign --json
crowdtensor volunteer campaign evaluate ./campaign --json
crowdtensor volunteer campaign finalize ./campaign --json
crowdtensor volunteer campaign export ./campaign adapter-export.zip --json
```

Pause stops new leases while preserving durable state. Finalize requires all
target rounds. Evaluation reports aggregate training metrics and checkpoint
integrity; it is not a held-out quality benchmark. Export includes the public
Campaign, status, evaluation, audit ledger, and canonical PEFT Adapter, but not
invites, credentials, leases, upload sessions, or private runtime state.

## Backup and Upgrade

Create a private mode-0600 backup and restore it into an empty directory:

```bash
crowdtensor volunteer campaign backup ./campaign campaign-private.tar.gz --json
crowdtensor volunteer campaign restore campaign-private.tar.gz ./restored --json
crowdtensor volunteer campaign validate ./restored --json
```

Restore rejects absolute, parent-traversal, and symbolic-link archive members,
rebases durable local paths, reloads the Coordinator, verifies every
content-addressed artifact, and verifies the audit-ledger hash chain. Opening a
v1 Coordinator state automatically migrates it to the versioned v2 Operator
state before requests are served.

Backups contain private credentials and runtime metadata. Do not publish them.

## Monitoring

The HTTPS service exposes:

- `GET /v1/volunteer/health`
- `GET /v1/volunteer/status`
- `GET /v1/volunteer/metrics`

The Prometheus text endpoint reports adapter/round progress, accepted and
rejected updates, expired leases, recoveries, uploaded bytes, active/revoked
credentials, and rate/replay rejections. It does not use Cell IDs, credential
IDs, prompts, dataset text, tensor names, or lease values as labels.

Alert on repeated Coordinator recovery, increasing expired leases, sustained
rate/replay rejection, stalled adapter version, MinIO health failure, and disk
capacity. Back up Coordinator and MinIO volumes before an upgrade.

## Verified Faults

The strict Operator Beta RC verifies on one host:

- 24 independent Cell protocol processes and three quorum rounds;
- one retained six-process, six-step, 96-token real SmolLM2/WikiText PEFT RC;
- slow-Cell lease expiration and reassignment;
- duplicate submission idempotency;
- credential scope rejection, revocation, replay rejection, and rate limits;
- Coordinator process and HTTPS proxy restart with an active lease preserved;
- MinIO unavailability and restart;
- interrupted chunk upload resume without retraining;
- credential and upload capacity rejection;
- lifecycle, export, private backup/restore, and v1-to-v2 migration;
- isolated wheel installation and a non-root project container; and
- removal of all gate processes, containers, S3 objects, and private temporary
  state.

## Boundaries

Operator Beta does not claim:

- independent physical multi-host operation;
- an open permissionless trust model;
- Sybil or model-poisoning safety;
- private or secure aggregation;
- useful model quality from the bounded gate;
- full-parameter training;
- General Availability, uptime, durability, performance, or security SLA.

The next external gate is the same ordinary Operator/Contributor workflow on
at least two independently administered Internet hosts, with measured WAN
behavior and independent reproduction. That external test must not weaken the
credential, cleanup, provenance, or public-safety checks in this RC.
