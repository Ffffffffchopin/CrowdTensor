# CrowdTensor

**Train a model together.**

CrowdTensor is an open-source system for **open campaigns for volunteer model
training**. Ordinary CPU, GPU, and TPU machines can contribute one bounded
update at a time. When compute arrives, training advances. When it leaves, the
shared checkpoint waits and resumes later.

[Contribute in your browser](https://crowdtensor.24.199.118.54.nip.io/join) | [Live dashboard](https://crowdtensor.24.199.118.54.nip.io/v1/volunteer/dashboard) | [Website](https://crowdtensor.24.199.118.54.nip.io) | [Governance](docs/volunteer-campaign-governance.md)

1. Request a controlled-beta pairing code.
2. Open `/join` for a verified browser task, or run the one-line native Agent.
3. The Coordinator validates the result; checkpoints remain when you leave.

> **Engineering beta:** the public Coordinator and website are live, but
> contributor enrollment is controlled. CrowdTensor does not yet claim
> permissionless admission, poisoning resistance, independent physical
> multi-host validation, or a production SLA.

## The Idea

A Campaign fixes the model, dataset revision, training method, evaluation, and
governance before work begins. Contributors receive small work units, train
locally, and submit LoRA deltas. The Coordinator validates each update,
aggregates accepted work, and commits an auditable checkpoint.

This makes long-running community training possible without requiring every
machine to stay online:

1. A contributor joins for minutes or hours.
2. Their machine completes a bounded local update.
3. The Coordinator validates and checkpoints it.
4. Training pauses when no eligible compute is present and resumes when new
   contributors arrive.

## Campaigns

| Campaign | Status | Scope |
| --- | --- | --- |
| Founding systems campaign | Live, controlled enrollment; first round seeded | Pinned SmolLM2-135M and WikiText-2 LoRA campaign used to harden the public contribution loop |
| [Community 7B campaign](docs/campaigns/qwen25-7b-gsm8k-rfc.md) | Draft RFC | Candidate Qwen2.5-7B GSM8K LoRA Campaign; ordinary-user runtime, governance, and external-host gates remain open |
| Qwen2.5-7B GSM8K proof run | Completed | 256-step elastic training showcase used as evidence for the larger Campaign path |

The website is the source of truth for live round progress, accepted updates,
active contributors, and checkpoint lineage. The 7B Campaign is not accepting
compute until its model, dataset, evaluation, moderation, and rollback proposal
is approved.

## Evidence

The strongest completed showcase fine-tuned pinned `Qwen/Qwen2.5-7B-Instruct`
on pinned `openai/gsm8k` for 256 exactly-once steps across two successive pairs
of Kaggle T4x2 runtimes. At step 128, every old Miner was removed; training
paused with zero Miners and resumed from the central checkpoint on fresh
runtimes.

- 262,144 non-padding training tokens
- Holdout accuracy: 71.875% base to 74.219% adapter (+2.34375 points)
- Validation loss: 1.389790 to 0.546368
- Standard 37 MB PEFT adapter export and reload
- Full cleanup and artifact hash verification

The accuracy gate passed, but its bootstrap interval includes zero; this is a
practical proof run, not a claim of statistical significance. See the
[showcase report](docs/qwen7b-gsm8k-elastic-showcase.md) and
[benchmark index](docs/benchmarks.md).

## Contribute Compute

The browser path needs no installation. It contributes a server-recomputed
scheduler calibration task through a Web Worker, preferring WebGPU with a
WASM/CPU fallback:

```bash
open https://crowdtensor.24.199.118.54.nip.io/join
```

For real CPU or CUDA LoRA work, Python 3.11+ users run one command after
receiving a one-time Agent code:

```bash
curl -fsSL https://crowdtensor.24.199.118.54.nip.io/downloads/install-contributor.sh | sh -s -- https://crowdtensor.24.199.118.54.nip.io CT-XXXX-XXXX-XXXX
```

The one-time code is exchanged for a short-lived, Cell-bound credential and is
not stored by the Coordinator. The native Agent checks workload and download
limits, auto-detects hardware, exposes a loopback-only status page, and stops
after the current atomic work unit on SIGINT/SIGTERM. Public status never
includes pairing codes, Cell identities, credentials, raw training data, token
IDs, tensor values, or private paths.

The installer selects CPU-only PyTorch wheels when no CUDA device is present.
CUDA providers can set `CROWDTENSOR_TORCH_INDEX_URL` for their managed runtime.

JAX/TPU training remains available through the managed heterogeneous-stage
Miner workflow; it is not yet part of this one-click Volunteer Agent.

[Request Founding Beta access](https://github.com/Ffffffffchopin/CrowdTensor/issues/new?template=beta_enrollment.yml)
without posting credentials or machine identifiers. Browser tasks validate the
contribution path but do not train or update the model; native Cells perform
the actual LoRA updates.

To rehearse the workflow without external hardware:

```bash
crowdtensor volunteer campaign create-local campaign-dir --target-rounds 2
crowdtensor volunteer serve campaign-dir --prepare-only
```

The broader manifest-driven workflow starts with:

```bash
crowdtensor community init training-run
crowdtensor community validate training-run --json
crowdtensor community plan training-run --json
```

## What Exists Today

- Durable HTTP Coordinator with short-lived Cell credentials and bounded work
  leases
- Real LoRA delta validation, aggregation, checkpoint lineage, pause/resume,
  and replacement recovery
- CPU, CUDA, and JAX/TPU training providers behind one heterogeneous scheduler
- Public Campaign dashboard, Prometheus metrics, backups, resumable uploads,
  cleanup, and evidence checks
- Pinned model adapters plus a fail-closed plugin conformance contract
- Kaggle logical multi-node validation for training and inference experiments

`formal_launch_ready` remains false until an independent physical multi-host
Campaign passes the external gate. Retained Kaggle evidence is useful systems
evidence, not proof of unrelated Internet contributors.

## Build Or Operate

- [Contributor quickstart](docs/community-quickstart.md)
- [Draft 7B Campaign RFC](docs/campaigns/qwen25-7b-gsm8k-rfc.md)
- [Community soft-launch posts](docs/community-soft-launch-posts.md)
- [Campaign governance](docs/volunteer-campaign-governance.md)
- [Operator runbook](docs/volunteer-training-operator-beta.md)
- [Project website deployment](docs/project-site.md)
- [Training architecture](docs/community-architecture.md)
- [Supported models](docs/model-adapters.md)
- [Threat model](docs/threat-model.md)
- [Provider matrix](docs/providers.md)
- [API reference](docs/api.md)
- [Historical engineering details](docs/project-memory.md)

For the older inference path, operational checks, prompt handling, and manual
multi-process examples, use [the detailed quickstart](docs/quickstart.md) and
[operations guide](docs/operations.md). The release gate and API contract are
documented there rather than duplicated in this README.

## License

Apache-2.0. Model and dataset licenses remain Campaign-specific and must be
approved before contributor recruitment.
