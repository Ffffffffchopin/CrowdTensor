# Quickstart

## Install

Python 3.11 or 3.12 is supported.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
crowdtensor --help
```

Install `.[hf]` for real PEFT work and `.[storage]` for S3-compatible uploads.
Do not install accelerator frameworks globally.

## Create A Workspace

This does not download a model:

```bash
crowdtensor train init ./project \
  --model Qwen/Qwen2.5-7B-Instruct \
  --model-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --dataset openai/gsm8k \
  --dataset-revision 740312add88f781978c0658806c59bc2815b9866 \
  --model-adapter qwen2_lora_v1 \
  --mode elastic-delta \
  --target-steps 100

crowdtensor train inspect ./project --json
crowdtensor train backends --json
crowdtensor adapters list --json
```

`pause`, `resume`, `status`, and `export` operate only on the small control
workspace. Export excludes weights, credentials, and private paths.

## Local Volunteer PEFT

Install the optional model stack first:

```bash
python -m pip install -e '.[hf]'
```

Create a tiny local evaluation Campaign:

```bash
crowdtensor volunteer campaign create-local ./campaign \
  --target-rounds 1 --local-steps 1
```

Start the user-owned Session in one terminal:

```bash
python -m build
crowdtensor release prepare ./campaign-release

crowdtensor train run ./operator \
  --campaign-dir ./campaign \
  --release-dir ./campaign-release
```

Contribute one bounded Work Unit in another terminal:

```bash
crowdtensor train join ./contributor \
  --invite ./campaign/.private/volunteer_invite.json \
  --device cpu --max-local-steps 1 --max-work-units 1
```

For remote contributors, terminate TLS in front of the Session and use
`--coordinator-url https://... --code <one-time-code>`. External plain HTTP is
rejected. Never publish the invite or pairing code.

Create a short-lived Agent code on the operator host:

```bash
crowdtensor volunteer pair-code ./campaign --mode agent --ttl-seconds 3600
```

The `/join` page constructs the installer command from its own HTTPS origin.
The installer verifies the exact wheel checksum and runs a resource preflight
before redeeming the code. A failed preflight therefore does not consume it.
Set `CROWDTENSOR_DEVICE=cpu` before the command to opt out of an otherwise
auto-detected GPU, or set an explicit `cuda:N` device.

On a slow or interrupted network, rerun the same installer command; its
Campaign wheel download resumes and pip uses a 10-minute read timeout with
bounded retries. When the official PyTorch wheel is already downloaded, set
`CROWDTENSOR_TORCH_WHEEL_PATH=/path/to/torch-2.11.0+cpu-*.whl` to install it
locally instead of downloading it again. The wheel must be from a trusted
source; the Campaign package itself is still checked against `SHA256SUMS`.

Run an explicit private held-out comparison after updates have committed:

```bash
crowdtensor volunteer campaign evaluate ./campaign \
  --heldout-quality --json
```

This reports before/after loss and perplexity. It never turns a small bounded
comparison into a statistical-significance claim.

For a reviewed community-data Campaign using the SmolLM3-3B adapter, follow the
[Commons 3B specification](commons-campaign.md). Its completed controlled CUDA
reference gate is recorded in the
[machine-readable report](evidence/commons-3b-kaggle-live.json). A new Campaign
must still pass its own accelerator, held-out, lineage, and export gates before
being presented as live evidence.

## Stable-Sharded Planning

Create with `--mode stable-sharded`, provide a public-safe stable capability
snapshot, and explicitly verify the upstream trainer contract:

```bash
crowdtensor train plan ./stable-project \
  --capability ./stable-capability.json \
  --runtime-probe \
  --trainer-entrypoint train.py \
  --trainer-contract-verified \
  --transformer-layer-class DecoderLayer \
  --materialize
```

Then run one bounded interval:

```bash
crowdtensor train run ./stable-project \
  --work-unit-steps 10 --max-work-units 1
```

See [the stable trainer contract](stable-sharded-trainer.md). A plan alone does
not prove that the model trained.

## Validate The Repository

```bash
python scripts/check_repository.py --json
crowdtensor release verify ./campaign-release --json
python -m pytest -q
python -m build --wheel
```
