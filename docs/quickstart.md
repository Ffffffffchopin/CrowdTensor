# Quickstart

This guide runs a local Coordinator and a Python Miner. It validates the current CrowdTensorD Alpha control-plane loop; it does not train a real LLM.

## Python Environment

Use Python 3.11 or newer.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

If your environment has no network access, preinstall `setuptools` and `wheel` in the virtualenv or use a base image that already includes them. The package install uses standard Python build metadata.

The install creates two console commands:

```bash
crowdtensord --help
crowdtensor-miner --help
```

## Run Coordinator

```bash
crowdtensord \
  --host 127.0.0.1 \
  --port 8787 \
  --state-dir state
```

The Coordinator creates and maintains the local checkpoint/event state under `state/`.

## Run One Miner

In another shell:

```bash
crowdtensor-miner \
  --coordinator http://127.0.0.1:8787 \
  --miner-id local-1 \
  --once
```

Expected behavior:

- Miner claims one task.
- Miner sends heartbeats while computing.
- Miner submits a validated result.
- Coordinator updates the tiny model state.
- Miner exits with a JSON summary.

## Token-Protected Local Run

For local demos, plaintext tokens are simplest. For remote demos, generate hashed token config values:

```bash
python3 scripts/hash_token.py local-miner
```

The Coordinator accepts either plaintext values or `sha256:` verifiers. Miners still send the original token.

Start Coordinator:

```bash
crowdtensord \
  --host 127.0.0.1 \
  --port 8787 \
  --state-dir state \
  --miner-token local-miner \
  --observer-token local-observer \
  --admin-token local-admin
```

Run Miner:

```bash
CROWDTENSOR_MINER_TOKEN=local-miner crowdtensor-miner \
  --coordinator http://127.0.0.1:8787 \
  --miner-id local-secure-1 \
  --once
```

Read metrics:

```bash
curl -H 'x-crowdtensor-observer-token: local-observer' \
  http://127.0.0.1:8787/metrics
```

## Runtime Acceptance Pack

Run the release gate to check Alpha packaging and documentation integrity:

```bash
python3 scripts/release_gate.py --json
```

This is a static open-source release check. It does not replace runtime acceptance.

Run the default non-browser smoke suite:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8910 \
  --report /tmp/crowdtensor_acceptance.json
```

Run the same suite with local auth enabled inside checks that support shared auth env vars:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8950 \
  --miner-token local-miner \
  --observer-token local-observer \
  --report /tmp/crowdtensor_auth_acceptance.json
```

## Remote Miner Demo

Generate a registry-backed invite and run a Miner on another Linux host or container:

```bash
python3 scripts/create_miner_invite.py \
  --registry state/miner_registry.json \
  --miner-id remote-linux-1 \
  --coordinator-url https://YOUR_COORDINATOR_HOST
```

See [Remote Miner Onboarding](remote-miner.md) for the full controlled remote demo flow and `scripts/remote_miner_join_check.py`.

## Docker Compose

Run the local stack:

```bash
docker compose up --build coordinator miner
```

Check health:

```bash
curl http://127.0.0.1:8787/health
curl http://127.0.0.1:8787/version
curl http://127.0.0.1:8787/ready
```

Check metrics:

```bash
curl -H 'x-crowdtensor-observer-token: local-observer' \
  http://127.0.0.1:8787/metrics
```

The default Compose tokens are for local demos only. Copy `.env.example` to `.env` and change the values before sharing a machine.

## Browser Experiments

Serve the static web directory:

```bash
python3 -m http.server 8765 --directory web
```

Open the WebRTC tensor tunnel:

```text
http://127.0.0.1:8765/index.html?role=receiver&room=demo
http://127.0.0.1:8765/index.html?role=sender&room=demo
```

Run the core browser acceptance pack when Playwright and a browser are available:

```bash
python3 scripts/browser_acceptance_pack.py \
  --base-port 9310 \
  --report /tmp/crowdtensor_browser_acceptance.json
```

It runs `webrtc_smoke.py`, `runtime_contract_check.py`, and `browser_miner_smoke.py`. Use `--allow-skip` in CI-style environments where Playwright or Chromium may be unavailable.

Run the broader browser smoke set:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8930 \
  --include-browser \
  --report /tmp/crowdtensor_browser_acceptance.json
```
