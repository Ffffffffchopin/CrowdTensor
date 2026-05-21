# CrowdTensorD

CrowdTensorD is an experimental control plane for fault-tolerant distributed AI workloads across untrusted Miner processes.

It currently validates the V1 mechanics needed for a future CrowdTensor network: task leasing, heartbeat timeout recovery, checkpoint replay, result validation, replay audit, Miner admission, and CPU-only training contracts.

## Alpha Status

This project is an **experimental alpha**. It is useful for developers who want to inspect or extend the control-plane mechanics behind distributed AI workers.

It is not yet:

- a production DePIN network
- a real LLM training or inference platform
- a reward, staking, or payment system
- a complete P2P/NAT traversal network
- a hardened public-internet security model
- a GPU, WebGPU, PyTorch, or Transformers benchmark

The current workloads are intentionally small and deterministic so the runtime can be tested without external model dependencies.

## Quickstart

Use Python 3.11 or newer.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Start the Coordinator:

```bash
crowdtensord --host 127.0.0.1 --port 8787 --state-dir state
```

In another shell, run one Miner task:

```bash
crowdtensor-miner \
  --coordinator http://127.0.0.1:8787 \
  --miner-id local-1 \
  --once
```

The Miner claims a task, runs a dependency-free local training loop, uploads a DiLoCo-style delta, and exits with a JSON summary.

For the full walkthrough, see [docs/quickstart.md](docs/quickstart.md).
For endpoint-level integration details, see [docs/api.md](docs/api.md).
For controlled remote Miner setup, see [docs/remote-miner.md](docs/remote-miner.md).

## Docker Compose

Run the local demo stack:

```bash
docker compose up --build coordinator miner
```

Check the Coordinator:

```bash
curl http://127.0.0.1:8787/health
curl http://127.0.0.1:8787/version
curl http://127.0.0.1:8787/ready
curl -H 'x-crowdtensor-observer-token: local-observer' \
  http://127.0.0.1:8787/metrics
```

The Compose file uses local demo tokens by default. Copy `.env.example` to `.env` to override them.

## Core Capabilities

- **Coordinator / Miner loop**: task claim, heartbeat, result submission, and bounded long-running Miner sessions.
- **Fault tolerance**: lease timeout requeue, stale result rejection, checkpoint recovery, and append-only event replay.
- **Runtime contracts**: deterministic CPU-only `diloco_train`, `cpu_lora_mock`, and `micro_transformer_lm` workloads.
- **Validation**: finite-value checks, shape checks, norm/loss gates, and optional deterministic replay audit.
- **Trust controls**: workload-scoped Miner scoring, quarantine, admin trust overrides, and redacted event tails.
- **Result traceability**: admin result ledger for accepted/rejected outcomes, validation, audit, model impact, and Miner score summaries.
- **Admission controls**: shared Miner token, observer token, admin token, per-Miner token registry, and hashed token configuration.
- **Remote Miner resilience**: startup `/ready` preflight, bounded retry for transient claim/heartbeat/result failures, result `idempotency_key`, and retry counters.
- **Browser experiments**: WebRTC tensor tunnel, browser Worker compute probe, and browser Miner bridge.
- **Acceptance pack**: repeatable smoke suite for runtime behavior and operator controls.

## Runtime Acceptance

Run the Alpha release gate first to verify package metadata, docs links, Docker/Compose shape, and CI wiring:

```bash
python3 scripts/release_gate.py --json
```

Run the offline security preflight before a controlled remote demo:

```bash
python3 scripts/security_preflight.py \
  --host 0.0.0.0 \
  --miner-token-registry state/miner_registry.json \
  --observer-token sha256:OBSERVER_DIGEST \
  --admin-token sha256:ADMIN_DIGEST \
  --json
```

Run the non-browser V1 acceptance pack from a normal Linux shell with localhost networking:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8910 \
  --report /tmp/crowdtensor_acceptance.json
```

It runs the core smoke checks sequentially:

- readiness/profile
- API contract
- chaos recovery
- trust quarantine
- replay audit
- operator control
- micro Transformer LM
- result idempotency
- result ledger
- Miner resilience
- Miner auth
- observer auth
- per-Miner registry auth
- hashed token auth
- outer optimizer contract
- compressed error-feedback delta transport

Browser-native checks are opt-in because they require Playwright and a Chromium-compatible browser:

```bash
python3 scripts/browser_acceptance_pack.py \
  --base-port 9310 \
  --report /tmp/crowdtensor_browser_acceptance.json
```

The browser acceptance pack runs the core browser checks: `webrtc_smoke.py`, `runtime_contract_check.py`, and `browser_miner_smoke.py`. CI uses `--allow-skip` so environments without Playwright or Chromium report a skipped browser pack instead of failing the whole job.

For the broader browser smoke set, use the runtime acceptance pack:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8930 \
  --include-browser \
  --report /tmp/crowdtensor_browser_acceptance.json
```

Some sandboxes block localhost client sockets. In that case, run unit tests inside the sandbox and run the acceptance pack in an unrestricted shell or CI job.

Run only the readiness/profile smoke:

```bash
python3 scripts/readiness_check.py --port 8890
```

Run only the API contract smoke:

```bash
python3 scripts/api_contract_check.py --port 8891
```

Run only the Miner resilience smoke:

```bash
python3 scripts/miner_resilience_check.py --port 8894
```

Run only the result idempotency smoke:

```bash
python3 scripts/result_idempotency_check.py --port 8896
```

Run only the result ledger smoke:

```bash
python3 scripts/result_ledger_check.py --port 8897
```

Run only the opt-in Nesterov outer optimizer smoke:

```bash
python3 scripts/outer_optimizer_check.py --port 8899
```

Run only the sign-compressed error-feedback transport smoke:

```bash
python3 scripts/compressed_error_feedback_check.py --port 8900
```

Run only the delta transport negotiation smoke:

```bash
python3 scripts/delta_transport_negotiation_check.py --port 8901
```

Run only the remote Miner invite/join smoke:

```bash
python3 scripts/remote_miner_join_check.py --port 8898
```

Run only the security preflight:

```bash
python3 scripts/security_preflight.py --json
```

## Security Model

CrowdTensorD has local-development admission controls, not a complete public network security model.

Coordinator supports:

- `--miner-token` / `CROWDTENSOR_MINER_TOKEN`
- `--miner-token-registry` / `CROWDTENSOR_MINER_TOKEN_REGISTRY`
- `--observer-token` / `CROWDTENSOR_OBSERVER_TOKEN`
- `--admin-token` / `CROWDTENSOR_ADMIN_TOKEN`

Token config values may be plaintext for local demos or `sha256:<digest>` verifiers for remote demos.

Miner startup checks `/ready` by default. Use `--skip-preflight` only for legacy Coordinators. Transient claim, heartbeat, and idempotent result upload failures are retried with `--max-request-attempts`; summaries include `request_retries`.

See [docs/security.md](docs/security.md) before exposing a Coordinator beyond localhost.

## Documentation

- [Quickstart](docs/quickstart.md)
- [API Reference](docs/api.md)
- [Remote Miner Onboarding](docs/remote-miner.md)
- [Architecture](docs/architecture.md)
- [Security](docs/security.md)
- [Operations](docs/operations.md)

## License

Apache-2.0. See [LICENSE](LICENSE).
