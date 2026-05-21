# Operations

This document collects common commands for local Alpha operation.

## Start Coordinator

```bash
crowdtensord \
  --host 127.0.0.1 \
  --port 8787 \
  --state-dir state
```

With local tokens:

```bash
crowdtensord \
  --host 127.0.0.1 \
  --port 8787 \
  --state-dir state \
  --miner-token local-miner \
  --observer-token local-observer \
  --admin-token local-admin
```

To avoid storing a usable token directly in Coordinator config, generate a hashed token verifier:

```bash
python3 scripts/hash_token.py local-miner
```

Then pass the printed `sha256:` value to `--miner-token`, `--observer-token`, `--admin-token`, or a per-Miner registry entry. Clients still send the original token.

## Run Miner

```bash
CROWDTENSOR_MINER_TOKEN=local-miner crowdtensor-miner \
  --coordinator http://127.0.0.1:8787 \
  --miner-id local-1 \
  --max-tasks 10 \
  --compute-seconds 0.2
```

Useful flags:

- `--once`: process one task and exit
- `--max-tasks N`: stop after N accepted tasks
- `--max-runtime-seconds N`: stop after a wall-clock budget
- `--heartbeat-interval N`: tune heartbeat cadence
- `--skip-preflight`: skip the startup `/ready` compatibility check
- `--max-request-attempts N`: retry transient `/ready`, claim, heartbeat, and idempotent result upload failures
- `--idle-sleep N`: sleep between failed or unavailable claims

The Miner summary includes `request_retries` and `preflight_failures` so operators can spot unstable links without parsing stderr.

Result uploads include an `idempotency_key`, so a lost response can be retried without applying the same model update twice.

## Health and Metrics

```bash
curl http://127.0.0.1:8787/health
```

```bash
curl http://127.0.0.1:8787/version
```

```bash
curl http://127.0.0.1:8787/ready
```

```bash
curl -H 'x-crowdtensor-observer-token: local-observer' \
  http://127.0.0.1:8787/state
```

```bash
curl -H 'x-crowdtensor-observer-token: local-observer' \
  http://127.0.0.1:8787/metrics
```

`/metrics` is aggregate-only and avoids lease tokens, task payloads, Miner IDs, and raw Miner metadata.

Admin result ledger:

```bash
curl -H 'x-crowdtensor-admin-token: local-admin' \
  'http://127.0.0.1:8787/admin/results?status=rejected&limit=20'
```

`GET /admin/results` is the safest operator view for result traceability. It includes validation, replay audit, model impact, and Miner workload score summaries, but avoids raw lease tokens, idempotency material, full result responses, and tensor deltas.

## Acceptance Checks

Alpha release gate:

```bash
python3 scripts/release_gate.py --json
```

Security preflight:

```bash
python3 scripts/security_preflight.py --json
```

Remote demo preflight:

```bash
python3 scripts/security_preflight.py \
  --host 0.0.0.0 \
  --miner-token-registry state/miner_registry.json \
  --observer-token sha256:OBSERVER_DIGEST \
  --admin-token sha256:ADMIN_DIGEST \
  --json
```

Readiness/profile smoke:

```bash
python3 scripts/readiness_check.py --port 8890
```

API contract smoke:

```bash
python3 scripts/api_contract_check.py --port 8891
```

Miner resilience smoke:

```bash
python3 scripts/miner_resilience_check.py --port 8894
```

Result idempotency smoke:

```bash
python3 scripts/result_idempotency_check.py --port 8896
```

Result ledger smoke:

```bash
python3 scripts/result_ledger_check.py --port 8897
```

Remote Miner invite/join smoke:

```bash
python3 scripts/remote_miner_join_check.py --port 8898
```

Unit tests:

```bash
python3 -m unittest discover -s tests -v
```

Default runtime acceptance:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8910 \
  --report /tmp/crowdtensor_acceptance.json
```

Remote-style Miner readiness:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8920 \
  --include-remote-miner \
  --miner-token local-miner \
  --observer-token local-observer \
  --report /tmp/crowdtensor_remote_acceptance.json
```

`--miner-token` and `--observer-token` are passed only to checks that explicitly support shared auth env vars. Auth-specific smoke tests keep their own local tokens so they can validate rejection paths deterministically.

Browser acceptance:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8930 \
  --include-browser \
  --report /tmp/crowdtensor_browser_acceptance.json
```

## Troubleshooting

**Address already in use**

Change `--port` or `--base-port`. The acceptance pack consumes a range of ports.

**Operation not permitted in sandbox**

Some restricted environments block localhost client sockets. Run unit tests there, then run acceptance checks on a normal shell or CI host.

**401 invalid miner token**

Confirm the Coordinator and Miner use the same `CROWDTENSOR_MINER_TOKEN`, or use the exact per-Miner token from the registry.

For registry-backed remote Miners, generate or rotate the entry with `scripts/create_miner_invite.py` and confirm the remote `--miner-id` matches the registry entry.

**401 invalid observer token**

Pass `x-crowdtensor-observer-token` when reading `/state` or `/metrics`.

**Security preflight fails**

Fix `error` findings before a remote demo. Typical causes are binding to `0.0.0.0` without Miner, observer, or admin tokens, using `local-*` demo tokens on a remote bind, or storing plaintext tokens in `--miner-token-registry`. Use `--strict` when warning-level findings should also block CI.

**503 no compatible queued task available**

The Miner capabilities do not match queued lanes. Check `--task-lane` values and the capabilities sent by the Miner/browser.

**Playwright browser not found**

Install browser dependencies or pass `--browser /path/to/chrome` to the browser smoke scripts.
