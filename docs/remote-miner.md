# Remote Miner Onboarding

This guide connects a remote Python Miner to a controlled CrowdTensorD Coordinator demo. It does not make the Coordinator safe for direct public-internet exposure.

Use HTTPS, a VPN, or a private network for any remote demo. Miner tokens are sent by clients as plaintext headers, even when the Coordinator stores only `sha256:` token verifiers.

## 1. Create a Miner Registry Entry

On the Coordinator host:

```bash
python3 scripts/create_miner_invite.py \
  --registry state/miner_registry.json \
  --miner-id remote-linux-1 \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --label "first remote linux miner"
```

The registry stores only the hashed verifier:

```json
{
  "miners": [
    {
      "enabled": true,
      "label": "first remote linux miner",
      "miner_id": "remote-linux-1",
      "token": "sha256:..."
    }
  ]
}
```

The script prints a one-time plaintext `CROWDTENSOR_MINER_TOKEN` and a remote `crowdtensor-miner` command. Keep the plaintext token out of the repository.

To rotate or replace an existing Miner token:

```bash
python3 scripts/create_miner_invite.py \
  --registry state/miner_registry.json \
  --miner-id remote-linux-1 \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --replace
```

## 2. Start Coordinator With Registry

Run the offline security preflight before binding beyond loopback:

```bash
python3 scripts/security_preflight.py \
  --host 0.0.0.0 \
  --miner-token-registry state/miner_registry.json \
  --observer-token sha256:OBSERVER_DIGEST \
  --admin-token sha256:ADMIN_DIGEST \
  --json
```

```bash
crowdtensord \
  --host 0.0.0.0 \
  --port 8787 \
  --state-dir state \
  --miner-token-registry state/miner_registry.json \
  --observer-token sha256:OBSERVER_DIGEST \
  --admin-token sha256:ADMIN_DIGEST
```

For controlled demos, put TLS or a VPN in front of the Coordinator. Do not expose the admin token path to the open internet.

Use `--strict` with `scripts/security_preflight.py` when warning-level findings should block a CI or release rehearsal.

## 3. Run the Remote Miner

On the remote machine:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

Use the invite output:

```bash
export CROWDTENSOR_MINER_TOKEN=PASTE_INVITE_TOKEN
crowdtensor-miner \
  --coordinator https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --max-tasks 1
```

For a longer controlled run:

```bash
crowdtensor-miner \
  --coordinator https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --max-tasks 10 \
  --max-request-attempts 5 \
  --compute-seconds 0.2
```

## 4. Verify Join Readiness

Local smoke for the registry invite flow:

```bash
python3 scripts/remote_miner_join_check.py --port 8898
```

Long-running remote-style Miner smoke:

```bash
python3 scripts/remote_miner_readiness_check.py \
  --port 8899 \
  --miner-token local-miner \
  --observer-token local-observer
```

This readiness smoke now runs all default Python Miner workloads: `diloco_train`, `cpu_lora_mock`, `micro_transformer_lm`, and `model_bundle_lm`.

The default runtime acceptance pack keeps the remote Miner check opt-in:

```bash
python3 scripts/runtime_acceptance_pack.py \
  --base-port 8920 \
  --include-remote-miner \
  --miner-token local-miner \
  --observer-token local-observer \
  --report /tmp/crowdtensor_remote_acceptance.json
```

## Troubleshooting

**401 invalid miner token**

Confirm the remote `miner_id` exactly matches the registry entry and the remote machine is using the plaintext token printed by the invite script.

**miner token is disabled**

The registry entry exists with `"enabled": false`. Re-enable it or generate a new invite.

**503 no compatible queued task available**

The Miner capabilities do not match queued lanes, or the Miner is blocked/quarantined for the workload. Check `/ready`, `/state`, and `GET /admin/results`.

**Coordinator unreachable or preflight timeout**

Check DNS, firewall rules, TLS proxy, and that `/ready` is reachable from the remote host. Use `--skip-preflight` only for legacy Coordinators; it should not be needed for current CrowdTensorD.
