# Security

CrowdTensorD Alpha includes local-development admission controls. They are useful for smoke tests and controlled demos, but they are not a complete public-internet security model.

Token configuration values can be plaintext for local demos or hashed token verifiers in the form `sha256:<64 hex digest>`. Clients still send the original token in headers; the Coordinator compares it against the configured plaintext or hashed token value.

## Token Surfaces

**Miner token**

`--miner-token` or `CROWDTENSOR_MINER_TOKEN` protects:

- `POST /tasks/claim`
- `POST /tasks/{task_id}/heartbeat`
- `POST /tasks/{task_id}/result`

Python Miners pass it with `--miner-token` or `CROWDTENSOR_MINER_TOKEN`.

Generate a hashed token verifier:

```bash
python3 scripts/hash_token.py local-miner
```

Use the printed `sha256:` value in Coordinator configuration:

```bash
crowdtensord \
  --host 127.0.0.1 \
  --port 8787 \
  --state-dir state \
  --miner-token sha256:REPLACE_WITH_DIGEST
```

The Miner still sends the original token:

```bash
CROWDTENSOR_MINER_TOKEN=local-miner crowdtensor-miner \
  --coordinator http://127.0.0.1:8787 \
  --miner-id local-hashed-token \
  --once
```

**Per-Miner registry**

`--miner-token-registry` or `CROWDTENSOR_MINER_TOKEN_REGISTRY` points to JSON like:

```json
{
  "miners": [
    {"miner_id": "kaggle-session-1", "token": "secret-1", "enabled": true, "label": "kaggle test"}
  ]
}
```

The `token` field may also use `sha256:<digest>`.

If a claiming `miner_id` exists in the registry, it must use its own token. Disabled miners are rejected. Unknown miners can still use the shared fallback token when configured.

**Observer token**

`--observer-token` or `CROWDTENSOR_OBSERVER_TOKEN` protects read-only `/state` and `/metrics`. `/health`, `/version`, and `/ready` remain public for process health checks and non-sensitive runtime profile checks.

**Admin token**

`--admin-token` or `CROWDTENSOR_ADMIN_TOKEN` protects operator endpoints such as event-log tail and trust overrides.

## What Is Protected

The current controls reduce accidental public access and keep local demo Miners separated from read-only observers and admins.

Hashed token configuration reduces the blast radius of accidentally leaking a config file because it avoids storing the usable token directly. It does not protect the token while it is sent over the network; use HTTPS for any remote demo.

They do not provide:

- TLS termination
- token rotation
- request signatures
- replay-resistant nonce checks
- account management
- RBAC
- Sybil resistance
- staking or economic penalties
- hardware attestation

## Public Exposure Guidance

Do not expose a Coordinator directly to the public internet.

For controlled remote demos:

- run behind HTTPS
- use long random tokens
- store Coordinator token config as `sha256:` verifiers where possible
- separate Miner, observer, and admin tokens
- prefer per-Miner registry tokens over a broad shared token
- bind admin access to a private network or VPN
- monitor `/metrics` and event tails
- rotate demo tokens after every test

## Replay Audit

`--replay-audit` verifies deterministic workload outputs against claim-time state. This is a control-plane integrity check for supported toy workloads, not a cryptographic proof of useful external compute.
