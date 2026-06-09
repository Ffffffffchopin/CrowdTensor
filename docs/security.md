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

Create registry entries without storing plaintext tokens:

```bash
python3 scripts/create_miner_invite.py \
  --registry state/miner_registry.json \
  --miner-id remote-linux-1 \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --claim-rate-limit 4 \
  --claim-rate-window-seconds 60
```

The invite output prints the plaintext token once for the remote Miner, while the registry stores only the `sha256:` verifier plus safe policy metadata. Positive `claim_rate_limit` / `claim_rate_window_seconds` values rate-limit claim events for that registered Miner before it can lease more work. See [Remote Miner Onboarding](remote-miner.md).

**Observer token**

`--observer-token` or `CROWDTENSOR_OBSERVER_TOKEN` protects read-only `/state` and `/metrics`. `/health`, `/version`, and `/ready` remain public for process health checks and non-sensitive runtime profile checks.

**Admin token**

`--admin-token` or `CROWDTENSOR_ADMIN_TOKEN` protects operator endpoints such as event-log tail and trust overrides.

For multi-operator demos, `--operator-token-registry` or
`CROWDTENSOR_OPERATOR_TOKEN_REGISTRY` can replace or supplement the legacy
admin token with per-operator roles. The registry stores token verifiers and
safe role metadata only:

```json
{
  "operators": [
    {
      "operator_id": "accounting-desk",
      "token": "sha256:ACCOUNTING_DIGEST",
      "roles": ["accounting"]
    },
    {
      "operator_id": "generate-desk",
      "token": "sha256:GENERATE_DIGEST",
      "roles": ["admin"],
      "session_policy": {
        "allowed_workloads": ["real_llm_sharded_infer"],
        "max_request_count": 2,
        "max_decode_steps": 1,
        "max_new_tokens": 8,
        "rate_limit": 30,
        "rate_window_seconds": 60
      }
    }
  ]
}
```

`accounting` can read `/admin/accounting` and `/admin/settlement`; `auditor`
can read event/result/stream audit views; `admin` and `owner` can use all admin
endpoints. Optional `session_policy` values restrict an admin/owner operator's
`/admin/inference-sessions` workload types, request sizes, token/decode bounds,
and create rate. `/ready` exposes only
`crowdtensor_operator_registry_summary_v1` operator IDs, labels, enabled flags,
roles, and safe session policy limits, never plaintext tokens.

To reduce request abuse on shared Coordinators, start the product Coordinator
with `--inference-session-rate-limit` plus
`--inference-session-rate-window-seconds`. The limit applies per legacy admin
or per operator-registry subject on `/admin/inference-sessions`; blocked creates
return `429` and append a safe `control_plane_blocked` audit event. Per-operator
`session_policy` blocks use the same audit event with
`operator_session_policy_*` reasons.

Accepted work created by `/admin/inference-sessions` carries the same safe admin
subject into `/admin/results`, `/admin/accounting`, and `/admin/settlement` as
`created_by_subject`, and accounting/settlement responses also expose
`created_by_subject_totals` grouped by subject/workload. Values are labels such
as `legacy-admin` or `operator:<operator_id>` for audit, chargeback, and draft
settlement attribution. The `created_by_subject` query parameter is an exact
label filter for accounting/settlement export, not a token lookup; plaintext
admin/operator tokens remain private and are never included in those rows or
totals.

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

Use `scripts/remote_miner_join_check.py` to validate registry-backed Miner onboarding in a local smoke test before sending invite commands to a remote machine.

## Support Bundle Hygiene

Use `scripts/support_bundle.py --json-out /tmp/crowdtensor_support_bundle.json` for issue reports and remote-demo troubleshooting. The Support Bundle redacts token, lease, idempotency, weight, and delta-shaped fields before writing output.

Do not upload raw `state/` directories, token registry files, shell history, raw `/state` output, or unredacted Coordinator logs to public issues.

Run the offline security preflight before binding beyond loopback:

```bash
python3 scripts/security_preflight.py \
  --host 0.0.0.0 \
  --miner-token-registry state/miner_registry.json \
  --observer-token sha256:OBSERVER_DIGEST \
  --admin-token sha256:ADMIN_DIGEST \
  --json
```

The preflight reads CLI/env-style configuration and the registry file without contacting a running Coordinator. It fails high-risk remote demo mistakes, including missing Miner/observer/admin auth, local demo tokens on a remote bind, invalid registry JSON, and plaintext registry token verifiers. Add `--strict` when warnings should also fail CI.

## Replay Audit

`--replay-audit` verifies deterministic workload outputs against claim-time state. For `sign_compressed` DiLoCo results, it recomputes the dense delta, applies the same deterministic sign compression/decode contract, and compares decoded deltas. `sign_compressed_ef` is deliberately rejected by replay audit with `error_feedback_replay_unsupported` because the error-feedback residual buffer is Miner-local mutable state. This is a control-plane integrity check for supported toy workloads, not a cryptographic proof of useful external compute.
