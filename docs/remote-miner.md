# Remote Miner Onboarding

This guide connects a remote Python Miner to a controlled CrowdTensorD Coordinator demo. It does not make the Coordinator safe for direct public-internet exposure.

Use HTTPS, a VPN, or a private network for any remote demo. Miner tokens are sent by clients as plaintext headers, even when the Coordinator stores only `sha256:` token verifiers.

## 1. Create a Miner Registry Entry

For the recommended high-level two-machine home-compute demo, start with `crowdtensor remote-demo prepare`. It creates the registry, `operator.private.env`, `miner.private.env`, public runbook, and `remote_home_compute_demo_v1` summary in one output directory:

```bash
crowdtensor remote-demo prepare \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --scenario-id route-baseline \
  --output-dir dist/remote-home-compute \
  --json
```

After the generated Coordinator and remote Miner commands are running, verify the read-only `model_bundle_infer` session:

```bash
. dist/remote-home-compute/operator.private.env
crowdtensor remote-demo doctor \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" \
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \
  --scenario-id route-baseline \
  --output-dir dist/remote-home-compute \
  --json

crowdtensor remote-demo verify \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" \
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \
  --scenario-id route-baseline \
  --output-dir dist/remote-home-compute \
  --json

crowdtensor remote-demo collect \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" \
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \
  --scenario-id route-baseline \
  --output-dir dist/remote-home-compute \
  --json
```

The wrapper uses `scripts/remote_home_compute_demo_pack.py` and CI validates the local stand-in with `scripts/remote_home_compute_demo_check.py`. The `remote_home_compute_demo_v1` artifact links the underlying `remote_demo_runbook_v1`, `POST /admin/inference-sessions` acceptance, `remote_compute_evidence_v1`, `remote_demo_observability_v1`, `model_bundle_infer`, and `remote_python_model_bundle_infer` evidence without exposing plaintext tokens. `remote-demo doctor` writes `remote_home_compute_doctor_v1`, `remote-demo collect` writes `remote_home_compute_collect_v1`, and `remote-demo clean` writes `remote_home_compute_cleanup_v1` while defaulting to dry-run cleanup. It is not production Swarm Inference and not P2P routing.

For an operator-owned local LLM runtime on the Miner host, use the explicit external LLM workload:

```bash
crowdtensor remote-demo prepare \
  --workload external-llm \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --mock \
  --output-dir dist/remote-home-compute-llm \
  --json

crowdtensor remote-demo verify \
  --workload external-llm \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" \
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \
  --mock \
  --output-dir dist/remote-home-compute-llm \
  --json

crowdtensor remote-demo collect \
  --workload external-llm \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" \
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \
  --mock \
  --output-dir dist/remote-home-compute-llm \
  --json
```

This queues read-only `external_llm_infer` through `POST /admin/inference-sessions`, verifies `remote_python_external_llm_infer`, and collects `remote_external_llm_evidence_v1` plus `remote_external_llm_observability_v1` through `scripts/remote_external_llm_evidence_pack.py`. The public artifacts do not include raw prompts, `output_text`, runtime URL, API key, lease token, or idempotency material. Replace `--mock` with `--llm-runtime-cmd` or `--llm-runtime-url` only when the operator owns the runtime. It is fixed-prompt runtime evidence, not public arbitrary prompt serving.

For the lower-level safe two-machine runbook that creates only the registry, private env files, and public operator commands:

```bash
crowdtensor remote-runbook \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --output-dir dist/remote-demo \
  --json
```

The `crowdtensor/cli.py` wrapper emits `remote_runbook_cli_v1` and delegates to `scripts/remote_demo_runbook_pack.py`. The `remote_demo_runbook_v1` output writes `operator.private.env` for observer/admin collection and `miner.private.env` for the remote Miner. Both files are created with `0600` permissions. Only copy `miner.private.env` to the remote Miner host. The public JSON/Markdown includes the `crowdtensord` and `crowdtensor-miner` commands, the `model_bundle_infer` lane, and `remote_compute_evidence_pack.py --mode collect`, but it does not include plaintext tokens. `scripts/remote_demo_runbook_check.py` validates this path in CI.

When rerunning the same runbook directory with the same `miner_id`, add `--replace` so the generated registry rotates that Miner entry instead of failing on the existing invite.

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

Safe, shareable remote-compute evidence:

```bash
python3 scripts/remote_compute_evidence_pack.py \
  --port 8912 \
  --request-count 4 \
  --json-out /tmp/crowdtensor_remote_evidence.json \
  --markdown-out /tmp/crowdtensor_remote_evidence.md
```

The default `local-loopback` mode is a CI-safe remote-style rehearsal: it creates a hashed registry invite, starts a registry-backed Coordinator, runs the invited Python Miner, and emits `remote_compute_evidence_v1` plus `remote_compute_observability_v1` for the read-only `model_bundle_infer` route `remote_python_model_bundle_infer`. For a real two-machine demo after the remote Miner has completed work, collect from the running Coordinator:

```bash
python3 scripts/remote_compute_evidence_pack.py \
  --mode collect \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --observer-token OBSERVER_TOKEN \
  --admin-token ADMIN_TOKEN \
  --json-out /tmp/crowdtensor_remote_evidence.json
```

`scripts/remote_compute_evidence_check.py` validates the local-loopback path. The runtime acceptance pack can opt in with `--include-remote-evidence`.

After running the safe two-machine runbook on real Coordinator/Miner hosts, use the acceptance pack to create a bounded read-only session, wait for the returned `task_id`, and collect shareable artifacts:

```bash
crowdtensor remote-acceptance \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id remote-linux-1 \
  --observer-token "$CROWDTENSOR_OBSERVER_TOKEN" \
  --admin-token "$CROWDTENSOR_ADMIN_TOKEN" \
  --create-session \
  --scenario-id route-baseline \
  --output-dir dist/remote-demo-acceptance \
  --json
```

The `crowdtensor/cli.py` wrapper emits `remote_acceptance_cli_v1`, applies token redaction to captured output, and delegates to `scripts/remote_demo_acceptance_pack.py`. With `--create-session`, the acceptance pack calls `POST /admin/inference-sessions`, queues one read-only `model_bundle_infer` task for the selected `model_bundle_inference_scenario_v1`, and waits for the returned `task_id` so stale accepted rows from earlier runs cannot satisfy the demo. The `remote_demo_acceptance_v1` output includes the top-level acceptance summary, `remote_compute_evidence_v1`, `remote_demo_observability_v1`, scenario match status, a redacted `support_bundle`, and stable `diagnosis_codes` for operator triage: `coordinator_unreachable`, `observer_auth_failed`, `admin_auth_failed`, `session_create_failed`, `miner_not_seen`, `task_lane_missing`, `workload_not_advertised`, `no_accepted_result`, `validation_failed`, `request_count_mismatch`, `artifact_collection_failed`, and `acceptance_ready`. This is not production Swarm Inference and not P2P routing. `scripts/remote_demo_acceptance_check.py` validates the local stand-in path in CI.

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
