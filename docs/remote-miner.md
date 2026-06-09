# Remote Miner Onboarding

This guide connects a remote Python Miner to a controlled CrowdTensorD Coordinator demo. It does not make the Coordinator safe for direct public-internet exposure.

Use HTTPS, a VPN, or a private network for any remote demo. Miner tokens are sent by clients as plaintext headers, even when the Coordinator stores only `sha256:` token verifiers.

For the product two-stage path, start with `crowdtensor swarm-bootstrap`. It
emits `crowdtensor_swarm_bootstrap_v1`, writes a private operator registry,
private Miner registry, coordinator/operator private env files, one operator
invite, stage0/stage1 Miner packages, executable `start_control_plane.sh`,
`install_operator.sh`, `operator_quickstart.sh`, optional `start_tunnel.sh`, `tunnel_doctor.sh`, `start_discovery.sh`,
`start_coordinator.sh`, stage
`install.sh` / `doctor.sh` / `check_join.sh` / `support_bundle.sh` / `join.sh` files, private `miner.join-code.txt` files,
private `stage0.miner-package.tar.gz` / `stage1.miner-package.tar.gz` archives,
matching `stage0.run-miner.sh` / `stage1.run-miner.sh` helpers,
`stage0.handoff.sha256` / `stage1.handoff.sha256`,
`stage_handoff_manifest.json`,
`handoff_doctor.sh`, `ready_for_handoff.sh`, `check_route.sh`, `operator_status.sh`,
`verify_bootstrap.sh`, generation scripts, and
`SWARM_BOOTSTRAP.md`.
If the Coordinator has no directly reachable public address, pass both a Miner-facing
`--coordinator-url` and a private `--tunnel-command`; the raw tunnel command is
stored only in `private/tunnel.private.env` and is started by `start_control_plane.sh`.
For common static tunnel setups, `--tunnel-provider ngrok` can generate an
`ngrok http <port> --url <coordinator-url>` command from the Miner-facing
`--coordinator-url`, while `--tunnel-provider cloudflare-token` generates a
`cloudflared tunnel run --token "$..."` command that references
`--tunnel-token-env` and expects the Cloudflare tunnel ingress to already route
to the local Coordinator upstream. These provider templates are only private
command generation; they do not register domains, create tunnel accounts, or
make quick/random tunnel URLs safe for Miner invites.
Run `tunnel_doctor.sh` or `crowdtensor swarm-tunnel-doctor --output-dir ...
--expect-remote-miners` first to emit `crowdtensor_swarm_tunnel_doctor_v1` and
`tunnel_doctor.json` checks for the private tunnel env, provider binary,
control-plane launcher, and Miner-facing URL without starting the tunnel.
`crowdtensor coordinator-route --coordinator-url ... --expect-remote-miners`
emits public-safe `join_options`, `recommended_join_option`, and
`recommended_setup_command` templates for public HTTPS/reverse-proxy, tunnel,
VPN/LAN, or explicit port-forwarding before any Miner package is shared.
Keep the generated operator invite and operator env on the operator host, use
the coordinator env only for the Coordinator process, and run
`operator_quickstart.sh` as the recommended Operator path. It runs
`install_operator.sh` when `crowdtensor`, `crowdtensord`, or
`crowdtensor-miner` is missing, starts `start_control_plane.sh` in the
background, writes `run/control_plane.pid` and `logs/control_plane.log`, waits
for `check_route.sh --check-ready`, and then runs `ready_for_handoff.sh` to
chain `tunnel_doctor.sh`, `verify_bootstrap.sh`, and `handoff_doctor.sh`.
`CROWDTENSOR_QUICKSTART_WAIT_SECONDS` adjusts the route wait and
`CROWDTENSOR_QUICKSTART_SKIP_INSTALL=1` skips installation for externally
managed runtimes. You can still run `install_operator.sh`,
`start_control_plane.sh`, `ready_for_handoff.sh`, `verify_bootstrap.sh`, and
`check_route.sh` separately to debug the advertised Coordinator URL and live
no-claim admission path,
run `operator_status.sh` on the operator host for read-only Coordinator,
trust, accounting, and settlement status,
and copy only the matching private stage archive plus `stageX.run-miner.sh` and
`stageX.handoff.sha256` to each Miner host. The runner verifies the checksum, validates and extracts the archive, and supports the recommended first run `./stageX.run-miner.sh --quickstart` (`CrowdTensor Miner quickstart`), which installs the local runtime, writes diagnostics, checks admission, and starts the Miner. Use `--setup` then `--start` for manual troubleshooting, `CROWDTENSOR_MINER_QUICKSTART_SKIP_INSTALL=1` when the Miner runtime is managed externally, `--install --dry-run` to preview installation, and `--doctor`, `--check-only`, or `--support-bundle` for troubleshooting. Stage `install.sh` creates
`.crowdtensor-venv` with the default `[hf]` runtime when `crowdtensor` is not
already on PATH. Stage `doctor.sh` writes `miner_support_bundle.json` and checks admission without starting the
Miner. Stage `check_join.sh` uses the private invite code file to verify
Coordinator reachability and admission without starting the Miner; stage `join.sh` uses the same path with
`--run`, while `miner.invite.json` remains private compatibility material. When
route or admission checks fail, stage `support_bundle.sh` writes safe
`miner_support_bundle.json` diagnostics to share instead of raw
`miner.join-code.txt` or `miner.invite.json`. The bundle includes
`crowdtensor_miner_local_environment_v1` with `local_environment_ready`,
`crowdtensor` CLI, `sha256sum`, Python, and optional torch/CUDA probes so the
Miner host can report setup failures without leaking tokens. `install_operator.sh`
creates `.crowdtensor-operator-venv` by default, can be relocated with
`CROWDTENSOR_OPERATOR_VENV`, checks `crowdtensor`, `crowdtensord`, and
`crowdtensor-miner`, and supports `--dry-run`, `CROWDTENSOR_INSTALL_SPEC`, and
`CROWDTENSOR_INSTALL_SOURCE`. It does not read `operator.private.env`, stage
invites, or join-code files, and it does not start services; generated
Coordinator/Operator-side scripts prefer that venv when present. If bootstrap is run
with `--peer-bootstrap`, the private invite embeds
`crowdtensor_miner_join_discovery_v1` so the stage join command can discover the
Coordinator through P2P-lite without the Miner operator manually passing
`--peer-bootstrap`. The Coordinator URL still must be reachable
by public HTTPS, VPN, trusted LAN, or tunnel; with `--expect-remote-miners`,
bootstrap fails before creating registry or invite files when the URL is
local-only. Before copying stage packages, run
`crowdtensor swarm-bootstrap-check --output-dir state/swarm-bootstrap --expect-remote-miners`;
it emits `crowdtensor_swarm_bootstrap_check_v1` and verifies required files,
`0600` private env/invite/join-code permissions, `0700` helper scripts, hashed
registries, env separation, stage invite Coordinator URL consistency,
non-local-only remote route readiness via `coordinator_url_remote_route_ready`,
stage join-code consistency, `stage_check_join_scripts_ready`,
`stage_install_scripts_ready`,
`stage_doctor_scripts_ready`,
`stage_support_bundle_scripts_ready`, `stage_package_archives_ready`,
`stage_archive_runner_scripts_ready`, `stage_setup_start_runner_ready`,
`stage_handoff_checksums_ready`, `tunnel_doctor_script_ready`,
`ready_for_handoff_script_ready`, `operator_install_script_ready`,
`operator_quickstart_script_ready`,
`operator_scripts_use_operator_venv`, and
plaintext token leakage in scripts or public Markdown.
After starting the Coordinator, rerun the same command with `--check-coordinator`
to call `/ready` and match both stage invites against the redacted registry
policy, or with `--check-admission` to also call token-backed `/tasks/preflight`
for both stage invites without claiming tasks. Wait for
`bootstrap_handoff.ready_to_copy_stage_packages=true`; until then the package is
generated but not live-verified for remote Miner handoff.
A clean live check reports `swarm_bootstrap_live_preflight_ready`.
For the Operator handoff summary, run `handoff_doctor.sh` or
`crowdtensor swarm-handoff-doctor`; it emits
`crowdtensor_swarm_handoff_doctor_v1` and writes public-safe
`handoff_doctor.json` / `handoff_doctor.md` with blockers and the exact stage
archive, runner, and checksum files to copy.
Bootstrap is not NAT traversal.

For a local maintainer check of the CPU inference Beta path before involving a second machine, run:

```bash
crowdtensor cpu-infer --mode remote-loopback --workload model-bundle --json
```

The `cpu_inference_beta_v1` aggregate report uses `scripts/cpu_inference_beta_pack.py` and is checked by `scripts/cpu_inference_beta_check.py`. The `--mode remote-loopback` path validates the same `remote-demo` contract on localhost; `--mode remote-existing` wraps an already running two-machine `remote-demo doctor/verify/collect` flow with explicit operator tokens. This remains CPU-only, read-only, not production Swarm Inference, and not P2P.

For the top-level CPU Inference Beta RC, run:

```bash
crowdtensor cpu-infer --mode beta-rc --json
```

The `cpu_inference_beta_rc_v1` report is built by `scripts/cpu_inference_beta_rc_pack.py` and validated by `scripts/cpu_inference_beta_rc_check.py`. It includes local CPU inference, remote-loopback inference, the Real two-machine CPU inference Beta rehearsal, Kaggle Remote Miner Beta artifact preparation, `miner_join_pack_v1`, and `demo_manifest_v1`. A ready report includes `cpu_inference_beta_rc_ready`, `local_cpu_inference_ready`, `remote_loopback_ready`, `two_machine_rehearsal_ready`, `kaggle_remote_miner_artifacts_ready`, `miner_join_pack_ready`, and `cpu_miner_beta_ready`. If you have already completed a live Kaggle run, pass `--kaggle-real-runtime-report dist/kaggle-real-runtime/kaggle_real_runtime_acceptance.json` to import `kaggle_real_runtime_acceptance_v1` and surface `real_runtime_evidence_ready`. It remains CPU-only, read-only, not production Swarm Inference, not P2P, and not a GPU/TPU workload path.

For the Real two-machine CPU inference Beta aggregate check used by maintainers and CI:

```bash
python3 scripts/remote_two_machine_beta_check.py --workload all --base-port 9050
```

The report schema is `remote_two_machine_beta_check_v1`. It runs local loopback stand-ins for the real Coordinator host and Miner host, requires `remote_two_machine_inference_ready`, `remote_two_machine_external_llm_ready`, and `remote_two_machine_beta_ready`, and checks token/redaction boundaries. It is a rehearsal for the 15-minute two-machine CPU inference Beta; real two-machine use still requires operator-provided TLS, VPN, tunnel, or trusted network. This is task-level remote CPU inference, not model sharding, not P2P, and not production Swarm Inference.

## 1. Create a Miner Registry Entry

For the product-shaped `serve` / `join` / `generate` path, the Coordinator host
can also create a private Miner join invite. The Coordinator URL must be
reachable from the Miner: use a public HTTPS endpoint, a VPN address, a trusted
private network, or a tunnel such as Cloudflare Tunnel, ngrok, frp, or a reverse
proxy. The Miner does not need to expose a public port because it connects
outbound to the Coordinator.

```bash
python scripts/create_miner_invite.py \
  --registry dist/public-swarm/miner_registry.json \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id stage0-gpu-1 \
  --stage stage0 \
  --backend cuda \
  --trust-tier probation \
  --quota-task-limit 25 \
  --claim-rate-limit 4 \
  --claim-rate-window-seconds 60 \
  --reward-account acct-stage0-private \
  --invite-file dist/public-swarm/stage0-gpu-1.invite.json \
  --json
```

Copy only the generated `*.invite.json` file to the Miner host. Before running
the Miner, check that the Coordinator URL is reachable from that host:

```bash
crowdtensor join --invite-file stage0-gpu-1.invite.json --check-admission --expect-remote-coordinator --json
```

If the Coordinator is behind a tunnel, VPN, or reverse proxy, start it with the
Miner-facing URL instead of relying on `--public-host` alone:

```bash
crowdtensor serve \
  --profile gpu-generation \
  --coordinator-public-url https://YOUR-TUNNEL.example \
  --expect-remote-miners \
  --run
```

The Coordinator process can still bind locally; `--coordinator-public-url` is
the advertised join/generate/P2P URL for Miner hosts. It is not NAT traversal or
a relay service, so the named tunnel, VPN, DNS, firewall, or reverse proxy must
already route to the Coordinator.

The preflight first calls the public `/ready` endpoint without sending the Miner
token, compares the invite's Miner ID, stage, backend, model, read-only workload,
quota, and claim-rate metadata with the Coordinator's redacted
`miner_policy_summary`, and then calls token-backed `/tasks/preflight` without
claiming work. It does not lease a task and does not submit a generation
request. A blocked report with `join_coordinator_unreachable` means the
Coordinator entry URL, DNS, firewall, VPN, tunnel, or reverse proxy must be
fixed before the Miner can join.
`coordinator_remote_route_required` means the invite or `--coordinator-url`
still points to a local-only address such as `127.0.0.1`; regenerate the invite
with the public/tunnel/VPN/LAN URL.
`join_invite_policy_miner_missing` or `join_invite_policy_mismatch` means the
Coordinator did not load the matching `miner_registry.json`, loaded an older
registry, or the invite was generated for a different Coordinator.
`join_admission_token_rejected` or `join_policy_*` means the token or advertised
stage/backend/model/capability would be rejected by claim-time policy. After the
preflight is ready, run:

```bash
crowdtensor join --invite-file stage0-gpu-1.invite.json --run
```

`--invite` also accepts the base64url invite code printed by the script, but the
file form is preferred because the invite contains the plaintext Miner token and
may otherwise end up in shell history. The Coordinator registry stores only the
token verifier plus `join_policy` metadata. `/ready` exposes a redacted
`miner_policy_summary` with the invited stage, backend, trust tier, quota limit,
claim-rate limit, claim-rate window, and reward-account presence, but never the
plaintext token or reward account.
For the higher-level two-stage bootstrap path, `crowdtensor swarm-bootstrap`
also accepts `--stage0-reward-account` and `--stage1-reward-account` to bind
private Beta accounting metadata to the generated stage invites while keeping
public bootstrap, check, support, and handoff reports limited to
reward-account presence.
`swarm-bootstrap-check` also verifies `operator_status_script_ready` so the
operator status helper stays executable and sources only the private operator
env instead of embedding plaintext credentials.
It verifies `check_route_script_ready` so the route helper stays executable and
continues to use the package's public Coordinator URL without embedding
credentials.
At claim time, the Coordinator enforces the invite's workload, stage, backend,
and model scope before the Miner can lease work; blocked claims are recorded in
the normal `blocked_claims` audit counters. A positive `quota_task_limit`
caps the number of leased, accepted, and rejected claims for that registered
Miner. A positive `claim_rate_limit` plus `claim_rate_window_seconds` rate
limits claim events for that registered Miner and returns `429` with
`join_policy_rate_limited` when the window is exhausted. Reward fields are still
accounting metadata for the current Beta; they are not production billing,
staking, or automatic economic settlement.

For the recommended high-level two-machine home-compute demo, start with `crowdtensor remote-demo prepare`. The Coordinator host runs `crowdtensord`; the Miner host runs `crowdtensor-miner`; the operator keeps `operator.private.env`; only `miner.private.env`, `miner_join.sh`, and `MINER_JOIN.md` are copied to the Miner host. It creates the registry, private env files, public runbook, `miner_join_pack_v1`, and `remote_home_compute_demo_v1` summary in one output directory:

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

The wrapper uses `scripts/remote_home_compute_demo_pack.py` and CI validates the local stand-in with `scripts/remote_home_compute_demo_check.py` plus `scripts/remote_two_machine_beta_check.py`. The `remote_home_compute_demo_v1` artifact links the underlying `remote_demo_runbook_v1`, `miner_join_pack_v1`, `POST /admin/inference-sessions` acceptance, `remote_compute_evidence_v1`, `remote_demo_observability_v1`, `model_bundle_infer`, and `remote_python_model_bundle_infer` evidence without exposing plaintext tokens. `remote-demo doctor` writes `remote_home_compute_doctor_v1`, `remote-demo collect` writes `remote_home_compute_collect_v1`, and `remote-demo clean` writes `remote_home_compute_cleanup_v1` while defaulting to dry-run cleanup. It is not model sharding, not production Swarm Inference, and not P2P routing.

For `micro-llm-sharded` stage-aware proofs, prepare two controlled Miner roles instead of one broad Miner: one join pack with `--stage-role stage0` and one with `--stage-role stage1`. The resulting Miners advertise `micro_llm_sharded_stage0` and `micro_llm_sharded_stage1`; verify with `crowdtensor remote-demo verify --workload micro-llm-sharded --stage-mode split --require-distinct-stage-miners ...`. Evidence should include `distinct_stage_miners` and `stage_assignment_valid`. This remains CPU-only, read-only, and operator-controlled.

## Kaggle Remote Miner Beta

Use Kaggle as an outbound temporary Miner when you want an external CPU runtime without configuring a second home machine:

```bash
crowdtensor remote-demo prepare \
  --target kaggle \
  --coordinator-url https://YOUR_COORDINATOR_HOST \
  --miner-id kaggle-cpu-1 \
  --scenario-id route-baseline \
  --output-dir dist/remote-home-compute-kaggle \
  --json
```

Copy only `miner.private.env` and the generated `kaggle_remote_miner.py` into the Kaggle Notebook. The generic `miner_join.sh` and `MINER_JOIN.md` are also generated for ordinary Linux Miner hosts. Keep `operator.private.env` on the Coordinator/operator side. In Kaggle, enable Internet, install the checkout with `python -m pip install -e .`, then run:

```bash
python kaggle_remote_miner.py
```

After the Kaggle Miner starts, run the normal operator-side `crowdtensor remote-demo doctor`, `crowdtensor remote-demo verify`, and `crowdtensor remote-demo collect` commands. The report carries `kaggle_remote_miner_prepare_ready` during prepare and the aggregate check emits `kaggle_remote_miner_beta_check_v1` with `kaggle_remote_miner_beta_ready`.

Kaggle is not the production substrate for CrowdTensor. This path uses Kaggle as an outbound Miner only; it does not expose a Coordinator from Kaggle, does not implement P2P, and does not enable any GPU/TPU workload. Kaggle GPU/TPU visibility may appear in Miner hardware hints later, but the current Beta remains CPU-only, read-only `model_bundle_infer`.

## Kaggle Real Runtime Acceptance

Kaggle Real Runtime Acceptance is the next step after artifact preparation: it proves a live Kaggle CPU Notebook can connect to an operator-owned public Coordinator. For the current server, the temporary HTTP target is `24.199.118.54`:

```bash
crowdtensor remote-demo kaggle-real \
  --action prepare \
  --public-host 24.199.118.54 \
  --port 9180 \
  --miner-id kaggle-cpu-1 \
  --output-dir dist/kaggle-real-runtime \
  --json
```

Run `dist/kaggle-real-runtime/start_coordinator.sh` on the Coordinator host. Upload only `kaggle-upload/miner.private.env` and `kaggle-upload/kaggle_remote_miner.py` to Kaggle, then run:

```bash
python kaggle_remote_miner.py --env-file miner.private.env
```

From the operator host, run:

```bash
crowdtensor remote-demo kaggle-real \
  --action verify \
  --public-host 24.199.118.54 \
  --port 9180 \
  --output-dir dist/kaggle-real-runtime \
  --json
```

`scripts/kaggle_real_runtime_acceptance_pack.py` emits `kaggle_real_runtime_acceptance_v1`; `scripts/kaggle_real_runtime_acceptance_check.py` validates only the generated artifacts in CI. A real Notebook success includes `kaggle_artifacts_ready`, `coordinator_public_ready`, `kaggle_miner_seen`, `kaggle_result_accepted`, and `kaggle_real_runtime_ready`. Keep `operator.private.env` off Kaggle, rotate tokens after the run because `token_rotation_required` is true, and treat temporary HTTP as non-production. This remains CPU-only/read-only `model_bundle_infer`, not production Swarm Inference, not P2P, and not GPU/TPU workload execution.

For `micro-llm-sharded`, prepare with `--workload micro-llm-sharded --stage-mode split --decode-steps 3`. The output contains `kaggle-upload-stage0` and `kaggle-upload-stage1`; upload them to two separate Kaggle CPU Notebooks so stage assignment can be verified. A successful split run reports `kaggle_micro_llm_stage0_seen`, `kaggle_micro_llm_stage1_seen`, `kaggle_micro_llm_stage_assignment_valid`, `stage_assignment_valid`, and `kaggle_micro_llm_sharded_ready`.

`scripts/kaggle_micro_llm_live_package.py` is the operator-only helper for Kaggle CLI runs. It packages the prepared stage upload directories as private Kaggle dataset/script-kernel folders. Default mode keeps source and stage env files in the private dataset; `--inline-kernel-payload` embeds the source tarball and stage `miner.private.env` in private kernel source when dataset mounting fails. Inline kernel source contains a usable Miner token, so never commit it, never publish it, delete the temporary Kaggle kernels/dataset after the proof, and rotate tokens. The completed artifact-backed live proof used public Coordinator `24.199.118.54:9180`, two private Kaggle CPU script kernels, and `micro_llm_artifact_v1`; the local evidence is `dist/kaggle-micro-llm-live/external-real/kaggle_real_runtime_acceptance.json` with `ok: true`, `artifact_loaded`, `micro_llm_artifact_ready`, `kaggle_micro_llm_stage0_seen`, `kaggle_micro_llm_stage1_seen`, `kaggle_micro_llm_stage_assignment_valid`, `baseline_match`, `decoded_tokens_match`, and `kaggle_micro_llm_sharded_ready`. This is a deterministic toy two-stage pipeline, not large-model sharding, not GGUF/llama.cpp serving, and not production Swarm Inference.

## Micro-LLM Live Two-Node RC

Use the RC wrapper when you want one report for the stage-aware split path:

```bash
crowdtensor micro-llm-live-rc --mode local-generated --port 9182 --request-count 2 --decode-steps 3 --json
python scripts/micro_llm_live_rc_check.py --base-port 9182 --request-count 2 --decode-steps 3
```

`scripts/micro_llm_live_rc_pack.py` emits `micro_llm_live_rc_v1`. The default `local-generated` mode creates `kaggle-upload-stage0` and `kaggle-upload-stage1`, starts a local Coordinator plus two independent stage Miner processes from those generated packages, and should report `local_generated_stage_upload_standins_ready`, `micro_llm_live_rc_ready`, `kaggle_micro_llm_sharded_ready`, and `stage_assignment_valid`. After two real Kaggle Notebooks or two real machines are already running against a public Coordinator, run `--mode external-existing` with the same output directory and tokens; only that mode may report `external_runtime_verified`. This remains CPU-only, read-only toy two-stage micro-LLM, not production Swarm Inference, not P2P, and not GGUF/llama.cpp serving.

## Real Small-LLM Sharded Inference Live RC

Use the real-weight RC wrapper when you want generated stage packages for the optional Hugging Face tiny GPT split path:

```bash
python -m pip install -e '.[hf]'
crowdtensor real-llm-live-rc --mode local-generated --port 9184 --request-count 1 --json
python scripts/real_llm_live_rc_check.py --base-port 9184 --request-count 1
```

`scripts/real_llm_live_rc_pack.py` emits `real_llm_live_rc_v1`. The default `local-generated` mode creates `kaggle-upload-real-llm-stage0` and `kaggle-upload-real-llm-stage1`, starts a local Coordinator plus two independent HF-enabled stage Miner processes from those generated packages, and should report `local_generated_real_llm_stage_upload_standins_ready`, `remote_real_llm_sharded_ready`, `real_llm_artifact_ready`, `stage_assignment_valid`, and `real_llm_live_rc_ready`. `kaggle-generated` prepares the packages and runbook only. After two real Kaggle Notebooks or two real machines are already running against a public Coordinator, run `--mode external-existing`; only that mode may report `external_runtime_verified` and `kaggle_real_llm_sharded_ready`. `scripts/kaggle_real_llm_live_package.py` emits `kaggle_real_llm_live_package_v1` private Kaggle dataset/script-kernel packages for this path; `--inline-kernel-payload` is a temporary fallback for Kaggle input-mount issues and embeds stage `miner.private.env` into private kernel source. A completed live proof is retained at `dist/real-llm-live-goal-external/real_llm_live_rc.json` with `kaggle_real_llm_stage0_seen`, `kaggle_real_llm_stage1_seen`, `distinct_stage_miners`, `stage_assignment_valid`, and `kaggle_real_llm_sharded_ready`. Generated Miners use `--enable-hf-tiny-gpt-runtime` and `--real-llm-stage-role stage0|stage1`. This remains CPU-only, read-only, not production Swarm Inference, not P2P, not GPU/TPU pooling, not GGUF/llama.cpp serving, and not large-model serving.

## Real Internet Swarm Inference Alpha

Use the Internet Alpha wrapper when you want one milestone report that combines the real-weight Live RC and stage failure recovery:

```bash
crowdtensor real-llm-internet-alpha --mode local-generated --port 9187 --base-port 9188 --request-count 1 --json
python scripts/real_llm_internet_alpha_check.py --port 9187 --base-port 9188 --request-count 1
```

`scripts/real_llm_internet_alpha_pack.py` emits `real_llm_internet_alpha_v1`. The `local-generated` mode invokes the Live RC and then runs local remote-loopback stand-ins with stage-specific failures so reports must include `real_llm_internet_alpha_ready`, `real_llm_stage_requeue_ready`, `stage_requeue_ready`, `real_llm_live_rc_ready`, `remote_real_llm_sharded_ready`, `real_llm_artifact_ready`, `activation_transport_ready`, `decoded_tokens_match`, `distinct_stage_miners`, and `stage_assignment_valid` without claiming `external_runtime_verified`. `package` prepares the public Coordinator runbook and stage upload directories only. `external-existing` verifies an already running public Coordinator plus two external stage Miners and only then may report `external_runtime_verified`. Temporary public HTTP runs report `token_rotation_required`; public artifacts redact raw prompts, activations, hidden states, logits, tokens, and lease material. This is CPU-only, read-only, not production Swarm Inference, not P2P, not GPU/TPU pooling, not GGUF/llama.cpp serving, and not large-model serving.

## Real Internet Swarm Inference Beta

Use the Internet Beta wrapper when the operator wants one automation command for the external Kaggle-backed proof:

```bash
crowdtensor real-llm-internet-beta --mode kaggle-auto --public-host 24.199.118.54 --port 9190 --base-port 9191 --request-count 2 --json
python scripts/real_llm_internet_beta_check.py --port 9190 --base-port 9191 --request-count 2
```

`scripts/real_llm_internet_beta_pack.py` emits `real_llm_internet_beta_v1`. `kaggle-auto` creates the Alpha package, starts the public Coordinator, pushes private Kaggle CPU script kernels by default or private Kaggle GPU kernels when `--real-llm-backend hf_transformers_cuda` is selected, runs external-existing verification, deletes the kernels, stops the Coordinator, and records `kaggle_kernels_deleted`. With `--failure-mode kill-stage0-after-claim` or `kill-stage1-after-claim`, it also creates target-stage victim/rescue kernels, observes the victim claim through `/state`, deletes the victim kernel, waits for lease timeout requeue, pushes the rescue kernel, and records `external_stage_requeue_ready`, `live_stage0_requeue_ready` / `live_stage1_requeue_ready`, and `live_requeue_summary`. A ready report must include `real_llm_internet_beta_ready`, `real_llm_internet_alpha_ready`, `external_runtime_verified`, both Kaggle stages seen, `decoded_tokens_match`, `distinct_stage_miners`, `stage_assignment_valid`, and `token_rotation_required`. With CUDA selected, the Coordinator may run on CPU and only the Kaggle stage Miners require torch CUDA. `scripts/real_llm_internet_beta_check.py` is the CI-safe fake-runner check and does not create Kaggle resources. This remains read-only optional tiny GPT evidence, not production Swarm Inference, not P2P, not GPU pooling, not GGUF/llama.cpp serving, and not large-model serving.

## Swarm Inference Beta

Use the user-facing Swarm Inference Beta when you want the same real tiny GPT split route as a reusable two-machine operator package:

```bash
python -m pip install -e '.[hf]'
crowdtensor swarm-infer-beta live \
  --public-host 24.199.118.54 \
  --port 9210 \
  --base-port 9211 \
  --request-count 2 \
  --output-dir dist/swarm-inference-beta-live \
  --json

crowdtensor swarm-infer-beta prepare --coordinator-url https://YOUR_COORDINATOR_HOST --json
crowdtensor swarm-infer-beta coordinator --json
crowdtensor swarm-infer-beta miner --stage stage0 --json
crowdtensor swarm-infer-beta miner --stage stage1 --json
crowdtensor swarm-infer-beta verify --coordinator-url https://YOUR_COORDINATOR_HOST --json
crowdtensor swarm-infer-beta collect --coordinator-url https://YOUR_COORDINATOR_HOST --json
```

The wrapper emits `swarm_inference_beta_v1` through `scripts/swarm_inference_beta_pack.py` and is checked by `scripts/swarm_inference_beta_check.py`. `swarm-infer-beta live` is the side-effectful `kaggle-auto` path around `real_llm_internet_beta_v1`; it starts a temporary public Coordinator, pushes private Kaggle CPU stage kernels, verifies `external_runtime_verified`, optionally verifies external victim/rescue requeue with `--failure-mode`, deletes the kernels, writes `support_bundle.json`, removes local live private artifacts and raw runtime state by default, and only then may report `swarm_inference_beta_live_ready`, `real_llm_internet_beta_ready`, `external_stage_requeue_ready` when requested, `live_requeue_summary`, `kaggle_kernels_deleted`, `swarm_inference_beta_live_private_artifacts_cleaned`, and `token_rotation_required`. `--keep-live-private-artifacts` is a debugging-only escape hatch. Prepare creates `operator.private.env`, stage0/stage1 `miner.private.env`, hashed `miner_registry.json`, and stage join packs. Verify wraps `remote_real_llm_sharded_beta_v1` and should report `real_llm_split_route_ready`, `decoded_tokens_match`, `distinct_stage_miners`, `stage_assignment_valid`, `swarm_inference_beta_ready`, and `two_machine_swarm_inference_ready`. Import retained `real_llm_internet_beta_v1` evidence with `--real-internet-beta-report` only as `external_beta_evidence_imported`. This is CPU-only, read-only, not production Swarm Inference, not P2P, and not large-model serving.

`crowdtensor swarm-session` is the Public Swarm Inference Alpha wrapper over the same controlled remote path. It emits `public_swarm_inference_alpha_v1` through `scripts/public_swarm_inference_alpha_pack.py` and is checked by `scripts/public_swarm_inference_alpha_check.py`. Use `--mode live-kaggle --failure-mode kill-stage0-after-claim` to combine the cleanup-backed external Kaggle proof, true external victim/rescue requeue evidence (`external_stage_requeue_ready`, `live_stage0_requeue_ready` / `live_stage1_requeue_ready`, `live_requeue_summary`), and mandatory `local-generated` requeue evidence; a ready report should include `public_swarm_inference_alpha_ready`, `public_swarm_session_ready`, `local_stage_requeue_ready`, `public_swarm_live_requeue_ready`, `public_swarm_live_kaggle_ready`, `stage_requeue_ready`, `external_runtime_verified`, `kaggle_kernels_deleted`, and `token_rotation_required`. Child debug artifacts are pruned by default, while `--keep-child-artifacts` is local debugging only. It is CPU-only, read-only, not production Swarm Inference, not P2P, and not large-model serving.

`crowdtensor public-swarm-alpha-rc` is the Public Swarm Inference Alpha RC evidence-import wrapper. It emits `public_swarm_inference_alpha_rc_v1` through `scripts/public_swarm_inference_alpha_rc_pack.py` and is checked by `scripts/public_swarm_inference_alpha_rc_check.py`. The `evidence-import` mode requires retained public reports for `stage0_live_requeue_evidence_ready`, `stage1_live_requeue_evidence_ready`, `public_swarm_live_requeue_evidence_ready`, `public_swarm_alpha_rc_evidence_imported`, `public_swarm_alpha_private_artifacts_absent`, and `public_swarm_inference_alpha_rc_ready`; `local-smoke` is the CI-safe contract check. The retained proof paths are `dist/public-swarm-inference-alpha-live-stage0-requeue-20260527165830/public_swarm_inference_alpha.json`, `dist/public-swarm-inference-alpha-live-stage1-requeue-20260527170600/public_swarm_inference_alpha.json`, and `dist/public-swarm-inference-alpha-live-requeue-summary.json`. This is CPU-only, read-only, not production Swarm Inference, not P2P, and not large-model serving.

`crowdtensor public-swarm-beta` is the Public Swarm Inference Beta user entrypoint over the current Coordinator-backed product surface. It emits `public_swarm_inference_beta_v1` through `scripts/public_swarm_inference_beta_pack.py` and is checked by `scripts/public_swarm_inference_beta_check.py`. Use `public-swarm-beta product-beta` for the aggregate product proof with `public_swarm_product_beta_ready`, `public_swarm_product_rc_ready`, `coordinator_product_surface_ready`, `session_protocol_ready`, `p2p_lite_discovery_ready`, `gpu_generation_evidence_import_ready`, and `cpu_fallback_ready`. Use `public-swarm-beta local-loopback` for a fresh localhost split proof with `two_stage_split_inference_ready`, `local_loopback_ready`, `decoded_tokens_match`, `distinct_stage_miners`, and `stage_assignment_valid`. Use `public-swarm-beta evidence-import` to import retained Alpha RC live evidence with `public_swarm_beta_evidence_import_ready`, `external_live_evidence_imported`, `stage0_live_requeue_evidence_ready`, and `stage1_live_requeue_evidence_ready`. The two-machine flow remains `prepare`, `coordinator`, `miner --stage stage0`, `miner --stage stage1`, `verify`, `collect`, and dry-run `clean`. This is Coordinator-backed and read-only, not production Swarm Inference, not libp2p/DHT/NAT traversal, not Hivemind-level serving, and not large-model serving.

`crowdtensor public-swarm-beta-rc` is the Public Swarm Inference Beta RC wrapper for operators who want one release-candidate artifact over product Beta. It emits `public_swarm_inference_beta_rc_v1` through `scripts/public_swarm_inference_beta_rc_pack.py` and is checked by `scripts/public_swarm_inference_beta_rc_check.py`. Use `public-swarm-beta-rc local-loopback` to run the product `serve` / `join` / `generate` loop and require `serve_join_generate_loop_ready`, `remote_generate_session_ready`, and `public_swarm_generate_ready`; use `public-swarm-beta-rc package --target kaggle` to prepare join material while keeping `private_artifacts_local_only` and `miner_join_pack_ready`; use `public-swarm-beta-rc external-existing` for an already running Coordinator plus stage Miners, where `external_runtime_verified` is only emitted after live verification. The RC also imports `public_swarm_product_beta_ready`, `p2p_lite_route_ready`, `p2p_lite_discovery_ready`, and `cpu_fallback_ready`. Missing optional HF dependencies should surface `hf_dependencies_missing`. It is CPU-only by default, read-only, not production Swarm Inference, not libp2p, not DHT, not NAT traversal, and not large-model serving.

`crowdtensor public-swarm-product-beta` is the ordinary user-facing two-stage product path. It emits `public_swarm_product_beta_v1` through `scripts/public_swarm_product_beta_pack.py` and is checked by `scripts/public_swarm_product_beta_check.py`. Use `public-swarm-product-beta local-loopback` after `python -m pip install -e '.[hf]'` to prove `serve_ready`, `stage0_join_ready`, `stage1_join_ready`, `generate_ready`, `support_bundle_ready`, `private_artifacts_cleaned`, `decoded_tokens_match`, `distinct_stage_miners`, and `stage_assignment_valid`; use `public-swarm-product-beta package --target kaggle` for join material with `private_artifacts_local_only` and `miner_join_pack_ready`; use `public-swarm-product-beta external-existing` only against an already running controlled runtime. Missing optional HF dependencies should surface `hf_dependencies_missing`. It is CPU-only by default, read-only, Coordinator-backed, not production Swarm Inference, not libp2p, not DHT, not NAT traversal, and not large-model serving.

`crowdtensor preview` is the Public Swarm Developer Preview wrapper over Product Beta for ordinary users. It emits `public_swarm_developer_preview_v1` through `scripts/public_swarm_developer_preview_pack.py` and is checked by `scripts/public_swarm_developer_preview_check.py`. Use `preview local` after `python -m pip install -e '.[hf]'` to prove `developer_preview_ready`, `public_swarm_developer_preview_ready`, `local_two_stage_generation_ready`, `serve_join_generate_ready`, `product_beta_ready`, `support_bundle_ready`, `cpu_fallback_ready`, and `local_cpu_inference_ready`; retained GPU generation evidence adds `gpu_generation_evidence_import_ready`. Use `preview package --target kaggle` for join material, `preview external-existing` only against an already running controlled runtime, and `preview evidence-import` for retained redacted evidence. Missing optional HF dependencies should surface `hf_dependencies_missing`. It is CPU-only by default, read-only, Coordinator-backed, not production Swarm Inference, not libp2p, not DHT, not NAT traversal, and not large-model serving.

`crowdtensor public-swarm-gpu-beta` is the optional Public Swarm GPU Inference Beta overlay for CUDA hosts. It emits `public_swarm_gpu_inference_beta_v1` through `scripts/public_swarm_gpu_inference_beta_pack.py` and is checked by `scripts/public_swarm_gpu_inference_beta_check.py`. Use `public-swarm-gpu-beta local-smoke` on any host to record safe CUDA availability with `public_swarm_gpu_beta_smoke_ready`; it does not claim `public_swarm_gpu_beta_ready` on CPU-only machines. Use `public-swarm-gpu-beta local-loopback` only after installing `[hf]` on CUDA-capable hosts; it selects `hf_transformers_cuda`, requires `cuda_runtime_available`, `hf_transformers_cuda_ready`, and `gpu_runtime_ready`, and schedules stage work only to Miners advertising `real_llm_sharded_cuda_stage0`, `real_llm_sharded_cuda_stage1`, or `real_llm_sharded_cuda_both`. Ready GPU reports should include `public_swarm_gpu_beta_ready`, `gpu_stage0_ready`, and `gpu_stage1_ready`. `public-swarm-gpu-beta kaggle-package` prepares private Kaggle GPU stage templates with `kaggle_gpu_package_ready`, and `public-swarm-gpu-beta evidence-import` imports a completed GPU report with `external_gpu_runtime_verified`. The side-effectful `kaggle-auto` path has retained successful evidence at `dist/public-swarm-gpu-beta-live-20260528-runtimepin/public_swarm_gpu_inference_beta_kaggle_auto.json` and pins generated Kaggle CUDA kernels to `torch==2.7.1+cu118`, `torchvision==0.22.1+cu118`, and `transformers==4.40.2`. This is read-only optional CUDA tiny GPT evidence, not production Swarm Inference, not P2P, not a GPU pooling marketplace, and not large-model serving.

The same real-weight split is also available through the high-level remote-demo wrapper. Prepare one stage0 join pack and one stage1 join pack:

```bash
python -m pip install -e '.[hf]'
crowdtensor remote-demo prepare --workload real-llm-sharded --stage-role stage0 --coordinator-url https://YOUR_COORDINATOR_HOST --miner-id real-stage0 --output-dir dist/remote-real-llm-stage0 --json
crowdtensor remote-demo prepare --workload real-llm-sharded --stage-role stage1 --coordinator-url https://YOUR_COORDINATOR_HOST --miner-id real-stage1 --output-dir dist/remote-real-llm-stage1 --json
```

Run each generated `miner_join.sh` or Kaggle launcher on a distinct CPU host, then verify with `crowdtensor remote-demo verify --workload real-llm-sharded --stage-mode split --require-distinct-stage-miners ...` and collect with the matching `collect` command. The acceptance path emits `remote_real_llm_sharded_acceptance_v1`, `remote_real_llm_sharded_observability_v1`, `remote_real_llm_sharded_beta_v1`, route `remote_python_real_llm_sharded_infer`, and readiness code `remote_two_machine_real_llm_sharded_ready`. If `transformers` or other optional HF runtime pieces are absent, diagnostics include `hf_dependencies_missing` and the operator action to install `python -m pip install -e '.[hf]'`.

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
