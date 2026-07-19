# Kaggle TPU v5e-8 Interactive Runtime Runbook

## Purpose

This runbook describes the most repeatable Kaggle TPU acquisition and execution
path found by this project. It is account-independent: use any Kaggle account
that is allowed to start a TPU v5e-8 Interactive Notebook, together with a
Notebook owned by or shared with that account.

The method is not an allocation guarantee. Kaggle can queue, cancel, reclaim,
or withhold TPU capacity. The reliable part is the control and verification
sequence:

1. Authenticate a browser with a private Playwright storage-state file.
2. Select `TPU v5e-8` in the Interactive Notebook UI and click Start Session.
3. Keep monitoring the visible queue and Active Events until the TPU event is
   Running.
4. Attach through the Notebook iframe's
   `window.jupyterapp.serviceManager`, not the root Kaggle `/api/kernels` URL.
5. Run a real JAX operation and a tiny Qwen-like JAX cell. Only start the model
   workload after both pass and JAX reports eight TPU devices.

Kaggle API tokens, `kaggle kernels push`, accelerator-name guessing, and an MCP
interactive-session ID did not reliably prove TPU allocation. In particular,
an interactive session can exist while `jax.devices()` still exposes CPU only.
The Web Interactive Notebook path is therefore the current recommended path.

## Proven Boundary

The retained successful channel evidence is:

- `dist/kaggle-web-tpu-execution-channel-probe-20260701-r2-force-new-session-after-running-event/kaggle_web_tpu_execution_channel_probe.json`
- `dist/kaggle-web-tpu-execution-channel-probe-20260702-r13-after-r20-session-started-before-cpuowner-kagglecpu-fp4-bridge/kaggle_web_tpu_execution_channel_probe.json`

Both record a real attached runtime, successful JAX execution, and eight
devices whose JAX `device_kind` is `TPU v5 lite`. Seeing eight `TPU v5 lite`
devices is the expected JAX view of the Kaggle `TPU v5e-8` selection.

This path has also executed real stage-owned Qwen 32B and Qwen 72B JAX/TPU
workloads. Those are model-stage proofs, not proof that every model can be
loaded, that full 72B inference works, or that long-running training is stable.

## Requirements

- A Kaggle account with Interactive Notebook access and available TPU quota.
- A private or controlled Kaggle Notebook URL of this form:
  `https://www.kaggle.com/code/<owner>/<notebook-slug>/edit`.
- An authenticated Playwright storage-state JSON for the same account.
- Python, Playwright, and Chrome or Chromium on the controlling machine.
- This repository if using the supplied monitor, checker, and execution tools.

Install browser support from this checkout:

```bash
python -m pip install -e '.[browser]'
```

The scripts default to `/usr/bin/google-chrome`. Pass
`--chrome-executable <path>` when Chrome or Chromium is elsewhere.

## Private Authentication Input

The Web path uses a browser login, not a Kaggle API token. The storage-state
file is an account credential and must be handled like a password.

Preferred preparation:

1. Start a Playwright browser in an environment where the operator can log in.
2. Navigate to `https://www.kaggle.com/` and complete login/MFA manually.
3. Save the browser context with
   `context.storage_state(path="/secure/path/kaggle-storage-state.json")`.
4. Set mode `0600` on the resulting file.

If the operator exports browser cookies instead, convert them privately into a
Playwright storage-state object with `cookies` and `origins` arrays. Cookie
export formats differ, so validate the result with the read-only UI probe
before attempting allocation. Never print cookie values, pass them on a command
line, upload them to Kaggle, place them in `dist/`, or commit them.

Basic local validation without displaying values:

```bash
chmod 600 /secure/path/kaggle-storage-state.json
jq -e '
  (.cookies | type == "array") and
  (.origins | type == "array") and
  (.cookies | length > 0)
' /secure/path/kaggle-storage-state.json >/dev/null
```

Use a separate storage-state file and browser context for each account. This
runbook does not authorize credential sharing or quota circumvention.

## Session Inputs

Set these variables in the controlling Session. Always pass the Notebook URL
explicitly because historical script defaults refer to a project-specific
Notebook.

```bash
export CROWDTENSOR_REPO=/path/to/CrowdTensor
export KAGGLE_NOTEBOOK_URL='https://www.kaggle.com/code/<owner>/<slug>/edit'
export KAGGLE_WEB_STORAGE_STATE='/secure/path/kaggle-storage-state.json'
export CHROME_EXECUTABLE='/usr/bin/google-chrome'
```

Before starting the session, attach any large Kaggle Model or Dataset through
the Notebook UI. This makes it available below `/kaggle/input` without first
downloading the complete asset into `/kaggle/working`. Do not assume a mounted
path; inspect `/kaggle/input` inside the Notebook.

## Phase 1: Read-Only Authentication and UI Check

Run a side-effect-free check first:

```bash
python "$CROWDTENSOR_REPO/scripts/kaggle_web_tpu_ui_state_probe.py" \
  --kaggle-notebook-url "$KAGGLE_NOTEBOOK_URL" \
  --kaggle-web-storage-state "$KAGGLE_WEB_STORAGE_STATE" \
  --chrome-executable "$CHROME_EXECUTABLE" \
  --output-dir dist/kaggle-tpu-ui-preflight \
  --json
```

Interpretation:

- A login page, HTTP 401/403, or missing Notebook editor means the storage state
  must be refreshed.
- `start_session_visible=true` means the Notebook is authenticated but no
  usable runtime is attached.
- `session_started_text_visible=true` alone is not readiness proof.
- A Running Active Event alone is a signal to attempt attachment, not proof
  that JAX can execute.

## Phase 2: Select TPU v5e-8 and Wait

The queue monitor can select TPU v5e-8, click Start Session, parse the visible
`You are #N in the queue` prompt, inspect Active Events, and write a public-safe
live status file.

```bash
python "$CROWDTENSOR_REPO/scripts/kaggle_web_tpu_queue_monitor_probe.py" \
  --kaggle-notebook-url "$KAGGLE_NOTEBOOK_URL" \
  --kaggle-web-storage-state "$KAGGLE_WEB_STORAGE_STATE" \
  --chrome-executable "$CHROME_EXECUTABLE" \
  --wait-seconds 21600 \
  --poll-seconds 60 \
  --observe-active-events-each-poll \
  --stop-after-session-started-polls 2 \
  --output-dir dist/kaggle-tpu-queue-current \
  --json
```

Validate the monitor artifact:

```bash
python "$CROWDTENSOR_REPO/scripts/kaggle_web_tpu_queue_monitor_check.py" \
  --report dist/kaggle-tpu-queue-current/kaggle_web_tpu_queue_monitor_probe.json \
  --json
```

Important monitor semantics:

- A valid not-ready queue report exits successfully because it is valid
  operational evidence. Do not treat process exit code 0 as TPU readiness.
- In the summary, read `web_tpu_runtime_ready`, `active_event_running`, and
  `queue_progress`. The incrementally updated
  `kaggle_web_tpu_queue_monitor_live_status.json` also exposes
  `session_started_handoff_candidate`.
- If a queue or Starting state already exists, the monitor does not click Start
  again unless `--force-start-click` is supplied.
- Do not use `--force-start-click` while a queued or Running TPU Active Event
  exists. Repeated start clicks can reset or confuse the allocation lifecycle.
- The maximum monitor window is six hours. If the report still shows a live
  queued/Starting event, immediately run another monitor window. Closing the
  local browser does not cancel the server-side queue.
- Use a new output directory for each monitor window, or archive the previous
  report before reusing `dist/kaggle-tpu-queue-current`, so queue history is not
  overwritten.
- A static queue position is not conclusive failure. Continue while Kaggle
  still shows a queued event, unless the user stops the run or the event is
  cancelled.

Do not declare the workload blocked merely because a visible queue has not yet
finished. Stop waiting only when one of these is true:

- the Active Event becomes Running or a runtime handoff signal appears;
- Kaggle cancels the event;
- authentication or Notebook access is lost;
- the user explicitly stops the attempt.

## Phase 3: Open a Running Active Event

If Active Events says Running but the Notebook has no visible Jupyter frame or
kernel, open the Running event and let Kaggle reattach the editor:

```bash
python "$CROWDTENSOR_REPO/scripts/kaggle_web_tpu_active_event_probe.py" \
  --kaggle-notebook-url "$KAGGLE_NOTEBOOK_URL" \
  --kaggle-web-storage-state "$KAGGLE_WEB_STORAGE_STATE" \
  --chrome-executable "$CHROME_EXECUTABLE" \
  --wait-seconds 7200 \
  --poll-seconds 30 \
  --attempt-open-running-event \
  --output-dir dist/kaggle-tpu-active-event-current \
  --json
```

Proceed to the execution gate once the Active Event is Running. It is valid to
try the execution gate even if the read-only page still cannot count a Jupyter
session; `--web-tpu-force-new-session` creates a fresh Jupyter session inside
the already allocated TPU runtime.

## Phase 4: Mandatory JAX Execution Gate

Run the channel probe before any expensive model load:

```bash
python "$CROWDTENSOR_REPO/scripts/kaggle_web_tpu_execution_channel_probe.py" \
  --kaggle-notebook-url "$KAGGLE_NOTEBOOK_URL" \
  --kaggle-web-storage-state "$KAGGLE_WEB_STORAGE_STATE" \
  --chrome-executable "$CHROME_EXECUTABLE" \
  --web-tpu-force-new-session \
  --web-tpu-execute-timeout-seconds 300 \
  --output-dir dist/kaggle-tpu-channel-current \
  --json

python "$CROWDTENSOR_REPO/scripts/kaggle_web_tpu_execution_channel_check.py" \
  --report dist/kaggle-tpu-channel-current/kaggle_web_tpu_execution_channel_probe.json \
  --json
```

The runtime is usable only when all of these are true:

```text
ok=true
web_tpu_execution_channel_ready=true
small_jax_cell_ready=true
tiny_qwen_like_cell_ready=true
tpu_runtime_attached=true
tpu_device_count=8
blocker_codes=[]
```

The proven access mode is `browser_iframe_service_manager`. The executor waits
for `globalThis.jupyterapp.serviceManager`, attaches to an existing Jupyter
session or starts one, and submits code through `kernel.requestExecute` with a
bounded timeout. Root Kaggle `/api/kernels` is not the primary path.

## Failure Classification

Use the failure code to choose the next action:

| Symptom | Meaning | Action |
| --- | --- | --- |
| Login/editor unavailable | Browser credential expired or wrong account | Refresh the private storage state |
| Queue/Starting visible | Allocation still pending | Continue monitor windows without repeated Start clicks |
| Active Event Cancelled | Kaggle ended the allocation | Start a new queue cycle |
| Running, no Jupyter frame | TPU exists but editor is detached | Open the Running event, then use `--web-tpu-force-new-session` |
| `jax_tpu_device_missing` | Session is CPU-backed or stale | Reattach/force a new Jupyter session; reacquire TPU if persistent |
| `web_tpu_jupyter_kernel_not_ready` | Jupyter kernel did not start | Retry one force-new session after the Running event is stable |
| `web_tpu_jupyter_execute_timeout` | Kernel/channel is stale or the cell exceeded the bound | Retry a fresh Jupyter session; split long work into checkpointed chunks |
| Small JAX passes, model cell fails | TPU allocation is healthy; model code/runtime is not | Fix model dependencies, shapes, sharding, or memory use without reacquiring first |

## Running Project Code

### Interactive Notebook path

For long inference batches or training, the simplest path is to put the
project code in the authenticated Notebook and run it after the channel gate.
Start every workload with this assertion:

```python
import jax

devices = list(jax.devices())
tpu_devices = [device for device in devices if device.platform == "tpu"]
assert len(tpu_devices) == 8, devices
print({
    "jax_version": jax.__version__,
    "device_count": len(tpu_devices),
    "device_kind": tpu_devices[0].device_kind,
})
```

Do not assume a PyTorch/CUDA model will run on this runtime. Use one of:

- a JAX/Flax/Optax model implementation;
- MaxText or another TPU-native runtime;
- a project-specific safetensors-to-JAX adapter;
- torch-xla only after a separate compatibility smoke passes.

Avoid upgrading `jax` or `jaxlib` blindly inside the Kaggle runtime. A wheel
that does not match Kaggle's TPU environment can turn a working TPU session
into a CPU-only or import-failing session.

### Automated bounded cells

Other Sessions can use the same proven service-manager executor. The submitted
cell must print one JSON line whose top-level object contains a `report` object:

```python
import os
from argparse import Namespace
from scripts.gpu_tpu_cpu_same_request_runtime_bridge_probe import (
    execute_web_tpu_code_via_iframe,
)

args = Namespace(
    kaggle_notebook_url=os.environ["KAGGLE_NOTEBOOK_URL"],
    kaggle_web_storage_state=os.environ["KAGGLE_WEB_STORAGE_STATE"],
    chrome_executable=os.environ["CHROME_EXECUTABLE"],
    web_tpu_execute_timeout_seconds=900.0,
    web_tpu_force_new_session=False,
)

code = r'''\
import json
import jax
import jax.numpy as jnp

tpu = [device for device in jax.devices() if device.platform == "tpu"]
report = {"ok": False, "tpu_device_count": len(tpu)}
if len(tpu) == 8:
    value = jax.device_put(jnp.arange(16).reshape(4, 4), tpu[0])
    result = (value @ value.T).block_until_ready()
    report.update({"ok": True, "shape": list(result.shape)})
print(json.dumps({"schema": "project_tpu_job_v1", "report": report}))
'''

result = execute_web_tpu_code_via_iframe(args, code)
if result.get("ok") is not True:
    raise RuntimeError(result.get("blockers") or "Kaggle Web TPU job failed")
```

Run this from a checkout where the repository root is on `PYTHONPATH`. Keep
cells bounded. The supplied execution-channel CLI caps one cell at 900 seconds;
long work should be divided into restartable chunks rather than hiding an
hours-long job behind one browser request.

## Inference Guidance

- Prefer BF16 and fixed shapes to limit recompilation.
- Warm up compilation with a small request before loading the full batch.
- Use `jax.sharding.Mesh`, `NamedSharding`, `pjit`, or `shard_map` to use all
  eight devices. Putting every parameter on only `tpu_devices[0]` wastes the
  v5e-8 topology.
- For models that do not fit as one replica, load only stage-owned or
  partition-owned safetensors keys and place arrays directly into their target
  sharding. Avoid materializing the full model several times on host RAM.
- Keep KV-cache shapes stable and preallocate where practical.
- Attach large weights as Kaggle Models/Datasets before allocation. Treat
  `/kaggle/working` as temporary capacity.

## Training Guidance

- Use JAX/Flax plus Optax, MaxText, or another TPU-native training stack.
- Choose data parallelism for models that fit per replica; add tensor/FSDP
  model partitioning when parameters or optimizer state do not fit.
- Verify one forward/backward/optimizer step and finite loss before starting a
  long run.
- Save model, optimizer, RNG, data cursor, and step metadata frequently to a
  durable destination. A checkpoint is not resumable if it contains weights
  but omits optimizer/RNG/data position.
- Use short, idempotent training segments. After each segment, persist a
  checkpoint and a manifest hash before starting the next one.
- Do not rely on the Interactive runtime as an always-on training worker. It
  can be reclaimed without notice, and there is no production SLA.

This repository has verified TPU execution and real model-stage inference. It
does not currently provide a generic production training adapter for arbitrary
Hugging Face PyTorch checkpoints. Each training project must validate its own
JAX checkpoint conversion, partitioning, optimizer semantics, and resume
equivalence.

## Output and Checkpoint Persistence

Before stopping the session:

1. Block until all pending JAX work completes.
2. Write the final checkpoint and an atomic completion manifest.
3. Copy or publish durable outputs to an authorized Kaggle Dataset/Model or
   another approved object store.
4. Verify hashes from the controlling Session.
5. Remove temporary files that contain credentials, prompts, private data, or
   model outputs that should not persist.

Do not place browser storage state, Kaggle cookies, Jupyter proxy URLs/tokens,
HF tokens, or service credentials in `/kaggle/working` or a Notebook cell.

## Stop and Release the TPU

The execution-channel probe may shut down a temporary Jupyter session that it
created, but it does not release the underlying Kaggle TPU Active Event.

After outputs are durable:

1. Open the Notebook with the authenticated account.
2. Use Stop Session/the session power control.
3. Confirm the `TPU v5e-8` Active Event is no longer Running.
4. Run the read-only UI or Active Event probe to verify it is stopped.

Do not delete unrelated Notebooks or sessions. Provider cleanup must target
only resources created for the current workload.

## Handoff Contract for Another Automated Session

Give the Session these private inputs, without pasting their contents into the
conversation or public artifacts:

```text
KAGGLE_NOTEBOOK_URL=<account-owned Notebook edit URL>
KAGGLE_WEB_STORAGE_STATE=<mode-0600 Playwright storage-state path>
CHROME_EXECUTABLE=<local Chrome/Chromium path>
```

Then instruct it to follow this state machine:

```text
AUTH_CHECK
  -> START_OR_REUSE_QUEUE
  -> WAIT_WHILE_QUEUED_OR_STARTING
  -> OPEN_RUNNING_ACTIVE_EVENT
  -> JAX_EXECUTION_GATE
  -> MODEL_PREFLIGHT
  -> CHECKPOINTED_INFERENCE_OR_TRAINING
  -> PERSIST_OUTPUTS
  -> STOP_TPU
```

The Session must not claim TPU readiness from a queue, a session ID, the text
`Session started`, or a Running event alone. Readiness requires the successful
JAX execution gate with eight TPU devices. It must not abandon a visibly live
queue merely because one monitor window expired, and it must not start the
expensive workload until the gate passes.

Suggested instruction for another project Session:

```text
Read <CROWDTENSOR_REPO>/docs/kaggle-tpu-v5e8-runbook.md and acquire one Kaggle
Interactive TPU v5e-8 runtime using the supplied Notebook URL and private
Playwright storage-state path. Always override the historical default Notebook
URL. Start with the read-only UI probe. If no TPU is running, use the queue
monitor and continue back-to-back monitor windows while Kaggle shows Queued or
Starting; do not mark the task blocked only because one wait window expires.
When the TPU Active Event is Running, open it and run the execution-channel
probe with --web-tpu-force-new-session. Do not begin the project workload until
small JAX, tiny Qwen-like execution, and all eight TPU devices are verified.
Run inference/training in bounded checkpointed segments, persist outputs, then
stop only the TPU session created for this workload. Never print or publish
cookies, storage state, Jupyter proxy material, credentials, private prompts,
model outputs, or checkpoint contents.
```

## Evidence and Tool Index

- Queue monitor: `scripts/kaggle_web_tpu_queue_monitor_probe.py`
- Queue checker: `scripts/kaggle_web_tpu_queue_monitor_check.py`
- Read-only UI probe: `scripts/kaggle_web_tpu_ui_state_probe.py`
- Active Event opener: `scripts/kaggle_web_tpu_active_event_probe.py`
- JAX execution gate: `scripts/kaggle_web_tpu_execution_channel_probe.py`
- Execution checker: `scripts/kaggle_web_tpu_execution_channel_check.py`
- Proven service-manager executor:
  `scripts/gpu_tpu_cpu_same_request_runtime_bridge_probe.py`

The UI and private Jupyter implementation can change. If Kaggle changes its
Notebook frontend, revalidate the queue parser, Active Events parser,
`jupyterapp.serviceManager` attachment, and the eight-device JAX gate before
using this runbook for model work.
