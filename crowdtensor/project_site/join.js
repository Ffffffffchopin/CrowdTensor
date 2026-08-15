"use strict";

const SESSION_KEY = "crowdtensor_browser_pairing_v1";
const state = {
  paired: null,
  worker: null,
  heartbeatTimer: null,
  currentTask: null,
  running: false,
};

function byId(id) { return document.getElementById(id); }
function setText(id, value) { const element = byId(id); if (element) element.textContent = String(value); }

function nonce() {
  return crypto.randomUUID() + "-" + crypto.randomUUID();
}

function browserCapability() {
  return {
    runtime: "browser-worker",
    secure_context: window.isSecureContext,
    webgpu_available: Boolean(navigator.gpu),
    webassembly_available: typeof WebAssembly === "object",
    logical_cpu_count: Number(navigator.hardwareConcurrency || 1),
    device_memory_gib: Number(navigator.deviceMemory || 0),
    model_training: false,
  };
}

function renderHardware() {
  const capability = browserCapability();
  setText("compute-path", capability.webgpu_available ? "WebGPU preferred" : capability.webassembly_available ? "WASM / CPU" : "JavaScript CPU");
  setText("cpu-count", capability.logical_cpu_count);
  byId("step-check").classList.add("complete");
}

function setRunState(value) { setText("run-state", value); }

function renderPaired() {
  const paired = Boolean(state.paired && state.paired.credential_token);
  byId("start-button").disabled = !paired || state.running;
  byId("pause-button").disabled = !state.running;
  byId("exit-button").disabled = !paired;
  byId("pairing-code").disabled = paired;
  byId("pair-button").disabled = paired;
  if (paired) {
    byId("step-pair").classList.add("complete");
    byId("step-run").classList.add("active");
    byId("pair-state").className = "field-state success";
    setText("pair-state", "Paired with a short-lived browser credential");
    if (!state.running) setRunState("Ready");
  }
}

function savePairing(value) {
  state.paired = value;
  sessionStorage.setItem(SESSION_KEY, JSON.stringify(value));
}

function clearPairing() {
  sessionStorage.removeItem(SESSION_KEY);
  state.paired = null;
}

function authHeaders() {
  return {
    "Authorization": "Bearer " + state.paired.credential_token,
    "Content-Type": "application/json",
    "X-CrowdTensor-Nonce": nonce(),
  };
}

async function jsonRequest(url, options) {
  const response = await fetch(url, { cache: "no-store", ...options });
  let value = {};
  try { value = await response.json(); } catch (_error) { value = {}; }
  if (!response.ok) {
    const error = new Error(String(value.error || `http_${response.status}`));
    error.code = String(value.error || "request_failed");
    throw error;
  }
  return value;
}

function friendlyError(code) {
  const labels = {
    volunteer_pairing_code_invalid: "Pairing code is not valid",
    volunteer_pairing_code_consumed: "Pairing code was already used",
    volunteer_pairing_code_expired: "Pairing code has expired",
    volunteer_cell_credential_expired: "This pairing expired; request a new code",
    volunteer_browser_probe_capacity_exceeded: "Browser task capacity is currently full",
    volunteer_browser_probe_output_invalid: "Result verification failed",
  };
  return labels[code] || "Contribution request could not be completed";
}

async function pairDevice(event) {
  event.preventDefault();
  const input = byId("pairing-code");
  const pairingCode = input.value.trim().toUpperCase();
  const cellId = "browser-" + crypto.randomUUID();
  byId("pair-button").disabled = true;
  byId("pair-state").className = "field-state";
  setText("pair-state", "Pairing");
  try {
    const value = await jsonRequest("/v1/volunteer/pairing/redeem", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pairing_code: pairingCode, cell_id: cellId, expected_mode: "browser" }),
    });
    if (value.pairing_mode !== "browser") {
      throw Object.assign(new Error("pairing_mode_mismatch"), { code: "pairing_mode_mismatch" });
    }
    savePairing({
      cell_id: cellId,
      credential_token: value.credential_token,
      credential_id: value.credential_id,
      expires_at: value.expires_at,
    });
    input.value = "";
    renderPaired();
  } catch (error) {
    clearPairing();
    byId("pair-state").className = "field-state error";
    setText("pair-state", friendlyError(error.code));
    byId("pair-button").disabled = false;
  }
}

async function heartbeat() {
  if (!state.currentTask || !state.paired) return;
  const task = state.currentTask;
  await jsonRequest("/v1/volunteer/browser/heartbeat", {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({
      cell_id: state.paired.cell_id,
      task_id: task.task_id,
      lease_generation: task.lease_generation,
      lease_token: task.lease_token,
    }),
  });
}

function stopHeartbeat() {
  if (state.heartbeatTimer) window.clearInterval(state.heartbeatTimer);
  state.heartbeatTimer = null;
}

async function submitResult(result) {
  const task = state.currentTask;
  const value = await jsonRequest("/v1/volunteer/browser/submit", {
    method: "POST",
    headers: authHeaders(),
    body: JSON.stringify({
      cell_id: state.paired.cell_id,
      task_id: task.task_id,
      lease_generation: task.lease_generation,
      lease_token: task.lease_token,
      output_sha256: result.output_sha256,
      runtime: result.runtime,
      duration_ms: result.duration_ms,
    }),
  });
  return value;
}

function finishRun(finalState) {
  stopHeartbeat();
  if (state.worker) state.worker.terminate();
  state.worker = null;
  state.currentTask = null;
  state.running = false;
  renderPaired();
  if (finalState) setRunState(finalState);
}

async function startContribution() {
  if (!state.paired || state.running) return;
  state.running = true;
  byId("result-line").hidden = true;
  renderPaired();
  setRunState("Claiming bounded task");
  try {
    const claim = await jsonRequest("/v1/volunteer/browser/claim", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ cell_id: state.paired.cell_id, capability: browserCapability() }),
    });
    if (!claim.task) {
      throw Object.assign(new Error(String(claim.state || "no_task")), { code: String(claim.state || "no_task") });
    }
    state.currentTask = claim.task;
    await heartbeat();
    state.heartbeatTimer = window.setInterval(() => heartbeat().catch(() => pauseContribution()), 30000);
    setRunState("Computing in isolated Worker");
    state.worker = new Worker("/assets/join_worker.js");
    state.worker.onerror = () => {
      setRunState("Worker failed; task paused");
      pauseContribution();
    };
    state.worker.onmessage = async (event) => {
      if (!event.data || event.data.type !== "complete") return;
      setRunState("Verifying with Coordinator");
      let finalState = "Ready";
      try {
        const accepted = await submitResult(event.data);
        byId("step-run").classList.add("complete");
        byId("result-line").hidden = false;
        setText("result-detail", `${accepted.runtime} / ${accepted.duration_ms} ms / model updates: 0`);
        finalState = "Accepted";
        await loadSnapshot();
      } catch (error) {
        finalState = friendlyError(error.code);
      } finally {
        finishRun(finalState);
      }
    };
    state.worker.postMessage({ type: "run", task: state.currentTask });
  } catch (error) {
    const finalState = friendlyError(error.code);
    if (String(error.code || "").includes("credential")) clearPairing();
    finishRun(finalState);
  }
}

function pauseContribution() {
  stopHeartbeat();
  if (state.worker) state.worker.terminate();
  state.worker = null;
  state.running = false;
  state.currentTask = null;
  renderPaired();
  setRunState("Paused; the lease can be reclaimed");
}

function exitContribution() {
  pauseContribution();
  clearPairing();
  byId("pairing-code").disabled = false;
  byId("pair-button").disabled = false;
  byId("exit-button").disabled = true;
  byId("start-button").disabled = true;
  byId("step-pair").classList.remove("complete");
  byId("step-run").classList.remove("active", "complete");
  byId("pair-state").className = "field-state";
  setText("pair-state", "Awaiting an approved beta code");
  setRunState("Not paired");
}

function humanCampaign(value) {
  return String(value || "Campaign").replace(/^crowdtensor-/, "").split(/[-_]+/).filter(Boolean).map((word) => word[0].toUpperCase() + word.slice(1)).join(" ");
}

async function loadSnapshot() {
  try {
    const snapshot = await jsonRequest("/v1/volunteer/public-snapshot", { headers: { Accept: "application/json" } });
    const campaign = snapshot.campaign || {};
    const progress = snapshot.progress || {};
    const completed = Number(progress.completed_rounds || 0);
    const target = Math.max(1, Number(progress.target_rounds || 1));
    setText("campaign-name", humanCampaign(campaign.campaign_id));
    setText("campaign-progress", `${completed} / ${target} rounds`);
    setText("training-updates", Number(progress.accepted_update_count || 0).toLocaleString());
    setText("browser-tasks", Number(progress.accepted_browser_task_count || 0).toLocaleString());
    setText("adapter-version", `v${Number(progress.adapter_version || 0)}`);
    byId("snapshot-fill").style.width = `${Math.max(0, Math.min(100, (completed / target) * 100))}%`;
  } catch (_error) {
    setText("campaign-name", "Campaign temporarily unavailable");
  }
}

function selectMode(mode) {
  const browser = mode === "browser";
  byId("browser-tab").classList.toggle("active", browser);
  byId("agent-tab").classList.toggle("active", !browser);
  byId("browser-tab").setAttribute("aria-selected", String(browser));
  byId("agent-tab").setAttribute("aria-selected", String(!browser));
  byId("browser-panel").hidden = !browser;
  byId("agent-panel").hidden = browser;
}

async function copyAgentCommand() {
  await navigator.clipboard.writeText(byId("agent-command").textContent);
  setText("copy-command", "Copied");
  window.setTimeout(() => setText("copy-command", "Copy"), 1200);
}

async function loadAgentRelease() {
  const origin = window.location.origin.replace(/\/$/, "");
  const command = `curl -fsSL ${origin}/downloads/install-contributor.sh | sh -s -- ${origin} CT-XXXX-XXXX-XXXX`;
  try {
    const health = await jsonRequest("/v1/volunteer/health", { headers: { Accept: "application/json" } });
    if (health.public_release_download !== true) throw new Error("release_unavailable");
    setText("agent-command", command);
    setText("agent-release-state", health.package_version || "Ready");
    byId("copy-command").disabled = false;
  } catch (_error) {
    setText("agent-command", "Contributor release not attached by this Campaign operator");
    setText("agent-release-state", "Unavailable");
    byId("copy-command").disabled = true;
  }
}

function restorePairing() {
  try {
    const value = JSON.parse(sessionStorage.getItem(SESSION_KEY) || "null");
    if (value && value.credential_token && Number(value.expires_at || 0) > Date.now() / 1000 + 5) {
      state.paired = value;
    } else {
      clearPairing();
    }
  } catch (_error) {
    clearPairing();
  }
}

byId("pair-form").addEventListener("submit", pairDevice);
byId("start-button").addEventListener("click", startContribution);
byId("pause-button").addEventListener("click", pauseContribution);
byId("exit-button").addEventListener("click", exitContribution);
byId("browser-tab").addEventListener("click", () => selectMode("browser"));
byId("agent-tab").addEventListener("click", () => selectMode("agent"));
byId("copy-command").addEventListener("click", () => copyAgentCommand().catch(() => {}));
window.addEventListener("beforeunload", stopHeartbeat);

restorePairing();
selectMode("agent");
renderHardware();
renderPaired();
loadSnapshot();
loadAgentRelease();
