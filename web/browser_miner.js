(() => {
  "use strict";

  const params = new URLSearchParams(window.location.search);
  const defaultMinerId = `browser-${Math.random().toString(16).slice(2, 10)}`;
  const holdMs = Math.max(0, Number.parseInt(params.get("hold_ms") || "0", 10));
  const resultMode = params.get("result_mode") === "large_delta" ? "large_delta" : "normal";

  const elements = {
    coordinatorInput: document.getElementById("coordinatorInput"),
    minerIdInput: document.getElementById("minerIdInput"),
    minerTokenInput: document.getElementById("minerTokenInput"),
    runOnceButton: document.getElementById("runOnceButton"),
    autoLoopButton: document.getElementById("autoLoopButton"),
    stopButton: document.getElementById("stopButton"),
    statusDot: document.getElementById("statusDot"),
    statusLabel: document.getElementById("statusLabel"),
    taskMetric: document.getElementById("taskMetric"),
    modelMetric: document.getElementById("modelMetric"),
    stepMetric: document.getElementById("stepMetric"),
    lossMetric: document.getElementById("lossMetric"),
    heartbeatMetric: document.getElementById("heartbeatMetric"),
    workerMetric: document.getElementById("workerMetric"),
    log: document.getElementById("log"),
  };

  const status = {
    running: false,
    autoLoop: false,
    phase: "booting",
    claimed: false,
    accepted: false,
    rejected: false,
    heartbeats: 0,
    taskId: "",
    attempt: 0,
    leaseExpiresAt: 0,
    globalStep: 0,
    modelVersion: 0,
    optimizerStep: 0,
    loss: null,
    innerLossStart: null,
    innerLossEnd: null,
    workerElapsedMs: 0,
    workloadType: "",
    probeHash: "",
    probeGops: 0,
    resultMode,
    holdMs,
    error: "",
  };
  window.__crowdTensorBrowserMinerStatus = status;

  let stopRequested = false;
  let activeHeartbeat = null;

  function coordinatorUrl() {
    return elements.coordinatorInput.value.trim().replace(/\/+$/, "");
  }

  function minerId() {
    return elements.minerIdInput.value.trim() || defaultMinerId;
  }

  function minerToken() {
    return elements.minerTokenInput.value.trim();
  }

  function minerCapabilities() {
    return {
      runtime: "browser",
      backend: "js-worker",
      supports_training_spec: true,
      protocol_version: "runtime_contract_v1",
      supported_workloads: ["diloco_train", "browser_probe"],
      user_agent: navigator.userAgent,
    };
  }

  function log(message) {
    const line = document.createElement("div");
    line.textContent = `[${new Date().toISOString().slice(11, 19)}] ${message}`;
    elements.log.appendChild(line);
    elements.log.scrollTop = elements.log.scrollHeight;
  }

  function setStatus(label, tone = "warn") {
    status.phase = label;
    elements.statusLabel.textContent = label;
    elements.statusDot.className = "status-dot";
    if (tone === "good" || tone === "bad") {
      elements.statusDot.classList.add(tone);
    }
  }

  function updateMetrics() {
    elements.taskMetric.textContent = status.taskId || "-";
    elements.modelMetric.textContent = String(status.modelVersion || 0);
    elements.stepMetric.textContent = String(status.globalStep || 0);
    elements.lossMetric.textContent = status.loss === null ? "-" : Number(status.loss).toFixed(6);
    elements.heartbeatMetric.textContent = String(status.heartbeats);
    elements.workerMetric.textContent = status.workerElapsedMs
      ? `${Math.round(status.workerElapsedMs)} ms`
      : "-";
  }

  async function postJson(path, payload) {
    const headers = { "content-type": "application/json" };
    const token = minerToken();
    if (token) {
      headers["x-crowdtensor-miner-token"] = token;
    }
    const response = await fetch(`${coordinatorUrl()}${path}`, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });
    const text = await response.text();
    const body = text ? JSON.parse(text) : {};
    if (!response.ok) {
      const error = new Error(`HTTP ${response.status}: ${text}`);
      error.status = response.status;
      error.body = body;
      throw error;
    }
    return body;
  }

  function runWorker(claim) {
    return new Promise((resolve, reject) => {
      const worker = new Worker(new URL("./diloco_worker.js", window.location.href));
      worker.onmessage = (event) => {
        const message = event.data || {};
        worker.terminate();
        if (message.type === "training-result") {
          resolve(message);
        } else {
          reject(new Error(message.error || "worker training failed"));
        }
      };
      worker.onerror = (event) => {
        worker.terminate();
        reject(new Error(event.message || "worker crashed"));
      };
      worker.postMessage({
        type: "train",
        claim,
        minerId: minerId(),
        holdMs,
      });
    });
  }

  function startHeartbeat(claim) {
    const intervalSeconds = Math.max(0.5, Number(claim.heartbeat_interval || 5));
    activeHeartbeat = window.setInterval(async () => {
      try {
        await postJson(`/tasks/${claim.task_id}/heartbeat`, {
          lease_token: claim.lease_token,
          attempt: claim.attempt,
          runtime_status: {
            phase: status.phase,
            heartbeats: status.heartbeats,
            worker_elapsed_ms: status.workerElapsedMs,
            workload_type: status.workloadType,
            result_mode: status.resultMode,
            hold_ms: status.holdMs,
          },
        });
        status.heartbeats += 1;
        updateMetrics();
      } catch (error) {
        status.error = error.message;
        log(`heartbeat failed: ${error.message}`);
      }
    }, intervalSeconds * 1000);
  }

  function stopHeartbeat() {
    if (activeHeartbeat !== null) {
      window.clearInterval(activeHeartbeat);
      activeHeartbeat = null;
    }
  }

  function resetAttemptState() {
    status.claimed = false;
    status.accepted = false;
    status.rejected = false;
    status.taskId = "";
    status.attempt = 0;
    status.leaseExpiresAt = 0;
    status.error = "";
    status.workerElapsedMs = 0;
    status.innerLossStart = null;
    status.innerLossEnd = null;
    status.workloadType = "";
    status.probeHash = "";
    status.probeGops = 0;
    updateMetrics();
  }

  function makeProbeBuffer(byteLength) {
    const bytes = Math.max(4, Number(byteLength || 1048576) - (Number(byteLength || 1048576) % 4));
    const floats = new Float32Array(bytes / 4);
    for (let index = 0; index < floats.length; index += 1) {
      floats[index] = Math.sin(index % 1024) * 0.5 + (index % 17) / 17;
    }
    return floats.buffer;
  }

  function runProbeWorker(claim) {
    return new Promise((resolve, reject) => {
      const spec = claim.workload_spec || {};
      const worker = new Worker(new URL("./compute_worker.js", window.location.href));
      worker.onmessage = (event) => {
        const message = event.data || {};
        worker.terminate();
        if (message.type === "compute-result") {
          if (message.error) {
            reject(new Error(message.error));
          } else {
            resolve({
              ...message,
              elapsed_ms: Number(message.elapsedMs || message.elapsed_ms || 0),
            });
          }
        } else {
          reject(new Error(message.error || "probe worker failed"));
        }
      };
      worker.onerror = (event) => {
        worker.terminate();
        reject(new Error(event.message || "probe worker crashed"));
      };
      const buffer = makeProbeBuffer(spec.bytes || 1048576);
      worker.postMessage({
        type: "compute",
        buffer,
        cols: spec.cols || 1024,
        iterations: spec.iterations || 8,
      }, [buffer]);
    });
  }

  async function runOnce() {
    if (status.running) return;
    status.running = true;
    stopRequested = false;
    resetAttemptState();
    setStatus("claiming", "warn");

    try {
      const claim = await postJson("/tasks/claim", {
        miner_id: minerId(),
        capabilities: minerCapabilities(),
      });
      status.claimed = true;
      status.taskId = claim.task_id;
      status.attempt = claim.attempt;
      status.leaseExpiresAt = claim.lease_expires_at;
      status.modelVersion = claim.model_version;
      status.workloadType = claim.workload_type || "diloco_train";
      updateMetrics();
      log(`claimed task=${claim.task_id} workload=${status.workloadType} model_version=${claim.model_version}`);

      startHeartbeat(claim);
      if (status.workloadType === "browser_probe") {
        setStatus("probing", "warn");
        const probe = await runProbeWorker(claim);
        stopHeartbeat();

        status.workerElapsedMs = probe.elapsed_ms;
        status.probeHash = probe.hash || "";
        status.probeGops = Number(probe.gops || 0);
        updateMetrics();

        if (stopRequested) {
          setStatus("stopped", "warn");
          return;
        }

        setStatus("uploading", "warn");
        const result = await postJson(`/tasks/${claim.task_id}/result`, {
          lease_token: claim.lease_token,
          attempt: claim.attempt,
          probe_result: probe,
          metrics: {
            backend: probe.backend,
            elapsed_ms: probe.elapsed_ms,
            gops: probe.gops,
            hash: probe.hash,
            ops: probe.ops,
            workload_type: status.workloadType,
          },
        });

        status.accepted = true;
        status.globalStep = result.global_step;
        status.modelVersion = result.model_version;
        status.optimizerStep = result.optimizer_step;
        status.loss = result.loss;
        updateMetrics();
        setStatus("accepted", "good");
        log(`accepted probe task=${claim.task_id} hash=${status.probeHash} gops=${status.probeGops.toFixed(3)}`);
        return;
      }

      if (status.workloadType !== "diloco_train") {
        throw new Error(`unsupported workload_type ${status.workloadType}`);
      }

      setStatus("training", "warn");
      const training = await runWorker(claim);
      stopHeartbeat();

      status.workerElapsedMs = training.metrics.elapsed_ms;
      status.innerLossStart = training.metrics.inner_loss_start;
      status.innerLossEnd = training.metrics.inner_loss_end;
      updateMetrics();

      if (stopRequested) {
        setStatus("stopped", "warn");
        return;
      }

      setStatus("uploading", "warn");
      const localDelta = resultMode === "large_delta"
        ? claim.weights.map(() => 1000.0)
        : training.local_delta;
      const result = await postJson(`/tasks/${claim.task_id}/result`, {
        lease_token: claim.lease_token,
        attempt: claim.attempt,
        local_delta: localDelta,
        metrics: {
          ...training.metrics,
          result_mode: resultMode,
        },
      });

      status.accepted = true;
      status.globalStep = result.global_step;
      status.modelVersion = result.model_version;
      status.optimizerStep = result.optimizer_step;
      status.loss = result.loss;
      updateMetrics();
      setStatus("accepted", "good");
      log(`accepted task=${claim.task_id} global_step=${result.global_step} loss=${result.loss.toFixed(6)}`);
    } catch (error) {
      stopHeartbeat();
      status.error = error.message || String(error);
      status.rejected = error.status === 422;
      setStatus(status.rejected ? "rejected" : "failed", "bad");
      log(status.error);
    } finally {
      stopHeartbeat();
      status.running = false;
      updateMetrics();
    }
  }

  async function runLoop() {
    if (status.running) return;
    status.autoLoop = true;
    stopRequested = false;
    while (status.autoLoop && !stopRequested) {
      await runOnce();
      if (!status.autoLoop || stopRequested) break;
      await new Promise((resolve) => window.setTimeout(resolve, 500));
    }
  }

  function stop() {
    stopRequested = true;
    status.autoLoop = false;
    stopHeartbeat();
    setStatus("stopping", "warn");
  }

  function boot() {
    elements.coordinatorInput.value = params.get("coordinator") || "http://127.0.0.1:8787";
    elements.minerIdInput.value = params.get("miner_id") || defaultMinerId;
    elements.minerTokenInput.value = params.get("miner_token") || "";
    elements.runOnceButton.addEventListener("click", () => runOnce());
    elements.autoLoopButton.addEventListener("click", () => runLoop());
    elements.stopButton.addEventListener("click", () => stop());
    updateMetrics();
    setStatus("idle", "warn");
    log(`ready miner_id=${minerId()}`);
    if (params.get("autorun") === "1") {
      runOnce();
    }
  }

  boot();
})();
