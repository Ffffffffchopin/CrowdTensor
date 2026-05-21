(() => {
  "use strict";

  const DEFAULT_BYTES = 5 * 1024 * 1024;
  const DEFAULT_CHUNK_BYTES = 16 * 1024;
  const MAX_BUFFERED_BYTES = 4 * 1024 * 1024;
  const LOW_BUFFERED_BYTES = 512 * 1024;

  const params = new URLSearchParams(window.location.search);
  const role = params.get("role") || "";
  const room = params.get("room") || "demo";
  const requestedBytes = Number.parseInt(params.get("bytes") || String(DEFAULT_BYTES), 10);
  const tensorBytes = Math.max(4, requestedBytes - (requestedBytes % 4));
  const chunkBytes = Math.max(1024, Number.parseInt(params.get("chunk") || String(DEFAULT_CHUNK_BYTES), 10));
  const computeMode = params.get("compute") === "off" ? "off" : "worker";
  const peerId = (crypto.randomUUID && crypto.randomUUID()) || `${Date.now()}-${Math.random()}`;
  const channelName = `crowdtensor-webrtc-${room}`;

  const elements = {
    roleLabel: document.getElementById("roleLabel"),
    roomLabel: document.getElementById("roomLabel"),
    statusDot: document.getElementById("statusDot"),
    statusLabel: document.getElementById("statusLabel"),
    headline: document.getElementById("headline"),
    subline: document.getElementById("subline"),
    progressBar: document.getElementById("progressBar"),
    bytesMetric: document.getElementById("bytesMetric"),
    chunksMetric: document.getElementById("chunksMetric"),
    mbpsMetric: document.getElementById("mbpsMetric"),
    elapsedMetric: document.getElementById("elapsedMetric"),
    checksumMetric: document.getElementById("checksumMetric"),
    verifiedMetric: document.getElementById("verifiedMetric"),
    computeMetric: document.getElementById("computeMetric"),
    computeTimeMetric: document.getElementById("computeTimeMetric"),
    computeRateMetric: document.getElementById("computeRateMetric"),
    sessionText: document.getElementById("sessionText"),
    senderLink: document.getElementById("senderLink"),
    receiverLink: document.getElementById("receiverLink"),
    restartButton: document.getElementById("restartButton"),
    log: document.getElementById("log"),
  };

  const status = {
    role,
    room,
    peerId,
    connected: false,
    verified: false,
    checksumMatch: false,
    bytesExpected: tensorBytes,
    bytesReceived: 0,
    bytesSent: 0,
    chunksExpected: Math.ceil(tensorBytes / chunkBytes),
    chunksReceived: 0,
    chunksSent: 0,
    elapsedMs: 0,
    mbps: 0,
    checksum: "",
    remoteChecksum: "",
    compute: {
      enabled: computeMode !== "off",
      backend: computeMode,
      verified: false,
      elapsedMs: 0,
      ops: 0,
      gops: 0,
      hash: "",
      error: "",
    },
    error: "",
  };
  window.__crowdTensorWebrtcStatus = status;

  let broadcast;
  let pc;
  let dataChannel;
  let helloTimer;
  let offered = false;
  let sendStarted = false;
  let receiveBuffer = null;
  let receiveHeader = null;
  let receiveOffset = 0;
  let receiveStartedAt = 0;
  const pendingCandidates = [];

  function log(message) {
    const line = document.createElement("div");
    line.textContent = `[${new Date().toISOString().slice(11, 19)}] ${message}`;
    elements.log.appendChild(line);
    elements.log.scrollTop = elements.log.scrollHeight;
  }

  function setStatus(label, tone = "warn") {
    elements.statusLabel.textContent = label;
    elements.statusDot.className = "status-dot";
    if (tone === "good" || tone === "bad") {
      elements.statusDot.classList.add(tone);
    }
  }

  function updateMetrics() {
    const bytes = Math.max(status.bytesReceived, status.bytesSent);
    const chunks = Math.max(status.chunksReceived, status.chunksSent);
    const progressBytes = status.bytesReceived || status.bytesSent;
    const progress = status.bytesExpected ? Math.min(100, (progressBytes / status.bytesExpected) * 100) : 0;
    elements.progressBar.style.width = `${progress.toFixed(2)}%`;
    elements.bytesMetric.textContent = `${bytes.toLocaleString()} / ${status.bytesExpected.toLocaleString()}`;
    elements.chunksMetric.textContent = `${chunks.toLocaleString()} / ${status.chunksExpected.toLocaleString()}`;
    elements.mbpsMetric.textContent = `${status.mbps.toFixed(2)} MB/s`;
    elements.elapsedMetric.textContent = `${Math.round(status.elapsedMs)} ms`;
    elements.checksumMetric.textContent = status.checksum || status.remoteChecksum || "-";
    elements.verifiedMetric.textContent = String(status.verified);
    if (elements.computeMetric) {
      if (!status.compute.enabled) {
        elements.computeMetric.textContent = "off";
      } else if (status.compute.error) {
        elements.computeMetric.textContent = "failed";
      } else if (status.compute.verified) {
        elements.computeMetric.textContent = status.compute.backend;
      } else {
        elements.computeMetric.textContent = "pending";
      }
    }
    if (elements.computeTimeMetric) {
      elements.computeTimeMetric.textContent = status.compute.elapsedMs
        ? `${Math.round(status.compute.elapsedMs)} ms`
        : "-";
    }
    if (elements.computeRateMetric) {
      elements.computeRateMetric.textContent = status.compute.gops
        ? `${status.compute.gops.toFixed(3)} GOPS`
        : "-";
    }
  }

  function checksumBytes(buffer) {
    const bytes = new Uint8Array(buffer);
    let hash = 2166136261 >>> 0;
    for (let index = 0; index < bytes.length; index += 1) {
      hash ^= bytes[index];
      hash = Math.imul(hash, 16777619) >>> 0;
    }
    return hash.toString(16).padStart(8, "0");
  }

  function makeTensorBuffer(byteLength) {
    const floats = new Float32Array(byteLength / 4);
    for (let index = 0; index < floats.length; index += 1) {
      floats[index] = Math.sin(index % 1024) * 0.5 + (index % 17) / 17;
    }
    return floats.buffer;
  }

  function signal(payload) {
    if (!broadcast) return;
    const message = { ...payload };
    if (message.sdp && typeof message.sdp.toJSON === "function") {
      message.sdp = message.sdp.toJSON();
    }
    if (message.candidate && typeof message.candidate.toJSON === "function") {
      message.candidate = message.candidate.toJSON();
    }
    broadcast.postMessage({
      ...message,
      room,
      from: peerId,
      role,
    });
  }

  function canUseSignal(message) {
    if (!message || message.room !== room || message.from === peerId) return false;
    if (message.to && message.to !== peerId) return false;
    return true;
  }

  async function addOrQueueCandidate(candidate) {
    if (!candidate) return;
    if (!pc.remoteDescription) {
      pendingCandidates.push(candidate);
      return;
    }
    try {
      await pc.addIceCandidate(candidate);
    } catch (error) {
      log(`candidate rejected: ${error.message}`);
    }
  }

  async function drainCandidates() {
    while (pendingCandidates.length > 0 && pc.remoteDescription) {
      await addOrQueueCandidate(pendingCandidates.shift());
    }
  }

  function createPeer() {
    const peer = new RTCPeerConnection({ iceServers: [] });
    peer.onicecandidate = (event) => {
      if (event.candidate) {
        signal({ type: "candidate", candidate: event.candidate });
      }
    };
    peer.onconnectionstatechange = () => {
      log(`peer state: ${peer.connectionState}`);
      if (peer.connectionState === "connected") {
        status.connected = true;
        setStatus("connected", "good");
      } else if (["failed", "closed", "disconnected"].includes(peer.connectionState)) {
        setStatus(peer.connectionState, peer.connectionState === "failed" ? "bad" : "warn");
      }
    };
    peer.ondatachannel = (event) => {
      setupDataChannel(event.channel);
    };
    return peer;
  }

  function setupDataChannel(channel) {
    dataChannel = channel;
    dataChannel.binaryType = "arraybuffer";
    dataChannel.bufferedAmountLowThreshold = LOW_BUFFERED_BYTES;
    dataChannel.onopen = () => {
      status.connected = true;
      setStatus("datachannel open", "good");
      log("datachannel open");
      if (role === "sender" && !sendStarted) {
        sendTensor().catch((error) => fail(error));
      }
    };
    dataChannel.onmessage = (event) => {
      handleDataMessage(event.data).catch((error) => fail(error));
    };
    dataChannel.onclose = () => log("datachannel closed");
  }

  async function waitForBufferedDrain() {
    if (!dataChannel || dataChannel.bufferedAmount <= MAX_BUFFERED_BYTES) return;
    await new Promise((resolve) => {
      dataChannel.onbufferedamountlow = () => {
        dataChannel.onbufferedamountlow = null;
        resolve();
      };
    });
  }

  async function sendTensor() {
    sendStarted = true;
    const startedAt = performance.now();
    setStatus("sending", "warn");
    log(`generating ${tensorBytes.toLocaleString()} bytes`);
    const buffer = makeTensorBuffer(tensorBytes);
    const checksum = checksumBytes(buffer);
    status.checksum = checksum;
    dataChannel.send(JSON.stringify({
      type: "tensor-header",
      bytes: tensorBytes,
      floats: tensorBytes / 4,
      chunkBytes,
      checksum,
      sentAt: Date.now(),
    }));

    for (let offset = 0; offset < tensorBytes; offset += chunkBytes) {
      const end = Math.min(offset + chunkBytes, tensorBytes);
      dataChannel.send(buffer.slice(offset, end));
      status.bytesSent = end;
      status.chunksSent += 1;
      status.elapsedMs = performance.now() - startedAt;
      status.mbps = (status.bytesSent / (1024 * 1024)) / Math.max(status.elapsedMs / 1000, 0.001);
      updateMetrics();
      await waitForBufferedDrain();
    }

    dataChannel.send(JSON.stringify({
      type: "tensor-done",
      bytes: tensorBytes,
      chunks: status.chunksSent,
      checksum,
      elapsedMs: performance.now() - startedAt,
    }));
    log(`sent ${status.chunksSent} chunks checksum=${checksum}`);
    setStatus("sent", "good");
  }

  async function handleDataMessage(data) {
    if (typeof data === "string") {
      const message = JSON.parse(data);
      if (message.type === "tensor-header") {
        receiveHeader = message;
        receiveBuffer = new Uint8Array(message.bytes);
        receiveOffset = 0;
        receiveStartedAt = performance.now();
        status.bytesExpected = message.bytes;
        status.chunksExpected = Math.ceil(message.bytes / message.chunkBytes);
        status.remoteChecksum = message.checksum;
        setStatus("receiving", "warn");
        log(`header bytes=${message.bytes} chunks=${status.chunksExpected} checksum=${message.checksum}`);
      } else if (message.type === "tensor-done") {
        await finishReceive(message);
      } else if (message.type === "ack") {
        if (message.compute) {
          status.compute = { ...status.compute, ...message.compute };
          updateMetrics();
        }
        log(`receiver ack verified=${message.verified} compute=${message.compute ? message.compute.verified : "n/a"}`);
      }
      return;
    }

    if (!receiveBuffer || !receiveHeader) {
      throw new Error("received binary chunk before tensor header");
    }
    const chunk = new Uint8Array(data);
    receiveBuffer.set(chunk, receiveOffset);
    receiveOffset += chunk.byteLength;
    status.bytesReceived = receiveOffset;
    status.chunksReceived += 1;
    status.elapsedMs = performance.now() - receiveStartedAt;
    status.mbps = (status.bytesReceived / (1024 * 1024)) / Math.max(status.elapsedMs / 1000, 0.001);
    updateMetrics();
  }

  function runComputeProbe(buffer) {
    if (!status.compute.enabled) {
      updateMetrics();
      return Promise.resolve();
    }

    if (!window.Worker) {
      status.compute.error = "Worker API unavailable";
      updateMetrics();
      return Promise.resolve();
    }

    status.compute = {
      ...status.compute,
      backend: "js-worker",
      verified: false,
      elapsedMs: 0,
      ops: 0,
      gops: 0,
      hash: "",
      error: "",
    };
    setStatus("computing", "warn");
    updateMetrics();

    return new Promise((resolve) => {
      const worker = new Worker(new URL("./compute_worker.js", window.location.href));
      const timeout = window.setTimeout(() => {
        worker.terminate();
        status.compute.error = "compute timeout";
        updateMetrics();
        resolve();
      }, 10000);

      worker.onmessage = (event) => {
        window.clearTimeout(timeout);
        const result = event.data || {};
        status.compute = {
          ...status.compute,
          ...result,
          enabled: true,
        };
        worker.terminate();
        updateMetrics();
        resolve();
      };

      worker.onerror = (event) => {
        window.clearTimeout(timeout);
        status.compute.error = event.message || "compute worker failed";
        worker.terminate();
        updateMetrics();
        resolve();
      };

      worker.postMessage({
        type: "compute",
        buffer,
        cols: 1024,
        iterations: 8,
      }, [buffer]);
    });
  }

  async function finishReceive(doneMessage) {
    const elapsed = performance.now() - receiveStartedAt;
    const checksum = checksumBytes(receiveBuffer.buffer);
    const bytesMatch = receiveOffset === receiveHeader.bytes && receiveOffset === doneMessage.bytes;
    const checksumMatch = checksum === receiveHeader.checksum && checksum === doneMessage.checksum;
    status.checksum = checksum;
    status.checksumMatch = checksumMatch;
    status.verified = bytesMatch && checksumMatch;
    status.elapsedMs = elapsed;
    status.mbps = (receiveOffset / (1024 * 1024)) / Math.max(elapsed / 1000, 0.001);
    updateMetrics();
    setStatus(status.verified ? "verified" : "failed", status.verified ? "good" : "bad");
    log(`done bytesMatch=${bytesMatch} checksumMatch=${checksumMatch} checksum=${checksum}`);
    if (status.verified) {
      await runComputeProbe(receiveBuffer.buffer);
      if (status.compute.enabled) {
        setStatus(status.compute.verified ? "computed" : "failed", status.compute.verified ? "good" : "bad");
        log(`compute backend=${status.compute.backend} verified=${status.compute.verified} hash=${status.compute.hash || "-"}`);
      }
    }
    if (dataChannel && dataChannel.readyState === "open") {
      dataChannel.send(JSON.stringify({
        type: "ack",
        verified: status.verified,
        bytes: receiveOffset,
        checksum,
        elapsedMs: elapsed,
        compute: status.compute,
      }));
    }
  }

  async function maybeOffer() {
    if (role !== "sender" || offered || !pc || pc.signalingState !== "stable") return;
    offered = true;
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    signal({ type: "offer", sdp: pc.localDescription });
    log("offer sent");
  }

  async function handleSignal(message) {
    if (!canUseSignal(message)) return;
    if (message.type === "hello") {
      if (role === "sender" && message.role === "receiver") {
        await maybeOffer();
      }
      return;
    }
    if (message.type === "offer" && role === "receiver") {
      log("offer received");
      await pc.setRemoteDescription(message.sdp);
      await drainCandidates();
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      signal({ type: "answer", sdp: pc.localDescription, to: message.from });
      log("answer sent");
      return;
    }
    if (message.type === "answer" && role === "sender") {
      log("answer received");
      await pc.setRemoteDescription(message.sdp);
      await drainCandidates();
      return;
    }
    if (message.type === "candidate") {
      await addOrQueueCandidate(message.candidate);
    }
  }

  function fail(error) {
    status.error = error.message || String(error);
    setStatus("failed", "bad");
    log(`error: ${status.error}`);
    console.error(error);
  }

  function setupLinks() {
    const base = `${window.location.pathname}?room=${encodeURIComponent(room)}&bytes=${tensorBytes}&chunk=${chunkBytes}&compute=${computeMode}`;
    elements.senderLink.href = `${base}&role=sender`;
    elements.receiverLink.href = `${base}&role=receiver`;
    elements.restartButton.addEventListener("click", () => window.location.reload());
  }

  async function boot() {
    setupLinks();
    elements.roleLabel.textContent = `role: ${role || "launcher"}`;
    elements.roomLabel.textContent = `room: ${room}`;
    updateMetrics();

    if (!role) {
      elements.headline.textContent = "Open a sender and receiver";
      elements.subline.textContent = "Both tabs use BroadcastChannel for local signaling, WebRTC DataChannel for transport, and an optional Worker probe for compute.";
      setStatus("launcher", "warn");
      return;
    }

    if (!["sender", "receiver"].includes(role)) {
      throw new Error(`unsupported role: ${role}`);
    }

    elements.headline.textContent = role === "sender" ? "Sender armed" : "Receiver waiting";
    elements.subline.textContent = `${tensorBytes.toLocaleString()} bytes in ${Math.ceil(tensorBytes / chunkBytes)} chunks; compute=${computeMode}.`;
    elements.sessionText.textContent = `Signaling room ${room}; peer ${peerId.slice(0, 8)}.`;
    setStatus("signaling", "warn");

    broadcast = new BroadcastChannel(channelName);
    broadcast.onmessage = (event) => {
      handleSignal(event.data).catch((error) => fail(error));
    };

    pc = createPeer();
    if (role === "sender") {
      setupDataChannel(pc.createDataChannel("tensor", { ordered: true }));
    }

    helloTimer = window.setInterval(() => {
      if (!status.connected) signal({ type: "hello" });
      else window.clearInterval(helloTimer);
    }, 500);
    signal({ type: "hello" });
    log(`boot role=${role} room=${room} bytes=${tensorBytes}`);
  }

  boot().catch((error) => fail(error));
})();
