"use strict";

const endpoint = "/v1/volunteer/public-snapshot";
const state = { snapshot: null, timer: null };

const elements = Object.fromEntries(
  [
    "connectionState",
    "refreshButton",
    "campaignProfile",
    "campaignTitle",
    "campaignPair",
    "lifecycleValue",
    "roundMetric",
    "roundMetricDetail",
    "updateMetric",
    "tokenMetricDetail",
    "activeMetric",
    "queueMetricDetail",
    "adapterMetric",
    "outerStepDetail",
    "progressPercent",
    "progressFill",
    "roundCanvas",
    "activityList",
    "expiredCount",
    "reassignedCount",
    "recoveryCount",
    "rejectedCount",
    "roundsBody",
    "sourceList",
    "checkpointList",
    "boundaryList",
    "updatedAt",
  ].map((id) => [id, document.getElementById(id)])
);

function formatNumber(value) {
  return new Intl.NumberFormat("en-US").format(Number(value || 0));
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let amount = bytes;
  let unit = -1;
  do {
    amount /= 1024;
    unit += 1;
  } while (amount >= 1024 && unit < units.length - 1);
  return `${amount.toFixed(amount >= 10 ? 1 : 2)} ${units[unit]}`;
}

function formatDuration(startedAt, completedAt) {
  if (!startedAt || !completedAt || completedAt < startedAt) return "In progress";
  const seconds = Math.max(0, completedAt - startedAt);
  if (seconds < 60) return `${Math.round(seconds)} sec`;
  if (seconds < 3600) return `${Math.round(seconds / 60)} min`;
  return `${(seconds / 3600).toFixed(1)} hr`;
}

function formatTime(value) {
  if (!value) return "Time unavailable";
  return new Intl.DateTimeFormat("en", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(Number(value) * 1000));
}

function humanize(value) {
  return String(value || "event")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function shortHash(value) {
  const text = String(value || "");
  if (text.length <= 24) return text || "Not reported";
  return `${text.slice(0, 15)}...${text.slice(-8)}`;
}

function setConnection(label, tone) {
  elements.connectionState.textContent = label;
  elements.connectionState.dataset.tone = tone;
}

function replaceChildren(parent, children) {
  parent.replaceChildren(...children);
}

function detailRow(term, description) {
  const wrapper = document.createElement("div");
  const dt = document.createElement("dt");
  const dd = document.createElement("dd");
  dt.textContent = term;
  dd.textContent = description;
  wrapper.append(dt, dd);
  return wrapper;
}

function drawRoundChart(rounds) {
  const canvas = elements.roundCanvas;
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(320, canvas.clientWidth);
  const height = Math.max(220, canvas.clientHeight);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);

  context.fillStyle = "#647069";
  context.font = "12px Inter, system-ui, sans-serif";
  if (!rounds.length) {
    context.fillText("No completed rounds", 16, 34);
    return;
  }

  const padding = { top: 22, right: 16, bottom: 42, left: 42 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  const maximum = Math.max(1, ...rounds.map((round) => Number(round.work_unit_count || 0)));
  const slot = chartWidth / rounds.length;
  const barWidth = Math.max(20, Math.min(64, slot * 0.58));

  context.strokeStyle = "#d9dfda";
  context.lineWidth = 1;
  for (let line = 0; line <= 4; line += 1) {
    const y = padding.top + (chartHeight * line) / 4;
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(width - padding.right, y);
    context.stroke();
  }

  rounds.forEach((round, index) => {
    const accepted = Number(round.accepted_update_count || 0);
    const barHeight = (accepted / maximum) * chartHeight;
    const x = padding.left + slot * index + (slot - barWidth) / 2;
    const y = padding.top + chartHeight - barHeight;
    context.fillStyle = round.state === "completed" ? "#16734b" : "#1e5aa8";
    context.fillRect(x, y, barWidth, Math.max(3, barHeight));
    context.fillStyle = "#647069";
    context.textAlign = "center";
    context.fillText(`R${Number(round.round_index) + 1}`, x + barWidth / 2, height - 16);
    context.fillStyle = "#17201c";
    context.fillText(String(accepted), x + barWidth / 2, Math.max(15, y - 7));
  });
  context.textAlign = "left";
}

function renderActivity(events) {
  const rows = events
    .slice()
    .reverse()
    .map((event) => {
      const item = document.createElement("li");
      const dot = document.createElement("span");
      const content = document.createElement("span");
      const type = document.createElement("span");
      const time = document.createElement("span");
      dot.className = "activity-dot";
      type.className = "activity-type";
      time.className = "activity-time";
      type.textContent = humanize(event.event_type);
      time.textContent = `Event ${formatNumber(event.sequence)} / ${formatTime(event.recorded_at)}`;
      content.append(type, time);
      item.append(dot, content);
      return item;
    });
  if (!rows.length) {
    const empty = document.createElement("li");
    empty.className = "empty-row";
    empty.textContent = "No activity reported";
    rows.push(empty);
  }
  replaceChildren(elements.activityList, rows);
}

function renderRounds(rounds) {
  const rows = rounds.map((round) => {
    const row = document.createElement("tr");
    const values = [
      `Round ${Number(round.round_index) + 1}`,
      humanize(round.state),
      formatNumber(round.accepted_update_count),
      formatNumber(round.distinct_contributor_count),
      `v${round.adapter_version_before} -> v${round.adapter_version_after}`,
      formatDuration(round.started_at, round.completed_at),
    ];
    values.forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      if (index === 1) cell.className = "state-cell";
      row.append(cell);
    });
    return row;
  });
  if (!rows.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 6;
    cell.className = "empty-row";
    cell.textContent = "No rounds reported";
    row.append(cell);
    rows.push(row);
  }
  replaceChildren(elements.roundsBody, rows);
}

function renderProvenance(snapshot) {
  const campaign = snapshot.campaign || {};
  const provenance = snapshot.provenance || {};
  const model = provenance.model_source || {};
  const dataset = provenance.dataset_source || {};
  const sourceRows = [
    detailRow("Model", model.model_id || campaign.model_id || "Not reported"),
    detailRow("Model revision", model.revision || String(campaign.model_revision || 0)),
    detailRow("Model license", model.license || "Not reported"),
    detailRow("Dataset", dataset.dataset_id || campaign.dataset_id || "Not reported"),
    detailRow("Dataset revision", dataset.revision || String(campaign.dataset_revision || 0)),
    detailRow("Dataset license", (dataset.licenses || []).join(", ") || "Not reported"),
    detailRow("Dataset snapshot", shortHash(provenance.dataset_snapshot_hash)),
  ];
  const checkpointRows = [
    detailRow("Adapter version", `v${snapshot.progress.adapter_version || 0}`),
    detailRow("Initial adapter", shortHash(provenance.initial_adapter_hash)),
    detailRow("Canonical adapter", shortHash(provenance.canonical_adapter_hash)),
    detailRow("Audit ledger", shortHash(provenance.append_only_ledger_head_hash)),
    detailRow(
      "Lineage",
      snapshot.evaluation.checkpoint_lineage_verified ? "Verified" : "Not verified"
    ),
    detailRow("Uploaded deltas", formatBytes(snapshot.progress.uploaded_delta_bytes)),
  ];
  replaceChildren(elements.sourceList, sourceRows);
  replaceChildren(elements.checkpointList, checkpointRows);

  const boundary = snapshot.trust_boundary || {};
  const labels = [
    ["Permissionless", boundary.permissionless],
    ["Sybil resistance", boundary.sybil_resistance],
    ["Poisoning safety", boundary.semantic_poisoning_safety],
    ["Secure aggregation", boundary.secure_aggregation],
    ["Physical multi-host", boundary.physical_multi_host_verified],
    ["Quality improvement", boundary.quality_improvement_verified],
  ];
  replaceChildren(
    elements.boundaryList,
    labels.map(([label, verified]) => {
      const item = document.createElement("span");
      item.className = "boundary-item";
      item.textContent = `${label}: ${verified ? "verified" : "not claimed"}`;
      return item;
    })
  );
}

function render(snapshot) {
  state.snapshot = snapshot;
  const campaign = snapshot.campaign || {};
  const progress = snapshot.progress || {};
  const reliability = snapshot.reliability || {};
  const rounds = snapshot.rounds || [];
  const fraction = Math.min(1, Math.max(0, Number(progress.progress_fraction || 0)));

  elements.campaignProfile.textContent = humanize(campaign.campaign_profile || "custom campaign");
  elements.campaignTitle.textContent = campaign.campaign_id || "Volunteer campaign";
  elements.campaignPair.textContent = `${campaign.model_id || "Model"} / ${campaign.dataset_id || "Dataset"}`;
  elements.lifecycleValue.textContent = humanize(progress.lifecycle || "unknown");
  elements.roundMetric.textContent = `${formatNumber(progress.completed_rounds)} / ${formatNumber(progress.target_rounds)}`;
  elements.roundMetricDetail.textContent = `${Math.round(fraction * 100)}% of target rounds`;
  elements.updateMetric.textContent = formatNumber(progress.accepted_update_count);
  elements.tokenMetricDetail.textContent = `${formatNumber(progress.accepted_token_count)} training tokens`;
  elements.activeMetric.textContent = formatNumber(progress.active_contributor_count);
  elements.queueMetricDetail.textContent = `${formatNumber(progress.queued_work_count)} work units queued`;
  elements.adapterMetric.textContent = `v${formatNumber(progress.adapter_version)}`;
  elements.outerStepDetail.textContent = `Outer step ${formatNumber(progress.outer_step)}`;
  elements.progressPercent.textContent = `${Math.round(fraction * 100)}%`;
  elements.progressFill.style.width = `${fraction * 100}%`;
  elements.expiredCount.textContent = formatNumber(reliability.expired_lease_count);
  elements.reassignedCount.textContent = formatNumber(reliability.reassigned_work_count);
  elements.recoveryCount.textContent = formatNumber(reliability.coordinator_recovery_count);
  elements.rejectedCount.textContent = formatNumber(reliability.rejected_update_count);
  elements.updatedAt.textContent = `Updated ${new Intl.DateTimeFormat("en", { hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(new Date())}`;

  drawRoundChart(rounds);
  renderActivity(snapshot.activity || []);
  renderRounds(rounds);
  renderProvenance(snapshot);
}

async function loadSnapshot() {
  setConnection("Refreshing", "loading");
  elements.refreshButton.disabled = true;
  try {
    const response = await fetch(endpoint, { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const snapshot = await response.json();
    if (!snapshot || snapshot.ok !== true) throw new Error("Invalid snapshot");
    render(snapshot);
    setConnection("Live", "ready");
  } catch (_error) {
    setConnection("Unavailable", "error");
  } finally {
    elements.refreshButton.disabled = false;
  }
}

document.querySelectorAll(".tab-button").forEach((button) => {
  button.addEventListener("click", () => {
    const view = button.dataset.view;
    document.querySelectorAll(".tab-button").forEach((candidate) => {
      candidate.setAttribute("aria-selected", String(candidate === button));
    });
    document.querySelectorAll("[data-view-panel]").forEach((panel) => {
      panel.hidden = panel.dataset.viewPanel !== view;
    });
    if (view === "overview" && state.snapshot) {
      requestAnimationFrame(() => drawRoundChart(state.snapshot.rounds || []));
    }
  });
});

elements.refreshButton.addEventListener("click", loadSnapshot);
window.addEventListener("resize", () => {
  if (state.snapshot) drawRoundChart(state.snapshot.rounds || []);
});

loadSnapshot();
state.timer = window.setInterval(loadSnapshot, 15000);
