"use strict";

const number = new Intl.NumberFormat("en-US");

function text(id, value) {
  const element = document.getElementById(id);
  if (element) {
    element.textContent = String(value);
  }
}

function boundedPercent(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return 0;
  }
  return Math.max(0, Math.min(100, parsed));
}

function humanizeCampaignId(value) {
  const words = String(value || "")
    .replace(/^crowdtensor-/, "")
    .split(/[-_]+/)
    .filter(Boolean);
  if (!words.length) {
    return "Founding systems campaign";
  }
  return words.map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join(" ");
}

function lifecycleLabel(value, complete) {
  if (complete) {
    return "Campaign complete";
  }
  const labels = {
    running: "Coordinator live",
    paused: "Campaign paused",
    created: "Preparing enrollment",
    finalized: "Campaign finalized",
  };
  return labels[String(value || "").toLowerCase()] || "Preparing enrollment";
}

function renderSnapshot(snapshot) {
  const campaign = snapshot && snapshot.campaign ? snapshot.campaign : {};
  const progress = snapshot && snapshot.progress ? snapshot.progress : {};
  const completed = Number(progress.completed_rounds || 0);
  const target = Math.max(1, Number(progress.target_rounds || 1));
  const fraction = boundedPercent(
    Number.isFinite(Number(progress.progress_fraction))
      ? Number(progress.progress_fraction) * 100
      : (completed / target) * 100,
  );
  const lifecycle = String(progress.lifecycle || "created").toLowerCase();
  const status = document.getElementById("campaign-status");

  text("campaign-title", humanizeCampaignId(campaign.campaign_id));
  text("campaign-model", campaign.model_id || "Pinned model pending");
  text("campaign-dataset", campaign.dataset_id || "Pinned dataset pending");
  text("round-progress", `${number.format(completed)} / ${number.format(target)}`);
  text("accepted-updates", number.format(Number(progress.accepted_update_count || 0)));
  text("accepted-tokens", number.format(Number(progress.accepted_token_count || 0)));
  text("active-contributors", number.format(Number(progress.active_contributor_count || 0)));
  text("adapter-version", `v${number.format(Number(progress.adapter_version || 0))}`);
  text("snapshot-state", "Live public snapshot");

  if (status) {
    status.dataset.state = progress.campaign_complete ? "complete" : lifecycle;
    const dot = status.querySelector(".status-dot");
    status.textContent = "";
    if (dot) {
      status.appendChild(dot);
    } else {
      const replacement = document.createElement("span");
      replacement.className = "status-dot";
      replacement.setAttribute("aria-hidden", "true");
      status.appendChild(replacement);
    }
    status.appendChild(document.createTextNode(lifecycleLabel(lifecycle, progress.campaign_complete)));
  }

  const fill = document.getElementById("progress-fill");
  if (fill) {
    fill.style.width = `${fraction}%`;
  }
  const track = document.getElementById("progress-track");
  if (track) {
    track.setAttribute("aria-valuenow", String(Math.round(fraction)));
  }
}

function renderUnavailable() {
  text("snapshot-state", "Public enrollment is being prepared");
}

async function loadCampaign() {
  try {
    const response = await fetch("/v1/volunteer/public-snapshot", {
      cache: "no-store",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      throw new Error(`snapshot_http_${response.status}`);
    }
    const snapshot = await response.json();
    if (!snapshot || snapshot.ok !== true) {
      throw new Error("snapshot_not_ready");
    }
    renderSnapshot(snapshot);
  } catch (_error) {
    renderUnavailable();
  }
}

loadCampaign();
