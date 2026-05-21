(() => {
  "use strict";

  const SCHEMA_VERSION = "diloco_mock_v1";
  const LOCAL_DELTA_SCALE = 0.1;
  const TARGET_WEIGHTS = [1.0, -2.0, 0.5];
  const FEATURES = [
    [1.0, 0.0, 0.5],
    [0.5, -1.0, 1.0],
    [-1.0, 0.25, 0.75],
    [0.25, 1.0, -0.5],
    [1.5, -0.5, 0.25],
    [-0.75, -1.25, 1.0],
  ];
  const TARGETS = FEATURES.map((features) => (
    TARGET_WEIGHTS.reduce((total, weight, index) => total + weight * features[index], 0)
  ));
  const DEFAULT_INNER_LR = 0.03;

  function stableOffset(...parts) {
    let hash = 2166136261 >>> 0;
    const payload = parts.map((part) => String(part)).join(":");
    for (let index = 0; index < payload.length; index += 1) {
      hash ^= payload.charCodeAt(index);
      hash = Math.imul(hash, 16777619) >>> 0;
    }
    return hash >>> 0;
  }

  function predict(weights, features) {
    return weights.reduce((total, weight, index) => total + weight * features[index], 0);
  }

  function syntheticLoss(weights, featuresSet = FEATURES, targets = TARGETS) {
    let total = 0;
    for (let index = 0; index < featuresSet.length; index += 1) {
      const error = predict(weights, featuresSet[index]) - targets[index];
      total += error * error;
    }
    return Math.sqrt(total / featuresSet.length);
  }

  function localGradient(weights, sampleIndex, featuresSet = FEATURES, targets = TARGETS) {
    const index = sampleIndex % featuresSet.length;
    const features = featuresSet[index];
    const error = predict(weights, features) - targets[index % targets.length];
    return features.map((feature) => 2.0 * error * feature);
  }

  function busyWaitUntil(startedAt, holdMs) {
    if (holdMs <= 0) return;
    while (performance.now() - startedAt < holdMs) {
      Math.imul(17, 31);
    }
  }

  function resolveTrainingSpec(claim, minerId) {
    const spec = claim.training_spec || {};
    const features = Array.isArray(spec.features) && spec.features.length > 0 ? spec.features : FEATURES;
    const targets = Array.isArray(spec.targets) && spec.targets.length > 0 ? spec.targets : TARGETS;
    const modelVersion = Number.parseInt(claim.model_version || 0, 10);
    const fallbackOffset = stableOffset(claim.task_id, minerId, modelVersion) % features.length;
    return {
      schemaVersion: spec.schema_version || SCHEMA_VERSION,
      features,
      targets,
      innerLr: Number(spec.inner_lr || DEFAULT_INNER_LR),
      localDeltaScale: Number(spec.local_delta_scale || DEFAULT_LOCAL_DELTA_SCALE),
      sampleOffset: Number.isFinite(Number(spec.sample_offset)) ? Number(spec.sample_offset) : fallbackOffset,
    };
  }

  function runInnerLoop(claim, minerId, holdMs) {
    const spec = resolveTrainingSpec(claim, minerId);
    const initial = claim.weights.map((value) => Number(value));
    const local = initial.slice();
    const innerSteps = Math.max(1, Number.parseInt(claim.inner_steps || 1, 10));
    const startedAt = performance.now();
    const innerLossStart = syntheticLoss(local, spec.features, spec.targets);

    for (let step = 0; step < innerSteps; step += 1) {
      const gradient = localGradient(local, spec.sampleOffset + step, spec.features, spec.targets);
      for (let index = 0; index < local.length; index += 1) {
        local[index] -= spec.innerLr * gradient[index];
      }
    }
    busyWaitUntil(startedAt, Math.max(0, Number(holdMs || 0)));

    const localDelta = local.map((value, index) => (value - initial[index]) * spec.localDeltaScale);
    return {
      schema_version: spec.schemaVersion,
      local_delta: localDelta,
      metrics: {
        backend: "browser-js-worker",
        inner_loss_start: innerLossStart,
        inner_loss_end: syntheticLoss(local, spec.features, spec.targets),
        samples_seen: innerSteps,
        inner_steps: innerSteps,
        inner_lr: spec.innerLr,
        elapsed_ms: performance.now() - startedAt,
        hold_ms: Math.max(0, Number(holdMs || 0)),
        local_delta_scale: spec.localDeltaScale,
        sample_offset: spec.sampleOffset,
      },
    };
  }

  self.onmessage = (event) => {
    const message = event.data || {};
    if (message.type !== "train") return;

    try {
      self.postMessage({
        type: "training-result",
        ...runInnerLoop(message.claim, message.minerId || "browser-miner", message.holdMs || 0),
      });
    } catch (error) {
      self.postMessage({
        type: "training-error",
        error: error.message || String(error),
      });
    }
  };
})();
