(() => {
  "use strict";

  const DEFAULT_COLS = 1024;
  const DEFAULT_ITERATIONS = 8;

  function hashNumber(hash, value) {
    const scaled = Math.trunc(Math.abs(value) * 1000000) >>> 0;
    hash ^= scaled;
    return Math.imul(hash, 16777619) >>> 0;
  }

  function runWorkload(buffer, cols, iterations) {
    const floats = new Float32Array(buffer);
    const rows = Math.floor(floats.length / cols);
    if (rows <= 0) {
      throw new Error(`tensor too small for cols=${cols}`);
    }

    const startedAt = performance.now();
    let hash = 2166136261 >>> 0;
    let accumulator = 0;

    for (let iteration = 0; iteration < iterations; iteration += 1) {
      const iterationBias = (iteration + 1) * 0.0001;
      for (let row = 0; row < rows; row += 1) {
        const base = row * cols;
        let sum = 0;
        for (let col = 0; col < cols; col += 1) {
          const weight = ((col % 23) - 11) * 0.03125 + iterationBias;
          sum += floats[base + col] * weight;
        }
        accumulator += Math.tanh(sum) * 0.001;
        hash = hashNumber(hash, sum + accumulator);
      }
    }

    const elapsedMs = performance.now() - startedAt;
    const ops = rows * cols * iterations * 2;
    const gops = (ops / Math.max(elapsedMs / 1000, 0.001)) / 1000000000;
    if (!Number.isFinite(accumulator) || !Number.isFinite(gops)) {
      throw new Error("non-finite compute result");
    }

    return {
      type: "compute-result",
      backend: "js-worker",
      rows,
      cols,
      iterations,
      ops,
      elapsedMs,
      gops,
      hash: hash.toString(16).padStart(8, "0"),
      verified: ops > 0 && elapsedMs > 0,
    };
  }

  self.onmessage = (event) => {
    const message = event.data || {};
    if (message.type !== "compute") return;

    try {
      const cols = Math.max(1, Number.parseInt(message.cols || DEFAULT_COLS, 10));
      const iterations = Math.max(1, Number.parseInt(message.iterations || DEFAULT_ITERATIONS, 10));
      self.postMessage(runWorkload(message.buffer, cols, iterations));
    } catch (error) {
      self.postMessage({
        type: "compute-result",
        backend: "js-worker",
        verified: false,
        error: error.message || String(error),
      });
    }
  };
})();
