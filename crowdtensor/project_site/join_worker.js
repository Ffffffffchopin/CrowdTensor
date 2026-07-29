"use strict";

const INDEX_MIX = 0x9e3779b9;
const ROUND_ADD = 0x6d2b79f5;

function mixCpu(seed, index, rounds, add) {
  let value = (seed ^ Math.imul(index, INDEX_MIX)) >>> 0;
  for (let round = 0; round < rounds; round += 1) {
    value = (value ^ (value << 13)) >>> 0;
    value = (value ^ (value >>> 17)) >>> 0;
    value = (value ^ (value << 5)) >>> 0;
    value = add(add(value, ROUND_ADD), (round + index) >>> 0) >>> 0;
  }
  return value >>> 0;
}

function canonicalBytes(values) {
  const bytes = new ArrayBuffer(values.length * 4);
  const view = new DataView(bytes);
  for (let index = 0; index < values.length; index += 1) {
    view.setUint32(index * 4, values[index], true);
  }
  return bytes;
}

async function digestValues(values) {
  const digest = await crypto.subtle.digest("SHA-256", canonicalBytes(values));
  return "sha256:" + Array.from(new Uint8Array(digest), (item) => item.toString(16).padStart(2, "0")).join("");
}

async function runWebGpu(task) {
  if (!self.navigator || !self.navigator.gpu) {
    throw new Error("webgpu_unavailable");
  }
  const adapter = await self.navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) {
    throw new Error("webgpu_adapter_unavailable");
  }
  const device = await adapter.requestDevice();
  const outputBytes = task.vector_length * 4;
  const output = device.createBuffer({
    size: outputBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    size: outputBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const params = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(params, 0, new Uint32Array([task.seed, task.vector_length, task.rounds, 0]));
  const shader = device.createShaderModule({
    code: `
      struct Params { seed: u32, length: u32, rounds: u32, padding: u32 }
      @group(0) @binding(0) var<storage, read_write> output: array<u32>;
      @group(0) @binding(1) var<uniform> params: Params;

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let index = id.x;
        if (index >= params.length) { return; }
        var value = params.seed ^ (index * 0x9e3779b9u);
        for (var round = 0u; round < params.rounds; round = round + 1u) {
          value = value ^ (value << 13u);
          value = value ^ (value >> 17u);
          value = value ^ (value << 5u);
          value = value + 0x6d2b79f5u + round + index;
        }
        output[index] = value;
      }
    `,
  });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: shader, entryPoint: "main" },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: output } },
      { binding: 1, resource: { buffer: params } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(task.vector_length / 64));
  pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, outputBytes);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const values = new Uint32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  output.destroy();
  readback.destroy();
  params.destroy();
  device.destroy();
  return values;
}

async function runWasmCpu(task) {
  const moduleBytes = new Uint8Array([
    0, 97, 115, 109, 1, 0, 0, 0, 1, 7, 1, 96, 2, 127, 127, 1, 127,
    3, 2, 1, 0, 7, 7, 1, 3, 97, 100, 100, 0, 0, 10, 9, 1, 7, 0,
    32, 0, 32, 1, 106, 11,
  ]);
  const instantiated = await WebAssembly.instantiate(moduleBytes);
  const add = instantiated.instance.exports.add;
  const values = new Uint32Array(task.vector_length);
  for (let index = 0; index < task.vector_length; index += 1) {
    values[index] = mixCpu(task.seed, index, task.rounds, add);
  }
  return values;
}

function runJavaScriptCpu(task) {
  const add = (left, right) => (left + right) >>> 0;
  const values = new Uint32Array(task.vector_length);
  for (let index = 0; index < task.vector_length; index += 1) {
    values[index] = mixCpu(task.seed, index, task.rounds, add);
  }
  return values;
}

self.onmessage = async (event) => {
  if (!event.data || event.data.type !== "run") {
    return;
  }
  const task = event.data.task;
  const started = performance.now();
  let runtime = "webgpu";
  let values;
  try {
    values = await runWebGpu(task);
  } catch (_webGpuError) {
    runtime = "wasm-cpu";
    try {
      values = await runWasmCpu(task);
    } catch (_wasmError) {
      runtime = "cpu-js";
      values = runJavaScriptCpu(task);
    }
  }
  const outputSha256 = await digestValues(values);
  self.postMessage({
    type: "complete",
    runtime,
    output_sha256: outputSha256,
    duration_ms: Math.max(1, Math.round(performance.now() - started)),
  });
};
