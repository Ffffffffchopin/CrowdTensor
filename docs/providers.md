# Training Provider Contract

Providers expose resources to the scheduler; they do not define model layers
or weaken acceptance gates.

| Provider | Capability | Current evidence boundary |
| --- | --- | --- |
| CPU | trainable stage, host memory, throughput telemetry | local and Kaggle Kernel |
| CUDA | one-GPU Miner, memory/compute telemetry, FP32/FP16 | Kaggle T4 logical nodes |
| JAX TPU | one v5e-8 resource group, mesh/sharding telemetry | retained Qwen Production RC; optional in Community short gate |

Every capability report contains only hashed device identity and bounded
telemetry. Scheduling uses free memory, throughput, utilization, health,
checkpoint freshness, transfer cost, and compile cost where applicable.

Community Kaggle workers install the project wheel and exact direct HF runtime
pins into `/kaggle/temp` (or the system temporary directory), never from the
repository workspace. Model caches and dependency trees stay outside
`/kaggle/working`, so only bounded public evidence is exported as Kernel output.
The base Torch/CUDA build remains provider-managed.

Kaggle is a validation Provider, not a permanent production backend. The
automation uses only authorized accounts, obeys concurrent-session and quota
limits, and has a 45-minute gate deadline. The original two full gates remain
immutable failures; a one-time explicit amendment permitted only a third
CPU+GPU gate while leaving every other boundary unchanged. Attempt 3 completed
100 contiguous steps in 578.93 seconds and deleted both temporary Kernels; no
further full gate is authorized. TPU acquisition is optional and bounded to two
60-minute windows; TPU unavailability cannot block forever.

No claim in this RC covers Colab, paid cloud, external object-storage SLA,
physical multi-machine independence, or uninterrupted public capacity.
