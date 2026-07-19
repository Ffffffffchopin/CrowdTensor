# Compatibility Matrix

## Runtime

| Surface | Supported | Rejection policy |
| --- | --- | --- |
| Python | 3.11, 3.12 | other versions unsupported |
| Package | `0.2.0rc1` | semantic version recorded in every release manifest |
| Community protocol | `community_training_v1.0` | family/major mismatch and newer minor rejected |
| Model Adapter API | `model_adapter_v1.0` | unknown adapter fails closed |
| Model Adapter plugins | `crowdtensor.model_adapters.v1`; official `mistral_lora_v1` Beta | invalid metadata, shadowing, name mismatch, or failed conformance rejects discovery |
| Evidence API | `community_evidence_v1.0` | unknown schema cannot satisfy strict RC |
| Linux CPU | CI and local gate | required |
| CUDA | explicit Provider | optional locally, required in Kaggle live |
| Kaggle HF runtime | exact direct pins in `requirements/community-kaggle-runtime.lock` | drifted direct versions rejected by evidence gate |
| Optional TorchAO | `>=0.16`, or ignored for ordinary dense weights when an older provider copy is present | old TorchAO-backed quantized weights fail closed |
| JAX TPU | pinned Qwen stage | optional in Community short gate |
| Checkpoint storage | local, S3/MinIO, mirrored | other backends rejected |

Same-major older protocol minors are accepted only when the local runtime
explicitly advertises that newer minor. There is no silent downgrade. Newer
peer minors are rejected until their semantics are known.

## Release Levels

- **experimental:** interface can change; smoke evidence only.
- **alpha:** bounded real path works; compatibility and recovery are incomplete.
- **beta:** ordinary workflow and fault gates work for named models/providers.
- **production-rc:** fixed topology passes long soak and release checker; no SLA.
- **GA:** stable support policy, external reproducibility, operations ownership,
  and published release process. CrowdTensor is not GA.
