# Compatibility Matrix

| Surface | Current support |
| --- | --- |
| Python | 3.11 and 3.12 |
| Package | `0.3.0a1` Training Architecture v2 alpha |
| Training workspace | v2 hashed JSON contracts |
| Modes | `elastic_delta`, `stable_sharded` |
| Backends | `volunteer_peft`, `accelerate_fsdp2` |
| Model Adapter API | `model_adapter_v1.0` |
| Backend plugins | `crowdtensor.training_backends.v2` |
| Model plugins | `crowdtensor.model_adapters.v1` |
| Community data | Data Pack v1, canonical instruction JSONL |
| Transformers | `>=4.53,<6` for native SmolLM3 support |
| Local storage | content-addressed filesystem |
| Optional storage | S3-compatible resumable upload |
| CPU PEFT | real-tested |
| Commons 3B | importer/CPU path tested; controlled Kaggle logical-node CUDA elastic gate completed ([evidence](evidence/commons-3b-kaggle-live.json)) |
| CPU FSDP2 | local two-rank recovery tested |
| CUDA stable group | planned/runtime contract implemented; live gate pending |
| JAX TPU | retained capability/manifest adapter, not one-click training |

Unknown schemas, plugin IDs, protocol majors, stale generations, and
unsupported launch topologies fail closed. There is no silent protocol or
training-mode downgrade.

The root compatibility modules for the former heterogeneous manifest/scheduler
names are import forwarders only. New code must use `crowdtensor.adapters`.
Historical CLI commands are not compatibility-supported; use Git history when
reproducing old evidence.
