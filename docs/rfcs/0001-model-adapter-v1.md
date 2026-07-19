# RFC 0001: Model Adapter v1

- Status: Accepted for Community RC
- Compatibility: additive to existing Qwen runtime

## Decision

Introduce `model_adapter_v1.0` as the only model-family boundary for new
Community training work. It owns config validation, stage partition/loading,
LoRA targets, estimates, checkpoint/export, reload, and capability reporting.

Qwen2 migrates behind the adapter without changing its achieved production
manifest. SmolLM2 is the second family and remains a bounded two-stage path.
The additive plugin extension uses the versioned
`crowdtensor.model_adapters.v1` Python entry-point group. Plugins must expose a
conformant `ModelAdapter`, cannot shadow built-ins, and carry public
distribution provenance. `mistral_lora_v1` is the first separately packaged
implementation and is bounded to its pinned 248M CPU/CUDA live proof.

## Alternatives

Direct `AutoModel` introspection was rejected because arbitrary architectures
cannot be safely partitioned or checkpointed by assumption. A per-script
adapter was rejected because it provides no compatibility or refusal surface.

## Migration And Rollback

Existing Qwen functions remain callable. Production manifest construction now
resolves `qwen2_lora_v1`; reverting that one delegation restores the previous
call path without changing persisted manifests.

External plugin discovery can be disabled without affecting built-ins by
setting `CROWDTENSOR_DISABLE_MODEL_ADAPTER_PLUGINS=1`.

## Non-Goals

No arbitrary architecture partition, full-parameter training, data parallel
runtime, in-flight migration, or parameter-limit search is introduced.
