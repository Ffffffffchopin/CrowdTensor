# Model Adapter Contract

The stable interface version is `model_adapter_v1.0` in
`crowdtensor/model_adapter.py`.

An adapter must provide:

- a versioned descriptor and exact supported model types/architectures;
- config discovery and fail-closed validation;
- contiguous stage partitioning and ownership of embedding, norm, and head;
- stage loading with `trust_remote_code=false`;
- explicit LoRA targets and PEFT export;
- adapter plus optimizer checkpoint/reload;
- memory/optimizer resource estimates and capability reporting.

Adapters may be built in or installed as Python entry-point plugins in the
versioned `crowdtensor.model_adapters.v1` group. Plugin discovery is
fail-closed: the entry-point name must equal `adapter_id`, distribution name
and version metadata are required, plugins cannot shadow built-ins, and every
discovered plugin must pass the same conformance checker. Operators can disable
all external plugins with `CROWDTENSOR_DISABLE_MODEL_ADAPTER_PLUGINS=1`.

## Support Matrix

| Adapter | Pinned model | LoRA | CPU | CUDA | JAX TPU | Production scheduler |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen2_lora_v1` | `Qwen/Qwen2.5-7B` | yes | yes | yes | fixed middle stage | yes |
| `smollm2_lora_v1` | `HuggingFaceTB/SmolLM2-135M@93efa2f...` | yes | yes | yes | no | bounded two-stage live only |
| `mistral_lora_v1` (plugin) | `Locutusque/TinyMistral-248M-v2@0f57b17c...` | yes | yes | yes | no | bounded two-stage live only |

List installed adapters and run conformance checks with:

```bash
crowdtensor community adapters list --json
crowdtensor community adapters check mistral_lora_v1 --json
```

The official Mistral plugin is a separate distribution under
`plugins/mistral_adapter`. The Model Ecosystem Beta RC clean-installed the core
and plugin wheels in isolated environments, then ran the pinned real trained
Mistral weights for eight contiguous LoRA steps across one Kaggle T4x2 Kernel
and one Kaggle CPU Kernel. The CUDA worker was replaced after step 4 and
restored LoRA plus Adam state; both stages checkpointed at steps 4 and 8, the
168 Adapter tensors merged into standard PEFT format, and a separate process
reloaded finite logits. The strict report is:

`dist/model-ecosystem-beta-20260717-r1/rc/model_ecosystem_beta_rc.json`

This is `Kaggle logical multi-node` evidence. Training used deterministic
private token sequences to verify mechanics; it is not model-quality or real
dataset evidence.

Unsupported adapter IDs, model types, architecture ambiguity, empty stages,
unknown protocol majors, and unsupported scheduler requests are errors. There
is no fallback to arbitrary `AutoModel` execution.

## Non-Capabilities

The RC does not support arbitrary architecture partitioning, full-parameter
training, data parallel training, in-flight stage migration, automatic
parameter-limit exploration, custom remote code, or semantic equivalence
between model families.

Adding a family requires an RFC, contract tests, public model/license review,
real-weight stage training, standard PEFT export, independent reload, and a
public-safe live artifact.

Installed plugins execute trusted Python in Coordinator and Miner processes.
Review and pin plugin distributions before enabling them. The Mistral Beta does
not verify Mistral-7B, arbitrary Mistral checkpoints, full-parameter training,
physical multi-machine independence, or an SLA.
