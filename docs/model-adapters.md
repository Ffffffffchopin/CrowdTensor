# Model Adapter Contract

`model_adapter_v1.0` is the trusted model-family boundary in
`crowdtensor/model_adapter.py`.

An adapter provides fail-closed config validation, contiguous partitioning,
resource estimates, explicit PEFT targets, model loading, Adapter export/reload,
and capability metadata. It does not make an unknown architecture safe by
falling back to arbitrary `AutoModel` execution.

Adapters may be built in or installed through the
`crowdtensor.model_adapters.v1` entry-point group. Plugin discovery requires:

- entry-point name equal to `adapter_id`;
- distribution name/version provenance;
- no shadowing of built-ins;
- a passing conformance report.

Disable external plugins with
`CROWDTENSOR_DISABLE_MODEL_ADAPTER_PLUGINS=1`.

## Current Adapters

| Adapter | Scope |
| --- | --- |
| `qwen2_lora_v1` | pinned Qwen2.5 config, PEFT, retained CPU/CUDA/JAX-TPU manifest |
| `smollm2_lora_v1` | pinned SmolLM2-135M Volunteer PEFT path |
| `smollm3_lora_v1` | pinned SmolLM3-3B Commons instruction PEFT path; live 3B gate pending |
| `mistral_lora_v1` | separately packaged TinyMistral plugin contract |

```bash
crowdtensor adapters list --json
crowdtensor adapters check qwen2_lora_v1 --json
```

The official example plugin is under `plugins/mistral_adapter`. Installed
plugins execute trusted Python in the user process and must be reviewed and
pinned.

Unsupported: arbitrary architecture partitioning, automatic full-parameter
volunteer training, in-flight stage migration, custom remote code, and semantic
equivalence between model families. Adding a family requires an RFC, license
review, contract tests, real-weight training evidence, standard export, and
independent reload.
