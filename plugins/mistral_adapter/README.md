# CrowdTensor Mistral Adapter

This separately installable package registers `mistral_lora_v1` through the
`crowdtensor.model_adapters.v1` entry-point group. It does not patch or import
private CrowdTensor registry state.

The pinned Beta model is
`Locutusque/TinyMistral-248M-v2@0f57b17cb317bb322c7c1466b669c681f80c058f`.
It is an Apache-2.0 `MistralForCausalLM` checkpoint trained on OpenWebText and
TM-DATA. The small checkpoint is used to verify real CPU/CUDA heterogeneous
LoRA mechanics; it is not a claim that Mistral-7B has completed the same live
gate.

Installed Python plugins execute trusted code in the Coordinator and Miner
processes. Operators must review and pin plugin distributions. Set
`CROWDTENSOR_DISABLE_MODEL_ADAPTER_PLUGINS=1` to disable all entry-point plugin
loading.
