#!/usr/bin/env python3
"""Compare the pinned Qwen2.5-7B base and a CrowdTensor PEFT adapter."""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from contextlib import ExitStack, nullcontext
from pathlib import Path


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
SYSTEM_PROMPT = (
    "You are a careful math solver. Show the reasoning, then end with exactly "
    "one line in the form #### <number>."
)


def _adapter_directory(value: str, stack: ExitStack) -> Path:
    source = Path(value).expanduser().resolve()
    if source.is_dir():
        destination = source
    elif source.is_file():
        destination = Path(
            stack.enter_context(tempfile.TemporaryDirectory(prefix="ct-q7b-adapter-"))
        )
        required = {"adapter_config.json", "adapter_model.safetensors"}
        with zipfile.ZipFile(source, "r") as archive:
            names = archive.namelist()
            if (
                len(names) != len(set(names))
                or not required.issubset(names)
                or any(
                    not name
                    or name.startswith(("/", "\\"))
                    or ".." in Path(name).parts
                    for name in names
                )
            ):
                raise ValueError("Adapter ZIP is not a safe standard PEFT archive")
            for name in sorted(required):
                (destination / name).write_bytes(archive.read(name))
    else:
        raise ValueError("Adapter must be a standard PEFT directory or ZIP")
    config_path = destination / "adapter_config.json"
    model_path = destination / "adapter_model.safetensors"
    if not config_path.is_file() or not model_path.is_file():
        raise ValueError("Adapter is missing standard PEFT files")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("base_model_name_or_path") != MODEL_ID
        or config.get("revision") != MODEL_REVISION
    ):
        raise ValueError("Adapter base model identity does not match this example")
    return destination


def _run(args: argparse.Namespace, adapter_dir: Path) -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
        quantization_config=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        base, adapter_dir, local_files_only=True, is_trainable=False
    )
    model.eval()
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": args.question},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(prompt, dict):
        prompt = prompt["input_ids"]
    prompt = prompt.to(model.get_input_embeddings().weight.device)

    def generate(adapter_enabled: bool) -> str:
        context = nullcontext() if adapter_enabled else model.disable_adapter()
        with context, torch.inference_mode():
            output = model.generate(
                input_ids=prompt,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            output[0, prompt.shape[1] :], skip_special_tokens=True
        )

    print("BASE\n" + generate(False))
    print("\nADAPTER\n" + generate(True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    with ExitStack() as stack:
        return _run(args, _adapter_directory(args.adapter, stack))


if __name__ == "__main__":
    raise SystemExit(main())
