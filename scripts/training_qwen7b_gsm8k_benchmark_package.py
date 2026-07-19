#!/usr/bin/env python3
"""Build a private Kaggle T4x2 package for isolated 7B GSM8K evaluation."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import textwrap
from pathlib import Path
from typing import Any

from crowdtensor.qwen15b_training import sha256_file, stable_hash
from crowdtensor.qwen7b_gsm8k_showcase import MODEL_ID, MODEL_REVISION


SCHEMA = "crowdtensor_qwen7b_gsm8k_benchmark_package_v1"
WORKER_REPORT = "training_qwen7b_gsm8k_benchmark_worker.json"


def _safe_slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9-]+", "-", str(value).lower()).strip("-")
    return re.sub(r"-+", "-", result)[:63].strip("-") or "ct-qwen7b-benchmark"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_kernel(
    *,
    mode: str,
    max_new_tokens: int,
    batch_size: int,
    expected_input_hashes: dict[str, str] | None = None,
) -> str:
    input_hashes = dict(expected_input_hashes or {})
    source = f'''from __future__ import annotations

import gc
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path


MODEL_ID = {MODEL_ID!r}
MODEL_REVISION = {MODEL_REVISION!r}
MODE = {mode!r}
MAX_NEW_TOKENS = {int(max_new_tokens)}
BATCH_SIZE = {int(batch_size)}
EXPECTED_INPUT_HASHES = {input_hashes!r}
INPUT_MATCH_COUNTS = {{}}
WORKING = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
REPORT_PATH = WORKING / {WORKER_REPORT!r}
PRIVATE_ROOT = WORKING / ".crowdtensor-qwen7b-benchmark-private"
ANSWER_PATTERN = re.compile(r"####\\s*([-+]?[$]?[0-9][0-9,]*(?:\\.[0-9]+)?)")


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def normalize(value):
    text = str(value or "").strip().replace("$", "").replace(",", "")
    try:
        number = float(text)
    except ValueError:
        return ""
    if number.is_integer():
        return str(int(number))
    return (f"{{number:.12f}}").rstrip("0").rstrip(".")


def extract(value, require_marker=False):
    text = str(value or "")
    matches = ANSWER_PATTERN.findall(text)
    if matches:
        return normalize(matches[-1])
    if require_marker:
        return ""
    fallback = re.findall(r"[-+]?[$]?[0-9][0-9,]*(?:\\.[0-9]+)?", text)
    return normalize(fallback[-1]) if fallback else ""


def locate(name):
    matches = sorted(Path("/kaggle/input").rglob(name))
    expected = EXPECTED_INPUT_HASHES.get(name, "")
    matching = [path for path in matches if expected and file_hash(path) == expected]
    INPUT_MATCH_COUNTS[name] = {{
        "candidate_count": len(matches),
        "hash_match_count": len(matching),
    }}
    if not matching:
        raise RuntimeError(
            "qwen7b_benchmark_private_input_missing_or_hash_mismatch_"
            + name
            + "_"
            + str(len(matches))
        )
    return matching[0]


def ensure_dependencies():
    requested = [
        "transformers==5.9.0",
        "peft==0.19.1",
        "accelerate==1.13.0",
        "safetensors==0.7.0",
        "bitsandbytes==0.49.2",
    ]
    step = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--quiet", *requested],
        text=True,
        capture_output=True,
        check=False,
        timeout=900,
    )
    if step.returncode != 0:
        raise RuntimeError("qwen7b_benchmark_dependency_install_failed")
    return {{"requested_hash": digest(requested), "installed": True}}


def padded_batch(torch, examples, start):
    values = examples[start : start + BATCH_SIZE]
    width = max(len(item["prompt_input_ids"]) for item in values)
    pad = 151643
    ids = []
    masks = []
    for item in values:
        row = [int(value) for value in item["prompt_input_ids"]]
        missing = width - len(row)
        ids.append([pad] * missing + row)
        masks.append([0] * missing + [1] * len(row))
    return (
        values,
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(masks, dtype=torch.long),
    )


def generation_pass(torch, tokenizer, model, examples, *, adapter_enabled):
    records = []
    generated_counts = []
    context = (
        nullcontext()
        if adapter_enabled or not hasattr(model, "disable_adapter")
        else model.disable_adapter()
    )
    with context, torch.inference_mode():
        for start in range(0, len(examples), BATCH_SIZE):
            values, ids, masks = padded_batch(torch, examples, start)
            first_device = model.get_input_embeddings().weight.device
            ids = ids.to(first_device)
            masks = masks.to(first_device)
            outputs = model.generate(
                input_ids=ids,
                attention_mask=masks,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True,
                pad_token_id=int(tokenizer.eos_token_id),
                eos_token_id=int(tokenizer.eos_token_id),
            )
            generated = outputs[:, ids.shape[1] :].detach().cpu()
            for offset, item in enumerate(values):
                tokens = generated[offset].tolist()
                if int(tokenizer.eos_token_id) in tokens:
                    tokens = tokens[: tokens.index(int(tokenizer.eos_token_id)) + 1]
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                parsed = extract(text)
                strict = extract(text, require_marker=True)
                gold = str(item["gold_answer"])
                generated_counts.append(len(tokens))
                records.append(
                    {{
                        "example_index": int(item["example_index"]),
                        "prompt_hash": digest(item["prompt_input_ids"]),
                        "gold_hash": digest({{"gold": gold}}),
                        "generated_text_hash": digest({{"text": text}}),
                        "parsed_answer_hash": digest({{"answer": parsed}}),
                        "answer_valid": bool(parsed),
                        "normalized_exact_match": parsed == gold,
                        "strict_marker_present": bool(strict),
                        "strict_exact_match": strict == gold,
                    }}
                )
    return {{
        "example_count": len(records),
        "normalized_exact_match_count": sum(item["normalized_exact_match"] for item in records),
        "normalized_exact_match": sum(item["normalized_exact_match"] for item in records) / len(records),
        "strict_exact_match_count": sum(item["strict_exact_match"] for item in records),
        "strict_exact_match": sum(item["strict_exact_match"] for item in records) / len(records),
        "valid_answer_count": sum(item["answer_valid"] for item in records),
        "valid_answer_rate": sum(item["answer_valid"] for item in records) / len(records),
        "strict_marker_count": sum(item["strict_marker_present"] for item in records),
        "strict_marker_rate": sum(item["strict_marker_present"] for item in records) / len(records),
        "generated_token_count": sum(generated_counts),
        "generated_token_count_min": min(generated_counts),
        "generated_token_count_max": max(generated_counts),
        "records": records,
        "records_hash": digest(records),
        "generated_text_public": False,
        "parsed_answers_public": False,
        "gold_answers_public": False,
        "token_ids_public": False,
    }}


def loss_pass(torch, model, rows, *, adapter_enabled):
    losses = []
    context = (
        nullcontext()
        if adapter_enabled or not hasattr(model, "disable_adapter")
        else model.disable_adapter()
    )
    with context, torch.inference_mode():
        for row in rows:
            device = model.get_input_embeddings().weight.device
            inputs = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            labels = torch.tensor([row["labels"]], dtype=torch.long, device=device)
            value = model(input_ids=inputs, labels=labels, use_cache=False)
            losses.append(float(value.loss.detach().float().item()))
    mean = sum(losses) / len(losses)
    return {{
        "sequence_count": len(losses),
        "mean_loss": mean,
        "perplexity": math.exp(min(20.0, mean)),
        "loss_trace_hash": digest(losses),
        "loss_values_public": False,
    }}


report = {{
    "schema": "crowdtensor_qwen7b_gsm8k_benchmark_worker_v1",
    "ok": False,
    "mode": MODE,
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "blockers": [],
    "started_at_epoch": time.time(),
    "kaggle_kernel": Path("/kaggle/working").is_dir(),
    "raw_text_public": False,
    "token_ids_public": False,
    "generated_text_public": False,
    "gold_answers_public": False,
    "adapter_tensor_values_public": False,
    "credentials_public": False,
    "private_paths_public": False,
    "public_artifact_safe": True,
}}
model = None
try:
    dependencies = ensure_dependencies()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("qwen7b_benchmark_t4x2_required")
    payload = json.loads(locate("qwen7b_gsm8k_benchmark_private.json").read_text())
    validation = json.loads(locate("qwen7b_gsm8k_validation_private.json").read_text())
    examples = list(payload.get("examples") or [])
    rows = list(validation.get("validation") or [])
    if len(examples) != 128 or len(rows) < 8:
        raise RuntimeError("qwen7b_benchmark_private_budget_invalid")
    adapter_dir = PRIVATE_ROOT / "adapter"
    if MODE in {{"adapter", "both"}}:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(locate("adapter_config.json"), adapter_dir / "adapter_config.json")
        shutil.copyfile(
            locate("adapter_model.safetensors"),
            adapter_dir / "adapter_model.safetensors",
        )
        adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text())
        if (
            adapter_config.get("base_model_name_or_path") != MODEL_ID
            or adapter_config.get("revision") != MODEL_REVISION
        ):
            raise RuntimeError("qwen7b_benchmark_adapter_identity_invalid")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
        quantization_config=quantization,
        device_map="auto",
        max_memory={{0: "14GiB", 1: "14GiB"}},
        low_cpu_mem_usage=True,
    )
    standard_peft_reload = False
    if MODE in {{"adapter", "both"}}:
        model = PeftModel.from_pretrained(
            model, adapter_dir, local_files_only=True, is_trainable=False
        )
        standard_peft_reload = any("lora_" in name for name, _value in model.named_parameters())
    model.eval()
    passes = {{}}
    if MODE in {{"base", "both"}}:
        passes["base"] = generation_pass(
            torch, tokenizer, model, examples, adapter_enabled=False
        )
        passes["base_validation"] = loss_pass(
            torch, model, rows, adapter_enabled=False
        )
    if MODE in {{"adapter", "both"}}:
        passes["adapter"] = generation_pass(
            torch, tokenizer, model, examples, adapter_enabled=True
        )
        passes["adapter_validation"] = loss_pass(
            torch, model, rows, adapter_enabled=True
        )
    report.update(
        {{
            "dependencies": dependencies,
            "cuda_device_count": int(torch.cuda.device_count()),
            "cuda_device_name_hashes": [
                digest({{"name": torch.cuda.get_device_name(index)}})
                for index in range(torch.cuda.device_count())
            ],
            "quantization": {{
                "load_in_4bit": True,
                "quant_type": "nf4",
                "double_quant": True,
                "compute_dtype": "float16",
            }},
            "standard_peft_reload_verified": standard_peft_reload,
            "adapter_file_hash": (
                file_hash(adapter_dir / "adapter_model.safetensors")
                if standard_peft_reload
                else ""
            ),
            "benchmark_example_count": len(examples),
            "benchmark_prompt_hash": digest([item["prompt_input_ids"] for item in examples]),
            "benchmark_gold_hash": digest([item["gold_answer"] for item in examples]),
            "passes": passes,
            "input_hashes_verified": bool(
                set(INPUT_MATCH_COUNTS) == set(EXPECTED_INPUT_HASHES)
                and all(
                    value.get("hash_match_count", 0) >= 1
                    for value in INPUT_MATCH_COUNTS.values()
                )
            ),
        }}
    )
    report["ok"] = bool(
        len(examples) == 128
        and (MODE == "base" or standard_peft_reload)
        and all(value.get("example_count") == 128 for key, value in passes.items() if key in {{"base", "adapter"}})
    )
except BaseException as exc:
    report["blockers"].append(
        "qwen7b_benchmark_" + type(exc).__name__.lower() + "_" + hashlib.sha256(str(exc).encode()).hexdigest()[:12]
    )
    report["error_class"] = type(exc).__name__
finally:
    report["input_match_counts"] = dict(INPUT_MATCH_COUNTS)
    report.setdefault("input_hashes_verified", False)
    report["finished_at_epoch"] = time.time()
    report["elapsed_seconds"] = report["finished_at_epoch"] - report["started_at_epoch"]
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except BaseException:
        pass
    shutil.rmtree(PRIVATE_ROOT, ignore_errors=True)
    report["private_runtime_removed"] = not PRIVATE_ROOT.exists()
    report["content_hash"] = digest(report)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n")

if not report["ok"]:
    raise SystemExit(2)
'''
    return textwrap.dedent(source)


def build_package(
    output_dir: str | Path,
    *,
    owner: str,
    slug: str,
    dataset_ref: str,
    mode: str,
    max_new_tokens: int = 256,
    batch_size: int = 8,
    expected_input_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    if mode not in {"base", "adapter", "both"}:
        raise ValueError("Qwen7B benchmark mode invalid")
    if not 64 <= int(max_new_tokens) <= 512 or not 1 <= int(batch_size) <= 16:
        raise ValueError("Qwen7B benchmark generation budget invalid")
    output = Path(output_dir).resolve()
    input_hashes = dict(expected_input_hashes or {})
    required_inputs = {
        "qwen7b_gsm8k_benchmark_private.json",
        "qwen7b_gsm8k_validation_private.json",
    }
    if mode in {"adapter", "both"}:
        required_inputs.update({"adapter_config.json", "adapter_model.safetensors"})
    input_hash_binding_ready = bool(
        set(input_hashes) == required_inputs
        and all(str(value).startswith("sha256:") for value in input_hashes.values())
    )
    if not input_hash_binding_ready:
        raise ValueError("Qwen7B benchmark expected input hashes invalid")
    if output.exists():
        shutil.rmtree(output)
    package = output / "private-kernel"
    package.mkdir(parents=True)
    package.chmod(0o700)
    owner_slug = _safe_slug(owner)
    kernel_slug = _safe_slug(slug)
    kernel_ref = f"{owner_slug}/{kernel_slug}"
    kernel = package / "kernel.py"
    kernel.write_text(
        render_kernel(
            mode=mode,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            expected_input_hashes=input_hashes,
        ),
        encoding="utf-8",
    )
    kernel.chmod(0o600)
    _write(
        package / "kernel-metadata.json",
        {
            "id": kernel_ref,
            "title": kernel_slug.replace("-", " ").title(),
            "code_file": "kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_tpu": "false",
            "enable_internet": "true",
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [str(dataset_ref)],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        },
    )
    report = {
        "schema": SCHEMA,
        "ok": True,
        "kernel_ref": kernel_ref,
        "package_dir": str(package),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "mode": mode,
        "max_new_tokens": int(max_new_tokens),
        "batch_size": int(batch_size),
        "dataset_ref_hash": stable_hash({"dataset_ref": dataset_ref}),
        "worker_report_name": WORKER_REPORT,
        "private_kernel": True,
        "dataset_private_expected": True,
        "input_hash_binding_ready": input_hash_binding_ready,
        "expected_input_hashes_hash": stable_hash(input_hashes),
        "raw_text_public": False,
        "token_ids_public": False,
        "generated_text_public": False,
        "credentials_public": False,
        "public_artifact_safe": True,
    }
    report["kernel_file_hash"] = sha256_file(kernel)
    _write(output / "training_qwen7b_gsm8k_benchmark_package.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--dataset-ref", required=True)
    parser.add_argument("--mode", choices=["base", "adapter", "both"], required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--expected-input-hashes-json", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    expected_input_hashes = json.loads(
        Path(args.expected_input_hashes_json).read_text(encoding="utf-8")
    )
    if not isinstance(expected_input_hashes, dict):
        parser.error("--expected-input-hashes-json must contain an object")
    report = build_package(
        args.output_dir,
        owner=args.owner,
        slug=args.slug,
        dataset_ref=args.dataset_ref,
        mode=args.mode,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        expected_input_hashes={
            str(key): str(value) for key, value in expected_input_hashes.items()
        },
    )
    if args.json:
        print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
