#!/usr/bin/env python3
"""Build one private Kaggle Kernel for the Qwen 1.5B four-GPU Alpha."""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import shutil
import sys
import textwrap
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SCHEMA = "crowdtensor_qwen15b_four_gpu_package_v1"
SOURCE_FILES = [
    "__init__.py",
    "version.py",
    "elastic_checkpoint_storage.py",
    "elastic_training_runtime.py",
    "elastic_training_client.py",
    "elastic_training_miner.py",
    "heterogeneous_jax_qwen_training.py",
    "heterogeneous_qwen_source.py",
    "heterogeneous_qwen_training.py",
    "heterogeneous_tensor_transport.py",
    "heterogeneous_training_checkpoint.py",
    "heterogeneous_training_manifest.py",
    "heterogeneous_training_miner.py",
    "heterogeneous_training_scheduler.py",
    "qwen15b_training.py",
    "qwen15b_four_gpu_runtime.py",
    "qwen15b_four_gpu_worker.py",
]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", str(value).lower()).strip("-")
    return re.sub(r"-+", "-", slug)[:63].strip("-") or "ct-qwen15b-training"


def _bundle_archive_b64(
    config: dict[str, Any],
    tokenized_payload: Path,
    *,
    include_private_inputs: bool = True,
    source_layout: Path | None = None,
) -> str:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in SOURCE_FILES:
            archive.writestr(
                f"crowdtensor/{filename}",
                (ROOT / "crowdtensor" / filename).read_bytes(),
            )
        if include_private_inputs:
            archive.writestr(
                "private/qwen_config.json",
                json.dumps(config, separators=(",", ":"), sort_keys=True).encode("utf-8"),
            )
            archive.writestr(
                "private/qwen15b_tokenized_private.json",
                tokenized_payload.read_bytes(),
            )
            if source_layout is not None:
                archive.writestr(
                    "private/qwen_source_layout.json",
                    source_layout.read_bytes(),
                )
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def render_kernel(
    *,
    role: str,
    bundle_archive_b64: str,
    private_env_b64: str,
    product_miner_mode: bool = False,
) -> str:
    source = f'''from __future__ import annotations

import base64
import gc
import hashlib
import importlib.metadata
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


WORKING = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
ROLE = "{role}"
REPORT_PATH = WORKING / "training_qwen15b_four_gpu_worker.json"
CHECKPOINT_PATH = WORKING / f"training_qwen15b_{{ROLE}}_checkpoint_bundle.zip"
ADAPTER_PATH = WORKING / "training_qwen15b_standard_peft_adapter.zip"
PRIVATE_ROOT = WORKING / f".crowdtensor-qwen15b-private-{{ROLE}}"
BUNDLE_ROOT = PRIVATE_ROOT / "bundle"
RUNTIME_ROOT = PRIVATE_ROOT / "runtime"
EXPORT_ROOT = PRIVATE_ROOT / "standard-peft-adapter"
BUNDLE_ARCHIVE_B64 = "{bundle_archive_b64}"
PRIVATE_ENV_B64 = "{private_env_b64}"
PRODUCT_MINER_MODE = {bool(product_miner_mode)!r}


def write_report(value):
    REPORT_PATH.write_text(json.dumps(value, indent=2, sort_keys=True) + "\\n", encoding="utf-8")


def public_blocker(exc):
    text = str(exc)
    if isinstance(exc, ModuleNotFoundError):
        module_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(getattr(exc, "name", "unknown")))
        return f"qwen15b_missing_module:{{module_name[:120]}}"
    if text.startswith((
        "qwen15b_stage_startup_failed:",
        "qwen15b_stage_request_failed:",
        "qwen15b_stage_shard_prepare_failed:",
    )):
        return text[:180]
    if text.startswith("elastic_"):
        return text.split(":", 1)[0][:180]
    known = (
        "qwen15b_dependency_smoke_failed",
        "qwen15b_cuda_mixed_precision_smoke_failed",
        "qwen15b_non_finite_stage_activation",
        "qwen15b_non_finite_stage_boundary_activation",
        "qwen15b_non_finite_logits",
        "qwen15b_non_finite_loss",
        "qwen15b_non_finite_activation_gradient",
        "qwen15b_non_finite_incoming_gradient",
        "qwen15b_non_finite_lora_gradient",
        "Attempting to unscale FP16 gradients",
        "qwen15b_training_peer_registration_timeout",
        "qwen15b_coordinator_http_",
        "qwen15b_training_private_payload_hash_mismatch",
        "Qwen Kernel A pipeline step timed out",
        "Qwen Kernel B pipeline step timed out",
    )
    for value in known:
        if value in text:
            return "qwen15b_worker_failed:" + re.sub(r"[^a-zA-Z0-9:_-]", "_", value)
    return f"qwen15b_worker_failed:{{type(exc).__name__}}"


def ensure_dependencies():
    torchao_before = ""
    incompatible_torchao_removed = False
    try:
        from packaging.version import Version
        torchao_version = importlib.metadata.version("torchao")
        torchao_before = str(torchao_version)
        if Version(torchao_version) < Version("0.16.0"):
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                check=True,
                timeout=300,
            )
            incompatible_torchao_removed = True
    except importlib.metadata.PackageNotFoundError:
        pass
    required = {{
        "transformers": "5.9.0",
        "peft": "0.19.1",
        "safetensors": "0.7.0",
    }}
    try:
        httpx_before = importlib.metadata.version("httpx")
    except importlib.metadata.PackageNotFoundError:
        httpx_before = ""
    installed = {{}}
    for name in required:
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = ""
    if any(installed[name] != version for name, version in required.items()) or not httpx_before:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-cache-dir",
                "transformers==5.9.0",
                "peft==0.19.1",
                "safetensors==0.7.0",
                "accelerate>=1.2,<2",
                "httpx>=0.24,<1",
            ],
            check=True,
            timeout=600,
        )
    try:
        torchao_after = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        torchao_after = ""
    try:
        httpx_after = importlib.metadata.version("httpx")
    except importlib.metadata.PackageNotFoundError:
        httpx_after = ""
    return {{
        **{{name: importlib.metadata.version(name) for name in required}},
        "httpx_before": httpx_before,
        "httpx_after": httpx_after,
        "torchao_before": torchao_before,
        "torchao_after": torchao_after,
        "incompatible_torchao_removed": incompatible_torchao_removed,
    }}


def run_dependency_smoke():
    """Exercise the exact PEFT injection path before materializing Qwen shards."""

    import torch
    from peft import LoraConfig, inject_adapter_in_model

    class TinyLoRASmoke(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 8, bias=False)
            self.v_proj = torch.nn.Linear(8, 8, bias=False)

        def forward(self, value):
            return self.v_proj(torch.tanh(self.q_proj(value)))

    started_ns = time.time_ns()
    torch.manual_seed(20260712)
    module = inject_adapter_in_model(
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
        TinyLoRASmoke(),
    )
    trainable = [
        (name, parameter)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable or any("lora_" not in name for name, _parameter in trainable):
        raise RuntimeError("qwen15b_dependency_smoke_failed:non_lora_trainable_parameters")
    value = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 16.0
    loss = module(value).float().square().mean()
    loss.backward()
    positive_gradient_count = sum(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and float(parameter.grad.detach().float().norm().item()) > 0.0
        for _name, parameter in trainable
    )
    if positive_gradient_count < 1 or not bool(torch.isfinite(loss.detach())):
        raise RuntimeError("qwen15b_dependency_smoke_failed:backward_invalid")
    ended_ns = time.time_ns()
    del module, value, loss, trainable
    gc.collect()
    return {{
        "schema": "crowdtensor_qwen15b_dependency_smoke_v1",
        "verified": True,
        "peft_import_verified": True,
        "lora_injection_verified": True,
        "forward_verified": True,
        "backward_verified": True,
        "only_lora_trainable": True,
        "positive_lora_gradient_count": int(positive_gradient_count),
        "started_ns": int(started_ns),
        "ended_ns": int(ended_ns),
        "tensor_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }}


def run_cuda_mixed_precision_smoke():
    """Verify CUDA FP32 compute, FP16 boundaries, and GradScaler before loading."""

    import torch
    from peft import LoraConfig, inject_adapter_in_model

    class TinyCUDALoRA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(16, 16, bias=False)
            self.v_proj = torch.nn.Linear(16, 16, bias=False)

        def forward(self, value):
            return self.v_proj(torch.tanh(self.q_proj(value)))

    started_ns = time.time_ns()
    torch.manual_seed(20260712)
    module = inject_adapter_in_model(
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
        TinyCUDALoRA(),
    ).to("cuda:0")
    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    for parameter in trainable:
        parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW(trainable, lr=0.001, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=128.0)
    value = torch.arange(32, device="cuda:0", dtype=torch.float32).reshape(2, 16) / 32.0
    optimizer.zero_grad(set_to_none=True)
    output = module(value)
    boundary = output.detach().to(dtype=torch.float16)
    loss = output.float().square().mean()
    if (
        output.dtype != torch.float32
        or boundary.dtype != torch.float16
        or not bool(torch.isfinite(boundary).all())
        or not bool(torch.isfinite(loss.detach()))
    ):
        raise RuntimeError("qwen15b_cuda_mixed_precision_smoke_failed:forward_invalid")
    scaler.scale(loss).backward()
    gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
    if (
        not gradients
        or any(gradient.dtype != torch.float32 for gradient in gradients)
        or any(not bool(torch.isfinite(gradient).all()) for gradient in gradients)
    ):
        raise RuntimeError("qwen15b_cuda_mixed_precision_smoke_failed:gradient_invalid")
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(0)
    ended_ns = time.time_ns()
    gradient_count = len(gradients)
    del module, optimizer, scaler, value, output, boundary, loss, gradients, trainable
    gc.collect()
    torch.cuda.empty_cache()
    return {{
        "schema": "crowdtensor_qwen15b_cuda_mixed_precision_smoke_v1",
        "verified": True,
        "cuda_live": True,
        "fp32_lora_parameters": True,
        "fp32_stable_compute": True,
        "fp16_stage_boundary": True,
        "grad_scaler_unscale_step_verified": True,
        "finite_gradient_count": int(gradient_count),
        "started_ns": int(started_ns),
        "ended_ns": int(ended_ns),
        "tensor_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }}


def archive_tree(source_root, destination, *, required_manifest_count=0):
    destination.unlink(missing_ok=True)
    files = [path for path in sorted(source_root.rglob("*")) if path.is_file()]
    if not files:
        return {{"present": False, "file_count": 0, "private_paths_public": False}}
    manifest_count = sum(path.name.endswith("_checkpoint.json") for path in files)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, str(path.relative_to(source_root)))
    digest = "sha256:" + hashlib.sha256(destination.read_bytes()).hexdigest()
    return {{
        "present": True,
        "file_name": destination.name,
        "file_hash": digest,
        "byte_count": destination.stat().st_size,
        "file_count": len(files),
        "checkpoint_manifest_count": manifest_count,
        "required_checkpoint_manifest_count": int(required_manifest_count),
        "all_required_checkpoint_manifests_present": manifest_count >= int(required_manifest_count),
        "checkpoint_values_public": False,
        "private_paths_public": False,
    }}


started = time.time()
report = {{
    "schema": "crowdtensor_qwen15b_four_gpu_worker_artifact_v1",
    "ok": False,
    "role": ROLE,
    "kaggle_kernel": bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE")),
    "blockers": [],
    "activation_values_public": False,
    "gradient_values_public": False,
    "adapter_tensor_values_public": False,
    "token_ids_public": False,
    "raw_training_text_public": False,
    "credentials_public": False,
    "coordinator_url_public": False,
    "private_paths_public": False,
    "public_artifact_safe": True,
}}
elastic_mode = False
elastic_final_segment = False
try:
    BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE_ARCHIVE_B64)), "r") as archive:
        archive.extractall(BUNDLE_ROOT)
    private_env = json.loads(base64.b64decode(PRIVATE_ENV_B64).decode("utf-8"))
    elastic_mode = bool(private_env.get("elastic_mode"))
    elastic_final_segment = bool(
        elastic_mode
        and int(private_env.get("segment_end_step", 0))
        == int(private_env.get("target_steps", 8))
    )
    report["elastic_mode"] = elastic_mode
    report["product_miner_mode"] = PRODUCT_MINER_MODE
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dependencies = ensure_dependencies()
    sys.path.insert(0, str(BUNDLE_ROOT))
    import torch
    report["dependency_smoke"] = run_dependency_smoke()
    from crowdtensor.qwen15b_four_gpu_worker import (
        run_elastic_kernel_role,
        run_kernel_role,
    )

    report["dependencies"] = {{
        **dependencies,
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda or ""),
    }}
    report["cuda_available"] = bool(torch.cuda.is_available())
    report["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if report["cuda_device_count"] < 2:
        raise RuntimeError("qwen15b_t4x2_required")
    report["cuda_mixed_precision_smoke"] = run_cuda_mixed_precision_smoke()
    config = (
        {{}}
        if PRODUCT_MINER_MODE
        else json.loads(
            (BUNDLE_ROOT / "private" / "qwen_config.json").read_text(
                encoding="utf-8"
            )
        )
    )
    report["stage_runtime_started_ns"] = time.time_ns()
    report["dependency_smoke_before_stage_runtime"] = bool(
        report["dependency_smoke"]["ended_ns"] <= report["stage_runtime_started_ns"]
        and report["cuda_mixed_precision_smoke"]["ended_ns"]
        <= report["stage_runtime_started_ns"]
    )
    common_worker = {{
        "role": ROLE,
        "coordinator_url": str(private_env["coordinator_url"]),
        "coordinator_token": str(private_env["coordinator_token"]),
        "run_id": str(private_env["run_id"]),
        "config": config,
        "tokenized_payload_path": BUNDLE_ROOT / "private" / "qwen15b_tokenized_private.json",
        "private_root": RUNTIME_ROOT,
        "export_dir": EXPORT_ROOT if ROLE == "kernel_b" else None,
        "microbatch_count": int(private_env.get("microbatch_count", 4)),
        "seed": int(private_env.get("seed", 20260712)),
        "learning_rate": float(private_env.get("learning_rate", 0.0005)),
        "lora_rank": int(private_env.get("lora_rank", 4)),
        "lora_alpha": int(private_env.get("lora_alpha", 8)),
        "wait_timeout": float(private_env.get("wait_timeout", 900.0)),
        "model_id": str(private_env.get("model_id") or ""),
        "model_revision": str(private_env.get("model_revision") or ""),
        "parameter_count": int(private_env.get("parameter_count") or 0),
        "source_layout_path": (
            BUNDLE_ROOT / "private" / "qwen_source_layout.json"
            if private_env.get("source_layout_present")
            else None
        ),
        "defer_evaluation": bool(private_env.get("defer_evaluation", False)),
    }}
    if PRODUCT_MINER_MODE:
        from argparse import Namespace
        from crowdtensor.elastic_training_miner import run_training_join

        worker = run_training_join(
            Namespace(
                training=True,
                coordinator=str(private_env["coordinator_url"]),
                invite="",
                token=str(private_env["coordinator_token"]),
                token_env="",
                miner_id=str(private_env.get("miner_id_hash") or ROLE),
                role=str(private_env.get("product_role") or "auto"),
                private_root=str(RUNTIME_ROOT),
                output_dir=str(WORKING / "elastic-training-miner-output"),
                drain_file="",
                max_steps=int(private_env.get("max_steps_per_session", 0)),
                wait_timeout=float(private_env.get("wait_timeout", 900.0)),
                http_timeout=min(
                    120.0, float(private_env.get("wait_timeout", 900.0))
                ),
                heartbeat_interval=float(
                    private_env.get("heartbeat_interval_seconds", 5.0)
                ),
                keep_private_cache=True,
                json=True,
            )
        )
    elif elastic_mode:
        worker = run_elastic_kernel_role(
            **common_worker,
            miner_id_hash=str(private_env["miner_id_hash"]),
            registration_nonce=str(private_env["registration_nonce"]),
            expected_start_step=int(private_env["expected_start_step"]),
            segment_end_step=int(private_env["segment_end_step"]),
            target_steps=int(private_env.get("target_steps", 8)),
            heartbeat_interval_seconds=float(
                private_env.get("heartbeat_interval_seconds", 5.0)
            ),
        )
    else:
        common_worker.pop("model_id", None)
        common_worker.pop("model_revision", None)
        common_worker.pop("parameter_count", None)
        common_worker.pop("source_layout_path", None)
        common_worker.pop("defer_evaluation", None)
        worker = run_kernel_role(
            **common_worker,
            steps=8,
            coordinator_restart_after_step=int(
                private_env.get("coordinator_restart_after_step", 0)
            ),
        )
    report["worker"] = worker
    report["gpu_live_verified"] = bool(
        worker.get("ok")
        and report["kaggle_kernel"]
        and report["cuda_available"]
        and report["cuda_device_count"] >= 2
    )
except BaseException as exc:
    report["blockers"].append(public_blocker(exc))
    report["error_class"] = type(exc).__name__
    report["error_code"] = public_blocker(exc)
finally:
    try:
        report["checkpoint_bundle"] = archive_tree(
            RUNTIME_ROOT,
            CHECKPOINT_PATH,
            required_manifest_count=2 if elastic_mode else 4,
        )
    except BaseException as exc:
        report["checkpoint_bundle"] = {{"present": False, "file_count": 0}}
        report["blockers"].append(public_blocker(exc))
    if ROLE == "kernel_b":
        try:
            report["adapter_bundle"] = archive_tree(EXPORT_ROOT, ADAPTER_PATH)
        except BaseException as exc:
            report["adapter_bundle"] = {{"present": False, "file_count": 0}}
            report["blockers"].append(public_blocker(exc))
    private_env = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except BaseException:
        pass
    shutil.rmtree(PRIVATE_ROOT, ignore_errors=True)
    report["cleanup"] = {{
        "private_runtime_removed": not PRIVATE_ROOT.exists(),
        "stage_shards_removed": not (RUNTIME_ROOT / "stage-shards").exists(),
    }}
    worker_ok = (report.get("worker") or {{}}).get("ok") is True
    checkpoint_ok = bool(
        (report.get("checkpoint_bundle") or {{}}).get("present")
        and (report.get("checkpoint_bundle") or {{}}).get(
            "all_required_checkpoint_manifests_present"
        )
    )
    adapter_required = bool(
        not PRODUCT_MINER_MODE
        and ROLE == "kernel_b"
        and (not elastic_mode or elastic_final_segment)
    )
    adapter_ok = bool(
        not adapter_required or (report.get("adapter_bundle") or {{}}).get("present")
    )
    report["ok"] = bool(
        worker_ok
        and report.get("gpu_live_verified")
        and checkpoint_ok
        and adapter_ok
        and report["cleanup"]["private_runtime_removed"]
    )
    if not report["ok"] and not report["blockers"]:
        report["blockers"].append("qwen15b_worker_acceptance_incomplete")
    report["elapsed_seconds"] = time.time() - started
    write_report(report)

if not report["ok"]:
    raise SystemExit(2)
'''
    marker = "started = time.time()"
    prefix, body = source.split(marker, 1)
    return (
        prefix
        + "def main():\n"
        + textwrap.indent(marker + body, "    ")
        + "\n\nif __name__ == '__main__':\n    main()\n"
    )


def build_package(
    output_dir: str | Path,
    *,
    owner: str,
    slug: str,
    role: str,
    config: dict[str, Any],
    tokenized_payload_path: str | Path,
    coordinator_url: str,
    coordinator_token: str,
    run_id: str,
    seed: int = 20260712,
    learning_rate: float = 5e-4,
    lora_rank: int = 4,
    lora_alpha: int = 8,
    wait_timeout: float = 900.0,
    coordinator_restart_after_step: int = 0,
    elastic_mode: bool = False,
    miner_id_hash: str = "",
    registration_nonce: str = "",
    expected_start_step: int = 0,
    segment_end_step: int = 8,
    target_steps: int = 8,
    microbatch_count: int = 4,
    heartbeat_interval_seconds: float = 5.0,
    product_miner_mode: bool = False,
    product_role: str = "auto",
    max_steps_per_session: int = 0,
    model_id: str | None = None,
    model_revision: str | None = None,
    parameter_count: int | None = None,
    source_layout_path: str | Path | None = None,
    defer_evaluation: bool = False,
) -> dict[str, Any]:
    from crowdtensor.qwen15b_training import (
        MODEL_ID,
        MODEL_PARAMETER_COUNT,
        MODEL_REVISION,
        sha256_file,
        stable_hash,
    )

    resolved_model_id = str(model_id or MODEL_ID)
    resolved_model_revision = str(model_revision or MODEL_REVISION)
    resolved_parameter_count = int(parameter_count or MODEL_PARAMETER_COUNT)

    if role not in {"kernel_a", "kernel_b"}:
        raise ValueError("--role must be kernel_a or kernel_b")
    if product_miner_mode and not elastic_mode:
        raise ValueError("product Miner package requires elastic mode")
    if product_role not in {"auto", "kernel_a", "kernel_b"}:
        raise ValueError("product Miner role invalid")
    if int(max_steps_per_session) < 0:
        raise ValueError("product Miner max steps invalid")
    if int(microbatch_count) < 1 or int(microbatch_count) > 16:
        raise ValueError("elastic package microbatch count invalid")
    if elastic_mode and (
        not str(miner_id_hash).startswith("sha256:")
        or not str(registration_nonce)
        or int(expected_start_step) < 0
        or int(segment_end_step) <= int(expected_start_step)
        or int(segment_end_step) > int(target_steps)
    ):
        raise ValueError("elastic package contract invalid")
    if str(config.get("model_type") or "") != "qwen2" or int(
        config.get("num_hidden_layers") or 0
    ) != 28:
        raise ValueError("package config must be pinned Qwen2.5-1.5B")
    tokenized = Path(tokenized_payload_path).resolve()
    if not tokenized.is_file():
        raise ValueError("private tokenized Qwen dataset is missing")
    private_payload = json.loads(tokenized.read_text(encoding="utf-8"))
    if private_payload.get("model_id") != resolved_model_id or private_payload.get(
        "model_revision"
    ) != resolved_model_revision:
        raise ValueError("private tokenized Qwen dataset source mismatch")
    source_layout = (
        Path(source_layout_path).resolve() if source_layout_path else None
    )
    if source_layout is not None:
        if not source_layout.is_file():
            raise ValueError("Qwen source layout is missing")
        layout = json.loads(source_layout.read_text(encoding="utf-8"))
        if (
            layout.get("model_id") != resolved_model_id
            or layout.get("model_revision") != resolved_model_revision
        ):
            raise ValueError("Qwen source layout identity mismatch")
    output = Path(output_dir).resolve()
    if output.exists():
        shutil.rmtree(output)
    package = output / "private-kernel"
    package.mkdir(parents=True, exist_ok=True)
    package.chmod(0o700)
    owner_slug = _safe_slug(owner)
    kernel_slug = _safe_slug(slug)
    kernel_ref = f"{owner_slug}/{kernel_slug}"
    private_env = base64.b64encode(
        json.dumps(
            {
                "coordinator_url": coordinator_url,
                "coordinator_token": coordinator_token,
                "run_id": run_id,
                "seed": int(seed),
                "learning_rate": float(learning_rate),
                "lora_rank": int(lora_rank),
                "lora_alpha": int(lora_alpha),
                "wait_timeout": float(wait_timeout),
                "coordinator_restart_after_step": int(coordinator_restart_after_step),
                "elastic_mode": bool(elastic_mode),
                "miner_id_hash": str(miner_id_hash),
                "registration_nonce": str(registration_nonce),
                "expected_start_step": int(expected_start_step),
                "segment_end_step": int(segment_end_step),
                "target_steps": int(target_steps),
                "microbatch_count": int(microbatch_count),
                "heartbeat_interval_seconds": float(heartbeat_interval_seconds),
                "product_miner_mode": bool(product_miner_mode),
                "product_role": str(product_role),
                "max_steps_per_session": int(max_steps_per_session),
                "model_id": resolved_model_id,
                "model_revision": resolved_model_revision,
                "parameter_count": resolved_parameter_count,
                "source_layout_present": source_layout is not None,
                "defer_evaluation": bool(defer_evaluation),
            },
            separators=(",", ":"),
        ).encode("utf-8")
    ).decode("ascii")
    kernel_path = package / "kernel.py"
    kernel_path.write_text(
        render_kernel(
            role=role,
            bundle_archive_b64=_bundle_archive_b64(
                config,
                tokenized,
                include_private_inputs=not product_miner_mode,
                source_layout=source_layout,
            ),
            private_env_b64=private_env,
            product_miner_mode=product_miner_mode,
        ),
        encoding="utf-8",
    )
    kernel_path.chmod(0o600)
    _write_json(
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
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        },
    )
    report = {
        "schema": (
            PACKAGE_SCHEMA
            if resolved_model_id == MODEL_ID
            and resolved_model_revision == MODEL_REVISION
            else "crowdtensor_qwen_four_gpu_package_v2"
        ),
        "ok": True,
        "kernel_ref": kernel_ref,
        "package_dir": str(package),
        "role": role,
        "model_id": resolved_model_id,
        "model_revision": resolved_model_revision,
        "parameter_count": resolved_parameter_count,
        "topology": "kaggle-2x-t4x2",
        "owned_stage_ids": [0, 1] if role == "kernel_a" else [2, 3],
        "steps": int(target_steps) if elastic_mode else 8,
        "microbatches_per_step": int(microbatch_count),
        "private_dataset_hash": sha256_file(tokenized),
        "config_hash": stable_hash(config),
        "worker_report_name": "training_qwen15b_four_gpu_worker.json",
        "checkpoint_bundle_name": f"training_qwen15b_{role}_checkpoint_bundle.zip",
        "adapter_bundle_name": (
            "training_qwen15b_standard_peft_adapter.zip" if role == "kernel_b" else ""
        ),
        "private_kernel": True,
        "private_inputs_embedded": not product_miner_mode,
        "private_inputs_fetched_from_authenticated_bootstrap": bool(
            product_miner_mode
        ),
        "dependency_smoke_before_stage_materialization": True,
        "dependency_smoke_schema": "crowdtensor_qwen15b_dependency_smoke_v1",
        "cuda_mixed_precision_smoke_before_stage_materialization": True,
        "coordinator_restart_after_step": int(coordinator_restart_after_step),
        "elastic_mode": bool(elastic_mode),
        "expected_start_step": int(expected_start_step),
        "segment_end_step": int(segment_end_step),
        "target_steps": int(target_steps),
        "miner_id_hash": str(miner_id_hash) if elastic_mode else "",
        "registration_nonce_public": False,
        "central_checkpoint_barrier": bool(elastic_mode),
        "product_miner_mode": bool(product_miner_mode),
        "product_role": str(product_role) if product_miner_mode else "",
        "max_steps_per_session": int(max_steps_per_session),
        "multi_file_source_layout": source_layout is not None,
        "source_layout_hash": sha256_file(source_layout) if source_layout else "",
        "evaluation_deferred_to_isolated_benchmark": bool(defer_evaluation),
        "bounded_coordinator_retry_enabled": True,
        "token_ids_public": False,
        "coordinator_url_public": False,
        "coordinator_token_public": False,
        "credentials_public": False,
        "public_artifact_safe": True,
    }
    _write_json(output / "training_qwen15b_four_gpu_package.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--role", choices=["kernel_a", "kernel_b"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenized-payload", required=True)
    parser.add_argument("--coordinator-url", required=True)
    parser.add_argument("--coordinator-token", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    report = build_package(
        args.output_dir,
        owner=args.owner,
        slug=args.slug,
        role=args.role,
        config=config,
        tokenized_payload_path=args.tokenized_payload,
        coordinator_url=args.coordinator_url,
        coordinator_token=args.coordinator_token,
        run_id=args.run_id,
    )
    if args.json:
        print(json.dumps({key: value for key, value in report.items() if key != "package_dir"}))
    else:
        print(report["package_dir"])


if __name__ == "__main__":
    main()
