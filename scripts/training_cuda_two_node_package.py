#!/usr/bin/env python3
"""Build one private Kaggle worker package for the two-node CUDA training gate."""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import sys
import textwrap
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SCHEMA = "crowdtensor_cuda_two_node_package_v1"
SOURCE_FILES = [
    "__init__.py",
    "cuda_training_worker.py",
    "hf_lora_training.py",
    "named_tensor_optimizer.py",
    "pipeline_lora_training.py",
    "training_contract.py",
]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_slug(value: str) -> str:
    import re

    slug = re.sub(r"[^a-z0-9-]+", "-", str(value).lower()).strip("-")
    return re.sub(r"-+", "-", slug)[:63].strip("-") or "ct-cuda-node"


def _bundle_archive_b64(fixture_dir: Path) -> str:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in SOURCE_FILES:
            archive.writestr(
                f"crowdtensor/{filename}",
                (ROOT / "crowdtensor" / filename).read_bytes(),
            )
        for path in sorted(fixture_dir.rglob("*")):
            if path.is_file():
                archive.write(path, f"fixture/{path.relative_to(fixture_dir)}")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def render_kernel(
    *,
    role: str,
    bundle_archive_b64: str,
    private_env_b64: str,
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
REPORT_PATH = WORKING / "training_cuda_two_node_worker.json"
CHECKPOINT_BUNDLE_PATH = WORKING / "training_cuda_two_node_{role}_checkpoint_bundle.zip"
PRIVATE_ROOT = WORKING / ".crowdtensor-cuda-two-node-private-{role}"
BUNDLE_ROOT = PRIVATE_ROOT / "bundle"
WORKER_ROLE = "{role}"
BUNDLE_ARCHIVE_B64 = "{bundle_archive_b64}"
PRIVATE_ENV_B64 = "{private_env_b64}"
BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)
if not (BUNDLE_ROOT / "crowdtensor" / "cuda_training_worker.py").is_file():
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE_ARCHIVE_B64)), "r") as archive:
        archive.extractall(BUNDLE_ROOT)
sys.path.insert(0, str(BUNDLE_ROOT))


def write_report(value):
    REPORT_PATH.write_text(json.dumps(value, indent=2, sort_keys=True) + "\\n", encoding="utf-8")


def public_blocker(exc):
    text = re.sub(r"https?://[^\\s]+", "<private-url>", str(exc)[:180])
    text = re.sub(r"/(?:root|tmp|home|kaggle)/[^\\s]+", "<private-path>", text)
    text = re.sub(r"(?i)(token|authorization|cookie)[=:][^\\s]+", r"\\1=<redacted>", text)
    return f"{{type(exc).__name__}}:{{text}}" if text else type(exc).__name__


def checkpoint_bundle():
    if CHECKPOINT_BUNDLE_PATH.exists():
        CHECKPOINT_BUNDLE_PATH.unlink()
    roots = {{
        "pipeline": PRIVATE_ROOT / "pipeline" / "checkpoint",
        "miner": PRIVATE_ROOT / "miner" / "checkpoint",
    }}
    embedded_roots = {{
        "single_baseline": PRIVATE_ROOT / "embedded-single-kernel" / "baseline" / "checkpoint",
        "single_resumed": PRIVATE_ROOT / "embedded-single-kernel" / "resumed" / "checkpoint",
    }}
    if WORKER_ROLE == "stage0":
        roots.update(embedded_roots)
    file_count = 0
    with zipfile.ZipFile(CHECKPOINT_BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for label, root in roots.items():
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, f"{{label}}/{{path.relative_to(root)}}")
                    file_count += 1
    if file_count == 0:
        CHECKPOINT_BUNDLE_PATH.unlink(missing_ok=True)
        return {{
            "present": False,
            "file_count": 0,
            "checkpoint_values_public": False,
            "private_paths_public": False,
        }}
    digest = hashlib.sha256(CHECKPOINT_BUNDLE_PATH.read_bytes()).hexdigest()
    return {{
        "present": True,
        "file_name": CHECKPOINT_BUNDLE_PATH.name,
        "file_hash": "sha256:" + digest,
        "byte_count": CHECKPOINT_BUNDLE_PATH.stat().st_size,
        "file_count": file_count,
        "contains_pipeline_and_miner_checkpoints": all(
            roots[label].is_dir() for label in ("pipeline", "miner")
        ),
        "contains_baseline_and_resumed_checkpoints": bool(
            WORKER_ROLE == "stage0"
            and all(root.is_dir() for root in embedded_roots.values())
        ),
        "checkpoint_values_public": False,
        "private_paths_public": False,
    }}


def ensure_dependencies():
    try:
        from packaging.version import Version
        torchao_version = importlib.metadata.version("torchao")
        if Version(torchao_version) < Version("0.16.0"):
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                check=True,
                timeout=300,
            )
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        import peft
        import safetensors
    except Exception:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "peft>=0.19,<1", "safetensors>=0.4,<1"],
            check=True,
            timeout=600,
        )
        import peft
        import safetensors
    return {{"peft": str(peft.__version__), "safetensors": str(safetensors.__version__)}}


def run_embedded_single_kernel_gate():
    from crowdtensor.pipeline_lora_training import compare_pipeline_runs, run_two_cuda_process_pipeline

    root = PRIVATE_ROOT / "embedded-single-kernel"
    baseline = run_two_cuda_process_pipeline(
        root / "baseline",
        total_steps=4,
        seed=20260710,
    )
    resumed = run_two_cuda_process_pipeline(
        root / "resumed",
        total_steps=4,
        interrupt_stage1_after_step=2,
        seed=20260710,
    )
    comparison = compare_pipeline_runs(baseline, resumed, atol=0.005, rtol=0.005)
    baseline_public = json.loads(Path(baseline["report_path"]).read_text(encoding="utf-8"))
    resumed_public = json.loads(Path(resumed["report_path"]).read_text(encoding="utf-8"))
    gate = {{
        "schema": "crowdtensor_cuda_single_kernel_gate_v1",
        "ok": False,
        "single_kernel_t4x2_verified": False,
        "kaggle_kernel": bool(
            os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE")
        ),
        "gpu_live_verified": bool(
            baseline_public.get("gpu_live_verified")
            and resumed_public.get("gpu_live_verified")
        ),
        "cuda_device_count": 2,
        "baseline": baseline_public,
        "resumed": resumed_public,
        "resume_equivalence": comparison,
        "two_distinct_processes": bool(baseline_public.get("distinct_stage_pids")),
        "two_distinct_cuda_devices": bool(baseline_public.get("distinct_cuda_devices")),
        "real_activation_transport": bool(baseline_public.get("real_activation_transport")),
        "real_backward_gradient_transport": bool(
            baseline_public.get("real_backward_gradient_transport")
        ),
        "real_cuda_backward": bool(baseline_public.get("real_cuda_backward")),
        "no_stage_loaded_full_model": bool(baseline_public.get("no_stage_loaded_full_model")),
        "base_weights_frozen": bool(
            baseline_public.get("base_weights_frozen")
            and resumed_public.get("base_weights_frozen")
        ),
        "positive_lora_gradient_norms": bool(
            baseline_public.get("positive_lora_gradient_norms")
            and resumed_public.get("positive_lora_gradient_norms")
        ),
        "positive_cuda_memory": bool(
            baseline_public.get("positive_cuda_memory")
            and resumed_public.get("positive_cuda_memory")
        ),
        "loss_reduced": bool(
            baseline_public.get("loss_reduced") and resumed_public.get("loss_reduced")
        ),
        "checkpoint_resume_verified": bool(comparison.get("checkpoint_resume_verified")),
        "controlled_stage_restart": bool(
            (resumed_public.get("interruption") or {{}}).get("worker_restarted")
        ),
        "execution_order": "before_cross_node_stage0",
        "coallocated_with_two_node_attempt": True,
        "source_role": "stage0",
        "activation_values_public": False,
        "gradient_values_public": False,
        "checkpoint_values_public": False,
        "raw_training_text_public": False,
        "credentials_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }}
    gate["single_kernel_t4x2_verified"] = bool(
        gate["kaggle_kernel"]
        and gate["gpu_live_verified"]
        and gate["cuda_device_count"] >= 2
        and gate["two_distinct_processes"]
        and gate["two_distinct_cuda_devices"]
        and gate["real_activation_transport"]
        and gate["real_backward_gradient_transport"]
        and gate["real_cuda_backward"]
        and gate["no_stage_loaded_full_model"]
        and gate["base_weights_frozen"]
        and gate["positive_lora_gradient_norms"]
        and gate["positive_cuda_memory"]
        and gate["loss_reduced"]
        and gate["checkpoint_resume_verified"]
        and gate["controlled_stage_restart"]
    )
    gate["ok"] = gate["single_kernel_t4x2_verified"]
    return gate


started = time.time()
role = WORKER_ROLE
report = {{
    "schema": "crowdtensor_cuda_two_node_worker_v1",
    "ok": False,
    "role": role,
    "kaggle_kernel": bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE")),
    "gpu_live_verified": False,
    "blockers": [],
    "activation_values_public": False,
    "gradient_values_public": False,
    "adapter_tensor_values_public": False,
    "evaluation_logits_public": False,
    "raw_training_text_public": False,
    "credentials_public": False,
    "coordinator_url_public": False,
    "private_paths_public": False,
    "public_artifact_safe": True,
}}
try:
    private_env = json.loads(base64.b64decode(PRIVATE_ENV_B64).decode("utf-8"))
    coordinator_url = str(private_env["coordinator_url"])
    coordinator_token = str(private_env["coordinator_token"])
    run_id = str(private_env["run_id"])
    dependencies = ensure_dependencies()
    import torch
    from crowdtensor.cuda_training_worker import (
        evaluate_global_adapter_on_cuda,
        run_cross_node_stage,
        run_remote_lora_miner,
        wait_global_adapter,
    )

    report["dependencies"] = {{
        **dependencies,
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda or ""),
    }}
    report["cuda_available"] = bool(torch.cuda.is_available())
    report["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if report["cuda_device_count"] < 1:
        raise RuntimeError("two_node_worker_requires_cuda")
    if role == "stage0":
        if report["cuda_device_count"] < 2:
            raise RuntimeError("embedded_single_kernel_gate_requires_two_cuda_devices")
        report["embedded_single_kernel_gate"] = run_embedded_single_kernel_gate()
        report["embedded_single_kernel_gate_verified"] = bool(
            report["embedded_single_kernel_gate"].get("single_kernel_t4x2_verified")
        )
        if not report["embedded_single_kernel_gate_verified"]:
            raise RuntimeError("embedded_single_kernel_t4x2_gate_incomplete")
        gc.collect()
        torch.cuda.empty_cache()
    pipeline = run_cross_node_stage(
        role=role,
        coordinator_url=coordinator_url,
        token=coordinator_token,
        run_id=run_id,
        output_dir=PRIVATE_ROOT / "pipeline",
        total_steps=4,
        wait_timeout=900,
    )
    miner = run_remote_lora_miner(
        role=role,
        coordinator_url=coordinator_url,
        token=coordinator_token,
        fixture_dir=BUNDLE_ROOT / "fixture",
        output_dir=PRIVATE_ROOT / "miner",
    )
    global_adapter = wait_global_adapter(
        coordinator_url=coordinator_url,
        token=coordinator_token,
        run_id=run_id,
        timeout=900,
    )
    evaluation = evaluate_global_adapter_on_cuda(
        role=role,
        coordinator_url=coordinator_url,
        token=coordinator_token,
        run_id=run_id,
        fixture_dir=BUNDLE_ROOT / "fixture",
        private_output_dir=PRIVATE_ROOT / "evaluation",
        global_adapter=global_adapter,
    )
    report.update({{
        "pipeline": pipeline,
        "miner": miner,
        "evaluation": evaluation,
        "global_adapter": {{
            "adapter_hash": global_adapter["adapter_hash"],
            "adapter_config_hash": global_adapter["adapter_config_hash"],
            "adapter_version": int(global_adapter["adapter_version"]),
            "outer_step": int(global_adapter["outer_step"]),
            "tensor_values_public": False,
        }},
    }})
    report["gpu_live_verified"] = bool(
        pipeline.get("real_cuda_forward")
        and pipeline.get("real_cuda_backward")
        and (miner.get("runtime") or {{}}).get("gpu_live_verified")
    )
    report["ok"] = bool(
        report["kaggle_kernel"]
        and report["gpu_live_verified"]
        and int(pipeline.get("steps_completed") or 0) >= 4
        and pipeline.get("positive_lora_gradient_norms") is True
        and pipeline.get("base_weights_frozen") is True
        and pipeline.get("no_full_model_loaded") is True
        and miner.get("coordinator_accepted") is True
        and miner.get("base_weights_frozen") is True
        and miner.get("only_lora_trainable") is True
        and miner.get("real_backward") is True
        and miner.get("loss_reduced") is True
        and int(miner.get("adapter_delta_tensor_count") or 0) > 0
        and int(global_adapter.get("adapter_version") or 0) == 1
        and int(global_adapter.get("outer_step") or 0) == 1
        and evaluation.get("standard_peft_cuda_load") is True
        and evaluation.get("adapter_changes_logits") is True
        and evaluation.get("validation_loss_reduced") is True
        and (
            role != "stage0"
            or report.get("embedded_single_kernel_gate_verified") is True
        )
    )
    if not report["ok"]:
        report["blockers"].append("two_node_worker_acceptance_incomplete")
except BaseException as exc:
    report["blockers"].append(public_blocker(exc))
    report["error_class"] = type(exc).__name__
finally:
    try:
        report["checkpoint_bundle"] = checkpoint_bundle()
    except BaseException as exc:
        report["checkpoint_bundle"] = {{
            "present": False,
            "file_count": 0,
            "checkpoint_values_public": False,
            "private_paths_public": False,
        }}
        report["blockers"].append(public_blocker(exc))
    embedded_gate = report.get("embedded_single_kernel_gate") or {{}}
    if embedded_gate:
        embedded_gate["checkpoint_bundle"] = report.get("checkpoint_bundle") or {{}}
        embedded_gate["single_kernel_t4x2_verified"] = bool(
            embedded_gate.get("single_kernel_t4x2_verified")
            and (report.get("checkpoint_bundle") or {{}}).get("present")
            and (report.get("checkpoint_bundle") or {{}}).get(
                "contains_baseline_and_resumed_checkpoints"
            )
        )
        embedded_gate["ok"] = embedded_gate["single_kernel_t4x2_verified"]
        report["embedded_single_kernel_gate_verified"] = embedded_gate["ok"]
    report["ok"] = bool(
        report.get("ok")
        and (report.get("checkpoint_bundle") or {{}}).get("present")
        and (report.get("checkpoint_bundle") or {{}}).get(
            "contains_pipeline_and_miner_checkpoints"
        )
        and (
            role != "stage0"
            or report.get("embedded_single_kernel_gate_verified") is True
        )
    )
    private_env = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    shutil.rmtree(PRIVATE_ROOT, ignore_errors=True)
    report["cleanup"] = {{"private_runtime_removed": not PRIVATE_ROOT.exists()}}
    report["elapsed_seconds"] = time.time() - started
    report["ok"] = bool(report.get("ok") and report["cleanup"]["private_runtime_removed"])
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
    fixture_dir: str | Path,
    coordinator_url: str,
    coordinator_token: str,
    run_id: str,
) -> dict[str, Any]:
    if role not in {"stage0", "stage1"}:
        raise ValueError("--role must be stage0 or stage1")
    fixture = Path(fixture_dir).resolve()
    required = [
        fixture / "base_model",
        fixture / "initial_adapter" / "adapter_model.safetensors",
        fixture / "initial_adapter" / "adapter_config.json",
        fixture / "private_dataset.jsonl",
    ]
    if not all(path.exists() for path in required):
        raise ValueError("fixture directory is incomplete")
    output = Path(output_dir).resolve()
    if output.exists():
        shutil.rmtree(output)
    package = output / "private-kernel"
    package.mkdir(parents=True, exist_ok=True)
    safe_owner = _safe_slug(owner)
    safe_kernel_slug = _safe_slug(slug)
    kernel_ref = f"{safe_owner}/{safe_kernel_slug}"
    private_env = base64.b64encode(
        json.dumps(
            {
                "coordinator_url": coordinator_url,
                "coordinator_token": coordinator_token,
                "run_id": run_id,
            },
            separators=(",", ":"),
        ).encode("utf-8")
    ).decode("ascii")
    (package / "kernel.py").write_text(
        render_kernel(
            role=role,
            bundle_archive_b64=_bundle_archive_b64(fixture),
            private_env_b64=private_env,
        ),
        encoding="utf-8",
    )
    _write_json(
        package / "kernel-metadata.json",
        {
            "id": kernel_ref,
            "title": safe_kernel_slug.replace("-", " ").title(),
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
        "schema": PACKAGE_SCHEMA,
        "ok": True,
        "kernel_ref": kernel_ref,
        "package_dir": str(package),
        "role": role,
        "worker_report_name": "training_cuda_two_node_worker.json",
        "private_kernel": True,
        "private_coordinator_inputs_embedded": True,
        "coordinator_url_public": False,
        "coordinator_token_public": False,
        "credentials_public": False,
        "public_artifact_safe": True,
    }
    _write_json(output / "training_cuda_two_node_package.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--role", choices=["stage0", "stage1"], required=True)
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--coordinator-url", required=True)
    parser.add_argument("--coordinator-token", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_package(
        args.output_dir,
        owner=args.owner,
        slug=args.slug,
        role=args.role,
        fixture_dir=args.fixture_dir,
        coordinator_url=args.coordinator_url,
        coordinator_token=args.coordinator_token,
        run_id=args.run_id,
    )
    if args.json:
        public = {key: value for key, value in report.items() if key != "package_dir"}
        print(json.dumps(public, sort_keys=True))
    else:
        print(report["package_dir"])


if __name__ == "__main__":
    main()
