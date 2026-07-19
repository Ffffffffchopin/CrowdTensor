#!/usr/bin/env python3
"""Build the private Kaggle T4x2 package for the CUDA pipeline gate."""

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
PACKAGE_SCHEMA = "crowdtensor_cuda_single_kernel_package_v1"
WORKER_SCHEMA = "crowdtensor_cuda_single_kernel_gate_v1"
SOURCE_FILES = [
    "__init__.py",
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
    return re.sub(r"-+", "-", slug)[:63].strip("-") or "ct-cuda-training"


def _source_archive_b64() -> str:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in SOURCE_FILES:
            archive.writestr(
                f"crowdtensor/{filename}",
                (ROOT / "crowdtensor" / filename).read_bytes(),
            )
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def render_kernel(*, total_steps: int, interrupt_after_step: int, source_archive_b64: str) -> str:
    source = f'''from __future__ import annotations

import base64
import hashlib
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
REPORT_PATH = WORKING / "training_cuda_single_kernel_gate.json"
CHECKPOINT_BUNDLE_PATH = WORKING / "training_cuda_single_kernel_checkpoint_bundle.zip"
PRIVATE_ROOT = WORKING / ".crowdtensor-cuda-single-private"
SOURCE_ROOT = PRIVATE_ROOT / "source"
SOURCE_ARCHIVE_B64 = "{source_archive_b64}"
SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
if not (SOURCE_ROOT / "crowdtensor" / "pipeline_lora_training.py").is_file():
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_ARCHIVE_B64)), "r") as archive:
        archive.extractall(SOURCE_ROOT)
sys.path.insert(0, str(SOURCE_ROOT))


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
        "baseline": PRIVATE_ROOT / "baseline" / "checkpoint",
        "resumed": PRIVATE_ROOT / "resumed" / "checkpoint",
    }}
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
        "contains_baseline_and_resumed_checkpoints": all(root.is_dir() for root in roots.values()),
        "checkpoint_values_public": False,
        "private_paths_public": False,
    }}


def ensure_dependencies():
    import importlib.metadata
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
        return {{"peft": str(peft.__version__), "safetensors": str(safetensors.__version__)}}
    except Exception:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "peft>=0.19,<1", "safetensors>=0.4,<1"],
            check=True,
            timeout=600,
        )
        import peft
        import safetensors
        return {{"peft": str(peft.__version__), "safetensors": str(safetensors.__version__)}}


started = time.time()
report = {{
    "schema": "{WORKER_SCHEMA}",
    "ok": False,
    "gpu_live_verified": False,
    "single_kernel_t4x2_verified": False,
    "kaggle_kernel": bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE")),
    "total_steps": {int(total_steps)},
    "interrupt_after_step": {int(interrupt_after_step)},
    "blockers": [],
    "activation_values_public": False,
    "gradient_values_public": False,
    "checkpoint_values_public": False,
    "raw_training_text_public": False,
    "credentials_public": False,
    "private_paths_public": False,
    "public_artifact_safe": True,
}}
try:
    dependencies = ensure_dependencies()
    import torch
    from crowdtensor.pipeline_lora_training import compare_pipeline_runs, run_two_cuda_process_pipeline

    report["dependencies"] = {{
        **dependencies,
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda or ""),
    }}
    report["cuda_available"] = bool(torch.cuda.is_available())
    report["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if report["cuda_device_count"] < 2:
        raise RuntimeError("single_kernel_gate_requires_two_cuda_devices")
    PRIVATE_ROOT.mkdir(parents=True, exist_ok=True)
    baseline = run_two_cuda_process_pipeline(
        PRIVATE_ROOT / "baseline",
        total_steps={int(total_steps)},
        seed=20260710,
    )
    resumed = run_two_cuda_process_pipeline(
        PRIVATE_ROOT / "resumed",
        total_steps={int(total_steps)},
        interrupt_stage1_after_step={int(interrupt_after_step)},
        seed=20260710,
    )
    comparison = compare_pipeline_runs(baseline, resumed, atol=0.005, rtol=0.005)
    baseline_public = json.loads(Path(baseline["report_path"]).read_text(encoding="utf-8"))
    resumed_public = json.loads(Path(resumed["report_path"]).read_text(encoding="utf-8"))
    report.update({{
        "baseline": baseline_public,
        "resumed": resumed_public,
        "resume_equivalence": comparison,
        "two_distinct_processes": bool(baseline_public.get("distinct_stage_pids")),
        "two_distinct_cuda_devices": bool(baseline_public.get("distinct_cuda_devices")),
        "real_activation_transport": bool(baseline_public.get("real_activation_transport")),
        "real_backward_gradient_transport": bool(baseline_public.get("real_backward_gradient_transport")),
        "real_cuda_backward": bool(baseline_public.get("real_cuda_backward")),
        "no_stage_loaded_full_model": bool(baseline_public.get("no_stage_loaded_full_model")),
        "base_weights_frozen": bool(baseline_public.get("base_weights_frozen") and resumed_public.get("base_weights_frozen")),
        "positive_lora_gradient_norms": bool(baseline_public.get("positive_lora_gradient_norms") and resumed_public.get("positive_lora_gradient_norms")),
        "positive_cuda_memory": bool(baseline_public.get("positive_cuda_memory") and resumed_public.get("positive_cuda_memory")),
        "loss_reduced": bool(baseline_public.get("loss_reduced") and resumed_public.get("loss_reduced")),
        "checkpoint_resume_verified": bool(comparison.get("checkpoint_resume_verified")),
        "controlled_stage_restart": bool((resumed_public.get("interruption") or {{}}).get("worker_restarted")),
    }})
    report["gpu_live_verified"] = bool(
        baseline_public.get("gpu_live_verified") and resumed_public.get("gpu_live_verified")
    )
    report["single_kernel_t4x2_verified"] = bool(
        report["kaggle_kernel"]
        and report["gpu_live_verified"]
        and report["cuda_device_count"] >= 2
        and report["two_distinct_processes"]
        and report["two_distinct_cuda_devices"]
        and report["real_activation_transport"]
        and report["real_backward_gradient_transport"]
        and report["real_cuda_backward"]
        and report["no_stage_loaded_full_model"]
        and report["base_weights_frozen"]
        and report["positive_lora_gradient_norms"]
        and report["positive_cuda_memory"]
        and report["loss_reduced"]
        and report["checkpoint_resume_verified"]
        and report["controlled_stage_restart"]
    )
    report["ok"] = report["single_kernel_t4x2_verified"]
    if not report["ok"]:
        report["blockers"].append("single_kernel_t4x2_gate_incomplete")
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
    report["single_kernel_t4x2_verified"] = bool(
        report.get("single_kernel_t4x2_verified")
        and (report.get("checkpoint_bundle") or {{}}).get("present")
        and (report.get("checkpoint_bundle") or {{}}).get(
            "contains_baseline_and_resumed_checkpoints"
        )
    )
    shutil.rmtree(PRIVATE_ROOT, ignore_errors=True)
    report["cleanup"] = {{
        "private_runtime_removed": not PRIVATE_ROOT.exists(),
        "worker_processes_stopped": bool(
            (report.get("baseline") or {{}}).get("cleanup", {{}}).get("all_worker_processes_stopped")
            and (report.get("resumed") or {{}}).get("cleanup", {{}}).get("all_worker_processes_stopped")
        ),
    }}
    report["elapsed_seconds"] = time.time() - started
    report["ok"] = bool(
        report.get("ok")
        and report["single_kernel_t4x2_verified"]
        and report["cleanup"]["private_runtime_removed"]
        and report["cleanup"]["worker_processes_stopped"]
    )
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
    total_steps: int = 4,
    interrupt_after_step: int = 2,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    if output.exists():
        shutil.rmtree(output)
    package = output / "private-kernel"
    source_dir = package / "crowdtensor"
    source_dir.mkdir(parents=True, exist_ok=True)
    for filename in SOURCE_FILES:
        shutil.copyfile(ROOT / "crowdtensor" / filename, source_dir / filename)
    (package / "kernel.py").write_text(
        render_kernel(
            total_steps=total_steps,
            interrupt_after_step=interrupt_after_step,
            source_archive_b64=_source_archive_b64(),
        ),
        encoding="utf-8",
    )
    safe_owner = _safe_slug(owner)
    safe_kernel_slug = _safe_slug(slug)
    kernel_ref = f"{safe_owner}/{safe_kernel_slug}"
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
        "worker_report_name": "training_cuda_single_kernel_gate.json",
        "total_steps": int(total_steps),
        "interrupt_after_step": int(interrupt_after_step),
        "source_files": SOURCE_FILES,
        "private_kernel": True,
        "credentials_embedded": False,
        "public_artifact_safe": True,
    }
    _write_json(output / "training_cuda_single_kernel_package.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--total-steps", type=int, default=4)
    parser.add_argument("--interrupt-after-step", type=int, default=2)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.total_steps < 4:
        parser.error("--total-steps must be at least 4")
    if args.interrupt_after_step <= 0 or args.interrupt_after_step >= args.total_steps:
        parser.error("--interrupt-after-step must be inside the training run")
    report = build_package(
        args.output_dir,
        owner=args.owner,
        slug=args.slug,
        total_steps=args.total_steps,
        interrupt_after_step=args.interrupt_after_step,
    )
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(report["package_dir"])


if __name__ == "__main__":
    main()
