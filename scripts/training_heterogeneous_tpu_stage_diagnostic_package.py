#!/usr/bin/env python3
"""Build a private Kaggle v5e-8 package for the bounded TPU stage diagnostic."""

from __future__ import annotations

import argparse
import base64
import json
import secrets
import shutil
import textwrap
from pathlib import Path
from typing import Any

from scripts.training_heterogeneous_beta_kaggle_package import (
    _bundle_archive_b64,
    _safe_slug,
)


PACKAGE_SCHEMA = "crowdtensor_heterogeneous_training_tpu_stage_diagnostic_package_v1"
KERNEL_SCHEMA = "crowdtensor_heterogeneous_training_tpu_stage_diagnostic_kernel_v1"
REPORT_NAME = "training_heterogeneous_tpu_stage_diagnostic_kernel.json"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def render_kernel(
    *,
    bundle_archive_b64: str,
    private_configuration_b64: str,
    diagnostic_nonce: str,
) -> str:
    source = f'''from __future__ import annotations

import base64
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
REPORT_PATH = WORKING / {REPORT_NAME!r}
PRIVATE_ROOT = WORKING / ".crowdtensor-tpu-stage-diagnostic"
BUNDLE_ROOT = PRIVATE_ROOT / "bundle"
PRIVATE_CONFIGURATION = PRIVATE_ROOT / "private_configuration.json"
BUNDLE_ARCHIVE_B64 = {bundle_archive_b64!r}
PRIVATE_CONFIGURATION_B64 = {private_configuration_b64!r}
DIAGNOSTIC_NONCE = {diagnostic_nonce!r}
KERNEL_SCHEMA = {KERNEL_SCHEMA!r}


def stable_hash(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def write_report(value):
    value.pop("content_hash", None)
    value["content_hash"] = stable_hash(value)
    REPORT_PATH.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
    )


def public_blocker(exc, phase):
    text = str(exc)
    if text.startswith(("heterogeneous_", "qwen15b_")):
        code = re.sub(r"[^a-zA-Z0-9:_-]", "_", text[:180])
    else:
        code = "heterogeneous_tpu_stage_diagnostic_failed:" + type(exc).__name__
    return f"{{phase}}:{{code}}"


def update_phase(report, phase, **progress):
    event = {{
        "phase": str(phase),
        "observed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **progress,
    }}
    report["phase"] = str(phase)
    report["progress"] = event
    report["phase_history"] = [*(report.get("phase_history") or []), event][-512:]
    write_report(report)


def ensure_dependencies():
    required = {{"safetensors": "0.7.0"}}
    installed = {{}}
    for name in required:
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = ""
    if any(installed[name] != version for name, version in required.items()):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-cache-dir",
                "safetensors==0.7.0",
            ],
            check=True,
            timeout=600,
        )
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, jax; "
                "d=[x for x in jax.devices() "
                "if str(getattr(x,'platform','')).lower()=='tpu']; "
                "print(json.dumps({{'jax':str(jax.__version__),'count':len(d)}}))"
            ),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
    )
    payload = {{}}
    for line in reversed((probe.stdout or "").splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    if probe.returncode != 0 or int(payload.get("count") or 0) != 8:
        raise RuntimeError("heterogeneous_jax_tpu_device_count_invalid")
    return {{
        "safetensors": importlib.metadata.version("safetensors"),
        "jax": str(payload.get("jax") or ""),
        "jax_tpu_device_count": int(payload["count"]),
        "tpu_probe_process_released": True,
    }}


def main():
    started = time.time()
    report = {{
        "schema": KERNEL_SCHEMA,
        "ok": False,
        "phase": "bootstrap",
        "phase_history": [],
        "progress": {{}},
        "model_id": "Qwen/Qwen2.5-7B",
        "model_revision": "d149729398750b98c0af14eb82c78cfe92750796",
        "stage_id": 2,
        "layer_start": 14,
        "layer_end": 20,
        "diagnostic_nonce_hash": stable_hash(DIAGNOSTIC_NONCE),
        "dependency_versions": {{}},
        "source_evidence": {{}},
        "shard_evidence": {{}},
        "jax_load_evidence": {{}},
        "training_step_evidence": {{}},
        "blockers": [],
        "synthetic_boundary_tensors_used": True,
        "full_training_gate_evidence": False,
        "same_job_three_accelerator_evidence": False,
        "credential_values_public": False,
        "credential_paths_public": False,
        "raw_training_text_public": False,
        "token_ids_public": False,
        "activation_values_public": False,
        "gradient_values_public": False,
        "checkpoint_tensor_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }}
    update_phase(report, "bootstrap")
    try:
        PRIVATE_ROOT.mkdir(parents=True, exist_ok=True)
        BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(
            io.BytesIO(base64.b64decode(BUNDLE_ARCHIVE_B64)), "r"
        ) as archive:
            archive.extractall(BUNDLE_ROOT)
        PRIVATE_CONFIGURATION.write_bytes(
            base64.b64decode(PRIVATE_CONFIGURATION_B64)
        )
        PRIVATE_CONFIGURATION.chmod(0o600)
        sys.path.insert(0, str(BUNDLE_ROOT))
        update_phase(report, "dependency_check")
        report["dependency_versions"] = ensure_dependencies()
        update_phase(
            report,
            "tpu_devices_verified",
            jax_tpu_device_count=report["dependency_versions"]["jax_tpu_device_count"],
        )

        from crowdtensor.heterogeneous_jax_qwen_training import JaxQwenStageTrainer
        from crowdtensor.heterogeneous_qwen_source import (
            materialize_qwen_stage_shard,
            resolve_qwen_source,
        )
        from crowdtensor.heterogeneous_training_manifest import (
            qwen25_7b_lora_tpu_manifest,
        )

        private = json.loads(PRIVATE_CONFIGURATION.read_text(encoding="utf-8"))
        hf_token = str(private.get("hf_token") or "")
        manifest = qwen25_7b_lora_tpu_manifest()
        update_phase(report, "source_metadata_resolving")
        config, _index, source = resolve_qwen_source(manifest, token=hf_token)
        report["source_evidence"] = source
        update_phase(
            report,
            "source_metadata_verified",
            source_verified=source.get("source_verified") is True,
            weight_file_count=int(source.get("weight_file_count") or 0),
        )

        shard_path = PRIVATE_ROOT / "stage2.safetensors"

        def shard_progress(event):
            update_phase(report, str(event.get("phase") or "stage_materializing"), **{{
                key: value for key, value in event.items() if key != "phase"
            }})

        shard = materialize_qwen_stage_shard(
            manifest,
            stage_id=2,
            output_path=shard_path,
            token=hf_token,
            max_group_bytes=128 * 1024 * 1024,
            progress_callback=shard_progress,
        )
        report["shard_evidence"] = {{
            key: value for key, value in shard.items() if key != "shard_path"
        }}
        update_phase(
            report,
            "jax_stage_loading",
            shard_byte_count=int(shard.get("shard_byte_count") or 0),
        )
        trainer = JaxQwenStageTrainer(
            training_manifest=manifest,
            config=config,
            stage_id=2,
            shard_path=shard_path,
            checkpoint_dir=PRIVATE_ROOT / "checkpoint",
            placement_generation=1,
            resume=False,
            require_tpu=True,
            expected_tpu_devices=8,
        )
        status_before = trainer.status()
        report["jax_load_evidence"] = {{
            **trainer.load_report,
            "adapter_hash_before": status_before["adapter_hash"],
            "jax_mesh_device_count": status_before["jax_mesh_device_count"],
            "all_mesh_devices_used": status_before["all_mesh_devices_used"],
        }}
        update_phase(
            report,
            "jax_stage_loaded",
            jax_mesh_device_count=status_before["jax_mesh_device_count"],
            all_mesh_devices_used=status_before["all_mesh_devices_used"],
        )

        import numpy as np

        shape = (1, 8, int(manifest["model"]["hidden_size"]))
        activation = np.full(shape, 0.001, dtype=np.float32)
        incoming_gradient = np.full(shape, 0.0001, dtype=np.float32)
        trainer.begin_step()
        update_phase(report, "forward_compiling")
        forward = trainer.forward(0, activation)
        update_phase(report, "backward_compiling")
        backward = trainer.backward(0, incoming_gradient)
        update_phase(report, "optimizer_updating")
        finish = trainer.finish_step(global_step=1, dataset_cursor=1)
        status_after = trainer.status()
        report["training_step_evidence"] = {{
            "forward_executed": True,
            "backward_executed": True,
            "optimizer_executed": True,
            "forward_activation_hash": str(forward.get("activation_hash") or ""),
            "input_gradient_hash": str(backward.get("input_gradient_hash") or ""),
            "lora_gradient_norm": float(finish.get("lora_gradient_norm") or 0.0),
            "adapter_hash_before": status_before["adapter_hash"],
            "adapter_hash_after": status_after["adapter_hash"],
            "adapter_hash_changed": status_before["adapter_hash"] != status_after["adapter_hash"],
            "compile_latency_ms": float(status_after.get("compile_latency_ms") or 0.0),
            "checkpoint_hash": str(finish.get("checkpoint_hash") or ""),
            "synthetic_boundary_tensors_used": True,
            "full_training_gate_evidence": False,
        }}
        report["ok"] = bool(
            report["jax_load_evidence"].get("jax_tpu_device_count") == 8
            and report["jax_load_evidence"].get("all_mesh_devices_used") is True
            and report["training_step_evidence"]["adapter_hash_changed"] is True
            and report["training_step_evidence"]["lora_gradient_norm"] > 0.0
        )
        if not report["ok"]:
            report["blockers"].append(
                "heterogeneous_tpu_stage_diagnostic_acceptance_incomplete"
            )
        update_phase(report, "diagnostic_completed", diagnostic_ok=report["ok"])
    except BaseException as exc:
        failure_phase = str(report.get("phase") or "unknown")
        report["failure_phase"] = failure_phase
        report["blockers"].append(public_blocker(exc, failure_phase))
    finally:
        shutil.rmtree(PRIVATE_ROOT, ignore_errors=True)
        report["private_runtime_removed"] = not PRIVATE_ROOT.exists()
        report["elapsed_seconds"] = round(time.time() - started, 3)
        report["blockers"] = sorted(set(report["blockers"]))
        report["ok"] = bool(
            report["ok"] and report["private_runtime_removed"] and not report["blockers"]
        )
        write_report(report)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
'''
    return textwrap.dedent(source)


def build_package(
    output_dir: str | Path,
    *,
    owner: str,
    slug: str,
    hf_token: str = "",
    diagnostic_nonce: str | None = None,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    shutil.rmtree(output, ignore_errors=True)
    package = output / "private-kernel"
    package.mkdir(parents=True, exist_ok=True)
    safe_owner = _safe_slug(owner)
    safe_kernel_slug = _safe_slug(slug)
    kernel_ref = f"{safe_owner}/{safe_kernel_slug}"
    private_configuration_b64 = base64.b64encode(
        json.dumps({"hf_token": str(hf_token)}, separators=(",", ":")).encode(
            "utf-8"
        )
    ).decode("ascii")
    nonce = str(diagnostic_nonce or secrets.token_urlsafe(24))
    (package / "kernel.py").write_text(
        render_kernel(
            bundle_archive_b64=_bundle_archive_b64(),
            private_configuration_b64=private_configuration_b64,
            diagnostic_nonce=nonce,
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
            "enable_gpu": "false",
            "enable_tpu": "true",
            "enable_internet": "true",
            "machine_shape": "tpuV5e8",
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
        "private_kernel": True,
        "requested_accelerator": "tpuV5e8",
        "expected_tpu_device_count": 8,
        "model_id": "Qwen/Qwen2.5-7B",
        "stage_id": 2,
        "layer_start": 14,
        "layer_end": 20,
        "diagnostic_only": True,
        "full_training_gate_evidence": False,
        "private_hf_token_embedded": bool(hf_token),
        "credential_values_public": False,
        "credential_paths_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    _write_json(output / "training_heterogeneous_tpu_stage_diagnostic_package.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    import os

    report = build_package(
        args.output_dir,
        owner=args.owner,
        slug=args.slug,
        hf_token=str(os.environ.get(args.hf_token_env) or ""),
    )
    public = {key: value for key, value in report.items() if key != "package_dir"}
    if args.json:
        print(json.dumps(public, sort_keys=True))
    else:
        print("training_heterogeneous_tpu_stage_diagnostic_package ok=True")


if __name__ == "__main__":
    main()
