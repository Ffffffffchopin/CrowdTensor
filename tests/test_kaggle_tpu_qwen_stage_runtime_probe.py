from __future__ import annotations

import argparse
import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import kaggle_tpu_qwen_stage_runtime_probe as probe


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=str(tmp_path),
        kaggle_owner="tester",
        kernel_slug_prefix="cttpu-qwen-stage",
        accelerators="tpuV5e8",
        stage_profile="qwen32b-one-layer",
        kernel_timeout_seconds=1800,
        kaggle_push_timeout_seconds=1.0,
        kaggle_status_timeout_seconds=1.0,
        kaggle_status_poll_interval=1.0,
        kaggle_output_timeout_seconds=1.0,
        kaggle_delete_timeout_seconds=1.0,
        skip_kaggle_cleanup=False,
        keep_private_package=False,
        json=False,
    )


def test_rendered_qwen_stage_kernel_compiles_for_profiles(tmp_path: Path) -> None:
    for profile in ["tiny-qwen-like", "qwen32b-one-layer"]:
        kernel_path = tmp_path / f"{profile}.py"
        kernel_path.write_text(probe.render_kernel("tpuV5e8", stage_profile=profile), encoding="utf-8")

        py_compile.compile(str(kernel_path), doraise=True)

        rendered = kernel_path.read_text(encoding="utf-8")
        assert "kaggle_tpu_qwen_stage_runtime_v1" in rendered
        assert "qwen_like_stage_runtime_ready" in rendered
        assert "stage_output_hash" in rendered
        assert "stage_output_payload_public" in rendered
        assert "'qwen32b_shape_profile':" in rendered
        assert '"qwen32b_shape_profile": false' not in rendered
        assert "false" not in rendered.split("PROFILE =", 1)[1].split("\n", 1)[0]


def test_build_package_requests_private_tpu_kernel(tmp_path: Path) -> None:
    args = _args(tmp_path)

    package = probe.build_package(args, output_dir=tmp_path, accelerator="tpuV5e8")

    metadata = json.loads((package["kernel_dir"] / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"].startswith("tester/cttpu-qwen-stage-")
    assert metadata["is_private"] == "true"
    assert metadata["enable_tpu"] == "true"
    assert metadata["enable_gpu"] == "false"
    assert metadata["machine_shape"] == "tpuV5e8"
    assert package["stage_profile"] == "qwen32b-one-layer"


def test_public_step_redacts_private_payload_path() -> None:
    step = {
        "name": "kaggle_kernel_push",
        "ok": True,
        "stdout_tail": "/tmp/private-kaggle-tpu-qwen-stage-kernels/pkg pushed",
        "stderr_tail": "using private-kaggle-tpu-qwen-stage-kernels",
        "command_public": ["kaggle", "kernels", "push", "-p", "/tmp/private-kaggle-tpu-qwen-stage-kernels/pkg"],
    }

    public = probe.public_step(step)

    encoded = json.dumps(public)
    assert "private-kaggle-tpu-qwen-stage-kernels" not in encoded
    assert "<private-payload-dir>" in encoded


def test_build_report_ready_requires_qwen_stage_and_cleanup(tmp_path: Path) -> None:
    args = _args(tmp_path)
    selected_report = {
        "schema": probe.STAGE_SCHEMA,
        "requested_accelerator": "tpuV5e8",
        "ok": True,
        "tpu_runtime_ready": True,
        "qwen_like_stage_runtime_ready": True,
        "qwen32b_single_layer_runtime_ready": True,
        "stage_local_kv_cache_verified": True,
        "diagnosis_codes": ["kaggle_tpu_qwen_stage_runtime_ready"],
        "jax_tpu_stage": {
            "shape_metadata": {
                "input_shape": [1, 1, 5120],
                "output_shape": [1, 1, 5120],
                "dtype": "bfloat16",
                "layout": "batch_seq_hidden",
            },
            "stage_input_hash": "sha256:in",
            "stage_output_hash": "sha256:out",
        },
    }
    attempts = [
        {
            "accelerator": "tpuV5e8",
            "kernel_ref": "tester/cttpu-qwen-stage",
            "steps": [
                {"name": "kaggle_kernel_push", "ok": True, "accepted": True, "stdout_tail": "Kernel version 1 successfully pushed"},
                {"name": "kaggle_kernel_delete", "ok": True},
            ],
        }
    ]

    report = probe.build_report(args, output_dir=tmp_path, accelerator_attempts=attempts, selected_report=selected_report)

    assert report["ok"] is True
    assert report["qwen_like_stage_runtime_ready"] is True
    assert report["qwen32b_single_layer_runtime_ready"] is True
    assert report["stage_local_kv_cache_verified"] is True
    assert report["tpu_32b_runtime_adapter_ready"] is False
    assert report["stage_output_hash"] == "sha256:out"
    assert report["kaggle_lifecycle"]["kernels_deleted"] is True
    assert report["kaggle_lifecycle"]["private_packages_removed"] is True


def test_build_report_accepts_web_jupyter_runtime_execution(tmp_path: Path) -> None:
    args = _args(tmp_path)
    selected_report = {
        "schema": probe.STAGE_SCHEMA,
        "requested_accelerator": "web-ui-tpu-v5e8",
        "ok": True,
        "tpu_runtime_ready": True,
        "qwen_like_stage_runtime_ready": True,
        "qwen32b_single_layer_runtime_ready": True,
        "stage_local_kv_cache_verified": True,
        "diagnosis_codes": ["kaggle_tpu_qwen_stage_runtime_ready"],
        "jax_tpu_stage": {
            "shape_metadata": {
                "input_shape": [1, 1, 5120],
                "output_shape": [1, 1, 5120],
                "dtype": "bfloat16",
                "layout": "batch_seq_hidden",
            },
            "stage_input_hash": "sha256:in",
            "stage_output_hash": "sha256:out",
        },
    }

    report = probe.build_report(
        args,
        output_dir=tmp_path,
        accelerator_attempts=[
            {
                "accelerator": "web-ui-tpu-v5e8",
                "kernel_ref": "kaggle-web-notebook-active-event",
                "steps": [
                    {"name": "jupyter_proxy_kernel_discovered", "ok": True, "accepted": True},
                    {"name": "jupyter_ws_execute", "ok": True},
                ],
            }
        ],
        selected_report=selected_report,
    )

    assert report["ok"] is True
    assert report["selected_accelerator"] == "web-ui-tpu-v5e8"
    assert report["kaggle_lifecycle"]["private_kernel_push_count"] == 0
    assert report["kaggle_lifecycle"]["web_runtime_execution_count"] == 1
    assert report["kaggle_lifecycle"]["kernels_deleted"] is True


def test_build_report_blocks_when_private_payload_retained(tmp_path: Path) -> None:
    args = _args(tmp_path)
    (tmp_path / "private-kaggle-tpu-qwen-stage-kernels").mkdir()

    report = probe.build_report(
        args,
        output_dir=tmp_path,
        accelerator_attempts=[
            {
                "accelerator": "tpuV5e8",
                "steps": [
                    {"name": "kaggle_kernel_push", "ok": True, "accepted": True},
                    {"name": "kaggle_kernel_delete", "ok": True},
                ],
            }
        ],
        selected_report={
            "requested_accelerator": "tpuV5e8",
            "ok": True,
            "tpu_runtime_ready": True,
            "qwen_like_stage_runtime_ready": True,
        },
    )

    assert report["ok"] is False
    assert "kaggle_tpu_private_package_retained" in report["blockers"]


def test_missing_runtime_report_preserves_queued_timeout(tmp_path: Path) -> None:
    package = {"accelerator": "tpuV5e8", "report_filename": "missing.json", "stage_profile": "qwen32b-one-layer"}
    stage_output = tmp_path / "kaggle-output" / "tpuv5e8"
    stage_output.mkdir(parents=True)
    steps = [{"name": "kaggle_kernel_status", "terminal": False, "status": "QUEUED", "ok": False}]

    report = probe.missing_runtime_report_if_needed(package=package, stage_output=stage_output, steps=steps)

    assert report is not None
    assert report["ok"] is False
    assert "kaggle_tpu_kernel_queued_timeout" in report["blockers"]
    assert "kaggle_tpu_qwen_stage_report_missing" in report["diagnosis_codes"]
