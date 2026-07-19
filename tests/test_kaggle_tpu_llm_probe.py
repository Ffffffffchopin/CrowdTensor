import argparse
import json
import py_compile
from pathlib import Path

from scripts import kaggle_tpu_llm_probe as probe


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=str(tmp_path),
        kaggle_owner="tester",
        kernel_slug_prefix="cttpu-llm",
        accelerators="TPU v5e-8",
        kernel_timeout_seconds=900,
        kaggle_push_timeout_seconds=1.0,
        kaggle_status_timeout_seconds=1.0,
        kaggle_status_poll_interval=1.0,
        kaggle_output_timeout_seconds=1.0,
        kaggle_delete_timeout_seconds=1.0,
        skip_kaggle_cleanup=False,
        keep_private_package=False,
        keep_kaggle_logs=False,
        json=False,
    )


def test_rendered_kernel_compiles(tmp_path: Path) -> None:
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text(probe.render_kernel("TPU v5e-8"), encoding="utf-8")

    py_compile.compile(str(kernel_path), doraise=True)

    rendered = kernel_path.read_text(encoding="utf-8")
    assert "kaggle_tpu_llm_runtime_v1" in rendered
    assert "synthetic_llm_ready" in rendered
    assert "raw_prompt_public" in rendered
    assert "next_token_hash" in rendered


def test_default_accelerators_prioritize_kaggle_internal_v5e_shape() -> None:
    accelerators = probe.parse_accelerators("")

    assert accelerators[0] == "tpuV5e8"
    assert "TPU v5e-8" in accelerators
    assert "Tpu1VmV38" in accelerators


def test_build_package_requests_private_tpu_kernel(tmp_path: Path) -> None:
    args = _args(tmp_path)

    package = probe.build_package(args, output_dir=tmp_path, accelerator="TPU v5e-8")

    metadata = json.loads((package["kernel_dir"] / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"].startswith("tester/cttpu-llm-")
    assert metadata["is_private"] == "true"
    assert metadata["enable_tpu"] == "true"
    assert metadata["enable_gpu"] == "false"
    assert metadata["machine_shape"] == "TPU v5e-8"
    assert package["metadata"]["title"] == metadata["title"]


def test_public_step_redacts_private_payload_path() -> None:
    step = {
        "name": "kaggle_kernel_push",
        "ok": True,
        "stdout_tail": "/tmp/private-kaggle-tpu-kernels/pkg pushed",
        "stderr_tail": "using private-kaggle-tpu-kernels",
        "command_public": ["kaggle", "kernels", "push", "-p", "/tmp/private-kaggle-tpu-kernels/pkg"],
    }

    public = probe.public_step(step)

    assert "private-kaggle-tpu-kernels" not in json.dumps(public)
    assert "<private-payload-dir>" in json.dumps(public)


def test_build_report_ready_requires_tpu_llm_and_cleanup(tmp_path: Path) -> None:
    args = _args(tmp_path)
    selected_report = {
        "schema": probe.STAGE_SCHEMA,
        "requested_accelerator": "TPU v5e-8",
        "ok": True,
        "tpu_runtime_ready": True,
        "llm_inference_ready": True,
        "diagnosis_codes": ["kaggle_tpu_synthetic_llm_ready"],
        "jax": {
            "synthetic_llm_runtime": "jax_tiny_causal_lm_jit",
            "generated_token_count": 1,
        },
    }
    attempts = [
        {
            "accelerator": "TPU v5e-8",
            "kernel_ref": "tester/cttpu-llm",
            "steps": [
                {
                    "name": "kaggle_kernel_push",
                    "ok": True,
                    "accepted": True,
                    "stdout_tail": "Kernel version 1 successfully pushed",
                },
                {"name": "kaggle_kernel_delete", "ok": True},
            ],
        }
    ]

    report = probe.build_report(
        args,
        output_dir=tmp_path,
        accelerator_attempts=attempts,
        selected_report=selected_report,
    )

    assert report["ok"] is True
    assert report["fresh_kaggle_run_performed"] is True
    assert report["selected_accelerator"] == "TPU v5e-8"
    assert report["tpu_runtime_ready"] is True
    assert report["llm_inference_ready"] is True
    assert report["generated_token_count"] == 1
    assert report["kaggle_lifecycle"]["kernels_deleted"] is True
    assert report["kaggle_lifecycle"]["private_packages_removed"] is True


def test_build_report_blocks_when_private_payload_retained(tmp_path: Path) -> None:
    args = _args(tmp_path)
    (tmp_path / "private-kaggle-tpu-kernels").mkdir()

    report = probe.build_report(
        args,
        output_dir=tmp_path,
        accelerator_attempts=[
            {
                "accelerator": "TPU v5e-8",
                "steps": [
                    {"name": "kaggle_kernel_push", "ok": True, "accepted": True},
                    {"name": "kaggle_kernel_delete", "ok": True},
                ],
            }
        ],
        selected_report={
            "requested_accelerator": "TPU v5e-8",
            "ok": True,
            "tpu_runtime_ready": True,
            "llm_inference_ready": True,
        },
    )

    assert report["ok"] is False
    assert "kaggle_tpu_private_package_retained" in report["blockers"]


def test_missing_runtime_report_preserves_queued_timeout(tmp_path: Path) -> None:
    args = _args(tmp_path)
    package = {"accelerator": "TPU v5e-8", "report_filename": "missing.json"}
    stage_output = tmp_path / "kaggle-output" / "tpu-v5e-8"
    stage_output.mkdir(parents=True)
    steps = [
        {
            "name": "kaggle_kernel_status",
            "terminal": False,
            "status": "QUEUED",
            "ok": False,
        }
    ]

    report = probe.missing_runtime_report_if_needed(
        package=package,
        stage_output=stage_output,
        steps=steps,
    )

    assert report is not None
    assert report["ok"] is False
    assert "kaggle_tpu_kernel_queued_timeout" in report["blockers"]
    assert "kaggle_tpu_report_missing" in report["diagnosis_codes"]
