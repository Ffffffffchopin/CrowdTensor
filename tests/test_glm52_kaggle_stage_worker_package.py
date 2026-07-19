from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from scripts import glm52_kaggle_stage_worker_package as package
from scripts import glm52_kaggle_stage_worker_package_check as check
from scripts import glm52_kaggle_stage_runtime_check as runtime_check


def _tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="ct_glm52_stage_worker_pkg_"))


def _plan() -> dict:
    return {
        "schema": "glm52_kaggle_stage_runtime_plan_v1",
        "ok": True,
        "glm52_stage_runtime_plan_ready": True,
        "stage_runtime_adapter_verified": False,
        "stage_specs": [
            {
                "provider": provider,
                "stage_id": index,
                "stage_layer_range": [index * 26, (index + 1) * 26],
                "compatible_weight_repo": "cyankiwi/GLM-5.2-AWQ-INT4",
                "runtime_adapter": f"{provider}_adapter",
                "public_artifact_safe": True,
            }
            for index, provider in enumerate(package.REQUIRED_PROVIDERS)
        ],
        "public_artifact_safe": True,
    }


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_build_report_renders_three_private_kaggle_packages() -> None:
    out = _tmp_dir()
    plan_path = _write(out / "plan.json", _plan())
    args = package.parse_args([
        "--output-dir",
        str(out / "pkg"),
        "--stage-runtime-plan-report",
        str(plan_path),
        "--kaggle-owner",
        "tester",
    ])

    report = package.build_report(args)

    assert report["glm52_stage_worker_package_ready"] is True
    assert report["live_run_performed"] is False
    assert report["stage_runtime_package_kind"] == "value_op"
    assert {entry["provider"] for entry in report["packages"]} == set(package.REQUIRED_PROVIDERS)
    for entry in report["packages"]:
        assert entry["stage_runtime_package_kind"] == "value_op"
        assert entry["bundled_runtime_files"] == []
        kernel = Path(entry["kernel_path"])
        metadata = Path(entry["metadata_path"])
        assert kernel.is_file()
        assert metadata.is_file()
        kernel_source = kernel.read_text(encoding="utf-8")
        compile(kernel_source, str(kernel), "exec")
        assert 'key_text in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}' in kernel_source
        assert 'os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")' in kernel_source
        assert 'request_headers["Authorization"] = "Bearer " + str(hf_token)' in kernel_source
        metadata_payload = json.loads(metadata.read_text(encoding="utf-8"))
        assert entry["kaggle_owner"] == "tester"
        assert metadata_payload["id"].startswith("tester/")
        assert entry["kernel_ref"] == metadata_payload["id"]
        assert metadata_payload["is_private"] == "true"
        assert metadata_payload["enable_internet"] == "true"
        if entry["provider"] == "kaggle_cuda":
            assert metadata_payload["machine_shape"] == "NvidiaTeslaT4"
        if entry["provider"] == "kaggle_jax_tpu":
            assert metadata_payload["machine_shape"] == "tpuV5e8"
    assert check.validate_report(report) == []


def test_build_report_can_bundle_full_prefix_stage_decode_runtime() -> None:
    out = _tmp_dir()
    plan_path = _write(out / "plan.json", _plan())
    args = package.parse_args([
        "--output-dir",
        str(out / "pkg"),
        "--stage-runtime-plan-report",
        str(plan_path),
        "--runtime-kind",
        "full_prefix_stage_decode",
    ])

    report = package.build_report(args)

    assert report["glm52_stage_worker_package_ready"] is True
    assert report["stage_runtime_package_kind"] == "full_prefix_stage_decode"
    assert report["full_prefix_probe_mode"] == "default"
    assert report["full_prefix_probe_full_stage_requested"] is False
    assert report["full_prefix_timeout_seconds"] == 3600
    assert report["full_prefix_runtime_bundle_required"] is True
    expected_probe_ranges = {
        "kaggle_cuda": [6, 8],
        "kaggle_jax_tpu": [26, 28],
        "kaggle_cpu": [54, 56],
    }
    for entry in report["packages"]:
        assert entry["stage_runtime_package_kind"] == "full_prefix_stage_decode"
        assert entry["stage_count"] == 3
        assert entry["full_prefix_probe_mode"] == "default"
        assert entry["full_prefix_timeout_seconds"] == 3600
        assert entry["full_prefix_probe_layer_range"] == expected_probe_ranges[entry["provider"]]
        assert entry["full_prefix_probe_covers_full_stage"] is False
        assert entry["full_prefix_runtime_bundle_present"] is True
        assert entry["embedded_runtime_bundle_present"] is True
        assert entry["embedded_runtime_bundle_file_count"] > 0
        bundle_names = {Path(item["relative_path"]).name for item in entry["bundled_runtime_files"]}
        assert "glm52_full_prefix_stage_decode_probe.py" in bundle_names
        assert "glm52_dsa_masked_layer_decode_probe.py" in bundle_names
        kernel_source = Path(entry["kernel_path"]).read_text(encoding="utf-8")
        assert "run_full_prefix_stage_adapter" in kernel_source
        assert "EMBEDDED_FULL_PREFIX_RUNTIME_BUNDLE" in kernel_source
        assert "runtime_work_dir" in kernel_source
        assert "ensure_embedded_full_prefix_bundle" in kernel_source
        assert "preferred_stage_tensor_keys" in kernel_source
        assert "load_stage_tensor_value" in kernel_source
        assert 'CT_GLM52_FULL_PREFIX_MAX_TENSOR_BYTES", str(512 * 1024 * 1024)' in kernel_source
        assert 'CT_GLM52_FULL_PREFIX_TIMEOUT_SECONDS", "3600"' in kernel_source
        assert '"full_prefix_timeout_seconds": 3600' in kernel_source
        assert "--skip-lm-head" in kernel_source
        assert '"stage_runtime_package_kind": "full_prefix_stage_decode"' in kernel_source
        assert f'"full_prefix_probe_layer_range": {expected_probe_ranges[entry["provider"]]}' in kernel_source
        assert (Path(entry["kernel_path"]).parent / "scripts" / "glm52_full_prefix_stage_decode_probe.py").is_file()
        compile(kernel_source, str(entry["kernel_path"]), "exec")
    assert "glm52_stage_worker_package_is_not_runtime_success" in report["blockers"]
    assert check.validate_report(report) == []


def test_build_report_can_raise_full_prefix_timeout() -> None:
    out = _tmp_dir()
    plan_path = _write(out / "plan.json", _plan())
    args = package.parse_args([
        "--output-dir",
        str(out / "pkg"),
        "--stage-runtime-plan-report",
        str(plan_path),
        "--runtime-kind",
        "full_prefix_stage_decode",
        "--full-prefix-timeout-seconds",
        "7200",
    ])

    report = package.build_report(args)

    assert report["full_prefix_timeout_seconds"] == 7200
    for entry in report["packages"]:
        assert entry["full_prefix_timeout_seconds"] == 7200
        kernel_source = Path(entry["kernel_path"]).read_text(encoding="utf-8")
        assert '"full_prefix_timeout_seconds": 7200' in kernel_source
        assert 'CT_GLM52_FULL_PREFIX_TIMEOUT_SECONDS", str(STAGE.get("full_prefix_timeout_seconds") or 3600)' in kernel_source
        compile(kernel_source, str(entry["kernel_path"]), "exec")
    assert check.validate_report(report) == []


def test_build_report_can_render_full_stage_full_prefix_attempt_packages() -> None:
    out = _tmp_dir()
    plan_path = _write(out / "plan.json", _plan())
    args = package.parse_args([
        "--output-dir",
        str(out / "pkg"),
        "--stage-runtime-plan-report",
        str(plan_path),
        "--runtime-kind",
        "full_prefix_stage_decode",
        "--full-prefix-probe-mode",
        "full-stage",
    ])

    report = package.build_report(args)

    assert report["glm52_stage_worker_package_ready"] is True
    assert report["stage_runtime_package_kind"] == "full_prefix_stage_decode"
    assert report["full_prefix_probe_mode"] == "full-stage"
    assert report["full_prefix_probe_full_stage_requested"] is True
    for entry in report["packages"]:
        assert entry["stage_count"] == 3
        assert entry["full_prefix_probe_mode"] == "full-stage"
        assert entry["full_prefix_probe_layer_range"] == entry["stage_layer_range"]
        assert entry["full_prefix_probe_covers_full_stage"] is True
        kernel_source = Path(entry["kernel_path"]).read_text(encoding="utf-8")
        assert f'"full_prefix_probe_layer_range": {entry["stage_layer_range"]}' in kernel_source
        compile(kernel_source, str(entry["kernel_path"]), "exec")
    assert "glm52_stage_worker_package_is_not_runtime_success" in report["blockers"]
    assert check.validate_report(report) == []


def test_build_report_can_group_cpu_stage_workers_without_merging_stage_specs() -> None:
    out = _tmp_dir()
    plan = {
        **_plan(),
        "stage_specs": [
            {
                "provider": provider,
                "stage_id": stage_id,
                "stage_count": 7,
                "stage_layer_range": layer_range,
                "compatible_weight_repo": "cyankiwi/GLM-5.2-AWQ-INT4",
                "runtime_adapter": f"{provider}_adapter",
                "public_artifact_safe": True,
            }
            for stage_id, provider, layer_range in [
                (0, "kaggle_cuda", [0, 2]),
                (1, "kaggle_cpu", [2, 4]),
                (2, "kaggle_cpu", [4, 6]),
                (3, "kaggle_jax_tpu", [6, 8]),
                (4, "kaggle_cpu", [8, 10]),
                (5, "kaggle_cpu", [10, 12]),
                (6, "kaggle_cpu", [12, 14]),
            ]
        ],
    }
    plan_path = _write(out / "plan.json", plan)
    request_hash = "sha256:" + ("d" * 64)
    args = package.parse_args([
        "--output-dir",
        str(out / "pkg"),
        "--stage-runtime-plan-report",
        str(plan_path),
        "--runtime-kind",
        "full_prefix_stage_decode",
        "--coordinator-request-id-hash",
        request_hash,
        "--cpu-stage-group-size",
        "2",
    ])

    report = package.build_report(args)

    assert report["cpu_stage_group_size"] == 2
    assert report["grouped_stage_worker_package_count"] == 2
    assert len(report["packages"]) == 5
    grouped = [entry for entry in report["packages"] if entry.get("grouped_stage_worker") is True]
    assert [entry["stage_ids"] for entry in grouped] == [[1, 2], [4, 5]]
    for entry in grouped:
        assert entry["provider"] == "kaggle_cpu"
        assert entry["stage_specs"]
        assert entry["grouped_stage_count"] == 2
        kernel_source = Path(entry["kernel_path"]).read_text(encoding="utf-8")
        assert "STAGE_GROUP" in kernel_source
        assert "run_one_stage_attempt" in kernel_source
        assert "CT_GLM52_PRIVATE_RUNTIME_ENV_INLINE = {}" in kernel_source
        compile(kernel_source, str(entry["kernel_path"]), "exec")
        package_dir = Path(entry["package_dir"])
        for stage_id in entry["stage_ids"]:
            stage_script = package_dir / f"kernel_stage_{stage_id}.py"
            assert stage_script.is_file()
            compile(stage_script.read_text(encoding="utf-8"), str(stage_script), "exec")
    single_cpu = [entry for entry in report["packages"] if entry["provider"] == "kaggle_cpu" and entry.get("grouped_stage_worker") is not True]
    assert [entry["stage_ids"] for entry in single_cpu] == [[6]]
    assert check.validate_report(report) == []


def test_full_prefix_worker_template_embeds_coordinator_decode_bridge() -> None:
    out = _tmp_dir()
    plan_path = _write(out / "plan.json", _plan())
    request_hash = "sha256:" + ("c" * 64)
    args = package.parse_args([
        "--output-dir",
        str(out / "pkg"),
        "--stage-runtime-plan-report",
        str(plan_path),
        "--runtime-kind",
        "full_prefix_stage_decode",
        "--full-prefix-probe-mode",
        "full-stage",
        "--coordinator-request-id-hash",
        request_hash,
    ])

    report = package.build_report(args)

    for entry in report["packages"]:
        kernel_source = Path(entry["kernel_path"]).read_text(encoding="utf-8")
        assert "coordinator_decode_enabled" in kernel_source
        assert "run_coordinator_decode_worker" in kernel_source
        assert 'CT_GLM52_COORDINATOR_STAGE_TASK_LIMIT", "1"' in kernel_source
        assert "coordinator_post_json" in kernel_source
        assert "load_private_runtime_env_file" in kernel_source
        assert "CT_GLM52_PRIVATE_RUNTIME_ENV_INLINE = {}" in kernel_source
        assert "ct_glm52_private_runtime_env.json" in kernel_source
        assert "CT_GLM52_COORDINATOR_URL" in kernel_source
        assert "CT_GLM52_COORDINATOR_TOKEN" in kernel_source
        assert "X-CrowdTensor-GLM52-Token" in kernel_source
        assert "CT_GLM52_INPUT_HIDDEN_B64" in kernel_source
        assert "CT_GLM52_OUTPUT_ACTIVATION_PATH" in kernel_source
        assert '"stage_decode_verified": bool(verified and coordinator_mode)' in kernel_source
        assert '"generated_token_hash"' in kernel_source
        assert "coordinator_stage_last_full_prefix_adapter_verified" in kernel_source
        assert "coordinator_stage_last_full_prefix_blocker" in kernel_source
        assert "coordinator_stage_last_full_prefix_probe_exit_code" in kernel_source
        assert "coordinator_stage_last_full_prefix_probe_blockers" in kernel_source
        assert "coordinator_stage_last_full_prefix_probe_errors" in kernel_source
        assert "coordinator_stage_last_full_prefix_stdout_hash" in kernel_source
        assert "coordinator_stage_last_full_prefix_stderr_hash" in kernel_source
        assert '"activation" = ' not in kernel_source
        compile(kernel_source, str(entry["kernel_path"]), "exec")
    assert check.validate_report(report) == []


def test_build_report_can_assign_kaggle_owner_per_provider() -> None:
    out = _tmp_dir()
    plan_path = _write(out / "plan.json", _plan())
    args = package.parse_args([
        "--output-dir",
        str(out / "pkg"),
        "--stage-runtime-plan-report",
        str(plan_path),
        "--kaggle-owner",
        "defaultowner",
        "--provider-owner-map",
        "kaggle_cuda=gpuowner,kaggle_jax_tpu=tpuowner,kaggle_cpu=cpuowner",
    ])

    report = package.build_report(args)

    owners = {entry["provider"]: entry["kaggle_owner"] for entry in report["packages"]}
    assert owners == {
        "kaggle_cuda": "gpuowner",
        "kaggle_jax_tpu": "tpuowner",
        "kaggle_cpu": "cpuowner",
    }
    for entry in report["packages"]:
        metadata_payload = json.loads(Path(entry["metadata_path"]).read_text(encoding="utf-8"))
        assert metadata_payload["id"].startswith(entry["kaggle_owner"] + "/")
        assert entry["kernel_ref"].startswith(entry["kaggle_owner"] + "/")
    assert report["provider_owner_map"] == owners
    assert check.validate_report(report) == []


def test_checker_rejects_package_as_verified_success() -> None:
    out = _tmp_dir()
    plan_path = _write(out / "plan.json", _plan())
    args = package.parse_args(["--output-dir", str(out / "pkg"), "--stage-runtime-plan-report", str(plan_path)])
    report = package.build_report(args)

    errors = check.validate_report(report, require_verified=True)

    assert "stage_worker_package_not_verified" in errors


def test_cli_writes_public_safe_manifest() -> None:
    out = _tmp_dir()
    plan_path = _write(out / "plan.json", _plan())
    code = package.main(["--output-dir", str(out / "pkg"), "--stage-runtime-plan-report", str(plan_path)])

    assert code == 0
    payload = json.loads((out / "pkg" / "glm52_kaggle_stage_worker_package.json").read_text(encoding="utf-8"))
    assert payload["public_artifact_safe"] is True
    assert "glm52_stage_worker_package_is_not_runtime_success" in payload["blockers"]
    assert check.validate_report(payload) == []


def test_rendered_kernel_requires_coordinator_hash_before_runtime_attempt() -> None:
    out = _tmp_dir()
    plan_path = _write(out / "plan.json", _plan())
    args = package.parse_args(["--output-dir", str(out / "pkg"), "--stage-runtime-plan-report", str(plan_path)])
    manifest = package.build_report(args)
    first = manifest["packages"][0]
    kernel = Path(first["kernel_path"])
    report_path = kernel.parent / "glm52_kaggle_stage_runtime_report.json"
    env = {
        **os.environ,
        "CT_GLM52_STAGE_REPORT_PATH": str(report_path),
    }

    completed = subprocess.run(["python", str(kernel)], check=False, env=env, text=True, capture_output=True)
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert completed.returncode == 0
    assert payload["schema"] == runtime_check.STAGE_SCHEMA
    assert payload["ok"] is False
    assert "glm52_stage_runtime_coordinator_request_hash_missing" in payload["blockers"]
    errors = runtime_check.validate_report(payload, require_verified=True)
    assert "stage_runtime_not_verified" in errors
    assert "live_run_not_performed" in errors


def test_rendered_kernel_can_bind_public_safe_request_hash_for_kaggle_push() -> None:
    out = _tmp_dir()
    plan_path = _write(out / "plan.json", _plan())
    request_hash = "sha256:" + ("a" * 64)
    args = package.parse_args([
        "--output-dir",
        str(out / "pkg"),
        "--stage-runtime-plan-report",
        str(plan_path),
        "--coordinator-request-id-hash",
        request_hash,
    ])

    manifest = package.build_report(args)
    first = manifest["packages"][0]
    kernel_source = Path(first["kernel_path"]).read_text(encoding="utf-8")

    assert manifest["coordinator_request_id_hash_bound"] is True
    assert manifest["coordinator_request_id_hash"] == request_hash
    assert first["coordinator_request_id_hash_bound"] is True
    assert f'DEFAULT_COORDINATOR_REQUEST_HASH = "{request_hash}"' in kernel_source
    assert "glm52_stage_worker_coordinator_request_hash_not_bound" not in manifest["blockers"]
    assert check.validate_report(manifest) == []
