#!/usr/bin/env python3
"""Run one isolated Qwen2.5-7B GSM8K benchmark on Kaggle T4x2."""

from __future__ import annotations

import argparse
import csv
import io
import json
import secrets
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

from crowdtensor.qwen15b_training import sha256_file, stable_hash
from crowdtensor.qwen7b_gsm8k_showcase import (
    DATASET_ID,
    DATASET_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    PRIVATE_BENCHMARK_SCHEMA,
    PRIVATE_TRAIN_SCHEMA,
)
from scripts.kaggle_gpu_token_weekly_quota_probe import clean_env
from scripts.training_cuda_kaggle_common import (
    extract_kernel_ref,
    public_safety_errors,
    push_accepted,
    run_command,
    safe_slug,
    utc_now,
)
from scripts.training_qwen15b_elastic_live_probe import (
    _delete_refs,
    _public_step,
    _wait_pair,
)
from scripts.training_qwen15b_four_gpu_probe import (
    _credential_sections,
    _load,
    preflight_accounts,
)
from scripts.training_qwen7b_gsm8k_benchmark_package import (
    WORKER_REPORT,
    build_package,
)


SCHEMA = "crowdtensor_qwen7b_gsm8k_benchmark_live_probe_v1"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _dataset_delete_ok(step: dict[str, Any]) -> bool:
    text = str(step.get("output_tail") or "").lower()
    return bool(
        step.get("ok") is True
        or "not found" in text
        or "does not exist" in text
        or "404" in text
    )


def _materialize_standard_peft_adapter(source: Path, destination: Path) -> None:
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
            raise RuntimeError("qwen7b_benchmark_adapter_archive_invalid")
        config = json.loads(archive.read("adapter_config.json"))
        if (
            config.get("base_model_name_or_path") != MODEL_ID
            or config.get("revision") != MODEL_REVISION
        ):
            raise RuntimeError("qwen7b_benchmark_adapter_identity_invalid")
        destination.mkdir(parents=True, exist_ok=True)
        for name in sorted(required):
            (destination / name).write_bytes(archive.read(name))


def _wait_dataset_attachment(
    *,
    dataset_ref: str,
    expected_names: set[str],
    env: dict[str, str],
    timeout_seconds: float = 180.0,
    poll_seconds: float = 5.0,
) -> dict[str, Any]:
    started = time.monotonic()
    observations: list[dict[str, Any]] = []
    ready = False
    while time.monotonic() - started < timeout_seconds:
        status = run_command(
            ["kaggle", "datasets", "status", dataset_ref, "--format", "json"],
            env=env,
            timeout=45.0,
        )
        files = run_command(
            [
                "kaggle",
                "datasets",
                "files",
                dataset_ref,
                "--page-size",
                "200",
                "-v",
            ],
            env=env,
            timeout=45.0,
        )
        rows = []
        if files.get("ok") is True:
            rows = list(
                csv.DictReader(io.StringIO(str(files.get("output_tail") or "")))
            )
        names = [
            str(row.get("name") or row.get("ref") or next(iter(row.values()), ""))
            for row in rows
        ]
        counts = {
            name: sum(Path(value).name == name for value in names)
            for name in sorted(expected_names)
        }
        status_ready = bool(
            status.get("ok") is True
            and "ready" in str(status.get("output_tail") or "").lower()
        )
        files_ready = all(value == 1 for value in counts.values())
        observations.append(
            {
                "status_ready": status_ready,
                "files_query_ok": files.get("ok") is True,
                "file_count": len(names),
                "expected_file_count": len(expected_names),
                "all_expected_files_unique": files_ready,
            }
        )
        if status_ready and files_ready:
            ready = True
            break
        time.sleep(poll_seconds)
    return {
        "ready": ready,
        "attempt_count": len(observations),
        "observations": observations[-12:],
        "stabilization_wait_seconds": 10.0 if ready else 0.0,
        "dataset_ref_public": False,
    }


def run_benchmark_live(
    output_dir: str | Path,
    *,
    token_files: list[str],
    raw_token_file: str = "",
    raw_token_username: str = "",
    benchmark_payload_path: str | Path,
    train_payload_path: str | Path,
    adapter_path: str | Path | None = None,
    mode: str = "base",
    allocation_timeout_seconds: float = 7200.0,
    status_timeout_seconds: float = 45.0,
    output_timeout_seconds: float = 600.0,
    delete_timeout_seconds: float = 180.0,
    poll_interval_seconds: float = 15.0,
    kernel_timeout_seconds: int = 7200,
    max_new_tokens: int = 256,
    batch_size: int = 8,
) -> dict[str, Any]:
    if mode not in {"base", "adapter", "both"}:
        raise ValueError("Qwen7B benchmark mode invalid")
    benchmark_path = Path(benchmark_payload_path).resolve()
    train_path = Path(train_payload_path).resolve()
    adapter = Path(adapter_path).resolve() if adapter_path else None
    benchmark = _load(benchmark_path)
    training = _load(train_path)
    if (
        benchmark.get("schema") != PRIVATE_BENCHMARK_SCHEMA
        or training.get("schema") != PRIVATE_TRAIN_SCHEMA
        or benchmark.get("model_id") != MODEL_ID
        or training.get("model_id") != MODEL_ID
        or benchmark.get("model_revision") != MODEL_REVISION
        or training.get("model_revision") != MODEL_REVISION
        or benchmark.get("dataset_id") != DATASET_ID
        or training.get("dataset_id") != DATASET_ID
        or benchmark.get("dataset_revision") != DATASET_REVISION
        or training.get("dataset_revision") != DATASET_REVISION
        or len(benchmark.get("examples") or []) != 128
        or len(training.get("validation") or []) < 8
    ):
        raise ValueError("Qwen7B benchmark private input contract invalid")
    if mode in {"adapter", "both"} and (adapter is None or not adapter.is_file()):
        raise ValueError("Qwen7B benchmark adapter is required")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    private = output / ".private-runtime"
    if private.exists():
        shutil.rmtree(private)
    private.mkdir(parents=True)
    private.chmod(0o700)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": False,
        "live_run_performed": True,
        "mode": mode,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "benchmark_example_count": 128,
        "benchmark_payload_hash": sha256_file(benchmark_path),
        "validation_payload_hash": stable_hash(training.get("validation") or []),
        "adapter_archive_hash": sha256_file(adapter) if adapter else "",
        "adapter_input_materialized": False,
        "blockers": [],
        "started_at": utc_now(),
        "account_preflight": [],
        "worker": {},
        "cleanup": {
            "kernel_deleted": False,
            "private_dataset_deleted": False,
            "private_runtime_removed": False,
            "live_resources_left_running": True,
        },
        "raw_text_public": False,
        "token_ids_public": False,
        "generated_text_public": False,
        "gold_answers_public": False,
        "adapter_tensor_values_public": False,
        "credentials_public": False,
        "credential_paths_public": False,
        "kernel_refs_public": False,
        "dataset_refs_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    selected_env: dict[str, str] = {}
    dataset_ref = ""
    kernel_ref = ""
    try:
        sections = _credential_sections(
            list(token_files),
            raw_token_file=raw_token_file,
            raw_token_username=raw_token_username,
        )
        preflight, candidates = preflight_accounts(sections)
        report["account_preflight"] = preflight
        report["eligible_account_count"] = len(candidates)
        if not candidates:
            raise RuntimeError("qwen7b_benchmark_kaggle_t4x2_unavailable")
        selected = candidates[0]
        selected_env = dict(selected["env_values"])
        owner = str(selected["owner"])
        suffix = f"{str(int(time.time()))[-8:]}-{secrets.token_hex(2)}"
        dataset_slug = safe_slug(f"ct-q7b-benchmark-input-{suffix}")
        dataset_ref = f"{safe_slug(owner)}/{dataset_slug}"
        kernel_slug = safe_slug(f"ct-q7b-benchmark-{mode}-{suffix}")
        kernel_ref = f"{safe_slug(owner)}/{kernel_slug}"
        report["selected_account"] = {
            "owner_hash": str(selected["owner_hash"]),
            "effective_remaining_seconds": float(selected["effective_remaining"]),
            "credentials_public": False,
        }
        with tempfile.TemporaryDirectory(
            prefix="ct-qwen7b-benchmark-kaggle-config-"
        ) as config_dir:
            env = clean_env(selected_env, config_dir=Path(config_dir))
            dataset_dir = private / "private-dataset"
            dataset_dir.mkdir()
            shutil.copyfile(
                benchmark_path,
                dataset_dir / "qwen7b_gsm8k_benchmark_private.json",
            )
            _write(
                dataset_dir / "qwen7b_gsm8k_validation_private.json",
                {
                    "schema": "crowdtensor_qwen7b_gsm8k_private_validation_v1",
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "validation": training["validation"],
                },
            )
            if adapter is not None:
                _materialize_standard_peft_adapter(
                    adapter,
                    dataset_dir,
                )
                report["adapter_input_materialized"] = True
            input_names = [
                "qwen7b_gsm8k_benchmark_private.json",
                "qwen7b_gsm8k_validation_private.json",
            ]
            if adapter is not None:
                input_names.extend(
                    ["adapter_config.json", "adapter_model.safetensors"]
                )
            expected_input_hashes = {
                name: sha256_file(dataset_dir / name) for name in input_names
            }
            _write(
                dataset_dir / "dataset-metadata.json",
                {
                    "title": dataset_slug.replace("-", " ").title(),
                    "id": dataset_ref,
                    "licenses": [{"name": "other"}],
                },
            )
            create = run_command(
                ["kaggle", "datasets", "create", "-p", str(dataset_dir), "-r", "zip"],
                env=env,
                timeout=900.0,
            )
            report["dataset_create"] = _public_step(create, role="benchmark_input")
            if create.get("ok") is not True:
                raise RuntimeError("qwen7b_benchmark_private_dataset_create_failed")
            attachment = _wait_dataset_attachment(
                dataset_ref=dataset_ref,
                expected_names=set(expected_input_hashes),
                env=env,
            )
            report["dataset_attachment_preflight"] = attachment
            if attachment.get("ready") is not True:
                raise RuntimeError("qwen7b_benchmark_dataset_attachment_not_ready")
            time.sleep(float(attachment["stabilization_wait_seconds"]))

            package = build_package(
                private / "package",
                owner=owner,
                slug=kernel_slug,
                dataset_ref=dataset_ref,
                mode=mode,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                expected_input_hashes=expected_input_hashes,
            )
            push = run_command(
                [
                    "kaggle",
                    "kernels",
                    "push",
                    "-p",
                    str(package["package_dir"]),
                    "-t",
                    str(int(kernel_timeout_seconds)),
                    "--accelerator",
                    "NvidiaTeslaT4",
                ],
                env=env,
                timeout=600.0,
            )
            pushed = push_accepted(push)
            kernel_ref = extract_kernel_ref(
                str(push.get("output_tail") or ""), kernel_ref
            )
            report["kernel_push"] = {
                **_public_step(push, role="benchmark"),
                "accepted": pushed,
                "kernel_ref_hash": stable_hash({"ref": kernel_ref}),
            }
            if not pushed:
                raise RuntimeError("qwen7b_benchmark_kernel_push_failed")
            terminal, observations, maximum_running = _wait_pair(
                [kernel_ref],
                env=env,
                timeout=allocation_timeout_seconds,
                status_timeout=status_timeout_seconds,
                poll_interval=poll_interval_seconds,
            )
            report["status_observations"] = observations
            report["maximum_running_kernel_count"] = maximum_running
            report["terminal_state"] = terminal[kernel_ref]
            private_output = private / "kernel-output"
            download = run_command(
                [
                    "kaggle",
                    "kernels",
                    "output",
                    kernel_ref,
                    "-p",
                    str(private_output),
                    "--force",
                    "--file-pattern",
                    WORKER_REPORT,
                ],
                env=env,
                timeout=output_timeout_seconds,
            )
            report["kernel_output"] = _public_step(download, role="benchmark")
            worker = _load(private_output / WORKER_REPORT)
            if worker:
                _write(output / WORKER_REPORT, worker)
            report["worker"] = worker
            report["ok"] = bool(
                terminal[kernel_ref] == "complete"
                and maximum_running == 1
                and worker.get("ok") is True
                and worker.get("mode") == mode
                and worker.get("model_id") == MODEL_ID
                and worker.get("model_revision") == MODEL_REVISION
                and int(worker.get("benchmark_example_count") or 0) == 128
                and worker.get("public_artifact_safe") is True
            )
            if not report["ok"]:
                raise RuntimeError("qwen7b_benchmark_worker_acceptance_incomplete")
    except BaseException as exc:
        report["blockers"].append(str(exc).split(":", 1)[0][:180])
        report["error_class"] = type(exc).__name__
    finally:
        if selected_env:
            with tempfile.TemporaryDirectory(
                prefix="ct-qwen7b-benchmark-cleanup-config-"
            ) as config_dir:
                env = clean_env(selected_env, config_dir=Path(config_dir))
                if kernel_ref:
                    deletions, deleted = _delete_refs(
                        [kernel_ref],
                        env=env,
                        timeout=delete_timeout_seconds,
                        role_by_ref={kernel_ref: "benchmark"},
                    )
                    report["kernel_deletions"] = deletions
                    report["cleanup"]["kernel_deleted"] = deleted
                else:
                    report["cleanup"]["kernel_deleted"] = True
                if dataset_ref:
                    delete_dataset = run_command(
                        ["kaggle", "datasets", "delete", dataset_ref, "-y"],
                        env=env,
                        timeout=delete_timeout_seconds,
                    )
                    report["dataset_delete"] = _public_step(
                        delete_dataset, role="benchmark_input"
                    )
                    report["cleanup"]["private_dataset_deleted"] = (
                        _dataset_delete_ok(delete_dataset)
                    )
                else:
                    report["cleanup"]["private_dataset_deleted"] = True
        else:
            report["cleanup"]["kernel_deleted"] = not kernel_ref
            report["cleanup"]["private_dataset_deleted"] = not dataset_ref
        shutil.rmtree(private, ignore_errors=True)
        report["cleanup"]["private_runtime_removed"] = not private.exists()
        report["cleanup"]["live_resources_left_running"] = not all(
            report["cleanup"].get(key) is True
            for key in (
                "kernel_deleted",
                "private_dataset_deleted",
                "private_runtime_removed",
            )
        )
        if report["cleanup"]["live_resources_left_running"]:
            report["ok"] = False
            report["blockers"].append("qwen7b_benchmark_cleanup_incomplete")
        report["blockers"] = sorted(set(report["blockers"]))
        report["finished_at"] = utc_now()
        safety = public_safety_errors(report)
        report["public_artifact_safe"] = not safety
        if safety:
            report["ok"] = False
            report["public_safety_error_count"] = len(safety)
        report["content_hash"] = stable_hash(
            {key: value for key, value in report.items() if key != "content_hash"}
        )
        _write(output / "training_qwen7b_gsm8k_benchmark_live_probe.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--token-file", action="append", default=[])
    parser.add_argument("--raw-token-file", default="")
    parser.add_argument("--raw-token-username", default="")
    parser.add_argument("--benchmark-payload", required=True)
    parser.add_argument("--train-payload", required=True)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--mode", choices=["base", "adapter", "both"], default="base")
    parser.add_argument("--allocation-timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--kernel-timeout-seconds", type=int, default=7200)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run_benchmark_live(
        args.output_dir,
        token_files=list(args.token_file),
        raw_token_file=args.raw_token_file,
        raw_token_username=args.raw_token_username,
        benchmark_payload_path=args.benchmark_payload,
        train_payload_path=args.train_payload,
        adapter_path=args.adapter or None,
        mode=args.mode,
        allocation_timeout_seconds=args.allocation_timeout_seconds,
        kernel_timeout_seconds=args.kernel_timeout_seconds,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "ok": report["ok"],
                    "mode": report["mode"],
                    "blockers": report["blockers"],
                    "cleanup": report["cleanup"],
                },
                sort_keys=True,
            )
        )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
