#!/usr/bin/env python3
"""CI-safe contract check for the real Internet Swarm Inference Beta automation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from scripts import real_llm_internet_beta_pack as pack  # noqa: E402


SCHEMA = "real_llm_internet_beta_check_v1"


def completed(payload: dict[str, Any], *, returncode: int = 0, stdout_suffix: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["cmd"],
        returncode=returncode,
        stdout=json.dumps(payload) + "\n" + stdout_suffix,
        stderr="",
    )


def option_value(command: list[str], option: str, default: str = "") -> str:
    if option not in command:
        return default
    index = command.index(option) + 1
    if index >= len(command):
        return default
    return command[index]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_alpha_package(output_dir: Path) -> None:
    live_dir = output_dir / "package-live-rc"
    runtime_dir = live_dir / "remote-real-llm-runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "operator.private.env").write_text(
        "export CROWDTENSOR_OBSERVER_TOKEN='observer-secret'\n"
        "export CROWDTENSOR_ADMIN_TOKEN='admin-secret'\n",
        encoding="utf-8",
    )
    for stage in pack.STAGES:
        upload = live_dir / f"kaggle-upload-real-llm-{stage}"
        upload.mkdir(parents=True, exist_ok=True)
        (upload / "miner.private.env").write_text(f"export CROWDTENSOR_MINER_TOKEN='{stage}-secret'\n", encoding="utf-8")
    (live_dir / "start_coordinator.sh").write_text("#!/usr/bin/env bash\nsleep 60\n", encoding="utf-8")
    write_json(live_dir / "real_llm_live_rc.json", {
        "schema": "real_llm_live_rc_v1",
        "ok": True,
        "mode": "kaggle-generated",
        "diagnosis_codes": ["real_llm_live_rc_prepare_ready", "kaggle_real_llm_stage_upload_packages_ready"],
    })
    write_json(output_dir / "real_llm_internet_alpha.json", {
        "schema": "real_llm_internet_alpha_v1",
        "ok": True,
        "mode": "package",
        "diagnosis_codes": ["real_llm_internet_alpha_package_ready"],
    })
    (output_dir / "real_llm_internet_alpha.md").write_text("# Alpha\n", encoding="utf-8")


def write_external_alpha(output_dir: Path) -> None:
    live_dir = output_dir / "live-rc"
    runtime_dir = live_dir / "remote-real-llm-runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    write_json(live_dir / "real_llm_live_rc.json", {
        "schema": "real_llm_live_rc_v1",
        "ok": True,
        "mode": "external-existing",
        "diagnosis_codes": [
            "external_runtime_verified",
            "kaggle_real_llm_stage0_seen",
            "kaggle_real_llm_stage1_seen",
            "kaggle_real_llm_sharded_ready",
            "decoded_tokens_match",
            "distinct_stage_miners",
            "stage_assignment_valid",
        ],
    })
    write_json(runtime_dir / "remote_real_llm_sharded_beta.json", {
        "schema": "remote_real_llm_sharded_beta_v1",
        "ok": True,
        "diagnosis_codes": ["remote_real_llm_sharded_ready"],
    })
    write_json(output_dir / "real_llm_internet_alpha.json", {
        "schema": "real_llm_internet_alpha_v1",
        "ok": True,
        "mode": "external-existing",
        "diagnosis_codes": [
            "real_llm_internet_alpha_ready",
            "external_runtime_verified",
            "kaggle_real_llm_stage0_seen",
            "kaggle_real_llm_stage1_seen",
            "kaggle_real_llm_sharded_ready",
            "decoded_tokens_match",
            "distinct_stage_miners",
            "stage_assignment_valid",
        ],
    })
    (output_dir / "real_llm_internet_alpha.md").write_text("# External Alpha\n", encoding="utf-8")


def output_scope_errors(label: str, payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    output_request = payload.get("output_request") if isinstance(payload.get("output_request"), dict) else {}
    prompt_scope = payload.get("prompt_scope") if isinstance(payload.get("prompt_scope"), dict) else {}
    answer_scope = payload.get("answer_scope") if isinstance(payload.get("answer_scope"), dict) else {}
    shareable = payload.get("shareable_summary") if isinstance(payload.get("shareable_summary"), dict) else {}
    if output_request.get("include_output") is not False:
        errors.append(f"{label}:output_request_include_output")
    if output_request.get("raw_prompt_public") is not False:
        errors.append(f"{label}:output_request_raw_prompt_public")
    if output_request.get("raw_generated_text_public") is not False:
        errors.append(f"{label}:output_request_raw_generated_text_public")
    if output_request.get("generated_token_ids_public") is not False:
        errors.append(f"{label}:output_request_generated_token_ids_public")
    if output_request.get("public_artifact_safe") is not True:
        errors.append(f"{label}:output_request_public_artifact_safe")
    if prompt_scope.get("source") != "built-in-default-prompts":
        errors.append(f"{label}:prompt_scope_source")
    if not isinstance(prompt_scope.get("prompt_count"), int) or prompt_scope.get("prompt_count") < 1:
        errors.append(f"{label}:prompt_scope_count")
    if prompt_scope.get("inline_prompt_text") is not False:
        errors.append(f"{label}:prompt_scope_inline_prompt_text")
    if prompt_scope.get("terminal_next_commands_local_private") is not False:
        errors.append(f"{label}:prompt_scope_terminal_next_commands_local_private")
    if prompt_scope.get("terminal_logs_local_private") is not False:
        errors.append(f"{label}:prompt_scope_terminal_logs_local_private")
    if prompt_scope.get("saved_artifacts_prompt_placeholders") is not True:
        errors.append(f"{label}:prompt_scope_saved_artifacts_prompt_placeholders")
    if prompt_scope.get("saved_artifacts_public_safe") is not True:
        errors.append(f"{label}:prompt_scope_saved_artifacts_public_safe")
    if prompt_scope.get("prefer_prompt_file_or_stdin_for_shareable_logs") is not False:
        errors.append(f"{label}:prompt_scope_prefer_prompt_file_or_stdin")
    if prompt_scope.get("prompt_file_path_public") is not False:
        errors.append(f"{label}:prompt_scope_prompt_file_path_public")
    if prompt_scope.get("raw_prompt_public") is not False:
        errors.append(f"{label}:prompt_scope_raw_prompt_public")
    if prompt_scope.get("public_artifact_safe") is not True:
        errors.append(f"{label}:prompt_scope_public_artifact_safe")
    if answer_scope.get("scope_state") != "no-local-answer":
        errors.append(f"{label}:answer_scope_state")
    if answer_scope.get("visible_in_terminal") is not False:
        errors.append(f"{label}:answer_visible_in_terminal")
    if answer_scope.get("terminal_only") is not False:
        errors.append(f"{label}:answer_terminal_only")
    if answer_scope.get("saved_json_display") != "hash-only":
        errors.append(f"{label}:answer_saved_json_display")
    if answer_scope.get("saved_markdown_display") != "hash-only":
        errors.append(f"{label}:answer_saved_markdown_display")
    if answer_scope.get("raw_prompt_public") is not False:
        errors.append(f"{label}:answer_raw_prompt_public")
    if answer_scope.get("raw_generated_text_public") is not False:
        errors.append(f"{label}:answer_raw_generated_text_public")
    if answer_scope.get("generated_token_ids_public") is not False:
        errors.append(f"{label}:answer_generated_token_ids_public")
    if answer_scope.get("public_artifact_safe") is not True:
        errors.append(f"{label}:answer_public_artifact_safe")
    if shareable.get("saved_artifacts_public_safe") is not True:
        errors.append(f"{label}:shareable_saved_artifacts")
    if shareable.get("raw_prompt_public") is not False:
        errors.append(f"{label}:shareable_raw_prompt_public")
    if shareable.get("raw_generated_text_public") is not False:
        errors.append(f"{label}:shareable_raw_generated_text_public")
    if shareable.get("generated_token_ids_public") is not False:
        errors.append(f"{label}:shareable_generated_token_ids_public")
    if shareable.get("answer_scope_state") != "no-local-answer":
        errors.append(f"{label}:shareable_answer_scope_state")
    if shareable.get("local_answer_terminal_only") is not False:
        errors.append(f"{label}:shareable_local_answer_terminal_only")
    if shareable.get("public_artifact_safe") is not True:
        errors.append(f"{label}:shareable_public_artifact_safe")
    return errors


def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
    joined = " ".join(command)
    if "real_llm_internet_alpha_pack.py" in joined and option_value(command, "--mode") == "package":
        output_dir = Path(option_value(command, "--output-dir"))
        write_alpha_package(output_dir)
        return completed({
            "schema": "real_llm_internet_alpha_v1",
            "ok": True,
            "mode": "package",
            "diagnosis_codes": ["real_llm_internet_alpha_package_ready"],
        })
    if "kaggle_real_llm_live_package.py" in joined:
        output_dir = Path(option_value(command, "--output-dir"))
        output_dir.mkdir(parents=True, exist_ok=True)
        owner = option_value(command, "--owner", "xuyuhaosuyi")
        prefix = option_value(command, "--kernel-slug-prefix", "crowdtensor-real-llm-beta-test")
        failure_mode = option_value(command, "--failure-mode", pack.FAILURE_NONE)
        stages = []
        if failure_mode == pack.FAILURE_KILL_STAGE0_AFTER_CLAIM:
            entries = [("stage0", "victim"), ("stage0", "rescue"), ("stage1", "normal")]
        elif failure_mode == pack.FAILURE_KILL_STAGE1_AFTER_CLAIM:
            entries = [("stage0", "normal"), ("stage1", "victim"), ("stage1", "rescue")]
        else:
            entries = [(stage, "normal") for stage in pack.STAGES]
        for stage, role in entries:
            key = stage if role == "normal" else f"{stage}-{role}"
            kernel_dir = output_dir / "kernels" / key
            kernel_dir.mkdir(parents=True, exist_ok=True)
            stages.append({
                "stage": stage,
                "role": role,
                "key": key,
                "kernel_ref": f"{owner}/{prefix}-{key}",
                "miner_id": f"swarm-beta-live-{key}",
                "inline_kernel_payload": True,
                "hf_runtime_enabled": True,
                "real_llm_stage_role_present": True,
            })
        write_json(output_dir / "kaggle_real_llm_live_package.json", {
            "schema": "kaggle_real_llm_live_package_v1",
            "ok": True,
            "dataset_ref": f"{owner}/{prefix}-dataset",
            "hf_model_id": "sshleifer/tiny-gpt2",
            "failure_mode": failure_mode,
            "max_tasks": int(option_value(command, "--max-tasks", "2")),
            "max_request_attempts": int(option_value(command, "--max-request-attempts", "240")),
            "stages": stages,
            "diagnosis_codes": ["kaggle_real_llm_live_package_ready"],
        })
        return completed(load_json(output_dir / "kaggle_real_llm_live_package.json"))
    if command[:3] == ["kaggle", "kernels", "push"]:
        stage = Path(command[-1]).name
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "Kernel version 1 successfully pushed.  Please check progress at "
                f"https://www.kaggle.com/code/xuyuhaosuyi/crowdtensor-real-llm-beta-check-{stage}\n"
            ),
            stderr="",
        )
    if command[:3] == ["kaggle", "kernels", "status"]:
        return subprocess.CompletedProcess(command, 0, stdout='has status "COMPLETE"\n', stderr="")
    if command[:3] == ["kaggle", "kernels", "output"]:
        output_dir = Path(option_value(command, "-p", tempfile.mkdtemp(prefix="crowdtensor_kaggle_output_")))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "crowdtensor_real_llm.log").write_text("fake Kaggle miner output\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="Downloaded 1 files\n", stderr="")
    if command[:3] == ["kaggle", "kernels", "delete"]:
        return subprocess.CompletedProcess(command, 0, stdout="deleted\n", stderr="")
    if "real_llm_internet_alpha_pack.py" in joined and option_value(command, "--mode") == "external-existing":
        output_dir = Path(option_value(command, "--output-dir"))
        write_external_alpha(output_dir)
        return completed(load_json(output_dir / "real_llm_internet_alpha.json"))
    raise AssertionError(command)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class FakePopen:
    pid = 4242
    returncode: int | None = None

    def __init__(self, command: list[str] | None = None, *_: object, **__: object) -> None:
        self.command = command or []

    def poll(self) -> int | None:
        return self.returncode

    def send_signal(self, _signal: int) -> None:
        self.returncode = -15

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = -15
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        if self.returncode is None:
            self.returncode = 0 if "real_llm_internet_alpha_pack.py" in " ".join(self.command) else -15
        if "real_llm_internet_alpha_pack.py" in " ".join(self.command):
            output_dir = Path(option_value(self.command, "--output-dir"))
            write_external_alpha(output_dir)
            return json.dumps(load_json(output_dir / "real_llm_internet_alpha.json")) + "\n", ""
        return "", ""


class FakeStateProbe:
    def __init__(self, *, target_stage: str, victim_miner_id: str, rescue_miner_id: str) -> None:
        self.target_stage = target_stage
        self.victim_miner_id = victim_miner_id
        self.rescue_miner_id = rescue_miner_id
        self.calls = 0
        self.requeue_seen = False

    def __call__(self, _args: argparse.Namespace, _observer_token: str, _secret_values: list[str]) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            status = "leased"
            miner_id = self.victim_miner_id
            attempt = 1
        elif self.calls == 2:
            status = "queued"
            miner_id = None
            attempt = 1
            self.requeue_seen = True
        else:
            status = "completed" if self.requeue_seen else "queued"
            miner_id = self.rescue_miner_id if self.requeue_seen else None
            attempt = 2
        return {
            "tasks": [
                {
                    "task_id": "task-live-requeue",
                    "workload_type": pack.WORKLOAD_TYPE,
                    "status": status,
                    "attempt": attempt,
                    "miner_id": miner_id,
                    "workload_metadata": {"stage_id": 0 if self.target_stage == "stage0" else 1},
                }
            ]
        }


def ready_probe(_url: str, _timeout: float, _poll: float) -> dict[str, Any]:
    return {"ok": True, "payload": {"ok": True}}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the real LLM Internet Beta automation contract without external side effects.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--port", type=int, default=9190)
    parser.add_argument("--base-port", type=int, default=9191)
    parser.add_argument("--request-count", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def build_check(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="crowdtensor_real_llm_internet_beta_check_"))
    pack_args = pack.parse_args([
        "--output-dir",
        str(output_dir),
        "--port",
        str(args.port),
        "--base-port",
        str(args.base_port),
        "--request-count",
        str(args.request_count),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--kaggle-owner",
        "xuyuhaosuyi",
        "--kernel-slug-prefix",
        "crowdtensor-real-llm-beta-check",
    ])
    report = pack.build_report(pack_args, runner=fake_runner, popen_factory=FakePopen, ready_probe=ready_probe)  # type: ignore[arg-type]
    required_codes = {
        "real_llm_internet_beta_ready",
        "real_llm_internet_alpha_ready",
        "external_runtime_verified",
        "kaggle_real_llm_stage0_seen",
        "kaggle_real_llm_stage1_seen",
        "decoded_tokens_match",
        "distinct_stage_miners",
        "stage_assignment_valid",
        "kaggle_kernels_deleted",
        "token_rotation_required",
    }
    serialized = json.dumps(report, sort_keys=True)
    forbidden = [
        "observer-secret",
        "admin-secret",
        "stage0-secret",
        "stage1-secret",
        "SOURCE_TARBALL_B64",
        "MINER_ENV_TEXT",
        "miner.private.env",
        "operator.private.env",
        "miner_registry.json",
    ]
    missing = sorted(required_codes - set(report.get("diagnosis_codes") or []))
    leaks = [item for item in forbidden if item in serialized]
    scope_errors = output_scope_errors("report", report)
    ok = bool(report.get("ok") and not missing and not leaks and not scope_errors)
    return {
        "schema": SCHEMA,
        "ok": ok,
        "output_dir": str(output_dir),
        "report_schema": report.get("schema"),
        "report_ok": report.get("ok"),
        "missing_codes": missing,
        "sensitive_leaks": leaks,
        "output_scope_errors": scope_errors,
        "diagnosis_codes": sorted(set(report.get("diagnosis_codes") or []) | (set() if ok else {"real_llm_internet_beta_check_failed"})),
        "artifacts": {
            "real_llm_internet_beta_json": pack.artifact_entry(
                output_dir / "real_llm_internet_beta.json",
                output_dir,
                kind="real_llm_internet_beta",
                schema=pack.SCHEMA,
                ok=report.get("ok"),
            ),
        },
        "limitations": [
            "This is a CI-safe fake-runner contract check; it does not create Kaggle kernels or verify external runtime.",
            "Use crowdtensor real-llm-internet-beta for the real side-effectful Kaggle automation.",
        ],
    }


def print_human(report: dict[str, Any]) -> None:
    print("CrowdTensor real LLM Internet Beta check")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")


def main() -> None:
    args = parse_args()
    report = build_check(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print_human(report)
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
