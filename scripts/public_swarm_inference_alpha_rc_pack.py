#!/usr/bin/env python3
"""Build the Public Swarm Inference Alpha release-candidate evidence pack.

This layer does not create live Kaggle resources.  It imports retained public
Alpha reports, verifies that both stage-specific live requeue proofs are
present and shareable, and emits a single release-candidate artifact.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import support_bundle  # noqa: E402


SCHEMA = "public_swarm_inference_alpha_rc_v1"
ALPHA_SCHEMA = "public_swarm_inference_alpha_v1"
SUMMARY_SCHEMA = "public_swarm_inference_alpha_live_requeue_summary_v1"
CHECK_SCHEMA = "public_swarm_inference_alpha_check_v1"
MODE_EVIDENCE_IMPORT = "evidence-import"
MODE_LOCAL_SMOKE = "local-smoke"
DEFAULT_OUTPUT_DIR = "dist/public-swarm-inference-alpha-rc"
DEFAULT_STAGE0_REPORT = (
    "dist/public-swarm-inference-alpha-live-stage0-requeue-20260527165830/"
    "public_swarm_inference_alpha.json"
)
DEFAULT_STAGE1_REPORT = (
    "dist/public-swarm-inference-alpha-live-stage1-requeue-20260527170600/"
    "public_swarm_inference_alpha.json"
)
DEFAULT_SUMMARY_REPORT = "dist/public-swarm-inference-alpha-live-requeue-summary.json"

Runner = Callable[..., subprocess.CompletedProcess[str]]

SECRET_FRAGMENTS = (
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
    "lease_token",
    "idempotency_key",
    "Bearer ",
    "hidden_state",
    "input_ids",
    "logits",
    "activation_results",
    "real_llm_sharded_result",
    "generated_text",
    "generated_token_ids",
)

PRIVATE_ARTIFACT_NAMES = {
    "operator.private.env",
    "miner.private.env",
    "miner_registry.json",
    "kernel.py",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def json_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("command emitted no JSON object")


def redact_text(text: str) -> str:
    redacted = str(text)
    for fragment in SECRET_FRAGMENTS:
        redacted = redacted.replace(fragment, "<redacted>")
    return redacted


def redact_values(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_values(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_values(item) for key, item in value.items()}
    return value


def diagnosis_codes(*payloads: dict[str, Any]) -> list[str]:
    codes: set[str] = set()
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for code in payload.get("diagnosis_codes") or []:
            if isinstance(code, str):
                codes.add(code)
        summaries = payload.get("payload_summaries") if isinstance(payload.get("payload_summaries"), dict) else {}
        for summary in summaries.values():
            if isinstance(summary, dict):
                for code in summary.get("diagnosis_codes") or []:
                    if isinstance(code, str):
                        codes.add(code)
    return sorted(codes)


def artifact_entry(path: Path, output_dir: Path, *, kind: str, schema: str = "", ok: bool | None = None) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        relative = str(path)
    entry: dict[str, Any] = {"kind": kind, "path": relative, "present": path.is_file()}
    if schema:
        entry["schema"] = schema
    if ok is not None:
        entry["ok"] = bool(ok)
    return entry


def output_request_summary() -> dict[str, Any]:
    return {
        "include_output": False,
        "raw_prompt_public": False,
        "raw_generation_public": False,
        "generation_ids_public": False,
        "local_output_display_only": False,
        "public_artifact_safe": True,
        "summary": (
            "Public Swarm Alpha RC artifacts summarize retained stage requeue "
            "readiness, cleanup state, hashes/counts from imported evidence, and "
            "safety diagnostics only. They do not include answer text."
        ),
    }


def answer_scope_summary() -> dict[str, Any]:
    return {
        "scope_state": "no-local-answer",
        "terminal_only": False,
        "visible_in_terminal": False,
        "saved_json_display": "hash-only",
        "saved_markdown_display": "hash-only",
        "json_stdout_display": "hash-only-json",
        "raw_prompt_public": False,
        "raw_generation_public": False,
        "generation_ids_public": False,
        "public_artifact_safe": True,
        "summary": (
            "This Alpha RC report is shareable retained-evidence audit output, not "
            "a local answer transcript; raw prompts, answer text, token ids, "
            "activations, leases, credentials, private env files, and raw runtime "
            "state are excluded."
        ),
    }


def shareable_summary() -> dict[str, Any]:
    return {
        "saved_artifacts_public_safe": True,
        "raw_prompt_public": False,
        "raw_generation_public": False,
        "generation_ids_public": False,
        "local_output_display_only": False,
        "answer_scope_state": "no-local-answer",
        "local_answer_terminal_only": False,
        "public_artifact_safe": True,
        "summary": (
            "Share public_swarm_inference_alpha_rc.json/md; they contain retained "
            "readiness evidence, cleanup state, hashes, counts, and diagnostics, "
            "not raw prompts or answers."
        ),
    }


def prompt_scope_summary(report: dict[str, Any] | None = None) -> dict[str, Any]:
    rc = report.get("release_candidate") if isinstance(report, dict) and isinstance(report.get("release_candidate"), dict) else {}
    prompt_count = 0
    for key in ("request_count", "prompt_count"):
        try:
            prompt_count = int(rc.get(key) or 0)
        except (TypeError, ValueError):
            prompt_count = 0
        if prompt_count:
            break
    return {
        "source": "imported-or-built-in-validation-prompts",
        "prompt_count": prompt_count,
        "inline_prompt_text": False,
        "terminal_next_commands_local_private": False,
        "terminal_logs_local_private": False,
        "saved_artifacts_prompt_placeholders": True,
        "saved_artifacts_public_safe": True,
        "prefer_prompt_file_or_stdin_for_shareable_logs": False,
        "prompt_file_path_public": False,
        "raw_prompt_public": False,
        "public_artifact_safe": True,
        "summary": (
            "Public Swarm Alpha RC imports retained validation evidence; public artifacts "
            "record prompt source/count only and exclude raw prompt text."
        ),
    }


def prompt_scope_text(prompt_scope: dict[str, Any]) -> str:
    return (
        f"source={prompt_scope.get('source') or 'unknown'} "
        f"count={prompt_scope.get('prompt_count')} "
        f"inline_prompt_text={bool(prompt_scope.get('inline_prompt_text'))} "
        f"terminal_next_commands_local_private={bool(prompt_scope.get('terminal_next_commands_local_private'))} "
        f"saved_artifacts_prompt_placeholders={bool(prompt_scope.get('saved_artifacts_prompt_placeholders'))} "
        f"prompt_file_path_public={bool(prompt_scope.get('prompt_file_path_public'))} "
        f"raw_prompt_public={bool(prompt_scope.get('raw_prompt_public'))} "
        f"public_artifact_safe={bool(prompt_scope.get('public_artifact_safe'))}"
    )


def prompt_scope_note(prompt_scope: dict[str, Any]) -> str:
    return str(
        prompt_scope.get("summary")
        or "Public artifacts record prompt source/count only and exclude raw prompt text."
    )


def answer_scope_note(answer_scope: dict[str, Any]) -> str:
    return str(
        answer_scope.get("summary")
        or "Public artifacts contain no local answer transcript or raw generated text."
    )


def shell_command(parts: list[Any]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts if str(part))


def command_entry(
    label: str,
    command: list[Any],
    *,
    reason: str = "",
    requires_private_credentials: bool = False,
    side_effectful: bool = False,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "label": label,
        "command": [str(part) for part in command],
        "command_line": shell_command(command),
        "public_artifact_safe": True,
    }
    if reason:
        entry["reason"] = reason
    if requires_private_credentials:
        entry["requires_private_credentials"] = True
        entry["credential_note"] = (
            "Use local private operator/Kaggle credentials when running this command; "
            "credential values are intentionally excluded from public artifacts."
        )
    if side_effectful:
        entry["side_effectful"] = True
    return entry


def artifact_command(output_dir: Path, filename: str, *, lines: str = "1,220p") -> list[str]:
    return ["sed", "-n", lines, str(output_dir / filename)]


def artifact_summary(output_dir: Path) -> dict[str, Any]:
    paths = {
        "inspect_first": output_dir / "public_swarm_inference_alpha_rc.md",
        "summary_json": output_dir / "public_swarm_inference_alpha_rc.json",
        "summary_markdown": output_dir / "public_swarm_inference_alpha_rc.md",
        "support_bundle": output_dir / "support_bundle.json",
    }
    present = sum(1 for path in paths.values() if path.is_file())
    return {
        **{name: str(path) for name, path in paths.items()},
        "artifact_count": len(paths),
        "present_artifact_count": present,
        "shareable_paths": [
            str(paths["summary_json"]),
            str(paths["summary_markdown"]),
            str(paths["support_bundle"]),
        ],
        "public_artifact_safe": True,
    }


def public_swarm_alpha_rc_command(args: argparse.Namespace, output_dir: Path, mode: str) -> list[Any]:
    return [
        "crowdtensor",
        "public-swarm-alpha-rc",
        "--mode",
        mode,
        "--output-dir",
        str(output_dir),
        "--stage0-report",
        getattr(args, "stage0_report", DEFAULT_STAGE0_REPORT),
        "--stage1-report",
        getattr(args, "stage1_report", DEFAULT_STAGE1_REPORT),
        "--summary-report",
        getattr(args, "summary_report", DEFAULT_SUMMARY_REPORT),
        "--request-count",
        str(getattr(args, "request_count", 2)),
        "--timeout-seconds",
        str(getattr(args, "timeout_seconds", 120.0)),
    ]


def find_private_artifacts(root: Path) -> list[str]:
    if not root.exists():
        return []
    matches: list[str] = []
    for child in root.rglob("*"):
        if child.name in PRIVATE_ARTIFACT_NAMES or child.suffix in {".tar", ".gz", ".tgz"}:
            try:
                matches.append(child.relative_to(root).as_posix())
            except ValueError:
                matches.append(str(child))
    return sorted(matches)


def live_requeue_summary(payload: dict[str, Any]) -> dict[str, Any]:
    session = payload.get("session") if isinstance(payload.get("session"), dict) else {}
    live_summary = session.get("live_summary") if isinstance(session.get("live_summary"), dict) else {}
    summary = live_summary.get("live_requeue_summary")
    if isinstance(summary, dict):
        return summary
    payload_summaries = payload.get("payload_summaries") if isinstance(payload.get("payload_summaries"), dict) else {}
    live_payload = payload_summaries.get("live_kaggle") if isinstance(payload_summaries.get("live_kaggle"), dict) else {}
    summary = live_payload.get("live_requeue_summary")
    if isinstance(summary, dict):
        return summary
    beta = live_payload.get("beta_summary") if isinstance(live_payload.get("beta_summary"), dict) else {}
    summary = beta.get("live_requeue_summary")
    return summary if isinstance(summary, dict) else {}


def validate_stage_report(path: Path, payload: dict[str, Any], *, stage: str) -> dict[str, Any]:
    expected_failure = f"kill-{stage}-after-claim"
    expected_code = f"live_{stage}_requeue_ready"
    codes = set(diagnosis_codes(payload))
    session = payload.get("session") if isinstance(payload.get("session"), dict) else {}
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    cleanup = payload.get("artifact_cleanup") if isinstance(payload.get("artifact_cleanup"), dict) else {}
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    private_artifacts = find_private_artifacts(path.parent) if path.parent.exists() else []
    summary = live_requeue_summary(payload)
    required_codes = {
        "public_swarm_inference_alpha_ready",
        "public_swarm_session_ready",
        "public_swarm_live_requeue_ready",
        "public_swarm_live_kaggle_ready",
        "external_stage_requeue_ready",
        "external_runtime_verified",
        "kaggle_kernels_deleted",
        "decoded_tokens_match",
        "distinct_stage_miners",
        "stage_assignment_valid",
        "token_rotation_required",
        expected_code,
    }
    missing_codes = sorted(required_codes - codes)
    failed_checks: list[str] = []
    if not path.is_file():
        failed_checks.append("report_missing")
    if payload.get("schema") != ALPHA_SCHEMA:
        failed_checks.append("schema_mismatch")
    if payload.get("ok") is not True:
        failed_checks.append("report_not_ok")
    if payload.get("mode") != "live-kaggle":
        failed_checks.append("mode_not_live_kaggle")
    if payload.get("failure_mode") != expected_failure:
        failed_checks.append("failure_mode_mismatch")
    if missing_codes:
        failed_checks.append("missing_readiness_codes")
    if session.get("live_external_runtime_verified") is not True:
        failed_checks.append("external_runtime_not_verified")
    if session.get("live_stage_requeue_verified") is not True:
        failed_checks.append("live_requeue_not_verified")
    if session.get("live_kaggle_kernels_deleted") is not True:
        failed_checks.append("kaggle_kernels_not_deleted")
    if safety.get("cpu_only") is not True:
        failed_checks.append("cpu_only_missing")
    if safety.get("read_only_workload") != "real_llm_sharded_infer":
        failed_checks.append("read_only_workload_missing")
    for key in ["not_production", "not_p2p", "not_large_model_serving", "not_public_prompt_serving"]:
        if safety.get(key) is not True:
            failed_checks.append(f"{key}_missing")
    if cleanup.get("child_artifacts_pruned") is not True:
        failed_checks.append("child_artifacts_not_pruned")
    for artifact_name in ["local_requeue_json", "live_swarm_beta_json", "live_support_bundle_json"]:
        artifact = artifacts.get(artifact_name) if isinstance(artifacts.get(artifact_name), dict) else {}
        if artifact.get("present") not in {False, None}:
            failed_checks.append(f"{artifact_name}_still_present")
    if private_artifacts:
        failed_checks.append("private_artifacts_present")
    if summary.get("target_stage") != stage:
        failed_checks.append("target_stage_mismatch")
    for key in ["claim_observed", "victim_kernel_deleted", "rescued_result"]:
        if summary.get(key) is not True:
            failed_checks.append(f"requeue_{key}_missing")
    if not summary.get("lease_expired"):
        failed_checks.append("requeue_lease_timeout_missing")
    if summary.get("victim_result_accepted") is not False:
        failed_checks.append("victim_result_not_rejected")
    return {
        "stage": stage,
        "path": str(path),
        "present": path.is_file(),
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "failure_mode": payload.get("failure_mode"),
        "ready": not failed_checks,
        "missing_codes": missing_codes,
        "failed_checks": sorted(set(failed_checks)),
        "diagnosis_codes": sorted(codes),
        "live_requeue_summary": {
            "target_stage": summary.get("target_stage"),
            "claim_observed": summary.get("claim_observed"),
            "victim_kernel_deleted": summary.get("victim_kernel_deleted"),
            "lease_expired": bool(summary.get("lease_expired")),
            "rescued_result": summary.get("rescued_result"),
            "victim_result_accepted": summary.get("victim_result_accepted"),
        },
        "artifact_cleanup": {
            "child_artifacts_pruned": cleanup.get("child_artifacts_pruned"),
            "retained_public_artifacts": cleanup.get("retained_public_artifacts"),
        },
        "private_artifacts_present": bool(private_artifacts),
        "private_artifact_count": len(private_artifacts),
    }


def validate_summary_report(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    failed: list[str] = []
    if not path.is_file():
        failed.append("summary_missing")
    if payload.get("schema") != SUMMARY_SCHEMA:
        failed.append("summary_schema_mismatch")
    if payload.get("ok") is not True:
        failed.append("summary_not_ok")
    proofs = payload.get("proofs") if isinstance(payload.get("proofs"), list) else []
    stages = {proof.get("target_stage") for proof in proofs if isinstance(proof, dict)}
    for stage in ["stage0", "stage1"]:
        if stage not in stages:
            failed.append(f"{stage}_summary_missing")
    for proof in proofs:
        if not isinstance(proof, dict):
            continue
        if proof.get("ok") is not True:
            failed.append(f"{proof.get('target_stage')}_summary_not_ok")
        for key in ["claim_observed", "victim_kernel_deleted", "rescued_result"]:
            if proof.get(key) is not True:
                failed.append(f"{proof.get('target_stage')}_{key}_missing")
        if proof.get("victim_result_accepted") is not False:
            failed.append(f"{proof.get('target_stage')}_victim_not_rejected")
    return {
        "path": str(path),
        "present": path.is_file(),
        "schema": payload.get("schema"),
        "ok": payload.get("ok"),
        "ready": not failed,
        "failed_checks": sorted(set(failed)),
        "proof_count": len(proofs),
        "target_stages": sorted(stage for stage in stages if isinstance(stage, str)),
    }


def build_evidence_import_report(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    stage0_path = Path(args.stage0_report).resolve()
    stage1_path = Path(args.stage1_report).resolve()
    summary_path = Path(args.summary_report).resolve()
    stage0_payload = load_json_file(stage0_path)
    stage1_payload = load_json_file(stage1_path)
    summary_payload = load_json_file(summary_path)
    stage0 = validate_stage_report(stage0_path, stage0_payload, stage="stage0")
    stage1 = validate_stage_report(stage1_path, stage1_payload, stage="stage1")
    summary = validate_summary_report(summary_path, summary_payload)
    imported_codes = set(diagnosis_codes(stage0_payload, stage1_payload))
    codes = set(imported_codes)
    if stage0.get("ready"):
        codes.add("stage0_live_requeue_evidence_ready")
    if stage1.get("ready"):
        codes.add("stage1_live_requeue_evidence_ready")
    if stage0.get("ready") and stage1.get("ready"):
        codes.add("public_swarm_live_requeue_evidence_ready")
    if summary.get("ready"):
        codes.add("public_swarm_live_requeue_summary_ready")
    private_clear = not stage0.get("private_artifacts_present") and not stage1.get("private_artifacts_present")
    if private_clear:
        codes.add("public_swarm_alpha_private_artifacts_absent")
    ok = bool(stage0.get("ready") and stage1.get("ready") and summary.get("ready") and private_clear)
    if ok:
        codes.update({
            "public_swarm_alpha_rc_evidence_imported",
            "public_swarm_alpha_rc_safety_ready",
            "public_swarm_inference_alpha_rc_ready",
        })
    else:
        codes.add("public_swarm_inference_alpha_rc_blocked")
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "mode": MODE_EVIDENCE_IMPORT,
        "output_dir": str(output_dir),
        "release_candidate": {
            "ready": ok,
            "model_id": "sshleifer/tiny-gpt2",
            "workload_type": "real_llm_sharded_infer",
            "stage_count": 2,
            "evidence_imported": True,
            "stage0_live_requeue_ready": bool(stage0.get("ready")),
            "stage1_live_requeue_ready": bool(stage1.get("ready")),
            "summary_ready": bool(summary.get("ready")),
        },
        "imported_reports": {
            "stage0": stage0,
            "stage1": stage1,
            "summary": summary,
        },
        "diagnosis_codes": sorted(codes),
        "artifacts": {
            "public_swarm_inference_alpha_rc_json": artifact_entry(
                output_dir / "public_swarm_inference_alpha_rc.json",
                output_dir,
                kind="public_swarm_inference_alpha_rc",
                schema=SCHEMA,
                ok=ok,
            ),
            "public_swarm_inference_alpha_rc_markdown": artifact_entry(
                output_dir / "public_swarm_inference_alpha_rc.md",
                output_dir,
                kind="public_swarm_inference_alpha_rc_markdown",
            ),
            "stage0_public_swarm_report": artifact_entry(
                stage0_path,
                output_dir,
                kind="public_swarm_inference_alpha_stage0",
                schema=ALPHA_SCHEMA,
                ok=stage0_payload.get("ok") if stage0_payload else None,
            ),
            "stage1_public_swarm_report": artifact_entry(
                stage1_path,
                output_dir,
                kind="public_swarm_inference_alpha_stage1",
                schema=ALPHA_SCHEMA,
                ok=stage1_payload.get("ok") if stage1_payload else None,
            ),
            "live_requeue_summary": artifact_entry(
                summary_path,
                output_dir,
                kind="public_swarm_inference_alpha_live_requeue_summary",
                schema=SUMMARY_SCHEMA,
                ok=summary_payload.get("ok") if summary_payload else None,
            ),
        },
        "safety": {
            "cpu_only": True,
            "read_only_workload": "real_llm_sharded_infer",
            "retained_public_artifacts_only": private_clear,
            "activation_payloads_redacted": True,
            "token_values_redacted": True,
            "child_debug_artifacts_pruned": private_clear,
            "not_production": True,
            "not_p2p": True,
            "not_gpu_tpu_pooling": True,
            "not_large_model_serving": True,
            "not_public_prompt_serving": True,
        },
        "operator_action": [
            "Use evidence-import as a release-candidate audit of retained public reports; it does not create a fresh Kaggle run.",
            "Rotate tokens after every temporary public HTTP/Kaggle run.",
        ],
        "limitations": [
            "Public Swarm Inference Alpha RC imports retained evidence; it is not a fresh live proof by itself.",
            "CPU-only read-only tiny GPT split evidence; not production Swarm Inference, P2P routing, NAT traversal, GPU/TPU pooling, GGUF/llama.cpp serving, large-model serving, training, payments, or staking.",
        ],
    }


def run_json_step(name: str, command: list[str], *, runner: Runner, timeout_seconds: float) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    try:
        completed = runner(
            command,
            cwd=str(ROOT),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "ok": False,
            "returncode": None,
            "duration_seconds": round(time.monotonic() - started, 3),
            "error": "timeout",
        }, {}
    step: dict[str, Any] = {
        "name": name,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
    }
    payload: dict[str, Any] = {}
    try:
        payload = json_from_stdout(completed.stdout)
    except ValueError as exc:
        step["ok"] = False
        step["error"] = str(exc)
    else:
        step["payload_schema"] = payload.get("schema")
        step["payload_ok"] = payload.get("ok")
        step["ok"] = bool(step["ok"] and payload.get("ok"))
    if not step["ok"]:
        if completed.stdout and not payload:
            step["stdout_tail"] = redact_text(completed.stdout[-1200:])
        if completed.stderr:
            step["stderr_tail"] = redact_text(completed.stderr[-1200:])
    return step, payload


def build_local_smoke_report(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> dict[str, Any]:
    child_dir = output_dir / "local-smoke"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "public_swarm_inference_alpha_check.py"),
        "--output-dir",
        str(child_dir),
        "--mode",
        "live-kaggle",
        "--request-count",
        str(args.request_count),
        "--json",
    ]
    step, payload = run_json_step("public_swarm_inference_alpha_contract_check", command, runner=runner, timeout_seconds=args.timeout_seconds)
    codes = set(diagnosis_codes(payload))
    ready = bool(step.get("ok") and payload.get("schema") == CHECK_SCHEMA and "public_swarm_inference_alpha_check_ready" in codes)
    if ready:
        codes.update({"public_swarm_alpha_rc_local_smoke_ready", "public_swarm_alpha_rc_contract_ready"})
    else:
        codes.add("public_swarm_alpha_rc_local_smoke_blocked")
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ready,
        "mode": MODE_LOCAL_SMOKE,
        "output_dir": str(output_dir),
        "release_candidate": {
            "ready": False,
            "evidence_imported": False,
            "local_smoke_ready": ready,
        },
        "steps": [step],
        "payload_summaries": {
            "public_swarm_inference_alpha_check": {
                "schema": payload.get("schema"),
                "ok": payload.get("ok"),
                "diagnosis_codes": diagnosis_codes(payload),
            },
        },
        "diagnosis_codes": sorted(codes),
        "artifacts": {
            "public_swarm_inference_alpha_rc_json": artifact_entry(
                output_dir / "public_swarm_inference_alpha_rc.json",
                output_dir,
                kind="public_swarm_inference_alpha_rc",
                schema=SCHEMA,
                ok=ready,
            ),
            "public_swarm_inference_alpha_rc_markdown": artifact_entry(
                output_dir / "public_swarm_inference_alpha_rc.md",
                output_dir,
                kind="public_swarm_inference_alpha_rc_markdown",
            ),
        },
        "safety": {
            "cpu_only": True,
            "read_only_workload": "real_llm_sharded_infer",
            "not_production": True,
            "not_p2p": True,
            "not_large_model_serving": True,
        },
        "limitations": [
            "Local smoke validates the RC contract without importing retained live evidence.",
            "It does not create Kaggle resources and does not prove a fresh external runtime.",
        ],
    }


def not_completed_items(report: dict[str, Any]) -> list[str]:
    rc = report.get("release_candidate") if isinstance(report.get("release_candidate"), dict) else {}
    imported = report.get("imported_reports") if isinstance(report.get("imported_reports"), dict) else {}
    safety = report.get("safety") if isinstance(report.get("safety"), dict) else {}
    codes = set(report.get("diagnosis_codes") or [])
    mode = str(report.get("mode") or "")
    existing = [str(item) for item in (report.get("not_completed") or []) if str(item)]
    items: list[tuple[str, Any]] = [
        ("Public Swarm Alpha RC ready", report.get("ok")),
        ("CPU-only safety boundary present", safety.get("cpu_only") is True),
        ("production/P2P boundary stated", safety.get("not_production") is True and safety.get("not_p2p") is True),
    ]
    if mode == MODE_EVIDENCE_IMPORT:
        items.extend([
            ("retained stage0 live requeue evidence ready", rc.get("stage0_live_requeue_ready") is True),
            ("retained stage1 live requeue evidence ready", rc.get("stage1_live_requeue_ready") is True),
            ("retained live requeue summary ready", rc.get("summary_ready") is True),
            ("retained public artifacts contain no private debug files", "public_swarm_alpha_private_artifacts_absent" in codes),
        ])
        for stage in ["stage0", "stage1"]:
            stage_report = imported.get(stage) if isinstance(imported.get(stage), dict) else {}
            for failed in stage_report.get("failed_checks") or []:
                items.append((f"{stage} check cleared: {failed}", False))
    else:
        items.append(("local smoke contract ready", "public_swarm_alpha_rc_contract_ready" in codes))
        for step in report.get("steps") or []:
            if isinstance(step, dict) and step.get("ok") is not True:
                items.append((f"step {step.get('name') or 'step'} passed", False))
    missing = list(existing)
    seen = set(missing)
    for label, ready in items:
        if ready is True or label in seen:
            continue
        missing.append(label)
        seen.add(label)
    return missing


def recommended_next_command(
    report: dict[str, Any],
    args: argparse.Namespace,
    *,
    output_dir: Path,
    missing: list[str],
) -> dict[str, Any]:
    mode = str(report.get("mode") or args.mode)
    if report.get("ok"):
        return command_entry(
            "inspect Public Swarm Alpha RC evidence",
            artifact_command(output_dir, "public_swarm_inference_alpha_rc.md"),
            reason="review_artifacts",
        )
    return command_entry(
        f"rerun Public Swarm Alpha RC {mode}",
        public_swarm_alpha_rc_command(args, output_dir, mode),
        reason="fix_retained_evidence_or_contract_blockers" if missing else "rerun_public_swarm_alpha_rc",
        side_effectful=mode == MODE_LOCAL_SMOKE,
    )


def next_commands(
    report: dict[str, Any],
    args: argparse.Namespace,
    *,
    output_dir: Path,
    recommended: dict[str, Any],
) -> list[dict[str, Any]]:
    mode = str(report.get("mode") or args.mode)
    commands = [
        command_entry(
            "inspect shareable RC summary",
            artifact_command(output_dir, "public_swarm_inference_alpha_rc.md"),
            reason="review_artifacts",
        ),
        command_entry(
            "inspect RC support bundle",
            artifact_command(output_dir, "support_bundle.json", lines="1,220p"),
            reason="inspect_diagnostics",
        ),
    ]
    if report.get("ok"):
        commands.append(command_entry(
            f"refresh {mode} RC",
            public_swarm_alpha_rc_command(args, output_dir, mode),
            reason="refresh_public_swarm_alpha_rc",
            side_effectful=mode == MODE_LOCAL_SMOKE,
        ))
    else:
        commands.append(dict(recommended))
    if mode != MODE_EVIDENCE_IMPORT:
        commands.append(command_entry(
            "audit retained Alpha live evidence",
            public_swarm_alpha_rc_command(args, output_dir, MODE_EVIDENCE_IMPORT),
            reason="promote_local_smoke_to_retained_evidence_audit",
        ))
    else:
        commands.append(command_entry(
            "run RC local smoke",
            public_swarm_alpha_rc_command(args, output_dir, MODE_LOCAL_SMOKE),
            reason="validate_rc_contract_locally",
            side_effectful=True,
        ))
    return commands


def user_status(*, ready: bool, mode: str, recommended: dict[str, Any], missing: list[str]) -> dict[str, Any]:
    if ready:
        state = "ready"
        headline = "Public Swarm Inference Alpha RC evidence is ready."
        next_step = "review_artifacts"
    elif mode == MODE_EVIDENCE_IMPORT:
        state = "evidence-import-blocked"
        headline = "Public Swarm Alpha RC retained evidence audit needs attention."
        next_step = "fix_retained_evidence_or_private_artifacts"
    else:
        state = "local-smoke-blocked"
        headline = "Public Swarm Alpha RC local smoke contract needs attention."
        next_step = "fix_local_smoke_contract"
    return {
        "state": state,
        "headline": headline,
        "next_step": next_step,
        "recommended_label": recommended.get("label") or "none",
        "recommended_reason": recommended.get("reason") or "none",
        "not_completed_count": len(missing),
        "public_artifact_safe": True,
    }


def review_summary(
    report: dict[str, Any],
    *,
    output_dir: Path,
    recommended: dict[str, Any],
    missing: list[str],
) -> dict[str, Any]:
    ready = bool(report.get("ok"))
    codes = [str(code) for code in (report.get("diagnosis_codes") or [])]
    return {
        "schema": "public_swarm_inference_alpha_rc_review_summary_v1",
        "state": "ready" if ready else "blocked",
        "headline": (
            "Public Swarm Inference Alpha RC evidence is ready."
            if ready
            else "Public Swarm Inference Alpha RC evidence needs attention."
        ),
        "mode": report.get("mode"),
        "next_step": "review_artifacts" if ready else "fix_blockers",
        "inspect_first": str(output_dir / "public_swarm_inference_alpha_rc.md"),
        "support_bundle": str(output_dir / "support_bundle.json"),
        "recommended_label": recommended.get("label") or "none",
        "recommended_reason": recommended.get("reason") or "none",
        "next_command": recommended.get("command_line") or "",
        "primary_code": "public_swarm_inference_alpha_rc_ready" if ready else (codes[0] if codes else "public_swarm_inference_alpha_rc_blocked"),
        "attention": "none" if ready else (missing[0] if missing else "public_swarm_inference_alpha_rc_blocked"),
        "attention_detail": "; ".join(missing[:6]),
        "not_completed_count": len(missing),
        "public_artifact_safe": True,
    }


def attach_user_guidance(report: dict[str, Any], args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    missing = not_completed_items(report)
    recommended = recommended_next_command(report, args, output_dir=output_dir, missing=missing)
    report["not_completed"] = missing
    report["recommended_next_command"] = recommended
    report["next_commands"] = next_commands(report, args, output_dir=output_dir, recommended=recommended)
    report["user_status"] = user_status(
        ready=bool(report.get("ok")),
        mode=str(report.get("mode") or args.mode),
        recommended=recommended,
        missing=missing,
    )
    report["review_summary"] = review_summary(
        report,
        output_dir=output_dir,
        recommended=recommended,
        missing=missing,
    )
    report["artifact_summary"] = artifact_summary(output_dir)
    return report


def ensure_user_guidance(report: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    if (
        isinstance(report.get("recommended_next_command"), dict)
        and isinstance(report.get("next_commands"), list)
        and isinstance(report.get("user_status"), dict)
        and isinstance(report.get("review_summary"), dict)
    ):
        report.setdefault("not_completed", not_completed_items(report))
        report.setdefault("artifact_summary", artifact_summary(output_dir))
        return report
    missing = not_completed_items(report)
    recommended = command_entry(
        "inspect Public Swarm Alpha RC evidence",
        artifact_command(output_dir, "public_swarm_inference_alpha_rc.md"),
        reason="review_artifacts" if report.get("ok") else "review_missing_evidence",
    )
    report["not_completed"] = missing
    report["recommended_next_command"] = recommended
    report["next_commands"] = [
        command_entry("inspect shareable RC summary", artifact_command(output_dir, "public_swarm_inference_alpha_rc.md"), reason="review_artifacts"),
        command_entry("inspect RC support bundle", artifact_command(output_dir, "support_bundle.json", lines="1,220p"), reason="inspect_diagnostics"),
    ]
    report["user_status"] = user_status(
        ready=bool(report.get("ok")),
        mode=str(report.get("mode") or ""),
        recommended=recommended,
        missing=missing,
    )
    report["review_summary"] = review_summary(
        report,
        output_dir=output_dir,
        recommended=recommended,
        missing=missing,
    )
    report["artifact_summary"] = artifact_summary(output_dir)
    return report


def support_bundle_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "public_swarm_inference_alpha_rc_support_bundle_v1",
        "ok": report.get("ok"),
        "mode": report.get("mode"),
        "output_dir": report.get("output_dir"),
        "release_candidate": report.get("release_candidate"),
        "imported_reports": report.get("imported_reports"),
        "steps": report.get("steps"),
        "payload_summaries": report.get("payload_summaries"),
        "diagnosis_codes": report.get("diagnosis_codes"),
        "review_summary": report.get("review_summary"),
        "user_status": report.get("user_status"),
        "recommended_next_command": report.get("recommended_next_command"),
        "next_commands": report.get("next_commands"),
        "artifact_summary": report.get("artifact_summary"),
        "not_completed": report.get("not_completed"),
        "output_request": report.get("output_request"),
        "prompt_scope": report.get("prompt_scope"),
        "answer_scope": report.get("answer_scope"),
        "shareable_summary": report.get("shareable_summary"),
        "safety": report.get("safety"),
        "limitations": report.get("limitations"),
        "public_artifact_safe": True,
    }


def persist_report(report: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    report.setdefault("output_request", output_request_summary())
    report.setdefault("prompt_scope", prompt_scope_summary(report))
    report.setdefault("answer_scope", answer_scope_summary())
    report.setdefault("shareable_summary", shareable_summary())
    report.setdefault("artifact_summary", artifact_summary(output_dir))
    report = ensure_user_guidance(report, output_dir=output_dir)
    report = support_bundle.sanitize(redact_values(report))
    encoded = json.dumps(report, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        report["ok"] = False
        report["diagnosis_codes"] = sorted(set(report.get("diagnosis_codes") or []) | {"sensitive_output_detected"})
        report["safety_error"] = "Public Swarm Inference Alpha RC report contained secret-like fragments"
    json_path = output_dir / "public_swarm_inference_alpha_rc.json"
    md_path = output_dir / "public_swarm_inference_alpha_rc.md"
    support_path = output_dir / "support_bundle.json"
    report.setdefault("artifacts", {})
    report["artifacts"]["support_bundle_json"] = artifact_entry(
        support_path,
        output_dir,
        kind="public_swarm_inference_alpha_rc_support_bundle",
        schema="public_swarm_inference_alpha_rc_support_bundle_v1",
        ok=report.get("ok"),
    )
    report["artifact_summary"] = artifact_summary(output_dir)
    if isinstance(report.get("review_summary"), dict):
        report["review_summary"]["inspect_first"] = report["artifact_summary"]["inspect_first"]
        report["review_summary"]["support_bundle"] = report["artifact_summary"]["support_bundle"]
    write_json(json_path, report)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    write_json(support_path, support_bundle_payload(report))
    if "artifacts" in report:
        report["artifacts"]["public_swarm_inference_alpha_rc_json"]["present"] = True
        report["artifacts"]["public_swarm_inference_alpha_rc_markdown"]["present"] = True
        report["artifacts"]["support_bundle_json"]["present"] = True
        report["artifact_summary"] = artifact_summary(output_dir)
        if isinstance(report.get("review_summary"), dict):
            report["review_summary"]["inspect_first"] = report["artifact_summary"]["inspect_first"]
            report["review_summary"]["support_bundle"] = report["artifact_summary"]["support_bundle"]
        write_json(json_path, report)
        md_path.write_text(render_markdown(report), encoding="utf-8")
        write_json(support_path, support_bundle_payload(report))
    return report


def render_markdown(report: dict[str, Any]) -> str:
    rc = report.get("release_candidate") if isinstance(report.get("release_candidate"), dict) else {}
    output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
    prompt_scope = report.get("prompt_scope") if isinstance(report.get("prompt_scope"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    shareable = report.get("shareable_summary") if isinstance(report.get("shareable_summary"), dict) else {}
    review = report.get("review_summary") if isinstance(report.get("review_summary"), dict) else {}
    user = report.get("user_status") if isinstance(report.get("user_status"), dict) else {}
    recommended = report.get("recommended_next_command") if isinstance(report.get("recommended_next_command"), dict) else {}
    artifact_report = report.get("artifact_summary") if isinstance(report.get("artifact_summary"), dict) else {}
    next_items = report.get("next_commands") if isinstance(report.get("next_commands"), list) else []
    lines = [
        "# CrowdTensor Public Swarm Inference Alpha RC",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{report.get('ok')}`",
        f"- mode: `{report.get('mode')}`",
        f"- ready: `{rc.get('ready')}`",
        f"- evidence_imported: `{rc.get('evidence_imported')}`",
        f"- model_id: `{rc.get('model_id')}`",
        f"- stage0_live_requeue_ready: `{rc.get('stage0_live_requeue_ready')}`",
        f"- stage1_live_requeue_ready: `{rc.get('stage1_live_requeue_ready')}`",
        "",
        "## Review",
        "",
        f"- state: `{review.get('state')}`",
        f"- status: `{user.get('headline')}`",
        f"- next step: `{review.get('next_step')}`",
        f"- inspect first: `{review.get('inspect_first')}`",
        f"- recommended: `{recommended.get('label')}` reason=`{recommended.get('reason')}`",
        f"- recommended command: `{recommended.get('command_line')}`",
        f"- not completed count: `{review.get('not_completed_count')}`",
        "",
        "## What To Do Next",
        "",
    ]
    if next_items:
        lines.extend(
            (
                f"- {item.get('label')}: `{item.get('command_line')}`"
                + (" (requires private credentials; see runbook)" if item.get("requires_private_credentials") else "")
                + (" side_effectful=`True`" if item.get("side_effectful") else "")
            )
            for item in next_items
            if isinstance(item, dict)
        )
    else:
        lines.append("- none")
    lines.extend([
        "",
        "## Output Scope",
        "",
        f"- include output: `{output_request.get('include_output')}`",
        f"- output request note: {output_request.get('summary') or 'Public artifacts summarize inference evidence only and do not include answer text.'}",
        f"- prompt scope: `{prompt_scope_text(prompt_scope)}`",
        f"- prompt scope note: {prompt_scope_note(prompt_scope)}",
        f"- answer scope: `{answer_scope.get('scope_state')}`",
        f"- answer scope note: {answer_scope_note(answer_scope)}",
        f"- saved JSON display: `{answer_scope.get('saved_json_display')}`",
        f"- saved Markdown display: `{answer_scope.get('saved_markdown_display')}`",
        f"- shareable: `saved_artifacts={shareable.get('saved_artifacts_public_safe')} raw_prompt_public={shareable.get('raw_prompt_public')} raw_generation_public={shareable.get('raw_generation_public')} generation_ids_public={shareable.get('generation_ids_public')} answer_scope_state={shareable.get('answer_scope_state')} local_answer_terminal_only={shareable.get('local_answer_terminal_only')}`",
        "",
        "## Artifact Summary",
        "",
        f"- inspect first: `{artifact_report.get('inspect_first')}`",
        f"- summary JSON: `{artifact_report.get('summary_json')}`",
        f"- summary Markdown: `{artifact_report.get('summary_markdown')}`",
        f"- support bundle: `{artifact_report.get('support_bundle')}`",
        f"- present: `{artifact_report.get('present_artifact_count')}` / `{artifact_report.get('artifact_count')}`",
        f"- public artifact safe: `{artifact_report.get('public_artifact_safe')}`",
        "",
        "## Diagnosis",
        "",
        ", ".join(f"`{code}`" for code in report.get("diagnosis_codes") or []) or "`none`",
        "",
        "## Not Completed",
        "",
    ])
    not_completed = report.get("not_completed") or []
    lines.extend(f"- {item}" for item in not_completed) if not_completed else lines.append("- none")
    lines.extend([
        "",
        "## Artifacts",
        "",
    ])
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        lines.append(f"- `{name}`: `{artifact.get('path')}` present=`{artifact.get('present')}`")
    lines.extend(["", "## Boundaries", ""])
    for limitation in report.get("limitations") or []:
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


def build_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == MODE_LOCAL_SMOKE:
        report = build_local_smoke_report(args, output_dir=output_dir, runner=runner)
    else:
        report = build_evidence_import_report(args, output_dir=output_dir)
    report = attach_user_guidance(report, args, output_dir=output_dir)
    return persist_report(report, output_dir=output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Public Swarm Inference Alpha RC evidence.")
    parser.add_argument("--mode", choices=[MODE_EVIDENCE_IMPORT, MODE_LOCAL_SMOKE], default=MODE_EVIDENCE_IMPORT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage0-report", default=DEFAULT_STAGE0_REPORT)
    parser.add_argument("--stage1-report", default=DEFAULT_STAGE1_REPORT)
    parser.add_argument("--summary-report", default=DEFAULT_SUMMARY_REPORT)
    parser.add_argument("--request-count", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.request_count < 1 or args.request_count > 4:
        raise SystemExit("--request-count must be between 1 and 4")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    return args


def print_human(report: dict[str, Any]) -> None:
    rc = report.get("release_candidate") if isinstance(report.get("release_candidate"), dict) else {}
    user_status_report = report.get("user_status") if isinstance(report.get("user_status"), dict) else {}
    review = report.get("review_summary") if isinstance(report.get("review_summary"), dict) else {}
    recommended = report.get("recommended_next_command") if isinstance(report.get("recommended_next_command"), dict) else {}
    output_request = report.get("output_request") if isinstance(report.get("output_request"), dict) else {}
    prompt_scope = report.get("prompt_scope") if isinstance(report.get("prompt_scope"), dict) else {}
    answer_scope = report.get("answer_scope") if isinstance(report.get("answer_scope"), dict) else {}
    artifact_report = report.get("artifact_summary") if isinstance(report.get("artifact_summary"), dict) else {}
    print("CrowdTensor Public Swarm Inference Alpha RC")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  ready: {rc.get('ready')}")
    print(f"  output: {report.get('output_dir')}")
    if user_status_report:
        print(
            "  status: "
            f"{user_status_report.get('state')} "
            f"next={user_status_report.get('next_step')} "
            f"recommended={user_status_report.get('recommended_label')}"
        )
    if review:
        print(
            "  review: "
            f"state={review.get('state')} next={review.get('next_step')} "
            f"inspect={review.get('inspect_first')} attention={review.get('attention')}"
        )
        print(f"  review_next: {review.get('next_command') or 'none'}")
    if recommended:
        print(
            "  recommended_next: "
            f"{recommended.get('label')} reason={recommended.get('reason')} {recommended.get('command_line')}"
        )
    if output_request:
        print(f"  output_request: include_output={bool(output_request.get('include_output'))} raw_generation_public={bool(output_request.get('raw_generation_public'))} public_artifact_safe={bool(output_request.get('public_artifact_safe'))}")
        print(f"  output_request_note: {output_request.get('summary') or 'Public artifacts summarize inference evidence only and do not include answer text.'}")
    if prompt_scope:
        print(f"  prompt_scope: {prompt_scope_text(prompt_scope)}")
        print(f"  prompt_scope_note: {prompt_scope_note(prompt_scope)}")
    if answer_scope:
        print(f"  answer_scope: {answer_scope.get('scope_state')}")
        print(f"  answer_scope_note: {answer_scope_note(answer_scope)}")
    for index, item in enumerate((report.get("next_commands") or []), start=1):
        if not isinstance(item, dict):
            continue
        suffix = ""
        if item.get("requires_private_credentials"):
            suffix += " (requires private credentials)"
        if item.get("side_effectful"):
            suffix += " side_effectful=True"
        print(f"  next[{index}] {item.get('label')}: {item.get('command_line')}{suffix}")
    if artifact_report:
        print(
            "  artifacts: "
            f"present={artifact_report.get('present_artifact_count')}/{artifact_report.get('artifact_count')} "
            f"support={artifact_report.get('support_bundle')} "
            f"public_artifact_safe={bool(artifact_report.get('public_artifact_safe'))}"
        )
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print_human(report)
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
