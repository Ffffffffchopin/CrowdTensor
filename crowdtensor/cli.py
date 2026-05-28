"""User-facing CrowdTensor command line entrypoints."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
for path in [ROOT, SCRIPTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import support_bundle  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover - fallback for unusual packaging layouts
    support_bundle = None

from crowdtensor.p2p_lite import PEER_SCHEMA, fetch_peer_catalog, post_announce, sanitize_peer
from crowdtensor.session_protocol import (
    build_route_decision,
    build_session_request,
    coordinator_payload_for_request,
    safe_generation_summary,
)


SUMMARY_SCHEMA = "local_proof_summary_v1"
CLEANUP_SCHEMA = "cleanup_report_v1"
REMOTE_RUNBOOK_CLI_SCHEMA = "remote_runbook_cli_v1"
REMOTE_ACCEPTANCE_CLI_SCHEMA = "remote_acceptance_cli_v1"
REMOTE_HOME_DEMO_SCHEMA = "remote_home_compute_demo_v1"
HOME_INFERENCE_CLI_SCHEMA = "home_inference_cli_v1"
LLM_INFERENCE_CLI_SCHEMA = "llm_inference_cli_v1"
CPU_INFERENCE_BETA_CLI_SCHEMA = "cpu_inference_beta_cli_v1"
CPU_INFERENCE_BETA_RC_CLI_SCHEMA = "cpu_inference_beta_rc_cli_v1"
SHARDED_INFERENCE_CLI_SCHEMA = "sharded_inference_cli_v1"
MICRO_LLM_SHARDED_CLI_SCHEMA = "micro_llm_sharded_cli_v1"
MICRO_LLM_ARTIFACT_CLI_SCHEMA = "micro_llm_artifact_cli_v1"
REAL_LLM_SHARDED_CLI_SCHEMA = "real_llm_sharded_cli_v1"
REMOTE_SHARDED_INFERENCE_BETA_CLI_SCHEMA = "remote_sharded_inference_beta_cli_v1"
REMOTE_MICRO_LLM_SHARDED_BETA_CLI_SCHEMA = "remote_micro_llm_sharded_beta_cli_v1"
REMOTE_REAL_LLM_SHARDED_BETA_CLI_SCHEMA = "remote_real_llm_sharded_beta_cli_v1"
KAGGLE_REAL_RUNTIME_SCHEMA = "kaggle_real_runtime_acceptance_v1"
MICRO_LLM_LIVE_RC_CLI_SCHEMA = "micro_llm_live_rc_cli_v1"
REAL_LLM_LIVE_RC_CLI_SCHEMA = "real_llm_live_rc_cli_v1"
REAL_LLM_INTERNET_ALPHA_CLI_SCHEMA = "real_llm_internet_alpha_cli_v1"
REAL_LLM_INTERNET_BETA_CLI_SCHEMA = "real_llm_internet_beta_cli_v1"
SWARM_INFERENCE_BETA_CLI_SCHEMA = "swarm_inference_beta_cli_v1"
PUBLIC_SWARM_INFERENCE_ALPHA_CLI_SCHEMA = "public_swarm_inference_alpha_cli_v1"
PUBLIC_SWARM_INFERENCE_ALPHA_RC_CLI_SCHEMA = "public_swarm_inference_alpha_rc_cli_v1"
PUBLIC_SWARM_INFERENCE_BETA_CLI_SCHEMA = "public_swarm_inference_beta_cli_v1"
PUBLIC_SWARM_INFERENCE_BETA_RC_CLI_SCHEMA = "public_swarm_inference_beta_rc_cli_v1"
PUBLIC_SWARM_PRODUCT_BETA_CLI_SCHEMA = "public_swarm_product_beta_cli_v1"
PUBLIC_SWARM_GPU_INFERENCE_BETA_CLI_SCHEMA = "public_swarm_gpu_inference_beta_cli_v1"
GPU_SHARDED_GENERATION_BETA_CLI_SCHEMA = "gpu_sharded_generation_beta_cli_v1"
PUBLIC_SWARM_PRODUCT_CLI_SCHEMA = "public_swarm_product_cli_v1"
P2P_LITE_CLI_SCHEMA = "p2p_lite_cli_v1"
SECRET_FRAGMENTS = (
    "CROWDTENSOR_MINER_TOKEN",
    "CROWDTENSOR_OBSERVER_TOKEN",
    "CROWDTENSOR_ADMIN_TOKEN",
    "lease_token",
    "idempotency_key",
    "inference_results",
    "sharded_inference_result",
    "activation_results",
    "logits",
    "external_llm_results",
    "output_text",
    "Bearer ",
)
REMOTE_DEMO_WORKLOADS = ["model-bundle", "external-llm", "sharded-model-bundle", "micro-llm-sharded", "real-llm-sharded"]
CLEANUP_TMP_DIR_PATTERNS = (
    "crowdtensor_local_proof*",
    "crowdtensor_demo_manifest_*",
    "crowdtensor_cli_test_*",
    "crowdtensor_*_test_*",
)
CLEANUP_REPORT_PATTERNS = (
    "crowdtensor_*.json",
    "crowdtensor_*.md",
)
PROTECTED_REPO_PARTS = {".git", ".venv", "venv", "state"}

Runner = Callable[..., subprocess.CompletedProcess[str]]


def request_json_url(
    method: str,
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    admin_token: str = "",
    timeout: float = 10.0,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"content-type": "application/json"}
    if admin_token:
        headers["x-crowdtensor-admin-token"] = admin_token
    request = Request(f"{base_url.rstrip('/')}{path}", data=body, headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def redacted_command(command: list[str], sensitive_flags: set[str]) -> list[str]:
    result: list[str] = []
    redact_next = False
    for item in command:
        if redact_next:
            result.append("<redacted>")
            redact_next = False
            continue
        result.append(item)
        if item in sensitive_flags:
            redact_next = True
    return result


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sanitize(value: Any) -> Any:
    if support_bundle is not None:
        return support_bundle.sanitize(value)
    return value


def redact_text(text: str, redact_values: list[str] | None = None) -> str:
    redacted = text
    for value in redact_values or []:
        if value:
            redacted = redacted.replace(value, "<redacted>")
    return redacted


def redact_values(value: Any, secret_values: list[str] | None = None) -> Any:
    if isinstance(value, str):
        return redact_text(value, secret_values)
    if isinstance(value, dict):
        return {key: redact_values(item, secret_values) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_values(item, secret_values) for item in value]
    return value


def json_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("command emitted no JSON object")


def diagnosis_codes(*payloads: dict[str, Any]) -> list[str]:
    codes: set[str] = set()
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for code in payload.get("diagnosis_codes") or []:
            if isinstance(code, str):
                codes.add(code)
        summary = payload.get("diagnosis_summary")
        if isinstance(summary, dict):
            for code in summary.get("codes") or []:
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


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def has_protected_repo_part(path: Path, root: Path) -> bool:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return any(part in PROTECTED_REPO_PARTS for part in relative.parts)


def path_size(path: Path) -> int:
    if path.is_symlink():
        return 0
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_symlink():
            continue
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def age_hours(path: Path, *, now: float | None = None) -> float:
    current = time.time() if now is None else now
    return max(0.0, (current - path.stat().st_mtime) / 3600.0)


def cleanup_candidate(
    path: Path,
    *,
    kind: str,
    reason: str,
    root: Path,
    tmp_root: Path,
    older_than_hours: float,
    include_reports: bool,
    age_gated: bool,
    report_gated: bool = False,
) -> dict[str, Any]:
    candidate = {
        "path": str(path.resolve()),
        "kind": kind,
        "reason": reason,
        "bytes": 0,
        "eligible": False,
        "skip_reason": "",
    }
    try:
        if path.is_symlink():
            candidate["skip_reason"] = "symlink"
            return candidate
        resolved = path.resolve()
        if not (is_relative_to(resolved, root) or is_relative_to(resolved, tmp_root)):
            candidate["skip_reason"] = "outside_allowed_roots"
            return candidate
        if is_relative_to(resolved, root) and has_protected_repo_part(resolved, root):
            candidate["skip_reason"] = "protected_repo_path"
            return candidate
        candidate["bytes"] = path_size(resolved)
        if report_gated and not include_reports:
            candidate["skip_reason"] = "requires_include_reports"
            return candidate
        if age_gated:
            candidate["age_hours"] = round(age_hours(resolved), 3)
            if float(candidate["age_hours"]) < older_than_hours:
                candidate["skip_reason"] = "too_new"
                return candidate
        candidate["eligible"] = True
        return candidate
    except OSError as exc:
        candidate["skip_reason"] = f"stat_failed: {exc}"
        return candidate


def discover_cleanup_candidates(
    *,
    root: Path = ROOT,
    tmp_root: Path = Path("/tmp"),
    older_than_hours: float = 24.0,
    include_reports: bool = False,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(candidate: dict[str, Any]) -> None:
        key = str(candidate.get("path"))
        if key in seen:
            return
        seen.add(key)
        candidates.append(candidate)

    for cache_dir in root.rglob("__pycache__"):
        add(cleanup_candidate(
            cache_dir,
            kind="python_cache",
            reason="repo __pycache__ directory",
            root=root,
            tmp_root=tmp_root,
            older_than_hours=older_than_hours,
            include_reports=include_reports,
            age_gated=False,
        ))
    for pyc_file in root.rglob("*.pyc"):
        add(cleanup_candidate(
            pyc_file,
            kind="python_cache",
            reason="repo .pyc file",
            root=root,
            tmp_root=tmp_root,
            older_than_hours=older_than_hours,
            include_reports=include_reports,
            age_gated=False,
        ))
    for pattern in CLEANUP_TMP_DIR_PATTERNS:
        for temp_dir in tmp_root.glob(pattern):
            if temp_dir.is_dir() or temp_dir.is_symlink():
                add(cleanup_candidate(
                    temp_dir,
                    kind="tmp_dir",
                    reason=f"temporary CrowdTensor artifact matching {pattern}",
                    root=root,
                    tmp_root=tmp_root,
                    older_than_hours=older_than_hours,
                    include_reports=include_reports,
                    age_gated=True,
                ))
    for pattern in CLEANUP_REPORT_PATTERNS:
        for report in tmp_root.glob(pattern):
            if report.is_file() or report.is_symlink():
                add(cleanup_candidate(
                    report,
                    kind="report",
                    reason=f"optional CrowdTensor report matching {pattern}",
                    root=root,
                    tmp_root=tmp_root,
                    older_than_hours=older_than_hours,
                    include_reports=include_reports,
                    age_gated=True,
                    report_gated=True,
                ))
    return sorted(candidates, key=lambda item: str(item.get("path")))


def delete_candidate(path: Path) -> None:
    if path.is_symlink():
        raise OSError("refusing to delete symlink")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def build_cleanup_report(
    args: argparse.Namespace,
    *,
    root: Path = ROOT,
    tmp_root: Path = Path("/tmp"),
) -> dict[str, Any]:
    candidates = discover_cleanup_candidates(
        root=root,
        tmp_root=tmp_root,
        older_than_hours=args.older_than_hours,
        include_reports=args.include_reports,
    )
    apply = bool(args.apply)
    deleted_bytes = 0
    errors: list[str] = []
    for candidate in candidates:
        candidate["action"] = "skipped"
        if not candidate.get("eligible"):
            continue
        if not apply:
            candidate["action"] = "dry_run"
            continue
        try:
            delete_candidate(Path(str(candidate["path"])))
        except OSError as exc:
            candidate["action"] = "error"
            candidate["error"] = str(exc)
            errors.append(str(candidate["path"]))
            continue
        candidate["action"] = "deleted"
        deleted_bytes += int(candidate.get("bytes") or 0)
    return sanitize({
        "schema": CLEANUP_SCHEMA,
        "generated_at": utc_now(),
        "ok": not errors,
        "mode": "apply" if apply else "dry_run",
        "root": str(root.resolve()),
        "tmp_root": str(tmp_root.resolve()),
        "older_than_hours": args.older_than_hours,
        "include_reports": bool(args.include_reports),
        "candidate_count": len(candidates),
        "deleted_bytes": deleted_bytes,
        "errors": errors,
        "candidates": candidates,
        "safety": {
            "dry_run_default": True,
            "reports_require_include_reports": True,
            "allowed_roots": [str(root.resolve()), str(tmp_root.resolve())],
            "protected_repo_parts": sorted(PROTECTED_REPO_PARTS),
        },
    })


def run_json_step(
    name: str,
    command: list[str],
    *,
    runner: Runner,
    cwd: Path,
    timeout_seconds: int,
    redact_secrets: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    try:
        completed = runner(
            command,
            cwd=str(cwd),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        elapsed = round(time.monotonic() - started, 3)
        return (
            {
                "name": name,
                "ok": False,
                "returncode": None,
                "duration_seconds": elapsed,
                "error": "timeout",
            },
            {},
        )
    elapsed = round(time.monotonic() - started, 3)
    step = {
        "name": name,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": elapsed,
    }
    payload: dict[str, Any] = {}
    try:
        payload = json_from_stdout(completed.stdout)
    except ValueError as exc:
        step["ok"] = False
        step["error"] = str(exc)
    if not step["ok"]:
        if completed.stderr:
            step["stderr_tail"] = sanitize(redact_text(completed.stderr[-1000:], redact_secrets))
        if completed.stdout and not payload:
            step["stdout_tail"] = sanitize(redact_text(completed.stdout[-1000:], redact_secrets))
    return step, payload


def build_local_proof(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    steps: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    doctor_payload: dict[str, Any] = {}
    if args.skip_doctor:
        steps.append({"name": "doctor", "ok": True, "skipped": True})
    else:
        doctor_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "doctor.py"),
            "--root",
            str(ROOT),
            "--state-dir",
            str(output_dir / "doctor-state"),
            "--port",
            "0",
            "--json",
        ]
        doctor_step, doctor_payload = run_json_step(
            "doctor",
            doctor_cmd,
            runner=runner,
            cwd=ROOT,
            timeout_seconds=args.timeout_seconds,
        )
        doctor_step["ok"] = bool(doctor_step.get("ok") and doctor_payload.get("ok"))
        if not doctor_step["ok"]:
            errors.append("doctor_failed")
        steps.append(doctor_step)

    matrix_cmd = [sys.executable, str(SCRIPTS_DIR / "runtime_matrix.py"), "--json"]
    matrix_step, matrix_payload = run_json_step(
        "runtime_matrix",
        matrix_cmd,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
    )
    matrix_step["ok"] = bool(matrix_step.get("ok") and matrix_payload.get("ok"))
    if not matrix_step["ok"]:
        errors.append("runtime_matrix_blocked")
    steps.append(matrix_step)

    home_payload: dict[str, Any] = {}
    manifest_payload: dict[str, Any] = {}
    if matrix_step["ok"]:
        home_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "home_compute_demo.py"),
            "--port",
            str(args.base_port),
            "--request-count",
            str(args.request_count),
            "--json",
        ]
        home_step, home_payload = run_json_step(
            "home_compute_demo",
            home_cmd,
            runner=runner,
            cwd=ROOT,
            timeout_seconds=args.timeout_seconds,
        )
        home_step["ok"] = bool(home_step.get("ok") and home_payload.get("ok"))
        if not home_step["ok"]:
            errors.append("home_compute_demo_failed")
        steps.append(home_step)

        if home_step["ok"]:
            manifest_cmd = [
                sys.executable,
                str(SCRIPTS_DIR / "demo_manifest_pack.py"),
                "--output-dir",
                str(output_dir),
                "--port",
                str(args.base_port + 1),
                "--request-count",
                str(args.request_count),
            ]
            manifest_step, manifest_payload = run_json_step(
                "demo_manifest",
                manifest_cmd,
                runner=runner,
                cwd=ROOT,
                timeout_seconds=args.timeout_seconds,
            )
            manifest_step["ok"] = bool(manifest_step.get("ok") and manifest_payload.get("ok"))
            if not manifest_step["ok"]:
                errors.append("demo_manifest_failed")
            steps.append(manifest_step)
    else:
        steps.append({"name": "home_compute_demo", "ok": False, "skipped": True, "reason": "runtime_matrix_blocked"})
        steps.append({"name": "demo_manifest", "ok": False, "skipped": True, "reason": "runtime_matrix_blocked"})

    manifest_json = output_dir / "demo_manifest.json"
    manifest_md = output_dir / "demo_manifest.md"
    summary_json = output_dir / "local_proof_summary.json"
    artifacts["demo_manifest_json"] = artifact_entry(
        manifest_json,
        output_dir,
        kind="demo_manifest",
        schema=str(manifest_payload.get("schema") or "demo_manifest_v1"),
        ok=manifest_payload.get("ok") if manifest_payload else None,
    )
    artifacts["demo_manifest_markdown"] = artifact_entry(manifest_md, output_dir, kind="demo_manifest_markdown")
    artifacts["local_proof_summary"] = {
        "kind": "local_proof_summary",
        "path": "local_proof_summary.json",
        "present": True,
        "schema": SUMMARY_SCHEMA,
    }

    summary = {
        "schema": SUMMARY_SCHEMA,
        "generated_at": utc_now(),
        "ok": not errors and all(bool(step.get("ok")) for step in steps if not step.get("skipped")),
        "output_dir": str(output_dir),
        "request_count": args.request_count,
        "base_port": args.base_port,
        "steps": steps,
        "diagnosis_codes": diagnosis_codes(matrix_payload, home_payload, manifest_payload),
        "artifacts": artifacts,
        "errors": errors,
        "limitations": [
            "CPU-only local proof; not production Swarm Inference",
            "Read-only model_bundle_infer rehearsal; not arbitrary prompt or real LLM serving",
            "No GPU pooling, WebGPU model shards, libp2p discovery, NAT traversal, or incentives are claimed",
        ],
        "recommended_next_commands": [
            "crowdtensor home-infer --json",
            "python3 scripts/demo_manifest_check.py --base-port 8914",
            "python3 scripts/runtime_acceptance_pack.py --base-port 8910 --report /tmp/crowdtensor_acceptance.json",
            "python3 scripts/remote_demo_runbook_pack.py --coordinator-url https://YOUR_COORDINATOR_HOST --miner-id remote-linux-1 --output-dir dist/remote-demo",
        ],
    }
    encoded = json.dumps(summary, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        summary["ok"] = False
        summary["errors"].append("sensitive_output_detected")
        summary["safety_error"] = "local proof summary contained secret-like fragments"

    summary = sanitize(summary)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _remote_summary_artifact(path: Path, output_dir: Path, *, kind: str, schema: str = "") -> dict[str, Any]:
    return artifact_entry(path, output_dir, kind=kind, schema=schema)


def build_home_inference(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_json = output_dir / "home_compute_evidence.json"
    evidence_md = output_dir / "home_compute_evidence.md"
    summary_json = output_dir / "home_inference_cli_summary.json"
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "home_compute_evidence_pack.py"),
        "--port",
        str(args.port),
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
        "--json-out",
        str(evidence_json),
        "--markdown-out",
        str(evidence_md),
    ]
    if args.runtime_report:
        command.extend(["--runtime-report", args.runtime_report])
    step, payload = run_json_step(
        "home_compute_evidence",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
    )
    step["ok"] = bool(step.get("ok") and payload.get("ok"))
    inference = payload.get("inference_summary") if isinstance(payload.get("inference_summary"), dict) else {}
    route = payload.get("route_decision") if isinstance(payload.get("route_decision"), dict) else {}
    scenario = payload.get("scenario") if isinstance(payload.get("scenario"), dict) else {}
    artifacts = {
        "home_compute_evidence_json": artifact_entry(
            evidence_json,
            output_dir,
            kind="home_compute_evidence",
            schema=str(payload.get("schema") or "home_compute_evidence_v1"),
            ok=payload.get("ok") if payload else None,
        ),
        "home_compute_evidence_markdown": artifact_entry(
            evidence_md,
            output_dir,
            kind="home_compute_evidence_markdown",
        ),
        "home_inference_cli_summary": {
            "kind": "home_inference_cli_summary",
            "path": "home_inference_cli_summary.json",
            "present": True,
            "schema": HOME_INFERENCE_CLI_SCHEMA,
        },
    }
    summary = {
        "schema": HOME_INFERENCE_CLI_SCHEMA,
        "generated_at": utc_now(),
        "ok": bool(step.get("ok")),
        "output_dir": str(output_dir),
        "port": args.port,
        "request_count": args.request_count,
        "scenario_id": args.scenario_id,
        "step": step,
        "evidence_schema": payload.get("schema") or "home_compute_evidence_v1",
        "route": {
            "name": route.get("name"),
            "target": route.get("target"),
            "workload": route.get("workload"),
            "confidence": route.get("confidence"),
            "usable_now": route.get("usable_now"),
        },
        "diagnosis_codes": diagnosis_codes(payload),
        "scenario": {
            "scenario_schema": scenario.get("scenario_schema") or inference.get("scenario_schema"),
            "scenario_id": scenario.get("scenario_id") or inference.get("scenario_id"),
            "scenario_description": scenario.get("scenario_description") or inference.get("scenario_description"),
            "scenario_request_count": scenario.get("scenario_request_count") or inference.get("scenario_request_count"),
        },
        "inference": {
            "present": inference.get("present"),
            "ok": inference.get("ok"),
            "workload_type": inference.get("workload_type"),
            "request_count": inference.get("request_count"),
            "request_trace_count": inference.get("request_trace_count"),
            "requests_per_second": inference.get("requests_per_second"),
            "read_only": inference.get("read_only"),
            "redaction_ok": inference.get("redaction_ok"),
        },
        "artifacts": artifacts,
        "safety": {
            "captured_output_redacted": True,
            "summary_excludes_raw_inference_payloads": True,
            "read_only_workload": "model_bundle_infer",
            "not_production": True,
        },
        "limitations": [
            "CPU-only read-only model_bundle_infer proof; not production Swarm Inference",
            "Does not provide arbitrary prompt serving, real LLM serving, GPU pooling, WebGPU shards, or P2P routing",
        ],
        "recommended_next_commands": [
            "python3 scripts/runtime_acceptance_pack.py --base-port 8910 --report /tmp/crowdtensor_acceptance.json",
            "crowdtensor remote-runbook --coordinator-url https://YOUR_COORDINATOR_HOST --miner-id remote-linux-1 --json",
        ],
    }
    summary = sanitize(summary)
    encoded = json.dumps(summary, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        summary["ok"] = False
        summary.setdefault("errors", []).append("sensitive_output_detected")
        summary["safety_error"] = "home inference summary contained secret-like fragments"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_llm_inference(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_json = output_dir / "external_llm_evidence.json"
    evidence_md = output_dir / "external_llm_evidence.md"
    summary_json = output_dir / "llm_inference_cli_summary.json"
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "external_llm_evidence_pack.py"),
        "--port",
        str(args.port),
        "--request-count",
        str(args.request_count),
        "--llm-runtime-model-id",
        args.llm_runtime_model_id,
        "--llm-runtime-timeout",
        str(args.llm_runtime_timeout),
        "--json-out",
        str(evidence_json),
        "--markdown-out",
        str(evidence_md),
    ]
    if args.mock:
        command.append("--mock")
    if args.llm_runtime_cmd:
        command.extend(["--llm-runtime-cmd", args.llm_runtime_cmd])
    if args.llm_runtime_url:
        command.extend(["--llm-runtime-url", args.llm_runtime_url])
    if args.llm_runtime_api_key:
        command.extend(["--llm-runtime-api-key", args.llm_runtime_api_key])
    secret_values = [args.llm_runtime_url, args.llm_runtime_api_key]
    step, payload = run_json_step(
        "external_llm_evidence",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
        redact_secrets=secret_values,
    )
    step["ok"] = bool(step.get("ok") and payload.get("ok"))
    adapter = payload.get("adapter") if isinstance(payload.get("adapter"), dict) else {}
    llm_summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    artifacts = {
        "external_llm_evidence_json": artifact_entry(
            evidence_json,
            output_dir,
            kind="external_llm_evidence",
            schema=str(payload.get("schema") or "external_llm_evidence_v1"),
            ok=payload.get("ok") if payload else None,
        ),
        "external_llm_evidence_markdown": artifact_entry(
            evidence_md,
            output_dir,
            kind="external_llm_evidence_markdown",
        ),
        "llm_inference_cli_summary": {
            "kind": "llm_inference_cli_summary",
            "path": "llm_inference_cli_summary.json",
            "present": True,
            "schema": LLM_INFERENCE_CLI_SCHEMA,
        },
    }
    summary = {
        "schema": LLM_INFERENCE_CLI_SCHEMA,
        "generated_at": utc_now(),
        "ok": bool(step.get("ok")),
        "output_dir": str(output_dir),
        "port": args.port,
        "request_count": args.request_count,
        "step": step,
        "evidence_schema": payload.get("schema") or "external_llm_evidence_v1",
        "adapter": {
            "kind": adapter.get("kind"),
            "model_id": adapter.get("model_id"),
            "operator_owned_runtime": adapter.get("operator_owned_runtime"),
        },
        "inference": {
            "request_count": llm_summary.get("request_count"),
            "completion_count": llm_summary.get("completion_count"),
            "output_chars": llm_summary.get("output_chars"),
            "requests_per_second": llm_summary.get("requests_per_second"),
        },
        "diagnosis_codes": diagnosis_codes(payload),
        "artifacts": artifacts,
        "safety": {
            "captured_output_redacted": True,
            "summary_excludes_raw_external_llm_payloads": True,
            "runtime_url_redacted": True,
            "api_credential_redacted": True,
            "read_only_workload": "external_llm_infer",
            "not_production": True,
        },
        "limitations": [
            "Local external_llm_infer proof; not production LLM serving",
            "Uses fixed claim-time prompts; not an arbitrary public prompt API",
            "Does not provide GPU pooling, WebGPU shards, P2P routing, or incentives",
        ],
        "recommended_next_commands": [
            "python3 scripts/external_llm_evidence_check.py --port 8919",
            "python3 scripts/runtime_acceptance_pack.py --base-port 8910 --report /tmp/crowdtensor_acceptance.json",
        ],
    }
    summary = sanitize(redact_values(summary, secret_values))
    encoded = json.dumps(summary, sort_keys=True)
    if any(secret and secret in encoded for secret in secret_values):
        summary["ok"] = False
        summary.setdefault("errors", []).append("sensitive_output_detected")
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        summary["ok"] = False
        summary.setdefault("errors", []).append("sensitive_output_detected")
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_cpu_inference_beta(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "beta-rc":
        command = [
            sys.executable,
            str(SCRIPTS_DIR / "cpu_inference_beta_rc_pack.py"),
            "--output-dir",
            str(output_dir),
            "--base-port",
            str(args.base_port),
            "--request-count",
            str(args.request_count),
            "--external-llm-request-count",
            str(args.external_llm_request_count),
            "--scenario-id",
            args.scenario_id,
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--remote-timeout-seconds",
            str(args.remote_timeout_seconds),
            "--json",
        ]
        if args.kaggle_real_runtime_report:
            command.extend(["--kaggle-real-runtime-report", args.kaggle_real_runtime_report])
        step, payload = run_json_step(
            "cpu_inference_beta_rc",
            command,
            runner=runner,
            cwd=ROOT,
            timeout_seconds=args.timeout_seconds * 3,
            redact_secrets=[],
        )
        if payload:
            payload = sanitize(payload)
            payload.setdefault("cli_schema", CPU_INFERENCE_BETA_RC_CLI_SCHEMA)
            return payload
        return sanitize({
            "schema": "cpu_inference_beta_rc_v1",
            "cli_schema": CPU_INFERENCE_BETA_RC_CLI_SCHEMA,
            "ok": False,
            "mode": args.mode,
            "output_dir": str(output_dir),
            "step": step,
            "diagnosis_codes": ["beta_rc_blocked"],
            "limitations": [
                "CPU Inference Beta RC evidence only; not production Swarm Inference",
                "Does not provide GPU/TPU workloads, P2P routing, NAT traversal, or arbitrary public prompt serving",
            ],
        })
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "cpu_inference_beta_pack.py"),
        "--mode",
        args.mode,
        "--workload",
        args.workload,
        "--output-dir",
        str(output_dir),
        "--base-port",
        str(args.base_port),
        "--request-count",
        str(args.request_count),
        "--external-llm-request-count",
        str(args.external_llm_request_count),
        "--scenario-id",
        args.scenario_id,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--poll-interval",
        str(args.poll_interval),
        "--json",
    ]
    if args.coordinator_url:
        command.extend(["--coordinator-url", args.coordinator_url])
    if args.miner_id:
        command.extend(["--miner-id", args.miner_id])
    if args.observer_token:
        command.extend(["--observer-token", args.observer_token])
    if args.admin_token:
        command.extend(["--admin-token", args.admin_token])
    if args.mock:
        command.append("--mock")
    if args.llm_runtime_cmd:
        command.extend(["--llm-runtime-cmd", args.llm_runtime_cmd])
    if args.llm_runtime_url:
        command.extend(["--llm-runtime-url", args.llm_runtime_url])
    if args.llm_runtime_api_key:
        command.extend(["--llm-runtime-api-key", args.llm_runtime_api_key])
    if args.llm_runtime_model_id:
        command.extend(["--llm-runtime-model-id", args.llm_runtime_model_id])
    if args.llm_runtime_timeout:
        command.extend(["--llm-runtime-timeout", str(args.llm_runtime_timeout)])
    secret_values = [args.observer_token, args.admin_token, args.llm_runtime_url, args.llm_runtime_api_key]
    step, payload = run_json_step(
        "cpu_inference_beta",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
        redact_secrets=secret_values,
    )
    if payload:
        payload = redact_values(payload, secret_values)
        payload.setdefault("cli_schema", CPU_INFERENCE_BETA_CLI_SCHEMA)
        encoded = json.dumps(payload, sort_keys=True)
        if any(secret and secret in encoded for secret in secret_values):
            payload["ok"] = False
            payload.setdefault("diagnosis_codes", [])
            if "sensitive_output_detected" not in payload["diagnosis_codes"]:
                payload["diagnosis_codes"].append("sensitive_output_detected")
        return payload
    return sanitize({
        "schema": "cpu_inference_beta_v1",
        "cli_schema": CPU_INFERENCE_BETA_CLI_SCHEMA,
        "ok": False,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["cpu_inference_beta_failed"],
        "limitations": [
            "CPU-only Beta inference proof; not production Swarm Inference",
            "Does not provide GPU pooling, WebGPU shards, P2P routing, NAT traversal, or arbitrary public prompt serving",
        ],
    })


def build_sharded_inference(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_json = output_dir / "sharded_inference_evidence.json"
    evidence_md = output_dir / "sharded_inference_evidence.md"
    summary_json = output_dir / "sharded_inference_cli_summary.json"
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "sharded_inference_evidence_pack.py"),
        "--port",
        str(args.port),
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
        "--failure-mode",
        args.failure_mode,
        "--stage-mode",
        args.stage_mode,
        "--json-out",
        str(evidence_json),
        "--markdown-out",
        str(evidence_md),
    ]
    if args.require_distinct_stage_miners:
        command.append("--require-distinct-stage-miners")
    step, payload = run_json_step(
        "sharded_inference_evidence",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
    )
    payload = sanitize(payload) if payload else {}
    step["ok"] = bool(step.get("ok") and payload.get("ok"))
    summary = {
        "schema": SHARDED_INFERENCE_CLI_SCHEMA,
        "generated_at": utc_now(),
        "ok": bool(step.get("ok")),
        "output_dir": str(output_dir),
        "evidence_schema": payload.get("schema") or "sharded_inference_evidence_v1",
        "failure_mode": args.failure_mode,
        "stage_mode": args.stage_mode,
        "require_distinct_stage_miners": bool(args.require_distinct_stage_miners),
        "request_count": args.request_count,
        "scenario_id": args.scenario_id,
        "step": step,
        "diagnosis_codes": diagnosis_codes(payload),
        "session": payload.get("session") if isinstance(payload.get("session"), dict) else {},
        "stage_summary": payload.get("stage_summary") if isinstance(payload.get("stage_summary"), dict) else {},
        "safety": payload.get("safety") if isinstance(payload.get("safety"), dict) else {},
        "artifacts": {
            "sharded_inference_evidence_json": artifact_entry(
                evidence_json,
                output_dir,
                kind="sharded_inference_evidence",
                schema=str(payload.get("schema") or "sharded_inference_evidence_v1"),
                ok=payload.get("ok") if payload else None,
            ),
            "sharded_inference_evidence_markdown": artifact_entry(
                evidence_md,
                output_dir,
                kind="sharded_inference_evidence_markdown",
            ),
            "sharded_inference_cli_summary": artifact_entry(
                summary_json,
                output_dir,
                kind="sharded_inference_cli_summary",
                schema=SHARDED_INFERENCE_CLI_SCHEMA,
                ok=bool(step.get("ok")),
            ),
        },
        "limitations": [
            "CPU-only fixed two-stage pipeline; not production Swarm Inference",
            "Does not provide GPU/TPU pooling, P2P routing, real LLM sharding, or arbitrary prompt serving",
        ],
    }
    summary = sanitize(summary)
    encoded = json.dumps(summary, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        summary["ok"] = False
        summary.setdefault("errors", []).append("sensitive_output_detected")
        summary["safety_error"] = "sharded inference summary contained secret-like fragments"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["artifacts"]["sharded_inference_cli_summary"]["present"] = True
    return summary


def build_micro_llm_sharded_inference(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_json = output_dir / "micro_llm_sharded_evidence.json"
    evidence_md = output_dir / "micro_llm_sharded_evidence.md"
    summary_json = output_dir / "micro_llm_sharded_cli_summary.json"
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "micro_llm_sharded_inference_evidence_pack.py"),
        "--port",
        str(args.port),
        "--request-count",
        str(args.request_count),
        "--decode-steps",
        str(args.decode_steps),
        "--failure-mode",
        args.failure_mode,
        "--stage-mode",
        args.stage_mode,
        "--json-out",
        str(evidence_json),
        "--markdown-out",
        str(evidence_md),
    ]
    if getattr(args, "micro_llm_artifact", ""):
        command.extend(["--micro-llm-artifact", args.micro_llm_artifact])
    if getattr(args, "prompt_texts", ""):
        command.extend(["--prompt-texts", args.prompt_texts])
    if args.require_distinct_stage_miners:
        command.append("--require-distinct-stage-miners")
    step, payload = run_json_step(
        "micro_llm_sharded_evidence",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
    )
    payload = sanitize(payload) if payload else {}
    step["ok"] = bool(step.get("ok") and payload.get("ok"))
    summary = {
        "schema": MICRO_LLM_SHARDED_CLI_SCHEMA,
        "generated_at": utc_now(),
        "ok": bool(step.get("ok")),
        "output_dir": str(output_dir),
        "evidence_schema": payload.get("schema") or "micro_llm_sharded_evidence_v1",
        "failure_mode": args.failure_mode,
        "stage_mode": args.stage_mode,
        "require_distinct_stage_miners": bool(args.require_distinct_stage_miners),
        "request_count": args.request_count,
        "decode_steps": args.decode_steps,
        "micro_llm_artifact": getattr(args, "micro_llm_artifact", ""),
        "step": step,
        "diagnosis_codes": diagnosis_codes(payload),
        "session": payload.get("session") if isinstance(payload.get("session"), dict) else {},
        "stage_summary": payload.get("stage_summary") if isinstance(payload.get("stage_summary"), dict) else {},
        "safety": payload.get("safety") if isinstance(payload.get("safety"), dict) else {},
        "artifacts": {
            "micro_llm_sharded_evidence_json": artifact_entry(
                evidence_json,
                output_dir,
                kind="micro_llm_sharded_evidence",
                schema=str(payload.get("schema") or "micro_llm_sharded_evidence_v1"),
                ok=payload.get("ok") if payload else None,
            ),
            "micro_llm_sharded_evidence_markdown": artifact_entry(
                evidence_md,
                output_dir,
                kind="micro_llm_sharded_evidence_markdown",
            ),
            "micro_llm_sharded_cli_summary": artifact_entry(
                summary_json,
                output_dir,
                kind="micro_llm_sharded_cli_summary",
                schema=MICRO_LLM_SHARDED_CLI_SCHEMA,
                ok=bool(step.get("ok")),
            ),
        },
        "limitations": [
            "CPU-only deterministic micro-LLM two-stage pipeline; not production Swarm Inference",
            "Does not provide GPU/TPU pooling, P2P routing, GGUF/llama.cpp serving, or arbitrary prompt serving",
        ],
    }
    summary = sanitize(summary)
    encoded = json.dumps(summary, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        summary["ok"] = False
        summary.setdefault("errors", []).append("sensitive_output_detected")
        summary["safety_error"] = "micro-LLM sharded inference summary contained secret-like fragments"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["artifacts"]["micro_llm_sharded_cli_summary"]["present"] = True
    return summary


def build_real_llm_sharded_inference(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_json = output_dir / "real_llm_sharded_evidence.json"
    evidence_md = output_dir / "real_llm_sharded_evidence.md"
    summary_json = output_dir / "real_llm_sharded_cli_summary.json"
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "real_llm_sharded_inference_evidence_pack.py"),
        "--port",
        str(args.port),
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(getattr(args, "max_new_tokens", 1)),
        "--failure-mode",
        args.failure_mode,
        "--stage-mode",
        args.stage_mode,
        "--hf-model-id",
        str(args.hf_model_id),
        "--real-llm-partition-mode",
        str(getattr(args, "real_llm_partition_mode", "full")),
        "--json-out",
        str(evidence_json),
        "--markdown-out",
        str(evidence_md),
    ]
    if getattr(args, "hf_cache_dir", ""):
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if getattr(args, "prompt_texts", ""):
        command.extend(["--prompt-texts", args.prompt_texts])
    if args.require_distinct_stage_miners:
        command.append("--require-distinct-stage-miners")
    step, payload = run_json_step(
        "real_llm_sharded_evidence",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
    )
    payload = sanitize(payload) if payload else {}
    step["ok"] = bool(step.get("ok") and payload.get("ok"))
    summary = {
        "schema": REAL_LLM_SHARDED_CLI_SCHEMA,
        "generated_at": utc_now(),
        "ok": bool(step.get("ok")),
        "output_dir": str(output_dir),
        "evidence_schema": payload.get("schema") or "real_llm_sharded_evidence_v1",
        "failure_mode": args.failure_mode,
        "stage_mode": args.stage_mode,
        "require_distinct_stage_miners": bool(args.require_distinct_stage_miners),
        "request_count": args.request_count,
        "max_new_tokens": getattr(args, "max_new_tokens", 1),
        "hf_model_id": args.hf_model_id,
        "real_llm_partition_mode": getattr(args, "real_llm_partition_mode", "full"),
        "step": step,
        "diagnosis_codes": diagnosis_codes(payload),
        "session": payload.get("session") if isinstance(payload.get("session"), dict) else {},
        "artifact": payload.get("artifact") if isinstance(payload.get("artifact"), dict) else {},
        "stage_summary": payload.get("stage_summary") if isinstance(payload.get("stage_summary"), dict) else {},
        "safety": payload.get("safety") if isinstance(payload.get("safety"), dict) else {},
        "artifacts": {
            "real_llm_sharded_evidence_json": artifact_entry(
                evidence_json,
                output_dir,
                kind="real_llm_sharded_evidence",
                schema=str(payload.get("schema") or "real_llm_sharded_evidence_v1"),
                ok=payload.get("ok") if payload else None,
            ),
            "real_llm_sharded_evidence_markdown": artifact_entry(
                evidence_md,
                output_dir,
                kind="real_llm_sharded_evidence_markdown",
            ),
            "real_llm_sharded_cli_summary": artifact_entry(
                summary_json,
                output_dir,
                kind="real_llm_sharded_cli_summary",
                schema=REAL_LLM_SHARDED_CLI_SCHEMA,
                ok=bool(step.get("ok")),
            ),
        },
        "limitations": [
            "CPU-only tiny Hugging Face GPT two-stage pipeline; not production Swarm Inference",
            "Does not provide GPU/TPU pooling, P2P routing, GGUF/llama.cpp serving, or arbitrary prompt serving",
        ],
    }
    summary = sanitize(summary)
    encoded = json.dumps(summary, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]
    if leaks:
        summary["ok"] = False
        summary.setdefault("errors", []).append("sensitive_output_detected")
        summary["safety_error"] = "real LLM sharded inference summary contained secret-like fragments"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["artifacts"]["real_llm_sharded_cli_summary"]["present"] = True
    return summary


def build_micro_llm_artifact(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "micro_llm_artifact_cli_summary.json"
    artifact_json = output_dir / "micro_llm_artifact.json"
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "micro_llm_artifact_pack.py"),
        "--output-dir",
        str(output_dir),
        "--artifact-id",
        args.artifact_id,
        "--version",
        str(args.version),
        "--json-out",
        str(artifact_json),
    ]
    if args.inspect:
        command.append("--inspect")
    step, payload = run_json_step(
        "micro_llm_artifact",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
    )
    payload = sanitize(payload) if payload else {}
    step["ok"] = bool(step.get("ok") and payload.get("ok"))
    summary = sanitize({
        "schema": MICRO_LLM_ARTIFACT_CLI_SCHEMA,
        "generated_at": utc_now(),
        "ok": bool(step.get("ok")),
        "output_dir": str(output_dir),
        "artifact_schema": payload.get("schema") or "micro_llm_artifact_v1",
        "artifact_id": payload.get("artifact_id"),
        "artifact_hash": payload.get("artifact_hash"),
        "artifact_version": payload.get("artifact_version"),
        "manifest_path": payload.get("manifest_path"),
        "step": step,
        "diagnosis_codes": ["micro_llm_artifact_ready"] if step.get("ok") else ["micro_llm_artifact_failed"],
        "artifacts": {
            "micro_llm_artifact_manifest": artifact_entry(output_dir / "manifest.json", output_dir, kind="micro_llm_artifact_manifest", schema="micro_llm_artifact_v1", ok=bool(step.get("ok"))),
            "micro_llm_artifact_json": artifact_entry(artifact_json, output_dir, kind="micro_llm_artifact", schema="micro_llm_artifact_v1", ok=payload.get("ok") if payload else None),
            "micro_llm_artifact_cli_summary": artifact_entry(summary_json, output_dir, kind="micro_llm_artifact_cli_summary", schema=MICRO_LLM_ARTIFACT_CLI_SCHEMA, ok=bool(step.get("ok"))),
        },
        "limitations": [
            "Dependency-free tiny Micro-LLM artifact; not a HF, GGUF, llama.cpp, or large-model artifact",
            "CPU-only/read-only proof boundary when used with micro-llm-shard-infer",
        ],
    })
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["artifacts"]["micro_llm_artifact_cli_summary"]["present"] = True
    return summary


def build_remote_sharded_inference_beta(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "remote_sharded_inference_beta_pack.py"),
        "--mode",
        args.mode,
        "--output-dir",
        str(output_dir),
        "--base-port",
        str(args.base_port),
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
        "--failure-mode",
        args.failure_mode,
        "--stage-mode",
        args.stage_mode,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
    ]
    if args.coordinator_url:
        command.extend(["--coordinator-url", args.coordinator_url])
    if args.observer_token:
        command.extend(["--observer-token", args.observer_token])
    if args.admin_token:
        command.extend(["--admin-token", args.admin_token])
    if args.require_distinct_stage_miners:
        command.append("--require-distinct-stage-miners")
    command.append("--json")
    secret_values = [args.observer_token, args.admin_token]
    step, payload = run_json_step(
        "remote_sharded_inference_beta",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
        redact_secrets=secret_values,
    )
    if payload:
        payload = sanitize(redact_values(payload, secret_values))
        payload.setdefault("cli_schema", REMOTE_SHARDED_INFERENCE_BETA_CLI_SCHEMA)
        encoded = json.dumps(payload, sort_keys=True)
        if any(secret and secret in encoded for secret in secret_values):
            payload["ok"] = False
            payload.setdefault("diagnosis_codes", [])
            if "sensitive_output_detected" not in payload["diagnosis_codes"]:
                payload["diagnosis_codes"].append("sensitive_output_detected")
        return payload
    return sanitize({
        "schema": "remote_sharded_inference_beta_v1",
        "cli_schema": REMOTE_SHARDED_INFERENCE_BETA_CLI_SCHEMA,
        "ok": False,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["remote_sharded_inference_failed"],
        "limitations": [
            "CPU-only two-stage pipeline-sharded inference Beta; not production Swarm Inference",
            "Does not provide GPU/TPU pooling, P2P routing, NAT traversal, real LLM sharding, or arbitrary prompt serving",
        ],
    })


def build_remote_micro_llm_sharded_beta(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "remote_micro_llm_sharded_beta_pack.py"),
        "--mode",
        args.mode,
        "--output-dir",
        str(output_dir),
        "--base-port",
        str(args.base_port),
        "--request-count",
        str(args.request_count),
        "--decode-steps",
        str(args.decode_steps),
        "--failure-mode",
        args.failure_mode,
        "--stage-mode",
        args.stage_mode,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
    ]
    if args.coordinator_url:
        command.extend(["--coordinator-url", args.coordinator_url])
    if args.observer_token:
        command.extend(["--observer-token", args.observer_token])
    if args.admin_token:
        command.extend(["--admin-token", args.admin_token])
    if args.require_distinct_stage_miners:
        command.append("--require-distinct-stage-miners")
    command.append("--json")
    secret_values = [args.observer_token, args.admin_token]
    step, payload = run_json_step(
        "remote_micro_llm_sharded_beta",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
        redact_secrets=secret_values,
    )
    if payload:
        payload = sanitize(redact_values(payload, secret_values))
        payload.setdefault("cli_schema", REMOTE_MICRO_LLM_SHARDED_BETA_CLI_SCHEMA)
        encoded = json.dumps(payload, sort_keys=True)
        if any(secret and secret in encoded for secret in secret_values):
            payload["ok"] = False
            payload.setdefault("diagnosis_codes", [])
            if "sensitive_output_detected" not in payload["diagnosis_codes"]:
                payload["diagnosis_codes"].append("sensitive_output_detected")
        return payload
    return sanitize({
        "schema": "remote_micro_llm_sharded_beta_v1",
        "cli_schema": REMOTE_MICRO_LLM_SHARDED_BETA_CLI_SCHEMA,
        "ok": False,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["remote_micro_llm_sharded_failed"],
        "limitations": [
            "CPU-only deterministic micro-LLM pipeline-sharded inference Beta; not production Swarm Inference",
            "Does not provide GPU/TPU pooling, P2P routing, GGUF/llama.cpp serving, or arbitrary prompt serving",
        ],
    })


def build_remote_real_llm_sharded_beta(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "remote_real_llm_sharded_beta_pack.py"),
        "--mode",
        args.mode,
        "--output-dir",
        str(output_dir),
        "--base-port",
        str(args.base_port),
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(getattr(args, "max_new_tokens", 1)),
        "--hf-model-id",
        str(args.hf_model_id),
        "--real-llm-partition-mode",
        str(getattr(args, "real_llm_partition_mode", "full")),
        "--failure-mode",
        args.failure_mode,
        "--stage-mode",
        args.stage_mode,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.prompt_texts:
        command.extend(["--prompt-texts", args.prompt_texts])
    if args.coordinator_url:
        command.extend(["--coordinator-url", args.coordinator_url])
    if args.observer_token:
        command.extend(["--observer-token", args.observer_token])
    if args.admin_token:
        command.extend(["--admin-token", args.admin_token])
    if args.require_distinct_stage_miners:
        command.append("--require-distinct-stage-miners")
    command.append("--json")
    secret_values = [args.observer_token, args.admin_token]
    step, payload = run_json_step(
        "remote_real_llm_sharded_beta",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
        redact_secrets=secret_values,
    )
    if payload:
        payload = sanitize(redact_values(payload, secret_values))
        payload.setdefault("cli_schema", REMOTE_REAL_LLM_SHARDED_BETA_CLI_SCHEMA)
        encoded = json.dumps(payload, sort_keys=True)
        if any(secret and secret in encoded for secret in secret_values):
            payload["ok"] = False
            payload.setdefault("diagnosis_codes", [])
            if "sensitive_output_detected" not in payload["diagnosis_codes"]:
                payload["diagnosis_codes"].append("sensitive_output_detected")
        return payload
    return sanitize({
        "schema": "remote_real_llm_sharded_beta_v1",
        "cli_schema": REMOTE_REAL_LLM_SHARDED_BETA_CLI_SCHEMA,
        "ok": False,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["remote_real_llm_sharded_failed"],
        "limitations": [
            "CPU-only tiny Hugging Face GPT pipeline-sharded inference Beta; not production Swarm Inference",
            "Does not provide GPU/TPU pooling, P2P routing, GGUF/llama.cpp serving, or arbitrary prompt serving",
        ],
    })


def build_micro_llm_live_rc(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "micro_llm_live_rc_pack.py"),
        "--mode",
        args.mode,
        "--output-dir",
        str(output_dir),
        "--public-host",
        args.public_host,
        "--port",
        str(args.port),
        "--miner-id",
        args.miner_id,
        "--request-count",
        str(args.request_count),
        "--decode-steps",
        str(args.decode_steps),
        "--micro-llm-artifact",
        str(args.micro_llm_artifact),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--startup-timeout",
        str(args.startup_timeout),
        "--process-exit-timeout",
        str(args.process_exit_timeout),
        "--poll-interval",
        str(args.poll_interval),
        "--http-timeout",
        str(args.http_timeout),
        "--artifact-timeout",
        str(args.artifact_timeout),
        "--admin-results-limit",
        str(args.admin_results_limit),
        "--lease-seconds",
        str(args.lease_seconds),
        "--compute-seconds",
        str(args.compute_seconds),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--idle-sleep",
        str(args.idle_sleep),
        "--max-request-attempts",
        str(args.max_request_attempts),
        "--failure-mode",
        getattr(args, "failure_mode", "none"),
        "--victim-compute-seconds",
        str(getattr(args, "victim_compute_seconds", 45.0)),
        "--claim-observe-timeout",
        str(getattr(args, "claim_observe_timeout", 180.0)),
        "--requeue-timeout",
        str(getattr(args, "requeue_timeout", 120.0)),
        "--json",
    ]
    if args.kaggle_output_dir:
        command.extend(["--kaggle-output-dir", args.kaggle_output_dir])
    if args.coordinator_url:
        command.extend(["--coordinator-url", args.coordinator_url])
    secret_values = [args.observer_token, args.admin_token]
    if args.observer_token:
        command.extend(["--observer-token", args.observer_token])
    if args.admin_token:
        command.extend(["--admin-token", args.admin_token])
    step, payload = run_json_step(
        "micro_llm_live_rc",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
        redact_secrets=secret_values,
    )
    if payload:
        payload = sanitize(redact_values(payload, secret_values))
        payload.setdefault("cli_schema", MICRO_LLM_LIVE_RC_CLI_SCHEMA)
        encoded = json.dumps(payload, sort_keys=True)
        if any(secret and secret in encoded for secret in secret_values):
            payload["ok"] = False
            payload.setdefault("diagnosis_codes", [])
            if "sensitive_output_detected" not in payload["diagnosis_codes"]:
                payload["diagnosis_codes"].append("sensitive_output_detected")
        return payload
    return sanitize({
        "schema": "micro_llm_live_rc_v1",
        "cli_schema": MICRO_LLM_LIVE_RC_CLI_SCHEMA,
        "ok": False,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["micro_llm_live_rc_blocked"],
        "limitations": [
            "CPU-only read-only micro-LLM live two-node RC; not production Swarm Inference",
            "Does not provide GPU/TPU pooling, P2P routing, NAT traversal, GGUF/llama.cpp serving, large-model sharding, or arbitrary prompt serving",
        ],
    })


def build_real_llm_live_rc(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "real_llm_live_rc_pack.py"),
        "--mode",
        args.mode,
        "--output-dir",
        str(output_dir),
        "--public-host",
        args.public_host,
        "--bind-host",
        args.bind_host,
        "--port",
        str(args.port),
        "--miner-id",
        args.miner_id,
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(getattr(args, "max_new_tokens", 1)),
        "--hf-model-id",
        args.hf_model_id,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--startup-timeout",
        str(args.startup_timeout),
        "--process-exit-timeout",
        str(args.process_exit_timeout),
        "--poll-interval",
        str(args.poll_interval),
        "--http-timeout",
        str(args.http_timeout),
        "--lease-seconds",
        str(args.lease_seconds),
        "--compute-seconds",
        str(args.compute_seconds),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--idle-sleep",
        str(args.idle_sleep),
        "--max-request-attempts",
        str(args.max_request_attempts),
        "--failure-mode",
        getattr(args, "failure_mode", "none"),
        "--victim-compute-seconds",
        str(getattr(args, "victim_compute_seconds", 45.0)),
        "--claim-observe-timeout",
        str(getattr(args, "claim_observe_timeout", 180.0)),
        "--requeue-timeout",
        str(getattr(args, "requeue_timeout", 120.0)),
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.coordinator_url:
        command.extend(["--coordinator-url", args.coordinator_url])
    secret_values = [args.observer_token, args.admin_token]
    if args.observer_token:
        command.extend(["--observer-token", args.observer_token])
    if args.admin_token:
        command.extend(["--admin-token", args.admin_token])
    step, payload = run_json_step(
        "real_llm_live_rc",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
        redact_secrets=secret_values,
    )
    if payload:
        payload = sanitize(redact_values(payload, secret_values))
        payload.setdefault("cli_schema", REAL_LLM_LIVE_RC_CLI_SCHEMA)
        encoded = json.dumps(payload, sort_keys=True)
        if any(secret and secret in encoded for secret in secret_values):
            payload["ok"] = False
            payload.setdefault("diagnosis_codes", [])
            if "sensitive_output_detected" not in payload["diagnosis_codes"]:
                payload["diagnosis_codes"].append("sensitive_output_detected")
        return payload
    return sanitize({
        "schema": "real_llm_live_rc_v1",
        "cli_schema": REAL_LLM_LIVE_RC_CLI_SCHEMA,
        "ok": False,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["real_llm_live_rc_blocked"],
        "limitations": [
            "CPU-only read-only real small-LLM live two-node RC; not production Swarm Inference",
            "Does not provide GPU/TPU pooling, P2P routing, NAT traversal, GGUF/llama.cpp serving, large-model serving, training, or arbitrary prompt serving",
        ],
    })


def build_real_llm_internet_alpha(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "real_llm_internet_alpha_pack.py"),
        "--mode",
        args.mode,
        "--output-dir",
        str(output_dir),
        "--public-host",
        args.public_host,
        "--bind-host",
        args.bind_host,
        "--port",
        str(args.port),
        "--base-port",
        str(args.base_port),
        "--miner-id",
        args.miner_id,
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(getattr(args, "max_new_tokens", 1)),
        "--hf-model-id",
        args.hf_model_id,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--startup-timeout",
        str(args.startup_timeout),
        "--process-exit-timeout",
        str(args.process_exit_timeout),
        "--poll-interval",
        str(args.poll_interval),
        "--http-timeout",
        str(args.http_timeout),
        "--lease-seconds",
        str(args.lease_seconds),
        "--compute-seconds",
        str(args.compute_seconds),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--idle-sleep",
        str(args.idle_sleep),
        "--max-request-attempts",
        str(args.max_request_attempts),
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.coordinator_url:
        command.extend(["--coordinator-url", args.coordinator_url])
    secret_values = [args.observer_token, args.admin_token]
    if args.observer_token:
        command.extend(["--observer-token", args.observer_token])
    if args.admin_token:
        command.extend(["--admin-token", args.admin_token])
    if args.skip_requeue:
        command.append("--skip-requeue")
    step, payload = run_json_step(
        "real_llm_internet_alpha",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(float(args.timeout_seconds), 240.0) * 3 + 120.0,
        redact_secrets=secret_values,
    )
    if payload:
        payload = sanitize(redact_values(payload, secret_values))
        payload.setdefault("cli_schema", REAL_LLM_INTERNET_ALPHA_CLI_SCHEMA)
        encoded = json.dumps(payload, sort_keys=True)
        if any(secret and secret in encoded for secret in secret_values):
            payload["ok"] = False
            payload.setdefault("diagnosis_codes", [])
            if "sensitive_output_detected" not in payload["diagnosis_codes"]:
                payload["diagnosis_codes"].append("sensitive_output_detected")
        return payload
    return sanitize({
        "schema": "real_llm_internet_alpha_v1",
        "cli_schema": REAL_LLM_INTERNET_ALPHA_CLI_SCHEMA,
        "ok": False,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["real_llm_internet_alpha_blocked"],
        "limitations": [
            "CPU-only read-only real small-LLM Internet Alpha; not production Swarm Inference",
            "Does not provide GPU/TPU pooling, P2P routing, NAT traversal, GGUF/llama.cpp serving, large-model serving, training, or arbitrary prompt serving",
        ],
    })


def build_real_llm_internet_beta(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "real_llm_internet_beta_pack.py"),
        "--mode",
        args.mode,
        "--output-dir",
        str(output_dir),
        "--public-host",
        args.public_host,
        "--bind-host",
        args.bind_host,
        "--port",
        str(args.port),
        "--base-port",
        str(args.base_port),
        "--miner-id",
        args.miner_id,
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(getattr(args, "max_new_tokens", 1)),
        "--hf-model-id",
        args.hf_model_id,
        "--dataset-title",
        args.dataset_title,
        "--kernel-title-prefix",
        args.kernel_title_prefix,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--startup-timeout",
        str(args.startup_timeout),
        "--process-exit-timeout",
        str(args.process_exit_timeout),
        "--poll-interval",
        str(args.poll_interval),
        "--http-timeout",
        str(args.http_timeout),
        "--kaggle-push-timeout-seconds",
        str(args.kaggle_push_timeout_seconds),
        "--kaggle-delete-timeout-seconds",
        str(args.kaggle_delete_timeout_seconds),
        "--kaggle-status-timeout-seconds",
        str(args.kaggle_status_timeout_seconds),
        "--kaggle-status-poll-interval",
        str(args.kaggle_status_poll_interval),
        "--lease-seconds",
        str(args.lease_seconds),
        "--compute-seconds",
        str(args.compute_seconds),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--idle-sleep",
        str(args.idle_sleep),
        "--max-request-attempts",
        str(args.max_request_attempts),
        "--json",
    ]
    if args.ready_url:
        command.extend(["--ready-url", args.ready_url])
    if args.coordinator_url:
        command.extend(["--coordinator-url", args.coordinator_url])
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.kaggle_owner:
        command.extend(["--kaggle-owner", args.kaggle_owner])
    if args.dataset_slug:
        command.extend(["--dataset-slug", args.dataset_slug])
    if args.kernel_slug_prefix:
        command.extend(["--kernel-slug-prefix", args.kernel_slug_prefix])
    if not args.inline_kernel_payload:
        command.append("--no-inline-kernel-payload")
    if args.skip_kaggle_cleanup:
        command.append("--skip-kaggle-cleanup")
    step, payload = run_json_step(
        "real_llm_internet_beta",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), 300.0) + 600.0,
    )
    if payload:
        payload = sanitize(redact_values(payload, []))
        payload.setdefault("cli_schema", REAL_LLM_INTERNET_BETA_CLI_SCHEMA)
        return payload
    return sanitize({
        "schema": "real_llm_internet_beta_v1",
        "cli_schema": REAL_LLM_INTERNET_BETA_CLI_SCHEMA,
        "ok": False,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["real_llm_internet_beta_blocked"],
        "limitations": [
            "CPU-only read-only real small-LLM Internet Beta; not production Swarm Inference",
            "Does not provide GPU/TPU pooling, P2P routing, NAT traversal, GGUF/llama.cpp serving, large-model serving, training, or arbitrary prompt serving",
        ],
    })


def build_swarm_inference_beta(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "swarm_inference_beta_pack.py"),
        args.swarm_action,
        "--output-dir",
        str(output_dir),
        "--json",
    ]
    if args.swarm_action == "live":
        command.extend([
            "--public-host",
            args.public_host,
            "--bind-host",
            args.bind_host,
            "--port",
            str(args.port),
            "--base-port",
            str(args.base_port),
            "--miner-id-prefix",
            args.miner_id_prefix,
            "--request-count",
            str(args.request_count),
            "--hf-model-id",
            args.hf_model_id,
            "--dataset-title",
            args.dataset_title,
            "--kernel-title-prefix",
            args.kernel_title_prefix,
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--remote-timeout-seconds",
            str(args.remote_timeout_seconds),
            "--startup-timeout",
            str(args.startup_timeout),
            "--process-exit-timeout",
            str(args.process_exit_timeout),
            "--poll-interval",
            str(args.poll_interval),
            "--http-timeout",
            str(args.http_timeout),
            "--kaggle-push-timeout-seconds",
            str(args.kaggle_push_timeout_seconds),
            "--kaggle-delete-timeout-seconds",
            str(args.kaggle_delete_timeout_seconds),
            "--kaggle-status-timeout-seconds",
            str(args.kaggle_status_timeout_seconds),
            "--kaggle-status-poll-interval",
            str(args.kaggle_status_poll_interval),
            "--lease-seconds",
            str(args.lease_seconds),
            "--compute-seconds",
            str(args.compute_seconds),
            "--heartbeat-interval",
            str(args.heartbeat_interval),
            "--idle-sleep",
            str(args.idle_sleep),
            "--max-request-attempts",
            str(args.max_request_attempts),
            "--failure-mode",
            getattr(args, "failure_mode", "none"),
            "--victim-compute-seconds",
            str(getattr(args, "victim_compute_seconds", 45.0)),
            "--claim-observe-timeout",
            str(getattr(args, "claim_observe_timeout", 180.0)),
            "--requeue-timeout",
            str(getattr(args, "requeue_timeout", 120.0)),
        ])
        if args.ready_url:
            command.extend(["--ready-url", args.ready_url])
        if args.coordinator_url:
            command.extend(["--coordinator-url", args.coordinator_url])
        if args.hf_cache_dir:
            command.extend(["--hf-cache-dir", args.hf_cache_dir])
        if args.kaggle_owner:
            command.extend(["--kaggle-owner", args.kaggle_owner])
        if args.dataset_slug:
            command.extend(["--dataset-slug", args.dataset_slug])
        if args.kernel_slug_prefix:
            command.extend(["--kernel-slug-prefix", args.kernel_slug_prefix])
        if args.inline_kernel_payload:
            command.append("--inline-kernel-payload")
        else:
            command.append("--no-inline-kernel-payload")
        if args.skip_kaggle_cleanup:
            command.append("--skip-kaggle-cleanup")
        if args.keep_live_private_artifacts:
            command.append("--keep-live-private-artifacts")
    elif args.swarm_action != "clean":
        command.extend([
            "--coordinator-url",
            args.coordinator_url,
            "--port",
            str(args.port),
            "--bind-host",
            args.bind_host,
            "--miner-id-prefix",
            args.miner_id_prefix,
            "--request-count",
            str(args.request_count),
            "--hf-model-id",
            args.hf_model_id,
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--remote-timeout-seconds",
            str(args.remote_timeout_seconds),
            "--http-timeout",
            str(args.http_timeout),
        ])
        if args.hf_cache_dir:
            command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.swarm_action == "prepare":
        command.extend([
            "--lease-seconds",
            str(args.lease_seconds),
            "--compute-seconds",
            str(args.compute_seconds),
            "--heartbeat-interval",
            str(args.heartbeat_interval),
            "--max-request-attempts",
            str(args.max_request_attempts),
        ])
        if args.observer_token:
            command.extend(["--observer-token", args.observer_token])
        if args.admin_token:
            command.extend(["--admin-token", args.admin_token])
        if args.replace:
            command.append("--replace")
    elif args.swarm_action == "coordinator":
        command.extend(["--lease-seconds", str(args.lease_seconds)])
        if args.observer_token:
            command.extend(["--observer-token", args.observer_token])
        if args.admin_token:
            command.extend(["--admin-token", args.admin_token])
        if args.run:
            command.append("--run")
    elif args.swarm_action == "miner":
        command.extend([
            "--stage",
            args.stage,
            "--compute-seconds",
            str(args.compute_seconds),
            "--heartbeat-interval",
            str(args.heartbeat_interval),
            "--max-request-attempts",
            str(args.max_request_attempts),
        ])
        if args.run:
            command.append("--run")
    elif args.swarm_action == "verify":
        if args.observer_token:
            command.extend(["--observer-token", args.observer_token])
        if args.admin_token:
            command.extend(["--admin-token", args.admin_token])
        if args.prompt_texts:
            command.extend(["--prompt-texts", args.prompt_texts])
        if args.real_internet_beta_report:
            command.extend(["--real-internet-beta-report", args.real_internet_beta_report])
    elif args.swarm_action == "collect":
        if args.observer_token:
            command.extend(["--observer-token", args.observer_token])
        if args.admin_token:
            command.extend(["--admin-token", args.admin_token])
        if args.miner_id:
            command.extend(["--miner-id", args.miner_id])
        command.extend(["--artifact-timeout", str(args.artifact_timeout)])
    elif args.swarm_action == "clean":
        if args.apply:
            command.append("--apply")
        if args.include_private:
            command.append("--include-private")
        if args.remove_empty_dir:
            command.append("--remove-empty-dir")
    secret_values = [getattr(args, "observer_token", ""), getattr(args, "admin_token", "")]
    step, payload = run_json_step(
        "swarm_inference_beta",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(float(getattr(args, "timeout_seconds", 60.0)), float(getattr(args, "remote_timeout_seconds", 60.0)), 60.0) + (600.0 if args.swarm_action == "live" else 0.0),
        redact_secrets=secret_values,
    )
    if payload:
        payload = sanitize(redact_values(payload, secret_values))
        payload.setdefault("cli_schema", SWARM_INFERENCE_BETA_CLI_SCHEMA)
        encoded = json.dumps(payload, sort_keys=True)
        if any(secret and secret in encoded for secret in secret_values):
            payload["ok"] = False
            payload.setdefault("diagnosis_codes", [])
            if "sensitive_output_detected" not in payload["diagnosis_codes"]:
                payload["diagnosis_codes"].append("sensitive_output_detected")
        return payload
    return sanitize({
        "schema": "swarm_inference_beta_v1",
        "cli_schema": SWARM_INFERENCE_BETA_CLI_SCHEMA,
        "ok": False,
        "mode": args.swarm_action,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["swarm_inference_beta_failed"],
        "limitations": [
            "CPU-only read-only real tiny-LLM Swarm Inference Beta; not production Swarm Inference",
            "Does not provide P2P routing, NAT traversal, GPU/TPU pooling, GGUF/llama.cpp serving, large-model serving, training, payments, or staking",
        ],
    })


def build_public_swarm_inference_alpha(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "public_swarm_inference_alpha_pack.py"),
        "--mode",
        args.mode,
        "--output-dir",
        str(output_dir),
        "--public-host",
        args.public_host,
        "--bind-host",
        args.bind_host,
        "--port",
        str(args.port),
        "--base-port",
        str(args.base_port),
        "--miner-id-prefix",
        args.miner_id_prefix,
        "--request-count",
        str(args.request_count),
        "--hf-model-id",
        args.hf_model_id,
        "--failure-mode",
        args.failure_mode,
        "--dataset-title",
        args.dataset_title,
        "--kernel-title-prefix",
        args.kernel_title_prefix,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--startup-timeout",
        str(args.startup_timeout),
        "--process-exit-timeout",
        str(args.process_exit_timeout),
        "--poll-interval",
        str(args.poll_interval),
        "--http-timeout",
        str(args.http_timeout),
        "--kaggle-push-timeout-seconds",
        str(args.kaggle_push_timeout_seconds),
        "--kaggle-delete-timeout-seconds",
        str(args.kaggle_delete_timeout_seconds),
        "--kaggle-status-timeout-seconds",
        str(args.kaggle_status_timeout_seconds),
        "--kaggle-status-poll-interval",
        str(args.kaggle_status_poll_interval),
        "--lease-seconds",
        str(args.lease_seconds),
        "--compute-seconds",
        str(args.compute_seconds),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--idle-sleep",
        str(args.idle_sleep),
        "--max-request-attempts",
        str(args.max_request_attempts),
        "--victim-compute-seconds",
        str(getattr(args, "victim_compute_seconds", 45.0)),
        "--claim-observe-timeout",
        str(getattr(args, "claim_observe_timeout", 180.0)),
        "--requeue-timeout",
        str(getattr(args, "requeue_timeout", 120.0)),
        "--json",
    ]
    if args.ready_url:
        command.extend(["--ready-url", args.ready_url])
    if args.coordinator_url:
        command.extend(["--coordinator-url", args.coordinator_url])
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.kaggle_owner:
        command.extend(["--kaggle-owner", args.kaggle_owner])
    if args.dataset_slug:
        command.extend(["--dataset-slug", args.dataset_slug])
    if args.kernel_slug_prefix:
        command.extend(["--kernel-slug-prefix", args.kernel_slug_prefix])
    if args.inline_kernel_payload:
        command.append("--inline-kernel-payload")
    else:
        command.append("--no-inline-kernel-payload")
    if args.skip_kaggle_cleanup:
        command.append("--skip-kaggle-cleanup")
    if args.keep_live_private_artifacts:
        command.append("--keep-live-private-artifacts")
    if args.keep_child_artifacts:
        command.append("--keep-child-artifacts")
    if args.skip_local_requeue:
        command.append("--skip-local-requeue")
    step, payload = run_json_step(
        "public_swarm_inference_alpha",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), 300.0) + 1200.0,
    )
    if payload:
        payload = sanitize(redact_values(payload, []))
        payload.setdefault("cli_schema", PUBLIC_SWARM_INFERENCE_ALPHA_CLI_SCHEMA)
        return payload
    return sanitize({
        "schema": "public_swarm_inference_alpha_v1",
        "cli_schema": PUBLIC_SWARM_INFERENCE_ALPHA_CLI_SCHEMA,
        "ok": False,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["public_swarm_inference_alpha_failed"],
        "limitations": [
            "CPU-only read-only real tiny-LLM Public Swarm Inference Alpha; not production Swarm Inference",
            "Does not provide P2P routing, NAT traversal, GPU/TPU pooling, GGUF/llama.cpp serving, large-model serving, training, payments, or staking",
        ],
    })


def build_public_swarm_inference_alpha_rc(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "public_swarm_inference_alpha_rc_pack.py"),
        "--mode",
        args.mode,
        "--output-dir",
        str(output_dir),
        "--stage0-report",
        args.stage0_report,
        "--stage1-report",
        args.stage1_report,
        "--summary-report",
        args.summary_report,
        "--request-count",
        str(args.request_count),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--json",
    ]
    step, payload = run_json_step(
        "public_swarm_inference_alpha_rc",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=float(args.timeout_seconds) + 60.0,
    )
    if payload:
        payload = sanitize(redact_values(payload, []))
        payload.setdefault("cli_schema", PUBLIC_SWARM_INFERENCE_ALPHA_RC_CLI_SCHEMA)
        return payload
    return sanitize({
        "schema": "public_swarm_inference_alpha_rc_v1",
        "cli_schema": PUBLIC_SWARM_INFERENCE_ALPHA_RC_CLI_SCHEMA,
        "ok": False,
        "mode": args.mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["public_swarm_inference_alpha_rc_failed"],
        "limitations": [
            "CPU-only read-only Public Swarm Inference Alpha RC evidence; not production Swarm Inference",
            "Does not provide P2P routing, NAT traversal, GPU/TPU pooling, GGUF/llama.cpp serving, large-model serving, training, payments, or staking",
        ],
    })


def build_public_swarm_inference_beta(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    action = getattr(args, "public_swarm_beta_action", "local-loopback")
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "public_swarm_inference_beta_pack.py"),
        action,
        "--output-dir",
        str(output_dir),
        "--request-count",
        str(args.request_count),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--json",
    ]
    if action == "product-beta":
        command.extend([
            "--base-port",
            str(args.base_port),
            "--hf-model-id",
            args.hf_model_id,
            "--gpu-report",
            args.gpu_report,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--cpu-request-count",
            str(args.cpu_request_count),
            "--external-llm-request-count",
            str(args.external_llm_request_count),
            "--scenario-id",
            args.scenario_id,
            "--cpu-timeout-seconds",
            str(args.cpu_timeout_seconds),
        ])
    if action in {"prepare", "coordinator", "miner", "verify", "collect", "local-loopback"}:
        command.extend([
            "--coordinator-url",
            args.coordinator_url,
            "--port",
            str(args.port),
            "--base-port",
            str(args.base_port),
            "--bind-host",
            args.bind_host,
            "--miner-id-prefix",
            args.miner_id_prefix,
            "--hf-model-id",
            args.hf_model_id,
            "--remote-timeout-seconds",
            str(args.remote_timeout_seconds),
            "--http-timeout",
            str(args.http_timeout),
        ])
        if args.hf_cache_dir:
            command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if action in {"prepare", "coordinator", "miner"}:
        command.extend(["--lease-seconds", str(args.lease_seconds)])
    if action in {"prepare", "miner"}:
        command.extend([
            "--compute-seconds",
            str(args.compute_seconds),
            "--heartbeat-interval",
            str(args.heartbeat_interval),
            "--max-request-attempts",
            str(args.max_request_attempts),
        ])
    if action in {"prepare", "coordinator", "verify", "collect"}:
        if args.observer_token:
            command.extend(["--observer-token", args.observer_token])
        if args.admin_token:
            command.extend(["--admin-token", args.admin_token])
    if action == "prepare" and args.replace:
        command.append("--replace")
    if action == "coordinator" and args.run:
        command.append("--run")
    if action == "miner":
        command.extend(["--stage", args.stage])
        if args.run:
            command.append("--run")
    if action in {"verify", "local-loopback"} and args.prompt_texts:
        command.extend(["--prompt-texts", args.prompt_texts])
    if action == "verify" and args.real_internet_beta_report:
        command.extend(["--real-internet-beta-report", args.real_internet_beta_report])
    if action == "collect":
        if args.miner_id:
            command.extend(["--miner-id", args.miner_id])
        command.extend(["--artifact-timeout", str(args.artifact_timeout)])
    if action == "clean":
        if args.apply:
            command.append("--apply")
        if args.include_private:
            command.append("--include-private")
        if args.remove_empty_dir:
            command.append("--remove-empty-dir")
    if action == "evidence-import":
        command.extend([
            "--alpha-rc-report",
            args.alpha_rc_report,
            "--stage0-report",
            args.stage0_report,
            "--stage1-report",
            args.stage1_report,
            "--summary-report",
            args.summary_report,
        ])
        if args.allow_missing_live_evidence:
            command.append("--allow-missing-live-evidence")
    secret_values = [getattr(args, "observer_token", ""), getattr(args, "admin_token", "")]
    step, payload = run_json_step(
        "public_swarm_inference_beta",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(float(args.timeout_seconds), float(getattr(args, "remote_timeout_seconds", 60.0)), 60.0) + 180.0,
        redact_secrets=secret_values,
    )
    if payload:
        payload = sanitize(redact_values(payload, secret_values))
        payload.setdefault("cli_schema", PUBLIC_SWARM_INFERENCE_BETA_CLI_SCHEMA)
        encoded = json.dumps(payload, sort_keys=True)
        if any(secret and secret in encoded for secret in secret_values):
            payload["ok"] = False
            payload.setdefault("diagnosis_codes", [])
            if "sensitive_output_detected" not in payload["diagnosis_codes"]:
                payload["diagnosis_codes"].append("sensitive_output_detected")
        return payload
    return sanitize({
        "schema": "public_swarm_inference_beta_v1",
        "cli_schema": PUBLIC_SWARM_INFERENCE_BETA_CLI_SCHEMA,
        "ok": False,
        "mode": action,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["public_swarm_inference_beta_failed"],
        "limitations": [
            "CPU-only read-only Public Swarm Inference Beta; not production Swarm Inference",
            "Does not provide P2P routing, NAT traversal, GPU/TPU pooling, GGUF/llama.cpp serving, large-model serving, training, payments, or staking",
        ],
    })


def build_public_swarm_inference_beta_rc(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = getattr(args, "public_swarm_beta_rc_mode", "local-loopback")
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "public_swarm_inference_beta_rc_pack.py"),
        mode,
        "--output-dir",
        str(output_dir),
        "--base-port",
        str(args.base_port),
        "--port",
        str(args.port),
        "--public-host",
        args.public_host,
        "--bind-host",
        args.bind_host,
        "--target",
        args.target,
        "--miner-id-prefix",
        args.miner_id_prefix,
        "--hf-model-id",
        args.hf_model_id,
        "--gpu-report",
        args.gpu_report,
        "--prompt-text",
        args.prompt_text,
        "--scenario-id",
        args.scenario_id,
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--cpu-request-count",
        str(args.cpu_request_count),
        "--external-llm-request-count",
        str(args.external_llm_request_count),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--cpu-timeout-seconds",
        str(args.cpu_timeout_seconds),
        "--startup-timeout",
        str(args.startup_timeout),
        "--process-exit-timeout",
        str(args.process_exit_timeout),
        "--poll-interval",
        str(args.poll_interval),
        "--http-timeout",
        str(args.http_timeout),
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.coordinator_url:
        command.extend(["--coordinator-url", args.coordinator_url])
    if args.observer_token:
        command.extend(["--observer-token", args.observer_token])
    if args.admin_token:
        command.extend(["--admin-token", args.admin_token])
    secret_values = [getattr(args, "observer_token", ""), getattr(args, "admin_token", "")]
    step, payload = run_json_step(
        "public_swarm_inference_beta_rc",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), float(args.cpu_timeout_seconds), 60.0) + 300.0,
        redact_secrets=secret_values,
    )
    if payload:
        payload = sanitize(redact_values(payload, secret_values))
        payload.setdefault("cli_schema", PUBLIC_SWARM_INFERENCE_BETA_RC_CLI_SCHEMA)
        encoded = json.dumps(payload, sort_keys=True)
        if any(secret and secret in encoded for secret in secret_values):
            payload["ok"] = False
            payload.setdefault("diagnosis_codes", [])
            if "sensitive_output_detected" not in payload["diagnosis_codes"]:
                payload["diagnosis_codes"].append("sensitive_output_detected")
        return payload
    return sanitize({
        "schema": "public_swarm_inference_beta_rc_v1",
        "cli_schema": PUBLIC_SWARM_INFERENCE_BETA_RC_CLI_SCHEMA,
        "ok": False,
        "mode": mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["public_swarm_inference_beta_rc_failed"],
        "limitations": [
            "Coordinator-backed Public Swarm Inference Beta RC; not production Swarm Inference",
            "Does not provide libp2p, DHT, NAT traversal, GPU marketplace, large-model serving, training, payments, or staking",
        ],
    })


def build_public_swarm_product_beta(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = getattr(args, "public_swarm_product_beta_mode", "local-loopback")
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "public_swarm_product_beta_pack.py"),
        mode,
        "--output-dir",
        str(output_dir),
        "--base-port",
        str(args.base_port),
        "--port",
        str(args.port),
        "--public-host",
        args.public_host,
        "--bind-host",
        args.bind_host,
        "--target",
        args.target,
        "--miner-id-prefix",
        args.miner_id_prefix,
        "--hf-model-id",
        args.hf_model_id,
        "--gpu-report",
        args.gpu_report,
        "--prompt-text",
        args.prompt_text,
        "--scenario-id",
        args.scenario_id,
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--cpu-request-count",
        str(args.cpu_request_count),
        "--external-llm-request-count",
        str(args.external_llm_request_count),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--cpu-timeout-seconds",
        str(args.cpu_timeout_seconds),
        "--startup-timeout",
        str(args.startup_timeout),
        "--process-exit-timeout",
        str(args.process_exit_timeout),
        "--poll-interval",
        str(args.poll_interval),
        "--http-timeout",
        str(args.http_timeout),
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.coordinator_url:
        command.extend(["--coordinator-url", args.coordinator_url])
    if args.observer_token:
        command.extend(["--observer-token", args.observer_token])
    if args.admin_token:
        command.extend(["--admin-token", args.admin_token])
    secret_values = [getattr(args, "observer_token", ""), getattr(args, "admin_token", "")]
    step, payload = run_json_step(
        "public_swarm_product_beta",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), float(args.cpu_timeout_seconds), 60.0) + 360.0,
        redact_secrets=secret_values,
    )
    if payload:
        payload = sanitize(redact_values(payload, secret_values))
        payload.setdefault("cli_schema", PUBLIC_SWARM_PRODUCT_BETA_CLI_SCHEMA)
        encoded = json.dumps(payload, sort_keys=True)
        if any(secret and secret in encoded for secret in secret_values):
            payload["ok"] = False
            payload.setdefault("diagnosis_codes", [])
            if "sensitive_output_detected" not in payload["diagnosis_codes"]:
                payload["diagnosis_codes"].append("sensitive_output_detected")
        return payload
    return sanitize({
        "schema": "public_swarm_product_beta_v1",
        "cli_schema": PUBLIC_SWARM_PRODUCT_BETA_CLI_SCHEMA,
        "ok": False,
        "mode": mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["public_swarm_product_beta_failed"],
        "limitations": [
            "Coordinator-backed Public Swarm Product Beta; not production Swarm Inference",
            "Does not provide libp2p, DHT, NAT traversal, GPU marketplace, large-model serving, training, payments, or staking",
        ],
    })


def build_public_swarm_gpu_inference_beta(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    action = getattr(args, "public_swarm_gpu_beta_action", "local-smoke")
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "public_swarm_gpu_inference_beta_pack.py"),
        action,
        "--output-dir",
        str(output_dir),
        "--request-count",
        str(args.request_count),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--json",
    ]
    if action in {"prepare", "coordinator", "miner", "verify", "collect", "local-loopback", "local-smoke", "kaggle-package", "kaggle-auto"}:
        command.extend([
            "--coordinator-url",
            args.coordinator_url,
            "--public-host",
            args.public_host,
            "--port",
            str(args.port),
            "--base-port",
            str(args.base_port),
            "--bind-host",
            args.bind_host,
            "--miner-id-prefix",
            args.miner_id_prefix,
            "--hf-model-id",
            args.hf_model_id,
            "--real-llm-partition-mode",
            getattr(args, "real_llm_partition_mode", "stage-local"),
            "--remote-timeout-seconds",
            str(args.remote_timeout_seconds),
            "--http-timeout",
            str(args.http_timeout),
        ])
        if args.hf_cache_dir:
            command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if action in {"prepare", "coordinator", "miner"}:
        command.extend(["--lease-seconds", str(args.lease_seconds)])
    if action == "kaggle-auto":
        command.extend([
            "--kaggle-owner",
            args.kaggle_owner,
            "--dataset-title",
            args.dataset_title,
            "--kernel-title-prefix",
            args.kernel_title_prefix,
            "--startup-timeout",
            str(args.startup_timeout),
            "--process-exit-timeout",
            str(args.process_exit_timeout),
            "--poll-interval",
            str(args.poll_interval),
            "--kaggle-push-timeout-seconds",
            str(args.kaggle_push_timeout_seconds),
            "--kaggle-delete-timeout-seconds",
            str(args.kaggle_delete_timeout_seconds),
            "--kaggle-status-timeout-seconds",
            str(args.kaggle_status_timeout_seconds),
            "--kaggle-status-poll-interval",
            str(args.kaggle_status_poll_interval),
        ])
        if args.dataset_slug:
            command.extend(["--dataset-slug", args.dataset_slug])
        if args.kernel_slug_prefix:
            command.extend(["--kernel-slug-prefix", args.kernel_slug_prefix])
        if args.inline_kernel_payload:
            command.append("--inline-kernel-payload")
        else:
            command.append("--no-inline-kernel-payload")
        if args.skip_kaggle_cleanup:
            command.append("--skip-kaggle-cleanup")
    if action in {"prepare", "miner"}:
        command.extend([
            "--compute-seconds",
            str(args.compute_seconds),
            "--heartbeat-interval",
            str(args.heartbeat_interval),
            "--max-request-attempts",
            str(args.max_request_attempts),
        ])
    if action == "kaggle-auto":
        command.extend([
            "--lease-seconds",
            str(args.lease_seconds),
            "--compute-seconds",
            str(args.compute_seconds),
            "--victim-compute-seconds",
            str(args.victim_compute_seconds),
            "--heartbeat-interval",
            str(args.heartbeat_interval),
            "--idle-sleep",
            str(args.idle_sleep),
            "--max-request-attempts",
            str(args.max_request_attempts),
        ])
    if action in {"prepare", "coordinator", "verify", "collect"}:
        if args.observer_token:
            command.extend(["--observer-token", args.observer_token])
        if args.admin_token:
            command.extend(["--admin-token", args.admin_token])
    if action == "prepare" and args.replace:
        command.append("--replace")
    if action == "coordinator" and args.run:
        command.append("--run")
    if action == "miner":
        command.extend(["--stage", args.stage])
        if args.run:
            command.append("--run")
    if action in {"verify", "local-loopback"} and args.prompt_texts:
        command.extend(["--prompt-texts", args.prompt_texts])
    if action == "collect":
        if args.miner_id:
            command.extend(["--miner-id", args.miner_id])
        command.extend(["--artifact-timeout", str(args.artifact_timeout)])
    if action == "clean":
        if args.apply:
            command.append("--apply")
        if args.include_private:
            command.append("--include-private")
        if args.remove_empty_dir:
            command.append("--remove-empty-dir")
    if action == "evidence-import":
        command.extend(["--gpu-report", args.gpu_report])
    secret_values = [getattr(args, "observer_token", ""), getattr(args, "admin_token", "")]
    step, payload = run_json_step(
        "public_swarm_gpu_inference_beta",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(float(args.timeout_seconds), float(getattr(args, "remote_timeout_seconds", 60.0)), 60.0) + 240.0,
        redact_secrets=secret_values,
    )
    if payload:
        payload = sanitize(redact_values(payload, secret_values))
        payload.setdefault("cli_schema", PUBLIC_SWARM_GPU_INFERENCE_BETA_CLI_SCHEMA)
        encoded = json.dumps(payload, sort_keys=True)
        if any(secret and secret in encoded for secret in secret_values):
            payload["ok"] = False
            payload.setdefault("diagnosis_codes", [])
            if "sensitive_output_detected" not in payload["diagnosis_codes"]:
                payload["diagnosis_codes"].append("sensitive_output_detected")
        return payload
    return sanitize({
        "schema": "public_swarm_gpu_inference_beta_v1",
        "cli_schema": PUBLIC_SWARM_GPU_INFERENCE_BETA_CLI_SCHEMA,
        "ok": False,
        "mode": action,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["public_swarm_gpu_inference_beta_failed"],
        "limitations": [
            "Optional CUDA read-only Public Swarm GPU Inference Beta; not production Swarm Inference",
            "Does not provide P2P routing, NAT traversal, GPU pooling marketplace, GGUF/llama.cpp serving, large-model serving, training, payments, or staking",
        ],
    })


def build_gpu_sharded_generation_beta(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = getattr(args, "gpu_generate_mode", "local-loopback")
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "gpu_sharded_generation_beta_pack.py"),
        mode,
        "--output-dir",
        str(output_dir),
        "--request-count",
        str(args.request_count),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--hf-model-id",
        args.hf_model_id,
        "--real-llm-partition-mode",
        args.real_llm_partition_mode,
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--remote-timeout-seconds",
        str(args.remote_timeout_seconds),
        "--public-host",
        args.public_host,
        "--port",
        str(args.port),
        "--base-port",
        str(args.base_port),
        "--bind-host",
        args.bind_host,
        "--miner-id-prefix",
        args.miner_id_prefix,
        "--json",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if mode == "kaggle-auto":
        command.extend([
            "--kaggle-owner",
            args.kaggle_owner,
            "--dataset-title",
            args.dataset_title,
            "--kernel-title-prefix",
            args.kernel_title_prefix,
            "--startup-timeout",
            str(args.startup_timeout),
            "--process-exit-timeout",
            str(args.process_exit_timeout),
            "--poll-interval",
            str(args.poll_interval),
            "--http-timeout",
            str(args.http_timeout),
            "--kaggle-push-timeout-seconds",
            str(args.kaggle_push_timeout_seconds),
            "--kaggle-delete-timeout-seconds",
            str(args.kaggle_delete_timeout_seconds),
            "--kaggle-status-timeout-seconds",
            str(args.kaggle_status_timeout_seconds),
            "--kaggle-status-poll-interval",
            str(args.kaggle_status_poll_interval),
            "--lease-seconds",
            str(args.lease_seconds),
            "--compute-seconds",
            str(args.compute_seconds),
            "--victim-compute-seconds",
            str(args.victim_compute_seconds),
            "--heartbeat-interval",
            str(args.heartbeat_interval),
            "--idle-sleep",
            str(args.idle_sleep),
            "--max-request-attempts",
            str(args.max_request_attempts),
        ])
        if args.dataset_slug:
            command.extend(["--dataset-slug", args.dataset_slug])
        if args.kernel_slug_prefix:
            command.extend(["--kernel-slug-prefix", args.kernel_slug_prefix])
        command.append("--inline-kernel-payload" if args.inline_kernel_payload else "--no-inline-kernel-payload")
        if args.skip_kaggle_cleanup:
            command.append("--skip-kaggle-cleanup")
    if mode == "evidence-import":
        command.extend(["--gpu-report", args.gpu_report])
    step, payload = run_json_step(
        "gpu_sharded_generation_beta",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=max(float(args.timeout_seconds), float(args.remote_timeout_seconds), float(getattr(args, "kaggle_status_timeout_seconds", 60.0))) + 600.0,
    )
    if payload:
        payload.setdefault("cli_schema", GPU_SHARDED_GENERATION_BETA_CLI_SCHEMA)
        return payload
    return sanitize({
        "schema": "gpu_sharded_generation_beta_v1",
        "cli_schema": GPU_SHARDED_GENERATION_BETA_CLI_SCHEMA,
        "ok": False,
        "mode": mode,
        "output_dir": str(output_dir),
        "step": step,
        "diagnosis_codes": ["gpu_sharded_generation_blocked"],
        "limitations": [
            "Tiny GPT CUDA sharded generation Beta; not production Swarm Inference",
            "Does not provide P2P routing, GPU marketplace, large-model serving, training, or arbitrary public prompt serving",
        ],
    })


def _serve_task_lane(args: argparse.Namespace) -> str:
    if args.profile == "gpu-generation":
        return f"python-cli:cuda:0:real_llm_sharded_infer"
    return f"python-cli:cpu:0:real_llm_sharded_infer"


def build_serve_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "coordinator.py"),
        "--host",
        args.bind_host,
        "--port",
        str(args.port),
        "--state-dir",
        args.state_dir,
        "--backlog",
        "0",
        "--task-lane",
        _serve_task_lane(args),
        "--admin-token",
        args.admin_token,
        "--real-llm-model-id",
        args.hf_model_id,
        "--real-llm-backend",
        "hf_transformers_cuda" if args.profile == "gpu-generation" else "hf_transformers_cpu",
        "--real-llm-partition-mode",
        "stage-local",
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.miner_token:
        command.extend(["--miner-token", args.miner_token])
    if args.observer_token:
        command.extend(["--observer-token", args.observer_token])
    return command


def build_product_serve(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    command = build_serve_command(args)
    public_bind = args.bind_host in {"0.0.0.0", "::"}
    if public_bind and not args.i_understand_public_bind:
        return sanitize({
            "schema": PUBLIC_SWARM_PRODUCT_CLI_SCHEMA,
            "ok": False,
            "mode": "serve",
            "profile": args.profile,
            "command": command,
            "diagnosis_codes": ["public_bind_requires_explicit_ack"],
            "safety": {"public_bind_requires_explicit_ack": True},
        })
    report = {
        "schema": PUBLIC_SWARM_PRODUCT_CLI_SCHEMA,
        "ok": True,
        "mode": "serve",
        "profile": args.profile,
        "coordinator_url": f"http://{args.public_host}:{args.port}",
        "command": redacted_command(command, {"--admin-token", "--miner-token", "--observer-token"}),
        "printed_only": not args.run,
        "diagnosis_codes": ["serve_command_ready"],
        "safety": {
            "admin_token_from_env_supported": bool(os.environ.get("CROWDTENSOR_ADMIN_TOKEN")),
            "public_bind_explicit": public_bind,
            "not_production": True,
            "not_p2p_task_execution": True,
        },
    }
    if not args.run:
        return sanitize(report)
    completed = runner(command, cwd=str(ROOT), text=True)
    report["returncode"] = completed.returncode
    report["ok"] = completed.returncode == 0
    return sanitize(report)


def build_join_command(args: argparse.Namespace, *, coordinator_url: str) -> list[str]:
    backend = "hf_transformers_cuda" if args.backend == "cuda" else "hf_transformers_cpu"
    command = [
        sys.executable,
        str(ROOT / "miner_cli.py"),
        "--coordinator",
        coordinator_url,
        "--miner-id",
        args.miner_id,
        "--enable-hf-tiny-gpt-runtime",
        "--real-llm-backend",
        backend,
        "--real-llm-stage-role",
        args.stage,
        "--real-llm-partition-mode",
        "stage-local",
        "--hf-model-id",
        args.hf_model_id,
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.miner_token:
        command.extend(["--miner-token", args.miner_token])
    if args.once:
        command.append("--once")
    if args.max_tasks > 0:
        command.extend(["--max-tasks", str(args.max_tasks)])
    return command


def resolve_coordinator_from_bootstrap(bootstrap_url: str, *, timeout: float = 5.0) -> tuple[str, list[dict[str, Any]]]:
    payload = fetch_peer_catalog(bootstrap_url, timeout=timeout)
    peers = payload.get("peers") if isinstance(payload, dict) else []
    peer_list = [peer for peer in peers if isinstance(peer, dict)]
    for peer in peer_list:
        if peer.get("role") != "coordinator":
            continue
        urls = peer.get("urls") if isinstance(peer.get("urls"), dict) else {}
        if urls.get("coordinator"):
            return str(urls["coordinator"]), peer_list
    return "", peer_list


def build_product_join(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    coordinator_url = args.coordinator_url
    peers: list[dict[str, Any]] = []
    if not coordinator_url and args.peer_bootstrap:
        coordinator_url, peers = resolve_coordinator_from_bootstrap(args.peer_bootstrap, timeout=args.http_timeout)
    if not coordinator_url:
        return sanitize({
            "schema": PUBLIC_SWARM_PRODUCT_CLI_SCHEMA,
            "ok": False,
            "mode": "join",
            "diagnosis_codes": ["coordinator_route_missing"],
        })
    command = build_join_command(args, coordinator_url=coordinator_url)
    report = {
        "schema": PUBLIC_SWARM_PRODUCT_CLI_SCHEMA,
        "ok": True,
        "mode": "join",
        "coordinator_url": coordinator_url,
        "peer_bootstrap_used": bool(args.peer_bootstrap),
        "peer_count": len(peers),
        "command": redacted_command(command, {"--miner-token"}),
        "printed_only": not args.run,
        "diagnosis_codes": ["join_command_ready"],
        "safety": {
            "tokens_redacted_in_report": True,
            "not_production": True,
            "not_p2p_task_execution": True,
        },
    }
    if args.run:
        completed = runner(command, cwd=str(ROOT), text=True)
        report["returncode"] = completed.returncode
        report["ok"] = completed.returncode == 0
    return sanitize(redact_values(report, [args.miner_token]))


def build_product_generate(args: argparse.Namespace) -> dict[str, Any]:
    prompt_text = str(args.prompt_text or "")
    session_request = build_session_request(
        prompt_text=prompt_text,
        backend=args.backend,
        stage_mode="split",
        max_new_tokens=args.max_new_tokens,
        scenario_id=args.scenario_id,
        route_source="peer-bootstrap" if args.peer_bootstrap else "coordinator-url",
    )
    coordinator_url = args.coordinator_url
    peers: list[dict[str, Any]] = []
    if args.peer_bootstrap:
        resolved_url, peers = resolve_coordinator_from_bootstrap(args.peer_bootstrap, timeout=args.http_timeout)
        coordinator_url = coordinator_url or resolved_url
    route = build_route_decision(session_request, coordinator_url=coordinator_url, peer_catalog=peers)
    if args.dry_run:
        return sanitize({
            "schema": PUBLIC_SWARM_PRODUCT_CLI_SCHEMA,
            "ok": bool(route.get("coordinator_url_present")),
            "mode": "generate",
            "dry_run": True,
            "session_request": session_request,
            "route": route,
            "diagnosis_codes": ["generate_dry_run_ready" if route.get("coordinator_url_present") else "coordinator_route_missing"],
        })
    if not coordinator_url:
        return sanitize({
            "schema": PUBLIC_SWARM_PRODUCT_CLI_SCHEMA,
            "ok": False,
            "mode": "generate",
            "session_request": session_request,
            "route": route,
            "diagnosis_codes": ["coordinator_route_missing"],
        })
    if not args.admin_token:
        return sanitize({
            "schema": PUBLIC_SWARM_PRODUCT_CLI_SCHEMA,
            "ok": False,
            "mode": "generate",
            "session_request": session_request,
            "route": route,
            "diagnosis_codes": ["admin_token_required"],
        })
    private_payload = coordinator_payload_for_request(session_request, prompt_text=prompt_text)
    try:
        session = request_json_url(
            "POST",
            coordinator_url,
            "/admin/inference-sessions",
            private_payload,
            admin_token=args.admin_token,
            timeout=args.http_timeout,
        )
    except Exception as exc:
        detail = str(exc)[:240]
        diagnosis = ["session_create_failed"]
        if isinstance(exc, HTTPError):
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            if body:
                detail = body[:240]
            if "requires optional Hugging Face dependencies" in body or "transformers" in body:
                diagnosis.append("hf_dependencies_missing")
                detail = "real_llm_sharded_infer requires optional Hugging Face dependencies; install with python -m pip install -e '.[hf]'"
        elif "requires optional Hugging Face dependencies" in str(exc) or "transformers" in str(exc):
            diagnosis.append("hf_dependencies_missing")
            detail = "real_llm_sharded_infer requires optional Hugging Face dependencies; install with python -m pip install -e '.[hf]'"
        return sanitize({
            "schema": PUBLIC_SWARM_PRODUCT_CLI_SCHEMA,
            "ok": False,
            "mode": "generate",
            "session_request": session_request,
            "route": route,
            "diagnosis_codes": diagnosis,
            "error": type(exc).__name__,
            "detail": detail,
        })
    result_row: dict[str, Any] | None = None
    deadline = time.monotonic() + args.timeout_seconds
    while time.monotonic() <= deadline:
        session_id = str(session.get("session_id") or "")
        query = f"/admin/results?status=accepted&workload_type=real_llm_sharded_infer&limit={args.admin_results_limit}"
        if session_id:
            query += f"&session_id={session_id}"
        try:
            ledger = request_json_url("GET", coordinator_url, query, admin_token=args.admin_token, timeout=args.http_timeout)
        except Exception:
            ledger = {}
        rows = ledger.get("results") if isinstance(ledger, dict) else []
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
                if int(validation.get("generated_token_count") or 0) >= args.max_new_tokens:
                    result_row = row
                    break
        if result_row:
            break
        time.sleep(args.poll_interval)
    generation = safe_generation_summary(result_row or {}, max_new_tokens=args.max_new_tokens)
    ok = bool(result_row and generation.get("multi_token_generation_ready"))
    report = {
        "schema": PUBLIC_SWARM_PRODUCT_CLI_SCHEMA,
        "ok": ok,
        "mode": "generate",
        "dry_run": False,
        "session_request": session_request,
        "route": route,
        "session": {
            "schema": session.get("schema"),
            "session_id": session.get("session_id"),
            "workload_type": session.get("workload_type"),
            "max_new_tokens": session.get("max_new_tokens"),
            "backend": session.get("backend"),
        },
        "generation": generation,
        "diagnosis_codes": ["public_swarm_generate_ready"] if ok else ["generation_timeout"],
        "safety": {
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "not_production": True,
            "not_p2p_task_execution": True,
        },
    }
    if args.include_output and result_row:
        report["local_output_note"] = "Raw generated text stays in Coordinator ledger; this public CLI summary only exposes hashes."
    return sanitize(redact_values(report, [args.admin_token]))


def build_peer_cli(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    action = args.peer_action
    if action == "check":
        command = [sys.executable, str(SCRIPTS_DIR / "p2p_lite_discovery_check.py"), "--json"]
        step, payload = run_json_step("p2p_lite_discovery_check", command, runner=runner, cwd=ROOT, timeout_seconds=args.timeout_seconds)
        return payload or sanitize({
            "schema": P2P_LITE_CLI_SCHEMA,
            "ok": False,
            "mode": "check",
            "step": step,
            "diagnosis_codes": ["p2p_lite_check_failed"],
        })
    if action == "daemon":
        command = [
            sys.executable,
            str(SCRIPTS_DIR / "p2p_lite_daemon.py"),
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--swarm-id",
            args.swarm_id,
            "--role",
            args.role,
            "--ttl-seconds",
            str(args.ttl_seconds),
        ]
        if args.peer_id:
            command.extend(["--peer-id", args.peer_id])
        if args.peer_url:
            command.extend(["--peer-url", args.peer_url])
        if args.coordinator_url:
            command.extend(["--coordinator-url", args.coordinator_url])
        if args.backend:
            command.extend(["--backend", args.backend])
        if args.stage_role:
            command.extend(["--stage-role", args.stage_role])
        for capability in args.stage_capability or []:
            command.extend(["--stage-capability", capability])
        for bootstrap in args.bootstrap or []:
            command.extend(["--bootstrap", bootstrap])
        if args.print_peer:
            command.append("--print-peer")
        if not args.run:
            return sanitize({
                "schema": P2P_LITE_CLI_SCHEMA,
                "ok": True,
                "mode": "daemon",
                "command": command,
                "printed_only": True,
                "diagnosis_codes": ["p2p_lite_daemon_command_ready"],
            })
        completed = runner(command, cwd=str(ROOT), text=True)
        return sanitize({
            "schema": P2P_LITE_CLI_SCHEMA,
            "ok": completed.returncode == 0,
            "mode": "daemon",
            "returncode": completed.returncode,
            "diagnosis_codes": ["p2p_lite_daemon_exited"],
        })
    if action == "resolve":
        session_request = build_session_request(
            prompt_text=args.prompt_text,
            backend=args.backend,
            stage_mode="split",
            max_new_tokens=args.max_new_tokens,
            route_source="peer-bootstrap",
        )
        coordinator_url = ""
        peers: list[dict[str, Any]] = []
        if args.bootstrap:
            coordinator_url, peers = resolve_coordinator_from_bootstrap(args.bootstrap, timeout=args.http_timeout)
        route = build_route_decision(session_request, coordinator_url=coordinator_url, peer_catalog=peers)
        return sanitize({
            "schema": P2P_LITE_CLI_SCHEMA,
            "ok": bool(route.get("usable_now")),
            "mode": "resolve",
            "session_request": session_request,
            "route": route,
            "peer_count": len(peers),
            "diagnosis_codes": list(route.get("diagnosis_codes") or []),
        })
    if action == "announce":
        peer = sanitize_peer({
            "schema": PEER_SCHEMA,
            "swarm_id": args.swarm_id,
            "peer_id": args.peer_id,
            "role": args.role,
            "urls": {"coordinator": args.coordinator_url, "peer": args.peer_url},
            "backend": args.backend,
            "stage_role": args.stage_role,
            "capabilities": {
                "runtime": "python-cli",
                "backend": args.backend,
                "real_llm_sharded_stage_role": args.stage_role,
                "real_llm_sharded_stage_capabilities": list(args.stage_capability or []),
            },
            "ttl_seconds": args.ttl_seconds,
        })
        payload = post_announce(args.bootstrap, peer, timeout=args.http_timeout)
        return sanitize({
            "schema": P2P_LITE_CLI_SCHEMA,
            "ok": bool(payload.get("ok")),
            "mode": "announce",
            "peer_id": peer.get("peer_id"),
            "diagnosis_codes": ["p2p_lite_announce_ready" if payload.get("ok") else "p2p_lite_announce_failed"],
        })
    raise SystemExit(f"unknown peer action: {action}")


def build_public_swarm_product_rc(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "public_swarm_product_rc_pack.py"),
        "--output-dir",
        args.output_dir,
        "--gpu-report",
        args.gpu_report,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--json",
    ]
    step, payload = run_json_step(
        "public_swarm_product_rc",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds + 60,
    )
    if payload:
        return payload
    return sanitize({
        "schema": "public_swarm_product_rc_v1",
        "cli_schema": PUBLIC_SWARM_PRODUCT_CLI_SCHEMA,
        "ok": False,
        "mode": "rc",
        "step": step,
        "diagnosis_codes": ["public_swarm_product_rc_failed"],
    })


def build_release_ready(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "release_readiness_pack.py"),
        "--output-dir",
        str(output_dir),
        "--host",
        args.host,
        "--base-port",
        str(args.base_port),
        "--request-count",
        str(args.request_count),
        "--external-llm-request-count",
        str(args.external_llm_request_count),
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]
    if args.allow_dirty:
        command.append("--allow-dirty")
    if args.skip_external_llm_evidence:
        command.append("--skip-external-llm-evidence")
    if args.runtime_report:
        command.extend(["--runtime-report", args.runtime_report])
    if args.browser_report:
        command.extend(["--browser-report", args.browser_report])
    if args.remote_report:
        command.extend(["--remote-report", args.remote_report])
    step, payload = run_json_step(
        "release_readiness",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
    )
    if payload:
        return sanitize(payload)
    return sanitize({
        "schema": "release_readiness_v1",
        "ok": False,
        "release_status": {
            "ready": False,
            "status": "blocked",
            "blocking_reasons": [step.get("error") or "release readiness command failed"],
            "diagnosis_codes": ["release_readiness_failed"],
        },
        "step": step,
    })


def build_remote_runbook(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "remote_demo_runbook_pack.py"),
        "--coordinator-url",
        args.coordinator_url,
        "--miner-id",
        args.miner_id,
        "--output-dir",
        str(output_dir),
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
    ]
    if args.replace:
        command.append("--replace")
    step, payload = run_json_step(
        "remote_demo_runbook",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
    )
    step["ok"] = bool(step.get("ok") and payload.get("ok"))

    runbook_json = output_dir / "remote_demo_runbook.json"
    runbook_md = output_dir / "remote_demo_runbook.md"
    operator_env = output_dir / "operator.private.env"
    miner_env = output_dir / "miner.private.env"
    summary_json = output_dir / "remote_runbook_cli_summary.json"
    artifacts = {
        "remote_demo_runbook_json": _remote_summary_artifact(
            runbook_json,
            output_dir,
            kind="remote_demo_runbook",
            schema=str(payload.get("schema") or "remote_demo_runbook_v1"),
        ),
        "remote_demo_runbook_markdown": _remote_summary_artifact(
            runbook_md,
            output_dir,
            kind="remote_demo_runbook_markdown",
        ),
        "operator_private_env": _remote_summary_artifact(operator_env, output_dir, kind="private_env"),
        "miner_private_env": _remote_summary_artifact(miner_env, output_dir, kind="private_env"),
        "remote_runbook_cli_summary": {
            "kind": "remote_runbook_cli_summary",
            "path": "remote_runbook_cli_summary.json",
            "present": True,
            "schema": REMOTE_RUNBOOK_CLI_SCHEMA,
        },
    }
    demo = payload.get("demo") if isinstance(payload.get("demo"), dict) else {}
    scenario = {
        "scenario_schema": demo.get("scenario_schema"),
        "scenario_id": demo.get("scenario_id"),
        "scenario_description": demo.get("scenario_description"),
        "scenario_request_count": demo.get("scenario_request_count"),
    }
    summary = {
        "schema": REMOTE_RUNBOOK_CLI_SCHEMA,
        "generated_at": utc_now(),
        "ok": bool(step.get("ok")),
        "output_dir": str(output_dir),
        "coordinator_url": args.coordinator_url.rstrip("/"),
        "miner_id": args.miner_id,
        "request_count": args.request_count,
        "scenario": scenario,
        "step": step,
        "runbook_schema": payload.get("schema") or "remote_demo_runbook_v1",
        "workload_type": demo.get("workload_type") or "model_bundle_infer",
        "artifacts": artifacts,
        "safety": {
            "private_env_files": ["operator.private.env", "miner.private.env"],
            "public_artifacts_exclude_plaintext_tokens": True,
            "captured_output_redacted": True,
            "not_production": True,
        },
        "limitations": [
            "Controlled two-machine demo runbook; not production Swarm Inference",
            "Requires operator-provided TLS, VPN, or private networking for non-local use",
            "Does not implement P2P/NAT traversal, GPU pooling, WebGPU model shards, or incentives",
        ],
        "recommended_next_commands": [
            f"source {output_dir / 'operator.private.env'}",
            (
                "crowdtensor remote-acceptance --coordinator-url https://YOUR_COORDINATOR_HOST "
                "--miner-id remote-linux-1 --observer-token \"$CROWDTENSOR_OBSERVER_TOKEN\" "
                "--admin-token \"$CROWDTENSOR_ADMIN_TOKEN\" --create-session "
                f"--scenario-id {args.scenario_id} --json"
            ),
        ],
    }
    summary = sanitize(redact_values(summary))
    encoded = json.dumps(summary, sort_keys=True)
    leaks = [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded and fragment not in {
        "CROWDTENSOR_OBSERVER_TOKEN",
        "CROWDTENSOR_ADMIN_TOKEN",
    }]
    if leaks:
        summary["ok"] = False
        summary.setdefault("errors", []).append("sensitive_output_detected")
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_remote_acceptance(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    secret_values = [args.observer_token, args.admin_token]
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "remote_demo_acceptance_pack.py"),
        "--coordinator-url",
        args.coordinator_url,
        "--miner-id",
        args.miner_id,
        "--observer-token",
        args.observer_token,
        "--admin-token",
        args.admin_token,
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
        "--timeout-seconds",
        str(args.remote_timeout_seconds),
        "--poll-interval",
        str(args.poll_interval),
        "--output-dir",
        str(output_dir),
    ]
    if args.create_session:
        command.append("--create-session")
    step, payload = run_json_step(
        "remote_demo_acceptance",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
        redact_secrets=secret_values,
    )
    step["ok"] = bool(step.get("ok") and payload.get("ok"))

    acceptance_json = output_dir / "remote_demo_acceptance.json"
    acceptance_md = output_dir / "remote_demo_acceptance.md"
    summary_json = output_dir / "remote_acceptance_cli_summary.json"
    artifacts = {
        "remote_demo_acceptance_json": _remote_summary_artifact(
            acceptance_json,
            output_dir,
            kind="remote_demo_acceptance",
            schema=str(payload.get("schema") or "remote_demo_acceptance_v1"),
        ),
        "remote_demo_acceptance_markdown": _remote_summary_artifact(
            acceptance_md,
            output_dir,
            kind="remote_demo_acceptance_markdown",
        ),
        "remote_acceptance_cli_summary": {
            "kind": "remote_acceptance_cli_summary",
            "path": "remote_acceptance_cli_summary.json",
            "present": True,
            "schema": REMOTE_ACCEPTANCE_CLI_SCHEMA,
        },
    }
    summary = {
        "schema": REMOTE_ACCEPTANCE_CLI_SCHEMA,
        "generated_at": utc_now(),
        "ok": bool(step.get("ok")),
        "output_dir": str(output_dir),
        "coordinator_url": args.coordinator_url.rstrip("/"),
        "miner_id": args.miner_id,
        "request_count": args.request_count,
        "scenario": payload.get("scenario") or {},
        "create_session": bool(args.create_session),
        "step": step,
        "acceptance_schema": payload.get("schema") or "remote_demo_acceptance_v1",
        "diagnosis_codes": diagnosis_codes(payload),
        "artifacts": artifacts,
        "safety": {
            "captured_output_redacted": True,
            "summary_excludes_plaintext_tokens": True,
            "read_only_workload": "model_bundle_infer",
            "not_production": True,
        },
        "limitations": [
            "Controlled two-machine acceptance wrapper; not production Swarm Inference",
            "Requires a running Coordinator and remote Miner already configured by the operator",
            "Does not implement P2P/NAT traversal, GPU pooling, WebGPU model shards, or incentives",
        ],
    }
    summary = sanitize(redact_values(summary, secret_values))
    encoded = json.dumps(summary, sort_keys=True)
    if any(secret and secret in encoded for secret in secret_values):
        summary["ok"] = False
        summary.setdefault("errors", []).append("sensitive_output_detected")
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_remote_demo(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.remote_demo_action == "kaggle-real":
        command = [
            sys.executable,
            str(SCRIPTS_DIR / "kaggle_real_runtime_acceptance_pack.py"),
            args.kaggle_real_action,
            "--public-host",
            args.public_host,
            "--port",
            str(args.port),
            "--miner-id",
            args.miner_id,
            "--workload",
            args.workload,
            "--output-dir",
            str(output_dir),
            "--request-count",
            str(args.request_count),
            "--scenario-id",
            args.scenario_id,
            "--decode-steps",
            str(args.decode_steps),
            "--stage-mode",
            args.stage_mode,
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--json",
        ]
        if getattr(args, "micro_llm_artifact", ""):
            command.extend(["--micro-llm-artifact", args.micro_llm_artifact])
        if getattr(args, "prompt_texts", ""):
            command.extend(["--prompt-texts", args.prompt_texts])
        if args.require_distinct_stage_miners:
            command.append("--require-distinct-stage-miners")
        if args.coordinator_url:
            command.extend(["--coordinator-url", args.coordinator_url])
        secret_values: list[str] = []
        if args.kaggle_real_action == "prepare":
            command.extend([
                "--bind-host",
                args.bind_host,
                "--backlog",
                str(args.backlog),
                "--lease-seconds",
                str(args.lease_seconds),
            ])
            for flag, value in [
                ("--miner-token", args.miner_token),
                ("--observer-token", args.observer_token),
                ("--admin-token", args.admin_token),
            ]:
                if value:
                    command.extend([flag, value])
                    secret_values.append(value)
            if args.replace:
                command.append("--replace")
        elif args.kaggle_real_action == "verify":
            for flag, value in [("--observer-token", args.observer_token), ("--admin-token", args.admin_token)]:
                if value:
                    command.extend([flag, value])
                    secret_values.append(value)
            command.extend([
                "--remote-timeout-seconds",
                str(args.remote_timeout_seconds),
                "--poll-interval",
                str(args.poll_interval),
                "--http-timeout",
                str(args.http_timeout),
                "--artifact-timeout",
                str(args.artifact_timeout),
                "--admin-results-limit",
                str(args.admin_results_limit),
            ])
            if args.require_existing_result:
                command.append("--require-existing-result")
            if args.collect_on_failure:
                command.append("--collect-on-failure")
        elif args.kaggle_real_action == "collect":
            for flag, value in [("--observer-token", args.observer_token), ("--admin-token", args.admin_token)]:
                if value:
                    command.extend([flag, value])
                    secret_values.append(value)
            command.extend([
                "--http-timeout",
                str(args.http_timeout),
                "--artifact-timeout",
                str(args.artifact_timeout),
                "--admin-results-limit",
                str(args.admin_results_limit),
            ])
            if args.task_id:
                command.extend(["--task-id", args.task_id])
        else:
            raise SystemExit(f"unknown kaggle-real action: {args.kaggle_real_action}")
        step, payload = run_json_step(
            "kaggle_real_runtime_acceptance",
            command,
            runner=runner,
            cwd=ROOT,
            timeout_seconds=args.timeout_seconds,
            redact_secrets=secret_values,
        )
        if not payload:
            payload = {
                "schema": KAGGLE_REAL_RUNTIME_SCHEMA,
                "generated_at": utc_now(),
                "ok": False,
                "mode": args.kaggle_real_action,
                "output_dir": str(output_dir),
                "coordinator_url": args.coordinator_url or f"http://{args.public_host}:{args.port}",
                "miner_id": args.miner_id,
                "step": step,
                "diagnosis_codes": ["kaggle_runtime_blocked"],
            }
        payload = redact_values(payload, secret_values)
        encoded = json.dumps(payload, sort_keys=True)
        if any(secret and secret in encoded for secret in secret_values):
            payload["ok"] = False
            payload.setdefault("diagnosis_codes", [])
            if "sensitive_output_detected" not in payload["diagnosis_codes"]:
                payload["diagnosis_codes"].append("sensitive_output_detected")
        return payload
    if args.remote_demo_action == "clean":
        command = [
            sys.executable,
            str(SCRIPTS_DIR / "remote_home_compute_demo_pack.py"),
            "clean",
            "--output-dir",
            str(output_dir),
            "--json",
        ]
        if args.apply:
            command.append("--apply")
        if args.include_private:
            command.append("--include-private")
        if args.remove_empty_dir:
            command.append("--remove-empty-dir")
        step, payload = run_json_step(
            "remote_home_compute_demo",
            command,
            runner=runner,
            cwd=ROOT,
            timeout_seconds=args.timeout_seconds,
        )
        if not payload:
            payload = {
                "schema": REMOTE_HOME_DEMO_SCHEMA,
                "generated_at": utc_now(),
                "ok": False,
                "mode": args.remote_demo_action,
                "output_dir": str(output_dir),
                "step": step,
                "diagnosis_codes": ["remote_home_compute_failed"],
            }
        return sanitize(payload)
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "remote_home_compute_demo_pack.py"),
        args.remote_demo_action,
        "--workload",
        args.workload,
        "--coordinator-url",
        args.coordinator_url,
        "--miner-id",
        args.miner_id,
        "--output-dir",
        str(output_dir),
        "--request-count",
        str(args.request_count),
        "--scenario-id",
        args.scenario_id,
        "--json",
    ]
    if hasattr(args, "decode_steps"):
        command.extend(["--decode-steps", str(args.decode_steps)])
    if hasattr(args, "stage_role"):
        command.extend(["--stage-role", args.stage_role])
    if hasattr(args, "stage_mode"):
        command.extend(["--stage-mode", args.stage_mode])
    if getattr(args, "require_distinct_stage_miners", False):
        command.append("--require-distinct-stage-miners")
    if args.workload == "micro-llm-sharded":
        if getattr(args, "micro_llm_artifact", ""):
            command.extend(["--micro-llm-artifact", args.micro_llm_artifact])
        if getattr(args, "prompt_texts", ""):
            command.extend(["--prompt-texts", args.prompt_texts])
    if args.workload == "real-llm-sharded":
        if getattr(args, "hf_model_id", ""):
            command.extend(["--hf-model-id", args.hf_model_id])
        if getattr(args, "hf_cache_dir", ""):
            command.extend(["--hf-cache-dir", args.hf_cache_dir])
        if getattr(args, "prompt_texts", ""):
            command.extend(["--prompt-texts", args.prompt_texts])
    if hasattr(args, "target"):
        command.extend(["--target", args.target])
    secret_values: list[str] = []
    if args.remote_demo_action in {"prepare", "verify"} and hasattr(args, "timeout_seconds"):
        command.extend(["--timeout-seconds", str(args.timeout_seconds)])
    if hasattr(args, "mock") and getattr(args, "mock", False):
        command.append("--mock")
    if hasattr(args, "llm_runtime_cmd") and getattr(args, "llm_runtime_cmd", ""):
        command.extend(["--llm-runtime-cmd", args.llm_runtime_cmd])
    if hasattr(args, "llm_runtime_url") and getattr(args, "llm_runtime_url", ""):
        command.extend(["--llm-runtime-url", args.llm_runtime_url])
    if hasattr(args, "llm_runtime_api_key") and getattr(args, "llm_runtime_api_key", ""):
        command.extend(["--llm-runtime-api-key", args.llm_runtime_api_key])
    if hasattr(args, "llm_runtime_model_id") and getattr(args, "llm_runtime_model_id", ""):
        command.extend(["--llm-runtime-model-id", args.llm_runtime_model_id])
    if hasattr(args, "llm_runtime_timeout") and getattr(args, "llm_runtime_timeout", None) is not None:
        command.extend(["--llm-runtime-timeout", str(args.llm_runtime_timeout)])
    if args.remote_demo_action == "prepare":
        if args.replace:
            command.append("--replace")
    elif args.remote_demo_action == "verify":
        secret_values = [args.observer_token, args.admin_token, args.llm_runtime_url, args.llm_runtime_api_key]
        command.extend([
            "--observer-token",
            args.observer_token,
            "--admin-token",
            args.admin_token,
            "--remote-timeout-seconds",
            str(args.remote_timeout_seconds),
            "--poll-interval",
            str(args.poll_interval),
            "--http-timeout",
            str(args.http_timeout),
            "--artifact-timeout",
            str(args.artifact_timeout),
            "--admin-results-limit",
            str(args.admin_results_limit),
        ])
        if args.create_session:
            command.append("--create-session")
        else:
            command.append("--no-create-session")
    elif args.remote_demo_action == "doctor":
        secret_values = [args.observer_token, args.admin_token]
        command.extend([
            "--observer-token",
            args.observer_token,
            "--admin-token",
            args.admin_token,
            "--http-timeout",
            str(args.http_timeout),
            "--admin-results-limit",
            str(args.admin_results_limit),
        ])
        if args.require_result:
            command.append("--require-result")
    elif args.remote_demo_action == "collect":
        secret_values = [args.observer_token, args.admin_token, args.llm_runtime_url, args.llm_runtime_api_key]
        command.extend([
            "--observer-token",
            args.observer_token,
            "--admin-token",
            args.admin_token,
            "--http-timeout",
            str(args.http_timeout),
            "--artifact-timeout",
            str(args.artifact_timeout),
            "--admin-results-limit",
            str(args.admin_results_limit),
        ])
        if args.task_id:
            command.extend(["--task-id", args.task_id])
    else:
        raise SystemExit(f"unknown remote-demo action: {args.remote_demo_action}")
    step, payload = run_json_step(
        "remote_home_compute_demo",
        command,
        runner=runner,
        cwd=ROOT,
        timeout_seconds=args.timeout_seconds,
        redact_secrets=secret_values,
    )
    if not payload:
        payload = {
            "schema": REMOTE_HOME_DEMO_SCHEMA,
            "generated_at": utc_now(),
            "ok": False,
            "mode": args.remote_demo_action,
            "output_dir": str(output_dir),
            "coordinator_url": args.coordinator_url.rstrip("/"),
            "miner_id": args.miner_id,
            "step": step,
            "diagnosis_codes": ["remote_home_compute_failed"],
        }
    payload = sanitize(redact_values(payload, secret_values))
    encoded = json.dumps(payload, sort_keys=True)
    if any(secret and secret in encoded for secret in secret_values):
        payload["ok"] = False
        payload.setdefault("diagnosis_codes", [])
        if "sensitive_output_detected" not in payload["diagnosis_codes"]:
            payload["diagnosis_codes"].append("sensitive_output_detected")
    return payload


def print_local_proof(summary: dict[str, Any]) -> None:
    print("CrowdTensor local proof")
    print(f"  ok: {summary.get('ok')}")
    print(f"  schema: {summary.get('schema')}")
    print(f"  output: {summary.get('output_dir')}")
    print(f"  diagnosis: {', '.join(summary.get('diagnosis_codes') or [])}")
    for step in summary.get("steps") or []:
        state = "skipped" if step.get("skipped") else step.get("ok")
        print(f"  step {step.get('name')}: {state}")
    artifacts = summary.get("artifacts") or {}
    manifest = artifacts.get("demo_manifest_json") or {}
    print(f"  demo manifest: {manifest.get('path')} present={manifest.get('present')}")


def print_cleanup_report(report: dict[str, Any]) -> None:
    print("CrowdTensor artifact cleanup")
    print(f"  ok: {report.get('ok')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  candidates: {report.get('candidate_count')}")
    print(f"  deleted_bytes: {report.get('deleted_bytes')}")
    for candidate in report.get("candidates") or []:
        print(
            "  "
            f"{candidate.get('action')} "
            f"{candidate.get('kind')} "
            f"{candidate.get('bytes')}B "
            f"{candidate.get('path')}"
        )


def print_home_inference(summary: dict[str, Any]) -> None:
    print("CrowdTensor home inference")
    print(f"  ok: {summary.get('ok')}")
    print(f"  schema: {summary.get('schema')}")
    print(f"  output: {summary.get('output_dir')}")
    route = summary.get("route") or {}
    print(f"  route: {route.get('name')} target={route.get('target')} confidence={route.get('confidence')}")
    scenario = summary.get("scenario") or {}
    if scenario.get("scenario_id"):
        print(f"  scenario: {scenario.get('scenario_id')} ({scenario.get('scenario_schema')})")
    print(f"  diagnosis: {', '.join(summary.get('diagnosis_codes') or [])}")
    inference = summary.get("inference") or {}
    print(
        "  inference: "
        f"present={inference.get('present')} "
        f"requests={inference.get('request_count')} "
        f"trace={inference.get('request_trace_count')} "
        f"rps={inference.get('requests_per_second')}"
    )
    for name, artifact in sorted((summary.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_llm_inference(summary: dict[str, Any]) -> None:
    print("CrowdTensor LLM inference")
    print(f"  ok: {summary.get('ok')}")
    print(f"  schema: {summary.get('schema')}")
    print(f"  output: {summary.get('output_dir')}")
    adapter = summary.get("adapter") or {}
    print(f"  adapter: {adapter.get('kind')} model={adapter.get('model_id')}")
    inference = summary.get("inference") or {}
    print(
        "  inference: "
        f"requests={inference.get('request_count')} "
        f"completions={inference.get('completion_count')} "
        f"chars={inference.get('output_chars')} "
        f"rps={inference.get('requests_per_second')}"
    )
    print(f"  diagnosis: {', '.join(summary.get('diagnosis_codes') or [])}")
    for name, artifact in sorted((summary.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_cpu_inference_beta(report: dict[str, Any]) -> None:
    print("CrowdTensor CPU inference Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for step in report.get("steps") or []:
        print(f"  step {step.get('name')}: {step.get('ok')} schema={step.get('payload_schema')}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_sharded_inference(summary: dict[str, Any]) -> None:
    print("CrowdTensor sharded inference")
    print(f"  ok: {summary.get('ok')}")
    print(f"  schema: {summary.get('schema')}")
    print(f"  output: {summary.get('output_dir')}")
    print(f"  failure_mode: {summary.get('failure_mode')}")
    print(f"  diagnosis: {', '.join(summary.get('diagnosis_codes') or [])}")
    session = summary.get("session") or {}
    print(f"  session: {session.get('session_id')} stages={session.get('stage_count')}")
    stage = summary.get("stage_summary") or {}
    stage0 = stage.get("stage_0") or {}
    stage1 = stage.get("stage_1") or {}
    print(f"  stage0: task={stage0.get('task_id')} miner={stage0.get('miner_id')} activations={stage0.get('activation_count')}")
    print(f"  stage1: task={stage1.get('task_id')} miner={stage1.get('miner_id')} baseline_match={stage1.get('baseline_match')}")
    for name, artifact in sorted((summary.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_micro_llm_sharded_inference(summary: dict[str, Any]) -> None:
    print("CrowdTensor micro-LLM sharded inference")
    print(f"  ok: {summary.get('ok')}")
    print(f"  schema: {summary.get('schema')}")
    print(f"  output: {summary.get('output_dir')}")
    print(f"  failure_mode: {summary.get('failure_mode')}")
    print(f"  diagnosis: {', '.join(summary.get('diagnosis_codes') or [])}")
    session = summary.get("session") or {}
    print(f"  session: {session.get('session_id')} stages={session.get('stage_count')} decode_steps={session.get('decode_steps')}")
    stage = summary.get("stage_summary") or {}
    stage0 = stage.get("stage_0") or {}
    stage1 = stage.get("stage_1") or {}
    print(f"  stage0: task={stage0.get('task_id')} miner={stage0.get('miner_id')} activations={stage0.get('activation_count')}")
    print(
        "  stage1: "
        f"task={stage1.get('task_id')} "
        f"miner={stage1.get('miner_id')} "
        f"baseline_match={stage1.get('baseline_match')} "
        f"decoded_tokens_match={stage1.get('decoded_tokens_match')}"
    )
    for name, artifact in sorted((summary.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_real_llm_sharded_inference(summary: dict[str, Any]) -> None:
    print("CrowdTensor real tiny-LLM sharded inference")
    print(f"  ok: {summary.get('ok')}")
    print(f"  schema: {summary.get('schema')}")
    print(f"  output: {summary.get('output_dir')}")
    print(f"  failure_mode: {summary.get('failure_mode')}")
    print(f"  diagnosis: {', '.join(summary.get('diagnosis_codes') or [])}")
    session = summary.get("session") or {}
    print(f"  session: {session.get('session_id')} stages={session.get('stage_count')} model={session.get('model_id')}")
    stage = summary.get("stage_summary") or {}
    stage0 = stage.get("stage_0") or {}
    stage1 = stage.get("stage_1") or {}
    print(f"  stage0: task={stage0.get('task_id')} miner={stage0.get('miner_id')} activations={stage0.get('activation_count')}")
    print(
        "  stage1: "
        f"task={stage1.get('task_id')} "
        f"miner={stage1.get('miner_id')} "
        f"baseline_match={stage1.get('baseline_match')} "
        f"decoded_tokens_match={stage1.get('decoded_tokens_match')}"
    )
    for name, artifact in sorted((summary.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_micro_llm_artifact(summary: dict[str, Any]) -> None:
    print("CrowdTensor micro-LLM artifact")
    print(f"  ok: {summary.get('ok')}")
    print(f"  schema: {summary.get('schema')}")
    print(f"  output: {summary.get('output_dir')}")
    print(f"  artifact: {summary.get('artifact_id')} {summary.get('artifact_hash')}")
    print(f"  diagnosis: {', '.join(summary.get('diagnosis_codes') or [])}")
    for name, artifact in sorted((summary.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_remote_sharded_inference_beta(report: dict[str, Any]) -> None:
    print("CrowdTensor remote sharded inference Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  failure_mode: {report.get('failure_mode')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for step in report.get("steps") or []:
        print(f"  step {step.get('name')}: {step.get('ok')} schema={step.get('payload_schema')}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_remote_micro_llm_sharded_beta(report: dict[str, Any]) -> None:
    print("CrowdTensor remote micro-LLM sharded inference Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  failure_mode: {report.get('failure_mode')}")
    print(f"  decode_steps: {report.get('decode_steps')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for step in report.get("steps") or []:
        print(f"  step {step.get('name')}: {step.get('ok')} schema={step.get('payload_schema')}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_remote_real_llm_sharded_beta(report: dict[str, Any]) -> None:
    print("CrowdTensor remote real tiny-LLM sharded inference Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  failure_mode: {report.get('failure_mode')}")
    print(f"  hf_model_id: {report.get('hf_model_id')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for step in report.get("steps") or []:
        print(f"  step {step.get('name')}: {step.get('ok')} schema={step.get('payload_schema')}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_micro_llm_live_rc(report: dict[str, Any]) -> None:
    print("CrowdTensor micro-LLM live two-node RC")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  coordinator: {report.get('coordinator_url')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    runtime = report.get("runtime_classification") or {}
    print(f"  local stand-in: {runtime.get('local_generated_stage_upload_standins')}")
    print(f"  external runtime: {runtime.get('external_runtime_verified')}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_real_llm_live_rc(report: dict[str, Any]) -> None:
    print("CrowdTensor real small-LLM live two-node RC")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  coordinator: {report.get('coordinator_url')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    runtime = report.get("runtime_classification") or {}
    print(f"  local stand-in: {runtime.get('local_generated_stage_upload_standins')}")
    print(f"  external runtime: {runtime.get('external_runtime_verified')}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_real_llm_internet_alpha(report: dict[str, Any]) -> None:
    print("CrowdTensor real Internet Swarm Inference Alpha")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  coordinator: {report.get('coordinator_url')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    runtime = report.get("runtime_classification") or {}
    print(f"  local stand-in: {runtime.get('local_generated_stage_upload_standins')}")
    print(f"  package only: {runtime.get('package_only')}")
    print(f"  external runtime: {runtime.get('external_runtime_verified')}")
    print(f"  stage requeue: {runtime.get('stage_requeue_verified')}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_real_llm_internet_beta(report: dict[str, Any]) -> None:
    print("CrowdTensor real Internet Swarm Inference Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  coordinator: {report.get('coordinator_url')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    runtime = report.get("runtime_classification") or {}
    lifecycle = report.get("kaggle_lifecycle") or {}
    print(f"  kaggle auto: {runtime.get('kaggle_auto')}")
    print(f"  external runtime: {runtime.get('external_runtime_verified')}")
    print(f"  kernels deleted: {lifecycle.get('kernels_deleted')}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_swarm_inference_beta(report: dict[str, Any]) -> None:
    print("CrowdTensor Swarm Inference Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    if report.get("command_text"):
        print(f"  command: {report.get('command_text')}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_public_swarm_inference_alpha(report: dict[str, Any]) -> None:
    session = report.get("session") if isinstance(report.get("session"), dict) else {}
    print("CrowdTensor Public Swarm Inference Alpha")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  model: {session.get('model_id')}")
    print(f"  external runtime: {session.get('live_external_runtime_verified')}")
    print(f"  local requeue: {session.get('local_stage_requeue_verified')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_public_swarm_inference_alpha_rc(report: dict[str, Any]) -> None:
    rc = report.get("release_candidate") if isinstance(report.get("release_candidate"), dict) else {}
    print("CrowdTensor Public Swarm Inference Alpha RC")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  ready: {rc.get('ready')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_public_swarm_inference_beta(report: dict[str, Any]) -> None:
    beta = report.get("beta") if isinstance(report.get("beta"), dict) else {}
    print("CrowdTensor Public Swarm Inference Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  ready: {beta.get('ready')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_public_swarm_inference_beta_rc(report: dict[str, Any]) -> None:
    rc = report.get("rc") if isinstance(report.get("rc"), dict) else {}
    print("CrowdTensor Public Swarm Inference Beta RC")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  ready: {rc.get('ready')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_public_swarm_product_beta(report: dict[str, Any]) -> None:
    beta = report.get("product_beta") if isinstance(report.get("product_beta"), dict) else {}
    print("CrowdTensor Public Swarm Product Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  ready: {beta.get('ready')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_public_swarm_gpu_inference_beta(report: dict[str, Any]) -> None:
    beta = report.get("beta") if isinstance(report.get("beta"), dict) else {}
    print("CrowdTensor Public Swarm GPU Inference Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  ready: {beta.get('ready')}")
    print(f"  backend: {beta.get('backend')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_gpu_sharded_generation_beta(report: dict[str, Any]) -> None:
    generation = report.get("generation") if isinstance(report.get("generation"), dict) else {}
    gpu = report.get("gpu") if isinstance(report.get("gpu"), dict) else {}
    print("CrowdTensor GPU sharded generation Beta")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  cli_schema: {report.get('cli_schema')}")
    print(f"  mode: {report.get('mode')}")
    print(f"  backend: {gpu.get('backend')}")
    print(f"  model: {gpu.get('model_id')}")
    print(f"  generated_tokens: {generation.get('generated_token_count')}/{generation.get('max_new_tokens')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  diagnosis: {', '.join(report.get('diagnosis_codes') or [])}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_release_ready(report: dict[str, Any]) -> None:
    status = report.get("release_status") or {}
    git = report.get("git") or {}
    print("CrowdTensor release readiness")
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  status: {status.get('status')}")
    print(f"  branch: {git.get('branch')} commit={git.get('commit')}")
    print(f"  dirty: {git.get('dirty')} status_count={git.get('status_count')}")
    print(f"  diagnosis: {', '.join(status.get('diagnosis_codes') or [])}")
    for reason in status.get("blocking_reasons") or []:
        print(f"  blocker: {reason}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def print_remote_cli_report(report: dict[str, Any], *, title: str) -> None:
    print(title)
    print(f"  ok: {report.get('ok')}")
    print(f"  schema: {report.get('schema')}")
    print(f"  output: {report.get('output_dir')}")
    print(f"  miner: {report.get('miner_id')}")
    codes = report.get("diagnosis_codes") or []
    if codes:
        print(f"  diagnosis: {', '.join(codes)}")
    step = report.get("step") or {}
    if step:
        print(f"  step {step.get('name')}: {step.get('ok')}")
    for name, artifact in sorted((report.get("artifacts") or {}).items()):
        print(f"  artifact {name}: {artifact.get('path')} present={artifact.get('present')}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="crowdtensor", description="CrowdTensor user-facing command line tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    local = subparsers.add_parser("local-proof", help="Run the CPU-only local proof and collect safe artifacts.")
    local.add_argument("--output-dir", default="dist/local-proof")
    local.add_argument("--base-port", type=int, default=8914)
    local.add_argument("--request-count", type=int, default=4)
    local.add_argument("--timeout-seconds", type=int, default=180)
    local.add_argument("--skip-doctor", action="store_true")
    local.add_argument("--json", action="store_true")
    clean = subparsers.add_parser("clean-artifacts", help="Safely clean generated CrowdTensor caches and temp artifacts.")
    clean.add_argument("--apply", action="store_true", help="delete eligible artifacts; default is dry-run")
    clean.add_argument("--dry-run", action="store_true", help="show candidates without deleting; this is the default")
    clean.add_argument("--include-reports", action="store_true", help="allow deletion of /tmp/crowdtensor_*.json/md reports")
    clean.add_argument("--older-than-hours", type=float, default=24.0)
    clean.add_argument("--json", action="store_true")

    serve = subparsers.add_parser("serve", help="Print or run a product-facing Coordinator command.")
    serve.add_argument("--profile", choices=["cpu-real-llm", "gpu-generation"], default="cpu-real-llm")
    serve.add_argument("--bind-host", default="127.0.0.1")
    serve.add_argument("--public-host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--state-dir", default="state")
    serve.add_argument("--admin-token", default=os.environ.get("CROWDTENSOR_ADMIN_TOKEN", "local-admin"))
    serve.add_argument("--miner-token", default=os.environ.get("CROWDTENSOR_MINER_TOKEN", ""))
    serve.add_argument("--observer-token", default=os.environ.get("CROWDTENSOR_OBSERVER_TOKEN", ""))
    serve.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    serve.add_argument("--hf-cache-dir", default="")
    serve.add_argument("--i-understand-public-bind", action="store_true")
    serve.add_argument("--run", action="store_true")
    serve.add_argument("--json", action="store_true")

    join = subparsers.add_parser("join", help="Print or run a product-facing Miner command.")
    join.add_argument("--coordinator-url", default="")
    join.add_argument("--peer-bootstrap", default="")
    join.add_argument("--miner-id", default="public-swarm-miner")
    join.add_argument("--stage", choices=["stage0", "stage1", "both"], default="both")
    join.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    join.add_argument("--miner-token", default=os.environ.get("CROWDTENSOR_MINER_TOKEN", ""))
    join.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    join.add_argument("--hf-cache-dir", default="")
    join.add_argument("--once", action="store_true")
    join.add_argument("--max-tasks", type=int, default=0)
    join.add_argument("--http-timeout", type=float, default=5.0)
    join.add_argument("--run", action="store_true")
    join.add_argument("--json", action="store_true")

    generate = subparsers.add_parser("generate", help="Create a bounded public product generation session.")
    generate.add_argument("--prompt-text", required=True)
    generate.add_argument("--scenario-id", default="public-swarm-product-rc")
    generate.add_argument("--max-new-tokens", type=int, default=16)
    generate.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    generate.add_argument("--coordinator-url", default="")
    generate.add_argument("--peer-bootstrap", default="")
    generate.add_argument("--admin-token", default=os.environ.get("CROWDTENSOR_ADMIN_TOKEN", ""))
    generate.add_argument("--timeout-seconds", type=float, default=120.0)
    generate.add_argument("--poll-interval", type=float, default=1.0)
    generate.add_argument("--http-timeout", type=float, default=10.0)
    generate.add_argument("--admin-results-limit", type=int, default=50)
    generate.add_argument("--dry-run", action="store_true")
    generate.add_argument("--include-output", action="store_true")
    generate.add_argument("--json", action="store_true")

    peer = subparsers.add_parser("peer", help="Run or query the P2P-lite discovery layer.")
    peer_subparsers = peer.add_subparsers(dest="peer_action", required=True)
    peer_check = peer_subparsers.add_parser("check", help="Run the P2P-lite discovery check.")
    peer_check.add_argument("--timeout-seconds", type=int, default=60)
    peer_check.add_argument("--json", action="store_true")

    peer_daemon = peer_subparsers.add_parser("daemon", help="Print or run the P2P-lite daemon command.")
    peer_daemon.add_argument("--host", default="127.0.0.1")
    peer_daemon.add_argument("--port", type=int, default=8788)
    peer_daemon.add_argument("--swarm-id", default="default")
    peer_daemon.add_argument("--peer-id", default="")
    peer_daemon.add_argument("--role", choices=["coordinator", "miner", "observer"], default="observer")
    peer_daemon.add_argument("--peer-url", default="")
    peer_daemon.add_argument("--coordinator-url", default="")
    peer_daemon.add_argument("--backend", choices=["", "cpu", "cuda"], default="")
    peer_daemon.add_argument("--stage-role", choices=["", "stage0", "stage1", "both"], default="")
    peer_daemon.add_argument("--stage-capability", action="append", default=[])
    peer_daemon.add_argument("--bootstrap", action="append", default=[])
    peer_daemon.add_argument("--ttl-seconds", type=float, default=60.0)
    peer_daemon.add_argument("--print-peer", action="store_true")
    peer_daemon.add_argument("--run", action="store_true")
    peer_daemon.add_argument("--json", action="store_true")

    peer_resolve = peer_subparsers.add_parser("resolve", help="Resolve a Coordinator route through a P2P-lite bootstrap.")
    peer_resolve.add_argument("--bootstrap", required=True)
    peer_resolve.add_argument("--prompt-text", default="CrowdTensor route probe")
    peer_resolve.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    peer_resolve.add_argument("--max-new-tokens", type=int, default=16)
    peer_resolve.add_argument("--http-timeout", type=float, default=5.0)
    peer_resolve.add_argument("--json", action="store_true")

    peer_announce = peer_subparsers.add_parser("announce", help="Announce one peer to a P2P-lite bootstrap.")
    peer_announce.add_argument("--bootstrap", required=True)
    peer_announce.add_argument("--swarm-id", default="default")
    peer_announce.add_argument("--peer-id", required=True)
    peer_announce.add_argument("--role", choices=["coordinator", "miner", "observer"], default="observer")
    peer_announce.add_argument("--peer-url", default="")
    peer_announce.add_argument("--coordinator-url", default="")
    peer_announce.add_argument("--backend", choices=["", "cpu", "cuda"], default="")
    peer_announce.add_argument("--stage-role", choices=["", "stage0", "stage1", "both"], default="")
    peer_announce.add_argument("--stage-capability", action="append", default=[])
    peer_announce.add_argument("--ttl-seconds", type=float, default=60.0)
    peer_announce.add_argument("--http-timeout", type=float, default=5.0)
    peer_announce.add_argument("--json", action="store_true")

    product_rc = subparsers.add_parser(
        "public-swarm-product-rc",
        help="Build the Coordinator product surface + session protocol + P2P-lite RC artifact.",
    )
    product_rc.add_argument("--output-dir", default="dist/public-swarm-product-rc")
    product_rc.add_argument(
        "--gpu-report",
        default=(
            "dist/gpu-sharded-generation-beta-kaggle-20260528095658/"
            "gpu_sharded_generation_beta_kaggle_auto.json"
        ),
    )
    product_rc.add_argument("--max-new-tokens", type=int, default=16)
    product_rc.add_argument("--timeout-seconds", type=int, default=120)
    product_rc.add_argument("--json", action="store_true")

    home = subparsers.add_parser(
        "home-infer",
        help="Run the CPU-only read-only home inference proof and collect safe artifacts.",
    )
    home.add_argument("--output-dir", default="dist/home-infer")
    home.add_argument("--port", type=int, default=8909)
    home.add_argument("--request-count", type=int, default=4)
    home.add_argument("--scenario-id", default="route-baseline")
    home.add_argument("--timeout-seconds", type=int, default=180)
    home.add_argument("--runtime-report", default="")
    home.add_argument("--json", action="store_true")
    llm = subparsers.add_parser(
        "llm-infer",
        help="Run a local external_llm_infer proof against mock or operator-owned LLM runtime.",
    )
    llm.add_argument("--output-dir", default="dist/llm-infer")
    llm.add_argument("--port", type=int, default=8919)
    llm.add_argument("--request-count", type=int, default=3)
    llm.add_argument("--timeout-seconds", type=int, default=180)
    llm.add_argument("--mock", action="store_true", help="use the deterministic built-in mock runtime")
    llm.add_argument("--llm-runtime-cmd", default="")
    llm.add_argument("--llm-runtime-url", default="")
    llm.add_argument("--llm-runtime-api-key", default="")
    llm.add_argument("--llm-runtime-model-id", default="external-llm-runtime")
    llm.add_argument("--llm-runtime-timeout", type=float, default=30.0)
    llm.add_argument("--json", action="store_true")
    cpu_infer = subparsers.add_parser(
        "cpu-infer",
        help="Run the CPU-only inference Beta aggregate proof.",
    )
    cpu_infer.add_argument("--mode", choices=["local", "remote-loopback", "remote-existing", "beta-rc"], default="local")
    cpu_infer.add_argument("--output-dir", default="dist/cpu-infer")
    cpu_infer.add_argument("--base-port", type=int, default=8970)
    cpu_infer.add_argument("--request-count", type=int, default=4)
    cpu_infer.add_argument("--external-llm-request-count", type=int, default=3)
    cpu_infer.add_argument("--scenario-id", default="route-baseline")
    cpu_infer.add_argument("--workload", choices=["model-bundle", "external-llm", "all"], default="all")
    cpu_infer.add_argument("--coordinator-url", default="")
    cpu_infer.add_argument("--miner-id", default="remote-linux-1")
    cpu_infer.add_argument("--observer-token", default="")
    cpu_infer.add_argument("--admin-token", default="")
    cpu_infer.add_argument("--timeout-seconds", type=int, default=240)
    cpu_infer.add_argument("--remote-timeout-seconds", type=float, default=60.0)
    cpu_infer.add_argument("--kaggle-real-runtime-report", default="")
    cpu_infer.add_argument("--poll-interval", type=float, default=1.0)
    cpu_infer.add_argument("--mock", action="store_true")
    cpu_infer.add_argument("--llm-runtime-cmd", default="")
    cpu_infer.add_argument("--llm-runtime-url", default="")
    cpu_infer.add_argument("--llm-runtime-api-key", default="")
    cpu_infer.add_argument("--llm-runtime-model-id", default="external-llm-runtime")
    cpu_infer.add_argument("--llm-runtime-timeout", type=float, default=30.0)
    cpu_infer.add_argument("--json", action="store_true")
    shard = subparsers.add_parser(
        "shard-infer",
        help="Run the CPU-only two-stage pipeline-sharded inference Alpha proof.",
    )
    shard.add_argument("--output-dir", default="dist/shard-infer")
    shard.add_argument("--port", type=int, default=9820)
    shard.add_argument("--request-count", type=int, default=4)
    shard.add_argument("--scenario-id", default="route-baseline")
    shard.add_argument(
        "--failure-mode",
        choices=["none", "kill-stage-after-claim", "kill-stage0-after-claim", "kill-stage1-after-claim"],
        default="none",
    )
    shard.add_argument("--stage-mode", choices=["both", "split"], default="both")
    shard.add_argument("--require-distinct-stage-miners", action="store_true")
    shard.add_argument("--timeout-seconds", type=int, default=120)
    shard.add_argument("--json", action="store_true")
    micro_shard = subparsers.add_parser(
        "micro-llm-shard-infer",
        help="Run the CPU-only deterministic micro-LLM pipeline-sharded inference Alpha proof.",
    )
    micro_shard.add_argument("--output-dir", default="dist/micro-llm-shard-infer")
    micro_shard.add_argument("--port", type=int, default=9860)
    micro_shard.add_argument("--request-count", type=int, default=4)
    micro_shard.add_argument("--decode-steps", type=int, default=4)
    micro_shard.add_argument(
        "--failure-mode",
        choices=["none", "kill-stage-after-claim", "kill-stage0-after-claim", "kill-stage1-after-claim"],
        default="none",
    )
    micro_shard.add_argument("--stage-mode", choices=["both", "split"], default="both")
    micro_shard.add_argument("--require-distinct-stage-miners", action="store_true")
    micro_shard.add_argument("--micro-llm-artifact", default="")
    micro_shard.add_argument("--prompt-texts", default="arn,ten")
    micro_shard.add_argument("--timeout-seconds", type=int, default=150)
    micro_shard.add_argument("--json", action="store_true")
    real_shard = subparsers.add_parser(
        "real-llm-shard-infer",
        help="Run the optional CPU-only tiny Hugging Face LLM pipeline-sharded inference Alpha proof.",
    )
    real_shard.add_argument("--output-dir", default="dist/real-llm-shard-infer")
    real_shard.add_argument("--port", type=int, default=9880)
    real_shard.add_argument("--request-count", type=int, default=1)
    real_shard.add_argument("--max-new-tokens", type=int, default=1)
    real_shard.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    real_shard.add_argument("--hf-cache-dir", default="")
    real_shard.add_argument("--real-llm-partition-mode", choices=["full", "stage-local", "stage_local"], default="full")
    real_shard.add_argument("--prompt-texts", default="CrowdTensor routes home CPU,A miner returns one token")
    real_shard.add_argument(
        "--failure-mode",
        choices=["none", "kill-stage-after-claim", "kill-stage0-after-claim", "kill-stage1-after-claim"],
        default="none",
    )
    real_shard.add_argument("--stage-mode", choices=["both", "split"], default="both")
    real_shard.add_argument("--require-distinct-stage-miners", action="store_true")
    real_shard.add_argument("--timeout-seconds", type=int, default=240)
    real_shard.add_argument("--json", action="store_true")
    micro_artifact = subparsers.add_parser(
        "micro-llm-artifact",
        help="Build or inspect the dependency-free file-backed Micro-LLM artifact.",
    )
    micro_artifact.add_argument("--output-dir", default="dist/micro-llm-artifact")
    micro_artifact.add_argument("--artifact-id", default="crowdtensor-micro-llm-alpha")
    micro_artifact.add_argument("--version", type=int, default=1)
    micro_artifact.add_argument("--inspect", action="store_true")
    micro_artifact.add_argument("--timeout-seconds", type=int, default=60)
    micro_artifact.add_argument("--json", action="store_true")
    shard_beta = subparsers.add_parser(
        "shard-infer-beta",
        help="Run the CPU-only remote pipeline-sharded inference Beta proof.",
    )
    shard_beta.add_argument("--mode", choices=["local", "remote-loopback", "remote-existing"], default="remote-loopback")
    shard_beta.add_argument("--output-dir", default="dist/remote-sharded-inference")
    shard_beta.add_argument("--base-port", type=int, default=9830)
    shard_beta.add_argument("--request-count", type=int, default=4)
    shard_beta.add_argument("--scenario-id", default="route-baseline")
    shard_beta.add_argument(
        "--failure-mode",
        choices=["none", "kill-stage-after-claim", "kill-stage0-after-claim", "kill-stage1-after-claim"],
        default="none",
    )
    shard_beta.add_argument("--stage-mode", choices=["both", "split"], default="both")
    shard_beta.add_argument("--require-distinct-stage-miners", action="store_true")
    shard_beta.add_argument("--coordinator-url", default="")
    shard_beta.add_argument("--observer-token", default="")
    shard_beta.add_argument("--admin-token", default="")
    shard_beta.add_argument("--timeout-seconds", type=int, default=180)
    shard_beta.add_argument("--remote-timeout-seconds", type=float, default=90.0)
    shard_beta.add_argument("--json", action="store_true")
    micro_shard_beta = subparsers.add_parser(
        "micro-llm-shard-infer-beta",
        help="Run the CPU-only remote micro-LLM pipeline-sharded inference Beta proof.",
    )
    micro_shard_beta.add_argument("--mode", choices=["local", "remote-loopback", "remote-existing"], default="remote-loopback")
    micro_shard_beta.add_argument("--output-dir", default="dist/remote-micro-llm-sharded-inference")
    micro_shard_beta.add_argument("--base-port", type=int, default=9870)
    micro_shard_beta.add_argument("--request-count", type=int, default=4)
    micro_shard_beta.add_argument("--decode-steps", type=int, default=4)
    micro_shard_beta.add_argument(
        "--failure-mode",
        choices=["none", "kill-stage-after-claim", "kill-stage0-after-claim", "kill-stage1-after-claim"],
        default="none",
    )
    micro_shard_beta.add_argument("--stage-mode", choices=["both", "split"], default="both")
    micro_shard_beta.add_argument("--require-distinct-stage-miners", action="store_true")
    micro_shard_beta.add_argument("--coordinator-url", default="")
    micro_shard_beta.add_argument("--observer-token", default="")
    micro_shard_beta.add_argument("--admin-token", default="")
    micro_shard_beta.add_argument("--timeout-seconds", type=int, default=180)
    micro_shard_beta.add_argument("--remote-timeout-seconds", type=float, default=90.0)
    micro_shard_beta.add_argument("--json", action="store_true")
    real_shard_beta = subparsers.add_parser(
        "real-llm-shard-infer-beta",
        help="Run the optional CPU-only remote tiny Hugging Face LLM pipeline-sharded inference Beta proof.",
    )
    real_shard_beta.add_argument("--mode", choices=["local", "remote-loopback", "remote-existing"], default="remote-loopback")
    real_shard_beta.add_argument("--output-dir", default="dist/remote-real-llm-sharded-inference")
    real_shard_beta.add_argument("--base-port", type=int, default=9890)
    real_shard_beta.add_argument("--request-count", type=int, default=1)
    real_shard_beta.add_argument("--max-new-tokens", type=int, default=1)
    real_shard_beta.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    real_shard_beta.add_argument("--hf-cache-dir", default="")
    real_shard_beta.add_argument("--real-llm-partition-mode", choices=["full", "stage-local", "stage_local"], default="full")
    real_shard_beta.add_argument("--prompt-texts", default="CrowdTensor routes home CPU,A miner returns one token")
    real_shard_beta.add_argument(
        "--failure-mode",
        choices=["none", "kill-stage-after-claim", "kill-stage0-after-claim", "kill-stage1-after-claim"],
        default="none",
    )
    real_shard_beta.add_argument("--stage-mode", choices=["both", "split"], default="split")
    real_shard_beta.add_argument("--require-distinct-stage-miners", action="store_true")
    real_shard_beta.add_argument("--coordinator-url", default="")
    real_shard_beta.add_argument("--observer-token", default="")
    real_shard_beta.add_argument("--admin-token", default="")
    real_shard_beta.add_argument("--timeout-seconds", type=int, default=300)
    real_shard_beta.add_argument("--remote-timeout-seconds", type=float, default=180.0)
    real_shard_beta.add_argument("--json", action="store_true")
    micro_live_rc = subparsers.add_parser(
        "micro-llm-live-rc",
        help="Run the stage-aware micro-LLM live two-node RC proof.",
    )
    micro_live_rc.add_argument("--mode", choices=["local-generated", "external-existing"], default="local-generated")
    micro_live_rc.add_argument("--output-dir", default="dist/micro-llm-live-rc")
    micro_live_rc.add_argument("--kaggle-output-dir", default="")
    micro_live_rc.add_argument("--public-host", default="24.199.118.54")
    micro_live_rc.add_argument("--port", type=int, default=9180)
    micro_live_rc.add_argument("--coordinator-url", default="")
    micro_live_rc.add_argument("--miner-id", default="kaggle-cpu-1")
    micro_live_rc.add_argument("--request-count", type=int, default=2)
    micro_live_rc.add_argument("--decode-steps", type=int, default=3)
    micro_live_rc.add_argument("--micro-llm-artifact", default="")
    micro_live_rc.add_argument("--timeout-seconds", type=int, default=240)
    micro_live_rc.add_argument("--remote-timeout-seconds", type=float, default=180.0)
    micro_live_rc.add_argument("--startup-timeout", type=float, default=20.0)
    micro_live_rc.add_argument("--process-exit-timeout", type=float, default=10.0)
    micro_live_rc.add_argument("--poll-interval", type=float, default=1.0)
    micro_live_rc.add_argument("--http-timeout", type=float, default=5.0)
    micro_live_rc.add_argument("--artifact-timeout", type=float, default=60.0)
    micro_live_rc.add_argument("--admin-results-limit", type=int, default=10)
    micro_live_rc.add_argument("--lease-seconds", type=float, default=15.0)
    micro_live_rc.add_argument("--compute-seconds", type=float, default=0.2)
    micro_live_rc.add_argument("--heartbeat-interval", type=float, default=0.1)
    micro_live_rc.add_argument("--idle-sleep", type=float, default=0.2)
    micro_live_rc.add_argument("--max-request-attempts", type=int, default=20)
    micro_live_rc.add_argument("--observer-token", default="")
    micro_live_rc.add_argument("--admin-token", default="")
    micro_live_rc.add_argument("--json", action="store_true")
    real_live_rc = subparsers.add_parser(
        "real-llm-live-rc",
        help="Run the real small-LLM live two-node RC proof with generated stage upload packages.",
    )
    real_live_rc.add_argument("--mode", choices=["local-generated", "kaggle-generated", "external-existing"], default="local-generated")
    real_live_rc.add_argument("--output-dir", default="dist/real-llm-live-rc")
    real_live_rc.add_argument("--public-host", default="24.199.118.54")
    real_live_rc.add_argument("--bind-host", default="0.0.0.0")
    real_live_rc.add_argument("--port", type=int, default=9184)
    real_live_rc.add_argument("--coordinator-url", default="")
    real_live_rc.add_argument("--miner-id", default="kaggle-real-llm")
    real_live_rc.add_argument("--request-count", type=int, default=1)
    real_live_rc.add_argument("--max-new-tokens", type=int, default=1)
    real_live_rc.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    real_live_rc.add_argument("--hf-cache-dir", default="")
    real_live_rc.add_argument("--timeout-seconds", type=int, default=300)
    real_live_rc.add_argument("--remote-timeout-seconds", type=float, default=180.0)
    real_live_rc.add_argument("--startup-timeout", type=float, default=30.0)
    real_live_rc.add_argument("--process-exit-timeout", type=float, default=20.0)
    real_live_rc.add_argument("--poll-interval", type=float, default=1.0)
    real_live_rc.add_argument("--http-timeout", type=float, default=30.0)
    real_live_rc.add_argument("--lease-seconds", type=float, default=15.0)
    real_live_rc.add_argument("--compute-seconds", type=float, default=0.2)
    real_live_rc.add_argument("--heartbeat-interval", type=float, default=0.1)
    real_live_rc.add_argument("--idle-sleep", type=float, default=0.5)
    real_live_rc.add_argument("--max-request-attempts", type=int, default=120)
    real_live_rc.add_argument("--observer-token", default="")
    real_live_rc.add_argument("--admin-token", default="")
    real_live_rc.add_argument("--json", action="store_true")
    real_internet_alpha = subparsers.add_parser(
        "real-llm-internet-alpha",
        help="Run the real Internet Swarm Inference Alpha proof with local requeue and external verification modes.",
    )
    real_internet_alpha.add_argument("--mode", choices=["local-generated", "package", "external-existing"], default="local-generated")
    real_internet_alpha.add_argument("--output-dir", default="dist/real-llm-internet-alpha")
    real_internet_alpha.add_argument("--public-host", default="24.199.118.54")
    real_internet_alpha.add_argument("--bind-host", default="0.0.0.0")
    real_internet_alpha.add_argument("--port", type=int, default=9186)
    real_internet_alpha.add_argument("--base-port", type=int, default=9188)
    real_internet_alpha.add_argument("--coordinator-url", default="")
    real_internet_alpha.add_argument("--miner-id", default="internet-real-llm")
    real_internet_alpha.add_argument("--request-count", type=int, default=1)
    real_internet_alpha.add_argument("--max-new-tokens", type=int, default=1)
    real_internet_alpha.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    real_internet_alpha.add_argument("--hf-cache-dir", default="")
    real_internet_alpha.add_argument("--timeout-seconds", type=int, default=300)
    real_internet_alpha.add_argument("--remote-timeout-seconds", type=float, default=180.0)
    real_internet_alpha.add_argument("--startup-timeout", type=float, default=30.0)
    real_internet_alpha.add_argument("--process-exit-timeout", type=float, default=20.0)
    real_internet_alpha.add_argument("--poll-interval", type=float, default=1.0)
    real_internet_alpha.add_argument("--http-timeout", type=float, default=30.0)
    real_internet_alpha.add_argument("--lease-seconds", type=float, default=15.0)
    real_internet_alpha.add_argument("--compute-seconds", type=float, default=0.2)
    real_internet_alpha.add_argument("--heartbeat-interval", type=float, default=0.1)
    real_internet_alpha.add_argument("--idle-sleep", type=float, default=0.5)
    real_internet_alpha.add_argument("--max-request-attempts", type=int, default=120)
    real_internet_alpha.add_argument("--observer-token", default="")
    real_internet_alpha.add_argument("--admin-token", default="")
    real_internet_alpha.add_argument("--skip-requeue", action="store_true")
    real_internet_alpha.add_argument("--json", action="store_true")
    real_internet_beta = subparsers.add_parser(
        "real-llm-internet-beta",
        help="Run the real Internet Swarm Inference Beta Kaggle automation with cleanup-backed evidence.",
    )
    real_internet_beta.add_argument("--mode", choices=["kaggle-auto"], default="kaggle-auto")
    real_internet_beta.add_argument("--output-dir", default="dist/real-llm-internet-beta-kaggle-auto")
    real_internet_beta.add_argument("--public-host", default="24.199.118.54")
    real_internet_beta.add_argument("--bind-host", default="0.0.0.0")
    real_internet_beta.add_argument("--port", type=int, default=9190)
    real_internet_beta.add_argument("--base-port", type=int, default=9191)
    real_internet_beta.add_argument("--ready-url", default="")
    real_internet_beta.add_argument("--coordinator-url", default="")
    real_internet_beta.add_argument("--miner-id", default="internet-real-llm-beta")
    real_internet_beta.add_argument("--request-count", type=int, default=2)
    real_internet_beta.add_argument("--max-new-tokens", type=int, default=1)
    real_internet_beta.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    real_internet_beta.add_argument("--hf-cache-dir", default="")
    real_internet_beta.add_argument("--kaggle-owner", default="")
    real_internet_beta.add_argument("--dataset-slug", default="")
    real_internet_beta.add_argument("--dataset-title", default="CrowdTensor Real LLM Internet Beta Package")
    real_internet_beta.add_argument("--kernel-slug-prefix", default="")
    real_internet_beta.add_argument("--kernel-title-prefix", default="CrowdTensor Real LLM Internet Beta Miner")
    real_internet_beta.add_argument("--inline-kernel-payload", dest="inline_kernel_payload", action="store_true", default=True)
    real_internet_beta.add_argument("--no-inline-kernel-payload", dest="inline_kernel_payload", action="store_false")
    real_internet_beta.add_argument("--skip-kaggle-cleanup", action="store_true")
    real_internet_beta.add_argument("--failure-mode", choices=["none", "kill-stage0-after-claim", "kill-stage1-after-claim"], default="none")
    real_internet_beta.add_argument("--timeout-seconds", type=float, default=900.0)
    real_internet_beta.add_argument("--remote-timeout-seconds", type=float, default=720.0)
    real_internet_beta.add_argument("--startup-timeout", type=float, default=45.0)
    real_internet_beta.add_argument("--process-exit-timeout", type=float, default=20.0)
    real_internet_beta.add_argument("--poll-interval", type=float, default=1.0)
    real_internet_beta.add_argument("--http-timeout", type=float, default=30.0)
    real_internet_beta.add_argument("--kaggle-push-timeout-seconds", type=float, default=180.0)
    real_internet_beta.add_argument("--kaggle-delete-timeout-seconds", type=float, default=120.0)
    real_internet_beta.add_argument("--kaggle-status-timeout-seconds", type=float, default=900.0)
    real_internet_beta.add_argument("--kaggle-status-poll-interval", type=float, default=15.0)
    real_internet_beta.add_argument("--lease-seconds", type=float, default=15.0)
    real_internet_beta.add_argument("--compute-seconds", type=float, default=0.2)
    real_internet_beta.add_argument("--victim-compute-seconds", type=float, default=45.0)
    real_internet_beta.add_argument("--heartbeat-interval", type=float, default=0.1)
    real_internet_beta.add_argument("--idle-sleep", type=float, default=0.5)
    real_internet_beta.add_argument("--claim-observe-timeout", type=float, default=180.0)
    real_internet_beta.add_argument("--requeue-timeout", type=float, default=120.0)
    real_internet_beta.add_argument("--max-request-attempts", type=int, default=240)
    real_internet_beta.add_argument("--json", action="store_true")
    swarm = subparsers.add_parser(
        "swarm-infer-beta",
        help="Prepare, run, verify, collect, or clean the user-facing real tiny-LLM Swarm Inference Beta.",
    )
    swarm_subparsers = swarm.add_subparsers(dest="swarm_action", required=True)

    def add_swarm_common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--output-dir", default="dist/swarm-inference-beta")
        target.add_argument("--coordinator-url", default="http://127.0.0.1:9200")
        target.add_argument("--port", type=int, default=9200)
        target.add_argument("--bind-host", default="0.0.0.0")
        target.add_argument("--miner-id-prefix", default="swarm-beta")
        target.add_argument("--request-count", type=int, default=2)
        target.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
        target.add_argument("--hf-cache-dir", default="")
        target.add_argument("--timeout-seconds", type=float, default=360.0)
        target.add_argument("--remote-timeout-seconds", type=float, default=240.0)
        target.add_argument("--http-timeout", type=float, default=30.0)
        target.add_argument("--json", action="store_true")

    swarm_prepare = swarm_subparsers.add_parser("prepare", help="Create operator runbook and stage0/stage1 Miner join packs.")
    add_swarm_common(swarm_prepare)
    swarm_prepare.add_argument("--observer-token", default="")
    swarm_prepare.add_argument("--admin-token", default="")
    swarm_prepare.add_argument("--lease-seconds", type=float, default=15.0)
    swarm_prepare.add_argument("--compute-seconds", type=float, default=0.2)
    swarm_prepare.add_argument("--heartbeat-interval", type=float, default=0.1)
    swarm_prepare.add_argument("--max-request-attempts", type=int, default=120)
    swarm_prepare.add_argument("--replace", action="store_true")

    swarm_coordinator = swarm_subparsers.add_parser("coordinator", help="Print or run the generated Coordinator command.")
    add_swarm_common(swarm_coordinator)
    swarm_coordinator.add_argument("--observer-token", default="")
    swarm_coordinator.add_argument("--admin-token", default="")
    swarm_coordinator.add_argument("--lease-seconds", type=float, default=15.0)
    swarm_coordinator.add_argument("--run", action="store_true")

    swarm_miner = swarm_subparsers.add_parser("miner", help="Print or run a generated stage Miner command.")
    add_swarm_common(swarm_miner)
    swarm_miner.add_argument("--stage", choices=["stage0", "stage1"], required=True)
    swarm_miner.add_argument("--compute-seconds", type=float, default=0.2)
    swarm_miner.add_argument("--heartbeat-interval", type=float, default=0.1)
    swarm_miner.add_argument("--max-request-attempts", type=int, default=120)
    swarm_miner.add_argument("--run", action="store_true")

    swarm_verify = swarm_subparsers.add_parser("verify", help="Verify a running two-stage Swarm Inference Beta session.")
    add_swarm_common(swarm_verify)
    swarm_verify.add_argument("--observer-token", default="")
    swarm_verify.add_argument("--admin-token", default="")
    swarm_verify.add_argument("--prompt-texts", default="CrowdTensor routes home CPU,A miner returns one token")
    swarm_verify.add_argument("--real-internet-beta-report", default="")

    swarm_collect = swarm_subparsers.add_parser("collect", help="Collect redacted evidence from a running Swarm Inference Beta.")
    add_swarm_common(swarm_collect)
    swarm_collect.add_argument("--observer-token", default="")
    swarm_collect.add_argument("--admin-token", default="")
    swarm_collect.add_argument("--miner-id", default="")
    swarm_collect.add_argument("--artifact-timeout", type=float, default=60.0)

    swarm_live = swarm_subparsers.add_parser("live", help="Run the side-effectful public Kaggle auto proof for Swarm Inference Beta.")
    swarm_live.add_argument("--output-dir", default="dist/swarm-inference-beta-live")
    swarm_live.add_argument("--public-host", default="24.199.118.54")
    swarm_live.add_argument("--bind-host", default="0.0.0.0")
    swarm_live.add_argument("--port", type=int, default=9210)
    swarm_live.add_argument("--base-port", type=int, default=9211)
    swarm_live.add_argument("--ready-url", default="")
    swarm_live.add_argument("--coordinator-url", default="")
    swarm_live.add_argument("--miner-id-prefix", default="swarm-beta-live")
    swarm_live.add_argument("--request-count", type=int, default=2)
    swarm_live.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    swarm_live.add_argument("--hf-cache-dir", default="")
    swarm_live.add_argument("--kaggle-owner", default="")
    swarm_live.add_argument("--dataset-slug", default="")
    swarm_live.add_argument("--dataset-title", default="CrowdTensor Swarm Inference Beta Live")
    swarm_live.add_argument("--kernel-slug-prefix", default="crowdtensor-swarm-inference-beta-live")
    swarm_live.add_argument("--kernel-title-prefix", default="CrowdTensor Swarm Inference Beta Live")
    swarm_live.add_argument("--inline-kernel-payload", action=argparse.BooleanOptionalAction, default=True)
    swarm_live.add_argument("--skip-kaggle-cleanup", action="store_true")
    swarm_live.add_argument("--keep-live-private-artifacts", action="store_true")
    swarm_live.add_argument("--failure-mode", choices=["none", "kill-stage0-after-claim", "kill-stage1-after-claim"], default="none")
    swarm_live.add_argument("--timeout-seconds", type=float, default=300.0)
    swarm_live.add_argument("--remote-timeout-seconds", type=float, default=300.0)
    swarm_live.add_argument("--startup-timeout", type=float, default=60.0)
    swarm_live.add_argument("--process-exit-timeout", type=float, default=10.0)
    swarm_live.add_argument("--poll-interval", type=float, default=1.0)
    swarm_live.add_argument("--http-timeout", type=float, default=30.0)
    swarm_live.add_argument("--kaggle-push-timeout-seconds", type=float, default=180.0)
    swarm_live.add_argument("--kaggle-delete-timeout-seconds", type=float, default=120.0)
    swarm_live.add_argument("--kaggle-status-timeout-seconds", type=float, default=300.0)
    swarm_live.add_argument("--kaggle-status-poll-interval", type=float, default=5.0)
    swarm_live.add_argument("--lease-seconds", type=float, default=15.0)
    swarm_live.add_argument("--compute-seconds", type=float, default=0.2)
    swarm_live.add_argument("--victim-compute-seconds", type=float, default=45.0)
    swarm_live.add_argument("--heartbeat-interval", type=float, default=0.1)
    swarm_live.add_argument("--idle-sleep", type=float, default=0.2)
    swarm_live.add_argument("--claim-observe-timeout", type=float, default=180.0)
    swarm_live.add_argument("--requeue-timeout", type=float, default=120.0)
    swarm_live.add_argument("--max-request-attempts", type=int, default=240)
    swarm_live.add_argument("--json", action="store_true")

    swarm_clean = swarm_subparsers.add_parser("clean", help="Dry-run or delete known Swarm Inference Beta artifacts.")
    swarm_clean.add_argument("--output-dir", default="dist/swarm-inference-beta")
    swarm_clean.add_argument("--timeout-seconds", type=float, default=60.0)
    swarm_clean.add_argument("--apply", action="store_true")
    swarm_clean.add_argument("--include-private", action="store_true")
    swarm_clean.add_argument("--remove-empty-dir", action="store_true")
    swarm_clean.add_argument("--json", action="store_true")
    swarm_session = subparsers.add_parser(
        "swarm-session",
        help="Run the Public Swarm Inference Alpha session wrapper around real tiny-LLM split inference.",
    )
    swarm_session.add_argument("--mode", choices=["local-generated", "live-kaggle"], default="live-kaggle")
    swarm_session.add_argument("--output-dir", default="dist/public-swarm-inference-alpha")
    swarm_session.add_argument("--public-host", default="24.199.118.54")
    swarm_session.add_argument("--bind-host", default="0.0.0.0")
    swarm_session.add_argument("--port", type=int, default=9220)
    swarm_session.add_argument("--base-port", type=int, default=9221)
    swarm_session.add_argument("--ready-url", default="")
    swarm_session.add_argument("--coordinator-url", default="")
    swarm_session.add_argument("--miner-id-prefix", default="public-swarm-alpha")
    swarm_session.add_argument("--request-count", type=int, default=2)
    swarm_session.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    swarm_session.add_argument("--hf-cache-dir", default="")
    swarm_session.add_argument("--failure-mode", choices=["none", "kill-stage0-after-claim", "kill-stage1-after-claim"], default="kill-stage0-after-claim")
    swarm_session.add_argument("--skip-local-requeue", action="store_true")
    swarm_session.add_argument("--kaggle-owner", default="")
    swarm_session.add_argument("--dataset-slug", default="")
    swarm_session.add_argument("--dataset-title", default="CrowdTensor Public Swarm Inference Alpha")
    swarm_session.add_argument("--kernel-slug-prefix", default="crowdtensor-public-swarm-alpha")
    swarm_session.add_argument("--kernel-title-prefix", default="CrowdTensor Public Swarm Inference Alpha")
    swarm_session.add_argument("--inline-kernel-payload", action=argparse.BooleanOptionalAction, default=True)
    swarm_session.add_argument("--skip-kaggle-cleanup", action="store_true")
    swarm_session.add_argument("--keep-live-private-artifacts", action="store_true")
    swarm_session.add_argument("--keep-child-artifacts", action="store_true")
    swarm_session.add_argument("--timeout-seconds", type=float, default=300.0)
    swarm_session.add_argument("--remote-timeout-seconds", type=float, default=300.0)
    swarm_session.add_argument("--startup-timeout", type=float, default=60.0)
    swarm_session.add_argument("--process-exit-timeout", type=float, default=10.0)
    swarm_session.add_argument("--poll-interval", type=float, default=1.0)
    swarm_session.add_argument("--http-timeout", type=float, default=30.0)
    swarm_session.add_argument("--kaggle-push-timeout-seconds", type=float, default=180.0)
    swarm_session.add_argument("--kaggle-delete-timeout-seconds", type=float, default=120.0)
    swarm_session.add_argument("--kaggle-status-timeout-seconds", type=float, default=300.0)
    swarm_session.add_argument("--kaggle-status-poll-interval", type=float, default=5.0)
    swarm_session.add_argument("--lease-seconds", type=float, default=15.0)
    swarm_session.add_argument("--compute-seconds", type=float, default=0.2)
    swarm_session.add_argument("--victim-compute-seconds", type=float, default=45.0)
    swarm_session.add_argument("--heartbeat-interval", type=float, default=0.1)
    swarm_session.add_argument("--idle-sleep", type=float, default=0.2)
    swarm_session.add_argument("--claim-observe-timeout", type=float, default=180.0)
    swarm_session.add_argument("--requeue-timeout", type=float, default=120.0)
    swarm_session.add_argument("--max-request-attempts", type=int, default=240)
    swarm_session.add_argument("--json", action="store_true")
    public_swarm_rc = subparsers.add_parser(
        "public-swarm-alpha-rc",
        help="Build the Public Swarm Inference Alpha release-candidate evidence artifact.",
    )
    public_swarm_rc.add_argument("--mode", choices=["evidence-import", "local-smoke"], default="evidence-import")
    public_swarm_rc.add_argument("--output-dir", default="dist/public-swarm-inference-alpha-rc")
    public_swarm_rc.add_argument(
        "--stage0-report",
        default=(
            "dist/public-swarm-inference-alpha-live-stage0-requeue-20260527165830/"
            "public_swarm_inference_alpha.json"
        ),
    )
    public_swarm_rc.add_argument(
        "--stage1-report",
        default=(
            "dist/public-swarm-inference-alpha-live-stage1-requeue-20260527170600/"
            "public_swarm_inference_alpha.json"
        ),
    )
    public_swarm_rc.add_argument("--summary-report", default="dist/public-swarm-inference-alpha-live-requeue-summary.json")
    public_swarm_rc.add_argument("--request-count", type=int, default=2)
    public_swarm_rc.add_argument("--timeout-seconds", type=float, default=120.0)
    public_swarm_rc.add_argument("--json", action="store_true")
    public_swarm_beta = subparsers.add_parser(
        "public-swarm-beta",
        help="Prepare, run, verify, collect, or validate Public Swarm Inference Beta.",
    )
    public_beta_subparsers = public_swarm_beta.add_subparsers(dest="public_swarm_beta_action", required=True)

    def add_public_beta_base(target: argparse.ArgumentParser) -> None:
        target.add_argument("--output-dir", default="dist/public-swarm-inference-beta")
        target.add_argument("--request-count", type=int, default=1)
        target.add_argument("--timeout-seconds", type=float, default=300.0)
        target.add_argument("--json", action="store_true")

    def add_public_beta_runtime(target: argparse.ArgumentParser) -> None:
        target.add_argument("--coordinator-url", default="http://127.0.0.1:9200")
        target.add_argument("--port", type=int, default=9200)
        target.add_argument("--base-port", type=int, default=9290)
        target.add_argument("--bind-host", default="0.0.0.0")
        target.add_argument("--miner-id-prefix", default="public-swarm-beta")
        target.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
        target.add_argument("--hf-cache-dir", default="")
        target.add_argument("--remote-timeout-seconds", type=float, default=240.0)
        target.add_argument("--http-timeout", type=float, default=30.0)

    def add_public_beta_tokens(target: argparse.ArgumentParser) -> None:
        target.add_argument("--observer-token", default="")
        target.add_argument("--admin-token", default="")

    public_beta_prepare = public_beta_subparsers.add_parser("prepare", help="Create Public Beta stage0/stage1 join packs.")
    add_public_beta_base(public_beta_prepare)
    add_public_beta_runtime(public_beta_prepare)
    add_public_beta_tokens(public_beta_prepare)
    public_beta_prepare.add_argument("--lease-seconds", type=float, default=15.0)
    public_beta_prepare.add_argument("--compute-seconds", type=float, default=0.2)
    public_beta_prepare.add_argument("--heartbeat-interval", type=float, default=0.1)
    public_beta_prepare.add_argument("--max-request-attempts", type=int, default=120)
    public_beta_prepare.add_argument("--replace", action="store_true")

    public_beta_coordinator = public_beta_subparsers.add_parser("coordinator", help="Print or run the Public Beta Coordinator command.")
    add_public_beta_base(public_beta_coordinator)
    add_public_beta_runtime(public_beta_coordinator)
    add_public_beta_tokens(public_beta_coordinator)
    public_beta_coordinator.add_argument("--lease-seconds", type=float, default=15.0)
    public_beta_coordinator.add_argument("--run", action="store_true")

    public_beta_miner = public_beta_subparsers.add_parser("miner", help="Print or run a Public Beta stage Miner command.")
    add_public_beta_base(public_beta_miner)
    add_public_beta_runtime(public_beta_miner)
    public_beta_miner.add_argument("--stage", choices=["stage0", "stage1"], required=True)
    public_beta_miner.add_argument("--compute-seconds", type=float, default=0.2)
    public_beta_miner.add_argument("--heartbeat-interval", type=float, default=0.1)
    public_beta_miner.add_argument("--max-request-attempts", type=int, default=120)
    public_beta_miner.add_argument("--run", action="store_true")

    public_beta_verify = public_beta_subparsers.add_parser("verify", help="Verify a running Public Beta two-stage session.")
    add_public_beta_base(public_beta_verify)
    add_public_beta_runtime(public_beta_verify)
    add_public_beta_tokens(public_beta_verify)
    public_beta_verify.add_argument("--prompt-texts", default="CrowdTensor routes home CPU,A miner returns one token")
    public_beta_verify.add_argument("--real-internet-beta-report", default="")

    public_beta_collect = public_beta_subparsers.add_parser("collect", help="Collect redacted Public Beta evidence.")
    add_public_beta_base(public_beta_collect)
    add_public_beta_runtime(public_beta_collect)
    add_public_beta_tokens(public_beta_collect)
    public_beta_collect.add_argument("--miner-id", default="")
    public_beta_collect.add_argument("--artifact-timeout", type=float, default=60.0)

    public_beta_clean = public_beta_subparsers.add_parser("clean", help="Dry-run or delete known Public Beta generated files.")
    add_public_beta_base(public_beta_clean)
    public_beta_clean.add_argument("--apply", action="store_true")
    public_beta_clean.add_argument("--include-private", action="store_true")
    public_beta_clean.add_argument("--remove-empty-dir", action="store_true")

    public_beta_product = public_beta_subparsers.add_parser("product-beta", help="Validate the product-shaped Public Swarm Inference Beta aggregate.")
    add_public_beta_base(public_beta_product)
    public_beta_product.add_argument("--base-port", type=int, default=9290)
    public_beta_product.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    public_beta_product.add_argument(
        "--gpu-report",
        default=(
            "dist/gpu-sharded-generation-beta-kaggle-20260528095658/"
            "gpu_sharded_generation_beta_kaggle_auto.json"
        ),
    )
    public_beta_product.add_argument("--max-new-tokens", type=int, default=16)
    public_beta_product.add_argument("--cpu-request-count", type=int, default=2)
    public_beta_product.add_argument("--external-llm-request-count", type=int, default=1)
    public_beta_product.add_argument("--scenario-id", default="route-baseline")
    public_beta_product.add_argument("--cpu-timeout-seconds", type=float, default=180.0)

    public_beta_loopback = public_beta_subparsers.add_parser("local-loopback", help="Run a fresh local two-stage CPU tiny GPT split proof.")
    add_public_beta_base(public_beta_loopback)
    add_public_beta_runtime(public_beta_loopback)
    public_beta_loopback.add_argument("--prompt-texts", default="CrowdTensor routes home CPU,A miner returns one token")

    public_beta_import = public_beta_subparsers.add_parser("evidence-import", help="Import retained Public Swarm Alpha RC live evidence.")
    add_public_beta_base(public_beta_import)
    public_beta_import.add_argument("--alpha-rc-report", default="dist/public-swarm-inference-alpha-rc/public_swarm_inference_alpha_rc.json")
    public_beta_import.add_argument(
        "--stage0-report",
        default=(
            "dist/public-swarm-inference-alpha-live-stage0-requeue-20260527165830/"
            "public_swarm_inference_alpha.json"
        ),
    )
    public_beta_import.add_argument(
        "--stage1-report",
        default=(
            "dist/public-swarm-inference-alpha-live-stage1-requeue-20260527170600/"
            "public_swarm_inference_alpha.json"
        ),
    )
    public_beta_import.add_argument("--summary-report", default="dist/public-swarm-inference-alpha-live-requeue-summary.json")
    public_beta_import.add_argument("--allow-missing-live-evidence", action="store_true")

    public_swarm_beta_rc = subparsers.add_parser(
        "public-swarm-beta-rc",
        help="Build the Coordinator-backed Public Swarm Inference Beta RC artifact.",
    )
    public_swarm_beta_rc.add_argument("public_swarm_beta_rc_mode", choices=["local-loopback", "package", "external-existing"])
    public_swarm_beta_rc.add_argument("--output-dir", default="dist/public-swarm-inference-beta-rc")
    public_swarm_beta_rc.add_argument("--base-port", type=int, default=9310)
    public_swarm_beta_rc.add_argument("--port", type=int, default=9310)
    public_swarm_beta_rc.add_argument("--public-host", default="127.0.0.1")
    public_swarm_beta_rc.add_argument("--bind-host", default="127.0.0.1")
    public_swarm_beta_rc.add_argument("--coordinator-url", default="")
    public_swarm_beta_rc.add_argument("--target", choices=["local", "kaggle"], default="local")
    public_swarm_beta_rc.add_argument("--miner-id-prefix", default="public-swarm-beta-rc")
    public_swarm_beta_rc.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    public_swarm_beta_rc.add_argument("--hf-cache-dir", default="")
    public_swarm_beta_rc.add_argument(
        "--gpu-report",
        default=(
            "dist/gpu-sharded-generation-beta-kaggle-20260528095658/"
            "gpu_sharded_generation_beta_kaggle_auto.json"
        ),
    )
    public_swarm_beta_rc.add_argument("--prompt-text", default="CrowdTensor public beta RC")
    public_swarm_beta_rc.add_argument("--scenario-id", default="route-baseline")
    public_swarm_beta_rc.add_argument("--request-count", type=int, default=1)
    public_swarm_beta_rc.add_argument("--max-new-tokens", type=int, default=2)
    public_swarm_beta_rc.add_argument("--cpu-request-count", type=int, default=1)
    public_swarm_beta_rc.add_argument("--external-llm-request-count", type=int, default=1)
    public_swarm_beta_rc.add_argument("--observer-token", default="")
    public_swarm_beta_rc.add_argument("--admin-token", default="")
    public_swarm_beta_rc.add_argument("--timeout-seconds", type=float, default=300.0)
    public_swarm_beta_rc.add_argument("--remote-timeout-seconds", type=float, default=180.0)
    public_swarm_beta_rc.add_argument("--cpu-timeout-seconds", type=float, default=180.0)
    public_swarm_beta_rc.add_argument("--startup-timeout", type=float, default=45.0)
    public_swarm_beta_rc.add_argument("--process-exit-timeout", type=float, default=20.0)
    public_swarm_beta_rc.add_argument("--poll-interval", type=float, default=1.0)
    public_swarm_beta_rc.add_argument("--http-timeout", type=float, default=10.0)
    public_swarm_beta_rc.add_argument("--json", action="store_true")

    public_swarm_product_beta = subparsers.add_parser(
        "public-swarm-product-beta",
        help="Build the user-facing Public Swarm Product Beta artifact.",
    )
    public_swarm_product_beta.add_argument("public_swarm_product_beta_mode", choices=["local-loopback", "package", "external-existing"])
    public_swarm_product_beta.add_argument("--output-dir", default="dist/public-swarm-product-beta")
    public_swarm_product_beta.add_argument("--base-port", type=int, default=9320)
    public_swarm_product_beta.add_argument("--port", type=int, default=9320)
    public_swarm_product_beta.add_argument("--public-host", default="127.0.0.1")
    public_swarm_product_beta.add_argument("--bind-host", default="127.0.0.1")
    public_swarm_product_beta.add_argument("--coordinator-url", default="")
    public_swarm_product_beta.add_argument("--target", choices=["local", "kaggle"], default="local")
    public_swarm_product_beta.add_argument("--miner-id-prefix", default="public-swarm-product-beta")
    public_swarm_product_beta.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    public_swarm_product_beta.add_argument("--hf-cache-dir", default="")
    public_swarm_product_beta.add_argument(
        "--gpu-report",
        default=(
            "dist/gpu-sharded-generation-beta-kaggle-20260528095658/"
            "gpu_sharded_generation_beta_kaggle_auto.json"
        ),
    )
    public_swarm_product_beta.add_argument("--prompt-text", default="CrowdTensor product beta")
    public_swarm_product_beta.add_argument("--scenario-id", default="route-baseline")
    public_swarm_product_beta.add_argument("--request-count", type=int, default=1)
    public_swarm_product_beta.add_argument("--max-new-tokens", type=int, default=2)
    public_swarm_product_beta.add_argument("--cpu-request-count", type=int, default=1)
    public_swarm_product_beta.add_argument("--external-llm-request-count", type=int, default=1)
    public_swarm_product_beta.add_argument("--observer-token", default="")
    public_swarm_product_beta.add_argument("--admin-token", default="")
    public_swarm_product_beta.add_argument("--timeout-seconds", type=float, default=300.0)
    public_swarm_product_beta.add_argument("--remote-timeout-seconds", type=float, default=180.0)
    public_swarm_product_beta.add_argument("--cpu-timeout-seconds", type=float, default=180.0)
    public_swarm_product_beta.add_argument("--startup-timeout", type=float, default=45.0)
    public_swarm_product_beta.add_argument("--process-exit-timeout", type=float, default=20.0)
    public_swarm_product_beta.add_argument("--poll-interval", type=float, default=1.0)
    public_swarm_product_beta.add_argument("--http-timeout", type=float, default=10.0)
    public_swarm_product_beta.add_argument("--json", action="store_true")

    public_swarm_gpu_beta = subparsers.add_parser(
        "public-swarm-gpu-beta",
        help="Prepare, smoke-check, or validate optional CUDA Public Swarm Inference Beta.",
    )
    public_gpu_subparsers = public_swarm_gpu_beta.add_subparsers(dest="public_swarm_gpu_beta_action", required=True)

    def add_public_gpu_base(target: argparse.ArgumentParser) -> None:
        target.add_argument("--output-dir", default="dist/public-swarm-gpu-inference-beta")
        target.add_argument("--request-count", type=int, default=1)
        target.add_argument("--timeout-seconds", type=float, default=300.0)
        target.add_argument("--json", action="store_true")

    def add_public_gpu_runtime(target: argparse.ArgumentParser) -> None:
        target.add_argument("--coordinator-url", default="http://127.0.0.1:9300")
        target.add_argument("--public-host", default="24.199.118.54")
        target.add_argument("--port", type=int, default=9320)
        target.add_argument("--base-port", type=int, default=9321)
        target.add_argument("--bind-host", default="0.0.0.0")
        target.add_argument("--miner-id-prefix", default="public-swarm-gpu-beta")
        target.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
        target.add_argument("--hf-cache-dir", default="")
        target.add_argument("--real-llm-partition-mode", choices=["full", "stage-local", "stage_local"], default="stage-local")
        target.add_argument("--remote-timeout-seconds", type=float, default=240.0)
        target.add_argument("--http-timeout", type=float, default=30.0)

    def add_public_gpu_tokens(target: argparse.ArgumentParser) -> None:
        target.add_argument("--observer-token", default="")
        target.add_argument("--admin-token", default="")

    public_gpu_smoke = public_gpu_subparsers.add_parser("local-smoke", help="Run CI-safe CUDA availability diagnostics without claiming GPU readiness.")
    add_public_gpu_base(public_gpu_smoke)
    add_public_gpu_runtime(public_gpu_smoke)

    public_gpu_loopback = public_gpu_subparsers.add_parser("local-loopback", help="Run a local CUDA tiny GPT split proof when CUDA is available.")
    add_public_gpu_base(public_gpu_loopback)
    add_public_gpu_runtime(public_gpu_loopback)
    public_gpu_loopback.add_argument("--prompt-texts", default="CrowdTensor routes home CPU,A miner returns one token")

    public_gpu_package = public_gpu_subparsers.add_parser("kaggle-package", help="Generate a private Kaggle GPU runbook/template package.")
    add_public_gpu_base(public_gpu_package)
    add_public_gpu_runtime(public_gpu_package)

    public_gpu_auto = public_gpu_subparsers.add_parser("kaggle-auto", help="Run the side-effectful private Kaggle GPU two-stage proof.")
    add_public_gpu_base(public_gpu_auto)
    add_public_gpu_runtime(public_gpu_auto)
    public_gpu_auto.add_argument("--kaggle-owner", default="")
    public_gpu_auto.add_argument("--dataset-slug", default="")
    public_gpu_auto.add_argument("--dataset-title", default="CrowdTensor Public Swarm GPU Beta Package")
    public_gpu_auto.add_argument("--kernel-slug-prefix", default="")
    public_gpu_auto.add_argument("--kernel-title-prefix", default="CrowdTensor Public Swarm GPU Beta Miner")
    public_gpu_auto.add_argument("--inline-kernel-payload", dest="inline_kernel_payload", action="store_true", default=True)
    public_gpu_auto.add_argument("--no-inline-kernel-payload", dest="inline_kernel_payload", action="store_false")
    public_gpu_auto.add_argument("--skip-kaggle-cleanup", action="store_true")
    public_gpu_auto.add_argument("--startup-timeout", type=float, default=45.0)
    public_gpu_auto.add_argument("--process-exit-timeout", type=float, default=20.0)
    public_gpu_auto.add_argument("--poll-interval", type=float, default=1.0)
    public_gpu_auto.add_argument("--kaggle-push-timeout-seconds", type=float, default=180.0)
    public_gpu_auto.add_argument("--kaggle-delete-timeout-seconds", type=float, default=120.0)
    public_gpu_auto.add_argument("--kaggle-status-timeout-seconds", type=float, default=900.0)
    public_gpu_auto.add_argument("--kaggle-status-poll-interval", type=float, default=15.0)
    public_gpu_auto.add_argument("--lease-seconds", type=float, default=15.0)
    public_gpu_auto.add_argument("--compute-seconds", type=float, default=0.2)
    public_gpu_auto.add_argument("--victim-compute-seconds", type=float, default=45.0)
    public_gpu_auto.add_argument("--heartbeat-interval", type=float, default=0.1)
    public_gpu_auto.add_argument("--idle-sleep", type=float, default=0.5)
    public_gpu_auto.add_argument("--max-request-attempts", type=int, default=120)

    public_gpu_import = public_gpu_subparsers.add_parser("evidence-import", help="Import retained Public Swarm GPU Beta evidence.")
    add_public_gpu_base(public_gpu_import)
    public_gpu_import.add_argument("--gpu-report", required=True)

    public_gpu_prepare = public_gpu_subparsers.add_parser("prepare", help="Create operator workflow artifacts using the existing Public Beta shape.")
    add_public_gpu_base(public_gpu_prepare)
    add_public_gpu_runtime(public_gpu_prepare)
    add_public_gpu_tokens(public_gpu_prepare)
    public_gpu_prepare.add_argument("--lease-seconds", type=float, default=15.0)
    public_gpu_prepare.add_argument("--compute-seconds", type=float, default=0.2)
    public_gpu_prepare.add_argument("--heartbeat-interval", type=float, default=0.1)
    public_gpu_prepare.add_argument("--max-request-attempts", type=int, default=120)
    public_gpu_prepare.add_argument("--replace", action="store_true")

    public_gpu_coordinator = public_gpu_subparsers.add_parser("coordinator", help="Print or run the generated Coordinator command.")
    add_public_gpu_base(public_gpu_coordinator)
    add_public_gpu_runtime(public_gpu_coordinator)
    add_public_gpu_tokens(public_gpu_coordinator)
    public_gpu_coordinator.add_argument("--lease-seconds", type=float, default=15.0)
    public_gpu_coordinator.add_argument("--run", action="store_true")

    public_gpu_miner = public_gpu_subparsers.add_parser("miner", help="Print or run a generated stage Miner command.")
    add_public_gpu_base(public_gpu_miner)
    add_public_gpu_runtime(public_gpu_miner)
    public_gpu_miner.add_argument("--stage", choices=["stage0", "stage1"], required=True)
    public_gpu_miner.add_argument("--compute-seconds", type=float, default=0.2)
    public_gpu_miner.add_argument("--heartbeat-interval", type=float, default=0.1)
    public_gpu_miner.add_argument("--max-request-attempts", type=int, default=120)
    public_gpu_miner.add_argument("--run", action="store_true")

    public_gpu_verify = public_gpu_subparsers.add_parser("verify", help="Verify a running operator workflow.")
    add_public_gpu_base(public_gpu_verify)
    add_public_gpu_runtime(public_gpu_verify)
    add_public_gpu_tokens(public_gpu_verify)
    public_gpu_verify.add_argument("--prompt-texts", default="CrowdTensor routes home CPU,A miner returns one token")

    public_gpu_collect = public_gpu_subparsers.add_parser("collect", help="Collect redacted Public GPU Beta evidence.")
    add_public_gpu_base(public_gpu_collect)
    add_public_gpu_runtime(public_gpu_collect)
    add_public_gpu_tokens(public_gpu_collect)
    public_gpu_collect.add_argument("--miner-id", default="")
    public_gpu_collect.add_argument("--artifact-timeout", type=float, default=60.0)

    public_gpu_clean = public_gpu_subparsers.add_parser("clean", help="Dry-run or delete known generated files.")
    add_public_gpu_base(public_gpu_clean)
    public_gpu_clean.add_argument("--apply", action="store_true")
    public_gpu_clean.add_argument("--include-private", action="store_true")
    public_gpu_clean.add_argument("--remove-empty-dir", action="store_true")

    gpu_generate = subparsers.add_parser(
        "gpu-generate",
        help="Run or import the optional CUDA multi-machine sharded generation Beta.",
    )
    gpu_generate_subparsers = gpu_generate.add_subparsers(dest="gpu_generate_mode", required=True)

    def add_gpu_generate_base(target: argparse.ArgumentParser) -> None:
        target.add_argument("--output-dir", default="dist/gpu-sharded-generation-beta")
        target.add_argument("--request-count", type=int, default=1)
        target.add_argument("--max-new-tokens", type=int, default=16)
        target.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
        target.add_argument("--hf-cache-dir", default="")
        target.add_argument("--real-llm-partition-mode", choices=["stage-local", "stage_local"], default="stage-local")
        target.add_argument("--timeout-seconds", type=float, default=900.0)
        target.add_argument("--remote-timeout-seconds", type=float, default=900.0)
        target.add_argument("--json", action="store_true")

    def add_gpu_generate_runtime(target: argparse.ArgumentParser) -> None:
        target.add_argument("--public-host", default="24.199.118.54")
        target.add_argument("--port", type=int, default=9340)
        target.add_argument("--base-port", type=int, default=9341)
        target.add_argument("--bind-host", default="0.0.0.0")
        target.add_argument("--miner-id-prefix", default="gpu-sharded-generation-beta")
        target.add_argument("--http-timeout", type=float, default=30.0)

    gpu_generate_loopback = gpu_generate_subparsers.add_parser("local-loopback", help="Run a local CUDA split multi-token generation proof.")
    add_gpu_generate_base(gpu_generate_loopback)
    add_gpu_generate_runtime(gpu_generate_loopback)

    gpu_generate_auto = gpu_generate_subparsers.add_parser("kaggle-auto", help="Run the side-effectful private Kaggle GPU multi-token proof.")
    add_gpu_generate_base(gpu_generate_auto)
    add_gpu_generate_runtime(gpu_generate_auto)
    gpu_generate_auto.add_argument("--kaggle-owner", default="")
    gpu_generate_auto.add_argument("--dataset-slug", default="")
    gpu_generate_auto.add_argument("--dataset-title", default="CrowdTensor GPU Sharded Generation Beta Package")
    gpu_generate_auto.add_argument("--kernel-slug-prefix", default="")
    gpu_generate_auto.add_argument("--kernel-title-prefix", default="CrowdTensor GPU Sharded Generation Beta Miner")
    gpu_generate_auto.add_argument("--inline-kernel-payload", dest="inline_kernel_payload", action="store_true", default=True)
    gpu_generate_auto.add_argument("--no-inline-kernel-payload", dest="inline_kernel_payload", action="store_false")
    gpu_generate_auto.add_argument("--skip-kaggle-cleanup", action="store_true")
    gpu_generate_auto.add_argument("--startup-timeout", type=float, default=45.0)
    gpu_generate_auto.add_argument("--process-exit-timeout", type=float, default=20.0)
    gpu_generate_auto.add_argument("--poll-interval", type=float, default=1.0)
    gpu_generate_auto.add_argument("--kaggle-push-timeout-seconds", type=float, default=180.0)
    gpu_generate_auto.add_argument("--kaggle-delete-timeout-seconds", type=float, default=120.0)
    gpu_generate_auto.add_argument("--kaggle-status-timeout-seconds", type=float, default=1200.0)
    gpu_generate_auto.add_argument("--kaggle-status-poll-interval", type=float, default=15.0)
    gpu_generate_auto.add_argument("--lease-seconds", type=float, default=15.0)
    gpu_generate_auto.add_argument("--compute-seconds", type=float, default=0.2)
    gpu_generate_auto.add_argument("--victim-compute-seconds", type=float, default=45.0)
    gpu_generate_auto.add_argument("--heartbeat-interval", type=float, default=0.1)
    gpu_generate_auto.add_argument("--idle-sleep", type=float, default=0.5)
    gpu_generate_auto.add_argument("--max-request-attempts", type=int, default=240)

    gpu_generate_import = gpu_generate_subparsers.add_parser("evidence-import", help="Import retained GPU sharded generation evidence.")
    add_gpu_generate_base(gpu_generate_import)
    add_gpu_generate_runtime(gpu_generate_import)
    gpu_generate_import.add_argument("--gpu-report", required=True)

    release_ready = subparsers.add_parser(
        "release-ready",
        help="Build the Alpha maintainer release readiness report.",
    )
    release_ready.add_argument("--output-dir", default="dist/release-readiness")
    release_ready.add_argument("--host", default="127.0.0.1")
    release_ready.add_argument("--base-port", type=int, default=8924)
    release_ready.add_argument("--request-count", type=int, default=4)
    release_ready.add_argument("--external-llm-request-count", type=int, default=3)
    release_ready.add_argument("--timeout-seconds", type=int, default=180)
    release_ready.add_argument("--allow-dirty", action="store_true")
    release_ready.add_argument("--skip-external-llm-evidence", action="store_true")
    release_ready.add_argument("--runtime-report", default="")
    release_ready.add_argument("--browser-report", default="")
    release_ready.add_argument("--remote-report", default="")
    release_ready.add_argument("--json", action="store_true")
    runbook = subparsers.add_parser(
        "remote-runbook",
        help="Build a safe controlled two-machine remote demo runbook through the user CLI.",
    )
    runbook.add_argument("--coordinator-url", default="http://127.0.0.1:8787")
    runbook.add_argument("--miner-id", default="remote-linux-1")
    runbook.add_argument("--output-dir", default="dist/remote-demo")
    runbook.add_argument("--request-count", type=int, default=4)
    runbook.add_argument("--scenario-id", default="route-baseline")
    runbook.add_argument("--timeout-seconds", type=int, default=180)
    runbook.add_argument("--replace", action="store_true", help="replace an existing Miner entry in the generated registry")
    runbook.add_argument("--json", action="store_true")
    remote = subparsers.add_parser(
        "remote-acceptance",
        help="Validate a running controlled two-machine remote demo through the user CLI.",
    )
    remote.add_argument("--coordinator-url", required=True)
    remote.add_argument("--miner-id", required=True)
    remote.add_argument("--observer-token", required=True)
    remote.add_argument("--admin-token", required=True)
    remote.add_argument("--output-dir", default="dist/remote-demo-acceptance")
    remote.add_argument("--request-count", type=int, default=4)
    remote.add_argument("--scenario-id", default="route-baseline")
    remote.add_argument("--timeout-seconds", type=int, default=180)
    remote.add_argument("--remote-timeout-seconds", type=float, default=120.0)
    remote.add_argument("--poll-interval", type=float, default=2.0)
    remote.add_argument("--create-session", dest="create_session", action="store_true", default=True)
    remote.add_argument("--no-create-session", dest="create_session", action="store_false")
    remote.add_argument("--json", action="store_true")
    remote_demo = subparsers.add_parser(
        "remote-demo",
        help="Prepare or verify the high-level two-machine home-compute remote Miner demo.",
    )
    remote_demo_subparsers = remote_demo.add_subparsers(dest="remote_demo_action", required=True)
    remote_demo_prepare = remote_demo_subparsers.add_parser(
        "prepare",
        help="Create the recommended remote home-compute runbook and private env files.",
    )
    remote_demo_prepare.add_argument("--workload", choices=REMOTE_DEMO_WORKLOADS, default="model-bundle")
    remote_demo_prepare.add_argument("--coordinator-url", default="http://127.0.0.1:8787")
    remote_demo_prepare.add_argument("--target", choices=["generic", "kaggle"], default="generic")
    remote_demo_prepare.add_argument("--miner-id", default="remote-linux-1")
    remote_demo_prepare.add_argument("--output-dir", default="dist/remote-home-compute")
    remote_demo_prepare.add_argument("--request-count", type=int, default=4)
    remote_demo_prepare.add_argument("--scenario-id", default="route-baseline")
    remote_demo_prepare.add_argument("--decode-steps", type=int, default=4)
    remote_demo_prepare.add_argument("--stage-role", choices=["stage0", "stage1", "both"], default="both")
    remote_demo_prepare.add_argument("--micro-llm-artifact", default="")
    remote_demo_prepare.add_argument("--prompt-texts", default="arn,ten")
    remote_demo_prepare.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    remote_demo_prepare.add_argument("--hf-cache-dir", default="")
    remote_demo_prepare.add_argument("--timeout-seconds", type=int, default=180)
    remote_demo_prepare.add_argument("--replace", action="store_true")
    remote_demo_prepare.add_argument("--mock", action="store_true")
    remote_demo_prepare.add_argument("--llm-runtime-cmd", default="")
    remote_demo_prepare.add_argument("--llm-runtime-url", default="")
    remote_demo_prepare.add_argument("--llm-runtime-api-key", default="")
    remote_demo_prepare.add_argument("--llm-runtime-model-id", default="external-llm-runtime")
    remote_demo_prepare.add_argument("--llm-runtime-timeout", type=float, default=30.0)
    remote_demo_prepare.add_argument("--json", action="store_true")
    remote_demo_doctor = remote_demo_subparsers.add_parser(
        "doctor",
        help="Check remote-demo files, token presence, Coordinator reachability, and route readiness.",
    )
    remote_demo_doctor.add_argument("--workload", choices=REMOTE_DEMO_WORKLOADS, default="model-bundle")
    remote_demo_doctor.add_argument("--coordinator-url", default="http://127.0.0.1:8787")
    remote_demo_doctor.add_argument("--miner-id", default="remote-linux-1")
    remote_demo_doctor.add_argument("--observer-token", default="")
    remote_demo_doctor.add_argument("--admin-token", default="")
    remote_demo_doctor.add_argument("--output-dir", default="dist/remote-home-compute")
    remote_demo_doctor.add_argument("--request-count", type=int, default=4)
    remote_demo_doctor.add_argument("--scenario-id", default="route-baseline")
    remote_demo_doctor.add_argument("--decode-steps", type=int, default=4)
    remote_demo_doctor.add_argument("--stage-mode", choices=["both", "split"], default="both")
    remote_demo_doctor.add_argument("--require-distinct-stage-miners", action="store_true")
    remote_demo_doctor.add_argument("--micro-llm-artifact", default="")
    remote_demo_doctor.add_argument("--prompt-texts", default="arn,ten")
    remote_demo_doctor.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    remote_demo_doctor.add_argument("--hf-cache-dir", default="")
    remote_demo_doctor.add_argument("--timeout-seconds", type=int, default=180)
    remote_demo_doctor.add_argument("--http-timeout", type=float, default=5.0)
    remote_demo_doctor.add_argument("--admin-results-limit", type=int, default=10)
    remote_demo_doctor.add_argument("--require-result", action="store_true")
    remote_demo_doctor.add_argument("--json", action="store_true")
    remote_demo_verify = remote_demo_subparsers.add_parser(
        "verify",
        help="Create and verify a read-only remote home-compute session.",
    )
    remote_demo_verify.add_argument("--workload", choices=REMOTE_DEMO_WORKLOADS, default="model-bundle")
    remote_demo_verify.add_argument("--coordinator-url", required=True)
    remote_demo_verify.add_argument("--miner-id", required=True)
    remote_demo_verify.add_argument("--observer-token", required=True)
    remote_demo_verify.add_argument("--admin-token", required=True)
    remote_demo_verify.add_argument("--output-dir", default="dist/remote-home-compute")
    remote_demo_verify.add_argument("--request-count", type=int, default=4)
    remote_demo_verify.add_argument("--scenario-id", default="route-baseline")
    remote_demo_verify.add_argument("--decode-steps", type=int, default=4)
    remote_demo_verify.add_argument("--stage-mode", choices=["both", "split"], default="both")
    remote_demo_verify.add_argument("--require-distinct-stage-miners", action="store_true")
    remote_demo_verify.add_argument("--micro-llm-artifact", default="")
    remote_demo_verify.add_argument("--prompt-texts", default="arn,ten")
    remote_demo_verify.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    remote_demo_verify.add_argument("--hf-cache-dir", default="")
    remote_demo_verify.add_argument("--timeout-seconds", type=int, default=180)
    remote_demo_verify.add_argument("--remote-timeout-seconds", type=float, default=120.0)
    remote_demo_verify.add_argument("--poll-interval", type=float, default=2.0)
    remote_demo_verify.add_argument("--http-timeout", type=float, default=5.0)
    remote_demo_verify.add_argument("--artifact-timeout", type=float, default=60.0)
    remote_demo_verify.add_argument("--admin-results-limit", type=int, default=10)
    remote_demo_verify.add_argument("--create-session", dest="create_session", action="store_true", default=True)
    remote_demo_verify.add_argument("--no-create-session", dest="create_session", action="store_false")
    remote_demo_verify.add_argument("--mock", action="store_true")
    remote_demo_verify.add_argument("--llm-runtime-cmd", default="")
    remote_demo_verify.add_argument("--llm-runtime-url", default="")
    remote_demo_verify.add_argument("--llm-runtime-api-key", default="")
    remote_demo_verify.add_argument("--llm-runtime-model-id", default="external-llm-runtime")
    remote_demo_verify.add_argument("--llm-runtime-timeout", type=float, default=30.0)
    remote_demo_verify.add_argument("--json", action="store_true")
    remote_demo_collect = remote_demo_subparsers.add_parser(
        "collect",
        help="Collect evidence and Support Bundle from an already running remote-demo.",
    )
    remote_demo_collect.add_argument("--workload", choices=REMOTE_DEMO_WORKLOADS, default="model-bundle")
    remote_demo_collect.add_argument("--coordinator-url", required=True)
    remote_demo_collect.add_argument("--miner-id", required=True)
    remote_demo_collect.add_argument("--observer-token", required=True)
    remote_demo_collect.add_argument("--admin-token", required=True)
    remote_demo_collect.add_argument("--output-dir", default="dist/remote-home-compute")
    remote_demo_collect.add_argument("--request-count", type=int, default=4)
    remote_demo_collect.add_argument("--scenario-id", default="route-baseline")
    remote_demo_collect.add_argument("--decode-steps", type=int, default=4)
    remote_demo_collect.add_argument("--stage-mode", choices=["both", "split"], default="both")
    remote_demo_collect.add_argument("--require-distinct-stage-miners", action="store_true")
    remote_demo_collect.add_argument("--micro-llm-artifact", default="")
    remote_demo_collect.add_argument("--prompt-texts", default="arn,ten")
    remote_demo_collect.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    remote_demo_collect.add_argument("--hf-cache-dir", default="")
    remote_demo_collect.add_argument("--task-id", default="")
    remote_demo_collect.add_argument("--timeout-seconds", type=int, default=180)
    remote_demo_collect.add_argument("--http-timeout", type=float, default=5.0)
    remote_demo_collect.add_argument("--artifact-timeout", type=float, default=60.0)
    remote_demo_collect.add_argument("--admin-results-limit", type=int, default=10)
    remote_demo_collect.add_argument("--mock", action="store_true")
    remote_demo_collect.add_argument("--llm-runtime-cmd", default="")
    remote_demo_collect.add_argument("--llm-runtime-url", default="")
    remote_demo_collect.add_argument("--llm-runtime-api-key", default="")
    remote_demo_collect.add_argument("--llm-runtime-model-id", default="external-llm-runtime")
    remote_demo_collect.add_argument("--llm-runtime-timeout", type=float, default=30.0)
    remote_demo_collect.add_argument("--json", action="store_true")
    remote_demo_clean = remote_demo_subparsers.add_parser(
        "clean",
        help="Dry-run or delete known files generated by remote-demo.",
    )
    remote_demo_clean.add_argument("--output-dir", default="dist/remote-home-compute")
    remote_demo_clean.add_argument("--timeout-seconds", type=int, default=60)
    remote_demo_clean.add_argument("--apply", action="store_true")
    remote_demo_clean.add_argument("--include-private", action="store_true")
    remote_demo_clean.add_argument("--remove-empty-dir", action="store_true")
    remote_demo_clean.add_argument("--json", action="store_true")
    remote_demo_kaggle_real = remote_demo_subparsers.add_parser(
        "kaggle-real",
        help="Prepare, verify, or collect the real Kaggle CPU Miner runtime acceptance.",
    )
    remote_demo_kaggle_real.add_argument("--action", dest="kaggle_real_action", choices=["prepare", "verify", "collect"], default="prepare")
    remote_demo_kaggle_real.add_argument("--public-host", default="24.199.118.54")
    remote_demo_kaggle_real.add_argument("--port", type=int, default=9180)
    remote_demo_kaggle_real.add_argument("--coordinator-url", default="")
    remote_demo_kaggle_real.add_argument("--miner-id", default="kaggle-cpu-1")
    remote_demo_kaggle_real.add_argument("--workload", choices=["model-bundle", "micro-llm-sharded"], default="model-bundle")
    remote_demo_kaggle_real.add_argument("--output-dir", default="dist/kaggle-real-runtime")
    remote_demo_kaggle_real.add_argument("--request-count", type=int, default=2)
    remote_demo_kaggle_real.add_argument("--scenario-id", default="route-baseline")
    remote_demo_kaggle_real.add_argument("--decode-steps", type=int, default=4)
    remote_demo_kaggle_real.add_argument("--stage-mode", choices=["both", "split"], default="both")
    remote_demo_kaggle_real.add_argument("--require-distinct-stage-miners", action="store_true")
    remote_demo_kaggle_real.add_argument("--micro-llm-artifact", default="")
    remote_demo_kaggle_real.add_argument("--prompt-texts", default="arn,ten")
    remote_demo_kaggle_real.add_argument("--timeout-seconds", type=int, default=240)
    remote_demo_kaggle_real.add_argument("--bind-host", default="0.0.0.0")
    remote_demo_kaggle_real.add_argument("--backlog", type=int, default=1)
    remote_demo_kaggle_real.add_argument("--lease-seconds", type=float, default=15.0)
    remote_demo_kaggle_real.add_argument("--miner-token", default="")
    remote_demo_kaggle_real.add_argument("--observer-token", default="")
    remote_demo_kaggle_real.add_argument("--admin-token", default="")
    remote_demo_kaggle_real.add_argument("--replace", action="store_true")
    remote_demo_kaggle_real.add_argument("--remote-timeout-seconds", type=float, default=180.0)
    remote_demo_kaggle_real.add_argument("--poll-interval", type=float, default=2.0)
    remote_demo_kaggle_real.add_argument("--http-timeout", type=float, default=5.0)
    remote_demo_kaggle_real.add_argument("--artifact-timeout", type=float, default=60.0)
    remote_demo_kaggle_real.add_argument("--admin-results-limit", type=int, default=10)
    remote_demo_kaggle_real.add_argument("--require-existing-result", action="store_true")
    remote_demo_kaggle_real.add_argument("--collect-on-failure", action="store_true")
    remote_demo_kaggle_real.add_argument("--task-id", default="")
    remote_demo_kaggle_real.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.command in {"local-proof", "serve", "join", "generate", "home-infer", "llm-infer", "cpu-infer", "shard-infer", "micro-llm-shard-infer", "real-llm-shard-infer", "micro-llm-artifact", "shard-infer-beta", "micro-llm-shard-infer-beta", "real-llm-shard-infer-beta", "micro-llm-live-rc", "real-llm-live-rc", "real-llm-internet-alpha", "real-llm-internet-beta", "swarm-session", "public-swarm-alpha-rc", "public-swarm-beta", "public-swarm-beta-rc", "public-swarm-product-beta", "public-swarm-gpu-beta", "gpu-generate", "release-ready", "remote-runbook", "remote-acceptance"} or (
        args.command == "remote-demo" and hasattr(args, "request_count")
    ):
        if hasattr(args, "request_count") and args.request_count < 1:
            raise SystemExit("--request-count must be at least 1")
        if hasattr(args, "timeout_seconds") and args.timeout_seconds < 1:
            raise SystemExit("--timeout-seconds must be at least 1")
    if args.command == "serve":
        if args.port < 1:
            raise SystemExit("--port must be positive")
    if args.command == "join":
        if not args.coordinator_url and not args.peer_bootstrap:
            raise SystemExit("join requires --coordinator-url or --peer-bootstrap")
        if args.http_timeout <= 0:
            raise SystemExit("--http-timeout must be positive")
        if args.max_tasks < 0:
            raise SystemExit("--max-tasks must be non-negative")
    if args.command == "generate":
        if args.max_new_tokens < 1 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 1 and 32")
        if len(args.prompt_text) > 256:
            raise SystemExit("--prompt-text must be at most 256 characters")
        if not args.coordinator_url and not args.peer_bootstrap:
            raise SystemExit("generate requires --coordinator-url or --peer-bootstrap")
        if args.poll_interval <= 0:
            raise SystemExit("--poll-interval must be positive")
        if args.http_timeout <= 0:
            raise SystemExit("--http-timeout must be positive")
        if args.admin_results_limit < 1:
            raise SystemExit("--admin-results-limit must be at least 1")
    if args.command == "peer":
        if args.peer_action == "daemon":
            if args.port < 1:
                raise SystemExit("--port must be positive")
            if args.ttl_seconds <= 0:
                raise SystemExit("--ttl-seconds must be positive")
        if args.peer_action in {"resolve", "announce"} and args.http_timeout <= 0:
            raise SystemExit("--http-timeout must be positive")
        if args.peer_action == "resolve" and (args.max_new_tokens < 1 or args.max_new_tokens > 32):
            raise SystemExit("--max-new-tokens must be between 1 and 32")
    if args.command == "public-swarm-product-rc":
        if args.max_new_tokens < 2 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 2 and 32")
        if args.timeout_seconds < 1:
            raise SystemExit("--timeout-seconds must be positive")
    if args.command == "release-ready":
        if args.base_port < 1:
            raise SystemExit("--base-port must be positive")
        if args.external_llm_request_count < 1:
            raise SystemExit("--external-llm-request-count must be at least 1")
    if args.command == "real-llm-internet-beta":
        if args.request_count > 4:
            raise SystemExit("--request-count must be between 1 and 4")
        if args.max_new_tokens < 1 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 1 and 32")
        if args.failure_mode != "none" and args.max_new_tokens != 1:
            raise SystemExit("--max-new-tokens greater than 1 is only supported with --failure-mode none")
        if args.port < 1 or args.base_port < 1:
            raise SystemExit("--port and --base-port must be positive")
        if args.max_request_attempts < 1:
            raise SystemExit("--max-request-attempts must be at least 1")
        for name in [
            "remote_timeout_seconds",
            "startup_timeout",
            "process_exit_timeout",
            "poll_interval",
            "http_timeout",
            "kaggle_push_timeout_seconds",
            "kaggle_delete_timeout_seconds",
            "kaggle_status_timeout_seconds",
            "kaggle_status_poll_interval",
            "lease_seconds",
            "victim_compute_seconds",
            "heartbeat_interval",
            "idle_sleep",
            "claim_observe_timeout",
            "requeue_timeout",
        ]:
            if getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if args.failure_mode != "none":
            if args.victim_compute_seconds <= args.lease_seconds:
                args.victim_compute_seconds = args.lease_seconds + 30.0
            if args.requeue_timeout <= args.lease_seconds:
                args.requeue_timeout = args.lease_seconds + 45.0
    if args.command == "swarm-infer-beta":
        if getattr(args, "timeout_seconds", 1) <= 0:
            raise SystemExit("--timeout-seconds must be positive")
        if args.swarm_action != "clean":
            if args.port < 1:
                raise SystemExit("--port must be positive")
            if args.request_count < 1 or args.request_count > 4:
                raise SystemExit("--request-count must be between 1 and 4")
            for name in ["remote_timeout_seconds", "http_timeout"]:
                if getattr(args, name) <= 0:
                    raise SystemExit(f"--{name.replace('_', '-')} must be positive")
            if hasattr(args, "lease_seconds") and args.lease_seconds <= 0:
                raise SystemExit("--lease-seconds must be positive")
            if hasattr(args, "compute_seconds") and args.compute_seconds < 0:
                raise SystemExit("--compute-seconds must be non-negative")
            if hasattr(args, "heartbeat_interval") and args.heartbeat_interval <= 0:
                raise SystemExit("--heartbeat-interval must be positive")
            if hasattr(args, "victim_compute_seconds") and args.victim_compute_seconds <= 0:
                raise SystemExit("--victim-compute-seconds must be positive")
            if hasattr(args, "claim_observe_timeout") and args.claim_observe_timeout <= 0:
                raise SystemExit("--claim-observe-timeout must be positive")
            if hasattr(args, "requeue_timeout") and args.requeue_timeout <= 0:
                raise SystemExit("--requeue-timeout must be positive")
            if hasattr(args, "max_request_attempts") and args.max_request_attempts < 1:
                raise SystemExit("--max-request-attempts must be at least 1")
            if hasattr(args, "artifact_timeout") and args.artifact_timeout <= 0:
                raise SystemExit("--artifact-timeout must be positive")
            if getattr(args, "swarm_action", "") == "live" and getattr(args, "failure_mode", "none") != "none":
                if args.victim_compute_seconds <= args.lease_seconds:
                    args.victim_compute_seconds = args.lease_seconds + 30.0
                if args.requeue_timeout <= args.lease_seconds:
                    args.requeue_timeout = args.lease_seconds + 45.0
    if args.command == "swarm-session":
        if args.port < 1 or args.base_port < 1:
            raise SystemExit("--port and --base-port must be positive")
        if args.request_count < 1 or args.request_count > 4:
            raise SystemExit("--request-count must be between 1 and 4")
        for name in [
            "remote_timeout_seconds",
            "startup_timeout",
            "process_exit_timeout",
            "poll_interval",
            "http_timeout",
            "kaggle_push_timeout_seconds",
            "kaggle_delete_timeout_seconds",
            "kaggle_status_timeout_seconds",
            "kaggle_status_poll_interval",
            "lease_seconds",
            "victim_compute_seconds",
            "heartbeat_interval",
            "idle_sleep",
            "claim_observe_timeout",
            "requeue_timeout",
        ]:
            if getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if args.compute_seconds < 0:
            raise SystemExit("--compute-seconds must be non-negative")
        if args.max_request_attempts < 1:
            raise SystemExit("--max-request-attempts must be at least 1")
        if args.failure_mode != "none":
            if args.victim_compute_seconds <= args.lease_seconds:
                args.victim_compute_seconds = args.lease_seconds + 30.0
            if args.requeue_timeout <= args.lease_seconds:
                args.requeue_timeout = args.lease_seconds + 45.0
    if args.command == "public-swarm-alpha-rc":
        if args.request_count > 4:
            raise SystemExit("--request-count must be between 1 and 4")
    if args.command == "public-swarm-beta":
        if args.request_count > 4:
            raise SystemExit("--request-count must be between 1 and 4")
        if getattr(args, "base_port", 1) < 1:
            raise SystemExit("--base-port must be positive")
        if getattr(args, "port", 1) < 1:
            raise SystemExit("--port must be positive")
        for name in ["remote_timeout_seconds", "http_timeout", "artifact_timeout", "lease_seconds", "heartbeat_interval"]:
            if hasattr(args, name) and getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if hasattr(args, "compute_seconds") and args.compute_seconds < 0:
            raise SystemExit("--compute-seconds must be non-negative")
        if hasattr(args, "max_request_attempts") and args.max_request_attempts < 1:
            raise SystemExit("--max-request-attempts must be at least 1")
        if hasattr(args, "max_new_tokens") and (args.max_new_tokens < 2 or args.max_new_tokens > 32):
            raise SystemExit("--max-new-tokens must be between 2 and 32")
        if hasattr(args, "cpu_request_count") and (args.cpu_request_count < 1 or args.cpu_request_count > 4):
            raise SystemExit("--cpu-request-count must be between 1 and 4")
        if hasattr(args, "external_llm_request_count") and (args.external_llm_request_count < 1 or args.external_llm_request_count > 4):
            raise SystemExit("--external-llm-request-count must be between 1 and 4")
        if hasattr(args, "cpu_timeout_seconds") and args.cpu_timeout_seconds <= 0:
            raise SystemExit("--cpu-timeout-seconds must be positive")
    if args.command == "public-swarm-beta-rc":
        if args.request_count < 1 or args.request_count > 4:
            raise SystemExit("--request-count must be between 1 and 4")
        if args.cpu_request_count < 1 or args.cpu_request_count > 4:
            raise SystemExit("--cpu-request-count must be between 1 and 4")
        if args.external_llm_request_count < 1 or args.external_llm_request_count > 4:
            raise SystemExit("--external-llm-request-count must be between 1 and 4")
        if args.max_new_tokens < 2 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 2 and 32")
        if args.base_port < 1 or args.port < 1:
            raise SystemExit("--base-port and --port must be positive")
        for name in [
            "timeout_seconds",
            "remote_timeout_seconds",
            "cpu_timeout_seconds",
            "startup_timeout",
            "process_exit_timeout",
            "poll_interval",
            "http_timeout",
        ]:
            if getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if args.public_swarm_beta_rc_mode == "external-existing":
            missing = [
                name for name in ["coordinator_url", "observer_token", "admin_token"]
                if not getattr(args, name)
            ]
            if missing:
                raise SystemExit(f"external-existing requires: {', '.join('--' + item.replace('_', '-') for item in missing)}")
    if args.command == "public-swarm-product-beta":
        if args.request_count < 1 or args.request_count > 4:
            raise SystemExit("--request-count must be between 1 and 4")
        if args.cpu_request_count < 1 or args.cpu_request_count > 4:
            raise SystemExit("--cpu-request-count must be between 1 and 4")
        if args.external_llm_request_count < 1 or args.external_llm_request_count > 4:
            raise SystemExit("--external-llm-request-count must be between 1 and 4")
        if args.max_new_tokens < 2 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 2 and 32")
        if args.base_port < 1 or args.port < 1:
            raise SystemExit("--base-port and --port must be positive")
        for name in [
            "timeout_seconds",
            "remote_timeout_seconds",
            "cpu_timeout_seconds",
            "startup_timeout",
            "process_exit_timeout",
            "poll_interval",
            "http_timeout",
        ]:
            if getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if args.public_swarm_product_beta_mode == "external-existing":
            missing = [
                name for name in ["coordinator_url", "observer_token", "admin_token"]
                if not getattr(args, name)
            ]
            if missing:
                raise SystemExit(f"external-existing requires: {', '.join('--' + item.replace('_', '-') for item in missing)}")
    if args.command == "public-swarm-gpu-beta":
        if args.request_count > 4:
            raise SystemExit("--request-count must be between 1 and 4")
        if getattr(args, "base_port", 1) < 1:
            raise SystemExit("--base-port must be positive")
        if getattr(args, "port", 1) < 1:
            raise SystemExit("--port must be positive")
        for name in ["remote_timeout_seconds", "http_timeout", "artifact_timeout", "lease_seconds", "heartbeat_interval"]:
            if hasattr(args, name) and getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if hasattr(args, "compute_seconds") and args.compute_seconds < 0:
            raise SystemExit("--compute-seconds must be non-negative")
        if hasattr(args, "max_request_attempts") and args.max_request_attempts < 1:
            raise SystemExit("--max-request-attempts must be at least 1")
    if args.command == "gpu-generate":
        if args.request_count != 1:
            raise SystemExit("--request-count must be 1 for gpu-generate Beta")
        if args.max_new_tokens < 2 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 2 and 32")
        if getattr(args, "base_port", 1) < 1:
            raise SystemExit("--base-port must be positive")
        if getattr(args, "port", 1) < 1:
            raise SystemExit("--port must be positive")
        for name in [
            "remote_timeout_seconds",
            "http_timeout",
            "lease_seconds",
            "heartbeat_interval",
            "idle_sleep",
        ]:
            if hasattr(args, name) and getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if hasattr(args, "compute_seconds") and args.compute_seconds < 0:
            raise SystemExit("--compute-seconds must be non-negative")
        if hasattr(args, "max_request_attempts") and args.max_request_attempts < 1:
            raise SystemExit("--max-request-attempts must be at least 1")
    if args.command == "llm-infer":
        if args.llm_runtime_cmd and args.llm_runtime_url:
            raise SystemExit("--llm-runtime-cmd and --llm-runtime-url are mutually exclusive")
        if args.llm_runtime_timeout <= 0:
            raise SystemExit("--llm-runtime-timeout must be positive")
    if args.command == "cpu-infer":
        if args.base_port < 1:
            raise SystemExit("--base-port must be positive")
        if args.external_llm_request_count < 1:
            raise SystemExit("--external-llm-request-count must be at least 1")
        if args.remote_timeout_seconds < 0:
            raise SystemExit("--remote-timeout-seconds must be non-negative")
        if args.poll_interval <= 0:
            raise SystemExit("--poll-interval must be positive")
        if args.llm_runtime_cmd and args.llm_runtime_url:
            raise SystemExit("--llm-runtime-cmd and --llm-runtime-url are mutually exclusive")
        if args.llm_runtime_timeout <= 0:
            raise SystemExit("--llm-runtime-timeout must be positive")
        if args.mode == "remote-existing":
            missing = [
                name for name in ["coordinator_url", "observer_token", "admin_token"]
                if not getattr(args, name)
            ]
            if missing:
                raise SystemExit(f"remote-existing requires: {', '.join('--' + item.replace('_', '-') for item in missing)}")
    if args.command == "shard-infer":
        if args.port < 1:
            raise SystemExit("--port must be positive")
    if args.command == "micro-llm-shard-infer":
        if args.port < 1:
            raise SystemExit("--port must be positive")
        if args.decode_steps < 1 or args.decode_steps > 4:
            raise SystemExit("--decode-steps must be between 1 and 4")
    if args.command == "real-llm-shard-infer":
        if args.port < 1:
            raise SystemExit("--port must be positive")
        if args.request_count > 4:
            raise SystemExit("--request-count must be at most 4")
        if args.max_new_tokens < 1 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 1 and 32")
        if args.failure_mode != "none" and args.max_new_tokens != 1:
            raise SystemExit("--max-new-tokens greater than 1 is only supported with --failure-mode none")
    if args.command == "micro-llm-artifact":
        if args.version < 1:
            raise SystemExit("--version must be at least 1")
    if args.command == "shard-infer-beta":
        if args.base_port < 1:
            raise SystemExit("--base-port must be positive")
        if args.remote_timeout_seconds <= 0:
            raise SystemExit("--remote-timeout-seconds must be positive")
        if args.mode == "remote-existing":
            missing = [
                name for name in ["coordinator_url", "observer_token", "admin_token"]
                if not getattr(args, name)
            ]
            if missing:
                raise SystemExit(f"remote-existing requires: {', '.join('--' + item.replace('_', '-') for item in missing)}")
            if args.failure_mode != "none":
                raise SystemExit("remote-existing does not orchestrate failure-mode kills; use remote-loopback for requeue proof")
    if args.command == "micro-llm-shard-infer-beta":
        if args.base_port < 1:
            raise SystemExit("--base-port must be positive")
        if args.decode_steps < 1 or args.decode_steps > 4:
            raise SystemExit("--decode-steps must be between 1 and 4")
        if args.remote_timeout_seconds <= 0:
            raise SystemExit("--remote-timeout-seconds must be positive")
        if args.mode == "remote-existing":
            missing = [
                name for name in ["coordinator_url", "observer_token", "admin_token"]
                if not getattr(args, name)
            ]
            if missing:
                raise SystemExit(f"remote-existing requires: {', '.join('--' + item.replace('_', '-') for item in missing)}")
            if args.failure_mode != "none":
                raise SystemExit("remote-existing does not orchestrate failure-mode kills; use remote-loopback for requeue proof")
    if args.command == "real-llm-shard-infer-beta":
        if args.base_port < 1:
            raise SystemExit("--base-port must be positive")
        if args.request_count > 4:
            raise SystemExit("--request-count must be at most 4")
        if args.max_new_tokens < 1 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 1 and 32")
        if args.failure_mode != "none" and args.max_new_tokens != 1:
            raise SystemExit("--max-new-tokens greater than 1 is only supported with --failure-mode none")
        if args.remote_timeout_seconds <= 0:
            raise SystemExit("--remote-timeout-seconds must be positive")
        if args.mode == "remote-existing":
            missing = [
                name for name in ["coordinator_url", "observer_token", "admin_token"]
                if not getattr(args, name)
            ]
            if missing:
                raise SystemExit(f"remote-existing requires: {', '.join('--' + item.replace('_', '-') for item in missing)}")
            if args.failure_mode != "none":
                raise SystemExit("remote-existing does not orchestrate failure-mode kills; use remote-loopback for requeue proof")
    if args.command == "micro-llm-live-rc":
        if args.port < 1:
            raise SystemExit("--port must be positive")
        if args.decode_steps < 1 or args.decode_steps > 4:
            raise SystemExit("--decode-steps must be between 1 and 4")
        for name in [
            "remote_timeout_seconds",
            "startup_timeout",
            "process_exit_timeout",
            "poll_interval",
            "http_timeout",
            "artifact_timeout",
            "lease_seconds",
            "heartbeat_interval",
            "idle_sleep",
        ]:
            if getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if args.compute_seconds < 0:
            raise SystemExit("--compute-seconds must be non-negative")
        if args.max_request_attempts < 1:
            raise SystemExit("--max-request-attempts must be at least 1")
        if args.admin_results_limit < 1:
            raise SystemExit("--admin-results-limit must be at least 1")
    if args.command == "real-llm-live-rc":
        if args.port < 1:
            raise SystemExit("--port must be positive")
        if args.request_count > 4:
            raise SystemExit("--request-count must be at most 4")
        if args.max_new_tokens < 1 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 1 and 32")
        for name in [
            "remote_timeout_seconds",
            "startup_timeout",
            "process_exit_timeout",
            "poll_interval",
            "http_timeout",
            "lease_seconds",
            "heartbeat_interval",
            "idle_sleep",
        ]:
            if getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if args.compute_seconds < 0:
            raise SystemExit("--compute-seconds must be non-negative")
        if args.max_request_attempts < 1:
            raise SystemExit("--max-request-attempts must be at least 1")
        if args.mode == "external-existing":
            missing = [
                name for name in ["coordinator_url", "observer_token", "admin_token"]
                if not getattr(args, name)
            ]
            if missing:
                raise SystemExit(f"external-existing requires: {', '.join('--' + item.replace('_', '-') for item in missing)}")
    if args.command == "real-llm-internet-alpha":
        if args.port < 1:
            raise SystemExit("--port must be positive")
        if args.base_port < 1:
            raise SystemExit("--base-port must be positive")
        if args.request_count > 4:
            raise SystemExit("--request-count must be at most 4")
        if args.max_new_tokens < 1 or args.max_new_tokens > 32:
            raise SystemExit("--max-new-tokens must be between 1 and 32")
        for name in [
            "remote_timeout_seconds",
            "startup_timeout",
            "process_exit_timeout",
            "poll_interval",
            "http_timeout",
            "lease_seconds",
            "heartbeat_interval",
            "idle_sleep",
        ]:
            if getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if args.compute_seconds < 0:
            raise SystemExit("--compute-seconds must be non-negative")
        if args.max_request_attempts < 1:
            raise SystemExit("--max-request-attempts must be at least 1")
        if args.mode == "external-existing":
            missing = [
                name for name in ["coordinator_url", "observer_token", "admin_token"]
                if not getattr(args, name)
            ]
            if missing:
                raise SystemExit(f"external-existing requires: {', '.join('--' + item.replace('_', '-') for item in missing)}")
    if args.command == "remote-acceptance":
        if args.remote_timeout_seconds < 0:
            raise SystemExit("--remote-timeout-seconds must be non-negative")
        if args.poll_interval <= 0:
            raise SystemExit("--poll-interval must be positive")
    if args.command == "remote-demo" and args.remote_demo_action in {"doctor", "verify", "collect"}:
        if args.http_timeout <= 0:
            raise SystemExit("--http-timeout must be positive")
        if args.admin_results_limit < 1:
            raise SystemExit("--admin-results-limit must be at least 1")
    if args.command == "remote-demo" and hasattr(args, "decode_steps"):
        if args.decode_steps < 1 or args.decode_steps > 4:
            raise SystemExit("--decode-steps must be between 1 and 4")
    if args.command == "remote-demo" and getattr(args, "workload", "") == "real-llm-sharded":
        if getattr(args, "prompt_texts", "") == "arn,ten":
            args.prompt_texts = "CrowdTensor routes home CPU,A miner returns one token"
        if args.remote_demo_action in {"doctor", "verify", "collect"} and getattr(args, "stage_mode", "both") == "both":
            args.stage_mode = "split"
        if args.remote_demo_action in {"doctor", "verify", "collect"} and getattr(args, "stage_mode", "") == "split":
            args.require_distinct_stage_miners = True
    if args.command == "remote-demo" and args.remote_demo_action == "kaggle-real":
        if args.port < 1:
            raise SystemExit("--port must be positive")
        if args.backlog < 1:
            raise SystemExit("--backlog must be at least 1")
        if args.decode_steps < 1 or args.decode_steps > 4:
            raise SystemExit("--decode-steps must be between 1 and 4")
        if args.workload == "micro-llm-sharded" and args.stage_mode == "split":
            args.require_distinct_stage_miners = True
        if args.lease_seconds <= 0:
            raise SystemExit("--lease-seconds must be positive")
        if args.remote_timeout_seconds < 0:
            raise SystemExit("--remote-timeout-seconds must be non-negative")
        if args.poll_interval <= 0:
            raise SystemExit("--poll-interval must be positive")
        if args.http_timeout <= 0:
            raise SystemExit("--http-timeout must be positive")
        if args.artifact_timeout <= 0:
            raise SystemExit("--artifact-timeout must be positive")
        if args.admin_results_limit < 1:
            raise SystemExit("--admin-results-limit must be at least 1")
    if args.command == "remote-demo" and args.remote_demo_action == "verify":
        if args.remote_timeout_seconds < 0:
            raise SystemExit("--remote-timeout-seconds must be non-negative")
        if args.poll_interval <= 0:
            raise SystemExit("--poll-interval must be positive")
        if args.artifact_timeout <= 0:
            raise SystemExit("--artifact-timeout must be positive")
    if args.command == "remote-demo" and args.remote_demo_action == "collect":
        if args.artifact_timeout <= 0:
            raise SystemExit("--artifact-timeout must be positive")
    if args.command == "remote-demo" and args.remote_demo_action in {"prepare", "verify", "collect"}:
        if args.llm_runtime_cmd and args.llm_runtime_url:
            raise SystemExit("--llm-runtime-cmd and --llm-runtime-url are mutually exclusive")
        if args.llm_runtime_timeout <= 0:
            raise SystemExit("--llm-runtime-timeout must be positive")
    if args.command == "clean-artifacts":
        if args.older_than_hours < 0:
            raise SystemExit("--older-than-hours must be non-negative")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "local-proof":
        summary = build_local_proof(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_local_proof(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "clean-artifacts":
        report = build_cleanup_report(args)
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            print_cleanup_report(report)
        raise SystemExit(0 if report.get("ok") else 1)
    if args.command == "serve":
        report = build_product_serve(args)
        print(json.dumps(report, sort_keys=True) if args.json else "\n".join(report.get("command") or []))
        raise SystemExit(0 if report.get("ok") else 1)
    if args.command == "join":
        report = build_product_join(args)
        print(json.dumps(report, sort_keys=True) if args.json else "\n".join(report.get("command") or []))
        raise SystemExit(0 if report.get("ok") else 1)
    if args.command == "generate":
        report = build_product_generate(args)
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            print(f"CrowdTensor generate ok={report.get('ok')} diagnosis={','.join(report.get('diagnosis_codes') or [])}")
        raise SystemExit(0 if report.get("ok") else 1)
    if args.command == "peer":
        report = build_peer_cli(args)
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            if report.get("command"):
                print("\n".join(report.get("command") or []))
            else:
                print(f"CrowdTensor peer ok={report.get('ok')} diagnosis={','.join(report.get('diagnosis_codes') or [])}")
        raise SystemExit(0 if report.get("ok") else 1)
    if args.command == "public-swarm-product-rc":
        report = build_public_swarm_product_rc(args)
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            print(f"CrowdTensor Public Swarm Product RC ok={report.get('ok')} diagnosis={','.join(report.get('diagnosis_codes') or [])}")
        raise SystemExit(0 if report.get("ok") else 1)
    if args.command == "home-infer":
        summary = build_home_inference(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_home_inference(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "llm-infer":
        summary = build_llm_inference(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_llm_inference(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "cpu-infer":
        summary = build_cpu_inference_beta(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_cpu_inference_beta(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "shard-infer":
        summary = build_sharded_inference(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_sharded_inference(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "micro-llm-shard-infer":
        summary = build_micro_llm_sharded_inference(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_micro_llm_sharded_inference(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "real-llm-shard-infer":
        summary = build_real_llm_sharded_inference(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_real_llm_sharded_inference(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "micro-llm-artifact":
        summary = build_micro_llm_artifact(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_micro_llm_artifact(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "shard-infer-beta":
        summary = build_remote_sharded_inference_beta(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_remote_sharded_inference_beta(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "micro-llm-shard-infer-beta":
        summary = build_remote_micro_llm_sharded_beta(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_remote_micro_llm_sharded_beta(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "real-llm-shard-infer-beta":
        summary = build_remote_real_llm_sharded_beta(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_remote_real_llm_sharded_beta(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "micro-llm-live-rc":
        summary = build_micro_llm_live_rc(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_micro_llm_live_rc(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "real-llm-live-rc":
        summary = build_real_llm_live_rc(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_real_llm_live_rc(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "real-llm-internet-alpha":
        summary = build_real_llm_internet_alpha(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_real_llm_internet_alpha(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "real-llm-internet-beta":
        summary = build_real_llm_internet_beta(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_real_llm_internet_beta(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "swarm-infer-beta":
        summary = build_swarm_inference_beta(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_swarm_inference_beta(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "swarm-session":
        summary = build_public_swarm_inference_alpha(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_public_swarm_inference_alpha(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "public-swarm-alpha-rc":
        summary = build_public_swarm_inference_alpha_rc(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_public_swarm_inference_alpha_rc(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "public-swarm-beta":
        summary = build_public_swarm_inference_beta(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_public_swarm_inference_beta(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "public-swarm-beta-rc":
        summary = build_public_swarm_inference_beta_rc(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_public_swarm_inference_beta_rc(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "public-swarm-product-beta":
        summary = build_public_swarm_product_beta(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_public_swarm_product_beta(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "public-swarm-gpu-beta":
        summary = build_public_swarm_gpu_inference_beta(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_public_swarm_gpu_inference_beta(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "gpu-generate":
        summary = build_gpu_sharded_generation_beta(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_gpu_sharded_generation_beta(summary)
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "release-ready":
        report = build_release_ready(args)
        if args.json:
            print(json.dumps(report, sort_keys=True))
        else:
            print_release_ready(report)
        raise SystemExit(0 if report.get("ok") else 1)
    if args.command == "remote-runbook":
        summary = build_remote_runbook(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_remote_cli_report(summary, title="CrowdTensor remote runbook")
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "remote-acceptance":
        summary = build_remote_acceptance(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_remote_cli_report(summary, title="CrowdTensor remote acceptance")
        raise SystemExit(0 if summary.get("ok") else 1)
    if args.command == "remote-demo":
        summary = build_remote_demo(args)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print_remote_cli_report(summary, title="CrowdTensor remote home-compute demo")
        raise SystemExit(0 if summary.get("ok") else 1)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
