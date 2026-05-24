"""User-facing CrowdTensor command line entrypoints."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
for path in [ROOT, SCRIPTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    import support_bundle  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover - fallback for unusual packaging layouts
    support_bundle = None


SUMMARY_SCHEMA = "local_proof_summary_v1"
CLEANUP_SCHEMA = "cleanup_report_v1"
REMOTE_RUNBOOK_CLI_SCHEMA = "remote_runbook_cli_v1"
REMOTE_ACCEPTANCE_CLI_SCHEMA = "remote_acceptance_cli_v1"
REMOTE_HOME_DEMO_SCHEMA = "remote_home_compute_demo_v1"
HOME_INFERENCE_CLI_SCHEMA = "home_inference_cli_v1"
LLM_INFERENCE_CLI_SCHEMA = "llm_inference_cli_v1"
SECRET_FRAGMENTS = (
    "CROWDTENSOR_MINER_TOKEN",
    "CROWDTENSOR_OBSERVER_TOKEN",
    "CROWDTENSOR_ADMIN_TOKEN",
    "lease_token",
    "idempotency_key",
    "inference_results",
    "external_llm_results",
    "output_text",
    "Bearer ",
)
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
    remote_demo_prepare.add_argument("--workload", choices=["model-bundle", "external-llm"], default="model-bundle")
    remote_demo_prepare.add_argument("--coordinator-url", default="http://127.0.0.1:8787")
    remote_demo_prepare.add_argument("--miner-id", default="remote-linux-1")
    remote_demo_prepare.add_argument("--output-dir", default="dist/remote-home-compute")
    remote_demo_prepare.add_argument("--request-count", type=int, default=4)
    remote_demo_prepare.add_argument("--scenario-id", default="route-baseline")
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
    remote_demo_doctor.add_argument("--workload", choices=["model-bundle", "external-llm"], default="model-bundle")
    remote_demo_doctor.add_argument("--coordinator-url", default="http://127.0.0.1:8787")
    remote_demo_doctor.add_argument("--miner-id", default="remote-linux-1")
    remote_demo_doctor.add_argument("--observer-token", default="")
    remote_demo_doctor.add_argument("--admin-token", default="")
    remote_demo_doctor.add_argument("--output-dir", default="dist/remote-home-compute")
    remote_demo_doctor.add_argument("--request-count", type=int, default=4)
    remote_demo_doctor.add_argument("--scenario-id", default="route-baseline")
    remote_demo_doctor.add_argument("--timeout-seconds", type=int, default=180)
    remote_demo_doctor.add_argument("--http-timeout", type=float, default=5.0)
    remote_demo_doctor.add_argument("--admin-results-limit", type=int, default=10)
    remote_demo_doctor.add_argument("--require-result", action="store_true")
    remote_demo_doctor.add_argument("--json", action="store_true")
    remote_demo_verify = remote_demo_subparsers.add_parser(
        "verify",
        help="Create and verify a read-only remote home-compute session.",
    )
    remote_demo_verify.add_argument("--workload", choices=["model-bundle", "external-llm"], default="model-bundle")
    remote_demo_verify.add_argument("--coordinator-url", required=True)
    remote_demo_verify.add_argument("--miner-id", required=True)
    remote_demo_verify.add_argument("--observer-token", required=True)
    remote_demo_verify.add_argument("--admin-token", required=True)
    remote_demo_verify.add_argument("--output-dir", default="dist/remote-home-compute")
    remote_demo_verify.add_argument("--request-count", type=int, default=4)
    remote_demo_verify.add_argument("--scenario-id", default="route-baseline")
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
    remote_demo_collect.add_argument("--workload", choices=["model-bundle", "external-llm"], default="model-bundle")
    remote_demo_collect.add_argument("--coordinator-url", required=True)
    remote_demo_collect.add_argument("--miner-id", required=True)
    remote_demo_collect.add_argument("--observer-token", required=True)
    remote_demo_collect.add_argument("--admin-token", required=True)
    remote_demo_collect.add_argument("--output-dir", default="dist/remote-home-compute")
    remote_demo_collect.add_argument("--request-count", type=int, default=4)
    remote_demo_collect.add_argument("--scenario-id", default="route-baseline")
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
    args = parser.parse_args(argv)
    if args.command in {"local-proof", "home-infer", "llm-infer", "release-ready", "remote-runbook", "remote-acceptance"} or (
        args.command == "remote-demo" and hasattr(args, "request_count")
    ):
        if args.request_count < 1:
            raise SystemExit("--request-count must be at least 1")
        if args.timeout_seconds < 1:
            raise SystemExit("--timeout-seconds must be at least 1")
    if args.command == "release-ready":
        if args.base_port < 1:
            raise SystemExit("--base-port must be positive")
        if args.external_llm_request_count < 1:
            raise SystemExit("--external-llm-request-count must be at least 1")
    if args.command == "llm-infer":
        if args.llm_runtime_cmd and args.llm_runtime_url:
            raise SystemExit("--llm-runtime-cmd and --llm-runtime-url are mutually exclusive")
        if args.llm_runtime_timeout <= 0:
            raise SystemExit("--llm-runtime-timeout must be positive")
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
