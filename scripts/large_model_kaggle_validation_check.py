#!/usr/bin/env python3
"""CI-safe checks for the large-model Kaggle validation evidence pack."""

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

from scripts import large_model_kaggle_validation_pack as pack  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} did not contain a JSON object")
    return loaded


def run_pack(output_dir: Path, *, mode: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "large_model_kaggle_validation_pack.py"),
            "--mode",
            mode,
            "--output-dir",
            str(output_dir),
            "--json",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    for line in reversed([line.strip() for line in completed.stdout.splitlines() if line.strip()]):
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return loaded
    raise SystemExit(
        f"pack emitted no JSON or failed before report\n"
        f"returncode={completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def artifact_path(report: dict[str, Any], name: str) -> Path:
    output_dir = Path(str(report.get("output_dir") or "."))
    artifact = (report.get("artifacts") or {}).get(name) if isinstance(report.get("artifacts"), dict) else {}
    if not isinstance(artifact, dict):
        raise SystemExit(f"missing artifact: {name}")
    path = Path(str(artifact.get("path") or ""))
    if not path.is_absolute():
        path = output_dir / path
    if not path.is_file():
        raise SystemExit(f"artifact missing on disk: {name} at {path}")
    return path


def validate_report(report: dict[str, Any], *, require_real_7b: bool = False, require_core_ready: bool = False) -> None:
    if report.get("schema") != pack.SCHEMA:
        raise SystemExit(f"unexpected schema: {report.get('schema')}")
    codes = set(report.get("diagnosis_codes") or [])
    if "large_model_kaggle_validation_ready" not in codes and "large_model_kaggle_validation_blocked" not in codes:
        raise SystemExit("missing validation readiness diagnosis")
    safety = report.get("safety") if isinstance(report.get("safety"), dict) else {}
    if safety.get("public_artifact_safe") is not True:
        raise SystemExit(f"public artifact safety failed: {safety}")
    errors = pack.public_redaction_errors(report)
    if errors:
        raise SystemExit(f"public report leaked sensitive fragments: {errors}")
    for name in ["summary_json", "summary_markdown", "support_bundle_json", "run_report_normalized"]:
        artifact_path(report, name)
    model = report.get("model") if isinstance(report.get("model"), dict) else {}
    runtime = report.get("runtime") if isinstance(report.get("runtime"), dict) else {}
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    if report.get("real_runtime_verified"):
        if not model.get("model_id"):
            raise SystemExit("real validation report must expose top-level model_id")
        if not runtime.get("backend"):
            raise SystemExit("real validation report must expose top-level runtime backend")
        if int(metrics.get("generated_token_count") or 0) <= 0:
            raise SystemExit("real validation report must expose generated token count")
        if not str(metrics.get("output_digest") or "").startswith("sha256:"):
            raise SystemExit("real validation report must expose output digest")
    if report.get("real_runtime_verified"):
        inference = report.get("inference_rc_report") if isinstance(report.get("inference_rc_report"), dict) else {}
        handoff = report.get("handoff_rc_report") if isinstance(report.get("handoff_rc_report"), dict) else {}
        if inference.get("real_runtime_verified") is not True:
            raise SystemExit("real validation must import Inference RC evidence")
        if handoff.get("real_runtime_verified") is not True:
            raise SystemExit("real validation must import Handoff RC evidence")
    if report.get("real_7b_runtime_verified"):
        if "large_model_7b_runtime_verified" not in codes:
            raise SystemExit("7B readiness flag/code mismatch")
        inference = report.get("inference_rc_report") if isinstance(report.get("inference_rc_report"), dict) else {}
        if inference.get("real_7b_runtime_verified") is not True:
            raise SystemExit("7B validation must propagate to Inference RC")
    else:
        if "large_model_7b_runtime_not_verified" not in codes:
            raise SystemExit("missing 7B not-verified code")
    if require_real_7b and report.get("real_7b_runtime_verified") is not True:
        raise SystemExit(f"7B validation required but missing: blockers={report.get('blockers')}")
    if report.get("gpu_runtime_verified") is not True:
        if "large_model_kaggle_gpu_runtime_not_verified" not in codes:
            raise SystemExit("missing GPU runtime not-verified code")
    if report.get("sharded_path_verified") is not True:
        blockers = set(report.get("blockers") or [])
        if "large_model_sharded_runtime_path_not_verified" not in blockers:
            raise SystemExit("missing sharded path blocker")
    if report.get("core_validation_ready"):
        if report.get("real_7b_runtime_verified") is not True:
            raise SystemExit("core readiness requires 7B validation")
        if report.get("gpu_runtime_verified") is not True:
            raise SystemExit("core readiness requires Kaggle GPU runtime")
        if report.get("sharded_path_verified") is not True:
            raise SystemExit("core readiness requires sharded path")
        if "large_model_core_validation_ready" not in codes:
            raise SystemExit("core readiness flag/code mismatch")
    else:
        if "large_model_core_validation_not_ready" not in codes:
            raise SystemExit("missing core validation not-ready code")
    if require_core_ready and report.get("core_validation_ready") is not True:
        raise SystemExit(f"core validation required but missing: blockers={report.get('blockers')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate large-model Kaggle validation evidence.")
    parser.add_argument("--report", default="")
    parser.add_argument("--mode", choices=["package", "fixture"], default="fixture")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--require-real-7b", action="store_true")
    parser.add_argument("--require-core-ready", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.report:
        report = load_json(Path(args.report))
    else:
        output_dir = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="crowdtensor_large_model_kaggle_check_"))
        report = run_pack(output_dir, mode=args.mode)
    validate_report(report, require_real_7b=args.require_real_7b, require_core_ready=args.require_core_ready)
    result = {
        "ok": True,
        "schema": "large_model_kaggle_validation_check_v1",
        "report_schema": report.get("schema"),
        "mode": report.get("mode"),
        "real_runtime_verified": bool(report.get("real_runtime_verified")),
        "real_7b_runtime_verified": bool(report.get("real_7b_runtime_verified")),
        "gpu_runtime_verified": bool(report.get("gpu_runtime_verified")),
        "sharded_path_verified": bool(report.get("sharded_path_verified")),
        "core_validation_ready": bool(report.get("core_validation_ready")),
        "diagnosis_codes": report.get("diagnosis_codes"),
        "output_dir": report.get("output_dir"),
    }
    print(json.dumps(result, sort_keys=True) if args.json else json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
