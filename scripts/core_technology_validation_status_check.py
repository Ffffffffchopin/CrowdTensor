#!/usr/bin/env python3
"""Validate the core technology validation status artifact."""

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

from scripts import core_technology_validation_status_pack as pack  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def artifact_path(report: dict[str, Any], name: str) -> Path:
    output_dir = Path(str(report.get("output_dir") or "."))
    artifact = (report.get("artifacts") or {}).get(name) if isinstance(report.get("artifacts"), dict) else {}
    if not isinstance(artifact, dict):
        raise SystemExit(f"missing artifact {name}")
    path = Path(str(artifact.get("path") or ""))
    if not path.is_absolute():
        path = output_dir / path
    if not path.is_file():
        raise SystemExit(f"artifact missing: {path}")
    return path


def run_pack(output_dir: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "core_technology_validation_status_pack.py"),
            "--output-dir",
            str(output_dir),
            "--json",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    for line in reversed([line.strip() for line in completed.stdout.splitlines() if line.strip()]):
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return loaded
    raise SystemExit(
        f"pack emitted no JSON\nreturncode={completed.returncode}\nstdout={completed.stdout}\nstderr={completed.stderr}"
    )


def validate_report(report: dict[str, Any], *, require_core_ready: bool = False) -> None:
    if report.get("schema") != pack.SCHEMA:
        raise SystemExit(f"unexpected schema: {report.get('schema')}")
    safety = report.get("safety") if isinstance(report.get("safety"), dict) else {}
    if safety.get("public_artifact_safe") is not True:
        raise SystemExit("status artifact is not public safe")
    if report.get("public_leak_paths"):
        raise SystemExit(f"public leak paths present: {report.get('public_leak_paths')}")
    for name in ["summary_json", "summary_markdown", "support_bundle_json"]:
        artifact_path(report, name)
    codes = set(report.get("diagnosis_codes") or [])
    if report.get("small_tier_gpu_validated"):
        if "core_small_tier_kaggle_gpu_validated" not in codes:
            raise SystemExit("small-tier flag/code mismatch")
    if report.get("seven_b_eight_b_validated"):
        if "core_7b_8b_kaggle_validation_ready" not in codes:
            raise SystemExit("7B/8B flag/code mismatch")
    else:
        if "core_7b_8b_kaggle_validation_not_ready" not in codes:
            raise SystemExit("missing 7B/8B not-ready code")
    truth = report.get("readiness_truth") if isinstance(report.get("readiness_truth"), dict) else {}
    if report.get("small_tier_gpu_validated") and not report.get("seven_b_eight_b_validated"):
        if truth.get("small_tier_success_is_not_7b_8b_completion") is not True:
            raise SystemExit("small-tier success must not be treated as 7B/8B completion")
        if report.get("core_validation_ready") is True:
            raise SystemExit("core readiness overclaimed from small-tier evidence")
    llama_local = report.get("llama_like_local_evidence") if isinstance(report.get("llama_like_local_evidence"), dict) else {}
    if llama_local.get("ready") and llama_local.get("large_model_validation") is not False:
        raise SystemExit("llama-like local smoke must not be treated as large-model validation")
    stage_selective = report.get("stage_selective_weight_loading_evidence") if isinstance(report.get("stage_selective_weight_loading_evidence"), dict) else {}
    if stage_selective.get("ready"):
        if stage_selective.get("large_model_validation") is not False:
            raise SystemExit("stage-selective loading must not be treated as large-model validation")
        if stage_selective.get("runtime_execution_validation") is not False:
            raise SystemExit("stage-selective loading must not be treated as runtime execution validation")
        truth = report.get("readiness_truth") if isinstance(report.get("readiness_truth"), dict) else {}
        if truth.get("stage_selective_weight_loading_is_not_7b_8b_completion") is not True:
            raise SystemExit("stage-selective loading must not be treated as 7B/8B completion")
        if stage_selective.get("partial_weight_tensor_application_ready") and truth.get("stage_selective_weight_application_is_not_runtime_execution") is not True:
            raise SystemExit("stage-selective application must not be treated as runtime execution")
    if llama_local.get("ready") and report.get("seven_b_eight_b_validated") is not True:
        truth = report.get("readiness_truth") if isinstance(report.get("readiness_truth"), dict) else {}
        if report.get("core_validation_ready") is True or truth.get("do_not_treat_core_layer_complete") is not True:
            raise SystemExit("llama-like local smoke overclaimed core readiness")
    if stage_selective.get("ready") and report.get("seven_b_eight_b_validated") is not True:
        truth = report.get("readiness_truth") if isinstance(report.get("readiness_truth"), dict) else {}
        if report.get("core_validation_ready") is True or truth.get("do_not_treat_core_layer_complete") is not True:
            raise SystemExit("stage-selective loading overclaimed core readiness")
    if report.get("core_validation_ready"):
        if "core_technology_validation_ready" not in codes:
            raise SystemExit("core readiness flag/code mismatch")
        if report.get("seven_b_eight_b_validated") is not True:
            raise SystemExit("core readiness requires 7B/8B validation")
    else:
        if "core_technology_validation_incomplete" not in codes:
            raise SystemExit("missing core incomplete code")
    if require_core_ready and report.get("core_validation_ready") is not True:
        raise SystemExit(f"core validation required but incomplete: {report.get('blockers')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate core technology validation status.")
    parser.add_argument("--report", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--require-core-ready", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = load_json(Path(args.report)) if args.report else run_pack(Path(args.output_dir or tempfile.mkdtemp(prefix="crowdtensor_core_status_check_")))
    validate_report(report, require_core_ready=args.require_core_ready)
    result = {
        "ok": True,
        "schema": "core_technology_validation_status_check_v1",
        "report_schema": report.get("schema"),
        "core_validation_ready": bool(report.get("core_validation_ready")),
        "small_tier_gpu_validated": bool(report.get("small_tier_gpu_validated")),
        "llama_like_local_ready": bool((report.get("llama_like_local_evidence") or {}).get("ready")) if isinstance(report.get("llama_like_local_evidence"), dict) else False,
        "stage_selective_weight_loading_ready": bool((report.get("stage_selective_weight_loading_evidence") or {}).get("ready")) if isinstance(report.get("stage_selective_weight_loading_evidence"), dict) else False,
        "seven_b_eight_b_validated": bool(report.get("seven_b_eight_b_validated")),
        "diagnosis_codes": report.get("diagnosis_codes"),
        "blockers": report.get("blockers"),
        "output_dir": report.get("output_dir"),
    }
    print(json.dumps(result, sort_keys=True) if args.json else json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
