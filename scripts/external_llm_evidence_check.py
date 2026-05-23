#!/usr/bin/env python3
"""CI-safe check for the external LLM evidence pack."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = "/tmp/crowdtensor_external_llm_evidence.json"
DEFAULT_MARKDOWN = "/tmp/crowdtensor_external_llm_evidence.md"


def validate_report(report: dict[str, Any], *, request_count: int) -> None:
    if report.get("schema") != "external_llm_evidence_v1" or report.get("ok") is not True:
        raise SystemExit(f"unexpected external LLM evidence report: {json.dumps(report, sort_keys=True)}")
    if "external_llm_evidence_ready" not in report.get("diagnosis_codes", []):
        raise SystemExit(f"missing success diagnosis: {report.get('diagnosis_codes')}")
    adapter = report.get("adapter") or {}
    if adapter.get("kind") != "mock":
        raise SystemExit(f"check expects mock adapter: {adapter}")
    summary = report.get("summary") or {}
    if int(summary.get("request_count") or 0) != request_count:
        raise SystemExit(f"request_count mismatch: {summary}")
    if int(summary.get("completion_count") or 0) != request_count:
        raise SystemExit(f"completion_count mismatch: {summary}")
    if int(summary.get("output_chars") or 0) <= 0:
        raise SystemExit(f"output chars missing: {summary}")
    safety = report.get("safety") or {}
    if not safety.get("read_only") or not safety.get("redaction_ok") or safety.get("raw_payloads_exposed"):
        raise SystemExit(f"unsafe external LLM evidence: {safety}")
    encoded = json.dumps(report, sort_keys=True)
    for fragment in [
        "external_llm_result",
        "external_llm_results",
        "output_text",
        "Bearer ",
        "local-runtime-key",
    ]:
        if fragment in encoded:
            raise SystemExit(f"external LLM evidence leaked secret-like payload: {fragment}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the external LLM evidence pack check.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8919)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--request-count", type=int, default=3)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--json-out", default=DEFAULT_REPORT)
    parser.add_argument("--markdown-out", default=DEFAULT_MARKDOWN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temp_dir = None
    state_dir = args.state_dir
    if not state_dir:
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_external_llm_evidence_check_")
        state_dir = temp_dir.name
    command = [
        sys.executable,
        str(ROOT / "scripts" / "external_llm_evidence_pack.py"),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--state-dir",
        state_dir,
        "--request-count",
        str(args.request_count),
        "--startup-timeout",
        str(args.startup_timeout),
        "--mock",
        "--json-out",
        args.json_out,
        "--markdown-out",
        args.markdown_out,
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=args.timeout,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "external_llm_evidence_pack.py failed\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("external_llm_evidence_pack.py emitted no JSON")
        report = json.loads(lines[-1])
        validate_report(report, request_count=args.request_count)
        report_path = Path(args.json_out)
        if not report_path.is_file():
            raise SystemExit(f"missing external LLM evidence JSON: {report_path}")
        persisted = json.loads(report_path.read_text(encoding="utf-8"))
        validate_report(persisted, request_count=args.request_count)
        markdown_path = Path(args.markdown_out)
        if not markdown_path.is_file() or "External LLM Evidence" not in markdown_path.read_text(encoding="utf-8"):
            raise SystemExit(f"missing external LLM evidence Markdown: {markdown_path}")
        print(json.dumps({
            "ok": True,
            "schema": report["schema"],
            "route": (report.get("route") or {}).get("name"),
            "adapter_kind": (report.get("adapter") or {}).get("kind"),
            "request_count": (report.get("summary") or {}).get("request_count"),
            "completion_count": (report.get("summary") or {}).get("completion_count"),
            "diagnosis_codes": report.get("diagnosis_codes", []),
            "report": str(report_path),
            "markdown": str(markdown_path),
        }, sort_keys=True))
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
