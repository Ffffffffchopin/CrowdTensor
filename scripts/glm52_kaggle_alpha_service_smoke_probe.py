#!/usr/bin/env python3
"""Smoke-check the local GLM 5.2 Kaggle Alpha HTTP service."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor import glm52_kaggle_alpha as alpha  # noqa: E402


SCHEMA = "glm52_kaggle_alpha_service_smoke_probe_v1"
DEFAULT_OUTPUT_DIR = "dist/glm52-kaggle-alpha-service-smoke"
SENSITIVE_FRAGMENTS = alpha.SENSITIVE_FRAGMENTS + (
    "service smoke prompt",
    "CrowdTensor GLM 5.2 Alpha service smoke",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {}
    loaded = json.loads(p.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return sorted({fragment for fragment in SENSITIVE_FRAGMENTS if fragment in encoded})


def resume_private_inputs_verified(payload: dict[str, Any]) -> bool:
    resume_private_inputs = _dict(payload.get("resume_private_inputs"))
    return bool(
        resume_private_inputs.get("schema") == alpha.RESUME_PRIVATE_INPUTS_SCHEMA
        and resume_private_inputs.get("required_for_live_resume") is True
        and resume_private_inputs.get("resume_command_omits_private_credentials") is True
        and resume_private_inputs.get("kaggle_credentials_required") is True
        and resume_private_inputs.get("kaggle_credential_values_public") is False
        and resume_private_inputs.get("hf_env_values_public") is False
        and resume_private_inputs.get("public_artifact_safe") is True
    )


def http_json(url: str, *, method: str = "GET", body: dict[str, Any] | None = None, timeout: float = 10.0) -> tuple[int, dict[str, Any]]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            loaded = json.loads(raw) if raw else {}
            return int(response.status), loaded if isinstance(loaded, dict) else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        loaded = json.loads(raw) if raw else {}
        return int(exc.code), loaded if isinstance(loaded, dict) else {}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    alpha_output_dir = Path(args.alpha_output_dir or args.output_dir)
    config = alpha.AlphaConfig(output_dir=alpha_output_dir)
    server = alpha.AlphaHTTPServer(host=str(args.host), port=int(args.port), config=config)
    server.start()
    base = f"http://{args.host}:{server.port}"
    prompt = str(args.prompt or "CrowdTensor GLM 5.2 Alpha service smoke")
    try:
        health_status, health = http_json(f"{base}/health", timeout=float(args.timeout_seconds))
        status_status, status = http_json(f"{base}/status", timeout=float(args.timeout_seconds))
        quota_blocked_before_generate = "kaggle_gpu_quota_unavailable" in _list(status.get("blockers")) or status.get("phase") == "blocked_gpu_quota"
        generate_status = 0
        generate: dict[str, Any] = {}
        generate_skipped_reason = ""
        if quota_blocked_before_generate or bool(args.allow_live_generate):
            generate_status, generate = http_json(
                f"{base}/generate",
                method="POST",
                body={
                    "prompt": prompt,
                    "max_new_tokens": int(args.max_new_tokens),
                    "timeout": float(args.generate_timeout_seconds),
                },
                timeout=float(args.timeout_seconds),
            )
        else:
            generate_skipped_reason = "live_generate_not_allowed_without_current_quota_blocker"
        cleanup_status, cleanup = http_json(
            f"{base}/cleanup",
            method="POST",
            body={},
            timeout=float(args.timeout_seconds),
        )
    finally:
        server.stop()

    generate_blockers = _list(generate.get("blockers"))
    quota_generate_verified = bool(
        generate_status == 503
        and "kaggle_gpu_quota_unavailable" in generate_blockers
        and generate.get("public_artifact_safe") is True
    )
    successful_generate_verified = bool(
        generate_status == 200
        and generate.get("same_request_decode_verified") is True
        and int(generate.get("generated_token_count") or 0) >= int(args.max_new_tokens)
        and generate.get("public_artifact_safe") is True
    )
    route_verified = quota_generate_verified or successful_generate_verified
    status_resume_private_inputs_verified = resume_private_inputs_verified(status)
    generate_resume_private_inputs_verified = resume_private_inputs_verified(generate)
    cleanup_route_verified = bool(
        cleanup_status == 200
        and cleanup.get("ok") is True
        and cleanup.get("temporary_kaggle_kernels_deleted") is True
        and cleanup.get("temporary_private_packages_removed") is True
        and cleanup.get("live_resources_left_running") is False
        and cleanup.get("public_artifact_safe") is True
    )
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": bool(health_status == 200 and status_status == 200 and route_verified and cleanup_route_verified),
        "service_http_smoke_verified": bool(
            health_status == 200 and status_status == 200 and route_verified and cleanup_route_verified
        ),
        "alpha_output_dir": str(alpha_output_dir),
        "server_bound_host": str(args.host),
        "server_bound_port": int(server.port),
        "routes_checked": [
            "GET /health",
            "GET /status",
            "POST /generate" if not generate_skipped_reason else "POST /generate skipped",
            "POST /cleanup",
        ],
        "health": {
            "http_status": health_status,
            "ok": health.get("ok") is True,
            "schema": str(health.get("schema") or ""),
            "public_artifact_safe": health.get("public_artifact_safe") is True,
        },
        "status": {
            "http_status": status_status,
            "ok": status.get("ok") is not False,
            "phase": str(status.get("phase") or ""),
            "alpha_report_present": status.get("alpha_report_present") is True,
            "glm52_kaggle_alpha_ready": status.get("glm52_kaggle_alpha_ready") is True,
            "blockers": generate_blockers if not status.get("blockers") else _list(status.get("blockers")),
            "resume_private_inputs_verified": status_resume_private_inputs_verified,
            "public_artifact_safe": status.get("public_artifact_safe") is True,
        },
        "generate": {
            "http_status": generate_status,
            "attempted": not bool(generate_skipped_reason),
            "skipped_reason": generate_skipped_reason,
            "ok": generate.get("ok") is True,
            "quota_blocker_verified": quota_generate_verified,
            "successful_generate_verified": successful_generate_verified,
            "request_prompt_hash": alpha.sha_text(prompt),
            "target_generated_token_count": int(args.max_new_tokens),
            "generated_token_count": int(generate.get("generated_token_count") or 0),
            "generated_token_hash_count": len(_list(generate.get("generated_token_hashes"))),
            "same_request_decode_verified": generate.get("same_request_decode_verified") is True,
            "accepted_providers": _list(generate.get("accepted_providers")),
            "blockers": generate_blockers,
            "resume_private_inputs_verified": generate_resume_private_inputs_verified,
            "raw_prompt_public": generate.get("raw_prompt_public") is True,
            "raw_generated_text_public": generate.get("raw_generated_text_public") is True,
            "generated_token_ids_public": generate.get("generated_token_ids_public") is True,
            "public_artifact_safe": generate.get("public_artifact_safe") is True,
        },
        "generate_route_reaches_service": not bool(generate_skipped_reason),
        "generate_route_quota_blocker_verified": quota_generate_verified,
        "generate_route_success_verified": successful_generate_verified,
        "status_resume_private_inputs_verified": status_resume_private_inputs_verified,
        "generate_resume_private_inputs_verified": bool(
            generate_resume_private_inputs_verified or successful_generate_verified
        ),
        "cleanup": {
            "http_status": cleanup_status,
            "ok": cleanup.get("ok") is True,
            "cleanup_evidence_source": str(cleanup.get("cleanup_evidence_source") or ""),
            "temporary_kaggle_kernels_deleted": cleanup.get("temporary_kaggle_kernels_deleted") is True,
            "temporary_private_packages_removed": cleanup.get("temporary_private_packages_removed") is True,
            "live_resources_left_running": cleanup.get("live_resources_left_running") is True,
            "public_artifact_safe": cleanup.get("public_artifact_safe") is True,
        },
        "cleanup_route_verified": cleanup_route_verified,
        "generate_route_live_not_started_without_quota_guard": bool(generate_skipped_reason),
        "completion_boundary": {
            "service_smoke_is_not_live_success": True,
            "quota_blocker_generate_is_not_multitoken_success": True,
            "strict_alpha_ready_still_requires_live_report": True,
        },
        "safety": alpha.safety_flags(),
        "public_artifact_safe": True,
    }
    leaks = public_redaction_errors(report)
    if leaks:
        report["ok"] = False
        report["service_http_smoke_verified"] = False
        report["public_artifact_safe"] = False
        report["redaction_errors"] = leaks
    write_json(output_dir / "glm52_kaggle_alpha_service_smoke_probe.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--alpha-output-dir", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--prompt", default="CrowdTensor GLM 5.2 Alpha service smoke")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--generate-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--allow-live-generate", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "glm52_kaggle_alpha_service_smoke_probe: "
            f"ok={report['ok']} verified={report['service_http_smoke_verified']}"
        )
    return 0 if report.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
