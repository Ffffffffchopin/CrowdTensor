#!/usr/bin/env python3
"""CI-safe check for CrowdTensor session_protocol_v1 public summaries."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor.session_protocol import (  # noqa: E402
    assert_public_safe,
    build_route_decision,
    build_session_request,
    coordinator_payload_for_request,
    public_leak_paths,
    safe_generation_summary,
)


SCHEMA = "session_protocol_check_v1"


def run_check(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir or tempfile.mkdtemp(prefix="crowdtensor_session_protocol_check_")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    session_request = build_session_request(
        prompt_text="CrowdTensor public product route",
        backend="cuda",
        stage_mode="split",
        max_new_tokens=8,
        route_source="peer-bootstrap",
    )
    private_payload = coordinator_payload_for_request(session_request, prompt_text="CrowdTensor public product route")
    peer_catalog = [
        {
            "peer_id": "coordinator-a",
            "role": "coordinator",
            "urls": {"coordinator": "http://127.0.0.1:8787"},
            "capabilities": {"runtime": "python-cli", "backend": "cuda"},
        },
        {
            "peer_id": "stage0",
            "role": "miner",
            "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_cuda_stage0"]},
        },
        {
            "peer_id": "stage1",
            "role": "miner",
            "capabilities": {"real_llm_sharded_stage_capabilities": ["real_llm_sharded_cuda_stage1"]},
        },
    ]
    route = build_route_decision(session_request, peer_catalog=peer_catalog)
    generation = safe_generation_summary(
        {
            "generation": {
                "max_new_tokens": 8,
                "generated_token_count": 8,
                "generated_text_hash": "sha256:generated",
                "decoded_tokens_match": True,
            }
        },
        max_new_tokens=8,
    )
    errors: list[str] = []
    if session_request.get("schema") != "session_protocol_v1":
        errors.append("session_schema_mismatch")
    if "prompt_hash" not in session_request or "CrowdTensor public product route" in json.dumps(session_request, sort_keys=True):
        errors.append("prompt_redaction_failed")
    if private_payload.get("prompt") != "CrowdTensor public product route":
        errors.append("private_payload_prompt_missing")
    if not route.get("usable_now"):
        errors.append("route_not_ready")
    if generation.get("generated_token_count") != 8 or generation.get("generated_text_hash") != "sha256:generated":
        errors.append("generation_summary_mismatch")
    try:
        assert_public_safe(session_request)
        assert_public_safe(generation)
    except ValueError as exc:
        errors.append(f"public_safety_failed:{exc}")
    leak_probe = public_leak_paths({"generated_text": "raw"})
    if not leak_probe:
        errors.append("leak_probe_failed")

    result = {
        "schema": SCHEMA,
        "ok": not errors,
        "output_dir": str(output_dir),
        "session_request": session_request,
        "route": route,
        "generation": generation,
        "errors": errors,
        "diagnosis_codes": ["session_protocol_ready"] if not errors else ["session_protocol_blocked"],
        "safety": {
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "private_payload_is_not_public": True,
        },
    }
    (output_dir / "session_protocol_check.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check session_protocol_v1 helpers.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = run_check(args)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"Session protocol ready: {result.get('ok')}")
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
