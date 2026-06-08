#!/usr/bin/env python3
"""CI-safe check for CrowdTensor P2P-lite discovery and route resolution."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor.p2p_lite import PeerCatalog, peer_leak_paths, write_catalog_file  # noqa: E402
from crowdtensor.protocol import (  # noqa: E402
    REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY,
    REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY,
    REAL_LLM_SHARDED_STAGE0_CAPABILITY,
    REAL_LLM_SHARDED_STAGE1_CAPABILITY,
)
from crowdtensor.session_protocol import build_session_request  # noqa: E402


SCHEMA = "p2p_lite_discovery_check_v1"


def output_request_summary() -> dict[str, Any]:
    return {
        "include_output": False,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "local_output_display_only": False,
        "public_artifact_safe": True,
        "summary": (
            "P2P-lite discovery check artifacts validate route matching and TTL "
            "pruning only; they do not include answer text."
        ),
    }


def prompt_scope_summary() -> dict[str, Any]:
    return {
        "source": "route-check-placeholder",
        "prompt_count": 1,
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
            "This discovery-only check uses an internal placeholder session request "
            "for route resolution; raw prompt text is not public."
        ),
    }


def answer_scope_summary() -> dict[str, Any]:
    return {
        "scope_state": "no-local-answer",
        "terminal_only": False,
        "visible_in_terminal": False,
        "saved_json_display": "route-only",
        "saved_markdown_display": "none",
        "json_stdout_display": "route-only-json",
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "public_artifact_safe": True,
        "summary": "Discovery checks are not answer transcripts.",
    }


def shareable_summary() -> dict[str, Any]:
    return {
        "saved_artifacts_public_safe": True,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "local_output_display_only": False,
        "answer_scope_state": "no-local-answer",
        "local_answer_terminal_only": False,
        "public_artifact_safe": True,
        "summary": "Share discovery readiness and route metadata only.",
    }


def stage_peer(peer_id: str, capability: str, *, backend: str, stage_role: str, now: float) -> dict[str, Any]:
    return {
        "schema": "p2p_lite_peer_v1",
        "swarm_id": "check-swarm",
        "peer_id": peer_id,
        "role": "miner",
        "urls": {"peer": f"http://127.0.0.1/{peer_id}"},
        "backend": backend,
        "stage_role": stage_role,
        "capabilities": {
            "runtime": "python-cli",
            "backend": backend,
            "real_llm_sharded_stage_role": stage_role,
            "real_llm_sharded_stage_capabilities": [capability],
        },
        "ttl_seconds": 30.0,
        "last_seen": now,
    }


def coordinator_peer(now: float) -> dict[str, Any]:
    return {
        "schema": "p2p_lite_peer_v1",
        "swarm_id": "check-swarm",
        "peer_id": "coordinator-a",
        "role": "coordinator",
        "urls": {"coordinator": "http://127.0.0.1:8787", "peer": "http://127.0.0.1:8788"},
        "capabilities": {"runtime": "python-cli", "backend": "cpu"},
        "ttl_seconds": 30.0,
        "last_seen": now,
    }


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir or tempfile.mkdtemp(prefix="crowdtensor_p2p_lite_check_")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    now = 1_800_000_000.0
    catalog = PeerCatalog(swarm_id="check-swarm", ttl_seconds=30.0)
    catalog.announce(coordinator_peer(now), now=now)
    catalog.announce(stage_peer("cpu-stage0", REAL_LLM_SHARDED_STAGE0_CAPABILITY, backend="cpu", stage_role="stage0", now=now), now=now)
    catalog.announce(stage_peer("cpu-stage1", REAL_LLM_SHARDED_STAGE1_CAPABILITY, backend="cpu", stage_role="stage1", now=now), now=now)
    catalog.announce(stage_peer("cuda-stage0", REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY, backend="cuda", stage_role="stage0", now=now), now=now)
    catalog.announce(stage_peer("cuda-stage1", REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY, backend="cuda", stage_role="stage1", now=now), now=now)

    cpu_request = build_session_request(
        prompt_text="CrowdTensor route check",
        backend="cpu",
        stage_mode="split",
        max_new_tokens=4,
        route_source="peer-bootstrap",
    )
    cuda_request = build_session_request(
        prompt_text="CrowdTensor route check",
        backend="cuda",
        stage_mode="split",
        max_new_tokens=4,
        route_source="peer-bootstrap",
    )
    cpu_route = catalog.resolve(cpu_request, now=now)
    cuda_route = catalog.resolve(cuda_request, now=now)
    peer_count_before = len(catalog.peers(now=now))
    expired = catalog.prune(now=now + 31.0)
    peer_count_after = len(catalog.peers(now=now + 31.0))

    payload = catalog.catalog_payload(now=now)
    leaks = peer_leak_paths(payload)
    errors: list[str] = []
    if not cpu_route.get("ok"):
        errors.append("cpu_route_not_ready")
    if not cuda_route.get("ok"):
        errors.append("cuda_route_not_ready")
    if peer_count_before != 5:
        errors.append("peer_count_before_mismatch")
    if expired != 5 or peer_count_after != 0:
        errors.append("ttl_prune_failed")
    if leaks:
        errors.append("private_metadata_leak")

    write_catalog_file(output_dir / "p2p_lite_catalog.json", catalog)
    result = {
        "schema": SCHEMA,
        "ok": not errors,
        "output_dir": str(output_dir),
        "peer_count_before_prune": peer_count_before,
        "expired_peer_count": expired,
        "peer_count_after_prune": peer_count_after,
        "cpu_route": cpu_route,
        "cuda_route": cuda_route,
        "errors": errors,
        "diagnosis_codes": ["p2p_lite_discovery_ready"] if not errors else ["p2p_lite_discovery_blocked"],
        "output_request": output_request_summary(),
        "prompt_scope": prompt_scope_summary(),
        "answer_scope": answer_scope_summary(),
        "shareable_summary": shareable_summary(),
        "safety": {
            "tokens_gossiped": False,
            "raw_prompts_gossiped": False,
            "activations_gossiped": False,
            "not_libp2p": True,
            "not_dht": True,
            "not_nat_traversal": True,
        },
        "artifacts": {
            "catalog": str(output_dir / "p2p_lite_catalog.json"),
        },
    }
    (output_dir / "p2p_lite_discovery_check.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check P2P-lite discovery route resolution.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = run_check(args)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"P2P-lite discovery ready: {result.get('ok')}")
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
