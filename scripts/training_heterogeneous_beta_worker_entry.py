#!/usr/bin/env python3
"""Private worker entry used inside a heterogeneous Training Beta Kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor.heterogeneous_training_manifest import (  # noqa: E402
    stable_hash,
    validate_training_manifest,
)
from crowdtensor.heterogeneous_training_miner import run_heterogeneous_miner  # noqa: E402


SCHEMA = "crowdtensor_heterogeneous_training_beta_remote_worker_v1"
BOOTSTRAP_SCHEMA = "crowdtensor_heterogeneous_training_beta_miner_bootstrap_v1"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _private_configuration(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("heterogeneous_remote_private_configuration_invalid") from exc
    if not isinstance(value, dict):
        raise RuntimeError("heterogeneous_remote_private_configuration_invalid")
    if not str(value.get("coordinator_url") or "") or not str(
        value.get("coordinator_token") or ""
    ):
        raise RuntimeError("heterogeneous_remote_private_configuration_incomplete")
    return value


def _request_json(
    coordinator_url: str,
    path: str,
    *,
    token: str,
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{str(coordinator_url).rstrip('/')}{path}",
        headers={
            "User-Agent": "crowdtensor-heterogeneous-training-kaggle/1",
            "x-crowdtensor-miner-token": str(token),
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"heterogeneous_remote_bootstrap_http_{int(exc.code)}"
        ) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError("heterogeneous_remote_coordinator_unreachable") from exc
    if not isinstance(value, dict):
        raise RuntimeError("heterogeneous_remote_bootstrap_response_invalid")
    return value


def _public_error(exc: BaseException) -> str:
    value = str(exc)
    if value.startswith(("heterogeneous_", "elastic_", "qwen15b_")):
        return re.sub(r"[^a-zA-Z0-9:_-]", "_", value[:180])
    return f"heterogeneous_remote_worker_failed:{type(exc).__name__}"


def run_worker(
    *,
    private_configuration_path: str | Path,
    output_path: str | Path,
    private_root: str | Path,
    deployment_role: str,
    identity_nonce: str,
    device_policy: str,
    max_steps: int,
    wait_timeout: float,
    operation_timeout: float = 1800.0,
    run_microbenchmark: bool = True,
    session_retries: int = 0,
    transport_optimization_after_step: int = -1,
) -> dict[str, Any]:
    started = time.time()
    private = _private_configuration(private_configuration_path)
    coordinator = str(private["coordinator_url"])
    token = str(private["coordinator_token"])
    bootstrap = _request_json(
        coordinator,
        "/elastic-training/bootstrap",
        token=token,
        timeout=min(180.0, float(wait_timeout)),
    )
    if bootstrap.get("schema") != BOOTSTRAP_SCHEMA:
        raise RuntimeError("heterogeneous_remote_bootstrap_schema_invalid")
    manifest = validate_training_manifest(bootstrap.get("training_manifest"))
    config = dict(bootstrap.get("config") or {})
    tokenized = dict(bootstrap.get("tokenized_payload") or {})
    if (
        stable_hash(config) != str(bootstrap.get("config_hash") or "")
        or stable_hash(tokenized)
        != str(bootstrap.get("tokenized_payload_hash") or "")
    ):
        raise RuntimeError("heterogeneous_remote_bootstrap_hash_mismatch")
    miner_id_hash = "sha256:" + hashlib.sha256(
        f"{deployment_role}:{identity_nonce}".encode("utf-8")
    ).hexdigest()
    worker: dict[str, Any] | None = None
    last_error: BaseException | None = None
    completed_attempt = 0
    for attempt in range(int(session_retries) + 1):
        completed_attempt = attempt
        try:
            worker = run_heterogeneous_miner(
                coordinator_url=coordinator,
                coordinator_token=token,
                run_id=str(bootstrap["run_id"]),
                miner_id_hash=miner_id_hash,
                registration_nonce="heterogeneous:" + str(identity_nonce),
                training_manifest=manifest,
                config=config,
                tokenized_payload=tokenized,
                private_root=private_root,
                device_policy=device_policy,
                cuda_devices=[0] if device_policy == "cuda" else None,
                max_stage_count=1,
                max_steps_per_session=int(max_steps),
                wait_timeout=float(wait_timeout),
                operation_timeout=float(operation_timeout),
                heartbeat_interval_seconds=5.0,
                hf_token=str(private.get("hf_token") or ""),
                attached_model_root=None,
                run_microbenchmark=bool(run_microbenchmark),
                transport_optimization_after_step=int(
                    transport_optimization_after_step
                ),
            )
            last_error = None
            break
        except BaseException as exc:
            last_error = exc
            if attempt < int(session_retries):
                time.sleep(min(10.0, 2.0 ** attempt))
    if worker is None:
        assert last_error is not None
        raise last_error
    report = {
        **worker,
        "schema": SCHEMA,
        "deployment_role": str(deployment_role),
        "single_stage_limit": True,
        "visible_cuda_device_count_expected": 1 if device_policy == "cuda" else 0,
        "jax_tpu_resource_group_expected": device_policy == "jax_tpu",
        "session_retry_count": completed_attempt,
        "session_retry_limit": int(session_retries),
        "transport_optimization_after_step": int(
            transport_optimization_after_step
        ),
        "operation_timeout_seconds": min(
            float(wait_timeout), max(30.0, float(operation_timeout))
        ),
        "elapsed_seconds": round(time.time() - started, 3),
        "credential_values_public": False,
        "coordinator_url_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    report.pop("content_hash", None)
    report["content_hash"] = stable_hash(report)
    _write_json(Path(output_path), report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-configuration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--deployment-role", required=True)
    parser.add_argument("--identity-nonce", required=True)
    parser.add_argument(
        "--device-policy", choices=["cpu", "cuda", "jax_tpu"], required=True
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--wait-timeout", type=float, default=10800.0)
    parser.add_argument("--operation-timeout", type=float, default=1800.0)
    parser.add_argument("--skip-microbenchmark", action="store_true")
    parser.add_argument("--session-retries", type=int, default=0)
    parser.add_argument("--transport-optimization-after-step", type=int, default=-1)
    args = parser.parse_args()
    output = Path(args.output)
    try:
        report = run_worker(
            private_configuration_path=args.private_configuration,
            output_path=output,
            private_root=args.private_root,
            deployment_role=args.deployment_role,
            identity_nonce=args.identity_nonce,
            device_policy=args.device_policy,
            max_steps=args.max_steps,
            wait_timeout=args.wait_timeout,
            operation_timeout=args.operation_timeout,
            run_microbenchmark=not args.skip_microbenchmark,
            session_retries=args.session_retries,
            transport_optimization_after_step=args.transport_optimization_after_step,
        )
    except BaseException as exc:
        report = {
            "schema": SCHEMA,
            "ok": False,
            "deployment_role": str(args.deployment_role),
            "blockers": [_public_error(exc)],
            "failure_detail_public": False,
            "credential_values_public": False,
            "coordinator_url_public": False,
            "raw_training_text_public": False,
            "token_ids_public": False,
            "activation_values_public": False,
            "gradient_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        report["content_hash"] = stable_hash(report)
        _write_json(output, report)
    if os.environ.get("CT_HETEROGENEOUS_WORKER_JSON") == "1":
        print(json.dumps(report, sort_keys=True))
    return 0 if report.get("ok") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
