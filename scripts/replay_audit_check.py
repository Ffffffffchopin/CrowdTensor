#!/usr/bin/env python3
"""Subprocess smoke test for deterministic replay audit mode."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DENSE_MINER_ID = "replay-audit-dense"
LORA_MINER_ID = "replay-audit-lora"
MICRO_TRANSFORMER_MINER_ID = "replay-audit-micro-transformer"
MODEL_BUNDLE_MINER_ID = "replay-audit-model-bundle"
for path in [ROOT, ROOT / "scripts"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from auth_headers import activate_miner_token, activate_observer_token, add_miner_token_arg, add_observer_token_arg, coordinator_env, json_headers, observer_headers  # noqa: E402
from crowdtensor.diloco import run_inner_loop  # noqa: E402
from crowdtensor.model_bundle import run_model_bundle_inner_loop  # noqa: E402
from crowdtensor.outer_optimizer import DELTA_FORMAT_SIGN_COMPRESSED, compress_sign_delta  # noqa: E402


def request_json(
    method: str,
    base_url: str,
    path: str,
    payload: dict | None = None,
    *,
    expected_status: int = 200,
    timeout: float = 5.0,
) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers=observer_headers() if method == "GET" else json_headers(),
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            if response.status != expected_status:
                raise RuntimeError(f"expected HTTP {expected_status}, got {response.status}: {raw}")
            return json.loads(raw) if raw else {}
    except HTTPError as exc:
        raw = exc.read().decode("utf-8")
        if exc.code != expected_status:
            raise RuntimeError(f"expected HTTP {expected_status}, got {exc.code}: {raw}") from exc
        return json.loads(raw) if raw else {}


def wait_health(base_url: str, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"coordinator exited early with code {proc.returncode}")
        try:
            health = request_json("GET", base_url, "/health", timeout=2.0)
            if health.get("ok") is True:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError(f"coordinator did not become healthy: {last_error}")


def start_coordinator(args: argparse.Namespace, state_dir: Path) -> subprocess.Popen:
    command = [
        sys.executable,
        str(ROOT / "coordinator.py"),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--state-dir",
        str(state_dir),
        "--lease-seconds",
        str(args.lease_seconds),
        "--inner-steps",
        str(args.inner_steps),
        "--backlog",
        "0",
        "--replay-audit",
        "--task-lane",
        "python-cli:cpu:1:diloco_train",
        "--task-lane",
        "python-cli:cpu:1:cpu_lora_mock",
        "--task-lane",
        "python-cli:cpu:1:micro_transformer_lm",
        "--task-lane",
        "python-cli:cpu:2:model_bundle_lm",
        "--delta-format",
        DELTA_FORMAT_SIGN_COMPRESSED,
    ]
    env = coordinator_env()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    wait_health(args.base_url, proc, args.startup_timeout)
    return proc


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def dense_capabilities() -> dict:
    return {
        "runtime": "python-cli",
        "backend": "cpu",
        "protocol_version": "runtime_contract_v1",
        "supported_workloads": ["diloco_train"],
        "supported_delta_formats": [DELTA_FORMAT_SIGN_COMPRESSED],
    }


def lora_capabilities() -> dict:
    return {
        "runtime": "python-cli",
        "backend": "cpu",
        "protocol_version": "runtime_contract_v1",
        "supported_workloads": ["cpu_lora_mock"],
    }


def micro_transformer_capabilities() -> dict:
    return {
        "runtime": "python-cli",
        "backend": "cpu",
        "protocol_version": "runtime_contract_v1",
        "supported_workloads": ["micro_transformer_lm"],
    }


def model_bundle_capabilities() -> dict:
    return {
        "runtime": "python-cli",
        "backend": "cpu",
        "protocol_version": "runtime_contract_v1",
        "supported_workloads": ["model_bundle_lm"],
    }


def reject_bad_dense(base_url: str) -> None:
    claim = request_json(
        "POST",
        base_url,
        "/tasks/claim",
        {"miner_id": DENSE_MINER_ID, "capabilities": dense_capabilities()},
    )
    if claim.get("audit_mode") != "replay" or claim.get("workload_type") != "diloco_train":
        raise RuntimeError(f"expected audited diloco_train claim, got {claim}")
    if (claim.get("optimizer_spec") or {}).get("delta_format") != DELTA_FORMAT_SIGN_COMPRESSED:
        raise RuntimeError(f"expected sign_compressed claim, got {claim}")

    inner_result = run_inner_loop(
        claim["weights"],
        task_id=claim["task_id"],
        miner_id=DENSE_MINER_ID,
        model_version=int(claim["model_version"]),
        inner_steps=int(claim["inner_steps"]),
        training_spec=claim["training_spec"],
    )
    compressed = compress_sign_delta(inner_result["local_delta"])
    compressed["signs"] = [-sign for sign in compressed["signs"]]

    rejected = request_json(
        "POST",
        base_url,
        f"/tasks/{claim['task_id']}/result",
        {
            "lease_token": claim["lease_token"],
            "attempt": claim["attempt"],
            "compressed_delta": compressed,
        },
        expected_status=422,
    )
    if rejected.get("detail", {}).get("code") != "local_delta_replay_mismatch":
        raise RuntimeError(f"unexpected dense rejection payload: {rejected}")


def accept_sign_compressed_dense(base_url: str) -> None:
    claim = request_json(
        "POST",
        base_url,
        "/tasks/claim",
        {"miner_id": f"{DENSE_MINER_ID}-compressed", "capabilities": dense_capabilities()},
    )
    if claim.get("audit_mode") != "replay" or claim.get("workload_type") != "diloco_train":
        raise RuntimeError(f"expected audited diloco_train claim, got {claim}")
    if (claim.get("optimizer_spec") or {}).get("delta_format") != DELTA_FORMAT_SIGN_COMPRESSED:
        raise RuntimeError(f"expected sign_compressed claim, got {claim}")

    inner_result = run_inner_loop(
        claim["weights"],
        task_id=claim["task_id"],
        miner_id=f"{DENSE_MINER_ID}-compressed",
        model_version=int(claim["model_version"]),
        inner_steps=int(claim["inner_steps"]),
        training_spec=claim["training_spec"],
    )
    accepted = request_json(
        "POST",
        base_url,
        f"/tasks/{claim['task_id']}/result",
        {
            "lease_token": claim["lease_token"],
            "attempt": claim["attempt"],
            "compressed_delta": compress_sign_delta(inner_result["local_delta"]),
            "metrics": {"delta_format": "sign_compressed"},
        },
    )
    if accepted.get("accepted") is not True:
        raise RuntimeError(f"expected compressed dense acceptance, got {accepted}")
    optimizer = accepted.get("optimizer") or {}
    if optimizer.get("delta_format") != "sign_compressed":
        raise RuntimeError(f"expected sign_compressed optimizer summary, got {accepted}")


def reject_bad_lora(base_url: str) -> None:
    claim = request_json(
        "POST",
        base_url,
        "/tasks/claim",
        {"miner_id": LORA_MINER_ID, "capabilities": lora_capabilities()},
    )
    if claim.get("audit_mode") != "replay" or claim.get("workload_type") != "cpu_lora_mock":
        raise RuntimeError(f"expected audited cpu_lora_mock claim, got {claim}")

    rejected = request_json(
        "POST",
        base_url,
        f"/tasks/{claim['task_id']}/result",
        {
            "lease_token": claim["lease_token"],
            "attempt": claim["attempt"],
            "adapter_delta": {"values": [0.0 for _ in claim["weights"]]},
        },
        expected_status=422,
    )
    if rejected.get("detail", {}).get("code") != "adapter_delta_replay_mismatch":
        raise RuntimeError(f"unexpected lora rejection payload: {rejected}")


def reject_bad_micro_transformer(base_url: str) -> None:
    claim = request_json(
        "POST",
        base_url,
        "/tasks/claim",
        {"miner_id": MICRO_TRANSFORMER_MINER_ID, "capabilities": micro_transformer_capabilities()},
    )
    if claim.get("audit_mode") != "replay" or claim.get("workload_type") != "micro_transformer_lm":
        raise RuntimeError(f"expected audited micro_transformer_lm claim, got {claim}")

    rejected = request_json(
        "POST",
        base_url,
        f"/tasks/{claim['task_id']}/result",
        {
            "lease_token": claim["lease_token"],
            "attempt": claim["attempt"],
            "local_delta": [0.0 for _ in claim["weights"]],
        },
        expected_status=422,
    )
    if rejected.get("detail", {}).get("code") != "micro_transformer_delta_replay_mismatch":
        raise RuntimeError(f"unexpected micro_transformer rejection payload: {rejected}")


def accept_model_bundle(base_url: str) -> None:
    claim = request_json(
        "POST",
        base_url,
        "/tasks/claim",
        {"miner_id": f"{MODEL_BUNDLE_MINER_ID}-accepted", "capabilities": model_bundle_capabilities()},
    )
    if claim.get("audit_mode") != "replay" or claim.get("workload_type") != "model_bundle_lm":
        raise RuntimeError(f"expected audited model_bundle_lm claim, got {claim}")

    inner_result = run_model_bundle_inner_loop(
        claim["workload_spec"],
        inner_steps=int(claim["inner_steps"]),
    )
    accepted = request_json(
        "POST",
        base_url,
        f"/tasks/{claim['task_id']}/result",
        {
            "lease_token": claim["lease_token"],
            "attempt": claim["attempt"],
            "bundle_delta": inner_result["bundle_delta"],
            "metrics": {
                "bundle_loss_start": inner_result["bundle_loss_start"],
                "bundle_loss_end": inner_result["bundle_loss_end"],
            },
        },
    )
    if accepted.get("accepted") is not True or accepted.get("model_bundle_updated") is not True:
        raise RuntimeError(f"expected model_bundle acceptance, got {accepted}")


def reject_bad_model_bundle(base_url: str) -> None:
    claim = request_json(
        "POST",
        base_url,
        "/tasks/claim",
        {"miner_id": MODEL_BUNDLE_MINER_ID, "capabilities": model_bundle_capabilities()},
    )
    if claim.get("audit_mode") != "replay" or claim.get("workload_type") != "model_bundle_lm":
        raise RuntimeError(f"expected audited model_bundle_lm claim, got {claim}")

    inner_result = run_model_bundle_inner_loop(
        claim["workload_spec"],
        inner_steps=int(claim["inner_steps"]),
    )
    bad_delta = dict(inner_result["bundle_delta"])
    values = list(bad_delta["values"])
    values[0] += 0.001
    bad_delta["values"] = values
    rejected = request_json(
        "POST",
        base_url,
        f"/tasks/{claim['task_id']}/result",
        {
            "lease_token": claim["lease_token"],
            "attempt": claim["attempt"],
            "bundle_delta": bad_delta,
        },
        expected_status=422,
    )
    if rejected.get("detail", {}).get("code") != "model_bundle_delta_replay_mismatch":
        raise RuntimeError(f"unexpected model_bundle rejection payload: {rejected}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run replay audit smoke test.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8894)
    parser.add_argument("--state-dir", default="")
    parser.add_argument("--lease-seconds", type=float, default=10.0)
    parser.add_argument("--inner-steps", type=int, default=20)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    add_miner_token_arg(parser)
    add_observer_token_arg(parser)
    args = parser.parse_args()
    activate_miner_token(args)
    activate_observer_token(args)
    args.base_url = f"http://{args.host}:{args.port}"
    return args


def main() -> None:
    args = parse_args()
    temp_dir = None
    state_dir = Path(args.state_dir) if args.state_dir else None
    if state_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="crowdtensor_replay_audit_")
        state_dir = Path(temp_dir.name)

    coordinator = None
    try:
        coordinator = start_coordinator(args, state_dir)
        reject_bad_dense(args.base_url)
        accept_sign_compressed_dense(args.base_url)
        reject_bad_lora(args.base_url)
        reject_bad_micro_transformer(args.base_url)
        accept_model_bundle(args.base_url)
        reject_bad_model_bundle(args.base_url)
        reject_bad_lora(args.base_url)

        blocked = request_json(
            "POST",
            args.base_url,
            "/tasks/claim",
            {"miner_id": LORA_MINER_ID, "capabilities": lora_capabilities()},
            expected_status=503,
        )
        if blocked.get("detail") != "miner quarantined for workload":
            raise SystemExit(f"unexpected blocked claim response: {blocked}")

        state = request_json("GET", args.base_url, "/state")
        lora_score = state["miner_workload_scores"][LORA_MINER_ID]["cpu_lora_mock"]
        if state.get("audit_results") != 7 or state.get("audit_rejections") != 5:
            raise SystemExit(f"unexpected audit counters: {json.dumps(state, sort_keys=True)}")
        if not lora_score.get("quarantined"):
            raise SystemExit(f"lora miner was not quarantined: {json.dumps(state, sort_keys=True)}")

        print(json.dumps({
            "audit_results": state["audit_results"],
            "audit_rejections": state["audit_rejections"],
            "compressed_dense_accepted": True,
            "dense_rejected": True,
            "lora_quarantined": lora_score["quarantined"],
            "lora_rejected": lora_score["rejected"],
            "model_bundle_accepted": True,
            "model_bundle_rejected": True,
            "micro_transformer_rejected": True,
        }, sort_keys=True))
    finally:
        stop_process(coordinator)
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
