#!/usr/bin/env python3
"""Validate in-process dual-stage KV-cache reuse with persistent real-LLM Miners."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor.real_llm import cuda_runtime_summary, missing_hf_dependencies  # noqa: E402
from crowdtensor.session_protocol import public_leak_paths  # noqa: E402


SCHEMA = "persistent_real_llm_kv_cache_check_v1"
ADMIN_TOKEN = "persistent-kv-admin"
MINER_TOKEN = "persistent-kv-miner"
OBSERVER_TOKEN = "persistent-kv-observer"
WORKLOAD_TYPE = "real_llm_sharded_infer"
SECRET_FRAGMENTS = (
    ADMIN_TOKEN,
    MINER_TOKEN,
    OBSERVER_TOKEN,
    "lease_token",
    "idempotency_key",
    '"hidden_state":',
    '"input_ids":',
    "activation_results",
    "inference_results",
    "sharded_inference_result",
    '"generated_text":',
    '"generated_token_ids":',
    '"prompt_text":',
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def request_json(
    method: str,
    base_url: str,
    path: str,
    *,
    admin_token: str = "",
    observer_token: str = "",
    timeout: float = 5.0,
) -> dict[str, Any]:
    headers = {"content-type": "application/json"}
    if admin_token:
        headers["x-crowdtensor-admin-token"] = admin_token
    if observer_token:
        headers["x-crowdtensor-observer-token"] = observer_token
    request = Request(f"{base_url.rstrip('/')}{path}", headers=headers, method=method)
    with urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def popen_command(command: list[str], *, cwd: Path = ROOT) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


def stop_process(proc: subprocess.Popen[str] | None) -> dict[str, Any]:
    if proc is None:
        return {}
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate(timeout=5)
    else:
        stdout, stderr = proc.communicate(timeout=1)
    return {
        "returncode": proc.returncode,
        "stdout_tail": (stdout or "")[-1200:],
        "stderr_tail": (stderr or "")[-1200:],
    }


def wait_health(base_url: str, proc: subprocess.Popen[str], timeout: float) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() <= deadline:
        if proc.poll() is not None:
            status = stop_process(proc)
            return False, f"serve exited early: {status}"
        try:
            if request_json("GET", base_url, "/health", timeout=2.0).get("ok") is True:
                return True, ""
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.2)
    return False, f"coordinator did not become healthy: {last_error}"


def parse_json_payload(stdout: str) -> dict[str, Any]:
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def finish_process(name: str, proc: subprocess.Popen[str], *, timeout: float) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        status = stop_process(proc)
        return {
            "name": name,
            "ok": False,
            "returncode": status.get("returncode"),
            "duration_seconds": round(time.monotonic() - started, 3),
            "error": "timeout",
            "stdout_tail": status.get("stdout_tail", ""),
            "stderr_tail": status.get("stderr_tail", ""),
        }, {}
    payload = parse_json_payload(stdout or "")
    step = {
        "name": name,
        "ok": proc.returncode == 0 and bool(payload),
        "returncode": proc.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
    }
    if not payload:
        step["error"] = "json_payload_missing"
    if not step["ok"]:
        step["stdout_tail"] = (stdout or "")[-1200:]
        step["stderr_tail"] = (stderr or "")[-1200:]
    return step, payload


def wait_for_stage_results(base_url: str, *, session_id: str, max_new_tokens: int, timeout: float) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() <= deadline:
        try:
            ledger = request_json(
                "GET",
                base_url,
                f"/admin/results?status=accepted&workload_type={WORKLOAD_TYPE}&limit=100&session_id={session_id}",
                admin_token=ADMIN_TOKEN,
                timeout=5.0,
            )
            rows = ledger.get("results") if isinstance(ledger.get("results"), list) else []
            stage0 = 0
            stage1 = 0
            final_ready = False
            for row in rows:
                if not isinstance(row, dict):
                    continue
                validation = row.get("validation") if isinstance(row.get("validation"), dict) else {}
                stage_id = int(validation.get("stage_id", -1))
                if stage_id == 0:
                    stage0 += 1
                if stage_id == 1:
                    stage1 += 1
                    if int(validation.get("generated_token_count") or 0) >= max_new_tokens:
                        final_ready = True
            if stage0 >= max_new_tokens and stage1 >= max_new_tokens and final_ready:
                return True, ""
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.5)
    return False, f"stage results not ready: {last_error}"


def stage_kv_rows(state_dir: Path, *, stage_id: int) -> list[dict[str, Any]]:
    path = state_dir / "tasks.jsonl"
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("type") != "task_completed":
            continue
        validation = event.get("validation") if isinstance(event.get("validation"), dict) else {}
        if int(validation.get("stage_id", -1)) != int(stage_id):
            continue
        if int(stage_id) == 0:
            cache_payload = (event.get("activation_results") or [{}])[0]
        else:
            cache_payload = (event.get("inference_results") or [{}])[0]
            if not isinstance(cache_payload, dict) or not cache_payload:
                result = event.get("sharded_inference_result")
                if isinstance(result, dict):
                    cache_payload = result.get("inference_result") or {}
        if not isinstance(cache_payload, dict):
            cache_payload = {}
        rows.append({
            "generation_step": int(validation.get("generation_step", cache_payload.get("generation_step", 0)) or 0),
            "miner_id": str(event.get("miner_id") or ""),
            "stage_id": int(stage_id),
            "kv_cache_schema": cache_payload.get("kv_cache_schema"),
            "kv_cache_stage": cache_payload.get("kv_cache_stage"),
            "kv_cache_ready": bool(cache_payload.get("kv_cache_ready")),
            "kv_cache_hit": bool(cache_payload.get("kv_cache_hit")),
            "kv_cache_tokens_before": int(cache_payload.get("kv_cache_tokens_before") or 0),
            "kv_cache_tokens_after": int(cache_payload.get("kv_cache_tokens_after") or 0),
            "generated_prefix_token_count": int(cache_payload.get("generated_prefix_token_count") or 0),
            "generated_token_count": int(cache_payload.get("generated_token_count") or 0),
            "activation_hash": cache_payload.get("activation_hash"),
            "output_hash": cache_payload.get("output_hash"),
        })
    return sorted(rows, key=lambda item: int(item.get("generation_step") or 0))


def stage0_kv_rows(state_dir: Path) -> list[dict[str, Any]]:
    return stage_kv_rows(state_dir, stage_id=0)


def stage1_kv_rows(state_dir: Path) -> list[dict[str, Any]]:
    return stage_kv_rows(state_dir, stage_id=1)


def kv_cache_summary(
    *,
    schema: str,
    stage: str,
    expected_hits: int,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    hit_rows = [row for row in rows if row.get("kv_cache_hit")]
    ready_rows = [row for row in rows if row.get("kv_cache_ready")]
    return {
        "schema": schema,
        "stage": stage,
        "expected_hit_count": expected_hits,
        "ready_count": len(ready_rows),
        "hit_count": len(hit_rows),
        "rows": [
            {
                key: value
                for key, value in row.items()
                if key not in {"activation_hash", "output_hash"}
            }
            for row in rows
        ],
        "activation_hashes_public": False,
        "output_hashes_public": False,
        "raw_hidden_state_public": False,
        "raw_input_ids_public": False,
    }


def validate_public_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    encoded = json.dumps(report, sort_keys=True)
    for fragment in SECRET_FRAGMENTS:
        if fragment and fragment in encoded:
            errors.append(f"sensitive_fragment:{fragment}")
    for path in public_leak_paths(report):
        if path.endswith(".prompt_hash") or ".safety." in path:
            continue
        errors.append(f"public_leak:{path}")
    return sorted(set(errors))


def degraded_report(args: argparse.Namespace, output_dir: Path, missing: list[str]) -> dict[str, Any]:
    ok = not bool(args.require_hf_runtime)
    codes = ["hf_dependencies_missing", "persistent_kv_cache_hf_runtime_missing"]
    if ok:
        codes.append("persistent_kv_cache_degraded_ready")
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "degraded": True,
        "output_dir": str(output_dir),
        "missing_dependencies": missing,
        "diagnosis_codes": codes,
        "operator_action": "Install optional runtime dependencies with: python -m pip install -e '.[hf]'",
        "safety": {
            "read_only_workload": WORKLOAD_TYPE,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "not_production": True,
            "not_large_model_serving": True,
        },
    }


def run_check(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    base_url = f"http://127.0.0.1:{args.port}"
    state_dir = output_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    serve_cmd = [
        sys.executable,
        "-m",
        "crowdtensor.cli",
        "serve",
        "--profile",
        "gpu-generation" if args.backend == "cuda" else "cpu-real-llm",
        "--bind-host",
        "127.0.0.1",
        "--public-host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--state-dir",
        str(state_dir),
        "--admin-token",
        ADMIN_TOKEN,
        "--miner-token",
        MINER_TOKEN,
        "--observer-token",
        OBSERVER_TOKEN,
        "--hf-model-id",
        args.hf_model_id,
        "--run",
        "--json",
    ]
    if args.hf_cache_dir:
        serve_cmd.extend(["--hf-cache-dir", args.hf_cache_dir])
    generate_cmd = [
        sys.executable,
        "-m",
        "crowdtensor.cli",
        "generate",
        "--coordinator-url",
        base_url,
        "--prompt-text",
        args.prompt_text,
        "--admin-token",
        ADMIN_TOKEN,
        "--backend",
        args.backend,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--timeout-seconds",
        str(args.generate_timeout),
        "--stream",
        "--json",
    ]
    stage0_cmd = [
        sys.executable,
        "-m",
        "crowdtensor.cli",
        "join",
        "--coordinator-url",
        base_url,
        "--miner-id",
        "persistent-kv-stage0",
        "--stage",
        "stage0",
        "--backend",
        args.backend,
        "--miner-token",
        MINER_TOKEN,
        "--hf-model-id",
        args.hf_model_id,
        "--max-tasks",
        str(args.max_new_tokens),
        "--idle-sleep",
        str(args.idle_sleep),
        "--run",
        "--json",
    ]
    stage1_cmd = [
        sys.executable,
        "-m",
        "crowdtensor.cli",
        "join",
        "--coordinator-url",
        base_url,
        "--miner-id",
        "persistent-kv-stage1",
        "--stage",
        "stage1",
        "--backend",
        args.backend,
        "--miner-token",
        MINER_TOKEN,
        "--hf-model-id",
        args.hf_model_id,
        "--max-tasks",
        str(args.max_new_tokens),
        "--idle-sleep",
        str(args.idle_sleep),
        "--run",
        "--json",
    ]
    if args.hf_cache_dir:
        generate_cmd.extend(["--hf-cache-dir", args.hf_cache_dir])
        stage0_cmd.extend(["--hf-cache-dir", args.hf_cache_dir])
        stage1_cmd.extend(["--hf-cache-dir", args.hf_cache_dir])

    serve_proc: subprocess.Popen[str] | None = None
    generate_proc: subprocess.Popen[str] | None = None
    stage0_proc: subprocess.Popen[str] | None = None
    stage1_proc: subprocess.Popen[str] | None = None
    steps: list[dict[str, Any]] = []
    payloads: dict[str, Any] = {}
    try:
        serve_proc = popen_command(serve_cmd)
        healthy, health_error = wait_health(base_url, serve_proc, args.startup_timeout)
        steps.append({"name": "serve", "ok": healthy, "error": health_error})
        if not healthy:
            return {
                "schema": SCHEMA,
                "generated_at": utc_now(),
                "ok": False,
                "output_dir": str(output_dir),
                "steps": steps,
                "diagnosis_codes": ["serve_start_failed"],
            }

        generate_proc = popen_command(generate_cmd)
        deadline = time.monotonic() + args.session_queue_timeout
        session_id = ""
        while time.monotonic() <= deadline:
            try:
                state = request_json("GET", base_url, "/state", observer_token=OBSERVER_TOKEN, timeout=2.0)
                tasks = state.get("tasks") if isinstance(state.get("tasks"), list) else []
                for task in tasks:
                    if not isinstance(task, dict):
                        continue
                    metadata = task.get("workload_metadata") if isinstance(task.get("workload_metadata"), dict) else {}
                    if task.get("workload_type") == WORKLOAD_TYPE and int(metadata.get("stage_id", -1)) == 0:
                        session_id = str(metadata.get("session_id") or "")
                        break
                if session_id:
                    break
            except Exception:
                pass
            time.sleep(0.2)
        steps.append({"name": "generate_session_created", "ok": bool(session_id), "session_id": session_id})
        if not session_id:
            step, payload = finish_process("generate", generate_proc, timeout=1.0)
            steps.append(step)
            payloads["generate"] = payload
            return {
                "schema": SCHEMA,
                "generated_at": utc_now(),
                "ok": False,
                "output_dir": str(output_dir),
                "steps": steps,
                "diagnosis_codes": ["session_create_failed"],
            }

        stage0_proc = popen_command(stage0_cmd)
        stage1_proc = popen_command(stage1_cmd)
        ready, ready_error = wait_for_stage_results(
            base_url,
            session_id=session_id,
            max_new_tokens=args.max_new_tokens,
            timeout=args.generate_timeout,
        )
        steps.append({"name": "stage_results_ready", "ok": ready, "error": ready_error})
        stage0_step, stage0_payload = finish_process("persistent_stage0", stage0_proc, timeout=args.miner_finish_timeout)
        stage1_step, stage1_payload = finish_process("persistent_stage1", stage1_proc, timeout=args.miner_finish_timeout)
        steps.extend([stage0_step, stage1_step])
        payloads["stage0"] = stage0_payload
        payloads["stage1"] = stage1_payload
        generate_step, generate_payload = finish_process("generate", generate_proc, timeout=args.generate_timeout + 10.0)
        steps.append(generate_step)
        payloads["generate"] = generate_payload

        stage0_rows = stage0_kv_rows(state_dir)
        stage1_rows = stage1_kv_rows(state_dir)
        stage0_hit_rows = [row for row in stage0_rows if row.get("kv_cache_hit")]
        stage0_ready_rows = [row for row in stage0_rows if row.get("kv_cache_ready")]
        stage1_hit_rows = [row for row in stage1_rows if row.get("kv_cache_hit")]
        stage1_ready_rows = [row for row in stage1_rows if row.get("kv_cache_ready")]
        expected_hits = max(0, int(args.max_new_tokens) - 1)
        generation = generate_payload.get("generation") if isinstance(generate_payload.get("generation"), dict) else {}
        ok = bool(
            ready
            and all(bool(step.get("ok")) for step in steps)
            and int(generation.get("generated_token_count") or 0) >= args.max_new_tokens
            and len(stage0_rows) >= args.max_new_tokens
            and len(stage0_ready_rows) >= args.max_new_tokens
            and len(stage0_hit_rows) >= expected_hits
            and len(stage1_rows) >= args.max_new_tokens
            and len(stage1_ready_rows) >= args.max_new_tokens
            and len(stage1_hit_rows) >= expected_hits
        )
        codes = {
            "persistent_real_llm_kv_cache_ready" if ok else "persistent_real_llm_kv_cache_blocked",
            "persistent_real_llm_dual_stage_kv_cache_ready" if ok else "persistent_real_llm_dual_stage_kv_cache_blocked",
            "persistent_stage_miners_ready" if stage0_step.get("ok") and stage1_step.get("ok") else "persistent_stage_miners_incomplete",
            "real_llm_stage0_kv_cache_v1_ready" if stage0_ready_rows else "real_llm_stage0_kv_cache_missing",
            "real_llm_stage1_kv_cache_v1_ready" if stage1_ready_rows else "real_llm_stage1_kv_cache_missing",
            "stage0_kv_cache_hits_ready" if len(stage0_hit_rows) >= expected_hits else "stage0_kv_cache_hits_missing",
            "stage1_kv_cache_hits_ready" if len(stage1_hit_rows) >= expected_hits else "stage1_kv_cache_hits_missing",
        }
        if generation.get("multi_token_generation_ready"):
            codes.add("multi_token_generation_ready")
        if generate_payload.get("stream", {}).get("event_count"):
            codes.add("generate_stream_events_ready")
        report = {
            "schema": SCHEMA,
            "generated_at": utc_now(),
            "ok": ok,
            "output_dir": str(output_dir),
            "backend": args.backend,
            "hf_model_id": args.hf_model_id,
            "max_new_tokens": args.max_new_tokens,
            "session": {
                "session_id": session_id,
                "workload_type": WORKLOAD_TYPE,
            },
            "generation": generation,
            "kv_cache": {
                "schema": "real_llm_dual_stage_kv_cache_summary_v1",
                "expected_hit_count_per_stage": expected_hits,
                "stage0": kv_cache_summary(
                    schema="real_llm_stage0_kv_cache_v1",
                    stage="stage0_prefix",
                    expected_hits=expected_hits,
                    rows=stage0_rows,
                ),
                "stage1": kv_cache_summary(
                    schema="real_llm_stage1_kv_cache_v1",
                    stage="stage1_suffix",
                    expected_hits=expected_hits,
                    rows=stage1_rows,
                ),
            },
            "runtime_resources": {
                "backend": args.backend,
                "cuda": cuda_runtime_summary(),
            },
            "steps": steps,
            "processes": {
                "serve": {},
                "stage0": stage0_payload,
                "stage1": stage1_payload,
            },
            "diagnosis_codes": sorted(codes),
            "safety": {
                "read_only_workload": WORKLOAD_TYPE,
                "raw_prompt_public": False,
                "raw_generated_text_public": False,
                "generated_token_ids_public": False,
                "activation_hashes_public": False,
                "not_production": True,
                "not_large_model_serving": True,
                "coordinator_backed_task_execution": True,
                "persistence_scope": "single_miner_process",
            },
        }
        errors = validate_public_report(report)
        if errors:
            report["ok"] = False
            report["diagnosis_codes"] = sorted(set(report["diagnosis_codes"]) | {"public_report_safety_failed"})
            report["safety_errors"] = errors
        return report
    finally:
        processes = {
            "stage0_process": stop_process(stage0_proc),
            "stage1_process": stop_process(stage1_proc),
            "generate_process": stop_process(generate_proc),
            "serve_process": stop_process(serve_proc),
        }
        if output_dir.exists():
            write_json(output_dir / "process_cleanup.json", processes)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate persistent real LLM dual-stage KV-cache reuse.")
    parser.add_argument("--output-dir", default="dist/persistent-real-llm-kv-cache-check")
    parser.add_argument("--port", type=int, default=8797)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--hf-model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--prompt-text", default="CrowdTensor routes home CPU")
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--session-queue-timeout", type=float, default=120.0)
    parser.add_argument("--generate-timeout", type=float, default=360.0)
    parser.add_argument("--miner-finish-timeout", type=float, default=60.0)
    parser.add_argument("--idle-sleep", type=float, default=0.2)
    parser.add_argument("--require-hf-runtime", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 2:
        raise SystemExit("--max-new-tokens must be at least 2 to prove cache hits")
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = missing_hf_dependencies()
    if missing:
        report = degraded_report(args, output_dir, missing)
    else:
        report = run_check(args, output_dir)
    write_json(output_dir / "persistent_real_llm_kv_cache_check.json", report)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"Persistent real LLM KV cache check ok={report.get('ok')}")
        print("diagnosis=" + ",".join(report.get("diagnosis_codes") or []))
        print(f"report={output_dir / 'persistent_real_llm_kv_cache_check.json'}")
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
