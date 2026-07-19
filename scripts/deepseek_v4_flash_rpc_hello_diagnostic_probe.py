#!/usr/bin/env python3
"""Run a bounded Kaggle llama.cpp RPC HELLO diagnostic for DeepSeek V4 work."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import deepseek_v4_flash_quantized_same_request_probe as same


SCHEMA = "deepseek_v4_flash_rpc_hello_diagnostic_probe_v1"
WORKER_SCHEMA = "deepseek_v4_flash_rpc_hello_diagnostic_worker_v1"
DEFAULT_OUTPUT_DIR = "dist/deepseek-v4-flash-rpc-hello-diagnostic-probe"
DEFAULT_ACCELERATOR = "NvidiaTeslaT4"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def render_kernel(args: argparse.Namespace, runtime_tarball_url: str) -> str:
    include_cpu_rpc = "True" if args.include_cpu_rpc else "False"
    return f'''from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import struct
import subprocess
import tarfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = {json.dumps(WORKER_SCHEMA)}
RUNTIME_TARBALL_URL = {json.dumps(runtime_tarball_url)}
RUNTIME_TARBALL_EXPECTED_SHA256 = {json.dumps(args.runtime_tarball_sha256)}
OUT = Path("/kaggle/working")
TEMP = Path("/kaggle/temp/ct_dsv4_rpc_hello_diag")
RUNTIME = TEMP / "runtime"
REPORT_PATH = OUT / "deepseek_v4_flash_rpc_hello_diagnostic_worker.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha_text(value):
    return "sha256:" + hashlib.sha256(str(value or "").encode("utf-8", errors="replace")).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def safe_tail(value, limit=1800):
    text = str(value or "")[-limit:]
    for fragment in ["KAGGLE_KEY", "KAGGLE_USERNAME", "KAGGLE_API_TOKEN", "HF_TOKEN", "Bearer ", "Authorization:", "Cookie:"]:
        text = text.replace(fragment, "<redacted>")
    return text


def write_report(stage, report):
    report["stage"] = stage
    report["updated_at"] = utc_now()
    tmp = REPORT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    tmp.replace(REPORT_PATH)


def run(command, *, timeout=120, env=None):
    started = time.monotonic()
    try:
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {{"ok": False, "error": "timeout", "duration_seconds": round(time.monotonic() - started, 3), "command_public": [str(item) for item in command]}}, "", ""
    return {{
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "stdout_digest": sha_text(completed.stdout),
        "stdout_chars": len(completed.stdout or ""),
        "stderr_tail": safe_tail(completed.stderr),
        "stderr_digest": sha_text(completed.stderr),
        "stderr_chars": len(completed.stderr or ""),
        "command_public": [str(item) for item in command],
    }}, completed.stdout or "", completed.stderr or ""


def find_binary(root, names):
    for name in names:
        for path in root.rglob(name):
            if path.is_file():
                path.chmod(path.stat().st_mode | 0o111)
                return path
    return None


def env_for_binary(binary, *, cuda_visible=""):
    env = os.environ.copy()
    dirs = [str(binary.parent)]
    sibling = binary.parent.parent / "lib"
    if sibling.is_dir():
        dirs.append(str(sibling))
    current = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(dirs + ([current] if current else []))
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible
    env.setdefault("CUDA_CACHE_DISABLE", "1")
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")
    return env


def rpc_hello_probe(host, port, *, timeout=10):
    conn_caps_size = 24
    response_size = 4 + conn_caps_size
    started = time.monotonic()
    try:
        with socket.create_connection((str(host), int(port)), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(struct.pack("<BQ", 14, conn_caps_size) + (b"\\0" * conn_caps_size))
            header = sock.recv(8)
            if len(header) != 8:
                return {{"ok": False, "error": "rpc_hello_response_header_short", "response_header_bytes": len(header), "duration_seconds": round(time.monotonic() - started, 3), "endpoint_public": False}}
            size = struct.unpack("<Q", header)[0]
            body = b""
            while len(body) < size:
                chunk = sock.recv(size - len(body))
                if not chunk:
                    break
                body += chunk
            if size != response_size or len(body) != size:
                return {{"ok": False, "error": "rpc_hello_response_size_mismatch", "response_size": int(size), "response_body_bytes": len(body), "duration_seconds": round(time.monotonic() - started, 3), "endpoint_public": False}}
            major, minor, patch, _padding = struct.unpack("<BBBB", body[:4])
            caps = body[4:]
            return {{"ok": major == 4 and minor <= 0, "major": int(major), "minor": int(minor), "patch": int(patch), "conn_caps_nonzero": any(caps), "conn_caps_size": len(caps), "duration_seconds": round(time.monotonic() - started, 3), "endpoint_public": False}}
    except Exception as exc:
        return {{"ok": False, "error_type": type(exc).__name__, "error_digest": sha_text(str(exc)), "duration_seconds": round(time.monotonic() - started, 3), "endpoint_public": False}}


def probe_hardware():
    step, stdout, _ = run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,compute_cap",
        "--format=csv,noheader,nounits",
    ], timeout=60)
    devices = []
    if step.get("ok"):
        for line in stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 5:
                try:
                    devices.append({{
                        "index": int(parts[0]),
                        "name_hash": sha_text(parts[1])[:24],
                        "name_public": False,
                        "memory_total_mb": int(float(parts[2])),
                        "memory_free_mb": int(float(parts[3])),
                        "compute_cap": parts[4],
                    }})
                except ValueError:
                    pass
    return {{"step": step, "gpu_count": len(devices), "devices": devices, "kaggle_cuda_verified": bool(devices)}}


def prepare_runtime():
    archive = TEMP / "runtime.tar.gz"
    extract = TEMP / "runtime-extract"
    started = time.monotonic()
    with urllib.request.urlopen(RUNTIME_TARBALL_URL, timeout=900) as response:
        with archive.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    digest = sha_file(archive)
    if RUNTIME_TARBALL_EXPECTED_SHA256 and digest != RUNTIME_TARBALL_EXPECTED_SHA256:
        return {{"ok": False, "error": "runtime_tarball_sha256_mismatch", "tarball_sha256": digest}}
    extract.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(extract)
    rpc = find_binary(extract, ["rpc-server", "llama-rpc-server"])
    if not rpc:
        return {{"ok": False, "error": "runtime_rpc_server_missing", "tarball_sha256": digest}}
    RUNTIME.mkdir(parents=True, exist_ok=True)
    for item in rpc.parent.iterdir():
        if item.is_file():
            target = RUNTIME / item.name
            shutil.copy2(item, target)
            if item.name == rpc.name:
                target.chmod(target.stat().st_mode | 0o111)
    return {{
        "ok": True,
        "duration_seconds": round(time.monotonic() - started, 3),
        "tarball_size_bytes": int(archive.stat().st_size),
        "tarball_sha256": digest,
        "runtime_dir_public": False,
    }}


def start_rpc(name, rpc, port, *, cuda_visible="", device_arg=""):
    stdout_log = OUT / (name + ".stdout.log")
    stderr_log = OUT / (name + ".stderr.log")
    stdout_handle = stdout_log.open("w", encoding="utf-8", errors="replace")
    stderr_handle = stderr_log.open("w", encoding="utf-8", errors="replace")
    command = [str(rpc), "-H", "127.0.0.1", "-p", str(port)]
    if device_arg:
        command.extend(["-d", device_arg])
    proc = subprocess.Popen(command, stdout=stdout_handle, stderr=stderr_handle, text=True, env=env_for_binary(rpc, cuda_visible=cuda_visible))
    return {{"name": name, "endpoint": "127.0.0.1:" + str(port), "command_public": command, "pid": proc.pid, "process": proc, "stdout_log": stdout_log, "stderr_log": stderr_log}}


def log_summary(server):
    values = {{}}
    for key in ["stdout_log", "stderr_log"]:
        path = server.get(key)
        if path and Path(path).is_file():
            text = Path(path).read_text(encoding="utf-8", errors="replace")
            values[str(key) + "_digest"] = sha_text(text)
            values[str(key) + "_tail"] = safe_tail(text)
    return values


report = {{
    "schema": SCHEMA,
    "ok": False,
    "started_at": utc_now(),
    "updated_at": utc_now(),
    "blockers": [],
    "diagnosis_codes": [],
    "public_artifact_safe": True,
    "runtime_tarball_url_public": False,
}}

processes = []
try:
    TEMP.mkdir(parents=True, exist_ok=True)
    report["hardware"] = probe_hardware()
    write_report("hardware_probe_complete", report)
    runtime = prepare_runtime()
    report["runtime"] = runtime
    write_report("runtime_ready" if runtime.get("ok") else "runtime_blocked", report)
    if not runtime.get("ok"):
        report["blockers"].append("runtime_tarball_prepare_failed")
    rpc = find_binary(RUNTIME, ["rpc-server", "llama-rpc-server"])
    report["rpc_server_present"] = bool(rpc)
    if not rpc:
        report["blockers"].append("rpc_server_missing")
    servers = []
    if rpc and not report["blockers"]:
        gpu_count = int(report["hardware"].get("gpu_count") or 0)
        if gpu_count >= 1:
            servers.append(start_rpc("kaggle-cuda0-rpc", rpc, 50152, cuda_visible="0,1", device_arg="CUDA0"))
        if gpu_count >= 2:
            servers.append(start_rpc("kaggle-cuda1-rpc", rpc, 50153, cuda_visible="0,1", device_arg="CUDA1"))
        if {include_cpu_rpc}:
            servers.append(start_rpc("kaggle-cpu-rpc", rpc, 50154, cuda_visible="", device_arg=""))
        processes = [server["process"] for server in servers]
        time.sleep(8)
        summaries = []
        for server in servers:
            alive = server["process"].poll() is None
            host, port = server["endpoint"].rsplit(":", 1)
            hello = rpc_hello_probe(host, port, timeout=10) if alive else {{"ok": False, "skipped": True}}
            summaries.append({{
                "name": server["name"],
                "endpoint_public": False,
                "alive": alive,
                "rpc_hello": hello,
                "log_summary": log_summary(server),
            }})
        report["servers"] = summaries
        if not summaries:
            report["blockers"].append("rpc_server_not_started")
        if any(not item.get("alive") for item in summaries):
            report["blockers"].append("rpc_server_process_not_alive")
        if any(item.get("alive") and not item.get("rpc_hello", {{}}).get("ok") for item in summaries):
            report["blockers"].append("rpc_hello_failed")
    report["rpc_hello_diagnostic_ready"] = bool(report.get("servers") and not report["blockers"])
    report["ok"] = report["rpc_hello_diagnostic_ready"]
    report["diagnosis_codes"].append("deepseek_v4_flash_rpc_hello_diagnostic_ready" if report["ok"] else "deepseek_v4_flash_rpc_hello_diagnostic_not_ready")
    write_report("complete" if report["ok"] else "blocked", report)
except Exception as exc:
    report["ok"] = False
    report["error_type"] = type(exc).__name__
    report["error_digest"] = sha_text(str(exc))
    report["blockers"].append("rpc_hello_diagnostic_exception")
    write_report("exception", report)
finally:
    for proc in processes:
        try:
            proc.terminate()
        except Exception:
            pass
    for proc in processes:
        try:
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    try:
        if TEMP.exists():
            shutil.rmtree(TEMP)
            report["temp_cleanup"] = {{"ok": True, "path_public": False}}
    except Exception as exc:
        report["temp_cleanup"] = {{"ok": False, "error_type": type(exc).__name__, "error_digest": sha_text(str(exc))}}
    write_report("final", report)
    print(json.dumps({{"schema": SCHEMA, "ok": report.get("ok"), "server_count": len(report.get("servers", [])), "blockers": report.get("blockers", [])}}, sort_keys=True), flush=True)
'''


def build_package(args: argparse.Namespace, *, output_dir: Path, runtime_tarball_url: str) -> dict[str, Any]:
    owner = args.kaggle_owner or same.default_kaggle_owner()
    if not owner:
        raise SystemExit("--kaggle-owner or ~/.kaggle/kaggle.json username is required")
    slug = same.safe_slug(args.kernel_slug_prefix)[:34] + "-" + str(int(time.time()))[-8:]
    slug = slug[:45].strip("-")
    kernel_dir = output_dir / "private-kaggle-kernel"
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text(render_kernel(args, runtime_tarball_url), encoding="utf-8")
    metadata = {
        "id": f"{owner}/{slug}",
        "title": f"CT DSV4 RPC Hello {slug[-8:]}",
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": args.accelerator,
    }
    write_json(kernel_dir / "kernel-metadata.json", metadata)
    return {"kernel_dir": kernel_dir, "declared_kernel_ref": metadata["id"], "kernel_ref": metadata["id"], "kernel_slug": slug, "metadata": metadata}


def summarize_worker(worker: dict[str, Any]) -> dict[str, Any]:
    servers = [item for item in worker.get("servers", []) if isinstance(item, dict)]
    return {
        "present": bool(worker),
        "schema": str(worker.get("schema") or ""),
        "ok": worker.get("ok") is True,
        "rpc_hello_diagnostic_ready": worker.get("rpc_hello_diagnostic_ready") is True,
        "server_count": len(servers),
        "server_names": [str(item.get("name") or "") for item in servers],
        "all_servers_alive": bool(servers) and all(item.get("alive") is True for item in servers),
        "all_rpc_hello_ok": bool(servers) and all((item.get("rpc_hello") or {}).get("ok") is True for item in servers),
        "blockers": [str(item) for item in worker.get("blockers", [])] if isinstance(worker.get("blockers"), list) else [],
        "diagnosis_codes": [str(item) for item in worker.get("diagnosis_codes", [])] if isinstance(worker.get("diagnosis_codes"), list) else [],
        "public_artifact_safe": worker.get("public_artifact_safe") is True,
    }


def build_report(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    package: dict[str, Any],
    steps: list[dict[str, Any]],
    worker_report: dict[str, Any],
) -> dict[str, Any]:
    worker = summarize_worker(worker_report)
    pushed = any(step.get("name") == "kaggle_kernel_push" and step.get("ok") for step in steps)
    output_downloaded = bool(worker_report)
    kernel_deleted = any(step.get("name") == "kaggle_kernel_delete" and step.get("ok") for step in steps)
    private_removed = not (output_dir / "private-kaggle-kernel").exists()
    ok = bool(pushed and output_downloaded and worker.get("rpc_hello_diagnostic_ready") and kernel_deleted and private_removed)
    blockers = set(worker.get("blockers") or [])
    if not pushed:
        blockers.add("kaggle_rpc_hello_kernel_push_failed")
    if pushed and not output_downloaded:
        blockers.add("kaggle_rpc_hello_worker_report_missing")
    if not kernel_deleted and not args.skip_kaggle_cleanup:
        blockers.add("kaggle_rpc_hello_kernel_cleanup_missing")
    if not private_removed:
        blockers.add("kaggle_rpc_hello_private_package_not_removed")
    return {
        "schema": SCHEMA,
        "generated_at": same.utc_now(),
        "ok": ok,
        "rpc_hello_diagnostic_ready": bool(ok and worker.get("rpc_hello_diagnostic_ready")),
        "fresh_kaggle_run_performed": pushed,
        "worker_summary": worker,
        "runtime": {
            "runtime_tarball_requested": bool(args.runtime_tarball_path or args.runtime_tarball_url),
            "runtime_tarball_sha256": args.runtime_tarball_sha256,
            "runtime_tarball_path_public": False,
            "runtime_tarball_url_public": False,
        },
        "kaggle_lifecycle": {
            "kernel_ref": package.get("kernel_ref"),
            "kernel_ref_public": False,
            "kernel_slug": package.get("kernel_slug"),
            "requested_accelerator": args.accelerator,
            "cleanup_attempted": any(step.get("name") == "kaggle_kernel_delete" for step in steps),
            "kernel_deleted": kernel_deleted,
            "private_package_removed": private_removed,
        },
        "steps": steps,
        "blockers": sorted(blockers),
        "diagnosis_codes": [
            "deepseek_v4_flash_rpc_hello_diagnostic_ready" if ok else "deepseek_v4_flash_rpc_hello_diagnostic_not_ready",
        ],
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "private_runtime_state_public": False,
            "private_kaggle_payload_public": False,
            "raw_logs_public": False,
        },
        "public_artifact_safe": True,
    }


def run_live_probe(args: argparse.Namespace, *, runner=same.subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps: list[dict[str, Any]] = []
    worker_report: dict[str, Any] = {}
    package: dict[str, Any] = {}
    original_url = args.runtime_tarball_url
    try:
        with same.maybe_runtime_tarball_server(args) as server:
            runtime_url = str(server.get("url") or "")
            args.runtime_tarball_url = runtime_url or args.runtime_tarball_url
            package = build_package(args, output_dir=output_dir, runtime_tarball_url=args.runtime_tarball_url)
            push_command = ["kaggle", "kernels", "push", "-p", str(package["kernel_dir"]), "-t", str(args.kernel_timeout_seconds)]
            if args.accelerator:
                push_command.extend(["--accelerator", args.accelerator])
            print(f"[{same.utc_now()}] pushing DeepSeek V4 RPC HELLO diagnostic Kaggle kernel {package['declared_kernel_ref']}", flush=True)
            push_step = same.run_step("kaggle_kernel_push", push_command, runner=runner, timeout_seconds=args.kaggle_push_timeout_seconds)
            steps.append(push_step)
            if push_step.get("ok"):
                kernel_ref, resolve_step = same.resolve_pushed_kernel_ref(package, push_step, runner=runner, timeout_seconds=args.kaggle_push_timeout_seconds)
                if resolve_step:
                    steps.append(resolve_step)
                package["kernel_ref"] = kernel_ref
                print(f"[{same.utc_now()}] waiting for DeepSeek V4 RPC HELLO diagnostic kernel {kernel_ref}", flush=True)
                status_step = same.wait_kaggle_terminal(
                    kernel_ref,
                    runner=runner,
                    timeout_seconds=args.kaggle_status_timeout_seconds,
                    poll_interval=args.kaggle_status_poll_interval,
                )
                steps.append(status_step)
                output_path = output_dir / "kaggle-output"
                output_step = same.run_step(
                    "kaggle_kernel_output",
                    [
                        "kaggle",
                        "kernels",
                        "output",
                        kernel_ref,
                        "-p",
                        str(output_path),
                        "--force",
                        "--file-pattern",
                        "deepseek_v4_flash_rpc_hello_diagnostic_worker.json",
                    ],
                    runner=runner,
                    timeout_seconds=args.kaggle_output_timeout_seconds,
                )
                steps.append(output_step)
                worker_report = load_json(output_path / "deepseek_v4_flash_rpc_hello_diagnostic_worker.json")
                if not args.skip_kaggle_cleanup:
                    print(f"[{same.utc_now()}] deleting DeepSeek V4 RPC HELLO diagnostic Kaggle kernel {kernel_ref}", flush=True)
                    steps.append(same.run_step(
                        "kaggle_kernel_delete",
                        ["kaggle", "kernels", "delete", kernel_ref, "-y"],
                        runner=runner,
                        timeout_seconds=args.kaggle_delete_timeout_seconds,
                    ))
    finally:
        args.runtime_tarball_url = original_url
        if not args.keep_private_package:
            shutil.rmtree(Path(args.output_dir) / "private-kaggle-kernel", ignore_errors=True)
    report = build_report(args, output_dir=output_dir, package=package, steps=steps, worker_report=worker_report)
    write_json(output_dir / "deepseek_v4_flash_rpc_hello_diagnostic_probe.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kaggle-owner", default=same.default_kaggle_owner())
    parser.add_argument("--kernel-slug-prefix", default="ct-dsv4-rpc-hello")
    parser.add_argument("--accelerator", default=DEFAULT_ACCELERATOR)
    parser.add_argument("--runtime-tarball-path", default="")
    parser.add_argument("--runtime-tarball-url", default="")
    parser.add_argument("--runtime-tarball-sha256", default="")
    parser.add_argument("--bore-url", default=same.DEFAULT_BORE_URL)
    parser.add_argument("--bore-server", default=same.DEFAULT_BORE_SERVER)
    parser.add_argument("--include-cpu-rpc", action="store_true")
    parser.add_argument("--kernel-timeout-seconds", type=int, default=1800)
    parser.add_argument("--kaggle-push-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--kaggle-status-timeout-seconds", type=float, default=2100.0)
    parser.add_argument("--kaggle-status-poll-interval", type=float, default=30.0)
    parser.add_argument("--kaggle-output-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--kaggle-delete-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--skip-kaggle-cleanup", action="store_true")
    parser.add_argument("--keep-private-package", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.runtime_tarball_path and args.runtime_tarball_url:
        raise SystemExit("--runtime-tarball-path and --runtime-tarball-url are mutually exclusive")
    if not args.runtime_tarball_path and not args.runtime_tarball_url:
        raise SystemExit("--runtime-tarball-path or --runtime-tarball-url is required")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_live_probe(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"RPC HELLO diagnostic ready: {report.get('rpc_hello_diagnostic_ready')}")
        print(f"Report: {Path(args.output_dir) / 'deepseek_v4_flash_rpc_hello_diagnostic_probe.json'}")
    return 0 if report.get("public_artifact_safe") else 1


if __name__ == "__main__":
    raise SystemExit(main())
