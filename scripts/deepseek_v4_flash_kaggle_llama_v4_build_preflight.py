#!/usr/bin/env python3
"""Build-check DeepSeek V4-aware llama.cpp CUDA/RPC runtime on Kaggle."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SCHEMA = "deepseek_v4_flash_kaggle_llama_v4_build_preflight_v1"
WORKER_SCHEMA = "deepseek_v4_flash_kaggle_llama_v4_build_worker_v1"
DEFAULT_OUTPUT_DIR = "dist/deepseek-v4-flash-kaggle-llama-v4-build-preflight"
DEFAULT_ACCELERATOR = "NvidiaTeslaT4"
DEFAULT_REPO_URL = "https://github.com/cchuter/llama.cpp.git"
DEFAULT_BRANCH = "feat/v4-port-cuda"
KAGGLE_CODE_URL = re.compile(r"https://www\.kaggle\.com/code/([^/\s]+)/([^/\s]+)")
KAGGLE_TABLE_SPLIT = re.compile(r"\s{2,}")
Runner = Callable[..., subprocess.CompletedProcess[str]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def safe_slug(value: str, *, default: str = "ct-dsv4-llama") -> str:
    cleaned: list[str] = []
    last_dash = False
    for char in str(value or "").lower():
        if char.isalnum():
            cleaned.append(char)
            last_dash = False
        elif not last_dash:
            cleaned.append("-")
            last_dash = True
    return "".join(cleaned).strip("-") or default


def default_kaggle_owner() -> str:
    if os.environ.get("KAGGLE_USERNAME"):
        return str(os.environ["KAGGLE_USERNAME"])
    config = Path.home() / ".kaggle" / "kaggle.json"
    if config.is_file():
        try:
            loaded = json.loads(config.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return ""
        if isinstance(loaded, dict):
            return str(loaded.get("username") or "")
    return ""


def shell_command(command: list[Any]) -> str:
    return shlex.join([str(part) for part in command])


def safe_tail(text: str, limit: int = 2000) -> str:
    redacted = str(text or "")[-limit:]
    for fragment in [
        "KAGGLE_KEY",
        "KAGGLE_USERNAME",
        "KAGGLE_API_TOKEN",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "Bearer ",
        "Authorization:",
        "Cookie:",
    ]:
        redacted = redacted.replace(fragment, "<redacted>")
    return redacted


def run_step(
    name: str,
    command: list[str],
    *,
    runner: Runner,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = runner(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "ok": False,
            "error": "timeout",
            "duration_seconds": round(time.monotonic() - started, 3),
            "command_line": shell_command(command),
        }
    output = f"{completed.stdout or ''}\n{completed.stderr or ''}"
    return {
        "name": name,
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "stdout_tail": safe_tail(completed.stdout or ""),
        "stderr_tail": safe_tail(completed.stderr or ""),
        "command_line": shell_command(command),
        "actual_kernel_ref": extract_kernel_ref(output),
    }


def extract_kernel_ref(text: str) -> str:
    match = KAGGLE_CODE_URL.search(text or "")
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return ""


def extract_status(text: str) -> str:
    upper = str(text or "").upper()
    for status in ["COMPLETE", "RUNNING", "QUEUED", "PENDING", "ERROR", "FAILED", "CANCELLED", "CANCELED"]:
        if status in upper:
            return status
    return "UNKNOWN"


def extract_kernel_list_ref(text: str, *, owner: str, title: str) -> str:
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("Warning:") or line.startswith("ref "):
            continue
        if set(line) <= {"-", " "}:
            continue
        parts = KAGGLE_TABLE_SPLIT.split(line)
        if len(parts) < 2:
            continue
        ref, row_title = parts[0], parts[1]
        if "/" not in ref:
            continue
        if owner and not ref.startswith(f"{owner}/"):
            continue
        if row_title == title:
            return ref
    return ""


def wait_kaggle_terminal(
    kernel_ref: str,
    *,
    runner: Runner,
    timeout_seconds: float,
    poll_interval: float,
) -> dict[str, Any]:
    started = time.monotonic()
    attempts = 0
    last_step: dict[str, Any] = {}
    while time.monotonic() - started <= timeout_seconds:
        attempts += 1
        last_step = run_step(
            "kaggle_kernel_status",
            ["kaggle", "kernels", "status", kernel_ref],
            runner=runner,
            timeout_seconds=60,
        )
        status = extract_status(f"{last_step.get('stdout_tail') or ''}\n{last_step.get('stderr_tail') or ''}")
        print(f"[{utc_now()}] DeepSeek V4 Kaggle llama build status attempt={attempts} status={status}", flush=True)
        if status in {"COMPLETE", "ERROR", "FAILED", "CANCELLED", "CANCELED"}:
            last_step.update({
                "duration_seconds": round(time.monotonic() - started, 3),
                "attempts": attempts,
                "status": status,
                "terminal": True,
                "kernel_ref": kernel_ref,
                "ok": bool(last_step.get("ok") and status == "COMPLETE"),
            })
            return last_step
        time.sleep(max(5.0, float(poll_interval)))
    last_step.update({
        "duration_seconds": round(time.monotonic() - started, 3),
        "attempts": attempts,
        "status": extract_status(f"{last_step.get('stdout_tail') or ''}\n{last_step.get('stderr_tail') or ''}"),
        "terminal": False,
        "kernel_ref": kernel_ref,
        "ok": False,
        "error": "timeout_waiting_for_terminal_kernel_status",
    })
    return last_step


def render_kernel(args: argparse.Namespace) -> str:
    repo_url = json.dumps(args.repo_url)
    branch = json.dumps(args.branch)
    cuda_architectures = json.dumps(args.cuda_architectures)
    patch_rpc_op_count_guard = "True" if args.patch_rpc_op_count_guard else "False"
    export_runtime_tarball = "True" if args.export_runtime_tarball else "False"
    return f'''from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = {json.dumps(WORKER_SCHEMA)}
REPO_URL = {repo_url}
BRANCH = {branch}
CUDA_ARCHITECTURES = {cuda_architectures}
CUDA_BUILD_JOBS = {int(args.cuda_build_jobs)}
CUDA_BUILD_TIMEOUT_SECONDS = {int(args.cuda_build_timeout_seconds)}
PATCH_RPC_OP_COUNT_GUARD = {patch_rpc_op_count_guard}
EXPORT_RUNTIME_TARBALL = {export_runtime_tarball}
OUT = Path("/kaggle/working")
ROOT = Path("/kaggle/temp/ct_dsv4_llama_v4_build")
SRC = ROOT / "llama.cpp"
BUILD = ROOT / "build"
REPORT_PATH = OUT / "deepseek_v4_flash_kaggle_llama_v4_build_worker.json"
RUNTIME_DIR = OUT / "deepseek-v4-flash-llama-v4-runtime"
RUNTIME_TARBALL = OUT / "deepseek-v4-flash-llama-v4-runtime.tar.gz"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha_text(value):
    return "sha256:" + hashlib.sha256(str(value or "").encode("utf-8", errors="replace")).hexdigest()


def safe_tail(value, limit=2000):
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


def disk_snapshot(path="/kaggle"):
    try:
        usage = shutil.disk_usage(str(path))
        return {{"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}}
    except Exception as exc:
        return {{"error_type": type(exc).__name__, "error_digest": sha_text(str(exc))}}


def run(command, *, timeout=1200, cwd=None, env=None):
    started = time.monotonic()
    try:
        completed = subprocess.run(command, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
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


def env_for_binary(binary):
    env = os.environ.copy()
    dirs = [str(binary.parent)]
    sibling = binary.parent.parent / "lib"
    if sibling.is_dir():
        dirs.append(str(sibling))
    current = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(dirs + ([current] if current else []))
    env.setdefault("CUDA_CACHE_DISABLE", "1")
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")
    return env


def probe_hardware():
    step, stdout, _ = run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,compute_cap",
        "--format=csv,noheader,nounits",
    ], timeout=60)
    devices = []
    if step.get("ok"):
        for line in stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 6:
                try:
                    devices.append({{
                        "index": int(parts[0]),
                        "name_hash": sha_text(parts[1])[:24],
                        "name_public": False,
                        "memory_total_mb": int(float(parts[2])),
                        "memory_used_mb": int(float(parts[3])),
                        "memory_free_mb": int(float(parts[4])),
                        "compute_cap": parts[5],
                    }})
                except ValueError:
                    pass
    return {{"step": step, "gpu_count": len(devices), "devices": devices, "kaggle_gpu_verified": bool(devices)}}


def patch_rpc_op_count_guard_if_requested():
    if not PATCH_RPC_OP_COUNT_GUARD:
        return {{"ok": True, "skipped": True}}
    path = SRC / "ggml" / "include" / "ggml-rpc.h"
    if not path.is_file():
        return {{"ok": False, "error": "ggml_rpc_header_missing"}}
    text = path.read_text(encoding="utf-8")
    original = 'static_assert(GGML_OP_COUNT == 101, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");'
    patched = 'static_assert(GGML_OP_COUNT == 101 || GGML_OP_COUNT == 102, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");'
    if original not in text:
        return {{"ok": False, "error": "ggml_rpc_static_assert_pattern_missing", "header_digest": sha_text(text[:4000])}}
    text = text.replace("#define RPC_PROTO_PATCH_VERSION    5", "#define RPC_PROTO_PATCH_VERSION    6")
    text = text.replace(original, patched)
    path.write_text(text, encoding="utf-8")
    return {{
        "ok": True,
        "patch": "rpc_op_count_guard_accepts_101_or_102_and_patch_version_6",
        "header_digest": sha_text(text[:4000]),
    }}


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def export_runtime_tarball_if_requested(cli, rpc):
    if not EXPORT_RUNTIME_TARBALL:
        return {{"ok": True, "skipped": True}}
    if not cli or not rpc:
        return {{"ok": False, "error": "runtime_binaries_missing"}}
    if RUNTIME_DIR.exists():
        shutil.rmtree(RUNTIME_DIR)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for binary in [cli, rpc]:
        target = RUNTIME_DIR / binary.name
        shutil.copy2(binary, target)
        target.chmod(target.stat().st_mode | 0o111)
        copied.append(target.name)
    for lib_dir in [BUILD / "bin", BUILD / "lib"]:
        if lib_dir.is_dir():
            for lib in lib_dir.glob("*.so*"):
                if lib.is_file():
                    target = RUNTIME_DIR / lib.name
                    shutil.copy2(lib, target)
                    copied.append(target.name)
    metadata = {{
        "schema": "deepseek_v4_flash_llama_v4_runtime_bundle_v1",
        "repo_url": REPO_URL,
        "branch": BRANCH,
        "commit_hash_public": report.get("commit_hash_public", ""),
        "cuda_architectures": CUDA_ARCHITECTURES,
        "patch_rpc_op_count_guard": PATCH_RPC_OP_COUNT_GUARD,
        "binaries": sorted(set(copied)),
        "public_artifact_safe": True,
    }}
    (RUNTIME_DIR / "runtime-bundle.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    with tarfile.open(RUNTIME_TARBALL, "w:gz") as tar:
        tar.add(RUNTIME_DIR, arcname="deepseek-v4-flash-llama-v4-runtime")
    return {{
        "ok": True,
        "tarball_name": RUNTIME_TARBALL.name,
        "tarball_size_bytes": int(RUNTIME_TARBALL.stat().st_size),
        "tarball_sha256": sha_file(RUNTIME_TARBALL),
        "runtime_dir_public": False,
        "file_count": len(list(RUNTIME_DIR.iterdir())),
        "binaries": sorted(set(copied)),
    }}


report = {{
    "schema": SCHEMA,
    "ok": False,
    "started_at": utc_now(),
    "updated_at": utc_now(),
    "repo_url_public": True,
    "repo_url": REPO_URL,
    "branch": BRANCH,
    "cuda_architectures": CUDA_ARCHITECTURES,
    "cuda_build_jobs": CUDA_BUILD_JOBS,
    "patch_rpc_op_count_guard": PATCH_RPC_OP_COUNT_GUARD,
    "steps": {{}},
    "blockers": [],
    "diagnosis_codes": [],
    "public_artifact_safe": True,
    "paths_public": False,
    "credentials_public": False,
    "private_runtime_state_public": False,
    "raw_build_logs_public": False,
}}

try:
    ROOT.mkdir(parents=True, exist_ok=True)
    report["disk_start"] = disk_snapshot(ROOT)
    report["hardware"] = probe_hardware()
    write_report("hardware_probe_complete", report)
    step, _, _ = run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(SRC)], timeout=900)
    report["steps"]["git_clone"] = step
    write_report("git_clone_complete", report)
    if not step.get("ok"):
        report["blockers"].append("kaggle_llama_v4_git_clone_failed")
    else:
        step, stdout, _ = run(["git", "rev-parse", "HEAD"], cwd=SRC, timeout=60)
        report["steps"]["git_rev_parse"] = step
        report["commit_hash_public"] = stdout.strip()[:40] if step.get("ok") else ""
        patch_step = patch_rpc_op_count_guard_if_requested()
        report["steps"]["patch_rpc_op_count_guard"] = patch_step
        if not patch_step.get("ok"):
            report["blockers"].append("kaggle_llama_v4_rpc_guard_patch_failed")
        configure = [
            "cmake",
            "-S",
            str(SRC),
            "-B",
            str(BUILD),
            "-DGGML_CUDA=ON",
            "-DGGML_RPC=ON",
            "-DLLAMA_CURL=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_CUDA_NO_VMM=ON",
        ]
        if CUDA_ARCHITECTURES:
            configure.append("-DCMAKE_CUDA_ARCHITECTURES=" + CUDA_ARCHITECTURES)
        step, _, _ = run(configure, timeout=900)
        report["steps"]["cmake_configure"] = step
        write_report("cmake_configure_complete", report)
        if not step.get("ok"):
            report["blockers"].append("kaggle_llama_v4_cmake_configure_failed")
        elif "kaggle_llama_v4_rpc_guard_patch_failed" not in report["blockers"]:
            step, _, _ = run([
                "cmake",
                "--build",
                str(BUILD),
                "--config",
                "Release",
                "-j",
                str(max(1, int(CUDA_BUILD_JOBS or 1))),
                "--target",
                "llama-cli",
                "rpc-server",
            ], timeout=max(1, int(CUDA_BUILD_TIMEOUT_SECONDS or 3600)))
            report["steps"]["cmake_build"] = step
            write_report("cmake_build_complete", report)
            if not step.get("ok"):
                report["blockers"].append("kaggle_llama_v4_cmake_build_failed")
    cli = find_binary(BUILD, ["llama-cli", "main"]) or find_binary(SRC, ["llama-cli", "main"])
    rpc = find_binary(BUILD, ["rpc-server", "llama-rpc-server"]) or find_binary(SRC, ["rpc-server", "llama-rpc-server"])
    report["llama_cli_present"] = bool(cli)
    report["rpc_server_present"] = bool(rpc)
    report["llama_cli_path_public"] = False
    report["rpc_server_path_public"] = False
    if cli:
        env = env_for_binary(cli)
        step, stdout, _ = run([str(cli), "--version"], timeout=60, env=env)
        report["steps"]["llama_cli_version"] = step
        report["llama_cli_version_digest"] = sha_text(stdout)
        step, stdout, _ = run([str(cli), "--help"], timeout=60, env=env)
        report["steps"]["llama_cli_help"] = step
        report["llama_cli_supports_rpc"] = "--rpc" in stdout
        report["llama_cli_supports_tensor_split"] = "tensor-split" in stdout
    if rpc:
        env = env_for_binary(rpc)
        step, stdout, _ = run([str(rpc), "--help"], timeout=60, env=env)
        report["steps"]["rpc_server_help"] = step
        report["rpc_server_help_digest"] = sha_text(stdout)
    report["runtime_tarball"] = export_runtime_tarball_if_requested(cli, rpc)
    if EXPORT_RUNTIME_TARBALL and not report["runtime_tarball"].get("ok"):
        report["blockers"].append("kaggle_llama_v4_runtime_tarball_export_failed")
    if not report.get("llama_cli_present"):
        report["blockers"].append("kaggle_llama_v4_llama_cli_missing")
    if not report.get("rpc_server_present"):
        report["blockers"].append("kaggle_llama_v4_rpc_server_missing")
    if report.get("llama_cli_present") and not report.get("llama_cli_supports_rpc"):
        report["blockers"].append("kaggle_llama_v4_rpc_flag_missing")
    report["ok"] = bool(
        report.get("llama_cli_present")
        and report.get("rpc_server_present")
        and report.get("llama_cli_supports_rpc")
        and not report["blockers"]
    )
    report["diagnosis_codes"].append("kaggle_llama_v4_rpc_build_ready" if report["ok"] else "kaggle_llama_v4_rpc_build_not_ready")
    report["disk_final"] = disk_snapshot(ROOT)
    write_report("complete" if report["ok"] else "blocked", report)
except Exception as exc:
    report["ok"] = False
    report["error_type"] = type(exc).__name__
    report["error_digest"] = sha_text(str(exc))
    report["blockers"].append("kaggle_llama_v4_build_preflight_exception")
    write_report("exception", report)
finally:
    try:
        if ROOT.exists():
            shutil.rmtree(ROOT)
            report["temp_cleanup"] = {{"ok": True, "path_public": False}}
    except Exception as exc:
        report["temp_cleanup"] = {{"ok": False, "error_type": type(exc).__name__, "error_digest": sha_text(str(exc))}}
    write_report("final", report)
    print(json.dumps({{"schema": SCHEMA, "ok": report.get("ok"), "stage": report.get("stage"), "diagnosis_codes": report.get("diagnosis_codes")}}, sort_keys=True))
'''


def build_package(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    owner = args.kaggle_owner or default_kaggle_owner()
    if not owner:
        raise SystemExit("--kaggle-owner or ~/.kaggle/kaggle.json username is required")
    slug = safe_slug(args.kernel_slug_prefix)[:34] + "-" + str(int(time.time()))[-8:]
    slug = slug[:45].strip("-")
    kernel_dir = output_dir / "private-kaggle-kernel"
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text(render_kernel(args), encoding="utf-8")
    title = f"CT DSV4 Llama Build {slug[-8:]}"
    metadata = {
        "id": f"{owner}/{slug}",
        "title": title,
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


def resolve_pushed_kernel_ref(
    package: dict[str, Any],
    push_step: dict[str, Any],
    *,
    runner: Runner,
    timeout_seconds: float,
) -> tuple[str, dict[str, Any] | None]:
    actual = str(push_step.get("actual_kernel_ref") or "")
    if actual:
        return actual, None
    metadata = package.get("metadata") if isinstance(package.get("metadata"), dict) else {}
    declared = str(package.get("declared_kernel_ref") or "")
    owner = declared.split("/", 1)[0]
    title = str(metadata.get("title") or "")
    step = run_step(
        "kaggle_kernel_ref_resolve",
        ["kaggle", "kernels", "list", "--user", owner, "--sort-by", "dateRun", "--page-size", "20"],
        runner=runner,
        timeout_seconds=timeout_seconds,
    )
    output = f"{step.get('stdout_tail') or ''}\n{step.get('stderr_tail') or ''}"
    resolved = extract_kernel_list_ref(output, owner=owner, title=title)
    step["resolved_kernel_ref"] = resolved
    step["resolved"] = bool(resolved)
    return (resolved or declared), step


def public_worker_summary(worker: dict[str, Any]) -> dict[str, Any]:
    hardware = worker.get("hardware") if isinstance(worker.get("hardware"), dict) else {}
    return {
        "schema": "deepseek_v4_flash_kaggle_llama_v4_build_worker_summary_v1",
        "present": bool(worker),
        "worker_schema": str(worker.get("schema") or ""),
        "worker_ok": worker.get("ok") is True,
        "stage": str(worker.get("stage") or ""),
        "repo_url": str(worker.get("repo_url") or ""),
        "branch": str(worker.get("branch") or ""),
        "commit_hash_public": str(worker.get("commit_hash_public") or ""),
        "cuda_architectures": str(worker.get("cuda_architectures") or ""),
        "gpu_count": int(hardware.get("gpu_count") or 0),
        "kaggle_gpu_verified": hardware.get("kaggle_gpu_verified") is True,
        "llama_cli_present": worker.get("llama_cli_present") is True,
        "rpc_server_present": worker.get("rpc_server_present") is True,
        "llama_cli_supports_rpc": worker.get("llama_cli_supports_rpc") is True,
        "llama_cli_supports_tensor_split": worker.get("llama_cli_supports_tensor_split") is True,
        "patch_rpc_op_count_guard": worker.get("patch_rpc_op_count_guard") is True,
        "patch_rpc_op_count_guard_ok": (worker.get("steps") or {}).get("patch_rpc_op_count_guard", {}).get("ok") is True if isinstance(worker.get("steps"), dict) else False,
        "runtime_tarball_exported": (worker.get("runtime_tarball") or {}).get("ok") is True if isinstance(worker.get("runtime_tarball"), dict) else False,
        "runtime_tarball_name": str((worker.get("runtime_tarball") or {}).get("tarball_name") or "") if isinstance(worker.get("runtime_tarball"), dict) else "",
        "runtime_tarball_size_bytes": int((worker.get("runtime_tarball") or {}).get("tarball_size_bytes") or 0) if isinstance(worker.get("runtime_tarball"), dict) else 0,
        "runtime_tarball_sha256": str((worker.get("runtime_tarball") or {}).get("tarball_sha256") or "") if isinstance(worker.get("runtime_tarball"), dict) else "",
        "cmake_configure_ok": (worker.get("steps") or {}).get("cmake_configure", {}).get("ok") is True if isinstance(worker.get("steps"), dict) else False,
        "cmake_build_ok": (worker.get("steps") or {}).get("cmake_build", {}).get("ok") is True if isinstance(worker.get("steps"), dict) else False,
        "temp_cleanup_ok": (worker.get("temp_cleanup") or {}).get("ok") is True if isinstance(worker.get("temp_cleanup"), dict) else False,
        "blockers": [str(item) for item in (worker.get("blockers") if isinstance(worker.get("blockers"), list) else [])],
        "diagnosis_codes": [str(item) for item in (worker.get("diagnosis_codes") if isinstance(worker.get("diagnosis_codes"), list) else [])],
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
    worker = public_worker_summary(worker_report)
    pushed = any(step.get("name") == "kaggle_kernel_push" and step.get("ok") for step in steps)
    output_downloaded = bool(worker_report)
    kernel_deleted = any(step.get("name") == "kaggle_kernel_delete" and step.get("ok") for step in steps)
    private_removed = not (output_dir / "private-kaggle-kernel").exists()
    runtime_tarball_exported = worker.get("runtime_tarball_exported") is True
    runtime_tarball_ready = bool((not args.export_runtime_tarball) or runtime_tarball_exported)
    ok = bool(
        pushed
        and output_downloaded
        and worker.get("worker_ok")
        and kernel_deleted
        and private_removed
        and runtime_tarball_ready
    )
    blockers = set(worker.get("blockers") or [])
    if not pushed:
        blockers.add("kaggle_llama_v4_kernel_push_failed")
    if pushed and not output_downloaded:
        blockers.add("kaggle_llama_v4_worker_report_missing")
    if args.export_runtime_tarball and not runtime_tarball_exported:
        blockers.add("kaggle_llama_v4_runtime_tarball_export_missing")
    if not kernel_deleted and not args.skip_kaggle_cleanup:
        blockers.add("kaggle_llama_v4_kernel_cleanup_missing")
    if not private_removed:
        blockers.add("kaggle_llama_v4_private_package_not_removed")
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "llama_v4_runtime_build_ready": bool(ok and worker.get("worker_ok")),
        "fresh_kaggle_run_performed": pushed,
        "output_dir": str(output_dir),
        "runtime": {
            "runtime_backend": "llama_cpp_v4_fork",
            "runtime_fork": f"{args.repo_url}@{args.branch}",
            "repo_url": args.repo_url,
            "branch": args.branch,
            "cuda_architectures": args.cuda_architectures,
            "patch_rpc_op_count_guard": bool(args.patch_rpc_op_count_guard),
            "accelerator": args.accelerator,
        },
        "worker_summary": worker,
        "runtime_artifact": {
            "runtime_tarball_requested": bool(args.export_runtime_tarball),
            "runtime_tarball_exported": runtime_tarball_exported,
            "runtime_tarball_name": worker.get("runtime_tarball_name") or "",
            "runtime_tarball_size_bytes": int(worker.get("runtime_tarball_size_bytes") or 0),
            "runtime_tarball_sha256": worker.get("runtime_tarball_sha256") or "",
            "runtime_tarball_path_public": False,
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
            "kaggle_llama_v4_runtime_build_ready" if ok else "kaggle_llama_v4_runtime_build_not_ready",
            "kaggle_llama_v4_gpu_hardware_verified" if worker.get("kaggle_gpu_verified") else "kaggle_llama_v4_gpu_hardware_not_verified",
        ],
        "safety": {
            "public_artifact_safe": True,
            "raw_build_logs_public": False,
            "credentials_public": False,
            "private_kernel_payload_public": False,
            "private_runtime_state_public": False,
            "weight_tensor_values_public": False,
        },
        "public_artifact_safe": True,
    }


def run_live_probe(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    package = build_package(args, output_dir=output_dir)
    steps: list[dict[str, Any]] = []
    worker_report: dict[str, Any] = {}
    try:
        push_command = ["kaggle", "kernels", "push", "-p", str(package["kernel_dir"]), "-t", str(args.kernel_timeout_seconds)]
        if args.accelerator:
            push_command.extend(["--accelerator", args.accelerator])
        print(f"[{utc_now()}] pushing DeepSeek V4 llama build Kaggle kernel {package['declared_kernel_ref']}", flush=True)
        push_step = run_step("kaggle_kernel_push", push_command, runner=runner, timeout_seconds=args.kaggle_push_timeout_seconds)
        steps.append(push_step)
        if push_step.get("ok"):
            kernel_ref, resolve_step = resolve_pushed_kernel_ref(
                package,
                push_step,
                runner=runner,
                timeout_seconds=args.kaggle_push_timeout_seconds,
            )
            if resolve_step:
                steps.append(resolve_step)
            package["kernel_ref"] = kernel_ref
            print(f"[{utc_now()}] waiting for {kernel_ref}", flush=True)
            status_step = wait_kaggle_terminal(
                kernel_ref,
                runner=runner,
                timeout_seconds=args.kaggle_status_timeout_seconds,
                poll_interval=args.kaggle_status_poll_interval,
            )
            steps.append(status_step)
            output_path = output_dir / "kaggle-output"
            output_step = run_step(
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
                    "deepseek_v4_flash_kaggle_llama_v4_build_worker.json|deepseek-v4-flash-llama-v4-runtime.tar.gz",
                ],
                runner=runner,
                timeout_seconds=args.kaggle_output_timeout_seconds,
            )
            steps.append(output_step)
            worker_report = load_json(output_path / "deepseek_v4_flash_kaggle_llama_v4_build_worker.json")
            if not args.skip_kaggle_cleanup:
                print(f"[{utc_now()}] deleting DeepSeek V4 llama build Kaggle kernel {kernel_ref}", flush=True)
                delete_step = run_step(
                    "kaggle_kernel_delete",
                    ["kaggle", "kernels", "delete", kernel_ref, "-y"],
                    runner=runner,
                    timeout_seconds=args.kaggle_delete_timeout_seconds,
                )
                steps.append(delete_step)
    finally:
        if not args.keep_private_package:
            shutil.rmtree(output_dir / "private-kaggle-kernel", ignore_errors=True)
    report = build_report(args, output_dir=output_dir, package=package, steps=steps, worker_report=worker_report)
    write_json(output_dir / "deepseek_v4_flash_kaggle_llama_v4_build_preflight.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kaggle-owner", default=default_kaggle_owner())
    parser.add_argument("--kernel-slug-prefix", default="ct-dsv4-llama-build")
    parser.add_argument("--accelerator", default=DEFAULT_ACCELERATOR)
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--cuda-architectures", default="75")
    parser.add_argument("--cuda-build-jobs", type=int, default=2)
    parser.add_argument("--cuda-build-timeout-seconds", type=int, default=3600)
    parser.add_argument("--patch-rpc-op-count-guard", action="store_true")
    parser.add_argument("--export-runtime-tarball", action="store_true")
    parser.add_argument("--kernel-timeout-seconds", type=int, default=5400)
    parser.add_argument("--kaggle-push-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--kaggle-status-timeout-seconds", type=float, default=5700.0)
    parser.add_argument("--kaggle-status-poll-interval", type=float, default=60.0)
    parser.add_argument("--kaggle-output-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--kaggle-delete-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--skip-kaggle-cleanup", action="store_true")
    parser.add_argument("--keep-private-package", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.cuda_build_jobs < 1 or args.cuda_build_jobs > 8:
        raise SystemExit("--cuda-build-jobs must be between 1 and 8")
    if args.cuda_build_timeout_seconds < 60 or args.cuda_build_timeout_seconds > 7200:
        raise SystemExit("--cuda-build-timeout-seconds must be between 60 and 7200")
    if args.kernel_timeout_seconds < 300 or args.kernel_timeout_seconds > 7200:
        raise SystemExit("--kernel-timeout-seconds must be between 300 and 7200")
    if args.kaggle_status_timeout_seconds < 60 or args.kaggle_status_timeout_seconds > 7500:
        raise SystemExit("--kaggle-status-timeout-seconds must be between 60 and 7500")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_live_probe(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"DeepSeek V4 llama build ready: {report.get('llama_v4_runtime_build_ready')}")
        print(f"Blockers: {','.join(report.get('blockers') or []) or 'none'}")
        print(f"Report: {Path(args.output_dir) / 'deepseek_v4_flash_kaggle_llama_v4_build_preflight.json'}")
    return 0 if report.get("public_artifact_safe") else 1


if __name__ == "__main__":
    raise SystemExit(main())
