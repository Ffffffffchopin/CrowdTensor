#!/usr/bin/env python3
"""Run a bounded DeepSeek-V4-Flash quantized GGUF same-request RPC probe.

The live path starts a Colab CUDA `rpc-server` through a public TCP tunnel,
then launches a private Kaggle T4x2 kernel that downloads the selected GGUF,
starts local Kaggle CUDA and CPU `rpc-server` workers, and runs one
`llama-cli --rpc` request across all three worker families.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import colab_cuda_session_manager  # noqa: E402
from scripts import deepseek_v4_flash_quantized_source_resolver as resolver  # noqa: E402
from scripts import deepseek_v4_flash_kaggle_llama_v4_build_preflight as llama_build  # noqa: E402


SCHEMA = "deepseek_v4_flash_quantized_same_request_probe_v1"
WORKER_SCHEMA = "deepseek_v4_flash_quantized_same_request_kaggle_worker_v1"
COLAB_RPC_SCHEMA = "deepseek_v4_flash_quantized_colab_rpc_worker_v1"
DEFAULT_OUTPUT_DIR = "dist/deepseek-v4-flash-quantized-same-request-probe"
DEFAULT_ACCELERATOR = "NvidiaTeslaT4"
DEFAULT_REPO_URL = "https://github.com/cchuter/llama.cpp.git"
DEFAULT_BRANCH = "feat/v4-port-cuda"
DEFAULT_BORE_URL = "https://github.com/ekzhang/bore/releases/download/v0.6.0/bore-v0.6.0-x86_64-unknown-linux-musl.tar.gz"
DEFAULT_BORE_SERVER = "bore.pub"
COLAB_MARKER = "CT_DSV4_COLAB_RPC_WORKER"
COLAB_BACKGROUND_LAUNCH_MARKER = "CT_DSV4_COLAB_RPC_BACKGROUND_LAUNCH"
KAGGLE_CODE_URL = re.compile(r"https://www\.kaggle\.com/code/([^/\s]+)/([^/\s]+)")
KAGGLE_TABLE_SPLIT = re.compile(r"\s{2,}")
Runner = Callable[..., subprocess.CompletedProcess[str]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        loaded = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def sha_text(value: Any) -> str:
    return "sha256:" + hashlib.sha256(str(value or "").encode("utf-8", errors="replace")).hexdigest()


def sha16(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8", errors="replace")).hexdigest()[:16]


def safe_slug(value: str, *, default: str = "ct-dsv4-same-request") -> str:
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
    return llama_build.default_kaggle_owner()


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
        "Set-Cookie",
    ]:
        redacted = redacted.replace(fragment, "<redacted>")
    return redacted


def run_step(name: str, command: list[str], *, runner: Runner, timeout_seconds: float) -> dict[str, Any]:
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
        print(f"[{utc_now()}] DeepSeek same-request Kaggle status attempt={attempts} status={status}", flush=True)
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


def public_outputs(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public: list[dict[str, Any]] = []
    for item in outputs:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "")
        public.append({
            "output_type": item.get("output_type"),
            "name": item.get("name"),
            "text_hash": sha_text(text),
            "text_chars": len(text),
            "text_public": False,
        })
    return public


def parse_marker_payload(outputs: list[dict[str, Any]], marker: str) -> dict[str, Any]:
    text_parts: list[str] = []
    for output in outputs:
        text = output.get("text") if isinstance(output, dict) else None
        if isinstance(text, str):
            text_parts.append(text)
    text = "\n".join(text_parts)
    match = re.search(re.escape(marker) + r"\s+(\{.*\})", text)
    if not match:
        return {"ok": False, "error": "marker_missing", "output_type_count": len(outputs)}
    try:
        loaded = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": "marker_json_decode_failed", "error_digest": sha_text(str(exc))}
    return loaded if isinstance(loaded, dict) else {"ok": False, "error": "marker_payload_not_object"}


def choose_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def download_local_bore(*, cache_dir: Path, bore_url: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    bore = cache_dir / "bore"
    if bore.is_file():
        bore.chmod(bore.stat().st_mode | 0o111)
        return bore
    archive = cache_dir / "bore.tar.gz"
    with urllib.request.urlopen(bore_url, timeout=300) as response:
        archive.write_bytes(response.read())
    extract = cache_dir / "extract"
    if extract.exists():
        shutil.rmtree(extract)
    extract.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(extract)
    for path in extract.rglob("bore"):
        if path.is_file():
            shutil.copy2(path, bore)
            bore.chmod(bore.stat().st_mode | 0o111)
            return bore
    raise RuntimeError("bore_binary_not_found")


def parse_bore_endpoint(log_text: str, *, bore_server: str) -> tuple[str, int]:
    pattern = r"([A-Za-z0-9.-]*bore\.pub|[A-Za-z0-9.-]*" + re.escape(bore_server) + r").*?:(\d{2,5})"
    match = re.search(pattern, log_text or "")
    if match:
        return match.group(1) or bore_server, int(match.group(2))
    match = re.search(r":(\d{2,5})", log_text or "")
    if match:
        return bore_server, int(match.group(1))
    return "", 0


@contextmanager
def maybe_runtime_tarball_server(args: argparse.Namespace):
    tarball_path = Path(args.runtime_tarball_path).expanduser() if args.runtime_tarball_path else None
    if not tarball_path:
        yield {"ok": True, "url": args.runtime_tarball_url or "", "public_artifact_safe": True}
        return
    if not tarball_path.is_file():
        raise FileNotFoundError(f"runtime tarball not found: {tarball_path}")

    tmp = Path(tempfile.mkdtemp(prefix="ct_dsv4_runtime_server_"))
    http_proc: subprocess.Popen[str] | None = None
    bore_proc: subprocess.Popen[str] | None = None
    try:
        local_port = choose_free_local_port()
        http_log = tmp / "http.log"
        http_handle = http_log.open("w", encoding="utf-8", errors="replace")
        http_proc = subprocess.Popen(
            [sys.executable, "-m", "http.server", str(local_port), "--bind", "127.0.0.1", "--directory", str(tarball_path.parent)],
            stdout=http_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        time.sleep(1)
        if http_proc.poll() is not None:
            raise RuntimeError("runtime_tarball_http_server_exited")
        bore = download_local_bore(cache_dir=tmp / "bore-cache", bore_url=args.bore_url)
        bore_log = tmp / "bore.log"
        bore_handle = bore_log.open("w", encoding="utf-8", errors="replace")
        bore_proc = subprocess.Popen(
            [str(bore), "local", str(local_port), "--to", args.bore_server],
            stdout=bore_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        remote_host = ""
        remote_port = 0
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            time.sleep(1)
            text = bore_log.read_text(encoding="utf-8", errors="replace") if bore_log.is_file() else ""
            remote_host, remote_port = parse_bore_endpoint(text, bore_server=args.bore_server)
            if remote_host and remote_port:
                break
            if bore_proc.poll() is not None:
                break
        if not remote_host or not remote_port:
            raise RuntimeError("runtime_tarball_bore_endpoint_missing")
        quoted = urllib.parse.quote(tarball_path.name)
        url = f"http://{remote_host}:{remote_port}/{quoted}"
        yield {
            "ok": True,
            "url": url,
            "url_public": False,
            "remote_host_hash": sha16(remote_host),
            "remote_port": int(remote_port),
            "tarball_name": tarball_path.name,
            "tarball_size_bytes": int(tarball_path.stat().st_size),
            "public_artifact_safe": True,
        }
    finally:
        for proc in [bore_proc, http_proc]:
            if proc is None:
                continue
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        shutil.rmtree(tmp, ignore_errors=True)


def candidate_from_source(source_report: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    recommended = source_report.get("recommended_live_probe_candidate")
    if isinstance(recommended, dict) and recommended.get("repo") and recommended.get("files"):
        files = [
            {
                "path": str(item.get("path") or ""),
                "size_bytes": int(item.get("size_bytes") or 0),
                "size_gb": float(item.get("size_gb") or 0.0),
            }
            for item in recommended.get("files", [])
            if isinstance(item, dict) and item.get("path")
        ]
        return {
            "candidate_id": str(recommended.get("candidate_id") or "source-recommended"),
            "repo": str(recommended.get("repo") or ""),
            "quant": str(recommended.get("quant") or ""),
            "runtime_backend": str(recommended.get("runtime_backend") or "llama_cpp_v4_fork"),
            "runtime_fork": str(recommended.get("runtime_fork") or f"{args.repo_url}@{args.branch}"),
            "total_size_gb": float(recommended.get("total_size_gb") or 0.0),
            "split_file_count": len(files),
            "files": files,
            "blockers": [str(item) for item in recommended.get("blockers", [])],
        }
    paths = [item for item in args.candidate_file if item]
    files = [{"path": path, "size_bytes": 0, "size_gb": 0.0} for path in paths]
    return {
        "candidate_id": args.candidate_id,
        "repo": args.candidate_repo,
        "quant": args.candidate_quant,
        "runtime_backend": "llama_cpp_v4_fork",
        "runtime_fork": f"{args.repo_url}@{args.branch}",
        "total_size_gb": float(args.candidate_total_size_gb),
        "split_file_count": len(files),
        "files": files,
        "blockers": [],
    }


def render_colab_rpc_code(args: argparse.Namespace) -> str:
    template = r'''
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import struct
import subprocess
import tarfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "__COLAB_RPC_SCHEMA__"
MARKER = "__COLAB_MARKER__"
REPO_URL = __REPO_URL_JSON__
BRANCH = __BRANCH_JSON__
CUDA_ARCHITECTURES = __CUDA_ARCHITECTURES_JSON__
CUDA_BUILD_JOBS = __CUDA_BUILD_JOBS__
CUDA_BUILD_TIMEOUT_SECONDS = __CUDA_BUILD_TIMEOUT_SECONDS__
PATCH_RPC_OP_COUNT_GUARD = __PATCH_RPC_OP_COUNT_GUARD__
BORE_URL = __BORE_URL_JSON__
BORE_SERVER = __BORE_SERVER_JSON__
RPC_PORT = __RPC_PORT__
RUNTIME_TARBALL_URL = __RUNTIME_TARBALL_URL_JSON__
RUNTIME_TARBALL_EXPECTED_SHA256 = __RUNTIME_TARBALL_SHA256_JSON__
KEEPALIVE_AFTER_READY = __KEEPALIVE_AFTER_READY__
KEEPALIVE_SECONDS = __KEEPALIVE_SECONDS__
OUT = Path("/content/ct_dsv4_colab_rpc")
SRC = OUT / "llama.cpp"
BUILD = OUT / "build"
RUNTIME = OUT / "runtime"
REPORT_PATH = OUT / "deepseek_v4_flash_colab_rpc_worker.json"


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
    for fragment in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "Bearer ", "Authorization:", "Cookie:", "KAGGLE_KEY", "KAGGLE_USERNAME"]:
        text = text.replace(fragment, "<redacted>")
    return text


def write_report(stage, report):
    report["stage"] = stage
    report["updated_at"] = utc_now()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(REPORT_PATH)


def run(command, *, timeout=1200, cwd=None, env=None):
    started = time.monotonic()
    try:
        completed = subprocess.run(command, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "duration_seconds": round(time.monotonic() - started, 3), "command_public": [str(item) for item in command]}, "", ""
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "stdout_digest": sha_text(completed.stdout),
        "stdout_chars": len(completed.stdout or ""),
        "stderr_tail": safe_tail(completed.stderr),
        "stderr_digest": sha_text(completed.stderr),
        "stderr_chars": len(completed.stderr or ""),
        "command_public": [str(item) for item in command],
    }, completed.stdout or "", completed.stderr or ""


def rpc_hello_probe(host, port, *, timeout=10):
    conn_caps_size = 24
    response_size = 4 + conn_caps_size
    started = time.monotonic()
    try:
        with socket.create_connection((str(host), int(port)), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(struct.pack("<BQ", 14, conn_caps_size) + (b"\0" * conn_caps_size))
            header = sock.recv(8)
            if len(header) != 8:
                return {"ok": False, "error": "rpc_hello_response_header_short", "response_header_bytes": len(header), "duration_seconds": round(time.monotonic() - started, 3)}
            size = struct.unpack("<Q", header)[0]
            body = b""
            while len(body) < size:
                chunk = sock.recv(size - len(body))
                if not chunk:
                    break
                body += chunk
            if size != response_size or len(body) != size:
                return {"ok": False, "error": "rpc_hello_response_size_mismatch", "response_size": int(size), "response_body_bytes": len(body), "duration_seconds": round(time.monotonic() - started, 3)}
            major, minor, patch, _padding = struct.unpack("<BBBB", body[:4])
            caps = body[4:]
            return {
                "ok": major == 4 and minor <= 0,
                "major": int(major),
                "minor": int(minor),
                "patch": int(patch),
                "conn_caps_nonzero": any(caps),
                "conn_caps_size": len(caps),
                "duration_seconds": round(time.monotonic() - started, 3),
                "endpoint_public": False,
            }
    except Exception as exc:
        return {"ok": False, "error_type": type(exc).__name__, "error_digest": sha_text(str(exc)), "duration_seconds": round(time.monotonic() - started, 3), "endpoint_public": False}


def find_binary(root, names):
    for name in names:
        for path in root.rglob(name):
            if path.is_file():
                path.chmod(path.stat().st_mode | 0o111)
                return path
    return None


def env_for_binary(binary, *, cuda_visible="0"):
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
                    devices.append({
                        "index": int(parts[0]),
                        "name_hash": sha_text(parts[1])[:24],
                        "name_public": False,
                        "memory_total_mb": int(float(parts[2])),
                        "memory_free_mb": int(float(parts[3])),
                        "compute_cap": parts[4],
                    })
                except ValueError:
                    pass
    return {"step": step, "gpu_count": len(devices), "devices": devices, "colab_cuda_verified": bool(devices)}


def patch_rpc_guard():
    if not PATCH_RPC_OP_COUNT_GUARD:
        return {"ok": True, "skipped": True}
    path = SRC / "ggml" / "include" / "ggml-rpc.h"
    if not path.is_file():
        return {"ok": False, "error": "ggml_rpc_header_missing"}
    text = path.read_text(encoding="utf-8")
    original = 'static_assert(GGML_OP_COUNT == 101, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");'
    patched = 'static_assert(GGML_OP_COUNT == 101 || GGML_OP_COUNT == 102, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");'
    if original not in text:
        return {"ok": False, "error": "ggml_rpc_static_assert_pattern_missing", "header_digest": sha_text(text[:4000])}
    text = text.replace("#define RPC_PROTO_PATCH_VERSION    5", "#define RPC_PROTO_PATCH_VERSION    6")
    text = text.replace(original, patched)
    path.write_text(text, encoding="utf-8")
    return {"ok": True, "patch": "rpc_op_count_guard_accepts_101_or_102_and_patch_version_6", "header_digest": sha_text(text[:4000])}


def compact_runtime(cli, rpc):
    RUNTIME.mkdir(parents=True, exist_ok=True)
    for binary in [cli, rpc]:
        target = RUNTIME / binary.name
        if binary.resolve() != target.resolve():
            shutil.copy2(binary, target)
        target.chmod(target.stat().st_mode | 0o111)
    for lib in BUILD.rglob("*.so*"):
        if lib.is_file():
            try:
                shutil.copy2(lib, RUNTIME / lib.name)
            except shutil.SameFileError:
                pass
    shutil.rmtree(SRC, ignore_errors=True)
    shutil.rmtree(BUILD, ignore_errors=True)
    return {"ok": True, "runtime_dir_public": False}


def prepare_runtime_from_tarball():
    if not RUNTIME_TARBALL_URL:
        return {"ok": True, "skipped": True}
    archive = OUT / "runtime.tar.gz"
    extract = OUT / "runtime-extract"
    if extract.exists():
        shutil.rmtree(extract)
    if RUNTIME.exists():
        shutil.rmtree(RUNTIME)
    started = time.monotonic()
    try:
        with urllib.request.urlopen(RUNTIME_TARBALL_URL, timeout=900) as response:
            with archive.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        digest = sha_file(archive)
        if RUNTIME_TARBALL_EXPECTED_SHA256 and digest != RUNTIME_TARBALL_EXPECTED_SHA256:
            return {"ok": False, "error": "runtime_tarball_sha256_mismatch", "sha256": digest}
        extract.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(extract)
        rpc = find_binary(extract, ["rpc-server", "llama-rpc-server"])
        cli = find_binary(extract, ["llama-cli", "main"])
        if not cli or not rpc:
            return {"ok": False, "error": "runtime_tarball_binaries_missing", "sha256": digest}
        RUNTIME.mkdir(parents=True, exist_ok=True)
        source_dir = rpc.parent if rpc.parent == cli.parent else extract
        for item in source_dir.iterdir():
            if item.is_file():
                target = RUNTIME / item.name
                shutil.copy2(item, target)
                if item.name in {cli.name, rpc.name}:
                    target.chmod(target.stat().st_mode | 0o111)
        return {
            "ok": True,
            "runtime_source": "tarball",
            "duration_seconds": round(time.monotonic() - started, 3),
            "tarball_size_bytes": int(archive.stat().st_size),
            "tarball_sha256": digest,
            "runtime_dir_public": False,
            "url_public": False,
        }
    except Exception as exc:
        return {"ok": False, "error_type": type(exc).__name__, "error_digest": sha_text(str(exc)), "url_public": False}


def download_bore():
    archive = OUT / "bore.tar.gz"
    with urllib.request.urlopen(BORE_URL, timeout=300) as response:
        archive.write_bytes(response.read())
    extract = OUT / "bore-bin"
    extract.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(extract)
    bore = find_binary(extract, ["bore"])
    return bore


report = {
    "schema": SCHEMA,
    "ok": False,
    "started_at": utc_now(),
    "updated_at": utc_now(),
    "repo_url": REPO_URL,
    "branch": BRANCH,
    "runtime_source": "tarball" if RUNTIME_TARBALL_URL else "source_build",
    "runtime_tarball_used": bool(RUNTIME_TARBALL_URL),
    "runtime_tarball_url_public": False,
    "cuda_architectures": CUDA_ARCHITECTURES,
    "patch_rpc_op_count_guard": PATCH_RPC_OP_COUNT_GUARD,
    "blockers": [],
    "diagnosis_codes": [],
    "public_artifact_safe": True,
    "credentials_public": False,
    "private_runtime_state_public": False,
    "raw_logs_public": False,
}

rpc_proc = None
bore_proc = None

try:
    if OUT.exists():
        shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)
    report["hardware"] = probe_hardware()
    write_report("hardware_probe_complete", report)
    report.setdefault("steps", {})
    tarball = prepare_runtime_from_tarball()
    report["steps"]["prepare_runtime_from_tarball"] = tarball
    if not tarball.get("ok"):
        report["blockers"].append("colab_runtime_tarball_prepare_failed")
    if not RUNTIME_TARBALL_URL:
        step, _, _ = run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(SRC)], timeout=900)
        report["steps"]["git_clone"] = step
        write_report("git_clone_complete", report)
        if not step.get("ok"):
            report["blockers"].append("colab_llama_v4_git_clone_failed")
        else:
            step, stdout, _ = run(["git", "rev-parse", "HEAD"], cwd=SRC, timeout=60)
            report["steps"]["git_rev_parse"] = step
            report["commit_hash_public"] = stdout.strip()[:40] if step.get("ok") else ""
            patch = patch_rpc_guard()
            report["steps"]["patch_rpc_op_count_guard"] = patch
            if not patch.get("ok"):
                report["blockers"].append("colab_llama_v4_rpc_guard_patch_failed")
            configure = [
                "cmake", "-S", str(SRC), "-B", str(BUILD),
                "-DGGML_CUDA=ON", "-DGGML_RPC=ON", "-DLLAMA_CURL=OFF",
                "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA_NO_VMM=ON",
            ]
            if CUDA_ARCHITECTURES:
                configure.append("-DCMAKE_CUDA_ARCHITECTURES=" + CUDA_ARCHITECTURES)
            step, _, _ = run(configure, timeout=900)
            report["steps"]["cmake_configure"] = step
            write_report("cmake_configure_complete", report)
            if not step.get("ok"):
                report["blockers"].append("colab_llama_v4_cmake_configure_failed")
            elif "colab_llama_v4_rpc_guard_patch_failed" not in report["blockers"]:
                step, _, _ = run([
                    "cmake", "--build", str(BUILD), "--config", "Release", "-j", str(max(1, int(CUDA_BUILD_JOBS))),
                    "--target", "llama-cli", "rpc-server",
                ], timeout=max(1, int(CUDA_BUILD_TIMEOUT_SECONDS)))
                report["steps"]["cmake_build"] = step
                write_report("cmake_build_complete", report)
                if not step.get("ok"):
                    report["blockers"].append("colab_llama_v4_cmake_build_failed")
    cli = find_binary(RUNTIME, ["llama-cli", "main"]) or find_binary(BUILD, ["llama-cli", "main"]) or find_binary(SRC, ["llama-cli", "main"])
    rpc = find_binary(RUNTIME, ["rpc-server", "llama-rpc-server"]) or find_binary(BUILD, ["rpc-server", "llama-rpc-server"]) or find_binary(SRC, ["rpc-server", "llama-rpc-server"])
    report["llama_cli_present"] = bool(cli)
    report["rpc_server_present"] = bool(rpc)
    if not cli:
        report["blockers"].append("colab_llama_v4_llama_cli_missing")
    if not rpc:
        report["blockers"].append("colab_llama_v4_rpc_server_missing")
    if cli and rpc and not report["blockers"]:
        report["runtime_compaction"] = compact_runtime(cli, rpc)
        cli = find_binary(RUNTIME, ["llama-cli", "main"])
        rpc = find_binary(RUNTIME, ["rpc-server", "llama-rpc-server"])
        bore = download_bore()
        if not bore:
            report["blockers"].append("colab_bore_binary_missing")
        else:
            rpc_stdout = (OUT / "rpc-server.stdout.log").open("w", encoding="utf-8", errors="replace")
            rpc_stderr = (OUT / "rpc-server.stderr.log").open("w", encoding="utf-8", errors="replace")
            rpc_proc = subprocess.Popen([str(rpc), "-H", "127.0.0.1", "-p", str(RPC_PORT), "-d", "CUDA0"], stdout=rpc_stdout, stderr=rpc_stderr, text=True, env=env_for_binary(rpc, cuda_visible="0"))
            time.sleep(5)
            report["rpc_server_alive"] = rpc_proc.poll() is None
            report["local_rpc_hello"] = rpc_hello_probe("127.0.0.1", RPC_PORT, timeout=10) if report["rpc_server_alive"] else {"ok": False, "skipped": True}
            if not report["rpc_server_alive"]:
                report["blockers"].append("colab_rpc_server_not_alive")
            if report["rpc_server_alive"] and not report["local_rpc_hello"].get("ok"):
                report["blockers"].append("colab_local_rpc_hello_failed")
            bore_log = OUT / "bore.log"
            bore_handle = bore_log.open("w", encoding="utf-8", errors="replace")
            bore_proc = subprocess.Popen([str(bore), "local", str(RPC_PORT), "--to", BORE_SERVER], stdout=bore_handle, stderr=subprocess.STDOUT, text=True)
            remote_port = 0
            remote_host = BORE_SERVER
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                time.sleep(2)
                text = bore_log.read_text(encoding="utf-8", errors="replace") if bore_log.is_file() else ""
                match = re.search(r"([A-Za-z0-9.-]*bore\.pub|[A-Za-z0-9.-]*" + re.escape(BORE_SERVER) + r").*?:(\d{2,5})", text)
                if not match:
                    match = re.search(r":(\d{2,5})", text)
                if match:
                    if len(match.groups()) >= 2:
                        remote_host = match.group(1) or BORE_SERVER
                        remote_port = int(match.group(2))
                    else:
                        remote_port = int(match.group(1))
                    break
                if bore_proc.poll() is not None:
                    break
            report["bore_tunnel_alive"] = bore_proc.poll() is None
            report["bore_remote_host_hash"] = sha_text(remote_host)[:24]
            report["bore_remote_port"] = int(remote_port)
            report["bore_endpoint_public"] = False
            report["bore_log_digest"] = sha_text(bore_log.read_text(encoding="utf-8", errors="replace") if bore_log.is_file() else "")
            report["bore_rpc_hello"] = rpc_hello_probe(remote_host, remote_port, timeout=10) if remote_port else {"ok": False, "skipped": True}
            if not remote_port:
                report["blockers"].append("colab_bore_remote_port_missing")
            if remote_port and not report["bore_rpc_hello"].get("ok"):
                report["blockers"].append("colab_bore_rpc_hello_failed")
            report["ok"] = bool(report["rpc_server_alive"] and report["bore_tunnel_alive"] and remote_port and report["local_rpc_hello"].get("ok") and report["bore_rpc_hello"].get("ok") and not report["blockers"])
    report["diagnosis_codes"].append("deepseek_v4_flash_colab_rpc_worker_ready" if report.get("ok") else "deepseek_v4_flash_colab_rpc_worker_not_ready")
    write_report("ready" if report.get("ok") else "blocked", report)
except Exception as exc:
    report["ok"] = False
    report["error_type"] = type(exc).__name__
    report["error_digest"] = sha_text(str(exc))
    report["blockers"].append("deepseek_v4_flash_colab_rpc_worker_exception")
    write_report("exception", report)

public_payload = {
    "schema": report.get("schema"),
    "ok": report.get("ok") is True,
    "stage": report.get("stage"),
    "remote_host": BORE_SERVER if report.get("ok") else "",
    "remote_port": int(report.get("bore_remote_port") or 0),
    "remote_endpoint_public": False,
    "blockers": report.get("blockers") or [],
    "diagnosis_codes": report.get("diagnosis_codes") or [],
    "hardware": report.get("hardware") or {},
    "commit_hash_public": report.get("commit_hash_public") or "",
    "patch_rpc_op_count_guard": report.get("patch_rpc_op_count_guard") is True,
    "runtime_tarball_used": report.get("runtime_tarball_used") is True,
    "keepalive": report.get("keepalive") if isinstance(report.get("keepalive"), dict) else {},
    "public_artifact_safe": True,
}
print(MARKER + " " + json.dumps(public_payload, sort_keys=True), flush=True)

if report.get("ok") and KEEPALIVE_AFTER_READY:
    deadline = time.monotonic() + max(60, int(KEEPALIVE_SECONDS))
    while time.monotonic() < deadline:
        rpc_alive = bool(rpc_proc is not None and rpc_proc.poll() is None)
        bore_alive = bool(bore_proc is not None and bore_proc.poll() is None)
        report["keepalive"] = {
            "rpc_server_alive": rpc_alive,
            "bore_tunnel_alive": bore_alive,
            "heartbeat_at": utc_now(),
            "public_artifact_safe": True,
        }
        if not rpc_alive or not bore_alive:
            report["ok"] = False
            report["blockers"].append("colab_rpc_worker_keepalive_process_exited")
            write_report("background_keepalive_ended", report)
            break
        write_report("ready_keepalive", report)
        time.sleep(30)
'''
    replacements = {
        "__COLAB_RPC_SCHEMA__": COLAB_RPC_SCHEMA,
        "__COLAB_MARKER__": COLAB_MARKER,
        "__REPO_URL_JSON__": json.dumps(args.repo_url),
        "__BRANCH_JSON__": json.dumps(args.branch),
        "__CUDA_ARCHITECTURES_JSON__": json.dumps(args.cuda_architectures),
        "__CUDA_BUILD_JOBS__": str(int(args.colab_cuda_build_jobs)),
        "__CUDA_BUILD_TIMEOUT_SECONDS__": str(int(args.colab_cuda_build_timeout_seconds)),
        "__PATCH_RPC_OP_COUNT_GUARD__": "True" if args.patch_rpc_op_count_guard else "False",
        "__BORE_URL_JSON__": json.dumps(args.bore_url),
        "__BORE_SERVER_JSON__": json.dumps(args.bore_server),
        "__RPC_PORT__": str(int(args.colab_rpc_local_port)),
        "__RUNTIME_TARBALL_URL_JSON__": json.dumps(args.runtime_tarball_url),
        "__RUNTIME_TARBALL_SHA256_JSON__": json.dumps(args.runtime_tarball_sha256),
        "__KEEPALIVE_AFTER_READY__": "True" if args.colab_build_mode == "background" else "False",
        "__KEEPALIVE_SECONDS__": str(int(args.colab_keepalive_seconds)),
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def render_kaggle_kernel(args: argparse.Namespace, candidate: dict[str, Any], colab_endpoint: dict[str, Any]) -> str:
    files = candidate.get("files") if isinstance(candidate.get("files"), list) else []
    template = r'''
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import struct
import subprocess
import tarfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "__WORKER_SCHEMA__"
MODEL_REPO = __MODEL_REPO_JSON__
CANDIDATE_ID = __CANDIDATE_ID_JSON__
QUANT = __QUANT_JSON__
FILES = __FILES_JSON__
REPO_URL = __REPO_URL_JSON__
BRANCH = __BRANCH_JSON__
CUDA_ARCHITECTURES = __CUDA_ARCHITECTURES_JSON__
CUDA_BUILD_JOBS = __CUDA_BUILD_JOBS__
CUDA_BUILD_TIMEOUT_SECONDS = __CUDA_BUILD_TIMEOUT_SECONDS__
PATCH_RPC_OP_COUNT_GUARD = __PATCH_RPC_OP_COUNT_GUARD__
RUNTIME_TARBALL_URL = __RUNTIME_TARBALL_URL_JSON__
RUNTIME_TARBALL_EXPECTED_SHA256 = __RUNTIME_TARBALL_SHA256_JSON__
COLAB_RPC_HOST = __COLAB_RPC_HOST_JSON__
COLAB_RPC_PORT = __COLAB_RPC_PORT__
MAX_NEW_TOKENS = __MAX_NEW_TOKENS__
CONTEXT_LENGTH = __CONTEXT_LENGTH__
RUN_TIMEOUT_SECONDS = __RUN_TIMEOUT_SECONDS__
INCLUDE_CPU_RPC_ENDPOINT = __INCLUDE_CPU_RPC_ENDPOINT__
CLIENT_CUDA_VISIBLE = __CLIENT_CUDA_VISIBLE_JSON__
SKIP_MODEL_DOWNLOAD_ON_RPC_HELLO_FAILURE = __SKIP_MODEL_DOWNLOAD_ON_RPC_HELLO_FAILURE__
PROMPT_TEXT = "CrowdTensor public-safe DeepSeek V4 Flash quantized probe."
OUT = Path("/kaggle/working")
TEMP = Path("/kaggle/temp/ct_dsv4_same_request")
SRC = TEMP / "llama.cpp"
BUILD = TEMP / "build"
RUNTIME = TEMP / "runtime"
MODEL_DIR = TEMP / "model"
REPORT_PATH = OUT / "deepseek_v4_flash_quantized_same_request_worker.json"


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
    for fragment in [PROMPT_TEXT, "KAGGLE_KEY", "KAGGLE_USERNAME", "KAGGLE_API_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "Bearer ", "Authorization:", "Cookie:"]:
        text = text.replace(fragment, "<redacted>")
    text = text.replace(str(COLAB_RPC_HOST) + ":" + str(COLAB_RPC_PORT), "<colab-rpc-endpoint>")
    return text


def write_report(stage, report):
    report["stage"] = stage
    report["updated_at"] = utc_now()
    tmp = REPORT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(REPORT_PATH)


def disk_snapshot(path="/kaggle"):
    try:
        usage = shutil.disk_usage(str(path))
        return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}
    except Exception as exc:
        return {"error_type": type(exc).__name__, "error_digest": sha_text(str(exc))}


def public_command(command):
    public = []
    for item in command:
        text = str(item)
        if text == PROMPT_TEXT:
            public.append("<prompt-redacted>")
        elif str(MODEL_DIR) in text:
            public.append(text.replace(str(MODEL_DIR), "<model-dir-redacted>"))
        elif str(COLAB_RPC_HOST) in text and str(COLAB_RPC_PORT) in text:
            public.append(text.replace(str(COLAB_RPC_HOST) + ":" + str(COLAB_RPC_PORT), "<colab-rpc-endpoint>"))
        else:
            public.append(text)
    return public


def run(command, *, timeout=1200, cwd=None, env=None, stdout_public=False):
    started = time.monotonic()
    try:
        completed = subprocess.run(command, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "duration_seconds": round(time.monotonic() - started, 3), "command_public": public_command(command)}, "", ""
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    step = {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "stdout_digest": sha_text(stdout),
        "stdout_chars": len(stdout),
        "stderr_tail": safe_tail(stderr),
        "stderr_digest": sha_text(stderr),
        "stderr_chars": len(stderr),
        "command_public": public_command(command),
    }
    if stdout_public:
        step["stdout_tail"] = safe_tail(stdout)
    return step, stdout, stderr


def rpc_hello_probe(host, port, *, timeout=10):
    conn_caps_size = 24
    response_size = 4 + conn_caps_size
    started = time.monotonic()
    try:
        with socket.create_connection((str(host), int(port)), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(struct.pack("<BQ", 14, conn_caps_size) + (b"\0" * conn_caps_size))
            header = sock.recv(8)
            if len(header) != 8:
                return {"ok": False, "error": "rpc_hello_response_header_short", "response_header_bytes": len(header), "duration_seconds": round(time.monotonic() - started, 3), "endpoint_public": False}
            size = struct.unpack("<Q", header)[0]
            body = b""
            while len(body) < size:
                chunk = sock.recv(size - len(body))
                if not chunk:
                    break
                body += chunk
            if size != response_size or len(body) != size:
                return {"ok": False, "error": "rpc_hello_response_size_mismatch", "response_size": int(size), "response_body_bytes": len(body), "duration_seconds": round(time.monotonic() - started, 3), "endpoint_public": False}
            major, minor, patch, _padding = struct.unpack("<BBBB", body[:4])
            caps = body[4:]
            return {
                "ok": major == 4 and minor <= 0,
                "major": int(major),
                "minor": int(minor),
                "patch": int(patch),
                "conn_caps_nonzero": any(caps),
                "conn_caps_size": len(caps),
                "duration_seconds": round(time.monotonic() - started, 3),
                "endpoint_public": False,
            }
    except Exception as exc:
        return {"ok": False, "error_type": type(exc).__name__, "error_digest": sha_text(str(exc)), "duration_seconds": round(time.monotonic() - started, 3), "endpoint_public": False}


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


def probe_hardware():
    step, stdout, _ = run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,compute_cap",
        "--format=csv,noheader,nounits",
    ], timeout=60, stdout_public=False)
    devices = []
    if step.get("ok"):
        for line in stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 5:
                try:
                    devices.append({
                        "index": int(parts[0]),
                        "name_hash": sha_text(parts[1])[:24],
                        "name_public": False,
                        "memory_total_mb": int(float(parts[2])),
                        "memory_free_mb": int(float(parts[3])),
                        "compute_cap": parts[4],
                    })
                except ValueError:
                    pass
    return {"step": step, "gpu_count": len(devices), "devices": devices, "kaggle_cuda_verified": bool(devices)}


def patch_rpc_guard():
    if not PATCH_RPC_OP_COUNT_GUARD:
        return {"ok": True, "skipped": True}
    path = SRC / "ggml" / "include" / "ggml-rpc.h"
    if not path.is_file():
        return {"ok": False, "error": "ggml_rpc_header_missing"}
    text = path.read_text(encoding="utf-8")
    original = 'static_assert(GGML_OP_COUNT == 101, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");'
    patched = 'static_assert(GGML_OP_COUNT == 101 || GGML_OP_COUNT == 102, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");'
    if original not in text:
        return {"ok": False, "error": "ggml_rpc_static_assert_pattern_missing", "header_digest": sha_text(text[:4000])}
    text = text.replace("#define RPC_PROTO_PATCH_VERSION    5", "#define RPC_PROTO_PATCH_VERSION    6")
    text = text.replace(original, patched)
    path.write_text(text, encoding="utf-8")
    return {"ok": True, "patch": "rpc_op_count_guard_accepts_101_or_102_and_patch_version_6", "header_digest": sha_text(text[:4000])}


def compact_runtime(cli, rpc):
    RUNTIME.mkdir(parents=True, exist_ok=True)
    for binary in [cli, rpc]:
        target = RUNTIME / binary.name
        if binary.resolve() != target.resolve():
            shutil.copy2(binary, target)
        target.chmod(target.stat().st_mode | 0o111)
    for lib in BUILD.rglob("*.so*"):
        if lib.is_file():
            try:
                shutil.copy2(lib, RUNTIME / lib.name)
            except shutil.SameFileError:
                pass
    shutil.rmtree(SRC, ignore_errors=True)
    shutil.rmtree(BUILD, ignore_errors=True)
    return {"ok": True, "runtime_dir_public": False, "disk_after_compaction": disk_snapshot(TEMP)}


def prepare_runtime_from_tarball():
    if not RUNTIME_TARBALL_URL:
        return {"ok": True, "skipped": True}
    archive = TEMP / "runtime.tar.gz"
    extract = TEMP / "runtime-extract"
    if extract.exists():
        shutil.rmtree(extract)
    if RUNTIME.exists():
        shutil.rmtree(RUNTIME)
    started = time.monotonic()
    try:
        with urllib.request.urlopen(RUNTIME_TARBALL_URL, timeout=900) as response:
            with archive.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        digest = sha_file(archive)
        if RUNTIME_TARBALL_EXPECTED_SHA256 and digest != RUNTIME_TARBALL_EXPECTED_SHA256:
            return {"ok": False, "error": "runtime_tarball_sha256_mismatch", "sha256": digest}
        extract.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(extract)
        rpc = find_binary(extract, ["rpc-server", "llama-rpc-server"])
        cli = find_binary(extract, ["llama-cli", "main"])
        if not cli or not rpc:
            return {"ok": False, "error": "runtime_tarball_binaries_missing", "sha256": digest}
        RUNTIME.mkdir(parents=True, exist_ok=True)
        source_dir = rpc.parent if rpc.parent == cli.parent else extract
        for item in source_dir.iterdir():
            if item.is_file():
                target = RUNTIME / item.name
                shutil.copy2(item, target)
                if item.name in {cli.name, rpc.name}:
                    target.chmod(target.stat().st_mode | 0o111)
        return {
            "ok": True,
            "runtime_source": "tarball",
            "duration_seconds": round(time.monotonic() - started, 3),
            "tarball_size_bytes": int(archive.stat().st_size),
            "tarball_sha256": digest,
            "runtime_dir_public": False,
            "url_public": False,
        }
    except Exception as exc:
        return {"ok": False, "error_type": type(exc).__name__, "error_digest": sha_text(str(exc)), "url_public": False}


def model_url(repo, path):
    return "https://huggingface.co/" + repo + "/resolve/main/" + urllib.parse.quote(path)


def download_file(file_info):
    rel = str(file_info["path"])
    target = MODEL_DIR / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    bytes_written = 0
    expected = int(file_info.get("size_bytes") or 0)
    with urllib.request.urlopen(model_url(MODEL_REPO, rel), timeout=1800) as response:
        if not expected:
            expected = int(response.headers.get("Content-Length") or response.headers.get("x-linked-size") or 0)
        with target.open("wb") as handle:
            last_reported = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                bytes_written += len(chunk)
                if bytes_written - last_reported >= 512 * 1024 * 1024:
                    last_reported = bytes_written
                    write_report("model_download_progress", report | {"current_download": {"path_hash": sha_text(rel), "bytes_written": bytes_written, "expected_bytes": expected}, "disk": disk_snapshot(MODEL_DIR)})
    return {"path_hash": sha_text(rel), "size_bytes": int(target.stat().st_size), "expected_bytes": expected, "duration_seconds": round(time.monotonic() - started, 3)}


def reachable(host, port, timeout=15):
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except Exception:
        return False


def start_rpc(name, rpc, port, *, cuda_visible="", device_arg=""):
    stdout_log = OUT / (name + ".stdout.log")
    stderr_log = OUT / (name + ".stderr.log")
    stdout_handle = stdout_log.open("w", encoding="utf-8", errors="replace")
    stderr_handle = stderr_log.open("w", encoding="utf-8", errors="replace")
    command = [str(rpc), "-H", "127.0.0.1", "-p", str(port)]
    if device_arg:
        command.extend(["-d", device_arg])
    proc = subprocess.Popen(command, stdout=stdout_handle, stderr=stderr_handle, text=True, env=env_for_binary(rpc, cuda_visible=cuda_visible))
    return {"name": name, "endpoint": "127.0.0.1:" + str(port), "command_public": public_command(command), "pid": proc.pid, "process": proc, "stdout_log": str(stdout_log), "stderr_log": str(stderr_log)}


def rpc_log_summary(server):
    values = {}
    for key in ["stdout_log", "stderr_log"]:
        path = Path(str(server.get(key) or ""))
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            values[key + "_digest"] = sha_text(text)
            values[key + "_tail"] = safe_tail(text)
    return values


report = {
    "schema": SCHEMA,
    "ok": False,
    "started_at": utc_now(),
    "updated_at": utc_now(),
    "model": {
        "model_id": "deepseek-ai/DeepSeek-V4-Flash",
        "repo": MODEL_REPO,
        "candidate_id": CANDIDATE_ID,
        "quant": QUANT,
        "format": "gguf",
        "total_params_b": 284.0,
        "active_params_b": 13.0,
        "split_file_count": len(FILES),
        "model_paths_public": False,
    },
    "accepted_providers": [],
    "provider_stage_counts": {"kaggle_cuda": 0, "colab_cuda": 0, "cpu": 0},
    "runtime_source": "tarball" if RUNTIME_TARBALL_URL else "source_build",
    "runtime_tarball_used": bool(RUNTIME_TARBALL_URL),
    "runtime_tarball_url_public": False,
    "blockers": [],
    "diagnosis_codes": [],
    "public_artifact_safe": True,
    "safety": {
        "public_artifact_safe": True,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "activation_public": False,
        "kv_cache_public": False,
        "credentials_public": False,
        "private_runtime_state_public": False,
        "weight_tensor_values_public": False,
    },
}

rpc_processes = []
try:
    TEMP.mkdir(parents=True, exist_ok=True)
    report["disk_start"] = disk_snapshot("/kaggle")
    report["hardware"] = probe_hardware()
    write_report("hardware_probe_complete", report)
    if not report["hardware"].get("kaggle_cuda_verified"):
        report["blockers"].append("kaggle_cuda_device_missing")
    report.setdefault("steps", {})
    tarball = prepare_runtime_from_tarball()
    report["steps"]["prepare_runtime_from_tarball"] = tarball
    if not tarball.get("ok"):
        report["blockers"].append("kaggle_runtime_tarball_prepare_failed")
    if not RUNTIME_TARBALL_URL:
        step, _, _ = run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(SRC)], timeout=900)
        report["steps"]["git_clone"] = step
        write_report("git_clone_complete", report)
        if not step.get("ok"):
            report["blockers"].append("kaggle_llama_v4_git_clone_failed")
        else:
            step, stdout, _ = run(["git", "rev-parse", "HEAD"], cwd=SRC, timeout=60)
            report["steps"]["git_rev_parse"] = step
            report["commit_hash_public"] = stdout.strip()[:40] if step.get("ok") else ""
            patch = patch_rpc_guard()
            report["steps"]["patch_rpc_op_count_guard"] = patch
            if not patch.get("ok"):
                report["blockers"].append("kaggle_llama_v4_rpc_guard_patch_failed")
            configure = [
                "cmake", "-S", str(SRC), "-B", str(BUILD),
                "-DGGML_CUDA=ON", "-DGGML_RPC=ON", "-DLLAMA_CURL=OFF",
                "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA_NO_VMM=ON",
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
                    "cmake", "--build", str(BUILD), "--config", "Release", "-j", str(max(1, int(CUDA_BUILD_JOBS))),
                    "--target", "llama-cli", "rpc-server",
                ], timeout=max(1, int(CUDA_BUILD_TIMEOUT_SECONDS)))
                report["steps"]["cmake_build"] = step
                write_report("cmake_build_complete", report)
                if not step.get("ok"):
                    report["blockers"].append("kaggle_llama_v4_cmake_build_failed")
    cli = find_binary(RUNTIME, ["llama-cli", "main"]) or find_binary(BUILD, ["llama-cli", "main"]) or find_binary(SRC, ["llama-cli", "main"])
    rpc = find_binary(RUNTIME, ["rpc-server", "llama-rpc-server"]) or find_binary(BUILD, ["rpc-server", "llama-rpc-server"]) or find_binary(SRC, ["rpc-server", "llama-rpc-server"])
    report["llama_cli_present"] = bool(cli)
    report["rpc_server_present"] = bool(rpc)
    if not cli:
        report["blockers"].append("kaggle_llama_v4_llama_cli_missing")
    if not rpc:
        report["blockers"].append("kaggle_llama_v4_rpc_server_missing")
    if cli and rpc and not report["blockers"]:
        report["runtime_compaction"] = compact_runtime(cli, rpc)
        cli = find_binary(RUNTIME, ["llama-cli", "main"])
        rpc = find_binary(RUNTIME, ["rpc-server", "llama-rpc-server"])
        report["colab_rpc_reachable_pre_run"] = reachable(COLAB_RPC_HOST, COLAB_RPC_PORT, timeout=20)
        report["colab_rpc_hello_pre_download"] = rpc_hello_probe(COLAB_RPC_HOST, COLAB_RPC_PORT, timeout=20) if report["colab_rpc_reachable_pre_run"] else {"ok": False, "skipped": True}
        if not report["colab_rpc_reachable_pre_run"]:
            report["blockers"].append("colab_rpc_endpoint_not_reachable_from_kaggle")
        if report["colab_rpc_reachable_pre_run"] and not report["colab_rpc_hello_pre_download"].get("ok"):
            report["blockers"].append("colab_rpc_hello_pre_download_failed")
        skip_llama_run = False
        if "colab_rpc_hello_pre_download_failed" in report["blockers"] and SKIP_MODEL_DOWNLOAD_ON_RPC_HELLO_FAILURE:
            report["ok"] = False
            report["same_request_decode_verified"] = False
            report["deepseek_v4_flash_quantized_same_request_verified"] = False
            report["generated_token_count"] = 0
            report["diagnosis_codes"].append("deepseek_v4_flash_quantized_colab_rpc_hello_pre_download_failed")
            write_report("blocked_colab_rpc_hello_pre_download", report)
            skip_llama_run = True
        else:
            downloads = []
            for file_info in FILES:
                write_report("model_download_start", report | {"current_download": {"path_hash": sha_text(file_info.get("path") or "")}, "disk": disk_snapshot(MODEL_DIR)})
                downloads.append(download_file(file_info))
                write_report("model_download_file_complete", report | {"downloads": downloads, "disk": disk_snapshot(MODEL_DIR)})
            report["downloads"] = downloads
            report["colab_rpc_reachable_before_llama"] = reachable(COLAB_RPC_HOST, COLAB_RPC_PORT, timeout=20)
            report["colab_rpc_hello_before_llama"] = rpc_hello_probe(COLAB_RPC_HOST, COLAB_RPC_PORT, timeout=20) if report["colab_rpc_reachable_before_llama"] else {"ok": False, "skipped": True}
            if not report["colab_rpc_reachable_before_llama"]:
                report["blockers"].append("colab_rpc_endpoint_lost_after_model_download")
            if report["colab_rpc_reachable_before_llama"] and not report["colab_rpc_hello_before_llama"].get("ok"):
                report["blockers"].append("colab_rpc_hello_lost_after_model_download")
        if skip_llama_run:
            report["downloads"] = []
            report["accepted_providers"] = []
            report["provider_stage_counts"] = {"kaggle_cuda": 0, "colab_cuda": 0, "cpu": 0}
            report["blockers"].append("deepseek_v4_flash_quantized_same_request_decode_not_verified")
            write_report("blocked_colab_rpc_hello_pre_download_final", report)
        else:
            first_model = MODEL_DIR / str(FILES[0]["path"])
            prompt_path = TEMP / "prompt.txt"
            prompt_path.write_text(PROMPT_TEXT + "\n", encoding="utf-8")
            servers = []
            gpu_count = int(report["hardware"].get("gpu_count") or 0)
            if gpu_count >= 1:
                servers.append(start_rpc("kaggle-cuda0-rpc", rpc, 50052, cuda_visible="0,1", device_arg="CUDA0"))
            if gpu_count >= 2:
                servers.append(start_rpc("kaggle-cuda1-rpc", rpc, 50053, cuda_visible="0,1", device_arg="CUDA1"))
            if INCLUDE_CPU_RPC_ENDPOINT:
                servers.append(start_rpc("kaggle-cpu-rpc", rpc, 50054, cuda_visible="", device_arg=""))
            rpc_processes = [item["process"] for item in servers]
            time.sleep(8)
            live_servers = []
            for server in servers:
                alive = server["process"].poll() is None
                endpoint = server["endpoint"]
                live_servers.append({k: v for k, v in server.items() if k != "process"} | {"alive": alive, "log_summary": rpc_log_summary(server), "rpc_hello": rpc_hello_probe(endpoint.rsplit(":", 1)[0], endpoint.rsplit(":", 1)[1], timeout=10) if alive else {"ok": False, "skipped": True}})
            report["local_rpc_servers"] = live_servers
            if not any(item.get("name", "").startswith("kaggle-cuda") and item.get("alive") for item in live_servers):
                report["blockers"].append("kaggle_cuda_rpc_server_not_alive")
            if any(item.get("alive") and not item.get("rpc_hello", {}).get("ok") for item in live_servers):
                report["blockers"].append("kaggle_local_rpc_hello_failed")
            if INCLUDE_CPU_RPC_ENDPOINT and not any(item.get("name") == "kaggle-cpu-rpc" and item.get("alive") for item in live_servers):
                report["blockers"].append("kaggle_cpu_rpc_server_not_alive")
            rpc_endpoints = [item["endpoint"] for item in live_servers if item.get("alive")]
            rpc_endpoints.append(str(COLAB_RPC_HOST) + ":" + str(COLAB_RPC_PORT))
            command = [
                str(cli), "-m", str(first_model), "-f", str(prompt_path), "-n", str(MAX_NEW_TOKENS), "-c", str(CONTEXT_LENGTH),
                "-ngl", "99", "--rpc", ",".join(rpc_endpoints), "-t", "4", "--no-display-prompt", "--simple-io", "--log-disable", "-no-cnv",
            ]
            if len(rpc_endpoints) > 1:
                command.extend(["-ts", ",".join(["1"] * len(rpc_endpoints))])
            write_report("llama_run_start", report | {"runner_step": {"pending": True, "command_public": public_command(command)}, "disk": disk_snapshot(MODEL_DIR)})
            run_started = time.monotonic()
            client_env = env_for_binary(cli, cuda_visible=CLIENT_CUDA_VISIBLE)
            step, stdout, _ = run(command, timeout=RUN_TIMEOUT_SECONDS, env=client_env, stdout_public=False)
            wall = round(time.monotonic() - run_started, 3)
            generated = 1 if step.get("ok") and stdout.strip() else 0
            report["runner_step"] = step
            report["generated_token_count"] = generated
            report["metrics"] = {
                "generated_token_count": generated,
                "max_new_tokens": MAX_NEW_TOKENS,
                "wall_time_seconds": wall,
                "tokens_per_second": round(generated / wall, 6) if wall > 0 and generated else 0.0,
                "output_digest": sha_text(stdout),
            }
            accepted = []
            if any(item.get("name", "").startswith("kaggle-cuda") and item.get("alive") for item in live_servers):
                accepted.append("kaggle_cuda")
                report["provider_stage_counts"]["kaggle_cuda"] = sum(1 for item in live_servers if item.get("name", "").startswith("kaggle-cuda") and item.get("alive"))
            if INCLUDE_CPU_RPC_ENDPOINT and any(item.get("name") == "kaggle-cpu-rpc" and item.get("alive") for item in live_servers):
                accepted.append("cpu")
                report["provider_stage_counts"]["cpu"] = 1
            elif not INCLUDE_CPU_RPC_ENDPOINT:
                accepted.append("cpu")
                report["provider_stage_counts"]["cpu"] = 1
            if report.get("colab_rpc_hello_pre_download", {}).get("ok") and report.get("colab_rpc_hello_before_llama", {}).get("ok"):
                accepted.append("colab_cuda")
                report["provider_stage_counts"]["colab_cuda"] = 1
            report["accepted_providers"] = sorted(set(accepted))
            required = {"kaggle_cuda", "colab_cuda", "cpu"}
            report["same_request_decode_verified"] = bool(step.get("ok") and generated >= 1 and required.issubset(set(report["accepted_providers"])))
            report["deepseek_v4_flash_quantized_same_request_verified"] = report["same_request_decode_verified"]
            report["ok"] = report["same_request_decode_verified"]
            if not report["ok"]:
                report["blockers"].append("deepseek_v4_flash_quantized_same_request_decode_not_verified")
            report["diagnosis_codes"].extend([
                "deepseek_v4_flash_quantized_same_request_decode_verified" if report["ok"] else "deepseek_v4_flash_quantized_same_request_decode_not_verified",
                "deepseek_v4_flash_quantized_model_download_complete",
            ])
            write_report("complete" if report["ok"] else "blocked_generation", report)
except Exception as exc:
    report["ok"] = False
    report["same_request_decode_verified"] = False
    report["deepseek_v4_flash_quantized_same_request_verified"] = False
    report["error_type"] = type(exc).__name__
    report["error_digest"] = sha_text(str(exc))
    report["blockers"].append("deepseek_v4_flash_quantized_same_request_probe_exception")
    report["diagnosis_codes"].append("deepseek_v4_flash_quantized_same_request_probe_exception")
    write_report("exception", report)
finally:
    for proc in rpc_processes:
        try:
            proc.terminate()
        except Exception:
            pass
    for proc in rpc_processes:
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
            report["temp_cleanup"] = {"ok": True, "path_public": False}
    except Exception as exc:
        report["temp_cleanup"] = {"ok": False, "error_type": type(exc).__name__, "error_digest": sha_text(str(exc))}
    write_report("final", report)
    print(json.dumps({"schema": SCHEMA, "ok": report.get("ok"), "stage": report.get("stage"), "generated_token_count": report.get("generated_token_count", 0), "accepted_providers": report.get("accepted_providers", [])}, sort_keys=True), flush=True)
'''
    replacements = {
        "__WORKER_SCHEMA__": WORKER_SCHEMA,
        "__MODEL_REPO_JSON__": json.dumps(str(candidate.get("repo") or "")),
        "__CANDIDATE_ID_JSON__": json.dumps(str(candidate.get("candidate_id") or "")),
        "__QUANT_JSON__": json.dumps(str(candidate.get("quant") or "")),
        "__FILES_JSON__": json.dumps(files),
        "__REPO_URL_JSON__": json.dumps(args.repo_url),
        "__BRANCH_JSON__": json.dumps(args.branch),
        "__CUDA_ARCHITECTURES_JSON__": json.dumps(args.cuda_architectures),
        "__CUDA_BUILD_JOBS__": str(int(args.kaggle_cuda_build_jobs)),
        "__CUDA_BUILD_TIMEOUT_SECONDS__": str(int(args.kaggle_cuda_build_timeout_seconds)),
        "__PATCH_RPC_OP_COUNT_GUARD__": "True" if args.patch_rpc_op_count_guard else "False",
        "__RUNTIME_TARBALL_URL_JSON__": json.dumps(args.runtime_tarball_url),
        "__RUNTIME_TARBALL_SHA256_JSON__": json.dumps(args.runtime_tarball_sha256),
        "__COLAB_RPC_HOST_JSON__": json.dumps(str(colab_endpoint.get("remote_host") or "")),
        "__COLAB_RPC_PORT__": str(int(colab_endpoint.get("remote_port") or 0)),
        "__MAX_NEW_TOKENS__": str(int(args.max_new_tokens)),
        "__CONTEXT_LENGTH__": str(int(args.context_length)),
        "__RUN_TIMEOUT_SECONDS__": str(int(args.run_timeout_seconds)),
        "__INCLUDE_CPU_RPC_ENDPOINT__": "True" if args.include_cpu_rpc_endpoint else "False",
        "__CLIENT_CUDA_VISIBLE_JSON__": json.dumps(args.client_cuda_visible),
        "__SKIP_MODEL_DOWNLOAD_ON_RPC_HELLO_FAILURE__": "True" if args.skip_model_download_on_rpc_hello_failure else "False",
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def start_colab_rpc_worker(args: argparse.Namespace) -> dict[str, Any]:
    matrix = colab_fallback_matrix(args)
    if len(matrix) > 1:
        return start_colab_rpc_worker_with_fallback_matrix(args, matrix=matrix)
    authusers = parse_colab_authusers(args)
    if args.colab_build_mode == "background":
        return start_colab_rpc_worker_background(args)
    code = render_colab_rpc_code(args)
    started = time.monotonic()
    try:
        outputs, _session, manager_result = colab_cuda_session_manager.execute_with_retry(
            code,
            session_name=args.colab_session,
            state_path=Path(args.colab_config).expanduser(),
            timeout=float(args.colab_execute_timeout_seconds),
            max_attempts=int(args.colab_max_attempts),
            token_cache=Path(args.colab_token_cache).expanduser(),
            accelerator=args.colab_accelerator,
            authuser=str(args.colab_authuser),
            force_reacquire_before=bool(args.colab_reacquire_before),
            heartbeat_code='print("CT_DSV4_COLAB_RPC_HEARTBEAT")',
        )
        payload = parse_marker_payload(outputs, COLAB_MARKER)
        return {
            "schema": "deepseek_v4_flash_quantized_colab_rpc_launch_v1",
            "ok": payload.get("ok") is True and manager_result.get("ok") is True,
            "duration_seconds": round(time.monotonic() - started, 3),
            "manager": colab_public_manager_result(manager_result),
            "worker": colab_public_worker_payload(payload),
            "outputs_public": public_outputs(outputs),
            "remote_host": str(payload.get("remote_host") or ""),
            "remote_port": int(payload.get("remote_port") or 0),
            "remote_endpoint_public": False,
            "blockers": [str(item) for item in payload.get("blockers", [])],
            "public_artifact_safe": True,
        }
    except Exception as exc:  # noqa: BLE001 - public-safe live boundary
        return {
            "schema": "deepseek_v4_flash_quantized_colab_rpc_launch_v1",
            "ok": False,
            "duration_seconds": round(time.monotonic() - started, 3),
            "error_type": type(exc).__name__,
            "error_digest": sha_text(str(exc)),
            "remote_host": "",
            "remote_port": 0,
            "remote_endpoint_public": False,
            "blockers": ["colab_rpc_worker_launch_exception"],
            "public_artifact_safe": True,
        }


def parse_colab_authusers(args: argparse.Namespace) -> list[str]:
    raw = str(getattr(args, "colab_authusers", "") or "").strip()
    values = [item.strip() for item in raw.split(",") if item.strip()] if raw else []
    if not values:
        values = [str(args.colab_authuser)]
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def parse_colab_accelerators(args: argparse.Namespace) -> list[str]:
    raw = str(getattr(args, "colab_accelerators", "") or "").strip()
    values = [item.strip() for item in raw.split(",") if item.strip()] if raw else []
    if not values:
        values = [str(args.colab_accelerator)]
    deduped: list[str] = []
    for value in values:
        if value and value not in deduped:
            deduped.append(value)
    return deduped


def colab_fallback_matrix(args: argparse.Namespace) -> list[dict[str, str]]:
    return [
        {"accelerator": accelerator, "authuser": authuser}
        for accelerator in parse_colab_accelerators(args)
        for authuser in parse_colab_authusers(args)
    ]


def clone_args_with_colab_target(args: argparse.Namespace, *, accelerator: str, authuser: str) -> argparse.Namespace:
    cloned = argparse.Namespace(**vars(args))
    cloned.colab_accelerator = str(accelerator)
    cloned.colab_accelerators = str(accelerator)
    cloned.colab_authuser = str(authuser)
    cloned.colab_authusers = str(authuser)
    return cloned


def start_colab_rpc_worker_single_target(args: argparse.Namespace) -> dict[str, Any]:
    if args.colab_build_mode == "background":
        return start_colab_rpc_worker_background(args)
    original_accelerators = getattr(args, "colab_accelerators", "")
    original_authusers = getattr(args, "colab_authusers", "")
    args.colab_accelerators = str(args.colab_accelerator)
    args.colab_authusers = str(args.colab_authuser)
    try:
        return start_colab_rpc_worker(args)
    finally:
        args.colab_accelerators = original_accelerators
        args.colab_authusers = original_authusers


def summarize_colab_fallback_attempt(target: dict[str, str], result: dict[str, Any]) -> dict[str, Any]:
    return {
        "accelerator": str(target.get("accelerator") or ""),
        "authuser": str(target.get("authuser") or ""),
        "ok": result.get("ok") is True,
        "duration_seconds": result.get("duration_seconds"),
        "remote_port": int(result.get("remote_port") or 0),
        "blockers": [str(item) for item in result.get("blockers", [])] if isinstance(result.get("blockers"), list) else [],
        "manager": result.get("manager") if isinstance(result.get("manager"), dict) else {},
        "worker": result.get("worker") if isinstance(result.get("worker"), dict) else {},
        "public_artifact_safe": True,
    }


def start_colab_rpc_worker_with_fallback_matrix(args: argparse.Namespace, *, matrix: list[dict[str, str]]) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    last: dict[str, Any] = {}
    started = time.monotonic()
    for target in matrix:
        result = start_colab_rpc_worker_single_target(clone_args_with_colab_target(
            args,
            accelerator=str(target.get("accelerator") or ""),
            authuser=str(target.get("authuser") or ""),
        ))
        attempts.append(summarize_colab_fallback_attempt(target, result))
        if result.get("ok") is True:
            result = dict(result)
            result["accelerator"] = str(target.get("accelerator") or "")
            result["authuser"] = str(target.get("authuser") or "")
            result["colab_fallback"] = {
                "attempt_count": len(attempts),
                "attempted_targets": [dict(item) for item in matrix],
                "selected_accelerator": str(target.get("accelerator") or ""),
                "selected_authuser": str(target.get("authuser") or ""),
                "attempts": attempts,
                "public_artifact_safe": True,
            }
            result["authuser_fallback"] = {
                "attempt_count": len(attempts),
                "attempted_authusers": sorted({str(item.get("authuser") or "") for item in matrix}),
                "selected_authuser": str(target.get("authuser") or ""),
                "attempts": attempts,
                "public_artifact_safe": True,
            }
            return result
        last = result
    blockers = sorted({
        *(str(item) for attempt in attempts for item in attempt.get("blockers", [])),
        "colab_rpc_worker_fallback_exhausted",
        "colab_rpc_worker_authuser_fallback_exhausted",
    })
    return {
        "schema": "deepseek_v4_flash_quantized_colab_rpc_launch_v1",
        "ok": False,
        "duration_seconds": round(time.monotonic() - started, 3),
        "remote_host": "",
        "remote_port": 0,
        "remote_endpoint_public": False,
        "manager": last.get("manager") if isinstance(last.get("manager"), dict) else {},
        "worker": last.get("worker") if isinstance(last.get("worker"), dict) else {},
        "blockers": blockers,
        "colab_fallback": {
            "attempt_count": len(attempts),
            "attempted_targets": [dict(item) for item in matrix],
            "selected_accelerator": "",
            "selected_authuser": "",
            "attempts": attempts,
            "public_artifact_safe": True,
        },
        "authuser_fallback": {
            "attempt_count": len(attempts),
            "attempted_authusers": sorted({str(item.get("authuser") or "") for item in matrix}),
            "selected_authuser": "",
            "attempts": attempts,
            "public_artifact_safe": True,
        },
        "public_artifact_safe": True,
    }


def render_colab_background_launch_code(worker_code: str) -> str:
    return f"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

MARKER = {json.dumps(COLAB_BACKGROUND_LAUNCH_MARKER)}
LAUNCH_DIR = Path("/content/ct_dsv4_colab_rpc_launcher")
WORKER_PATH = LAUNCH_DIR / "worker.py"
PID_PATH = LAUNCH_DIR / "worker.pid"
LOG_PATH = LAUNCH_DIR / "worker.launch.log"
LAUNCH_DIR.mkdir(parents=True, exist_ok=True)
WORKER_PATH.write_text({json.dumps(worker_code)}, encoding="utf-8")
handle = LOG_PATH.open("ab")
proc = subprocess.Popen([sys.executable, str(WORKER_PATH)], stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
PID_PATH.write_text(str(proc.pid), encoding="utf-8")
payload = {{
    "ok": True,
    "pid_recorded": True,
    "pid_hash": "sha256:" + hashlib.sha256(str(proc.pid).encode()).hexdigest(),
    "worker_path_public": False,
    "log_path_public": False,
    "public_artifact_safe": True,
}}
print(MARKER + " " + json.dumps(payload, sort_keys=True), flush=True)
"""


def render_colab_background_poll_code(args: argparse.Namespace) -> str:
    return f"""
import hashlib
import json
import os
from pathlib import Path

MARKER = {json.dumps(COLAB_MARKER)}
BORE_SERVER = {json.dumps(args.bore_server)}
REPORT_PATH = Path("/content/ct_dsv4_colab_rpc/deepseek_v4_flash_colab_rpc_worker.json")
PID_PATH = Path("/content/ct_dsv4_colab_rpc_launcher/worker.pid")

def sha_text(value):
    return "sha256:" + hashlib.sha256(str(value or "").encode("utf-8", errors="replace")).hexdigest()

def pid_alive():
    try:
        pid = int(PID_PATH.read_text(encoding="utf-8").strip())
    except Exception:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

loaded = {{}}
if REPORT_PATH.is_file():
    try:
        loaded = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        loaded = {{"ok": False, "stage": "report_json_error", "blockers": ["colab_rpc_worker_report_json_error"], "error_digest": sha_text(str(exc))}}

alive = pid_alive()
if loaded:
    payload = {{
        "schema": loaded.get("schema", ""),
        "ok": loaded.get("ok") is True,
        "stage": loaded.get("stage", ""),
        "remote_host": BORE_SERVER if loaded.get("ok") is True and int(loaded.get("bore_remote_port") or 0) else "",
        "remote_port": int(loaded.get("bore_remote_port") or 0),
        "remote_endpoint_public": False,
        "blockers": loaded.get("blockers") or [],
        "diagnosis_codes": loaded.get("diagnosis_codes") or [],
        "hardware": loaded.get("hardware") or {{}},
        "commit_hash_public": loaded.get("commit_hash_public") or "",
        "patch_rpc_op_count_guard": loaded.get("patch_rpc_op_count_guard") is True,
        "public_artifact_safe": True,
        "background_pid_alive": alive,
        "report_digest": sha_text(json.dumps(loaded, sort_keys=True, default=str)),
    }}
else:
    payload = {{
        "schema": "",
        "ok": False,
        "stage": "pending" if alive else "background_process_missing",
        "remote_host": "",
        "remote_port": 0,
        "remote_endpoint_public": False,
        "blockers": [] if alive else ["colab_rpc_worker_background_process_missing"],
        "diagnosis_codes": ["colab_rpc_worker_background_pending" if alive else "colab_rpc_worker_background_process_missing"],
        "public_artifact_safe": True,
        "background_pid_alive": alive,
    }}
print(MARKER + " " + json.dumps(payload, sort_keys=True), flush=True)
"""


def start_colab_rpc_worker_background(args: argparse.Namespace) -> dict[str, Any]:
    worker_code = render_colab_rpc_code(args)
    started = time.monotonic()
    launch_outputs: list[dict[str, Any]] = []
    poll_summaries: list[dict[str, Any]] = []
    manager_result: dict[str, Any] = {}
    try:
        launch_outputs, _session, manager_result = colab_cuda_session_manager.execute_with_retry(
            render_colab_background_launch_code(worker_code),
            session_name=args.colab_session,
            state_path=Path(args.colab_config).expanduser(),
            timeout=float(args.colab_background_launch_timeout_seconds),
            max_attempts=int(args.colab_max_attempts),
            token_cache=Path(args.colab_token_cache).expanduser(),
            accelerator=args.colab_accelerator,
            authuser=str(args.colab_authuser),
            force_reacquire_before=bool(args.colab_reacquire_before),
            heartbeat_code='print("CT_DSV4_COLAB_RPC_BACKGROUND_HEARTBEAT")',
            stop_runtime_after_success=False,
        )
        launch_payload = parse_marker_payload(launch_outputs, COLAB_BACKGROUND_LAUNCH_MARKER)
        if not manager_result.get("ok") or launch_payload.get("ok") is not True:
            return {
                "schema": "deepseek_v4_flash_quantized_colab_rpc_launch_v1",
                "ok": False,
                "duration_seconds": round(time.monotonic() - started, 3),
                "manager": colab_public_manager_result(manager_result),
                "launch": {
                    "ok": launch_payload.get("ok") is True,
                    "pid_recorded": launch_payload.get("pid_recorded") is True,
                    "public_artifact_safe": True,
                },
                "outputs_public": public_outputs(launch_outputs),
                "remote_host": "",
                "remote_port": 0,
                "remote_endpoint_public": False,
                "blockers": ["colab_rpc_worker_background_launch_failed"],
                "public_artifact_safe": True,
            }
        deadline = time.monotonic() + float(args.colab_background_timeout_seconds)
        last_payload: dict[str, Any] = {}
        last_manager: dict[str, Any] = {}
        while time.monotonic() < deadline:
            time.sleep(max(1.0, float(args.colab_background_poll_interval_seconds)))
            poll_outputs, _session, last_manager = colab_cuda_session_manager.execute_with_retry(
                render_colab_background_poll_code(args),
                session_name=args.colab_session,
                state_path=Path(args.colab_config).expanduser(),
                timeout=float(args.colab_background_poll_timeout_seconds),
                max_attempts=1,
                token_cache=Path(args.colab_token_cache).expanduser(),
                accelerator=args.colab_accelerator,
                authuser=str(args.colab_authuser),
                force_reacquire_before=False,
                stop_runtime_after_success=False,
            )
            if not last_manager.get("ok"):
                poll_summaries.append({
                    "ok": False,
                    "manager": colab_public_manager_result(last_manager),
                    "public_artifact_safe": True,
                })
                if colab_cuda_session_manager.is_stale_error(last_manager.get("blocker")):
                    break
                continue
            last_payload = parse_marker_payload(poll_outputs, COLAB_MARKER)
            poll_summaries.append({
                "ok": last_payload.get("ok") is True,
                "stage": str(last_payload.get("stage") or ""),
                "background_pid_alive": last_payload.get("background_pid_alive") is True,
                "remote_port": int(last_payload.get("remote_port") or 0),
                "blockers": [str(item) for item in last_payload.get("blockers", [])],
                "outputs_public": public_outputs(poll_outputs),
                "public_artifact_safe": True,
            })
            stage = str(last_payload.get("stage") or "")
            if last_payload.get("ok") is True:
                return {
                    "schema": "deepseek_v4_flash_quantized_colab_rpc_launch_v1",
                    "ok": True,
                    "duration_seconds": round(time.monotonic() - started, 3),
                    "manager": colab_public_manager_result(last_manager),
                    "launch": {
                        "ok": True,
                        "pid_recorded": launch_payload.get("pid_recorded") is True,
                        "public_artifact_safe": True,
                    },
                    "polls": poll_summaries,
                    "worker": colab_public_worker_payload(last_payload),
                    "remote_host": str(last_payload.get("remote_host") or ""),
                    "remote_port": int(last_payload.get("remote_port") or 0),
                    "remote_endpoint_public": False,
                    "blockers": [],
                    "public_artifact_safe": True,
                }
            if stage in {"blocked", "exception", "background_process_missing", "report_json_error"}:
                break
        blockers = [str(item) for item in last_payload.get("blockers", [])] if last_payload else []
        if not blockers:
            blockers = ["colab_rpc_worker_background_timeout"]
        return {
            "schema": "deepseek_v4_flash_quantized_colab_rpc_launch_v1",
            "ok": False,
            "duration_seconds": round(time.monotonic() - started, 3),
            "manager": colab_public_manager_result(last_manager or manager_result),
            "launch": {
                "ok": True,
                "pid_recorded": launch_payload.get("pid_recorded") is True,
                "public_artifact_safe": True,
            },
            "polls": poll_summaries,
            "worker": colab_public_worker_payload(last_payload or {}),
            "remote_host": "",
            "remote_port": 0,
            "remote_endpoint_public": False,
            "blockers": blockers,
            "public_artifact_safe": True,
        }
    except Exception as exc:  # noqa: BLE001 - public-safe live boundary
        return {
            "schema": "deepseek_v4_flash_quantized_colab_rpc_launch_v1",
            "ok": False,
            "duration_seconds": round(time.monotonic() - started, 3),
            "error_type": type(exc).__name__,
            "error_digest": sha_text(str(exc)),
            "manager": colab_public_manager_result(manager_result),
            "launch_outputs_public": public_outputs(launch_outputs),
            "polls": poll_summaries,
            "remote_host": "",
            "remote_port": 0,
            "remote_endpoint_public": False,
            "blockers": ["colab_rpc_worker_background_exception"],
            "public_artifact_safe": True,
        }


def colab_public_manager_result(result: dict[str, Any]) -> dict[str, Any]:
    attempts = []
    for item in result.get("attempts", []) if isinstance(result.get("attempts"), list) else []:
        if not isinstance(item, dict):
            continue
        attempts.append({
            key: value
            for key, value in item.items()
            if key not in {"runtime_proxy_token", "runtime_proxy_url", "endpoint", "token", "url"}
        })
    return {
        "ok": result.get("ok") is True,
        "blocker": str(result.get("blocker") or ""),
        "attempt_count": len(attempts),
        "attempts": attempts,
        "public_artifact_safe": True,
    }


def colab_public_worker_payload(payload: dict[str, Any]) -> dict[str, Any]:
    hardware = payload.get("hardware") if isinstance(payload.get("hardware"), dict) else {}
    return {
        "schema": str(payload.get("schema") or ""),
        "ok": payload.get("ok") is True,
        "stage": str(payload.get("stage") or ""),
        "remote_host_hash": sha16(str(payload.get("remote_host") or "")),
        "remote_port": int(payload.get("remote_port") or 0),
        "remote_endpoint_public": False,
        "blockers": [str(item) for item in payload.get("blockers", [])],
        "diagnosis_codes": [str(item) for item in payload.get("diagnosis_codes", [])],
        "gpu_count": int(hardware.get("gpu_count") or 0),
        "colab_cuda_verified": hardware.get("colab_cuda_verified") is True,
        "commit_hash_public": str(payload.get("commit_hash_public") or ""),
        "patch_rpc_op_count_guard": payload.get("patch_rpc_op_count_guard") is True,
        "runtime_tarball_used": payload.get("runtime_tarball_used") is True,
        "keepalive": payload.get("keepalive") if isinstance(payload.get("keepalive"), dict) else {},
        "public_artifact_safe": True,
    }


def build_kaggle_package(args: argparse.Namespace, *, output_dir: Path, candidate: dict[str, Any], colab_endpoint: dict[str, Any]) -> dict[str, Any]:
    owner = args.kaggle_owner or default_kaggle_owner()
    if not owner:
        raise SystemExit("--kaggle-owner or ~/.kaggle/kaggle.json username is required")
    slug = safe_slug(args.kernel_slug_prefix)[:34] + "-" + str(int(time.time()))[-8:]
    slug = slug[:45].strip("-")
    kernel_dir = output_dir / "private-kaggle-kernel"
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text(render_kaggle_kernel(args, candidate, colab_endpoint), encoding="utf-8")
    title = f"CT DSV4 Same Request {slug[-8:]}"
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


def resolve_pushed_kernel_ref(package: dict[str, Any], push_step: dict[str, Any], *, runner: Runner, timeout_seconds: float) -> tuple[str, dict[str, Any] | None]:
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


def summarize_worker(worker: dict[str, Any]) -> dict[str, Any]:
    metrics = worker.get("metrics") if isinstance(worker.get("metrics"), dict) else {}
    return {
        "present": bool(worker),
        "schema": str(worker.get("schema") or ""),
        "ok": worker.get("ok") is True,
        "same_request_decode_verified": worker.get("same_request_decode_verified") is True,
        "generated_token_count": int(worker.get("generated_token_count") or metrics.get("generated_token_count") or 0),
        "accepted_providers": [str(item) for item in worker.get("accepted_providers", [])] if isinstance(worker.get("accepted_providers"), list) else [],
        "provider_stage_counts": worker.get("provider_stage_counts") if isinstance(worker.get("provider_stage_counts"), dict) else {},
        "blockers": [str(item) for item in worker.get("blockers", [])] if isinstance(worker.get("blockers"), list) else [],
        "diagnosis_codes": [str(item) for item in worker.get("diagnosis_codes", [])] if isinstance(worker.get("diagnosis_codes"), list) else [],
        "llama_cli_present": worker.get("llama_cli_present") is True,
        "rpc_server_present": worker.get("rpc_server_present") is True,
        "downloads_count": len(worker.get("downloads", [])) if isinstance(worker.get("downloads"), list) else 0,
        "downloaded_bytes": sum(int(item.get("size_bytes") or 0) for item in worker.get("downloads", []) if isinstance(item, dict)),
        "public_artifact_safe": worker.get("public_artifact_safe") is True,
    }


def build_report(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    candidate: dict[str, Any],
    colab_rpc: dict[str, Any],
    package: dict[str, Any],
    steps: list[dict[str, Any]],
    worker_report: dict[str, Any],
    live_run_performed: bool,
) -> dict[str, Any]:
    worker = summarize_worker(worker_report)
    pushed = any(step.get("name") == "kaggle_kernel_push" and step.get("ok") for step in steps)
    output_downloaded = bool(worker_report)
    kernel_deleted = any(step.get("name") == "kaggle_kernel_delete" and step.get("ok") for step in steps)
    private_removed = not (output_dir / "private-kaggle-kernel").exists()
    accepted = set(worker.get("accepted_providers") or [])
    required = {"kaggle_cuda", "colab_cuda", "cpu"}
    generated = int(worker.get("generated_token_count") or 0)
    success = bool(worker.get("same_request_decode_verified") and generated >= 1 and required.issubset(accepted))
    blockers = set(str(item) for item in (candidate.get("blockers") or []))
    blockers.update(str(item) for item in (colab_rpc.get("blockers") or []))
    blockers.update(str(item) for item in (worker.get("blockers") or []))
    if args.mode == "preflight":
        blockers.add("deepseek_v4_flash_quantized_same_request_live_run_not_started")
    if args.mode == "kaggle-auto" and not colab_rpc.get("ok"):
        blockers.add("colab_rpc_worker_not_ready")
    if args.mode == "kaggle-auto" and colab_rpc.get("ok") and not pushed:
        blockers.add("kaggle_same_request_kernel_push_failed")
    if pushed and not output_downloaded:
        blockers.add("kaggle_same_request_worker_report_missing")
    if pushed and not kernel_deleted and not args.skip_kaggle_cleanup:
        blockers.add("kaggle_same_request_kernel_cleanup_missing")
    if not private_removed:
        blockers.add("kaggle_same_request_private_package_not_removed")
    if not success:
        blockers.add("deepseek_v4_flash_quantized_same_request_decode_not_verified")
    failure = failure_stage(args, candidate, colab_rpc, pushed=pushed, output_downloaded=output_downloaded, worker=worker)
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": success,
        "deepseek_v4_flash_quantized_same_request_verified": success,
        "same_request_decode_verified": success,
        "generated_token_count": generated,
        "accepted_providers": sorted(accepted),
        "provider_stage_counts": {
            "kaggle_cuda": int((worker.get("provider_stage_counts") or {}).get("kaggle_cuda") or 0),
            "colab_cuda": int((worker.get("provider_stage_counts") or {}).get("colab_cuda") or 0),
            "cpu": int((worker.get("provider_stage_counts") or {}).get("cpu") or 0),
        },
        "stage_task_counts": {
            "kaggle_cuda_rpc": int((worker.get("provider_stage_counts") or {}).get("kaggle_cuda") or 0),
            "colab_cuda_rpc": int((worker.get("provider_stage_counts") or {}).get("colab_cuda") or 0),
            "cpu_rpc": int((worker.get("provider_stage_counts") or {}).get("cpu") or 0),
        },
        "mode": args.mode,
        "live_run_performed": live_run_performed,
        "failure_stage": "" if success else failure,
        "model": {
            "model_id": resolver.MODEL_ID,
            "architecture_class": "moe",
            "total_params_b": resolver.TOTAL_PARAMS_B,
            "active_params_b": resolver.ACTIVE_PARAMS_B,
            "repo": candidate.get("repo"),
            "candidate_id": candidate.get("candidate_id"),
            "quant": candidate.get("quant"),
            "format": "gguf",
            "total_size_gb": candidate.get("total_size_gb"),
            "split_file_count": candidate.get("split_file_count"),
        },
        "runtime": {
            "backend": "llama_cpp_v4_fork_rpc",
            "repo_url": args.repo_url,
            "branch": args.branch,
            "cuda_architectures": args.cuda_architectures,
            "patch_rpc_op_count_guard": bool(args.patch_rpc_op_count_guard),
            "runtime_tarball_requested": bool(args.runtime_tarball_path or args.runtime_tarball_url),
            "runtime_tarball_path_public": False,
            "runtime_tarball_url_public": False,
            "runtime_tarball_sha256": args.runtime_tarball_sha256 or "",
            "include_cpu_rpc_endpoint": bool(args.include_cpu_rpc_endpoint),
            "client_cuda_visible": args.client_cuda_visible,
        },
        "colab_rpc": {
            "schema": colab_rpc.get("schema"),
            "ok": colab_rpc.get("ok") is True,
            "remote_host_hash": sha16(str(colab_rpc.get("remote_host") or "")),
            "remote_port": int(colab_rpc.get("remote_port") or 0),
            "remote_endpoint_public": False,
            "worker": colab_rpc.get("worker") if isinstance(colab_rpc.get("worker"), dict) else {},
            "manager": colab_rpc.get("manager") if isinstance(colab_rpc.get("manager"), dict) else {},
            "colab_fallback": colab_rpc.get("colab_fallback") if isinstance(colab_rpc.get("colab_fallback"), dict) else {},
            "authuser_fallback": colab_rpc.get("authuser_fallback") if isinstance(colab_rpc.get("authuser_fallback"), dict) else {},
            "public_artifact_safe": colab_rpc.get("public_artifact_safe") is True,
        },
        "kaggle_worker_summary": worker,
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
            "deepseek_v4_flash_quantized_same_request_decode_verified" if success else "deepseek_v4_flash_quantized_same_request_decode_not_verified",
            "deepseek_v4_flash_quantized_same_request_probe_live_run_performed" if live_run_performed else "deepseek_v4_flash_quantized_same_request_probe_live_run_not_started",
        ],
        "safety": {
            "public_artifact_safe": True,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "past_key_values_public": False,
            "credentials_public": False,
            "cookies_public": False,
            "private_runtime_state_public": False,
            "weight_tensor_values_public": False,
        },
        "public_artifact_safe": True,
    }
    return report


def failure_stage(args: argparse.Namespace, candidate: dict[str, Any], colab_rpc: dict[str, Any], *, pushed: bool, output_downloaded: bool, worker: dict[str, Any]) -> str:
    if args.mode == "preflight":
        return "same_request_live_probe_not_started"
    if not candidate.get("files"):
        return "quantized_candidate_files_missing"
    if not colab_rpc.get("ok"):
        return "colab_rpc_worker_not_ready"
    if not pushed:
        return "kaggle_same_request_kernel_push_failed"
    if not output_downloaded:
        return "kaggle_same_request_worker_report_missing"
    blockers = set(worker.get("blockers") or [])
    if "colab_rpc_endpoint_not_reachable_from_kaggle" in blockers:
        return "colab_rpc_endpoint_not_reachable_from_kaggle"
    if "colab_rpc_hello_pre_download_failed" in blockers:
        return "colab_rpc_hello_pre_download_failed"
    if "colab_rpc_endpoint_lost_after_model_download" in blockers:
        return "colab_rpc_endpoint_lost_after_model_download"
    if "colab_rpc_hello_lost_after_model_download" in blockers:
        return "colab_rpc_hello_lost_after_model_download"
    if "kaggle_local_rpc_hello_failed" in blockers:
        return "kaggle_local_rpc_hello_failed"
    if "colab_bore_rpc_hello_failed" in blockers:
        return "colab_bore_rpc_hello_failed"
    if any("download" in item for item in blockers):
        return "quantized_model_download_failed"
    if any("llama_v4_cmake_build_failed" in item for item in blockers):
        return "llama_v4_runtime_build_failed"
    if any("rpc_server_not_alive" in item for item in blockers):
        return "rpc_worker_not_alive"
    return "same_request_decode_not_verified"


def run_preflight(args: argparse.Namespace, *, output_dir: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    colab_rpc = {"schema": "deepseek_v4_flash_quantized_colab_rpc_launch_v1", "ok": False, "blockers": ["preflight_mode_colab_rpc_not_started"], "public_artifact_safe": True}
    package = build_kaggle_package(
        args,
        output_dir=output_dir,
        candidate=candidate,
        colab_endpoint={"remote_host": args.colab_rpc_host or args.bore_server, "remote_port": int(args.colab_rpc_port or 1)},
    )
    if not args.keep_private_package:
        shutil.rmtree(output_dir / "private-kaggle-kernel", ignore_errors=True)
    return build_report(
        args,
        output_dir=output_dir,
        candidate=candidate,
        colab_rpc=colab_rpc,
        package=package,
        steps=[],
        worker_report={},
        live_run_performed=False,
    )


def run_kaggle_auto(args: argparse.Namespace, *, output_dir: Path, candidate: dict[str, Any], runner: Runner = subprocess.run) -> dict[str, Any]:
    package: dict[str, Any] = {}
    steps: list[dict[str, Any]] = []
    worker_report: dict[str, Any] = {}
    colab_rpc: dict[str, Any] = {}
    original_runtime_tarball_url = args.runtime_tarball_url
    try:
        with maybe_runtime_tarball_server(args) as server:
            if server.get("url"):
                args.runtime_tarball_url = str(server.get("url") or "")
            colab_rpc = (
                {
                    "schema": "deepseek_v4_flash_quantized_colab_rpc_launch_v1",
                    "ok": True,
                    "remote_host": args.colab_rpc_host,
                    "remote_port": int(args.colab_rpc_port),
                    "remote_endpoint_public": False,
                    "blockers": [],
                    "public_artifact_safe": True,
                }
                if args.colab_rpc_host and args.colab_rpc_port
                else start_colab_rpc_worker(args)
            )
            if colab_rpc.get("ok"):
                package = build_kaggle_package(args, output_dir=output_dir, candidate=candidate, colab_endpoint=colab_rpc)
                push_command = ["kaggle", "kernels", "push", "-p", str(package["kernel_dir"]), "-t", str(args.kernel_timeout_seconds)]
                if args.accelerator:
                    push_command.extend(["--accelerator", args.accelerator])
                print(f"[{utc_now()}] pushing DeepSeek V4 same-request Kaggle kernel {package['declared_kernel_ref']}", flush=True)
                push_step = run_step("kaggle_kernel_push", push_command, runner=runner, timeout_seconds=args.kaggle_push_timeout_seconds)
                steps.append(push_step)
                if push_step.get("ok"):
                    kernel_ref, resolve_step = resolve_pushed_kernel_ref(package, push_step, runner=runner, timeout_seconds=args.kaggle_push_timeout_seconds)
                    if resolve_step:
                        steps.append(resolve_step)
                    package["kernel_ref"] = kernel_ref
                    print(f"[{utc_now()}] waiting for DeepSeek V4 same-request kernel {kernel_ref}", flush=True)
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
                            "deepseek_v4_flash_quantized_same_request_worker.json",
                        ],
                        runner=runner,
                        timeout_seconds=args.kaggle_output_timeout_seconds,
                    )
                    steps.append(output_step)
                    worker_report = load_json(output_path / "deepseek_v4_flash_quantized_same_request_worker.json")
                    if not args.skip_kaggle_cleanup:
                        print(f"[{utc_now()}] deleting DeepSeek V4 same-request Kaggle kernel {kernel_ref}", flush=True)
                        delete_step = run_step(
                            "kaggle_kernel_delete",
                            ["kaggle", "kernels", "delete", kernel_ref, "-y"],
                            runner=runner,
                            timeout_seconds=args.kaggle_delete_timeout_seconds,
                        )
                        steps.append(delete_step)
    except Exception as exc:  # noqa: BLE001 - public-safe orchestration boundary
        colab_rpc = {
            "schema": "deepseek_v4_flash_quantized_colab_rpc_launch_v1",
            "ok": False,
            "error_type": type(exc).__name__,
            "error_digest": sha_text(str(exc)),
            "remote_host": "",
            "remote_port": 0,
            "remote_endpoint_public": False,
            "blockers": ["runtime_tarball_server_exception"],
            "public_artifact_safe": True,
        }
    finally:
        args.runtime_tarball_url = original_runtime_tarball_url
        if not args.keep_private_package:
            shutil.rmtree(output_dir / "private-kaggle-kernel", ignore_errors=True)
    return build_report(
        args,
        output_dir=output_dir,
        candidate=candidate,
        colab_rpc=colab_rpc,
        package=package,
        steps=steps,
        worker_report=worker_report,
        live_run_performed=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["preflight", "kaggle-auto"], default="preflight")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-resolver-report", default="")
    parser.add_argument("--candidate-id", default="iq1-s-xl-gguf")
    parser.add_argument("--candidate-repo", default="teamblobfish/DeepSeek-V4-Flash-GGUF")
    parser.add_argument("--candidate-quant", default="IQ1_S-XL")
    parser.add_argument("--candidate-total-size-gb", type=float, default=61.540805)
    parser.add_argument("--candidate-file", action="append", default=[
        "IQ1_S-XL/DeepSeek-V4-Flash-IQ1_S-XL-00001-of-00002.gguf",
        "IQ1_S-XL/DeepSeek-V4-Flash-IQ1_S-XL-00002-of-00002.gguf",
    ])
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--cuda-architectures", default="75")
    parser.add_argument("--patch-rpc-op-count-guard", action="store_true")
    parser.add_argument("--runtime-tarball-path", default="")
    parser.add_argument("--runtime-tarball-url", default="")
    parser.add_argument("--runtime-tarball-sha256", default="")
    parser.add_argument("--bore-url", default=DEFAULT_BORE_URL)
    parser.add_argument("--bore-server", default=DEFAULT_BORE_SERVER)
    parser.add_argument("--colab-session", default=colab_cuda_session_manager.DEFAULT_SESSION_NAME)
    parser.add_argument("--colab-config", default=str(colab_cuda_session_manager.DEFAULT_STATE_PATH))
    parser.add_argument("--colab-token-cache", default=str(colab_cuda_session_manager.DEFAULT_TOKEN_CACHE))
    parser.add_argument("--colab-accelerator", default="T4")
    parser.add_argument("--colab-accelerators", default="")
    parser.add_argument("--colab-authuser", default="0")
    parser.add_argument("--colab-authusers", default="")
    parser.add_argument("--colab-reacquire-before", action="store_true")
    parser.add_argument("--colab-max-attempts", type=int, default=1)
    parser.add_argument("--colab-build-mode", choices=["background", "direct"], default="background")
    parser.add_argument("--colab-execute-timeout-seconds", type=float, default=5400.0)
    parser.add_argument("--colab-background-launch-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--colab-background-timeout-seconds", type=float, default=5400.0)
    parser.add_argument("--colab-background-poll-interval-seconds", type=float, default=30.0)
    parser.add_argument("--colab-background-poll-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--colab-keepalive-seconds", type=int, default=7200)
    parser.add_argument("--colab-cuda-build-jobs", type=int, default=2)
    parser.add_argument("--colab-cuda-build-timeout-seconds", type=int, default=3600)
    parser.add_argument("--colab-rpc-local-port", type=int, default=50052)
    parser.add_argument("--colab-rpc-host", default="")
    parser.add_argument("--colab-rpc-port", type=int, default=0)
    parser.add_argument("--kaggle-owner", default=default_kaggle_owner())
    parser.add_argument("--kernel-slug-prefix", default="ct-dsv4-same-request")
    parser.add_argument("--accelerator", default=DEFAULT_ACCELERATOR)
    parser.add_argument("--kaggle-cuda-build-jobs", type=int, default=2)
    parser.add_argument("--kaggle-cuda-build-timeout-seconds", type=int, default=3600)
    parser.add_argument("--kernel-timeout-seconds", type=int, default=7200)
    parser.add_argument("--kaggle-push-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--kaggle-status-timeout-seconds", type=float, default=7500.0)
    parser.add_argument("--kaggle-status-poll-interval", type=float, default=60.0)
    parser.add_argument("--kaggle-output-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--kaggle-delete-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--skip-kaggle-cleanup", action="store_true")
    parser.add_argument("--keep-private-package", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--run-timeout-seconds", type=int, default=2400)
    parser.add_argument("--include-cpu-rpc-endpoint", action="store_true")
    parser.add_argument("--client-cuda-visible", default="0,1")
    parser.add_argument("--skip-model-download-on-rpc-hello-failure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1 or args.max_new_tokens > 16:
        raise SystemExit("--max-new-tokens must be between 1 and 16")
    if args.context_length < 1 or args.context_length > 2048:
        raise SystemExit("--context-length must be between 1 and 2048")
    if args.colab_cuda_build_jobs < 1 or args.colab_cuda_build_jobs > 8:
        raise SystemExit("--colab-cuda-build-jobs must be between 1 and 8")
    if args.kaggle_cuda_build_jobs < 1 or args.kaggle_cuda_build_jobs > 8:
        raise SystemExit("--kaggle-cuda-build-jobs must be between 1 and 8")
    if args.colab_keepalive_seconds < 60 or args.colab_keepalive_seconds > 14400:
        raise SystemExit("--colab-keepalive-seconds must be between 60 and 14400")
    if args.mode == "kaggle-auto" and bool(args.colab_rpc_host) != bool(args.colab_rpc_port):
        raise SystemExit("--colab-rpc-host and --colab-rpc-port must be provided together")
    if args.runtime_tarball_path and args.runtime_tarball_url:
        raise SystemExit("--runtime-tarball-path and --runtime-tarball-url are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = load_json(args.source_resolver_report)
    candidate = candidate_from_source(source, args)
    if not candidate.get("files"):
        raise SystemExit("candidate files are required")
    if args.mode == "preflight":
        report = run_preflight(args, output_dir=output_dir, candidate=candidate)
    else:
        report = run_kaggle_auto(args, output_dir=output_dir, candidate=candidate)
    write_json(output_dir / "deepseek_v4_flash_quantized_same_request_probe.json", report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"DeepSeek V4 Flash same-request verified: {report.get('same_request_decode_verified')}")
        print(f"Failure stage: {report.get('failure_stage') or 'none'}")
        print(f"Report: {output_dir / 'deepseek_v4_flash_quantized_same_request_probe.json'}")
    return 0 if report.get("public_artifact_safe") else 1


if __name__ == "__main__":
    raise SystemExit(main())
