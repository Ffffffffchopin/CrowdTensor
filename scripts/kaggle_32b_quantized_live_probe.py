#!/usr/bin/env python3
"""Run a bounded private Kaggle probe for a 32B-class quantized GGUF model."""

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


SCHEMA = "kaggle_32b_quantized_live_probe_v1"
DEFAULT_OUTPUT_DIR = "dist/kaggle-32b-quantized-live-probe"
DEFAULT_OWNER = ""
DEFAULT_ACCELERATOR = "NvidiaTeslaT4"
DEFAULT_REPO = "Qwen/Qwen2.5-32B-Instruct-GGUF"
DEFAULT_QUANT = "q2_k"
DEFAULT_LLAMA_RELEASE = "b9728"
KAGGLE_TABLE_SPLIT = re.compile(r"\s{2,}")
KAGGLE_CODE_URL = re.compile(r"https://www\.kaggle\.com/code/([^/\s]+)/([^/\s]+)")
Runner = Callable[..., subprocess.CompletedProcess[str]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def default_kaggle_owner() -> str:
    if os.environ.get("KAGGLE_USERNAME"):
        return str(os.environ["KAGGLE_USERNAME"])
    config = Path.home() / ".kaggle" / "kaggle.json"
    if config.is_file():
        try:
            loaded = json.loads(config.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return DEFAULT_OWNER
        if isinstance(loaded, dict):
            return str(loaded.get("username") or DEFAULT_OWNER)
    return DEFAULT_OWNER


def safe_slug(value: str, *, default: str = "ct32bprobe") -> str:
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


def shell_command(command: list[Any]) -> str:
    return shlex.join([str(part) for part in command])


def safe_tail(text: str, limit: int = 1600) -> str:
    redacted = str(text or "")[-limit:]
    for fragment in [
        "KAGGLE_KEY",
        "KAGGLE_USERNAME",
        "Bearer ",
        "CROWDTENSOR_MINER_TOKEN=",
        "CROWDTENSOR_OBSERVER_TOKEN=",
        "CROWDTENSOR_ADMIN_TOKEN=",
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
        print(f"[{utc_now()}] status attempt={attempts} status={status}", flush=True)
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


def quantized_filenames(quant: str) -> list[str]:
    normalized = quant.lower().strip()
    if normalized == "q2_k":
        return [f"qwen2.5-32b-instruct-q2_k-{index:05d}-of-00004.gguf" for index in range(1, 5)]
    if normalized == "q3_k_m":
        return [f"qwen2.5-32b-instruct-q3_k_m-{index:05d}-of-00005.gguf" for index in range(1, 6)]
    if normalized == "q4_k_m":
        return [f"qwen2.5-32b-instruct-q4_k_m-{index:05d}-of-00005.gguf" for index in range(1, 6)]
    raise SystemExit("--quant must be one of q2_k, q3_k_m, q4_k_m")


def render_kernel(args: argparse.Namespace, filenames: list[str]) -> str:
    filenames_json = json.dumps(filenames)
    repo_json = json.dumps(args.model_repo)
    llama_release_json = json.dumps(args.llama_release)
    return f'''from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "{SCHEMA}"
MODEL_REPO = {repo_json}
FILENAMES = {filenames_json}
QUANT = "{args.quant}"
MAX_NEW_TOKENS = {int(args.max_new_tokens)}
CONTEXT_LENGTH = {int(args.context_length)}
LLAMA_RELEASE = {llama_release_json}
BACKEND = "{args.backend}"
CUDA_ARCHITECTURES = "{args.cuda_architectures}"
CUDA_BUILD_JOBS = {int(args.cuda_build_jobs)}
CUDA_BUILD_TIMEOUT_SECONDS = {int(args.cuda_build_timeout_seconds)}
RUN_TIMEOUT_SECONDS = {int(args.run_timeout_seconds)}
OUT = Path("/kaggle/working")
TEMP = Path("/kaggle/temp/ct_32b_probe")
PROMPT_TEXT = "CrowdTensor bounded public safe probe."
REPORT_PATH = OUT / "ct_32b_probe_report.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def safe_tail(value: str, limit: int = 1800) -> str:
    text = str(value or "")[-limit:]
    for fragment in [PROMPT_TEXT, "KAGGLE_KEY", "KAGGLE_USERNAME", "Bearer "]:
        text = text.replace(fragment, "<redacted>")
    return text


def disk_snapshot(path: Path = TEMP):
    try:
        usage = shutil.disk_usage(str(path if path.exists() else Path("/kaggle")))
        return {{"total_bytes": int(usage.total), "used_bytes": int(usage.used), "free_bytes": int(usage.free)}}
    except Exception as exc:
        return {{"error_type": type(exc).__name__, "error_digest": sha_text(str(exc))}}


def run(command, *, timeout=1200, env=None, stdout_public=False):
    started = time.monotonic()
    try:
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {{"ok": False, "error": "timeout", "duration_seconds": round(time.monotonic() - started, 3), "command_public": public_command(command)}}, "", ""
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    step = {{
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "stdout_digest": sha_text(stdout),
        "stdout_chars": len(stdout),
        "stdout_public": bool(stdout_public),
        "stderr_tail": safe_tail(stderr),
        "stderr_digest": sha_text(stderr),
        "stderr_chars": len(stderr),
        "command_public": public_command(command),
    }}
    if stdout_public:
        step["stdout_tail"] = safe_tail(stdout)
    return step, stdout, stderr


def public_command(command):
    public = []
    for item in command:
        text = str(item)
        if text == PROMPT_TEXT:
            public.append("<prompt-redacted>")
        elif text.startswith("/kaggle/temp/ct_32b_probe/models"):
            public.append("<model-path-redacted>")
        else:
            public.append(text)
    return public


report = {{
    "schema": SCHEMA,
    "ok": False,
    "started_at": utc_now(),
    "updated_at": utc_now(),
    "stage": "start",
    "model": {{
        "repo": MODEL_REPO,
        "parameter_count_b": 32,
        "quant": QUANT,
        "split_file_count": len(FILENAMES),
        "filenames": FILENAMES,
        "model_paths_public": False,
    }},
    "runtime": {{
        "backend": BACKEND,
        "llama_release": LLAMA_RELEASE,
        "gpu_offload_requested": BACKEND == "source-cuda",
        "cross_kernel_sharded": False,
    }},
    "safety": {{
        "public_artifact_safe": True,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "activation_public": False,
        "kv_cache_public": False,
        "credentials_public": False,
    }},
    "diagnosis_codes": [],
    "blockers": [],
}}


def write_report(stage: str, **updates):
    report["stage"] = stage
    report["updated_at"] = utc_now()
    report.update(updates)
    tmp = REPORT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    tmp.replace(REPORT_PATH)


def probe_hardware():
    step, stdout, _ = run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ], timeout=60, stdout_public=True)
    devices = []
    if step.get("ok"):
        for line in stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 5:
                try:
                    devices.append({{
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_total_mb": int(float(parts[2])),
                        "memory_used_mb": int(float(parts[3])),
                        "memory_free_mb": int(float(parts[4])),
                    }})
                except ValueError:
                    pass
    return {{
        "provider": "kaggle",
        "kaggle_gpu_verified": bool(devices),
        "gpu_count": len(devices),
        "gpu_names": [item["name"] for item in devices],
        "devices": devices,
        "nvidia_smi_step": step,
    }}


def download(url: str, path: Path, *, timeout=1800):
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    bytes_written = 0
    with urllib.request.urlopen(url, timeout=timeout) as response:
        expected = int(response.headers.get("Content-Length") or response.headers.get("x-linked-size") or 0)
        with path.open("wb") as handle:
            last_reported = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                bytes_written += len(chunk)
                if bytes_written - last_reported >= 256 * 1024 * 1024:
                    last_reported = bytes_written
                    write_report(
                        "model_download_progress",
                        current_download={{"filename": path.name, "bytes_written": bytes_written, "expected_bytes": expected}},
                        disk=disk_snapshot(path.parent),
                    )
    return {{
        "filename": path.name,
        "size_bytes": int(path.stat().st_size),
        "size_mb": int(path.stat().st_size // (1024 * 1024)),
        "expected_bytes": expected,
        "duration_seconds": round(time.monotonic() - started, 3),
    }}


def find_binary(root: Path, names):
    for name in names:
        for path in root.rglob(name):
            if path.is_file():
                path.chmod(path.stat().st_mode | 0o111)
                return path
    return None


def env_for_binary(binary: Path):
    env = os.environ.copy()
    dirs = [str(binary.parent)]
    sibling = binary.parent.parent / "lib"
    if sibling.is_dir():
        dirs.append(str(sibling))
    current = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(dirs + ([current] if current else []))
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    return env


def prepare_llama():
    if BACKEND == "source-cuda":
        return prepare_llama_source_cuda()
    archive_url = "https://github.com/ggml-org/llama.cpp/releases/download/" + LLAMA_RELEASE + "/llama-" + LLAMA_RELEASE + "-bin-ubuntu-x64.tar.gz"
    archive = TEMP / "llama.tar.gz"
    info = download(archive_url, archive, timeout=900)
    extract = TEMP / "llama-bin"
    extract.mkdir(parents=True, exist_ok=True)
    step, _, _ = run(["tar", "-xzf", str(archive), "-C", str(extract)], timeout=300, stdout_public=True)
    cli = find_binary(extract, ["llama-cli", "main"])
    version_step = {{"ok": False}}
    if cli:
        version_step, _, _ = run([str(cli), "--version"], timeout=60, env=env_for_binary(cli), stdout_public=True)
    return {{
        "ok": bool(step.get("ok") and cli and version_step.get("ok")),
        "archive_download": info,
        "extract_step": step,
        "llama_cli_present": bool(cli),
        "llama_cli": str(cli) if cli else "",
        "version_step": version_step,
    }}


def prepare_llama_source_cuda():
    source_url = "https://github.com/ggml-org/llama.cpp/archive/refs/tags/" + LLAMA_RELEASE + ".tar.gz"
    archive = TEMP / "llama-source.tar.gz"
    info = download(source_url, archive, timeout=900)
    source_root = TEMP / "llama-source"
    source_root.mkdir(parents=True, exist_ok=True)
    extract_step, _, _ = run(["tar", "-xzf", str(archive), "-C", str(source_root), "--strip-components", "1"], timeout=300, stdout_public=True)
    build_dir = TEMP / "llama-build-cuda"
    configure_command = [
        "cmake",
        "-S",
        str(source_root),
        "-B",
        str(build_dir),
        "-DGGML_CUDA=ON",
        "-DLLAMA_CURL=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_CUDA_NO_VMM=ON",
    ]
    if CUDA_ARCHITECTURES:
        configure_command.append("-DCMAKE_CUDA_ARCHITECTURES=" + CUDA_ARCHITECTURES)
    configure_step, _, _ = run(configure_command, timeout=900, stdout_public=True)
    write_report("llama_source_cuda_configure_complete", llama={{"configure_step": configure_step}}, disk=disk_snapshot())
    build_step, _, _ = run([
        "cmake",
        "--build",
        str(build_dir),
        "--config",
        "Release",
        "-j",
        str(max(1, int(CUDA_BUILD_JOBS or 1))),
        "--target",
        "llama-cli",
    ], timeout=max(1, int(CUDA_BUILD_TIMEOUT_SECONDS or 2400)), stdout_public=True)
    cli = find_binary(build_dir, ["llama-cli", "main"])
    version_step = {{"ok": False}}
    if cli:
        version_step, _, _ = run([str(cli), "--version"], timeout=60, env=env_for_binary(cli), stdout_public=True)
    return {{
        "ok": bool(extract_step.get("ok") and configure_step.get("ok") and build_step.get("ok") and cli and version_step.get("ok")),
        "source_download": info,
        "extract_step": extract_step,
        "configure_step": configure_step,
        "build_step": build_step,
        "llama_cli_present": bool(cli),
        "llama_cli": str(cli) if cli else "",
        "version_step": version_step,
        "cuda_architectures": CUDA_ARCHITECTURES,
        "cuda_build_jobs": CUDA_BUILD_JOBS,
    }}


try:
    TEMP.mkdir(parents=True, exist_ok=True)
    write_report("hardware_probe_start", disk=disk_snapshot())
    hardware = probe_hardware()
    write_report("hardware_probe_complete", hardware=hardware, disk=disk_snapshot())
    llama = prepare_llama()
    write_report("llama_prepare_complete", llama=llama, disk=disk_snapshot())
    if not llama.get("ok"):
        report["blockers"].append("llama_cpp_release_prepare_failed")
        report["diagnosis_codes"].append("llama_cpp_release_prepare_failed")
        write_report("blocked_llama_prepare")
    else:
        model_dir = TEMP / "models" / QUANT
        downloads = []
        for filename in FILENAMES:
            write_report("model_download_start", current_download={{"filename": filename}}, disk=disk_snapshot(model_dir))
            url = "https://huggingface.co/" + MODEL_REPO + "/resolve/main/" + filename
            downloads.append(download(url, model_dir / filename, timeout=1800))
            write_report("model_download_file_complete", downloads=downloads, disk=disk_snapshot(model_dir))
        first_model = model_dir / FILENAMES[0]
        prompt_path = TEMP / "prompt.txt"
        prompt_path.write_text(PROMPT_TEXT + "\\n", encoding="utf-8")
        gpu_count = int((report.get("hardware") or {{}}).get("gpu_count") or 0)
        command = [
            str(llama["llama_cli"]),
            "-m",
            str(first_model),
            "-f",
            str(prompt_path),
            "-n",
            str(MAX_NEW_TOKENS),
            "-c",
            str(CONTEXT_LENGTH),
            "-ngl",
            "99" if BACKEND == "source-cuda" else "0",
            "-t",
            "4",
            "--no-display-prompt",
            "--simple-io",
            "--log-disable",
            "-no-cnv",
        ]
        if BACKEND == "source-cuda" and gpu_count >= 2:
            command.extend(["-ts", ",".join(["1"] * gpu_count)])
        write_report("llama_run_start", downloads=downloads, runner_step={{"pending": True, "command_public": public_command(command)}}, disk=disk_snapshot(model_dir))
        run_started = time.monotonic()
        env = env_for_binary(Path(llama["llama_cli"]))
        env["CUDA_VISIBLE_DEVICES"] = "0,1" if BACKEND == "source-cuda" and gpu_count >= 2 else ("0" if BACKEND == "source-cuda" else "")
        step, stdout, stderr = run(command, timeout=RUN_TIMEOUT_SECONDS, env=env, stdout_public=False)
        wall = round(time.monotonic() - run_started, 3)
        generated_token_count = 1 if step.get("ok") and stdout.strip() else 0
        report["ok"] = bool(step.get("ok") and generated_token_count > 0)
        report["diagnosis_codes"].extend([
            "kaggle_32b_gpu_hardware_verified" if hardware.get("kaggle_gpu_verified") else "kaggle_32b_gpu_hardware_not_verified",
            "kaggle_32b_model_download_complete",
            "kaggle_32b_one_token_generation_verified" if report["ok"] else "kaggle_32b_one_token_generation_not_verified",
        ])
        if not report["ok"]:
            report["blockers"].append("kaggle_32b_one_token_generation_failed")
        write_report(
            "complete" if report["ok"] else "blocked_generation",
            downloads=downloads,
            runner_step=step,
            metrics={{
                "generated_token_count": generated_token_count,
                "max_new_tokens": MAX_NEW_TOKENS,
                "wall_time_seconds": wall,
                "tokens_per_second": round(generated_token_count / wall, 6) if wall > 0 and generated_token_count else 0.0,
                "output_digest": sha_text(stdout),
            }},
            disk=disk_snapshot(model_dir),
        )
except Exception as exc:
    report["ok"] = False
    report["error_type"] = type(exc).__name__
    report["error_digest"] = sha_text(str(exc))
    report["blockers"].append("kaggle_32b_probe_exception")
    report["diagnosis_codes"].append("kaggle_32b_probe_exception")
    write_report("exception")
finally:
    try:
        if TEMP.exists():
            shutil.rmtree(TEMP)
            report["temp_cleanup"] = {{"ok": True, "path_public": False}}
    except Exception as exc:
        report["temp_cleanup"] = {{"ok": False, "error_type": type(exc).__name__, "error_digest": sha_text(str(exc))}}
    write_report("final")
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
    filenames = quantized_filenames(args.quant)
    (kernel_dir / "kernel.py").write_text(render_kernel(args, filenames), encoding="utf-8")
    title = f"CT 32B Probe {slug[-8:]}"
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
    return {
        "kernel_dir": kernel_dir,
        "declared_kernel_ref": metadata["id"],
        "kernel_ref": metadata["id"],
        "kernel_slug": slug,
        "metadata": metadata,
        "filenames": filenames,
    }


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


def summarize_probe_report(probe: dict[str, Any]) -> dict[str, Any]:
    hardware = probe.get("hardware") if isinstance(probe.get("hardware"), dict) else {}
    metrics = probe.get("metrics") if isinstance(probe.get("metrics"), dict) else {}
    downloads = probe.get("downloads") if isinstance(probe.get("downloads"), list) else []
    return {
        "probe_schema": probe.get("schema"),
        "probe_ok": probe.get("ok") is True,
        "probe_stage": probe.get("stage"),
        "gpu_verified": hardware.get("kaggle_gpu_verified") is True,
        "gpu_count": hardware.get("gpu_count"),
        "gpu_names": hardware.get("gpu_names") or [],
        "downloaded_file_count": len(downloads),
        "downloaded_mb": sum(int(item.get("size_mb") or 0) for item in downloads if isinstance(item, dict)),
        "generated_token_count": metrics.get("generated_token_count"),
        "diagnosis_codes": probe.get("diagnosis_codes") or [],
        "blockers": probe.get("blockers") or [],
    }


def build_report(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    package: dict[str, Any],
    steps: list[dict[str, Any]],
    probe_report: dict[str, Any],
) -> dict[str, Any]:
    probe_summary = summarize_probe_report(probe_report)
    lifecycle = {
        "kernel_ref": package.get("kernel_ref"),
        "kernel_slug": package.get("kernel_slug"),
        "requested_accelerator": args.accelerator,
        "cleanup_attempted": any(step.get("name") == "kaggle_kernel_delete" for step in steps),
        "kernel_deleted": any(step.get("name") == "kaggle_kernel_delete" and step.get("ok") for step in steps),
        "private_package_removed": not (output_dir / "private-kaggle-kernel").exists(),
    }
    pushed = any(step.get("name") == "kaggle_kernel_push" and step.get("ok") for step in steps)
    output_downloaded = bool(probe_report)
    blocked_reason = ""
    if not pushed:
        blocked_reason = "kaggle_kernel_push_failed"
    elif not output_downloaded:
        blocked_reason = "kaggle_probe_output_not_downloaded"
    elif probe_summary["probe_ok"]:
        blocked_reason = ""
    elif "kaggle_32b_model_download_complete" not in probe_summary["diagnosis_codes"]:
        blocked_reason = "kaggle_32b_model_download_or_runtime_prepare_blocked"
    else:
        blocked_reason = "kaggle_32b_one_token_generation_failed"
    return {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": bool(pushed and output_downloaded),
        "probe_success": probe_summary["probe_ok"],
        "fresh_kaggle_run_performed": pushed,
        "one_token_generation_verified": probe_summary["probe_ok"],
        "blocked_reason": blocked_reason,
        "output_dir": str(output_dir),
        "model": {
            "repo": args.model_repo,
            "parameter_count_b": 32,
            "quant": args.quant,
            "split_file_count": len(quantized_filenames(args.quant)),
        },
        "runtime": {
            "backend": args.backend,
            "llama_release": args.llama_release,
            "cross_kernel_sharded": False,
            "gpu_offload_requested": args.backend == "source-cuda",
        },
        "probe_summary": probe_summary,
        "kaggle_lifecycle": lifecycle,
        "steps": steps,
        "safety": {
            "public_artifact_safe": True,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "kv_cache_public": False,
            "credentials_public": False,
            "private_kernel_payload_public": False,
        },
        "limitations": [
            "This is a single private Kaggle kernel 32B GGUF probe, not CrowdTensor cross-kernel sharded inference.",
            "The default runtime uses llama.cpp Linux release CPU binaries because Linux CUDA release binaries are not available for the selected llama.cpp release.",
            "A failed 32B probe is still useful blocker evidence for download, disk, container memory, model loading, or runtime execution.",
        ],
    }


def run_live_probe(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    package = build_package(args, output_dir=output_dir)
    steps: list[dict[str, Any]] = []
    try:
        push_command = ["kaggle", "kernels", "push", "-p", str(package["kernel_dir"]), "-t", str(args.kernel_timeout_seconds)]
        if args.accelerator:
            push_command.extend(["--accelerator", args.accelerator])
        print(f"[{utc_now()}] pushing private Kaggle kernel {package['declared_kernel_ref']}", flush=True)
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
                    "ct_32b_probe_report.json",
                ],
                runner=runner,
                timeout_seconds=args.kaggle_output_timeout_seconds,
            )
            steps.append(output_step)
            if not args.skip_kaggle_cleanup:
                print(f"[{utc_now()}] deleting private Kaggle kernel {kernel_ref}", flush=True)
                delete_step = run_step(
                    "kaggle_kernel_delete",
                    ["kaggle", "kernels", "delete", kernel_ref, "-y"],
                    runner=runner,
                    timeout_seconds=args.kaggle_delete_timeout_seconds,
                )
                steps.append(delete_step)
        probe_report = load_json(output_dir / "kaggle-output" / "ct_32b_probe_report.json")
    finally:
        if not args.keep_private_package:
            shutil.rmtree(output_dir / "private-kaggle-kernel", ignore_errors=True)
    report = build_report(args, output_dir=output_dir, package=package, steps=steps, probe_report=probe_report)
    write_json(output_dir / "kaggle_32b_quantized_live_probe.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded private Kaggle 32B quantized live probe.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kaggle-owner", default=default_kaggle_owner())
    parser.add_argument("--kernel-slug-prefix", default="ct32bprobe")
    parser.add_argument("--accelerator", default=DEFAULT_ACCELERATOR)
    parser.add_argument("--model-repo", default=DEFAULT_REPO)
    parser.add_argument("--quant", choices=["q2_k", "q3_k_m", "q4_k_m"], default=DEFAULT_QUANT)
    parser.add_argument("--llama-release", default=DEFAULT_LLAMA_RELEASE)
    parser.add_argument("--backend", choices=["release-cpu", "source-cuda"], default="release-cpu")
    parser.add_argument("--cuda-architectures", default="75")
    parser.add_argument("--cuda-build-jobs", type=int, default=2)
    parser.add_argument("--cuda-build-timeout-seconds", type=int, default=2400)
    parser.add_argument("--run-timeout-seconds", type=int, default=1800)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--kernel-timeout-seconds", type=int, default=3600)
    parser.add_argument("--kaggle-push-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--kaggle-status-timeout-seconds", type=float, default=3900.0)
    parser.add_argument("--kaggle-status-poll-interval", type=float, default=60.0)
    parser.add_argument("--kaggle-output-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--kaggle-delete-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--skip-kaggle-cleanup", action="store_true")
    parser.add_argument("--keep-private-package", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1 or args.max_new_tokens > 16:
        raise SystemExit("--max-new-tokens must be between 1 and 16")
    if args.context_length < 1 or args.context_length > 2048:
        raise SystemExit("--context-length must be between 1 and 2048")
    if args.kernel_timeout_seconds > 3600:
        raise SystemExit("--kernel-timeout-seconds must be <= 3600")
    if args.kaggle_status_timeout_seconds > 4200:
        raise SystemExit("--kaggle-status-timeout-seconds must be <= 4200")
    if args.cuda_build_timeout_seconds > 3600:
        raise SystemExit("--cuda-build-timeout-seconds must be <= 3600")
    if args.run_timeout_seconds > 2400:
        raise SystemExit("--run-timeout-seconds must be <= 2400")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_live_probe(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(f"Kaggle 32B probe fresh run: {report.get('fresh_kaggle_run_performed')}")
        print(f"Probe success: {report.get('probe_success')}")
        print(f"Blocked reason: {report.get('blocked_reason')}")
        summary = report.get("probe_summary") if isinstance(report.get("probe_summary"), dict) else {}
        print(f"GPU: {summary.get('gpu_names')} count={summary.get('gpu_count')}")
        print(f"Downloaded MB: {summary.get('downloaded_mb')}")
        print(f"Generated tokens: {summary.get('generated_token_count')}")
        print(f"Report: {Path(args.output_dir) / 'kaggle_32b_quantized_live_probe.json'}")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
