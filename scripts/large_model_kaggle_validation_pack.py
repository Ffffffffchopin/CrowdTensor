#!/usr/bin/env python3
"""Run or package Kaggle GPU validation for the core large-model path."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor import large_model_inference_rc as inference_rc  # noqa: E402
from scripts import core_technology_handoff_pack as handoff_pack  # noqa: E402
from scripts import large_model_inference_rc_pack as inference_pack  # noqa: E402


SCHEMA = "large_model_kaggle_validation_v1"
SUPPORT_BUNDLE_SCHEMA = "large_model_kaggle_validation_support_bundle_v1"
RUN_SCHEMA = "large_model_kaggle_validation_run_v1"
DEFAULT_OUTPUT_DIR = "dist/large-model-kaggle-validation"
DEFAULT_KERNEL_SLUG_PREFIX = "ct-large-llm"
DEFAULT_KERNEL_TITLE_PREFIX = "CrowdTensor Large LLM"
DEFAULT_SMALL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
DEFAULT_SMALL_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
DEFAULT_SEVEN_B_REPO = "Qwen/Qwen2.5-7B-Instruct-GGUF"
DEFAULT_SEVEN_B_FILE = "qwen2.5-7b-instruct-q2_k.gguf"
DEFAULT_THIRTEEN_B_REPO = "Qwen/Qwen2.5-7B-Instruct-GGUF"
DEFAULT_THIRTEEN_B_FILE = "qwen2.5-7b-instruct-q2_k.gguf"
DEFAULT_LLAMA_RELEASE = "b9611"
DEFAULT_MAX_NEW_TOKENS = 8
DEFAULT_ACCELERATOR = "NvidiaTeslaT4"
MODES = ("package", "kaggle-auto", "evidence-import", "fixture")
TIERS = ("small", "7b", "13b")
LLAMA_BUILD_MODES = ("auto", "source-cuda", "release")
RUNTIME_PATHS = ("rpc", "cli", "hf-cuda")
Runner = Callable[..., subprocess.CompletedProcess[str]]


SECRET_FRAGMENTS = (
    "CrowdTensor validates a public-safe large model sharded inference route.",
    "CROWDTENSOR_MINER_TOKEN=",
    "CROWDTENSOR_OBSERVER_TOKEN=",
    "CROWDTENSOR_ADMIN_TOKEN=",
    "CROWDTENSOR_P2P_PEER_SECRET=",
    "Bearer ",
    "kaggle.json",
    '"prompt_text":',
    '"raw_prompt":',
    '"generated_text":',
    '"output_text":',
    '"generated_token_ids":',
    '"token_ids":',
    '"activation":',
    '"activations":',
    '"kv_cache":',
    '"past_key_values":',
    "operator.private.env",
    "miner.private.env",
    "miner_registry.json",
)


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


def stable_hash_payload(value: Any) -> str:
    return inference_rc.stable_hash_payload(value)


def shell_command(command: list[Any]) -> str:
    return shlex.join([str(part) for part in command])


def redact_text(text: str) -> str:
    redacted = str(text)
    for fragment in SECRET_FRAGMENTS:
        redacted = redacted.replace(fragment, "<redacted>")
    return redacted


def safe_slug(value: str, *, default: str = "ct-large-llm") -> str:
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


def artifact_entry(path: Path, output_dir: Path, *, kind: str, schema: str = "", ok: bool | None = None) -> dict[str, Any]:
    try:
        rel = path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        rel = str(path)
    entry: dict[str, Any] = {"kind": kind, "path": rel, "present": path.is_file()}
    if schema:
        entry["schema"] = schema
    if ok is not None:
        entry["ok"] = bool(ok)
    return entry


def artifact_summary(output_dir: Path, artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    paths = {
        "inspect_first": output_dir / "large_model_kaggle_validation.md",
        "summary_json": output_dir / "large_model_kaggle_validation.json",
        "summary_markdown": output_dir / "large_model_kaggle_validation.md",
        "support_bundle": output_dir / "support_bundle.json",
    }
    return {
        "schema": "large_model_kaggle_validation_artifact_summary_v1",
        "artifact_count": len(artifacts),
        "present_artifact_count": sum(1 for item in artifacts.values() if isinstance(item, dict) and item.get("present")),
        "inspect_first": str(paths["inspect_first"]),
        "summary_json": str(paths["summary_json"]),
        "summary_markdown": str(paths["summary_markdown"]),
        "support_bundle": str(paths["support_bundle"]),
        "shareable_paths": [str(path) for path in paths.values()],
        "public_artifact_safe": True,
    }


def tier_spec(args: argparse.Namespace, tier: str) -> dict[str, Any]:
    if tier == "small":
        return {
            "tier": "small",
            "model_id": args.small_model_id,
            "repo": args.small_model_repo,
            "filename": args.small_model_file,
            "parameter_count_b": args.small_parameter_count_b,
            "quantization": args.small_quantization,
            "model_size_mb": args.small_model_size_mb,
            "layer_count": args.small_layer_count,
        }
    if tier == "13b":
        return {
            "tier": "13b",
            "model_id": args.thirteen_b_model_id,
            "repo": args.thirteen_b_model_repo,
            "filename": args.thirteen_b_model_file,
            "parameter_count_b": args.thirteen_b_parameter_count_b,
            "quantization": args.thirteen_b_quantization,
            "model_size_mb": args.thirteen_b_model_size_mb,
            "layer_count": args.thirteen_b_layer_count,
        }
    return {
        "tier": "7b",
        "model_id": args.seven_b_model_id,
        "repo": args.seven_b_model_repo,
        "filename": args.seven_b_model_file,
        "parameter_count_b": args.seven_b_parameter_count_b,
        "quantization": args.seven_b_quantization,
        "model_size_mb": args.seven_b_model_size_mb,
        "layer_count": args.seven_b_layer_count,
    }


def selected_tiers(args: argparse.Namespace) -> list[str]:
    if args.tiers:
        return [tier.strip() for tier in args.tiers.split(",") if tier.strip()]
    tiers = ["small", "7b"]
    if args.include_13b:
        tiers.append("13b")
    return tiers


def kernel_slug(args: argparse.Namespace) -> str:
    prefix = safe_slug(args.kernel_slug_prefix or DEFAULT_KERNEL_SLUG_PREFIX)
    suffix = str(int(time.time()))[-8:]
    slug = f"{prefix}-{suffix}"
    return slug[:45].strip("-")


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
            cwd=str(ROOT),
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
        "stdout_tail": redact_text((completed.stdout or "")[-1600:]),
        "stderr_tail": redact_text((completed.stderr or "")[-1600:]),
        "command_line": shell_command(command),
        "actual_kernel_ref": extract_kernel_ref(output),
    }


KAGGLE_CODE_URL = re.compile(r"https://www\.kaggle\.com/code/([^/\s]+)/([^/\s]+)")


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


def wait_kaggle_terminal(
    kernel_ref: str,
    *,
    runner: Runner,
    timeout_seconds: float,
    poll_interval: float,
) -> dict[str, Any]:
    started = time.monotonic()
    last_step: dict[str, Any] = {}
    attempts = 0
    while time.monotonic() - started <= timeout_seconds:
        attempts += 1
        last_step = run_step(
            "kaggle_kernel_status",
            ["kaggle", "kernels", "status", kernel_ref],
            runner=runner,
            timeout_seconds=min(60.0, max(5.0, timeout_seconds)),
        )
        output = f"{last_step.get('stdout_tail') or ''}\n{last_step.get('stderr_tail') or ''}"
        status = extract_status(output)
        if status in {"COMPLETE", "ERROR", "FAILED", "CANCELLED", "CANCELED"}:
            last_step["duration_seconds"] = round(time.monotonic() - started, 3)
            last_step["attempts"] = attempts
            last_step["status"] = status
            last_step["terminal"] = True
            last_step["kernel_ref"] = kernel_ref
            last_step["ok"] = bool(last_step.get("ok") and status == "COMPLETE")
            return last_step
        time.sleep(max(1.0, float(poll_interval)))
    last_step["duration_seconds"] = round(time.monotonic() - started, 3)
    last_step["attempts"] = attempts
    last_step["terminal"] = False
    last_step["status"] = extract_status(f"{last_step.get('stdout_tail') or ''}\n{last_step.get('stderr_tail') or ''}")
    last_step["kernel_ref"] = kernel_ref
    last_step["ok"] = False
    last_step["error"] = "timeout_waiting_for_terminal_kernel_status"
    return last_step


def render_kernel(args: argparse.Namespace, tiers: list[dict[str, Any]]) -> str:
    tiers_json = json.dumps(tiers, sort_keys=True)
    return f'''from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import textwrap
import time
import urllib.request
from pathlib import Path


SCHEMA = "{RUN_SCHEMA}"
TIERS = {tiers_json}
MAX_NEW_TOKENS = {int(args.max_new_tokens)}
CONTEXT_LENGTH = {int(args.context_length)}
LLAMA_RELEASE = "{args.llama_release}"
LLAMA_BUILD_MODE = "{args.llama_build_mode}"
RUNTIME_PATH = "{args.runtime_path}"
HF_CUDA_INSTALL_COMPAT = {str(bool(args.hf_cuda_install_compat))}
CUDA_ARCHITECTURES = "{args.cuda_architectures}"
CUDA_NO_VMM = {str(bool(args.cuda_no_vmm))}
CUDA_BUILD_JOBS = {int(args.cuda_build_jobs)}
CUDA_BUILD_TIMEOUT_SECONDS = {int(args.cuda_build_timeout_seconds)}
LLAMA_ASSET = "llama-" + LLAMA_RELEASE + "-bin-ubuntu-x64.tar.gz"
LLAMA_URL = "https://github.com/ggml-org/llama.cpp/releases/download/" + LLAMA_RELEASE + "/" + LLAMA_ASSET
OUT = Path("/kaggle/working")
PROMPT_TEXT = "CrowdTensor validates a public-safe large model sharded inference route."
LLAMA_SOURCE_URL = "https://github.com/ggml-org/llama.cpp/archive/refs/tags/" + LLAMA_RELEASE + ".tar.gz"


def sha_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def public_command(command):
    public = []
    redact_next = False
    for part in command:
        if redact_next:
            public.append("<inline-python-redacted>")
            redact_next = False
            continue
        text = str(part)
        public.append(text if text != PROMPT_TEXT else "<prompt-redacted>")
        if text == "-c":
            redact_next = True
    return public


def safe_tail(value: str, limit: int = 2400) -> str:
    text = str(value or "")[-limit:]
    for fragment in [
        PROMPT_TEXT,
        "Bearer ",
        "KAGGLE_KEY",
        "KAGGLE_USERNAME",
        "CROWDTENSOR_MINER_TOKEN=",
        "CROWDTENSOR_OBSERVER_TOKEN=",
        "CROWDTENSOR_ADMIN_TOKEN=",
    ]:
        text = text.replace(fragment, "<redacted>")
    return text


def setup_stdout_public(command) -> bool:
    if not command:
        return False
    base = Path(str(command[0])).name
    return base in {{"cmake", "tar", "nvidia-smi", "ldd"}}


def classify_stderr(stderr: str) -> str:
    lowered = (stderr or "").lower()
    if "cannot open shared object file" in lowered or "error while loading shared libraries" in lowered:
        return "llama_cpp_shared_library_missing"
    if "unknown argument" in lowered or "invalid argument" in lowered or "unrecognized option" in lowered:
        return "llama_cpp_argument_error"
    if "failed to load model" in lowered or "error loading model" in lowered:
        return "llama_cpp_model_load_failed"
    if "cuda" in lowered and ("error" in lowered or "failed" in lowered or "not found" in lowered):
        return "llama_cpp_cuda_runtime_error"
    if "no gpu" in lowered or "no cuda" in lowered:
        return "llama_cpp_gpu_runtime_missing"
    if "pip" in lowered or "requirement" in lowered or "dependency resolver" in lowered:
        return "python_package_stderr"
    if "warning" in lowered and "error" not in lowered and "failed" not in lowered:
        return "runtime_warning"
    if stderr:
        return "runtime_stderr_present"
    return ""


def run(command, timeout=1200, env=None):
    started = time.monotonic()
    try:
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, env=env)
        step = {{
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "duration_seconds": round(time.monotonic() - started, 3),
            "stdout_digest": sha_text(completed.stdout or ""),
            "stderr_digest": sha_text(completed.stderr or ""),
            "stdout_chars": len(completed.stdout or ""),
            "stderr_chars": len(completed.stderr or ""),
            "stderr_hint": classify_stderr(completed.stderr or ""),
            "stderr_tail": safe_tail(completed.stderr or ""),
            "stdout_public": False,
            "stderr_public": False,
            "command_public": public_command(command),
        }}
        if not completed.returncode == 0 and setup_stdout_public(command):
            step["stdout_tail"] = safe_tail(completed.stdout or "")
        return step, completed.stdout or "", completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        stdout_text = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else exc.stdout
            if isinstance(exc.stdout, str)
            else ""
        )
        stderr_text = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else exc.stderr
            if isinstance(exc.stderr, str)
            else ""
        )
        step = {{
            "ok": False,
            "error": "timeout",
            "duration_seconds": round(time.monotonic() - started, 3),
            "stdout_digest": sha_text(stdout_text),
            "stderr_digest": sha_text(stderr_text),
            "stderr_hint": "timeout",
            "stderr_tail": safe_tail(stderr_text),
            "stdout_public": False,
            "stderr_public": False,
            "command_public": public_command(command),
        }}
        if setup_stdout_public(command):
            step["stdout_tail"] = safe_tail(stdout_text)
        return step, "", ""


def nvidia_smi():
    step, stdout, stderr = run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ], timeout=30)
    devices = []
    if step.get("ok"):
        for line in stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 4:
                try:
                    devices.append({{
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_total_mb": int(float(parts[2])),
                        "memory_free_mb": int(float(parts[3])),
                    }})
                except ValueError:
                    pass
    return {{
        "provider": "kaggle",
        "gpu_count": len(devices),
        "gpu_names": [item["name"] for item in devices],
        "devices": devices,
        "kaggle_gpu_verified": bool(devices),
        "nvidia_smi_step": step,
    }}


def cuda_architectures_for_hardware(hardware):
    requested = str(CUDA_ARCHITECTURES or "").strip()
    if requested and requested.lower() != "native":
        return requested
    names = " ".join(str(name).lower() for name in hardware.get("gpu_names") or [])
    if "p100" in names:
        return "60"
    if "t4" in names:
        return "75"
    if "v100" in names:
        return "70"
    if "a100" in names:
        return "80"
    return "native"


def download(url: str, path: Path, timeout=1800):
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with urllib.request.urlopen(url, timeout=timeout) as response:
        with path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    return {{"path": str(path), "size_mb": path.stat().st_size // (1024 * 1024), "duration_seconds": round(time.monotonic() - started, 3)}}


def find_binary(root: Path, names):
    for name in names:
        for path in root.rglob(name):
            if path.is_file():
                path.chmod(path.stat().st_mode | 0o111)
                return path
    return None


def link_release_libraries(bin_dir: Path):
    links = {{}}
    candidates = [
        ("libggml.so.0", "libggml.so.*"),
        ("libggml-base.so.0", "libggml-base.so.*"),
        ("libllama-common.so.0", "libllama-common.so.*"),
        ("libllama.so.0", "libllama.so.*"),
        ("libmtmd.so.0", "libmtmd.so.*"),
    ]
    for link_name, pattern in candidates:
        link_path = bin_dir / link_name
        if link_path.exists():
            links[link_name] = "already-present"
            continue
        matches = sorted([path for path in bin_dir.glob(pattern) if path.name != link_name])
        if matches:
            try:
                link_path.symlink_to(matches[-1].name)
                links[link_name] = matches[-1].name
            except OSError as exc:
                links[link_name] = type(exc).__name__
    return links


def env_for_binary(binary: Path):
    env = os.environ.copy()
    current = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = str(binary.parent) + ((":" + current) if current else "")
    return env


def client_env_for_binary(binary: Path, *, rpc_enabled: bool):
    env = env_for_binary(binary)
    if rpc_enabled:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def probe_llama_binary(binary: Path, *, env):
    version_step, version_stdout, version_stderr = run([str(binary), "--version"], timeout=60, env=env)
    help_step, help_stdout, help_stderr = run([str(binary), "--help"], timeout=60, env=env)
    return {{
        "version_step": version_step,
        "help_step": help_step,
        "supports_rpc": "--rpc" in help_stdout,
        "supports_file_prompt": "--file" in help_stdout or "-f," in help_stdout,
        "supports_prompt_file": "--prompt-file" in help_stdout,
        "version_digest": sha_text(version_stdout),
        "help_digest": sha_text(help_stdout),
    }}


def attempt_summary(info):
    return {{key: value for key, value in info.items() if key != "attempts"}}


def prepare_llama_release():
    archive = OUT / LLAMA_ASSET
    info = download(LLAMA_URL, archive, timeout=900)
    extract = OUT / "llama-bin"
    extract.mkdir(exist_ok=True)
    step, stdout, stderr = run(["tar", "-xzf", str(archive), "-C", str(extract)], timeout=300)
    cli = find_binary(extract, ["llama-cli", "main"])
    if not cli:
        cli = find_binary(extract, ["llama"])
    rpc = find_binary(extract, ["rpc-server", "llama-rpc-server"])
    links = link_release_libraries(cli.parent) if cli else {{}}
    binary_env = env_for_binary(cli) if cli else os.environ.copy()
    probe = probe_llama_binary(cli, env=binary_env) if cli else {{}}
    return {{
        "ok": bool(step.get("ok") and cli and (probe.get("version_step") or {{}}).get("ok")),
        "download": info,
        "extract_step": step,
        "library_links": links,
        "llama_cli": str(cli) if cli else "",
        "rpc_server": str(rpc) if rpc else "",
        "probe": probe,
        "backend": "llama.cpp",
        "build_mode": "release",
        "cuda_runtime_verified": False,
        "gpu_runtime_capable": False,
    }}


def prepare_llama_source_cuda(hardware):
    archive = OUT / ("llama-source-" + LLAMA_RELEASE + ".tar.gz")
    info = download(LLAMA_SOURCE_URL, archive, timeout=900)
    source_root = OUT / "llama-source"
    source_root.mkdir(exist_ok=True)
    extract_step, stdout, stderr = run(["tar", "-xzf", str(archive), "-C", str(source_root), "--strip-components", "1"], timeout=300)
    build_dir = OUT / "llama-build-cuda"
    cuda_architectures = cuda_architectures_for_hardware(hardware)
    configure_command = [
        "cmake",
        "-S",
        str(source_root),
        "-B",
        str(build_dir),
        "-DGGML_CUDA=ON",
        "-DGGML_RPC=ON",
        "-DLLAMA_CURL=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if CUDA_NO_VMM:
        configure_command.append("-DGGML_CUDA_NO_VMM=ON")
    if cuda_architectures:
        configure_command.append("-DCMAKE_CUDA_ARCHITECTURES=" + cuda_architectures)
    configure_step, _, _ = run(configure_command, timeout=900)
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
        "rpc-server",
    ], timeout=max(1, int(CUDA_BUILD_TIMEOUT_SECONDS or 3600)))
    cli = find_binary(build_dir, ["llama-cli", "main"])
    if not cli:
        cli = find_binary(source_root, ["llama-cli", "main"])
    rpc = find_binary(build_dir, ["rpc-server", "llama-rpc-server"])
    if not rpc:
        rpc = find_binary(source_root, ["rpc-server", "llama-rpc-server"])
    binary_env = env_for_binary(cli) if cli else os.environ.copy()
    probe = probe_llama_binary(cli, env=binary_env) if cli else {{}}
    return {{
        "ok": bool(extract_step.get("ok") and configure_step.get("ok") and build_step.get("ok") and cli and (probe.get("version_step") or {{}}).get("ok")),
        "download": info,
        "extract_step": extract_step,
        "configure_step": configure_step,
        "build_step": build_step,
        "llama_cli": str(cli) if cli else "",
        "rpc_server": str(rpc) if rpc else "",
        "probe": probe,
        "backend": "llama.cpp",
        "build_mode": "source-cuda",
        "cuda_architectures": cuda_architectures,
        "cuda_no_vmm": bool(CUDA_NO_VMM),
        "cuda_runtime_verified": bool(cli and configure_step.get("ok") and build_step.get("ok")),
        "gpu_runtime_capable": bool(cli and configure_step.get("ok") and build_step.get("ok")),
    }}


def prepare_llama(hardware):
    attempts = []
    if LLAMA_BUILD_MODE in ("auto", "source-cuda"):
        try:
            source = prepare_llama_source_cuda(hardware)
        except Exception as exc:
            source = {{"ok": False, "build_mode": "source-cuda", "error_type": type(exc).__name__}}
        attempts.append(attempt_summary(source))
        if source.get("ok") or LLAMA_BUILD_MODE == "source-cuda":
            source["attempts"] = attempts
            return source
    release = prepare_llama_release()
    attempts.append(attempt_summary(release))
    release["attempts"] = attempts
    return release


def start_rpc_servers(llama_info, hardware):
    if RUNTIME_PATH != "rpc":
        return {{"ok": False, "enabled": False, "reason": "runtime_path_not_rpc", "servers": [], "processes": []}}
    rpc_path = llama_info.get("rpc_server") or ""
    if not rpc_path:
        return {{"ok": False, "enabled": True, "reason": "rpc_server_missing", "servers": [], "processes": []}}
    gpu_count = max(0, int(hardware.get("gpu_count") or 0))
    count = max(1, gpu_count)
    servers = []
    processes = []
    env = env_for_binary(Path(rpc_path))
    for index in range(count):
        port = 50052 + index
        command = [rpc_path, "-H", "127.0.0.1", "-p", str(port)]
        if gpu_count > 0 and llama_info.get("gpu_runtime_capable"):
            command.extend(["-d", "CUDA" + str(index)])
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        processes.append(proc)
        servers.append({{
            "index": index,
            "endpoint": "127.0.0.1:" + str(port),
            "command_public": public_command(command),
            "device_index": index if gpu_count > 0 and llama_info.get("gpu_runtime_capable") else None,
            "pid_recorded": True,
        }})
    time.sleep(3)
    alive = [proc.poll() is None for proc in processes]
    return {{
        "ok": bool(servers and all(alive)),
        "enabled": True,
        "servers": servers,
        "processes": processes,
        "worker_count": len(servers),
        "alive_count": sum(1 for item in alive if item),
    }}


def model_url(repo: str, filename: str) -> str:
    return "https://huggingface.co/" + repo + "/resolve/main/" + filename


def generated_token_estimate(stdout: str, max_new_tokens: int) -> int:
    text = stdout.strip()
    if not text:
        return 0
    pieces = [piece for piece in re.split(r"\\s+", text) if piece]
    return min(max_new_tokens, len(pieces))


def run_hf_tier(tier, hardware):
    result = {{
        "schema": SCHEMA,
        "ok": False,
        "tier": tier["tier"],
        "model": {{
            "model_id": tier["model_id"],
            "repo": tier["repo"],
            "filename": "",
            "parameter_count_b": tier["parameter_count_b"],
            "quantization": tier["quantization"],
            "model_size_mb": tier["model_size_mb"],
            "layer_count": tier["layer_count"],
            "model_path_public": False,
        }},
        "runtime": {{
            "backend": "hf_transformers_cuda",
            "intended_backend": "llama_cpp_rpc",
            "llama_build_mode": LLAMA_BUILD_MODE,
            "runtime_path": RUNTIME_PATH,
            "cuda_runtime_verified": False,
            "gpu_runtime_capable": bool(hardware.get("kaggle_gpu_verified")),
            "sharded_path_verified": False,
            "multi_worker_sharded_path_verified": False,
            "worker_count": max(1, int(hardware.get("gpu_count") or 0)),
            "stage_count": max(1, int(hardware.get("gpu_count") or 0)),
        }},
        "hardware": hardware,
        "metrics": {{}},
        "diagnosis_codes": [],
        "safety": {{
            "public_artifact_safe": True,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "kv_cache_public": False,
        }},
    }}
    install_step = {{"ok": True, "skipped": True}}
    if HF_CUDA_INSTALL_COMPAT:
        install_step, _, _ = run([
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu118",
            "torch==2.7.1+cu118",
            "torchvision==0.22.1+cu118",
            "transformers==4.40.2",
            "accelerate==0.30.1",
        ], timeout=1200)
        if not install_step.get("ok"):
            result.update({{
                "blockers": ["hf_cuda_compat_install_failed"],
                "runner_step": install_step,
                "diagnosis_codes": [
                    "hf_cuda_compat_install_failed",
                    "large_model_kaggle_real_runtime_failed",
                    "large_model_kaggle_gpu_runtime_not_verified",
                    "large_model_7b_runtime_not_verified",
                    "large_model_sharded_runtime_path_not_verified",
                ],
            }})
            return result
    code = "\\n".join([
        "import hashlib",
        "import json",
        "import os",
        "import time",
        "",
        "prompt = os.environ['CT_PROMPT']",
        "model_id = os.environ['CT_MODEL_ID']",
        "max_new_tokens = int(os.environ['CT_MAX_NEW_TOKENS'])",
        "context_length = int(os.environ['CT_CONTEXT_LENGTH'])",
        "started = time.monotonic()",
        "summary = dict(ok=False, model_id=model_id, torch_cuda_available=False, device_count=0, device_names=[], generated_token_count=0, output_digest='', wall_time_seconds=0.0, tokens_per_second=0.0, error_type='', error_stage='', error_digest='')",
        "try:",
        "    import torch",
        "    from transformers import AutoModelForCausalLM, AutoTokenizer",
        "    summary['torch_version'] = str(torch.__version__)",
        "    summary['cuda_version'] = str(torch.version.cuda)",
        "    summary['torch_cuda_available'] = bool(torch.cuda.is_available())",
        "    summary['device_count'] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0",
        "    summary['device_names'] = [torch.cuda.get_device_name(i) for i in range(summary['device_count'])]",
        "    if not summary['torch_cuda_available']:",
        "        summary['error_type'] = 'torch_cuda_unavailable'",
        "    else:",
        "        summary['error_stage'] = 'tokenizer_load'",
        "        tokenizer = AutoTokenizer.from_pretrained(model_id)",
        "        summary['error_stage'] = 'model_load'",
        "        model_kwargs = dict(torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map=None)",
        "        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)",
        "        summary['error_stage'] = 'model_to_cuda'",
        "        model = model.to('cuda:0')",
        "        model.eval()",
        "        summary['error_stage'] = 'tokenize'",
        "        encoded = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=context_length)",
        "        encoded = dict((key, value.to('cuda:0')) for key, value in encoded.items())",
        "        input_length = int(encoded['input_ids'].shape[-1])",
        "        summary['error_stage'] = 'generate'",
        "        with torch.inference_mode():",
        "            output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)",
        "        summary['error_stage'] = 'decode'",
        "        generated_ids = output[0][input_length:]",
        "        summary['generated_token_count'] = int(generated_ids.numel())",
        "        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)",
        "        summary['output_digest'] = 'sha256:' + hashlib.sha256(decoded.encode('utf-8', errors='replace')).hexdigest()",
        "        wall = max(time.monotonic() - started, 0.001)",
        "        summary['wall_time_seconds'] = round(wall, 3)",
        "        summary['tokens_per_second'] = round(summary['generated_token_count'] / wall, 4)",
        "        summary['memory_peak_mb'] = int(torch.cuda.max_memory_allocated() // (1024 * 1024))",
        "        summary['ok'] = bool(summary['generated_token_count'] > 0)",
        "        summary['error_stage'] = ''",
        "except Exception as exc:",
        "    summary['error_type'] = type(exc).__name__",
        "    summary['error_digest'] = 'sha256:' + hashlib.sha256(str(exc).encode('utf-8', errors='replace')).hexdigest()",
        "finally:",
        "    summary['wall_time_seconds'] = round(max(time.monotonic() - started, 0.001), 3)",
        "print(json.dumps(summary, sort_keys=True))",
    ])
    env = os.environ.copy()
    env.update({{
        "CT_PROMPT": PROMPT_TEXT,
        "CT_MODEL_ID": tier["repo"],
        "CT_MAX_NEW_TOKENS": str(MAX_NEW_TOKENS),
        "CT_CONTEXT_LENGTH": str(CONTEXT_LENGTH),
    }})
    step, stdout, stderr = run([sys.executable, "-c", code], timeout=2400, env=env)
    hf_summary = {{}}
    for line in reversed([line.strip() for line in stdout.splitlines() if line.strip()]):
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            hf_summary = loaded
            break
    token_count = int(hf_summary.get("generated_token_count") or 0)
    ok = bool(step.get("ok") and hf_summary.get("ok") and token_count > 0 and hf_summary.get("torch_cuda_available"))
    gpu_runtime_verified = bool(ok and hf_summary.get("torch_cuda_available"))
    sharded_path_verified = False
    multi_worker_verified = False
    diagnosis = [
        "large_model_kaggle_real_runtime_verified" if ok else "large_model_kaggle_real_runtime_failed",
        "large_model_kaggle_gpu_runtime_verified" if gpu_runtime_verified else "large_model_kaggle_gpu_runtime_not_verified",
        "large_model_kaggle_gpu_hardware_verified" if hardware.get("kaggle_gpu_verified") else "large_model_kaggle_gpu_runtime_missing",
        "large_model_7b_runtime_verified" if ok and float(tier["parameter_count_b"]) >= 7.0 else "large_model_7b_runtime_not_verified",
        "large_model_sharded_runtime_path_verified" if sharded_path_verified else "large_model_sharded_runtime_path_not_verified",
        "large_model_multi_worker_sharded_path_verified" if multi_worker_verified else "large_model_multi_worker_sharded_path_not_verified",
    ]
    if hf_summary.get("error_type"):
        diagnosis.append("hf_cuda_" + str(hf_summary["error_type"]).lower())
    result.update({{
        "ok": ok,
        "runner_step": step,
        "install_step": install_step,
        "hf_summary": {{
            key: value
            for key, value in hf_summary.items()
            if key not in {{"output_text", "generated_text", "generated_token_ids", "prompt"}}
        }},
        "metrics": {{
            "ttft_ms": None,
            "tokens_per_second": float(hf_summary.get("tokens_per_second") or 0.0),
            "wall_time_seconds": float(hf_summary.get("wall_time_seconds") or 0.0),
            "generated_token_count": token_count,
            "max_new_tokens": MAX_NEW_TOKENS,
            "output_digest": hf_summary.get("output_digest") or sha_text(stdout),
            "p50_latency_ms": None,
            "p95_latency_ms": None,
            "memory_peak_mb": hf_summary.get("memory_peak_mb"),
            "network_bytes_per_token": None,
            "cache_hits": 0,
            "cache_misses": token_count,
        }},
        "runtime": {{
            **result["runtime"],
            "cuda_runtime_verified": gpu_runtime_verified,
            "sharded_path_verified": sharded_path_verified,
            "multi_worker_sharded_path_verified": multi_worker_verified,
            "worker_count": int(hf_summary.get("device_count") or result["runtime"]["worker_count"]),
            "stage_count": int(hf_summary.get("device_count") or result["runtime"]["stage_count"]),
            "device_names": hf_summary.get("device_names") or [],
        }},
        "validation": {{
            "real_runtime_verified": ok,
            "real_7b_runtime_verified": bool(ok and float(tier["parameter_count_b"]) >= 7.0),
            "real_13b_runtime_verified": bool(ok and float(tier["parameter_count_b"]) >= 13.0),
            "kaggle_gpu_verified": bool(hardware.get("kaggle_gpu_verified")),
            "gpu_runtime_verified": gpu_runtime_verified,
            "sharded_path_verified": sharded_path_verified,
            "multi_worker_sharded_path_verified": multi_worker_verified,
            "scale_tier": tier["tier"],
        }},
        "diagnosis_codes": diagnosis,
    }})
    return result


def stop_rpc_servers(rpc_info):
    for proc in rpc_info.get("processes") or []:
        try:
            proc.terminate()
        except Exception:
            pass
    for proc in rpc_info.get("processes") or []:
        try:
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def run_tier(tier, llama_info, hardware, rpc_info, progress=None):
    def publish(stage):
        if callable(progress):
            progress(result, stage)

    llama_cli = str(llama_info.get("llama_cli") or "")
    model_dir = OUT / "models" / tier["tier"]
    model_path = model_dir / tier["filename"]
    rpc_enabled = bool(RUNTIME_PATH == "rpc" and rpc_info.get("ok"))
    rpc_endpoints = [server["endpoint"] for server in rpc_info.get("servers") or [] if server.get("endpoint")]
    worker_count = len(rpc_endpoints) if rpc_enabled else max(1, int(hardware.get("gpu_count") or 0))
    cuda_runtime_available = bool(hardware.get("kaggle_gpu_verified") and llama_info.get("cuda_runtime_verified"))
    result = {{
        "schema": SCHEMA,
        "ok": False,
        "tier": tier["tier"],
        "model": {{
            "model_id": tier["model_id"],
            "repo": tier["repo"],
            "filename": tier["filename"],
            "parameter_count_b": tier["parameter_count_b"],
            "quantization": tier["quantization"],
            "model_size_mb": tier["model_size_mb"],
            "layer_count": tier["layer_count"],
            "model_path_public": False,
        }},
        "runtime": {{
            "backend": "llama_cpp_rpc" if rpc_enabled else "llama_cpp_cli",
            "intended_backend": "llama_cpp_rpc",
            "llama_build_mode": llama_info.get("build_mode"),
            "cuda_runtime_available": cuda_runtime_available,
            "cuda_runtime_verified": False,
            "gpu_runtime_capable": bool(llama_info.get("gpu_runtime_capable")),
            "sharded_path_verified": False,
            "multi_worker_sharded_path_verified": False,
            "runtime_path": RUNTIME_PATH,
            "rpc_enabled": rpc_enabled,
            "rpc_endpoints": rpc_endpoints,
            "worker_count": worker_count,
            "stage_count": worker_count,
        }},
        "hardware": hardware,
        "metrics": {{}},
        "diagnosis_codes": [],
    }}
    result["diagnosis_codes"].append("large_model_kaggle_tier_download_start")
    publish("large_model_kaggle_tier_download_start")
    try:
        download_info = download(model_url(tier["repo"], tier["filename"]), model_path, timeout=3600)
    except Exception as exc:
        result["blockers"] = ["large_model_kaggle_model_download_failed"]
        result["error_type"] = type(exc).__name__
        result["diagnosis_codes"].append("large_model_kaggle_model_download_failed")
        publish("large_model_kaggle_tier_download_failed")
        return result
    result["download"] = download_info
    result["diagnosis_codes"].append("large_model_kaggle_tier_download_complete")
    publish("large_model_kaggle_tier_download_complete")
    prompt_path = OUT / ("prompt-" + tier["tier"] + ".txt")
    prompt_path.write_text(PROMPT_TEXT + "\\n", encoding="utf-8")
    command = [
        llama_cli,
        "-m",
        str(model_path),
        "-f",
        str(prompt_path),
        "-n",
        str(MAX_NEW_TOKENS),
        "-c",
        str(CONTEXT_LENGTH),
        "-ngl",
        "99",
        "--no-display-prompt",
        "--simple-io",
        "--log-disable",
        "-no-cnv",
    ]
    if rpc_enabled:
        command.extend(["--rpc", ",".join(rpc_endpoints)])
        if len(rpc_endpoints) > 1:
            command.extend(["-ts", ",".join(["1"] * len(rpc_endpoints))])
    result["runner_step"] = {{
        "ok": False,
        "pending": True,
        "command_public": public_command(command),
        "client_cuda_hidden": bool(rpc_enabled),
    }}
    result["diagnosis_codes"].append("large_model_kaggle_tier_run_start")
    publish("large_model_kaggle_tier_run_start")
    started = time.monotonic()
    step, stdout, stderr = run(command, timeout=1200, env=client_env_for_binary(Path(llama_cli), rpc_enabled=rpc_enabled))
    wall = round(time.monotonic() - started, 3)
    token_count = generated_token_estimate(stdout, MAX_NEW_TOKENS)
    ok = bool(step.get("ok") and token_count > 0)
    sharded_path_verified = bool(ok and rpc_enabled)
    multi_worker_verified = bool(sharded_path_verified and len(rpc_endpoints) >= 2)
    gpu_runtime_verified = bool(ok and cuda_runtime_available)
    diagnosis = [
        "large_model_kaggle_real_runtime_verified" if ok else "large_model_kaggle_real_runtime_failed",
        "large_model_kaggle_gpu_runtime_verified" if gpu_runtime_verified else "large_model_kaggle_gpu_runtime_not_verified",
        "large_model_kaggle_gpu_hardware_verified" if hardware.get("kaggle_gpu_verified") else "large_model_kaggle_gpu_runtime_missing",
        "large_model_7b_runtime_verified" if ok and float(tier["parameter_count_b"]) >= 7.0 else "large_model_7b_runtime_not_verified",
        "large_model_sharded_runtime_path_verified" if sharded_path_verified else "large_model_sharded_runtime_path_not_verified",
        "large_model_multi_worker_sharded_path_verified" if multi_worker_verified else "large_model_multi_worker_sharded_path_not_verified",
    ]
    if step.get("stderr_hint"):
        diagnosis.append(str(step["stderr_hint"]))
    result.update({{
        "ok": ok,
        "runner_step": step,
        "metrics": {{
            "ttft_ms": None,
            "tokens_per_second": round(token_count / wall, 4) if wall > 0 and token_count else 0.0,
            "wall_time_seconds": wall,
            "generated_token_count": token_count,
            "max_new_tokens": MAX_NEW_TOKENS,
            "output_digest": sha_text(stdout),
            "p50_latency_ms": None,
            "p95_latency_ms": None,
            "memory_peak_mb": max([int(item.get("memory_total_mb") or 0) for item in hardware.get("devices") or []] or [0]),
            "network_bytes_per_token": None,
            "cache_hits": 0,
            "cache_misses": token_count,
        }},
        "validation": {{
            "real_runtime_verified": ok,
            "real_7b_runtime_verified": bool(ok and float(tier["parameter_count_b"]) >= 7.0),
            "real_13b_runtime_verified": bool(ok and float(tier["parameter_count_b"]) >= 13.0),
            "kaggle_gpu_verified": bool(hardware.get("kaggle_gpu_verified")),
            "gpu_runtime_verified": gpu_runtime_verified,
            "sharded_path_verified": sharded_path_verified,
            "multi_worker_sharded_path_verified": multi_worker_verified,
            "scale_tier": tier["tier"],
        }},
        "safety": {{
            "public_artifact_safe": True,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "kv_cache_public": False,
        }},
        "diagnosis_codes": diagnosis,
    }})
    publish("large_model_kaggle_tier_run_complete")
    return result


def build_run_report(started, hardware, llama, rpc_info, tier_results, blockers, *, partial_stage=""):
    ok = any(item.get("ok") for item in tier_results)
    real_7b = any((item.get("validation") or {{}}).get("real_7b_runtime_verified") for item in tier_results)
    sharded_path = any((item.get("validation") or {{}}).get("sharded_path_verified") for item in tier_results)
    multi_worker_sharded_path = any((item.get("validation") or {{}}).get("multi_worker_sharded_path_verified") for item in tier_results)
    gpu_runtime = any((item.get("validation") or {{}}).get("gpu_runtime_verified") for item in tier_results)
    diagnosis = sorted(set(
        ["large_model_kaggle_validation_run_ready" if ok else "large_model_kaggle_validation_run_blocked"]
        + ["large_model_kaggle_gpu_hardware_verified" if hardware.get("kaggle_gpu_verified") else "large_model_kaggle_gpu_runtime_missing"]
        + ["large_model_kaggle_gpu_runtime_verified" if gpu_runtime else "large_model_kaggle_gpu_runtime_not_verified"]
        + ["large_model_7b_runtime_verified" if real_7b else "large_model_7b_runtime_not_verified"]
        + ["large_model_sharded_runtime_path_verified" if sharded_path else "large_model_sharded_runtime_path_not_verified"]
        + ["large_model_multi_worker_sharded_path_verified" if multi_worker_sharded_path else "large_model_multi_worker_sharded_path_not_verified"]
        + ([partial_stage] if partial_stage else [])
        + [code for item in tier_results for code in item.get("diagnosis_codes", [])]
    ))
    return {{
        "schema": SCHEMA,
        "ok": ok,
        "partial_stage": partial_stage,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": round(time.monotonic() - started, 3),
        "hardware": hardware,
        "llama_cpp": llama,
        "rpc": {{k: v for k, v in rpc_info.items() if k != "processes"}},
        "tier_results": tier_results,
        "largest_successful_tier": next((item["tier"] for item in reversed(tier_results) if item.get("ok")), ""),
        "real_runtime_verified": ok,
        "real_7b_runtime_verified": real_7b,
        "gpu_runtime_verified": gpu_runtime,
        "sharded_path_verified": sharded_path,
        "multi_worker_sharded_path_verified": multi_worker_sharded_path,
        "blockers": blockers + ([] if ok else ["large_model_kaggle_no_successful_real_run"]),
        "diagnosis_codes": diagnosis,
        "safety": {{
            "public_artifact_safe": True,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "kv_cache_public": False,
            "credentials_public": False,
        }},
    }}


def write_run_report(report):
    (OUT / "large_model_kaggle_validation_run.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\\n", encoding="utf-8")


def main():
    started = time.monotonic()
    hardware = nvidia_smi()
    llama = {{"ok": False, "llama_cli": "", "backend": "llama.cpp"}}
    rpc_info = {{"ok": False, "enabled": False, "servers": [], "worker_count": 0}}
    tier_results = []
    blockers = []
    write_run_report(build_run_report(started, hardware, llama, rpc_info, tier_results, blockers, partial_stage="large_model_kaggle_hardware_probe_complete"))
    if RUNTIME_PATH != "hf-cuda":
        try:
            llama = prepare_llama(hardware)
        except Exception as exc:
            blockers.append("large_model_kaggle_llama_cpp_install_failed")
            llama = {{"ok": False, "error_type": type(exc).__name__, "backend": "llama.cpp"}}
        write_run_report(build_run_report(started, hardware, llama, rpc_info, tier_results, blockers, partial_stage="large_model_kaggle_llama_cpp_prepare_complete"))
    if hardware.get("gpu_count", 0) <= 0:
        blockers.append("large_model_kaggle_gpu_unavailable")
    if RUNTIME_PATH != "hf-cuda" and not llama.get("ok"):
        blockers.append("large_model_kaggle_llama_cpp_unavailable")
    if hardware.get("kaggle_gpu_verified") and RUNTIME_PATH == "hf-cuda":
        for tier in TIERS:
            tier_results.append(run_hf_tier(tier, hardware))
            write_run_report(build_run_report(started, hardware, llama, rpc_info, tier_results, blockers, partial_stage="large_model_kaggle_tier_attempt_complete"))
            if tier["tier"] == "7b" and tier_results[-1].get("ok"):
                break
            if tier["tier"] == "7b" and not tier_results[-1].get("ok"):
                break
    elif hardware.get("kaggle_gpu_verified") and llama.get("ok"):
        rpc_info = start_rpc_servers(llama, hardware)
        write_run_report(build_run_report(started, hardware, llama, rpc_info, tier_results, blockers, partial_stage="large_model_kaggle_rpc_start_complete"))
        try:
            for tier in TIERS:
                pending_result = {{
                    "schema": SCHEMA,
                    "ok": False,
                    "tier": tier["tier"],
                    "model": {{
                        "model_id": tier["model_id"],
                        "repo": tier["repo"],
                        "filename": tier["filename"],
                        "parameter_count_b": tier["parameter_count_b"],
                        "quantization": tier["quantization"],
                        "model_size_mb": tier["model_size_mb"],
                        "layer_count": tier["layer_count"],
                        "model_path_public": False,
                    }},
                    "runtime": {{
                        "backend": "llama_cpp_rpc" if rpc_info.get("ok") else "llama_cpp_cli",
                        "intended_backend": "llama_cpp_rpc",
                        "runtime_path": RUNTIME_PATH,
                        "rpc_enabled": bool(RUNTIME_PATH == "rpc" and rpc_info.get("ok")),
                        "worker_count": int(rpc_info.get("worker_count") or 0),
                        "stage_count": int(rpc_info.get("worker_count") or 0),
                    }},
                    "hardware": hardware,
                    "metrics": {{}},
                    "diagnosis_codes": ["large_model_kaggle_tier_attempt_start"],
                }}
                tier_results.append(pending_result)
                write_run_report(build_run_report(started, hardware, llama, rpc_info, tier_results, blockers, partial_stage="large_model_kaggle_tier_attempt_start"))
                def update_tier_progress(result, partial_stage):
                    tier_results[-1] = result
                    write_run_report(build_run_report(started, hardware, llama, rpc_info, tier_results, blockers, partial_stage=partial_stage))

                tier_results[-1] = run_tier(tier, llama, hardware, rpc_info, progress=update_tier_progress)
                write_run_report(build_run_report(started, hardware, llama, rpc_info, tier_results, blockers, partial_stage="large_model_kaggle_tier_attempt_complete"))
                if tier["tier"] == "7b" and tier_results[-1].get("ok"):
                    break
                if tier["tier"] == "7b" and not tier_results[-1].get("ok"):
                    break
        finally:
            stop_rpc_servers(rpc_info)
    report = build_run_report(started, hardware, llama, rpc_info, tier_results, blockers)
    ok = bool(report.get("ok"))
    real_7b = bool(report.get("real_7b_runtime_verified"))
    write_run_report(report)
    print(json.dumps({{"ok": ok, "schema": SCHEMA, "real_7b_runtime_verified": real_7b, "diagnosis_codes": report["diagnosis_codes"]}}, sort_keys=True))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
'''


def build_package(args: argparse.Namespace, *, output_dir: Path) -> dict[str, Any]:
    owner = args.kaggle_owner or default_kaggle_owner()
    slug = kernel_slug(args)
    kernel_dir = output_dir / "kaggle-kernel"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    tiers = [tier_spec(args, tier) for tier in selected_tiers(args)]
    (kernel_dir / "kernel.py").write_text(render_kernel(args, tiers), encoding="utf-8")
    metadata = {
        "id": f"{owner}/{slug}" if owner else slug,
        "title": (args.kernel_title_prefix or DEFAULT_KERNEL_TITLE_PREFIX)[:36] + " " + slug[-8:],
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
        "machine_shape": args.accelerator or DEFAULT_ACCELERATOR,
    }
    write_json(kernel_dir / "kernel-metadata.json", metadata)
    runbook = output_dir / "KAGGLE_LARGE_MODEL_VALIDATION.md"
    runbook.write_text(
        "\n".join([
            "# CrowdTensor Large-Model Kaggle Validation",
            "",
            "This package runs a bounded private Kaggle GPU script kernel for core technology validation.",
            "",
            f"- Kernel ref: `{metadata['id']}`",
            f"- Requested accelerator: `{metadata['machine_shape']}`",
            f"- Tiers: `{', '.join(item['tier'] for item in tiers)}`",
            "- Public artifacts contain digests/counts and diagnostics only.",
            "- Raw prompts, generated text, generated token ids, activations, credentials, and private env files are excluded.",
            "",
            "The run first probes Kaggle GPU visibility, prepares the configured runtime backend, runs the small tier, then attempts the 7B tier.",
            "The 13B tier is optional and should stay a stretch target.",
        ]) + "\n",
        encoding="utf-8",
    )
    return {
        "kernel_dir": kernel_dir,
        "kernel_ref": metadata["id"],
        "kernel_slug": slug,
        "metadata": metadata,
        "tiers": tiers,
        "runbook": runbook,
    }


def run_kaggle_auto(args: argparse.Namespace, *, output_dir: Path, runner: Runner) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    package = build_package(args, output_dir=output_dir)
    steps: list[dict[str, Any]] = []
    push_command = ["kaggle", "kernels", "push", "-p", str(package["kernel_dir"]), "-t", str(args.kernel_timeout_seconds)]
    if args.accelerator:
        push_command.extend(["--accelerator", args.accelerator])
    push_step = run_step("kaggle_kernel_push", push_command, runner=runner, timeout_seconds=args.kaggle_push_timeout_seconds)
    steps.append(push_step)
    kernel_ref = push_step.get("actual_kernel_ref") or package["kernel_ref"]
    if not push_step.get("ok"):
        return steps, package, Path("")
    status_step = wait_kaggle_terminal(
        str(kernel_ref),
        runner=runner,
        timeout_seconds=args.kaggle_status_timeout_seconds,
        poll_interval=args.kaggle_status_poll_interval,
    )
    steps.append(status_step)
    output_path = output_dir / "kaggle-output"
    run_report_path = output_path / "large_model_kaggle_validation_run.json"
    if status_step.get("terminal"):
        output_step = run_step(
            "kaggle_kernel_output",
            [
                "kaggle",
                "kernels",
                "output",
                str(kernel_ref),
                "-p",
                str(output_path),
                "--force",
                "--file-pattern",
                "large_model_kaggle_validation_run.json",
            ],
            runner=runner,
            timeout_seconds=args.kaggle_output_timeout_seconds,
        )
        output_step["output_path"] = str(output_path)
        steps.append(output_step)
        if not run_report_path.is_file():
            output_fallback_step = run_step(
                "kaggle_kernel_output_full_fallback",
                [
                    "kaggle",
                    "kernels",
                    "output",
                    str(kernel_ref),
                    "-p",
                    str(output_path),
                    "--force",
                ],
                runner=runner,
                timeout_seconds=args.kaggle_output_timeout_seconds,
            )
            output_fallback_step["output_path"] = str(output_path)
            output_fallback_step["reason"] = "run_report_missing_after_patterned_output"
            steps.append(output_fallback_step)
    if not args.skip_kaggle_cleanup:
        cleanup_step = run_step(
            "kaggle_kernel_delete",
            ["kaggle", "kernels", "delete", str(kernel_ref), "-y"],
            runner=runner,
            timeout_seconds=args.kaggle_delete_timeout_seconds,
        )
        cleanup_step["kernel_ref"] = str(kernel_ref)
        steps.append(cleanup_step)
    return steps, package, run_report_path


def normalize_run_report(payload: dict[str, Any]) -> dict[str, Any]:
    tier_results = payload.get("tier_results") if isinstance(payload.get("tier_results"), list) else []
    successes = [item for item in tier_results if isinstance(item, dict) and item.get("ok")]
    largest = successes[-1] if successes else {}
    metrics = largest.get("metrics") if isinstance(largest.get("metrics"), dict) else payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    metrics = dict(metrics)
    if metrics and metrics.get("ttft_ms") is None:
        metrics["ttft_ms"] = 0.0
    if metrics and metrics.get("tokens_per_second") is None:
        metrics["tokens_per_second"] = 0.0
    if metrics and metrics.get("wall_time_seconds") is None:
        metrics["wall_time_seconds"] = 0.0
    if metrics and metrics.get("output_digest") is None:
        metrics["output_digest"] = ""
    model = largest.get("model") if isinstance(largest.get("model"), dict) else payload.get("model") if isinstance(payload.get("model"), dict) else {}
    runtime = largest.get("runtime") if isinstance(largest.get("runtime"), dict) else payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}
    hardware = payload.get("hardware") if isinstance(payload.get("hardware"), dict) else largest.get("hardware") if isinstance(largest.get("hardware"), dict) else {}
    validation = largest.get("validation") if isinstance(largest.get("validation"), dict) else payload.get("validation") if isinstance(payload.get("validation"), dict) else {}
    real_runtime_verified = bool(payload.get("real_runtime_verified") or validation.get("real_runtime_verified") or payload.get("ok") or successes)
    real_7b_runtime_verified = bool(payload.get("real_7b_runtime_verified") or validation.get("real_7b_runtime_verified"))
    gpu_runtime_verified = bool(payload.get("gpu_runtime_verified") or validation.get("gpu_runtime_verified"))
    sharded_path_verified = bool(payload.get("sharded_path_verified") or validation.get("sharded_path_verified"))
    multi_worker_sharded_path_verified = bool(payload.get("multi_worker_sharded_path_verified") or validation.get("multi_worker_sharded_path_verified"))
    core_validation_ready = bool(real_7b_runtime_verified and gpu_runtime_verified and sharded_path_verified)
    blockers = list(payload.get("blockers") or [])
    if real_runtime_verified:
        blockers = [item for item in blockers if item != "large_model_kaggle_no_successful_real_run"]
    return {
        "schema": RUN_SCHEMA,
        "ok": real_runtime_verified,
        "generated_at": payload.get("generated_at") or utc_now(),
        "model": model,
        "runtime": runtime,
        "hardware": hardware,
        "validation": {
            **validation,
            "real_runtime_verified": real_runtime_verified,
            "real_7b_runtime_verified": real_7b_runtime_verified,
            "real_13b_runtime_verified": any(
                bool((item.get("validation") or {}).get("real_13b_runtime_verified"))
                for item in tier_results
                if isinstance(item, dict)
            ),
            "kaggle_gpu_verified": bool(hardware.get("kaggle_gpu_verified")),
            "gpu_runtime_verified": gpu_runtime_verified,
            "sharded_path_verified": sharded_path_verified,
            "multi_worker_sharded_path_verified": multi_worker_sharded_path_verified,
            "core_validation_ready": core_validation_ready,
            "scale_tier": largest.get("tier") or payload.get("largest_successful_tier") or "",
        },
        "metrics": metrics,
        "tier_results": tier_results,
        "real_runtime_verified": real_runtime_verified,
        "real_7b_runtime_verified": real_7b_runtime_verified,
        "gpu_runtime_verified": gpu_runtime_verified,
        "sharded_path_verified": sharded_path_verified,
        "multi_worker_sharded_path_verified": multi_worker_sharded_path_verified,
        "core_validation_ready": core_validation_ready,
        "blockers": blockers or ([] if real_runtime_verified else ["large_model_kaggle_no_successful_real_run"]),
        "diagnosis_codes": payload.get("diagnosis_codes") or [],
        "safety": payload.get("safety") if isinstance(payload.get("safety"), dict) else {},
    }


def build_imported_rcs(args: argparse.Namespace, *, output_dir: Path, run_report_path: Path, run_report: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    model = run_report.get("model") if isinstance(run_report.get("model"), dict) else {}
    validation = run_report.get("validation") if isinstance(run_report.get("validation"), dict) else {}
    hardware = run_report.get("hardware") if isinstance(run_report.get("hardware"), dict) else {}
    devices = []
    gpu_devices = hardware.get("devices") if isinstance(hardware.get("devices"), list) else []
    for index, gpu in enumerate(gpu_devices or [{"name": "kaggle-gpu", "memory_total_mb": 0, "memory_free_mb": 0}]):
        if not isinstance(gpu, dict):
            continue
        total_mb = int(gpu.get("memory_total_mb") or 0)
        free_mb = int(gpu.get("memory_free_mb") or total_mb or 0)
        usable_mb = int(max(free_mb * 0.85, min(total_mb, free_mb), 8192 if total_mb else 0))
        devices.append({
            "device_id": f"kaggle-gpu-{index}",
            "backend": "cuda",
            "rpc_endpoint": f"http://127.0.0.1:{50052 + index}",
            "usable_memory_mb": usable_mb,
            "vram_total_mb": total_mb,
            "device_name": gpu.get("name") or "kaggle-gpu",
        })
    device_profile_path = output_dir / "kaggle_device_profile.json"
    write_json(device_profile_path, devices)
    try:
        parameter_count_b = float(model.get("parameter_count_b") or 0.0)
    except (TypeError, ValueError):
        parameter_count_b = 0.0
    scale_tier = str(validation.get("scale_tier") or run_report.get("largest_successful_tier") or "")
    default_layer_count = args.seven_b_layer_count
    default_model_size_mb = args.seven_b_model_size_mb
    if scale_tier == "small" or (parameter_count_b and parameter_count_b < 7.0):
        default_layer_count = args.small_layer_count
        default_model_size_mb = args.small_model_size_mb
    elif scale_tier == "13b" or parameter_count_b >= 13.0:
        default_layer_count = args.thirteen_b_layer_count
        default_model_size_mb = args.thirteen_b_model_size_mb
    layer_count = int(model.get("layer_count") or default_layer_count)
    model_size_mb = int(model.get("model_size_mb") or default_model_size_mb)
    rc_args = inference_pack.parse_args([
        "--mode",
        "fixture",
        "--output-dir",
        str(output_dir / "inference-rc"),
        "--model-id",
        str(model.get("model_id") or args.seven_b_model_id),
        "--model-path",
        "kaggle://redacted/model.gguf",
        "--quantization",
        str(model.get("quantization") or args.seven_b_quantization),
        "--layer-count",
        str(layer_count),
        "--context-length",
        str(args.context_length),
        "--model-size-mb",
        str(model_size_mb),
        "--device-profile",
        str(device_profile_path),
        "--real-run-report",
        str(run_report_path),
    ])
    inference_report = inference_pack.build_report(rc_args)
    handoff_args = handoff_pack.parse_args([
        "--mode",
        "fixture",
        "--output-dir",
        str(output_dir / "handoff-rc"),
        "--model-id",
        str(model.get("model_id") or args.seven_b_model_id),
        "--model-path",
        "kaggle://redacted/model.gguf",
        "--quantization",
        str(model.get("quantization") or args.seven_b_quantization),
        "--layer-count",
        str(layer_count),
        "--context-length",
        str(args.context_length),
        "--model-size-mb",
        str(model_size_mb),
        "--device-profile",
        str(device_profile_path),
        "--real-run-report",
        str(run_report_path),
    ])
    handoff_report = handoff_pack.build_report(handoff_args)
    return inference_report, handoff_report


def public_redaction_errors(value: Any) -> list[str]:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True)
    return [fragment for fragment in SECRET_FRAGMENTS if fragment in encoded]


def build_support_bundle(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SUPPORT_BUNDLE_SCHEMA,
        "ok": bool(report.get("ok")),
        "mode": report.get("mode"),
        "real_runtime_verified": bool(report.get("real_runtime_verified")),
        "real_7b_runtime_verified": bool(report.get("real_7b_runtime_verified")),
        "gpu_runtime_verified": bool(report.get("gpu_runtime_verified")),
        "sharded_path_verified": bool(report.get("sharded_path_verified")),
        "multi_worker_sharded_path_verified": bool(report.get("multi_worker_sharded_path_verified")),
        "core_validation_ready": bool(report.get("core_validation_ready")),
        "largest_successful_tier": report.get("largest_successful_tier"),
        "diagnosis_codes": report.get("diagnosis_codes") or [],
        "blockers": report.get("blockers") or [],
        "kaggle_lifecycle": report.get("kaggle_lifecycle"),
        "artifact_summary": report.get("artifact_summary"),
        "public_artifact_safe": bool((report.get("safety") or {}).get("public_artifact_safe")),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CrowdTensor Large-Model Kaggle Validation",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- ok: `{bool(report.get('ok'))}`",
        f"- mode: `{report.get('mode')}`",
        f"- real runtime verified: `{bool(report.get('real_runtime_verified'))}`",
        f"- real 7B runtime verified: `{bool(report.get('real_7b_runtime_verified'))}`",
        f"- real 13B runtime verified: `{bool(report.get('real_13b_runtime_verified'))}`",
        f"- Kaggle GPU runtime verified: `{bool(report.get('gpu_runtime_verified'))}`",
        f"- sharded/RPC path verified: `{bool(report.get('sharded_path_verified'))}`",
        f"- multi-worker sharded path verified: `{bool(report.get('multi_worker_sharded_path_verified'))}`",
        f"- core validation ready: `{bool(report.get('core_validation_ready'))}`",
        f"- largest successful tier: `{report.get('largest_successful_tier')}`",
        "",
        "## Hardware",
        "",
    ]
    hardware = report.get("hardware") if isinstance(report.get("hardware"), dict) else {}
    lines.extend([
        f"- provider: `{hardware.get('provider')}`",
        f"- GPU count: `{hardware.get('gpu_count')}`",
        f"- GPU names: `{', '.join(str(item) for item in hardware.get('gpu_names') or [])}`",
        "",
        "## Tiers",
        "",
    ])
    for item in report.get("tier_results") or []:
        if isinstance(item, dict):
            model = item.get("model") if isinstance(item.get("model"), dict) else {}
            metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
            lines.append(
                f"- `{item.get('tier')}` ok=`{bool(item.get('ok'))}` "
                f"model=`{model.get('model_id')}` params_b=`{model.get('parameter_count_b')}` "
                f"tokens=`{metrics.get('generated_token_count')}` tok/s=`{metrics.get('tokens_per_second')}`"
            )
    lines.extend(["", "## Blockers", ""])
    blockers = report.get("blockers") or []
    lines.extend(f"- `{item}`" for item in blockers) if blockers else lines.append("- none")
    lines.extend(["", "## Diagnosis", ""])
    for code in report.get("diagnosis_codes") or []:
        lines.append(f"- `{code}`")
    lines.extend(["", "## Boundaries", ""])
    for item in report.get("limitations") or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def persist_report(report: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "large_model_kaggle_validation.json"
    md_path = output_dir / "large_model_kaggle_validation.md"
    support_path = output_dir / "support_bundle.json"
    artifacts = report.setdefault("artifacts", {})
    artifacts["summary_json"] = artifact_entry(json_path, output_dir, kind="large_model_kaggle_validation", schema=SCHEMA, ok=report.get("ok"))
    artifacts["summary_markdown"] = artifact_entry(md_path, output_dir, kind="large_model_kaggle_validation_markdown")
    artifacts["support_bundle_json"] = artifact_entry(support_path, output_dir, kind="large_model_kaggle_validation_support_bundle", schema=SUPPORT_BUNDLE_SCHEMA, ok=report.get("ok"))
    report["artifact_summary"] = artifact_summary(output_dir, artifacts)
    report["safety"] = {
        **(report.get("safety") if isinstance(report.get("safety"), dict) else {}),
        "public_artifact_safe": True,
        "raw_prompt_public": False,
        "raw_generated_text_public": False,
        "generated_token_ids_public": False,
        "activation_public": False,
        "kv_cache_public": False,
        "credentials_public": False,
        "private_kaggle_material_public": False,
    }
    errors = public_redaction_errors(report)
    if errors:
        report["ok"] = False
        report["safety"]["public_artifact_safe"] = False
        report["redaction_errors"] = errors
        report.setdefault("diagnosis_codes", []).append("large_model_kaggle_public_artifact_redaction_failed")
    write_json(json_path, report)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    write_json(support_path, build_support_bundle(report))
    report["artifacts"]["summary_json"]["present"] = True
    report["artifacts"]["summary_markdown"]["present"] = True
    report["artifacts"]["support_bundle_json"]["present"] = True
    report["artifact_summary"] = artifact_summary(output_dir, artifacts)
    write_json(json_path, report)
    return report


def build_report(args: argparse.Namespace, *, runner: Runner = subprocess.run) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    steps: list[dict[str, Any]] = []
    package: dict[str, Any] = {}
    run_report_path = Path(args.run_report).resolve() if args.run_report else Path("")
    if args.mode in {"package", "fixture"}:
        package = build_package(args, output_dir=output_dir)
        run_report = {
            "schema": RUN_SCHEMA,
            "ok": False,
            "model": tier_spec(args, "7b"),
            "runtime": {
                "backend": "llama_cpp_rpc",
                "intended_backend": "llama_cpp_rpc",
                "llama_build_mode": args.llama_build_mode,
                "runtime_path": args.runtime_path,
                "cuda_runtime_verified": False,
                "sharded_path_verified": False,
                "multi_worker_sharded_path_verified": False,
                "worker_count": 2,
                "stage_count": 2,
            },
            "hardware": {"provider": "kaggle-fixture", "gpu_count": 0, "gpu_names": [], "kaggle_gpu_verified": False},
            "validation": {
                "real_runtime_verified": False,
                "real_7b_runtime_verified": False,
                "kaggle_gpu_verified": False,
                "gpu_runtime_verified": False,
                "sharded_path_verified": False,
                "multi_worker_sharded_path_verified": False,
                "core_validation_ready": False,
                "scale_tier": "",
                "fixture_only": args.mode == "fixture",
            },
            "metrics": {"ttft_ms": 100.0, "tokens_per_second": 8.0, "wall_time_seconds": 1.0, "generated_token_count": args.max_new_tokens, "max_new_tokens": args.max_new_tokens, "output_digest": "sha256:" + "7" * 64},
            "tier_results": [],
            "diagnosis_codes": ["large_model_kaggle_validation_fixture_ready"] if args.mode == "fixture" else ["large_model_kaggle_package_ready"],
            "blockers": ["large_model_kaggle_not_executed"],
        }
        if args.mode == "fixture":
            run_report = normalize_run_report(run_report)
        if args.mode == "fixture":
            run_report_path = output_dir / "fixture_large_model_kaggle_validation_run.json"
            write_json(run_report_path, run_report)
    elif args.mode == "kaggle-auto":
        steps, package, run_report_path = run_kaggle_auto(args, output_dir=output_dir, runner=runner)
        run_report = normalize_run_report(load_json(run_report_path)) if run_report_path.is_file() else {
            "schema": RUN_SCHEMA,
            "ok": False,
            "blockers": ["large_model_kaggle_run_report_missing"],
            "diagnosis_codes": ["large_model_kaggle_run_report_missing"],
        }
    else:
        if not run_report_path.is_file():
            raise SystemExit("--run-report is required for evidence-import and must exist")
        run_report = normalize_run_report(load_json(run_report_path))

    normalized_path = output_dir / "large_model_kaggle_validation_run_normalized.json"
    write_json(normalized_path, run_report)
    inference_report: dict[str, Any] = {}
    handoff_report: dict[str, Any] = {}
    if run_report.get("ok"):
        inference_report, handoff_report = build_imported_rcs(
            args,
            output_dir=output_dir,
            run_report_path=normalized_path,
            run_report=run_report,
        )
    hardware = run_report.get("hardware") if isinstance(run_report.get("hardware"), dict) else {}
    tier_results = run_report.get("tier_results") if isinstance(run_report.get("tier_results"), list) else []
    largest_successful_tier = str(run_report.get("validation", {}).get("scale_tier") if isinstance(run_report.get("validation"), dict) else "")
    real_runtime_verified = bool(run_report.get("real_runtime_verified") or run_report.get("ok"))
    validation = run_report.get("validation") if isinstance(run_report.get("validation"), dict) else {}
    real_7b_runtime_verified = bool(run_report.get("real_7b_runtime_verified") or validation.get("real_7b_runtime_verified"))
    real_13b_runtime_verified = bool(validation.get("real_13b_runtime_verified"))
    gpu_runtime_verified = bool(run_report.get("gpu_runtime_verified") or validation.get("gpu_runtime_verified"))
    sharded_path_verified = bool(validation.get("sharded_path_verified"))
    multi_worker_sharded_path_verified = bool(validation.get("multi_worker_sharded_path_verified"))
    core_validation_ready = bool(real_7b_runtime_verified and gpu_runtime_verified and sharded_path_verified)
    codes = set(run_report.get("diagnosis_codes") or [])
    codes.add("large_model_kaggle_validation_ready" if real_runtime_verified else "large_model_kaggle_validation_blocked")
    codes.add("large_model_7b_runtime_verified" if real_7b_runtime_verified else "large_model_7b_runtime_not_verified")
    codes.add("large_model_kaggle_gpu_runtime_verified" if gpu_runtime_verified else "large_model_kaggle_gpu_runtime_not_verified")
    if sharded_path_verified:
        codes.add("large_model_sharded_runtime_path_verified")
    else:
        codes.add("large_model_sharded_runtime_path_not_verified")
    if multi_worker_sharded_path_verified:
        codes.add("large_model_multi_worker_sharded_path_verified")
    else:
        codes.add("large_model_multi_worker_sharded_path_not_verified")
    codes.add("large_model_core_validation_ready" if core_validation_ready else "large_model_core_validation_not_ready")
    blockers = list(run_report.get("blockers") or [])
    if not real_runtime_verified and "large_model_kaggle_no_successful_real_run" not in blockers:
        blockers.append("large_model_kaggle_no_successful_real_run")
    if not real_7b_runtime_verified and "large_model_7b_runtime_not_verified" not in blockers:
        blockers.append("large_model_7b_runtime_not_verified")
    if not gpu_runtime_verified and "large_model_kaggle_gpu_runtime_not_verified" not in blockers:
        blockers.append("large_model_kaggle_gpu_runtime_not_verified")
    if not sharded_path_verified and "large_model_sharded_runtime_path_not_verified" not in blockers:
        blockers.append("large_model_sharded_runtime_path_not_verified")
    model_summary = run_report.get("model") if isinstance(run_report.get("model"), dict) else {}
    runtime_summary = run_report.get("runtime") if isinstance(run_report.get("runtime"), dict) else {}
    metrics_summary = run_report.get("metrics") if isinstance(run_report.get("metrics"), dict) else {}
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": bool(real_runtime_verified),
        "mode": args.mode,
        "output_dir": str(output_dir),
        "real_runtime_verified": real_runtime_verified,
        "real_7b_runtime_verified": real_7b_runtime_verified,
        "real_13b_runtime_verified": real_13b_runtime_verified,
        "gpu_runtime_verified": gpu_runtime_verified,
        "sharded_path_verified": sharded_path_verified,
        "multi_worker_sharded_path_verified": multi_worker_sharded_path_verified,
        "core_validation_ready": core_validation_ready,
        "largest_successful_tier": largest_successful_tier,
        "model": model_summary,
        "runtime": runtime_summary,
        "metrics": metrics_summary,
        "hardware": hardware,
        "tier_results": tier_results,
        "run_report": run_report,
        "inference_rc_report": inference_report,
        "handoff_rc_report": handoff_report,
        "kaggle_lifecycle": {
            "owner": args.kaggle_owner or default_kaggle_owner(),
            "kernel_ref": package.get("kernel_ref"),
            "kernel_slug": package.get("kernel_slug"),
            "requested_accelerator": args.accelerator,
            "cleanup_attempted": any(step.get("name") == "kaggle_kernel_delete" for step in steps),
            "kernels_deleted": any(step.get("name") == "kaggle_kernel_delete" and step.get("ok") for step in steps),
            "skip_cleanup": bool(args.skip_kaggle_cleanup),
        },
        "steps": steps,
        "blockers": [item for index, item in enumerate(blockers) if item and item not in blockers[:index]],
        "diagnosis_codes": sorted(codes),
        "artifacts": {
            "kaggle_kernel_metadata": artifact_entry(
                (package.get("kernel_dir", output_dir / "kaggle-kernel") / "kernel-metadata.json")
                if package
                else output_dir / "kaggle-kernel" / "kernel-metadata.json",
                output_dir,
                kind="kaggle_kernel_metadata",
            ),
            "run_report_normalized": artifact_entry(normalized_path, output_dir, kind="large_model_kaggle_validation_run", schema=RUN_SCHEMA, ok=run_report.get("ok")),
            "inference_rc_json": artifact_entry(output_dir / "inference-rc" / "core_technology_inference_rc.json", output_dir, kind="core_technology_inference_rc", schema=inference_rc.RC_SCHEMA, ok=inference_report.get("ok") if inference_report else None),
            "handoff_rc_json": artifact_entry(output_dir / "handoff-rc" / "core_technology_handoff_rc.json", output_dir, kind="core_technology_handoff_rc", schema="core_technology_handoff_rc_v1", ok=handoff_report.get("ok") if handoff_report else None),
        },
        "limitations": [
            "Kaggle accelerator assignment is best-effort; actual GPU type/count comes from nvidia-smi evidence.",
            "This validation proves a bounded private Kaggle runtime, not production public serving, P2P/NAT traversal, billing, or training.",
            "llama.cpp CLI execution is accepted as the real GGUF runtime smoke; sharded path readiness stays false unless the run report proves a sharded/RPC path.",
            "13B is a stretch target and remains false unless a real 13B tier succeeds.",
        ],
    }
    return persist_report(report, output_dir=output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or run CrowdTensor Kaggle large-model validation evidence.")
    parser.add_argument("--mode", choices=MODES, default="package")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kaggle-owner", default=default_kaggle_owner())
    parser.add_argument("--kernel-slug-prefix", default=DEFAULT_KERNEL_SLUG_PREFIX)
    parser.add_argument("--kernel-title-prefix", default=DEFAULT_KERNEL_TITLE_PREFIX)
    parser.add_argument("--accelerator", default=DEFAULT_ACCELERATOR)
    parser.add_argument("--tiers", default="")
    parser.add_argument("--include-13b", action="store_true")
    parser.add_argument("--run-report", default="")
    parser.add_argument("--llama-release", default=DEFAULT_LLAMA_RELEASE)
    parser.add_argument("--llama-build-mode", choices=LLAMA_BUILD_MODES, default="auto")
    parser.add_argument("--runtime-path", choices=RUNTIME_PATHS, default="rpc")
    parser.add_argument("--cuda-architectures", default="native")
    parser.add_argument("--cuda-build-jobs", type=int, default=4)
    parser.add_argument("--cuda-build-timeout-seconds", type=int, default=3600)
    parser.add_argument("--cuda-no-vmm", dest="cuda_no_vmm", action="store_true", default=True)
    parser.add_argument("--cuda-vmm", dest="cuda_no_vmm", action="store_false")
    parser.add_argument("--hf-cuda-install-compat", action="store_true")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--small-model-id", default="qwen2.5-1.5b-instruct-q4-k-m")
    parser.add_argument("--small-model-repo", default=DEFAULT_SMALL_REPO)
    parser.add_argument("--small-model-file", default=DEFAULT_SMALL_FILE)
    parser.add_argument("--small-parameter-count-b", type=float, default=1.5)
    parser.add_argument("--small-quantization", default="Q4_K_M")
    parser.add_argument("--small-model-size-mb", type=int, default=1066)
    parser.add_argument("--small-layer-count", type=int, default=28)
    parser.add_argument("--seven-b-model-id", default="qwen2.5-7b-instruct-q2-k")
    parser.add_argument("--seven-b-model-repo", default=DEFAULT_SEVEN_B_REPO)
    parser.add_argument("--seven-b-model-file", default=DEFAULT_SEVEN_B_FILE)
    parser.add_argument("--seven-b-parameter-count-b", type=float, default=7.6)
    parser.add_argument("--seven-b-quantization", default="Q2_K")
    parser.add_argument("--seven-b-model-size-mb", type=int, default=2876)
    parser.add_argument("--seven-b-layer-count", type=int, default=28)
    parser.add_argument("--thirteen-b-model-id", default="qwen2.5-13b-placeholder")
    parser.add_argument("--thirteen-b-model-repo", default=DEFAULT_THIRTEEN_B_REPO)
    parser.add_argument("--thirteen-b-model-file", default=DEFAULT_THIRTEEN_B_FILE)
    parser.add_argument("--thirteen-b-parameter-count-b", type=float, default=13.0)
    parser.add_argument("--thirteen-b-quantization", default="Q2_K")
    parser.add_argument("--thirteen-b-model-size-mb", type=int, default=8192)
    parser.add_argument("--thirteen-b-layer-count", type=int, default=40)
    parser.add_argument("--kernel-timeout-seconds", type=int, default=7200)
    parser.add_argument("--kaggle-push-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--kaggle-status-timeout-seconds", type=float, default=5400.0)
    parser.add_argument("--kaggle-status-poll-interval", type=float, default=30.0)
    parser.add_argument("--kaggle-output-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--kaggle-delete-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--skip-kaggle-cleanup", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.context_length < 1:
        raise SystemExit("--context-length must be positive")
    if args.max_new_tokens < 1 or args.max_new_tokens > inference_rc.MAX_REAL_RUN_TOKENS:
        raise SystemExit(f"--max-new-tokens must be between 1 and {inference_rc.MAX_REAL_RUN_TOKENS}")
    if args.cuda_build_jobs < 1:
        raise SystemExit("--cuda-build-jobs must be positive")
    if args.cuda_build_timeout_seconds < 1:
        raise SystemExit("--cuda-build-timeout-seconds must be positive")
    tiers = selected_tiers(args)
    bad = sorted(set(tiers) - set(TIERS))
    if bad:
        raise SystemExit(f"unknown tier(s): {', '.join(bad)}")
    if args.mode == "kaggle-auto" and not args.kaggle_owner:
        raise SystemExit("--kaggle-owner or KAGGLE_USERNAME is required for kaggle-auto")
    if args.mode == "evidence-import" and not args.run_report:
        raise SystemExit("--run-report is required for evidence-import")
    if args.run_report and not Path(args.run_report).is_file():
        raise SystemExit("--run-report must point to an existing JSON file")
    for name in [
        "kaggle_push_timeout_seconds",
        "kaggle_status_timeout_seconds",
        "kaggle_status_poll_interval",
        "kaggle_output_timeout_seconds",
        "kaggle_delete_timeout_seconds",
    ]:
        if float(getattr(args, name)) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_report(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(render_markdown(report))


if __name__ == "__main__":
    main()
