#!/usr/bin/env python3
"""Build Kaggle dataset/kernel upload folders for the real tiny-LLM live proof."""

from __future__ import annotations

import argparse
import base64
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "kaggle_real_llm_live_package_v1"
STAGES = ("stage0", "stage1")
ROLE_NORMAL = "normal"
ROLE_VICTIM = "victim"
ROLE_RESCUE = "rescue"
FAILURE_NONE = "none"
FAILURE_KILL_STAGE0_AFTER_CLAIM = "kill-stage0-after-claim"
FAILURE_KILL_STAGE1_AFTER_CLAIM = "kill-stage1-after-claim"
FAILURE_MODES = {
    FAILURE_NONE,
    FAILURE_KILL_STAGE0_AFTER_CLAIM,
    FAILURE_KILL_STAGE1_AFTER_CLAIM,
}
DEFAULT_CUDA_TORCH_SPEC = "torch==2.7.1+cu118"
DEFAULT_CUDA_TORCHVISION_SPEC = "torchvision==0.22.1+cu118"
DEFAULT_CUDA_TORCH_RUNTIME_SPEC = f"{DEFAULT_CUDA_TORCH_SPEC} {DEFAULT_CUDA_TORCHVISION_SPEC}"
DEFAULT_CUDA_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu118"
DEFAULT_TRANSFORMERS_SPEC = "transformers==4.40.2"
KAGGLE_KERNEL_SLUG_MAX_LENGTH = 45
PARTITION_MODE_ALIASES = {
    "full": "full",
    "full-model": "full",
    "stage-local": "stage_local",
    "stage_local": "stage_local",
    "stage": "stage_local",
    "partitioned": "stage_local",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def safe_slug(value: str) -> str:
    cleaned: list[str] = []
    last_dash = False
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
            last_dash = False
        elif not last_dash:
            cleaned.append("-")
            last_dash = True
    return "".join(cleaned).strip("-") or "crowdtensor-real-llm-live"


def validate_kernel_slug_lengths(kernel_slug_prefix: str, keys: list[str]) -> None:
    too_long = [
        f"{kernel_slug_prefix}-{key}"
        for key in keys
        if len(f"{kernel_slug_prefix}-{key}") > KAGGLE_KERNEL_SLUG_MAX_LENGTH
    ]
    if too_long:
        longest = max(too_long, key=len)
        raise SystemExit(
            "--kernel-slug-prefix is too long for Kaggle kernel slugs: "
            f"{longest!r} has {len(longest)} characters, "
            f"limit is {KAGGLE_KERNEL_SLUG_MAX_LENGTH}. Use a shorter prefix."
        )


def relative_entry(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def add_source_tree(tar: tarfile.TarFile) -> list[str]:
    included: list[str] = []
    for path in [
        ROOT / "pyproject.toml",
        ROOT / "LICENSE",
        ROOT / "README.md",
        ROOT / "coordinator.py",
        ROOT / "miner_cli.py",
    ]:
        if path.is_file():
            arcname = relative_entry(path, ROOT)
            tar.add(path, arcname=arcname)
            included.append(arcname)
    for package_file in sorted((ROOT / "crowdtensor").rglob("*.py")):
        if "__pycache__" in package_file.parts:
            continue
        arcname = relative_entry(package_file, ROOT)
        tar.add(package_file, arcname=arcname)
        included.append(arcname)
    return included


def build_source_tarball(path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        included = add_source_tree(tar)
    return {
        "path": str(path),
        "file_count": len(included),
        "included_roots": sorted({item.split("/", 1)[0] for item in included}),
    }


def stage_miner_id(base_miner_id: str, stage: str) -> str:
    suffix = f"-{stage}"
    return base_miner_id if base_miner_id.endswith(suffix) else f"{base_miner_id}{suffix}"


def role_miner_id(base_miner_id: str, stage: str, role: str) -> str:
    if role == ROLE_NORMAL:
        return stage_miner_id(base_miner_id, stage)
    return f"{base_miner_id}-{stage}-{role}"


def entry_key(stage: str, role: str) -> str:
    return stage if role == ROLE_NORMAL else f"{stage}-{role}"


def upload_name(stage: str, role: str) -> str:
    return f"kaggle-upload-real-llm-{entry_key(stage, role)}"


def kernel_title_for_slug(kernel_slug: str) -> str:
    """Kaggle requires the title's clean URL to match metadata.id."""

    return kernel_slug.replace("-", " ").title()


def kernel_entries(args: argparse.Namespace) -> list[dict[str, Any]]:
    failure_mode = str(getattr(args, "failure_mode", FAILURE_NONE) or FAILURE_NONE)
    entries: list[dict[str, Any]] = []
    for stage in STAGES:
        target_role = ROLE_NORMAL
        if failure_mode == FAILURE_KILL_STAGE0_AFTER_CLAIM and stage == "stage0":
            target_role = ""
        if failure_mode == FAILURE_KILL_STAGE1_AFTER_CLAIM and stage == "stage1":
            target_role = ""
        if target_role == ROLE_NORMAL:
            entries.append({"stage": stage, "role": ROLE_NORMAL})
        else:
            entries.append({"stage": stage, "role": ROLE_VICTIM})
            entries.append({"stage": stage, "role": ROLE_RESCUE})
    return entries


def normalize_partition_mode(value: str) -> str:
    normalized = str(value or "full").strip().lower().replace("_", "-")
    if normalized not in PARTITION_MODE_ALIASES:
        raise SystemExit("--real-llm-partition-mode must be full or stage-local")
    return PARTITION_MODE_ALIASES[normalized]


def render_kernel(
    stage: str,
    *,
    coordinator_url: str,
    miner_id: str,
    hf_model_id: str,
    hf_cache_dir: str,
    max_tasks: int,
    max_request_attempts: int,
    compute_seconds: float,
    heartbeat_interval: float,
    idle_sleep: float,
    real_llm_backend: str = "hf_transformers_cpu",
    real_llm_partition_mode: str = "full",
    torch_spec: str = "",
    torch_index_url: str = "",
    transformers_spec: str = DEFAULT_TRANSFORMERS_SPEC,
) -> str:
    gpu_lines = ""
    if real_llm_backend == "hf_transformers_cuda":
        gpu_lines = '''
import torch
if not torch.cuda.is_available():
    raise SystemExit("CrowdTensor GPU proof requires Kaggle Accelerator=GPU and torch.cuda.is_available()")
print("CrowdTensor Kaggle CUDA preflight:", torch.cuda.get_device_name(0), flush=True)
'''
    return f'''from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

{gpu_lines}

STAGE = "{stage}"
COORDINATOR_URL = "{coordinator_url}"
MINER_ID = "{miner_id}"
HF_MODEL_ID = "{hf_model_id}"
HF_CACHE_DIR = "{hf_cache_dir}"
REAL_LLM_PARTITION_MODE = "{real_llm_partition_mode}"
TORCH_SPEC = "{torch_spec}"
TORCH_INDEX_URL = "{torch_index_url}"
TRANSFORMERS_SPEC = "{transformers_spec}"


def find_input_root() -> Path:
    for root in Path("/kaggle/input").glob("*"):
        if (root / "crowdtensor_source").is_dir() or (root / "crowdtensor_source.tar.gz").is_file():
            return root
    raise SystemExit("crowdtensor_source.tar.gz not found in Kaggle inputs")


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {{}}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        parsed = shlex.split(raw_value)
        values[key] = parsed[0] if parsed else ""
    return values


input_root = find_input_root()
if (input_root / "crowdtensor_source").is_dir():
    src_dir = input_root / "crowdtensor_source"
else:
    src_dir = Path("/kaggle/working/crowdtensor-src")
    src_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(input_root / "crowdtensor_source.tar.gz", "r:gz") as tar:
        tar.extractall(src_dir)

stage_dir = input_root / STAGE
env = os.environ.copy()
env.update(load_env(stage_dir / "miner.private.env"))
env["PYTHONPATH"] = str(src_dir)
env["CROWDTENSOR_REMOTE_ENVIRONMENT"] = "kaggle-real-llm"
env.setdefault("PYTHONUNBUFFERED", "1")
log_path = Path("/kaggle/working") / f"crowdtensor_real_llm_{{STAGE}}.log"

with log_path.open("a", encoding="utf-8") as log:
    log.write("CrowdTensor Kaggle real LLM stage miner start\\n")
    log.write(f"stage={{STAGE}} miner_id={{MINER_ID}} backend={real_llm_backend}\\n")
    log.flush()
    if TORCH_SPEC:
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--force-reinstall",
        ]
        command.extend(shlex.split(TORCH_SPEC))
        if TORCH_INDEX_URL:
            command.extend(["--index-url", TORCH_INDEX_URL])
        subprocess.check_call(command, stdout=log, stderr=subprocess.STDOUT)
    transformers_command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
    ]
    transformers_command.extend(shlex.split(TRANSFORMERS_SPEC or "{DEFAULT_TRANSFORMERS_SPEC}"))
    subprocess.check_call(transformers_command, stdout=log, stderr=subprocess.STDOUT)

command = [
    sys.executable,
    str(src_dir / "miner_cli.py"),
    "--coordinator",
    COORDINATOR_URL,
    "--miner-id",
    MINER_ID,
    "--max-tasks",
    "{max_tasks}",
    "--compute-seconds",
    "{compute_seconds}",
    "--heartbeat-interval",
    "{heartbeat_interval}",
    "--enable-hf-tiny-gpt-runtime",
    "--hf-model-id",
    HF_MODEL_ID,
    "--hf-cache-dir",
    HF_CACHE_DIR,
    "--real-llm-backend",
    "{real_llm_backend}",
    "--real-llm-partition-mode",
    REAL_LLM_PARTITION_MODE,
    "--real-llm-stage-role",
    STAGE,
    "--result-timeout",
    "60.0",
    "--heartbeat-timeout",
    "10.0",
    "--retry-base-sleep",
    "1.0",
    "--retry-max-sleep",
    "5.0",
    "--debug-tracebacks",
    "--max-request-attempts",
    "{max_request_attempts}",
    "--idle-sleep",
    "{idle_sleep}",
]
print("Starting CrowdTensor Kaggle real LLM stage miner:", " ".join(command), flush=True)
with log_path.open("a", encoding="utf-8") as log:
    log.write("Starting command: " + " ".join(command) + "\\n")
    log.flush()
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    raise SystemExit(process.wait())
'''


def render_inline_kernel(
    stage: str,
    *,
    coordinator_url: str,
    miner_id: str,
    hf_model_id: str,
    hf_cache_dir: str,
    max_tasks: int,
    max_request_attempts: int,
    compute_seconds: float,
    heartbeat_interval: float,
    idle_sleep: float,
    source_tarball_b64: str,
    miner_env_text: str,
    real_llm_backend: str = "hf_transformers_cpu",
    real_llm_partition_mode: str = "full",
    torch_spec: str = "",
    torch_index_url: str = "",
    transformers_spec: str = DEFAULT_TRANSFORMERS_SPEC,
) -> str:
    gpu_lines = ""
    if real_llm_backend == "hf_transformers_cuda":
        gpu_lines = '''
import torch
if not torch.cuda.is_available():
    raise SystemExit("CrowdTensor GPU proof requires Kaggle Accelerator=GPU and torch.cuda.is_available()")
print("CrowdTensor Kaggle CUDA preflight:", torch.cuda.get_device_name(0), flush=True)
'''
    return f'''from __future__ import annotations

import base64
import os
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

{gpu_lines}

STAGE = "{stage}"
COORDINATOR_URL = "{coordinator_url}"
MINER_ID = "{miner_id}"
HF_MODEL_ID = "{hf_model_id}"
HF_CACHE_DIR = "{hf_cache_dir}"
REAL_LLM_PARTITION_MODE = "{real_llm_partition_mode}"
TORCH_SPEC = "{torch_spec}"
TORCH_INDEX_URL = "{torch_index_url}"
TRANSFORMERS_SPEC = "{transformers_spec}"
SOURCE_TARBALL_B64 = """{source_tarball_b64}"""
MINER_ENV_TEXT = {miner_env_text!r}


def load_env_text(text: str) -> dict[str, str]:
    values: dict[str, str] = {{}}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        parsed = shlex.split(raw_value)
        values[key] = parsed[0] if parsed else ""
    return values


src_dir = Path("/kaggle/working/crowdtensor-src")
src_dir.mkdir(parents=True, exist_ok=True)
archive = Path("/kaggle/working/crowdtensor_source.tar.gz")
archive.write_bytes(base64.b64decode(SOURCE_TARBALL_B64.encode("ascii")))
with tarfile.open(archive, "r:gz") as tar:
    tar.extractall(src_dir)

env = os.environ.copy()
env.update(load_env_text(MINER_ENV_TEXT))
env["PYTHONPATH"] = str(src_dir)
env["CROWDTENSOR_REMOTE_ENVIRONMENT"] = "kaggle-real-llm"
env.setdefault("PYTHONUNBUFFERED", "1")
log_path = Path("/kaggle/working") / f"crowdtensor_real_llm_{{STAGE}}.log"

with log_path.open("a", encoding="utf-8") as log:
    log.write("CrowdTensor Kaggle real LLM stage miner start\\n")
    log.write(f"stage={{STAGE}} miner_id={{MINER_ID}} backend={real_llm_backend}\\n")
    log.flush()
    if TORCH_SPEC:
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--force-reinstall",
        ]
        command.extend(shlex.split(TORCH_SPEC))
        if TORCH_INDEX_URL:
            command.extend(["--index-url", TORCH_INDEX_URL])
        subprocess.check_call(command, stdout=log, stderr=subprocess.STDOUT)
    transformers_command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
    ]
    transformers_command.extend(shlex.split(TRANSFORMERS_SPEC or "{DEFAULT_TRANSFORMERS_SPEC}"))
    subprocess.check_call(transformers_command, stdout=log, stderr=subprocess.STDOUT)

command = [
    sys.executable,
    str(src_dir / "miner_cli.py"),
    "--coordinator",
    COORDINATOR_URL,
    "--miner-id",
    MINER_ID,
    "--max-tasks",
    "{max_tasks}",
    "--compute-seconds",
    "{compute_seconds}",
    "--heartbeat-interval",
    "{heartbeat_interval}",
    "--enable-hf-tiny-gpt-runtime",
    "--hf-model-id",
    HF_MODEL_ID,
    "--hf-cache-dir",
    HF_CACHE_DIR,
    "--real-llm-backend",
    "{real_llm_backend}",
    "--real-llm-partition-mode",
    REAL_LLM_PARTITION_MODE,
    "--real-llm-stage-role",
    STAGE,
    "--result-timeout",
    "60.0",
    "--heartbeat-timeout",
    "10.0",
    "--retry-base-sleep",
    "1.0",
    "--retry-max-sleep",
    "5.0",
    "--debug-tracebacks",
    "--max-request-attempts",
    "{max_request_attempts}",
    "--idle-sleep",
    "{idle_sleep}",
]
print("Starting CrowdTensor Kaggle real LLM stage miner:", " ".join(command), flush=True)
with log_path.open("a", encoding="utf-8") as log:
    log.write("Starting command: " + " ".join(command) + "\\n")
    log.flush()
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    raise SystemExit(process.wait())
'''


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_package(args: argparse.Namespace) -> dict[str, Any]:
    args.failure_mode = getattr(args, "failure_mode", FAILURE_NONE)
    args.compute_seconds = float(getattr(args, "compute_seconds", 0.2))
    args.victim_compute_seconds = float(getattr(args, "victim_compute_seconds", 30.0))
    args.heartbeat_interval = float(getattr(args, "heartbeat_interval", 0.1))
    args.idle_sleep = float(getattr(args, "idle_sleep", 1.0))
    args.real_llm_backend = str(getattr(args, "real_llm_backend", "hf_transformers_cpu") or "hf_transformers_cpu")
    args.real_llm_partition_mode = normalize_partition_mode(getattr(args, "real_llm_partition_mode", "full"))
    args.torch_spec = str(getattr(args, "torch_spec", "") or "")
    args.torch_index_url = str(getattr(args, "torch_index_url", "") or "")
    args.transformers_spec = str(getattr(args, "transformers_spec", "") or DEFAULT_TRANSFORMERS_SPEC)
    if args.real_llm_backend == "hf_transformers_cuda" and not args.torch_spec:
        args.torch_spec = DEFAULT_CUDA_TORCH_RUNTIME_SPEC
        args.torch_index_url = args.torch_index_url or DEFAULT_CUDA_TORCH_INDEX_URL
    real_llm_dir = Path(args.real_llm_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    dataset_dir = output_dir / "dataset"
    kernels_dir = output_dir / "kernels"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    kernels_dir.mkdir(parents=True, exist_ok=True)

    prepare = load_json(real_llm_dir / "real_llm_live_rc.json")
    coordinator_url = args.coordinator_url or str(prepare.get("coordinator_url") or "")
    if not coordinator_url:
        raise SystemExit("--coordinator-url is required when the prepare report is missing one")
    miner_id = args.miner_id or str(prepare.get("miner_id") or "kaggle-real-llm")
    hf_model_id = args.hf_model_id or str((prepare.get("workload") or {}).get("hf_model_id") or "sshleifer/tiny-gpt2")
    owner = args.owner.strip()
    dataset_slug = safe_slug(args.dataset_slug)
    kernel_slug_prefix = safe_slug(args.kernel_slug_prefix)
    dataset_ref = f"{owner}/{dataset_slug}"
    hf_cache_dir = args.hf_cache_dir or "/kaggle/working/crowdtensor-hf-cache-real-llm"
    entries = kernel_entries(args)
    validate_kernel_slug_lengths(kernel_slug_prefix, [entry_key(str(item["stage"]), str(item["role"])) for item in entries])

    source = build_source_tarball(dataset_dir / "crowdtensor_source.tar.gz")
    source_tarball_b64 = ""
    if args.inline_kernel_payload:
        source_tarball_b64 = base64.b64encode((dataset_dir / "crowdtensor_source.tar.gz").read_bytes()).decode("ascii")
    write_json(dataset_dir / "dataset-metadata.json", {
        "title": args.dataset_title,
        "id": dataset_ref,
        "licenses": [{"name": "CC0-1.0"}],
    })

    stage_reports: list[dict[str, Any]] = []
    for entry in entries:
        stage = str(entry["stage"])
        role = str(entry["role"])
        key = entry_key(stage, role)
        upload = real_llm_dir / upload_name(stage, role)
        stage_dataset_dir = dataset_dir / key
        stage_dataset_dir.mkdir(parents=True, exist_ok=True)
        for name in ["miner.private.env", "kaggle_remote_miner.py", "KAGGLE_RUN.md"]:
            source_file = upload / name
            if source_file.is_file():
                (stage_dataset_dir / name).write_text(source_file.read_text(encoding="utf-8"), encoding="utf-8")

        kernel_dir = kernels_dir / key
        kernel_dir.mkdir(parents=True, exist_ok=True)
        kernel_slug = f"{kernel_slug_prefix}-{key}"
        code_path = kernel_dir / "kernel.py"
        miner_env_text = (stage_dataset_dir / "miner.private.env").read_text(encoding="utf-8") if (stage_dataset_dir / "miner.private.env").is_file() else ""
        compute_seconds = float(args.victim_compute_seconds) if role == ROLE_VICTIM else float(args.compute_seconds)
        max_tasks = 1 if role == ROLE_VICTIM else int(args.max_tasks)
        if args.inline_kernel_payload:
            code = render_inline_kernel(
                stage,
                coordinator_url=coordinator_url,
                miner_id=role_miner_id(miner_id, stage, role),
                hf_model_id=hf_model_id,
                hf_cache_dir=hf_cache_dir,
                max_tasks=max_tasks,
                max_request_attempts=args.max_request_attempts,
                compute_seconds=compute_seconds,
                heartbeat_interval=args.heartbeat_interval,
                idle_sleep=args.idle_sleep,
                source_tarball_b64=source_tarball_b64,
                miner_env_text=miner_env_text,
                real_llm_backend=args.real_llm_backend,
                real_llm_partition_mode=args.real_llm_partition_mode,
                torch_spec=args.torch_spec,
                torch_index_url=args.torch_index_url,
                transformers_spec=args.transformers_spec,
            )
        else:
            code = render_kernel(
                stage,
                coordinator_url=coordinator_url,
                miner_id=role_miner_id(miner_id, stage, role),
                hf_model_id=hf_model_id,
                hf_cache_dir=hf_cache_dir,
                max_tasks=max_tasks,
                max_request_attempts=args.max_request_attempts,
                compute_seconds=compute_seconds,
                heartbeat_interval=args.heartbeat_interval,
                idle_sleep=args.idle_sleep,
                real_llm_backend=args.real_llm_backend,
                real_llm_partition_mode=args.real_llm_partition_mode,
                torch_spec=args.torch_spec,
                torch_index_url=args.torch_index_url,
                transformers_spec=args.transformers_spec,
            )
        code_path.write_text(code, encoding="utf-8")
        write_json(kernel_dir / "kernel-metadata.json", {
            "id": f"{owner}/{kernel_slug}",
            "title": kernel_title_for_slug(kernel_slug),
            "code_file": "kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true" if args.real_llm_backend == "hf_transformers_cuda" else "false",
            "enable_tpu": "false",
            "enable_internet": "true",
            "dataset_sources": [] if args.inline_kernel_payload else [dataset_ref],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        })
        stage_reports.append({
            "stage": stage,
            "role": role,
            "key": key,
            "kernel_ref": f"{owner}/{kernel_slug}",
            "kernel_dir": str(kernel_dir),
            "miner_id": role_miner_id(miner_id, stage, role),
            "miner_env_present": (stage_dataset_dir / "miner.private.env").is_file(),
            "operator_env_excluded": not (stage_dataset_dir / "operator.private.env").exists(),
            "registry_excluded": not (stage_dataset_dir / "miner_registry.json").exists(),
            "inline_kernel_payload": bool(args.inline_kernel_payload),
            "max_tasks": max_tasks,
            "max_request_attempts": args.max_request_attempts,
            "compute_seconds": compute_seconds,
            "hf_runtime_enabled": "--enable-hf-tiny-gpt-runtime" in code,
            "real_llm_stage_role_present": "--real-llm-stage-role" in code and stage in code,
            "real_llm_backend": args.real_llm_backend,
            "real_llm_partition_mode": args.real_llm_partition_mode,
            "gpu_accelerator_enabled": args.real_llm_backend == "hf_transformers_cuda",
            "cuda_preflight_present": "torch.cuda.is_available()" in code if args.real_llm_backend == "hf_transformers_cuda" else False,
            "torch_spec": args.torch_spec,
            "torch_index_url": args.torch_index_url,
            "transformers_spec": args.transformers_spec,
        })

    ok = all(
        item["miner_env_present"]
        and item["operator_env_excluded"]
        and item["registry_excluded"]
        and item["hf_runtime_enabled"]
        and item["real_llm_stage_role_present"]
        for item in stage_reports
    )
    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": ok,
        "output_dir": str(output_dir),
        "real_llm_dir": str(real_llm_dir),
        "dataset_ref": dataset_ref,
        "dataset_dir": str(dataset_dir),
        "source": source,
        "coordinator_url": coordinator_url,
        "miner_id": miner_id,
        "hf_model_id": hf_model_id,
        "failure_mode": args.failure_mode,
        "max_tasks": args.max_tasks,
        "max_request_attempts": args.max_request_attempts,
        "real_llm_backend": args.real_llm_backend,
        "real_llm_partition_mode": args.real_llm_partition_mode,
        "torch_spec": args.torch_spec,
        "torch_index_url": args.torch_index_url,
        "transformers_spec": args.transformers_spec,
        "compute_seconds": args.compute_seconds,
        "victim_compute_seconds": args.victim_compute_seconds,
        "stages": stage_reports,
        "diagnosis_codes": [
            "kaggle_real_llm_live_package_ready",
            "kaggle_dataset_package_ready",
            "kaggle_kernel_package_ready",
            "kaggle_real_llm_requeue_kernel_package_ready" if args.failure_mode != FAILURE_NONE else "kaggle_real_llm_normal_kernel_package_ready",
        ] if ok else ["kaggle_real_llm_live_package_blocked"],
        "safety": {
            "dataset_private_expected": True,
            "operator_env_excluded": True,
            "registry_excluded": True,
            "source_tarball_excludes_git_and_dist": True,
            "private_kernel_payload_contains_miner_env": bool(args.inline_kernel_payload),
            "cpu_only": args.real_llm_backend != "hf_transformers_cuda",
            "gpu_backend_selected": args.real_llm_backend == "hf_transformers_cuda",
            "stage_local_partition": args.real_llm_partition_mode == "stage_local",
            "kaggle_gpu_accelerator_expected": args.real_llm_backend == "hf_transformers_cuda",
            "cuda_torch_wheel_pinned": bool(args.torch_spec) if args.real_llm_backend == "hf_transformers_cuda" else False,
            "read_only": True,
            "not_production": True,
            "not_p2p": True,
            "not_large_model_serving": True,
        },
        "commands": {
            "create_dataset": f"kaggle datasets create -p {dataset_dir} -r zip",
            **{
                f"push_{item['key']}_kernel": f"kaggle kernels push -p {kernels_dir / item['key']}"
                for item in stage_reports
            },
        },
    }
    write_json(output_dir / "kaggle_real_llm_live_package.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Kaggle upload folders for the real small-LLM live proof.")
    parser.add_argument("--real-llm-dir", default="dist/real-llm-live-goal")
    parser.add_argument("--output-dir", default="dist/real-llm-live-goal/kaggle-cli-package")
    parser.add_argument("--owner", required=True)
    parser.add_argument("--dataset-slug", default="crowdtensor-real-llm-live")
    parser.add_argument("--dataset-title", default="CrowdTensor Real LLM Live Package")
    parser.add_argument("--kernel-slug-prefix", default="crowdtensor-real-llm-live")
    parser.add_argument("--kernel-title-prefix", default="CrowdTensor Real LLM Live Miner")
    parser.add_argument("--coordinator-url", default="")
    parser.add_argument("--miner-id", default="")
    parser.add_argument("--hf-model-id", default="")
    parser.add_argument("--hf-cache-dir", default="")
    parser.add_argument("--real-llm-backend", choices=["hf_transformers_cpu", "hf_transformers_cuda"], default="hf_transformers_cpu")
    parser.add_argument("--real-llm-partition-mode", choices=["full", "stage-local", "stage_local"], default="full")
    parser.add_argument(
        "--torch-spec",
        default="",
        help=(
            "Optional torch package spec installed inside generated Kaggle kernels before the miner runs. "
            f"CUDA kernels default to {DEFAULT_CUDA_TORCH_SPEC!r} plus {DEFAULT_CUDA_TORCHVISION_SPEC!r} "
            "for older Kaggle GPUs."
        ),
    )
    parser.add_argument(
        "--torch-index-url",
        default="",
        help=(
            "Optional pip index URL used with --torch-spec. "
            f"CUDA kernels default to {DEFAULT_CUDA_TORCH_INDEX_URL!r}."
        ),
    )
    parser.add_argument(
        "--transformers-spec",
        default=DEFAULT_TRANSFORMERS_SPEC,
        help="Transformers package spec installed inside generated Kaggle kernels.",
    )
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--max-request-attempts", type=int, default=120)
    parser.add_argument("--compute-seconds", type=float, default=0.2)
    parser.add_argument("--victim-compute-seconds", type=float, default=30.0)
    parser.add_argument("--heartbeat-interval", type=float, default=0.1)
    parser.add_argument("--idle-sleep", type=float, default=1.0)
    parser.add_argument("--failure-mode", choices=sorted(FAILURE_MODES), default=FAILURE_NONE)
    parser.add_argument("--inline-kernel-payload", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.max_tasks < 1:
        raise SystemExit("--max-tasks must be at least 1")
    if args.max_request_attempts < 1:
        raise SystemExit("--max-request-attempts must be at least 1")
    if args.compute_seconds < 0:
        raise SystemExit("--compute-seconds must be non-negative")
    if args.victim_compute_seconds <= 0:
        raise SystemExit("--victim-compute-seconds must be positive")
    if args.heartbeat_interval <= 0:
        raise SystemExit("--heartbeat-interval must be positive")
    if args.idle_sleep <= 0:
        raise SystemExit("--idle-sleep must be positive")
    return args


def main() -> None:
    report = build_package(parse_args())
    print(json.dumps(report, sort_keys=True))
    if not report.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
