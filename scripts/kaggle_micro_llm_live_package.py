#!/usr/bin/env python3
"""Build Kaggle dataset/kernel upload folders for the micro-LLM live proof."""

from __future__ import annotations

import argparse
import base64
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "kaggle_micro_llm_live_package_v1"


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
    cleaned = []
    last_dash = False
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
            last_dash = False
        elif not last_dash:
            cleaned.append("-")
            last_dash = True
    slug = "".join(cleaned).strip("-")
    return slug or "crowdtensor-micro-llm-live"


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


def render_kernel(stage: str, *, coordinator_url: str, miner_id: str) -> str:
    return f'''from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path


STAGE = "{stage}"
COORDINATOR_URL = "{coordinator_url}"
MINER_ID = "{miner_id}"


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
env["CROWDTENSOR_REMOTE_ENVIRONMENT"] = "kaggle"
env.setdefault("PYTHONUNBUFFERED", "1")

command = [
    sys.executable,
    str(src_dir / "miner_cli.py"),
    "--coordinator",
    COORDINATOR_URL,
    "--miner-id",
    MINER_ID,
    "--max-tasks",
    "1",
    "--compute-seconds",
    "0.2",
    "--heartbeat-interval",
    "0.1",
    "--micro-llm-stage-role",
    STAGE,
    "--max-request-attempts",
    "120",
    "--idle-sleep",
    "1.0",
]
print("Starting CrowdTensor Kaggle stage miner:", " ".join(command), flush=True)
raise SystemExit(subprocess.call(command, env=env))
'''


def render_inline_kernel(
    stage: str,
    *,
    coordinator_url: str,
    miner_id: str,
    source_tarball_b64: str,
    miner_env_text: str,
) -> str:
    return f'''from __future__ import annotations

import base64
import os
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path


STAGE = "{stage}"
COORDINATOR_URL = "{coordinator_url}"
MINER_ID = "{miner_id}"
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
env["CROWDTENSOR_REMOTE_ENVIRONMENT"] = "kaggle"
env.setdefault("PYTHONUNBUFFERED", "1")

command = [
    sys.executable,
    str(src_dir / "miner_cli.py"),
    "--coordinator",
    COORDINATOR_URL,
    "--miner-id",
    MINER_ID,
    "--max-tasks",
    "1",
    "--compute-seconds",
    "0.2",
    "--heartbeat-interval",
    "0.1",
    "--micro-llm-stage-role",
    STAGE,
    "--max-request-attempts",
    "120",
    "--idle-sleep",
    "1.0",
]
print("Starting CrowdTensor Kaggle stage miner:", " ".join(command), flush=True)
raise SystemExit(subprocess.call(command, env=env))
'''


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_package(args: argparse.Namespace) -> dict[str, Any]:
    kaggle_dir = Path(args.kaggle_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    dataset_dir = output_dir / "dataset"
    kernels_dir = output_dir / "kernels"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    kernels_dir.mkdir(parents=True, exist_ok=True)

    prepare = load_json(kaggle_dir / "kaggle_real_runtime_acceptance.json")
    coordinator_url = args.coordinator_url or str(prepare.get("coordinator_url") or "")
    if not coordinator_url:
        raise SystemExit("--coordinator-url is required when the prepare report is missing one")
    miner_id = args.miner_id or str(prepare.get("miner_id") or "kaggle-cpu-1")
    owner = args.owner.strip()
    dataset_slug = safe_slug(args.dataset_slug)
    kernel_slug_prefix = safe_slug(args.kernel_slug_prefix)
    dataset_ref = f"{owner}/{dataset_slug}"

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
    for stage in ["stage0", "stage1"]:
        upload = kaggle_dir / f"kaggle-upload-{stage}"
        stage_dataset_dir = dataset_dir / stage
        stage_dataset_dir.mkdir(parents=True, exist_ok=True)
        for name in ["miner.private.env", "kaggle_remote_miner.py", "KAGGLE_RUN.md"]:
            source_file = upload / name
            if source_file.is_file():
                (stage_dataset_dir / name).write_text(source_file.read_text(encoding="utf-8"), encoding="utf-8")
        kernel_dir = kernels_dir / stage
        kernel_dir.mkdir(parents=True, exist_ok=True)
        kernel_slug = f"{kernel_slug_prefix}-{stage}"
        code_path = kernel_dir / "kernel.py"
        miner_env_text = (stage_dataset_dir / "miner.private.env").read_text(encoding="utf-8") if (stage_dataset_dir / "miner.private.env").is_file() else ""
        if args.inline_kernel_payload:
            code = render_inline_kernel(
                stage,
                coordinator_url=coordinator_url,
                miner_id=stage_miner_id(miner_id, stage),
                source_tarball_b64=source_tarball_b64,
                miner_env_text=miner_env_text,
            )
        else:
            code = render_kernel(stage, coordinator_url=coordinator_url, miner_id=stage_miner_id(miner_id, stage))
        code_path.write_text(code, encoding="utf-8")
        write_json(kernel_dir / "kernel-metadata.json", {
            "id": f"{owner}/{kernel_slug}",
            "title": f"{args.kernel_title_prefix} {stage}",
            "code_file": "kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "false",
            "enable_tpu": "false",
            "enable_internet": "true",
            "dataset_sources": [] if args.inline_kernel_payload else [dataset_ref],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        })
        stage_reports.append({
            "stage": stage,
            "kernel_ref": f"{owner}/{kernel_slug}",
            "kernel_dir": str(kernel_dir),
            "miner_env_present": (stage_dataset_dir / "miner.private.env").is_file(),
            "operator_env_excluded": not (stage_dataset_dir / "operator.private.env").exists(),
            "registry_excluded": not (stage_dataset_dir / "miner_registry.json").exists(),
            "inline_kernel_payload": bool(args.inline_kernel_payload),
        })

    report = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "ok": all(item["miner_env_present"] and item["operator_env_excluded"] and item["registry_excluded"] for item in stage_reports),
        "output_dir": str(output_dir),
        "kaggle_dir": str(kaggle_dir),
        "dataset_ref": dataset_ref,
        "dataset_dir": str(dataset_dir),
        "source": source,
        "coordinator_url": coordinator_url,
        "miner_id": miner_id,
        "stages": stage_reports,
        "diagnosis_codes": [
            "kaggle_micro_llm_live_package_ready",
            "kaggle_dataset_package_ready",
            "kaggle_kernel_package_ready",
        ],
        "safety": {
            "dataset_private_expected": True,
            "operator_env_excluded": True,
            "registry_excluded": True,
            "source_tarball_excludes_git_and_dist": True,
            "private_kernel_payload_contains_miner_env": bool(args.inline_kernel_payload),
            "cpu_only": True,
            "not_production": True,
            "not_p2p": True,
        },
        "commands": {
            "create_dataset": f"kaggle datasets create -p {dataset_dir} -r zip",
            "push_stage0_kernel": f"kaggle kernels push -p {kernels_dir / 'stage0'}",
            "push_stage1_kernel": f"kaggle kernels push -p {kernels_dir / 'stage1'}",
        },
    }
    write_json(output_dir / "kaggle_micro_llm_live_package.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Kaggle upload folders for the micro-LLM live proof.")
    parser.add_argument("--kaggle-dir", default="dist/kaggle-micro-llm-live/external-real")
    parser.add_argument("--output-dir", default="dist/kaggle-micro-llm-live/kaggle-cli-package")
    parser.add_argument("--owner", required=True)
    parser.add_argument("--dataset-slug", default="crowdtensor-micro-llm-live")
    parser.add_argument("--dataset-title", default="CrowdTensor Micro LLM Live Package")
    parser.add_argument("--kernel-slug-prefix", default="crowdtensor-micro-llm-live")
    parser.add_argument("--kernel-title-prefix", default="CrowdTensor Micro LLM Live Miner")
    parser.add_argument("--coordinator-url", default="")
    parser.add_argument("--miner-id", default="")
    parser.add_argument("--inline-kernel-payload", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    report = build_package(parse_args())
    if report.get("ok"):
        print(json.dumps(report, sort_keys=True))
    else:
        print(json.dumps(report, sort_keys=True))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
