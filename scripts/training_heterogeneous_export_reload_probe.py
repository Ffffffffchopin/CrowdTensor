#!/usr/bin/env python3
"""Reload a completed PEFT export and run one full stagewise Qwen forward."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowdtensor.heterogeneous_qwen_source import (  # noqa: E402
    materialize_qwen_stage_shard,
    qwen_stage_spec,
)
from crowdtensor.heterogeneous_training_manifest import (  # noqa: E402
    stable_hash,
    validate_training_manifest,
)
from crowdtensor.qwen15b_training import (  # noqa: E402
    load_qwen_pipeline_stage,
    sha256_bytes,
    sha256_file,
)


SCHEMA = "crowdtensor_heterogeneous_training_export_reload_probe_v1"
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
LORA_RE = re.compile(r"\.lora_(A|B)\.weight$")
MAX_EXPORT_BYTES = 512 * 1024 * 1024


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _private_configuration(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("heterogeneous_export_private_configuration_invalid")
    if not value.get("coordinator_url") or not value.get("coordinator_token"):
        raise RuntimeError("heterogeneous_export_private_configuration_incomplete")
    return value


def _request(
    private: dict[str, Any],
    path: str,
    *,
    timeout: float,
) -> tuple[bytes, dict[str, str]]:
    request = urllib.request.Request(
        str(private["coordinator_url"]).rstrip("/") + path,
        headers={
            "User-Agent": "crowdtensor-heterogeneous-export-reload/1",
            "x-crowdtensor-miner-token": str(private["coordinator_token"]),
        },
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=float(timeout)) as response:
        payload = response.read(MAX_EXPORT_BYTES + 1)
        headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
    if len(payload) > MAX_EXPORT_BYTES:
        raise RuntimeError("heterogeneous_export_bundle_too_large")
    return payload, headers


def _bootstrap(private: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    payload, _headers = _request(
        private, "/elastic-training/bootstrap", timeout=timeout
    )
    value = json.loads(payload)
    if (
        not isinstance(value, dict)
        or value.get("schema")
        != "crowdtensor_heterogeneous_training_beta_miner_bootstrap_v1"
    ):
        raise RuntimeError("heterogeneous_export_bootstrap_invalid")
    return value


def _download_export(
    private: dict[str, Any],
    *,
    timeout: float,
) -> tuple[bytes, dict[str, str]]:
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        try:
            payload, headers = _request(
                private,
                "/elastic-training/export-bundle",
                timeout=min(180.0, max(1.0, deadline - time.monotonic())),
            )
            expected = str(headers.get("x-crowdtensor-export-hash") or "")
            if expected != sha256_bytes(payload):
                raise RuntimeError("heterogeneous_export_bundle_hash_mismatch")
            return payload, headers
        except urllib.error.HTTPError as exc:
            if int(exc.code) != 409:
                raise
        time.sleep(5.0)
    raise TimeoutError("heterogeneous_export_bundle_wait_timeout")


def _extract_export(payload: bytes, output: Path) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    expected = {"adapter_config.json", "adapter_model.safetensors"}
    with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
        names = {str(PurePosixPath(info.filename)) for info in archive.infolist()}
        if names != expected or any(info.is_dir() for info in archive.infolist()):
            raise RuntimeError("heterogeneous_export_bundle_members_invalid")
        if sum(int(info.file_size) for info in archive.infolist()) > MAX_EXPORT_BYTES:
            raise RuntimeError("heterogeneous_export_bundle_unpacked_size_invalid")
        for info in archive.infolist():
            target = output / str(PurePosixPath(info.filename))
            target.write_bytes(archive.read(info))
    return output / "adapter_config.json", output / "adapter_model.safetensors"


def _stage_adapter_state(
    exported: dict[str, Any],
    *,
    layer_start: int,
    layer_end: int,
) -> tuple[dict[str, Any], list[str]]:
    selected: dict[str, Any] = {}
    names = []
    for name, tensor in exported.items():
        match = LAYER_RE.search(str(name))
        if match is None or not int(layer_start) <= int(match.group(1)) < int(
            layer_end
        ):
            continue
        if not str(name).startswith("base_model.model.model.layers."):
            raise RuntimeError("heterogeneous_export_adapter_name_invalid")
        target = str(name).removeprefix("base_model.model.")
        target = LORA_RE.sub(r".lora_\1.default.weight", target)
        selected[target] = tensor
        names.append(str(name))
    if not selected:
        raise RuntimeError("heterogeneous_export_stage_adapter_missing")
    return selected, sorted(names)


def _tensor_hash(value: Any) -> str:
    raw = value.detach().to("cpu").contiguous().view(__import__("torch").uint8)
    return "sha256:" + hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def verify_exported_forward(
    *,
    private_configuration_path: str | Path,
    private_root: str | Path,
    output_path: str | Path,
    wait_timeout: float,
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    started = time.time()
    private = _private_configuration(private_configuration_path)
    bootstrap = _bootstrap(private, timeout=min(180.0, float(wait_timeout)))
    manifest = validate_training_manifest(bootstrap["training_manifest"])
    config = dict(bootstrap["config"])
    tokenized = dict(bootstrap["tokenized_payload"])
    if (
        stable_hash(config) != str(bootstrap.get("config_hash") or "")
        or stable_hash(tokenized)
        != str(bootstrap.get("tokenized_payload_hash") or "")
    ):
        raise RuntimeError("heterogeneous_export_bootstrap_hash_mismatch")
    root = Path(private_root).resolve()
    shutil.rmtree(root, ignore_errors=True)
    export_root = root / "adapter"
    shard_root = root / "shards"
    root.mkdir(parents=True, exist_ok=True)
    payload, _headers = _download_export(
        private, timeout=max(60.0, float(wait_timeout) * 0.1)
    )
    config_path, adapter_path = _extract_export(payload, export_root)
    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        adapter_config.get("base_model_name_or_path")
        != manifest["model"]["model_id"]
        or adapter_config.get("revision") != manifest["model"]["model_revision"]
        or int(adapter_config.get("r") or 0) != int(manifest["lora"]["rank"])
        or int(adapter_config.get("lora_alpha") or 0)
        != int(manifest["lora"]["alpha"])
    ):
        raise RuntimeError("heterogeneous_export_adapter_model_binding_invalid")
    exported = load_file(str(adapter_path), device="cpu")
    layer_indexes = sorted(
        {
            int(match.group(1))
            for name in exported
            if (match := LAYER_RE.search(name)) is not None
        }
    )
    if layer_indexes != list(range(int(manifest["model"]["num_hidden_layers"]))):
        raise RuntimeError("heterogeneous_export_adapter_layer_coverage_invalid")
    rows = list(tokenized.get("validation") or [])
    if not rows:
        raise RuntimeError("heterogeneous_export_validation_tokens_missing")
    hidden: Any = torch.as_tensor([rows[0]], dtype=torch.long)
    stage_reports = []
    torch.manual_seed(int(manifest["training"]["seed"]))
    torch.use_deterministic_algorithms(True, warn_only=True)
    for stage in manifest["stages"]:
        stage_id = int(stage["stage_id"])
        shard = shard_root / f"stage{stage_id}.safetensors"
        shard_report = materialize_qwen_stage_shard(
            manifest,
            stage_id=stage_id,
            output_path=shard,
            token=str(private.get("hf_token") or ""),
            source_root=source_root,
        )
        module, load_report = load_qwen_pipeline_stage(
            config,
            qwen_stage_spec(manifest, stage_id=stage_id),
            shard,
            device="cpu",
            compute_dtype="float32",
            inject_lora=True,
            lora_rank=int(manifest["lora"]["rank"]),
            lora_alpha=int(manifest["lora"]["alpha"]),
            lora_target_modules=manifest["lora"]["target_modules"],
            lora_dropout=0.0,
            lora_seed=int(manifest["training"]["seed"]),
            gradient_checkpointing=False,
            model_id=manifest["model"]["model_id"],
            model_revision=manifest["model"]["model_revision"],
        )
        selected, selected_names = _stage_adapter_state(
            exported,
            layer_start=int(stage["layer_start"]),
            layer_end=int(stage["layer_end"]),
        )
        module_state = module.state_dict()
        if set(selected) - set(module_state):
            raise RuntimeError("heterogeneous_export_stage_adapter_target_missing")
        for name, tensor in selected.items():
            if list(module_state[name].shape) != list(tensor.shape):
                raise RuntimeError("heterogeneous_export_stage_adapter_shape_mismatch")
        incompatible = module.load_state_dict(selected, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError("heterogeneous_export_stage_adapter_unexpected_key")
        module.eval()
        with torch.inference_mode():
            hidden = module(hidden)
        if not bool(torch.isfinite(hidden.float()).all().item()):
            raise RuntimeError("heterogeneous_export_non_finite_forward")
        stage_reports.append(
            {
                "stage_id": stage_id,
                "device_type": "cpu",
                "loaded_layer_indexes": load_report["loaded_layer_indexes"],
                "adapter_tensor_count": len(selected),
                "adapter_tensor_names_hash": stable_hash(selected_names),
                "source_shard_hash": shard_report["shard_file_hash"],
                "output_shape": list(hidden.shape),
                "output_hash": _tensor_hash(hidden),
                "finite_output": True,
                "tensor_values_public": False,
                "public_artifact_safe": True,
            }
        )
        del module, module_state, selected
        shard.unlink(missing_ok=True)
        gc.collect()
    report = {
        "schema": SCHEMA,
        "ok": True,
        "model_id": manifest["model"]["model_id"],
        "model_revision": manifest["model"]["model_revision"],
        "training_manifest_hash": manifest["content_hash"],
        "standard_peft_format": True,
        "model_binding_verified": True,
        "adapter_reload_verified": True,
        "forward_inference_verified": True,
        "finite_logits_verified": True,
        "all_five_stages_present": len(stage_reports) == 5,
        "adapter_file_hash": sha256_file(adapter_path),
        "adapter_tensor_count": len(exported),
        "adapter_tensor_names_hash": stable_hash(sorted(exported)),
        "layer_indexes": layer_indexes,
        "stage_reports": stage_reports,
        "final_logits_shape": list(hidden.shape),
        "final_logits_hash": _tensor_hash(hidden),
        "elapsed_seconds": round(time.time() - started, 3),
        "raw_training_text_public": False,
        "token_ids_public": False,
        "logit_values_public": False,
        "adapter_tensor_values_public": False,
        "credential_values_public": False,
        "coordinator_url_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    report["content_hash"] = stable_hash(report)
    _write_json(Path(output_path), report)
    shutil.rmtree(root, ignore_errors=True)
    return report


def _public_error(exc: BaseException) -> str:
    value = str(exc)
    if value.startswith("heterogeneous_"):
        return re.sub(r"[^a-zA-Z0-9:_-]", "_", value[:180])
    return f"heterogeneous_export_reload_failed:{type(exc).__name__}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-configuration", required=True)
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wait-timeout", type=float, default=10800.0)
    parser.add_argument("--attached-model-root", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        report = verify_exported_forward(
            private_configuration_path=args.private_configuration,
            private_root=args.private_root,
            output_path=args.output,
            wait_timeout=args.wait_timeout,
            source_root=args.attached_model_root or None,
        )
    except BaseException as exc:
        report = {
            "schema": SCHEMA,
            "ok": False,
            "standard_peft_format": False,
            "adapter_reload_verified": False,
            "forward_inference_verified": False,
            "finite_logits_verified": False,
            "blockers": [_public_error(exc)],
            "failure_detail_public": False,
            "raw_training_text_public": False,
            "token_ids_public": False,
            "logit_values_public": False,
            "adapter_tensor_values_public": False,
            "credential_values_public": False,
            "coordinator_url_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        report["content_hash"] = stable_hash(report)
        _write_json(Path(args.output), report)
    finally:
        shutil.rmtree(Path(args.private_root), ignore_errors=True)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    return 0 if report.get("ok") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
