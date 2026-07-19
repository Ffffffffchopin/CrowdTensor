#!/usr/bin/env python3
"""Pack plugin, live training, quality, and cleanup proof into a portable RC."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from crowdtensor.community_security import scan_public_value
from crowdtensor.model_adapter import stable_hash
from scripts.mistral_kaggle_live_check import check_report as check_live
from scripts.mistral_kaggle_live_ledger import validate_ledger
from scripts.model_ecosystem_beta_rc_check import check_report


SCHEMA = "crowdtensor_model_ecosystem_beta_rc_v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("model_ecosystem_rc_source_invalid")
    return value


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy(name: str, source: Path, output: Path) -> tuple[Path, dict[str, Any]]:
    destination = output / "evidence" / name / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination, {
        "relative_path": str(destination.relative_to(output)),
        "sha256": _hash(destination),
        "size_bytes": destination.stat().st_size,
    }


def pack(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        "core_wheel": Path(args.core_wheel).expanduser().resolve(),
        "adapter_wheel": Path(args.adapter_wheel).expanduser().resolve(),
        "plugin_smoke": Path(args.plugin_smoke).expanduser().resolve(),
        "live_report": Path(args.live_report).expanduser().resolve(),
        "live_check": Path(args.live_check).expanduser().resolve(),
        "attempt_ledger": Path(args.attempt_ledger).expanduser().resolve(),
        "local_gate": Path(args.local_gate).expanduser().resolve(),
        "adapter_config": Path(args.adapter_dir).expanduser().resolve()
        / "adapter_config.json",
        "adapter_weights": Path(args.adapter_dir).expanduser().resolve()
        / "adapter_model.safetensors",
    }
    if any(not path.is_file() for path in sources.values()):
        raise ValueError("model_ecosystem_rc_required_source_missing")
    artifacts: dict[str, dict[str, Any]] = {}
    portable: dict[str, Path] = {}
    for name, source in sources.items():
        portable[name], artifacts[name] = _copy(name, source, output)
    plugin = _read(portable["plugin_smoke"])
    live = _read(portable["live_report"])
    live_check = _read(portable["live_check"])
    ledger = _read(portable["attempt_ledger"])
    local = _read(portable["local_gate"])
    strict_live = check_live(live)
    plugin_ready = bool(
        plugin.get("ok") is True
        and plugin.get("registration_kind") == "entry_point_plugin"
        and plugin.get("conformance_verified") is True
        and plugin.get("workspace_import_used") is False
    )
    live_ready = bool(
        strict_live.get("ok") is True
        and live_check.get("ok") is True
        and live_check.get("mistral_live_verified") is True
    )
    ledger_ready = not validate_ledger(ledger) and (
        (ledger.get("attempts") or [{}])[-1].get("outcome") == "achieved"
    ) and (
        (ledger.get("attempts") or [{}])[-1].get("strict_status") == "current"
    ) and all(
        item.get("strict_status") == "superseded"
        for item in (ledger.get("attempts") or [])[:-1]
    )
    quality_ready = bool(
        local.get("ok") is True
        and int(local.get("passed") or 0) >= 40
        and int(local.get("failed") or 0) == 0
    )
    cleanup = dict(live.get("cleanup") or {})
    cleanup_ready = bool(
        live.get("cleanup_verified") is True
        and cleanup.get("all_remote_kernels_deleted") is True
        and cleanup.get("live_resources_left_running") is False
        and cleanup.get("private_runtime_removed") is True
        and ledger.get("community_maturity_ledger_modified") is False
    )
    replacement = dict(live.get("gpu_worker_replacement") or {})
    exported = dict(live.get("export") or {})
    reload = dict(live.get("reload") or {})
    report = {
        "schema": SCHEMA,
        "model_adapter_ecosystem_ready": plugin_ready and live_ready,
        "entry_point_plugin_contract_ready": plugin_ready,
        "mistral_adapter_ready": plugin_ready and live_ready,
        "mistral_real_heterogeneous_training_verified": live_ready,
        "dual_wheel_clean_install_verified": bool(
            plugin_ready
            and (live.get("plugin_installation") or {}).get(
                "both_wheels_installed_in_fresh_environment"
            )
            is True
        ),
        "checkpoint_replacement_verified": bool(
            replacement.get("verified") is True
            and replacement.get("restored_checkpoint_step") == 4
            and replacement.get("optimizer_state_restored") is True
        ),
        "peft_export_reload_verified": bool(
            exported.get("standard_peft_format") is True
            and int(exported.get("adapter_tensor_count") or 0) == 168
            and reload.get("adapter_reload_verified") is True
            and reload.get("independent_process_reload") is True
        ),
        "regression_gate_verified": quality_ready,
        "cleanup_verified": cleanup_ready,
        "goal_achieved": bool(
            plugin_ready and live_ready and ledger_ready and quality_ready and cleanup_ready
        ),
        "supported_model_families": sorted(
            plugin.get("supported_model_families") or []
        ),
        "plugin_smoke_summary": {
            key: plugin.get(key)
            for key in (
                "ok",
                "adapter_id",
                "family",
                "model_id",
                "model_revision",
                "registration_kind",
                "entry_point_group",
                "conformance_verified",
                "partition_verified",
                "isolated_venv",
                "wheel_install_no_deps",
                "workspace_import_used",
            )
        },
        "live_summary": {
            "strict_check_ok": strict_live.get("ok") is True,
            "mistral_live_verified": live_check.get("mistral_live_verified") is True,
            "live_run_performed": live.get("live_run_performed") is True,
            "committed_step_ids": list(
                (live.get("final_status") or {}).get("committed_step_ids") or []
            ),
            "accepted_providers": list(live.get("accepted_providers") or []),
            "gpu_worker_replacement_verified": replacement.get("verified") is True,
            "restored_checkpoint_step": int(
                replacement.get("restored_checkpoint_step") or 0
            ),
            "optimizer_state_restored": replacement.get("optimizer_state_restored")
            is True,
            "adapter_tensor_count": int(exported.get("adapter_tensor_count") or 0),
            "adapter_reload_verified": reload.get("adapter_reload_verified") is True,
            "cleanup_verified": cleanup_ready,
            "public_safety_verified": (live.get("public_safety") or {}).get("ok")
            is True,
            "duration_seconds": float(
                (live.get("benchmark") or {}).get("duration_seconds") or 0.0
            ),
            "source_content_hash": str(live.get("content_hash") or ""),
        },
        "quality_summary": {
            "ok": local.get("ok") is True,
            "passed": int(local.get("passed") or 0),
            "failed": int(local.get("failed") or 0),
            "py_compile_ok": (local.get("py_compile") or {}).get("ok") is True,
            "plugin_registry_tests_included": local.get(
                "plugin_registry_tests_included"
            )
            is True,
            "mistral_architecture_tests_included": local.get(
                "mistral_architecture_tests_included"
            )
            is True,
            "live_report_checker_tests_included": local.get(
                "live_report_checker_tests_included"
            )
            is True,
        },
        "cleanup": {
            "all_remote_kernels_deleted": cleanup.get("all_remote_kernels_deleted")
            is True,
            "live_resources_left_running": cleanup.get("live_resources_left_running")
            is True,
            "private_runtime_removed": cleanup.get("private_runtime_removed") is True,
            "community_maturity_ledger_modified": ledger.get(
                "community_maturity_ledger_modified"
            )
            is True,
        },
        "artifacts": artifacts,
        "unsupported_claims": {
            "arbitrary_architecture_support_verified": False,
            "full_parameter_training_verified": False,
            "mistral_7b_live_verified": False,
            "physical_multi_machine_verified": False,
            "production_sla_verified": False,
        },
        "credential_values_public": False,
        "credential_paths_public": False,
        "coordinator_url_public": False,
        "raw_training_text_public": False,
        "token_ids_public": False,
        "activation_values_public": False,
        "gradient_values_public": False,
        "checkpoint_tensor_values_public": False,
        "adapter_tensor_values_public": False,
        "private_paths_public": False,
        "public_artifact_safe": True,
    }
    safety = scan_public_value(report)
    report["public_safety"] = safety
    report["public_artifact_safe"] = safety["ok"] is True
    report["goal_achieved"] = bool(report["goal_achieved"] and safety["ok"])
    report["content_hash"] = stable_hash(report)
    destination = output / "model_ecosystem_beta_rc.json"
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checked = check_report(report, artifact_root=output)
    (output / "model_ecosystem_beta_rc_check.json").write_text(
        json.dumps(checked, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if checked["ok"] is not True:
        raise RuntimeError("model_ecosystem_beta_rc_check_failed:" + checked["errors"][0])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--core-wheel", required=True)
    parser.add_argument("--adapter-wheel", required=True)
    parser.add_argument("--plugin-smoke", required=True)
    parser.add_argument("--live-report", required=True)
    parser.add_argument("--live-check", required=True)
    parser.add_argument("--attempt-ledger", required=True)
    parser.add_argument("--local-gate", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = pack(args)
    print(json.dumps(report, sort_keys=True) if args.json else f"goal_achieved={report['goal_achieved']}")
    return 0 if report["goal_achieved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
