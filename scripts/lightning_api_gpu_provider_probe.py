#!/usr/bin/env python3
"""Probe Lightning AI GPU availability through the Python API.

The probe is read-only by default. It intentionally does not create or start a
Studio because those calls can consume paid credits when the account is not
eligible for free Studio sessions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


SCHEMA = "lightning_api_gpu_provider_probe_v1"


def sha16(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8", errors="replace")).hexdigest()[:16]


def parse_token_file(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    values: dict[str, str] = {}
    for key in ["LIGHTNING_API_KEY", "LIGHTNING_USER_ID"]:
        match = re.search(rf"(?m)^\s*{key}\s*[:=]\s*(.+?)\s*$", text)
        if match:
            values[key] = match.group(1).strip().strip("\"'")
    return values


def scrub_error(exc: BaseException) -> dict[str, Any]:
    body = str(getattr(exc, "body", "") or "")
    return {
        "type": type(exc).__name__,
        "status": getattr(exc, "status", None),
        "reason": getattr(exc, "reason", None),
        "body_hash": sha16(body) if body else "",
        "body_chars": len(body),
    }


def call_readonly(name: str, fn: Any) -> dict[str, Any]:
    started = time.monotonic()
    try:
        obj = fn()
    except Exception as exc:  # pragma: no cover - exercised by live API only
        return {
            "name": name,
            "ok": False,
            "duration_seconds": round(time.monotonic() - started, 3),
            "error": scrub_error(exc),
        }
    return {
        "name": name,
        "ok": True,
        "duration_seconds": round(time.monotonic() - started, 3),
        "object": obj,
    }


def to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "to_dict"):
        data = obj.to_dict()
        return data if isinstance(data, dict) else {}
    if isinstance(obj, dict):
        return obj
    return {}


def finite_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_accelerators(response: Any) -> dict[str, Any]:
    data = to_dict(response)
    items = data.get("accelerator") if isinstance(data.get("accelerator"), list) else []
    summarized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        resources = item.get("resources") if isinstance(item.get("resources"), dict) else {}
        is_gpu = str(item.get("accelerator_type") or "").upper() == "GPU" or int(resources.get("gpu") or 0) > 0
        if not is_gpu:
            continue
        summarized.append(
            {
                "slug": str(item.get("slug") or ""),
                "slug_multi_cloud": str(item.get("slug_multi_cloud") or ""),
                "display_name": str(item.get("display_name") or ""),
                "family": str(item.get("family") or ""),
                "provider": str(item.get("provider") or ""),
                "enabled": item.get("enabled") is True,
                "out_of_capacity": item.get("out_of_capacity") is True,
                "is_tier_restricted": item.get("is_tier_restricted") is True,
                "cost_per_hour": finite_float(item.get("cost")),
                "spot_price_per_hour": finite_float(item.get("spot_price")),
                "available_in_seconds": str(item.get("available_in_seconds") or ""),
                "available_in_seconds_spot": str(item.get("available_in_seconds_spot") or ""),
                "quota_value": str(item.get("quota_value") or ""),
                "quota_utilization": str(item.get("quota_utilization") or ""),
                "max_available_quota": str(item.get("max_available_quota") or ""),
                "resources": {
                    "cpu": resources.get("cpu"),
                    "gpu": resources.get("gpu"),
                    "gpu_type": str(resources.get("gpu_type") or ""),
                    "memory_mb": str(resources.get("memory_mb") or ""),
                    "storage_gb": str(resources.get("storage_gb") or ""),
                },
            }
        )
    enabled = [item for item in summarized if item["enabled"] and not item["out_of_capacity"]]
    costs = [item["cost_per_hour"] for item in enabled if item.get("cost_per_hour") is not None]
    cheapest = min(costs) if costs else None
    return {
        "total_accelerator_count": len(items),
        "gpu_accelerator_count": len(summarized),
        "enabled_gpu_not_out_of_capacity_count": len(enabled),
        "cheapest_enabled_gpu_cost_per_hour": cheapest,
        "all_enabled_gpu_costs_positive": bool(costs) and all(cost > 0 for cost in costs),
        "gpu_skus": summarized[:40],
    }


def summarize_memberships(response: Any) -> dict[str, Any]:
    data = to_dict(response)
    memberships = data.get("memberships") if isinstance(data.get("memberships"), list) else []
    projects: list[dict[str, Any]] = []
    for item in memberships:
        if not isinstance(item, dict):
            continue
        project_id = item.get("project_id")
        projects.append(
            {
                "project_id_hash": sha16(project_id),
                "project_id_public": False,
                "role": str(item.get("role") or ""),
                "name_hash": sha16(item.get("name")),
                "display_name_hash": sha16(item.get("display_name")),
            }
        )
    return {
        "membership_count": len(memberships),
        "projects": projects,
        "project_ids_public": False,
    }


def summarize_user(response: Any) -> dict[str, Any]:
    data = to_dict(response)
    return {
        "user_id_hash": sha16(data.get("id")),
        "organizations_count": len(data.get("organizations") or []) if isinstance(data.get("organizations"), list) else 0,
        "user_id_public": False,
    }


def summarize_balance(response: Any) -> dict[str, Any]:
    data = to_dict(response)
    balance = finite_float(data.get("balance"))
    balance_limit = finite_float(data.get("balance_limit"))
    return {
        "balance": balance,
        "balance_limit": balance_limit,
        "account_id_hash": sha16(data.get("account_id")),
        "project_id_hash": sha16(data.get("project_id")),
        "transactions_count": len(data.get("transactions") or []) if isinstance(data.get("transactions"), list) else 0,
        "ids_public": False,
    }


def summarize_cloud_space_instances(response: Any) -> dict[str, Any]:
    data = to_dict(response)
    instances = data.get("cloudspace_instances") if isinstance(data.get("cloudspace_instances"), list) else []
    return {
        "can_start_free_cloud_space": data.get("can_start_free_cloud_space") is True,
        "cloudspace_instance_count": len(instances),
        "instance_ids_public": False,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    token_path = Path(args.token_file).expanduser()
    token_values = parse_token_file(token_path)
    for key, value in token_values.items():
        os.environ[key] = value

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "objective": "Determine whether the provided Lightning AI account can supply a stable free/zero-balance GPU worker.",
        "token_file": {
            "path": str(token_path),
            "exists": token_path.is_file(),
            "api_key_present": bool(token_values.get("LIGHTNING_API_KEY")),
            "user_id_present": bool(token_values.get("LIGHTNING_USER_ID")),
            "api_key_hash": sha16(token_values.get("LIGHTNING_API_KEY")) if token_values.get("LIGHTNING_API_KEY") else "",
            "user_id_hash": sha16(token_values.get("LIGHTNING_USER_ID")) if token_values.get("LIGHTNING_USER_ID") else "",
            "secret_values_public": False,
        },
        "api_auth_verified": False,
        "read_only": True,
        "create_or_start_attempted": False,
        "credentials_public": False,
        "private_runtime_state_public": False,
        "public_artifact_safe": True,
        "blockers": [],
    }
    if not token_values.get("LIGHTNING_API_KEY") or not token_values.get("LIGHTNING_USER_ID"):
        report["blockers"].append("lightning_token_file_missing_required_keys")
        return report

    try:
        from lightning_cloud import rest_client
    except Exception as exc:  # pragma: no cover - environment dependent
        report["blockers"].append("lightning_sdk_import_failed")
        report["sdk_import_error"] = scrub_error(exc)
        return report

    try:
        client = rest_client.LightningClient(retry=False, with_auth=True)
    except Exception as exc:  # pragma: no cover - live auth only
        report["blockers"].append("lightning_client_creation_failed")
        report["client_error"] = scrub_error(exc)
        return report

    user_call = call_readonly("auth_service_get_user", client.auth_service_get_user)
    if user_call["ok"]:
        report["api_auth_verified"] = True
        report["user"] = summarize_user(user_call.pop("object"))
    else:
        report["blockers"].append("lightning_api_auth_failed")
        report["auth_error"] = user_call["error"]
        return report

    balance_call = call_readonly("billing_service_get_user_balance", client.billing_service_get_user_balance)
    if balance_call["ok"]:
        user_balance = finite_float(to_dict(balance_call.pop("object")).get("balance"))
        report["user_balance"] = {"balance": user_balance}
        if user_balance is None or user_balance <= 0:
            report["blockers"].append("lightning_zero_or_unknown_user_balance")
    else:
        report["user_balance_error"] = balance_call["error"]
        report["blockers"].append("lightning_user_balance_unavailable")

    membership_call = call_readonly("projects_service_list_memberships", client.projects_service_list_memberships)
    project_ids: list[str] = []
    if membership_call["ok"]:
        membership_obj = membership_call.pop("object")
        membership_data = to_dict(membership_obj)
        report["project_memberships"] = summarize_memberships(membership_obj)
        for item in membership_data.get("memberships") or []:
            if isinstance(item, dict) and item.get("project_id"):
                project_ids.append(str(item["project_id"]))
    else:
        report["project_membership_error"] = membership_call["error"]
        report["blockers"].append("lightning_project_memberships_unavailable")

    accel_call = call_readonly("cluster_service_list_default_cluster_accelerators", client.cluster_service_list_default_cluster_accelerators)
    if accel_call["ok"]:
        accel_summary = summarize_accelerators(accel_call.pop("object"))
        report["default_accelerators"] = accel_summary
        if accel_summary["gpu_accelerator_count"] == 0:
            report["blockers"].append("lightning_no_gpu_skus_visible")
        if accel_summary["all_enabled_gpu_costs_positive"]:
            report["blockers"].append("lightning_enabled_gpu_skus_have_positive_cost")
    else:
        report["default_accelerators_error"] = accel_call["error"]
        report["blockers"].append("lightning_default_accelerators_unavailable")

    instances_call = call_readonly("cloud_space_service_list_cloud_space_instances", client.cloud_space_service_list_cloud_space_instances)
    if instances_call["ok"]:
        instances_summary = summarize_cloud_space_instances(instances_call.pop("object"))
        report["cloud_space_instances"] = instances_summary
        if not instances_summary["can_start_free_cloud_space"]:
            report["blockers"].append("lightning_free_cloud_space_start_not_allowed")
    else:
        report["cloud_space_instances_error"] = instances_call["error"]
        report["blockers"].append("lightning_cloud_space_instances_unavailable")

    project_summaries: list[dict[str, Any]] = []
    for project_id in project_ids[:5]:
        item: dict[str, Any] = {"project_id_hash": sha16(project_id), "project_id_public": False}
        project_balance = call_readonly(
            "billing_service_get_project_balance",
            lambda project_id=project_id: client.billing_service_get_project_balance(project_id=project_id),
        )
        if project_balance["ok"]:
            item["balance"] = summarize_balance(project_balance.pop("object"))
            balance = item["balance"].get("balance")
            limit = item["balance"].get("balance_limit")
            if balance is None or balance <= 0:
                item.setdefault("blockers", []).append("project_zero_or_unknown_balance")
            if limit is None or limit <= 0:
                item.setdefault("blockers", []).append("project_zero_or_unknown_balance_limit")
        else:
            item["balance_error"] = project_balance["error"]
        cloud_spaces = call_readonly(
            "cloud_space_service_list_cloud_spaces",
            lambda project_id=project_id: client.cloud_space_service_list_cloud_spaces(project_id=project_id),
        )
        if cloud_spaces["ok"]:
            data = to_dict(cloud_spaces.pop("object"))
            spaces = data.get("cloudspaces") if isinstance(data.get("cloudspaces"), list) else []
            item["cloudspace_count"] = len(spaces)
        else:
            item["cloudspaces_error"] = cloud_spaces["error"]
        project_summaries.append(item)
    report["projects"] = project_summaries

    project_has_positive_balance = any(
        (project.get("balance") or {}).get("balance", 0) and (project.get("balance") or {}).get("balance", 0) > 0
        for project in project_summaries
    )
    can_free = bool((report.get("cloud_space_instances") or {}).get("can_start_free_cloud_space"))
    cheapest = (report.get("default_accelerators") or {}).get("cheapest_enabled_gpu_cost_per_hour")
    report["safe_to_attempt_zero_balance_gpu_start"] = bool(
        report["api_auth_verified"] and (can_free or project_has_positive_balance or cheapest == 0)
    )
    if not report["safe_to_attempt_zero_balance_gpu_start"]:
        report["blockers"].append("lightning_no_zero_balance_gpu_start_path_visible")
    report["recommended_action"] = (
        "do_not_start_gpu_without_credits_or_free_cloud_space_flag"
        if not report["safe_to_attempt_zero_balance_gpu_start"]
        else "eligible_for_bounded_gpu_start_probe"
    )
    report["blockers"] = sorted(set(str(item) for item in report["blockers"] if item))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-file", default="~/.config/crowdtensor/lightning-token.md")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args)
    path = output_dir / "lightning_api_gpu_provider_probe.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Report: {path}")
    return 0 if report.get("public_artifact_safe") else 1


if __name__ == "__main__":
    raise SystemExit(main())
