#!/usr/bin/env python3
"""Static release gate for the CrowdTensorD Alpha package."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "AGENTS.md",
    "README.md",
    "CHANGELOG.md",
    "ROADMAP.md",
    "LICENSE",
    "pyproject.toml",
    "crowdtensor/cli.py",
    "requirements.txt",
    "Dockerfile",
    "compose.yaml",
    ".dockerignore",
    ".env.example",
    ".github/release.yml",
    ".github/ISSUE_TEMPLATE/bug_report.md",
    ".github/ISSUE_TEMPLATE/feature_request.md",
    ".github/pull_request_template.md",
    ".github/workflows/ci.yml",
    "scripts/api_contract_check.py",
    "scripts/result_idempotency_check.py",
    "scripts/result_ledger_check.py",
    "scripts/create_miner_invite.py",
    "scripts/create_operator_invite.py",
    "scripts/remote_miner_join_check.py",
    "scripts/miner_resilience_check.py",
    "scripts/readiness_check.py",
    "scripts/remote_demo_runbook_pack.py",
    "scripts/remote_demo_runbook_check.py",
    "scripts/remote_demo_acceptance_pack.py",
    "scripts/remote_demo_acceptance_check.py",
    "scripts/remote_home_compute_demo_pack.py",
    "scripts/remote_home_compute_demo_check.py",
    "scripts/remote_two_machine_beta_check.py",
    "scripts/kaggle_remote_miner_beta_check.py",
    "scripts/kaggle_real_runtime_acceptance_pack.py",
    "scripts/kaggle_real_runtime_acceptance_check.py",
    "scripts/demo_manifest_pack.py",
    "scripts/demo_manifest_check.py",
    "scripts/onboarding_gate.py",
    "scripts/release_readiness_pack.py",
    "scripts/release_readiness_check.py",
    "scripts/remote_compute_evidence_pack.py",
    "scripts/remote_compute_evidence_check.py",
    "scripts/multi_miner_scenario_sweep.py",
    "scripts/multi_miner_scenario_sweep_check.py",
    "scripts/runtime_matrix.py",
    "scripts/runtime_matrix_check.py",
    "scripts/user_friendly_inference_frontdoor_check.py",
    "scripts/cpu_inference_beta_pack.py",
    "scripts/cpu_inference_beta_check.py",
    "scripts/cpu_inference_beta_rc_pack.py",
    "scripts/cpu_inference_beta_rc_check.py",
    "scripts/sharded_inference_evidence_pack.py",
    "scripts/sharded_inference_check.py",
    "scripts/remote_sharded_inference_beta_pack.py",
    "scripts/remote_sharded_inference_beta_check.py",
    "scripts/micro_llm_sharded_inference_evidence_pack.py",
    "scripts/micro_llm_sharded_inference_check.py",
    "scripts/stage_aware_micro_llm_sharded_check.py",
    "scripts/remote_micro_llm_sharded_beta_pack.py",
    "scripts/remote_micro_llm_sharded_beta_check.py",
    "scripts/real_llm_sharded_inference_evidence_pack.py",
    "scripts/remote_real_llm_sharded_beta_pack.py",
    "scripts/remote_real_llm_sharded_beta_check.py",
    "scripts/real_llm_live_rc_pack.py",
    "scripts/real_llm_live_rc_check.py",
    "scripts/real_llm_internet_alpha_pack.py",
    "scripts/real_llm_internet_alpha_check.py",
    "scripts/real_llm_internet_beta_pack.py",
    "scripts/real_llm_internet_beta_check.py",
    "scripts/swarm_inference_beta_pack.py",
    "scripts/swarm_inference_beta_check.py",
    "scripts/public_swarm_inference_beta_pack.py",
    "scripts/public_swarm_inference_beta_check.py",
    "scripts/public_swarm_inference_beta_rc_pack.py",
    "scripts/public_swarm_inference_beta_rc_check.py",
    "scripts/public_swarm_product_beta_pack.py",
    "scripts/public_swarm_product_beta_check.py",
    "scripts/public_swarm_developer_preview_pack.py",
    "scripts/public_swarm_developer_preview_check.py",
    "scripts/public_swarm_live_preview_rc_pack.py",
    "scripts/public_swarm_live_preview_rc_check.py",
    "scripts/public_swarm_operator_preview_pack.py",
    "scripts/public_swarm_operator_preview_check.py",
    "scripts/public_swarm_trial_pack.py",
    "scripts/public_swarm_trial_check.py",
    "scripts/public_swarm_gpu_inference_beta_pack.py",
    "scripts/public_swarm_gpu_inference_beta_check.py",
    "scripts/kaggle_real_llm_live_package.py",
    "scripts/home_compute_demo.py",
    "scripts/home_compute_demo_check.py",
    "scripts/home_compute_evidence_pack.py",
    "scripts/home_compute_evidence_check.py",
    "scripts/doctor.py",
    "scripts/support_bundle.py",
    "scripts/hash_token.py",
    "scripts/token_hash_auth_check.py",
    "scripts/security_preflight.py",
    "scripts/browser_acceptance_pack.py",
    "scripts/release_evidence_pack.py",
    "scripts/outer_optimizer_check.py",
    "scripts/compressed_error_feedback_check.py",
    "scripts/delta_transport_negotiation_check.py",
    "scripts/model_bundle_smoke.py",
    "scripts/model_bundle_inference_smoke.py",
    "scripts/inference_session_demo.py",
    "scripts/inference_session_client.py",
    "scripts/inference_session_client_check.py",
    "scripts/admin_inference_session_check.py",
    "scripts/external_llm_inference_smoke.py",
    "scripts/external_llm_http_adapter_smoke.py",
    "scripts/external_llm_evidence_pack.py",
    "scripts/external_llm_evidence_check.py",
    "docs/api.md",
    "docs/protocol.md",
    "docs/project-memory.md",
    "docs/use-cases.md",
    "docs/quickstart.md",
    "docs/remote-miner.md",
    "docs/architecture.md",
    "docs/security.md",
    "docs/operations.md",
    "docs/release.md",
    "site/index.html",
]

MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def check_result(name: str, ok: bool, details: list[str] | None = None) -> dict[str, Any]:
    return {"name": name, "ok": ok, "details": details or []}


def read_text(root: Path, relative_path: str) -> str:
    return (root / relative_path).read_text(encoding="utf-8")


def check_required_files(root: Path) -> dict[str, Any]:
    missing = [path for path in REQUIRED_FILES if not (root / path).is_file()]
    return check_result("required_files", not missing, missing)


def check_pyproject(root: Path) -> dict[str, Any]:
    details: list[str] = []
    try:
        payload = tomllib.loads(read_text(root, "pyproject.toml"))
    except Exception as exc:
        return check_result("pyproject", False, [f"could not parse pyproject.toml: {exc}"])

    project = payload.get("project", {})
    if project.get("name") != "crowdtensord":
        details.append("project.name must be crowdtensord")
    version = str(project.get("version", ""))
    if "a" not in version:
        details.append("project.version must be an alpha version")
    if project.get("requires-python") != ">=3.11":
        details.append("project.requires-python must be >=3.11")

    dependencies = [str(dep).lower() for dep in project.get("dependencies", [])]
    if not any(dep.startswith("fastapi") for dep in dependencies):
        details.append("project.dependencies must include fastapi")
    if not any(dep.startswith("uvicorn") for dep in dependencies):
        details.append("project.dependencies must include uvicorn")

    scripts = project.get("scripts", {})
    if scripts.get("crowdtensord") != "coordinator:main":
        details.append("project.scripts.crowdtensord must point to coordinator:main")
    if scripts.get("crowdtensor-miner") != "miner_cli:main":
        details.append("project.scripts.crowdtensor-miner must point to miner_cli:main")
    if scripts.get("crowdtensor") != "crowdtensor.cli:main":
        details.append("project.scripts.crowdtensor must point to crowdtensor.cli:main")

    build_requires = [str(req).lower() for req in payload.get("build-system", {}).get("requires", [])]
    if not any(req.startswith("setuptools") for req in build_requires):
        details.append("build-system.requires must include setuptools")
    if not any(req == "wheel" or req.startswith("wheel") for req in build_requires):
        details.append("build-system.requires must include wheel")

    return check_result("pyproject", not details, details)


def local_markdown_targets(root: Path) -> list[Path]:
    return [root / "README.md", *sorted((root / "docs").glob("*.md"))]


def is_external_link(target: str) -> bool:
    lowered = target.lower()
    return lowered.startswith(("http://", "https://", "mailto:", "tel:"))


def normalize_markdown_target(raw_target: str) -> str:
    target = raw_target.strip().strip("<>")
    target = target.split("#", 1)[0]
    return unquote(target)


def check_markdown_links(root: Path) -> dict[str, Any]:
    details: list[str] = []
    for markdown_path in local_markdown_targets(root):
        if not markdown_path.exists():
            continue
        text = markdown_path.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(text):
            raw_target = match.group(1)
            if is_external_link(raw_target):
                continue
            target = normalize_markdown_target(raw_target)
            if not target:
                continue
            if target.startswith("/"):
                details.append(f"{markdown_path.relative_to(root)} uses absolute local link {raw_target}")
                continue
            resolved = (markdown_path.parent / target).resolve()
            try:
                resolved.relative_to(root.resolve())
            except ValueError:
                details.append(f"{markdown_path.relative_to(root)} links outside repo: {raw_target}")
                continue
            if not resolved.exists():
                details.append(f"{markdown_path.relative_to(root)} has broken link: {raw_target}")
    return check_result("markdown_links", not details, details)


def check_security_docs(root: Path) -> dict[str, Any]:
    try:
        text = read_text(root, "docs/security.md")
    except OSError as exc:
        return check_result("security_docs", False, [str(exc)])
    details: list[str] = []
    for fragment in [
        "sha256:",
        "scripts/hash_token.py",
        "hashed token",
        "--operator-token-registry",
        "roles",
        "crowdtensor operator-invite",
        "crowdtensor swarm-bootstrap",
        "crowdtensor swarm-bootstrap-check",
        "--expect-remote-miners",
        "--check-admission",
        "coordinator.private.env",
        "operator.private.env",
        "tunnel.private.env",
        "--tunnel-command",
        "bootstrap_handoff",
        "ready_to_copy_stage_packages",
        "start_control_plane.sh",
        "start_tunnel.sh",
        "start_discovery.sh",
        "start_coordinator.sh",
        "verify_bootstrap.sh",
        "handoff_doctor.sh",
        "crowdtensor swarm-handoff-doctor",
        "crowdtensor_swarm_handoff_doctor_v1",
        "handoff_doctor.json",
        "handoff_doctor.md",
        "check_join.sh",
        "support_bundle.sh",
        "miner_support_bundle.json",
        "crowdtensor_miner_local_environment_v1",
        "local_environment_ready",
        "stage0.miner-package.tar.gz",
        "stage0.run-miner.sh",
        "--doctor",
        "stage0.handoff.sha256",
        "stage_handoff_manifest.json",
        "stage_handoff_checksums_ready",
        "miner.join-code.txt",
        "--invite-code-file",
        "--peer-bootstrap",
        "crowdtensor_miner_join_discovery_v1",
        "join.sh",
        "scripts/create_operator_invite.py",
        "crowdtensor_operator_invite_v1",
        "crowdtensor settlement",
        "crowdtensor_settlement_cli_v1",
    ]:
        if fragment not in text:
            details.append(f"docs/security.md must mention {fragment}")
    return check_result("security_docs", not details, details)


def check_readiness_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in ["README.md", "docs/operations.md"]
        if (root / path).exists()
    )
    for fragment in ["/ready", "/version", "scripts/readiness_check.py"]:
        if fragment not in combined:
            details.append(f"README.md or docs/operations.md must mention {fragment}")
    return check_result("readiness_docs", not details, details)


def check_api_docs(root: Path) -> dict[str, Any]:
    try:
        text = read_text(root, "docs/api.md")
    except OSError as exc:
        return check_result("api_docs", False, [str(exc)])
    details: list[str] = []
    required_fragments = [
        "GET /health",
        "GET /version",
        "GET /ready",
        "GET /state",
        "GET /metrics",
        "GET /admin/events",
        "GET /admin/results",
        "GET /admin/accounting",
        "GET /admin/settlement",
        "miner_accounting_summary_v1",
        "miner_settlement_draft_v1",
        "crowdtensor settlement",
        "crowdtensor_settlement_cli_v1",
        "--operator-token-registry",
        "--miner-token-registry",
        "crowdtensor operator-invite",
        "crowdtensor operator-status",
        "crowdtensor_operator_status_cli_v1",
        "operator_status.json",
        "operator_status.md",
        "crowdtensor trust",
        "crowdtensor_trust_cli_v1",
        "trust_summary.json",
        "trust_summary.md",
        "crowdtensor swarm-bootstrap",
        "--tunnel-command",
        "tunnel.private.env",
        "bootstrap_handoff",
        "ready_to_copy_stage_packages",
        "start_control_plane.sh",
        "start_tunnel.sh",
        "start_discovery.sh",
        "verify_bootstrap.sh",
        "handoff_doctor.sh",
        "crowdtensor swarm-handoff-doctor",
        "crowdtensor_swarm_handoff_doctor_v1",
        "handoff_doctor.json",
        "handoff_doctor.md",
        "check_join.sh",
        "support_bundle.sh",
        "miner_support_bundle.json",
        "crowdtensor_miner_local_environment_v1",
        "local_environment_ready",
        "stage0.miner-package.tar.gz",
        "stage0.run-miner.sh",
        "--doctor",
        "stage0.handoff.sha256",
        "stage_handoff_manifest.json",
        "stage_handoff_checksums_ready",
        "miner.join-code.txt",
        "--invite-code-file",
        "--peer-bootstrap",
        "crowdtensor_miner_join_discovery_v1",
        "crowdtensor swarm-bootstrap-check",
        "--expect-remote-miners",
        "--check-admission",
        "/tasks/preflight",
        "operator_registry_summary",
        "POST /admin/trust-overrides",
        "POST /tasks/claim",
        "POST /tasks/{task_id}/heartbeat",
        "POST /tasks/{task_id}/result",
        "x-crowdtensor-miner-token",
        "x-crowdtensor-observer-token",
        "x-crowdtensor-admin-token",
        "scripts/api_contract_check.py",
    ]
    for fragment in required_fragments:
        if fragment not in text:
            details.append(f"docs/api.md must mention {fragment}")
    combined = "\n".join(
        read_text(root, path)
        for path in ["README.md", "docs/operations.md", "scripts/runtime_acceptance_pack.py"]
        if (root / path).exists()
    )
    for fragment in ["docs/api.md", "api_contract_check.py", "api_contract"]:
        if fragment not in combined:
            details.append(f"README.md, docs/operations.md, or runtime_acceptance_pack.py must mention {fragment}")
    return check_result("api_docs", not details, details)


def check_miner_resilience_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in ["README.md", "docs/operations.md", "scripts/runtime_acceptance_pack.py"]
        if (root / path).exists()
    )
    for fragment in [
        "scripts/miner_resilience_check.py",
        "miner_resilience",
        "--skip-preflight",
        "--max-request-attempts",
        "request_retries",
    ]:
        if fragment not in combined:
            details.append(f"README.md, docs/operations.md, or runtime_acceptance_pack.py must mention {fragment}")
    return check_result("miner_resilience_docs", not details, details)


def check_result_idempotency_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in ["README.md", "docs/api.md", "docs/operations.md", "scripts/runtime_acceptance_pack.py"]
        if (root / path).exists()
    )
    for fragment in [
        "idempotency_key",
        "scripts/result_idempotency_check.py",
        "result_idempotency",
        "--skip-result-idempotency",
    ]:
        if fragment not in combined:
            details.append(f"README.md, docs/api.md, docs/operations.md, or runtime_acceptance_pack.py must mention {fragment}")
    return check_result("result_idempotency_docs", not details, details)


def check_result_ledger_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in ["README.md", "docs/api.md", "docs/operations.md", "scripts/runtime_acceptance_pack.py"]
        if (root / path).exists()
    )
    for fragment in [
        "GET /admin/results",
        "scripts/result_ledger_check.py",
        "result_ledger",
        "--skip-result-ledger",
    ]:
        if fragment not in combined:
            details.append(f"README.md, docs/api.md, docs/operations.md, or runtime_acceptance_pack.py must mention {fragment}")
    return check_result("result_ledger_docs", not details, details)


def check_remote_miner_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/quickstart.md",
            "docs/remote-miner.md",
            "docs/operations.md",
            "docs/security.md",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "scripts/create_miner_invite.py",
        "scripts/create_operator_invite.py",
        "crowdtensor swarm-bootstrap",
        "crowdtensor_swarm_bootstrap_v1",
        "crowdtensor swarm-bootstrap-check",
        "crowdtensor_swarm_bootstrap_check_v1",
        "coordinator_url_remote_route_ready",
        "swarm_bootstrap_live_preflight_ready",
        "--expect-remote-miners",
        "--check-admission",
        "/tasks/preflight",
        "coordinator.private.env",
        "operator.private.env",
        "tunnel.private.env",
        "--tunnel-command",
        "bootstrap_handoff",
        "ready_to_copy_stage_packages",
        "start_control_plane.sh",
        "start_tunnel.sh",
        "start_discovery.sh",
        "start_coordinator.sh",
        "verify_bootstrap.sh",
        "handoff_doctor.sh",
        "crowdtensor swarm-handoff-doctor",
        "crowdtensor_swarm_handoff_doctor_v1",
        "handoff_doctor.json",
        "handoff_doctor.md",
        "check_join.sh",
        "stage_check_join_scripts_ready",
        "support_bundle.sh",
        "miner_support_bundle.json",
        "stage_support_bundle_scripts_ready",
        "crowdtensor_miner_local_environment_v1",
        "local_environment_ready",
        "stage0.miner-package.tar.gz",
        "stage_package_archives_ready",
        "stage0.run-miner.sh",
        "stage_archive_runner_scripts_ready",
        "--doctor",
        "stage0.handoff.sha256",
        "stage_handoff_manifest.json",
        "stage_handoff_checksums_ready",
        "miner.join-code.txt",
        "--invite-code-file",
        "--peer-bootstrap",
        "crowdtensor_miner_join_discovery_v1",
        "join.sh",
        "scripts/remote_miner_join_check.py",
        "scripts/remote_miner_readiness_check.py",
        "model_bundle_lm",
        "--miner-token-registry",
        "sha256:",
        "Remote Miner Onboarding",
    ]:
        if fragment not in combined:
            details.append(f"remote Miner docs must mention {fragment}")
    return check_result("remote_miner_docs", not details, details)


def check_security_preflight_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/quickstart.md",
            "docs/remote-miner.md",
            "docs/operations.md",
            "docs/security.md",
            ".github/workflows/ci.yml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "scripts/security_preflight.py",
        "--strict",
        "--miner-token-registry",
        "security preflight",
    ]:
        if fragment not in combined:
            details.append(f"security preflight docs or CI must mention {fragment}")
    return check_result("security_preflight_docs", not details, details)


def check_browser_acceptance_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/quickstart.md",
            "docs/operations.md",
            ".github/workflows/ci.yml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "scripts/browser_acceptance_pack.py",
        "--allow-skip",
        "webrtc_smoke.py",
        "browser_miner_smoke.py",
        "runtime_contract_check.py",
    ]:
        if fragment not in combined:
            details.append(f"browser acceptance docs or CI must mention {fragment}")
    return check_result("browser_acceptance_docs", not details, details)


def check_release_evidence_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            ".github/workflows/ci.yml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "scripts/release_evidence_pack.py",
        "--runtime-report",
        "release-evidence.json",
        "Release Evidence",
        "diagnosis_summary",
        "diagnosis_by_check",
        "observability_summaries",
        "remote_compute_observability_v1",
        "remote_demo_observability_v1",
    ]:
        if fragment not in combined:
            details.append(f"release evidence docs or CI must mention {fragment}")
    return check_result("release_evidence_docs", not details, details)


def check_release_readiness_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/release.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
            "pyproject.toml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "crowdtensor release-ready",
        "release_readiness_v1",
        "scripts/release_readiness_pack.py",
        "scripts/release_readiness_check.py",
        "--allow-dirty",
        "git_dirty",
        "demo_manifest_v1",
        "release gate",
        "not production",
        "crowdtensor/cli.py",
    ]:
        if fragment not in combined:
            details.append(f"release readiness docs/CI must mention {fragment}")
    return check_result("release_readiness_docs", not details, details)


def check_onboarding_gate_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/release.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "scripts/onboarding_gate.py",
        "onboarding_gate_v1",
        "python3 -m venv",
        "python -m pip install -e .[dev,hf]",
        "crowdtensor --help",
        "crowdtensord --help",
        "crowdtensor-miner --help",
        "crowdtensor infer --prompt-stdin --shareable-terminal",
        "user_infer_smoke",
        "user_infer_smoke_validation_v1",
        "answer=shareable-terminal-redacted",
        "gpu=local-cpu-only",
        "fresh_kaggle_gpu=False",
        "crowdtensor local-proof",
        "crowdtensor home-infer",
        "crowdtensor llm-infer --mock",
        "crowdtensor release-ready",
        "--quick",
        "not production",
    ]:
        if fragment not in combined:
            details.append(f"onboarding gate docs/CI must mention {fragment}")
    return check_result("onboarding_gate_docs", not details, details)


def check_doctor_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/quickstart.md",
            "docs/operations.md",
            "CONTRIBUTING.md",
            ".github/workflows/ci.yml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "scripts/doctor.py",
        "--remote-demo",
        "--browser",
        "First-run Doctor",
    ]:
        if fragment not in combined:
            details.append(f"doctor docs or CI must mention {fragment}")
    return check_result("doctor_docs", not details, details)


def check_support_bundle_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/security.md",
            ".github/workflows/ci.yml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "scripts/support_bundle.py",
        "--json-out",
        "Support Bundle",
        "support_bundle",
        "diagnosis_summary",
        "diagnosis_by_check",
        "observability_summaries",
        "remote_compute_observability_v1",
        "remote_demo_observability_v1",
    ]:
        if fragment not in combined:
            details.append(f"support bundle docs or CI must mention {fragment}")
    return check_result("support_bundle_docs", not details, details)


def check_home_compute_evidence_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/use-cases.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            "scripts/runtime_acceptance_pack.py",
            ".github/workflows/ci.yml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "home_compute_evidence_pack.py",
        "home_compute_evidence_check.py",
        "home_compute_evidence_v1",
        "--skip-home-compute-evidence",
        "route_decision",
        "request_trace",
        "matched_capabilities",
        "diagnosis_codes",
        "home_compute_ready",
        "runtime_matrix_blocked",
        "safe, shareable",
    ]:
        if fragment not in combined:
            details.append(f"home-compute evidence docs/CI must mention {fragment}")
    return check_result("home_compute_evidence_docs", not details, details)


def check_remote_compute_evidence_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/remote-miner.md",
            "docs/use-cases.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            "scripts/runtime_acceptance_pack.py",
            ".github/workflows/ci.yml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "remote_compute_evidence_pack.py",
        "remote_compute_evidence_check.py",
        "remote_compute_evidence_v1",
        "remote_compute_observability_v1",
        "--include-remote-evidence",
        "remote_python_model_bundle_infer",
        "model_bundle_infer",
        "safe, shareable",
    ]:
        if fragment not in combined:
            details.append(f"remote-compute evidence docs/CI must mention {fragment}")
    return check_result("remote_compute_evidence_docs", not details, details)


def check_remote_demo_runbook_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/remote-miner.md",
            "docs/use-cases.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "remote_demo_runbook_pack.py",
        "remote_demo_runbook_check.py",
        "remote_demo_runbook_v1",
        "operator.private.env",
        "miner.private.env",
        "remote_compute_evidence_pack.py --mode collect",
        "--scenario-id route-baseline",
        "model_bundle_inference_scenario_v1",
        "model_bundle_infer",
        "safe two-machine",
    ]:
        if fragment not in combined:
            details.append(f"remote demo runbook docs/CI must mention {fragment}")
    return check_result("remote_demo_runbook_docs", not details, details)


def check_remote_demo_acceptance_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/remote-miner.md",
            "docs/use-cases.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "remote_demo_acceptance_pack.py",
        "remote_demo_acceptance_check.py",
        "remote_demo_acceptance_v1",
        "remote_demo_observability_v1",
        "--create-session",
        "POST /admin/inference-sessions",
        "--scenario-id route-baseline",
        "model_bundle_inference_scenario_v1",
        "scenario match",
        "session_create_failed",
        "remote_compute_evidence_v1",
        "support_bundle",
        "diagnosis_codes",
        "coordinator_unreachable",
        "observer_auth_failed",
        "artifact_collection_failed",
        "model_bundle_infer",
        "safe two-machine",
    ]:
        if fragment not in combined:
            details.append(f"remote demo acceptance docs/CI must mention {fragment}")
    return check_result("remote_demo_acceptance_docs", not details, details)


def check_demo_manifest_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/release.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "demo_manifest_pack.py",
        "demo_manifest_check.py",
        "demo_manifest_v1",
        "Demo Manifest",
        "runtime_matrix.json",
        "remote_compute_evidence_v1",
        "external_llm_evidence_v1",
        "support_bundle",
        "remote_compute_observability_v1",
        "latest output artifact",
        "local-loopback",
    ]:
        if fragment not in combined:
            details.append(f"demo manifest docs/CI must mention {fragment}")
    return check_result("demo_manifest_docs", not details, details)


def check_local_proof_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
            "pyproject.toml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "crowdtensor local-proof",
        "local_proof_summary_v1",
        "crowdtensor/cli.py",
        "Demo Manifest",
        "CPU-only",
        "read-only",
        "not production Swarm Inference",
    ]:
        if fragment not in combined:
            details.append(f"local proof CLI docs/CI must mention {fragment}")
    return check_result("local_proof_docs", not details, details)


def check_cleanup_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "CHANGELOG.md",
            "pyproject.toml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "crowdtensor clean-artifacts",
        "cleanup_report_v1",
        "--apply",
        "--include-reports",
        "dry-run",
        "__pycache__",
        "does not delete state",
    ]:
        if fragment not in combined:
            details.append(f"cleanup docs must mention {fragment}")
    return check_result("cleanup_docs", not details, details)


def check_remote_cli_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/remote-miner.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            "pyproject.toml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "crowdtensor remote-runbook",
        "crowdtensor remote-acceptance",
        "remote_runbook_cli_v1",
        "remote_acceptance_cli_v1",
        "--create-session",
        "token redaction",
        "not production",
        "not P2P",
        "crowdtensor/cli.py",
    ]:
        if fragment not in combined:
            details.append(f"remote CLI docs must mention {fragment}")
    return check_result("remote_cli_docs", not details, details)


def check_remote_home_compute_demo_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/remote-miner.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
            "pyproject.toml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "crowdtensor remote-demo",
        "remote-demo doctor",
        "remote-demo collect",
        "remote-demo clean",
        "remote_home_compute_demo_v1",
        "remote_home_compute_doctor_v1",
        "remote_home_compute_collect_v1",
        "remote_home_compute_cleanup_v1",
        "remote_home_compute_demo_pack.py",
        "remote_home_compute_demo_check.py",
        "model_bundle_infer",
        "remote_python_model_bundle_infer",
        "POST /admin/inference-sessions",
        "remote_compute_evidence_v1",
        "remote_demo_observability_v1",
        "external_llm_infer",
        "remote_python_external_llm_infer",
        "remote_external_llm_evidence_v1",
        "remote_external_llm_observability_v1",
        "--workload external-llm",
        "--workload real-llm-sharded",
        "remote_python_real_llm_sharded_infer",
        "remote_real_llm_sharded_runbook_v1",
        "remote_real_llm_sharded_acceptance_v1",
        "remote_real_llm_sharded_observability_v1",
        "remote_real_llm_sharded_beta_v1",
        "remote_two_machine_real_llm_sharded_ready",
        "operator.private.env",
        "miner.private.env",
        "not production",
        "not P2P",
    ]:
        if fragment not in combined:
            details.append(f"remote home-compute demo docs/CI must mention {fragment}")
    return check_result("remote_home_compute_demo_docs", not details, details)


def check_remote_two_machine_beta_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/remote-miner.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "Real two-machine CPU inference Beta",
        "remote_two_machine_beta_check.py",
        "remote_two_machine_beta_check_v1",
        "remote_two_machine_inference_ready",
        "remote_two_machine_external_llm_ready",
        "remote_two_machine_beta_ready",
        "15-minute two-machine CPU inference Beta",
        "Coordinator host",
        "Miner host",
        "operator-provided TLS, VPN, tunnel, or trusted network",
        "not model sharding",
        "not P2P",
    ]:
        if fragment not in combined:
            details.append(f"remote two-machine Beta docs/CI must mention {fragment}")
    return check_result("remote_two_machine_beta_docs", not details, details)


def check_kaggle_remote_miner_beta_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/remote-miner.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "Kaggle Remote Miner Beta",
        "kaggle_remote_miner_beta_check.py",
        "kaggle_remote_miner_beta_check_v1",
        "--target kaggle",
        "kaggle_remote_miner.py",
        "kaggle_remote_miner_prepare_ready",
        "kaggle_remote_miner_beta_ready",
        "miner.private.env",
        "operator.private.env",
        "outbound",
        "GPU/TPU workload",
        "not production",
        "not P2P",
    ]:
        if fragment not in combined:
            details.append(f"Kaggle Remote Miner Beta docs/CI must mention {fragment}")
    return check_result("kaggle_remote_miner_beta_docs", not details, details)


def check_kaggle_real_runtime_acceptance_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/remote-miner.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "Kaggle Real Runtime Acceptance",
        "kaggle_real_runtime_acceptance_v1",
        "kaggle_real_runtime_acceptance_pack.py",
        "kaggle_real_runtime_acceptance_check.py",
        "remote-demo kaggle-real",
        "24.199.118.54",
        "temporary HTTP",
        "kaggle_real_runtime_ready",
        "coordinator_public_ready",
        "kaggle_artifacts_ready",
        "kaggle_miner_seen",
        "kaggle_result_accepted",
        "micro-llm-sharded",
        "kaggle-upload-stage0",
        "kaggle-upload-stage1",
        "kaggle_micro_llm_sharded_ready",
        "kaggle_micro_llm_stage0_seen",
        "kaggle_micro_llm_stage1_seen",
        "kaggle_micro_llm_stage_assignment_valid",
        "stage_assignment_valid",
        "toy two-stage pipeline",
        "not large-model sharding",
        "token_rotation_required",
        "operator.private.env",
        "miner.private.env",
        "not production",
        "not P2P",
    ]:
        if fragment not in combined:
            details.append(f"Kaggle real runtime acceptance docs/CI must mention {fragment}")
    return check_result("kaggle_real_runtime_acceptance_docs", not details, details)


def check_home_inference_cli_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            "pyproject.toml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "crowdtensor home-infer",
        "home_inference_cli_v1",
        "home_compute_evidence_v1",
        "model_bundle_inference_scenario_v1",
        "route-baseline",
        "request_trace",
        "model_bundle_infer",
        "read-only",
        "arbitrary prompt",
        "not production Swarm Inference",
        "crowdtensor/cli.py",
    ]:
        if fragment not in combined:
            details.append(f"home inference CLI docs must mention {fragment}")
    return check_result("home_inference_cli_docs", not details, details)


def check_cpu_inference_beta_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/remote-miner.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            "scripts/onboarding_gate.py",
            ".github/workflows/ci.yml",
            "pyproject.toml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "crowdtensor cpu-infer",
        "cpu_inference_beta_v1",
        "cpu_inference_beta_pack.py",
        "cpu_inference_beta_check.py",
        "--mode local",
        "--mode remote-loopback",
        "--mode remote-existing",
        "CPU-only",
        "read-only",
        "home-infer",
        "llm-infer",
        "remote-demo",
        "not production Swarm Inference",
        "not P2P",
        "crowdtensor/cli.py",
    ]:
        if fragment not in combined:
            details.append(f"CPU inference Beta docs/CI must mention {fragment}")
    return check_result("cpu_inference_beta_docs", not details, details)


def check_cpu_inference_beta_rc_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/remote-miner.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
            "pyproject.toml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "CPU Inference Beta RC",
        "cpu_inference_beta_rc_v1",
        "cpu_inference_beta_rc_pack.py",
        "cpu_inference_beta_rc_check.py",
        "--mode beta-rc",
        "cpu_inference_beta_rc_ready",
        "local_cpu_inference_ready",
        "remote_loopback_ready",
        "two_machine_rehearsal_ready",
        "kaggle_remote_miner_artifacts_ready",
        "miner_join_pack_v1",
        "miner_join_pack_ready",
        "cpu_miner_beta_ready",
        "--kaggle-real-runtime-report",
        "real_runtime_evidence_ready",
        "Kaggle Remote Miner Beta",
        "demo_manifest_v1",
        "CPU-only",
        "read-only",
        "not production Swarm Inference",
        "not P2P",
    ]:
        if fragment not in combined:
            details.append(f"CPU Inference Beta RC docs/CI must mention {fragment}")
    return check_result("cpu_inference_beta_rc_docs", not details, details)


def check_sharded_inference_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
            "pyproject.toml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "Pipeline-Sharded Inference Alpha",
        "sharded_model_bundle_infer",
        "sharded_model_bundle_infer_v1",
        "sharded_inference_session_v1",
        "sharded_inference_evidence_v1",
        "sharded_inference_check.py",
        "crowdtensor shard-infer",
        "stage_0_accepted",
        "stage_1_accepted",
        "activation_transport_ready",
        "baseline_match",
        "stage_requeue_ready",
        "CPU-only",
        "read-only",
        "not production Swarm Inference",
        "not P2P",
    ]:
        if fragment not in combined:
            details.append(f"sharded inference docs/CI must mention {fragment}")
    return check_result("sharded_inference_docs", not details, details)


def check_remote_sharded_inference_beta_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
            "pyproject.toml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "CPU Pipeline-Sharded Inference Beta",
        "remote_sharded_inference_beta_v1",
        "remote_sharded_inference_beta_check.py",
        "remote_sharded_inference_beta_pack.py",
        "crowdtensor shard-infer-beta",
        "--mode remote-loopback",
        "remote_sharded_inference_ready",
        "remote_sharded_loopback_ready",
        "local_sharded_inference_ready",
        "sharded-model-bundle",
        "remote_sharded_inference_acceptance_v1",
        "remote_sharded_inference_observability_v1",
        "remote_python_sharded_model_bundle_infer",
        "remote_two_machine_sharded_ready",
        "stage_requeue_ready",
        "activation hashes",
        "baseline_match",
        "CPU-only",
        "read-only",
        "not production Swarm Inference",
        "not P2P",
        "not real LLM sharding",
    ]:
        if fragment not in combined:
            details.append(f"remote sharded inference Beta docs/CI must mention {fragment}")
    return check_result("remote_sharded_inference_beta_docs", not details, details)


def check_micro_llm_sharded_inference_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
            "pyproject.toml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "Micro-LLM Pipeline-Sharded Inference Alpha",
        "micro_llm_sharded_infer",
        "micro_llm_sharded_infer_v1",
        "micro_llm_sharded_session_v1",
        "micro_llm_sharded_evidence_v1",
        "micro_llm_sharded_inference_check.py",
        "stage_aware_micro_llm_sharded_check.py",
        "crowdtensor micro-llm-shard-infer",
        "micro_llm_sharded_stage0",
        "micro_llm_sharded_stage1",
        "--require-distinct-stage-miners",
        "distinct_stage_miners",
        "stage_assignment_valid",
        "decode_steps",
        "decoded_tokens_match",
        "activation_transport_ready",
        "baseline_match",
        "stage_requeue_ready",
        "CPU-only",
        "read-only",
        "not production Swarm Inference",
        "not P2P",
        "not GGUF/llama.cpp",
    ]:
        if fragment not in combined:
            details.append(f"micro-LLM sharded inference docs/CI must mention {fragment}")
    return check_result("micro_llm_sharded_inference_docs", not details, details)


def check_remote_micro_llm_sharded_beta_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
            "pyproject.toml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "Remote Micro-LLM Pipeline-Sharded Inference Beta",
        "remote_micro_llm_sharded_beta_v1",
        "remote_micro_llm_sharded_beta_check.py",
        "remote_micro_llm_sharded_beta_pack.py",
        "crowdtensor micro-llm-shard-infer-beta",
        "--mode remote-loopback",
        "--stage-mode split",
        "--require-distinct-stage-miners",
        "remote_micro_llm_sharded_ready",
        "remote_micro_llm_sharded_loopback_ready",
        "local_micro_llm_sharded_inference_ready",
        "micro-llm-sharded",
        "remote_micro_llm_sharded_acceptance_v1",
        "remote_micro_llm_sharded_observability_v1",
        "remote_python_micro_llm_sharded_infer",
        "remote_two_machine_micro_llm_sharded_ready",
        "decoded_tokens_match",
        "distinct_stage_miners",
        "stage_assignment_valid",
        "activation hashes",
        "baseline_match",
        "CPU-only",
        "read-only",
        "not production Swarm Inference",
        "not P2P",
        "not GGUF/llama.cpp",
    ]:
        if fragment not in combined:
            details.append(f"remote micro-LLM sharded Beta docs/CI must mention {fragment}")
    return check_result("remote_micro_llm_sharded_beta_docs", not details, details)


def check_micro_llm_live_rc_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/remote-miner.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
            "pyproject.toml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "Micro-LLM Live Two-Node RC",
        "micro_llm_live_rc_v1",
        "micro_llm_live_rc_check.py",
        "micro_llm_live_rc_pack.py",
        "crowdtensor micro-llm-live-rc",
        "local-generated",
        "external-existing",
        "local_generated_stage_upload_standins_ready",
        "external_runtime_verified",
        "micro_llm_live_rc_ready",
        "kaggle_micro_llm_sharded_ready",
        "kaggle-upload-stage0",
        "kaggle-upload-stage1",
        "stage_assignment_valid",
        "toy two-stage micro-LLM",
        "CPU-only",
        "read-only",
        "not production Swarm Inference",
        "not P2P",
        "not GGUF/llama.cpp",
    ]:
        if fragment not in combined:
            details.append(f"micro-LLM live RC docs/CI must mention {fragment}")
    return check_result("micro_llm_live_rc_docs", not details, details)


def check_real_llm_sharded_beta_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/protocol.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
            "pyproject.toml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "Real Small-LLM Sharded Inference Beta",
        "real_llm_sharded_infer",
        "real_llm_sharded_infer_v1",
        "real_llm_artifact_v1",
        "real_llm_sharded_evidence_v1",
        "remote_real_llm_sharded_beta_v1",
        "real_llm_sharded_inference_evidence_pack.py",
        "remote_real_llm_sharded_beta_pack.py",
        "remote_real_llm_sharded_beta_check.py",
        "crowdtensor real-llm-shard-infer",
        "crowdtensor real-llm-shard-infer-beta",
        "crowdtensor remote-demo --workload real-llm-sharded",
        "--enable-hf-tiny-gpt-runtime",
        "--hf-cache-dir",
        "--real-llm-stage-role",
        "real_llm_sharded_stage0",
        "real_llm_sharded_stage1",
        "real_llm_sharded_both",
        "real_llm_artifact_ready",
        "activation_transport_ready",
        "baseline_match",
        "decoded_tokens_match",
        "stage_assignment_valid",
        "remote_real_llm_sharded_ready",
        "remote_two_machine_real_llm_sharded_ready",
        "remote_real_llm_sharded_acceptance_v1",
        "remote_real_llm_sharded_observability_v1",
        "remote_python_real_llm_sharded_infer",
        "hf_dependencies_missing",
        "hf_transformers_cpu",
        "optional [hf]",
        "CPU-only",
        "read-only",
        "not production Swarm Inference",
        "not P2P",
        "not GGUF/llama.cpp",
        "not large-model",
    ]:
        if fragment not in combined:
            details.append(f"real LLM sharded Beta docs/CI must mention {fragment}")
    return check_result("real_llm_sharded_beta_docs", not details, details)


def check_real_llm_live_rc_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/remote-miner.md",
            "docs/release.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
            "pyproject.toml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "Real Small-LLM Sharded Inference Live RC",
        "real_llm_live_rc_v1",
        "real_llm_live_rc_check.py",
        "real_llm_live_rc_pack.py",
        "kaggle_real_llm_live_package.py",
        "kaggle_real_llm_live_package_v1",
        "crowdtensor real-llm-live-rc",
        "local-generated",
        "kaggle-generated",
        "external-existing",
        "kaggle_real_llm_live_package_ready",
        "kaggle-upload-real-llm-stage0",
        "kaggle-upload-real-llm-stage1",
        "local_generated_real_llm_stage_upload_standins_ready",
        "external_runtime_verified",
        "kaggle_real_llm_stage0_seen",
        "kaggle_real_llm_stage1_seen",
        "kaggle_real_llm_sharded_ready",
        "real_llm_artifact_ready",
        "--enable-hf-tiny-gpt-runtime",
        "--real-llm-stage-role",
        "CPU-only",
        "read-only",
        "not production Swarm Inference",
        "not P2P",
        "not large-model",
    ]:
        if fragment not in combined:
            details.append(f"real LLM live RC docs/CI must mention {fragment}")
    return check_result("real_llm_live_rc_docs", not details, details)


def check_real_llm_internet_alpha_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/remote-miner.md",
            "docs/release.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
            "pyproject.toml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "Real Internet Swarm Inference Alpha",
        "real_llm_internet_alpha_v1",
        "real_llm_internet_alpha_check.py",
        "real_llm_internet_alpha_pack.py",
        "crowdtensor real-llm-internet-alpha",
        "local-generated",
        "package",
        "external-existing",
        "real_llm_internet_alpha_ready",
        "real_llm_stage_requeue_ready",
        "stage_requeue_ready",
        "external_runtime_verified",
        "token_rotation_required",
        "CPU-only",
        "read-only",
        "not production Swarm Inference",
        "not P2P",
        "not large-model",
    ]:
        if fragment not in combined:
            details.append(f"real LLM Internet Alpha docs/CI must mention {fragment}")
    return check_result("real_llm_internet_alpha_docs", not details, details)


def check_real_llm_internet_beta_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/remote-miner.md",
            "docs/release.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
            "pyproject.toml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "Real Internet Swarm Inference Beta",
        "real_llm_internet_beta_v1",
        "real_llm_internet_beta_check.py",
        "real_llm_internet_beta_pack.py",
        "crowdtensor real-llm-internet-beta",
        "kaggle-auto",
        "kaggle_kernels_deleted",
        "real_llm_internet_beta_ready",
        "real_llm_internet_alpha_ready",
        "external_runtime_verified",
        "external_stage_requeue_ready",
        "live_stage0_requeue_ready",
        "live_stage1_requeue_ready",
        "live_requeue_summary",
        "--failure-mode",
        "token_rotation_required",
        "CPU-only",
        "read-only",
        "not production Swarm Inference",
        "not P2P",
        "not large-model",
    ]:
        if fragment not in combined:
            details.append(f"real LLM Internet Beta docs/CI must mention {fragment}")
    return check_result("real_llm_internet_beta_docs", not details, details)


def check_swarm_inference_beta_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/remote-miner.md",
            "docs/release.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
            "pyproject.toml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "Swarm Inference Beta",
        "swarm_inference_beta_v1",
        "swarm_inference_beta_check.py",
        "swarm_inference_beta_pack.py",
        "crowdtensor swarm-infer-beta",
        "swarm-infer-beta live",
        "kaggle-auto",
        "swarm-infer-beta prepare",
        "swarm-infer-beta verify",
        "swarm-infer-beta collect",
        "swarm_inference_beta_ready",
        "swarm_inference_beta_live_ready",
        "two_machine_swarm_inference_ready",
        "real_llm_split_route_ready",
        "real_llm_internet_beta_ready",
        "kaggle_kernels_deleted",
        "external_stage_requeue_ready",
        "live_requeue_summary",
        "swarm_inference_beta_live_private_artifacts_cleaned",
        "--failure-mode",
        "--keep-live-private-artifacts",
        "token_rotation_required",
        "external_beta_evidence_imported",
        "stage0",
        "stage1",
        "operator.private.env",
        "miner.private.env",
        "miner_registry.json",
        "CPU-only",
        "read-only",
        "not production Swarm Inference",
        "not P2P",
        "not large-model",
    ]:
        if fragment not in combined:
            details.append(f"Swarm Inference Beta docs/CI must mention {fragment}")
    return check_result("swarm_inference_beta_docs", not details, details)


def check_public_swarm_inference_alpha_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/remote-miner.md",
            "docs/release.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
            "pyproject.toml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "Public Swarm Inference Alpha",
        "public_swarm_inference_alpha_v1",
        "public_swarm_inference_alpha_check.py",
        "public_swarm_inference_alpha_pack.py",
        "crowdtensor swarm-session",
        "live-kaggle",
        "local-generated",
        "public_swarm_inference_alpha_ready",
        "public_swarm_session_ready",
        "local_stage_requeue_ready",
        "public_swarm_live_requeue_ready",
        "external_stage_requeue_ready",
        "public_swarm_live_kaggle_ready",
        "stage_requeue_ready",
        "external_runtime_verified",
        "kaggle_kernels_deleted",
        "token_rotation_required",
        "CPU-only",
        "read-only",
        "not production Swarm Inference",
        "not P2P",
        "not large-model",
    ]:
        if fragment not in combined:
            details.append(f"Public Swarm Inference Alpha docs/CI must mention {fragment}")
    return check_result("public_swarm_inference_alpha_docs", not details, details)


def check_public_swarm_inference_alpha_rc_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/remote-miner.md",
            "docs/release.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
            "pyproject.toml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "Public Swarm Inference Alpha RC",
        "public_swarm_inference_alpha_rc_v1",
        "public_swarm_inference_alpha_rc_check.py",
        "public_swarm_inference_alpha_rc_pack.py",
        "crowdtensor public-swarm-alpha-rc",
        "evidence-import",
        "local-smoke",
        "public_swarm_inference_alpha_rc_ready",
        "public_swarm_alpha_rc_evidence_imported",
        "stage0_live_requeue_evidence_ready",
        "stage1_live_requeue_evidence_ready",
        "public_swarm_live_requeue_evidence_ready",
        "public_swarm_alpha_private_artifacts_absent",
        "CPU-only",
        "read-only",
        "not production Swarm Inference",
        "not P2P",
        "not large-model",
    ]:
        if fragment not in combined:
            details.append(f"Public Swarm Inference Alpha RC docs/CI must mention {fragment}")
    return check_result("public_swarm_inference_alpha_rc_docs", not details, details)


def check_public_swarm_inference_beta_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/remote-miner.md",
            "docs/release.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
            "pyproject.toml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "Public Swarm Inference Beta",
        "public_swarm_inference_beta_v1",
        "public_swarm_inference_beta_check.py",
        "public_swarm_inference_beta_pack.py",
        "crowdtensor public-swarm-beta",
        "public-swarm-beta product-beta",
        "public-swarm-beta local-loopback",
        "public-swarm-beta evidence-import",
        "public_swarm_inference_beta_ready",
        "public_swarm_product_beta_ready",
        "public_swarm_product_rc_ready",
        "coordinator_product_surface_ready",
        "session_protocol_ready",
        "p2p_lite_discovery_ready",
        "gpu_generation_evidence_import_ready",
        "cpu_fallback_ready",
        "public_swarm_beta_evidence_import_ready",
        "two_stage_split_inference_ready",
        "local_loopback_ready",
        "external_live_evidence_imported",
        "stage0_live_requeue_evidence_ready",
        "stage1_live_requeue_evidence_ready",
        "decoded_tokens_match",
        "distinct_stage_miners",
        "stage_assignment_valid",
        "prepare",
        "coordinator",
        "miner",
        "verify",
        "collect",
        "clean",
        "CPU-only",
        "read-only",
        "not production Swarm Inference",
        "not libp2p",
        "not DHT",
        "not NAT traversal",
        "not large-model",
    ]:
        if fragment not in combined:
            details.append(f"Public Swarm Inference Beta docs/CI must mention {fragment}")
    return check_result("public_swarm_inference_beta_docs", not details, details)


def check_public_swarm_inference_beta_rc_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/remote-miner.md",
            "docs/release.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
            "pyproject.toml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "Public Swarm Inference Beta RC",
        "public_swarm_inference_beta_rc_v1",
        "public_swarm_inference_beta_rc_check.py",
        "public_swarm_inference_beta_rc_pack.py",
        "crowdtensor public-swarm-beta-rc",
        "public-swarm-beta-rc local-loopback",
        "public-swarm-beta-rc package",
        "public-swarm-beta-rc external-existing",
        "public_swarm_inference_beta_rc_ready",
        "serve_join_generate_loop_ready",
        "remote_generate_session_ready",
        "public_swarm_generate_ready",
        "public_swarm_product_beta_ready",
        "p2p_lite_route_ready",
        "p2p_lite_discovery_ready",
        "cpu_fallback_ready",
        "private_artifacts_local_only",
        "external_runtime_verified",
        "hf_dependencies_missing",
        "CPU-only",
        "read-only",
        "not production Swarm Inference",
        "not libp2p",
        "not DHT",
        "not NAT traversal",
        "not large-model",
    ]:
        if fragment not in combined:
            details.append(f"Public Swarm Inference Beta RC docs/CI must mention {fragment}")
    return check_result("public_swarm_inference_beta_rc_docs", not details, details)


def check_public_swarm_product_beta_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/remote-miner.md",
            "docs/release.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
            "pyproject.toml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "Public Swarm Product Beta",
        "public_swarm_product_beta_v1",
        "public_swarm_product_beta_check.py",
        "public_swarm_product_beta_pack.py",
        "crowdtensor public-swarm-product-beta",
        "public-swarm-product-beta local-loopback",
        "public-swarm-product-beta package",
        "public-swarm-product-beta external-existing",
        "public_swarm_product_beta_ready",
        "public_swarm_product_beta_user_path_ready",
        "serve_ready",
        "stage0_join_ready",
        "stage1_join_ready",
        "generate_ready",
        "support_bundle_ready",
        "private_artifacts_cleaned",
        "decoded_tokens_match",
        "distinct_stage_miners",
        "stage_assignment_valid",
        "hf_dependencies_missing",
        "CPU-only",
        "read-only",
        "not production Swarm Inference",
        "not libp2p",
        "not DHT",
        "not NAT traversal",
        "not large-model",
    ]:
        if fragment not in combined:
            details.append(f"Public Swarm Product Beta docs/CI must mention {fragment}")
    return check_result("public_swarm_product_beta_docs", not details, details)


def check_public_swarm_developer_preview_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/remote-miner.md",
            "docs/release.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
            "pyproject.toml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "Public Swarm Developer Preview",
        "public_swarm_developer_preview_v1",
        "public_swarm_developer_preview_check.py",
        "public_swarm_developer_preview_pack.py",
        "crowdtensor preview",
        "preview local",
        "preview package",
        "preview external-existing",
        "preview evidence-import",
        "developer_preview_ready",
        "public_swarm_developer_preview_ready",
        "local_two_stage_generation_ready",
        "serve_join_generate_ready",
        "product_beta_ready",
        "support_bundle_ready",
        "cpu_fallback_ready",
        "gpu_generation_evidence_import_ready",
        "hf_dependencies_missing",
        "CPU-only",
        "read-only",
        "not production Swarm Inference",
        "not libp2p",
        "not DHT",
        "not NAT traversal",
        "not large-model",
    ]:
        if fragment not in combined:
            details.append(f"Public Swarm Developer Preview docs/CI must mention {fragment}")
    return check_result("public_swarm_developer_preview_docs", not details, details)


def check_public_swarm_live_preview_rc_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/remote-miner.md",
            "docs/release.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
            "pyproject.toml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "Public Swarm Live Preview RC",
        "public_swarm_live_preview_rc_v1",
        "public_swarm_live_preview_rc_check.py",
        "public_swarm_live_preview_rc_pack.py",
        "crowdtensor live-preview",
        "live-preview local-smoke",
        "live-preview package",
        "live-preview live-kaggle",
        "live-preview evidence-import",
        "public_swarm_live_preview_rc_ready",
        "public_swarm_live_preview_local_smoke_ready",
        "public_swarm_live_preview_package_ready",
        "public_swarm_live_preview_live_kaggle_ready",
        "public_swarm_live_preview_evidence_import_ready",
        "external_stage_requeue_ready",
        "kaggle_kernels_deleted",
        "private_artifacts_cleaned",
        "token_rotation_required",
        "gpu_generation_evidence_import_ready",
        "CPU-only",
        "read-only",
        "not production Swarm Inference",
        "not libp2p",
        "not DHT",
        "not NAT traversal",
        "not large-model",
    ]:
        if fragment not in combined:
            details.append(f"Public Swarm Live Preview RC docs/CI must mention {fragment}")
    return check_result("public_swarm_live_preview_rc_docs", not details, details)


def check_public_swarm_operator_preview_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/remote-miner.md",
            "docs/release.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
            "pyproject.toml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "Public Swarm v0.1 Operator Preview",
        "public_swarm_operator_preview_v1",
        "public_swarm_operator_preview_check.py",
        "public_swarm_operator_preview_pack.py",
        "crowdtensor operator-preview",
        "operator-preview local-smoke",
        "operator-preview package",
        "operator-preview live-kaggle",
        "operator-preview evidence-import",
        "public_swarm_operator_preview_ready",
        "operator_preview_user_path_ready",
        "operator_preview_local_smoke_ready",
        "operator_preview_package_ready",
        "operator_preview_live_kaggle_ready",
        "operator_preview_evidence_import_ready",
        "serve_join_generate_ready",
        "miner_join_pack_ready",
        "cpu_fallback_ready",
        "developer_preview_degraded",
        "operator_preview_cpu_fallback_user_path_ready",
        "operator_preview_retained_evidence_ready",
        "live_preview_ready",
        "support_bundle_ready",
        "release_readiness_ready",
        "gpu_generation_evidence_import_ready",
        "external_runtime_blocked",
        "CPU-only",
        "read-only",
        "not production Swarm Inference",
        "not libp2p",
        "not DHT",
        "not NAT traversal",
        "not large-model",
    ]:
        if fragment not in combined:
            details.append(f"Public Swarm v0.1 Operator Preview docs/CI must mention {fragment}")
    return check_result("public_swarm_operator_preview_docs", not details, details)


def check_public_swarm_trial_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/remote-miner.md",
            "docs/release.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
            "pyproject.toml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "Public Swarm v0.2 Usable Inference Trial",
        "public_swarm_trial_v1",
        "public_swarm_trial_check.py",
        "public_swarm_trial_pack.py",
        "crowdtensor swarm-trial",
        "swarm-trial local-loopback",
        "swarm-trial package",
        "swarm-trial live-kaggle",
        "swarm-trial evidence-import",
        "public_swarm_trial_ready",
        "serve_join_generate_trial_ready",
        "stage0_join_ready",
        "stage1_join_ready",
        "generate_ready",
        "generated_token_count_ready",
        "support_bundle_ready",
        "cpu_fallback_ready",
        "private_artifacts_cleaned",
        "operator_preview_import_ready",
        "gpu_generation_evidence_import_ready",
        "swarm_trial_degraded_cpu_fallback_ready",
        "external_runtime_blocked",
        "token_rotation_required",
        "CPU-only",
        "read-only",
        "not production Swarm Inference",
        "not libp2p",
        "not DHT",
        "not NAT traversal",
        "not GPU marketplace",
        "not large-model",
    ]:
        if fragment not in combined:
            details.append(f"Public Swarm v0.2 Usable Inference Trial docs/CI must mention {fragment}")
    return check_result("public_swarm_trial_docs", not details, details)


def check_public_swarm_gpu_inference_beta_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/remote-miner.md",
            "docs/release.md",
            "docs/protocol.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
            "pyproject.toml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "Public Swarm GPU Inference Beta",
        "public_swarm_gpu_inference_beta_v1",
        "public_swarm_gpu_inference_beta_check.py",
        "public_swarm_gpu_inference_beta_pack.py",
        "crowdtensor public-swarm-gpu-beta",
        "public-swarm-gpu-beta local-smoke",
        "public-swarm-gpu-beta local-loopback",
        "public-swarm-gpu-beta kaggle-package",
        "public-swarm-gpu-beta kaggle-auto",
        "public-swarm-gpu-beta evidence-import",
        "hf_transformers_cuda",
        "real_llm_sharded_cuda_stage0",
        "real_llm_sharded_cuda_stage1",
        "real_llm_sharded_cuda_both",
        "public_swarm_gpu_beta_smoke_ready",
        "public_swarm_gpu_beta_ready",
        "public_swarm_gpu_beta_kaggle_auto_ready",
        "gpu_runtime_ready",
        "cuda_runtime_available",
        "hf_transformers_cuda_ready",
        "gpu_stage0_ready",
        "gpu_stage1_ready",
        "kaggle_gpu_package_ready",
        "kaggle_kernels_deleted",
        "token_rotation_required",
        "external_gpu_runtime_verified",
        "read-only",
        "not production Swarm Inference",
        "not P2P",
        "not a GPU pooling marketplace",
        "not large-model",
    ]:
        if fragment not in combined:
            details.append(f"Public Swarm GPU Inference Beta docs/CI must mention {fragment}")
    return check_result("public_swarm_gpu_inference_beta_docs", not details, details)


def check_release_materials(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            "docs/release.md",
            ".github/release.yml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "0.1.0a0",
        "CHANGELOG.md",
        "docs/release.md",
        "scripts/release_gate.py",
        "scripts/runtime_acceptance_pack.py",
        "scripts/browser_acceptance_pack.py",
        "scripts/release_evidence_pack.py",
        "scripts/support_bundle.py",
        "git tag",
        "changelog:",
        "Runtime",
        "Security",
    ]:
        if fragment not in combined:
            details.append(f"release materials must mention {fragment}")
    return check_result("release_materials", not details, details)


def check_open_source_entrypoints(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "ROADMAP.md",
            "docs/protocol.md",
            "docs/use-cases.md",
            "docs/architecture.md",
            "site/index.html",
            ".github/ISSUE_TEMPLATE/bug_report.md",
            ".github/ISSUE_TEMPLATE/feature_request.md",
            ".github/pull_request_template.md",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "CrowdTensor",
        "CrowdTensorD",
        "home compute",
        "5-minute local swarm demo",
        "What Works Today",
        "What Is Not Ready",
        "ROADMAP.md",
        "docs/protocol.md",
        "docs/use-cases.md",
        "site/index.html",
        "runtime_contract_v1",
        "Swarm Inference",
        "Support Bundle",
        "Protocol boundary changed",
    ]:
        if fragment not in combined:
            details.append(f"open-source entrypoints must mention {fragment}")
    return check_result("open_source_entrypoints", not details, details)


def check_project_memory(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "AGENTS.md",
            "docs/project-memory.md",
            "README.md",
            "CONTRIBUTING.md",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "CrowdTensor",
        "CrowdTensorD",
        "runtime_contract_v1",
        "Swarm Inference",
        "Swarm Training",
        "not yet",
        "P2P/NAT",
        "CPU-only",
        "Support Bundle",
        "release gate",
        "home compute",
        "operator",
        "network/control-plane code",
        "project memory",
    ]:
        if fragment not in combined:
            details.append(f"project memory must mention {fragment}")
    return check_result("project_memory", not details, details)


def check_model_bundle_inference_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/api.md",
            "docs/protocol.md",
            "docs/architecture.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            "scripts/runtime_acceptance_pack.py",
            ".github/workflows/ci.yml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "model_bundle_infer",
        "model_bundle_infer_v1",
        "model_bundle_inference_smoke.py",
        "inference_session_demo.py",
        "home_compute_demo.py",
        "home_compute_demo_check.py",
        "--skip-home-compute-demo",
        "--skip-inference-session-demo",
        "--skip-model-bundle-inference",
        "--request-count",
        "inference_results",
        "request_trace",
        "request_count",
        "requests_per_second",
        "hardware_profile",
        "Swarm Inference",
        "read-only",
    ]:
        if fragment not in combined:
            details.append(f"model bundle inference docs/CI must mention {fragment}")
    return check_result("model_bundle_inference_docs", not details, details)


def check_admin_inference_session_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/api.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            "scripts/runtime_acceptance_pack.py",
            ".github/workflows/ci.yml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "POST /admin/inference-sessions",
        "admin_inference_session_check.py",
        "inference_session_request_v1",
        "task_id",
        "model_bundle_infer",
        "read-only",
        "--skip-admin-inference-session",
    ]:
        if fragment not in combined:
            details.append(f"admin inference session docs/CI must mention {fragment}")
    return check_result("admin_inference_session_docs", not details, details)


def check_inference_session_client_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/api.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            "scripts/runtime_acceptance_pack.py",
            ".github/workflows/ci.yml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "inference_session_client.py",
        "inference_session_client_check.py",
        "inference_session_client_v1",
        "session_client_ready",
        "--skip-inference-session-client",
        "POST /admin/inference-sessions",
        "task_id",
        "read-only",
    ]:
        if fragment not in combined:
            details.append(f"inference session client docs/CI must mention {fragment}")
    return check_result("inference_session_client_docs", not details, details)


def check_runtime_matrix_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/use-cases.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            "scripts/runtime_acceptance_pack.py",
            ".github/workflows/ci.yml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "runtime_matrix.py",
        "runtime_matrix_check.py",
        "home_compute_demo.py",
        "--skip-runtime-matrix",
        "runtime capability matrix",
        "hardware/runtime matrix",
        "hardware_targets",
        "recommended_routes",
        "route_decision",
        "matched_capabilities",
        "missing_capabilities",
        "diagnosis_summary",
        "hardware_diagnosis_summary",
        "diagnosis_codes",
        "operator_action",
        "hardware_profile",
        "CROWDTENSOR_LLM_RUNTIME_URL",
        "CPU-only",
    ]:
        if fragment not in combined:
            details.append(f"runtime matrix docs/CI must mention {fragment}")
    return check_result("runtime_matrix_docs", not details, details)


def check_external_llm_inference_docs(root: Path) -> dict[str, Any]:
    details: list[str] = []
    combined = "\n".join(
        read_text(root, path)
        for path in [
            "README.md",
            "docs/api.md",
            "docs/protocol.md",
            "docs/architecture.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            "scripts/runtime_acceptance_pack.py",
            ".github/workflows/ci.yml",
        ]
        if (root / path).exists()
    )
    for fragment in [
        "external_llm_infer",
        "external_llm_infer_v1",
        "external_llm_inference_smoke.py",
        "external_llm_http_adapter_smoke.py",
        "external_llm_evidence_pack.py",
        "external_llm_evidence_check.py",
        "external_llm_evidence_v1",
        "remote_external_llm_evidence_pack.py",
        "remote_external_llm_evidence_v1",
        "remote_external_llm_observability_v1",
        "crowdtensor llm-infer",
        "crowdtensor remote-demo",
        "llm_inference_cli_v1",
        "--skip-external-llm-inference",
        "--skip-external-llm-http-adapter",
        "--skip-external-llm-evidence",
        "--enable-mock-llm-runtime",
        "--llm-runtime-cmd",
        "--llm-runtime-url",
        "CROWDTENSOR_LLM_RUNTIME_URL",
        "CROWDTENSOR_LLM_RUNTIME_CMD",
        "external_llm_results",
        "completion_count",
        "output_chars",
        "adapter_kind",
        "read-only",
    ]:
        if fragment not in combined:
            details.append(f"external LLM inference docs/CI must mention {fragment}")
    return check_result("external_llm_inference_docs", not details, details)


def check_dockerfile(root: Path) -> dict[str, Any]:
    try:
        text = read_text(root, "Dockerfile")
    except OSError as exc:
        return check_result("dockerfile", False, [str(exc)])
    details: list[str] = []
    required_fragments = {
        "python:3.12-slim": "Dockerfile must use the Python slim base image",
        "useradd": "Dockerfile must create a non-root user",
        "USER crowdtensor": "Dockerfile must run as the crowdtensor user",
        'VOLUME ["/data"]': "Dockerfile must declare /data as a volume",
        "EXPOSE 8787": "Dockerfile must expose 8787",
        "crowdtensord": "Dockerfile must default to the crowdtensord entrypoint",
    }
    for fragment, message in required_fragments.items():
        if fragment not in text:
            details.append(message)
    return check_result("dockerfile", not details, details)


def check_compose(root: Path) -> dict[str, Any]:
    try:
        text = read_text(root, "compose.yaml")
    except OSError as exc:
        return check_result("compose", False, [str(exc)])
    details: list[str] = []
    required_fragments = {
        "services:": "compose.yaml must define services",
        "  coordinator:": "compose.yaml must define a coordinator service",
        "  miner:": "compose.yaml must define a miner service",
        "  web:": "compose.yaml must define a web service",
        '"8787:8787"': "coordinator service must expose 8787",
        '"8765:8765"': "web service must expose 8765",
        "http://coordinator:8787": "miner service must target the coordinator service URL",
        "crowdtensor-state": "compose.yaml must define persistent coordinator state",
    }
    for fragment, message in required_fragments.items():
        if fragment not in text:
            details.append(message)
    return check_result("compose", not details, details)


def check_dockerignore(root: Path) -> dict[str, Any]:
    try:
        lines = [
            line.strip()
            for line in read_text(root, ".dockerignore").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    except OSError as exc:
        return check_result("dockerignore", False, [str(exc)])
    details: list[str] = []
    if "Dockerfile" in lines:
        details.append(".dockerignore must not exclude Dockerfile")
    if "compose.yaml" in lines:
        details.append(".dockerignore must not exclude compose.yaml")
    for expected in ["__pycache__", ".pytest_cache", "state", ".env"]:
        if expected not in lines:
            details.append(f".dockerignore should exclude {expected}")
    return check_result("dockerignore", not details, details)


def check_ci_workflow(root: Path) -> dict[str, Any]:
    try:
        text = read_text(root, ".github/workflows/ci.yml")
    except OSError as exc:
        return check_result("ci_workflow", False, [str(exc)])
    details: list[str] = []
    required_fragments = {
        "python -m pip install -e .[dev]": "CI must install the dev package",
        "python -m py_compile": "CI must compile Python entrypoints",
        "python scripts/release_gate.py --json": "CI must run the release gate",
        "python scripts/release_readiness_check.py --allow-dirty": "CI must run the release readiness check",
        "python scripts/onboarding_gate.py --quick": "CI must run the fresh clone onboarding gate",
        "python scripts/user_friendly_inference_frontdoor_check.py --json": "CI must run the user-friendly inference frontdoor report check",
        "python scripts/doctor.py --json": "CI must run first-run doctor",
        "python scripts/runtime_matrix_check.py": "CI must run the runtime matrix check",
        "python scripts/home_compute_demo_check.py": "CI must run the home-compute demo check",
        "python scripts/home_compute_evidence_check.py": "CI must run the home-compute evidence check",
        "python scripts/remote_compute_evidence_check.py": "CI must run the remote-compute evidence check",
        "python scripts/remote_demo_runbook_check.py": "CI must run the remote demo runbook check",
        "python scripts/remote_demo_acceptance_check.py": "CI must run the remote demo acceptance check",
        "python scripts/remote_home_compute_demo_check.py": "CI must run the remote home-compute demo check",
        "python scripts/remote_two_machine_beta_check.py": "CI must run the remote two-machine Beta aggregate check",
        "python scripts/kaggle_remote_miner_beta_check.py": "CI must run the Kaggle Remote Miner Beta check",
        "python scripts/cpu_inference_beta_rc_check.py": "CI must run the CPU Inference Beta RC check",
        "python scripts/sharded_inference_check.py": "CI must run the sharded inference check",
        "python scripts/remote_sharded_inference_beta_check.py": "CI must run the remote sharded inference Beta check",
        "python scripts/micro_llm_sharded_inference_check.py": "CI must run the micro-LLM sharded inference check",
        "python scripts/remote_micro_llm_sharded_beta_check.py": "CI must run the remote micro-LLM sharded inference Beta check",
        "python scripts/stage_aware_micro_llm_sharded_check.py": "CI must run the stage-aware micro-LLM sharded inference check",
        "python scripts/micro_llm_live_rc_check.py": "CI must run the micro-LLM live two-node RC check",
        "python scripts/remote_real_llm_sharded_beta_check.py": "CI must run the real LLM sharded Beta check",
        "python scripts/real_llm_live_rc_check.py": "CI must run the real LLM live RC check",
        "python scripts/real_llm_internet_alpha_check.py": "CI must run the real LLM Internet Alpha check",
        "python scripts/real_llm_internet_beta_check.py": "CI must run the real LLM Internet Beta check",
        "python scripts/swarm_inference_beta_check.py": "CI must run the Swarm Inference Beta check",
        "python scripts/public_swarm_inference_beta_check.py": "CI must run the Public Swarm Inference Beta check",
        "python scripts/public_swarm_inference_beta_rc_check.py": "CI must run the Public Swarm Inference Beta RC local-loopback check",
        "python scripts/public_swarm_inference_beta_rc_check.py --mode package": "CI must run the Public Swarm Inference Beta RC package check",
        "python scripts/public_swarm_inference_beta_rc_check.py --mode external-existing": "CI must run the Public Swarm Inference Beta RC external-existing check",
        "python scripts/public_swarm_product_beta_check.py": "CI must run the Public Swarm Product Beta local-loopback check",
        "python scripts/public_swarm_product_beta_check.py --mode package": "CI must run the Public Swarm Product Beta package check",
        "python scripts/public_swarm_product_beta_check.py --mode external-existing": "CI must run the Public Swarm Product Beta external-existing check",
        "python scripts/public_swarm_developer_preview_check.py": "CI must run the Public Swarm Developer Preview local check",
        "python scripts/public_swarm_developer_preview_check.py --mode package": "CI must run the Public Swarm Developer Preview package check",
        "python scripts/public_swarm_developer_preview_check.py --mode external-existing": "CI must run the Public Swarm Developer Preview external-existing check",
        "python scripts/public_swarm_developer_preview_check.py --mode evidence-import": "CI must run the Public Swarm Developer Preview evidence-import check",
        "python scripts/public_swarm_live_preview_rc_check.py": "CI must run the Public Swarm Live Preview RC local-smoke check",
        "python scripts/public_swarm_live_preview_rc_check.py --mode package": "CI must run the Public Swarm Live Preview RC package check",
        "python scripts/public_swarm_live_preview_rc_check.py --mode live-kaggle": "CI must run the Public Swarm Live Preview RC live-kaggle fake-runner check",
        "python scripts/public_swarm_live_preview_rc_check.py --mode evidence-import": "CI must run the Public Swarm Live Preview RC evidence-import check",
        "python scripts/public_swarm_operator_preview_check.py": "CI must run the Public Swarm v0.1 Operator Preview local-smoke check",
        "python scripts/public_swarm_operator_preview_check.py --mode package": "CI must run the Public Swarm v0.1 Operator Preview package check",
        "python scripts/public_swarm_operator_preview_check.py --mode live-kaggle": "CI must run the Public Swarm v0.1 Operator Preview live-kaggle fake-runner check",
        "python scripts/public_swarm_operator_preview_check.py --mode evidence-import": "CI must run the Public Swarm v0.1 Operator Preview evidence-import check",
        "python scripts/public_swarm_trial_check.py": "CI must run the Public Swarm v0.2 Usable Inference Trial local-loopback check",
        "python scripts/public_swarm_trial_check.py --mode package": "CI must run the Public Swarm v0.2 Usable Inference Trial package check",
        "python scripts/public_swarm_trial_check.py --mode live-kaggle": "CI must run the Public Swarm v0.2 Usable Inference Trial live-kaggle fake-runner check",
        "python scripts/public_swarm_trial_check.py --mode evidence-import": "CI must run the Public Swarm v0.2 Usable Inference Trial evidence-import check",
        "python scripts/public_swarm_gpu_inference_beta_check.py": "CI must run the Public Swarm GPU Inference Beta smoke check",
        "python scripts/public_swarm_gpu_inference_beta_check.py --mode kaggle-package": "CI must run the Public Swarm GPU Inference Beta Kaggle package check",
        "python scripts/public_swarm_gpu_inference_beta_check.py --mode kaggle-auto": "CI must run the Public Swarm GPU Inference Beta Kaggle auto fake-runner check",
        "python scripts/remote_home_compute_demo_check.py --port 9185 --workload real-llm-sharded": "CI must run the real LLM sharded remote-demo loopback check",
        "remote-demo doctor": "CI/docs must cover the remote-demo doctor workflow",
        "remote-demo collect": "CI/docs must cover the remote-demo collect workflow",
        "remote-demo clean": "CI/docs must cover the remote-demo clean workflow",
        "--workload external-llm": "CI must run the external LLM remote-demo loopback check",
        "--workload micro-llm-sharded": "CI must run the micro-LLM sharded remote-demo loopback check",
        "python scripts/support_bundle.py": "CI must build support bundle",
        "python scripts/security_preflight.py --json": "CI must run the security preflight",
        "browser_acceptance_pack.py": "CI must run or skip the browser acceptance pack",
        "outer_optimizer_check.py": "CI must run the outer optimizer smoke through runtime acceptance",
        "compressed_error_feedback_check.py": "CI must run the compressed error-feedback smoke through runtime acceptance",
        "delta_transport_negotiation_check.py": "CI must run the delta transport negotiation smoke through runtime acceptance",
        "model_bundle_smoke.py": "CI must run the model bundle smoke through runtime acceptance",
        "model_bundle_inference_smoke.py": "CI must run the model bundle inference smoke through runtime acceptance",
        "inference_session_client_check.py": "CI must run the inference session client check through runtime acceptance",
        "admin_inference_session_check.py": "CI must run the admin inference session check through runtime acceptance",
        "external_llm_inference_smoke.py": "CI must run the external LLM inference smoke through runtime acceptance",
        "external_llm_http_adapter_smoke.py": "CI must run the external LLM HTTP adapter smoke through runtime acceptance",
        "release_evidence_pack.py": "CI must build release evidence",
        "release-evidence.json": "CI must write release evidence JSON",
        "python -m unittest discover -s tests -v": "CI must run unit tests",
        "runtime_acceptance_pack.py": "CI must run the runtime acceptance pack",
    }
    for fragment, message in required_fragments.items():
        if fragment not in text:
            details.append(message)
    return check_result("ci_workflow", not details, details)


def run_release_gate(root: str | Path = ROOT) -> dict[str, Any]:
    gate_root = Path(root).resolve()
    checks = [
        check_required_files(gate_root),
        check_pyproject(gate_root),
        check_markdown_links(gate_root),
        check_security_docs(gate_root),
        check_readiness_docs(gate_root),
        check_api_docs(gate_root),
        check_miner_resilience_docs(gate_root),
        check_result_idempotency_docs(gate_root),
        check_result_ledger_docs(gate_root),
        check_remote_miner_docs(gate_root),
        check_security_preflight_docs(gate_root),
        check_browser_acceptance_docs(gate_root),
        check_release_evidence_docs(gate_root),
        check_release_readiness_docs(gate_root),
        check_onboarding_gate_docs(gate_root),
        check_doctor_docs(gate_root),
        check_support_bundle_docs(gate_root),
        check_home_compute_evidence_docs(gate_root),
        check_remote_compute_evidence_docs(gate_root),
        check_remote_demo_runbook_docs(gate_root),
        check_remote_demo_acceptance_docs(gate_root),
        check_demo_manifest_docs(gate_root),
        check_local_proof_docs(gate_root),
        check_cleanup_docs(gate_root),
        check_remote_cli_docs(gate_root),
        check_remote_home_compute_demo_docs(gate_root),
        check_remote_two_machine_beta_docs(gate_root),
        check_kaggle_remote_miner_beta_docs(gate_root),
        check_kaggle_real_runtime_acceptance_docs(gate_root),
        check_home_inference_cli_docs(gate_root),
        check_cpu_inference_beta_docs(gate_root),
        check_cpu_inference_beta_rc_docs(gate_root),
        check_sharded_inference_docs(gate_root),
        check_remote_sharded_inference_beta_docs(gate_root),
        check_micro_llm_sharded_inference_docs(gate_root),
        check_remote_micro_llm_sharded_beta_docs(gate_root),
        check_micro_llm_live_rc_docs(gate_root),
        check_real_llm_sharded_beta_docs(gate_root),
        check_real_llm_live_rc_docs(gate_root),
        check_real_llm_internet_alpha_docs(gate_root),
        check_real_llm_internet_beta_docs(gate_root),
        check_swarm_inference_beta_docs(gate_root),
        check_public_swarm_inference_alpha_docs(gate_root),
        check_public_swarm_inference_alpha_rc_docs(gate_root),
        check_public_swarm_inference_beta_docs(gate_root),
        check_public_swarm_inference_beta_rc_docs(gate_root),
        check_public_swarm_product_beta_docs(gate_root),
        check_public_swarm_developer_preview_docs(gate_root),
        check_public_swarm_live_preview_rc_docs(gate_root),
        check_public_swarm_operator_preview_docs(gate_root),
        check_public_swarm_trial_docs(gate_root),
        check_public_swarm_gpu_inference_beta_docs(gate_root),
        check_release_materials(gate_root),
        check_open_source_entrypoints(gate_root),
        check_project_memory(gate_root),
        check_model_bundle_inference_docs(gate_root),
        check_admin_inference_session_docs(gate_root),
        check_inference_session_client_docs(gate_root),
        check_runtime_matrix_docs(gate_root),
        check_external_llm_inference_docs(gate_root),
        check_dockerfile(gate_root),
        check_compose(gate_root),
        check_dockerignore(gate_root),
        check_ci_workflow(gate_root),
    ]
    return {
        "ok": all(check["ok"] for check in checks),
        "root": str(gate_root),
        "checks": checks,
    }


def print_human(report: dict[str, Any]) -> None:
    for check in report["checks"]:
        status = "ok" if check["ok"] else "FAIL"
        print(f"{status} {check['name']}")
        for detail in check["details"]:
            print(f"  - {detail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CrowdTensorD Alpha release gate checks.")
    parser.add_argument("--root", default=str(ROOT), help="repository root to check")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_release_gate(args.root)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print_human(report)
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
