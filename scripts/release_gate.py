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
    "scripts/remote_miner_join_check.py",
    "scripts/miner_resilience_check.py",
    "scripts/readiness_check.py",
    "scripts/runtime_matrix.py",
    "scripts/runtime_matrix_check.py",
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
    "scripts/external_llm_inference_smoke.py",
    "scripts/external_llm_http_adapter_smoke.py",
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
    for fragment in ["sha256:", "scripts/hash_token.py", "hashed token"]:
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
    ]:
        if fragment not in combined:
            details.append(f"release evidence docs or CI must mention {fragment}")
    return check_result("release_evidence_docs", not details, details)


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
    ]:
        if fragment not in combined:
            details.append(f"support bundle docs or CI must mention {fragment}")
    return check_result("support_bundle_docs", not details, details)


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
        "--skip-inference-session-demo",
        "--skip-model-bundle-inference",
        "--request-count",
        "inference_results",
        "request_count",
        "requests_per_second",
        "hardware_profile",
        "Swarm Inference",
        "read-only",
    ]:
        if fragment not in combined:
            details.append(f"model bundle inference docs/CI must mention {fragment}")
    return check_result("model_bundle_inference_docs", not details, details)


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
        "--skip-runtime-matrix",
        "runtime capability matrix",
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
        "--skip-external-llm-inference",
        "--skip-external-llm-http-adapter",
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
        "python scripts/doctor.py --json": "CI must run first-run doctor",
        "python scripts/runtime_matrix_check.py": "CI must run the runtime matrix check",
        "python scripts/support_bundle.py": "CI must build support bundle",
        "python scripts/security_preflight.py --json": "CI must run the security preflight",
        "browser_acceptance_pack.py": "CI must run or skip the browser acceptance pack",
        "outer_optimizer_check.py": "CI must run the outer optimizer smoke through runtime acceptance",
        "compressed_error_feedback_check.py": "CI must run the compressed error-feedback smoke through runtime acceptance",
        "delta_transport_negotiation_check.py": "CI must run the delta transport negotiation smoke through runtime acceptance",
        "model_bundle_smoke.py": "CI must run the model bundle smoke through runtime acceptance",
        "model_bundle_inference_smoke.py": "CI must run the model bundle inference smoke through runtime acceptance",
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
        check_doctor_docs(gate_root),
        check_support_bundle_docs(gate_root),
        check_release_materials(gate_root),
        check_open_source_entrypoints(gate_root),
        check_project_memory(gate_root),
        check_model_bundle_inference_docs(gate_root),
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
