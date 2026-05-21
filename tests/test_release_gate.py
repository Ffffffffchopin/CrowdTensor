from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "release_gate.py"
SPEC = importlib.util.spec_from_file_location("release_gate", SCRIPT_PATH)
release_gate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(release_gate)


def copy_release_fixture(tmp_root: Path) -> Path:
    for relative in release_gate.REQUIRED_FILES:
        source = ROOT / relative
        target = tmp_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return tmp_root


def failed_details(report: dict, check_name: str) -> list[str]:
    check = next(check for check in report["checks"] if check["name"] == check_name)
    return check["details"]


class ReleaseGateTests(unittest.TestCase):
    def test_current_tree_passes_release_gate(self) -> None:
        report = release_gate.run_release_gate(ROOT)

        self.assertTrue(report["ok"], report)
        self.assertEqual(
            [check["name"] for check in report["checks"]],
            [
                "required_files",
                "pyproject",
                "markdown_links",
                "security_docs",
                "readiness_docs",
                "api_docs",
                "miner_resilience_docs",
                "result_idempotency_docs",
                "result_ledger_docs",
                "remote_miner_docs",
                "security_preflight_docs",
                "browser_acceptance_docs",
                "release_evidence_docs",
                "doctor_docs",
                "support_bundle_docs",
                "release_materials",
                "open_source_entrypoints",
                "project_memory",
                "model_bundle_inference_docs",
                "dockerfile",
                "compose",
                "dockerignore",
                "ci_workflow",
            ],
        )

    def test_missing_required_file_fails_with_path(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        (tmp_root / "LICENSE").unlink()

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        self.assertIn("LICENSE", failed_details(report, "required_files"))

    def test_broken_markdown_link_fails(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        readme = tmp_root / "README.md"
        readme.write_text(readme.read_text(encoding="utf-8") + "\n[Broken](docs/missing.md)\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        self.assertTrue(any("docs/missing.md" in detail for detail in failed_details(report, "markdown_links")))

    def test_missing_console_script_fails_pyproject_check(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        pyproject = tmp_root / "pyproject.toml"
        pyproject.write_text(
            pyproject.read_text(encoding="utf-8").replace('crowdtensord = "coordinator:main"\n', ""),
            encoding="utf-8",
        )

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        self.assertTrue(any("crowdtensord" in detail for detail in failed_details(report, "pyproject")))

    def test_missing_compose_miner_service_fails(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        compose = tmp_root / "compose.yaml"
        compose.write_text(
            compose.read_text(encoding="utf-8").replace("  miner:", "  removed_miner:"),
            encoding="utf-8",
        )

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        self.assertTrue(any("miner service" in detail for detail in failed_details(report, "compose")))

    def test_dockerignore_must_not_exclude_release_inputs(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        dockerignore = tmp_root / ".dockerignore"
        dockerignore.write_text(dockerignore.read_text(encoding="utf-8") + "\nDockerfile\ncompose.yaml\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "dockerignore")
        self.assertTrue(any("Dockerfile" in detail for detail in details))
        self.assertTrue(any("compose.yaml" in detail for detail in details))

    def test_security_docs_must_describe_hashed_tokens(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        security = tmp_root / "docs" / "security.md"
        security.write_text("No hashed token docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "security_docs")
        self.assertTrue(any("sha256:" in detail for detail in details))

    def test_readiness_docs_must_describe_public_profile_endpoints(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        (tmp_root / "README.md").write_text("No readiness docs here.\n", encoding="utf-8")
        (tmp_root / "docs" / "operations.md").write_text("No readiness docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "readiness_docs")
        self.assertTrue(any("/ready" in detail for detail in details))

    def test_api_docs_must_describe_endpoint_contract(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        (tmp_root / "docs" / "api.md").write_text("No API contract docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "api_docs")
        self.assertTrue(any("POST /tasks/claim" in detail for detail in details))

    def test_miner_resilience_docs_must_describe_retry_controls(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        (tmp_root / "README.md").write_text("No Miner resilience docs here.\n", encoding="utf-8")
        (tmp_root / "docs" / "operations.md").write_text("No Miner resilience docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "miner_resilience_docs")
        self.assertTrue(any("--max-request-attempts" in detail for detail in details))

    def test_result_idempotency_docs_must_describe_result_retry_contract(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        (tmp_root / "README.md").write_text("No result idempotency docs here.\n", encoding="utf-8")
        (tmp_root / "docs" / "operations.md").write_text("No result idempotency docs here.\n", encoding="utf-8")
        (tmp_root / "docs" / "api.md").write_text("No result idempotency docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "result_idempotency_docs")
        self.assertTrue(any("idempotency_key" in detail for detail in details))

    def test_result_ledger_docs_must_describe_operator_ledger(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        (tmp_root / "README.md").write_text("No result ledger docs here.\n", encoding="utf-8")
        (tmp_root / "docs" / "operations.md").write_text("No result ledger docs here.\n", encoding="utf-8")
        (tmp_root / "docs" / "api.md").write_text("No result ledger docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "result_ledger_docs")
        self.assertTrue(any("GET /admin/results" in detail for detail in details))

    def test_remote_miner_docs_must_describe_invite_flow(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        (tmp_root / "README.md").write_text("No remote Miner docs here.\n", encoding="utf-8")
        (tmp_root / "docs" / "quickstart.md").write_text("No remote Miner docs here.\n", encoding="utf-8")
        (tmp_root / "docs" / "remote-miner.md").write_text("No remote Miner docs here.\n", encoding="utf-8")
        (tmp_root / "docs" / "operations.md").write_text("No remote Miner docs here.\n", encoding="utf-8")
        (tmp_root / "docs" / "security.md").write_text("No remote Miner docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "remote_miner_docs")
        self.assertTrue(any("scripts/create_miner_invite.py" in detail for detail in details))

    def test_security_preflight_docs_must_describe_preflight_gate(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        (tmp_root / "README.md").write_text("No security preflight docs here.\n", encoding="utf-8")
        (tmp_root / "docs" / "quickstart.md").write_text("No security preflight docs here.\n", encoding="utf-8")
        (tmp_root / "docs" / "remote-miner.md").write_text("No security preflight docs here.\n", encoding="utf-8")
        (tmp_root / "docs" / "operations.md").write_text("No security preflight docs here.\n", encoding="utf-8")
        (tmp_root / "docs" / "security.md").write_text("No security preflight docs here.\n", encoding="utf-8")
        (tmp_root / ".github" / "workflows" / "ci.yml").write_text("No security preflight docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "security_preflight_docs")
        self.assertTrue(any("scripts/security_preflight.py" in detail for detail in details))

    def test_browser_acceptance_docs_must_describe_browser_pack(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        (tmp_root / "README.md").write_text("No browser acceptance docs here.\n", encoding="utf-8")
        (tmp_root / "docs" / "quickstart.md").write_text("No browser acceptance docs here.\n", encoding="utf-8")
        (tmp_root / "docs" / "operations.md").write_text("No browser acceptance docs here.\n", encoding="utf-8")
        (tmp_root / ".github" / "workflows" / "ci.yml").write_text("No browser acceptance docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "browser_acceptance_docs")
        self.assertTrue(any("scripts/browser_acceptance_pack.py" in detail for detail in details))

    def test_release_evidence_docs_must_describe_evidence_pack(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        (tmp_root / "README.md").write_text("No release evidence docs here.\n", encoding="utf-8")
        (tmp_root / "docs" / "operations.md").write_text("No release evidence docs here.\n", encoding="utf-8")
        (tmp_root / ".github" / "workflows" / "ci.yml").write_text("No release evidence docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "release_evidence_docs")
        self.assertTrue(any("scripts/release_evidence_pack.py" in detail for detail in details))

    def test_doctor_docs_must_describe_first_run_diagnostics(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        (tmp_root / "README.md").write_text("No doctor docs here.\n", encoding="utf-8")
        (tmp_root / "docs" / "quickstart.md").write_text("No doctor docs here.\n", encoding="utf-8")
        (tmp_root / "docs" / "operations.md").write_text("No doctor docs here.\n", encoding="utf-8")
        (tmp_root / "CONTRIBUTING.md").write_text("No doctor docs here.\n", encoding="utf-8")
        (tmp_root / ".github" / "workflows" / "ci.yml").write_text("No doctor docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "doctor_docs")
        self.assertTrue(any("scripts/doctor.py" in detail for detail in details))

    def test_support_bundle_docs_must_describe_operator_bundle(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        (tmp_root / "README.md").write_text("No support bundle docs here.\n", encoding="utf-8")
        (tmp_root / "docs" / "operations.md").write_text("No support bundle docs here.\n", encoding="utf-8")
        (tmp_root / "docs" / "security.md").write_text("No support bundle docs here.\n", encoding="utf-8")
        (tmp_root / ".github" / "workflows" / "ci.yml").write_text("No support bundle docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "support_bundle_docs")
        self.assertTrue(any("scripts/support_bundle.py" in detail for detail in details))

    def test_release_materials_must_describe_release_flow(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        (tmp_root / "README.md").write_text("No release materials here.\n", encoding="utf-8")
        (tmp_root / "CHANGELOG.md").write_text("No release materials here.\n", encoding="utf-8")
        (tmp_root / "CONTRIBUTING.md").write_text("No release materials here.\n", encoding="utf-8")
        (tmp_root / "docs" / "release.md").write_text("No release materials here.\n", encoding="utf-8")
        (tmp_root / ".github" / "release.yml").write_text("No release materials here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "release_materials")
        self.assertTrue(any("scripts/release_gate.py" in detail for detail in details))
        self.assertTrue(any("git tag" in detail for detail in details))

    def test_open_source_entrypoints_must_describe_public_positioning(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
            "README.md",
            "ROADMAP.md",
            "docs/protocol.md",
            "docs/use-cases.md",
            "docs/architecture.md",
            "site/index.html",
            ".github/ISSUE_TEMPLATE/bug_report.md",
            ".github/ISSUE_TEMPLATE/feature_request.md",
            ".github/pull_request_template.md",
        ]:
            (tmp_root / relative).write_text("No open-source positioning here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "open_source_entrypoints")
        self.assertTrue(any("home compute" in detail for detail in details))
        self.assertTrue(any("runtime_contract_v1" in detail for detail in details))

    def test_project_memory_must_preserve_long_term_context(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        (tmp_root / "AGENTS.md").write_text("No project memory here.\n", encoding="utf-8")
        (tmp_root / "docs" / "project-memory.md").write_text("No project memory here.\n", encoding="utf-8")
        (tmp_root / "README.md").write_text("No project memory here.\n", encoding="utf-8")
        (tmp_root / "CONTRIBUTING.md").write_text("No project memory here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "project_memory")
        self.assertTrue(any("runtime_contract_v1" in detail for detail in details))
        self.assertTrue(any("P2P/NAT" in detail for detail in details))

    def test_model_bundle_inference_docs_must_describe_probe_and_acceptance(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
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
        ]:
            (tmp_root / relative).write_text("No model bundle inference docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "model_bundle_inference_docs")
        self.assertTrue(any("model_bundle_infer" in detail for detail in details))
        self.assertTrue(any("--skip-model-bundle-inference" in detail for detail in details))
        self.assertTrue(any("--request-count" in detail for detail in details))

    def _tmp_dir(self) -> str:
        path = Path(self.id().replace(".", "_").replace("/", "_"))
        tmp_root = Path("/tmp") / f"crowdtensor_release_gate_{path.name}"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True)
        self.addCleanup(lambda: shutil.rmtree(tmp_root, ignore_errors=True))
        return str(tmp_root)


if __name__ == "__main__":
    unittest.main()
