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
                "release_readiness_docs",
                "onboarding_gate_docs",
                "doctor_docs",
                "support_bundle_docs",
                "home_compute_evidence_docs",
                "remote_compute_evidence_docs",
                "remote_demo_runbook_docs",
                "remote_demo_acceptance_docs",
                "demo_manifest_docs",
                "local_proof_docs",
                "cleanup_docs",
                "remote_cli_docs",
                "remote_home_compute_demo_docs",
                "remote_two_machine_beta_docs",
                "kaggle_remote_miner_beta_docs",
                "kaggle_real_runtime_acceptance_docs",
                "home_inference_cli_docs",
                "cpu_inference_beta_docs",
                "cpu_inference_beta_rc_docs",
                "sharded_inference_docs",
                "remote_sharded_inference_beta_docs",
                "micro_llm_sharded_inference_docs",
                "remote_micro_llm_sharded_beta_docs",
                "micro_llm_live_rc_docs",
                "real_llm_sharded_beta_docs",
                "real_llm_live_rc_docs",
                "real_llm_internet_alpha_docs",
                "real_llm_internet_beta_docs",
                "swarm_inference_beta_docs",
                "public_swarm_inference_alpha_docs",
                "public_swarm_inference_alpha_rc_docs",
                "public_swarm_inference_beta_docs",
                "public_swarm_inference_beta_rc_docs",
                "public_swarm_product_beta_docs",
                "public_swarm_developer_preview_docs",
                "public_swarm_gpu_inference_beta_docs",
                "release_materials",
                "open_source_entrypoints",
                "project_memory",
                "model_bundle_inference_docs",
                "admin_inference_session_docs",
                "inference_session_client_docs",
                "runtime_matrix_docs",
                "external_llm_inference_docs",
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
        self.assertTrue(any("diagnosis_summary" in detail for detail in details))

    def test_release_readiness_docs_must_describe_maintainer_gate(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
            "README.md",
            "docs/operations.md",
            "docs/release.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
            "pyproject.toml",
        ]:
            (tmp_root / relative).write_text("No release readiness docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "release_readiness_docs")
        self.assertTrue(any("crowdtensor release-ready" in detail for detail in details))
        self.assertTrue(any("release_readiness_v1" in detail for detail in details))
        self.assertTrue(any("--allow-dirty" in detail for detail in details))

    def test_onboarding_gate_docs_must_describe_fresh_clone_path(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/release.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
        ]:
            (tmp_root / relative).write_text("No onboarding docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "onboarding_gate_docs")
        self.assertTrue(any("scripts/onboarding_gate.py" in detail for detail in details))
        self.assertTrue(any("onboarding_gate_v1" in detail for detail in details))
        self.assertTrue(any("crowdtensor local-proof" in detail for detail in details))

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
        self.assertTrue(any("diagnosis_by_check" in detail for detail in details))

    def test_home_compute_evidence_docs_must_describe_shareable_pack(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
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
        ]:
            (tmp_root / relative).write_text("No home-compute evidence docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "home_compute_evidence_docs")
        self.assertTrue(any("home_compute_evidence_pack.py" in detail for detail in details))
        self.assertTrue(any("--skip-home-compute-evidence" in detail for detail in details))
        self.assertTrue(any("home_compute_ready" in detail for detail in details))

    def test_remote_compute_evidence_docs_must_describe_remote_pack(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
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
        ]:
            (tmp_root / relative).write_text("No remote-compute evidence docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "remote_compute_evidence_docs")
        self.assertTrue(any("remote_compute_evidence_pack.py" in detail for detail in details))
        self.assertTrue(any("--include-remote-evidence" in detail for detail in details))

    def test_remote_demo_runbook_docs_must_describe_two_machine_pack(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
            "README.md",
            "docs/operations.md",
            "docs/remote-miner.md",
            "docs/use-cases.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
        ]:
            (tmp_root / relative).write_text("No remote demo runbook docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "remote_demo_runbook_docs")
        self.assertTrue(any("remote_demo_runbook_pack.py" in detail for detail in details))
        self.assertTrue(any("operator.private.env" in detail for detail in details))

    def test_remote_demo_acceptance_docs_must_describe_acceptance_pack(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
            "README.md",
            "docs/operations.md",
            "docs/remote-miner.md",
            "docs/use-cases.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
        ]:
            (tmp_root / relative).write_text("No remote demo acceptance docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "remote_demo_acceptance_docs")
        self.assertTrue(any("remote_demo_acceptance_pack.py" in detail for detail in details))
        self.assertTrue(any("remote_demo_acceptance_v1" in detail for detail in details))
        self.assertTrue(any("coordinator_unreachable" in detail for detail in details))
        self.assertTrue(any("--create-session" in detail for detail in details))
        self.assertTrue(any("session_create_failed" in detail for detail in details))

    def test_demo_manifest_docs_must_describe_latest_artifact_entrypoint(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/release.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
        ]:
            (tmp_root / relative).write_text("No demo manifest docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "demo_manifest_docs")
        self.assertTrue(any("demo_manifest_pack.py" in detail for detail in details))
        self.assertTrue(any("demo_manifest_v1" in detail for detail in details))
        self.assertTrue(any("external_llm_evidence_v1" in detail for detail in details))
        self.assertTrue(any("latest output artifact" in detail for detail in details))

    def test_local_proof_docs_must_describe_one_command_entrypoint(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
            "pyproject.toml",
        ]:
            (tmp_root / relative).write_text("No local proof docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "local_proof_docs")
        self.assertTrue(any("crowdtensor local-proof" in detail for detail in details))
        self.assertTrue(any("local_proof_summary_v1" in detail for detail in details))
        self.assertTrue(any("crowdtensor/cli.py" in detail for detail in details))

    def test_cleanup_docs_must_describe_safe_artifact_cleanup(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "CHANGELOG.md",
            "pyproject.toml",
        ]:
            (tmp_root / relative).write_text("No cleanup docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "cleanup_docs")
        self.assertTrue(any("crowdtensor clean-artifacts" in detail for detail in details))
        self.assertTrue(any("cleanup_report_v1" in detail for detail in details))
        self.assertTrue(any("--include-reports" in detail for detail in details))

    def test_remote_cli_docs_must_describe_operator_entrypoints(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/remote-miner.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            "pyproject.toml",
        ]:
            (tmp_root / relative).write_text("No remote CLI docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "remote_cli_docs")
        self.assertTrue(any("crowdtensor remote-runbook" in detail for detail in details))
        self.assertTrue(any("remote_acceptance_cli_v1" in detail for detail in details))
        self.assertTrue(any("token redaction" in detail for detail in details))

    def test_home_inference_cli_docs_must_describe_user_entrypoint(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            "pyproject.toml",
        ]:
            (tmp_root / relative).write_text("No home inference CLI docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "home_inference_cli_docs")
        self.assertTrue(any("crowdtensor home-infer" in detail for detail in details))
        self.assertTrue(any("home_inference_cli_v1" in detail for detail in details))
        self.assertTrue(any("model_bundle_inference_scenario_v1" in detail for detail in details))
        self.assertTrue(any("request_trace" in detail for detail in details))

    def test_cpu_inference_beta_docs_must_describe_beta_entrypoint(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
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
        ]:
            (tmp_root / relative).write_text("No CPU inference Beta docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "cpu_inference_beta_docs")
        self.assertTrue(any("crowdtensor cpu-infer" in detail for detail in details))
        self.assertTrue(any("cpu_inference_beta_v1" in detail for detail in details))
        self.assertTrue(any("--mode remote-loopback" in detail for detail in details))

    def test_cpu_inference_beta_rc_docs_must_describe_release_candidate_entrypoint(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
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
        ]:
            (tmp_root / relative).write_text("No CPU Inference Beta RC docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "cpu_inference_beta_rc_docs")
        self.assertTrue(any("CPU Inference Beta RC" in detail for detail in details))
        self.assertTrue(any("cpu_inference_beta_rc_v1" in detail for detail in details))
        self.assertTrue(any("--mode beta-rc" in detail for detail in details))

    def test_kaggle_remote_miner_beta_docs_must_describe_kaggle_target(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
            "README.md",
            "docs/operations.md",
            "docs/quickstart.md",
            "docs/remote-miner.md",
            "docs/project-memory.md",
            "AGENTS.md",
            "ROADMAP.md",
            "CHANGELOG.md",
            ".github/workflows/ci.yml",
        ]:
            (tmp_root / relative).write_text("No Kaggle Remote Miner Beta docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "kaggle_remote_miner_beta_docs")
        self.assertTrue(any("Kaggle Remote Miner Beta" in detail for detail in details))
        self.assertTrue(any("--target kaggle" in detail for detail in details))
        self.assertTrue(any("kaggle_remote_miner_beta_check_v1" in detail for detail in details))

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
        self.assertTrue(any("inference_session_demo.py" in detail for detail in details))
        self.assertTrue(any("--skip-inference-session-demo" in detail for detail in details))
        self.assertTrue(any("--request-count" in detail for detail in details))

    def test_admin_inference_session_docs_must_describe_api_and_acceptance(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
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
        ]:
            (tmp_root / relative).write_text("No admin inference docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "admin_inference_session_docs")
        self.assertTrue(any("POST /admin/inference-sessions" in detail for detail in details))
        self.assertTrue(any("admin_inference_session_check.py" in detail for detail in details))
        self.assertTrue(any("inference_session_request_v1" in detail for detail in details))
        self.assertTrue(any("--skip-admin-inference-session" in detail for detail in details))

    def test_inference_session_client_docs_must_describe_user_client_and_acceptance(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
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
        ]:
            (tmp_root / relative).write_text("No inference session client docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "inference_session_client_docs")
        self.assertTrue(any("inference_session_client.py" in detail for detail in details))
        self.assertTrue(any("inference_session_client_v1" in detail for detail in details))
        self.assertTrue(any("session_client_ready" in detail for detail in details))
        self.assertTrue(any("--skip-inference-session-client" in detail for detail in details))

    def test_external_llm_inference_docs_must_describe_adapter_and_acceptance(self) -> None:
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
            (tmp_root / relative).write_text("No external LLM inference docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "external_llm_inference_docs")
        self.assertTrue(any("external_llm_infer" in detail for detail in details))
        self.assertTrue(any("external_llm_http_adapter_smoke.py" in detail for detail in details))
        self.assertTrue(any("external_llm_evidence_pack.py" in detail for detail in details))
        self.assertTrue(any("crowdtensor llm-infer" in detail for detail in details))
        self.assertTrue(any("--enable-mock-llm-runtime" in detail for detail in details))
        self.assertTrue(any("--llm-runtime-cmd" in detail for detail in details))
        self.assertTrue(any("--llm-runtime-url" in detail for detail in details))
        self.assertTrue(any("--skip-external-llm-inference" in detail for detail in details))
        self.assertTrue(any("--skip-external-llm-http-adapter" in detail for detail in details))
        self.assertTrue(any("--skip-external-llm-evidence" in detail for detail in details))

    def test_runtime_matrix_docs_must_describe_user_capability_path(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
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
        ]:
            (tmp_root / relative).write_text("No runtime matrix docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "runtime_matrix_docs")
        self.assertTrue(any("runtime_matrix.py" in detail for detail in details))
        self.assertTrue(any("runtime_matrix_check.py" in detail for detail in details))
        self.assertTrue(any("--skip-runtime-matrix" in detail for detail in details))
        self.assertTrue(any("diagnosis_summary" in detail for detail in details))
        self.assertTrue(any("operator_action" in detail for detail in details))

    def test_real_llm_sharded_beta_docs_must_describe_optional_hf_split_path(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
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
        ]:
            (tmp_root / relative).write_text("No real LLM sharded docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "real_llm_sharded_beta_docs")
        self.assertTrue(any("real_llm_sharded_infer" in detail for detail in details))
        self.assertTrue(any("real_llm_artifact_v1" in detail for detail in details))
        self.assertTrue(any("remote_real_llm_sharded_beta_check.py" in detail for detail in details))
        self.assertTrue(any("crowdtensor remote-demo --workload real-llm-sharded" in detail for detail in details))
        self.assertTrue(any("remote_real_llm_sharded_acceptance_v1" in detail for detail in details))
        self.assertTrue(any("remote_python_real_llm_sharded_infer" in detail for detail in details))
        self.assertTrue(any("hf_dependencies_missing" in detail for detail in details))
        self.assertTrue(any("--enable-hf-tiny-gpt-runtime" in detail for detail in details))
        self.assertTrue(any("real_llm_sharded_stage0" in detail for detail in details))
        self.assertTrue(any("optional [hf]" in detail for detail in details))

    def test_real_llm_live_rc_docs_must_describe_generated_external_path(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
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
        ]:
            (tmp_root / relative).write_text("No real LLM live RC docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "real_llm_live_rc_docs")
        self.assertTrue(any("real_llm_live_rc_v1" in detail for detail in details))
        self.assertTrue(any("real_llm_live_rc_check.py" in detail for detail in details))
        self.assertTrue(any("crowdtensor real-llm-live-rc" in detail for detail in details))
        self.assertTrue(any("kaggle-upload-real-llm-stage0" in detail for detail in details))
        self.assertTrue(any("external_runtime_verified" in detail for detail in details))

    def test_real_llm_internet_alpha_docs_must_describe_ready_and_requeue_path(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
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
        ]:
            (tmp_root / relative).write_text("No real LLM Internet Alpha docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "real_llm_internet_alpha_docs")
        self.assertTrue(any("real_llm_internet_alpha_v1" in detail for detail in details))
        self.assertTrue(any("real_llm_internet_alpha_check.py" in detail for detail in details))
        self.assertTrue(any("crowdtensor real-llm-internet-alpha" in detail for detail in details))
        self.assertTrue(any("real_llm_stage_requeue_ready" in detail for detail in details))
        self.assertTrue(any("stage_requeue_ready" in detail for detail in details))

    def test_real_llm_internet_beta_docs_must_describe_kaggle_auto_cleanup_path(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
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
        ]:
            (tmp_root / relative).write_text("No real LLM Internet Beta docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "real_llm_internet_beta_docs")
        self.assertTrue(any("real_llm_internet_beta_v1" in detail for detail in details))
        self.assertTrue(any("real_llm_internet_beta_check.py" in detail for detail in details))
        self.assertTrue(any("crowdtensor real-llm-internet-beta" in detail for detail in details))
        self.assertTrue(any("kaggle-auto" in detail for detail in details))
        self.assertTrue(any("kaggle_kernels_deleted" in detail for detail in details))
        self.assertTrue(any("external_stage_requeue_ready" in detail for detail in details))

    def test_public_swarm_inference_alpha_docs_must_describe_session_entrypoint(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
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
        ]:
            (tmp_root / relative).write_text("No Public Swarm Inference Alpha docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "public_swarm_inference_alpha_docs")
        self.assertTrue(any("public_swarm_inference_alpha_v1" in detail for detail in details))
        self.assertTrue(any("public_swarm_inference_alpha_check.py" in detail for detail in details))
        self.assertTrue(any("crowdtensor swarm-session" in detail for detail in details))
        self.assertTrue(any("local_stage_requeue_ready" in detail for detail in details))
        self.assertTrue(any("public_swarm_live_requeue_ready" in detail for detail in details))
        self.assertTrue(any("live-kaggle" in detail for detail in details))

    def test_public_swarm_inference_alpha_rc_docs_must_describe_release_candidate(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
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
        ]:
            (tmp_root / relative).write_text("No Public Swarm Inference Alpha RC docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "public_swarm_inference_alpha_rc_docs")
        self.assertTrue(any("public_swarm_inference_alpha_rc_v1" in detail for detail in details))
        self.assertTrue(any("public_swarm_inference_alpha_rc_check.py" in detail for detail in details))
        self.assertTrue(any("crowdtensor public-swarm-alpha-rc" in detail for detail in details))
        self.assertTrue(any("stage0_live_requeue_evidence_ready" in detail for detail in details))
        self.assertTrue(any("evidence-import" in detail for detail in details))

    def test_public_swarm_inference_beta_docs_must_describe_user_beta(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
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
        ]:
            (tmp_root / relative).write_text("No Public Swarm Inference Beta docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "public_swarm_inference_beta_docs")
        self.assertTrue(any("public_swarm_inference_beta_v1" in detail for detail in details))
        self.assertTrue(any("public_swarm_inference_beta_check.py" in detail for detail in details))
        self.assertTrue(any("crowdtensor public-swarm-beta" in detail for detail in details))
        self.assertTrue(any("public-swarm-beta product-beta" in detail for detail in details))
        self.assertTrue(any("public_swarm_product_beta_ready" in detail for detail in details))
        self.assertTrue(any("p2p_lite_discovery_ready" in detail for detail in details))
        self.assertTrue(any("cpu_fallback_ready" in detail for detail in details))
        self.assertTrue(any("local_loopback_ready" in detail for detail in details))
        self.assertTrue(any("external_live_evidence_imported" in detail for detail in details))

    def test_public_swarm_inference_beta_rc_docs_must_describe_release_candidate(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
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
        ]:
            (tmp_root / relative).write_text("No Public Swarm Inference Beta RC docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "public_swarm_inference_beta_rc_docs")
        self.assertTrue(any("public_swarm_inference_beta_rc_v1" in detail for detail in details))
        self.assertTrue(any("public_swarm_inference_beta_rc_check.py" in detail for detail in details))
        self.assertTrue(any("crowdtensor public-swarm-beta-rc" in detail for detail in details))
        self.assertTrue(any("serve_join_generate_loop_ready" in detail for detail in details))
        self.assertTrue(any("p2p_lite_route_ready" in detail for detail in details))
        self.assertTrue(any("private_artifacts_local_only" in detail for detail in details))
        self.assertTrue(any("external_runtime_verified" in detail for detail in details))
        self.assertTrue(any("hf_dependencies_missing" in detail for detail in details))
        ci_details = failed_details(report, "ci_workflow")
        self.assertTrue(any("Public Swarm Inference Beta RC package" in detail for detail in ci_details))
        self.assertTrue(any("Public Swarm Inference Beta RC external-existing" in detail for detail in ci_details))

    def test_public_swarm_product_beta_docs_must_describe_user_product_path(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
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
        ]:
            (tmp_root / relative).write_text("No Public Swarm Product Beta docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "public_swarm_product_beta_docs")
        self.assertTrue(any("public_swarm_product_beta_v1" in detail for detail in details))
        self.assertTrue(any("public_swarm_product_beta_check.py" in detail for detail in details))
        self.assertTrue(any("crowdtensor public-swarm-product-beta" in detail for detail in details))
        self.assertTrue(any("serve_ready" in detail for detail in details))
        self.assertTrue(any("stage0_join_ready" in detail for detail in details))
        self.assertTrue(any("support_bundle_ready" in detail for detail in details))
        self.assertTrue(any("private_artifacts_cleaned" in detail for detail in details))
        ci_details = failed_details(report, "ci_workflow")
        self.assertTrue(any("Public Swarm Product Beta package" in detail for detail in ci_details))
        self.assertTrue(any("Public Swarm Product Beta external-existing" in detail for detail in ci_details))

    def test_public_swarm_developer_preview_docs_must_describe_user_preview_path(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
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
        ]:
            (tmp_root / relative).write_text("No Public Swarm Developer Preview docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "public_swarm_developer_preview_docs")
        self.assertTrue(any("public_swarm_developer_preview_v1" in detail for detail in details))
        self.assertTrue(any("public_swarm_developer_preview_check.py" in detail for detail in details))
        self.assertTrue(any("crowdtensor preview" in detail for detail in details))
        self.assertTrue(any("local_two_stage_generation_ready" in detail for detail in details))
        self.assertTrue(any("support_bundle_ready" in detail for detail in details))
        self.assertTrue(any("gpu_generation_evidence_import_ready" in detail for detail in details))
        ci_details = failed_details(report, "ci_workflow")
        self.assertTrue(any("Public Swarm Developer Preview package" in detail for detail in ci_details))
        self.assertTrue(any("Public Swarm Developer Preview evidence-import" in detail for detail in ci_details))

    def test_public_swarm_gpu_inference_beta_docs_must_describe_optional_cuda_beta(self) -> None:
        tmp_root = copy_release_fixture(Path(self._tmp_dir()))
        for relative in [
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
        ]:
            (tmp_root / relative).write_text("No Public Swarm GPU Inference Beta docs here.\n", encoding="utf-8")

        report = release_gate.run_release_gate(tmp_root)

        self.assertFalse(report["ok"])
        details = failed_details(report, "public_swarm_gpu_inference_beta_docs")
        self.assertTrue(any("public_swarm_gpu_inference_beta_v1" in detail for detail in details))
        self.assertTrue(any("public_swarm_gpu_inference_beta_check.py" in detail for detail in details))
        self.assertTrue(any("crowdtensor public-swarm-gpu-beta" in detail for detail in details))
        self.assertTrue(any("hf_transformers_cuda" in detail for detail in details))
        self.assertTrue(any("kaggle_gpu_package_ready" in detail for detail in details))
        self.assertTrue(any("public_swarm_gpu_beta_kaggle_auto_ready" in detail for detail in details))
        ci_details = failed_details(report, "ci_workflow")
        self.assertTrue(any("Public Swarm GPU Inference Beta Kaggle package" in detail for detail in ci_details))
        self.assertTrue(any("Public Swarm GPU Inference Beta Kaggle auto" in detail for detail in ci_details))

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
