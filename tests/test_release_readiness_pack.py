from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "release_readiness_pack.py"
SPEC = importlib.util.spec_from_file_location("release_readiness_pack", SCRIPT_PATH)
release_readiness_pack = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(release_readiness_pack)


def completed(payload: dict, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["cmd"], returncode=returncode, stdout=json.dumps(payload) + "\n", stderr="")


def args_for(output_dir: Path, **overrides: object) -> argparse.Namespace:
    values = {
        "root": str(ROOT),
        "output_dir": str(output_dir),
        "host": "127.0.0.1",
        "base_port": 9024,
        "request_count": 4,
        "external_llm_request_count": 3,
        "timeout_seconds": 30,
        "allow_dirty": False,
        "skip_demo_manifest": False,
        "skip_external_llm_evidence": False,
        "runtime_report": "",
        "browser_report": "",
        "remote_report": "",
        "git_dir": "",
        "work_tree": "",
        "security_host": "127.0.0.1",
        "miner_token": "",
        "observer_token": "",
        "admin_token": "",
        "miner_token_registry": "",
        "cors_origins": [],
        "security_strict": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def release_gate_report(ok: bool = True) -> dict:
    return {"ok": ok, "checks": [{"name": "required_files", "ok": ok, "details": [] if ok else ["missing"]}]}


def security_report(ok: bool = True) -> dict:
    return {"ok": ok, "checks": [{"id": "bind_host", "ok": ok, "severity": "info"}]}


def git_info(*, dirty: bool = False, available: bool = True) -> dict:
    return {
        "available": available,
        "commit": "abc123",
        "branch": "main",
        "remote_origin": "https://token@github.com/Ffffffffchopin/CrowdTensor.git",
        "dirty": dirty,
        "status": [" M README.md"] if dirty else [],
        "error": "" if available else "not a git repository",
    }


def demo_manifest(ok: bool = True) -> dict:
    return {
        "schema": "demo_manifest_v1",
        "mode": "local-loopback",
        "ok": ok,
        "artifacts": {
            "runtime_matrix": {"present": True},
            "remote_compute_evidence_json": {"present": True},
            "support_bundle_json": {"present": True},
            "demo_manifest_markdown": {"present": True},
        },
        "summaries": {
            "remote_compute_evidence": {"ok": True},
            "external_llm_evidence": {"ok": True},
            "support_bundle": {"release_gate_ok": True},
        },
    }


class ReleaseReadinessPackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_readiness_test_"))

    def tearDown(self) -> None:
        temp_root = Path(tempfile.gettempdir())
        for path in temp_root.glob("crowdtensor_readiness_test_*"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)

    def test_clean_inputs_build_ready_report_and_artifacts(self) -> None:
        output_dir = self._tmp_dir()
        manifest_dir = output_dir / "demo-manifest"
        manifest_dir.mkdir(parents=True)
        (manifest_dir / "demo_manifest.json").write_text("{}", encoding="utf-8")
        (manifest_dir / "demo_manifest.md").write_text("# Demo\n", encoding="utf-8")

        report = release_readiness_pack.build_readiness(
            args_for(output_dir, allow_dirty=False),
            release_gate_report=release_gate_report(),
            security_report=security_report(),
            git_info=git_info(dirty=False),
            demo_step={"name": "demo_manifest", "ok": True},
            demo_manifest=demo_manifest(),
        )

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["schema"], "release_readiness_v1")
        self.assertIn("release_ready", report["release_status"]["diagnosis_codes"])
        self.assertIn("runtime_report_missing", report["release_status"]["warnings"])
        self.assertEqual(report["git"]["remote_origin"], "https://<redacted>@github.com/Ffffffffchopin/CrowdTensor.git")
        self.assertTrue((output_dir / "release_readiness.json").is_file())
        self.assertTrue((output_dir / "release_readiness.md").is_file())
        self.assertTrue(report["artifacts"]["demo_manifest_json"]["present"])

    def test_dirty_tree_blocks_without_allow_dirty(self) -> None:
        output_dir = self._tmp_dir()
        report = release_readiness_pack.build_readiness(
            args_for(output_dir, allow_dirty=False),
            release_gate_report=release_gate_report(),
            security_report=security_report(),
            git_info=git_info(dirty=True),
            demo_step={"name": "demo_manifest", "ok": True},
            demo_manifest=demo_manifest(),
        )

        self.assertFalse(report["ok"])
        self.assertIn("git_dirty", report["release_status"]["diagnosis_codes"])
        self.assertIn("git worktree is dirty", report["release_status"]["blocking_reasons"])

    def test_dirty_tree_can_be_allowed_for_development_smoke(self) -> None:
        output_dir = self._tmp_dir()
        report = release_readiness_pack.build_readiness(
            args_for(output_dir, allow_dirty=True),
            release_gate_report=release_gate_report(),
            security_report=security_report(),
            git_info=git_info(dirty=True),
            demo_step={"name": "demo_manifest", "ok": True},
            demo_manifest=demo_manifest(),
        )

        self.assertTrue(report["ok"], report)
        self.assertIn("release_ready", report["release_status"]["diagnosis_codes"])

    def test_release_gate_and_demo_manifest_failures_are_blockers(self) -> None:
        output_dir = self._tmp_dir()
        report = release_readiness_pack.build_readiness(
            args_for(output_dir, allow_dirty=True),
            release_gate_report=release_gate_report(ok=False),
            security_report=security_report(),
            git_info=git_info(dirty=False),
            demo_step={"name": "demo_manifest", "ok": False},
            demo_manifest=demo_manifest(ok=False),
        )

        self.assertFalse(report["ok"])
        self.assertIn("release_gate_failed", report["release_status"]["diagnosis_codes"])
        self.assertIn("demo_manifest_failed", report["release_status"]["diagnosis_codes"])

    def test_runner_invokes_demo_manifest_pack_when_not_injected(self) -> None:
        output_dir = self._tmp_dir()

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            self.assertIn("demo_manifest_pack.py", command[1])
            manifest_dir = output_dir / "demo-manifest"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            (manifest_dir / "demo_manifest.json").write_text("{}", encoding="utf-8")
            (manifest_dir / "demo_manifest.md").write_text("# Demo\n", encoding="utf-8")
            return completed(demo_manifest())

        report = release_readiness_pack.build_readiness(
            args_for(output_dir, allow_dirty=True),
            runner=fake_runner,
            release_gate_report=release_gate_report(),
            security_report=security_report(),
            git_info=git_info(dirty=False),
        )

        self.assertTrue(report["ok"], report)
        self.assertEqual(report["reports"]["demo_manifest"]["summary"]["schema"], "demo_manifest_v1")


if __name__ == "__main__":
    unittest.main()
