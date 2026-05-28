from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import public_swarm_inference_beta_rc_check as check
from scripts import public_swarm_inference_beta_rc_pack as pack


class PublicSwarmInferenceBetaRcPackTests(unittest.TestCase):
    def _tmp_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="crowdtensor_public_swarm_beta_rc_test_"))

    def test_check_builds_ready_local_loopback_rc_with_fake_runner(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "local-loopback",
            "--output-dir",
            str(self._tmp_dir()),
            "--max-new-tokens",
            "2",
        ]))

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["schema"], "public_swarm_inference_beta_rc_check_v1")

    def test_check_builds_ready_kaggle_package_rc_with_fake_runner(self) -> None:
        result = check.run_check(check.parse_args([
            "--mode",
            "package",
            "--target",
            "kaggle",
            "--output-dir",
            str(self._tmp_dir()),
        ]))

        self.assertTrue(result["ok"], result)

    def test_external_existing_requires_auth(self) -> None:
        with self.assertRaises(SystemExit):
            pack.parse_args([
                "external-existing",
                "--coordinator-url",
                "http://127.0.0.1:9999",
            ])

    def test_common_report_blocks_when_generate_loop_missing(self) -> None:
        output_dir = self._tmp_dir()

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            return check.fake_runner(command, **_)

        args = pack.parse_args([
            "local-loopback",
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            "2",
        ])
        product_step, product_payload = pack.run_product_beta(args, output_dir=output_dir / "product-beta", runner=fake_runner)
        p2p_step, p2p_payload = pack.run_p2p_route(args, output_dir=output_dir / "p2p-lite", runner=fake_runner)
        cpu_step, cpu_payload = pack.run_cpu_fallback(args, output_dir=output_dir / "cpu-fallback", runner=fake_runner)

        report = pack.build_common_report(
            args,
            output_dir=output_dir,
            product_step=product_step,
            product_payload=product_payload,
            p2p_step=p2p_step,
            p2p_payload=p2p_payload,
            cpu_step=cpu_step,
            cpu_payload=cpu_payload,
            mode_body={"ok": False, "diagnosis_codes": ["generation_timeout"]},
            mode_steps=[{"name": "serve_join_generate_loop", "ok": False}],
        )

        self.assertFalse(report["ok"], report)
        self.assertIn("public_swarm_inference_beta_rc_blocked", report["diagnosis_codes"])
        self.assertIn("generation_timeout", report["diagnosis_codes"])

    def test_report_redacts_secret_fragments(self) -> None:
        output_dir = self._tmp_dir()
        args = pack.parse_args(["package", "--output-dir", str(output_dir)])
        report = pack.persist_report(
            {
                "schema": pack.SCHEMA,
                "ok": True,
                "mode": "package",
                "diagnosis_codes": ["public_swarm_inference_beta_rc_ready"],
                "rc": {"ready": True},
                "steps": [],
                "limitations": [],
                "safety": {},
                "secret": "admin-secret",
            },
            output_dir=output_dir,
            secret_values=["admin-secret"],
        )

        encoded = json.dumps(report, sort_keys=True)
        self.assertNotIn("admin-secret", encoded)


if __name__ == "__main__":
    unittest.main()
