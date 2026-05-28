from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import public_swarm_product_rc_check as check
from scripts import public_swarm_product_rc_pack as pack


class PublicSwarmProductRcPackTests(unittest.TestCase):
    def test_check_builds_ready_rc_from_synthetic_evidence(self) -> None:
        result = check.run_check(check.parse_args([
            "--output-dir",
            tempfile.mkdtemp(prefix="crowdtensor_product_rc_check_test_"),
            "--max-new-tokens",
            "4",
        ]))

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["schema"], "public_swarm_product_rc_check_v1")

    def test_pack_blocks_missing_gpu_evidence(self) -> None:
        output_dir = Path(tempfile.mkdtemp(prefix="crowdtensor_product_rc_missing_gpu_test_"))

        def fake_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            if "session_protocol_check.py" in " ".join(command):
                return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"ok": True, "schema": "session_protocol_check_v1"}) + "\n", stderr="")
            if "p2p_lite_discovery_check.py" in " ".join(command):
                return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"ok": True, "schema": "p2p_lite_discovery_check_v1"}) + "\n", stderr="")
            raise AssertionError(command)

        report = pack.build_report(pack.parse_args([
            "--output-dir",
            str(output_dir),
            "--gpu-report",
            str(output_dir / "missing.json"),
            "--max-new-tokens",
            "4",
        ]), runner=fake_runner)

        self.assertFalse(report["ok"])
        self.assertIn("gpu_generation_evidence_import_blocked", report["diagnosis_codes"])


if __name__ == "__main__":
    unittest.main()
