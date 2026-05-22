from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "runtime_matrix.py"
SPEC = importlib.util.spec_from_file_location("runtime_matrix", SCRIPT_PATH)
runtime_matrix = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runtime_matrix)


def workload_by_name(matrix: dict, name: str) -> dict:
    return next(row for row in matrix["workloads"] if row["name"] == name)


class RuntimeMatrixTests(unittest.TestCase):
    def test_cpu_baseline_is_available_without_optional_runtimes(self) -> None:
        matrix = runtime_matrix.build_matrix(root=ROOT, env={})

        self.assertTrue(matrix["ok"], matrix)
        self.assertEqual(workload_by_name(matrix, "diloco_train")["status"], "available")
        self.assertEqual(workload_by_name(matrix, "model_bundle_infer")["status"], "available")
        self.assertEqual(workload_by_name(matrix, "external_llm_infer_mock")["status"], "available")
        self.assertEqual(workload_by_name(matrix, "external_llm_infer_http")["status"], "optional_missing")
        self.assertNotIn("CROWDTENSOR_LLM_RUNTIME_API_KEY=", json.dumps(matrix, sort_keys=True))

    def test_http_runtime_configuration_is_reported_without_secret_value(self) -> None:
        matrix = runtime_matrix.build_matrix(
            root=ROOT,
            env={
                "CROWDTENSOR_LLM_RUNTIME_URL": "http://127.0.0.1:11434/v1/chat/completions",
                "CROWDTENSOR_LLM_RUNTIME_API_KEY": "super-secret-key",
            },
        )

        self.assertEqual(workload_by_name(matrix, "external_llm_infer_http")["status"], "configured")
        self.assertTrue(matrix["configured_runtimes"]["external_llm_http"]["api_key_configured"])
        self.assertNotIn("super-secret-key", json.dumps(matrix, sort_keys=True))

    def test_missing_project_files_block_required_workloads(self) -> None:
        with patch.object(runtime_matrix, "module_available", return_value=True):
            matrix = runtime_matrix.build_matrix(root=Path("/tmp/crowdtensor-missing-root"), env={})

        self.assertFalse(matrix["ok"])
        self.assertEqual(workload_by_name(matrix, "diloco_train")["status"], "blocked")
        self.assertIn("diloco_train", matrix["summary"]["blocked_workloads"])

    def test_cli_json_outputs_machine_readable_matrix(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--root", str(ROOT), "--json"],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["ok"], payload)
        self.assertIn("recommended_next_commands", payload)


if __name__ == "__main__":
    unittest.main()
