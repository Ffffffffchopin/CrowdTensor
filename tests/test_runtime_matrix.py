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


def target_by_name(matrix: dict, name: str) -> dict:
    return next(row for row in matrix["hardware_targets"] if row["name"] == name)


def route_by_name(matrix: dict, name: str) -> dict:
    return next(row for row in matrix["recommended_routes"] if row["name"] == name)


class RuntimeMatrixTests(unittest.TestCase):
    def test_cpu_baseline_is_available_without_optional_runtimes(self) -> None:
        matrix = runtime_matrix.build_matrix(root=ROOT, env={})

        self.assertTrue(matrix["ok"], matrix)
        self.assertEqual(workload_by_name(matrix, "diloco_train")["status"], "available")
        self.assertEqual(workload_by_name(matrix, "model_bundle_infer")["status"], "available")
        self.assertEqual(workload_by_name(matrix, "external_llm_infer_mock")["status"], "available")
        self.assertEqual(workload_by_name(matrix, "external_llm_infer_http")["status"], "optional_missing")
        self.assertEqual(target_by_name(matrix, "cpu_baseline")["status"], "available")
        self.assertTrue(target_by_name(matrix, "cpu_baseline")["usable_now"])
        self.assertFalse(target_by_name(matrix, "nvidia_cuda")["usable_now"])
        self.assertEqual(route_by_name(matrix, "local_cpu_model_bundle_infer")["status"], "available")
        self.assertTrue(route_by_name(matrix, "local_cpu_model_bundle_infer")["usable_now"])
        self.assertEqual(route_by_name(matrix, "local_cpu_model_bundle_infer")["confidence"], "ready")
        self.assertIn("python_runtime", route_by_name(matrix, "local_cpu_model_bundle_infer")["matched_capabilities"])
        self.assertEqual(route_by_name(matrix, "local_cpu_model_bundle_infer")["missing_capabilities"], [])
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
        self.assertEqual(target_by_name(matrix, "external_llm_http")["status"], "configured")
        self.assertTrue(target_by_name(matrix, "external_llm_http")["usable_now"])
        self.assertEqual(route_by_name(matrix, "external_llm_http_adapter")["status"], "configured")
        self.assertTrue(route_by_name(matrix, "external_llm_http_adapter")["usable_now"])
        self.assertEqual(route_by_name(matrix, "external_llm_http_adapter")["confidence"], "configured")
        self.assertIn("target:external_llm_http", route_by_name(matrix, "external_llm_http_adapter")["matched_capabilities"])
        self.assertTrue(matrix["configured_runtimes"]["external_llm_http"]["api_key_configured"])
        self.assertNotIn("super-secret-key", json.dumps(matrix, sort_keys=True))

    def test_missing_project_files_block_required_workloads(self) -> None:
        with patch.object(runtime_matrix, "module_available", return_value=True):
            matrix = runtime_matrix.build_matrix(root=Path("/tmp/crowdtensor-missing-root"), env={})

        self.assertFalse(matrix["ok"])
        self.assertEqual(workload_by_name(matrix, "diloco_train")["status"], "blocked")
        self.assertIn("diloco_train", matrix["summary"]["blocked_workloads"])
        self.assertEqual(target_by_name(matrix, "cpu_baseline")["status"], "blocked")
        self.assertFalse(route_by_name(matrix, "local_cpu_model_bundle_infer")["usable_now"])

    def test_hardware_targets_report_detected_future_paths(self) -> None:
        with patch.object(runtime_matrix, "executable_available", side_effect=lambda name: name in {"nvidia-smi", "rocminfo"}):
            with patch.object(runtime_matrix.platform, "system", return_value="Darwin"):
                with patch.object(runtime_matrix.platform, "machine", return_value="arm64"):
                    matrix = runtime_matrix.build_matrix(
                        root=ROOT,
                        env={"KAGGLE_KERNEL_RUN_TYPE": "Interactive"},
                    )

        self.assertEqual(target_by_name(matrix, "nvidia_cuda")["status"], "detected")
        self.assertEqual(target_by_name(matrix, "amd_rocm")["status"], "detected")
        self.assertEqual(target_by_name(matrix, "apple_metal")["status"], "detected")
        self.assertEqual(target_by_name(matrix, "remote_container")["status"], "detected")
        self.assertFalse(target_by_name(matrix, "nvidia_cuda")["usable_now"])

        route = route_by_name(matrix, "browser_probe")
        self.assertIn(route["confidence"], {"ready", "future"})
        self.assertIn("reason", route)
        self.assertIn("matched_capabilities", route)
        self.assertIn("missing_capabilities", route)

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
        self.assertIn("hardware_targets", payload)
        self.assertIn("recommended_routes", payload)


if __name__ == "__main__":
    unittest.main()
