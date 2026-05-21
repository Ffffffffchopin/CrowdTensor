from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "remote_miner_readiness_check.py"
SPEC = importlib.util.spec_from_file_location("remote_miner_readiness_check", SCRIPT_PATH)
remote_miner_readiness_check = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(remote_miner_readiness_check)


class RemoteMinerReadinessTests(unittest.TestCase):
    def test_default_python_workload_set_includes_model_bundle(self) -> None:
        self.assertEqual(
            remote_miner_readiness_check.WORKLOADS,
            {"diloco_train", "cpu_lora_mock", "micro_transformer_lm", "model_bundle_lm"},
        )

    def test_default_max_tasks_matches_workload_count(self) -> None:
        args = remote_miner_readiness_check.parse_args([])

        self.assertEqual(args.max_tasks, len(remote_miner_readiness_check.WORKLOADS))


if __name__ == "__main__":
    unittest.main()
