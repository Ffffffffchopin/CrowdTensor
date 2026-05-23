from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "runtime_acceptance_pack.py"
SPEC = importlib.util.spec_from_file_location("runtime_acceptance_pack", SCRIPT_PATH)
runtime_acceptance_pack = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runtime_acceptance_pack)

DEFAULT_CHECK_NAMES = [
    "readiness",
    "runtime_matrix",
    "home_compute_demo",
    "home_compute_evidence",
    "api_contract",
    "chaos",
    "trust_quarantine",
    "replay_audit",
    "operator_control",
    "micro_transformer",
    "model_bundle",
    "result_idempotency",
    "result_ledger",
    "miner_resilience",
    "miner_auth",
    "observer_auth",
    "miner_registry_auth",
    "token_hash_auth",
    "outer_optimizer",
    "compressed_error_feedback",
    "delta_transport_negotiation",
    "model_bundle_inference",
    "inference_session_demo",
    "inference_session_client",
    "external_llm_inference",
    "external_llm_http_adapter",
    "external_llm_evidence",
    "admin_inference_session",
]

DEFAULT_PORTS = [
    9010,
    9044,
    9045,
    9046,
    9011,
    9012,
    9013,
    9014,
    9015,
    9016,
    9017,
    9018,
    9019,
    9020,
    9021,
    9022,
    9023,
    9024,
    9025,
    9026,
    9027,
    9040,
    9041,
    9049,
    9042,
    9043,
    9052,
    9048,
]

TOKEN_AWARE_NAMES = {
    "chaos",
    "trust_quarantine",
    "replay_audit",
    "operator_control",
    "micro_transformer",
    "model_bundle",
    "home_compute_demo",
    "home_compute_evidence",
    "remote_compute_evidence",
    "multi_miner_scenario_sweep",
    "multi_miner_requeue",
    "model_bundle_inference",
    "inference_session_demo",
    "external_llm_inference",
    "external_llm_http_adapter",
    "external_llm_evidence",
}


def acceptance_args(**overrides):
    values = {
        "host": "127.0.0.1",
        "base_port": 9010,
        "admin_token": "admin-test",
        "miner_token": "",
        "observer_token": "",
        "include_browser": False,
        "include_browser_chaos": False,
        "include_micro_transformer": False,
        "include_remote_miner": False,
        "include_remote_evidence": False,
        "include_multi_miner_sweep": False,
        "include_multi_miner_requeue": False,
        "browser": "",
        "headful": False,
        "browser_timeout": 20.0,
        "skip_readiness": False,
        "skip_runtime_matrix": False,
        "skip_home_compute_demo": False,
        "skip_home_compute_evidence": False,
        "skip_api_contract": False,
        "skip_chaos": False,
        "skip_trust": False,
        "skip_replay_audit": False,
        "skip_operator": False,
        "skip_micro_transformer": False,
        "skip_model_bundle": False,
        "skip_model_bundle_inference": False,
        "skip_inference_session_demo": False,
        "skip_inference_session_client": False,
        "skip_admin_inference_session": False,
        "skip_external_llm_inference": False,
        "skip_external_llm_http_adapter": False,
        "skip_external_llm_evidence": False,
        "skip_result_idempotency": False,
        "skip_result_ledger": False,
        "skip_miner_resilience": False,
        "skip_miner_auth": False,
        "skip_observer_auth": False,
        "skip_miner_registry_auth": False,
        "skip_token_hash_auth": False,
        "skip_outer_optimizer": False,
        "skip_compressed_error_feedback": False,
        "skip_delta_transport_negotiation": False,
        "skip_remote_miner": False,
        "skip_remote_evidence": False,
        "skip_multi_miner_sweep": False,
        "skip_multi_miner_requeue": False,
        "skip_webrtc": False,
        "skip_runtime_contract": False,
        "skip_browser_miner": False,
        "skip_browser_probe": False,
        "skip_capability_ledger": False,
        "skip_browser_chaos": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class RuntimeAcceptancePackTests(unittest.TestCase):
    def test_build_report_aggregates_check_statuses(self) -> None:
        passing = runtime_acceptance_pack.build_report(
            "2026-05-19T00:00:00+00:00",
            "2026-05-19T00:00:01+00:00",
            1.2345,
            [{"name": "chaos", "ok": True}, {"name": "trust", "ok": True}],
        )
        failing = runtime_acceptance_pack.build_report(
            "2026-05-19T00:00:00+00:00",
            "2026-05-19T00:00:01+00:00",
            1.0,
            [{"name": "chaos", "ok": True}, {"name": "operator", "ok": False}],
        )

        self.assertTrue(passing["ok"])
        self.assertEqual(passing["duration_seconds"], 1.234)
        self.assertFalse(failing["ok"])
        self.assertEqual([check["name"] for check in failing["checks"]], ["chaos", "operator"])
        self.assertEqual(failing["diagnosis_summary"]["failed_checks"], ["operator"])

    def test_build_report_aggregates_diagnosis_codes(self) -> None:
        report = runtime_acceptance_pack.build_report(
            "2026-05-19T00:00:00+00:00",
            "2026-05-19T00:00:01+00:00",
            1.0,
            [
                {"name": "home_compute_demo", "ok": True, "diagnosis_codes": ["home_compute_ready"]},
                {"name": "remote_demo", "ok": False, "diagnosis_codes": ["coordinator_unreachable"]},
            ],
        )

        self.assertEqual(
            report["diagnosis_summary"],
            {
                "codes": ["coordinator_unreachable", "home_compute_ready"],
                "by_check": {
                    "home_compute_demo": ["home_compute_ready"],
                    "remote_demo": ["coordinator_unreachable"],
                },
                "failed_checks": ["remote_demo"],
            },
        )

    def test_parse_summary_json_keeps_safe_fields_and_diagnosis(self) -> None:
        summary = runtime_acceptance_pack.parse_summary_json(
            'setup line\n{"ok": true, "schema": "home_compute_evidence_v1", '
            '"route": "local_cpu_model_bundle_infer", "request_count": 4, '
            '"diagnosis_codes": ["home_compute_ready"], "token": "secret", "weights": [1, 2]}\n'
        )

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["schema"], "home_compute_evidence_v1")
        self.assertEqual(summary["route"], "local_cpu_model_bundle_infer")
        self.assertEqual(summary["request_count"], 4)
        self.assertEqual(summary["diagnosis_codes"], ["home_compute_ready"])
        self.assertNotIn("token", summary)
        self.assertNotIn("weights", summary)

    def test_parse_summary_json_keeps_external_llm_safe_fields(self) -> None:
        summary = runtime_acceptance_pack.parse_summary_json(
            'setup line\n{"ok": true, "schema": "external_llm_evidence_v1", '
            '"route": "local_external_llm_infer", "adapter_kind": "mock", '
            '"request_count": 3, "completion_count": 3, "output_chars": 128, '
            '"diagnosis_codes": ["external_llm_evidence_ready"], '
            '"output_text": "raw", "llm_runtime_api_key": "secret"}\n'
        )

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["schema"], "external_llm_evidence_v1")
        self.assertEqual(summary["route"], "local_external_llm_infer")
        self.assertEqual(summary["adapter_kind"], "mock")
        self.assertEqual(summary["request_count"], 3)
        self.assertEqual(summary["completion_count"], 3)
        self.assertEqual(summary["output_chars"], 128)
        self.assertEqual(summary["diagnosis_codes"], ["external_llm_evidence_ready"])
        self.assertNotIn("output_text", summary)
        self.assertNotIn("llm_runtime_api_key", summary)

    def test_parse_summary_json_ignores_non_json_stdout(self) -> None:
        self.assertIsNone(runtime_acceptance_pack.parse_summary_json("plain success\n"))

    def test_run_check_extracts_summary_json_and_diagnosis_codes(self) -> None:
        completed = SimpleNamespace(
            returncode=0,
            stdout='{"ok": true, "schema": "home_compute_demo_v1", "route": "local_cpu_model_bundle_infer", "diagnosis": "home_compute_ready", "request_count": 4}\n',
            stderr="",
        )

        with TemporaryDirectory(prefix="crowdtensor_runtime_acceptance_") as tmp_root:
            check = {
                "name": "home_compute_demo",
                "port": 9910,
                "state_dir": str(Path(tmp_root) / "state"),
                "command": ["fake-check"],
            }
            with patch.object(runtime_acceptance_pack.subprocess, "run", return_value=completed):
                result = runtime_acceptance_pack.run_check(check, timeout_seconds=1.0)

        self.assertTrue(result["ok"])
        self.assertEqual(result["summary_json"]["schema"], "home_compute_demo_v1")
        self.assertEqual(result["route"], "local_cpu_model_bundle_infer")
        self.assertEqual(result["request_count"], 4)
        self.assertEqual(result["diagnosis_codes"], ["home_compute_ready"])

    def test_run_check_extracts_failed_check_diagnosis_codes(self) -> None:
        completed = SimpleNamespace(
            returncode=1,
            stdout='{"ok": false, "diagnosis": {"primary_code": "coordinator_unreachable", "severity": "error"}}\n',
            stderr="failed",
        )

        with TemporaryDirectory(prefix="crowdtensor_runtime_acceptance_") as tmp_root:
            check = {
                "name": "remote_demo_acceptance",
                "port": 9911,
                "state_dir": str(Path(tmp_root) / "state"),
                "command": ["fake-check"],
            }
            with patch.object(runtime_acceptance_pack.subprocess, "run", return_value=completed):
                result = runtime_acceptance_pack.run_check(check, timeout_seconds=1.0)

        self.assertFalse(result["ok"])
        self.assertEqual(result["diagnosis_codes"], ["coordinator_unreachable"])

    def test_selected_checks_respects_skips_and_ports(self) -> None:
        args = acceptance_args(skip_trust=True)

        checks = runtime_acceptance_pack.selected_checks(args, Path("/tmp/crowdtensor_acceptance_test"))

        self.assertEqual([check["name"] for check in checks], [name for name in DEFAULT_CHECK_NAMES if name != "trust_quarantine"])
        self.assertEqual([check["port"] for check in checks], [port for name, port in zip(DEFAULT_CHECK_NAMES, DEFAULT_PORTS) if name != "trust_quarantine"])
        operator = next(check for check in checks if check["name"] == "operator_control")
        self.assertIn("--admin-token", operator["command"])
        self.assertIn("admin-test", operator["command"])
        self.assertTrue(all(check.get("miner_token") == "" for check in checks))
        self.assertTrue(all(check.get("observer_token") == "" for check in checks))

    def test_browser_checks_are_opt_in(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        self.assertEqual([check["name"] for check in checks], DEFAULT_CHECK_NAMES)

    def test_readiness_check_is_default_and_skippable(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        readiness = next(check for check in checks if check["name"] == "readiness")
        self.assertEqual(readiness["port"], 9010)
        self.assertIn("readiness_check.py", readiness["command"][1])

        skipped = runtime_acceptance_pack.selected_checks(
            acceptance_args(skip_readiness=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )
        self.assertNotIn("readiness", [check["name"] for check in skipped])

    def test_runtime_matrix_check_is_default_and_skippable(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        matrix = next(check for check in checks if check["name"] == "runtime_matrix")
        self.assertEqual(matrix["port"], 9044)
        self.assertIn("runtime_matrix_check.py", matrix["command"][1])

        skipped = runtime_acceptance_pack.selected_checks(
            acceptance_args(skip_runtime_matrix=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )
        self.assertNotIn("runtime_matrix", [check["name"] for check in skipped])

    def test_api_contract_check_is_default_and_skippable(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        api_contract = next(check for check in checks if check["name"] == "api_contract")
        self.assertEqual(api_contract["port"], 9011)
        self.assertIn("api_contract_check.py", api_contract["command"][1])

        skipped = runtime_acceptance_pack.selected_checks(
            acceptance_args(skip_api_contract=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )
        self.assertNotIn("api_contract", [check["name"] for check in skipped])

    def test_home_compute_demo_check_is_default_and_skippable(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        demo = next(check for check in checks if check["name"] == "home_compute_demo")
        self.assertEqual(demo["port"], 9045)
        self.assertIn("home_compute_demo_check.py", demo["command"][1])
        self.assertIn("--admin-token", demo["command"])
        self.assertIn("admin-test", demo["command"])

        skipped = runtime_acceptance_pack.selected_checks(
            acceptance_args(skip_home_compute_demo=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )
        self.assertNotIn("home_compute_demo", [check["name"] for check in skipped])

    def test_home_compute_evidence_check_is_default_and_skippable(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        evidence = next(check for check in checks if check["name"] == "home_compute_evidence")
        self.assertEqual(evidence["port"], 9046)
        self.assertIn("home_compute_evidence_check.py", evidence["command"][1])
        self.assertIn("--admin-token", evidence["command"])
        self.assertIn("admin-test", evidence["command"])

        skipped = runtime_acceptance_pack.selected_checks(
            acceptance_args(skip_home_compute_evidence=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )
        self.assertNotIn("home_compute_evidence", [check["name"] for check in skipped])

    def test_micro_transformer_check_is_default_and_legacy_include_is_noop(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(include_micro_transformer=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        self.assertEqual([check["name"] for check in checks], DEFAULT_CHECK_NAMES)
        micro = next(check for check in checks if check["name"] == "micro_transformer")
        self.assertEqual(micro["port"], 9016)
        self.assertIn("micro_transformer_smoke.py", micro["command"][1])

    def test_micro_transformer_skip_removes_default_check(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(skip_micro_transformer=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        self.assertEqual([check["name"] for check in checks], [name for name in DEFAULT_CHECK_NAMES if name != "micro_transformer"])


    def test_model_bundle_check_is_default_and_skippable(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        model_bundle = next(check for check in checks if check["name"] == "model_bundle")
        self.assertEqual(model_bundle["port"], 9017)
        self.assertIn("model_bundle_smoke.py", model_bundle["command"][1])

        skipped = runtime_acceptance_pack.selected_checks(
            acceptance_args(skip_model_bundle=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )
        self.assertNotIn("model_bundle", [check["name"] for check in skipped])

    def test_model_bundle_inference_check_is_default_and_skippable(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        inference = next(check for check in checks if check["name"] == "model_bundle_inference")
        self.assertEqual(inference["port"], 9040)
        self.assertIn("model_bundle_inference_smoke.py", inference["command"][1])

        skipped = runtime_acceptance_pack.selected_checks(
            acceptance_args(skip_model_bundle_inference=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )
        self.assertNotIn("model_bundle_inference", [check["name"] for check in skipped])

    def test_inference_session_demo_check_is_default_and_skippable(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        demo = next(check for check in checks if check["name"] == "inference_session_demo")
        self.assertEqual(demo["port"], 9041)
        self.assertIn("inference_session_demo.py", demo["command"][1])
        self.assertIn("--json", demo["command"])

        skipped = runtime_acceptance_pack.selected_checks(
            acceptance_args(skip_inference_session_demo=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )
        self.assertNotIn("inference_session_demo", [check["name"] for check in skipped])

    def test_inference_session_client_check_is_default_and_skippable(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        client = next(check for check in checks if check["name"] == "inference_session_client")
        self.assertEqual(client["port"], 9049)
        self.assertIn("inference_session_client_check.py", client["command"][1])

        skipped = runtime_acceptance_pack.selected_checks(
            acceptance_args(skip_inference_session_client=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )
        self.assertNotIn("inference_session_client", [check["name"] for check in skipped])

    def test_admin_inference_session_check_is_default_and_skippable(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        admin_session = next(check for check in checks if check["name"] == "admin_inference_session")
        self.assertEqual(admin_session["port"], 9048)
        self.assertIn("admin_inference_session_check.py", admin_session["command"][1])
        self.assertIn("--admin-token", admin_session["command"])
        self.assertIn("admin-test", admin_session["command"])

        skipped = runtime_acceptance_pack.selected_checks(
            acceptance_args(skip_admin_inference_session=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )
        self.assertNotIn("admin_inference_session", [check["name"] for check in skipped])

    def test_external_llm_inference_check_is_default_and_skippable(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        external = next(check for check in checks if check["name"] == "external_llm_inference")
        self.assertEqual(external["port"], 9042)
        self.assertIn("external_llm_inference_smoke.py", external["command"][1])

        skipped = runtime_acceptance_pack.selected_checks(
            acceptance_args(skip_external_llm_inference=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )
        self.assertNotIn("external_llm_inference", [check["name"] for check in skipped])

    def test_external_llm_http_adapter_check_is_default_and_skippable(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        external_http = next(check for check in checks if check["name"] == "external_llm_http_adapter")
        self.assertEqual(external_http["port"], 9043)
        self.assertIn("external_llm_http_adapter_smoke.py", external_http["command"][1])

        skipped = runtime_acceptance_pack.selected_checks(
            acceptance_args(skip_external_llm_http_adapter=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )
        self.assertNotIn("external_llm_http_adapter", [check["name"] for check in skipped])

    def test_external_llm_evidence_check_is_default_and_skippable(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        evidence = next(check for check in checks if check["name"] == "external_llm_evidence")
        self.assertEqual(evidence["port"], 9052)
        self.assertIn("external_llm_evidence_check.py", evidence["command"][1])

        skipped = runtime_acceptance_pack.selected_checks(
            acceptance_args(skip_external_llm_evidence=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )
        self.assertNotIn("external_llm_evidence", [check["name"] for check in skipped])

    def test_result_idempotency_check_is_default_and_skippable(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        result_idempotency = next(check for check in checks if check["name"] == "result_idempotency")
        self.assertEqual(result_idempotency["port"], 9018)
        self.assertIn("result_idempotency_check.py", result_idempotency["command"][1])

        skipped = runtime_acceptance_pack.selected_checks(
            acceptance_args(skip_result_idempotency=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )
        self.assertNotIn("result_idempotency", [check["name"] for check in skipped])

    def test_result_ledger_check_is_default_and_skippable(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        result_ledger = next(check for check in checks if check["name"] == "result_ledger")
        self.assertEqual(result_ledger["port"], 9019)
        self.assertIn("result_ledger_check.py", result_ledger["command"][1])

        skipped = runtime_acceptance_pack.selected_checks(
            acceptance_args(skip_result_ledger=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )
        self.assertNotIn("result_ledger", [check["name"] for check in skipped])

    def test_miner_resilience_check_is_default_and_skippable(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        resilience = next(check for check in checks if check["name"] == "miner_resilience")
        self.assertEqual(resilience["port"], 9020)
        self.assertIn("miner_resilience_check.py", resilience["command"][1])

        skipped = runtime_acceptance_pack.selected_checks(
            acceptance_args(skip_miner_resilience=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )
        self.assertNotIn("miner_resilience", [check["name"] for check in skipped])

    def test_miner_auth_check_is_default_and_skippable(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        miner_auth = next(check for check in checks if check["name"] == "miner_auth")
        self.assertEqual(miner_auth["port"], 9021)
        self.assertIn("miner_auth_check.py", miner_auth["command"][1])
        self.assertNotIn("miner-test", miner_auth["command"])

        skipped = runtime_acceptance_pack.selected_checks(
            acceptance_args(skip_miner_auth=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )
        self.assertNotIn("miner_auth", [check["name"] for check in skipped])

    def test_explicit_miner_token_is_not_written_into_commands(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(miner_token="miner-test"),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        for check in checks:
            expected = "miner-test" if check["name"] in TOKEN_AWARE_NAMES else ""
            self.assertEqual(check.get("miner_token"), expected)
        self.assertTrue(all("miner-test" not in check["command"] for check in checks))

    def test_observer_auth_check_is_default_and_skippable(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        observer_auth = next(check for check in checks if check["name"] == "observer_auth")
        self.assertEqual(observer_auth["port"], 9022)
        self.assertIn("observer_auth_check.py", observer_auth["command"][1])

        skipped = runtime_acceptance_pack.selected_checks(
            acceptance_args(skip_observer_auth=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )
        self.assertNotIn("observer_auth", [check["name"] for check in skipped])

    def test_miner_registry_auth_check_is_default_and_skippable(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        registry_auth = next(check for check in checks if check["name"] == "miner_registry_auth")
        self.assertEqual(registry_auth["port"], 9023)
        self.assertIn("miner_registry_auth_check.py", registry_auth["command"][1])

        skipped = runtime_acceptance_pack.selected_checks(
            acceptance_args(skip_miner_registry_auth=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )
        self.assertNotIn("miner_registry_auth", [check["name"] for check in skipped])

    def test_token_hash_auth_check_is_default_and_skippable(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        hash_auth = next(check for check in checks if check["name"] == "token_hash_auth")
        self.assertEqual(hash_auth["port"], 9024)
        self.assertIn("token_hash_auth_check.py", hash_auth["command"][1])

        skipped = runtime_acceptance_pack.selected_checks(
            acceptance_args(skip_token_hash_auth=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )
        self.assertNotIn("token_hash_auth", [check["name"] for check in skipped])

    def test_outer_optimizer_check_is_default_and_skippable(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        outer_optimizer = next(check for check in checks if check["name"] == "outer_optimizer")
        self.assertEqual(outer_optimizer["port"], 9025)
        self.assertIn("outer_optimizer_check.py", outer_optimizer["command"][1])

        skipped = runtime_acceptance_pack.selected_checks(
            acceptance_args(skip_outer_optimizer=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )
        self.assertNotIn("outer_optimizer", [check["name"] for check in skipped])

    def test_compressed_error_feedback_check_is_default_and_skippable(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        compressed_error_feedback = next(check for check in checks if check["name"] == "compressed_error_feedback")
        self.assertEqual(compressed_error_feedback["port"], 9026)
        self.assertIn("compressed_error_feedback_check.py", compressed_error_feedback["command"][1])

        skipped = runtime_acceptance_pack.selected_checks(
            acceptance_args(skip_compressed_error_feedback=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )
        self.assertNotIn("compressed_error_feedback", [check["name"] for check in skipped])

    def test_delta_transport_negotiation_check_is_default_and_skippable(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        delta_transport = next(check for check in checks if check["name"] == "delta_transport_negotiation")
        self.assertEqual(delta_transport["port"], 9027)
        self.assertIn("delta_transport_negotiation_check.py", delta_transport["command"][1])

        skipped = runtime_acceptance_pack.selected_checks(
            acceptance_args(skip_delta_transport_negotiation=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )
        self.assertNotIn("delta_transport_negotiation", [check["name"] for check in skipped])

    def test_explicit_observer_token_is_not_written_into_commands(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(observer_token="observer-test"),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        for check in checks:
            expected = "observer-test" if check["name"] in TOKEN_AWARE_NAMES else ""
            self.assertEqual(check.get("observer_token"), expected)
        self.assertTrue(all("observer-test" not in check["command"] for check in checks))

    def test_remote_miner_readiness_is_opt_in(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(include_remote_miner=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        self.assertEqual([check["name"] for check in checks], DEFAULT_CHECK_NAMES + ["remote_miner"])
        self.assertEqual(checks[-1]["port"], 9028)
        self.assertIn("remote_miner_readiness_check.py", checks[-1]["command"][1])

        authenticated = runtime_acceptance_pack.selected_checks(
            acceptance_args(include_remote_miner=True, miner_token="miner-test", observer_token="observer-test"),
            Path("/tmp/crowdtensor_acceptance_test"),
        )
        remote = next(check for check in authenticated if check["name"] == "remote_miner")
        self.assertEqual(remote.get("miner_token"), "miner-test")
        self.assertEqual(remote.get("observer_token"), "observer-test")
        self.assertTrue(all("miner-test" not in check["command"] for check in authenticated))
        self.assertTrue(all("observer-test" not in check["command"] for check in authenticated))

    def test_remote_miner_skip_removes_opt_in_check(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(include_remote_miner=True, skip_remote_miner=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        self.assertNotIn("remote_miner", [check["name"] for check in checks])

    def test_remote_compute_evidence_is_opt_in(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(include_remote_evidence=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        evidence = next(check for check in checks if check["name"] == "remote_compute_evidence")
        self.assertEqual(evidence["port"], 9047)
        self.assertIn("remote_compute_evidence_check.py", evidence["command"][1])

        authenticated = runtime_acceptance_pack.selected_checks(
            acceptance_args(include_remote_evidence=True, miner_token="miner-test", observer_token="observer-test"),
            Path("/tmp/crowdtensor_acceptance_test"),
        )
        remote = next(check for check in authenticated if check["name"] == "remote_compute_evidence")
        self.assertEqual(remote.get("miner_token"), "miner-test")
        self.assertEqual(remote.get("observer_token"), "observer-test")

    def test_remote_compute_evidence_skip_removes_opt_in_check(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(include_remote_evidence=True, skip_remote_evidence=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        self.assertNotIn("remote_compute_evidence", [check["name"] for check in checks])

    def test_multi_miner_scenario_sweep_is_opt_in(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(include_multi_miner_sweep=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        sweep = next(check for check in checks if check["name"] == "multi_miner_scenario_sweep")
        self.assertEqual(sweep["port"], 9050)
        self.assertIn("multi_miner_scenario_sweep_check.py", sweep["command"][1])
        self.assertIn("--execution-mode", sweep["command"])
        self.assertIn("concurrent", sweep["command"])
        self.assertIn("--json-out", sweep["command"])
        self.assertTrue(any(item.endswith("multi_miner_scenario_sweep.json") for item in sweep["command"]))

    def test_multi_miner_scenario_sweep_skip_removes_opt_in_check(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(include_multi_miner_sweep=True, skip_multi_miner_sweep=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        self.assertNotIn("multi_miner_scenario_sweep", [check["name"] for check in checks])

    def test_multi_miner_requeue_is_opt_in(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(include_multi_miner_requeue=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        requeue = next(check for check in checks if check["name"] == "multi_miner_requeue")
        self.assertEqual(requeue["port"], 9051)
        self.assertIn("multi_miner_scenario_sweep_check.py", requeue["command"][1])
        self.assertIn("--execution-mode", requeue["command"])
        self.assertIn("concurrent", requeue["command"])
        self.assertIn("--failure-mode", requeue["command"])
        self.assertIn("kill-after-claim", requeue["command"])
        self.assertIn("--json-out", requeue["command"])
        self.assertTrue(any(item.endswith("multi_miner_requeue.json") for item in requeue["command"]))

    def test_multi_miner_requeue_skip_removes_opt_in_check(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(include_multi_miner_requeue=True, skip_multi_miner_requeue=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        self.assertNotIn("multi_miner_requeue", [check["name"] for check in checks])

    def test_include_browser_adds_lightweight_browser_pack(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(include_browser=True, browser="/usr/bin/chromium", headful=True, browser_timeout=7.5),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        browser_checks = checks[len(DEFAULT_CHECK_NAMES):]
        self.assertEqual(
            [check["name"] for check in browser_checks],
            ["webrtc", "runtime_contract", "browser_miner", "browser_probe", "capability_ledger"],
        )
        self.assertEqual([check["port"] for check in browser_checks], [9029, 9030, 9032, 9034, 9036])
        self.assertEqual([check.get("web_port") for check in browser_checks], [None, 9031, 9033, 9035, 9037])
        for check in browser_checks:
            self.assertIn("--browser", check["command"])
            self.assertIn("/usr/bin/chromium", check["command"])
            self.assertIn("--headful", check["command"])
        self.assertIn("--timeout", browser_checks[0]["command"])
        self.assertIn("7.5", browser_checks[0]["command"])

    def test_browser_auth_tokens_only_target_token_aware_checks(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(include_browser=True, miner_token="miner-test", observer_token="observer-test"),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        token_aware_names = TOKEN_AWARE_NAMES | {
            "runtime_contract",
            "browser_miner",
            "browser_probe",
            "capability_ledger",
        }
        for check in checks:
            expected_miner = "miner-test" if check["name"] in token_aware_names else ""
            expected_observer = "observer-test" if check["name"] in token_aware_names else ""
            self.assertEqual(check.get("miner_token"), expected_miner, check["name"])
            self.assertEqual(check.get("observer_token"), expected_observer, check["name"])

    def test_include_browser_chaos_adds_full_browser_pack_tail(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(include_browser=True, include_browser_chaos=True, browser="/usr/bin/chromium"),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        browser_checks = checks[len(DEFAULT_CHECK_NAMES):]
        self.assertEqual(
            [check["name"] for check in browser_checks],
            ["webrtc", "runtime_contract", "browser_miner", "browser_probe", "capability_ledger", "browser_chaos"],
        )
        chaos = browser_checks[-1]
        self.assertEqual(chaos["port"], 9038)
        self.assertEqual(chaos["web_port"], 9039)
        self.assertIn("browser_miner_chaos.py", chaos["command"][1])
        self.assertIn("--coordinator-port", chaos["command"])
        self.assertIn("9038", chaos["command"])
        self.assertIn("--web-port", chaos["command"])
        self.assertIn("9039", chaos["command"])
        self.assertIn("--browser", chaos["command"])
        self.assertIn("/usr/bin/chromium", chaos["command"])

    def test_browser_chaos_skip_removes_full_browser_tail(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(include_browser=True, include_browser_chaos=True, skip_browser_chaos=True),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        self.assertNotIn("browser_chaos", [check["name"] for check in checks])

    def test_browser_skip_flags_remove_individual_checks(self) -> None:
        checks = runtime_acceptance_pack.selected_checks(
            acceptance_args(
                include_browser=True,
                skip_webrtc=True,
                skip_browser_probe=True,
                skip_capability_ledger=True,
            ),
            Path("/tmp/crowdtensor_acceptance_test"),
        )

        self.assertEqual(
            [check["name"] for check in checks],
            [
                "readiness",
                "runtime_matrix",
                "home_compute_demo",
                "home_compute_evidence",
                "api_contract",
                "chaos",
                "trust_quarantine",
                "replay_audit",
                "operator_control",
                "micro_transformer",
                "model_bundle",
                "result_idempotency",
                "result_ledger",
                "miner_resilience",
                "miner_auth",
                "observer_auth",
                "miner_registry_auth",
                "token_hash_auth",
                "outer_optimizer",
                "compressed_error_feedback",
                "delta_transport_negotiation",
                "model_bundle_inference",
                "inference_session_demo",
                "inference_session_client",
                "external_llm_inference",
                "external_llm_http_adapter",
                "external_llm_evidence",
                "admin_inference_session",
                "runtime_contract",
                "browser_miner",
            ],
        )


if __name__ == "__main__":
    unittest.main()
