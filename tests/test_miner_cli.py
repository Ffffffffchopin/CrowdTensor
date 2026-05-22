from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import importlib.util
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "miner_cli.py"
SPEC = importlib.util.spec_from_file_location("miner_cli", SCRIPT_PATH)
miner_cli = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(miner_cli)


def miner_args(**overrides):
    values = {
        "coordinator": "http://coordinator.test",
        "skip_preflight": False,
        "preflight_timeout": 1.0,
        "max_request_attempts": 3,
        "retry_base_sleep": 0.0,
        "retry_max_sleep": 0.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class MinerCliTests(unittest.TestCase):
    def test_retry_policy_only_retries_transport_and_selected_5xx(self) -> None:
        self.assertTrue(miner_cli.should_retry_error(miner_cli.CoordinatorTransportError("offline")))
        self.assertTrue(miner_cli.should_retry_error(miner_cli.CoordinatorHTTPError(500, "boom")))
        self.assertTrue(miner_cli.should_retry_error(miner_cli.CoordinatorHTTPError(502, "bad gateway")))
        self.assertTrue(miner_cli.should_retry_error(miner_cli.CoordinatorHTTPError(504, "timeout")))
        self.assertFalse(miner_cli.should_retry_error(miner_cli.CoordinatorHTTPError(401, "auth")))
        self.assertFalse(miner_cli.should_retry_error(miner_cli.CoordinatorHTTPError(409, "stale")))
        self.assertFalse(miner_cli.should_retry_error(miner_cli.CoordinatorHTTPError(422, "invalid")))
        self.assertFalse(miner_cli.should_retry_error(miner_cli.CoordinatorHTTPError(503, "no task")))

    def test_retry_sleep_caps_exponential_backoff(self) -> None:
        args = miner_args(retry_base_sleep=0.5, retry_max_sleep=1.0)

        self.assertEqual(miner_cli.retry_sleep_seconds(args, 1), 0.5)
        self.assertEqual(miner_cli.retry_sleep_seconds(args, 2), 1.0)
        self.assertEqual(miner_cli.retry_sleep_seconds(args, 3), 1.0)

    def test_request_retries_retryable_errors_and_counts_retries(self) -> None:
        args = miner_args()
        counters: Counter = Counter()
        calls = {"count": 0}

        def fake_request(*_args, **_kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise miner_cli.CoordinatorHTTPError(500, "temporary")
            return {"ok": True}

        with patch.object(miner_cli, "request_json", fake_request):
            payload = miner_cli.request_json_with_retries(
                "GET",
                "http://coordinator.test",
                "/ready",
                timeout=1.0,
                args=args,
                counters=counters,
            )

        self.assertEqual(payload, {"ok": True})
        self.assertEqual(calls["count"], 2)
        self.assertEqual(counters["request_retries"], 1)

    def test_request_retries_idempotent_result_upload(self) -> None:
        args = miner_args(max_request_attempts=5)
        counters: Counter = Counter()
        calls = {"count": 0}

        def fake_request(*_args, **_kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise miner_cli.CoordinatorHTTPError(500, "temporary")
            return {"accepted": True}

        with patch.object(miner_cli, "request_json", fake_request):
            payload = miner_cli.request_json_with_retries(
                "POST",
                "http://coordinator.test",
                "/tasks/task-1/result",
                {"lease_token": "lease", "attempt": 1, "idempotency_key": "key"},
                timeout=1.0,
                args=args,
                counters=counters,
                retry_result_upload=True,
            )

        self.assertEqual(payload, {"accepted": True})
        self.assertEqual(calls["count"], 2)
        self.assertEqual(counters["request_retries"], 1)

    def test_request_does_not_retry_non_idempotent_result_upload(self) -> None:
        args = miner_args(max_request_attempts=5)
        counters: Counter = Counter()
        calls = {"count": 0}

        def fake_request(*_args, **_kwargs):
            calls["count"] += 1
            raise miner_cli.CoordinatorHTTPError(500, "temporary")

        with patch.object(miner_cli, "request_json", fake_request):
            with self.assertRaises(miner_cli.CoordinatorHTTPError):
                miner_cli.request_json_with_retries(
                    "POST",
                    "http://coordinator.test",
                    "/tasks/task-1/result",
                    {"lease_token": "lease", "attempt": 1},
                    timeout=1.0,
                    args=args,
                    counters=counters,
                )

        self.assertEqual(calls["count"], 1)
        self.assertEqual(counters["request_retries"], 0)

    def test_preflight_validates_ready_payload(self) -> None:
        args = miner_args()
        counters: Counter = Counter()

        with patch.object(
            miner_cli,
            "request_json_with_retries",
            return_value={
                "ok": True,
                "service": "crowdtensord-coordinator",
                "version": "0.1.0a0",
                "protocol_version": miner_cli.EXPECTED_PROTOCOL_VERSION,
            },
        ):
            miner_cli.preflight(args, counters)

        self.assertEqual(counters["preflight_failures"], 0)

    def test_preflight_rejects_protocol_mismatch(self) -> None:
        args = miner_args()
        counters: Counter = Counter()

        with patch.object(
            miner_cli,
            "request_json_with_retries",
            return_value={"ok": True, "protocol_version": "old"},
        ):
            with self.assertRaisesRegex(RuntimeError, "protocol mismatch"):
                miner_cli.preflight(args, counters)

        self.assertEqual(counters["preflight_failures"], 1)

    def test_summary_includes_resilience_counters(self) -> None:
        args = argparse.Namespace(miner_id="miner-test")
        counters = Counter({
            "accepted_tasks": 1,
            "request_retries": 2,
            "preflight_failures": 0,
            "workload:diloco_train": 1,
        })

        payload = miner_cli.summary_payload(args, counters, 0.0)

        self.assertEqual(payload["request_retries"], 2)
        self.assertEqual(payload["preflight_failures"], 0)
        self.assertEqual(payload["workloads"], {"diloco_train": 1})

    def test_capabilities_advertise_delta_formats(self) -> None:
        capabilities = miner_cli.miner_capabilities()

        self.assertIn("dense_float", capabilities["supported_delta_formats"])
        self.assertIn("sign_compressed", capabilities["supported_delta_formats"])
        self.assertIn("sign_compressed_ef", capabilities["supported_delta_formats"])
        self.assertIn("model_bundle_lm", capabilities["supported_workloads"])
        self.assertIn("model_bundle_infer", capabilities["supported_workloads"])
        self.assertNotIn("external_llm_infer", capabilities["supported_workloads"])
        self.assertEqual(capabilities["external_llm_runtime"], {})
        self.assertEqual(capabilities["backend"], "cpu")
        profile = capabilities["hardware_profile"]
        self.assertGreaterEqual(profile["cpu_count"], 1)
        self.assertTrue(profile["os"])
        self.assertTrue(profile["python_version"])

    def test_capabilities_advertise_external_llm_only_when_enabled(self) -> None:
        mock_capabilities = miner_cli.miner_capabilities(enable_mock_llm_runtime=True)
        command_capabilities = miner_cli.miner_capabilities(
            llm_runtime_cmd="/usr/local/bin/local-llm",
            llm_runtime_model_id="local-model",
        )
        http_capabilities = miner_cli.miner_capabilities(
            llm_runtime_url="http://127.0.0.1:11434/v1/chat/completions",
            llm_runtime_model_id="http-model",
        )

        self.assertIn("external_llm_infer", mock_capabilities["supported_workloads"])
        self.assertEqual(mock_capabilities["external_llm_runtime"]["adapter_kind"], "mock")
        self.assertIn("external_llm_infer", command_capabilities["supported_workloads"])
        self.assertEqual(command_capabilities["external_llm_runtime"]["adapter_kind"], "command")
        self.assertEqual(command_capabilities["external_llm_runtime"]["model_id"], "local-model")
        self.assertIn("external_llm_infer", http_capabilities["supported_workloads"])
        self.assertEqual(http_capabilities["external_llm_runtime"]["adapter_kind"], "http_openai_chat")
        self.assertEqual(http_capabilities["external_llm_runtime"]["model_id"], "http-model")
        self.assertNotIn("runtime_url", http_capabilities["external_llm_runtime"])

    def test_auto_delta_format_follows_claim_optimizer_spec(self) -> None:
        self.assertEqual(
            miner_cli.delta_format_for_claim(
                {"optimizer_spec": {"delta_format": "sign_compressed"}},
                "auto",
            ),
            "sign_compressed",
        )
        self.assertEqual(miner_cli.delta_format_for_claim({}, "auto"), "dense_float")
        self.assertEqual(miner_cli.delta_format_for_claim({}, "sign_compressed_ef"), "sign_compressed_ef")

    def test_build_result_payload_uses_sign_compressed_delta(self) -> None:
        payload, next_residual = miner_cli.build_result_payload(
            {"lease_token": "lease", "attempt": 1, "workload_type": "diloco_train"},
            {"local_delta": [0.1, -0.3, 0.0], "inner_loss_start": 1.0},
            delta_format="sign_compressed",
            elapsed_ms=12.5,
        )

        self.assertIsNone(next_residual)
        self.assertNotIn("local_delta", payload)
        self.assertEqual(payload["compressed_delta"]["format"], "sign_compressed")
        self.assertEqual(payload["compressed_delta"]["signs"], [1, -1, 0])
        self.assertEqual(payload["metrics"]["delta_format"], "sign_compressed")
        self.assertEqual(payload["metrics"]["elapsed_ms"], 12.5)

    def test_build_result_payload_uses_error_feedback_delta(self) -> None:
        payload, next_residual = miner_cli.build_result_payload(
            {"lease_token": "lease", "attempt": 1, "workload_type": "diloco_train"},
            {"local_delta": [0.1, -0.3, 0.0], "inner_loss_start": 1.0},
            delta_format="sign_compressed_ef",
            elapsed_ms=12.5,
            residual=[0.05, -0.05, 0.02],
        )

        self.assertIsNotNone(next_residual)
        self.assertEqual(len(next_residual or []), 3)
        self.assertNotIn("local_delta", payload)
        self.assertEqual(payload["compressed_delta"]["format"], "sign_compressed_ef")
        self.assertIn("error_feedback", payload["compressed_delta"])
        self.assertEqual(payload["metrics"]["delta_format"], "sign_compressed_ef")

    def test_build_result_payload_keeps_lora_adapter_delta(self) -> None:
        payload, next_residual = miner_cli.build_result_payload(
            {"lease_token": "lease", "attempt": 1, "workload_type": "cpu_lora_mock"},
            {"adapter_delta": {"values": [0.1], "rank": 1}, "adapter_loss_start": 1.0},
            delta_format="sign_compressed",
            elapsed_ms=1.0,
        )

        self.assertIsNone(next_residual)
        self.assertIn("adapter_delta", payload)
        self.assertNotIn("compressed_delta", payload)

    def test_build_result_payload_keeps_micro_transformer_delta_dense(self) -> None:
        payload, next_residual = miner_cli.build_result_payload(
            {"lease_token": "lease", "attempt": 1, "workload_type": "micro_transformer_lm"},
            {"local_delta": [0.1, -0.2, 0.0], "lm_loss_start": 1.0},
            delta_format="sign_compressed_ef",
            elapsed_ms=1.0,
        )

        self.assertIsNone(next_residual)
        self.assertEqual(payload["local_delta"], [0.1, -0.2, 0.0])
        self.assertNotIn("compressed_delta", payload)

    def test_build_result_payload_keeps_model_bundle_delta_dense(self) -> None:
        bundle_delta = {
            "schema_version": "model_bundle_lm_v1",
            "bundle_id": "bundle",
            "base_bundle_version": 0,
            "artifact_hash": "sha256:abc",
            "values": [0.1, -0.2],
        }

        payload, next_residual = miner_cli.build_result_payload(
            {"lease_token": "lease", "attempt": 1, "workload_type": "model_bundle_lm"},
            {"bundle_delta": bundle_delta, "bundle_loss_start": 1.0},
            delta_format="sign_compressed_ef",
            elapsed_ms=1.0,
        )

        self.assertIsNone(next_residual)
        self.assertEqual(payload["bundle_delta"], bundle_delta)
        self.assertNotIn("compressed_delta", payload)

    def test_build_result_payload_keeps_model_bundle_inference_result(self) -> None:
        inference_result = {
            "schema_version": "model_bundle_infer_v1",
            "bundle_id": "bundle",
            "base_bundle_version": 0,
            "artifact_hash": "sha256:abc",
            "prompt_token_ids": [1, 2, 3, 4],
            "target_token_id": 5,
            "predicted_token_id": 5,
            "top_k": [{"token_id": 5, "probability": 0.25}],
            "correct": True,
        }

        payload, next_residual = miner_cli.build_result_payload(
            {"lease_token": "lease", "attempt": 1, "workload_type": "model_bundle_infer"},
            {
                "inference_result": inference_result,
                "inference_results": [inference_result, {**inference_result, "request_id": "req-2"}],
                "prediction_correct": True,
                "request_count": 2,
            },
            delta_format="sign_compressed_ef",
            elapsed_ms=1.0,
        )

        self.assertIsNone(next_residual)
        self.assertEqual(payload["inference_result"], inference_result)
        self.assertEqual(len(payload["inference_results"]), 2)
        self.assertNotIn("compressed_delta", payload)
        self.assertNotIn("inference_result", payload["metrics"])
        self.assertNotIn("inference_results", payload["metrics"])
        self.assertEqual(payload["metrics"]["request_count"], 2)
        self.assertTrue(payload["metrics"]["prediction_correct"])

    def test_build_result_payload_keeps_external_llm_result(self) -> None:
        external_result = {
            "schema_version": "external_llm_infer_v1",
            "request_id": "req-1",
            "prompt_hash": "sha256:abc",
            "adapter_kind": "mock",
            "model_id": "mock-external-llm",
            "output_text": "mock completion",
            "output_chars": 15,
        }

        payload, next_residual = miner_cli.build_result_payload(
            {"lease_token": "lease", "attempt": 1, "workload_type": "external_llm_infer"},
            {
                "external_llm_result": external_result,
                "external_llm_results": [external_result, {**external_result, "request_id": "req-2"}],
                "request_count": 2,
                "completion_count": 2,
                "adapter_kind": "mock",
                "model_id": "mock-external-llm",
            },
            delta_format="sign_compressed_ef",
            elapsed_ms=1.0,
        )

        self.assertIsNone(next_residual)
        self.assertEqual(payload["external_llm_result"], external_result)
        self.assertEqual(len(payload["external_llm_results"]), 2)
        self.assertNotIn("compressed_delta", payload)
        self.assertNotIn("external_llm_result", payload["metrics"])
        self.assertNotIn("external_llm_results", payload["metrics"])
        self.assertEqual(payload["metrics"]["request_count"], 2)
        self.assertEqual(payload["metrics"]["completion_count"], 2)
        self.assertEqual(payload["metrics"]["adapter_kind"], "mock")


if __name__ == "__main__":
    unittest.main()
