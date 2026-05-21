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

    def test_build_result_payload_uses_sign_compressed_delta(self) -> None:
        payload = miner_cli.build_result_payload(
            {"lease_token": "lease", "attempt": 1, "workload_type": "diloco_train"},
            {"local_delta": [0.1, -0.3, 0.0], "inner_loss_start": 1.0},
            delta_format="sign_compressed",
            elapsed_ms=12.5,
        )

        self.assertNotIn("local_delta", payload)
        self.assertEqual(payload["compressed_delta"]["format"], "sign_compressed")
        self.assertEqual(payload["compressed_delta"]["signs"], [1, -1, 0])
        self.assertEqual(payload["metrics"]["delta_format"], "sign_compressed")
        self.assertEqual(payload["metrics"]["elapsed_ms"], 12.5)

    def test_build_result_payload_keeps_lora_adapter_delta(self) -> None:
        payload = miner_cli.build_result_payload(
            {"lease_token": "lease", "attempt": 1, "workload_type": "cpu_lora_mock"},
            {"adapter_delta": {"values": [0.1], "rank": 1}, "adapter_loss_start": 1.0},
            delta_format="sign_compressed",
            elapsed_ms=1.0,
        )

        self.assertIn("adapter_delta", payload)
        self.assertNotIn("compressed_delta", payload)


if __name__ == "__main__":
    unittest.main()
