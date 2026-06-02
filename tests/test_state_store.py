from __future__ import annotations

import json
import hashlib
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from crowdtensor.protocol import LeaseConflict, NoTaskAvailable, ResultRejected
from crowdtensor.state_store import StateStore
from crowdtensor.diloco import run_inner_loop
from crowdtensor.external_llm import WORKLOAD_TYPE as WORKLOAD_EXTERNAL_LLM_INFER
from crowdtensor.external_llm import run_mock_external_llm_inference
from crowdtensor.micro_transformer import MICRO_LLM_SHARDED_WORKLOAD_TYPE as WORKLOAD_MICRO_LLM_SHARDED_INFER
from crowdtensor.micro_transformer import WORKLOAD_TYPE as WORKLOAD_MICRO_TRANSFORMER_LM
from crowdtensor.micro_transformer import run_micro_llm_sharded_inference, run_micro_transformer_inner_loop
from crowdtensor.micro_llm_artifact import build_default_micro_llm_artifact
from crowdtensor.protocol import (
    MICRO_LLM_SHARDED_BOTH_CAPABILITY,
    MICRO_LLM_SHARDED_STAGE0_CAPABILITY,
    MICRO_LLM_SHARDED_STAGE1_CAPABILITY,
    REAL_LLM_SHARDED_BOTH_CAPABILITY,
    REAL_LLM_SHARDED_CUDA_BOTH_CAPABILITY,
    REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY,
    REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY,
    REAL_LLM_SHARDED_STAGE0_CAPABILITY,
    REAL_LLM_SHARDED_STAGE1_CAPABILITY,
    STATUS_COMPLETED,
)
from crowdtensor.real_llm import WORKLOAD_TYPE as WORKLOAD_REAL_LLM_SHARDED_INFER
from crowdtensor.model_bundle import WORKLOAD_TYPE as WORKLOAD_MODEL_BUNDLE_LM
from crowdtensor.model_bundle import INFERENCE_WORKLOAD_TYPE as WORKLOAD_MODEL_BUNDLE_INFER
from crowdtensor.model_bundle import SHARDED_INFERENCE_WORKLOAD_TYPE as WORKLOAD_SHARDED_MODEL_BUNDLE_INFER
from crowdtensor.model_bundle import run_model_bundle_inference
from crowdtensor.model_bundle import run_model_bundle_inner_loop
from crowdtensor.model_bundle import run_sharded_model_bundle_inference
from crowdtensor.outer_optimizer import (
    DELTA_FORMAT_DENSE_FLOAT,
    DELTA_FORMAT_SIGN_COMPRESSED,
    DELTA_FORMAT_SIGN_COMPRESSED_EF,
    OPTIMIZER_DILOCO_NESTEROV,
    compress_sign_delta,
    compress_sign_delta_with_error_feedback,
)
from crowdtensor.toy_compute import compute_pseudo_gradient


class StateStoreTests(unittest.TestCase):
    def _complete_claim(self, store: StateStore, claim: dict, miner_id: str) -> dict:
        inner_result = run_inner_loop(
            claim["weights"],
            task_id=claim["task_id"],
            miner_id=miner_id,
            model_version=claim["model_version"],
            inner_steps=claim["inner_steps"],
        )
        return store.complete_task(
            claim["task_id"],
            lease_token=claim["lease_token"],
            attempt=claim["attempt"],
            local_delta=inner_result["local_delta"],
            metrics=inner_result,
        )

    def _lora_capabilities(self) -> dict:
        return {
            "runtime": "python-cli",
            "backend": "cpu",
            "protocol_version": "runtime_contract_v1",
            "supported_workloads": ["cpu_lora_mock"],
        }

    def _python_capabilities(self, workloads: list[str] | None = None) -> dict:
        return {
            "runtime": "python-cli",
            "backend": "cpu",
            "protocol_version": "runtime_contract_v1",
            "supported_workloads": workloads or ["diloco_train"],
        }

    def _micro_transformer_capabilities(self) -> dict:
        return self._python_capabilities([WORKLOAD_MICRO_TRANSFORMER_LM])

    def _model_bundle_capabilities(self) -> dict:
        return self._python_capabilities([WORKLOAD_MODEL_BUNDLE_LM])

    def _model_bundle_inference_capabilities(self) -> dict:
        return self._python_capabilities([WORKLOAD_MODEL_BUNDLE_INFER])

    def _sharded_model_bundle_inference_capabilities(self) -> dict:
        return self._python_capabilities([WORKLOAD_SHARDED_MODEL_BUNDLE_INFER])

    def _micro_llm_sharded_inference_capabilities(self) -> dict:
        capabilities = self._python_capabilities([WORKLOAD_MICRO_LLM_SHARDED_INFER])
        capabilities["micro_llm_sharded_stage_role"] = "both"
        capabilities["micro_llm_sharded_stage_capabilities"] = [
            MICRO_LLM_SHARDED_STAGE0_CAPABILITY,
            MICRO_LLM_SHARDED_STAGE1_CAPABILITY,
            MICRO_LLM_SHARDED_BOTH_CAPABILITY,
        ]
        return capabilities

    def _micro_llm_stage_capabilities(self, stage_role: str) -> dict:
        capabilities = self._python_capabilities([WORKLOAD_MICRO_LLM_SHARDED_INFER])
        capabilities["micro_llm_sharded_stage_role"] = stage_role
        if stage_role == "stage0":
            capabilities["micro_llm_sharded_stage_capabilities"] = [MICRO_LLM_SHARDED_STAGE0_CAPABILITY]
        elif stage_role == "stage1":
            capabilities["micro_llm_sharded_stage_capabilities"] = [MICRO_LLM_SHARDED_STAGE1_CAPABILITY]
        else:
            capabilities["micro_llm_sharded_stage_capabilities"] = [
                MICRO_LLM_SHARDED_STAGE0_CAPABILITY,
                MICRO_LLM_SHARDED_STAGE1_CAPABILITY,
                MICRO_LLM_SHARDED_BOTH_CAPABILITY,
            ]
        return capabilities

    def _real_llm_stage_capabilities(self, stage_role: str) -> dict:
        capabilities = self._python_capabilities([WORKLOAD_REAL_LLM_SHARDED_INFER])
        capabilities["real_llm_sharded_stage_role"] = stage_role
        if stage_role == "stage0":
            capabilities["real_llm_sharded_stage_capabilities"] = [REAL_LLM_SHARDED_STAGE0_CAPABILITY]
        elif stage_role == "stage1":
            capabilities["real_llm_sharded_stage_capabilities"] = [REAL_LLM_SHARDED_STAGE1_CAPABILITY]
        else:
            capabilities["real_llm_sharded_stage_capabilities"] = [
                REAL_LLM_SHARDED_STAGE0_CAPABILITY,
                REAL_LLM_SHARDED_STAGE1_CAPABILITY,
                REAL_LLM_SHARDED_BOTH_CAPABILITY,
            ]
        return capabilities

    def _real_llm_cuda_stage_capabilities(self, stage_role: str) -> dict:
        capabilities = self._python_capabilities([WORKLOAD_REAL_LLM_SHARDED_INFER])
        capabilities["backend"] = "cuda"
        capabilities["real_llm_runtime"] = {"adapter_kind": "hf_transformers_cuda"}
        capabilities["real_llm_sharded_stage_role"] = stage_role
        if stage_role == "stage0":
            capabilities["real_llm_sharded_stage_capabilities"] = [REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY]
        elif stage_role == "stage1":
            capabilities["real_llm_sharded_stage_capabilities"] = [REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY]
        else:
            capabilities["real_llm_sharded_stage_capabilities"] = [
                REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY,
                REAL_LLM_SHARDED_CUDA_STAGE1_CAPABILITY,
                REAL_LLM_SHARDED_CUDA_BOTH_CAPABILITY,
            ]
        return capabilities

    def _external_llm_inference_capabilities(self) -> dict:
        return self._python_capabilities([WORKLOAD_EXTERNAL_LLM_INFER])

    def test_claim_complete_checkpoints_and_creates_next_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=1)
            claim = store.claim_task("miner-a")
            inner_result = run_inner_loop(
                claim["weights"],
                task_id=claim["task_id"],
                miner_id="miner-a",
                model_version=claim["model_version"],
                inner_steps=claim["inner_steps"],
            )

            result = store.complete_task(
                claim["task_id"],
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                local_delta=inner_result["local_delta"],
                metrics=inner_result,
            )

            self.assertEqual(result["global_step"], 1)
            self.assertEqual(result["model_version"], 1)
            self.assertEqual(result["optimizer_step"], 1)
            self.assertTrue((Path(tmp) / "global_model.json").exists())
            summary = store.summary()
            self.assertEqual(summary["model"]["schema_version"], "diloco_mock_v1")
            self.assertEqual(summary["task_counts"]["completed"], 1)
            self.assertEqual(summary["task_counts"]["queued"], 1)
            self.assertEqual(summary["accepted_results"], 1)
            self.assertEqual(summary["model_updates"], 1)
            self.assertEqual(summary["max_staleness"], 0)
            self.assertEqual(summary["last_completed"]["task_id"], claim["task_id"])
            self.assertEqual(summary["task_lanes"][0]["runtime"], "any")
            self.assertEqual(summary["task_lanes"][0]["count"], 1)
            self.assertEqual(summary["audit_results"], 0)
            self.assertEqual(summary["audit_rejections"], 0)
            self.assertEqual(claim["workload_type"], "diloco_train")
            self.assertEqual(claim["audit_mode"], "none")
            self.assertEqual(claim["optimizer_spec"]["contract_version"], "outer_optimizer_contract_v1")
            self.assertEqual(claim["optimizer_spec"]["optimizer_type"], "diloco_momentum")
            self.assertEqual(claim["optimizer_spec"]["delta_format"], "dense_float")
            self.assertEqual(result["optimizer"]["optimizer_step_before"], 0)
            self.assertEqual(result["optimizer"]["optimizer_step_after"], 1)
            self.assertGreater(result["optimizer"]["delta_norm"], 0.0)
            self.assertEqual(summary["last_completed"]["optimizer"]["optimizer_step_after"], 1)
            self.assertIn("micro_transformer", summary["model"])

    def test_claim_advertises_configured_delta_format_without_mutating_model_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=1,
                delta_format=DELTA_FORMAT_SIGN_COMPRESSED_EF,
            )
            claim = store.claim_task(
                "compressed-capable",
                capabilities={
                    **self._python_capabilities(),
                    "supported_delta_formats": [DELTA_FORMAT_SIGN_COMPRESSED_EF],
                },
            )

            self.assertEqual(claim["optimizer_spec"]["delta_format"], DELTA_FORMAT_SIGN_COMPRESSED_EF)
            self.assertEqual(
                store.summary()["model"]["outer_optimizer_contract"]["delta_format"],
                DELTA_FORMAT_DENSE_FLOAT,
            )

    def test_compressed_delta_claim_requires_matching_capability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=1,
                delta_format=DELTA_FORMAT_SIGN_COMPRESSED_EF,
            )

            with self.assertRaises(NoTaskAvailable):
                store.claim_task("legacy-miner", capabilities=self._python_capabilities())

            claim = store.claim_task(
                "compressed-capable",
                capabilities={
                    **self._python_capabilities(),
                    "supported_delta_formats": [DELTA_FORMAT_SIGN_COMPRESSED_EF],
                },
            )
            self.assertEqual(claim["optimizer_spec"]["delta_format"], DELTA_FORMAT_SIGN_COMPRESSED_EF)

    def test_dense_claim_accepts_legacy_missing_delta_capability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=1)

            claim = store.claim_task("legacy-miner", capabilities=self._python_capabilities())

            self.assertEqual(claim["optimizer_spec"]["delta_format"], DELTA_FORMAT_DENSE_FLOAT)

    def test_idempotent_completed_result_does_not_update_twice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=1)
            claim = store.claim_task("idempotent-miner")
            inner_result = run_inner_loop(
                claim["weights"],
                task_id=claim["task_id"],
                miner_id="idempotent-miner",
                model_version=claim["model_version"],
                inner_steps=claim["inner_steps"],
            )
            first = store.complete_task(
                claim["task_id"],
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                idempotency_key="result-key",
                local_delta=inner_result["local_delta"],
                metrics=inner_result,
            )
            duplicate = store.complete_task(
                claim["task_id"],
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                idempotency_key="result-key",
                local_delta=inner_result["local_delta"],
                metrics=inner_result,
            )
            with self.assertRaises(LeaseConflict):
                store.complete_task(
                    claim["task_id"],
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    idempotency_key="different-key",
                    local_delta=inner_result["local_delta"],
                    metrics=inner_result,
                )

            summary = store.summary()
            self.assertEqual(first, duplicate)
            self.assertEqual(summary["accepted_results"], 1)
            self.assertEqual(summary["model"]["global_step"], 1)
            self.assertNotIn("result_idempotency_key_hash", json.dumps(summary))
            self.assertNotIn("result_lease_token_hash", json.dumps(summary))

    def test_idempotent_rejected_result_does_not_reject_twice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=1)
            claim = store.claim_task("bad-idempotent-miner")
            with self.assertRaises(ResultRejected) as first:
                store.complete_task(
                    claim["task_id"],
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    idempotency_key="bad-result-key",
                    local_delta=[100.0 for _ in claim["weights"]],
                )
            with self.assertRaises(ResultRejected) as duplicate:
                store.complete_task(
                    claim["task_id"],
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    idempotency_key="bad-result-key",
                    local_delta=[100.0 for _ in claim["weights"]],
                )

            summary = store.summary()
            self.assertEqual(first.exception.validation, duplicate.exception.validation)
            self.assertEqual(summary["rejected_results"], 1)
            self.assertEqual(summary["accepted_results"], 0)

    def test_idempotent_result_survives_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=1)
            claim = store.claim_task("restart-idempotent-miner")
            inner_result = run_inner_loop(
                claim["weights"],
                task_id=claim["task_id"],
                miner_id="restart-idempotent-miner",
                model_version=claim["model_version"],
                inner_steps=claim["inner_steps"],
            )
            first = store.complete_task(
                claim["task_id"],
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                idempotency_key="restart-result-key",
                local_delta=inner_result["local_delta"],
                metrics=inner_result,
            )

            recovered = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=1)
            duplicate = recovered.complete_task(
                claim["task_id"],
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                idempotency_key="restart-result-key",
                local_delta=inner_result["local_delta"],
                metrics=inner_result,
            )
            summary = recovered.summary()

            self.assertEqual(first, duplicate)
            self.assertEqual(summary["accepted_results"], 1)
            self.assertEqual(summary["model"]["global_step"], 1)

    def test_result_ledger_summarizes_terminal_results_and_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=1)
            accepted_claim = store.claim_task("ledger-accepted")
            inner_result = run_inner_loop(
                accepted_claim["weights"],
                task_id=accepted_claim["task_id"],
                miner_id="ledger-accepted",
                model_version=accepted_claim["model_version"],
                inner_steps=accepted_claim["inner_steps"],
            )
            store.complete_task(
                accepted_claim["task_id"],
                lease_token=accepted_claim["lease_token"],
                attempt=accepted_claim["attempt"],
                idempotency_key="ledger-key",
                local_delta=inner_result["local_delta"],
                metrics=inner_result,
            )

            rejected_claim = store.claim_task("ledger-rejected")
            with self.assertRaises(ResultRejected):
                store.complete_task(
                    rejected_claim["task_id"],
                    lease_token=rejected_claim["lease_token"],
                    attempt=rejected_claim["attempt"],
                    idempotency_key="bad-ledger-key",
                    local_delta=[100.0 for _ in rejected_claim["weights"]],
                )

            rows = store.result_ledger(limit=10)
            self.assertEqual(len(rows), 2)
            self.assertGreater(rows[0]["event_index"], rows[1]["event_index"])
            accepted = store.result_ledger(status="accepted")[0]
            rejected = store.result_ledger(status="rejected")[0]
            self.assertEqual(accepted["task_id"], accepted_claim["task_id"])
            self.assertEqual(accepted["validation"]["code"], "ok")
            self.assertEqual(accepted["audit"], {})
            self.assertEqual(accepted["optimizer"]["contract_version"], "outer_optimizer_contract_v1")
            self.assertEqual(accepted["optimizer"]["optimizer_type"], "diloco_momentum")
            self.assertEqual(accepted["optimizer"]["optimizer_step_after"], 1)
            self.assertTrue(accepted["idempotent"])
            self.assertTrue(accepted["model_updated"])
            self.assertEqual(rejected["task_id"], rejected_claim["task_id"])
            self.assertFalse(rejected["accepted"])
            self.assertEqual(rejected["validation"]["code"], "delta_norm_too_large")
            self.assertEqual(rejected["miner_workload_score"]["rejected"], 1)
            self.assertEqual(
                store.result_ledger(miner_id="ledger-accepted", workload_type="diloco_train")[0]["task_id"],
                accepted_claim["task_id"],
            )
            self.assertEqual(store.result_ledger(task_id=accepted_claim["task_id"])[0]["task_id"], accepted_claim["task_id"])
            self.assertEqual(store.result_ledger(task_id="missing-task"), [])
            self.assertEqual(store.result_ledger(limit=0), [])
            with self.assertRaises(ValueError):
                store.result_ledger(status="broken")

            public_text = json.dumps(rows, sort_keys=True)
            for forbidden in [
                "ledger-key",
                "bad-ledger-key",
                '"lease_token"',
                '"result_idempotency_key_hash"',
                '"result_lease_token_hash"',
                '"result_response"',
                '"local_delta"',
                '"adapter_delta"',
            ]:
                self.assertNotIn(forbidden, public_text)

    def test_sign_compressed_result_updates_model_and_ledger_safely(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=1,
                delta_format=DELTA_FORMAT_SIGN_COMPRESSED,
            )
            claim = store.claim_task(
                "compressed-miner",
                capabilities={
                    **self._python_capabilities(),
                    "supported_delta_formats": [DELTA_FORMAT_SIGN_COMPRESSED],
                },
            )
            inner_result = run_inner_loop(
                claim["weights"],
                task_id=claim["task_id"],
                miner_id="compressed-miner",
                model_version=claim["model_version"],
                inner_steps=claim["inner_steps"],
            )

            result = store.complete_task(
                claim["task_id"],
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                compressed_delta=compress_sign_delta(inner_result["local_delta"]),
                metrics={"transport": "sign_compressed"},
            )

            self.assertTrue(result["accepted"])
            self.assertEqual(result["optimizer"]["delta_format"], "sign_compressed")
            self.assertIn("decoded_delta_norm", result["optimizer"])
            self.assertIn("compression_ratio_estimate", result["optimizer"])
            row = store.result_ledger(status="accepted")[0]
            self.assertEqual(row["optimizer"]["delta_format"], "sign_compressed")
            public_text = json.dumps(row, sort_keys=True)
            self.assertNotIn("compressed_delta", public_text)
            self.assertNotIn("signs", public_text)

    def test_sign_compressed_error_feedback_updates_model_and_ledger_safely(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=1,
                delta_format=DELTA_FORMAT_SIGN_COMPRESSED_EF,
            )
            claim = store.claim_task(
                "compressed-ef-miner",
                capabilities={
                    **self._python_capabilities(),
                    "supported_delta_formats": [DELTA_FORMAT_SIGN_COMPRESSED_EF],
                },
            )
            inner_result = run_inner_loop(
                claim["weights"],
                task_id=claim["task_id"],
                miner_id="compressed-ef-miner",
                model_version=claim["model_version"],
                inner_steps=claim["inner_steps"],
            )
            compressed, residual = compress_sign_delta_with_error_feedback(
                inner_result["local_delta"],
                residual=[0.01, -0.02, 0.03],
            )

            result = store.complete_task(
                claim["task_id"],
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                compressed_delta=compressed,
                metrics={"transport": DELTA_FORMAT_SIGN_COMPRESSED_EF},
            )

            self.assertTrue(result["accepted"])
            self.assertEqual(len(residual), len(claim["weights"]))
            self.assertEqual(result["optimizer"]["delta_format"], DELTA_FORMAT_SIGN_COMPRESSED_EF)
            self.assertTrue(result["optimizer"]["error_feedback"])
            self.assertIn("residual_norm", result["optimizer"])
            self.assertIn("corrected_delta_norm", result["optimizer"])
            row = store.result_ledger(status="accepted")[0]
            self.assertEqual(row["optimizer"]["delta_format"], DELTA_FORMAT_SIGN_COMPRESSED_EF)
            self.assertTrue(row["optimizer"]["error_feedback"])
            self.assertEqual(row["validation"]["delta_format"], DELTA_FORMAT_SIGN_COMPRESSED_EF)
            public_text = json.dumps(row, sort_keys=True)
            self.assertNotIn("compressed_delta", public_text)
            self.assertNotIn("signs", public_text)

    def test_delta_format_mismatch_rejects_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=1,
                delta_format=DELTA_FORMAT_SIGN_COMPRESSED_EF,
            )
            claim = store.claim_task(
                "wrong-transport-miner",
                capabilities={
                    **self._python_capabilities(),
                    "supported_delta_formats": [DELTA_FORMAT_SIGN_COMPRESSED_EF],
                },
            )
            inner_result = run_inner_loop(
                claim["weights"],
                task_id=claim["task_id"],
                miner_id="wrong-transport-miner",
                model_version=claim["model_version"],
                inner_steps=claim["inner_steps"],
            )

            with self.assertRaises(ResultRejected) as raised:
                store.complete_task(
                    claim["task_id"],
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    local_delta=inner_result["local_delta"],
                )

            self.assertEqual(raised.exception.validation["code"], "delta_format_mismatch")
            self.assertEqual(raised.exception.validation["expected_delta_format"], DELTA_FORMAT_SIGN_COMPRESSED_EF)
            self.assertEqual(raised.exception.validation["actual_delta_format"], DELTA_FORMAT_DENSE_FLOAT)

    def test_replay_audit_rejects_error_feedback_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "sign_compressed_ef"):
                StateStore(
                    tmp,
                    lease_seconds=5,
                    inner_steps=10,
                    backlog=1,
                    replay_audit=True,
                    delta_format=DELTA_FORMAT_SIGN_COMPRESSED_EF,
                )

    def test_result_ledger_survives_restart_with_event_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=1)
            claim = store.claim_task("ledger-restart")
            inner_result = run_inner_loop(
                claim["weights"],
                task_id=claim["task_id"],
                miner_id="ledger-restart",
                model_version=claim["model_version"],
                inner_steps=claim["inner_steps"],
            )
            store.complete_task(
                claim["task_id"],
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                idempotency_key="restart-ledger-key",
                local_delta=inner_result["local_delta"],
                metrics=inner_result,
            )
            before = store.result_ledger()[0]

            restarted = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=1)
            after = restarted.result_ledger()[0]

            self.assertEqual(after["task_id"], before["task_id"])
            self.assertEqual(after["event_index"], before["event_index"])
            self.assertTrue(after["idempotent"])

    def test_nesterov_outer_optimizer_updates_and_survives_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=1,
                outer_optimizer=OPTIMIZER_DILOCO_NESTEROV,
            )
            claim = store.claim_task("nesterov-miner")
            self.assertEqual(claim["optimizer_spec"]["optimizer_type"], OPTIMIZER_DILOCO_NESTEROV)
            inner_result = run_inner_loop(
                claim["weights"],
                task_id=claim["task_id"],
                miner_id="nesterov-miner",
                model_version=claim["model_version"],
                inner_steps=claim["inner_steps"],
            )

            result = store.complete_task(
                claim["task_id"],
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                local_delta=inner_result["local_delta"],
                metrics=inner_result,
            )

            self.assertEqual(result["optimizer"]["optimizer_type"], OPTIMIZER_DILOCO_NESTEROV)
            self.assertIn("outer_update_norm", result["optimizer"])
            self.assertEqual(store.summary()["model"]["outer_optimizer_type"], OPTIMIZER_DILOCO_NESTEROV)

            restarted = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=1)
            summary = restarted.summary()
            self.assertEqual(summary["model"]["outer_optimizer_type"], OPTIMIZER_DILOCO_NESTEROV)
            self.assertEqual(summary["last_completed"]["optimizer"]["optimizer_type"], OPTIMIZER_DILOCO_NESTEROV)
            self.assertIn("outer_update_norm", summary["last_completed"]["optimizer"])

    def test_legacy_completed_event_replays_with_momentum_even_when_nesterov_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=1)
            claim = store.claim_task("legacy-replay")
            inner_result = run_inner_loop(
                claim["weights"],
                task_id=claim["task_id"],
                miner_id="legacy-replay",
                model_version=claim["model_version"],
                inner_steps=claim["inner_steps"],
            )
            store.complete_task(
                claim["task_id"],
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                local_delta=inner_result["local_delta"],
                metrics=inner_result,
            )
            expected_weights = store.summary()["model"]["weights"]
            checkpoint = Path(tmp) / "global_model.json"
            checkpoint.unlink()
            task_log = Path(tmp) / "tasks.jsonl"
            lines = []
            for line in task_log.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                if event.get("type") == "task_completed":
                    event.pop("optimizer", None)
                lines.append(json.dumps(event, sort_keys=True))
            task_log.write_text("\n".join(lines) + "\n", encoding="utf-8")

            recovered = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=1,
                outer_optimizer=OPTIMIZER_DILOCO_NESTEROV,
            )

            self.assertEqual(recovered.summary()["model"]["weights"], expected_weights)
            self.assertEqual(recovered.summary()["model"]["outer_optimizer_type"], "diloco_momentum")

    def test_micro_transformer_workload_updates_nested_lm_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=4,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": WORKLOAD_MICRO_TRANSFORMER_LM,
                    },
                ],
            )
            claim = store.claim_task("lm-miner", capabilities=self._micro_transformer_capabilities())
            inner_result = run_micro_transformer_inner_loop(claim["workload_spec"], inner_steps=claim["inner_steps"])
            result = store.complete_task(
                claim["task_id"],
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                local_delta=inner_result["local_delta"],
                metrics=inner_result,
            )

            summary = store.summary()
            self.assertEqual(claim["workload_type"], WORKLOAD_MICRO_TRANSFORMER_LM)
            self.assertEqual(result["workload_type"], WORKLOAD_MICRO_TRANSFORMER_LM)
            self.assertTrue(result["micro_transformer_updated"])
            self.assertFalse(result["model_updated"])
            self.assertEqual(summary["model"]["global_step"], 0)
            self.assertEqual(summary["model"]["micro_transformer"]["version"], 1)
            self.assertEqual(summary["model_updates"], 0)
            self.assertEqual(summary["micro_transformer_updates"], 1)
            self.assertEqual(summary["accepted_results"], 1)

    def test_micro_transformer_replay_audit_rejects_mismatch_and_quarantines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=4,
                backlog=0,
                replay_audit=True,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": WORKLOAD_MICRO_TRANSFORMER_LM,
                    },
                ],
            )
            for _ in range(2):
                claim = store.claim_task("bad-lm-miner", capabilities=self._micro_transformer_capabilities())
                bad_delta = [0.0 for _ in claim["weights"]]
                with self.assertRaises(ResultRejected) as raised:
                    store.complete_task(
                        claim["task_id"],
                        lease_token=claim["lease_token"],
                        attempt=claim["attempt"],
                        local_delta=bad_delta,
                    )
                self.assertEqual(raised.exception.validation["code"], "micro_transformer_delta_replay_mismatch")

            summary = store.summary()
            self.assertEqual(summary["audit_rejections"], 2)
            self.assertTrue(
                summary["miner_workload_scores"]["bad-lm-miner"][WORKLOAD_MICRO_TRANSFORMER_LM]["quarantined"]
            )
            with self.assertRaises(NoTaskAvailable):
                store.claim_task("bad-lm-miner", capabilities=self._micro_transformer_capabilities())

    def test_model_bundle_workload_updates_nested_bundle_state_and_survives_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=6,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": WORKLOAD_MODEL_BUNDLE_LM,
                    },
                ],
            )
            claim = store.claim_task("bundle-miner", capabilities=self._model_bundle_capabilities())
            inner_result = run_model_bundle_inner_loop(
                claim["workload_spec"],
                inner_steps=claim["inner_steps"],
            )
            result = store.complete_task(
                claim["task_id"],
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                bundle_delta=inner_result["bundle_delta"],
                metrics=inner_result,
            )

            summary = store.summary()
            self.assertEqual(claim["workload_type"], WORKLOAD_MODEL_BUNDLE_LM)
            self.assertEqual(result["workload_type"], WORKLOAD_MODEL_BUNDLE_LM)
            self.assertTrue(result["model_bundle_updated"])
            self.assertFalse(result["model_updated"])
            self.assertEqual(summary["model"]["global_step"], 0)
            self.assertEqual(summary["model"]["model_bundle"]["version"], 1)
            self.assertEqual(summary["model_updates"], 0)
            self.assertEqual(summary["model_bundle_updates"], 1)
            self.assertEqual(summary["accepted_results"], 1)
            row = store.result_ledger(status="accepted")[0]
            self.assertTrue(row["model_bundle_updated"])
            self.assertEqual(row["workload_type"], WORKLOAD_MODEL_BUNDLE_LM)
            self.assertEqual(row["validation"]["bundle_id"], result["bundle_id"])
            self.assertNotIn("bundle_delta", json.dumps(summary["tasks"], sort_keys=True))

            recovered = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=6,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": WORKLOAD_MODEL_BUNDLE_LM,
                    },
                ],
            )
            recovered_summary = recovered.summary()
            self.assertEqual(recovered_summary["model"]["model_bundle"]["version"], 1)
            self.assertEqual(recovered_summary["model"]["model_bundle"]["optimizer_step"], 1)

    def test_model_bundle_replay_audit_rejects_mismatch_and_quarantines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=6,
                backlog=0,
                replay_audit=True,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": WORKLOAD_MODEL_BUNDLE_LM,
                    },
                ],
            )
            for _ in range(2):
                claim = store.claim_task("bad-bundle-miner", capabilities=self._model_bundle_capabilities())
                inner_result = run_model_bundle_inner_loop(
                    claim["workload_spec"],
                    inner_steps=claim["inner_steps"],
                )
                bad_delta = dict(inner_result["bundle_delta"])
                bad_values = list(bad_delta["values"])
                bad_values[0] += 0.001
                bad_delta["values"] = bad_values
                with self.assertRaises(ResultRejected) as raised:
                    store.complete_task(
                        claim["task_id"],
                        lease_token=claim["lease_token"],
                        attempt=claim["attempt"],
                        bundle_delta=bad_delta,
                        metrics=inner_result,
                    )
                self.assertEqual(raised.exception.validation["code"], "model_bundle_delta_replay_mismatch")

            summary = store.summary()
            self.assertEqual(summary["audit_rejections"], 2)
            self.assertTrue(
                summary["miner_workload_scores"]["bad-bundle-miner"][WORKLOAD_MODEL_BUNDLE_LM]["quarantined"]
            )
            with self.assertRaises(NoTaskAvailable):
                store.claim_task("bad-bundle-miner", capabilities=self._model_bundle_capabilities())

    def test_model_bundle_inference_is_read_only_and_ledgered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=1,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": WORKLOAD_MODEL_BUNDLE_INFER,
                    },
                ],
            )
            claim = store.claim_task("bundle-infer-miner", capabilities=self._model_bundle_inference_capabilities())
            self.assertEqual(claim["workload_spec"]["request_count"], 1)
            inner_result = run_model_bundle_inference(claim["workload_spec"])

            result = store.complete_task(
                claim["task_id"],
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                inference_result=inner_result["inference_result"],
                inference_results=inner_result["inference_results"],
                metrics=inner_result,
            )

            summary = store.summary()
            self.assertEqual(claim["workload_type"], WORKLOAD_MODEL_BUNDLE_INFER)
            self.assertEqual(result["workload_type"], WORKLOAD_MODEL_BUNDLE_INFER)
            self.assertFalse(result["model_updated"])
            self.assertFalse(result["model_bundle_updated"])
            self.assertEqual(summary["model"]["global_step"], 0)
            self.assertEqual(summary["model"]["model_bundle"]["version"], 0)
            self.assertEqual(summary["accepted_results"], 1)
            self.assertEqual(summary["model_updates"], 0)
            self.assertEqual(summary["model_bundle_updates"], 0)
            self.assertEqual(result["request_count"], 1)
            self.assertEqual(result["correct_count"], inner_result["correct_count"])
            row = store.result_ledger(status="accepted", workload_type=WORKLOAD_MODEL_BUNDLE_INFER)[0]
            self.assertEqual(row["workload_type"], WORKLOAD_MODEL_BUNDLE_INFER)
            self.assertFalse(row["model_updated"])
            self.assertFalse(row["model_bundle_updated"])
            self.assertEqual(row["validation"]["predicted_token_id"], result["predicted_token_id"])
            self.assertEqual(row["validation"]["request_count"], 1)
            self.assertEqual(row["validation"]["request_trace_count"], 1)
            self.assertFalse(row["validation"]["request_trace_truncated"])
            self.assertEqual(len(row["validation"]["request_trace"]), 1)
            self.assertIn("prompt", row["validation"]["request_trace"][0])
            self.assertIn("top_k", row["validation"]["request_trace"][0])
            self.assertEqual(
                summary["tasks"][0]["validation"]["request_trace"],
                row["validation"]["request_trace"],
            )
            self.assertNotIn("inference_result", json.dumps(summary["tasks"], sort_keys=True))
            self.assertNotIn("inference_results", json.dumps(summary["tasks"], sort_keys=True))

    def test_create_readonly_inference_task_enqueues_cpu_model_bundle_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=0)

            session = store.create_readonly_inference_task(request_count=3, scenario_id="route-baseline")
            claim = store.claim_task("admin-session-miner", capabilities=self._model_bundle_inference_capabilities())
            inner_result = run_model_bundle_inference(claim["workload_spec"])
            result = store.complete_task(
                claim["task_id"],
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                inference_result=inner_result["inference_result"],
                inference_results=inner_result["inference_results"],
                metrics=inner_result,
            )

            self.assertEqual(session["schema"], "inference_session_request_v1")
            self.assertEqual(session["task_id"], claim["task_id"])
            self.assertEqual(session["request_count"], 3)
            self.assertEqual(session["scenario_id"], "route-baseline")
            self.assertEqual(session["task_requirements"]["runtime"], "python-cli")
            self.assertEqual(session["task_requirements"]["backend"], "cpu")
            self.assertEqual(claim["workload_type"], WORKLOAD_MODEL_BUNDLE_INFER)
            self.assertEqual(claim["workload_spec"]["request_count"], 3)
            self.assertEqual(claim["workload_spec"]["scenario_id"], "route-baseline")
            self.assertFalse(result["model_updated"])
            self.assertFalse(result["model_bundle_updated"])
            self.assertEqual(result["scenario_id"], "route-baseline")
            row = store.result_ledger(task_id=session["task_id"])[0]
            self.assertEqual(row["task_id"], session["task_id"])
            self.assertEqual(row["validation"]["request_count"], 3)
            self.assertEqual(row["validation"]["scenario_id"], "route-baseline")
            self.assertEqual(row["session_metrics"]["request_count"], 3)
            self.assertEqual(row["session_metrics"]["scenario_id"], "route-baseline")
            self.assertFalse(row["model_updated"])

    def test_create_readonly_external_llm_task_enqueues_cpu_read_only_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=0)

            session = store.create_readonly_external_llm_task(request_count=3)
            claim = store.claim_task("admin-external-llm-miner", capabilities=self._external_llm_inference_capabilities())
            inner_result = run_mock_external_llm_inference(claim["workload_spec"])
            result = store.complete_task(
                claim["task_id"],
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                external_llm_result=inner_result["external_llm_result"],
                external_llm_results=inner_result["external_llm_results"],
                metrics=inner_result,
            )

            self.assertEqual(session["schema"], "inference_session_request_v1")
            self.assertEqual(session["task_id"], claim["task_id"])
            self.assertEqual(session["workload_type"], WORKLOAD_EXTERNAL_LLM_INFER)
            self.assertEqual(session["request_count"], 3)
            self.assertEqual(session["scenario_id"], "")
            self.assertEqual(session["task_requirements"]["runtime"], "python-cli")
            self.assertEqual(session["task_requirements"]["backend"], "cpu")
            self.assertEqual(claim["workload_type"], WORKLOAD_EXTERNAL_LLM_INFER)
            self.assertEqual(claim["workload_spec"]["request_count"], 3)
            self.assertTrue(result["accepted"])
            self.assertFalse(result["model_updated"])
            self.assertFalse(result["model_bundle_updated"])
            row = store.result_ledger(task_id=session["task_id"])[0]
            self.assertEqual(row["task_id"], session["task_id"])
            self.assertEqual(row["workload_type"], WORKLOAD_EXTERNAL_LLM_INFER)
            self.assertEqual(row["validation"]["request_count"], 3)
            self.assertEqual(row["session_metrics"]["completion_count"], 3)
            self.assertFalse(row["model_updated"])
            public = json.dumps(store.summary(), sort_keys=True)
            self.assertNotIn("external_llm_results", public)
            self.assertNotIn("output_text", public)

    def test_create_readonly_inference_task_rejects_non_cpu_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=0)

            with self.assertRaises(ValueError):
                store.create_readonly_inference_task(required_runtime="browser")
            with self.assertRaises(ValueError):
                store.create_readonly_inference_task(required_backend="cuda")
            with self.assertRaises(ValueError):
                store.create_readonly_inference_task(scenario_id="freeform-prompt")
            with self.assertRaises(ValueError):
                store.create_readonly_external_llm_task(required_runtime="browser")
            with self.assertRaises(ValueError):
                store.create_readonly_external_llm_task(required_backend="cuda")

    def test_model_bundle_inference_multi_request_session_is_read_only_and_ledgered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=4,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": WORKLOAD_MODEL_BUNDLE_INFER,
                    },
                ],
            )
            claim = store.claim_task("bundle-infer-session-miner", capabilities=self._model_bundle_inference_capabilities())
            inner_result = run_model_bundle_inference(claim["workload_spec"])

            result = store.complete_task(
                claim["task_id"],
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                inference_result=inner_result["inference_result"],
                inference_results=inner_result["inference_results"],
                metrics=inner_result,
            )

            summary = store.summary()
            self.assertEqual(claim["workload_spec"]["request_count"], 4)
            self.assertEqual(result["request_count"], 4)
            self.assertEqual(result["correct_count"], inner_result["correct_count"])
            self.assertEqual(result["accuracy"], inner_result["accuracy"])
            self.assertGreater(inner_result["requests_per_second"], 0.0)
            self.assertEqual(summary["model"]["global_step"], 0)
            self.assertEqual(summary["model"]["model_bundle"]["version"], 0)
            row = store.result_ledger(status="accepted", workload_type=WORKLOAD_MODEL_BUNDLE_INFER)[0]
            self.assertEqual(row["validation"]["request_count"], 4)
            self.assertEqual(row["validation"]["correct_count"], inner_result["correct_count"])
            self.assertEqual(row["validation"]["accuracy"], inner_result["accuracy"])
            self.assertEqual(row["validation"]["request_trace_count"], 4)
            self.assertFalse(row["validation"]["request_trace_truncated"])
            self.assertEqual(len(row["validation"]["request_trace"]), 4)
            self.assertEqual(row["session_metrics"]["request_count"], 4)
            self.assertEqual(row["session_metrics"]["correct_count"], inner_result["correct_count"])
            self.assertEqual(row["session_metrics"]["accuracy"], inner_result["accuracy"])
            self.assertGreaterEqual(row["session_metrics"]["elapsed_ms"], 0.0)
            self.assertGreater(row["session_metrics"]["requests_per_second"], 0.0)
            self.assertNotIn("inference_results", json.dumps(summary["tasks"], sort_keys=True))

    def test_model_bundle_inference_rejects_tampered_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=1,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": WORKLOAD_MODEL_BUNDLE_INFER,
                    },
                ],
            )
            claim = store.claim_task("bad-infer-miner", capabilities=self._model_bundle_inference_capabilities())
            inner_result = run_model_bundle_inference(claim["workload_spec"])
            bad_result = dict(inner_result["inference_result"])
            bad_result["predicted_token_id"] = (
                int(bad_result["predicted_token_id"]) + 1
            ) % claim["workload_spec"]["config"]["vocab_size"]

            with self.assertRaises(ResultRejected) as raised:
                store.complete_task(
                    claim["task_id"],
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    inference_result=bad_result,
                    metrics=inner_result,
                )

            self.assertEqual(raised.exception.validation["code"], "model_bundle_inference_prediction_mismatch")
            summary = store.summary()
            self.assertEqual(summary["rejected_results"], 1)
            row = store.result_ledger(status="rejected", workload_type=WORKLOAD_MODEL_BUNDLE_INFER)[0]
            self.assertEqual(row["validation"]["code"], "model_bundle_inference_prediction_mismatch")

    def test_model_bundle_inference_rejects_result_for_wrong_claim_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=2,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": WORKLOAD_MODEL_BUNDLE_INFER,
                    },
                ],
            )
            claim = store.claim_task("wrong-prompt-infer-miner", capabilities=self._model_bundle_inference_capabilities())
            alternate = dict(claim["workload_spec"])
            alternate_requests = list(alternate["requests"])
            alternate["requests"] = [alternate_requests[1], alternate_requests[0]]
            inner_result = run_model_bundle_inference(alternate)

            with self.assertRaises(ResultRejected) as raised:
                store.complete_task(
                    claim["task_id"],
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    inference_result=inner_result["inference_result"],
                    inference_results=inner_result["inference_results"],
                    metrics=inner_result,
                )

            self.assertEqual(raised.exception.validation["code"], "model_bundle_inference_request_id_mismatch")

    def test_model_bundle_inference_rejects_one_tampered_multi_request_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=3,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": WORKLOAD_MODEL_BUNDLE_INFER,
                    },
                ],
            )
            claim = store.claim_task("bad-infer-session-miner", capabilities=self._model_bundle_inference_capabilities())
            inner_result = run_model_bundle_inference(claim["workload_spec"])
            bad_results = [dict(row) for row in inner_result["inference_results"]]
            bad_results[-1]["predicted_token_id"] = (
                int(bad_results[-1]["predicted_token_id"]) + 1
            ) % claim["workload_spec"]["config"]["vocab_size"]

            with self.assertRaises(ResultRejected) as raised:
                store.complete_task(
                    claim["task_id"],
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    inference_result=bad_results[0],
                    inference_results=bad_results,
                    metrics=inner_result,
                )

            self.assertEqual(raised.exception.validation["code"], "model_bundle_inference_prediction_mismatch")

    def test_sharded_model_bundle_inference_runs_two_stage_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp, lease_seconds=5, inner_steps=1, backlog=0)
            session = store.create_sharded_inference_session(request_count=3, scenario_id="route-baseline")

            stage0_claim = store.claim_task("stage-0-miner", capabilities=self._sharded_model_bundle_inference_capabilities())
            self.assertEqual(stage0_claim["workload_type"], WORKLOAD_SHARDED_MODEL_BUNDLE_INFER)
            self.assertEqual(stage0_claim["workload_spec"]["stage_id"], 0)
            stage0_result = run_sharded_model_bundle_inference(stage0_claim["workload_spec"])
            accepted0 = store.complete_task(
                stage0_claim["task_id"],
                lease_token=stage0_claim["lease_token"],
                attempt=stage0_claim["attempt"],
                sharded_inference_result=stage0_result,
                metrics=stage0_result,
            )

            stage1_claim = store.claim_task("stage-1-miner", capabilities=self._sharded_model_bundle_inference_capabilities())
            self.assertEqual(stage1_claim["workload_spec"]["stage_id"], 1)
            self.assertEqual(stage1_claim["workload_spec"]["parent_task_id"], stage0_claim["task_id"])
            self.assertEqual(stage1_claim["workload_spec"]["activation_count"], 3)
            stage1_result = run_sharded_model_bundle_inference(stage1_claim["workload_spec"])
            accepted1 = store.complete_task(
                stage1_claim["task_id"],
                lease_token=stage1_claim["lease_token"],
                attempt=stage1_claim["attempt"],
                sharded_inference_result=stage1_result,
                metrics=stage1_result,
            )

            self.assertEqual(session["schema"], "sharded_inference_session_v1")
            self.assertEqual(accepted0["stage_id"], 0)
            self.assertEqual(accepted0["activation_count"], 3)
            self.assertTrue(accepted0["activation_transport_ready"])
            self.assertEqual(accepted1["stage_id"], 1)
            self.assertTrue(accepted1["baseline_match"])
            self.assertFalse(accepted1["model_updated"])
            rows = store.result_ledger(workload_type=WORKLOAD_SHARDED_MODEL_BUNDLE_INFER, session_id=session["session_id"])
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["stage_id"] for row in rows}, {0, 1})
            public = json.dumps(store.summary(), sort_keys=True)
            self.assertNotIn("activation_results", public)
            self.assertNotIn("logits", public)
            self.assertNotIn("sharded_inference_result", public)
            self.assertIn("activation_hashes", public)

    def test_sharded_model_bundle_inference_rejects_tampered_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp, lease_seconds=5, inner_steps=1, backlog=0)
            store.create_sharded_inference_session(request_count=2, scenario_id="route-baseline")
            claim = store.claim_task("bad-stage-0-miner", capabilities=self._sharded_model_bundle_inference_capabilities())
            inner_result = run_sharded_model_bundle_inference(claim["workload_spec"])
            inner_result["activation_results"][0]["logits"][0] += 1.0

            with self.assertRaises(ResultRejected) as raised:
                store.complete_task(
                    claim["task_id"],
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    sharded_inference_result=inner_result,
                    metrics=inner_result,
                )

            self.assertEqual(raised.exception.validation["code"], "sharded_model_bundle_activation_logits_mismatch")

    def test_micro_llm_sharded_inference_runs_two_stage_decode_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp, lease_seconds=5, inner_steps=1, backlog=0)
            session = store.create_micro_llm_sharded_inference_session(request_count=2, decode_steps=4)

            stage0_claim = store.claim_task("micro-stage-0-miner", capabilities=self._micro_llm_sharded_inference_capabilities())
            self.assertEqual(stage0_claim["workload_type"], WORKLOAD_MICRO_LLM_SHARDED_INFER)
            self.assertEqual(stage0_claim["workload_spec"]["stage_id"], 0)
            self.assertEqual(stage0_claim["workload_spec"]["decode_steps"], 4)
            stage0_result = run_micro_llm_sharded_inference(stage0_claim["workload_spec"])
            accepted0 = store.complete_task(
                stage0_claim["task_id"],
                lease_token=stage0_claim["lease_token"],
                attempt=stage0_claim["attempt"],
                sharded_inference_result=stage0_result,
                metrics=stage0_result,
            )

            stage1_claim = store.claim_task("micro-stage-1-miner", capabilities=self._micro_llm_sharded_inference_capabilities())
            self.assertEqual(stage1_claim["workload_spec"]["stage_id"], 1)
            self.assertEqual(stage1_claim["workload_spec"]["parent_task_id"], stage0_claim["task_id"])
            self.assertEqual(stage1_claim["workload_spec"]["activation_count"], 8)
            stage1_result = run_micro_llm_sharded_inference(stage1_claim["workload_spec"])
            accepted1 = store.complete_task(
                stage1_claim["task_id"],
                lease_token=stage1_claim["lease_token"],
                attempt=stage1_claim["attempt"],
                sharded_inference_result=stage1_result,
                metrics=stage1_result,
            )

            self.assertEqual(session["schema"], "micro_llm_sharded_session_v1")
            self.assertEqual(accepted0["stage_id"], 0)
            self.assertEqual(accepted0["activation_count"], 8)
            self.assertTrue(accepted0["activation_transport_ready"])
            self.assertEqual(accepted1["stage_id"], 1)
            self.assertTrue(accepted1["baseline_match"])
            self.assertTrue(accepted1["decoded_tokens_match"])
            self.assertEqual(accepted1["decode_steps"], 4)
            self.assertEqual(accepted1["generated_token_count"], 8)
            self.assertFalse(accepted1["model_updated"])
            self.assertFalse(accepted1["micro_transformer_updated"])
            rows = store.result_ledger(workload_type=WORKLOAD_MICRO_LLM_SHARDED_INFER, session_id=session["session_id"])
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["stage_id"] for row in rows}, {0, 1})
            public = json.dumps(store.summary(), sort_keys=True)
            self.assertNotIn("activation_results", public)
            self.assertNotIn("hidden_state", public)
            self.assertNotIn("sharded_inference_result", public)
            self.assertIn("decoded_tokens_match", public)
            self.assertIn("activation_hashes", public)

    def test_micro_llm_sharded_inference_rejects_tampered_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp, lease_seconds=5, inner_steps=1, backlog=0)
            store.create_micro_llm_sharded_inference_session(request_count=1, decode_steps=2)
            claim = store.claim_task("bad-micro-stage-0-miner", capabilities=self._micro_llm_sharded_inference_capabilities())
            inner_result = run_micro_llm_sharded_inference(claim["workload_spec"])
            inner_result["activation_results"][0]["hidden_state"][0] += 1.0

            with self.assertRaises(ResultRejected) as raised:
                store.complete_task(
                    claim["task_id"],
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    sharded_inference_result=inner_result,
                    metrics=inner_result,
                )

            self.assertEqual(raised.exception.validation["code"], "micro_llm_activation_hidden_mismatch")

    def test_micro_llm_sharded_stage_capabilities_route_split_miners(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp, lease_seconds=5, inner_steps=1, backlog=0)
            session = store.create_micro_llm_sharded_inference_session(request_count=2, decode_steps=2)

            with self.assertRaises(NoTaskAvailable):
                store.claim_task("stage1-too-early", capabilities=self._micro_llm_stage_capabilities("stage1"))

            stage0_claim = store.claim_task(
                "stage0-only",
                capabilities=self._micro_llm_stage_capabilities("stage0"),
            )
            self.assertEqual(stage0_claim["task_requirements"]["stage_capability"], MICRO_LLM_SHARDED_STAGE0_CAPABILITY)
            stage0_result = run_micro_llm_sharded_inference(stage0_claim["workload_spec"])
            store.complete_task(
                stage0_claim["task_id"],
                lease_token=stage0_claim["lease_token"],
                attempt=stage0_claim["attempt"],
                sharded_inference_result=stage0_result,
                metrics=stage0_result,
            )

            with self.assertRaises(NoTaskAvailable):
                store.claim_task("stage0-too-late", capabilities=self._micro_llm_stage_capabilities("stage0"))

            stage1_claim = store.claim_task(
                "stage1-only",
                capabilities=self._micro_llm_stage_capabilities("stage1"),
            )
            self.assertEqual(stage1_claim["task_requirements"]["stage_capability"], MICRO_LLM_SHARDED_STAGE1_CAPABILITY)
            self.assertEqual(stage1_claim["workload_spec"]["parent_task_id"], stage0_claim["task_id"])
            stage1_result = run_micro_llm_sharded_inference(stage1_claim["workload_spec"])
            accepted1 = store.complete_task(
                stage1_claim["task_id"],
                lease_token=stage1_claim["lease_token"],
                attempt=stage1_claim["attempt"],
                sharded_inference_result=stage1_result,
                metrics=stage1_result,
            )

            self.assertTrue(accepted1["baseline_match"])
            rows = store.result_ledger(workload_type=WORKLOAD_MICRO_LLM_SHARDED_INFER, session_id=session["session_id"])
            self.assertEqual({row["miner_id"] for row in rows}, {"stage0-only", "stage1-only"})

    def test_micro_llm_sharded_session_can_use_file_backed_artifact_and_prompt_texts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp) / "artifact"
            artifact = build_default_micro_llm_artifact(artifact_dir)
            store = StateStore(
                Path(tmp) / "state",
                lease_seconds=5,
                inner_steps=1,
                backlog=0,
                micro_llm_artifact=artifact_dir,
            )
            session = store.create_micro_llm_sharded_inference_session(
                request_count=2,
                decode_steps=3,
                prompt_texts=["arn", "ten"],
            )

            self.assertEqual(session["artifact_hash"], artifact["artifact_hash"])
            self.assertEqual(session["artifact_id"], artifact["artifact_id"])
            stage0_claim = store.claim_task("artifact-stage0", capabilities=self._micro_llm_stage_capabilities("stage0"))
            self.assertEqual(stage0_claim["workload_spec"]["artifact_hash"], artifact["artifact_hash"])
            self.assertEqual([row["prompt_token_ids"] for row in stage0_claim["workload_spec"]["requests"]], [[1, 7, 5], [9, 2, 5]])
            stage0_result = run_micro_llm_sharded_inference(stage0_claim["workload_spec"])
            accepted0 = store.complete_task(
                stage0_claim["task_id"],
                lease_token=stage0_claim["lease_token"],
                attempt=stage0_claim["attempt"],
                sharded_inference_result=stage0_result,
                metrics=stage0_result,
            )
            stage1_claim = store.claim_task("artifact-stage1", capabilities=self._micro_llm_stage_capabilities("stage1"))
            stage1_result = run_micro_llm_sharded_inference(stage1_claim["workload_spec"])
            accepted1 = store.complete_task(
                stage1_claim["task_id"],
                lease_token=stage1_claim["lease_token"],
                attempt=stage1_claim["attempt"],
                sharded_inference_result=stage1_result,
                metrics=stage1_result,
            )

            self.assertEqual(accepted0["artifact_hash"], artifact["artifact_hash"])
            self.assertEqual(accepted1["artifact_hash"], artifact["artifact_hash"])
            self.assertEqual(accepted1["artifact_id"], artifact["artifact_id"])
            self.assertTrue(accepted1["baseline_match"])
            rows = store.result_ledger(workload_type=WORKLOAD_MICRO_LLM_SHARDED_INFER, session_id=session["session_id"])
            self.assertEqual({row["validation"]["artifact_hash"] for row in rows}, {artifact["artifact_hash"]})

    def test_external_llm_inference_is_read_only_ledgered_and_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=3,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": WORKLOAD_EXTERNAL_LLM_INFER,
                    },
                ],
            )
            claim = store.claim_task("external-llm-miner", capabilities=self._external_llm_inference_capabilities())
            self.assertEqual(claim["workload_type"], WORKLOAD_EXTERNAL_LLM_INFER)
            self.assertEqual(claim["workload_spec"]["request_count"], 3)
            self.assertEqual(claim["weights"], [])
            raw_prompts = [row["prompt"] for row in claim["workload_spec"]["requests"]]
            claim_events = json.dumps(store.event_tail(limit=20), sort_keys=True)
            for prompt in raw_prompts:
                self.assertNotIn(prompt, claim_events)
            self.assertIn("<redacted>", claim_events)
            self.assertIn("prompt_hash", claim_events)
            inner_result = run_mock_external_llm_inference(claim["workload_spec"])

            result = store.complete_task(
                claim["task_id"],
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                external_llm_result=inner_result["external_llm_result"],
                external_llm_results=inner_result["external_llm_results"],
                metrics=inner_result,
            )

            summary = store.summary()
            self.assertFalse(result["model_updated"])
            self.assertFalse(result["model_bundle_updated"])
            self.assertEqual(result["workload_type"], WORKLOAD_EXTERNAL_LLM_INFER)
            self.assertEqual(result["request_count"], 3)
            self.assertEqual(result["completion_count"], 3)
            self.assertEqual(result["adapter_kind"], "mock")
            self.assertGreater(result["output_chars"], 0)
            self.assertEqual(summary["model"]["global_step"], 0)
            self.assertEqual(summary["model_updates"], 0)
            row = store.result_ledger(status="accepted", workload_type=WORKLOAD_EXTERNAL_LLM_INFER)[0]
            self.assertEqual(row["validation"]["request_count"], 3)
            self.assertEqual(row["validation"]["completion_count"], 3)
            self.assertEqual(row["validation"]["adapter_kind"], "mock")
            self.assertEqual(row["validation"]["output_preview"], "<redacted>")
            self.assertEqual(row["session_metrics"]["request_count"], 3)
            self.assertEqual(row["session_metrics"]["completion_count"], 3)
            self.assertEqual(row["session_metrics"]["adapter_kind"], "mock")
            public_tasks = json.dumps(summary["tasks"], sort_keys=True)
            self.assertNotIn("external_llm_result", public_tasks)
            self.assertNotIn("external_llm_results", public_tasks)
            self.assertNotIn("output_text", public_tasks)
            for prompt in raw_prompts:
                self.assertNotIn(prompt, public_tasks)
            self.assertIn("<redacted>", public_tasks)
            self.assertIn("prompt_hash", public_tasks)
            ledger_payload = json.dumps(row, sort_keys=True)
            for prompt in raw_prompts:
                self.assertNotIn(prompt, ledger_payload)
            completed_events = json.dumps(store.event_tail(limit=20), sort_keys=True)
            for prompt in raw_prompts:
                self.assertNotIn(prompt, completed_events)

    def test_external_llm_inference_rejects_wrong_prompt_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=1,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": WORKLOAD_EXTERNAL_LLM_INFER,
                    },
                ],
            )
            claim = store.claim_task("external-bad-miner", capabilities=self._external_llm_inference_capabilities())
            inner_result = run_mock_external_llm_inference(claim["workload_spec"])
            bad_result = dict(inner_result["external_llm_result"])
            bad_result["prompt_hash"] = "sha256:wrong"

            with self.assertRaises(ResultRejected) as raised:
                store.complete_task(
                    claim["task_id"],
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    external_llm_result=bad_result,
                    external_llm_results=[bad_result],
                    metrics=inner_result,
                )

            self.assertEqual(raised.exception.validation["code"], "external_llm_prompt_hash_mismatch")
            row = store.result_ledger(status="rejected", workload_type=WORKLOAD_EXTERNAL_LLM_INFER)[0]
            self.assertEqual(row["validation"]["code"], "external_llm_prompt_hash_mismatch")
            self.assertEqual(raised.exception.validation["request_index"], 0)

    def test_backlog_allows_async_claims_and_records_staleness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=3)
            claims = [
                store.claim_task(f"miner-{index}")
                for index in range(3)
            ]

            self.assertEqual(len({claim["task_id"] for claim in claims}), 3)
            self.assertEqual([claim["model_version"] for claim in claims], [0, 0, 0])

            results = [
                self._complete_claim(store, claim, f"miner-{index}")
                for index, claim in enumerate(claims)
            ]
            self.assertEqual([result["staleness"] for result in results], [0, 1, 2])

            summary = store.summary()
            self.assertEqual(summary["model"]["global_step"], 3)
            self.assertEqual(summary["model"]["optimizer_step"], 3)
            self.assertEqual(summary["accepted_results"], 3)
            self.assertEqual(summary["max_staleness"], 2)
            self.assertAlmostEqual(summary["avg_staleness"], 1.0)
            self.assertEqual(summary["task_counts"]["completed"], 3)
            self.assertEqual(summary["task_counts"]["queued"], 3)

            completed = [
                task for task in summary["tasks"]
                if task["status"] == "completed"
            ]
            self.assertEqual(
                [task["base_model_version"] for task in completed],
                [0, 0, 0],
            )
            self.assertEqual(
                [task["result_model_version"] for task in completed],
                [1, 2, 3],
            )
            self.assertEqual(
                [task["staleness"] for task in completed],
                [0, 1, 2],
            )

    def test_expired_task_requeues_and_rejects_stale_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp, lease_seconds=0.05, inner_steps=10, backlog=1)
            stale = store.claim_task("miner-a")
            time.sleep(0.08)

            expired = store.reap_expired()
            fresh = store.claim_task("miner-b")

            self.assertEqual(expired, [stale["task_id"]])
            self.assertEqual(fresh["task_id"], stale["task_id"])
            self.assertEqual(fresh["attempt"], stale["attempt"] + 1)

            gradient = compute_pseudo_gradient(
                stale["weights"],
                task_id=stale["task_id"],
                miner_id="miner-a",
                inner_steps=stale["inner_steps"],
            )
            with self.assertRaises(LeaseConflict):
                store.complete_task(
                    stale["task_id"],
                    lease_token=stale["lease_token"],
                    attempt=stale["attempt"],
                    pseudo_gradient=gradient,
                )

    def test_restart_restores_model_and_requeues_inflight_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=1)
            claim = store.claim_task("miner-a")
            inner_result = run_inner_loop(
                claim["weights"],
                task_id=claim["task_id"],
                miner_id="miner-a",
                model_version=claim["model_version"],
                inner_steps=claim["inner_steps"],
            )
            store.complete_task(
                claim["task_id"],
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                local_delta=inner_result["local_delta"],
                metrics=inner_result,
            )

            in_flight = store.claim_task("miner-b")
            recovered = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=1)
            summary = recovered.summary()

            self.assertEqual(summary["model"]["global_step"], 1)
            self.assertEqual(summary["model"]["optimizer_step"], 1)
            self.assertIn("outer_velocity", summary["model"])
            self.assertEqual(summary["model"]["outer_optimizer_contract"]["optimizer_step"], 1)
            self.assertEqual(summary["accepted_results"], 1)
            self.assertEqual(summary["task_counts"]["queued"], 1)
            self.assertEqual(summary["task_counts"]["leased"], 0)

            reclaimed = recovered.claim_task("miner-c")
            self.assertEqual(reclaimed["task_id"], in_flight["task_id"])
            self.assertEqual(reclaimed["attempt"], in_flight["attempt"] + 1)

    def test_quality_rejection_does_not_update_model_and_updates_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=1)
            claim = store.claim_task("bad-miner")

            with self.assertRaises(ResultRejected) as raised:
                store.complete_task(
                    claim["task_id"],
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    local_delta=[1000.0, 1000.0, 1000.0],
                )

            self.assertEqual(raised.exception.validation["code"], "delta_norm_too_large")
            summary = store.summary()
            self.assertEqual(summary["model"]["global_step"], 0)
            self.assertEqual(summary["model"]["optimizer_step"], 0)
            self.assertEqual(summary["accepted_results"], 0)
            self.assertEqual(summary["rejected_results"], 1)
            self.assertEqual(summary["task_counts"]["rejected"], 1)
            self.assertEqual(summary["task_counts"]["queued"], 1)
            self.assertFalse((Path(tmp) / "global_model.json").exists())
            self.assertEqual(summary["last_rejected"]["task_id"], claim["task_id"])
            self.assertEqual(summary["last_rejected"]["validation"]["code"], "delta_norm_too_large")
            self.assertEqual(summary["miner_scores"]["bad-miner"]["rejected"], 1)
            self.assertLess(summary["miner_scores"]["bad-miner"]["score"], 0)

    def test_rejected_result_replays_from_event_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=1)
            claim = store.claim_task("bad-miner")
            with self.assertRaises(ResultRejected):
                store.complete_task(
                    claim["task_id"],
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    local_delta=[1000.0, 1000.0, 1000.0],
                )

            recovered = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=1)
            summary = recovered.summary()

            self.assertEqual(summary["model"]["global_step"], 0)
            self.assertEqual(summary["rejected_results"], 1)
            self.assertEqual(summary["miner_scores"]["bad-miner"]["rejected"], 1)

    def test_miner_profiles_summarize_capabilities_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=1)
            claim = store.claim_task(
                "profile-miner",
                capabilities={
                    "runtime": "python-cli",
                    "backend": "cpu",
                    "hardware_profile": {
                        "os": "Linux",
                        "platform": "Linux",
                        "machine": "x86_64",
                        "processor": "x86_64",
                        "cpu_count": 8,
                        "python_version": "3.12.0",
                    },
                    "supports_training_spec": True,
                },
            )
            heartbeat = store.heartbeat(
                claim["task_id"],
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                runtime_status={"phase": "training", "pid": 123},
            )
            self.assertGreater(heartbeat["lease_expires_at"], claim["lease_expires_at"])
            inner_result = run_inner_loop(
                claim["weights"],
                task_id=claim["task_id"],
                miner_id="profile-miner",
                model_version=claim["model_version"],
                inner_steps=claim["inner_steps"],
                training_spec=claim["training_spec"],
            )
            store.complete_task(
                claim["task_id"],
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                local_delta=inner_result["local_delta"],
                metrics={**inner_result, "elapsed_ms": 12.5},
            )

            profile = store.summary()["miner_profiles"]["profile-miner"]
            self.assertEqual(profile["runtime"], "python-cli")
            self.assertEqual(profile["backend"], "cpu")
            self.assertEqual(profile["last_capabilities"]["hardware_profile"]["cpu_count"], 8)
            self.assertEqual(profile["last_capabilities"]["hardware_profile"]["os"], "Linux")
            self.assertEqual(profile["accepted"], 1)
            self.assertEqual(profile["rejected"], 0)
            self.assertEqual(profile["avg_worker_elapsed_ms"], 12.5)
            self.assertEqual(profile["last_runtime_status"]["phase"], "training")

    def test_miner_profiles_count_rejections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=1)
            claim = store.claim_task(
                "bad-profile-miner",
                capabilities={"runtime": "browser", "backend": "js-worker"},
            )
            with self.assertRaises(ResultRejected):
                store.complete_task(
                    claim["task_id"],
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    local_delta=[1000.0, 1000.0, 1000.0],
                )

            profile = store.summary()["miner_profiles"]["bad-profile-miner"]
            self.assertEqual(profile["runtime"], "browser")
            self.assertEqual(profile["backend"], "js-worker")
            self.assertEqual(profile["accepted"], 0)
            self.assertEqual(profile["rejected"], 1)

    def test_capability_scheduler_skips_incompatible_older_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=0)
            browser_task_id = store._create_task(  # noqa: SLF001 - scheduler fixture
                required_runtime="browser",
                required_backend="js-worker",
            )
            python_task_id = store._create_task(  # noqa: SLF001 - scheduler fixture
                required_runtime="python-cli",
                required_backend="cpu",
            )

            claim = store.claim_task(
                "python-miner",
                capabilities={
                    "runtime": "python-cli",
                    "backend": "cpu",
                    "protocol_version": "runtime_contract_v1",
                },
            )

            self.assertNotEqual(browser_task_id, python_task_id)
            self.assertEqual(claim["task_id"], python_task_id)
            self.assertEqual(claim["task_requirements"]["runtime"], "python-cli")
            summary = store.summary()
            self.assertEqual(summary["task_counts"]["queued"], 1)
            self.assertEqual(summary["task_counts"]["leased"], 1)
            self.assertEqual(summary["incompatible_claims"], 0)

    def test_task_lanes_create_separate_backlogs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                task_lanes=[
                    {"runtime": "python-cli", "backend": "cpu", "count": 2},
                    {"runtime": "browser", "backend": "js-worker", "count": 1},
                ],
            )

            summary = store.summary()

            self.assertEqual(summary["task_counts"]["queued"], 3)
            self.assertEqual(summary["task_lanes"][0]["workload_type"], "diloco_train")
            self.assertEqual(
                summary["task_counts_by_requirement"]["python-cli/cpu/runtime_contract_v1"]["queued"],
                2,
            )
            self.assertEqual(
                summary["task_counts_by_requirement"]["browser/js-worker/runtime_contract_v1"]["queued"],
                1,
            )
            self.assertEqual(
                summary["task_counts_by_lane"]["browser/js-worker/runtime_contract_v1/diloco_train"]["queued"],
                1,
            )

    def test_task_lanes_route_miners_and_replenish_completed_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                task_lanes=[
                    {"runtime": "python-cli", "backend": "cpu", "count": 1},
                    {"runtime": "browser", "backend": "js-worker", "count": 1},
                ],
            )
            python_claim = store.claim_task(
                "python-lane-miner",
                capabilities={
                    "runtime": "python-cli",
                    "backend": "cpu",
                    "protocol_version": "runtime_contract_v1",
                },
            )
            browser_claim = store.claim_task(
                "browser-lane-miner",
                capabilities={
                    "runtime": "browser",
                    "backend": "js-worker",
                    "protocol_version": "runtime_contract_v1",
                },
            )

            self.assertEqual(python_claim["task_requirements"]["runtime"], "python-cli")
            self.assertEqual(browser_claim["task_requirements"]["runtime"], "browser")
            self.assertEqual(python_claim["workload_type"], "diloco_train")

            self._complete_claim(store, python_claim, "python-lane-miner")
            summary = store.summary()

            self.assertEqual(
                summary["task_counts_by_requirement"]["python-cli/cpu/runtime_contract_v1"]["completed"],
                1,
            )
            self.assertEqual(
                summary["task_counts_by_requirement"]["python-cli/cpu/runtime_contract_v1"]["queued"],
                1,
            )
            self.assertEqual(
                summary["task_counts_by_requirement"]["browser/js-worker/runtime_contract_v1"]["leased"],
                1,
            )

    def test_task_lanes_replenish_rejected_lane(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                task_lanes=[
                    {"runtime": "browser", "backend": "js-worker", "count": 1},
                ],
            )
            claim = store.claim_task(
                "bad-browser-lane-miner",
                capabilities={
                    "runtime": "browser",
                    "backend": "js-worker",
                    "protocol_version": "runtime_contract_v1",
                },
            )

            with self.assertRaises(ResultRejected):
                store.complete_task(
                    claim["task_id"],
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    local_delta=[1000.0, 1000.0, 1000.0],
                )
            summary = store.summary()

            self.assertEqual(
                summary["task_counts_by_requirement"]["browser/js-worker/runtime_contract_v1"]["rejected"],
                1,
            )
            self.assertEqual(
                summary["task_counts_by_requirement"]["browser/js-worker/runtime_contract_v1"]["queued"],
                1,
            )

    def test_browser_probe_completes_without_model_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "browser",
                        "backend": "js-worker",
                        "count": 1,
                        "workload_type": "browser_probe",
                    },
                ],
            )
            claim = store.claim_task(
                "probe-browser",
                capabilities={
                    "runtime": "browser",
                    "backend": "js-worker",
                    "protocol_version": "runtime_contract_v1",
                    "supported_workloads": ["diloco_train", "browser_probe"],
                },
            )

            result = store.complete_task(
                claim["task_id"],
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                probe_result={
                    "verified": True,
                    "ops": 1024,
                    "elapsed_ms": 3.5,
                    "hash": "abcd1234",
                    "gops": 0.1,
                },
            )
            summary = store.summary()

            self.assertTrue(result["accepted"])
            self.assertFalse(result["model_updated"])
            self.assertEqual(claim["workload_type"], "browser_probe")
            self.assertEqual(claim["workload_spec"]["buffer_pattern"], "sin_mod_v1")
            self.assertEqual(summary["accepted_results"], 1)
            self.assertEqual(summary["model_updates"], 0)
            self.assertEqual(summary["model"]["global_step"], 0)
            self.assertEqual(
                summary["task_counts_by_lane"]["browser/js-worker/runtime_contract_v1/browser_probe"]["completed"],
                1,
            )
            self.assertEqual(
                summary["task_counts_by_lane"]["browser/js-worker/runtime_contract_v1/browser_probe"]["queued"],
                1,
            )

    def test_browser_probe_rejects_invalid_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "browser",
                        "backend": "js-worker",
                        "count": 1,
                        "workload_type": "browser_probe",
                    },
                ],
            )
            claim = store.claim_task(
                "bad-probe-browser",
                capabilities={
                    "runtime": "browser",
                    "backend": "js-worker",
                    "protocol_version": "runtime_contract_v1",
                    "supported_workloads": ["browser_probe"],
                },
            )

            with self.assertRaises(ResultRejected) as raised:
                store.complete_task(
                    claim["task_id"],
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    probe_result={"verified": False, "ops": 0, "elapsed_ms": 0, "hash": ""},
                )
            summary = store.summary()

            self.assertEqual(raised.exception.validation["code"], "probe_result_invalid")
            self.assertEqual(summary["accepted_results"], 0)
            self.assertEqual(summary["rejected_results"], 1)
            self.assertEqual(summary["model_updates"], 0)
            workload_score = summary["miner_workload_scores"]["bad-probe-browser"]["browser_probe"]
            self.assertEqual(workload_score["rejected"], 1)
            self.assertFalse(workload_score["quarantined"])
            self.assertEqual(workload_score["last_rejection_code"], "probe_result_invalid")

    def test_browser_probe_requires_supported_workload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "browser",
                        "backend": "js-worker",
                        "count": 1,
                        "workload_type": "browser_probe",
                    },
                ],
            )

            with self.assertRaises(NoTaskAvailable):
                store.claim_task(
                    "legacy-browser",
                    capabilities={
                        "runtime": "browser",
                        "backend": "js-worker",
                        "protocol_version": "runtime_contract_v1",
                    },
                )

    def test_cpu_lora_mock_updates_adapter_without_dense_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=40,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": "cpu_lora_mock",
                    },
                ],
            )
            claim = store.claim_task(
                "lora-miner",
                capabilities={
                    "runtime": "python-cli",
                    "backend": "cpu",
                    "protocol_version": "runtime_contract_v1",
                    "supported_workloads": ["cpu_lora_mock"],
                },
            )
            from crowdtensor.lora_mock import run_lora_inner_loop
            inner_result = run_lora_inner_loop(claim["workload_spec"], inner_steps=claim["inner_steps"])

            result = store.complete_task(
                claim["task_id"],
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                adapter_delta=inner_result["adapter_delta"],
                metrics=inner_result,
            )
            summary = store.summary()

            self.assertEqual(claim["workload_type"], "cpu_lora_mock")
            self.assertEqual(claim["workload_spec"]["type"], "cpu_lora_mock")
            self.assertTrue(result["adapter_updated"])
            self.assertFalse(result["model_updated"])
            self.assertEqual(summary["accepted_results"], 1)
            self.assertEqual(summary["adapter_updates"], 1)
            self.assertEqual(summary["model_updates"], 0)
            self.assertEqual(summary["model"]["global_step"], 0)
            self.assertEqual(summary["model"]["adapter_step"], 1)
            self.assertNotEqual(summary["model"]["lora_adapter"]["values"], [0.0, 0.0, 0.0])

    def test_replay_audit_accepts_diloco_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                replay_audit=True,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": "diloco_train",
                    },
                ],
            )
            claim = store.claim_task("audit-diloco", capabilities=self._python_capabilities())
            inner_result = run_inner_loop(
                claim["weights"],
                task_id=claim["task_id"],
                miner_id="audit-diloco",
                model_version=claim["model_version"],
                inner_steps=claim["inner_steps"],
                training_spec=claim["training_spec"],
            )

            result = store.complete_task(
                claim["task_id"],
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                local_delta=inner_result["local_delta"],
                metrics=inner_result,
            )
            summary = store.summary()

            self.assertEqual(claim["audit_mode"], "replay")
            self.assertEqual(result["global_step"], 1)
            self.assertEqual(summary["audit_results"], 1)
            self.assertEqual(summary["audit_rejections"], 0)
            self.assertEqual(summary["last_completed"]["validation"]["audit_code"], "ok")

    def test_replay_audit_accepts_sign_compressed_diloco_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                replay_audit=True,
                delta_format=DELTA_FORMAT_SIGN_COMPRESSED,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": "diloco_train",
                    },
                ],
            )
            claim = store.claim_task(
                "audit-compressed-diloco",
                capabilities={
                    **self._python_capabilities(),
                    "supported_delta_formats": [DELTA_FORMAT_SIGN_COMPRESSED],
                },
            )
            inner_result = run_inner_loop(
                claim["weights"],
                task_id=claim["task_id"],
                miner_id="audit-compressed-diloco",
                model_version=claim["model_version"],
                inner_steps=claim["inner_steps"],
                training_spec=claim["training_spec"],
            )

            result = store.complete_task(
                claim["task_id"],
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                compressed_delta=compress_sign_delta(inner_result["local_delta"]),
                metrics={"transport": "sign_compressed"},
            )
            summary = store.summary()
            row = store.result_ledger(status="accepted")[0]

            self.assertEqual(result["optimizer"]["delta_format"], "sign_compressed")
            self.assertEqual(summary["audit_results"], 1)
            self.assertEqual(summary["audit_rejections"], 0)
            self.assertEqual(summary["last_completed"]["validation"]["audit_code"], "ok")
            self.assertEqual(summary["last_completed"]["validation"]["audit_delta_format"], "sign_compressed")
            self.assertEqual(row["audit"]["audit_delta_format"], "sign_compressed")
            self.assertIn("audit_expected_delta_norm", row["audit"])

    def test_nesterov_sign_compressed_result_passes_replay_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                replay_audit=True,
                outer_optimizer=OPTIMIZER_DILOCO_NESTEROV,
                delta_format=DELTA_FORMAT_SIGN_COMPRESSED,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": "diloco_train",
                    },
                ],
            )
            claim = store.claim_task(
                "nesterov-compressed",
                capabilities={
                    **self._python_capabilities(),
                    "supported_delta_formats": [DELTA_FORMAT_SIGN_COMPRESSED],
                },
            )
            inner_result = run_inner_loop(
                claim["weights"],
                task_id=claim["task_id"],
                miner_id="nesterov-compressed",
                model_version=claim["model_version"],
                inner_steps=claim["inner_steps"],
                training_spec=claim["training_spec"],
            )

            result = store.complete_task(
                claim["task_id"],
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                compressed_delta=compress_sign_delta(inner_result["local_delta"]),
                metrics={"delta_format": "sign_compressed"},
            )
            summary = store.summary()

            self.assertEqual(claim["optimizer_spec"]["optimizer_type"], OPTIMIZER_DILOCO_NESTEROV)
            self.assertEqual(result["optimizer"]["optimizer_type"], OPTIMIZER_DILOCO_NESTEROV)
            self.assertEqual(result["optimizer"]["delta_format"], "sign_compressed")
            self.assertEqual(summary["audit_results"], 1)
            self.assertEqual(summary["audit_rejections"], 0)
            self.assertEqual(summary["last_completed"]["validation"]["audit_code"], "ok")

    def test_replay_audit_rejects_wrong_sign_compressed_diloco_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                replay_audit=True,
                delta_format=DELTA_FORMAT_SIGN_COMPRESSED,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": "diloco_train",
                    },
                ],
            )
            claim = store.claim_task(
                "bad-compressed-diloco",
                capabilities={
                    **self._python_capabilities(),
                    "supported_delta_formats": [DELTA_FORMAT_SIGN_COMPRESSED],
                },
            )
            inner_result = run_inner_loop(
                claim["weights"],
                task_id=claim["task_id"],
                miner_id="bad-compressed-diloco",
                model_version=claim["model_version"],
                inner_steps=claim["inner_steps"],
                training_spec=claim["training_spec"],
            )
            compressed = compress_sign_delta(inner_result["local_delta"])
            compressed["signs"] = [-sign for sign in compressed["signs"]]

            with self.assertRaises(ResultRejected) as raised:
                store.complete_task(
                    claim["task_id"],
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    compressed_delta=compressed,
                )
            summary = store.summary()

            self.assertEqual(raised.exception.validation["code"], "local_delta_replay_mismatch")
            self.assertEqual(raised.exception.validation["audit_delta_format"], "sign_compressed")
            self.assertEqual(summary["audit_results"], 1)
            self.assertEqual(summary["audit_rejections"], 1)

    def test_replay_audit_rejects_sign_compressed_error_feedback_diloco_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                replay_audit=True,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": "diloco_train",
                    },
                ],
            )
            claim = store.claim_task("audit-compressed-ef-diloco", capabilities=self._python_capabilities())
            inner_result = run_inner_loop(
                claim["weights"],
                task_id=claim["task_id"],
                miner_id="audit-compressed-ef-diloco",
                model_version=claim["model_version"],
                inner_steps=claim["inner_steps"],
                training_spec=claim["training_spec"],
            )
            compressed, _residual = compress_sign_delta_with_error_feedback(inner_result["local_delta"])

            with self.assertRaises(ResultRejected) as raised:
                store.complete_task(
                    claim["task_id"],
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    compressed_delta=compressed,
                )
            summary = store.summary()
            row = store.result_ledger(status="rejected")[0]

            self.assertEqual(raised.exception.validation["code"], "local_delta_replay_mismatch")
            self.assertEqual(raised.exception.validation["audit_code"], "error_feedback_replay_unsupported")
            self.assertEqual(raised.exception.validation["audit_delta_format"], DELTA_FORMAT_SIGN_COMPRESSED_EF)
            self.assertEqual(summary["audit_results"], 1)
            self.assertEqual(summary["audit_rejections"], 1)
            self.assertEqual(row["audit"]["audit_code"], "error_feedback_replay_unsupported")

    def test_replay_audit_rejects_small_wrong_diloco_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                replay_audit=True,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": "diloco_train",
                    },
                ],
            )
            claim = store.claim_task("bad-audit-diloco", capabilities=self._python_capabilities())

            with self.assertRaises(ResultRejected) as raised:
                store.complete_task(
                    claim["task_id"],
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    local_delta=[0.0 for _ in claim["weights"]],
                )
            summary = store.summary()

            self.assertEqual(raised.exception.validation["code"], "local_delta_replay_mismatch")
            self.assertEqual(raised.exception.validation["audit_code"], "local_delta_replay_mismatch")
            self.assertEqual(summary["model"]["global_step"], 0)
            self.assertEqual(summary["audit_results"], 1)
            self.assertEqual(summary["audit_rejections"], 1)
            self.assertEqual(summary["last_rejected"]["validation"]["audit_code"], "local_delta_replay_mismatch")

    def test_replay_audit_accepts_lora_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=20,
                backlog=0,
                replay_audit=True,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": "cpu_lora_mock",
                    },
                ],
            )
            claim = store.claim_task("audit-lora", capabilities=self._lora_capabilities())
            from crowdtensor.lora_mock import run_lora_inner_loop
            inner_result = run_lora_inner_loop(claim["workload_spec"], inner_steps=claim["inner_steps"])

            result = store.complete_task(
                claim["task_id"],
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                adapter_delta=inner_result["adapter_delta"],
                metrics=inner_result,
            )
            summary = store.summary()

            self.assertTrue(result["adapter_updated"])
            self.assertEqual(claim["audit_mode"], "replay")
            self.assertEqual(summary["audit_results"], 1)
            self.assertEqual(summary["audit_rejections"], 0)
            self.assertEqual(summary["last_completed"]["validation"]["audit_code"], "ok")

    def test_replay_audit_rejects_lora_mismatch_and_quarantines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=20,
                backlog=0,
                replay_audit=True,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": "cpu_lora_mock",
                    },
                ],
            )
            for _ in range(2):
                claim = store.claim_task("bad-audit-lora", capabilities=self._lora_capabilities())
                with self.assertRaises(ResultRejected) as raised:
                    store.complete_task(
                        claim["task_id"],
                        lease_token=claim["lease_token"],
                        attempt=claim["attempt"],
                        adapter_delta={"values": [0.0 for _ in claim["weights"]]},
                    )
                self.assertEqual(raised.exception.validation["code"], "adapter_delta_replay_mismatch")

            with self.assertRaises(NoTaskAvailable) as blocked:
                store.claim_task("bad-audit-lora", capabilities=self._lora_capabilities())
            summary = store.summary()

            self.assertEqual(str(blocked.exception), "miner quarantined for workload")
            self.assertEqual(summary["audit_results"], 2)
            self.assertEqual(summary["audit_rejections"], 2)
            self.assertTrue(
                summary["miner_workload_scores"]["bad-audit-lora"]["cpu_lora_mock"]["quarantined"]
            )

    def test_replay_audit_rejection_replays_from_event_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                replay_audit=True,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": "diloco_train",
                    },
                ],
            )
            claim = store.claim_task("replay-audit-bad", capabilities=self._python_capabilities())
            with self.assertRaises(ResultRejected):
                store.complete_task(
                    claim["task_id"],
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    local_delta=[0.0 for _ in claim["weights"]],
                )

            recovered = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                replay_audit=True,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": "diloco_train",
                    },
                ],
            )
            summary = recovered.summary()

            self.assertEqual(summary["audit_results"], 1)
            self.assertEqual(summary["audit_rejections"], 1)
            self.assertEqual(summary["last_rejected"]["validation"]["audit_code"], "local_delta_replay_mismatch")

    def test_cpu_lora_mock_rejects_invalid_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": "cpu_lora_mock",
                    },
                ],
            )
            claim = store.claim_task(
                "bad-lora-miner",
                capabilities={
                    "runtime": "python-cli",
                    "backend": "cpu",
                    "protocol_version": "runtime_contract_v1",
                    "supported_workloads": ["cpu_lora_mock"],
                },
            )

            with self.assertRaises(ResultRejected) as raised:
                store.complete_task(
                    claim["task_id"],
                    lease_token=claim["lease_token"],
                    attempt=claim["attempt"],
                    adapter_delta={"values": [100.0, 0.0, 0.0]},
                )
            summary = store.summary()

            self.assertEqual(raised.exception.validation["code"], "adapter_delta_norm_too_large")
            self.assertEqual(summary["adapter_updates"], 0)
            self.assertEqual(summary["rejected_results"], 1)

    def test_cpu_lora_mock_requires_supported_workload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": "cpu_lora_mock",
                    },
                ],
            )

            with self.assertRaises(NoTaskAvailable):
                store.claim_task(
                    "old-python-miner",
                    capabilities={
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "protocol_version": "runtime_contract_v1",
                        "supported_workloads": ["diloco_train"],
                    },
                )

    def test_cpu_lora_mock_replays_adapter_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=20,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": "cpu_lora_mock",
                    },
                ],
            )
            claim = store.claim_task(
                "replay-lora-miner",
                capabilities={
                    "runtime": "python-cli",
                    "backend": "cpu",
                    "protocol_version": "runtime_contract_v1",
                    "supported_workloads": ["cpu_lora_mock"],
                },
            )
            from crowdtensor.lora_mock import run_lora_inner_loop
            inner_result = run_lora_inner_loop(claim["workload_spec"], inner_steps=claim["inner_steps"])
            store.complete_task(
                claim["task_id"],
                lease_token=claim["lease_token"],
                attempt=claim["attempt"],
                adapter_delta=inner_result["adapter_delta"],
                metrics=inner_result,
            )

            recovered = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=20,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": "cpu_lora_mock",
                    },
                ],
            )
            summary = recovered.summary()

            self.assertEqual(summary["model"]["adapter_step"], 1)
            self.assertEqual(summary["adapter_updates"], 1)

    def test_miner_workload_score_quarantines_lora_after_two_rejections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": "cpu_lora_mock",
                    },
                ],
            )

            for _ in range(2):
                claim = store.claim_task("bad-lora-repeat", capabilities=self._lora_capabilities())
                with self.assertRaises(ResultRejected):
                    store.complete_task(
                        claim["task_id"],
                        lease_token=claim["lease_token"],
                        attempt=claim["attempt"],
                        adapter_delta={"values": [100.0, 0.0, 0.0]},
                    )

            summary = store.summary()
            workload_score = summary["miner_workload_scores"]["bad-lora-repeat"]["cpu_lora_mock"]
            self.assertEqual(workload_score["rejected"], 2)
            self.assertEqual(workload_score["consecutive_rejections"], 2)
            self.assertEqual(workload_score["last_rejection_code"], "adapter_delta_norm_too_large")
            self.assertTrue(workload_score["quarantined"])
            self.assertIn("cpu_lora_mock", summary["quarantined_miners"]["bad-lora-repeat"])

            with self.assertRaises(NoTaskAvailable) as raised:
                store.claim_task("bad-lora-repeat", capabilities=self._lora_capabilities())

            self.assertEqual(str(raised.exception), "miner quarantined for workload")
            summary = store.summary()
            self.assertEqual(summary["blocked_claims"], 1)
            self.assertEqual(summary["last_blocked_claim"]["miner_id"], "bad-lora-repeat")
            self.assertEqual(summary["last_blocked_claim"]["blocked_workloads"], ["cpu_lora_mock"])

    def test_quarantine_is_scoped_to_workload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": "cpu_lora_mock",
                    },
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": "diloco_train",
                    },
                ],
            )

            for _ in range(2):
                claim = store.claim_task("scoped-bad-miner", capabilities=self._lora_capabilities())
                with self.assertRaises(ResultRejected):
                    store.complete_task(
                        claim["task_id"],
                        lease_token=claim["lease_token"],
                        attempt=claim["attempt"],
                        adapter_delta={"values": [100.0, 0.0, 0.0]},
                    )

            claim = store.claim_task(
                "scoped-bad-miner",
                capabilities={
                    "runtime": "python-cli",
                    "backend": "cpu",
                    "protocol_version": "runtime_contract_v1",
                    "supported_workloads": ["cpu_lora_mock", "diloco_train"],
                },
            )

            self.assertEqual(claim["workload_type"], "diloco_train")
            summary = store.summary()
            self.assertEqual(summary["blocked_claims"], 0)
            self.assertIn("cpu_lora_mock", summary["quarantined_miners"]["scoped-bad-miner"])

    def test_quarantine_replays_from_event_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": "cpu_lora_mock",
                    },
                ],
            )
            for _ in range(2):
                claim = store.claim_task("replay-bad-lora", capabilities=self._lora_capabilities())
                with self.assertRaises(ResultRejected):
                    store.complete_task(
                        claim["task_id"],
                        lease_token=claim["lease_token"],
                        attempt=claim["attempt"],
                        adapter_delta={"values": [100.0, 0.0, 0.0]},
                    )
            with self.assertRaises(NoTaskAvailable):
                store.claim_task("replay-bad-lora", capabilities=self._lora_capabilities())

            recovered = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": "cpu_lora_mock",
                    },
                ],
            )
            summary = recovered.summary()

            self.assertEqual(summary["blocked_claims"], 1)
            self.assertTrue(
                summary["miner_workload_scores"]["replay-bad-lora"]["cpu_lora_mock"]["quarantined"]
            )
            self.assertIn("cpu_lora_mock", summary["quarantined_miners"]["replay-bad-lora"])

    def test_trust_override_allow_bypasses_and_none_restores_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": "cpu_lora_mock",
                    },
                ],
            )
            for _ in range(2):
                claim = store.claim_task("override-lora", capabilities=self._lora_capabilities())
                with self.assertRaises(ResultRejected):
                    store.complete_task(
                        claim["task_id"],
                        lease_token=claim["lease_token"],
                        attempt=claim["attempt"],
                        adapter_delta={"values": [100.0, 0.0, 0.0]},
                    )

            with self.assertRaises(NoTaskAvailable):
                store.claim_task("override-lora", capabilities=self._lora_capabilities())

            override = store.set_trust_override(
                "override-lora",
                "cpu_lora_mock",
                "allow",
                reason="manual audit accepted",
            )
            self.assertTrue(override["accepted"])
            allowed_claim = store.claim_task("override-lora", capabilities=self._lora_capabilities())
            self.assertEqual(allowed_claim["workload_type"], "cpu_lora_mock")
            summary = store.summary()
            self.assertEqual(summary["miner_trust_overrides"]["override-lora"]["cpu_lora_mock"]["mode"], "allow")
            self.assertNotIn("override-lora", summary["effective_quarantined_miners"])

            store._create_task(  # noqa: SLF001 - keep a queued lora task available after the allowed lease
                required_runtime="python-cli",
                required_backend="cpu",
                workload_type="cpu_lora_mock",
            )
            store.set_trust_override("override-lora", "cpu_lora_mock", "none", reason="restore auto")
            with self.assertRaises(NoTaskAvailable) as blocked:
                store.claim_task("override-lora", capabilities=self._lora_capabilities())

            self.assertEqual(str(blocked.exception), "miner quarantined for workload")
            summary = store.summary()
            self.assertNotIn("override-lora", summary["miner_trust_overrides"])
            self.assertIn("cpu_lora_mock", summary["effective_quarantined_miners"]["override-lora"])

    def test_trust_override_block_prevents_healthy_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": "diloco_train",
                    },
                ],
            )
            store.set_trust_override("manual-blocked", "diloco_train", "block", reason="operator test")

            with self.assertRaises(NoTaskAvailable) as blocked:
                store.claim_task("manual-blocked", capabilities=self._python_capabilities())
            summary = store.summary()

            self.assertEqual(str(blocked.exception), "miner manually blocked for workload")
            self.assertEqual(summary["blocked_claims"], 1)
            self.assertEqual(summary["last_blocked_claim"]["reason"], "miner manually blocked for workload")
            self.assertIn("diloco_train", summary["manual_blocked_miners"]["manual-blocked"])
            self.assertIn("diloco_train", summary["effective_quarantined_miners"]["manual-blocked"])

    def test_trust_override_replays_from_event_log_and_event_tail_redacts_lease(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": "diloco_train",
                    },
                ],
            )
            store.set_trust_override("replay-blocked", "diloco_train", "block", reason="persist")
            with self.assertRaises(NoTaskAvailable):
                store.claim_task("replay-blocked", capabilities=self._python_capabilities())
            store.set_trust_override("healthy-tail", "diloco_train", "allow", reason="tail")
            store.claim_task("healthy-tail", capabilities=self._python_capabilities())

            recovered = StateStore(
                tmp,
                lease_seconds=5,
                inner_steps=10,
                backlog=0,
                task_lanes=[
                    {
                        "runtime": "python-cli",
                        "backend": "cpu",
                        "count": 1,
                        "workload_type": "diloco_train",
                    },
                ],
            )
            summary = recovered.summary()
            events = recovered.event_tail(limit=20)

            self.assertEqual(summary["miner_trust_overrides"]["replay-blocked"]["diloco_train"]["mode"], "block")
            self.assertEqual(summary["miner_trust_overrides"]["healthy-tail"]["diloco_train"]["mode"], "allow")
            claimed_events = [event for event in events if event["type"] == "task_claimed"]
            self.assertTrue(claimed_events)
            self.assertEqual(claimed_events[-1]["lease_token"], "<redacted>")

    def test_capability_scheduler_records_incompatible_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=0)
            store._create_task(  # noqa: SLF001 - scheduler fixture
                required_runtime="python-cli",
                required_backend="cpu",
            )

            with self.assertRaises(NoTaskAvailable) as raised:
                store.claim_task(
                    "browser-miner",
                    capabilities={
                        "runtime": "browser",
                        "backend": "js-worker",
                        "protocol_version": "runtime_contract_v1",
                    },
                )

            self.assertEqual(str(raised.exception), "no compatible queued task available")
            summary = store.summary()
            self.assertEqual(summary["task_counts"]["queued"], 1)
            self.assertEqual(summary["incompatible_claims"], 1)
            self.assertEqual(summary["last_incompatible_claim"]["miner_id"], "browser-miner")
            self.assertEqual(
                summary["last_incompatible_claim"]["queued_requirements"],
                ["python-cli/cpu/runtime_contract_v1"],
            )

    def test_real_llm_stage_capabilities_route_split_miners_without_loading_hf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=0)
            task_id = store._create_task(  # noqa: SLF001 - scheduler fixture
                required_runtime="python-cli",
                required_backend="cpu",
                workload_type=WORKLOAD_REAL_LLM_SHARDED_INFER,
                inner_steps=1,
                workload_metadata={
                    "session_id": "real-llm-session-test",
                    "stage_id": 0,
                    "stage_count": 2,
                    "artifact_schema": "real_llm_artifact_v1",
                    "artifact_hash": "sha256:test-real-llm-artifact",
                    "model_id": "sshleifer/tiny-gpt2",
                    "backend": "hf_transformers_cpu",
                    "split_index": 1,
                    "num_hidden_layers": 2,
                    "hidden_size": 2,
                },
            )

            with self.assertRaises(NoTaskAvailable):
                store.claim_task("stage1-only", capabilities=self._real_llm_stage_capabilities("stage1"))

            claim = store.claim_task("stage0-only", capabilities=self._real_llm_stage_capabilities("stage0"))

            self.assertEqual(claim["task_id"], task_id)
            self.assertEqual(claim["task_requirements"]["stage_capability"], REAL_LLM_SHARDED_STAGE0_CAPABILITY)

    def test_real_llm_cuda_stage_capabilities_require_cuda_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=0)
            task_id = store._create_task(  # noqa: SLF001 - scheduler fixture
                required_runtime="python-cli",
                required_backend="cuda",
                workload_type=WORKLOAD_REAL_LLM_SHARDED_INFER,
                inner_steps=1,
                workload_metadata={
                    "session_id": "real-llm-cuda-session-test",
                    "stage_id": 0,
                    "stage_count": 2,
                    "artifact_schema": "real_llm_artifact_v1",
                    "artifact_hash": "sha256:test-real-llm-cuda-artifact",
                    "model_id": "sshleifer/tiny-gpt2",
                    "backend": "hf_transformers_cuda",
                    "split_index": 1,
                    "num_hidden_layers": 2,
                    "hidden_size": 2,
                },
            )

            with self.assertRaises(NoTaskAvailable):
                store.claim_task("cpu-stage0", capabilities=self._real_llm_stage_capabilities("stage0"))

            claim = store.claim_task("cuda-stage0", capabilities=self._real_llm_cuda_stage_capabilities("stage0"))

            self.assertEqual(claim["task_id"], task_id)
            self.assertEqual(claim["task_requirements"]["stage_capability"], REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY)

    def test_real_llm_cuda_session_creation_defers_cuda_runtime_to_miner(self) -> None:
        artifact = {
            "schema": "real_llm_artifact_v1",
            "artifact_hash": "sha256:test-real-llm-cuda-artifact",
            "model_id": "sshleifer/tiny-gpt2",
            "backend": "hf_transformers_cuda",
            "split_index": 1,
            "num_hidden_layers": 2,
            "hidden_size": 2,
            "vocab_size": 128,
            "read_only": True,
            "metadata_only": True,
            "cuda_runtime": {
                "backend": "hf_transformers_cuda",
                "coordinator_runtime_required": False,
                "miner_runtime_required": True,
                "diagnosis_codes": ["cuda_runtime_deferred_to_miner"],
            },
        }
        with tempfile.TemporaryDirectory() as tmp, patch(
            "crowdtensor.state_store.inspect_real_llm_artifact",
            return_value=artifact,
        ) as inspect_artifact:
            store = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=0)

            session = store.create_real_llm_sharded_inference_session(
                request_count=1,
                required_runtime="python-cli",
                required_backend="cuda",
                llm_backend="hf_transformers_cuda",
            )

            inspect_artifact.assert_called_once()
            self.assertFalse(inspect_artifact.call_args.kwargs["require_runtime"])
            self.assertEqual(session["backend"], "hf_transformers_cuda")
            self.assertEqual(session["task_requirements"]["backend"], "cuda")
            self.assertEqual(session["task_requirements"]["stage_capability"], REAL_LLM_SHARDED_CUDA_STAGE0_CAPABILITY)
            with self.assertRaises(NoTaskAvailable):
                store.claim_task("cpu-stage0", capabilities=self._real_llm_stage_capabilities("stage0"))
            claim = store.claim_task("cuda-stage0", capabilities=self._real_llm_cuda_stage_capabilities("stage0"))
            self.assertEqual(claim["task_id"], session["stage_0_task_id"])

    def test_real_llm_generation_session_chains_next_stage0_until_max_new_tokens(self) -> None:
        artifact = {
            "schema": "real_llm_artifact_v1",
            "artifact_hash": "sha256:test-real-llm-generation-artifact",
            "model_id": "sshleifer/tiny-gpt2",
            "backend": "hf_transformers_cuda",
            "split_index": 1,
            "num_hidden_layers": 2,
            "hidden_size": 2,
            "vocab_size": 128,
            "read_only": True,
        }
        with tempfile.TemporaryDirectory() as tmp, patch(
            "crowdtensor.state_store.inspect_real_llm_artifact",
            return_value=artifact,
        ):
            store = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=0)
            session = store.create_real_llm_sharded_inference_session(
                request_count=1,
                max_new_tokens=2,
                prompt_texts=["The future of open AI is"],
                required_runtime="python-cli",
                required_backend="cuda",
                llm_backend="hf_transformers_cuda",
            )
            stage0_claim = store.claim_task("stage0-a", capabilities=self._real_llm_cuda_stage_capabilities("stage0"))
            self.assertEqual(stage0_claim["workload_spec"]["max_new_tokens"], 2)
            self.assertEqual(stage0_claim["workload_spec"]["generation_step"], 0)
            activation = {
                "schema_version": "real_llm_activation_v1",
                "session_id": session["session_id"],
                "request_id": "req-1",
                "prompt_hash": stage0_claim["workload_spec"]["requests"][0]["prompt_hash"],
                "model_id": "sshleifer/tiny-gpt2",
                "artifact_hash": artifact["artifact_hash"],
                "split_index": 1,
                "generation_step": 0,
                "max_new_tokens": 2,
                "generated_token_ids": [],
                "generated_text": "",
                "input_ids": [1, 2],
                "position_ids": [0, 1],
                "hidden_shape": [1, 2, 2],
                "hidden_state": [[[0.1, 0.2], [0.3, 0.4]]],
            }
            activation["activation_hash"] = real_activation_hash = "sha256:" + hashlib.sha256(
                json.dumps({
                    "artifact_hash": activation["artifact_hash"],
                    "hidden_shape": activation["hidden_shape"],
                    "hidden_state": activation["hidden_state"],
                    "input_ids": activation["input_ids"],
                    "model_id": activation["model_id"],
                    "position_ids": activation["position_ids"],
                    "request_id": activation["request_id"],
                    "schema_version": activation["schema_version"],
                    "session_id": activation["session_id"],
                    "split_index": activation["split_index"],
                }, separators=(",", ":"), sort_keys=True).encode("utf-8")
            ).hexdigest()
            stage0_result = {
                "schema_version": "real_llm_sharded_infer_v1",
                "type": WORKLOAD_REAL_LLM_SHARDED_INFER,
                "session_id": session["session_id"],
                "stage_id": 0,
                "stage_count": 2,
                "model_id": "sshleifer/tiny-gpt2",
                "backend": "hf_transformers_cuda",
                "partition_mode": "full",
                "artifact_schema": "real_llm_artifact_v1",
                "artifact_hash": artifact["artifact_hash"],
                "split_index": 1,
                "max_new_tokens": 2,
                "generation_step": 0,
                "request_count": 1,
                "activation_count": 1,
                "activation_bytes": 10,
                "activation_hashes": [real_activation_hash],
                "activation_transport_ready": True,
                "activation_results": [activation],
                "real_llm_artifact_ready": True,
            }
            store.complete_task(
                stage0_claim["task_id"],
                lease_token=stage0_claim["lease_token"],
                attempt=stage0_claim["attempt"],
                sharded_inference_result=stage0_result,
            )
            stage1_claim = store.claim_task("stage1-a", capabilities=self._real_llm_cuda_stage_capabilities("stage1"))
            self.assertEqual(stage1_claim["workload_spec"]["generation_step"], 0)
            self.assertEqual(stage1_claim["workload_spec"]["activation_hashes"], [real_activation_hash])
            stage1_result_row = {
                "request_id": "req-1",
                "prompt_hash": activation["prompt_hash"],
                "model_id": "sshleifer/tiny-gpt2",
                "artifact_hash": artifact["artifact_hash"],
                "activation_hash": real_activation_hash,
                "generation_step": 0,
                "max_new_tokens": 2,
                "next_token_id": 42,
                "next_token_text": " useful",
                "baseline_next_token_id": 42,
                "baseline_next_token_text": " useful",
                "generated_token_ids": [42],
                "generated_token_count": 1,
                "generated_text": " useful",
                "generated_text_hash": "sha256:" + hashlib.sha256(b" useful").hexdigest(),
                "baseline_match": True,
            }
            stage1_result_row["output_hash"] = "sha256:" + hashlib.sha256(
                json.dumps({
                    "activation_hash": real_activation_hash,
                    "artifact_hash": artifact["artifact_hash"],
                    "baseline_match": True,
                    "baseline_next_token_id": 42,
                    "model_id": "sshleifer/tiny-gpt2",
                    "next_token_id": 42,
                    "request_id": "req-1",
                }, separators=(",", ":"), sort_keys=True).encode("utf-8")
            ).hexdigest()
            stage1_result = {
                "schema_version": "real_llm_sharded_infer_v1",
                "type": WORKLOAD_REAL_LLM_SHARDED_INFER,
                "session_id": session["session_id"],
                "stage_id": 1,
                "stage_count": 2,
                "model_id": "sshleifer/tiny-gpt2",
                "backend": "hf_transformers_cuda",
                "partition_mode": "full",
                "artifact_schema": "real_llm_artifact_v1",
                "artifact_hash": artifact["artifact_hash"],
                "split_index": 1,
                "max_new_tokens": 2,
                "generation_step": 0,
                "request_count": 1,
                "activation_count": 1,
                "activation_hashes": [real_activation_hash],
                "activation_transport_ready": True,
                "inference_results": [stage1_result_row],
                "inference_result": stage1_result_row,
                "baseline_match": True,
                "decoded_tokens_match": True,
                "generated_token_ids": [42],
                "generated_token_count": 1,
                "generated_text": " useful",
                "generated_text_hash": stage1_result_row["generated_text_hash"],
                "real_llm_artifact_ready": True,
            }
            store.complete_task(
                stage1_claim["task_id"],
                lease_token=stage1_claim["lease_token"],
                attempt=stage1_claim["attempt"],
                sharded_inference_result=stage1_result,
            )
            stream_events = store.session_stream_events(
                session_id=session["session_id"],
                max_new_tokens=2,
            )
            self.assertEqual(len(stream_events), 1)
            self.assertEqual(stream_events[0]["schema"], "session_stream_event_v1")
            self.assertEqual(stream_events[0]["generated_token_count"], 1)
            self.assertEqual(stream_events[0]["generation_step"], 0)
            self.assertEqual(stream_events[0]["generated_text_hash"], stage1_result_row["generated_text_hash"])
            public_text = json.dumps(stream_events, sort_keys=True)
            self.assertNotIn(" useful", public_text)
            self.assertNotIn('"generated_token_ids":', public_text)
            with self.assertRaises(NoTaskAvailable):
                store.claim_task("stage0-b", capabilities=self._real_llm_cuda_stage_capabilities("stage0"))
            next_stage0_claim = store.claim_task("stage0-a", capabilities=self._real_llm_cuda_stage_capabilities("stage0"))
            self.assertEqual(next_stage0_claim["workload_spec"]["generation_step"], 1)
            next_request = next_stage0_claim["workload_spec"]["requests"][0]
            self.assertEqual(next_request["prompt"], "The future of open AI is")
            self.assertEqual(next_request["generated_token_ids"], [42])
            self.assertEqual(next_request["generated_text"], " useful")

    def test_session_stream_events_keep_batch_request_progress_separate(self) -> None:
        session_id = "session-batch-stream"
        rows = [
            {
                "task_id": "batch-step0",
                "session_id": session_id,
                "miner_id": "stage1-a",
                "terminal_at": 1.0,
                "validation": {
                    "session_id": session_id,
                    "stage_id": 1,
                    "generation_step": 0,
                    "generated_token_count": 1,
                    "max_new_tokens": 2,
                    "generated_text_hash": "sha256:aggregate",
                    "decoded_tokens_match": True,
                    "inference_results": [
                        {
                            "request_id": "req-1",
                            "prompt_hash": "sha256:p1",
                            "generation_step": 0,
                            "generated_token_count": 1,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:r1s0",
                            "generated_text": " raw one",
                            "generated_token_ids": [1],
                            "decoded_tokens_match": True,
                        },
                        {
                            "request_id": "req-2",
                            "prompt_hash": "sha256:p2",
                            "generation_step": 0,
                            "generated_token_count": 1,
                            "max_new_tokens": 2,
                            "generated_text_hash": "sha256:r2s0",
                            "generated_text": " raw two",
                            "generated_token_ids": [2],
                            "decoded_tokens_match": True,
                        },
                    ],
                },
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(Path(tmp), lease_seconds=5, inner_steps=10, backlog=1)
            store._tasks = {
                row["task_id"]: {
                    "task_id": row["task_id"],
                    "status": STATUS_COMPLETED,
                    "workload_type": WORKLOAD_REAL_LLM_SHARDED_INFER,
                    "workload_metadata": {"session_id": session_id},
                }
                for row in rows
            }
            store._result_ledger_entry = lambda task, _event: next(row for row in rows if row["task_id"] == task["task_id"])  # type: ignore[method-assign]

            stream_events = store.session_stream_events(session_id=session_id, max_new_tokens=2)

        encoded = json.dumps(stream_events, sort_keys=True)
        self.assertEqual(len(stream_events), 2)
        self.assertEqual(
            [(event["request_id"], event["generated_token_count"]) for event in stream_events],
            [("req-1", 1), ("req-2", 1)],
        )
        self.assertNotIn(" raw ", encoded)
        self.assertNotIn('"generated_token_ids":', encoded)

    def test_real_llm_generation_stage_affinity_keeps_session_stage_on_same_miner(self) -> None:
        artifact = {
            "schema": "real_llm_artifact_v1",
            "artifact_hash": "sha256:test-real-llm-affinity-artifact",
            "model_id": "sshleifer/tiny-gpt2",
            "backend": "hf_transformers_cuda",
            "split_index": 1,
            "num_hidden_layers": 2,
            "hidden_size": 2,
            "vocab_size": 128,
            "read_only": True,
        }
        with tempfile.TemporaryDirectory() as tmp, patch(
            "crowdtensor.state_store.inspect_real_llm_artifact",
            return_value=artifact,
        ):
            store = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=0)
            session = store.create_real_llm_sharded_inference_session(
                request_count=1,
                max_new_tokens=2,
                prompt_texts=["The future of open AI is"],
                required_runtime="python-cli",
                required_backend="cuda",
                llm_backend="hf_transformers_cuda",
            )

            stage0_claim = store.claim_task("stage0-a", capabilities=self._real_llm_cuda_stage_capabilities("stage0"))
            activation = {
                "schema_version": "real_llm_activation_v1",
                "session_id": session["session_id"],
                "request_id": "req-1",
                "prompt_hash": stage0_claim["workload_spec"]["requests"][0]["prompt_hash"],
                "model_id": "sshleifer/tiny-gpt2",
                "artifact_hash": artifact["artifact_hash"],
                "split_index": 1,
                "generation_step": 0,
                "max_new_tokens": 2,
                "generated_token_ids": [],
                "generated_text": "",
                "input_ids": [1, 2],
                "position_ids": [0, 1],
                "hidden_shape": [1, 2, 2],
                "hidden_state": [[[0.1, 0.2], [0.3, 0.4]]],
            }
            activation["activation_hash"] = real_activation_hash = "sha256:" + hashlib.sha256(
                json.dumps({
                    "artifact_hash": activation["artifact_hash"],
                    "hidden_shape": activation["hidden_shape"],
                    "hidden_state": activation["hidden_state"],
                    "input_ids": activation["input_ids"],
                    "model_id": activation["model_id"],
                    "position_ids": activation["position_ids"],
                    "request_id": activation["request_id"],
                    "schema_version": activation["schema_version"],
                    "session_id": activation["session_id"],
                    "split_index": activation["split_index"],
                }, separators=(",", ":"), sort_keys=True).encode("utf-8")
            ).hexdigest()
            store.complete_task(
                stage0_claim["task_id"],
                lease_token=stage0_claim["lease_token"],
                attempt=stage0_claim["attempt"],
                sharded_inference_result={
                    "schema_version": "real_llm_sharded_infer_v1",
                    "type": WORKLOAD_REAL_LLM_SHARDED_INFER,
                    "session_id": session["session_id"],
                    "stage_id": 0,
                    "stage_count": 2,
                    "model_id": "sshleifer/tiny-gpt2",
                    "backend": "hf_transformers_cuda",
                    "partition_mode": "full",
                    "artifact_schema": "real_llm_artifact_v1",
                    "artifact_hash": artifact["artifact_hash"],
                    "split_index": 1,
                    "max_new_tokens": 2,
                    "generation_step": 0,
                    "request_count": 1,
                    "activation_count": 1,
                    "activation_bytes": 10,
                    "activation_hashes": [real_activation_hash],
                    "activation_transport_ready": True,
                    "activation_results": [activation],
                    "real_llm_artifact_ready": True,
                },
            )
            stage1_claim = store.claim_task("stage1-a", capabilities=self._real_llm_cuda_stage_capabilities("stage1"))
            self.assertEqual(stage1_claim["workload_spec"]["stage_affinity"]["miner_id"], "stage1-a")
            stage1_result_row = {
                "request_id": "req-1",
                "prompt_hash": activation["prompt_hash"],
                "model_id": "sshleifer/tiny-gpt2",
                "artifact_hash": artifact["artifact_hash"],
                "activation_hash": real_activation_hash,
                "generation_step": 0,
                "max_new_tokens": 2,
                "next_token_id": 42,
                "next_token_text": " useful",
                "baseline_next_token_id": 42,
                "baseline_next_token_text": " useful",
                "generated_token_ids": [42],
                "generated_token_count": 1,
                "generated_text": " useful",
                "generated_text_hash": "sha256:" + hashlib.sha256(b" useful").hexdigest(),
                "baseline_match": True,
            }
            stage1_result_row["output_hash"] = "sha256:" + hashlib.sha256(
                json.dumps({
                    "activation_hash": real_activation_hash,
                    "artifact_hash": artifact["artifact_hash"],
                    "baseline_match": True,
                    "baseline_next_token_id": 42,
                    "model_id": "sshleifer/tiny-gpt2",
                    "next_token_id": 42,
                    "request_id": "req-1",
                }, separators=(",", ":"), sort_keys=True).encode("utf-8")
            ).hexdigest()
            store.complete_task(
                stage1_claim["task_id"],
                lease_token=stage1_claim["lease_token"],
                attempt=stage1_claim["attempt"],
                sharded_inference_result={
                    "schema_version": "real_llm_sharded_infer_v1",
                    "type": WORKLOAD_REAL_LLM_SHARDED_INFER,
                    "session_id": session["session_id"],
                    "stage_id": 1,
                    "stage_count": 2,
                    "model_id": "sshleifer/tiny-gpt2",
                    "backend": "hf_transformers_cuda",
                    "partition_mode": "full",
                    "artifact_schema": "real_llm_artifact_v1",
                    "artifact_hash": artifact["artifact_hash"],
                    "split_index": 1,
                    "max_new_tokens": 2,
                    "generation_step": 0,
                    "request_count": 1,
                    "activation_count": 1,
                    "activation_hashes": [real_activation_hash],
                    "activation_transport_ready": True,
                    "inference_results": [stage1_result_row],
                    "inference_result": stage1_result_row,
                    "baseline_match": True,
                    "decoded_tokens_match": True,
                    "generated_token_ids": [42],
                    "generated_token_count": 1,
                    "generated_text": " useful",
                    "generated_text_hash": stage1_result_row["generated_text_hash"],
                    "real_llm_artifact_ready": True,
                },
            )

            with self.assertRaises(NoTaskAvailable):
                store.claim_task("stage0-b", capabilities=self._real_llm_cuda_stage_capabilities("stage0"))
            next_stage0_claim = store.claim_task("stage0-a", capabilities=self._real_llm_cuda_stage_capabilities("stage0"))
            self.assertEqual(next_stage0_claim["workload_spec"]["generation_step"], 1)
            self.assertEqual(next_stage0_claim["workload_spec"]["stage_affinity"]["miner_id"], "stage0-a")

    def test_real_llm_stage_affinity_allows_rescue_after_requeue(self) -> None:
        artifact = {
            "schema": "real_llm_artifact_v1",
            "artifact_hash": "sha256:test-real-llm-affinity-rescue-artifact",
            "model_id": "sshleifer/tiny-gpt2",
            "backend": "hf_transformers_cuda",
            "split_index": 1,
            "num_hidden_layers": 2,
            "hidden_size": 2,
            "vocab_size": 128,
            "read_only": True,
        }
        with tempfile.TemporaryDirectory() as tmp, patch(
            "crowdtensor.state_store.inspect_real_llm_artifact",
            return_value=artifact,
        ):
            store = StateStore(tmp, lease_seconds=0.01, inner_steps=10, backlog=0)
            store.create_real_llm_sharded_inference_session(
                request_count=1,
                max_new_tokens=1,
                required_runtime="python-cli",
                required_backend="cuda",
                llm_backend="hf_transformers_cuda",
            )
            first_claim = store.claim_task("stage0-a", capabilities=self._real_llm_cuda_stage_capabilities("stage0"))
            self.assertEqual(first_claim["workload_spec"]["stage_affinity"]["miner_id"], "stage0-a")
            store.reap_expired(now=first_claim["lease_expires_at"] + 0.001)

            rescue_claim = store.claim_task("stage0-rescue", capabilities=self._real_llm_cuda_stage_capabilities("stage0"))

            self.assertEqual(rescue_claim["attempt"], 2)
            self.assertEqual(rescue_claim["workload_spec"]["stage_affinity"]["miner_id"], "stage0-rescue")

    def test_requeue_preserves_task_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(tmp, lease_seconds=0.05, inner_steps=10, backlog=0)
            task_id = store._create_task(  # noqa: SLF001 - scheduler fixture
                required_runtime="browser",
                required_backend="js-worker",
            )
            claim = store.claim_task(
                "browser-miner",
                capabilities={
                    "runtime": "browser",
                    "backend": "js-worker",
                    "protocol_version": "runtime_contract_v1",
                },
            )
            time.sleep(0.08)

            expired = store.reap_expired()
            summary = store.summary()

            self.assertEqual(claim["task_id"], task_id)
            self.assertEqual(expired, [task_id])
            task = summary["tasks"][0]
            self.assertEqual(task["status"], "queued")
            self.assertEqual(task["required_runtime"], "browser")
            self.assertEqual(task["required_backend"], "js-worker")
            self.assertEqual(task["capabilities"], {})

    def test_legacy_task_log_defaults_to_any_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task_log = Path(tmp) / "tasks.jsonl"
            task_log.write_text(
                json.dumps({
                    "event_index": 1,
                    "type": "task_created",
                    "task_id": "legacy-task",
                    "inner_steps": 10,
                    "ts": time.time(),
                }) + "\n",
                encoding="utf-8",
            )
            store = StateStore(tmp, lease_seconds=5, inner_steps=10, backlog=0)

            claim = store.claim_task(
                "legacy-miner",
                capabilities={"runtime": "browser", "backend": "js-worker"},
            )

            self.assertEqual(claim["task_id"], "legacy-task")
            self.assertEqual(claim["task_requirements"]["runtime"], "any")
            self.assertEqual(claim["task_requirements"]["backend"], "any")
            self.assertEqual(
                claim["task_requirements"]["protocol_version"],
                "runtime_contract_v1",
            )


if __name__ == "__main__":
    unittest.main()
