from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from crowdtensor.protocol import LeaseConflict, NoTaskAvailable, ResultRejected
from crowdtensor.state_store import StateStore
from crowdtensor.diloco import run_inner_loop
from crowdtensor.micro_transformer import WORKLOAD_TYPE as WORKLOAD_MICRO_TRANSFORMER_LM
from crowdtensor.micro_transformer import run_micro_transformer_inner_loop
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
