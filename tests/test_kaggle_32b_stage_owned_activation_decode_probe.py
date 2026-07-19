from __future__ import annotations

import json
import py_compile
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import kaggle_32b_stage_owned_activation_decode_probe as probe


class Kaggle32BStageOwnedActivationDecodeProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="ct32b_activation_decode_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _args(self) -> object:
        return probe.parse_args([
            "--output-dir",
            str(self.tmp),
            "--kaggle-owner",
            "tester",
            "--kaggle-status-timeout-seconds",
            "5",
        ])

    def test_build_report_requires_activation_handoff_and_cleanup(self) -> None:
        args = self._args()
        stage0 = {
            "schema": probe.STAGE_REPORT_SCHEMA,
            "mode": "stage0",
            "stage_id": 0,
            "ok": True,
            "activation_ready": True,
            "activation_hash": "sha256:activation",
            "awq_stage_preparation": {"awq_stage_model_prepared": True},
            "selection": {"stage_layer_range": [0, 32], "assigned_weight_key_count": 833, "assigned_weight_file_count": 3},
            "stage_weight_load": {"loaded_weight_key_count": 833, "loaded_tensor_gb": 9.0},
            "hardware": {"kaggle_gpu_verified": True, "gpu_count": 2, "gpu_names": ["Tesla T4", "Tesla T4"]},
            "diagnosis_codes": ["kaggle_32b_stage0_activation_ready"],
        }
        stage1 = {
            "schema": probe.STAGE_REPORT_SCHEMA,
            "mode": "stage1",
            "stage_id": 1,
            "ok": True,
            "activation_ready": True,
            "stage1_decode_ready": True,
            "generated_token_count": 1,
            "activation_hash": "sha256:activation",
            "output_hash": "sha256:output",
            "awq_stage_preparation": {"awq_stage_model_prepared": True},
            "selection": {"stage_layer_range": [32, 64], "assigned_weight_key_count": 834, "assigned_weight_file_count": 3},
            "stage_weight_load": {"loaded_weight_key_count": 834, "loaded_tensor_gb": 9.0},
            "hardware": {"kaggle_gpu_verified": True, "gpu_count": 2, "gpu_names": ["Tesla T4", "Tesla T4"]},
            "diagnosis_codes": ["kaggle_32b_stage1_decode_ready"],
        }
        report = probe.build_report(
            args,
            output_dir=self.tmp,
            stage0_report=stage0,
            stage1_report=stage1,
            stage_runs=[
                {"mode": "stage0", "kernel_ref": "tester/stage0", "steps": [{"name": "kaggle_kernel_push", "ok": True}, {"name": "kaggle_kernel_delete", "ok": True}]},
                {"mode": "stage1", "kernel_ref": "tester/stage1", "steps": [{"name": "kaggle_kernel_push", "ok": True}, {"name": "kaggle_kernel_delete", "ok": True}]},
            ],
        )

        self.assertTrue(report["ok"])
        self.assertTrue(report["cross_kernel_activation_decode_verified"])
        self.assertTrue(report["one_token_generation_verified"])
        self.assertTrue(report["stage_owned_awq_runtime_verified"])
        self.assertTrue(report["activation_handoff_verified"])
        self.assertFalse(report["safety"]["activation_public"])
        encoded = json.dumps(report)
        self.assertNotIn("hidden_b64", encoded)
        self.assertNotIn('"hidden_state":', encoded)
        self.assertNotIn("next_token_id", encoded)

    def test_rendered_kaggle_kernel_is_valid_python(self) -> None:
        args = probe.parse_args([
            "--output-dir",
            str(self.tmp),
            "--kaggle-owner",
            "tester",
            "--execution-mode",
            "coordinator",
            "--max-new-tokens",
            "2",
            "--kaggle-status-timeout-seconds",
            "5",
        ])
        rendered = probe.render_kernel(
            args,
            mode="stage0",
            stage_id=0,
            coordinator_token="secret-token",
        )
        path = self.tmp / "kernel.py"
        path.write_text(rendered, encoding="utf-8")
        py_compile.compile(str(path), doraise=True)

    def test_rendered_four_stage_shard_kernel_is_valid_python(self) -> None:
        args = probe.parse_args([
            "--output-dir",
            str(self.tmp),
            "--kaggle-owner",
            "tester",
            "--execution-mode",
            "coordinator",
            "--stage-count",
            "4",
            "--max-new-tokens",
            "1",
            "--single-baseline-placement",
            "strict_stage_count",
            "--kaggle-status-timeout-seconds",
            "5",
        ])
        rendered = probe.render_kernel(
            args,
            mode="shard0",
            stage_id=0,
            stage_ids=[0, 1],
            coordinator_token="secret-token",
        )
        path = self.tmp / "kernel4.py"
        path.write_text(rendered, encoding="utf-8")
        py_compile.compile(str(path), doraise=True)

    def test_run_live_probe_uses_stage0_activation_then_removes_private_payload(self) -> None:
        args = self._args()
        activation = {
            "schema": "kaggle_32b_stage0_private_activation_v1",
            "activation_hash": "sha256:activation",
            "hidden_b64": "private",
            "hidden_shape": [1, 1, 8],
        }
        stage0 = {
            "schema": probe.STAGE_REPORT_SCHEMA,
            "mode": "stage0",
            "stage_id": 0,
            "ok": True,
            "activation_ready": True,
            "activation_hash": "sha256:activation",
            "awq_stage_preparation": {"awq_stage_model_prepared": True},
            "selection": {"stage_layer_range": [0, 32]},
            "hardware": {"kaggle_gpu_verified": True},
        }
        stage1 = {
            "schema": probe.STAGE_REPORT_SCHEMA,
            "mode": "stage1",
            "stage_id": 1,
            "ok": True,
            "activation_ready": True,
            "stage1_decode_ready": True,
            "generated_token_count": 1,
            "activation_hash": "sha256:activation",
            "output_hash": "sha256:output",
            "awq_stage_preparation": {"awq_stage_model_prepared": True},
            "selection": {"stage_layer_range": [32, 64]},
            "hardware": {"kaggle_gpu_verified": True},
        }

        calls: list[str] = []

        def fake_run(args_obj, *, package, output_dir, runner, file_patterns):
            mode = str(package["mode"])
            calls.append(mode)
            stage_dir = output_dir / "kaggle-output" / mode
            stage_dir.mkdir(parents=True, exist_ok=True)
            if mode == "stage0":
                probe.write_json(stage_dir / "ct_32b_stage0_activation_private.json", activation)
                return stage0, [{"name": "kaggle_kernel_push", "ok": True}, {"name": "kaggle_kernel_delete", "ok": True}]
            return stage1, [{"name": "kaggle_kernel_push", "ok": True}, {"name": "kaggle_kernel_delete", "ok": True}]

        with mock.patch.object(probe, "run_kaggle_package", side_effect=fake_run):
            report = probe.run_live_probe(args)

        self.assertEqual(calls, ["stage0", "stage1"])
        self.assertTrue(report["ok"])
        self.assertFalse((self.tmp / "kaggle-output" / "stage0" / "ct_32b_stage0_activation_private.json").exists())
        self.assertTrue(report["kaggle_lifecycle"]["private_activation_removed"])

    def test_coordinator_state_drives_two_token_stage_loop_without_public_private_payload(self) -> None:
        state = probe.StageCoordinatorState(prompt="Hi", max_new_tokens=2)
        claim0 = state.claim(miner_id="stage0", stage_id=0)
        self.assertFalse(claim0["done"])
        self.assertEqual(claim0["task"]["stage_id"], 0)
        state.submit({
            "task_id": claim0["task"]["task_id"],
            "stage_id": 0,
            "generation_step": 0,
            "activation": {
                "activation_hash": "sha256:a0",
                "input_ids": [1, 2],
                "hidden_b64": "private",
            },
            "duration_seconds": 1.0,
        })
        claim1 = state.claim(miner_id="stage1", stage_id=1)
        self.assertEqual(claim1["task"]["stage_id"], 1)
        state.submit({
            "task_id": claim1["task"]["task_id"],
            "stage_id": 1,
            "generation_step": 0,
            "activation_hash": "sha256:a0",
            "next_token_id_private": 3,
            "next_token_hash": "sha256:t0",
            "output_hash": "sha256:o0",
            "generated_token_count": 1,
            "duration_seconds": 1.5,
        })
        claim2 = state.claim(miner_id="stage0", stage_id=0)
        self.assertEqual(claim2["task"]["stage_id"], 0)
        self.assertEqual(claim2["task"]["input_ids"], [1, 2, 3])
        state.submit({
            "task_id": claim2["task"]["task_id"],
            "stage_id": 0,
            "generation_step": 1,
            "activation": {
                "activation_hash": "sha256:a1",
                "input_ids": [1, 2, 3],
                "hidden_b64": "private2",
            },
            "duration_seconds": 1.1,
        })
        claim3 = state.claim(miner_id="stage1", stage_id=1)
        state.submit({
            "task_id": claim3["task"]["task_id"],
            "stage_id": 1,
            "generation_step": 1,
            "activation_hash": "sha256:a1",
            "next_token_id_private": 4,
            "next_token_hash": "sha256:t1",
            "output_hash": "sha256:o1",
            "generated_token_count": 1,
            "duration_seconds": 1.6,
        })

        status = state.public_status()
        self.assertTrue(status["ready"])
        self.assertEqual(status["generated_token_count"], 2)
        encoded = json.dumps(status)
        self.assertNotIn("hidden_b64", encoded)
        self.assertNotIn("next_token_id_private", encoded)
        self.assertFalse(status["activation_public"])

    def test_four_stage_coordinator_state_routes_middle_activations(self) -> None:
        state = probe.StageCoordinatorState(prompt="Hi", max_new_tokens=1, stage_count=4)
        previous_hash = ""
        for stage_id in range(4):
            claim = state.claim(miner_id=f"stage{stage_id}", stage_id=stage_id)
            self.assertEqual(claim["task"]["stage_id"], stage_id)
            if stage_id < 3:
                previous_hash = f"sha256:a{stage_id}"
                response = state.submit({
                    "task_id": claim["task"]["task_id"],
                    "stage_id": stage_id,
                    "generation_step": 0,
                    "activation": {
                        "activation_hash": previous_hash,
                        "input_ids": [1, 2],
                        "hidden_b64": f"private{stage_id}",
                    },
                    "duration_seconds": 1.0 + stage_id,
                })
                self.assertTrue(response["accepted"])
                self.assertFalse(response["ready"])
            else:
                response = state.submit({
                    "task_id": claim["task"]["task_id"],
                    "stage_id": stage_id,
                    "generation_step": 0,
                    "activation_hash": previous_hash,
                    "next_token_id_private": 3,
                    "next_token_hash": "sha256:t0",
                    "output_hash": "sha256:o0",
                    "generated_token_count": 1,
                    "duration_seconds": 4.0,
                })
                self.assertTrue(response["accepted"])
                self.assertTrue(response["ready"])

        status = state.public_status()
        self.assertTrue(status["ready"])
        self.assertEqual(status["stage_seen"], [0, 1, 2, 3])
        self.assertEqual(status["stage_task_counts"], {"stage0": 1, "stage1": 1, "stage2": 1, "stage3": 1})
        encoded = json.dumps(status)
        self.assertNotIn("hidden_b64", encoded)
        self.assertNotIn("next_token_id_private", encoded)

    def test_build_coordinator_report_records_multitoken_and_single_kernel_comparison(self) -> None:
        args = probe.parse_args([
            "--output-dir",
            str(self.tmp),
            "--kaggle-owner",
            "tester",
            "--execution-mode",
            "coordinator",
            "--max-new-tokens",
            "2",
            "--run-single-kernel-baseline",
            "--kaggle-status-timeout-seconds",
            "5",
        ])
        coordinator_status = {
            "ready": True,
            "prompt_hash": "sha256:prompt",
            "generated_token_count": 2,
            "generated_token_hashes": ["sha256:t0", "sha256:t1"],
            "output_hashes": ["sha256:o0", "sha256:o1"],
            "activation_hashes": ["sha256:a0", "sha256:a1"],
            "completed_task_count": 4,
            "stage0_task_count": 2,
            "stage1_task_count": 2,
            "stage_seen": [0, 1],
            "completed_tasks": [
                {"task_id": "s0", "stage_id": 0, "generation_step": 0, "duration_seconds": 1.0},
                {"task_id": "s1", "stage_id": 1, "generation_step": 0, "duration_seconds": 1.5, "generated_token_count": 1},
                {"task_id": "s2", "stage_id": 0, "generation_step": 1, "duration_seconds": 1.1},
                {"task_id": "s3", "stage_id": 1, "generation_step": 1, "duration_seconds": 1.6, "generated_token_count": 1},
            ],
            "public_artifact_safe": True,
        }
        stage0_report = {
            "ok": True,
            "stage_id": 0,
            "hardware": {"kaggle_gpu_verified": True, "gpu_count": 2},
            "awq_stage_preparation": {"awq_stage_model_prepared": True},
            "selection": {"stage_layer_range": [0, 32]},
            "stage_weight_load": {"loaded_tensor_gb": 9.0},
            "cuda_memory_after_load": {"cuda_available": True, "allocated_mb": 9275.0},
            "cuda_memory_after_execution": {"cuda_available": True, "max_allocated_mb": 10904.0},
        }
        stage1_report = {
            **stage0_report,
            "stage_id": 1,
            "selection": {"stage_layer_range": [32, 64]},
            "stage_weight_load": {"loaded_tensor_gb": 9.1},
        }
        single = {
            "ok": False,
            "schema": "kaggle_32b_single_t4x2_stage_split_baseline_v1",
            "metrics": {"generated_token_count": 0, "wall_time_seconds": 1800.0, "tokens_per_second": 0.0},
            "stage0": {"loaded_tensor_gb": 9.0, "loaded_weight_key_count": 833},
            "stage1": {"loaded_tensor_gb": 9.0, "loaded_weight_key_count": 834},
            "blockers": ["single_kernel_timeout"],
            "diagnosis_codes": ["single_kernel_timeout"],
            "model": {"repo": args.model_repo},
        }
        report = probe.build_coordinator_report(
            args,
            output_dir=self.tmp,
            coordinator_status=coordinator_status,
            stage_reports=[stage0_report, stage1_report],
            stage_runs=[
                {"mode": "stage0", "steps": [{"name": "kaggle_kernel_push", "ok": True}, {"name": "kaggle_kernel_delete", "ok": True}]},
                {"mode": "stage1", "steps": [{"name": "kaggle_kernel_push", "ok": True}, {"name": "kaggle_kernel_delete", "ok": True}]},
            ],
            single_kernel_report=single,
        )

        self.assertTrue(report["ok"])
        self.assertTrue(report["coordinator_direct_management_verified"])
        self.assertTrue(report["multi_token_decode_verified"])
        self.assertEqual(report["generated_token_count"], 2)
        self.assertEqual(report["comparison"]["single_kernel_stability"], "failed_or_killed")
        self.assertEqual(report["comparison"]["two_kernel_stage_memory"]["stage0"]["loaded_tensor_gb"], 9.0)
        self.assertEqual(
            report["comparison"]["two_kernel_stage_memory"]["stage0"]["cuda_memory_after_load"]["allocated_mb"],
            9275.0,
        )
        self.assertEqual(report["comparison"]["single_kernel_stage_memory"]["stage1"]["loaded_weight_key_count"], 834)
        encoded = json.dumps(report)
        self.assertNotIn("next_token_id_private", encoded)
        self.assertNotIn("hidden_b64", encoded)

    def test_build_coordinator_report_marks_four_stage_upper_bound_crossing(self) -> None:
        args = probe.parse_args([
            "--output-dir",
            str(self.tmp),
            "--kaggle-owner",
            "tester",
            "--execution-mode",
            "coordinator",
            "--stage-count",
            "4",
            "--max-new-tokens",
            "1",
            "--run-single-kernel-baseline",
            "--single-baseline-placement",
            "strict_stage_count",
            "--kaggle-status-timeout-seconds",
            "5",
        ])
        coordinator_status = {
            "ready": True,
            "prompt_hash": "sha256:prompt",
            "stage_count": 4,
            "generated_token_count": 1,
            "generated_token_hashes": ["sha256:t0"],
            "output_hashes": ["sha256:o0"],
            "activation_hashes": ["sha256:a0", "sha256:a1", "sha256:a2"],
            "completed_task_count": 4,
            "stage_task_counts": {"stage0": 1, "stage1": 1, "stage2": 1, "stage3": 1},
            "stage_seen": [0, 1, 2, 3],
            "completed_tasks": [
                {"task_id": f"s{stage_id}", "stage_id": stage_id, "generation_step": 0, "duration_seconds": 1.0 + stage_id}
                for stage_id in range(4)
            ],
            "public_artifact_safe": True,
        }
        stage_reports = [
            {
                "ok": True,
                "mode": "shard0",
                "hardware": {"kaggle_gpu_verified": True, "gpu_count": 2},
                "stage_runtime_summaries": [
                    {
                        "stage_id": 0,
                        "device": "cuda:0",
                        "selection": {"stage_layer_range": [0, 16]},
                        "awq_stage_preparation": {"awq_stage_model_prepared": True},
                        "runtime_buffers": {"ready": True},
                        "stage_weight_load": {"loaded_tensor_gb": 4.5, "loaded_weight_key_count": 420},
                        "cuda_memory_after_load": {"cuda_available": True, "allocated_mb": 5200.0},
                    },
                    {
                        "stage_id": 1,
                        "device": "cuda:1",
                        "selection": {"stage_layer_range": [16, 32]},
                        "awq_stage_preparation": {"awq_stage_model_prepared": True},
                        "runtime_buffers": {"ready": True},
                        "stage_weight_load": {"loaded_tensor_gb": 4.6, "loaded_weight_key_count": 421},
                        "cuda_memory_after_load": {"cuda_available": True, "allocated_mb": 5300.0},
                    },
                ],
            },
            {
                "ok": True,
                "mode": "shard1",
                "hardware": {"kaggle_gpu_verified": True, "gpu_count": 2},
                "stage_runtime_summaries": [
                    {
                        "stage_id": 2,
                        "device": "cuda:0",
                        "selection": {"stage_layer_range": [32, 48]},
                        "awq_stage_preparation": {"awq_stage_model_prepared": True},
                        "runtime_buffers": {"ready": True},
                        "stage_weight_load": {"loaded_tensor_gb": 4.6, "loaded_weight_key_count": 421},
                        "cuda_memory_after_load": {"cuda_available": True, "allocated_mb": 5300.0},
                    },
                    {
                        "stage_id": 3,
                        "device": "cuda:1",
                        "selection": {"stage_layer_range": [48, 64]},
                        "awq_stage_preparation": {"awq_stage_model_prepared": True},
                        "runtime_buffers": {"ready": True},
                        "stage_weight_load": {"loaded_tensor_gb": 4.7, "loaded_weight_key_count": 422},
                        "cuda_memory_after_load": {"cuda_available": True, "allocated_mb": 5400.0},
                    },
                ],
            },
        ]
        single = {
            "ok": False,
            "schema": "kaggle_32b_single_t4x2_stage_split_baseline_v1",
            "metrics": {"generated_token_count": 0},
            "blockers": ["single_kernel_t4x2_gpu_count_below_required_stage_count"],
            "diagnosis_codes": ["single_kernel_t4x2_exceeds_gpu_count"],
            "model": {"repo": args.model_repo},
        }
        report = probe.build_coordinator_report(
            args,
            output_dir=self.tmp,
            coordinator_status=coordinator_status,
            stage_reports=stage_reports,
            stage_runs=[
                {"mode": "shard0", "steps": [{"name": "kaggle_kernel_push", "ok": True}, {"name": "kaggle_kernel_delete", "ok": True}]},
                {"mode": "shard1", "steps": [{"name": "kaggle_kernel_push", "ok": True}, {"name": "kaggle_kernel_delete", "ok": True}]},
            ],
            single_kernel_report=single,
        )

        self.assertTrue(report["ok"])
        self.assertTrue(report["upper_bound_crossing_verified"])
        self.assertTrue(report["comparison"]["upper_bound_crossing_verified"])
        self.assertEqual(report["stage_task_counts"], {"stage0": 1, "stage1": 1, "stage2": 1, "stage3": 1})
        self.assertEqual(len(report["stage_summaries"]), 4)
        self.assertEqual(report["comparison"]["two_kernel_stage_memory"]["stage3"]["loaded_tensor_gb"], 4.7)
        encoded = json.dumps(report)
        self.assertNotIn("hidden_b64", encoded)
        self.assertNotIn("next_token_id_private", encoded)

    def test_run_coordinator_probe_assembles_public_safe_report_from_workers(self) -> None:
        args = probe.parse_args([
            "--output-dir",
            str(self.tmp),
            "--kaggle-owner",
            "tester",
            "--execution-mode",
            "coordinator",
            "--max-new-tokens",
            "2",
            "--run-single-kernel-baseline",
            "--coordinator-timeout-seconds",
            "5",
            "--port",
            "0",
            "--kaggle-status-timeout-seconds",
            "5",
        ])

        stage_report = {
            "ok": True,
            "worker_loop_ready": True,
            "processed_task_count": 2,
            "hardware": {"kaggle_gpu_verified": True, "gpu_count": 2},
            "awq_stage_preparation": {"awq_stage_model_prepared": True},
            "selection": {"stage_layer_range": [0, 32]},
            "stage_weight_load": {"loaded_tensor_gb": 9.0},
        }
        single = {
            "ok": False,
            "schema": "kaggle_32b_single_t4x2_stage_split_baseline_v1",
            "metrics": {"generated_token_count": 0},
            "blockers": ["single_kernel_timeout"],
            "diagnosis_codes": ["single_kernel_timeout"],
            "model": {"repo": args.model_repo},
        }

        def fake_run_kaggle(args_obj, *, package, output_dir, runner, file_patterns, cleanup=True):
            mode = str(package["mode"])
            if mode == "stage0":
                deadline = 100
                while deadline > 0 and not state.ready():
                    deadline -= 1
                    task = state.claim(miner_id="stage0", stage_id=0)["task"]
                    if not task:
                        probe.time.sleep(0.001)
                        continue
                    state.submit({
                        "task_id": task["task_id"],
                        "stage_id": 0,
                        "generation_step": task["generation_step"],
                        "activation": {
                            "activation_hash": f"sha256:a{task['generation_step']}",
                            "input_ids": task.get("input_ids") or [1, 2],
                            "hidden_b64": "private",
                        },
                        "duration_seconds": 1.0,
                    })
                return {**stage_report, "mode": "stage0", "stage_id": 0}, [{"name": "kaggle_kernel_push", "ok": True}, {"name": "kaggle_kernel_delete", "ok": True}]
            if mode == "stage1":
                deadline = 100
                while deadline > 0 and not state.ready():
                    deadline -= 1
                    task = state.claim(miner_id="stage1", stage_id=1)["task"]
                    if not task:
                        probe.time.sleep(0.001)
                        continue
                    step = int(task["generation_step"])
                    state.submit({
                        "task_id": task["task_id"],
                        "stage_id": 1,
                        "generation_step": step,
                        "activation_hash": task.get("activation_hash"),
                        "next_token_id_private": 10 + step,
                        "next_token_hash": f"sha256:t{step}",
                        "output_hash": f"sha256:o{step}",
                        "generated_token_count": 1,
                        "duration_seconds": 1.5,
                    })
                return {**stage_report, "mode": "stage1", "stage_id": 1}, [{"name": "kaggle_kernel_push", "ok": True}, {"name": "kaggle_kernel_delete", "ok": True}]
            return {}, []

        captured: dict[str, probe.StageCoordinatorState] = {}
        original_init = probe.ProbeCoordinatorServer.__init__

        def init_spy(self, *, host, port, token, state):
            captured["state"] = state
            return original_init(self, host=host, port=port, token=token, state=state)

        with mock.patch.object(probe.ProbeCoordinatorServer, "__init__", init_spy):
            with mock.patch.object(probe, "run_single_kernel_baseline", return_value=single):
                with mock.patch.object(probe, "run_kaggle_package") as mocked_run:
                    def side_effect(*call_args, **kwargs):
                        nonlocal state
                        state = captured["state"]
                        return fake_run_kaggle(*call_args, **kwargs)

                    state = probe.StageCoordinatorState(prompt="unused", max_new_tokens=2)
                    mocked_run.side_effect = side_effect
                    report = probe.run_coordinator_probe(args)

        self.assertTrue(report["ok"])
        self.assertTrue(report["multi_token_decode_verified"])
        self.assertTrue(report["coordinator_direct_management_verified"])
        self.assertEqual(report["generated_token_count"], 2)
        self.assertFalse((self.tmp / "coordinator-private-state.json").exists())
        encoded = json.dumps(report)
        self.assertNotIn("hidden_b64", encoded)
        self.assertNotIn("next_token_id_private", encoded)

    def test_run_four_stage_coordinator_probe_assembles_upper_bound_report(self) -> None:
        args = probe.parse_args([
            "--output-dir",
            str(self.tmp),
            "--kaggle-owner",
            "tester",
            "--execution-mode",
            "coordinator",
            "--stage-count",
            "4",
            "--max-new-tokens",
            "1",
            "--run-single-kernel-baseline",
            "--single-baseline-placement",
            "strict_stage_count",
            "--coordinator-timeout-seconds",
            "5",
            "--port",
            "0",
            "--kaggle-status-timeout-seconds",
            "5",
        ])

        def report_for(mode: str) -> dict[str, object]:
            stage_ids = [0, 1] if mode == "shard0" else [2, 3]
            return {
                "ok": True,
                "mode": mode,
                "hardware": {"kaggle_gpu_verified": True, "gpu_count": 2, "gpu_names": ["Tesla T4", "Tesla T4"]},
                "stage_runtime_summaries": [
                    {
                        "stage_id": stage_id,
                        "device": f"cuda:{index}",
                        "selection": {"stage_layer_range": [stage_id * 16, (stage_id + 1) * 16]},
                        "awq_stage_preparation": {"awq_stage_model_prepared": True},
                        "runtime_buffers": {"ready": True},
                        "stage_weight_load": {"loaded_tensor_gb": 4.5 + stage_id / 10, "loaded_weight_key_count": 420 + stage_id},
                        "cuda_memory_after_load": {"cuda_available": True, "allocated_mb": 5200 + stage_id},
                    }
                    for index, stage_id in enumerate(stage_ids)
                ],
            }

        single = {
            "ok": False,
            "schema": "kaggle_32b_single_t4x2_stage_split_baseline_v1",
            "metrics": {"generated_token_count": 0},
            "blockers": ["single_kernel_t4x2_gpu_count_below_required_stage_count"],
            "diagnosis_codes": ["single_kernel_t4x2_exceeds_gpu_count"],
            "model": {"repo": args.model_repo},
        }

        def fake_run_kaggle(args_obj, *, package, output_dir, runner, file_patterns, cleanup=True):
            mode = str(package["mode"])
            while not state.ready():
                progressed = False
                for stage_id in list(package.get("stage_ids") or []):
                    task = state.claim(miner_id=mode, stage_id=int(stage_id))["task"]
                    if not task:
                        continue
                    progressed = True
                    if int(stage_id) < 3:
                        state.submit({
                            "task_id": task["task_id"],
                            "stage_id": int(stage_id),
                            "generation_step": task["generation_step"],
                            "activation": {
                                "activation_hash": f"sha256:a{stage_id}",
                                "input_ids": task.get("input_ids") or [1, 2],
                                "hidden_b64": "private",
                            },
                            "duration_seconds": 1.0,
                        })
                    else:
                        state.submit({
                            "task_id": task["task_id"],
                            "stage_id": int(stage_id),
                            "generation_step": task["generation_step"],
                            "activation_hash": task.get("activation_hash"),
                            "next_token_id_private": 3,
                            "next_token_hash": "sha256:t0",
                            "output_hash": "sha256:o0",
                            "generated_token_count": 1,
                            "duration_seconds": 1.0,
                        })
                if not progressed:
                    probe.time.sleep(0.001)
            return report_for(mode), [
                {"name": "kaggle_kernel_push", "ok": True, "command_line": f"kaggle kernels push -p {self.tmp}/private-kaggle-kernels/{mode}"},
                {"name": "kaggle_kernel_delete", "ok": True},
            ]

        captured: dict[str, probe.StageCoordinatorState] = {}
        original_init = probe.ProbeCoordinatorServer.__init__

        def init_spy(self, *, host, port, token, state):
            captured["state"] = state
            return original_init(self, host=host, port=port, token=token, state=state)

        with mock.patch.object(probe.ProbeCoordinatorServer, "__init__", init_spy):
            with mock.patch.object(probe, "run_single_kernel_baseline", return_value=single):
                with mock.patch.object(probe, "run_kaggle_package") as mocked_run:
                    def side_effect(*call_args, **kwargs):
                        nonlocal state
                        state = captured["state"]
                        return fake_run_kaggle(*call_args, **kwargs)

                    state = probe.StageCoordinatorState(prompt="unused", max_new_tokens=1, stage_count=4)
                    mocked_run.side_effect = side_effect
                    report = probe.run_coordinator_probe(args)

        self.assertTrue(report["ok"])
        self.assertTrue(report["upper_bound_crossing_verified"])
        self.assertEqual(report["stage_task_counts"], {"stage0": 1, "stage1": 1, "stage2": 1, "stage3": 1})
        self.assertEqual(len(report["stage_summaries"]), 4)
        self.assertTrue(report["kaggle_lifecycle"]["private_packages_removed"])
        encoded = json.dumps(report)
        self.assertNotIn("hidden_b64", encoded)
        self.assertNotIn("next_token_id_private", encoded)

    def test_failed_single_kernel_report_is_preserved_as_baseline_blocker(self) -> None:
        args = probe.parse_args([
            "--output-dir",
            str(self.tmp),
            "--kaggle-owner",
            "tester",
            "--execution-mode",
            "coordinator",
            "--stage-count",
            "4",
            "--run-single-kernel-baseline",
            "--single-baseline-placement",
            "strict_stage_count",
            "--kaggle-status-timeout-seconds",
            "5",
        ])

        def fake_run_kaggle(args_obj, *, package, output_dir, runner, file_patterns, cleanup=True):
            return {
                "ok": False,
                "schema": probe.STAGE_REPORT_SCHEMA,
                "mode": "single_baseline",
                "generated_token_count": 0,
                "blockers": ["kernel_stage_count_exceeds_cuda_device_count"],
                "diagnosis_codes": ["kaggle_32b_kernel_stage_count_exceeds_cuda_device_count"],
            }, [{"name": "kaggle_kernel_push", "ok": True}, {"name": "kaggle_kernel_delete", "ok": True}]

        with mock.patch.object(probe, "run_kaggle_package", side_effect=fake_run_kaggle):
            baseline = probe.run_single_kernel_baseline(args, output_dir=self.tmp)

        self.assertFalse(baseline["ok"])
        self.assertEqual(baseline["metrics"]["generated_token_count"], 0)
        self.assertEqual(baseline["blockers"], ["kernel_stage_count_exceeds_cuda_device_count"])
        self.assertNotIn("single_kernel_baseline_report_missing", baseline["blockers"])

    def test_single_kernel_baseline_cleans_private_package_dir(self) -> None:
        args = probe.parse_args([
            "--output-dir",
            str(self.tmp),
            "--kaggle-owner",
            "tester",
            "--execution-mode",
            "coordinator",
            "--stage-count",
            "4",
            "--run-single-kernel-baseline",
            "--single-baseline-placement",
            "strict_stage_count",
            "--kaggle-status-timeout-seconds",
            "5",
        ])
        private_dir = self.tmp / "private-kaggle-kernels" / "single_baseline"

        def fake_run_kaggle(args_obj, *, package, output_dir, runner, file_patterns, cleanup=True):
            private_dir.mkdir(parents=True, exist_ok=True)
            (private_dir / "kernel.py").write_text("private", encoding="utf-8")
            return {
                "ok": False,
                "schema": probe.STAGE_REPORT_SCHEMA,
                "mode": "single_baseline",
                "single_kernel_baseline": {
                    "ok": False,
                    "generated_token_count": 0,
                    "blockers": ["single_kernel_t4x2_gpu_count_below_required_stage_count"],
                    "diagnosis_codes": ["single_kernel_t4x2_exceeds_gpu_count"],
                },
            }, [{"name": "kaggle_kernel_push", "ok": True}, {"name": "kaggle_kernel_delete", "ok": True}]

        with mock.patch.object(probe, "run_kaggle_package", side_effect=fake_run_kaggle):
            baseline = probe.run_single_kernel_baseline(args, output_dir=self.tmp)

        self.assertEqual(baseline["blockers"], ["single_kernel_t4x2_gpu_count_below_required_stage_count"])
        self.assertFalse((self.tmp / "private-kaggle-kernels").exists())


if __name__ == "__main__":
    unittest.main()
