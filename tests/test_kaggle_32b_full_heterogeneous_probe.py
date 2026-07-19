from __future__ import annotations

import json
import py_compile
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import kaggle_32b_full_heterogeneous_probe as probe


class Kaggle32BFullHeterogeneousProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="ct32b_full_het_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _args(self) -> object:
        return probe.parse_args([
            "--output-dir",
            str(self.tmp),
            "--kaggle-owner",
            "tester",
            "--coordinator-timeout-seconds",
            "5",
            "--kaggle-status-timeout-seconds",
            "5",
            "--port",
            "0",
            "--max-new-tokens",
            "2",
        ])

    def test_rendered_gpu_and_cpu_kernels_are_valid_python(self) -> None:
        args = self._args()
        gpu = probe.render_kernel(
            args,
            mode="gpu-shard0",
            stage_ids=[0, 1],
            resource_kind="gpu",
            coordinator_token="secret-token",
        )
        cpu = probe.render_kernel(
            args,
            mode="cpu-stage4",
            stage_ids=[4],
            resource_kind="cpu",
            coordinator_token="secret-token",
        )
        gpu_path = self.tmp / "gpu_kernel.py"
        cpu_path = self.tmp / "cpu_kernel.py"
        gpu_path.write_text(gpu, encoding="utf-8")
        cpu_path.write_text(cpu, encoding="utf-8")
        py_compile.compile(str(gpu_path), doraise=True)
        py_compile.compile(str(cpu_path), doraise=True)

    def test_rendered_llama_stage_uses_delta_token_on_kv_cache_hit(self) -> None:
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional hf dependency missing: {exc}")
        args = self._args()
        rendered = probe.render_kernel(
            args,
            mode="cpu-stage0",
            stage_ids=[0],
            resource_kind="cpu",
            coordinator_token="secret-token",
        )
        namespace: dict[str, object] = {}
        exec(compile(rendered.replace("\nmain()\n", "\n"), "<rendered-32b-kernel>", "exec"), namespace)
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
        model = AutoModelForCausalLM.from_config(config).eval()
        selection = {"stage_id": 0, "stage_count": 2, "stage_layer_range": [0, 1]}
        cache = {"stage_id": 0}

        activation0, kv0 = namespace["run_stage_activation"](
            model,
            None,
            selection,
            torch.device("cpu"),
            input_ids_values=[1, 2, 3],
            generation_step=0,
            stage_cache=cache,
        )
        activation1, kv1 = namespace["run_stage_activation"](
            model,
            None,
            selection,
            torch.device("cpu"),
            input_ids_values=[1, 2, 3, 4],
            generation_step=1,
            stage_cache=cache,
        )

        self.assertEqual(activation0["hidden_shape"][1], 3)
        self.assertEqual(activation1["hidden_shape"][1], 1)
        self.assertFalse(kv0["last_cache_hit"])
        self.assertTrue(kv1["last_cache_hit"])
        self.assertEqual(kv1["hit_count"], 1)
        self.assertEqual(kv1["tokens_before"], 3)
        self.assertEqual(kv1["tokens_after"], 4)

    def test_rendered_bloom_stage_can_run_stage_activation_and_final_decode(self) -> None:
        try:
            import torch
            from transformers import BloomConfig, BloomForCausalLM
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional hf dependency missing: {exc}")
        args = probe.parse_args([
            "--output-dir",
            str(self.tmp),
            "--kaggle-owner",
            "tester",
            "--model-repo",
            "bigscience/bloom",
            "--stage-ranges-json",
            "[[0,1],[1,2]]",
            "--stage-groups-json",
            json.dumps([
                {"mode": "gpu-stage0", "stage_ids": [0], "resource_kind": "gpu"},
                {"mode": "cpu-stage1", "stage_ids": [1], "resource_kind": "cpu"},
            ]),
            "--max-new-tokens",
            "1",
        ])
        rendered = probe.render_kernel(
            args,
            mode="cpu-stage0",
            stage_ids=[0],
            resource_kind="cpu",
            coordinator_token="secret-token",
        )
        namespace: dict[str, object] = {}
        exec(compile(rendered.replace("\nmain()\n", "\n"), "<rendered-bloom-kernel>", "exec"), namespace)
        config = BloomConfig(n_layer=2, hidden_size=32, n_head=4, vocab_size=99, use_cache=False)
        model = BloomForCausalLM(config).to(dtype=torch.bfloat16).eval()

        prefixes0 = namespace["stage_prefixes"](0, 2, [0, 1], config.to_dict())
        prefixes1 = namespace["stage_prefixes"](1, 2, [1, 2], config.to_dict())
        self.assertIn("word_embeddings.", prefixes0)
        self.assertIn("h.0.", prefixes0)
        self.assertIn("h.1.", prefixes1)
        self.assertIn("ln_f.", prefixes1)
        self.assertIn("lm_head.", prefixes1)

        activation, kv0 = namespace["run_stage_activation"](
            model,
            None,
            {"stage_id": 0, "stage_count": 2, "stage_layer_range": [0, 1]},
            torch.device("cpu"),
            input_ids_values=[1, 2],
            generation_step=0,
            stage_cache={"stage_id": 0},
        )
        class FakeTokenizer:
            def decode(self, token_ids, skip_special_tokens=False):
                return "tok-" + "-".join(str(value) for value in token_ids)

        decoded = namespace["final_stage_decode"](
            model,
            FakeTokenizer(),
            {"stage_id": 1, "stage_count": 2, "stage_layer_range": [1, 2]},
            activation,
            torch.device("cpu"),
            stage_cache={"stage_id": 1},
            generation_step=0,
        )

        self.assertEqual(activation["hidden_shape"], [1, 2, 32])
        self.assertEqual(decoded["generated_token_count"], 1)
        self.assertIn("next_token_hash", decoded)
        self.assertTrue(kv0["hit_target_ready"])

    def test_rendered_worker_exits_after_own_stage_finishes_before_global_ready(self) -> None:
        args = probe.parse_args([
            "--output-dir",
            str(self.tmp),
            "--kaggle-owner",
            "tester",
            "--stage-ranges-json",
            "[[0,2],[2,4],[4,6]]",
            "--stage-groups-json",
            json.dumps([
                {"mode": "gpu-stage0", "stage_ids": [0], "resource_kind": "gpu"},
                {"mode": "middle-stage", "stage_ids": [1], "resource_kind": "cpu"},
                {"mode": "cpu-stage2", "stage_ids": [2], "resource_kind": "cpu"},
            ]),
            "--max-new-tokens",
            "1",
            "--task-idle-timeout-seconds",
            "60",
            "--task-poll-interval-seconds",
            "5",
        ])
        rendered = probe.render_kernel(
            args,
            mode="middle-stage",
            stage_ids=[1],
            resource_kind="cpu",
            coordinator_token="secret-token",
        )
        namespace: dict[str, object] = {}
        exec(compile(rendered.replace("\nmain()\n", "\n"), "<rendered-32b-worker>", "exec"), namespace)
        calls = {"claim": 0}

        def fake_http_json(method: str, path: str, payload: dict, timeout: float = 120) -> dict:
            self.assertEqual(method, "POST")
            if path == "/claim":
                calls["claim"] += 1
                if calls["claim"] == 1:
                    return {
                        "ok": True,
                        "done": False,
                        "task": {
                            "task_id": "stage1-task",
                            "stage_id": 1,
                            "generation_step": 0,
                            "activation": {"activation_hash": "sha256:in", "input_ids": [1]},
                        },
                    }
                return {"ok": True, "done": False, "task": None}
            if path == "/submit":
                self.assertEqual(payload["task_id"], "stage1-task")
                return {"ok": True, "accepted": True, "ready": False}
            raise AssertionError(path)

        def fake_run_stage_activation(*args, **kwargs):
            return (
                {
                    "activation_hash": "sha256:out",
                    "input_ids": [1],
                    "activation_payload_public": False,
                },
                {
                    "schema": "kaggle_32b_full_stage_local_kv_cache_v1",
                    "stage_id": 1,
                    "ready": True,
                    "hit_count": 0,
                    "expected_hit_count": 0,
                    "hit_target_ready": True,
                    "past_key_values_public": False,
                    "cache_tensors_public": False,
                },
            )

        namespace["http_json"] = fake_http_json
        namespace["run_stage_activation"] = fake_run_stage_activation
        namespace["cuda_memory_summary"] = lambda *args, **kwargs: {"cuda_available": False}
        with mock.patch.object(namespace["time"], "sleep") as sleep_mock:
            processed = namespace["worker_loop"](
                [
                    {
                        "stage_id": 1,
                        "model": object(),
                        "selection": {"stage_id": 1, "stage_count": 3},
                        "device": "cpu",
                        "kv_cache": {"stage_id": 1},
                    }
                ],
                None,
                {},
            )

        self.assertEqual(calls["claim"], 1)
        self.assertEqual(len(processed), 1)
        self.assertTrue(processed[0]["accepted"])
        sleep_mock.assert_not_called()

    def test_stage_coordinator_routes_all_nine_stages_without_private_payload(self) -> None:
        state = probe.StageCoordinatorState(prompt="Hi", max_new_tokens=2, stage_count=9)
        previous_hash = ""
        for step in range(2):
            for stage_id in range(9):
                claim = state.claim(miner_id=f"stage{stage_id}", stage_id=stage_id)
                self.assertEqual(claim["task"]["stage_id"], stage_id)
                self.assertEqual(claim["task"]["generation_step"], step)
                kv_cache = {
                    "schema": "kaggle_32b_full_stage_local_kv_cache_v1",
                    "stage_id": stage_id,
                    "ready": True,
                    "hit_count": step,
                    "miss_count": 1,
                    "prefill_count": 1,
                    "expected_hit_count": 1,
                    "hit_target_ready": step >= 1,
                    "tokens_before": 2 + step - 1 if step else 0,
                    "tokens_after": 2 + step,
                    "past_key_values_public": False,
                    "cache_tensors_public": False,
                }
                if stage_id < 8:
                    previous_hash = f"sha256:a{stage_id}-{step}"
                    response = state.submit({
                        "task_id": claim["task"]["task_id"],
                        "stage_id": stage_id,
                        "generation_step": step,
                        "activation": {
                            "activation_hash": previous_hash,
                            "input_ids": [1, 2, *([3] if step else [])],
                            "hidden_b64": f"private{stage_id}",
                        },
                        "kv_cache": kv_cache,
                        "duration_seconds": 1.0,
                    })
                    self.assertTrue(response["accepted"])
                    self.assertFalse(response["ready"])
                else:
                    response = state.submit({
                        "task_id": claim["task"]["task_id"],
                        "stage_id": stage_id,
                        "generation_step": step,
                        "activation_hash": previous_hash,
                        "next_token_id_private": 3 + step,
                        "next_token_hash": f"sha256:t{step}",
                        "output_hash": f"sha256:o{step}",
                        "generated_token_count": 1,
                        "kv_cache": kv_cache,
                        "duration_seconds": 1.0,
                    })
                    self.assertTrue(response["accepted"])
                    self.assertEqual(response["ready"], step == 1)

        status = state.public_status()
        self.assertTrue(status["ready"])
        self.assertTrue(status["kv_cache_ready"])
        self.assertEqual(status["generated_token_count"], 2)
        self.assertEqual(status["stage_seen"], list(range(9)))
        self.assertEqual(status["stage_task_counts"], {f"stage{i}": 2 for i in range(9)})
        self.assertEqual(status["stage_kv_cache"]["stage0"]["hit_count"], 1)
        encoded = json.dumps(status)
        self.assertNotIn("hidden_b64", encoded)
        self.assertNotIn("next_token_id_private", encoded)
        self.assertNotIn('"past_key_values":', encoded)

    def test_wait_for_coordinator_ready_returns_when_worker_report_fails(self) -> None:
        state = probe.StageCoordinatorState(prompt="Hi", max_new_tokens=1, stage_count=3)
        reports = {"gpu-shard0": {"ok": False, "blockers": ["cuda_runtime_missing"]}}

        with mock.patch.object(probe.time, "sleep") as sleep_mock:
            status = probe.wait_for_coordinator_ready(
                state,
                timeout_seconds=60,
                stage_reports_by_mode=reports,
                errors=[],
            )

        self.assertFalse(status["ready"])
        sleep_mock.assert_not_called()

    def test_wait_for_coordinator_ready_ignores_no_task_worker_reports(self) -> None:
        state = probe.StageCoordinatorState(prompt="Hi", max_new_tokens=1, stage_count=3)
        reports = {"cpu-tail": {"ok": False, "blockers": ["coordinator_worker_processed_no_tasks"]}}
        sleeps = {"count": 0}

        def fake_sleep(_seconds: float) -> None:
            sleeps["count"] += 1
            if sleeps["count"] >= 2:
                stage0 = state.claim(miner_id="stage0", stage_id=0)["task"]
                state.submit({
                    "stage_id": 0,
                    "task_id": stage0["task_id"],
                    "activation": {"activation_hash": "sha256:0"},
                    "activation_hash": "sha256:0",
                    "kv_cache": {"ready": True, "stage_id": 0},
                })
                stage1 = state.claim(miner_id="stage1", stage_id=1)["task"]
                state.submit({
                    "stage_id": 1,
                    "task_id": stage1["task_id"],
                    "activation": {"activation_hash": "sha256:1"},
                    "activation_hash": "sha256:1",
                    "kv_cache": {"ready": True, "stage_id": 1},
                })
                stage2 = state.claim(miner_id="stage2", stage_id=2)["task"]
                state.submit({
                    "stage_id": 2,
                    "task_id": stage2["task_id"],
                    "generated_token_count": 1,
                    "next_token_id_private": 1,
                    "next_token_hash": "sha256:token",
                    "output_hash": "sha256:out",
                    "kv_cache": {"ready": True, "stage_id": 2},
                })

        with mock.patch.object(probe.time, "sleep", side_effect=fake_sleep):
            status = probe.wait_for_coordinator_ready(
                state,
                timeout_seconds=60,
                stage_reports_by_mode=reports,
                errors=[],
            )

        self.assertTrue(status["ready"])
        self.assertGreaterEqual(sleeps["count"], 1)

    def test_stage_coordinator_accepts_initial_input_ids(self) -> None:
        state = probe.StageCoordinatorState(prompt="Hi", max_new_tokens=1, stage_count=3, initial_input_ids=[1, 2])

        claim = state.claim(miner_id="stage0", stage_id=0)

        self.assertEqual(claim["task"]["input_ids"], [1, 2])
        status = state.public_status()
        self.assertEqual(status["initial_input_token_count"], 2)
        self.assertFalse(status["input_token_ids_public"])

    def test_stage_groups_json_can_describe_72b_gpu_tpu_cpu_topology(self) -> None:
        args = probe.parse_args([
            "--output-dir",
            str(self.tmp),
            "--kaggle-owner",
            "tester",
            "--model-repo",
            "Qwen/Qwen2.5-72B-Instruct",
            "--stage-ranges-json",
            "[[0,8],[8,16],[16,24],[24,32],[32,40],[40,48],[48,56],[56,64],[64,72],[72,80]]",
            "--stage-groups-json",
            json.dumps([
                {"mode": "gpu-shard0", "stage_ids": [0, 1], "resource_kind": "gpu"},
                {"mode": "gpu-shard1", "stage_ids": [2, 3], "resource_kind": "gpu"},
                {"mode": "web-tpu-stage4", "stage_ids": [4], "resource_kind": "web_tpu"},
                {"mode": "cpu-stage5", "stage_ids": [5], "resource_kind": "cpu"},
                {"mode": "cpu-stage6", "stage_ids": [6], "resource_kind": "cpu"},
                {"mode": "cpu-stage7", "stage_ids": [7], "resource_kind": "cpu"},
                {"mode": "cpu-stage8", "stage_ids": [8], "resource_kind": "cpu"},
                {"mode": "cpu-stage9", "stage_ids": [9], "resource_kind": "cpu"},
            ]),
        ])

        groups = probe.stage_groups_for(args)

        self.assertEqual(len(probe.stage_ranges_from_args(args)), 10)
        self.assertEqual([group["resource_kind"] for group in groups], [
            "gpu",
            "gpu",
            "web_tpu",
            "cpu",
            "cpu",
            "cpu",
            "cpu",
            "cpu",
        ])
        self.assertEqual([stage for group in groups for stage in group["stage_ids"]], list(range(10)))

    def test_parse_args_accepts_colab_tpu_provider(self) -> None:
        args = probe.parse_args([
            "--output-dir",
            str(self.tmp),
            "--kaggle-owner",
            "tester",
            "--tpu-provider",
            "colab_cli",
            "--colab-session-name",
            "ct-colab-tpu-v5e1",
        ])

        self.assertEqual(args.tpu_provider, "colab_cli")
        self.assertEqual(args.colab_session_name, "ct-colab-tpu-v5e1")
        self.assertTrue(str(args.colab_session_config).endswith("sessions.json"))

    def test_stage_groups_json_can_describe_kaggle_colab_gpu_cpu_topology(self) -> None:
        args = probe.parse_args([
            "--output-dir",
            str(self.tmp),
            "--kaggle-owner",
            "tester",
            "--model-repo",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--stage-ranges-json",
            "[[0,8],[8,16],[16,24]]",
            "--stage-groups-json",
            json.dumps([
                {"mode": "kaggle-gpu-stage0", "stage_ids": [0], "resource_kind": "gpu"},
                {"mode": "colab-gpu-stage1", "stage_ids": [1], "resource_kind": "colab_gpu"},
                {"mode": "cpu-stage2", "stage_ids": [2], "resource_kind": "cpu"},
            ]),
            "--colab-gpu-session-name",
            "ct-colab-cuda-gpu",
        ])

        groups = probe.stage_groups_for(args)

        self.assertEqual([group["resource_kind"] for group in groups], ["gpu", "colab_gpu", "cpu"])
        self.assertEqual(args.colab_gpu_session_name, "ct-colab-cuda-gpu")
        self.assertTrue(args.colab_gpu_reacquire_before)
        self.assertEqual(args.colab_gpu_max_attempts, 3)

    def test_kaggle_model_sources_are_written_to_kernel_metadata_and_worker(self) -> None:
        args = probe.parse_args([
            "--output-dir",
            str(self.tmp),
            "--kaggle-owner",
            "tester",
            "--model-repo",
            "bigscience/bloom",
            "--kaggle-model-sources-json",
            '["bigscience/bloom/Transformers/default/1"]',
            "--kaggle-attached-model-paths-json",
            '["/kaggle/input/models/bigscience/bloom/transformers/default/1","/kaggle/input/bloom/transformers/default/1"]',
        ])

        package = probe.build_package(
            args,
            output_dir=self.tmp,
            mode="gpu-stage0",
            stage_ids=[0, 1],
            resource_kind="gpu",
            coordinator_token="token",
        )
        metadata = json.loads((package["kernel_dir"] / "kernel-metadata.json").read_text(encoding="utf-8"))
        rendered = (package["kernel_dir"] / "kernel.py").read_text(encoding="utf-8")

        self.assertEqual(metadata["model_sources"], ["bigscience/bloom/Transformers/default/1"])
        self.assertIn("/kaggle/input/models/bigscience/bloom/transformers/default/1", rendered)
        self.assertIn("KAGGLE_ATTACHED_MODEL_PATHS", rendered)

    def test_parse_args_accepts_stage_launch_stagger(self) -> None:
        args = probe.parse_args([
            "--output-dir",
            str(self.tmp),
            "--kaggle-owner",
            "tester",
            "--stage-launch-stagger-seconds",
            "3",
        ])

        self.assertEqual(args.stage_launch_stagger_seconds, 3)

    def test_non_accepted_push_attempts_cleanup_when_kernel_ref_resolves(self) -> None:
        args = self._args()
        package = {
            "kernel_dir": self.tmp / "kernel",
            "resource_kind": "gpu",
            "declared_kernel_ref": "tester/not-accepted",
            "kernel_ref": "tester/not-accepted",
            "mode": "gpu-shard1",
            "report_filename": "missing.json",
            "metadata": {"title": "Not Accepted"},
        }
        calls: list[list[str]] = []

        def fake_runner(command, **kwargs):
            calls.append([str(part) for part in command])
            if command[:3] == ["kaggle", "kernels", "push"]:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout="Kernel push error: Maximum batch GPU session count of 2 reached.\n",
                    stderr="",
                )
            if command[:3] == ["kaggle", "kernels", "list"]:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout="tester/not-accepted  Not Accepted  tester  now  0\n",
                    stderr="",
                )
            if command[:3] == ["kaggle", "kernels", "delete"]:
                return subprocess.CompletedProcess(command, 0, stdout="deleted\n", stderr="")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        report, steps = probe.run_kaggle_package(
            args,
            package=package,
            output_dir=self.tmp,
            runner=fake_runner,
            cleanup=True,
        )

        self.assertEqual(report, {})
        self.assertFalse(steps[0]["accepted"])
        self.assertTrue(any(step.get("name") == "kaggle_kernel_delete" and step.get("ok") for step in steps))
        self.assertTrue(any(call[:3] == ["kaggle", "kernels", "delete"] for call in calls))

    def test_colab_tpu_worker_uses_colab_runtime_loader_report(self) -> None:
        args = probe.parse_args([
            "--output-dir",
            str(self.tmp),
            "--kaggle-owner",
            "tester",
            "--model-repo",
            "Qwen/Qwen2.5-72B-Instruct",
            "--stage-ranges-json",
            "[[0,8],[8,16],[16,24],[24,32],[32,36],[36,44],[44,52],[52,60],[60,70],[70,80]]",
            "--stage-groups-json",
            json.dumps([
                {"mode": "gpu-shard0", "stage_ids": [0, 1], "resource_kind": "gpu"},
                {"mode": "gpu-shard1", "stage_ids": [2, 3], "resource_kind": "gpu"},
                {"mode": "web-tpu-stage4", "stage_ids": [4], "resource_kind": "web_tpu"},
                {"mode": "cpu-stage5", "stage_ids": [5], "resource_kind": "cpu"},
                {"mode": "cpu-stage6", "stage_ids": [6], "resource_kind": "cpu"},
                {"mode": "cpu-stage7", "stage_ids": [7], "resource_kind": "cpu"},
                {"mode": "cpu-stage8", "stage_ids": [8], "resource_kind": "cpu"},
                {"mode": "cpu-stage9", "stage_ids": [9], "resource_kind": "cpu"},
            ]),
            "--tpu-provider",
            "colab_cli",
            "--web-tpu-execute-layer-count",
            "4",
            "--web-tpu-execute-timeout-seconds",
            "300",
            "--task-idle-timeout-seconds",
            "30",
            "--port",
            "0",
        ])
        state = probe.StageCoordinatorState(prompt="Hi", max_new_tokens=1, stage_count=10, initial_input_ids=[1, 2])
        for stage_id in range(4):
            claim = state.claim(miner_id=f"pre-stage{stage_id}", stage_id=stage_id)
            response = state.submit(
                {
                    "task_id": claim["task"]["task_id"],
                    "stage_id": stage_id,
                    "generation_step": 0,
                    "activation": {
                        "schema": "kaggle_32b_full_activation_v1",
                        "activation_hash": f"sha256:pre{stage_id}",
                        "activation_payload_public": False,
                    },
                    "activation_hash": f"sha256:pre{stage_id}",
                    "output_hash": f"sha256:out{stage_id}",
                    "kv_cache": {
                        "schema": "kaggle_32b_full_stage_local_kv_cache_v1",
                        "stage_id": stage_id,
                        "ready": True,
                        "hit_count": 0,
                        "expected_hit_count": 0,
                        "hit_target_ready": True,
                        "past_key_values_public": False,
                        "cache_tensors_public": False,
                    },
                }
            )
            self.assertTrue(response["accepted"])
        server = probe.ProbeCoordinatorServer(host="127.0.0.1", port=0, token="token", state=state)

        class FakeRuntime:
            def __init__(self, *args, **kwargs):
                pass

            def execute_code(self, code: str, timeout: float):
                return [
                    {
                        "text": json.dumps(
                            {
                                "schema": "kaggle_tpu_32b_stage_owned_loader_probe_v1",
                                "report": {
                                    "schema": "kaggle_tpu_32b_stage_owned_loader_probe_v1",
                                    "ok": True,
                                    "model_repo": "Qwen/Qwen2.5-72B-Instruct",
                                    "stage_owned_header_verified": True,
                                    "partial_tensor_to_tpu_verified": True,
                                    "full_stage_owned_tpu_loader_ready": True,
                                    "tpu_32b_runtime_adapter_ready": True,
                                    "input_activation_consumed": True,
                                    "output_activation_private_present": True,
                                    "output_activation_private": {
                                        "schema": "gpu_tpu_cpu_bridge_activation_v1",
                                        "activation_hash": "sha256:tpuout",
                                        "activation_payload_public": False,
                                    },
                                    "executed_layer_count": 4,
                                    "missing_stage_key_count": 0,
                                    "assigned_weight_key_count": 48,
                                    "assigned_weight_file_count": 3,
                                    "loaded_execution_tensor_key_count": 48,
                                    "loaded_execution_tensor_gb": 6.5,
                                    "stage_local_kv_cache_verified": True,
                                    "stage_output_hash": "sha256:stageout",
                                    "tpu_device_count": 1,
                                    "tpu_device_kind": "TPU v5 lite",
                                    "public_artifact_safe": True,
                                },
                            }
                        )
                    }
                ]

        with mock.patch.object(probe.web_tpu_bridge, "load_colab_session", return_value={"url": "http://colab", "token": "secret", "endpoint": "ep"}), \
            mock.patch.object(probe, "load_colab_runtime_class", return_value=FakeRuntime):
            server.start()
            try:
                args.coordinator_url = f"http://127.0.0.1:{server.port}"
                report = probe.web_tpu_stage_worker(args, stage_id=4, token="token", timeout_seconds=30)
            finally:
                server.stop()

        self.assertTrue(report["ok"])
        self.assertEqual(report["tpu_provider"], "colab_cli")
        self.assertEqual(report["processed_task_count"], 1)
        self.assertTrue(report["full_stage_owned_tpu_loader_ready"])
        self.assertEqual(report["tpu_device_count"], 1)
        self.assertEqual(state.public_status()["stage_task_counts"]["stage4"], 1)

    def test_build_report_marks_72b_full_topology_only_when_all_stages_ready(self) -> None:
        args = probe.parse_args([
            "--output-dir",
            str(self.tmp),
            "--kaggle-owner",
            "tester",
            "--model-repo",
            "Qwen/Qwen2.5-72B-Instruct",
            "--max-new-tokens",
            "1",
            "--stage-ranges-json",
            "[[0,8],[8,16],[16,24],[24,32],[32,40],[40,48],[48,56],[56,64],[64,72],[72,80]]",
            "--stage-groups-json",
            json.dumps([
                {"mode": "gpu-shard0", "stage_ids": [0, 1], "resource_kind": "gpu"},
                {"mode": "gpu-shard1", "stage_ids": [2, 3], "resource_kind": "gpu"},
                {"mode": "web-tpu-stage4", "stage_ids": [4], "resource_kind": "web_tpu"},
                {"mode": "cpu-stage5", "stage_ids": [5], "resource_kind": "cpu"},
                {"mode": "cpu-stage6", "stage_ids": [6], "resource_kind": "cpu"},
                {"mode": "cpu-stage7", "stage_ids": [7], "resource_kind": "cpu"},
                {"mode": "cpu-stage8", "stage_ids": [8], "resource_kind": "cpu"},
                {"mode": "cpu-stage9", "stage_ids": [9], "resource_kind": "cpu"},
            ]),
        ])
        coordinator_status = {
            "ready": True,
            "generated_token_count": 1,
            "activation_hashes": [f"sha256:a{i}" for i in range(9)],
            "stage_seen": list(range(10)),
            "stage_task_counts": {f"stage{i}": 1 for i in range(10)},
            "stage_kv_cache": {
                f"stage{i}": {
                    "schema": "kaggle_32b_full_stage_local_kv_cache_v1",
                    "stage_id": i,
                    "ready": True,
                    "hit_count": 0,
                    "expected_hit_count": 0,
                    "hit_target_ready": True,
                    "past_key_values_public": False,
                    "cache_tensors_public": False,
                }
                for i in range(10)
            },
            "completed_tasks": [],
        }
        ranges_72b = [[0,8],[8,16],[16,24],[24,32],[32,40],[40,48],[48,56],[56,64],[64,72],[72,80]]
        stage_reports = [
            self._report_for("gpu-shard0", [0, 1], "gpu", stage_ranges=ranges_72b),
            self._report_for("gpu-shard1", [2, 3], "gpu", stage_ranges=ranges_72b),
            self._report_for("web-tpu-stage4", [4], "web_tpu", stage_ranges=ranges_72b),
            *[self._report_for(f"cpu-stage{i}", [i], "cpu", stage_ranges=ranges_72b) for i in range(5, 10)],
        ]
        stage_runs = [
            {"mode": "gpu-shard0", "stage_ids": [0, 1], "resource_kind": "gpu", "steps": [{"name": "kaggle_kernel_push", "accepted": True}, {"name": "kaggle_kernel_delete", "ok": True}]},
            {"mode": "gpu-shard1", "stage_ids": [2, 3], "resource_kind": "gpu", "steps": [{"name": "kaggle_kernel_push", "accepted": True}, {"name": "kaggle_kernel_delete", "ok": True}]},
            {"mode": "web-tpu-stage4", "stage_ids": [4], "resource_kind": "web_tpu", "steps": [{"name": "web_tpu_stage_worker", "accepted": True, "ok": True}]},
            *[
                {"mode": f"cpu-stage{i}", "stage_ids": [i], "resource_kind": "cpu", "steps": [{"name": "kaggle_kernel_push", "accepted": True}, {"name": "kaggle_kernel_delete", "ok": True}]}
                for i in range(5, 10)
            ],
        ]

        report = probe.build_report(
            args,
            output_dir=self.tmp,
            coordinator_status=coordinator_status,
            stage_reports=stage_reports,
            stage_runs=stage_runs,
            errors=[],
        )

        self.assertTrue(report["ok"])
        self.assertTrue(report["gpu_tpu_cpu_72b_same_request_verified"])
        self.assertTrue(report["same_request_72b_full_model_verified"])
        self.assertTrue(report["full_72b_weight_loading_public_claim"])
        self.assertTrue(report["full_72b_layer_coverage_verified"])
        self.assertTrue(report["gpu_tpu_cpu_72b_full_topology_verified"])
        self.assertTrue(report["four_t4_one_tpu_five_cpu_topology_verified"])

    def test_build_report_accepts_72b_five_t4_colab_gpu_cpu_topology_without_tpu_claim(self) -> None:
        stage_ranges = [[0, 2], [2, 8], [8, 14], [14, 20], [20, 26], [26, 32], [32, 44], [44, 56], [56, 68], [68, 80]]
        args = probe.parse_args([
            "--output-dir",
            str(self.tmp),
            "--kaggle-owner",
            "tester",
            "--model-repo",
            "Qwen/Qwen2.5-72B-Instruct",
            "--max-new-tokens",
            "1",
            "--stage-ranges-json",
            json.dumps(stage_ranges),
            "--stage-groups-json",
            json.dumps([
                {"mode": "cpu-stage0", "stage_ids": [0], "resource_kind": "cpu"},
                {"mode": "kaggle-gpu-a", "stage_ids": [1, 2], "resource_kind": "gpu"},
                {"mode": "kaggle-gpu-b", "stage_ids": [3, 4], "resource_kind": "gpu"},
                {"mode": "colab-gpu-c", "stage_ids": [5], "resource_kind": "colab_gpu"},
                {"mode": "cpu-tail-a", "stage_ids": [6], "resource_kind": "cpu"},
                {"mode": "cpu-tail-b", "stage_ids": [7], "resource_kind": "cpu"},
                {"mode": "cpu-tail-c", "stage_ids": [8], "resource_kind": "cpu"},
                {"mode": "cpu-tail-d", "stage_ids": [9], "resource_kind": "cpu"},
            ]),
        ])
        coordinator_status = {
            "ready": True,
            "generated_token_count": 1,
            "activation_hashes": [f"sha256:a{i}" for i in range(9)],
            "stage_seen": list(range(10)),
            "stage_task_counts": {f"stage{i}": 1 for i in range(10)},
            "stage_kv_cache": {
                f"stage{i}": {
                    "schema": "kaggle_32b_full_stage_local_kv_cache_v1",
                    "stage_id": i,
                    "ready": True,
                    "hit_count": 0,
                    "expected_hit_count": 0,
                    "hit_target_ready": True,
                    "past_key_values_public": False,
                    "cache_tensors_public": False,
                }
                for i in range(10)
            },
            "completed_tasks": [],
        }
        stage_reports = [
            self._report_for("cpu-stage0", [0], "cpu", stage_ranges=stage_ranges),
            self._report_for("kaggle-gpu-a", [1, 2], "gpu", stage_ranges=stage_ranges),
            self._report_for("kaggle-gpu-b", [3, 4], "gpu", stage_ranges=stage_ranges),
            self._report_for("colab-gpu-c", [5], "colab_gpu", stage_ranges=stage_ranges),
            self._report_for("cpu-tail-a", [6], "cpu", stage_ranges=stage_ranges),
            self._report_for("cpu-tail-b", [7], "cpu", stage_ranges=stage_ranges),
            self._report_for("cpu-tail-c", [8], "cpu", stage_ranges=stage_ranges),
            self._report_for("cpu-tail-d", [9], "cpu", stage_ranges=stage_ranges),
        ]
        stage_runs = [
            {"mode": "cpu-stage0", "stage_ids": [0], "resource_kind": "cpu", "steps": [{"name": "kaggle_kernel_push", "accepted": True}, {"name": "kaggle_kernel_delete", "ok": True}]},
            {"mode": "kaggle-gpu-a", "stage_ids": [1, 2], "resource_kind": "gpu", "steps": [{"name": "kaggle_kernel_push", "accepted": True}, {"name": "kaggle_kernel_delete", "ok": True}]},
            {"mode": "kaggle-gpu-b", "stage_ids": [3, 4], "resource_kind": "gpu", "steps": [{"name": "kaggle_kernel_push", "accepted": True}, {"name": "kaggle_kernel_delete", "ok": True}]},
            {"mode": "colab-gpu-c", "stage_ids": [5], "resource_kind": "colab_gpu", "steps": [{"name": "colab_cuda_stage_worker", "accepted": True, "ok": True}]},
            {"mode": "cpu-tail-a", "stage_ids": [6], "resource_kind": "cpu", "steps": [{"name": "kaggle_kernel_push", "accepted": True}, {"name": "kaggle_kernel_delete", "ok": True}]},
            {"mode": "cpu-tail-b", "stage_ids": [7], "resource_kind": "cpu", "steps": [{"name": "kaggle_kernel_push", "accepted": True}, {"name": "kaggle_kernel_delete", "ok": True}]},
            {"mode": "cpu-tail-c", "stage_ids": [8], "resource_kind": "cpu", "steps": [{"name": "kaggle_kernel_push", "accepted": True}, {"name": "kaggle_kernel_delete", "ok": True}]},
            {"mode": "cpu-tail-d", "stage_ids": [9], "resource_kind": "cpu", "steps": [{"name": "kaggle_kernel_push", "accepted": True}, {"name": "kaggle_kernel_delete", "ok": True}]},
        ]

        report = probe.build_report(
            args,
            output_dir=self.tmp,
            coordinator_status=coordinator_status,
            stage_reports=stage_reports,
            stage_runs=stage_runs,
            errors=[],
        )

        self.assertTrue(report["ok"])
        self.assertTrue(report["kaggle_colab_gpu_cpu_same_request_verified"])
        self.assertTrue(report["same_request_72b_kaggle_colab_gpu_cpu_full_model_verified"])
        self.assertFalse(report["gpu_tpu_cpu_72b_same_request_verified"])
        self.assertFalse(report["same_request_72b_full_model_verified"])
        self.assertEqual(report["provider_stage_counts"], {"kaggle_cuda": 4, "colab_cuda": 1, "cpu": 5, "web_tpu": 0})
        self.assertEqual(report["kaggle_lifecycle"]["requested_topology"], "4KaggleGPU_stages_1ColabGPU_stages_0WebTPU_stages_5CPU_stages")

    def test_build_report_accepts_72b_full_layer_coverage_with_extra_cpu_stages(self) -> None:
        ranges_72b = [
            [0, 6],
            [6, 12],
            [12, 18],
            [18, 24],
            [24, 32],
            [32, 38],
            [38, 44],
            [44, 50],
            [50, 56],
            [56, 62],
            [62, 68],
            [68, 74],
            [74, 80],
        ]
        groups = [
            {"mode": "gpu-shard0", "stage_ids": [0, 1], "resource_kind": "gpu"},
            {"mode": "gpu-shard1", "stage_ids": [2, 3], "resource_kind": "gpu"},
            {"mode": "web-tpu-stage4", "stage_ids": [4], "resource_kind": "web_tpu"},
            *[
                {"mode": f"cpu-stage{i}", "stage_ids": [i], "resource_kind": "cpu"}
                for i in range(5, 13)
            ],
        ]
        args = probe.parse_args([
            "--output-dir",
            str(self.tmp),
            "--kaggle-owner",
            "tester",
            "--model-repo",
            "Qwen/Qwen2.5-72B-Instruct",
            "--max-new-tokens",
            "1",
            "--stage-ranges-json",
            json.dumps(ranges_72b),
            "--stage-groups-json",
            json.dumps(groups),
        ])
        coordinator_status = {
            "ready": True,
            "generated_token_count": 1,
            "activation_hashes": [f"sha256:a{i}" for i in range(12)],
            "stage_seen": list(range(13)),
            "stage_task_counts": {f"stage{i}": 1 for i in range(13)},
            "stage_kv_cache": {
                f"stage{i}": {
                    "schema": "kaggle_32b_full_stage_local_kv_cache_v1",
                    "stage_id": i,
                    "ready": True,
                    "hit_count": 0,
                    "expected_hit_count": 0,
                    "hit_target_ready": True,
                    "past_key_values_public": False,
                    "cache_tensors_public": False,
                }
                for i in range(13)
            },
            "completed_tasks": [],
        }
        stage_reports = [
            self._report_for("gpu-shard0", [0, 1], "gpu", stage_ranges=ranges_72b),
            self._report_for("gpu-shard1", [2, 3], "gpu", stage_ranges=ranges_72b),
            self._report_for("web-tpu-stage4", [4], "web_tpu", stage_ranges=ranges_72b),
            *[self._report_for(f"cpu-stage{i}", [i], "cpu", stage_ranges=ranges_72b) for i in range(5, 13)],
        ]
        stage_runs = [
            {"mode": "gpu-shard0", "stage_ids": [0, 1], "resource_kind": "gpu", "steps": [{"name": "kaggle_kernel_push", "accepted": True}, {"name": "kaggle_kernel_delete", "ok": True}]},
            {"mode": "gpu-shard1", "stage_ids": [2, 3], "resource_kind": "gpu", "steps": [{"name": "kaggle_kernel_push", "accepted": True}, {"name": "kaggle_kernel_delete", "ok": True}]},
            {"mode": "web-tpu-stage4", "stage_ids": [4], "resource_kind": "web_tpu", "steps": [{"name": "web_tpu_stage_worker", "accepted": True, "ok": True}]},
            *[
                {"mode": f"cpu-stage{i}", "stage_ids": [i], "resource_kind": "cpu", "steps": [{"name": "kaggle_kernel_push", "accepted": True}, {"name": "kaggle_kernel_delete", "ok": True}]}
                for i in range(5, 13)
            ],
        ]

        report = probe.build_report(
            args,
            output_dir=self.tmp,
            coordinator_status=coordinator_status,
            stage_reports=stage_reports,
            stage_runs=stage_runs,
            errors=[],
        )

        self.assertTrue(report["ok"])
        self.assertTrue(report["gpu_tpu_cpu_72b_same_request_verified"])
        self.assertTrue(report["same_request_72b_full_model_verified"])
        self.assertTrue(report["full_72b_layer_coverage_verified"])
        self.assertTrue(report["gpu_tpu_cpu_72b_full_topology_verified"])
        self.assertFalse(report["four_t4_one_tpu_five_cpu_topology_verified"])
        self.assertEqual(report["model"]["stage_count"], 13)
        self.assertEqual(report["model"]["expected_layer_count"], 80)
        self.assertEqual(report["kaggle_lifecycle"]["requested_cpu_kernels"], 8)
        self.assertIn("4GPU_stages_1WebTPU_stages_8CPU_stages", report["kaggle_lifecycle"]["requested_topology"])

    def test_build_report_rejects_72b_gap_in_layer_coverage(self) -> None:
        ranges_72b = [[0, 8], [8, 16], [20, 28]]
        groups = [
            {"mode": "gpu-stage0", "stage_ids": [0], "resource_kind": "gpu"},
            {"mode": "web-tpu-stage1", "stage_ids": [1], "resource_kind": "web_tpu"},
            {"mode": "cpu-stage2", "stage_ids": [2], "resource_kind": "cpu"},
        ]
        args = probe.parse_args([
            "--output-dir",
            str(self.tmp),
            "--kaggle-owner",
            "tester",
            "--model-repo",
            "Qwen/Qwen2.5-72B-Instruct",
            "--max-new-tokens",
            "1",
            "--stage-ranges-json",
            json.dumps(ranges_72b),
            "--stage-groups-json",
            json.dumps(groups),
        ])
        coordinator_status = {
            "ready": True,
            "generated_token_count": 1,
            "activation_hashes": ["sha256:a0", "sha256:a1"],
            "stage_seen": [0, 1, 2],
            "stage_task_counts": {"stage0": 1, "stage1": 1, "stage2": 1},
            "stage_kv_cache": {
                f"stage{i}": {
                    "schema": "kaggle_32b_full_stage_local_kv_cache_v1",
                    "stage_id": i,
                    "ready": True,
                    "hit_count": 0,
                    "expected_hit_count": 0,
                    "hit_target_ready": True,
                    "past_key_values_public": False,
                    "cache_tensors_public": False,
                }
                for i in range(3)
            },
            "completed_tasks": [],
        }
        stage_reports = [
            self._report_for("gpu-stage0", [0], "gpu", stage_ranges=ranges_72b),
            self._report_for("web-tpu-stage1", [1], "web_tpu", stage_ranges=ranges_72b),
            self._report_for("cpu-stage2", [2], "cpu", stage_ranges=ranges_72b),
        ]
        stage_runs = [
            {"mode": "gpu-stage0", "stage_ids": [0], "resource_kind": "gpu", "steps": [{"name": "kaggle_kernel_push", "accepted": True}, {"name": "kaggle_kernel_delete", "ok": True}]},
            {"mode": "web-tpu-stage1", "stage_ids": [1], "resource_kind": "web_tpu", "steps": [{"name": "web_tpu_stage_worker", "accepted": True, "ok": True}]},
            {"mode": "cpu-stage2", "stage_ids": [2], "resource_kind": "cpu", "steps": [{"name": "kaggle_kernel_push", "accepted": True}, {"name": "kaggle_kernel_delete", "ok": True}]},
        ]

        report = probe.build_report(
            args,
            output_dir=self.tmp,
            coordinator_status=coordinator_status,
            stage_reports=stage_reports,
            stage_runs=stage_runs,
            errors=[],
        )

        self.assertFalse(report["ok"])
        self.assertFalse(report["gpu_tpu_cpu_72b_same_request_verified"])
        self.assertFalse(report["same_request_72b_full_model_verified"])
        self.assertFalse(report["full_72b_layer_coverage_verified"])
        self.assertIn("dense_full_model_layer_coverage_not_verified", report["blockers"])

    def test_build_report_marks_full_heterogeneous_topology_ready(self) -> None:
        args = self._args()
        coordinator_status = {
            "ready": True,
            "generated_token_count": 2,
            "activation_hashes": [f"sha256:a{i}" for i in range(8)],
            "stage_seen": list(range(9)),
            "stage_task_counts": {f"stage{i}": 2 for i in range(9)},
            "stage_kv_cache": {
                f"stage{i}": {
                    "schema": "kaggle_32b_full_stage_local_kv_cache_v1",
                    "stage_id": i,
                    "ready": True,
                    "hit_count": 1,
                    "expected_hit_count": 1,
                    "hit_target_ready": True,
                    "past_key_values_public": False,
                    "cache_tensors_public": False,
                }
                for i in range(9)
            },
            "completed_tasks": [],
        }
        stage_reports = [
            self._report_for("gpu-shard0", [0, 1], "gpu"),
            self._report_for("gpu-shard1", [2, 3], "gpu"),
            *[self._report_for(f"cpu-stage{i}", [i], "cpu") for i in range(4, 9)],
        ]
        stage_runs = [
            {
                "mode": "gpu-shard0",
                "stage_ids": [0, 1],
                "resource_kind": "gpu",
                "steps": [{"name": "kaggle_kernel_push", "accepted": True}, {"name": "kaggle_kernel_delete", "ok": True}],
            },
            {
                "mode": "gpu-shard1",
                "stage_ids": [2, 3],
                "resource_kind": "gpu",
                "steps": [{"name": "kaggle_kernel_push", "accepted": True}, {"name": "kaggle_kernel_delete", "ok": True}],
            },
            *[
                {
                    "mode": f"cpu-stage{i}",
                    "stage_ids": [i],
                    "resource_kind": "cpu",
                    "steps": [{"name": "kaggle_kernel_push", "accepted": True}, {"name": "kaggle_kernel_delete", "ok": True}],
                }
                for i in range(4, 9)
            ],
        ]
        report = probe.build_report(
            args,
            output_dir=self.tmp,
            coordinator_status=coordinator_status,
            stage_reports=stage_reports,
            stage_runs=stage_runs,
            errors=[],
        )

        self.assertTrue(report["ok"])
        self.assertTrue(report["four_t4_five_cpu_topology_verified"])
        self.assertTrue(report["one_token_generation_verified"])
        self.assertTrue(report["multi_token_generation_verified"])
        self.assertTrue(report["stage_local_kv_cache_verified"])
        self.assertEqual(report["generated_token_count"], 2)
        self.assertEqual(len(report["stage_summaries"]), 9)
        self.assertEqual(report["kaggle_lifecycle"]["actual_gpu_push_count"], 2)
        self.assertEqual(report["kaggle_lifecycle"]["actual_cpu_push_count"], 5)
        encoded = json.dumps(report)
        self.assertNotIn("hidden_b64", encoded)
        self.assertNotIn("next_token_id_private", encoded)

    def test_build_report_accepts_72b_colab_tpu_with_single_cpu_tail_kernel(self) -> None:
        stage_ranges = [[0, 8], [8, 16], [16, 24], [24, 32], [32, 36], [36, 44], [44, 52], [52, 60], [60, 70], [70, 80]]
        args = probe.parse_args([
            "--output-dir",
            str(self.tmp),
            "--kaggle-owner",
            "tester",
            "--model-repo",
            "Qwen/Qwen2.5-72B-Instruct",
            "--stage-ranges-json",
            json.dumps(stage_ranges),
            "--stage-groups-json",
            json.dumps([
                {"mode": "gpu-shard0", "stage_ids": [0, 1], "resource_kind": "gpu"},
                {"mode": "gpu-shard1", "stage_ids": [2, 3], "resource_kind": "gpu"},
                {"mode": "web-tpu-stage4", "stage_ids": [4], "resource_kind": "web_tpu"},
                {"mode": "cpu-tail", "stage_ids": [5, 6, 7, 8, 9], "resource_kind": "cpu"},
            ]),
            "--tpu-provider",
            "colab_cli",
            "--max-new-tokens",
            "1",
            "--coordinator-timeout-seconds",
            "5",
            "--kaggle-status-timeout-seconds",
            "5",
        ])
        coordinator_status = {
            "ready": True,
            "generated_token_count": 1,
            "activation_hashes": [f"sha256:a{i}" for i in range(9)],
            "stage_seen": list(range(10)),
            "stage_task_counts": {f"stage{i}": 1 for i in range(10)},
            "stage_kv_cache": {
                f"stage{i}": {
                    "schema": "kaggle_32b_full_stage_local_kv_cache_v1",
                    "stage_id": i,
                    "ready": True,
                    "hit_count": 0,
                    "expected_hit_count": 0,
                    "hit_target_ready": True,
                    "past_key_values_public": False,
                    "cache_tensors_public": False,
                }
                for i in range(10)
            },
            "completed_tasks": [],
        }
        stage_reports = [
            self._report_for("gpu-shard0", [0, 1], "gpu", stage_ranges=stage_ranges),
            self._report_for("gpu-shard1", [2, 3], "gpu", stage_ranges=stage_ranges),
            self._report_for("web-tpu-stage4", [4], "web_tpu", stage_ranges=stage_ranges),
            self._report_for("cpu-tail", [5, 6, 7, 8, 9], "cpu", stage_ranges=stage_ranges),
        ]
        stage_runs = [
            {
                "mode": "gpu-shard0",
                "stage_ids": [0, 1],
                "resource_kind": "gpu",
                "steps": [{"name": "kaggle_kernel_push", "accepted": True}, {"name": "kaggle_kernel_delete", "ok": True}],
            },
            {
                "mode": "gpu-shard1",
                "stage_ids": [2, 3],
                "resource_kind": "gpu",
                "steps": [{"name": "kaggle_kernel_push", "accepted": True}, {"name": "kaggle_kernel_delete", "ok": True}],
            },
            {
                "mode": "web-tpu-stage4",
                "stage_ids": [4],
                "resource_kind": "web_tpu",
                "steps": [{"name": "web_tpu_stage_worker", "ok": True, "accepted": True, "resource_kind": "web_tpu"}],
            },
            {
                "mode": "cpu-tail",
                "stage_ids": [5, 6, 7, 8, 9],
                "resource_kind": "cpu",
                "steps": [{"name": "kaggle_kernel_push", "accepted": True}, {"name": "kaggle_kernel_delete", "ok": True}],
            },
        ]

        report = probe.build_report(
            args,
            output_dir=self.tmp,
            coordinator_status=coordinator_status,
            stage_reports=stage_reports,
            stage_runs=stage_runs,
            errors=[],
        )

        self.assertTrue(report["ok"])
        self.assertTrue(report["gpu_tpu_cpu_72b_same_request_verified"])
        self.assertTrue(report["same_request_72b_full_model_verified"])
        self.assertTrue(report["full_72b_weight_loading_public_claim"])
        self.assertTrue(report["full_72b_layer_coverage_verified"])
        self.assertTrue(report["gpu_tpu_cpu_72b_full_topology_verified"])
        self.assertTrue(report["four_t4_one_tpu_five_cpu_topology_verified"])
        self.assertEqual(report["kaggle_lifecycle"]["actual_cpu_push_count"], 1)
        self.assertEqual(report["kaggle_lifecycle"]["requested_topology"], "4GPU_stages_1WebTPU_stages_5CPU_stages")
        self.assertEqual(report["generated_token_count"], 1)

    def test_build_report_accepts_72b_single_gpu_web_tpu_cpu_tail_topology(self) -> None:
        stage_ranges = [[0, 8], [8, 16], [16, 24], [24, 32], [32, 36], [36, 44], [44, 52], [52, 60], [60, 70], [70, 80]]
        args = probe.parse_args([
            "--output-dir",
            str(self.tmp),
            "--kaggle-owner",
            "tester",
            "--model-repo",
            "Qwen/Qwen2.5-72B-Instruct",
            "--stage-ranges-json",
            json.dumps(stage_ranges),
            "--stage-groups-json",
            json.dumps([
                {"mode": "gpu-shard0", "stage_ids": [0, 1], "resource_kind": "gpu"},
                {"mode": "web-tpu-stage4", "stage_ids": [4], "resource_kind": "web_tpu"},
                {"mode": "cpu-tail", "stage_ids": [2, 3, 5, 6, 7, 8, 9], "resource_kind": "cpu"},
            ]),
            "--tpu-provider",
            "kaggle_web",
            "--max-new-tokens",
            "1",
            "--coordinator-timeout-seconds",
            "5",
            "--kaggle-status-timeout-seconds",
            "5",
        ])
        coordinator_status = {
            "ready": True,
            "generated_token_count": 1,
            "activation_hashes": [f"sha256:a{i}" for i in range(9)],
            "stage_seen": list(range(10)),
            "stage_task_counts": {f"stage{i}": 1 for i in range(10)},
            "stage_kv_cache": {
                f"stage{i}": {
                    "schema": "kaggle_32b_full_stage_local_kv_cache_v1",
                    "stage_id": i,
                    "ready": True,
                    "hit_count": 0,
                    "expected_hit_count": 0,
                    "hit_target_ready": True,
                    "past_key_values_public": False,
                    "cache_tensors_public": False,
                }
                for i in range(10)
            },
            "completed_tasks": [],
        }
        stage_reports = [
            self._report_for("gpu-shard0", [0, 1], "gpu", stage_ranges=stage_ranges),
            self._report_for("web-tpu-stage4", [4], "web_tpu", stage_ranges=stage_ranges),
            self._report_for("cpu-tail", [2, 3, 5, 6, 7, 8, 9], "cpu", stage_ranges=stage_ranges),
        ]
        stage_runs = [
            {
                "mode": "gpu-shard0",
                "stage_ids": [0, 1],
                "resource_kind": "gpu",
                "steps": [{"name": "kaggle_kernel_push", "accepted": True}, {"name": "kaggle_kernel_delete", "ok": True}],
            },
            {
                "mode": "web-tpu-stage4",
                "stage_ids": [4],
                "resource_kind": "web_tpu",
                "steps": [{"name": "web_tpu_stage_worker", "ok": True, "accepted": True, "resource_kind": "web_tpu"}],
            },
            {
                "mode": "cpu-tail",
                "stage_ids": [2, 3, 5, 6, 7, 8, 9],
                "resource_kind": "cpu",
                "steps": [{"name": "kaggle_kernel_push", "accepted": True}, {"name": "kaggle_kernel_delete", "ok": True}],
            },
        ]

        report = probe.build_report(
            args,
            output_dir=self.tmp,
            coordinator_status=coordinator_status,
            stage_reports=stage_reports,
            stage_runs=stage_runs,
            errors=[],
        )

        self.assertTrue(report["ok"])
        self.assertTrue(report["gpu_tpu_cpu_72b_same_request_verified"])
        self.assertTrue(report["same_request_72b_full_model_verified"])
        self.assertTrue(report["full_72b_layer_coverage_verified"])
        self.assertTrue(report["gpu_tpu_cpu_72b_full_topology_verified"])
        self.assertFalse(report["four_t4_one_tpu_five_cpu_topology_verified"])
        self.assertEqual(report["kaggle_lifecycle"]["actual_gpu_push_count"], 1)
        self.assertEqual(report["kaggle_lifecycle"]["actual_cpu_push_count"], 1)
        self.assertEqual(report["kaggle_lifecycle"]["requested_topology"], "2GPU_stages_1WebTPU_stages_7CPU_stages")
        self.assertEqual(report["generated_token_count"], 1)

    def test_build_report_accepts_72b_cpu_embedding_small_gpu_web_tpu_topology(self) -> None:
        stage_ranges = [[0, 1], [1, 5], [5, 9], [9, 32], [32, 36], [36, 50], [50, 65], [65, 80]]
        groups = [
            {"mode": "cpu-embed-stage0", "stage_ids": [0], "resource_kind": "cpu"},
            {"mode": "gpu-shard0", "stage_ids": [1, 2], "resource_kind": "gpu"},
            {"mode": "cpu-mid-stage3", "stage_ids": [3], "resource_kind": "cpu"},
            {"mode": "web-tpu-stage4", "stage_ids": [4], "resource_kind": "web_tpu"},
            {"mode": "cpu-tail-a", "stage_ids": [5], "resource_kind": "cpu"},
            {"mode": "cpu-tail-b", "stage_ids": [6], "resource_kind": "cpu"},
            {"mode": "cpu-tail-c", "stage_ids": [7], "resource_kind": "cpu"},
        ]
        args = probe.parse_args([
            "--output-dir",
            str(self.tmp),
            "--kaggle-owner",
            "tester",
            "--model-repo",
            "Qwen/Qwen2.5-72B-Instruct",
            "--stage-ranges-json",
            json.dumps(stage_ranges),
            "--stage-groups-json",
            json.dumps(groups),
            "--tpu-provider",
            "kaggle_web",
            "--max-new-tokens",
            "1",
            "--coordinator-timeout-seconds",
            "5",
            "--kaggle-status-timeout-seconds",
            "5",
        ])
        coordinator_status = {
            "ready": True,
            "generated_token_count": 1,
            "activation_hashes": [f"sha256:a{i}" for i in range(7)],
            "stage_seen": list(range(8)),
            "stage_task_counts": {f"stage{i}": 1 for i in range(8)},
            "stage_kv_cache": {
                f"stage{i}": {
                    "schema": "kaggle_32b_full_stage_local_kv_cache_v1",
                    "stage_id": i,
                    "ready": True,
                    "hit_count": 0,
                    "expected_hit_count": 0,
                    "hit_target_ready": True,
                    "past_key_values_public": False,
                    "cache_tensors_public": False,
                }
                for i in range(8)
            },
            "completed_tasks": [],
        }
        stage_reports = [
            self._report_for("cpu-embed-stage0", [0], "cpu", stage_ranges=stage_ranges),
            self._report_for("gpu-shard0", [1, 2], "gpu", stage_ranges=stage_ranges),
            self._report_for("cpu-mid-stage3", [3], "cpu", stage_ranges=stage_ranges),
            self._report_for("web-tpu-stage4", [4], "web_tpu", stage_ranges=stage_ranges),
            self._report_for("cpu-tail-a", [5], "cpu", stage_ranges=stage_ranges),
            self._report_for("cpu-tail-b", [6], "cpu", stage_ranges=stage_ranges),
            self._report_for("cpu-tail-c", [7], "cpu", stage_ranges=stage_ranges),
        ]
        stage_runs = [
            {"mode": "cpu-embed-stage0", "stage_ids": [0], "resource_kind": "cpu", "steps": [{"name": "kaggle_kernel_push", "accepted": True}, {"name": "kaggle_kernel_delete", "ok": True}]},
            {"mode": "gpu-shard0", "stage_ids": [1, 2], "resource_kind": "gpu", "steps": [{"name": "kaggle_kernel_push", "accepted": True}, {"name": "kaggle_kernel_delete", "ok": True}]},
            {"mode": "cpu-mid-stage3", "stage_ids": [3], "resource_kind": "cpu", "steps": [{"name": "kaggle_kernel_push", "accepted": True}, {"name": "kaggle_kernel_delete", "ok": True}]},
            {"mode": "web-tpu-stage4", "stage_ids": [4], "resource_kind": "web_tpu", "steps": [{"name": "web_tpu_stage_worker", "ok": True, "accepted": True, "resource_kind": "web_tpu"}]},
            {"mode": "cpu-tail-a", "stage_ids": [5], "resource_kind": "cpu", "steps": [{"name": "kaggle_kernel_push", "accepted": True}, {"name": "kaggle_kernel_delete", "ok": True}]},
            {"mode": "cpu-tail-b", "stage_ids": [6], "resource_kind": "cpu", "steps": [{"name": "kaggle_kernel_push", "accepted": True}, {"name": "kaggle_kernel_delete", "ok": True}]},
            {"mode": "cpu-tail-c", "stage_ids": [7], "resource_kind": "cpu", "steps": [{"name": "kaggle_kernel_push", "accepted": True}, {"name": "kaggle_kernel_delete", "ok": True}]},
        ]

        report = probe.build_report(
            args,
            output_dir=self.tmp,
            coordinator_status=coordinator_status,
            stage_reports=stage_reports,
            stage_runs=stage_runs,
            errors=[],
        )

        self.assertTrue(report["ok"])
        self.assertTrue(report["gpu_tpu_cpu_72b_same_request_verified"])
        self.assertTrue(report["same_request_72b_full_model_verified"])
        self.assertTrue(report["full_72b_layer_coverage_verified"])
        self.assertTrue(report["gpu_tpu_cpu_72b_full_topology_verified"])
        self.assertFalse(report["four_t4_one_tpu_five_cpu_topology_verified"])
        self.assertEqual(report["kaggle_lifecycle"]["requested_topology"], "2GPU_stages_1WebTPU_stages_5CPU_stages")
        self.assertEqual(report["kaggle_lifecycle"]["actual_cpu_push_count"], 5)
        self.assertEqual(report["generated_token_count"], 1)

    def test_run_coordinator_probe_assembles_public_safe_report_from_workers(self) -> None:
        args = self._args()

        def fake_run_kaggle(args_obj, *, package, output_dir, runner, cleanup=True):
            mode = str(package["mode"])
            deadline = 1000
            while not state.ready() and deadline > 0:
                deadline -= 1
                progressed = False
                for stage_id in list(package.get("stage_ids") or []):
                    task = state.claim(miner_id=mode, stage_id=int(stage_id))["task"]
                    if not task:
                        continue
                    progressed = True
                    step = int(task["generation_step"])
                    kv_cache = {
                        "schema": "kaggle_32b_full_stage_local_kv_cache_v1",
                        "stage_id": int(stage_id),
                        "ready": True,
                        "hit_count": step,
                        "miss_count": 1,
                        "prefill_count": 1,
                        "expected_hit_count": 1,
                        "hit_target_ready": step >= 1,
                        "tokens_before": 2 + step - 1 if step else 0,
                        "tokens_after": 2 + step,
                        "past_key_values_public": False,
                        "cache_tensors_public": False,
                    }
                    if int(stage_id) < 8:
                        state.submit({
                            "task_id": task["task_id"],
                            "stage_id": int(stage_id),
                            "generation_step": step,
                            "activation": {
                                "activation_hash": f"sha256:a{stage_id}",
                                "input_ids": task.get("input_ids") or [1, 2],
                                "hidden_b64": "private",
                            },
                            "kv_cache": kv_cache,
                            "duration_seconds": 1.0,
                        })
                    else:
                        state.submit({
                            "task_id": task["task_id"],
                            "stage_id": int(stage_id),
                            "generation_step": step,
                            "activation_hash": task.get("activation_hash"),
                            "next_token_id_private": 3,
                            "next_token_hash": f"sha256:t{step}",
                            "output_hash": f"sha256:o{step}",
                            "generated_token_count": 1,
                            "kv_cache": kv_cache,
                            "duration_seconds": 1.0,
                        })
                if not progressed:
                    probe.time.sleep(0.001)
            return self._report_for(mode, list(package.get("stage_ids") or []), str(package.get("resource_kind"))), [
                {"name": "kaggle_kernel_push", "accepted": True},
                {"name": "kaggle_kernel_delete", "ok": True},
            ]

        captured: dict[str, probe.StageCoordinatorState] = {}
        original_init = probe.ProbeCoordinatorServer.__init__

        def init_spy(self, *, host, port, token, state):
            captured["state"] = state
            return original_init(self, host=host, port=port, token=token, state=state)

        with mock.patch.object(probe.ProbeCoordinatorServer, "__init__", init_spy):
            with mock.patch.object(probe, "run_kaggle_package") as mocked_run:
                def side_effect(*call_args, **kwargs):
                    nonlocal state
                    state = captured["state"]
                    return fake_run_kaggle(*call_args, **kwargs)

                state = probe.StageCoordinatorState(prompt="unused", max_new_tokens=2, stage_count=9)
                mocked_run.side_effect = side_effect
                report = probe.run_coordinator_probe(args)

        self.assertTrue(report["ok"])
        self.assertTrue(report["four_t4_five_cpu_topology_verified"])
        self.assertTrue(report["stage_local_kv_cache_verified"])
        self.assertFalse((self.tmp / "coordinator-private-state.json").exists())
        encoded = json.dumps(report)
        self.assertNotIn("hidden_b64", encoded)
        self.assertNotIn("next_token_id_private", encoded)

    def test_run_coordinator_probe_can_assemble_single_gpu_72b_topology(self) -> None:
        stage_ranges = [[0, 8], [8, 16], [16, 24], [24, 32], [32, 36], [36, 44], [44, 52], [52, 60], [60, 70], [70, 80]]
        args = probe.parse_args([
            "--output-dir",
            str(self.tmp),
            "--kaggle-owner",
            "tester",
            "--model-repo",
            "Qwen/Qwen2.5-72B-Instruct",
            "--stage-ranges-json",
            json.dumps(stage_ranges),
            "--stage-groups-json",
            json.dumps([
                {"mode": "gpu-shard0", "stage_ids": [0, 1], "resource_kind": "gpu"},
                {"mode": "web-tpu-stage4", "stage_ids": [4], "resource_kind": "web_tpu"},
                {"mode": "cpu-tail", "stage_ids": [2, 3, 5, 6, 7, 8, 9], "resource_kind": "cpu"},
            ]),
            "--port",
            "0",
            "--max-new-tokens",
            "1",
            "--coordinator-timeout-seconds",
            "5",
            "--kaggle-status-timeout-seconds",
            "5",
        ])

        def submit_task(state: probe.StageCoordinatorState, task: dict, stage_id: int) -> None:
            kv_cache = {
                "schema": "kaggle_32b_full_stage_local_kv_cache_v1",
                "stage_id": stage_id,
                "ready": True,
                "hit_count": 0,
                "expected_hit_count": 0,
                "hit_target_ready": True,
                "past_key_values_public": False,
                "cache_tensors_public": False,
            }
            if stage_id < 9:
                state.submit({
                    "task_id": task["task_id"],
                    "stage_id": stage_id,
                    "generation_step": 0,
                    "activation": {
                        "activation_hash": f"sha256:a{stage_id}",
                        "input_ids": task.get("input_ids") or [1, 2],
                        "hidden_b64": "private",
                    },
                    "kv_cache": kv_cache,
                    "duration_seconds": 1.0,
                })
            else:
                state.submit({
                    "task_id": task["task_id"],
                    "stage_id": stage_id,
                    "generation_step": 0,
                    "activation_hash": task.get("activation_hash"),
                    "next_token_id_private": 3,
                    "next_token_hash": "sha256:t0",
                    "output_hash": "sha256:o0",
                    "generated_token_count": 1,
                    "kv_cache": kv_cache,
                    "duration_seconds": 1.0,
                })

        captured: dict[str, probe.StageCoordinatorState] = {}
        original_init = probe.ProbeCoordinatorServer.__init__

        def init_spy(self, *, host, port, token, state):
            captured["state"] = state
            return original_init(self, host=host, port=port, token=token, state=state)

        def fake_run_kaggle(args_obj, *, package, output_dir, runner, cleanup=True):
            state = captured["state"]
            mode = str(package["mode"])
            deadline = 1000
            while not state.ready() and deadline > 0:
                deadline -= 1
                progressed = False
                for stage_id in list(package.get("stage_ids") or []):
                    claim = state.claim(miner_id=mode, stage_id=int(stage_id))
                    task = claim.get("task") if isinstance(claim.get("task"), dict) else {}
                    if not task:
                        continue
                    progressed = True
                    submit_task(state, task, int(stage_id))
                if not progressed:
                    probe.time.sleep(0.001)
            return self._report_for(mode, list(package.get("stage_ids") or []), str(package.get("resource_kind")), stage_ranges=stage_ranges), [
                {"name": "kaggle_kernel_push", "accepted": True},
                {"name": "kaggle_kernel_delete", "ok": True},
            ]

        def fake_web_tpu_worker(args_obj, *, stage_id, token, timeout_seconds):
            state = captured["state"]
            deadline = 1000
            while not state.ready() and deadline > 0:
                deadline -= 1
                claim = state.claim(miner_id=f"web-tpu-stage{stage_id}", stage_id=int(stage_id))
                task = claim.get("task") if isinstance(claim.get("task"), dict) else {}
                if task:
                    submit_task(state, task, int(stage_id))
                    break
                probe.time.sleep(0.001)
            return self._report_for(f"web-tpu-stage{stage_id}", [int(stage_id)], "web_tpu", stage_ranges=stage_ranges)

        with mock.patch.object(probe.ProbeCoordinatorServer, "__init__", init_spy):
            with mock.patch.object(probe, "run_kaggle_package", side_effect=fake_run_kaggle):
                with mock.patch.object(probe, "web_tpu_stage_worker", side_effect=fake_web_tpu_worker):
                    report = probe.run_coordinator_probe(args)

        self.assertTrue(report["ok"])
        self.assertTrue(report["gpu_tpu_cpu_72b_same_request_verified"])
        self.assertTrue(report["same_request_72b_full_model_verified"])
        self.assertEqual(report["kaggle_lifecycle"]["actual_gpu_push_count"], 1)
        self.assertEqual(report["kaggle_lifecycle"]["actual_cpu_push_count"], 1)
        self.assertEqual(report["kaggle_lifecycle"]["requested_topology"], "2GPU_stages_1WebTPU_stages_7CPU_stages")
        self.assertFalse(report["four_t4_one_tpu_five_cpu_topology_verified"])

    def test_run_coordinator_probe_can_assemble_kaggle_colab_gpu_cpu_topology(self) -> None:
        stage_ranges = [[0, 8], [8, 16], [16, 24]]
        args = probe.parse_args([
            "--output-dir",
            str(self.tmp),
            "--kaggle-owner",
            "tester",
            "--model-repo",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--stage-ranges-json",
            json.dumps(stage_ranges),
            "--stage-groups-json",
            json.dumps([
                {"mode": "kaggle-gpu-stage0", "stage_ids": [0], "resource_kind": "gpu"},
                {"mode": "colab-gpu-stage1", "stage_ids": [1], "resource_kind": "colab_gpu"},
                {"mode": "cpu-stage2", "stage_ids": [2], "resource_kind": "cpu"},
            ]),
            "--port",
            "0",
            "--max-new-tokens",
            "1",
            "--coordinator-timeout-seconds",
            "5",
            "--kaggle-status-timeout-seconds",
            "5",
        ])

        def submit_task(state: probe.StageCoordinatorState, task: dict, stage_id: int) -> None:
            kv_cache = {
                "schema": "kaggle_32b_full_stage_local_kv_cache_v1",
                "stage_id": stage_id,
                "ready": True,
                "hit_count": 0,
                "expected_hit_count": 0,
                "hit_target_ready": True,
                "past_key_values_public": False,
                "cache_tensors_public": False,
            }
            if stage_id < 2:
                state.submit({
                    "task_id": task["task_id"],
                    "stage_id": stage_id,
                    "generation_step": 0,
                    "activation": {
                        "activation_hash": f"sha256:a{stage_id}",
                        "input_ids": task.get("input_ids") or [1, 2],
                        "hidden_b64": "private",
                    },
                    "kv_cache": kv_cache,
                    "duration_seconds": 1.0,
                })
            else:
                state.submit({
                    "task_id": task["task_id"],
                    "stage_id": stage_id,
                    "generation_step": 0,
                    "activation_hash": task.get("activation_hash"),
                    "next_token_id_private": 3,
                    "next_token_hash": "sha256:t0",
                    "output_hash": "sha256:o0",
                    "generated_token_count": 1,
                    "kv_cache": kv_cache,
                    "duration_seconds": 1.0,
                })

        captured: dict[str, probe.StageCoordinatorState] = {}
        original_init = probe.ProbeCoordinatorServer.__init__

        def init_spy(self, *, host, port, token, state):
            captured["state"] = state
            return original_init(self, host=host, port=port, token=token, state=state)

        def fake_run_kaggle(args_obj, *, package, output_dir, runner, cleanup=True):
            state = captured["state"]
            mode = str(package["mode"])
            deadline = 1000
            while not state.ready() and deadline > 0:
                deadline -= 1
                progressed = False
                for stage_id in list(package.get("stage_ids") or []):
                    claim = state.claim(miner_id=mode, stage_id=int(stage_id))
                    task = claim.get("task") if isinstance(claim.get("task"), dict) else {}
                    if not task:
                        continue
                    progressed = True
                    submit_task(state, task, int(stage_id))
                if not progressed:
                    probe.time.sleep(0.001)
            return self._report_for(mode, list(package.get("stage_ids") or []), str(package.get("resource_kind")), stage_ranges=stage_ranges), [
                {"name": "kaggle_kernel_push", "accepted": True},
                {"name": "kaggle_kernel_delete", "ok": True},
            ]

        def fake_colab_worker(args_obj, *, mode, stage_ids, token):
            state = captured["state"]
            deadline = 1000
            while not state.ready() and deadline > 0:
                deadline -= 1
                progressed = False
                for stage_id in list(stage_ids):
                    claim = state.claim(miner_id=mode, stage_id=int(stage_id))
                    task = claim.get("task") if isinstance(claim.get("task"), dict) else {}
                    if not task:
                        continue
                    progressed = True
                    submit_task(state, task, int(stage_id))
                if not progressed:
                    probe.time.sleep(0.001)
            return self._report_for(mode, list(stage_ids), "colab_gpu", stage_ranges=stage_ranges)

        with mock.patch.object(probe.ProbeCoordinatorServer, "__init__", init_spy):
            with mock.patch.object(probe, "run_kaggle_package", side_effect=fake_run_kaggle):
                with mock.patch.object(probe, "colab_cuda_stage_worker", side_effect=fake_colab_worker):
                    report = probe.run_coordinator_probe(args)

        self.assertTrue(report["ok"])
        self.assertTrue(report["kaggle_colab_gpu_cpu_same_request_verified"])
        self.assertTrue(report["colab_cuda_provider_verified"])
        self.assertIn("kaggle_cuda", report["accepted_providers"])
        self.assertIn("colab_cuda", report["accepted_providers"])
        self.assertIn("cpu", report["accepted_providers"])
        self.assertEqual(report["kaggle_lifecycle"]["actual_gpu_push_count"], 1)
        self.assertEqual(report["kaggle_lifecycle"]["actual_colab_gpu_runtime_count"], 1)
        self.assertEqual(report["kaggle_lifecycle"]["actual_cpu_push_count"], 1)
        self.assertEqual(report["kaggle_lifecycle"]["requested_topology"], "1KaggleGPU_stages_1ColabGPU_stages_0WebTPU_stages_1CPU_stages")

    def test_run_coordinator_probe_waits_for_slow_worker_reports_after_ready(self) -> None:
        stage_ranges = [[0, 8], [8, 16], [16, 24]]
        args = probe.parse_args([
            "--output-dir",
            str(self.tmp),
            "--kaggle-owner",
            "tester",
            "--model-repo",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--stage-ranges-json",
            json.dumps(stage_ranges),
            "--stage-groups-json",
            json.dumps([
                {"mode": "kaggle-gpu-stage0", "stage_ids": [0], "resource_kind": "gpu"},
                {"mode": "colab-gpu-stage1", "stage_ids": [1], "resource_kind": "colab_gpu"},
                {"mode": "cpu-stage2", "stage_ids": [2], "resource_kind": "cpu"},
            ]),
            "--port",
            "0",
            "--max-new-tokens",
            "1",
            "--coordinator-timeout-seconds",
            "5",
            "--kaggle-status-timeout-seconds",
            "1",
        ])

        captured: dict[str, probe.StageCoordinatorState] = {}
        original_init = probe.ProbeCoordinatorServer.__init__

        def init_spy(self, *, host, port, token, state):
            captured["state"] = state
            return original_init(self, host=host, port=port, token=token, state=state)

        def submit_task(state: probe.StageCoordinatorState, task: dict, stage_id: int) -> None:
            kv_cache = {
                "schema": "kaggle_32b_full_stage_local_kv_cache_v1",
                "stage_id": stage_id,
                "ready": True,
                "hit_count": 0,
                "expected_hit_count": 0,
                "hit_target_ready": True,
                "past_key_values_public": False,
                "cache_tensors_public": False,
            }
            if stage_id < 2:
                state.submit({
                    "task_id": task["task_id"],
                    "stage_id": stage_id,
                    "generation_step": 0,
                    "activation": {
                        "activation_hash": f"sha256:a{stage_id}",
                        "input_ids": task.get("input_ids") or [1, 2],
                        "hidden_b64": "private",
                    },
                    "kv_cache": kv_cache,
                    "duration_seconds": 1.0,
                })
            else:
                state.submit({
                    "task_id": task["task_id"],
                    "stage_id": stage_id,
                    "generation_step": 0,
                    "activation_hash": task.get("activation_hash"),
                    "next_token_id_private": 3,
                    "next_token_hash": "sha256:t0",
                    "output_hash": "sha256:o0",
                    "generated_token_count": 1,
                    "kv_cache": kv_cache,
                    "duration_seconds": 1.0,
                })

        def fake_run_kaggle(args_obj, *, package, output_dir, runner, cleanup=True):
            state = captured["state"]
            mode = str(package["mode"])
            while True:
                progressed = False
                for stage_id in list(package.get("stage_ids") or []):
                    claim = state.claim(miner_id=mode, stage_id=int(stage_id))
                    task = claim.get("task") if isinstance(claim.get("task"), dict) else {}
                    if not task:
                        continue
                    progressed = True
                    submit_task(state, task, int(stage_id))
                if state.ready() or progressed:
                    break
                probe.time.sleep(0.001)
            return self._report_for(mode, list(package.get("stage_ids") or []), str(package.get("resource_kind")), stage_ranges=stage_ranges), [
                {"name": "kaggle_kernel_push", "accepted": True},
                {"name": "kaggle_kernel_delete", "ok": True},
            ]

        def fake_colab_worker(args_obj, *, mode, stage_ids, token):
            state = captured["state"]
            while True:
                progressed = False
                for stage_id in list(stage_ids):
                    claim = state.claim(miner_id=mode, stage_id=int(stage_id))
                    task = claim.get("task") if isinstance(claim.get("task"), dict) else {}
                    if not task:
                        continue
                    progressed = True
                    submit_task(state, task, int(stage_id))
                if state.ready() or progressed:
                    break
                probe.time.sleep(0.001)
            return self._report_for(mode, list(stage_ids), "colab_gpu", stage_ranges=stage_ranges)

        with mock.patch.object(probe.ProbeCoordinatorServer, "__init__", init_spy):
            with mock.patch.object(probe, "run_kaggle_package", side_effect=fake_run_kaggle):
                with mock.patch.object(probe, "colab_cuda_stage_worker", side_effect=fake_colab_worker):
                    report = probe.run_coordinator_probe(args)

        self.assertTrue(report["ok"])
        self.assertEqual(len(report["stage_runs"]), 3)
        self.assertEqual(len(report["stage_summaries"]), 3)
        self.assertEqual(report["accepted_providers"], ["kaggle_cuda", "colab_cuda", "cpu"])

    def test_run_coordinator_probe_preserves_colab_failure_diagnostics(self) -> None:
        args = probe.parse_args([
            "--output-dir",
            str(self.tmp),
            "--kaggle-owner",
            "tester",
            "--model-repo",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--stage-ranges-json",
            "[[0,8],[8,16],[16,24]]",
            "--stage-groups-json",
            json.dumps([
                {"mode": "kaggle-gpu-stage0", "stage_ids": [0], "resource_kind": "gpu"},
                {"mode": "colab-gpu-stage1", "stage_ids": [1], "resource_kind": "colab_gpu"},
                {"mode": "cpu-stage2", "stage_ids": [2], "resource_kind": "cpu"},
            ]),
            "--port",
            "0",
            "--max-new-tokens",
            "1",
            "--coordinator-timeout-seconds",
            "5",
            "--kaggle-status-timeout-seconds",
            "5",
        ])

        failure = {
            "ok": False,
            "mode": "colab-gpu-stage1",
            "resource_kind": "colab_gpu",
            "provider_kind": "colab_cuda",
            "stage_ids": [1],
            "blockers": ["colab_cuda_execute_failed"],
            "diagnosis_codes": ["kaggle_full_heterogeneous_colab_cuda_stage_execute_failed"],
            "elapsed_seconds": 12.5,
            "session_manager": {
                "ok": False,
                "blocker": "colab_cuda_execute_failed",
                "attempt_count": 1,
                "attempts": [{"attempt": 1, "ok": False, "stale_detected": True}],
                "public_artifact_safe": True,
            },
            "public_artifact_safe": True,
        }

        def fake_run_kaggle(args_obj, *, package, output_dir, runner, cleanup=True):
            return self._report_for(
                str(package["mode"]),
                list(package.get("stage_ids") or []),
                str(package.get("resource_kind")),
                stage_ranges=[[0, 8], [8, 16], [16, 24]],
            ), [
                {"name": "kaggle_kernel_push", "accepted": True},
                {"name": "kaggle_kernel_delete", "ok": True},
            ]

        with mock.patch.object(probe, "run_kaggle_package", side_effect=fake_run_kaggle):
            with mock.patch.object(probe, "colab_cuda_stage_worker", return_value=failure):
                with mock.patch.object(probe.time, "sleep"):
                    report = probe.run_coordinator_probe(args)

        colab_run = [run for run in report["stage_runs"] if run["resource_kind"] == "colab_gpu"][0]
        step = colab_run["steps"][0]
        self.assertFalse(step["ok"])
        self.assertEqual(step["blockers"], ["colab_cuda_execute_failed"])
        self.assertEqual(step["diagnosis_codes"], ["kaggle_full_heterogeneous_colab_cuda_stage_execute_failed"])
        self.assertEqual(step["session_manager"]["blocker"], "colab_cuda_execute_failed")
        self.assertEqual(step["session_manager"]["attempts"][0]["stale_detected"], True)
        encoded = json.dumps(report)
        self.assertNotIn("runtime_proxy_token", encoded)
        self.assertNotIn("runtime_proxy_url", encoded)

    def _report_for(self, mode: str, stage_ids: list[int], resource_kind: str, *, stage_ranges: list[list[int]] | None = None) -> dict[str, object]:
        ranges = stage_ranges or probe.DEFAULT_STAGE_RANGES
        return {
            "ok": True,
            "mode": mode,
            "resource_kind": resource_kind,
            "stage_runtime_summaries": [
                {
                    "stage_id": stage_id,
                    "resource_kind": resource_kind,
                    "device": f"cuda:{index}" if resource_kind in {"gpu", "colab_gpu"} else ("jax_tpu" if resource_kind == "web_tpu" else "cpu"),
                    "selection": {
                        "stage_layer_range": ranges[stage_id],
                        "assigned_weight_key_count": 100,
                        "assigned_weight_file_count": 2,
                    },
                    "runtime_buffers": {"ready": True},
                    "stage_weight_load": {
                        "ready": True,
                        "loaded_weight_key_count": 100,
                        "loaded_tensor_gb": 6.0,
                        "prepared_tensor_gb": 6.0,
                        "runtime_dtype": "float16" if resource_kind in {"gpu", "colab_gpu"} else "bfloat16",
                    "loads_only_stage_weight_keys": True,
                },
                "kv_cache": {
                    "schema": "kaggle_32b_full_stage_local_kv_cache_v1",
                    "stage_id": stage_id,
                    "ready": True,
                    "hit_count": 1,
                    "expected_hit_count": 1,
                    "hit_target_ready": True,
                    "past_key_values_public": False,
                    "cache_tensors_public": False,
                },
                "cuda_memory_after_load": {"cuda_available": resource_kind in {"gpu", "colab_gpu"}},
                    "memory_after_load": {"mem_available_mb": 10000},
                }
                for index, stage_id in enumerate(stage_ids)
            ],
            "diagnosis_codes": ["kaggle_32b_full_stage_owned_runtime_ready"],
            "blockers": [],
        }


if __name__ == "__main__":
    unittest.main()
