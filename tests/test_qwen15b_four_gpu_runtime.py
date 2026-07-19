from __future__ import annotations

import json
import threading
import time
import urllib.error

import pytest
import torch

from crowdtensor.qwen15b_four_gpu_runtime import (
    QwenHTTPTransport,
    StageProcessClient,
    deserialize_private_tensors,
    four_stage_overlap_summary,
    public_runtime_report,
    run_kernel_a_once,
    run_kernel_b_once,
    serialize_private_tensors,
)
from crowdtensor.qwen15b_training import (
    QwenStageSpec,
    materialize_stage_shard,
    read_safetensors_header,
)
from crowdtensor.qwen15b_four_gpu_worker import compare_adapter_states, compare_losses
from safetensors.torch import save_file


class _FakeTransport:
    def __init__(self) -> None:
        self.payloads = {}
        self.events = []
        self.lock = threading.Lock()

    def put_tensors(self, **kwargs):
        key = (
            kwargs["run_kind"],
            kwargs["kind"],
            kwargs["step"],
            kwargs["microbatch"],
        )
        payload = serialize_private_tensors(kwargs["tensors"])
        with self.lock:
            self.payloads[key] = payload
        return {"payload_hash": "sha256:" + __import__("hashlib").sha256(payload).hexdigest()}

    def get_tensors(self, **kwargs):
        key = (
            kwargs["run_kind"],
            kwargs["kind"],
            kwargs["step"],
            kwargs["microbatch"],
        )
        with self.lock:
            payload = self.payloads.get(key)
        if payload is None:
            return None
        return deserialize_private_tensors(payload), {
            "payload_hash": "sha256:" + __import__("hashlib").sha256(payload).hexdigest(),
            "byte_count": len(payload),
            "tensor_count": len(deserialize_private_tensors(payload)),
        }

    def event(self, **kwargs):
        with self.lock:
            self.events.append(dict(kwargs))
        return {"ok": True}

    def register(self, **kwargs):
        return {"ok": True, "role": kwargs["role"]}

    def wait_generation(self, **_kwargs):
        return {"coordinator_generation": 1}


class _JSONResponse:
    def __init__(self, value: dict) -> None:
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.value).encode("utf-8")


class _FakeStage:
    def __init__(
        self,
        spec: QwenStageSpec,
        *,
        pid: int | None = None,
        resumed_step: int = 0,
        resumed_cursor: int = 0,
    ) -> None:
        self.spec = spec
        self.device = spec.device
        self.ready = {
            "pid": int(pid or (100 + spec.stage_id)),
            "stage_id": spec.stage_id,
            "device": spec.device,
            "cuda_live": True,
            "cuda_device_name_hash": f"sha256:gpu-{spec.stage_id}",
            "resumed": resumed_step > 0,
            "resumed_global_step": resumed_step,
            "resumed_dataset_cursor": resumed_cursor,
            "loaded_checkpoint_hash": "sha256:checkpoint",
        }
        self.busy = None
        self._response = None
        self._clock = time.time_ns()
        self._finish_count = 0

    @property
    def pid(self):
        return self.ready["pid"]

    def send(self, operation, **payload):
        assert self.busy is None
        self.busy = {"request_id": 1, "operation": operation, **payload}
        microbatch = int(payload.get("microbatch_id", -1))
        self._clock += 100
        interval = {
            "stage_id": self.spec.stage_id,
            "microbatch_id": microbatch,
            "started_ns": self._clock,
            "ended_ns": self._clock + 50,
        }
        if operation == "begin_step":
            result = {"begun": True}
        elif operation == "forward":
            value = torch.as_tensor(payload["value"]).float()
            if value.ndim == 2:
                value = value.unsqueeze(-1)
            result = {"activation": value + self.spec.stage_id + 1, "compute_interval": interval}
        elif operation == "loss_backward":
            hidden = torch.as_tensor(payload["hidden_states"])
            result = {
                "activation_gradient": torch.ones_like(hidden),
                "loss": 10.0 - 0.1 * self._finish_count - 0.01 * microbatch,
                "compute_interval": interval,
            }
        elif operation == "backward":
            incoming = torch.as_tensor(payload["activation_gradient"])
            result = {
                "activation_gradient": None if self.spec.stage_id == 0 else incoming,
                "compute_interval": interval,
            }
        elif operation == "finish_step":
            self._finish_count += 1
            result = {
                "global_step": payload["global_step"],
                "dataset_cursor": payload["dataset_cursor"],
                "lora_gradient_norm": 1.0,
                "checkpoint_hash": f"sha256:checkpoint-{self.spec.stage_id}-{self._finish_count}",
                "adapter_tensor_hash": f"sha256:adapter-{self.spec.stage_id}-{self._finish_count}",
            }
        elif operation == "adapter_state":
            result = {
                "adapter_state": {
                    f"model.layers.{self.spec.stage_id}.x.lora_A.weight": torch.ones(1)
                },
                "adapter_hash": f"sha256:adapter-{self.spec.stage_id}",
            }
        elif operation == "status":
            result = {
                "base_hash_before": f"sha256:base-{self.spec.stage_id}",
                "base_hash_after": f"sha256:base-{self.spec.stage_id}",
                "compute_intervals": [],
            }
        else:
            raise AssertionError(operation)
        self._response = (dict(self.busy), result)

    def poll(self, _timeout=0.0):
        return self._response is not None

    def recv(self):
        response = self._response
        self._response = None
        self.busy = None
        return response

    def call(self, operation, **kwargs):
        kwargs.pop("timeout", None)
        self.send(operation, **kwargs)
        return self.recv()[1]

    def force_stop(self):
        return True


def test_private_safetensors_transport_round_trip() -> None:
    values = {
        "activation": torch.randn(1, 4, 8, dtype=torch.float16),
        "labels": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
    }
    payload = serialize_private_tensors(values)
    loaded = deserialize_private_tensors(payload)
    assert set(loaded) == set(values)
    assert all(torch.equal(values[name], loaded[name]) for name in values)


def test_http_transport_retries_disconnect_and_reregisters_before_replay(
    monkeypatch,
) -> None:
    paths = []
    failed_event = False

    def urlopen(request, **_kwargs):
        nonlocal failed_event
        path = request.full_url.split("example.invalid", 1)[-1]
        paths.append(path)
        if path == "/qwen15b-training/event" and not failed_event:
            failed_event = True
            raise urllib.error.URLError("temporary disconnect")
        return _JSONResponse({"ok": True})

    monkeypatch.setattr("crowdtensor.qwen15b_four_gpu_runtime.urllib.request.urlopen", urlopen)
    transport = QwenHTTPTransport(
        coordinator_url="https://example.invalid",
        token="private",
        run_id="run",
        retry_attempts=3,
        retry_base_seconds=0,
        retry_max_seconds=0,
    )
    ready = [
        {
            "pid": 10 + stage,
            "stage_id": stage,
            "device": f"cuda:{stage}",
            "cuda_live": True,
            "cuda_device_name_hash": f"sha256:gpu{stage}",
        }
        for stage in (0, 1)
    ]
    transport.register(role="kernel_a", ready=ready)
    transport.event(
        role="kernel_a",
        run_kind="resumed",
        operation="checkpoint",
        stage_id=0,
        step=4,
    )
    assert paths == [
        "/qwen15b-training/register",
        "/qwen15b-training/event",
        "/qwen15b-training/register",
        "/qwen15b-training/event",
    ]
    report = transport.public_retry_report()
    assert report["retry_count"] == 1
    assert report["reconnect_registration_count"] == 1
    assert report["transient_error_classes"] == ["URLError"]
    assert "private" not in json.dumps(report)


def test_http_transport_does_not_retry_nontransient_http_400(monkeypatch) -> None:
    calls = 0

    def urlopen(request, **_kwargs):
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(request.full_url, 400, "bad", {}, None)

    monkeypatch.setattr("crowdtensor.qwen15b_four_gpu_runtime.urllib.request.urlopen", urlopen)
    transport = QwenHTTPTransport(
        coordinator_url="https://example.invalid",
        token="private",
        run_id="run",
        retry_attempts=3,
        retry_base_seconds=0,
    )
    with pytest.raises(RuntimeError, match="coordinator_http_400"):
        transport.status()
    assert calls == 1
    assert transport.public_retry_report()["retry_count"] == 0


def test_two_kernel_event_loops_exchange_real_tensor_payloads() -> None:
    transport = _FakeTransport()
    specs = [
        QwenStageSpec(0, "A", 0, 0, 1, owns_embedding=True),
        QwenStageSpec(1, "A", 1, 1, 2),
        QwenStageSpec(2, "B", 0, 2, 3),
        QwenStageSpec(3, "B", 1, 3, 4, owns_norm=True, owns_lm_head=True),
    ]
    results = {}

    def run_a():
        results["a"] = run_kernel_a_once(
            run_kind="baseline",
            clients=[_FakeStage(specs[0]), _FakeStage(specs[1])],
            transport=transport,
            train_rows=[
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
            steps=2,
            microbatch_count=2,
            wait_timeout=5,
        )

    def run_b():
        results["b"] = run_kernel_b_once(
            run_kind="baseline",
            clients=[_FakeStage(specs[2]), _FakeStage(specs[3])],
            transport=transport,
            steps=2,
            microbatch_count=2,
            wait_timeout=5,
        )

    threads = [threading.Thread(target=run_a), threading.Thread(target=run_b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert results["a"]["steps_completed"] == 2
    assert results["a"]["dataset_row_indexes"] == [0, 1, 2, 3]
    assert results["b"]["steps_completed"] == 2
    assert results["a"]["real_backward"] is True
    assert results["b"]["loss_reduced"] is True
    assert len([key for key in transport.payloads if key[1] == "activation"]) == 4
    assert len([key for key in transport.payloads if key[1] == "gradient"]) == 4


def test_elastic_segment_uses_absolute_steps_and_barrier_callbacks() -> None:
    transport = _FakeTransport()
    specs = [
        QwenStageSpec(0, "A", 0, 0, 1, owns_embedding=True),
        QwenStageSpec(1, "A", 1, 1, 2),
        QwenStageSpec(2, "B", 0, 2, 3),
        QwenStageSpec(3, "B", 1, 3, 4, owns_norm=True, owns_lm_head=True),
    ]
    results = {}
    callbacks = {"a": [], "b": []}

    def callback(role):
        def commit(step, cursor, stages):
            callbacks[role].append(
                (step, cursor, sorted(item["stage_id"] for item in stages))
            )
            return {"barrier_committed": True, "global_step": step}

        return commit

    def run_a():
        results["a"] = run_kernel_a_once(
            run_kind="elastic",
            clients=[
                _FakeStage(specs[0], resumed_step=2, resumed_cursor=4),
                _FakeStage(specs[1], resumed_step=2, resumed_cursor=4),
            ],
            transport=transport,
            train_rows=[[index, index + 1] for index in range(8)],
            steps=2,
            start_step=2,
            microbatch_count=2,
            wait_timeout=5,
            step_commit_callback=callback("a"),
        )

    def run_b():
        results["b"] = run_kernel_b_once(
            run_kind="elastic",
            clients=[
                _FakeStage(specs[2], resumed_step=2, resumed_cursor=4),
                _FakeStage(specs[3], resumed_step=2, resumed_cursor=4),
            ],
            transport=transport,
            steps=2,
            start_step=2,
            microbatch_count=2,
            wait_timeout=5,
            step_commit_callback=callback("b"),
        )

    threads = [threading.Thread(target=run_a), threading.Thread(target=run_b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert callbacks["a"] == [(3, 6, [0, 1]), (4, 8, [0, 1])]
    assert callbacks["b"] == [(3, 6, [2, 3]), (4, 8, [2, 3])]
    assert results["a"]["start_step"] == 2
    assert results["a"]["end_step"] == 4
    assert results["a"]["dataset_row_indexes"] == [4, 5, 6, 7]
    assert {
        key[2] for key in transport.payloads if key[1] in {"activation", "gradient"}
    } == {2, 3}
    assert [item["step"] for item in results["b"]["step_reports"]] == [3, 4]


def test_two_kernel_runtime_restarts_all_four_stages_from_step_checkpoint() -> None:
    transport = _FakeTransport()
    specs = [
        QwenStageSpec(0, "A", 0, 0, 1, owns_embedding=True),
        QwenStageSpec(1, "A", 1, 1, 2),
        QwenStageSpec(2, "B", 0, 2, 3),
        QwenStageSpec(3, "B", 1, 3, 4, owns_norm=True, owns_lm_head=True),
    ]
    results = {}

    def replacements(stage_ids):
        return [
            _FakeStage(specs[stage], pid=1000 + stage, resumed_step=1, resumed_cursor=2)
            for stage in stage_ids
        ]

    def run_a():
        results["a"] = run_kernel_a_once(
            run_kind="resumed",
            clients=[_FakeStage(specs[0]), _FakeStage(specs[1])],
            transport=transport,
            train_rows=[[1, 2, 3, 4]] * 4,
            steps=2,
            microbatch_count=2,
            wait_timeout=5,
            restart_pair_after_step=1,
            restart_pair_factory=lambda: replacements((0, 1)),
        )

    def run_b():
        results["b"] = run_kernel_b_once(
            run_kind="resumed",
            clients=[_FakeStage(specs[2]), _FakeStage(specs[3])],
            transport=transport,
            steps=2,
            microbatch_count=2,
            wait_timeout=5,
            restart_pair_after_step=1,
            restart_pair_factory=lambda: replacements((2, 3)),
        )

    threads = [threading.Thread(target=run_a), threading.Thread(target=run_b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert all(not thread.is_alive() for thread in threads)
    assert {
        item["stage_id"] for item in results["a"]["coordinator_restart_stage_recoveries"]
    } == {0, 1}
    assert {
        item["stage_id"] for item in results["b"]["coordinator_restart_stage_recoveries"]
    } == {2, 3}
    assert results["a"]["coordinator_restart_all_owned_stages_verified"] is True
    assert results["b"]["coordinator_restart_all_owned_stages_verified"] is True
    assert results["b"]["controlled_restart_verified"] is True
    assert results["a"]["steps_completed"] == 2
    assert results["b"]["steps_completed"] == 2


def test_resumed_kernel_b_forces_stage2_restart_from_step_checkpoint() -> None:
    transport = _FakeTransport()
    specs = [
        QwenStageSpec(0, "A", 0, 0, 1, owns_embedding=True),
        QwenStageSpec(1, "A", 1, 1, 2),
        QwenStageSpec(2, "B", 0, 2, 3),
        QwenStageSpec(3, "B", 1, 3, 4, owns_norm=True, owns_lm_head=True),
    ]
    results = {}

    def run_a():
        results["a"] = run_kernel_a_once(
            run_kind="resumed",
            clients=[_FakeStage(specs[0]), _FakeStage(specs[1])],
            transport=transport,
            train_rows=[
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
            steps=2,
            microbatch_count=2,
            wait_timeout=5,
        )

    def run_b():
        clients = [_FakeStage(specs[2], pid=202), _FakeStage(specs[3], pid=203)]
        results["b"] = run_kernel_b_once(
            run_kind="resumed",
            clients=clients,
            transport=transport,
            steps=2,
            microbatch_count=2,
            wait_timeout=5,
            restart_stage2_after_step=1,
            restart_stage2_factory=lambda: _FakeStage(
                specs[2], pid=302, resumed_step=1, resumed_cursor=2
            ),
        )

    threads = [threading.Thread(target=run_a), threading.Thread(target=run_b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert results["b"]["controlled_restart_verified"] is True
    restart = results["b"]["controlled_restarts"][0]
    assert restart["old_pid"] == 202
    assert restart["new_pid"] == 302
    assert restart["resumed_global_step"] == 1
    assert restart["resumed_dataset_cursor"] == 2


def test_overlap_gate_requires_all_four_distinct_stages_at_same_time() -> None:
    events = [
        {
            "run_kind": "baseline",
            "step": 0,
            "stage_id": stage,
            "started_ns": 100 + stage * 5,
            "ended_ns": 200 - stage * 5,
        }
        for stage in range(4)
    ]
    result = four_stage_overlap_summary(events)
    assert result["four_stage_compute_overlap_verified"] is True
    assert result["maximum_four_stage_overlap"]["duration_ns"] == 70
    events[-1]["started_ns"] = 300
    events[-1]["ended_ns"] = 400
    assert four_stage_overlap_summary(events)["four_stage_compute_overlap_verified"] is False


def test_public_runtime_report_removes_tensors_and_paths() -> None:
    report = public_runtime_report(
        {
            "adapter_states_private": [{"secret": torch.ones(1)}],
            "nested": {
                "activation": torch.ones(1),
                "checkpoint_path": "/private/checkpoint",
                "hash": "sha256:ok",
            },
        }
    )
    encoded = json.dumps(report, sort_keys=True)
    assert "secret" not in encoded
    assert "/private" not in encoded
    assert "activation" not in report["nested"]
    assert report["activation_values_public"] is False


def test_real_stage_process_pipeline_and_checkpoint_restart_on_tiny_qwen(tmp_path) -> None:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(311)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        attention_dropout=0.0,
        use_cache=False,
    )
    model = Qwen2ForCausalLM(config)
    model.tie_weights()
    source = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if name != "lm_head.weight"
    }
    source_path = tmp_path / "tiny-model.safetensors"
    save_file(source, source_path)
    header_length, header = read_safetensors_header(source_path)
    raw = source_path.read_bytes()
    specs = [
        QwenStageSpec(0, "A", 0, 0, 1, owns_embedding=True),
        QwenStageSpec(1, "A", 1, 1, 2),
        QwenStageSpec(2, "B", 0, 2, 3),
        QwenStageSpec(3, "B", 1, 3, 4, owns_norm=True, owns_lm_head=True),
    ]
    shards = {}
    for spec in specs:
        path = tmp_path / f"stage{spec.stage_id}.safetensors"
        materialize_stage_shard(
            spec=spec,
            header_length=header_length,
            header=header,
            output_path=path,
            range_reader=lambda _url, start, end: raw[start : end + 1],
        )
        shards[spec.stage_id] = path

    def start(run_kind: str, stage: int, *, resume: bool = False, root=None):
        checkpoint_root = root or (tmp_path / run_kind / "checkpoints")
        return StageProcessClient(
            config=config.to_dict(),
            spec=specs[stage],
            shard_path=shards[stage],
            checkpoint_dir=checkpoint_root,
            resume=resume,
            seed=404,
            learning_rate=0.001,
            lora_rank=2,
            lora_alpha=4,
            ready_timeout=60,
            device_override="cpu",
        )

    rows = [
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12],
        [13, 14, 15, 16, 17, 18],
        [19, 20, 21, 22, 23, 24],
    ]

    def execute(run_kind: str, *, restart: bool):
        transport = _FakeTransport()
        clients = [start(run_kind, stage) for stage in range(4)]
        results = {}

        def run_a():
            results["a"] = run_kernel_a_once(
                run_kind=run_kind,
                clients=clients[:2],
                transport=transport,
                train_rows=rows,
                steps=2,
                microbatch_count=2,
                wait_timeout=30,
            )

        def run_b():
            checkpoint_root = tmp_path / run_kind / "checkpoints"
            b_clients = clients[2:]
            results["b"] = run_kernel_b_once(
                run_kind=run_kind,
                clients=b_clients,
                transport=transport,
                steps=2,
                microbatch_count=2,
                wait_timeout=30,
                restart_stage2_after_step=1 if restart else 0,
                restart_stage2_factory=(
                    lambda: start(
                        run_kind,
                        2,
                        resume=True,
                        root=checkpoint_root,
                    )
                )
                if restart
                else None,
            )
            clients[2:] = b_clients

        threads = [threading.Thread(target=run_a), threading.Thread(target=run_b)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
            assert not thread.is_alive()
        for client in clients:
            if client.process.is_alive():
                client.stop(timeout=30)
        return results

    baseline = execute("baseline", restart=False)
    resumed = execute("resumed", restart=True)
    assert resumed["b"]["controlled_restart_verified"] is True
    assert compare_adapter_states(
        baseline["a"]["adapter_states_private"],
        resumed["a"]["adapter_states_private"],
    )["verified"] is True
    assert compare_adapter_states(
        baseline["b"]["adapter_states_private"],
        resumed["b"]["adapter_states_private"],
    )["verified"] is True
    assert compare_losses(baseline["b"]["losses"], resumed["b"]["losses"])[
        "verified"
    ] is True
    assert all(
        float(stage["lora_gradient_norm"]) > 0
        for result in baseline.values()
        for step in result["step_reports"]
        for stage in step["stages"]
    )
