"""Kaggle worker lifecycle for the Qwen 1.5B four-GPU Training Alpha."""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from .elastic_training_client import ElasticTrainingHTTPClient
from .qwen15b_four_gpu_runtime import (
    DEFAULT_MICROBATCHES,
    DEFAULT_STEPS,
    QwenHTTPTransport,
    StageProcessClient,
    public_runtime_report,
    run_kernel_a_once,
    run_kernel_b_once,
    _stage_error_code,
)
from .qwen15b_training import (
    MODEL_ID,
    MODEL_PARAMETER_COUNT,
    MODEL_REVISION,
    QwenStageSpec,
    canonical_stage_specs,
    export_qwen_standard_peft_adapter,
    fetch_safetensors_header,
    materialize_stage_shard,
    materialize_stage_shard_from_layout,
    sha256_file,
    stable_hash,
)


WORKER_SCHEMA = "crowdtensor_qwen15b_four_gpu_worker_v1"


def _role_specs(role: str, *, layer_count: int = 28) -> list[QwenStageSpec]:
    specs = canonical_stage_specs(layer_count)
    if role == "kernel_a":
        return specs[:2]
    if role == "kernel_b":
        return specs[2:]
    raise ValueError("Qwen four-GPU worker role must be kernel_a or kernel_b")


def prepare_role_stage_shards(
    *,
    role: str,
    output_dir: str | Path,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
    source_layout: dict[str, Any] | None = None,
    layer_count: int = 28,
) -> tuple[dict[int, Path], list[dict[str, Any]]]:
    """Materialize only the two stages owned by this Kaggle Kernel."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    header_length = 0
    header: dict[str, Any] = {}
    if source_layout is None:
        header_length, header = fetch_safetensors_header(
            model_id=model_id,
            revision=model_revision,
        )
    specs = _role_specs(role, layer_count=layer_count)

    def materialize(spec: QwenStageSpec) -> tuple[int, Path, dict[str, Any]]:
        shard = output / f"stage{spec.stage_id}.safetensors"
        if source_layout is not None:
            report = materialize_stage_shard_from_layout(
                spec=spec,
                source_layout=source_layout,
                output_path=shard,
            )
        else:
            report = materialize_stage_shard(
                spec=spec,
                header_length=header_length,
                header=header,
                output_path=shard,
                model_id=model_id,
                revision=model_revision,
            )
        return spec.stage_id, shard, report

    results = []
    try:
        # Each shard materializer retains its complete tensor dictionary until
        # safetensors is written. Keep this startup phase serial so two >600 MiB
        # dictionaries and two HTTP range buffers never coexist in one Kernel.
        for spec in specs:
            results.append(materialize(spec))
    except BaseException as exc:
        raise RuntimeError(
            f"qwen15b_stage_shard_prepare_failed:{_stage_error_code(exc)}"
        ) from exc
    results.sort(key=lambda item: item[0])
    shards = {stage_id: path for stage_id, path, _report in results}
    reports = [
        {key: value for key, value in report.items() if key != "shard_path"}
        for _stage_id, _path, report in results
    ]
    return shards, reports


def compare_adapter_states(
    baseline: list[dict[str, Any]],
    resumed: list[dict[str, Any]],
    *,
    atol: float = 5e-3,
    rtol: float = 5e-3,
) -> dict[str, Any]:
    import torch

    if len(baseline) != len(resumed):
        return {
            "verified": False,
            "reason": "stage_count_mismatch",
            "atol": float(atol),
            "rtol": float(rtol),
        }
    maximum = 0.0
    compared = 0
    exact = True
    for first, second in zip(baseline, resumed, strict=True):
        if set(first) != set(second):
            return {
                "verified": False,
                "reason": "adapter_tensor_names_mismatch",
                "atol": float(atol),
                "rtol": float(rtol),
            }
        for name in sorted(first):
            left = first[name].detach().float().cpu()
            right = second[name].detach().float().cpu()
            difference = float((left - right).abs().max().item()) if left.numel() else 0.0
            maximum = max(maximum, difference)
            compared += 1
            exact = bool(exact and torch.equal(left, right))
            if not torch.allclose(left, right, atol=float(atol), rtol=float(rtol)):
                return {
                    "verified": False,
                    "reason": "adapter_tolerance_exceeded",
                    "maximum_absolute_difference": maximum,
                    "compared_tensor_count": compared,
                    "atol": float(atol),
                    "rtol": float(rtol),
                }
    return {
        "verified": compared > 0,
        "exact_match": exact,
        "maximum_absolute_difference": maximum,
        "compared_tensor_count": compared,
        "atol": float(atol),
        "rtol": float(rtol),
    }


def compare_losses(
    baseline: list[float],
    resumed: list[float],
    *,
    atol: float = 5e-3,
    rtol: float = 5e-3,
) -> dict[str, Any]:
    import torch

    left = torch.tensor(baseline, dtype=torch.float64)
    right = torch.tensor(resumed, dtype=torch.float64)
    same_shape = left.shape == right.shape and left.numel() > 0
    maximum = float((left - right).abs().max().item()) if same_shape else None
    return {
        "verified": bool(same_shape and torch.allclose(left, right, atol=atol, rtol=rtol)),
        "loss_count": int(left.numel()) if same_shape else 0,
        "maximum_absolute_difference": maximum,
        "atol": float(atol),
        "rtol": float(rtol),
    }


def _start_pair(
    *,
    role: str,
    config: dict[str, Any],
    shards: dict[int, Path],
    run_root: Path,
    resume_stage_ids: set[int] | None = None,
    seed: int,
    learning_rate: float,
    lora_rank: int,
    lora_alpha: int,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
    layer_count: int = 28,
) -> list[StageProcessClient]:
    resume_ids = set(resume_stage_ids or set())
    clients = []
    for spec in _role_specs(role, layer_count=layer_count):
        clients.append(
            StageProcessClient(
                config=config,
                spec=spec,
                shard_path=shards[spec.stage_id],
                checkpoint_dir=run_root / "checkpoints",
                resume=spec.stage_id in resume_ids,
                seed=seed,
                learning_rate=learning_rate,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                model_id=model_id,
                model_revision=model_revision,
            )
        )
    return clients


def _register_pair(
    transport: QwenHTTPTransport,
    *,
    role: str,
    run_kind: str,
    clients: list[StageProcessClient],
) -> None:
    transport.register(role=role, ready=[client.ready for client in clients])
    for client in clients:
        transport.event(
            role=role,
            run_kind=run_kind,
            operation="stage_loaded",
            stage_id=client.spec.stage_id,
            pid=client.pid,
            device=client.spec.device,
        )


def _stop_pair(clients: list[StageProcessClient]) -> list[dict[str, Any]]:
    reports = []
    for client in clients:
        try:
            reports.append(client.stop())
        except BaseException:
            reports.append(
                {
                    "stage_id": int(client.spec.stage_id),
                    "stopped": client.force_stop(),
                }
            )
    return reports


def _wait_stage_adapter(
    transport: QwenHTTPTransport,
    *,
    timeout: float,
    run_kind: str = "resumed",
    step: int = DEFAULT_STEPS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        value = transport.get_tensors(
            run_kind=run_kind,
            kind="stage_adapter",
            step=int(step),
            microbatch=-1,
        )
        if value is not None:
            return value
        time.sleep(1.0)
    raise TimeoutError("Qwen Kernel B timed out waiting for Kernel A adapter stages")


def evaluate_standard_adapter(
    *,
    adapter_dir: str | Path,
    validation_rows: list[list[int]],
    cache_dir: str | Path,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
) -> dict[str, Any]:
    """Reload standard PEFT on CPU and CUDA, then run held-out validation."""

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen adapter evaluation requires live CUDA")
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    adapter = Path(adapter_dir)
    cpu_loaded = False
    cuda_loaded = False
    base = None
    model = None
    try:
        cpu_base = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=model_revision,
            cache_dir=cache,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        cpu_model = PeftModel.from_pretrained(
            cpu_base,
            adapter,
            local_files_only=True,
            is_trainable=False,
        )
        cpu_loaded = bool(
            next(cpu_model.parameters()).device.type == "cpu"
            and any("lora_" in name for name, _value in cpu_model.named_parameters())
        )
        del cpu_model, cpu_base
        gc.collect()

        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=model_revision,
            cache_dir=cache,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        ).to("cuda:0")
        base.eval()
        rows = [torch.tensor([row], dtype=torch.long, device="cuda:0") for row in validation_rows]

        def run_loss(current: Any) -> tuple[float, str]:
            losses = []
            digest = hashlib.sha256()
            with torch.no_grad():
                for index, tokens in enumerate(rows):
                    result = current(input_ids=tokens, labels=tokens, use_cache=False)
                    losses.append(float(result.loss.detach().float().item()))
                    if index == 0:
                        probe = result.logits[0, -1, :256].detach().cpu().contiguous()
                        digest.update(probe.view(torch.uint8).numpy().tobytes())
            return sum(losses) / len(losses), "sha256:" + digest.hexdigest()

        before_loss, before_logits_hash = run_loss(base)
        model = PeftModel.from_pretrained(
            base,
            adapter,
            local_files_only=True,
            is_trainable=False,
        )
        model.eval()
        cuda_loaded = bool(
            next(model.parameters()).device.type == "cuda"
            and any(
                "lora_" in name and parameter.device.type == "cuda"
                for name, parameter in model.named_parameters()
            )
        )
        after_loss, after_logits_hash = run_loss(model)
        report = {
            "schema": "crowdtensor_qwen15b_peft_evaluation_v1",
            "model_id": str(model_id),
            "model_revision": str(model_revision),
            "validation_sequence_count": len(rows),
            "standard_peft_cpu_load": cpu_loaded,
            "standard_peft_cuda_load": cuda_loaded,
            "before_validation_loss": before_loss,
            "after_validation_loss": after_loss,
            "validation_loss_reduced": after_loss < before_loss,
            "before_logits_hash": before_logits_hash,
            "after_logits_hash": after_logits_hash,
            "adapter_changes_logits": before_logits_hash != after_logits_hash,
            "cuda_device": "cuda:0",
            "cuda_compute_dtype": "float32",
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "logits_values_public": False,
            "token_ids_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        report["evaluation_verified"] = bool(
            report["standard_peft_cpu_load"]
            and report["standard_peft_cuda_load"]
            and report["validation_loss_reduced"]
            and report["adapter_changes_logits"]
        )
        report["content_hash"] = stable_hash(report)
        return report
    finally:
        del model, base
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        shutil.rmtree(cache, ignore_errors=True)


def run_kernel_role(
    *,
    role: str,
    coordinator_url: str,
    coordinator_token: str,
    run_id: str,
    config: dict[str, Any],
    tokenized_payload_path: str | Path,
    private_root: str | Path,
    export_dir: str | Path | None = None,
    steps: int = DEFAULT_STEPS,
    microbatch_count: int = DEFAULT_MICROBATCHES,
    seed: int = 20260712,
    learning_rate: float = 5e-4,
    lora_rank: int = 4,
    lora_alpha: int = 8,
    wait_timeout: float = 900.0,
    coordinator_restart_after_step: int = 0,
) -> dict[str, Any]:
    import torch

    if role not in {"kernel_a", "kernel_b"}:
        raise ValueError("Qwen worker role invalid")
    if int(steps) != DEFAULT_STEPS or int(microbatch_count) != DEFAULT_MICROBATCHES:
        raise ValueError("Qwen Alpha requires exactly 8 steps and four microbatches")
    if str(config.get("model_type") or "") != "qwen2" or int(
        config.get("num_hidden_layers") or 0
    ) != 28:
        raise ValueError("Qwen Alpha config does not resolve the pinned 28-layer model")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Qwen four-GPU worker requires one live T4x2 Kernel")
    root = Path(private_root)
    root.mkdir(parents=True, exist_ok=True)
    payload = json.loads(Path(tokenized_payload_path).read_text(encoding="utf-8"))
    train_rows = list(payload.get("train") or [])
    validation_rows = list(payload.get("validation") or [])
    if len(train_rows) < 32 or len(validation_rows) < 4:
        raise RuntimeError("Qwen private tokenized dataset is incomplete")
    transport = QwenHTTPTransport(
        coordinator_url=coordinator_url,
        token=coordinator_token,
        run_id=run_id,
    )
    shards, shard_reports = prepare_role_stage_shards(
        role=role,
        output_dir=root / "stage-shards",
    )
    run_results: dict[str, dict[str, Any]] = {}
    all_ready: dict[str, list[dict[str, Any]]] = {}
    all_stop: dict[str, list[dict[str, Any]]] = {}
    clients: list[StageProcessClient] = []
    started = time.time()
    try:
        for run_kind in ("baseline", "resumed"):
            run_root = root / run_kind
            clients = _start_pair(
                role=role,
                config=config,
                shards=shards,
                run_root=run_root,
                seed=seed,
                learning_rate=learning_rate,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
            )
            all_ready[run_kind] = [dict(client.ready) for client in clients]
            _register_pair(
                transport,
                role=role,
                run_kind=run_kind,
                clients=clients,
            )
            if run_kind == "baseline":
                transport.wait_roles(timeout=wait_timeout)
            if role == "kernel_a":
                def restart_pair_a() -> list[StageProcessClient]:
                    return _start_pair(
                        role=role,
                        config=config,
                        shards=shards,
                        run_root=run_root,
                        resume_stage_ids={0, 1},
                        seed=seed,
                        learning_rate=learning_rate,
                        lora_rank=lora_rank,
                        lora_alpha=lora_alpha,
                    )

                result = run_kernel_a_once(
                    run_kind=run_kind,
                    clients=clients,
                    transport=transport,
                    train_rows=train_rows[: steps * microbatch_count],
                    steps=steps,
                    microbatch_count=microbatch_count,
                    wait_timeout=wait_timeout,
                    restart_pair_after_step=(
                        int(coordinator_restart_after_step) if run_kind == "resumed" else 0
                    ),
                    restart_pair_factory=(
                        restart_pair_a
                        if run_kind == "resumed" and coordinator_restart_after_step
                        else None
                    ),
                )
            else:

                def restart_stage2() -> StageProcessClient:
                    spec = _role_specs(role)[0]
                    return StageProcessClient(
                        config=config,
                        spec=spec,
                        shard_path=shards[spec.stage_id],
                        checkpoint_dir=run_root / "checkpoints",
                        resume=True,
                        seed=seed,
                        learning_rate=learning_rate,
                        lora_rank=lora_rank,
                        lora_alpha=lora_alpha,
                    )

                def restart_pair_b() -> list[StageProcessClient]:
                    return _start_pair(
                        role=role,
                        config=config,
                        shards=shards,
                        run_root=run_root,
                        resume_stage_ids={2, 3},
                        seed=seed,
                        learning_rate=learning_rate,
                        lora_rank=lora_rank,
                        lora_alpha=lora_alpha,
                    )

                result = run_kernel_b_once(
                    run_kind=run_kind,
                    clients=clients,
                    transport=transport,
                    steps=steps,
                    microbatch_count=microbatch_count,
                    wait_timeout=wait_timeout,
                    restart_stage2_after_step=(
                        4
                        if run_kind == "resumed" and not coordinator_restart_after_step
                        else 0
                    ),
                    restart_stage2_factory=(
                        restart_stage2
                        if run_kind == "resumed" and not coordinator_restart_after_step
                        else None
                    ),
                    restart_pair_after_step=(
                        int(coordinator_restart_after_step) if run_kind == "resumed" else 0
                    ),
                    restart_pair_factory=(
                        restart_pair_b
                        if run_kind == "resumed" and coordinator_restart_after_step
                        else None
                    ),
                )
            run_results[run_kind] = result
            all_stop[run_kind] = _stop_pair(clients)
            clients = []

        comparison = compare_adapter_states(
            run_results["baseline"]["adapter_states_private"],
            run_results["resumed"]["adapter_states_private"],
        )
        loss_comparison: dict[str, Any] = {}
        evaluation: dict[str, Any] = {}
        export: dict[str, Any] = {}
        adapter_transfer: dict[str, Any] = {}
        if role == "kernel_a":
            combined = {}
            for state in run_results["resumed"]["adapter_states_private"]:
                combined.update(state)
            adapter_transfer = transport.put_tensors(
                role="kernel_a",
                run_kind="resumed",
                kind="stage_adapter",
                step=steps,
                microbatch=-1,
                tensors=combined,
            )
            adapter_transfer = {
                "payload_hash": adapter_transfer["payload_hash"],
                "stage_ids": [0, 1],
                "adapter_tensor_count": len(combined),
                "tensor_values_public": False,
            }
        else:
            loss_comparison = compare_losses(
                run_results["baseline"]["losses"],
                run_results["resumed"]["losses"],
            )
            remote, metadata = _wait_stage_adapter(transport, timeout=wait_timeout)
            stage01 = remote
            stage23 = run_results["resumed"]["adapter_states_private"]
            target_export = Path(export_dir or (root / "standard-peft-adapter"))
            export = export_qwen_standard_peft_adapter(
                [stage01, *stage23],
                target_export,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
            )
            adapter_transfer = {
                **metadata,
                "stage_ids": [0, 1],
                "adapter_tensor_count": len(stage01),
                "tensor_values_public": False,
            }
            evaluation = evaluate_standard_adapter(
                adapter_dir=export["adapter_dir"],
                validation_rows=validation_rows,
                cache_dir=root / "hf-evaluation-cache",
            )
            transport.event(
                role=role,
                run_kind="resumed",
                operation="evaluation",
                stage_id=3,
                step=steps,
                pid=int(all_ready["resumed"][1]["pid"]),
                device="cuda:0",
                adapter_hash=str(export.get("adapter_file_hash") or ""),
            )
            transport.event(
                role=role,
                run_kind="resumed",
                operation="export",
                stage_id=3,
                step=steps,
                pid=int(all_ready["resumed"][1]["pid"]),
                device="cuda:0",
                adapter_hash=str(export.get("adapter_file_hash") or ""),
            )

        base_frozen = all(
            status.get("base_hash_before") == status.get("base_hash_after")
            for run in run_results.values()
            for status in run.get("stage_statuses") or []
        )
        positive_gradients = all(
            float(stage.get("lora_gradient_norm") or 0.0) > 0
            for run in run_results.values()
            for step_report in run.get("step_reports") or []
            for stage in step_report.get("stages") or []
        )
        resumed_restart = run_results["resumed"].get("controlled_restart_verified")
        coordinator_restart_recoveries = list(
            run_results["resumed"].get("coordinator_restart_stage_recoveries") or []
        )
        expected_owned = {0, 1} if role == "kernel_a" else {2, 3}
        coordinator_restart_owned_stages_verified = bool(
            not coordinator_restart_after_step
            or {
                int(item.get("stage_id", -1)) for item in coordinator_restart_recoveries
            }
            == expected_owned
        )
        role_ok = bool(
            comparison.get("verified")
            and base_frozen
            and positive_gradients
            and all(run.get("steps_completed") == steps for run in run_results.values())
            and coordinator_restart_owned_stages_verified
            and (
                role == "kernel_a"
                or (
                    resumed_restart is True
                    and loss_comparison.get("verified") is True
                    and run_results["baseline"].get("loss_reduced") is True
                    and run_results["resumed"].get("loss_reduced") is True
                    and evaluation.get("evaluation_verified") is True
                    and export.get("standard_peft_format") is True
                )
            )
        )
        report = {
            "schema": WORKER_SCHEMA,
            "ok": role_ok,
            "role": role,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "parameter_count": MODEL_PARAMETER_COUNT,
            "steps": int(steps),
            "microbatches_per_step": int(microbatch_count),
            "stage_shards": shard_reports,
            "stage_ready": all_ready,
            "runs": {
                key: public_runtime_report(value) for key, value in run_results.items()
            },
            "resume_adapter_equivalence": comparison,
            "resume_loss_equivalence": loss_comparison,
            "adapter_transfer": adapter_transfer,
            "export": {key: value for key, value in export.items() if key != "adapter_dir"},
            "evaluation": evaluation,
            "base_weights_frozen": base_frozen,
            "positive_lora_gradient_norms": positive_gradients,
            "controlled_restart_verified": bool(resumed_restart) if role == "kernel_b" else None,
            "coordinator_restart_after_step": int(coordinator_restart_after_step),
            "coordinator_restart_stage_recoveries": coordinator_restart_recoveries,
            "coordinator_restart_owned_stages_verified": coordinator_restart_owned_stages_verified,
            "transport_reliability": transport.public_retry_report(),
            "process_shutdown": all_stop,
            "elapsed_seconds": time.time() - started,
            "activation_values_public": False,
            "gradient_values_public": False,
            "adapter_tensor_values_public": False,
            "token_ids_public": False,
            "raw_training_text_public": False,
            "credentials_public": False,
            "coordinator_url_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        report["content_hash"] = stable_hash(report)
        transport.complete(
            role=role,
            summary={
                "ok": report["ok"],
                "baseline_steps_completed": int(
                    report["runs"]["baseline"]["steps_completed"]
                ),
                "resumed_steps_completed": int(
                    report["runs"]["resumed"]["steps_completed"]
                ),
                "stage_ids": [spec.stage_id for spec in _role_specs(role)],
                "final_adapter_hashes": report["runs"]["resumed"]["adapter_hashes"],
                "controlled_restart_verified": report.get("controlled_restart_verified") is True,
                "evaluation_verified": evaluation.get("evaluation_verified") is True,
                "export_verified": export.get("standard_peft_format") is True,
                "coordinator_restart_owned_stages_verified": (
                    coordinator_restart_owned_stages_verified
                ),
                "transport_retry_count": int(
                    report["transport_reliability"].get("retry_count") or 0
                ),
                "transport_reconnect_registration_count": int(
                    report["transport_reliability"].get(
                        "reconnect_registration_count"
                    )
                    or 0
                ),
            },
        )
        return report
    finally:
        for client in clients:
            try:
                client.force_stop()
            except BaseException:
                pass
        for shard in shards.values():
            shard.unlink(missing_ok=True)


def run_elastic_kernel_role(
    *,
    role: str,
    coordinator_url: str,
    coordinator_token: str,
    run_id: str,
    miner_id_hash: str,
    registration_nonce: str,
    expected_start_step: int,
    segment_end_step: int,
    config: dict[str, Any],
    tokenized_payload_path: str | Path,
    private_root: str | Path,
    export_dir: str | Path | None = None,
    target_steps: int = DEFAULT_STEPS,
    microbatch_count: int = DEFAULT_MICROBATCHES,
    seed: int = 20260712,
    learning_rate: float = 5e-4,
    lora_rank: int = 4,
    lora_alpha: int = 8,
    wait_timeout: float = 900.0,
    heartbeat_interval_seconds: float = 5.0,
    drain_requested: Callable[[], bool] | None = None,
    max_steps_per_session: int = 0,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
    parameter_count: int = MODEL_PARAMETER_COUNT,
    source_layout_path: str | Path | None = None,
    defer_evaluation: bool = False,
) -> dict[str, Any]:
    """Run one bounded elastic segment under central checkpoint barriers."""

    import torch

    requested_role = str(role)
    if requested_role not in {"auto", "kernel_a", "kernel_b"}:
        raise ValueError("Qwen elastic worker role invalid")
    if (
        int(target_steps) < 1
        or int(target_steps) > 2048
        or int(microbatch_count) < 1
        or int(microbatch_count) > 16
        or int(expected_start_step) < 0
        or int(segment_end_step) <= int(expected_start_step)
        or int(segment_end_step) > int(target_steps)
    ):
        raise ValueError("Qwen elastic segment contract invalid")
    layer_count = int(config.get("num_hidden_layers") or 0)
    if str(config.get("model_type") or "") != "qwen2" or layer_count != 28:
        raise ValueError("Qwen elastic config does not resolve the pinned model")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Qwen elastic worker requires one live T4x2 Kernel")
    root = Path(private_root)
    root.mkdir(parents=True, exist_ok=True)
    run_root = root / "elastic-segment"
    checkpoint_dir = run_root / "checkpoints"
    checkpoint_dir_preexisted = checkpoint_dir.exists()
    payload = json.loads(Path(tokenized_payload_path).read_text(encoding="utf-8"))
    if (
        str(payload.get("model_id") or "") != str(model_id)
        or str(payload.get("model_revision") or "") != str(model_revision)
    ):
        raise RuntimeError("Qwen elastic private dataset model identity mismatch")
    train_rows = list(payload.get("train") or [])
    validation_rows = list(payload.get("validation") or [])
    if len(train_rows) < int(target_steps) * int(microbatch_count) or len(
        validation_rows
    ) < 4:
        raise RuntimeError("Qwen elastic private tokenized dataset is incomplete")
    expected_stage_ids = (
        [0, 1, 2, 3]
        if requested_role == "auto"
        else [
            int(spec.stage_id)
            for spec in _role_specs(requested_role, layer_count=layer_count)
        ]
    )
    elastic = ElasticTrainingHTTPClient(
        coordinator_url=coordinator_url,
        coordinator_token=coordinator_token,
        run_id=run_id,
        miner_id_hash=miner_id_hash,
        registration_nonce=registration_nonce,
        supported_stage_ids=expected_stage_ids,
        slot_count=2,
        accelerator="cuda",
        timeout=min(120.0, float(wait_timeout)),
        heartbeat_interval_seconds=float(heartbeat_interval_seconds),
    )
    transport = QwenHTTPTransport(
        coordinator_url=coordinator_url,
        token=coordinator_token,
        run_id=run_id,
    )
    clients: list[StageProcessClient] = []
    shards: dict[int, Path] = {}
    stop_reports: list[dict[str, Any]] = []
    offline_report: dict[str, Any] = {}
    barrier_records: list[dict[str, Any]] = []
    restore_reports: list[dict[str, Any]] = []
    started = time.time()
    try:
        registration = elastic.register()
        elastic.start_heartbeat()
        assignment_response = elastic.wait_for_assignments(
            expected_stage_ids=(
                None if requested_role == "auto" else expected_stage_ids
            ),
            expected_base_step=int(expected_start_step),
            timeout=wait_timeout,
            expected_assignment_count=2,
            allowed_stage_groups=[[0, 1], [2, 3]],
        )
        assignments = {
            int(item["stage_id"]): dict(item)
            for item in assignment_response.get("assignments") or []
        }
        assigned_stage_ids = sorted(assignments)
        if requested_role == "auto":
            if assigned_stage_ids == [0, 1]:
                role = "kernel_a"
            elif assigned_stage_ids == [2, 3]:
                role = "kernel_b"
            else:
                raise RuntimeError("elastic_qwen_auto_role_assignment_invalid")
            expected_stage_ids = assigned_stage_ids
        else:
            role = requested_role
        specs = _role_specs(role, layer_count=layer_count)
        if set(assignments) != set(expected_stage_ids):
            raise RuntimeError("elastic_qwen_stage_assignment_incomplete")
        base_steps = {int(item["base_step"]) for item in assignments.values()}
        if base_steps != {int(expected_start_step)}:
            raise RuntimeError("elastic_qwen_resume_step_mismatch")
        if int(expected_start_step) > 0:
            if checkpoint_dir_preexisted:
                raise RuntimeError("elastic_qwen_checkpoint_directory_not_fresh")
            for stage_id in expected_stage_ids:
                restore_reports.append(
                    elastic.download_checkpoint(
                        assignments[stage_id], checkpoint_dir=checkpoint_dir
                    )
                )
        source_layout = None
        if source_layout_path:
            source_layout = json.loads(
                Path(source_layout_path).read_text(encoding="utf-8")
            )
            if (
                str(source_layout.get("model_id") or "") != str(model_id)
                or str(source_layout.get("model_revision") or "")
                != str(model_revision)
            ):
                raise RuntimeError("Qwen elastic source layout model mismatch")
        shards, shard_reports = prepare_role_stage_shards(
            role=role,
            output_dir=root / "stage-shards",
            model_id=model_id,
            model_revision=model_revision,
            source_layout=source_layout,
            layer_count=layer_count,
        )
        clients = _start_pair(
            role=role,
            config=config,
            shards=shards,
            run_root=run_root,
            resume_stage_ids=(
                set(expected_stage_ids) if int(expected_start_step) > 0 else set()
            ),
            seed=seed,
            learning_rate=learning_rate,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            model_id=model_id,
            model_revision=model_revision,
            layer_count=layer_count,
        )
        ready = [dict(client.ready) for client in clients]
        if int(expected_start_step) > 0 and any(
            item.get("resumed") is not True
            or int(item.get("resumed_global_step") or 0) != int(expected_start_step)
            or int(item.get("resumed_dataset_cursor") or 0)
            != int(expected_start_step) * int(microbatch_count)
            or not str(item.get("loaded_checkpoint_hash") or "").startswith("sha256:")
            for item in ready
        ):
            raise RuntimeError("elastic_qwen_stage_restore_contract_invalid")
        _register_pair(
            transport,
            role=role,
            run_kind="elastic",
            clients=clients,
        )
        transport.wait_roles(timeout=wait_timeout)
        graceful_drain_applied = False

        def commit_step(
            global_step: int,
            dataset_cursor: int,
            finishes: list[dict[str, Any]],
        ) -> dict[str, Any]:
            nonlocal assignments, graceful_drain_applied

            if (
                {int(item["stage_id"]) for item in finishes}
                != set(expected_stage_ids)
                or {int(item["global_step"]) for item in finishes} != {global_step}
                or {int(item["dataset_cursor"]) for item in finishes}
                != {dataset_cursor}
                or {int(item["target_step"]) for item in assignments.values()}
                != {global_step}
            ):
                raise RuntimeError("elastic_qwen_local_checkpoint_barrier_mismatch")
            submissions = []
            archive_reports = []
            epoch_ids = {int(item["epoch_id"]) for item in assignments.values()}
            if len(epoch_ids) != 1:
                raise RuntimeError("elastic_qwen_epoch_assignment_divergence")
            epoch_id = next(iter(epoch_ids))
            for stage_id in expected_stage_ids:
                submission, archive_report = elastic.submit_checkpoint(
                    assignments[stage_id], checkpoint_dir=checkpoint_dir
                )
                submissions.append(
                    {
                        "stage_id": stage_id,
                        "archive_hash": str(archive_report["archive_hash"]),
                        "checkpoint_content_hash": str(
                            archive_report["checkpoint_content_hash"]
                        ),
                        "idempotent": submission.get("idempotent") is True,
                        "global_commit_created": submission.get(
                            "global_commit_created"
                        )
                        is True,
                    }
                )
                archive_reports.append(archive_report)
            barrier = elastic.wait_barrier(epoch_id=epoch_id, timeout=wait_timeout)
            if (
                barrier.get("state") != "committed"
                or int(barrier.get("committed_step") or 0) != global_step
                or int(barrier.get("submitted_stage_count") or 0) != 4
            ):
                raise RuntimeError("elastic_qwen_global_barrier_not_committed")
            record = {
                "epoch_id": epoch_id,
                "global_step": global_step,
                "dataset_cursor": dataset_cursor,
                "owned_stage_ids": list(expected_stage_ids),
                "owned_stage_checkpoint_submissions": submissions,
                "owned_stage_checkpoint_count": len(archive_reports),
                "global_stage_checkpoint_count": int(
                    barrier["submitted_stage_count"]
                ),
                "barrier_committed": True,
                "exactly_once_commit": True,
                "assignment_tokens_public": False,
                "checkpoint_values_public": False,
                "public_artifact_safe": True,
            }
            should_drain = bool(
                (drain_requested is not None and drain_requested())
                or (
                    int(max_steps_per_session) > 0
                    and global_step - int(expected_start_step)
                    >= int(max_steps_per_session)
                )
            )
            graceful_drain_applied = bool(
                should_drain and global_step < int(segment_end_step)
            )
            record["continue_training"] = bool(
                global_step < int(segment_end_step) and not should_drain
            )
            record["graceful_drain_after_barrier"] = graceful_drain_applied
            barrier_records.append(record)
            if record["continue_training"]:
                next_response = elastic.wait_for_assignments(
                    expected_stage_ids=expected_stage_ids,
                    expected_base_step=global_step,
                    timeout=wait_timeout,
                )
                assignments = {
                    int(item["stage_id"]): dict(item)
                    for item in next_response.get("assignments") or []
                }
            return record

        segment_steps = int(segment_end_step) - int(expected_start_step)
        if role == "kernel_a":
            result = run_kernel_a_once(
                run_kind="elastic",
                clients=clients,
                transport=transport,
                train_rows=train_rows,
                steps=segment_steps,
                start_step=int(expected_start_step),
                microbatch_count=int(microbatch_count),
                wait_timeout=wait_timeout,
                step_commit_callback=commit_step,
            )
        else:
            result = run_kernel_b_once(
                run_kind="elastic",
                clients=clients,
                transport=transport,
                steps=segment_steps,
                start_step=int(expected_start_step),
                microbatch_count=int(microbatch_count),
                wait_timeout=wait_timeout,
                step_commit_callback=commit_step,
            )

        export: dict[str, Any] = {}
        evaluation: dict[str, Any] = {}
        adapter_transfer: dict[str, Any] = {}
        actual_end_step = int(result.get("end_step") or expected_start_step)
        if actual_end_step == int(target_steps):
            if role == "kernel_a":
                combined: dict[str, Any] = {}
                for state in result["adapter_states_private"]:
                    combined.update(state)
                transfer = transport.put_tensors(
                    role=role,
                    run_kind="elastic",
                    kind="stage_adapter",
                    step=int(target_steps),
                    microbatch=-1,
                    tensors=combined,
                )
                adapter_transfer = {
                    "payload_hash": transfer["payload_hash"],
                    "stage_ids": [0, 1],
                    "adapter_tensor_count": len(combined),
                    "tensor_values_public": False,
                }
            else:
                remote, metadata = _wait_stage_adapter(
                    transport,
                    timeout=wait_timeout,
                    run_kind="elastic",
                    step=int(target_steps),
                )
                target_export = Path(export_dir or (root / "standard-peft-adapter"))
                export = export_qwen_standard_peft_adapter(
                    [remote, *result["adapter_states_private"]],
                    target_export,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    model_id=model_id,
                    model_revision=model_revision,
                )
                adapter_transfer = {
                    **metadata,
                    "stage_ids": [0, 1],
                    "adapter_tensor_count": len(remote),
                    "tensor_values_public": False,
                }
                if defer_evaluation:
                    evaluation = {
                        "schema": "crowdtensor_qwen_deferred_evaluation_v1",
                        "evaluation_verified": False,
                        "evaluation_deferred_to_isolated_benchmark": True,
                        "model_id": str(model_id),
                        "model_revision": str(model_revision),
                        "public_artifact_safe": True,
                    }
                else:
                    evaluation = evaluate_standard_adapter(
                        adapter_dir=export["adapter_dir"],
                        validation_rows=validation_rows,
                        cache_dir=root / "hf-evaluation-cache",
                        model_id=model_id,
                        model_revision=model_revision,
                    )
        base_frozen = all(
            status.get("base_hash_before") == status.get("base_hash_after")
            for status in result.get("stage_statuses") or []
        )
        positive_gradients = all(
            float(stage.get("lora_gradient_norm") or 0.0) > 0
            for step_report in result.get("step_reports") or []
            for stage in step_report.get("stages") or []
        )
        restore_verified = bool(
            int(expected_start_step) == 0
            or (
                len(restore_reports) == 2
                and all(
                    int(item["global_step"]) == int(expected_start_step)
                    for item in restore_reports
                )
                and all(item.get("resumed") is True for item in ready)
            )
        )
        role_ok = bool(
            actual_end_step
            == (
                int(expected_start_step) + len(barrier_records)
            )
            and len(barrier_records) >= 1
            and (
                actual_end_step == int(segment_end_step)
                or graceful_drain_applied
            )
            and all(item["barrier_committed"] for item in barrier_records)
            and base_frozen
            and positive_gradients
            and restore_verified
            and (
                actual_end_step < int(target_steps)
                or role == "kernel_a"
                or (
                    export.get("standard_peft_format") is True
                    and (
                        evaluation.get("evaluation_verified") is True
                        or (
                            defer_evaluation
                            and evaluation.get(
                                "evaluation_deferred_to_isolated_benchmark"
                            )
                            is True
                        )
                    )
                )
            )
        )
        report = {
            "schema": "crowdtensor_qwen15b_elastic_worker_v1",
            "ok": role_ok,
            "role": role,
            "requested_role": requested_role,
            "model_id": str(model_id),
            "model_revision": str(model_revision),
            "parameter_count": int(parameter_count),
            "expected_start_step": int(expected_start_step),
            "segment_end_step": actual_end_step,
            "requested_segment_end_step": int(segment_end_step),
            "target_steps": int(target_steps),
            "stage_ids": expected_stage_ids,
            "stage_shards": shard_reports,
            "stage_ready": ready,
            "central_checkpoint_restore": restore_reports,
            "central_checkpoint_restore_verified": restore_verified,
            "fresh_checkpoint_directory_before_restore": not checkpoint_dir_preexisted,
            "old_kernel_local_checkpoint_dependency": False,
            "runtime": public_runtime_report(result),
            "barrier_commits": barrier_records,
            "all_segment_barriers_committed": len(barrier_records) == segment_steps,
            "all_completed_barriers_committed": all(
                item["barrier_committed"] for item in barrier_records
            ),
            "graceful_drain_applied": graceful_drain_applied,
            "base_weights_frozen": base_frozen,
            "positive_lora_gradient_norms": positive_gradients,
            "adapter_transfer": adapter_transfer,
            "export": {key: value for key, value in export.items() if key != "adapter_dir"},
            "evaluation": evaluation,
            "evaluation_deferred_to_isolated_benchmark": bool(defer_evaluation),
            "elastic_client": elastic.public_report(),
            "elapsed_seconds": time.time() - started,
            "activation_values_public": False,
            "gradient_values_public": False,
            "adapter_tensor_values_public": False,
            "checkpoint_tensor_values_public": False,
            "token_ids_public": False,
            "raw_training_text_public": False,
            "credentials_public": False,
            "coordinator_url_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        report["content_hash"] = stable_hash(report)
        return report
    finally:
        if clients:
            stop_reports = _stop_pair(clients)
            clients = []
        for shard in shards.values():
            shard.unlink(missing_ok=True)
        try:
            if elastic.public_report()["registered"]:
                offline_report = elastic.offline()
        except BaseException:
            elastic.stop_heartbeat()
        # These reports are intentionally retained only in the private runtime;
        # the returned worker report contains no session or fencing token.
        if stop_reports:
            (root / "elastic-process-stop-private.json").write_text(
                json.dumps(stop_reports, sort_keys=True), encoding="utf-8"
            )
        if offline_report:
            (root / "elastic-offline-private.json").write_text(
                json.dumps(offline_report, sort_keys=True), encoding="utf-8"
            )
