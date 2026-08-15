"""Tiny real FSDP2 trainer used only by the stable-session integration test."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.fsdp import FSDPModule, fully_shard

from crowdtensor.backends.stable_session import STABLE_TRAINER_RESULT_SCHEMA
from crowdtensor.core.contracts import CheckpointRef, WorkUnit, stable_hash


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crowdtensor-project", required=True)
    parser.add_argument("--crowdtensor-checkpoint-dir", required=True)
    parser.add_argument("--crowdtensor-work-unit", required=True)
    parser.add_argument("--crowdtensor-base-checkpoint", required=True)
    parser.add_argument("--crowdtensor-base-payload", required=True)
    parser.add_argument("--crowdtensor-output-checkpoint", required=True)
    parser.add_argument("--crowdtensor-result", required=True)
    parser.add_argument("--fail-once-marker", default="")
    return parser.parse_args()


def _load(path: str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("fixture_contract_invalid")
    return value


def _atomic_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _distributed_parameter_sum(model: nn.Module) -> float:
    local = torch.zeros((), dtype=torch.float64)
    for parameter in model.parameters():
        value = parameter.detach()
        if hasattr(value, "to_local"):
            value = value.to_local()
        local += value.to(dtype=torch.float64).sum().cpu()
    dist.all_reduce(local, op=dist.ReduceOp.SUM)
    return float(local.item())


def _train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    global_step: int,
    rank: int,
    batch_size: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(9000 + global_step * 101 + rank)
    inputs = torch.randn(batch_size, 8, generator=generator)
    targets = torch.randn(batch_size, 4, generator=generator)
    optimizer.zero_grad(set_to_none=True)
    loss = (model(inputs) - targets).square().mean()
    loss.backward()
    optimizer.step()
    return loss.detach().to(dtype=torch.float64)


def main() -> None:
    args = _arguments()
    work = WorkUnit.from_dict(_load(args.crowdtensor_work_unit))
    base = CheckpointRef.from_dict(_load(args.crowdtensor_base_checkpoint))
    if work.base_checkpoint_hash != base.content_hash:
        raise ValueError("fixture_base_checkpoint_mismatch")

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.set_num_threads(1)
    torch.manual_seed(20260814)
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.GELU(),
        nn.Linear(16, 4),
    )
    fully_shard(model)
    if not isinstance(model, FSDPModule):
        raise RuntimeError("fixture_fsdp2_not_active")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    options = StateDictOptions(full_state_dict=False, cpu_offload=False)

    if base.step > 0:
        model_state, optimizer_state = get_state_dict(
            model, optimizer, options=options
        )
        state = {"model": model_state, "optimizer": optimizer_state}
        dcp.load(state, checkpoint_id=args.crowdtensor_base_payload)
        set_state_dict(
            model,
            optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
            options=options,
        )

    should_fail = torch.zeros(1, dtype=torch.int64)
    marker = Path(args.fail_once_marker) if args.fail_once_marker else None
    if rank == 0 and base.step > 0 and marker is not None and marker.is_file():
        marker.unlink()
        should_fail.fill_(1)
    dist.broadcast(should_fail, src=0)
    if should_fail.item():
        _train_step(
            model,
            optimizer,
            global_step=work.step_start,
            rank=rank,
            batch_size=2,
        )
        dist.destroy_process_group()
        raise SystemExit(23)

    final_loss = torch.zeros((), dtype=torch.float64)
    batch_size = 2
    for offset in range(work.step_count):
        global_step = work.step_start + offset
        final_loss = _train_step(
            model,
            optimizer,
            global_step=global_step,
            rank=rank,
            batch_size=batch_size,
        )

    output = Path(args.crowdtensor_output_checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
    dcp.save(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id=output,
    )
    dist.barrier()

    dist.all_reduce(final_loss, op=dist.ReduceOp.SUM)
    final_loss /= world_size
    parameter_sum = _distributed_parameter_sum(model)
    if rank == 0:
        result = {
            "schema": STABLE_TRAINER_RESULT_SCHEMA,
            "work_unit_hash": work.content_hash,
            "base_checkpoint_hash": base.content_hash,
            "step_start": work.step_start,
            "steps_completed": work.step_count,
            "rank_count": world_size,
            "distributed_type": "fsdp2",
            "device_type": "cpu",
            "restored_step": base.step,
            "samples": work.step_count * batch_size * world_size,
            "tokens": work.step_count * batch_size * world_size * 8,
            "metrics": [
                ["loss", float(final_loss.item())],
                ["parameter_sum", parameter_sum],
            ],
            "credential_values_public": False,
            "private_paths_public": False,
            "public_artifact_safe": True,
        }
        result["content_hash"] = stable_hash(result)
        _atomic_result(Path(args.crowdtensor_result), result)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
