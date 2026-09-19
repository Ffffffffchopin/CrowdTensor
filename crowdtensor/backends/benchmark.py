"""Real, offline CPU calibration for the intermittent-training study.

Tiny synthetic data tests the measurement and recovery path, not model quality
on a language benchmark. Transformers owns the checkpoint reference trainer.
"""

from __future__ import annotations

import copy
from importlib.metadata import version
from pathlib import Path
import random
import time
from typing import Any

from crowdtensor.core.benchmark import BenchmarkError, sealed, write_json
from crowdtensor.core.controller import SessionController
from crowdtensor.hf_lora_training import (
    _load_causal_lm,
    _load_peft_adapter,
    configure_cpu_determinism,
    create_local_training_fixture,
    evaluate_adapter,
    load_token_rows,
)
from crowdtensor.named_tensor_optimizer import load_tensors, named_tensor_hash
from crowdtensor.training_contract import public_training_spec, sha256_json
from crowdtensor.volunteer_training_cell import LocalVolunteerTransport, VolunteerTrainingCell
from crowdtensor.volunteer_training_coordinator import VolunteerTrainingCoordinator
from crowdtensor.volunteer_training_protocol import VolunteerProtocolError

from .elastic_peft import VolunteerControllerTransport


class _DiscardedResult(RuntimeError):
    pass


class _RecordingTransport(LocalVolunteerTransport):
    def __init__(self, coordinator, token, *, discard: bool = False):
        super().__init__(coordinator, token)
        self.discard = discard
        self.last_submission: dict[str, Any] | None = None
        self.copied_artifact_bytes = 0

    def download_artifact(self, ref, destination, *, max_bytes):
        count = super().download_artifact(ref, destination, max_bytes=max_bytes)
        self.copied_artifact_bytes += count
        return count

    def submit(self, **kwargs):
        self.last_submission = kwargs
        if self.discard:
            self.discard = False
            raise _DiscardedResult("benchmark_result_discarded_before_submission")
        return super().submit(**kwargs)


def _evaluate(fixture: dict[str, Any], adapter: Path) -> dict[str, Any]:
    result = evaluate_adapter(
        base_model_path=fixture["model"]["base_model_path"],
        adapter_path=adapter,
        dataset_path=fixture["dataset"]["private_validation_dataset_path"],
        sample_indexes=list(range(fixture["dataset"]["validation_sample_count"])),
        device="cpu",
    )
    return {
        "metric": "synthetic_heldout_token_loss", "mean_loss": result["mean_loss"],
        "sample_count": result["sample_count"], "logits_hash": result["logits_hash"],
        "dataset_hash": fixture["dataset"]["validation_file_hash"],
    }


def _upstream_trial(fixture, root: Path, *, interrupted: bool, total_steps: int):
    import torch
    from peft import PeftModel
    from transformers import Trainer, TrainerCallback, TrainingArguments, default_data_collator
    from transformers.trainer_callback import PrinterCallback

    configure_cpu_determinism(fixture["seed"])
    boundary = total_steps // 2

    class StopAfterUncommittedStep(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == boundary + 1:
                control.should_training_stop = True
                control.should_save = False
            return control

    rows = load_token_rows(fixture["dataset"]["private_dataset_path"])
    dataset = [{"input_ids": row["input_ids"], "labels": row["input_ids"]} for row in rows]
    arguments = TrainingArguments(
        output_dir=str(root / "checkpoints"), use_cpu=True,
        per_device_train_batch_size=fixture["local_training"]["batch_size"],
        max_steps=total_steps, learning_rate=fixture["local_training"]["learning_rate"],
        weight_decay=0.0, lr_scheduler_type="constant", optim="adamw_torch",
        max_grad_norm=0.0, save_strategy="steps", save_steps=boundary,
        logging_strategy="no", report_to=[], disable_tqdm=True,
        seed=fixture["seed"], data_seed=fixture["seed"],
        dataloader_num_workers=0, dataloader_pin_memory=False,
    )

    def new_trainer(*, stop: bool = False):
        base = _load_causal_lm(fixture["model"]["base_model_path"])
        model, _ = _load_peft_adapter(
            base, PeftModel, fixture["lora"]["adapter_path"], is_trainable=True
        )
        trainer = Trainer(
            model=model, args=arguments, train_dataset=dataset,
            data_collator=default_data_collator,
            callbacks=[StopAfterUncommittedStep()] if stop else [],
        )
        trainer.remove_callback(PrinterCallback)
        return trainer

    started = time.monotonic()
    trainer = new_trainer(stop=interrupted)
    trainer.train()
    recovery_seconds = None
    if interrupted:
        if trainer.state.global_step != boundary + 1:
            raise BenchmarkError("benchmark_upstream_interruption_not_exercised")
        checkpoint = root / "checkpoints" / f"checkpoint-{boundary}"
        for required in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
            if not (checkpoint / required).is_file():
                raise BenchmarkError("benchmark_upstream_checkpoint_incomplete")
        del trainer
        recovery_started = time.monotonic()
        trainer = new_trainer()
        # Trainer restores optimizer, scheduler, RNG, and the data-loader position.
        trainer.train(resume_from_checkpoint=str(checkpoint))
        recovery_seconds = time.monotonic() - recovery_started
    if trainer.state.global_step != total_steps:
        raise BenchmarkError("benchmark_upstream_step_budget_mismatch")
    adapter = root / "adapter"
    trainer.model.save_pretrained(adapter, safe_serialization=True)
    elapsed = time.monotonic() - started
    adapter_hash = named_tensor_hash(load_tensors(adapter / "adapter_model.safetensors"))
    del trainer
    return {
        "metrics": {
            "accepted_optimizer_steps": total_steps,
            "discarded_optimizer_steps": int(interrupted),
            "accepted_tokens": total_steps * fixture["local_training"]["batch_size"] * fixture["local_training"]["sequence_length"],
            "elapsed_seconds": elapsed,
            "recovery_and_remaining_training_seconds": recovery_seconds,
            "uploaded_delta_bytes": None,
            "checkpoint_bytes_on_disk": sum(p.stat().st_size for p in (root / "checkpoints").rglob("*") if p.is_file()),
        },
        "adapter_hash": adapter_hash,
        "quality": _evaluate(fixture, adapter),
        "recovery_kind": "trainer_checkpoint_plus_optimizer_rng_cursor" if interrupted else "none",
        "torch_version": torch.__version__,
    }


def _elastic_trial(fixture, root: Path, *, interrupted: bool, steps: int, total_steps: int):
    job = copy.deepcopy(fixture)
    job["local_training"].update(local_steps=steps, step_end=steps)
    job["job_hash"] = sha256_json(public_training_spec({k: v for k, v in job.items() if k != "job_hash"}))
    clock = [time.time()]
    coordinator = VolunteerTrainingCoordinator.create_from_fixture(
        root / "campaign", job, campaign_id="benchmark-fixture",
        target_rounds=total_steps // (2 * steps), minimum_quorum=2,
        lease_seconds=3600, max_loss_increase=100.0,
        clock=lambda: clock[0],
    )
    token = coordinator.private_invite()["invite_token"]
    local = _RecordingTransport(coordinator, token, discard=interrupted)
    transports = [local]
    transport = VolunteerControllerTransport(local, root / "controller")
    completed = []
    attempted_steps = 0
    stale_rejected = False
    idempotent = False
    recovery_seconds = None
    started = time.monotonic()
    attempts = total_steps // steps + int(interrupted)
    for index in range(attempts):
        cell = VolunteerTrainingCell(
            transport, root / f"cell-{index}", cell_id=f"logical-{index}",
            device="cpu", max_local_steps=steps, cache_dir=root / "cache",
        )
        attempted_steps += steps
        try:
            result = cell.join_once()
        except _DiscardedResult:
            recovery_started = time.monotonic()
            discarded = local.last_submission
            clock[0] = time.time() + 7200
            coordinator = VolunteerTrainingCoordinator(root / "campaign", clock=lambda: clock[0])
            coordinator.recover_after_restart()
            coordinator.expire_leases(invite_token=token)
            local = _RecordingTransport(coordinator, token)
            transports.append(local)
            transport = VolunteerControllerTransport(local, root / "controller")
            try:
                transport.submit(**discarded)
            except VolunteerProtocolError as exc:
                if exc.code not in {"volunteer_lease_not_active", "volunteer_lease_expired"}:
                    raise
                stale_rejected = True
            else:
                raise BenchmarkError("benchmark_expired_result_was_accepted")
            recovery_seconds = time.monotonic() - recovery_started
            continue
        if not result.get("work_completed") or not result["submission"].get("accepted"):
            raise BenchmarkError("benchmark_work_not_accepted")
        completed.append(result)
        if not idempotent:
            replay = transport.submit(**local.last_submission)
            if replay.get("idempotent_replay") is not True:
                raise BenchmarkError("benchmark_duplicate_not_idempotent")
            idempotent = True
    elapsed = time.monotonic() - started
    evaluation = coordinator.evaluate_campaign(heldout_quality=True)
    status = coordinator.status()
    controller = SessionController(root / "controller").status()
    accepted_steps = sum(item["optimizer_steps"] for item in completed)
    if (
        accepted_steps != total_steps or not status["campaign_complete"]
        or controller["terminal_count"] != len(completed)
        or not coordinator.verify_ledger()["ok"]
    ):
        raise BenchmarkError("benchmark_elastic_accounting_mismatch")
    return {
        "metrics": {
            "accepted_optimizer_steps": accepted_steps,
            "discarded_optimizer_steps": attempted_steps - accepted_steps,
            "accepted_tokens": status["accepted_tokens_seen"],
            "elapsed_seconds": elapsed, "controller_recovery_seconds": recovery_seconds,
            "uploaded_delta_bytes": status["uploaded_delta_bytes"],
            "local_artifact_copy_bytes_including_failed_attempts": sum(item.copied_artifact_bytes for item in transports),
            "accepted_artifact_download_bytes": sum(item["artifact_download_bytes"] for item in completed),
            "completed_rounds": status["completed_rounds"],
            "expired_leases": status["expired_lease_count"],
            "reassignments": status["reassigned_work_count"],
        },
        "adapter_hash": status["canonical_adapter_hash"],
        "quality": {
            "metric": "synthetic_heldout_token_loss",
            "mean_loss": evaluation["quality"]["candidate_mean_loss"],
            "sample_count": evaluation["quality"]["heldout_sample_count"],
            "dataset_hash": evaluation["quality"]["heldout_dataset_hash"],
        },
        "duplicate_receipt_idempotent": idempotent,
        "stale_result_rejected": stale_rejected if interrupted else None,
        "checkpoint_lineage_verified": evaluation["checkpoint_lineage_verified"],
        "recovery_kind": "discarded_result_expiry_and_reassignment" if interrupted else "none",
    }


def run_cpu_calibration(protocol: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    private = root / ".private"
    private.mkdir(mode=0o700)
    tasks = [
        (seed, condition, policy)
        for seed in protocol["seeds"]
        for condition in protocol["conditions"]
        for policy in ("upstream_checkpoint", *protocol["fixed_steps"])
    ]
    random.Random(0).shuffle(tasks)
    fixtures = {}
    baselines = {}
    trials = []
    for index, (seed, condition, policy) in enumerate(tasks):
        if seed not in fixtures:
            fixtures[seed] = create_local_training_fixture(
                private / f"fixture-{seed}", seed=seed, row_count=8,
                sequence_length=16, local_steps=1, learning_rate=0.01,
            )
            baselines[seed] = _evaluate(fixtures[seed], Path(fixtures[seed]["lora"]["adapter_path"]))
        fixture = fixtures[seed]
        trial_root = private / f"trial-{index}"
        trial_root.mkdir(mode=0o700)
        kwargs = {
            "interrupted": condition == "intermittent",
            "total_steps": protocol["cpu_total_steps"],
        }
        try:
            trial = (
                _upstream_trial(fixture, trial_root, **kwargs)
                if policy == "upstream_checkpoint"
                else _elastic_trial(fixture, trial_root, steps=protocol["fixed_steps"][policy], **kwargs)
            )
        except Exception as exc:
            write_json(root / f"trial-{index:03d}-failed.json", sealed({
                "policy": policy, "condition": condition, "seed": seed,
                "protocol_hash": protocol["content_hash"], "execution_order": index,
                "state": "failed", "error_type": type(exc).__name__,
            }))
            raise
        trial.update(
            policy=policy, condition=condition, seed=seed,
            protocol_hash=protocol["content_hash"], execution_order=index,
            evidence_kind="real_cpu_tiny_fixture",
            model_parameter_count=fixture["model"]["parameter_count"],
            model_weights_hash=fixture["model"]["base_model_hash"],
            training_data_hash=fixture["dataset"]["dataset_file_hash"],
            baseline_quality=baselines[seed],
            versions={package: version(package) for package in ("torch", "transformers", "peft", "accelerate")},
            simultaneous_workers=1, natural_language_benchmark=False,
        )
        trial = sealed(trial)
        write_json(root / f"trial-{index:03d}.json", trial)
        trials.append(trial)
    for seed in protocol["seeds"]:
        for policy in ("upstream_checkpoint", *protocol["fixed_steps"]):
            pair = [trial for trial in trials if trial["seed"] == seed and trial["policy"] == policy]
            if len({trial["adapter_hash"] for trial in pair}) != 1:
                raise BenchmarkError("benchmark_recovery_changed_final_adapter")
    return trials
