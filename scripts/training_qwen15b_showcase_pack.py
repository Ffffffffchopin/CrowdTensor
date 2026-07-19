#!/usr/bin/env python3
"""Package public-safe evidence for a real Qwen 1.5B training showcase."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from scripts.training_cuda_kaggle_common import public_safety_errors
from crowdtensor.qwen15b_training import stable_hash


SCHEMA = "crowdtensor_qwen15b_training_showcase_v1"


def _load(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("showcase live report must be an object")
    return value


def _worker(report: dict[str, Any], role: str) -> dict[str, Any]:
    for item in report.get("worker_reports") or []:
        worker = dict(item.get("worker") or {})
        if worker.get("role") == role:
            return worker
    return {}


def _loss_summary(old_worker: dict[str, Any], new_worker: dict[str, Any]) -> dict[str, Any]:
    old_runtime = dict(old_worker.get("runtime") or {})
    new_runtime = dict(new_worker.get("runtime") or {})
    old_losses = [float(value) for value in old_runtime.get("step_mean_losses") or []]
    new_losses = [float(value) for value in new_runtime.get("step_mean_losses") or []]
    losses = old_losses + new_losses
    return {
        "optimizer_step_loss_count": len(losses),
        "optimizer_step_loss_start": losses[0] if losses else None,
        "optimizer_step_loss_end": losses[-1] if losses else None,
        "optimizer_step_loss_min": min(losses) if losses else None,
        "optimizer_step_loss_max": max(losses) if losses else None,
        "optimizer_step_loss_values_public": False,
        "optimizer_step_loss_trace_hash": stable_hash(losses),
    }


def build_showcase_report(
    live: dict[str, Any],
    *,
    minimum_steps: int = 128,
    minimum_training_tokens: int = 65536,
    minimum_relative_validation_improvement: float = 0.01,
) -> dict[str, Any]:
    target_steps = int(live.get("target_steps") or 0)
    replacement_step = int(live.get("replacement_generation_start_step") or 0)
    microbatches = int(live.get("microbatches_per_step") or 0)
    source = dict(live.get("source") or {})
    budget = dict(live.get("training_budget") or {})
    sequence_length = int(
        budget.get("sequence_length") or source.get("sequence_length") or 0
    )
    token_count = int(
        budget.get("training_token_count")
        or target_steps * microbatches * sequence_length
    )
    old = dict(live.get("old_generation") or {})
    new = dict(live.get("new_generation") or {})
    old_b = _worker(old, "kernel_b")
    new_b = _worker(new, "kernel_b")
    evaluation = dict(new_b.get("evaluation") or {})
    before_loss = float(evaluation.get("before_validation_loss") or 0.0)
    after_loss = float(evaluation.get("after_validation_loss") or 0.0)
    relative_improvement = (
        (before_loss - after_loss) / before_loss if before_loss > 0 else 0.0
    )
    evidence = dict(live.get("evidence") or {})
    cleanup = dict(live.get("cleanup") or {})
    old_refs = set(old.get("kernel_ref_hashes") or [])
    new_refs = set(new.get("kernel_ref_hashes") or [])
    gates = {
        "real_live_run": live.get("live_run_performed") is True
        and live.get("mock_runtime_used") is False
        and live.get("tiny_or_random_model_used") is False,
        "pinned_qwen_model": source.get("model_id") == "Qwen/Qwen2.5-1.5B"
        and bool(source.get("model_revision"))
        and int(source.get("parameter_count") or 0) >= 1_000_000_000,
        "pinned_public_dataset": bool(source.get("dataset_id"))
        and bool(source.get("dataset_revision"))
        and bool(source.get("train_token_hash"))
        and bool(source.get("validation_token_hash")),
        "large_bounded_training_budget": target_steps >= int(minimum_steps)
        and token_count >= int(minimum_training_tokens)
        and sequence_length >= 64,
        "two_generation_replacement": replacement_step > 0
        and replacement_step < target_steps
        and evidence.get("old_kernels_deleted_before_replacement") is True
        and evidence.get("entirely_new_miner_sessions_verified") is True
        and evidence.get("bounded_no_miner_pause_verified") is True,
        "exactly_once_contiguous_steps": evidence.get(
            "exactly_once_optimizer_commits_verified"
        ) is True
        and evidence.get("final_target_step_completed") is True
        and evidence.get("rendezvous_full_pipeline_verified") is True,
        "four_real_t4x2_kernel_sessions": len(old_refs) == 2
        and len(new_refs) == 2
        and old_refs.isdisjoint(new_refs)
        and evidence.get("real_cuda_only_verified") is True,
        "positive_real_lora_updates": bool(
            old_b.get("positive_lora_gradient_norms") is True
            and new_b.get("positive_lora_gradient_norms") is True
        ),
        "held_out_validation_improved": evaluation.get("evaluation_verified") is True
        and evaluation.get("validation_loss_reduced") is True
        and relative_improvement >= float(minimum_relative_validation_improvement)
        and math.isfinite(before_loss)
        and math.isfinite(after_loss),
        "standard_peft_export_and_reload": evaluation.get("standard_peft_cpu_load") is True
        and evaluation.get("standard_peft_cuda_load") is True
        and evaluation.get("standard_peft_format") is True
        and bool(dict(new.get("adapter_bundle") or {}).get("verified"))
        or bool(dict(new_b.get("export") or {}).get("standard_peft_format"))
        and evaluation.get("standard_peft_cpu_load") is True
        and evaluation.get("standard_peft_cuda_load") is True,
        "complete_cleanup": cleanup.get("live_resources_left_running") is False
        and cleanup.get("all_four_kernels_deleted") is True
        and cleanup.get("private_runtime_removed") is True
        and cleanup.get("rendezvous_payloads_removed") is True
        and cleanup.get("coordinator_stopped") is True
        and cleanup.get("tunnel_stopped") is True,
        "public_safe": live.get("public_artifact_safe") is True
        and not public_safety_errors(live),
    }
    # The boolean expression above intentionally accepts either the generation
    # summary or the worker export summary; normalize it for the public gate.
    gates["standard_peft_export_and_reload"] = bool(
        gates["standard_peft_export_and_reload"]
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": False,
        "showcase_ready": False,
        "goal_achieved": False,
        "execution_boundary": "Kaggle logical multi-node; not independent physical multi-host evidence",
        "model": {
            "model_id": source.get("model_id"),
            "model_revision": source.get("model_revision"),
            "parameter_count": int(source.get("parameter_count") or 0),
        },
        "dataset": {
            "dataset_id": source.get("dataset_id"),
            "dataset_revision": source.get("dataset_revision"),
            "sequence_length": sequence_length,
            "train_sequence_count": int(source.get("train_sequence_count") or 0),
            "validation_sequence_count": int(source.get("validation_sequence_count") or 0),
            "train_token_hash": source.get("train_token_hash"),
            "validation_token_hash": source.get("validation_token_hash"),
            "raw_text_public": False,
            "token_ids_public": False,
        },
        "training": {
            "target_optimizer_steps": target_steps,
            "replacement_step": replacement_step,
            "microbatches_per_step": microbatches,
            "training_token_count": token_count,
            "lora_rank": int(live.get("lora_rank") or 0),
            "lora_alpha": int(live.get("lora_alpha") or 0),
            "learning_rate": float(live.get("learning_rate") or 0.0),
            "loss_summary": _loss_summary(old_b, new_b),
        },
        "evaluation": {
            "validation_sequence_count": int(evaluation.get("validation_sequence_count") or 0),
            "before_validation_loss": before_loss,
            "after_validation_loss": after_loss,
            "before_validation_perplexity": math.exp(before_loss) if before_loss < 100 else None,
            "after_validation_perplexity": math.exp(after_loss) if after_loss < 100 else None,
            "relative_validation_loss_improvement": relative_improvement,
            "validation_loss_reduced": evaluation.get("validation_loss_reduced") is True,
            "standard_peft_cpu_load": evaluation.get("standard_peft_cpu_load") is True,
            "standard_peft_cuda_load": evaluation.get("standard_peft_cuda_load") is True,
            "evaluation_verified": evaluation.get("evaluation_verified") is True,
            "logits_values_public": False,
        },
        "elastic_continuation": {
            "old_generation_kernel_count": len(old_refs),
            "replacement_generation_kernel_count": len(new_refs),
            "old_kernel_refs_public": False,
            "new_kernel_refs_public": False,
            "old_kernels_deleted_before_replacement": evidence.get(
                "old_kernels_deleted_before_replacement"
            ) is True,
            "zero_miner_pause_verified": evidence.get("bounded_no_miner_pause_verified") is True,
            "new_miner_checkpoint_restore_verified": evidence.get(
                "new_miners_restore_step4_verified"
            ) is True,
            "exactly_once_contiguous_steps": evidence.get(
                "exactly_once_optimizer_commits_verified"
            ) is True,
        },
        "adapter": {
            "standard_peft_format": evaluation.get("standard_peft_format") is True
            or dict(new_b.get("export") or {}).get("standard_peft_format") is True,
            "archive_verified": dict(new.get("adapter_bundle") or {}).get("verified") is True,
            "adapter_file_hash": dict(new.get("adapter_bundle") or {}).get("adapter_file_hash", ""),
            "adapter_tensor_values_public": False,
        },
        "gates": gates,
        "cleanup": {
            "live_resources_left_running": cleanup.get("live_resources_left_running"),
            "all_four_kernels_deleted": cleanup.get("all_four_kernels_deleted"),
            "private_runtime_removed": cleanup.get("private_runtime_removed"),
            "rendezvous_payloads_removed": cleanup.get("rendezvous_payloads_removed"),
            "coordinator_stopped": cleanup.get("coordinator_stopped"),
            "tunnel_stopped": cleanup.get("tunnel_stopped"),
        },
        "public_artifact_safe": False,
        "blockers": [],
    }
    result["blockers"] = sorted(
        f"showcase_gate_{name}" for name, passed in gates.items() if not passed
    )
    result["ok"] = not result["blockers"]
    result["showcase_ready"] = result["ok"]
    result["goal_achieved"] = result["ok"]
    result["public_artifact_safe"] = not public_safety_errors(result)
    if not result["public_artifact_safe"]:
        result["blockers"].append("public_artifact_safety_violation")
        result["ok"] = False
        result["showcase_ready"] = False
        result["goal_achieved"] = False
    result["content_hash"] = stable_hash(
        {key: value for key, value in result.items() if key != "content_hash"}
    )
    return result


def write_markdown(report: dict[str, Any], path: Path) -> None:
    model = report["model"]
    dataset = report["dataset"]
    training = report["training"]
    evaluation = report["evaluation"]
    continuation = report["elastic_continuation"]
    status = "READY" if report.get("showcase_ready") else "BLOCKED"
    lines = [
        "# CrowdTensor Qwen2.5-1.5B Elastic Training Showcase",
        "",
        f"Status: **{status}**",
        "",
        "This artifact describes a real LoRA causal-language-model adaptation run "
        "through CrowdTensor's Kaggle logical multi-node training path.",
        "",
        "## Configuration",
        "",
        f"- Model: `{model['model_id']}` ({model['parameter_count']:,} parameters)",
        f"- Dataset: `{dataset['dataset_id']}` at a pinned revision",
        f"- Training budget: `{training['target_optimizer_steps']}` optimizer steps, "
        f"`{training['training_token_count']:,}` training tokens",
        f"- Replacement boundary: step `{training['replacement_step']}`",
        "- Placement: two concurrent T4x2 Kernel generations, four stage sessions total",
        "",
        "## Held-Out Result",
        "",
        f"- Base validation loss: `{evaluation['before_validation_loss']:.6f}`",
        f"- Adapter validation loss: `{evaluation['after_validation_loss']:.6f}`",
        f"- Relative loss improvement: `{evaluation['relative_validation_loss_improvement']:.2%}`",
        f"- Base perplexity: `{evaluation['before_validation_perplexity']:.4f}`",
        f"- Adapter perplexity: `{evaluation['after_validation_perplexity']:.4f}`",
        "",
        "The exported adapter was reloaded in standard PEFT format on CPU and CUDA "
        "before the result was accepted. Raw prompts, token IDs, weights, activations, "
        "credentials, and runtime-private URLs are excluded from the public artifact.",
        "",
        "## Elastic Evidence",
        "",
        f"- Original generation deleted before replacement: `{continuation['old_kernels_deleted_before_replacement']}`",
        f"- Zero-Miner pause observed: `{continuation['zero_miner_pause_verified']}`",
        f"- Replacement checkpoint restore: `{continuation['new_miner_checkpoint_restore_verified']}`",
        f"- Exactly-once contiguous optimizer steps: `{continuation['exactly_once_contiguous_steps']}`",
        "",
        "This is logical multi-node evidence using Kaggle runtimes. It does not claim "
        "independent physical multi-host network performance.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-steps", type=int, default=128)
    parser.add_argument("--minimum-training-tokens", type=int, default=65536)
    parser.add_argument("--minimum-relative-validation-improvement", type=float, default=0.01)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = build_showcase_report(
        _load(args.live_report),
        minimum_steps=int(args.minimum_steps),
        minimum_training_tokens=int(args.minimum_training_tokens),
        minimum_relative_validation_improvement=float(
            args.minimum_relative_validation_improvement
        ),
    )
    report_path = output / "training_qwen15b_showcase.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, output / "TRAINING_SHOWCASE.md")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "training_qwen15b_showcase "
            f"ready={report['showcase_ready']} blockers={','.join(report['blockers']) or 'none'}"
        )
    return 0 if report["public_artifact_safe"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
