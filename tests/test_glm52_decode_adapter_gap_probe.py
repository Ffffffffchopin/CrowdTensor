from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scripts import glm52_decode_adapter_gap_check as check
from scripts import glm52_decode_adapter_gap_probe as probe


def _tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="ct_glm52_decode_gap_"))


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def _metadata() -> dict:
    return {
        "model_id": probe.MODEL_ID,
        "model_repo": probe.DEFAULT_MODEL_REPO,
        "config_ready": True,
        "index_ready": True,
        "model_type": "glm_moe_dsa",
        "architectures": ["GlmMoeDsaForCausalLM"],
        "num_hidden_layers": 78,
        "hidden_size": 6144,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 192,
        "v_head_dim": 256,
        "first_k_dense_replace": 3,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048,
        "weight_key_count": 232269,
        "total_weight_size_bytes": 440335957008,
        "total_weight_size_gb": 440.335957,
        "family_hits": {
            "attention_low_rank": True,
            "rope_nope_attention": True,
            "dense_mlp": True,
            "moe_experts": True,
            "moe_router": True,
            "awq_int4_tensors": True,
            "lm_head": True,
        },
        "metadata_errors": [],
    }


def _stage(provider: str, stage_id: int, *, decode: bool = False, runtime_kind: str = "") -> dict:
    return {
        "schema": "glm52_kaggle_stage_runtime_report_v1",
        "ok": True,
        "model_id": probe.MODEL_ID,
        "compatible_weight_repo": probe.DEFAULT_MODEL_REPO,
        "provider": provider,
        "stage_id": stage_id,
        "stage_layer_range": [stage_id * 26, (stage_id + 1) * 26],
        "coordinator_request_id_hash": _hash("b"),
        "stage_execution_verified": True,
        "stage_decode_verified": decode,
        "same_request_route_verified": decode,
        "stage_runtime_kind": runtime_kind,
        "stage_output_hash": _hash(str(stage_id)),
        "weight_tensor_values_loaded": True,
        "weight_value_byte_count": 16,
        "weight_value_sha256": _hash("c"),
        "weight_tensor_values_public": False,
        "live_run_performed": True,
        "stage_smoke_only": False,
        "public_artifact_safe": True,
        "blockers": [] if decode else ["glm52_stage_value_provider_op_is_not_full_decode"],
    }


def _same(*, verified: bool = False) -> dict:
    return {
        "schema": "glm52_kaggle_same_request_probe_v1",
        "ok": verified,
        "glm52_kaggle_same_request_verified": verified,
        "same_request_decode_verified": verified,
        "live_run_performed": verified,
        "public_artifact_safe": True,
        "model": {"model_id": probe.MODEL_ID},
        "success": {
            "same_request_decode_verified": verified,
            "generated_token_count": 1 if verified else 0,
            "generated_token_hash": _hash("a") if verified else "",
            "accepted_providers": probe.REQUIRED_PROVIDERS if verified else [],
        },
        "same_request": {
            "coordinator_request_verified": verified,
            "coordinator_request_id_hash": _hash("b"),
            "model_id": probe.MODEL_ID,
        },
        "cleanup": {
            "temporary_kaggle_kernels_deleted": verified,
            "temporary_private_packages_removed": verified,
            "live_resources_left_running": False if verified else None,
            "public_artifact_safe": True,
        },
    }


def _activation_handoff() -> dict:
    return {
        "schema": "glm52_stage_activation_handoff_probe_v1",
        "ok": True,
        "glm52_stage_activation_handoff_probe_ready": True,
        "stage_activation_handoff_runtime_verified": True,
        "stage_activation_handoff_contract_verified": True,
        "same_request_decode_verified": False,
        "stage_decode_verified": False,
        "generated_token_verified": False,
        "model_id": probe.MODEL_ID,
        "stage_count": 3,
        "handoff_count": 2,
        "stage_runtime_provider_coverage": probe.REQUIRED_PROVIDERS,
        "blockers": [],
        "public_artifact_safe": True,
    }


def _component_report(schema: str, **fields: object) -> dict:
    payload = {
        "schema": schema,
        "ok": True,
        "model_id": probe.MODEL_ID,
        "model_repo": probe.DEFAULT_MODEL_REPO,
        "model_type": "glm_moe_dsa",
        "stage_decode_verified": False,
        "same_request_decode_verified": False,
        "generated_token_verified": False,
        "public_artifact_safe": True,
        "blockers": ["glm52_stage_decode_not_verified"],
    }
    payload.update(fields)
    return payload


def _component_reports(base: Path) -> dict[str, Path]:
    return {
        "attention_projection": _write(
            base / "attention_projection.json",
            _component_report(
                "glm52_attention_projection_probe_v1",
                attention_projection_verified=True,
                input_layernorm_verified=True,
                q_lora_projection_verified=True,
                kv_lora_projection_verified=True,
            ),
        ),
        "attention_single": _write(
            base / "attention_single.json",
            _component_report(
                "glm52_attention_single_token_probe_v1",
                single_token_attention_verified=True,
                rope_applied=True,
                attention_scores_verified=True,
                attention_weights_verified=True,
                o_proj_verified=True,
            ),
        ),
        "kv_cache": _write(
            base / "kv_cache.json",
            _component_report(
                "glm52_kv_cache_decode_probe_v1",
                kv_cache_prefill_verified=True,
                kv_cache_update_verified=True,
                kv_cache_decode_attention_verified=True,
                o_proj_verified=True,
            ),
        ),
        "dsa_masked": _write(
            base / "dsa_masked.json",
            _component_report(
                "glm52_dsa_masked_layer_decode_probe_v1",
                layer_decode_verified=True,
                attention_decode_verified=True,
                dsa_masked_attention_integrated=True,
                dsa_indexer_verified=True,
                full_moe_mlp_verified=True,
                kv_cache_prefill_verified=True,
                kv_cache_update_verified=True,
            ),
        ),
        "lm_head": _write(
            base / "lm_head.json",
            _component_report(
                "glm52_lm_head_token_probe_v1",
                lm_head_logits_token_selection_verified=True,
                lm_head_streamed_full_vocab=True,
            ),
        ),
        "stage_hidden_lm_head": _write(
            base / "stage_hidden_lm_head.json",
            _component_report(
                "glm52_stage_hidden_lm_head_probe_v1",
                stage_hidden_lm_head_token_selection_verified=True,
                stage_hidden_to_lm_head_verified=True,
                stage_layer_decode_verified=True,
                lm_head_streamed_full_vocab=True,
            ),
        ),
        "dequant": _write(
            base / "dequant.json",
            _component_report(
                "glm52_pack_quantized_dequant_probe_v1",
                pack_quantized_dequant_verified=True,
                pack_quantized_linear_slice_verified=True,
            ),
        ),
        "expert_mlp": _write(
            base / "expert_mlp.json",
            _component_report(
                "glm52_pack_quantized_expert_mlp_probe_v1",
                pack_quantized_expert_mlp_verified=True,
                single_expert_mlp_verified=True,
            ),
        ),
        "router_gather": _write(
            base / "router_gather.json",
            _component_report(
                "glm52_pack_quantized_router_gather_probe_v1",
                router_topk_verified=True,
                routed_expert_subset_verified=True,
                executed_expert_count=8,
            ),
        ),
        "moe_mlp": _write(
            base / "moe_mlp.json",
            _component_report(
                "glm52_pack_quantized_moe_mlp_probe_v1",
                full_moe_mlp_verified=True,
                shared_experts_mlp_verified=True,
                router_topk_verified=True,
                routed_expert_gather_verified=True,
                executed_expert_count=8,
            ),
        ),
    }


def test_gap_probe_reports_stage_value_op_as_not_full_decode(monkeypatch) -> None:
    monkeypatch.setattr(probe, "model_metadata", lambda _args: _metadata())
    base = _tmp_dir()
    stage_paths = [
        _write(base / "gpu.json", _stage("kaggle_cuda", 0, runtime_kind="glm52_awq_stage_value_provider_op")),
        _write(base / "tpu.json", _stage("kaggle_jax_tpu", 1, runtime_kind="glm52_awq_stage_value_provider_op")),
        _write(base / "cpu.json", _stage("kaggle_cpu", 2, runtime_kind="glm52_awq_stage_value_provider_op")),
    ]
    same = _write(base / "same.json", _same(verified=False))
    argv = ["--output-dir", str(base / "gap"), "--same-request-report", str(same)]
    for path in stage_paths:
        argv.extend(["--stage-report", str(path)])

    report = probe.build_report(probe.parse_args(argv))

    assert report["decode_adapter_ready"] is False
    assert "glm52_stage_value_provider_op_is_not_full_decode" in report["blockers"]
    assert "glm52_same_request_decode_not_verified" in report["blockers"]
    assert "glm52_decode_capability_missing:awq_int4_dequant_linear_runtime" in report["blockers"]
    assert set(report["stage_runtime_provider_coverage"]) == set(probe.REQUIRED_PROVIDERS)
    assert report["stage_decode_provider_coverage"] == []
    assert check.validate_report(report) == []
    assert "decode_adapter_not_ready" in check.validate_report(report, require_ready=True)


def test_gap_probe_accepts_real_same_request_decode_contract(monkeypatch) -> None:
    monkeypatch.setattr(probe, "model_metadata", lambda _args: _metadata())
    base = _tmp_dir()
    stage_paths = [
        _write(base / "gpu.json", _stage("kaggle_cuda", 0, decode=True)),
        _write(base / "tpu.json", _stage("kaggle_jax_tpu", 1, decode=True)),
        _write(base / "cpu.json", _stage("kaggle_cpu", 2, decode=True)),
    ]
    same = _write(base / "same.json", _same(verified=True))
    argv = ["--output-dir", str(base / "gap"), "--same-request-report", str(same)]
    for path in stage_paths:
        argv.extend(["--stage-report", str(path)])

    report = probe.build_report(probe.parse_args(argv))

    assert report["decode_adapter_ready"] is True
    assert report["blockers"] == []
    assert {item["capability"] for item in report["required_capabilities"]} == set(probe.REQUIRED_CAPABILITIES)
    assert all(item["verified"] is True for item in report["required_capabilities"])
    assert check.validate_report(report, require_ready=True) == []


def test_gap_probe_can_credit_activation_handoff_without_claiming_decode(monkeypatch) -> None:
    monkeypatch.setattr(probe, "model_metadata", lambda _args: _metadata())
    base = _tmp_dir()
    stage_paths = [
        _write(base / "gpu.json", _stage("kaggle_cuda", 0, runtime_kind="glm52_awq_stage_value_provider_op")),
        _write(base / "tpu.json", _stage("kaggle_jax_tpu", 1, runtime_kind="glm52_awq_stage_value_provider_op")),
        _write(base / "cpu.json", _stage("kaggle_cpu", 2, runtime_kind="glm52_awq_stage_value_provider_op")),
    ]
    same = _write(base / "same.json", _same(verified=False))
    activation_handoff = _write(base / "handoff.json", _activation_handoff())
    argv = [
        "--output-dir",
        str(base / "gap"),
        "--same-request-report",
        str(same),
        "--activation-handoff-report",
        str(activation_handoff),
    ]
    for path in stage_paths:
        argv.extend(["--stage-report", str(path)])

    report = probe.build_report(probe.parse_args(argv))

    statuses = {item["capability"]: item for item in report["required_capabilities"]}
    assert report["decode_adapter_ready"] is False
    assert statuses["stage_activation_handoff_runtime"]["verified"] is True
    assert statuses["stage_activation_handoff_runtime"]["evidence"] == "stage_activation_handoff_probe"
    assert "glm52_decode_capability_missing:stage_activation_handoff_runtime" not in report["blockers"]
    assert "glm52_decode_capability_missing:awq_int4_dequant_linear_runtime" in report["blockers"]
    assert check.validate_report(report) == []


def test_gap_probe_can_credit_component_runtime_proofs_without_claiming_same_request(monkeypatch) -> None:
    monkeypatch.setattr(probe, "model_metadata", lambda _args: _metadata())
    base = _tmp_dir()
    stage_paths = [
        _write(base / "gpu.json", _stage("kaggle_cuda", 0, runtime_kind="glm52_awq_stage_value_provider_op")),
        _write(base / "tpu.json", _stage("kaggle_jax_tpu", 1, runtime_kind="glm52_awq_stage_value_provider_op")),
        _write(base / "cpu.json", _stage("kaggle_cpu", 2, runtime_kind="glm52_awq_stage_value_provider_op")),
    ]
    same = _write(base / "same.json", _same(verified=False))
    activation_handoff = _write(base / "handoff.json", _activation_handoff())
    components = _component_reports(base / "components")
    argv = [
        "--output-dir",
        str(base / "gap"),
        "--same-request-report",
        str(same),
        "--activation-handoff-report",
        str(activation_handoff),
        "--attention-projection-report",
        str(components["attention_projection"]),
        "--attention-single-token-report",
        str(components["attention_single"]),
        "--kv-cache-decode-report",
        str(components["kv_cache"]),
        "--dsa-masked-layer-decode-report",
        str(components["dsa_masked"]),
        "--lm-head-token-report",
        str(components["lm_head"]),
        "--stage-hidden-lm-head-report",
        str(components["stage_hidden_lm_head"]),
        "--pack-quantized-dequant-report",
        str(components["dequant"]),
        "--pack-quantized-expert-mlp-report",
        str(components["expert_mlp"]),
        "--pack-quantized-router-gather-report",
        str(components["router_gather"]),
        "--pack-quantized-moe-mlp-report",
        str(components["moe_mlp"]),
    ]
    for path in stage_paths:
        argv.extend(["--stage-report", str(path)])

    report = probe.build_report(probe.parse_args(argv))

    statuses = {item["capability"]: item for item in report["required_capabilities"]}
    assert report["decode_adapter_ready"] is False
    assert report["same_request_decode_ready"] is False
    assert statuses["coordinator_same_request_decode_runtime"]["verified"] is False
    for capability in probe.REQUIRED_CAPABILITIES:
        if capability != "coordinator_same_request_decode_runtime":
            assert statuses[capability]["verified"] is True, capability
            assert "glm52_decode_capability_missing:" + capability not in report["blockers"]
    assert "glm52_decode_capability_missing:coordinator_same_request_decode_runtime" in report["blockers"]
    assert "glm52_same_request_decode_not_verified" in report["blockers"]
    assert check.validate_report(report) == []
