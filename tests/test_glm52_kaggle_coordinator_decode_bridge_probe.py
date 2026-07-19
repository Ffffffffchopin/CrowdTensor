from __future__ import annotations

import json
import tempfile
from pathlib import Path

from scripts import glm52_kaggle_coordinator_decode_bridge_check as bridge_check
from scripts import glm52_kaggle_coordinator_decode_bridge_probe as bridge
from scripts import glm52_kaggle_same_request_check as same_request_check
from scripts import glm52_kaggle_same_request_probe as same_request


def _tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="ct_glm52_coordinator_bridge_"))


def _hash(label: str) -> str:
    return bridge.sha_json({"test": label})


def _stage_specs() -> list[dict]:
    return [
        {
            "stage_id": 0,
            "stage_count": 3,
            "provider": "kaggle_cuda",
            "stage_layer_range": [0, 2],
            "compatible_weight_repo": bridge.COMPATIBLE_WEIGHT_REPO,
        },
        {
            "stage_id": 1,
            "stage_count": 3,
            "provider": "kaggle_jax_tpu",
            "stage_layer_range": [2, 4],
            "compatible_weight_repo": bridge.COMPATIBLE_WEIGHT_REPO,
        },
        {
            "stage_id": 2,
            "stage_count": 3,
            "provider": "kaggle_cpu",
            "stage_layer_range": [4, 6],
            "compatible_weight_repo": bridge.COMPATIBLE_WEIGHT_REPO,
        },
    ]


def _activation(stage_id: int) -> dict:
    return {
        "schema": "glm52_private_stage_activation_v1",
        "activation_hash": _hash(f"activation-{stage_id}"),
        "hidden_shape": [3, 8],
        "hidden_dtype": "float16",
        "hidden_b64": f"PRIVATE_HIDDEN_PAYLOAD_{stage_id}",
        "activation_public": False,
    }


def _submit_payload(task: dict, *, final: bool = False) -> dict:
    stage_id = int(task["stage_id"])
    payload = {
        "task_id": task["task_id"],
        "stage_id": stage_id,
        "generation_step": int(task.get("generation_step") or 0),
        "public_artifact_safe": True,
        "stage_decode_verified": True,
        "stage_output_hash": _hash(f"stage-output-{stage_id}"),
        "output_hash": _hash(f"output-{stage_id}"),
        "weight_value_sha256": _hash(f"weight-{stage_id}"),
        "weight_value_byte_count": 16 + stage_id,
        "provider_runtime_verified": True,
        "provider_device_count": 1,
    }
    if final:
        payload["generated_token_hash"] = _hash("generated-token")
    else:
        payload["activation"] = _activation(stage_id)
        payload["activation_hash"] = payload["activation"]["activation_hash"]
    return payload


def test_state_routes_three_provider_stages_without_public_activation_payload() -> None:
    state = bridge.Glm52CoordinatorState(
        stage_specs=_stage_specs(),
        coordinator_request_id_hash=_hash("request"),
        max_new_tokens=1,
    )

    task0 = state.claim(miner_id="cuda-worker-secret", stage_id=0)["task"]
    assert state.submit(_submit_payload(task0))["accepted"] is True
    task1 = state.claim(miner_id="tpu-worker-secret", stage_id=1)["task"]
    assert task1["activation"]["hidden_b64"] == "PRIVATE_HIDDEN_PAYLOAD_0"
    assert state.submit(_submit_payload(task1))["accepted"] is True
    task2 = state.claim(miner_id="cpu-worker-secret", stage_id=2)["task"]
    assert task2["activation"]["hidden_b64"] == "PRIVATE_HIDDEN_PAYLOAD_1"
    assert state.submit(_submit_payload(task2, final=True))["ready"] is True

    status = state.public_status()
    encoded_status = json.dumps(status, sort_keys=True)
    assert "PRIVATE_HIDDEN_PAYLOAD" not in encoded_status
    assert '"hidden_b64":' not in encoded_status
    assert bridge.public_redaction_errors(status) == []
    assert status["generated_token_count"] == 1
    assert status["activation_public"] is False

    same_args = same_request.parse_args(["--mode", "assemble"])
    same_report = same_request.build_report(
        same_args,
        stage_reports=state.same_request_stage_reports(),
        coordinator_report=state.coordinator_report(),
        cleanup_report=bridge.cleanup_report(cleaned=True),
    )
    assert same_report["same_request_decode_verified"] is True
    assert same_request_check.validate_report(same_report, require_verified=True) == []


def test_state_rejects_nonfinal_submit_without_private_activation() -> None:
    state = bridge.Glm52CoordinatorState(
        stage_specs=_stage_specs(),
        coordinator_request_id_hash=_hash("request"),
    )
    task = state.claim(miner_id="cuda-worker", stage_id=0)["task"]
    payload = _submit_payload(task)
    payload.pop("activation")

    result = state.submit(payload)

    assert result["accepted"] is False
    assert result["reason"] == "activation_missing"
    assert state.ready() is False


def test_state_routes_by_layer_order_not_stage_id_order() -> None:
    specs = [
        {
            "stage_id": 30,
            "stage_count": 3,
            "provider": "kaggle_jax_tpu",
            "stage_layer_range": [2, 4],
            "compatible_weight_repo": bridge.COMPATIBLE_WEIGHT_REPO,
        },
        {
            "stage_id": 20,
            "stage_count": 3,
            "provider": "kaggle_cpu",
            "stage_layer_range": [4, 6],
            "compatible_weight_repo": bridge.COMPATIBLE_WEIGHT_REPO,
        },
        {
            "stage_id": 10,
            "stage_count": 3,
            "provider": "kaggle_cuda",
            "stage_layer_range": [0, 2],
            "compatible_weight_repo": bridge.COMPATIBLE_WEIGHT_REPO,
        },
    ]
    state = bridge.Glm52CoordinatorState(
        stage_specs=specs,
        coordinator_request_id_hash=_hash("request"),
    )

    first = state.claim(miner_id="cuda", stage_id=10)["task"]
    assert first["sequence_index"] == 0
    assert first["next_stage_id"] == 30
    assert first["is_final_stage"] is False
    assert state.submit(_submit_payload(first))["accepted"] is True

    second = state.claim(miner_id="tpu", stage_id=30)["task"]
    assert second["sequence_index"] == 1
    assert second["next_stage_id"] == 20
    assert second["is_final_stage"] is False
    assert state.submit(_submit_payload(second))["accepted"] is True

    final = state.claim(miner_id="cpu", stage_id=20)["task"]
    assert final["sequence_index"] == 2
    assert final["next_stage_id"] is None
    assert final["is_final_stage"] is True
    assert state.submit(_submit_payload(final, final=True))["ready"] is True
    assert state.public_status()["stage_order"] == [10, 30, 20]
    assert [item["stage_id"] for item in state.same_request_stage_reports()] == [10, 30, 20]


def test_contract_artifact_is_public_safe_and_not_success() -> None:
    out = _tmp_dir()
    request_hash = _hash("contract-request")
    args = bridge.parse_args([
        "--output-dir",
        str(out),
        "--stage-count",
        "39",
        "--coordinator-request-id-hash",
        request_hash,
    ])

    report = bridge.build_contract_report(args)
    bridge.write_json(out / "glm52_kaggle_coordinator_decode_bridge_probe.json", report)

    assert report["coordinator_bridge_contract_ready"] is True
    assert report["same_request_decode_verified"] is False
    assert report["live_run_performed"] is False
    assert report["coordinator_request_id_hash"] == request_hash
    assert "glm52_live_kaggle_same_request_not_run" in report["blockers"]
    assert bridge_check.validate_report(report, require_contract=True) == []


def test_checker_rejects_contract_overclaim() -> None:
    args = bridge.parse_args([])
    report = bridge.build_contract_report(args)
    report["same_request_decode_verified"] = True

    errors = bridge_check.validate_report(report, require_contract=True)

    assert "contract_artifact_overclaims_same_request_success" in errors
