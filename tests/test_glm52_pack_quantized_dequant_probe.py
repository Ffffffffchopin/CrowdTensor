from __future__ import annotations

import struct

import torch

from scripts import glm52_pack_quantized_dequant_check as check
from scripts import glm52_pack_quantized_dequant_probe as probe


def _hash(char: str) -> str:
    return "sha256:" + char * 64


def test_unpack_from_int32_matches_pack_order_for_int4() -> None:
    values = torch.tensor([[-8, -7, -1, 0, 1, 2, 6, 7]], dtype=torch.int8)
    packed = torch.tensor([[0]], dtype=torch.int32)
    for index, value in enumerate((values + 8).to(torch.int32)[0]):
        packed[0, 0] |= value << (4 * index)

    unpacked = probe.unpack_from_int32(packed, 4, (1, 8), packed_dim=1)

    assert torch.equal(unpacked, values)


def test_dequantize_group_slice_uses_unpacked_zero_points_and_scale() -> None:
    # Two rows, one group, four int4 values per row. Only the first packed int32
    # lanes are used by the requested shape.
    packed = torch.zeros((2, 1), dtype=torch.int32)
    raw = torch.tensor([[8, 9, 10, 11], [7, 6, 5, 4]], dtype=torch.int32)
    for row in range(2):
        for col in range(4):
            packed[row, 0] |= raw[row, col] << (4 * col)
    zp_raw = torch.tensor([[8], [7]], dtype=torch.int32)
    zp = torch.zeros((1, 1), dtype=torch.int32)
    zp[0, 0] = int(zp_raw[0, 0] | (zp_raw[1, 0] << 4))
    scale = torch.tensor([[0.5], [0.25]], dtype=torch.float32)
    shape = torch.tensor([2, 4], dtype=torch.int64)

    q, unpacked_zp, dequant = probe.dequantize_group_slice(
        packed=packed,
        scale=scale,
        zero_point=zp,
        weight_shape=shape,
        row_count=2,
        group_count=1,
        num_bits=4,
    )

    assert q.tolist() == [[0, 1, 2, 3], [-1, -2, -3, -4]]
    assert unpacked_zp.tolist() == [[0], [-1]]
    assert torch.allclose(dequant, torch.tensor([[0.0, 0.5, 1.0, 1.5], [0.0, -0.25, -0.5, -0.75]]))


def test_checker_accepts_dequant_slice_without_stage_decode_overclaim() -> None:
    report = {
        "schema": probe.SCHEMA,
        "public_artifact_safe": True,
        "model_id": probe.MODEL_ID,
        "model_type": "glm_moe_dsa",
        "quantization_format": "pack-quantized",
        "pack_quantized_group_loaded": True,
        "pack_quantized_dequant_verified": True,
        "pack_quantized_linear_slice_verified": True,
        "stage_decode_verified": False,
        "q_unpacked_hash": _hash("a"),
        "zero_point_unpacked_hash": _hash("b"),
        "dequant_slice_hash": _hash("c"),
        "linear_slice_hash": _hash("d"),
        "dequant_slice_shape": [4, 64],
        "linear_slice_shape": [4],
        "row_count": 4,
        "group_count": 2,
        "completion_boundary": {
            "dequant_slice_is_not_full_layer": True,
            "linear_slice_is_not_stage_decode": True,
            "weight_values_not_public": True,
            "requires_full_projection_runtime": True,
            "requires_transformer_block_runtime": True,
            "requires_stage_decode_verified": True,
        },
        "safety": {
            "public_artifact_safe": True,
            "credentials_public": False,
            "cookies_public": False,
            "signed_url_public": False,
            "raw_prompt_public": False,
            "raw_generated_text_public": False,
            "generated_token_ids_public": False,
            "activation_public": False,
            "hidden_state_public": False,
            "logits_public": False,
            "kv_cache_public": False,
            "weight_tensor_values_public": False,
            "safetensors_header_payload_public": False,
        },
    }

    assert check.validate_report(report, require_verified=True) == []


def test_fetch_hf_json_retries_empty_json_payload(monkeypatch) -> None:
    calls: list[int] = []

    def fake_fetch(*args, **kwargs) -> bytes:
        calls.append(1)
        return b"" if len(calls) == 1 else b'{"ok": true}'

    monkeypatch.setattr(probe, "fetch_url_bytes", fake_fetch)
    monkeypatch.setattr(probe.time, "sleep", lambda _seconds: None)

    assert probe.fetch_hf_json("repo", "config.json", timeout_seconds=1) == {"ok": True}
    assert len(calls) == 2


def test_fetch_url_bytes_uses_configurable_empty_response_retries(monkeypatch) -> None:
    calls: list[int] = []

    class FakeResponse:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def read(self, *args) -> bytes:
            return self.payload

    def fake_urlopen(*args, **kwargs):
        calls.append(1)
        return FakeResponse(b"" if len(calls) < 4 else b"ok")

    monkeypatch.setenv("CT_GLM52_HF_FETCH_ATTEMPTS", "5")
    monkeypatch.setattr(probe.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(probe.time, "sleep", lambda _seconds: None)

    assert probe.fetch_url_bytes("https://example.test/model", timeout_seconds=1) == b"ok"
    assert len(calls) == 4


def test_safetensors_header_loader_retries_short_header_payload(monkeypatch) -> None:
    header_calls = 0

    def fake_range(repo, filename, start, end, *, timeout_seconds, max_bytes) -> bytes:
        nonlocal header_calls
        if start == 0 and end == 7:
            return struct.pack("<Q", 2)
        header_calls += 1
        return b"" if header_calls == 1 else b"{}"

    monkeypatch.setattr(probe, "read_hf_range", fake_range)
    monkeypatch.setattr(probe.time, "sleep", lambda _seconds: None)

    header_len, header = probe.load_safetensors_header_with_len(
        "repo",
        "model.safetensors",
        timeout_seconds=1,
        max_header_bytes=8,
    )

    assert header_len == 2
    assert header == {}
    assert header_calls == 2
