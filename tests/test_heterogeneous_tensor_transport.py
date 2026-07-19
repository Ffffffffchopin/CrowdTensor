import time

import pytest
import torch

from crowdtensor.heterogeneous_tensor_transport import (
    ChunkedTensorStore,
    TensorTransportError,
    decode_tensor_payload,
    decode_tensor_payload_to_jax,
    deliver_chunks_with_retry,
    encode_tensor_message,
)


def message(*, direction: str = "forward_activation", clock=time.time, ttl=300.0):
    source, target = (0, 1) if direction == "forward_activation" else (1, 0)
    return encode_tensor_message(
        {
            "hidden_states": torch.arange(96, dtype=torch.float32).reshape(2, 3, 16),
        },
        job_id="job-1",
        manifest_hash="sha256:" + "1" * 64,
        global_step=1,
        microbatch_id=0,
        source_stage_id=source,
        target_stage_id=target,
        direction=direction,
        placement_generation=3,
        assignment_token_hash="sha256:" + "2" * 64,
        chunk_bytes=128,
        ttl_seconds=ttl,
        clock=clock,
    )


def test_activation_and_gradient_round_trip_with_dtype_conversion(tmp_path) -> None:
    for direction in ("forward_activation", "backward_gradient"):
        envelope, chunks = message(direction=direction)
        store = ChunkedTensorStore(tmp_path / direction, max_chunk_bytes=128)
        store.begin(envelope, expected_generation=3)
        for index, chunk in reversed(list(enumerate(chunks))):
            store.put_chunk(
                envelope["message_id"], index, chunk, expected_generation=3
            )
        output = store.assemble(
            envelope["message_id"],
            expected_generation=3,
            consumer_id_hash="sha256:" + "3" * 64,
            target_device="cpu",
            target_dtype="float16",
        )
        assert output["hidden_states"].dtype == torch.float16
        assert output["hidden_states"].shape == (2, 3, 16)
        assert store.status(envelope["message_id"])["complete"] is True


def test_chunk_hash_conflict_size_and_generation_fail_closed(tmp_path) -> None:
    envelope, chunks = message()
    store = ChunkedTensorStore(tmp_path, max_chunk_bytes=128)

    with pytest.raises(TensorTransportError, match="stale_generation"):
        store.begin(envelope, expected_generation=2)
    store.begin(envelope, expected_generation=3)
    with pytest.raises(TensorTransportError, match="chunk_length_invalid"):
        store.put_chunk(
            envelope["message_id"], 0, chunks[0][:-1], expected_generation=3
        )
    damaged = bytearray(chunks[0])
    damaged[0] ^= 1
    with pytest.raises(TensorTransportError, match="chunk_hash_mismatch"):
        store.put_chunk(
            envelope["message_id"], 0, bytes(damaged), expected_generation=3
        )
    with pytest.raises(TensorTransportError, match="stale_generation"):
        store.put_chunk(envelope["message_id"], 0, chunks[0], expected_generation=4)


def test_identical_chunk_replay_is_idempotent_and_conflicting_consumer_rejected(
    tmp_path,
) -> None:
    envelope, chunks = message()
    store = ChunkedTensorStore(tmp_path, max_chunk_bytes=128)
    store.begin(envelope, expected_generation=3)
    first = store.put_chunk(
        envelope["message_id"], 0, chunks[0], expected_generation=3
    )
    second = store.put_chunk(
        envelope["message_id"], 0, chunks[0], expected_generation=3
    )
    assert first["idempotent_replay"] is False
    assert second["idempotent_replay"] is True
    for index, chunk in list(enumerate(chunks))[1:]:
        store.put_chunk(envelope["message_id"], index, chunk, expected_generation=3)
    store.assemble(
        envelope["message_id"],
        expected_generation=3,
        consumer_id_hash="sha256:" + "4" * 64,
    )
    with pytest.raises(TensorTransportError, match="already_consumed"):
        store.assemble(
            envelope["message_id"],
            expected_generation=3,
            consumer_id_hash="sha256:" + "5" * 64,
        )


def test_non_finite_pickle_and_corrupted_payload_are_rejected(tmp_path) -> None:
    with pytest.raises(TensorTransportError, match="non_finite"):
        encode_tensor_message(
            {"gradient": torch.tensor([float("nan")])},
            job_id="job",
            manifest_hash="sha256:" + "1" * 64,
            global_step=1,
            microbatch_id=0,
            source_stage_id=1,
            target_stage_id=0,
            direction="backward_gradient",
            placement_generation=1,
            assignment_token_hash="sha256:" + "2" * 64,
        )
    envelope, chunks = message()
    payload = b"".join(chunks)
    corrupted = payload[:-1] + bytes([payload[-1] ^ 1])
    with pytest.raises(TensorTransportError, match="payload_hash_mismatch"):
        decode_tensor_payload(corrupted, envelope)
    with pytest.raises(TensorTransportError):
        decode_tensor_payload(b"\x80\x04pickle", envelope)


def test_expiry_timeout_and_cleanup_are_bounded(tmp_path) -> None:
    current = [100.0]
    envelope, _chunks = message(clock=lambda: current[0], ttl=1.0)
    store = ChunkedTensorStore(
        tmp_path, max_chunk_bytes=128, clock=lambda: current[0]
    )
    store.begin(envelope, expected_generation=3)
    with pytest.raises(TimeoutError, match="wait_timeout"):
        store.wait_for_complete(
            envelope["message_id"],
            expected_generation=3,
            timeout=0.02,
            poll_interval=0.005,
        )
    current[0] = 102.0
    with pytest.raises(TensorTransportError, match="message_expired"):
        store.assemble(envelope["message_id"], expected_generation=3)
    assert store.cleanup_expired()["expired_messages_removed"] == 1


def test_delivery_uses_finite_retry_and_stops_at_limit() -> None:
    envelope, chunks = message()
    calls = {}

    def flaky(_envelope, index, _chunk):
        calls[index] = calls.get(index, 0) + 1
        if calls[index] < 2:
            raise TimeoutError("temporary")

    report = deliver_chunks_with_retry(
        envelope, chunks, flaky, sleep=lambda _seconds: None
    )
    assert report["delivery_complete"] is True
    assert all(item["attempt_count"] == 2 for item in report["attempts"])

    def failed(_envelope, _index, _chunk):
        raise ConnectionError("down")

    with pytest.raises(TensorTransportError, match="retry_limit_exceeded"):
        deliver_chunks_with_retry(
            envelope, chunks, failed, sleep=lambda _seconds: None
        )


def test_lookup_index_rebuilds_and_avoids_legacy_directory_scan(tmp_path) -> None:
    root = tmp_path / "indexed"
    store = ChunkedTensorStore(root)
    envelopes = []
    for step in range(1, 31):
        envelope, _chunks = encode_tensor_message(
            {"hidden_states": torch.ones((1, 2, 8), dtype=torch.float32)},
            job_id="indexed-job",
            manifest_hash="sha256:" + "1" * 64,
            global_step=step,
            microbatch_id=0,
            source_stage_id=0,
            target_stage_id=1,
            direction="forward_activation",
            placement_generation=1,
            assignment_token_hash="sha256:" + "2" * 64,
        )
        store.begin(envelope, expected_generation=1)
        envelopes.append(envelope)
    legacy = store.find_message(
        job_id="indexed-job",
        global_step=30,
        microbatch_id=0,
        source_stage_id=0,
        target_stage_id=1,
        direction="forward_activation",
        placement_generation=1,
        use_index=False,
    )
    indexed = store.find_message(
        job_id="indexed-job",
        global_step=30,
        microbatch_id=0,
        source_stage_id=0,
        target_stage_id=1,
        direction="forward_activation",
        placement_generation=1,
        use_index=True,
    )
    report = store.lookup_performance_report()
    reopened = ChunkedTensorStore(root)
    rebuilt = reopened.find_message(
        job_id="indexed-job",
        global_step=30,
        microbatch_id=0,
        source_stage_id=0,
        target_stage_id=1,
        direction="forward_activation",
        placement_generation=1,
    )

    assert legacy["message_id"] == envelopes[-1]["message_id"]
    assert indexed["message_id"] == envelopes[-1]["message_id"]
    assert report["legacy_scanned_directory_count"] == 30
    assert report["indexed_lookup_count"] == 1
    assert rebuilt["message_id"] == envelopes[-1]["message_id"]
    assert reopened.lookup_performance_report()["indexed_key_count"] == 30


def test_jax_array_round_trip_uses_same_safetensors_contract() -> None:
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import numpy as np

    source = jnp.arange(96, dtype=jnp.bfloat16).reshape(2, 3, 16)
    envelope, chunks = encode_tensor_message(
        {"hidden_states": source},
        job_id="job-jax",
        manifest_hash="sha256:" + "1" * 64,
        global_step=1,
        microbatch_id=0,
        source_stage_id=1,
        target_stage_id=2,
        direction="forward_activation",
        placement_generation=3,
        assignment_token_hash="sha256:" + "2" * 64,
        chunk_bytes=128,
    )
    torch_decoded = decode_tensor_payload(
        b"".join(chunks), envelope, target_dtype="float32"
    )
    jax_decoded = decode_tensor_payload_to_jax(
        b"".join(chunks), envelope, target_dtype="bfloat16"
    )

    assert envelope["tensor_specs"][0]["dtype"] == "bfloat16"
    assert torch_decoded["hidden_states"].dtype == torch.float32
    assert str(jax_decoded["hidden_states"].dtype) == "bfloat16"
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(jax_decoded["hidden_states"]), dtype=np.float32),
        np.asarray(jax.device_get(source), dtype=np.float32),
    )
