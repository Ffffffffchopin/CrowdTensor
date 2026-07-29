"""Deterministic browser capability work for Volunteer Campaigns.

Browser tasks calibrate an untrusted contributor runtime. They never mutate a
model checkpoint and are deliberately separate from PEFT training work.
"""

from __future__ import annotations

import hashlib
import struct


BROWSER_PROBE_SCHEMA = "crowdtensor_volunteer_browser_probe_v1"
BROWSER_PROBE_RESULT_SCHEMA = "crowdtensor_volunteer_browser_probe_result_v1"
BROWSER_PROBE_VECTOR_LENGTH = 32_768
BROWSER_PROBE_ROUNDS = 64
BROWSER_PROBE_LEASE_SECONDS = 120.0
BROWSER_PROBE_MAX_ACTIVE = 64
BROWSER_PROBE_RESULT_RETENTION_SECONDS = 24 * 60 * 60
BROWSER_PROBE_RUNTIMES = frozenset({"webgpu", "wasm-cpu", "cpu-js"})

_MASK32 = 0xFFFFFFFF
_INDEX_MIX = 0x9E3779B9
_ROUND_ADD = 0x6D2B79F5


def browser_probe_value(seed: int, index: int, rounds: int) -> int:
    """Return one portable u32 result shared by Python, WGSL, and JavaScript."""

    value = (int(seed) ^ ((int(index) * _INDEX_MIX) & _MASK32)) & _MASK32
    for round_index in range(int(rounds)):
        value ^= (value << 13) & _MASK32
        value ^= value >> 17
        value ^= (value << 5) & _MASK32
        value = (value + _ROUND_ADD + round_index + int(index)) & _MASK32
    return value


def browser_probe_digest(*, seed: int, vector_length: int, rounds: int) -> str:
    """Hash the canonical little-endian output without retaining tensor values."""

    length = int(vector_length)
    round_count = int(rounds)
    if length < 1 or length > BROWSER_PROBE_VECTOR_LENGTH:
        raise ValueError("volunteer_browser_probe_vector_length_invalid")
    if round_count < 1 or round_count > BROWSER_PROBE_ROUNDS:
        raise ValueError("volunteer_browser_probe_round_count_invalid")
    digest = hashlib.sha256()
    for index in range(length):
        digest.update(struct.pack("<I", browser_probe_value(seed, index, round_count)))
    return "sha256:" + digest.hexdigest()

