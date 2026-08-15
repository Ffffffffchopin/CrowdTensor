"""Small model-neutral text preparation helpers used by Campaign imports."""

from __future__ import annotations

from typing import Any


def tokenize_fixed_sequences(
    texts: list[Any],
    tokenizer: Any,
    *,
    sequence_length: int,
    sequence_count: int,
) -> tuple[list[list[int]], list[int]]:
    """Tokenize a bounded prefix into deterministic fixed-length sequences."""

    length = int(sequence_length)
    count = int(sequence_count)
    if length <= 0 or count <= 0:
        raise ValueError("fixed_sequence_shape_invalid")
    required = length * count
    tokens: list[int] = []
    row_indexes: list[int] = []
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        raise ValueError("tokenizer_eos_token_required")
    for index, value in enumerate(texts):
        text = str(value or "").strip()
        if not text:
            continue
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if not encoded:
            continue
        row_indexes.append(index)
        tokens.extend(int(token) for token in encoded)
        tokens.append(int(eos))
        if len(tokens) >= required:
            break
    if len(tokens) < required:
        raise RuntimeError("text_split_did_not_provide_enough_fixed_tokens")
    rows = [
        tokens[offset : offset + length]
        for offset in range(0, required, length)
    ]
    return rows, row_indexes
