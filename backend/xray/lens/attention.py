"""Incremental attention capture and heuristic head classification.

The engine feeds one forward pass per text delta (possibly several tokens) with
``output_attentions=True`` and a KV cache.  This module turns the resulting
attention tensors into the ``Probe.attention`` payload (apps/desktop/src/types.ts)
and classifies each head of the two focus layers.

All taxonomy labels are heuristics from ordinary statistics (first-column mass,
entropy, distance to punctuation, entity repetition, row argmax).  They are
documented as heuristics in the UI, never as ground truth.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .common import (
    HEAD_TYPES,
    entropy,
    is_punctuation,
    looks_like_entity,
    normalize_token,
    round_floats,
)


def forward_with_attention(model: Any, input_ids: Any, past_key_values: Any = None) -> Any:
    """Single incremental forward with eager attention and hidden states.

    Returns the transformers ``ModelOutput``.  Kept here so the behaviour
    (eager attentions via ``output_attentions=True``) lives next to the code
    that consumes the tensors.
    """
    return model(
        input_ids,
        past_key_values=past_key_values,
        output_attentions=True,
        output_hidden_states=True,
        use_cache=True,
        return_dict=True,
    )


def window_attention_row(
    row: np.ndarray, max_context: int
) -> tuple[np.ndarray, float]:
    """Keep the most recent ``max_context`` columns.

    Dropped mass is folded into the first kept column so each row still sums to
    one; the UI shows the retained context window only.
    """
    arr = np.asarray(row, dtype=np.float32).reshape(-1)
    if arr.size <= max_context:
        return arr, 0.0
    dropped = arr[: arr.size - max_context]
    kept = arr[arr.size - max_context :].copy()
    folded = float(dropped.sum())
    kept[0] += folded
    return kept, folded


def _probability_row(row: np.ndarray) -> np.ndarray:
    arr = np.asarray(row, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(arr.sum())
    if total > 1e-6:
        arr = arr / total
    return arr.astype(np.float32)


def _nearest_punctuation_distance(tokens: Sequence[str], index: int) -> int:
    best = 10**6
    for offset in range(1, len(tokens)):
        for candidate in (index - offset, index + offset):
            if 0 <= candidate < len(tokens) and is_punctuation(tokens[candidate]):
                best = min(best, offset)
    return best


def classify_head(
    row: Sequence[float],
    tokens: Sequence[str],
    current_index: int | None = None,
) -> str:
    """Classify one attention row into one of the six documented head types."""
    probs = _probability_row(np.asarray(row, dtype=np.float32))
    n = int(probs.size)
    if n == 0:
        return "diffuse"
    index = n - 1 if current_index is None else int(current_index)
    index = max(0, min(index, n - 1))

    ent = entropy(probs)
    max_ent = math.log(n) if n > 1 else 1.0
    peak = float(probs.max())
    argmax = int(probs.argmax())
    first_mass = float(probs[0]) if n else 0.0
    tail = probs[max(0, index - 3) : index + 1]
    local_mass = float(tail.sum())
    prev_mass = float(probs[index - 1]) if index >= 1 else 0.0
    punct_mask = np.asarray([is_punctuation(token) for token in tokens], dtype=bool)
    if punct_mask.size != n:
        punct_mask = np.zeros(n, dtype=bool)
    punct_mass = float(probs[punct_mask].sum()) if punct_mask.any() else 0.0
    entity_mask = np.asarray([looks_like_entity(token) for token in tokens], dtype=bool)
    if entity_mask.size != n:
        entity_mask = np.zeros(n, dtype=bool)
    entity_mass = float(probs[entity_mask].sum()) if entity_mask.any() else 0.0
    current_text = normalize_token(tokens[index]) if tokens else ""
    argmax_text = normalize_token(tokens[argmax]) if argmax < len(tokens) else ""
    repeated = bool(current_text) and current_text.lower() == argmax_text.lower() and argmax < index
    punct_distance = _nearest_punctuation_distance(tokens, index)

    # 1. Previous-token heads: mass on the immediately preceding token.
    if prev_mass >= 0.35 or (argmax == index - 1 and prev_mass >= 0.22):
        return "previous-token"
    # 2. Attention sinks / first-column dominance.
    if first_mass >= 0.35 and n >= 4:
        return "global"
    # 3. Syntactic heads: punctuation (near the current position).
    if punct_mass >= 0.30 and punct_distance <= 3:
        return "syntactic"
    if punct_mask[argmax] and peak >= 0.25:
        return "syntactic"
    # 4. Entity heads: repeat the current entity or collect digit/CJK tokens.
    if repeated and peak >= 0.18:
        return "entity"
    if entity_mass >= 0.35 and looks_like_entity(tokens[argmax] if argmax < len(tokens) else ""):
        return "entity"
    # 5. Diffuse: nearly flat across a long context.
    if ent >= 0.88 * max_ent and n >= 6:
        return "diffuse"
    # 6. Local window heads.
    if local_mass >= 0.55 or (peak >= 0.3 and (index - argmax) <= 3):
        return "local"
    if argmax <= 1 and n >= 6:
        return "global"
    return "local"


def build_attention_payload(
    attentions: Sequence[Any],
    position: int,
    kv_len: int,
    layer_indices: Sequence[int],
    focus_layers: Sequence[int],
    head_count: int,
    window_tokens: Sequence[str],
    current_index: int,
    max_context: int = 256,
) -> dict:
    """Build the ``Probe.attention`` dict for one new token.

    ``attentions`` is the per-layer tuple returned by transformers; each tensor
    has shape ``[1, heads, q_len, kv_len]``.  ``position`` is the query row for
    this token and ``kv_len`` its visible context length (causal mask).
    """
    focus = {int(layer) for layer in focus_layers}
    selected: dict[str, dict] = {}
    available: list[int] = []
    token_count = len(window_tokens)

    for layer in layer_indices:
        try:
            tensor = attentions[int(layer)]
        except (IndexError, TypeError):
            continue
        row_all = tensor[0, :, int(position), : int(kv_len)]
        row_all = row_all.detach().float().cpu().numpy()
        mean_row = _probability_row(row_all.mean(axis=0))
        mean_window, _ = window_attention_row(mean_row, max_context)
        if mean_window.size != token_count:
            # Align defensively (window tokens are already truncated by caller).
            mean_window = _fit_length(mean_window, token_count)
        available.append(int(layer))
        entry: dict = {"layer": int(layer), "mean": round_floats(mean_window)}
        if int(layer) in focus:
            rows = row_all[:, : int(kv_len)]
            heads: list[list[float]] = []
            head_types: list[str] = []
            for head in range(rows.shape[0]):
                row = _probability_row(rows[head])
                window, _ = window_attention_row(row, max_context)
                window = _fit_length(window, token_count)
                heads.append(round_floats(window))
                head_types.append(classify_head(window, window_tokens, current_index))
            entry["heads"] = heads
            entry["head_types"] = head_types
        selected[str(int(layer))] = entry

    return {
        "layers": available or list(layer_indices),
        "selected": selected,
        "head_count": int(head_count),
    }


def _fit_length(values: np.ndarray, length: int) -> np.ndarray:
    """Pad or truncate a probability row to ``length`` preserving the sum."""
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == length:
        return arr
    if arr.size == 0:
        return np.full(length, 1.0 / max(length, 1), dtype=np.float32)
    if arr.size > length:
        kept, _ = window_attention_row(arr, length)
        return kept
    padded = np.zeros(length, dtype=np.float32)
    padded[: arr.size] = arr
    padded[-1] += 1.0 - float(padded.sum())
    return _probability_row(padded)


def head_type_counts(attention_payload: dict) -> dict[str, int]:
    counts = {name: 0 for name in HEAD_TYPES}
    for entry in attention_payload.get("selected", {}).values():
        for head_type in entry.get("head_types", []) or []:
            if head_type in counts:
                counts[head_type] += 1
    return counts
