"""Deterministic synthetic telemetry for demo mode (no model required).

Every value is derived from the token strings and ``step`` with a seeded hash,
so the same input always produces the same probe.  The data is *plausible*:
attention rows sum to 1, probabilities sum close to 1, layers/heads/types use
the same shapes as the real surrogate engine, and labels mention the actual
token text so the UI panels look alive in demo mode.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence

import numpy as np

from .common import HEAD_TYPES, PUNCT_CHARS, entropy, round_floats, softmax

DEMO_MODEL = "demo/synthetic-surrogate"
DEMO_LAYERS = (3, 7, 10)
DEMO_HEADS = 8
DEMO_HEAD_COUNT = 8
DEMO_VOCAB = 64
MAX_CONTEXT = 256


def _stable_seed(*parts: Any) -> int:
    key = "|".join(str(part) for part in parts).encode("utf-8", errors="ignore")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**32)


def _rng(*parts: Any) -> np.random.Generator:
    return np.random.default_rng(_stable_seed(*parts))


def _token_pieces(tokens: Sequence[str]) -> list[str]:
    return [str(token) if str(token) else "·" for token in tokens]


def _hash_float(token: str, salt: int) -> float:
    digest = hashlib.blake2b(f"{salt}:{token}".encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64)


def _demo_vocab(tokens: Sequence[str]) -> list[str]:
    base = [piece for piece in _token_pieces(tokens)][-24:]
    extras = [
        " the", " a", " of", " and", " to", " is", " in", " that", " it", " with",
        " you", " we", " this", " as", " for", " on", " be", " are", " can", " not",
        " 紫色", " violet", " 注意", " attention", " layer", " token", " model", " lens",
        " residual", " feature", " head", " context", " next", " answer", " think", " step",
        " first", " second", " because", " therefore",
    ]
    vocab: list[str] = []
    for piece in base + extras:
        if piece not in vocab:
            vocab.append(piece)
    return vocab[:DEMO_VOCAB]


def _demo_token_options(tokens: Sequence[str], rng: np.random.Generator, count: int = 8) -> list[dict]:
    vocab = _demo_vocab(tokens)
    size = len(vocab)
    logits = rng.normal(0.0, 2.6, size=size)
    if tokens:
        current = _stable_seed("chosen", tokens[-1], len(tokens)) % size
        logits[current] += 3.0
    probs = softmax(logits)
    order = np.argsort(-probs)[: max(1, min(count, size))]
    options = []
    for rank, token_id in enumerate(order, start=1):
        prob = float(probs[token_id])
        options.append(
            {
                "token": vocab[int(token_id)],
                "id": int(token_id),
                "logprob": round(float(math.log(max(prob, 1e-12))), 5),
                "prob": round(prob, 5),
                "rank": rank,
            }
        )
    return options


def _attention_profile(
    tokens: Sequence[str],
    head: int,
    layer: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    """Return one attention row over the visible context and its head type."""
    n = len(tokens)
    row = np.zeros(n, dtype=np.float64)
    current = n - 1
    kind = HEAD_TYPES[(head + layer) % len(HEAD_TYPES)]
    previous = max(0, current - 1)
    if kind == "previous-token":
        row[previous] += 0.62
        row[current] += 0.18
        if current >= 2:
            row[current - 2] += 0.08
    elif kind == "local":
        for offset in range(0, min(5, n)):
            row[current - offset] += 0.55 / (offset + 1)
    elif kind == "syntactic":
        punct = [i for i, token in enumerate(tokens) if str(token).strip()[-1:] in PUNCT_CHARS]
        if punct:
            for i in punct[-2:]:
                row[i] += 0.6
        row[current] += 0.2
    elif kind == "entity":
        current_token = str(tokens[current])
        matches = [i for i, token in enumerate(tokens) if str(token) == current_token]
        for i in matches:
            row[i] += 0.55
        if not matches:
            row[current] += 0.5
    elif kind == "global":
        row[0] += 0.5
        row[current] += 0.12
        row[max(0, n // 3)] += 0.15
    else:  # diffuse
        row += rng.uniform(0.6, 1.4, size=n)
    row += rng.uniform(0.0, 0.02, size=n)
    row = np.clip(row, 0.0, None)
    if row.sum() <= 0:
        row = np.ones(n, dtype=np.float64)
    row /= row.sum()
    return row, kind


def _history_mass(tokens: Sequence[str], text: str, position: int) -> float:
    """Fraction of earlier tokens whose text overlaps ``text``; stable across steps."""
    if not text:
        return 0.0
    target = text.strip().lower()
    if not target:
        return 0.0
    hits = 0
    for token in tokens[:position]:
        piece = str(token).strip().lower()
        if piece and (piece in target or target in piece):
            hits += 1
    return hits / max(1, position)


def synthetic_probe(step: int, tokens: list[str], seed: int = 7) -> dict:
    """Create a full ``Probe`` dict (types.ts) without any model.

    Args:
        step: token index in the generation (used when no explicit context index).
        tokens: visible context tokens, prompt + generated, ending at the token
            this probe describes.  At least one token is recommended.
        seed: varies the synthetic distribution deterministically.
    """
    pieces = _token_pieces(tokens) or ["<empty>"]
    pieces = pieces[-MAX_CONTEXT:]
    step = int(step)
    n = len(pieces)
    current_index = n - 1
    current_token = pieces[current_index]
    rng = _rng("probe", seed, step, current_token, n)

    # --- attention: two focus layers with per-head rows -------------------
    focus_layers = (3, 10)
    selected: dict[str, dict] = {}
    for layer in (0, 3, 7, 10):
        heads: list[list[float]] = []
        head_types: list[str] = []
        means = []
        for head in range(DEMO_HEADS):
            row, head_kind = _attention_profile(pieces, head, layer, _rng("head", seed, layer, head, step))
            heads.append(row)
            head_types.append(head_kind)
            means.append(row)
        mean_row = np.mean(np.stack(means), axis=0)
        mean_row /= max(mean_row.sum(), 1e-9)
        entry: dict = {"layer": layer, "mean": round_floats(mean_row)}
        if layer in focus_layers:
            entry["heads"] = [round_floats(row) for row in heads]
            entry["head_types"] = head_types
        selected[str(layer)] = entry

    # --- lens: three layers, per-layer token distribution ----------------
    lens = []
    for layer in DEMO_LAYERS:
        options = _demo_token_options(pieces, _rng("lens", seed, layer, step, current_token))
        agreement = 0.35 + 0.45 * _hash_float(current_token, layer + step)
        tuned = layer != DEMO_LAYERS[0]
        if tuned:
            # Bias tuned top-k towards the actual current token.
            for option in options:
                if option["token"] == current_token and option["rank"] > 1:
                    options.remove(option)
                    options.insert(0, option)
                    for rank, opt in enumerate(options, start=1):
                        opt["rank"] = rank
                    break
        lens.append(
            {
                "layer": layer,
                "topk": options,
                "tuned": tuned,
                "agreement": round(agreement, 3),
            }
        )

    # --- hidden (16 dims) -------------------------------------------------
    hidden = []
    for dim in range(16):
        base = math.sin((dim + 1) * 0.37 + step * 0.11 + _hash_float(current_token, dim) * math.pi)
        hidden.append(base * 0.9)

    # --- residual: one row per demo layer --------------------------------
    residual = []
    for layer in (0, 3, 7, 10):
        local = _rng("residual", seed, layer, step)
        attn_norm = 0.4 + 1.6 * float(local.random())
        mlp_norm = 0.3 + 1.4 * float(local.random())
        overlap = _history_mass(pieces, current_token, current_index)
        residual.append(
            {
                "layer": layer,
                "attn_norm": round(attn_norm, 4),
                "mlp_norm": round(mlp_norm, 4),
                "attn_logit": round((float(local.random()) * 2 - 1) * attn_norm * (0.3 + overlap), 4),
                "mlp_logit": round((float(local.random()) * 2 - 1) * mlp_norm * 0.8, 4),
            }
        )

    # --- sparse features (probe, not SAE) --------------------------------
    vocab = _demo_vocab(pieces)
    features = []
    used_labels: set[str] = set()
    for slot in range(6):
        local = _rng("feature", seed, step, slot)
        primary = pieces[(slot * 7 + slot) % n]
        secondary = vocab[int(local.integers(0, len(vocab)))]
        label = f"{primary} / {secondary}"
        if label in used_labels:
            continue
        used_labels.add(label)
        activation = 0.2 + 0.8 * float(local.random())
        features.append(
            {
                "id": int(100 + (step * 13 + slot * 7) % 900),
                "label": label,
                "activation": round(activation, 4),
                "tokens": [primary, secondary],
            }
        )

    # --- kv / vector ------------------------------------------------------
    used = n
    head_dim = 64
    kv_heads = 2
    kv_layers = 12
    bytes_per_token = kv_layers * kv_heads * head_dim * 2 * 4
    kv = {
        "used": used,
        "capacity": 8192,
        "bytes": used * bytes_per_token,
        "bytes_per_token": bytes_per_token,
    }

    hash_x = _hash_float(current_token, 1) * 2 - 1
    hash_y = _hash_float(current_token, 2) * 2 - 1
    hash_z = _hash_float(current_token, 3) * 2 - 1
    rope = math.sin(n * 0.21 + step * 0.13)
    vector = {"x": round(0.6 * hash_x + 0.4 * rope, 5), "y": round(0.6 * hash_y, 5), "z": round(0.6 * hash_z, 5)}

    # Demo streams call one probe per generated token, so the trailing
    # ``step + 1`` context tokens are the generated ones; the rest is prompt.
    generated_count = min(n, max(0, step + 1))
    context_kinds = ["prompt"] * (n - generated_count) + ["generated"] * generated_count

    return {
        "step": step,
        "model": DEMO_MODEL,
        "surrogate": True,
        "same_weights": False,
        "latency_ms": round(1.0 + 2.5 * _hash_float(current_token, 9), 2),
        "context": {"tokens": pieces, "kinds": context_kinds},
        "attention": {
            "layers": [0, 3, 7, 10],
            "selected": selected,
            "head_count": DEMO_HEAD_COUNT,
        },
        "lens": lens,
        "hidden": [round(value, 5) for value in hidden],
        "residual": residual,
        "features": features,
        "kv": kv,
        "vector": vector,
    }


def synthetic_projection(points: Sequence[dict], seed: int = 7) -> list[dict]:
    """Attach smooth UMAP-looking 3D coordinates to projection points."""
    result: list[dict] = []
    for position, point in enumerate(points):
        token = str(point.get("token", ""))
        index = int(point.get("index", position))
        jitter = _hash_float(token, seed)
        angle = index * 0.41 + jitter * math.tau
        x = 0.72 * math.sin(angle) + 0.18 * math.cos(angle * 2.3)
        y = 0.68 * math.cos(angle * 0.83) + 0.2 * math.sin(angle * 1.7)
        z = 0.6 * math.sin(angle * 0.37 + math.pi / 3) + 0.25 * _hash_float(token, seed + 1) - 0.12
        result.append(
            {
                "id": str(point.get("id") or f"t{index}"),
                "token": token,
                "index": index,
                "kind": point.get("kind", "generated"),
                "step": point.get("step"),
                "x": round(max(-1.0, min(1.0, x)), 5),
                "y": round(max(-1.0, min(1.0, y)), 5),
                "z": round(max(-1.0, min(1.0, z)), 5),
            }
        )
    return result


def synthetic_projection_event(points: Sequence[dict], fitted_at: int | None = None) -> dict:
    """Convenience ``ProjectionEvent`` payload (types.ts)."""
    return {
        "points": synthetic_projection(points),
        "reducer": "UMAP",
        "fitted_at": int(fitted_at if fitted_at is not None else len(points)),
    }


def demo_lens_status() -> dict:
    """Lens status payload for demo mode: nothing loaded, data is synthetic."""
    return {
        "status": "disabled",
        "model": DEMO_MODEL,
        "device": None,
        "surrogate": True,
        "same_weights": False,
        "calibrated_layers": 0,
        "message": "demo 模式：合成遥测，未加载真实模型",
        "threads": 0,
    }


def token_entropy(tokens: Sequence[str]) -> float:
    """Small helper used by tests / demo panels: normalised token entropy."""
    probs = softmax(np.linspace(-1.0, 1.0, max(1, len(tokens))))
    return round(entropy(probs), 5)
