"""Shared helpers for the surrogate lens package.

This module is intentionally torch-free at import time so the backend can import
``xray.lens.api`` even when the optional lens dependencies are not installed.
"""

from __future__ import annotations

import math
import re
import time
from typing import Any, Iterable, Sequence

import numpy as np

#: Heuristic head taxonomy shared with the frontend (apps/desktop/src/types.ts).
HEAD_TYPES = (
    "previous-token",
    "local",
    "syntactic",
    "entity",
    "global",
    "diffuse",
)

#: Punctuation used by the syntactic head heuristic.
PUNCT_CHARS = set(".,;:!?()[]{}<>\"'`~@#$%^&*-_+=/\\|")
NEWLINE_CHARS = set("\n\r")


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def round_floats(values: Iterable[float], ndigits: int = 6) -> list[float]:
    """JSON friendly rounding of an array-like of floats."""
    out: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        if not math.isfinite(number):
            number = 0.0
        out.append(round(number, ndigits))
    return out


def softmax(logits: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr.astype(np.float32)
    arr = arr - np.nanmax(arr)
    exp = np.exp(arr)
    total = float(exp.sum())
    if not math.isfinite(total) or total <= 0.0:
        return np.full(arr.shape, 1.0 / arr.size, dtype=np.float32)
    return (exp / total).astype(np.float32)


def entropy(probs: Sequence[float] | np.ndarray) -> float:
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    p = p[p > 1e-12]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def normalize_token(token: Any) -> str:
    """Normalize a token/piece for comparison heuristics."""
    text = str(token)
    text = text.replace("\u0120", " ").replace("Ġ", " ")
    return text.strip()


def is_punctuation(token: Any) -> bool:
    text = normalize_token(token)
    if not text:
        return False
    return text[-1] in PUNCT_CHARS or text[-1] in NEWLINE_CHARS


def looks_like_entity(token: Any) -> bool:
    """Digit runs, capitalized words and non-ASCII (e.g. CJK) tokens."""
    text = normalize_token(token)
    if not text:
        return False
    if any(ch.isdigit() for ch in text):
        return True
    if len(text) >= 2 and text[0].isupper() and text.isalpha():
        return True
    if any(ord(ch) > 0x2E7F for ch in text):  # CJK / other non-Latin scripts
        return True
    return False


def downsample_vector(vector: Any, dims: int = 16) -> list[float]:
    """Reduce an arbitrary hidden vector to ``dims`` values for the UI strip.

    Uses contiguous mean bins so the result is deterministic and cheap; short
    vectors are linearly resampled instead of zero padded.
    """
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    n = int(arr.size)
    out = np.zeros(dims, dtype=np.float32)
    if n == 0:
        return round_floats(out)
    if n >= dims:
        edges = np.linspace(0, n, dims + 1).astype(int)
        for i in range(dims):
            lo, hi = int(edges[i]), max(int(edges[i + 1]), int(edges[i]) + 1)
            out[i] = float(arr[lo:hi].mean())
    else:
        positions = np.linspace(0, n - 1, dims)
        out = np.interp(positions, np.arange(n), arr).astype(np.float32)
    return round_floats(out)


def token_options_from_logits(
    logits: Any,
    tokenizer: Any,
    top_k: int = 8,
) -> list[dict]:
    """Build the TokenOption list (types.ts) from a 1-D logits vector."""
    arr = np.asarray(logits, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return []
    probs = softmax(arr)
    k = max(1, min(int(top_k), probs.size))
    # argpartition is O(V) and keeps prompt-time latency low.
    candidates = np.argpartition(-probs, k - 1)[:k]
    candidates = candidates[np.argsort(-probs[candidates])]
    options: list[dict] = []
    for rank, token_id in enumerate(candidates, start=1):
        prob = float(probs[token_id])
        options.append(
            {
                "token": token_text(tokenizer, int(token_id)),
                "id": int(token_id),
                "logprob": round(float(math.log(max(prob, 1e-12))), 6),
                "prob": round(prob, 6),
                "rank": rank,
            }
        )
    return options


def token_text(tokenizer: Any, token_id: int) -> str:
    """Best-effort single-token text; never raises.

    Byte-level BPE tokenizers (Qwen, GPT-2, Llama) expose mojibake through
    ``convert_ids_to_tokens``. ``decode`` is the canonical path and returns the
    actual text; only fall back to the raw token when decoding fails.
    """
    try:
        text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
        if isinstance(text, str):
            return text
    except Exception:
        pass
    try:
        pieces = tokenizer.convert_ids_to_tokens([int(token_id)])
        if isinstance(pieces, str):
            return pieces
        if pieces:
            return str(pieces[0])
    except Exception:
        pass
    return str(token_id)


def model_slug(model_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_name or "model")).strip("_")
    return slug[:80] or "model"


def compare_model_names(left: str | None, right: str | None) -> bool:
    """True when two model identifiers look like the same local weights."""
    if not left or not right:
        return False

    def norm(value: str) -> str:
        base = value.replace("\\", "/").rstrip("/").split("/")[-1]
        return re.sub(r"[^a-z0-9]+", "", base.lower())

    a, b = norm(left), norm(right)
    return bool(a and b and a == b)


# ---------------------------------------------------------------------------
# Architecture discovery (transformers 4.x and 5.x friendly, no torch import).
# ---------------------------------------------------------------------------


def _dig(module: Any, dotted: str) -> Any:
    current = module
    for part in dotted.split("."):
        if current is None:
            return None
        try:
            current = getattr(current, part, None)
        except Exception:
            return None
    return current


def _as_layer_list(value: Any) -> list | None:
    """Coerce an nn.ModuleList / tuple / list of layers into a plain list."""
    if value is None or isinstance(value, (str, bytes, dict)):
        return None
    if not hasattr(value, "__len__") or not hasattr(value, "__getitem__"):
        return None
    try:
        if len(value) == 0:
            return None
        return [value[i] for i in range(len(value))]
    except (TypeError, KeyError, IndexError, AttributeError):
        return None


def find_decoder_layers(model: Any) -> list:
    """Locate the decoder layer list across common HF causal-LM architectures."""
    candidates = (
        "model.layers",  # Llama / Qwen2 / Mistral / Gemma (transformers 5.x)
        "transformer.h",  # GPT-2
        "gpt_neox.layers",
        "model.decoder.layers",
        "transformer.blocks",
        "model.transformer.h",
        "language_model.model.layers",
    )
    for path in candidates:
        layers = _as_layer_list(_dig(model, path))
        if layers:
            return layers
    # Last resort: group named modules by their shared decoder-layer prefix.
    try:
        names = [name for name, _ in model.named_modules()]
    except Exception:
        return []
    prefixes: dict[str, list[str]] = {}
    for name in names:
        parts = name.split(".")
        if len(parts) < 2:
            continue
        prefix, leaf = parts[-2], parts[-1]
        if leaf.isdigit():
            prefixes.setdefault(f"{prefix}.", []).append(leaf)
    for marker, indices in prefixes.items():
        if len(indices) < 2:
            continue
        prefix = marker.rstrip(".")
        found = []
        for index in range(len(indices)):
            layers = _as_layer_list(_dig(model, f"{prefix}.{index}"))
            module = layers[0] if layers else None
            if module is None:
                break
            found.append(module)
        if len(found) >= 2:
            return found
    return []


def find_final_norm(model: Any) -> Any:
    for path in (
        "model.norm",
        "transformer.ln_f",
        "gpt_neox.final_layer_norm",
        "model.decoder.final_layer_norm",
        "language_model.model.norm",
    ):
        module = _dig(model, path)
        if module is not None:
            return module
    return None


def find_lm_head(model: Any) -> Any:
    for path in ("lm_head", "embed_out", "language_model.lm_head"):
        module = _dig(model, path)
        if module is not None:
            return module
    try:
        return model.get_output_embeddings()
    except Exception:
        return None


def find_layer_branches(layer: Any) -> tuple[Any, Any]:
    """Return ``(attention, mlp)`` submodules of one decoder layer."""
    attention = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
    mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
    return attention, mlp


def read_config_int(config: Any, name: str, default: int) -> int:
    value = getattr(config, name, None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def norm_weight_of(norm_module: Any) -> np.ndarray | None:
    """Extract LayerNorm/RMSNorm gain as float32 numpy (or None)."""
    if norm_module is None:
        return None
    weight = getattr(norm_module, "weight", None)
    if weight is None:
        return None
    try:
        array = weight.detach().to("cpu", dtype=None).float().numpy()
    except Exception:
        try:
            array = np.asarray(weight, dtype=np.float32)
        except Exception:
            return None
    return np.asarray(array, dtype=np.float32).reshape(-1)


def norm_eps_of(norm_module: Any, default: float = 1e-6) -> float:
    for attr in ("variance_epsilon", "eps", "epsilon"):
        value = getattr(norm_module, attr, None)
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return float(default)
