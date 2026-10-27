"""Pure sampling math shared by the backend and the demo generator.

LM Studio returns the *raw* top-k log-probabilities next to every token.  The
front end re-applies temperature / top-k / top-p / repetition penalty locally
when the user drags a slider, without asking the model again.  This module is
the server-side reference implementation of that maths, used by the demo
generator and by tests.  Nothing here performs I/O; everything is a pure
function so it is trivial to unit test.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

#: How many recent tokens the repetition penalty looks at (see docs/SPEC.md §7).
REPETITION_WINDOW = 64

_LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class SamplingParams:
    """Sampling parameters, matching ``SamplingParams`` in types.ts."""

    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repetition_penalty: float = 1.05
    max_tokens: int = 4096
    seed: int | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None, *, base: "SamplingParams | None" = None) -> "SamplingParams":
        """Tolerant parser: bad / missing values fall back to defaults (or ``base``)."""
        src = dict(base.to_dict() if base is not None else asdict(cls()))
        for key in ("temperature", "top_p", "top_k", "repetition_penalty", "max_tokens", "seed"):
            if data is not None and key in data and data[key] is not None:
                src[key] = data[key]
        try:
            temperature = float(src["temperature"])
        except (TypeError, ValueError):
            temperature = 0.7
        try:
            top_p = float(src["top_p"])
        except (TypeError, ValueError):
            top_p = 0.95
        try:
            top_k = int(src["top_k"])
        except (TypeError, ValueError):
            top_k = 40
        try:
            repetition_penalty = float(src["repetition_penalty"])
        except (TypeError, ValueError):
            repetition_penalty = 1.05
        try:
            max_tokens = int(src["max_tokens"])
        except (TypeError, ValueError):
            max_tokens = 4096
        seed = src.get("seed")
        if seed is not None:
            try:
                seed = int(seed)
            except (TypeError, ValueError):
                seed = None
        return cls(
            temperature=_clamp(temperature, 0.0, 4.0),
            top_p=_clamp(top_p, 0.01, 1.0),
            top_k=int(_clamp(float(top_k), 0.0, 2048.0)) if top_k else 0,
            repetition_penalty=_clamp(repetition_penalty, 0.5, 2.0),
            max_tokens=int(_clamp(float(max_tokens), 1.0, 262144.0)),
            seed=seed,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _as_params(params: SamplingParams | Mapping[str, Any] | None) -> SamplingParams:
    if isinstance(params, SamplingParams):
        return params
    if params is None:
        return SamplingParams()
    return SamplingParams.from_dict(params)


def _decode_bytes(value: Any) -> str | None:
    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover - defensive
            return None
    if isinstance(value, list) and value and all(isinstance(item, int) for item in value[:8]):
        try:
            return bytes(item & 0xFF for item in value).decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover - defensive
            return None
    return None


def normalize_options(topk: Iterable[Any] | None) -> list[dict]:
    """Normalise loosely-shaped top-k entries into ``{"token","id","logprob"}``.

    LM Studio / OpenAI compatible servers report log-probs in a few different
    shapes (dicts, tuples, ``top_logprobs`` mappings, raw ``bytes``).  Bad
    entries are dropped instead of raising: telemetry must never break a live
    stream.
    """
    options: list[dict] = []
    for raw in topk or []:
        token: str | None = None
        token_id = None
        logprob: float | None = None
        if isinstance(raw, Mapping):
            token = raw.get("token", raw.get("text", raw.get("piece")))
            if not isinstance(token, str):
                token = _decode_bytes(raw.get("bytes"))
            logprob = raw.get("logprob", raw.get("log_prob", raw.get("score")))
            token_id = raw.get("id", raw.get("token_id"))
        elif isinstance(raw, (tuple, list)) and len(raw) >= 2:
            token, logprob = raw[0], raw[1]
        else:
            continue
        if token is None:
            continue
        if logprob is None:
            continue
        try:
            token = str(token)
            logprob = float(logprob)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(logprob):
            continue
        if isinstance(token_id, bool) or not isinstance(token_id, (int, float)):
            token_id = None
        options.append({"token": token, "id": token_id, "logprob": logprob})
    # Merge duplicate tokens, keeping the highest log-probability.
    merged: dict[str, dict] = {}
    for option in options:
        previous = merged.get(option["token"])
        if previous is None or option["logprob"] > previous["logprob"]:
            merged[option["token"]] = option
    return list(merged.values())


def _repetition_counts(recent_tokens: Iterable[str] | Mapping[str, int] | None) -> Counter:
    if recent_tokens is None:
        return Counter()
    if isinstance(recent_tokens, Mapping):
        return Counter({str(k): int(v) for k, v in recent_tokens.items() if int(v) > 0})
    tokens = [str(t) for t in recent_tokens if t is not None]
    return Counter(tokens[-REPETITION_WINDOW:])


def _apply_repetition(logit: float, token: str, counts: Counter, penalty: float) -> float:
    count = counts.get(token, 0)
    if not count or penalty <= 0 or abs(penalty - 1.0) < 1e-9:
        return logit
    scaled = penalty ** min(count, 16)
    return logit / scaled if logit > 0 else logit * scaled


def _softmax(logits: Sequence[float]) -> list[float]:
    if not logits:
        return []
    peak = max(logits)
    exps = [math.exp(v - peak) for v in logits]
    total = sum(exps)
    if total <= 0 or not math.isfinite(total):
        return [1.0 / len(logits)] * len(logits)
    return [v / total for v in exps]


def entropy_of(probs: Iterable[float]) -> float:
    """Shannon entropy in nats."""
    total = 0.0
    for p in probs:
        if p > 0:
            total -= p * math.log(p)
    return max(0.0, total)


def apply_sampling(
    topk: Iterable[Any] | None,
    params: SamplingParams | Mapping[str, Any] | None = None,
    recent_tokens: Iterable[str] | Mapping[str, int] | None = None,
    *,
    chosen_token: str | None = None,
) -> dict:
    """Re-apply repetition penalty, temperature, top-k, top-p and softmax.

    Returns a dict with:

    ``topk``                 kept candidates (``token``/``id``/``logprob``/``prob``/``rank``)
    ``probabilities``        same probabilities in rank order
    ``entropy``              entropy of the *kept + renormalised* distribution, nats
    ``chosen_token``         echo of ``chosen_token``
    ``chosen_logprob``       adjusted log-probability of the chosen token, if kept
    ``chosen_prob``          renormalised probability of the chosen token, if kept
    ``chosen_rank``          1-based rank of the chosen token, if kept
    ``kept``                 number of candidates after truncation
    """
    settings = _as_params(params)
    options = normalize_options(topk)
    result: dict = {
        "topk": [],
        "probabilities": [],
        "entropy": None,
        "chosen_token": chosen_token,
        "chosen_logprob": None,
        "chosen_prob": None,
        "chosen_rank": None,
        "kept": 0,
        "temperature": settings.temperature,
        "repetition_penalty": settings.repetition_penalty,
    }
    if not options:
        return result

    counts = _repetition_counts(recent_tokens)
    adjusted: list[dict] = []
    for option in options:
        logit = _apply_repetition(option["logprob"], option["token"], counts, settings.repetition_penalty)
        adjusted.append({**option, "logit": logit})

    # Temperature.  temperature == 0 means greedy decoding (deterministic).
    if settings.temperature <= 0:
        adjusted.sort(key=lambda item: item["logit"], reverse=True)
        best = adjusted[0]
        chosen = None
        if chosen_token is not None:
            for index, item in enumerate(adjusted):
                if item["token"] == chosen_token:
                    chosen = (item, index)
                    break
        result.update(
            topk=[
                {
                    "token": best["token"],
                    "id": best["id"],
                    "logprob": best["logprob"],
                    "logit": best["logit"],
                    "prob": 1.0,
                    "rank": 1,
                }
            ],
            probabilities=[1.0],
            entropy=0.0,
            kept=1,
        )
        if chosen is not None:
            item, index = chosen
            result.update(
                chosen_logprob=item["logprob"] if index == 0 else None,
                chosen_prob=1.0 if index == 0 else 0.0,
                chosen_rank=index + 1,
            )
        return result

    for item in adjusted:
        item["logit"] = item["logit"] / settings.temperature
    adjusted.sort(key=lambda item: item["logit"], reverse=True)

    # Top-K truncation.
    if settings.top_k and settings.top_k > 0:
        adjusted = adjusted[: settings.top_k]

    # Top-P (nucleus) truncation over the temperature-scaled distribution.
    probs = _softmax([item["logit"] for item in adjusted])
    kept: list[dict] = []
    cumulative = 0.0
    for item, prob in zip(adjusted, probs):
        kept.append({**item, "p": prob})
        cumulative += prob
        if cumulative >= settings.top_p:
            break
    if not kept:
        kept = [{**adjusted[0], "p": 1.0}]

    kept_probs = [item["p"] for item in kept]
    total = sum(kept_probs)
    if total > 0:
        kept_probs = [p / total for p in kept_probs]
    else:  # pragma: no cover - extremely defensive
        kept_probs = [1.0 / len(kept)] * len(kept)

    topk_out = []
    for rank, (item, prob) in enumerate(zip(kept, kept_probs), start=1):
        topk_out.append(
            {
                "token": item["token"],
                "id": item["id"],
                "logprob": item["logprob"],
                "logit": item["logit"],
                "prob": prob,
                "rank": rank,
            }
        )

    result.update(
        topk=topk_out,
        probabilities=[round(p, 10) for p in kept_probs],
        entropy=entropy_of(kept_probs),
        kept=len(kept_probs),
    )

    if chosen_token is not None:
        for rank, item in enumerate(topk_out, start=1):
            if item["token"] == chosen_token:
                result.update(
                    chosen_logprob=item["logprob"],
                    chosen_prob=item["prob"],
                    chosen_rank=rank,
                )
                break
    return result


def probs_from_logprobs(topk: Iterable[Any] | None, hidden_mass: float = 0.0) -> tuple[list[dict], float]:
    """Attach approximate probabilities to raw top-k log-probabilities.

    ``hidden_mass`` reserves probability for the rest of the vocabulary so a
    demo stream does not claim that the top-k candidates are the whole
    distribution.  Used by :mod:`xray.demo` only; no I/O.
    """
    options = normalize_options(topk)
    if not options:
        return [], 0.0
    exps = [math.exp(item["logprob"]) for item in options]
    denom = sum(exps) + max(0.0, hidden_mass)
    probs = [value / denom for value in exps]
    enriched = [{**item, "prob": prob} for item, prob in zip(options, probs)]
    entropy = entropy_of([p / sum(probs) for p in probs]) if probs else 0.0
    return enriched, entropy


def approximate_prompt_tokens(text: str) -> int:
    """Cheap token estimate for usage reporting when the server omits it."""
    if not text:
        return 0
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    other = len(text) - ascii_chars
    return max(1, int(ascii_chars / 4 + other * 0.7))
