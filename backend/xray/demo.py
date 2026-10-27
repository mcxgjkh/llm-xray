"""Offline demo stream.

The demo mode is the development and onboarding path: it needs no LM Studio, no
lens model and no GPU, yet produces a believable token stream (top-k
distribution, entropy, timing, surrogate probes and a projection) so the whole
UI can be exercised end to end.

The stream is deliberately *incremental*: one token per tick with a 35–90 ms
gap, never one giant string.  It is cancellable through :meth:`DemoStream.stop`
or by cancelling the consuming task.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from typing import Any, AsyncIterator, Iterable, Sequence

from .config import DEFAULT_LENS_MODEL
from .sampling import SamplingParams, entropy_of

#: A short Chinese explanation of the residual stream metaphor.
DEMO_TOKENS: tuple[str, ...] = (
    "残", "差", "流", "像", "一条", "河", "：", "每层", "注意力", "与", " MLP", "在",
    "主", "河道", "旁", "汇入", "支流", "，", "不会", "冲掉", "原来", "的", "水流", "。",
    "层", "归一化", "像", "河床", "，", "约束", "水位", "；", "跳", "连接", "像", "长堤",
    "，", "把", "早期", "信息", "送到", "下游", "。", "第", "24", "层", "里", "仍",
    "能", "听见", "第", "1", "层", "的", "水声", "。",
)

DEMO_PROMPT = "为什么说残差流像一条河？"

#: Plausible alternative tokens for the synthetic candidate lists.
_ALT_TOKENS = (
    "是", "的", "在", "一个", "表示", "信息", "河流", "支流", "网络", "层",
    "水流", "特征", "上下文", "残差", "注意力", "模型", "语言", "数据", "结构", "连接",
    "像", "会", "可以", "因此", "但", "也", "更", "将", "被", "从",
)

_HEAD_TYPES = ("previous-token", "local", "syntactic", "global")
_FEATURE_LABELS = (
    ("水 / river", ("水", "河", "流")),
    ("层级 / layer", ("层", "第", "深度")),
    ("连接 / connect", ("连接", "跳", "汇入")),
    ("上下文 / context", ("上下文", "信息", "早期")),
)
_RESIDUAL_LAYERS = (0, 6, 12, 18, 23)
_ATTENTION_LAYERS = (12, 23)


def _resolve_lens_demo():
    """Try to use the lens engine's synthetic fixtures when available."""
    try:  # pragma: no cover - depends on the optional lens package
        from .lens.demo import synthetic_probe, synthetic_projection

        return synthetic_probe, synthetic_projection
    except Exception:
        return None


def _is_valid_probe(probe: Any) -> bool:
    if not isinstance(probe, dict):
        return False
    attention = probe.get("attention")
    lens = probe.get("lens")
    return isinstance(attention, dict) and isinstance(lens, list) and len(lens) >= 1


def _normalise_row(rng: random.Random, length: int) -> list[float]:
    raw = [rng.random() ** 2 + 0.01 for _ in range(max(1, length))]
    total = sum(raw)
    return [round(value / total, 5) for value in raw]


def _unit_vector(rng: random.Random) -> dict[str, float]:
    x, y, z = rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)
    norm = math.sqrt(x * x + y * y + z * z) or 1.0
    return {"x": round(x / norm, 4), "y": round(y / norm, 4), "z": round(z / norm, 4)}


def _candidate_token(rng: random.Random, *, step: int, chosen: str) -> str:
    for _ in range(8):
        token = _ALT_TOKENS[(rng.randrange(len(_ALT_TOKENS)) + step) % len(_ALT_TOKENS)]
        if token != chosen:
            return token
    return "的"


def _token_options(rng: random.Random, step: int, chosen: str, hidden_mass: float = 0.12) -> tuple[list[dict], float, float, float]:
    """Build a credible raw top-k distribution around ``chosen``.

    Returns ``(topk, chosen_logprob, chosen_prob, entropy)`` where probabilities
    reserve ``hidden_mass`` for the rest of the vocabulary.
    """
    chosen_logprob = -rng.uniform(0.35, 1.9)
    entries: list[dict[str, Any]] = [{"token": chosen, "id": None, "logprob": chosen_logprob}]
    count = rng.randint(6, 11)
    seen = {chosen}
    offset = 0.15
    for _ in range(count):
        token = _candidate_token(rng, step=step, chosen=chosen)
        if token in seen:
            continue
        seen.add(token)
        offset += rng.uniform(0.15, 1.1)
        entries.append({"token": token, "id": None, "logprob": chosen_logprob - offset})
    entries.sort(key=lambda item: item["logprob"], reverse=True)

    exps = [math.exp(item["logprob"]) for item in entries]
    hidden = max(0.0, hidden_mass)
    denominator = sum(exps) + hidden
    probs = [value / denominator for value in exps]
    hidden_prob = hidden / denominator
    entropy = entropy_of(probs + [hidden_prob])

    topk: list[dict[str, Any]] = []
    chosen_prob = probs[0]
    chosen_logprob = entries[0]["logprob"]
    for rank, (item, prob) in enumerate(zip(entries, probs), start=1):
        option = {
            "token": item["token"],
            "id": item["id"],
            "logprob": round(item["logprob"], 4),
            "prob": round(prob, 5),
            "rank": rank,
        }
        topk.append(option)
        if item["token"] == chosen:
            chosen_prob = prob
            chosen_logprob = item["logprob"]
    return topk, round(chosen_logprob, 4), round(chosen_prob, 5), round(entropy, 4)


def _synthetic_vector(step: int, token: str, rng: random.Random) -> dict[str, float]:
    base = _unit_vector(rng)
    wobble = 0.15 * math.sin(step * 0.7)
    return {axis: round(max(-1.0, min(1.0, value + wobble)), 4) for axis, value in base.items()}


# ---------------------------------------------------------------------------
# Built-in probe (used when backend/xray/lens/demo.py is absent)
# ---------------------------------------------------------------------------


def builtin_probe(
    step: int,
    context_tokens: Sequence[str],
    kinds: Sequence[str],
    *,
    lens_model: str = DEFAULT_LENS_MODEL,
    seed: int = 7,
) -> dict[str, Any]:
    rng = random.Random(seed * 1_000_003 + step * 9_973)
    context_len = max(1, len(context_tokens))
    selected: dict[str, Any] = {}
    for layer in _ATTENTION_LAYERS:
        heads = [_normalise_row(rng, context_len) for _ in range(len(_HEAD_TYPES))]
        selected[str(layer)] = {
            "layer": layer,
            "mean": _normalise_row(rng, context_len),
            "heads": heads,
            "head_types": list(_HEAD_TYPES),
        }
    lens_layers = []
    for index, layer in enumerate((6, 12, 23)):
        options: list[dict[str, Any]] = []
        logprob = -rng.uniform(0.2, 1.2)
        for rank in range(1, 6):
            options.append(
                {
                    "token": _ALT_TOKENS[(step + rank * 3 + layer) % len(_ALT_TOKENS)],
                    "id": None,
                    "logprob": round(logprob - 0.9 * (rank - 1), 4),
                    "prob": round(math.exp(-0.9 * (rank - 1)) / 3.2, 5),
                    "rank": rank,
                }
            )
        lens_layers.append(
            {
                "layer": layer,
                "topk": options,
                "tuned": index > 0,
                "agreement": round(0.45 + rng.random() * 0.4, 3),
            }
        )
    residual = []
    for layer in _RESIDUAL_LAYERS:
        residual.append(
            {
                "layer": layer,
                "attn_norm": round(0.4 + rng.random(), 3),
                "mlp_norm": round(0.25 + rng.random(), 3),
                "attn_logit": round(rng.uniform(-0.5, 0.9), 3),
                "mlp_logit": round(rng.uniform(-0.4, 0.7), 3),
            }
        )
    features = []
    for index, (label, tokens) in enumerate(_FEATURE_LABELS):
        features.append(
            {
                "id": 100 + index * 7 + step % 3,
                "label": label,
                "activation": round(rng.random(), 3),
                "tokens": list(tokens),
            }
        )
    used = min(8192, context_len + step + 8)
    bytes_per_token = 131072
    probe = {
        "step": step,
        "model": lens_model,
        "surrogate": True,
        "latency_ms": round(24 + rng.random() * 36, 1),
        "context": {"tokens": list(context_tokens)[-128:], "kinds": list(kinds)[-128:]},
        "attention": {
            "layers": list(_ATTENTION_LAYERS),
            "selected": selected,
            "head_count": len(_HEAD_TYPES),
        },
        "lens": lens_layers,
        "hidden": [round(rng.uniform(-1, 1), 3) for _ in range(16)],
        "residual": residual,
        "features": features,
        "kv": {
            "used": used,
            "capacity": 8192,
            "bytes": used * bytes_per_token,
            "bytes_per_token": bytes_per_token,
            "layers": 32,
            "kv_heads": 8,
            "head_dim": 128,
            "dtype": "q8_0",
            "estimated": True,
        },
        "vector": _unit_vector(rng),
    }
    return probe


def build_probe(
    step: int,
    context_tokens: Sequence[str],
    kinds: Sequence[str],
    *,
    lens_model: str = DEFAULT_LENS_MODEL,
    seed: int = 7,
    synthetic_probe: Any = None,
) -> dict[str, Any]:
    """Use the lens fixtures when installed, otherwise the built-in probe."""
    if synthetic_probe is not None:
        try:
            probe = synthetic_probe(step, list(context_tokens), seed=seed)
            if _is_valid_probe(probe):
                probe = dict(probe)
                probe.setdefault("step", step)
                probe.setdefault("model", lens_model)
                probe.setdefault("surrogate", True)
                if "vector" not in probe or not isinstance(probe.get("vector"), dict):
                    probe["vector"] = builtin_probe(step, context_tokens, kinds, lens_model=lens_model, seed=seed)["vector"]
                return probe
        except Exception:
            pass
    return builtin_probe(step, context_tokens, kinds, lens_model=lens_model, seed=seed)


# ---------------------------------------------------------------------------
# Demo stream
# ---------------------------------------------------------------------------


class DemoStream:
    """Async iterator yielding ``token`` / ``probe`` / ``projection`` events."""

    def __init__(
        self,
        sampling: SamplingParams | dict | None = None,
        *,
        lens_model: str = DEFAULT_LENS_MODEL,
        prompt_text: str = "",
        probe_every: int = 2,
        projection_every: int = 8,
        seed: int = 7,
    ) -> None:
        self.params = sampling if isinstance(sampling, SamplingParams) else SamplingParams.from_dict(sampling)
        self.lens_model = lens_model
        self.prompt_text = prompt_text or DEMO_PROMPT
        self.probe_every = max(1, int(probe_every))
        self.projection_every = max(1, int(projection_every))
        self.seed = int(seed)
        self._stopped = False
        self._rng = random.Random(seed * 7_919)
        lens = _resolve_lens_demo()
        self._synthetic_probe = lens[0] if lens else None
        self._synthetic_projection = lens[1] if lens else None
        self._context_tokens: list[str] = self._prompt_tokens()
        self._context_kinds: list[str] = ["prompt"] * len(self._context_tokens)
        self._points: list[dict[str, Any]] = []
        self._answer = ""

    # -- control --------------------------------------------------------
    def stop(self) -> None:
        self._stopped = True

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def answer(self) -> str:
        return self._answer

    @property
    def token_count(self) -> int:
        return len([kind for kind in self._context_kinds if kind == "generated"])

    # -- prompt context -------------------------------------------------
    def _prompt_tokens(self) -> list[str]:
        pieces: list[str] = ["<|im_start|>", "user"]
        text = self.prompt_text.strip()
        for index in range(0, len(text), 2):
            pieces.append(text[index : index + 2])
        pieces.extend(["<|im_end|>", "<|im_start|>", "assistant"])
        return pieces

    def _projection(self, fitted_at: int) -> dict[str, Any]:
        points = [dict(point) for point in self._points]
        reducer = "PCA" if len(points) >= 8 else "deterministic"
        if self._synthetic_projection is not None:
            try:
                synthetic = self._synthetic_projection(points)
                if isinstance(synthetic, dict) and isinstance(synthetic.get("points"), list):
                    return {
                        "points": synthetic["points"],
                        "reducer": synthetic.get("reducer") or reducer,
                        "fitted_at": int(synthetic.get("fitted_at") or fitted_at),
                    }
                if isinstance(synthetic, list) and synthetic:
                    return {"points": synthetic, "reducer": reducer, "fitted_at": fitted_at}
            except Exception:
                pass
        return {"points": points, "reducer": reducer, "fitted_at": fitted_at}

    # -- iteration ------------------------------------------------------
    async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        started = time.monotonic()
        total = min(len(DEMO_TOKENS), max(1, int(self.params.max_tokens)))
        ttft_ms: float | None = None
        text_len = 0
        for index, delta in enumerate(DEMO_TOKENS[:total]):
            if self._stopped:
                break
            interval = self._rng.uniform(0.035, 0.090)
            await asyncio.sleep(interval)
            if self._stopped:
                break
            now = time.monotonic()
            elapsed_ms = (now - started) * 1000.0
            if ttft_ms is None:
                ttft_ms = elapsed_ms
            text_len += len(delta)
            self._answer += delta
            self._context_tokens.append(delta)
            self._context_kinds.append("generated")
            topk, chosen_logprob, chosen_prob, entropy = _token_options(self._rng, index, delta)
            step = {
                "index": index,
                "delta": delta,
                "kind": "answer",
                "topk": topk,
                "chosen_logprob": chosen_logprob,
                "chosen_prob": chosen_prob,
                "entropy": entropy,
                "at": round(elapsed_ms, 2),
                "ttft_ms": round(ttft_ms, 1),
                "tps": round((index + 1) / max(0.001, now - started), 2),
                "generated": index + 1,
                "text_len": text_len,
                "_vector": _synthetic_vector(index, delta, self._rng),
            }
            yield {"type": "token", "step": step}

            if (index + 1) % self.probe_every == 0 or index == total - 1:
                probe = build_probe(
                    index,
                    self._context_tokens,
                    self._context_kinds,
                    lens_model=self.lens_model,
                    seed=self.seed,
                    synthetic_probe=self._synthetic_probe,
                )
                vector = probe.get("vector") if isinstance(probe.get("vector"), dict) else None
                if vector is None:
                    vector = step["_vector"]
                self._points.append(
                    {
                        "id": f"t{index}",
                        "token": delta,
                        "index": index,
                        "kind": "generated",
                        "x": round(float(vector.get("x", 0.0)), 4),
                        "y": round(float(vector.get("y", 0.0)), 4),
                        "z": round(float(vector.get("z", 0.0)), 4),
                        "step": index,
                    }
                )
                yield {"type": "probe", "step": index, "probe": probe}

            if (index + 1) % self.projection_every == 0:
                projection = self._projection(index)
                yield {
                    "type": "projection",
                    "projection": projection,
                    "points": projection["points"],
                    "reducer": projection["reducer"],
                    "fitted_at": projection["fitted_at"],
                }

        if self._points and len(self._points) < 8:
            # Make sure a projection exists even for very short generations.
            projection = self._projection(len(self._points) - 1)
            yield {
                "type": "projection",
                "projection": projection,
                "points": projection["points"],
                "reducer": projection["reducer"],
                "fitted_at": projection["fitted_at"],
            }


def demo_text() -> str:
    return "".join(DEMO_TOKENS)


def demo_stream(
    sampling: SamplingParams | dict | None = None,
    *,
    lens_model: str = DEFAULT_LENS_MODEL,
    prompt_text: str = "",
) -> DemoStream:
    return DemoStream(sampling, lens_model=lens_model, prompt_text=prompt_text)


__all__ = [
    "DEMO_TOKENS",
    "DEMO_PROMPT",
    "DemoStream",
    "builtin_probe",
    "build_probe",
    "demo_stream",
    "demo_text",
]
