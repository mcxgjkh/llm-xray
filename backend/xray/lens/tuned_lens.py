"""Tuned Lens: per-layer affine translators fitted with closed-form ridge regression.

Logit Lens applies the final norm + ``lm_head`` directly to an intermediate
residual stream.  Tuned Lens learns ``W_l`` / ``b_l`` mapping each layer's
residual into the final-layer residual space, then applies the same final norm
and ``lm_head``.

Target (per SPEC 8.3): the pre-final-norm last-layer residual
(``hidden_states[-2]`` in transformers, i.e. the last decoder layer output).
Calibration corpus: short sentences from ``docs/*.md`` and ``memory/*.md``;
24-48 segments, a few hundred tokens total, so CPU calibration stays quick.
Fitted translators are cached in ``data/cache/tuned_lens_<slug>.pt``.
"""

from __future__ import annotations

import glob
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .common import model_slug, token_options_from_logits

RIDGE = 1.0
DEFAULT_CACHE_DIR = os.path.join("data", "cache")


# ---------------------------------------------------------------------------
# Pure functions (unit-testable without a model).
# ---------------------------------------------------------------------------


def fit_translator(
    hidden_l: Any,
    target: Any,
    ridge: float = RIDGE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form ridge regression ``hidden_l @ W + b ~= target``.

    Args:
        hidden_l: ``[N, D]`` intermediate residual stream.
        target: ``[N, D]`` final-layer residual stream (pre final norm).
        ridge: L2 penalty on W (does not penalize the bias).

    Returns:
        ``(W, b)`` float32 CPU tensors with shapes ``[D, D]`` and ``[D]``.
    """
    x = torch.as_tensor(hidden_l, dtype=torch.float32).reshape(-1, _last_dim(hidden_l))
    y = torch.as_tensor(target, dtype=torch.float32).reshape(x.shape[0], -1)
    if x.shape[0] == 0 or y.shape[0] != x.shape[0]:
        raise ValueError("fit_translator needs matching non-empty [N, D] tensors")
    dim = int(x.shape[1])
    if int(y.shape[1]) != dim:
        raise ValueError(f"target dim {y.shape[1]} != hidden dim {dim}")
    x_mean = x.mean(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)
    xc = x - x_mean
    yc = y - y_mean
    # (X^T X + ridge I) W = X^T Y  -- solve on CPU, deterministic.
    gram = xc.transpose(0, 1) @ xc + float(ridge) * torch.eye(dim, dtype=torch.float32)
    rhs = xc.transpose(0, 1) @ yc
    try:
        w = torch.linalg.solve(gram, rhs)
    except Exception:
        w = torch.linalg.lstsq(gram, rhs).solution
    bias = (y_mean - x_mean @ w).reshape(dim)
    return w.contiguous().float(), bias.contiguous().float()


def apply_translator(hidden_l: Any, w: Any, b: Any) -> torch.Tensor:
    """Apply a fitted translator to ``[N, D]`` or ``[D]`` hidden states."""
    x = torch.as_tensor(hidden_l, dtype=torch.float32)
    weight = torch.as_tensor(w, dtype=torch.float32, device=x.device)
    bias = torch.as_tensor(b, dtype=torch.float32, device=x.device)
    return x @ weight + bias


def _last_dim(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape:
        return int(shape[-1])
    array = np.asarray(value)
    return int(array.shape[-1])


# ---------------------------------------------------------------------------
# Corpus collection.
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    # backend/xray/lens/tuned_lens.py -> repo root
    return Path(__file__).resolve().parents[3]


def _clean_markdown(text: str) -> str:
    lines = []
    in_fence = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not line:
            lines.append("")
            continue
        if line.startswith("#") or line.startswith("|"):
            continue
        if re.match(r"^[-*+]\s+", line):
            line = re.sub(r"^[-*+]\s+", "", line)
        if re.match(r"^\d+\.\s+", line):
            line = re.sub(r"^\d+\.\s+", "", line)
        line = re.sub(r"`([^`]*)`", r"\1", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
        line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
        line = re.sub(r"\s+", " ", line)
        lines.append(line)
    return "\n".join(lines)


def collect_calibration_corpus(
    root: str | Path | None = None,
    max_segments: int = 32,
    min_chars: int = 28,
    max_chars: int = 180,
) -> list[str]:
    """Extract short clean sentences from repository docs and memory files."""
    base = Path(root) if root else _repo_root()
    paths: list[str] = []
    for pattern in ("docs/*.md", "memory/*.md", "README.md", "CHANGELOG.md"):
        paths.extend(sorted(glob.glob(str(base / pattern))))
    segments: list[str] = []
    for path in paths:
        try:
            raw = Path(path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line in _clean_markdown(raw).splitlines():
            for piece in re.split(r"(?<=[。！？!?;；])\s*", line):
                sentence = piece.strip(" \t-")
                if min_chars <= len(sentence) <= max_chars:
                    segments.append(sentence)
                if len(segments) >= max_segments:
                    return segments
    if not segments:
        # Deterministic fallback so calibration still works on a bare checkout.
        segments = [
            "The quick brown fox jumps over the lazy dog near the river.",
            "Attention heads sometimes copy the previous token in a sequence.",
            "A tuned lens projects intermediate residual states into the final layer.",
        ]
    return segments[:max_segments]


# ---------------------------------------------------------------------------
# TunedLens.
# ---------------------------------------------------------------------------


class TunedLens:
    """Owns translators + logit-lens fallback for one loaded surrogate model."""

    def __init__(
        self,
        model_name: str,
        n_layers: int,
        hidden_size: int,
        final_norm: Any,
        lm_head: Any,
        tokenizer: Any,
        device: str = "cpu",
        cache_dir: str | None = None,
        ridge: float = RIDGE,
    ) -> None:
        self.model_name = model_name
        self.n_layers = int(n_layers)
        self.hidden_size = int(hidden_size)
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.tokenizer = tokenizer
        self.device = device
        self.ridge = float(ridge)
        self._lock = threading.RLock()
        self._translators: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._agreements: dict[int, float] = {}
        self.cache_dir = cache_dir or str(_repo_root() / DEFAULT_CACHE_DIR)
        self.cache_path = os.path.join(
            self.cache_dir, f"tuned_lens_{model_slug(model_name)}.pt"
        )
        self.corpus_size = 0
        self.last_error: str | None = None
        self.fitted_at: float | None = None

    # -- layer selection --------------------------------------------------

    def lens_layers(self, count: int = 3) -> list[int]:
        """Evenly spaced decoder layers including the last one (deduplicated)."""
        n = max(1, self.n_layers)
        if n <= count:
            return list(range(n))
        picks = [round((i + 1) * (n - 1) / count) for i in range(count)]
        result: list[int] = []
        for layer in picks:
            layer = max(0, min(n - 1, int(layer)))
            if layer not in result:
                result.append(layer)
        if n - 1 not in result:
            result[-1] = n - 1
        return sorted(set(result))

    @property
    def calibrated_layers(self) -> int:
        with self._lock:
            return len(self._translators)

    def calibrated_layer_indices(self) -> list[int]:
        with self._lock:
            return sorted(self._translators)

    # -- inference --------------------------------------------------------

    def _logits_from_hidden(self, hidden: torch.Tensor) -> np.ndarray:
        with torch.inference_mode():
            value = hidden.to(self.device)
            if value.dim() == 1:
                value = value.unsqueeze(0)
            normalized = self.final_norm(value)
            logits = self.lm_head(normalized)
        return logits.detach().float().cpu().numpy().reshape(-1)

    def predict(
        self,
        hidden_states: Sequence[torch.Tensor],
        layer: int,
        position: int = -1,
        top_k: int = 8,
    ) -> tuple[list[dict], bool, float | None]:
        """Return ``(topk, tuned, agreement)`` for one layer / token position."""
        results = self.predict_many(hidden_states, [int(layer)], position, top_k)
        return results[0]

    def predict_many(
        self,
        hidden_states: Sequence[torch.Tensor],
        layers: Sequence[int],
        position: int = -1,
        top_k: int = 8,
    ) -> list[tuple[list[dict], bool, float | None]]:
        """Batched Logit/Tuned Lens for several layers at one token position.

        Stacking the layers means ``lm_head`` (the largest matrix in a small
        model) is read once per token instead of once per lens layer, which
        matters a lot on CPU.
        """
        layers = [int(max(0, min(int(layer), self.n_layers - 1))) for layer in layers]
        with self._lock:
            translators = {layer: self._translators.get(layer) for layer in layers}
            agreements = {layer: self._agreements.get(layer) for layer in layers}
        with torch.inference_mode():
            rows = torch.stack(
                [
                    hidden_states[layer + 1][0, int(position)].float()
                    for layer in layers
                ]
            )
            if self.device != "cpu":
                rows = rows.to(self.device)
            mapped: list[torch.Tensor] = []
            for index, layer in enumerate(layers):
                translator = translators.get(layer)
                if translator is not None:
                    mapped.append(
                        apply_translator(
                            rows[index].to("cpu"), translator[0], translator[1]
                        ).to(rows.device)
                    )
                else:
                    mapped.append(rows[index])
            stacked = torch.stack(mapped)
            logits = self.lm_head(self.final_norm(stacked))
            logits_np = logits.detach().float().cpu().numpy()
        results: list[tuple[list[dict], bool, float | None]] = []
        for index, layer in enumerate(layers):
            tuned = translators.get(layer) is not None
            agreement = agreements.get(layer)
            results.append(
                (
                    token_options_from_logits(logits_np[index], self.tokenizer, top_k),
                    tuned,
                    round(float(agreement), 4) if agreement is not None else None,
                )
            )
        return results

    # -- calibration ------------------------------------------------------

    def load_cache(self) -> bool:
        """Load a validated translator cache.  Returns True when usable."""
        path = self.cache_path
        if not os.path.isfile(path):
            return False
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                return False
            if int(payload.get("hidden_size", -1)) != self.hidden_size:
                self.last_error = "缓存维度与模型不匹配"
                return False
            if str(payload.get("model", "")) != self.model_name:
                # Same slug is enough; keep going but note the mismatch.
                pass
            layers = payload.get("layers", {})
            translators: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            agreements: dict[int, float] = {}
            for key, item in layers.items():
                layer = int(key)
                if layer < 0 or layer >= self.n_layers:
                    continue
                w = torch.as_tensor(item["W"], dtype=torch.float32)
                b = torch.as_tensor(item["b"], dtype=torch.float32)
                if tuple(w.shape) != (self.hidden_size, self.hidden_size):
                    continue
                if tuple(b.shape) != (self.hidden_size,):
                    continue
                translators[layer] = (w, b)
                agreements[layer] = float(item.get("agreement", 0.0))
            if not translators:
                return False
            with self._lock:
                self._translators = translators
                self._agreements = agreements
            self.fitted_at = float(payload.get("created_at", time.time()))
            return True
        except Exception as exc:
            self.last_error = f"缓存加载失败：{exc}"
            return False

    def save_cache(self) -> bool:
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with self._lock:
                layers = {
                    str(layer): {
                        "W": w.detach().cpu().clone(),
                        "b": b.detach().cpu().clone(),
                        "agreement": round(float(self._agreements.get(layer, 0.0)), 6),
                    }
                    for layer, (w, b) in self._translators.items()
                }
            payload = {
                "format": "llm-xray/tuned-lens/1",
                "model": self.model_name,
                "hidden_size": self.hidden_size,
                "n_layers": self.n_layers,
                "ridge": self.ridge,
                "created_at": time.time(),
                "layers": layers,
            }
            tmp = f"{self.cache_path}.tmp"
            torch.save(payload, tmp)
            os.replace(tmp, self.cache_path)
            return True
        except Exception as exc:
            self.last_error = f"缓存写入失败：{exc}"
            return False

    def calibrate(
        self,
        model: Any,
        corpus: Sequence[str],
        max_seq_tokens: int = 48,
        max_total_tokens: int = 640,
        time_budget_s: float = 20.0,
    ) -> int:
        """Fit translators for every layer and cache them.

        Runs in a background thread from the engine.  Returns the number of
        calibrated layers; on failure it leaves Logit Lens in place.
        """
        started = time.time()
        layers = list(range(self.n_layers))
        xs: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
        targets: list[torch.Tensor] = []
        total_tokens = 0
        used_segments = 0
        token_ids: list[list[int]] = []
        for text in corpus:
            try:
                ids = self._encode(text)
            except Exception:
                continue
            if len(ids) < 4:
                continue
            token_ids.append(ids[:max_seq_tokens])
            total_tokens += len(ids[:max_seq_tokens])
            used_segments += 1
            if total_tokens >= max_total_tokens:
                break
        if not token_ids:
            self.last_error = "校准语料为空"
            return 0

        try:
            with torch.inference_mode():
                for ids in token_ids:
                    if time.time() - started > time_budget_s * 2:
                        break
                    input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
                    out = model(
                        input_ids,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )
                    hidden_states = out.hidden_states
                    target = hidden_states[-2][0].detach().float().cpu()
                    targets.append(target)
                    for layer in layers:
                        xs[layer].append(
                            hidden_states[layer + 1][0].detach().float().cpu()
                        )
            if not targets:
                self.last_error = "校准前向未产生隐藏状态"
                return 0
            target_all = torch.cat(targets, dim=0)
            new_translators: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            new_agreements: dict[int, float] = {}
            for layer in layers:
                if time.time() - started > time_budget_s * 3:
                    break
                try:
                    layer_all = torch.cat(xs[layer], dim=0)
                    w, b = fit_translator(layer_all, target_all, self.ridge)
                    agreement = self._agreement(layer_all, target_all, w, b)
                    new_translators[layer] = (w, b)
                    new_agreements[layer] = agreement
                except Exception as exc:
                    self.last_error = f"第 {layer} 层拟合失败：{exc}"
                    continue
            if not new_translators:
                return 0
            with self._lock:
                self._translators = new_translators
                self._agreements = new_agreements
            self.corpus_size = used_segments
            self.fitted_at = time.time()
            self.save_cache()
            return len(new_translators)
        except Exception as exc:
            self.last_error = f"校准失败：{exc}"
            return 0

    def _agreement(
        self,
        layer_all: torch.Tensor,
        target_all: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
    ) -> float:
        """Top-1 agreement between tuned projection and the true final logits."""
        with torch.inference_mode():
            mapped = apply_translator(layer_all, w, b)
            tuned_logits = self.lm_head(self.final_norm(mapped.float()))
            true_logits = self.lm_head(self.final_norm(target_all.float()))
            tuned_top = tuned_logits.argmax(dim=-1)
            true_top = true_logits.argmax(dim=-1)
            return float((tuned_top == true_top).float().mean().item())

    def _encode(self, text: str) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=False)
        ids = _ids_from_encoding(encoded)
        return [int(i) for i in ids]


def _ids_from_encoding(encoded: Any) -> list[int]:
    if isinstance(encoded, dict):
        ids = encoded.get("input_ids", [])
    else:
        ids = getattr(encoded, "input_ids", encoded)
    if isinstance(ids, torch.Tensor):
        ids = ids.reshape(-1).tolist()
    if isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return list(ids or [])
