"""Residual-stream branch capture and signed logit contributions.

For every decoder layer we capture the attention branch output (after ``o_proj``)
and the MLP branch output (after ``down_proj``) with forward hooks.  Those are
the exact increments added to the residual stream.

Logit contribution to the current chosen token uses the first-order
linearisation of the final norm at the final residual vector:

    logit_j ~= lm_head_weight[j] . (gamma * delta / rms(final_hidden))

which is the standard RMSNorm Jacobian ignoring the (small) second-order term.
Values are signed: positive pushes the chosen token up, negative down.

transformers 5.x Qwen2 paths are ``model.model.layers[i].self_attn`` and
``model.model.layers[i].mlp``; discovery is name tolerant (see common.py).
"""

from __future__ import annotations

import math
import threading
from typing import Any, Sequence

import numpy as np

from .common import find_decoder_layers, find_layer_branches, norm_eps_of, norm_weight_of


def _as_tensor_list(output: Any) -> Any:
    if isinstance(output, (tuple, list)):
        return output[0] if output else None
    return output


class ResidualCapture:
    """Forward hooks that record attention/MLP branch outputs of one forward."""

    def __init__(self, model: Any) -> None:
        self.layers = find_decoder_layers(model)
        self._handles: list[Any] = []
        self._lock = threading.RLock()
        self._attn: dict[int, Any] = {}
        self._mlp: dict[int, Any] = {}
        self.available = False

    def register(self) -> bool:
        self.close()
        for index, layer in enumerate(self.layers):
            attention, mlp = find_layer_branches(layer)
            if attention is not None:
                self._handles.append(
                    attention.register_forward_hook(self._make_hook(index, self._attn))
                )
            if mlp is not None:
                self._handles.append(
                    mlp.register_forward_hook(self._make_hook(index, self._mlp))
                )
        self.available = bool(self._handles)
        return self.available

    def _make_hook(self, index: int, store: dict[int, Any]):
        def hook(_module, _inputs, output):
            tensor = _as_tensor_list(output)
            if tensor is not None and hasattr(tensor, "detach"):
                with self._lock:
                    # Keep only the newest forward; observe() clears before each run.
                    store[index] = tensor.detach()

        return hook

    def prepare(self) -> None:
        with self._lock:
            self._attn.clear()
            self._mlp.clear()

    def take(self, n_new: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Return ``{layer: (attn_delta[n_new, D], mlp_delta[n_new, D])}``."""
        captured: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        with self._lock:
            layers = sorted(set(self._attn) & set(self._mlp))
            for index in layers:
                try:
                    attn = self._attn[index][0, :n_new].float().cpu().numpy()
                    mlp = self._mlp[index][0, :n_new].float().cpu().numpy()
                except Exception:
                    continue
                captured[index] = (
                    np.asarray(attn, dtype=np.float32),
                    np.asarray(mlp, dtype=np.float32),
                )
        return captured

    def close(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles = []
        with self._lock:
            self._attn.clear()
            self._mlp.clear()


def _l2(tensor: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(tensor, dtype=np.float32)))


def contribution_logits(
    final_hidden: Any,
    branch_delta: Any,
    lm_head_weight: Any,
    chosen_id: int,
    final_norm_weight: Any,
    eps: float = 1e-6,
) -> float:
    """First-order signed contribution of a branch delta to one vocab logit.

    Args:
        final_hidden: ``[D]`` pre-final-norm residual vector at the chosen token.
        branch_delta: ``[D]`` attention or MLP branch output at the same position.
        lm_head_weight: ``[V, D]`` output embedding matrix.
        chosen_id: vocab id of the chosen token.
        final_norm_weight: ``[D]`` RMSNorm/LayerNorm gain.
        eps: norm epsilon (RMSNorm ``variance_epsilon``).
    """
    hidden = np.asarray(final_hidden, dtype=np.float32).reshape(-1)
    delta = np.asarray(branch_delta, dtype=np.float32).reshape(-1)
    weight = np.asarray(lm_head_weight, dtype=np.float32)
    gain = np.asarray(final_norm_weight, dtype=np.float32).reshape(-1)
    if hidden.size == 0 or delta.size != hidden.size:
        return 0.0
    if weight.ndim == 1:
        weight = weight.reshape(1, -1)
    if gain.size != hidden.size:
        gain = np.ones_like(hidden)
    chosen_id = int(chosen_id)
    if chosen_id < 0 or chosen_id >= weight.shape[0]:
        return 0.0
    rms = float(np.sqrt(np.mean(hidden * hidden) + eps))
    if not math.isfinite(rms) or rms <= 0.0:
        return 0.0
    # d(normalized)/d(h) ~= gain / rms  (first-order approximation)
    direction = weight[chosen_id] * (gain / rms)
    return float(np.dot(direction, delta))


def summarize_residual(
    captured: dict[int, tuple[np.ndarray, np.ndarray]],
    position: int,
    final_hidden: Any,
    lm_head_weight: Any,
    final_norm: Any,
    chosen_id: int,
) -> list[dict]:
    """Build the per-layer ``Probe.residual`` list for one token.

    ``position`` indexes into the captured (new-token) rows of this forward.
    """
    gain = norm_weight_of(final_norm)
    eps = norm_eps_of(final_norm)
    hidden_row = np.asarray(final_hidden, dtype=np.float32).reshape(-1)
    rows: list[dict] = []
    for layer in sorted(captured):
        attn, mlp = captured[layer]
        if position >= attn.shape[0] or position >= mlp.shape[0]:
            continue
        attn_row = attn[position]
        mlp_row = mlp[position]
        attn_logit = contribution_logits(
            hidden_row, attn_row, lm_head_weight, chosen_id, gain, eps
        )
        mlp_logit = contribution_logits(
            hidden_row, mlp_row, lm_head_weight, chosen_id, gain, eps
        )
        rows.append(
            {
                "layer": int(layer),
                "attn_norm": round(_l2(attn_row), 5),
                "mlp_norm": round(_l2(mlp_row), 5),
                "attn_logit": round(float(attn_logit), 5),
                "mlp_logit": round(float(mlp_logit), 5),
            }
        )
    return rows
