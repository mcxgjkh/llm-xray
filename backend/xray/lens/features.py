"""Sparse MLP feature probe (honest alternative to a real SAE).

Without external sparse-autoencoder weights we take the top-k neuron
activations of a middle-layer MLP and decode each neuron through its ``down_proj``
column and the ``lm_head`` directions, producing a readable label such as
``"紫色 / violet"`` plus a normalised activation.  The UI must label this as
``probe`` -- it is not an SAE.

An optional ``sae_lens`` adapter is reserved: when the package and a matching
checkpoint are available the engine can switch to real SAE features.  Import
failures never affect the rest of the lens.
"""

from __future__ import annotations

import threading
from typing import Any, Sequence

import numpy as np

from .common import find_decoder_layers, find_layer_branches, token_text


def try_import_sae_lens() -> Any | None:
    try:
        import sae_lens  # type: ignore

        return sae_lens
    except Exception:
        return None


class SAEAdapter:
    """Minimal placeholder adapter around an optional ``sae_lens`` install.

    Usage (future): ``SAEAdapter.load(release, sae_id, device="cpu")`` then
    ``encode(activations)``.  When ``sae_lens`` is missing every method returns
    ``None`` / raises a clear ``RuntimeError`` only if explicitly requested.
    """

    def __init__(self, release: str | None = None, sae_id: str | None = None, device: str = "cpu"):
        self.release = release
        self.sae_id = sae_id
        self.device = device
        self._sae = None
        self._module = try_import_sae_lens()

    @property
    def available(self) -> bool:
        return self._module is not None and self._sae is not None

    @property
    def installed(self) -> bool:
        return self._module is not None

    def load(self) -> bool:
        if self._module is None:
            return False
        try:
            self._sae = self._module.load_from_pretrained(
                release=self.release, sae_id=self.sae_id, device=self.device
            )
            return True
        except Exception:
            self._sae = None
            return False

    def encode(self, activations: Any) -> Any | None:
        if self._sae is None:
            return None
        try:
            import torch

            tensor = torch.as_tensor(activations, dtype=torch.float32)
            with torch.inference_mode():
                return self._sae.encode(tensor)
        except Exception:
            return None


class FeatureProbe:
    """Captures middle-layer MLP neuron activations via a ``down_proj`` hook."""

    def __init__(self, model: Any, layer_index: int | None = None) -> None:
        self.layers = find_decoder_layers(model)
        self.layer_index = (
            int(layer_index)
            if layer_index is not None
            else max(0, len(self.layers) // 2)
        )
        self._lock = threading.RLock()
        self._activation: Any = None
        self._handle: Any = None
        self.available = False
        self.down_proj = None
        self.lm_head_weight: np.ndarray | None = None

    def register(self) -> bool:
        self.close()
        self.available = False
        if not self.layers:
            return False
        self.layer_index = max(0, min(self.layer_index, len(self.layers) - 1))
        layer = self.layers[self.layer_index]
        _attention, mlp = find_layer_branches(layer)
        if mlp is None:
            return False
        down_proj = getattr(mlp, "down_proj", None) or getattr(mlp, "c_proj", None)
        if down_proj is None:
            return False
        self.down_proj = down_proj
        self._handle = down_proj.register_forward_hook(self._hook)
        self.available = True
        return True

    def _hook(self, _module, inputs, _output):
        if not inputs:
            return
        tensor = inputs[0]
        if tensor is not None and hasattr(tensor, "detach"):
            with self._lock:
                self._activation = tensor.detach()

    def set_lm_head(self, lm_head: Any) -> None:
        try:
            weight = lm_head.weight
            self.lm_head_weight = weight.detach().float().cpu().numpy()
        except Exception:
            self.lm_head_weight = None

    def prepare(self) -> None:
        with self._lock:
            self._activation = None

    def take(self, n_new: int) -> np.ndarray | None:
        with self._lock:
            activation = self._activation
        if activation is None:
            return None
        try:
            rows = activation[0, : int(n_new)].float().cpu().numpy()
        except Exception:
            return None
        return np.asarray(rows, dtype=np.float32)

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.remove()
            except Exception:
                pass
        self._handle = None
        self.available = False
        with self._lock:
            self._activation = None


def decode_features(
    activations: Any,
    down_proj_weight: Any,
    lm_head_weight: Any,
    tokenizer: Any,
    top_k: int = 6,
    label_tokens: int = 2,
) -> list[dict]:
    """Decode top-k MLP neurons into readable ``FeatureActivation`` dicts.

    Args:
        activations: ``[intermediate]`` neuron activations at one token.
        down_proj_weight: ``[D, intermediate]`` (transformers linear layout).
        lm_head_weight: ``[V, D]`` output embedding.
        tokenizer: tokenizer used to turn ids into text.
    """
    acts = np.asarray(activations, dtype=np.float32).reshape(-1)
    if acts.size == 0:
        return []
    down = np.asarray(down_proj_weight, dtype=np.float32)
    lm_head = np.asarray(lm_head_weight, dtype=np.float32)
    if down.ndim != 2 or lm_head.ndim != 2:
        return []
    if down.shape[1] != acts.size:
        return []
    k = max(1, min(int(top_k), acts.size))
    candidates = np.argpartition(-np.abs(acts), k - 1)[:k]
    candidates = candidates[np.argsort(-np.abs(acts[candidates]))]
    scale = float(np.abs(acts).max()) or 1.0
    vocab = lm_head.shape[0]
    if vocab == 0:
        return []
    # One matmul for all top neurons reads the (large) lm_head once instead of
    # once per neuron -- important on CPU where it is memory-bandwidth bound.
    directions = np.asarray(down[:, candidates], dtype=np.float32)  # [D, k]
    norms = np.linalg.norm(directions, axis=0)
    norms[norms <= 1e-9] = 1.0
    scores = lm_head @ (directions / norms)  # [V, k]
    features: list[dict] = []
    top_n = min(int(label_tokens), vocab)
    for column, neuron in enumerate(candidates):
        activation = float(acts[int(neuron)]) / scale
        column_scores = scores[:, column]
        ids = np.argpartition(-column_scores, top_n - 1)[:top_n]
        ids = ids[np.argsort(-column_scores[ids])]
        tokens = [token_text(tokenizer, int(token_id)) for token_id in ids]
        tokens = [token for token in tokens if token]
        label = " / ".join(tokens) if tokens else f"neuron {int(neuron)}"
        features.append(
            {
                "id": int(neuron),
                "label": label,
                "activation": round(max(-1.0, min(1.0, activation)), 4),
                "tokens": tokens,
            }
        )
    return features


def decode_features_for_token(
    probe: FeatureProbe,
    position: int,
    tokenizer: Any,
    top_k: int = 6,
) -> list[dict]:
    """Convenience wrapper used by the engine for one observed token."""
    if not probe.available or probe.down_proj is None or probe.lm_head_weight is None:
        return []
    rows = probe.take(position + 1)
    if rows is None or position >= rows.shape[0]:
        return []
    weight = getattr(probe.down_proj, "weight", None)
    if weight is None:
        return []
    try:
        down = weight.detach().float().cpu().numpy()
    except Exception:
        return []
    return decode_features(
        rows[position], down, probe.lm_head_weight, tokenizer, top_k=top_k
    )
