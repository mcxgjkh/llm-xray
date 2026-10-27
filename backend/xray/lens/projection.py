"""3D projection of final-layer hidden states (UMAP -> PCA -> deterministic).

The backend calls :meth:`Projector.add` for prompt and generated tokens and
:meth:`Projector.maybe_fit` every 8 points.  Fitting runs on a daemon thread so
token generation never blocks; until a fit finishes every point already has a
stable deterministic coordinate, so the UI always has something to draw.

Reducers:
* UMAP (umap-learn) for >= 8 points, ``n_neighbors=min(12, n-1)``,
  ``min_dist=0.25``, ``metric="cosine"``, ``random_state=7``, ``n_jobs=1``.
* PCA (numpy SVD) when UMAP is missing/fails or n < 8.
* Deterministic hash projection for n < 3 or degenerate input.
"""

from __future__ import annotations

import hashlib
import math
import threading
from typing import Any, Sequence

import numpy as np

MAX_POINTS = 256
FIT_EVERY = 8
MAX_HIDDEN_DIM = 4096


def _hash_coord(token: str, index: int, component: int, seed: int = 7) -> float:
    key = f"{seed}:{component}:{index}:{token}".encode("utf-8", errors="ignore")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    value = int.from_bytes(digest, "big") / float(2**64)
    return value * 2.0 - 1.0


def deterministic_projection(
    points: Sequence[dict], seed: int = 7
) -> list[dict]:
    """Stable pseudo-3D coordinates derived from token text + index."""
    result = []
    for point in points:
        token = str(point.get("token", ""))
        index = int(point.get("index", len(result)))
        base = index * 0.37
        coords = []
        for component in range(3):
            jitter = _hash_coord(token, index, component, seed)
            wave = math.sin(base * (1.0 + 0.6 * component) + jitter * math.pi)
            coords.append(max(-1.0, min(1.0, 0.75 * wave + 0.25 * jitter)))
        result.append(
            {
                "id": str(point.get("id") or f"t{index}"),
                "token": token,
                "index": index,
                "kind": point.get("kind", "generated"),
                "step": point.get("step"),
                "x": round(coords[0], 5),
                "y": round(coords[1], 5),
                "z": round(coords[2], 5),
            }
        )
    return result


def pca_projection(matrix: np.ndarray, components: int = 3) -> np.ndarray:
    """Numpy SVD PCA to at most ``components`` dims (zero padded)."""
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, components), dtype=np.float32)
    centered = arr - arr.mean(axis=0, keepdims=True)
    try:
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros((arr.shape[0], components), dtype=np.float32)
    take = min(components, vt.shape[0])
    coords = centered @ vt[:take].T if take > 0 else np.zeros((arr.shape[0], 0))
    out = np.zeros((arr.shape[0], components), dtype=np.float32)
    out[:, :take] = coords[:, :take]
    return out


def umap_projection(matrix: np.ndarray) -> np.ndarray | None:
    """Fit 3-D UMAP; returns ``None`` when unavailable or it fails."""
    arr = np.asarray(matrix, dtype=np.float32)
    n = int(arr.shape[0])
    if n < 8:
        return None
    try:
        from umap import UMAP  # imported lazily: heavy dependency

        n_neighbors = min(12, n - 1)
        reducer = UMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            min_dist=0.25,
            metric="cosine",
            random_state=7,
            n_jobs=1,
        )
        coords = reducer.fit_transform(arr)
    except Exception:
        return None
    coords = np.asarray(coords, dtype=np.float32)
    if coords.shape != (n, 3) or not np.isfinite(coords).all():
        return None
    return coords


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    """Min-max normalise every axis into [-1, 1] (constant axis -> 0)."""
    arr = np.asarray(coords, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return arr
    out = np.zeros_like(arr)
    for axis in range(arr.shape[1]):
        column = arr[:, axis]
        lo = float(column.min())
        hi = float(column.max())
        if not math.isfinite(lo) or not math.isfinite(hi) or hi - lo <= 1e-9:
            out[:, axis] = 0.0
        else:
            out[:, axis] = (column - lo) / (hi - lo) * 2.0 - 1.0
    return out


class Projector:
    """Thread-safe accumulator of hidden-state points and their 3D projection."""

    def __init__(
        self,
        max_points: int = MAX_POINTS,
        fit_every: int = FIT_EVERY,
        seed: int = 7,
        use_umap: bool = True,
    ) -> None:
        self.max_points = int(max_points)
        self.fit_every = int(fit_every)
        self.seed = int(seed)
        self.use_umap = bool(use_umap)
        self._lock = threading.RLock()
        self._points: list[dict] = []
        self._hidden: list[np.ndarray] = []
        self._coords: dict[str, tuple[float, float, float]] = {}
        self._reducer = "deterministic"
        self._fitted_at: int | None = None
        self._dirty = 0
        self._fitting = False
        self._thread: threading.Thread | None = None
        self._hidden_dim = 0
        self._generation = 0

    # -- collection -------------------------------------------------------

    def reset(self) -> None:
        with self._lock:
            self._points = []
            self._hidden = []
            self._coords = {}
            self._reducer = "deterministic"
            self._fitted_at = None
            self._dirty = 0
            self._hidden_dim = 0
            self._generation += 1

    def _point_id(self, index: int) -> str:
        return f"t{int(index)}"

    def _trim_locked(self) -> None:
        while len(self._points) > self.max_points:
            # Keep the first prompt point, drop the second-oldest generated point.
            if len(self._points) > 2:
                del self._points[1]
                del self._hidden[1]
            else:
                break
            if len(self._points) > self.max_points:
                continue
        # Recompute coordinates for the surviving set so ids stay attached.
        if len(self._points) > self.max_points:
            overflow = len(self._points) - self.max_points
            del self._points[:overflow]
            del self._hidden[:overflow]

    def add(
        self,
        hidden: Any,
        token: str,
        index: int,
        kind: str = "generated",
        step: int | None = None,
    ) -> dict:
        vector = np.asarray(hidden, dtype=np.float32).reshape(-1)
        if vector.size == 0:
            vector = np.zeros(1, dtype=np.float32)
        if vector.size > MAX_HIDDEN_DIM:
            vector = vector[:MAX_HIDDEN_DIM]
        point_id = self._point_id(index)
        coords = (
            _hash_coord(token, index, 0, self.seed),
            _hash_coord(token, index, 1, self.seed),
            _hash_coord(token, index, 2, self.seed),
        )
        with self._lock:
            self._points.append(
                {
                    "id": point_id,
                    "token": str(token),
                    "index": int(index),
                    "kind": "prompt" if kind == "prompt" else "generated",
                    "step": None if step is None else int(step),
                }
            )
            self._hidden.append(vector.astype(np.float32))
            self._hidden_dim = int(vector.size)
            self._coords[point_id] = coords
            self._trim_locked()
            self._dirty += 1
        vector_dict = {"x": round(coords[0], 5), "y": round(coords[1], 5), "z": round(coords[2], 5)}
        if point_id not in self._coords:
            # Trim removed the just-added point (max_points < 2 edge case).
            return vector_dict
        return self.vector_for(point_id) or vector_dict

    def vector_for(self, point_id: str) -> dict | None:
        with self._lock:
            coords = self._coords.get(str(point_id))
        if coords is None:
            return None
        return {"x": round(coords[0], 5), "y": round(coords[1], 5), "z": round(coords[2], 5)}

    # -- fitting ----------------------------------------------------------

    def should_fit(self) -> bool:
        with self._lock:
            return len(self._points) >= 2 and self._dirty >= self.fit_every and not self._fitting

    def maybe_fit(self, force: bool = False) -> bool:
        """Schedule a background fit.  Returns True when a fit was scheduled."""
        with self._lock:
            if self._fitting:
                return False
            ready = len(self._points) >= 3
            enough = force or (self.should_fit() if ready else False)
            if not enough:
                return False
            self._fitting = True
            self._dirty = 0
            generation = self._generation
            points = list(self._points)
            matrix = (
                np.stack(self._hidden)
                if self._hidden
                else np.zeros((len(points), 1), dtype=np.float32)
            )
        self._thread = threading.Thread(
            target=self._fit_worker,
            args=(points, matrix, generation),
            name="xray-projection",
            daemon=True,
        )
        self._thread.start()
        return True

    def _fit_worker(self, points: list[dict], matrix: np.ndarray, generation: int) -> None:
        try:
            coords, reducer = self._compute_coords(matrix)
        except Exception:
            coords, reducer = None, "deterministic"
        with self._lock:
            if generation != self._generation:
                self._fitting = False
                return
            if coords is None or len(points) != coords.shape[0]:
                result = deterministic_projection(points, self.seed)
            else:
                result = []
                for point, coord in zip(points, coords):
                    result.append(
                        {
                            "id": point["id"],
                            "token": point["token"],
                            "index": point["index"],
                            "kind": point["kind"],
                            "step": point["step"],
                            "x": round(float(coord[0]), 5),
                            "y": round(float(coord[1]), 5),
                            "z": round(float(coord[2]), 5),
                        }
                    )
                self._reducer = reducer
            for point in result:
                self._coords[point["id"]] = (point["x"], point["y"], point["z"])
            self._fitted_at = len(self._points)
            self._fitting = False

    def fit_sync(self) -> None:
        """Synchronous fit, used by tests and by the final flush."""
        with self._lock:
            points = list(self._points)
            matrix = (
                np.stack(self._hidden)
                if self._hidden
                else np.zeros((len(points), 1), dtype=np.float32)
            )
            generation = self._generation
        coords, reducer = self._compute_coords(matrix)
        with self._lock:
            if generation != self._generation:
                return
            if coords is None or len(points) != coords.shape[0]:
                result = deterministic_projection(points, self.seed)
            else:
                result = [
                    {
                        "id": point["id"],
                        "token": point["token"],
                        "index": point["index"],
                        "kind": point["kind"],
                        "step": point["step"],
                        "x": round(float(coord[0]), 5),
                        "y": round(float(coord[1]), 5),
                        "z": round(float(coord[2]), 5),
                    }
                    for point, coord in zip(points, coords)
                ]
                self._reducer = reducer
            for point in result:
                self._coords[point["id"]] = (point["x"], point["y"], point["z"])
            self._fitted_at = len(self._points)

    def _compute_coords(self, matrix: np.ndarray) -> tuple[np.ndarray | None, str]:
        arr = np.asarray(matrix, dtype=np.float32)
        n = int(arr.shape[0])
        if n == 0:
            return np.zeros((0, 3), dtype=np.float32), "deterministic"
        if arr.ndim != 2 or not np.isfinite(arr).all():
            arr = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0)
        if n >= 8 and self.use_umap:
            coords = umap_projection(arr)
            if coords is not None:
                return normalize_coords(coords), "UMAP"
        if n >= 3:
            return normalize_coords(pca_projection(arr, 3)), "PCA"
        return normalize_coords(deterministic_projection_matrix(arr, n)), "deterministic"

    # -- output -----------------------------------------------------------

    def wait(self, timeout: float | None = 5.0) -> None:
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def snapshot(self) -> dict:
        with self._lock:
            points = []
            for point in self._points:
                coords = self._coords.get(point["id"], (0.0, 0.0, 0.0))
                points.append(
                    {
                        "id": point["id"],
                        "token": point["token"],
                        "index": point["index"],
                        "kind": point["kind"],
                        "step": point["step"],
                        "x": round(float(coords[0]), 5),
                        "y": round(float(coords[1]), 5),
                        "z": round(float(coords[2]), 5),
                    }
                )
            return {
                "points": points,
                "reducer": self._reducer,
                "fitted_at": self._fitted_at,
            }

    @property
    def length(self) -> int:
        with self._lock:
            return len(self._points)


def deterministic_projection_matrix(matrix: np.ndarray, n: int) -> np.ndarray:
    """Deterministic 3-D coordinates when nothing else applies."""
    arr = np.asarray(matrix, dtype=np.float32)
    out = np.zeros((n, 3), dtype=np.float32)
    for row in range(n):
        for component in range(3):
            if row < arr.shape[0] and arr.shape[1] > component:
                base = float(arr[row, component])
            else:
                base = 0.0
            digest = hashlib.blake2b(
                f"{row}:{component}:{base:.6f}".encode(), digest_size=8
            ).digest()
            out[row, component] = int.from_bytes(digest, "big") / float(2**64) * 2 - 1
    return out
