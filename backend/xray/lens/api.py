"""Public API of the surrogate lens package.

The backend core imports only from here::

    from xray.lens.api import LensEngine, get_engine

``LensEngine`` is safe to construct before torch/models exist and never raises
into the event loop: failures surface through ``status()["message"]``.  Demo
mode uses the model-free synthesizers re-exported below.
"""

from __future__ import annotations

import threading

from .demo import (
    DEMO_MODEL,
    demo_lens_status,
    synthetic_probe,
    synthetic_projection,
    synthetic_projection_event,
)
from .engine import DEFAULT_MODEL, LensEngine

__all__ = [
    "LensEngine",
    "get_engine",
    "reset_engine",
    "DEFAULT_MODEL",
    "DEMO_MODEL",
    "synthetic_probe",
    "synthetic_projection",
    "synthetic_projection_event",
    "demo_lens_status",
]

_engine: LensEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> LensEngine:
    """Process-wide lens engine singleton (lazy, never loads a model)."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = LensEngine()
        return _engine


def reset_engine() -> None:
    """Drop/replace the singleton (tests, settings changes)."""
    global _engine
    with _engine_lock:
        engine = _engine
        _engine = None
    if engine is not None:
        try:
            engine.unload()
        except Exception:
            pass
