"""Optional surrogate lens for internal model telemetry.

LM Studio does not expose hidden states or attention. This package observes a
small local Hugging Face model on the same text and clearly labels the result
as surrogate telemetry in the UI.

Only :mod:`xray.lens.api` should be imported by the backend core; importing it
does not import torch/transformers (those load lazily in ``LensEngine.load``).
"""

from .api import (
    DEMO_MODEL,
    DEFAULT_MODEL,
    LensEngine,
    demo_lens_status,
    get_engine,
    reset_engine,
    synthetic_probe,
    synthetic_projection,
    synthetic_projection_event,
)

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
