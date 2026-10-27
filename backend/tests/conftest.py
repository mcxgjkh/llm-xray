"""Pytest configuration: make ``import xray`` work without installing the package."""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent

for path in (str(BACKEND_DIR),):
    if path not in sys.path:
        sys.path.insert(0, path)


def make_test_settings(tmp_path: Path, **overrides):
    """Settings pointing all writable paths at ``tmp_path`` and LM Studio at a dead port."""
    from xray.config import Settings

    values = dict(
        data_dir=tmp_path / "data",
        memory_dir=tmp_path / "memory",
        docs_dir=PROJECT_ROOT / "docs",
        static_dir=tmp_path / "dist",
        lmstudio_base_url="http://127.0.0.1:9",
        model="demo-model",
        lens_model="test-lens-model",
        # Keep the suite fast: this host takes ~2 s to refuse loopback connections.
        lmstudio_connect_timeout=0.2,
        embed_enabled=False,
    )
    values.update(overrides)
    return Settings(**values)
