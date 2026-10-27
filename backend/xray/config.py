"""Runtime configuration for LLM-Xray.

All paths are derived from the project root (never from the process CWD) so the
backend behaves the same whether it is started from ``backend/``, the repo root
or from the packaged desktop shell.

Environment variables (all optional):

===================== ================================================= =========================
``LLM_XRAY_HOST``     bind address for uvicorn                           127.0.0.1
``LLM_XRAY_PORT``     bind port for uvicorn                             8765
``LMSTUDIO_BASE_URL`` LM Studio server base URL                         http://127.0.0.1:1234
``LLM_XRAY_MODEL``    default LM Studio model key                       deepseek-r1-distill-llama-8b
``LLM_XRAY_PROVIDER`` streaming provider: auto | responses | chat | native   auto
``LLM_XRAY_LENS_MODEL`` surrogate lens HF model or local path           Qwen/Qwen2.5-0.5B-Instruct
``LLM_XRAY_DATA_DIR`` runtime data directory (sessions/cache)          <root>/data
===================== ================================================= =========================
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path

from . import __version__

# ``backend/xray/config.py`` -> backend/xray -> backend -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_LMSTUDIO_URL = "http://127.0.0.1:1234"
DEFAULT_MODEL = "deepseek-r1-distill-llama-8b"
DEFAULT_LENS_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"
VALID_PROVIDERS = ("auto", "responses", "chat", "native")

#: Sampling defaults, mirrored from apps/desktop/src/types.ts (DEFAULT_SAMPLING).
DEFAULT_SAMPLING: dict = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "repetition_penalty": 1.05,
    "max_tokens": 4096,
    "seed": None,
}


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _read_version() -> str:
    try:
        text = (PROJECT_ROOT / "VERSION").read_text(encoding="utf-8").strip()
        return text or __version__
    except OSError:
        return __version__


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings. Use :meth:`from_env` or :func:`dataclasses.replace`."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    lmstudio_base_url: str = DEFAULT_LMSTUDIO_URL
    model: str = DEFAULT_MODEL
    provider: str = "auto"
    lens_model: str = DEFAULT_LENS_MODEL
    embed_model: str = DEFAULT_EMBED_MODEL
    data_dir: Path = PROJECT_ROOT / "data"
    memory_dir: Path = PROJECT_ROOT / "memory"
    docs_dir: Path = PROJECT_ROOT / "docs"
    static_dir: Path = PROJECT_ROOT / "apps" / "desktop" / "dist"
    version: str = dataclasses.field(default_factory=_read_version)
    #: Connection timeout for LM Studio metadata probes (fast failure when the
    #: local server is not running keeps ``hello`` and ``/api/health`` snappy).
    lmstudio_connect_timeout: float = 3.0
    #: Whether RAG may call ``/v1/embeddings`` (falls back to pure BM25 when off).
    embed_enabled: bool = True
    cors_origins: tuple[str, ...] = (
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    )

    # ---- derived paths -------------------------------------------------
    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    # ---- constructors --------------------------------------------------
    @classmethod
    def from_env(cls) -> "Settings":
        provider = _env("LLM_XRAY_PROVIDER", "auto").lower()
        if provider not in VALID_PROVIDERS:
            provider = "auto"
        data_dir = Path(_env("LLM_XRAY_DATA_DIR", str(PROJECT_ROOT / "data"))).expanduser()
        if not data_dir.is_absolute():
            data_dir = (PROJECT_ROOT / data_dir).resolve()
        try:
            connect_timeout = float(_env("LLM_XRAY_LMSTUDIO_CONNECT_TIMEOUT", "3.0"))
        except ValueError:
            connect_timeout = 3.0
        return cls(
            host=_env("LLM_XRAY_HOST", DEFAULT_HOST),
            port=_env_int("LLM_XRAY_PORT", DEFAULT_PORT),
            lmstudio_base_url=_env("LMSTUDIO_BASE_URL", DEFAULT_LMSTUDIO_URL).rstrip("/"),
            model=_env("LLM_XRAY_MODEL", DEFAULT_MODEL),
            provider=provider,
            lens_model=_env("LLM_XRAY_LENS_MODEL", DEFAULT_LENS_MODEL),
            embed_model=_env("LLM_XRAY_EMBED_MODEL", DEFAULT_EMBED_MODEL),
            lmstudio_connect_timeout=max(0.05, connect_timeout),
            embed_enabled=_env("LLM_XRAY_EMBEDDINGS", "1").lower() not in ("0", "false", "no", "off"),
            data_dir=data_dir,
            memory_dir=Path(_env("LLM_XRAY_MEMORY_DIR", str(PROJECT_ROOT / "memory"))),
            docs_dir=Path(_env("LLM_XRAY_DOCS_DIR", str(PROJECT_ROOT / "docs"))),
            static_dir=Path(
                _env("LLM_XRAY_STATIC_DIR", str(PROJECT_ROOT / "apps" / "desktop" / "dist"))
            ),
        )

    def with_overrides(self, **kwargs) -> "Settings":
        """Return a copy with the given fields replaced (convenience for tests)."""
        return dataclasses.replace(self, **kwargs)

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.sessions_dir, self.cache_dir):
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError:
                # Read-only environments: the app still runs, persistence degrades.
                pass

    # ---- serialization -------------------------------------------------
    def public_dict(self) -> dict:
        """Paths and settings safe to expose through ``GET /api/config``."""
        return {
            "version": self.version,
            "lmstudio_url": self.lmstudio_base_url,
            "default_model": self.model,
            "default_mode": "lmstudio",
            "provider": self.provider,
            "lens_model": self.lens_model,
            "embed_model": self.embed_model,
            "data_dir": str(self.data_dir),
            "memory_dir": str(self.memory_dir),
            "sessions_dir": str(self.sessions_dir),
            "static_dir": str(self.static_dir),
            "sampling_defaults": dict(DEFAULT_SAMPLING),
        }


_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide settings (cached after the first call)."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def reset_settings() -> Settings:
    """Drop the cache and re-read the environment (used by tests)."""
    global _settings
    _settings = Settings.from_env()
    return _settings
