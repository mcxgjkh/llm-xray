"""Uvicorn entry point for the LLM-Xray backend.

Usage::

    python backend/run.py                 # 127.0.0.1:8765
    python backend/run.py --port 8800
    python backend/run.py --reload        # development

Environment variables are documented in :mod:`xray.config`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent


def _ensure_import_path() -> None:
    """Make ``import xray`` work regardless of the caller's working directory."""
    backend = str(BACKEND_DIR)
    if backend not in sys.path:
        sys.path.insert(0, backend)
    # uvicorn's reload subprocess re-imports the app string; pass the path along.
    current = os.environ.get("PYTHONPATH", "")
    parts = [part for part in current.split(os.pathsep) if part]
    if backend not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([backend, *parts]) if parts else backend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动 LLM-Xray 后端（FastAPI + LM Studio 适配）")
    parser.add_argument("--host", default=None, help="监听地址，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=None, help="监听端口，默认 8765")
    parser.add_argument("--reload", action="store_true", help="开发模式：代码变更自动重载")
    parser.add_argument("--log-level", default="info", choices=["critical", "error", "warning", "info", "debug", "trace"])
    parser.add_argument("--no-access-log", action="store_true", help="关闭 Uvicorn 访问日志")
    return parser


def main(argv: list[str] | None = None) -> int:
    _ensure_import_path()
    from xray.config import get_settings

    settings = get_settings()
    args = build_parser().parse_args(argv)
    host = args.host or settings.host
    port = args.port or settings.port

    import uvicorn

    print(f"LLM-Xray {settings.version} · http://{host}:{port}")
    print(f"LM Studio: {settings.lmstudio_base_url} · model: {settings.model} · provider: {settings.provider}")
    print(f"数据目录: {settings.data_dir}")

    uvicorn.run(
        "xray.app:app",
        host=host,
        port=port,
        reload=bool(args.reload),
        reload_dirs=[str(BACKEND_DIR)] if args.reload else None,
        log_level=args.log_level,
        access_log=not args.no_access_log,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
