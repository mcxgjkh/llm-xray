"""Optional bridge to the surrogate lens engine (``backend/xray/lens/``).

The lens package is an optional, independently developed component.  The core
backend must run — and the demo/UI must stay fully usable — when it is missing,
still under construction, or raises.  Every interaction is therefore wrapped:

* imports happen lazily, so the module can appear in a running installation;
* ``load`` is a background task (never auto-started);
* ``observe`` is executed via ``asyncio.to_thread`` so a slow CPU forward pass
  can never block the LM Studio token stream;
* all errors become a ``disabled``/``error`` status dict, never an exception on
  the core path.
"""

from __future__ import annotations

import asyncio
import importlib
import os
from typing import Any

from .config import Settings

LENS_THREADS = 2


def _same_weights(lens_model: str | None, runtime_model: str | None) -> bool:
    if not lens_model or not runtime_model:
        return False

    def normalise(value: str) -> str:
        base = os.path.basename(value.replace("\\", "/")).lower()
        for suffix in (".gguf", ".safetensors", ".bin", ".pt"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
        return base

    left, right = normalise(lens_model), normalise(runtime_model)
    return bool(left) and bool(right) and (left == right or left in right or right in left)


class LensBridge:
    """Covers the optional ``LensEngine`` contract from docs/SPEC.md §8."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._engine: Any = None
        self._engine_class: Any = None
        self._import_error: str | None = None
        self._state = "unknown"
        self._message = ""
        self._device: str | None = None
        self._model: str | None = None
        self._load_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # engine discovery
    # ------------------------------------------------------------------
    def _resolve_engine_class(self) -> Any:
        if self._engine_class is not None:
            return self._engine_class
        package = __package__ or "xray"
        errors: list[str] = []
        for module_name in (".lens.api", ".lens", ".lens.engine"):
            try:
                module = importlib.import_module(module_name, package=package)
            except Exception as exc:  # ImportError, but also broken optional deps
                errors.append(f"{module_name}: {exc.__class__.__name__}: {exc}")
                continue
            engine_class = getattr(module, "LensEngine", None)
            if engine_class is None:
                continue
            self._engine_class = engine_class
            if self._state in ("unknown", "disabled"):
                self._state = "unloaded"
            return engine_class
        self._state = "disabled"
        self._import_error = "; ".join(errors) or "backend/xray/lens 未安装"
        return None

    def _make_engine(self) -> Any:
        engine_class = self._resolve_engine_class()
        if engine_class is None:
            return None
        try:
            return engine_class()
        except TypeError:
            try:
                return engine_class(model_name=self.settings.lens_model, device="cpu")
            except Exception:
                return engine_class()
        except Exception:
            return None

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------
    def _base_status(self, status: str, *, message: str = "") -> dict[str, Any]:
        model = self._model or self.settings.lens_model
        return {
            "status": status,
            "model": model,
            "device": self._device,
            "surrogate": True,
            "same_weights": _same_weights(model, self.settings.model),
            "calibrated_layers": 0,
            "message": message,
            "threads": LENS_THREADS,
        }

    def status(self) -> dict[str, Any]:
        if self._engine is None:
            engine_class = self._resolve_engine_class()
            if engine_class is None:
                return self._base_status(
                    "disabled",
                    message="透镜模块不可用（backend/xray/lens 尚未就绪）；演示与实时通道不受影响",
                )
            if self._state == "loading":
                return self._base_status("loading", message=self._message or "正在加载透镜模型")
            if self._state == "error":
                return self._base_status("error", message=self._message or "透镜加载失败")
            return self._base_status("unloaded", message="透镜模型未加载；可在设置中显式加载")
        raw: dict[str, Any] = {}
        try:
            reported = self._engine.status()
            if isinstance(reported, dict):
                raw = reported
        except Exception as exc:
            return self._base_status("error", message=f"读取透镜状态失败: {exc}")
        status = raw.get("status")
        if self._state == "loading":
            status = "loading"
        if status not in ("unloaded", "loading", "ready", "error", "disabled"):
            status = "ready" if self.is_ready() else "unloaded"
        merged = self._base_status(str(status), message=str(raw.get("message") or self._message or ""))
        for key in ("model", "device", "surrogate", "same_weights", "calibrated_layers", "threads"):
            if key in raw and raw[key] is not None:
                if key == "same_weights" and not raw[key]:
                    merged[key] = _same_weights(merged.get("model"), self.settings.model)
                else:
                    merged[key] = raw[key]
        merged["same_weights"] = _same_weights(merged.get("model"), self.settings.model) or bool(
            merged.get("same_weights")
        )
        return merged

    def available(self) -> bool:
        return self._resolve_engine_class() is not None

    def is_ready(self) -> bool:
        if self._engine is None:
            return False
        try:
            return bool(self._engine.is_ready())
        except Exception:
            return False

    # ------------------------------------------------------------------
    # load / unload (background)
    # ------------------------------------------------------------------
    def start_load(self, model: str | None = None, device: str = "cpu") -> dict[str, Any]:
        """Kick off loading in the background and return the immediate status."""
        engine_class = self._resolve_engine_class()
        if engine_class is None:
            return self.status()
        if self.is_ready():
            return self.status()
        if self._load_task is not None and not self._load_task.done():
            return self.status()
        self._state = "loading"
        self._message = "正在加载透镜模型（低优先级 CPU 线程）"
        self._model = model or self._model or self.settings.lens_model
        self._device = device
        self._load_task = asyncio.create_task(self._load(model, device), name="xray-lens-load")
        return self.status()

    async def _load(self, model: str | None, device: str) -> None:
        try:
            await asyncio.to_thread(self._load_blocking, model, device)
        except asyncio.CancelledError:
            self._state = "unloaded"
            raise
        except Exception as exc:
            self._state = "error"
            self._message = f"透镜加载失败: {exc}"

    def _load_blocking(self, model: str | None, device: str) -> None:
        if self._engine is None:
            self._engine = self._make_engine()
        engine = self._engine
        if engine is None:
            self._state = "disabled"
            self._message = "无法构造透镜引擎"
            return
        self._model = model or self._model or self.settings.lens_model
        self._device = device
        try:
            result = engine.load(self._model, device)
        except TypeError:
            result = engine.load(self._model)
        if isinstance(result, dict):
            if isinstance(result.get("device"), str):
                self._device = result["device"]
            if isinstance(result.get("model"), str):
                self._model = result["model"]
            if result.get("status") == "error":
                self._state = "error"
                self._message = str(result.get("message") or "透镜加载失败")
                return
        if self.is_ready():
            self._state = "ready"
            self._message = ""
        else:
            self._state = "error"
            self._message = "透镜引擎未报告 ready"

    async def wait_loaded(self, timeout: float = 120.0) -> dict[str, Any]:
        task = self._load_task
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
            except (asyncio.TimeoutError, Exception):  # noqa: B014 - defensive
                pass
        return self.status()

    async def unload(self) -> dict[str, Any]:
        task = self._load_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: B014 - defensive
                pass
        self._load_task = None
        engine = self._engine
        if engine is not None:
            try:
                await asyncio.to_thread(engine.unload)
            except Exception as exc:
                self._message = f"卸载透镜失败: {exc}"
        self._state = "unloaded"
        self._message = "透镜模型已释放"
        return self.status()

    # ------------------------------------------------------------------
    # inference path (blocking calls pushed to threads)
    # ------------------------------------------------------------------
    async def reset(self, prompt_text: str, system: str = "") -> bool:
        if not self.is_ready():
            return False
        engine = self._engine
        try:
            await asyncio.to_thread(engine.reset, prompt_text, system)
            return True
        except Exception as exc:
            self._message = f"透镜 reset 失败: {exc}"
            return False

    async def observe_async(self, delta_text: str, step: int) -> list[dict[str, Any]]:
        """Run one (blocking) lens observation in a worker thread."""
        if not self.is_ready():
            return []
        engine = self._engine
        try:
            probes = await asyncio.to_thread(engine.observe, delta_text, step)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._message = f"透镜 observe 失败: {exc}"
            self._state = "error"
            return []
        if isinstance(probes, dict):
            return [probes]
        if isinstance(probes, (list, tuple)):
            return [probe for probe in probes if isinstance(probe, dict)]
        return []

    def projection(self) -> dict[str, Any] | None:
        if not self.is_ready():
            return None
        try:
            projection = self._engine.projection()
            return projection if isinstance(projection, dict) else None
        except Exception:
            return None

    async def shutdown(self) -> None:
        task = self._load_task
        self._load_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: B014 - defensive
                pass
