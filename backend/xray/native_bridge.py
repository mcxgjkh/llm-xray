"""Native "exact" inference bridge: Transformers generation + telemetry.

This is the second runtime channel next to LM Studio.  The backend loads a
local Hugging Face causal LM in-process (``local_files_only=True`` — never
downloads), then generates one token at a time.  Every token is produced by the
*same* forward pass that yields the hidden states, attention rows, residual
deltas, Tuned/Logit Lens and projection points, so the probe is exact
(``surrogate=False``, ``same_weights=True``) instead of a surrogate model's
guess at the text.

The heavy lifting lives in :class:`xray.lens.engine.LensEngine` (``prefill`` /
``sample_next``).  This module only adds:

* dependency probing and a small status dict the backend can render;
* ``asyncio.to_thread`` wrappers so a CPU forward never blocks the event loop;
* clear, graceful errors for missing dependencies / models / 4-bit requests.

It never contacts LM Studio and never downloads weights.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import os
from typing import Any

from .sampling import SamplingParams

try:  # Reuse the lens resolution logic / default model without importing torch.
    from .lens.engine import DEFAULT_MODEL
except Exception:  # pragma: no cover - lens package should always be present
    DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

NATIVE_MODEL_ENV = "XRAY_NATIVE_MODEL"
NATIVE_THREADS = 2


def _find_spec(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # pragma: no cover - broken environment
        return False


class NativeBridge:
    """Owns one in-process Transformers engine for exact generation.

    Typical use from ``app.py``::

        bridge = NativeBridge(settings)
        status = await bridge.load(payload.model)       # local HF path only
        await bridge.prefill(payload.messages, system=payload.system)
        for step in range(max_tokens):
            out = await bridge.sample_next(sampling, step)
            if not out["ok"] or out["eos"]:
                break
            # out["text"], out["topk"], out["probabilities"], out["entropy"],
            # out["probe"] (surrogate=False) are all from one forward pass.

    Pass an existing ``LensEngine`` as ``engine`` to reuse one loaded model;
    by default the bridge constructs its own engine so it cannot steal the
    surrogate lens channel's model or state.
    """

    def __init__(
        self,
        settings: Any = None,
        engine: Any = None,
        model_name: str | None = None,
        use_umap: bool = False,
        max_context: int | None = None,
    ) -> None:
        self.settings = settings
        self._engine = engine
        self._engine_class: Any = None
        self._use_umap = bool(use_umap)
        self._max_context = int(max_context) if max_context else None
        self._default_model = (
            model_name
            or os.environ.get(NATIVE_MODEL_ENV)
            or getattr(settings, "lens_model", None)
            or DEFAULT_MODEL
        )
        self._model: str | None = None
        self._device: str | None = None
        self._state = "unloaded"
        self._message = (
            "原生精确通道未加载；需要本地 HF 因果模型目录（不会自动下载）"
        )
        self._deps_ok: bool | None = None
        self._deps_error: str | None = None
        self._prompt_tokens = 0
        self._engine_info: dict[str, Any] = {}
        self._generation = False
        self._load_lock: asyncio.Lock | None = None

    # ------------------------------------------------------------------
    # dependencies / engine discovery
    # ------------------------------------------------------------------

    def _check_deps(self) -> str | None:
        """Return a human-readable dependency error, or ``None`` when usable."""
        if self._deps_ok is not None:
            return self._deps_error
        missing: list[str] = []
        if not _find_spec("torch"):
            missing.append("torch")
        if not _find_spec("transformers"):
            missing.append("transformers")
        if missing:
            self._deps_ok = False
            self._deps_error = (
                "原生通道不可用：缺少 "
                + "、".join(missing)
                + "。请安装 backend/requirements-lens.txt（CPU 版 torch 即可）。"
            )
            return self._deps_error
        self._deps_ok = True
        self._deps_error = None
        return None

    def _resolve_engine_class(self) -> Any:
        if self._engine_class is not None:
            return self._engine_class
        if self._check_deps() is not None:
            return None
        try:
            module = importlib.import_module(
                ".lens.api", package=__package__ or "xray"
            )
            self._engine_class = getattr(module, "LensEngine", None)
        except Exception as exc:  # pragma: no cover - broken install
            self._message = f"透镜引擎导入失败：{exc}"
            self._engine_class = None
        return self._engine_class

    def available(self) -> bool:
        """True when torch/transformers and the lens engine can be imported."""
        if self._check_deps() is not None:
            return False
        return self._resolve_engine_class() is not None

    def supports_generation(self) -> bool:
        """True when a local model is loaded and can generate tokens."""
        return bool(self._state == "ready" and self._engine is not None and self._generation)

    def is_ready(self) -> bool:
        return self._state == "ready" and self._engine is not None

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        dependency_error = self._check_deps()
        state = self._state
        if dependency_error is not None:
            state = "disabled"
            message = dependency_error
        else:
            message = self._message
        engine_status: dict[str, Any] = dict(self._engine_info)
        return {
            "available": dependency_error is None,
            "status": state,
            "model": self._model or self._default_model,
            "device": self._device if state == "ready" else None,
            "surrogate": False,
            "exact": bool(state == "ready"),
            "same_weights": bool(state == "ready"),
            "calibrated_layers": int(engine_status.get("calibrated_layers") or 0),
            "threads": int(engine_status.get("threads") or NATIVE_THREADS),
            "generation": bool(
                state == "ready" and self.supports_generation()
            ),
            "message": message,
        }

    def _fail(self, message: str, *, state: str = "error") -> dict[str, Any]:
        self._state = state
        self._message = str(message)
        self._generation = False
        return self.status()

    # ------------------------------------------------------------------
    # load / unload
    # ------------------------------------------------------------------

    async def load(
        self,
        model_name: str | None = None,
        device: str = "cpu",
        load_in_4bit: bool = False,
        device_map: str | None = None,
    ) -> dict[str, Any]:
        """Load a local HF causal LM (blocking work pushed to a thread)."""
        if self._load_lock is None:
            self._load_lock = asyncio.Lock()
        async with self._load_lock:
            desired = (model_name or self._default_model or "").strip()
            if self.is_ready() and not load_in_4bit:
                if not desired or desired == (self._model or ""):
                    return self.status()
                # A different local model was requested: free the old weights.
                self.unload()
            self._state = "loading"
            self._generation = False
            requested = desired
            self._model = requested or self._model
            self._device = device
            self._message = f"正在加载原生模型 {requested}（local_files_only，CPU {NATIVE_THREADS} 线程）"
            try:
                return await asyncio.to_thread(
                    self._load_blocking, requested, device, load_in_4bit, device_map
                )
            except asyncio.CancelledError:
                self._state = "unloaded"
                raise
            except Exception as exc:  # pragma: no cover - defensive
                return self._fail(f"原生通道加载失败：{exc}")

    def _load_blocking(
        self,
        model_name: str,
        device: str,
        load_in_4bit: bool,
        device_map: str | None,
    ) -> dict[str, Any]:
        dependency_error = self._check_deps()
        if dependency_error is not None:
            return self._fail(dependency_error, state="disabled")
        engine_class = self._resolve_engine_class()
        if engine_class is None:
            return self._fail(self._message or "透镜引擎不可用", state="disabled")
        if load_in_4bit:
            if not _find_spec("bitsandbytes"):
                return self._fail(
                    "请求了 4bit 加载，但未安装 bitsandbytes（当前 torch 为 CPU 版，"
                    "4bit 需要 CUDA torch + bitsandbytes）。请改用 device='cpu' 的 "
                    "float32 精确通道，或安装 bitsandbytes 后重试。",
                    state="error",
                )
            return self._fail(
                "4bit 量化尚未接入 NativeBridge：LensEngine 当前以 CPU float32 "
                "加载以保证隐藏状态/注意力精确。请去掉 load_in_4bit 再试。",
                state="error",
            )
        notes: list[str] = []
        if device_map and str(device_map).lower() not in ("cpu", "auto"):
            notes.append(f"device_map={device_map!r} 已忽略（原生通道当前固定由 LensEngine 管理设备）")
        requested_device = str(device or "auto").lower()
        if requested_device == "auto":
            try:
                import torch as _torch
                device = "cuda" if _torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
            notes.append(f"auto → {device}")
        elif not requested_device.startswith("cuda"):
            device = "cpu"
        try:
            if self._engine is None:
                if self._max_context:
                    self._engine = engine_class(
                        use_umap=self._use_umap, max_context=self._max_context
                    )
                else:
                    self._engine = engine_class(use_umap=self._use_umap)
            result = self._engine.load(
                model_name,
                device=device,
                low_priority=True,
                schedule_calibration=False,
            )
        except Exception as exc:
            return self._fail(f"原生模型加载失败（local_files_only，不会下载）：{exc}")
        if not isinstance(result, dict):
            return self._fail("透镜引擎未返回状态字典")
        if result.get("status") == "error" or not self._engine.is_ready():
            return self._fail(
                str(result.get("message") or "原生模型加载失败（local_files_only，不会自动下载）")
            )
        display = str(result.get("model") or model_name)
        self._model = display
        self._device = str(result.get("device") or getattr(self._engine, "_device", "cpu"))
        # Same weights power generation and telemetry: this is not a surrogate.
        self._engine.set_native_mode(True)
        self._engine.set_runtime_model(display)
        self._state = "ready"
        message = "原生精确通道就绪：生成与观测共用同一次前向（surrogate=false）"
        if not self._engine._tokenizer_has_chat_template():
            message += "；tokenizer 无 chat template，已使用 role: content 文本回退"
        if notes:
            message += "；" + "；".join(notes)
        self._message = message
        # Cache the immutable bits so status() never waits on the engine lock
        # (a generation step holds it for a few hundred ms on CPU).
        try:
            reported = self._engine.status()
            self._engine_info = dict(reported) if isinstance(reported, dict) else {}
            self._generation = bool(self._engine.supports_generation())
        except Exception:
            self._engine_info = {}
            self._generation = True
        return self.status()

    def unload(self) -> dict[str, Any]:
        """Release the model and reset bridge state (sync; safe to call twice)."""
        engine = self._engine
        self._engine = None
        self._state = "unloaded"
        self._device = None
        self._prompt_tokens = 0
        self._engine_info = {}
        self._generation = False
        self._message = "原生精确通道已卸载"
        if engine is not None:
            try:
                engine.unload()
            except Exception as exc:  # pragma: no cover - defensive
                self._message = f"原生通道卸载失败：{exc}"
        return self.status()

    # ------------------------------------------------------------------
    # inference (blocking calls pushed to threads)
    # ------------------------------------------------------------------

    async def prefill(
        self,
        messages: list[dict[str, str]] | str,
        *,
        max_context: int | None = None,
        system: str | None = None,
    ) -> dict[str, Any]:
        """Run one forward over the chat prompt and keep the KV cache."""
        engine = self._engine
        if engine is None or self._state != "ready":
            return {
                "ok": False,
                "error": "原生通道未就绪：请先 load(本地 HF 模型路径)",
                "status": self._state,
            }
        try:
            result = await asyncio.to_thread(
                engine.prefill,
                messages,
                max_context=max_context,
                system=system,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            return {"ok": False, "error": f"prefill 失败：{exc}"}
        if isinstance(result, dict) and result.get("ok"):
            self._prompt_tokens = int(result.get("tokens") or 0)
        return result if isinstance(result, dict) else {"ok": False, "error": "prefill 无返回"}

    async def sample_next(
        self, params: SamplingParams | dict[str, Any] | None, step: int = 0
    ) -> dict[str, Any]:
        """Sample one token; probe/text/topk all come from the same forward."""
        engine = self._engine
        if engine is None or self._state != "ready":
            return {
                "ok": False,
                "error": "原生通道未就绪：请先 load + prefill",
                "status": self._state,
            }
        settings = (
            params
            if isinstance(params, SamplingParams)
            else SamplingParams.from_dict(params if isinstance(params, dict) else None)
        )
        try:
            result = await asyncio.to_thread(engine.sample_next, settings, int(step))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            return {"ok": False, "error": f"sample_next 失败：{exc}"}
        return result if isinstance(result, dict) else {"ok": False, "error": "sample_next 无返回"}

    async def reset(self, prompt_text: str = "", system: str = "") -> dict[str, Any]:
        """Clear KV cache/probes; with text, seed the legacy text reset instead."""
        engine = self._engine
        if engine is None or self._state != "ready":
            return {"ok": False, "status": self._state, "message": "原生通道未就绪"}
        try:
            if prompt_text:
                result = await asyncio.to_thread(engine.reset, prompt_text, system)
            else:
                result = await asyncio.to_thread(engine.reset_generation)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            return {"ok": False, "error": f"reset 失败：{exc}"}
        self._prompt_tokens = 0
        return result if isinstance(result, dict) else {"ok": True}

    # ------------------------------------------------------------------
    # optional extras for the UI
    # ------------------------------------------------------------------

    def eos_token_ids(self) -> list[int]:
        """EOS ids declared by the loaded tokenizer/config (generation stop)."""
        engine = self._engine
        if engine is None:
            return []
        try:
            return list(engine.eos_token_ids())
        except Exception:
            return []

    def decode_token(self, token_id: int) -> str:
        """Decoded text for one id (token labels / debugging)."""
        engine = self._engine
        if engine is None:
            return ""
        try:
            return str(engine.decode_token(int(token_id)))
        except Exception:
            return ""

    @property
    def engine(self) -> Any:
        """The underlying ``LensEngine`` (advanced integration / tests)."""
        return self._engine

    def projection(self) -> dict[str, Any] | None:
        """Current 3D projection payload (surrogate-free: real token states)."""
        engine = self._engine
        if engine is None or self._state != "ready":
            return None
        try:
            projection = engine.projection()
            return projection if isinstance(projection, dict) else None
        except Exception:
            return None

    @property
    def prompt_tokens(self) -> int:
        return int(self._prompt_tokens)


__all__ = ["NativeBridge", "DEFAULT_MODEL", "NATIVE_THREADS"]
