"""FastAPI application: REST, WebSocket event protocol and static front end.

Event contract: docs/SPEC.md §4 and apps/desktop/src/types.ts.
Design rules that matter for this file:

* one WebSocket = one UI session; several connections may coexist but each
  connection runs at most one generation;
* every outbound frame goes through a per-connection lock;
* no model is ever loaded implicitly — only ``POST /v1/...`` generation starts
  a request, and only after the client sends ``{"type":"start"}``;
* the demo mode touches neither LM Studio nor the lens package;
* failures are emitted as ``{"type":"error", code, message, recoverable}``.
"""

from __future__ import annotations

import asyncio
import logging
import math
import mimetypes
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import DEFAULT_SAMPLING, PROJECT_ROOT, Settings, get_settings
from .demo import DemoStream, demo_text
from .lens_bridge import LensBridge
from .lmstudio import (
    LMStudioClient,
    LMStudioError,
    LMStudioHTTPError,
    LMStudioStreamError,
    LMStudioUnreachable,
    StreamEvent,
)
from .metrics import MetricsCollector
from .rag import RAGStore
from .sampling import SamplingParams, approximate_prompt_tokens, entropy_of
from .store import SessionStore, now_ms

try:  # Optional in-process exact runtime; the core still runs without it.
    from .native_bridge import NativeBridge
except Exception:  # pragma: no cover - broken optional install
    NativeBridge = None  # type: ignore[assignment]

logger = logging.getLogger("xray.app")

METRICS_EVERY_S = 1.2
PROBE_EVERY_TOKENS = 2
PROJECTION_EVERY_TOKENS = 8
MAX_STEP_QUERY_LIMIT = 2000
#: How many initial LM Studio tokens are inspected before warning about empty top-k.
LOGPROBS_PROBE_STEPS = 5
NO_LOGPROBS_MESSAGE = "LM Studio 未返回 top_logprobs；概率云为空，可切换原生精确模式"


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class RagSearchRequest(BaseModel):
    query: str = ""
    top_k: int = 3


class LensLoadRequest(BaseModel):
    model: str | None = None
    device: str = "cpu"


class NativeLoadRequest(BaseModel):
    model: str | None = None
    device: str = "cpu"
    load_in_4bit: bool = False
    device_map: str | None = None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _error(code: str, message: str, recoverable: bool = True) -> dict[str, Any]:
    return {"type": "error", "code": code, "message": message, "recoverable": bool(recoverable)}


def _public_step(step: dict[str, Any]) -> dict[str, Any]:
    """Strip internal keys and round floats for the ``token`` event."""
    def rounded(value: Any, digits: int) -> Any:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return round(float(value), digits)
        return value

    return {
        "index": int(step.get("index") or 0),
        "delta": str(step.get("delta") or ""),
        "kind": step.get("kind") if step.get("kind") in ("answer", "reasoning") else "answer",
        "topk": list(step.get("topk") or []),
        "chosen_logprob": rounded(step.get("chosen_logprob"), 6),
        "chosen_prob": rounded(step.get("chosen_prob"), 6),
        "entropy": rounded(step.get("entropy"), 4),
        "at": rounded(step.get("at"), 2),
        "ttft_ms": rounded(step.get("ttft_ms"), 1),
        "tps": rounded(step.get("tps"), 2),
        "generated": int(step.get("generated") or 0),
        "text_len": int(step.get("text_len") or 0),
    }


def _entropy_from_topk(topk: list[dict[str, Any]]) -> float | None:
    if not topk:
        return None
    probs: list[float] = []
    for item in topk:
        prob = item.get("prob")
        if not isinstance(prob, (int, float)):
            logprob = item.get("logprob")
            prob = math.exp(float(logprob)) if isinstance(logprob, (int, float)) else 0.0
        probs.append(max(0.0, min(1.0, float(prob))))
    hidden = max(0.0, 1.0 - sum(probs))
    return entropy_of(probs + ([hidden] if hidden > 0 else []))


def _build_step(
    index: int,
    *,
    delta: str,
    kind: str,
    topk: list[dict[str, Any]],
    chosen_logprob: float | None,
    chosen_prob: float | None,
    at_ms: float,
    ttft_ms: float | None,
    tps: float | None,
    generated: int,
    text_len: int,
) -> dict[str, Any]:
    return _public_step(
        {
            "index": index,
            "delta": delta,
            "kind": kind,
            "topk": topk,
            "chosen_logprob": chosen_logprob,
            "chosen_prob": chosen_prob,
            "entropy": _entropy_from_topk(topk),
            "at": at_ms,
            "ttft_ms": ttft_ms,
            "tps": tps,
            "generated": generated,
            "text_len": text_len,
        }
    )


def _normalise_messages(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    messages: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user").strip().lower()
        if role not in ("system", "user", "assistant"):
            role = "user"
        content = item.get("content")
        if isinstance(content, list):
            content = "".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        messages.append({"role": role, "content": content})
    return messages


def _last_user_text(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content") or ""
    return messages[-1]["content"] if messages else ""


def _flatten_messages(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(f"{message['role']}: {message['content']}" for message in messages)


def _inject_memory(system: str | None, items: list[dict[str, Any]]) -> str:
    block = "\n\n".join(
        f"[记忆 {index + 1} · {item.get('source', '')}] {item.get('text', '')}"
        for index, item in enumerate(items)
    )
    header = "以下内容来自本机记忆库，仅作参考：\n\n"
    body = header + block
    if system:
        return f"{system.rstrip()}\n\n{body}"
    return body


async def _lens_status(app: FastAPI) -> dict[str, Any]:
    """Read lens status off the event loop (importing the optional package can be slow)."""
    bridge: LensBridge = app.state.lens
    try:
        return await asyncio.wait_for(asyncio.to_thread(bridge.status), timeout=4.0)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        return {
            "status": "loading",
            "model": app.state.settings.lens_model,
            "device": None,
            "surrogate": True,
            "same_weights": False,
            "calibrated_layers": 0,
            "message": "透镜模块初始化超时",
            "threads": 2,
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "status": "disabled",
            "model": app.state.settings.lens_model,
            "device": None,
            "surrogate": True,
            "same_weights": False,
            "calibrated_layers": 0,
            "message": f"透镜状态不可用: {exc}",
            "threads": 2,
        }


def _disabled_native_status(message: str = "原生精确通道不可用（缺少 backend/xray/native_bridge.py 或 torch/transformers）") -> dict[str, Any]:
    return {
        "available": False,
        "status": "disabled",
        "model": None,
        "device": None,
        "surrogate": False,
        "exact": False,
        "same_weights": False,
        "calibrated_layers": 0,
        "threads": 2,
        "generation": False,
        "message": message,
    }


async def _native_status(app: FastAPI) -> dict[str, Any]:
    """Native (exact) runtime status, read off the event loop."""
    bridge = getattr(app.state, "native", None)
    if bridge is None:
        return _disabled_native_status()
    try:
        status = await asyncio.wait_for(asyncio.to_thread(bridge.status), timeout=5.0)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        status = _disabled_native_status("原生精确通道状态读取超时")
        status["status"] = "error"
        return status
    except Exception as exc:  # pragma: no cover - defensive
        status = _disabled_native_status(f"原生精确通道状态不可用: {exc}")
        status["status"] = "error"
        return status
    if not isinstance(status, dict):
        return _disabled_native_status()
    # Contract with types.ts: exact only when the same weights are loaded.
    status["surrogate"] = False
    status["exact"] = bool(status.get("status") == "ready" and status.get("generation"))
    return status


async def _refresh_models(app: FastAPI, *, force: bool = False, timeout: float = 2.5) -> dict[str, Any]:
    """Cached ``GET /api/v0/models`` (metadata only — never loads a model)."""
    cache = app.state.models_cache
    now = time.monotonic()
    ttl = 15.0 if cache.get("reachable") else 5.0
    if not force and cache.get("at") and now - cache["at"] < ttl:
        return cache["data"]
    client: LMStudioClient = app.state.lmstudio
    result = await client.ping(timeout=timeout)
    app.state.models_cache = {"at": time.monotonic(), "reachable": result["reachable"], "data": result}
    return result


async def _build_config(app: FastAPI) -> dict[str, Any]:
    settings: Settings = app.state.settings
    lens = await _lens_status(app)
    native = await _native_status(app)
    metrics: MetricsCollector = app.state.metrics
    lens_usable = lens.get("status") != "disabled"
    native_usable = bool(native.get("available"))
    return {
        "version": settings.version,
        "lmstudio_url": settings.lmstudio_base_url,
        "default_model": settings.model,
        "default_mode": "lmstudio",
        "lens": lens,
        "native": native,
        "capabilities": {
            "logprobs": True,
            "attention": lens_usable or native_usable,
            "lens": lens_usable,
            "rag": True,
            "gpu": bool(metrics.gpu_available),
        },
        "exact_available": native_usable,
        "sampling_defaults": dict(DEFAULT_SAMPLING),
        "demo_available": True,
    }


async def _build_hello(app: FastAPI) -> dict[str, Any]:
    settings: Settings = app.state.settings
    models = await _refresh_models(app)
    config = await _build_config(app)
    return {
        "type": "hello",
        "version": settings.version,
        "config": config,
        "lmstudio": {
            "reachable": models.get("reachable", False),
            "models": models.get("models", []),
            "error": models.get("error"),
        },
        "lens": config["lens"],
        "native": config["native"],
        "demo_text": demo_text(),
    }


# ---------------------------------------------------------------------------
# Generation bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class GenerationHandle:
    session_id: str
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None
    demo: DemoStream | None = None
    connection: "Connection | None" = None


class ModelNotLoaded(LMStudioError):
    pass


class NativeRuntimeError(RuntimeError):
    """Failure of the in-process exact (native) runtime."""


# ---------------------------------------------------------------------------
# WebSocket connection
# ---------------------------------------------------------------------------


class Connection:
    """One UI connection: receives commands, streams generation events."""

    def __init__(self, websocket: WebSocket, app: FastAPI) -> None:
        self.ws = websocket
        self.app = app
        self._send_lock = asyncio.Lock()
        self.closed = False
        self.gen_task: asyncio.Task | None = None
        self.handle: GenerationHandle | None = None
        # Set to False once the first LM Studio tokens reveal missing top-logprobs.
        self._logprobs_available: bool | None = None

    # -- sending --------------------------------------------------------
    async def send(self, event: dict[str, Any]) -> bool:
        if self.closed:
            return False
        async with self._send_lock:
            if self.closed:
                return False
            try:
                await self.ws.send_json(event)
                return True
            except (WebSocketDisconnect, RuntimeError, OSError):
                self.closed = True
                return False

    # -- main loop ------------------------------------------------------
    async def run(self) -> None:
        try:
            await self.send(await _build_hello(self.app))
            while not self.closed:
                try:
                    message = await self.ws.receive_json()
                except (WebSocketDisconnect, RuntimeError):
                    break
                if not isinstance(message, dict):
                    continue
                kind = message.get("type")
                if kind == "ping":
                    await self.send({"type": "pong", "at": now_ms()})
                elif kind == "start":
                    await self.start(message.get("payload") or {})
                elif kind == "stop":
                    await self.stop_generation()
                else:
                    await self.send(_error("UNKNOWN_MESSAGE", f"未知消息类型: {kind!r}", True))
        except WebSocketDisconnect:
            pass
        except Exception:  # pragma: no cover - a broken socket must not kill the app
            logger.exception("WebSocket connection failed")
        finally:
            await self.cleanup()

    # -- commands -------------------------------------------------------
    async def start(self, payload: Any) -> None:
        if self.gen_task is not None and not self.gen_task.done():
            await self.send(_error("BUSY", "当前连接已有生成在进行中", True))
            return
        if not isinstance(payload, dict):
            await self.send(_error("INVALID_PAYLOAD", "start 载荷必须是对象", True))
            return
        messages = _normalise_messages(payload.get("messages"))
        if not messages:
            await self.send(_error("INVALID_PAYLOAD", "messages 不能为空", True))
            return
        settings: Settings = self.app.state.settings
        store: SessionStore = self.app.state.store
        rag: RAGStore = self.app.state.rag
        raw_mode = payload.get("mode")
        mode = raw_mode if raw_mode in ("lmstudio", "demo", "native") else "lmstudio"
        native_status: dict[str, Any] = {}
        if mode == "native":
            native_status = await _native_status(self.app)
            model = str(payload.get("model") or "").strip() or str(native_status.get("model") or "")
            if not model:
                model = settings.lens_model
        else:
            model = str(payload.get("model") or settings.model)
        sampling = SamplingParams.from_dict(payload.get("sampling"))
        system = payload.get("system")
        system = str(system) if isinstance(system, str) and system.strip() else None
        rag_conf = payload.get("rag") if isinstance(payload.get("rag"), dict) else {}
        lens_conf = payload.get("lens") if isinstance(payload.get("lens"), dict) else {}
        lens_status = await _lens_status(self.app)
        lens_ready = lens_status.get("status") == "ready"
        lens_requested = bool(lens_conf.get("enabled"))
        if mode == "demo":
            capabilities = {"logprobs": True, "attention": True, "lens": True}
        elif mode == "native":
            capabilities = {
                "logprobs": True,
                "attention": True,
                "lens": True,
                "exact": True,
                "surrogate": False,
                "runtime": "native",
            }
            lens_requested = False  # native probes come from the same forward pass
        else:
            capabilities = {
                "logprobs": True,
                "attention": lens_ready and lens_requested,
                "lens": lens_ready and lens_requested,
                "exact": False,
                "surrogate": bool(lens_ready and lens_requested),
                "runtime": "lmstudio",
            }
        if mode == "native":
            active_native = getattr(self.app.state, "native_generation", None)
            if active_native is not None:
                await self.send(
                    _error("BUSY", "原生精确通道正在被另一个会话使用，请等待其结束或先停止", True)
                )
                return
            if NativeBridge is None or not native_status.get("available"):
                await self.send(
                    _error(
                        "NATIVE_UNAVAILABLE",
                        str(native_status.get("message") or "原生精确通道不可用：缺少 torch/transformers 或 native_bridge"),
                        True,
                    )
                )
                return
        stored_messages = list(messages)
        if system:
            stored_messages = [{"role": "system", "content": system}] + stored_messages
        meta = store.create_session(
            {
                "mode": mode,
                "model": model,
                "started_at": now_ms(),
                "capabilities": capabilities,
                "sampling": sampling.to_dict(),
                "rag_enabled": bool(rag_conf.get("enabled")),
                "lens_enabled": bool(lens_requested),
            },
            stored_messages,
        )
        if mode == "native":
            self.app.state.native_generation = meta["id"]
        self.handle = GenerationHandle(session_id=meta["id"], connection=self)
        self.app.state.generations[meta["id"]] = self.handle
        await self.send({"type": "started", "session": meta})

        rag_items: list[dict[str, Any]] = []
        injected = False
        if rag_conf.get("enabled"):
            query = _last_user_text(messages)
            try:
                rag_items = await rag.search(query, int(rag_conf.get("top_k") or 3))
            except Exception:  # pragma: no cover - retrieval must never block chat
                logger.debug("RAG search failed", exc_info=True)
                rag_items = []
            injected = bool(rag_conf.get("inject") and rag_items)
            if injected:
                system = _inject_memory(system, rag_items)
            await self.send(
                {
                    "type": "rag",
                    "items": rag_items,
                    "query": query,
                    "injected": injected,
                }
            )
        if lens_requested and not lens_ready:
            await self.send(
                {
                    "type": "status",
                    "scope": "lens",
                    "state": lens_status.get("status") or "unloaded",
                    "message": "透镜未就绪：请在设置页加载替代透镜模型。实时通道与演示不受影响。",
                }
            )
        self.gen_task = asyncio.create_task(
            self._generate(
                meta=meta,
                mode=mode,
                model=model,
                messages=messages,
                sampling=sampling,
                system=system,
                lens_ready=lens_ready and lens_requested,
                rag_items=rag_items,
            ),
            name=f"xray-generation-{meta['id']}",
        )
        self.app.state.active_tasks.add(self.gen_task)
        self.gen_task.add_done_callback(self.app.state.active_tasks.discard)

    async def stop_generation(self) -> None:
        handle = self.handle
        if handle is None:
            await self.send({"type": "status", "scope": "generation", "state": "idle", "message": "没有正在进行的生成"})
            return
        handle.stop.set()
        if handle.demo is not None:
            handle.demo.stop()
        await self.send({"type": "status", "scope": "generation", "state": "stopping", "message": "正在停止生成"})

    # -- generation drivers --------------------------------------------
    async def _generate(
        self,
        *,
        meta: dict[str, Any],
        mode: str,
        model: str,
        messages: list[dict[str, str]],
        sampling: SamplingParams,
        system: str | None,
        lens_ready: bool,
        rag_items: list[dict[str, Any]],
    ) -> None:
        store: SessionStore = self.app.state.store
        metrics: MetricsCollector = self.app.state.metrics
        metrics.reset_generation()
        self._logprobs_available = None
        started = time.monotonic()
        finish_reason = "stop"
        usage: dict[str, Any] | None = None
        cancelled = False
        try:
            if mode == "demo":
                finish_reason, usage = await self._run_demo(meta, sampling, started, messages)
            elif mode == "native":
                finish_reason, usage = await self._run_native(meta, model, messages, sampling, system, started)
            else:
                finish_reason, usage = await self._run_lmstudio(meta, model, messages, sampling, system, lens_ready, started)
        except ModelNotLoaded as exc:
            finish_reason = "model_not_loaded"
            await self.send(_error("MODEL_NOT_LOADED", str(exc), True))
        except NativeRuntimeError as exc:
            finish_reason = "error"
            await self.send(_error("NATIVE_RUNTIME_ERROR", str(exc), True))
        except LMStudioUnreachable as exc:
            finish_reason = "unreachable"
            await self.send(_error("LMSTUDIO_UNREACHABLE", str(exc), True))
        except LMStudioHTTPError as exc:
            finish_reason = "http_error"
            await self.send(
                _error("LMSTUDIO_HTTP_ERROR", f"{exc.status_code}: {exc}", True)
            )
        except LMStudioStreamError as exc:
            finish_reason = "error"
            await self.send(_error("LMSTUDIO_STREAM_ERROR", str(exc), True))
        except asyncio.CancelledError:
            cancelled = True
            finish_reason = "cancelled"
            try:
                store.finish_session(meta["id"], None, "cancelled", time.monotonic() - started)
            except Exception:  # pragma: no cover - defensive
                pass
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("generation failed")
            finish_reason = "error"
            await self.send(_error("INTERNAL", f"生成失败: {exc}", True))
        finally:
            if not cancelled:
                elapsed = time.monotonic() - started
                if usage is None:
                    usage = self._estimate_usage(store, meta["id"], messages)
                try:
                    await asyncio.wait_for(
                        asyncio.shield(
                            asyncio.to_thread(store.finish_session, meta["id"], usage, finish_reason, elapsed)
                        ),
                        timeout=10.0,
                    )
                except (asyncio.TimeoutError, Exception):  # noqa: B014 - persistence is best effort
                    try:
                        store.finish_session(meta["id"], usage, finish_reason, elapsed)
                    except Exception:
                        pass
                await self.send(
                    {
                        "type": "done",
                        "usage": usage,
                        "finish_reason": finish_reason,
                        "elapsed_s": round(elapsed, 3),
                        **self._done_extra(mode),
                    }
                )
                await self.send({"type": "metrics", "metrics": metrics.snapshot()})
            if mode == "native":
                self.app.state.native_generation = None
            self.app.state.generations.pop(meta["id"], None)
            self.handle = None
            self.gen_task = None

    def _done_extra(self, mode: str) -> dict[str, Any]:
        """Extra capability flags on ``done`` (exact channel / missing logprobs)."""
        if mode == "native":
            return {
                "capabilities": {
                    "logprobs": True,
                    "attention": True,
                    "lens": True,
                    "exact": True,
                    "surrogate": False,
                    "runtime": "native",
                },
                "surrogate": False,
                "exact": True,
            }
        if mode == "lmstudio" and self._logprobs_available is False:
            return {"capabilities": {"logprobs": False}}
        return {}

    def _estimate_usage(self, store: SessionStore, session_id: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        state = store.get(session_id)
        completion = state.tokens if state is not None else 0
        prompt = sum(approximate_prompt_tokens(message.get("content") or "") for message in messages) + 8
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }

    # -- demo driver ----------------------------------------------------
    async def _run_demo(
        self,
        meta: dict[str, Any],
        sampling: SamplingParams,
        started: float,
        messages: list[dict[str, str]],
    ) -> tuple[str, dict[str, Any] | None]:
        store: SessionStore = self.app.state.store
        metrics: MetricsCollector = self.app.state.metrics
        settings: Settings = self.app.state.settings
        demo = DemoStream(sampling, lens_model=settings.lens_model, prompt_text=_last_user_text(messages))
        if self.handle is not None:
            self.handle.demo = demo
        last_metrics = time.monotonic()
        generated = 0
        finish_reason = "stop"
        async for event in demo:
            if self.handle is None or self.handle.stop.is_set():
                demo.stop()
                break
            event_type = event.get("type")
            if event_type == "token":
                step = _public_step(event["step"])
                store.append_step(meta["id"], step)
                generated += 1
                metrics.note_token()
                metrics.update_generation(generated, time.monotonic() - started, step.get("ttft_ms"))
                await self.send({"type": "token", "step": step})
                now = time.monotonic()
                if now - last_metrics >= METRICS_EVERY_S:
                    await self.send({"type": "metrics", "metrics": metrics.snapshot()})
                    last_metrics = now
            elif event_type == "probe":
                probe = event.get("probe")
                if isinstance(probe, dict):
                    store.attach_probe(meta["id"], int(event.get("step") or 0), probe)
                    await self.send({"type": "probe", "step": int(event.get("step") or 0), "probe": probe})
                    if isinstance(probe.get("kv"), dict):
                        await self.send({"type": "kv", "kv": probe["kv"]})
            elif event_type == "projection":
                projection = event.get("projection") or {}
                await self.send(
                    {
                        "type": "projection",
                        "projection": projection,
                        "points": projection.get("points", event.get("points", [])),
                        "reducer": projection.get("reducer", event.get("reducer", "deterministic")),
                        "fitted_at": projection.get("fitted_at", event.get("fitted_at", 0)),
                    }
                )
        if self.handle is not None and self.handle.stop.is_set():
            finish_reason = "stopped"
        elif demo.stopped:
            finish_reason = "stopped"
        if demo.token_count:
            metrics.update_generation(demo.token_count, time.monotonic() - started)
        return finish_reason, None

    # -- native exact driver --------------------------------------------
    async def _run_native(
        self,
        meta: dict[str, Any],
        model: str,
        messages: list[dict[str, str]],
        sampling: SamplingParams,
        system: str | None,
        started: float,
    ) -> tuple[str, dict[str, Any] | None]:
        """Generate with the in-process exact runtime: text + tensors, one forward.

        No lens worker is involved here: the probe is produced by the same
        forward pass that samples the token (``surrogate=false``).
        """
        store: SessionStore = self.app.state.store
        metrics: MetricsCollector = self.app.state.metrics
        bridge = getattr(self.app.state, "native", None)
        if bridge is None:
            raise NativeRuntimeError("原生精确通道不可用：native_bridge 未安装")
        if not bridge.available():
            raise NativeRuntimeError("原生精确通道不可用：需要本机安装 torch 与 transformers")

        # -- load / switch model (local_files_only: never downloads) -----
        native_device = os.environ.get("XRAY_NATIVE_DEVICE", "auto")
        if not bridge.is_ready():
            await self.send(
                {
                    "type": "status",
                    "scope": "native",
                    "state": "loading",
                    "message": "正在加载原生精确模型（本地权重，不会下载；生成与观测共用同一次前向）…",
                }
            )
            status = await bridge.load(model or None, native_device)
            if not isinstance(status, dict) or status.get("status") != "ready":
                message = str((status or {}).get("message") or "原生模型加载失败")
                await self.send({"type": "status", "scope": "native", "state": "error", "message": message})
                raise NativeRuntimeError(message)
            await self.send(
                {
                    "type": "status",
                    "scope": "native",
                    "state": "ready",
                    "message": str(status.get("message") or "原生精确通道就绪"),
                }
            )
        else:
            current = bridge.status()
            current_model = str(current.get("model") or "")
            target = (model or "").strip()
            if target and current_model and target != current_model and Path(target).name != Path(current_model).name:
                await self.send(
                    {
                        "type": "status",
                        "scope": "native",
                        "state": "loading",
                        "message": f"切换原生精确模型到 {target}…",
                    }
                )
                await asyncio.to_thread(bridge.unload)
                status = await bridge.load(target, native_device)
                if not isinstance(status, dict) or status.get("status") != "ready":
                    message = str((status or {}).get("message") or "原生模型加载失败")
                    await self.send({"type": "status", "scope": "native", "state": "error", "message": message})
                    raise NativeRuntimeError(message)
        if self.handle is not None and self.handle.stop.is_set():
            return "stopped", None

        # -- prefill ------------------------------------------------------
        await self.send(
            {
                "type": "status",
                "scope": "native",
                "state": "prefill",
                "message": "正在编码提示词并建立 KV cache…",
            }
        )
        prefill_started = time.monotonic()
        prefill = await bridge.prefill(messages, system=system)
        if not isinstance(prefill, dict) or not prefill.get("ok"):
            message = str(
                (prefill or {}).get("error") or (prefill or {}).get("message") or "prefill 失败"
            )
            await self.send({"type": "status", "scope": "native", "state": "error", "message": message})
            raise NativeRuntimeError(message)
        prompt_tokens = int(prefill.get("tokens") or bridge.prompt_tokens or 0)
        metrics.set_context(prompt_tokens)
        await self.send(
            {
                "type": "status",
                "scope": "native",
                "state": "generating",
                "message": f"原生精确生成中 · prompt {prompt_tokens} tokens · surrogate=false",
            }
        )

        # -- token loop ---------------------------------------------------
        generation_started = time.monotonic()
        max_tokens = max(1, int(sampling.max_tokens))
        finish_reason = "stop"
        generated = 0
        text_len = 0
        ttft_ms: float | None = None
        last_metrics = time.monotonic()
        try:
            for step_index in range(max_tokens):
                if self.handle is None or self.handle.stop.is_set():
                    finish_reason = "stopped"
                    break
                out = await bridge.sample_next(sampling, step_index)
                if not isinstance(out, dict) or not out.get("ok"):
                    raise NativeRuntimeError(
                        str(
                            (out or {}).get("error")
                            or (out or {}).get("message")
                            or "sample_next 失败"
                        )
                    )
                if out.get("eos"):
                    finish_reason = "stop"
                    break
                now = time.monotonic()
                stream_elapsed = max(0.001, now - generation_started)
                session_elapsed = max(0.001, now - started)
                if ttft_ms is None:
                    # Include prompt prefill; model-load time is reported separately.
                    ttft_ms = max(0.001, now - prefill_started) * 1000.0
                delta = str(out.get("text") or "")
                generated += 1
                text_len += len(delta)
                step = _build_step(
                    step_index,
                    delta=delta,
                    kind="answer",
                    topk=list(out.get("topk") or []),
                    chosen_logprob=out.get("chosen_logprob"),
                    chosen_prob=out.get("chosen_prob"),
                    at_ms=session_elapsed * 1000.0,
                    ttft_ms=ttft_ms,
                    tps=generated / stream_elapsed,
                    generated=generated,
                    text_len=text_len,
                )
                entropy = out.get("entropy")
                if isinstance(entropy, (int, float)) and not isinstance(entropy, bool):
                    step["entropy"] = round(float(entropy), 4)
                store.append_step(meta["id"], step)
                await self.send({"type": "token", "step": step})

                probe = out.get("probe") if isinstance(out.get("probe"), dict) else None
                if probe is not None:
                    probe = dict(probe)
                    probe.setdefault("step", step_index)
                    probe["surrogate"] = False
                    probe["same_weights"] = True
                    store.attach_probe(meta["id"], step_index, probe)
                    await self.send({"type": "probe", "step": step_index, "probe": probe})
                    if isinstance(probe.get("kv"), dict):
                        await self.send({"type": "kv", "kv": probe["kv"]})

                if step_index % PROJECTION_EVERY_TOKENS == 0:
                    projection = bridge.projection()
                    if isinstance(projection, dict):
                        await self.send(
                            {
                                "type": "projection",
                                "projection": projection,
                                "points": projection.get("points", []),
                                "reducer": projection.get("reducer", "deterministic"),
                                "fitted_at": projection.get("fitted_at", 0),
                            }
                        )
                metrics.note_token()
                metrics.update_generation(generated, stream_elapsed, ttft_ms)
                if now - last_metrics >= METRICS_EVERY_S:
                    await self.send({"type": "metrics", "metrics": metrics.snapshot()})
                    last_metrics = now
        finally:
            try:
                await bridge.reset()
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - defensive
                pass
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": generated,
            "total_tokens": prompt_tokens + generated,
        }
        return finish_reason, usage

    # -- LM Studio driver ----------------------------------------------
    async def _run_lmstudio(
        self,
        meta: dict[str, Any],
        model: str,
        messages: list[dict[str, str]],
        sampling: SamplingParams,
        system: str | None,
        lens_enabled: bool,
        started: float,
    ) -> tuple[str, dict[str, Any] | None]:
        store: SessionStore = self.app.state.store
        metrics: MetricsCollector = self.app.state.metrics
        settings: Settings = self.app.state.settings
        client: LMStudioClient = self.app.state.lmstudio
        lens: LensBridge = self.app.state.lens

        models = await _refresh_models(self.app)
        if not models.get("reachable"):
            raise LMStudioUnreachable(
                "无法连接 LM Studio（" + settings.lmstudio_base_url + "）。请确认 LM Studio 本地服务已启动。"
            )
        entry = next((item for item in models.get("models", []) if item.get("id") == model), None)
        if entry is not None and entry.get("state") == "not-loaded":
            raise ModelNotLoaded(
                f"模型 {model} 当前未加载。为避免卡顿，后端不会自动加载模型；"
                "请先在 LM Studio 中加载该模型，然后重新发送。"
            )
        metrics.set_context(sum(approximate_prompt_tokens(message["content"]) for message in messages))

        queue: asyncio.Queue | None = None
        lens_task: asyncio.Task | None = None
        if lens_enabled and lens.is_ready():
            await lens.reset(_flatten_messages(messages), system or "")
            queue = asyncio.Queue(maxsize=2)
            lens_task = asyncio.create_task(self._lens_worker(queue, meta["id"], lens), name="xray-lens-worker")
        elif lens_enabled:
            lens_enabled = False

        stream = client.stream(
            messages,
            sampling,
            model=model,
            provider=settings.provider,
            system=system,
        )
        step_index = 0
        generated = 0
        text_len = 0
        answer_steps = 0
        empty_topk_answer_steps = 0
        ttft_ms: float | None = None
        usage: dict[str, Any] | None = None
        finish_reason = "stop"
        last_metrics = time.monotonic()
        try:
            async for event in stream:
                if self.handle is None or self.handle.stop.is_set():
                    finish_reason = "stopped"
                    break
                if event.kind == "meta":
                    provider_name = (event.meta or {}).get("provider", "unknown")
                    await self.send(
                        {
                            "type": "status",
                            "scope": "runtime",
                            "state": "streaming",
                            "message": f"LM Studio · {provider_name} · top-logprobs "
                            + ("可用" if (event.meta or {}).get("supports_logprobs") else "不可用"),
                        }
                    )
                    continue
                if event.kind == "fallback":
                    meta_info = event.meta or {}
                    await self.send(
                        {
                            "type": "status",
                            "scope": "runtime",
                            "state": "degraded",
                            "message": f"{meta_info.get('from')} 不受支持，已降级到 {meta_info.get('to')}",
                        }
                    )
                    continue
                if event.kind == "usage":
                    usage = event.usage
                    continue
                if event.kind == "finish":
                    if event.finish_reason:
                        finish_reason = event.finish_reason
                    continue
                if event.kind == "error":
                    raise LMStudioStreamError(event.message or "LM Studio 流返回错误")
                if event.kind != "delta":
                    continue
                now = time.monotonic()
                elapsed_s = max(0.001, now - started)
                if ttft_ms is None:
                    ttft_ms = elapsed_s * 1000.0
                text_len += len(event.text)
                generated += 1
                step = _build_step(
                    step_index,
                    delta=event.text,
                    kind="reasoning" if event.token_kind == "reasoning" else "answer",
                    topk=event.topk,
                    chosen_logprob=event.chosen_logprob,
                    chosen_prob=event.chosen_prob,
                    at_ms=elapsed_s * 1000.0,
                    ttft_ms=ttft_ms,
                    tps=generated / elapsed_s,
                    generated=generated,
                    text_len=text_len,
                )
                store.append_step(meta["id"], step)
                await self.send({"type": "token", "step": step})
                # LM Studio sometimes accepts include=logprobs but returns none.
                # After a few answer tokens without any top-k, tell the UI and
                # mark the persisted session capability as logprobs=false.
                if step["kind"] == "answer":
                    if step["topk"]:
                        if self._logprobs_available is None:
                            self._logprobs_available = True
                    else:
                        empty_topk_answer_steps += 1
                    answer_steps += 1
                    if (
                        self._logprobs_available is None
                        and answer_steps >= LOGPROBS_PROBE_STEPS
                        and empty_topk_answer_steps >= LOGPROBS_PROBE_STEPS
                    ):
                        self._logprobs_available = False
                        try:
                            state = store.get(meta["id"])
                            if state is not None:
                                state.meta.setdefault("capabilities", {})["logprobs"] = False
                        except Exception:  # pragma: no cover - defensive
                            pass
                        await self.send(
                            {
                                "type": "status",
                                "scope": "runtime",
                                "state": "no_logprobs",
                                "message": NO_LOGPROBS_MESSAGE,
                            }
                        )
                metrics.note_token()
                metrics.update_generation(generated, elapsed_s, ttft_ms)
                if queue is not None and event.text and step_index % PROBE_EVERY_TOKENS == 0:
                    while True:
                        try:
                            queue.put_nowait((step_index, event.text))
                            break
                        except asyncio.QueueFull:
                            try:
                                queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                if now - last_metrics >= METRICS_EVERY_S:
                    await self.send({"type": "metrics", "metrics": metrics.snapshot()})
                    last_metrics = now
                step_index += 1
        finally:
            try:
                await stream.aclose()
            except Exception:  # pragma: no cover - defensive
                pass
            if queue is not None:
                try:
                    queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass
            if lens_task is not None:
                try:
                    await asyncio.wait_for(lens_task, timeout=4.0)
                except asyncio.TimeoutError:
                    lens_task.cancel()
                except asyncio.CancelledError:
                    raise
                except Exception:  # pragma: no cover - defensive
                    pass
        return finish_reason, usage

    async def _lens_worker(self, queue: asyncio.Queue, session_id: str, lens: LensBridge) -> None:
        store: SessionStore = self.app.state.store
        metrics: MetricsCollector = self.app.state.metrics
        while True:
            item = await queue.get()
            if item is None:
                return
            step_index, delta = item
            started = time.monotonic()
            try:
                probes = await lens.observe_async(delta, step_index)
            except asyncio.CancelledError:
                raise
            metrics.lens_lag_ms = (time.monotonic() - started) * 1000.0
            for probe in probes:
                if not isinstance(probe.get("step"), int):
                    probe["step"] = step_index
                store.attach_probe(session_id, step_index, probe)
                await self.send({"type": "probe", "step": probe["step"], "probe": probe})
                if isinstance(probe.get("kv"), dict):
                    await self.send({"type": "kv", "kv": probe["kv"]})
            if step_index % PROJECTION_EVERY_TOKENS == 0:
                projection = lens.projection()
                if projection:
                    await self.send(
                        {
                            "type": "projection",
                            "projection": projection,
                            "points": projection.get("points", []),
                            "reducer": projection.get("reducer", "deterministic"),
                            "fitted_at": projection.get("fitted_at", 0),
                        }
                    )

    # -- teardown -------------------------------------------------------
    async def cleanup(self) -> None:
        self.closed = True
        handle = self.handle
        if handle is not None:
            handle.stop.set()
            if handle.demo is not None:
                handle.demo.stop()
        task = self.gen_task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):  # noqa: B014 - defensive
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: B014 - defensive
                    pass
        if handle is not None:
            self.app.state.generations.pop(handle.session_id, None)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or get_settings()
    cfg.ensure_dirs()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = cfg
        app.state.started_at = time.time()
        app.state.store = SessionStore(cfg.sessions_dir)
        app.state.rag = RAGStore(
            cfg.memory_dir,
            embed_base_url=cfg.lmstudio_base_url if cfg.embed_enabled else None,
            embed_model=cfg.embed_model,
        )
        app.state.lens = LensBridge(cfg)
        native_context = max(512, int(os.environ.get("XRAY_NATIVE_CONTEXT", "32768")))
        app.state.native = (
            NativeBridge(cfg, max_context=native_context) if NativeBridge is not None else None
        )
        app.state.native_generation = None
        app.state.metrics = MetricsCollector()
        app.state.lmstudio = LMStudioClient(
            cfg.lmstudio_base_url,
            connect_timeout=cfg.lmstudio_connect_timeout,
        )
        app.state.models_cache = {"at": 0.0, "reachable": False, "data": {"reachable": False, "models": [], "error": None}}
        app.state.generations = {}
        app.state.active_tasks = set()
        await app.state.metrics.start()
        try:
            yield
        finally:
            for task in list(app.state.active_tasks):
                task.cancel()
            if app.state.active_tasks:
                await asyncio.gather(*app.state.active_tasks, return_exceptions=True)
            await app.state.metrics.stop()
            await app.state.lens.shutdown()
            native_bridge = getattr(app.state, "native", None)
            if native_bridge is not None:
                try:
                    await asyncio.to_thread(native_bridge.unload)
                except Exception:  # pragma: no cover - shutdown is best effort
                    pass
            await app.state.rag.aclose()
            await app.state.lmstudio.aclose()

    app = FastAPI(title="LLM-Xray", version=cfg.version, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # -- REST -----------------------------------------------------------
    @app.get("/api/health")
    async def health(request: Request) -> dict[str, Any]:
        state = request.app.state
        models = await _refresh_models(request.app)
        lens = await _lens_status(request.app)
        native = await _native_status(request.app)
        snapshot = state.metrics.snapshot()
        return {
            "status": "ok",
            "version": cfg.version,
            "uptime_s": round(time.time() - state.started_at, 1),
            "lmstudio": {
                "reachable": models.get("reachable", False),
                "error": models.get("error"),
                "models": models.get("models", []),
                "loaded": [item["id"] for item in models.get("models", []) if item.get("state") == "loaded"],
            },
            "lens": lens,
            "native": native,
            "gpu": {
                "available": bool(state.metrics.gpu_available),
                "util": snapshot.get("gpu"),
                "vram_used": snapshot.get("vram_used"),
                "vram_total": snapshot.get("vram_total"),
            },
            "metrics": snapshot,
            "sessions": state.store.session_count(),
            "demo_available": True,
        }

    @app.get("/api/config")
    async def config(request: Request) -> dict[str, Any]:
        return await _build_config(request.app)

    @app.get("/api/models")
    async def models(request: Request, refresh: bool = Query(False)) -> dict[str, Any]:
        result = await _refresh_models(request.app, force=bool(refresh))
        return {"reachable": result.get("reachable", False), "models": result.get("models", []), "error": result.get("error")}

    @app.get("/api/sessions")
    async def sessions(request: Request, limit: int = Query(20, ge=1, le=200)) -> list[dict[str, Any]]:
        return request.app.state.store.list_sessions(limit)

    @app.get("/api/sessions/{session_id}")
    async def session_detail(session_id: str, request: Request) -> dict[str, Any]:
        store: SessionStore = request.app.state.store
        state = store.get(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        detail = state.meta_public()
        detail["messages"] = state.messages
        detail["step_count"] = len(state.steps)
        detail["has_probe"] = state.has_lens
        return detail

    @app.get("/api/sessions/{session_id}/steps")
    async def session_steps(
        session_id: str,
        request: Request,
        offset: int = Query(0, ge=0),
        limit: int = Query(200, ge=1, le=MAX_STEP_QUERY_LIMIT),
    ) -> dict[str, Any]:
        store: SessionStore = request.app.state.store
        state = store.get(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        total = len(state.steps)
        return {
            "session_id": session_id,
            "steps": state.compact_steps(offset, limit),
            "total": total,
            "offset": offset,
            "limit": limit,
        }

    @app.get("/api/sessions/{session_id}/steps/{index}")
    async def session_step(session_id: str, index: int, request: Request) -> dict[str, Any]:
        store: SessionStore = request.app.state.store
        step = store.get_step(session_id, index)
        if step is None:
            raise HTTPException(status_code=404, detail="步骤不存在")
        return step

    @app.get("/api/sessions/{session_id}/export")
    async def session_export(session_id: str, request: Request) -> Response:
        store: SessionStore = request.app.state.store
        exported = store.export_jsonl(session_id)
        if exported is None:
            raise HTTPException(status_code=404, detail="会话不存在或没有可导出的步骤")
        filename, content = exported
        return Response(
            content,
            media_type="application/x-ndjson; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.post("/api/sessions/{session_id}/stop")
    async def session_stop(session_id: str, request: Request) -> dict[str, Any]:
        handle = request.app.state.generations.get(session_id)
        if handle is None:
            return {"stopped": False, "session_id": session_id, "reason": "no_active_generation"}
        handle.stop.set()
        if handle.demo is not None:
            handle.demo.stop()
        return {"stopped": True, "session_id": session_id}

    @app.post("/api/rag/search")
    async def rag_search(body: RagSearchRequest, request: Request) -> dict[str, Any]:
        query = (body.query or "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="query 不能为空")
        rag: RAGStore = request.app.state.rag
        items = await rag.search(query, body.top_k)
        return {"query": query, "items": items, "backend": rag.backend}

    @app.get("/api/rag/documents")
    async def rag_documents(request: Request) -> dict[str, Any]:
        rag: RAGStore = request.app.state.rag
        return await rag.documents()

    @app.post("/api/rag/reindex")
    async def rag_reindex(request: Request) -> dict[str, Any]:
        rag: RAGStore = request.app.state.rag
        stats = await rag.reindex()
        return {"ok": True, "stats": stats, "backend": rag.backend}

    @app.get("/api/lens/status")
    async def lens_status(request: Request) -> dict[str, Any]:
        return await _lens_status(request.app)

    @app.post("/api/lens/load")
    async def lens_load(body: LensLoadRequest, request: Request) -> dict[str, Any]:
        bridge: LensBridge = request.app.state.lens
        return bridge.start_load(body.model, body.device or "cpu")

    @app.post("/api/lens/unload")
    async def lens_unload(request: Request) -> dict[str, Any]:
        bridge: LensBridge = request.app.state.lens
        return await bridge.unload()

    # -- native exact runtime -------------------------------------------
    @app.get("/api/native/status")
    async def native_status(request: Request) -> dict[str, Any]:
        return await _native_status(request.app)

    @app.post("/api/native/load")
    async def native_load(body: NativeLoadRequest, request: Request) -> dict[str, Any]:
        bridge = getattr(request.app.state, "native", None)
        if bridge is None:
            return _disabled_native_status()
        active = getattr(request.app.state, "native_generation", None)
        if active is not None:
            status = await _native_status(request.app)
            status["ok"] = False
            status["message"] = "原生精确通道正在生成中，请先停止当前生成再加载模型"
            return status
        task = asyncio.create_task(
            bridge.load(
                body.model,
                body.device or "cpu",
                bool(body.load_in_4bit),
                body.device_map,
            ),
            name="xray-native-load",
        )
        request.app.state.active_tasks.add(task)
        task.add_done_callback(request.app.state.active_tasks.discard)
        await asyncio.sleep(0)  # let the coroutine transition to "loading"
        return await _native_status(request.app)

    @app.post("/api/native/unload")
    async def native_unload(request: Request) -> dict[str, Any]:
        bridge = getattr(request.app.state, "native", None)
        if bridge is None:
            return _disabled_native_status()
        active = getattr(request.app.state, "native_generation", None)
        if active is not None:
            status = await _native_status(request.app)
            status["ok"] = False
            status["message"] = "原生精确通道正在生成中，请先停止当前生成再卸载模型"
            return status
        await asyncio.to_thread(bridge.unload)
        return await _native_status(request.app)

    @app.get("/api/docs")
    async def docs_list() -> dict[str, Any]:
        documents: list[dict[str, Any]] = []
        try:
            for path in sorted(cfg.docs_dir.glob("*.md"), key=lambda item: item.name.lower()):
                documents.append({"name": path.name, "size": path.stat().st_size})
        except OSError:
            pass
        return {"documents": documents}

    @app.get("/api/docs/{name}")
    async def docs_read(name: str) -> PlainTextResponse:
        if not name or "/" in name or "\\" in name or ".." in name or not name.endswith(".md"):
            raise HTTPException(status_code=404, detail="文档不存在")
        try:
            base = cfg.docs_dir.resolve()
            path = (cfg.docs_dir / name).resolve()
        except OSError:
            raise HTTPException(status_code=404, detail="文档不存在")
        if base not in path.parents or not path.is_file():
            raise HTTPException(status_code=404, detail="文档不存在")
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            raise HTTPException(status_code=404, detail="文档不可读")
        return PlainTextResponse(content, media_type="text/markdown; charset=utf-8")

    # -- WebSocket ------------------------------------------------------
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        await Connection(websocket, app).run()

    # -- static documentation site (docs/index.html, docs/*.css) --------
    if cfg.docs_dir.is_dir():
        app.mount(
            "/documentation",
            StaticFiles(directory=str(cfg.docs_dir), html=True),
            name="documentation",
        )

    # -- static frontend (mounted last, so /api and /ws win) ------------
    @app.get("/{full_path:path}", include_in_schema=False)
    async def static_frontend(full_path: str):
        static_dir = cfg.static_dir
        if static_dir.is_dir():
            try:
                base = static_dir.resolve()
                candidate = (static_dir / full_path).resolve() if full_path else base
            except OSError:
                candidate = None
            if candidate is not None and base in candidate.parents and candidate.is_file():
                media_type, _ = mimetypes.guess_type(str(candidate))
                return FileResponse(candidate, media_type=media_type)
            index = static_dir / "index.html"
            if index.is_file():
                return FileResponse(index, media_type="text/html")
        if full_path in ("", "index.html"):
            return JSONResponse(
                {
                    "name": "LLM-Xray",
                    "version": cfg.version,
                    "message": "前端尚未构建：请在 apps/desktop 下执行构建，或访问 /api/health。",
                    "api": "/api/health",
                    "docs": "/api/docs",
                    "demo": "WebSocket /ws 发送 {type:'start', payload:{mode:'demo'}}",
                }
            )
        raise HTTPException(status_code=404, detail="Not found")

    return app


app = create_app()


__all__ = ["app", "create_app"]
