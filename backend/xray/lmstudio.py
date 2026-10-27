"""LM Studio HTTP adapter.

LM Studio exposes three useful streaming surfaces (docs/SPEC.md §6):

1. ``POST /v1/responses`` — the only official route that returns *text and*
   ``top_logprobs`` together (with ``include=["message.output_text.logprobs"]``).
2. ``POST /v1/chat/completions`` — OpenAI compatible fallback with
   ``logprobs=true`` / ``top_logprobs=12``.
3. ``POST /api/v1/chat`` — native SSE, text only.

This module is deliberately forgiving about shapes: LM Studio versions differ
slightly in event names, logprob containers and usage keys.  Every parser
returns zero or more :class:`StreamEvent` objects and ignores unknown events
instead of raising, because a dashboard must not die on an unexpected field.

Nothing here loads a model.  ``GET /api/v0/models`` is metadata only and is the
only request allowed to run implicitly.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterable, Mapping

import httpx

from .sampling import SamplingParams

TOP_LOGPROBS = 12
RESPONSES_ENDPOINT = "/v1/responses"
CHAT_ENDPOINT = "/v1/chat/completions"
NATIVE_ENDPOINT = "/api/v1/chat"
MODELS_ENDPOINTS = ("/api/v0/models", "/v1/models")

_UNSUPPORTED_STATUS = (400, 404, 405)
_UNSUPPORTED_FRAGMENTS = (
    "unsupported",
    "not supported",
    "unknown parameter",
    "unrecognized",
    "invalid parameter",
    "unknown field",
)


class LMStudioError(Exception):
    """Base class for adapter failures."""


class LMStudioUnreachable(LMStudioError):
    """The server refused the connection or timed out before responding."""


class LMStudioHTTPError(LMStudioError):
    """The server answered with a non-degradable HTTP error."""

    def __init__(self, status_code: int, message: str, *, endpoint: str = "", body: str = "") -> None:
        super().__init__(f"{endpoint or 'LM Studio'} HTTP {status_code}: {message}")
        self.status_code = int(status_code)
        self.endpoint = endpoint
        self.body = body


class LMStudioStreamError(LMStudioError):
    """A stream event explicitly reported an error."""


# ---------------------------------------------------------------------------
# Normalised stream events
# ---------------------------------------------------------------------------


@dataclass
class StreamEvent:
    """One normalised event produced by any of the three providers."""

    kind: str = "delta"  # meta | fallback | delta | usage | finish | error
    text: str = ""
    token_kind: str = "answer"  # answer | reasoning
    topk: list[dict[str, Any]] = field(default_factory=list)
    chosen_logprob: float | None = None
    chosen_prob: float | None = None
    token_id: int | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    meta: dict[str, Any] | None = None
    code: str | None = None
    message: str | None = None

    @property
    def is_delta(self) -> bool:
        return self.kind == "delta"


# ---------------------------------------------------------------------------
# SSE decoding
# ---------------------------------------------------------------------------


class SSEDecoder:
    """Incremental ``text/event-stream`` decoder tolerant of split chunks."""

    def __init__(self) -> None:
        self._buffer = ""

    @property
    def buffered(self) -> str:
        return self._buffer

    def feed(self, chunk: str) -> list[tuple[str | None, str]]:
        if not chunk:
            return []
        self._buffer += chunk.replace("\r\n", "\n").replace("\r", "\n")
        events: list[tuple[str | None, str]] = []
        while "\n\n" in self._buffer:
            block, self._buffer = self._buffer.split("\n\n", 1)
            parsed = _parse_sse_block(block)
            if parsed is not None:
                events.append(parsed)
        return events

    def flush(self) -> list[tuple[str | None, str]]:
        """Emit a trailing block that arrived without the final blank line."""
        if not self._buffer.strip():
            self._buffer = ""
            return []
        block, self._buffer = self._buffer, ""
        parsed = _parse_sse_block(block)
        return [parsed] if parsed is not None else []


def _parse_sse_block(block: str) -> tuple[str | None, str] | None:
    event: str | None = None
    data_lines: list[str] = []
    for line in block.split("\n"):
        if not line or line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line[6:].strip() or None
        elif line.startswith("data:"):
            value = line[5:]
            if value.startswith(" "):
                value = value[1:]
            data_lines.append(value)
        else:
            # Some servers stream bare JSON lines; keep them.
            data_lines.append(line.strip())
    if not data_lines:
        return None
    return event, "\n".join(data_lines)


def decode_sse(text: str, *, provider: str = "responses") -> list[StreamEvent]:
    """Decode a complete SSE payload (fixture friendly, used by tests)."""
    decoder = SSEDecoder()
    events: list[StreamEvent] = []
    for event_name, data in decoder.feed(text) + decoder.flush():
        events.extend(parse_sse_data(event_name, data, provider=provider))
    return events


def parse_sse_data(event_name: str | None, data: str, *, provider: str = "responses") -> list[StreamEvent]:
    """Parse one SSE ``data:`` payload into normalised events."""
    data = data.strip()
    if not data or data == "[DONE]":
        return []
    try:
        payload = json.loads(data)
    except ValueError:
        # Not JSON: some servers emit a plain text chunk.
        return [StreamEvent(kind="delta", text=data)]
    if isinstance(payload, list):
        payload = payload[-1] if payload else None
    if not isinstance(payload, dict):
        return []
    etype = str(payload.get("type") or event_name or "")
    if _looks_like_chat_payload(payload):
        return parse_chat_chunk(payload)
    if etype.startswith("response.") or etype == "response":
        return parse_responses_event(etype, payload)
    if provider == "native" or _looks_like_native(etype, payload):
        return parse_native_event(etype, payload)
    if provider == "chat":
        return parse_chat_chunk(payload)
    return parse_responses_event(etype, payload)


def _looks_like_chat_payload(payload: Mapping[str, Any]) -> bool:
    return "choices" in payload or ("object" in payload and payload.get("object") == "chat.completion.chunk")


def _looks_like_native(etype: str, payload: Mapping[str, Any]) -> bool:
    return etype in {
        "message.start",
        "message.delta",
        "message.end",
        "chat.start",
        "chat.end",
        "prompt_processing.start",
        "prompt_processing.progress",
        "prompt_processing.end",
        "reasoning.delta",
        "reasoning.end",
    }


# ---------------------------------------------------------------------------
# Logprob normalisation
# ---------------------------------------------------------------------------


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def _decode_bytes(value: Any) -> str | None:
    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover - defensive
            return None
    if isinstance(value, list) and value and all(isinstance(item, int) for item in value[:8]):
        try:
            return bytes(item & 0xFF for item in value).decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover - defensive
            return None
    return None


def _token_text(raw: Mapping[str, Any]) -> str | None:
    for key in ("token", "text", "piece", "content", "word"):
        value = raw.get(key)
        if isinstance(value, str):
            return value
    return _decode_bytes(raw.get("bytes"))


def normalize_logprob_entry(raw: Any) -> dict[str, Any] | None:
    """Normalise one log-probability record from any endpoint."""
    if not isinstance(raw, Mapping):
        return None
    token = _token_text(raw)
    logprob = None
    for key in ("logprob", "log_prob", "logprobs", "chosen_logprob", "score"):
        candidate = raw.get(key)
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            logprob = float(candidate)
            break
        if isinstance(candidate, Mapping):
            logprob = _as_float(candidate.get("logprob"))
            if logprob is not None:
                break
    if logprob is None:
        logprob = _as_float(raw.get("logprob"))
    token_id = raw.get("id", raw.get("token_id", raw.get("tok_id")))
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        token_id = None
    top_raw = raw.get("top_logprobs", raw.get("topLogprobs", raw.get("alternatives")))
    top: list[dict[str, Any]] = []
    if isinstance(top_raw, Mapping):
        iterator: Iterable[tuple[str, Any]] = top_raw.items()
        for tok, lp in iterator:
            number = _as_float(lp if not isinstance(lp, Mapping) else lp.get("logprob"))
            if number is not None:
                top.append({"token": str(tok), "id": None, "logprob": number})
    elif isinstance(top_raw, Iterable) and not isinstance(top_raw, (str, bytes)):
        for item in top_raw:
            entry = normalize_logprob_entry(item)
            if entry is not None:
                top.append({"token": entry["token"], "id": entry["id"], "logprob": entry["logprob"]})
    if token is None and not top:
        return None
    if token is None:
        token = top[0]["token"]
        logprob = top[0]["logprob"] if logprob is None else logprob
    assert logprob is not None
    return {"token": token, "id": token_id, "logprob": logprob, "top_logprobs": top}


def build_topk(
    entry: Any,
    *,
    limit: int = TOP_LOGPROBS,
) -> tuple[list[dict[str, Any]], float | None, float | None, int | None]:
    """Return ``(topk, chosen_logprob, chosen_prob, chosen_id)`` from a logprob record."""
    normalised = normalize_logprob_entry(entry)
    if normalised is None:
        return [], None, None, None
    candidates: dict[str, dict[str, Any]] = {}
    for item in normalised["top_logprobs"]:
        candidates.setdefault(item["token"], item)
    chosen_token = normalised["token"]
    if chosen_token not in candidates:
        candidates[chosen_token] = {
            "token": chosen_token,
            "id": normalised["id"],
            "logprob": normalised["logprob"],
        }
    ordered = sorted(candidates.values(), key=lambda item: item["logprob"], reverse=True)
    if limit and limit > 0:
        ordered = ordered[:limit]
    topk: list[dict[str, Any]] = []
    for rank, item in enumerate(ordered, start=1):
        prob = _prob_from_logprob(item["logprob"])
        topk.append(
            {
                "token": item["token"],
                "id": item["id"],
                "logprob": item["logprob"],
                "prob": prob,
                "rank": rank,
            }
        )
    chosen_logprob = normalised["logprob"]
    chosen_prob = _prob_from_logprob(chosen_logprob)
    chosen_rank = next((item["rank"] for item in topk if item["token"] == chosen_token), None)
    return topk, chosen_logprob, chosen_prob, (normalised["id"] if chosen_token == normalised["token"] else None)


def _prob_from_logprob(logprob: float | None) -> float | None:
    if logprob is None:
        return None
    try:
        prob = math.exp(logprob)
    except OverflowError:  # pragma: no cover - defensive
        return 1.0
    return min(1.0, max(0.0, prob))


def _first_logprob(payload: Mapping[str, Any]) -> Any:
    """Find the logprob record inside a response payload."""
    for key in ("logprobs", "logprob", "logprobs_content"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            for inner in ("content", "items", "data"):
                if isinstance(value.get(inner), list) and value[inner]:
                    return value[inner][0]
            return value
        if isinstance(value, list) and value:
            return value[0]
    delta = payload.get("delta")
    if isinstance(delta, Mapping):
        return _first_logprob(delta)
    return None


def _all_logprobs(payload: Mapping[str, Any]) -> list[Any]:
    for key in ("logprobs", "logprobs_content"):
        value = payload.get(key)
        if isinstance(value, Mapping) and isinstance(value.get("content"), list):
            return list(value["content"])
        if isinstance(value, list):
            return list(value)
    return []


# ---------------------------------------------------------------------------
# Response / chat / native parsers
# ---------------------------------------------------------------------------


def parse_responses_event(etype: str, payload: Mapping[str, Any]) -> list[StreamEvent]:
    """Parse a ``/v1/responses`` server-sent event."""
    events: list[StreamEvent] = []
    data = payload.get("delta")
    text = ""
    if isinstance(data, str):
        text = data
    elif isinstance(data, Mapping):
        for key in ("text", "content", "delta"):
            value = data.get(key)
            if isinstance(value, str):
                text = value
                break

    lower = etype.lower()
    if "reasoning" in lower and ("delta" in lower or "text" in lower):
        if text:
            events.append(StreamEvent(kind="delta", text=text, token_kind="reasoning"))
        return events

    if "output_text.delta" in lower or lower.endswith("text.delta") or etype in ("response.text.delta",):
        logprob = _first_logprob(payload)
        topk, chosen_logprob, chosen_prob, token_id = build_topk(logprob)
        if not text and logprob is not None:
            normalised = normalize_logprob_entry(logprob)
            if normalised is not None:
                text = normalised["token"]
        if text or topk:
            events.append(
                StreamEvent(
                    kind="delta",
                    text=text,
                    token_kind="answer",
                    topk=topk,
                    chosen_logprob=chosen_logprob,
                    chosen_prob=chosen_prob,
                    token_id=token_id,
                )
            )
        return events

    if etype in ("response.completed", "response.done", "response.incomplete"):
        response = payload.get("response")
        if not isinstance(response, Mapping):
            response = payload
        usage = normalize_usage(response.get("usage"))
        finish_reason = _responses_finish_reason(response)
        if usage:
            events.append(StreamEvent(kind="usage", usage=usage))
        events.append(StreamEvent(kind="finish", finish_reason=finish_reason))
        return events

    if etype in ("response.failed", "response.error", "error"):
        events.append(_error_from_payload(payload, fallback_code="LMSTUDIO_STREAM_ERROR"))
        return events

    if etype in ("response.created", "response.in_progress", "response.output_item.added"):
        events.append(StreamEvent(kind="meta", meta={"type": etype}))
        return events

    # Unknown event: if it still carries text, keep it; otherwise ignore.
    if text:
        events.append(StreamEvent(kind="delta", text=text, token_kind="answer"))
    return events


def _responses_finish_reason(response: Mapping[str, Any]) -> str:
    status = response.get("status")
    if isinstance(status, str) and status in ("completed", "incomplete"):
        details = response.get("incomplete_details")
        if isinstance(details, Mapping) and details.get("reason"):
            return str(details["reason"])
        return "stop"
    for key in ("finish_reason", "stop_reason"):
        value = response.get(key)
        if isinstance(value, str) and value:
            return value
    return "stop"


def _usage_from_mapping(raw: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    prompt = _as_int(raw.get("input_tokens", raw.get("prompt_tokens")))
    completion = _as_int(raw.get("output_tokens", raw.get("completion_tokens")))
    total = _as_int(raw.get("total_tokens"))
    if total is None and (prompt is not None or completion is not None):
        total = (prompt or 0) + (completion or 0)
    if prompt is None and completion is None and total is None:
        return None
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def normalize_usage(raw: Any) -> dict[str, Any] | None:
    """Normalise usage counters from any endpoint."""
    if isinstance(raw, Mapping):
        return _usage_from_mapping(raw)
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _error_from_payload(payload: Mapping[str, Any], *, fallback_code: str = "LMSTUDIO_ERROR") -> StreamEvent:
    error = payload.get("error")
    message: str
    code = fallback_code
    if isinstance(error, Mapping):
        message = str(error.get("message") or error.get("code") or "未知的 LM Studio 错误")
        if isinstance(error.get("code"), str):
            code = str(error["code"])
        elif isinstance(error.get("type"), str):
            code = str(error["type"])
    elif isinstance(error, str):
        message = error
    else:
        message = str(payload.get("message") or "未知的 LM Studio 错误")
    return StreamEvent(kind="error", code=code, message=message, meta={"raw": dict(payload)})


def parse_chat_chunk(payload: Mapping[str, Any]) -> list[StreamEvent]:
    """Parse one ``/v1/chat/completions`` stream chunk."""
    events: list[StreamEvent] = []
    if isinstance(payload.get("error"), (Mapping, str)):
        events.append(_error_from_payload(payload))
        return events
    choices = payload.get("choices")
    if not isinstance(choices, list):
        choices = []
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, Mapping):
            delta = {}
        reasoning = None
        for key in ("reasoning_content", "reasoning", "thinking"):
            value = delta.get(key)
            if isinstance(value, str) and value:
                reasoning = value
                break
        if reasoning:
            events.append(StreamEvent(kind="delta", text=reasoning, token_kind="reasoning"))
        content = delta.get("content")
        if isinstance(content, list):  # multimodal-ish content parts
            content = "".join(
                str(part.get("text") or "") for part in content if isinstance(part, Mapping)
            )
        if isinstance(content, str):
            logprob_record = None
            logprobs = choice.get("logprobs")
            if isinstance(logprobs, Mapping):
                content_logprobs = logprobs.get("content")
                if isinstance(content_logprobs, list) and content_logprobs:
                    logprob_record = content_logprobs[0]
            elif isinstance(logprobs, list) and logprobs:
                logprob_record = logprobs[0]
            if logprob_record is None:
                logprob_record = _first_logprob(delta)
            topk, chosen_logprob, chosen_prob, token_id = build_topk(logprob_record)
            if content or topk:
                events.append(
                    StreamEvent(
                        kind="delta",
                        text=content or "",
                        token_kind="answer",
                        topk=topk,
                        chosen_logprob=chosen_logprob,
                        chosen_prob=chosen_prob,
                        token_id=token_id,
                    )
                )
        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason:
            events.append(StreamEvent(kind="finish", finish_reason=finish_reason))
    usage = normalize_usage(payload.get("usage"))
    if usage:
        events.append(StreamEvent(kind="usage", usage=usage))
    return events


def parse_native_event(etype: str, payload: Mapping[str, Any]) -> list[StreamEvent]:
    """Parse LM Studio's native ``/api/v1/chat`` SSE."""
    events: list[StreamEvent] = []
    lower = etype.lower()
    if isinstance(payload.get("error"), (Mapping, str)):
        events.append(_error_from_payload(payload))
        return events
    text = ""
    for key in ("content", "text", "delta"):
        value = payload.get(key)
        if isinstance(value, str):
            text = value
            break
    if lower.startswith("reasoning") and text:
        events.append(StreamEvent(kind="delta", text=text, token_kind="reasoning"))
        return events
    if "delta" in lower and text:
        events.append(StreamEvent(kind="delta", text=text, token_kind="answer"))
        return events
    if lower in ("chat.end", "message.end", "response.end", "chat.completed") or lower.endswith(".end"):
        result = payload.get("result")
        if not isinstance(result, Mapping):
            result = payload
        usage = normalize_usage(result.get("usage") or result.get("stats"))
        finish_reason = "stop"
        for key in ("stop_reason", "finish_reason"):
            value = result.get(key)
            if isinstance(value, str) and value:
                finish_reason = value
                break
        if usage:
            events.append(StreamEvent(kind="usage", usage=usage))
        events.append(StreamEvent(kind="finish", finish_reason=finish_reason))
        return events
    if text:
        events.append(StreamEvent(kind="delta", text=text, token_kind="answer"))
    return events


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------


def normalize_models(payload: Any) -> list[dict[str, Any]]:
    """Normalise ``/api/v0/models`` or ``/v1/models`` payloads."""
    raw_items: list[Any] = []
    if isinstance(payload, Mapping):
        for key in ("data", "models"):
            value = payload.get(key)
            if isinstance(value, list):
                raw_items = value
                break
    elif isinstance(payload, list):
        raw_items = payload
    models: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_items:
        if not isinstance(item, Mapping):
            continue
        model_id = item.get("id") or item.get("model_key") or item.get("name")
        if not model_id:
            continue
        model_id = str(model_id)
        if model_id in seen:
            continue
        seen.add(model_id)
        model: dict[str, Any] = {"id": model_id}
        for key in ("object", "type", "publisher", "arch", "quantization", "state", "compatibility_type"):
            value = item.get(key)
            if isinstance(value, str):
                model[key] = value
        for key in ("max_context_length", "context_length", "context"):
            value = item.get(key)
            if isinstance(value, int):
                model["max_context_length"] = value
                break
        models.append(model)
    return models


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class LMStudioClient:
    """Thin async adapter around an LM Studio server."""

    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        connect_timeout: float = 3.0,
        read_timeout: float = 180.0,
    ) -> None:
        self.base_url = (base_url or "http://127.0.0.1:1234").rstrip("/")
        self._connect_timeout = float(connect_timeout)
        timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=30.0,
            pool=5.0,
        )
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
            headers={"User-Agent": "LLM-Xray/1.0"},
        )

    # -- lifecycle ------------------------------------------------------
    async def aclose(self) -> None:
        try:
            await self._client.aclose()
        except Exception:  # pragma: no cover - defensive
            pass

    async def __aenter__(self) -> "LMStudioClient":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    # -- metadata (never loads a model) ---------------------------------
    async def list_models(self, *, timeout: float = 2.5) -> list[dict[str, Any]]:
        last_error: LMStudioHTTPError | None = None
        for path in MODELS_ENDPOINTS:
            try:
                request_timeout = httpx.Timeout(timeout, connect=min(float(timeout), self._connect_timeout))
                response = await self._client.get(
                    path, timeout=request_timeout, headers={"Accept": "application/json"}
                )
            except httpx.TransportError as exc:
                raise LMStudioUnreachable(f"无法连接 LM Studio ({self.base_url}): {exc}") from exc
            if response.status_code == 404:
                continue
            if response.status_code >= 400:
                last_error = LMStudioHTTPError(
                    response.status_code,
                    f"读取模型列表失败: {response.text[:200]}",
                    endpoint=path,
                    body=response.text[:1000],
                )
                continue
            try:
                payload = response.json()
            except ValueError:
                continue
            return normalize_models(payload)
        if last_error is not None:
            raise last_error
        return []

    async def ping(self, *, timeout: float = 2.0) -> dict[str, Any]:
        """Return ``{"reachable": bool, "models": [...], "error": str|None}``."""
        try:
            models = await self.list_models(timeout=timeout)
        except LMStudioError as exc:
            return {"reachable": False, "models": [], "error": str(exc)}
        return {"reachable": True, "models": models, "error": None}

    # -- streaming ------------------------------------------------------
    def _provider_candidates(self, provider: str) -> list[str]:
        provider = (provider or "auto").lower()
        if provider in ("responses", "chat", "native"):
            return [provider]
        return ["responses", "chat", "native"]

    def _endpoint(self, provider: str) -> str:
        return {"responses": RESPONSES_ENDPOINT, "chat": CHAT_ENDPOINT, "native": NATIVE_ENDPOINT}[provider]

    def _build_body(
        self,
        provider: str,
        messages: list[dict[str, Any]],
        params: SamplingParams,
        model: str,
        system: str | None,
    ) -> dict[str, Any]:
        chat_messages = _with_system(messages, system)
        if provider == "responses":
            return {
                "model": model,
                "input": chat_messages,
                "stream": True,
                "temperature": params.temperature,
                "top_p": params.top_p,
                "max_output_tokens": params.max_tokens,
                "store": False,
                "include": ["message.output_text.logprobs"],
                "top_logprobs": TOP_LOGPROBS,
            }
        if provider == "chat":
            body: dict[str, Any] = {
                "model": model,
                "messages": chat_messages,
                "stream": True,
                "temperature": params.temperature,
                "top_p": params.top_p,
                "max_tokens": params.max_tokens,
                "logprobs": True,
                "top_logprobs": TOP_LOGPROBS,
                "top_k": params.top_k,
                "repeat_penalty": params.repetition_penalty,
            }
            if params.seed is not None:
                body["seed"] = params.seed
            return body
        prompt = "\n\n".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in chat_messages)
        body = {
            "model": model,
            "input": prompt,
            "stream": True,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
        }
        if params.seed is not None:
            body["seed"] = params.seed
        return body

    async def stream(
        self,
        messages: list[dict[str, Any]],
        sampling: SamplingParams | Mapping[str, Any] | None = None,
        *,
        model: str,
        provider: str = "auto",
        system: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a completion, degrading providers only on explicit 400/404/405.

        Raises :class:`LMStudioUnreachable` for connection problems.  The
        generator is cancellable: cancelling the consuming task closes the
        HTTP stream (``finally`` block).
        """
        params = sampling if isinstance(sampling, SamplingParams) else SamplingParams.from_dict(sampling)
        candidates = self._provider_candidates(provider)
        last_error: LMStudioError | None = None
        for position, candidate in enumerate(candidates):
            body = self._build_body(candidate, messages, params, model, system)
            endpoint = self._endpoint(candidate)
            try:
                cm = self._client.stream("POST", endpoint, json=body)
                response = await cm.__aenter__()
            except httpx.TransportError as exc:
                # Connection errors must not loop across providers (SPEC §6).
                raise LMStudioUnreachable(
                    f"无法连接 LM Studio ({self.base_url}): {exc}"
                ) from exc
            try:
                if response.status_code >= 400:
                    raw = await _read_stream_body(response)
                    if _is_unsupported(response.status_code, raw) and position + 1 < len(candidates):
                        last_error = LMStudioHTTPError(
                            response.status_code,
                            raw[:300] or "unsupported",
                            endpoint=endpoint,
                            body=raw[:2000],
                        )
                        yield StreamEvent(
                            kind="fallback",
                            meta={
                                "from": candidate,
                                "to": candidates[position + 1],
                                "status": response.status_code,
                                "reason": raw[:300],
                            },
                        )
                        continue
                    raise LMStudioHTTPError(
                        response.status_code,
                        raw[:500] or response.reason_phrase,
                        endpoint=endpoint,
                        body=raw[:2000],
                    )
                yield StreamEvent(
                    kind="meta",
                    meta={
                        "provider": candidate,
                        "endpoint": endpoint,
                        "supports_logprobs": candidate != "native",
                        "model": model,
                    },
                )
                decoder = SSEDecoder()
                async for chunk in response.aiter_text():
                    for event_name, data in decoder.feed(chunk):
                        for event in parse_sse_data(event_name, data, provider=candidate):
                            if event.kind == "error":
                                yield event
                                return
                            yield event
                for event_name, data in decoder.flush():
                    for event in parse_sse_data(event_name, data, provider=candidate):
                        if event.kind == "error":
                            yield event
                            return
                        yield event
                return
            finally:
                try:
                    await cm.__aexit__(None, None, None)
                except Exception:  # pragma: no cover - defensive
                    pass
        if last_error is not None:
            raise last_error
        raise LMStudioStreamError("LM Studio 的流式接口没有返回任何数据")


def _with_system(messages: list[dict[str, Any]], system: str | None) -> list[dict[str, Any]]:
    normalized = [
        {"role": str(item.get("role") or "user"), "content": str(item.get("content") or "")}
        for item in messages
        if isinstance(item, Mapping)
    ]
    if system and not any(item["role"] == "system" for item in normalized):
        normalized.insert(0, {"role": "system", "content": system})
    return normalized


async def _read_stream_body(response: httpx.Response) -> str:
    try:
        raw = await response.aread()
    except Exception:  # pragma: no cover - defensive
        return ""
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover - defensive
        return ""


def _is_unsupported(status_code: int, body: str) -> bool:
    if status_code in _UNSUPPORTED_STATUS:
        return True
    lowered = (body or "").lower()
    return any(fragment in lowered for fragment in _UNSUPPORTED_FRAGMENTS)
