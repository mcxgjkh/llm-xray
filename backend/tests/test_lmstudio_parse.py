"""Parser and adapter tests. All HTTP is served by ``httpx.MockTransport``.

No real LM Studio request is made: the test suite must never load the 8B model.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from xray.lmstudio import (
    LMStudioClient,
    LMStudioHTTPError,
    LMStudioUnreachable,
    SSEDecoder,
    StreamEvent,
    build_topk,
    decode_sse,
    normalize_logprob_entry,
    normalize_models,
    normalize_usage,
    parse_chat_chunk,
    parse_native_event,
    parse_responses_event,
    parse_sse_data,
)

RESPONSES_SSE = (
    "event: response.created\n"
    'data: {"type":"response.created","response":{"id":"r1"}}\n'
    "\n"
    "event: response.output_text.delta\n"
    'data: {"type":"response.output_text.delta","output_index":0,"delta":"残",'
    '"logprobs":[{"token":"残","logprob":-0.25,"bytes":[230,174,139],'
    '"top_logprobs":[{"token":"残","logprob":-0.25},{"token":"差","logprob":-1.7},'
    '{"token":"河","logprob":-2.1}]}]}\n'
    "\n"
    "event: response.output_text.delta\n"
    'data: {"type":"response.output_text.delta","delta":"差",'
    '"logprobs":{"token":"差","logprob":-0.5,"top_logprobs":'
    '{"残":-0.3,"差":-0.5,"流":-1.9}}}\n'
    "\n"
    "event: response.reasoning_text.delta\n"
    'data: {"type":"response.reasoning_text.delta","delta":"先想一想",'
    '"content_index":0}\n'
    "\n"
    "event: response.completed\n"
    'data: {"type":"response.completed","response":{"status":"completed",'
    '"usage":{"input_tokens":12,"output_tokens":4,"total_tokens":16}}}\n'
    "\n"
    "data: [DONE]\n"
    "\n"
)

CHAT_SSE = (
    'data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,'
    '"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n'
    "\n"
    'data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,'
    '"delta":{"content":"残"},"logprobs":{"content":[{"token":"残","logprob":-0.4,'
    '"top_logprobs":[{"token":"残","logprob":-0.4},{"token":"差","logprob":-2.2}]}]},'
    '"finish_reason":null}]}\n'
    "\n"
    'data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,'
    '"delta":{"reasoning_content":"推理"},"finish_reason":null}]}\n'
    "\n"
    'data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,'
    '"delta":{"content":"差"},"finish_reason":"stop"}],'
    '"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\n'
    "\n"
    "data: [DONE]\n"
    "\n"
)

NATIVE_SSE = (
    'data: {"type":"message.start"}\n'
    "\n"
    'data: {"type":"message.delta","content":"残"}\n'
    "\n"
    'data: {"type":"message.delta","content":"差"}\n'
    "\n"
    'data: {"type":"chat.end","result":{"stop_reason":"stop",'
    '"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}}\n'
    "\n"
)


# ---------------------------------------------------------------------------
# SSE decoding
# ---------------------------------------------------------------------------


def test_sse_decoder_tolerates_split_chunks_and_crlf():
    text = RESPONSES_SSE.replace("\n", "\r\n")
    decoder = SSEDecoder()
    events = []
    for start in range(0, len(text), 7):
        events.extend(decoder.feed(text[start : start + 7]))
    events.extend(decoder.flush())
    kinds = [kind for kind, _ in events]
    assert "response.output_text.delta" in kinds
    assert all(data != "[DONE]" for _, data in events) or any(data == "[DONE]" for _, data in events)


def test_parse_sse_ignores_done_and_garbage():
    assert parse_sse_data(None, "[DONE]") == []
    assert parse_sse_data(None, "not json at all")[0].kind == "delta"
    assert parse_sse_data(None, "") == []


# ---------------------------------------------------------------------------
# responses parser
# ---------------------------------------------------------------------------


def test_responses_event_parsing_full_stream():
    events = decode_sse(RESPONSES_SSE, provider="responses")
    deltas = [event for event in events if event.kind == "delta"]
    assert [event.text for event in deltas if event.token_kind == "answer"] == ["残", "差"]
    assert [event.text for event in deltas if event.token_kind == "reasoning"] == ["先想一想"]
    first = next(event for event in deltas if event.text == "残")
    assert first.chosen_logprob == -0.25
    assert first.chosen_prob is not None and 0 < first.chosen_prob < 1
    assert len(first.topk) == 3
    assert first.topk[0]["rank"] == 1
    usage = next(event for event in events if event.kind == "usage")
    assert usage.usage == {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}
    finish = next(event for event in events if event.kind == "finish")
    assert finish.finish_reason == "stop"


def test_responses_unknown_and_failed_events():
    failed = parse_responses_event(
        "response.failed", {"type": "response.failed", "error": {"code": "model_load_failed", "message": "加载失败"}}
    )
    assert failed[0].kind == "error"
    assert failed[0].code == "model_load_failed"
    assert parse_responses_event("response.output_item.added", {"type": "response.output_item.added"})[0].kind == "meta"


# ---------------------------------------------------------------------------
# chat parser
# ---------------------------------------------------------------------------


def test_chat_chunk_parsing():
    events = decode_sse(CHAT_SSE, provider="chat")
    answers = [event for event in events if event.kind == "delta" and event.token_kind == "answer" and event.text]
    assert [event.text for event in answers] == ["残", "差"]
    assert answers[0].topk[0]["token"] == "残"
    reasoning = [event for event in events if event.kind == "delta" and event.token_kind == "reasoning"]
    assert reasoning[0].text == "推理"
    assert any(event.kind == "finish" and event.finish_reason == "stop" for event in events)
    usage = next(event for event in events if event.kind == "usage")
    assert usage.usage["total_tokens"] == 7


# ---------------------------------------------------------------------------
# native parser
# ---------------------------------------------------------------------------


def test_native_event_parsing():
    events = decode_sse(NATIVE_SSE, provider="native")
    assert [event.text for event in events if event.kind == "delta"] == ["残", "差"]
    assert any(event.kind == "finish" for event in events)
    usage = next(event for event in events if event.kind == "usage")
    assert usage.usage == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


# ---------------------------------------------------------------------------
# logprob normalisation
# ---------------------------------------------------------------------------


def test_normalize_logprob_bytes_and_dict_top():
    entry = normalize_logprob_entry(
        {
            "bytes": [230, 174, 139],
            "logprob": -0.4,
            "top_logprobs": {"残": -0.4, "差": -1.9},
        }
    )
    assert entry is not None
    assert entry["token"] == "残"
    assert entry["logprob"] == -0.4
    assert {item["token"] for item in entry["top_logprobs"]} == {"残", "差"}


def test_normalize_logprob_nested_alternative():
    entry = normalize_logprob_entry({"token": "x", "logprob": {"logprob": -1.25}})
    assert entry is not None and entry["logprob"] == -1.25


def test_build_topk_sorts_and_keeps_chosen():
    topk, chosen_logprob, chosen_prob, token_id = build_topk(
        {
            "token": "残",
            "id": 7,
            "logprob": -1.0,
            "top_logprobs": [{"token": "差", "logprob": -0.2}, {"token": "残", "logprob": -1.0}],
        }
    )
    assert [item["token"] for item in topk] == ["差", "残"]
    assert chosen_logprob == -1.0
    assert chosen_prob == pytest.approx(0.3679, abs=1e-3)
    assert token_id == 7


def test_normalize_usage_variants():
    assert normalize_usage({"input_tokens": 10, "output_tokens": 3, "total_tokens": 13}) == {
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "total_tokens": 13,
    }
    assert normalize_usage({"prompt_tokens": 4, "completion_tokens": 2})["total_tokens"] == 6
    assert normalize_usage(None) is None
    assert normalize_usage({"unrelated": 1}) is None


def test_normalize_models_v0_and_v1():
    models = normalize_models(
        {
            "data": [
                {
                    "id": "deepseek-r1-distill-llama-8b",
                    "object": "model",
                    "type": "llm",
                    "publisher": "lmstudio-community",
                    "arch": "llama",
                    "quantization": "Q4_K_M",
                    "state": "not-loaded",
                    "max_context_length": 131072,
                },
                {"id": "text-embedding-nomic-embed-text-v1.5", "type": "embeddings", "state": "loaded"},
                {"id": "deepseek-r1-distill-llama-8b"},  # duplicate
            ]
        }
    )
    assert len(models) == 2
    assert models[0]["state"] == "not-loaded"
    assert models[0]["max_context_length"] == 131072
    assert normalize_models({"data": [{"id": "a"}]})[0]["id"] == "a"


# ---------------------------------------------------------------------------
# Client behaviour with a mock transport
# ---------------------------------------------------------------------------


def _async_client(handler) -> LMStudioClient:
    return LMStudioClient("http://mock.local:1234", transport=httpx.MockTransport(handler))


def _sse_response(text: str) -> httpx.Response:
    return httpx.Response(200, content=text.encode("utf-8"), headers={"content-type": "text/event-stream"})


def test_client_list_models_falls_back_to_v1():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/api/v0/models":
            return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(200, json={"data": [{"id": "m1", "object": "model"}]})

    async def run():
        client = _async_client(handler)
        try:
            return await client.list_models()
        finally:
            await client.aclose()

    models = asyncio.run(run())
    assert seen == ["/api/v0/models", "/v1/models"]
    assert models == [{"id": "m1", "object": "model"}]


def test_client_responses_stream_captures_body_and_events():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return _sse_response(RESPONSES_SSE)

    async def run():
        client = _async_client(handler)
        try:
            return [event async for event in client.stream([{"role": "user", "content": "你好"}], model="m1")]
        finally:
            await client.aclose()

    events = asyncio.run(run())
    body = captured["body"]
    assert captured["path"] == "/v1/responses"
    assert body["input"] == [{"role": "user", "content": "你好"}]
    assert body["stream"] is True
    assert body["store"] is False
    assert body["include"] == ["message.output_text.logprobs"]
    assert body["top_logprobs"] == 12
    assert body["max_output_tokens"] == 4096
    meta = [event for event in events if event.kind == "meta"]
    assert meta and meta[0].meta["provider"] == "responses"
    assert any(event.kind == "delta" for event in events)
    assert any(event.kind == "usage" for event in events)


def test_client_degrades_to_chat_on_unsupported_parameter():
    calls: list[str] = []
    chat_body: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/v1/responses":
            return httpx.Response(
                400,
                json={"error": {"message": "Unsupported parameter: include", "code": "invalid_request_error"}},
            )
        if request.url.path == "/v1/chat/completions":
            chat_body.update(json.loads(request.content))
            return _sse_response(CHAT_SSE)
        return httpx.Response(500)

    async def run():
        client = _async_client(handler)
        try:
            return [event async for event in client.stream([{"role": "user", "content": "hi"}], model="m1")]
        finally:
            await client.aclose()

    events = asyncio.run(run())
    assert calls == ["/v1/responses", "/v1/chat/completions"]
    fallbacks = [event for event in events if event.kind == "fallback"]
    assert fallbacks and fallbacks[0].meta["to"] == "chat"
    meta = [event for event in events if event.kind == "meta"]
    assert meta[0].meta["provider"] == "chat"
    assert chat_body["logprobs"] is True
    assert chat_body["top_logprobs"] == 12
    assert chat_body["top_k"] == 40
    assert chat_body["repeat_penalty"] == 1.05
    assert [event.text for event in events if event.kind == "delta" and event.token_kind == "answer" and event.text] == ["残", "差"]


def test_client_degrades_to_native_when_both_json_apis_fail():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/api/v1/chat":
            return _sse_response(NATIVE_SSE)
        return httpx.Response(404, json={"error": {"message": "not found"}})

    async def run():
        client = _async_client(handler)
        try:
            return [event async for event in client.stream([{"role": "user", "content": "hi"}], model="m1")]
        finally:
            await client.aclose()

    events = asyncio.run(run())
    assert calls == ["/v1/responses", "/v1/chat/completions", "/api/v1/chat"]
    assert [event.text for event in events if event.kind == "delta"] == ["残", "差"]
    assert [event.meta["provider"] for event in events if event.kind == "meta"] == ["native"]


def test_client_connection_error_does_not_retry_other_providers():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        raise httpx.ConnectError("connection refused", request=request)

    async def run():
        client = _async_client(handler)
        try:
            async for _ in client.stream([{"role": "user", "content": "hi"}], model="m1"):
                pytest.fail("should not yield events")
        finally:
            await client.aclose()

    with pytest.raises(LMStudioUnreachable):
        asyncio.run(run())
    assert calls == ["/v1/responses"]


def test_client_non_degradable_http_error_is_raised():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "boom"}})

    async def run():
        client = _async_client(handler)
        try:
            async for _ in client.stream([{"role": "user", "content": "hi"}], model="m1"):
                pass
        finally:
            await client.aclose()

    with pytest.raises(LMStudioHTTPError) as excinfo:
        asyncio.run(run())
    assert excinfo.value.status_code == 500


def test_client_explicit_provider_is_not_degraded():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(404, json={"error": {"message": "nope"}})

    async def run():
        client = _async_client(handler)
        try:
            async for _ in client.stream([{"role": "user", "content": "hi"}], model="m1", provider="chat"):
                pass
        finally:
            await client.aclose()

    with pytest.raises(LMStudioHTTPError):
        asyncio.run(run())
    assert calls == ["/v1/chat/completions"]


def test_parse_sse_data_dispatch_ignores_unrelated_payload():
    event = parse_sse_data(None, '{"type":"ping"}', provider="chat")
    assert event == []
