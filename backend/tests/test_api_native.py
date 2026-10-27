"""Integration tests for the native (exact) runtime channel.

These tests never download a model and never call LM Studio:

* status / config / hello shape is checked without loading anything;
* ``POST /api/native/load`` with a bogus local path must fail gracefully;
* one short end-to-end WS generation uses the *already downloaded* local
  ``Qwen2.5-0.5B-Instruct`` directory and stops after 3 tokens.

The real-model test is skipped when ``models/lens/Qwen__Qwen2.5-0.5B-Instruct``
is absent (run ``scripts/fetch_lens_model.ps1`` to enable it).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import PROJECT_ROOT, make_test_settings
from xray.app import create_app
from xray.lmstudio import StreamEvent

QWEN_DIR = PROJECT_ROOT / "models" / "lens" / "Qwen__Qwen2.5-0.5B-Instruct"
QWEN_AVAILABLE = (QWEN_DIR / "model.safetensors").is_file()
#: Qwen2.5-0.5B: 24 layers x 14 heads (matches lens tests).
QWEN_HEADS = 14


def _client(tmp_path):
    return TestClient(create_app(make_test_settings(tmp_path)))


def test_native_status_is_exposed_everywhere(tmp_path):
    with _client(tmp_path) as client:
        status = client.get("/api/native/status").json()
        assert status["surrogate"] is False  # never claim a surrogate for exact mode
        assert "available" in status and isinstance(status["available"], bool)
        assert status["status"] in ("unloaded", "loading", "ready", "error", "disabled")
        assert status["generation"] is False

        config = client.get("/api/config").json()
        assert config["native"]["surrogate"] is False
        assert config["native"]["status"] == status["status"]
        assert "exact_available" in config

        health = client.get("/api/health").json()
        assert health["native"]["surrogate"] is False

        with client.websocket_connect("/ws") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello"
            assert hello["native"]["surrogate"] is False
            assert hello["config"]["native"]["status"] == status["status"]
            ws.send_json({"type": "ping"})
            assert ws.receive_json()["type"] == "pong"


class _EmptyLogprobLMStudio:
    """Fake LM Studio stream: valid text but every top-k list is empty.

    Reproduces the field report: ``top_logprobs`` advertised but all topk=[].
    """

    def __init__(self) -> None:
        self.calls = 0

    async def ping(self, *, timeout: float = 2.5) -> dict:
        return {
            "reachable": True,
            "models": [{"id": "demo-model", "state": "loaded"}],
            "error": None,
        }

    async def stream(self, messages, sampling=None, *, model, provider="auto", system=None):
        self.calls += 1
        yield StreamEvent(
            kind="meta",
            meta={"provider": "responses", "supports_logprobs": True, "model": model},
        )
        for index in range(6):
            yield StreamEvent(
                kind="delta",
                text=f"字{index}",
                token_kind="answer",
                topk=[],
                chosen_logprob=None,
                chosen_prob=None,
            )
        yield StreamEvent(kind="finish", finish_reason="stop")

    async def aclose(self) -> None:
        return None


def test_lmstudio_empty_topk_warns_and_marks_session_logprobs_false(tmp_path):
    """When LM Studio returns no top_logprobs, warn and persist logprobs=false."""
    app = create_app(make_test_settings(tmp_path))
    with TestClient(app) as client:
        fake = _EmptyLogprobLMStudio()
        app.state.lmstudio = fake
        app.state.models_cache = {"at": 0.0, "reachable": False, "data": {"reachable": False, "models": [], "error": None}}
        with client.websocket_connect("/ws") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_json(
                {
                    "type": "start",
                    "payload": {
                        "mode": "lmstudio",
                        "model": "demo-model",
                        "messages": [{"role": "user", "content": "测试空 logprobs"}],
                        "sampling": {"max_tokens": 8},
                    },
                }
            )
            events: list[dict] = []
            for _ in range(200):
                event = ws.receive_json()
                events.append(event)
                if event["type"] in ("done", "error"):
                    break
        assert fake.calls == 1
        assert events[-1]["type"] == "done", events[-1]
        session_id = events[0]["session"]["id"]

        warnings = [
            event
            for event in events
            if event["type"] == "status" and event.get("scope") == "runtime" and event.get("state") == "no_logprobs"
        ]
        assert warnings, "expected one runtime warning about missing top_logprobs"
        assert "top_logprobs" in warnings[0]["message"]
        assert "原生精确模式" in warnings[0]["message"]

        done = events[-1]
        assert done["capabilities"]["logprobs"] is False
        tokens = [event for event in events if event["type"] == "token"]
        assert tokens
        assert all(step["step"]["topk"] == [] for step in tokens)

        detail = client.get(f"/api/sessions/{session_id}").json()
        assert detail["capabilities"]["logprobs"] is False


def test_native_load_missing_local_model_is_graceful(tmp_path):
    """No download: a missing local model ends in status=error with a message."""
    with _client(tmp_path) as client:
        if not client.get("/api/native/status").json().get("available"):
            pytest.skip("torch/transformers not installed on this host")
        response = client.post(
            "/api/native/load",
            json={"model": "xray-test/native-model-that-does-not-exist"},
        )
        assert response.status_code == 200
        status = response.json()
        assert status["status"] in ("loading", "error")

        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            status = client.get("/api/native/status").json()
            if status["status"] not in ("loading",):
                break
            time.sleep(0.2)

        assert status["status"] == "error", status
        assert status["message"]
        assert status["generation"] is False

        unloaded = client.post("/api/native/unload").json()
        assert unloaded["status"] == "unloaded"


@pytest.mark.skipif(
    not QWEN_AVAILABLE,
    reason="models/lens/Qwen__Qwen2.5-0.5B-Instruct not downloaded",
)
def test_native_ws_generates_three_tokens_with_exact_probe(tmp_path):
    """mode=native: tokens and probes come from one forward (surrogate=false)."""
    with _client(tmp_path) as client:
        assert client.get("/api/native/status").json()["available"] is True

        with client.websocket_connect("/ws") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello"

            ws.send_json(
                {
                    "type": "start",
                    "payload": {
                        "mode": "native",
                        "model": str(QWEN_DIR),
                        "messages": [{"role": "user", "content": "用一句话说你好"}],
                        "sampling": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "top_k": 40,
                            "repetition_penalty": 1.05,
                            "max_tokens": 3,
                            "seed": 42,
                        },
                    },
                }
            )
            events: list[dict] = []
            for _ in range(400):
                event = ws.receive_json()
                events.append(event)
                if event["type"] in ("done", "error"):
                    break
            assert events[-1]["type"] == "done", events[-1]
            assert not any(event["type"] == "error" for event in events)

        started = events[0]
        assert started["type"] == "started"
        session = started["session"]
        session_id = session["id"]
        assert session["mode"] == "native"
        capabilities = session["capabilities"]
        assert capabilities["logprobs"] is True
        assert capabilities["attention"] is True
        assert capabilities["lens"] is True
        assert capabilities["exact"] is True
        assert capabilities["surrogate"] is False
        assert session["model"] == str(QWEN_DIR)

        native_statuses = [
            event for event in events if event["type"] == "status" and event.get("scope") == "native"
        ]
        assert any(event.get("state") == "loading" for event in native_statuses)
        assert any(event.get("state") == "generating" for event in native_statuses)

        tokens = [event for event in events if event["type"] == "token"]
        assert 1 <= len(tokens) <= 3
        for token in tokens:
            step = token["step"]
            assert step["topk"], "native mode must expose real top-k logprobs"
            assert all("logprob" in option and "prob" in option for option in step["topk"])
            assert step["topk"][0]["rank"] == 1
            assert step["entropy"] is not None and step["entropy"] >= 0.0
            assert step["chosen_logprob"] is not None and step["chosen_logprob"] <= 0.0
            assert 0.0 < step["chosen_prob"] <= 1.0
            assert step["ttft_ms"] is not None and step["ttft_ms"] >= 0.0
            assert step["tps"] is not None and step["tps"] > 0.0
        assert [token["step"]["index"] for token in tokens] == list(range(len(tokens)))

        probes = [event for event in events if event["type"] == "probe"]
        assert probes, "native mode must attach probes from the same forward pass"
        assert len(probes) == len(tokens)
        probe = probes[0]["probe"]
        assert probe["surrogate"] is False
        assert probe["same_weights"] is True
        assert probe["model"] == str(QWEN_DIR)
        assert probe["context"]["tokens"] and len(probe["context"]["tokens"]) == len(probe["context"]["kinds"])
        assert set(probe["context"]["kinds"]) <= {"prompt", "generated"}
        assert probe["attention"]["head_count"] == QWEN_HEADS
        assert probe["attention"]["selected"]
        assert all(abs(sum(entry["mean"]) - 1.0) < 1e-3 for entry in probe["attention"]["selected"].values())
        assert len(probe["hidden"]) == 16
        assert len(probe["lens"]) >= 2
        assert any("heads" in entry for entry in probe["attention"]["selected"].values())
        assert probe["residual"]
        assert probe["kv"] and probe["kv"]["used"] > 0
        assert set(probe["vector"]) == {"x", "y", "z"}

        kv_events = [event for event in events if event["type"] == "kv"]
        assert kv_events, "kv event should follow the exact probe"
        projections = [event for event in events if event["type"] == "projection"]
        assert projections, "native mode should emit projection points"
        assert projections[0]["points"]
        assert projections[0]["reducer"] in ("UMAP", "PCA", "deterministic")

        done = events[-1]
        assert done["finish_reason"] in ("stop", "stopped")
        assert done["usage"]["prompt_tokens"] > 0
        assert done["usage"]["completion_tokens"] == len(tokens)
        assert done["usage"]["total_tokens"] == done["usage"]["prompt_tokens"] + len(tokens)
        assert done["capabilities"]["exact"] is True
        assert done["capabilities"]["surrogate"] is False

        # ---- persisted session keeps the exact capabilities / probe ----
        detail = client.get(f"/api/sessions/{session_id}").json()
        assert detail["mode"] == "native"
        assert detail["capabilities"]["exact"] is True
        assert detail["tokens"] == len(tokens)

        steps = client.get(f"/api/sessions/{session_id}/steps").json()
        assert steps["total"] == len(tokens)
        first_step = client.get(f"/api/sessions/{session_id}/steps/0").json()
        assert first_step["probe"] is not None
        assert first_step["probe"]["surrogate"] is False
        assert first_step["probe"]["same_weights"] is True

        # ---- REST unload releases the in-process model -----------------
        assert client.get("/api/native/status").json()["status"] == "ready"
        unloaded = client.post("/api/native/unload").json()
        assert unloaded["status"] == "unloaded"
        assert unloaded["generation"] is False
