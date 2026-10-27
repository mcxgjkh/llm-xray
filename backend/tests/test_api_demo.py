"""End-to-end API tests using the offline demo stream (no LM Studio calls).

The demo stream is intentionally paced like a real generation (35–90 ms per
token), so the full WebSocket test takes a few seconds.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from conftest import make_test_settings
from xray.app import create_app
from xray.demo import DEMO_TOKENS, demo_text


def _client(tmp_path):
    app = create_app(make_test_settings(tmp_path))
    return TestClient(app)


def _collect_until_done(ws, limit: int = 400) -> list[dict]:
    events: list[dict] = []
    for _ in range(limit):
        event = ws.receive_json()
        events.append(event)
        if event.get("type") in ("done", "error"):
            break
    return events


def test_health_config_docs_and_root(tmp_path):
    with _client(tmp_path) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        payload = health.json()
        assert payload["status"] == "ok"
        assert payload["version"]
        assert payload["lmstudio"]["reachable"] is False  # dead port, no retry storm
        assert payload["sessions"] >= 0

        config = client.get("/api/config").json()
        assert config["default_mode"] == "lmstudio"
        assert config["demo_available"] is True
        assert set(config["capabilities"]) == {"logprobs", "attention", "lens", "rag", "gpu"}
        assert set(config["sampling_defaults"]) == {
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "max_tokens",
            "seed",
        }
        assert config["lens"]["status"] in ("disabled", "unloaded", "loading", "ready", "error")

        models = client.get("/api/models").json()
        assert models["reachable"] is False
        assert models["models"] == []

        spec = client.get("/api/docs/SPEC.md")
        assert spec.status_code == 200
        assert "LLM-Xray" in spec.text
        assert client.get("/api/docs/VERSION").status_code == 404
        assert client.get("/api/docs/nope.md").status_code == 404
        assert client.get("/api/docs/%2E%2E%2FVERSION").status_code == 404

        root = client.get("/")
        assert root.status_code == 200
        assert root.json()["name"] == "LLM-Xray"

        documentation = client.get("/documentation/")
        assert documentation.status_code == 200
        assert "LLM-Xray" in documentation.text

        rag = client.post("/api/rag/search", json={"query": "残差流", "top_k": 2})
        assert rag.status_code == 200
        assert isinstance(rag.json()["items"], list)
        assert client.post("/api/rag/search", json={"query": ""}).status_code == 400

        documents = client.get("/api/rag/documents").json()
        assert len(documents["documents"]) == 2  # generated sample memories
        reindexed = client.post("/api/rag/reindex").json()
        assert reindexed["ok"] is True

        lens = client.get("/api/lens/status").json()
        assert lens["surrogate"] is True
        unloaded = client.post("/api/lens/unload").json()
        assert unloaded["status"] in ("unloaded", "disabled")


def test_demo_websocket_full_flow_and_export(tmp_path):
    expected_tokens = len(DEMO_TOKENS)
    with _client(tmp_path) as client:
        with client.websocket_connect("/ws") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello"
            assert hello["version"]
            assert hello["config"]["demo_available"] is True
            assert "reachable" in hello["lmstudio"]

            ws.send_json({"type": "ping"})
            pong = ws.receive_json()
            assert pong["type"] == "pong" and isinstance(pong["at"], (int, float))

            ws.send_json(
                {
                    "type": "start",
                    "payload": {
                        "mode": "demo",
                        "messages": [{"role": "user", "content": "为什么残差流像一条河？"}],
                        "sampling": {"temperature": 0.7, "max_tokens": 128},
                    },
                }
            )
            events = _collect_until_done(ws)

        assert events[0]["type"] == "started", events[0]
        session = events[0]["session"]
        session_id = session["id"]
        assert session["mode"] == "demo"
        assert session["capabilities"]["logprobs"] is True
        assert events[-1]["type"] == "done", events[-1]
        assert not any(event["type"] == "error" for event in events)

        tokens = [event for event in events if event["type"] == "token"]
        assert len(tokens) >= 36
        assert len(tokens) == expected_tokens
        indices = [event["step"]["index"] for event in tokens]
        assert indices == list(range(len(tokens)))
        first = tokens[0]["step"]
        for key in (
            "index",
            "delta",
            "kind",
            "topk",
            "chosen_prob",
            "chosen_logprob",
            "entropy",
            "at",
            "ttft_ms",
            "tps",
            "generated",
            "text_len",
        ):
            assert key in first, key
        assert first["at"] >= 0
        assert first["topk"] and first["topk"][0]["rank"] == 1
        rebuilt = "".join(event["step"]["delta"] for event in tokens)
        assert rebuilt == demo_text()
        assert tokens[-1]["step"]["generated"] == len(tokens)

        probes = [event for event in events if event["type"] == "probe"]
        assert probes, "demo mode must emit synthetic probes"
        probe = probes[0]["probe"]
        assert probe["surrogate"] is True
        assert probe["attention"]["selected"]
        assert len(probe["lens"]) >= 1
        assert probe["residual"]
        assert probe["features"]
        assert probe["vector"] is not None

        projections = [event for event in events if event["type"] == "projection"]
        assert projections, "demo mode must emit projections"
        assert projections[0]["reducer"] in ("UMAP", "PCA", "deterministic")
        assert projections[0]["points"]

        metrics = [event for event in events if event["type"] == "metrics"]
        assert metrics, "demo mode must emit metrics"
        assert "cpu" in metrics[0]["metrics"]

        done = events[-1]
        assert done["finish_reason"] == "stop"
        assert done["usage"]["completion_tokens"] == len(tokens)
        assert done["elapsed_s"] > 0

        # ---- persistence / REST replay ----
        sessions = client.get("/api/sessions").json()
        assert any(item["id"] == session_id for item in sessions)

        detail = client.get(f"/api/sessions/{session_id}").json()
        assert detail["mode"] == "demo"
        assert detail["tokens"] == len(tokens)
        assert detail["usage"]["completion_tokens"] == len(tokens)
        # SessionMeta fields must survive the round trip (frontend replay needs them).
        assert detail["capabilities"]["logprobs"] is True
        assert detail["sampling"]["temperature"] == 0.7
        assert isinstance(detail["rag_enabled"], bool)
        assert isinstance(detail["lens_enabled"], bool)

        steps = client.get(f"/api/sessions/{session_id}/steps?offset=0&limit=10").json()
        assert steps["total"] == len(tokens)
        assert len(steps["steps"]) == 10
        assert steps["steps"][0]["probe"] is None

        probed_steps = [
            index
            for index in range(min(12, len(tokens)))
            if client.get(f"/api/sessions/{session_id}/steps/{index}").json().get("probe")
        ]
        assert probed_steps, "expected at least one step with a probe snapshot"

        export = client.get(f"/api/sessions/{session_id}/export")
        assert export.status_code == 200
        assert "jsonl" in export.headers["content-disposition"]
        lines = [json.loads(line) for line in export.text.strip().splitlines()]
        assert len(lines) == len(tokens)
        assert lines[0]["delta"] == tokens[0]["step"]["delta"]
        assert any(line.get("probe") for line in lines)

        assert client.get("/api/sessions/does-not-exist").status_code == 404
        assert client.get("/api/sessions/does-not-exist/export").status_code == 404


def test_demo_stop_and_invalid_payload(tmp_path):
    with _client(tmp_path) as client:
        with client.websocket_connect("/ws") as ws:
            assert ws.receive_json()["type"] == "hello"

            ws.send_json({"type": "start", "payload": {"messages": []}})
            error = ws.receive_json()
            assert error["type"] == "error"
            assert error["code"] == "INVALID_PAYLOAD"

            ws.send_json(
                {
                    "type": "start",
                    "payload": {"mode": "demo", "messages": [{"role": "user", "content": "停止测试"}]},
                }
            )
            events = [ws.receive_json()]
            while len([event for event in events if event["type"] == "token"]) < 3:
                event = ws.receive_json()
                events.append(event)
                assert event["type"] != "error", event
            ws.send_json({"type": "stop"})
            events.extend(_collect_until_done(ws))

        assert events[0]["type"] == "started"
        assert events[-1]["type"] == "done"
        assert events[-1]["finish_reason"] == "stopped"
        tokens = [event for event in events if event["type"] == "token"]
        assert 3 <= len(tokens) < len(demo_text())
