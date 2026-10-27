"""Session store: in-memory streaming, probes, JSONL persistence and guards."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from xray.store import SessionStore, now_ms


def make_step(index: int, delta: str = "字", *, probe: dict | None = None) -> dict:
    step = {
        "index": index,
        "delta": delta,
        "kind": "answer",
        "topk": [{"token": delta, "logprob": -0.2, "prob": 0.82, "rank": 1}],
        "chosen_logprob": -0.2,
        "chosen_prob": 0.82,
        "entropy": 0.7,
        "at": index * 40.0,
        "ttft_ms": 120.0,
        "tps": 12.4,
        "generated": index + 1,
        "text_len": (index + 1) * len(delta),
    }
    if probe is not None:
        step["probe"] = probe
    return step


def create(store: SessionStore) -> dict:
    return store.create_session(
        {
            "mode": "demo",
            "model": "demo-model",
            "started_at": now_ms(),
            "capabilities": {"logprobs": True, "attention": True, "lens": True},
            "sampling": {"temperature": 0.7, "top_p": 0.95, "top_k": 40, "repetition_penalty": 1.05, "max_tokens": 512, "seed": None},
            "rag_enabled": False,
            "lens_enabled": False,
        },
        [{"role": "user", "content": "为什么残差流像一条河？"}],
    )


def test_streaming_steps_and_probe_attachment(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    meta = create(store)
    session_id = meta["id"]
    assert store.append_step(session_id, make_step(0, "残"))
    assert store.append_step(session_id, make_step(1, "差"))
    probe = {"step": 1, "surrogate": True, "attention": {}, "lens": []}
    assert store.attach_probe(session_id, 1, probe)

    state = store.get(session_id)
    assert state is not None
    assert len(state.steps) == 2
    assert state.steps[0]["probe"] is None
    assert state.steps[1]["probe"] == probe
    assert state.preview == "残差"
    assert state.meta["capabilities"]["lens"] is True

    compact = state.compact_steps(0, 10)
    assert len(compact) == 2
    assert compact[1]["probe"] is None
    assert compact[1]["delta"] == "差"
    full = store.get_step(session_id, 1)
    assert full is not None and full["probe"] == probe
    assert store.get_step(session_id, 99) is None


def test_finish_writes_jsonl_and_meta(tmp_path):
    sessions_dir = tmp_path / "sessions"
    store = SessionStore(sessions_dir)
    meta = create(store)
    session_id = meta["id"]
    for index, delta in enumerate("残差流"):
        store.append_step(session_id, make_step(index, delta))
    store.attach_probe(session_id, 1, {"step": 1, "attention": {}, "lens": []})
    path = store.finish_session(session_id, {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6}, "stop", 1.5)
    assert path is not None
    jsonl = sessions_dir / f"{session_id}.jsonl"
    meta_file = sessions_dir / f"{session_id}.meta.json"
    assert jsonl.is_file() and meta_file.is_file()

    lines = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 3
    assert lines[1]["probe"]["step"] == 1
    record = json.loads(meta_file.read_text(encoding="utf-8"))
    assert record["meta"]["mode"] == "demo"
    assert record["finish_reason"] == "stop"
    assert record["has_lens"] is True

    exported = store.export_jsonl(session_id)
    assert exported is not None
    filename, content = exported
    assert filename == f"{session_id}.jsonl"
    assert len(content.strip().splitlines()) == 3


def test_list_sessions_and_reload_from_disk(tmp_path):
    sessions_dir = tmp_path / "sessions"
    store = SessionStore(sessions_dir)
    meta = create(store)
    store.append_step(meta["id"], make_step(0, "你好"))
    store.finish_session(meta["id"], None, "stop", 0.2)

    summaries = store.list_sessions()
    assert len(summaries) == 1
    assert summaries[0]["id"] == meta["id"]
    assert summaries[0]["tokens"] == 1
    assert summaries[0]["mode"] == "demo"

    fresh = SessionStore(sessions_dir)
    loaded = fresh.get(meta["id"])
    assert loaded is not None and len(loaded.steps) == 1
    assert fresh.read_session(meta["id"])["messages"][0]["role"] == "user"
    assert fresh.list_sessions()[0]["id"] == meta["id"]
    assert fresh.session_count() >= 1


def test_memory_limits_truncate_without_crashing(tmp_path):
    store = SessionStore(tmp_path / "sessions", max_steps=2, max_chars=10_000)
    meta = create(store)
    session_id = meta["id"]
    assert store.append_step(session_id, make_step(0, "一"))
    assert store.append_step(session_id, make_step(1, "二"))
    assert not store.append_step(session_id, make_step(2, "三"))
    assert store.get(session_id).truncated is True
    store.finish_session(session_id, None, "stop", 0.1)
    assert len(store.get(session_id).steps) == 2

    char_store = SessionStore(tmp_path / "sessions2", max_chars=4)
    char_meta = create(char_store)
    assert char_store.append_step(char_meta["id"], make_step(0, "四个字符"))
    assert not char_store.append_step(char_meta["id"], make_step(1, "六"))

    eviction = SessionStore(tmp_path / "sessions3", max_sessions=2)
    ids = [create(eviction)["id"] for _ in range(4)]
    assert eviction.session_count() >= 2
    assert eviction.exists(ids[-1])


def test_invalid_session_ids_are_rejected(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    assert store.get("../../etc/passwd") is None
    assert store.get("") is None
    assert store.export_jsonl("../../evil") is None
    assert store.get_step("nope", 0) is None


def test_thread_safe_appends(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    meta = create(store)
    session_id = meta["id"]

    def worker(offset: int) -> None:
        for index in range(offset, offset + 50):
            store.append_step(session_id, make_step(index, "x"))

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, (0, 50, 100, 150)))
    assert len(store.get(session_id).steps) == 200


def test_finish_unknown_session_returns_none(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    assert store.finish_session("does-not-exist") is None
    assert store.export_jsonl("does-not-exist") is None
    assert store.read_session("does-not-exist") is None
