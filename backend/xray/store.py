"""Session state and JSONL persistence.

Design constraints from docs/SPEC.md §10: no synchronous disk writes while a
stream is running.  Steps are appended to memory in O(1); the whole session is
flushed to ``data/sessions/<id>.jsonl`` (one compact JSON step per line, probe
snapshot included) plus a small ``<id>.meta.json`` once generation finishes.
Both files are written atomically.

The store is protected by a re-entrant lock and is safe to call from the event
loop, from ``asyncio.to_thread`` workers and from test threads.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")

#: In-memory guard rails (SPEC §5: replay stays bounded even for absurd sessions).
DEFAULT_MAX_STEPS = 20000
DEFAULT_MAX_CHARS = 4_000_000
DEFAULT_MAX_SESSIONS = 32

STEP_KEYS = (
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
)


def now_ms() -> float:
    """Wall-clock epoch milliseconds (used for ``started_at`` / ``created_at``)."""
    return time.time() * 1000.0


def _new_session_id(created_at_ms: float) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(created_at_ms / 1000.0))
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def normalize_step(step: dict[str, Any]) -> dict[str, Any]:
    """Return a compact, JSON-safe step with every contract key present."""
    out: dict[str, Any] = {}
    for key in STEP_KEYS:
        out[key] = step.get(key)
    out["index"] = int(out.get("index") or 0)
    out["delta"] = str(out.get("delta") or "")
    kind = out.get("kind") or "answer"
    out["kind"] = kind if kind in ("answer", "reasoning") else "answer"
    out["topk"] = list(out.get("topk") or [])
    out["generated"] = int(out.get("generated") or out["index"] + 1)
    out["text_len"] = int(out.get("text_len") or 0)
    at = out.get("at")
    out["at"] = float(at) if isinstance(at, (int, float)) else float(out["index"])
    # Always present so JSONL replay sees a stable schema ("probe": null, no probe).
    out["probe"] = step.get("probe")
    return out


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=_json_default)


def _json_default(value: Any) -> Any:
    # numpy scalars / arrays show up when probes are produced by the lens engine.
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


@dataclass
class SessionState:
    """A live or recently finished session held in memory."""

    id: str
    meta: dict[str, Any]
    messages: list[dict[str, Any]] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    finish_reason: str = ""
    elapsed_s: float = 0.0
    created_at: float = field(default_factory=now_ms)
    preview: str = ""
    total_chars: int = 0
    truncated: bool = False
    path: str | None = None

    # ---- convenience ---------------------------------------------------
    @property
    def tokens(self) -> int:
        return len([s for s in self.steps if s.get("kind") == "answer"])

    @property
    def has_lens(self) -> bool:
        return any(step.get("probe") for step in self.steps)

    def summary(self) -> dict[str, Any]:
        mode = self.meta.get("mode") or "lmstudio"
        return {
            "id": self.id,
            "model": self.meta.get("model") or "",
            "mode": mode,
            "created_at": self.created_at,
            "tokens": len(self.steps),
            "preview": self.preview[:160],
            "has_lens": self.has_lens,
            "path": self.path,
        }

    def meta_public(self) -> dict[str, Any]:
        return {
            **self.meta,
            "id": self.id,
            "tokens": len(self.steps),
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "elapsed_s": self.elapsed_s,
            "has_lens": self.has_lens,
            "preview": self.preview[:200],
            "truncated": self.truncated,
        }

    def compact_steps(self, offset: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        window = self.steps[offset : offset + limit]
        return [_compact_step(step) for step in window]

    def full_step(self, index: int) -> dict[str, Any] | None:
        if index < 0 or index >= len(self.steps):
            return None
        return self.steps[index]


def _compact_step(step: dict[str, Any]) -> dict[str, Any]:
    """Step without the (large) probe payload, matching ``ReplayStep``."""
    return {
        "index": step.get("index"),
        "delta": step.get("delta"),
        "kind": step.get("kind"),
        "topk": step.get("topk") or [],
        "chosen_prob": step.get("chosen_prob"),
        "chosen_logprob": step.get("chosen_logprob"),
        "entropy": step.get("entropy"),
        "at": step.get("at"),
        "ttft_ms": step.get("ttft_ms"),
        "tps": step.get("tps"),
        "generated": step.get("generated"),
        "text_len": step.get("text_len"),
        "probe": None,
    }


class SessionStore:
    """Thread-safe in-memory session store with atomic JSONL persistence."""

    def __init__(
        self,
        sessions_dir: Path | str,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_chars: int = DEFAULT_MAX_CHARS,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
    ) -> None:
        self.sessions_dir = Path(sessions_dir)
        self.max_steps = int(max_steps)
        self.max_chars = int(max_chars)
        self.max_sessions = int(max_sessions)
        self._sessions: "OrderedDict[str, SessionState]" = OrderedDict()
        self._lock = threading.RLock()
        try:
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # creation / lookup
    # ------------------------------------------------------------------
    def create_session(self, meta: dict[str, Any], messages: Iterable[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Create a session and return its public meta (``SessionMeta`` shape)."""
        created = float(meta.get("started_at") or now_ms())
        session_id = str(meta.get("id") or _new_session_id(created))
        if not _SESSION_ID_RE.match(session_id):
            session_id = _new_session_id(created)
        normalized_meta = {
            "id": session_id,
            "mode": meta.get("mode") or "lmstudio",
            "model": str(meta.get("model") or ""),
            "started_at": created,
            "capabilities": dict(meta.get("capabilities") or {}),
            "sampling": dict(meta.get("sampling") or {}),
            "rag_enabled": bool(meta.get("rag_enabled")),
            "lens_enabled": bool(meta.get("lens_enabled")),
        }
        state = SessionState(
            id=session_id,
            meta=normalized_meta,
            messages=[dict(m) for m in (messages or [])],
            created_at=created,
        )
        with self._lock:
            self._sessions[session_id] = state
            self._evict_locked()
        return state.meta

    def get(self, session_id: str) -> SessionState | None:
        """In-memory session, loading it lazily from disk if necessary."""
        if not self._valid_id(session_id):
            return None
        with self._lock:
            state = self._sessions.get(session_id)
            if state is not None:
                self._sessions.move_to_end(session_id)
                return state
        record = self._load_record(session_id)
        if record is None:
            return None
        state = SessionState(
            id=session_id,
            meta=record["meta"],
            messages=record.get("messages") or [],
            steps=record.get("steps") or [],
            usage=record.get("usage"),
            finish_reason=record.get("finish_reason") or "",
            elapsed_s=float(record.get("elapsed_s") or 0.0),
            created_at=float(record["meta"].get("started_at") or now_ms()),
            preview=record.get("preview") or _preview_from_steps(record.get("steps") or []),
            total_chars=sum(len(str(s.get("delta") or "")) for s in (record.get("steps") or [])),
            path=str(self.sessions_dir / f"{session_id}.jsonl"),
        )
        with self._lock:
            self._sessions.setdefault(session_id, state)
            self._evict_locked()
        return state

    def exists(self, session_id: str) -> bool:
        return self.get(session_id) is not None

    # ------------------------------------------------------------------
    # streaming writes (memory only)
    # ------------------------------------------------------------------
    def append_step(self, session_id: str, step: dict[str, Any], probe: dict[str, Any] | None = None) -> bool:
        """Append one compact step. Returns False once the memory guard trips."""
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return False
            stats = normalize_step(step)
            if probe is not None:
                stats["probe"] = probe
            if len(state.steps) >= self.max_steps or state.total_chars + len(stats["delta"]) > self.max_chars:
                state.truncated = True
                return False
            state.steps.append(stats)
            if stats["kind"] == "answer":
                state.preview = (state.preview + stats["delta"])[-400:]
            state.total_chars += len(stats["delta"])
            self._sessions.move_to_end(session_id)
            return True

    def attach_probe(self, session_id: str, index: int, probe: dict[str, Any]) -> bool:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None or index < 0 or index >= len(state.steps):
                return False
            state.steps[index]["probe"] = probe
            return True

    def set_usage(
        self,
        session_id: str,
        usage: dict[str, Any] | None,
        finish_reason: str,
        elapsed_s: float,
    ) -> None:
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return
            state.usage = usage
            state.finish_reason = finish_reason
            state.elapsed_s = float(elapsed_s)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def finish_session(
        self,
        session_id: str,
        usage: dict[str, Any] | None = None,
        finish_reason: str = "stop",
        elapsed_s: float = 0.0,
    ) -> str | None:
        """Flush a finished session to disk; safe to call more than once."""
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return None
            state.usage = usage
            state.finish_reason = finish_reason
            state.elapsed_s = float(elapsed_s)
            steps = list(state.steps)
            meta = dict(state.meta)
            messages = list(state.messages)
            record_meta = {
                "id": state.id,
                "meta": meta,
                "messages": messages,
                "usage": usage,
                "finish_reason": finish_reason,
                "elapsed_s": float(elapsed_s),
                "tokens": len(steps),
                "preview": state.preview[:400],
                "has_lens": state.has_lens,
                "truncated": state.truncated,
                "saved_at": now_ms(),
            }
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = self.sessions_dir / f"{session_id}.jsonl"
            meta_path = self.sessions_dir / f"{session_id}.meta.json"
            try:
                body = "".join(_json_dumps(step) + "\n" for step in steps)
                _atomic_write(jsonl_path, body)
                _atomic_write(meta_path, _json_dumps(record_meta))
                state.path = str(jsonl_path)
                state.finish_reason = finish_reason
                state.usage = usage
                state.elapsed_s = float(elapsed_s)
            except OSError:
                state.path = None
                return None
            return str(jsonl_path)

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------
    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """Recent session summaries, in-memory and on-disk sessions merged."""
        limit = max(1, min(int(limit or 20), 200))
        summaries: dict[str, dict[str, Any]] = {}
        with self._lock:
            for state in self._sessions.values():
                if state.steps or state.finish_reason:
                    summaries[state.id] = state.summary()
        for meta_path in self._iter_meta_files():
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            session_id = str(data.get("id") or meta_path.name.split(".")[0])
            if session_id in summaries or not _SESSION_ID_RE.match(session_id):
                continue
            meta = data.get("meta") or {}
            summaries[session_id] = {
                "id": session_id,
                "model": meta.get("model") or "",
                "mode": meta.get("mode") or "lmstudio",
                "created_at": float(meta.get("started_at") or 0.0),
                "tokens": int(data.get("tokens") or 0),
                "preview": str(data.get("preview") or ""),
                "has_lens": bool(data.get("has_lens", False)) or _meta_has_lens(data),
                "path": str(self.sessions_dir / f"{session_id}.jsonl"),
            }
        ordered = sorted(summaries.values(), key=lambda item: item.get("created_at") or 0, reverse=True)
        return ordered[:limit]

    def session_count(self) -> int:
        with self._lock:
            count = len(self._sessions)
        return max(count, len(self._iter_meta_files()))

    def read_session(self, session_id: str) -> dict[str, Any] | None:
        """Full replay record (``SessionRecord`` shape) or ``None``."""
        if not self._valid_id(session_id):
            return None
        state = self.get(session_id)
        if state is None:
            return None
        return {
            "meta": state.meta,
            "steps": [_replay_step(step) for step in state.steps],
            "messages": state.messages,
            "usage": state.usage,
            "finish_reason": state.finish_reason,
            "elapsed_s": state.elapsed_s,
            "tokens": len(state.steps),
            "preview": state.preview[:200],
            "path": state.path,
        }

    def get_step(self, session_id: str, index: int) -> dict[str, Any] | None:
        state = self.get(session_id)
        if state is None:
            return None
        step = state.full_step(int(index))
        return _replay_step(step) if step is not None else None

    def export_jsonl(self, session_id: str) -> tuple[str, str] | None:
        """``(filename, content)`` of the JSONL export, one step per line."""
        if not self._valid_id(session_id):
            return None
        state = self.get(session_id)
        if state is None:
            return None
        path = self.sessions_dir / f"{session_id}.jsonl"
        if path.is_file():
            try:
                return f"{session_id}.jsonl", path.read_text(encoding="utf-8")
            except OSError:
                pass
        if not state.steps:
            return None
        return f"{session_id}.jsonl", "".join(_json_dumps(step) + "\n" for step in state.steps)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _valid_id(self, session_id: str) -> bool:
        return bool(session_id) and bool(_SESSION_ID_RE.match(session_id)) and ".." not in session_id

    def _iter_meta_files(self) -> list[Path]:
        try:
            return sorted(self.sessions_dir.glob("*.meta.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            return []

    def _load_record(self, session_id: str) -> dict[str, Any] | None:
        meta_path = self.sessions_dir / f"{session_id}.meta.json"
        jsonl_path = self.sessions_dir / f"{session_id}.jsonl"
        if not meta_path.is_file() and not jsonl_path.is_file():
            return None
        record: dict[str, Any] = {}
        if meta_path.is_file():
            try:
                record = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                record = {}
        steps: list[dict[str, Any]] = []
        if jsonl_path.is_file():
            try:
                with jsonl_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            steps.append(normalize_step(json.loads(line)))
                        except ValueError:
                            continue
            except OSError:
                steps = []
        if not record and not steps:
            return None
        return {
            "meta": record.get("meta") or {"id": session_id},
            "steps": steps,
            "messages": record.get("messages") or [],
            "usage": record.get("usage"),
            "finish_reason": record.get("finish_reason") or "",
            "elapsed_s": float(record.get("elapsed_s") or 0.0),
            "preview": record.get("preview") or "",
        }

    def _evict_locked(self) -> None:
        if self.max_sessions <= 0:
            return
        while len(self._sessions) > self.max_sessions:
            evicted_id, evicted = next(iter(self._sessions.items()))
            # Keep the most recent session even if it is the only one.
            if evicted_id == next(reversed(self._sessions)):
                break
            del self._sessions[evicted_id]
            if evicted.steps and not evicted.path:
                try:
                    self.finish_session(evicted_id)
                except Exception:  # pragma: no cover - defensive
                    pass


def _replay_step(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": step.get("index"),
        "delta": step.get("delta"),
        "kind": step.get("kind"),
        "topk": step.get("topk") or [],
        "chosen_prob": step.get("chosen_prob"),
        "chosen_logprob": step.get("chosen_logprob"),
        "entropy": step.get("entropy"),
        "at": step.get("at"),
        "ttft_ms": step.get("ttft_ms"),
        "tps": step.get("tps"),
        "generated": step.get("generated"),
        "text_len": step.get("text_len"),
        "probe": step.get("probe"),
    }


def _preview_from_steps(steps: Iterable[dict[str, Any]]) -> str:
    text = "".join(str(step.get("delta") or "") for step in steps if step.get("kind") == "answer")
    return text[-400:]


def _meta_has_lens(record: dict[str, Any]) -> bool:
    return bool(record.get("lens"))


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
