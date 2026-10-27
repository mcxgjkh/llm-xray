"""Hybrid retrieval over the local Markdown memory bank.

* Corpus: ``memory/*.md`` (top level), split at paragraphs into ~320 character
  chunks with 48 characters of overlap.
* Lexical score: BM25 over character 2–4-grams (works well for Chinese without a
  tokenizer dependency).
* Semantic score (optional): cosine similarity over LM Studio embeddings via
  ``/v1/embeddings`` with a 5 s timeout.  When the endpoint or the embedding
  model is unavailable the whole index degrades to pure BM25 — no exception, no
  user-visible error, no retry storm.
* Hybrid: ``0.55 * cosine_norm + 0.45 * bm25_norm`` (SPEC §9).

The index is built lazily and kept in memory; repeated queries never touch the
network unless embeddings are in use.  Logging is intentionally quiet: this code
sits on the hot path of a query.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx
import numpy as np

from .config import DEFAULT_EMBED_MODEL

CHUNK_CHARS = 320
CHUNK_OVERLAP = 48
EMBED_BATCH = 32
EMBED_TIMEOUT = 5.0
HYBRID_COSINE_WEIGHT = 0.55
HYBRID_BM25_WEIGHT = 0.45
TOP_K_DEFAULT = 3

_SAMPLE_MEMORIES: dict[str, str] = {
    "项目原则.md": (
        "# 项目原则\n\n"
        "LLM-Xray 的第一原则是诚实。LM Studio 只暴露文本、采样参数与 top-logprobs，"
        "不暴露隐藏状态、注意力矩阵或 MLP 输出，因此实时通道展示的一切概率都来自真实采样，"
        "而内部张量通道必须被明确标注为替代透镜（surrogate）。\n\n"
        "第二原则是低开销。界面可以炫目，但不能为了炫目拖慢本机推理；演示模式、"
        "mock 流与合成透镜数据让开发和测试不需要调用真实 8B 模型。\n\n"
        "第三原则是本地优先。记忆库、会话记录和导出文件都留在本机，默认不注入检索内容，"
        "避免额外 Token 费用和隐私外泄。\n"
    ),
    "本地推理笔记.md": (
        "# 本地推理笔记\n\n"
        "本机通过 LM Studio 提供 deepseek-r1-distill-llama-8b。推理接口优先使用 "
        "/v1/responses 以获得 top-logprobs，失败时按 400/404/405 降级到 "
        "/v1/chat/completions，最后才使用原生 /api/v1/chat。连接错误不重试，"
        "避免在模型加载期间反复请求造成卡顿。\n\n"
        "采样重算在前端完成：重复惩罚按最近若干 Token 的出现次数调整 logit，"
        "再依次应用温度、Top-K、Top-P 与 Softmax，不重新请求模型。\n\n"
        "如果只想演示界面，使用 demo 模式；它不加载任何模型，逐 Token 产生可信的 "
        "top-k 分布、熵与时间戳。\n"
    ),
}


# ---------------------------------------------------------------------------
# Text processing helpers (pure functions, unit tested)
# ---------------------------------------------------------------------------


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def char_ngrams(text: str, min_n: int = 2, max_n: int = 4) -> list[str]:
    """Character n-grams used by BM25 (Chinese friendly, no tokenizer)."""
    normalised = _normalise_text(text)
    grams: list[str] = []
    length = len(normalised)
    for size in range(min_n, max_n + 1):
        if length < size:
            continue
        for start in range(length - size + 1):
            gram = normalised[start : start + size]
            if gram.strip():
                grams.append(gram)
    return grams


def _split_blocks(text: str) -> list[dict[str, str]]:
    """Split Markdown into paragraphs, tracking the nearest heading."""
    blocks: list[dict[str, str]] = []
    heading = ""
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        body = "\n".join(line for line in buffer if line.strip()).strip()
        if body:
            blocks.append({"heading": heading, "text": body})
        buffer = []

    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            flush()
            heading = stripped.lstrip("#").strip()
            continue
        if not stripped:
            flush()
            continue
        buffer.append(line.rstrip())
    flush()
    return blocks


def _windows(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        return [text]
    if len(text) <= size:
        return [text]
    step = max(1, size - max(0, overlap))
    return [text[start : start + size] for start in range(0, len(text), step) if text[start : start + size]]


def chunk_document(
    doc_id: str,
    source: str,
    text: str,
    *,
    chunk_chars: int = CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Split one Markdown document into paragraph chunks (~320 chars, 48 overlap)."""
    size = max(64, int(chunk_chars))
    gap = max(0, min(int(overlap), size - 1))
    blocks = _split_blocks(text)
    chunks: list[dict[str, Any]] = []
    current: list[str] = []
    current_heading = blocks[0]["heading"] if blocks else ""
    counter = 0

    def emit() -> None:
        nonlocal current, counter
        body = "\n\n".join(part for part in current if part).strip()
        if not body:
            current = []
            return
        chunks.append(
            {
                "id": f"{doc_id}#{counter}",
                "source": source,
                "heading": current_heading,
                "text": body,
                "chars": len(body),
            }
        )
        counter += 1
        current = []

    for block in blocks:
        body = block["text"]
        heading = block["heading"]
        if heading != current_heading:
            emit()
            current_heading = heading
        if len(body) > size:
            emit()
            for window in _windows(body, size, gap):
                chunks.append(
                    {
                        "id": f"{doc_id}#{counter}",
                        "source": source,
                        "heading": heading,
                        "text": window,
                        "chars": len(window),
                    }
                )
                counter += 1
            current_heading = heading
            continue
        projected = sum(len(part) + 2 for part in current) + len(body)
        if current and projected > size:
            tail = chunks[-1]["text"][-gap:] if chunks and gap else ""
            emit()
            if tail:
                current.append(tail)
            current_heading = heading
        current.append(body)
    emit()
    return chunks


def _bm25_scores(
    query_counts: Counter,
    chunk_counts: list[Counter],
    df: Counter,
    total_docs: int,
    avgdl: float,
) -> list[float]:
    scores = [0.0] * len(chunk_counts)
    if total_docs <= 0 or avgdl <= 0:
        return scores
    k1, b = 1.5, 0.75
    idf: dict[str, float] = {}
    for term in query_counts:
        doc_freq = df.get(term, 0)
        if doc_freq <= 0:
            continue
        idf[term] = math.log(1.0 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
    for index, counts in enumerate(chunk_counts):
        doc_len = sum(counts.values()) or 1
        score = 0.0
        for term, query_freq in query_counts.items():
            term_freq = counts.get(term, 0)
            if not term_freq:
                continue
            numerator = term_freq * (k1 + 1.0)
            denominator = term_freq + k1 * (1.0 - b + b * doc_len / avgdl)
            score += idf.get(term, 0.0) * (numerator / denominator) * query_freq
        scores[index] = score
    return scores


def _min_max(values: Iterable[float]) -> list[float]:
    numbers = [float(value) for value in values]
    if not numbers:
        return []
    low, high = min(numbers), max(numbers)
    if high <= low:
        if high <= 0:
            return [0.0 for _ in numbers]
        return [min(1.0, value / high) for value in numbers]
    return [(value - low) / (high - low) for value in numbers]


def _extract_hits(query: str, text: str, limit: int = 6) -> list[str]:
    """Readable hit words: longest query n-grams that appear in the chunk."""
    normalised_query = _normalise_text(query)
    normalised_text = _normalise_text(text)
    if not normalised_query or not normalised_text:
        return []
    candidates: list[tuple[int, str]] = []
    length = len(normalised_query)
    for size in range(min(4, length), 1, -1):
        for start in range(length - size + 1):
            gram = normalised_query[start : start + size]
            if gram.strip() and gram in normalised_text:
                candidates.append((size, gram))
    candidates.sort(key=lambda item: (-item[0], normalised_query.find(item[1])))
    hits: list[str] = []
    for _, gram in candidates:
        if any(gram in existing for existing in hits):
            continue
        hits.append(gram)
        if len(hits) >= limit:
            break
    return hits


# ---------------------------------------------------------------------------
# Index + store
# ---------------------------------------------------------------------------


@dataclass
class _Index:
    chunks: list[dict[str, Any]]
    vectorizer: Any
    matrix: Any
    counts: list[Counter]
    df: Counter
    avgdl: float
    embeddings: np.ndarray | None
    backend: str
    built_at: float


class RAGStore:
    """Lazy hybrid index over ``memory/*.md``."""

    def __init__(
        self,
        memory_dir: Path | str,
        *,
        embed_base_url: str | None = "http://127.0.0.1:1234",
        embed_model: str = DEFAULT_EMBED_MODEL,
        embed_timeout: float = EMBED_TIMEOUT,
        chunk_chars: int = CHUNK_CHARS,
        overlap: int = CHUNK_OVERLAP,
    ) -> None:
        self.memory_dir = Path(memory_dir)
        self.embed_base_url = (embed_base_url or "").rstrip("/") or None
        self.embed_model = embed_model
        self.embed_timeout = float(embed_timeout)
        self.chunk_chars = int(chunk_chars)
        self.overlap = int(overlap)
        self._index: _Index | None = None
        self._lock = asyncio.Lock()
        self._embed_client: httpx.AsyncClient | None = None

    # -- lifecycle ------------------------------------------------------
    async def aclose(self) -> None:
        if self._embed_client is not None:
            try:
                await self._embed_client.aclose()
            except Exception:  # pragma: no cover - defensive
                pass
            self._embed_client = None

    # -- documents ------------------------------------------------------
    def _ensure_sample_memories(self) -> None:
        try:
            if self.memory_dir.is_dir():
                existing = list(self.memory_dir.glob("*.md"))
                if existing:
                    self._remove_empty_placeholder()
                    return
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            for name, content in _SAMPLE_MEMORIES.items():
                path = self.memory_dir / name
                if not path.exists():
                    path.write_text(content, encoding="utf-8")
        except OSError:
            pass

    def _remove_empty_placeholder(self) -> None:
        return

    def _scan_documents(self) -> list[dict[str, Any]]:
        self._ensure_sample_memories()
        documents: list[dict[str, Any]] = []
        try:
            paths = sorted(self.memory_dir.glob("*.md"), key=lambda path: path.name.lower())
        except OSError:
            return documents
        for path in paths:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if not text.strip():
                continue
            chunks = chunk_document(
                path.stem,
                path.name,
                text,
                chunk_chars=self.chunk_chars,
                overlap=self.overlap,
            )
            if chunks:
                documents.append({"path": path, "source": path.name, "text": text, "chunks": chunks})
        return documents

    # -- index ----------------------------------------------------------
    async def _embed_texts(self, texts: list[str]) -> np.ndarray | None:
        if not self.embed_base_url or not texts:
            return None
        if self._embed_client is None:
            self._embed_client = httpx.AsyncClient(timeout=httpx.Timeout(self.embed_timeout))
        vectors: list[list[float]] = []
        try:
            for start in range(0, len(texts), EMBED_BATCH):
                batch = texts[start : start + EMBED_BATCH]
                response = await self._embed_client.post(
                    f"{self.embed_base_url}/v1/embeddings",
                    json={"model": self.embed_model, "input": batch},
                )
                if response.status_code >= 400:
                    return None
                payload = response.json()
                items = payload.get("data") if isinstance(payload, dict) else None
                if not isinstance(items, list) or len(items) != len(batch):
                    return None
                ordered = sorted(items, key=lambda item: item.get("index", 0))
                for item in ordered:
                    embedding = item.get("embedding") if isinstance(item, dict) else None
                    if not isinstance(embedding, list) or not embedding:
                        return None
                    vectors.append([float(value) for value in embedding])
        except (httpx.HTTPError, ValueError, TypeError, KeyError):
            return None
        except Exception:  # pragma: no cover - defensive
            return None
        if not vectors:
            return None
        array = np.asarray(vectors, dtype=np.float32)
        norm = np.linalg.norm(array, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return array / norm

    async def _build_index(self) -> _Index:
        from sklearn.feature_extraction.text import TfidfVectorizer

        documents = self._scan_documents()
        chunks: list[dict[str, Any]] = []
        for document in documents:
            chunks.extend(document["chunks"])
        if not chunks:
            return _Index([], None, None, [], Counter(), 0.0, None, "empty", time.time())
        texts = [chunk["text"] for chunk in chunks]
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), sublinear_tf=True, lowercase=True)
        matrix = None
        try:
            matrix = vectorizer.fit_transform(texts)
        except ValueError:
            # Corpus too short for char 2-4-grams (e.g. a one-character memory):
            # keep the documents visible but rely on BM25 only.
            vectorizer = None
            matrix = None
        counts = [Counter(char_ngrams(text)) for text in texts]
        df: Counter = Counter()
        for counter in counts:
            df.update(counter.keys())
        avgdl = sum(sum(counter.values()) for counter in counts) / max(1, len(counts))
        embeddings = await self._embed_texts(texts)
        backend = "hybrid" if embeddings is not None else "bm25"
        return _Index(chunks, vectorizer, matrix, counts, df, avgdl, embeddings, backend, time.time())

    async def _ensure_index(self) -> _Index:
        if self._index is not None:
            return self._index
        async with self._lock:
            if self._index is not None:
                return self._index
            self._index = await self._build_index()
            return self._index

    # -- public async API ----------------------------------------------
    async def reindex(self) -> dict[str, Any]:
        async with self._lock:
            self._index = await self._build_index()
            index = self._index
        return self._stats(index)

    async def documents(self) -> dict[str, Any]:
        index = await self._ensure_index()
        grouped: dict[str, dict[str, Any]] = {}
        for chunk in index.chunks:
            entry = grouped.setdefault(
                chunk["source"],
                {"id": chunk["source"].rsplit(".", 1)[0], "source": chunk["source"], "chunks": 0, "chars": 0, "headings": []},
            )
            entry["chunks"] += 1
            entry["chars"] += chunk["chars"]
            if chunk["heading"] and chunk["heading"] not in entry["headings"]:
                entry["headings"].append(chunk["heading"])
        documents = sorted(grouped.values(), key=lambda item: item["source"].lower())
        return {
            "documents": documents,
            "stats": self._stats(index),
            "memory_dir": str(self.memory_dir),
        }

    async def search(self, query: str, top_k: int = TOP_K_DEFAULT) -> list[dict[str, Any]]:
        text = (query or "").strip()
        top_k = max(1, min(int(top_k or TOP_K_DEFAULT), 50))
        if not text:
            return []
        index = await self._ensure_index()
        if not index.chunks:
            return []

        query_counts = Counter(char_ngrams(text))
        bm25_scores = _bm25_scores(query_counts, index.counts, index.df, len(index.chunks), index.avgdl)
        bm25_norm = _min_max(bm25_scores)

        cosine_scores: np.ndarray | None = None
        backend = "bm25"
        if index.backend == "hybrid" and self.embed_base_url:
            query_vectors = await self._embed_texts([text])
            if query_vectors is not None and index.embeddings is not None:
                cosine_scores = index.embeddings @ query_vectors[0]
                backend = "hybrid"
        if cosine_scores is None:
            if index.vectorizer is not None and index.matrix is not None:
                query_vector = index.vectorizer.transform([text])
                cosine_scores = (index.matrix @ query_vector.T).toarray().ravel()
            else:
                cosine_scores = np.zeros(len(index.chunks), dtype=np.float32)
        cosine_norm = _min_max(cosine_scores.tolist())
        use_cosine = backend == "hybrid"

        ranked: list[tuple[float, int]] = []
        for position, chunk in enumerate(index.chunks):
            if use_cosine:
                hybrid = HYBRID_COSINE_WEIGHT * cosine_norm[position] + HYBRID_BM25_WEIGHT * bm25_norm[position]
            else:
                hybrid = bm25_norm[position]
            ranked.append((hybrid, position))
        ranked.sort(key=lambda item: (-item[0], item[1]))

        results: list[dict[str, Any]] = []
        for hybrid, position in ranked[:top_k]:
            chunk = index.chunks[position]
            results.append(
                {
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "heading": chunk["heading"] or None,
                    "cosine": round(float(cosine_scores[position]), 4) if use_cosine else None,
                    "bm25": round(float(bm25_scores[position]), 4),
                    "hybrid": round(float(hybrid), 4),
                    "hits": _extract_hits(text, chunk["text"]),
                }
            )
        return results

    # -- internals ------------------------------------------------------
    @property
    def backend(self) -> str:
        if self._index is not None and self._index.backend == "hybrid":
            return "hybrid"
        return "bm25"

    def _stats(self, index: _Index) -> dict[str, Any]:
        sources = {chunk["source"] for chunk in index.chunks}
        return {
            "documents": len(sources),
            "chunks": len(index.chunks),
            "chars": int(sum(chunk["chars"] for chunk in index.chunks)),
            "backend": "bm25" if index.backend in ("bm25", "empty") else "hybrid",
            "embedding_model": self.embed_model if index.backend == "hybrid" else None,
            "chunk_chars": self.chunk_chars,
            "overlap": self.overlap,
            "built_at": round(index.built_at * 1000.0, 1),
        }
