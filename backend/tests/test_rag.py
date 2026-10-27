"""RAG tests: chunking, sample memories, BM25/hybrid scoring, embedding fallback.

Embedding calls are disabled (``embed_base_url=None``) or point at a dead port,
so no LM Studio embedding request can happen during tests.
"""

from __future__ import annotations

import asyncio

from xray.rag import RAGStore, _extract_hits, char_ngrams, chunk_document


def run(coro):
    return asyncio.run(coro)


def test_char_ngrams_are_chinese_friendly():
    grams = char_ngrams("残差流", min_n=2, max_n=3)
    assert "残差" in grams
    assert "残差流" in grams
    assert all(gram.strip() for gram in grams)


def test_chunk_document_splits_long_paragraphs_with_overlap():
    text = "残差流像一条河。" * 90  # 720 characters, one paragraph
    chunks = chunk_document("doc", "doc.md", text, chunk_chars=120, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk["text"]) <= 140 for chunk in chunks)
    assert len({chunk["id"] for chunk in chunks}) == len(chunks)
    # consecutive windows overlap
    assert chunks[0]["text"][-20:] == chunks[1]["text"][:20]
    assert all(chunk["source"] == "doc.md" for chunk in chunks)


def test_chunk_document_tracks_headings_and_paragraphs():
    text = "# 第一节\n\n第一段内容。\n\n第二段内容。\n\n# 第二节\n\n第三段内容。"
    chunks = chunk_document("doc", "doc.md", text, chunk_chars=320, overlap=48)
    headings = [chunk["heading"] for chunk in chunks]
    assert "第一节" in headings and "第二节" in headings
    assert all(chunk["chars"] <= 320 for chunk in chunks)


def test_sample_memories_are_created_when_missing(tmp_path):
    memory_dir = tmp_path / "memory"

    async def go():
        store = RAGStore(memory_dir, embed_base_url=None)
        try:
            return await store.documents()
        finally:
            await store.aclose()

    result = run(go())
    files = sorted(path.name for path in memory_dir.glob("*.md"))
    assert len(files) == 2
    assert len(result["documents"]) == 2
    assert result["stats"]["documents"] == 2
    assert result["stats"]["chunks"] >= 2
    assert result["stats"]["backend"] == "bm25"
    assert result["memory_dir"] == str(memory_dir)


def test_search_returns_hybrid_fields_and_hits(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "透镜笔记.md").write_text(
        "# 替代透镜\n\n替代透镜模型用于观察内部张量，注意力与残差贡献都是合成数据，必须标注 surrogate。\n\n"
        "# 接口限制\n\nLM Studio 不暴露隐藏状态与注意力矩阵，只暴露文本与 top-logprobs。\n",
        encoding="utf-8",
    )

    async def go():
        store = RAGStore(memory_dir, embed_base_url=None)
        try:
            items = await store.search("替代透镜 注意力", top_k=2)
            docs = await store.documents()
            return items, docs, store.backend
        finally:
            await store.aclose()

    items, docs, backend = run(go())
    assert items, "expected at least one hit"
    assert backend == "bm25"
    assert items[0]["source"] == "透镜笔记.md"
    assert items[0]["cosine"] is None
    assert 0.0 <= items[0]["hybrid"] <= 1.0
    assert items[0]["bm25"] > 0
    assert items[0]["hits"], "expected readable hit words"
    assert any("透镜" in hit for hit in items[0]["hits"])
    assert docs["documents"][0]["source"] == "透镜笔记.md"
    assert docs["stats"]["documents"] == 1


def test_user_memories_are_never_overwritten(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    custom = memory_dir / "我的记忆.md"
    custom.write_text("# 私有\n\n这是用户自己写的记忆，不能被示例覆盖。", encoding="utf-8")

    async def go():
        store = RAGStore(memory_dir, embed_base_url=None)
        try:
            return await store.documents()
        finally:
            await store.aclose()

    result = run(go())
    assert custom.read_text(encoding="utf-8").startswith("# 私有")
    assert "我的记忆.md" in [document["source"] for document in result["documents"]]
    assert "项目原则.md" not in [document["source"] for document in result["documents"]]


def test_embedding_unavailable_falls_back_to_bm25(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "记忆.md").write_text("# 检索\n\n混合检索由 BM25 与 embedding 组成。", encoding="utf-8")

    async def go():
        store = RAGStore(memory_dir, embed_base_url="http://127.0.0.1:9", embed_timeout=0.5)
        try:
            items = await store.search("混合检索", top_k=1)
            stats = await store.documents()
            return items, stats["stats"], store.backend
        finally:
            await store.aclose()

    items, stats, backend = run(go())
    assert items
    assert backend == "bm25"
    assert stats["backend"] == "bm25"
    assert stats["embedding_model"] is None


def test_reindex_picks_up_new_files(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()

    async def go():
        store = RAGStore(memory_dir, embed_base_url=None)
        try:
            first = await store.documents()
            (memory_dir / "新增记忆.md").write_text("# 新增\n\n这里是后来加入的稀疏特征说明。", encoding="utf-8")
            stats = await store.reindex()
            items = await store.search("稀疏特征", top_k=1)
            return first, stats, items
        finally:
            await store.aclose()

    first, stats, items = run(go())
    assert len(first["documents"]) == 2  # two generated samples
    assert stats["documents"] == 3
    assert stats["chunks"] > first["stats"]["chunks"]
    assert items and items[0]["source"] == "新增记忆.md"


def test_extract_hits_prefers_longer_terms():
    hits = _extract_hits("替代透镜与残差流", "这里讲替代透镜，也讲残差流。")
    assert hits
    assert "替代透镜" in hits
    assert "透镜" not in hits  # contained in a longer hit


def test_empty_query_returns_no_results(tmp_path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "记忆.md").write_text("# x\n\ny", encoding="utf-8")

    async def go():
        store = RAGStore(memory_dir, embed_base_url=None)
        try:
            return await store.search("   ", top_k=3)
        finally:
            await store.aclose()

    assert run(go()) == []
