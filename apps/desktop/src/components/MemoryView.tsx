import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { Xray } from "../state/useXray";
import { Button, IconButton, Tag, Toggle } from "./Controls";
import { Icon } from "./Icon";
import { EmptyState, Panel } from "./Panel";
import { KVMemoryMap } from "./visuals/KVMemoryMap";
import { resolveKv } from "../lib/kv";

interface DocInfo {
  source: string;
  chunks: number;
  chars: number;
}

export function MemoryView({ xray }: { xray: Xray }) {
  const [query, setQuery] = useState("");
  const [docs, setDocs] = useState<DocInfo[]>([]);
  const [chunks, setChunks] = useState(0);
  const [embedding, setEmbedding] = useState("BM25");
  const [busy, setBusy] = useState(false);
  const [localError, setLocalError] = useState("");
  const [reindexed, setReindexed] = useState<number | null>(null);

  const loadDocs = async () => {
    try {
      const res = await api.ragDocuments();
      setDocs(res.documents ?? []);
      setChunks(res.chunks ?? 0);
      setEmbedding(res.embedding ?? "BM25");
    } catch {
      // The memory view still works as a search UI without the index summary.
    }
  };

  useEffect(() => {
    void loadDocs();
  }, []);

  const search = async () => {
    if (!query.trim() || busy) return;
    setBusy(true);
    setLocalError("");
    try {
      await xray.searchMemory(query);
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : "检索失败");
    } finally {
      setBusy(false);
    }
  };

  const reindex = async () => {
    setBusy(true);
    setLocalError("");
    try {
      const res = await api.reindexRag();
      setReindexed(res.chunks);
      await loadDocs();
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : "重建索引失败");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="view view-memory">
      <div className="section-head">
        <div>
          <h1>记忆宫殿</h1>
          <p>本地 Markdown 记忆库，BM25 与向量余弦的混合检索。召回过程可见，默认不注入提示。</p>
        </div>
        <div className="section-head-meta">
          <Tag tone="neutral">{embedding}</Tag>
          <span className="mono">{chunks} chunks · {docs.length} docs</span>
        </div>
      </div>

      <div className="memory-grid">
        <div className="memory-main">
          <Panel
            title="检索"
            kicker="混合召回 · 本地记忆"
            aside={
              <div className="memory-actions">
                <Toggle checked={xray.ragEnabled} onChange={xray.setRagEnabled} label="生成时检索" description="" />
                <Toggle checked={xray.ragInject} onChange={xray.setRagInject} label="注入提示" description="" />
                <IconButton label="重建记忆索引" onClick={reindex} disabled={busy}>
                  <Icon name="refresh" size={14} />
                </IconButton>
              </div>
            }
          >
            <form
              className="search-row"
              onSubmit={(event) => {
                event.preventDefault();
                void search();
              }}
            >
              <Icon name="search" size={15} />
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="输入查询，例如：残差流 与 注意力 的关系"
                aria-label="检索本地记忆"
              />
              <Button variant="outline" size="sm" type="submit" disabled={!query.trim() || busy}>
                {busy ? "检索中" : "检索"}
              </Button>
            </form>
            {localError ? <p className="inline-error">{localError}</p> : null}
            {reindexed !== null ? <p className="inline-note">索引已重建：{reindexed} 个片段。</p> : null}

            {xray.ragItems.length === 0 ? (
              <EmptyState
                title="还没有召回结果"
                body="输入查询，查看每个记忆片段的余弦相似度、BM25 与混合分数。生成时启用检索，召回会自动出现在这里。"
                compact
              />
            ) : (
              <div className="rag-results">
                {xray.ragItems.map((item, index) => (
                  <article key={item.id} className="rag-item">
                    <header className="rag-head">
                      <span className="rag-rank num">{String(index + 1).padStart(2, "0")}</span>
                      <span className="rag-source mono">{item.source}{item.heading ? ` · ${item.heading}` : ""}</span>
                      <span className="rag-hybrid num" title="混合得分">{item.hybrid.toFixed(3)}</span>
                    </header>
                    <p className="rag-text">{item.text}</p>
                    <footer className="rag-foot">
                      <span className="rag-score">
                        <span className="rag-score-label">cos</span>
                        <span className="rag-score-track"><span style={{ width: `${Math.max(0, (item.cosine ?? 0) * 100)}%` }} /></span>
                        <span className="num">{item.cosine === null ? "—" : item.cosine.toFixed(2)}</span>
                      </span>
                      <span className="rag-score">
                        <span className="rag-score-label">BM25</span>
                        <span className="rag-score-track"><span style={{ width: `${Math.min(100, item.bm25 * 100)}%` }} /></span>
                        <span className="num">{item.bm25.toFixed(2)}</span>
                      </span>
                      {item.hits.length ? <span className="rag-hits">命中：{item.hits.slice(0, 5).join(" / ")}</span> : null}
                    </footer>
                  </article>
                ))}
              </div>
            )}
          </Panel>
        </div>
        <div className="memory-side">
          <Panel title="记忆库" kicker="memory/*.md" aside={<span className="panel-note">{docs.length} 个文件</span>}>
            {docs.length === 0 ? (
              <EmptyState title="记忆库为空" body="在 memory/ 中放入 Markdown 文件，点击重建索引即可参与检索。" compact />
            ) : (
              <div className="doc-list">
                {docs.map((doc) => (
                  <div key={doc.source} className="doc-row">
                    <span className="doc-name mono">{doc.source}</span>
                    <span className="doc-meta num">{doc.chunks} 块</span>
                    <span className="doc-meta num">{doc.chars.toLocaleString("zh-CN")} 字</span>
                  </div>
                ))}
              </div>
            )}
            <p className="panel-copy faint">
              记忆只在本机读取；使用 LM Studio 的 embedding 端点生成向量，不可用时自动退化为字符 BM25。
            </p>
          </Panel>
          <KVMemoryMap kv={resolveKv(xray.kv, xray.latestProbe?.kv)} contextTokens={xray.contextTokens.length} height={180} />
        </div>
      </div>
    </div>
  );
}
