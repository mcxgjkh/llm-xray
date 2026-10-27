import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { EmptyState } from "./Panel";
import { Markdown } from "./Markdown";

const DOCS = [
  { id: "USAGE", label: "使用说明" },
  { id: "ARCHITECTURE", label: "架构" },
  { id: "INTERPRETABILITY", label: "可解释性" },
  { id: "API", label: "接口" },
  { id: "LIMITATIONS", label: "边界与排错" },
  { id: "VERSIONS", label: "版本记录" },
];

const FALLBACK = `# 使用说明尚未载入

后端会从 \`docs/\` 提供 Markdown。若这里为空，请确认：

1. 已从项目根目录启动后端；
2. \`docs/USAGE.md\` 存在；
3. 当前版本与 \`VERSION\` 一致。

核心操作：在右侧对话窗发送消息；左侧观测台同步显示概率云、注意力、透镜与残差；底部时间轴可以回放每一步。`;

export function DocsView() {
  const [current, setCurrent] = useState("USAGE");
  const [content, setContent] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (content[current]) return;
    let cancelled = false;
    setLoading(true);
    const load = async () => {
      for (const name of [`${current}.md`, current]) {
        try {
          const text = await api.doc(name);
          if (text && !text.trim().startsWith("{")) {
            if (!cancelled) setContent((previous) => ({ ...previous, [current]: text }));
            return;
          }
        } catch {
          // try the other spelling
        }
      }
      if (!cancelled) setContent((previous) => ({ ...previous, [current]: FALLBACK }));
    };
    void load().finally(() => {
      if (!cancelled) setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [content, current]);

  return (
    <div className="view view-docs">
      <div className="docs-layout">
        <aside className="docs-nav">
          <p className="docs-nav-title">技术文档</p>
          {DOCS.map((doc) => (
            <button
              key={doc.id}
              className={current === doc.id ? "is-active" : ""}
              onClick={() => setCurrent(doc.id)}
            >
              {doc.label}
            </button>
          ))}
          <p className="docs-nav-note">
            文档与代码同版本维护。正式发布前执行版本一致性检查。
          </p>
          <a className="docs-external" href="/documentation/" target="_blank" rel="noreferrer">
            打开独立文档站
          </a>
        </aside>
        <article className="docs-reader">
          <header className="docs-reader-head">
            <h1>{DOCS.find((doc) => doc.id === current)?.label ?? current}</h1>
            <span className="mono">docs/{current}.md</span>
          </header>
          {loading && !content[current] ? <p className="muted">正在读取文档</p> : null}
          {content[current] ? (
            <Markdown source={content[current]} className="docs-markdown" />
          ) : (
            <EmptyState title="没有找到文档" body="后端文档接口不可用。可以直接阅读项目 docs/ 目录中的 Markdown。" compact />
          )}
        </article>
      </div>
    </div>
  );
}
