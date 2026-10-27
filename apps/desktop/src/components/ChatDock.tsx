import { useEffect, useMemo, useRef, useState } from "react";
import { clamp, int, tokenLabel } from "../lib/format";
import { estimateMemory } from "../lib/memory";
import type { Xray } from "../state/useXray";
import { Button, IconButton, SliderRow, Tag, Toggle } from "./Controls";
import { Icon } from "./Icon";
import { Markdown } from "./Markdown";
import { SonificationToggle } from "./visuals/SonificationToggle";

const SUGGESTIONS = [
  "用一句话解释 Transformer 的残差流，为什么它像一条河？",
  "设计一个实验，判断某个注意力头是否在做指代消解。",
  "把“调优透镜”讲给一个刚学完线性代数的学生。",
];

export function ChatDock({ xray }: { xray: Xray }) {
  const [draft, setDraft] = useState("");
  const [dragging, setDragging] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [openReasoning, setOpenReasoning] = useState<Set<string>>(new Set());
  const [nearBottom, setNearBottom] = useState(true);
  const listRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const { chat, generating, sampling, connection, error, statusMessage, session } = xray;
  const selectedModelState = xray.models.find((model) => model.id === xray.selectedModel)?.state;
  const needsModelLoad =
    xray.mode === "lmstudio" && connection === "open" && selectedModelState === "not-loaded";
  const maxOutputLimit =
    xray.mode === "native"
      ? 32768
      : xray.models.find((model) => model.id === xray.selectedModel)?.max_context_length ?? 131072;
  const selectedModelInfo = xray.models.find((model) => model.id === xray.selectedModel);
  const memoryEstimate = estimateMemory(xray.mode, selectedModelInfo, xray.nativeModel, sampling.max_tokens, xray.nativeDevice);

  useEffect(() => {
    const node = listRef.current;
    if (!node || !nearBottom) return;
    node.scrollTop = node.scrollHeight;
  }, [chat, nearBottom]);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "0px";
    textarea.style.height = `${Math.min(160, textarea.scrollHeight)}px`;
  }, [draft]);

  useEffect(() => {
    if (!dragging) return;
    const move = (event: PointerEvent) => {
      const next = clamp(window.innerWidth - event.clientX, 320, Math.min(680, window.innerWidth - 560));
      xray.setChatWidth(next);
    };
    const up = () => setDragging(false);
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
  }, [dragging, xray]);

  const send = () => {
    if (!draft.trim() || generating) return;
    xray.send(draft);
    setDraft("");
  };

  const statusLabel = useMemo(() => {
    if (connection === "connecting") return "连接后端";
    if (connection === "error") return "连接异常";
    if (connection !== "open") return "后端离线";
    if (xray.mode === "demo") return "离线演示";
    if (xray.mode === "native") {
      if (xray.native.status === "ready") return "原生精确就绪";
      if (xray.native.status === "loading") return "原生模型加载中";
      return "原生模型未加载";
    }
    return xray.lmReachable ? "LM Studio 在线" : "LM Studio 未连接";
  }, [connection, xray.lmReachable, xray.mode, xray.native.status]);

  if (xray.chatCollapsed) {
    return (
      <aside className="chat-dock is-collapsed" aria-label="对话">
        <button className="chat-expand" onClick={() => xray.setChatCollapsed(false)} aria-label="展开对话">
          <Icon name="expand" size={15} />
          <span>对话</span>
          {chat.length ? <span className="chat-collapsed-count num">{chat.filter((entry) => entry.role === "assistant").length}</span> : null}
        </button>
      </aside>
    );
  }

  return (
    <aside className="chat-dock" style={{ width: xray.chatWidth }} aria-label="LLM 对话">
      <div
        className={`chat-resize ${dragging ? "is-dragging" : ""}`.trim()}
        role="separator"
        aria-orientation="vertical"
        aria-label="调节对话窗宽度"
        onPointerDown={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
      />
      <header className="chat-head">
        <div className="chat-head-main">
          <div className="chat-head-title">
            <span className={`conn-dot conn-${connection}`} aria-hidden="true" />
            <h2>对话</h2>
            <Tag tone={xray.mode === "demo" ? "warn" : "accent"}>
              {xray.mode === "demo" ? "演示" : xray.mode === "native" ? "原生精确" : "真实遥测"}
            </Tag>
          </div>
          <span className="chat-head-sub">
            {statusLabel}
            {session ? ` · ${session.model}` : ""}
          </span>
        </div>
        <div className="chat-head-actions">
          <IconButton label="新建对话" onClick={xray.resetConversation}>
            <Icon name="plus" size={15} />
          </IconButton>
          <IconButton
            label={xray.exporting ? "正在导出 JSONL" : "导出 JSONL"}
            onClick={xray.exportSession}
            disabled={!xray.sessionId || xray.exporting}
          >
            <Icon name="download" size={15} />
          </IconButton>
          <IconButton label="收起对话窗" onClick={() => xray.setChatCollapsed(true)}>
            <Icon name="collapse" size={15} />
          </IconButton>
        </div>
      </header>

      <div className="chat-flags">
        <span className="flag" title="当前模型与采样参数">
          {session?.mode === "demo"
            ? "demo · 离线演示"
            : xray.mode === "native" || session?.mode === "native"
              ? `原生 · ${xray.native.model ?? xray.nativeModel}`
              : xray.selectedModel || "未选择模型"}
        </span>
        {xray.mode !== "native" ? (
          <button
            className="flag flag-button"
            title="LM Studio 不公开内部张量；透镜来自独立小模型，界面已标注。点击加载/查看状态。"
            onClick={() => {
              if (xray.lens.status === "unloaded" || xray.lens.status === "error") void xray.loadLens();
            }}
          >
            透镜 <span className={xray.lens.status === "ready" ? "flag-on" : ""}>{lensWord(xray.lens.status)}</span>
          </button>
        ) : (
          <span className="flag flag-on" title="生成与观测来自同一次前向">
            精确张量
          </span>
        )}
        {xray.ragEnabled ? <span className="flag">记忆检索</span> : null}
      </div>

      {xray.mode === "lmstudio" && xray.lens.status === "unloaded" && !xray.lensEnabled ? (
        <button className="lens-cta" onClick={() => xray.setLensEnabled(true)}>
          已加载的透镜尚未启用：点击开启“允许透镜采样”
        </button>
      ) : null}
      {xray.mode === "lmstudio" && xray.lens.status === "unloaded" && xray.lensEnabled ? (
        <button className="lens-cta" onClick={() => void xray.loadLens()}>
          注意力/Tuned Lens 尚未启用：点击加载本地替代透镜（CPU 低优先级，约 20-30 秒）
        </button>
      ) : null}
      {xray.lens.status === "loading" ? (
        <div className="lens-cta is-loading">正在加载透镜，完成后下一次生成会携带注意力与 Tuned Lens 数据</div>
      ) : null}
      {xray.lens.status === "error" ? (
        <div className="lens-cta is-error">
          透镜不可用：{xray.lens.message ?? "请运行 scripts/fetch_lens_model.ps1"}
          <button onClick={() => void xray.loadLens()}>重试</button>
        </div>
      ) : null}

      {error ? (
        <div className="chat-error" role="alert">
          <Icon name="warning" size={14} />
          <span>{error}</span>
          <button onClick={() => xray.setError("")} aria-label="关闭错误提示">
            <Icon name="close" size={12} />
          </button>
        </div>
      ) : null}
      {!error && statusMessage ? <div className="chat-status">{statusMessage}</div> : null}
      {needsModelLoad && !generating ? (
        <div className="chat-status chat-model-hint">
          模型尚未加载。请先在 LM Studio 中加载 <span className="mono">{xray.selectedModel}</span>，应用不会为了探测而自动装载 8B 模型。
        </div>
      ) : null}
      {session?.mode === "lmstudio" && session.capabilities.logprobs === false && !generating ? (
        <button
          className="lens-cta is-error"
          onClick={() => {
            xray.setMode("native");
            xray.setSelectedTab("settings");
          }}
        >
          当前 LM Studio 没有返回 top_logprobs，概率云为空。切换到原生精确模式可得到完整的概率、注意力与 Tuned Lens。点击前往设置。
        </button>
      ) : null}

      <div
        className="chat-list"
        ref={listRef}
        onScroll={(event) => {
          const el = event.currentTarget;
          setNearBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 60);
        }}
      >
        {chat.length === 0 ? (
          <div className="chat-empty">
            <p className="chat-empty-title">观测一次生成</p>
            <p className="chat-empty-body">
              发送一个问题。左侧会同步展开概率云、注意力与透镜数据；右侧只负责对话，不遮挡观测。
            </p>
            <div className="chat-suggestions">
              {SUGGESTIONS.map((prompt) => (
                <button key={prompt} onClick={() => xray.send(prompt)} disabled={generating}>
                  {prompt}
                </button>
              ))}
            </div>
            <div className="chat-empty-actions">
              <Button variant="outline" size="sm" onClick={() => xray.send(SUGGESTIONS[0], { mode: "demo" })} disabled={generating}>
                <Icon name="spark" size={13} /> 演示模式
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => xray.send(SUGGESTIONS[0], { mode: "native" })}
                disabled={generating || !xray.native.available}
                title="在进程内运行本地 Qwen2.5-0.5B，生成与注意力/概率来自同一次前向；CPU 较慢"
              >
                <Icon name="layers" size={13} /> 原生精确
              </Button>
              <span className="muted">演示不加载模型；原生精确首次加载约 10-30 秒</span>
            </div>
          </div>
        ) : null}

        {chat.map((entry) => (
          <article key={entry.id} className={`chat-entry chat-${entry.role}`}>
            <header className="chat-meta">
              <span>{entry.role === "user" ? "你" : "模型"}</span>
              {entry.role === "assistant" && entry.kind ? (
                <span className="chat-kind">
                  {entry.kind === "demo" ? "演示" : entry.kind === "native" ? "原生精确" : "LM Studio"}
                </span>
              ) : null}
            </header>
            {entry.role === "assistant" && entry.reasoning ? (
              <div className={`reasoning ${openReasoning.has(entry.id) ? "is-open" : ""}`.trim()}>
                <button
                  className="reasoning-toggle"
                  onClick={() =>
                    setOpenReasoning((previous) => {
                      const next = new Set(previous);
                      if (next.has(entry.id)) next.delete(entry.id);
                      else next.add(entry.id);
                      return next;
                    })
                  }
                >
                  <Icon name={openReasoning.has(entry.id) ? "chevronDown" : "chevronRight"} size={13} />
                  思考过程
                  <span className="reasoning-count num">{entry.reasoning.length}</span>
                </button>
                {openReasoning.has(entry.id) ? <pre className="reasoning-text">{entry.reasoning}</pre> : null}
              </div>
            ) : null}
            {entry.role === "assistant" ? (
              entry.streaming ? (
                <div className="chat-text is-streaming">
                  {entry.content || (!entry.reasoning ? <span className="chat-waiting">等待第一个 Token</span> : null)}
                  {generating ? <span className="stream-caret" aria-hidden="true" /> : null}
                </div>
              ) : entry.content ? (
                <Markdown source={entry.content} className="chat-text" />
              ) : (
                <div className="chat-text muted">没有返回文本。</div>
              )
            ) : (
              <p className="chat-text">{entry.content}</p>
            )}
            <footer className="chat-entry-foot">
              <span>
                {entry.role === "assistant"
                  ? entry.kind === "demo"
                    ? "演示生成"
                    : entry.kind === "native"
                      ? "原生精确生成"
                      : "LM Studio 生成"
                  : "本轮输入"}
              </span>
              {entry.role === "assistant" && entry.content && !entry.streaming ? (
                <button
                  className="chat-copy"
                  onClick={() => navigator.clipboard?.writeText(entry.content).catch(() => undefined)}
                >
                  复制
                </button>
              ) : null}
            </footer>
          </article>
        ))}
        {generating ? (
          <div className="chat-generating">
            <span className="pulse-dot" />
            {xray.activeStep ? `第 ${int(xray.activeStep.index + 1)} 步 · ${tokenLabel(xray.activeStep.delta)}` : "模型正在处理"}
          </div>
        ) : null}
      </div>

      <div className="chat-controls">
        <div className="chat-control-row">
          <button className={`chat-param-toggle ${advanced ? "is-open" : ""}`.trim()} onClick={() => setAdvanced(!advanced)}>
            <Icon name="settings" size={13} />
            采样
            <span className="num">
              T {sampling.temperature.toFixed(2)} · P {sampling.top_p.toFixed(2)} · K {sampling.top_k} · {sampling.max_tokens}
            </span>
          </button>
          <SonificationToggle
            enabled={xray.sound}
            onToggle={xray.setSound}
            entropy={xray.activeStep?.entropy ?? null}
            chosenProb={xray.activeStep?.chosen_prob ?? null}
          />
          <Toggle checked={xray.crt} onChange={xray.setCrt} label="CRT" description="" />
        </div>
        {advanced ? (
          <div className="chat-advanced">
            <SliderRow
              label="温度"
              value={sampling.temperature}
              min={0.05}
              max={2}
              step={0.05}
              format={(value) => value.toFixed(2)}
              onChange={(temperature) => xray.setSampling((current) => ({ ...current, temperature }))}
              hint="低温度更确定，高温度更平坦"
            />
            <SliderRow
              label="Top-P"
              value={sampling.top_p}
              min={0.05}
              max={1}
              step={0.01}
              format={(value) => value.toFixed(2)}
              onChange={(top_p) => xray.setSampling((current) => ({ ...current, top_p }))}
            />
            <SliderRow
              label="Top-K"
              value={sampling.top_k}
              min={1}
              max={100}
              step={1}
              format={(value) => String(Math.round(value))}
              onChange={(top_k) => xray.setSampling((current) => ({ ...current, top_k }))}
            />
            <SliderRow
              label="重复惩罚"
              value={sampling.repetition_penalty}
              min={1}
              max={1.5}
              step={0.01}
              format={(value) => value.toFixed(2)}
              onChange={(repetition_penalty) => xray.setSampling((current) => ({ ...current, repetition_penalty }))}
            />
            <SliderRow
              label="最大输出"
              value={sampling.max_tokens}
              min={256}
              max={maxOutputLimit}
              step={256}
              format={(value) => `${Math.round(value)} tokens`}
              onChange={(max_tokens) => xray.setSampling((current) => ({ ...current, max_tokens }))}
              hint="上限取当前模型的 max_context_length；实际输出还受已加载上下文限制"
            />
            <div className="memory-estimate" title={memoryEstimate.detail}>
              <span>Estimated Memory Usage</span>
              <strong className="num">~{memoryEstimate.totalGb.toFixed(1)} GB</strong>
              <small>{memoryEstimate.detail}</small>
            </div>
            <div className="chat-advanced-toggles">
              <Toggle checked={xray.ragEnabled} onChange={xray.setRagEnabled} label="记忆检索" description="查询本地记忆库" />
              <Toggle checked={xray.lensEnabled} onChange={xray.setLensEnabled} label="透镜采样" description="独立小模型，CPU 低优先级" disabled={xray.lens.status !== "ready"} />
              <Toggle checked={xray.ragInject} onChange={xray.setRagInject} label="注入记忆" description="把召回内容放入系统提示，会增加 Token" />
            </div>
            <p className="chat-param-note">显示参数即时重算概率云；实际采样参数在下一个请求生效。</p>
          </div>
        ) : null}
        <div className="chat-composer">
          <textarea
            ref={textareaRef}
            value={draft}
            placeholder={generating ? "正在生成，可先构思下一句" : "输入消息，Enter 发送，Shift+Enter 换行"}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                send();
              }
            }}
            rows={1}
          />
          {generating ? (
            <Button variant="danger" size="sm" onClick={xray.stop} aria-label="停止生成">
              <Icon name="stop" size={13} /> 停止
            </Button>
          ) : (
            <Button variant="primary" size="sm" onClick={send} disabled={!draft.trim() || connection !== "open"}>
              <Icon name="send" size={13} /> 发送
            </Button>
          )}
        </div>
        <div className="chat-footnote">
          <span>{xray.usage ? `${int(xray.usage.total_tokens)} tokens` : "本地运行 · 数据不离开这台电脑"}</span>
          <span className="mono">{xray.session ? xray.session.id.slice(0, 18) : "未开始"}</span>
        </div>
      </div>
    </aside>
  );
}

function lensWord(status: string) {
  if (status === "ready") return "就绪";
  if (status === "loading") return "加载中";
  if (status === "error") return "异常";
  return "未启用";
}
