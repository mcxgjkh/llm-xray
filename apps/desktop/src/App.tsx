import { useEffect, useMemo, useRef, useState } from "react";
import { useXray, type TabId } from "./state/useXray";
import { Icon, BrandMark } from "./components/Icon";
import { IconButton, Tag } from "./components/Controls";
import { ChatDock } from "./components/ChatDock";
import { PerformanceStrip } from "./components/PerformanceStrip";
import { Timeline } from "./components/Timeline";
import { ObservatoryView } from "./components/ObservatoryView";
import { AttentionView } from "./components/AttentionView";
import { VectorsView } from "./components/VectorsView";
import { MemoryView } from "./components/MemoryView";
import { ReplayView } from "./components/ReplayView";
import { SettingsView } from "./components/SettingsView";
import { DocsView } from "./components/DocsView";
import { useSonification } from "./components/visuals/Sonification";
import { seconds } from "./lib/format";
import "./styles/app.css";

const NAV: { id: TabId; label: string; icon: string }[] = [
  { id: "observatory", label: "观测台", icon: "observatory" },
  { id: "attention", label: "注意力", icon: "attention" },
  { id: "vectors", label: "向量空间", icon: "vectors" },
  { id: "memory", label: "记忆宫殿", icon: "memory" },
  { id: "replay", label: "时间回放", icon: "replay" },
];

const META: Record<TabId, { title: string; subtitle: string }> = {
  observatory: {
    title: "观测台",
    subtitle: "实时观察一次生成：概率、注意力、透镜与残差流在同一条时间线上。",
  },
  attention: { title: "注意力", subtitle: "逐层、逐头查看当前 Token 的上下文分配。" },
  vectors: { title: "向量空间", subtitle: "语义投影、隐状态摘要与稀疏特征。" },
  memory: { title: "记忆宫殿", subtitle: "本地记忆的混合检索与召回轨迹。" },
  replay: { title: "时间回放", subtitle: "按步保存，按步回看。" },
  settings: { title: "设置", subtitle: "连接、透镜资源与本地偏好。" },
  docs: { title: "技术文档", subtitle: "使用说明、架构、可解释性与版本记录。" },
};

export default function App() {
  const xray = useXray();
  const [selectedLayer, setSelectedLayer] = useState(12);
  const [selectedHead, setSelectedHead] = useState<number | "mean">("mean");
  const [playing, setPlaying] = useState(false);
  const autoDemo = useRef(false);
  const tab = xray.selectedTab;

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const requestedTab = params.get("tab") as TabId | null;
    if (requestedTab && META[requestedTab]) xray.setSelectedTab(requestedTab);
    // A deterministic visual smoke test, never used by normal launches:
    // /?demo=1 starts the offline demo once, without touching LM Studio.
    if (params.get("demo") === "1" && xray.connection === "open" && !autoDemo.current) {
      autoDemo.current = true;
      window.setTimeout(() => {
        xray.send("为什么说残差流像一条河？", { mode: "demo", model: "demo" });
      }, 350);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [xray.connection]);
  const meta = META[tab];
  const dataTab = tab !== "settings" && tab !== "docs";
  const sessionAge = xray.startedAt ? Math.max(0, (Date.now() - xray.startedAt) / 1000) : 0;

  useSonification({
    enabled: xray.sound,
    entropy: xray.activeStep?.entropy ?? null,
    chosenProb: xray.activeStep?.chosen_prob ?? null,
    step: xray.activeStep?.index ?? null,
  });

  const modelState = useMemo(() => {
    const model = xray.models.find((item) => item.id === xray.selectedModel);
    return model?.state ?? "unknown";
  }, [xray.models, xray.selectedModel]);

  const probeLayers = xray.activeProbe?.attention.layers;
  const probeHeads = xray.activeProbe?.attention.head_count ?? 0;
  useEffect(() => {
    if (!probeLayers?.length) return;
    if (!probeLayers.includes(selectedLayer)) {
      const preferred = probeLayers.includes(12) ? 12 : probeLayers[probeLayers.length - 1];
      setSelectedLayer(preferred);
    }
  }, [probeLayers, selectedLayer]);
  useEffect(() => {
    if (selectedHead !== "mean" && probeHeads > 0 && selectedHead >= probeHeads) setSelectedHead("mean");
  }, [probeHeads, selectedHead]);

  return (
    <div className="app-shell">
      <nav className="rail" aria-label="主导航">
        <div className="rail-brand" title="LLM-Xray">
          <BrandMark size={26} />
        </div>
        <div className="rail-group">
          {NAV.map((item) => (
            <button
              key={item.id}
              className={`rail-item ${tab === item.id ? "is-active" : ""}`.trim()}
              aria-label={item.label}
              aria-current={tab === item.id ? "page" : undefined}
              title={item.label}
              onClick={() => {
                xray.setSelectedTab(item.id);
                setPlaying(false);
              }}
            >
              <Icon name={item.icon} size={17} />
            </button>
          ))}
        </div>
        <div className="rail-spacer" />
        <div className="rail-group rail-bottom">
          <button
            className={`rail-item ${tab === "docs" ? "is-active" : ""}`.trim()}
            aria-label="技术文档"
            title="技术文档"
            onClick={() => xray.setSelectedTab("docs")}
          >
            <Icon name="docs" size={17} />
          </button>
          <button
            className={`rail-item ${tab === "settings" ? "is-active" : ""}`.trim()}
            aria-label="设置"
            title="设置"
            onClick={() => xray.setSelectedTab("settings")}
          >
            <Icon name="settings" size={17} />
          </button>
        </div>
        <div className="rail-version mono">v{xray.config?.version ?? "1.0.0"}</div>
      </nav>

      <div className="app-body">
        <header className="topbar">
          <div className="wordmark">
            <span className="wordmark-name">LLM-Xray</span>
            <span className="wordmark-sub">本地可解释性观测台</span>
          </div>
          <div className="topbar-center">
            <label className="model-select-wrap">
              <span className="topbar-label">引擎</span>
              <select
                value={xray.mode}
                onChange={(event) => xray.setMode(event.target.value as typeof xray.mode)}
                aria-label="选择推理引擎"
              >
                <option value="lmstudio">LM Studio</option>
                <option value="native">原生精确</option>
                <option value="demo">演示</option>
              </select>
            </label>
            {xray.mode === "native" ? (
              <label className="model-select-wrap">
                <span className="topbar-label">精确模型</span>
                <select
                  value={xray.nativeModel}
                  onChange={(event) => xray.setNativeModel(event.target.value)}
                  aria-label="选择原生精确模型"
                >
                  <option value="Qwen/Qwen2.5-0.5B-Instruct">Qwen2.5-0.5B（本地）</option>
                  <option value="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B">R1-Distill-1.5B（需本地）</option>
                  {!["Qwen/Qwen2.5-0.5B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"].includes(xray.nativeModel) ? (
                    <option value={xray.nativeModel}>{xray.nativeModel}</option>
                  ) : null}
                </select>
              </label>
            ) : (
              <label className="model-select-wrap">
                <span className="topbar-label">模型</span>
                <select
                  value={xray.selectedModel}
                  onChange={(event) => xray.setSelectedModel(event.target.value)}
                  aria-label="选择 LM Studio 模型"
                >
                  {xray.models.length === 0 ? <option value="">未发现模型</option> : null}
                  {xray.models.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.id}
                    </option>
                  ))}
                </select>
              </label>
            )}
            <span className={`conn-status conn-${xray.connection}`}>
              <span className="conn-dot" aria-hidden="true" />
              {xray.mode === "native"
                ? xray.native.status === "ready"
                  ? "原生精确就绪"
                  : xray.native.status === "loading"
                    ? "原生模型加载中"
                    : "原生模型未加载"
                : xray.connection === "open"
                  ? xray.lmReachable
                    ? "LM Studio 在线"
                    : "后端在线 · 模型离线"
                  : xray.connection === "connecting"
                    ? "正在连接后端"
                    : "后端离线"}
            </span>
            {xray.mode === "native" ? (
              <Tag tone={xray.native.status === "ready" ? "positive" : "warn"} title="原生精确张量">
                {xray.native.status === "ready" ? "精确张量" : "未加载"}
              </Tag>
            ) : (
              <Tag tone={modelState === "loaded" ? "positive" : "muted"} title="LM Studio 模型加载状态">
                {modelState === "loaded" ? "已加载" : modelState === "not-loaded" ? "未加载" : "模型状态未知"}
              </Tag>
            )}
          </div>
          <div className="topbar-right">
            <span className="topbar-session">
              {xray.session ? (
                <>
                  <span className="mono">{xray.session.id.slice(0, 14)}</span>
                  <span className="faint">{xray.generating ? `生成 ${seconds(sessionAge)}` : xray.elapsed ? `用时 ${seconds(xray.elapsed)}` : "已就绪"}</span>
                </>
              ) : (
                <span className="faint">等待第一次生成</span>
              )}
            </span>
            {xray.chatCollapsed ? (
              <IconButton label="展开对话窗" onClick={() => xray.setChatCollapsed(false)}>
                <Icon name="expand" size={15} />
              </IconButton>
            ) : null}
          </div>
        </header>

        <div className="workspace">
          <main className="canvas">
            {tab === "observatory" ? (
              <div className="view-head">
                <div>
                  <h1>{meta.title}</h1>
                  <p>{meta.subtitle}</p>
                </div>
                <div className="view-head-actions">
                  <span className={`live-stamp ${xray.generating ? "is-live" : ""}`.trim()}>
                    <span className="live-dot" aria-hidden="true" />
                    {xray.generating ? "实时流" : xray.finishReason ? "已结束" : "待命"}
                  </span>
                  <Tag tone={xray.activeProbe?.surrogate ? "warn" : xray.activeProbe ? "positive" : "neutral"}>
                    {xray.activeProbe ? (xray.activeProbe.surrogate ? "替代透镜" : "精确张量") : "仅 API 遥测"}
                  </Tag>
                  <IconButton label="导出 JSONL" onClick={xray.exportSession} disabled={!xray.sessionId}>
                    <Icon name="download" size={15} />
                  </IconButton>
                  <IconButton label="新建对话" onClick={xray.resetConversation}>
                    <Icon name="plus" size={15} />
                  </IconButton>
                </div>
              </div>
            ) : null}

            {dataTab ? (
              <PerformanceStrip
                metrics={xray.metrics}
                history={xray.metricHistory}
                tokens={xray.steps.length}
                elapsed={xray.elapsed}
                ttft={xray.steps[0]?.ttft_ms ?? null}
                usage={xray.usage}
                session={xray.session}
                finishReason={xray.finishReason}
                generating={xray.generating}
                contextTokens={xray.contextTokens.length}
              />
            ) : null}

            <div className="canvas-scroll">
              {tab === "observatory" ? (
                <ObservatoryView
                  xray={xray}
                  selectedLayer={selectedLayer}
                  setSelectedLayer={setSelectedLayer}
                  selectedHead={selectedHead}
                  setSelectedHead={setSelectedHead}
                />
              ) : null}
              {tab === "attention" ? (
                <AttentionView
                  xray={xray}
                  selectedLayer={selectedLayer}
                  setSelectedLayer={setSelectedLayer}
                  selectedHead={selectedHead}
                  setSelectedHead={setSelectedHead}
                />
              ) : null}
              {tab === "vectors" ? <VectorsView xray={xray} /> : null}
              {tab === "memory" ? <MemoryView xray={xray} /> : null}
              {tab === "replay" ? <ReplayView xray={xray} /> : null}
              {tab === "settings" ? <SettingsView xray={xray} /> : null}
              {tab === "docs" ? <DocsView /> : null}
            </div>

            {dataTab ? (
              <Timeline
                steps={xray.steps}
                probes={xray.probes}
                activeStep={xray.activeStepIndex}
                onSelectStep={(index) => {
                  setPlaying(false);
                  void xray.selectStep(index);
                }}
                session={xray.session}
                exporting={xray.exporting}
                onExport={xray.exportSession}
                replayMode={xray.replayMode}
                playing={playing}
                onPlayingChange={setPlaying}
              />
            ) : null}
          </main>
          <ChatDock xray={xray} />
        </div>
      </div>
      <div className="crt-overlay" aria-hidden="true" />
      {xray.exportNotice ? (
        <div className="app-toast" role="status" onClick={() => xray.setExportNotice("")}>
          {xray.exportNotice}
        </div>
      ) : null}
    </div>
  );
}
