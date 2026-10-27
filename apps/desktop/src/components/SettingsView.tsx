import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { estimateMemory } from "../lib/memory";
import type { Xray } from "../state/useXray";
import { Button, Field, IconButton, KeyValue, Segmented, SliderRow, Tag, Toggle } from "./Controls";
import { Icon } from "./Icon";
import { Panel } from "./Panel";

export function SettingsView({ xray }: { xray: Xray }) {
  const [lensModel, setLensModel] = useState("Qwen/Qwen2.5-0.5B-Instruct");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [docs, setDocs] = useState<{ source: string; chunks: number }[]>([]);
  const [chunks, setChunks] = useState<number | null>(null);

  const lens = xray.lens;
  const native = xray.native;
  const maxOutputLimit =
    xray.mode === "native"
      ? 32768
      : xray.models.find((model) => model.id === xray.selectedModel)?.max_context_length ?? 131072;
  const selectedModelInfo = xray.models.find((model) => model.id === xray.selectedModel);
  const memoryEstimate = estimateMemory(xray.mode, selectedModelInfo, xray.nativeModel, xray.sampling.max_tokens, xray.nativeDevice);
  const [nativeBusy, setNativeBusy] = useState(false);
  const [nativeMessage, setNativeMessage] = useState("");

  useEffect(() => {
    void api
      .ragDocuments()
      .then((res) => {
        setDocs(res.documents ?? []);
        setChunks(res.chunks ?? 0);
      })
      .catch(() => undefined);
  }, []);

  const loadLens = async () => {
    setBusy(true);
    setMessage("");
    try {
      const res = await xray.loadLens(lensModel.trim() || null);
      setMessage(res?.message ?? "透镜模型已加载");
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "透镜加载失败");
    } finally {
      setBusy(false);
    }
  };

  const loadNative = async () => {
    setNativeBusy(true);
    setNativeMessage("");
    try {
      const res = await xray.loadNative(xray.nativeModel.trim() || null);
      setNativeMessage(res.message ?? "原生精确模型已加载");
    } catch (err) {
      setNativeMessage(err instanceof Error ? err.message : "原生模型加载失败");
    } finally {
      setNativeBusy(false);
    }
  };

  const unloadNative = async () => {
    setNativeBusy(true);
    try {
      await xray.unloadNative();
      setNativeMessage("原生精确模型已卸载，资源已释放");
    } finally {
      setNativeBusy(false);
    }
  };

  const unloadLens = async () => {
    setBusy(true);
    try {
      await xray.unloadLens();
      setMessage("透镜模型已卸载，资源已释放");
    } finally {
      setBusy(false);
    }
  };

  const reindex = async () => {
    setBusy(true);
    try {
      const res = await api.reindexRag();
      setChunks(res.chunks);
      setMessage(`记忆索引已重建：${res.chunks} 个片段`);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "重建索引失败");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="view view-settings">
      <div className="section-head">
        <div>
          <h1>设置</h1>
          <p>连接、透镜资源、采样默认值和本地数据。所有设置都不离开这台电脑。</p>
        </div>
        <div className="section-head-meta">
          <Tag tone="neutral">v{xray.config?.version ?? "1.0.0"}</Tag>
          <span className="mono">{xray.connection === "open" ? "后端在线" : "后端离线"}</span>
        </div>
      </div>

      <div className="settings-column">
        <Panel title="LM Studio 连接" kicker="本地推理通道" aside={<Tag tone={xray.lmReachable ? "positive" : "warn"}>{xray.lmReachable ? "可达" : "不可达"}</Tag>}>
          <div className="settings-grid">
            <Field label="服务地址" hint="只允许本机地址；可在后端环境变量中修改">
              <input value={xray.config?.lmstudio_url ?? "http://127.0.0.1:1234"} readOnly />
            </Field>
            <Field label="默认模型" hint="仅读取模型列表，不加载权重">
              <select value={xray.selectedModel} onChange={(event) => xray.setSelectedModel(event.target.value)}>
                {xray.models.length === 0 ? <option value="">未发现模型</option> : null}
                {xray.models.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.id}{model.state === "loaded" ? " · 已加载" : ""}{model.quantization ? ` · ${model.quantization}` : ""}
                  </option>
                ))}
              </select>
            </Field>
          </div>
          <div className="settings-actions">
            <Button variant="outline" size="sm" onClick={xray.refreshModels}>
              <Icon name="refresh" size={13} /> 刷新模型列表
            </Button>
            <span className="settings-hint">发送消息前不会加载模型。LM Studio 的模型加载进度由其自身窗口显示。</span>
          </div>
          <div className="settings-kv-grid">
            <KeyValue label="架构">{xray.models.find((item) => item.id === xray.selectedModel)?.arch ?? "—"}</KeyValue>
            <KeyValue label="量化">{xray.models.find((item) => item.id === xray.selectedModel)?.quantization ?? "—"}</KeyValue>
            <KeyValue label="最大上下文">
              <span className="num">{xray.models.find((item) => item.id === xray.selectedModel)?.max_context_length?.toLocaleString("zh-CN") ?? "—"}</span>
            </KeyValue>
            <KeyValue label="推理通道"><span className="mono">/v1/responses → chat → native</span></KeyValue>
          </div>
        </Panel>

        <Panel
          title="推理引擎"
          kicker="当前生效的运行时"
          aside={<Tag tone={xray.mode === "native" ? "accent" : "neutral"}>{xray.mode === "native" ? "原生精确" : xray.mode === "demo" ? "演示" : "LM Studio"}</Tag>}
        >
          <Segmented
            ariaLabel="选择推理引擎"
            value={xray.mode}
            onChange={(mode) => xray.setMode(mode)}
            options={[
              { value: "lmstudio", label: "LM Studio", title: "速度快；只保证 Token 与采样参数，内部张量不可得" },
              { value: "native", label: "原生精确", title: "在进程内运行本地 HF 模型；全部张量来自同一次前向，CPU 较慢" },
              { value: "demo", label: "演示", title: "不加载任何模型；用于预览界面" },
            ]}
          />
          <p className="panel-copy">
            LM Studio 的 HTTP API 不返回隐藏状态与注意力；部分版本在实际流式响应中也不返回 top_logprobs。
            原生精确模式把模型放进 LLM-Xray 进程内运行，生成与观测共用同一次前向，概率云、注意力、Tuned Lens、
            残差和向量云都来自同一模型。代价是 CPU 推理明显更慢，且模型必须能装进本机内存。
          </p>
        </Panel>

        <Panel
          title="原生精确模型"
          kicker="in-process · local_files_only"
          aside={
            <Tag tone={native.status === "ready" ? "positive" : native.status === "error" ? "warn" : "muted"}>
              {native.status}
            </Tag>
          }
        >
          <div className="settings-grid">
            <Field label="本地 HF 模型目录或模型 ID" hint="默认解析 models/lens 下的 Qwen2.5-0.5B；不会自动下载">
              <input
                value={xray.nativeModel}
                onChange={(event) => xray.setNativeModel(event.target.value)}
                placeholder="Qwen/Qwen2.5-0.5B-Instruct 或 D:\models\Qwen2.5-0.5B-Instruct"
              />
            </Field>
            <Field label="设备" hint="CUDA 需要 CUDA 版 PyTorch；Auto 优先使用 GPU，失败回退 CPU">
              <Segmented
                ariaLabel="原生模型设备"
                value={xray.nativeDevice}
                onChange={xray.setNativeDevice}
                options={[
                  { value: "auto", label: "Auto" },
                  { value: "cuda", label: "CUDA" },
                  { value: "cpu", label: "CPU" },
                ]}
              />
            </Field>
            <div className="settings-status-block">
              <KeyValue label="设备">{native.device ?? "cpu"}</KeyValue>
              <KeyValue label="权重关系">{native.status === "ready" ? "生成与观测同一权重" : "—"}</KeyValue>
              <KeyValue label="张量">{native.status === "ready" ? "精确（非替代）" : "—"}</KeyValue>
              <KeyValue label="线程"><span className="num">{native.threads ?? 2}</span></KeyValue>
            </div>
          </div>
          <div className="settings-actions">
            <Button variant="primary" size="sm" onClick={loadNative} disabled={nativeBusy || native.status === "loading"}>
              <Icon name="plug" size={13} /> {native.status === "loading" ? "加载中" : "加载原生模型"}
            </Button>
            <Button variant="outline" size="sm" onClick={unloadNative} disabled={nativeBusy || native.status !== "ready"}>
              卸载并释放
            </Button>
            <span className="settings-hint">
              0.5B CPU 首次加载约 20-30 秒；8B FP32 不适合 16GB 内存，需要量化 GPU 环境。
            </span>
          </div>
          {native.message ? <p className="inline-note">{native.message}</p> : null}
          {nativeMessage ? <p className="inline-note">{nativeMessage}</p> : null}
        </Panel>

        <Panel
          title="替代透镜"
          kicker="内部张量通道"
          aside={<Tag tone={lens.status === "ready" ? "positive" : lens.status === "error" ? "warn" : "muted"}>{lens.status}</Tag>}
        >
          <p className="panel-copy">
            LM Studio 不公开隐藏状态与注意力。透镜通道在本地加载一个小模型，以 CPU 两线程、低优先级观测同一段文本；
            它不会阻塞 LM Studio 的 Token 流，也不会自动下载权重。界面会把结果标为“替代透镜”。
          </p>
          <div className="settings-grid">
            <Field label="本地模型 ID 或路径" hint="必须已存在本地缓存或绝对路径；后端以 local_files_only 加载">
              <input value={lensModel} onChange={(event) => setLensModel(event.target.value)} placeholder="Qwen/Qwen2.5-0.5B-Instruct" />
            </Field>
            <div className="settings-status-block">
              <KeyValue label="状态">{lens.status}</KeyValue>
              <KeyValue label="设备">{lens.device ?? "cpu"}</KeyValue>
              <KeyValue label="已校准层"><span className="num">{lens.calibrated_layers}</span></KeyValue>
              <KeyValue label="权重关系">{lens.same_weights ? "与主模型相同" : "替代模型"}</KeyValue>
            </div>
          </div>
          <div className="settings-actions">
            <Button variant="primary" size="sm" onClick={loadLens} disabled={busy || lens.status === "loading"}>
              <Icon name="plug" size={13} /> {lens.status === "loading" ? "加载中" : "加载透镜"}
            </Button>
            <Button variant="outline" size="sm" onClick={unloadLens} disabled={busy || lens.status !== "ready"}>
              卸载并释放
            </Button>
            {lens.message ? <span className="settings-hint">{lens.message}</span> : null}
          </div>
          {message ? <p className="inline-note">{message}</p> : null}
        </Panel>

        <Panel title="采样默认值" kicker="下一次请求生效" aside={<span className="panel-note">实时重算仅改变显示</span>}>
          <div className="settings-sliders">
            <SliderRow
              label="温度"
              value={xray.sampling.temperature}
              min={0.05}
              max={2}
              step={0.05}
              format={(value) => value.toFixed(2)}
              onChange={(temperature) => xray.setSampling((current) => ({ ...current, temperature }))}
            />
            <SliderRow
              label="Top-P"
              value={xray.sampling.top_p}
              min={0.05}
              max={1}
              step={0.01}
              format={(value) => value.toFixed(2)}
              onChange={(top_p) => xray.setSampling((current) => ({ ...current, top_p }))}
            />
            <SliderRow
              label="Top-K"
              value={xray.sampling.top_k}
              min={1}
              max={100}
              step={1}
              format={(value) => String(Math.round(value))}
              onChange={(top_k) => xray.setSampling((current) => ({ ...current, top_k }))}
            />
            <SliderRow
              label="重复惩罚"
              value={xray.sampling.repetition_penalty}
              min={1}
              max={1.5}
              step={0.01}
              format={(value) => value.toFixed(2)}
              onChange={(repetition_penalty) => xray.setSampling((current) => ({ ...current, repetition_penalty }))}
            />
            <SliderRow
              label="最大输出"
              value={xray.sampling.max_tokens}
              min={256}
              max={maxOutputLimit}
              step={256}
              format={(value) => `${Math.round(value)} tokens`}
              onChange={(max_tokens) => xray.setSampling((current) => ({ ...current, max_tokens }))}
              hint="上限取当前模型的 max_context_length；仍受已加载上下文限制"
            />
            <div className="memory-estimate" title={memoryEstimate.detail}>
              <span>Estimated Memory Usage</span>
              <strong className="num">~{memoryEstimate.totalGb.toFixed(1)} GB</strong>
              <small>{memoryEstimate.detail}</small>
            </div>
          </div>
          <Field label="系统提示" hint="会随下一次请求发送；不会写入日志">
            <textarea
              className="settings-textarea"
              rows={3}
              value={xray.systemPrompt}
              onChange={(event) => xray.setSystemPrompt(event.target.value)}
              placeholder="例如：回答尽量简洁，并给出可验证的推理。"
            />
          </Field>
        </Panel>

        <Panel title="观测与界面" kicker="本地偏好">
          <div className="settings-toggles">
            <Toggle checked={xray.crt} onChange={xray.setCrt} label="CRT 覆膜" description="扫描线噪点，关闭时完全不会消耗渲染资源" />
            <Toggle checked={xray.sound} onChange={xray.setSound} label="熵声化" description="低熵清脆高音，高熵低频噪声；需要一次用户手势" />
            <Toggle checked={xray.ragEnabled} onChange={xray.setRagEnabled} label="生成时检索记忆" description="召回展示，不默认注入提示" />
            <Toggle checked={xray.lensEnabled} onChange={xray.setLensEnabled} label="允许透镜采样" description="仅当透镜模型就绪时生效" disabled={lens.status !== "ready"} />
          </div>
          <div className="settings-actions">
            <Button variant="outline" size="sm" onClick={() => xray.setChatWidth(390)}>重置对话窗宽度</Button>
            <Button variant="outline" size="sm" onClick={() => xray.setChatCollapsed(true)}>收起对话窗</Button>
          </div>
        </Panel>

        <Panel title="本地记忆" kicker="memory/*.md" aside={<IconButton label="重建索引" onClick={reindex} disabled={busy}><Icon name="refresh" size={14} /></IconButton>}>
          <div className="settings-kv-grid">
            <KeyValue label="文档数"><span className="num">{docs.length}</span></KeyValue>
            <KeyValue label="片段数"><span className="num">{chunks ?? "—"}</span></KeyValue>
            <KeyValue label="检索"><span className="mono">BM25 + cosine</span></KeyValue>
            <KeyValue label="注入"><span>{xray.ragInject ? "开启" : "关闭"}</span></KeyValue>
          </div>
          <p className="panel-copy faint">
            记忆库是普通 Markdown，可直接编辑。也可以把研究笔记放入 memory/，下一次检索会自动切块。
          </p>
        </Panel>

        <Panel title="关于" kicker="LLM-Xray">
          <div className="about-grid">
            <div>
              <KeyValue label="版本"><span className="num">{xray.config?.version ?? "1.0.0"}</span></KeyValue>
              <KeyValue label="桌面外壳">pywebview + WebView2</KeyValue>
              <KeyValue label="前端">React 19 + Three.js</KeyValue>
              <KeyValue label="后端">FastAPI + httpx</KeyValue>
            </div>
            <div>
              <KeyValue label="数据目录"><span className="mono">data/sessions</span></KeyValue>
              <KeyValue label="文档"><span className="mono">docs/</span></KeyValue>
              <KeyValue label="网络"><span>仅本机回环</span></KeyValue>
              <KeyValue label="模型调用"><span>只在点击发送时</span></KeyValue>
            </div>
          </div>
          <p className="panel-copy faint">
            概率、token 与速度来自 LM Studio。注意力、Tuned Lens、残差、稀疏特征与 UMAP 来自替代透镜，两者在界面中始终分开标注。
          </p>
        </Panel>
      </div>
    </div>
  );
}
