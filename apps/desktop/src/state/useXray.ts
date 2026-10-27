import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type StartPayload, XraySocket } from "../lib/api";
import { desktopSaveApi } from "../lib/desktop";
import {
  DEFAULT_SAMPLING,
  type AppConfig,
  type ChatMessage,
  type KVInfo,
  type LensStatus,
  type Metrics,
  type ModelInfo,
  type NativeStatus,
  type ProjectionPoint,
  type Probe,
  type ReplayStep,
  type SamplingParams,
  type ServerEvent,
  type SessionMeta,
  type SessionSummary,
  type StepToken,
  type Usage,
  type RuntimeMode,
} from "../types";

export type Connection = "connecting" | "open" | "closed" | "error";
export type TabId = "observatory" | "attention" | "vectors" | "memory" | "replay" | "settings" | "docs";

export interface ChatEntry {
  id: string;
  role: "user" | "assistant";
  content: string;
  reasoning: string;
  streaming: boolean;
  kind?: "demo" | "lmstudio" | "native";
}

export interface MetricPoint {
  t: number;
  tps: number | null;
  gpu: number | null;
  entropy: number | null;
}

const EMPTY_LENS: LensStatus = {
  status: "unloaded",
  model: null,
  device: null,
  surrogate: true,
  same_weights: false,
  calibrated_layers: 0,
};

const EMPTY_NATIVE: NativeStatus = {
  available: false,
  status: "unloaded",
  model: "Qwen/Qwen2.5-0.5B-Instruct",
  device: null,
  surrogate: false,
  exact: false,
  same_weights: false,
  generation: false,
  calibrated_layers: 0,
  message: "原生精确通道未加载",
};

function makeId() {
  return `s-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
}

function readPref<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function loadInitialSampling(): SamplingParams {
  const stored = readPref<Partial<SamplingParams> | null>("xray.sampling", null);
  const merged: SamplingParams = { ...DEFAULT_SAMPLING, ...(stored ?? {}) };
  // v1.0 persisted the old hard cap of 512 tokens and had no UI to change it.
  // Migrate that legacy value once; after this marker the user's choice wins.
  try {
    if (localStorage.getItem("xray.maxTokensMigrated") !== "1") {
      if (!stored?.max_tokens || stored.max_tokens <= 512) {
        merged.max_tokens = DEFAULT_SAMPLING.max_tokens;
      }
      localStorage.setItem("xray.maxTokensMigrated", "1");
    }
  } catch {
    // localStorage can be unavailable in hardened webviews; defaults still apply.
  }
  return merged;
}

export function useXray() {
  const [connection, setConnection] = useState<Connection>("connecting");
  const [connectionDetail, setConnectionDetail] = useState<string>("");
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [lmReachable, setLmReachable] = useState<boolean | null>(null);
  const [lens, setLens] = useState<LensStatus>(EMPTY_LENS);
  const [native, setNative] = useState<NativeStatus>(EMPTY_NATIVE);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [nativeModel, setNativeModel] = useState<string>(
    () => localStorage.getItem("xray.nativeModel") ?? "Qwen/Qwen2.5-0.5B-Instruct",
  );
  const [nativeDevice, setNativeDevice] = useState<string>(() => localStorage.getItem("xray.nativeDevice") ?? "auto");
  const [mode, setMode] = useState<RuntimeMode>(() => {
    const stored = localStorage.getItem("xray.runtime");
    return stored === "native" || stored === "demo" ? stored : "lmstudio";
  });
  const [sampling, setSampling] = useState<SamplingParams>(loadInitialSampling);
  const [systemPrompt, setSystemPrompt] = useState<string>(
    () => localStorage.getItem("xray.system") ?? "",
  );
  const [ragEnabled, setRagEnabled] = useState<boolean>(() => readPref("xray.rag", true));
  const [ragInject, setRagInject] = useState<boolean>(() => readPref("xray.ragInject", false));
  const [lensEnabled, setLensEnabled] = useState<boolean>(() => readPref("xray.lens", true));

  const [chat, setChat] = useState<ChatEntry[]>([]);
  const [steps, setSteps] = useState<StepToken[]>([]);
  const [probes, setProbes] = useState<Map<number, Probe>>(new Map());
  const [projection, setProjection] = useState<ProjectionPoint[]>([]);
  const [projectionReducer, setProjectionReducer] = useState<string>("UMAP");
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [metricHistory, setMetricHistory] = useState<MetricPoint[]>([]);
  const [kv, setKv] = useState<KVInfo | null>(null);
  const [usage, setUsage] = useState<Usage | null>(null);
  const [finishReason, setFinishReason] = useState<string>("");
  const [elapsed, setElapsed] = useState<number>(0);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [session, setSession] = useState<SessionMeta | null>(null);
  const [generating, setGenerating] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [exporting, setExporting] = useState(false);
  const [exportNotice, setExportNotice] = useState<string>("");
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [replayMode, setReplayMode] = useState(false);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [selectedTab, setSelectedTab] = useState<TabId>("observatory");
  const [chatWidth, setChatWidth] = useState<number>(() => readPref("xray.chatWidth", 390));
  const [chatCollapsed, setChatCollapsed] = useState(false);
  const [crt, setCrt] = useState<boolean>(() => readPref("xray.crt", false));
  const [sound, setSound] = useState<boolean>(() => readPref("xray.sound", false));
  const [ragItems, setRagItems] = useState<import("../types").RagItem[]>([]);
  const [ragQuery, setRagQuery] = useState("");
  const socketRef = useRef<XraySocket | null>(null);
  const stepRef = useRef<StepToken[]>([]);
  const chatRef = useRef<ChatEntry[]>([]);
  const sessionRef = useRef<SessionMeta | null>(null);

  useEffect(() => {
    stepRef.current = steps;
  }, [steps]);
  useEffect(() => {
    chatRef.current = chat;
  }, [chat]);
  useEffect(() => {
    sessionRef.current = session;
  }, [session]);
  useEffect(() => {
    localStorage.setItem("xray.sampling", JSON.stringify(sampling));
  }, [sampling]);
  useEffect(() => {
    localStorage.setItem("xray.chatWidth", JSON.stringify(chatWidth));
  }, [chatWidth]);
  useEffect(() => {
    localStorage.setItem("xray.crt", JSON.stringify(crt));
    document.body.classList.toggle("crt-on", crt);
  }, [crt]);
  useEffect(() => {
    localStorage.setItem("xray.sound", JSON.stringify(sound));
  }, [sound]);
  useEffect(() => {
    localStorage.setItem("xray.rag", JSON.stringify(ragEnabled));
    localStorage.setItem("xray.ragInject", JSON.stringify(ragInject));
  }, [ragEnabled, ragInject]);
  useEffect(() => {
    localStorage.setItem("xray.lens", JSON.stringify(lensEnabled));
  }, [lensEnabled]);
  useEffect(() => {
    localStorage.setItem("xray.nativeModel", nativeModel);
  }, [nativeModel]);
  useEffect(() => {
    localStorage.setItem("xray.nativeDevice", nativeDevice);
  }, [nativeDevice]);
  useEffect(() => {
    if (mode !== "demo") localStorage.setItem("xray.runtime", mode);
  }, [mode]);
  useEffect(() => {
    localStorage.setItem("xray.system", systemPrompt);
  }, [systemPrompt]);

  const applyHello = useCallback((event: Extract<ServerEvent, { type: "hello" }>) => {
    setConfig(event.config);
    setModels(event.lmstudio.models ?? []);
    setLmReachable(event.lmstudio.reachable);
    setLens(event.lens ?? event.config?.lens ?? EMPTY_LENS);
    setNative(event.native ?? event.config?.native ?? EMPTY_NATIVE);
    setSelectedModel((current) => {
      if (current) return current;
      const loaded = (event.lmstudio.models ?? []).find((m) => m.state === "loaded");
      return loaded?.id ?? event.config?.default_model ?? event.lmstudio.models?.[0]?.id ?? "";
    });
    if (event.config?.sampling_defaults) {
      setSampling((current) => ({ ...event.config.sampling_defaults, ...current }));
    }
  }, []);

  const handleEvent = useCallback((event: ServerEvent) => {
    switch (event.type) {
      case "hello":
        applyHello(event);
        break;
      case "started":
        setSession(event.session);
        sessionRef.current = event.session;
        setSteps([]);
        stepRef.current = [];
        setProbes(new Map());
        setProjection([]);
        setUsage(null);
        setFinishReason("");
        setElapsed(0);
        setStartedAt(event.session.started_at);
        setGenerating(true);
        setSelectedStep(null);
        setReplayMode(false);
        setStatusMessage("");
        break;
      case "rag":
        setRagItems(event.items);
        break;
      case "token": {
        const step = event.step;
        setSteps((previous) => {
          const next = [...previous, step];
          stepRef.current = next;
          return next;
        });
        setChat((entries) => {
          const updated = [...entries];
          for (let i = updated.length - 1; i >= 0; i -= 1) {
            if (updated[i].role === "assistant" && updated[i].streaming) {
              updated[i] = {
                ...updated[i],
                content: updated[i].content + (step.kind === "answer" ? step.delta : ""),
                reasoning: updated[i].reasoning + (step.kind === "reasoning" ? step.delta : ""),
              };
              break;
            }
          }
          chatRef.current = updated;
          return updated;
        });
        break;
      }
      case "probe":
        setProbes((previous) => {
          const next = new Map(previous);
          next.set(event.step, event.probe);
          return next;
        });
        break;
      case "projection":
        setProjection(event.projection.points);
        setProjectionReducer(event.projection.reducer);
        break;
      case "metrics":
        setMetrics(event.metrics);
        setMetricHistory((history) => {
          const point: MetricPoint = {
            t: Date.now(),
            tps: event.metrics.tps,
            gpu: event.metrics.gpu,
            entropy: stepRef.current.at(-1)?.entropy ?? null,
          };
          return [...history.slice(-119), point];
        });
        break;
      case "kv":
        setKv(event.kv);
        break;
      case "status":
        setStatusMessage(event.message);
        if (event.scope === "lens") {
          setLens((current) => ({
            ...current,
            status: event.state === "ready" ? "ready" : event.state === "error" ? "error" : "loading",
            message: event.message,
          }));
        }
        if (event.scope === "native") {
          setNative((current) => ({
            ...current,
            status:
              event.state === "error"
                ? "error"
                : event.state === "loading"
                  ? "loading"
                  : "ready",
            exact: event.state !== "error",
            same_weights: event.state !== "error",
            message: event.message,
          }));
        }
        if (event.scope === "runtime" && event.state === "no_logprobs") {
          setSession((current) =>
            current
              ? { ...current, capabilities: { ...current.capabilities, logprobs: false } }
              : current,
          );
        }
        break;
      case "done":
        setUsage(event.usage);
        setFinishReason(event.finish_reason);
        setElapsed(event.elapsed_s);
        setGenerating(false);
        if (event.capabilities?.logprobs === false) {
          setSession((current) =>
            current
              ? { ...current, capabilities: { ...current.capabilities, logprobs: false } }
              : current,
          );
        }
        setChat((entries) =>
          entries.map((entry) => (entry.streaming ? { ...entry, streaming: false } : entry)),
        );
        void api.sessions().then((res) => setSessions(res.sessions)).catch(() => undefined);
        break;
      case "error":
        setError(event.message);
        setGenerating(false);
        setChat((entries) =>
          entries.map((entry) => (entry.streaming ? { ...entry, streaming: false } : entry)),
        );
        break;
      case "pong":
        break;
      default:
        break;
    }
  }, [applyHello]);

  useEffect(() => {
    const socket = new XraySocket({
      onEvent: (event) => {
        if (event.type === "hello") {
          setConnection("open");
          setConnectionDetail("");
        }
        handleEvent(event);
      },
      onStatus: (status, detail) => {
        setConnection(status);
        if (detail) setConnectionDetail(detail);
        if (status === "closed" || status === "error") {
          setGenerating(false);
          setChat((entries) => entries.map((entry) => (entry.streaming ? { ...entry, streaming: false } : entry)));
        }
      },
    });
    socketRef.current = socket;
    socket.connect();
    return () => {
      socket.dispose();
      socketRef.current = null;
    };
  }, [handleEvent]);

  useEffect(() => {
    if (lens.status !== "loading") return;
    let cancelled = false;
    const timer = window.setInterval(() => {
      void api
        .lensStatus()
        .then((status) => {
          if (cancelled) return;
          setLens(status);
          if (status.status !== "loading") {
            window.clearInterval(timer);
            setStatusMessage(status.status === "ready" ? "透镜已就绪" : status.message ?? "");
          }
        })
        .catch(() => undefined);
    }, 1200);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [lens.status]);

  useEffect(() => {
    if (native.status !== "loading") return;
    let cancelled = false;
    const timer = window.setInterval(() => {
      void api
        .nativeStatus()
        .then((status) => {
          if (cancelled) return;
          setNative(status);
          if (status.status !== "loading") {
            window.clearInterval(timer);
            setStatusMessage(status.status === "ready" ? "原生精确模型已就绪" : status.message ?? "");
          }
        })
        .catch(() => undefined);
    }, 1500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [native.status]);

  useEffect(() => {
    void api
      .sessions()
      .then((res) => setSessions(res.sessions))
      .catch(() => undefined);
    void api
      .health()
      .then((health) => {
        setLmReachable(health.lmstudio.reachable);
        if (health.lens) setLens(health.lens);
        if (health.native) setNative(health.native);
      })
      .catch(() => undefined);
  }, []);

  const refreshModels = useCallback(async () => {
    try {
      const res = await api.models();
      setModels(res.data ?? []);
      setLmReachable(res.lmstudio_reachable ?? true);
      setError("");
    } catch (err) {
      setLmReachable(false);
      setError(err instanceof Error ? err.message : "无法读取 LM Studio 模型列表");
    }
  }, []);

  const send = useCallback(
    (text: string, options?: { mode?: RuntimeMode; model?: string }) => {
      const content = text.trim();
      if (!content || generating) return;
      const nextMode = options?.mode ?? mode;
      const userEntry: ChatEntry = {
        id: makeId(),
        role: "user",
        content,
        reasoning: "",
        streaming: false,
      };
      const assistantEntry: ChatEntry = {
        id: makeId(),
        role: "assistant",
        content: "",
        reasoning: "",
        streaming: true,
        kind: nextMode,
      };
      const history = chatRef.current.filter((entry) => entry.content.trim().length > 0);
      const nextChat = [...history, userEntry, assistantEntry];
      setChat(nextChat);
      chatRef.current = nextChat;
      setError("");
      setGenerating(true);
      setSteps([]);
      stepRef.current = [];
      setProbes(new Map());
      setProjection([]);
      setProjectionReducer("UMAP");
      setMetricHistory([]);
      setSelectedStep(null);
      setReplayMode(false);
      setSelectedTab("observatory");
      setMode(nextMode);
      const messages: ChatMessage[] = history.map((entry) => ({
        role: entry.role,
        content: entry.content,
      }));
      messages.push({ role: "user", content });
      const payload: StartPayload = {
        messages,
        model:
          nextMode === "demo"
            ? "demo"
            : nextMode === "native"
              ? nativeModel.trim() || undefined
              : options?.model ?? (selectedModel || undefined),
        mode: nextMode,
        sampling,
        rag: { enabled: ragEnabled, top_k: 4, inject: ragInject },
        lens: { enabled: nextMode === "lmstudio" && lensEnabled && lens.status === "ready" },
        system: systemPrompt.trim() || undefined,
      };
      socketRef.current?.start(payload);
    },
    [generating, lens.status, lensEnabled, mode, nativeModel, ragEnabled, ragInject, selectedModel, sampling, systemPrompt],
  );

  const stop = useCallback(() => {
    if (sessionRef.current) void api.stop(sessionRef.current.id).catch(() => undefined);
    socketRef.current?.stop();
    setGenerating(false);
    setChat((entries) => entries.map((entry) => (entry.streaming ? { ...entry, streaming: false } : entry)));
  }, []);

  const resetConversation = useCallback(() => {
    if (generating) stop();
    setChat([]);
    setSteps([]);
    setProbes(new Map());
    setProjection([]);
    setUsage(null);
    setMetrics(null);
    setKv(null);
    setSelectedStep(null);
    setReplayMode(false);
    setSession(null);
    setError("");
    setFinishReason("");
  }, [generating, stop]);

  const loadLens = useCallback(async (model?: string | null) => {
    try {
      setLens((current) => ({ ...current, status: "loading", message: "正在加载透镜模型" }));
      const result = await api.loadLens({ model: model ?? null, device: "cpu" });
      setLens(result.lens);
      setStatusMessage(result.message ?? "透镜已就绪");
      return result;
    } catch (err) {
      setLens((current) => ({
        ...current,
        status: "error",
        message: err instanceof Error ? err.message : "透镜加载失败",
      }));
      setError(err instanceof Error ? err.message : "透镜加载失败");
      throw err;
    }
  }, []);

  const unloadLens = useCallback(async () => {
    try {
      const result = await api.unloadLens();
      setLens(result.lens);
    } catch (err) {
      setError(err instanceof Error ? err.message : "透镜卸载失败");
    }
  }, []);

  const loadNative = useCallback(async (model?: string | null) => {
    try {
      setNative((current) => ({ ...current, status: "loading", message: "正在加载原生模型" }));
      setStatusMessage("正在加载原生精确模型（CPU，可能需要 20-40 秒）");
      const result = await api.loadNative({ model: model ?? nativeModel.trim() ?? null, device: nativeDevice });
      setNative(result);
      if (result.model) setNativeModel(result.model);
      setStatusMessage(result.message ?? "原生精确模型已就绪");
      return result;
    } catch (err) {
      setNative((current) => ({
        ...current,
        status: "error",
        message: err instanceof Error ? err.message : "原生模型加载失败",
      }));
      setError(err instanceof Error ? err.message : "原生模型加载失败");
      throw err;
    }
  }, [nativeModel, nativeDevice]);

  const unloadNative = useCallback(async () => {
    try {
      const result = await api.unloadNative();
      setNative(result);
      setStatusMessage("原生精确模型已卸载");
    } catch (err) {
      setError(err instanceof Error ? err.message : "原生模型卸载失败");
    }
  }, []);

  const searchMemory = useCallback(async (query: string) => {
    const q = query.trim();
    if (!q) return;
    setRagQuery(q);
    try {
      const res = await api.ragSearch(q);
      setRagItems(res.items);
    } catch (err) {
      setError(err instanceof Error ? err.message : "记忆检索失败");
    }
  }, []);

  const loadSavedSession = useCallback(async (id: string) => {
    try {
      setStatusMessage("正在读取历史会话");
      const [detail, res] = await Promise.all([api.session(id), api.steps(id)]);
      const mapped: StepToken[] = res.steps.map((step: ReplayStep) => ({
        index: step.index,
        delta: step.delta,
        kind: step.kind,
        topk: step.topk ?? [],
        chosen_logprob: null,
        chosen_prob: step.chosen_prob,
        entropy: step.entropy,
        at: step.at,
        ttft_ms: null,
        tps: null,
        generated: step.index + 1,
        text_len: 0,
      }));
      const probeMap = new Map<number, Probe>();
      for (const step of res.steps) if (step.probe) probeMap.set(step.index, step.probe);
      const meta = detail.meta as SessionMeta | undefined;
      if (meta) {
        setSession(meta);
        sessionRef.current = meta;
        setStartedAt(meta.started_at);
      }
      setSteps(mapped);
      stepRef.current = mapped;
      setProbes(probeMap);
      if (mapped.length) {
        const lastIndex = mapped[mapped.length - 1].index;
        if (!probeMap.has(lastIndex)) {
          try {
            const full = await api.step(id, lastIndex);
            if (full.step?.probe) {
              probeMap.set(lastIndex, full.step.probe);
              setProbes(new Map(probeMap));
            }
          } catch {
            // A session can be truncated; the timeline still works without a probe.
          }
        }
      }
      const messages = detail.messages as ChatMessage[] | undefined;
      if (Array.isArray(messages)) {
        const restored: ChatEntry[] = messages
          .filter((message) => message.role !== "system" && message.content)
          .map((message) => ({
            id: makeId(),
            role: message.role === "assistant" ? ("assistant" as const) : ("user" as const),
            content: message.content,
            reasoning: "",
            streaming: false,
          }));
        const answerText = mapped
          .filter((step) => step.kind === "answer")
          .map((step) => step.delta)
          .join("");
        const reasoningText = mapped
          .filter((step) => step.kind === "reasoning")
          .map((step) => step.delta)
          .join("");
        if (answerText && restored.at(-1)?.role !== "assistant") {
          restored.push({
            id: makeId(),
            role: "assistant",
            content: answerText,
            reasoning: reasoningText,
            streaming: false,
            kind: meta?.mode === "demo" ? "demo" : "lmstudio",
          });
        }
        setChat(restored);
      }
      setSelectedStep(mapped.length ? mapped[mapped.length - 1].index : null);
      setReplayMode(true);
      setGenerating(false);
      setSelectedTab("replay");
      setStatusMessage("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "历史会话读取失败");
      setStatusMessage("");
    }
  }, []);

  const selectStep = useCallback(
    async (index: number | null) => {
      setSelectedStep(index);
      if (index === null) {
        setReplayMode(false);
        return;
      }
      setReplayMode(true);
      if (!probes.has(index) && sessionRef.current) {
        try {
          const res = await api.step(sessionRef.current.id, index);
          const probe = res.step?.probe;
          if (probe) {
            setProbes((previous) => {
              const next = new Map(previous);
              next.set(index, probe);
              return next;
            });
          }
        } catch {
          // Historical detail is best-effort; the scan remains usable.
        }
      }
    },
    [probes],
  );

  const activeStepIndex = selectedStep ?? (steps.length ? steps[steps.length - 1].index : null);
  const activeStep = useMemo(
    () => (activeStepIndex === null ? null : steps.find((step) => step.index === activeStepIndex) ?? steps.at(-1) ?? null),
    [activeStepIndex, steps],
  );
  const activeProbe = useMemo(() => {
    if (activeStepIndex === null) return null;
    if (probes.has(activeStepIndex)) return probes.get(activeStepIndex) ?? null;
    let best: Probe | null = null;
    for (const [index, probe] of probes) if (index <= activeStepIndex && (!best || index > best.step)) best = probe;
    return best;
  }, [activeStepIndex, probes]);
  const latestProbe = useMemo(() => {
    if (!probes.size) return null;
    let best: Probe | null = null;
    for (const probe of probes.values()) if (!best || probe.step > best.step) best = probe;
    return best;
  }, [probes]);

  const contextTokens = useMemo(() => {
    if (activeProbe) return activeProbe.context.tokens;
    return steps.slice(0, 40).map((step) => step.delta);
  }, [activeProbe, steps]);

  const recentTokens = useMemo(() => {
    if (activeStepIndex === null) return [];
    return steps.filter((step) => step.index <= activeStepIndex).map((step) => step.delta);
  }, [activeStepIndex, steps]);

  const sessionId = session?.id ?? null;

  const exportSession = useCallback(async () => {
    if (!sessionId || exporting) return;
    setExporting(true);
    setExportNotice("");
    const filename = `llm-xray-${sessionId}.jsonl`;
    try {
      const response = await fetch(api.exportUrl(sessionId));
      if (!response.ok) throw new Error(`导出失败：HTTP ${response.status}`);
      const text = await response.text();
      const native = desktopSaveApi();
      if (native?.save_text_file) {
        const path = await native.save_text_file(filename, text);
        setExportNotice(path ? `JSONL 已导出：${path}` : "已取消导出");
      } else {
        const blob = new Blob([text], { type: "application/x-ndjson;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = filename;
        anchor.style.display = "none";
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        window.setTimeout(() => URL.revokeObjectURL(url), 4000);
        setExportNotice(`JSONL 已下载：${filename}`);
      }
      window.setTimeout(() => setExportNotice(""), 7000);
    } catch (err) {
      setExportNotice(err instanceof Error ? err.message : "导出失败");
    } finally {
      setExporting(false);
    }
  }, [exporting, sessionId]);

  return {
    connection,
    connectionDetail,
    config,
    models,
    lmReachable,
    lens,
    native,
    nativeModel,
    setNativeModel,
    nativeDevice,
    setNativeDevice,
    selectedModel,
    setSelectedModel,
    mode,
    setMode,
    sampling,
    setSampling,
    systemPrompt,
    setSystemPrompt,
    ragEnabled,
    setRagEnabled,
    ragInject,
    setRagInject,
    lensEnabled,
    setLensEnabled,
    chat,
    steps,
    probes,
    projection,
    projectionReducer,
    metrics,
    metricHistory,
    kv,
    usage,
    finishReason,
    elapsed,
    startedAt,
    session,
    sessionId,
    generating,
    statusMessage,
    error,
    setError,
    exporting,
    exportNotice,
    setExportNotice,
    selectedStep,
    activeStepIndex,
    activeStep,
    activeProbe,
    latestProbe,
    contextTokens,
    recentTokens,
    replayMode,
    sessions,
    selectedTab,
    setSelectedTab,
    chatWidth,
    setChatWidth,
    chatCollapsed,
    setChatCollapsed,
    crt,
    setCrt,
    sound,
    setSound,
    ragItems,
    ragQuery,
    send,
    stop,
    resetConversation,
    refreshModels,
    loadLens,
    unloadLens,
    loadNative,
    unloadNative,
    searchMemory,
    loadSavedSession,
    selectStep,
    exportSession,
  };
}

export type Xray = ReturnType<typeof useXray>;
