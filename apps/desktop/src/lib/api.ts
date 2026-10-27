import type {
  AppConfig,
  ClientEvent,
  LensStatus,
  ModelInfo,
  RagItem,
  ReplayStep,
  SamplingParams,
  ServerEvent,
  SessionSummary,
} from "../types";

const base = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${base}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}${text ? `: ${text.slice(0, 240)}` : ""}`);
  }
  return (await res.json()) as T;
}

export interface Health {
  ok: boolean;
  version: string;
  uptime_s: number;
  lmstudio: { reachable: boolean; models?: number; error?: string | null };
  lens: LensStatus;
  native?: import("../types").NativeStatus;
  gpu?: string | null;
}

export const api = {
  health: () => json<Health>("/api/health"),
  config: () => json<AppConfig>("/api/config"),
  models: async () => {
    const raw = await json<{
      data?: ModelInfo[];
      models?: ModelInfo[];
      reachable?: boolean;
      lmstudio_reachable?: boolean;
      error?: string;
    }>("/api/models");
    return {
      data: raw.data ?? raw.models ?? [],
      lmstudio_reachable: raw.reachable ?? raw.lmstudio_reachable ?? true,
      error: raw.error,
    };
  },
  lensStatus: () => json<LensStatus>("/api/lens/status"),
  nativeStatus: () => json<import("../types").NativeStatus>("/api/native/status"),
  loadNative: async (body: { model?: string | null; device?: string; load_in_4bit?: boolean } = {}) => {
    const raw = await json<import("../types").NativeStatus & { native?: import("../types").NativeStatus }>(
      "/api/native/load",
      { method: "POST", body: JSON.stringify(body) },
    );
    return raw.native ?? raw;
  },
  unloadNative: async () => {
    const raw = await json<import("../types").NativeStatus & { native?: import("../types").NativeStatus }>(
      "/api/native/unload",
      { method: "POST" },
    );
    return raw.native ?? raw;
  },
  loadLens: async (body: { model?: string | null; device?: string } = {}) => {
    const raw = await json<LensStatus & { lens?: LensStatus; message?: string; ok?: boolean }>("/api/lens/load", {
      method: "POST",
      body: JSON.stringify(body),
    });
    const lens = raw.lens ?? raw;
    return { ok: lens.status !== "disabled" && lens.status !== "error", lens, message: raw.message ?? lens.message };
  },
  unloadLens: async () => {
    const raw = await json<LensStatus & { lens?: LensStatus }>("/api/lens/unload", { method: "POST" });
    return { ok: raw.status !== "error", lens: raw.lens ?? raw };
  },
  sessions: async () => {
    const raw = await json<SessionSummary[] | { sessions: SessionSummary[] }>("/api/sessions");
    return { sessions: Array.isArray(raw) ? raw : raw.sessions ?? [] };
  },
  session: async (id: string) => {
    const raw = await json<Record<string, unknown> & { meta?: unknown; messages?: unknown; steps?: ReplayStep[] }>(
      `/api/sessions/${id}`,
    );
    return {
      meta: raw.meta ?? raw,
      steps: raw.steps ?? [],
      messages: raw.messages ?? [],
    };
  },
  steps: async (id: string, offset = 0, limit = 500) => {
    const raw = await json<ReplayStep[] | { steps: ReplayStep[]; total?: number }>(
      `/api/sessions/${id}/steps?offset=${offset}&limit=${limit}`,
    );
    return { steps: Array.isArray(raw) ? raw : raw.steps ?? [], total: Array.isArray(raw) ? raw.length : raw.total ?? 0 };
  },
  step: async (id: string, index: number) => {
    const raw = await json<ReplayStep & { step?: ReplayStep }>(`/api/sessions/${id}/steps/${index}`);
    return { step: raw.step ?? raw };
  },
  exportUrl: (id: string) => `${base}/api/sessions/${id}/export`,
  ragSearch: async (query: string, topK = 4) => {
    const raw = await json<{ items?: RagItem[]; query?: string }>("/api/rag/search", {
      method: "POST",
      body: JSON.stringify({ query, top_k: topK }),
    });
    return { items: raw.items ?? [], query: raw.query ?? query };
  },
  ragDocuments: async () => {
    const raw = await json<{
      documents?: { source: string; chunks: number; chars: number }[];
      stats?: { chunks?: number; backend?: string; embedding_model?: string | null };
      chunks?: number;
      embedding?: string;
    }>("/api/rag/documents");
    return {
      documents: raw.documents ?? [],
      chunks: raw.stats?.chunks ?? raw.chunks ?? 0,
      embedding: raw.stats?.embedding_model ?? raw.stats?.backend ?? raw.embedding ?? "BM25",
    };
  },
  reindexRag: async () => {
    const raw = await json<{ ok: boolean; stats?: { chunks?: number }; chunks?: number }>("/api/rag/reindex", {
      method: "POST",
    });
    return { ok: raw.ok, chunks: raw.stats?.chunks ?? raw.chunks ?? 0 };
  },
  doc: (name: string) => fetch(`${base}/api/docs/${name}`).then((r) => r.text()),
  stop: (id: string) => json<{ ok?: boolean; stopped?: boolean }>(`/api/sessions/${id}/stop`, { method: "POST" }),
};

export interface StartPayload {
  messages: { role: "system" | "user" | "assistant"; content: string }[];
  model?: string;
  mode?: "lmstudio" | "demo" | "native";
  sampling?: Partial<SamplingParams>;
  rag?: { enabled: boolean; top_k: number; inject: boolean };
  lens?: { enabled: boolean; model?: string | null };
  system?: string;
}

type SocketHandlers = {
  onEvent: (event: ServerEvent) => void;
  onStatus: (status: "connecting" | "open" | "closed" | "error", detail?: string) => void;
};

export class XraySocket {
  private ws: WebSocket | null = null;
  private timer: number | null = null;
  private closedByUser = false;
  private queue: ClientEvent[] = [];

  constructor(private handlers: SocketHandlers) {}

  connect() {
    this.closedByUser = false;
    this.handlers.onStatus("connecting");
    const scheme = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${scheme}//${location.host}/ws`;
    try {
      this.ws = new WebSocket(url);
    } catch {
      this.handlers.onStatus("error", "无法创建 WebSocket");
      this.scheduleReconnect();
      return;
    }
    this.ws.onopen = () => {
      this.handlers.onStatus("open");
      for (const item of this.queue.splice(0)) this.ws?.send(JSON.stringify(item));
    };
    this.ws.onmessage = (message) => {
      try {
        const data = JSON.parse(String(message.data)) as ServerEvent;
        this.handlers.onEvent(data);
      } catch {
        this.handlers.onStatus("error", "收到无法解析的事件");
      }
    };
    this.ws.onerror = () => this.handlers.onStatus("error", "后端连接失败");
    this.ws.onclose = () => {
      this.ws = null;
      if (!this.closedByUser) {
        this.handlers.onStatus("closed");
        this.scheduleReconnect();
      }
    };
  }

  private scheduleReconnect() {
    if (this.timer !== null || this.closedByUser) return;
    this.timer = window.setTimeout(() => {
      this.timer = null;
      if (!this.closedByUser) this.connect();
    }, 1800);
  }

  send(event: ClientEvent) {
    const text = JSON.stringify(event);
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(text);
    else this.queue.push(event);
  }

  start(payload: StartPayload) {
    this.send({ type: "start", payload });
  }

  stop() {
    this.send({ type: "stop" });
  }

  ping() {
    this.send({ type: "ping" });
  }

  dispose() {
    this.closedByUser = true;
    if (this.timer !== null) window.clearTimeout(this.timer);
    this.ws?.close();
    this.ws = null;
  }
}
