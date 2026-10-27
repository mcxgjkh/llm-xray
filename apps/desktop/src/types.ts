// Shared contract between the Python backend and the React shell.
// Source of truth: docs/SPEC.md. Keep changes backwards compatible within v1.

export type Role = "system" | "user" | "assistant";

export interface ChatMessage {
  role: Role;
  content: string;
}

export type RuntimeMode = "lmstudio" | "demo" | "native";
export type TokenKind = "answer" | "reasoning";

export interface SamplingParams {
  temperature: number;
  top_p: number;
  top_k: number;
  repetition_penalty: number;
  max_tokens: number;
  seed: number | null;
}

export const DEFAULT_SAMPLING: SamplingParams = {
  temperature: 0.7,
  top_p: 0.95,
  top_k: 40,
  repetition_penalty: 1.05,
  max_tokens: 4096,
  seed: null,
};

export interface TokenOption {
  token: string;
  id?: number | null;
  logprob: number;
  prob: number;
  rank: number;
}

export interface ModelInfo {
  id: string;
  object?: string;
  type?: string;
  publisher?: string;
  arch?: string;
  quantization?: string;
  state?: "loaded" | "not-loaded" | string;
  max_context_length?: number;
}

export interface LensStatus {
  status: "unloaded" | "loading" | "ready" | "error" | "disabled";
  model: string | null;
  device: string | null;
  surrogate: boolean;
  same_weights: boolean;
  calibrated_layers: number;
  message?: string;
  threads?: number;
}

export interface NativeStatus {
  available: boolean;
  status: "unloaded" | "loading" | "ready" | "error" | "disabled";
  model: string | null;
  device: string | null;
  surrogate: false;
  exact?: boolean;
  same_weights: boolean;
  generation: boolean;
  calibrated_layers: number;
  message?: string;
  threads?: number;
}

export interface AppConfig {
  version: string;
  lmstudio_url: string;
  default_model: string;
  default_mode: RuntimeMode;
  lens: LensStatus;
  native?: NativeStatus;
  capabilities: {
    logprobs: boolean;
    attention: boolean;
    lens: boolean;
    rag: boolean;
    gpu: boolean;
  };
  sampling_defaults: SamplingParams;
  demo_available: boolean;
}

export interface SessionMeta {
  id: string;
  mode: RuntimeMode;
  model: string;
  started_at: number;
  capabilities: {
    logprobs: boolean;
    attention: boolean;
    lens: boolean;
  };
  sampling: SamplingParams;
  rag_enabled: boolean;
  lens_enabled: boolean;
  exact?: boolean;
  native?: boolean;
}

export interface StepToken {
  index: number;
  delta: string;
  kind: TokenKind;
  topk: TokenOption[];
  chosen_logprob: number | null;
  chosen_prob: number | null;
  entropy: number | null;
  at: number;
  ttft_ms: number | null;
  tps: number | null;
  generated: number;
  text_len: number;
}

export interface AttentionLayer {
  layer: number;
  mean: number[];
  heads?: number[][];
  head_types?: HeadType[];
}

export type HeadType =
  | "previous-token"
  | "local"
  | "syntactic"
  | "entity"
  | "global"
  | "diffuse";

export interface Probe {
  step: number;
  model: string;
  surrogate: boolean;
  same_weights?: boolean;
  latency_ms: number;
  context: {
    tokens: string[];
    kinds: ("prompt" | "generated")[];
  };
  attention: {
    layers: number[];
    selected: Record<string, AttentionLayer>;
    head_count: number;
  };
  lens: {
    layer: number;
    topk: TokenOption[];
    tuned: boolean;
    agreement: number | null;
  }[];
  hidden: number[];
  residual: {
    layer: number;
    attn_norm: number;
    mlp_norm: number;
    attn_logit: number;
    mlp_logit: number;
  }[];
  features: FeatureActivation[];
  kv: {
    used: number;
    capacity: number;
    bytes: number;
    bytes_per_token: number;
  } | null;
  vector: Vec3 | null;
}

export interface FeatureActivation {
  id: number;
  label: string;
  activation: number;
  tokens: string[];
}

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface ProjectionPoint extends Vec3 {
  id: string;
  token: string;
  index: number;
  kind: "prompt" | "generated";
  step: number | null;
}

export interface ProjectionEvent {
  points: ProjectionPoint[];
  reducer: "UMAP" | "PCA" | "deterministic";
  fitted_at: number;
}

export interface RagItem {
  id: string;
  text: string;
  source: string;
  heading?: string;
  cosine: number | null;
  bm25: number;
  hybrid: number;
  hits: string[];
}

export interface Metrics {
  cpu: number | null;
  ram: number | null;
  ram_total: number | null;
  gpu: number | null;
  vram_used: number | null;
  vram_total: number | null;
  tps: number | null;
  ttft_ms: number | null;
  tokens: number;
  context: number;
  lens_lag_ms?: number | null;
}

export interface Usage {
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
}

export interface KVInfo {
  used: number;
  capacity: number;
  bytes: number;
  bytes_per_token: number;
  layers: number;
  kv_heads: number;
  head_dim: number;
  dtype: string;
  estimated: boolean;
}

export interface SessionSummary {
  id: string;
  model: string;
  mode: RuntimeMode;
  created_at: number;
  tokens: number;
  preview: string;
  has_lens: boolean;
  path?: string;
}

export interface ReplayStep {
  index: number;
  delta: string;
  kind: TokenKind;
  topk: TokenOption[];
  chosen_prob: number | null;
  entropy: number | null;
  at: number;
  probe: Probe | null;
}

export type ServerEvent =
  | {
      type: "hello";
      version: string;
      config: AppConfig;
      lmstudio: { reachable: boolean; models: ModelInfo[]; error?: string | null };
      lens: LensStatus;
      native?: NativeStatus;
    }
  | {
      type: "started";
      session: SessionMeta;
      resumed?: boolean;
    }
  | { type: "rag"; items: RagItem[]; query?: string; injected: boolean }
  | { type: "token"; step: StepToken }
  | { type: "probe"; step: number; probe: Probe }
  | { type: "projection"; projection: ProjectionEvent }
  | { type: "metrics"; metrics: Metrics }
  | { type: "kv"; kv: KVInfo }
  | {
      type: "done";
      usage: Usage;
      finish_reason: string;
      elapsed_s: number;
      capabilities?: { logprobs?: boolean };
      exact?: boolean;
      surrogate?: boolean;
    }
  | { type: "status"; scope: string; state: string; message: string }
  | { type: "error"; code: string; message: string; recoverable?: boolean }
  | { type: "pong"; at: number };

export type ClientEvent =
  | {
      type: "start";
      payload: {
        messages: ChatMessage[];
        model?: string;
        mode?: RuntimeMode;
        sampling?: Partial<SamplingParams>;
        rag?: { enabled: boolean; top_k: number; inject: boolean };
        lens?: { enabled: boolean; model?: string | null };
        system?: string;
      };
    }
  | { type: "stop" }
  | { type: "ping" };

export interface SessionRecord {
  meta: SessionMeta;
  steps: ReplayStep[];
  messages: ChatMessage[];
  usage: Usage | null;
  finish_reason: string;
  elapsed_s: number;
}

export interface LayerView {
  layer: number;
  mean: number[];
  heads: number[][];
  headTypes: HeadType[];
}
