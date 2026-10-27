import type { ReactNode } from "react";
import type { Metrics, SessionMeta, Usage } from "../types";
import type { MetricPoint } from "../state/useXray";
import { int, ms, num, seconds } from "../lib/format";
import { Sparkline } from "./Sparkline";

function Cell({
  label,
  value,
  unit,
  sub,
  spark,
  title,
}: {
  label: string;
  value: string;
  unit?: string;
  sub?: string;
  spark?: ReactNode;
  title?: string;
}) {
  return (
    <div className="perf-cell" title={title}>
      <span className="perf-label">{label}</span>
      <span className="perf-value num">
        {value}
        {unit ? <span className="perf-unit">{unit}</span> : null}
      </span>
      {spark ? <span className="perf-spark">{spark}</span> : null}
      {sub ? <span className="perf-sub">{sub}</span> : null}
    </div>
  );
}

export function PerformanceStrip({
  metrics,
  history,
  tokens,
  elapsed,
  ttft,
  usage,
  session,
  finishReason,
  generating,
  contextTokens,
}: {
  metrics: Metrics | null;
  history: MetricPoint[];
  tokens: number;
  elapsed: number;
  ttft: number | null;
  usage: Usage | null;
  session: SessionMeta | null;
  finishReason: string;
  generating: boolean;
  contextTokens?: number;
}) {
  const tpsValues = history.map((point) => point.tps);
  const gpuValues = history.map((point) => point.gpu);
  const context = contextTokens || metrics?.context || tokens;
  const usageText = usage
    ? `prompt ${int(usage.prompt_tokens)} · completion ${int(usage.completion_tokens)}`
    : session
      ? session.mode === "demo"
        ? "演示会话"
        : session.mode === "native"
          ? "原生精确"
          : "LM Studio"
      : "尚未开始";
  return (
    <section className="performance-strip" aria-label="推理性能">
      <Cell
        label="生成速度"
        value={metrics?.tps != null ? num(metrics.tps, 1) : "—"}
        unit="tok/s"
        spark={<Sparkline values={tpsValues} min={0} color="var(--accent)" label="生成速度趋势" />}
        sub={generating ? "正在采样" : finishReason ? `结束：${finishReason}` : "待命"}
      />
      <Cell
        label="首字延迟"
        value={ttft != null ? String(Math.round(ttft)) : metrics?.ttft_ms != null ? String(Math.round(metrics.ttft_ms)) : "—"}
        unit={ttft != null || metrics?.ttft_ms != null ? "ms" : ""}
        sub={elapsed ? `已运行 ${seconds(elapsed)}` : "TTFT"}
      />
      <Cell
        label="上下文 / 步数"
        value={int(context)}
        sub={`${int(tokens)} 个生成步 · ${usageText}`}
      />
      <Cell
        label="GPU 利用率"
        value={metrics?.gpu != null ? num(metrics.gpu, 0) : "—"}
        unit={metrics?.gpu != null ? "%" : ""}
        spark={<Sparkline values={gpuValues} min={0} max={100} color="var(--mlp)" label="GPU 利用率趋势" />}
        sub={metrics?.gpu == null ? "未检测到 nvidia-smi" : "整卡利用率"}
      />
      <Cell
        label="显存"
        value={metrics?.vram_used != null ? num(metrics.vram_used, 1) : "—"}
        unit={metrics?.vram_used != null ? ` / ${num(metrics.vram_total, 1)} GiB` : ""}
        sub={metrics?.ram != null ? `系统内存 ${num(metrics.ram, 1)} GiB` : "VRAM"}
      />
      <Cell
        label="透镜状态"
        value={
          session?.mode === "demo"
            ? "演示"
            : session?.mode === "native"
              ? "精确"
              : session?.capabilities.lens
                ? "替代"
                : "关闭"
        }
        sub={
          session?.mode === "demo"
            ? "合成探针"
            : session?.mode === "native"
              ? "同一次前向"
              : metrics?.lens_lag_ms != null
                ? `延迟 ${ms(metrics.lens_lag_ms)}`
                : session?.model ?? "未连接"
        }
        title={session?.model}
      />
    </section>
  );
}
