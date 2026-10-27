import { useMemo } from "react";
import { formatProb } from "../lib/sampling";
import { num, tokenLabel } from "../lib/format";
import { resample } from "../lib/sampling";
import type { SamplingParams, StepToken } from "../types";
import { SliderRow } from "./Controls";
import { EmptyState, Panel } from "./Panel";

export function ProbabilityCloud({
  step,
  sampling,
  onSamplingChange,
  recentTokens,
  height = "default",
}: {
  step: StepToken | null;
  sampling: SamplingParams;
  onSamplingChange?: (patch: Partial<SamplingParams>) => void;
  recentTokens: string[];
  height?: "default" | "compact";
}) {
  const adjusted = useMemo(
    () => resample(step?.topk ?? [], sampling, recentTokens, step?.delta),
    [step, sampling, recentTokens],
  );
  const raw = useMemo(
    () =>
      resample(
        step?.topk ?? [],
        { temperature: 1, top_p: 1, top_k: 9999, repetition_penalty: 1 },
        [],
        step?.delta,
      ),
    [step],
  );
  const rawMap = useMemo(() => new Map(raw.options.map((item) => [item.token, item.prob])), [raw.options]);
  const maxProb = adjusted.options[0]?.prob ?? 1;

  return (
    <Panel
      title="概率云"
      kicker={step ? `第 ${step.index + 1} 步 · 原始 top-${step.topk.length}` : "等待生成"}
      aside={
        <div className="prob-head-stats">
          <span>
            熵 <span className="num">{step?.entropy != null ? step.entropy.toFixed(2) : "—"}</span> bit
          </span>
          <span>
            选中 <span className="num">{formatProb(adjusted.chosenProb)}</span>
          </span>
        </div>
      }
      className="panel-probability"
      bodyClassName="prob-body"
    >
      {!step || !step.topk.length ? (
        <EmptyState
          title="还没有可观测的候选分布"
          body="发送一条消息后，这里会显示每个生成步骤的 top-k 概率。LM Studio 未返回 logprobs 时会保持空态，而不是伪造分布。"
          compact
        />
      ) : (
        <>
          <div className={`prob-list ${height === "compact" ? "prob-list-compact" : ""}`.trim()}>
            {adjusted.options.slice(0, height === "compact" ? 7 : 12).map((option) => {
              const rawProb = rawMap.get(option.token) ?? 0;
              const isChosen = option.token === step.delta;
              const width = Math.max(0.8, (option.prob / maxProb) * 100);
              return (
                <div key={`${option.rank}-${option.token}`} className={`prob-row ${isChosen ? "is-chosen" : ""}`.trim()}>
                  <span className="prob-rank num">{String(option.rank).padStart(2, "0")}</span>
                  <span className="prob-token" title={option.token}>
                    {tokenLabel(option.token)}
                  </span>
                  <span className="prob-track" aria-hidden="true">
                    <span className="prob-fill" style={{ width: `${width}%` }} />
                    <span className="prob-raw-tick" style={{ left: `${Math.min(100, rawProb * 100)}%` }} />
                  </span>
                  <span className="prob-value num">{formatProb(option.prob)}</span>
                  <span className="prob-logprob num">{option.logprob.toFixed(2)}</span>
                </div>
              );
            })}
          </div>
          <div className="prob-legend">
            <span className="legend-swatch swatch-fill" /> 实时重算
            <span className="legend-swatch swatch-tick" /> 模型原始概率
            <span className="prob-legend-note">拖动参数只改变观测分布；下一 Token 生效</span>
          </div>
          {onSamplingChange ? (
            <div className="prob-controls">
              <SliderRow
                label="温度"
                value={sampling.temperature}
                min={0.05}
                max={2}
                step={0.05}
                onChange={(temperature) => onSamplingChange({ temperature })}
                format={(value) => value.toFixed(2)}
              />
              <SliderRow
                label="Top-P"
                value={sampling.top_p}
                min={0.05}
                max={1}
                step={0.01}
                onChange={(top_p) => onSamplingChange({ top_p })}
                format={(value) => value.toFixed(2)}
              />
              <SliderRow
                label="Top-K"
                value={sampling.top_k}
                min={1}
                max={100}
                step={1}
                onChange={(top_k) => onSamplingChange({ top_k })}
                format={(value) => String(Math.round(value))}
              />
              <div className="prob-control-stat">
                <span className="muted">保留候选</span>
                <span className="num">{adjusted.kept}</span>
                <span className="muted">·</span>
                <span className="muted">温度</span>
                <span className="num">{num(sampling.temperature, 2)}</span>
              </div>
            </div>
          ) : null}
        </>
      )}
    </Panel>
  );
}
