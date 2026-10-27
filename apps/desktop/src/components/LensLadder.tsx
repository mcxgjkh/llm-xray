import { formatPct } from "../lib/sampling";
import { tokenLabel } from "../lib/format";
import type { Probe } from "../types";
import { EmptyState, Panel } from "./Panel";

export function LensLadder({
  probe,
  selectedLayer,
  onSelectLayer,
  compact = false,
}: {
  probe: Probe | null;
  selectedLayer: number;
  onSelectLayer: (layer: number) => void;
  compact?: boolean;
}) {
  const layers = probe?.lens ?? [];
  const agreement = layers.find((layer) => layer.agreement !== null)?.agreement ?? null;
  const tuned = layers.some((layer) => layer.tuned);
  return (
    <Panel
      title="透镜梯"
      kicker={
        probe
          ? probe.surrogate
            ? tuned
              ? "Tuned Lens · 仿射校准"
              : "Logit Lens · 未校准"
            : "Tuned Lens · 原生精确"
          : "等待透镜"
      }
      aside={
        <span className="panel-note">
          {agreement !== null ? (
            <>
              top-1 一致 <span className="num">{(agreement * 100).toFixed(0)}%</span>
            </>
          ) : (
            probe && !probe.surrogate ? "精确张量" : "替代模型"
          )}
        </span>
      }
      className={`panel-lens ${compact ? "panel-lens-compact" : ""}`.trim()}
      bodyClassName="lens-body"
    >
      {!probe || !layers.length ? (
        <EmptyState
          title="透镜通道没有开启"
          body="LM Studio 不暴露逐层隐状态。开启替代透镜或切换到原生精确模式后，这里显示每一层经 Tuned Lens 投影后的候选词，并明确标注数据来源。"
          compact
        />
      ) : (
        <div className="lens-list">
          {layers.map((layer) => {
            const top = layer.topk.slice(0, 3);
            return (
              <button
                key={layer.layer}
                className={`lens-row ${selectedLayer === layer.layer ? "is-selected" : ""}`.trim()}
                onClick={() => onSelectLayer(layer.layer)}
                title={`选择第 ${layer.layer} 层`}
              >
                <span className="lens-layer num">L{String(layer.layer).padStart(2, "0")}</span>
                <span className="lens-bar" aria-hidden="true">
                  <span style={{ width: `${Math.max(1, (top[0]?.prob ?? 0) * 100)}%` }} />
                </span>
                <span className="lens-tokens">
                  {top.map((option, index) => (
                    <span key={`${option.token}-${index}`} className={`lens-token rank-${index}`}>
                      {tokenLabel(option.token)}
                      <span className="lens-prob num">{formatPct(option.prob, 0)}</span>
                    </span>
                  ))}
                </span>
                <span className="lens-top-prob num">{top[0] ? formatPct(top[0].prob, 0) : "—"}</span>
              </button>
            );
          })}
        </div>
      )}
    </Panel>
  );
}
