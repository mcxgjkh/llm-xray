import { useMemo } from "react";
import type { HeadType } from "../types";
import type { Xray } from "../state/useXray";
import { EmptyState, Panel } from "./Panel";
import { Tag } from "./Controls";
import { AttentionHeatmap } from "./visuals/AttentionHeatmap";
import { NeuralTopology } from "./visuals/NeuralTopology";
import { ResidualBars } from "./visuals/ResidualBars";

const HEAD_LABEL: Record<HeadType, string> = {
  "previous-token": "前一 Token",
  local: "局部窗口",
  syntactic: "语法/标点",
  entity: "实体回指",
  global: "长程汇聚",
  diffuse: "弥散",
};

export function AttentionView({
  xray,
  selectedLayer,
  setSelectedLayer,
  selectedHead,
  setSelectedHead,
}: {
  xray: Xray;
  selectedLayer: number;
  setSelectedLayer: (layer: number) => void;
  selectedHead: number | "mean";
  setSelectedHead: (head: number | "mean") => void;
}) {
  const probe = xray.activeProbe;
  const selected = probe?.attention.selected?.[String(selectedLayer)] ?? null;
  const availableLayers = probe?.attention.layers ?? [];
  const counts = useMemo(() => {
    const map = new Map<HeadType, number>();
    for (const type of selected?.head_types ?? []) map.set(type, (map.get(type) ?? 0) + 1);
    return [...map.entries()].sort((a, b) => b[1] - a[1]);
  }, [selected]);

  return (
    <div className="view view-attention">
      <div className="section-head">
        <div>
          <h1>注意力</h1>
          <p>逐层观察当前 Token 把目光放在哪里。热图颜色是权重，不是因果证据。</p>
        </div>
        <div className="section-head-meta">
          <Tag tone={probe?.surrogate ? "warn" : "positive"}>
            {probe ? (probe.surrogate ? "替代透镜" : "精确张量") : "等待数据"}
          </Tag>
          <span className="mono">{probe ? `上下文 ${probe.context.tokens.length} · ${probe.attention.head_count} heads` : "等待数据"}</span>
        </div>
      </div>

      <div className="attention-grid">
        <div className="attention-main">
          <AttentionHeatmap
            probe={probe}
            history={Array.from(xray.probes.values()).sort((a, b) => a.step - b.step)}
            layer={selectedLayer}
            head={selectedHead}
            maxColumns={48}
            onLayerChange={setSelectedLayer}
            onHeadChange={setSelectedHead}
            height={410}
          />
        </div>
        <div className="attention-side">
          <Panel
            title="焦点层"
            kicker={availableLayers.length ? `${availableLayers.length} 个已捕获层` : "无"}
            bodyClassName="layer-picks-body"
          >
            {!availableLayers.length ? (
              <EmptyState title="尚无层快照" body="开启替代透镜并产生至少一个 Token。" compact />
            ) : (
              <div className="layer-picks">
                {availableLayers.map((layer) => {
                  const active = layer === selectedLayer;
                  const layerData = probe?.attention.selected?.[String(layer)];
                  return (
                    <button
                      key={layer}
                      className={`layer-pick ${active ? "is-active" : ""}`.trim()}
                      onClick={() => setSelectedLayer(layer)}
                    >
                      <span className="num">L{String(layer).padStart(2, "0")}</span>
                      <span className="layer-pick-bar">
                        <span style={{ width: `${Math.min(100, (layerData?.mean.at(-1) ?? 0) * 260)}%` }} />
                      </span>
                      <span className="num layer-pick-value">{(layerData?.mean.at(-1) ?? 0).toFixed(2)}</span>
                    </button>
                  );
                })}
              </div>
            )}
          </Panel>
          <Panel title="头功能分类" kicker="启发式统计">
            {!counts.length ? (
              <p className="panel-copy muted">分类器会依据首列权重、熵与跨实体模式，把每个头归入一个可读类型。它用于导航，不是定论。</p>
            ) : (
              <div className="head-type-list">
                {counts.map(([type, count]) => (
                  <div key={type} className="head-type-row">
                    <span className={`head-type-swatch type-${type}`} aria-hidden="true" />
                    <span>{HEAD_LABEL[type]}</span>
                    <span className="num">{count}</span>
                  </div>
                ))}
                <p className="panel-copy faint">当前层共 {selected?.head_types?.length ?? 0} 个头；分类随上下文变化。</p>
              </div>
            )}
          </Panel>
        </div>
      </div>

      <div className="attention-bottom">
        <ResidualBars residual={probe?.residual ?? []} selectedToken={xray.activeStep?.delta} height={214} />
        <NeuralTopology probe={probe} activeToken={xray.activeStep?.delta} height={260} />
      </div>
    </div>
  );
}
