import type { Xray } from "../state/useXray";
import { EmptyState, Panel } from "./Panel";
import { ProbabilityCloud } from "./ProbabilityCloud";
import { LensLadder } from "./LensLadder";
import { DecodingTree } from "./DecodingTree";
import { AttentionHeatmap } from "./visuals/AttentionHeatmap";
import { ResidualBars } from "./visuals/ResidualBars";
import { KVMemoryMap } from "./visuals/KVMemoryMap";
import { Tag } from "./Controls";
import { resolveKv } from "../lib/kv";

export function ObservatoryView({
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
  return (
    <div className="view view-observatory">
      <div className="obs-top">
        <ProbabilityCloud
          step={xray.activeStep}
          sampling={xray.sampling}
          recentTokens={xray.recentTokens}
          onSamplingChange={(patch) => xray.setSampling((current) => ({ ...current, ...patch }))}
        />
        <div className="obs-lens-stack">
          <LensLadder probe={probe} selectedLayer={selectedLayer} onSelectLayer={setSelectedLayer} />
          <Panel title="观测注记" kicker="如何读这台仪器" className="panel-note-card" bodyClassName="note-body">
            <div className="readout-note">
              <Tag tone={probe?.surrogate ? "warn" : probe ? "positive" : "neutral"}>
                {probe ? (probe.surrogate ? "替代透镜" : "精确张量") : "无透镜"}
              </Tag>
              <p>
                {probe && !probe.surrogate
                  ? "精确模式：概率、注意力、Tuned Lens、残差与向量都来自当前原生模型的同一次前向，不是替代模型。"
                  : probe
                    ? "注意力、隐状态与残差贡献来自独立观测模型，不是右侧 LM Studio 模型的内部张量。概率云与速度来自真实 API。"
                    : "概率云、解码树与性能数据来自 LM Studio。开启透镜或切换到原生精确模式后，才能观察逐层注意力与残差流。"}
              </p>
            </div>
          </Panel>
        </div>
      </div>

      <div className="obs-bottom">
        <AttentionHeatmap
          probe={probe}
          history={Array.from(xray.probes.values()).sort((a, b) => a.step - b.step)}
          layer={selectedLayer}
          head={selectedHead}
          compact
          onLayerChange={setSelectedLayer}
          onHeadChange={setSelectedHead}
          height={286}
        />
        <div className="obs-side">
          <ResidualBars residual={probe?.residual ?? []} selectedToken={xray.activeStep?.delta} height={152} />
          <KVMemoryMap kv={resolveKv(xray.kv, probe?.kv)} contextTokens={xray.contextTokens.length} height={122} />
        </div>
      </div>

      <div className="obs-third">
        <DecodingTree
          steps={xray.steps}
          sampling={xray.sampling}
          selectedStep={xray.replayMode ? xray.activeStepIndex : null}
          onSelectStep={xray.selectStep}
        />
        <Panel title="稀疏特征探针" kicker={probe ? "top MLP 神经元 · 非 SAE" : "等待透镜"} className="panel-features">
          {!probe ? (
            <EmptyState title="没有特征读数" body="开启替代透镜或切到原生精确模式后，这里列出当前 Token 激活的稀疏特征。它是神经元探针，不冒充预训练 SAE。" compact />
          ) : (
            <div className="feature-list">
              {probe.features.slice(0, 6).map((feature) => (
                <div key={feature.id} className="feature-row">
                  <span className="feature-label">{feature.label}</span>
                  <span className="feature-track" aria-hidden="true">
                    <span style={{ width: `${Math.max(2, Math.min(100, feature.activation * 100))}%` }} />
                  </span>
                  <span className="feature-value num">{feature.activation.toFixed(2)}</span>
                </div>
              ))}
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}
