import { useMemo } from "react";
import type { Xray } from "../state/useXray";
import { EmptyState, Panel } from "./Panel";
import { Tag } from "./Controls";
import { ParticleCloud3D } from "./visuals/ParticleCloud3D";
import { KVMemoryMap } from "./visuals/KVMemoryMap";
import { resolveKv } from "../lib/kv";

export function VectorsView({ xray }: { xray: Xray }) {
  const probe = xray.activeProbe;
  const maxAbs = useMemo(
    () => Math.max(0.01, ...(probe?.hidden ?? []).map((value) => Math.abs(value))),
    [probe],
  );
  const generated = xray.projection.filter((point) => point.kind === "generated").length;

  return (
    <div className="view view-vectors">
      <div className="section-head">
        <div>
          <h1>向量空间</h1>
          <p>把替代模型的最终层隐状态压到三维。距离是语义线索，不是原模型的内部坐标。</p>
        </div>
        <div className="section-head-meta">
          <Tag tone="warn">{xray.projectionReducer}</Tag>
          <span className="mono">{xray.projection.length} 点 · {generated} 生成</span>
        </div>
      </div>

      <div className="vectors-grid">
        <ParticleCloud3D points={xray.projection} activeStep={xray.activeStep?.index ?? null} reducer={xray.projectionReducer} height={488} />
        <div className="vectors-side">
          <Panel
            title="隐状态摘要"
            kicker={probe ? "最终层 · 16 维降采样" : "等待透镜"}
            aside={probe ? <span className="panel-note mono">step {probe.step + 1}</span> : null}
          >
            {!probe ? (
              <EmptyState title="没有向量" body="开启替代透镜后，每个 Token 的最终层隐状态会在这里展开。" compact />
            ) : (
              <>
                <div className="vector-strip" role="img" aria-label="最终层隐状态十六维摘要">
                  {probe.hidden.map((value, index) => {
                    const ratio = Math.min(1, Math.abs(value) / maxAbs);
                    const height = 8 + ratio * 42;
                    return (
                      <span key={index} className="vector-bar-wrap" title={`维度 ${index + 1}: ${value.toFixed(3)}`}>
                        <span
                          className={`vector-bar ${value >= 0 ? "is-positive" : "is-negative"}`.trim()}
                          style={{ height }}
                        />
                      </span>
                    );
                  })}
                </div>
                <div className="vector-axis">
                  <span>−</span>
                  <span>最终层残差 · 归一化显示</span>
                  <span>+</span>
                </div>
              </>
            )}
          </Panel>
          <Panel title="稀疏特征" kicker="MLP 探针 · 非 SAE" className="panel-feature-tall">
            {!probe ? (
              <EmptyState title="没有激活" body="透镜就绪后显示与当前 Token 最相关的稀疏神经元方向。" compact />
            ) : (
              <div className="feature-list feature-list-roomy">
                {probe.features.map((feature) => (
                  <div key={feature.id} className="feature-row">
                    <span className="feature-label">
                      {feature.label}
                      <span className="feature-tokens">{feature.tokens.slice(0, 3).join(" · ")}</span>
                    </span>
                    <span className="feature-track" aria-hidden="true">
                      <span style={{ width: `${Math.max(2, Math.min(100, feature.activation * 100))}%` }} />
                    </span>
                    <span className="feature-value num">{feature.activation.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            )}
          </Panel>
          <KVMemoryMap kv={resolveKv(xray.kv, probe?.kv)} contextTokens={xray.contextTokens.length} height={138} />
        </div>
      </div>
    </div>
  );
}
