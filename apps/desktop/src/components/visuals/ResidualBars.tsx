import { useMemo, useState } from "react";
import type { Probe } from "../../types";
import { fmtFixed, fmtInt, fmtSigned } from "./format";

export interface ResidualBarsProps {
  residual: Probe["residual"];
  selectedToken?: string;
  height?: number;
}

interface ResidualRow {
  layer: number;
  attnNorm: number;
  mlpNorm: number;
  attnLogit: number;
  mlpLogit: number;
}

const EMPTY_ROWS = 6;

function toRows(residual: Probe["residual"]): ResidualRow[] {
  if (!Array.isArray(residual)) return [];
  return residual
    .filter(
      (row) =>
        Number.isFinite(row.layer) &&
        Number.isFinite(row.attn_norm) &&
        Number.isFinite(row.mlp_norm) &&
        Number.isFinite(row.attn_logit) &&
        Number.isFinite(row.mlp_logit),
    )
    .map((row) => ({
      layer: row.layer,
      attnNorm: row.attn_norm,
      mlpNorm: row.mlp_norm,
      attnLogit: row.attn_logit,
      mlpLogit: row.mlp_logit,
    }))
    .sort((a, b) => a.layer - b.layer);
}

export function ResidualBars({ residual, selectedToken, height = 300 }: ResidualBarsProps) {
  const rows = useMemo(() => toRows(residual), [residual]);
  const [hover, setHover] = useState<ResidualRow | null>(null);

  const maxAbs = useMemo(
    () => rows.reduce((max, row) => Math.max(max, Math.abs(row.attnLogit), Math.abs(row.mlpLogit)), 0),
    [rows],
  );
  const scale = maxAbs > 0 ? maxAbs : 1;

  const barStyle = (value: number): { width: string; left?: string; right?: string } => {
    const width = `${Math.min(50, (Math.abs(value) / scale) * 50)}%`;
    return value >= 0 ? { width, left: "50%" } : { width, right: "50%" };
  };

  return (
    <section className="lx-panel lx-bars" aria-label="残差贡献柱">
      <header className="lx-panel__head">
        <div className="lx-panel__heading">
          <h3 className="lx-panel__title">残差贡献</h3>
          <p className="lx-panel__sub">
            注意力分支与 MLP 分支的 signed logit 一阶近似
            {selectedToken ? ` · 选中 ${selectedToken}` : ""}
          </p>
        </div>
        <div className="lx-panel__meta lx-panel__meta--inline">
          <span className="lx-chip num">层 {fmtInt(rows.length)}</span>
          <span className="lx-chip num">最大 |logit| {fmtFixed(maxAbs, 3)}</span>
        </div>
      </header>

      <div className="lx-bars__legend">
        <span className="lx-legend-key">
          <span className="lx-bars__swatch lx-bars__swatch--attn" aria-hidden="true" />
          注意力
        </span>
        <span className="lx-legend-key">
          <span className="lx-bars__swatch lx-bars__swatch--mlp" aria-hidden="true" />
          MLP
        </span>
        <span className="lx-bars__legend-zero" aria-hidden="true" />
        <span className="lx-panel__note">负值向左，正值向右；零线居中</span>
      </div>

      <div className="lx-bars__head" aria-hidden="true">
        <span className="lx-bars__head-layer">层</span>
        <span className="lx-bars__head-track">贡献</span>
        <span className="lx-bars__head-value">attn</span>
        <span className="lx-bars__head-value lx-bars__head-value--mlp">mlp</span>
      </div>

      <div className="lx-bars__list" style={{ maxHeight: `${height}px` }}>
        {rows.length > 0 ? (
          rows.map((row) => {
            const active = hover?.layer === row.layer;
            return (
              <div
                className={`lx-bars__row${active ? " is-active" : ""}`}
                key={row.layer}
                title={`第 ${row.layer} 层 · attn_norm ${fmtFixed(row.attnNorm, 3)} · mlp_norm ${fmtFixed(
                  row.mlpNorm,
                  3,
                )} · attn_logit ${fmtSigned(row.attnLogit, 3)} · mlp_logit ${fmtSigned(row.mlpLogit, 3)}`}
                onMouseEnter={() => setHover(row)}
                onMouseLeave={() => setHover(null)}
              >
                <span className="lx-bars__layer num">{row.layer}</span>
                <div className="lx-bars__track">
                  <span className="lx-bars__zero" aria-hidden="true" />
                  <span
                    className="lx-bars__bar lx-bars__bar--attn"
                    style={barStyle(row.attnLogit)}
                    aria-hidden="true"
                  />
                  <span
                    className="lx-bars__bar lx-bars__bar--mlp"
                    style={barStyle(row.mlpLogit)}
                    aria-hidden="true"
                  />
                </div>
                <span className="lx-bars__value num">{fmtSigned(row.attnLogit, 3)}</span>
                <span className="lx-bars__value lx-bars__value--mlp num">{fmtSigned(row.mlpLogit, 3)}</span>
              </div>
            );
          })
        ) : (
          <>
            {Array.from({ length: EMPTY_ROWS }, (_, index) => (
              <div className="lx-bars__row lx-bars__row--placeholder" key={`placeholder-${index}`} aria-hidden="true">
                <span className="lx-bars__layer num">{index}</span>
                <div className="lx-bars__track">
                  <span className="lx-bars__zero" />
                </div>
                <span className="lx-bars__value">—</span>
                <span className="lx-bars__value">—</span>
              </div>
            ))}
            <div className="lx-empty lx-bars__empty">
              <p className="lx-empty__title">尚无残差数据</p>
              <p className="lx-empty__hint">透镜启用后，每层的 attention / MLP 分支贡献会在这里逐行出现。</p>
            </div>
          </>
        )}
      </div>

      <div className="lx-bars__readout" aria-live="polite">
        {hover ? (
          <dl className="lx-bars__metrics">
            <div>
              <dt>attn_norm</dt>
              <dd className="num">{fmtFixed(hover.attnNorm, 3)}</dd>
            </div>
            <div>
              <dt>mlp_norm</dt>
              <dd className="num">{fmtFixed(hover.mlpNorm, 3)}</dd>
            </div>
            <div>
              <dt>attn_logit</dt>
              <dd className="num">{fmtSigned(hover.attnLogit, 3)}</dd>
            </div>
            <div>
              <dt>mlp_logit</dt>
              <dd className="num">{fmtSigned(hover.mlpLogit, 3)}</dd>
            </div>
            <div>
              <dt>层</dt>
              <dd className="num">{hover.layer}</dd>
            </div>
          </dl>
        ) : (
          <span className="muted">悬停层查看 attn_norm / mlp_norm / attn_logit / mlp_logit</span>
        )}
      </div>

      <footer className="lx-bars__foot">
        <p className="lx-panel__note">
          柱长为分支增量经最终 LayerNorm 投影到词表方向的一阶近似，不是严格的因果贡献；范数只用于悬停读数。
        </p>
        {selectedToken ? (
          <p className="lx-panel__note num">选中 Token：{selectedToken}</p>
        ) : (
          <p className="lx-panel__note">未选中 Token：贡献针对当前最新 Token。</p>
        )}
      </footer>
    </section>
  );
}

export default ResidualBars;
export type { ResidualRow };
