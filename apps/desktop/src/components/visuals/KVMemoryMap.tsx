import { useState } from "react";
import type { KVInfo } from "../../types";
import { clamp, fmtBytes, fmtInt, fmtPercent } from "./format";

export interface KVMemoryMapProps {
  kv: KVInfo | null;
  contextTokens?: number;
  height?: number;
}

type ColumnMode = "layer" | "head";

const GRID_ROWS = 16;
const CELL = 14;
const GAP = 2;
const MAX_COLUMNS = 32;
const AXIS_WIDTH = 34;

interface CellSpec {
  row: number;
  column: number;
  used: boolean;
  partial: boolean;
  usedTokens: number;
  blockTokens: number;
}

function buildCells(columns: number, used: number, capacity: number): CellSpec[] {
  const cells: CellSpec[] = [];
  if (columns <= 0 || capacity <= 0) return cells;
  const blockTokens = capacity / GRID_ROWS;
  for (let column = 0; column < columns; column += 1) {
    for (let row = 0; row < GRID_ROWS; row += 1) {
      const usedTokens = clamp(used - row * blockTokens, 0, blockTokens);
      cells.push({
        row,
        column,
        used: usedTokens >= blockTokens - 1e-9,
        partial: usedTokens > 0 && usedTokens < blockTokens - 1e-9,
        usedTokens,
        blockTokens,
      });
    }
  }
  return cells;
}

export function KVMemoryMap({ kv, contextTokens, height = 280 }: KVMemoryMapProps) {
  const [mode, setMode] = useState<ColumnMode>("layer");

  const layers = Math.max(1, Math.round(kv?.layers ?? 0));
  const kvHeads = Math.max(1, Math.round(kv?.kv_heads ?? 0));
  const rawColumns = mode === "layer" ? layers : kvHeads;
  const columns = clamp(rawColumns, 1, MAX_COLUMNS);
  const capacity = Math.max(0, kv?.capacity ?? 0);
  const used = clamp(kv?.used ?? 0, 0, capacity);
  const cells = buildCells(columns, used, capacity);
  const perColumn = rawColumns / columns;
  const blockTokens = capacity > 0 ? capacity / GRID_ROWS : 0;
  const usedRows = cells.filter((cell) => cell.used || cell.partial).reduce((max, cell) => Math.max(max, cell.row + 1), 0);
  const occupied = cells.filter((cell) => cell.used).length;
  const partialCount = cells.filter((cell) => cell.partial).length;
  const free = cells.length - occupied - partialCount;

  const columnLabel = (column: number): string => {
    if (mode === "layer") {
      const from = Math.floor(column * perColumn);
      const to = Math.max(from, Math.ceil((column + 1) * perColumn) - 1);
      return from === to ? `第 ${from} 层` : `第 ${from}–${to} 层`;
    }
    const from = Math.floor(column * perColumn);
    const to = Math.max(from, Math.ceil((column + 1) * perColumn) - 1);
    return from === to ? `头 ${from}` : `头 ${from}–${to}`;
  };

  const viewWidth = columns * (CELL + GAP) + AXIS_WIDTH;
  const viewHeight = GRID_ROWS * (CELL + GAP);
  const occupancy = fmtPercent(used, capacity, 1);

  if (!kv) {
    return (
      <section className="lx-panel lx-kv" aria-label="KV 缓存映射">
        <header className="lx-panel__head">
          <div className="lx-panel__heading">
            <h3 className="lx-panel__title">KV Cache</h3>
            <p className="lx-panel__sub">上下文显存映射</p>
          </div>
        </header>
        <div className="lx-kv__stage" style={{ minHeight: `${height}px` }}>
          <div className="lx-kv__grid" aria-hidden="true">
            {Array.from({ length: GRID_ROWS * 12 }, (_, index) => (
              <span className="lx-kv__placeholder" key={index} />
            ))}
          </div>
          <div className="lx-empty">
            <p className="lx-empty__title">KV 数据不可用</p>
            <p className="lx-empty__hint">选择会话并收到 kv 事件后显示缓存占用。</p>
            <p className="lx-empty__hint">LM Studio 未公开真实 KV 缓存，本图只按模型结构估算。</p>
          </div>
        </div>
        <footer className="lx-kv__foot">
          <p className="lx-panel__note">
            LM Studio 未公开真实 KV 缓存；真实读数需以显存监控为准，本图不做假精确。
          </p>
          {contextTokens != null ? (
            <p className="lx-panel__note num">当前上下文 {fmtInt(contextTokens)} Token</p>
          ) : null}
        </footer>
      </section>
    );
  }

  return (
    <section className="lx-panel lx-kv" aria-label="KV 缓存映射">
      <header className="lx-panel__head">
        <div className="lx-panel__heading">
          <h3 className="lx-panel__title">KV Cache</h3>
          <p className="lx-panel__sub">
            上下文 × {mode === "layer" ? "层" : "头"} 的占用块
          </p>
        </div>
        <div className="lx-panel__controls" role="group" aria-label="KV 列视图">
          <button
            type="button"
            className={`lx-button${mode === "layer" ? " is-active" : ""}`}
            aria-pressed={mode === "layer"}
            onClick={() => setMode("layer")}
          >
            层
          </button>
          <button
            type="button"
            className={`lx-button${mode === "head" ? " is-active" : ""}`}
            aria-pressed={mode === "head"}
            onClick={() => setMode("head")}
          >
            头
          </button>
        </div>
      </header>

      <div className="lx-panel__meta">
        <span className="lx-chip num">估算 {fmtBytes(kv.bytes)}</span>
        <span className="lx-chip num">每 Token {fmtBytes(kv.bytes_per_token)}</span>
        <span className="lx-chip">{kv.dtype}</span>
        <span className="lx-chip num">
          上下文 {fmtInt(used)} / {fmtInt(capacity)}
        </span>
        <span className={`lx-chip ${kv.estimated ? "lx-chip--warn" : ""}`}>
          {kv.estimated ? "结构估算" : "上报值"}
        </span>
        {contextTokens != null ? <span className="lx-chip num">界面上下文 {fmtInt(contextTokens)}</span> : null}
      </div>

      <div className="lx-kv__bar" aria-hidden="true">
        <span className="lx-kv__bar-fill" style={{ width: `${clamp(capacity > 0 ? (used / capacity) * 100 : 0, 0, 100)}%` }} />
      </div>

      <div className="lx-kv__stage" style={{ maxHeight: `${height}px` }} role="img" aria-label={`KV 缓存占用矩阵，${columns} 列 ${GRID_ROWS} 行`}>
        <svg
          className="lx-kv__svg"
          viewBox={`0 0 ${viewWidth} ${viewHeight}`}
          preserveAspectRatio="xMidYMin meet"
          role="presentation"
        >
          {Array.from({ length: GRID_ROWS }, (_, row) => {
            const tokenStart = Math.round(row * blockTokens);
            return (
              <text
                key={`axis-${row}`}
                x={AXIS_WIDTH - 6}
                y={row * (CELL + GAP) + CELL - 3}
                className="lx-kv__axis"
                textAnchor="end"
              >
                {row % 2 === 0 ? tokenStart : ""}
              </text>
            );
          })}
          {cells.map((cell) => {
            const x = AXIS_WIDTH + cell.column * (CELL + GAP);
            const y = cell.row * (CELL + GAP);
            const tokenStart = Math.round(cell.row * cell.blockTokens);
            const tokenEnd = Math.round((cell.row + 1) * cell.blockTokens);
            const columnLabelText = columnLabel(cell.column);
            const ratio = cell.blockTokens > 0 ? cell.usedTokens / cell.blockTokens : 0;
            return (
              <g key={`${cell.column}-${cell.row}`}>
                <title>{`${columnLabelText} · Token ${tokenStart}–${tokenEnd} · ${
                  cell.used ? "已用" : cell.partial ? "部分已用" : "空闲"
                }`}</title>
                <rect className="lx-kv__cell lx-kv__cell--free" x={x} y={y} width={CELL} height={CELL} rx={1} />
                {ratio > 0 ? (
                  <rect
                    className="lx-kv__cell lx-kv__cell--used"
                    x={x}
                    y={y + CELL * (1 - ratio)}
                    width={CELL}
                    height={Math.max(1, CELL * ratio)}
                  />
                ) : null}
              </g>
            );
          })}
        </svg>
      </div>

      <div className="lx-kv__legend">
        <span className="lx-legend-key">
          <span className="lx-kv__swatch lx-kv__swatch--used" aria-hidden="true" />
          已用 {fmtInt(occupied + partialCount)} 块
        </span>
        <span className="lx-legend-key">
          <span className="lx-kv__swatch lx-kv__swatch--partial" aria-hidden="true" />
          其中边界 {fmtInt(partialCount)} 块
        </span>
        <span className="lx-legend-key">
          <span className="lx-kv__swatch lx-kv__swatch--free" aria-hidden="true" />
          空闲 {fmtInt(free)} 块
        </span>
        <span className="lx-kv__legend-spacer" />
        <span className="lx-kv__occupancy num">上下文占比 {occupancy}</span>
      </div>

      <dl className="lx-kv__stats">
        <div>
          <dt>结构</dt>
          <dd className="num">
            {fmtInt(layers)} 层 × {fmtInt(kvHeads)} KV 头 × {fmtInt(kv.head_dim)} 维
          </dd>
        </div>
        <div>
          <dt>dtype</dt>
          <dd className="num">{kv.dtype}</dd>
        </div>
        <div>
          <dt>每 Token</dt>
          <dd className="num">{fmtBytes(kv.bytes_per_token)}</dd>
        </div>
        <div>
          <dt>已用块行</dt>
          <dd className="num">
            {usedRows} / {GRID_ROWS}
          </dd>
        </div>
      </dl>

      <footer className="lx-kv__foot">
        <p className="lx-panel__note">
          LM Studio 未公开真实 KV 缓存；此处按模型结构（层 × KV 头 × head_dim × dtype）与上下文位置估算，
          不是显存实时读数。
        </p>
        {contextTokens != null && contextTokens !== used ? (
          <p className="lx-panel__note num">
            界面上下文 {fmtInt(contextTokens)} Token 与估算占用 {fmtInt(used)} Token 存在差异，属于正常范围。
          </p>
        ) : null}
      </footer>
    </section>
  );
}

export default KVMemoryMap;
export type { ColumnMode };
