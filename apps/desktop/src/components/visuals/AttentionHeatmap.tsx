import { useEffect, useMemo, useState } from "react";
import type { AttentionLayer, HeadType, Probe } from "../../types";
import { fmtFixed, fmtInt, fmtWeight } from "./format";

const MAX_ROWS = 48;
const MAX_VISIBLE_COLUMNS = 64;
const MIN_VISIBLE_COLUMNS = 8;
const DEFAULT_VISIBLE_COLUMNS = 48;
const COMPACT_VISIBLE_COLUMNS = 16;
const CELL_WIDTH = 26;
const LABEL_WIDTH = 64;
const HEAT_LEVELS = 9;
const MAX_HISTORY_PROBES = 96;

export interface AttentionHeatmapProps {
  probe: Probe | null;
  layer: number;
  head: number | "mean";
  /**
   * Previous probes of the same session, oldest first. Each probe contributes
   * its newest attention row for the selected layer/head, so the panel can draw
   * a real generated-token x context matrix instead of a single stripe.
   */
  history?: readonly Probe[];
  /** Columns kept visible (window is the tail). Default 48, compact default 16. */
  maxColumns?: number;
  /** Shortcut for dense Observatory slots: 16 visible columns. */
  compact?: boolean;
  onLayerChange?(l: number): void;
  onHeadChange?(h: number | "mean"): void;
  height?: number;
}

interface HeadTypeMeta {
  type: HeadType;
  label: string;
  hint: string;
}

const HEAD_TYPES: readonly HeadTypeMeta[] = [
  { type: "previous-token", label: "前一词", hint: "峰值落在上一个 Token，常见于复制与续写" },
  { type: "local", label: "局部", hint: "集中在邻近窗口，负责局部语法与拼写" },
  { type: "syntactic", label: "句法", hint: "跨过标点连到从句或短语中心，呈结构依赖" },
  { type: "entity", label: "实体", hint: "反复指向同一实体，维持指代一致" },
  { type: "global", label: "全局", hint: "权重散布到远端 Token，承担长程汇总" },
  { type: "diffuse", label: "弥散", hint: "没有明显峰值，接近均匀分布" },
];

const HEAD_TYPE_META = new Map<HeadType, HeadTypeMeta>(HEAD_TYPES.map((meta) => [meta.type, meta]));

export interface HeatRow {
  contextIndex: number;
  token: string;
  step?: number;
  values: (number | null)[];
}

interface HoverCell {
  rowIndex: number;
  contextIndex: number;
  columnIndex: number;
  rowToken: string;
  columnToken: string;
  value: number | null;
  step?: number;
}

function shortToken(token: string, max = 2): string {
  const cleaned = token.replace(/\s+/g, "").trim();
  if (cleaned.length === 0) return "∅";
  return Array.from(cleaned).slice(0, max).join("");
}

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}

function lastGeneratedIndex(kinds: readonly string[]): number {
  for (let index = kinds.length - 1; index >= 0; index -= 1) {
    if (kinds[index] === "generated") return index;
  }
  return kinds.length - 1;
}

interface ProbeRow {
  step: number;
  tokenIndex: number;
  token: string;
  values: number[];
}

/**
 * Pull the newest attention row for one probe. `mean` and per-head rows are
 * already single causal rows in the wire format; if a replay payload carries a
 * flattened block, its last row is the newest token.
 */
function rowFromProbe(probe: Probe, layer: number, head: number | "mean"): ProbeRow | null {
  const layerView = probe.attention?.selected?.[String(layer)];
  if (!layerView) return null;
  const tokens = probe.context?.tokens ?? [];
  const contextLength = tokens.length;
  if (contextLength === 0) return null;
  const headRows = Array.isArray(layerView.heads) ? layerView.heads : [];
  const vector = head === "mean" ? layerView.mean : headRows[head];
  if (!vector || vector.length === 0) return null;
  const values = vector.length >= contextLength ? vector.slice(vector.length - contextLength) : vector.slice();
  const tokenIndex = contextLength - 1;
  return {
    step: probe.step,
    tokenIndex,
    token: tokens[tokenIndex] ?? "",
    values,
  };
}

/**
 * Accumulate one generated-token row per probe, left-aligned onto the latest
 * context. Columns beyond that step's context are void, which is exactly the
 * lower-triangular shape of causal attention.
 */
function buildAccumulatedRows(
  probes: readonly Probe[],
  layer: number,
  head: number | "mean",
  contextLength: number,
): HeatRow[] {  if (contextLength === 0) return [];
  const byStep = new Map<number, ProbeRow>();
  for (const probe of probes) {
    const row = rowFromProbe(probe, layer, head);
    if (!row) continue;
    byStep.set(row.step, row);
  }
  const sourceRows = Array.from(byStep.values())
    .sort((a, b) => a.step - b.step)
    .slice(-MAX_ROWS);
  return sourceRows.map((source) => {
    const values: (number | null)[] = Array.from({ length: contextLength }, (_, column) => {
      const value = source.values[column];
      return value != null && Number.isFinite(value) ? value : null;
    });
    return {
      contextIndex: Math.min(source.tokenIndex, contextLength - 1),
      token: source.token,
      step: source.step,
      values,
    };
  });
}

/** Fallback for a single probe that carries a flattened block of rows. */
function buildRowsFromVector(
  tokens: readonly string[],
  kinds: readonly string[],
  vector: readonly number[] | undefined,
): HeatRow[] {
  const contextLength = tokens.length;
  if (contextLength === 0 || !vector || vector.length === 0) return [];
  const rowCount = Math.floor(vector.length / contextLength);
  if (rowCount <= 1) {
    const rowIndex = Math.max(0, Math.min(contextLength - 1, lastGeneratedIndex(kinds)));
    const values = tokens.map((_, column) => (column < vector.length ? vector[column] ?? null : null));
    return [{ contextIndex: rowIndex, token: tokens[rowIndex] ?? "", values }];
  }
  const firstRow = Math.max(0, contextLength - rowCount);
  const rows: HeatRow[] = [];
  for (let contextIndex = firstRow; contextIndex < contextLength; contextIndex += 1) {
    const offset = (contextIndex - firstRow) * contextLength;
    const values = tokens.map((_, column) =>
      column <= contextIndex ? vector[offset + column] ?? null : null,
    );
    rows.push({ contextIndex, token: tokens[contextIndex] ?? "", values });
  }
  return rows.slice(-MAX_ROWS);
}

function levelOf(value: number | null, max: number): number {
  if (value == null || !Number.isFinite(value) || max <= 0) return -1;
  const normalized = Math.min(1, Math.max(0, value / max));
  return Math.min(HEAT_LEVELS - 1, Math.floor(normalized * HEAT_LEVELS));
}

export function AttentionHeatmap({
  probe,
  layer,
  head,
  history,
  maxColumns,
  compact = false,
  onLayerChange,
  onHeadChange,
  height = 340,
}: AttentionHeatmapProps) {
  const [activeLayerState, setActiveLayerState] = useState<number>(layer);
  const [activeHeadState, setActiveHeadState] = useState<number | "mean">(head);
  const [hover, setHover] = useState<HoverCell | null>(null);

  useEffect(() => {
    setActiveLayerState(layer);
  }, [layer]);

  useEffect(() => {
    setActiveHeadState(head);
  }, [head]);

  const sourceProbes = useMemo(() => {
    const list: Probe[] = [];
    if (history && history.length > 0) list.push(...history.slice(-MAX_HISTORY_PROBES));
    if (probe) list.push(probe);
    return list;
  }, [history, probe]);
  const referenceProbe: Probe | null =
    probe ?? (sourceProbes.length > 0 ? sourceProbes[sourceProbes.length - 1] : null);

  const layerOptions = useMemo(() => {
    const selected = referenceProbe?.attention?.selected ?? {};
    const keys = Object.keys(selected)
      .map((key) => Number(key))
      .filter((value) => Number.isFinite(value));
    const declared = referenceProbe?.attention?.layers ?? [];
    const present = declared.filter((value) => Number.isFinite(value) && selected[String(value)] != null);
    const source = present.length > 0 ? present : keys;
    return Array.from(new Set(source)).sort((a, b) => a - b);
  }, [referenceProbe]);

  // Keep the controlled parent in sync when it points at a layer that this
  // snapshot did not capture; otherwise a <select> would show an empty option.
  useEffect(() => {
    if (layerOptions.length > 0 && !layerOptions.includes(layer)) {
      onLayerChange?.(layerOptions[0]);
    }
  }, [layer, layerOptions, onLayerChange]);

  // Controlled callbacks win over local state, so the selectors always mirror
  // the parent's selectedLayer / selectedHead. Local state keeps the panel
  // usable when it is rendered standalone without callbacks.
  const requestedLayer = onLayerChange ? layer : activeLayerState;
  const activeLayer = layerOptions.includes(requestedLayer)
    ? requestedLayer
    : layerOptions[0] ?? requestedLayer;
  const layerView: AttentionLayer | undefined = referenceProbe?.attention?.selected?.[String(activeLayer)];
  const headCountFromProbe = Math.max(0, referenceProbe?.attention?.head_count ?? 0);
  const availableHeads = useMemo(() => {
    const list = Array.isArray(layerView?.heads) ? layerView.heads : [];
    return list.slice(0, headCountFromProbe > 0 ? headCountFromProbe : list.length);
  }, [layerView, headCountFromProbe]);
  const requestedHead = onHeadChange ? head : activeHeadState;
  const activeHead: number | "mean" =
    requestedHead !== "mean" && availableHeads[requestedHead] != null ? requestedHead : "mean";
  const headType = activeHead === "mean" ? null : layerView?.head_types?.[activeHead] ?? null;
  const headMeta = headType ? HEAD_TYPE_META.get(headType) ?? null : null;

  // Same idea for a controlled head selection that does not exist on this layer.
  useEffect(() => {
    if (typeof head === "number" && availableHeads[head] == null) {
      onHeadChange?.("mean");
    }
  }, [head, availableHeads.length, onHeadChange]);

  const tokens = referenceProbe?.context?.tokens ?? [];
  const kinds = referenceProbe?.context?.kinds ?? [];
  const visibleColumnCount = clamp(
    maxColumns ?? (compact ? COMPACT_VISIBLE_COLUMNS : DEFAULT_VISIBLE_COLUMNS),
    MIN_VISIBLE_COLUMNS,
    MAX_VISIBLE_COLUMNS,
  );
  const columnStart = Math.max(0, tokens.length - visibleColumnCount);

  const rows = useMemo(() => {
    if (!referenceProbe || tokens.length === 0) return [];
    if (sourceProbes.length > 1) {
      const accumulated = buildAccumulatedRows(sourceProbes, activeLayer, activeHead, tokens.length);
      if (accumulated.length > 0) {
        // Rows older than the visible column window would render as empty
        // stripes, so keep only rows that still intersect the window.
        return accumulated.filter((row) => row.contextIndex >= columnStart);
      }
    }
    const fallbackVector =
      activeHead === "mean" ? layerView?.mean : availableHeads[activeHead] ?? layerView?.mean;
    return buildRowsFromVector(tokens, kinds, fallbackVector).filter((row) => row.contextIndex >= columnStart);
  }, [
    referenceProbe,
    sourceProbes,
    tokens,
    kinds,
    activeLayer,
    activeHead,
    layerView,
    availableHeads,
    columnStart,
  ]);

  const columns = useMemo(
    () => Array.from({ length: tokens.length - columnStart }, (_, index) => columnStart + index),
    [columnStart, tokens.length],
  );
  const maxWeight = useMemo(() => {
    let max = 0;
    for (const row of rows) {
      for (const value of row.values) {
        if (value != null && Number.isFinite(value) && value > max) max = value;
      }
    }
    return max;
  }, [rows]);
  const frameMessage =
    referenceProbe == null ? "透镜未启用" : rows.length === 0 ? "尚无注意力数据" : null;
  const accumulatedSteps = useMemo(
    () => new Set(sourceProbes.map((item) => item.step)).size,
    [sourceProbes],
  );

  const setLayer = (next: number) => {
    setActiveLayerState(next);
    onLayerChange?.(next);
  };
  const setHead = (next: number | "mean") => {
    setActiveHeadState(next);
    onHeadChange?.(next);
  };

  return (
    <section className="lx-panel lx-heatmap" aria-label="注意力热图">
      <header className="lx-panel__head">
        <div className="lx-panel__heading">
          <h3 className="lx-panel__title">注意力</h3>
          <p className="lx-panel__sub">
            逐 Token 对上下文的权重 · 下三角矩阵
            {referenceProbe ? ` · 步 ${referenceProbe.step} · ${referenceProbe.model}` : ""}
            {sourceProbes.length > 1 ? ` · 累积 ${fmtInt(accumulatedSteps)} 步` : ""}
          </p>
        </div>
        <div className="lx-panel__controls">
          <label className="lx-field">
            <span className="lx-field__label">层</span>
            <select
              className="lx-select num"
              value={layerOptions.length > 0 ? String(activeLayer) : ""}
              disabled={layerOptions.length === 0}
              onChange={(event) => {
                const next = Number(event.target.value);
                if (Number.isFinite(next)) setLayer(next);
              }}
              aria-label="注意力层"
            >
              {layerOptions.length === 0 ? <option value="">—</option> : null}
              {layerOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <label className="lx-field">
            <span className="lx-field__label">头</span>
            <select
              className="lx-select num"
              value={activeHead === "mean" ? "mean" : String(activeHead)}
              disabled={availableHeads.length === 0}
              onChange={(event) => {
                const raw = event.target.value;
                const next = raw === "mean" ? "mean" : Number(raw);
                if (next === "mean" || Number.isFinite(next)) setHead(next);
              }}
              aria-label="注意力头；未捕获分头数据时仅显示均值"
            >
              <option value="mean">均值</option>
              {availableHeads.map((_, index) => (
                <option key={index} value={index}>
                  #{index + 1}
                </option>
              ))}
            </select>
          </label>
        </div>
      </header>

      <div className="lx-panel__meta">
        <span className="lx-chip">层 {layerOptions.length > 0 ? activeLayer : "—"}</span>
        <span className="lx-chip">头 {activeHead === "mean" ? "均值" : `#${activeHead + 1}`}</span>
        {headMeta ? <span className="lx-chip lx-chip--accent">{headMeta.label}</span> : null}
        {rows.length === 1 ? <span className="lx-chip">仅最新行</span> : null}
        {rows.length !== accumulatedSteps && accumulatedSteps > 1 ? (
          <span className="lx-chip num">窗口 {fmtInt(rows.length)} 行</span>
        ) : null}
        {referenceProbe?.surrogate ? <span className="lx-chip">替代透镜</span> : null}
        {referenceProbe?.same_weights ? <span className="lx-chip lx-chip--warn">同权重</span> : null}
        {referenceProbe != null ? (
          <span className="lx-chip num">延迟 {fmtFixed(referenceProbe.latency_ms, 1)} ms</span>
        ) : null}
        {tokens.length > columns.length ? (
          <span className="lx-panel__note num">窗口 {columns.length} 列 · 可横向滚动</span>
        ) : null}
        {accumulatedSteps > MAX_ROWS ? (
          <span className="lx-panel__note num">仅保留最近 {MAX_ROWS} 步</span>
        ) : null}
      </div>

      <div className="lx-heatmap__body" style={{ maxHeight: `${height}px` }}>
        {rows.length > 0 ? (
          <div className="lx-heatmap__scroller">
            <div
              className="lx-heatmap__grid"
              style={{ gridTemplateColumns: `${LABEL_WIDTH}px repeat(${columns.length}, ${CELL_WIDTH}px)` }}
              role="img"
              aria-label={`注意力矩阵，${rows.length} 行，${columns.length} 列`}
            >
              <div className="lx-heatmap__corner" aria-hidden="true">
                <span className="num">行 / 列</span>
              </div>
              {columns.map((column) => {
                const token = tokens[column] ?? "";
                const generated = kinds[column] === "generated";
                return (
                  <div
                    key={`col-${column}`}
                    className={`lx-heatmap__col-label${generated ? " is-generated" : ""}`}
                    title={`上下文 #${column} · ${token || "∅"} · ${generated ? "生成" : "提示"}`}
                  >
                    <span>{shortToken(token, 2)}</span>
                  </div>
                );
              })}

              {rows.map((row, rowIndex) => (
                <div
                  className="lx-heatmap__row"
                  key={`row-${row.step ?? row.contextIndex}-${rowIndex}`}
                  style={{ display: "contents" }}
                >
                  <div
                    className="lx-heatmap__row-label"
                    title={`${row.step != null ? `步 ${row.step} · ` : ""}上下文 #${row.contextIndex} · ${
                      row.token || "∅"
                    }`}
                  >
                    <span className="lx-heatmap__row-index num">{row.contextIndex}</span>
                    <span className="lx-heatmap__row-token">{shortToken(row.token, 2)}</span>
                  </div>
                  {columns.map((column) => {
                    const value = row.values[column] ?? null;
                    const level = levelOf(value, maxWeight);
                    const isDiagonal = column === row.contextIndex;
                    return (
                      <div
                        key={`cell-${rowIndex}-${column}`}
                        className={`lx-heatmap__cell${level < 0 ? " lx-heat--void" : ` lx-heat-${level}`}${
                          isDiagonal ? " is-diagonal" : ""
                        }`}
                        title={`${row.step != null ? `步 ${row.step} · ` : ""}第 ${row.contextIndex} 行 · 第 ${column} 列 · ${
                          row.token || "∅"
                        } ← ${tokens[column] ?? "∅"} · 权重 ${fmtWeight(value)}`}
                        onMouseEnter={() =>
                          setHover({
                            rowIndex,
                            contextIndex: row.contextIndex,
                            columnIndex: column,
                            rowToken: row.token,
                            columnToken: tokens[column] ?? "",
                            value,
                            step: row.step,
                          })
                        }
                        onMouseLeave={() => setHover(null)}
                      />
                    );
                  })}
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="lx-heatmap__placeholder" aria-hidden="true">
            {Array.from({ length: 3 }, (_, rowIndex) => (
              <div className="lx-heatmap__placeholder-row" key={rowIndex}>
                {Array.from({ length: 12 }, (_, columnIndex) => (
                  <span className="lx-heatmap__placeholder-cell" key={columnIndex} />
                ))}
              </div>
            ))}
          </div>
        )}

        {frameMessage ? (
          <div className="lx-empty">
            <p className="lx-empty__title">{frameMessage}</p>
            <p className="lx-empty__hint">
              {referenceProbe == null
                ? "开启替代透镜后，这里逐行累积每个生成 Token 的注意力。"
                : "该层在历史探针里尚无注意力行；切换层或等待下一个探测。"}
            </p>
            <p className="lx-empty__hint">原始模型不公开注意力，本图为透镜模型的观测结果。</p>
          </div>
        ) : null}
      </div>

      <div className="lx-heatmap__readout" aria-live="polite">
        {hover ? (
          <>
            <span className="lx-heatmap__readout-pair">
              行 {hover.rowIndex + 1} #{hover.contextIndex}{" "}
              <span className="lx-heatmap__readout-token">{hover.rowToken || "∅"}</span>
            </span>
            <span className="lx-heatmap__readout-arrow" aria-hidden="true">
              ←
            </span>
            <span className="lx-heatmap__readout-pair">
              列 {hover.columnIndex} <span className="lx-heatmap__readout-token">{hover.columnToken || "∅"}</span>
            </span>
            <span className="lx-heatmap__readout-weight num">{fmtWeight(hover.value)}</span>
            {hover.step != null ? <span className="muted num">步 {hover.step}</span> : null}
          </>
        ) : (
          <span className="muted">
            悬停单元格查看完整 Token 与权重
            {maxWeight > 0 ? ` · 当前行内最大 ${fmtFixed(maxWeight, 3)}` : ""}
          </span>
        )}
      </div>

      <div className="lx-heatmap__foot">
        <div className="lx-heatmap__scale" aria-label="权重色阶">
          <span className="lx-heatmap__scale-label muted">低</span>
          {Array.from({ length: HEAT_LEVELS }, (_, level) => (
            <span key={level} className={`lx-heatmap__scale-swatch lx-heat-${level}`} />
          ))}
          <span className="lx-heatmap__scale-label muted">高</span>
          <span className="lx-heatmap__scale-caption">权重</span>
        </div>
        <div className="lx-heatmap__legend">
          {HEAD_TYPES.map((meta) => (
            <div className="lx-heatmap__legend-item" key={meta.type} title={meta.hint}>
              <span className={`lx-headtype lx-headtype--${meta.type}`} aria-hidden="true" />
              <span className="lx-heatmap__legend-label">{meta.label}</span>
              <span className="lx-heatmap__legend-hint">{meta.hint}</span>
            </div>
          ))}
          <p className="lx-heatmap__legend-note">
            头分类由首行权重、熵、标点距离与实体重复等统计量启发式推断，不是模型自带标签。
          </p>
        </div>
        <dl className="lx-heatmap__stats">
          <div>
            <dt>上下文</dt>
            <dd className="num">{fmtInt(tokens.length)}</dd>
          </div>
          <div>
            <dt>累积步</dt>
            <dd className="num">{fmtInt(accumulatedSteps)}</dd>
          </div>
          <div>
            <dt>可见行</dt>
            <dd className="num">{fmtInt(rows.length)}</dd>
          </div>
          <div>
            <dt>可见列</dt>
            <dd className="num">{fmtInt(columns.length)}</dd>
          </div>
          <div>
            <dt>最新步</dt>
            <dd className="num">{referenceProbe ? referenceProbe.step : "—"}</dd>
          </div>
        </dl>
      </div>
    </section>
  );
}

export default AttentionHeatmap;
export { HEAD_TYPES };
export type { HeadTypeMeta };
