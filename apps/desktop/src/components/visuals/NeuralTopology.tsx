import { useEffect, useRef, useState } from "react";
import type { FeatureActivation, Probe } from "../../types";
import { clamp, fmtInt, shorten } from "./format";
import { readPalette, withAlpha, type VisualPalette } from "./palette";

export interface NeuralTopologyProps {
  probe: Probe | null;
  activeToken?: string;
  height?: number;
}

interface LayerGroup {
  from: number;
  to: number;
  attn: number;
  mlp: number;
  attnLogit: number;
  mlpLogit: number;
}

const DEFAULT_HEIGHT = 300;
const MAX_DISPLAY_LAYERS = 12;
const MAX_INPUT_TOKENS = 10;
const MAX_FEATURES = 5;

function aggregateLayers(residual: Probe["residual"], max: number): LayerGroup[] {
  if (residual.length === 0) return [];
  if (residual.length <= max) {
    return residual.map((row) => ({
      from: row.layer,
      to: row.layer,
      attn: row.attn_norm,
      mlp: row.mlp_norm,
      attnLogit: row.attn_logit,
      mlpLogit: row.mlp_logit,
    }));
  }
  const groups: LayerGroup[] = [];
  for (let index = 0; index < max; index += 1) {
    const start = Math.floor((index * residual.length) / max);
    const end = Math.max(start + 1, Math.floor(((index + 1) * residual.length) / max));
    const slice = residual.slice(start, end);
    const average = (pick: (row: Probe["residual"][number]) => number) =>
      slice.reduce((sum, row) => sum + pick(row), 0) / slice.length;
    groups.push({
      from: slice[0]?.layer ?? index,
      to: slice[slice.length - 1]?.layer ?? index,
      attn: average((row) => row.attn_norm),
      mlp: average((row) => row.mlp_norm),
      attnLogit: average((row) => row.attn_logit),
      mlpLogit: average((row) => row.mlp_logit),
    });
  }
  return groups;
}

function fitText(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string {
  if (maxWidth <= 0) return "";
  if (ctx.measureText(text).width <= maxWidth) return text;
  let out = text;
  while (out.length > 1 && ctx.measureText(`${out}…`).width > maxWidth) {
    out = out.slice(0, -1);
  }
  return `${out}…`;
}

function line(
  ctx: CanvasRenderingContext2D,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  color: string,
  width = 1,
): void {
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}

function dot(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  radius: number,
  fill: string,
  stroke?: string,
  strokeWidth = 1,
): void {
  ctx.beginPath();
  ctx.arc(x, y, Math.max(0.5, radius), 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = strokeWidth;
    ctx.stroke();
  }
}

function pickInputIndices(
  tokens: readonly string[],
  kinds: readonly string[],
  activeToken: string | undefined,
  limit: number,
): number[] {
  if (tokens.length === 0) return [];
  const indices: number[] = [];
  const wanted = Math.max(1, limit);
  const active = activeToken?.trim();
  if (active) {
    for (let index = tokens.length - 1; index >= 0 && indices.length < 2; index -= 1) {
      if ((tokens[index] ?? "").trim() === active) indices.push(index);
    }
  }
  for (let index = tokens.length - 1; index >= 0 && indices.length < wanted; index -= 1) {
    if (kinds[index] === "generated" && !indices.includes(index)) indices.push(index);
  }
  for (let index = tokens.length - 1; index >= 0 && indices.length < wanted; index -= 1) {
    if (!indices.includes(index)) indices.push(index);
  }
  return indices.sort((a, b) => a - b);
}

function drawTopology(
  canvas: HTMLCanvasElement,
  host: HTMLElement,
  width: number,
  height: number,
  probe: Probe | null,
  activeToken: string | undefined,
): void {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.round(width));
  const h = Math.max(1, Math.round(height));
  const pixelWidth = Math.round(w * dpr);
  const pixelHeight = Math.round(h * dpr);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const palette: VisualPalette = readPalette(host);

  const top = 18;
  const bottom = 26;
  const usableHeight = Math.max(24, h - top - bottom);

  // Faint layer guides keep the empty frame legible without animation.
  const inputX = 62;
  const featureX = w - 30;
  const layerStart = 104;
  const layerEnd = Math.max(layerStart + 30, w - 110);
  const layerArea = layerEnd - layerStart;
  const maxGroups = clamp(Math.floor(layerArea / 26), 3, MAX_DISPLAY_LAYERS);

  const residual = (probe?.residual ?? [])
    .filter(
      (row) =>
        Number.isFinite(row.layer) &&
        Number.isFinite(row.attn_norm) &&
        Number.isFinite(row.mlp_norm) &&
        Number.isFinite(row.attn_logit) &&
        Number.isFinite(row.mlp_logit),
    )
    .slice()
    .sort((a, b) => a.layer - b.layer);
  const groups = aggregateLayers(residual, maxGroups);

  if (groups.length > 0) {
    groups.forEach((_, index) => {
      const x = groups.length === 1 ? layerStart + layerArea / 2 : layerStart + (index / (groups.length - 1)) * layerArea;
      line(ctx, x, top, x, h - bottom + 4, withAlpha(palette.line, 0.7), 1);
    });
  }

  if (!probe || residual.length === 0 || groups.length === 0) {
    return;
  }

  const tokens = probe.context?.tokens ?? [];
  const kinds = probe.context?.kinds ?? [];
  const features = [...(probe.features ?? [])]
    .filter((feature) => Number.isFinite(feature.activation))
    .sort((a, b) => b.activation - a.activation)
    .slice(0, MAX_FEATURES);
  const featuresActivation = features.reduce((max, feature) => Math.max(max, feature.activation), 0) || 1;
  const inputLimit = Math.max(3, Math.min(MAX_INPUT_TOKENS, Math.floor(usableHeight / 24)));
  const inputIndices = pickInputIndices(tokens, kinds, activeToken, inputLimit);
  const active = activeToken?.trim();
  let selectedInput = -1;
  if (active) {
    selectedInput = inputIndices.findIndex((index) => (tokens[index] ?? "").trim() === active);
    if (selectedInput < 0) {
      for (let index = inputIndices.length - 1; index >= 0; index -= 1) {
        if (kinds[inputIndices[index]] === "generated") {
          selectedInput = index;
          break;
        }
      }
    }
  }

  const layerX = (index: number) =>
    groups.length === 1 ? layerStart + layerArea / 2 : layerStart + (index / (groups.length - 1)) * layerArea;
  const attnY = top + usableHeight * 0.36;
  const mlpY = top + usableHeight * 0.64;
  const inputY = (index: number) =>
    inputIndices.length === 1
      ? top + usableHeight / 2
      : top + 4 + (index / (inputIndices.length - 1)) * (usableHeight - 8);
  const maxNorm = groups.reduce((max, group) => Math.max(max, group.attn, group.mlp), 0) || 1;
  const intensity = (value: number) => clamp(value / maxNorm, 0.08, 1);

  const last = groups.length - 1;
  const outputY = h - bottom - 2;
  const outputToken = activeToken ?? probe.lens?.[probe.lens.length - 1]?.topk?.[0]?.token ?? "";
  const featureY = (index: number) =>
    top + 4 + (features.length === 1 ? usableHeight * 0.12 : (index / Math.max(1, features.length - 1)) * usableHeight * 0.34);

  // Faint structural edges first: selected paths are drawn last and stay on top.
  const normalEdge = withAlpha(palette.lineStrong, 0.85);
  for (let index = 0; index < groups.length; index += 1) {
    const x = layerX(index);
    if (index > 0) {
      const px = layerX(index - 1);
      line(ctx, px, attnY, x, attnY, normalEdge);
      line(ctx, px, mlpY, x, mlpY, normalEdge);
      line(ctx, px, attnY, x, mlpY, withAlpha(palette.line, 0.9));
      line(ctx, px, mlpY, x, attnY, withAlpha(palette.line, 0.9));
    }
  }
  inputIndices.forEach((contextIndex, index) => {
    const y = inputY(index);
    line(ctx, inputX, y, layerStart, attnY, withAlpha(palette.line, 0.75));
    line(ctx, inputX, y, layerStart, mlpY, withAlpha(palette.line, 0.75));
  });
  line(ctx, layerEnd, attnY, featureX - 6, outputY, withAlpha(palette.line, 0.8));
  line(ctx, layerEnd, mlpY, featureX - 6, outputY, withAlpha(palette.line, 0.8));
  features.forEach((feature, index) => {
    line(ctx, layerEnd, attnY, featureX - 6, featureY(index), withAlpha(palette.line, 0.7));
  });

  // Copper path for the active token, everything else stays ash.
  if (selectedInput >= 0) {
    const pathColor = withAlpha(palette.accent, 0.95);
    const inputNodeY = inputY(selectedInput);
    line(ctx, inputX, inputNodeY, layerStart, attnY, pathColor, 1.5);
    line(ctx, inputX, inputNodeY, layerStart, mlpY, pathColor, 1.5);
    for (let index = 1; index < groups.length; index += 1) {
      const px = layerX(index - 1);
      const x = layerX(index);
      line(ctx, px, attnY, x, attnY, pathColor, 1.25);
      line(ctx, px, mlpY, x, mlpY, pathColor, 1.25);
    }
    line(ctx, layerEnd, attnY, featureX - 6, outputY, pathColor, 1.25);
    line(ctx, layerEnd, mlpY, featureX - 6, outputY, pathColor, 1.25);
  }

  // Input tokens.
  ctx.font = "11px 'Segoe UI Variable Text', 'Segoe UI', 'Microsoft YaHei UI', sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  inputIndices.forEach((contextIndex, index) => {
    const y = inputY(index);
    const selected = index === selectedInput;
    const label = shorten(tokens[contextIndex] ?? "∅", 5);
    dot(
      ctx,
      inputX,
      y,
      selected ? 4 : 2.5,
      selected ? withAlpha(palette.accent, 1) : withAlpha(palette.muted, 0.85),
      selected ? withAlpha(palette.inkStrong, 0.9) : undefined,
    );
    ctx.fillStyle = selected ? palette.inkStrong : withAlpha(palette.muted, selectedInput >= 0 ? 0.65 : 0.9);
    ctx.fillText(fitText(ctx, label, 44), inputX - 9, y);
  });
  if (tokens.length === 0) {
    ctx.fillStyle = withAlpha(palette.faint, 0.9);
    ctx.font = "10px 'Cascadia Mono', Consolas, monospace";
    ctx.fillText("无上下文", inputX - 9, top + usableHeight / 2);
  }

  // Layer columns: attention (copper) and MLP (olive) nodes, brightness from norms.
  ctx.textAlign = "center";
  groups.forEach((group, index) => {
    const x = layerX(index);
    const onPath = selectedInput >= 0;
    const attnRadius = 2.5 + 7 * intensity(group.attn);
    const mlpRadius = 2.5 + 7 * intensity(group.mlp);
    if (index === 0) {
      dot(ctx, x, attnY, attnRadius, withAlpha(palette.attention, 0.35 + 0.65 * intensity(group.attn)));
      dot(ctx, x, mlpY, mlpRadius, withAlpha(palette.mlp, 0.35 + 0.65 * intensity(group.mlp)));
    } else {
      dot(
        ctx,
        x,
        attnY,
        attnRadius,
        withAlpha(palette.attention, 0.28 + 0.72 * intensity(group.attn)),
        onPath ? withAlpha(palette.accentSoft, 0.95) : undefined,
        onPath ? 1.25 : 1,
      );
      dot(
        ctx,
        x,
        mlpY,
        mlpRadius,
        withAlpha(palette.mlp, 0.28 + 0.72 * intensity(group.mlp)),
        onPath ? withAlpha(palette.accentSoft, 0.95) : undefined,
        onPath ? 1.25 : 1,
      );
    }
    ctx.font = "9px 'Cascadia Mono', Consolas, monospace";
    ctx.fillStyle = withAlpha(palette.faint, 0.95);
    // Aggregated columns are labelled by their first layer so 12 narrow columns
    // do not collide; the header already says how many columns are aggregated.
    const label = `L${group.from}`;
    ctx.fillText(fitText(ctx, label, groups.length <= 1 ? 70 : Math.max(16, layerArea / groups.length - 4)), x, h - bottom + 12);
  });

  // Sparse features (feature-probe, never labelled SAE) orbit the last layer.
  ctx.font = "9px 'Cascadia Mono', Consolas, monospace";
  ctx.textAlign = "right";
  features.forEach((feature: FeatureActivation, index) => {
    const y = featureY(index);
    const ratio = clamp(feature.activation / featuresActivation, 0.1, 1);
    dot(ctx, featureX - 6, y, 1.8 + 4.5 * ratio, withAlpha(palette.warning, 0.35 + 0.65 * ratio));
    ctx.fillStyle = withAlpha(palette.warning, 0.9);
    ctx.fillText(`#${feature.id}`, featureX - 15, y);
  });

  // Output token node.
  dot(ctx, featureX - 6, outputY, 4.5, withAlpha(palette.accentSoft, 1), withAlpha(palette.inkStrong, 0.8), 1);
  ctx.textAlign = "left";
  ctx.font = "10px 'Segoe UI Variable Text', 'Segoe UI', 'Microsoft YaHei UI', sans-serif";
  ctx.fillStyle = withAlpha(palette.ink, 0.9);
  ctx.fillText(fitText(ctx, shorten(outputToken || "—", 6), 70), featureX + 1, outputY + 14);
}

export function NeuralTopology({ probe, activeToken, height = DEFAULT_HEIGHT }: NeuralTopologyProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const host = hostRef.current;
    if (!host || typeof ResizeObserver === "undefined") return undefined;
    let frame = 0;
    const observer = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (!rect) return;
      if (frame !== 0) window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(() => {
        frame = 0;
        const width = Math.round(rect.width);
        const nextHeight = Math.round(rect.height);
        setSize((previous) =>
          previous.width === width && previous.height === nextHeight
            ? previous
            : { width, height: nextHeight },
        );
      });
    });
    observer.observe(host);
    return () => {
      if (frame !== 0) window.cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const host = hostRef.current;
    if (!canvas || !host) return;
    if (size.width <= 0 || size.height <= 0) return;
    drawTopology(canvas, host, size.width, size.height, probe, activeToken);
  }, [probe, activeToken, size]);

  const residualCount = probe?.residual?.length ?? 0;
  const headCount = probe?.attention?.head_count ?? 0;
  const featureCount = probe?.features?.length ?? 0;
  const availableColumns = clamp(
    Math.floor((Math.max(1, size.width) - 214) / 26),
    3,
    MAX_DISPLAY_LAYERS,
  );
  const shownLayers = Math.min(residualCount, availableColumns);
  const empty = probe == null || residualCount === 0;

  return (
    <section className="lx-panel lx-topo" aria-label="神经元拓扑">
      <header className="lx-panel__head">
        <div className="lx-panel__heading">
          <h3 className="lx-panel__title">神经元拓扑</h3>
          <p className="lx-panel__sub">残差通路 · 注意力与 MLP 分支</p>
        </div>
        <dl className="lx-topo__stats">
          <div>
            <dt>层</dt>
            <dd className="num">{fmtInt(residualCount)}</dd>
          </div>
          <div>
            <dt>头</dt>
            <dd className="num">{fmtInt(headCount)}</dd>
          </div>
          <div>
            <dt>激活特征</dt>
            <dd className="num">{fmtInt(featureCount)}</dd>
          </div>
        </dl>
      </header>

      <div className="lx-panel__meta">
        <span className="lx-legend-key">
          <span className="lx-legend-key__dot lx-legend-key__dot--attn" aria-hidden="true" />
          注意力
        </span>
        <span className="lx-legend-key">
          <span className="lx-legend-key__dot lx-legend-key__dot--mlp" aria-hidden="true" />
          MLP
        </span>
        <span className="lx-legend-key">
          <span className="lx-legend-key__dot lx-legend-key__dot--feature" aria-hidden="true" />
          稀疏特征
        </span>
        <span className="lx-panel__note">
          {activeToken ? `高亮路径：${shorten(activeToken, 8)}` : "选中 Token 后高亮其路径"}
        </span>
        {residualCount > shownLayers ? (
          <span className="lx-panel__note">显示时聚合为 {shownLayers} 列</span>
        ) : null}
      </div>

      <div className="lx-topo__stage" ref={hostRef} style={{ height: `${height}px` }}>
        <canvas ref={canvasRef} className="lx-topo__canvas" role="img" aria-label="分层神经元拓扑示意" />
        {empty ? (
          <div className="lx-empty lx-topo__empty">
            <p className="lx-empty__title">{probe == null ? "透镜未启用" : "尚无拓扑数据"}</p>
            <p className="lx-empty__hint">
              {probe == null
                ? "开启替代透镜后，残差范数与稀疏特征会绘制为分层网络。"
                : "探针尚未返回残差分支范数；等待下一个探测。"}
            </p>
          </div>
        ) : null}
      </div>

      <footer className="lx-topo__foot">
        <p className="lx-panel__note">
          节点亮度取自 attention / MLP 分支的 L2 范数；特征为 feature-probe 的 top-k 神经元解码，不是 SAE。
        </p>
        <p className="lx-panel__note">仅由数据变化重绘，没有动画噪点；窄面板会自动聚合层列。</p>
      </footer>
    </section>
  );
}

export default NeuralTopology;
