import { useMemo } from "react";
import { formatPct } from "../lib/sampling";
import { tokenLabel } from "../lib/format";
import { resample } from "../lib/sampling";
import type { SamplingParams, StepToken } from "../types";
import { EmptyState, Panel } from "./Panel";

const WIDTH = 560;
const HEIGHT = 218;
const LEFT = 30;
const TOP = 24;
const STEP_GAP = 104;
const ROW_GAP = 42;

export function DecodingTree({
  steps,
  sampling,
  selectedStep,
  onSelectStep,
  windowSize = 5,
}: {
  steps: StepToken[];
  sampling: SamplingParams;
  selectedStep: number | null;
  onSelectStep: (index: number) => void;
  windowSize?: number;
}) {
  const nodes = useMemo(() => {
    const visible = steps.slice(-windowSize);
    return visible.map((step, column) => {
      const dist = resample(step.topk, sampling, steps.slice(0, step.index).map((item) => item.delta), step.delta);
      const candidates = dist.options.slice(0, 4);
      return {
        step,
        column,
        candidates,
        x: LEFT + column * STEP_GAP,
      };
    });
  }, [sampling, steps, windowSize]);

  return (
    <Panel
      title="解码树"
      kicker="候选展开 · 最近五步"
      aside={<span className="panel-note">只有主干被执行</span>}
      className="panel-tree"
      bodyClassName="tree-body"
    >
      {!nodes.length ? (
        <EmptyState title="尚未产生分叉" body="每个步骤的 top-k 候选会在这里展开。铜色主干是模型实际走过的 Token，灰色分支是未被选中的可能路径。" compact />
      ) : (
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          className="decode-tree"
          role="img"
          aria-label="最近生成步骤的解码候选树"
        >
          {nodes.map((node, columnIndex) => {
            const next = nodes[columnIndex + 1];
            return (
              <g key={node.step.index}>
                {columnIndex > 0 ? (
                  <line
                    x1={LEFT + (columnIndex - 1) * STEP_GAP}
                    y1={TOP + ROW_GAP * 1.5}
                    x2={node.x}
                    y2={TOP + ROW_GAP * 1.5}
                    stroke="var(--line)"
                    strokeWidth="1"
                  />
                ) : null}
                {node.candidates.map((candidate, rank) => {
                  const y = TOP + ROW_GAP * (rank + 0.9);
                  const chosen = candidate.token === node.step.delta;
                  const active = selectedStep === node.step.index;
                  const p = Math.max(0.02, candidate.prob);
                  const radius = 3 + Math.min(5, p * 14);
                  return (
                    <g
                      key={`${node.step.index}-${rank}-${candidate.token}`}
                      className="tree-node"
                      role="button"
                      tabIndex={0}
                      onClick={() => onSelectStep(node.step.index)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") onSelectStep(node.step.index);
                      }}
                    >
                      <circle
                        cx={node.x}
                        cy={y}
                        r={radius}
                        fill={chosen ? "var(--accent)" : "var(--raised-2)"}
                        stroke={chosen ? "var(--accent-soft)" : "var(--line-strong)"}
                        strokeWidth={active ? 1.6 : 1}
                      />
                      <text x={node.x + 11} y={y - 2} className={chosen ? "tree-label is-chosen" : "tree-label"}>
                        {truncate(tokenLabel(candidate.token), 10)}
                      </text>
                      <text x={node.x + 11} y={y + 11} className="tree-prob">
                        {formatPct(candidate.prob, 1)}
                      </text>
                      {next && chosen ? (
                        <line
                          x1={node.x}
                          y1={y}
                          x2={next.x}
                          y2={TOP + ROW_GAP * 1.5}
                          stroke="var(--accent-deep)"
                          strokeWidth="1.2"
                        />
                      ) : null}
                    </g>
                  );
                })}
                <text x={node.x} y={HEIGHT - 8} textAnchor="middle" className="tree-step">
                  {String(node.step.index + 1).padStart(3, "0")}
                </text>
              </g>
            );
          })}
        </svg>
      )}
    </Panel>
  );
}

function truncate(text: string, max: number) {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}
