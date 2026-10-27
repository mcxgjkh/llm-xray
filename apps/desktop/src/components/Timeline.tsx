import { useEffect, useMemo, useRef } from "react";
import type { Probe, SessionMeta, StepToken } from "../types";
import { tokenLabel } from "../lib/format";
import { Icon } from "./Icon";
import { Button, IconButton } from "./Controls";

export function Timeline({
  steps,
  probes,
  activeStep,
  onSelectStep,
  session,
  exporting,
  onExport,
  replayMode,
  playing,
  onPlayingChange,
}: {
  steps: StepToken[];
  probes: Map<number, Probe>;
  activeStep: number | null;
  onSelectStep: (index: number | null) => void;
  session: SessionMeta | null;
  exporting: boolean;
  onExport: () => void;
  replayMode: boolean;
  playing: boolean;
  onPlayingChange: (playing: boolean) => void;
}) {
  const timerRef = useRef<number | null>(null);
  const last = steps.at(-1)?.index ?? null;
  const current = activeStep ?? last;
  const currentIndex = useMemo(
    () => (current === null ? -1 : steps.findIndex((step) => step.index === current)),
    [current, steps],
  );

  useEffect(() => {
    if (!playing) {
      if (timerRef.current !== null) window.clearInterval(timerRef.current);
      timerRef.current = null;
      return;
    }
    timerRef.current = window.setInterval(() => {
      const index = current === null ? -1 : steps.findIndex((step) => step.index === current);
      if (index < 0 || index >= steps.length - 1) {
        onPlayingChange(false);
        return;
      }
      onSelectStep(steps[index + 1].index);
    }, 280);
    return () => {
      if (timerRef.current !== null) window.clearInterval(timerRef.current);
      timerRef.current = null;
    };
  }, [current, onPlayingChange, onSelectStep, playing, steps]);

  const labelStep = currentIndex >= 0 ? steps[currentIndex] : null;

  return (
    <section className={`timeline ${replayMode ? "is-replay" : ""}`.trim()} aria-label="生成时间轴">
      <div className="timeline-head">
        <div className="timeline-title-wrap">
          <span className="timeline-title">时间轴</span>
          <span className={`timeline-mode ${replayMode ? "mode-replay" : "mode-live"}`.trim()}>
            {replayMode ? `回放 · 第 ${current !== null ? current + 1 : 0} 步` : "实时"}
          </span>
          <span className="timeline-session mono">{session?.id ?? "无会话"}</span>
        </div>
        <div className="timeline-actions">
          <IconButton
            label={playing ? "暂停回放" : "播放回放"}
            onClick={() => onPlayingChange(!playing)}
            disabled={steps.length < 2}
            active={playing}
          >
            <Icon name={playing ? "pause" : "play"} size={14} />
          </IconButton>
          <button className="timeline-jump" onClick={() => onSelectStep(null)} disabled={!replayMode}>
            回到实时
          </button>
          <Button size="sm" variant="outline" onClick={onExport} disabled={!session || exporting}>
            <Icon name="download" size={13} /> 导出 JSONL
          </Button>
        </div>
      </div>
      <div className="timeline-track-wrap">
        <div className="timeline-ticks" role="list">
          {steps.length === 0 ? <span className="timeline-empty">生成开始后，每一个 Token 会成为时间轴上的一个刻度。</span> : null}
          {steps.map((step) => {
            const active = step.index === current;
            const hasProbe = probes.has(step.index);
            const height = 8 + Math.min(26, Math.max(0, (step.entropy ?? 0.4) * 16));
            return (
              <button
                key={step.index}
                className={`timeline-tick ${active ? "is-active" : ""} ${hasProbe ? "has-probe" : ""}`.trim()}
                style={{ height }}
                title={`第 ${step.index + 1} 步 · ${tokenLabel(step.delta)} · 熵 ${step.entropy?.toFixed(2) ?? "—"}`}
                onClick={() => onSelectStep(step.index)}
                role="listitem"
                aria-label={`跳到第 ${step.index + 1} 步`}
              />
            );
          })}
        </div>
        <div className="timeline-readout">
          <span className="timeline-token">{labelStep ? tokenLabel(labelStep.delta) : "—"}</span>
          <span className="timeline-detail">
            {labelStep
              ? `第 ${labelStep.index + 1} 步 · 熵 ${labelStep.entropy?.toFixed(2) ?? "—"} · p ${labelStep.chosen_prob?.toFixed(3) ?? "—"}`
              : "尚未选择"}
          </span>
        </div>
      </div>
      <input
        className="timeline-range"
        type="range"
        min={0}
        max={Math.max(0, steps.length - 1)}
        step={1}
        value={Math.max(0, currentIndex)}
        disabled={!steps.length}
        aria-label="拖动时间轴"
        onChange={(event) => {
          const step = steps[Number(event.target.value)];
          if (step) onSelectStep(step.index);
        }}
        onPointerDown={() => onPlayingChange(false)}
      />
    </section>
  );
}
