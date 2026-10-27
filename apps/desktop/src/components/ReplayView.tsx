import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { clock, seconds, tokenLabel } from "../lib/format";
import type { Xray } from "../state/useXray";
import { Button, IconButton, Tag } from "./Controls";
import { Icon } from "./Icon";
import { EmptyState, Panel } from "./Panel";

export function ReplayView({ xray }: { xray: Xray }) {
  const [sessions, setSessions] = useState(xray.sessions);
  const [loading, setLoading] = useState<string | null>(null);
  const [localError, setLocalError] = useState("");

  const refresh = async () => {
    try {
      const result = await api.sessions();
      setSessions(result.sessions);
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : "无法读取历史会话");
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  const open = async (id: string) => {
    setLoading(id);
    setLocalError("");
    try {
      await xray.loadSavedSession(id);
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : "回放失败");
    } finally {
      setLoading(null);
    }
  };

  return (
    <div className="view view-replay">
      <div className="section-head">
        <div>
          <h1>时间回放</h1>
          <p>每个 Token 的候选、熵与透镜快照都按步保存。拖动底部时间轴，像回看一段录音一样检查第 N 步。</p>
        </div>
        <div className="section-head-meta">
          <Button variant="outline" size="sm" onClick={xray.exportSession} disabled={!xray.sessionId} type="button">
            <Icon name="download" size={13} /> 导出当前会话
          </Button>
          <IconButton label="刷新历史" onClick={refresh}>
            <Icon name="refresh" size={14} />
          </IconButton>
        </div>
      </div>

      {localError ? <p className="inline-error">{localError}</p> : null}

      <div className="replay-grid">
        <Panel
          title="当前会话步骤"
          kicker={xray.session ? `${xray.session.model} · ${xray.steps.length} 步` : "尚未开始"}
          aside={xray.replayMode ? <Tag tone="accent">回放中</Tag> : <Tag tone="muted">实时</Tag>}
          className="panel-steps"
        >
          {xray.steps.length === 0 ? (
            <EmptyState title="没有可回放的步骤" body="生成一段回答后，逐步数据会保留在当前会话中；完成的会话也会自动写入 data/sessions。" compact />
          ) : (
            <div className="step-table-wrap">
              <table className="step-table">
                <thead>
                  <tr>
                    <th scope="col">步</th>
                    <th scope="col">Token</th>
                    <th scope="col" className="num-col">熵</th>
                    <th scope="col" className="num-col">p</th>
                    <th scope="col">探测</th>
                  </tr>
                </thead>
                <tbody>
                  {xray.steps.slice(-120).map((step) => (
                    <tr
                      key={step.index}
                      className={xray.activeStepIndex === step.index ? "is-selected" : ""}
                      onClick={() => void xray.selectStep(step.index)}
                      tabIndex={0}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") void xray.selectStep(step.index);
                      }}
                    >
                      <td className="num">{String(step.index + 1).padStart(3, "0")}</td>
                      <td className="token-cell">{tokenLabel(step.delta)}</td>
                      <td className="num num-col">{step.entropy?.toFixed(2) ?? "—"}</td>
                      <td className="num num-col">{step.chosen_prob?.toFixed(3) ?? "—"}</td>
                      <td>{xray.probes.has(step.index) ? <span className="probe-mark">有</span> : <span className="faint">—</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {xray.session ? (
            <footer className="step-foot">
              <span>{xray.finishReason ? `结束：${xray.finishReason}` : xray.generating ? "生成中" : "已结束"}</span>
              <span className="num">{seconds(xray.elapsed)}</span>
              <span className="mono">{xray.session.id}</span>
            </footer>
          ) : null}
        </Panel>

        <Panel title="历史会话" kicker="data/sessions" aside={<span className="panel-note">JSONL</span>}>
          {sessions.length === 0 ? (
            <EmptyState title="还没有落盘会话" body="一次生成结束后，完整快照会写入本地 data/sessions，可在这里重新打开。" compact />
          ) : (
            <div className="session-list">
              {sessions.map((item) => (
                <button
                  key={item.id}
                  className="session-row"
                  onClick={() => void open(item.id)}
                  disabled={loading !== null}
                >
                  <span className="session-when">
                    <span className="num">{clock(item.created_at)}</span>
                    <span className="faint">{item.mode === "demo" ? "演示" : "LM Studio"}</span>
                  </span>
                  <span className="session-preview">
                    <span className="session-model">{item.model}</span>
                    <span className="session-text">{item.preview || "无预览"}</span>
                  </span>
                  <span className="session-meta num">{item.tokens} 步</span>
                  <span className="session-open">{loading === item.id ? "读取中" : "打开"}</span>
                </button>
              ))}
            </div>
          )}
          <p className="panel-copy faint">
            回放只读取本地快照，不重新推理模型。导出文件包含每一步的 top-k、注意力、透镜与残差字段，适合离线分析。
          </p>
        </Panel>
      </div>
    </div>
  );
}
