import { Component, type ReactNode } from "react";

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error) {
    console.error("LLM-Xray UI error", error);
  }

  render() {
    if (this.state.error) {
      return (
        <main className="fatal-screen">
          <p className="fatal-kicker">界面异常</p>
          <h1>观测台的一部分没有完成渲染</h1>
          <p>
            这段数据可能缺少字段，或者回放快照来自旧版本。刷新后会重新连接后端；当前会话仍在本地
            <span className="mono"> data/sessions </span>中。
          </p>
          <pre>{this.state.error.message}</pre>
          <button type="button" onClick={() => window.location.reload()}>
            重新加载
          </button>
        </main>
      );
    }
    return this.props.children;
  }
}
