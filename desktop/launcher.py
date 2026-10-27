"""LLM-Xray desktop launcher.

Starts the local FastAPI server in a child process, waits for its health
endpoint, then opens a native window with Edge WebView2 through pywebview.
No bundled Chromium: the shell is the platform webview plus Python.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
DIST = ROOT / "apps" / "desktop" / "dist"


class DesktopApi:
    """Small native bridge exposed to the web UI as ``window.pywebview.api``.

    WebView2 blocks ordinary browser downloads in some configurations. The UI
    calls this method to open a native save dialog and write the JSONL export
    directly. A path may be supplied for automated tests.
    """

    def save_text_file(self, filename: str, content: str, path: str | None = None) -> str | None:
        target = Path(path) if path else None
        if target is None:
            try:
                import webview  # type: ignore
            except ImportError:
                return None
            window = webview.windows[0] if webview.windows else None
            if window is None:
                return None
            result = window.create_file_dialog(
                webview.SAVE_DIALOG,
                save_filename=filename,
                file_types=("JSON Lines (*.jsonl)", "All files (*.*)"),
            )
            if not result:
                return None
            target = Path(result if isinstance(result, str) else result[0])
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError:
            return None
        return str(target)


def health(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{url}/api/health", timeout=1.2) as response:
            if response.status != 200:
                return None
            import json

            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


def wait_for_server(url: str, timeout: float = 40.0) -> dict:
    deadline = time.time() + timeout
    last: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=1.5) as response:
                if response.status == 200:
                    import json

                    return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - startup polling
            last = exc
        time.sleep(0.25)
    raise RuntimeError(f"后端未能在 {timeout:.0f} 秒内启动: {last}")


def spawn_server(port: int, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("LLM_XRAY_HOST", "127.0.0.1")
    env.setdefault("LLM_XRAY_PORT", str(port))
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "xray.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    log = open(log_path, "ab", buffering=0)  # noqa: SIM115 - lifetime equals child process
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.Popen(
        command,
        cwd=str(ROOT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-Xray desktop launcher")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-window", action="store_true", help="只启动后端，便于浏览器调试")
    parser.add_argument("--browser", action="store_true", help="用系统浏览器打开，而不是原生窗口")
    parser.add_argument("--reload", action="store_true", help="开发时透传给 uvicorn 的 reload（仅后端）")
    args = parser.parse_args()

    if not DIST.exists() and not args.no_window and not args.browser:
        print("未找到前端构建产物：apps/desktop/dist")
        print("请先执行：npm --prefix apps/desktop install，然后 npm --prefix apps/desktop run build")
        print("开发模式可使用 scripts/dev.ps1，或执行 python desktop/launcher.py --browser 并先启动 Vite。")
        return 2

    url = f"http://127.0.0.1:{args.port}"
    existing = health(url)
    child: subprocess.Popen | None = None
    if existing:
        print(f"检测到已有后端 {url}，直接复用。")
    else:
        log_path = ROOT / "data" / "server.log"
        child = spawn_server(args.port, log_path)
        print(f"后端启动中，日志：{log_path}")

    try:
        info = wait_for_server(url)
        print(f"LLM-Xray {info.get('version', 'unknown')} 已就绪：{url}")
        if args.no_window:
            print("按 Ctrl+C 停止。")
            while True:
                time.sleep(1)
        if args.browser:
            webbrowser.open(url)
            return 0

        try:
            import webview  # type: ignore
        except ImportError:
            print("未安装 pywebview，退回系统浏览器。安装：pip install pywebview")
            webbrowser.open(url)
            return 0

        window = webview.create_window(
            "LLM-Xray · 本地模型观测台",
            url,
            width=1500,
            height=940,
            min_size=(1120, 700),
            background_color="#17130f",
            text_select=True,
            js_api=DesktopApi(),
        )
        webview.start(debug=False, private_mode=False)
        del window
    except KeyboardInterrupt:
        return 0
    except Exception as exc:  # noqa: BLE001 - desktop boundary
        print(f"启动失败：{exc}")
        return 1
    finally:
        if child and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=4)
            except subprocess.TimeoutExpired:
                child.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
