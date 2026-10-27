"""Performance telemetry (CPU / RAM / optional NVIDIA GPU).

Sampling runs in one background asyncio task every 1.5 s.  ``nvidia-smi`` is a
short-lived subprocess with a 1.2 s timeout, cached result and failure backoff:
a missing driver or a busy machine must never turn into a per-token subprocess
storm (SPEC §10).  Everything degrades to ``None`` instead of raising.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from typing import Any

try:  # psutil is a declared dependency, but telemetry must never break the app.
    import psutil
except Exception:  # pragma: no cover - extremely defensive
    psutil = None  # type: ignore[assignment]

SAMPLE_INTERVAL = 1.5
NVIDIA_TIMEOUT = 1.2
GPU_CACHE_SECONDS = 1.5
GPU_FAILURE_BACKOFF = 15.0
GPU_DISABLE_AFTER_FAILURES = 5


class MetricsCollector:
    """Non-blocking system metrics used by the metrics event and REST health."""

    def __init__(
        self,
        *,
        interval: float = SAMPLE_INTERVAL,
        gpu_timeout: float = NVIDIA_TIMEOUT,
    ) -> None:
        self.interval = float(interval)
        self.gpu_timeout = float(gpu_timeout)
        self.gpu_available = shutil.which("nvidia-smi") is not None
        self._task: asyncio.Task | None = None
        self._cpu: float | None = None
        self._ram: float | None = None
        self._ram_total: float | None = None
        self._gpu: int | None = None
        self._vram_used: float | None = None
        self._vram_total: float | None = None
        self._gpu_next_at = 0.0
        self._gpu_failures = 0
        # generation counters (reset per session)
        self.tokens = 0
        self.tps: float | None = None
        self.ttft_ms: float | None = None
        self.context = 0
        self.lens_lag_ms: float | None = None

    # -- background loop ------------------------------------------------
    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        if psutil is not None:
            try:
                psutil.cpu_percent(interval=None)  # prime the counter
            except Exception:  # pragma: no cover - defensive
                pass
        self._task = asyncio.create_task(self._loop(), name="xray-metrics")

    async def _loop(self) -> None:
        try:
            while True:
                self.sample_system()
                await self._sample_gpu()
                await asyncio.sleep(self.interval)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - never kill the sampler
            await asyncio.sleep(self.interval)

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: B014 - defensive
                pass

    # -- sampling -------------------------------------------------------
    def sample_system(self) -> None:
        if psutil is None:
            return
        try:
            self._cpu = float(psutil.cpu_percent(interval=None))
        except Exception:  # pragma: no cover - defensive
            self._cpu = None
        try:
            memory = psutil.virtual_memory()
            self._ram = round((memory.total - memory.available) / 1e9, 2)
            self._ram_total = round(memory.total / 1e9, 2)
        except Exception:  # pragma: no cover - defensive
            self._ram = None
            self._ram_total = None

    async def _sample_gpu(self) -> None:
        if not self.gpu_available:
            return
        now = time.monotonic()
        if now < self._gpu_next_at:
            return
        self._gpu_next_at = now + GPU_CACHE_SECONDS
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=self.gpu_timeout)
            first_line = stdout.decode("utf-8", errors="replace").strip().splitlines()[0]
            parts = [part.strip() for part in first_line.split(",")]
            if len(parts) < 3:
                raise ValueError("unexpected nvidia-smi output")
            self._gpu = int(float(parts[0]))
            self._vram_used = round(float(parts[1]) / 1024.0, 2)
            self._vram_total = round(float(parts[2]) / 1024.0, 2)
            self._gpu_failures = 0
        except (asyncio.TimeoutError, ValueError, IndexError, OSError, TypeError):
            self._gpu = None
            self._vram_used = None
            self._vram_total = None
            self._gpu_failures += 1
            self._gpu_next_at = time.monotonic() + GPU_FAILURE_BACKOFF
            if self._gpu_failures >= GPU_DISABLE_AFTER_FAILURES:
                self.gpu_available = False
        except Exception:  # pragma: no cover - defensive
            self._gpu = None
        finally:
            if process is not None:
                try:
                    if process.returncode is None:
                        process.kill()
                        await process.wait()
                except Exception:  # pragma: no cover - defensive
                    pass

    # -- generation counters -------------------------------------------
    def reset_generation(self) -> None:
        self.tokens = 0
        self.tps = None
        self.ttft_ms = None
        self.lens_lag_ms = None

    def note_token(self) -> None:
        self.tokens += 1

    def update_generation(self, generated: int, elapsed_s: float, ttft_ms: float | None = None) -> None:
        self.tokens = int(generated)
        self.tps = round(generated / elapsed_s, 2) if elapsed_s > 0 else None
        if ttft_ms is not None:
            self.ttft_ms = round(float(ttft_ms), 1)

    def set_context(self, tokens: int) -> None:
        self.context = max(0, int(tokens))

    # -- read -----------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        return {
            "cpu": round(self._cpu, 1) if self._cpu is not None else None,
            "ram": self._ram,
            "ram_total": self._ram_total,
            "gpu": self._gpu,
            "vram_used": self._vram_used,
            "vram_total": self._vram_total,
            "tps": round(self.tps, 2) if self.tps is not None else None,
            "ttft_ms": round(self.ttft_ms, 1) if self.ttft_ms is not None else None,
            "tokens": self.tokens,
            "context": self.context,
            "lens_lag_ms": round(self.lens_lag_ms, 1) if self.lens_lag_ms is not None else None,
        }

    @property
    def gpu_util(self) -> int | None:
        return self._gpu
