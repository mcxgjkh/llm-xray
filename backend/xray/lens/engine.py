"""LensEngine: optional surrogate causal model for internal telemetry.

The lens channel is a *surrogate*: a small local HF model observes the same text
as the real LM Studio model and exposes hidden states, attention, Tuned Lens,
residual contributions, sparse MLP features and a UMAP projection.  It is never
presented as the real model's internals (see docs/SPEC.md section 8).

Design constraints implemented here:
* Loading never downloads: ``local_files_only=True``; a missing model is a
  graceful error the backend can ignore.
* CPU friendly: 2 torch threads, optional below-normal process priority, eager
  attention, float32, capped context (256 by default).
* Incremental: one forward per text delta with a KV cache.  If the delta's
  tokenizer boundary changes the prefix, the engine rebuilds the cache and
  reports the newest tokens instead of guessing.
* Never raises into the backend: every failure becomes a ``status()`` message.
* Probe construction is split into small modules (attention, tuned_lens,
  residual, features, projection) so each piece is testable in isolation.
"""

from __future__ import annotations

import gc
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .common import (
    compare_model_names,
    downsample_vector,
    find_decoder_layers,
    find_final_norm,
    find_lm_head,
    find_layer_branches,
    now_ms,
    read_config_int,
    round_floats,
    token_text,
)
from .projection import Projector

try:  # sampling is pure Python; importing it here keeps lens and backend in sync.
    from ..sampling import SamplingParams
except Exception:  # pragma: no cover - only if sampling.py is missing/corrupt
    SamplingParams = None  # type: ignore[assignment,misc]

try:  # psutil is a hard backend dependency, but stay import tolerant.
    import psutil
except Exception:  # pragma: no cover - optional niceness only
    psutil = None

DEFAULT_MODEL = os.environ.get("XRAY_LENS_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DEFAULT_MAX_CONTEXT = int(os.environ.get("XRAY_LENS_MAX_CONTEXT", "256"))
DEFAULT_THREADS = int(os.environ.get("XRAY_LENS_THREADS", "2"))
#: Decode sparse features every N-th token only (1 = every token).  Set
#: XRAY_LENS_FEATURES_EVERY=2/3 when the 0.5B CPU budget is tight.
FEATURES_EVERY = max(1, int(os.environ.get("XRAY_LENS_FEATURES_EVERY", "1")))
#: Populated by scripts/fetch_lens_model.ps1 (override in tests / custom installs).
MODELS_DIR_ENV = "LLM_XRAY_MODELS_DIR"
PROJECTION_EVERY = 8
#: Raw-distribution candidates returned per native token (probability cloud).
NATIVE_TOPK = max(4, int(os.environ.get("XRAY_NATIVE_TOPK", "20")))
#: Repetition penalty window for native sampling (same as SPEC §7 / sampling.py).
REPETITION_WINDOW = 64

_torch_module = None


def _get_torch():
    """Lazily import torch so the backend starts without optional deps."""
    global _torch_module
    if _torch_module is None:
        try:
            import torch as _torch

            _torch_module = _torch
        except Exception:
            return None
    return _torch_module


def _encoding_ids(encoded: Any) -> list[int]:
    """Extract a flat int id list from a tokenizer call result."""
    if isinstance(encoded, dict):
        ids = encoded.get("input_ids", [])
    else:
        ids = getattr(encoded, "input_ids", encoded)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    result: list[int] = []
    for value in ids or []:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            continue
    return result


def _tensor_to_numpy(value: Any) -> np.ndarray:
    try:
        return value.detach().float().cpu().numpy()
    except Exception:
        return np.asarray(value, dtype=np.float32)


def _coerce_sampling_params(params: Any) -> Any:
    """Accept ``SamplingParams``, a dict, or ``None`` (defaults)."""
    if SamplingParams is not None:
        if isinstance(params, SamplingParams):
            return params
        if isinstance(params, dict):
            return SamplingParams.from_dict(params)
        return SamplingParams()
    if isinstance(params, dict):
        fallback = _FallbackSamplingParams()
        for key, value in params.items():
            if hasattr(fallback, key) and value is not None:
                setattr(fallback, key, value)
        return fallback
    return params if params is not None else _FallbackSamplingParams()


class _FallbackSamplingParams:
    """Last-resort defaults if ``xray.sampling`` cannot be imported."""

    temperature = 0.7
    top_p = 0.95
    top_k = 40
    repetition_penalty = 1.05
    max_tokens = 512
    seed = None


def repo_root_path() -> Path:
    """Repository root (backend/xray/lens/engine.py -> repo root)."""
    return Path(__file__).resolve().parents[3]


def is_model_dir(path: Path) -> bool:
    """A directory that ``from_pretrained`` can load offline."""
    try:
        candidate = Path(path)
    except TypeError:
        return False
    if candidate.is_file():
        return True
    if not candidate.is_dir():
        return False
    if (candidate / "config.json").is_file():
        return True
    for pattern in ("*.safetensors", "pytorch_model.bin"):
        if any(candidate.glob(pattern)):
            return True
    return False


def has_weight_files(path: Path) -> bool:
    """True when a local model directory actually contains weights."""
    try:
        candidate = Path(path)
    except TypeError:
        return False
    if candidate.is_file():
        return True
    if not candidate.is_dir():
        return False
    for pattern in ("*.safetensors", "pytorch_model*.bin", "*.pt", "*.pth"):
        try:
            if any(candidate.glob(pattern)):
                return True
        except OSError:
            continue
    return False


def sanitized_model_dir_name(model_name: str) -> str:
    """Match scripts/fetch_lens_model.ps1: 'org/model' -> 'org__model'."""
    return re.sub(r"[/\\:]", "__", str(model_name))


def project_model_candidates(model_name: str) -> list[Path]:
    """Candidate local paths used by the fetch script / manual installs."""
    root = repo_root_path()
    models_dir = Path(os.environ.get(MODELS_DIR_ENV) or (root / "models"))
    lens_dir = models_dir / "lens"
    safe = sanitized_model_dir_name(model_name)
    basename = Path(str(model_name)).name
    candidates: list[Path] = []
    for base in (Path.cwd(), root):
        candidates.append(base / model_name)
    candidates.extend(
        [
            lens_dir / safe,
            models_dir / safe,
            lens_dir / basename,
            models_dir / basename,
        ]
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def resolve_model_source(model_name: str) -> tuple[str | None, list[str]]:
    """Resolve a requested model to a local, offline-loadable source.

    Order: explicit existing path -> project ``models/lens/<sanitized>`` (as
    written by scripts/fetch_lens_model.ps1) -> Hugging Face cache (checked by
    the caller via ``local_files_only=True``).  Returns ``(source, searched)``;
    ``source`` is ``None`` when nothing local was found, and ``model_name``
    itself is returned when the HF cache already holds it.
    """
    raw = str(model_name).strip()
    searched: list[str] = []
    expanded = Path(os.path.expandvars(os.path.expanduser(raw)))
    try:
        if expanded.exists():
            if is_model_dir(expanded):
                return str(expanded.resolve()), searched
            searched.append(str(expanded))
    except OSError:
        searched.append(str(expanded))
    for candidate in project_model_candidates(raw):
        text = str(candidate)
        if text not in searched:
            searched.append(text)
        try:
            if is_model_dir(candidate):
                return str(candidate), searched
        except OSError:
            continue
    # Hugging Face cache check (read-only; never downloads).
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(raw, "config.json")
        if isinstance(cached, str) and Path(cached).is_file():
            return raw, searched
    except Exception:
        pass
    return None, searched


class LensEngine:
    """Optional surrogate lens; safe to construct before torch/models exist."""

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        model_name: str | None = None,
        max_context: int = DEFAULT_MAX_CONTEXT,
        cache_dir: str | None = None,
        use_umap: bool = True,
    ) -> None:
        self._lock = threading.RLock()
        self._status = "unloaded"
        self._message = "透镜未加载"
        self._model: Any = None
        self._tokenizer: Any = None
        self._model_name: str | None = None
        self._runtime_model: str | None = None
        self._same_weights = False
        self._device = "cpu"
        self._threads = DEFAULT_THREADS
        self._max_context = max(8, int(max_context))
        self._capacity = self._max_context
        self._cache_dir = cache_dir
        self._use_umap = bool(use_umap)

        self._layers: list = []
        self._n_layers = 0
        self._n_heads = 1
        self._n_kv_heads = 1
        self._head_dim = 0
        self._hidden_size = 0
        self._vocab_size = 0
        self._elem_bytes = 4
        self._focus_layers: list[int] = []
        self._final_norm: Any = None
        self._lm_head: Any = None
        self._lm_head_weight: np.ndarray | None = None

        # Incremental context state (token ids/tokens/kinds held by the cache).
        self._past: Any = None
        self._ctx_ids: list[int] = []
        self._ctx_tokens: list[str] = []
        self._ctx_kinds: list[str] = []
        self._all_text = ""
        self._seen_ids: list[int] = []
        self._prompt_tokens = 0
        self._point_counter = 0

        # Native generation state: prefill seeds ``_next_logits`` and
        # ``_pending_probe``; each sample_next() forwards exactly one token and
        # returns the probe of that same forward (no second pass).
        self._native_mode = False
        self._next_logits: Any = None
        self._pending_probe: dict | None = None
        self._pending_forward_id: int | None = None
        self._gen_ids: list[int] = []
        self._gen_decoded = ""
        self._sample_generator: Any = None
        self._sample_seed: int | None = None

        self._projector = Projector(use_umap=self._use_umap)
        self._tuned: Any = None
        self._residual: Any = None
        self._features: Any = None
        self._calibration_thread: threading.Thread | None = None
        self._calibrating = False
        self._last_latency_ms: float | None = None
        self._latency_samples: list[float] = []

        if model is not None and tokenizer is not None:
            self.attach_model(
                model,
                tokenizer,
                model_name=model_name or "attached-local-model",
                schedule_calibration=False,
            )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict:
        with self._lock:
            calibrated = 0
            if self._tuned is not None:
                try:
                    calibrated = int(self._tuned.calibrated_layers)
                except Exception:
                    calibrated = 0
            return {
                "status": self._status,
                "model": self._model_name,
                "device": self._device if self._status == "ready" else None,
                "surrogate": not bool(self._native_mode),
                "same_weights": bool(self._same_weights),
                "calibrated_layers": calibrated,
                "message": self._message or "",
                "threads": int(self._threads),
            }

    def is_ready(self) -> bool:
        with self._lock:
            return self._status == "ready" and self._model is not None

    @property
    def latency_ms(self) -> float | None:
        with self._lock:
            return self._last_latency_ms

    def latency_stats(self) -> dict:
        with self._lock:
            samples = list(self._latency_samples)
        if not samples:
            return {"last": None, "avg": None, "count": 0}
        return {
            "last": round(samples[-1], 2),
            "avg": round(sum(samples) / len(samples), 2),
            "count": len(samples),
        }

    def set_runtime_model(self, name: str | None) -> None:
        """Record the LM Studio model so ``same_weights`` can be computed."""
        with self._lock:
            self._runtime_model = name
            self._same_weights = compare_model_names(name, self._model_name)

    def note_runtime_model(self, name: str | None) -> None:
        self.set_runtime_model(name)

    def resolve_model_path(self, model_name: str | None = None) -> str | None:
        """Return the local source ``load()`` would use, or ``None``.

        Useful for ``GET /api/lens/status`` diagnostics without touching the
        network; exposes the same search order as :meth:`load`.
        """
        requested = (
            model_name or os.environ.get("LLM_XRAY_LENS_MODEL") or DEFAULT_MODEL
        ).strip()
        source, _searched = resolve_model_source(requested)
        return source

    # ------------------------------------------------------------------
    # Load / unload
    # ------------------------------------------------------------------

    def load(
        self,
        model_name: str | None = None,
        device: str = "cpu",
        low_priority: bool = True,
        schedule_calibration: bool = True,
    ) -> dict:
        """Blocking load of a local HF causal LM.  Never downloads weights.

        Resolution order: explicit existing path -> project
        ``models/lens/<sanitized model>`` (fetch script destination) -> HF cache
        (``local_files_only=True``) -> clear error naming the fetch script.
        """
        requested = (
            model_name
            or os.environ.get("LLM_XRAY_LENS_MODEL")
            or DEFAULT_MODEL
        ).strip()
        source, searched = resolve_model_source(requested)
        if source is None:
            searched_text = "、".join(searched[:6]) if searched else "（无）"
            return self._fail(
                "未找到本地透镜模型（local_files_only，不会自动下载）。"
                f"已尝试：{searched_text}。请运行 "
                "powershell -ExecutionPolicy Bypass -File scripts/fetch_lens_model.ps1 "
                "（下载到 models/lens/），或设置 LLM_XRAY_LENS_MODEL=<本地模型目录>；"
                "也可以使用 demo 模式。"
            )
        name = requested
        resolved_note = ""
        try:
            if str(Path(source)) != str(Path(requested)):
                resolved_note = f"来源 {source}"
            if Path(source).is_dir() and not has_weight_files(Path(source)):
                return self._fail(
                    "本地透镜模型目录缺少权重文件（可能仍在下载，或下载未完成）："
                    f"{source}。请等待 "
                    "powershell -ExecutionPolicy Bypass -File scripts/fetch_lens_model.ps1 "
                    "完成；也可设置 LLM_XRAY_LENS_MODEL=<本地模型目录> 或使用 demo 模式。"
                )
        except OSError:
            resolved_note = ""
        with self._lock:
            if self._status == "ready":
                return self.status()
            self._status = "loading"
            self._message = f"正在加载透镜模型 {name}（local_files_only）"

        torch = _get_torch()
        if torch is None:
            return self._fail("PyTorch 未安装：透镜通道不可用；请使用 demo 模式或安装 backend/requirements-lens.txt")
        try:
            threads = self._configure_torch(torch)
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - depends on environment
            return self._fail(f"透镜依赖不可用：{exc}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                source, local_files_only=True, trust_remote_code=False
            )
            config = AutoConfig.from_pretrained(
                source, local_files_only=True, trust_remote_code=False
            )
            # transformers 5.x: set eager before construction so output_attentions
            # actually returns matrices (SDPA silently drops them).
            try:
                config._attn_implementation = "eager"
            except Exception:
                pass
            model = AutoModelForCausalLM.from_pretrained(
                source,
                config=config,
                local_files_only=True,
                trust_remote_code=False,
                attn_implementation="eager",
                dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
        except Exception as exc:
            return self._fail(
                "本地透镜模型加载失败（local_files_only，不会自动下载）："
                f"{exc}. 来源：{source}。请运行 "
                "powershell -ExecutionPolicy Bypass -File scripts/fetch_lens_model.ps1，"
                "或设置 LLM_XRAY_LENS_MODEL=<本地模型目录>；也可以使用 demo 模式。"
            )

        try:
            model.eval()
            try:
                model.config._attn_implementation = "eager"
            except Exception:
                pass
            target_device = "cpu"
            note = ""
            if str(device).lower().startswith("cuda"):
                target_device, note = self._maybe_cuda(torch, model, device)
            model.to(target_device)
            with self._lock:
                self._threads = threads
            self.attach_model(
                model,
                tokenizer,
                model_name=name,
                device=target_device,
                low_priority=low_priority,
                schedule_calibration=schedule_calibration,
                extra_message="；".join(part for part in (note, resolved_note) if part),
            )
            return self.status()
        except Exception as exc:
            return self._fail(f"透镜初始化失败：{exc}")

    def _maybe_cuda(self, torch: Any, model: Any, requested: str) -> tuple[str, str]:
        """Use CUDA only when explicitly requested and memory looks sufficient."""
        try:
            if not torch.cuda.is_available():
                return "cpu", "请求了 CUDA 但不可用，已回退 CPU"
            free_bytes, _total = torch.cuda.mem_get_info()
            params = sum(int(p.numel()) for p in model.parameters())
            estimate = params * 4 * 1.3 + 256 * 1024 * 1024
            if free_bytes < estimate:
                return "cpu", "CUDA 显存不足，已回退 CPU"
            return str(requested), ""
        except Exception:
            return "cpu", "CUDA 设备检查失败，已回退 CPU"

    def _configure_torch(self, torch: Any) -> int:
        try:
            torch.set_num_threads(max(1, DEFAULT_THREADS))
        except Exception:
            pass
        try:
            torch.set_grad_enabled(False)
        except Exception:
            pass
        try:
            return int(torch.get_num_threads())
        except Exception:
            return DEFAULT_THREADS

    def _lower_priority(self) -> bool:
        if psutil is None:
            return False
        try:
            process = psutil.Process()
            if os.name == "nt":
                process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            else:
                process.nice(10)
            return True
        except Exception:
            return False

    def attach_model(
        self,
        model: Any,
        tokenizer: Any,
        model_name: str | None = None,
        device: str = "cpu",
        low_priority: bool = False,
        schedule_calibration: bool = False,
        extra_message: str = "",
    ) -> dict:
        """Attach an already-loaded model/tokenizer.

        Public mainly for tests and for callers that manage HF weights
        themselves; ``load()`` uses the same path after ``from_pretrained``.
        """
        with self._lock:
            self._teardown_locked()
            self._model = model
            self._tokenizer = tokenizer
            self._model_name = model_name or getattr(model, "name_or_path", None) or "local-model"
            self._device = str(device or "cpu")
            config = getattr(model, "config", None)
            self._layers = find_decoder_layers(model)
            self._n_layers = len(self._layers) or read_config_int(config, "num_hidden_layers", 0)
            self._hidden_size = read_config_int(config, "hidden_size", 0)
            self._n_heads = max(1, read_config_int(config, "num_attention_heads", 1))
            self._n_kv_heads = max(
                1, read_config_int(config, "num_key_value_heads", self._n_heads)
            )
            head_dim = read_config_int(config, "head_dim", 0)
            if head_dim <= 0 and self._hidden_size:
                head_dim = max(1, self._hidden_size // self._n_heads)
            self._head_dim = int(head_dim)
            self._final_norm = find_final_norm(model)
            self._lm_head = find_lm_head(model)
            self._vocab_size = read_config_int(config, "vocab_size", 0)
            if self._vocab_size <= 0:
                try:
                    self._vocab_size = int(getattr(self._lm_head, "out_features", 0))
                except Exception:
                    self._vocab_size = 0
            self._lm_head_weight = self._read_lm_head_weight()
            self._elem_bytes = self._read_element_bytes(model)
            max_pos = read_config_int(config, "max_position_embeddings", 8192) or 8192
            self._capacity = max(8, min(self._max_context, int(max_pos)))
            self._projector = Projector(
                max_points=256, fit_every=PROJECTION_EVERY, use_umap=self._use_umap
            )
            self._focus_layers = self._compute_focus_layers()
            self._threads = int(self._configure_torch(_get_torch())) if _get_torch() else DEFAULT_THREADS
            self._reset_context_locked()
            self._native_mode = False
            self._same_weights = compare_model_names(self._runtime_model, self._model_name)

            # Optional heavy submodules, imported lazily.
            from .residual import ResidualCapture

            self._residual = ResidualCapture(model)
            self._residual.register()
            try:
                from .tuned_lens import TunedLens

                self._tuned = TunedLens(
                    model_name=self._model_name,
                    n_layers=max(1, self._n_layers),
                    hidden_size=self._hidden_size or self._head_dim * self._n_heads,
                    final_norm=self._final_norm,
                    lm_head=self._lm_head,
                    tokenizer=tokenizer,
                    device=self._device,
                    cache_dir=self._cache_dir,
                )
                cached = self._tuned.load_cache()
            except Exception as exc:
                self._tuned = None
                cached = False
                self._message = f"Tuned Lens 不可用：{exc}"
            try:
                from .features import FeatureProbe

                self._features = FeatureProbe(model)
                self._features.register()
                if self._lm_head is not None:
                    self._features.set_lm_head(self._lm_head)
            except Exception:
                self._features = None

            self._status = "ready"
            if self._tuned is None:
                self._message = "透镜已就绪（Logit Lens；Tuned Lens 不可用）"
            elif cached:
                self._message = f"透镜已就绪，Tuned Lens 缓存 {self._tuned.calibrated_layers} 层"
            else:
                self._message = "透镜已就绪，Tuned Lens 后台校准中"
            if extra_message:
                self._message = f"{self._message}；{extra_message}"
            if low_priority and self._lower_priority():
                self._message += "；进程低优先级"
            if schedule_calibration and self._tuned is not None and not cached:
                self._start_calibration_locked()
            return self.status()

    def _read_lm_head_weight(self) -> np.ndarray | None:
        if self._lm_head is None:
            return None
        try:
            weight = getattr(self._lm_head, "weight", None)
            if weight is None:
                return None
            return _tensor_to_numpy(weight).astype(np.float32)
        except Exception:
            return None

    @staticmethod
    def _read_element_bytes(model: Any) -> int:
        try:
            parameter = next(model.parameters())
            dtype = getattr(parameter, "dtype", None)
            size = getattr(dtype, "itemsize", None)
            if size:
                return int(size)
        except Exception:
            pass
        return 4

    def _compute_focus_layers(self) -> list[int]:
        n = max(1, int(self._n_layers))
        if n == 1:
            return [0]
        focus = {min(12, n - 1), n - 1}
        if len(focus) < 2:
            # Two-layer test models: pair the last layer with the one before it.
            focus.add(max(0, min(12, n - 1) - 1))
        return sorted(focus)

    def unload(self) -> dict:
        with self._lock:
            self._teardown_locked()
            self._projector.reset()
            self._reset_context_locked()
            self._model = None
            self._tokenizer = None
            self._tuned = None
            self._status = "unloaded"
            self._message = "透镜已卸载"
            self._same_weights = False
            self._native_mode = False
        try:
            torch = _get_torch()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        return self.status()

    def _teardown_locked(self) -> None:
        for module in (self._residual, self._features):
            if module is not None:
                try:
                    module.close()
                except Exception:
                    pass
        self._residual = None
        self._features = None
        self._tuned = None

    def _fail(self, message: str) -> dict:
        with self._lock:
            self._status = "error"
            self._message = str(message)
        return self.status()

    def _reset_context_locked(self) -> None:
        self._past = None
        self._ctx_ids = []
        self._ctx_tokens = []
        self._ctx_kinds = []
        self._all_text = ""
        self._seen_ids = []
        self._prompt_tokens = 0
        self._point_counter = 0
        self._next_logits = None
        self._pending_probe = None
        self._pending_forward_id = None
        self._gen_ids = []
        self._gen_decoded = ""
        self._sample_generator = None
        self._sample_seed = None

    # ------------------------------------------------------------------
    # Calibration (background)
    # ------------------------------------------------------------------

    def _start_calibration_locked(self) -> None:
        if self._calibrating or self._tuned is None or self._model is None:
            return
        self._calibrating = True
        model = self._model
        tuned = self._tuned
        device = self._device

        def worker() -> None:
            try:
                from .tuned_lens import collect_calibration_corpus

                torch = _get_torch()
                if torch is not None:
                    try:
                        torch.set_num_threads(max(1, DEFAULT_THREADS))
                    except Exception:
                        pass
                corpus = collect_calibration_corpus()
                count = tuned.calibrate(model, corpus)
                with self._lock:
                    if count > 0:
                        self._message = f"Tuned Lens 校准完成：{count} 层，{tuned.corpus_size} 段语料"
                    else:
                        self._message = (
                            "Tuned Lens 校准失败，继续使用 Logit Lens"
                            + (f"：{tuned.last_error}" if tuned.last_error else "")
                        )
            except Exception as exc:
                with self._lock:
                    self._message = f"Tuned Lens 后台校准异常：{exc}"
            finally:
                with self._lock:
                    self._calibrating = False

        thread = threading.Thread(
            target=worker, name="xray-tuned-lens-calibration", daemon=True
        )
        self._calibration_thread = thread
        thread.start()

    def _calibrate_now(self, corpus: Sequence[str] | None = None, max_total_tokens: int = 640) -> int:
        """Synchronous calibration helper (tests / CLI)."""
        with self._lock:
            if self._status != "ready" or self._tuned is None or self._model is None:
                return 0
            tuned = self._tuned
            model = self._model
        if corpus is None:
            from .tuned_lens import collect_calibration_corpus

            corpus = collect_calibration_corpus()
        return int(tuned.calibrate(model, corpus, max_total_tokens=max_total_tokens))

    # ------------------------------------------------------------------
    # Reset / observe
    # ------------------------------------------------------------------

    def reset(self, prompt_text: str, system: str = "") -> dict:
        """Seed the KV cache and token alignment with a prompt."""
        with self._lock:
            if self._status != "ready" or self._model is None:
                return {"ok": False, "message": self._message or "透镜未就绪"}
            torch = _get_torch()
            if torch is None:
                return {"ok": False, "message": "PyTorch 不可用"}
            try:
                text = str(prompt_text or "")
                if system:
                    system_text = str(system).strip()
                    if system_text:
                        text = f"{system_text}\n\n{text}" if text else system_text
                self._reset_context_locked()
                self._native_mode = False
                self._all_text = text
                full_ids = self._encode(text)
                if not full_ids:
                    full_ids = [self._fallback_token_id()]
                self._seen_ids = list(full_ids)
                self._prompt_tokens = len(full_ids)
                keep = full_ids[-self._capacity :]
                offset = len(full_ids) - len(keep)
                self._projector.reset()
                self._point_counter = 0
                with torch.inference_mode():
                    input_ids = torch.tensor([keep], dtype=torch.long, device=self._device)
                    out = self._forward_raw(input_ids, past=None, capture=False)
                self._past = getattr(out, "past_key_values", None)
                self._ctx_ids = list(keep)
                self._ctx_tokens = [self._piece(token_id) for token_id in keep]
                self._ctx_kinds = ["prompt"] * len(keep)
                hidden_states = getattr(out, "hidden_states", None)
                if hidden_states:
                    final_residual = _tensor_to_numpy(hidden_states[-2][0])
                    for row, token_id in enumerate(keep):
                        token = self._ctx_tokens[row]
                        index = self._point_counter
                        self._point_counter += 1
                        vector = final_residual[row] if row < final_residual.shape[0] else None
                        if vector is not None:
                            self._projector.add(vector, token, index, "prompt", None)
                self._projector.maybe_fit()
                if self._past is None:
                    self._message = "模型未返回 KV cache：将使用全量重算（较慢）"
                return {
                    "ok": True,
                    "tokens": len(keep),
                    "truncated": offset > 0 or len(full_ids) > len(keep),
                }
            except Exception as exc:
                self._message = f"reset 失败：{exc}"
                return {"ok": False, "message": self._message}

    # ------------------------------------------------------------------
    # Native generation: one forward serves both text and telemetry
    # ------------------------------------------------------------------

    def supports_generation(self) -> bool:
        """True when this engine can generate tokens itself (local model ready)."""
        with self._lock:
            return bool(
                self._status == "ready"
                and self._model is not None
                and self._tokenizer is not None
            )

    def set_native_mode(self, enabled: bool) -> dict:
        """Mark probes as exact (``surrogate=False``) for native generation."""
        with self._lock:
            self._native_mode = bool(enabled)
        return self.status()

    def reset_generation(self) -> dict:
        """Drop native KV cache / logits / pending probe (keeps the model)."""
        with self._lock:
            ready = self._status == "ready" and self._model is not None
            self._reset_context_locked()
            self._projector.reset()
            return {"ok": ready, "tokens": 0, "message": self._message}

    def prefill(
        self,
        messages: Sequence[dict[str, str]] | str,
        *,
        max_context: int | None = None,
        system: str | None = None,
        capture: bool = True,
    ) -> dict:
        """Tokenise a chat prompt, run one forward and keep the KV cache.

        The chat template uses ``add_generation_prompt=True``.  The returned
        probe for the last prompt token is kept internally and returned by the
        first :meth:`sample_next`; the 151k-float logits vector stays in the
        engine (the bridge samples from it) so it never crosses the wire.
        """
        with self._lock:
            if self._status != "ready" or self._model is None:
                return {"ok": False, "tokens": 0, "error": self._message or "透镜未就绪"}
            torch = _get_torch()
            if torch is None:
                return {"ok": False, "tokens": 0, "error": "PyTorch 不可用"}
            started = now_ms()
            try:
                chat = self._normalise_chat_messages(messages, system)
                ids = self._chat_template_ids(chat)
                if not ids:
                    ids = [self._fallback_token_id()]
                limit = self._capacity
                if max_context is not None:
                    try:
                        limit = max(1, min(int(max_context), self._capacity))
                    except (TypeError, ValueError):
                        pass
                keep = [int(token_id) for token_id in ids[-limit:]]
                self._reset_context_locked()
                self._native_mode = True
                self._prompt_tokens = len(keep)
                self._projector.reset()
                self._point_counter = 0
                with torch.inference_mode():
                    input_ids = torch.tensor([keep], dtype=torch.long, device=self._device)
                    out = self._forward_raw(input_ids, past=None, capture=capture)
                self._past = getattr(out, "past_key_values", None)
                self._ctx_ids = list(keep)
                self._ctx_tokens = [self._token_label(token_id) for token_id in keep]
                self._ctx_kinds = ["prompt"] * len(keep)
                self._next_logits = out.logits[0, -1].detach().float().cpu().clone()
                self._pending_forward_id = None
                probes = self._probes_from_output(
                    out,
                    ctx_start=0,
                    rows=[len(keep) - 1],
                    step_for_row=lambda _row: 0,
                    kind="prompt",
                    started=started,
                )
                self._pending_probe = probes[0] if probes else None
                self._projector.maybe_fit()
                elapsed = max(0.0, now_ms() - started)
                self._notice_latency(elapsed)
                return {
                    "ok": True,
                    "tokens": len(keep),
                    "prompt_tokens": len(ids),
                    "truncated": len(ids) > len(keep),
                    "logits": int(self._next_logits.numel()) if self._next_logits is not None else 0,
                    "latency_ms": round(elapsed, 2),
                    "probe_ready": self._pending_probe is not None,
                    "chat_template": self._tokenizer_has_chat_template(),
                }
            except Exception as exc:
                self._message = f"prefill 失败：{exc}"
                self._notice_latency(max(0.0, now_ms() - started))
                return {"ok": False, "tokens": 0, "error": str(exc), "message": self._message}

    def sample_next(
        self,
        params: Any = None,
        step: int = 0,
        *,
        token_id: int | None = None,
        capture: bool = True,
        decode_features: bool = True,
    ) -> dict:
        """Sample one token, forwarding the previously sampled token.

        Call repeatedly after :meth:`prefill`.  Each call appends the token
        chosen by the previous call to the KV cache, builds that position's
        probe from the *same* forward, and samples the next token from the
        freshly produced logits.  ``token_id`` overrides which token is
        appended (tests / external samplers).

        Returns ``token_id``/``text``/``topk``/``probabilities``/``entropy``/
        ``chosen_logprob``/``probe``/``eos`` plus a small timing field.
        ``topk`` keeps the raw model distribution; ``probabilities`` is the
        renormalised distribution actually sampled from (sums to 1).
        """
        with self._lock:
            if self._status != "ready" or self._model is None:
                return {"ok": False, "error": self._message or "透镜未就绪"}
            torch = _get_torch()
            if torch is None:
                return {"ok": False, "error": "PyTorch 不可用"}
            settings = _coerce_sampling_params(params)
            started = now_ms()
            try:
                if token_id is not None:
                    self._pending_forward_id = int(token_id)
                if (
                    self._pending_forward_id is None
                    and (self._next_logits is None or not self._ctx_ids)
                ):
                    return {"ok": False, "error": "尚未 prefill：请先调用 prefill(messages)"}
                self._advance_one_token(
                    torch,
                    int(step),
                    capture=capture,
                    decode_features=decode_features,
                )
                if self._next_logits is None:
                    return {"ok": False, "error": "模型未返回 logits"}
                sampled = self._apply_sampling(self._next_logits, settings, self._gen_ids)
                chosen_id = int(sampled["token_id"])
                self._gen_ids.append(chosen_id)
                delta = self._next_text_delta(self._decode_generated(self._gen_ids))
                probe = self._pending_probe
                self._pending_probe = None
                if isinstance(probe, dict):
                    probe["step"] = int(step)
                self._pending_forward_id = chosen_id
                eos_ids = self.eos_token_ids()
                elapsed = max(0.0, now_ms() - started)
                self._notice_latency(elapsed)
                return {
                    "ok": True,
                    "step": int(step),
                    "token_id": chosen_id,
                    "text": delta,
                    "delta": delta,
                    "piece": self._token_label(chosen_id),
                    "topk": sampled["topk"],
                    "probabilities": sampled["probabilities"],
                    "chosen_logprob": sampled["chosen_logprob"],
                    "chosen_prob": sampled["chosen_prob"],
                    "chosen_rank": sampled["chosen_rank"],
                    "sampled_logprob": sampled["sampled_logprob"],
                    "entropy": sampled["entropy"],
                    "raw_entropy": sampled["raw_entropy"],
                    "temperature": float(getattr(settings, "temperature", 0.7)),
                    "repetition_penalty": float(getattr(settings, "repetition_penalty", 1.05)),
                    "kept": sampled["kept"],
                    "eos": chosen_id in eos_ids,
                    "finished": chosen_id in eos_ids,
                    "probe": probe,
                    "latency_ms": round(elapsed, 2),
                    "context_tokens": len(self._ctx_ids),
                }
            except Exception as exc:
                self._message = f"sample_next 失败：{exc}"
                self._notice_latency(max(0.0, now_ms() - started))
                return {"ok": False, "error": str(exc), "message": self._message}

    def decode_token(self, token_id: int) -> str:
        """Decoded text for one vocabulary id (cleaner than BPE token labels)."""
        return self._token_label(int(token_id))

    def eos_token_ids(self) -> list[int]:
        """Every EOS id the tokenizer/config/generation-config declares."""
        values: list[int] = []
        for holder in (
            self._tokenizer,
            getattr(self._model, "generation_config", None),
            getattr(self._model, "config", None),
        ):
            if holder is None:
                continue
            raw = getattr(holder, "eos_token_id", None)
            if isinstance(raw, (list, tuple, set)):
                for value in raw:
                    if isinstance(value, int) and not isinstance(value, bool):
                        values.append(int(value))
            elif isinstance(raw, int) and not isinstance(raw, bool):
                values.append(int(raw))
        return sorted({value for value in values if value >= 0})

    # -- native generation internals ------------------------------------

    def _advance_one_token(
        self,
        torch: Any,
        step: int,
        *,
        capture: bool,
        decode_features: bool,
    ) -> None:
        """Forward the token chosen last call, keeping past/context/probe in sync."""
        if self._pending_forward_id is None:
            return
        token = int(self._pending_forward_id)
        if len(self._ctx_ids) + 1 > self._capacity:
            self._compact(torch, keep=max(1, self._capacity - 8))
        started = now_ms()
        with torch.inference_mode():
            input_ids = torch.tensor([[token]], dtype=torch.long, device=self._device)
            out = self._forward_raw(input_ids, past=self._past, capture=capture)
        past = getattr(out, "past_key_values", None)
        if past is not None:
            self._past = past
        base = len(self._ctx_ids)
        self._ctx_ids.append(token)
        self._ctx_tokens.append(self._token_label(token))
        self._ctx_kinds.append("generated")
        self._next_logits = out.logits[0, -1].detach().float().cpu().clone()
        self._pending_forward_id = None
        probes = self._probes_from_output(
            out,
            ctx_start=base,
            rows=[0],
            step_for_row=lambda _row: int(step),
            kind="generated",
            started=started,
            decode_features=decode_features,
        )
        self._pending_probe = probes[0] if probes else None
        self._projector.maybe_fit()

    def _apply_sampling(self, logits: Any, settings: Any, recent_ids: Sequence[int]) -> dict:
        """Temperature/top-k/top-p/repetition sampling on the real vocab logits."""
        torch = _get_torch()
        raw = logits.detach().float().reshape(-1)
        vocab = int(raw.numel())
        adjusted = raw.clone()
        penalty = float(getattr(settings, "repetition_penalty", 1.0) or 1.0)
        if abs(penalty - 1.0) > 1e-9 and recent_ids:
            from collections import Counter

            counts = Counter(int(value) for value in list(recent_ids)[-REPETITION_WINDOW:])
            for recent_id, count in counts.items():
                if 0 <= recent_id < vocab:
                    factor = penalty ** min(int(count), 16)
                    value = adjusted[recent_id]
                    adjusted[recent_id] = value / factor if value > 0 else value * factor
        raw_probs = torch.softmax(raw, dim=-1)
        raw_logprobs = torch.log(raw_probs.clamp_min(1e-300))
        top_count = min(NATIVE_TOPK, vocab)
        top_probs, top_ids = torch.topk(raw_probs, top_count)
        topk: list[dict] = []
        for rank, (token_id, prob) in enumerate(
            zip(top_ids.tolist(), top_probs.tolist()), start=1
        ):
            token_id = int(token_id)
            topk.append(
                {
                    "token": self._token_label(token_id),
                    "id": token_id,
                    "logprob": round(float(raw_logprobs[token_id].item()), 6),
                    "prob": round(float(prob), 6),
                    "rank": rank,
                }
            )
        raw_entropy = max(0.0, float(-(raw_probs * raw_logprobs).sum().item()))
        temperature = float(getattr(settings, "temperature", 0.7))
        top_p = float(getattr(settings, "top_p", 0.95))
        top_k_param = int(getattr(settings, "top_k", 0) or 0)
        seed = getattr(settings, "seed", None)
        if temperature <= 0.0:
            # Greedy decoding: deterministic argmax over penalised logits.
            chosen_id = int(torch.argmax(adjusted).item())
            probabilities = [1.0]
            entropy = 0.0
            kept = 1
            sampled_logprob = 0.0
        else:
            scaled = adjusted / temperature
            if top_k_param > 0:
                values, ids = torch.topk(scaled, min(top_k_param, vocab))
            else:
                values, ids = torch.sort(scaled, descending=True)
            probs = torch.softmax(values, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            threshold = torch.tensor(
                min(max(top_p, 1e-6), 1.0), dtype=probs.dtype, device=probs.device
            )
            keep_n = int(torch.searchsorted(cumulative, threshold).item()) + 1
            keep_n = max(1, min(keep_n, int(probs.numel())))
            probs = probs[:keep_n]
            ids = ids[:keep_n]
            total = float(probs.sum().item())
            probs = probs / total if total > 0 else torch.full_like(probs, 1.0 / keep_n)
            generator = self._generator_for_seed(torch, seed)
            if generator is None:
                position = int(torch.multinomial(probs, 1).item())
            else:
                position = int(torch.multinomial(probs, 1, generator=generator).item())
            chosen_id = int(ids[position].item())
            probabilities = [float(value) for value in probs.tolist()]
            entropy = max(
                0.0,
                float(-(probs * torch.log(probs.clamp_min(1e-300))).sum().item()),
            )
            kept = keep_n
            sampled_logprob = math.log(max(float(probs[position].item()), 1e-300))
        chosen_raw_prob = float(raw_probs[chosen_id].item())
        chosen_rank = None
        for item in topk:
            if int(item["id"]) == chosen_id:
                chosen_rank = int(item["rank"])
                break
        return {
            "token_id": chosen_id,
            "topk": topk,
            "probabilities": [round(value, 10) for value in probabilities],
            "entropy": round(entropy, 6),
            "raw_entropy": round(raw_entropy, 6),
            "chosen_logprob": round(math.log(max(chosen_raw_prob, 1e-300)), 6),
            "chosen_prob": round(chosen_raw_prob, 6),
            "chosen_rank": chosen_rank,
            "sampled_logprob": round(sampled_logprob, 6),
            "kept": int(kept),
        }

    def _generator_for_seed(self, torch: Any, seed: Any) -> Any:
        """Per-generation seeded RNG; ``None`` uses the global torch RNG."""
        if seed is None:
            return None
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            return None
        if self._sample_generator is None or self._sample_seed != seed:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            self._sample_generator = generator
            self._sample_seed = seed
        return self._sample_generator

    def _next_text_delta(self, full_text: str) -> str:
        """Incremental text for a byte-level tokenizer (hold back partial chars)."""
        stable = full_text
        while stable.endswith("\ufffd"):
            stable = stable[:-1]
        if stable.startswith(self._gen_decoded):
            delta = stable[len(self._gen_decoded):]
        else:
            delta = stable
        self._gen_decoded = stable
        return delta

    def _decode_generated(self, ids: Sequence[int]) -> str:
        if self._tokenizer is None:
            return ""
        try:
            return str(
                self._tokenizer.decode(
                    [int(token_id) for token_id in ids], skip_special_tokens=True
                )
            )
        except Exception:
            return "".join(self._token_label(int(token_id)) for token_id in ids)

    def _token_label(self, token_id: int) -> str:
        """Readable single-token text (prefers ``decode`` over BPE byte labels)."""
        tokenizer = self._tokenizer
        if tokenizer is None:
            return str(token_id)
        try:
            text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
            if isinstance(text, str) and text:
                return text
        except Exception:
            pass
        return self._piece(int(token_id))

    def _tokenizer_has_chat_template(self) -> bool:
        return bool(getattr(self._tokenizer, "apply_chat_template", None))

    def _normalise_chat_messages(
        self,
        messages: Sequence[dict[str, str]] | str,
        system: str | None = None,
    ) -> list[dict[str, str]]:
        if isinstance(messages, str):
            raw: list[dict[str, str]] = [{"role": "user", "content": messages}]
        else:
            raw = []
            for item in messages or []:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role") or "user").strip().lower()
                if role not in ("system", "user", "assistant"):
                    role = "user"
                content = item.get("content")
                if not isinstance(content, str):
                    content = "" if content is None else str(content)
                raw.append({"role": role, "content": content})
        system_text = str(system or "").strip()
        if system_text and not any(item["role"] == "system" for item in raw):
            raw.insert(0, {"role": "system", "content": system_text})
        if not raw:
            raw = [{"role": "user", "content": ""}]
        return raw

    def _chat_template_ids(self, messages: Sequence[dict[str, str]]) -> list[int]:
        tokenizer = self._tokenizer
        if tokenizer is None:
            return []
        if self._tokenizer_has_chat_template():
            try:
                encoded = tokenizer.apply_chat_template(
                    list(messages), add_generation_prompt=True, tokenize=True
                )
                ids = _encoding_ids(encoded)
                if ids:
                    return ids
            except Exception as exc:
                self._message = f"chat template 失败，已回退文本拼接：{exc}"
        lines = [f"{item['role']}: {item['content']}" for item in messages]
        lines.append("assistant:")
        return self._encode("\n".join(lines))

    def _forward_raw(self, input_ids: Any, past: Any, capture: bool = True) -> Any:
        if capture:
            if self._residual is not None:
                self._residual.prepare()
            if self._features is not None:
                self._features.prepare()
        model = self._model
        return model(
            input_ids,
            past_key_values=past,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
        )

    def observe(self, delta_text: str, step: int) -> list[dict]:
        """Observe newly generated text; returns one probe per new token."""
        with self._lock:
            if self._status != "ready" or self._model is None:
                return []
            delta = "" if delta_text is None else str(delta_text)
            if not delta:
                return []
            torch = _get_torch()
            if torch is None:
                return []
            started = now_ms()
            try:
                new_text = self._all_text + delta
                self._projector.maybe_fit()
                full_ids = self._encode(new_text)
                if not full_ids:
                    return []
                prev_len = len(self._seen_ids)
                prefix_ok = (
                    prev_len <= len(full_ids)
                    and full_ids[:prev_len] == self._seen_ids
                )
                if not prefix_ok or self._past is None:
                    probes = self._rebuild_and_probe(
                        torch, full_ids, new_text, int(step), started, prefix_changed=not prefix_ok
                    )
                else:
                    new_ids = full_ids[prev_len:]
                    if not new_ids:
                        self._seen_ids = list(full_ids)
                        self._all_text = new_text
                        return []
                    probes = self._incremental_probe(
                        torch, new_ids, full_ids, new_text, int(step), started
                    )
                return probes
            except Exception as exc:
                elapsed = max(0.0, now_ms() - started)
                self._message = f"observe 失败：{exc}"
                self._notice_latency(elapsed)
                return []

    def _incremental_probe(
        self,
        torch: Any,
        new_ids: list[int],
        full_ids: list[int],
        new_text: str,
        step: int,
        started: float,
    ) -> list[dict]:
        if len(self._ctx_ids) + len(new_ids) > self._capacity:
            self._compact(torch, keep=max(8, self._capacity - len(new_ids) - 32))
        base = len(self._ctx_ids)
        with torch.inference_mode():
            input_ids = torch.tensor([new_ids], dtype=torch.long, device=self._device)
            out = self._forward_raw(input_ids, past=self._past, capture=True)
        past = getattr(out, "past_key_values", None)
        if past is not None:
            self._past = past
        for row, token_id in enumerate(new_ids):
            self._ctx_ids.append(int(token_id))
            self._ctx_tokens.append(self._piece(token_id))
            self._ctx_kinds.append("generated")
        rows = list(range(len(new_ids)))
        probes = self._probes_from_output(
            out,
            ctx_start=base,
            rows=rows,
            step_for_row=lambda row: step + row,
            kind="generated",
            started=started,
        )
        self._seen_ids = list(full_ids)
        self._all_text = new_text
        self._projector.maybe_fit()
        self._notice_latency(max(0.0, now_ms() - started))
        return probes

    def _rebuild_and_probe(
        self,
        torch: Any,
        full_ids: list[int],
        new_text: str,
        step: int,
        started: float,
        prefix_changed: bool,
    ) -> list[dict]:
        """Fallback: rebuild the cache from full text, probe the newest rows."""
        prev_len = len(self._seen_ids)
        keep = full_ids[-self._capacity :]
        offset = len(full_ids) - len(keep)
        old_ctx_tokens = list(self._ctx_tokens)
        old_ctx_kinds = list(self._ctx_kinds)
        old_len = len(old_ctx_tokens)
        old_point = self._point_counter
        with torch.inference_mode():
            input_ids = torch.tensor([keep], dtype=torch.long, device=self._device)
            out = self._forward_raw(input_ids, past=None, capture=True)
        self._past = getattr(out, "past_key_values", None)
        self._ctx_ids = list(keep)
        self._ctx_tokens = [self._piece(token_id) for token_id in keep]
        prompt_count = max(0, int(self._prompt_tokens) - offset)
        prompt_count = min(prompt_count, len(keep))
        self._ctx_kinds = ["prompt"] * prompt_count + ["generated"] * (len(keep) - prompt_count)
        # Probe rows covering the changed tail: the re-tokenized previous last
        # token plus every token appended by this delta.
        tail_start_abs = max(0, prev_len - 1) if prefix_changed else prev_len
        tail_start = max(0, tail_start_abs - offset)
        if tail_start >= len(keep):
            tail_start = max(0, len(keep) - 1)
        rows = list(range(tail_start, len(keep)))
        last_row = len(keep) - 1

        def step_for_row(row: int) -> int:
            return step - (last_row - row)

        probes = self._probes_from_output(
            out,
            ctx_start=0,
            rows=rows,
            step_for_row=step_for_row,
            kind="generated",
            started=started,
        )
        # Keep projection indices monotonic and add the re-probed points.
        self._point_counter = max(self._point_counter, old_point, old_len)
        self._seen_ids = list(full_ids)
        self._all_text = new_text
        self._projector.maybe_fit()
        self._notice_latency(max(0.0, now_ms() - started))
        return probes

    def _probes_from_output(
        self,
        out: Any,
        ctx_start: int,
        rows: Sequence[int],
        step_for_row: Callable[[int], int],
        kind: str,
        started: float,
        *,
        surrogate: bool | None = None,
        decode_features: bool = True,
    ) -> list[dict]:
        hidden_states = getattr(out, "hidden_states", None)
        attentions = getattr(out, "attentions", None)
        if not hidden_states:
            return []
        final_residual = _tensor_to_numpy(hidden_states[-2][0])
        final_normalized = _tensor_to_numpy(hidden_states[-1][0])
        row_count = int(final_residual.shape[0])
        captured = self._residual.take(row_count) if self._residual is not None else {}
        feature_acts = None
        if self._features is not None and self._features.available:
            feature_acts = self._features.take(row_count)
        down_weight = None
        if self._features is not None and self._features.down_proj is not None:
            try:
                down_weight = _tensor_to_numpy(self._features.down_proj.weight)
            except Exception:
                down_weight = None

        probes: list[dict] = []
        for row in rows:
            row = int(row)
            if row < 0 or row >= row_count:
                continue
            ctx_index = int(ctx_start) + row
            if ctx_index < 0 or ctx_index >= len(self._ctx_ids):
                continue
            token = self._ctx_tokens[ctx_index]
            step_value = int(step_for_row(row))
            window_start = max(0, ctx_index + 1 - self._capacity)
            window_tokens = self._ctx_tokens[window_start : ctx_index + 1]
            window_kinds = self._ctx_kinds[window_start : ctx_index + 1]
            point = self._projector.add(
                final_residual[row],
                token,
                self._point_counter,
                kind,
                step_value,
            )
            self._point_counter += 1

            if attentions:
                try:
                    attention_payload = self._build_attention(
                        attentions, row, ctx_index + 1, window_tokens, ctx_index - window_start
                    )
                except Exception:
                    attention_payload = self._uniform_attention(len(window_tokens), ctx_index - window_start)
            else:
                attention_payload = self._uniform_attention(len(window_tokens), ctx_index - window_start)

            lens: list[dict] = []
            if self._tuned is not None:
                try:
                    layers = self._tuned.lens_layers(3)
                    for layer, (topk, tuned_flag, agreement) in zip(
                        layers,
                        self._tuned.predict_many(hidden_states, layers, row, top_k=8),
                    ):
                        lens.append(
                            {
                                "layer": int(layer),
                                "topk": topk,
                                "tuned": bool(tuned_flag),
                                "agreement": agreement,
                            }
                        )
                except Exception:
                    lens = []

            residual: list[dict] = []
            if captured and self._lm_head_weight is not None:
                try:
                    from .residual import summarize_residual

                    residual = summarize_residual(
                        captured,
                        row,
                        final_residual[row],
                        self._lm_head_weight,
                        self._final_norm,
                        int(self._ctx_ids[ctx_index]),
                    )
                except Exception:
                    residual = []

            features: list[dict] = []
            decode_features_now = decode_features and (step_value % FEATURES_EVERY) == 0
            if (
                decode_features_now
                and feature_acts is not None
                and row < feature_acts.shape[0]
                and down_weight is not None
                and self._lm_head_weight is not None
            ):
                try:
                    from .features import decode_features

                    features = decode_features(
                        feature_acts[row],
                        down_weight,
                        self._lm_head_weight,
                        self._tokenizer,
                        top_k=6,
                    )
                except Exception:
                    features = []

            probes.append(
                {
                    "step": step_value,
                    "model": self._model_name or "local-model",
                    "surrogate": bool(
                        (not self._native_mode) if surrogate is None else surrogate
                    ),
                    "same_weights": bool(self._same_weights),
                    "latency_ms": 0.0,
                    "context": {
                        "tokens": window_tokens,
                        "kinds": window_kinds,
                    },
                    "attention": attention_payload,
                    "lens": lens,
                    "hidden": downsample_vector(final_normalized[row], 16),
                    "residual": residual,
                    "features": features,
                    "kv": self._kv_info(ctx_index + 1),
                    "vector": point if isinstance(point, dict) else None,
                }
            )
        elapsed = max(0.0, now_ms() - started)
        per_token = round(elapsed / max(1, len(probes)), 2)
        for probe in probes:
            probe["latency_ms"] = per_token
        return probes

    def _build_attention(
        self,
        attentions: Sequence[Any],
        row: int,
        kv_len: int,
        window_tokens: Sequence[str],
        current_index: int,
    ) -> dict:
        from .attention import build_attention_payload

        return build_attention_payload(
            attentions,
            position=row,
            kv_len=kv_len,
            layer_indices=list(range(len(attentions))),
            focus_layers=self._focus_layers,
            head_count=self._n_heads,
            window_tokens=window_tokens,
            current_index=current_index,
            max_context=self._capacity,
        )

    def _uniform_attention(self, context_len: int, current_index: int) -> dict:
        n = max(1, int(context_len))
        row = [round(1.0 / n, 6)] * n
        selected = {
            str(layer): {"layer": layer, "mean": list(row)}
            for layer in range(max(1, self._n_layers))
        }
        return {
            "layers": list(range(max(1, self._n_layers))),
            "selected": selected,
            "head_count": int(self._n_heads),
        }

    def _kv_info(self, used: int) -> dict:
        bytes_per_token = max(
            1,
            int(self._n_layers)
            * max(1, int(self._n_kv_heads))
            * max(1, int(self._head_dim))
            * 2
            * max(1, int(self._elem_bytes)),
        )
        used = int(max(0, used))
        return {
            "used": used,
            "capacity": int(self._capacity),
            "bytes": int(used * bytes_per_token),
            "bytes_per_token": int(bytes_per_token),
        }

    # ------------------------------------------------------------------
    # Compaction / tokenizer helpers
    # ------------------------------------------------------------------

    def _compact(self, torch: Any, keep: int) -> None:
        """Recompute the KV cache from the most recent tokens (sliding window)."""
        keep = max(1, min(int(keep), len(self._ctx_ids)))
        if keep >= len(self._ctx_ids):
            return
        self._ctx_ids = self._ctx_ids[-keep:]
        self._ctx_tokens = self._ctx_tokens[-keep:]
        self._ctx_kinds = self._ctx_kinds[-keep:]
        keep_ids = list(self._ctx_ids)
        with torch.inference_mode():
            input_ids = torch.tensor([keep_ids], dtype=torch.long, device=self._device)
            out = self._forward_raw(input_ids, past=None, capture=False)
        self._past = getattr(out, "past_key_values", None)

    def _encode(self, text: str) -> list[int]:
        if self._tokenizer is None:
            return []
        try:
            encoded = self._tokenizer(str(text), add_special_tokens=False)
        except TypeError:
            encoded = self._tokenizer(str(text))
        return _encoding_ids(encoded)

    def _piece(self, token_id: int) -> str:
        if self._tokenizer is None:
            return str(token_id)
        return token_text(self._tokenizer, int(token_id))

    def _fallback_token_id(self) -> int:
        for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
            value = getattr(self._tokenizer, attr, None)
            if isinstance(value, int) and value >= 0:
                return int(value)
        return 0

    def _notice_latency(self, elapsed_ms: float) -> None:
        self._last_latency_ms = round(float(elapsed_ms), 2)
        self._latency_samples.append(float(elapsed_ms))
        if len(self._latency_samples) > 120:
            del self._latency_samples[: len(self._latency_samples) - 120]

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def projection(self) -> dict:
        """Current UMAP/PCA projection payload (ProjectionEvent shape)."""
        return self._projector.snapshot()

    def projection_ready(self) -> bool:
        return self._projector.length >= 8
