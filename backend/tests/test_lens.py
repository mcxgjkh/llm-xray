"""Tests for the surrogate lens package.

Everything here runs offline: a tiny randomly-initialised Qwen2 model and a
character-level tokenizer stand in for the real 0.5B surrogate.  No Hugging
Face weights are downloaded and LM Studio is never contacted.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from xray.lens.api import (  # noqa: E402
    DEMO_MODEL,
    LensEngine,
    demo_lens_status,
    get_engine,
    synthetic_probe,
    synthetic_projection,
    synthetic_projection_event,
)
from xray.lens.attention import classify_head  # noqa: E402
from xray.lens.common import HEAD_TYPES, softmax  # noqa: E402
from xray.lens.features import decode_features  # noqa: E402
from xray.lens.projection import Projector, deterministic_projection  # noqa: E402
from xray.lens.residual import contribution_logits  # noqa: E402
from xray.lens.tuned_lens import (  # noqa: E402
    TunedLens,
    apply_translator,
    collect_calibration_corpus,
    fit_translator,
)

TINY_CFG = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=128,
)

PROBE_KEYS = {
    "step",
    "model",
    "surrogate",
    "latency_ms",
    "context",
    "attention",
    "lens",
    "hidden",
    "residual",
    "features",
    "kv",
    "vector",
}


class CharTokenizer:
    """Tiny deterministic tokenizer: one id per character."""

    def __init__(self) -> None:
        self.vocab_size = 128
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0

    def __len__(self) -> int:
        return self.vocab_size

    def _id(self, char: str) -> int:
        return 8 + (ord(char) % 120)

    def __call__(self, text: str, add_special_tokens: bool = False):
        return {"input_ids": [self._id(ch) for ch in str(text)]}

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "".join(self._char(int(i)) for i in ids)

    def _char(self, token_id: int) -> str:
        return chr(32 + ((int(token_id) - 8) % 95))

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        return [self._char(int(i)) for i in ids]


def build_tiny_model():
    config = Qwen2Config(**TINY_CFG)
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model


def save_tiny_model_dir(directory: Path) -> Path:
    """Write a tiny HF checkpoint + char tokenizer, fully offline."""
    from tokenizers import Tokenizer as RawTokenizer
    from tokenizers import models as raw_models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    alphabet = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-"
    vocab = {char: index for index, char in enumerate(alphabet)}
    vocab["[UNK]"] = len(vocab)
    raw = RawTokenizer(raw_models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]")
    tokenizer.save_pretrained(str(directory))
    model = build_tiny_model()
    model.save_pretrained(str(directory))
    return directory


@pytest.fixture()
def engine(tmp_path):
    model = build_tiny_model()
    tokenizer = CharTokenizer()
    lens = LensEngine(
        model=model,
        tokenizer=tokenizer,
        model_name="tiny/random-qwen2",
        max_context=16,
        cache_dir=str(tmp_path),
        use_umap=False,  # numba/JIT warmup is slow; UMAP path covered separately
    )
    yield lens
    lens.unload()


def test_engine_status_contract(engine):
    status = engine.status()
    assert set(status) >= {
        "status",
        "model",
        "device",
        "surrogate",
        "same_weights",
        "calibrated_layers",
        "message",
        "threads",
    }
    assert status["status"] == "ready"
    assert status["surrogate"] is True
    assert status["model"] == "tiny/random-qwen2"
    assert engine.is_ready()
    engine.set_runtime_model("tiny/random-qwen2")
    assert engine.status()["same_weights"] is True
    engine.set_runtime_model("deepseek-r1-distill-llama-8b")
    assert engine.status()["same_weights"] is False


def test_load_local_checkpoint_without_network(tmp_path):
    """Exercise the real ``load()`` path against a locally saved tiny model."""
    model_dir = save_tiny_model_dir(tmp_path / "tiny-hf")
    lens = LensEngine(cache_dir=str(tmp_path / "cache"), use_umap=False)
    status = lens.load(
        str(model_dir), device="cpu", low_priority=False, schedule_calibration=False
    )
    assert status["status"] == "ready", status["message"]
    assert lens.is_ready()
    assert status["model"] == str(model_dir)
    assert status["device"] == "cpu"
    assert status["calibrated_layers"] == 0
    assert lens.reset("hello")["ok"] is True
    probes = lens.observe("world", step=0)
    assert len(probes) == 5  # char-level tokenizer: one probe per token
    assert probes[-1]["step"] == 4
    assert all(PROBE_KEYS <= set(probe) for probe in probes)
    lens.unload()
    assert lens.status()["status"] == "unloaded"


def test_load_resolves_project_models_dir(tmp_path, monkeypatch):
    """Mirror scripts/fetch_lens_model.ps1: models/lens/<org__model>."""
    models_dir = tmp_path / "models"
    save_tiny_model_dir(models_dir / "lens" / "TestOrg__TinyLocal")
    monkeypatch.setenv("LLM_XRAY_MODELS_DIR", str(models_dir))
    lens = LensEngine(cache_dir=str(tmp_path / "cache"), use_umap=False)
    status = lens.load(
        "TestOrg/TinyLocal", low_priority=False, schedule_calibration=False
    )
    assert status["status"] == "ready", status["message"]
    # Display name stays the repo id, so same_weights can match the runtime model.
    assert status["model"] == "TestOrg/TinyLocal"
    lens.set_runtime_model("TinyLocal")
    assert lens.status()["same_weights"] is True
    assert lens.reset("abc")["ok"] is True
    assert len(lens.observe("de", 0)) == 2
    lens.unload()


def test_load_accepts_env_model_path(tmp_path, monkeypatch):
    model_dir = save_tiny_model_dir(tmp_path / "env-model")
    monkeypatch.setenv("LLM_XRAY_LENS_MODEL", str(model_dir))
    lens = LensEngine(cache_dir=str(tmp_path / "cache"), use_umap=False)
    status = lens.load(low_priority=False, schedule_calibration=False)
    assert status["status"] == "ready", status["message"]
    assert status["model"] == str(model_dir)
    lens.unload()


def test_load_detects_incomplete_model_dir(tmp_path, monkeypatch):
    """A half-downloaded models/lens dir must not look like a ready model."""
    model_dir = tmp_path / "models" / "lens" / "Partial__Tiny"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")
    monkeypatch.setenv("LLM_XRAY_MODELS_DIR", str(tmp_path / "models"))
    lens = LensEngine(cache_dir=str(tmp_path / "cache"), use_umap=False)
    status = lens.load("Partial/Tiny", low_priority=False, schedule_calibration=False)
    assert status["status"] == "error"
    assert "权重" in status["message"]
    lens.unload()


def test_unloaded_engine_is_safe(tmp_path):
    lens = LensEngine(cache_dir=str(tmp_path))
    assert lens.status()["status"] == "unloaded"
    assert lens.is_ready() is False
    assert lens.reset("hello")["ok"] is False
    assert lens.observe("x", 0) == []
    # Loading a model that is not present locally must fail gracefully
    # (local_files_only=True: no network, no download).
    status = lens.load("xray-test/not-a-local-model", device="cpu", low_priority=False)
    assert status["status"] == "error"
    assert status["message"]
    assert "fetch_lens_model.ps1" in status["message"]
    assert lens.observe("x", 0) == []
    assert lens.projection()["points"] == []


def _assert_attention_contract(payload, expected_heads):
    assert isinstance(payload, dict)
    assert set(payload) >= {"layers", "selected", "head_count"}
    assert payload["head_count"] == expected_heads
    assert payload["layers"]
    assert payload["selected"]
    for key, entry in payload["selected"].items():
        assert int(key) == entry["layer"]
        assert abs(sum(entry["mean"]) - 1.0) < 1e-3
        if "heads" in entry:
            assert len(entry["heads"]) == expected_heads
            assert len(entry["head_types"]) == expected_heads
            for row in entry["heads"]:
                assert len(row) == len(entry["mean"])
                assert abs(sum(row) - 1.0) < 1e-3
            for head_type in entry["head_types"]:
                assert head_type in HEAD_TYPES
    # exactly two focus layers carry per-head rows (tiny model has 2 layers)
    focus_count = sum(1 for entry in payload["selected"].values() if "heads" in entry)
    assert focus_count == 2


def test_observe_single_and_multi_token(engine):
    engine.reset("hello world", system="be nice")
    first = engine.observe("A", step=0)
    assert len(first) == 1
    probe = first[0]
    assert PROBE_KEYS <= set(probe)
    assert probe["step"] == 0
    assert probe["surrogate"] is True
    assert probe["model"] == "tiny/random-qwen2"
    assert probe["latency_ms"] >= 0.0
    assert probe["context"]["tokens"]
    assert len(probe["context"]["kinds"]) == len(probe["context"]["tokens"])
    assert probe["context"]["kinds"][-1] == "generated"
    assert set(probe["context"]["kinds"]) <= {"prompt", "generated"}
    _assert_attention_contract(probe["attention"], TINY_CFG["num_attention_heads"])
    assert len(probe["lens"]) >= 2
    for entry in probe["lens"]:
        assert entry["layer"] in range(TINY_CFG["num_hidden_layers"])
        assert entry["topk"]
        probs = [option["prob"] for option in entry["topk"]]
        assert 0.0 <= sum(probs) <= 1.0 + 1e-6
        assert all(0.0 <= prob <= 1.0 for prob in probs)
        for rank, option in enumerate(entry["topk"], start=1):
            assert option["rank"] == rank
            assert isinstance(option["token"], str)
            assert 0 <= option["id"] < TINY_CFG["vocab_size"]
    assert len(probe["hidden"]) == 16
    assert all(math.isfinite(value) for value in probe["hidden"])
    assert len(probe["residual"]) == TINY_CFG["num_hidden_layers"]
    for row in probe["residual"]:
        assert row["attn_norm"] >= 0.0 and row["mlp_norm"] >= 0.0
        assert math.isfinite(row["attn_logit"]) and math.isfinite(row["mlp_logit"])
    assert probe["features"]
    for feature in probe["features"]:
        assert isinstance(feature["label"], str) and feature["label"]
        assert 0.0 <= abs(feature["activation"]) <= 1.0
    assert probe["kv"]["used"] == len(probe["context"]["tokens"])
    assert probe["kv"]["capacity"] == 16
    assert probe["kv"]["bytes"] > 0
    assert probe["kv"]["bytes_per_token"] > 0
    assert set(probe["vector"]) == {"x", "y", "z"}

    # A delta with several tokens returns one probe per token, steps advance.
    multi = engine.observe("bc", step=1)
    assert len(multi) == 2
    assert [item["step"] for item in multi] == [1, 2]
    assert len(multi[1]["context"]["tokens"]) == len(multi[0]["context"]["tokens"]) + 1
    for item in multi:
        _assert_attention_contract(item["attention"], TINY_CFG["num_attention_heads"])
    assert engine.latency_ms is not None


def test_observe_context_is_capped(engine):
    engine.reset("abcdefghijklmnopqrstuvwxyz")
    # max_context=16: prompt is truncated and long generation triggers compaction.
    for step in range(24):
        probes = engine.observe(chr(ord("a") + step % 26), step)
        for probe in probes:
            assert len(probe["context"]["tokens"]) <= 16
            assert probe["kv"]["used"] <= 16
            _assert_attention_contract(probe["attention"], TINY_CFG["num_attention_heads"])


def test_tokenizer_boundary_fallback_does_not_crash(engine):
    engine.reset("hello")
    engine.observe("a", 0)
    # Simulate a tokenizer that re-tokenizes the whole prefix differently.
    original = engine._tokenizer

    class ShiftingTokenizer(CharTokenizer):
        def __call__(self, text, add_special_tokens=False):
            ids = super().__call__(text, add_special_tokens)["input_ids"]
            if len(ids) > 2:
                ids = ids[:-2] + [ids[-2]] + [ids[-1]]  # same length, force prefix diff
            return {"input_ids": ids}

    shifting = ShiftingTokenizer()
    engine._tokenizer = shifting
    probes = engine.observe("bc", 1)
    engine._tokenizer = original
    assert isinstance(probes, list)
    for probe in probes:
        assert PROBE_KEYS <= set(probe)
        _assert_attention_contract(probe["attention"], TINY_CFG["num_attention_heads"])


def test_classify_head_heuristics():
    tokens = ["<s>", "Hello", "world", ",", "this", "is", "fine"]
    previous = np.zeros(len(tokens))
    previous[-2] = 0.9
    previous[-1] = 0.1
    assert classify_head(previous, tokens, len(tokens) - 1) == "previous-token"
    uniform = np.full(len(tokens), 1.0 / len(tokens))
    assert classify_head(uniform, tokens, len(tokens) - 1) in {"diffuse", "global"}
    punctuation = np.zeros(len(tokens))
    punctuation[3] = 0.8
    punctuation[-1] = 0.2
    assert classify_head(punctuation, tokens, len(tokens) - 1) == "syntactic"
    local = np.zeros(len(tokens))
    local[-4:] = 0.25
    plain_tokens = ["<s>", "Hello", "world", "this", "is", "all", "fine"]
    assert classify_head(local, plain_tokens, len(plain_tokens) - 1) == "local"


def test_tuned_lens_pure_functions():
    torch.manual_seed(0)
    n, dim = 48, 16
    x = torch.randn(n, dim)
    true_w = torch.randn(dim, dim) * 0.1
    true_b = torch.randn(dim) * 0.1
    y = x @ true_w + true_b
    w, b = fit_translator(x, y, ridge=1e-4)
    assert w.shape == (dim, dim)
    assert b.shape == (dim,)
    fitted = apply_translator(x, w, b)
    error = float((fitted - y).abs().mean())
    assert error < 0.02
    single = apply_translator(x[0], w, b)
    assert single.shape == (dim,)


def test_tuned_lens_calibration_and_cache(engine, tmp_path):
    tuned = engine._tuned
    assert tuned is not None
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Attention heads copy the previous token in a sequence.",
        "A tuned lens maps residual states into the final layer.",
        "Ridge regression gives a closed form affine translator.",
    ]
    count = tuned.calibrate(engine._model, corpus, max_total_tokens=128)
    assert count == TINY_CFG["num_hidden_layers"]
    assert tuned.calibrated_layers == count
    assert tuned.cache_path.endswith(".pt")
    assert Path(tuned.cache_path).is_file()
    assert all(0.0 <= agreement <= 1.0 for agreement in tuned._agreements.values())

    # Probe now uses tuned lens entries with an agreement value.
    engine.reset("hello")
    probe = engine.observe("x", 0)[0]
    assert any(entry["tuned"] for entry in probe["lens"])
    assert all(entry["agreement"] is None or 0.0 <= entry["agreement"] <= 1.0 for entry in probe["lens"])

    # Cache reload validates shapes.
    fresh = TunedLens(
        model_name="tiny/random-qwen2",
        n_layers=TINY_CFG["num_hidden_layers"],
        hidden_size=TINY_CFG["hidden_size"],
        final_norm=engine._final_norm,
        lm_head=engine._lm_head,
        tokenizer=engine._tokenizer,
        cache_dir=str(tmp_path),
    )
    assert fresh.load_cache() is True
    assert fresh.calibrated_layers == count


def test_collect_calibration_corpus():
    corpus = collect_calibration_corpus(max_segments=12)
    assert isinstance(corpus, list)
    assert 0 < len(corpus) <= 12
    assert all(isinstance(item, str) and item.strip() for item in corpus)


def test_residual_contribution_math():
    final_hidden = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    branch = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    lm_head = np.zeros((4, 3), dtype=np.float32)
    lm_head[2] = [1.0, 0.0, 0.0]
    gain = np.ones(3, dtype=np.float32)
    value = contribution_logits(final_hidden, branch, lm_head, 2, gain, eps=1e-6)
    rms = math.sqrt((1 + 4 + 0.25) / 3)
    expected = 0.1 / rms
    assert abs(value - expected) < 1e-5
    # Sign flips with the branch direction; out-of-range chosen id is neutral.
    assert contribution_logits(final_hidden, -branch, lm_head, 2, gain) == pytest.approx(-value, abs=1e-6)
    assert contribution_logits(final_hidden, branch, lm_head, 99, gain) == 0.0


def test_features_decode():
    rng = np.random.default_rng(0)
    acts = rng.normal(size=32).astype(np.float32)
    down = rng.normal(size=(16, 32)).astype(np.float32)
    lm_head = rng.normal(size=(128, 16)).astype(np.float32)
    tokenizer = CharTokenizer()
    features = decode_features(acts, down, lm_head, tokenizer, top_k=4)
    assert len(features) == 4
    ids = [feature["id"] for feature in features]
    assert len(set(ids)) == len(ids)
    assert all(0 <= feature["id"] < 32 for feature in features)
    assert all(0.0 <= abs(feature["activation"]) <= 1.0 for feature in features)
    assert all(feature["tokens"] for feature in features)


def test_projection_umap_or_pca(engine):
    projector = Projector(max_points=32, fit_every=8, use_umap=False)
    rng = np.random.default_rng(3)
    for index in range(10):
        hidden = rng.normal(size=16)
        kind = "prompt" if index < 3 else "generated"
        point = projector.add(hidden, f"tok{index}", index, kind, None if kind == "prompt" else index - 3)
        assert set(point) == {"x", "y", "z"}
    projector.maybe_fit(force=True)
    projector.wait(timeout=30.0)
    snapshot = projector.snapshot()
    assert len(snapshot["points"]) == 10
    assert snapshot["reducer"] in {"UMAP", "PCA", "deterministic"}
    assert snapshot["fitted_at"] is not None
    kinds = {point["kind"] for point in snapshot["points"]}
    assert "prompt" in kinds and "generated" in kinds
    for point in snapshot["points"]:
        for axis in ("x", "y", "z"):
            assert -1.0 - 1e-6 <= point[axis] <= 1.0 + 1e-6
            assert math.isfinite(point[axis])
        assert set(point) >= {"id", "token", "index", "kind", "step", "x", "y", "z"}

    # Engine projection() is also a ProjectionEvent payload and contains prompt+generated.
    engine.reset("hello")
    engine.observe("world", 0)
    projection = engine.projection()
    assert set(projection) == {"points", "reducer", "fitted_at"}
    assert {point["kind"] for point in projection["points"]} == {"prompt", "generated"}


def test_projection_fallback_and_determinism():
    points = [
        {"id": f"t{i}", "token": f"tok{i}", "index": i, "kind": "generated", "step": i}
        for i in range(4)
    ]
    first = deterministic_projection(points)
    second = deterministic_projection(points)
    assert first == second
    assert len(first) == len(points)
    projector = Projector(max_points=8, fit_every=4)
    for point in points:
        projector.add(np.zeros(8), point["token"], point["index"], "generated", point["step"])
    projector.fit_sync()
    snapshot = projector.snapshot()
    assert snapshot["reducer"] in {"PCA", "deterministic"}
    assert len(snapshot["points"]) == len(points)


def test_demo_probe_contract_and_stability():
    tokens = ["<|im_start|>", "user", "紫色", "violet", "是什么", "?"]
    probe = synthetic_probe(3, tokens)
    repeated = synthetic_probe(3, tokens)
    assert probe == repeated
    assert set(probe) >= PROBE_KEYS
    assert probe["model"] == DEMO_MODEL
    assert probe["surrogate"] is True
    assert probe["step"] == 3
    assert probe["context"]["tokens"] == tokens
    assert len(probe["context"]["kinds"]) == len(tokens)
    assert set(probe["context"]["kinds"]) <= {"prompt", "generated"}
    attention = probe["attention"]
    assert attention["head_count"] > 0
    assert len(attention["layers"]) >= 2
    assert attention["selected"]
    focus_with_heads = 0
    for entry in attention["selected"].values():
        assert abs(sum(entry["mean"]) - 1.0) < 1e-3
        if "heads" in entry:
            focus_with_heads += 1
            assert len(entry["heads"]) == attention["head_count"]
            assert len(entry["head_types"]) == attention["head_count"]
            for row in entry["heads"]:
                assert abs(sum(row) - 1.0) < 1e-3
            assert all(head_type in HEAD_TYPES for head_type in entry["head_types"])
    assert focus_with_heads == 2
    assert len(probe["lens"]) == 3
    for entry in probe["lens"]:
        probs = [option["prob"] for option in entry["topk"]]
        assert 0.0 < sum(probs) <= 1.0 + 1e-6
        assert sum(probs) > 0.7  # plausible distribution over top-k
        assert [option["rank"] for option in entry["topk"]] == list(range(1, len(entry["topk"]) + 1))
    assert len(probe["hidden"]) == 16
    assert all(math.isfinite(value) for value in probe["hidden"])
    assert probe["residual"]
    assert all(math.isfinite(row["attn_norm"]) and row["attn_norm"] >= 0 for row in probe["residual"])
    assert probe["features"]
    labels = [feature["label"] for feature in probe["features"]]
    assert any("紫色" in label or "violet" in label for label in labels)
    assert probe["kv"]["used"] == len(tokens)
    assert probe["kv"]["used"] <= probe["kv"]["capacity"]
    assert set(probe["vector"]) == {"x", "y", "z"}

    # Empty / single-token inputs remain contract-valid.
    empty = synthetic_probe(0, [])
    assert empty["context"]["tokens"] == ["<empty>"]
    assert empty["attention"]["selected"]


def test_demo_projection_helpers():
    points = [
        {"id": "t0", "token": "a", "index": 0, "kind": "prompt", "step": None},
        {"id": "t1", "token": "b", "index": 1, "kind": "generated", "step": 0},
    ]
    projected = synthetic_projection(points)
    assert len(projected) == 2
    for point in projected:
        assert -1.0 <= point["x"] <= 1.0
        assert -1.0 <= point["y"] <= 1.0
        assert -1.0 <= point["z"] <= 1.0
    event = synthetic_projection_event(points)
    assert event["reducer"] == "UMAP"
    assert len(event["points"]) == 2
    status = demo_lens_status()
    assert status["status"] in {"disabled", "ready"}
    assert status["surrogate"] is True
    assert status["model"] == DEMO_MODEL


def test_get_engine_is_singleton():
    first = get_engine()
    second = get_engine()
    assert first is second
    assert isinstance(first, LensEngine)
    assert first.status()["status"] == "unloaded"
