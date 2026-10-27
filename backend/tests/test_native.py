"""Tests for the native exact inference channel.

Everything runs offline:

* a tiny randomly-initialised Qwen2 model checks the full interface contract
  (``prefill`` / ``sample_next`` / probe fields / attention rows / EOS / caps);
* the already-downloaded ``Qwen__Qwen2.5-0.5B-Instruct`` under ``models/lens``
  checks one short real generation through :class:`NativeBridge`.

No Hugging Face downloads, no LM Studio calls, no 8B model.
"""

from __future__ import annotations

import asyncio
import importlib.util
import math
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from xray.lens.engine import LensEngine  # noqa: E402
from xray.native_bridge import NativeBridge  # noqa: E402
from xray.sampling import SamplingParams  # noqa: E402

TINY_CFG = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=128,
)

QWEN_DIR = ROOT / "models" / "lens" / "Qwen__Qwen2.5-0.5B-Instruct"
#: Qwen2.5-0.5B: 24 layers x 14 heads.
QWEN_LAYERS = 24
QWEN_HEADS = 14

PROBE_KEYS = {
    "step",
    "model",
    "surrogate",
    "same_weights",
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
    """Tiny deterministic tokenizer: one id per character (no chat template)."""

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
        if isinstance(ids, int):
            ids = [ids]
        return "".join(self._char(int(i)) for i in ids)

    def _char(self, token_id: int) -> str:
        return chr(32 + ((int(token_id) - 8) % 95))

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        return [self._char(int(i)) for i in ids]


def build_tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(**TINY_CFG)
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model


@pytest.fixture()
def tiny_engine(tmp_path):
    engine = LensEngine(
        model=build_tiny_model(),
        tokenizer=CharTokenizer(),
        model_name="tiny/native-qwen2",
        max_context=16,
        cache_dir=str(tmp_path),
        use_umap=False,
    )
    # Native generation is the runtime model itself: exact, not surrogate.
    engine.set_runtime_model("tiny/native-qwen2")
    engine.set_native_mode(True)
    yield engine
    engine.unload()


def _assert_attention_contract(payload, expected_heads):
    assert isinstance(payload, dict)
    assert set(payload) >= {"layers", "selected", "head_count"}
    assert payload["head_count"] == expected_heads
    assert payload["layers"] and payload["selected"]
    for key, entry in payload["selected"].items():
        assert int(key) == entry["layer"]
        assert abs(sum(entry["mean"]) - 1.0) < 1e-3
        if "heads" in entry:
            assert len(entry["heads"]) == expected_heads
            assert len(entry["head_types"]) == expected_heads
            for row in entry["heads"]:
                assert len(row) == len(entry["mean"])
                assert abs(sum(row) - 1.0) < 1e-3


def test_native_tiny_prefill_and_generate(tiny_engine):
    engine = tiny_engine
    assert engine.supports_generation() is True
    assert engine.status()["surrogate"] is False
    assert engine.eos_token_ids() == [2]
    assert isinstance(engine.decode_token(9), str)

    pre = engine.prefill(
        [{"role": "user", "content": "hello world"}], max_context=16
    )
    assert pre["ok"], pre
    assert pre["tokens"] == 16  # fallback text is longer than the 16-token cap
    assert pre["truncated"] is True
    assert pre["logits"] == TINY_CFG["vocab_size"]
    assert pre["probe_ready"] is True
    assert pre["chat_template"] is False  # CharTokenizer has no chat template

    params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.05,
        max_tokens=4,
        seed=123,
    )
    outputs = []
    for step in range(3):
        out = engine.sample_next(params, step)
        assert out["ok"], out
        outputs.append(out)
        assert out["step"] == step
        assert 0 <= out["token_id"] < TINY_CFG["vocab_size"]
        assert out["piece"]
        # Raw top-k distribution.
        assert out["topk"]
        assert len(out["topk"]) == min(20, TINY_CFG["vocab_size"])
        for rank, option in enumerate(out["topk"], start=1):
            assert option["rank"] == rank
            assert 0 <= option["id"] < TINY_CFG["vocab_size"]
            assert isinstance(option["token"], str) and option["token"]
            assert math.isfinite(option["logprob"]) and option["logprob"] <= 0.0
            assert 0.0 <= option["prob"] <= 1.0
        # Effective (renormalised) sampling distribution sums to one.
        assert abs(sum(out["probabilities"]) - 1.0) < 1e-6
        assert out["entropy"] >= 0.0
        assert out["raw_entropy"] >= 0.0
        assert out["chosen_logprob"] <= 0.0
        assert out["temperature"] == 0.7
        assert isinstance(out["eos"], bool)

        probe = out["probe"]
        assert probe is not None
        assert PROBE_KEYS <= set(probe)
        assert probe["step"] == step
        assert probe["surrogate"] is False
        assert probe["same_weights"] is True
        assert probe["model"] == "tiny/native-qwen2"
        assert probe["latency_ms"] >= 0.0
        assert probe["context"]["tokens"]
        assert len(probe["context"]["tokens"]) == len(probe["context"]["kinds"])
        assert set(probe["context"]["kinds"]) <= {"prompt", "generated"}
        assert len(probe["hidden"]) == 16
        assert all(math.isfinite(value) for value in probe["hidden"])
        _assert_attention_contract(probe["attention"], TINY_CFG["num_attention_heads"])
        assert len(probe["lens"]) >= 2
        for entry in probe["lens"]:
            assert entry["layer"] in range(TINY_CFG["num_hidden_layers"])
            assert entry["topk"]
        assert len(probe["residual"]) == TINY_CFG["num_hidden_layers"]
        assert probe["kv"]["used"] <= 16
        assert probe["kv"]["capacity"] == 16
        assert set(probe["vector"]) == {"x", "y", "z"}

    # First probe comes from prefill (all prompt), later ones from generated rows.
    assert set(outputs[0]["probe"]["context"]["kinds"]) == {"prompt"}
    assert outputs[-1]["probe"]["context"]["kinds"][-1] == "generated"
    # Sliding-window cap is enforced across generation.
    assert len(engine._ctx_ids) <= 16

    # Greedy decoding must agree with the raw argmax when no repetition applies.
    engine.reset_generation()
    assert engine.prefill([{"role": "user", "content": "hi"}])["ok"] is True
    greedy = engine.sample_next(
        SamplingParams(temperature=0.0, top_k=40, repetition_penalty=1.0, seed=None), 0
    )
    assert greedy["ok"], greedy
    assert greedy["probabilities"] == [1.0]
    assert greedy["entropy"] == 0.0
    assert greedy["token_id"] == greedy["topk"][0]["id"]


def test_native_bridge_status_contract():
    bridge = NativeBridge(model_name="Xray/NotLoadedYet")
    status = bridge.status()
    assert status["status"] == "unloaded"
    assert status["available"] is True  # torch + transformers are installed
    assert status["surrogate"] is False
    assert status["exact"] is False
    assert status["same_weights"] is False
    assert status["generation"] is False
    assert status["model"] == "Xray/NotLoadedYet"
    assert "本地" in status["message"] or "加载" in status["message"]
    assert bridge.supports_generation() is False
    assert bridge.projection() is None


def test_native_load_missing_local_model_is_graceful():
    """No download is attempted; the error names the fetch script."""
    async def flow():
        bridge = NativeBridge()
        return await bridge.load("xray-test/native-model-that-is-not-local")

    status = asyncio.run(flow())
    assert status["status"] == "error"
    assert status["message"]
    assert "fetch_lens_model.ps1" in status["message"]


def test_native_bridge_4bit_without_bitsandbytes():
    if importlib.util.find_spec("bitsandbytes") is not None:
        pytest.skip("bitsandbytes installed on this host")
    async def flow():
        bridge = NativeBridge()
        return await bridge.load(load_in_4bit=True)

    status = asyncio.run(flow())
    assert status["status"] == "error"
    assert "bitsandbytes" in status["message"]


@pytest.mark.skipif(
    not (QWEN_DIR / "model.safetensors").is_file(),
    reason="models/lens/Qwen__Qwen2.5-0.5B-Instruct not downloaded",
)
def test_native_bridge_qwen_0_5b_short_generation():
    """One real short generation with all tensors from the same forward."""
    started = time.monotonic()

    async def flow():
        bridge = NativeBridge(model_name=str(QWEN_DIR))
        status = await bridge.load(str(QWEN_DIR))
        assert status["status"] == "ready", status
        assert status["surrogate"] is False
        assert status["exact"] is True
        assert status["same_weights"] is True
        assert status["generation"] is True
        assert status["model"]

        pre = await bridge.prefill(
            [{"role": "user", "content": "用一句话说你好"}], max_context=128
        )
        assert pre["ok"], pre
        assert pre["tokens"] > 0
        assert pre["prompt_tokens"] == pre["tokens"]
        assert pre["logits"] == 151936
        assert pre["chat_template"] is True

        params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.05,
            max_tokens=4,
            seed=42,
        )
        outputs = []
        for step in range(3):
            out = await bridge.sample_next(params, step)
            assert out["ok"], out
            outputs.append(out)
            assert 0 <= out["token_id"] < 151936
            assert out["topk"]
            assert all(math.isfinite(option["logprob"]) for option in out["topk"])
            assert abs(sum(out["probabilities"]) - 1.0) < 1e-6
            assert out["entropy"] >= 0.0
            assert out["chosen_logprob"] <= 0.0
            assert isinstance(out["eos"], bool)
            probe = out["probe"]
            assert probe is not None
            assert PROBE_KEYS <= set(probe)
            assert probe["surrogate"] is False
            assert probe["same_weights"] is True
            assert probe["context"]["tokens"]
            assert len(probe["context"]["tokens"]) == len(probe["context"]["kinds"])
            assert set(probe["context"]["kinds"]) <= {"prompt", "generated"}
            assert probe["attention"]["head_count"] == QWEN_HEADS
            assert probe["attention"]["layers"]
            for entry in probe["attention"]["selected"].values():
                assert abs(sum(entry["mean"]) - 1.0) < 1e-3
            assert any("heads" in entry for entry in probe["attention"]["selected"].values())
            assert len(probe["hidden"]) == 16
            assert len(probe["lens"]) >= 2
            for entry in probe["lens"]:
                assert entry["layer"] in range(QWEN_LAYERS)
                assert entry["topk"]
            assert probe["residual"]
            assert probe["kv"]["used"] <= 256
            assert set(probe["vector"]) == {"x", "y", "z"}

        combined = "".join(out["text"] for out in outputs)
        assert combined.strip(), [out["text"] for out in outputs]
        assert bridge.projection() is not None

        bridge.unload()
        assert bridge.status()["status"] == "unloaded"
        return pre, outputs

    pre, outputs = asyncio.run(flow())
    elapsed = time.monotonic() - started
    # 0.5B CPU 2 threads: load + prefill + 3 tokens must stay interactive.
    assert elapsed < 120.0, f"native 0.5B test took {elapsed:.1f}s"
