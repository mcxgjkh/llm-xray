"""Unit tests for the pure sampling maths (no network, no models)."""

from __future__ import annotations

import math

from xray.sampling import (
    SamplingParams,
    apply_sampling,
    approximate_prompt_tokens,
    entropy_of,
    normalize_options,
)

RAW = [
    {"token": "你", "id": 1, "logprob": -0.1},
    {"token": "好", "id": 2, "logprob": -1.2},
    {"token": "吗", "id": 3, "logprob": -2.5},
]


def neutral(**overrides) -> SamplingParams:
    values = dict(temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0)
    values.update(overrides)
    return SamplingParams(**values)


def test_normalize_options_handles_loose_shapes():
    options = normalize_options(
        [
            {"token": "a", "logprob": -1.0},
            ("b", -2.0),
            {"text": "c", "score": -3.0},
            {"bytes": [228, 189, 160], "logprob": -0.5},
            {"token": "a", "logprob": -0.2},  # duplicate: keep best
            "garbage",
            None,
        ]
    )
    tokens = {option["token"] for option in options}
    assert tokens == {"a", "b", "c", "你"}
    assert next(option for option in options if option["token"] == "a")["logprob"] == -0.2


def test_softmax_probabilities_sum_to_one_and_entropy_positive():
    result = apply_sampling(RAW, neutral())
    assert result["kept"] == 3
    assert math.isclose(sum(result["probabilities"]), 1.0, rel_tol=1e-9)
    probs = [item["prob"] for item in result["topk"]]
    assert probs == sorted(probs, reverse=True)
    assert result["entropy"] is not None and result["entropy"] > 0
    assert result["topk"][0]["rank"] == 1


def test_chosen_token_rank_probability():
    result = apply_sampling(RAW, neutral(), chosen_token="好")
    assert result["chosen_rank"] == 2
    assert result["chosen_prob"] is not None and 0 < result["chosen_prob"] < 1
    assert result["chosen_logprob"] == -1.2

    missing = apply_sampling(RAW, neutral(), chosen_token="不在候选里")
    assert missing["chosen_prob"] is None
    assert missing["chosen_rank"] is None


def test_top_k_truncation():
    result = apply_sampling(RAW, neutral(top_k=2))
    assert result["kept"] == 2
    assert [item["token"] for item in result["topk"]] == ["你", "好"]


def test_top_p_truncation():
    result = apply_sampling(RAW, neutral(top_p=0.5))
    assert result["kept"] < 3
    assert result["probabilities"][0] > 0.5
    assert math.isclose(sum(result["probabilities"]), 1.0, rel_tol=1e-9)


def test_temperature_sharpens_distribution():
    cold = apply_sampling(RAW, neutral(temperature=0.2))
    hot = apply_sampling(RAW, neutral(temperature=2.0))
    assert cold["topk"][0]["prob"] > hot["topk"][0]["prob"]
    assert cold["entropy"] < hot["entropy"]


def test_greedy_when_temperature_zero():
    result = apply_sampling(RAW, neutral(temperature=0.0), chosen_token="你")
    assert result["kept"] == 1
    assert result["probabilities"] == [1.0]
    assert result["entropy"] == 0.0
    assert result["topk"][0]["token"] == "你"
    assert result["chosen_prob"] == 1.0


def test_repetition_penalty_lowers_repeated_token_probability():
    baseline = apply_sampling(RAW, neutral(), chosen_token="你")
    repeated = apply_sampling(RAW, neutral(repetition_penalty=1.8), recent_tokens=["你", "你"], chosen_token="你")
    assert repeated["chosen_prob"] < baseline["chosen_prob"]

    # A single occurrence uses the penalty once; two occurrences use it twice.
    once = apply_sampling(RAW, neutral(repetition_penalty=1.8), recent_tokens=["你"], chosen_token="你")
    twice = apply_sampling(RAW, neutral(repetition_penalty=1.8), recent_tokens=["你", "你"], chosen_token="你")
    assert twice["chosen_prob"] < once["chosen_prob"] < baseline["chosen_prob"]


def test_repetition_window_is_limited_to_recent_tokens():
    recent = ["你"] * 100
    result = apply_sampling(RAW, neutral(repetition_penalty=1.05), recent_tokens=recent, chosen_token="你")
    assert result["chosen_prob"] is not None
    assert result["chosen_prob"] > 0


def test_empty_and_invalid_input_returns_empty_result():
    for value in (None, [], [{"token": "x"}], [{"token": "x", "logprob": float("nan")}]):
        result = apply_sampling(value, neutral())
        assert result["topk"] == []
        assert result["entropy"] is None


def test_sampling_params_clamp_and_parse():
    params = SamplingParams.from_dict(
        {
            "temperature": 99,
            "top_p": 0,
            "top_k": -12,
            "repetition_penalty": 9,
            "max_tokens": 0,
            "seed": "42",
        }
    )
    assert params.temperature == 4.0
    assert params.top_p == 0.01
    assert params.top_k == 0
    assert params.repetition_penalty == 2.0
    assert params.max_tokens == 1
    assert params.seed == 42

    bad = SamplingParams.from_dict({"temperature": "hot", "top_k": "many", "seed": "x"})
    assert bad.temperature == 0.7
    assert bad.top_k == 40
    assert bad.seed is None


def test_entropy_helpers():
    assert entropy_of([1.0]) == 0.0
    assert math.isclose(entropy_of([0.5, 0.5]), math.log(2.0), rel_tol=1e-9)
    assert entropy_of([0.0, 1.0]) == 0.0
    assert approximate_prompt_tokens("") == 0
    assert approximate_prompt_tokens("你好，世界") > 0
    assert approximate_prompt_tokens("hello") >= 1
