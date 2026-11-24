"""Every count that is compared within one request comes from ONE counter (#5037).

``tokens_saved`` is ``tokens_before - tokens_after``. When the two operands are
counted by two independently resolved counters, the subtraction is a
difference of two rulers: it reports savings (or inflation) on a request that
nothing touched.

On ``main`` every provider caches one counter per model, so two lookups usually
return the same object. Nothing in the ``Provider`` contract promises that,
though: a provider that hands out a fallback counter while a vocabulary cannot
load, and the real one once it can (#4990), returns two different rulers to two
lookups in one request. These tests use exactly such a provider, so they pin
the contract (resolve once, pass it through) rather than an accident of
caching.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from headroom.client import HeadroomClient
from headroom.providers.openai import OpenAIProvider
from headroom.tokenizer import Tokenizer
from headroom.tokenizers import EstimatingTokenCounter
from headroom.transforms.pipeline import TransformPipeline

MODEL = "gpt-4o"
MESSAGES: list[dict[str, Any]] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize the release notes for version 2.4. " * 12},
]


class _FlippingProvider(OpenAIProvider):
    """Hands out a different counter on every lookup, like an uncached fallback.

    Odd lookups count ~4 chars per token, even lookups ~2 chars per token, so
    any pairing of two lookups shows up as a large, deterministic gap.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lookups = 0

    def get_token_counter(self, model: str) -> Any:
        self.lookups += 1
        return EstimatingTokenCounter(chars_per_token=4.0 if self.lookups % 2 else 2.0)


def _open_breaker(pipeline: TransformPipeline) -> None:
    pipeline._breaker_threshold = 1
    pipeline._breaker_open_until = time.monotonic() + 600


# ---------------------------------------------------------------------------
# TransformPipeline.apply(token_counter=...)
# ---------------------------------------------------------------------------


def test_pipeline_counts_with_the_counter_the_caller_passes() -> None:
    pipeline = TransformPipeline(transforms=[], provider=_FlippingProvider())
    mine = EstimatingTokenCounter(chars_per_token=3.0)

    result = pipeline.apply(MESSAGES, MODEL, token_counter=mine, model_limit=128_000)

    assert result.tokens_before == mine.count_messages(MESSAGES)
    assert result.tokens_after == mine.count_messages(result.messages)


def test_pipeline_accepts_a_tokenizer_wrapper_too() -> None:
    pipeline = TransformPipeline(transforms=[], provider=_FlippingProvider())
    mine = Tokenizer(EstimatingTokenCounter(chars_per_token=3.0), MODEL)

    result = pipeline.apply(MESSAGES, MODEL, token_counter=mine, model_limit=128_000)

    assert result.tokens_before == mine.count_messages(MESSAGES)


def test_open_breaker_passthrough_uses_the_callers_counter() -> None:
    pipeline = TransformPipeline(transforms=[], provider=_FlippingProvider())
    _open_breaker(pipeline)
    mine = EstimatingTokenCounter(chars_per_token=3.0)

    result = pipeline.apply(MESSAGES, MODEL, token_counter=mine, model_limit=128_000)

    assert result.transforms_applied == ["pipeline:circuit_open"]
    assert result.tokens_before == result.tokens_after == mine.count_messages(MESSAGES)


def test_simulate_forwards_the_callers_counter() -> None:
    pipeline = TransformPipeline(transforms=[], provider=_FlippingProvider())
    mine = EstimatingTokenCounter(chars_per_token=3.0)

    result = pipeline.simulate(MESSAGES, MODEL, token_counter=mine, model_limit=128_000)

    assert result.tokens_before == mine.count_messages(MESSAGES)


def test_token_counter_is_not_forwarded_to_transforms() -> None:
    """The argument is the pipeline's, not a transform kwarg."""
    seen: dict[str, Any] = {}

    class _Spy:
        name = "spy"

        def should_apply(self, messages: Any, tokenizer: Any, **kwargs: Any) -> bool:
            seen.update(kwargs)
            seen["tokenizer"] = tokenizer
            return False

    pipeline = TransformPipeline(transforms=[_Spy()], provider=_FlippingProvider())  # type: ignore[list-item]
    mine = EstimatingTokenCounter(chars_per_token=3.0)

    pipeline.apply(MESSAGES, MODEL, token_counter=mine, model_limit=128_000)

    assert "token_counter" not in seen
    assert seen["tokenizer"].count_messages(MESSAGES) == mine.count_messages(MESSAGES)


# ---------------------------------------------------------------------------
# HeadroomClient: tokens_before is the client's count, tokens_after the
# pipeline's. They must be one ruler.
# ---------------------------------------------------------------------------


def _client(tmp_path: Path) -> HeadroomClient:
    original = MagicMock()
    original.chat.completions.create.return_value = MagicMock()
    return HeadroomClient(
        original_client=original,
        provider=_FlippingProvider(),
        store_url=f"sqlite:///{tmp_path / 'metrics.db'}",
        default_mode="optimize",
        enable_cache_optimizer=False,
    )


def test_sdk_simulate_reports_zero_savings_for_an_untouched_request(tmp_path: Path) -> None:
    client = _client(tmp_path)
    _open_breaker(client._pipeline)

    sim = client.chat.completions.simulate(model=MODEL, messages=MESSAGES)

    assert sim.tokens_before == sim.tokens_after
    assert sim.tokens_saved == 0


def test_sdk_create_records_equal_totals_for_an_untouched_request(tmp_path: Path) -> None:
    client = _client(tmp_path)
    _open_breaker(client._pipeline)
    saved: list[Any] = []
    client._storage.save = saved.append  # type: ignore[method-assign]

    client.chat.completions.create(model=MODEL, messages=MESSAGES)

    (metrics,) = saved
    assert metrics.tokens_input_before == metrics.tokens_input_after


def test_sdk_input_compressed_event_sees_one_ruler(tmp_path: Path) -> None:
    """Extensions get ``tokens_before``/``tokens_after`` before any recount."""
    from headroom.pipeline import PipelineStage

    client = _client(tmp_path)
    _open_breaker(client._pipeline)
    client._storage.save = lambda _m: None  # type: ignore[method-assign]
    seen: list[dict[str, Any]] = []

    class _Spy:
        def on_pipeline_event(self, event: Any) -> None:
            if event.stage == PipelineStage.INPUT_COMPRESSED:
                seen.append(dict(event.metadata))

    client._pipeline_extensions.add(_Spy())

    client.chat.completions.create(model=MODEL, messages=MESSAGES)

    (metadata,) = seen
    assert metadata["tokens_before"] == metadata["tokens_after"]


# ---------------------------------------------------------------------------
# compress(): agent / verify_lossless mode recounts after the pipeline.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode_kwargs", [{"mode": "agent"}, {"verify_lossless": True}])
def test_compress_recount_uses_the_pipelines_counter(
    monkeypatch: pytest.MonkeyPatch, mode_kwargs: dict[str, Any]
) -> None:
    import importlib

    compress_mod = importlib.import_module("headroom.compress")
    from headroom.compress import CompressConfig, compress

    lookups = {"n": 0}

    def _flipping_get_tokenizer(self: TransformPipeline, model: str) -> Tokenizer:
        lookups["n"] += 1
        cpt = 4.0 if lookups["n"] % 2 else 2.0
        return Tokenizer(EstimatingTokenCounter(chars_per_token=cpt), model)

    monkeypatch.setattr(TransformPipeline, "_get_tokenizer", _flipping_get_tokenizer)
    pipeline = TransformPipeline(transforms=[])
    monkeypatch.setattr(compress_mod, "_get_pipeline", lambda: pipeline)
    monkeypatch.setattr(compress_mod, "_get_agent_pipeline", lambda: pipeline)

    result = compress(MESSAGES, model=MODEL, config=CompressConfig(**mode_kwargs))

    # Nothing was compressed (no transforms), so one ruler reports no change.
    assert result.tokens_before == result.tokens_after
    assert result.tokens_saved == 0
