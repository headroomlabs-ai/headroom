"""Compaction memoization must respect the compressor selected by the caller."""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import pytest

from headroom.transforms import thinking_compactor as tc


class Compressor:
    def __init__(self, output):
        self.output = output
        self.calls = 0

    def compress(self, text, **kwargs):
        self.calls += 1
        return SimpleNamespace(compressed=self.output)


@pytest.fixture(autouse=True)
def clear_cache():
    tc._COMPACT_CACHE.clear()
    yield
    tc._COMPACT_CACHE.clear()


def compact(compressor, shape="field"):
    reasoning = "reasoning " * 60
    if shape == "field":
        messages = [{"role": "assistant", "content": "answer", "reasoning_content": reasoning}]
        result, _ = tc.compact_reasoning_openai_chat(
            messages, kompress=compressor, keep_last_turns=0
        )
        return result[0]["reasoning_content"]
    messages = [{"role": "assistant", "content": [{"type": "thinking", "thinking": reasoning}]}]
    result, _ = tc.compact_thinking_to_text(messages, kompress=compressor, keep_last_turns=0)
    return result[0]["content"][0]["text"]


@pytest.mark.parametrize("shape", ["field", "thinking"])
def test_different_compressors_do_not_share_compacted_output(shape):
    first = Compressor("first backend summary")
    second = Compressor("second backend summary")

    assert "first backend summary" in compact(first, shape)
    assert "second backend summary" in compact(second, shape)
    assert first.calls == second.calls == 1


def test_repeated_calls_keep_the_selected_compressors_memo():
    compressor = Compressor("stable summary")
    assert compact(compressor) == compact(compressor)
    assert compressor.calls == 1


def test_cache_does_not_keep_unused_compressors_alive():
    compressor = Compressor("summary")
    compact(compressor)
    reference = weakref.ref(compressor)
    del compressor
    gc.collect()
    assert reference() is None


def test_compressor_without_weakref_support_still_works():
    class SlottedCompressor:
        __slots__ = ()

        def compress(self, text, **kwargs):
            return SimpleNamespace(compressed="slotted summary")

    compressor = SlottedCompressor()
    assert compact(compressor) == "slotted summary"
    assert compact(compressor) == "slotted summary"


def test_cache_limit_is_shared_across_compressors(monkeypatch):
    monkeypatch.setattr(tc, "_CACHE_CAP", 1)
    first, second = Compressor("first summary"), Compressor("second summary")
    compact(first)
    compact(second)
    assert len(tc._COMPACT_CACHE) == 1
    assert compact(first) == "first summary"
    assert first.calls == 2
