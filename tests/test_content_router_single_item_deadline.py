from __future__ import annotations

import time

import headroom.transforms.kompress_compressor as kc
from headroom.transforms.content_detector import ContentType
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    RouterCompressionResult,
    RoutingDecision,
)
from headroom.transforms.kompress_compressor import KompressCompressor, KompressConfig


class _Tokenizer:
    def count_text(self, content: str) -> int:
        return len(content.split())


def _compression_result(content: str, compressed: str) -> RouterCompressionResult:
    return RouterCompressionResult(
        compressed=compressed,
        original=content,
        strategy_used=CompressionStrategy.TEXT,
        routing_log=[
            RoutingDecision(
                content_type=ContentType.PLAIN_TEXT,
                strategy=CompressionStrategy.TEXT,
                original_tokens=len(content.split()),
                compressed_tokens=len(compressed.split()),
            )
        ],
    )


def _router() -> ContentRouter:
    return ContentRouter(
        ContentRouterConfig(
            protect_recent_code=0,
            protect_analysis_context=False,
            skip_user_messages=False,
        )
    )


def _messages() -> list[dict[str, str]]:
    return [
        {"role": "assistant", "content": "frozen prefix content remains unchanged"},
        {
            "role": "assistant",
            "content": "pending cache miss content takes the inline compression branch today",
        },
    ]


def test_single_cache_miss_fails_open_at_deadline(monkeypatch, caplog):
    router = _router()

    def slow_compress(content, *, context="", bias=1.0):
        time.sleep(0.2)
        return _compression_result(content, "compressed output")

    monkeypatch.setattr(router, "compress", slow_compress)
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "10")

    started = time.perf_counter()
    result = router.apply(
        _messages(),
        _Tokenizer(),
        frozen_message_count=1,
        min_tokens_to_compress=1,
    )

    assert time.perf_counter() - started < 0.12
    assert result.messages[1]["content"] == _messages()[1]["content"]
    assert "failing open via PASSTHROUGH" in caplog.text


def test_single_cache_miss_preserves_under_deadline_output(monkeypatch):
    router = _router()
    monkeypatch.setattr(
        router,
        "compress",
        lambda content, *, context="", bias=1.0: _compression_result(content, "compressed output"),
    )
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "1000")

    result = router.apply(
        _messages(),
        _Tokenizer(),
        frozen_message_count=1,
        min_tokens_to_compress=1,
    )

    assert result.messages[1]["content"] == "compressed output"


def test_single_cache_miss_preserves_disabled_deadline(monkeypatch):
    router = _router()
    monkeypatch.setattr(
        router,
        "compress",
        lambda content, *, context="", bias=1.0: _compression_result(content, "compressed output"),
    )
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "0")

    result = router.apply(
        _messages(),
        _Tokenizer(),
        frozen_message_count=1,
        min_tokens_to_compress=1,
    )

    assert result.messages[1]["content"] == "compressed output"


def test_single_cache_miss_deadline_starts_before_kompress_load(monkeypatch, caplog):
    router = _router()

    class _Encoding(dict):
        def __init__(self, rows: list[list[str]]):
            super().__init__(
                input_ids=[[0] * len(row) for row in rows],
                attention_mask=[[1] * len(row) for row in rows],
            )
            self._rows = rows

        def word_ids(self, batch_index: int = 0):
            return list(range(len(self._rows[batch_index])))

    class _Tokenizer:
        def count_text(self, content: str) -> int:
            return len(content.split())

        def __call__(self, words, **_kwargs):
            rows = words if words and isinstance(words[0], list) else [words]
            return _Encoding(rows)

    class _Model:
        def __init__(self):
            self.calls = 0

        def get_keep_mask(self, input_ids, attention_mask):
            self.calls += 1
            return [[i % 2 == 0 for i in range(len(row))] for row in input_ids]

    model = _Model()
    compressor = KompressCompressor(config=KompressConfig(enable_ccr=False))
    monkeypatch.setattr(compressor, "_should_batch_single_content", lambda *a, **k: False)
    load_state = {"calls": 0}

    def _slow_load(*_args, **_kwargs):
        load_state["calls"] += 1
        time.sleep(0.05)
        return model, _Tokenizer(), "onnx"

    monkeypatch.setattr(kc, "_load_kompress", _slow_load)
    monkeypatch.setattr(
        router,
        "compress",
        lambda content, *, context="", bias=1.0: _compression_result(
            content,
            compressor.compress(content).compressed,
        ),
    )
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "10")

    started = time.perf_counter()
    result = router.apply(
        _messages(),
        _Tokenizer(),
        frozen_message_count=1,
        min_tokens_to_compress=1,
    )
    elapsed = time.perf_counter() - started
    time.sleep(0.1)

    assert elapsed < 0.12
    assert result.messages[1]["content"] == _messages()[1]["content"]
    assert "failing open via PASSTHROUGH" in caplog.text
    assert load_state["calls"] == 1
    assert model.calls == 0
