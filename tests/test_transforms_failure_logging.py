"""Compression failure paths log no exception text at any level.

Each path catches the failure and passes content through unchanged. The
WARNING names the exception type and counts, and the DEBUG detail line gives
only exception types and code locations, because the exception text (and its
cause chain) can quote tool output, stored payloads or credentials.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator

import pytest

import headroom.cache.compression_store as compression_store
from headroom.transforms import (
    compression_batches,
    cross_turn_dedup,
    kompress_compressor,
    lossless_compaction,
    smart_crusher,
)
from headroom.transforms.compression_batches import (
    CompressionBatchEntry,
    build_compression_batches,
    compress_batch_with_router,
)
from headroom.transforms.compression_units import CompressionUnit, RoutedCompressionUnit

CONTENT = "customer-tool-output-canary-7f3a"
CREDENTIAL = "sk-test-credential-canary-91bd"


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def capture() -> Iterator[Callable[[logging.Logger], _Capture]]:
    attached: list[tuple[logging.Logger, _Capture, int]] = []

    def attach(logger: logging.Logger) -> _Capture:
        handler = _Capture()
        attached.append((logger, handler, logger.level))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        return handler

    yield attach
    for logger, handler, level in attached:
        logger.removeHandler(handler)
        logger.setLevel(level)


def _boom(*_args: object, **_kwargs: object) -> object:
    try:
        raise ValueError(f"backend rejected {CONTENT} for key {CREDENTIAL}")
    except ValueError as cause:
        raise RuntimeError(f"failed while handling {CONTENT} with {CREDENTIAL}") from cause


def _run_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    def entry(index: int) -> CompressionBatchEntry:
        unit = CompressionUnit(
            text="x" * 150,
            provider="openai",
            endpoint="responses",
            role="tool",
            item_type="local_shell_call_output",
            cache_zone="live",
            mutable=True,
            min_bytes=512,
        )
        return CompressionBatchEntry(
            entry_id=f"u{index}",
            routed=RoutedCompressionUnit(unit=unit, slot=(index, ("output", None))),
        )

    class _Router:
        compress = staticmethod(_boom)

    class _Counter:
        def count_text(self, text: str) -> int:
            return len(text)

    batches, _ = build_compression_batches([entry(i) for i in range(4)], min_batch_bytes=512)
    results = compress_batch_with_router(batches[0], router=_Router(), tokenizer=_Counter())
    assert [result.compressed for _, result in results] == ["x" * 150] * 4


def _run_dedup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cross_turn_dedup, "_longest_match", _boom)
    blocks = [
        cross_turn_dedup.DedupBlock(text="line a\nline b\nline c\nline d", turn=i, protected=False)
        for i in range(2)
    ]
    out, stats = cross_turn_dedup.dedup_blocks(blocks)
    assert out == blocks
    assert stats["error"] is True


def _run_lossless(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lossless_compaction, "collapse_runs", _boom)
    log = "worker 7 retrying connection to upstream\n" * 5
    assert lossless_compaction.compact_lossless(log, "log") == log


def _run_kompress_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compression_store, "get_compression_store", _boom)
    assert kompress_compressor.store_kompress_in_ccr("original words", "compressed", 2) is None


def _run_crusher_mirror(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Store:
        store = staticmethod(_boom)

    class _Rust:
        def ccr_get(self, _hash: str) -> str:
            return "[1, 2, 3]"

    monkeypatch.setattr(compression_store, "get_compression_store", lambda: _Store())
    crusher = smart_crusher.SmartCrusher()
    monkeypatch.setattr(crusher, "_rust", _Rust())
    crusher._mirror_single_hash_to_python_store("a" * 24, "smart_crusher", "", None)


@pytest.mark.parametrize(
    ("logger", "run", "expected"),
    [
        (compression_batches.logger, _run_batch, "Batch compression failed (RuntimeError)"),
        (cross_turn_dedup.logger, _run_dedup, "Cross-turn dedup failed (RuntimeError)"),
        (lossless_compaction.logger, _run_lossless, "Lossless compaction (log) failed"),
        (
            kompress_compressor.logger,
            _run_kompress_store,
            "Kompress CCR store write failed (RuntimeError)",
        ),
        (smart_crusher.logger, _run_crusher_mirror, "CCR mirror: store write failed"),
    ],
    ids=["batch", "dedup", "lossless", "kompress-ccr", "crusher-mirror"],
)
def test_failure_logs_no_exception_text_at_any_level(
    capture: Callable[[logging.Logger], _Capture],
    monkeypatch: pytest.MonkeyPatch,
    logger: logging.Logger,
    run: Callable[[pytest.MonkeyPatch], None],
    expected: str,
) -> None:
    monkeypatch.delenv("HEADROOM_DEBUG_DUMP", raising=False)
    handler = capture(logger)
    run(monkeypatch)

    formatter = logging.Formatter()
    for record in handler.records:
        rendered = formatter.format(record)
        assert CONTENT not in rendered
        assert CREDENTIAL not in rendered

    warnings = [r for r in handler.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert expected in message
    assert "RuntimeError" in message

    details = [r for r in handler.records if r.levelno == logging.DEBUG]
    assert len(details) == 1
    assert "RuntimeError" in details[0].getMessage()
    assert "caused by ValueError" in details[0].getMessage()
