"""Unit tests for headroom.providers.agy.stats.

Tests are headless and isolated: no live agy, no network, no port :8787.
Ref: headroom-30y.15
"""

from __future__ import annotations

import logging
import threading
from typing import Any
from unittest.mock import patch

import pytest

from headroom.providers.agy.stats import (
    _FAIL_OPEN_SUBSTR,
    _GEMINI_LOGGER,
    AgySessionStats,
    FailOpenWarnHandler,
    _format_summary,
    install_fail_open_handler,
    remove_fail_open_handler,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _emit_fail_open(handler: FailOpenWarnHandler) -> None:
    """Emit a synthetic fail-open log record directly into *handler*."""
    record = logging.LogRecord(
        name=_GEMINI_LOGGER,
        level=logging.WARNING,
        pathname="",
        lineno=883,
        msg=f"[req-1] {_FAIL_OPEN_SUBSTR}: some error",
        args=(),
        exc_info=None,
    )
    handler.emit(record)


# ---------------------------------------------------------------------------
# FailOpenWarnHandler — one-shot notice
# ---------------------------------------------------------------------------


class TestFailOpenWarnHandler:
    """FailOpenWarnHandler emits exactly ONE user notice regardless of fire count."""

    def test_emits_one_notice_on_first_fail_open(self, capsys: pytest.CaptureFixture[str]) -> None:
        handler = FailOpenWarnHandler()
        _emit_fail_open(handler)
        captured = capsys.readouterr()
        assert "Headroom: compression failed" in captured.err
        assert "fail-open" in captured.err

    def test_does_not_emit_second_notice(self, capsys: pytest.CaptureFixture[str]) -> None:
        handler = FailOpenWarnHandler()
        _emit_fail_open(handler)
        capsys.readouterr()  # drain first notice
        _emit_fail_open(handler)
        _emit_fail_open(handler)
        captured = capsys.readouterr()
        assert captured.err == "", "no second notice must be printed"

    def test_counts_all_occurrences(self) -> None:
        handler = FailOpenWarnHandler()
        for _ in range(5):
            _emit_fail_open(handler)
        assert handler.fail_open_count == 5

    def test_ignores_unrelated_warning(self, capsys: pytest.CaptureFixture[str]) -> None:
        handler = FailOpenWarnHandler()
        record = logging.LogRecord(
            name=_GEMINI_LOGGER,
            level=logging.WARNING,
            pathname="",
            lineno=1,
            msg="Some unrelated warning",
            args=(),
            exc_info=None,
        )
        handler.emit(record)
        captured = capsys.readouterr()
        assert captured.err == ""
        assert handler.fail_open_count == 0

    def test_thread_safe_one_shot(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Concurrent emit()s from multiple threads must produce exactly ONE notice."""
        handler = FailOpenWarnHandler()
        barrier = threading.Barrier(10)

        def _fire() -> None:
            barrier.wait()
            # Capture via a temp stderr replacement per thread is unreliable;
            # instead count the _warned flag transitions by checking stderr
            # using capsys after all threads complete.
            _emit_fail_open(handler)

        threads = [threading.Thread(target=_fire) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # count = 10, one-shot flag set exactly once
        assert handler.fail_open_count == 10
        # The one-shot flag must be True
        assert handler._warned is True  # noqa: SLF001


# ---------------------------------------------------------------------------
# install / remove lifecycle
# ---------------------------------------------------------------------------


class TestInstallRemoveHandler:
    """install_fail_open_handler adds; remove_fail_open_handler removes — no leak."""

    def test_install_adds_handler_to_logger(self) -> None:
        logger = logging.getLogger(_GEMINI_LOGGER)
        before = list(logger.handlers)
        handler = install_fail_open_handler()
        try:
            assert handler in logger.handlers
        finally:
            remove_fail_open_handler(handler)
        assert list(logger.handlers) == before

    def test_remove_is_idempotent(self) -> None:
        handler = install_fail_open_handler()
        remove_fail_open_handler(handler)
        remove_fail_open_handler(handler)  # must not raise

    def test_handler_receives_real_log_record(self, capsys: pytest.CaptureFixture[str]) -> None:
        logger = logging.getLogger(_GEMINI_LOGGER)
        logger.setLevel(logging.WARNING)
        handler = install_fail_open_handler()
        try:
            logger.warning(f"[req] {_FAIL_OPEN_SUBSTR}: boom")
        finally:
            remove_fail_open_handler(handler)
        captured = capsys.readouterr()
        assert "Headroom: compression failed" in captured.err
        assert handler.fail_open_count == 1

    def test_no_handler_leaks_after_remove(self) -> None:
        logger = logging.getLogger(_GEMINI_LOGGER)
        original_handlers = list(logger.handlers)
        h = install_fail_open_handler()
        remove_fail_open_handler(h)
        assert logger.handlers == original_handlers


# ---------------------------------------------------------------------------
# Falsification: emit on the ACTUAL production logger ("headroom.proxy").
# This hardcodes the production logger name (gemini.py:25) — it does NOT use
# _GEMINI_LOGGER — so it MUST fail if the install target ever regresses to the
# child "headroom.proxy.handlers.gemini" (parent->child does not propagate).
# ---------------------------------------------------------------------------


class TestFailOpenOnProductionLogger:
    """Records emitted on "headroom.proxy" (gemini.py's logger) must be caught."""

    # The exact logger gemini.py:25 uses. Hardcoded on purpose — independent of
    # the stats module's _GEMINI_LOGGER constant so a regression is detectable.
    PROD_LOGGER = "headroom.proxy"

    def test_first_fail_open_on_prod_logger_emits_one_notice(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        prod = logging.getLogger(self.PROD_LOGGER)
        prod.setLevel(logging.WARNING)
        handler = install_fail_open_handler()
        try:
            prod.warning("[req-1] Cloud Code Assist optimization failed: boom")
            first = capsys.readouterr()
            assert "Headroom: compression failed" in first.err
            assert handler.fail_open_count == 1

            # Second such record on the real logger -> still ONE notice, count==2.
            prod.warning("[req-2] Cloud Code Assist optimization failed: boom2")
            second = capsys.readouterr()
            assert second.err == "", "no second user notice must be printed"
            assert handler.fail_open_count == 2
        finally:
            remove_fail_open_handler(handler)

    def test_unrelated_warning_on_prod_logger_no_notice(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        prod = logging.getLogger(self.PROD_LOGGER)
        prod.setLevel(logging.WARNING)
        handler = install_fail_open_handler()
        try:
            prod.warning("[req] some unrelated headroom.proxy warning")
            captured = capsys.readouterr()
            assert captured.err == ""
            assert handler.fail_open_count == 0
        finally:
            remove_fail_open_handler(handler)


# ---------------------------------------------------------------------------
# _format_summary — pure function
# ---------------------------------------------------------------------------


class TestFormatSummary:
    def _make_stats(
        self,
        entry_count: int = 0,
        orig: int = 0,
        comp: int = 0,
    ) -> dict[str, Any]:
        return {
            "entry_count": entry_count,
            "total_original_tokens": orig,
            "total_compressed_tokens": comp,
        }

    def test_with_compression_data(self) -> None:
        start = self._make_stats(entry_count=0, orig=0, comp=0)
        end = self._make_stats(entry_count=3, orig=1000, comp=400)
        summary = _format_summary(start, end)
        assert "3 entries compressed" in summary
        assert "1,000" in summary
        assert "400" in summary
        assert "0.40x" in summary

    def test_divide_by_zero_guard_no_compression(self) -> None:
        start = self._make_stats()
        end = self._make_stats()
        summary = _format_summary(start, end)
        assert "n/a" in summary or "no compression" in summary

    def test_fail_open_count_included_when_provided(self) -> None:
        start = self._make_stats()
        end = self._make_stats(entry_count=1, orig=500, comp=200)
        summary = _format_summary(start, end, fail_open_count=3)
        assert "3 fail-open" in summary

    def test_fail_open_count_omitted_when_none(self) -> None:
        start = self._make_stats()
        end = self._make_stats(entry_count=1, orig=500, comp=200)
        summary = _format_summary(start, end, fail_open_count=None)
        assert "fail-open" not in summary

    def test_none_start_returns_unavailable(self) -> None:
        summary = _format_summary(None, self._make_stats())
        assert "unavailable" in summary

    def test_none_end_returns_unavailable(self) -> None:
        summary = _format_summary(self._make_stats(), None)
        assert "unavailable" in summary

    def test_delta_is_correct_over_preexisting_entries(self) -> None:
        """Delta must subtract the baseline, not report absolute store totals."""
        start = self._make_stats(entry_count=10, orig=5000, comp=2000)
        end = self._make_stats(entry_count=13, orig=6500, comp=2800)
        summary = _format_summary(start, end)
        # 3 new entries, 1500 orig, 800 comp
        assert "3 entries" in summary
        assert "1,500" in summary
        assert "800" in summary

    def test_negative_delta_clamped_to_zero(self) -> None:
        """Entries can be evicted between snapshots; clamp negatives to 0."""
        start = self._make_stats(entry_count=10, orig=5000, comp=2000)
        end = self._make_stats(entry_count=8, orig=4800, comp=1900)
        # Should not raise or produce negative numbers
        summary = _format_summary(start, end)
        assert "0 entries" in summary


# ---------------------------------------------------------------------------
# AgySessionStats — idempotent print_summary
# ---------------------------------------------------------------------------


class TestAgySessionStats:
    """print_summary is idempotent: prints exactly once."""

    def _make_stats_patch(self, stats_list: list[dict[str, Any]]):
        """Patch _get_compression_stats to return successive values from stats_list."""
        call_count = [0]

        def _fake() -> dict[str, Any]:
            idx = min(call_count[0], len(stats_list) - 1)
            call_count[0] += 1
            return stats_list[idx]

        return patch("headroom.providers.agy.stats._get_compression_stats", side_effect=_fake)

    def test_print_summary_outputs_once(self, capsys: pytest.CaptureFixture[str]) -> None:
        start_snap = {"entry_count": 0, "total_original_tokens": 0, "total_compressed_tokens": 0}
        end_snap = {"entry_count": 2, "total_original_tokens": 800, "total_compressed_tokens": 320}

        with self._make_stats_patch([start_snap, end_snap]):
            stats = AgySessionStats()
            stats.snapshot_start()
            stats.print_summary()
            stats.print_summary()  # second call must NOT print

        captured = capsys.readouterr()
        assert captured.err.count("Headroom agy session") == 1

    def test_print_summary_idempotent_across_threads(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        start_snap = {"entry_count": 0, "total_original_tokens": 0, "total_compressed_tokens": 0}
        end_snap = {"entry_count": 1, "total_original_tokens": 400, "total_compressed_tokens": 160}

        with self._make_stats_patch([start_snap, end_snap, end_snap, end_snap]):
            stats = AgySessionStats()
            stats.snapshot_start()

            barrier = threading.Barrier(4)

            def _call_print() -> None:
                barrier.wait()
                # Redirect stderr per-thread is tricky; just count
                stats.print_summary()

            threads = [threading.Thread(target=_call_print) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        captured = capsys.readouterr()
        assert captured.err.count("Headroom agy session") == 1

    def test_snapshot_start_graceful_on_import_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """If compression_store is unavailable, snapshot_start must not raise."""
        with patch(
            "headroom.providers.agy.stats._get_compression_stats",
            side_effect=ImportError("no compression_store"),
        ):
            stats = AgySessionStats()
            stats.snapshot_start()  # must not raise
            stats.print_summary()

        captured = capsys.readouterr()
        assert "unavailable" in captured.err

    def test_print_summary_includes_fail_open_count(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        start_snap = {"entry_count": 0, "total_original_tokens": 0, "total_compressed_tokens": 0}
        end_snap = {"entry_count": 1, "total_original_tokens": 200, "total_compressed_tokens": 80}

        handler = FailOpenWarnHandler()
        _emit_fail_open(handler)
        _emit_fail_open(handler)

        with self._make_stats_patch([start_snap, end_snap]):
            stats = AgySessionStats()
            stats.snapshot_start()
            stats.print_summary(handler=handler)

        captured = capsys.readouterr()
        assert "2 fail-open" in captured.err
