"""A failing compression hook is visible once at warning, then quiet at debug."""

from __future__ import annotations

import logging

import pytest

from headroom.proxy.handlers import _hook_failures
from headroom.proxy.handlers._hook_failures import log_hook_failure


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_hook_failures, "_WARNED", set())
    log = logging.getLogger("headroom.proxy")
    handler = _Capture()
    old_level = log.level
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        log.removeHandler(handler)
        log.setLevel(old_level)


def test_first_failure_warns_with_traceback_and_repeats_drop_to_debug(capture) -> None:
    for request_id in ("req-1", "req-2"):
        try:
            raise ValueError("bad bias")
        except ValueError as err:
            log_hook_failure(request_id, "compute_biases", err)

    first, second = capture.records
    assert first.levelno == logging.WARNING
    assert first.exc_info is not None
    assert "[req-1] compute_biases hook failed" in first.getMessage()
    assert "ValueError: bad bias" in first.getMessage()
    assert second.levelno == logging.DEBUG
    assert not second.exc_info
    assert "[req-2]" in second.getMessage()


def test_a_different_stage_or_error_type_warns_again(capture) -> None:
    log_hook_failure("req-1", "pre_compress", ValueError("x"))
    log_hook_failure("req-2", "post_compress", ValueError("x"))
    log_hook_failure("req-3", "pre_compress", KeyError("x"))
    log_hook_failure("req-4", "pre_compress", ValueError("y"))

    levels = [r.levelno for r in capture.records]
    assert levels == [logging.WARNING, logging.WARNING, logging.WARNING, logging.DEBUG]
