"""Tests for the payload-safe log helpers in ``headroom.log_safety``."""

from __future__ import annotations

import logging
import os

import pytest

from headroom import log_safety
from headroom.log_safety import WarnOnce, describe_exception, redact_url, safe_id

CANARY = "sk-live-CANARY tool output: patient record 42"


def _raise_chain() -> BaseException:
    try:
        try:
            raise KeyError(CANARY)
        except KeyError as inner:
            raise ValueError(f"could not parse {CANARY}") from inner
    except ValueError as exc:
        return exc


def test_exception_description_names_types_and_places_but_not_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HEADROOM_DEBUG_DUMP", raising=False)
    text = describe_exception(_raise_chain())

    assert "CANARY" not in text
    assert text.startswith("ValueError at ")
    assert "test_log_safety.py:" in text
    assert "in _raise_chain" in text
    assert "; caused by KeyError at " in text


def test_oserror_keeps_errno_but_not_its_filename(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADROOM_DEBUG_DUMP", raising=False)
    text = describe_exception(PermissionError(13, "Permission denied", f"/tmp/{CANARY}"))

    assert text == "PermissionError [Errno 13] Permission denied"


def test_oserror_free_text_strerror_is_replaced_by_the_system_text() -> None:
    text = describe_exception(OSError(2, f"token={CANARY}"))

    assert "CANARY" not in text
    assert text == f"FileNotFoundError [Errno 2] {os.strerror(2)}"


def test_content_opt_in_gives_the_full_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_DEBUG_DUMP", "full")

    text = describe_exception(_raise_chain())

    assert "Traceback" in text
    assert CANARY in text


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://user:hunter2@proxy.example:8443/v1?key=abc#frag",
            "https://proxy.example:8443/v1?<redacted>",
        ),
        ("http://127.0.0.1:8787/stats", "http://127.0.0.1:8787/stats"),
        ("http://[::1]:8787/", "http://[::1]:8787/"),
        ("http://[::1", "<unparseable url>"),
    ],
)
def test_redact_url_drops_userinfo_and_query(url: str, expected: str) -> None:
    assert redact_url(url) == expected


def test_safe_id_escapes_newlines_and_bounds_length() -> None:
    assert safe_id("m1\nWARNING forged") == "'m1\\nWARNING forged'"
    long = safe_id("x" * 10_000)
    assert len(long) < 120
    assert long.endswith("chars)")


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_warn_once_stays_bounded_across_repeated_passes(caplog: pytest.LogCaptureFixture) -> None:
    log = logging.getLogger("headroom.test_log_safety.bounded")
    guard = WarnOnce(limit=64, what="unwritable ledgers")

    with caplog.at_level(logging.INFO, logger=log.name):
        for _ in range(3):
            for path in range(65):
                if guard.first(path, log):
                    log.warning("ledger %d failed", path)

    messages = _warnings(caplog)
    assert len(messages) == 65  # 64 keys plus one overflow notice, not 65 per pass
    assert messages[-1] == (
        "More than 64 distinct unwritable ledgers; further ones are logged at debug only "
        "for the next 60 minutes"
    )


def test_warn_once_keeps_the_key_while_warnings_are_disabled() -> None:
    log = logging.getLogger("headroom.test_log_safety.level")
    guard = WarnOnce(limit=4, what="failures")
    log.setLevel(logging.ERROR)
    try:
        assert guard.first("k", log) is False
        log.setLevel(logging.INFO)
        assert guard.first("k", log) is True
        assert guard.first("k", log) is False
    finally:
        log.setLevel(logging.NOTSET)


def test_forget_rearms_a_key() -> None:
    log = logging.getLogger("headroom.test_log_safety.forget")
    guard = WarnOnce(limit=4, what="failures")

    assert guard.first("path", log) is True
    guard.forget("path")
    assert guard.first("path", log) is True


def test_frame_paths_are_package_relative_for_headroom_and_bare_otherwise() -> None:
    package_dir = os.path.dirname(log_safety.__file__)

    assert log_safety._short_path(os.path.join(package_dir, "proxy", "server.py")) == (
        "headroom/proxy/server.py"
    )
    assert log_safety._short_path("/home/runner/work/headroom/headroom/tests/test_x.py") == (
        "test_x.py"
    )
    assert log_safety._short_path("/repo/headroom/.venv/lib/site-packages/httpx/_client.py") == (
        "_client.py"
    )


def test_raise_from_none_hides_the_suppressed_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADROOM_DEBUG_DUMP", raising=False)
    try:
        try:
            raise KeyError(CANARY)
        except KeyError:
            raise ValueError("bad") from None
    except ValueError as exc:
        text = describe_exception(exc)

    assert "KeyError" not in text


def test_a_cause_cycle_is_described_once() -> None:
    first = ValueError(CANARY)
    second = KeyError(CANARY)
    first.__cause__ = second
    second.__cause__ = first

    assert describe_exception(first) == "ValueError; caused by KeyError"


def test_forgetting_after_overflow_does_not_reopen_the_cap(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log = logging.getLogger("headroom.test_log_safety.reopen")
    guard = WarnOnce(limit=2, what="failures")

    with caplog.at_level(logging.INFO, logger=log.name):
        assert guard.first("a", log) and guard.first("b", log)
        assert guard.first("c", log) is False  # overflow notice
        for cycle in range(5):
            guard.forget("a")
            assert guard.first(f"new-{cycle}", log) is False

    assert len(_warnings(caplog)) == 1


def test_warn_once_starts_over_after_the_overflow_window(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = logging.getLogger("headroom.test_log_safety.window")
    guard = WarnOnce(limit=2, what="hook failures", window_seconds=600)
    clock = [1000.0]
    monkeypatch.setattr(log_safety.time, "monotonic", lambda: clock[0])

    results = []
    with caplog.at_level(logging.INFO, logger=log.name):
        for key, advance in (("a", 0), ("b", 0), ("c", 0), ("d", 599), ("new-hook", 1)):
            clock[0] += advance
            results.append(guard.first(key, log))

    # a and b warn, c triggers the overflow notice, d is inside the window,
    # and a newly broken key after the window warns again.
    assert results == [True, True, False, False, True]
    assert len(_warnings(caplog)) == 1


def test_huge_errno_does_not_break_the_description() -> None:
    exc = OSError()
    exc.errno = 2**63

    assert describe_exception(exc) == f"OSError [Errno {2**63}] unknown error"
