"""Tests for cache_control breakpoint diagnostics and log-privacy switches.

Covers the three pieces added for the uncached-tail investigation:
- ``count_cache_breakpoints`` / ``log_cache_breakpoints`` (proxy helpers)
- the ``HEADROOM_LOG_PAYLOAD_PREVIEW`` kill switch (compression store)
- the injection guard that keeps proactive expansion out of breakpointed blocks
"""

from __future__ import annotations

import logging
import os
import stat
from contextlib import contextmanager
from pathlib import Path

import pytest

from headroom import fileperms
from headroom import paths as _paths
from headroom.cache.compression_store import _payload_for_retrieval_log
from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
from headroom.proxy.helpers import (
    _OwnerOnlyRotatingFileHandler,
    count_cache_breakpoints,
    log_cache_breakpoints,
)

_CC = {"cache_control": {"type": "ephemeral"}}


def _claude_code_style_request() -> tuple[list[dict], list[dict], list[dict]]:
    """System/messages/tools shaped like a real Claude Code request."""
    system = [
        {"type": "text", "text": "You are Claude Code."},
        {"type": "text", "text": "project instructions", **_CC},
    ]
    tools = [
        {"name": "Bash", "input_schema": {}},
        {"name": "Read", "input_schema": {}, **_CC},
    ]
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi", **_CC}]},
        {"role": "assistant", "content": [{"type": "text", "text": "ack"}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [{"type": "text", "text": "big output"}],
                    **_CC,
                }
            ],
        },
    ]
    return system, messages, tools


def test_count_cache_breakpoints_counts_all_sections() -> None:
    system, messages, tools = _claude_code_style_request()
    stats = count_cache_breakpoints(system, messages, tools)
    assert stats["system"] == 1
    assert stats["tools"] == 1
    assert stats["messages"] == 2
    assert stats["total"] == 4
    assert stats["message_count"] == 3
    assert stats["last_marker_tail"] == 0  # last message carries a marker


def test_count_cache_breakpoints_counts_nested_tool_result_markers() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [{"type": "text", "text": "out", **_CC}],
                }
            ],
        }
    ]
    stats = count_cache_breakpoints("plain system string", messages, None)
    assert stats["system"] == 0
    assert stats["tools"] == 0
    assert stats["messages"] == 1
    assert stats["last_marker_tail"] == 0


def test_count_cache_breakpoints_tail_tracks_last_marker() -> None:
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "a", **_CC}]},
        {"role": "assistant", "content": [{"type": "text", "text": "b"}]},
        {"role": "user", "content": [{"type": "text", "text": "c"}]},
    ]
    stats = count_cache_breakpoints(None, messages, None)
    assert stats["last_marker_tail"] == 2
    assert count_cache_breakpoints(None, [], None)["last_marker_tail"] == -1


def test_log_cache_breakpoints_warns_on_dropped_marker(caplog) -> None:
    system, messages, tools = _claude_code_style_request()
    inbound = count_cache_breakpoints(system, messages, tools)
    # Transform "lost" the final breakpoint: strip it from the last message.
    stripped = [dict(m) for m in messages]
    stripped[2] = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "compressed"}],
    }
    outbound = count_cache_breakpoints(system, stripped, tools)
    with caplog.at_level(logging.INFO, logger="headroom.proxy"):
        log_cache_breakpoints(request_id="r1", inbound=inbound, outbound=outbound)
    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert "dropped=true" in record.getMessage()
    assert "tail_grew=true" in record.getMessage()


def test_log_cache_breakpoints_info_when_preserved(caplog) -> None:
    system, messages, tools = _claude_code_style_request()
    stats = count_cache_breakpoints(system, messages, tools)
    with caplog.at_level(logging.INFO, logger="headroom.proxy"):
        log_cache_breakpoints(request_id="r1", inbound=stats, outbound=stats)
    [record] = caplog.records
    assert record.levelno == logging.INFO
    assert "dropped=false" in record.getMessage()


def test_payload_preview_disabled_omits_content(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_LOG_PAYLOAD_PREVIEW", "0")
    payload = "secret file contents: api_key=sk-abcdefghijklmnop"
    event = _payload_for_retrieval_log(payload)
    assert event["payload_preview"] == ""
    assert event["payload_preview_chars"] == 0
    assert event["payload_chars"] == len(payload)
    assert event["payload_truncated"] is True


def test_payload_preview_disabled_by_default(monkeypatch) -> None:
    """Unset means off: the log gets byte counts, never the content."""
    monkeypatch.delenv("HEADROOM_LOG_PAYLOAD_PREVIEW", raising=False)
    event = _payload_for_retrieval_log("hello world")
    assert event["payload_preview"] == ""
    assert event["payload_preview_chars"] == 0
    assert event["payload_chars"] == len("hello world")


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_payload_preview_opt_in_values(monkeypatch, value: str) -> None:
    monkeypatch.setenv("HEADROOM_LOG_PAYLOAD_PREVIEW", value)
    assert _payload_for_retrieval_log("hello world")["payload_preview"] == "hello world"


@pytest.mark.parametrize("value", ["", "0", "off", "no", "maybe", "  "])
def test_payload_preview_stays_off_for_anything_else(monkeypatch, value: str) -> None:
    """Only an explicit opt-in turns previews on — a typo must not."""
    monkeypatch.setenv("HEADROOM_LOG_PAYLOAD_PREVIEW", value)
    assert _payload_for_retrieval_log("hello world")["payload_preview"] == ""


def test_append_context_skips_breakpointed_text_block() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "breakpointed", **_CC},
                {"type": "text", "text": "free"},
            ],
        }
    ]
    result = AnthropicHandlerMixin._append_context_to_latest_non_frozen_user_turn(
        messages, "CTX", frozen_message_count=0
    )
    blocks = result[0]["content"]
    assert blocks[0]["text"] == "breakpointed"  # untouched
    assert blocks[1]["text"].endswith("CTX")


def test_append_context_no_eligible_block_returns_unchanged() -> None:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "breakpointed", **_CC}],
        }
    ]
    result = AnthropicHandlerMixin._append_context_to_latest_non_frozen_user_turn(
        messages, "CTX", frozen_message_count=0
    )
    assert result == messages


def test_count_cache_breakpoints_tolerates_malformed_shapes() -> None:
    messages = [
        "not-a-dict",
        {"role": "user", "content": ["scalar-block", {"type": "text", "text": "x", **_CC}]},
        {"role": "user", "content": "plain string"},
    ]
    stats = count_cache_breakpoints("system-as-string", messages, "tools-as-string")
    assert stats["system"] == 0
    assert stats["tools"] == 0
    assert stats["messages"] == 1
    assert stats["message_count"] == 3
    assert stats["last_marker_tail"] == 1

    empty = count_cache_breakpoints(None, None, None)
    assert empty["total"] == 0
    assert empty["message_count"] == 0


# --- the runtime log file itself -------------------------------------------
#
# _payload_for_retrieval_log decides what goes into the record;
# _setup_file_logging decides who can read the file it lands in. Both halves
# of the default-off guarantee are checked against a real log on disk.


@contextmanager
def _proxy_log(tmp_path, monkeypatch, port: int):
    """Point the workspace at *tmp_path*, install the real proxy log handler."""
    from headroom.proxy.helpers import _PROXY_LOG_HANDLER_NAME, _setup_file_logging

    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    headroom_logger = logging.getLogger("headroom")
    before = list(headroom_logger.handlers)
    propagate = headroom_logger.propagate
    try:
        _setup_file_logging(port)
        [handler] = [h for h in headroom_logger.handlers if h.name == _PROXY_LOG_HANDLER_NAME]
        yield Path(handler.baseFilename)
    finally:
        for handler in list(headroom_logger.handlers):
            if handler not in before:
                headroom_logger.removeHandler(handler)
                handler.close()
        headroom_logger.propagate = propagate


def test_runtime_log_holds_no_payload_text_at_default_settings(tmp_path, monkeypatch) -> None:
    """A retrieval on default settings leaves byte counts in the log, not content."""
    from headroom.cache.compression_store import CompressionStore

    monkeypatch.delenv("HEADROOM_LOG_PAYLOAD_PREVIEW", raising=False)
    secret = "BEGIN-CUSTOMER-DATA ssn=123-45-6789 def sekrit(): pass END-CUSTOMER-DATA"

    with _proxy_log(tmp_path, monkeypatch, 18801) as log_path:
        store = CompressionStore(enable_feedback=False)
        assert store.retrieve(store.store(original=secret, compressed="[compressed]")) is not None
        logging.getLogger("headroom").handlers[-1].flush()
        text = log_path.read_text(encoding="utf-8")

    assert "event=headroom_retrieve" in text, "the retrieval was not logged at all"
    assert secret not in text
    assert "123-45-6789" not in text
    assert f'"payload_chars":{len(secret)}' in text
    assert '"payload_preview":""' in text


_posix_only = pytest.mark.skipif(
    not fileperms.OWNER_ONLY_SUPPORTED,
    reason=(
        "asserts POSIX mode bits, which do not control read access on this platform; "
        "the scope of the guarantee is asserted instead by "
        "test_owner_only_support_matches_what_the_platform_can_enforce and "
        "test_no_owner_only_claim_is_made_off_posix"
    ),
)


@pytest.fixture
def predictable_umask():
    """Pin the umask so "would have been world-readable" is not luck.

    Without this the fail-before evidence for these tests depends on whatever
    umask the runner happens to have; 0o022 is the stock developer value that
    makes an unhardened log 0644.
    """
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


@_posix_only
def test_runtime_log_is_owner_only_with_previews_off(
    tmp_path, monkeypatch, predictable_umask
) -> None:
    """The file's permissions are not keyed off the payload-preview switch.

    Previews are one of several sources of request content in this file:
    ``--log-messages`` bodies, wire-debug dumps and query logging land here
    too, each behind its own switch. Hardening only when previews are on left
    every other combination creating a sensitive log at the umask.
    """
    monkeypatch.delenv("HEADROOM_LOG_PAYLOAD_PREVIEW", raising=False)
    with _proxy_log(tmp_path, monkeypatch, 18804) as log_path:
        assert log_path.exists()
        assert stat.S_IMODE(log_path.stat().st_mode) == 0o600


@_posix_only
def test_runtime_log_is_owner_only_when_preview_enabled(tmp_path, monkeypatch) -> None:
    """Opting in to previews hardens the log the previews land in."""
    monkeypatch.setenv("HEADROOM_LOG_PAYLOAD_PREVIEW", "1")
    with _proxy_log(tmp_path, monkeypatch, 18802) as log_path:
        assert log_path.exists()
        assert stat.S_IMODE(log_path.stat().st_mode) == 0o600


@_posix_only
def test_runtime_log_hardening_survives_a_pre_existing_world_readable_log(
    tmp_path, monkeypatch, predictable_umask
) -> None:
    """O_CREAT's mode does not apply to an existing file; the fchmod must."""
    monkeypatch.delenv("HEADROOM_LOG_PAYLOAD_PREVIEW", raising=False)
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    stale = _paths.proxy_log_path(18803)
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("from an older, unhardened run\n", encoding="utf-8")
    stale.chmod(0o644)

    with _proxy_log(tmp_path, monkeypatch, 18803) as log_path:
        assert stat.S_IMODE(log_path.stat().st_mode) == 0o600


@_posix_only
def test_rotated_backups_are_owner_only(tmp_path, monkeypatch, predictable_umask) -> None:
    """Rotation must not launder the mode away.

    ``doRollover`` renames the base file and opens a fresh one; a backup that
    ends up 0644 exposes exactly the content the base file was protecting, and
    keeps exposing it for ``backupCount`` rotations. Exercised through the real
    handler the proxy installs (with ``maxBytes`` shrunk so a rollover is
    reachable in a test), not a hand-built one, so the handler *selection* is
    part of what is asserted.
    """
    monkeypatch.delenv("HEADROOM_LOG_PAYLOAD_PREVIEW", raising=False)
    with _proxy_log(tmp_path, monkeypatch, 18805) as log_path:
        handler = next(
            h for h in logging.getLogger("headroom").handlers if h.name == "headroom.proxy.file"
        )
        handler.maxBytes = 256
        for i in range(60):
            handler.emit(
                logging.LogRecord("headroom", logging.INFO, __file__, i, "x" * 64, None, None)
            )
        handler.flush()
        backups = sorted(log_path.parent.glob(f"{log_path.name}.*"))
        assert backups, "no rollover happened — the test did not exercise the path it claims to"
        for path in [log_path, *backups]:
            assert stat.S_IMODE(path.stat().st_mode) == 0o600, path


@_posix_only
def test_pre_existing_backups_are_tightened_when_the_handler_opens(
    tmp_path, predictable_umask
) -> None:
    """Backups written by an older, unhardened build are still on disk."""
    log_path = tmp_path / "proxy-18806.log"
    stale = tmp_path / "proxy-18806.log.2"
    stale.write_text("payload from before the fix\n", encoding="utf-8")
    stale.chmod(0o644)

    handler = _OwnerOnlyRotatingFileHandler(
        log_path, maxBytes=1024, backupCount=5, encoding="utf-8"
    )
    handler.close()

    assert stat.S_IMODE(stale.stat().st_mode) == 0o600


@pytest.mark.skipif(
    os.name != "posix",
    reason="creating a symlink needs elevation on Windows, and O_NOFOLLOW does not exist there",
)
def test_runtime_log_refuses_a_symlinked_path(tmp_path, monkeypatch) -> None:
    """A planted symlink must not redirect the log — or the mode we set on it."""
    from headroom.proxy.helpers import _PROXY_LOG_HANDLER_NAME, _setup_file_logging

    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    log_path = _paths.proxy_log_path(18807)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    elsewhere = tmp_path / "attacker-readable.log"
    elsewhere.write_text("", encoding="utf-8")
    elsewhere.chmod(0o666)
    log_path.symlink_to(elsewhere)

    headroom_logger = logging.getLogger("headroom")
    before = list(headroom_logger.handlers)
    try:
        _setup_file_logging(18807)
        attached = [h for h in headroom_logger.handlers if h.name == _PROXY_LOG_HANDLER_NAME]
        assert not attached, "logging was wired up through the symlink"
        logging.getLogger("headroom").info("a record that must not be written")
    finally:
        for handler in list(headroom_logger.handlers):
            if handler not in before:
                headroom_logger.removeHandler(handler)
                handler.close()

    assert elsewhere.read_text(encoding="utf-8") == ""
    assert stat.S_IMODE(elsewhere.stat().st_mode) == 0o666, "the symlink target was chmodded"


@pytest.mark.skipif(
    os.name != "posix",
    reason="creating a symlink needs elevation on Windows, and O_NOFOLLOW does not exist there",
)
def test_open_owner_only_fails_closed_on_a_symlink(tmp_path) -> None:
    """The kernel-enforced half of the same refusal, at the open() itself."""
    target = tmp_path / "target"
    target.write_text("", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(target)

    with pytest.raises(OSError):
        fileperms.open_owner_only(link).close()


@_posix_only
def test_jsonl_request_log_is_owner_only(tmp_path, predictable_umask) -> None:
    """``--log-file`` with ``--log-messages`` writes whole bodies to this file."""
    from headroom.proxy.models import RequestLog
    from headroom.proxy.request_logger import RequestLogger

    log_file = tmp_path / "requests.jsonl"
    entry = RequestLog(
        request_id="r1",
        timestamp="2026-09-23T00:00:00Z",
        provider="anthropic",
        model="claude-opus-4-20250514",
        input_tokens_original=10,
        input_tokens_optimized=8,
        output_tokens=2,
        tokens_saved=2,
        savings_percent=20.0,
        optimization_latency_ms=1.0,
        total_latency_ms=2.0,
        tags={},
        cache_hit=False,
        transforms_applied=[],
        request_messages=[{"role": "user", "content": "ssn=123-45-6789"}],
    )
    RequestLogger(log_file=str(log_file), log_full_messages=True).log(entry)

    assert "123-45-6789" in log_file.read_text(encoding="utf-8")
    assert stat.S_IMODE(log_file.stat().st_mode) == 0o600


# --- what the guarantee is, and is not, on Windows -------------------------


def test_owner_only_support_matches_what_the_platform_can_enforce() -> None:
    """The scope is a value in the code, not a claim in a comment.

    POSIX mode bits decide read access; on Windows an NTFS ACL does, and
    ``os.chmod`` there only flips the read-only attribute — a
    ``chmod(0o600)`` succeeds while ``stat.S_IMODE`` still reports ``0666``.
    Python ships no ACL API, so Headroom reports that it cannot make the
    promise instead of making it and not keeping it.
    """
    assert fileperms.OWNER_ONLY_SUPPORTED is (os.name == "posix")


@pytest.mark.skipif(
    fileperms.OWNER_ONLY_SUPPORTED,
    reason="asserts the *absence* of the mode guarantee; only meaningful off POSIX",
)
def test_no_owner_only_claim_is_made_off_posix(tmp_path) -> None:
    """On Windows the handler still logs — it just does not claim 0600."""
    log_path = tmp_path / "proxy-18808.log"
    handler = _OwnerOnlyRotatingFileHandler(
        log_path, maxBytes=1024, backupCount=1, encoding="utf-8"
    )
    handler.close()

    assert log_path.exists(), "logging must keep working where hardening cannot"
    assert fileperms.restrict_path_to_owner(log_path) is False


def test_unsupported_platform_says_so_rather_than_silently_not_protecting(
    tmp_path, monkeypatch
) -> None:
    """Simulates the Windows path on any host, since CI cannot be both.

    A control that quietly does nothing on a supported platform is the thing
    to avoid, so the operator is told once per process.
    """
    from headroom.proxy import helpers as _helpers

    monkeypatch.setattr(fileperms, "OWNER_ONLY_SUPPORTED", False)
    monkeypatch.setattr(_helpers, "_owner_only_warning_emitted", False)
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))

    records: list[logging.LogRecord] = []
    sink = logging.Handler()
    sink.emit = records.append  # type: ignore[method-assign]
    emitter = logging.getLogger(_helpers.logger.name)
    emitter.addHandler(sink)
    headroom_logger = logging.getLogger("headroom")
    before = list(headroom_logger.handlers)
    try:
        _helpers._setup_file_logging(18809)
    finally:
        emitter.removeHandler(sink)
        for handler in list(headroom_logger.handlers):
            if handler not in before:
                headroom_logger.removeHandler(handler)
                handler.close()

    warnings = [r for r in records if r.levelno == logging.WARNING]
    assert warnings, "the unsupported platform was not reported at all"
    message = warnings[0].getMessage()
    assert "owner-only" in message
    assert "proxy-18809.log" in message
    # And only once per process, so it is a notice and not a per-worker flood.
    assert _helpers._owner_only_warning_emitted is True
