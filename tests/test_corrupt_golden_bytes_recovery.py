"""Regression tests for corrupt golden bytes fail-open recovery.

Guards against:
- Bug: corrupt (invalid JSON/bytes) golden tool definitions previously raised
  RuntimeError, permanently breaking the session until proxy restart.
- Fix: log at ERROR and recover — skip the corrupt memory tool entry, or
  regenerate a fresh CCR definition — rather than propagating RuntimeError.

Covers both injection sites:
  1. apply_session_sticky_memory_tools (memory tool golden bytes)
  2. apply_session_sticky_ccr_tool (CCR golden bytes)
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import pytest

from headroom.proxy.helpers import (
    _reset_session_ccr_tracker_for_test,
    _reset_session_tool_tracker_for_test,
    apply_session_sticky_ccr_tool,
    apply_session_sticky_memory_tools,
    get_session_ccr_tracker,
    get_session_tool_tracker,
    serialize_tool_definition_canonical,
)

# ---------------------------------------------------------------------------
# Test isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_trackers() -> None:
    _reset_session_tool_tracker_for_test()
    _reset_session_ccr_tracker_for_test()
    yield
    _reset_session_tool_tracker_for_test()
    _reset_session_ccr_tracker_for_test()


@pytest.fixture(autouse=True)
def _enable_headroom_log_propagation() -> Iterator[None]:
    """Re-enable propagation on the headroom logger during tests.

    configure_proxy_logging() calls headroom_logger.propagate = False to prevent
    duplicate writes when the proxy redirects stderr to a log file. In CI the proxy
    may initialise its logging before the test suite runs, leaving propagation
    disabled. pytest's caplog attaches its handler to the root logger, so it only
    sees records that propagate all the way up. This fixture re-enables propagation
    for the duration of each test so caplog captures headroom log records correctly.
    """
    headroom_logger = logging.getLogger("headroom")
    original = headroom_logger.propagate
    headroom_logger.propagate = True
    yield
    headroom_logger.propagate = original


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory_tool_def(name: str) -> dict[str, Any]:
    return {"name": name, "description": "test tool", "input_schema": {"type": "object"}}


def _seed_memory_tool(
    provider: str,
    session_id: str,
    tool_name: str,
    tool_bytes: bytes,
) -> None:
    tracker = get_session_tool_tracker()
    tracker.record_injection(
        provider=provider,
        session_id=session_id,
        tool_name=tool_name,
        tool_definition_bytes=tool_bytes,
    )


def _seed_ccr_done(
    provider: str,
    session_id: str,
    golden_bytes: bytes,
) -> None:
    tracker = get_session_ccr_tracker()
    tracker.record_ccr_done(provider, session_id, golden_bytes)


# ---------------------------------------------------------------------------
# Fix 1a: apply_session_sticky_memory_tools — corrupt memory golden bytes
# ---------------------------------------------------------------------------


class TestCorruptMemoryGoldenBytes:
    """Corrupt memory tool bytes must not raise RuntimeError."""

    def test_corrupt_bytes_does_not_raise(self) -> None:
        """Invalid JSON golden bytes: no RuntimeError, returns normally."""
        session_id = "sess-corrupt-mem-1"
        _seed_memory_tool("anthropic", session_id, "memory_save", b"NOT_VALID_JSON{{{")

        # Must not raise.
        tools_out, was_injected = apply_session_sticky_memory_tools(
            provider="anthropic",
            session_id=session_id,
            request_id="req-1",
            existing_tools=None,
            memory_tools_to_inject=[_make_memory_tool_def("memory_save")],
            inject_this_turn=True,
        )
        # The corrupt entry is skipped; no tool injected from golden bytes.
        assert isinstance(tools_out, list)

    def test_corrupt_bytes_logs_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """Corrupt golden bytes are logged at ERROR level with exc_info."""
        session_id = "sess-corrupt-mem-2"
        _seed_memory_tool("anthropic", session_id, "memory_search", b"\xff\xfe invalid")

        with caplog.at_level(logging.ERROR, logger="headroom.proxy"):
            apply_session_sticky_memory_tools(
                provider="anthropic",
                session_id=session_id,
                request_id="req-2",
                existing_tools=None,
                memory_tools_to_inject=[_make_memory_tool_def("memory_search")],
                inject_this_turn=True,
            )

        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert error_records, "Expected at least one ERROR log for corrupt golden bytes"
        assert any("corrupt" in r.getMessage().lower() for r in error_records)
        # exc_info must be attached so the traceback appears in logs.
        assert any(r.exc_info is not None for r in error_records)

    def test_valid_tool_survives_alongside_corrupt_entry(self) -> None:
        """A valid second tool is still injected even when one entry is corrupt."""
        session_id = "sess-corrupt-mem-3"
        valid_def = _make_memory_tool_def("memory_search")
        valid_bytes = serialize_tool_definition_canonical(valid_def)

        _seed_memory_tool("anthropic", session_id, "memory_save", b"CORRUPT_JSON")
        _seed_memory_tool("anthropic", session_id, "memory_search", valid_bytes)

        tools_out, was_injected = apply_session_sticky_memory_tools(
            provider="anthropic",
            session_id=session_id,
            request_id="req-3",
            existing_tools=None,
            memory_tools_to_inject=[
                _make_memory_tool_def("memory_save"),
                _make_memory_tool_def("memory_search"),
            ],
            inject_this_turn=True,
        )

        # Valid entry must be present.
        names = [t.get("name") for t in tools_out]
        assert "memory_search" in names, "valid tool should survive corrupt sibling"

    def test_unicode_decode_error_handled(self, caplog: pytest.LogCaptureFixture) -> None:
        """UnicodeDecodeError from non-UTF-8 bytes is also handled gracefully."""
        session_id = "sess-corrupt-mem-4"
        # Bytes that are not valid UTF-8.
        _seed_memory_tool("anthropic", session_id, "memory_save", b"\x80\x81\x82\x83")

        with caplog.at_level(logging.ERROR, logger="headroom.proxy"):
            # Must not raise RuntimeError or UnicodeDecodeError.
            apply_session_sticky_memory_tools(
                provider="anthropic",
                session_id=session_id,
                request_id="req-4",
                existing_tools=None,
                memory_tools_to_inject=[_make_memory_tool_def("memory_save")],
                inject_this_turn=True,
            )

        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert error_records


# ---------------------------------------------------------------------------
# Fix 1b: apply_session_sticky_ccr_tool — corrupt CCR golden bytes
# ---------------------------------------------------------------------------


class TestCorruptCcrGoldenBytes:
    """Corrupt CCR tool bytes must not raise RuntimeError; fresh def regenerated."""

    def test_corrupt_bytes_does_not_raise(self) -> None:
        """Invalid JSON in CCR golden bytes: no RuntimeError."""
        session_id = "sess-corrupt-ccr-1"
        _seed_ccr_done("anthropic", session_id, b"NOT_VALID_JSON{{{")

        # Must not raise.
        tools_out, was_injected = apply_session_sticky_ccr_tool(
            provider="anthropic",
            session_id=session_id,
            request_id="req-1",
            existing_tools=None,
            has_compressed_content_this_turn=False,
        )
        assert isinstance(tools_out, list)

    def test_corrupt_bytes_logs_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """Corrupt CCR golden bytes are logged at ERROR level with exc_info."""
        session_id = "sess-corrupt-ccr-2"
        _seed_ccr_done("anthropic", session_id, b"bad bytes {")

        with caplog.at_level(logging.ERROR, logger="headroom.proxy"):
            apply_session_sticky_ccr_tool(
                provider="anthropic",
                session_id=session_id,
                request_id="req-2",
                existing_tools=None,
                has_compressed_content_this_turn=False,
            )

        error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert error_records, "Expected at least one ERROR log for corrupt CCR golden bytes"
        assert any("corrupt" in r.getMessage().lower() for r in error_records)
        assert any(r.exc_info is not None for r in error_records)

    def test_corrupt_bytes_falls_through_to_fresh_definition(self) -> None:
        """After corrupt bytes, a valid fresh CCR tool definition is injected."""
        session_id = "sess-corrupt-ccr-3"
        _seed_ccr_done("anthropic", session_id, b"CORRUPT_JSON_HERE")

        tools_out, was_injected = apply_session_sticky_ccr_tool(
            provider="anthropic",
            session_id=session_id,
            request_id="req-3",
            existing_tools=None,
            has_compressed_content_this_turn=False,
        )

        # A fresh CCR tool definition must have been regenerated and injected.
        assert was_injected is True, "Should still inject a fresh CCR tool after corrupt bytes"
        assert len(tools_out) == 1, "Exactly one CCR tool should be injected"
        # Verify it is a valid tool definition (JSON-serializable with a name).
        tool = tools_out[0]
        name = tool.get("name") or tool.get("function", {}).get("name")
        assert name is not None, "Regenerated tool definition must have a name"

    def test_unicode_decode_error_falls_through_to_fresh_definition(self) -> None:
        """UnicodeDecodeError in CCR golden bytes also triggers fresh definition."""
        session_id = "sess-corrupt-ccr-4"
        # Non-UTF-8 bytes.
        _seed_ccr_done("anthropic", session_id, b"\x80\x81\x82\x83")

        tools_out, was_injected = apply_session_sticky_ccr_tool(
            provider="anthropic",
            session_id=session_id,
            request_id="req-4",
            existing_tools=None,
            has_compressed_content_this_turn=False,
        )

        assert was_injected is True
        assert len(tools_out) == 1

    def test_valid_golden_bytes_still_injected(self) -> None:
        """Sanity check: valid CCR golden bytes still work correctly after fix."""
        from headroom.ccr.tool_injection import create_ccr_tool_definition
        from headroom.proxy.helpers import serialize_tool_definition_canonical

        session_id = "sess-valid-ccr"
        valid_def = create_ccr_tool_definition("anthropic")
        valid_bytes = serialize_tool_definition_canonical(valid_def)
        _seed_ccr_done("anthropic", session_id, valid_bytes)

        tools_out, was_injected = apply_session_sticky_ccr_tool(
            provider="anthropic",
            session_id=session_id,
            request_id="req-valid",
            existing_tools=None,
            has_compressed_content_this_turn=False,
        )

        assert was_injected is True
        assert tools_out == [valid_def]
