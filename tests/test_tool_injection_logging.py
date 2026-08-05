from __future__ import annotations

import logging

import pytest

from headroom.proxy.helpers import log_tool_injection_decision as helper_log_tool_injection_decision
from headroom.proxy.tool_injection_logging import log_tool_injection_decision


def test_logs_tool_injection_decision_without_contents(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("headroom.proxy.test_tool_injection_logging")

    with caplog.at_level(logging.INFO, logger=logger.name):
        log_tool_injection_decision(
            logger=logger,
            provider="anthropic",
            session_id="session-1",
            decision="inject_sticky_replay",
            tool_definition_bytes_count=123,
            request_id="req-1",
        )

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "event=tool_injection_decision" in message
    assert "provider=anthropic" in message
    assert "session_id=session-1" in message
    assert "decision=inject_sticky_replay" in message
    assert "tool_definition_bytes_count=123" in message
    assert "request_id=req-1" in message
    assert "memory_save" not in message
    assert "headroom_retrieve" not in message


def test_helper_wrapper_uses_proxy_logger(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="headroom.proxy"):
        helper_log_tool_injection_decision(
            provider="openai",
            session_id=None,
            decision="skip",
            tool_definition_bytes_count=0,
            request_id=None,
        )

    assert len(caplog.records) == 1
    assert caplog.records[0].name == "headroom.proxy"
    assert "session_id= decision=skip" in caplog.records[0].getMessage()
