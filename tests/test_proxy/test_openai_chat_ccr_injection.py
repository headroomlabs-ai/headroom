"""Streaming chat CCR injection must not advertise an unredeemable tool."""

from headroom.proxy.handlers.openai import _should_inject_openai_chat_ccr_tool


def test_streaming_chat_does_not_inject_ccr_tool() -> None:
    assert _should_inject_openai_chat_ccr_tool(ccr_inject_tool=True, stream=True) is False


def test_non_streaming_chat_still_injects_ccr_tool() -> None:
    assert _should_inject_openai_chat_ccr_tool(ccr_inject_tool=True, stream=False) is True


def test_disabled_ccr_never_injects_tool() -> None:
    assert _should_inject_openai_chat_ccr_tool(ccr_inject_tool=False, stream=False) is False
    assert _should_inject_openai_chat_ccr_tool(ccr_inject_tool=False, stream=True) is False
