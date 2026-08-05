from __future__ import annotations

from headroom.ccr import CCR_TOOL_NAME
from headroom.proxy.handlers.openai import _should_buffer_openai_responses_stream_ccr

_TOOL_TYPE_FUNCTION = "function"
_UNRELATED_TOOL_NAME = "unrelated_tool"


def _ccr_tool() -> dict[str, str]:
    return {"type": _TOOL_TYPE_FUNCTION, "name": CCR_TOOL_NAME}


def _unrelated_tool() -> dict[str, str]:
    return {"type": _TOOL_TYPE_FUNCTION, "name": _UNRELATED_TOOL_NAME}


def _should_buffer(*, tools: list[dict[str, str]], is_chatgpt_auth: bool) -> bool:
    return _should_buffer_openai_responses_stream_ccr(
        stream=True,
        ccr_response_handler_enabled=True,
        tools=tools,
        is_chatgpt_auth=is_chatgpt_auth,
    )


def test_responses_ccr_buffers_streaming_openai_requests() -> None:
    assert _should_buffer(tools=[_ccr_tool()], is_chatgpt_auth=False)


def test_responses_ccr_keeps_chatgpt_oauth_requests_streaming() -> None:
    assert not _should_buffer(tools=[_ccr_tool()], is_chatgpt_auth=True)


def test_responses_ccr_ignores_requests_without_retrieve_tool() -> None:
    assert not _should_buffer(tools=[_unrelated_tool()], is_chatgpt_auth=False)
