import copy

import pytest

from headroom.cache.compression_cache import CompressionCache
from headroom.proxy.session_engine import FREEZE_POLICY_CONFIRMED_CLAMP, prepare_turn


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_recovered_original_bypasses_cached_summary(provider, wrapped):
    name = "mcp_daemon" if wrapped else "headroom_retrieve"
    args = (
        {"action": "call", "selector": "headroom.headroom_retrieve", "args": {"hash": "test"}}
        if wrapped
        else {"hash": "test"}
    )
    if provider == "openai":
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "recover",
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "recover", "content": "Exact original"},
        ]
    else:
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "recover", "name": name, "input": args}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "recover", "content": "Exact original"}
                ],
            },
        ]
    cache = CompressionCache()
    cache.store_compressed(cache.content_hash("Exact original"), "Earlier summary", 1)
    before = copy.deepcopy(messages)
    result = prepare_turn(cache, messages, policy=FREEZE_POLICY_CONFIRMED_CLAMP)
    assert result.pipeline_input == before
    assert messages == before


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_cached_text_does_not_remove_native_images(provider):
    blocks = [
        {"type": "text", "text": "Exact original"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}},
    ]
    message = (
        {"role": "tool", "tool_call_id": "read", "content": blocks}
        if provider == "openai"
        else {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "read", "content": blocks}],
        }
    )
    cache = CompressionCache()
    cache.store_compressed(cache.content_hash("Exact original"), "Earlier summary", 1)
    before = copy.deepcopy(message)
    result = prepare_turn(cache, [message], policy=FREEZE_POLICY_CONFIRMED_CLAMP)
    assert result.pipeline_input == [before]
    assert message == before
