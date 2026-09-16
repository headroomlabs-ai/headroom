import json

from headroom.providers.ccr import CCR_TOOL_NAME, get_ccr_adapter


def test_ccr_adapters_create_provider_native_tool_definitions() -> None:
    anthropic = get_ccr_adapter("anthropic").tool_definition()
    openai = get_ccr_adapter("openai").tool_definition()
    google = get_ccr_adapter("google").tool_definition()

    assert anthropic["name"] == CCR_TOOL_NAME
    assert "input_schema" in anthropic

    assert openai["function"]["name"] == CCR_TOOL_NAME
    assert openai["type"] == "function"

    assert google["name"] == CCR_TOOL_NAME
    assert "parameters" in google


def test_ccr_adapters_parse_provider_native_tool_calls() -> None:
    valid_hash = "abcdef123456abcdef123456"

    assert get_ccr_adapter("anthropic").parse_tool_call(
        {"name": CCR_TOOL_NAME, "input": {"hash": valid_hash, "query": "needle"}}
    ) == (CCR_TOOL_NAME, {"hash": valid_hash, "query": "needle"})

    assert get_ccr_adapter("openai").parse_tool_call(
        {
            "function": {
                "name": CCR_TOOL_NAME,
                "arguments": json.dumps({"hash": valid_hash}),
            }
        }
    ) == (CCR_TOOL_NAME, {"hash": valid_hash})

    assert get_ccr_adapter("google").parse_tool_call(
        {"functionCall": {"name": CCR_TOOL_NAME, "args": {"hash": valid_hash}}}
    ) == (CCR_TOOL_NAME, {"hash": valid_hash})


def test_google_ccr_adapter_converts_standard_messages_to_contents() -> None:
    adapter = get_ccr_adapter("google")

    contents = adapter.messages_to_contents(  # type: ignore[attr-defined]
        [
            {"role": "system", "content": "ignore here"},
            {"role": "assistant", "content": "model text"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": "done",
                    },
                ],
            },
        ]
    )

    assert contents == [
        {"role": "model", "parts": [{"text": "model text"}]},
        {
            "role": "user",
            "parts": [
                {"text": "hello"},
                {
                    "functionResponse": {
                        "name": "call_1",
                        "response": {"content": "done"},
                    }
                },
            ],
        },
    ]
