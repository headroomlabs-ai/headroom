"""CCR retrievals through Codex code-mode must reach the model verbatim."""

import json
from types import MethodType, SimpleNamespace

import pytest

from headroom.proxy.handlers.openai import OpenAIHandlerMixin
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    RouterCompressionResult,
)


class TokenCounter:
    def count_text(self, text: str) -> int:
        return len(text.split())


def _handler_with_router(router: ContentRouter) -> OpenAIHandlerMixin:
    handler = OpenAIHandlerMixin()
    handler.openai_pipeline = SimpleNamespace(transforms=[router])
    handler.openai_provider = SimpleNamespace(
        get_token_counter=lambda _model: TokenCounter(),
    )
    return handler


def _lossy_router() -> ContentRouter:
    """Router whose compress() always 'lossy-compresses' any candidate it sees."""

    router = ContentRouter()

    def compress(self, content: str, **_kwargs):
        return RouterCompressionResult(
            compressed="kept words",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )

    router.compress = MethodType(compress, router)
    return router


def _run(handler: OpenAIHandlerMixin, payload: dict):
    return handler._compress_openai_responses_live_text_units_with_router(
        payload,
        model="gpt-5",
        request_id="req_read_protection",
    )


@pytest.mark.parametrize(
    "call_type,name,field",
    [
        ("custom_tool_call", "exec", "input"),
        ("custom_tool_call", "functions.exec", "input"),
        ("function_call", "functions.exec", "arguments"),
    ],
)
@pytest.mark.parametrize(
    "original",
    [
        "\n".join(
            f"Instruction {i}: preserve exact whitespace and punctuation." for i in range(90)
        ),
        "\n".join(f"def function_{i}():\n    return {i} + 1" for i in range(90)),
        json.dumps(
            [{"id": i, "result": "complete", "score": i / 100} for i in range(90)], indent=2
        ),
    ],
    ids=["instructions", "source", "structured"],
)
@pytest.mark.parametrize("as_parts", [False, True])
def test_wrapped_retrieval_stays_verbatim(call_type, name, field, original, as_parts):
    handler = _handler_with_router(_lossy_router())
    output = json.dumps(
        {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "hash": "abc123",
                            "original_content": original,
                            "source": "local",
                        }
                    ),
                }
            ],
            "isError": False,
        }
    )
    if as_parts:
        output = [{"type": "output_text", "text": output}]
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": call_type,
                "call_id": "retrieve",
                "name": name,
                field: (
                    "const result = await tools.mcp__headroom__headroom_retrieve("
                    '{hash: "abc123"}); text(result);'
                ),
            },
            {
                "type": "custom_tool_call_output"
                if call_type == "custom_tool_call"
                else "function_call_output",
                "call_id": "retrieve",
                "output": output,
            },
        ],
    }
    forwarded, *_ = _run(handler, payload)
    assert forwarded["input"][1]["output"] == output


def test_ordinary_exec_remains_compressible():
    handler = _handler_with_router(_lossy_router())
    payload = {
        "input": [
            {
                "type": "custom_tool_call",
                "name": "functions.exec",
                "call_id": "normal",
                "input": 'text(await tools.exec_command({cmd: "pytest"}));',
            },
            {
                "type": "custom_tool_call_output",
                "call_id": "normal",
                "output": "test passed successfully\n" * 200,
            },
        ]
    }
    forwarded, modified, *_ = _run(handler, payload)
    assert modified
    assert forwarded["input"][1]["output"] == "kept words"


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("headroom_retrieve", '{"hash": "abc123"}'),
        ("functions.exec", 'text(await tools.mcp__headroom__headroom_retrieve({hash: "abc123"}));'),
    ],
)
def test_retrieval_is_not_replaced_with_cross_turn_pointer(name, arguments):
    router = _lossy_router()
    router._cross_turn_dedup_enabled = True
    handler = _handler_with_router(router)
    original = "Retrieved original instruction, preserve it verbatim.\n" * 200
    payload = {"input": []}
    for call_id in ("first", "second"):
        payload["input"].extend(
            [
                {"type": "function_call", "name": name, "call_id": call_id, "arguments": arguments},
                {"type": "function_call_output", "call_id": call_id, "output": original},
            ]
        )
    payload["input"].append(
        {
            "type": "function_call_output",
            "call_id": "ordinary",
            "output": "ordinary test output for compression\n" * 100,
        }
    )
    forwarded, *_ = _run(handler, payload)
    assert forwarded["input"][1]["output"] == original
    assert forwarded["input"][3]["output"] == original
