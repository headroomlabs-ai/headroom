from __future__ import annotations

from collections.abc import Callable
from types import MethodType, SimpleNamespace
from typing import Any, cast

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
    handler_any = cast(Any, handler)
    handler_any.openai_pipeline = SimpleNamespace(transforms=[router])
    handler_any.openai_provider = SimpleNamespace(
        get_token_counter=lambda _model: TokenCounter(),
    )
    return handler


def _bind_router_compress(
    router: ContentRouter,
    compress: Callable[..., RouterCompressionResult],
) -> None:
    cast(Any, router).compress = MethodType(compress, router)


def test_openai_responses_adapter_compresses_only_live_text_slots() -> None:
    router = ContentRouter()

    def compress(self: Any, content: str, **_kwargs: Any) -> RouterCompressionResult:
        return RouterCompressionResult(
            compressed="kept words",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )

    _bind_router_compress(router, compress)
    handler = _handler_with_router(router)
    long_text = " ".join(f"word{i}" for i in range(2000))
    payload = {
        "model": "gpt-5",
        "input": [
            {"type": "reasoning", "encrypted_content": long_text},
            {"type": "function_call", "arguments": long_text},
            {"type": "local_shell_call_output", "call_id": "c1", "output": long_text},
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": long_text}],
            },
        ],
    }

    new_payload, modified, saved, transforms, units_by_category, strategy_chain, _attempted = (
        handler._compress_openai_responses_live_text_units_with_router(
            payload,
            model="gpt-5",
            request_id="req_test",
        )
    )

    assert modified is True
    assert saved > 0
    assert new_payload["input"][0]["encrypted_content"] == long_text
    assert new_payload["input"][1]["arguments"] == long_text
    assert new_payload["input"][2]["output"] == "kept words"
    assert new_payload["input"][3]["content"][0]["text"] == long_text
    assert any(t.startswith("router:openai:responses:") for t in transforms)
    assert units_by_category == {"applied": 1}
    assert strategy_chain == []


def test_openai_responses_adapter_compresses_custom_tool_call_output() -> None:
    router = ContentRouter()

    def compress(self: Any, content: str, **_kwargs: Any) -> RouterCompressionResult:
        return RouterCompressionResult(
            compressed="custom output summary",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )

    _bind_router_compress(router, compress)
    handler = _handler_with_router(router)
    long_text = " ".join(f"word{i}" for i in range(2000))
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "custom_tool_call_output",
                "call_id": "c1",
                "output": long_text,
            }
        ],
    }

    new_payload, modified, saved, transforms, units_by_category, strategy_chain, _attempted = (
        handler._compress_openai_responses_live_text_units_with_router(
            payload,
            model="gpt-5",
            request_id="req_test",
        )
    )

    assert modified is True
    assert saved > 0
    assert new_payload["input"][0]["output"] == "custom output summary"
    assert "router:openai:responses:custom_tool_call_output:kompress" in transforms
    assert units_by_category == {"applied": 1}
    assert strategy_chain == []


def test_openai_responses_adapter_accepts_empty_input_list() -> None:
    router = ContentRouter()
    handler = _handler_with_router(router)
    payload = {"model": "gpt-5", "input": [], "tools": []}

    new_payload, modified, saved, transforms, units_by_category, strategy_chain, _attempted = (
        handler._compress_openai_responses_live_text_units_with_router(
            payload,
            model="gpt-5",
            request_id="req_test",
        )
    )

    assert new_payload == payload
    assert modified is False
    assert saved == 0
    assert transforms == []
    assert units_by_category == {}
    assert strategy_chain == []


def test_openai_responses_adapter_preserves_headroom_retrieve_outputs() -> None:
    router = ContentRouter()

    def compress(self: Any, content: str, **_kwargs: Any) -> RouterCompressionResult:
        return RouterCompressionResult(
            compressed="compressed retrieve output",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )

    _bind_router_compress(router, compress)
    handler = _handler_with_router(router)
    retrieved = " ".join(f"retrieved{i}" for i in range(2000))
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_retrieve",
                "name": "mcp__headroom__headroom_retrieve",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_retrieve",
                "output": retrieved,
            },
        ],
    }

    new_payload, modified, saved, transforms, units_by_category, strategy_chain, _attempted = (
        handler._compress_openai_responses_live_text_units_with_router(
            payload,
            model="gpt-5",
            request_id="req_test",
        )
    )

    assert modified is False
    assert saved == 0
    assert transforms == []
    assert new_payload == payload
    assert units_by_category == {}
    assert strategy_chain == []


def test_openai_responses_adapter_keeps_small_and_opaque_items() -> None:
    router = ContentRouter()

    def compress(self: Any, content: str, **_kwargs: Any) -> RouterCompressionResult:
        return RouterCompressionResult(
            compressed="short",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )

    _bind_router_compress(router, compress)
    handler = _handler_with_router(router)
    payload = {
        "model": "gpt-5",
        "input": [
            {"type": "local_shell_call_output", "call_id": "c1", "output": "too small"},
            {"type": "compaction", "encrypted_content": " ".join(["secret"] * 200)},
        ],
    }

    new_payload, modified, saved, transforms, units_by_category, strategy_chain, _attempted = (
        handler._compress_openai_responses_live_text_units_with_router(
            payload,
            model="gpt-5",
            request_id="req_test",
        )
    )

    assert modified is False
    assert saved == 0
    assert transforms == []
    assert new_payload == payload
    assert units_by_category == {"size_floor": 1}
    assert strategy_chain == []


def test_openai_responses_payload_routes_through_content_router_without_rust(
    monkeypatch: Any,
) -> None:
    router = ContentRouter()

    def compress(self: Any, content: str, **_kwargs: Any) -> RouterCompressionResult:
        return RouterCompressionResult(
            compressed="compressed fallback",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )

    _bind_router_compress(router, compress)
    handler = _handler_with_router(router)

    import headroom._core as core

    def rust_must_not_run(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Responses payload compression should route through ContentRouter")

    monkeypatch.setattr(
        core,
        "compress_openai_responses_live_zone",
        rust_must_not_run,
        raising=False,
    )

    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "local_shell_call_output",
                "call_id": "c1",
                "output": " ".join(f"word{i}" for i in range(2000)),
            }
        ],
    }

    new_payload, modified, saved, transforms, reason, _, _, _ = (
        handler._compress_openai_responses_payload(
            payload,
            model="gpt-5",
            request_id="req_router",
        )
    )

    assert modified is True
    assert saved > 0
    assert reason is None
    assert new_payload["input"][0]["output"] == "compressed fallback"
    assert any(t.startswith("router:openai:responses:") for t in transforms)


def test_openai_responses_adapter_compresses_historical_message_text_but_not_current_user() -> None:
    router = ContentRouter()

    def compress(self: Any, content: str, **_kwargs: Any) -> RouterCompressionResult:
        return RouterCompressionResult(
            compressed="compressed history",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )

    _bind_router_compress(router, compress)
    handler = _handler_with_router(router)
    long_text = " ".join(f"word{i}" for i in range(2000))
    current_text = " ".join(f"current{i}" for i in range(2000))
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": long_text}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": long_text}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": current_text}],
            },
        ],
    }

    new_payload, modified, saved, transforms, units_by_category, strategy_chain, _attempted = (
        handler._compress_openai_responses_live_text_units_with_router(
            payload, model="gpt-5", request_id="req_history"
        )
    )

    assert modified is True
    assert saved > 0
    assert new_payload["input"][0]["content"][0]["text"] == "compressed history"
    assert new_payload["input"][1]["content"][0]["text"] == "compressed history"
    assert new_payload["input"][2]["content"][0]["text"] == current_text
    assert units_by_category == {"applied": 2}
    assert strategy_chain == []
    assert any(t.startswith("router:openai:responses:message:") for t in transforms)


def disabled_test_openai_responses_adapter_keeps_message_floor_high_but_allows_tool_outputs() -> (
    None
):
    router = ContentRouter()

    def compress(self: Any, content: str, **_kwargs: Any) -> RouterCompressionResult:
        return RouterCompressionResult(
            compressed="compressed medium",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )

    _bind_router_compress(router, compress)
    handler = _handler_with_router(router)
    medium_text = " ".join(f"word{i}" for i in range(700))
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "local_shell_call_output",
                "call_id": "c1",
                "output": medium_text,
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": medium_text}],
            },
        ],
    }

    new_payload, modified, saved, transforms, units_by_category, strategy_chain, _attempted = (
        handler._compress_openai_responses_live_text_units_with_router(
            payload, model="gpt-5", request_id="req_floor"
        )
    )

    assert modified is True
    assert saved > 0
    assert new_payload["input"][0]["output"] == "compressed medium"
    assert new_payload["input"][1]["content"][0]["text"] == medium_text
    assert units_by_category == {"applied": 1, "size_floor": 1}
    assert strategy_chain == []
