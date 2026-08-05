"""Regression tests for web-tool result passthrough."""

from __future__ import annotations

from headroom.config import DEFAULT_EXCLUDE_TOOLS
from headroom.proxy.server import HeadroomProxy, ProxyConfig
from headroom.transforms.content_detector import ContentType
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    RouterCompressionResult,
    RoutingDecision,
)


class _Tokenizer:
    def count_text(self, text: str) -> int:
        return max(1, len(text) // 4)


def _messages(tool_name: str, payload: str) -> list[dict[str, object]]:
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": tool_name,
                    "input": {},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool-1",
                    "content": payload,
                }
            ],
        },
    ]


def _router() -> ContentRouter:
    proxy = HeadroomProxy(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            code_aware_enabled=False,
            mode="token",
        )
    )
    router = proxy.anthropic_pipeline.transforms[-1]
    assert isinstance(router, ContentRouter)
    router.config.min_section_tokens = 1
    router.config.min_chars_for_block_compression = 1
    return router


def test_web_tools_are_default_exclusions() -> None:
    assert {"WebSearch", "WebFetch", "web_search", "web_fetch"} <= DEFAULT_EXCLUDE_TOOLS


def test_web_tool_results_bypass_compressor() -> None:
    router = _router()
    calls = 0

    def fake_compress(*args: object, **kwargs: object) -> RouterCompressionResult:
        nonlocal calls
        calls += 1
        content = str(args[0])
        return RouterCompressionResult(
            compressed="mutated",
            original=content,
            strategy_used=CompressionStrategy.TEXT,
            routing_log=[
                RoutingDecision(ContentType.PLAIN_TEXT, CompressionStrategy.TEXT, 100, 10)
            ],
        )

    router.compress = fake_compress  # type: ignore[method-assign]
    payload = (
        "{\n"
        '  "results": [\n'
        '    {"title": "Headroom", "snippet": "reference payload reference payload reference payload"},\n'
        '    {"title": "Docs", "snippet": "structured web payload with spacing that must remain verbatim"}\n'
        "  ],\n"
        '  "source": "web"\n'
        "}"
    )

    for tool_name in ("WebSearch", "WebFetch", "web_search", "web_fetch"):
        messages = _messages(tool_name, payload)
        result = router.apply(messages, _Tokenizer())

        tool_result = result.messages[1]["content"][0]  # type: ignore[index]
        assert tool_result["content"] == payload  # type: ignore[index]
        assert "router:excluded:tool" in result.transforms_applied

    assert calls == 0


def test_web_tool_results_stay_verbatim_outside_token_age_window() -> None:
    router = _router()
    calls = 0

    def fake_compress(*args: object, **kwargs: object) -> RouterCompressionResult:
        nonlocal calls
        calls += 1
        content = str(args[0])
        return RouterCompressionResult(
            compressed="mutated",
            original=content,
            strategy_used=CompressionStrategy.TEXT,
            routing_log=[
                RoutingDecision(ContentType.PLAIN_TEXT, CompressionStrategy.TEXT, 100, 10)
            ],
        )

    router.compress = fake_compress  # type: ignore[method-assign]
    payload = (
        "{\n"
        '  "results": [\n'
        '    {"title": "Headroom", "snippet": "reference payload reference payload reference payload"}\n'
        "  ]\n"
        "}"
    )
    messages = _messages("WebSearch", payload)
    messages.extend({"role": "user", "content": f"later turn {i}"} for i in range(18))

    result = router.apply(messages, _Tokenizer())
    tool_result = result.messages[1]["content"][0]  # type: ignore[index]

    assert tool_result["content"] == payload  # type: ignore[index]
    assert calls == 0


def test_web_tool_results_skip_cross_turn_dedup() -> None:
    router = _router()
    payload = (
        "{\n"
        '  "results": [\n'
        '    {"title": "Headroom", "snippet": "structured web payload with spacing that must remain verbatim"}\n'
        "  ]\n"
        "}"
    )
    messages = _messages("WebSearch", payload) + _messages("WebSearch", payload)

    result = router.apply(messages, _Tokenizer())

    first = result.messages[1]["content"][0]  # type: ignore[index]
    second = result.messages[3]["content"][0]  # type: ignore[index]
    assert first["content"] == payload  # type: ignore[index]
    assert second["content"] == payload  # type: ignore[index]


def test_bash_remains_compressible() -> None:
    router = _router()
    calls = 0

    def fake_compress(*args: object, **kwargs: object) -> RouterCompressionResult:
        nonlocal calls
        calls += 1
        content = str(args[0])
        return RouterCompressionResult(
            compressed="compressed bash output",
            original=content,
            strategy_used=CompressionStrategy.TEXT,
            routing_log=[
                RoutingDecision(ContentType.PLAIN_TEXT, CompressionStrategy.TEXT, 100, 10)
            ],
        )

    router.compress = fake_compress  # type: ignore[method-assign]
    payload = "bash output " * 100
    result = router.apply(_messages("Bash", payload), _Tokenizer())

    assert calls == 1
    assert "router:excluded:tool" not in result.transforms_applied
