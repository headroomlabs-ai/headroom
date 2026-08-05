"""Regression tests for Hermes Studio scoped coding-agent passthrough compression.

Verifies the new compression logic in ``handle_passthrough`` that rewrites
``/api/codex-proxy/.../v1/responses`` and ``/api/claude-code-proxy/.../v1/messages``
request bodies before forwarding upstream.

Key invariants tested:
- Chat messages (role=user/assistant) are compressed
- Non-chat items (tool, function, reasoning, system) are preserved byte-stable
- Non-dict items in the input array are preserved
- x-headroom-bypass header skips compression entirely
- Malformed/unsupported payloads are forwarded unchanged
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.providers.hermes import (  # noqa: E402
    compress_scoped_passthrough_body,
    is_scoped_coding_agent_path,
)
from headroom.proxy.loopback_guard import require_loopback  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


def _make_app(**kwargs: Any):
    """Create a minimal test app with loopback guard bypassed."""
    config = ProxyConfig(
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        **kwargs,
    )
    app = create_app(config)
    app.dependency_overrides[require_loopback] = lambda: None
    return app


def _mock_upstream(
    router: respx.MockRouter, upstream_url: str = "https://api.openai.com"
) -> dict[str, Any]:
    """Install a mock upstream that captures the forwarded request body."""
    captured: dict[str, Any] = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        try:
            captured["body"] = json.loads(request.content)
        except (json.JSONDecodeError, UnicodeDecodeError):
            captured["body"] = request.content
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={"id": "resp_1", "output": [], "usage": {"input_tokens": 10, "output_tokens": 1}},
        )

    # Mock any upstream path
    router.route(method="POST", url__startswith=upstream_url).mock(side_effect=_capture)
    router.route(method="GET", url__startswith=upstream_url).mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    return captured


# ── Codex proxy (Responses API) ──────────────────────────────────────────────


@respx.mock
def test_codex_proxy_preserves_tool_and_function_items() -> None:
    """Tool/function/reasoning items are preserved after compression."""
    original_input: list[Any] = [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Write a sort function."},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "read_file",
            "arguments": '{"path":"/tmp/x.py"}',
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "def foo(): pass"},
        {"role": "assistant", "content": "I see the file contains a foo function."},
        {"role": "user", "content": "Add error handling."},
    ]

    app = _make_app()
    with TestClient(app) as client:
        captured = _mock_upstream(respx)

        response = client.post(
            "/api/codex-proxy/some-session/v1/responses",
            headers={"authorization": "Bearer test-key"},
            json={"model": "gpt-4o-mini", "input": original_input},
        )

    assert response.status_code == 200
    forwarded_input = captured["body"]["input"]

    # Must have same length (all items preserved)
    assert len(forwarded_input) == len(original_input), (
        f"Input length changed: {len(original_input)} -> {len(forwarded_input)}"
    )

    # Non-chat items must be identical
    for idx in [0, 2, 3]:  # system, function_call, function_call_output
        assert forwarded_input[idx] == original_input[idx], (
            f"Item {idx} was mutated: {forwarded_input[idx]} != {original_input[idx]}"
        )

    # Chat items (user/assistant) should still be present (may be compressed)
    for idx in [1, 4, 5]:
        assert isinstance(forwarded_input[idx], dict), f"Item {idx} is no longer a dict"
        assert forwarded_input[idx].get("role") == original_input[idx].get("role"), (
            f"Item {idx} role changed"
        )


@respx.mock
def test_codex_proxy_preserves_nondict_items() -> None:
    """Non-dict items in the input array survive compression."""
    original_input: list[Any] = [
        {"role": "user", "content": "hello"},
        "a plain string that is not a dict",
        42,
        {"role": "assistant", "content": "hi there"},
    ]

    app = _make_app()
    with TestClient(app) as client:
        captured = _mock_upstream(respx)

        response = client.post(
            "/api/codex-proxy/session-1/v1/responses",
            headers={"authorization": "Bearer test-key"},
            json={"model": "gpt-4o-mini", "input": original_input},
        )

    assert response.status_code == 200
    forwarded_input = captured["body"]["input"]

    assert len(forwarded_input) == len(original_input)
    # Non-dict items preserved exactly
    assert forwarded_input[1] == "a plain string that is not a dict"
    assert forwarded_input[2] == 42
    # Dict items still present
    assert forwarded_input[0].get("role") == "user"
    assert forwarded_input[3].get("role") == "assistant"


@respx.mock
def test_codex_proxy_bypass_header_skips_compression() -> None:
    """x-headroom-bypass: true prevents any body mutation."""
    original_input = [
        {"role": "user", "content": "compressible " + "text " * 200},
        {"role": "assistant", "content": "response"},
    ]

    app = _make_app()
    with TestClient(app) as client:
        captured = _mock_upstream(respx)

        response = client.post(
            "/api/codex-proxy/session-1/v1/responses",
            headers={
                "authorization": "Bearer test-key",
                "x-headroom-bypass": "true",
            },
            json={"model": "gpt-4o-mini", "input": original_input},
        )

    assert response.status_code == 200
    # Body must be identical (no compression)
    assert captured["body"]["input"] == original_input


@respx.mock
def test_codex_proxy_malformed_input_preserved() -> None:
    """Malformed input (no model) is forwarded without mutation."""
    original_input = [
        {"role": "user", "content": "hello"},
    ]

    app = _make_app()
    with TestClient(app) as client:
        captured = _mock_upstream(respx)

        response = client.post(
            "/api/codex-proxy/session-1/v1/responses",
            headers={"authorization": "Bearer test-key"},
            json={"input": original_input},  # no "model" key
        )

    assert response.status_code == 200
    # Must be forwarded unchanged
    assert captured["body"]["input"] == original_input


@respx.mock
def test_codex_proxy_compression_applies_to_chat_messages() -> None:
    """Chat messages are compressed when model is present and bypass is off."""
    app = _make_app()
    with TestClient(app) as client:
        captured = _mock_upstream(respx)

        response = client.post(
            "/api/codex-proxy/session-1/v1/responses",
            headers={"authorization": "Bearer test-key"},
            json={
                "model": "gpt-4o-mini",
                "input": [
                    {"role": "user", "content": "hello world"},
                    {"role": "assistant", "content": "hi"},
                ],
            },
        )

    assert response.status_code == 200
    forwarded_input = captured["body"]["input"]
    assert len(forwarded_input) >= 2  # at least original count
    # Roles preserved
    assert forwarded_input[0].get("role") == "user"
    assert forwarded_input[1].get("role") == "assistant"


# ── Claude Code proxy (Anthropic Messages API) ───────────────────────────────


@respx.mock
def test_claude_proxy_preserves_tool_use_items() -> None:
    """tool_use and tool_result messages are preserved after compression."""
    original_messages = [
        {"role": "user", "content": [{"type": "text", "text": "Read the file."}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "read_file",
                    "input": {"path": "/tmp/x.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "print('hello')"}
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "The file prints hello."}]},
    ]

    app = _make_app()
    with TestClient(app) as client:
        captured = _mock_upstream(
            respx, upstream_url="https://httpbin.org"
        )  # any URL, overridden by header

        response = client.post(
            "/api/claude-code-proxy/session-1/v1/messages",
            headers={
                "authorization": "Bearer test-key",
                "x-headroom-base-url": "https://httpbin.org",
            },
            json={"model": "claude-sonnet-4-5-20250929", "messages": original_messages},
        )

    assert response.status_code == 200
    forwarded_messages = captured["body"]["messages"]

    assert len(forwarded_messages) == len(original_messages)

    # tool_use message preserved
    assert forwarded_messages[1]["role"] == "assistant"
    assert forwarded_messages[1]["content"][0]["type"] == "tool_use"

    # tool_result message preserved
    assert forwarded_messages[2]["role"] == "user"
    assert forwarded_messages[2]["content"][0]["type"] == "tool_result"


@respx.mock
def test_claude_proxy_bypass_header_skips_compression() -> None:
    """x-headroom-bypass: true prevents any body mutation on Claude proxy."""
    original_messages = [
        {"role": "user", "content": "compressible " + "text " * 200},
        {"role": "assistant", "content": "response"},
    ]

    app = _make_app()
    with TestClient(app) as client:
        captured = _mock_upstream(respx, upstream_url="https://httpbin.org")

        response = client.post(
            "/api/claude-code-proxy/session-1/v1/messages",
            headers={
                "authorization": "Bearer test-key",
                "x-headroom-bypass": "true",
                "x-headroom-base-url": "https://httpbin.org",
            },
            json={"model": "claude-sonnet-4-5-20250929", "messages": original_messages},
        )

    assert response.status_code == 200
    assert captured["body"]["messages"] == original_messages


@respx.mock
def test_claude_proxy_no_model_forwarded_unchanged() -> None:
    """Missing model → forwarded without compression."""
    original_messages = [
        {"role": "user", "content": "hello"},
    ]

    app = _make_app()
    with TestClient(app) as client:
        captured = _mock_upstream(respx, upstream_url="https://httpbin.org")

        response = client.post(
            "/api/claude-code-proxy/session-1/v1/messages",
            headers={
                "authorization": "Bearer test-key",
                "x-headroom-base-url": "https://httpbin.org",
            },
            json={"messages": original_messages},
        )

    assert response.status_code == 200
    assert captured["body"]["messages"] == original_messages


@respx.mock
def test_claude_proxy_compression_applies_to_chat_messages() -> None:
    """Chat messages (user/assistant) are compressed."""
    app = _make_app()
    with TestClient(app) as client:
        captured = _mock_upstream(respx, upstream_url="https://httpbin.org")

        response = client.post(
            "/api/claude-code-proxy/session-1/v1/messages",
            headers={
                "authorization": "Bearer test-key",
                "x-headroom-base-url": "https://httpbin.org",
            },
            json={
                "model": "claude-sonnet-4-5-20250929",
                "messages": [
                    {"role": "user", "content": "hello world"},
                    {"role": "assistant", "content": "hi"},
                ],
            },
        )

    assert response.status_code == 200
    forwarded_messages = captured["body"]["messages"]
    assert len(forwarded_messages) >= 2
    assert forwarded_messages[0].get("role") == "user"
    assert forwarded_messages[1].get("role") == "assistant"


# ── Generic passthrough (non-Hermes routes) ──────────────────────────────────


@respx.mock
def test_non_hermes_routes_not_affected() -> None:
    """Non-Hermes passthrough routes are not touched."""
    app = _make_app()
    with TestClient(app) as client:
        captured = _mock_upstream(respx)

        response = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer test-key"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    # Normal passthrough: body should just have "messages"
    assert "messages" in captured["body"]


def test_hermes_adapter_ignores_non_hermes_paths() -> None:
    assert not is_scoped_coding_agent_path("/v1/responses")
    assert not is_scoped_coding_agent_path("/api/codex-proxy/session/v1/chat/completions")
    assert is_scoped_coding_agent_path("/api/codex-proxy/session/v1/responses")
    assert is_scoped_coding_agent_path("/api/claude-code-proxy/session/v1/messages")


def test_hermes_adapter_preserves_structured_messages_when_compressor_changes_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_compress(*, messages: list[dict[str, Any]], **_: Any):
        class Result:
            pass

        result = Result()
        result.messages = [{**message, "content": "compressed"} for message in messages]
        return result

    monkeypatch.setattr("headroom.compress", fake_compress)
    original = {
        "model": "claude-sonnet-4-5-20250929",
        "messages": [
            {"role": "user", "content": "compress this"},
            {"role": "assistant", "content": [{"type": "tool_use", "name": "read_file"}]},
            {"role": "user", "content": [{"type": "tool_result", "content": "secret"}]},
        ],
    }
    body = json.dumps(original).encode()

    transformed = compress_scoped_passthrough_body(
        "/api/claude-code-proxy/session/v1/messages", body, optimize=True, bypass=False
    )

    messages = json.loads(transformed)["messages"]
    assert messages[0]["content"] == "compressed"
    assert messages[1] == original["messages"][1]
    assert messages[2] == original["messages"][2]


def test_codex_adapter_compresses_canonical_responses_input_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, list[dict[str, Any]]] = {}

    def fake_compress(*, messages: list[dict[str, Any]], **_: Any):
        class Result:
            pass

        seen["messages"] = messages
        result = Result()
        result.messages = [
            {
                **messages[0],
                "content": [{**messages[0]["content"][0], "text": "compressed prompt"}],
            }
        ]
        return result

    monkeypatch.setattr("headroom.compress", fake_compress)
    original = {
        "model": "gpt-5.5",
        "input": [
            {
                "type": "message",
                "role": "user",
                "metadata": {"source": "codex"},
                "content": [{"type": "input_text", "text": "long prompt", "annotations": []}],
            }
        ],
    }
    body = json.dumps(original).encode()

    transformed = compress_scoped_passthrough_body(
        "/api/codex-proxy/session/v1/responses", body, optimize=True, bypass=False
    )

    assert transformed != body
    assert seen["messages"][0]["type"] == "message"
    assert seen["messages"][0]["content"] == [
        {"type": "text", "text": "long prompt", "annotations": []}
    ]
    forwarded = json.loads(transformed)["input"][0]
    assert forwarded == {
        "type": "message",
        "role": "user",
        "metadata": {"source": "codex"},
        "content": [{"type": "input_text", "text": "compressed prompt", "annotations": []}],
    }
