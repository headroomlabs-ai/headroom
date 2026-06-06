import importlib
from types import SimpleNamespace
from typing import Any

import httpx
from click.testing import CliRunner
from fastapi.testclient import TestClient

from headroom.cli.wrap import _build_hermes_launch_env, wrap
from headroom.proxy.handlers.openai import (
    _compact_openai_chat_tools_for_hermes,
    _openai_chat_response_shape,
    _prepare_hermes_packed_history_messages,
    _restore_hermes_packed_history_roles,
    _should_inject_ccr_tool_for_request,
)
from headroom.proxy.server import ProxyConfig, create_app

wrap_module = importlib.import_module("headroom.cli.wrap")


class _DummyTokenizer:
    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        return sum(len(str(message.get("content", ""))) for message in messages)

    def count_text(self, text: str) -> int:
        return len(text)


def _capture_openai_chat_request(
    client: TestClient,
    *,
    headers: dict[str, str] | None = None,
    body: dict[str, Any],
) -> tuple[dict[str, Any], httpx.Response]:
    captured: dict[str, Any] = {}
    proxy = client.app.state.proxy

    async def _fake_retry(
        method: str,
        url: str,
        request_headers: dict[str, str],
        request_body: dict[str, Any],
        stream: bool = False,
        **kwargs: Any,
    ) -> httpx.Response:
        captured["method"] = method
        captured["url"] = url
        captured["headers"] = request_headers
        captured["body"] = request_body
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_hermes_test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            },
        )

    proxy._retry_request = _fake_retry
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer sk-test", **(headers or {})},
        json=body,
    )
    return captured, response


def test_hermes_openrouter_env_routes_via_openrouter_base_url():
    env, display = _build_hermes_launch_env(
        8787,
        {"OPENROUTER_API_KEY": "sk-openrouter"},
        provider_mode="openrouter",
    )

    assert env["OPENROUTER_BASE_URL"] == "http://127.0.0.1:8787/v1"
    assert "CUSTOM_BASE_URL" not in env
    assert "HERMES_INFERENCE_PROVIDER" not in env
    assert "OPENROUTER_BASE_URL=http://127.0.0.1:8787/v1" in display


def test_hermes_custom_env_does_not_set_openrouter_base_url():
    env, display = _build_hermes_launch_env(
        8788,
        {"OPENROUTER_API_KEY": "sk-openrouter"},
        provider_mode="custom",
        debug_shape=True,
    )

    assert env["HERMES_INFERENCE_PROVIDER"] == "custom"
    assert env["CUSTOM_BASE_URL"] == "http://127.0.0.1:8788/v1"
    assert "OPENROUTER_BASE_URL" not in env
    assert env["HEADROOM_HERMES_DEBUG_SHAPE"] == "1"
    assert "CUSTOM_BASE_URL=http://127.0.0.1:8788/v1" in display


def test_hermes_prepare_only_defaults_backend_to_openrouter(monkeypatch):
    monkeypatch.delenv("HEADROOM_BACKEND", raising=False)
    result = CliRunner().invoke(wrap, ["hermes", "--prepare-only"])

    assert result.exit_code == 0
    assert "OPENROUTER_BASE_URL=http://127.0.0.1:8787/v1" in result.output
    assert "HEADROOM_BACKEND=openrouter" in result.output


def test_hermes_launch_passes_debug_env_to_proxy(monkeypatch):
    captured = {}

    def fake_launch_tool(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(wrap_module.shutil, "which", lambda name: "/bin/echo")
    monkeypatch.setattr(wrap_module, "_launch_tool", fake_launch_tool)

    result = CliRunner().invoke(wrap, ["hermes", "--debug-shape", "--", "-z", "hi"])

    assert result.exit_code == 0
    assert captured["agent_type"] == "hermes"
    assert captured["backend"] == "openrouter"
    assert captured["proxy_env_overrides"] == {
        "HEADROOM_HERMES_MODE": "1",
        "HEADROOM_HERMES_DEBUG_SHAPE": "1",
    }
    assert "startup_timeout_seconds" not in captured


def test_compact_chat_tools_preserves_function_shape():
    body = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "run",
                    "description": "  Run   a command.  ",
                    "parameters": {
                        "type": "object",
                        "title": "ToolInput",
                        "properties": {
                            "cmd": {
                                "type": "string",
                                "description": "  Shell command.  ",
                                "examples": ["ls"],
                            }
                        },
                    },
                },
            }
        ]
    }

    compacted, changed, before, after = _compact_openai_chat_tools_for_hermes(body)

    assert changed is True
    assert after < before
    assert compacted["tools"][0]["type"] == "function"
    assert compacted["tools"][0]["function"]["name"] == "run"
    assert "title" not in compacted["tools"][0]["function"]["parameters"]
    assert "examples" not in compacted["tools"][0]["function"]["parameters"]["properties"]["cmd"]


def test_hermes_disables_ccr_tool_injection_until_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("HEADROOM_HERMES_CCR_TOOL", raising=False)

    assert (
        _should_inject_ccr_tool_for_request(
            config_enabled=True,
            hermes_request=True,
        )
        is False
    )
    assert (
        _should_inject_ccr_tool_for_request(
            config_enabled=True,
            hermes_request=False,
        )
        is True
    )

    monkeypatch.setenv("HEADROOM_HERMES_CCR_TOOL", "1")

    assert (
        _should_inject_ccr_tool_for_request(
            config_enabled=True,
            hermes_request=True,
        )
        is True
    )


def test_packed_user_history_is_retagged_then_restored():
    packed = (
        '{"role":"tool","tool_call_id":"abc","content":"terminal output"}\n'
        "stdout: line\nstderr: none\nexit_code: 0\n"
    ) * 40
    messages = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": packed},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "current task must stay protected"},
    ]

    prepared, indices = _prepare_hermes_packed_history_messages(messages)

    assert indices == [1]
    assert prepared[1]["role"] == "tool"
    assert prepared[3]["role"] == "user"

    restored = _restore_hermes_packed_history_roles(prepared, indices)

    assert restored[1]["role"] == "user"
    assert "tool_call_id" not in restored[1]
    assert restored[1]["content"] == packed


def test_response_shape_preserves_content_and_tool_call_signals():
    content_shape = _openai_chat_response_shape(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "hello"},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }
    )
    tool_shape = _openai_chat_response_shape(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": "call_1"}],
                    },
                }
            ]
        }
    )

    assert content_shape["has_content"] is True
    assert content_shape["finish_reason"] == "stop"
    assert tool_shape["has_tool_calls"] is True
    assert tool_shape["tool_calls_count"] == 1


def test_hermes_request_path_suppresses_headroom_ccr_tool(monkeypatch):
    monkeypatch.delenv("HEADROOM_HERMES_CCR_TOOL", raising=False)
    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())
    app = create_app(
        ProxyConfig(
            optimize=False,
            image_optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=True,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
            discover_pipeline_extensions=False,
        )
    )

    with TestClient(app) as client:
        captured, response = _capture_openai_chat_request(
            client,
            headers={"User-Agent": "Hermes/0.15"},
            body={
                "model": "openrouter/test",
                "messages": [
                    {
                        "role": "user",
                        "content": "[10 items compressed. Retrieve more: hash=abc123]",
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "run_shell",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
        )

    assert response.status_code == 200, response.text
    forwarded_tools = captured["body"].get("tools") or []
    tool_names = {
        tool.get("function", {}).get("name")
        for tool in forwarded_tools
        if isinstance(tool, dict)
    }
    assert "run_shell" in tool_names
    assert "headroom_retrieve" not in tool_names


def test_hermes_request_path_compacts_openai_chat_tools(monkeypatch):
    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())
    app = create_app(
        ProxyConfig(
            optimize=False,
            image_optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
            discover_pipeline_extensions=False,
        )
    )

    with TestClient(app) as client:
        captured, response = _capture_openai_chat_request(
            client,
            headers={"User-Agent": "Hermes/0.15"},
            body={
                "model": "openrouter/test",
                "messages": [{"role": "user", "content": "Run a command"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "run_shell",
                            "description": "  Run   a shell command.  ",
                            "parameters": {
                                "type": "object",
                                "title": "RunShellInput",
                                "properties": {
                                    "cmd": {
                                        "type": "string",
                                        "description": "  Command to run.  ",
                                        "examples": ["ls -la"],
                                    }
                                },
                            },
                        },
                    }
                ],
            },
        )

    assert response.status_code == 200, response.text
    forwarded_tool = captured["body"]["tools"][0]
    forwarded_params = forwarded_tool["function"]["parameters"]
    assert forwarded_tool["function"]["name"] == "run_shell"
    assert "title" not in forwarded_params
    assert "examples" not in forwarded_params["properties"]["cmd"]


def test_hermes_request_path_retags_then_restores_packed_user_history(monkeypatch):
    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())
    packed = (
        '{"role":"tool","tool_call_id":"abc","content":"terminal output"}\n'
        "stdout: many lines\nstderr: none\nexit_code: 0\n"
    ) * 40
    pipeline_seen: dict[str, Any] = {}

    def _pipeline_apply(messages, model, **kwargs):  # noqa: ANN001
        pipeline_seen["messages"] = messages
        rewritten = [dict(message) for message in messages]
        rewritten[1]["content"] = "[packed history compressed]"
        return SimpleNamespace(
            messages=rewritten,
            transforms_applied=["router:text:kompress"],
            tokens_before=2000,
            tokens_after=20,
            timing={},
            waste_signals=None,
        )

    app = create_app(
        ProxyConfig(
            optimize=True,
            mode="message",
            image_optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
            discover_pipeline_extensions=False,
        )
    )

    with TestClient(app) as client:
        client.app.state.proxy.openai_pipeline = SimpleNamespace(apply=_pipeline_apply)
        captured, response = _capture_openai_chat_request(
            client,
            headers={"User-Agent": "Hermes/0.15"},
            body={
                "model": "openrouter/test",
                "messages": [
                    {"role": "system", "content": "instructions"},
                    {"role": "user", "content": packed},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "current task must stay protected"},
                ],
            },
        )

    assert response.status_code == 200, response.text
    assert pipeline_seen["messages"][1]["role"] == "tool"
    assert pipeline_seen["messages"][1]["tool_call_id"] == "headroom_hermes_history_1"
    assert pipeline_seen["messages"][3]["role"] == "user"

    forwarded_messages = captured["body"]["messages"]
    assert forwarded_messages[1]["role"] == "user"
    assert "tool_call_id" not in forwarded_messages[1]
    assert forwarded_messages[1]["content"] == "[packed history compressed]"
    assert forwarded_messages[3]["role"] == "user"
    assert forwarded_messages[3]["content"] == "current task must stay protected"
