from __future__ import annotations

import copy
import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from headroom.cache.compression_store import get_compression_store, reset_compression_store
from headroom.pipeline import PipelineStage
from headroom.proxy.server import ProxyConfig, create_app
from headroom.proxy.turn_hooks import TurnContext, clear_turn_hooks, register_turn_hook


def _tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "tool_search_tool_regex_20251119",
            "name": "tool_search_tool_regex",
        },
        {
            "name": "WaitForMcpServers",
            "description": "Wait for MCP servers to become ready.",
            "input_schema": {"type": "object", "properties": {"server": {"type": "string"}}},
        },
        {
            "name": "work_items",
            "description": "Read work items from Azure DevOps.",
            "input_schema": {"type": "object", "properties": {"project": {"type": "string"}}},
        },
    ]


def _workitems() -> str:
    return json.dumps(
        [
            {
                "id": index,
                "rev": 1,
                "fields": {"System.Title": f"work item {index}", "System.State": "New"},
                "url": f"https://dev.azure.com/example/_apis/wit/workItems/{index}",
            }
            for index in range(150)
        ]
    )


def _messages(result: str | None = None) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "server_tool_use",
            "id": "search_1",
            "name": "tool_search_tool_regex",
            "input": {"pattern": "Azure DevOps", "limit": 5},
        },
        {
            "type": "tool_search_tool_result",
            "tool_use_id": "search_1",
            "content": {
                "type": "tool_search_tool_search_result",
                "tool_references": [
                    {"type": "tool_reference", "tool_name": "WaitForMcpServers"},
                    {"type": "tool_reference", "tool_name": "work_items"},
                ],
            },
        },
        {
            "type": "tool_use",
            "id": "work-1",
            "name": "work_items",
            "input": {"project": "example"},
        },
    ]
    messages: list[dict[str, Any]] = [{"role": "assistant", "content": content}]
    if result is not None:
        messages.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "work-1", "content": result}],
            }
        )
    return messages


def _body(
    *, result: str | None = None, tools: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": _messages(result),
        "tools": copy.deepcopy(_tools() if tools is None else tools),
    }


def _response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_tool_search",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 10, "output_tokens": 3},
        },
    )


def _client(
    *,
    optimize: bool = True,
    mode: str = "token",
    ccr_inject_tool: bool = False,
    ccr_inject_system_instructions: bool = False,
) -> TestClient:
    return TestClient(
        create_app(
            ProxyConfig(
                optimize=optimize,
                mode=mode,
                cache_enabled=False,
                rate_limit_enabled=False,
                cost_tracking_enabled=False,
                log_requests=False,
                ccr_inject_tool=ccr_inject_tool,
                ccr_inject_system_instructions=ccr_inject_system_instructions,
                ccr_handle_responses=False,
                ccr_context_tracking=False,
                image_optimize=False,
                pipeline_extensions=[],
                discover_pipeline_extensions=False,
            )
        )
    )


def _capture(
    client: TestClient, body: dict[str, Any], **headers: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    captured: dict[str, Any] = {}

    async def fake_retry(
        method: str,
        url: str,
        request_headers: dict[str, str],
        request_body: dict[str, Any],
        **kwargs: Any,
    ) -> httpx.Response:
        captured["body"] = copy.deepcopy(request_body)
        captured["kwargs"] = kwargs
        return _response()

    client.app.state.proxy._retry_request = fake_retry
    request_headers = {
        "x-api-key": "test-key",
        "anthropic-version": "2023-06-01",
        "x-client": "claude-code",
        **headers,
    }
    response = client.post("/v1/messages", headers=request_headers, json=body)
    assert response.status_code == 200, response.text
    return captured["body"], captured.get("kwargs", {})


@pytest.fixture(autouse=True)
def _clean_turn_hooks() -> None:
    clear_turn_hooks()
    yield
    clear_turn_hooks()


class _ReplacementExtension:
    def on_pipeline_event(self, event: Any) -> Any:
        if event.stage in {
            PipelineStage.INPUT_RECEIVED,
            PipelineStage.INPUT_REMEMBERED,
            PipelineStage.PRE_SEND,
        }:
            event.tools = [{"name": "replacement", "input_schema": {"type": "object"}}]
        return event


class _ProviderHistoryMutationExtension:
    def __init__(self) -> None:
        self.stages: list[PipelineStage] = []

    def on_pipeline_event(self, event: Any) -> Any:
        if event.messages is not None:
            self.stages.append(event.stage)
            replaced = copy.deepcopy(event.messages)
            provider_blocks = [
                block
                for message in replaced
                for block in message.get("content", [])
                if block.get("type") in {"server_tool_use", "tool_search_tool_result"}
            ]
            provider_blocks[0]["id"] = "mutated-id"
            provider_blocks[0]["input"]["pattern"] = "mutated"
            provider_blocks[1]["tool_use_id"] = "mutated-tool-use-id"
            provider_blocks[1]["content"]["tool_references"][0]["tool_name"] = "mutated"
            event.messages = replaced
        return event


def test_route_contract_preserves_tools_and_nested_history_while_compressing() -> None:
    inbound = _body(result=_workitems())
    with _client() as client:
        outbound, _ = _capture(client, inbound)

    assert outbound["tools"] == inbound["tools"]
    assert outbound["messages"][0]["content"] == inbound["messages"][0]["content"]
    result = outbound["messages"][1]["content"][0]["content"]
    assert isinstance(result, str)
    assert len(result) < len(_workitems())


def test_string_tool_result_is_compressed() -> None:
    inbound = _body(result=_workitems())
    with _client() as client:
        outbound, _ = _capture(client, inbound)
    assert len(outbound["messages"][1]["content"][0]["content"]) < len(_workitems())


def test_replacement_attempts_cannot_replace_active_tools() -> None:
    class Hook:
        name = "replace-tools"

        def on_request(self, ctx: TurnContext) -> None:
            ctx.tools = [{"name": "hook-replacement"}]
            ctx.messages[0]["content"][0]["input"]["pattern"] = "hook-mutated"
            ctx.messages[0]["content"][1]["content"]["tool_references"][0]["tool_name"] = (
                "hook-mutated"
            )

    register_turn_hook(Hook())
    inbound = _body(result=_workitems())
    with _client() as client:
        client.app.state.proxy.pipeline_extensions._extensions = [_ReplacementExtension()]
        outbound, _ = _capture(client, inbound)
    assert outbound["tools"] == inbound["tools"]
    assert outbound["messages"][0] == inbound["messages"][0]


def test_active_route_injection_cannot_add_proxy_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_TOOL_SEARCH", "true")
    marker_hash = "abc123def456abc123def456"
    reset_compression_store()
    get_compression_store().store(
        original="expanded tool result",
        compressed="[100 items compressed to 10]",
        explicit_hash=marker_hash,
    )
    try:
        inbound = _body(result=f"[100 items compressed to 10. Retrieve more: hash={marker_hash}]")
        with _client(
            ccr_inject_tool=True,
            ccr_inject_system_instructions=True,
        ) as client:
            outbound, _ = _capture(client, inbound)
    finally:
        reset_compression_store()

    serialized = json.dumps(outbound)
    assert "headroom_retrieve" not in serialized
    assert "expanded tool result" in serialized
    assert "Retrieve more: hash=" not in serialized
    assert outbound["tools"] == inbound["tools"]


def test_active_route_with_memory_enabled_keeps_memory_state_initialized() -> None:
    class MemoryConfig:
        inject_context = False
        inject_tools = True
        project_root_override = ""

    class MemoryHandler:
        config = MemoryConfig()
        initialized = False
        backend = None

        def has_memory_tool_calls(self, response: dict[str, Any], provider: str) -> bool:
            return False

        def compute_memory_tool_definitions(self, provider: str) -> list[dict[str, Any]]:
            return [{"name": "memory_search", "input_schema": {"type": "object"}}]

    inbound = _body(result=_workitems())
    with _client() as client:
        client.app.state.proxy.memory_handler = MemoryHandler()
        outbound, _ = _capture(client, inbound, **{"x-headroom-user-id": "test-user"})
    assert outbound["tools"] == inbound["tools"]


def test_active_route_beta_marker_locks_tools_without_tool_search_definition() -> None:
    inbound = _body(result=_workitems())
    inbound["tools"][0] = {
        "name": "ordinary_tool",
        "description": "A provider-owned tool.",
        "input_schema": {"type": "object", "properties": {"value": {"type": "string"}}},
    }
    with _client() as client:
        outbound, _ = _capture(
            client,
            inbound,
            **{"anthropic-beta": "advanced-tool-use-2025-11-20"},
        )
    assert outbound["tools"] == inbound["tools"]


@pytest.mark.parametrize(
    "marker_template",
    [
        "<<ccr:{hash}>>",
        "[100 items compressed to 10. Retrieve more: hash={hash}]",
        "[Read content stale. Retrieve original: hash={hash}]",
    ],
)
def test_active_route_resolves_ccr_marker_without_adding_retrieve_tool(
    marker_template: str,
) -> None:
    marker_hash = "abc123def456abc123def456"
    reset_compression_store()
    get_compression_store().store(
        original="expanded tool result",
        compressed="[100 items compressed to 10]",
        explicit_hash=marker_hash,
    )
    try:
        inbound = _body(result=marker_template.format(hash=marker_hash))
        with _client(
            ccr_inject_tool=False,
            ccr_inject_system_instructions=False,
        ) as client:
            outbound, _ = _capture(client, inbound)
        assert outbound["tools"] == inbound["tools"]
        result = outbound["messages"][1]["content"][0]["content"]
        assert "Retrieve more: hash=" not in result
        assert "Retrieve original: hash=" not in result
        assert result == "expanded tool result"
    finally:
        reset_compression_store()


def test_provider_history_is_unchanged() -> None:
    inbound = _body(result=_workitems())
    with _client() as client:
        outbound, _ = _capture(client, inbound)
    assert outbound["messages"][0] == inbound["messages"][0]


def test_pipeline_history_mutations_restore_nested_provider_blocks() -> None:
    inbound = _body(result=_workitems())
    extension = _ProviderHistoryMutationExtension()
    with _client() as client:
        client.app.state.proxy.pipeline_extensions._extensions = [extension]
        outbound, _ = _capture(client, inbound)

    assert outbound["messages"][0]["content"] == inbound["messages"][0]["content"]
    assert PipelineStage.INPUT_COMPRESSED in extension.stages
    if PipelineStage.INPUT_ROUTED in extension.stages:
        assert extension.stages.count(PipelineStage.INPUT_ROUTED) == 1


def test_non_list_message_replacement_keeps_replacement_and_provider_history() -> None:
    class Extension:
        def on_pipeline_event(self, event: Any) -> Any:
            if event.stage == PipelineStage.INPUT_RECEIVED:
                event.messages = [{"role": "user", "content": "replacement"}]
            return event

    inbound = _body(result=_workitems())
    with _client() as client:
        client.app.state.proxy.pipeline_extensions._extensions = [Extension()]
        outbound, _ = _capture(client, inbound)

    assert outbound["messages"][0]["content"] == inbound["messages"][0]["content"][:2]
    assert outbound["messages"][-1] == {"role": "user", "content": "replacement"}


def test_list_message_replacement_preserves_provider_message_role() -> None:
    class Extension:
        def on_pipeline_event(self, event: Any) -> Any:
            if event.stage == PipelineStage.INPUT_RECEIVED:
                event.messages = [
                    {"role": "user", "content": copy.deepcopy(event.messages[0]["content"])}
                ]
            return event

    inbound = _body(result=_workitems())
    with _client() as client:
        client.app.state.proxy.pipeline_extensions._extensions = [Extension()]
        outbound, _ = _capture(client, inbound)

    assert outbound["messages"][0]["role"] == "assistant"
    assert outbound["messages"][0]["content"] == inbound["messages"][0]["content"][:2]
    assert outbound["messages"][-1]["role"] == "user"


def test_text_list_tool_result_remains_compressible_and_list_shaped() -> None:
    inbound = _body(result=None)
    inbound["messages"].append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "work-1",
                    "content": [{"type": "text", "text": _workitems()}],
                }
            ],
        }
    )
    with _client() as client:
        outbound, _ = _capture(client, inbound)
    content = outbound["messages"][1]["content"][0]["content"]
    assert isinstance(content, list)
    assert all(block.get("type") == "text" for block in content)
    assert len("".join(block["text"] for block in content)) < len(_workitems())


def test_continuation_tools_use_locked_snapshot() -> None:
    inbound = _body(result=_workitems())
    continuation_capture: dict[str, Any] = {}

    class FakeCcr:
        def has_ccr_tool_calls(self, response: dict[str, Any], provider: str) -> bool:
            return True

        async def handle_response(
            self,
            response: dict[str, Any],
            messages: list[dict[str, Any]],
            tools: Any,
            api_call_fn: Any,
            provider: str,
        ) -> dict[str, Any]:
            mutated = copy.deepcopy(messages)
            mutated[0]["content"][0]["input"]["pattern"] = "ccr-mutated"
            mutated[0]["content"][1]["content"]["tool_references"][0]["tool_name"] = "ccr-mutated"
            fresh_history = [
                {
                    "type": "server_tool_use",
                    "id": "fresh-search",
                    "name": "tool_search_tool_regex",
                    "input": {"pattern": "fresh"},
                },
                {
                    "type": "tool_search_tool_result",
                    "tool_use_id": "fresh-search",
                    "content": {
                        "type": "tool_search_tool_search_result",
                        "tool_references": [{"type": "tool_reference", "tool_name": "work_items"}],
                    },
                },
            ]
            continuation_capture["fresh_history"] = fresh_history
            mutated.append({"role": "assistant", "content": fresh_history})
            await api_call_fn(mutated, [{"name": "replacement"}])
            return response

        def residual_ccr_status(self, response: dict[str, Any], provider: str) -> None:
            return None

    class FakeHttpClient:
        async def post(
            self, url: str, *, content: bytes, headers: dict[str, str], timeout: Any
        ) -> httpx.Response:
            continuation_capture["body"] = json.loads(content)
            return _response()

        async def aclose(self) -> None:
            return None

    with _client() as client:
        proxy = client.app.state.proxy
        proxy.ccr_response_handler = FakeCcr()
        proxy.http_client = FakeHttpClient()
        outbound, _ = _capture(client, inbound)
    assert outbound["tools"] == inbound["tools"]
    assert outbound["messages"][0] == inbound["messages"][0]
    assert continuation_capture["body"]["tools"] == inbound["tools"]
    assert continuation_capture["body"]["messages"][0] == inbound["messages"][0]
    assert (
        continuation_capture["body"]["messages"][-1]["content"]
        == continuation_capture["fresh_history"]
    )


def test_continuation_reapplies_unsupported_history_repair() -> None:
    inbound = _body(result=_workitems())
    inbound["tools"] = [
        tool for tool in inbound["tools"] if tool.get("name") != "WaitForMcpServers"
    ]
    continuation_capture: dict[str, Any] = {}

    class FakeCcr:
        def has_ccr_tool_calls(self, response: dict[str, Any], provider: str) -> bool:
            return True

        async def handle_response(
            self,
            response: dict[str, Any],
            messages: list[dict[str, Any]],
            tools: Any,
            api_call_fn: Any,
            provider: str,
        ) -> dict[str, Any]:
            await api_call_fn(copy.deepcopy(messages), None)
            return response

        def residual_ccr_status(self, response: dict[str, Any], provider: str) -> None:
            return None

    class FakeHttpClient:
        async def post(
            self, url: str, *, content: bytes, headers: dict[str, str], timeout: Any
        ) -> httpx.Response:
            continuation_capture["body"] = json.loads(content)
            return _response()

        async def aclose(self) -> None:
            return None

    with _client() as client:
        proxy = client.app.state.proxy
        proxy.ccr_response_handler = FakeCcr()
        proxy.http_client = FakeHttpClient()
        _capture(client, inbound)

    assert "WaitForMcpServers" not in json.dumps(continuation_capture["body"]["messages"])


def test_memory_continuation_uses_locked_snapshot() -> None:
    inbound = _body(result=_workitems())

    class MemoryConfig:
        inject_context = False
        inject_tools = True
        project_root_override = ""

    class FakeMemory:
        config = MemoryConfig()
        initialized = True
        backend = object()

        def has_memory_tool_calls(self, response: dict[str, Any], provider: str) -> bool:
            return True

        async def handle_memory_tool_calls(
            self,
            response: dict[str, Any],
            user_id: str,
            provider: str,
            *,
            request_context: Any,
        ) -> list[dict[str, Any]]:
            return [{"type": "tool_result", "content": "memory result"}]

        def compute_memory_tool_definitions(self, provider: str) -> list[dict[str, Any]]:
            return [{"name": "memory_search", "input_schema": {"type": "object"}}]

    with _client() as client:
        client.app.state.proxy.memory_handler = FakeMemory()
        outbound, _ = _capture(client, inbound, **{"x-headroom-user-id": "test-user"})

    assert outbound["tools"] == inbound["tools"]
    assert outbound["messages"][-1]["content"] == [
        {"type": "tool_result", "content": "memory result"}
    ]


def test_response_hook_continuation_uses_locked_snapshot() -> None:
    inbound = _body(result=_workitems())

    class ResponseHook:
        name = "response-reload"

        async def on_response(
            self,
            ctx: TurnContext,
            response: dict[str, Any],
            call_model: Any,
        ) -> dict[str, Any]:
            ctx.tools = [{"name": "response-hook-replacement"}]
            replacement = copy.deepcopy(ctx.messages)
            replacement[0]["content"][0]["input"]["pattern"] = "hook-mutated"
            replacement[0]["content"][1]["content"]["tool_references"][0]["tool_name"] = (
                "hook-mutated"
            )
            replacement.append({"role": "user", "content": "reload"})
            return await call_model(replacement)

    register_turn_hook(ResponseHook())
    with _client() as client:
        outbound, _ = _capture(client, inbound)

    assert outbound["tools"] == inbound["tools"]
    assert outbound["messages"][0] == inbound["messages"][0]
    assert outbound["messages"][-1] == {"role": "user", "content": "reload"}


def test_active_route_description_compaction_leaves_tools_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import headroom.proxy.tool_schema_compaction as compaction

    monkeypatch.setenv("HEADROOM_TOOL_DESC_MAX_CHARS", "20")
    monkeypatch.setattr(compaction, "_TOOL_DESC_MAX_CHARS", None)
    inbound = _body(result=_workitems())
    inbound["tools"][1]["description"] = (
        "This description is deliberately much longer than twenty characters."
    )
    with _client() as client:
        outbound, _ = _capture(client, inbound)
    assert outbound["tools"] == inbound["tools"]


def test_cache_mode_route_capture_keeps_active_token_lock_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_TOOL_SEARCH", "false")
    inbound = _body(result=_workitems())
    with _client(mode="cache") as client:
        client.app.state.proxy.pipeline_extensions._extensions = [_ReplacementExtension()]
        outbound, _ = _capture(client, inbound)
    assert outbound["tools"] != inbound["tools"]
    assert outbound["tools"][0]["name"] == "replacement"


def test_custom_anthropic_upstream_strips_tool_search_without_lock_restore() -> None:
    inbound = _body(result=_workitems())
    with _client() as client:
        outbound, _ = _capture(
            client,
            inbound,
            **{"x-headroom-base-url": "https://api.deepseek.com/anthropic"},
        )
    assert all(
        not str(tool.get("type", "")).startswith("tool_search_tool_") for tool in outbound["tools"]
    )


def test_active_route_keeps_tools_exact_when_ttl_repair_would_rewrite_tools() -> None:
    inbound = _body(result=_workitems())
    inbound["tools"][0]["cache_control"] = {"type": "ephemeral"}
    inbound["messages"].append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "later",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
        }
    )
    with _client() as client:
        outbound, _ = _capture(client, inbound)
    assert outbound["tools"] == inbound["tools"]


def test_bypass_bytes_preserve_original_body() -> None:
    raw = '{ "tools": [], "messages": [], "model": "claude-sonnet-4-6", "max_tokens": 1 }'
    captured: dict[str, Any] = {}

    async def fake_retry(
        method: str, url: str, request_headers: dict[str, str], body: dict[str, Any], **kwargs: Any
    ) -> httpx.Response:
        captured.update(kwargs)
        return _response()

    with _client() as client:
        client.app.state.proxy._retry_request = fake_retry
        response = client.post(
            "/v1/messages",
            content=raw.encode(),
            headers={
                "x-api-key": "test-key",
                "x-client": "claude-code",
                "x-headroom-bypass": "true",
                "content-type": "application/json",
            },
        )
        assert response.status_code == 200, response.text
    assert captured["original_body_bytes"] == raw.encode()
