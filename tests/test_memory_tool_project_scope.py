"""Unresolved project requests must not acquire Headroom memory tools."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.memory.storage_router import BackendRouter, BackendRouterConfig, MemoryStorageMode
from headroom.proxy.memory_handler import MEMORY_TOOL_NAMES
from headroom.proxy.server import ProxyConfig, create_app


@pytest.mark.parametrize(
    ("storage_mode", "fallback", "allow_unresolved"),
    [
        (MemoryStorageMode.PROJECT, "empty", False),
        (MemoryStorageMode.PROJECT, "global", True),
        (MemoryStorageMode.USER, "empty", True),
        (MemoryStorageMode.GLOBAL, "empty", True),
    ],
)
def test_memory_tools_follow_request_scope_without_losing_client_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    storage_mode: MemoryStorageMode,
    fallback: str,
    allow_unresolved: bool,
) -> None:
    monkeypatch.setenv("HEADROOM_MEMORY_INJECTION_MODE", "live_zone_tail")
    monkeypatch.delenv("HEADROOM_TOOL_INJECTION_STICKY", raising=False)
    # Isolate paths without changing the process-global stateless policy.
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("HEADROOM_BEACON", "off")
    config = ProxyConfig(
        optimize=False,
        memory_enabled=True,
        memory_db_path=str(tmp_path / "memory.db"),
        memory_storage_mode=storage_mode.value,
        memory_inject_context=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        subscription_tracking_enabled=False,
    )
    captured: list[dict] = []

    async def upstream(method, url, headers, body, stream=False, **kwargs):
        captured.append(deepcopy(body))
        return httpx.Response(
            200,
            json={
                "id": "msg_scope",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-4-5",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        )

    client_tools = [
        {
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "memory_search",
            "description": "Client-owned search, not Headroom memory",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    ]
    payload = {
        "model": "claude-haiku-4-5",
        "max_tokens": 16,
        "system": "You are helpful.",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": client_tools,
    }
    headers = {
        "x-api-key": "local-test-key",
        "anthropic-version": "2023-06-01",
        "x-headroom-user-id": "scope-test-user",
    }
    with TestClient(create_app(config)) as client:
        proxy = client.app.state.proxy
        handler = proxy.memory_handler
        assert handler is not None
        if fallback == "global":
            handler._router = BackendRouter(
                BackendRouterConfig(
                    mode=storage_mode,
                    root_dir=tmp_path / "memories",
                    global_db_path=tmp_path / "memory.db",
                    unresolved_project_fallback=fallback,
                )
            )
        monkeypatch.setattr(proxy, "_retry_request", upstream)
        monkeypatch.setattr(
            proxy.session_tracker_store,
            "compute_session_id",
            lambda request, model, messages: f"memory-scope-{tmp_path.name}",
        )

        for project in (None, "project-a", None, "project-a"):
            request_headers = dict(headers)
            if project:
                request_headers["x-headroom-project-id"] = project
            response = client.post("/v1/messages", headers=request_headers, json=payload)
            assert response.status_code == 200
            forwarded = captured[-1]["tools"]
            names = {tool["name"] for tool in forwarded}
            expected = {"get_weather", "memory_search"}
            if project or allow_unresolved:
                expected |= MEMORY_TOOL_NAMES
            assert names == expected
            assert (
                next(tool for tool in forwarded if tool["name"] == "memory_search")
                == client_tools[1]
            )

        # Losing scope must not replay tools, but restoring it keeps their golden bytes.
        assert captured[1]["tools"] == captured[3]["tools"]
