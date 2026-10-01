"""Claude Code server-stored Threads (``thread: {"type": "continue", ...}``).

A continue turn carries only the new delta: ``system`` is often just the
billing block, ``tools`` is omitted, and the tool_use naming a tool_result is
upstream. Each test pins one place where request-local assumptions broke that.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom import tool_name_registry  # noqa: E402
from headroom.proxy.anthropic_threads import is_thread_continue, thread_scope  # noqa: E402
from headroom.proxy.output_shaper import OutputShaperSettings, shape_request  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402
from headroom.transforms.content_router import ContentRouter  # noqa: E402

CONTINUE = {"type": "continue", "previous_message_id": "msg_prev"}
BILLING = {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.286"}
ENABLED = OutputShaperSettings(enabled=True)


def test_is_thread_continue() -> None:
    assert is_thread_continue({"thread": CONTINUE})
    assert not is_thread_continue({"thread": {"type": "create"}})
    assert not is_thread_continue({"messages": []})
    assert not is_thread_continue({"thread": "continue"})


# 1. output shaper ----------------------------------------------------------


def test_shaper_skips_steering_on_continue_with_billing_only_system() -> None:
    body = {"thread": CONTINUE, "system": [dict(BILLING)], "messages": []}
    result = shape_request(body, ENABLED, level_override=2)
    assert body["system"] == [BILLING]
    assert not result.changed


def test_shaper_never_steers_continue_even_with_full_system_resent() -> None:
    body = {"thread": CONTINUE, "system": [dict(BILLING), {"type": "text", "text": "full"}]}
    shape_request(body, ENABLED, level_override=2)
    assert len(body["system"]) == 2


def test_shaper_still_steers_create_turn() -> None:
    body = {"thread": {"type": "create"}, "system": [dict(BILLING)]}
    shape_request(body, ENABLED, level_override=2)
    assert len(body["system"]) == 2


# 2-5. handler --------------------------------------------------------------


class _FakeTracker:
    """Reports a confirmed frozen prefix so a wrongly-used tracker is visible."""

    _cached_token_count = 0

    def get_frozen_message_count(self) -> int:
        return 2

    def get_last_original_messages(self) -> list:
        return []

    def get_last_forwarded_messages(self) -> list:
        return []

    def update_from_response(self, **kwargs) -> None:  # noqa: ANN003
        return None


def _response(msg_id: str = "msg_1") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 10, "output_tokens": 3},
        },
    )


def _client(**overrides) -> TestClient:  # noqa: ANN003
    cfg = {
        "optimize": True,
        "cache_enabled": False,
        "rate_limit_enabled": False,
        "cost_tracking_enabled": False,
        "log_requests": False,
        "ccr_inject_tool": False,
        "ccr_handle_responses": False,
        "ccr_context_tracking": False,
        "image_optimize": False,
    }
    cfg.update(overrides)
    return TestClient(create_app(ProxyConfig(**cfg)))


HEADERS = {"x-api-key": "test-key", "anthropic-version": "2023-06-01"}


# Scope the proxy derives for HEADERS on the default tenant and Anthropic upstream.
S = thread_scope(HEADERS, "global", "https://api.anthropic.com")


def _post(client: TestClient, body: dict, headers: dict | None = None) -> httpx.Response:
    return client.post("/v1/messages", headers=headers or HEADERS, json=body)


def _continue_body(messages: list, **extra) -> dict:  # noqa: ANN003
    return {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "thread": dict(CONTINUE),
        "system": [dict(BILLING)],
        "messages": messages,
        **extra,
    }


def _instrument(proxy, captured: dict) -> None:  # noqa: ANN001
    def _apply(**kwargs):  # noqa: ANN003
        captured["frozen_message_count"] = kwargs.get("frozen_message_count")
        captured["tool_scope"] = kwargs.get("tool_scope")
        return SimpleNamespace(
            messages=kwargs["messages"],
            transforms_applied=[],
            timing={},
            tokens_before=50,
            tokens_after=50,
            waste_signals=None,
        )

    proxy.anthropic_pipeline.apply = _apply

    async def _retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001, ANN003
        captured["body"] = body
        captured["calls"] = captured.get("calls", 0) + 1
        return _response(captured.get("resp_id", "msg_1"))

    proxy._retry_request = _retry

    async def _stream(url, headers, body, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        from fastapi.responses import JSONResponse

        captured["body"] = body
        return JSONResponse(content={"ok": True})

    proxy._stream_response = _stream


def test_continue_without_tools_injects_no_tools(monkeypatch) -> None:  # noqa: ANN001
    """Sticky CCR injection must not add `tools` to a tools-less continue turn."""
    injected = {"name": "headroom_retrieve", "description": "d", "input_schema": {"type": "object"}}
    monkeypatch.setattr(
        "headroom.proxy.helpers.apply_session_sticky_ccr_tool",
        lambda **kw: ([*(kw["existing_tools"] or []), injected], True),
    )
    messages = [{"role": "user", "content": "delta"}]
    sent = {}
    for label, thread in (("continue", True), ("control", False)):
        captured: dict = {}
        with _client(ccr_inject_tool=True) as client:
            proxy = client.app.state.proxy
            proxy.config.mode = "token"
            _instrument(proxy, captured)
            body = _continue_body(messages)
            if not thread:
                del body["thread"]
            assert _post(client, body).status_code == 200
        sent[label] = captured["body"]
    assert "tools" not in sent["continue"]
    assert sent["control"].get("tools") == [injected], "control must inject, so the probe bites"


def test_continue_skips_tool_search_and_ccr_history_repairs() -> None:
    """tools is absent on a continue turn; the repairs must not judge blocks by it."""
    captured: dict = {}
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_r", "name": "headroom_retrieve", "input": {}}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_s",
                    "content": [{"type": "tool_reference", "tool_name": "mcp__x__y"}],
                },
                {"type": "tool_result", "tool_use_id": "toolu_r", "content": "retrieved"},
            ],
        },
    ]
    with _client() as client:
        proxy = client.app.state.proxy
        proxy.config.mode = "token"
        _instrument(proxy, captured)
        assert _post(client, _continue_body(messages)).status_code == 200
    sent = captured["body"]["messages"]
    assert sent[0]["content"][0]["type"] == "tool_use"
    assert sent[1]["content"][0]["content"] == [
        {"type": "tool_reference", "tool_name": "mcp__x__y"}
    ]


def test_continue_never_injects_ccr_system_instructions(monkeypatch) -> None:  # noqa: ANN001
    """frozen_message_count is 0 on continue, so only the thread guard stops this."""
    monkeypatch.setattr(
        "headroom.ccr.tool_injection.CCRToolInjector.verify_ownership",
        lambda self, store=None: self._detected_hashes,
    )
    marker = "[10 items compressed to 2. Retrieve more: hash=" + "a" * 24 + "]"
    messages = [{"role": "user", "content": [{"type": "text", "text": marker}]}]
    sent = {}
    for label, thread in (("continue", True), ("control", False)):
        captured: dict = {}
        with _client(ccr_inject_system_instructions=True) as client:
            proxy = client.app.state.proxy
            proxy.config.mode = "token"
            _instrument(proxy, captured)
            body = _continue_body(messages)
            if not thread:
                del body["thread"]
            assert _post(client, body).status_code == 200
        sent[label] = captured["body"]
    assert sent["continue"]["system"] == [BILLING]
    assert sent["control"]["system"] != [BILLING], "control must inject, so the probe bites"


def test_continue_ttl_order_repair_leaves_system_and_tools_untouched() -> None:
    """A 5m system marker before a 1h message marker violates the TTL order; on
    continue only the delta messages may be repaired, never system/tools."""
    five = {"type": "ephemeral"}
    one = {"type": "ephemeral", "ttl": "1h"}
    system = [{**BILLING, "cache_control": dict(five)}]
    tools = [
        {
            "name": "t",
            "description": "d",
            "input_schema": {"type": "object"},
            "cache_control": dict(five),
        }
    ]
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "delta", "cache_control": dict(one)}]}
    ]
    sent = {}
    for label, thread in (("continue", True), ("control", False)):
        captured: dict = {}
        with _client() as client:
            proxy = client.app.state.proxy
            proxy.config.mode = "token"
            _instrument(proxy, captured)
            body = _continue_body(messages, tools=tools)
            body["system"] = system
            if not thread:
                del body["thread"]
            assert _post(client, body).status_code == 200
        sent[label] = captured["body"]
    assert sent["continue"]["system"] == system
    assert sent["continue"]["tools"] == tools
    assert sent["control"]["system"] != system, "control must repair, so the probe bites"


def _rewriting_system_compaction(body, **kwargs):  # noqa: ANN001, ANN003, ANN202
    out = dict(body)
    out["system"] = [{"type": "text", "text": "REWRITTEN"}]
    return out, True, 100, 10


def _send_with_rewrites(monkeypatch, body: dict) -> dict:  # noqa: ANN001
    """Run ``body`` with default tool schema compaction and a forced system rewrite."""
    monkeypatch.setenv("HEADROOM_SYSTEM_COMPACT", "1")
    monkeypatch.setattr(
        "headroom.proxy.system_compaction.compact_system_prompt", _rewriting_system_compaction
    )
    captured: dict = {}
    with _client() as client:
        proxy = client.app.state.proxy
        proxy.config.mode = "token"
        _instrument(proxy, captured)
        assert _post(client, body).status_code == 200
    return captured["body"]


def _tools_needing_compaction() -> list:
    return [
        {
            "name": "t",
            "description": "Does   a\n\n   thing   with   lots   of   whitespace.",
            "input_schema": {"type": "object", "title": "T", "properties": {}},
        }
    ]


def test_continue_restores_system_and_tools_byte_identical(monkeypatch) -> None:  # noqa: ANN001
    tools = _tools_needing_compaction()
    system = [{**BILLING, "text": "billing  header"}]
    messages = [{"role": "user", "content": "delta"}]

    def body() -> dict:
        b = _continue_body(messages, tools=copy.deepcopy(tools))
        b["system"] = copy.deepcopy(system)
        return b

    control = body()
    del control["thread"]
    sent_control = _send_with_rewrites(monkeypatch, control)
    assert sent_control["system"] != system and sent_control["tools"] != tools, (
        "control must be rewritten, so the probe bites"
    )

    sent = _send_with_rewrites(monkeypatch, body())
    assert json.dumps(sent["system"]) == json.dumps(system)
    assert json.dumps(sent["tools"]) == json.dumps(tools)

    # Without the restore the same continue turn is rewritten.
    monkeypatch.setattr("headroom.proxy.handlers.anthropic.restore_thread_pinned", lambda *a: False)
    unguarded = _send_with_rewrites(monkeypatch, body())
    assert unguarded["system"] != system or unguarded["tools"] != tools


def test_continue_stream_restores_system_and_tools_byte_identical(monkeypatch) -> None:  # noqa: ANN001
    tools = _tools_needing_compaction()
    system = [{**BILLING, "text": "billing  header"}]
    body = _continue_body([{"role": "user", "content": "delta"}], tools=copy.deepcopy(tools))
    body["system"] = copy.deepcopy(system)
    body["stream"] = True

    control = copy.deepcopy(body)
    del control["thread"]
    sent_control = _send_with_rewrites(monkeypatch, control)
    assert sent_control["system"] != system and sent_control["tools"] != tools, (
        "control must be rewritten, so the probe bites"
    )

    sent = _send_with_rewrites(monkeypatch, body)
    assert json.dumps(sent["system"]) == json.dumps(system)
    assert json.dumps(sent["tools"]) == json.dumps(tools)


def test_continue_tool_reference_history_is_repaired_against_restored_tools(monkeypatch) -> None:  # noqa: ANN001
    """A mutation adds a tool, so a reference to it looks resolvable mid-request; the
    restore removes it again, so the repair must judge against the stored tools."""
    from headroom.proxy.turn_hooks import clear_turn_hooks, register_turn_hook

    added = {"name": "added", "description": "d", "input_schema": {"type": "object"}}
    ran: list[int] = []

    class _AddToolHook:
        def on_request(self, ctx) -> None:  # noqa: ANN001
            ctx.tools = [*(ctx.tools or []), copy.deepcopy(added)]
            ran.append(1)

    clear_turn_hooks()
    register_turn_hook(_AddToolHook())
    tools = [{"name": "t", "description": "d", "input_schema": {"type": "object"}}]
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_s",
                    "content": [{"type": "tool_reference", "tool_name": "added"}],
                }
            ],
        }
    ]
    captured: dict = {}
    with _client() as client:
        proxy = client.app.state.proxy
        proxy.config.mode = "token"
        _instrument(proxy, captured)
        assert _post(client, _continue_body(messages, tools=tools)).status_code == 200
    sent = captured["body"]
    assert ran, "the mutation must run, so the probe bites"
    assert sent["tools"] == tools
    assert "tool_reference" not in json.dumps(sent["messages"])
    clear_turn_hooks()


def _run_turn(body: dict, resp_id: str, patches=None, headers: dict | None = None) -> dict:  # noqa: ANN001
    """POST ``body`` through the handler and return the forwarded body."""
    captured: dict = {"resp_id": resp_id}
    with _client(ccr_inject_tool=True) as client:
        proxy = client.app.state.proxy
        proxy.config.mode = "token"
        _instrument(proxy, captured)
        assert _post(client, body, headers).status_code == 200
    return captured["body"]


_RETRIEVE = {"name": "headroom_retrieve", "description": "d", "input_schema": {"type": "object"}}


def _sticky(monkeypatch, inject: bool, front: bool = False) -> None:  # noqa: ANN001
    def _apply(**kw):  # noqa: ANN003, ANN202
        tools = list(kw["existing_tools"] or [])
        if not inject:
            return tools, False
        return ([_RETRIEVE, *tools] if front else [*tools, _RETRIEVE]), True

    monkeypatch.setattr("headroom.proxy.helpers.apply_session_sticky_ccr_tool", _apply)


def test_continue_pins_system_and_tools_forwarded_on_the_create_turn(monkeypatch) -> None:  # noqa: ANN001
    """Live shape: every turn's system is rewritten differently (shaper stratum) and
    tools are compacted; the API compares against what was FORWARDED last turn."""
    calls: list[int] = []

    def _shaping(body, **kwargs):  # noqa: ANN001, ANN003, ANN202
        calls.append(1)
        out = dict(body)
        out["system"] = [*body["system"], {"type": "text", "text": f"steer {len(calls)}"}]
        return out, True, 100, 10

    monkeypatch.setenv("HEADROOM_SYSTEM_COMPACT", "1")
    monkeypatch.setattr("headroom.proxy.system_compaction.compact_system_prompt", _shaping)
    _sticky(monkeypatch, inject=False)
    tools = _tools_needing_compaction()
    system = [dict(BILLING), {"type": "text", "text": "full  system"}]
    messages = [{"role": "user", "content": "x"}]

    create = _continue_body(messages, tools=copy.deepcopy(tools))
    create["system"] = copy.deepcopy(system)
    create["thread"] = {"type": "create"}
    forwarded_create = _run_turn(create, "msg_live_1")
    assert forwarded_create["system"] != system and forwarded_create["tools"] != tools

    cont = _continue_body(messages, tools=copy.deepcopy(tools))
    cont["system"] = copy.deepcopy(system)
    cont["thread"] = {"type": "continue", "previous_message_id": "msg_live_1"}
    sent = _run_turn(cont, "msg_live_2")
    assert len(calls) == 2, "the continue turn must be rewritten too, so the probe bites"
    assert json.dumps(sent["system"]) == json.dumps(forwarded_create["system"])
    assert json.dumps(sent["tools"]) == json.dumps(forwarded_create["tools"])

    # Chained: the next continue pins the previous continue's forwarded values.
    cont2 = copy.deepcopy(cont)
    cont2["thread"] = {"type": "continue", "previous_message_id": "msg_live_2"}
    sent2 = _run_turn(cont2, "msg_live_3")
    assert json.dumps(sent2["system"]) == json.dumps(forwarded_create["system"])


def test_continue_pins_the_tools_recorded_from_the_create_turn(monkeypatch) -> None:  # noqa: ANN001
    """Create forwarded no retrieve tool; a later continue with tools must not gain one."""
    tools = _tools_needing_compaction()
    messages = [{"role": "user", "content": "x"}]
    _sticky(monkeypatch, inject=False)
    create = _continue_body(messages, tools=copy.deepcopy(tools))
    create["thread"] = {"type": "create"}
    forwarded_create = _run_turn(create, "msg_create_a")
    assert forwarded_create["tools"] != tools, "create must be compacted, so the probe bites"

    _sticky(monkeypatch, inject=True)
    cont = _continue_body(messages, tools=copy.deepcopy(tools))
    cont["thread"] = {"type": "continue", "previous_message_id": "msg_create_a"}
    sent = _run_turn(cont, "msg_cont_a")
    assert json.dumps(sent["tools"]) == json.dumps(forwarded_create["tools"])
    assert all(t["name"] != "headroom_retrieve" for t in sent["tools"])


def test_continue_pins_recorded_tools_with_retrieve_and_compaction(monkeypatch) -> None:  # noqa: ANN001
    tools = [
        {"name": "zeta", "description": "a   b", "input_schema": {"type": "object", "title": "T"}},
        *_tools_needing_compaction(),
    ]
    messages = [{"role": "user", "content": "x"}]
    _sticky(monkeypatch, inject=True, front=True)
    create = _continue_body(messages, tools=copy.deepcopy(tools))
    create["thread"] = {"type": "create"}
    forwarded_create = _run_turn(create, "msg_create_b")
    assert any(t["name"] == "headroom_retrieve" for t in forwarded_create["tools"])
    assert forwarded_create["tools"] != tools

    _sticky(monkeypatch, inject=True, front=False)  # would land elsewhere if not pinned
    cont = _continue_body(messages, tools=copy.deepcopy(tools))
    cont["thread"] = {"type": "continue", "previous_message_id": "msg_create_b"}
    sent = _run_turn(cont, "msg_cont_b")
    assert json.dumps(sent["tools"]) == json.dumps(forwarded_create["tools"])


def test_continue_without_record_keeps_retrieve_only_if_history_references_it(
    monkeypatch,  # noqa: ANN001
) -> None:
    tools = _tools_needing_compaction()
    _sticky(monkeypatch, inject=True)
    plain = [{"role": "user", "content": "x"}]
    refs = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_r", "name": "headroom_retrieve", "input": {}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_r", "content": "r"}],
        },
    ]
    names = {}
    for label, messages in (("plain", plain), ("refs", refs)):
        cont = _continue_body(messages, tools=copy.deepcopy(tools))
        cont["thread"] = {"type": "continue", "previous_message_id": "msg_never_seen"}
        names[label] = [t["name"] for t in _run_turn(cont, "msg_x_" + label)["tools"]]
    assert "headroom_retrieve" not in names["plain"]
    assert "headroom_retrieve" in names["refs"]


def _capture_outcomes(proxy) -> list:  # noqa: ANN001
    outcomes: list = []
    original = proxy._record_request_outcome

    async def _spy(outcome):  # noqa: ANN001, ANN202
        outcomes.append(outcome)
        return await original(outcome)

    proxy._record_request_outcome = _spy
    return outcomes


def test_continue_reports_no_savings_for_discarded_tool_compaction() -> None:
    tools = _tools_needing_compaction()
    saved = {}
    for label, thread in (("continue", True), ("control", False)):
        captured: dict = {}
        with _client() as client:
            proxy = client.app.state.proxy
            proxy.config.mode = "token"
            _instrument(proxy, captured)
            outcomes = _capture_outcomes(proxy)
            body = _continue_body(
                [{"role": "user", "content": "delta"}], tools=copy.deepcopy(tools)
            )
            if not thread:
                del body["thread"]
            assert _post(client, body).status_code == 200
        saved[label] = outcomes[-1].tokens_saved
    assert saved["control"] > 0, "control must book tool-compaction savings, so the probe bites"
    assert saved["continue"] == 0


def test_continue_never_repairs_the_stored_tools_array() -> None:
    search = {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}
    ref = {"type": "tool_reference", "tool_name": "tool_search_tool_regex"}
    tools = [search, ref]
    labels = {}
    for label, thread in (("continue", True), ("control", False)):
        captured: dict = {}
        with _client() as client:
            proxy = client.app.state.proxy
            proxy.config.mode = "token"
            _instrument(proxy, captured)
            outcomes = _capture_outcomes(proxy)
            body = _continue_body([{"role": "user", "content": "d"}], tools=copy.deepcopy(tools))
            if not thread:
                del body["thread"]
            assert _post(client, body).status_code == 200
        labels[label] = [t for t in outcomes[-1].transforms_applied if "ref_repair" in t]
        if thread:
            assert captured["body"]["tools"] == tools
    assert labels["control"], "control must repair, so the probe bites"
    assert labels["continue"] == []


def _create_then_continue(monkeypatch, create_overrides: dict, cont_overrides: dict):  # noqa: ANN001, ANN202
    """Run a create turn then a continue naming it; return (forwarded create, forwarded continue)."""
    _sticky(monkeypatch, inject=False)
    tools = _tools_needing_compaction()
    messages = [{"role": "user", "content": "x"}]
    create = _continue_body(messages, tools=copy.deepcopy(tools))
    create["thread"] = {"type": "create"}
    create.update(create_overrides)
    forwarded_create = _run_turn(create, "msg_pin_a")
    cont = _continue_body(messages, tools=copy.deepcopy(tools))
    cont["thread"] = {"type": "continue", "previous_message_id": "msg_pin_a"}
    cont.update(cont_overrides)
    return forwarded_create, _run_turn(cont, "msg_pin_b")


def test_continue_pins_an_absent_recorded_system_even_if_client_sends_one(monkeypatch) -> None:  # noqa: ANN001
    body_no_system = {"system": None}

    def _drop(body, **kwargs):  # noqa: ANN001, ANN003, ANN202
        out = dict(body)
        del out["system"]
        return out, True, 100, 10

    monkeypatch.setenv("HEADROOM_SYSTEM_COMPACT", "1")
    monkeypatch.setattr("headroom.proxy.system_compaction.compact_system_prompt", _drop)
    created, sent = _create_then_continue(monkeypatch, {}, {})
    assert "system" not in created
    assert "system" not in sent, body_no_system


def test_recorded_tools_win_over_the_history_reference_fallback(monkeypatch) -> None:  # noqa: ANN001
    _sticky(monkeypatch, inject=False)
    tools = _tools_needing_compaction()
    create = _continue_body([{"role": "user", "content": "x"}], tools=copy.deepcopy(tools))
    create["thread"] = {"type": "create"}
    forwarded_create = _run_turn(create, "msg_pin_c")
    _sticky(monkeypatch, inject=True)
    refs = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_r", "name": "headroom_retrieve", "input": {}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_r", "content": "r"}],
        },
    ]
    cont = _continue_body(refs, tools=copy.deepcopy(tools))
    cont["thread"] = {"type": "continue", "previous_message_id": "msg_pin_c"}
    sent = _run_turn(cont, "msg_pin_d")
    assert json.dumps(sent["tools"]) == json.dumps(forwarded_create["tools"])
    assert all(t["name"] != "headroom_retrieve" for t in sent["tools"])


def test_recording_is_capped_and_the_oldest_falls_back_to_client_values(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(tool_name_registry, "_THREAD_MAX", 2)
    _sticky(monkeypatch, inject=False)
    tools = _tools_needing_compaction()
    messages = [{"role": "user", "content": "x"}]
    forwarded = {}
    for i in range(3):
        create = _continue_body(messages, tools=copy.deepcopy(tools))
        create["thread"] = {"type": "create"}
        forwarded[i] = _run_turn(create, f"msg_cap_{i}")
    for i, pinned in ((2, True), (0, False)):
        cont = _continue_body(messages, tools=copy.deepcopy(tools))
        cont["thread"] = {"type": "continue", "previous_message_id": f"msg_cap_{i}"}
        sent = _run_turn(cont, f"msg_cap_next_{i}")
        assert (sent["tools"] == forwarded[i]["tools"]) is pinned
        if not pinned:
            assert sent["tools"] == tools, "evicted: falls back to the client's own tools"


class _FakeUpstream:
    """Stands in for proxy.http_client on the streaming path; records request bodies."""

    def __init__(self, sse: bytes) -> None:
        self.sse = sse
        self.bodies: list[dict] = []

    def build_request(self, method, url, content=None, headers=None):  # noqa: ANN001, ANN201
        self.bodies.append(json.loads(content))
        return httpx.Request(method, url)

    async def send(self, request, stream=False):  # noqa: ANN001, ANN201, ARG002
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=self.sse)


def _sse(msg_id: str) -> bytes:
    events = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": 5, "output_tokens": 1},
                },
            },
        ),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 2},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return b"".join(f"event: {n}\ndata: {json.dumps(d)}\n\n".encode() for n, d in events)


def test_streaming_create_then_continue_pins_forwarded_system_and_tools(monkeypatch) -> None:  # noqa: ANN001
    calls: list[int] = []

    def _shaping(body, **kwargs):  # noqa: ANN001, ANN003, ANN202
        calls.append(1)
        out = dict(body)
        out["system"] = [*body["system"], {"type": "text", "text": f"steer {len(calls)}"}]
        return out, True, 100, 10

    monkeypatch.setenv("HEADROOM_SYSTEM_COMPACT", "1")
    monkeypatch.setattr("headroom.proxy.system_compaction.compact_system_prompt", _shaping)
    tools = _tools_needing_compaction()
    system = [dict(BILLING), {"type": "text", "text": "full  system"}]
    messages = [{"role": "user", "content": "x"}]
    upstream = _FakeUpstream(_sse("msg_stream_1"))
    with _client() as client:
        proxy = client.app.state.proxy
        proxy.config.mode = "token"
        _instrument(proxy, {})
        del proxy._stream_response  # the real streaming handler under test
        proxy.http_client = upstream
        create = _continue_body(messages, tools=copy.deepcopy(tools), stream=True)
        create["system"] = copy.deepcopy(system)
        create["thread"] = {"type": "create"}
        assert _post(client, create).status_code == 200
        cont = _continue_body(messages, tools=copy.deepcopy(tools), stream=True)
        cont["system"] = copy.deepcopy(system)
        cont["thread"] = {"type": "continue", "previous_message_id": "msg_stream_1"}
        assert _post(client, cont).status_code == 200
    created, continued = upstream.bodies
    assert created["system"] != system and created["tools"] != tools
    assert len(calls) == 2, "the continue turn must be rewritten too, so the probe bites"
    assert json.dumps(continued["system"]) == json.dumps(created["system"])
    assert json.dumps(continued["tools"]) == json.dumps(created["tools"])


class _FakeCcrHandler:
    config = SimpleNamespace(enabled=True)

    def has_ccr_tool_calls(self, resp, provider) -> bool:  # noqa: ANN001
        return resp.get("id") == "msg_ccr_first"

    def residual_ccr_status(self, resp, provider):  # noqa: ANN001, ANN201
        return None

    async def handle_response(self, resp, messages, tools, api_call_fn, provider="anthropic"):  # noqa: ANN001, ANN201
        other_tools = [{"name": "other", "description": "d", "input_schema": {"type": "object"}}]
        return await api_call_fn([*messages, {"role": "user", "content": "more"}], other_tools)


def test_ccr_continuation_on_a_continue_turn_keeps_pinned_tools_and_records_final_id(
    monkeypatch,  # noqa: ANN001
) -> None:
    tools = _tools_needing_compaction()
    messages = [{"role": "user", "content": "x"}]
    posted: list[dict] = []

    class _Http:
        async def post(self, url, content=None, headers=None, timeout=None):  # noqa: ANN001, ANN201, ARG002
            posted.append(json.loads(content))
            return _response("msg_ccr_final")

    captured: dict = {"resp_id": "msg_ccr_first"}
    with _client() as client:
        proxy = client.app.state.proxy
        proxy.config.mode = "token"
        _instrument(proxy, captured)
        proxy.ccr_response_handler = _FakeCcrHandler()
        proxy.http_client = _Http()
        cont = _continue_body(messages, tools=copy.deepcopy(tools))
        cont["thread"] = {"type": "continue", "previous_message_id": "msg_prev_unrecorded"}
        resp = _post(client, cont)
        assert resp.status_code == 200, resp.text
        assert resp.json()["id"] == "msg_ccr_final"
        assert json.dumps(posted[0]["tools"]) == json.dumps(tools)
        assert [t["name"] for t in posted[0]["tools"]] != ["other"]
        # The id the client received is the one a later continue can name.
        assert tool_name_registry.lookup_thread_pinned(S, "msg_ccr_first") is None
        assert tool_name_registry.lookup_thread_pinned(S, "msg_ccr_final") is not None
        captured["resp_id"] = "msg_after"
        nxt = _continue_body(messages, tools=copy.deepcopy(tools))
        nxt["thread"] = {"type": "continue", "previous_message_id": "msg_ccr_final"}
        assert _post(client, nxt).status_code == 200
    assert json.dumps(captured["body"]["tools"]) == json.dumps(tools)


def test_ccr_continuation_on_a_create_turn_records_the_tools_it_actually_sent(monkeypatch) -> None:  # noqa: ANN001
    _sticky(monkeypatch, inject=False)
    tools = _tools_needing_compaction()
    messages = [{"role": "user", "content": "x"}]
    posted: list[dict] = []

    class _Http:
        async def post(self, url, content=None, headers=None, timeout=None):  # noqa: ANN001, ANN201, ARG002
            posted.append(json.loads(content))
            return _response("msg_ccr_final2")

    captured: dict = {"resp_id": "msg_ccr_first"}
    with _client() as client:
        proxy = client.app.state.proxy
        proxy.config.mode = "token"
        _instrument(proxy, captured)
        proxy.ccr_response_handler = _FakeCcrHandler()
        proxy.http_client = _Http()
        create = _continue_body(messages, tools=copy.deepcopy(tools))
        create["thread"] = {"type": "create"}
        assert _post(client, create).json()["id"] == "msg_ccr_final2"
        assert [t["name"] for t in posted[0]["tools"]] == ["other"], (
            "create keeps the handler's tools"
        )
        captured["resp_id"] = "msg_after2"
        cont = _continue_body(messages, tools=copy.deepcopy(tools))
        cont["thread"] = {"type": "continue", "previous_message_id": "msg_ccr_final2"}
        assert _post(client, cont).status_code == 200
    assert [t["name"] for t in captured["body"]["tools"]] == ["other"]


class _FailingCcrHandler(_FakeCcrHandler):
    """Like the real handler: a failed continuation returns the earlier response."""

    async def handle_response(self, resp, messages, tools, api_call_fn, provider="anthropic"):  # noqa: ANN001, ANN201
        try:
            return await super().handle_response(resp, messages, tools, api_call_fn, provider)
        except Exception:
            return resp


def test_failed_ccr_continuation_keeps_the_original_id_and_its_forwarded_tools(monkeypatch) -> None:  # noqa: ANN001
    _sticky(monkeypatch, inject=False)
    tools = _tools_needing_compaction()
    messages = [{"role": "user", "content": "x"}]

    class _Http:
        async def post(self, url, content=None, headers=None, timeout=None):  # noqa: ANN001, ANN201, ARG002
            raise httpx.ConnectError("boom")

    captured: dict = {"resp_id": "msg_ccr_first"}
    with _client() as client:
        proxy = client.app.state.proxy
        proxy.config.mode = "token"
        _instrument(proxy, captured)
        proxy.ccr_response_handler = _FailingCcrHandler()
        proxy.http_client = _Http()
        create = _continue_body(messages, tools=copy.deepcopy(tools))
        create["thread"] = {"type": "create"}
        resp = _post(client, create)
        assert resp.status_code == 200, resp.text
        assert resp.json()["id"] == "msg_ccr_first"
        forwarded = captured["body"]["tools"]
        recorded = tool_name_registry.lookup_thread_pinned(S, "msg_ccr_first")
    assert [t["name"] for t in forwarded] != ["other"]
    assert json.dumps(recorded["tools"]) == json.dumps(forwarded)


class _FakeMemory:
    config = SimpleNamespace(project_root_override="")

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def has_memory_tool_calls(self, resp, provider) -> bool:  # noqa: ANN001
        return resp.get("id") == "msg_mem_first"

    async def handle_memory_tool_calls(
        self, resp, user_id, provider="anthropic", request_context=None
    ):  # noqa: ANN001, ANN201
        return [{"type": "tool_result", "tool_use_id": "toolu_m", "content": "ok"}]


DIVERGED = [{"name": "diverged", "description": "d", "input_schema": {"type": "object"}}]


def _memory_turn(monkeypatch, body: dict, fail: bool) -> tuple[dict, dict]:  # noqa: ANN001
    """POST ``body`` with a fake memory handler whose continuation succeeds or raises.

    The handler keeps ``tools`` (a local) in step with ``body["tools"]`` everywhere, so
    the two are made to differ here: a stand-in restore rewrites ``body["tools"]`` only.
    Which of the two the memory continuation sends and records is then observable.
    """

    def _diverge(b, snap, keep=None):  # noqa: ANN001, ANN202, ARG001
        b["tools"] = copy.deepcopy(DIVERGED)
        return False

    monkeypatch.setattr("headroom.proxy.handlers.anthropic.restore_thread_pinned", _diverge)
    for name in ("_thread_pinned", "_blobs", "_blob_refs"):  # the registry is process-global
        monkeypatch.setattr(tool_name_registry, name, type(getattr(tool_name_registry, name))())
    monkeypatch.setattr("headroom.proxy.handlers.anthropic.resolve_memory_identity", lambda r: "u")
    monkeypatch.setattr(
        "headroom.proxy.handlers.anthropic.MemoryDecision.decide",
        lambda **kw: SimpleNamespace(apply_to_tags=lambda tags: None, inject=False),
    )
    captured: dict = {"resp_id": "msg_mem_first"}
    sent: list[dict] = []
    with _client() as client:
        proxy = client.app.state.proxy
        proxy.config.mode = "token"
        _instrument(proxy, captured)
        first = proxy._retry_request

        async def _retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001, ANN003
            sent.append(body)
            if len(sent) == 1:
                return await first(method, url, headers, body, stream=stream, **kwargs)
            if fail:
                raise httpx.ConnectError("boom")
            return _response("msg_mem_final")

        proxy._retry_request = _retry
        proxy.memory_handler = _FakeMemory()
        resp = _post(client, body)
        assert resp.status_code == 200, resp.text
        recorded = {
            i: tool_name_registry.lookup_thread_pinned(S, i)
            for i in ("msg_mem_first", "msg_mem_final")
        }
    return {"sent": sent, "id": resp.json()["id"]}, recorded


def test_memory_continuation_on_a_continue_turn_keeps_pinned_tools_and_records_final_id(
    monkeypatch,  # noqa: ANN001
) -> None:
    tools = _tools_needing_compaction()
    cont = _continue_body([{"role": "user", "content": "x"}], tools=copy.deepcopy(tools))
    cont["thread"] = {"type": "continue", "previous_message_id": "msg_unrecorded"}
    out, recorded = _memory_turn(monkeypatch, cont, fail=False)
    assert out["id"] == "msg_mem_final" and len(out["sent"]) == 2
    assert out["sent"][0]["tools"] == DIVERGED
    assert out["sent"][1]["tools"] == DIVERGED, (
        "a continue keeps the pinned tools, not the stale local"
    )
    assert recorded["msg_mem_first"] is None and recorded["msg_mem_final"]["tools"] == DIVERGED


def test_memory_continuation_on_a_create_turn_records_the_continuation_it_sent(monkeypatch) -> None:  # noqa: ANN001
    _sticky(monkeypatch, inject=False)
    create = _continue_body([{"role": "user", "content": "x"}], tools=_tools_needing_compaction())
    create["thread"] = {"type": "create"}
    out, recorded = _memory_turn(monkeypatch, create, fail=False)
    assert out["id"] == "msg_mem_final"
    assert out["sent"][0]["tools"] == DIVERGED and out["sent"][1]["tools"] != DIVERGED
    assert recorded["msg_mem_final"]["tools"] == out["sent"][1]["tools"]


def test_failed_memory_continuation_keeps_the_original_id_and_forwarded_values(monkeypatch) -> None:  # noqa: ANN001
    _sticky(monkeypatch, inject=False)
    create = _continue_body([{"role": "user", "content": "x"}], tools=_tools_needing_compaction())
    create["thread"] = {"type": "create"}
    out, recorded = _memory_turn(monkeypatch, create, fail=True)
    assert out["id"] == "msg_mem_first"
    assert recorded["msg_mem_final"] is None
    assert out["sent"][1]["tools"] != DIVERGED
    assert recorded["msg_mem_first"]["tools"] == out["sent"][0]["tools"] == DIVERGED


class _RewriteAfterRepairHook:
    """Turn hook: adds a tool, so tools differ from the stored thread."""

    def on_request(self, ctx) -> None:  # noqa: ANN001
        ctx.tools = [*(ctx.tools or []), {"name": "x", "description": "d", "input_schema": {}}]


@pytest.mark.parametrize("via_backend", [False, True])
def test_late_restore_undoes_system_relocation(monkeypatch, via_backend: bool) -> None:  # noqa: ANN001, FBT001
    """A role=system message in the delta is relocated into ``system`` after the early
    restore; only the restore before the backend fork puts the stored system back."""
    from headroom.backends.base import BackendResponse
    from headroom.proxy.turn_hooks import clear_turn_hooks, register_turn_hook

    sent: dict = {}
    if via_backend:

        class _Backend:
            name = "fake"

            async def send_message(self, body, headers):  # noqa: ANN001, ANN202
                sent["body"] = body
                return BackendResponse(body=_response().json(), status_code=200)

        monkeypatch.setattr(
            "headroom.proxy.route_advice.resolver_for",
            lambda proxy: SimpleNamespace(for_request=lambda request, body=None: _Backend()),
        )
    clear_turn_hooks()
    register_turn_hook(_RewriteAfterRepairHook())
    tools = _tools_needing_compaction()
    messages = [
        {"role": "system", "content": "relocate me"},
        {"role": "user", "content": "delta"},
    ]
    body = _continue_body(messages, tools=copy.deepcopy(tools))
    try:
        with _client() as client:
            proxy = client.app.state.proxy
            proxy.config.mode = "token"
            _instrument(proxy, sent)
            assert _post(client, body).status_code == 200
    finally:
        clear_turn_hooks()
    out = sent["body"]
    assert out["system"] == [BILLING]
    assert json.dumps(out["tools"]) == json.dumps(tools)


def test_continue_absent_tools_stays_absent_through_restore(monkeypatch) -> None:  # noqa: ANN001
    sent = _send_with_rewrites(monkeypatch, _continue_body([{"role": "user", "content": "d"}]))
    assert "tools" not in sent


def test_continue_forces_frozen_count_zero_and_bypasses_tracker_store() -> None:
    captured: dict = {}
    store_calls: list = []
    messages = [
        {"role": "user", "content": "earlier delta"},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "x"}]},
    ]
    with _client() as client:
        proxy = client.app.state.proxy
        proxy.config.mode = "cache"
        _instrument(proxy, captured)
        store = proxy.session_tracker_store
        store.compute_session_id = lambda request, model, messages: "sess"

        def _fake_get(session_id, provider):  # noqa: ANN001
            store_calls.append(session_id)
            return _FakeTracker()

        store.get_or_create = _fake_get
        assert _post(client, _continue_body(messages)).status_code == 200
    assert captured["frozen_message_count"] == 0
    assert store_calls == [], "continue deltas must never touch the content-keyed store"


def test_non_continue_still_uses_the_tracker_store(monkeypatch) -> None:  # noqa: ANN001
    """Control: the same request without `thread` resolves a stored, fully frozen tracker."""
    decision = SimpleNamespace(should_compress=True, passthrough_reason=None)
    decision.apply_to_tags = lambda tags: None
    monkeypatch.setattr(
        "headroom.proxy.handlers.anthropic.CompressionDecision.decide", lambda **kw: decision
    )
    captured: dict = {}
    store_calls: list = []
    messages = [
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
    ]
    with _client() as client:
        proxy = client.app.state.proxy
        proxy.config.mode = "cache"
        _instrument(proxy, captured)
        store = proxy.session_tracker_store
        store.compute_session_id = lambda request, model, messages: "sess"

        def _fake_get(session_id, provider):  # noqa: ANN001
            store_calls.append(session_id)
            return _FakeTracker()

        store.get_or_create = _fake_get
        body = _continue_body(messages)
        del body["thread"]
        assert _post(client, body).status_code == 200
    # Everything frozen -> the pipeline has nothing mutable and is skipped: the
    # exact outcome a continue delta used to get (56/56 measured turns).
    assert store_calls and "frozen_message_count" not in captured


def test_response_cache_key_includes_thread() -> None:
    captured: dict = {}
    delta = [
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}]}
    ]
    with _client(cache_enabled=True) as client:
        proxy = client.app.state.proxy
        proxy.config.mode = "token"
        _instrument(proxy, captured)
        a = _continue_body(delta)
        b = _continue_body(delta)
        b["thread"] = {"type": "continue", "previous_message_id": "msg_OTHER"}
        assert _post(client, a).status_code == 200
        assert _post(client, b).status_code == 200
    assert captured["calls"] == 2, "different threads must not share a cached response"


# 6. tool-name registry -----------------------------------------------------


def _sse_start(tool_id: str, name: str) -> bytes:
    return (
        b"event: content_block_start\ndata: "
        b'{"type":"content_block_start","index":1,"content_block":'
        b'{"type":"tool_use","id":"'
        + tool_id.encode()
        + b'","name":"'
        + name.encode()
        + b'","input":{}}}\n\n'
    )


def test_registry_records_and_looks_up_from_sse() -> None:
    tool_name_registry.record_from_sse(S, _sse_start("toolu_reg1", "Edit"))
    assert tool_name_registry.lookup(S, "toolu_reg1") == "Edit"
    assert tool_name_registry.lookup(S, "toolu_missing") is None


def test_registry_sse_is_field_order_independent() -> None:
    raw = (
        b'data: {"content_block":{"name":"Glob","input":{},"id":"toolu_order","type":"tool_use"},'
        b'"index":0,"type":"content_block_start"}\n\n'
    )
    tool_name_registry.record_from_sse(S, raw)
    assert tool_name_registry.lookup(S, "toolu_order") == "Glob"


def test_registry_finds_event_split_across_chunks_via_returned_rest() -> None:
    raw = _sse_start("toolu_split", "Grep")
    cut = len(raw) // 2
    rest = tool_name_registry.record_from_sse(S, raw[:cut])
    assert tool_name_registry.lookup(S, "toolu_split") is None
    tool_name_registry.record_from_sse(S, rest + raw[cut:])
    assert tool_name_registry.lookup(S, "toolu_split") == "Grep"


def test_registry_records_from_non_streaming_json() -> None:
    tool_name_registry.record_from_json(
        S,
        {
            "type": "message",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "tool_use", "name": "Write", "id": "toolu_json", "input": {}},
            ],
        },
    )
    assert tool_name_registry.lookup(S, "toolu_json") == "Write"
    tool_name_registry.record_from_json(S, None)  # tolerated


def test_registry_records_forwarded_system_and_tools_by_message_id() -> None:
    body = {"thread": CONTINUE, "system": [{"type": "text", "text": "s"}], "tools": [{"name": "t"}]}
    pinned = tool_name_registry.thread_pinned_of(body)
    raw = b'data: {"type":"message_start","message":{"id":"msg_sse_pin"}}\n\n'
    tool_name_registry.record_from_sse(S, raw, pinned)
    body["tools"].append({"name": "mutated after"})
    found = tool_name_registry.lookup_thread_pinned(S, "msg_sse_pin")
    assert found == {"system": [{"type": "text", "text": "s"}], "tools": [{"name": "t"}]}
    found["tools"].append("caller mutation")
    assert tool_name_registry.lookup_thread_pinned(S, "msg_sse_pin")["tools"] == [{"name": "t"}]
    assert tool_name_registry.lookup_thread_pinned(S, "msg_unknown") is None
    assert tool_name_registry.thread_pinned_of({"system": "no thread"}) is None


def test_registry_interns_blobs_and_frees_unreferenced_ones(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(tool_name_registry, "_THREAD_MAX", 2)
    for name in ("_thread_pinned", "_blobs", "_blob_refs"):
        monkeypatch.setattr(tool_name_registry, name, type(getattr(tool_name_registry, name))())
    shared = {"thread": CONTINUE, "system": "same", "tools": [{"name": "same"}]}
    for i in range(2):
        tool_name_registry.record_thread_pinned(
            S, f"msg_int_{i}", tool_name_registry.thread_pinned_of(shared)
        )
    assert len(tool_name_registry._blobs) == 2, "identical blobs are held once"
    for i in range(2, 4):  # evicts both, with their blobs
        tool_name_registry.record_thread_pinned(
            S,
            f"msg_int_{i}",
            tool_name_registry.thread_pinned_of({"thread": CONTINUE, "system": i}),
        )
    assert tool_name_registry.lookup_thread_pinned(S, "msg_int_0") is None
    assert len(tool_name_registry._blobs) == 2  # only the two newest "system" blobs


def test_registry_is_bounded(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(tool_name_registry, "_MAX", 3)
    for i in range(5):
        tool_name_registry.record(S, f"toolu_lru{i}", "Bash")
    assert tool_name_registry.lookup(S, "toolu_lru0") is None
    assert tool_name_registry.lookup(S, "toolu_lru4") == "Bash"


def _result_msg(tool_id: str) -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": "body"}],
    }


def test_router_name_map_uses_registry_then_fails_safe_to_read_on_continue() -> None:
    tool_name_registry.record(S, "toolu_known", "Grep")
    router = ContentRouter()
    mapping = router._build_tool_name_map(
        [_result_msg("toolu_known"), _result_msg("toolu_never_seen")],
        thread_continue=True,
        tool_scope=S,
    )
    assert mapping["toolu_known"] == "Grep"
    assert mapping["toolu_never_seen"] == "Read"


def test_router_orphan_tool_result_without_continue_flag_gets_no_read_fallback() -> None:
    """Non-thread requests keep their old orphan behavior (no name, so not excluded)."""
    mapping = ContentRouter()._build_tool_name_map([_result_msg("toolu_orphan_plain")])
    assert "toolu_orphan_plain" not in mapping


def test_orphan_tool_result_unknown_id_in_continue_maps_to_read() -> None:
    mapping = ContentRouter()._build_tool_name_map(
        [_result_msg("toolu_orphan_unknown_zz")], thread_continue=True
    )
    assert mapping == {"toolu_orphan_unknown_zz": "Read"}


def test_router_name_map_prefers_in_request_tool_use_and_handles_openai_tool_role() -> None:
    router = ContentRouter()
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_here", "name": "Bash", "input": {}}],
        },
        _result_msg("toolu_here"),
        {"role": "tool", "tool_call_id": "call_unknown", "content": "x"},
    ]
    mapping = router._build_tool_name_map(messages)
    assert mapping["toolu_here"] == "Bash"
    assert "call_unknown" not in mapping  # OpenAI orphans keep prior behavior
    tool_name_registry.record(S, "call_learned", "Edit")
    learned = [{"role": "tool", "tool_call_id": "call_learned", "content": "x"}]
    assert router._build_tool_name_map(learned, tool_scope=S)["call_learned"] == "Edit"


# ---- Caller scoping: identical ids from two principals must not see each other ----

HEADERS_B = {"x-api-key": "other-key", "anthropic-version": "2023-06-01"}
SB = thread_scope(HEADERS_B, "global", "https://api.anthropic.com")


def test_thread_scope_separates_credential_tenant_and_upstream_and_hides_the_secret() -> None:
    up = "https://api.anthropic.com"
    base = thread_scope(HEADERS, "global", up)
    assert base == thread_scope(dict(HEADERS), "global", up)
    assert len({base, thread_scope(HEADERS_B, "global", up), thread_scope(HEADERS, "t1", up)}) == 3
    assert thread_scope(HEADERS, "global", "https://gw.example") != base
    bearer = {"authorization": "Bearer test-key"}
    assert thread_scope(bearer, "global", up) != base
    assert thread_scope(bearer, "global", up) != thread_scope(
        {"authorization": "Bearer other-key"}, "global", up
    )
    assert "test-key" not in base


def test_registry_pins_blobs_and_names_do_not_cross_principals(monkeypatch) -> None:  # noqa: ANN001
    for name in ("_names", "_thread_pinned", "_blobs", "_blob_refs"):
        monkeypatch.setattr(tool_name_registry, name, type(getattr(tool_name_registry, name))())
    a_body = {"thread": CONTINUE, "system": "A secret system", "tools": [{"name": "a"}]}
    b_body = {"thread": CONTINUE, "system": "B system", "tools": [{"name": "b"}]}
    tool_name_registry.record_thread_pinned(
        S, "msg_same", tool_name_registry.thread_pinned_of(a_body)
    )
    tool_name_registry.record(S, "toolu_same", "Edit")
    assert tool_name_registry.lookup_thread_pinned(SB, "msg_same") is None
    assert tool_name_registry.lookup(SB, "toolu_same") is None
    # B recording the SAME ids neither replaces nor frees A's entries.
    tool_name_registry.record_thread_pinned(
        SB, "msg_same", tool_name_registry.thread_pinned_of(b_body)
    )
    tool_name_registry.record(SB, "toolu_same", "Bash")
    assert tool_name_registry.lookup_thread_pinned(S, "msg_same")["system"] == "A secret system"
    assert tool_name_registry.lookup_thread_pinned(SB, "msg_same")["system"] == "B system"
    assert tool_name_registry.lookup(S, "toolu_same") == "Edit"
    assert tool_name_registry.lookup(SB, "toolu_same") == "Bash"
    # Identical blobs are interned per scope, never shared across principals.
    tool_name_registry.record_thread_pinned(
        SB, "msg_b2", tool_name_registry.thread_pinned_of(a_body)
    )
    assert len(tool_name_registry._blobs) == 6
    # Same principal still round-trips; SSE recording is scoped the same way.
    raw = _sse("msg_sse_scoped") + _sse_start("toolu_sse_scoped", "Grep")
    tool_name_registry.record_from_sse(S, raw, tool_name_registry.thread_pinned_of(a_body))
    assert tool_name_registry.lookup(S, "toolu_sse_scoped") == "Grep"
    assert tool_name_registry.lookup(SB, "toolu_sse_scoped") is None
    assert tool_name_registry.lookup_thread_pinned(SB, "msg_sse_scoped") is None
    assert tool_name_registry.lookup_thread_pinned(S, "msg_sse_scoped") is not None
    tool_name_registry.record_from_json(
        S, {"content": [{"type": "tool_use", "id": "tj", "name": "W"}]}
    )
    assert tool_name_registry.lookup(SB, "tj") is None


def test_router_names_learned_by_one_principal_do_not_reach_another() -> None:
    tool_name_registry.record(S, "toolu_scoped_name", "Grep")
    msgs = [_result_msg("toolu_scoped_name")]
    mine = ContentRouter()._build_tool_name_map(msgs, thread_continue=True, tool_scope=S)
    theirs = ContentRouter()._build_tool_name_map(msgs, thread_continue=True, tool_scope=SB)
    assert mine["toolu_scoped_name"] == "Grep"
    assert theirs["toolu_scoped_name"] == "Read"  # fail-safe fallback, not A's name


def _shape_system(monkeypatch) -> None:  # noqa: ANN001
    def _shaping(body, **kwargs):  # noqa: ANN001, ANN003, ANN202
        out = dict(body)
        out["system"] = [*body["system"], {"type": "text", "text": "steer"}]
        return out, True, 100, 10

    monkeypatch.setenv("HEADROOM_SYSTEM_COMPACT", "1")
    monkeypatch.setattr("headroom.proxy.system_compaction.compact_system_prompt", _shaping)


def test_colliding_message_id_does_not_pin_another_principals_blobs_non_streaming(
    monkeypatch,  # noqa: ANN001
) -> None:
    _shape_system(monkeypatch)
    _sticky(monkeypatch, inject=False)
    tools = _tools_needing_compaction()
    system = [dict(BILLING), {"type": "text", "text": "full  system"}]
    messages = [{"role": "user", "content": "x"}]
    create = _continue_body(messages, tools=copy.deepcopy(tools))
    create["system"] = copy.deepcopy(system)
    create["thread"] = {"type": "create"}
    forwarded_a = _run_turn(create, "msg_collide")
    assert forwarded_a["system"] != system

    def _continue(headers: dict) -> dict:
        cont = _continue_body(messages, tools=copy.deepcopy(tools))
        cont["system"] = copy.deepcopy(system)
        cont["thread"] = {"type": "continue", "previous_message_id": "msg_collide"}
        return _run_turn(cont, "msg_next", headers=headers)

    sent_a = _continue(HEADERS)
    assert json.dumps(sent_a["system"]) == json.dumps(forwarded_a["system"])
    assert json.dumps(sent_a["tools"]) == json.dumps(forwarded_a["tools"])
    sent_b = _continue(HEADERS_B)  # same id, other principal: A's blobs must not leak
    assert json.dumps(sent_b["system"]) != json.dumps(forwarded_a["system"])
    assert json.dumps(sent_b["tools"]) != json.dumps(forwarded_a["tools"])


def test_colliding_message_id_does_not_pin_another_principals_blobs_streaming(
    monkeypatch,  # noqa: ANN001
) -> None:
    _shape_system(monkeypatch)
    tools = _tools_needing_compaction()
    system = [dict(BILLING), {"type": "text", "text": "full  system"}]
    messages = [{"role": "user", "content": "x"}]
    upstream = _FakeUpstream(_sse("msg_stream_collide"))
    with _client() as client:
        proxy = client.app.state.proxy
        proxy.config.mode = "token"
        _instrument(proxy, {})
        del proxy._stream_response
        proxy.http_client = upstream

        def _send(thread: dict, headers: dict) -> None:
            body = _continue_body(messages, tools=copy.deepcopy(tools), stream=True)
            body["system"] = copy.deepcopy(system)
            body["thread"] = thread
            assert _post(client, body, headers).status_code == 200

        _send({"type": "create"}, HEADERS)
        cont = {"type": "continue", "previous_message_id": "msg_stream_collide"}
        _send(cont, HEADERS)
        _send(cont, HEADERS_B)
    created, continued_a, continued_b = upstream.bodies
    assert json.dumps(continued_a["system"]) == json.dumps(created["system"])
    assert json.dumps(continued_b["system"]) != json.dumps(created["system"])
    assert json.dumps(continued_b["tools"]) != json.dumps(created["tools"])


def test_same_principal_rerecord_releases_its_own_old_blobs(monkeypatch) -> None:  # noqa: ANN001
    for name in ("_names", "_thread_pinned", "_blobs", "_blob_refs"):
        monkeypatch.setattr(tool_name_registry, name, type(getattr(tool_name_registry, name))())
    for system in ("one", "two"):
        pinned = tool_name_registry.thread_pinned_of({"thread": CONTINUE, "system": system})
        tool_name_registry.record_thread_pinned(S, "msg_again", pinned)
    assert len(tool_name_registry._blobs) == 1
    assert tool_name_registry.lookup_thread_pinned(S, "msg_again")["system"] == "two"


def test_non_streaming_response_names_are_recorded_under_the_callers_scope(monkeypatch) -> None:  # noqa: ANN001
    _sticky(monkeypatch, inject=False)
    resp = _response("msg_names_json").json()
    resp["content"] = [{"type": "tool_use", "id": "toolu_scoped_json", "name": "Edit", "input": {}}]
    captured: dict = {}
    with _client() as client:
        proxy = client.app.state.proxy
        _instrument(proxy, captured)

        async def _retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001, ANN003
            return httpx.Response(200, json=resp)

        proxy._retry_request = _retry
        body = _continue_body([{"role": "user", "content": "x"}])
        body["thread"] = {"type": "create"}
        assert _post(client, body).status_code == 200
    assert tool_name_registry.lookup(S, "toolu_scoped_json") == "Edit"
    assert tool_name_registry.lookup(SB, "toolu_scoped_json") is None


def test_tenant_header_partitions_otherwise_identical_callers(monkeypatch) -> None:  # noqa: ANN001
    _shape_system(monkeypatch)
    _sticky(monkeypatch, inject=False)
    tools = _tools_needing_compaction()
    system = [dict(BILLING), {"type": "text", "text": "full  system"}]
    messages = [{"role": "user", "content": "x"}]
    tenant_a = {**HEADERS, "X-Headroom-Tenant-ID": "tenant-a"}
    tenant_b = {**HEADERS, "X-Headroom-Tenant-ID": "tenant-b"}
    create = _continue_body(messages, tools=copy.deepcopy(tools))
    create["system"] = copy.deepcopy(system)
    create["thread"] = {"type": "create"}
    forwarded = _run_turn(create, "msg_tenant", headers=tenant_a)

    def _continue(headers: dict) -> dict:
        cont = _continue_body(messages, tools=copy.deepcopy(tools))
        cont["system"] = copy.deepcopy(system)
        cont["thread"] = {"type": "continue", "previous_message_id": "msg_tenant"}
        return _run_turn(cont, "msg_tenant_next", headers=headers)

    assert json.dumps(_continue(tenant_a)["system"]) == json.dumps(forwarded["system"])
    assert json.dumps(_continue(tenant_b)["system"]) != json.dumps(forwarded["system"])


def test_pipeline_receives_the_callers_scope_for_tool_name_lookups() -> None:
    body = _continue_body([_result_msg("toolu_x")])
    seen = {}
    for who, headers in (("a", HEADERS), ("b", HEADERS_B)):
        captured: dict = {}
        with _client() as client:
            _instrument(client.app.state.proxy, captured)
            assert _post(client, copy.deepcopy(body), headers).status_code == 200
        seen[who] = captured["tool_scope"]
    assert seen == {"a": S, "b": SB}


def test_router_apply_forwards_tool_scope_to_the_name_map(monkeypatch) -> None:  # noqa: ANN001
    seen: list = []

    class _Stop(Exception):
        pass

    def _spy(self, messages, thread_continue=False, tool_scope=""):  # noqa: ANN001, ANN202
        seen.append(tool_scope)
        raise _Stop

    monkeypatch.setattr(ContentRouter, "_build_tool_name_map", _spy)
    with pytest.raises(_Stop):
        ContentRouter().apply(
            [_result_msg("toolu_spy")],
            SimpleNamespace(count_text=lambda t: len(t) // 4),
            thread_continue=True,
            tool_scope=S,
        )
    assert seen == [S]
