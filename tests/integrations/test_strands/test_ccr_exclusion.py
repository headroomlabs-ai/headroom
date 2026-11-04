"""Regression tests for qualified CCR names in Strands compression."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("headroom._core")

from headroom.integrations.strands import hooks


@pytest.fixture
def hook(monkeypatch: pytest.MonkeyPatch) -> hooks.HeadroomHookProvider:
    monkeypatch.setattr(hooks, "STRANDS_AVAILABLE", True)
    provider = hooks.HeadroomHookProvider(min_tokens_to_compress=0)
    provider._crusher = Mock()
    provider._crusher.crush.return_value = SimpleNamespace(
        compressed="compressed", was_modified=True
    )
    return provider


class _Registry:
    """Stand-in for Strands' HookRegistry that dispatches like the real one."""

    def __init__(self) -> None:
        self._callbacks: dict[object, list] = {}

    def add_callback(self, event_type: object, callback) -> None:  # noqa: ANN001
        self._callbacks.setdefault(event_type, []).append(callback)

    def dispatch(self, event_type: object, event: object) -> None:
        for callback in self._callbacks[event_type]:
            callback(event)


def _event(tool_name: str, content: str) -> SimpleNamespace:
    return SimpleNamespace(
        tool_use={"name": tool_name, "toolUseId": "tool_1"},
        result={"content": [{"text": content}]},
    )


def test_qualified_ccr_result_is_preserved_through_the_registered_hook(
    hook: hooks.HeadroomHookProvider,
) -> None:
    """Drive the seam Strands actually drives: register_hooks, then dispatch."""
    registry = _Registry()
    hook.register_hooks(registry)

    content = "x" * 400
    event = _event("mcp__headroom__headroom_retrieve", content)
    registry.dispatch(hooks.AfterToolCallEvent, event)

    assert event.result["content"][0]["text"] == content
    assert hook.metrics_history[0].skip_reason == "tool_excluded"
    hook._crusher.crush.assert_not_called()


@pytest.mark.parametrize(
    "tool_name",
    [
        "mcp__headroom__headroom_retrieve",
        "mcp_headroom_headroom_retrieve",
        "headroom_retrieve",
    ],
)
def test_qualified_ccr_tool_result_is_preserved(
    hook: hooks.HeadroomHookProvider, tool_name: str
) -> None:
    content = "x" * 400
    event = _event(tool_name, content)

    hook._compress_tool_result(event)

    assert event.result["content"][0]["text"] == content
    assert hook.metrics_history[0].skip_reason == "tool_excluded"
    hook._crusher.crush.assert_not_called()


def test_near_match_tool_name_still_compresses(hook: hooks.HeadroomHookProvider) -> None:
    event = _event("mcp__headroom__headroom_retrieve_extra", "x" * 400)

    hook._compress_tool_result(event)

    assert event.result["content"][0]["text"] == "compressed"
    assert hook.metrics_history[0].was_compressed is True
    hook._crusher.crush.assert_called_once()
