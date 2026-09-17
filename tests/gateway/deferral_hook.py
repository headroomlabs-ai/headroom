"""Reference native tool-search deferral hook for the gateway contract tests.

OSS core no longer defers tool schemas on the gateway path; the licensed
tool-search extension's native tier does, through the ordinary turn-hook
contract. This module is the smallest hook that exercises every seam that tier
relies on, so the contract tests prove core carries a hook's deferral end to
end without depending on the extension package:

* the OSS primitives (``inject_tool_search_deferral`` and the two gate helpers
  in ``headroom.proxy.helpers``) applied as a pure function of ``ctx.tools``;
* the savings tags core's ``/stats`` headline reads
  (``tool_search_deferred_tools`` / ``tool_search_deferred_tokens``) and the
  ``tool_search`` attribution entry;
* ``ctx.provider_headers`` for the ``anthropic-beta`` token, merged by core
  behind the client's own value and surfaced as the response's ``headers``;
* ``stream_safe = True`` so it runs when the gateway cannot re-drive (every
  streamed Claude Code turn), and ``priority = 200`` so a router that changes
  ``ctx.model`` has run before the first-party gate is evaluated.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from headroom.proxy.helpers import (
    _TOOL_SEARCH_CORE_TOOLS,
    anthropic_model_is_first_party,
    inject_tool_search_deferral,
    tools_are_anthropic_shaped,
)
from headroom.proxy.turn_hooks import register_turn_hook

BETA_TOKEN = "advanced-tool-use-2025-11-20"
_ON = ("1", "true", "yes", "on", "auto")


@dataclass
class DeferralHook:
    name: str = "test_native_deferral"
    savings_source: str = "tool_search"
    stream_safe: bool = True
    priority: int = 200
    # Extra provider headers to request (tests the allowlist).
    extra_headers: dict[str, str] | None = None
    request_calls: int = 0
    applied: int = 0

    def on_request(self, ctx: Any) -> None:
        self.request_calls += 1
        if ctx.provider != "anthropic":
            return
        if os.environ.get("HEADROOM_TOOL_SEARCH", "1").strip().lower() not in _ON:
            return
        if not anthropic_model_is_first_party(ctx.model):
            return
        if not tools_are_anthropic_shaped(ctx.tools):
            return
        after = inject_tool_search_deferral(ctx.tools, core_tools=_TOOL_SEARCH_CORE_TOOLS)
        if after is ctx.tools:
            return
        ctx.tools = after
        deferred = [t for t in after if isinstance(t, dict) and t.get("defer_loading")]
        saved = 0
        if ctx.count_tools is not None:
            try:
                saved = int(ctx.count_tools(deferred))
            except Exception:
                saved = 0
        ctx.tags["tool_search_deferred_tools"] = len(deferred)
        ctx.tags["tool_search_deferred_tokens"] = saved
        ctx.record_savings("tool_search", tokens=saved)
        ctx.provider_headers["anthropic-beta"] = BETA_TOKEN
        for key, value in (self.extra_headers or {}).items():
            ctx.provider_headers[key] = value
        self.applied += 1


@dataclass
class BufferedDeferralHook(DeferralHook):
    """Same deferral, but declared NOT stream-safe: the gateway request half
    must skip it whenever it cannot re-drive."""

    name: str = "test_native_deferral_buffered"
    stream_safe: bool = False


@dataclass
class ModelRouterHook:
    """A router that rewrites ``ctx.model``; ``priority`` decides whether the
    deferral hook sees the routed id or the original."""

    target_model: str
    name: str = "test_model_router"
    stream_safe: bool = True
    priority: int = 10

    def on_request(self, ctx: Any) -> None:
        ctx.model = self.target_model


def register(hook: Any | None = None, **kwargs: Any) -> Any:
    hook = hook if hook is not None else DeferralHook(**kwargs)
    register_turn_hook(hook)
    return hook


def install(app: Any, config: Any) -> None:
    """``headroom.proxy_extension`` entry-point shape."""
    register()
