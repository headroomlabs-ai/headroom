"""A minimal re-driving turn hook with the SAME loop shape as headroom-tool-search.

Shrink (``on_request``): every tool whose name starts with ``defer_prefix``
(default ``"deferred_"``) is removed from ``ctx.tools`` and remembered in a
catalog; a single ``search_tools`` function is appended whose description lists
the deferred names. The output is a pure function of the input tools, so a
gateway session re-sending the same tools gets byte-identical shrunk tools every
turn.

Reload (``on_response``): if the model called ``search_tools`` - OpenAI chat
shape (``choices[0].message.tool_calls[]``) or Anthropic shape
(``content[]`` ``tool_use`` block) - resolve the query against the catalog
(substring match on name/description; no match -> every deferred tool), restore
the resolved definitions into ``ctx.tools`` (appended after the search tool,
sorted by name, exactly like the tool-search adapter's I2/I3 rules), append the
assistant call plus a tool result naming the loaded tools to ``ctx.messages``,
``await call_model(messages)`` and loop, up to ``max_rounds``. Returns the final
provider response, or ``None`` when it never drove the model.

``stream_safe = False``: the reload needs a re-drive, so the proxy (and the
gateway request half) must skip this hook whenever nothing can re-drive.

Entry points:

* ``register(**kwargs) -> RedriveHook`` - build + ``register_turn_hook`` (tests).
* ``install(app, config)`` - the ``headroom.proxy_extension`` entry-point shape.
  The OSS loader discovers extensions through ``importlib.metadata`` entry points
  only, so a subprocess driver (the Kong docker test) that wants this hook in a
  uvicorn process must either register a temporary entry point or start the app
  from Python after calling ``register()``. Env knobs honoured by ``install``:
  ``HEADROOM_TEST_REDRIVE_PREFIX`` and ``HEADROOM_TEST_REDRIVE_MAX_ROUNDS``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from headroom.proxy.turn_hooks import register_turn_hook

SEARCH_TOOL_NAME = "search_tools"
SEARCH_TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "What capability you need, in a few words."}
    },
    "required": ["query"],
}
LOADED_PREFIX = "Loaded tools: "


def _name(tool: Any) -> str:
    if not isinstance(tool, dict):
        return ""
    fn = tool.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn["name"])
    return str(tool.get("name") or "")


def _description(tool: Any) -> str:
    if not isinstance(tool, dict):
        return ""
    fn = tool.get("function")
    if isinstance(fn, dict):
        return str(fn.get("description") or "")
    return str(tool.get("description") or "")


def _is_openai_shape(tools: list[Any]) -> bool:
    """Chat-completions tools nest the function; Anthropic tools are flat with
    ``input_schema``. Decide from the tools, not the provider string, so an
    Anthropic-shaped body sent with an OpenAI-looking model still round-trips."""
    for t in tools:
        if isinstance(t, dict):
            if isinstance(t.get("function"), dict):
                return True
            if "input_schema" in t:
                return False
    return True


def search_tool(openai_shape: bool, deferred_names: list[str]) -> dict[str, Any]:
    description = (
        "Search for additional tools by capability. Hidden tools: "
        + ", ".join(sorted(deferred_names))
        + ". Call this with a short description of what you need and the matching "
        "tools will be loaded so you can call them next."
    )
    if openai_shape:
        return {
            "type": "function",
            "function": {
                "name": SEARCH_TOOL_NAME,
                "description": description,
                "parameters": SEARCH_TOOL_PARAMETERS,
            },
        }
    return {
        "name": SEARCH_TOOL_NAME,
        "description": description,
        "input_schema": SEARCH_TOOL_PARAMETERS,
    }


@dataclass
class SearchCall:
    query: str
    call_id: str
    raw: dict[str, Any]
    shape: str  # "openai" | "anthropic"


def detect_search_call(response: Any) -> SearchCall | None:
    """Find a ``search_tools`` call in an OpenAI chat or Anthropic response."""
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        msg = (choices[0] or {}).get("message") or {}
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if fn.get("name") == SEARCH_TOOL_NAME:
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args or "{}")
                    except ValueError:
                        args = {"query": args}
                query = str((args or {}).get("query") or "") if isinstance(args, dict) else ""
                return SearchCall(
                    query=query, call_id=str(tc.get("id") or ""), raw=tc, shape="openai"
                )
    content = response.get("content")
    if isinstance(content, list):
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("name") == SEARCH_TOOL_NAME
            ):
                inp = block.get("input") or {}
                query = str(inp.get("query") or "") if isinstance(inp, dict) else ""
                return SearchCall(
                    query=query, call_id=str(block.get("id") or ""), raw=block, shape="anthropic"
                )
    return None


@dataclass
class RedriveHook:
    name: str = "test_redrive"
    savings_source: str = "test_redrive"
    stream_safe: bool = False
    defer_prefix: str = "deferred_"
    max_rounds: int = 4
    # Observability for tests.
    catalog: dict[str, dict[str, Any]] = field(default_factory=dict)
    shrink_calls: int = 0
    response_calls: int = 0
    rounds_driven: int = 0
    queries: list[str] = field(default_factory=list)

    # --- request: shrink ---------------------------------------------------

    def on_request(self, ctx: Any) -> None:
        tools = ctx.tools
        if not isinstance(tools, list) or not tools:
            return
        deferred = [t for t in tools if _name(t).startswith(self.defer_prefix)]
        if not deferred:
            return
        self.shrink_calls += 1
        for t in deferred:
            self.catalog[_name(t)] = t
        kept = [t for t in tools if not _name(t).startswith(self.defer_prefix)]
        kept = [t for t in kept if _name(t) != SEARCH_TOOL_NAME]
        # Rebind, never mutate in place: ctx.tools is the caller's list.
        ctx.tools = [*kept, search_tool(_is_openai_shape(tools), [_name(t) for t in deferred])]

    # --- response: reload loop ---------------------------------------------

    def _resolve(self, query: str) -> list[dict[str, Any]]:
        q = query.lower().strip()
        hits = [
            d
            for n, d in self.catalog.items()
            if q and (q in n.lower() or any(w in _description(d).lower() for w in q.split()))
        ]
        return hits or list(self.catalog.values())

    def _restore(self, ctx: Any, resolved: list[dict[str, Any]]) -> None:
        tools = ctx.tools if isinstance(ctx.tools, list) else []
        head: list[Any] = []
        tail: list[Any] = []
        seen_search = False
        for t in tools:
            (tail if seen_search else head).append(t)
            if _name(t) == SEARCH_TOOL_NAME:
                seen_search = True
        head_names = {_name(t) for t in head}
        block = {_name(t): t for t in tail}
        for d in resolved:
            if _name(d) not in head_names:
                block.setdefault(_name(d), d)
        rebuilt = [*head, *(block[n] for n in sorted(block))]
        if rebuilt != tools:
            ctx.tools = rebuilt

    @staticmethod
    def _reload_items(
        response: dict[str, Any], call: SearchCall, loaded: str
    ) -> list[dict[str, Any]]:
        text = f"{LOADED_PREFIX}{loaded}. They are now available - call the one you need."
        if call.shape == "openai":
            assistant = (response.get("choices") or [{}])[0].get("message") or {
                "role": "assistant",
                "content": None,
                "tool_calls": [call.raw],
            }
            return [assistant, {"role": "tool", "tool_call_id": call.call_id, "content": text}]
        return [
            {"role": "assistant", "content": list(response.get("content") or [call.raw])},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": call.call_id, "content": text}],
            },
        ]

    async def on_response(self, ctx: Any, response: dict[str, Any], call_model: Any) -> Any:
        self.response_calls += 1
        current = response
        drove = False
        for _ in range(self.max_rounds):
            call = detect_search_call(current)
            if call is None:
                return current if drove else None
            self.queries.append(call.query)
            resolved = self._resolve(call.query)
            self._restore(ctx, resolved)
            loaded = ", ".join(sorted(_name(d) for d in resolved)) or "(no match)"
            messages = [*ctx.messages, *self._reload_items(current, call, loaded)]
            ctx.messages = messages
            current = await call_model(messages)
            self.rounds_driven += 1
            drove = True
        return current  # hit the round cap while the model was still searching


def register(**kwargs: Any) -> RedriveHook:
    """Build a hook, register it with the proxy's turn-hook registry, return it."""
    hook = RedriveHook(**kwargs)
    register_turn_hook(hook)
    return hook


def install(app: Any, config: Any) -> None:
    """``headroom.proxy_extension`` entry-point shape."""
    prefix = os.environ.get("HEADROOM_TEST_REDRIVE_PREFIX", "deferred_")
    try:
        rounds = int(os.environ.get("HEADROOM_TEST_REDRIVE_MAX_ROUNDS", "4"))
    except ValueError:
        rounds = 4
    register(defer_prefix=prefix, max_rounds=max(1, rounds))
