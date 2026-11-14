"""Tool-schema savings attribution policy for proxy stats.

Headroom saves input tokens in two accounting shapes, and the split is not a
style choice — it follows from what each transform can observe:

* **Compaction** (``*:tool_schema_compaction``, ``*:tool_desc_compaction``)
  rewrites the tool array in place, so both endpoints are countable. Handlers
  fold the delta into ``original_tokens``/``optimized_tokens``, which keeps
  ``tok_before - tok_after == tok_saved`` coherent in the PERF line. It is
  therefore ALREADY inside ``tokens_saved`` and must never be added again.
* **Deferral / hook shrink** (tool search, turn hooks) removes schemas that
  ``count_messages`` never saw, so it cannot move ``original_tokens``. It is
  recorded in per-request tags and is ADDITIVE to ``tokens_saved``.

The one rule a caller needs: the headline is
``tokens_saved + tool_schema_saved_from_tags(tags)``. Use
:func:`headline_tokens_saved` rather than open-coding it — three surfaces had
drifted inline copies of that sum, and two harnesses were silently dropping
their compaction savings entirely because the convention was never written down.

Adding a new tool-schema-shrinking feature? If it moves the tool array, fold it
in the handler like the compaction sites do. If it defers schemas, add its tag
name to :data:`TOOL_SCHEMA_SAVINGS_TAGS` and every surface picks it up.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from headroom.log_safety import WarnOnce, describe_exception

logger = logging.getLogger(__name__)

# A failure that repeats on every request warns once per exception type; every
# occurrence also goes to debug.
_reconcile_warned = WarnOnce(32, "tool-schema reconcile failure types")


def _log_reconcile_failure(exc: Exception, booked: int) -> None:
    # The exception comes from tokenizing client tool schemas and can quote
    # them, so it is described by type and code location only, at every level.
    where = describe_exception(exc)
    if _reconcile_warned.first(type(exc).__qualname__, logger):
        logger.warning(
            "tool-schema deferral reconcile failed (%s); credit left at %d booked tokens, "
            "so /stats over-reports tool-search savings for this request. Please report "
            "this as a bug.",
            where,
            booked,
        )
    logger.debug("tool-schema deferral reconcile failed; booked=%d: %s", booked, where)


def without_deferral_flags(tools: object) -> object:
    """The tool array as it is MEASURED for a hook's tool delta.

    Deferral is reported through ``tool_search_deferred_tokens`` (it removes
    schemas from the billed context while leaving them in the array, flagged
    ``defer_loading``). Measuring the raw array would see only the flag
    appear or disappear: un-deferring a tool would shorten the JSON by the
    flag text and be booked as a small saving although the bill goes UP, and
    deferring one would add a phantom cost. So the flag is ignored here, and a
    hook's measured tool delta covers only real removals and rewrites. Tools
    nested in a Responses ``namespace`` group (Codex) are measured the same
    way. Never mutates its input.
    """
    return _measured_view(tools, frozenset())


def unbilled_deferred_tool_names(tools: object, messages: object) -> frozenset[str]:
    """Names of the ``defer_loading`` tools in ``tools`` that are not billed.

    A deferred tool costs nothing until a ``tool_reference`` naming it is in
    the history; from then on its full definition is billed (measured on
    Copilot's Claude API). ``messages`` is the request's Anthropic message
    list; pass ``None`` where loading cannot be read from the history (OpenAI
    tool search), and every tool is then treated as billed, the old behaviour.
    """
    if messages is None or not isinstance(tools, list):
        return frozenset()
    deferred = _deferred_names(tools)
    if not deferred:
        return frozenset()
    from headroom.proxy.helpers import (
        referenced_tool_names,
        strip_unsupported_tool_search_blocks,
    )

    # What the provider receives: repair drops a reference this request's tools
    # cannot support, and that one loads nothing.
    repaired, _ = strip_unsupported_tool_search_blocks(messages, tools)
    return frozenset(deferred - referenced_tool_names(repaired))


def _tool_names(tools: object) -> set[str]:
    from headroom.proxy.helpers import iter_tool_entries

    return {str(t["name"]) for t in iter_tool_entries(tools) if t.get("name")}


def _deferred_names(tools: object) -> frozenset[str]:
    from headroom.proxy.helpers import iter_tool_entries

    return frozenset(
        str(t["name"]) for t in iter_tool_entries(tools) if t.get("defer_loading") and t.get("name")
    )


def _measured_view(tools: object, excluded: frozenset[str]) -> object:
    """``tools`` without ``defer_loading`` flags or the ``excluded`` tools, the
    members of a ``namespace`` group included. Never mutates its input."""
    if not isinstance(tools, list):
        return tools
    view: list[Any] = []
    for tool in tools:
        if isinstance(tool, dict):
            if tool.get("name") and str(tool["name"]) in excluded:
                continue
            if "defer_loading" in tool:
                tool = {k: v for k, v in tool.items() if k != "defer_loading"}
            if tool.get("type") == "namespace" and isinstance(tool.get("tools"), list):
                tool = {**tool, "tools": _measured_view(tool["tools"], excluded)}
        view.append(tool)
    return view


@dataclass(frozen=True)
class ToolsBeforeHook:
    """What :func:`hook_tool_counts` needs from the tools before a hook ran."""

    names: frozenset[str]
    unbilled: frozenset[str]
    # Tools whose schema is growth once the hook puts them on the bill: those
    # not billed before it, less those the deferral credit covers.
    chargeable: frozenset[str] = frozenset()
    # A copy of the tools, kept only when some tool was unbilled or chargeable:
    # whether it is left out of the count depends on what the hook does to it.
    tools: object = None
    # Counted up front otherwise (nothing to decide later).
    count: int | None = None


def tools_before_hook(
    tools: object,
    messages: object,
    count_tools: Callable[[Any], int],
    tags: object = None,
    *,
    deferred_billed: bool = False,
) -> ToolsBeforeHook:
    """Snapshot ``tools`` for a hook's measured tool delta; see :func:`hook_tool_counts`.

    ``messages`` as for :func:`unbilled_deferred_tool_names`; ``tags`` are the
    request's tags, read for the deferral credit. ``deferred_billed``: a
    translating backend bills every tool, deferred or not. May raise whatever
    ``count_tools`` raises.
    """
    unbilled = unbilled_deferred_tool_names(tools, messages)
    names = frozenset(_tool_names(tools))
    if deferred_billed:
        chargeable: frozenset[str] = frozenset()
    else:
        # Where loading is unreadable (``messages`` is None, OpenAI tool
        # search) a deferred tool the hook un-defers may have been unloaded:
        # it is charged as if it were, which under-books at worst. (Condensing
        # or removing one there is still booked as if billed, as before.)
        # With a deferral credit, every deferred tool is left to
        # reconcile_deferred_tokens: the proxy's deferrers stand down when the
        # client already defers, so the credit's tools are all of them.
        chargeable = unbilled if messages is not None else _deferred_names(tools)
        if isinstance(tags, dict) and "tool_search_deferred_tokens" in tags:
            chargeable -= _deferred_names(tools)
    if not unbilled and not chargeable:
        return ToolsBeforeHook(
            names, unbilled, count=int(count_tools(without_deferral_flags(tools)))
        )
    return ToolsBeforeHook(names, unbilled, chargeable, tools=copy.deepcopy(tools))


def hook_tool_counts(
    before: ToolsBeforeHook,
    tools: object,
    messages: object,
    count_tools: Callable[[Any], int],
) -> tuple[int, int]:
    """Return the ``(before, after)`` tool counts a hook's delta is taken from.

    Measured as :func:`without_deferral_flags` does, and additionally without
    every tool that is billed on neither side: deferred and not loaded before
    the hook (or absent), and deferred and not loaded after it (or absent).
    Condensing, adding or removing such a tool changes nothing on the bill, so
    it books nothing. A chargeable tool (:class:`ToolsBeforeHook`) billed
    after the hook, because it un-deferred or loaded it, is left out of the
    ``before`` count only: its whole schema is growth, netted against the
    hook's other savings. Any other tool the hook defers or un-defers stays in
    both counts: the flag itself is never booked, and the deferral credit
    (``tool_search_deferred_tokens``, reconciled by
    :func:`reconcile_deferred_tokens`) carries that change. May raise
    whatever ``count_tools`` raises.
    """
    unbilled_after = unbilled_deferred_tool_names(tools, messages)
    billed_before = before.names - before.unbilled
    billed_after = _tool_names(tools) - unbilled_after
    excluded = frozenset((before.unbilled | unbilled_after) - billed_before - billed_after)
    if messages is None:
        # Unreadable loading: only un-deferring is seen to bill a tool.
        billed_after -= _deferred_names(tools)
    newly_billed = before.chargeable & billed_after
    if before.count is not None:
        before_count = before.count
    else:
        before_count = int(count_tools(_measured_view(before.tools, excluded | newly_billed)))
    return before_count, int(count_tools(_measured_view(tools, excluded)))


def billed_tools(tools: object, history: object) -> object:
    """The tools that enter the billed context, for compaction counts.

    ``history`` is :func:`loading_history`'s result. With one, a deferred tool
    counts once a ``tool_reference`` has loaded it; without one (loading is
    unreadable), every ``defer_loading`` tool is left out, since a client's
    own deferral keeps it off the bill until the model asks for it.
    """
    if not isinstance(tools, list):
        return tools
    if history is None:
        return [t for t in tools if not (isinstance(t, dict) and t.get("defer_loading"))]
    unbilled = unbilled_deferred_tool_names(tools, history)
    if not unbilled:
        return tools
    return [
        t
        for t in tools
        if not (isinstance(t, dict) and t.get("name") and str(t["name"]) in unbilled)
    ]


def loading_history(provider: str, messages: object) -> object:
    """The history that says which deferred tools are loaded, or ``None``.

    Only an Anthropic history can be read (``tool_reference`` blocks); OpenAI
    tool search loads tools outside it.
    """
    return messages if provider == "anthropic" else None


def proxy_added_tools(tools: object, client_tools: object, name: str) -> list[Any]:
    """The tool named ``name`` in the outbound ``tools`` when the client did not
    declare it: a definition the proxy added and the request pays for."""
    if name in _tool_names(client_tools):
        return []
    return [
        t
        for t in (tools if isinstance(tools, list) else [])
        if isinstance(t, dict) and t.get("name") == name
    ]


DeferredBooking = tuple[int, dict[str, Any], "dict[str, Any] | None", frozenset[str]]


def deferred_booking(tags: object, tools: object, messages: object) -> DeferredBooking | None:
    """Snapshot the deferral currently booked: the tag, the tools it covers,
    its own ledger entry and which of those tools no ``tool_reference`` in
    ``messages`` has loaded yet (as for :func:`unbilled_deferred_tool_names`).

    ``None`` when nothing is booked. The schemas are copied as they were when
    booked, so a later in-place edit cannot change what the credit stands for.
    The ledger entry is the ``tool_search`` row marked ``estimated``: both the
    core deferral and the native deferral hook book that way (rows coalesce, so
    they share it), and a hook's own measured saving never is, so it is never
    the one debited.
    """
    if not isinstance(tags, dict) or "tool_search_deferred_tokens" not in tags:
        return None
    try:
        booked = int(tags.get("tool_search_deferred_tokens") or 0)
    except (TypeError, ValueError):
        return None
    snapshot = {
        str(t["name"]): copy.deepcopy(t)
        for t in (tools if isinstance(tools, list) else [])
        if isinstance(t, dict) and t.get("defer_loading") and t.get("name")
    }
    return booked, snapshot, _deferral_entry(tags), unbilled_deferred_tool_names(tools, messages)


def _deferral_entry(tags: dict[str, Any]) -> dict[str, Any] | None:
    """The ledger's estimated ``tool_search`` row, if there is one."""
    from headroom.proxy.savings_attribution import SAVINGS_ATTRIBUTION_TAG

    ledger = tags.get(SAVINGS_ATTRIBUTION_TAG)
    return next(
        (
            item
            for item in reversed(ledger if isinstance(ledger, list) else [])
            if isinstance(item, dict)
            and item.get("source") == "tool_search"
            and item.get("estimated")
        ),
        None,
    )


def reconcile_deferred_tokens(
    tags: object,
    booking: DeferredBooking | None,
    tools: object,
    count_tools: object,
    messages: object,
) -> None:
    """Cap the deferral credit at the booked tools still deferred after hooks.

    A hook may un-defer tools the deferral booked (tool search's hot tools);
    those are billed, so their share of the credit goes. Each tool still
    deferred is credited at the SMALLER of its booked and its final schema:
    growing another schema cannot keep an un-deferred tool's credit alive, and
    a schema a hook shortened (and booked as its own saving) is not credited
    twice. A tool unloaded both when booked and in the final ``messages`` is
    the exception: billed at neither point, :func:`hook_tool_counts` books no
    hook saving for it, so it keeps its booked schema. ``messages`` as for
    :func:`unbilled_deferred_tool_names`. Only ever lowers the tag, and lowers
    the deferral's own ledger entry by the same amount so ``/stats`` by-source
    agrees with the headline. Never raises.
    """
    if booking is None or not isinstance(tags, dict) or not callable(count_tools):
        return
    booked, snapshot, entry, unbilled_when_booked = booking
    try:
        final = {
            str(t["name"]): t
            for t in (tools if isinstance(tools, list) else [])
            if isinstance(t, dict) and t.get("defer_loading") and t.get("name")
        }
        unbilled = unbilled_deferred_tool_names(tools, messages)
        kept = []
        for name, schema in snapshot.items():
            if name not in final:
                continue
            now = final[name]
            if name in unbilled and name in unbilled_when_booked:
                kept.append(schema)
                continue
            kept.append(now if int(count_tools([now])) < int(count_tools([schema])) else schema)
        measured = int(count_tools(kept)) if kept else 0
        released = booked - measured
        if released <= 0:
            return
        tags["tool_search_deferred_tokens"] = measured
        tags["tool_search_deferred_tools"] = len(kept)
        if isinstance(entry, dict):
            entry["tokens"] = max(0, int(entry.get("tokens") or 0) - released)
    except Exception as exc:  # accounting must never break a request
        _log_reconcile_failure(exc, booked)
        return


# System-prompt text a turn hook removed: like a tool schema it sits in the
# cached prefix, so it is priced as these tags are, not as live-zone content.
TURN_HOOK_SYSTEM_SAVED_TAG = "turn_hook_system_saved_tokens"
TOOL_SCHEMA_SAVINGS_TAGS: tuple[str, ...] = (
    "tool_search_deferred_tokens",
    "turn_hook_tools_saved_tokens",
    TURN_HOOK_SYSTEM_SAVED_TAG,
)


def tool_schema_saved_from_tags(tags: object) -> int:
    """Return prefix tokens (tool definitions, a hook's system-prompt cut) Headroom
    kept out of context for one request.

    The summed tags are set only on paths where Headroom performed the deferral,
    so clients that already had tool search enabled contribute zero here.

    These tags are additive to ``tokens_saved`` — see the module docstring.
    """
    if not isinstance(tags, dict):
        return 0

    total = 0
    for key in TOOL_SCHEMA_SAVINGS_TAGS:
        try:
            total += int(tags.get(key, 0) or 0)
        except (TypeError, ValueError):
            continue
    return total


def net_hook_tool_saving(raw_saved: int, tags: object, hook_tool_saved: int) -> int:
    """Net a request's message-side saving against the tool saving its turn
    hooks booked on ``turn_hook_tools_saved_tokens``; return it clamped at 0.

    The headline is ``tokens_saved`` (clamped) plus that tag. A hook that adds
    message text while removing tools leaves ``raw_saved`` negative, and the
    clamp alone would drop the growth while the tag kept the full removal. The
    deficit is debited from the tag first (at most what the hooks booked on
    this request), then from the system-prompt cut they booked, then from the
    tool-search credit and its ledger row, so headline == messages + tools + system,
    signed, clamped once.
    """
    raw = int(raw_saved)
    booked = max(0, int(hook_tool_saved or 0))
    if raw < 0 and booked > 0 and isinstance(tags, dict):
        offset = min(-raw, booked)
        current = int(tags.get("turn_hook_tools_saved_tokens", 0) or 0)
        tags["turn_hook_tools_saved_tokens"] = max(0, current - offset)
        raw += offset
    if raw < 0 and isinstance(tags, dict):
        system_cut = max(0, int(tags.get(TURN_HOOK_SYSTEM_SAVED_TAG, 0) or 0))
        offset = min(-raw, system_cut)
        if offset:
            tags[TURN_HOOK_SYSTEM_SAVED_TAG] = system_cut - offset
            raw += offset
    if raw < 0 and isinstance(tags, dict):
        try:
            credit = max(0, int(tags.get("tool_search_deferred_tokens") or 0))
        except (TypeError, ValueError):
            credit = 0
        offset = min(-raw, credit)
        if offset:
            tags["tool_search_deferred_tokens"] = credit - offset
            entry = _deferral_entry(tags)
            if entry is not None:
                entry["tokens"] = max(0, int(entry.get("tokens") or 0) - offset)
            raw += offset
    return max(0, raw)


def accumulate_tool_schema_saved(into: dict[str, int], tags: object) -> None:
    """Add one request's ``TOOL_SCHEMA_SAVINGS_TAGS`` into ``into``, for a
    transport that books several requests' savings on one later outcome."""
    if not isinstance(tags, dict):
        return
    for key in TOOL_SCHEMA_SAVINGS_TAGS:
        try:
            amount = int(tags.get(key, 0) or 0)
        except (TypeError, ValueError):
            continue
        if amount > 0:
            into[key] = into.get(key, 0) + amount


def headline_tokens_saved(tokens_saved: object, tags: object) -> int:
    """Return the single "Tokens saved" figure for one request.

    This is the only correct total to show a user: message compression plus the
    tool-definition tokens that never entered the context. Every reporting
    surface (PERF line, ``headroom perf``, ``/api/stats``, dashboard, session
    summary) must route through here so they cannot disagree.

    Clamped at zero: handlers already revert any inflation before forwarding, so
    a negative is a token-count artifact that never reached the model.
    """
    base = 0
    if isinstance(tokens_saved, (int, float, str)) and not isinstance(tokens_saved, bool):
        try:
            base = int(tokens_saved)
        except (TypeError, ValueError):
            base = 0
    return max(0, base + tool_schema_saved_from_tags(tags))
