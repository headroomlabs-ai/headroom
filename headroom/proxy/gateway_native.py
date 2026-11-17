"""Gemini ``generateContent`` and Bedrock Converse bodies on the gateway turn.

Same approach as :mod:`headroom.proxy.gateway_responses`, and for the same
reason: a *view*, never a translation. Both shapes carry parts the provider
needs back byte-for-byte (Gemini ``thoughtSignature``, ``inlineData``,
``functionCall``; Converse ``cachePoint``, ``reasoningContent``, ``toolUse``),
so the body is never rebuilt from chat messages.

* Every compressible text slot becomes one chat-shaped pseudo message, in
  document order. A slot is a path into the body; putting text back walks the
  path and replaces one leaf, so nothing else can change.
* A tool call (``functionCall``, ``toolUse``) becomes an assistant message with
  ``tool_calls`` and no slot. Nothing is written back from it; it is there so
  tool-name rules (``Read`` stays verbatim, read lifecycle) see which tool
  produced the result that follows.
* A JSON slot (a Gemini ``functionResponse.response`` with no string field, a
  Converse ``{"json": ...}`` block) is written back only when the new text
  still parses to the same JSON type.
* If the pipeline hands back a list that does not line up with the slots, the
  body is forwarded unchanged.

Converse bodies carry ``messages`` too, so the original list is kept under the
view marker while the view sits in ``messages``. The marker is a Headroom
control key and never reaches a provider.

Only a gateway turn (a body with a ``gateway`` block) gets a view; a legacy
caller's body goes through unchanged. Message counts the caller states
(``frozen_message_count``, ``protect_recent``) are in native messages, so
:func:`view_count`, :func:`view_recent` and :func:`native_count` convert
between native and view indices.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

from headroom.proxy.gateway_responses import message_text

GEMINI = "gemini"
CONVERSE = "converse"

#: Control key holding ``{"shape": ..., "messages": <original Converse list>}``.
VIEW_MARKER = "_headroom_native_view"

_THOUGHT_KEYS = ("thought", "thoughtSignature", "thought_signature")
_CONVERSE_BLOCK_KEYS = frozenset(
    {
        "text",
        "toolUse",
        "toolResult",
        "image",
        "document",
        "video",
        "cachePoint",
        "reasoningContent",
        "guardContent",
    }
)


@dataclass(frozen=True)
class _Slot:
    """``kind`` is ``text`` (a string leaf), ``json`` (a JSON value leaf) or
    ``call`` (a tool call shown for context; never written back)."""

    kind: str
    path: tuple[str | int, ...] = ()


def _is_converse(body: dict[str, Any]) -> bool:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    seen = False
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
            return False
        content = message.get("content")
        if not isinstance(content, list):
            return False
        for block in content:
            # Anthropic and OpenAI blocks always carry `type`; Converse never.
            if not isinstance(block, dict) or "type" in block:
                return False
            seen = seen or bool(_CONVERSE_BLOCK_KEYS.intersection(block))
    return seen


def native_shape(body: Any) -> str | None:
    """``"gemini"`` or ``"converse"`` for a body in that wire shape, else None."""
    if not isinstance(body, dict) or "input" in body:
        return None
    if isinstance(body.get("contents"), list) and "messages" not in body:
        return GEMINI
    return CONVERSE if _is_converse(body) else None


def view_shape(body: Any) -> str | None:
    """The shape :func:`attach_view` recorded on *body*, if any."""
    marker = body.get(VIEW_MARKER) if isinstance(body, dict) else None
    return marker.get("shape") if isinstance(marker, dict) else None


class _ViewBuilder:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.slots: list[_Slot] = []
        # starts[i]: index of the first view message for native message i,
        # with one extra entry for the end of the view.
        self.starts: list[int] = []
        self._pending: dict[str, list[str]] = {}

    def begin_message(self) -> None:
        self.starts.append(len(self.messages))

    def text(
        self,
        role: str,
        text: Any,
        path: tuple[str | int, ...],
        call_id: str | None = None,
        kind: str = "text",
    ) -> None:
        if not isinstance(text, str) or not text:
            return
        message: dict[str, Any] = {"role": role, "content": text}
        if call_id:
            message["tool_call_id"] = call_id
        self.messages.append(message)
        self.slots.append(_Slot(kind, path))

    def json(self, value: Any, path: tuple[str | int, ...], call_id: str | None) -> None:
        self.text("tool", json.dumps(value, ensure_ascii=False), path, call_id, kind="json")

    def call(self, call_id: str, name: Any, arguments: Any) -> None:
        name = name if isinstance(name, str) else ""
        self._pending.setdefault(name, []).append(call_id)
        self.messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(arguments or {}, ensure_ascii=False),
                        },
                    }
                ],
            }
        )
        self.slots.append(_Slot("call"))

    def result_id(self, explicit: Any, name: Any, fallback: str) -> str:
        """Gemini pairs a response with its call by id when present, else by name."""
        queue = self._pending.get(name if isinstance(name, str) else "", [])
        if isinstance(explicit, str) and explicit in queue:
            queue.remove(explicit)
            return explicit
        # A call with no id got a generated one; pair by name instead.
        if queue:
            return queue.pop(0)
        return explicit if isinstance(explicit, str) and explicit else fallback


def _gemini_view(body: dict[str, Any], view: _ViewBuilder) -> None:
    for key in ("systemInstruction", "system_instruction"):
        system = body.get(key)
        parts = system.get("parts") if isinstance(system, dict) else None
        for k, part in enumerate(parts if isinstance(parts, list) else ()):
            if isinstance(part, dict) and not any(t in part for t in _THOUGHT_KEYS):
                view.text("system", part.get("text"), (key, "parts", k, "text"))
    for i, content in enumerate(body["contents"]):
        view.begin_message()
        if not isinstance(content, dict) or not isinstance(content.get("parts"), list):
            continue
        role = "assistant" if content.get("role") == "model" else "user"
        for k, part in enumerate(content["parts"]):
            if not isinstance(part, dict):
                continue
            here: tuple[str | int, ...] = ("contents", i, "parts", k)
            call_key = "functionCall" if "functionCall" in part else "function_call"
            response_key = "functionResponse" if "functionResponse" in part else "function_response"
            if isinstance(part.get(call_key), dict):
                call = part[call_key]
                call_id = call.get("id") if isinstance(call.get("id"), str) else f"call_{i}_{k}"
                view.call(call_id, call.get("name"), call.get("args"))
            elif isinstance(part.get(response_key), dict):
                result = part[response_key]
                call_id = view.result_id(result.get("id"), result.get("name"), f"call_{i}_{k}")
                response = result.get("response")
                strings = (
                    [f for f, v in response.items() if isinstance(v, str) and v]
                    if isinstance(response, dict)
                    else []
                )
                for field in strings:
                    view.text(
                        "tool", response[field], (*here, response_key, "response", field), call_id
                    )
                if not strings and isinstance(response, dict):
                    view.json(response, (*here, response_key, "response"), call_id)
            elif not any(t in part for t in _THOUGHT_KEYS):
                # A signed or thought text part is the model's own; leave it.
                view.text(role, part.get("text"), (*here, "text"))


def _converse_view(body: dict[str, Any], messages: list[Any], view: _ViewBuilder) -> None:
    system = body.get("system")
    for k, block in enumerate(system if isinstance(system, list) else ()):
        if isinstance(block, dict):
            view.text("system", block.get("text"), ("system", k, "text"))
    for i, message in enumerate(messages):
        view.begin_message()
        role = "assistant" if message.get("role") == "assistant" else "user"
        for j, block in enumerate(message["content"]):
            here: tuple[str | int, ...] = ("messages", i, "content", j)
            if isinstance(block.get("toolUse"), dict):
                use = block["toolUse"]
                call_id = use.get("toolUseId")
                call_id = call_id if isinstance(call_id, str) else f"call_{i}_{j}"
                view.call(call_id, use.get("name"), use.get("input"))
            elif isinstance(block.get("toolResult"), dict):
                result = block["toolResult"]
                call_id = result.get("toolUseId")
                call_id = call_id if isinstance(call_id, str) else None
                inner = result.get("content")
                for k, item in enumerate(inner if isinstance(inner, list) else ()):
                    if not isinstance(item, dict):
                        continue
                    path = (*here, "toolResult", "content", k)
                    if "text" in item:
                        view.text("tool", item["text"], (*path, "text"), call_id)
                    elif "json" in item:
                        view.json(item["json"], (*path, "json"), call_id)
            else:
                view.text(role, block.get("text"), (*here, "text"))


def _build_view(shape: str, body: dict[str, Any], messages: Any) -> _ViewBuilder:
    view = _ViewBuilder()
    if shape == GEMINI:
        _gemini_view(body, view)
    else:
        _converse_view(body, messages, view)
    view.starts.append(len(view.messages))
    return view


def attach_view(body: dict[str, Any]) -> bool:
    """Give a gateway turn's native body a chat-shaped view under ``messages``.

    True if attached. A marker the client sent is dropped first: only this
    function may set it.
    """
    body.pop(VIEW_MARKER, None)
    shape = native_shape(body)
    if shape is None or not isinstance(body.get("gateway"), dict):
        return False
    original = body.get("messages") if shape == CONVERSE else None
    view = _build_view(shape, body, original)
    body["messages"] = view.messages
    body[VIEW_MARKER] = {"shape": shape, "messages": original, "starts": view.starts}
    return True


def _starts(body: Any) -> list[int] | None:
    marker = body.get(VIEW_MARKER) if isinstance(body, dict) else None
    return marker.get("starts") if isinstance(marker, dict) else None


def view_count(body: Any, native: int) -> int:
    """View messages covering the first *native* native messages."""
    starts = _starts(body)
    if starts is None or native <= 0:
        return native
    return starts[min(native, len(starts) - 1)]


def view_recent(body: Any, native: int) -> int:
    """View messages covering the last *native* native messages."""
    starts = _starts(body)
    if starts is None or native <= 0:
        return native
    return starts[-1] - starts[max(0, len(starts) - 1 - native)]


def native_count(body: Any, view: int) -> int:
    """Native messages wholly inside the first *view* view messages."""
    starts = _starts(body)
    if starts is None:
        return view
    return max(i for i, start in enumerate(starts) if i == 0 or start <= view)


def _set(root: Any, path: tuple[str | int, ...], kind: str, text: str) -> None:
    node = root
    for step in path[:-1]:
        node = node[step]
    leaf = path[-1]
    if kind == "json":
        try:
            value = json.loads(text)
        except ValueError:
            return
        if type(value) is not type(node[leaf]):
            return
    else:
        value = text
    node[leaf] = value


def apply_view(
    body: dict[str, Any], marker: dict[str, Any], messages: list[dict[str, Any]]
) -> tuple[dict[str, Any], bool]:
    """The native provider body with the pipeline's text written back.

    *body* is the provider body without control keys; *marker* is what
    :func:`attach_view` stored. Returns ``(body, aligned)``; when the list does
    not line up with the slots the original transcript comes back untouched.
    """
    out = dict(body)
    shape = marker.get("shape")
    if shape == CONVERSE:
        out["messages"] = marker.get("messages")
    else:
        out.pop("messages", None)
    slots = _build_view(str(shape), out, out.get("messages")).slots
    if len(messages) != len(slots):
        return out, False
    out = copy.deepcopy(out)
    for message, slot in zip(messages, slots):
        text = message_text(message)
        if slot.kind != "call" and text is not None:
            _set(out, slot.path, slot.kind, text)
    return out, True
