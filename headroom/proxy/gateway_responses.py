"""The OpenAI Responses wire shape on the gateway turn.

``POST /v1/compress`` grew up on two shapes that both carry a ``messages``
list: Anthropic Messages and OpenAI Chat Completions. Codex speaks the third
one — OpenAI Responses — where the transcript is ``input`` (a list of typed
items) and the system prompt is ``instructions``. That body reached the
endpoint and was rejected outright: ``Missing required field: messages``.

The tempting fix is to translate Responses into chat messages, compress, and
translate back. It is the wrong fix, and this module deliberately does not do
it. A Responses transcript carries fields that are opaque to us and that the
provider requires back byte-for-byte — ``reasoning.encrypted_content`` above
all, plus ``call_id`` pairing and Codex's own
``internal_chat_message_metadata_passthrough``. A translation round trip is
lossy by construction, and the loss is silent: the request still looks valid
and the provider rejects it, or worse, accepts it with the model's reasoning
chain quietly severed. ``_responses_input_to_waste_messages`` in the OpenAI
handler carries the same warning for the same reason — it is telemetry-only,
"never used as a compression input".

So this module builds a *view*, not a translation:

* Every compressible text slot in the body becomes one chat-shaped pseudo
  message. One slot, one message — never joined, so putting the text back
  never has to guess how to split it again.
* Every item that is not a text slot — ``reasoning`` above all — produces no
  message at all. It is not shown to the compressor, cannot be rewritten by
  it, and survives in the outgoing body exactly as it arrived.
* A call item's input (``custom_tool_call.input``, ``function_call.arguments``)
  *is* a text slot: 9.2% of transcript bytes across 51 real Codex sessions,
  and resent on every turn like everything else. ``arguments`` is a JSON
  document the provider parses, so it carries one extra condition on the way
  back (``_json_safe``).
* Putting it back is by position: view message *i* owns slot *i*. The slots
  are rebuilt from the original ``input`` rather than carried along, because
  ``build_view`` is a pure function of the body — recomputing is cheaper than
  threading state through the turn and cannot go stale.

If the pipeline ever hands back a list that does not line up with the slots,
``apply_view`` returns the body unchanged rather than guessing. Compression is
an optimisation; a mangled transcript is a broken request.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

#: ``*_call_output`` items: a tool's result coming back into the transcript.
#: This is where nearly all the compressible bulk in an agent session lives.
OUTPUT_ITEM_TYPES: frozenset[str] = frozenset(
    {
        "function_call_output",
        "custom_tool_call_output",
        "local_shell_call_output",
        "apply_patch_call_output",
    }
)

#: Call items, and the field carrying what the model passed to the tool.
#: Measured over 51 real Codex sessions these are 9.2% of transcript bytes --
#: worth taking, but they are not tool *output* and the two differ in kind:
#: output is data the tool produced, while this is what the model said. It is
#: still resent on every turn, and ``custom_tool_call.input`` in particular
#: carries whole scripts and patches.
#:
#: ``function_call.arguments`` is a JSON string the provider parses, so it is
#: guarded separately on the way back (see ``_json_safe``); it is also only
#: 0.3% of bytes, and real ones have been seen carrying encrypted payloads,
#: so very little is lost when the guard refuses.
CALL_INPUT_FIELDS: dict[str, str] = {
    "custom_tool_call": "input",
    "local_shell_call": "input",
    "apply_patch_call": "input",
    "function_call": "arguments",
}

#: Content part types that hold plain text we may rewrite. Responses uses
#: ``input_text`` on the way in and ``output_text`` on the way out; both appear
#: in a resent transcript.
TEXT_PART_TYPES: frozenset[str] = frozenset({"input_text", "output_text", "text"})


#: Where a piece of text lived, so it can be put back.
#: ``kind`` is one of:
#:   ``instructions``  — the top-level ``instructions`` string
#:   ``content``       — ``input[i]["content"][j]["text"]``
#:   ``content_str``   — ``input[i]["content"]`` held a bare string
#:   ``output``        — ``input[i]["output"][j]["text"]``
#:   ``output_str``    — ``input[i]["output"]`` held a bare string
#:   ``input_str``     — the whole ``input`` was a bare string
#:   ``call``          — ``input[i][<the item's call-input field>]``
@dataclass(frozen=True)
class Slot:
    kind: str
    item: int = -1
    part: int = -1
    field: str = ""


@dataclass
class ResponsesView:
    """A chat-shaped read of a Responses body, plus how to undo it."""

    messages: list[dict[str, Any]]
    slots: list[Slot]

    def __len__(self) -> int:
        return len(self.slots)


def is_responses_body(body: Any) -> bool:
    """True for a Responses-shaped request body.

    ``input`` is the discriminator and ``messages`` is the veto: a body
    carrying both is a chat body that happens to have an ``input`` field, and
    the chat path already knows what to do with it.
    """
    if not isinstance(body, dict):
        return False
    return "input" in body and "messages" not in body


#: Set on the body when a view is attached, so rebuilding the provider body
#: knows to fold it back rather than having to re-derive the wire shape. It is
#: a Headroom control key and is stripped like the rest of them, so it never
#: reaches a provider.
#:
#: Sniffing for ``input`` instead looks equivalent and is not: by then the view
#: is sitting under ``messages``, so the arrival-time predicate cannot be
#: reused, and a chat body that merely happens to carry an ``input`` field
#: would be mistaken for a Responses turn and have its ``messages`` dropped.
#: The marker says what actually happened rather than inferring it.
VIEW_MARKER = "_headroom_responses_view"


def mark_view(body: dict[str, Any]) -> None:
    body[VIEW_MARKER] = True


def carries_view(body: Any) -> bool:
    """True when :func:`mark_view` attached a view to this body."""
    return isinstance(body, dict) and body.get(VIEW_MARKER) is True


def _part_text(part: Any) -> str | None:
    if not isinstance(part, dict):
        return None
    if part.get("type") not in TEXT_PART_TYPES:
        return None
    text = part.get("text")
    return text if isinstance(text, str) else None


def build_view(body: dict[str, Any]) -> ResponsesView:
    """A chat-shaped view of every text slot in *body*, in document order.

    Pure: the same body always yields the same slots, which is what lets
    ``apply_view`` rebuild them instead of being handed them.
    """
    messages: list[dict[str, Any]] = []
    slots: list[Slot] = []

    def emit(role: str, text: str, slot: Slot, *, tool_call_id: str | None = None) -> None:
        message: dict[str, Any] = {"role": role, "content": text}
        if tool_call_id:
            # The pipeline keys tool results off this; without it a fold that
            # groups by call would treat every result as the same call.
            message["tool_call_id"] = tool_call_id
        messages.append(message)
        slots.append(slot)

    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions:
        emit("system", instructions, Slot("instructions"))

    data = body.get("input")
    if isinstance(data, str):
        if data:
            emit("user", data, Slot("input_str"))
        return ResponsesView(messages, slots)
    if not isinstance(data, list):
        return ResponsesView(messages, slots)

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        kind = item.get("type")

        if kind in OUTPUT_ITEM_TYPES:
            call_id = item.get("call_id")
            call_id = call_id if isinstance(call_id, str) else None
            output = item.get("output")
            if isinstance(output, str):
                if output:
                    emit("tool", output, Slot("output_str", index), tool_call_id=call_id)
            elif isinstance(output, list):
                for part_index, part in enumerate(output):
                    text = _part_text(part)
                    if text:
                        emit(
                            "tool",
                            text,
                            Slot("output", index, part_index),
                            tool_call_id=call_id,
                        )
            continue

        call_field = CALL_INPUT_FIELDS.get(kind or "")
        if call_field is not None:
            value = item.get(call_field)
            if isinstance(value, str) and value:
                call_id = item.get("call_id")
                emit(
                    "assistant",
                    value,
                    Slot("call", index, field=call_field),
                    tool_call_id=call_id if isinstance(call_id, str) else None,
                )
            continue

        # A `message` item. Anything else -- `reasoning` above all -- is opaque
        # on purpose and contributes no slot.
        if kind is not None and kind != "message":
            continue

        role = item.get("role")
        role = role if isinstance(role, str) and role else "user"
        content = item.get("content")
        if isinstance(content, str):
            if content:
                emit(role, content, Slot("content_str", index))
        elif isinstance(content, list):
            for part_index, part in enumerate(content):
                text = _part_text(part)
                if text:
                    emit(role, text, Slot("content", index, part_index))

    return ResponsesView(messages, slots)


def _json_safe(original: Any, replacement: str) -> bool:
    """May *replacement* stand in for *original* in a call-input field?

    ``function_call.arguments`` is a JSON document the provider parses to
    dispatch the call. A compressor works on text and has no reason to emit
    valid JSON, so shortening one would usually turn a working request into a
    parse error at the provider -- the exact failure this module exists to
    avoid. When the original parses as JSON the replacement must too, or the
    field keeps what it had.

    A field that was never JSON (``custom_tool_call.input`` holds shell
    commands and patches) has nothing to protect and passes straight through.
    """
    if not isinstance(original, str):
        return False
    try:
        json.loads(original)
    except (ValueError, TypeError):
        return True
    try:
        json.loads(replacement)
    except (ValueError, TypeError):
        return False
    return True


def _message_text(message: Any) -> str | None:
    """The text the pipeline left in a view message.

    It may hand back a bare string or a content list, depending on which
    stage last touched it, so both are accepted.
    """
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [t for t in (_part_text(p) for p in content) if t is not None]
        if parts:
            return "\n".join(parts)
        return ""
    return None


def apply_view(
    body: dict[str, Any],
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    """Write the view's text back into a copy of *body*.

    Returns ``(body, applied)``. ``applied`` is False when the messages do not
    line up with the slots rebuilt from *body*, in which case the body comes
    back untouched: a transcript we cannot map confidently is one we must not
    rewrite.
    """
    view = build_view(body)
    if len(messages) != len(view.slots):
        return body, False

    out = copy.deepcopy(body)
    data = out.get("input")
    changed = False

    for message, slot in zip(messages, view.slots):
        text = _message_text(message)
        if text is None:
            continue

        if slot.kind == "instructions":
            if out.get("instructions") != text:
                out["instructions"] = text
                changed = True
            continue

        if slot.kind == "input_str":
            if out.get("input") != text:
                out["input"] = text
                changed = True
            continue

        if not isinstance(data, list) or not (0 <= slot.item < len(data)):
            continue
        item = data[slot.item]
        if not isinstance(item, dict):
            continue

        if slot.kind == "call":
            if not _json_safe(item.get(slot.field), text):
                continue
            if item.get(slot.field) != text:
                item[slot.field] = text
                changed = True
            continue

        if slot.kind in ("content_str", "output_str"):
            field = "content" if slot.kind == "content_str" else "output"
            if item.get(field) != text:
                item[field] = text
                changed = True
            continue

        field = "content" if slot.kind == "content" else "output"
        parts = item.get(field)
        if not isinstance(parts, list) or not (0 <= slot.part < len(parts)):
            continue
        part = parts[slot.part]
        if isinstance(part, dict) and part.get("text") != text:
            part["text"] = text
            changed = True

    return out, changed
