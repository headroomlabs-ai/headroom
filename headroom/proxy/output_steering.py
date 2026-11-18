"""Byte-stable output verbosity steering helpers."""

from __future__ import annotations

from typing import Any

from headroom.proxy.output_verbosity_policy import (
    STEERING_SENTINEL as _STEERING_SENTINEL,
)
from headroom.proxy.output_verbosity_policy import (
    STEERING_SUFFIX as _STEERING_SUFFIX,
)
from headroom.proxy.output_verbosity_policy import (
    replace_or_append_steering_block,
    steering_text,
)


def apply_verbosity_steering(body: dict[str, Any], level: int) -> bool:
    """Append the steering block to the tail of the Anthropic system prompt.

    Appending after the last system block keeps any ``cache_control``
    breakpoint on an earlier block intact: the cached prefix is unchanged and
    only the small, byte-stable steering block is reprocessed.
    """
    text = steering_text(level)
    if text is None:
        return False

    system = body.get("system")
    if system is None:
        body["system"] = [{"type": "text", "text": text}]
        return True
    if isinstance(system, str):
        body["system"] = [
            {"type": "text", "text": system},
            {"type": "text", "text": text},
        ]
        return True
    if isinstance(system, list):
        for block in system:
            # Guard the text is a string before ``startswith``: a malformed
            # client block (``{"type": "text", "text": null}``) would otherwise
            # raise ``AttributeError`` here and 500 the request. The OpenAI chat
            # sibling below already guards this exact case.
            block_text = block.get("text") if isinstance(block, dict) else None
            if isinstance(block_text, str) and block_text.startswith(_STEERING_SENTINEL):
                if block_text == text:
                    return False
                block["text"] = text
                return True
        system.append({"type": "text", "text": text})
        return True
    return False


def apply_openai_chat_verbosity_steering(
    body: dict[str, Any],
    level: int,
) -> bool:
    """Append or replace the steering block in an OpenAI chat/completions body.

    OpenAI ``/v1/chat/completions`` carries the system prompt as a
    ``role: "system"`` (or ``"developer"``) message inside ``messages`` rather
    than a top-level field, so it needs its own injector (the Anthropic
    ``system`` and Responses ``instructions`` variants do not reach it — the
    root cause of GitHub Copilot CLI seeing zero output savings, #2302).

    The block is appended to the tail of the last system/developer message so a
    treatment conversation's steering stays byte-stable across turns (and
    re-applies idempotently via the sentinel). When the request carries no
    system message at all, one is inserted at the front. Returns True only when
    the body actually changed.
    """
    text = steering_text(level)
    if text is None:
        return False

    messages = body.get("messages")
    if not isinstance(messages, list):
        return False

    target: dict[str, Any] | None = None
    for message in messages:
        if isinstance(message, dict) and message.get("role") in ("system", "developer"):
            target = message
    if target is None:
        # No system prompt to append to — insert one carrying just the block.
        messages.insert(0, {"role": "system", "content": text})
        return True

    content = target.get("content")
    if content is None:
        target["content"] = text
        return True
    if isinstance(content, str):
        updated, changed = replace_or_append_steering_block(content, text)
        if changed:
            target["content"] = updated
        return changed
    if isinstance(content, list):
        # OpenAI also accepts a content-part list ([{"type": "text", ...}]).
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
                and part["text"].startswith(_STEERING_SENTINEL)
            ):
                if part["text"] == text:
                    return False
                part["text"] = text
                return True
        content.append({"type": "text", "text": text})
        return True
    return False


def _input_already_steered(items: Any) -> bool:
    """True when a user message in a Responses ``input`` ends with a steering block.

    Only user messages are read, and only a block at the very end counts: that is where
    ``replace_or_append_steering_block`` puts it. Tool outputs and quoted source can contain the
    marker text (this module's own policy file does) without anything having been steered, and
    must not switch core's steering off.
    """
    if isinstance(items, str):  # Responses accepts a bare string as one user message
        items = [{"role": "user", "content": items}]
    if not isinstance(items, list):
        return False
    for item in items:
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        if item.get("type") not in (None, "message"):
            continue
        content = item.get("content")
        texts = (
            [content]
            if isinstance(content, str)
            else [
                part.get("text")
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            ]
            if isinstance(content, list)
            else []
        )
        for text in texts:
            stripped = text.rstrip()
            if stripped.endswith(_STEERING_SUFFIX) and _STEERING_SENTINEL in stripped:
                return True
    return False


def apply_openai_responses_verbosity_steering(
    body: dict[str, Any],
    level: int,
) -> bool:
    """Append or replace steering in OpenAI Responses ``instructions``.

    Leaves the body alone when a steering block already rides in ``input``. A
    client whose upstream rejects ``instructions`` (Kiro's runtime returns 400
    for it) has its steering placed in the user turns by an extension's
    ``on_responses_create`` hook; adding a second copy here would break that
    request outright.
    """
    text = steering_text(level)
    if text is None:
        return False
    if _input_already_steered(body.get("input")):
        return False

    instructions = body.get("instructions")
    if instructions is None:
        body["instructions"] = text
        return True
    if not isinstance(instructions, str):
        return False

    updated, changed = replace_or_append_steering_block(instructions, text)
    if changed:
        body["instructions"] = updated
    return changed
