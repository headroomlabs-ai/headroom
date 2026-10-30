"""A stream-safe, fold-only turn hook: truncates long tool results deterministically.

``stream_safe = True`` because ``on_request`` needs no later re-drive, so the
proxy runs it on streaming turns and the gateway request half runs it even when
``can_redrive`` is false.

Idempotent by construction: a result that already carries the fold marker is
left alone, so in session mode (where the hook's output is what the next turn's
prefix replays) turn N+1 reproduces turn N's bytes exactly. The hook rebinds
``ctx.messages`` to a new list with fresh dicts for the messages it changed and
never mutates the caller's objects.

Handles the chat-completions ``role: "tool"`` string content and Anthropic
``tool_result`` blocks (string content or a list of text blocks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from headroom.proxy.turn_hooks import register_turn_hook

FOLD_MARKER = " ...[folded by test_fold_only]"


def fold_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars or text.endswith(FOLD_MARKER):
        return text
    return text[:max_chars] + FOLD_MARKER


@dataclass
class FoldOnlyHook:
    name: str = "test_fold_only"
    savings_source: str = "test_fold_only"
    stream_safe: bool = True
    max_chars: int = 200
    request_calls: int = 0
    folded: int = 0
    seen_messages: list[list[dict[str, Any]]] = field(default_factory=list)

    def _fold_block(self, block: Any) -> Any:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            return block
        content = block.get("content")
        if isinstance(content, str):
            new = fold_text(content, self.max_chars)
            if new is not content:
                self.folded += 1
                return {**block, "content": new}
            return block
        if isinstance(content, list):
            changed = False
            parts = []
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                ):
                    new = fold_text(part["text"], self.max_chars)
                    if new is not part["text"]:
                        changed = True
                        part = {**part, "text": new}
                parts.append(part)
            if changed:
                self.folded += 1
                return {**block, "content": parts}
        return block

    def on_request(self, ctx: Any) -> None:
        self.request_calls += 1
        messages = ctx.messages
        if not isinstance(messages, list):
            return
        self.seen_messages.append(messages)
        out: list[Any] = []
        changed = False
        for m in messages:
            if not isinstance(m, dict):
                out.append(m)
                continue
            if m.get("role") == "tool" and isinstance(m.get("content"), str):
                new = fold_text(m["content"], self.max_chars)
                if new is not m["content"]:
                    self.folded += 1
                    changed = True
                    out.append({**m, "content": new})
                    continue
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                blocks = [self._fold_block(b) for b in m["content"]]
                if any(a is not b for a, b in zip(blocks, m["content"])):
                    changed = True
                    out.append({**m, "content": blocks})
                    continue
            out.append(m)
        if changed:
            ctx.messages = out


def register(**kwargs: Any) -> FoldOnlyHook:
    hook = FoldOnlyHook(**kwargs)
    register_turn_hook(hook)
    return hook


def install(app: Any, config: Any) -> None:
    """``headroom.proxy_extension`` entry-point shape."""
    register()
