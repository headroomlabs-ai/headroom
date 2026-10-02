"""Hide proxy-handled memory tool calls from an Anthropic SSE stream.

The proxy injects its own memory tools (``memory_save``, ``memory_search``,
...) into requests. The client never declared them, so when the model calls
one the client cannot run it: Claude Code answers ``No such tool available``
and the model concludes the save or search failed (GH #2195).

``MemoryToolStreamFilter`` sits between the upstream SSE stream and the
client. It forwards every event live except:

* the ``content_block_*`` events of a ``tool_use`` block whose name is a
  proxy-owned memory tool, which it withholds;
* ``message_delta`` / ``message_stop``, which it holds until the caller
  knows whether the turn ends here or continues with a server-side
  continuation round.

Content block indices are rewritten so the blocks the client sees stay
contiguous, including across continuation rounds whose blocks are appended
to the same client message. Frames that need no rewrite are forwarded as the
exact upstream bytes.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Collection
from typing import Any

from headroom.proxy.sse_byte_buffer_policy import (
    SSE_DATA_LINE_PREFIX,
    SSE_EVENT_LINE_PREFIX,
    find_sse_event_terminator,
)

logger = logging.getLogger(__name__)

_BLOCK_EVENTS = ("content_block_start", "content_block_delta", "content_block_stop")
_TAIL_EVENTS = ("message_delta", "message_stop")


def _parse_frame(frame: bytes) -> tuple[str | None, dict[str, Any] | None]:
    """Return ``(event_name, payload)`` for one complete SSE frame."""
    event_name: str | None = None
    data_lines: list[str] = []
    for line in frame.decode("utf-8").splitlines():
        if line.startswith(SSE_EVENT_LINE_PREFIX):
            event_name = line[len(SSE_EVENT_LINE_PREFIX) :].strip()
        elif line.startswith(SSE_DATA_LINE_PREFIX):
            data_lines.append(line[len(SSE_DATA_LINE_PREFIX) :].lstrip())
    if not data_lines:
        return event_name, None
    try:
        payload = json.loads("\n".join(data_lines))
    except json.JSONDecodeError:
        return event_name, None
    if not isinstance(payload, dict):
        return event_name, None
    return event_name or payload.get("type"), payload


def _render_frame(event_name: str, payload: dict[str, Any]) -> bytes:
    return f"event: {event_name}\ndata: {json.dumps(payload)}\n\n".encode()


class MemoryToolStreamFilter:
    """Per-round SSE filter; see the module docstring.

    Args:
        tool_names: Memory tool names the proxy injected and will execute
            itself. Tools the client declared are never withheld.
        index_offset: Client-side index of this round's first visible block.
            Zero for the first round; ``next_index`` of the previous round
            for a continuation round.
        forward_message_start: False for continuation rounds, whose blocks
            extend the message the client already opened.
    """

    def __init__(
        self,
        tool_names: Collection[str],
        *,
        index_offset: int = 0,
        forward_message_start: bool = True,
    ) -> None:
        self._tool_names = frozenset(tool_names)
        self._forward_message_start = forward_message_start
        self._buffer = bytearray()
        self._index_map: dict[int, int] = {}
        self._hidden_indices: set[int] = set()
        self.next_index = index_offset
        self.hidden_tool_names: list[str] = []
        self.visible_tool_use = False
        self.stop_reason: str | None = None
        self._tail: list[tuple[str, dict[str, Any] | None, bytes]] = []

    @property
    def hid_tool_calls(self) -> bool:
        return bool(self.hidden_tool_names)

    def feed(self, chunk: bytes) -> list[bytes]:
        """Consume upstream bytes; return the frames to forward now."""
        self._buffer.extend(chunk)
        out: list[bytes] = []
        while (match := find_sse_event_terminator(self._buffer)) is not None:
            idx, terminator_len = match
            frame = bytes(self._buffer[: idx + terminator_len])
            del self._buffer[: idx + terminator_len]
            forwarded = self._route(frame)
            if forwarded is not None:
                out.append(forwarded)
        return out

    def _route(self, frame: bytes) -> bytes | None:
        event_name, payload = _parse_frame(frame)
        if payload is None or event_name is None:
            return frame

        if event_name == "message_start":
            return frame if self._forward_message_start else None

        if event_name in _TAIL_EVENTS:
            if event_name == "message_delta":
                delta = payload.get("delta")
                if isinstance(delta, dict) and delta.get("stop_reason"):
                    self.stop_reason = delta["stop_reason"]
            self._tail.append((event_name, payload, frame))
            return None

        if event_name not in _BLOCK_EVENTS:
            return frame

        upstream_index = payload.get("index")
        if not isinstance(upstream_index, int):
            return frame

        if event_name == "content_block_start":
            block = payload.get("content_block")
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("name") in self._tool_names
            ):
                self._hidden_indices.add(upstream_index)
                self.hidden_tool_names.append(block["name"])
                return None
            if isinstance(block, dict) and block.get("type") == "tool_use":
                self.visible_tool_use = True
            self._index_map[upstream_index] = self.next_index
            self.next_index += 1

        if upstream_index in self._hidden_indices:
            return None
        client_index = self._index_map.get(upstream_index, upstream_index)
        if client_index == upstream_index:
            return frame
        return _render_frame(event_name, {**payload, "index": client_index})

    def closing_frames(self) -> list[bytes]:
        """End the client message after this round without a continuation."""
        ends_on_hidden_call = (
            self.hid_tool_calls and not self.visible_tool_use and self.stop_reason == "tool_use"
        )
        return self.tail_frames(stop_reason="end_turn" if ends_on_hidden_call else None)

    def tail_frames(self, *, stop_reason: str | None = None) -> list[bytes]:
        """Return the held ``message_delta`` / ``message_stop`` frames.

        ``stop_reason`` replaces the upstream stop reason, e.g. ``end_turn``
        when the only tool calls were withheld memory calls: a ``tool_use``
        stop with no visible ``tool_use`` block leaves the client waiting on
        a tool it was never shown.
        """
        frames: list[bytes] = []
        for event_name, payload, raw in self._tail:
            if (
                stop_reason is not None
                and event_name == "message_delta"
                and payload is not None
                and isinstance(payload.get("delta"), dict)
            ):
                frames.append(
                    _render_frame(
                        event_name,
                        {**payload, "delta": {**payload["delta"], "stop_reason": stop_reason}},
                    )
                )
            else:
                frames.append(raw)
        if self._buffer:
            # An unterminated trailing frame: pass it through untouched.
            frames.append(bytes(self._buffer))
            self._buffer.clear()
        return frames
