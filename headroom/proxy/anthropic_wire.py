"""Loss-minimizing helpers for the native Anthropic Messages wire envelope.

The proxy is allowed to understand the message content it optimizes, but it
must not need to understand every field Anthropic adds to the surrounding
protocol.  This module keeps that distinction explicit: known message events
are reconstructed into a normal response dictionary, while every SSE frame the
reconstruction cannot use (unknown events, and known events whose bytes or
payload are malformed) is retained as opaque bytes and replayed by
:meth:`AnthropicSSEEnvelope.render`.

The opaque data is deliberately kept out of the provider JSON.  It is an
internal rendering detail and can therefore never accidentally be serialized
upstream as a synthetic Headroom field.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Collection
from dataclasses import dataclass, field, replace
from typing import Any

from headroom.copilot_auth import build_copilot_upstream_url

_DANGEROUS_TOOL_USE_RE = re.compile(r"(?:^|,)\s*dangerous-tool-use-[^,\s]+", re.IGNORECASE)
# Members of each known event (and delta type) that reconstruction interprets.
# Anything else on a known frame is an extension carried by ``_FrameExtensions``.
_EVENT_FIELDS: dict[str, frozenset[str]] = {
    "message_start": frozenset({"type", "message"}),
    "content_block_start": frozenset({"type", "index", "content_block"}),
    "content_block_delta": frozenset({"type", "index", "delta"}),
    "content_block_stop": frozenset({"type", "index"}),
    "message_delta": frozenset({"type", "delta", "usage"}),
    "message_stop": frozenset({"type"}),
}
_DELTA_FIELDS: dict[str, frozenset[str]] = {
    "text_delta": frozenset({"type", "text"}),
    "input_json_delta": frozenset({"type", "partial_json"}),
    "thinking_delta": frozenset({"type", "thinking"}),
    "signature_delta": frozenset({"type", "signature"}),
    "citations_delta": frozenset({"type", "citation"}),
}
_MESSAGE_DELTA_FIELDS = ("stop_reason", "stop_sequence", "stop_details")
# The payload member each known delta type must carry, and its JSON type.
_DELTA_PAYLOADS: dict[str, tuple[str, type]] = {
    "text_delta": ("text", str),
    "input_json_delta": ("partial_json", str),
    "thinking_delta": ("thinking", str),
    "signature_delta": ("signature", str),
    "citations_delta": ("citation", dict),
}
# The block field each string delta is concatenated onto.
_ACCUMULATED_FIELDS = {
    "text_delta": "text",
    "input_json_delta": "_partial_json",
    "thinking_delta": "thinking",
}
_KNOWN_MESSAGE_FIELDS = frozenset(
    {
        "id",
        "type",
        "role",
        "model",
        "content",
        "stop_reason",
        "stop_sequence",
        "stop_details",
        "usage",
    }
)


@dataclass(frozen=True)
class _SSEFrame:
    """One raw SSE frame and its decoded payload, when it has one."""

    raw: bytes
    event_type: str
    payload: dict[str, Any] | None


def build_anthropic_upstream_url(base_url: str, path: str, raw_query: str = "") -> str:
    """Build an Anthropic upstream URL without dropping the incoming query.

    ``raw_query`` is intentionally not parsed and re-encoded.  Beta query
    parameters are protocol data, so preserving their spelling and ordering is
    safer than normalizing them through a mapping.
    """

    url = build_copilot_upstream_url(base_url, path)
    if raw_query:
        return f"{url}?{raw_query}"
    return url


def has_dangerous_tool_use_beta(value: str | None) -> bool:
    """Return whether a beta header contains an auto-mode capability token."""

    return bool(value and _DANGEROUS_TOOL_USE_RE.search(value))


def is_safeguard_capable_request(body: Any, anthropic_beta: str | None) -> bool:
    """Classify a request without inspecting or retaining the safeguard value."""

    return isinstance(body, dict) and (
        "safeguards" in body or has_dangerous_tool_use_beta(anthropic_beta)
    )


def strip_safeguard_payload(value: Any) -> Any:
    """Copy an observation while removing classifier payload fields.

    This is only for logs, pipeline events, and diagnostics.  The actual
    upstream response is never passed through this helper, so wire fidelity is
    unaffected.
    """

    if isinstance(value, dict):
        return {
            key: strip_safeguard_payload(item)
            for key, item in value.items()
            if key not in {"safeguards", "safeguard_results"}
        }
    if isinstance(value, list):
        return [strip_safeguard_payload(item) for item in value]
    return value


def preserve_opaque_response_fields(
    original: dict[str, Any], replacement: dict[str, Any]
) -> dict[str, Any]:
    """Retain unknown Anthropic response attachments across a transform.

    Response transforms such as CCR may replace the message produced by the
    first upstream call with a continuation message.  Capability attachments
    belong to the surrounding protocol rather than to Headroom's mutable
    message view, so keep them unless the replacement explicitly supplies its
    own value.
    """

    merged = copy.deepcopy(replacement)
    for key, value in original.items():
        if key not in _KNOWN_MESSAGE_FIELDS:
            merged.setdefault(key, copy.deepcopy(value))
    return merged


def _split_frames(raw: bytes) -> list[bytes]:
    """Split SSE bytes while retaining each frame's original delimiters."""

    frames: list[bytes] = []
    start = 0
    for match in re.finditer(rb"\r\n\r\n|\n\n|\r\r", raw):
        end = match.end()
        if raw[start : match.start()].strip():
            frames.append(raw[start:end])
        start = end
    if raw[start:].strip():
        frames.append(raw[start:])
    return frames


def _frame_parts(raw: bytes) -> tuple[str, str | None]:
    event_name = ""
    data_lines: list[bytes] = []
    for line in raw.splitlines():
        if line.startswith(b"event:"):
            try:
                event_name = line[6:].strip().decode("utf-8")
            except UnicodeDecodeError:
                return "", None
        elif line.startswith(b"data:"):
            data_lines.append(line[5:].lstrip())
    try:
        return event_name, b"\n".join(data_lines).decode("utf-8")
    except UnicodeDecodeError:
        # Leave the complete frame opaque so rendering can replay its exact
        # bytes. Replacing malformed input would violate the proxy's fail-open
        # wire-fidelity contract.
        return event_name, None


def _json_payload(raw: bytes) -> tuple[str, dict[str, Any] | None]:
    event_name, data = _frame_parts(raw)
    if data is None:
        return event_name, None
    if not data or data == "[DONE]":
        return event_name or "[DONE]", None
    try:
        payload = json.loads(data)
    except (TypeError, json.JSONDecodeError):
        return event_name, None
    if not isinstance(payload, dict):
        return event_name, None
    return event_name or str(payload.get("type", "")), payload


def _sse_event(payload: dict[str, Any], *, event_name: str | None = None) -> bytes:
    name = event_name or str(payload.get("type", "message"))
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


def _wire_index(value: Any) -> int | None:
    """Return a content block index, or None when it is not a valid one.

    ``index`` is provider JSON and may be any JSON value.  Only a non-negative
    integer identifies a block; anything else (including an unhashable object
    or array) must leave its frame unreconstructable instead of reaching a
    dictionary lookup.
    """

    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return int(value)
    return None


def _extensions(payload: dict[str, Any], known: Collection[str]) -> dict[str, Any]:
    """Return a copy of the members of ``payload`` this module does not interpret."""

    return {key: copy.deepcopy(value) for key, value in payload.items() if key not in known}


@dataclass
class _BlockExtensions:
    """Unknown members carried by one content block's known frames."""

    start: dict[str, Any] = field(default_factory=dict)
    stop: dict[str, Any] = field(default_factory=dict)
    # delta type -> (members on the event, members inside its ``delta``)
    deltas: dict[str, tuple[dict[str, Any], dict[str, Any]]] = field(default_factory=dict)


@dataclass
class _FrameExtensions:
    """Unknown members of known frames, keyed by the canonical frame they rode on.

    Rendering regenerates known frames from the message dictionary, which has
    no place for members added to the events themselves.  They are kept here
    and re-attached to the corresponding regenerated frame instead.
    """

    message_start: dict[str, Any] = field(default_factory=dict)
    message_delta: dict[str, Any] = field(default_factory=dict)
    message_delta_delta: dict[str, Any] = field(default_factory=dict)
    # ``message_delta.usage`` keys upstream sent; values come from the
    # rendered response so a CCR continuation reports its own usage.
    message_delta_usage_keys: list[str] = field(default_factory=list)
    message_stop: dict[str, Any] = field(default_factory=dict)
    # Keyed by position in ``message["content"]``, not by wire index.
    blocks: dict[int, _BlockExtensions] = field(default_factory=dict)

    def for_content(self, original: Any, rendered: Any) -> _FrameExtensions:
        """Keep block members only where the rendered block is the original one.

        A response transform such as CCR can replace content blocks; members
        observed on the original block's frames do not describe its
        replacement.  Message-level members belong to the surrounding protocol
        and are always kept.
        """

        before = original if isinstance(original, list) else []
        after = rendered if isinstance(rendered, list) else []
        blocks = {
            position: extensions
            for position, extensions in self.blocks.items()
            if position < len(before)
            and position < len(after)
            and after[position] == before[position]
        }
        return replace(self, blocks=blocks)


@dataclass
class _Reconstruction:
    message: dict[str, Any]
    extensions: _FrameExtensions
    saw_start: bool
    saw_stop: bool
    saw_error: bool
    open_blocks: set[int]
    # Positions of the frames the message (plus extensions) was built from.
    consumed: set[int]


def _response_from_events(frames: list[_SSEFrame]) -> _Reconstruction:
    """Reconstruct the message and report which frames it was built from.

    The ``consumed`` positions are the only frames that
    :func:`_render_known_response` can regenerate.  Every other frame,
    including a known event whose payload is malformed or does not fit the
    stream's structure, must be replayed verbatim.
    """

    response: dict[str, Any] = {"content": [], "usage": {}}
    extensions = _FrameExtensions()
    blocks_by_index: dict[int, dict[str, Any]] = {}
    # Keyed by ``id()`` of the block dictionary; the dictionaries stay alive
    # in ``blocks_by_index`` or ``response["content"]`` for the whole parse.
    block_extensions: dict[int, _BlockExtensions] = {}
    current_block: dict[str, Any] | None = None
    appended: set[int] = set()
    open_blocks: set[int] = set()
    consumed: set[int] = set()
    saw_start = saw_stop = saw_error = False

    for position, frame in enumerate(frames):
        data = frame.payload
        event_type = frame.event_type
        if data is None:
            continue
        if event_type == "message_start":
            saw_start = True
            message = data.get("message")
            usage = message.get("usage") if isinstance(message, dict) else None
            # ``usage`` is merged into by later message_delta frames, so a
            # non-object value would make the message unreconstructable.
            if isinstance(message, dict) and (usage is None or isinstance(usage, dict)):
                consumed.add(position)
                extensions.message_start.update(_extensions(data, _EVENT_FIELDS[event_type]))
                # Start with the complete upstream message object.  This is
                # what preserves future top-level attachments.
                response.update(copy.deepcopy(message))
                # The stream's content blocks below are authoritative.  A
                # defensive non-empty content array in message_start must not
                # be duplicated when those blocks are replayed.
                response["content"] = []
                if response.get("usage") is None:
                    response["usage"] = {}
        elif event_type == "content_block_start":
            block = data.get("content_block")
            if not isinstance(block, dict):
                continue
            raw_index = data.get("index")
            if raw_index is None:
                index = len(response["content"])
            else:
                parsed_index = _wire_index(raw_index)
                if parsed_index is None:
                    continue
                index = parsed_index
            current_block = copy.deepcopy(block)
            blocks_by_index[index] = current_block
            block_extensions[id(current_block)] = _BlockExtensions(
                start=_extensions(data, _EVENT_FIELDS[event_type])
            )
            open_blocks.add(index)
            consumed.add(position)
        elif event_type == "content_block_delta":
            delta = data.get("delta")
            if not isinstance(delta, dict):
                continue
            raw_index = data.get("index")
            block_index = _wire_index(raw_index)
            if raw_index is not None and block_index is None:
                continue
            target = blocks_by_index.get(block_index) if block_index is not None else current_block
            dtype = delta.get("type")
            if target is None or not isinstance(dtype, str) or dtype not in _DELTA_FIELDS:
                continue
            # Validate the raw member: a missing, null or falsy non-string value
            # is not an empty delta, and must not be consumed as one.
            member, member_type = _DELTA_PAYLOADS[dtype]
            payload = delta.get(member)
            if not isinstance(payload, member_type):
                continue
            accumulated = _ACCUMULATED_FIELDS.get(dtype)
            if accumulated is not None and not isinstance(target.get(accumulated, ""), str):
                continue
            if dtype == "citations_delta" and not isinstance(target.get("citations", []), list):
                continue
            consumed.add(position)
            event_extensions, delta_extensions = block_extensions[id(target)].deltas.setdefault(
                dtype, ({}, {})
            )
            event_extensions.update(_extensions(data, _EVENT_FIELDS[event_type]))
            delta_extensions.update(_extensions(delta, _DELTA_FIELDS[dtype]))
            if accumulated is not None:
                target[accumulated] = target.get(accumulated, "") + payload
            elif dtype == "signature_delta":
                target["signature"] = payload
            elif dtype == "citations_delta":
                target.setdefault("citations", []).append(payload)
        elif event_type == "content_block_stop":
            raw_index = data.get("index")
            block_index = _wire_index(raw_index)
            if raw_index is not None and block_index is None:
                continue
            target = blocks_by_index.get(block_index) if block_index is not None else current_block
            if target is None:
                continue
            consumed.add(position)
            block_extensions[id(target)].stop.update(_extensions(data, _EVENT_FIELDS[event_type]))
            partial = target.pop("_partial_json", None)
            if partial is not None:
                try:
                    target["input"] = json.loads(partial) if partial else {}
                except (TypeError, json.JSONDecodeError):
                    target["input"] = {}
            appended_key = block_index if block_index is not None else id(target)
            if appended_key not in appended:
                response["content"].append(target)
                appended.add(appended_key)
            if block_index is not None:
                open_blocks.discard(block_index)
            current_block = None
        elif event_type == "message_delta":
            delta = data.get("delta")
            usage = data.get("usage")
            if (delta is not None and not isinstance(delta, dict)) or (
                usage is not None and not isinstance(usage, dict)
            ):
                continue
            consumed.add(position)
            if isinstance(delta, dict):
                for key in _MESSAGE_DELTA_FIELDS:
                    if key in delta:
                        response[key] = copy.deepcopy(delta[key])
                extensions.message_delta_delta.update(_extensions(delta, _MESSAGE_DELTA_FIELDS))
            if isinstance(usage, dict):
                response.setdefault("usage", {}).update(copy.deepcopy(usage))
                for key in usage:
                    if key not in extensions.message_delta_usage_keys:
                        extensions.message_delta_usage_keys.append(key)
            # Some Anthropic additions are attached directly to the delta
            # event rather than nested below ``delta``. Preserve them as
            # response fields without interpreting their schemas.
            event_extensions = _extensions(data, _EVENT_FIELDS[event_type])
            response.update(copy.deepcopy(event_extensions))
            extensions.message_delta.update(event_extensions)
        elif event_type == "message_stop":
            saw_stop = True
            consumed.add(position)
            extensions.message_stop.update(_extensions(data, _EVENT_FIELDS[event_type]))
        elif event_type == "error":
            saw_error = True

    extensions.blocks = {
        index: block_extensions[id(block)]
        for index, block in enumerate(response["content"])
        if id(block) in block_extensions
    }
    return _Reconstruction(
        response, extensions, saw_start, saw_stop, saw_error, open_blocks, consumed
    )


@dataclass
class AnthropicSSEEnvelope:
    """Parsed Anthropic response plus opaque frames retained for replay."""

    message: dict[str, Any]
    # (number of reconstructed frames before it, raw bytes) for every frame
    # that the reconstructed message cannot regenerate.
    _opaque_frames: list[tuple[int, bytes]]
    _extensions: _FrameExtensions
    saw_message_start: bool
    saw_message_stop: bool
    saw_error: bool
    saw_unreconstructable_frame: bool
    open_block_indices: set[int]

    @classmethod
    def parse(cls, raw_sse_bytes: bytes) -> AnthropicSSEEnvelope:
        """Parse Anthropic SSE bytes while retaining opaque frames verbatim.

        Args:
            raw_sse_bytes: Complete or partial Anthropic SSE wire bytes.

        Returns:
            A request-local envelope containing the reconstructed message and
            every original frame needed for loss-minimizing replay.
        """

        frames = [_SSEFrame(raw, *_json_payload(raw)) for raw in _split_frames(raw_sse_bytes)]
        reconstruction = _response_from_events(frames)

        # Anything the reconstruction did not consume is replayed verbatim.
        # A recognized event name is not enough: a known frame with malformed
        # bytes or an unusable payload would otherwise be neither rendered
        # nor replayed, and would silently disappear from the stream.
        opaque_frames: list[tuple[int, bytes]] = []
        known_count = 0
        saw_unreconstructable_frame = False
        for position, frame in enumerate(frames):
            if position in reconstruction.consumed:
                known_count += 1
                continue
            if frame.event_type in _EVENT_FIELDS:
                saw_unreconstructable_frame = True
            opaque_frames.append((known_count, frame.raw))

        return cls(
            reconstruction.message,
            opaque_frames,
            reconstruction.extensions,
            reconstruction.saw_start,
            reconstruction.saw_stop,
            reconstruction.saw_error,
            saw_unreconstructable_frame,
            reconstruction.open_blocks,
        )

    @classmethod
    def from_events(cls, events: list[dict[str, Any]]) -> AnthropicSSEEnvelope:
        """Build an envelope for callers that already decoded event JSON."""

        raw = b"".join(_sse_event(event) for event in events if isinstance(event, dict))
        return cls.parse(raw)

    def is_complete(self) -> bool:
        """Return whether the stream has complete structural framing."""

        return (
            self.saw_message_start
            and self.saw_message_stop
            and not self.saw_error
            and not self.open_block_indices
        )

    def is_message_reconstructable(self) -> bool:
        """Return whether strict callers can safely use the message dictionary."""

        return self.is_complete() and not self.saw_unreconstructable_frame

    def render(self, message: dict[str, Any] | None = None) -> list[bytes]:
        """Render a changed message and replay opaque frames at their anchors."""

        response = preserve_opaque_response_fields(
            self.message,
            message if isinstance(message, dict) else self.message,
        )
        extensions = self._extensions.for_content(
            self.message.get("content"), response.get("content")
        )
        standard = _render_known_response(response, extensions)
        if self._opaque_frames:
            by_anchor: dict[int, list[bytes]] = {}
            for known_before, raw in self._opaque_frames:
                by_anchor.setdefault(min(known_before, len(standard)), []).append(raw)
            out: list[bytes] = []
            for index in range(len(standard) + 1):
                out.extend(by_anchor.get(index, []))
                if index < len(standard):
                    out.append(standard[index])
            return out
        return standard


def _block_delta(index: int, delta: dict[str, Any], extensions: _BlockExtensions) -> bytes:
    event_extensions, delta_extensions = extensions.deltas.get(delta["type"], ({}, {}))
    return _sse_event(
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {**delta, **copy.deepcopy(delta_extensions)},
            **copy.deepcopy(event_extensions),
        }
    )


def _render_known_response(
    response: dict[str, Any], extensions: _FrameExtensions | None = None
) -> list[bytes]:
    ext = extensions if extensions is not None else _FrameExtensions()
    message = copy.deepcopy(response)
    content = message.pop("content", [])
    raw_usage = message.get("usage")
    usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
    msg_start_message = copy.deepcopy(message)
    msg_start_message.setdefault("type", "message")
    msg_start_message.setdefault("role", "assistant")
    msg_start_message.setdefault("content", [])
    msg_start_message.setdefault("stop_reason", None)
    msg_start_message["usage"] = usage
    events = [
        _sse_event(
            {
                "type": "message_start",
                "message": msg_start_message,
                **copy.deepcopy(ext.message_start),
            }
        )
    ]

    if isinstance(content, list):
        for index, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            block_ext = ext.blocks.get(index, _BlockExtensions())
            block_type = block.get("type")
            start_block = copy.deepcopy(block)
            if block_type == "text":
                start_block["text"] = ""
                start_block.pop("citations", None)
            elif block_type == "tool_use":
                start_block["input"] = {}
            elif block_type == "thinking":
                start_block["thinking"] = ""
            # Any other block, including ``server_tool_use``, is emitted
            # complete in its start frame: its input is not streamed.
            events.append(
                _sse_event(
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": start_block,
                        **copy.deepcopy(block_ext.start),
                    }
                )
            )
            if block_type == "text" and block.get("text"):
                events.append(
                    _block_delta(index, {"type": "text_delta", "text": block["text"]}, block_ext)
                )
                for citation in block.get("citations", []) or []:
                    events.append(
                        _block_delta(
                            index, {"type": "citations_delta", "citation": citation}, block_ext
                        )
                    )
            elif block_type == "tool_use" and "input" in block:
                partial_json = json.dumps(block.get("input") or {}, ensure_ascii=False)
                events.append(
                    _block_delta(
                        index, {"type": "input_json_delta", "partial_json": partial_json}, block_ext
                    )
                )
            elif block_type == "thinking":
                if block.get("thinking"):
                    events.append(
                        _block_delta(
                            index,
                            {"type": "thinking_delta", "thinking": block["thinking"]},
                            block_ext,
                        )
                    )
                if block.get("signature"):
                    events.append(
                        _block_delta(
                            index,
                            {"type": "signature_delta", "signature": block["signature"]},
                            block_ext,
                        )
                    )
            events.append(
                _sse_event(
                    {
                        "type": "content_block_stop",
                        "index": index,
                        **copy.deepcopy(block_ext.stop),
                    }
                )
            )

    delta_usage = {"output_tokens": usage.get("output_tokens", 0)}
    for key in ext.message_delta_usage_keys:
        if key in usage:
            delta_usage[key] = copy.deepcopy(usage[key])
    delta: dict[str, Any] = {"type": "message_delta", "delta": {}, "usage": delta_usage}
    for key in _MESSAGE_DELTA_FIELDS:
        if key in response:
            delta["delta"][key] = response[key]
    delta["delta"].update(copy.deepcopy(ext.message_delta_delta))
    delta.update(copy.deepcopy(ext.message_delta))
    events.append(_sse_event(delta))
    events.append(_sse_event({"type": "message_stop", **copy.deepcopy(ext.message_stop)}))
    return events


def render_anthropic_sse_response(response: dict[str, Any]) -> list[bytes]:
    """Render an Anthropic response dictionary as canonical SSE frames.

    Args:
        response: Anthropic Messages API response dictionary.

    Returns:
        Canonically encoded Anthropic SSE frames.
    """

    return _render_known_response(response)
