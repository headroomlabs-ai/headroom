"""Pull-driven, bounded HTTP entity observation without native reserialization."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, replace
from typing import Any

from headroom.proxy.gateway.config import LimitsConfig
from headroom.proxy.gateway.errors import GatewayPublicError
from headroom.proxy.gateway.usage import UsageObservation, normalize_usage


def stream_error(reason: str) -> GatewayPublicError:
    return GatewayPublicError(
        status_code=502, code="gateway_stream_" + reason, message="Upstream stream failed"
    )


class SSEFrames:
    """One bounded partial frame; output retains line endings and field bytes."""

    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self.pending = bytearray()

    def feed(self, chunk: bytes) -> Iterator[bytes]:
        start = 0
        while start < len(chunk):
            newline = chunk.find(b"\n", start)
            end = len(chunk) if newline < 0 else newline + 1
            if len(self.pending) + end - start > self.maximum:
                raise stream_error("too_large")
            self.pending.extend(chunk[start:end])
            start = end
            if self.pending.endswith((b"\n\n", b"\n\r\n")) or self.pending in (b"\n", b"\r\n"):
                frame = bytes(self.pending)
                self.pending.clear()
                yield frame


@dataclass(frozen=True, slots=True)
class SSEEvent:
    name: str
    data: str


def parse_event(frame: bytes) -> SSEEvent:
    try:
        lines = frame.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        raise stream_error("malformed") from None
    names = [
        line.partition(":")[2].removeprefix(" ") for line in lines if line.startswith("event:")
    ]
    return SSEEvent(
        names[-1] if names else "",
        "\n".join(line[5:].removeprefix(" ") for line in lines if line.startswith("data:")),
    )


def event_data(frame: bytes) -> str:
    event = parse_event(frame)
    if event.name == "error":
        raise stream_error("upstream_error")
    return event.data


def responses_refusal(payload: dict[str, Any]) -> bool:
    """Inspect documented output roles, never arbitrary nested user text."""
    output = payload.get("output", [])
    if not isinstance(output, list):
        return False
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "refusal":
            return True
        content = item.get("content", [])
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") == "refusal" for part in content
        ):
            return True
    return False


class StreamObserver:
    def __init__(
        self,
        protocol: str,
        limits: LimitsConfig,
        deadline: float,
        *,
        expected_candidates: int | None = None,
    ) -> None:
        if expected_candidates is not None and not 1 <= expected_candidates <= 8:
            raise ValueError("unsupported candidate count")
        self.protocol, self.limits, self.deadline = protocol, limits, deadline
        self.frames = SSEFrames(limits.max_frame_bytes)
        self.usage = UsageObservation()
        self.terminal: str | None = None
        self.last_event: dict[str, Any] | None = None
        self._choices: dict[int, bool] = dict.fromkeys(range(expected_candidates or 0), False)
        self._expected_candidates = expected_candidates
        self._usage_final = False
        self._chat_final_usage = False
        self._refused = False
        self._content_at = time.monotonic()
        self._partial_at: float | None = None

    def _apply_finality(self) -> None:
        complete = all(
            value is not None
            for value in (
                self.usage.input_tokens,
                self.usage.output_tokens,
                self.usage.cache_read_tokens,
                self.usage.cache_create_tokens,
            )
        )
        if self.usage.availability != "unknown":
            self.usage = replace(
                self.usage, availability="complete" if self._usage_final and complete else "partial"
            )

    def inspect(self, frame: bytes) -> bool:
        self.last_event = None
        parsed = parse_event(frame)
        data = parsed.data
        if not data:
            if parsed.name == "error":
                self.terminal = "failed"
                raise stream_error("upstream_error")
            return False
        if data == "[DONE]":
            if (
                self.protocol != "openai-chat"
                or not self._choices
                or not all(self._choices.values())
            ):
                raise stream_error("truncated")
            self.terminal = "success"
            self._usage_final = self._chat_final_usage
            self._apply_finality()
            return True
        try:
            event = json.loads(data)
        except (ValueError, RecursionError):
            raise stream_error("malformed") from None
        if not isinstance(event, dict):
            raise stream_error("malformed")
        self.last_event = event
        kind = event.get("type")
        failure = bool(
            parsed.name == "error"
            or event.get("error")
            or kind
            in {
                "error",
                "response.failed",
                "response.error",
                "response.incomplete",
                "response.cancelled",
                "refusal",
            }
        )
        direct = normalize_usage(self.protocol, event)
        self.usage = normalize_usage(self.protocol, event, previous=self.usage)
        if direct.availability != "unknown":
            # Complete fields in an intermediate snapshot are still lower bounds.
            self._usage_final = failure and direct.availability == "complete"
        self._apply_finality()
        if failure:
            self.terminal = "failed"
            raise stream_error("upstream_error")
        if kind in {"ping", "heartbeat"}:
            return False
        if self.protocol == "openai-chat":
            choices = event.get("choices", [])
            if not isinstance(choices, list):
                raise stream_error("malformed")
            for choice in choices:
                if not isinstance(choice, dict) or type(choice.get("index", 0)) is not int:
                    raise stream_error("malformed")
                index = choice.get("index", 0)
                if not 0 <= index < 128:
                    raise stream_error("malformed")
                if self._choices.get(index):
                    self.terminal = "failed"
                    raise stream_error("malformed")
                self._choices[index] = choice.get("finish_reason") is not None
                if choice.get("finish_reason") == "content_filter" or choice.get("delta", {}).get(
                    "refusal"
                ):
                    self.terminal = "failed"
                    raise stream_error("upstream_error")
            if direct.availability != "unknown":
                self._chat_final_usage = (
                    direct.availability == "complete"
                    and bool(self._choices)
                    and all(self._choices.values())
                )
        elif self.protocol == "openai-responses":
            part = event.get("part")
            item = event.get("item")
            response = event.get("response")
            if (
                kind in {"response.refusal.delta", "response.refusal.done"}
                or kind in {"response.content_part.added", "response.content_part.done"}
                and isinstance(part, dict)
                and part.get("type") == "refusal"
                or kind in {"response.output_item.added", "response.output_item.done"}
                and isinstance(item, dict)
                and responses_refusal({"output": [item]})
                or isinstance(response, dict)
                and responses_refusal(response)
            ):
                self._refused = True
            if kind == "response.completed":
                self._usage_final = direct.availability == "complete"
                self._apply_finality()
                response = event.get("response", {})
                if self._refused or (isinstance(response, dict) and responses_refusal(response)):
                    self.terminal = "failed"
                    raise stream_error("upstream_error")
                self.terminal = "success"
        elif self.protocol == "anthropic-messages":
            delta = event.get("delta", {})
            if not isinstance(delta, dict):
                raise stream_error("malformed")
            if (
                kind == "message_delta"
                and delta.get("stop_reason") is not None
                and direct.output_tokens is not None
            ):
                self._usage_final = True
                self._apply_finality()
            if delta.get("stop_reason") in {
                "refusal",
                "model_context_window_exceeded",
            }:
                self.terminal = "failed"
                raise stream_error("upstream_error")
            if kind == "message_stop":
                self.terminal = "success"
        elif self.protocol in {"gemini-generate", "vertex-generate"}:
            candidates = event.get("candidates", [])
            if not isinstance(candidates, list):
                raise stream_error("malformed")
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    raise stream_error("malformed")
                index = candidate.get("index", 0)
                if type(index) is not int or not 0 <= index < (self._expected_candidates or 8):
                    raise stream_error("malformed")
                finish = candidate.get("finishReason")
                if finish is not None and finish not in {"STOP", "MAX_TOKENS"}:
                    self.terminal = "failed"
                    raise stream_error("upstream_error")
                self._choices[index] = finish in {"STOP", "MAX_TOKENS"} or self._choices.get(
                    index, False
                )
            # A permitted usage-only tail must be consumed before closing.
            if self._choices and all(self._choices.values()) and direct.availability == "complete":
                self._usage_final = True
                self._apply_finality()
                self.terminal = "success"
        return True

    def _check_absolute_deadline(self) -> None:
        if time.monotonic() >= self.deadline:
            self.terminal = "failed"
            raise stream_error("timeout")

    async def observe(self, chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        iterator = chunks.__aiter__()
        first_read = True
        try:
            while True:
                now = time.monotonic()
                if first_read:
                    self._content_at = now
                    first_read = False
                due = min(self.deadline, self._content_at + self.limits.stream_content_idle_seconds)
                if self._partial_at is not None:
                    due = min(due, self._partial_at + self.limits.partial_frame_seconds)
                try:
                    chunk = await asyncio.wait_for(anext(iterator), max(0, due - now))
                except StopAsyncIteration:
                    break
                except TimeoutError:
                    raise stream_error("timeout") from None
                had_partial = bool(self.frames.pending)
                for frame in self.frames.feed(chunk):
                    self._check_absolute_deadline()
                    self._partial_at = None
                    if self.inspect(frame):
                        self._content_at = time.monotonic()
                    # Observe refusal streams to retain final usage, but never
                    # expose refusal-bearing frames or later content to clients.
                    if not self._refused:
                        yield frame
                    self._check_absolute_deadline()
                    if self.terminal == "success":
                        return
                if self.frames.pending and (not had_partial or self._partial_at is None):
                    self._partial_at = time.monotonic()
            self._check_absolute_deadline()
            if self._refused:
                self.terminal = "failed"
                raise stream_error("upstream_error")
            if (
                not self.frames.pending
                and self.protocol in {"gemini-generate", "vertex-generate"}
                and self._choices
                and all(self._choices.values())
            ):
                self.terminal = "success"
            if self.frames.pending or self.terminal is None:
                raise stream_error("truncated")
        finally:
            await iterator.aclose()  # type: ignore[attr-defined]


async def observed_body(chunks: AsyncIterator[bytes], *, maximum: int, deadline: float) -> bytes:
    body = bytearray()
    iterator = chunks.__aiter__()
    try:
        while True:
            try:
                chunk = await asyncio.wait_for(anext(iterator), max(0, deadline - time.monotonic()))
            except StopAsyncIteration:
                return bytes(body)
            except TimeoutError:
                raise stream_error("timeout") from None
            if len(body) + len(chunk) > maximum:
                raise stream_error("too_large")
            body.extend(chunk)
    finally:
        await iterator.aclose()  # type: ignore[attr-defined]
