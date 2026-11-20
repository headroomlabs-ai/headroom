"""Typed provider-neutral content used only at qualified translation boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class ContentBlock:
    kind: Literal["text", "image"]
    text: str | None = None
    media_type: str | None = None
    data: str | None = None


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str | None
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str
    content: tuple[ContentBlock, ...]


@dataclass(frozen=True, slots=True)
class StructuredOutput:
    name: str
    schema: dict[str, Any]
    strict: bool


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["user", "assistant"]
    content: tuple[ContentBlock, ...]
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()


@dataclass(frozen=True, slots=True)
class Conversation:
    model: str | None
    system: tuple[ContentBlock, ...]
    messages: tuple[Message, ...]
    tools: tuple[ToolDefinition, ...] = ()
    structured_output: StructuredOutput | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False
