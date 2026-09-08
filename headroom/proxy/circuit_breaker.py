"""Tool-call repetition loop cycle detector and proxy circuit breaker.

Protects wrapped autonomous agent CLIs from runaway tool-call repetition loops
caused by autoregressive attractor states in models lacking harness-level turn limits.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from collections import deque
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException

logger = logging.getLogger("headroom.proxy.circuit_breaker")

CIRCUIT_BREAKER_ENV = "HEADROOM_CIRCUIT_BREAKER"
DEFAULT_CIRCUIT_BREAKER_MODE = "warn"
VALID_CIRCUIT_BREAKER_MODES = frozenset({"enforce", "off", "warn"})
MAX_HISTORY_PER_SESSION = 16
DEFAULT_MIN_REPETITIONS = 3
DEFAULT_MAX_PERIOD = 4
MAX_TRACKED_SESSIONS = 1000


def resolve_circuit_breaker_mode(raw: str | None) -> str:
    """Resolve circuit breaker mode from raw CLI flag or environment variable."""
    if not raw:
        return DEFAULT_CIRCUIT_BREAKER_MODE
    val = raw.strip().lower()
    if val in ("enforce", "block", "true", "1"):
        return "enforce"
    if val in ("off", "disabled", "false", "0"):
        return "off"
    if val in ("warn", "warning", "log"):
        return "warn"
    raise ValueError(
        f"Invalid circuit breaker mode: {raw!r}. Valid modes: {sorted(VALID_CIRCUIT_BREAKER_MODES)}"
    )


def hash_tool_call(name: str, arguments: Any) -> tuple[str, str]:
    """Compute a canonical (tool_name, sha256_short_hash) tuple for a tool call.

    Normalizes dictionary keys and JSON string whitespace so equivalent invocations
    produce identical hashes.
    """
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            arg_bytes = json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode("utf-8")
        except Exception:
            arg_bytes = arguments.strip().encode("utf-8")
    elif isinstance(arguments, dict):
        try:
            arg_bytes = json.dumps(arguments, sort_keys=True, separators=(",", ":")).encode("utf-8")
        except Exception:
            arg_bytes = str(arguments).encode("utf-8")
    elif arguments is None:
        arg_bytes = b""
    else:
        arg_bytes = str(arguments).encode("utf-8")

    digest = hashlib.sha256(arg_bytes).hexdigest()[:16]
    return (name, digest)


@dataclass(frozen=True)
class ToolLoopDetection:
    """Result of a detected tool-call repetition loop."""

    tool_name: str
    period: int
    count: int
    pattern: list[tuple[str, str]]


def detect_repetition_cycle(
    history: Sequence[tuple[str, str]],
    min_repetitions: int = DEFAULT_MIN_REPETITIONS,
    max_period: int = DEFAULT_MAX_PERIOD,
) -> ToolLoopDetection | None:
    """Detect if the trailing sequence of tool calls repeats a period-1..max_period cycle.

    Periods are checked in ascending order (1, 2, 3, 4) to ensure the fundamental minimal
    period is selected.
    """
    items = list(history)
    total = len(items)
    for p in range(1, max_period + 1):
        if total < p * min_repetitions:
            continue

        pattern = items[-p:]
        k = 0
        while total - (k + 1) * p >= 0:
            block = items[total - (k + 1) * p : total - k * p]
            if block == pattern:
                k += 1
            else:
                break

        if k >= min_repetitions:
            return ToolLoopDetection(
                tool_name=pattern[0][0],
                period=p,
                count=k,
                pattern=pattern,
            )

    return None


def extract_tool_calls_from_messages(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Extract tool calls from OpenAI or Anthropic shaped message arrays."""
    calls: list[tuple[str, str]] = []
    if not isinstance(messages, list):
        return calls

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role != "assistant":
            continue

        # OpenAI tool_calls: [{"function": {"name": "...", "arguments": "..."}}]
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function")
                    if isinstance(fn, dict) and "name" in fn:
                        calls.append(hash_tool_call(str(fn["name"]), fn.get("arguments")))

        # OpenAI legacy function_call: {"name": "...", "arguments": "..."}
        func_call = msg.get("function_call")
        if isinstance(func_call, dict) and "name" in func_call:
            calls.append(hash_tool_call(str(func_call["name"]), func_call.get("arguments")))

        # Anthropic tool_use content blocks: [{"type": "tool_use", "name": "...", "input": {...}}]
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use" and "name" in block:
                    calls.append(hash_tool_call(str(block["name"]), block.get("input")))

    return calls


class ToolLoopCircuitBreaker:
    """Thread-safe rolling cycle detector per conversation session."""

    def __init__(
        self,
        max_history: int = MAX_HISTORY_PER_SESSION,
        min_repetitions: int = DEFAULT_MIN_REPETITIONS,
        max_period: int = DEFAULT_MAX_PERIOD,
    ) -> None:
        self.max_history = max_history
        self.min_repetitions = min_repetitions
        self.max_period = max_period
        self._sessions: dict[str, deque[tuple[str, str]]] = {}
        self._lock = threading.Lock()

    def record_tool_call(
        self,
        session_id: str,
        name: str,
        arguments: Any,
    ) -> ToolLoopDetection | None:
        """Record one tool call and return a detection if a loop is formed."""
        sig = hash_tool_call(name, arguments)
        with self._lock:
            q = self._get_or_create_queue(session_id)
            q.append(sig)
            return detect_repetition_cycle(q, self.min_repetitions, self.max_period)

    def check_session(self, session_id: str) -> ToolLoopDetection | None:
        """Check if the session history currently contains a repetition loop."""
        with self._lock:
            q = self._sessions.get(session_id)
            if not q:
                return None
            return detect_repetition_cycle(q, self.min_repetitions, self.max_period)

    def check_messages(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> ToolLoopDetection | None:
        """Update session state from request messages and check for repetition loops."""
        extracted = extract_tool_calls_from_messages(messages)
        with self._lock:
            q = self._get_or_create_queue(session_id)
            if extracted:
                # Replace with the latest window from messages
                q.clear()
                q.extend(extracted[-self.max_history :])
            if not q:
                return None
            return detect_repetition_cycle(q, self.min_repetitions, self.max_period)

    def clear_session(self, session_id: str) -> None:
        """Clear state for a session."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def _get_or_create_queue(self, session_id: str) -> deque[tuple[str, str]]:
        if session_id in self._sessions:
            return self._sessions[session_id]
        if len(self._sessions) >= MAX_TRACKED_SESSIONS:
            # Simple FIFO eviction of oldest tracked session
            oldest = next(iter(self._sessions))
            del self._sessions[oldest]
        q: deque[tuple[str, str]] = deque(maxlen=self.max_history)
        self._sessions[session_id] = q
        return q


async def execute_circuit_breaker_policy(
    circuit_breaker: ToolLoopCircuitBreaker,
    mode: str,
    session_id: str,
    messages: Sequence[Mapping[str, Any]] | None,
    request_id: str = "",
    metrics: Any = None,
    on_enforce_abort: Callable[[], Coroutine[Any, Any, None]] | None = None,
) -> ToolLoopDetection | None:
    """Evaluate messages against circuit breaker according to configured mode.

    Modes:
      - "off": Bypass detection entirely.
      - "warn": Log warning and record metrics counter, allow request to proceed.
      - "enforce": Log warning, record metrics counter, and raise HTTPException(429).
    """
    mode = resolve_circuit_breaker_mode(mode)
    if mode == "off":
        return None

    if not messages:
        return None

    detection = circuit_breaker.check_messages(session_id, list(messages))
    if detection is None:
        return None

    logger.warning(
        f"[{request_id}] Tool repetition loop detected in session {session_id}: "
        f"tool {detection.tool_name!r} period {detection.period} repeated {detection.count} times"
    )

    if metrics is not None and hasattr(metrics, "record_tool_loop_detected"):
        res = metrics.record_tool_loop_detected(tool=detection.tool_name, period=detection.period)
        if asyncio.iscoroutine(res):
            await res

    if mode == "enforce":
        if on_enforce_abort is not None:
            await on_enforce_abort()

        if detection.period == 1:
            detail = (
                f"Tool call loop detected: tool '{detection.tool_name}' executed with "
                f"identical arguments {detection.count} consecutive times. Break the cycle "
                f"by varying the probe or synthesizing existing findings."
            )
        else:
            detail = (
                f"Tool call loop detected: repeating sequence of {detection.period} tool "
                f"calls executed {detection.count} consecutive times. Break the cycle by "
                f"varying the probe or synthesizing existing findings."
            )
        raise HTTPException(status_code=429, detail=detail)

    return detection
