"""Mechanism B: hold-back Read maturation — compress a Read once it is done.

A fresh Read is held verbatim while its file is active, so the model sees
the full content the whole time it is working with the file. Once the
file has been quiet for `quiesce_turns`, the content is replaced with a
CCR-backed marker, if the Read is still outside the provider-cached
(frozen) prefix.

The held Read is cached like any other message: the client's cache
breakpoints are forwarded where the client put them. The prefix cache
bills everything *after* the first changed byte, so a Read the frozen
prefix already covers stays verbatim: maturing it would re-write the
whole tail after it at the cache-write rate to save a cache read of the
Read. An earlier version kept the verbatim Read out of the cache instead,
by moving the trailing breakpoint in front of it for the whole hold.
Claude Code marks the last two messages, so that dropped its tail
breakpoint, and every held turn re-sent the Read and everything after it
at the full input rate (one 13-request hold grew its uncached tail from
~3.9K to ~14.6K tokens per request) to save a single cache write.

Timeline for a Read of file F (quiesce_turns=5):

    turn T:      model reads F — verbatim, cached with the turn
    T+1..T+k:    model edits / re-reads F — read stays verbatim (every
                 touch resets the quiet clock)
    T+k+5:       F has been quiet 5 turns → read matures into a marker
                 if it is still outside the frozen prefix, else it stays
                 verbatim for good
    later turns: the forwarded form is read from cache

Why activity-based instead of a fixed hold window: the audit-reads
simulation over real traffic showed touch gaps are fat-tailed (next-touch
p50 = 4 turns, p90 = 81) — no fixed window covers the tail. The quiesce
rule covers the activity cluster, `max_hold_turns` bounds the hold cost
for pathologically busy files, and the tail self-heals through the
model's *observed* habit of re-reading ranges from disk: 95% of re-reads
in real traffic are partial-range reads made while the full text was
still in context. The recovery path is the model's existing behavior.

Two invariants:

1. **The frozen prefix is never mutated.** A Read only matures while it
   is outside the provider-confirmed prefix.
2. **Replay is deterministic.** Once matured, the same marker is applied
   on every subsequent request (state is session-scoped), so the cached
   prefix stays byte-stable for the rest of the session.

Recovery contract (same as read_lifecycle): the full original is stored
in the CCR compression store, the marker carries the retrieval hash and
the file path, and the file itself remains on disk — a confused model
re-reads at the cost of one tool call.

State (matured markers only — holding is derived from the conversation
itself, so it survives state loss) lives with the session's prefix
tracker. Per-process, like all session state: multi-worker deployments
need sticky sessions (existing constraint).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from ..config import ReadMaturationConfig

logger = logging.getLogger(__name__)

# Tool names whose results are eligible for maturation.
_READ_TOOLS = frozenset({"Read", "read"})
# Tool names that count as file activity (reset the quiet clock).
_TOUCH_TOOLS = frozenset(
    {"Read", "read", "Edit", "edit", "Write", "write", "MultiEdit", "NotebookEdit"}
)


@dataclass
class MaturedRead:
    """Replayed replacement for a matured Read."""

    marker: str
    ccr_hash: str


@dataclass
class _Activity:
    """Per-request scan of tool activity, in assistant-turn units."""

    # tool_use_id -> (file_path, assistant turn of the Read tool_use)
    read_calls: dict[str, tuple[str, int]] = field(default_factory=dict)
    # file_path -> assistant turn of its most recent touch (read or edit)
    file_last_touch: dict[str, int] = field(default_factory=dict)
    # Total assistant messages in the conversation ("now").
    assistant_count: int = 0


@dataclass
class MaturationResult:
    """Output of one per-request maturation pass."""

    messages: list[dict[str, Any]]
    # Message indices that contain still-holding Reads (forwarded verbatim
    # this request).
    holding_msg_indices: list[int] = field(default_factory=list)
    holding_reads: int = 0
    newly_matured: int = 0
    replacements_applied: int = 0
    bytes_saved: int = 0


def _iter_tool_results(messages: list[dict[str, Any]]) -> Iterator[tuple[str, str]]:
    """(tool_call_id, content) for every string tool result, both formats."""
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if msg.get("role") == "tool":
            if isinstance(content, str):
                yield str(msg.get("tool_call_id", "")), content
            continue
        if isinstance(content, list):
            for b in content:
                if (
                    isinstance(b, dict)
                    and b.get("type") == "tool_result"
                    and isinstance(b.get("content"), str)
                ):
                    yield str(b.get("tool_use_id", "")), b["content"]


class ReadMaturationManager:
    """Per-session Read maturation state machine.

    Construct once per session (or hold in a session-scoped container)
    and call :meth:`apply` on every request, after read_lifecycle and
    before breakpoint placement.
    """

    def __init__(
        self,
        config: ReadMaturationConfig,
        compression_store: Any | None = None,
    ):
        self.config = config
        self.store = compression_store
        self._matured: dict[str, MaturedRead] = {}
        # tool_call_ids whose removal has already been booked as savings.
        self._booked: set[str] = set()
        # tool_call_id -> token delta of its replayed marker, tokenized
        # once per session (content and marker are stable per tool call).
        self._replay_token_deltas: dict[str, int] = {}

    # ─── Per-request entry point ────────────────────────────────────────

    def apply(
        self,
        messages: list[dict[str, Any]],
        frozen_message_count: int = 0,
    ) -> MaturationResult:
        """Hold active Reads, mature quiet ones, replay matured markers.

        Args:
            messages: Conversation messages (Anthropic content-block or
                OpenAI role="tool" formats).
            frozen_message_count: Provider-cached message count. Reads
                inside the frozen prefix are already cache-written
                verbatim (held Reads are cached with their turn) — they
                are never touched.
        """
        result = MaturationResult(messages=messages)
        if not self.config.enabled:
            return result

        activity = self._scan_activity(messages)
        out: list[dict[str, Any]] = []
        any_changed = False

        for i, msg in enumerate(messages):
            if i < frozen_message_count:
                out.append(msg)
                continue
            new_msg, msg_holding = self._process_message(msg, activity, result)
            out.append(new_msg)
            if new_msg is not msg:
                any_changed = True
            if msg_holding:
                result.holding_msg_indices.append(i)

        if any_changed:
            result.messages = out
        return result

    def replayed_token_debt(
        self,
        original_messages: list[dict[str, Any]],
        outbound_messages: list[dict[str, Any]],
        count_text: Callable[[str], int],
    ) -> int:
        """Tokens this request re-saved by re-removing content whose removal
        was already booked on an EARLIER request.

        The client re-sends the raw conversation every turn, so a plain
        original-vs-optimized token diff books a matured Read's removal
        again on every request until end of session; one long session
        inflated its new-content savings rate from 31.8% to 78.15% the
        day maturation turned on, with no new removal behind the jump.
        Subtracting this debt makes the figure first-appearance: matured
        content books exactly once, on the turn it matures.

        Measured on the request's own endpoints (the raw client snapshot
        vs what is actually forwarded) rather than on this pass's
        replacements, because after a Read matures its marker usually
        reaches the wire through the cached-prefix replay instead of
        through :meth:`apply` — the replacement this manager makes is
        only one of the paths that re-remove it.

        Call this ONCE per request, on the final outbound messages:
        the first booking is recorded as a side effect, so a second call
        would charge the request for its own first appearance.
        ``count_text`` should be the tokenizer the caller diffs with, so
        the subtraction lands on the booked scale. Deltas are tokenized
        once per tool call and cached for the session.
        """
        if not self._matured:
            return 0
        forwarded = dict(_iter_tool_results(outbound_messages))
        debt = 0
        for tc_id, content in _iter_tool_results(original_messages):
            matured = self._matured.get(tc_id)
            if matured is None or forwarded.get(tc_id) != matured.marker:
                continue  # not replaced on the wire this request
            if content == matured.marker:
                continue  # the client already held the marker: nothing removed
            if tc_id not in self._booked:
                self._booked.add(tc_id)  # first appearance: books in full
                continue
            delta = self._replay_token_deltas.get(tc_id)
            if delta is None:
                delta = max(0, count_text(content) - count_text(matured.marker))
                self._replay_token_deltas[tc_id] = delta
            debt += delta
        return debt

    # ─── Internals ──────────────────────────────────────────────────────

    def _scan_activity(self, messages: list[dict[str, Any]]) -> _Activity:
        """One pass over assistant messages: read calls, per-file last
        touch, and the current assistant-turn count."""
        act = _Activity()
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            act.assistant_count += 1
            turn = act.assistant_count

            for tc in msg.get("tool_calls", []) or []:
                if not isinstance(tc, dict):
                    continue
                func = tc.get("function", {})
                name = func.get("name", "")
                if name not in _TOUCH_TOOLS:
                    continue
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except (ValueError, TypeError):
                    args = {}
                fp = args.get("file_path") or args.get("path") or ""
                if fp:
                    act.file_last_touch[fp] = turn
                if name in _READ_TOOLS:
                    act.read_calls[tc.get("id", "")] = (fp, turn)

            content = msg.get("content")
            if isinstance(content, list):
                for b in content:
                    if not (isinstance(b, dict) and b.get("type") == "tool_use"):
                        continue
                    name = b.get("name", "")
                    if name not in _TOUCH_TOOLS:
                        continue
                    inp = b.get("input") or {}
                    fp = inp.get("file_path") or inp.get("path") or ""
                    if fp:
                        act.file_last_touch[fp] = turn
                    if name in _READ_TOOLS:
                        act.read_calls[b.get("id", "")] = (fp, turn)
        return act

    def _process_message(
        self,
        msg: dict[str, Any],
        activity: _Activity,
        result: MaturationResult,
    ) -> tuple[dict[str, Any], bool]:
        """Returns (possibly-replaced message, message_still_holding)."""
        role = msg.get("role", "")
        content = msg.get("content", "")

        # OpenAI format: whole message is one tool result.
        if role == "tool":
            tc_id = msg.get("tool_call_id", "")
            if tc_id in activity.read_calls and isinstance(content, str):
                new_content, holding = self._handle_read(tc_id, content, activity, result)
                if new_content is not None:
                    return {**msg, "content": new_content}, holding
                return msg, holding
            return msg, False

        # Anthropic format: tool_result blocks inside a user message.
        # NOTE: blocks carrying a client cache_control are NOT skipped —
        # Claude Code parks its tail breakpoint on the newest content
        # block, which right after a Read is the Read's tool_result
        # itself. The breakpoint stays where the client put it; a
        # replaced block keeps its cache_control.
        if isinstance(content, list):
            new_blocks: list[Any] = []
            changed = False
            holding_any = False
            for b in content:
                if (
                    isinstance(b, dict)
                    and b.get("type") == "tool_result"
                    and b.get("tool_use_id", "") in activity.read_calls
                    and isinstance(b.get("content"), str)
                ):
                    tc_id = b["tool_use_id"]
                    new_content, holding = self._handle_read(tc_id, b["content"], activity, result)
                    holding_any = holding_any or holding
                    if new_content is not None:
                        new_blocks.append({**b, "content": new_content})
                        changed = True
                        continue
                new_blocks.append(b)
            if changed:
                return {**msg, "content": new_blocks}, holding_any
            return msg, holding_any

        return msg, False

    def _handle_read(
        self,
        tc_id: str,
        content: str,
        activity: _Activity,
        result: MaturationResult,
    ) -> tuple[str | None, bool]:
        """Returns (replacement_content | None, still_holding)."""
        matured = self._matured.get(tc_id)

        # Matured earlier: replay the recorded marker deterministically.
        if matured is not None:
            if content == matured.marker:
                return None, False
            result.replacements_applied += 1
            result.bytes_saved += max(0, len(content) - len(matured.marker))
            return matured.marker, False

        size = len(content.encode("utf-8", errors="replace"))
        if size < self.config.min_size_bytes:
            return None, False
        # Lifecycle markers (stale/superseded) are already compact — and
        # read_lifecycle runs first, so respect its replacement.
        if "Retrieve original: hash=" in content or "Retrieve more: hash=" in content:
            return None, False

        file_path, read_turn = activity.read_calls[tc_id]
        last_touch = activity.file_last_touch.get(file_path, read_turn)
        quiet_turns = activity.assistant_count - last_touch
        held_turns = activity.assistant_count - read_turn

        if quiet_turns < self.config.quiesce_turns and held_turns < self.config.max_hold_turns:
            result.holding_reads += 1
            return None, True  # file still active — keep verbatim, uncached

        # File quiesced (or hold cap hit): mature.
        ccr_hash = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:24]
        if self.store is not None:
            try:
                ccr_hash = self.store.store(
                    original=content,
                    compressed="",
                    tool_name="Read",
                    tool_call_id=tc_id,
                    compression_strategy="read_maturation",
                )
            except Exception as e:  # noqa: BLE001 - storage failure must not break the request
                logger.warning("read_maturation: CCR store failed for %s: %s", tc_id, e)

        file_display = file_path or "unknown"
        # NOTE: "Retrieve original: hash=" is load-bearing (marker-
        # preserving regex + ContentRouter compression pinning).
        marker = (
            f"[Read of {file_display} compressed after use — re-read the file "
            f"if needed. Retrieve original: hash={ccr_hash}]"
        )
        self._matured[tc_id] = MaturedRead(marker=marker, ccr_hash=ccr_hash)
        result.newly_matured += 1
        result.replacements_applied += 1
        result.bytes_saved += max(0, len(content) - len(marker))
        return marker, False
