"""Recover output-token counts from a finished SSE stream.

When an upstream omits a usage chunk, the proxy has to estimate how many output
tokens the turn produced. The estimate was ``total_bytes // 40`` over the RAW
SSE WIRE — every ``data:`` prefix, every JSON envelope, every ``role``/
``finish_reason``/``id``/``model`` field, blank-line framing included. On a
Copilot chat turn that produced a short answer the log read:

    Could not parse output_tokens from SSE, estimating 8 from 334 bytes

334 bytes of wire is mostly envelope; the generated text inside it was a
fraction of that. The divisor 40 is a fudge for "bytes per token INCLUDING
framing overhead", so its error scales with how chatty the framing is rather
than with the answer — a stream split into many small deltas is punished for
the split, and one delivered in a few large chunks is not.

The stream's own text is right there in the buffer, so extract it and count
that instead. ``bytes // 40`` survives only as the last resort for a stream
whose text could not be recovered at all.

Pure and I/O-free so the parsing is testable without a proxy or a network.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

# Bytes per token when nothing better is available. Applied to the raw wire, so
# it must absorb SSE framing overhead as well as the text — which is exactly why
# it is a poor estimator and a last resort.
WIRE_BYTES_PER_TOKEN = 40

# Characters per token for extracted text. ~4 is the usual English/code
# approximation and is applied to generated text ONLY, with no framing in it.
TEXT_CHARS_PER_TOKEN = 4


def _iter_sse_payloads(sse: str) -> Iterator[Any]:
    """Yield each ``data:`` payload in an SSE stream as a parsed object.

    Tolerates the two things real streams do that a naive split does not:
    an event whose ``data:`` is spread over multiple lines, and ``[DONE]``.
    """
    for block in sse.split("\n\n"):
        lines = [ln for ln in block.split("\n") if ln.startswith("data:")]
        if not lines:
            continue
        # Multi-line data: fields concatenate, per the SSE spec.
        raw = "".join(ln[5:].lstrip() for ln in lines).strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            yield json.loads(raw)
        except (ValueError, TypeError):
            continue


def extract_stream_text(sse: str) -> str:
    """Return the assistant text carried by a completed SSE stream.

    Handles the three surfaces this proxy forwards:

    * OpenAI chat completions — ``choices[].delta.content``
    * OpenAI responses       — ``response.output_text.delta`` / ``delta``
    * Anthropic messages     — ``content_block_delta.delta.text``

    Reasoning/thinking deltas are counted too: the provider bills them as
    output tokens, so omitting them would under-count exactly the turns where
    output is most expensive.
    """
    if not sse:
        return ""

    parts: list[str] = []
    for obj in _iter_sse_payloads(sse):
        if not isinstance(obj, dict):
            continue

        # --- OpenAI chat completions -------------------------------------- #
        choices = obj.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue
                for key in ("content", "reasoning_content", "refusal"):
                    value = delta.get(key)
                    if isinstance(value, str):
                        parts.append(value)
                # Tool-call arguments stream as text and are billed as output.
                tool_calls = delta.get("tool_calls")
                if isinstance(tool_calls, list):
                    for call in tool_calls:
                        fn = call.get("function") if isinstance(call, dict) else None
                        args = fn.get("arguments") if isinstance(fn, dict) else None
                        if isinstance(args, str):
                            parts.append(args)
            continue

        obj_type = obj.get("type")

        # --- Anthropic messages ------------------------------------------- #
        if obj_type == "content_block_delta":
            delta = obj.get("delta")
            if isinstance(delta, dict):
                for key in ("text", "thinking", "partial_json"):
                    value = delta.get(key)
                    if isinstance(value, str):
                        parts.append(value)
            continue

        # --- OpenAI responses --------------------------------------------- #
        if isinstance(obj_type, str) and obj_type.endswith(".delta"):
            value = obj.get("delta")
            if isinstance(value, str):
                parts.append(value)
            continue

    return "".join(parts)


# Terminal stop reasons that mean "the output-token ceiling cut this turn off".
# Anthropic messages say ``max_tokens``; OpenAI chat completions say ``length``.
_CEILING_STOP_REASONS = frozenset({"max_tokens", "length", "max_output_tokens"})

# Request fields that carry that ceiling, per surface. All three are denominated
# in OUTPUT tokens (reasoning included, where a model emits it), which is what
# makes the ceiling an exact count rather than a lower bound.
_CEILING_FIELDS = ("max_tokens", "max_completion_tokens", "max_output_tokens")


def _hit_output_ceiling(sse_text: str) -> bool:
    """Whether the stream ended because it hit the request's output ceiling."""
    for obj in _iter_sse_payloads(sse_text):
        if not isinstance(obj, dict):
            continue

        # OpenAI chat completions — choices[].finish_reason
        choices = obj.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if isinstance(choice, dict) and choice.get("finish_reason") in (
                    _CEILING_STOP_REASONS
                ):
                    return True
            continue

        # Anthropic messages — message_delta.delta.stop_reason
        delta = obj.get("delta")
        if isinstance(delta, dict) and delta.get("stop_reason") in _CEILING_STOP_REASONS:
            return True

        # OpenAI responses — response.incomplete_details.reason
        response = obj.get("response")
        if isinstance(response, dict):
            details = response.get("incomplete_details")
            if isinstance(details, dict) and details.get("reason") in _CEILING_STOP_REASONS:
                return True
    return False


def _requested_ceiling(body: Any) -> int | None:
    """The output-token ceiling the request asked for, or ``None``."""
    if not isinstance(body, dict):
        return None
    for field in _CEILING_FIELDS:
        value = body.get(field)
        # bool is an int subclass and would sail through as 0/1.
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def estimate_output_tokens(*, sse_text: str, total_bytes: int, body: Any = None) -> tuple[int, str]:
    """Return ``(tokens, source)`` for a stream with no provider usage chunk.

    ``source`` names which rung of the ladder produced the number so the caller
    can log it honestly rather than implying the provider reported it:

    * ``exact_ceiling``   — the turn hit its output-token ceiling (exact)
    * ``estimated_text``  — counted from the generated text (good)
    * ``estimated_bytes`` — the raw-wire fallback (poor, last resort)

    The top rung exists because counting the stream's text is catastrophically
    wrong for exactly one shape of turn: a TOOL CALL cut off by ``max_tokens``.
    The arguments stream as ``input_json_delta`` fragments, and an upstream that
    stops mid-object drops the incomplete remainder rather than emit unparseable
    JSON — while still billing every token it generated. Measured against
    Anthropic: a turn billed 1500 output tokens left ~45 characters of
    recoverable text, which the text rung reads as ~11 tokens. A 100x
    under-count, landing on the most expensive turns there are.

    Nothing needs estimating there. The ceiling is denominated in output tokens,
    so a turn that stopped *because* it hit the ceiling produced exactly that
    many. ``body`` is the provider request as sent, so the value read here is
    the ceiling the upstream actually enforced — not whatever the client
    originally asked for before the proxy's own levers touched it.
    """
    if _hit_output_ceiling(sse_text):
        ceiling = _requested_ceiling(body)
        if ceiling is not None:
            return ceiling, "exact_ceiling"

    text = extract_stream_text(sse_text)
    if text:
        # At least one token for any non-empty answer: integer division would
        # report 0 for a 1-3 character reply ("OK", "42"), and a turn that
        # produced output must never be recorded as having produced none.
        return max(1, len(text) // TEXT_CHARS_PER_TOKEN), "estimated_text"
    return max(0, total_bytes) // WIRE_BYTES_PER_TOKEN, "estimated_bytes"
