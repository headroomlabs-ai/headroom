"""Separate thinking tokens from visible output tokens.

``output_tokens`` is a single number today, and it pools two quantities that
different levers move in opposite directions. Reasoning-effort routing cuts
*thinking*; verbosity steering cuts *visible text*. Summed into one counter,
neither can be attributed: a request whose thinking dropped 4,000 tokens while
its prose grew 200 looks identical to one where nothing happened.

That is not a reporting nicety. It is the reason the output shaper cannot say
which of its levers works, including the ones already shipped.

What each provider actually tells us
------------------------------------
* **OpenAI chat** — ``usage.completion_tokens_details.reasoning_tokens``.
* **OpenAI Responses** — ``usage.output_tokens_details.reasoning_tokens``.
* **Gemini** — ``usageMetadata.thoughtsTokenCount``. Note this is already read
  by :func:`headroom.proxy.token_counting.gemini_output_tokens`, but only to
  fold into the output total; keeping it separate here loses nothing and gains
  the split.
* **Anthropic** — *nothing*. The Messages API reports no thinking count at all.
  The thinking text is in the response content, so it can be estimated, but an
  estimate must never be handed back as if the provider had reported it.

Unknown is not zero
-------------------
Returning ``0`` for Anthropic would assert "no thinking happened", which is
false whenever extended thinking is on — and it would quietly corrupt every
average computed over mixed-provider traffic. So the result is ``None`` for
"we cannot tell" and an ``int`` for "we know", with :attr:`ThinkingTokens.inferred`
marking a count Headroom derived rather than received.

That mirrors ``RequestOutcome.cache_inferred``, which exists for the same
reason on the input side: a derived number is useful, and pretending it was
measured is not.

Pure module: no tokenizer import at module scope and no I/O. The estimator is
injected, so callers that have a tokenizer can pass one and callers that do not
still get a correct ``None``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ThinkingTokens",
    "anthropic_thinking_text",
    "extract_from_usage",
    "extract_thinking_tokens",
]


@dataclass(frozen=True)
class ThinkingTokens:
    """A thinking-token count, and whether it was reported or derived.

    Attributes:
        tokens: The count, or ``None`` when the provider reported nothing and
            no estimate could be made. ``0`` means "the provider told us there
            was no thinking" — a genuinely different claim from ``None``.
        inferred: True when Headroom derived the number (by tokenizing the
            response's thinking blocks) rather than reading it from usage.
    """

    tokens: int | None = None
    inferred: bool = False

    @property
    def known(self) -> bool:
        return self.tokens is not None

    def visible_from(self, output_tokens: int) -> int | None:
        """Visible output tokens, or ``None`` when the split is unknown.

        Clamped at zero: an inferred count uses Headroom's tokenizer while
        ``output_tokens`` is on the provider's scale, so the two can disagree
        slightly and a naive subtraction can go negative on a short response.
        """
        if self.tokens is None:
            return None
        return max(0, output_tokens - self.tokens)


def _as_int(value: Any) -> int | None:
    """Coerce a usage field to a non-negative int, or ``None`` if absent/bad.

    Distinguishes a missing field from a zero one, which is the whole point of
    this module — so it deliberately does not use the ``_usage_int`` pattern
    elsewhere in the proxy, which floors everything to 0.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return max(parsed, 0)


def _details_reasoning(usage: dict[str, Any]) -> int | None:
    """Read ``reasoning_tokens`` from whichever details block is present.

    Chat spells the container ``completion_tokens_details``; Responses spells
    it ``output_tokens_details``. Some gateways echo both, so check each and
    take the first that carries a usable number rather than assuming a format.
    """
    for container_key in ("output_tokens_details", "completion_tokens_details"):
        container = usage.get(container_key)
        if not isinstance(container, dict):
            continue
        parsed = _as_int(container.get("reasoning_tokens"))
        if parsed is not None:
            return parsed
    return None


def anthropic_thinking_text(payload: dict[str, Any]) -> str:
    """Concatenate the text of every thinking block in an Anthropic response.

    Anthropic reports no thinking count, but it does return the thinking
    itself, so the content is the only available basis for an estimate.
    Handles both ``thinking`` and ``redacted_thinking`` blocks; a redacted
    block carries no readable text but still cost output tokens, so its
    ``data`` payload is included — an approximation, and flagged as inferred
    by the caller.
    """
    content = payload.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "thinking":
            text = block.get("thinking")
            if isinstance(text, str):
                parts.append(text)
        elif btype == "redacted_thinking":
            data = block.get("data")
            if isinstance(data, str):
                parts.append(data)
    return "\n".join(parts)


def extract_from_usage(usage: Any) -> ThinkingTokens:
    """Read the thinking count from a bare provider ``usage`` dict.

    The full-payload :func:`extract_thinking_tokens` is the general entry
    point, but several proxy call sites have already destructured ``usage``
    out of the response and never keep the body around. Giving them a direct
    path avoids forcing an artificial re-wrap — and avoids the temptation to
    pass ``{}`` and silently record a zero.
    """
    if not isinstance(usage, dict):
        return ThinkingTokens()
    reported = _details_reasoning(usage)
    if reported is not None:
        return ThinkingTokens(tokens=reported)
    thoughts = _as_int(usage.get("thoughtsTokenCount"))
    if thoughts is not None:
        return ThinkingTokens(tokens=thoughts)
    return ThinkingTokens()


def extract_thinking_tokens(
    payload: Any,
    *,
    estimator: Callable[[str], int] | None = None,
) -> ThinkingTokens:
    """Extract the thinking-token count from any provider response shape.

    Args:
        payload: A parsed provider response body.
        estimator: Optional ``text -> token count``. Supplied by callers that
            have a tokenizer; used only for Anthropic, which reports no count
            of its own. Without it, Anthropic responses return ``None``
            (unknown) rather than a fabricated zero.

    Returns:
        A :class:`ThinkingTokens`. ``tokens is None`` means the split could not
        be determined — callers must treat that as "unknown", never as zero.
    """
    if not isinstance(payload, dict):
        return ThinkingTokens()

    # Gemini: a top-level usageMetadata rather than usage.
    usage_meta = payload.get("usageMetadata")
    if isinstance(usage_meta, dict):
        return ThinkingTokens(tokens=_as_int(usage_meta.get("thoughtsTokenCount")))

    usage = payload.get("usage")
    if isinstance(usage, dict):
        reported = _details_reasoning(usage)
        if reported is not None:
            return ThinkingTokens(tokens=reported)

    # Anthropic: no usage field for this, so fall back to the content.
    text = anthropic_thinking_text(payload)
    if not text:
        # Distinguish "an Anthropic response with no thinking blocks" — which
        # is a genuine zero — from "not an Anthropic response at all", where we
        # know nothing. Only a response that actually carries content blocks
        # can support the former claim.
        if isinstance(payload.get("content"), list):
            return ThinkingTokens(tokens=0)
        return ThinkingTokens()

    if estimator is None:
        return ThinkingTokens()
    try:
        return ThinkingTokens(tokens=max(0, int(estimator(text))), inferred=True)
    except Exception:  # noqa: BLE001 — a bad estimator must not break a response
        return ThinkingTokens()
