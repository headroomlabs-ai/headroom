"""Pure output-effort policy decisions.

The output shaper mutates provider request bodies. This module owns the
provider-neutral decisions behind those mutations so rank comparisons and
legacy budget clamping stay testable without request dictionaries.
"""

from __future__ import annotations

EFFORT_RANK = {"low": 0, "medium": 1, "high": 2, "xhigh": 3, "max": 4}
TEXT_VERBOSITY_RANK = {"low": 0, "medium": 1, "high": 2}
LEGACY_THINKING_FLOOR = 1024


def lower_effort_value(current: object, target: str) -> str | None:
    """Return ``target`` when an existing effort should be lowered."""
    if not isinstance(current, str):
        return None
    if current not in EFFORT_RANK or target not in EFFORT_RANK:
        return None
    if EFFORT_RANK[current] <= EFFORT_RANK[target]:
        return None
    return target


def clamp_legacy_thinking_budget(
    *,
    thinking_type: object,
    budget_tokens: object,
    floor: int = LEGACY_THINKING_FLOOR,
) -> int | None:
    """Return the clamped budget for legacy enabled thinking, else ``None``."""
    if thinking_type != "enabled":
        return None
    if not isinstance(budget_tokens, int):
        return None
    if budget_tokens <= floor:
        return None
    return floor


def can_create_openai_text_verbosity(model: object) -> bool:
    """Whether it is safe to create a new OpenAI ``text.verbosity`` block."""
    return str(model or "").lower().startswith("gpt-5")


def lower_text_verbosity_value(current: object) -> str | None:
    """Return ``low`` when an existing OpenAI text verbosity should be lowered."""
    if not isinstance(current, str):
        return None
    if current not in TEXT_VERBOSITY_RANK:
        return None
    if TEXT_VERBOSITY_RANK[current] <= TEXT_VERBOSITY_RANK["low"]:
        return None
    return "low"
