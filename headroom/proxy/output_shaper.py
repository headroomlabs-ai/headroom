"""Output token shaping for proxied Anthropic requests.

Headroom's transforms compress what goes INTO the model. This module is the
first request-side lever on what comes OUT of it. The proxy never generates
output tokens, so every lever here works by reshaping the request:

1. Verbosity steering — a deterministic instruction block appended to the
   TAIL of the system prompt (after any ``cache_control`` breakpoint, so the
   provider prefix cache is preserved). Five levels, from "no ceremony" to
   full caveman.

2. Effort routing — agentic loops are mostly mechanical continuations (the
   last message is a clean tool_result: a file read, a passing test). Thinking
   bills as output tokens, and harnesses like Claude Code pin
   ``output_config.effort`` at ``xhigh`` for every turn. On turns classified
   as mechanical we lower an explicitly-present effort; on errors or new user
   asks we leave it alone. For legacy models still sending
   ``thinking.budget_tokens`` we clamp the budget to the API floor instead.

Safety rules (each prevents a concrete failure mode):
- Never INJECT ``output_config.effort`` where the client didn't send it —
  models without effort support 400 on it. Lowering an existing value is
  always valid.
- Never toggle ``thinking.type`` — disabling thinking while history carries
  thinking blocks 400s on some models, and the toggle busts the messages
  cache tier.
- Steering text is byte-stable per level and applied idempotently, so
  repeated requests keep an identical prefix.

Turn classification is purely structural (block types, roles, ``is_error``
flags) — no content regexes or keyword patterns.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Documented Anthropic API minimum for thinking.budget_tokens on models
# that still accept the legacy enabled/budget_tokens form.
LEGACY_THINKING_FLOOR = 1024

# Ordering for output_config.effort values. Unknown values are left alone.
_EFFORT_RANK = {"low": 0, "medium": 1, "high": 2, "xhigh": 3, "max": 4}

# Sentinel prefix marks the steering block so application is idempotent and
# the block is recognizable in logs/diffs.
_STEERING_SENTINEL = "<headroom_output_shaping>"
_STEERING_SUFFIX = "</headroom_output_shaping>"

# Levels are cumulative: each includes everything above it. Text must stay
# byte-stable across releases for prefix-cache friendliness — treat edits to
# these strings as cache-busting changes.
_VERBOSITY_LEVELS = {
    1: (
        "Skip preamble and postamble. Do not announce what you are about to "
        "do or recap what you just did; start with the substance."
    ),
    2: (
        "Skip preamble and postamble; start with the substance. Never restate "
        "code, file contents, diffs, or tool output that already appear in "
        "this conversation — reference them by path and line instead. After a "
        "tool call succeeds, continue without narrating the result."
    ),
    3: (
        "Skip preamble and postamble. Never restate code, file contents, "
        "diffs, or tool output already in this conversation — reference by "
        "path and line. Give conclusions only; omit rationale unless the user "
        "asks why. Prefer the smallest edit over rewriting whole files. Keep "
        "prose to the minimum needed to be unambiguous."
    ),
    4: (
        "Minimum tokens. Fragments fine. No preamble, no postamble, no "
        "restating context, no rationale. Answer, smallest-possible edits, "
        "nothing else."
    ),
}


class TurnKind(Enum):
    """Structural classification of the latest conversation turn."""

    NEW_USER_ASK = "new_user_ask"
    MECHANICAL_CONTINUATION = "mechanical_continuation"
    ERROR_CONTINUATION = "error_continuation"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class OutputShaperSettings:
    """Runtime settings, resolved once per request from the environment.

    Env-driven (like HEADROOM_INTERCEPT_ENABLED) so the proxy picks it up
    without config plumbing through the server. Off by default.
    """

    enabled: bool = False
    verbosity_level: int = 2
    effort_router_enabled: bool = True
    mechanical_effort: str = "low"

    @classmethod
    def from_env(cls) -> OutputShaperSettings:
        enabled = os.environ.get("HEADROOM_OUTPUT_SHAPER", "").lower() in (
            "1",
            "true",
            "yes",
        )
        try:
            level = int(os.environ.get("HEADROOM_VERBOSITY_LEVEL", "2"))
        except ValueError:
            level = 2
        level = max(0, min(4, level))
        router = os.environ.get("HEADROOM_EFFORT_ROUTER", "1").lower() not in (
            "0",
            "false",
            "no",
        )
        mech = os.environ.get("HEADROOM_MECHANICAL_EFFORT", "low")
        if mech not in _EFFORT_RANK:
            mech = "low"
        return cls(
            enabled=enabled,
            verbosity_level=level,
            effort_router_enabled=router,
            mechanical_effort=mech,
        )


def resolve_verbosity_level(settings: OutputShaperSettings) -> tuple[int, str]:
    """Resolve the live verbosity level and its source.

    Precedence:
      1. ``HEADROOM_VERBOSITY_LEVEL`` set explicitly → manual override.
      2. AIMD controller state (when ``HEADROOM_VERBOSITY_AUTOTUNE`` is on).
      3. Learned ``verbosity.json`` from ``learn --verbosity``.
      4. The settings default.

    Returns ``(level, source)``. Kept separate from :func:`shape_request` so the
    body-mutating core stays a pure function of an explicit level.
    """
    import os

    if os.environ.get("HEADROOM_VERBOSITY_LEVEL"):
        return settings.verbosity_level, "env"

    try:
        from ..paths import workspace_dir

        ws = workspace_dir()
    except Exception:
        return settings.verbosity_level, "default"

    autotune = os.environ.get("HEADROOM_VERBOSITY_AUTOTUNE", "").lower() in ("1", "true", "yes")
    if autotune:
        ctrl_path = ws / "verbosity_controller.json"
        if ctrl_path.exists():
            try:
                import json as _json

                level = int(
                    _json.loads(ctrl_path.read_text()).get("level", settings.verbosity_level)
                )
                return max(0, min(4, level)), "controller"
            except (OSError, ValueError):
                pass

    prof_path = ws / "verbosity.json"
    if prof_path.exists():
        try:
            import json as _json

            level = int(_json.loads(prof_path.read_text()).get("verbosity_level", -1))
            if 0 <= level <= 4:
                return level, "learned"
        except (OSError, ValueError):
            pass

    return settings.verbosity_level, "default"


@dataclass
class ShapeResult:
    """What the shaper did to a request body."""

    changed: bool = False
    labels: list[str] | None = None

    def __post_init__(self) -> None:
        if self.labels is None:
            self.labels = []


def classify_turn(messages: list[dict[str, Any]]) -> TurnKind:
    """Classify the latest turn from message structure alone.

    - Any text block in the last user message → the user is asking something
      new: full effort.
    - Only tool_result blocks, none flagged ``is_error`` → mechanical
      continuation: the model is resuming after a routine tool call.
    - Any tool_result with ``is_error: true`` → error continuation: the model
      must reason about a failure, keep full effort.
    """
    if not messages:
        return TurnKind.UNKNOWN
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return TurnKind.UNKNOWN

    content = last.get("content")
    if isinstance(content, str):
        return TurnKind.NEW_USER_ASK if content.strip() else TurnKind.UNKNOWN
    if not isinstance(content, list) or not content:
        return TurnKind.UNKNOWN

    saw_tool_result = False
    saw_error = False
    for block in content:
        if not isinstance(block, dict):
            return TurnKind.UNKNOWN
        btype = block.get("type")
        if btype == "tool_result":
            saw_tool_result = True
            if block.get("is_error") is True:
                saw_error = True
        elif btype == "text":
            # Fresh user text alongside (or instead of) tool results means
            # the user interjected — treat as a new ask.
            return TurnKind.NEW_USER_ASK
        elif btype in ("image", "document"):
            return TurnKind.NEW_USER_ASK
        # Unknown block types are ignored rather than guessed at.

    if saw_error:
        return TurnKind.ERROR_CONTINUATION
    if saw_tool_result:
        return TurnKind.MECHANICAL_CONTINUATION
    return TurnKind.UNKNOWN


def steering_text(level: int) -> str | None:
    """The full steering block for a verbosity level, or None for level 0."""
    text = _VERBOSITY_LEVELS.get(level)
    if text is None:
        return None
    return f"{_STEERING_SENTINEL}\n{text}\n{_STEERING_SUFFIX}"


def apply_verbosity_steering(body: dict[str, Any], level: int) -> bool:
    """Append the steering block to the tail of the system prompt.

    Appending AFTER the last system block keeps any ``cache_control``
    breakpoint on an earlier block intact — the cached prefix is unchanged
    and only the (small, byte-stable) steering block is reprocessed.

    A string system prompt is converted to block form so the original text
    keeps its exact bytes as the first block.
    """
    text = steering_text(level)
    if text is None:
        return False

    system = body.get("system")
    if system is None:
        body["system"] = [{"type": "text", "text": text}]
        return True
    if isinstance(system, str):
        body["system"] = [
            {"type": "text", "text": system},
            {"type": "text", "text": text},
        ]
        return True
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get("text", "").startswith(_STEERING_SENTINEL):
                if block["text"] == text:
                    return False  # already applied at this level
                block["text"] = text  # level changed mid-session
                return True
        system.append({"type": "text", "text": text})
        return True
    return False


def route_effort(
    body: dict[str, Any],
    kind: TurnKind,
    settings: OutputShaperSettings,
) -> list[str]:
    """Lower thinking/effort spend on mechanical continuations.

    Returns labels for each mutation made (empty list = untouched).
    """
    if kind is not TurnKind.MECHANICAL_CONTINUATION:
        return []

    labels: list[str] = []

    # Modern lever: output_config.effort. Only lower a value the client
    # explicitly sent — presence proves the target model accepts the param.
    output_config = body.get("output_config")
    if isinstance(output_config, dict):
        effort = output_config.get("effort")
        if (
            isinstance(effort, str)
            and effort in _EFFORT_RANK
            and _EFFORT_RANK[effort] > _EFFORT_RANK[settings.mechanical_effort]
        ):
            output_config["effort"] = settings.mechanical_effort
            labels.append(f"output_shaper:effort:{effort}->{settings.mechanical_effort}")

    # Legacy lever: clamp thinking.budget_tokens on models still using the
    # enabled/budget_tokens form. The type field itself is never touched.
    thinking = body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "enabled":
        budget = thinking.get("budget_tokens")
        if isinstance(budget, int) and budget > LEGACY_THINKING_FLOOR:
            thinking["budget_tokens"] = LEGACY_THINKING_FLOOR
            labels.append(f"output_shaper:thinking_budget:{budget}->{LEGACY_THINKING_FLOOR}")

    return labels


def shape_request(
    body: dict[str, Any],
    settings: OutputShaperSettings | None = None,
    level_override: int | None = None,
) -> ShapeResult:
    """Apply all output-shaping levers to an Anthropic request body in place.

    ``level_override`` supersedes ``settings.verbosity_level`` when given — the
    handler passes the level resolved by :func:`resolve_verbosity_level` (learned
    profile / controller / env) so the body-mutating core stays level-agnostic.
    """
    if settings is None:
        settings = OutputShaperSettings.from_env()
    result = ShapeResult()
    if not settings.enabled:
        return result

    assert result.labels is not None  # __post_init__ guarantees this

    level = settings.verbosity_level if level_override is None else level_override
    if level > 0 and apply_verbosity_steering(body, level):
        result.changed = True
        result.labels.append(f"output_shaper:verbosity:L{level}")

    if settings.effort_router_enabled:
        kind = classify_turn(body.get("messages", []))
        labels = route_effort(body, kind, settings)
        if labels:
            result.changed = True
            result.labels.extend(labels)
        logger.debug("OutputShaper: turn=%s mutations=%s", kind.value, labels)

    return result
