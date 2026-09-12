"""Output token shaping for proxied Anthropic and OpenAI Responses requests.

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

The same lever exists for the OpenAI Responses format (Codex et al.):
:func:`apply_responses_verbosity_steering` appends the byte-stable steering
block to the tail of the ``instructions`` string, and
:func:`shape_responses_request` is the Responses-format counterpart of
:func:`shape_request`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from headroom.proxy import runtime_env
from headroom.proxy.output_steering import (
    apply_openai_chat_verbosity_steering,
    apply_openai_responses_verbosity_steering,
    apply_verbosity_steering,
    replace_or_append_steering_block,
    steering_text,
)
from headroom.proxy.output_turn_policy import (
    TurnKind,
    classify_openai_responses_input,
    classify_turn,
)
from headroom.rollout import FeatureDecisionReason

logger = logging.getLogger(__name__)

__all__ = [
    "OutputShaperSettings",
    "ShapeResult",
    "TurnKind",
    "apply_openai_chat_verbosity_steering",
    "apply_openai_responses_verbosity_steering",
    "apply_verbosity_steering",
    "classify_openai_responses_input",
    "classify_turn",
    "resolve_verbosity_level",
    "shape_openai_chat_request",
    "shape_openai_responses_request",
    "shape_request",
    "steering_text",
]

_replace_or_append_steering_block = replace_or_append_steering_block


@dataclass(frozen=True)
class OutputShaperSettings:
    """Output-shaping settings with rollout enablement injected by the proxy."""

    enabled: bool = False
    verbosity_level: int = 3
    # False in ``mode="cache"``. Steering is the one lever that writes into the
    # provider prefix-cache key (it appends to the system-prompt tail, and on a
    # body with no system field it creates one); effort routing and the
    # thinking budget sit outside the key and stay on. See
    # :func:`steering_allowed_for`.
    steering_enabled: bool = True

    @classmethod
    def from_env(
        cls, *, enabled: bool | None = None, steering_enabled: bool = True
    ) -> OutputShaperSettings:
        """Resolve tuning; running proxies always inject the resolved gate.

        ``None`` preserves the helper's direct-call compatibility for SDK/tests,
        but proxy request paths never use it and therefore never re-resolve the
        rollout alias.
        """
        if enabled is None:
            enabled = runtime_env.getenv("HEADROOM_OUTPUT_SHAPER", "").lower() in (
                "1",
                "true",
                "yes",
            )
        try:
            level = int(runtime_env.getenv("HEADROOM_VERBOSITY_LEVEL", "3"))
        except ValueError:
            level = 2
        level = max(0, min(4, level))
        return cls(
            enabled=enabled,
            verbosity_level=level,
            steering_enabled=steering_enabled,
        )


def shaper_enabled_for(config: Any) -> bool | None:
    """Resolve the output-shaper gate for a proxy config.

    Output shaping is deliberately independent of input compression — an
    operator can run ``optimize=False`` and still want terser responses, and
    the WS/Responses shaper tests pin that combination. So ``optimize`` does
    not veto shaping outright. What it vetoes is shaping that nobody asked
    for.

    Since the feature defaults on, ``optimize=False`` plus *no* explicit
    request means an operator who turned every transform off would silently
    start getting a steering block appended to their system-prompt tail — and
    on a request carrying no ``system`` field at all, would have one created.
    That breaks the byte-faithful forwarding invariant. So in that one
    combination the default loses:

    * enabled explicitly (``HEADROOM_OUTPUT_SHAPER=1``, ``HEADROOM_FEATURES``)
      → shape, whatever ``optimize`` says;
    * enabled only by default, with ``optimize=False`` → do not shape;
    * enabled only by default, with ``optimize=True`` → shape.

    Returns ``None`` when there is no rollout snapshot to consult, which
    preserves :meth:`OutputShaperSettings.from_env`'s env-var fallback for the
    SDK and test callers that construct a config without one.
    """
    rollout = getattr(config, "rollout", None)
    if rollout is None:
        return None
    try:
        decision = rollout.decision("proxy_output_shaper")
    except (KeyError, AttributeError):
        return None
    if not decision.enabled:
        return False
    if getattr(config, "optimize", True):
        return True
    return decision.reason is not FeatureDecisionReason.DEFAULT


def steering_allowed_for(config: Any) -> bool:
    """False when the proxy is in prefix-freezing cache mode.

    ``mode="cache"`` freezes prior turns specifically to keep the provider's
    prefix-cache key byte-stable (see ``ProxyConfig.mode``). Verbosity steering
    writes into that key, so running it there trades a large, certain cache
    cost for a small, uncertain output saving — the wrong side of a roughly
    60x margin on a long context. Effort routing and the thinking budget are
    unaffected: they ride request parameters outside the cache key, so they
    keep saving in cache mode.
    """
    return getattr(config, "mode", None) != "cache"


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
    if not settings.steering_enabled:
        # Level 0 is the documented "no steering" value, so this disables the
        # only cache-key-mutating lever while leaving effort routing on. It
        # deliberately outranks the manual override below: a level set in the
        # environment must not reintroduce a prefix mutation the mode exists
        # to prevent.
        return 0, "cache_mode"
    if runtime_env.getenv("HEADROOM_VERBOSITY_LEVEL"):
        return settings.verbosity_level, "env"

    try:
        from ..paths import workspace_dir

        ws = workspace_dir()
    except Exception:
        return settings.verbosity_level, "default"

    autotune = runtime_env.getenv("HEADROOM_VERBOSITY_AUTOTUNE", "").lower() in ("1", "true", "yes")
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


def shape_openai_responses_request(
    body: dict[str, Any],
    settings: OutputShaperSettings | None = None,
    level_override: int | None = None,
) -> ShapeResult:
    """Apply OpenAI Responses output-shaping levers in place."""
    if settings is None:
        settings = OutputShaperSettings.from_env()
    result = ShapeResult()
    if not settings.enabled:
        return result

    assert result.labels is not None  # __post_init__ guarantees

    level = settings.verbosity_level if level_override is None else level_override
    if level > 0 and apply_openai_responses_verbosity_steering(body, level):
        result.changed = True
        result.labels.append(f"output_shaper:verbosity:L{level}")

    return result


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

    return result


def shape_openai_chat_request(
    body: dict[str, Any],
    settings: OutputShaperSettings | None = None,
    level_override: int | None = None,
) -> ShapeResult:
    """Apply output-shaping levers to an OpenAI chat/completions body in place.

    The chat counterpart of :func:`shape_request`. Chat carries the system
    prompt as a ``role: "system"`` message, so verbosity steering uses the
    chat-specific injector.
    """
    if settings is None:
        settings = OutputShaperSettings.from_env()
    result = ShapeResult()
    if not settings.enabled:
        return result

    assert result.labels is not None  # __post_init__ guarantees this

    level = settings.verbosity_level if level_override is None else level_override
    if level > 0 and apply_openai_chat_verbosity_steering(body, level):
        result.changed = True
        result.labels.append(f"output_shaper:verbosity:L{level}")

    return result


# ---------------------------------------------------------------------------
# OpenAI Responses format (Codex, /v1/responses HTTP + WebSocket)
# ---------------------------------------------------------------------------

# Trailing ``input`` item types that represent tool output coming back to the
# model — the Responses counterpart of an Anthropic ``tool_result`` block.
_RESPONSES_TOOL_OUTPUT_TYPES = frozenset(
    {
        "custom_tool_call_output",
        "function_call_output",
        "local_shell_call_output",
        "apply_patch_call_output",
    }
)


def _responses_tool_output_is_error(item: dict[str, Any]) -> bool:
    """Structural error sniff on a Responses tool-output item.

    The Responses format has no ``is_error`` flag, but agent harnesses encode
    failure structurally in the ``output`` payload: a JSON object with a
    nonzero ``exit_code``, ``success: false``, or a truthy ``error`` field.
    Only those JSON fields are inspected — never prose content.
    """
    output = item.get("output")
    data: Any = output
    if isinstance(output, str):
        stripped = output.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            return False
        try:
            data = json.loads(stripped)
        except (ValueError, TypeError):
            return False
    if not isinstance(data, dict):
        return False
    # Direct fields, plus the common {"output": ..., "metadata": {...}} nesting.
    scopes: list[dict[str, Any]] = [data]
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        scopes.append(metadata)
    for scope in scopes:
        exit_code = scope.get("exit_code")
        if isinstance(exit_code, int) and exit_code != 0:
            return True
        if scope.get("success") is False:
            return True
        if scope.get("error"):
            return True
    return False


def classify_responses_turn(input_data: Any) -> TurnKind:
    """Classify a Responses request's turn from its ``input`` field.

    Mirrors :func:`classify_turn` semantics on the Responses item list: the
    trailing run of tool-output items decides the turn. A trailing user
    message is a new ask; tool outputs are mechanical unless any carries a
    structural error marker. Purely structural — item types and JSON fields,
    no content regexes.
    """
    if isinstance(input_data, str):
        return TurnKind.NEW_USER_ASK if input_data.strip() else TurnKind.UNKNOWN
    if not isinstance(input_data, list) or not input_data:
        return TurnKind.UNKNOWN

    saw_tool_output = False
    saw_error = False
    for item in reversed(input_data):
        if not isinstance(item, dict):
            return TurnKind.UNKNOWN
        itype = item.get("type")
        if itype in _RESPONSES_TOOL_OUTPUT_TYPES:
            saw_tool_output = True
            if _responses_tool_output_is_error(item):
                saw_error = True
            continue
        # First non-tool-output item ends the trailing run.
        if saw_tool_output:
            break
        if itype == "message" or (itype is None and "role" in item):
            role = item.get("role")
            if role == "user":
                return TurnKind.NEW_USER_ASK
            return TurnKind.UNKNOWN
        return TurnKind.UNKNOWN

    if saw_error:
        return TurnKind.ERROR_CONTINUATION
    if saw_tool_output:
        return TurnKind.MECHANICAL_CONTINUATION
    return TurnKind.UNKNOWN


def apply_responses_verbosity_steering(body: dict[str, Any], level: int) -> bool:
    """Append the steering block to the tail of ``instructions``.

    ``instructions`` is the Responses cache hot zone: the appended block is
    byte-stable per level, so within a conversation every shaped turn sends
    identical instructions bytes and the provider prefix cache stays hot
    after the first shaped turn (the same contract as the Anthropic
    system-tail append).
    """
    return apply_openai_responses_verbosity_steering(body, level)


def shape_responses_request(
    body: dict[str, Any],
    settings: OutputShaperSettings | None = None,
    level_override: int | None = None,
) -> ShapeResult:
    """Apply all output-shaping levers to a Responses payload in place.

    The Responses counterpart of :func:`shape_request`: same settings, same
    labels, same level-resolution contract.
    """
    if settings is None:
        settings = OutputShaperSettings.from_env()
    result = ShapeResult()
    if not settings.enabled:
        return result

    assert result.labels is not None  # __post_init__ guarantees this

    level = settings.verbosity_level if level_override is None else level_override
    if level > 0 and apply_responses_verbosity_steering(body, level):
        result.changed = True
        result.labels.append(f"output_shaper:verbosity:L{level}")

    return result
