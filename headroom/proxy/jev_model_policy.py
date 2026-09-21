"""Optional Jev-backed semantic model routing (issue #3690).

Complements the heuristic :class:`~headroom.proxy.model_router.ModelRouter`
(#1706) with a TypeSafe System One (Jev) policy: classify truncated user text
into economy / standard / frontier tiers, then map tiers to operator-configured
model IDs.

Off by default. Fail-open: any missing config, HTTP error, low confidence, or
unknown tier leaves the original model unchanged (or lets the static router
run). Uses existing ``httpx`` — no new dependency.

Docs: https://docs.typesafe.ai/introduction
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from headroom.proxy.model_router import ModelDecision

logger = logging.getLogger(__name__)

TYPESAFE_SYSTEM_ONE_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
DEFAULT_CONFIDENCE_MIN = 0.6
DEFAULT_TIMEOUT_S = 2.0
DEFAULT_STATE_CHARS = 2000

TIERS = ("economy", "standard", "frontier")

_TIER_QUESTIONS: dict[str, Any] = {
    "tier": {
        "type": "choice",
        "instructions": (
            "Which model capability tier should handle this user request? "
            "economy = trivial edits, lookups, renames, formatting; "
            "standard = typical coding or reasoning; "
            "frontier = hard design, security-sensitive, multi-file architecture, "
            "or ambiguous high-stakes work."
        ),
        "criteria": {
            "economy": "Trivial, low-risk, small local change",
            "standard": "Normal coding or analysis task",
            "frontier": "Hard, ambiguous, or high-stakes work",
        },
    },
    "difficulty": {
        "type": "score",
        "instructions": "How difficult is this request for a coding agent?",
        "criteria": [
            "Trivial one-step change",
            "Moderate multi-step work",
            "Hard design or high-risk work",
        ],
    },
}


@dataclass(frozen=True)
class JevModelPolicyConfig:
    """Configuration for :class:`JevModelPolicy`. Disabled by default."""

    enabled: bool = False
    api_key: str = ""
    tier_models: Mapping[str, str] | None = None
    confidence_min: float = DEFAULT_CONFIDENCE_MIN
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_state_chars: int = DEFAULT_STATE_CHARS
    system_one_url: str = TYPESAFE_SYSTEM_ONE_URL
    jev_model: str = DEFAULT_MODEL

    @classmethod
    def from_env(cls) -> JevModelPolicyConfig:
        """Build from env, failing open to disabled on any bad config."""
        enabled = _truthy(os.environ.get("HEADROOM_JEV_MODEL_ROUTER"))
        if not enabled:
            return cls(enabled=False)

        api_key = (os.environ.get("TYPESAFE_API_KEY") or "").strip()
        tier_models = _parse_tier_models(os.environ.get("HEADROOM_JEV_TIER_MODELS"))
        confidence_min = _parse_float(
            os.environ.get("HEADROOM_JEV_CONFIDENCE_MIN"),
            DEFAULT_CONFIDENCE_MIN,
        )
        timeout_s = _parse_float(
            os.environ.get("HEADROOM_JEV_TIMEOUT_S"),
            DEFAULT_TIMEOUT_S,
        )
        max_state_chars = int(
            _parse_float(
                os.environ.get("HEADROOM_JEV_MAX_STATE_CHARS"),
                float(DEFAULT_STATE_CHARS),
            )
        )

        if not api_key:
            logger.warning(
                "HEADROOM_JEV_MODEL_ROUTER enabled but TYPESAFE_API_KEY missing; disabling"
            )
            return cls(enabled=False)
        if not tier_models:
            logger.warning(
                "HEADROOM_JEV_MODEL_ROUTER enabled but HEADROOM_JEV_TIER_MODELS missing/invalid; "
                "disabling"
            )
            return cls(enabled=False)

        return cls(
            enabled=True,
            api_key=api_key,
            tier_models=tier_models,
            confidence_min=confidence_min,
            timeout_s=timeout_s,
            max_state_chars=max(200, max_state_chars),
        )


class JevModelPolicy:
    """Selects an outgoing model via TypeSafe Jev; returns :class:`ModelDecision`."""

    def __init__(
        self,
        config: JevModelPolicyConfig | None = None,
        *,
        post: Callable[..., httpx.Response] | None = None,
    ) -> None:
        self._config = config or JevModelPolicyConfig()
        # Injectable for tests; production uses httpx.post.
        self._post = post or httpx.post

    @property
    def enabled(self) -> bool:
        return bool(self._config.enabled and self._config.api_key and self._config.tier_models)

    def select(self, *, model: str, user_text: str) -> ModelDecision:
        """Return a routing decision. Never raises."""
        if not self.enabled:
            return ModelDecision(model, model, matched=False, reason="jev router disabled")
        if not isinstance(model, str) or not model:
            return ModelDecision(model, model, matched=False, reason="jev: no source model")

        state = (user_text or "").strip()
        if not state:
            return ModelDecision(model, model, matched=False, reason="jev: empty user text")
        if len(state) > self._config.max_state_chars:
            state = state[: self._config.max_state_chars]

        try:
            answers = self._system_one(state)
        except Exception as exc:  # noqa: BLE001 — fail open
            logger.warning("jev model router request failed: %s", exc)
            return ModelDecision(
                model, model, matched=False, reason=f"jev error: {type(exc).__name__}"
            )

        tier_ans = answers.get("tier") if isinstance(answers, dict) else None
        if not isinstance(tier_ans, dict):
            return ModelDecision(model, model, matched=False, reason="jev: missing tier answer")

        tier = tier_ans.get("choice")
        confidence = tier_ans.get("confidence")
        try:
            conf = float(confidence) if confidence is not None else 0.0
        except (TypeError, ValueError):
            conf = 0.0

        if not isinstance(tier, str) or tier not in TIERS:
            return ModelDecision(
                model,
                model,
                matched=True,
                reason=f"jev: unknown tier {tier!r}; keeping {model}",
                rule_name="jev-tier",
            )

        if conf < self._config.confidence_min:
            return ModelDecision(
                model,
                model,
                matched=True,
                reason=(
                    f"jev: tier={tier} confidence={conf:.3f} "
                    f"< {self._config.confidence_min}; keeping {model}"
                ),
                rule_name="jev-tier",
            )

        tier_models = self._config.tier_models or {}
        routed = tier_models.get(tier)
        if not routed:
            return ModelDecision(
                model,
                model,
                matched=True,
                reason=f"jev: tier={tier} has no mapped model; keeping {model}",
                rule_name="jev-tier",
            )

        difficulty = ""
        diff_ans = answers.get("difficulty") if isinstance(answers, dict) else None
        if isinstance(diff_ans, dict) and "score" in diff_ans:
            difficulty = f", difficulty={diff_ans.get('score')}"

        reason = f"jev matched tier={tier} confidence={conf:.3f}{difficulty}: {model} -> {routed}"
        return ModelDecision(
            original_model=model,
            routed_model=routed,
            matched=True,
            reason=reason,
            rule_name="jev-tier",
        )

    def _system_one(self, state: str) -> dict[str, Any]:
        payload = {
            "model": self._config.jev_model,
            "state": state,
            "questions": _TIER_QUESTIONS,
        }
        response = self._post(
            self._config.system_one_url,
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self._config.timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        answers = data.get("answers") if isinstance(data, dict) else None
        if not isinstance(answers, dict):
            raise ValueError("systemone response missing answers object")
        return answers


def latest_user_text(messages: object, *, max_chars: int = DEFAULT_STATE_CHARS) -> str:
    """Extract the newest user turn text for Jev state (truncated). Never raises."""
    try:
        if not isinstance(messages, list):
            return ""
        for msg in reversed(messages):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            text = _content_to_text(content)
            if text:
                return text if len(text) <= max_chars else text[:max_chars]
        return ""
    except Exception:  # noqa: BLE001
        return ""


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = str(block.get("text", "")).strip()
                if t and not t.startswith("<system-reminder"):
                    parts.append(t)
            elif isinstance(block, str) and block.strip():
                parts.append(block.strip())
        return "\n".join(parts).strip()
    return str(content).strip() if content else ""


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on", "enable", "enabled"}


def _parse_tier_models(raw: str | None) -> dict[str, str] | None:
    if not raw or not raw.strip():
        return None
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError) as exc:
        logger.warning("invalid HEADROOM_JEV_TIER_MODELS JSON; ignoring: %s", exc)
        return None
    if not isinstance(parsed, dict):
        logger.warning("HEADROOM_JEV_TIER_MODELS must be a JSON object; ignoring")
        return None
    out: dict[str, str] = {}
    for key, value in parsed.items():
        if key not in TIERS:
            logger.warning("HEADROOM_JEV_TIER_MODELS unknown tier %r; skipping", key)
            continue
        if not isinstance(value, str) or not value.strip():
            logger.warning("HEADROOM_JEV_TIER_MODELS tier %r needs a non-empty string model", key)
            continue
        out[key] = value.strip()
    return out or None


def _parse_float(raw: str | None, default: float) -> float:
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("invalid float %r; using default %s", raw, default)
        return default
