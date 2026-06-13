"""Shared token-savings profiles for coding agents."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

AGENT_90_PROFILE = "agent-90"


class CompressConfigLike(Protocol):
    compress_user_messages: bool
    compress_system_messages: bool
    protect_recent: int
    protect_analysis_context: bool
    target_ratio: float | None
    min_tokens_to_compress: int


@dataclass(frozen=True)
class AgentSavingsProfile:
    """Reusable policy for high-savings agent compression."""

    name: str
    target_savings: float
    target_ratio: float
    compress_user_messages: bool
    compress_system_messages: bool
    protect_recent: int
    protect_analysis_context: bool
    min_tokens_to_compress: int
    max_items_after_crush: int
    smart_crusher_with_compaction: bool
    force_kompress: bool
    proxy_mode: str
    accuracy_guard: str

    @property
    def savings_percent(self) -> int:
        return round(self.target_savings * 100)

    def proxy_env(self) -> dict[str, str]:
        """Return env vars for Headroom proxy/wrapper entry points."""

        return {
            "HEADROOM_MODE": self.proxy_mode,
            "HEADROOM_SAVINGS_PROFILE": self.name,
            "HEADROOM_SAVINGS_TARGET": f"{self.target_savings:.2f}",
            "HEADROOM_TARGET_RATIO": f"{self.target_ratio:.2f}",
            "HEADROOM_COMPRESS_USER_MESSAGES": ("1" if self.compress_user_messages else "0"),
            "HEADROOM_COMPRESS_SYSTEM_MESSAGES": ("1" if self.compress_system_messages else "0"),
            "HEADROOM_PROTECT_RECENT": str(self.protect_recent),
            "HEADROOM_PROTECT_ANALYSIS_CONTEXT": ("1" if self.protect_analysis_context else "0"),
            "HEADROOM_MIN_TOKENS": str(self.min_tokens_to_compress),
            "HEADROOM_MAX_ITEMS": str(self.max_items_after_crush),
            "HEADROOM_SMART_CRUSHER_COMPACTION": (
                "1" if self.smart_crusher_with_compaction else "0"
            ),
            "HEADROOM_FORCE_KOMPRESS": "1" if self.force_kompress else "0",
            "HEADROOM_ACCURACY_GUARD": self.accuracy_guard,
        }

    def apply_proxy_env_defaults(self, env: dict[str, str]) -> dict[str, str]:
        """Seed proxy env defaults without overriding explicit user settings."""

        for key, value in self.proxy_env().items():
            env.setdefault(key, value)
        return env


_PROFILES: dict[str, AgentSavingsProfile] = {
    AGENT_90_PROFILE: AgentSavingsProfile(
        name=AGENT_90_PROFILE,
        target_savings=0.90,
        target_ratio=0.10,
        compress_user_messages=True,
        compress_system_messages=True,
        protect_recent=2,
        protect_analysis_context=True,
        min_tokens_to_compress=120,
        max_items_after_crush=8,
        smart_crusher_with_compaction=False,
        force_kompress=True,
        proxy_mode="token",
        accuracy_guard="strict",
    ),
    "balanced": AgentSavingsProfile(
        name="balanced",
        target_savings=0.70,
        target_ratio=0.30,
        compress_user_messages=False,
        compress_system_messages=False,
        protect_recent=4,
        protect_analysis_context=True,
        min_tokens_to_compress=250,
        max_items_after_crush=15,
        smart_crusher_with_compaction=True,
        force_kompress=False,
        proxy_mode="token",
        accuracy_guard="strict",
    ),
}


def get_agent_savings_profile(name: str | None = None) -> AgentSavingsProfile:
    """Return a named agent savings profile."""

    key = (name or AGENT_90_PROFILE).strip().lower()
    try:
        return _PROFILES[key]
    except KeyError as exc:
        valid = ", ".join(sorted(_PROFILES))
        raise ValueError(f"unknown savings profile {name!r}; expected one of: {valid}") from exc


def apply_agent_savings_env_defaults(
    env: dict[str, str],
    profile: AgentSavingsProfile | str | None = None,
) -> dict[str, str]:
    """Apply agent savings env defaults to a proxy subprocess environment."""

    resolved = (
        get_agent_savings_profile(profile)
        if isinstance(profile, str) or profile is None
        else profile
    )
    return resolved.apply_proxy_env_defaults(env)


def apply_agent_savings_profile(
    config: CompressConfigLike,
    profile: AgentSavingsProfile | str | None = None,
) -> CompressConfigLike:
    """Apply a profile to an existing ``CompressConfig``-like object."""

    resolved = (
        get_agent_savings_profile(profile)
        if isinstance(profile, str) or profile is None
        else profile
    )
    config.compress_user_messages = resolved.compress_user_messages
    config.compress_system_messages = resolved.compress_system_messages
    config.protect_recent = resolved.protect_recent
    config.protect_analysis_context = resolved.protect_analysis_context
    config.target_ratio = resolved.target_ratio
    config.min_tokens_to_compress = resolved.min_tokens_to_compress
    return config


def proxy_pipeline_kwargs(config: object) -> dict[str, object]:
    """Build per-request pipeline kwargs from proxy config and savings profile.

    The proxy has provider-specific handlers, but the accuracy-sensitive
    compression knobs should be consistent across Claude, Codex, and Cursor.
    """

    kwargs: dict[str, object] = {}
    profile_name = getattr(config, "savings_profile", None)
    if profile_name:
        profile = get_agent_savings_profile(str(profile_name))
        kwargs.update(
            {
                "compress_user_messages": profile.compress_user_messages,
                "compress_system_messages": profile.compress_system_messages,
                "protect_recent": profile.protect_recent,
                "protect_analysis_context": profile.protect_analysis_context,
                "target_ratio": profile.target_ratio,
                "min_tokens_to_compress": profile.min_tokens_to_compress,
                "max_items_after_crush": profile.max_items_after_crush,
                "smart_crusher_with_compaction": profile.smart_crusher_with_compaction,
                "force_kompress": profile.force_kompress,
                "read_protection_window": profile.protect_recent,
            }
        )

    if getattr(config, "compress_user_messages", False):
        kwargs["compress_user_messages"] = True

    compress_system_messages = getattr(config, "compress_system_messages", None)
    if compress_system_messages is not None:
        kwargs["compress_system_messages"] = bool(compress_system_messages)

    protect_recent = getattr(config, "protect_recent", None)
    if protect_recent is not None:
        kwargs["protect_recent"] = int(protect_recent)

    protect_analysis_context = getattr(config, "protect_analysis_context", None)
    if protect_analysis_context is not None:
        kwargs["protect_analysis_context"] = bool(protect_analysis_context)

    target_ratio = getattr(config, "target_ratio", None)
    if target_ratio is not None:
        kwargs["target_ratio"] = float(target_ratio)

    min_tokens = getattr(config, "min_tokens_to_crush", None)
    if min_tokens is not None and (not profile_name or int(min_tokens) != 500):
        kwargs["min_tokens_to_compress"] = int(min_tokens)

    max_items = getattr(config, "max_items_after_crush", None)
    if max_items is not None and (not profile_name or int(max_items) != 50):
        kwargs["max_items_after_crush"] = int(max_items)

    smart_crusher_with_compaction = getattr(
        config,
        "smart_crusher_with_compaction",
        None,
    )
    if smart_crusher_with_compaction is not None:
        kwargs["smart_crusher_with_compaction"] = bool(smart_crusher_with_compaction)

    return kwargs


def with_target_savings(
    profile: AgentSavingsProfile,
    target_savings: float,
) -> AgentSavingsProfile:
    """Return a copy of ``profile`` adjusted to a specific savings target."""

    if not 0 < target_savings < 1:
        raise ValueError("target_savings must be between 0 and 1")
    return replace(
        profile,
        target_savings=target_savings,
        target_ratio=round(1 - target_savings, 4),
    )
