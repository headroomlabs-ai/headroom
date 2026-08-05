"""Shared token-savings profiles for coding agents."""

from __future__ import annotations

import logging
import os
from collections.abc import MutableMapping
from dataclasses import dataclass, replace
from typing import Protocol

logger = logging.getLogger(__name__)

AGENT_90_PROFILE = "agent-90"
FALLBACK_PROFILE = "balanced"
# Out-of-the-box default profile when none is requested. Headroom's primary
# workload is coding agents, and the "coding" profile encodes the cache-mode
# delta posture (see below).
DEFAULT_PROFILE = "coding"


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
    # None = don't pin a keep-ratio; let Kompress decide adaptively (and any
    # ambient HEADROOM_TARGET_RATIO / proxy default still applies). Workload
    # personas leave this unset so savings emerge from lossless + relevance.
    target_ratio: float | None
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
    # Standalone router/handler toggles carried through the profile so a single
    # named profile seeds Headroom's full posture. Defaults below preserve the
    # current global behavior, so only a profile that opts in changes anything.
    tool_search: bool = False
    cross_turn_dedup: bool = False
    lossless_then_lossy: bool = False
    protect_reads: bool = False
    code_aware: bool = True
    effort_router: bool = True
    lossless: bool = False
    min_chars_for_block: int | None = None

    @property
    def savings_percent(self) -> int:
        return round(self.target_savings * 100)

    def proxy_env(self) -> dict[str, str]:
        """Return env vars for Headroom proxy/wrapper entry points."""

        env = {
            "HEADROOM_MODE": self.proxy_mode,
            "HEADROOM_SAVINGS_PROFILE": self.name,
            "HEADROOM_SAVINGS_TARGET": f"{self.target_savings:.2f}",
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
            # Standalone router/handler toggles.
            "HEADROOM_TOOL_SEARCH": "1" if self.tool_search else "0",
            "HEADROOM_DEDUPE": "1" if self.cross_turn_dedup else "0",
            "HEADROOM_LOSSLESS_THEN_LOSSY": "1" if self.lossless_then_lossy else "0",
            "HEADROOM_PROTECT_READS": "1" if self.protect_reads else "0",
            "HEADROOM_CODE_AWARE_ENABLED": "1" if self.code_aware else "0",
            "HEADROOM_EFFORT_ROUTER": "1" if self.effort_router else "0",
            "HEADROOM_LOSSLESS": "1" if self.lossless else "0",
        }
        # Only pin a keep-ratio when the profile sets one; workload personas
        # leave it unset so Kompress decides and the ambient default applies.
        if self.target_ratio is not None:
            env["HEADROOM_TARGET_RATIO"] = f"{self.target_ratio:.2f}"
        # Block-compression char floor: only emit when the profile pins one.
        if self.min_chars_for_block is not None:
            env["HEADROOM_MIN_CHARS_FOR_BLOCK"] = str(self.min_chars_for_block)
        return env

    def apply_proxy_env_defaults(self, env: MutableMapping[str, str]) -> MutableMapping[str, str]:
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
    # Workload personas: compress aggressively while holding the three
    # invariants — no accuracy loss, no extra turns, no prefix-cache bust.
    # They rely on the defaults that already deliver this (relevance split on,
    # user/system messages protected, read-maturation off, lossless structural
    # compaction) and only set the workload-specific + visibility knobs. The
    # MCP-vs-airgapped axis is the separate HEADROOM_LOSSLESS toggle: markers
    # when a retrieve tool exists, marker-free lossless-first when air-gapped.
    # target_ratio is unset — savings emerge from lossless + relevance, and
    # Kompress decides its own keep. min_tokens is low so compression is
    # actually exercised/visible on modest outputs.
    "coding": AgentSavingsProfile(
        name="coding",
        target_savings=0.50,  # nominal (display only); savings are emergent
        target_ratio=None,
        # Cache mode compresses only the newest delta — a tool/user OBSERVATION —
        # so compress_user must be ON or there is nothing to compress. Prefix
        # stability (no bust) is preserved by the delta engine (frozen prefix +
        # append-only forwarding), not by refusing to touch user turns.
        compress_user_messages=True,
        compress_system_messages=False,  # system prompt is the hottest cache
        # Compress the newest observation delta. In cache mode the delta IS the
        # newest turn(s), so a positional guard here suppresses the only thing
        # cache mode can compress (large recent grep/test/build tool output).
        # The code working set stays byte-exact via protect_reads below — a TYPE
        # guard on file reads — which is the real fidelity protection, so 0 loses
        # no correctness while unblocking compression of recent non-read output.
        protect_recent=0,
        protect_analysis_context=True,
        min_tokens_to_compress=10,  # low → even modest deltas are eligible
        max_items_after_crush=15,
        smart_crusher_with_compaction=True,
        force_kompress=False,  # don't override diff/log lossless with lossy ML
        proxy_mode="cache",  # delta-only compression at ~0 prefix-cache busts
        accuracy_guard="strict",
        # Coding posture: defer non-core tool schemas, dedupe re-reads, extend
        # lossy coverage when lossless found nothing, and NEVER lossy-compress a
        # file read (the agent patches exact bytes). CCR stays ON (unset) so any
        # lossy loss is recoverable. Effort router off; low block floor so modest
        # deltas are eligible.
        tool_search=True,
        cross_turn_dedup=True,
        lossless_then_lossy=True,
        protect_reads=True,
        code_aware=True,
        effort_router=False,
        lossless=False,
        min_chars_for_block=25,
    ),
    "general": AgentSavingsProfile(
        name="general",
        target_savings=0.60,  # nominal (display only); savings are emergent
        target_ratio=None,
        compress_user_messages=False,
        compress_system_messages=False,
        protect_recent=0,  # little code; nothing positional to protect
        protect_analysis_context=True,
        min_tokens_to_compress=25,
        max_items_after_crush=15,
        smart_crusher_with_compaction=True,
        force_kompress=False,
        proxy_mode="token",
        accuracy_guard="strict",
    ),
}


def get_agent_savings_profile(name: str | None = None) -> AgentSavingsProfile:
    """Return a named agent savings profile.

    An unrecognized name falls back to the ``balanced`` profile with a warning
    instead of raising. The savings profile is a soft config knob, but it is
    resolved during proxy startup (``proxy_pipeline_kwargs`` -> ``create_app``),
    so raising here takes the whole proxy down before it can open its port. That
    happens on desktop/runtime version skew: a newer client requests a profile
    (e.g. ``coding``) that an older pinned or fallback runtime predates. Degrade
    to ``balanced`` rather than leaving the user with no proxy at all.
    """

    key = (name or DEFAULT_PROFILE).strip().lower()
    profile = _PROFILES.get(key)
    if profile is not None:
        return profile
    valid = ", ".join(sorted(_PROFILES))
    logger.warning(
        "unknown savings profile %r; falling back to %r (known: %s)",
        name,
        FALLBACK_PROFILE,
        valid,
    )
    return _PROFILES[FALLBACK_PROFILE]


def apply_agent_savings_env_defaults(
    env: MutableMapping[str, str],
    profile: AgentSavingsProfile | str | None = None,
) -> MutableMapping[str, str]:
    """Apply agent savings env defaults to a proxy subprocess environment.

    When ``profile`` is not given, an explicit ``HEADROOM_SAVINGS_PROFILE`` already
    in ``env`` is honored; only when that too is absent do we fall back to the
    out-of-box default (:data:`DEFAULT_PROFILE`, i.e. ``coding``).
    """

    if profile is None:
        profile = env.get("HEADROOM_SAVINGS_PROFILE")
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
    if resolved.target_ratio is not None:
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
                "min_tokens_to_compress": profile.min_tokens_to_compress,
                "max_items_after_crush": profile.max_items_after_crush,
                "smart_crusher_with_compaction": profile.smart_crusher_with_compaction,
                "force_kompress": profile.force_kompress,
                "read_protection_window": profile.protect_recent,
            }
        )
        # Only pin a keep-ratio when the profile sets one (personas leave it
        # unset → Kompress decides / ambient default applies).
        if profile.target_ratio is not None:
            kwargs["target_ratio"] = profile.target_ratio

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

    # Lower the block-compression char floor (default 500) so modest tool outputs
    # are eligible for the LOSSY path too. Matters in cache mode, where the only
    # compressible content each turn is a single (often small) delta observation;
    # a 500-char floor buckets most of them as "small" (skipped). Env-gated so it
    # only changes behavior when explicitly set; lossless folding has no floor and
    # is unaffected.
    _min_chars_block = os.environ.get("HEADROOM_MIN_CHARS_FOR_BLOCK")
    if _min_chars_block:
        try:
            kwargs["min_chars_for_block_compression"] = int(_min_chars_block)
        except ValueError:
            pass

    return kwargs


def seed_proxy_env_defaults(env: MutableMapping[str, str] | None = None) -> None:
    """Seed the process env with the savings-profile defaults (default: coding).

    Call at proxy EXECUTABLE entry points (the ``headroom proxy`` command and the
    uvicorn ``create_app_from_env`` factory) BEFORE building config from env. Uses
    ``setdefault`` semantics via :func:`apply_agent_savings_env_defaults`, so any
    explicit user setting wins and the call is idempotent.

    Deliberately NOT called from ``_proxy_config_from_env`` / ``ContentRouter`` or
    any other library-level builder that unit tests construct directly, so those
    keep clean (unseeded) defaults and test isolation is preserved.
    """
    target = os.environ if env is None else env
    # apply_agent_savings_env_defaults honors an explicit HEADROOM_SAVINGS_PROFILE
    # already in the env and otherwise falls back to DEFAULT_PROFILE (coding).
    apply_agent_savings_env_defaults(target)


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
