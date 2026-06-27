"""Release-channel and feature-flag policy for Headroom.

The rollout layer is intentionally small: it answers whether a named feature is
eligible in the current release channel and whether it is enabled by default or
explicit request. Composition roots should depend on this module instead of
reading feature env vars directly.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "off", "disabled"}


class ReleaseChannel(str, Enum):
    """Ordered stability channels used to bake features before stable release."""

    STABLE = "stable"
    BETA = "beta"
    CANARY = "canary"
    DEV = "dev"

    @classmethod
    def parse(cls, value: str | None) -> ReleaseChannel:
        if not value:
            return cls.STABLE
        normalized = value.strip().lower().replace("-", "_")
        aliases = {
            "prod": cls.STABLE,
            "production": cls.STABLE,
            "preview": cls.BETA,
            "nightly": cls.CANARY,
            "development": cls.DEV,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError:
            return cls.STABLE

    @property
    def order(self) -> int:
        return {
            ReleaseChannel.STABLE: 0,
            ReleaseChannel.BETA: 1,
            ReleaseChannel.CANARY: 2,
            ReleaseChannel.DEV: 3,
        }[self]

    def allows(self, required: ReleaseChannel) -> bool:
        return self.order >= required.order


@dataclass(frozen=True)
class FeatureSpec:
    """Static policy for one rollout-managed feature."""

    name: str
    available_in: ReleaseChannel
    default_enabled_in: ReleaseChannel | None = None
    legacy_env: tuple[str, ...] = ()
    description: str = ""

    def default_enabled(self, channel: ReleaseChannel) -> bool:
        return self.default_enabled_in is not None and channel.allows(self.default_enabled_in)


FEATURES: dict[str, FeatureSpec] = {
    "tool_result_interceptors": FeatureSpec(
        name="tool_result_interceptors",
        available_in=ReleaseChannel.CANARY,
        default_enabled_in=None,
        legacy_env=("HEADROOM_INTERCEPT_ENABLED",),
        description="AST-aware Read/tool-result interceptors used before compression.",
    ),
    "proxy_output_shaper": FeatureSpec(
        name="proxy_output_shaper",
        available_in=ReleaseChannel.BETA,
        default_enabled_in=None,
        legacy_env=("HEADROOM_OUTPUT_SHAPER",),
        description="Proxy output-shaping path for response-side experiments.",
    ),
    "read_maturation": FeatureSpec(
        name="read_maturation",
        available_in=ReleaseChannel.BETA,
        default_enabled_in=None,
        legacy_env=("HEADROOM_READ_MATURATION",),
        description="Hold-back Read maturation before provider cache entry.",
    ),
}


def _split_names(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {
        part.strip().lower().replace("-", "_")
        for part in raw.replace(";", ",").split(",")
        if part.strip()
    }


def _truthy(value: str | None) -> bool:
    return bool(value and value.strip().lower() in _TRUE_VALUES)


def _falsey(value: str | None) -> bool:
    return bool(value and value.strip().lower() in _FALSE_VALUES)


@dataclass(frozen=True)
class Rollout:
    """Resolved runtime rollout state."""

    channel: ReleaseChannel
    enabled: frozenset[str]
    disabled: frozenset[str]
    unsafe_allow_unstable: bool = False

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Rollout:
        env = os.environ if environ is None else environ
        return cls(
            channel=ReleaseChannel.parse(env.get("HEADROOM_RELEASE_CHANNEL")),
            enabled=frozenset(_split_names(env.get("HEADROOM_FEATURES"))),
            disabled=frozenset(_split_names(env.get("HEADROOM_DISABLE_FEATURES"))),
            unsafe_allow_unstable=_truthy(env.get("HEADROOM_UNSAFE_ALLOW_UNSTABLE_FEATURES")),
        )

    def is_available(self, feature: str) -> bool:
        spec = FEATURES[feature]
        return self.channel.allows(spec.available_in) or self.unsafe_allow_unstable

    def is_enabled(
        self,
        feature: str,
        *,
        explicit: bool = False,
        environ: Mapping[str, str] | None = None,
    ) -> bool:
        """Return whether a feature may run in this process.

        `explicit=True` represents a typed configuration or CLI option. Legacy
        per-feature env vars are treated as explicit requests for migration, but
        they still obey the channel gate.
        """
        spec = FEATURES[feature]
        if feature in self.disabled:
            return False
        if not self.is_available(feature):
            return False
        if feature in self.enabled:
            return True
        if explicit:
            return True

        env = os.environ if environ is None else environ
        for name in spec.legacy_env:
            raw = env.get(name)
            if _falsey(raw):
                return False
            if _truthy(raw):
                return True

        return spec.default_enabled(self.channel)


def current_rollout(environ: Mapping[str, str] | None = None) -> Rollout:
    """Resolve rollout state from the process environment."""

    return Rollout.from_env(environ)


def feature_enabled(
    feature: str,
    *,
    explicit: bool = False,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Convenience wrapper for composition roots."""

    return current_rollout(environ).is_enabled(feature, explicit=explicit, environ=environ)
