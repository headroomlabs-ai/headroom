"""Detached-mode capability contract for the proxy runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

from headroom import paths
from headroom.proxy.models import ProxyConfig

DetachedProfile = Literal["strict", "lenient", "silent"]
Dependency = Literal["none", "optional", "required"]
FeatureState = Literal["full", "degraded", "disabled"]

DETACHED_PROFILE_ENV = "HEADROOM_DETACHED_PROFILE"
_VALID_PROFILES: set[str] = {"strict", "lenient", "silent"}
_REMOTE_BACKENDS = {"redis", "http", "https"}


@dataclass(frozen=True)
class FeatureCapability:
    feature: str
    label: str
    local_state_dependency: Dependency
    state: FeatureState
    enabled: bool
    degradation_mode: str
    reason: str
    backend: str | None = None
    strict_required: bool = False

    @property
    def degraded(self) -> bool:
        return self.state in {"degraded", "disabled"}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "feature": self.feature,
            "label": self.label,
            "local_state_dependency": self.local_state_dependency,
            "state": self.state,
            "enabled": self.enabled,
            "degraded": self.degraded,
            "degradation_mode": self.degradation_mode,
            "reason": self.reason,
            "strict_required": self.strict_required,
        }
        if self.backend is not None:
            payload["backend"] = self.backend
        return payload


@dataclass(frozen=True)
class CapabilityReport:
    detached: bool
    profile: DetachedProfile
    local_state_available: bool
    local_state_reason: str
    workspace_dir: str
    features: tuple[FeatureCapability, ...]

    @property
    def strict_violations(self) -> tuple[FeatureCapability, ...]:
        if self.profile != "strict":
            return ()
        return tuple(
            feature for feature in self.features if feature.strict_required and feature.degraded
        )

    def to_dict(self) -> dict[str, Any]:
        violations = self.strict_violations
        return {
            "detached": self.detached,
            "profile": self.profile,
            "local_state": {
                "available": self.local_state_available,
                "reason": self.local_state_reason,
                "workspace_dir": self.workspace_dir,
            },
            "features": [feature.to_dict() for feature in self.features],
            "strict_violations": [feature.to_dict() for feature in violations],
        }


class DetachedModeError(RuntimeError):
    """Raised when strict detached mode refuses a degraded requested feature."""

    def __init__(self, report: CapabilityReport) -> None:
        features = ", ".join(feature.feature for feature in report.strict_violations)
        super().__init__(f"Detached profile 'strict' refuses degraded feature(s): {features}")
        self.report = report


def normalize_detached_profile(value: str | None = None) -> DetachedProfile:
    raw = (value or os.environ.get(DETACHED_PROFILE_ENV) or "lenient").strip().lower()
    if raw not in _VALID_PROFILES:
        return "lenient"
    return raw  # type: ignore[return-value]


def _probe_local_state(config: ProxyConfig) -> tuple[bool, str]:
    if config.stateless:
        return False, "HEADROOM_STATELESS/--stateless requested no filesystem writes"

    root = paths.workspace_dir()
    probe_path = root / ".headroom-capability-probe"
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe_path.write_text("ok", encoding="utf-8")
        probe_path.unlink(missing_ok=True)
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, "workspace state is writable"


def _backend_from_env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip().lower() or default


def _is_remote_backend(value: str) -> bool:
    return value in _REMOTE_BACKENDS or value.startswith(("redis://", "http://", "https://"))


def build_capability_report(config: ProxyConfig) -> CapabilityReport:
    profile = normalize_detached_profile(config.detached_profile)
    local_state_available, local_state_reason = _probe_local_state(config)
    detached = config.stateless or not local_state_available
    workspace_dir = str(paths.workspace_dir())
    toin_backend = _backend_from_env("HEADROOM_TOIN_BACKEND", "filesystem")
    ccr_backend = _backend_from_env("HEADROOM_CCR_BACKEND", "memory")

    def local_optional(
        *,
        feature: str,
        label: str,
        enabled: bool = True,
        full_reason: str,
        degraded_mode: str,
        disabled_reason: str,
        backend: str | None = None,
        strict_required: bool = False,
    ) -> FeatureCapability:
        if not enabled:
            return FeatureCapability(
                feature=feature,
                label=label,
                local_state_dependency="optional",
                state="disabled",
                enabled=False,
                degradation_mode="disabled",
                reason=disabled_reason,
                backend=backend,
                strict_required=strict_required,
            )
        if local_state_available:
            return FeatureCapability(
                feature=feature,
                label=label,
                local_state_dependency="optional",
                state="full",
                enabled=True,
                degradation_mode="full",
                reason=full_reason,
                backend=backend,
                strict_required=strict_required,
            )
        return FeatureCapability(
            feature=feature,
            label=label,
            local_state_dependency="optional",
            state="degraded",
            enabled=True,
            degradation_mode=degraded_mode,
            reason=local_state_reason,
            backend=backend,
            strict_required=strict_required,
        )

    features: list[FeatureCapability] = [
        FeatureCapability(
            feature="proxy_request_handling",
            label="Proxy request handling",
            local_state_dependency="none",
            state="full",
            enabled=True,
            degradation_mode="full",
            reason="request forwarding and compression routing do not require local state",
        ),
        FeatureCapability(
            feature="compression",
            label="Compression pipeline",
            local_state_dependency="none",
            state="full" if config.optimize else "disabled",
            enabled=config.optimize,
            degradation_mode="full" if config.optimize else "disabled",
            reason=(
                "compression does not require local state"
                if config.optimize
                else "optimization disabled by configuration"
            ),
        ),
        local_optional(
            feature="savings_tracker",
            label="Savings tracker",
            enabled=config.cost_tracking_enabled,
            full_reason="persistent savings ledger is available",
            degraded_mode="memory-only",
            disabled_reason="cost tracking disabled by configuration",
        ),
    ]

    if _is_remote_backend(toin_backend):
        features.append(
            FeatureCapability(
                feature="toin_tagging",
                label="TOIN tagging",
                local_state_dependency="optional",
                state="full",
                enabled=True,
                degradation_mode="remote-backed",
                reason=f"remote TOIN backend configured: {toin_backend}",
                backend=toin_backend,
            )
        )
    else:
        features.append(
            local_optional(
                feature="toin_tagging",
                label="TOIN tagging",
                enabled=toin_backend != "none",
                full_reason="filesystem TOIN backend is available",
                degraded_mode="disabled",
                disabled_reason="TOIN backend disabled",
                backend=toin_backend,
            )
        )

    features.append(
        FeatureCapability(
            feature="ccr_retrieval",
            label="CCR retrieval",
            local_state_dependency="optional",
            state="full" if _is_remote_backend(ccr_backend) else "degraded",
            enabled=True,
            degradation_mode="remote-backed" if _is_remote_backend(ccr_backend) else "memory-only",
            reason=(
                f"remote CCR backend configured: {ccr_backend}"
                if _is_remote_backend(ccr_backend)
                else "CCR store is process-local and will not survive restart"
            ),
            backend=ccr_backend,
        )
    )

    if config.memory_enabled:
        remote_memory = config.memory_backend == "qdrant-neo4j"
        features.append(
            FeatureCapability(
                feature="memory",
                label="Persistent memory",
                local_state_dependency="optional",
                state="full" if (remote_memory or local_state_available) else "disabled",
                enabled=remote_memory or local_state_available,
                degradation_mode=(
                    "remote-backed"
                    if remote_memory
                    else ("full" if local_state_available else "disabled")
                ),
                reason=(
                    "remote memory backend configured"
                    if remote_memory
                    else (
                        "local memory store is available"
                        if local_state_available
                        else local_state_reason
                    )
                ),
                backend=config.memory_backend,
                strict_required=True,
            )
        )
    else:
        features.append(
            FeatureCapability(
                feature="memory",
                label="Persistent memory",
                local_state_dependency="optional",
                state="disabled",
                enabled=False,
                degradation_mode="disabled",
                reason="memory disabled by configuration",
                backend=config.memory_backend,
            )
        )

    features.append(
        FeatureCapability(
            feature="learn_plugins",
            label="Learn plugins",
            local_state_dependency="required",
            state="full"
            if (config.traffic_learning_enabled and local_state_available)
            else "disabled",
            enabled=config.traffic_learning_enabled and local_state_available,
            degradation_mode="full"
            if (config.traffic_learning_enabled and local_state_available)
            else "disabled",
            reason=(
                "traffic learning can write learned patterns"
                if (config.traffic_learning_enabled and local_state_available)
                else (
                    local_state_reason
                    if config.traffic_learning_enabled
                    else "traffic learning disabled by configuration"
                )
            ),
            strict_required=config.traffic_learning_enabled,
        )
    )
    features.append(
        local_optional(
            feature="dashboard_live_data",
            label="Dashboard live data",
            enabled=True,
            full_reason="request log and stats state are available",
            degraded_mode="memory-only",
            disabled_reason="dashboard disabled",
        )
    )
    features.append(
        FeatureCapability(
            feature="session_aggregation",
            label="Session aggregation",
            local_state_dependency="required",
            state="full" if local_state_available else "disabled",
            enabled=local_state_available,
            degradation_mode="full" if local_state_available else "disabled",
            reason=(
                "workspace session stats are available"
                if local_state_available
                else local_state_reason
            ),
        )
    )

    return CapabilityReport(
        detached=detached,
        profile=profile,
        local_state_available=local_state_available,
        local_state_reason=local_state_reason,
        workspace_dir=workspace_dir,
        features=tuple(features),
    )


def enforce_detached_profile(report: CapabilityReport) -> None:
    if report.strict_violations:
        raise DetachedModeError(report)


def render_capability_matrix(report: CapabilityReport) -> str:
    rows = [
        "Detached Capability Matrix:",
        f"  Profile: {report.profile}",
        f"  Local state: {'available' if report.local_state_available else 'unavailable'} ({report.local_state_reason})",
    ]
    for feature in report.features:
        rows.append(
            "  - "
            f"{feature.label}: {feature.state} "
            f"(dependency={feature.local_state_dependency}, degradation={feature.degradation_mode})"
        )
    return "\n".join(rows)
