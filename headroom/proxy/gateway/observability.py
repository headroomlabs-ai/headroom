"""Bounded, content-free telemetry for the gateway profile."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, TypeAlias, get_args

from headroom.proxy.gateway.usage import CostEvaluation, UsageObservation

if TYPE_CHECKING:
    from headroom.proxy.gateway.admission import AdmissionController

IngressProtocol: TypeAlias = Literal[
    "openai-chat",
    "openai-responses",
    "anthropic-messages",
    "gemini-generate",
    "vertex-generate",
    "bedrock-invoke",
]
RouteClass: TypeAlias = Literal["public-api", "private-compatible", "cloud-workload"]
Adapter: TypeAlias = Literal["strict-native", "routed-native", "translated"]
CredentialSource: TypeAlias = Literal["env", "gcp-adc", "aws-chain", "none"]
FailureOrigin: TypeAlias = Literal["none", "client", "gateway", "identity", "network", "upstream"]
RetryReason: TypeAlias = Literal[
    "none", "connect", "rate-limit", "unavailable", "credential-refresh"
]
TerminalResult: TypeAlias = Literal["success", "rejected", "cancelled", "failed", "unknown"]
GatewayDimensions: TypeAlias = tuple[str, str, str, str, str, str, str]


@dataclass(frozen=True, slots=True)
class GatewayEvent:
    ingress_protocol: IngressProtocol
    route_class: RouteClass
    adapter: Adapter
    credential_source: CredentialSource
    failure_origin: FailureOrigin
    retry_reason: RetryReason
    terminal_result: TerminalResult

    def __post_init__(self) -> None:
        domains = (
            IngressProtocol,
            RouteClass,
            Adapter,
            CredentialSource,
            FailureOrigin,
            RetryReason,
            TerminalResult,
        )
        if any(value not in get_args(domain) for value, domain in zip(self.dimensions(), domains)):
            raise ValueError("invalid gateway metric dimension")

    def dimensions(self) -> GatewayDimensions:
        return (
            self.ingress_protocol,
            self.route_class,
            self.adapter,
            self.credential_source,
            self.failure_origin,
            self.retry_reason,
            self.terminal_result,
        )


class GatewayObservability:
    """Aggregate events without retaining request or response content."""

    def __init__(self) -> None:
        self._counts: Counter[GatewayDimensions] = Counter()
        self._attempt_counts: Counter[GatewayDimensions] = Counter()
        self._totals: Counter[str] = Counter()
        self._admission: AdmissionController | None = None

    def bind_admission(self, admission: AdmissionController) -> None:
        self._admission = admission

    def record(self, event: GatewayEvent) -> None:
        self._counts[event.dimensions()] += 1
        self._totals["logical_requests"] += 1

    def record_attempt(
        self, event: GatewayEvent, usage: UsageObservation, cost: CostEvaluation
    ) -> None:
        if cost.basis not in {"provider_reported", "configured_tariff", "unknown"}:
            raise ValueError("invalid currency basis")
        self._attempt_counts[event.dimensions()] += 1
        self._totals["attempts"] += 1
        self._totals["known_micro_usd"] += cost.known_micro_usd or 0
        self._totals["known_micro_usd_" + cost.basis] += cost.known_micro_usd or 0
        self._totals["unknown_usage_attempts"] += int(not cost.complete)
        self._totals["allowance_observations"] += int(usage.allowance_units is not None)
        for dimension in (
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_create_tokens",
        ):
            count = getattr(usage, dimension)
            if count is not None:
                self._totals[dimension] += count

    def state(self, *, active: int, queued: int, unresolved_micro_usd: int) -> None:
        if min(active, queued, unresolved_micro_usd) < 0:
            raise ValueError("invalid gateway metric gauge")
        self._totals["active"] = active
        self._totals["queued"] = queued
        self._totals["unresolved_micro_usd"] = unresolved_micro_usd

    def totals(self) -> MappingProxyType[str, int]:
        if self._admission is not None:
            self.state(
                active=self._admission.active_count,
                queued=self._admission.queued_count,
                unresolved_micro_usd=self._admission.snapshot().unresolved_micro_usd,
            )
        return MappingProxyType(dict(self._totals))

    def attempt_snapshot(self) -> MappingProxyType[GatewayDimensions, int]:
        return MappingProxyType(dict(self._attempt_counts))

    def snapshot(self) -> MappingProxyType[GatewayDimensions, int]:
        return MappingProxyType(dict(self._counts))
