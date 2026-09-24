"""Deterministic account selection and conservative retry classification."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from enum import Enum
from typing import NoReturn

from headroom.proxy.gateway.config import CredentialConfig, RetryConfig, RouteConfig
from headroom.proxy.gateway.context import GatewayPrincipal
from headroom.proxy.gateway.errors import GatewayCredentialUnavailable
from headroom.proxy.gateway.resources import ResourceBinding


@dataclass(frozen=True, slots=True)
class AccountSelection:
    account_ref: str
    sticky: bool


@dataclass(frozen=True, slots=True)
class TransportFailure:
    kind: str
    retry_after: float | str | None


@dataclass(frozen=True, slots=True)
class ProviderContract:
    max_attempts: int
    retryable_failures: frozenset[str]
    max_retry_after: float = 0.0
    rejected_failures: frozenset[str] = frozenset()


class ExposureState(str, Enum):
    UNSENT = "unsent"
    PROVEN_REJECTED = "proven_rejected"
    ACCEPTANCE_UNKNOWN = "acceptance_unknown"
    ACCEPTED = "accepted"
    OUTPUT_EXPOSED = "output_exposed"


@dataclass(frozen=True, slots=True)
class RetryDecision:
    allowed: bool
    reason: str
    delay: float = 0

    @classmethod
    def decide(
        cls,
        failure: TransportFailure,
        exposure: str,
        provider_contract: ProviderContract,
        *,
        attempt_count: int,
        deadline: float,
        now: float,
        policy: RetryConfig,
        wall_time: float | None = None,
    ) -> RetryDecision:
        if exposure not in {"unsent", "proven_rejected"}:
            return cls(False, "exposure")
        if (
            exposure == "proven_rejected"
            and failure.kind not in provider_contract.rejected_failures
        ):
            return cls(False, "contract")
        if failure.kind not in provider_contract.retryable_failures:
            return cls(False, "contract")
        if attempt_count < 1 or attempt_count >= min(
            policy.max_attempts, provider_contract.max_attempts, 3
        ):
            return cls(False, "attempt_limit")
        delay = max(0.001, policy.base_backoff_seconds * 2 ** (attempt_count - 1))
        if failure.retry_after is not None:
            try:
                raw = failure.retry_after
                if isinstance(raw, str):
                    try:
                        retry_after = float(raw)
                    except ValueError:
                        parsed = parsedate_to_datetime(raw)
                        if parsed.tzinfo is None:
                            return cls(False, "retry_after")
                        retry_after = max(
                            0,
                            parsed.timestamp() - (time.time() if wall_time is None else wall_time),
                        )
                else:
                    retry_after = raw
            except (ValueError, TypeError, OverflowError):
                return cls(False, "retry_after")
            if (
                not math.isfinite(retry_after)
                or retry_after < 0
                or retry_after
                > min(policy.max_retry_after_seconds, provider_contract.max_retry_after)
            ):
                return cls(False, "retry_after")
            delay = max(delay, retry_after)
        if not math.isfinite(deadline) or not math.isfinite(now) or delay >= deadline - now:
            return cls(False, "deadline")
        return cls(
            True,
            {"connect": "connect", "rate_limit": "rate-limit", "unavailable": "unavailable"}.get(
                failure.kind, "unavailable"
            ),
            delay,
        )

    @staticmethod
    def classify(
        failure: TransportFailure,
        exposure: str,
        provider_contract: ProviderContract,
    ) -> bool:
        if provider_contract.max_attempts < 2 or exposure != "none":
            return False
        if failure.kind not in provider_contract.retryable_failures:
            return False
        return failure.retry_after is None or (
            isinstance(failure.retry_after, (int, float))
            and math.isfinite(failure.retry_after)
            and 0 <= failure.retry_after <= provider_contract.max_retry_after
        )


class AccountRouter:
    _credentials: dict[str, CredentialConfig] | None

    def update_accounts(
        self,
        available_accounts: set[str],
        identities: dict[str, str] | None = None,
        *,
        credentials: dict[str, CredentialConfig] | None = None,
    ) -> None:
        self._available = frozenset(available_accounts)
        self._identities = identities or {}
        self._credentials = credentials

    def __init__(self, *, available_accounts: set[str]) -> None:
        self._available = frozenset(available_accounts)
        self._identities = {}
        self._credentials = None
        self._positions: dict[tuple[str, str], int] = {}
        self._cooldowns: dict[tuple[str, str], float] = {}

    def cool_down(self, account_ref: str, *, quota_key: str, until: float) -> None:
        self._cooldowns[(self._identities.get(account_ref, account_ref), quota_key)] = until

    def select(
        self,
        route: RouteConfig,
        principal: GatewayPrincipal,
        resource_binding: ResourceBinding | None = None,
        *,
        now: float | None = None,
        eligible_accounts: frozenset[str] | None = None,
        authority_keys: dict[str, str] | None = None,
        target_key: str | None = None,
    ) -> AccountSelection:
        if (
            not route.enabled
            or route.id not in principal.routes
            or "inference" not in principal.scopes
        ):
            self._unavailable()
        available = (
            self._available if eligible_accounts is None else self._available & eligible_accounts
        )
        identities = self._identities if authority_keys is None else authority_keys
        current = time.time() if now is None else now
        quota = route.selection.quota_group or route.id
        anchor = self._credentials.get(route.credentials[0]) if self._credentials else None

        def equivalent(account: str) -> bool:
            if self._credentials is None:
                return True
            candidate = self._credentials.get(account)
            if candidate is None or anchor is None or not candidate.enabled:
                return False
            if (
                route.selection.required_residency is not None
                and candidate.residency != route.selection.required_residency
            ):
                return False
            return (
                candidate.provider,
                candidate.owner_group or candidate.id,
                candidate.billing_group or candidate.id,
                candidate.residency,
            ) == (
                anchor.provider,
                anchor.owner_group or anchor.id,
                anchor.billing_group or anchor.id,
                anchor.residency,
            )

        candidates = [
            account
            for account in route.credentials
            if account in available
            and equivalent(account)
            and self._cooldowns.get((identities.get(account, account), quota), 0) <= current
        ]
        if resource_binding is not None:
            if (
                resource_binding.principal_id != principal.id
                or resource_binding.route_id != route.id
                or resource_binding.account_ref not in candidates
                or resource_binding.expires_at is not None
                and resource_binding.expires_at <= current
                or resource_binding.authority_fingerprint is not None
                and resource_binding.authority_fingerprint
                != identities.get(resource_binding.account_ref)
                or resource_binding.target_fingerprint is not None
                and resource_binding.target_fingerprint != target_key
            ):
                self._unavailable()
            return AccountSelection(resource_binding.account_ref, sticky=True)
        if not candidates:
            self._unavailable()
        strategy = route.selection.strategy
        if strategy == "priority":
            account = max(candidates, key=lambda key: route.selection.priority.get(key, 0))
        else:
            cycle = [
                account
                for account in candidates
                for _ in range(
                    route.selection.weight.get(account, 1) if strategy == "weighted" else 1
                )
            ]
            key = (principal.id, route.id)
            position = self._positions.get(key, 0)
            account = cycle[position % len(cycle)]
            self._positions[key] = (position + 1) % len(cycle)
        return AccountSelection(account, sticky=False)

    @staticmethod
    def _unavailable() -> NoReturn:
        raise GatewayCredentialUnavailable(
            status_code=503,
            code="credential_unavailable",
            message="No eligible gateway account is available",
        )
