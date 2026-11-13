"""Closed-by-default provider identity admission catalog."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    admitted: bool
    public_capabilities: tuple[str, ...]
    reason: str


class ProviderAdmissionCatalog:
    """Admit only identity contracts qualified for gateway v1."""

    _ADMITTED = frozenset(
        {
            ("openai", "api-key"),
            ("anthropic", "api-key"),
            ("gemini", "api-key"),
            ("vertex", "workload"),
            ("bedrock", "workload"),
            ("compatible", "api-key"),
        }
    )

    def check(
        self,
        provider: str,
        identity_kind: str,
        product_class: str,
        audience: str,
    ) -> AdmissionDecision:
        if audience != "inference":
            return AdmissionDecision(False, (), "audience_not_admitted")
        if product_class not in {"public-api", "cloud-workload"}:
            return AdmissionDecision(False, (), "product_not_admitted")
        if (provider, identity_kind) not in self._ADMITTED:
            return AdmissionDecision(False, (), "identity_not_admitted")
        return AdmissionDecision(True, ("inference",), "admitted")

    def discover(self) -> tuple[tuple[str, str], ...]:
        """Return metadata only; discovery never reads or imports credentials."""
        return tuple(sorted(self._ADMITTED))
