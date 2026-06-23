"""Shared pytest fixtures for agent-evals."""

from __future__ import annotations

import pytest

from agent_evals.models import ArmName, ArmSpec, Pricing, Provider, ProxyMode


@pytest.fixture
def pricing() -> Pricing:
    """A simple, exact pricing for deterministic cost-derivation tests."""

    return Pricing(input_usd_per_1m=2.0, output_usd_per_1m=10.0)


@pytest.fixture
def three_arms() -> list[ArmSpec]:
    """The canonical Phase-0 three-arm set (Anthropic)."""

    return [
        ArmSpec(
            name=ArmName.A0_DIRECT, provider=Provider.ANTHROPIC, proxy_mode=None, label="direct"
        ),
        ArmSpec(
            name=ArmName.A1_PASSTHROUGH,
            provider=Provider.ANTHROPIC,
            proxy_mode=ProxyMode.OFF,
            label="passthrough",
        ),
        ArmSpec(
            name=ArmName.B_HEADROOM,
            provider=Provider.ANTHROPIC,
            proxy_mode=ProxyMode.TOKEN,
            label="headroom",
        ),
    ]
