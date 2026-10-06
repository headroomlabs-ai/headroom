"""Fluent testing harness for Headroom scenarios.

The testing package is intentionally small at import time. It builds real
``HeadroomConfig`` and ``ProxyConfig`` instances and exposes the same surface to
bench runners, smoke tests, and local simulations.
"""

from __future__ import annotations

from .harness import (
    AgentEvalsManifest,
    AgentEvalsPricing,
    ArmName,
    BenchArm,
    BenchManifestFragment,
    Configurator,
    ContractAudit,
    DeploymentHandle,
    FieldContract,
    Guarantee,
    GuaranteeResult,
    HarnessScenario,
    Headroom,
    HeadroomSuite,
    LocalProxyDeployment,
    PlatformTarget,
    ProviderTarget,
    ProxyDeploymentPlan,
    ScenarioCaseResult,
    ScenarioContract,
    ScenarioOrchestrator,
    ScenarioResult,
    ScenarioRunReport,
    ScenarioTask,
    SuiteManifestBundle,
    guarantee_messages_remain_non_empty,
    guarantee_tokens_do_not_increase,
)

__all__ = [
    "AgentEvalsManifest",
    "AgentEvalsPricing",
    "ArmName",
    "BenchArm",
    "BenchManifestFragment",
    "Configurator",
    "ContractAudit",
    "DeploymentHandle",
    "FieldContract",
    "Guarantee",
    "GuaranteeResult",
    "Headroom",
    "HarnessScenario",
    "HeadroomSuite",
    "LocalProxyDeployment",
    "PlatformTarget",
    "ProviderTarget",
    "ProxyDeploymentPlan",
    "ScenarioCaseResult",
    "ScenarioContract",
    "ScenarioOrchestrator",
    "ScenarioResult",
    "ScenarioRunReport",
    "ScenarioTask",
    "SuiteManifestBundle",
    "guarantee_messages_remain_non_empty",
    "guarantee_tokens_do_not_increase",
]
