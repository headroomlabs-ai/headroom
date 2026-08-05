"""Authoritative Headroom test-harness contract.

This module gives tests and external benches one fluent API for configuring the
whole Headroom surface area. It does not mirror a hand-maintained subset:
``HeadroomConfig`` and ``ProxyConfig`` are instantiated directly, and their
dataclass fields are exposed through ``ScenarioContract`` so drift is visible to
tests and downstream harnesses.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import MISSING, asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, cast

from headroom._subprocess import run as subprocess_run
from headroom.config import HeadroomConfig, HeadroomMode
from headroom.proxy.models import ProxyConfig

ConfigCallback = Callable[["Configurator"], None]
ProxyCallback = Callable[[ProxyConfig], None]
SdkCallback = Callable[[HeadroomConfig], None]
Guarantee = Callable[["HarnessScenario", "ScenarioTask", "ScenarioResult"], "GuaranteeResult"]


class ProviderTarget(str, Enum):
    """Provider/backend targets that Headroom currently exposes through the proxy."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    CLOUDCODE = "cloudcode"
    VERTEX = "vertex"
    BEDROCK = "bedrock"
    ANYLLM = "anyllm"
    LITELLM = "litellm"


class PlatformTarget(str, Enum):
    """Execution platform metadata for reproducible test scenarios."""

    LOCAL = "local"
    APPLE_SILICON = "apple_silicon"
    LINUX_X86_64 = "linux_x86_64"
    CONTAINER = "container"


class ArmName(str, Enum):
    """Bench arm names used by headroom-bench."""

    A0_DIRECT = "a0_direct"
    A1_PASSTHROUGH = "a1_passthrough"
    B_HEADROOM = "b_headroom"
    B_ABLATE = "b_ablate"


@dataclass(frozen=True)
class FieldContract:
    """One dataclass field in the Headroom contract."""

    owner: Literal["headroom", "proxy"]
    name: str
    type_repr: str
    has_default: bool
    default_repr: str | None


@dataclass(frozen=True)
class ScenarioContract:
    """The auditable configuration surface for a built scenario."""

    headroom_fields: tuple[FieldContract, ...]
    proxy_fields: tuple[FieldContract, ...]

    def field_names(self, owner: Literal["headroom", "proxy"]) -> set[str]:
        source = self.headroom_fields if owner == "headroom" else self.proxy_fields
        return {field.name for field in source}


@dataclass(frozen=True)
class ContractAudit:
    """Machine-readable audit of a scenario against current Headroom contracts."""

    scenario: str
    provider: str
    platform: str
    headroom_fields_total: int
    proxy_fields_total: int
    headroom_payload_fields: tuple[str, ...]
    proxy_payload_fields: tuple[str, ...]
    missing_headroom_payload_fields: tuple[str, ...]
    missing_proxy_payload_fields: tuple[str, ...]
    extra_headroom_payload_fields: tuple[str, ...]
    extra_proxy_payload_fields: tuple[str, ...]
    notes: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not (
            self.missing_headroom_payload_fields
            or self.missing_proxy_payload_fields
            or self.extra_headroom_payload_fields
            or self.extra_proxy_payload_fields
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "provider": self.provider,
            "platform": self.platform,
            "passed": self.passed,
            "headroom_fields_total": self.headroom_fields_total,
            "proxy_fields_total": self.proxy_fields_total,
            "headroom_payload_fields": list(self.headroom_payload_fields),
            "proxy_payload_fields": list(self.proxy_payload_fields),
            "missing_headroom_payload_fields": list(self.missing_headroom_payload_fields),
            "missing_proxy_payload_fields": list(self.missing_proxy_payload_fields),
            "extra_headroom_payload_fields": list(self.extra_headroom_payload_fields),
            "extra_proxy_payload_fields": list(self.extra_proxy_payload_fields),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class BenchArm:
    """Bench-ready arm specification compatible with headroom-bench's model."""

    name: ArmName
    provider: Literal["anthropic", "openai"]
    proxy_mode: Literal["off", "token", "cache"] | None
    proxy_flags: tuple[str, ...]
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "provider": self.provider,
            "proxy_mode": self.proxy_mode,
            "proxy_flags": list(self.proxy_flags),
            "label": self.label,
        }


@dataclass(frozen=True)
class BenchManifestFragment:
    """Portable scenario metadata a bench repo can merge into its RunManifest."""

    harness: str
    harness_version: str
    provider: str
    platform: str
    arms: tuple[BenchArm, ...]
    proxy_config: dict[str, Any]
    headroom_config: dict[str, Any]
    env: dict[str, str]
    deployment_plan: dict[str, Any]
    contract_audit: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "harness": self.harness,
            "harness_version": self.harness_version,
            "provider": self.provider,
            "platform": self.platform,
            "arms": [arm.to_dict() for arm in self.arms],
            "proxy_config": dict(self.proxy_config),
            "headroom_config": dict(self.headroom_config),
            "env": dict(self.env),
            "deployment_plan": dict(self.deployment_plan),
            "contract_audit": dict(self.contract_audit),
        }


@dataclass(frozen=True)
class AgentEvalsPricing:
    """Pricing block matching headroom-bench's ``Pricing`` model."""

    input_usd_per_1m: float = 3.0
    output_usd_per_1m: float = 15.0

    def to_dict(self) -> dict[str, float]:
        return {
            "input_usd_per_1m": self.input_usd_per_1m,
            "output_usd_per_1m": self.output_usd_per_1m,
        }


@dataclass(frozen=True)
class AgentEvalsManifest:
    """Bench-native manifest shape compatible with ``agent_evals.models.RunManifest``."""

    experiment_id: str
    created_at: datetime
    headroom_git_sha: str
    agent_evals_git_sha: str
    model_snapshot: str
    provider: Literal["anthropic", "openai"]
    auth_mode: str
    benchmark: str
    benchmark_ref: str
    harness: str
    harness_version: str
    docker_digests: dict[str, str]
    arms: tuple[BenchArm, ...]
    k_runs: int
    temperature: float
    seeds: tuple[int, ...]
    alpha: float
    margins: dict[str, float]
    pricing: AgentEvalsPricing

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "created_at": self.created_at.isoformat(),
            "headroom_git_sha": self.headroom_git_sha,
            "agent_evals_git_sha": self.agent_evals_git_sha,
            "model_snapshot": self.model_snapshot,
            "provider": self.provider,
            "auth_mode": self.auth_mode,
            "benchmark": self.benchmark,
            "benchmark_ref": self.benchmark_ref,
            "harness": self.harness,
            "harness_version": self.harness_version,
            "docker_digests": dict(self.docker_digests),
            "arms": [arm.to_dict() for arm in self.arms],
            "k_runs": self.k_runs,
            "temperature": self.temperature,
            "seeds": list(self.seeds),
            "alpha": self.alpha,
            "margins": dict(self.margins),
            "pricing": self.pricing.to_dict(),
        }


@dataclass(frozen=True)
class SuiteManifestBundle:
    """JSON-ready bundle for a group of Headroom scenarios."""

    name: str
    harness: str
    harness_version: str
    scenarios: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "harness": self.harness,
            "harness_version": self.harness_version,
            "scenarios": [dict(scenario) for scenario in self.scenarios],
        }


@dataclass(frozen=True)
class ScenarioResult:
    """Dry-run result produced without upstream API keys."""

    tokens_before: int
    tokens_after: int
    tokens_saved: int
    transforms: tuple[str, ...]
    messages: list[dict[str, Any]]


@dataclass(frozen=True)
class ScenarioTask:
    """One no-key local simulation task for a Headroom scenario."""

    task_id: str
    messages: list[dict[str, Any]]
    model: str = "gpt-4o"
    provider: Literal["openai", "anthropic"] = "openai"
    output_buffer_tokens: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class GuaranteeResult:
    """Verdict for one scenario guarantee."""

    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class ScenarioCaseResult:
    """Result for one scenario x task cell."""

    scenario: str
    task_id: str
    result: ScenarioResult
    guarantees: tuple[GuaranteeResult, ...]

    @property
    def passed(self) -> bool:
        return all(guarantee.passed for guarantee in self.guarantees)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "task_id": self.task_id,
            "passed": self.passed,
            "result": {
                "tokens_before": self.result.tokens_before,
                "tokens_after": self.result.tokens_after,
                "tokens_saved": self.result.tokens_saved,
                "transforms": list(self.result.transforms),
                "messages": self.result.messages,
            },
            "guarantees": [guarantee.to_dict() for guarantee in self.guarantees],
        }


@dataclass(frozen=True)
class ScenarioRunReport:
    """Deterministic report for an orchestrated no-key scenario run."""

    cases: tuple[ScenarioCaseResult, ...]

    @property
    def passed(self) -> bool:
        return all(case.passed for case in self.cases)

    @property
    def total_tokens_before(self) -> int:
        return sum(case.result.tokens_before for case in self.cases)

    @property
    def total_tokens_after(self) -> int:
        return sum(case.result.tokens_after for case in self.cases)

    @property
    def total_tokens_saved(self) -> int:
        return self.total_tokens_before - self.total_tokens_after

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "total_cases": len(self.cases),
            "total_tokens_before": self.total_tokens_before,
            "total_tokens_after": self.total_tokens_after,
            "total_tokens_saved": self.total_tokens_saved,
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True)
class DeploymentHandle:
    """Live local proxy metadata."""

    base_url: str
    env: dict[str, str]
    command: tuple[str, ...]
    process_id: int


@dataclass(frozen=True)
class ProxyDeploymentPlan:
    """Concrete proxy deployment inputs for local runners and headroom-bench."""

    command: tuple[str, ...]
    env: dict[str, str]
    config_payload: dict[str, Any]
    config_env_var: str = "HEADROOM_PROXY_CONFIG_JSON"

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": list(self.command),
            "env": dict(self.env),
            "config_payload": dict(self.config_payload),
            "config_env_var": self.config_env_var,
        }

    def validate(self) -> None:
        """Fail if the environment does not round-trip the full config payload."""

        raw = self.env.get(self.config_env_var)
        if raw is None:
            raise ValueError(f"deployment env missing {self.config_env_var}")
        parsed = json.loads(raw)
        if parsed != self.config_payload:
            raise ValueError(f"{self.config_env_var} does not match config_payload")


def _field_contract(
    owner: Literal["headroom", "proxy"], cls: type[Any]
) -> tuple[FieldContract, ...]:
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} must be a dataclass")
    out: list[FieldContract] = []
    for field in cls.__dataclass_fields__.values():
        default: Any = MISSING
        if field.default is not MISSING:
            default = field.default
        elif field.default_factory is not MISSING:  # type: ignore[attr-defined]
            default = "<factory>"
        out.append(
            FieldContract(
                owner=owner,
                name=field.name,
                type_repr=str(field.type),
                has_default=default is not MISSING,
                default_repr=None if default is MISSING else repr(default),
            )
        )
    return tuple(out)


def _public_dataclass_dict(value: Any) -> dict[str, Any]:
    raw = asdict(value)
    out = {key: _portable_value(val) for key, val in raw.items() if not key.startswith("_")}
    for name, field in value.__dataclass_fields__.items():
        if name in out or name.startswith("_"):
            continue
        default = None if field.default is MISSING else field.default
        out[name] = _portable_value(default)
    return out


def _portable_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _portable_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _portable_value(val) for key, val in value.items()}
    if isinstance(value, set | frozenset):
        return sorted(_portable_value(item) for item in value)
    if isinstance(value, tuple | list):
        return [_portable_value(item) for item in value]
    return repr(value)


def _git_sha(repo_path: str | Path) -> str:
    try:
        proc = subprocess_run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return proc.stdout.strip() or "unknown"


def _coerce_headroom_mode(value: str | HeadroomMode) -> HeadroomMode:
    if isinstance(value, HeadroomMode):
        return value
    return HeadroomMode(value)


def guarantee_tokens_do_not_increase(
    scenario: HarnessScenario,
    task: ScenarioTask,
    result: ScenarioResult,
) -> GuaranteeResult:
    """Guarantee that local optimization never grows the input token count."""

    passed = result.tokens_after <= result.tokens_before
    return GuaranteeResult(
        name="tokens_do_not_increase",
        passed=passed,
        detail=(
            f"{scenario.name}/{task.task_id}: "
            f"{result.tokens_before} -> {result.tokens_after} tokens"
        ),
    )


def guarantee_messages_remain_non_empty(
    scenario: HarnessScenario,
    task: ScenarioTask,
    result: ScenarioResult,
) -> GuaranteeResult:
    """Guarantee that optimization keeps a non-empty message sequence."""

    passed = len(result.messages) > 0
    return GuaranteeResult(
        name="messages_remain_non_empty",
        passed=passed,
        detail=f"{scenario.name}/{task.task_id}: {len(result.messages)} messages",
    )


DEFAULT_GUARANTEES: tuple[Guarantee, ...] = (
    guarantee_tokens_do_not_increase,
    guarantee_messages_remain_non_empty,
)


class Configurator:
    """Mutation facade used by ``Headroom.configure`` callbacks.

    Unknown fields fail fast. Known lower-level dataclass fields can be written
    via ``proxy.<field>`` / ``headroom.<field>``, while common cross-surface
    concepts are exposed as concise properties.
    """

    headroom: HeadroomConfig
    proxy: ProxyConfig

    def __init__(self, headroom: HeadroomConfig, proxy: ProxyConfig) -> None:
        object.__setattr__(self, "headroom", headroom)
        object.__setattr__(self, "proxy", proxy)

    @property
    def kompress_enabled(self) -> bool:
        return not self.proxy.disable_kompress

    @kompress_enabled.setter
    def kompress_enabled(self, value: bool) -> None:
        self.proxy.disable_kompress = not bool(value)

    @property
    def optimize(self) -> bool:
        return cast(bool, self.proxy.optimize)

    @optimize.setter
    def optimize(self, value: bool) -> None:
        self.proxy.optimize = bool(value)

    @property
    def mode(self) -> str:
        return cast(str, self.proxy.mode)

    @mode.setter
    def mode(self, value: Literal["token", "cache"]) -> None:
        if value not in {"token", "cache"}:
            raise ValueError("mode must be 'token' or 'cache'")
        self.proxy.mode = value

    @property
    def default_mode(self) -> HeadroomMode:
        return self.headroom.default_mode

    @default_mode.setter
    def default_mode(self, value: str | HeadroomMode) -> None:
        self.headroom.default_mode = _coerce_headroom_mode(value)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"headroom", "proxy"}:
            raise AttributeError(f"{name} is read-only; mutate its fields instead")
        descriptor = getattr(type(self), name, None)
        if isinstance(descriptor, property) and descriptor.fset is not None:
            descriptor.fset(self, value)
            return
        if hasattr(self.proxy, name):
            setattr(self.proxy, name, value)
            return
        if hasattr(self.headroom, name):
            setattr(self.headroom, name, value)
            return
        raise AttributeError(f"unknown Headroom harness config field: {name}")


class Headroom:
    """Fluent scenario builder.

    Example:
        ``Headroom.with_bedrock(region="us-east-1").on_apple_silicon().configure(...)``
    """

    HARNESS_VERSION = "1"

    def __init__(
        self,
        *,
        name: str = "headroom-scenario",
        provider: ProviderTarget = ProviderTarget.ANTHROPIC,
        platform: PlatformTarget = PlatformTarget.LOCAL,
        headroom_config: HeadroomConfig | None = None,
        proxy_config: ProxyConfig | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._name = name
        self._provider = provider
        self._platform = platform
        self._headroom_config = headroom_config or HeadroomConfig()
        self._proxy_config = proxy_config or ProxyConfig()
        self._metadata = dict(metadata or {})

    @classmethod
    def scenario(cls, name: str = "headroom-scenario") -> Headroom:
        return cls(name=name)

    @classmethod
    def suite(cls, name: str = "headroom-suite") -> HeadroomSuite:
        return HeadroomSuite(name=name)

    @classmethod
    def with_bedrock(cls, **config: Any) -> Headroom:
        return cls.scenario().with_provider(ProviderTarget.BEDROCK, **config)

    @classmethod
    def with_openai(cls, **config: Any) -> Headroom:
        return cls.scenario().with_provider(ProviderTarget.OPENAI, **config)

    @classmethod
    def with_anthropic(cls, **config: Any) -> Headroom:
        return cls.scenario().with_provider(ProviderTarget.ANTHROPIC, **config)

    @classmethod
    def with_vertex(cls, **config: Any) -> Headroom:
        return cls.scenario().with_provider(ProviderTarget.VERTEX, **config)

    @classmethod
    def with_gemini(cls, **config: Any) -> Headroom:
        return cls.scenario().with_provider(ProviderTarget.GEMINI, **config)

    @classmethod
    def with_cloudcode(cls, **config: Any) -> Headroom:
        return cls.scenario().with_provider(ProviderTarget.CLOUDCODE, **config)

    @classmethod
    def with_anyllm(cls, **config: Any) -> Headroom:
        return cls.scenario().with_provider(ProviderTarget.ANYLLM, **config)

    @classmethod
    def with_litellm(cls, **config: Any) -> Headroom:
        return cls.scenario().with_provider(ProviderTarget.LITELLM, **config)

    # PascalCase aliases mirror the shape requested by downstream .NET-style examples.
    WithBedrock = with_bedrock
    WithOpenAI = with_openai
    WithAnthropic = with_anthropic
    WithVertex = with_vertex
    WithGemini = with_gemini
    WithCloudCode = with_cloudcode
    WithAnyLLM = with_anyllm
    WithLiteLLM = with_litellm

    def named(self, name: str) -> Headroom:
        self._name = name
        return self

    def with_provider(self, target: ProviderTarget | str, **config: Any) -> Headroom:
        target = ProviderTarget(target)
        self._provider = target
        if target is ProviderTarget.BEDROCK:
            self._proxy_config.backend = "bedrock"
            if "region" in config:
                self._proxy_config.bedrock_region = str(config["region"])
            if "profile" in config:
                self._proxy_config.bedrock_profile = str(config["profile"])
            if "api_url" in config:
                self._proxy_config.bedrock_api_url = str(config["api_url"])
        elif target is ProviderTarget.VERTEX:
            self._proxy_config.backend = "litellm-vertex"
            if "api_url" in config:
                self._proxy_config.vertex_api_url = str(config["api_url"])
        elif target is ProviderTarget.OPENAI:
            self._proxy_config.backend = "anthropic"
            if "api_url" in config:
                self._proxy_config.openai_api_url = str(config["api_url"])
        elif target is ProviderTarget.ANTHROPIC:
            self._proxy_config.backend = "anthropic"
            if "api_url" in config:
                self._proxy_config.anthropic_api_url = str(config["api_url"])
        elif target is ProviderTarget.GEMINI:
            if "api_url" in config:
                self._proxy_config.gemini_api_url = str(config["api_url"])
        elif target is ProviderTarget.CLOUDCODE:
            if "api_url" in config:
                self._proxy_config.cloudcode_api_url = str(config["api_url"])
        elif target is ProviderTarget.ANYLLM:
            self._proxy_config.backend = "anyllm"
            self._proxy_config.anyllm_provider = str(config.get("provider", "openai"))
        elif target is ProviderTarget.LITELLM:
            litellm_provider = str(config.get("provider", "openai"))
            self._proxy_config.backend = f"litellm-{litellm_provider}"
        self._metadata.setdefault("provider_config", {}).update(config)
        return self

    WithProvider = with_provider

    def on_apple_silicon(self, **config: Any) -> Headroom:
        self._platform = PlatformTarget.APPLE_SILICON
        self._metadata.setdefault("platform_config", {}).update(config)
        return self

    OnAppleSilicon = on_apple_silicon

    def on_platform(self, platform: PlatformTarget | str, **config: Any) -> Headroom:
        self._platform = PlatformTarget(platform)
        self._metadata.setdefault("platform_config", {}).update(config)
        return self

    OnPlatform = on_platform

    def configure(self, callback: ConfigCallback | None = None, **overrides: Any) -> Headroom:
        configurator = Configurator(self._headroom_config, self._proxy_config)
        if callback is not None:
            callback(configurator)
        for key, value in overrides.items():
            setattr(configurator, key, value)
        return self

    Configure = configure

    def configure_proxy(self, callback: ProxyCallback | None = None, **overrides: Any) -> Headroom:
        if callback is not None:
            callback(self._proxy_config)
        for key, value in overrides.items():
            if not hasattr(self._proxy_config, key):
                raise AttributeError(f"unknown ProxyConfig field: {key}")
            setattr(self._proxy_config, key, value)
        return self

    ConfigureProxy = configure_proxy

    def configure_headroom(self, callback: SdkCallback | None = None, **overrides: Any) -> Headroom:
        if callback is not None:
            callback(self._headroom_config)
        for key, value in overrides.items():
            if not hasattr(self._headroom_config, key):
                raise AttributeError(f"unknown HeadroomConfig field: {key}")
            setattr(self._headroom_config, key, value)
        return self

    ConfigureHeadroom = configure_headroom

    def with_compression(
        self,
        *,
        mode: Literal["token", "cache"] | None = None,
        kompress: bool | None = None,
        force_kompress_all: bool | None = None,
        lossless: bool | None = None,
        compressors: set[str] | Sequence[str] | Literal["*"] | None = None,
        min_tokens: int | None = None,
        max_items: int | None = None,
        savings_profile: str | None = None,
    ) -> Headroom:
        """Configure proxy and SDK compression posture with real config fields."""

        if mode is not None:
            self._proxy_config.mode = mode
        if kompress is not None:
            self._proxy_config.disable_kompress = not kompress
        if force_kompress_all is not None:
            self._proxy_config.force_kompress_all = force_kompress_all
        if lossless is not None:
            self._proxy_config.lossless = lossless
            self._headroom_config.smart_crusher.lossless_only = lossless
        if compressors is not None:
            if compressors == "*":
                self._proxy_config.compressors = {"*"}
            else:
                self._proxy_config.compressors = set(compressors)
        if min_tokens is not None:
            self._proxy_config.min_tokens_to_crush = min_tokens
            self._headroom_config.smart_crusher.min_tokens_to_crush = min_tokens
        if max_items is not None:
            self._proxy_config.max_items_after_crush = max_items
            self._headroom_config.smart_crusher.max_items_after_crush = max_items
        if savings_profile is not None:
            self._proxy_config.savings_profile = savings_profile
        return self

    WithCompression = with_compression

    def with_ccr(
        self,
        *,
        enabled: bool = True,
        inject_tool: bool | None = None,
        inject_marker: bool | None = None,
        handle_responses: bool | None = None,
        proactive_expansion: bool | None = None,
        max_retrieval_rounds: int | None = None,
    ) -> Headroom:
        """Configure Compress-Cache-Retrieve across SDK and proxy surfaces."""

        self._headroom_config.ccr.enabled = enabled
        self._proxy_config.ccr_inject_tool = enabled if inject_tool is None else inject_tool
        self._proxy_config.ccr_inject_marker = enabled if inject_marker is None else inject_marker
        if inject_tool is not None:
            self._headroom_config.ccr.inject_tool = inject_tool
        if inject_marker is not None:
            self._headroom_config.ccr.inject_retrieval_marker = inject_marker
        if handle_responses is not None:
            self._proxy_config.ccr_handle_responses = handle_responses
        if proactive_expansion is not None:
            self._proxy_config.ccr_proactive_expansion = proactive_expansion
        if max_retrieval_rounds is not None:
            self._proxy_config.ccr_max_retrieval_rounds = max_retrieval_rounds
        return self

    WithCCR = with_ccr

    def with_cache(
        self,
        *,
        enabled: bool = True,
        semantic: bool | None = None,
        ttl_seconds: int | None = None,
        max_entries: int | None = None,
    ) -> Headroom:
        """Configure proxy cache and SDK cache optimizer knobs."""

        self._proxy_config.cache_enabled = enabled
        self._headroom_config.cache_optimizer.enabled = enabled
        if semantic is not None:
            self._headroom_config.cache_optimizer.enable_semantic_cache = semantic
        if ttl_seconds is not None:
            self._proxy_config.cache_ttl_seconds = ttl_seconds
            self._headroom_config.cache_optimizer.semantic_cache_ttl_seconds = ttl_seconds
        if max_entries is not None:
            self._proxy_config.cache_max_entries = max_entries
            self._headroom_config.cache_optimizer.semantic_cache_max_entries = max_entries
        return self

    WithCache = with_cache

    def with_prefix_freeze(
        self,
        *,
        enabled: bool = True,
        session_ttl_seconds: int | None = None,
    ) -> Headroom:
        """Configure cache-aware prefix freezing for proxy and SDK paths."""

        self._proxy_config.prefix_freeze_enabled = enabled
        self._headroom_config.prefix_freeze.enabled = enabled
        if session_ttl_seconds is not None:
            self._proxy_config.prefix_freeze_session_ttl = session_ttl_seconds
            self._headroom_config.prefix_freeze.session_ttl_seconds = session_ttl_seconds
        return self

    WithPrefixFreeze = with_prefix_freeze

    def with_read_maturation(
        self,
        *,
        enabled: bool = True,
        quiesce_turns: int | None = None,
        max_hold_turns: int | None = None,
        min_size_bytes: int | None = None,
    ) -> Headroom:
        """Configure activity-based Read maturation on the proxy surface."""

        self._proxy_config.read_maturation = enabled
        read_maturation_metadata = self._metadata.setdefault("read_maturation", {})
        read_maturation_metadata["enabled"] = enabled
        if quiesce_turns is not None:
            self._proxy_config.read_maturation_quiesce_turns = quiesce_turns
            read_maturation_metadata["quiesce_turns"] = quiesce_turns
        if max_hold_turns is not None:
            self._proxy_config.read_maturation_max_hold_turns = max_hold_turns
            read_maturation_metadata["max_hold_turns"] = max_hold_turns
        if min_size_bytes is not None:
            self._proxy_config.read_maturation_min_size_bytes = min_size_bytes
            read_maturation_metadata["min_size_bytes"] = min_size_bytes
        return self

    WithReadMaturation = with_read_maturation

    def with_memory(
        self,
        *,
        enabled: bool = True,
        backend: Literal["local", "qdrant-neo4j"] | None = None,
        mode: Literal["auto_tail", "tool"] | None = None,
        top_k: int | None = None,
        min_similarity: float | None = None,
        inject_tools: bool | None = None,
        inject_context: bool | None = None,
        storage_mode: Literal["project", "user", "global"] | None = None,
    ) -> Headroom:
        """Configure the proxy memory subsystem for agent scenarios."""

        self._proxy_config.memory_enabled = enabled
        if backend is not None:
            self._proxy_config.memory_backend = backend
        if mode is not None:
            self._proxy_config.memory_mode = mode
        if top_k is not None:
            self._proxy_config.memory_top_k = top_k
        if min_similarity is not None:
            self._proxy_config.memory_min_similarity = min_similarity
        if inject_tools is not None:
            self._proxy_config.memory_inject_tools = inject_tools
        if inject_context is not None:
            self._proxy_config.memory_inject_context = inject_context
        if storage_mode is not None:
            self._proxy_config.memory_storage_mode = storage_mode
        return self

    WithMemory = with_memory

    def build(self) -> HarnessScenario:
        return HarnessScenario(
            name=self._name,
            provider=self._provider,
            platform=self._platform,
            headroom_config=replace(self._headroom_config),
            proxy_config=replace(self._proxy_config),
            metadata=dict(self._metadata),
        )

    Build = build
    Suite = suite


class HeadroomSuite:
    """Declarative matrix of built scenarios for bench and local no-key runs."""

    def __init__(self, *, name: str = "headroom-suite") -> None:
        self.name = name
        self._scenarios: list[HarnessScenario] = []

    @property
    def scenarios(self) -> tuple[HarnessScenario, ...]:
        return tuple(self._scenarios)

    def add(self, scenario: HarnessScenario | Headroom) -> HeadroomSuite:
        built = scenario.Build() if isinstance(scenario, Headroom) else scenario
        if any(existing.name == built.name for existing in self._scenarios):
            raise ValueError(f"duplicate scenario name in suite: {built.name}")
        self._scenarios.append(built)
        return self

    Add = add

    def extend(self, scenarios: Sequence[HarnessScenario | Headroom]) -> HeadroomSuite:
        for scenario in scenarios:
            self.add(scenario)
        return self

    Extend = extend

    def orchestrate(
        self,
        tasks: Sequence[ScenarioTask],
        *,
        guarantees: Sequence[Guarantee] = DEFAULT_GUARANTEES,
    ) -> ScenarioRunReport:
        return ScenarioOrchestrator(self._require_scenarios(), guarantees=guarantees).run(tasks)

    Orchestrate = orchestrate

    def deployment_plans(
        self,
        *,
        port_start: int = 8787,
        headroom_cmd: Sequence[str] = ("headroom", "proxy"),
        skip_upstream_check: bool = True,
    ) -> dict[str, ProxyDeploymentPlan]:
        plans: dict[str, ProxyDeploymentPlan] = {}
        for offset, scenario in enumerate(self._require_scenarios()):
            plans[scenario.name] = scenario.deployment_plan(
                port=port_start + offset,
                headroom_cmd=headroom_cmd,
                skip_upstream_check=skip_upstream_check,
            )
        return plans

    DeploymentPlans = deployment_plans

    def manifest_bundle(
        self,
        *,
        provider: Literal["anthropic", "openai"] = "anthropic",
        port_start: int = 8787,
    ) -> SuiteManifestBundle:
        scenarios: list[dict[str, Any]] = []
        for offset, scenario in enumerate(self._require_scenarios()):
            fragment = scenario.bench_manifest_fragment(provider=provider).to_dict()
            fragment["suite_port"] = port_start + offset
            fragment["deployment_plan"] = scenario.deployment_plan(
                port=port_start + offset
            ).to_dict()
            scenarios.append(fragment)
        return SuiteManifestBundle(
            name=self.name,
            harness="headroom.testing",
            harness_version=Headroom.HARNESS_VERSION,
            scenarios=tuple(scenarios),
        )

    ManifestBundle = manifest_bundle

    def write_manifest_bundle(
        self,
        path: str | Path,
        *,
        provider: Literal["anthropic", "openai"] = "anthropic",
        port_start: int = 8787,
    ) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.manifest_bundle(provider=provider, port_start=port_start).to_dict()
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target

    WriteManifestBundle = write_manifest_bundle

    def agent_evals_manifests(
        self,
        *,
        benchmark: str,
        benchmark_ref: str,
        provider: Literal["anthropic", "openai"] = "anthropic",
        now: datetime | None = None,
        model_snapshot: str = "claude-sonnet-4-6",
        headroom_repo_path: str | Path = ".",
        agent_evals_repo_path: str | Path = ".",
        auth_mode: str = "payg",
        temperature: float = 0.0,
        k_runs: int = 10,
        seeds: Sequence[int] | None = None,
        alpha: float = 0.05,
        margins: Mapping[str, float] | None = None,
        pricing: AgentEvalsPricing | None = None,
        docker_digests: Mapping[str, str] | None = None,
    ) -> dict[str, AgentEvalsManifest]:
        stamp = now or datetime.now(timezone.utc)
        return {
            scenario.name: scenario.agent_evals_manifest(
                benchmark=benchmark,
                benchmark_ref=benchmark_ref,
                provider=provider,
                now=stamp,
                model_snapshot=model_snapshot,
                headroom_repo_path=headroom_repo_path,
                agent_evals_repo_path=agent_evals_repo_path,
                auth_mode=auth_mode,
                temperature=temperature,
                k_runs=k_runs,
                seeds=seeds,
                alpha=alpha,
                margins=margins,
                pricing=pricing,
                docker_digests=docker_digests,
            )
            for scenario in self._require_scenarios()
        }

    AgentEvalsManifests = agent_evals_manifests

    def write_agent_evals_manifests(
        self,
        directory: str | Path,
        *,
        benchmark: str,
        benchmark_ref: str,
        provider: Literal["anthropic", "openai"] = "anthropic",
        now: datetime | None = None,
    ) -> tuple[Path, ...]:
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        manifests = self.agent_evals_manifests(
            benchmark=benchmark,
            benchmark_ref=benchmark_ref,
            provider=provider,
            now=now,
        )
        written: list[Path] = []
        for scenario_name, manifest in manifests.items():
            target = target_dir / f"{scenario_name}.agent-evals.json"
            target.write_text(
                json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            written.append(target)
        return tuple(written)

    WriteAgentEvalsManifests = write_agent_evals_manifests

    def _require_scenarios(self) -> tuple[HarnessScenario, ...]:
        if not self._scenarios:
            raise ValueError("HeadroomSuite requires at least one scenario")
        return tuple(self._scenarios)


@dataclass(frozen=True)
class HarnessScenario:
    """A fully-built, immutable Headroom test scenario."""

    name: str
    provider: ProviderTarget
    platform: PlatformTarget
    headroom_config: HeadroomConfig
    proxy_config: ProxyConfig
    metadata: dict[str, Any]

    @property
    def contract(self) -> ScenarioContract:
        return ScenarioContract(
            headroom_fields=_field_contract("headroom", HeadroomConfig),
            proxy_fields=_field_contract("proxy", ProxyConfig),
        )

    def env(self, *, base_url: str | None = None) -> dict[str, str]:
        env: dict[str, str] = {
            "HEADROOM_MODE": self.proxy_config.mode,
            "HEADROOM_BACKEND": self.proxy_config.backend,
            "HEADROOM_DISABLE_KOMPRESS": "1" if self.proxy_config.disable_kompress else "0",
            "HEADROOM_SAVINGS_PROFILE": self.proxy_config.savings_profile or "",
        }
        if self.proxy_config.bedrock_region:
            env["HEADROOM_BEDROCK_REGION"] = self.proxy_config.bedrock_region
        if self.proxy_config.bedrock_profile:
            env["AWS_PROFILE"] = self.proxy_config.bedrock_profile
        if base_url:
            if self.provider is ProviderTarget.OPENAI:
                openai_url = base_url if base_url.rstrip("/").endswith("/v1") else f"{base_url}/v1"
                env["OPENAI_BASE_URL"] = openai_url
                env["OPENAI_API_BASE"] = openai_url
            elif self.provider is ProviderTarget.ANTHROPIC:
                env["ANTHROPIC_BASE_URL"] = base_url
        return {key: value for key, value in env.items() if value != ""}

    def proxy_command(
        self,
        *,
        port: int | None = None,
        headroom_cmd: Sequence[str] = ("headroom", "proxy"),
        extra_flags: Sequence[str] = (),
    ) -> tuple[str, ...]:
        cmd = list(headroom_cmd)
        if port is not None:
            cmd += ["--port", str(port)]
        if not self.proxy_config.optimize:
            cmd.append("--no-optimize")
        else:
            cmd += ["--mode", self.proxy_config.mode]
        if self.proxy_config.backend != "anthropic":
            cmd += ["--backend", self.proxy_config.backend]
        if self.proxy_config.disable_kompress:
            cmd.append("--disable-kompress")
        if self.proxy_config.lossless:
            cmd.append("--lossless")
        if self.proxy_config.bedrock_region:
            cmd += ["--bedrock-region", self.proxy_config.bedrock_region]
        if self.proxy_config.bedrock_profile:
            cmd += ["--bedrock-profile", self.proxy_config.bedrock_profile]
        cmd += list(extra_flags)
        return tuple(cmd)

    def bench_arms(
        self, *, provider: Literal["anthropic", "openai"] = "anthropic"
    ) -> tuple[BenchArm, ...]:
        headroom_proxy_mode: Literal["off", "token", "cache"]
        if self.proxy_config.optimize:
            raw_mode = cast(str, self.proxy_config.mode)
            if raw_mode not in {"token", "cache"}:
                raise ValueError(f"unsupported headroom-bench proxy mode: {raw_mode!r}")
            headroom_proxy_mode = cast(Literal["token", "cache"], raw_mode)
        else:
            headroom_proxy_mode = "off"
        return (
            BenchArm(
                name=ArmName.A0_DIRECT,
                provider=provider,
                proxy_mode=None,
                proxy_flags=(),
                label="Direct provider API",
            ),
            BenchArm(
                name=ArmName.A1_PASSTHROUGH,
                provider=provider,
                proxy_mode="off",
                proxy_flags=(),
                label="Headroom proxy passthrough",
            ),
            BenchArm(
                name=ArmName.B_HEADROOM,
                provider=provider,
                proxy_mode=headroom_proxy_mode,
                proxy_flags=self._bench_proxy_flags(),
                label="Headroom compression",
            ),
        )

    def bench_manifest_fragment(
        self, *, provider: Literal["anthropic", "openai"] = "anthropic"
    ) -> BenchManifestFragment:
        deployment_plan = self.deployment_plan()
        contract_audit = self.audit_contract()
        return BenchManifestFragment(
            harness="headroom.testing",
            harness_version=Headroom.HARNESS_VERSION,
            provider=self.provider.value,
            platform=self.platform.value,
            arms=self.bench_arms(provider=provider),
            proxy_config=_public_dataclass_dict(self.proxy_config),
            headroom_config=_public_dataclass_dict(self.headroom_config),
            env=self.env(),
            deployment_plan=deployment_plan.to_dict(),
            contract_audit=contract_audit.to_dict(),
        )

    def agent_evals_manifest(
        self,
        *,
        benchmark: str,
        benchmark_ref: str,
        provider: Literal["anthropic", "openai"] = "anthropic",
        now: datetime | None = None,
        model_snapshot: str = "claude-sonnet-4-6",
        headroom_repo_path: str | Path = ".",
        agent_evals_repo_path: str | Path = ".",
        auth_mode: str = "payg",
        temperature: float = 0.0,
        k_runs: int = 10,
        seeds: Sequence[int] | None = None,
        alpha: float = 0.05,
        margins: Mapping[str, float] | None = None,
        pricing: AgentEvalsPricing | None = None,
        docker_digests: Mapping[str, str] | None = None,
    ) -> AgentEvalsManifest:
        created_at = now or datetime.now(timezone.utc)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        else:
            created_at = created_at.astimezone(timezone.utc)
        run_seeds = tuple(seeds) if seeds is not None else tuple(range(k_runs))
        return AgentEvalsManifest(
            experiment_id=f"{benchmark}-{self.name}-{created_at:%Y%m%dT%H%M%SZ}",
            created_at=created_at,
            headroom_git_sha=_git_sha(headroom_repo_path),
            agent_evals_git_sha=_git_sha(agent_evals_repo_path),
            model_snapshot=model_snapshot,
            provider=provider,
            auth_mode=auth_mode,
            benchmark=benchmark,
            benchmark_ref=benchmark_ref,
            harness="headroom.testing",
            harness_version=Headroom.HARNESS_VERSION,
            docker_digests=dict(docker_digests or {}),
            arms=self.bench_arms(provider=provider),
            k_runs=k_runs,
            temperature=temperature,
            seeds=run_seeds,
            alpha=alpha,
            margins=dict(margins or {"ccr": 0.0, "lossy": 2.0}),
            pricing=pricing or AgentEvalsPricing(),
        )

    AgentEvalsManifest = agent_evals_manifest

    def write_agent_evals_manifest(
        self,
        path: str | Path,
        *,
        benchmark: str,
        benchmark_ref: str,
        provider: Literal["anthropic", "openai"] = "anthropic",
        now: datetime | None = None,
    ) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.agent_evals_manifest(
            benchmark=benchmark,
            benchmark_ref=benchmark_ref,
            provider=provider,
            now=now,
        ).to_dict()
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target

    WriteAgentEvalsManifest = write_agent_evals_manifest

    def audit_contract(self) -> ContractAudit:
        """Audit scenario payloads against the current dataclass contract."""

        contract = self.contract
        headroom_contract = contract.field_names("headroom")
        proxy_contract = contract.field_names("proxy")
        headroom_payload = set(_public_dataclass_dict(self.headroom_config))
        proxy_payload = set(_public_dataclass_dict(self.proxy_config))
        notes: list[str] = []
        if self.metadata.get("read_maturation"):
            notes.append("read_maturation is currently a proxy-only surface")
        return ContractAudit(
            scenario=self.name,
            provider=self.provider.value,
            platform=self.platform.value,
            headroom_fields_total=len(headroom_contract),
            proxy_fields_total=len(proxy_contract),
            headroom_payload_fields=tuple(sorted(headroom_payload)),
            proxy_payload_fields=tuple(sorted(proxy_payload)),
            missing_headroom_payload_fields=tuple(sorted(headroom_contract - headroom_payload)),
            missing_proxy_payload_fields=tuple(sorted(proxy_contract - proxy_payload)),
            extra_headroom_payload_fields=tuple(sorted(headroom_payload - headroom_contract)),
            extra_proxy_payload_fields=tuple(sorted(proxy_payload - proxy_contract)),
            notes=tuple(notes),
        )

    def write_manifest_fragment(
        self,
        path: str | Path,
        *,
        provider: Literal["anthropic", "openai"] = "anthropic",
    ) -> Path:
        """Write a JSON manifest fragment that headroom-bench can consume."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.bench_manifest_fragment(provider=provider).to_dict()
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target

    def deployment_plan(
        self,
        *,
        port: int | None = None,
        headroom_cmd: Sequence[str] = ("headroom", "proxy"),
        extra_flags: Sequence[str] = (),
        skip_upstream_check: bool = True,
    ) -> ProxyDeploymentPlan:
        """Build complete command/env/config inputs for a proxy deployment."""

        command = self.proxy_command(port=port, headroom_cmd=headroom_cmd, extra_flags=extra_flags)
        config_payload = _public_dataclass_dict(self.proxy_config)
        env = self.env(base_url=f"http://127.0.0.1:{port}" if port is not None else None)
        env["HEADROOM_PROXY_CONFIG_JSON"] = json.dumps(
            config_payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        if skip_upstream_check:
            env["HEADROOM_SKIP_UPSTREAM_CHECK"] = "1"
        plan = ProxyDeploymentPlan(command=command, env=env, config_payload=config_payload)
        plan.validate()
        return plan

    def simulate(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str = "gpt-4o",
        provider: Literal["openai", "anthropic"] = "openai",
        output_buffer_tokens: int | None = None,
    ) -> ScenarioResult:
        """Run the real Headroom SDK transform pipeline without calling an upstream API."""

        from headroom.client import HeadroomClient
        from headroom.providers.anthropic import AnthropicProvider
        from headroom.providers.openai import OpenAIProvider

        provider_obj = (
            AnthropicProvider(warn=False) if provider == "anthropic" else OpenAIProvider()
        )
        config = replace(self.headroom_config)
        temp_metrics_path = (
            Path(tempfile.gettempdir()) / f"headroom-testing-{uuid.uuid4().hex}.jsonl"
        )
        store_url = f"jsonl://{temp_metrics_path}"
        config.store_url = store_url
        client = HeadroomClient(
            original_client=_NoopClient(),
            provider=provider_obj,
            store_url=store_url,
            config=config,
            default_mode=self.headroom_config.default_mode.value,
            enable_cache_optimizer=False,
        )
        result = client.chat.completions.simulate(
            model=model,
            messages=messages,
            headroom_mode=HeadroomMode.OPTIMIZE.value,
            headroom_output_buffer_tokens=output_buffer_tokens,
        )
        return ScenarioResult(
            tokens_before=result.tokens_before,
            tokens_after=result.tokens_after,
            tokens_saved=result.tokens_saved,
            transforms=tuple(result.transforms),
            messages=result.messages_optimized,
        )

    def orchestrate(
        self,
        tasks: Sequence[ScenarioTask],
        *,
        guarantees: Sequence[Guarantee] = DEFAULT_GUARANTEES,
    ) -> ScenarioRunReport:
        """Run no-key simulations for this scenario and evaluate guarantees."""

        return ScenarioOrchestrator([self], guarantees=guarantees).run(tasks)

    def deploy_local(
        self,
        *,
        port: int = 8787,
        headroom_cmd: Sequence[str] = ("headroom", "proxy"),
        ready_path: str = "/readyz",
        timeout_s: float = 30.0,
        log_path: str | Path | None = None,
    ) -> LocalProxyDeployment:
        return LocalProxyDeployment(
            self,
            port=port,
            headroom_cmd=headroom_cmd,
            ready_path=ready_path,
            timeout_s=timeout_s,
            log_path=log_path,
        )

    def _bench_proxy_flags(self) -> tuple[str, ...]:
        flags: list[str] = []
        if self.proxy_config.disable_kompress:
            flags.append("--disable-kompress")
        if self.proxy_config.lossless:
            flags.append("--lossless")
        return tuple(flags)


class LocalProxyDeployment:
    """Context manager that launches a configured local Headroom proxy."""

    def __init__(
        self,
        scenario: HarnessScenario,
        *,
        port: int,
        headroom_cmd: Sequence[str],
        ready_path: str,
        timeout_s: float,
        log_path: str | Path | None,
    ) -> None:
        self.scenario = scenario
        self.port = port
        self.headroom_cmd = tuple(headroom_cmd)
        self.ready_path = ready_path
        self.timeout_s = timeout_s
        self.log_path = Path(log_path) if log_path is not None else None
        self._process: subprocess.Popen[bytes] | None = None
        self._log_file: Any = None

    def __enter__(self) -> DeploymentHandle:
        plan = self.scenario.deployment_plan(port=self.port, headroom_cmd=self.headroom_cmd)
        command = plan.command
        env = os.environ.copy()
        env.update(plan.env)
        stdout: Any = subprocess.DEVNULL
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = self.log_path.open("ab")
            stdout = self._log_file
        self._process = subprocess.Popen(command, stdout=stdout, stderr=stdout, env=env)
        try:
            self._wait_ready()
        except Exception:
            self.__exit__(None, None, None)
            raise
        return DeploymentHandle(
            base_url=f"http://127.0.0.1:{self.port}",
            env=plan.env,
            command=command,
            process_id=self._process.pid,
        )

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def _wait_ready(self) -> None:
        assert self._process is not None
        url = f"http://127.0.0.1:{self.port}{self.ready_path}"
        deadline = time.monotonic() + self.timeout_s
        last_error: str | None = None
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(f"headroom proxy exited with code {self._process.returncode}")
            try:
                with urllib.request.urlopen(url, timeout=0.5) as response:
                    if response.status == 200:
                        return
            except (OSError, urllib.error.URLError) as exc:
                last_error = str(exc)
            time.sleep(0.25)
        raise TimeoutError(f"headroom proxy was not ready at {url}: {last_error}")


class _NoopClient:
    """Minimal original-client object for SDK simulations without upstream I/O."""


class ScenarioOrchestrator:
    """Runs built scenarios against local no-key tasks and evaluates guarantees."""

    def __init__(
        self,
        scenarios: Sequence[HarnessScenario],
        *,
        guarantees: Sequence[Guarantee] = DEFAULT_GUARANTEES,
    ) -> None:
        if not scenarios:
            raise ValueError("ScenarioOrchestrator requires at least one scenario")
        self._scenarios = tuple(scenarios)
        self._guarantees = tuple(guarantees)

    def run(self, tasks: Sequence[ScenarioTask]) -> ScenarioRunReport:
        if not tasks:
            raise ValueError("ScenarioOrchestrator.run requires at least one task")

        cases: list[ScenarioCaseResult] = []
        for scenario in self._scenarios:
            for task in tasks:
                result = scenario.simulate(
                    task.messages,
                    model=task.model,
                    provider=task.provider,
                    output_buffer_tokens=task.output_buffer_tokens,
                )
                guarantees = tuple(
                    guarantee(scenario, task, result) for guarantee in self._guarantees
                )
                cases.append(
                    ScenarioCaseResult(
                        scenario=scenario.name,
                        task_id=task.task_id,
                        result=result,
                        guarantees=guarantees,
                    )
                )
        return ScenarioRunReport(cases=tuple(cases))
