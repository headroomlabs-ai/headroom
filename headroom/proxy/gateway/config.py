"""Immutable version-1 gateway configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from headroom.proxy.gateway.destinations import (
    SOURCE_KINDS,
    https_destination,
    normalized_origin,
    path_within,
    validate_audience,
    validate_path,
)

Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")]
Protocol = Literal[
    "openai-chat",
    "openai-responses",
    "anthropic-messages",
    "gemini-generate",
    "vertex-generate",
    "bedrock-invoke",
]
Provider = Literal["openai", "anthropic", "gemini", "vertex", "bedrock", "compatible"]


class FrozenDict(dict):
    """Pydantic-serializable mapping whose published contents cannot be edited."""

    def _immutable(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("configuration mapping is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __deepcopy__(self, memo: Any) -> FrozenDict:
        return self


def _freeze(value: Any) -> Any:
    return (
        FrozenDict({key: _freeze(item) for key, item in value.items()})
        if isinstance(value, dict)
        else value
    )


class FrozenModel(BaseModel):
    """Strict immutable base for one published configuration snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def freeze_mappings(self) -> FrozenModel:
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, dict):
                object.__setattr__(self, name, _freeze(value))
        return self


class RuntimeConfig(FrozenModel):
    profile: Literal["gateway"]
    engine: Literal["python"]
    bind: Literal["127.0.0.1"]
    port: int = Field(ge=1, le=65535)
    workers: Literal[1]
    remote_enabled: Literal[False]
    stateless: bool


class TransformConfig(FrozenModel):
    mode: Literal["off"]


class PrivacyConfig(FrozenModel):
    beacon: Literal[False]
    payload_logging: Literal[False]
    metrics: Literal["off", "local"]


Money = Annotated[str, Field(pattern=r"^(0|[1-9][0-9]*)(\.[0-9]{1,12})?$")]
PositiveSeconds = Annotated[float, Field(gt=0, le=86400, allow_inf_nan=False)]
Feature = Literal[
    "text",
    "tools",
    "parallel_tools",
    "inline_images",
    "structured_output",
    "signed_state",
    "hosted_tools",
]
Transport = Literal["http-json", "http-stream", "websocket"]


class AdmissionPolicy(FrozenModel):
    max_concurrency: int = Field(default=128, ge=1, le=10000)
    queue_limit: int = Field(default=128, ge=0, le=10000)
    queue_timeout_seconds: float = Field(default=0, ge=0, le=300, allow_inf_nan=False)
    budget_usd: Money | None = None
    budget_period: Literal["process"] = "process"
    unknown_cost_policy: Literal["allow", "block"] = "allow"

    @model_validator(mode="after")
    def strict_budget(self) -> AdmissionPolicy:
        if self.budget_usd is not None and self.unknown_cost_policy != "block":
            raise ValueError("finite budget requires unknown_cost_policy=block")
        return self


class PrincipalAdmissionPolicy(AdmissionPolicy):
    max_concurrency: int = Field(default=8, ge=1, le=10000)
    reserved_concurrency: int = Field(default=0, ge=0, le=10000)
    queue_limit: int = Field(default=8, ge=0, le=10000)

    @model_validator(mode="after")
    def reservation_bound(self) -> PrincipalAdmissionPolicy:
        if self.reserved_concurrency > self.max_concurrency:
            raise ValueError("reserved concurrency exceeds principal maximum")
        return self


class LimitsConfig(FrozenModel):
    request_deadline_seconds: PositiveSeconds = 120
    stream_content_idle_seconds: PositiveSeconds = 30
    partial_frame_seconds: PositiveSeconds = 10
    max_frame_bytes: int = Field(default=1048576, ge=1024, le=16777216)
    max_observed_json_bytes: int = Field(default=8388608, ge=1024, le=67108864)
    websocket_idle_seconds: PositiveSeconds = 300
    resource_ttl_seconds: PositiveSeconds = 3600
    max_resource_bindings: int = Field(default=10000, ge=1, le=1000000)
    shutdown_drain_seconds: PositiveSeconds = 2
    shutdown_cleanup_seconds: PositiveSeconds = 3


class SelectionConfig(FrozenModel):
    strategy: Literal["round_robin", "priority", "weighted"] = "round_robin"
    priority: dict[Identifier, int] = Field(default_factory=dict)
    weight: dict[Identifier, Annotated[int, Field(ge=1, le=100)]] = Field(default_factory=dict)
    quota_group: Identifier | None = None
    required_residency: Identifier | None = None


class CatalogConfig(FrozenModel):
    source: Literal["configured", "provider"] = "configured"
    ttl_seconds: PositiveSeconds = 300
    stale_if_error_seconds: float = Field(default=0, ge=0, le=86400, allow_inf_nan=False)
    refresh_timeout_seconds: Annotated[float, Field(gt=0, le=60, allow_inf_nan=False)] = 5
    entitlements: dict[Identifier, Literal["allowed", "denied", "unknown"]] = Field(
        default_factory=dict
    )


class CapabilityConfig(FrozenModel):
    operations: tuple[Literal["generate"], ...] = Field(default=("generate",), min_length=1)
    features: tuple[Feature, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique(self) -> CapabilityConfig:
        _require_unique("capability features", self.features)
        _require_unique("capability operations", self.operations)
        return self


class PricingConfig(FrozenModel):
    currency: Literal["USD"] = "USD"
    input_usd_per_million: Money
    output_usd_per_million: Money
    cache_read_usd_per_million: Money | None = None
    cache_create_usd_per_million: Money | None = None
    revision: Identifier


class ModelBounds(FrozenModel):
    max_input_tokens: int = Field(ge=1, le=100000000)
    max_output_tokens: int = Field(ge=1, le=100000000)
    default_max_output_tokens: int | None = Field(default=None, ge=1, le=100000000)
    provider_contract: Identifier

    @model_validator(mode="after")
    def output_bound(self) -> ModelBounds:
        if (
            self.default_max_output_tokens is not None
            and self.default_max_output_tokens > self.max_output_tokens
        ):
            raise ValueError("default output exceeds qualified bound")
        return self


class PrincipalConfig(FrozenModel):
    id: Identifier
    secret_ref: Annotated[str, Field(pattern=r"^env:[A-Z][A-Z0-9_]*$")]
    scopes: tuple[Literal["inference", "models", "admin"], ...] = Field(min_length=1)
    routes: tuple[Identifier, ...]
    enabled: bool = True
    admission: PrincipalAdmissionPolicy = PrincipalAdmissionPolicy()

    @model_validator(mode="after")
    def unique_values(self) -> PrincipalConfig:
        _require_unique("principal scopes", self.scopes)
        _require_unique("principal routes", self.routes)
        if not self.routes and set(self.scopes) != {"admin"}:
            raise ValueError("inference/models principals require route grants")
        return self


class ClientAuthConfig(FrozenModel):
    required: Literal[True]
    principals: tuple[PrincipalConfig, ...] = Field(min_length=1)


class EnvironmentSource(FrozenModel):
    kind: Literal["env"]
    ref: Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]*$")]


class GcpAdcSource(FrozenModel):
    kind: Literal["gcp-adc"]
    project: Annotated[str, Field(min_length=1)]


class AwsChainSource(FrozenModel):
    kind: Literal["aws-chain"]
    region: Annotated[str, Field(min_length=1)]
    profile: Annotated[str, Field(min_length=1)]


class NoCredentialSource(FrozenModel):
    kind: Literal["none"]


CredentialSource = Annotated[
    EnvironmentSource | GcpAdcSource | AwsChainSource | NoCredentialSource,
    Field(discriminator="kind"),
]


class CredentialConfig(FrozenModel):
    id: Identifier
    provider: Provider
    source: CredentialSource
    refresh_owner: Literal["none", "sdk"]
    allowed_origins: tuple[Annotated[str, Field(pattern=r"^https://")], ...] = Field(min_length=1)
    allowed_path_prefixes: tuple[Annotated[str, Field(pattern=r"^/")], ...] = Field(min_length=1)
    enabled: bool
    billing_group: Identifier | None = None
    owner_group: Identifier | None = None
    residency: Identifier | None = None
    max_concurrency: int = Field(default=128, ge=1, le=10000)

    @model_validator(mode="after")
    def validate_source_contract(self) -> CredentialConfig:
        if SOURCE_KINDS.get(self.provider) != self.source.kind:
            raise ValueError("provider credential source is not admitted")
        if self.source.kind in {"gcp-adc", "aws-chain"} and self.refresh_owner != "sdk":
            raise ValueError(f"{self.source.kind} requires refresh_owner='sdk'")
        if self.source.kind in {"env", "none"} and self.refresh_owner != "none":
            raise ValueError(f"{self.source.kind} requires refresh_owner='none'")
        if self.provider == "vertex" and self.source.kind != "gcp-adc":
            raise ValueError("vertex credentials require gcp-adc")
        if self.provider == "bedrock" and self.source.kind != "aws-chain":
            raise ValueError("bedrock credentials require aws-chain")
        if self.source.kind == "none" and self.provider != "compatible":
            raise ValueError("credential source 'none' is restricted to compatible providers")
        _require_unique("allowed origins", self.allowed_origins)
        _require_unique("allowed path prefixes", self.allowed_path_prefixes)
        for origin in self.allowed_origins:
            parsed = https_destination(origin)
            if parsed.path or parsed.query:
                raise ValueError("credential origin must contain only an HTTPS authority")
            for prefix in self.allowed_path_prefixes:
                validate_path(prefix)
                validate_audience(
                    self.provider,
                    origin + prefix,
                    project=getattr(self.source, "project", None),
                    region=getattr(self.source, "region", None),
                )
        return self


class RetryConfig(FrozenModel):
    max_attempts: int = Field(default=1, ge=1, le=3)
    max_retry_after_seconds: float = Field(default=5, ge=0, le=60, allow_inf_nan=False)
    base_backoff_seconds: float = Field(default=0.1, ge=0, le=5, allow_inf_nan=False)
    ambiguous_commit: Literal["never"]
    after_output: Literal["never"]


class BillingConfig(FrozenModel):
    allow_paid_fallback: Literal[False]


class RouteConfig(FrozenModel):
    id: Identifier
    public_model: Annotated[str, Field(min_length=1)]
    upstream_model: Annotated[str, Field(min_length=1)]
    provider: Provider
    upstream_origin: Annotated[str, Field(pattern=r"^https://")]
    upstream_path_prefix: Annotated[str, Field(pattern=r"^/")]
    credentials: tuple[Identifier, ...] = Field(min_length=1)
    ingress_protocols: tuple[Protocol, ...] = Field(min_length=1)
    native_protocols: tuple[Protocol, ...] = Field(min_length=1)
    translation: Literal["disabled", "qualified"]
    body_contract: Literal["strict-native", "routed-native"]
    private_network: bool
    retry: RetryConfig
    billing: BillingConfig
    enabled: bool = True
    selection: SelectionConfig = SelectionConfig()
    capabilities: dict[Protocol, dict[Transport, CapabilityConfig]] = Field(min_length=1)
    catalog: CatalogConfig = CatalogConfig()
    pricing: PricingConfig | None = None
    model_bounds: ModelBounds | None = None

    @model_validator(mode="after")
    def unique_values(self) -> RouteConfig:
        from headroom.proxy.gateway.capabilities import implemented_features

        if not self.capabilities or any(
            not transports for transports in self.capabilities.values()
        ):
            raise ValueError("routes require explicit capability declarations")
        for mapping in (self.selection.priority, self.selection.weight, self.catalog.entitlements):
            if mapping and set(mapping) != set(self.credentials):
                raise ValueError("candidate policy map must contain exactly the route credentials")
        for protocol, transports in self.capabilities.items():
            if protocol not in self.ingress_protocols:
                raise ValueError("capability protocol is not an ingress protocol")
            for transport, declaration in transports.items():
                if not set(declaration.features) <= implemented_features(
                    protocol,
                    transport,
                    protocol in self.native_protocols,
                    self.native_protocols[0] if len(self.native_protocols) == 1 else None,
                ):
                    raise ValueError("capability declaration exceeds implemented contract")
        parsed = https_destination(self.upstream_origin)
        if parsed.path or parsed.query:
            raise ValueError("route origin must contain only an HTTPS authority")
        validate_path(self.upstream_path_prefix)
        if self.private_network and self.provider != "compatible":
            raise ValueError("private destinations require a compatible environment credential")
        _require_unique("route credentials", self.credentials)
        _require_unique("ingress protocols", self.ingress_protocols)
        _require_unique("native protocols", self.native_protocols)
        non_native = set(self.ingress_protocols) - set(self.native_protocols)
        if non_native and self.translation == "disabled":
            raise ValueError("non-native ingress requires qualified translation")
        if self.translation == "qualified":
            supported_pairs = {
                ("openai-chat", "anthropic-messages"),
                ("anthropic-messages", "openai-chat"),
                ("gemini-generate", "openai-chat"),
            }
            if len(self.native_protocols) != 1 or not non_native:
                raise ValueError("qualified translation requires one native target")
            target = self.native_protocols[0]
            if any((source, target) not in supported_pairs for source in non_native):
                raise ValueError("translation direction is not qualified")
        return self


class TransportConfig(FrozenModel):
    ca_bundle: str | None = None


class GatewayConfigSnapshot(FrozenModel):
    """A validated configuration generation with no resolved secret material."""

    version: Literal[1]
    runtime: RuntimeConfig
    transforms: TransformConfig
    privacy: PrivacyConfig
    client_auth: ClientAuthConfig
    credentials: tuple[CredentialConfig, ...] = Field(min_length=1)
    routes: tuple[RouteConfig, ...] = Field(min_length=1)
    transport: TransportConfig = TransportConfig()
    admission: AdmissionPolicy = AdmissionPolicy()
    limits: LimitsConfig = LimitsConfig()

    @classmethod
    def load(cls, path: Path) -> GatewayConfigSnapshot:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cast("GatewayConfigSnapshot", cls.model_validate(raw))

    def redacted_dict(self) -> dict[str, object]:
        # Version 1 stores references only. Credential values are deliberately not
        # accepted by any model, so a regular dump is already secret-free.
        return cast("dict[str, object]", self.model_dump(mode="json"))

    @model_validator(mode="after")
    def validate_references(self) -> GatewayConfigSnapshot:
        if (
            sum(p.admission.reserved_concurrency for p in self.client_auth.principals if p.enabled)
            > self.admission.max_concurrency
        ):
            raise ValueError("principal reservations exceed global concurrency")
        credential_ids = _index_unique("credential id", self.credentials, key=lambda item: item.id)
        route_ids = _index_unique("route id", self.routes, key=lambda item: item.id)
        _index_unique("principal id", self.client_auth.principals, key=lambda item: item.id)
        _index_unique("public_model", self.routes, key=lambda item: item.public_model)

        for principal in self.client_auth.principals:
            unknown_routes = sorted(set(principal.routes) - route_ids.keys())
            if unknown_routes:
                raise ValueError(
                    f"principal {principal.id} references unknown route: {unknown_routes}"
                )

        for route in self.routes:
            groups = {
                (
                    credential_ids[account].owner_group or account,
                    credential_ids[account].billing_group or account,
                    credential_ids[account].residency,
                )
                for account in route.credentials
                if account in credential_ids
            }
            if len(groups) > 1:
                raise ValueError(
                    "route candidates require explicit equivalent owner/billing/residency"
                )
            for credential_id in route.credentials:
                credential = credential_ids.get(credential_id)
                if credential is None:
                    raise ValueError(
                        f"route {route.id} references unknown credential: {credential_id}"
                    )
                if (
                    route.selection.required_residency is not None
                    and credential.residency != route.selection.required_residency
                ):
                    raise ValueError("credential does not satisfy route residency")
                if credential.provider != route.provider:
                    raise ValueError(
                        f"route {route.id} provider {route.provider} does not match "
                        f"credential {credential_id} provider {credential.provider}"
                    )
                if normalized_origin(route.upstream_origin) not in {
                    normalized_origin(origin) for origin in credential.allowed_origins
                }:
                    raise ValueError(
                        f"route {route.id} origin is outside credential {credential_id} origins"
                    )
                if not any(
                    path_within(route.upstream_path_prefix, prefix)
                    for prefix in credential.allowed_path_prefixes
                ):
                    raise ValueError(
                        f"route {route.id} path is outside credential {credential_id} prefixes"
                    )
        return self


def gateway_proxy_overrides(snapshot: GatewayConfigSnapshot) -> dict[str, object]:
    """Return authoritative legacy ProxyConfig values for gateway pure mode."""

    return {
        "host": snapshot.runtime.bind,
        "port": snapshot.runtime.port,
        "optimize": False,
        "cache_enabled": False,
        "memory_enabled": False,
        "traffic_learning_enabled": False,
        "ccr_inject_tool": False,
        "ccr_inject_marker": False,
        "ccr_handle_responses": False,
        "code_graph_watcher": False,
        "image_optimize": False,
        "retry_enabled": False,
        "stateless": snapshot.runtime.stateless,
    }


def _require_unique(label: str, values: tuple[object, ...]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label}")


def _index_unique(label: str, values: tuple[object, ...], *, key):  # type: ignore[no-untyped-def]
    result = {}
    for value in values:
        item_key = key(value)
        if item_key in result:
            raise ValueError(f"duplicate {label}: {item_key}")
        result[item_key] = value
    return result
