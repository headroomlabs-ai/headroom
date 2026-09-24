"""Atomic policy publication, with process-lifetime mutable ownership."""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from headroom.proxy.gateway.admission import AdmissionController
from headroom.proxy.gateway.auth import GatewayAuthenticator, GatewayAuthorizer
from headroom.proxy.gateway.config import CredentialConfig, GatewayConfigSnapshot, RouteConfig
from headroom.proxy.gateway.control import RedactedGatewayStatus
from headroom.proxy.gateway.credentials import CredentialBroker
from headroom.proxy.gateway.egress import EgressPolicy, build_managed_upstream_headers
from headroom.proxy.gateway.errors import GatewayAuthorizationError
from headroom.proxy.gateway.models import (
    AccountAvailability,
    CatalogSnapshot,
    ProviderModelMetadata,
)
from headroom.proxy.gateway.observability import GatewayObservability
from headroom.proxy.gateway.resources import ResourceBinding, ResourceRegistry
from headroom.proxy.gateway.routing import AccountRouter
from headroom.proxy.gateway.transport import http_client, private_transport, tls_context


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _provider_metadata(provider: str, model_id: str, item: dict[str, Any]) -> ProviderModelMetadata:
    """Retain positive provider evidence; an ID alone proves no operation."""
    operations: set[str] = set()
    evidence = item.get("supportedFeatures")
    if provider == "gemini":
        methods = item.get("supportedGenerationMethods", [])
        if not isinstance(methods, list) or any(not isinstance(v, str) for v in methods):
            raise ValueError("invalid metadata operations")
        if "generateContent" in methods:
            operations.add("generate")
        if "streamGenerateContent" in methods:
            operations.add("stream")
    elif provider == "anthropic" and item.get("type") == "model":
        # Anthropic's ModelInfo contract identifies Messages models, whose
        # documented minimum is text generation with optional SSE streaming.
        operations.update({"generate", "stream"})
    elif provider in {"openai", "compatible"}:
        capabilities = item.get("capabilities", {})
        if not isinstance(capabilities, dict):
            raise ValueError("invalid metadata capabilities")
        for operation in ("generate", "stream"):
            if capabilities.get(operation) is True:
                operations.add(operation)
        evidence = capabilities.get("features", evidence)
    if evidence is None:
        features = frozenset({"text"}) if "generate" in operations else frozenset()
    else:
        if not isinstance(evidence, list) or any(not isinstance(v, str) for v in evidence):
            raise ValueError("invalid metadata features")
        features = frozenset(evidence)
    return ProviderModelMetadata(model_id, frozenset(operations), features)


def _source_state(
    credential: CredentialConfig, environ: Mapping[str, str]
) -> Literal["available", "unavailable", "unknown"]:
    if not credential.enabled:
        return "unavailable"
    source = credential.source
    if source.kind == "none" or source.kind == "env" and source.ref in environ:
        return "available"
    return "unknown"


@dataclass(frozen=True, slots=True)
class ReloadResult:
    applied: bool
    generation: int
    error: str | None = None


@dataclass(slots=True)
class RuntimeDependencies:
    """Explicit embedded/test seams; no request can supply these dependencies."""

    clock: Callable[[], float] = time.time
    egress_policy: EgressPolicy = field(default_factory=EgressPolicy)
    http_client: Any = None
    broker: Any = None
    qualified_cost_contracts: frozenset[str] = frozenset()
    metadata_reader: (
        Callable[[Any, RouteConfig, str], Awaitable[tuple[ProviderModelMetadata, ...]]] | None
    ) = None


@dataclass(frozen=True, slots=True)
class RuntimeGeneration:
    number: int
    snapshot: GatewayConfigSnapshot
    authenticator: GatewayAuthenticator
    authorizer: GatewayAuthorizer
    catalog: CatalogSnapshot
    broker: CredentialBroker
    config_digest: str
    authorities: tuple[tuple[str, str], ...]
    targets: tuple[tuple[str, str], ...]
    http_client: Any
    tls_context: Any
    egress_policy: EgressPolicy

    @property
    def models(self) -> CatalogSnapshot:
        return self.catalog

    def account_key(self, account_ref: str) -> str:
        return dict(self.authorities)[account_ref]

    def target_key(self, route_id: str) -> str:
        return dict(self.targets)[route_id]

    def validate_binding(self, binding: ResourceBinding) -> None:
        if binding.authority_fingerprint != self.account_key(
            binding.account_ref
        ) or binding.target_fingerprint != self.target_key(binding.route_id):
            raise GatewayAuthorizationError(
                status_code=404,
                code="gateway_resource_not_found",
                message="Stateful resource not found",
            )


class GatewayRuntime:
    def __init__(
        self,
        snapshot: GatewayConfigSnapshot,
        *,
        environ: Mapping[str, str],
        config_path: Path | None = None,
        dependencies: RuntimeDependencies | None = None,
    ) -> None:
        self._environ = environ
        self.config_path = config_path.resolve() if config_path is not None else None
        self.dependencies = dependencies or RuntimeDependencies()
        self._reload_lock = asyncio.Lock()
        self.resources = ResourceRegistry(max_entries=snapshot.limits.max_resource_bindings)
        self.router = AccountRouter(
            available_accounts={c.id for c in snapshot.credentials if c.enabled}
        )
        self.observability = GatewayObservability()
        self.active_work: dict[str, Any] = {}
        self._work_empty = asyncio.Event()
        self._work_empty.set()
        self._refresh_tasks: dict[tuple[int, str, str], asyncio.Task[dict[str, object]]] = {}
        self._retired: list[RuntimeGeneration] = []
        self._generation_users: dict[int, int] = {}
        self._generations_empty = asyncio.Event()
        self._generations_empty.set()
        self._shutdown_task: asyncio.Task[None] | None = None
        self._cleanup_tasks: set[asyncio.Task[None]] = set()
        self._cleanup_deadline: float | None = None
        self._pseudonyms: dict[str, str] = {}
        self._ready = True
        self._generation = self._build_generation(snapshot, 1)
        self.admission = AdmissionController.from_policy(
            snapshot.admission,
            principals={p.id: p.admission for p in snapshot.client_auth.principals if p.enabled},
            account_limits={
                self._generation.account_key(c.id): c.max_concurrency
                for c in snapshot.credentials
                if c.enabled
            },
            generation=1,
            grants={
                p.id: frozenset(p.routes) for p in snapshot.client_auth.principals if p.enabled
            },
            routes=frozenset(r.id for r in snapshot.routes if r.enabled),
        )
        self.observability.bind_admission(self.admission)
        self.router.update_accounts(
            {c.id for c in snapshot.credentials if c.enabled},
            dict(self._generation.authorities),
            credentials={c.id: c for c in snapshot.credentials},
        )

    @property
    def generation(self) -> int:
        return self._generation.number

    @property
    def snapshot(self) -> GatewayConfigSnapshot:
        return self._generation.snapshot

    @property
    def authenticator(self) -> GatewayAuthenticator:
        return self._generation.authenticator

    @property
    def authorizer(self) -> GatewayAuthorizer:
        return self.capture().authorizer

    @property
    def models(self) -> CatalogSnapshot:
        return self.capture().catalog

    @property
    def broker(self) -> CredentialBroker:
        return self.capture().broker

    def capture(self) -> RuntimeGeneration:
        generation = self._generation
        catalog = CatalogSnapshot(
            generation.snapshot,
            accounts=generation.catalog.accounts,
            published_at=generation.catalog.published_at,
            revision=generation.catalog.revision,
            generation=generation.number,
            now=self.dependencies.clock(),
        )
        return replace(
            generation,
            catalog=catalog,
            authorizer=GatewayAuthorizer(generation.snapshot, catalog),
            broker=self.dependencies.broker or generation.broker,
            http_client=self.dependencies.http_client or generation.http_client,
            egress_policy=self.dependencies.egress_policy,
        )

    async def reload(self, path: Path | None = None) -> ReloadResult:
        if not self._ready:
            return ReloadResult(False, self.generation, "gateway_shutting_down")
        # The Python API accepts a trusted path; HTTP controls never accept one.
        selected = self.config_path if path is None else path
        if selected is None:
            return ReloadResult(False, self.generation, "reload_unavailable")
        try:
            snapshot = GatewayConfigSnapshot.load(selected)
            candidate = self._build_generation(snapshot, self.generation + 1)
        except (OSError, ValueError):
            return ReloadResult(False, self.generation, "gateway_configuration_invalid")
        async with self._reload_lock:
            if not self._ready:
                await self._close_generation(candidate)
                return ReloadResult(False, self.generation, "gateway_shutting_down")
            candidate = replace(candidate, number=self.generation + 1)
            catalog = CatalogSnapshot(
                candidate.snapshot,
                accounts=candidate.catalog.accounts,
                revision=self._generation.catalog.revision + 1,
                generation=candidate.number,
                now=self.dependencies.clock(),
            )
            candidate = replace(
                candidate,
                catalog=catalog,
                authorizer=GatewayAuthorizer(candidate.snapshot, catalog),
            )
            try:
                await self._publish(candidate)
            except GatewayAuthorizationError as exc:
                await self._close_generation(candidate)
                return ReloadResult(False, self.generation, exc.code)
            except ValueError:
                await self._close_generation(candidate)
                return ReloadResult(False, self.generation, "invalid_configuration")
        return ReloadResult(True, candidate.number)

    async def _publish(self, candidate: RuntimeGeneration) -> None:
        snapshot = candidate.snapshot

        def publish() -> None:
            current = self._generation
            enabled = {candidate.account_key(c.id) for c in snapshot.credentials if c.enabled}
            principals = {p.id: p for p in snapshot.client_auth.principals if p.enabled}
            revoked = (
                any(
                    p.enabled
                    and (
                        p.id not in principals
                        or bool(set(p.routes) - set(principals[p.id].routes))
                        or bool(set(p.scopes) - set(principals[p.id].scopes))
                    )
                    for p in current.snapshot.client_auth.principals
                )
                or bool(
                    {r.id for r in current.snapshot.routes if r.enabled}
                    - {r.id for r in snapshot.routes if r.enabled}
                )
                or bool(
                    {current.account_key(c.id) for c in current.snapshot.credentials if c.enabled}
                    - enabled
                )
            )
            if revoked:
                # A refresh spans accounts/routes. Cancel the whole bounded
                # refresh when any captured authority is revoked; ordinary
                # reloads still let it finish against its retained snapshot.
                for refresh_task in self._refresh_tasks.values():
                    refresh_task.cancel()
            for retired in [current, *self._retired]:
                for account, authority in retired.authorities:
                    if authority not in enabled:
                        retired.broker.revoke(account)
            self._retired.append(self._generation)
            self._generation = candidate
            self.router.update_accounts(
                {c.id for c in snapshot.credentials if c.enabled},
                dict(candidate.authorities),
                credentials={c.id: c for c in snapshot.credentials},
            )

        await self.admission.update_policy(
            snapshot.admission,
            principals={p.id: p.admission for p in snapshot.client_auth.principals if p.enabled},
            account_limits={
                candidate.account_key(c.id): c.max_concurrency
                for c in snapshot.credentials
                if c.enabled
            },
            generation=candidate.number,
            grants={
                p.id: frozenset(p.routes) if "inference" in p.scopes else frozenset()
                for p in snapshot.client_auth.principals
                if p.enabled
            },
            routes=frozenset(r.id for r in snapshot.routes if r.enabled),
            publish=publish,
            validate=self._require_ready,
        )
        await self._cancel_revoked(candidate)
        await self._retire_unused()

    def retain(self, generation: RuntimeGeneration) -> None:
        self._generations_empty.clear()
        self._generation_users[generation.number] = (
            self._generation_users.get(generation.number, 0) + 1
        )

    async def release(
        self, generation: RuntimeGeneration, *, cleanup_deadline: float | None = None
    ) -> None:
        count = self._generation_users.get(generation.number, 0)
        if count <= 1:
            self._generation_users.pop(generation.number, None)
        else:
            self._generation_users[generation.number] = count - 1
        if not self._generation_users:
            self._generations_empty.set()
        await self._retire_unused(cleanup_deadline=cleanup_deadline)

    async def _close_generation(self, generation: RuntimeGeneration) -> None:
        await asyncio.gather(generation.http_client.aclose(), generation.broker.aclose())

    async def _retire_unused(self, *, cleanup_deadline: float | None = None) -> None:
        unused = [
            generation
            for generation in self._retired
            if not self._generation_users.get(generation.number)
        ]
        self._retired = [generation for generation in self._retired if generation not in unused]
        tasks = [asyncio.create_task(self._close_generation(generation)) for generation in unused]
        for task in tasks:
            self._cleanup_tasks.add(task)
            task.add_done_callback(self._cleanup_finished)
        if tasks:
            # An SDK thread cannot be force-stopped safely. Keep ownership and
            # close its source after the guarded late result, within a bounded
            # wait for the caller (never publish that result into a new broker).
            timeout = self._cleanup_remaining()
            if cleanup_deadline is not None:
                timeout = min(timeout, max(0, cleanup_deadline - time.monotonic()))
            await asyncio.wait(tasks, timeout=timeout)

    def _cleanup_remaining(self) -> float:
        if self._cleanup_deadline is None:
            return self.snapshot.limits.shutdown_cleanup_seconds
        return max(0, self._cleanup_deadline - time.monotonic())

    def _cleanup_finished(self, task: asyncio.Task[None]) -> None:
        self._cleanup_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    async def _cancel_revoked(self, candidate: RuntimeGeneration) -> None:
        principals = {p.id: p for p in candidate.snapshot.client_auth.principals if p.enabled}
        routes = {r.id for r in candidate.snapshot.routes if r.enabled}
        accounts = {
            candidate.account_key(c.id) for c in candidate.snapshot.credentials if c.enabled
        }
        caller = asyncio.current_task()
        owners = [
            owner
            for owner in tuple(self.active_work.values())
            if (
                owner.principal.id not in principals
                or "inference" not in principals[owner.principal.id].scopes
                or owner.route is not None
                and (
                    owner.route.id not in routes
                    or owner.route.id not in principals[owner.principal.id].routes
                )
                or owner.selected_account_key is not None
                and owner.selected_account_key not in accounts
            )
        ]
        tasks = [
            asyncio.create_task(
                owner.cancel(
                    cancel_owner=owner.owner_task is not caller
                    and caller not in getattr(owner, "tasks", ())
                )
            )
            for owner in owners
        ]
        for task in tasks:
            self._cleanup_tasks.add(task)
            task.add_done_callback(self._cleanup_finished)
        if tasks:
            await asyncio.wait(tasks, timeout=candidate.snapshot.limits.shutdown_cleanup_seconds)

    def status(self) -> RedactedGatewayStatus:
        generation = self.capture()
        return RedactedGatewayStatus(
            service="headroom",
            profile="gateway",
            generation=generation.number,
            ready=self._ready,
            route_count=len(generation.snapshot.routes),
            credential_count=len(generation.snapshot.credentials),
            config_digest=generation.config_digest,
            catalog_revision=generation.catalog.revision,
        )

    def account_status(self) -> list[dict[str, object]]:
        generation = self.capture()
        return [
            {
                "account": self._pseudonyms.setdefault(record.authority, secrets.token_hex(8)),
                "state": record.state(generation.catalog.captured_at),
                "source_state": record.source_state,
                "provenance": record.provenance,
                "entitlement": record.entitlement,
            }
            for record in generation.catalog.accounts
        ]

    async def revoke(
        self,
        *,
        principal_id: str | None = None,
        route_id: str | None = None,
        account_id: str | None = None,
    ) -> ReloadResult:
        if sum(value is not None for value in (principal_id, route_id, account_id)) != 1:
            raise ValueError("one revocation selector required")
        async with self._reload_lock:
            if not self._ready:
                return ReloadResult(False, self.generation, "gateway_shutting_down")
            # Derive mutations while holding the publication lock: concurrent
            # revocations must compose instead of restoring an older snapshot.
            raw = self.snapshot.model_dump(mode="json")
            items = (
                raw["client_auth"]["principals"]
                if principal_id
                else raw["routes"]
                if route_id
                else raw["credentials"]
            )
            selector = principal_id or route_id or account_id
            if not any(item["id"] == selector for item in items):
                raise ValueError("unknown revocation selector")
            for item in items:
                if item["id"] == selector:
                    item["enabled"] = False
            snapshot = GatewayConfigSnapshot.model_validate(raw)
            candidate = self._build_generation(snapshot, self.generation + 1)
            try:
                await self._publish(candidate)
            except GatewayAuthorizationError as exc:
                await self._close_generation(candidate)
                return ReloadResult(False, self.generation, exc.code)
        return ReloadResult(True, candidate.number)

    def _require_ready(self) -> None:
        if not self._ready:
            raise GatewayAuthorizationError(
                status_code=503,
                code="gateway_shutting_down",
                message="Gateway is shutting down",
            )

    async def refresh_catalog(self) -> dict[str, object]:
        # This check and task registration do not yield. Publication performs
        # its own check after its lock wait, coordinated with shutdown's
        # synchronous readiness transition on the same event loop.
        self._require_ready()
        generation = self.capture()
        key = (generation.number, "catalog", "complete")
        task = self._refresh_tasks.get(key)
        if task is None:
            self.retain(generation)
            task = asyncio.create_task(self._refresh_complete(generation))
            self._refresh_tasks[key] = task
            task.add_done_callback(lambda done: self._refresh_finished(key, generation, done))
        return await asyncio.shield(task)

    def _refresh_finished(
        self,
        key: tuple[int, str, str],
        generation: RuntimeGeneration,
        task: asyncio.Task[dict[str, object]],
    ) -> None:
        self._refresh_tasks.pop(key, None)
        # A coroutine cancelled before its first step never enters a finally
        # block. Its registered generation lease must still be released.
        cleanup = asyncio.create_task(self.release(generation))
        self._cleanup_tasks.add(cleanup)
        cleanup.add_done_callback(self._cleanup_finished)
        if not task.cancelled():
            task.exception()

    async def _refresh_complete(self, generation: RuntimeGeneration) -> dict[str, object]:
        self._require_ready()
        semaphore = asyncio.Semaphore(8)

        async def probe(
            route: RouteConfig, account: str
        ) -> tuple[str, str, tuple[ProviderModelMetadata, ...] | None]:
            try:
                async with semaphore:
                    self._require_ready()
                    reader = self.dependencies.metadata_reader or self._read_metadata
                    models = await asyncio.wait_for(
                        reader(generation, route, account),
                        timeout=route.catalog.refresh_timeout_seconds,
                    )
                if (
                    not isinstance(models, tuple)
                    or len(models) > 10000
                    or any(
                        not isinstance(model, ProviderModelMetadata)
                        or not model.id
                        or len(model.id) > 1024
                        for model in models
                    )
                ):
                    raise ValueError("invalid metadata")
                return route.id, account, models
            except Exception:
                return route.id, account, None

        results = await asyncio.gather(
            *(
                probe(route, account)
                for route in generation.snapshot.routes
                if route.enabled and route.catalog.source == "provider"
                for account in route.credentials
                if any(
                    r.route_id == route.id
                    and r.account_ref == account
                    and r.source_state != "unavailable"
                    for r in generation.catalog.accounts
                )
            )
        )
        async with self._reload_lock:
            self._require_ready()
            current = self._generation
            if current.number != generation.number:
                return {
                    "generation": current.number,
                    "refreshed": 0,
                    "failed": len(results),
                    "obsolete": True,
                }
            now = self.dependencies.clock()
            records = list(current.catalog.accounts)
            for route_id, account, models in results:
                route = next(r for r in current.snapshot.routes if r.id == route_id)
                for index, record in enumerate(records):
                    if record.route_id != route_id or record.account_ref != account:
                        continue
                    if models is not None:
                        metadata = next((m for m in models if m.id == route.upstream_model), None)
                        records[index] = replace(
                            record,
                            metadata_available=metadata is not None
                            and "generate" in metadata.operations,
                            features=metadata.features if metadata is not None else frozenset(),
                            operations=metadata.operations if metadata is not None else frozenset(),
                            observed_at=now,
                            expires_at=now + route.catalog.ttl_seconds,
                            stale_until=now
                            + route.catalog.ttl_seconds
                            + route.catalog.stale_if_error_seconds,
                            refresh_failed=False,
                        )
                    else:
                        records[index] = replace(record, refresh_failed=True)
            if results:
                catalog = CatalogSnapshot(
                    current.snapshot,
                    accounts=tuple(records),
                    revision=current.catalog.revision + 1,
                    generation=current.number,
                    now=now,
                )
                self._generation = replace(
                    current,
                    catalog=catalog,
                    authorizer=GatewayAuthorizer(current.snapshot, catalog),
                )
        return {
            "generation": current.number,
            "catalog_revision": self._generation.catalog.revision,
            "refreshed": sum(models is not None for _route, _account, models in results),
            "failed": sum(models is None for _route, _account, models in results),
        }

    async def _read_metadata(
        self, generation: RuntimeGeneration, route: RouteConfig, account: str
    ) -> tuple[ProviderModelMetadata, ...]:
        if route.provider not in {"openai", "compatible", "anthropic", "gemini"}:
            raise ValueError("provider metadata adapter unavailable")
        lease = await generation.broker.acquire(route, account_ref=account)
        target = (
            route.upstream_origin.rstrip("/") + route.upstream_path_prefix.rstrip("/") + "/models"
        )
        destination = await asyncio.to_thread(
            generation.egress_policy.authorize, lease, target, route=route
        )
        headers = build_managed_upstream_headers(
            {}, lease, target, resolved_addresses=destination.addresses, route=route
        )
        if route.provider == "anthropic":
            headers["anthropic-version"] = "2023-06-01"
        request = generation.http_client.build_request("GET", target, headers=headers)
        request.extensions["gateway_destination"] = destination
        with private_transport():
            response = await generation.http_client.send(
                request, stream=True, follow_redirects=False
            )
            try:
                if response.status_code != 200:
                    raise ValueError("metadata unavailable")
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > 1048576:
                        raise ValueError("metadata response too large")
                payload = json.loads(body)
            finally:
                await response.aclose()
        if not isinstance(payload, dict):
            raise ValueError("invalid metadata")
        # A bounded first page is not a complete catalog. Keep the prior snapshot
        # on pagination rather than publishing false absence or partial authority.
        if route.provider == "gemini":
            token = payload.get("nextPageToken")
            if token is not None and (not isinstance(token, str) or token):
                raise ValueError("incomplete metadata")
        if route.provider == "anthropic":
            has_more = payload.get("has_more", False)
            if not isinstance(has_more, bool) or has_more:
                raise ValueError("incomplete metadata")
        items = payload.get("models" if route.provider == "gemini" else "data")
        if not isinstance(items, list) or len(items) > 10000:
            raise ValueError("invalid metadata")
        models = []
        seen = set()
        for item in items:
            value = (
                item.get("name" if route.provider == "gemini" else "id")
                if isinstance(item, dict)
                else None
            )
            if not isinstance(value, str) or not value or len(value) > 1024:
                raise ValueError("invalid metadata model")
            model_id = value.removeprefix("models/")
            if model_id in seen:
                raise ValueError("duplicate metadata model")
            seen.add(model_id)
            models.append(_provider_metadata(route.provider, model_id, item))
        return tuple(models)

    async def shutdown(self) -> None:
        if self._shutdown_task is None:
            self._ready = False
            self._shutdown_task = asyncio.create_task(self._shutdown())
        await asyncio.shield(self._shutdown_task)

    async def _shutdown(self) -> None:
        self._ready = False
        for refresh_task in self._refresh_tasks.values():
            refresh_task.cancel()
        await self.admission.shutdown()
        try:
            await asyncio.wait_for(
                self._work_empty.wait(), self.snapshot.limits.shutdown_drain_seconds
            )
        except asyncio.TimeoutError:
            pass
        self._cleanup_deadline = time.monotonic() + self.snapshot.limits.shutdown_cleanup_seconds
        if self.active_work:
            caller = asyncio.current_task()
            owners = tuple(self.active_work.values())
            tasks = [
                asyncio.create_task(owner.cancel(cancel_owner=owner.owner_task is not caller))
                for owner in owners
            ]
            for task in tasks:
                self._cleanup_tasks.add(task)
                task.add_done_callback(self._cleanup_finished)
            await asyncio.wait(tasks, timeout=self._cleanup_remaining())
        self._retired.append(self._generation)
        await self._retire_unused()
        cleanup = asyncio.create_task(self._close_dependencies())
        self._cleanup_tasks.add(cleanup)
        cleanup.add_done_callback(self._cleanup_finished)
        pending = {*self._cleanup_tasks, *self._refresh_tasks.values()}
        if pending:
            await asyncio.wait(pending, timeout=self._cleanup_remaining())

    async def _close_dependencies(self) -> None:
        await self._work_empty.wait()
        # Catalog refreshes aren't logical inference operations, but their
        # captured generations can still be using these injected dependencies.
        await self._generations_empty.wait()
        await self.resources.clear()
        for dependency, owned in (
            (self.dependencies.http_client, self._generation.http_client),
            (self.dependencies.broker, self._generation.broker),
        ):
            close = getattr(dependency, "aclose", None)
            if dependency is not owned and close is not None:
                await close()

    def _build_generation(self, snapshot: GatewayConfigSnapshot, number: int) -> RuntimeGeneration:
        authorities = tuple(
            (
                c.id,
                _fingerprint(
                    {
                        "id": c.id,
                        "provider": c.provider,
                        "source": c.source.model_dump(),
                        "origins": c.allowed_origins,
                        "paths": c.allowed_path_prefixes,
                        "owner": c.owner_group or c.id,
                        "billing": c.billing_group or c.id,
                    }
                ),
            )
            for c in snapshot.credentials
        )
        targets = tuple(
            (
                r.id,
                _fingerprint(
                    {
                        "provider": r.provider,
                        "origin": r.upstream_origin,
                        "path": r.upstream_path_prefix,
                        "model": r.upstream_model,
                        "protocols": r.native_protocols,
                        "contract": r.body_contract,
                    }
                ),
            )
            for r in snapshot.routes
        )
        credentials = {c.id: c for c in snapshot.credentials}
        records = tuple(
            AccountAvailability(
                r.id,
                account,
                dict(authorities)[account],
                _source_state(credentials[account], self._environ),
                r.catalog.entitlements.get(account, "unknown"),
                r.catalog.source,
                metadata_available=r.catalog.source == "configured",
            )
            for r in snapshot.routes
            for account in r.credentials
        )
        catalog = CatalogSnapshot(
            snapshot, accounts=records, generation=number, now=self.dependencies.clock()
        )
        authenticator = GatewayAuthenticator(snapshot, self._environ)
        context = tls_context(snapshot)
        return RuntimeGeneration(
            number,
            snapshot,
            authenticator,
            GatewayAuthorizer(snapshot, catalog),
            catalog,
            CredentialBroker.from_snapshot(snapshot, environ=self._environ),
            _fingerprint(snapshot.redacted_dict()),
            authorities,
            targets,
            http_client(snapshot),
            context,
            self.dependencies.egress_policy,
        )
