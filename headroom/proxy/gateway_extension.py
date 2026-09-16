"""The gateway turn contract as a compress-turn extension.

Everything a gateway (Kong, Envoy, LiteLLM, Cloudflare …) needs from
``POST /v1/compress`` — the ``gateway`` capability block, the full provider
``body``, ``turn_id`` / ``route`` / ``obligations`` / ``headers``, the
``POST /v1/compress/response`` second half, the pending-turn registry and the
parked re-drive hooks — is installed here through the same seams a third-party
contract would use (:mod:`headroom.proxy.compress_turn`). The contract LOGIC
stays in :mod:`headroom.proxy.gateway_turn`; this module is the glue that used
to be inlined in the compress handler, plus the route and registry setup.

Installed by ``create_app`` by default; ``HEADROOM_GATEWAY_CONTRACT=0`` leaves
it out, in which case a body's ``gateway`` block is ignored and the endpoint
answers in legacy shape (a gateway then fails open on its side).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from starlette.requests import Request

from headroom.proxy.compress_turn import (
    CompressTurnError,
    FinishedTurn,
    register_compress_turn_extension,
    registered_compress_turn_extensions,
)
from headroom.proxy.gateway_turn import (
    OBLIGATION_RELAY_USAGE,
    GatewayCapabilities,
    GatewayRequestError,
    PendingTurnRegistry,
    RequestTransformer,
    arm_ccr_redrive,
    compute_obligations,
    gateway_response_fields,
    handle_compress_response,
    new_turn_id,
    on_turn_expired,
    parse_gateway_block,
    provider_for_model,
    register_pending_turn,
    shape_gateway_body,
)
from headroom.proxy.outcome import RequestOutcome

log = logging.getLogger(__name__)

ENV_GATEWAY_CONTRACT = "HEADROOM_GATEWAY_CONTRACT"
EXTENSION_NAME = "gateway_turn_contract"


def gateway_contract_enabled() -> bool:
    return os.environ.get(ENV_GATEWAY_CONTRACT, "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


class GatewayTurn:
    """One gateway-mode request, from ``begin`` to ``commit``."""

    def __init__(
        self,
        *,
        proxy: Any,
        request: Any,
        body: dict[str, Any],
        caps: GatewayCapabilities,
        client: str | None,
    ) -> None:
        self.proxy = proxy
        self.request = request
        self.body = body
        self.caps = caps
        self.client = client
        self.provider = provider_for_model(str(body.get("model") or ""))
        self.model_name = str(body.get("model") or "")
        self.tags: dict[str, Any] = {}
        self._transformer: RequestTransformer | None = None
        self._fields: dict[str, Any] | None = None
        self._obligations: list[str] = []
        self._turn_id: str | None = None
        self._ccr_armed = False

    # -- fail-open ---------------------------------------------------------
    def fail_open_fields(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        # Originals, no obligations: the gateway forwards the body as sent.
        return gateway_response_fields(
            request=self.request,
            body=self.body,
            caps=self.caps,
            provider=self.provider,
            messages=messages,
            tools=self.body.get("tools"),
            obligations=[],
            turn_id=new_turn_id(),
        )

    # -- before the executor ----------------------------------------------
    def prepare(self, *, model_name: str, tags: dict[str, Any], config: Any) -> None:
        from headroom.proxy.savings_attribution import bind_scope

        # bind_scope first so a hook's savings attribute to this outcome,
        # exactly as the chat path does.
        bind_scope(tags, self.request.scope)
        self.model_name = model_name
        self.tags = tags
        self.provider = provider_for_model(model_name)
        self._transformer = RequestTransformer(
            provider=self.provider,
            model_name=model_name,
            tools=self.body.get("tools"),
            config=config,
            tags=tags,
            caps=self.caps,
        )

    # -- executor side -----------------------------------------------------
    def transform(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self._transformer is None:
            return messages
        return self._transformer.run(messages)

    @property
    def folded_messages(self) -> bool:
        return self._transformer is not None and self._transformer.folded_messages

    def count_messages(self, messages: list[dict[str, Any]], fallback: int) -> int:
        if self._transformer is None:
            return fallback
        return self._transformer.count_messages(messages, fallback)

    # -- event-loop side ---------------------------------------------------
    def finish(
        self,
        *,
        messages: list[dict[str, Any]],
        tokens_before: int,
        transforms_applied: list[str],
        mode: str | None,
        ccr_hashes: list[str],
    ) -> FinishedTurn:
        result = self._transformer.result if self._transformer is not None else None
        if result is None:
            return FinishedTurn(fields={}, transforms=list(transforms_applied), messages=None)
        # mode="ccr" markers are only useful if the model can resolve them:
        # inject headroom_retrieve and let the response half answer it (a
        # re-drive), or leave it to the caller otherwise.
        self._ccr_armed = arm_ccr_redrive(
            result,
            provider=self.provider,
            caps=self.caps,
            mode=mode,
            ccr_hashes=ccr_hashes,
        )
        transforms = list(transforms_applied or ()) + list(result.transforms)
        # Gateway fields are built BEFORE the outcome: output shaping runs on
        # the provider body and its (arm, stratum) label must be on the
        # outcome's transforms for the response half's relayed output_tokens
        # to become output savings.
        self._obligations = compute_obligations(
            self.caps, redrive_armed=result.redrive_armed or self._ccr_armed
        )
        self._turn_id = new_turn_id()
        self._fields = gateway_response_fields(
            request=self.request,
            body=self.body,
            caps=self.caps,
            provider=self.provider,
            messages=messages,
            tools=result.tools,
            obligations=self._obligations,
            turn_id=self._turn_id,
            headers=result.headers,
        )
        shape_gateway_body(
            self._fields["body"],
            provider=self.provider,
            model_name=self.model_name,
            config=self.proxy.config,
            input_tokens=tokens_before,
            transforms=transforms,
        )
        # I-BODY: top-level `messages` and `body.messages` are one list.
        return FinishedTurn(
            fields=self._fields,
            transforms=transforms,
            messages=self._fields["body"]["messages"],
        )

    def commit(
        self,
        outcome: RequestOutcome,
        *,
        session_key: str | None,
        session_id: str | None,
    ) -> bool:
        if self._fields is None or self._transformer is None or self._transformer.result is None:
            return False
        # With relay_usage owed, the outcome waits for the response half
        # (billed input/output, cache counts, provider status); the registry
        # records the draft as-is if that never comes.
        defer = OBLIGATION_RELAY_USAGE in self._obligations
        register_pending_turn(
            self.proxy,
            turn_id=self._turn_id or new_turn_id(),
            session_key=session_key,
            session_id=session_id,
            provider=self.provider,
            model=str(self._fields["body"].get("model")),
            body=self._fields["body"],
            ctx=self._transformer.result.ctx,
            obligations=self._obligations,
            outcome_draft=outcome if defer else None,
            tags=self.tags,
            client=self.client,
            ccr_armed=self._ccr_armed,
        )
        return defer


class GatewayTurnExtension:
    """Claims every ``/v1/compress`` body that carries a ``gateway`` block."""

    name = EXTENSION_NAME

    def begin(
        self, *, proxy: Any, request: Any, body: dict[str, Any], client: str | None
    ) -> GatewayTurn | None:
        try:
            caps = parse_gateway_block(body)
        except GatewayRequestError as e:
            raise CompressTurnError(e.message) from e
        if caps is None:
            return None
        return GatewayTurn(proxy=proxy, request=request, body=body, caps=caps, client=client)


def install(app: Any, config: Any, *, route_dependencies: list[Any] | None = None) -> None:
    """Install the contract: registry on the proxy, the extension, the route.

    ``route_dependencies`` is the same FastAPI dependency list ``/v1/compress``
    uses (loopback-only unless ``HEADROOM_COMPRESS_ALLOW_REMOTE``), so the two
    halves keep one exposure policy. Idempotent per app.
    """
    proxy = getattr(app.state, "proxy", None)
    if proxy is None:
        raise RuntimeError("gateway contract: app.state.proxy is not set yet")
    if getattr(proxy, "gateway_turns", None) is None:
        # Turns whose response half is still owed by an external gateway. Same
        # replica affinity as the session caches. Expiry records the deferred
        # outcome draft and drops any parked re-drive coroutine.
        proxy.gateway_turns = PendingTurnRegistry(
            on_expire=lambda turn: on_turn_expired(proxy, turn)
        )
    if not any(
        getattr(ext, "name", None) == EXTENSION_NAME
        for ext in registered_compress_turn_extensions(proxy)
    ):
        register_compress_turn_extension(proxy, GatewayTurnExtension())

    # Response half: the gateway posts the provider's answer (or just its
    # usage) under the turn_id the request half returned. Same exposure
    # policy as /v1/compress — the two halves are one contract.
    async def compress_response(request: Request) -> Any:
        return await handle_compress_response(proxy, request)

    app.add_api_route(
        "/v1/compress/response",
        compress_response,
        methods=["POST"],
        dependencies=list(route_dependencies or []),
        name="compress_response",
    )
    log.info(
        "gateway turn contract installed (request half via compress-turn seam, response half at /v1/compress/response)"
    )


def install_builtin(app: Any, config: Any, *, route_dependencies: list[Any] | None = None) -> bool:
    """``create_app``'s default install, honouring ``HEADROOM_GATEWAY_CONTRACT``."""
    if not gateway_contract_enabled():
        log.info(
            "gateway turn contract disabled (%s=0); `gateway` blocks are ignored",
            ENV_GATEWAY_CONTRACT,
        )
        return False
    install(app, config, route_dependencies=route_dependencies)
    return True
