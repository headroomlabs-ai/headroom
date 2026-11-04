"""A Python "Kong": drives the gateway turn contract exactly as the Lua plugin does.

Algorithm (spec section 5, ``access``/``body_filter``/``log`` phases collapsed
into one synchronous ``turn``):

1. POST ``{...body, config: {session_id, ...}, gateway: {...}}`` to
   ``<headroom>/v1/compress``. ``can_redrive`` is ``allow_redrive and not
   body.stream`` - a streaming client cannot be re-driven.
2. Any transport error, non-200, or malformed answer from headroom -> FAIL OPEN:
   forward the ORIGINAL body to the provider, no relay.
3. Otherwise the provider-bound body is ``compress["body"]`` (legacy answers
   without ``body`` fall back to ``{...body, messages: compress["messages"]}``).
4. ``"redrive" in obligations`` -> LOOP PATH: call the provider, POST
   ``{turn_id, status, latency_ms, usage, response}`` to
   ``/v1/compress/response``; while ``action == "redrive"`` POST ``request`` to
   the provider and post the answer back; on ``done`` the client sees
   ``done.response`` or, when that is null, the last provider response.
5. Else PROXY PATH: forward once; if ``relay`` and ``can_relay_response`` and a
   ``turn_id`` came back, POST ``{turn_id, status, latency_ms, model, usage}``
   (usage pulled from the JSON body, or from the SSE frames the way the Lua
   ``body_filter`` does).

Headers: the client headers handed to ``turn(..., headers=...)`` travel to
headroom, and the ones headroom may need to merge (``anthropic-beta``) are
echoed as ``gateway.request_headers``. The provider-bound headers are the
constructor's ``provider_headers``, then the client's passthrough set
(``authorization``, ``x-api-key``, ``anthropic-version``, ``anthropic-beta``,
``openai-organization``, ``openai-project``), then whatever headroom returned
in ``compress["headers"]`` - which wins on the same name, on both paths.

Both clients need only ``.post(path, json=..., headers=...)`` returning an
object with ``.status_code``, ``.headers``, ``.content`` and ``.json()`` -
``fastapi.testclient.TestClient`` and ``httpx.Client`` both qualify.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from tests.gateway.fake_provider import ANTHROPIC_MESSAGES_PATH, OPENAI_CHAT_PATH, extract_usage

COMPRESS_PATH = "/v1/compress"
RESPONSE_HALF_PATH = "/v1/compress/response"

# Client headers echoed to headroom as ``gateway.request_headers`` (the Lua
# plugin's ``GATEWAY_REQUEST_HEADERS``).
GATEWAY_REQUEST_HEADERS = ("anthropic-beta",)
# Client headers copied onto the provider call (the plugin's
# ``provider_auth_header_passthrough`` set).
PROVIDER_PASSTHROUGH_HEADERS = (
    "authorization",
    "x-api-key",
    "anthropic-version",
    "anthropic-beta",
    "openai-organization",
    "openai-project",
)


def _lower(headers: dict[str, str] | None) -> dict[str, str]:
    return {str(k).lower(): v for k, v in (headers or {}).items()}


def request_headers_block(client_headers: dict[str, str] | None) -> dict[str, str] | None:
    """``gateway.request_headers`` for these client headers, or None when empty."""
    lowered = _lower(client_headers)
    picked = {name: lowered[name] for name in GATEWAY_REQUEST_HEADERS if lowered.get(name)}
    return picked or None


def required_provider_headers(compress: Any) -> dict[str, str]:
    """The ``headers`` object of a /v1/compress answer, lower-cased; ``{}`` when
    absent or malformed (older headroom)."""
    if not isinstance(compress, dict):
        return {}
    raw = compress.get("headers")
    if not isinstance(raw, dict):
        return {}
    return {str(k).lower(): str(v) for k, v in raw.items() if isinstance(v, (str, int, float))}


@dataclass
class ResponseHalfExchange:
    """One ``POST /v1/compress/response`` round trip."""

    request: dict[str, Any]
    status: int
    response: dict[str, Any] | None


@dataclass
class ProviderExchange:
    """One provider-bound request and what came back."""

    request: dict[str, Any]
    status: int
    headers: dict[str, str]
    content: bytes
    json: dict[str, Any] | None
    latency_ms: float
    request_headers: dict[str, str] = field(default_factory=dict)  # what we sent, lower-cased

    @property
    def usage(self) -> dict[str, Any] | None:
        return extract_usage(self.content, self.headers.get("content-type", ""))


@dataclass
class GatewayTurnResult:
    """Everything a test needs to assert about one gateway turn."""

    path: str  # "loop" | "proxy" | "fail_open"
    compress_status: int | None
    compress_response: dict[str, Any] | None
    provider_exchanges: list[ProviderExchange] = field(default_factory=list)
    response_half_calls: list[ResponseHalfExchange] = field(default_factory=list)
    final_status: int = 0
    final_headers: dict[str, str] = field(default_factory=dict)
    final_content: bytes = b""
    final_response: dict[str, Any] | None = None
    fail_open_reason: str | None = None

    # --- conveniences ------------------------------------------------------

    @property
    def provider_calls(self) -> list[dict[str, Any]]:
        """The provider-bound request bodies, in order."""
        return [x.request for x in self.provider_exchanges]

    @property
    def provider_headers_sent(self) -> list[dict[str, str]]:
        """The provider-bound request headers, in order (lower-cased)."""
        return [x.request_headers for x in self.provider_exchanges]

    @property
    def required_headers(self) -> dict[str, str]:
        """Headers headroom asked the gateway to set (``compress["headers"]``)."""
        return required_provider_headers(self.compress_response)

    @property
    def turn_id(self) -> str | None:
        if isinstance(self.compress_response, dict):
            tid = self.compress_response.get("turn_id")
            return tid if isinstance(tid, str) else None
        return None

    @property
    def obligations(self) -> list[str]:
        if isinstance(self.compress_response, dict):
            obs = self.compress_response.get("obligations")
            return list(obs) if isinstance(obs, list) else []
        return []

    @property
    def rounds(self) -> int:
        """Redrive rounds actually driven (provider calls beyond the first)."""
        return max(0, len(self.provider_exchanges) - 1)

    @property
    def done(self) -> dict[str, Any] | None:
        for x in reversed(self.response_half_calls):
            if isinstance(x.response, dict) and x.response.get("action") == "done":
                return x.response
        return None


def provider_path_for(body: dict[str, Any]) -> str:
    model = str(body.get("model") or "").lower()
    if "claude" in model or "anthropic" in model:
        return ANTHROPIC_MESSAGES_PATH
    return OPENAI_CHAT_PATH


class FakeGateway:
    def __init__(
        self,
        headroom_client: Any,
        provider_client: Any,
        *,
        can_redrive: bool,
        can_relay_response: bool,
        session_affinity: bool = True,
        relay: bool = True,
        plugin_version: str = "fake-gateway/0.1.0",
        headroom_headers: dict[str, str] | None = None,
        provider_headers: dict[str, str] | None = None,
        provider_path: str | None = None,
        compress_mode: str | None = None,
        max_redrives: int = 8,
    ) -> None:
        self.headroom = headroom_client
        self.provider = provider_client
        self.can_redrive = can_redrive
        self.can_relay_response = can_relay_response
        self.session_affinity = session_affinity
        self.relay = relay
        self.plugin_version = plugin_version
        self.headroom_headers = dict(headroom_headers or {})
        self.provider_headers = dict(provider_headers or {})
        self.provider_path = provider_path
        self.compress_mode = compress_mode
        self.max_redrives = max_redrives
        self.turns: list[GatewayTurnResult] = []

    # --- helpers -------------------------------------------------------------

    def gateway_block(
        self, body: dict[str, Any], client_headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        block = {
            "can_redrive": bool(self.can_redrive and not body.get("stream")),
            "can_relay_response": bool(self.can_relay_response),
            "session_affinity": bool(self.session_affinity),
            "plugin_version": self.plugin_version,
        }
        request_headers = request_headers_block(client_headers)
        if request_headers is not None:
            block["request_headers"] = request_headers
        return block

    def compress_request(
        self,
        body: dict[str, Any],
        session_id: str | None,
        config: dict[str, Any] | None,
        client_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        cfg: dict[str, Any] = dict(config or {})
        if session_id is not None:
            cfg["session_id"] = session_id
        if self.compress_mode is not None:
            cfg.setdefault("mode", self.compress_mode)
        return {**body, "config": cfg, "gateway": self.gateway_block(body, client_headers)}

    def provider_call_headers(
        self,
        client_headers: dict[str, str] | None = None,
        required: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Provider-bound headers: constructor set, client passthrough, then
        headroom's ``headers`` (which win on the same name)."""
        out = _lower(self.provider_headers)
        lowered = _lower(client_headers)
        for name in PROVIDER_PASSTHROUGH_HEADERS:
            if lowered.get(name):
                out[name] = lowered[name]
        out.update(_lower(required))
        return out

    def _call_provider(
        self, body: dict[str, Any], headers: dict[str, str] | None = None
    ) -> ProviderExchange:
        path = self.provider_path or provider_path_for(body)
        sent_headers = _lower(headers) if headers is not None else _lower(self.provider_headers)
        started = time.perf_counter()
        resp = self.provider.post(path, json=body, headers=sent_headers or None)
        latency_ms = (time.perf_counter() - started) * 1000
        headers = {k.lower(): v for k, v in resp.headers.items()}
        parsed: dict[str, Any] | None = None
        if not headers.get("content-type", "").startswith("text/event-stream"):
            try:
                candidate = resp.json()
            except ValueError:
                candidate = None
            parsed = candidate if isinstance(candidate, dict) else None
        return ProviderExchange(
            request=body,
            status=resp.status_code,
            headers=headers,
            content=resp.content,
            json=parsed,
            latency_ms=latency_ms,
            request_headers=sent_headers,
        )

    def _post_response_half(
        self, result: GatewayTurnResult, payload: dict[str, Any]
    ) -> ResponseHalfExchange:
        try:
            resp = self.headroom.post(
                RESPONSE_HALF_PATH, json=payload, headers=self.headroom_headers or None
            )
            try:
                parsed = resp.json()
            except ValueError:
                parsed = None
            exchange = ResponseHalfExchange(
                request=payload,
                status=resp.status_code,
                response=parsed if isinstance(parsed, dict) else None,
            )
        except Exception as exc:  # the plugin never lets a relay failure reach the client
            exchange = ResponseHalfExchange(
                request=payload, status=0, response={"error": {"message": str(exc)}}
            )
        result.response_half_calls.append(exchange)
        return exchange

    @staticmethod
    def _finish(result: GatewayTurnResult, exchange: ProviderExchange, override: Any = None):
        if isinstance(override, dict):
            result.final_status = 200
            result.final_headers = {"content-type": "application/json"}
            result.final_content = json.dumps(override).encode()
            result.final_response = override
        else:
            result.final_status = exchange.status
            result.final_headers = dict(exchange.headers)
            result.final_content = exchange.content
            result.final_response = exchange.json
        return result

    # --- the turn ------------------------------------------------------------

    def turn(
        self,
        body: dict[str, Any],
        session_id: str | None,
        *,
        config: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> GatewayTurnResult:
        """Run one client request through the gateway; see the module docstring."""
        req_headers = {**self.headroom_headers, **(headers or {})}
        compress_body = self.compress_request(body, session_id, config, headers)
        # Fail-open and pre-contract answers send the client's own headers only.
        client_provider_headers = self.provider_call_headers(headers)

        # --- request half ------------------------------------------------------
        try:
            resp = self.headroom.post(
                COMPRESS_PATH, json=compress_body, headers=req_headers or None
            )
            status = resp.status_code
            try:
                compress = resp.json()
            except ValueError:
                compress = None
        except Exception as exc:  # headroom unreachable -> fail open
            status, compress = None, None
            reason = f"transport: {exc!r}"
        else:
            reason = None if status == 200 and isinstance(compress, dict) else f"status {status}"

        if reason is not None or not isinstance(compress, dict):
            result = GatewayTurnResult(
                path="fail_open",
                compress_status=status,
                compress_response=compress if isinstance(compress, dict) else None,
                fail_open_reason=reason or "malformed compress response",
            )
            exchange = self._call_provider(body, client_provider_headers)
            result.provider_exchanges.append(exchange)
            self.turns.append(self._finish(result, exchange))
            return result

        provider_body = compress.get("body")
        if not isinstance(provider_body, dict):
            # Legacy (pre-contract) answer: only ``messages`` came back.
            provider_body = {**body, "messages": compress.get("messages", body.get("messages"))}
        turn_id = compress.get("turn_id")
        obligations = (
            compress.get("obligations") if isinstance(compress.get("obligations"), list) else []
        )
        # ``headers``: what headroom needs on the provider request (today the
        # merged ``anthropic-beta``); applied on both paths, over the client's own.
        provider_headers = self.provider_call_headers(headers, required_provider_headers(compress))

        # --- loop path -----------------------------------------------------------
        if "redrive" in obligations and isinstance(turn_id, str):
            result = GatewayTurnResult(
                path="loop", compress_status=status, compress_response=compress
            )
            exchange = self._call_provider(provider_body, provider_headers)
            result.provider_exchanges.append(exchange)
            final_override: Any = None
            round_no = 0  # which provider call this answers; the plugin sends it too
            for _ in range(self.max_redrives + 1):
                if exchange.json is None:
                    break  # non-JSON provider answer: nothing headroom can drive
                step = self._post_response_half(
                    result,
                    {
                        "turn_id": turn_id,
                        "round": round_no,
                        "status": exchange.status,
                        "latency_ms": exchange.latency_ms,
                        "model": exchange.json.get("model"),
                        "usage": exchange.usage,
                        "response": exchange.json,
                    },
                )
                answer = step.response or {}
                if step.status != 200:
                    break  # headroom refused/expired: client gets the last provider answer
                if answer.get("action") == "redrive" and isinstance(answer.get("request"), dict):
                    round_no = int(answer.get("round", round_no + 1))
                    exchange = self._call_provider(answer["request"], provider_headers)
                    result.provider_exchanges.append(exchange)
                    continue
                if answer.get("action") == "done":
                    final_override = answer.get("response")
                break
            self.turns.append(self._finish(result, exchange, final_override))
            return result

        # --- proxy path ----------------------------------------------------------
        result = GatewayTurnResult(path="proxy", compress_status=status, compress_response=compress)
        exchange = self._call_provider(provider_body, provider_headers)
        result.provider_exchanges.append(exchange)
        self._finish(result, exchange)
        if self.relay and self.can_relay_response and isinstance(turn_id, str):
            model = exchange.json.get("model") if isinstance(exchange.json, dict) else None
            self._post_response_half(
                result,
                {
                    "turn_id": turn_id,
                    "status": exchange.status,
                    "latency_ms": exchange.latency_ms,
                    "model": model,
                    "usage": exchange.usage,
                },
            )
        self.turns.append(result)
        return result
