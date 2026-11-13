from __future__ import annotations

import pytest

from headroom.proxy.gateway.dispatch import GatewayDispatcher, rewrite_routed_native_model
from headroom.proxy.gateway.errors import GatewayAuthorizationError
from headroom.proxy.gateway.models import Capability


def test_strict_native_does_not_parse_or_reserialize_body() -> None:
    body = b'{ "model":"same", "unknown":1.00, "unicode":"\xe2\x98\x83" }'

    assert rewrite_routed_native_model(body, public_model="same", upstream_model="same") is body


def test_routed_native_changes_only_model_field() -> None:
    body = b'{"model":"public","input":"hello"}'

    rewritten = rewrite_routed_native_model(
        body,
        public_model="public",
        upstream_model="provider-model",
    )

    assert rewritten == b'{"model":"provider-model","input":"hello"}'


def test_routed_native_rejects_body_whose_model_disagrees_with_route() -> None:
    with pytest.raises(GatewayAuthorizationError, match="model"):
        rewrite_routed_native_model(
            b'{"model":"different"}',
            public_model="public",
            upstream_model="provider-model",
        )


def test_dispatch_plan_declares_routed_native_model_mutation() -> None:
    plan = GatewayDispatcher.resolve_body(
        b'{ "model" : "public-model", "input" : "hi" }',
        public_model="public-model",
        upstream_model="provider-model",
        declared_contract="routed-native",
    )

    assert plan.contract == "routed-native"
    assert plan.capabilities == frozenset({Capability.GENERATE})
    assert plan.mutation_reasons == ("model_alias",)
    assert plan.body == b'{ "model" : "provider-model", "input" : "hi" }'


def test_strict_native_dispatch_plan_preserves_entity_identity() -> None:
    original = b'{ "model" : "same-model", "unknown" : 1.00 }'

    plan = GatewayDispatcher.resolve_body(
        original,
        public_model="same-model",
        upstream_model="same-model",
        declared_contract="strict-native",
    )

    assert plan.contract == "strict-native"
    assert plan.mutation_reasons == ()
    assert plan.body is original
