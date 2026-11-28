from __future__ import annotations

from headroom.proxy.gateway.observability import GatewayEvent, GatewayObservability


def test_gateway_observability_accepts_only_bounded_content_free_dimensions() -> None:
    telemetry = GatewayObservability()
    telemetry.record(
        GatewayEvent(
            ingress_protocol="openai-responses",
            route_class="public-api",
            adapter="strict-native",
            credential_source="env",
            failure_origin="none",
            retry_reason="none",
            terminal_result="success",
        )
    )

    snapshot = telemetry.snapshot()
    assert snapshot == {
        (
            "openai-responses",
            "public-api",
            "strict-native",
            "env",
            "none",
            "none",
            "success",
        ): 1
    }
    assert "sentinel-secret-or-prompt" not in repr(snapshot)
