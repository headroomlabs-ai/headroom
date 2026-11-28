from __future__ import annotations

from pathlib import Path

import pytest

from headroom.proxy.gateway.admission import AdmissionRequest
from headroom.proxy.gateway.config import GatewayConfigSnapshot
from headroom.proxy.gateway.runtime import GatewayRuntime

EXAMPLE = (
    Path(__file__).parents[2]
    / "docs"
    / "proposals"
    / "unified-api-gateway"
    / "examples"
    / "gateway.api-keys.json"
)


@pytest.mark.asyncio
async def test_shutdown_stops_new_admission_and_releases_runtime_state() -> None:
    runtime = GatewayRuntime(
        GatewayConfigSnapshot.load(EXAMPLE),
        environ={"HEADROOM_GATEWAY_CLIENT_TOKEN": "client-secret"},
    )

    await runtime.shutdown()
    result = await runtime.admission.try_reserve(AdmissionRequest("local-app", estimated_cost=None))

    assert result.allowed is False
    assert result.reason == "shutdown"
