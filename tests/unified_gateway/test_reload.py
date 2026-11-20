from __future__ import annotations

import json
from pathlib import Path

import pytest

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
async def test_invalid_reload_keeps_previous_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-secret")
    runtime = GatewayRuntime(
        GatewayConfigSnapshot.load(EXAMPLE),
        environ={"HEADROOM_GATEWAY_CLIENT_TOKEN": "client-secret"},
    )
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"version":1}', encoding="utf-8")

    result = await runtime.reload(invalid)

    assert result.applied is False
    assert runtime.generation == 1
    assert runtime.snapshot.routes[0].id == "openai-native"


@pytest.mark.asyncio
async def test_valid_reload_publishes_one_complete_generation(tmp_path: Path) -> None:
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw["routes"][0]["public_model"] = "reloaded-model"
    path = tmp_path / "gateway.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    runtime = GatewayRuntime(
        GatewayConfigSnapshot.load(EXAMPLE),
        environ={"HEADROOM_GATEWAY_CLIENT_TOKEN": "client-secret"},
    )

    result = await runtime.reload(path)

    assert result.applied is True
    assert runtime.generation == 2
    assert runtime.snapshot.routes[0].public_model == "reloaded-model"
