from __future__ import annotations

from pathlib import Path

from headroom.proxy.gateway.config import GatewayConfigSnapshot, gateway_proxy_overrides
from headroom.proxy.models import ProxyConfig

EXAMPLE = (
    Path(__file__).parents[2]
    / "docs"
    / "proposals"
    / "unified-api-gateway"
    / "examples"
    / "gateway.api-keys.json"
)


def test_gateway_profile_disables_every_hidden_transform() -> None:
    overrides = gateway_proxy_overrides(GatewayConfigSnapshot.load(EXAMPLE))

    assert overrides == {
        "host": "127.0.0.1",
        "port": 8787,
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
        "stateless": False,
    }


def test_proxy_config_applies_gateway_profile_for_non_cli_callers() -> None:
    snapshot = GatewayConfigSnapshot.load(EXAMPLE)

    config = ProxyConfig(
        gateway=snapshot,
        host="0.0.0.0",
        optimize=True,
        cache_enabled=True,
        memory_enabled=True,
        traffic_learning_enabled=True,
        ccr_inject_tool=True,
        retry_enabled=True,
    )

    assert config.host == "127.0.0.1"
    assert config.optimize is False
    assert config.cache_enabled is False
    assert config.memory_enabled is False
    assert config.traffic_learning_enabled is False
    assert config.ccr_inject_tool is False
    assert config.retry_enabled is False
