"""HEADROOM_BACKGROUND_COMPRESSION_WORKERS sizes the off-path compression pool.

The background (off-path, Phase 3 / #1171) compression executor was hardcoded to
a single worker, so a burst of concurrent cold-start sessions drained one job at
a time. This locks that the pool size is env-configurable, defaults to 1, and
clamps junk/low values to at least 1.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from headroom.proxy.server import ProxyConfig, create_app


def _make_proxy():
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    return create_app(config).state.proxy


def test_default_is_single_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADROOM_BACKGROUND_COMPRESSION_WORKERS", raising=False)
    proxy = _make_proxy()
    assert proxy._background_compression_workers == 1
    assert proxy._background_compression_executor._max_workers == 1


def test_env_overrides_worker_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_BACKGROUND_COMPRESSION_WORKERS", "3")
    proxy = _make_proxy()
    assert proxy._background_compression_workers == 3
    assert proxy._background_compression_executor._max_workers == 3


@pytest.mark.parametrize("bad", ["0", "-4", "notanint"])
def test_junk_and_low_values_clamp_to_one(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.setenv("HEADROOM_BACKGROUND_COMPRESSION_WORKERS", bad)
    proxy = _make_proxy()
    assert proxy._background_compression_workers == 1
