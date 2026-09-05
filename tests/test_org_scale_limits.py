"""Org-scale sizing knobs: shared-process stores must be tunable and safe.

One Headroom process shared by many users (gateway sidecar/pool) stresses
stores that were sized for a single user's workload:

* the per-session compression-cache entry cap
  (``HEADROOM_COMPRESSION_CACHE_MAX_ENTRIES``),
* the process-wide frozen-verdicts store
  (``HEADROOM_FROZEN_VERDICTS_MAX``), and
* the session registry under churn (active sessions must survive a flood
  of transient ones — the LRU property at scale).

Registry TTL/LRU mechanics live in ``test_compression_cache_registry.py``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")


def _make_proxy():
    from headroom.proxy.server import ProxyConfig, create_app

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
    app = create_app(config)
    return app.state.proxy


# --------------------------------------------------------------------------- #
# Per-session entry cap is plumbed through and env-tunable.                    #
# --------------------------------------------------------------------------- #


def test_compression_cache_entry_cap_is_plumbed(monkeypatch) -> None:
    import headroom.proxy.server as server_mod

    monkeypatch.setattr(server_mod, "COMPRESSION_CACHE_MAX_ENTRIES", 123)
    proxy = _make_proxy()
    assert proxy._get_compression_cache("s").max_entries == 123


def test_compression_cache_entry_cap_env_parsing(monkeypatch) -> None:
    import importlib

    import headroom.proxy.helpers as helpers_mod

    monkeypatch.setenv("HEADROOM_COMPRESSION_CACHE_MAX_ENTRIES", "50000")
    importlib.reload(helpers_mod)
    assert helpers_mod.COMPRESSION_CACHE_MAX_ENTRIES == 50000

    # Floor: an absurdly small value cannot disable the cache.
    monkeypatch.setenv("HEADROOM_COMPRESSION_CACHE_MAX_ENTRIES", "1")
    importlib.reload(helpers_mod)
    assert helpers_mod.COMPRESSION_CACHE_MAX_ENTRIES == 100

    # Garbage falls back to the default.
    monkeypatch.setenv("HEADROOM_COMPRESSION_CACHE_MAX_ENTRIES", "banana")
    importlib.reload(helpers_mod)
    assert helpers_mod.COMPRESSION_CACHE_MAX_ENTRIES == 10000

    monkeypatch.delenv("HEADROOM_COMPRESSION_CACHE_MAX_ENTRIES")
    importlib.reload(helpers_mod)
    assert helpers_mod.COMPRESSION_CACHE_MAX_ENTRIES == 10000


def test_compression_cache_ttl_env_rejects_non_finite(monkeypatch) -> None:
    """'nan'/'inf' parse as floats but poison every idle comparison — they
    must fall back to the default like any other unparseable value."""
    import importlib

    import headroom.proxy.helpers as helpers_mod

    for bad in ("nan", "inf", "-inf"):
        monkeypatch.setenv("HEADROOM_COMPRESSION_CACHE_TTL_SECONDS", bad)
        importlib.reload(helpers_mod)
        assert helpers_mod.COMPRESSION_CACHE_TTL_SECONDS == 3900.0, bad

    # Below the 600s floor clamps up; above it passes through.
    monkeypatch.setenv("HEADROOM_COMPRESSION_CACHE_TTL_SECONDS", "60")
    importlib.reload(helpers_mod)
    assert helpers_mod.COMPRESSION_CACHE_TTL_SECONDS == 600.0

    monkeypatch.delenv("HEADROOM_COMPRESSION_CACHE_TTL_SECONDS")
    importlib.reload(helpers_mod)
    assert helpers_mod.COMPRESSION_CACHE_TTL_SECONDS == 3900.0


# --------------------------------------------------------------------------- #
# Frozen-verdicts store: process-wide, so it must be sizeable per deployment.  #
# --------------------------------------------------------------------------- #


def test_frozen_verdicts_cap_env(monkeypatch) -> None:
    from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

    monkeypatch.setenv("HEADROOM_FROZEN_VERDICTS_MAX", "65536")
    assert ContentRouter(ContentRouterConfig())._frozen_verdicts_max == 65536

    # Floor: cannot be sized below 256.
    monkeypatch.setenv("HEADROOM_FROZEN_VERDICTS_MAX", "1")
    assert ContentRouter(ContentRouterConfig())._frozen_verdicts_max == 256

    # Garbage falls back to the default.
    monkeypatch.setenv("HEADROOM_FROZEN_VERDICTS_MAX", "banana")
    assert ContentRouter(ContentRouterConfig())._frozen_verdicts_max == 4096

    monkeypatch.delenv("HEADROOM_FROZEN_VERDICTS_MAX")
    assert ContentRouter(ContentRouterConfig())._frozen_verdicts_max == 4096


def test_frozen_verdicts_eviction_honors_configured_cap(monkeypatch) -> None:
    from headroom.transforms.content_router import ContentRouter, ContentRouterConfig

    monkeypatch.setenv("HEADROOM_FROZEN_VERDICTS_MAX", "256")
    router = ContentRouter(ContentRouterConfig())
    for key in range(300):
        router._record_frozen_verdict(key, True)
    assert len(router._frozen_verdicts) == 256
    # FIFO: the oldest keys were evicted, the newest survive.
    assert 0 not in router._frozen_verdicts
    assert 299 in router._frozen_verdicts


# --------------------------------------------------------------------------- #
# Session registry under org-scale churn: active sessions always survive a     #
# flood of transient ones (the property that keeps busts away at capacity).    #
# --------------------------------------------------------------------------- #


def test_active_sessions_survive_transient_flood(monkeypatch) -> None:
    import headroom.proxy.server as server_mod

    monkeypatch.setattr(server_mod, "MAX_COMPRESSION_CACHE_SESSIONS", 100)
    proxy = _make_proxy()

    active = [f"active-{i}" for i in range(40)]
    active_caches = {sid: proxy._get_compression_cache(sid) for sid in active}

    # 400 transient sessions arrive interleaved with active-session traffic —
    # 4x the cap, forcing repeated capacity evictions along the way.
    for i in range(400):
        proxy._get_compression_cache(f"transient-{i}")
        if i % 5 == 0:  # active sessions keep making requests
            for sid in active:
                proxy._get_compression_cache(sid)

    # Every active session survived with its instance (and therefore its
    # byte-replay state) intact; evictions only ever hit transient sessions.
    for sid in active:
        assert proxy._get_compression_cache(sid) is active_caches[sid], (
            f"active session {sid} lost its cache to transient churn"
        )
    assert len(proxy._compression_caches) <= 100 + len(active)
