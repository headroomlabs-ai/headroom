"""Hotfix-A0 smoke tests: deployment-stage Rust core verification.

Background — Finding #2 of HEADROOM_PROXY_LOG_FINDINGS_2026_05_03.md.
A customer's production proxy was silently running without the
`headroom._core` PyO3 extension because the Docker image never built it
into the runtime layer. Diff compression failed 54 times in one day;
optimization failed 379 times. Once the failure rate hit ~100%, every
Rust port we'd shipped was providing zero customer value.

These tests pin the contract for the fix:

1. The PyO3 module exposes a `hello()` marker function returning
   ``"headroom-core"`` so the deployment smoke test has something stable
   to latch onto. Reusing an existing function instead of inventing a
   new one means we don't drift the Rust API surface for diagnostic
   purposes.

2. The proxy lifespan refuses to start when ``headroom._core`` is
   unimportable, exiting with `sysexits.h` ``EX_CONFIG`` (78) so process
   supervisors recognize this as a deliberate configuration failure
   rather than a crash they should retry forever.

3. An explicit opt-out — ``HEADROOM_REQUIRE_RUST_CORE=false`` — keeps
   the dev-time `pip install -e .` workflow alive without forcing every
   contributor to run maturin.

The opt-out test also verifies that `/health` surfaces the rust core
state so operators can alert on `rust_core != "loaded"` in production.
"""

from __future__ import annotations

import pytest


# ──────────────────────────────────────────────────────────────────────────
# 1. The Rust extension's `hello()` marker is stable and importable.
# ──────────────────────────────────────────────────────────────────────────
def test_rust_core_imports() -> None:
    """`headroom._core.hello()` returns the documented sentinel.

    The deployment smoke test (`headroom.proxy.server._check_rust_core`)
    asserts on the exact return value so a stale or mis-linked .so is
    caught — not just a complete `ImportError`. If you change the marker
    string, you also need to update the lifespan check to match.
    """
    from headroom._core import hello

    assert hello() == "headroom-core"


# ──────────────────────────────────────────────────────────────────────────
# 2. Default behavior: missing extension blocks startup with exit 78.
# ──────────────────────────────────────────────────────────────────────────
def test_proxy_refuses_to_start_when_rust_core_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When `headroom._core` raises ImportError on import and the opt-out
    env var is not set, the lifespan smoke test must call `sys.exit(78)`.

    We invoke the helper directly (rather than spinning up FastAPI)
    because the helper is the single source of truth for the policy and
    `sys.exit` propagates through the lifespan context manager
    transparently. Hitting it directly keeps the test tight and avoids
    the lifespan's 30+ side effects (OTel, Langfuse, beacon, ...).
    """
    from headroom.proxy import server

    # Ensure the env var is absent so the default fail-loud path runs.
    monkeypatch.delenv("HEADROOM_REQUIRE_RUST_CORE", raising=False)

    # Force the import to raise. Patching the symbol on the module is
    # not enough because `_check_rust_core` runs `from headroom._core
    # import hello` inside its body — we need the import machinery to
    # raise. `sys.modules` is the cleanest hook for that.
    import sys

    # Pre-purge any cached binding so the import statement re-runs.
    sys.modules.pop("headroom._core", None)
    # Sentinel that pretends to be the module but raises on attribute
    # access. `from X import Y` first looks up `X` in sys.modules; if
    # present it skips re-import and uses it directly. Setting
    # sys.modules["headroom._core"] to None forces a real ImportError.
    sys.modules["headroom._core"] = None  # type: ignore[assignment]

    try:
        with pytest.raises(SystemExit) as exc_info:
            server._check_rust_core()
        assert exc_info.value.code == 78, (
            f"expected exit code 78 (EX_CONFIG), got {exc_info.value.code!r}"
        )
    finally:
        # Restore the real module so subsequent tests can import it.
        sys.modules.pop("headroom._core", None)


# ──────────────────────────────────────────────────────────────────────────
# 3. Opt-out path: HEADROOM_REQUIRE_RUST_CORE=false → degraded mode + /health.
# ──────────────────────────────────────────────────────────────────────────
def test_proxy_starts_in_degraded_mode_when_opt_out_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the opt-out env var set, the lifespan must start successfully
    even when `headroom._core` is missing, and `/health` must surface
    `rust_core: "disabled"` so operators can detect the degraded mode.

    We spin up a real FastAPI app via `TestClient` because the spec
    requires the health endpoint to reflect the lifespan state — the
    helper alone doesn't tell us the wiring is right.
    """
    from fastapi.testclient import TestClient

    from headroom.proxy.server import ProxyConfig, create_app

    monkeypatch.setenv("HEADROOM_REQUIRE_RUST_CORE", "false")

    # Force the import to fail at lifespan time. Same trick as above:
    # sys.modules["headroom._core"] = None makes Python treat the module
    # as known-unimportable, raising ImportError on the next `from ...
    # import` without re-trying the loader.
    import sys

    sys.modules.pop("headroom._core", None)
    sys.modules["headroom._core"] = None  # type: ignore[assignment]

    config = ProxyConfig(
        optimize=False,
        image_optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
    )
    app = create_app(config)

    try:
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            payload = response.json()
            assert payload["rust_core"] == "disabled", (
                f"expected rust_core=disabled, got {payload.get('rust_core')!r}; "
                f"full payload keys: {sorted(payload.keys())}"
            )
            # The error reason should be carried through so operators can
            # see *why* the extension isn't loaded.
            assert "rust_core_error" in payload
            assert (
                "ModuleNotFoundError" in payload["rust_core_error"]
                or "ImportError" in payload["rust_core_error"]
            )
    finally:
        sys.modules.pop("headroom._core", None)


# ──────────────────────────────────────────────────────────────────────────
# 4. Happy path through the helper: real extension present → status=loaded.
# ──────────────────────────────────────────────────────────────────────────
def test_check_rust_core_returns_loaded_when_extension_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When `headroom._core` is loadable and `hello()` returns the
    expected sentinel, `_check_rust_core` returns ``("loaded", None)``.

    This covers the production-happy path so a future change that
    breaks the marker check (e.g. tightening to require a JSON dict
    return) will fail this test rather than degrading silently.
    """
    from headroom.proxy import server

    # Make sure the env var doesn't tip us into the disabled branch.
    monkeypatch.delenv("HEADROOM_REQUIRE_RUST_CORE", raising=False)

    status, error = server._check_rust_core()
    assert status == "loaded"
    assert error is None
