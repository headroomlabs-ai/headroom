"""The Headroom for Teams offer shows on unlicensed installs only.

"Licensed" means the running proxy's effective licence, from any source the
proxy accepts: an explicit ``ProxyConfig.license_key``, ``HEADROOM_LICENSE``, or
the deprecated ``HEADROOM_LICENSE_KEY`` alias.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from headroom.dashboard import get_dashboard_html
from headroom.proxy.server import ProxyConfig, create_app

_LICENSE_ENV = ("HEADROOM_LICENSE", "HEADROOM_LICENSE_KEY")


@pytest.fixture(autouse=True)
def _no_license_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _LICENSE_ENV:
        monkeypatch.delenv(name, raising=False)


def _assert_offer(html: str, shown: bool) -> None:
    assert ('data-testid="teams-perk"' in html) is shown
    assert ("headroom-perks.vercel.app" in html) is shown
    assert ("teams-perk:start" in html) is shown
    # Only the offer block ever goes; the dashboard around it is intact.
    assert 'data-testid="session-view"' in html
    assert html.count("<main") == 1


def _dashboard(monkeypatch: pytest.MonkeyPatch, **config: object) -> str:
    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            http2=False,
            **config,
        )
    )
    with TestClient(app) as client:
        response = client.get("/dashboard")
    assert response.status_code == 200
    return response.text


# --- renderer ---------------------------------------------------------------


def test_offer_shown_without_license() -> None:
    _assert_offer(get_dashboard_html(), shown=True)


@pytest.mark.parametrize("env", _LICENSE_ENV)
def test_offer_removed_with_license_env(monkeypatch: pytest.MonkeyPatch, env: str) -> None:
    monkeypatch.setenv(env, "hlk_test")
    _assert_offer(get_dashboard_html(), shown=False)


def test_explicit_licensed_flag_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _assert_offer(get_dashboard_html(licensed=True), shown=False)
    monkeypatch.setenv("HEADROOM_LICENSE", "hlk_test")
    _assert_offer(get_dashboard_html(licensed=False), shown=True)


# --- /dashboard route: the effective licence of the running proxy -------------


def test_route_shows_offer_unlicensed(monkeypatch: pytest.MonkeyPatch) -> None:
    _assert_offer(_dashboard(monkeypatch), shown=True)


def test_route_hides_offer_with_configured_license_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _assert_offer(_dashboard(monkeypatch, license_key="hlk_configured"), shown=False)


@pytest.mark.parametrize("env", _LICENSE_ENV)
def test_route_hides_offer_with_license_env(monkeypatch: pytest.MonkeyPatch, env: str) -> None:
    monkeypatch.setenv(env, "hlk_test")
    _assert_offer(_dashboard(monkeypatch), shown=False)
