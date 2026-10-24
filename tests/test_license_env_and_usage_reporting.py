"""D-12: one licence variable, and usage reporting only on explicit opt-in.

Before this change core read ``HEADROOM_LICENSE_KEY`` while every extension
read ``HEADROOM_LICENSE``, and the cloud usage reporter started whenever a key
was present. Unifying the variable without the opt-in would have switched the
reporter on in every licensed enterprise deployment. These tests pin both
halves: the variable is ``HEADROOM_LICENSE`` (old name is a warned alias), and
nothing is sent to the Headroom cloud unless ``HEADROOM_USAGE_REPORTING=1``.
"""

from __future__ import annotations

import logging

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.license_env import (  # noqa: E402
    resolve_license_token,
    usage_reporting_enabled,
)
from headroom.proxy.models import ProxyConfig  # noqa: E402
from headroom.proxy.server import create_app  # noqa: E402

CLOUD = "https://cloud.headroom.test"


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def license_log():
    # Attach directly: headroom.* loggers are not reliably visible to caplog.
    logger = logging.getLogger("headroom.license_env")
    handler = _Capture()
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)


# --- variable resolution -------------------------------------------------


def test_headroom_license_is_the_variable(license_log):
    assert resolve_license_token({"HEADROOM_LICENSE": "tok-new"}) == "tok-new"
    assert license_log.messages == []


def test_legacy_name_is_an_alias_with_a_warning(license_log):
    assert resolve_license_token({"HEADROOM_LICENSE_KEY": "tok-old"}) == "tok-old"
    assert any("license_env_deprecated" in m for m in license_log.messages)


def test_new_name_wins_on_conflict_and_says_so(license_log):
    env = {"HEADROOM_LICENSE": "tok-new", "HEADROOM_LICENSE_KEY": "tok-old"}
    assert resolve_license_token(env) == "tok-new"
    assert any("license_env_conflict" in m for m in license_log.messages)


def test_same_value_under_both_names_is_quiet(license_log):
    env = {"HEADROOM_LICENSE": "tok", "HEADROOM_LICENSE_KEY": "tok"}
    assert resolve_license_token(env) == "tok"
    assert license_log.messages == []


def test_blank_values_mean_unset():
    assert resolve_license_token({"HEADROOM_LICENSE": "  ", "HEADROOM_LICENSE_KEY": ""}) is None


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_usage_reporting_opt_in_values(value):
    assert usage_reporting_enabled({"HEADROOM_USAGE_REPORTING": value}) is True


@pytest.mark.parametrize("value", [None, "", "0", "false", "no", "off", "maybe"])
def test_usage_reporting_is_off_unless_opted_in(value):
    env = {} if value is None else {"HEADROOM_USAGE_REPORTING": value}
    assert usage_reporting_enabled(env) is False


# --- outbound traffic ----------------------------------------------------


@pytest.fixture
def cloud_calls(monkeypatch):
    """Record every request any httpx.AsyncClient sends to the fake cloud."""
    calls: list[httpx.Request] = []
    real_send = httpx.AsyncClient.send

    async def send(self, request, *args, **kwargs):
        if str(request.url).startswith(CLOUD):
            calls.append(request)
            return httpx.Response(200, json={"status": "active"}, request=request)
        return await real_send(self, request, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
    return calls


def _config(**overrides) -> ProxyConfig:
    return ProxyConfig(
        license_key="tok-licensed",
        license_cloud_url=CLOUD,
        license_report_interval=3600,
        optimize=False,
        **overrides,
    )


def test_licence_alone_sends_nothing(cloud_calls):
    app = create_app(_config())
    with TestClient(app) as client:
        client.get("/livez")
    assert cloud_calls == []


def test_opt_in_reports_to_the_cloud(cloud_calls):
    app = create_app(_config(usage_reporting=True))
    with TestClient(app) as client:
        client.get("/livez")
    assert [c.url.path for c in cloud_calls][:1] == ["/v1/license/validate"]


def test_offline_beats_opt_in(cloud_calls, monkeypatch):
    monkeypatch.setenv("HEADROOM_OFFLINE", "1")
    app = create_app(_config(usage_reporting=True))
    with TestClient(app) as client:
        client.get("/livez")
    assert cloud_calls == []
