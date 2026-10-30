"""Issue #1779: the HTTPS_PROXY route that keeps Remote Control with Headroom.

``examples/claude_code_remote_control/route_to_headroom.py`` is a mitmproxy addon:
Claude Code keeps ``ANTHROPIC_BASE_URL`` unset (so its Remote Control gate passes)
and reaches Headroom through ``HTTPS_PROXY``. These tests pin the routing rules
with stand-in flow objects, so mitmproxy is not a test dependency.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from headroom.proxy.helpers import _strip_internal_headers

ADDON = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "claude_code_remote_control"
    / "route_to_headroom.py"
)


@pytest.fixture()
def addon(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.delenv("HEADROOM_RC_UPSTREAM", raising=False)
    spec = importlib.util.spec_from_file_location("route_to_headroom", ADDON)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _flow(host: str, path: str, headers: dict[str, str] | None = None) -> SimpleNamespace:
    request = SimpleNamespace(
        pretty_host=host,
        host=host,
        port=443,
        scheme="https",
        path=path,
        headers=dict(headers or {}),
    )
    return SimpleNamespace(request=request, response=SimpleNamespace(stream=False))


def test_messages_call_goes_to_headroom(addon: ModuleType) -> None:
    flow = _flow("api.anthropic.com", "/v1/messages?beta=true")
    addon.request(flow)
    assert (flow.request.scheme, flow.request.host, flow.request.port) == (
        "http",
        "127.0.0.1",
        8787,
    )
    assert flow.request.path == "/v1/messages?beta=true"
    assert flow.request.headers[addon.HOP_HEADER] == "1"


def test_count_tokens_goes_to_headroom(addon: ModuleType) -> None:
    flow = _flow("api.anthropic.com", "/v1/messages/count_tokens")
    addon.request(flow)
    assert flow.request.host == "127.0.0.1"


@pytest.mark.parametrize(
    ("host", "path"),
    [
        ("api.anthropic.com", "/api/oauth/profile"),
        ("api.anthropic.com", "/v1/models"),
        ("claude.ai", "/v1/messages"),
        ("api.anthropic.com.evil.example", "/v1/messages"),
    ],
)
def test_everything_else_is_untouched(addon: ModuleType, host: str, path: str) -> None:
    flow = _flow(host, path)
    addon.request(flow)
    assert (flow.request.scheme, flow.request.host, flow.request.port) == ("https", host, 443)
    assert addon.HOP_HEADER not in flow.request.headers


def test_second_pass_from_headroom_goes_to_anthropic(addon: ModuleType) -> None:
    """A Headroom that inherited HTTPS_PROXY must not loop back into itself."""
    flow = _flow("api.anthropic.com", "/v1/messages", {addon.HOP_HEADER: "1"})
    addon.request(flow)
    assert (flow.request.scheme, flow.request.host, flow.request.port) == (
        "https",
        "api.anthropic.com",
        443,
    )
    assert addon.HOP_HEADER not in flow.request.headers


def test_hop_marker_survives_headroom_header_stripping(addon: ModuleType) -> None:
    """Headroom strips ``x-headroom-*`` upstream; a marker named so would loop."""
    assert not addon.HOP_HEADER.startswith("x-headroom-")
    forwarded = _strip_internal_headers({addon.HOP_HEADER: "1", "x-headroom-mode": "token"})
    assert forwarded == {addon.HOP_HEADER: "1"}


def test_upstream_is_configurable(addon: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_RC_UPSTREAM", "10.0.0.5:9999")
    flow = _flow("api.anthropic.com", "/v1/messages")
    addon.request(flow)
    assert (flow.request.host, flow.request.port) == ("10.0.0.5", 9999)


def test_responses_are_streamed(addon: ModuleType) -> None:
    flow = _flow("api.anthropic.com", "/v1/messages")
    addon.responseheaders(flow)
    assert flow.response.stream is True
