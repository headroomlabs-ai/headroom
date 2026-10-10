"""Offline checks for the Copilot enterprise test kit scripts in tools/copilot-test."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit, urlunsplit

import httpx
import pytest

from headroom import copilot_auth

KIT = Path(__file__).resolve().parents[1] / "tools" / "copilot-test"
LEAKED = "gho_ECHOEDSECRET123"


class _FakeProvider:
    def __init__(self, token: str | None, api_url: str = "https://api.example.test") -> None:
        self.token = token
        self.api_url = api_url

    async def get_api_token(self, *, integration_id: str | None = None) -> SimpleNamespace:
        if self.token is None:
            raise RuntimeError("no credential")
        return SimpleNamespace(token=self.token, api_url=self.api_url)


class _FakeClient:
    """Stands in for httpx.Client; records every request the doctor makes."""

    requests: list[tuple[str, dict[str, str]]] = []
    broken_model: str | None = None  # this model's reply is malformed JSON

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def get(self, url: str, headers: dict[str, str] | None = None) -> httpx.Response:
        self.requests.append((url, dict(headers or {})))
        return httpx.Response(200, json={"data": [{"id": "gpt-4o"}]})

    def post(self, url: str, headers: dict[str, str] | None = None, json: object = None):
        self.requests.append((url, dict(headers or {})))
        model = json["model"]  # type: ignore[index]
        if model == self.broken_model:
            return httpx.Response(
                200, content=b"<html>", headers={"content-type": "application/json"}
            )
        return httpx.Response(400, json={"error": {"message": f"bad token {LEAKED}"}})


def _run_doctor(
    monkeypatch: pytest.MonkeyPatch,
    token: str | None,
    broken_model: str | None = None,
    api_url: str = "https://api.example.test",
) -> list[tuple[str, dict]]:
    _FakeClient.requests = []
    _FakeClient.broken_model = broken_model
    monkeypatch.setenv("GITHUB_COPILOT_INTEGRATION_ID", "my-cli")
    monkeypatch.setattr(
        copilot_auth, "get_copilot_token_provider", lambda: _FakeProvider(token, api_url)
    )
    monkeypatch.setattr(copilot_auth, "iter_oauth_token_candidates", lambda: [])
    monkeypatch.setattr(copilot_auth, "read_cached_oauth_token", lambda: None)
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    runpy.run_path(str(KIT / "copilot_doctor.py"), run_name="__main__")
    return _FakeClient.requests


def test_doctor_probes_with_the_token_integration_id(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = _run_doctor(monkeypatch, "tid_minted_for_my_cli")

    assert requests  # catalog + inference probes ran
    assert {h["Copilot-Integration-Id"] for _, h in requests} == {"my-cli"}


def test_doctor_survives_a_failed_probe_and_redacts_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _run_doctor(monkeypatch, "tid_minted_for_my_cli", broken_model="gpt-5.5")
    out = capsys.readouterr().out

    # gpt-5.5 answered with malformed JSON, yet claude was still probed and the verdict printed.
    assert "claude-sonnet-4.6" in out.split("[7]")[1]
    assert "VERDICT" in out
    assert "request error: JSONDecodeError" in out
    # The upstream error echoed a credential; only its type prefix may be shown.
    assert "gho_…" in out
    assert "ECHOEDSECRET" not in out


def test_doctor_reports_no_token_when_discovery_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _run_doctor(monkeypatch, None)
    out = capsys.readouterr().out

    assert "Token forwarded     : none" in out


def test_harness_refuses_an_occupied_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200))

    def no_spawn(*args: object, **kwargs: object) -> None:
        raise AssertionError("must not start a proxy on an occupied port")

    monkeypatch.setattr("subprocess.Popen", no_spawn)

    with pytest.raises(SystemExit, match="already in use"):
        runpy.run_path(str(KIT / "enterprise_proxy_test.py"), run_name="__main__")


def test_harness_stops_when_its_proxy_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the spawned proxy dies, a server that comes up later must not be probed."""

    def port_free(*args: object, **kwargs: object) -> None:
        raise httpx.ConnectError("refused")

    exited = SimpleNamespace(poll=lambda: 1, terminate=lambda: None, wait=lambda timeout: 1)
    monkeypatch.setattr(httpx, "get", port_free)
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: exited)
    _FakeClient.requests = []
    monkeypatch.setattr(httpx, "Client", _FakeClient)

    with pytest.raises(SystemExit, match="did not become ready"):
        runpy.run_path(str(KIT / "enterprise_proxy_test.py"), run_name="__main__")
    assert _FakeClient.requests == []


# One synthetic secret per URL component that can carry a credential.
SECRET_URLS = {
    "userinfo": ("https://user:USERINFO_SECRET@api.example.test:8443/v1", "USERINFO_SECRET"),
    "query": ("https://api.example.test:8443/v1?access_token=QUERY_SECRET", "QUERY_SECRET"),
    "fragment": ("https://api.example.test:8443/v1#FRAGMENT_SECRET", "FRAGMENT_SECRET"),
}


@pytest.mark.parametrize("kind", sorted(SECRET_URLS))
def test_doctor_output_drops_url_credentials(
    kind: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Env display, outbound request line and verdict show the host, never URL secrets."""
    url, secret = SECRET_URLS[kind]
    monkeypatch.setenv("GITHUB_COPILOT_API_URL", url)
    monkeypatch.setenv("GITHUB_COPILOT_ENTERPRISE_URL", url)

    requests = _run_doctor(monkeypatch, "tid_minted_for_my_cli", api_url=url)
    out = capsys.readouterr().out

    assert secret not in out
    assert "user:" not in out
    env_section = out.split("[1]")[1].split("[2]")[0]
    assert "GITHUB_COPILOT_API_URL" in env_section
    assert "https://api.example.test:8443/v1" in env_section
    assert "POST https://api.example.test:8443/v1/chat/completions" in out
    assert "API host            : https://api.example.test:8443/v1" in out
    # Requests still go to the original, unsanitized endpoint.
    assert requests
    assert all(u.startswith(url) for u, _ in requests)


class _FakeProc:
    def __init__(self) -> None:
        self.env: dict[str, str] = {}

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        return None

    def wait(self, timeout: float) -> int:
        return 0


@pytest.mark.parametrize("kind", sorted(SECRET_URLS))
def test_harness_output_drops_url_credentials(
    kind: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Banner, capture lines and verdict show the host, never URL secrets."""
    url, secret = SECRET_URLS[kind]
    cap = Path("/tmp/hr_enterprise_capture.jsonl")
    proc = _FakeProc()

    def spawn(*args: object, env: dict[str, str], **kwargs: object) -> _FakeProc:
        proc.env = env
        # A capture record carrying the raw URL: the harness must not echo its secrets.
        parts = urlsplit(url)
        raw = urlunsplit(parts._replace(path=parts.path + "/chat/completions"))
        record = {"host": "api.example.test:8443", "url": raw}
        record.update(auth_scheme="Bearer", token_kind="tid_***")
        cap.write_text(json.dumps(record) + "\n")
        return proc

    def port_free(*args: object, **kwargs: object) -> None:
        raise httpx.ConnectError("refused")

    monkeypatch.setenv("GITHUB_COPILOT_API_URL", url)
    monkeypatch.setattr(httpx, "get", port_free)
    monkeypatch.setattr("subprocess.Popen", spawn)
    _FakeClient.requests = []
    _FakeClient.broken_model = None
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    try:
        runpy.run_path(str(KIT / "enterprise_proxy_test.py"), run_name="__main__")
    finally:
        cap.unlink(missing_ok=True)
    out = capsys.readouterr().out

    assert secret not in out
    assert "user:" not in out
    assert "host = https://api.example.test:8443/v1" in out
    assert "→ https://api.example.test:8443/v1/chat/completions" in out
    assert "host reached        : https://api.example.test:8443/v1" in out
    # The proxy is still pointed at the original, unsanitized endpoint.
    assert proc.env["GITHUB_COPILOT_API_URL"] == url
    assert proc.env["OPENAI_TARGET_API_URL"] == url
