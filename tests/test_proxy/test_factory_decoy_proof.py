"""Local decoy proof for Factory Droid routing, over real sockets.

Two local HTTP servers stand in for the outside world:

* a **decoy**, configured as the proxy's Anthropic and OpenAI upstream (and
  allowlisted as a client-named base), which must stay silent for every
  Factory-bound request, and
* a **Factory stand-in**, configured as ``factory_api_url``, which must
  receive Droid's inference request and its non-inference REST calls.

The proxy runs under a real uvicorn server. ``headroom wrap droid --no-proxy``
then probes real listeners and either refuses or launches a fake ``droid``
executable. Nothing here contacts Factory or needs a Factory token.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import stat
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
uvicorn = pytest.importorskip("uvicorn")

from click.testing import CliRunner  # noqa: E402

from headroom.cli.main import main  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

MESSAGES = "/api/llm/a/v1/messages"
UNCONTACTED_FACTORY = "https://factory.invalid"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class _Seen:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes


@dataclass
class _Upstream:
    name: str
    server: ThreadingHTTPServer
    seen: list[_Seen] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"


def _start_upstream(name: str) -> _Upstream:
    upstream: _Upstream

    class _Handler(BaseHTTPRequestHandler):
        def _serve(self) -> None:
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length) if length else b""
            upstream.seen.append(
                _Seen(
                    self.command,
                    self.path,
                    {k.lower(): v for k, v in self.headers.items()},
                    body,
                )
            )
            if self.path.split("?")[0].endswith("/v1/messages"):
                reply: dict[str, Any] = {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5",
                    "content": [{"type": "text", "text": f"from {name}"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 3},
                }
            else:
                reply = {"served_by": name, "path": self.path}
            payload = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_GET = do_POST = do_PATCH = do_PUT = do_DELETE = _serve

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    upstream = _Upstream(name, server)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return upstream


class _ProxyThread:
    def __init__(self, config: ProxyConfig) -> None:
        self.port = _free_port()
        config.port = self.port
        self.server = uvicorn.Server(
            uvicorn.Config(
                create_app(config),
                host="127.0.0.1",
                port=self.port,
                log_level="warning",
                loop="asyncio",
                lifespan="on",
                ws="none",
            )
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self.thread.start()
        deadline = time.perf_counter() + 30.0
        while time.perf_counter() < deadline:
            if self.server.started:
                return
            time.sleep(0.05)
        raise RuntimeError("proxy failed to start within 30s")

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10.0)


def _config(**overrides: Any) -> ProxyConfig:
    base: dict[str, Any] = {
        "optimize": True,
        "cache_enabled": False,
        "rate_limit_enabled": False,
        "cost_tracking_enabled": False,
        "log_requests": False,
    }
    base.update(overrides)
    return ProxyConfig(**base)


@dataclass
class _Stack:
    decoy: _Upstream
    factory: _Upstream
    factory_proxy: _ProxyThread
    plain_proxy: _ProxyThread
    uncontacted_factory_proxy: _ProxyThread


@pytest.fixture
def stack(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[_Stack]:
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / ".headroom"))
    # Allowlist the loopback decoy as a client-named base, so a spoofed
    # x-headroom-base-url pointing at it would pass the SSRF guard if it were
    # honoured. The decoy staying silent then proves the route ignores it.
    monkeypatch.setenv("HEADROOM_ALLOWED_BASE_URLS", "127.0.0.1")
    for name in ("FACTORY_API_BASE_URL", "FACTORY_TARGET_API_URL"):
        monkeypatch.delenv(name, raising=False)
    saved_signals = {
        sig: signal.getsignal(sig)
        for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None))
        if sig is not None
    }

    decoy = _start_upstream("decoy")
    factory = _start_upstream("factory")
    # Provider targets are process-wide, so every in-process proxy gets the
    # decoy; otherwise the last one created would reset them to the public
    # defaults and stray traffic would leave the machine.
    decoy_targets = {"anthropic_api_url": decoy.url, "openai_api_url": decoy.url}
    proxies = [
        _ProxyThread(_config(factory_api_url=factory.url, **decoy_targets)),
        _ProxyThread(_config(**decoy_targets)),
        _ProxyThread(_config(factory_api_url=UNCONTACTED_FACTORY, **decoy_targets)),
    ]
    try:
        for proxy in proxies:
            proxy.start()
        yield _Stack(decoy, factory, *proxies)
    finally:
        for proxy in proxies:
            proxy.stop()
        decoy.server.shutdown()
        factory.server.shutdown()
        for sig, handler in saved_signals.items():
            signal.signal(sig, handler)


def test_inference_reaches_factory_stand_in_and_decoy_stays_silent(stack: _Stack) -> None:
    body = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "lorem ipsum dolor " * 400}],
    }
    response = httpx.post(
        f"{stack.factory_proxy.url}{MESSAGES}",
        headers={
            "authorization": "Bearer fk-local-dummy",
            "anthropic-version": "2023-06-01",
            # Spoofed Headroom control headers: none may redirect the request.
            "x-headroom-base-url": stack.decoy.url,
            "x-headroom-provider": "openai",
            "x-headroom-target": stack.decoy.url,
        },
        json=body,
        timeout=60,
    )

    assert response.status_code == 200, response.text
    assert response.json()["content"][0]["text"] == "from factory"
    assert stack.decoy.seen == []
    [seen] = stack.factory.seen
    assert (seen.method, seen.path) == ("POST", MESSAGES)
    assert seen.headers["authorization"] == "Bearer fk-local-dummy"
    assert not [k for k in seen.headers if k.startswith("x-headroom-")]


def test_non_inference_factory_paths_pass_through(stack: _Stack) -> None:
    get = httpx.get(
        f"{stack.factory_proxy.url}/api/sessions?limit=5",
        headers={"authorization": "Bearer fk-local-dummy"},
        timeout=30,
    )
    patch = httpx.patch(
        f"{stack.factory_proxy.url}/api/sessions/s1",
        headers={"authorization": "Bearer fk-local-dummy"},
        content=b'{"title":"x"}',
        timeout=30,
    )

    assert get.json() == {"served_by": "factory", "path": "/api/sessions?limit=5"}
    assert patch.json()["served_by"] == "factory"
    assert [(s.method, s.path) for s in stack.factory.seen] == [
        ("GET", "/api/sessions?limit=5"),
        ("PATCH", "/api/sessions/s1"),
    ]
    assert stack.factory.seen[1].body == b'{"title":"x"}'
    assert stack.decoy.seen == []


def test_unmatched_traffic_never_reaches_factory(stack: _Stack) -> None:
    unrouted = httpx.get(
        f"{stack.factory_proxy.url}/totally/unrouted",
        headers={"authorization": "Bearer sk-other-tool"},
        timeout=30,
    )
    hermes = httpx.post(
        f"{stack.factory_proxy.url}/api/codex-proxy/s1/v1/responses",
        headers={"authorization": "Bearer sk-hermes"},
        json={"model": "gpt-4o-mini", "input": "hi"},
        timeout=30,
    )

    assert unrouted.json()["served_by"] == "decoy"
    assert hermes.json()["served_by"] == "decoy"
    assert stack.factory.seen == []
    assert [s.path for s in stack.decoy.seen] == [
        "/totally/unrouted",
        "/api/codex-proxy/s1/v1/responses",
    ]


@pytest.fixture
def fake_droid(tmp_path, monkeypatch: pytest.MonkeyPatch):
    record = tmp_path / "droid-env.txt"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "droid"
    script.write_text(f'#!/bin/sh\nprintf "%s" "$FACTORY_API_BASE_URL" > "{record}"\n')
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    return record


def _wrap_droid_no_proxy(port: int, upstream: str):
    return CliRunner().invoke(
        main,
        ["wrap", "droid", "--no-proxy", "--port", str(port), "--factory-api-url", upstream],
    )


def test_no_proxy_refuses_non_factory_listener(stack: _Stack, fake_droid) -> None:
    result = _wrap_droid_no_proxy(stack.plain_proxy.port, UNCONTACTED_FACTORY)

    assert result.exit_code != 0
    assert "--no-proxy requires a Headroom proxy" in result.output
    assert not fake_droid.exists()


def test_no_proxy_refuses_mismatched_factory_listener(stack: _Stack, fake_droid) -> None:
    result = _wrap_droid_no_proxy(
        stack.uncontacted_factory_proxy.port, "https://eu.factory.example"
    )

    assert result.exit_code != 0
    assert "--no-proxy requires a Headroom proxy" in result.output
    assert not fake_droid.exists()


def test_no_proxy_refuses_absent_listener(stack: _Stack, fake_droid) -> None:
    result = _wrap_droid_no_proxy(_free_port(), UNCONTACTED_FACTORY)

    assert result.exit_code != 0
    assert not fake_droid.exists()


def test_no_proxy_launches_against_matching_listener(stack: _Stack, fake_droid) -> None:
    port = stack.uncontacted_factory_proxy.port
    result = _wrap_droid_no_proxy(port, f"{UNCONTACTED_FACTORY}/")

    assert result.exit_code == 0, result.output
    assert fake_droid.read_text() == f"http://127.0.0.1:{port}"
