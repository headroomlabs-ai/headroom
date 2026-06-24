from __future__ import annotations

import json
import os
import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
import uvicorn

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from headroom.cli import init as init_cli
from headroom.install.models import ConfigScope, DeploymentManifest
from headroom.providers.codex import build_launch_env, proxy_base_url
from headroom.providers.codex.install import apply_provider_scope, build_install_env
from headroom.proxy.server import ProxyConfig, create_app


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _MockOpenAIServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int]) -> None:
        super().__init__(server_address, _MockOpenAIHandler)
        self.requests: list[dict[str, object]] = []


class _MockOpenAIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return

    def _write_json(self, status_code: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _record(self, body: dict[str, object] | None = None) -> None:
        server = self.server
        assert isinstance(server, _MockOpenAIServer)
        server.requests.append(
            {
                "method": self.command,
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "body": body,
            }
        )

    def do_GET(self) -> None:  # noqa: N802
        self._record()
        if self.path == "/v1/models":
            self._write_json(
                200,
                {
                    "object": "list",
                    "data": [{"id": "gpt-4o-mini", "object": "model", "owned_by": "openai"}],
                },
            )
            return
        self._write_json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length) if length else b""
        payload = json.loads(raw_body.decode("utf-8") or "{}")
        self._record(body=payload)
        if self.path == "/v1/chat/completions":
            self._write_json(
                200,
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": payload.get("model", "gpt-4o-mini"),
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "mock completion from upstream",
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 5,
                        "total_tokens": 17,
                    },
                },
            )
            return
        self._write_json(404, {"error": {"message": "not found"}})


class _ProxyThread:
    def __init__(self, port: int, upstream_port: int) -> None:
        app = create_app(
            ProxyConfig(
                host="127.0.0.1",
                port=port,
                optimize=False,
                cache_enabled=False,
                rate_limit_enabled=False,
                openai_api_url=f"http://127.0.0.1:{upstream_port}",
            )
        )
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            loop="asyncio",
            lifespan="on",
            ws="none",
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, name="codex-proxy", daemon=True)

    def start(self) -> None:
        self.thread.start()
        deadline = time.perf_counter() + 10.0
        while time.perf_counter() < deadline:
            if self.server.started:
                return
            time.sleep(0.05)
        raise RuntimeError("proxy failed to start within 10s")

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10.0)


@dataclass
class _CodexProxyStack:
    proxy_port: int
    upstream: _MockOpenAIServer
    proxy: _ProxyThread

    @property
    def base_url(self) -> str:
        return proxy_base_url(self.proxy_port)

    @property
    def stats_url(self) -> str:
        return f"http://127.0.0.1:{self.proxy_port}/stats"


@pytest.fixture(scope="module")
def codex_proxy_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_CodexProxyStack]:
    temp_home = tmp_path_factory.mktemp("codex-proxy-home")
    previous_env = {name: os.environ.get(name) for name in ("HEADROOM_REQUIRE_RUST_CORE", "HOME")}
    os.environ["HEADROOM_REQUIRE_RUST_CORE"] = "false"
    os.environ["HOME"] = str(temp_home)

    upstream_port = _free_port()
    upstream = _MockOpenAIServer(("127.0.0.1", upstream_port))
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()

    proxy_port = _free_port()
    proxy = _ProxyThread(proxy_port, upstream_port)
    proxy.start()

    try:
        _wait_for_stats(f"http://127.0.0.1:{proxy_port}/stats")
        yield _CodexProxyStack(proxy_port=proxy_port, upstream=upstream, proxy=proxy)
    finally:
        proxy.stop()
        upstream.shutdown()
        upstream_thread.join(timeout=5.0)
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _wait_for_stats(url: str) -> None:
    deadline = time.time() + 10.0
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=2.0)
            if response.status_code == 200:
                return
        except Exception:  # pragma: no cover - best effort poll
            pass
        time.sleep(0.1)
    raise RuntimeError(f"Timed out waiting for {url}")


def _model_count(stats_url: str, model: str) -> int:
    response = httpx.get(stats_url, timeout=5.0)
    response.raise_for_status()
    payload = response.json()
    return int(payload["requests"]["by_model"].get(model, 0))


def _send_probe(base_url: str, *, model: str, content: str) -> dict[str, object]:
    response = httpx.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": content}],
        },
        timeout=10.0,
    )
    response.raise_for_status()
    return response.json()


def _assert_delivery(
    stack: _CodexProxyStack,
    *,
    base_url: str,
    model: str,
    content: str,
) -> None:
    before = _model_count(stack.stats_url, model)
    payload = _send_probe(base_url, model=model, content=content)

    deadline = time.time() + 10.0
    while time.time() < deadline:
        if _model_count(stack.stats_url, model) >= before + 1:
            break
        time.sleep(0.1)
    else:
        raise AssertionError(f"Headroom never recorded model {model!r} in /stats")

    assert payload["choices"][0]["message"]["content"] == "mock completion from upstream"
    assert any(
        item["path"] == "/v1/chat/completions"
        and isinstance(item.get("body"), dict)
        and item["body"].get("model") == model
        and item["body"].get("messages") == [{"role": "user", "content": content}]
        for item in stack.upstream.requests
    )


def _codex_base_url_from_config(path: Path) -> str:
    parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    assert parsed["model_provider"] == "headroom"
    return str(parsed["model_providers"]["headroom"]["base_url"])


def _manifest(tmp_path: Path, *, port: int) -> DeploymentManifest:
    return DeploymentManifest(
        profile="default",
        preset="persistent-service",
        runtime_kind="python",
        supervisor_kind="service",
        scope=ConfigScope.PROVIDER.value,
        provider_mode="manual",
        targets=["codex"],
        port=port,
        host="127.0.0.1",
        backend="openai",
        memory_db_path=str(tmp_path / "memory.db"),
        tool_envs={"codex": {"OPENAI_BASE_URL": proxy_base_url(port)}},
    )


def test_codex_proxy_base_url_and_launch_env() -> None:
    env, lines = build_launch_env(9999, {"OPENAI_API_KEY": "sk-test"})

    assert proxy_base_url(9999) == "http://127.0.0.1:9999/v1"
    assert env["OPENAI_API_KEY"] == "sk-test"
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:9999/v1"
    assert lines == ["OPENAI_BASE_URL=http://127.0.0.1:9999/v1"]


def test_codex_launch_env_routes_messages_through_headroom(
    codex_proxy_stack: _CodexProxyStack,
) -> None:
    env, _ = build_launch_env(codex_proxy_stack.proxy_port, {"OPENAI_API_KEY": "sk-test"})

    _assert_delivery(
        codex_proxy_stack,
        base_url=str(env["OPENAI_BASE_URL"]),
        model="wrap-launch-delivery-probe",
        content="Verify temporary launch env traffic reaches Headroom.",
    )


def test_codex_install_env_routes_messages_through_headroom(
    codex_proxy_stack: _CodexProxyStack,
) -> None:
    env = build_install_env(port=codex_proxy_stack.proxy_port, backend="ignored")

    assert env == {"OPENAI_BASE_URL": codex_proxy_stack.base_url}
    _assert_delivery(
        codex_proxy_stack,
        base_url=env["OPENAI_BASE_URL"],
        model="persistent-install-delivery-probe",
        content="Verify persistent install env traffic reaches Headroom.",
    )


def test_init_codex_config_routes_messages_through_headroom(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    codex_proxy_stack: _CodexProxyStack,
) -> None:
    monkeypatch.chdir(tmp_path)

    init_cli._init_codex(
        global_scope=False,
        profile="init-local-demo",
        port=codex_proxy_stack.proxy_port,
    )

    config_path = tmp_path / ".codex" / "config.toml"
    content = config_path.read_text(encoding="utf-8")
    assert 'env_key = "OPENAI_API_KEY"' not in content
    # Bug 3 (#406): requires_openai_auth must be absent from headroom provider blocks.
    assert "requires_openai_auth" not in content, (
        f"requires_openai_auth must not appear in init-generated Codex config:\n{content}"
    )

    _assert_delivery(
        codex_proxy_stack,
        base_url=_codex_base_url_from_config(config_path),
        model="init-config-delivery-probe",
        content="Verify init-generated Codex config sends traffic to Headroom.",
    )


def test_provider_scope_codex_config_routes_messages_through_headroom(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    codex_proxy_stack: _CodexProxyStack,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('model = "gpt-4o"\n', encoding="utf-8")
    monkeypatch.setattr("headroom.providers.codex.install.codex_config_path", lambda: config_path)

    mutation = apply_provider_scope(_manifest(tmp_path, port=codex_proxy_stack.proxy_port))

    assert mutation is not None
    content = config_path.read_text(encoding="utf-8")
    assert 'env_key = "OPENAI_API_KEY"' not in content
    assert 'model_provider = "headroom"' in content

    _assert_delivery(
        codex_proxy_stack,
        base_url=_codex_base_url_from_config(config_path),
        model="provider-scope-delivery-probe",
        content="Verify persistent provider config sends traffic to Headroom.",
    )
