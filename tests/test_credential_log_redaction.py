"""Credentials stay out of the proxy log and the upstream-error dumps."""

from __future__ import annotations

import json
import logging
import os
import stat
from unittest.mock import AsyncMock, patch

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.models import ProxyConfig  # noqa: E402
from headroom.proxy.server import create_app  # noqa: E402
from headroom.proxy.wire_debug_redaction_policy import should_redact_key  # noqa: E402

FAKE_KEY = "AIzaSyFAKEKEY0123456789"


def _config() -> ProxyConfig:
    return ProxyConfig(
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


def test_inbound_request_log_drops_query_values_and_key_headers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO), TestClient(create_app(_config())) as client:
        client.get(
            f"/stats?key={FAKE_KEY}&alt=sse",
            headers={"Ocp-Apim-Subscription-Key": "FAKE-OCP", "apikey": "FAKE-APIKEY"},
        )
    lines = [r.getMessage() for r in caplog.records if "proxy_inbound_request" in r.getMessage()]
    assert lines
    for line in lines:
        assert FAKE_KEY not in line
        assert "FAKE-OCP" not in line
        assert "FAKE-APIKEY" not in line
    assert any("query=key=<redacted>&alt=<redacted>" in line for line in lines)


def test_streaming_upstream_error_log_drops_query(caplog: pytest.LogCaptureFixture) -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad"}})

    body = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}
    with caplog.at_level(logging.WARNING), TestClient(create_app(_config())) as client:
        client.app.state.proxy.http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(upstream)
        )
        client.post(
            f"/v1beta/models/gemini-2.5-flash:streamGenerateContent?key={FAKE_KEY}&alt=sse",
            json=body,
        )
    messages = [r.getMessage() for r in caplog.records]
    assert any("Upstream provider returned an error" in m for m in messages), messages
    assert not any(FAKE_KEY in m for m in messages)


@pytest.mark.parametrize(
    "name",
    ["Ocp-Apim-Subscription-Key", "X-Functions-Key", "apikey", "x-goog-api-key", "x-api-key"],
)
def test_wire_debug_redacts_key_headers(name: str) -> None:
    assert should_redact_key(name)


@pytest.mark.parametrize("name", ["max_tokens", "input_tokens", "keyboard", "monkey", "key_id"])
def test_wire_debug_key_rule_does_not_overreach(name: str) -> None:
    assert not should_redact_key(name)


@pytest.mark.parametrize(
    ("path", "headers"),
    [
        (
            "/v1/messages",
            {"x-api-key": "sk-ant-FAKESECRET-0123456789", "anthropic-version": "2023-06-01"},
        ),
        ("/v1/chat/completions", {"authorization": "Bearer sk-FAKESECRET-0123456789"}),
    ],
)
def test_upstream_error_dump_masks_credentials_and_is_owner_only(
    tmp_path, monkeypatch: pytest.MonkeyPatch, path: str, headers: dict[str, str]
) -> None:
    from headroom import paths

    dump_dir = tmp_path / "debug_400"
    monkeypatch.setattr(paths, "debug_400_dir", lambda: dump_dir)
    monkeypatch.setenv("HEADROOM_DEBUG_DUMP", "1")
    error = httpx.Response(
        400,
        content=json.dumps({"error": {"type": "invalid_request_error", "message": "bad"}}).encode(),
        headers={"content-type": "application/json"},
    )
    body = {"model": "m", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}
    with patch("headroom.proxy.server.AnyLLMBackend"):
        with TestClient(create_app(_config())) as client:
            client.app.state.proxy._retry_request = AsyncMock(return_value=error)
            client.post(
                path, headers={**headers, "Ocp-Apim-Subscription-Key": "FAKE-OCP"}, json=body
            )

    (dump,) = dump_dir.glob("*.json")
    text = dump.read_text(encoding="utf-8")
    assert "FAKESECRET" not in text
    assert "sk-ant-FAKE" not in text and "Bearer sk-F" not in text
    assert "FAKE-OCP" not in text
    if os.name != "nt":  # POSIX permission bits
        assert stat.S_IMODE(dump.stat().st_mode) == 0o600


def test_query_names_cannot_split_the_log_line() -> None:
    from headroom.proxy.query_log_policy import redact_query_string_for_log

    rendered = redact_query_string_for_log("a%0D%0Aforged%20line=1&key=secret")

    assert "\n" not in rendered and "\r" not in rendered
    assert "secret" not in rendered
    assert rendered == "a%0D%0Aforged%20line=<redacted>&key=<redacted>"


def test_uvicorn_access_log_drops_query_values() -> None:
    """Real uvicorn at info level, with no filter but headroom's own on the logger."""
    import asyncio
    import socket

    import uvicorn

    app = create_app(_config())
    access = logging.getLogger("uvicorn.access")
    others = [f for f in access.filters if not type(f).__module__.startswith("headroom.")]
    for f in others:  # e.g. LiteLLM's, which is only present when litellm is imported
        access.removeFilter(f)
    lines: list[str] = []

    class _Lines(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(record.getMessage())

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info", lifespan="off")
    )

    async def run() -> None:
        task = asyncio.create_task(server.serve())

        async def started() -> None:
            while not server.started:
                if task.done():  # e.g. the port could not be bound
                    pytest.fail(f"uvicorn stopped before starting: {task.exception()!r}")
                await asyncio.sleep(0.02)

        await asyncio.wait_for(started(), timeout=10)
        handler = _Lines()
        access.addHandler(handler)
        try:
            async with httpx.AsyncClient(trust_env=False) as client:
                await client.get(f"http://127.0.0.1:{port}/livez?key={FAKE_KEY}&alt=sse")
        finally:
            access.removeHandler(handler)
            server.should_exit = True
            await task

    try:
        asyncio.run(run())
    finally:
        for f in others:
            access.addFilter(f)

    assert any("/livez?" in line for line in lines)
    assert not any(FAKE_KEY in line for line in lines)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_owner_only_write_tightens_an_existing_directory_and_file(tmp_path) -> None:
    from headroom.proxy.handlers._debug_dump import write_owner_only_json

    dump_dir = tmp_path / "debug_400"
    dump_dir.mkdir(mode=0o755)
    os.chmod(dump_dir, 0o755)
    dump = dump_dir / "dump.json"
    dump.write_text("{}", encoding="utf-8")
    os.chmod(dump, 0o644)

    write_owner_only_json(dump, {"headers": {"authorization": "<redacted>"}})

    assert stat.S_IMODE(dump_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(dump.stat().st_mode) == 0o600
    assert json.loads(dump.read_text(encoding="utf-8"))["headers"]["authorization"] == "<redacted>"


@pytest.mark.skipif(not os.path.isdir("/dev/fd"), reason="needs /dev/fd to count descriptors")
def test_owner_only_write_closes_and_keeps_the_file_when_chmod_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from headroom.proxy.handlers import _debug_dump

    dump = tmp_path / "dump.json"
    dump.write_text('{"previous": true}', encoding="utf-8")

    def refuse(fd: int, mode: int) -> None:  # e.g. a file owned by another user
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(_debug_dump.os, "fchmod", refuse, raising=False)
    open_before = len(os.listdir("/dev/fd"))

    with pytest.raises(PermissionError):
        _debug_dump.write_owner_only_json(dump, {"x": 1})

    assert len(os.listdir("/dev/fd")) == open_before
    assert dump.read_text(encoding="utf-8") == '{"previous": true}'
