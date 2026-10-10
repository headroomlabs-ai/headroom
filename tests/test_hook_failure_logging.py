"""Request-path failure logging: named at warning/error, no payload at any level."""

from __future__ import annotations

import contextlib
import logging
import socket
import threading

import httpx
import pytest

from headroom.hooks import CompressionHooks
from headroom.log_safety import WarnOnce
from headroom.proxy.handlers import _failure_logging
from headroom.proxy.handlers._failure_logging import log_hook_failure, log_request_failure

CONTENT = "CANARY-USER-CONTENT"
CREDENTIAL = "sk-CANARY-CREDENTIAL"


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def at(self, level: int) -> list[logging.LogRecord]:
        return [r for r in self.records if r.levelno == level]

    def hook_warnings(self) -> list[str]:
        return [r.getMessage() for r in self.at(logging.WARNING) if "] hook " in r.getMessage()]

    def assert_no_canaries(self) -> None:
        """Every record, rendered the way a log file renders it (traceback included)."""
        formatter = logging.Formatter()
        assert self.records
        for record in self.records:
            text = formatter.format(record)
            assert CONTENT not in text
            assert CREDENTIAL not in text


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HEADROOM_DEBUG_DUMP", raising=False)
    monkeypatch.setattr(_failure_logging, "_HOOK_WARNINGS", WarnOnce(256, "failing hooks"))
    log = logging.getLogger("headroom.proxy")
    handler = _Capture()
    old_level = log.level
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        log.removeHandler(handler)
        log.setLevel(old_level)


class _HooksA(CompressionHooks):
    pass


class _HooksB(CompressionHooks):
    pass


def _raised(exc: BaseException) -> BaseException:
    """``exc`` with a traceback, as a caught exception has."""
    try:
        raise exc
    except BaseException as caught:  # noqa: BLE001 - test helper
        return caught


def test_hook_failure_warns_once_and_no_record_carries_the_payload(capture) -> None:
    for request_id in ("req-1", "req-2"):
        err = _raised(ValueError(f"bad message {CONTENT} key={CREDENTIAL}"))
        log_hook_failure(request_id, "compute_biases", _HooksA(), err)

    (warning,) = capture.hook_warnings()
    assert warning.startswith("[req-1] hook ")
    assert "_HooksA compute_biases failed; continuing without it: ValueError" in warning
    (repeat,) = capture.at(logging.DEBUG)
    assert repeat.getMessage().startswith("[req-2] hook ")
    capture.assert_no_canaries()


def test_hook_identity_stage_and_error_type_each_get_their_own_warning(capture) -> None:
    log_hook_failure("r1", "pre_compress", _HooksA(), ValueError("x"))
    log_hook_failure("r2", "pre_compress", _HooksB(), ValueError("x"))  # other hook
    log_hook_failure("r3", "post_compress", _HooksA(), ValueError("x"))  # other stage
    log_hook_failure("r4", "pre_compress", _HooksA(), KeyError("x"))  # other type
    log_hook_failure("r5", "pre_compress", _HooksA(), ValueError("y"))  # repeat

    assert [w.split()[0] for w in capture.hook_warnings()] == ["[r1]", "[r2]", "[r3]", "[r4]"]


def test_a_suppressed_warning_is_still_emitted_later(capture) -> None:
    log = logging.getLogger("headroom.proxy")
    log.setLevel(logging.ERROR)
    log_hook_failure("r1", "post_compress", _HooksA(), ValueError("x"))
    log.setLevel(logging.INFO)
    log_hook_failure("r2", "post_compress", _HooksA(), ValueError("x"))

    assert [w.split()[0] for w in capture.hook_warnings()] == ["[r2]"]


def test_request_failure_names_the_failure_without_payload(capture) -> None:
    log_request_failure(
        "r1", "Request", _raised(ValueError(CONTENT)), provider="anthropic", model="m"
    )

    (error,) = (r.getMessage() for r in capture.at(logging.ERROR))
    assert error.startswith("[r1] Request failed: provider='anthropic' model='m' ValueError")
    capture.assert_no_canaries()


@contextlib.contextmanager
def _proxy_rejecting_connect(reason: str):
    """A real local HTTP proxy that answers every CONNECT with ``502 <reason>``."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    server.settimeout(5)

    def serve() -> None:
        try:
            conn, _ = server.accept()
        except OSError:  # closed before a client came
            return
        with conn:
            conn.recv(65536)
            conn.sendall(f"HTTP/1.1 502 {reason}\r\nContent-Length: 0\r\n\r\n".encode())

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.getsockname()[1]}"
    finally:
        server.close()
        thread.join(timeout=5)


def _proxy_error(reason: str) -> httpx.ProxyError:
    with _proxy_rejecting_connect(reason) as proxy_url:
        with httpx.Client(proxy=proxy_url, timeout=5) as client:
            with pytest.raises(httpx.ProxyError) as caught:
                client.get("https://upstream.example/v1/messages")
    assert reason in str(caught.value)  # the peer's text really is in the exception
    return caught.value


def test_a_proxy_reason_phrase_stays_out_of_the_request_failure_line(capture) -> None:
    log_request_failure("r1", "Request", _proxy_error(CONTENT), provider="anthropic", model="m")

    (error,) = (r.getMessage() for r in capture.at(logging.ERROR))
    assert error.startswith("[r1] Request failed: provider='anthropic' model='m' ProxyError")
    capture.assert_no_canaries()


def test_the_content_opt_in_restores_the_proxy_text(
    capture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HEADROOM_DEBUG_DUMP", "full")
    log_request_failure("r1", "Request", _proxy_error(CONTENT), model="m")

    (error,) = capture.at(logging.ERROR)
    assert CONTENT in logging.Formatter().format(error)


def _proxy_app(hooks: CompressionHooks, fake_retry, *, optimize: bool):  # noqa: ANN001, ANN202
    pytest.importorskip("fastapi")
    from headroom.proxy.server import ProxyConfig, create_app

    app = create_app(
        ProxyConfig(optimize=optimize, cache_enabled=False, rate_limit_enabled=False, hooks=hooks)
    )
    return app


def _chat_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "c",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        },
    )


def test_openai_chat_warns_for_each_failing_stage(capture) -> None:
    """Through the real handler: after a pre_compress failure, a later compute_biases
    failure of the same type still warns, under its own stage."""
    from fastapi.testclient import TestClient

    class Flaky(CompressionHooks):
        failing = "pre_compress"

        def pre_compress(self, messages, ctx):  # noqa: ANN001, ANN201
            if self.failing == "pre_compress":
                raise ValueError(CONTENT)
            return messages

        def compute_biases(self, messages, ctx):  # noqa: ANN001, ANN201
            raise ValueError(CONTENT)

    async def fake_retry(method, url, headers, body, *args, **kwargs):  # noqa: ANN001, ANN202
        return _chat_response()

    hooks = Flaky()
    with TestClient(_proxy_app(hooks, fake_retry, optimize=False)) as client:
        # Proxy startup may reset logger levels; re-assert capture afterwards.
        logging.getLogger("headroom.proxy").setLevel(logging.DEBUG)
        client.app.state.proxy._retry_request = fake_retry
        for failing in ("pre_compress", "compute_biases"):
            hooks.failing = failing
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": CONTENT}]},
                headers={"Authorization": f"Bearer {CREDENTIAL}"},
            )
            assert resp.status_code == 200, resp.text

    warnings = capture.hook_warnings()
    assert len(warnings) == 2
    assert "Flaky pre_compress failed" in warnings[0]
    assert "Flaky compute_biases failed" in warnings[1]
    capture.assert_no_canaries()


def test_anthropic_bias_failure_says_compression_is_skipped(capture) -> None:
    """Through the real handler: Anthropic aborts optimization when a bias hook fails,
    so the warning must say compression is skipped, not that it continues."""
    from fastapi.testclient import TestClient

    class BrokenBiases(CompressionHooks):
        def compute_biases(self, messages, ctx):  # noqa: ANN001, ANN201
            raise ValueError(CONTENT)

    async def fake_retry(method, url, headers, body, *args, **kwargs):  # noqa: ANN001, ANN202
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 1},
            },
        )

    with TestClient(_proxy_app(BrokenBiases(), fake_retry, optimize=True)) as client:
        logging.getLogger("headroom.proxy").setLevel(logging.DEBUG)
        client.app.state.proxy._retry_request = fake_retry
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": CONTENT}],
            },
            headers={"x-api-key": CREDENTIAL, "anthropic-version": "2023-06-01"},
        )

    assert resp.status_code == 200, resp.text
    (warning,) = capture.hook_warnings()
    assert "BrokenBiases compute_biases failed; skipping compression for this turn" in warning
    assert "continuing without it" not in warning
    capture.assert_no_canaries()
