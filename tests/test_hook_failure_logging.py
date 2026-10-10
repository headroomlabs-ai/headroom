"""Request-path failure logging: visible at warning/error, content only at debug."""

from __future__ import annotations

import logging

import httpx
import pytest

from headroom.hooks import CompressionHooks
from headroom.proxy.handlers import _failure_logging
from headroom.proxy.handlers._failure_logging import log_hook_failure, log_request_failure

SECRET = "SECRET-USER-CONTENT"


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def at(self, level: int) -> list[logging.LogRecord]:
        return [r for r in self.records if r.levelno == level]


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_failure_logging, "_WARNED", set())
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


def test_hook_failure_warns_once_without_the_error_text(capture) -> None:
    for request_id in ("req-1", "req-2"):
        log_hook_failure(request_id, "compute_biases", _HooksA(), ValueError(SECRET))

    (warning,) = capture.at(logging.WARNING)
    assert "[req-1] hook" in warning.getMessage()
    assert "_HooksA compute_biases failed" in warning.getMessage()
    assert "ValueError" in warning.getMessage()
    assert SECRET not in warning.getMessage()
    assert warning.exc_info is None
    # The detail, text and traceback included, is at debug for both failures.
    debug = capture.at(logging.DEBUG)
    assert len(debug) == 2
    assert all(r.exc_info and SECRET in str(r.exc_info[1]) for r in debug)


def test_hook_identity_stage_and_error_type_each_get_their_own_warning(capture) -> None:
    log_hook_failure("r1", "pre_compress", _HooksA(), ValueError("x"))
    log_hook_failure("r2", "pre_compress", _HooksB(), ValueError("x"))  # other hook
    log_hook_failure("r3", "post_compress", _HooksA(), ValueError("x"))  # other stage
    log_hook_failure("r4", "pre_compress", _HooksA(), KeyError("x"))  # other type
    log_hook_failure("r5", "pre_compress", _HooksA(), ValueError("y"))  # repeat

    warned = [r.getMessage().split()[0] for r in capture.at(logging.WARNING)]
    assert warned == ["[r1]", "[r2]", "[r3]", "[r4]"]


def test_a_suppressed_warning_does_not_use_up_the_once_slot(capture) -> None:
    log = logging.getLogger("headroom.proxy")
    log.setLevel(logging.ERROR)
    log_hook_failure("r1", "post_compress", _HooksA(), ValueError("x"))
    log.setLevel(logging.DEBUG)
    log_hook_failure("r2", "post_compress", _HooksA(), ValueError("x"))

    assert [r.getMessage().split()[0] for r in capture.at(logging.WARNING)] == ["[r2]"]


def test_the_once_set_is_bounded(capture, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_failure_logging, "_MAX_WARNED", 2)
    for stage in ("a", "b", "c"):
        log_hook_failure("r", stage, _HooksA(), ValueError("x"))
    assert len(_failure_logging._WARNED) <= 2


def test_request_failure_keeps_transport_text_and_hides_other_text(capture) -> None:
    log_request_failure("r1", "Request", ValueError(SECRET), provider="anthropic", model="m")
    log_request_failure("r2", "Request", httpx.ConnectError("Connection refused"), model="m")

    first, second = capture.at(logging.ERROR)
    assert first.getMessage() == "[r1] Request failed: provider=anthropic model=m ValueError"
    assert second.getMessage() == "[r2] Request failed: model=m ConnectError: Connection refused"
    assert first.exc_info is None and second.exc_info is None
    assert any(r.exc_info and SECRET in str(r.exc_info[1]) for r in capture.at(logging.DEBUG))


def test_openai_chat_names_the_failing_hook_stage(capture) -> None:
    """Through the real handler: the stage that raised is the one logged."""
    fastapi = pytest.importorskip("fastapi")  # noqa: F841
    from fastapi.testclient import TestClient

    from headroom.proxy.server import ProxyConfig, create_app

    class BrokenBiases(CompressionHooks):
        def compute_biases(self, messages, ctx):  # noqa: ANN001, ANN201
            raise ValueError(SECRET)

    async def fake_retry(method, url, headers, body, *args, **kwargs):  # noqa: ANN001, ANN202
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

    config = ProxyConfig(
        optimize=False, cache_enabled=False, rate_limit_enabled=False, hooks=BrokenBiases()
    )
    app = create_app(config)
    with TestClient(app) as client:
        # Proxy startup may reset logger levels; re-assert capture afterwards.
        logging.getLogger("headroom.proxy").setLevel(logging.DEBUG)
        client.app.state.proxy._retry_request = fake_retry
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": SECRET}]},
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 200, resp.text
    hook_warnings = [r for r in capture.at(logging.WARNING) if "hook" in r.getMessage()]
    assert len(hook_warnings) == 1
    assert "BrokenBiases compute_biases failed" in hook_warnings[0].getMessage()
    assert "pre_compress" not in hook_warnings[0].getMessage()
    assert all(
        SECRET not in r.getMessage() for r in capture.records if r.levelno >= logging.WARNING
    )


def test_anthropic_bias_failure_says_compression_is_skipped(capture) -> None:
    """Through the real handler: Anthropic aborts optimization when a bias hook fails,
    so the warning must say compression is skipped, not that it continues."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from headroom.proxy.server import ProxyConfig, create_app

    class BrokenBiases(CompressionHooks):
        def compute_biases(self, messages, ctx):  # noqa: ANN001, ANN201
            raise ValueError(SECRET)

    sent: list[dict] = []

    async def fake_retry(method, url, headers, body, *args, **kwargs):  # noqa: ANN001, ANN202
        sent.append(body)
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

    app = create_app(
        ProxyConfig(
            optimize=True, cache_enabled=False, rate_limit_enabled=False, hooks=BrokenBiases()
        )
    )
    with TestClient(app) as client:
        logging.getLogger("headroom.proxy").setLevel(logging.DEBUG)
        client.app.state.proxy._retry_request = fake_retry
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": SECRET}],
            },
            headers={"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"},
        )

    assert resp.status_code == 200, resp.text
    (warning,) = [r for r in capture.at(logging.WARNING) if "hook" in r.getMessage()]
    message = warning.getMessage()
    assert "BrokenBiases compute_biases failed; skipping compression for this turn" in message
    assert "continuing without it" not in message
    assert SECRET not in message
