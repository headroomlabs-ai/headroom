"""Diagnostics for Anthropic compression-stage observability (issue #296).

These tests verify two diagnostic improvements that let bug reports
distinguish a real pipeline failure from a thread-pool starvation timeout:

1. ``request_id`` is plumbed into ``pipeline.apply`` so its log lines
   ("Pipeline: freezing first ...", "Pipeline complete: ...") can be
   correlated with a specific request rather than guessed at from
   interleaved concurrent logs.
2. When ``compression_first_stage`` raises, the warning includes the
   exception type — ``str(asyncio.TimeoutError())`` is empty, which is
   why issue #296 shows ``Optimization failed:`` with nothing after the
   colon.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app


def _make_proxy_client() -> TestClient:
    config = ProxyConfig(
        optimize=True,
        mode="token",
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    app = create_app(config)
    return TestClient(app)


def _ok_response(msg_id: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 3,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    )


def test_request_id_plumbed_to_pipeline_apply() -> None:
    """The handler must pass request_id into pipeline.apply so the
    pipeline's log lines can be correlated with a specific request."""
    captured: dict[str, object] = {}
    with _make_proxy_client() as client:
        proxy = client.app.state.proxy

        def _fake_apply(**kwargs):
            captured["request_id"] = kwargs.get("request_id")
            return SimpleNamespace(
                messages=kwargs["messages"],
                transforms_applied=[],
                timing={},
                tokens_before=10,
                tokens_after=10,
                waste_signals=None,
            )

        proxy.anthropic_pipeline.apply = _fake_apply

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            return _ok_response("msg_diag_1")

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        assert response.status_code == 200
        assert isinstance(captured["request_id"], str)
        assert captured["request_id"]  # non-empty


def test_optimization_failure_logs_exception_type() -> None:
    """When pipeline.apply raises, the warning must include the
    exception type — issue #296 reported ``Optimization failed:`` with
    an empty message because asyncio.TimeoutError has no str repr.

    We patch the handler module's ``logger.warning`` directly rather than
    relying on logging propagation: the headroom logger sets
    ``propagate=False`` (see proxy/helpers.py) and per-test mutations of
    handler chains have proven brittle in CI.
    """
    from headroom.proxy.handlers import anthropic as anth_handler

    with _make_proxy_client() as client:
        proxy = client.app.state.proxy

        def _raise_timeout(**kwargs):
            raise asyncio.TimeoutError()

        proxy.anthropic_pipeline.apply = _raise_timeout

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            return _ok_response("msg_diag_2")

        proxy._retry_request = _fake_retry

        with patch.object(anth_handler.logger, "warning") as mock_warning:
            response = client.post(
                "/v1/messages",
                headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

        assert response.status_code == 200, response.text
        warning_msgs = [
            call.args[0]
            for call in mock_warning.call_args_list
            if call.args and "Optimization failed" in str(call.args[0])
        ]
        assert warning_msgs, (
            f"expected an 'Optimization failed' warning, got calls: {mock_warning.call_args_list!r}"
        )
        msg = warning_msgs[0]
        assert "TimeoutError" in msg, f"expected exception type in warning, got: {msg!r}"
