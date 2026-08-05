"""Regression: ``ccr_workspace_key`` UnboundLocalError when CCR inject is off.

``handle_anthropic_messages`` previously assigned ``ccr_workspace_key`` /
``ccr_workspace_label`` only inside the ``if (ccr_inject_tool or
ccr_inject_system_instructions) and not _bypass:`` block, but referenced
``ccr_workspace_key`` unconditionally later (the proactive-expansion gate). When
the proxy is started with ``--no-ccr-inject-tool`` and
``ccr_inject_system_instructions`` left at its ``False`` default — a real,
user-supported configuration — the assignment block was skipped and the later
reference raised ``UnboundLocalError``. FastAPI translated that into HTTP 500 on
every ``/v1/messages`` request.

Reproducer config matches the deployment that surfaced the bug:

  * ``ccr_inject_tool=False``                  (user passed ``--no-ccr-inject-tool``)
  * ``ccr_inject_system_instructions=False``  (default)
  * ``ccr_context_tracking=True``             (default — installs the tracker)
  * ``ccr_proactive_expansion=True``          (default — reaches the gate)
"""

from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app


def _client() -> TestClient:
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        # The two flags that gate the assignment block:
        ccr_inject_tool=False,
        ccr_inject_system_instructions=False,
        # Tracker + proactive expansion enabled (defaults) reach the unbound use:
        ccr_context_tracking=True,
        ccr_proactive_expansion=True,
        image_optimize=False,
    )
    return TestClient(create_app(config))


def test_proactive_expansion_does_not_raise_when_ccr_inject_disabled() -> None:
    with _client() as client:
        proxy = client.app.state.proxy
        # Sanity: the tracker is wired up (necessary for the bug to trigger).
        assert proxy.ccr_context_tracker is not None

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            return httpx.Response(
                200,
                json={
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        assert response.status_code == 200, response.text
