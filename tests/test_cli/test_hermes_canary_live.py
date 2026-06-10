"""Optional live test for the persistent Headroom→Hermes canary on spot-tech-ci.

Requires a running user unit (``headroom-hermes.service``) or any Headroom proxy
already bound to ``HEADROOM_CANARY_PORT`` (default ``18787`` on spot-tech-ci):

    HEADROOM_LIVE_HERMES=1 HEADROOM_LIVE_CANARY=1 \\
        pytest tests/test_cli/test_hermes_canary_live.py -v
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import pytest

from tests.test_cli.hermes_support import (
    assert_compression_delta,
    hermes_health_url,
    hermes_reachable,
)
from tests.test_cli.hermes_workloads import agent_tool_messages

_LIVE = os.environ.get("HEADROOM_LIVE_HERMES") == "1"
_CANARY = os.environ.get("HEADROOM_LIVE_CANARY") == "1"
HERMES_BASE = os.environ.get("HEADROOM_HERMES_BASE_URL", "http://127.0.0.1:38765/v1").rstrip("/")
CANARY_PORT = int(os.environ.get("HEADROOM_CANARY_PORT", "18787"))
HERMES_MODEL = os.environ.get("HEADROOM_HERMES_MODEL", "grok-4.3")

pytestmark = pytest.mark.skipif(
    not (_LIVE and _CANARY),
    reason="set HEADROOM_LIVE_HERMES=1 and HEADROOM_LIVE_CANARY=1",
)


def _chat(client: httpx.Client, base_url: str, messages: list[dict]) -> dict[str, Any]:
    payload = {
        "model": HERMES_MODEL,
        "messages": messages,
        "max_tokens": 24,
        "temperature": 0,
        "stream": False,
    }
    resp = client.post(f"{base_url}/chat/completions", json=payload, timeout=180.0)
    resp.raise_for_status()
    return resp.json()


def _canary_ready(client: httpx.Client) -> bool:
    for path in ("/readyz", "/health", "/livez"):
        try:
            resp = client.get(f"http://127.0.0.1:{CANARY_PORT}{path}", timeout=5.0)
            if resp.status_code == 200:
                return True
        except Exception:
            continue
    return False


@pytest.fixture(scope="module")
def hermes_available() -> None:
    if not hermes_reachable(HERMES_BASE):
        pytest.skip(f"Hermes not reachable at {hermes_health_url(HERMES_BASE)}")


@pytest.fixture(scope="module")
def canary_available(hermes_available: None) -> None:  # noqa: ARG001
    with httpx.Client() as client:
        if not _canary_ready(client):
            pytest.skip(f"canary not ready on port {CANARY_PORT}")


def test_live_canary_compresses_agent_tool_output(
    hermes_available: None,  # noqa: ARG001
    canary_available: None,  # noqa: ARG001
) -> None:
    messages = agent_tool_messages()
    with httpx.Client() as client:
        plain = _chat(client, HERMES_BASE, messages)
        wrapped = _chat(client, f"http://127.0.0.1:{CANARY_PORT}/v1", messages)

        plain_tokens = int((plain.get("usage") or {}).get("prompt_tokens") or 0)
        wrapped_tokens = int((wrapped.get("usage") or {}).get("prompt_tokens") or 0)
        content = wrapped["choices"][0]["message"]["content"]

        stats_resp = client.get(f"http://127.0.0.1:{CANARY_PORT}/stats", timeout=30.0)
        stats_resp.raise_for_status()
        stats = stats_resp.json()
        saved = int((stats.get("tokens") or {}).get("saved") or 0)
        strategy_saved = int((stats.get("tokens_saved_by_strategy") or {}).get("smart_crusher") or 0)

    assert_compression_delta(
        plain_prompt_tokens=plain_tokens,
        wrapped_prompt_tokens=wrapped_tokens,
        tokens_saved=saved,
        smart_crusher_saved=strategy_saved,
        content=content,
    )