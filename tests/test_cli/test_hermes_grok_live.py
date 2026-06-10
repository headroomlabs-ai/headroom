"""Optional live Hermes + Headroom integration tests (OpenAI chat path).

Run on a host where Hermes llm-proxy is up (default ``:38765``):

    HEADROOM_LIVE_HERMES=1 pytest tests/test_cli/test_hermes_grok_live.py -v

spot-tech-ci minimal venv::

    uv venv .venv-live && source .venv-live/bin/activate
    uv pip install httpx pytest
    HEADROOM_LIVE_HERMES=1 pytest tests/test_cli/test_hermes_grok_live.py -v
    python scripts/bench_hermes_headroom.py
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.test_cli.hermes_support import (
    assert_compression_delta,
    hermes_health_url,
    hermes_reachable,
    pick_free_port,
    start_headroom_proxy,
    stop_process,
    wait_proxy_ready,
)
from tests.test_cli.hermes_workloads import agent_tool_messages

_LIVE = os.environ.get("HEADROOM_LIVE_HERMES") == "1"
HERMES_BASE = os.environ.get("HEADROOM_HERMES_BASE_URL", "http://127.0.0.1:38765/v1").rstrip("/")
HERMES_MODEL = os.environ.get("HEADROOM_HERMES_MODEL", "grok-4.3")
REPLY_TOKEN = os.environ.get("HEADROOM_HERMES_REPLY_TOKEN", "HERMES_OK")

pytestmark = pytest.mark.skipif(not _LIVE, reason="set HEADROOM_LIVE_HERMES=1 to run live Hermes tests")


def _chat(
    client: httpx.Client,
    base_url: str,
    messages: list[dict],
    *,
    max_tokens: int = 24,
) -> dict[str, Any]:
    payload = {
        "model": HERMES_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    resp = client.post(f"{base_url}/chat/completions", json=payload, timeout=180.0)
    resp.raise_for_status()
    body = resp.json()
    body["_backend"] = resp.headers.get("x-llm-backend")
    return body


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def hermes_available() -> None:
    if not hermes_reachable(HERMES_BASE):
        pytest.skip(f"Hermes not reachable at {hermes_health_url(HERMES_BASE)}")


@pytest.fixture(scope="module")
def hermes_client(hermes_available: None) -> Iterator[httpx.Client]:  # noqa: ARG001
    with httpx.Client() as client:
        yield client


@pytest.fixture
def headroom_proxy(repo_root: Path, hermes_available: None) -> Iterator[int]:  # noqa: ARG001
    port = pick_free_port()
    proc = start_headroom_proxy(port=port, hermes_base=HERMES_BASE, repo_root=repo_root)
    try:
        wait_proxy_ready(port)
        yield port
    finally:
        stop_process(proc)


def test_live_hermes_health(hermes_client: httpx.Client, hermes_available: None) -> None:  # noqa: ARG001
    resp = hermes_client.get(hermes_health_url(HERMES_BASE))
    assert resp.status_code == 200
    assert resp.text.strip().lower() in {"ok", '"ok"'}


def test_live_plain_hermes_chat_completion(
    hermes_client: httpx.Client, hermes_available: None
) -> None:  # noqa: ARG001
    body = _chat(
        hermes_client,
        HERMES_BASE,
        [{"role": "user", "content": f"Reply with exactly one word: {REPLY_TOKEN}"}],
    )
    content = body["choices"][0]["message"]["content"]
    usage = body.get("usage") or {}
    assert REPLY_TOKEN in content
    assert int(usage.get("prompt_tokens") or 0) > 0


def test_live_headroom_proxy_routes_to_hermes(
    hermes_client: httpx.Client,
    headroom_proxy: int,
    hermes_available: None,
) -> None:  # noqa: ARG001
    body = _chat(
        hermes_client,
        f"http://127.0.0.1:{headroom_proxy}/v1",
        [{"role": "user", "content": f"Reply with exactly one word: {REPLY_TOKEN}"}],
    )
    content = body["choices"][0]["message"]["content"]
    assert REPLY_TOKEN in content

    stats_resp = hermes_client.get(f"http://127.0.0.1:{headroom_proxy}/stats", timeout=30.0)
    stats_resp.raise_for_status()
    stats = stats_resp.json()
    requests_total = int((stats.get("requests") or {}).get("total") or 0)
    assert requests_total >= 1


def test_live_headroom_openai_upstream_points_at_hermes(
    hermes_client: httpx.Client,
    headroom_proxy: int,
    hermes_available: None,
) -> None:  # noqa: ARG001
    health_resp = hermes_client.get(f"http://127.0.0.1:{headroom_proxy}/health", timeout=30.0)
    health_resp.raise_for_status()
    health = health_resp.json()
    config = health.get("config") if isinstance(health.get("config"), dict) else health
    assert config.get("openai_api_url") == HERMES_BASE


def test_live_headroom_compresses_agent_tool_output(
    hermes_client: httpx.Client,
    headroom_proxy: int,
    hermes_available: None,
) -> None:  # noqa: ARG001
    messages = agent_tool_messages()
    plain = _chat(hermes_client, HERMES_BASE, messages)
    wrapped = _chat(hermes_client, f"http://127.0.0.1:{headroom_proxy}/v1", messages)

    plain_tokens = int((plain.get("usage") or {}).get("prompt_tokens") or 0)
    wrapped_tokens = int((wrapped.get("usage") or {}).get("prompt_tokens") or 0)
    content = wrapped["choices"][0]["message"]["content"]

    stats_resp = hermes_client.get(f"http://127.0.0.1:{headroom_proxy}/stats", timeout=30.0)
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