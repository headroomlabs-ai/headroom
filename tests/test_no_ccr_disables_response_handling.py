"""`--no-ccr` has to disable server-side CCR handling too (#3082).

The flag advertises "Disable CCR entirely", and names the case it exists for:
streaming / non-MCP clients that cannot resolve an injected tool. It mapped onto
only two of the three CCR subsystems, though — markers and tool injection —
leaving ``ccr_handle_responses`` on, which has no flag and no env var of its own.

That mattered because the buffered ``stream: false`` path keys off
``headroom_retrieve`` being present in the *request's* tools, and the client can
put it there itself: the bundled OpenCode plugin registers the tool
unconditionally. So `--no-ccr` left the buffered path fully armed for exactly
the clients it was recommended to, and a turn whose history still held a
redeemable marker kept being flipped to buffered.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

click = pytest.importorskip("click")
pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from click.testing import CliRunner  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from headroom.cache.backends import InMemoryBackend  # noqa: E402
from headroom.cache.compression_store import (  # noqa: E402
    get_compression_store,
    reset_compression_store,
)
from headroom.ccr.tool_injection import create_ccr_tool_definition  # noqa: E402
from headroom.cli.main import main  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


def _config_for(args: list[str], env: dict[str, str] | None = None) -> ProxyConfig:
    """Run `headroom proxy ...` far enough to capture the ProxyConfig it builds."""
    captured: dict[str, ProxyConfig] = {}

    def mock_run_server(config, **kwargs):  # noqa: ANN001, ANN003
        captured["config"] = config

    with patch("headroom.proxy.server.run_server", mock_run_server):
        result = CliRunner().invoke(main, args, env=env or {}, catch_exceptions=False)

    assert result.exit_code == 0, result.output
    return captured["config"]


# --------------------------------------------------------------------------- #
# The flag -> config mapping
# --------------------------------------------------------------------------- #
def test_no_ccr_flag_disables_response_handling() -> None:
    config = _config_for(["proxy", "--no-ccr"])

    assert config.ccr_inject_tool is False
    assert config.ccr_inject_marker is False
    # The half that used to survive the switch.
    assert config.ccr_handle_responses is False


def test_no_ccr_env_var_disables_response_handling() -> None:
    config = _config_for(["proxy"], env={"HEADROOM_NO_CCR": "1"})

    assert config.ccr_inject_tool is False
    assert config.ccr_inject_marker is False
    assert config.ccr_handle_responses is False


def test_default_keeps_ccr_fully_on() -> None:
    """The switch must not leak into the default posture."""
    config = _config_for(["proxy"])

    assert config.ccr_inject_tool is True
    assert config.ccr_inject_marker is True
    assert config.ccr_handle_responses is True


# --------------------------------------------------------------------------- #
# What that mapping buys: no buffered flip, even when the client offers the tool
# --------------------------------------------------------------------------- #
@pytest.fixture
def _store():
    reset_compression_store()
    get_compression_store(backend=InMemoryBackend())
    try:
        yield
    finally:
        reset_compression_store()


def _upstream_stream_field(*, ccr_handle_responses: bool) -> object:
    """Drive one streaming turn and report the `stream` value sent upstream."""
    marker = get_compression_store().store(
        original=json.dumps({"earlier": "tool output"}),
        compressed="{}",
        original_item_count=400,
    )
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        memory_enabled=False,
        # `--no-ccr` posture: nothing injected by us.
        ccr_inject_tool=False,
        ccr_inject_marker=False,
        ccr_handle_responses=ccr_handle_responses,
        ccr_event_streaming=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    seen: dict[str, object] = {}
    app = create_app(config)
    with TestClient(app) as client:
        proxy = client.app.state.proxy

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            sent = json.loads(body) if isinstance(body, (str, bytes)) else body
            seen["stream"] = sent.get("stream")
            return httpx.Response(
                200,
                json={
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-6",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )

        proxy._retry_request = _fake_retry  # type: ignore[assignment]
        client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "stream": True,
                # The client advertises the tool itself, as the OpenCode plugin does.
                "tools": [create_ccr_tool_definition("anthropic")],
                "messages": [{"role": "user", "content": f"go <<ccr:{marker}>>"}],
            },
            headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
        )
    return seen.get("stream")


def test_client_offered_tool_still_buffers_when_handling_is_on(_store) -> None:  # noqa: ANN001
    """The behaviour being switched off — pinned so the switch is meaningful."""
    assert _upstream_stream_field(ccr_handle_responses=True) is False


def test_client_offered_tool_does_not_buffer_under_no_ccr(_store) -> None:  # noqa: ANN001
    """With the switch honoured, the turn keeps streaming despite the tool."""
    assert _upstream_stream_field(ccr_handle_responses=False) is not False
