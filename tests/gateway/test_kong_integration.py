"""End-to-end: Kong 3.9 + the `headroom` Lua plugin against a live Headroom.

Skipped unless ``HEADROOM_KONG_E2E=1``. Needs Docker Desktop (the container
reaches the host as ``host.docker.internal``). Runs Headroom in-process on a
uvicorn thread (``HEADROOM_COMPRESS_ALLOW_REMOTE=1``) and the scripted fake
provider on another, brings up the kong-plugin-headroom repo's ``docker/`` compose with
``docker compose up -d --wait``, and sends requests through Kong's ``/openai``
route.

Four checks, mirroring the spec (section 5):

a. proxy path: the provider receives a compressed body and the usage relay
   completes the pending turn (registry drained, outcome recorded);
b. loop path: a re-driving hook makes Headroom ask for a redrive; the client
   sees only the final answer and the provider saw two calls;
c. fail-open: with Headroom stopped the request still reaches the provider;
d. header contract: an Anthropic request through ``/anthropic`` with a large
   tool set is natively deferred, and the ``anthropic-beta`` header Headroom
   returns (client betas + ``advanced-tool-use``) reaches the provider.

Ports (env, defaults): ``HEADROOM_PORT=18787``, ``PROVIDER_PORT=18081``,
``KONG_PROXY_PORT=18000``, ``KONG_ADMIN_PORT=18001``.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from headroom.proxy.models import ProxyConfig
from headroom.proxy.turn_hooks import clear_turn_hooks
from tests.gateway.fake_provider import (
    FakeProvider,
    anthropic_text_response,
    openai_text_response,
    openai_tool_call_response,
    openai_usage,
)
from tests.gateway.samples import big_tool_history, openai_tools, tool_names

try:  # owned by the harness; may land after this file
    from tests.gateway.redrive_hook_ext import register as register_redrive_hook
except ImportError:  # pragma: no cover - harness not present yet
    register_redrive_hook = None

pytestmark = pytest.mark.skipif(
    os.environ.get("HEADROOM_KONG_E2E") != "1",
    reason="set HEADROOM_KONG_E2E=1 to run the Kong docker integration",
)

REPO = Path(__file__).resolve().parents[2]
# The Kong plugin lives in its own repo (kong-plugin-headroom); by default a
# sibling checkout of this one. Point HEADROOM_KONG_PLUGIN_DIR elsewhere if not.
PLUGIN_DIR = Path(
    os.environ.get("HEADROOM_KONG_PLUGIN_DIR") or str(REPO.parent / "kong-plugin-headroom")
).resolve()
COMPOSE_FILE = PLUGIN_DIR / "docker" / "docker-compose.yml"
COMPOSE_PROJECT = "headroom-kong-e2e"

HEADROOM_PORT = int(os.environ.get("HEADROOM_PORT", "18787"))
PROVIDER_PORT = int(os.environ.get("PROVIDER_PORT", "18081"))
KONG_PROXY_PORT = int(os.environ.get("KONG_PROXY_PORT", "18000"))
KONG_ADMIN_PORT = int(os.environ.get("KONG_ADMIN_PORT", "18001"))
SESSION_HEADER = "x-headroom-session-id"


# --------------------------------------------------------------------------- #
# Headroom on a uvicorn thread                                                 #
# --------------------------------------------------------------------------- #


class HeadroomServer:
    """Headroom's FastAPI app served on ``0.0.0.0:<port>`` from a daemon thread."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.app: Any = None
        self.proxy: Any = None
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self.recorded_outcomes: list[Any] = []

    def start(self) -> str:
        import uvicorn

        from headroom.proxy.server import create_app

        # The loopback guard on /v1/compress is decided when the app is built.
        os.environ["HEADROOM_COMPRESS_ALLOW_REMOTE"] = "1"
        self.app = create_app(
            ProxyConfig(
                host="0.0.0.0",
                port=self.port,
                optimize=True,
                cache_enabled=False,
                rate_limit_enabled=False,
                cost_tracking_enabled=False,
                log_requests=False,
                image_optimize=False,
            )
        )
        self.proxy = self.app.state.proxy
        self._capture_outcomes()
        config = uvicorn.Config(self.app, host="0.0.0.0", port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, name="headroom-e2e", daemon=True)
        self._thread.start()
        deadline = time.time() + 20
        while not self._server.started:
            if time.time() > deadline or not self._thread.is_alive():
                raise RuntimeError("headroom did not start")
            time.sleep(0.05)
        return f"http://127.0.0.1:{self.port}"

    def _capture_outcomes(self) -> None:
        """Wrap the proxy's outcome recorder so the relay's effect is observable."""
        original = getattr(self.proxy, "_record_request_outcome", None)
        if original is None:
            return

        def recording(outcome: Any, *args: Any, **kwargs: Any) -> Any:
            self.recorded_outcomes.append(outcome)
            return original(outcome, *args, **kwargs)

        self.proxy._record_request_outcome = recording

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(10)
        self._server = None
        self._thread = None

    def pending_turns(self) -> int | None:
        registry = getattr(self.proxy, "gateway_turns", None)
        return None if registry is None else len(registry)


def _wait_port_closed(port: int, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return
        time.sleep(0.1)


# --------------------------------------------------------------------------- #
# docker compose                                                               #
# --------------------------------------------------------------------------- #


def _compose(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = dict(
        os.environ,
        HEADROOM_PORT=str(HEADROOM_PORT),
        PROVIDER_PORT=str(PROVIDER_PORT),
        KONG_PROXY_PORT=str(KONG_PROXY_PORT),
        KONG_ADMIN_PORT=str(KONG_ADMIN_PORT),
    )
    return subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "-p", COMPOSE_PROJECT, *args],
        env=env,
        capture_output=True,
        text=True,
        check=check,
        timeout=300,
    )


def kong_logs(tail: int = 80) -> str:
    result = _compose("logs", "--no-color", "--tail", str(tail), "kong", check=False)
    return result.stdout + result.stderr


@dataclass
class Stack:
    provider: FakeProvider
    headroom: HeadroomServer
    kong_url: str

    def chat(
        self, body: dict[str, Any], session: str | None, timeout: float = 90
    ) -> httpx.Response:
        headers = {"content-type": "application/json", "authorization": "Bearer sk-test"}
        if session:
            headers[SESSION_HEADER] = session
        return httpx.post(
            f"{self.kong_url}/openai/v1/chat/completions",
            content=json.dumps(body),
            headers=headers,
            timeout=timeout,
        )

    def messages(
        self,
        body: dict[str, Any],
        session: str | None,
        extra_headers: dict[str, str] | None = None,
        timeout: float = 90,
    ) -> httpx.Response:
        """POST an Anthropic Messages body through Kong's ``/anthropic`` route."""
        headers = {
            "content-type": "application/json",
            "x-api-key": "sk-ant-test",
            "anthropic-version": "2023-06-01",
        }
        if session:
            headers[SESSION_HEADER] = session
        headers.update(extra_headers or {})
        return httpx.post(
            f"{self.kong_url}/anthropic/v1/messages",
            content=json.dumps(body),
            headers=headers,
            timeout=timeout,
        )


@pytest.fixture(scope="module")
def stack() -> Iterator[Stack]:
    if not COMPOSE_FILE.exists():
        pytest.skip(f"missing {COMPOSE_FILE}")
    provider = FakeProvider()
    provider.start(PROVIDER_PORT, host="0.0.0.0")
    stack_obj = Stack(
        provider=provider,
        headroom=HeadroomServer(HEADROOM_PORT),
        kong_url=f"http://127.0.0.1:{KONG_PROXY_PORT}",
    )
    try:
        stack_obj.headroom.start()
        try:
            _compose("up", "-d", "--wait", "--force-recreate")
        except subprocess.CalledProcessError as exc:
            pytest.fail(f"compose failed: {exc.stderr}\n--- kong logs ---\n{kong_logs()}")
        yield stack_obj
    finally:
        # `down` even after a failed `up`: it may have created the container.
        _compose("down", "-v", "--remove-orphans", check=False)
        stack_obj.headroom.stop()  # the fail-open test may have swapped this
        provider.stop()
        os.environ.pop("HEADROOM_COMPRESS_ALLOW_REMOTE", None)


@pytest.fixture(autouse=True)
def _arm_redrive_hook() -> Iterator[None]:
    """Register the re-driving hook for every test in this module.

    Function-scoped on purpose: ``tests/gateway/conftest.py`` clears turn hooks
    around each test, and module fixtures run before that clear. Hooks are
    process-global, so the uvicorn thread sees this registration.
    """
    if register_redrive_hook is not None:
        register_redrive_hook()
    yield
    clear_turn_hooks()


def _poll(predicate: Any, timeout: float = 10.0, interval: float = 0.1) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def _sse_frames(text: str) -> list[dict[str, Any]]:
    frames = []
    for line in text.splitlines():
        if line.startswith("data:"):
            data = line[5:].strip()
            if data and data != "[DONE]":
                frames.append(json.loads(data))
    return frames


# --------------------------------------------------------------------------- #
# (a) proxy path                                                               #
# --------------------------------------------------------------------------- #


def test_proxy_path_compresses_and_relays_usage(stack: Stack) -> None:
    """Streaming turn: Kong proxies, provider sees the compressed body, and the
    body_filter/log relay closes the turn on Headroom's side."""
    stack.provider.reset()
    stack.headroom.recorded_outcomes.clear()
    stack.provider.script(
        [openai_text_response("streamed answer", usage=openai_usage(120, 7, cached=64))]
    )
    body = {
        "model": "gpt-4o",
        "messages": big_tool_history(),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    sent_bytes = len(json.dumps(body["messages"]))

    resp = stack.chat(body, session="kong-e2e-proxy")
    assert resp.status_code == 200, f"{resp.text}\n{kong_logs()}"
    assert resp.headers["content-type"].startswith("text/event-stream")
    frames = _sse_frames(resp.text)
    assert any(f.get("usage") for f in frames), "provider's final SSE chunk should carry usage"

    assert len(stack.provider.calls) == 1, kong_logs()
    received = stack.provider.calls[0]
    assert received.stream is True
    received_bytes = len(json.dumps(received.messages))
    assert received_bytes < sent_bytes, (
        f"provider got {received_bytes}B, client sent {sent_bytes}B: not compressed\n{kong_logs()}"
    )
    # The relay carries the plugin version; the body must not leak Headroom-only keys.
    assert "gateway" not in received.body and "config" not in received.body

    # Response half: the log-phase timer posts usage; Headroom drains the turn
    # and records one outcome. Requires the server side of the contract.
    pending = stack.headroom.pending_turns
    if pending() is None:
        pytest.fail(
            "TODO(gateway-turn-contract): proxy has no `gateway_turns` registry yet; "
            "the usage relay cannot be asserted until the server side lands"
        )
    assert _poll(lambda: pending() == 0 and len(stack.headroom.recorded_outcomes) >= 1), (
        f"relay did not land: pending={pending()} outcomes={len(stack.headroom.recorded_outcomes)}\n"
        f"{kong_logs()}"
    )
    outcome = stack.headroom.recorded_outcomes[-1]
    assert getattr(outcome, "output_tokens", None) == 7


# --------------------------------------------------------------------------- #
# (b) loop path                                                                #
# --------------------------------------------------------------------------- #


def test_loop_path_redrives_and_hides_search_tools(stack: Stack) -> None:
    """Non-streaming turn with deferrable tools: Headroom asks for a redrive,
    the plugin runs it in `access`, the client sees only the final answer."""
    if register_redrive_hook is None:
        pytest.skip("tests/gateway/redrive_hook_ext.py not present; no re-driving hook to arm")
    stack.provider.reset()
    stack.provider.script(
        [
            openai_tool_call_response("search_tools", {"query": "deferred"}),
            openai_text_response("final answer", usage=openai_usage(30, 5, cached=0)),
        ]
    )
    tools = openai_tools()
    body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "use a deferred tool please"}],
        "tools": tools,
    }
    resp = stack.chat(body, session="kong-e2e-loop")
    assert resp.status_code == 200, f"{resp.text}\n{kong_logs()}"
    final = resp.json()
    assert "search_tools" not in resp.text, f"client saw the search_tools call:\n{resp.text}"
    assert final["choices"][0]["message"]["content"] == "final answer"

    calls = stack.provider.calls
    assert len(calls) == 2, f"provider saw {len(calls)} calls\n{kong_logs()}"
    first_names = tool_names(calls[0].tools)
    assert "search_tools" in first_names
    assert not any(n.startswith("deferred_") for n in first_names)
    second_names = tool_names(calls[1].tools)
    assert any(n.startswith("deferred_") for n in second_names), second_names
    # Both provider calls carried the client's auth on the loop path.
    assert all(c.headers.get("authorization") == "Bearer sk-test" for c in calls)
    assert _poll(lambda: stack.headroom.pending_turns() in (0, None)), "turn not drained"


# --------------------------------------------------------------------------- #
# (c) fail-open                                                                #
# --------------------------------------------------------------------------- #


def test_fail_open_when_headroom_is_down(stack: Stack) -> None:
    stack.provider.reset()
    stack.provider.script([openai_text_response("still here")])
    body = {"model": "gpt-4o", "messages": big_tool_history(n_items=20)}

    stack.headroom.stop()
    _wait_port_closed(HEADROOM_PORT)
    try:
        resp = stack.chat(body, session="kong-e2e-down")
    finally:
        # Order-independent: leave the stack as we found it.
        stack.headroom = HeadroomServer(HEADROOM_PORT)
        stack.headroom.start()

    assert resp.status_code == 200, f"{resp.text}\n{kong_logs()}"
    assert resp.json()["choices"][0]["message"]["content"] == "still here"
    assert len(stack.provider.calls) == 1
    # Original body, untouched.
    assert stack.provider.calls[0].messages == body["messages"]
    assert "gateway" not in stack.provider.calls[0].body


# --------------------------------------------------------------------------- #
# (d) header contract: native tool-search deferral + merged anthropic-beta     #
# --------------------------------------------------------------------------- #

ADVANCED_TOOL_USE_TOKEN = "advanced-tool-use"
CLIENT_BETA = "some-other-beta"


def anthropic_tool_set() -> list[dict[str, Any]]:
    """Thirteen Anthropic-shaped tools: three core coding tools that must stay
    resident plus ten integration tools Headroom is expected to defer."""
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "What to do. " * 8}},
        "required": ["query"],
    }
    names = [
        "Bash",
        "Read",
        "Edit",
        "jira_create_issue",
        "jira_search_issues",
        "slack_post_message",
        "slack_list_channels",
        "linear_create_ticket",
        "sentry_list_events",
        "notion_search_pages",
        "snowflake_run_query",
        "github_open_pull_request",
        "pagerduty_trigger_incident",
    ]
    return [{"name": n, "description": f"{n} tool.", "input_schema": schema} for n in names]


def _compress_contract_probe(body: dict[str, Any]) -> dict[str, Any]:
    """Ask Headroom directly what its /v1/compress answer looks like for this
    body, offering no capabilities so no pending turn is registered. Used only
    to tell "core not landed" apart from "plugin broke"."""
    payload = {
        **body,
        "config": {"session_id": "kong-e2e-anthropic-probe"},
        "gateway": {
            "can_redrive": False,
            "can_relay_response": False,
            "session_affinity": True,
            "plugin_version": "e2e-probe",
            "request_headers": {"anthropic-beta": CLIENT_BETA},
        },
    }
    resp = httpx.post(f"http://127.0.0.1:{HEADROOM_PORT}/v1/compress", json=payload, timeout=60)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert isinstance(data, dict)
    return data


def test_anthropic_native_deferral_sets_merged_beta_header(stack: Stack) -> None:
    """Anthropic-shaped request with 13 tools and a client ``anthropic-beta``:
    the provider must see the non-core tools deferred behind a regex tool-search
    tool, and an ``anthropic-beta`` header carrying both the client's beta and
    Anthropic's ``advanced-tool-use`` token (the plugin sets what Headroom
    returned in ``headers``)."""
    stack.provider.reset()
    stack.headroom.recorded_outcomes.clear()
    stack.provider.script([anthropic_text_response("deferred answer")])
    # Deferral is a turn hook's job on this path: the tool-search extension's
    # native tier when it is importable here, else the reference hook. Headroom
    # runs in-process, so registering here is what an installed extension does.
    from headroom.proxy.turn_hooks import register_turn_hook

    try:
        from headroom_tool_search.native import NativeDeferralHook

        register_turn_hook(NativeDeferralHook(None))
    except Exception:
        from tests.gateway.deferral_hook import DeferralHook

        register_turn_hook(DeferralHook())
    body = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "open a jira ticket for the flaky build"}],
        "tools": anthropic_tool_set(),
    }

    resp = stack.messages(
        body, session="kong-e2e-anthropic", extra_headers={"anthropic-beta": CLIENT_BETA}
    )
    assert resp.status_code == 200, f"{resp.text}\n{kong_logs()}"
    assert resp.json()["content"][0]["text"] == "deferred answer"

    calls = stack.provider.calls
    assert len(calls) == 1, f"provider saw {len(calls)} calls\n{kong_logs()}"
    received = calls[0]
    assert received.path.endswith("/v1/messages")
    assert "gateway" not in received.body and "config" not in received.body
    # The plugin's own header handling, independent of the contract addition:
    # the client's auth passthrough must not be disturbed by header merging.
    assert received.headers.get("x-api-key") == "sk-ant-test"

    probe = _compress_contract_probe(body)
    if "headers" not in probe:
        pytest.xfail(
            "core pending: /v1/compress answered without `headers` "
            f"(keys: {sorted(probe)}); native deferral + anthropic-beta merge not landed"
        )

    # (a) native tool-search deferral reached the provider.
    tools = received.tools
    names = tool_names(tools)
    assert any(n.startswith("tool_search_tool_regex") for n in names), names
    deferred = {t["name"] for t in tools if isinstance(t, dict) and t.get("defer_loading") is True}
    resident = {
        t["name"]
        for t in tools
        if isinstance(t, dict) and t.get("name") and not t.get("defer_loading")
    }
    assert "jira_create_issue" in deferred and "slack_post_message" in deferred, tools
    assert {"Bash", "Read", "Edit"} <= resident, tools
    assert not any(n.startswith("tool_search_tool_regex") for n in deferred)

    # (b) the merged anthropic-beta header: client beta kept, Anthropic's token added.
    beta = received.headers.get("anthropic-beta", "")
    tokens = [t.strip() for t in beta.split(",") if t.strip()]
    assert CLIENT_BETA in tokens, f"client beta dropped: {beta!r}\n{kong_logs()}"
    assert any(t.startswith(ADVANCED_TOOL_USE_TOKEN) for t in tokens), (
        f"advanced-tool-use missing from {beta!r}; headroom returned {probe.get('headers')!r}\n"
        f"{kong_logs()}"
    )
    assert _poll(lambda: stack.headroom.pending_turns() in (0, None)), "turn not drained"
