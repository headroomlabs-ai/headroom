"""Headroom protocol fixture, NOT a replacement for the real Headroom integration gate.

The dataclass/deque shape matches the inspected main SHA. Guard test doubles
record dependency execution and simulate denied network peers/origins. Live tests
use the real package when it is installed (test_real_headroom.py).
"""

import sys
import types
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

SID = "11111111-1111-4111-8111-111111111111"
OTHER = "22222222-2222-4222-8222-222222222222"


@dataclass
class Log:
    request_id: str = "req-1"
    timestamp: str = "2026-10-05T20:00:00Z"
    provider: str = "anthropic"
    model: str = "claude-test-fixture"
    input_tokens_original: int = 1000
    input_tokens_optimized: int = 600
    output_tokens: int = 100
    tokens_saved: int = 400
    savings_percent: float = 40
    optimization_latency_ms: float = 8
    total_latency_ms: float = 50
    tags: dict = field(default_factory=lambda: {"mod-session": SID})
    cache_hit: bool = False
    transforms_applied: list = field(default_factory=lambda: ["smart_crusher"])
    savings_breakdown: list = field(default_factory=list)
    cache_read_tokens: int = 200
    cache_write_tokens: int = 100
    uncached_input_tokens: int = 300
    request_messages: list | None = field(
        default_factory=lambda: [{"role": "user", "content": "original tool output"}]
    )
    compressed_messages: list | None = field(
        default_factory=lambda: [{"role": "user", "content": "compressed tool output"}]
    )
    response_content: str | None = "MUST_NOT_BE_SERVED"
    error: str | None = None
    turn_id: str | None = None


def make_app(monkeypatch, *, capture=True, entries=None, capacity=10000):
    # These named doubles prove the extension actually wires both dependencies.
    guard = types.ModuleType("headroom.proxy.loopback_guard")
    calls = []

    def require_loopback(request: Request):
        calls.append("peer-and-host")
        if request.client.host != "127.0.0.1" or request.url.hostname != "127.0.0.1":
            raise HTTPException(404)

    def require_same_origin(request: Request):
        calls.append("origin")
        if request.headers.get("origin", "http://127.0.0.1") != "http://127.0.0.1":
            raise HTTPException(403)

    guard.require_loopback = require_loopback
    guard.require_same_origin = require_same_origin
    monkeypatch.setitem(sys.modules, "headroom.proxy.loopback_guard", guard)
    from headroom_claude_mod.extension import install

    app = FastAPI()
    logger = SimpleNamespace(
        _logs=deque(entries if entries is not None else [Log()], maxlen=capacity),
        MESSAGE_WINDOW=100,
    )
    app.state.proxy = SimpleNamespace(logger=logger)
    config = SimpleNamespace(log_full_messages=capture)
    install(app, config)
    return app, logger, calls, config


@pytest.fixture
def system(monkeypatch):
    app, logger, calls, config = make_app(monkeypatch)
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345)) as client:
        yield client, logger, calls, config
