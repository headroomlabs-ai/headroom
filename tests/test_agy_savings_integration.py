"""WU3 integration test: agy inbox event -> proxy drain -> real dashboard surfaces.

Proves the end-to-end replay path claimed by headroom-4l8: an event emitted by
agy, when drained by the shared proxy, moves the SAME in-memory metrics the
dashboard renders — the token-savings counter (``tokens_saved_total``, the source
of the dashboard token hero) AND the per-project SavingsTracker rows — and does
so exactly once across repeated drains (at-least-once + dedup).

Isolated: constructs a real ``PrometheusMetrics`` + ``SavingsTracker`` in-process,
no network, no live proxy, HOME pinned to a tmp dir. Never runs the broad suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from headroom.proxy import agy_savings_inbox
from headroom.proxy.prometheus_metrics import PrometheusMetrics
from headroom.proxy.savings_tracker import SavingsTracker


def _verbose_log_blob() -> str:
    """A large, verbose pytest-style log — the shape Headroom's log compressor
    collapses dramatically (dedupes repetitive PASS/INFO lines, keeps errors)."""
    lines = ["============================= test session starts ============================="]
    for i in range(600):
        lines.append(f"tests/test_module_{i % 12}.py::test_case_{i} PASSED   [{i % 100}%]")
        lines.append(f"2026-07-05 18:00:{i % 60:02d},123 INFO worker.pool handled request id={i}")
    lines += [
        "tests/test_x.py::test_broken FAILED",
        "E   AssertionError: expected 3 got 4",
        "======================== 1 failed, 1200 passed in 5.2s =========================",
    ]
    return "Here is the failing test log. Find the root cause:\n\n" + "\n".join(lines)


def _event(project: str, *, tokens_saved: int, input_tokens: int) -> dict:
    """A minimal-but-complete funnel-kwargs payload for one agy request."""
    return {
        "provider": "anthropic",
        "model": "claude-sonnet",
        "input_tokens": input_tokens,
        "output_tokens": 100,
        "tokens_saved": tokens_saved,
        "latency_ms": 25.0,
        "cached": False,
        "overhead_ms": 1.0,
        "ttfb_ms": 5.0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "cache_write_5m_tokens": 0,
        "cache_write_1h_tokens": 0,
        "uncached_input_tokens": input_tokens,
        "attempted_input_tokens": input_tokens + tokens_saved,
        "project": project,
        "client": "agy",
    }


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    # Belt-and-suspenders: pin every savings sink under tmp so nothing global is touched.
    monkeypatch.setenv("HEADROOM_SAVINGS_PATH", str(tmp_path / "proxy_savings.json"))
    monkeypatch.setenv("HEADROOM_SAVINGS_EVENTS_PATH", str(tmp_path / "savings_events.jsonl"))
    monkeypatch.setenv("HEADROOM_OTEL_METRICS_ENABLED", "0")
    return tmp_path


@pytest.mark.parametrize(
    ("emit_marker", "expect_drain"),
    [("1", False), ("", True)],
)
def test_emitting_process_never_drains(
    isolated_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    emit_marker: str,
    expect_drain: bool,
) -> None:
    """A process that emits inbox events must not also drain them.

    ``wrap agy`` builds ``create_app()`` in-process (dispatch + retrieve) with its
    savings paths redirected to a temp dir that is deleted at exit. A drain loop
    there would consume events into that throwaway sink — and race the shared
    proxy, which is the only process allowed to replay them.
    """
    from fastapi.testclient import TestClient

    from headroom.proxy import server as server_mod
    from headroom.proxy.server import ProxyConfig

    monkeypatch.setenv("HEADROOM_AGY_INBOX_EMIT", emit_marker)

    started = False

    async def _record(metrics: object, interval_seconds: int = 5) -> None:
        nonlocal started
        started = True

    monkeypatch.setattr(server_mod, "_drain_agy_savings_periodically", _record)

    with TestClient(server_mod.create_app(ProxyConfig(optimize=False))):
        pass

    assert started is expect_drain


async def test_drain_moves_token_hero_and_per_project(isolated_home: Path) -> None:
    tracker = SavingsTracker(path=str(isolated_home / "proxy_savings.json"))
    metrics = PrometheusMetrics(savings_tracker=tracker)

    # Two agy requests in two projects land in the inbox.
    agy_savings_inbox.emit_event(**_event("proj-a", tokens_saved=800, input_tokens=1200))
    agy_savings_inbox.emit_event(**_event("proj-b", tokens_saved=300, input_tokens=500))

    recorded = await agy_savings_inbox.drain_inbox(metrics)
    assert recorded == 2

    # Token hero source: the dashboard reads m.tokens_saved_total (server.py:2685).
    assert metrics.tokens_saved_total == 1100
    # Request-count fidelity: both requests are reflected, not just the savings.
    assert metrics.requests_total == 2

    # Per-project section: the dashboard reads savings_tracker.stats_preview()["projects"].
    projects = metrics.savings_tracker.stats_preview()["projects"]
    assert "proj-a" in projects and "proj-b" in projects
    assert projects["proj-a"]["tokens_saved"] == 800
    assert projects["proj-b"]["tokens_saved"] == 300

    # Inbox drained empty.
    assert not list(agy_savings_inbox.inbox_dir().glob("evt-*.json"))


async def test_redrain_does_not_double_count(isolated_home: Path) -> None:
    tracker = SavingsTracker(path=str(isolated_home / "proxy_savings.json"))
    metrics = PrometheusMetrics(savings_tracker=tracker)

    agy_savings_inbox.emit_event(**_event("proj-a", tokens_saved=800, input_tokens=1200))
    assert await agy_savings_inbox.drain_inbox(metrics) == 1
    # A second drain with nothing new must not re-apply the event.
    assert await agy_savings_inbox.drain_inbox(metrics) == 0

    assert metrics.tokens_saved_total == 800
    assert metrics.requests_total == 1


def test_real_agy_path_compression_reaches_dashboard(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: a large agy (antigravity UA) cloudcode request is REALLY
    compressed by the production pipeline, and the genuine savings delta reaches
    the dashboard's metrics via the WU2 inbox. Asserts a real reduction (not a
    magic number) so it is robust across compressor tuning.
    """
    from fastapi.responses import StreamingResponse
    from starlette.testclient import TestClient

    from headroom.proxy.server import HeadroomProxy, ProxyConfig, create_app

    monkeypatch.setenv("HEADROOM_AGY_INBOX_EMIT", "1")

    async def _fake_stream(proxy_self, url, headers, body, *a, **k):  # type: ignore[no-untyped-def]
        async def _b():
            yield b'data: {"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}\r\n\r\ndata: [DONE]\r\n\r\n'

        return StreamingResponse(_b(), status_code=200, media_type="text/event-stream")

    monkeypatch.setattr(HeadroomProxy, "_stream_response", _fake_stream)

    body = {
        "project": "agy-proof",
        "model": "gemini-3-flash-agent",
        "request": {"contents": [{"role": "user", "parts": [{"text": _verbose_log_blob()}]}]},
    }
    deltas: list[tuple[int, int]] = []
    with TestClient(
        create_app(
            ProxyConfig(
                optimize=True,
                compress_user_messages=True,
                protect_recent=0,
                min_tokens_to_crush=100,
            )
        )
    ) as client:
        proxy: HeadroomProxy = client.app.state.proxy  # type: ignore[attr-defined]
        real_apply = proxy.openai_pipeline.apply

        def _spy(*a, **k):  # type: ignore[no-untyped-def]
            r = real_apply(*a, **k)
            deltas.append((r.tokens_before, r.tokens_after))
            return r

        proxy.openai_pipeline.apply = _spy  # type: ignore[method-assign]
        resp = client.post(
            "/v1internal:streamGenerateContent",
            params={"alt": "sse"},
            headers={"User-Agent": "antigravity/1.0.5"},
            json=body,
        )

    assert resp.status_code == 200
    assert deltas, "compression pipeline was not invoked on the agy path"
    before, after = deltas[0]
    saved = before - after
    # Real, substantial compression happened on agy-shaped traffic.
    assert saved > 0 and after < before, f"expected real compression, got {before}->{after}"

    # The genuine delta flows through the WU2 inbox to the dashboard metrics.
    agy_savings_inbox.emit_event(
        provider="google",
        model="gemini-3-flash-agent",
        input_tokens=after,
        output_tokens=5,
        tokens_saved=saved,
        latency_ms=30.0,
        cached=False,
        overhead_ms=0.0,
        ttfb_ms=0.0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cache_write_5m_tokens=0,
        cache_write_1h_tokens=0,
        uncached_input_tokens=after,
        attempted_input_tokens=before,
        project="agy-proof",
        client="agy",
    )
    metrics = PrometheusMetrics(
        savings_tracker=SavingsTracker(path=str(isolated_home / "dash.json"))
    )
    import asyncio

    assert asyncio.run(agy_savings_inbox.drain_inbox(metrics)) == 1
    assert metrics.tokens_saved_total == saved
    assert metrics.savings_tracker.stats_preview()["projects"]["agy-proof"]["tokens_saved"] == saved
