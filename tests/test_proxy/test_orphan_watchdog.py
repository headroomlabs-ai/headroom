"""Tests for the wrap-spawned proxy orphan watchdog."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from starlette.responses import StreamingResponse

import headroom.proxy.server as server
from headroom.proxy import orphan_watchdog as ow
from headroom.proxy.server import ProxyConfig


def _write_marker(clients_dir: Path, pid: int, **extra: object) -> Path:
    clients_dir.mkdir(parents=True, exist_ok=True)
    marker = clients_dir / f"{pid}.json"
    payload = {"pid": pid, "started_at": 1.0}
    payload.update(extra)
    marker.write_text(json.dumps(payload), encoding="utf-8")
    return marker


class _Proxy:
    def __init__(self, *, port: int = 8787, active_sessions: int = 0) -> None:
        self.config = SimpleNamespace(port=port)
        self.ws_sessions = SimpleNamespace(active_count=lambda: active_sessions)
        self._active_requests = 0
        self._activity_generation = 0

    @property
    def active_request_count(self) -> int:
        return self._active_requests

    @property
    def activity_generation(self) -> int:
        return self._activity_generation


class TestEnvConfig:
    def test_enabled_requires_wrap_owned_flag(self, monkeypatch) -> None:
        monkeypatch.delenv(ow.WRAP_OWNED_ENV, raising=False)
        assert ow.orphan_watchdog_enabled() is False
        monkeypatch.setenv(ow.WRAP_OWNED_ENV, "1")
        assert ow.orphan_watchdog_enabled() is True
        monkeypatch.setenv(ow.WRAP_OWNED_ENV, "0")
        assert ow.orphan_watchdog_enabled() is False
        # Explicit mapping wins over os.environ.
        assert ow.orphan_watchdog_enabled({ow.WRAP_OWNED_ENV: "true"}) is True

    def test_grace_seconds_default_override_and_floor(self, monkeypatch) -> None:
        monkeypatch.delenv(ow.GRACE_SECONDS_ENV, raising=False)
        assert ow.orphan_grace_seconds() == ow.DEFAULT_GRACE_SECONDS
        monkeypatch.setenv(ow.GRACE_SECONDS_ENV, "120")
        assert ow.orphan_grace_seconds() == 120.0
        # Below the floor: clamped, so a typo cannot exit a proxy under a
        # client that has not registered yet.
        monkeypatch.setenv(ow.GRACE_SECONDS_ENV, "5")
        assert ow.orphan_grace_seconds() == ow.MIN_GRACE_SECONDS
        monkeypatch.setenv(ow.GRACE_SECONDS_ENV, "not-a-number")
        assert ow.orphan_grace_seconds() == ow.DEFAULT_GRACE_SECONDS


class TestLiveClientPids:
    def test_live_marker_is_kept(self, tmp_path) -> None:
        _write_marker(tmp_path, os.getpid())
        assert ow.live_client_pids(tmp_path) == [os.getpid()]

    def test_dead_pid_marker_is_pruned(self, tmp_path) -> None:
        # PID 2**31 - 1 is never alive.
        marker = _write_marker(tmp_path, 2**31 - 1)
        assert ow.live_client_pids(tmp_path) == []
        assert not marker.exists()

    def test_recycled_pid_marker_is_pruned(self, tmp_path, monkeypatch) -> None:
        # Live PID (ours) but a recorded identity that provably differs.
        monkeypatch.setattr(ow, "identity_mismatch", lambda src, recorded, pid: True)
        marker = _write_marker(tmp_path, os.getpid(), start_src="psutil", start_time=1.0)
        assert ow.live_client_pids(tmp_path) == []
        assert not marker.exists()

    def test_garbage_and_non_numeric_markers_are_ignored(self, tmp_path) -> None:
        (tmp_path / "not-a-pid.json").write_text("{}", encoding="utf-8")
        (tmp_path / f"{os.getpid()}.json").write_text("not json", encoding="utf-8")
        # Unparseable marker JSON is not proof of recycling: PID is alive, kept.
        assert ow.live_client_pids(tmp_path) == [os.getpid()]

    def test_oversized_numeric_marker_is_pruned(self, tmp_path) -> None:
        marker = _write_marker(tmp_path, int("9" * 100))

        assert ow.live_client_pids(tmp_path) == []
        assert not marker.exists()

    def test_non_dict_marker_is_not_recycling_proof(self, tmp_path) -> None:
        (tmp_path / f"{os.getpid()}.json").write_text("[]", encoding="utf-8")
        assert ow.live_client_pids(tmp_path) == [os.getpid()]

    def test_missing_dir_is_empty(self, tmp_path) -> None:
        assert ow.live_client_pids(tmp_path / "nope") == []

    def test_client_scan_error_is_unknown(self, tmp_path, monkeypatch) -> None:
        tmp_path.mkdir(exist_ok=True)

        def fail_scandir(path: object):  # type: ignore[no-untyped-def]
            raise PermissionError("scan unavailable")

        monkeypatch.setattr(ow.os, "scandir", fail_scandir)
        assert ow.live_client_pids(tmp_path) is None


class TestWatchdogLoop:
    @staticmethod
    def _proxy(port: int = 8787, active_sessions: int = 0) -> _Proxy:
        return _Proxy(port=port, active_sessions=active_sessions)

    def test_stops_after_grace_with_no_clients(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(ow, "proxy_clients_dir", lambda port: tmp_path)
        stopped: list[bool] = []

        asyncio.run(
            ow.orphan_watchdog_loop(
                self._proxy(),
                grace_seconds=0.05,
                interval_seconds=0.01,
                stop=lambda: stopped.append(True),
            )
        )

        assert stopped == [True]

    def test_no_stop_while_client_alive(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(ow, "proxy_clients_dir", lambda port: tmp_path)
        _write_marker(tmp_path, os.getpid())
        stopped: list[bool] = []

        async def run() -> None:
            task = asyncio.create_task(
                ow.orphan_watchdog_loop(
                    self._proxy(),
                    grace_seconds=0.3,
                    interval_seconds=0.01,
                    stop=lambda: stopped.append(True),
                )
            )
            await asyncio.sleep(0.4)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(run())

        assert stopped == []

    def test_no_stop_while_ws_session_active(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(ow, "proxy_clients_dir", lambda port: tmp_path)
        stopped: list[bool] = []

        async def run() -> None:
            task = asyncio.create_task(
                ow.orphan_watchdog_loop(
                    self._proxy(active_sessions=1),
                    grace_seconds=0.3,
                    interval_seconds=0.01,
                    stop=lambda: stopped.append(True),
                )
            )
            await asyncio.sleep(0.4)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(run())

        assert stopped == []

    def test_no_stop_while_request_activity_continues(self, tmp_path, monkeypatch) -> None:
        """Direct-HTTP / SSH-forwarded clients leave no marker and may hold no
        WS session; activity generation movement must still hold the exit."""
        monkeypatch.setattr(ow, "proxy_clients_dir", lambda port: tmp_path)
        counter = {"n": 0}
        proxy = self._proxy()
        stopped: list[bool] = []

        async def serve_traffic() -> None:
            while True:
                await asyncio.sleep(0.05)
                counter["n"] += 1
                proxy._activity_generation = counter["n"]

        async def run() -> None:
            traffic = asyncio.create_task(serve_traffic())
            task = asyncio.create_task(
                ow.orphan_watchdog_loop(
                    proxy,
                    grace_seconds=0.15,
                    interval_seconds=0.01,
                    stop=lambda: stopped.append(True),
                )
            )
            await asyncio.sleep(0.4)
            task.cancel()
            traffic.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(run())

        assert stopped == []

    def test_streaming_request_resets_full_grace_after_completion(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(ow, "proxy_clients_dir", lambda port: tmp_path)
        proxy = self._proxy()
        app = FastAPI()
        app.add_middleware(server.ActivityMiddleware, proxy=proxy)
        stream_started = asyncio.Event()
        release_stream = asyncio.Event()
        stopped: list[bool] = []

        @app.get("/stream")
        async def stream() -> StreamingResponse:
            async def body():  # type: ignore[no-untyped-def]
                stream_started.set()
                yield b"started"
                await release_stream.wait()
                yield b"finished"

            return StreamingResponse(body())

        async def run() -> None:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                watchdog = asyncio.create_task(
                    ow.orphan_watchdog_loop(
                        proxy,
                        grace_seconds=0.08,
                        interval_seconds=0.01,
                        stop=lambda: stopped.append(True),
                    )
                )
                request = asyncio.create_task(client.get("/stream"))
                await asyncio.wait_for(stream_started.wait(), timeout=1)
                await asyncio.sleep(0.12)
                assert stopped == []
                assert proxy.active_request_count == 1

                release_stream.set()
                await request
                await asyncio.sleep(0.04)
                assert stopped == []
                await asyncio.wait_for(watchdog, timeout=0.2)

        asyncio.run(run())

        assert stopped == [True]

    def test_short_http_pulse_restarts_full_grace(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(ow, "proxy_clients_dir", lambda port: tmp_path)
        proxy = self._proxy()
        app = FastAPI()
        app.add_middleware(server.ActivityMiddleware, proxy=proxy)
        stopped: list[bool] = []

        @app.get("/pulse")
        async def pulse() -> dict[str, bool]:
            return {"ok": True}

        async def run() -> None:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                watchdog = asyncio.create_task(
                    ow.orphan_watchdog_loop(
                        proxy,
                        grace_seconds=0.08,
                        interval_seconds=0.02,
                        stop=lambda: stopped.append(True),
                    )
                )
                await asyncio.sleep(0.05)
                before = proxy.activity_generation
                response = await client.get("/pulse")
                assert response.status_code == 200
                assert proxy.active_request_count == 0
                assert proxy.activity_generation == before + 2

                # The pre-pulse idle window would have expired by now. The
                # observed generation change must start a full new window.
                await asyncio.sleep(0.06)
                assert stopped == []
                await asyncio.wait_for(watchdog, timeout=0.2)

        asyncio.run(run())

        assert stopped == [True]

    def test_activity_middleware_tracks_held_websocket_scope(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(ow, "proxy_clients_dir", lambda port: tmp_path)
        proxy = self._proxy()
        entered = asyncio.Event()
        release = asyncio.Event()
        stopped: list[bool] = []

        async def held_app(scope, receive, send):  # type: ignore[no-untyped-def]
            entered.set()
            await release.wait()

        async def unused():  # type: ignore[no-untyped-def]
            return {}

        async def run() -> None:
            middleware = server.ActivityMiddleware(held_app, proxy=proxy)
            watchdog = asyncio.create_task(
                ow.orphan_watchdog_loop(
                    proxy,
                    grace_seconds=0.05,
                    interval_seconds=0.01,
                    stop=lambda: stopped.append(True),
                )
            )
            task = asyncio.create_task(middleware({"type": "websocket"}, unused, unused))
            await asyncio.wait_for(entered.wait(), timeout=1)
            assert proxy.active_request_count == 1
            assert proxy.activity_generation == 1
            await asyncio.sleep(0.08)
            assert stopped == []

            release.set()
            await task
            assert proxy.active_request_count == 0
            assert proxy.activity_generation == 2
            await asyncio.wait_for(watchdog, timeout=0.2)

        asyncio.run(run())
        assert stopped == [True]

    def test_registry_failure_resets_grace_until_activity_is_observable(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(ow, "proxy_clients_dir", lambda port: tmp_path)
        observable = False

        def active_count() -> int:
            if not observable:
                raise RuntimeError("registry unavailable")
            return 0

        proxy = self._proxy()
        proxy.ws_sessions = SimpleNamespace(active_count=active_count)
        stopped: list[bool] = []

        async def run() -> None:
            nonlocal observable
            watchdog = asyncio.create_task(
                ow.orphan_watchdog_loop(
                    proxy,
                    grace_seconds=0.08,
                    interval_seconds=0.01,
                    stop=lambda: stopped.append(True),
                )
            )
            await asyncio.sleep(0.12)
            assert stopped == []

            observable = True
            await asyncio.sleep(0.04)
            assert stopped == []
            await asyncio.wait_for(watchdog, timeout=0.2)

        asyncio.run(run())

        assert stopped == [True]

    @pytest.mark.parametrize("unknown_signal", ["clients", "activity", "generation"])
    def test_unknown_activity_signal_resets_grace_until_observable(
        self, unknown_signal, tmp_path, monkeypatch
    ) -> None:
        observable = False
        proxy = self._proxy()
        if unknown_signal == "clients":
            monkeypatch.setattr(
                ow,
                "live_client_pids",
                lambda clients_dir: [] if observable else None,
            )
        elif unknown_signal == "activity":
            monkeypatch.setattr(ow, "proxy_clients_dir", lambda port: tmp_path)
            proxy._active_requests = None
        else:
            monkeypatch.setattr(ow, "proxy_clients_dir", lambda port: tmp_path)
            proxy._activity_generation = None
        stopped: list[bool] = []

        async def run() -> None:
            nonlocal observable
            watchdog = asyncio.create_task(
                ow.orphan_watchdog_loop(
                    proxy,
                    grace_seconds=0.08,
                    interval_seconds=0.01,
                    stop=lambda: stopped.append(True),
                )
            )
            await asyncio.sleep(0.12)
            assert stopped == []

            observable = True
            proxy._active_requests = 0
            proxy._activity_generation = 0
            await asyncio.sleep(0.04)
            assert stopped == []
            await asyncio.wait_for(watchdog, timeout=0.2)

        asyncio.run(run())

        assert stopped == [True]

    def test_grace_restarts_when_client_returns(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(ow, "proxy_clients_dir", lambda port: tmp_path)
        stopped: list[bool] = []

        async def run() -> None:
            task = asyncio.create_task(
                ow.orphan_watchdog_loop(
                    self._proxy(),
                    grace_seconds=0.5,
                    interval_seconds=0.01,
                    stop=lambda: stopped.append(True),
                )
            )
            # Let the idle clock accumulate, then register a live client: the
            # clock must reset, not fire late.
            await asyncio.sleep(0.2)
            _write_marker(tmp_path, os.getpid())
            await asyncio.sleep(0.7)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(run())

        assert stopped == []

    def test_missing_ws_registry_counts_as_idle(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(ow, "proxy_clients_dir", lambda port: tmp_path)
        proxy = self._proxy()
        proxy.ws_sessions = None
        stopped: list[bool] = []

        asyncio.run(
            ow.orphan_watchdog_loop(
                proxy,
                grace_seconds=0.05,
                interval_seconds=0.01,
                stop=lambda: stopped.append(True),
            )
        )

        assert stopped == [True]


class TestDefensiveHelpers:
    def test_non_dict_marker_json_is_not_recycling_proof(self, tmp_path) -> None:
        # Valid JSON but not an object: only a dict record can prove recycling.
        marker = _write_marker(tmp_path, os.getpid())
        marker.write_text("[1, 2]", encoding="utf-8")
        assert ow._marker_pid_recycled(marker, os.getpid()) is False

    def test_unlink_oserror_is_tolerated(self, tmp_path, monkeypatch) -> None:
        dead_pid = 2**31 - 1  # never alive
        marker = _write_marker(tmp_path, dead_pid)
        real_unlink = Path.unlink

        def flaky_unlink(self: Path, *args: object, **kwargs: object) -> None:
            if self == marker:
                raise OSError("simulated EPERM")
            real_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", flaky_unlink)
        # The scan tolerates the failed prune and still reports no live clients.
        assert ow.live_client_pids(tmp_path) == []
        assert marker.exists()

    def test_active_session_count_reports_registry_errors_as_unknown(self) -> None:
        def boom() -> int:
            raise RuntimeError("registry exploded")

        proxy = SimpleNamespace(ws_sessions=SimpleNamespace(active_count=boom))
        assert ow._active_session_count(proxy) is None

    def test_active_request_count_is_unknown_when_unavailable(self) -> None:
        class BrokenProxy:
            @property
            def active_request_count(self) -> int:
                raise RuntimeError("counter unavailable")

        assert ow._active_request_count(SimpleNamespace()) is None
        assert ow._active_request_count(BrokenProxy()) is None

    def test_activity_generation_is_unknown_when_unavailable(self) -> None:
        assert ow._activity_generation(SimpleNamespace()) is None

    def test_default_stop_raises_sigterm(self, monkeypatch) -> None:
        sent: list[int] = []
        monkeypatch.setattr(ow.signal, "raise_signal", lambda sig: sent.append(sig))
        ow._default_stop()
        assert sent == [ow.signal.SIGTERM]


class TestServerWiring:
    """create_app must start the watchdog only for wrap-spawned single-worker
    proxies, and cancel it cleanly on shutdown."""

    @staticmethod
    def _config() -> ProxyConfig:
        return ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            periodic_malloc_trim_enabled=False,
        )

    @staticmethod
    async def _fake_loop(*args: object, **kwargs: object) -> None:
        await asyncio.sleep(3600)

    def test_starts_and_stops_watchdog_when_wrap_owned(self, monkeypatch) -> None:
        monkeypatch.setenv(ow.WRAP_OWNED_ENV, "1")
        monkeypatch.delenv(server._MULTI_WORKER_CONFIG_ENV, raising=False)
        monkeypatch.setattr(server, "orphan_watchdog_loop", self._fake_loop)

        app = server.create_app(self._config())
        with TestClient(app) as client:
            task = client.app.state.orphan_watchdog_task
            assert task is not None
            assert not task.done()
        # Lifespan teardown cancelled it and cleared the state slot.
        assert app.state.orphan_watchdog_task is None

    def test_activity_middleware_is_outermost(self) -> None:
        app = server.create_app(self._config())
        assert app.user_middleware[0].cls is server.ActivityMiddleware
        assert app.state.proxy.active_request_count == 0
        assert app.state.proxy.activity_generation == 0

    def test_skips_watchdog_for_multi_worker(self, monkeypatch) -> None:
        monkeypatch.setenv(ow.WRAP_OWNED_ENV, "1")
        monkeypatch.setenv(server._MULTI_WORKER_CONFIG_ENV, "{}")
        monkeypatch.setattr(server, "orphan_watchdog_loop", self._fake_loop)

        app = server.create_app(self._config())
        with TestClient(app) as client:
            assert client.app.state.orphan_watchdog_task is None

    def test_skips_watchdog_when_not_wrap_owned(self, monkeypatch) -> None:
        monkeypatch.delenv(ow.WRAP_OWNED_ENV, raising=False)
        monkeypatch.delenv(server._MULTI_WORKER_CONFIG_ENV, raising=False)
        monkeypatch.setattr(server, "orphan_watchdog_loop", self._fake_loop)

        app = server.create_app(self._config())
        with TestClient(app) as client:
            assert client.app.state.orphan_watchdog_task is None
