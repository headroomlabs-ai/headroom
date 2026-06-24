#!/usr/bin/env python3
"""Repro harness: multi-agent Codex reconnect storm against a local proxy.

This script reproduces the failure class described in
``wiki/plans/2026-04-17-codex-proxy-runtime-analysis.md`` ("Latest Correction"):
a burst of concurrent Codex WebSocket sessions plus a parallel burst of large
Anthropic ``/v1/messages`` replays immediately after a fresh proxy restart, with
``/livez`` probed continuously to detect event-loop starvation.

Usage::

    python scripts/repro_codex_replay.py \\
        --url http://127.0.0.1:8787 \\
        --ws-clients 8 \\
        --anthropic-clients 4 \\
        --duration 30

Exit code is ``0`` iff the warmup phase succeeded (or was skipped), the storm
phase ran for the requested duration, and ``/livez`` p99 stayed at or below the
configured threshold (``--livez-threshold-ms``, default 500ms).

The harness adds no new pip dependencies: it uses ``asyncio`` + ``websockets``
+ ``httpx`` only, all already available in the Headroom dev environment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx

try:
    from websockets.asyncio.client import connect as ws_connect
except ImportError:  # pragma: no cover - older websockets fallback
    from websockets.client import connect as ws_connect  # type: ignore[no-redef]

# asyncio.timeout was added in Python 3.11; provide a thin shim for 3.10.
if sys.version_info >= (3, 11):
    _asyncio_timeout = asyncio.timeout
else:

    @asynccontextmanager
    async def _asyncio_timeout(delay: float | None):  # type: ignore[no-redef]
        """Minimal shim that mimics ``asyncio.timeout`` using ``wait_for``."""
        # For the single call-site we have, wrapping the entire body in a task
        # and cancelling it on timeout is sufficient.  We yield control, and if
        # the caller's block exceeds *delay* seconds the enclosing task is
        # cancelled.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + delay if delay is not None else None
        task = asyncio.current_task()
        handle = None
        if deadline is not None and task is not None:
            handle = loop.call_at(deadline, task.cancel)
        try:
            yield
        except asyncio.CancelledError:
            raise asyncio.TimeoutError() from None
        finally:
            if handle is not None:
                handle.cancel()


from websockets.exceptions import ConnectionClosed, InvalidStatus, WebSocketException

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WS_FRAME_FIXTURE = SCRIPT_DIR / "fixtures" / "codex_response_create_frame.json"
DEFAULT_ANTHROPIC_BODY_FIXTURE = SCRIPT_DIR / "fixtures" / "anthropic_replay_body.json"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class LatencyHistogram:
    """Minimal fixed-list histogram with p50/p95/p99/max computation."""

    samples_ms: list[float] = field(default_factory=list)

    def record(self, value_ms: float) -> None:
        self.samples_ms.append(value_ms)

    @property
    def count(self) -> int:
        return len(self.samples_ms)

    def percentile(self, p: float) -> float:
        if not self.samples_ms:
            return 0.0
        ordered = sorted(self.samples_ms)
        if p <= 0:
            return ordered[0]
        if p >= 100:
            return ordered[-1]
        idx = max(0, min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1)))))
        return ordered[idx]

    def as_summary(self) -> dict[str, float]:
        if not self.samples_ms:
            return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
        return {
            "count": self.count,
            "p50": self.percentile(50),
            "p95": self.percentile(95),
            "p99": self.percentile(99),
            "max": max(self.samples_ms),
        }


@dataclass
class CodexWsStats:
    opened: int = 0
    response_completed: int = 0
    errors: dict[str, int] = field(default_factory=dict)

    def record_error(self, kind: str) -> None:
        self.errors[kind] = self.errors.get(kind, 0) + 1


@dataclass
class AnthropicHttpStats:
    attempted: int = 0
    ok_2xx: int = 0
    non_2xx: int = 0
    timed_out: int = 0
    errors: int = 0
    first_byte_latency_ms: list[float] = field(default_factory=list)

    @property
    def avg_first_byte_ms(self) -> float:
        if not self.first_byte_latency_ms:
            return 0.0
        return statistics.mean(self.first_byte_latency_ms)


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def _http_to_ws_url(http_url: str, path: str) -> str:
    parsed = urlparse(http_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    # Preserve host:port; override scheme + path. Ensure path starts with a single "/".
    normalized_path = "/" + path.lstrip("/") if path else ""
    return urlunparse((scheme, parsed.netloc, normalized_path, "", "", ""))


# ---------------------------------------------------------------------------
# Warmup phase
# ---------------------------------------------------------------------------


async def warmup_probe(
    url: str,
    ws_frame: dict[str, Any],
    timeout_s: float = 10.0,
) -> tuple[bool, float, str]:
    """Open one WS, send the response.create frame, wait briefly.

    Returns ``(success, elapsed_ms, note)``. Success is true if the proxy
    accepts the WebSocket handshake and the initial frame is sent without
    error. We do not require ``response.completed`` — upstream auth will
    often fail locally and the proxy will send an error frame; the important
    signal is that the handshake + relay wiring worked.
    """
    ws_url = _http_to_ws_url(url, "/v1/responses")
    start = time.perf_counter()
    handshake_ok = False
    frame_sent = False
    note = "unknown"
    try:
        async with _asyncio_timeout(timeout_s):
            async with ws_connect(
                ws_url,
                additional_headers={"Authorization": "Bearer repro-harness"},
            ) as ws:
                handshake_ok = True
                try:
                    await ws.send(json.dumps(ws_frame))
                    frame_sent = True
                except (ConnectionClosed, WebSocketException) as exc:
                    note = f"send_failed:{type(exc).__name__}"
                    return False, (time.perf_counter() - start) * 1000.0, note
                # Drain until terminal event, close, or deadline.
                deadline_local = time.perf_counter() + timeout_s
                while time.perf_counter() < deadline_local:
                    remaining = deadline_local - time.perf_counter()
                    if remaining <= 0:
                        break
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                    except asyncio.TimeoutError:
                        note = "timeout_waiting_for_completion"
                        break
                    except ConnectionClosed:
                        # Upstream (or mock) closed cleanly after accepting our
                        # frame — this still counts as a working pre-upstream
                        # path; the warmup's purpose is handshake + send, not
                        # demanding real upstream output.
                        note = "upstream_closed_after_send"
                        break
                    try:
                        evt = json.loads(raw) if isinstance(raw, str) else {}
                    except json.JSONDecodeError:
                        continue
                    etype = evt.get("type", "")
                    if etype == "response.completed":
                        note = "response.completed"
                        break
                    if etype in {"error", "response.failed"}:
                        note = f"terminal:{etype}"
                        break
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return frame_sent, elapsed_ms, note
    except (OSError, WebSocketException, InvalidStatus, asyncio.TimeoutError) as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        # If the handshake completed and we sent the frame, a subsequent close
        # error from the context manager is not a warmup failure.
        if frame_sent:
            return True, elapsed_ms, f"post_send_close:{type(exc).__name__}"
        if handshake_ok:
            return False, elapsed_ms, f"post_handshake_error:{type(exc).__name__}: {exc}"
        return False, elapsed_ms, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Storm phase
# ---------------------------------------------------------------------------


async def _ws_client(
    idx: int,
    url: str,
    ws_frame: dict[str, Any],
    deadline: float,
    stats: CodexWsStats,
) -> None:
    ws_url = _http_to_ws_url(url, "/v1/responses")
    try:
        async with ws_connect(
            ws_url,
            additional_headers={"Authorization": f"Bearer repro-harness-ws-{idx}"},
            open_timeout=10,
        ) as ws:
            stats.opened += 1
            try:
                await ws.send(json.dumps(ws_frame))
            except (ConnectionClosed, WebSocketException) as exc:
                stats.record_error(f"send:{type(exc).__name__}")
                return
            # Hold the session open, draining events until deadline or close.
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 5.0))
                except asyncio.TimeoutError:
                    continue
                except ConnectionClosed:
                    break
                try:
                    evt = json.loads(raw) if isinstance(raw, str) else {}
                except json.JSONDecodeError:
                    continue
                etype = evt.get("type", "")
                if etype == "response.completed":
                    stats.response_completed += 1
                    break
                if etype in {"error", "response.failed"}:
                    stats.record_error(f"upstream:{etype}")
                    break
    except (OSError, InvalidStatus) as exc:
        stats.record_error(f"connect:{type(exc).__name__}")
    except WebSocketException as exc:
        stats.record_error(f"ws:{type(exc).__name__}")
    except Exception as exc:  # pragma: no cover - defensive
        stats.record_error(f"unexpected:{type(exc).__name__}")


async def _anthropic_client(
    idx: int,
    url: str,
    body: dict[str, Any],
    deadline: float,
    stats: AnthropicHttpStats,
) -> None:
    endpoint = url.rstrip("/") + "/v1/messages?beta=true"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer repro-harness-anthropic-{idx}",
        "anthropic-version": "2023-06-01",
        "x-api-key": f"repro-harness-anthropic-{idx}",
    }
    retry_cutoff = min(deadline, time.perf_counter() + 60.0)
    attempt = 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0)) as client:
        while time.perf_counter() < deadline:
            attempt += 1
            stats.attempted += 1
            start = time.perf_counter()
            try:
                resp = await client.post(endpoint, headers=headers, json=body)
                first_byte_ms = (time.perf_counter() - start) * 1000.0
                stats.first_byte_latency_ms.append(first_byte_ms)
                # Drain body so the connection is released.
                with suppress(Exception):
                    _ = resp.content
                if 200 <= resp.status_code < 300:
                    stats.ok_2xx += 1
                    return
                stats.non_2xx += 1
                if resp.status_code < 500:
                    # Client error — no retry, agent would surface this.
                    return
            except httpx.TimeoutException:
                stats.timed_out += 1
            except (httpx.HTTPError, OSError):
                stats.errors += 1
            # Retry loop — mimic agent behavior with bounded wall clock.
            if time.perf_counter() >= retry_cutoff:
                return
            # Exponential backoff with 50-150% jitter, mirroring the
            # ``jitter_delay_ms(base_ms=250, max_ms=5000, attempt=n)``
            # helper used inside the proxy. Keeping the formula inline
            # so this script stays dependency-free of the proxy package.
            _base_ms = 250
            _max_ms = 5000
            _delay_ms = min(_base_ms * (2 ** (attempt - 1)), _max_ms) * (0.5 + random.random())
            await asyncio.sleep(_delay_ms / 1000.0)


async def _livez_prober(
    url: str,
    histogram: LatencyHistogram,
    deadline: float,
    interval_ms: int = 250,
) -> None:
    endpoint = url.rstrip("/") + "/livez"
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=2.0)) as client:
        while time.perf_counter() < deadline:
            start = time.perf_counter()
            try:
                resp = await client.get(endpoint)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                if resp.status_code == 200:
                    histogram.record(elapsed_ms)
                else:
                    # Treat non-200 as worst-case for our threshold.
                    histogram.record(max(elapsed_ms, 5000.0))
            except (httpx.HTTPError, OSError):
                histogram.record(5000.0)
            # Tick every interval_ms regardless of probe latency.
            sleep_for = interval_ms / 1000.0
            next_wake = time.perf_counter() + sleep_for
            remaining = max(0.0, next_wake - time.perf_counter())
            if remaining > 0:
                await asyncio.sleep(remaining)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def _check_reachable(url: str, timeout_s: float = 5.0) -> tuple[bool, str]:
    endpoint = url.rstrip("/") + "/livez"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=timeout_s)) as client:
            resp = await client.get(endpoint)
            return (True, f"HTTP {resp.status_code}")
    except (httpx.ConnectError, ConnectionRefusedError, OSError) as exc:
        return (False, f"{type(exc).__name__}: {exc}")
    except httpx.HTTPError as exc:
        return (True, f"reachable-but-error: {type(exc).__name__}: {exc}")


async def run_harness(args: argparse.Namespace) -> dict[str, Any]:
    ws_frame = _load_json(Path(args.ws_frame_fixture))
    anthropic_body = _load_json(Path(args.anthropic_body_fixture))

    # Reachability gate — quick, clear failure if the proxy is not up.
    reachable, reach_note = await _check_reachable(args.url, timeout_s=5.0)
    if not reachable:
        return {
            "ok": False,
            "reason": "proxy_unreachable",
            "detail": reach_note,
            "url": args.url,
        }

    # Phase 1: warmup
    warmup_result: dict[str, Any]
    if args.no_warmup:
        warmup_result = {"skipped": True}
    else:
        success, elapsed_ms, note = await warmup_probe(
            args.url, ws_frame, timeout_s=args.warmup_timeout
        )
        warmup_result = {
            "skipped": False,
            "success": success,
            "elapsed_ms": round(elapsed_ms, 2),
            "note": note,
        }

    # Phase 2: storm
    livez_hist = LatencyHistogram()
    ws_stats = CodexWsStats()
    http_stats = AnthropicHttpStats()

    storm_start = time.perf_counter()
    deadline = storm_start + args.duration

    livez_task = asyncio.create_task(
        _livez_prober(args.url, livez_hist, deadline, interval_ms=args.livez_interval_ms),
        name="repro-livez-prober",
    )
    ws_tasks = [
        asyncio.create_task(
            _ws_client(i, args.url, ws_frame, deadline, ws_stats),
            name=f"repro-ws-{i}",
        )
        for i in range(args.ws_clients)
    ]
    http_tasks = [
        asyncio.create_task(
            _anthropic_client(i, args.url, anthropic_body, deadline, http_stats),
            name=f"repro-http-{i}",
        )
        for i in range(args.anthropic_clients)
    ]

    # Let the storm run. Gather storm tasks but keep probing /livez for the
    # full requested duration regardless of when the clients exit — the whole
    # point is to observe event-loop health across the window.
    storm_tasks = ws_tasks + http_tasks
    try:
        await asyncio.wait_for(
            asyncio.gather(*storm_tasks, return_exceptions=True),
            timeout=args.duration + 10.0,
        )
    except asyncio.TimeoutError:
        for t in storm_tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*storm_tasks, return_exceptions=True)

    # Let the /livez prober run to its own deadline so the histogram covers
    # the full window, not just the interval where storm tasks were alive.
    remaining = deadline - time.perf_counter()
    if remaining > 0:
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(livez_task), timeout=remaining + 2.0)
    if not livez_task.done():
        livez_task.cancel()
    with suppress(asyncio.CancelledError, Exception):
        await livez_task

    storm_duration_s = time.perf_counter() - storm_start
    livez_summary = livez_hist.as_summary()

    # Soft assertion: livez p99 under threshold.
    threshold_ms = args.livez_threshold_ms
    livez_ok = livez_summary["p99"] <= threshold_ms

    return {
        "ok": (warmup_result.get("skipped", False) or warmup_result.get("success", False))
        and livez_ok,
        "warmup": warmup_result,
        "storm": {
            "ws_clients": args.ws_clients,
            "anthropic_clients": args.anthropic_clients,
            "requested_duration_s": args.duration,
            "actual_duration_s": round(storm_duration_s, 3),
        },
        "livez": {
            **livez_summary,
            "threshold_ms": threshold_ms,
            "threshold_ok": livez_ok,
        },
        "codex_ws": {
            "opened": ws_stats.opened,
            "response_completed": ws_stats.response_completed,
            "errors": dict(ws_stats.errors),
        },
        "anthropic_http": {
            "attempted": http_stats.attempted,
            "ok_2xx": http_stats.ok_2xx,
            "non_2xx": http_stats.non_2xx,
            "timed_out": http_stats.timed_out,
            "errors": http_stats.errors,
            "avg_first_byte_ms": round(http_stats.avg_first_byte_ms, 2),
        },
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def format_summary(result: dict[str, Any]) -> str:
    if result.get("reason") == "proxy_unreachable":
        return (
            "Proxy unreachable at {url}.\n"
            "  {detail}\n"
            "  Hint: start the proxy with `headroom proxy` and retry."
        ).format(**result)

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("Codex proxy reconnect-storm repro harness — summary")
    lines.append("=" * 72)
    warm = result.get("warmup", {})
    if warm.get("skipped"):
        lines.append("Warmup:         skipped")
    else:
        lines.append(
            "Warmup:         success={success} elapsed_ms={elapsed_ms} note={note}".format(**warm)
        )
    storm = result.get("storm", {})
    lines.append(
        "Storm:          ws_clients={ws_clients} anthropic_clients={anthropic_clients} "
        "requested={requested_duration_s}s actual={actual_duration_s}s".format(**storm)
    )
    livez = result.get("livez", {})
    lines.append(
        "/livez:         count={count} p50={p50:.2f}ms p95={p95:.2f}ms "
        "p99={p99:.2f}ms max={max:.2f}ms (threshold={threshold_ms}ms, "
        "ok={threshold_ok})".format(**livez)
    )
    ws = result.get("codex_ws", {})
    lines.append(
        "Codex WS:       opened={opened} response.completed={response_completed} "
        "errors={errors}".format(**ws)
    )
    http = result.get("anthropic_http", {})
    lines.append(
        "Anthropic HTTP: attempted={attempted} ok_2xx={ok_2xx} non_2xx={non_2xx} "
        "timed_out={timed_out} errors={errors} avg_first_byte_ms={avg_first_byte_ms}".format(**http)
    )
    lines.append("=" * 72)
    lines.append("RESULT: {}".format("OK" if result.get("ok") else "FAIL"))
    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducibly exercise the multi-agent Codex reconnect/retry "
        "storm against a local Headroom proxy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8787",
        help="Base URL of the running proxy.",
    )
    parser.add_argument(
        "--ws-clients",
        type=int,
        default=8,
        help="Number of concurrent Codex WS connections to open during the storm.",
    )
    parser.add_argument(
        "--anthropic-clients",
        type=int,
        default=4,
        help="Number of concurrent Anthropic /v1/messages POST clients.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Total storm phase length, in seconds.",
    )
    parser.add_argument(
        "--livez-threshold-ms",
        type=float,
        default=500.0,
        help="Soft assertion threshold for /livez p99 (ms). Exit non-zero if exceeded.",
    )
    parser.add_argument(
        "--livez-interval-ms",
        type=int,
        default=250,
        help="Interval between /livez probes, in milliseconds.",
    )
    parser.add_argument(
        "--warmup-timeout",
        type=float,
        default=10.0,
        help="Max seconds to wait for the warmup WS session to complete.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the warmup probe phase.",
    )
    parser.add_argument(
        "--ws-frame-fixture",
        default=str(DEFAULT_WS_FRAME_FIXTURE),
        help="Path to the Codex response.create frame JSON fixture.",
    )
    parser.add_argument(
        "--anthropic-body-fixture",
        default=str(DEFAULT_ANTHROPIC_BODY_FIXTURE),
        help="Path to the Anthropic /v1/messages request body JSON fixture.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit the full summary as JSON on stdout; the human-readable "
            "summary is routed to stderr. Without --json, the human summary "
            "is on stdout and nothing goes to JSON."
        ),
    )
    return parser


# Exit codes. Keep POSIX-style — 0 means success, non-zero categorises
# the failure so callers (CI, smoke tests, shell wrappers) can branch.
EXIT_OK = 0
EXIT_CRASH = 1
EXIT_PROXY_UNREACHABLE = 2
EXIT_LIVEZ_THRESHOLD = 3
EXIT_WARMUP_FAILED = 4
EXIT_SIGINT = 130


def _classify_exit(result: dict[str, Any]) -> int:
    """Map a harness result dict onto one of the exit constants above."""
    if result.get("reason") == "proxy_unreachable":
        return EXIT_PROXY_UNREACHABLE
    warm = result.get("warmup", {})
    if not warm.get("skipped", False) and not warm.get("success", True):
        return EXIT_WARMUP_FAILED
    livez = result.get("livez", {})
    if livez and not livez.get("threshold_ok", True):
        return EXIT_LIVEZ_THRESHOLD
    return EXIT_OK if result.get("ok") else EXIT_CRASH


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = asyncio.run(run_harness(args))
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return EXIT_SIGINT
    except Exception as exc:  # noqa: BLE001 - top-level guard
        print(f"Harness crashed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_CRASH

    # Output routing: when --json is set, JSON on stdout (machine-readable)
    # and human summary on stderr (so operators still see what happened
    # without polluting stdout). Without --json, human summary on stdout.
    if args.json:
        print(format_summary(result), file=sys.stderr)
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(format_summary(result))

    return _classify_exit(result)


if __name__ == "__main__":
    sys.exit(main())
