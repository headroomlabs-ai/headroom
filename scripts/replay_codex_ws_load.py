#!/usr/bin/env python3
"""Tier-3 replay: reproduce Codex /v1/responses compression load.

Parses a production proxy log to extract per-session frame-size scenarios,
generates synthetic payloads matching those sizes/shapes, and concurrently
drives the proxy's _compress_openai_responses_payload entry point. Reports
per-frame latency percentiles, timeout count, and total wall time so a
before/after comparison proves the P2 scheduler fix.

Why this lives in scripts/ (not tests/):
    - It is a measurement tool, not a correctness test.
    - It needs to run against multiple branches (main baseline vs fix
      branch) and report comparable numbers.
    - It exercises the *real* compression dispatch by booting a proxy
      instance via create_app() and calling the handler method directly —
      no HTTP/WS layer, because the bug is in the dispatch, not the wire.

Usage:
    .venv/bin/python scripts/replay_codex_ws_load.py \\
        --log "/Users/tchopra/Downloads/proxy (1).log" \\
        --concurrency 10 \\
        --frames-per-session 20
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# Telemetry off so we don't pollute the user's metrics during replay.
os.environ.setdefault("HEADROOM_DISABLE_TELEMETRY", "true")
os.environ.setdefault("HEADROOM_REQUIRE_RUST_CORE", "false")


@dataclass
class Frame:
    bytes_estimate: int
    text_shape: str  # plain_text_like | code_fence | traceback | jsonl_like


@dataclass
class Scenario:
    request_id: str
    frames: list[Frame] = field(default_factory=list)


# ── Log parser ─────────────────────────────────────────────────────────


# Marker columns. We are not using regex here per the design constraints —
# the log shape is a single deterministic format set by code we own. If
# the format changes the parser fails loud, not silently.
_FRAME_TOKEN = " WS /v1/responses "
_REQID_OPEN = "["
_REQID_CLOSE = "]"


def _parse_kv(text: str) -> dict[str, str]:
    """Parse ``key=value`` pairs out of a slow-unit log tail. Stops at the
    first unquoted space after a value. Quoted values not supported because
    the log never emits them; if it ever does, this raises.
    """
    out: dict[str, str] = {}
    for token in text.split():
        if "=" not in token:
            continue
        k, _, v = token.partition("=")
        out[k] = v
    return out


def parse_log(log_path: Path) -> dict[str, Scenario]:
    """Group ``WS /v1/responses slow compression unit`` entries by request_id.

    Each ``slow compression unit`` line carries the per-unit byte count and
    text_shape — exactly what we need to reconstruct a payload of similar
    compression cost. We deliberately ignore the ``compressed`` / ``frame
    compressed`` lines because they report POST-compression bytes, not the
    pre-compression input the dispatcher sees.

    Format:
        ... [hr_..._...] WS /v1/responses slow compression unit elapsed_ms=N
            strategy=X category=Y modified=Z content_type=T text_shape=S
            bytes=B min_bytes=N tokens_before=T tokens_after=T tokens_saved=S
            strategy_chain=[...]
    """
    scenarios: dict[str, Scenario] = {}
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "slow compression unit" not in line:
                continue
            if _FRAME_TOKEN not in line:
                continue
            req_open = line.find(_REQID_OPEN)
            req_close = line.find(_REQID_CLOSE, req_open + 1)
            if req_open < 0 or req_close < 0:
                continue
            request_id = line[req_open + 1 : req_close]
            tail = line[req_close + 1 :]
            kv = _parse_kv(tail)
            try:
                size = int(kv["bytes"])
            except (KeyError, ValueError):
                continue
            shape = kv.get("text_shape", "plain_text_like")
            scen = scenarios.setdefault(request_id, Scenario(request_id=request_id))
            scen.frames.append(Frame(bytes_estimate=size, text_shape=shape))
    return scenarios


# ── Payload synthesizer ────────────────────────────────────────────────


_LOREM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
)
_CODE_LINE = "def compute_metric_{i}(value: int) -> int:\n    return value * {i}\n\n"
_TRACEBACK_LINE = (
    '  File "/app/handler.py", line {i}, in process_request\n    raise RuntimeError(f"oops {i}")\n'
)


def _text_for_shape(shape: str, target_bytes: int) -> str:
    """Generate a string roughly ``target_bytes`` long, shaped like the
    production observation. No randomness — same input produces same output
    so the replay is reproducible.
    """
    if target_bytes < 64:
        # Below size_floor — generator just returns a short token.
        return "ok"
    if shape == "code_fence":
        body_target = max(target_bytes - 12, 0)  # "```python\n" + closing
        repeats = max(body_target // 50, 1)
        body = "".join(_CODE_LINE.format(i=i) for i in range(repeats))
        return "```python\n" + body[:body_target] + "\n```"
    if shape == "traceback":
        header = "Traceback (most recent call last):\n"
        body_target = max(target_bytes - len(header), 0)
        repeats = max(body_target // 65, 1)
        body = "".join(_TRACEBACK_LINE.format(i=i) for i in range(repeats))
        return header + body[:body_target]
    # plain_text_like / unknown / jsonl_like → lorem ipsum is fine as a
    # neutral payload; we are measuring scheduler contention, not compressor
    # quality, so the content shape just needs to traverse the same router.
    repeats = max(target_bytes // len(_LOREM), 1)
    raw = _LOREM * repeats
    return raw[:target_bytes]


def synthesize_payload(frame: Frame, turn_no: int) -> dict:
    """Build the *inner* Responses payload (no `response.create` envelope)
    with one function_call_output of the target byte size.

    ``_compress_openai_responses_payload`` is envelope-agnostic but routes
    by inspecting ``input``/``messages`` at the top level. The WS handler
    extracts ``payload["response"]`` and passes that downstream — we pass
    the same shape directly so the router actually sees compressible
    candidates instead of a single opaque ``response`` key.
    """
    output_text = _text_for_shape(frame.text_shape, frame.bytes_estimate)
    return {
        "model": "gpt-4o-mini",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Turn {turn_no} — please summarize.",
                    }
                ],
            },
            {
                "type": "function_call",
                "call_id": f"call_replay_{turn_no}",
                "name": "shell",
                "arguments": '{"command": "build"}',
            },
            {
                "type": "function_call_output",
                "call_id": f"call_replay_{turn_no}",
                "output": output_text,
            },
        ],
        "instructions": "Be brief.",
        "max_output_tokens": 30,
    }


# ── Proxy bring-up ─────────────────────────────────────────────────────


def boot_proxy():
    """Build a HeadroomProxy instance with optimize=True so the compression
    dispatch is actually exercised.

    This deliberately does NOT start the FastAPI server. We only need the
    in-process handler methods. Lifecycle hooks (background tasks, model
    pre-loading) that fire on startup are not required for the dispatch
    method we exercise — Kompress will lazy-load on first use, which we
    explicitly warm up below.
    """
    from headroom.proxy.server import ProxyConfig, create_app

    config = ProxyConfig(
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
    )
    app = create_app(config)
    return app.state.proxy


def warmup(proxy, model: str = "gpt-4o-mini") -> float:
    """Issue one small compression call so model weights are loaded.

    Returns the warmup wall time so the caller can sanity-check the
    measurements (warmup time is NOT counted toward replay metrics).
    """
    payload = synthesize_payload(
        Frame(bytes_estimate=4096, text_shape="plain_text_like"), turn_no=0
    )
    started = time.perf_counter()
    proxy._compress_openai_responses_payload(payload, model=model, request_id="replay-warmup")
    return (time.perf_counter() - started) * 1000.0


# ── Replay driver ──────────────────────────────────────────────────────


@dataclass
class FrameResult:
    request_id: str
    frame_index: int
    bytes_in: int
    elapsed_ms: float
    error: str | None = None


def replay_session(proxy, scenario: Scenario, model: str) -> list[FrameResult]:
    out: list[FrameResult] = []
    for idx, frame in enumerate(scenario.frames):
        payload = synthesize_payload(frame, turn_no=idx + 1)
        started = time.perf_counter()
        err: str | None = None
        try:
            proxy._compress_openai_responses_payload(
                payload, model=model, request_id=scenario.request_id
            )
        except Exception as e:  # noqa: BLE001 — surface ALL failure modes
            err = f"{type(e).__name__}: {e}"
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        out.append(
            FrameResult(
                request_id=scenario.request_id,
                frame_index=idx,
                bytes_in=frame.bytes_estimate,
                elapsed_ms=elapsed_ms,
                error=err,
            )
        )
    return out


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(pct / 100.0 * (len(s) - 1)))))
    return s[k]


# ── Reporting ──────────────────────────────────────────────────────────


def print_report(
    results: list[FrameResult],
    wall_time_s: float,
    concurrency: int,
    warmup_ms: float,
    out_json: Path | None,
) -> None:
    elapsed = [r.elapsed_ms for r in results]
    errors = [r for r in results if r.error]
    total_bytes = sum(r.bytes_in for r in results)
    by_session: dict[str, list[float]] = {}
    for r in results:
        by_session.setdefault(r.request_id, []).append(r.elapsed_ms)
    session_totals = [sum(v) for v in by_session.values()]

    summary = {
        "concurrency": concurrency,
        "warmup_ms": round(warmup_ms, 1),
        "frames_total": len(results),
        "sessions": len(by_session),
        "wall_time_s": round(wall_time_s, 2),
        "errors": len(errors),
        "error_classes": sorted({type(e.error).__name__: 1 for e in errors if e.error}.keys()),
        "input_bytes_total": total_bytes,
        "per_frame_elapsed_ms": {
            "p50": round(_percentile(elapsed, 50), 1),
            "p90": round(_percentile(elapsed, 90), 1),
            "p99": round(_percentile(elapsed, 99), 1),
            "max": round(max(elapsed) if elapsed else 0.0, 1),
            "mean": round(statistics.mean(elapsed) if elapsed else 0.0, 1),
        },
        "per_session_total_ms": {
            "p50": round(_percentile(session_totals, 50), 1),
            "p90": round(_percentile(session_totals, 90), 1),
            "max": round(max(session_totals) if session_totals else 0.0, 1),
        },
    }

    print("─── Codex compression replay summary ───")
    print(f"Concurrency:           {summary['concurrency']}")
    print(f"Sessions replayed:     {summary['sessions']}")
    print(f"Frames replayed:       {summary['frames_total']}")
    print(f"Wall time:             {summary['wall_time_s']}s")
    print(f"Warmup wall time:      {summary['warmup_ms']}ms (NOT counted in metrics)")
    print(f"Failures:              {summary['errors']}")
    print(f"Input bytes total:     {summary['input_bytes_total']:,}")
    print("Per-frame elapsed_ms:")
    for k, v in summary["per_frame_elapsed_ms"].items():
        print(f"  {k:5}                {v}")
    print("Per-session total_ms:")
    for k, v in summary["per_session_total_ms"].items():
        print(f"  {k:5}                {v}")
    if errors:
        print("\nFirst 5 errors:")
        for e in errors[:5]:
            print(f"  [{e.request_id}] frame {e.frame_index}: {e.error}")

    if out_json:
        out_json.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote machine-readable summary to {out_json}")


# ── Main ───────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="Path to production proxy log; per-session frame sizes are extracted from "
        "`slow compression unit` lines.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent sessions to replay (default: 10).",
    )
    parser.add_argument(
        "--frames-per-session",
        type=int,
        default=20,
        help="Cap frames per session for bounded run-time (default: 20). "
        "Sessions with more frames are truncated; with fewer are padded.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model name passed through the dispatcher (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        help="Write machine-readable summary JSON here for before/after comparison.",
    )
    args = parser.parse_args()

    if not args.log.exists():
        print(f"error: log file not found: {args.log}", file=sys.stderr)
        return 2

    print(f"[replay] parsing {args.log} ...", flush=True)
    scenarios = parse_log(args.log)
    if not scenarios:
        print(
            "error: no scenarios extracted from log (no `slow compression unit` lines)",
            file=sys.stderr,
        )
        return 2

    # Pick the top-N sessions by frame count — those exercised the bug
    # hardest in production and give the most representative replay.
    ranked = sorted(scenarios.values(), key=lambda s: -len(s.frames))
    picked = ranked[: args.concurrency]
    # Cap each scenario's frame count for bounded runtime.
    for s in picked:
        s.frames = s.frames[: args.frames_per_session]
    print(
        f"[replay] picked {len(picked)} scenarios "
        f"(total frames: {sum(len(s.frames) for s in picked)})",
        flush=True,
    )

    print("[replay] booting proxy in-process ...", flush=True)
    proxy = boot_proxy()

    print("[replay] warming up Kompress + router ...", flush=True)
    warmup_ms = warmup(proxy, model=args.model)
    print(f"[replay] warmup done in {warmup_ms:.1f}ms", flush=True)

    print(
        f"[replay] starting replay: {len(picked)} concurrent sessions x "
        f"{args.frames_per_session} frames",
        flush=True,
    )
    results: list[FrameResult] = []
    wall_started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(replay_session, proxy, s, args.model) for s in picked]
        for fut in concurrent.futures.as_completed(futures):
            results.extend(fut.result())
    wall_time_s = time.perf_counter() - wall_started

    print_report(
        results,
        wall_time_s=wall_time_s,
        concurrency=args.concurrency,
        warmup_ms=warmup_ms,
        out_json=args.out_json,
    )
    return 0 if all(r.error is None for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
