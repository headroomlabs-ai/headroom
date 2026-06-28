#!/usr/bin/env python
"""Deep diagnostic for Kompress v2 (kompress-v2-base) inside Headroom.

Answers three questions:
  1. What model/backend is actually loaded, and what does it report per-compress?
  2. How does the ratio behave on realistic prose at different target_ratios
     (vs the repetitive-text case that looked weak)?
  3. WHY did Kompress never fire in the pilot dry-run? (routing: JSON -> SmartCrusher,
     prose -> Kompress) — demonstrated end-to-end through compress().
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from headroom import CompressConfig, compress  # noqa: E402
from headroom.transforms.kompress_compressor import KompressCompressor  # noqa: E402

# --- realistic, NON-repetitive technical prose (varied entropy) -------------
DOC = """\
Postmortem: Elevated p99 latency on the checkout service (2025-12-02)

Summary. Between 14:10 and 15:42 UTC the checkout service served p99 latencies
above nine seconds, well over the 800ms SLO. Roughly 4.3% of checkout attempts
timed out client-side. Revenue impact is estimated at forty thousand dollars in
abandoned carts. No data was lost and no customer records were exposed.

Timeline. At 14:08 a routine deploy rolled out connection-pool tuning intended to
raise throughput during the holiday surge. The change lowered the idle-connection
reaper interval from sixty seconds to five, and simultaneously raised the maximum
pool size from two hundred to eight hundred. Within ninety seconds the database
reported a sharp climb in active connections and the checkout service began
queueing requests behind a saturated pool.

Root cause. The reaper interval and the max-pool change interacted badly. The
aggressive reaper closed warm connections that were about to be reused, forcing
the service to renegotiate TLS on nearly every request. Each renegotiation added
roughly two hundred milliseconds, and under load the handshakes serialized on a
single lock in the driver. The larger pool ceiling masked the problem at first by
absorbing the backlog, then amplified it when the database hit its own connection
limit and started refusing new sessions.

Detection. The first alert fired at 14:19 when synthetic checkout probes breached
the latency SLO for three consecutive minutes. On-call acknowledged at 14:22 and
opened an incident channel. Dashboards showed normal CPU and memory but a pegged
connection-wait metric, which pointed away from the application code and toward
the pool configuration.

Resolution. At 15:30 the team reverted the deploy. Connection counts normalized
within four minutes as warm connections were re-established and TLS sessions were
reused. Latency returned to baseline by 15:42. A follow-up patch restores the
sixty-second reaper interval and caps the pool at three hundred, a value validated
in load testing.

Action items. First, add a guardrail test that fails CI if the reaper interval is
below thirty seconds. Second, expose the connection-wait metric on the primary
checkout dashboard so it is visible during the first minute of any incident.
Third, document the TLS-renegotiation cost in the driver runbook so future tuning
accounts for it. Fourth, stage pool changes behind a feature flag with automatic
rollback tied to the latency SLO.
"""

REPETITIVE = (
    "The payment service returned elevated latencies and engineers investigated. "
) * 40


def section(title: str) -> None:
    print("\n" + "=" * 76 + f"\n{title}\n" + "=" * 76)


def show_result(label: str, r) -> None:  # noqa: ANN001
    print(
        f"  {label:<28} model={getattr(r, 'model_used', '?'):<22} "
        f"orig_tok={r.original_tokens:>6}  comp_tok={r.compressed_tokens:>6}  "
        f"saved={r.savings_percentage:>5.1f}%  ratio={r.compression_ratio:.3f}"
    )


def main() -> None:
    k = KompressCompressor()
    backend = k.preload(allow_download=True)

    section("1. What Kompress v2 actually loaded")
    print(f"  preload backend : {backend}")
    cfg = getattr(k, "config", None)
    if cfg is not None:
        for attr in ("model_id", "onnx_model", "target_ratio", "min_tokens", "device"):
            if hasattr(cfg, attr):
                print(f"  config.{attr:<14}: {getattr(cfg, attr)}")
    # which onnx files exist locally
    onnx_dir = _REPO / "onnx"
    if onnx_dir.exists():
        print("  local onnx files:", ", ".join(p.name for p in sorted(onnx_dir.glob("*.onnx"))))

    section("2. Kompress v2 on realistic prose at different target_ratios")
    print(f"  [varied technical prose, {len(DOC)} chars]")
    for tr in (None, 0.5, 0.3, 0.15):
        r = k.compress(DOC, context="what was the root cause of the latency incident?", target_ratio=tr)
        show_result(f"target_ratio={tr}", r)
    print(f"\n  [highly repetitive prose, {len(REPETITIVE)} chars] (why the first test looked weak)")
    for tr in (None, 0.3):
        r = k.compress(REPETITIVE, context="root cause?", target_ratio=tr)
        show_result(f"target_ratio={tr}", r)
    print("\n  NOTE: repetitive low-entropy text has little to drop; varied prose compresses far more.")

    section("3. WHY Kompress didn't fire in the pilot: routing (JSON vs prose)")
    import json

    records = [
        {"id": f"r{i}", "score": 0.9 - i * 0.01, "status": "ok", "value": i * 7}
        for i in range(120)
    ]
    json_blob = json.dumps(records, indent=2)
    prose_blob = DOC * 3  # a big retrieved document, as text

    def route(content: str, label: str) -> None:
        msgs = [
            {"role": "user", "content": "Analyze this."},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "ok"},
                    {"type": "tool_use", "id": "toolu_demo01", "name": "fetch", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_demo01", "content": content}],
            },
        ]
        res = compress(
            msgs,
            model="claude-haiku-4-5-20251001",
            config=CompressConfig(compress_user_messages=True, protect_recent=0),
        )
        print(
            f"  {label:<22} {res.tokens_before:>7,} -> {res.tokens_after:>7,} tok "
            f"({100 * res.compression_ratio:>4.1f}% saved)  transforms={res.transforms_applied}"
        )

    route(json_blob, "JSON array (120 recs)")
    route(prose_blob, "prose document")
    print("\n  => JSON routes to SmartCrusher; prose routes to Kompress. The pilot's payloads")
    print("     were all JSON, so Kompress never engaged. Adding prose tool_results fixes it.")


if __name__ == "__main__":
    main()
