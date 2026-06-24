"""Record SmartCrusher parity fixtures.

Standalone recorder for `SmartCrusher.crush(content, query, bias)`. The
generic `recorder.py` only captures one positional arg; this script
captures all three so the Rust comparator gets the same inputs.

Fixture schema (consumed by `SmartCrusherComparator` in
`crates/headroom-parity/src/lib.rs`):

```
{
  "transform": "smart_crusher",
  "input":  { "content": "<JSON string>", "query": "<str>", "bias": 1.0 },
  "config": { ...SmartCrusherConfig fields... },
  "output": { "compressed": "...", "original": "...",
              "was_modified": <bool>, "strategy": "..." },
  "recorded_at":  "<iso>",
  "input_sha256": "<hex>"
}
```

Initial fixture suite focuses on empty-query paths so embedding
nondeterminism between Python `onnxruntime` and Rust `ort` does not
factor in. With `query=""`, both BM25 and embedding scorers short-
circuit to 0.0, no items get pinned by relevance, and the output is a
deterministic function of the input.

Run from repo root:
    python tests/parity/record_smart_crusher.py
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FIXTURES_DIR = _REPO_ROOT / "tests" / "parity" / "fixtures" / "smart_crusher"


def _digest(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _record(
    label: str,
    content: str,
    query: str,
    bias: float,
    config: SmartCrusherConfig | None = None,
) -> Path:
    cfg = config or SmartCrusherConfig()
    crusher = SmartCrusher(config=cfg)
    result = crusher.crush(content, query=query, bias=bias)

    payload_input = {"content": content, "query": query, "bias": bias}
    payload_config = asdict(cfg)
    payload_output = {
        "compressed": result.compressed,
        "original": result.original,
        "was_modified": result.was_modified,
        "strategy": result.strategy,
    }

    digest_source = {
        "transform": "smart_crusher",
        "label": label,
        "input": payload_input,
        "config": payload_config,
    }
    digest = _digest(digest_source)

    fixture = {
        "transform": "smart_crusher",
        "label": label,
        "input": payload_input,
        "config": payload_config,
        "output": payload_output,
        "recorded_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "input_sha256": digest,
    }

    _FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    target = _FIXTURES_DIR / f"{label}_{digest[:12]}.json"
    target.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    return target


def _scenarios() -> list[tuple[str, str, str, float]]:
    """Initial parity scenarios. All use `query=""` to keep embeddings
    out of the comparison until we resolve the ~0.0002 numeric drift
    between Python `onnxruntime` and Rust `ort`."""
    out: list[tuple[str, str, str, float]] = []

    # 1. Non-JSON content → passthrough. The crusher returns the input
    # unchanged; trivially byte-equal.
    out.append(("non_json_passthrough", "this is not json at all", "", 1.0))

    # 2. JSON object with no array fields long enough to crush.
    out.append(
        (
            "small_object_passthrough",
            json.dumps({"a": 1, "b": 2, "c": "hello"}),
            "",
            1.0,
        )
    )

    # 3. Short array (below min_items_to_analyze=5) → passthrough.
    out.append(
        (
            "short_array_passthrough",
            json.dumps([1, 2, 3]),
            "",
            1.0,
        )
    )

    # 4. Dict array with 30 items, varied integer status field.
    # Exercises crush_array's adaptive_k → smart_sample / top_n path.
    items_30_dict = [
        {"id": i, "status": "ok" if i % 5 != 0 else "error", "msg": f"line {i}"} for i in range(30)
    ]
    out.append(("dict_array_30", json.dumps(items_30_dict), "", 1.0))

    # 5. Pure string array of 25 items.
    string_arr_25 = [f"event {i}: something happened at index {i}" for i in range(25)]
    out.append(("string_array_25", json.dumps(string_arr_25), "", 1.0))

    # 6. Pure number array of 40 items with a clear change point.
    number_arr_40 = [10 + (i % 3) for i in range(20)] + [50 + i for i in range(20)]
    out.append(("number_array_40_changepoint", json.dumps(number_arr_40), "", 1.0))

    # 7. Mixed array (strings + ints).
    mixed_arr = ["start"] + list(range(20)) + ["middle"] + ["end"] * 5
    out.append(("mixed_array", json.dumps(mixed_arr), "", 1.0))

    # 8. Nested: top-level dict whose `events` field is a long dict array.
    nested = {
        "request_id": "req-1",
        "events": [{"step": i, "kind": "trace", "msg": f"e{i}"} for i in range(20)],
    }
    out.append(("nested_object_with_array", json.dumps(nested), "", 1.0))

    # 9. Bias > 1 (keep more) on the 30-dict case.
    out.append(("dict_array_30_bias_high", json.dumps(items_30_dict), "", 1.5))

    # 10. Bias < 1 (keep fewer) on the 30-dict case.
    out.append(("dict_array_30_bias_low", json.dumps(items_30_dict), "", 0.7))

    # 11. Unicode payload — exercises the `ensure_ascii=False` path in
    # Python's safe_json_dumps. Rust's python_safe_json_dumps must emit
    # raw UTF-8 bytes here, not `\uXXXX` escapes.
    unicode_items = [{"id": i, "msg": f"hello 中文 русский {i}", "tag": "тест"} for i in range(20)]
    out.append(("unicode_dict_array", json.dumps(unicode_items), "", 1.0))

    # 12. Larger dict array (100 items) with a strong sequential `id`
    # field — exercises top_n strategy via field stats.
    big_seq = [
        {"id": i, "level": "info" if i % 7 != 0 else "warn", "message": f"seq {i}"}
        for i in range(100)
    ]
    out.append(("dict_array_100_sequential", json.dumps(big_seq), "", 1.0))

    # 13. Time-series-like payload: monotonic timestamp + float metric.
    ts = [{"ts": 1000 + i, "metric": float(i * 1.5), "host": f"host-{i % 3}"} for i in range(50)]
    out.append(("time_series_50", json.dumps(ts), "", 1.0))

    # 14. Many duplicate items — exercises dedup_identical_items.
    dups = [{"event": "heartbeat", "ok": True} for _ in range(40)]
    out.append(("duplicate_dicts_40", json.dumps(dups), "", 1.0))

    # 15. Empty array — boundary case, must round-trip cleanly.
    out.append(("empty_array", json.dumps([]), "", 1.0))

    # 16. Array of nulls and bools — non-crushable mixed type.
    out.append(
        (
            "nulls_and_bools",
            json.dumps([None, True, False, None, True, False, None]),
            "",
            1.0,
        )
    )

    # 17. Deeply nested structure: 3-level depth with arrays at each
    # level. Exercises process_value's recursion.
    deep = {"a": {"b": {"events": [{"i": i, "kind": "deep", "v": f"x{i}"} for i in range(15)]}}}
    out.append(("nested_3deep_with_array", json.dumps(deep), "", 1.0))

    return out


def main() -> int:
    written: list[Path] = []
    for label, content, query, bias in _scenarios():
        path = _record(label, content, query, bias)
        written.append(path)
        print(f"  + {path.relative_to(_REPO_ROOT)}")
    print(f"wrote {len(written)} fixture(s) → {_FIXTURES_DIR.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
