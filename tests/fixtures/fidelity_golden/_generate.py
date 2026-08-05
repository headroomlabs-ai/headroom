"""Deterministic generator for the fidelity-regression golden fixtures.

Run to (re)create ``cases.json``::

    python tests/fixtures/fidelity_golden/_generate.py

Each case is a JSON tool output (a list of records) plus the question it should
let a model answer and the evidence that must survive compression. Critical
``answer_evidence`` is always placed in an *error* or *anomaly* row, because
those are the rows SmartCrusher formally guarantees to retain
(see ``tests/test_quality_retention.py``) — so the hard ``recall == 1.0`` gate
holds on a healthy tree. ``supporting_facts`` are softer details that may live
in droppable rows; they feed the aggregate-recall baseline, not the hard gate.

No models, no network: content is built from fixed strings only.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "cases.json"


def _filler(n: int, msg: str) -> list[dict]:
    """n benign, near-identical INFO rows (droppable noise)."""
    return [{"seq": i, "level": "INFO", "status": "ok", "msg": f"{msg} {i}"} for i in range(n)]


def _cases() -> list[dict]:
    cases: list[dict] = []

    # 1. OOM crash buried in benign logs — error-keyword retention.
    rows = _filler(28, "heartbeat ping")
    rows.insert(
        17,
        {
            "seq": 17,
            "level": "ERROR",
            "status": "failed",
            "msg": "OOM killed worker 3 exceeded memory limit 8GiB",
        },
    )
    rows.insert(
        9, {"seq": 9, "level": "INFO", "status": "ok", "msg": "checkpoint saved at step 4500"}
    )
    cases.append(
        {
            "id": "logs_oom",
            "question": "Why did the job fail?",
            "content_type": "json_array",
            "compress": {"with_compaction": False, "max_items_after_crush": 10},
            "answer_evidence": ["OOM killed worker 3", "exceeded memory limit"],
            "supporting_facts": ["checkpoint saved at step 4500"],
            "content": rows,
        }
    )

    # 2. Payment exception — error-keyword retention.
    rows = _filler(30, "GET /healthz 200")
    rows.insert(
        21,
        {
            "seq": 21,
            "level": "ERROR",
            "status": "failed",
            "msg": "exception: NullPointerException at PaymentService.charge line 88",
        },
    )
    cases.append(
        {
            "id": "payment_exception",
            "question": "Which service threw an exception and where?",
            "content_type": "json_array",
            "compress": {"with_compaction": False, "max_items_after_crush": 8},
            "answer_evidence": ["NullPointerException", "PaymentService.charge line 88"],
            "supporting_facts": [],
            "content": rows,
        }
    )

    # 3. Latency anomaly — anomaly (>2 sigma) retention. Values cluster near 100ms,
    #    one row spikes to 99999ms.
    rows = [
        {"seq": i, "region": "us-west-2", "latency_ms": 95 + (i % 11), "status": "ok"}
        for i in range(30)
    ]
    rows[19] = {
        "seq": 19,
        "region": "us-east-1",
        "latency_ms": 99999,
        "status": "ok",
        "note": "latency spike us-east-1",
    }
    cases.append(
        {
            "id": "latency_anomaly",
            "question": "Which region had the latency spike, and how high?",
            "content_type": "json_array",
            "compress": {"with_compaction": False, "max_items_after_crush": 8},
            "answer_evidence": ["us-east-1", "99999"],
            "supporting_facts": [],
            "content": rows,
        }
    )

    # 4. CI test failure among many passes — "failed" keyword retention.
    rows = [
        {"seq": i, "test": f"test_module_{i}", "outcome": "passed", "duration_ms": 5 + i}
        for i in range(30)
    ]
    rows[12] = {
        "seq": 12,
        "test": "test_auth_token_refresh",
        "outcome": "failed",
        "duration_ms": 41,
        "error": "AssertionError: expected 200 got 401 in test_auth_token_refresh",
    }
    cases.append(
        {
            "id": "ci_test_failures",
            "question": "Which test failed and why?",
            "content_type": "json_array",
            "compress": {"with_compaction": False, "max_items_after_crush": 9},
            "answer_evidence": ["test_auth_token_refresh", "expected 200 got 401"],
            "supporting_facts": [],
            "content": rows,
        }
    )

    return cases


def main() -> None:
    OUT.write_text(json.dumps(_cases(), indent=2) + "\n")
    print(f"wrote {OUT} ({len(_cases())} cases)")


if __name__ == "__main__":
    main()
