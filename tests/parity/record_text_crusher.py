"""Record TextCrusher parity fixtures (Phase 2, #1171).

Locks the Rust core's ``compress`` output for a fixed set of deterministic
scenarios so a future change to the Rust algorithm is caught as a regression.
The Python wrapper delegates to ``headroom._core.TextCrusher``, so these
fixtures are recorded from (and verified against) the native implementation.

Re-record after an intentional algorithm change:
    python tests/parity/record_text_crusher.py
"""

from __future__ import annotations

import hashlib
import json
import os

from headroom.transforms.text_crusher import TextCrusher

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "text_crusher")


def _prose(n: int) -> str:
    return " ".join(
        f"Sentence number {i} explains how distributed systems reconcile state across topic {i}."
        for i in range(n)
    )


def _redundant() -> str:
    dup = "The quick brown fox jumps over the very lazy dog every single morning."
    uniques = [f"A distinct fact about subsystem {i} is recorded plainly here." for i in range(8)]
    return "\n".join([dup] * 10 + uniques)


def _salient() -> str:
    return "\n".join(
        [
            "ERROR connection refused at host 10.0.0.42 after 3 retries.",
            "The authentication module validated tokens against auth.registry before forwarding.",
            "A traceback was logged with code 500 and request_id req-9182.",
            "Just some generic filler text without any specific identifiers here.",
            "More plain filler describing the overall behavior in vague terms.",
            "Warning: cache hit ratio dropped to 71 percent during the spike.",
            "Another unremarkable sentence with no salient tokens at all today.",
            "The pipeline.apply call returned 42 kept rows out of 1000 total.",
        ]
    )


# (label, content, context, target_ratio)
SCENARIOS: list[tuple[str, str, str, float | None]] = [
    ("plain_prose", _prose(30), "how do distributed systems reconcile state", 0.3),
    ("plain_prose_no_query", _prose(30), "", 0.5),
    ("redundant", _redundant(), "", 0.9),
    ("salient_heavy", _salient(), "authentication tokens errors", 0.4),
    ("short_passthrough", "one thing. two thing. three thing.", "", None),
    (
        "unicode",
        " ".join(f"句子 {i} 描述了系统在主题 {i} 上的行为细节。" for i in range(12)),
        "系统",
        0.4,
    ),
]


def record() -> None:
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    tc = TextCrusher()
    for label, content, context, ratio in SCENARIOS:
        r = tc.compress(content, context, ratio)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        fixture = {
            "transform": "text_crusher",
            "label": label,
            "input": {"content": content, "context": context, "target_ratio": ratio},
            "output": {
                "compressed": r.compressed,
                "original_tokens": r.original_tokens,
                "compressed_tokens": r.compressed_tokens,
                "compression_ratio": r.compression_ratio,
                "kept_segments": r.kept_segments,
                "total_segments": r.total_segments,
            },
            "input_sha256": digest,
        }
        path = os.path.join(FIXTURE_DIR, f"{label}_{digest[:12]}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(fixture, fh, indent=2, ensure_ascii=False)
        print(f"wrote {path}")


if __name__ == "__main__":
    record()
