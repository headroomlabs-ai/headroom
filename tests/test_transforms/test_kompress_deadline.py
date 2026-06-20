"""Phase 1 (#1171): kompress cooperative chunk-boundary deadline.

Kompress ONNX inference is O(tokens) and non-preemptible once the request's
asyncio timeout fires, so one large block can run for minutes holding a worker
(the leak -> executor-saturation -> queue-timeout cascade). compress() checks a
wall-clock budget at each chunk boundary and, when over, keeps the unprocessed
tail verbatim and returns -- a partial compression that returns now beats a full
one that leaks.
"""

from __future__ import annotations

from headroom.transforms import kompress_compressor as kc


def test_compress_bails_at_deadline_keeping_tail_verbatim(monkeypatch):
    # Fake clock: the pre-loop stamp reads 0s, the first loop-top check reads
    # 999s elapsed -> deadline trips on chunk 0 before any model/tokenizer use.
    clock = iter([0.0] + [999.0] * 50)
    monkeypatch.setattr(kc.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(kc, "_load_kompress", lambda *a, **k: (object(), object(), "onnx"))
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "20000")

    comp = kc.KompressCompressor()
    monkeypatch.setattr(comp, "_should_batch_single_content", lambda *a, **k: False)

    content = " ".join(f"w{i}" for i in range(1000))
    result = comp.compress(content)

    # Deadline tripped on the first chunk -> nothing dropped, tail kept verbatim.
    assert result.compressed_tokens == 1000
    assert result.compressed.split() == content.split()
