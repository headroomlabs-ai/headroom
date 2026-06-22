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


def test_compress_partial_run_keeps_processed_head_plus_verbatim_tail(monkeypatch):
    # The real partial case: chunk 0 processes (gets compressed), chunk 1 trips
    # the deadline (kept verbatim). Output must be compressed-head + verbatim-tail.
    # Clock: call1=t_deadline(0); calls 2-4 are chunk-0's check+inference reads
    # (under budget); call 5+ is chunk-1's check -> trips.
    # Robust clock: jump past the deadline only AFTER chunk 0 is processed
    # (tracked via the model mock), so adding perf_counter calls inside the chunk
    # body -- e.g. sub-stage timing -- can't shift when the deadline trips.
    state = {"chunks_done": 0}

    def fake_clock():
        return 999.0 if state["chunks_done"] >= 1 else 0.0

    monkeypatch.setattr(kc.time, "perf_counter", fake_clock)

    class _Enc(dict):
        def word_ids(self, batch_index=0):
            return self["_word_ids"]

    class _Tok:
        def __call__(self, chunk_words, **kw):
            n = len(chunk_words)
            return _Enc(input_ids=[[0] * n], attention_mask=[[1] * n], _word_ids=list(range(n)))

    class _Model:
        def get_keep_mask(self, input_ids, attention_mask):
            n = len(input_ids[0])
            mask = [[i < n // 2 for i in range(n)]]  # keep first half of the chunk
            state["chunks_done"] += 1  # after chunk 0, the clock trips the deadline
            return mask

    monkeypatch.setattr(kc, "_load_kompress", lambda *a, **k: (_Model(), _Tok(), "onnx"))
    monkeypatch.setattr(kc, "_model_device_type", lambda *a, **k: "cpu")
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "20000")

    comp = kc.KompressCompressor()
    comp.config.chunk_words = 10  # 20 words -> 2 chunks
    monkeypatch.setattr(comp, "_should_batch_single_content", lambda *a, **k: False)

    words = [f"w{i}" for i in range(20)]
    out = comp.compress(" ".join(words)).compressed.split()

    # chunk 0 processed: first half kept (w0..w4), second half dropped (w5..w9)
    assert "w0" in out and "w4" in out
    assert "w5" not in out and "w9" not in out
    # chunk 1 tripped the deadline -> its words kept verbatim (w10..w19 all present)
    for i in range(10, 20):
        assert f"w{i}" in out
