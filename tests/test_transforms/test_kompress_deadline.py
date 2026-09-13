"""Phase 1 (#1171): kompress cooperative chunk-boundary deadline.

Kompress ONNX inference is O(tokens) and non-preemptible once the request's
asyncio timeout fires, so one large block can run for minutes holding a worker
(the leak -> executor-saturation -> queue-timeout cascade). compress() checks a
wall-clock budget at each chunk boundary and, when over, keeps the unprocessed
tail verbatim and returns -- a partial compression that returns now beats a full
one that leaks.
"""

from __future__ import annotations

import hashlib

import pytest

from headroom.transforms import kompress_compressor as kc


def test_compress_bails_at_deadline_keeping_tail_verbatim(monkeypatch):
    # Fake clock: the pre-loop stamp reads 0s, the first loop-top check reads
    # 999s elapsed -> deadline trips on chunk 0 before any model/tokenizer use.
    clock = iter([0.0] + [999.0] * 50)
    monkeypatch.setattr(kc.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(kc, "_load_kompress", lambda *a, **k: (object(), object(), "onnx"))
    monkeypatch.setenv("HEADROOM_COMPRESSION_DEADLINE_MS", "20000")

    comp = kc.KompressCompressor(kc.KompressConfig(min_input_words=10))
    monkeypatch.setattr(comp, "_should_batch_single_content", lambda *a, **k: False)

    content = " ".join(f"w{i}" for i in range(1000))
    result = comp.compress(content)

    # Deadline tripped on the first chunk -> nothing dropped, tail kept verbatim.
    assert result.compressed_tokens == 1000
    assert result.compressed.split() == content.split()


@pytest.mark.parametrize(
    ("n_words", "net_saving"),
    [(200, True), (20, False)],
    ids=["net-saving", "no-net-saving"],
)
def test_compress_partial_run_keeps_processed_head_plus_verbatim_tail(
    monkeypatch, n_words, net_saving
):
    # real partial case: chunk 0 processes (gets compressed), chunk 1 trips the
    # deadline (kept verbatim). Output must be compressed-head + verbatim-tail.
    # Clock: call 1 = t_deadline (0); calls 2-4 chunk-0's check + inference
    # reads (under budget); call 5+ chunk-1's check -> trips.
    # Robust clock: jump past the deadline only AFTER chunk 0 processed
    # (tracked via the model mock), so adding perf_counter calls inside the
    # chunk body -- e.g. sub-stage timing -- can't shift when the deadline trips.
    #
    # Two sizes: at 200 words the marked partial result is smaller than the
    # original and ships; at 20 words (150 -> 300-odd tokens either way, plus a
    # ~43-token marker) the CCR gate finds no net saving and passes the whole
    # payload through, which is the other half of the contract.
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

    comp = kc.KompressCompressor(kc.KompressConfig(min_input_words=10))
    comp.config.chunk_words = n_words // 2  # two chunks
    monkeypatch.setattr(comp, "_should_batch_single_content", lambda *a, **k: False)
    monkeypatch.setattr(
        comp,
        "_store_in_ccr",
        lambda source, *a, **k: hashlib.sha256(source.encode()).hexdigest()[:24],
    )

    words = [f"w{i}" for i in range(n_words)]
    result = comp.compress(" ".join(words))
    if not net_saving:
        assert result.compressed == " ".join(words)
        assert result.compression_ratio == 1.0
        return

    out = result.compressed.split()
    half = n_words // 2
    assert result.cache_key is not None and "Retrieve more" in result.compressed
    # chunk 0 processed: its first half kept, its second half dropped
    assert "w0" in out and f"w{half // 2 - 1}" in out
    assert f"w{half // 2}" not in out and f"w{half - 1}" not in out
    # chunk 1 tripped the deadline -> its words kept verbatim (all present)
    for i in range(half, n_words):
        assert f"w{i}" in out
