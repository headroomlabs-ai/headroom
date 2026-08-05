"""No-CCR lossy: with markers OFF, unmarked lossy output is accepted (not skipped).

The reversibility guard skips lossy-unmarked tool output to keep it recoverable —
but only makes sense when retrieval markers are ON. In no-CCR mode
(ccr_inject_marker=False) recovery is deliberately disabled, so the unmarked lossy
result IS the intended output. This tests both directions without needing the
ModernBERT model (self.compress is mocked to a lossy result).
"""

from types import SimpleNamespace

from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
)


def _run(marker_on):
    orig = "some_identifier = compute_value(x)\n" * 60  # long, lossy-eligible
    fake = SimpleNamespace(
        compressed="<<lossy summary, marker-free>>",
        compression_ratio=0.3,  # < min_ratio → enters accept branch
        strategy_used=CompressionStrategy.KOMPRESS,  # lossy + unmarked
        strategy_chain=["kompress"],
    )
    r = ContentRouter(ContentRouterConfig(ccr_inject_marker=marker_on))
    r.compress = lambda content, context=None, bias=1.0: fake  # type: ignore
    tr, rc = [], {}
    out, was = r._compress_block_content(
        orig,
        hash((orig, marker_on)),
        "",
        1.0,
        1.0,
        None,
        tr,
        rc,
        [],
        "tool_result",
        "tool",
        enforce_reversibility=True,
    )
    return was, rc


def test_markers_on_skips_unmarked_lossy():
    was, rc = _run(marker_on=True)
    assert was is False  # guard skips: unrecoverable
    assert rc.get("lossy_unrecoverable_skipped", 0) == 1


def test_no_ccr_mode_accepts_unmarked_lossy():
    was, rc = _run(marker_on=False)
    assert was is True  # accepted: no-CCR mode wants unmarked lossy
    assert rc.get("lossy_unrecoverable_skipped", 0) == 0
