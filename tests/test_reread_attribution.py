"""Over-compression attribution for reread waste (issue #899).

``parse_messages(compressed_messages=...)`` splits the existing ``reread``
signal: repeats whose first serve was replaced by a CCR retrieval marker in
the transformed output count into ``reread_compressed_tokens`` — re-reads
attributable to Headroom rather than agent behavior. Lossless reshaping
(no marker) and intact first serves are deliberately not attributed.
"""

from __future__ import annotations

import json

import pytest

from headroom import OpenAIProvider, Tokenizer
from headroom.config import HeadroomConfig, WasteSignals
from headroom.parser import parse_messages
from headroom.transforms.pipeline import TransformPipeline

_provider = OpenAIProvider()


@pytest.fixture
def tokenizer() -> Tokenizer:
    return Tokenizer(_provider.get_token_counter("gpt-4o"), "gpt-4o")


def _uniform_rows(rows: int = 200) -> str:
    return json.dumps(
        [{"id": i, "name": f"item_{i}", "status": "ok", "score": i * 3.14} for i in range(rows)]
    )


_MARKER = "[200 items compressed to 12. Retrieve more: hash=abc123def4567890abcdef12]"


def _conversation(first_serve: str, repeat: str) -> list[dict]:
    """First serve at index 1, repeat at index 7 (gap 6 > REREAD_ADJACENT_GAP)."""
    filler = [{"role": "user", "content": f"step {i}"} for i in range(5)]
    return [
        {"role": "user", "content": "read the data"},
        {"role": "tool", "content": first_serve},
        *filler,
        {"role": "tool", "content": repeat},
    ]


class TestRereadAttribution:
    def test_markerized_first_serve_attributes(self, tokenizer):
        content = _uniform_rows()
        messages = _conversation(content, content)
        compressed = [dict(m) for m in messages]
        compressed[1] = {"role": "tool", "content": _MARKER}

        _, _, waste = parse_messages(messages, tokenizer, compressed_messages=compressed)
        assert waste.reread_tokens > 0
        assert waste.reread_compressed_tokens == waste.reread_tokens

    def test_intact_first_serve_not_attributed(self, tokenizer):
        content = _uniform_rows()
        messages = _conversation(content, content)

        _, _, waste = parse_messages(
            messages, tokenizer, compressed_messages=[dict(m) for m in messages]
        )
        assert waste.reread_tokens > 0
        assert waste.reread_compressed_tokens == 0

    def test_lossless_reshape_without_marker_not_attributed(self, tokenizer):
        content = _uniform_rows()
        messages = _conversation(content, content)
        compressed = [dict(m) for m in messages]
        # CSV-style compaction: content reshaped, all data retained, no marker.
        compressed[1] = {"role": "tool", "content": "id,name,status,score\n0,item_0,ok,0.0"}

        _, _, waste = parse_messages(messages, tokenizer, compressed_messages=compressed)
        assert waste.reread_tokens > 0
        assert waste.reread_compressed_tokens == 0

    def test_marker_with_original_still_present_not_attributed(self, tokenizer):
        # Marker appended but full original retained (e.g. partial compression
        # of a different span in the same message) — model saw everything.
        content = _uniform_rows()
        messages = _conversation(content, content)
        compressed = [dict(m) for m in messages]
        compressed[1] = {"role": "tool", "content": content + "\n" + _MARKER}

        _, _, waste = parse_messages(messages, tokenizer, compressed_messages=compressed)
        assert waste.reread_compressed_tokens == 0

    def test_message_count_mismatch_skips_attribution(self, tokenizer):
        content = _uniform_rows()
        messages = _conversation(content, content)
        compressed = [dict(m) for m in messages]
        compressed[1] = {"role": "tool", "content": _MARKER}
        compressed.pop(0)

        _, _, waste = parse_messages(messages, tokenizer, compressed_messages=compressed)
        assert waste.reread_tokens > 0
        assert waste.reread_compressed_tokens == 0

    def test_default_no_compressed_messages(self, tokenizer):
        content = _uniform_rows()
        _, _, waste = parse_messages(_conversation(content, content), tokenizer)
        assert waste.reread_tokens > 0
        assert waste.reread_compressed_tokens == 0

    def test_polling_repeats_not_attributed(self, tokenizer):
        # Adjacent repeats (gap <= REREAD_ADJACENT_GAP) are polling, not
        # rereads — attribution never runs for groups with no counted waste.
        content = _uniform_rows()
        messages = [
            {"role": "tool", "content": content},
            {"role": "user", "content": "poll"},
            {"role": "tool", "content": content},
        ]
        compressed = [dict(m) for m in messages]
        compressed[0] = {"role": "tool", "content": _MARKER}

        _, _, waste = parse_messages(messages, tokenizer, compressed_messages=compressed)
        assert waste.reread_tokens == 0
        assert waste.reread_compressed_tokens == 0

    def test_ccr_inline_marker_form_attributes(self, tokenizer):
        content = _uniform_rows()
        messages = _conversation(content, content)
        compressed = [dict(m) for m in messages]
        compressed[1] = {"role": "tool", "content": "<<ccr:a703e0aaa98f,string,1.1KB>>"}

        _, _, waste = parse_messages(messages, tokenizer, compressed_messages=compressed)
        assert waste.reread_compressed_tokens == waste.reread_tokens > 0


class TestWasteSignalsContract:
    def test_to_dict_exports_reread_compressed(self):
        ws = WasteSignals(reread_tokens=100, reread_compressed_tokens=60)
        d = ws.to_dict()
        assert d["reread"] == 100
        assert d["reread_compressed"] == 60

    def test_total_excludes_reread_compressed(self):
        # reread_compressed is a subset of reread — adding it to total()
        # would double count.
        ws = WasteSignals(reread_tokens=100, reread_compressed_tokens=60)
        assert ws.total() == 100


class TestPipelineAttribution:
    def test_pipeline_passes_compressed_messages(self, tokenizer):
        # End-to-end through TransformPipeline.apply: a large duplicated tool
        # result far from its first serve produces reread waste, and
        # reread_compressed is consistent (either 0 or the full group —
        # never more than reread).
        content = _uniform_rows(400)
        filler = [{"role": "user", "content": f"working on step {i}"} for i in range(5)]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "tool", "content": content},
            *filler,
            {"role": "tool", "content": content},
            {"role": "user", "content": "continue"},
        ]
        result = TransformPipeline(HeadroomConfig()).apply(
            [dict(m) for m in messages], model="gpt-4o", model_limit=128000
        )
        assert result.waste_signals is not None
        assert result.waste_signals.reread_tokens > 0
        assert (
            0
            <= result.waste_signals.reread_compressed_tokens
            <= (result.waste_signals.reread_tokens)
        )
