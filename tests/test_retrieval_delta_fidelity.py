import json

import pytest

from headroom.cache.compression_cache import CompressionCache
from headroom.transforms.compression_units import CompressionUnit, compress_unit_with_router
from headroom.transforms.content_router import ContentRouter


@pytest.mark.parametrize("wrapped", [False, True])
def test_delta_retrieval_stays_exact_without_call_history(wrapped):
    original = json.dumps(
        {
            "messages": [
                f"[{i}:00] Speaker: We reviewed the layout and discussed the report. The team needs time to compare the options."
                for i in range(180)
            ]
        }
    )
    text = json.dumps(
        {"hash": "abcdef123456", "original_content": original, "retrieval_count": 1}, indent=2
    )
    if wrapped:
        text = json.dumps({"content": [{"type": "text", "text": text}]})
    router = ContentRouter()
    assert router.compress(text).compressed == text
    unit = CompressionUnit(
        text=text,
        provider="openai",
        endpoint="responses",
        role="tool",
        item_type="function_call_output",
        min_bytes=0,
    )

    class Counter:
        def count_text(self, s):
            return len(s)

    assert compress_unit_with_router(unit, router=router, tokenizer=Counter()).compressed == text
    cache = CompressionCache()
    cache.store_compressed(cache.content_hash(text), "Stale summary", 1)
    message = {"role": "tool", "tool_call_id": "delta-no-history", "content": text}
    assert cache.apply_cached([message]) == [message]
