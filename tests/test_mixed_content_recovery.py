import json
import re

import pytest

from headroom.cache import compression_store
from headroom.cache.compression_store import CompressionStore
from headroom.transforms.content_router import ContentRouter


@pytest.mark.parametrize("storage_fails", [False, True])
def test_mixed_json_compression_preserves_full_recovery(monkeypatch, storage_fails):
    rows = [
        f"[{i}:00] Speaker: We reviewed the layout and discussed the report. The team needs time to compare the options."
        for i in range(180)
    ]
    rows[17] = (
        "[17:00] Decision: The launch is NOT approved. Reference ZX-9031. The budget is AUD 137.42, not AUD 173.42."
    )
    original = json.dumps({"messages": rows})
    store = CompressionStore()
    if storage_fails:

        def fail(*args, **kwargs):
            raise OSError("Storage unavailable")

        monkeypatch.setattr(store, "store", fail)
    monkeypatch.setattr(compression_store, "get_compression_store", lambda: store)
    result = ContentRouter().compress(original)
    if storage_fails:
        assert result.compressed == original
    else:
        hashes = re.findall(r"Retrieve original: hash=([a-f0-9]+)", result.compressed)
        assert hashes, "Mixed output has no complete-original recovery reference"
        assert store.get_stats()["total_retrievals"] == 0
        assert store.retrieve(hashes[-1]).original_content == original
