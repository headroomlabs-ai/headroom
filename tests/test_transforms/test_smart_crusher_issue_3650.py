"""Regression tests for headroomlabs-ai/headroom#3650.

Scalar (string/number/mixed) arrays used to drop items silently: the
crusher returned only the retained items with no marker, so e.g. 120
slugs came back as 16 with no trace of the other 104. Dropped scalar
items now get a visible, retrievable CCR sentinel appended to the
array.
"""

from __future__ import annotations

import json
import re

import pytest

from headroom.transforms.content_router import ContentRouter, ContentRouterConfig
from headroom.transforms.smart_crusher import (
    is_ccr_sentinel,
    strip_ccr_sentinels,
)

_CCR_RE = re.compile(r"<<ccr:([0-9a-f]{12}) ")


def _issue_payload() -> str:
    # The exact shape from the issue report.
    return json.dumps({"slugs": [f"r{i}" for i in range(120)], "total": 120})


def test_scalar_sentinel_recognized_by_python_shim() -> None:
    """Pure-Python coverage: the new string sentinel is a CCR sentinel."""
    sentinel = "… 104 more items <<ccr:4f03caae30bb 104_items_offloaded>>"
    assert is_ccr_sentinel(sentinel)
    # The dict sentinel keeps working.
    assert is_ccr_sentinel({"_ccr_dropped": "<<ccr:abc123 5_rows_offloaded>>"})
    # Real data is untouched.
    assert not is_ccr_sentinel("r42")
    assert not is_ccr_sentinel({"slugs": ["r1"]})
    assert not is_ccr_sentinel(42)

    items = ["r0", sentinel, "r1", {"_ccr_dropped": "<<ccr:abc123 1_rows_offloaded>>"}]
    assert strip_ccr_sentinels(items) == ["r0", "r1"]
    assert strip_ccr_sentinels("not-a-list") == "not-a-list"


def test_issue_3650_router_emits_visible_retrievable_sentinel() -> None:
    """End-to-end: the issue's payload truncates visibly and retrievably."""
    pytest.importorskip("headroom._core")  # needs the native extension
    router = ContentRouter(ContentRouterConfig())
    result = router.compress(_issue_payload())
    compressed = result.compressed

    # 1. The truncation is visible: a sentinel names the drop count and
    #    carries a CCR marker. (Before the fix, 104 slugs vanished with
    #    no marker at all.)
    assert "more items" in compressed, compressed[:300]
    m = _CCR_RE.search(compressed)
    assert m, f"no CCR marker in compressed output: {compressed[:300]}"
    ccr_hash = m.group(1)

    # 2. The marker retrieves the FULL original array: the Python
    #    SmartCrusher mirrors Rust CCR markers into the process
    #    compression store.
    from headroom.cache.compression_store import get_compression_store

    entry = get_compression_store().retrieve(ccr_hash)
    assert entry is not None, "CCR marker hash must resolve in the compression store"
    original = json.loads(entry.original_content)
    assert len(original) == 120
    assert original[0] == "r0"
    assert original[119] == "r119"

    # 3. The sibling field is untouched and the output still parses.
    doc = json.loads(compressed)
    assert doc["total"] == 120
    assert isinstance(doc["slugs"], list)
    assert len(doc["slugs"]) < 120  # some were dropped…
    assert any(is_ccr_sentinel(x) for x in doc["slugs"])  # …but marked
