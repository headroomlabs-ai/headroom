"""`header_safe_transforms` keeps the comma-joined transforms header splittable.

`x-headroom-transforms` is built as ``",".join(transforms_applied)``. The
enriched ``read_lifecycle:<state>:<path>`` and ``smart_crush:<n>:<names>`` tags
can contain commas, which would make that header ambiguous; the helper collapses
them back to their legacy counter shape so each header token stays comma-free.
"""

from __future__ import annotations

from headroom.proxy.cost import header_safe_transforms


def test_strips_smart_crush_tool_names():
    assert header_safe_transforms(["smart_crush:2:Bash,Grep"]) == ["smart_crush:2"]


def test_strips_read_lifecycle_path():
    assert header_safe_transforms(["read_lifecycle:stale:/src/App.tsx"]) == ["read_lifecycle:stale"]


def test_strips_read_lifecycle_path_with_comma():
    # A path containing a comma is exactly the case that would corrupt the header.
    assert header_safe_transforms(["read_lifecycle:superseded:/tmp/a,b/x.py"]) == [
        "read_lifecycle:superseded"
    ]


def test_passes_through_legacy_and_unrelated_tags():
    tags = ["smart_crush:3", "read_lifecycle:stale", "router:excluded:tool", "smart:lossless:table"]
    assert header_safe_transforms(tags) == tags


def test_joined_header_remains_unambiguous():
    tags = ["smart_crush:2:Bash,Grep", "read_lifecycle:stale:/a,b.py", "router:excluded:tool"]
    header = ",".join(header_safe_transforms(tags))
    # One token per tag — no stray commas leaking in from enriched detail.
    assert header.split(",") == ["smart_crush:2", "read_lifecycle:stale", "router:excluded:tool"]
