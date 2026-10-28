"""A fenced block passed through verbatim (it holds a protected tag) keeps its fences."""

from __future__ import annotations

from headroom.transforms.content_router import ContentRouter

CONTENT = """Here is the Template file that we wrote for the project today.
```xml
<config-block>
  <name>demo</name>
</config-block>
```
Please Review this carefully and tell me what you think about it.
The Second paragraph explains the Design choice that we made here.
{"name": "demo", "version": 1, "debug": true}
Then the Search results came back from the Second query run today.
src/main.c:10: int main() {
src/util.c:22: void helper() {
"""


def test_mixed_passthrough_of_protected_fenced_block_keeps_fence_lines():
    result = ContentRouter()._compress_mixed(CONTENT, context="")

    lines = result.compressed.splitlines()
    assert "```xml" in lines
    opening = lines.index("```xml")
    assert lines[opening + 1 : opening + 5] == [
        "<config-block>",
        "  <name>demo</name>",
        "</config-block>",
        "```",
    ]


UNCLOSED_BARE = """Here is the Template file that we wrote for the project today.
Please Review this carefully and tell me what you think about it.
{"name": "demo", "version": 1, "debug": true}
Then the Search results came back from the Second query run today.
src/main.c:10: int main() {
```
<config-block>
  <name>demo</name>
</config-block>"""


def test_mixed_passthrough_keeps_an_unclosed_bare_fence_as_written():
    """Passthrough is verbatim: no closing fence is invented and a bare fence stays bare."""
    result = ContentRouter()._compress_mixed(UNCLOSED_BARE, context="")

    lines = result.compressed.splitlines()
    assert lines[-4:] == ["```", "<config-block>", "  <name>demo</name>", "</config-block>"]
    assert "```unknown" not in lines
