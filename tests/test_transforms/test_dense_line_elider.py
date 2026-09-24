"""Dense-line elision: minified/base64/RSC lines that no compressor can shrink."""

from __future__ import annotations

import pytest

from headroom.transforms.content_router import ContentRouter, ContentRouterConfig
from headroom.transforms.dense_line_elider import elide_dense_lines


@pytest.fixture
def tokenizer():
    from headroom.providers import OpenAIProvider
    from headroom.tokenizer import Tokenizer

    return Tokenizer(OpenAIProvider().get_token_counter("gpt-4o"), "gpt-4o")


MINIFIED_JS = (
    "!function(){var e=window,t=e.document,n=t.createElement('div');"
    "n.className='x';for(var r=0;r<1e3;r++){n.appendChild(t.createTextNode(r))}"
    "e.__x=n;var a=e.location.search.replace(/^\\?/,'').split('&');"
) * 16
BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    * 12
)
PROSE = "This is an ordinary paragraph with plenty of spaces in it, " * 12
CODE = "\n".join(f"    value_{i} = compute(arg_{i}, other_{i})  # comment {i}" for i in range(40))


def test_prose_code_and_short_lines_untouched():
    for text in (PROSE, CODE, "short", "a" * 299):
        assert elide_dense_lines(text) == (text, 0)


def test_dense_lines_elided_others_kept():
    text = f"Script completed\nOutput:\n{MINIFIED_JS}\n{PROSE}\n{BASE64}\n"
    out, n = elide_dense_lines(text)
    assert n == 2
    lines = out.split("\n")
    assert lines[0] == "Script completed" and lines[3] == PROSE and lines[5] == ""
    assert lines[2].startswith(MINIFIED_JS[:160]) and lines[2].endswith(MINIFIED_JS[-80:])
    assert "chars of dense machine-generated content elided" in lines[2]
    assert len(out) < len(text) // 2


def test_router_applies_elision_when_no_compressor_wins():
    html = (
        "<!doctype html><html><head><script>" + MINIFIED_JS + "</script></head><body></body></html>"
    )
    router = ContentRouter(ContentRouterConfig(enable_kompress=False, min_section_tokens=10))
    result = router.compress(html, context="tool_result")
    assert "content elided" in result.compressed
    assert len(result.compressed) < len(html) // 2
    assert result.compressed.strip(), "must never blank a non-empty block"

    off = ContentRouter(
        ContentRouterConfig(
            enable_kompress=False, enable_dense_line_elision=False, min_section_tokens=10
        )
    )
    assert "content elided" not in off.compress(html, context="tool_result").compressed


def test_elided_block_carries_a_retrievable_marker():
    from headroom.cache.compression_store import get_compression_store

    html = "<html><head><script>" + MINIFIED_JS + "</script></head><body>hi there</body></html>"
    router = ContentRouter(ContentRouterConfig(enable_kompress=False, min_section_tokens=10))
    out = router.compress(html, context="tool_result").compressed
    assert "Retrieve original: hash=" in out
    key = out.rsplit("hash=", 1)[1].rstrip("]\n")
    assert get_compression_store().retrieve(key) is not None


def test_messages_path_keeps_elided_tool_result(tokenizer):
    """The Claude path discards marker-less lossy results (#1307); elision must survive it."""
    body = (
        "<!doctype html><html><head><script>"
        + MINIFIED_JS * 3
        + "</script></head><body><p>x</p></body></html>"
    )
    messages = [
        {"role": "user", "content": "check the site"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "curl -s x"}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": body}],
        },
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "and now?"},
    ]
    router = ContentRouter(ContentRouterConfig(enable_kompress=False, min_section_tokens=10))
    result = router.apply(messages, tokenizer)
    block = result.messages[2]["content"][0]["content"]
    text = block if isinstance(block, str) else block[0]["text"]
    assert "content elided" in text and "Retrieve original: hash=" in text
    assert len(text) < len(body) // 2


def test_single_dense_values_and_tsv_are_kept():
    """A lone JWT / signed URL / PATH / modulus is a value the agent asked for; TSV is data."""
    jwt = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9." + "eyJzdWIiOiIxMjM0NTY3ODkwIn0" * 40 + ".sig"
    url = "https://d1.cloudfront.net/v.mp4?Policy=" + "A" * 500 + "&Signature=" + "b" * 128
    path = "PATH=" + ":".join(f"/opt/homebrew/opt/package{i}/bin" for i in range(24))
    modulus = "Modulus: " + ":".join("ab" for _ in range(256))
    tsv = "\n".join("\t".join(f"column_{i}_value_{j}" for i in range(35)) for j in range(5))
    for text in (jwt, url, path, modulus, tsv, "Output:\n" + tsv * 3):
        assert elide_dense_lines(text) == (text, 0)


def test_compact_json_lines_are_left_to_smart_crusher():
    objs = " ".join(
        f'{{"file":"src/m_{i}.py","line":{i},"text":"repeated search payload"}}' for i in range(160)
    )
    assert elide_dense_lines(objs) == (objs, 0)
    assert elide_dense_lines("[" + ",".join(f'{{"id":{i}}}' for i in range(200)) + "]")[1] == 0


def test_lossless_mode_never_elides():
    html = "<html><head><script>" + MINIFIED_JS + "</script></head><body>hi</body></html>"
    router = ContentRouter(
        ContentRouterConfig(enable_kompress=False, lossless=True, min_section_tokens=10)
    )
    assert "content elided" not in router.compress(html, context="tool_result").compressed


def test_search_output_keeps_line_structure():
    grep = "src/bundle.js\n1:" + MINIFIED_JS + "\n2:export default x\nsrc/app.py\n10:def main():\n"
    out, n = elide_dense_lines(grep)
    lines = out.split("\n")
    assert n == 1 and lines[0] == "src/bundle.js" and lines[1].startswith("1:!function")
    assert lines[2:] == ["2:export default x", "src/app.py", "10:def main():", ""]
