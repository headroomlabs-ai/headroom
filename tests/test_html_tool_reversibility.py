"""Real HTML extraction must not erase unrecoverable tool ground truth (#3775)."""

import pytest

pytest.importorskip("trafilatura")

from headroom.parser import CCR_RETRIEVAL_MARKER_RE
from headroom.providers import OpenAIProvider
from headroom.tokenizer import Tokenizer
from headroom.transforms.compression_units import CompressionUnit, compress_unit_with_router
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
)


def _script_heavy_html() -> str:
    js = "var a=1;function f(x){return x*2};" * 300
    return (
        "<!doctype html><html><head><script>"
        + js
        + "</script></head><body><p>the answer is 42</p></body></html>"
    )


def _tokenizer() -> Tokenizer:
    return Tokenizer(OpenAIProvider().get_token_counter("gpt-4o"), "gpt-4o")


@pytest.mark.parametrize("shape", ["tool_result", "role_tool"])
def test_html_tool_ground_truth_is_recoverable(shape: str) -> None:
    html = _script_heavy_html()
    router = ContentRouter(ContentRouterConfig(enable_kompress=False, min_section_tokens=10))
    extracted = router.compress(html, context="tool_result")
    assert extracted.strategy_used is CompressionStrategy.HTML
    assert extracted.compressed != html
    assert router._frozen_verdict_recoverable(
        CompressionStrategy.HTML, extracted.compressed
    ) == bool(CCR_RETRIEVAL_MARKER_RE.search(extracted.compressed))

    if shape == "tool_result":
        messages = [
            {"role": "user", "content": "check the site"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Bash",
                        "input": {"command": "curl -s x"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": html}],
            },
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "and now?"},
        ]
        result = router.apply(messages, _tokenizer())
        block = result.messages[2]["content"][0]["content"]
        output = block if isinstance(block, str) else block[0]["text"]
    else:
        result = router.apply(
            [{"role": "tool", "tool_call_id": "call_bash_1", "content": html}],
            _tokenizer(),
            protect_recent=0,
            protect_analysis_context=False,
        )
        output = result.messages[0]["content"]

    assert output == html or CCR_RETRIEVAL_MARKER_RE.search(output)


def test_html_provider_shell_unit_is_recoverable() -> None:
    html = _script_heavy_html().replace("><", ">\n<")
    router = ContentRouter(ContentRouterConfig(enable_kompress=False, min_section_tokens=10))
    extracted = router.compress(html)
    assert extracted.strategy_used is CompressionStrategy.HTML
    assert extracted.compressed != html

    result = compress_unit_with_router(
        CompressionUnit(
            text=html,
            provider="openai",
            endpoint="responses",
            role="tool",
            item_type="local_shell_call_output",
            min_bytes=1,
        ),
        router=router,
        tokenizer=_tokenizer(),
    )

    assert result.compressed == html or CCR_RETRIEVAL_MARKER_RE.search(result.compressed)
