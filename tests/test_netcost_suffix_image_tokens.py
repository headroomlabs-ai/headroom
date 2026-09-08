"""An image must not inflate the net-cost suffix (S).

``_netcost_message_tokens`` used to walk block-list content itself and fall back
to ``str(block)`` for anything that was not ``text`` or ``tool_result``, on the
stated assumption that such blocks "rarely dominate a suffix". An ``image`` block
breaks that assumption completely: ``str()`` embeds the whole base64 payload.

    512x512 PNG        20,034 counted   vs  ~349 real    57x
    1092x1092 shot    100,034 counted   vs ~1,589 real   63x
    1568x1568        233,367 counted   vs ~1,600 real  146x

S is the cache-bust cost -- the tokens re-written if message *j* is mutated -- so
an image inflated S for **every message before it**, and the break-even gate then
declined to compress any of them. A single screenshot could switch off net-cost
compression for the whole earlier conversation.
"""

from __future__ import annotations

import base64

import pytest

from headroom.tokenizer import Tokenizer
from headroom.tokenizers import get_tokenizer
from headroom.transforms.content_router import _netcost_message_tokens


@pytest.fixture
def tok() -> Tokenizer:
    return Tokenizer(get_tokenizer("claude-sonnet-4-6"), "claude-sonnet-4-6")


def _image_block(payload_bytes: int) -> dict:
    data = base64.b64encode(b"\x89PNG" + b"\x00" * payload_bytes).decode()
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": data},
    }


@pytest.mark.parametrize("payload_bytes", [120_000, 600_000, 1_400_000])
def test_image_is_not_counted_as_its_base64_payload(tok: Tokenizer, payload_bytes: int) -> None:
    """Cost must not scale with the base64 length."""
    message = {"role": "user", "content": [_image_block(payload_bytes)]}

    counted = _netcost_message_tokens(message, tok)

    # Anthropic caps image cost around 1600 tokens; anything in the tens of
    # thousands means the payload is being counted as text.
    assert counted <= 2000, f"image counted as {counted:,} tokens"


def test_image_cost_does_not_grow_with_payload_size(tok: Tokenizer) -> None:
    """A 12x larger payload must not cost ~12x more."""
    small = _netcost_message_tokens({"role": "user", "content": [_image_block(120_000)]}, tok)
    large = _netcost_message_tokens({"role": "user", "content": [_image_block(1_400_000)]}, tok)

    assert large == small


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "text", "text": "hello world " * 50}],
        [{"type": "tool_result", "content": "result text " * 40}],
        [{"type": "tool_result", "content": [{"type": "text", "text": "x " * 60}]}],
    ],
    ids=["text", "tool_result_str", "tool_result_list"],
)
def test_text_bearing_blocks_are_unchanged(tok: Tokenizer, content: list) -> None:
    """Delegation must be behaviour-preserving for what already worked.

    These are the shapes the old local walk handled correctly; pinning them
    keeps the delegation from quietly changing suffix sizes on normal traffic.
    """
    counted = _netcost_message_tokens({"role": "user", "content": content}, tok)
    text = "".join(
        block.get("text", "")
        or (block.get("content") if isinstance(block.get("content"), str) else "")
        or "".join(
            sub.get("text", "") for sub in (block.get("content") or []) if isinstance(sub, dict)
        )
        for block in content
    )

    assert counted == pytest.approx(tok.count_text(text), abs=2)


def test_plain_string_content_still_counted(tok: Tokenizer) -> None:
    message = {"role": "user", "content": "plain " * 20}

    assert _netcost_message_tokens(message, tok) == tok.count_text(message["content"])
