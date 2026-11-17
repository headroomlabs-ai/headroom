"""An install with an affected Pillow never decodes an untrusted image.

`pip install headroom-ai[all]` resolves Pillow 12 under fastembed's <13 cap, and
Python 3.10 has no Pillow 13. Those installs must forward images unchanged.
"""

from __future__ import annotations

import base64
import io
import logging

import pytest

pytest.importorskip("PIL")

import PIL  # noqa: E402
from PIL import Image  # noqa: E402

from headroom.image import ImageCompressor, decoder_guard  # noqa: E402
from headroom.tokenizers.base import BaseTokenizer  # noqa: E402


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, format="PNG")
    return buf.getvalue()


_PNG_BYTES = _png()
_PNG = base64.b64encode(_PNG_BYTES).decode()


@pytest.fixture
def pillow_version(monkeypatch):
    def use(installed: str) -> None:
        monkeypatch.setattr(PIL, "__version__", installed)
        decoder_guard.pillow_is_safe.cache_clear()

    yield use
    decoder_guard.pillow_is_safe.cache_clear()


@pytest.fixture
def image_opens(monkeypatch):
    opened: list[object] = []
    real_open = Image.open
    monkeypatch.setattr(Image, "open", lambda *a, **k: opened.append(a) or real_open(*a, **k))
    return opened


def _messages() -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What does this show?"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": _PNG},
                },
            ],
        }
    ]


def test_an_affected_pillow_forwards_images_without_opening_them(
    pillow_version, image_opens, caplog
):
    pillow_version("12.3.0")
    messages = _messages()

    with caplog.at_level(logging.WARNING, logger=decoder_guard.__name__):
        assert ImageCompressor().compress(messages, provider="anthropic") is messages
        assert ImageCompressor().compress(_messages(), provider="anthropic") == _messages()
        tokens = BaseTokenizer._estimate_image_tokens({"source": {"bytes": _PNG_BYTES}})

    assert image_opens == []
    assert tokens > 0
    assert len(caplog.records) == 1  # once per process, not per image
    assert "Pillow 12.3.0" in caplog.records[0].getMessage()


def test_a_fixed_pillow_still_decodes(pillow_version, image_opens):
    pillow_version("13.0.0")

    BaseTokenizer._estimate_image_tokens({"source": {"bytes": _PNG_BYTES}})

    assert image_opens
