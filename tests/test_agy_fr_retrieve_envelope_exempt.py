"""headroom-8tm: the headroom_retrieve result envelope must NOT be re-compressed.

On agy the retrieve-result functionResponse carries a name that matches neither
``is_headroom_retrieve_name`` nor ``_args_mention_retrieve``, so BOTH the
name-based fr exemption and the functionCall hash-collection miss it. Left
unfixed, the resolved envelope re-compresses into a marker every turn (the model
re-retrieves it, L1) and the ORIGINAL resent leaf keeps re-retrieving (the 236x,
L2). These tests pin the name-INDEPENDENT, content-based exemption.

Fixtures mirror the real ``ccr.mcp_server._retrieve_content`` serialization
(``json.dumps(result, indent=2)``) plus the agy ``Created At:/Completed At:``
text wrapper observed in the fry run3 store.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

import headroom.transforms.agy_fr_compressor as mod
from headroom.cache.compression_store import (
    default_ccr_hash,
    get_compression_store,
    reset_compression_store,
)
from headroom.tokenizers import get_tokenizer
from headroom.transforms.agy_fr_compressor import (
    _FR_CCR_MARKER_PREFIX,
    _ccr_envelope_hash,
    _collect_retrieved_hashes,
    compress_function_response_leaves,
)

_MODEL = "gemini-3-flash-agent"
# Big, non-repeating original content -- well above the compression floor, so if
# it were NOT exempt it would compress to a marker.
_ORIGINAL = "watermark CRIMSON-WALRUS " + ("archive row alpha beta gamma delta epsilon " * 90)


@pytest.fixture
def tok() -> Any:
    return get_tokenizer(_MODEL)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    reset_compression_store()
    s = get_compression_store()
    yield s
    reset_compression_store()


def _local_envelope_dict(hash_key: str, original: str = _ORIGINAL) -> dict:
    return {
        "hash": hash_key,
        "source": "local",
        "original_content": original,
        "original_item_count": 10,
        "compressed_item_count": 1,
        "retrieval_count": 1,
    }


def _proxy_envelope_dict(hash_key: str, original: str = _ORIGINAL) -> dict:
    # WU-2b: source is the leading key on the proxy path too.
    return {
        "source": "proxy",
        "hash": hash_key,
        "original_content": original,
        "original_tokens": 500,
        "original_item_count": 10,
        "compressed_item_count": 1,
        "tool_name": "headroom_retrieve",
        "retrieval_count": 1,
    }


def _agy_text(envelope: dict) -> str:
    """The envelope as agy renders it: a timestamped text wrapper + indent=2 JSON."""
    body = json.dumps(envelope, indent=2)
    return f"Created At: 2026-07-11T15:59:27+02:00\nCompleted At: 2026-07-11T15:59:28+02:00\n{body}"


# --- Detector -------------------------------------------------------------
class TestDetector:
    def test_local_dict(self) -> None:
        h = default_ccr_hash(_ORIGINAL)
        assert _ccr_envelope_hash(_local_envelope_dict(h)) == h

    def test_proxy_dict(self) -> None:
        h = default_ccr_hash(_ORIGINAL)
        assert _ccr_envelope_hash(_proxy_envelope_dict(h)) == h

    def test_local_text(self) -> None:
        h = default_ccr_hash(_ORIGINAL)
        assert _ccr_envelope_hash(_agy_text(_local_envelope_dict(h))) == h

    def test_proxy_text(self) -> None:
        h = default_ccr_hash(_ORIGINAL)
        assert _ccr_envelope_hash(_agy_text(_proxy_envelope_dict(h))) == h

    def test_file_pointer_variant_still_detected(self) -> None:
        # agy replaced a large original_content with a saved-to-file pointer;
        # the LEADING hash+source keys survive.
        h = "a" * 24
        env = {
            "hash": h,
            "source": "local",
            "original_content": "The output was large and was saved to: file:///tmp/x",
        }
        assert _ccr_envelope_hash(_agy_text(env)) == h

    def test_source_read_is_not_an_envelope(self) -> None:
        # A leaf that READS headroom's own source: key NAMES present, but the
        # hash value is a variable (`hash_key`), not a 24-hex literal.
        leaf = (
            'return {\n  "hash": hash_key,\n  "source": "local",\n  "original_content": entry.x,\n}'
        )
        assert _ccr_envelope_hash(leaf) is None

    def test_uppercase_hash_rejected(self) -> None:
        env = _local_envelope_dict("A" * 24)
        assert _ccr_envelope_hash(env) is None
        assert _ccr_envelope_hash(_agy_text(env)) is None

    def test_forty_hex_rejected(self) -> None:
        # A 40-hex git sha must not match the {24} anchor (hex boundary).
        leaf = '{\n  "hash": "' + ("d" * 40) + '",\n  "source": "local"\n}'
        assert _ccr_envelope_hash(leaf) is None

    def test_plain_text_is_none(self) -> None:
        assert _ccr_envelope_hash("just some large file content\n" * 50) is None


# --- L1: envelope leaf never compressed -----------------------------------
def _fr_contents(name: str, leaf: Any) -> list[dict]:
    return [
        {
            "role": "user",
            "parts": [{"functionResponse": {"name": name, "response": {"output": leaf}}}],
        }
    ]


class TestL1Exempt:
    @pytest.mark.parametrize(
        "name", ["headroom.headroom_retrieve", "headroom", "call_mcp_tool", None]
    )
    def test_text_envelope_not_compressed_regardless_of_name(
        self, tok: Any, store: Any, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        h = default_ccr_hash(_ORIGINAL)
        env_text = _agy_text(_local_envelope_dict(h))
        contents = _fr_contents(name, env_text)

        calls: list = []
        orig = mod._compress_fr_leaf
        monkeypatch.setattr(
            mod, "_compress_fr_leaf", lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
        )

        before, after, leaves = compress_function_response_leaves(contents, "ccr", tok, store)
        out = contents[0]["parts"][0]["functionResponse"]["response"]["output"]
        assert out == env_text  # verbatim -- not a marker
        assert not out.startswith(_FR_CCR_MARKER_PREFIX)
        assert leaves == 0
        assert calls == []  # _compress_fr_leaf never called for the envelope

    def test_proxy_and_file_pointer_variants_exempt(self, tok: Any, store: Any) -> None:
        h = default_ccr_hash(_ORIGINAL)
        for leaf in (
            _agy_text(_proxy_envelope_dict(h)),
            _agy_text(
                {"hash": "b" * 24, "source": "local", "original_content": "saved to: file:///tmp/y"}
            ),
        ):
            contents = _fr_contents("headroom.headroom_retrieve", leaf)
            compress_function_response_leaves(contents, "ccr", tok, store)
            assert contents[0]["parts"][0]["functionResponse"]["response"]["output"] == leaf

    def test_dict_envelope_response_exempt(self, tok: Any, store: Any) -> None:
        # response IS the envelope dict (structured, not text-rendered).
        h = default_ccr_hash(_ORIGINAL)
        env = _local_envelope_dict(h)
        contents = [
            {"role": "user", "parts": [{"functionResponse": {"name": "headroom", "response": env}}]}
        ]
        compress_function_response_leaves(contents, "ccr", tok, store)
        # original_content left verbatim (dict exempt as a whole)
        assert (
            contents[0]["parts"][0]["functionResponse"]["response"]["original_content"] == _ORIGINAL
        )


# --- L2: envelope hash exempts the ORIGINAL resent leaf --------------------
class TestL2Exempt:
    def test_envelope_hash_collected(self) -> None:
        h = default_ccr_hash(_ORIGINAL)
        contents = _fr_contents("headroom.headroom_retrieve", _agy_text(_local_envelope_dict(h)))
        assert h in _collect_retrieved_hashes(contents)

    def test_original_leaf_exempt_when_envelope_present(self, tok: Any, store: Any) -> None:
        h = default_ccr_hash(_ORIGINAL)
        contents = [
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "headroom.headroom_retrieve",
                            "response": {"output": _agy_text(_local_envelope_dict(h))},
                        }
                    },
                ],
            },
            {
                "role": "user",
                "parts": [
                    {"functionResponse": {"name": "read_file", "response": {"output": _ORIGINAL}}},
                ],
            },
        ]
        compress_function_response_leaves(contents, "ccr", tok, store)
        # the resent ORIGINAL leaf is exempt (its hash is in retrieved_hashes)
        assert contents[1]["parts"][0]["functionResponse"]["response"]["output"] == _ORIGINAL


# --- Negatives: normal / false-positive leaves STILL compress -------------
class TestStillCompresses:
    def test_normal_large_leaf_compresses(self, tok: Any, store: Any) -> None:
        contents = _fr_contents("read_file", _ORIGINAL)
        before, after, leaves = compress_function_response_leaves(contents, "ccr", tok, store)
        out = contents[0]["parts"][0]["functionResponse"]["response"]["output"]
        assert leaves == 1
        assert out.startswith(_FR_CCR_MARKER_PREFIX)

    def test_source_read_leaf_still_compresses(self, tok: Any, store: Any) -> None:
        # Large leaf mentioning the key NAMES but no 24-hex hash value.
        src = (
            'def build():\n  return {\n    "hash": hash_key,\n    "source": "local",\n'
            '    "original_content": entry.original_content,\n  }\n'
        ) * 40
        contents = _fr_contents("read_file", src)
        before, after, leaves = compress_function_response_leaves(contents, "ccr", tok, store)
        out = contents[0]["parts"][0]["functionResponse"]["response"]["output"]
        assert leaves == 1
        assert out.startswith(_FR_CCR_MARKER_PREFIX)
