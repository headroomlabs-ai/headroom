"""Tests for routing a selected EXTERNAL compressor through the content router.

Scope 3 of the pluggable-compressor work: an opt-in ``headroom.compressor``
entry point, when SELECTED, actually compresses matching real traffic through
:meth:`ContentRouter._apply_strategy_to_content`, with fail-open fallback to the
built-in dispatch. BACKWARD COMPATIBILITY is the hard requirement — with nothing
selected the branch is inert and the default request path is byte-identical.

Covers:
  (a) a selected external compressor compresses a matching block end-to-end;
  (b) its ``recoverable`` (hash -> original) map is retrievable from the CCR store;
  (c) fail-open: an external that raises / malforms / expands falls back to the
      built-in path (never breaks the request);
  (d) NOT selected, or a non-matching content type, leaves the built-in path
      untouched (the external is never even invoked);
plus the proxy seam that threads external names to the router.
"""

from __future__ import annotations

import hashlib

import pytest

from headroom.cache.compression_store import (
    get_compression_store,
    reset_compression_store,
)
from headroom.proxy.server import _external_compressor_selection
from headroom.transforms.compressor_registry import (
    CompressInput,
    CompressorDescriptor,
    CompressOutput,
)
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
)

# A JSON array reliably routes to SMART_CRUSHER (content type application/json)
# and is not touched by the STAGE-0 lossless fold, so it reaches the external
# dispatch branch. Big enough that the reference compressor's output shrinks it.
_JSON_ARRAY = (
    "["
    + ",".join(
        f'{{"id":{i},"name":"item-{i}","status":"active","value":{i * 7},'
        f'"note":"a fairly long descriptive field number {i} to add bulk"}}'
        for i in range(40)
    )
    + "]"
)


@pytest.fixture
def _memory_ccr(monkeypatch):
    """Isolated in-memory CCR store + offline content detection per test.

    ``HEADROOM_DETECT_BACKEND=python`` forces the pure-Python regex detector so
    ``compress()`` never touches the native Magika/ONNX detector (which needs a
    model download and blocks in this offline environment). The external-dispatch
    branch under test is independent of the detector backend.
    """
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    monkeypatch.setenv("HEADROOM_DETECT_BACKEND", "python")
    reset_compression_store()
    yield
    reset_compression_store()


class _WordTokenizer:
    """Word-count tokenizer stub — no model, deterministic, offline-safe."""

    def count_text(self, text: object) -> int:
        return len(str(text).split())

    def count_messages(self, messages: list[dict]) -> int:
        return sum(self.count_text(m.get("content", "")) for m in messages)


def _json_array(tag: str) -> str:
    """A distinct, spaced JSON array (>50 word-tokens, >500 chars) that the
    pure-Python detector routes to SMART_CRUSHER (application/json)."""
    return (
        "["
        + ",".join(
            f'{{"id":{i},"tag":"{tag}","note":"long descriptive field number {i} '
            f'to add real bulk here"}}'
            for i in range(40)
        )
        + "]"
    )


def _tool_msg(call_id: str, content: str) -> dict:
    # tool_call_id with no matching assistant tool_calls -> not excluded -> the
    # non-frozen one reaches compression (matches Bash/shell output).
    return {"role": "tool", "tool_call_id": call_id, "content": content}


class _RecordingExternal:
    """Reference in-process external ``Compressor`` for the router tests.

    Deterministically shrinks its input to a short marker string and, unless
    ``recoverable=False``, returns a ``{hash: original}`` recovery map keyed by
    the same hash it embeds in the output — mirroring how SmartCrusher's markers
    point back into the CCR store.
    """

    def __init__(
        self,
        name: str = "ext_json",
        content_types: tuple[str, ...] = ("application/json",),
        *,
        lossless: bool = False,
        recoverable: bool = True,
        raises: bool = False,
        expand: bool = False,
        malformed: bool = False,
        empty: bool = False,
        bad_hash: bool = False,
    ) -> None:
        self._name = name
        self._content_types = list(content_types)
        self._lossless = lossless
        self._recoverable = recoverable
        self._raises = raises
        self._expand = expand
        self._malformed = malformed
        self._empty = empty
        self._bad_hash = bad_hash
        self.calls: list[CompressInput] = []

    @property
    def descriptor(self) -> CompressorDescriptor:
        return CompressorDescriptor(
            name=self._name,
            content_types=self._content_types,
            lossless=self._lossless,
            cost_tier="fast",
            recoverable=self._recoverable,
        )

    def compress(self, inp: CompressInput):  # noqa: ANN201 - matches protocol
        self.calls.append(inp)
        if self._raises:
            raise RuntimeError("external boom")
        if self._malformed:
            return {"not": "a CompressOutput"}
        original = inp.content
        digest = hashlib.sha256(original.encode()).hexdigest()[:24]
        key = "zznothex" if self._bad_hash else digest
        if self._empty:
            content = ""
        elif self._expand:
            content = original + (" PADDING" * 200)
        else:
            content = f"[external-compressed <<ccr:{digest}>>]"
        recoverable = {key: original} if self._recoverable else {}
        return CompressOutput(
            content=content,
            tokens_before=len(original.split()),
            tokens_after=len(content.split()),
            lossless=self._lossless,
            markers=[f"external:{self._name}"],
            recoverable=recoverable,
            warnings=["reference-warning"],
        )


def _cfg(**kwargs) -> ContentRouterConfig:
    """Router config with Kompress OFF so the built-in fallback path never
    loads the ModernBERT ML model (keeps these unit tests fast and offline).
    The external-dispatch branch under test is independent of this flag."""
    kwargs.setdefault("enable_kompress", False)
    return ContentRouterConfig(**kwargs)


def _router_with_external(comp: _RecordingExternal, selection):
    """Build a router with ``comp`` registered and ``selection`` active."""
    router = ContentRouter(_cfg(active_external_compressors=selection))
    router.compressor_registry.register(comp, replace=True)
    # Re-resolve now that the external compressor is registered (the router
    # resolves the selection once at construction, before this injection).
    router._active_external_compressors = router._resolve_active_external_compressors()
    return router


# ─────────────────────────── (a) end-to-end ──────────────────────────────────


def test_selected_external_compresses_matching_block_end_to_end(_memory_ccr):
    comp = _RecordingExternal(name="ext_json", content_types=("application/json",))
    router = _router_with_external(comp, ["ext_json"])

    result = router.compress(_JSON_ARRAY)

    assert comp.calls, "external compressor should have been invoked"
    assert "external:ext_json" in result.strategy_chain
    assert "external-compressed" in result.compressed
    assert len(result.compressed) < len(_JSON_ARRAY)
    # The CompressInput carried the block + its detected MIME content type.
    assert comp.calls[0].content == _JSON_ARRAY
    assert comp.calls[0].content_type == "application/json"


def test_external_dispatch_via_apply_strategy_returns_normal_shape(_memory_ccr):
    comp = _RecordingExternal(name="ext_json", content_types=("application/json",))
    router = _router_with_external(comp, ["ext_json"])

    compressed, tokens, chain = router._apply_strategy_to_content(
        _JSON_ARRAY, CompressionStrategy.SMART_CRUSHER, ""
    )

    assert chain == ["external:ext_json"]
    assert "external-compressed" in compressed
    # Tokens counted with the router's OWN estimator (a positive int), not the
    # compressor's self-report.
    assert isinstance(tokens, int) and tokens > 0


# ─────────────────────────── (b) recoverable map ─────────────────────────────


def test_recoverable_map_is_retrievable(_memory_ccr):
    comp = _RecordingExternal(name="ext_json", content_types=("application/json",))
    router = _router_with_external(comp, ["ext_json"])

    router.compress(_JSON_ARRAY)

    digest = hashlib.sha256(_JSON_ARRAY.encode()).hexdigest()[:24]
    entry = get_compression_store().retrieve(digest)
    assert entry is not None, "recoverable entry should be in the CCR store"
    assert entry.original_content == _JSON_ARRAY
    assert entry.compression_strategy == "external:ext_json"


def test_non_hex_recoverable_hash_is_skipped_without_breaking(_memory_ccr):
    # A malformed (non-hex) recovery hash must not break the request; the block
    # is still compressed, only that entry is not retrievable.
    comp = _RecordingExternal(name="ext_json", content_types=("application/json",), bad_hash=True)
    router = _router_with_external(comp, ["ext_json"])

    result = router.compress(_JSON_ARRAY)

    assert "external:ext_json" in result.strategy_chain
    assert get_compression_store().retrieve("zznothex") is None


# ─────────────────────────── (c) fail-open ───────────────────────────────────


def test_external_raise_falls_back_to_builtin(_memory_ccr):
    comp = _RecordingExternal(name="ext_json", content_types=("application/json",), raises=True)
    router = _router_with_external(comp, ["ext_json"])

    result = router.compress(_JSON_ARRAY)

    assert comp.calls, "external should have been attempted"
    # Fell back: no external marker in the chain, request not broken.
    assert "external:ext_json" not in result.strategy_chain
    assert result.compressed and result.compressed.strip()


def test_external_malformed_output_falls_back_to_builtin(_memory_ccr):
    comp = _RecordingExternal(name="ext_json", content_types=("application/json",), malformed=True)
    router = _router_with_external(comp, ["ext_json"])

    result = router.compress(_JSON_ARRAY)

    assert comp.calls
    assert "external:ext_json" not in result.strategy_chain


def test_external_expansion_falls_back_to_builtin(_memory_ccr):
    comp = _RecordingExternal(name="ext_json", content_types=("application/json",), expand=True)
    router = _router_with_external(comp, ["ext_json"])

    result = router.compress(_JSON_ARRAY)

    assert comp.calls
    assert "external:ext_json" not in result.strategy_chain
    # Never expands: the returned block is no larger than the input.
    assert len(result.compressed) <= len(_JSON_ARRAY)


def test_external_empty_output_falls_back_to_builtin(_memory_ccr):
    comp = _RecordingExternal(name="ext_json", content_types=("application/json",), empty=True)
    router = _router_with_external(comp, ["ext_json"])

    result = router.compress(_JSON_ARRAY)

    assert comp.calls
    assert "external:ext_json" not in result.strategy_chain
    assert result.compressed.strip(), "non-empty input must never blank out"


# ─────────────── (d) not selected / non-matching → built-in ──────────────────


def test_not_selected_leaves_builtin_path_unchanged(_memory_ccr):
    comp = _RecordingExternal(name="ext_json", content_types=("application/json",))
    # Registered but NOT selected.
    router = _router_with_external(comp, None)

    baseline = ContentRouter(_cfg()).compress(_JSON_ARRAY)
    result = router.compress(_JSON_ARRAY)

    assert comp.calls == [], "unselected external must never be invoked"
    assert "external:ext_json" not in result.strategy_chain
    # Byte-identical to a plain router with no external registered at all.
    assert result.compressed == baseline.compressed
    assert result.strategy_chain == baseline.strategy_chain


def test_non_matching_content_type_leaves_builtin_path_unchanged(_memory_ccr):
    # Selected, but declares a content type the JSON block never has.
    comp = _RecordingExternal(name="ext_diff", content_types=("text/x-diff",))
    router = _router_with_external(comp, ["ext_diff"])

    baseline = ContentRouter(_cfg()).compress(_JSON_ARRAY)
    result = router.compress(_JSON_ARRAY)

    assert comp.calls == [], "non-matching external must never be invoked"
    assert "external:ext_diff" not in result.strategy_chain
    assert result.compressed == baseline.compressed


def test_default_config_has_no_external_selection():
    assert ContentRouterConfig().active_external_compressors is None
    router = ContentRouter(ContentRouterConfig())
    assert router._active_external_compressors == []


# ─────────────────────────── proxy seam ──────────────────────────────────────


@pytest.mark.parametrize(
    "selection,expected",
    [
        (None, None),
        (set(), None),
        ({"", "  "}, None),
        ({"smart_crusher"}, None),  # built-in only → no external
        ({"smart_crusher", "kompress"}, None),
        ({"my_ext"}, ["my_ext"]),
        ({"kompress", "my_ext"}, ["my_ext"]),
        ({"b_ext", "a_ext"}, ["a_ext", "b_ext"]),  # sorted
        ({"*"}, ["*"]),
        ({"*", "my_ext"}, ["*"]),  # wildcard wins
    ],
)
def test_external_compressor_selection_helper(selection, expected):
    assert _external_compressor_selection(selection) == expected


def test_wildcard_selection_activates_registered_external(_memory_ccr):
    comp = _RecordingExternal(name="ext_json", content_types=("application/json",))
    router = _router_with_external(comp, ["*"])

    result = router.compress(_JSON_ARRAY)

    assert comp.calls
    assert "external:ext_json" in result.strategy_chain
