"""RemoteKompressCompressor must really be a drop-in for KompressCompressor.

Its module docstring promises the class "mirrors KompressCompressor's public
surface (``is_ready`` / ``preload`` / ``ensure_background_load`` / ``compress``),
so it is a drop-in at the ContentRouter seam". That promise silently lapsed:
the local ``compress`` gained a ``ccr_original`` keyword and the remote one did
not.

ContentRouter passes ``ccr_original`` whenever custom tags are protected. On any
deployment with ``HEADROOM_KOMPRESS_ENDPOINT`` set — which is exactly the
sandboxed/enterprise install the remote compressor exists for — every such
request raised

    TypeError: RemoteKompressCompressor.compress() got an unexpected keyword
    argument 'ccr_original'

ContentRouter caught it with a broad ``except Exception`` and logged
``Kompress failed: ...`` at WARNING. The request then forwarded uncompressed
with ``tok_saved=0`` and the proxy reported success, so the deployment lost ALL
ML compression while every dashboard read "working, 0 saved".

From a field log (Copilot Chat on Windows, 0.36.x), on every single request:

    WARNING Kompress failed: RemoteKompressCompressor.compress() got an
            unexpected keyword argument 'ccr_original'
    INFO    [router] route_counts={...} compressed=0 frozen=1 msgs=2
    INFO    PERF ... tok_before=1623 tok_after=1623 tok_saved=0 savings=none
"""

from __future__ import annotations

import pytest
import tiktoken

from headroom.transforms.kompress_remote import RemoteKompressCompressor


# --------------------------------------------------------------------------- #
# The reported failure, end to end through the real call shape
# --------------------------------------------------------------------------- #
class _FakeResponse:
    status_code = 200

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.calls: list[dict] = []

    def post(self, url, headers=None, json=None):  # noqa: A002, ANN001
        self.calls.append(json or {})
        return _FakeResponse(self._payload)

    def close(self) -> None:
        return None


@pytest.fixture
def ccr_store(monkeypatch):
    from headroom.cache.compression_store import CompressionStore
    from headroom.telemetry import TOINConfig, ToolIntelligenceNetwork

    store = CompressionStore(enable_feedback=False)
    toin = ToolIntelligenceNetwork(config=TOINConfig(enabled=False, storage_path=""))
    monkeypatch.setattr("headroom.cache.compression_store.get_compression_store", lambda: store)
    monkeypatch.setattr("headroom.telemetry.get_toin", lambda: toin)
    return store


def _compressor(monkeypatch, *, enable_ccr: bool, payload: dict):
    monkeypatch.setenv("HEADROOM_KOMPRESS_ENDPOINT", "https://ml.example.invalid")
    c = RemoteKompressCompressor("https://ml.example.invalid")
    c._client = _FakeClient(payload)  # type: ignore[assignment]
    c.config.enable_ccr = enable_ccr
    # The 60-word fixtures below sit under the production word floor
    # (min_input_words=64); drop it to the clamp so the seam under test runs.
    c.config.min_input_words = 10
    return c


# 60 words: "short" saves 59, past the marker cost gate (CCR_MARKER_COST_WORDS).
ORIGINAL = "real secret block " * 60
PLACEHOLDER = "{{HEADROOM_TAG_0}} " * 60


@pytest.mark.parametrize("with_override", [False, True])
def test_remote_ccr_recovers_original_not_protection_placeholders(
    monkeypatch, ccr_store, with_override
) -> None:
    c = _compressor(
        monkeypatch,
        enable_ccr=True,
        payload={"compressed": "short", "compression_ratio": 0.2, "original_tokens": 999},
    )
    source = PLACEHOLDER if with_override else ORIGINAL
    result = c.compress(source, ccr_original=ORIGINAL if with_override else None)

    assert result.cache_key is not None
    recovered = ccr_store.retrieve(result.cache_key)
    assert recovered is not None
    assert recovered.original_content == ORIGINAL
    encoding = tiktoken.get_encoding("cl100k_base")
    assert recovered.original_tokens == len(encoding.encode(ORIGINAL, disallowed_special=()))
    assert recovered.original_item_count == recovered.compressed_item_count == 0


def test_remote_marker_gate_measures_the_whole_payload_like_local(monkeypatch, ccr_store) -> None:
    """Parity with KompressCompressor: the whole original against the whole
    marked candidate in one token unit. Both boundary sources pass through
    (41 single-token words saved against a 43-token marker; a head word that
    retokenizes: 100 -> 103), and a real saving reports that measurement."""
    from headroom.ccr.tool_injection import CCRToolInjector

    for source, drop in (
        (" ".join(["alpha"] * 99 + ["nfs"]), 41),
        (" ".join(["alpha"] * 36 + ["bureaucratic"] + ["alpha"] * 63), 36),
    ):
        kept = " ".join(source.split()[drop:])
        c = _compressor(monkeypatch, enable_ccr=True, payload={"compressed": kept})
        result = c.compress(source)
        assert result.compressed == source
        assert result.cache_key is None
        assert result.compression_ratio == 1.0

    big = " ".join(["alpha"] * 299 + ["nfs"])
    kept = " ".join(big.split()[60:])
    c = _compressor(monkeypatch, enable_ccr=True, payload={"compressed": kept})
    result = c.compress(big)
    injector = CCRToolInjector(
        provider="anthropic", inject_tool=False, inject_system_instructions=False
    )
    injector.scan_for_markers([{"role": "user", "content": result.compressed}])
    assert injector.detected_hashes == [result.cache_key]
    recovered = ccr_store.retrieve(injector.detected_hashes[0])
    assert recovered is not None and recovered.original_content == big
    encoding = tiktoken.get_encoding("cl100k_base")
    assert (result.original_tokens, result.compressed_tokens) == (
        len(encoding.encode(big, disallowed_special=())),
        len(encoding.encode(result.compressed, disallowed_special=())),
    )
