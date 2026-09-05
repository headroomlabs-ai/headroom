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

import inspect

import pytest

from headroom.transforms.kompress_compressor import KompressCompressor
from headroom.transforms.kompress_remote import RemoteKompressCompressor


def _kwargs(fn) -> set[str]:
    """Public keywords only.

    ``_deadline_started_at`` is underscore-prefixed and only ever passed by
    kompress_compressor to itself on its recursive batch path — it never crosses
    the ContentRouter seam, so it is genuinely private and not part of the
    drop-in contract.
    """
    return {
        name
        for name, p in inspect.signature(fn).parameters.items()
        if name != "self"
        and not name.startswith("_")
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }


# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #
def test_remote_compress_accepts_every_local_keyword() -> None:
    """The drift guard. This is what would have caught the reported bug."""
    local = _kwargs(KompressCompressor.compress)
    remote = _kwargs(RemoteKompressCompressor.compress)

    missing = local - remote
    assert not missing, (
        f"RemoteKompressCompressor.compress is missing {sorted(missing)}. "
        "ContentRouter calls both through one seam, so a keyword the local "
        "compressor accepts and the remote one does not becomes a TypeError "
        "that ContentRouter swallows into a warning — silently disabling "
        "compression for the whole deployment."
    )


@pytest.mark.parametrize("method", ["is_ready", "preload", "ensure_background_load", "compress"])
def test_the_promised_public_surface_exists(method: str) -> None:
    assert callable(getattr(RemoteKompressCompressor, method, None))


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


def _compressor(monkeypatch, *, enable_ccr: bool, payload: dict):
    monkeypatch.setenv("HEADROOM_KOMPRESS_ENDPOINT", "https://ml.example.invalid")
    c = RemoteKompressCompressor("https://ml.example.invalid")
    c._client = _FakeClient(payload)  # type: ignore[assignment]
    c.config.enable_ccr = enable_ccr
    # The 60-word fixtures below sit under the production word floor
    # (min_input_words=64); drop it to the clamp so the seam under test runs.
    c.config.min_input_words = 10
    return c


ORIGINAL = "real secret block " * 20
PLACEHOLDER = "{{HEADROOM_TAG_0}} " * 20


def test_passing_ccr_original_no_longer_raises(monkeypatch) -> None:
    """The bug itself: this call is what ContentRouter makes."""
    c = _compressor(
        monkeypatch,
        enable_ccr=False,
        payload={"compressed": "short", "compression_ratio": 0.2},
    )

    result = c.compress(
        PLACEHOLDER,
        context="",
        question=None,
        target_ratio=0.5,
        allow_download=False,
        ccr_original=ORIGINAL,
    )

    assert result.compressed == "short"


def test_ccr_stores_the_pre_protection_text_not_the_placeholder(monkeypatch) -> None:
    """Fixing only the TypeError would leave retrieval returning a placeholder."""
    stored: dict = {}

    def _fake_store(original, compressed, original_tokens):  # noqa: ANN001
        stored["original"] = original
        stored["tokens"] = original_tokens
        return "cafebabe"

    monkeypatch.setattr("headroom.transforms.kompress_remote.store_kompress_in_ccr", _fake_store)
    c = _compressor(
        monkeypatch,
        enable_ccr=True,
        payload={"compressed": "short", "compression_ratio": 0.2},
    )

    result = c.compress(PLACEHOLDER, ccr_original=ORIGINAL)

    assert stored["original"] == ORIGINAL
    assert "HEADROOM_TAG" not in stored["original"]
    # Token count describes what was actually stored, not the placeholder.
    assert stored["tokens"] == len(ORIGINAL.split())
    assert result.cache_key == "cafebabe"


def test_the_common_path_without_an_override_is_unchanged(monkeypatch) -> None:
    stored: dict = {}

    def _fake_store(original, compressed, original_tokens):  # noqa: ANN001
        stored["original"] = original
        stored["tokens"] = original_tokens
        return "d00d"

    monkeypatch.setattr("headroom.transforms.kompress_remote.store_kompress_in_ccr", _fake_store)
    c = _compressor(
        monkeypatch,
        enable_ccr=True,
        payload={
            "compressed": "short",
            "compression_ratio": 0.2,
            "original_tokens": 999,
        },
    )

    c.compress(ORIGINAL)

    assert stored["original"] == ORIGINAL
    # Still the endpoint's own count when no override was supplied.
    assert stored["tokens"] == 999
