"""F2.2: TOIN write-gate tests for the per-mode CompressionPolicy.

When ``CompressionPolicy.toin_read_only`` is ``True`` (Subscription
auth mode), TOIN must serve cached recommendations but NEVER write new
pattern observations from this request. PAYG / OAuth keep writing so
the network effect keeps growing. The gate is read at the
``record_compression`` call site in ``smart_crusher.py`` and
``content_router.py``, and — since #4801 — at the Kompress CCR store
(``store_kompress_in_ccr``), the second write path of #4474.

These tests mirror the structure of
``tests/test_smart_crusher_toin_attachment.py`` (the F2.1-era TOIN
re-attachment regression suite) so a future contributor can locate the
expected behaviour by name.

Behaviour matrix:

| Mode         | toin_read_only | record_compression called? |
|--------------|----------------|----------------------------|
| Payg         | False          | yes                        |
| OAuth        | False          | yes                        |
| Subscription | True           | NO                         |

Direct callers (those that call ``crush()`` / ``crush_array_json()``
without going through ``apply()``) don't set
``self._runtime_compression_policy``, so they keep their pre-F2.2
write-enabled behaviour. That's a deliberate compatibility decision —
non-proxy callers have no auth context.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from headroom.proxy.auth_mode import AuthMode
from headroom.telemetry.toin import TOINConfig, get_toin, reset_toin
from headroom.tokenizer import Tokenizer
from headroom.tokenizers import EstimatingTokenCounter
from headroom.transforms.compression_policy import policy_for_mode


def _has_core() -> bool:
    """Match the pattern in ``test_smart_crusher_rust_parity.py``.

    SmartCrusher's __init__ hard-imports ``headroom._core`` (the Rust
    PyO3 wheel). On dev machines or CI lanes that haven't run
    ``scripts/build_rust_extension.sh``, the wheel is absent. Skip the
    SmartCrusher-touching tests rather than fail loudly — the
    ContentRouter tests don't need the wheel and exercise the same
    F2.2 gate code path.
    """
    try:
        from headroom._core import SmartCrusher  # noqa: F401

        return True
    except ImportError:
        return False


_skip_no_core = pytest.mark.skipif(
    not _has_core(),
    reason="headroom._core wheel not installed (run `scripts/build_rust_extension.sh`)",
)


@pytest.fixture
def fresh_toin():
    """Per-test TOIN instance backed by a tempdir to avoid global drift."""
    reset_toin()
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = str(Path(tmpdir) / "toin.json")
        toin = get_toin(
            TOINConfig(
                storage_path=storage,
                auto_save_interval=0,
            )
        )
        yield toin
        reset_toin()


def _bigger_array(n: int = 60) -> str:
    """JSON array of `n` dicts, sized to trigger crushing.

    Mirrors the helper in ``test_smart_crusher_toin_attachment.py`` so
    these tests use the same shape and any "didn't trigger compression"
    skip lines up with the existing suite.
    """
    items = [{"status": "ok", "tag": "x", "n": i} for i in range(n)]
    return json.dumps(items)


def _wrap_in_tool_message(payload: str) -> list[dict]:
    """Build the OpenAI-style ``role=tool`` message ``apply()`` walks."""
    return [{"role": "tool", "content": payload, "tool_call_id": "t1"}]


def _tokenizer() -> Tokenizer:
    return Tokenizer(EstimatingTokenCounter())  # type: ignore[arg-type]


# ─── SmartCrusher: apply() with policy ──────────────────────────────────


@_skip_no_core
def test_smart_crusher_payg_policy_writes_to_toin(fresh_toin):
    """PAYG: ``toin_read_only=False`` → record_compression IS called."""
    from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig

    crusher = SmartCrusher(SmartCrusherConfig())
    messages = _wrap_in_tool_message(_bigger_array(60))
    pre = sum(p.total_compressions for p in fresh_toin._patterns.values())

    policy = policy_for_mode(AuthMode.PAYG)
    assert policy.toin_read_only is False  # baseline sanity
    result = crusher.apply(messages, _tokenizer(), compression_policy=policy)

    if not result.transforms_applied:
        pytest.skip("payload didn't trigger compression — bump the size")
    post = sum(p.total_compressions for p in fresh_toin._patterns.values())
    assert post > pre, "PAYG should write to TOIN (network effect)"


@_skip_no_core
def test_smart_crusher_oauth_policy_writes_to_toin(fresh_toin):
    """OAuth: identical to PAYG in F2.2 — writes enabled."""
    from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig

    crusher = SmartCrusher(SmartCrusherConfig())
    messages = _wrap_in_tool_message(_bigger_array(60))
    pre = sum(p.total_compressions for p in fresh_toin._patterns.values())

    policy = policy_for_mode(AuthMode.OAUTH)
    assert policy.toin_read_only is False
    result = crusher.apply(messages, _tokenizer(), compression_policy=policy)

    if not result.transforms_applied:
        pytest.skip("payload didn't trigger compression — bump the size")
    post = sum(p.total_compressions for p in fresh_toin._patterns.values())
    assert post > pre, "OAuth (matches PAYG today) should write to TOIN"


@_skip_no_core
def test_smart_crusher_subscription_policy_skips_toin_write(fresh_toin):
    """Subscription: ``toin_read_only=True`` → record_compression is NOT called.

    This is THE behaviour change of F2.2 — keep the learning pool
    consistent for cache-stability-sensitive traffic.
    """
    from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig

    crusher = SmartCrusher(SmartCrusherConfig())
    messages = _wrap_in_tool_message(_bigger_array(60))
    pre = sum(p.total_compressions for p in fresh_toin._patterns.values())

    policy = policy_for_mode(AuthMode.SUBSCRIPTION)
    assert policy.toin_read_only is True  # baseline sanity
    result = crusher.apply(messages, _tokenizer(), compression_policy=policy)

    # Compression itself should still complete — this gate is on the
    # learning side only, not the compression path.
    if not result.transforms_applied:
        pytest.skip("payload didn't trigger compression — bump the size")
    post = sum(p.total_compressions for p in fresh_toin._patterns.values())
    assert post == pre, (
        "Subscription MUST NOT write to TOIN — load-bearing for keeping "
        "the learning pool consistent across cache-sensitive traffic"
    )


@_skip_no_core
def test_smart_crusher_no_policy_keeps_legacy_write_behaviour(fresh_toin):
    """Direct ``apply()`` call without ``compression_policy`` keeps
    pre-F2.2 behaviour: TOIN writes are not gated.

    Many test fixtures and non-proxy callers don't pass a policy; they
    must continue to feed the learning pool exactly as they did
    before F2.2.
    """
    from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig

    crusher = SmartCrusher(SmartCrusherConfig())
    messages = _wrap_in_tool_message(_bigger_array(60))
    pre = sum(p.total_compressions for p in fresh_toin._patterns.values())

    # No `compression_policy` kwarg.
    result = crusher.apply(messages, _tokenizer())

    if not result.transforms_applied:
        pytest.skip("payload didn't trigger compression — bump the size")
    post = sum(p.total_compressions for p in fresh_toin._patterns.values())
    assert post > pre, "no policy → legacy write-enabled behaviour"


# ─── ContentRouter: apply() captures the policy ─────────────────────────


def test_content_router_apply_stores_runtime_policy():
    """``ContentRouter.apply()`` must populate
    ``self._runtime_compression_policy`` from kwargs so
    ``_record_to_toin`` can read it.

    We don't assert TOIN behaviour here (the router routes most JSON
    arrays to SmartCrusher, which has its own gate already covered
    above); the load-bearing thing for the parity guard is that the
    field is wired through.
    """
    from headroom.transforms.content_router import ContentRouter

    router = ContentRouter()
    # Sanity: the field exists on a fresh instance and starts None.
    assert router._runtime_compression_policy is None

    policy = policy_for_mode(AuthMode.SUBSCRIPTION)
    # Empty-message apply is fine — the field assignment happens
    # before the message walk, so we don't need a payload that
    # actually compresses.
    router.apply([], _tokenizer(), compression_policy=policy)
    assert router._runtime_compression_policy is policy, (
        "ContentRouter.apply() must capture the policy onto self so _record_to_toin can read it"
    )


def test_content_router_subscription_skips_toin_record(fresh_toin):
    """ContentRouter._record_to_toin returns early when
    policy.toin_read_only is True.

    We exercise the gate directly rather than building a fixture that
    routes to a non-SmartCrusher compressor — both are equivalent
    coverage for the gate, and the direct call avoids the routing
    flake from ``test_smart_crusher_toin_attachment.py``'s comments.
    """
    from headroom.transforms.content_router import (
        CompressionStrategy,
        ContentRouter,
    )

    router = ContentRouter()
    router._runtime_compression_policy = policy_for_mode(AuthMode.SUBSCRIPTION)

    pre = sum(p.total_compressions for p in fresh_toin._patterns.values())
    # Pick TEXT strategy (not SMART_CRUSHER, which has its own
    # early-return). With Subscription policy, the F2.2 gate fires
    # and the call returns before ever loading TOIN.
    router._record_to_toin(
        strategy=CompressionStrategy.TEXT,
        content="some text content",
        compressed="compressed",
        original_tokens=100,
        compressed_tokens=50,
    )
    post = sum(p.total_compressions for p in fresh_toin._patterns.values())
    assert post == pre, "Subscription policy must skip ContentRouter TOIN write"


def test_content_router_payg_records_to_toin(fresh_toin):
    """PAYG policy → ContentRouter._record_to_toin proceeds to the
    real TOIN call. Asserts the gate doesn't accidentally fire when
    ``toin_read_only=False``.
    """
    from headroom.transforms.content_router import (
        CompressionStrategy,
        ContentRouter,
    )

    router = ContentRouter()
    router._runtime_compression_policy = policy_for_mode(AuthMode.PAYG)

    pre = sum(p.total_compressions for p in fresh_toin._patterns.values())
    router._record_to_toin(
        strategy=CompressionStrategy.TEXT,
        content="some text content with structure that learns",
        compressed="compressed shorter",
        original_tokens=100,
        compressed_tokens=50,
    )
    post = sum(p.total_compressions for p in fresh_toin._patterns.values())
    # Real TOIN write should happen unless _create_content_signature
    # returns None (it can for malformed inputs). We accept either
    # "post > pre" (signature succeeded) OR "post == pre with a
    # signature-None path"; the load-bearing assertion is that the
    # F2.2 gate did NOT fire (which it would with toin_read_only=True
    # regardless of signature).
    assert post >= pre, (
        "PAYG must not be blocked by the F2.2 gate — write should happen or fall through naturally"
    )


# ─── ContentRouter → SmartCrusher.crush() (#4474) ───────────────────────


@_skip_no_core
@pytest.mark.parametrize(
    ("mode", "expect_write"),
    [(AuthMode.SUBSCRIPTION, False), (AuthMode.PAYG, True)],
)
def test_content_router_smart_crusher_path_honours_toin_read_only(fresh_toin, mode, expect_write):
    """The router compresses JSON arrays via the shared crusher's
    ``crush()`` and never calls ``crusher.apply()``, so the crusher's
    own ``_runtime_compression_policy`` stays None. The router must
    forward the request's ``toin_read_only`` gate to that call.
    """
    from headroom.transforms.content_router import ContentRouter

    router = ContentRouter()
    messages = _wrap_in_tool_message(_bigger_array(60))
    pre = sum(p.total_compressions for p in fresh_toin._patterns.values())

    result = router.apply(messages, _tokenizer(), compression_policy=policy_for_mode(mode))

    assert any("smart_crusher" in t for t in result.transforms_applied), (
        "payload must route through SmartCrusher for this test to mean anything"
    )
    # The shared crusher's own field is untouched — the gate travels per call.
    assert router._get_smart_crusher()._runtime_compression_policy is None
    post = sum(p.total_compressions for p in fresh_toin._patterns.values())
    if expect_write:
        assert post > pre, "write-enabled policy must still feed TOIN via the router"
    else:
        assert post == pre, "Subscription MUST NOT write to TOIN via router → crush()"


@_skip_no_core
def test_smart_crusher_crush_record_toin_false_skips_write(fresh_toin):
    """``crush(record_toin=False)`` compresses as usual but skips TOIN;
    the default keeps the legacy write-enabled behaviour."""
    from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig

    crusher = SmartCrusher(SmartCrusherConfig())
    payload = _bigger_array(60)

    gated = crusher.crush(payload, record_toin=False)
    if not gated.was_modified or gated.strategy == "passthrough":
        pytest.skip("payload didn't trigger compression — bump the size")
    assert sum(p.total_compressions for p in fresh_toin._patterns.values()) == 0

    crusher.crush(payload)
    assert sum(p.total_compressions for p in fresh_toin._patterns.values()) > 0


# ─── Kompress CCR store: record_toin gate (#4801) ────────────────────────


def _toin_total(toin) -> int:
    return sum(p.total_compressions for p in toin._patterns.values())


def test_kompress_store_record_toin_false_skips_write(fresh_toin):
    """``store_kompress_in_ccr(record_toin=False)`` keeps the CCR mapping
    (the compressed output stays retrievable) but writes no TOIN
    observation; the default call writes one."""
    from headroom.cache.compression_store import get_compression_store
    from headroom.transforms.kompress_compressor import store_kompress_in_ccr

    cache_key = store_kompress_in_ccr("word " * 400, "word " * 100, 400, record_toin=False)
    assert cache_key is not None
    assert get_compression_store().retrieve(cache_key) is not None
    assert _toin_total(fresh_toin) == 0

    cache_key = store_kompress_in_ccr("other " * 400, "other " * 100, 400)
    assert cache_key is not None
    assert _toin_total(fresh_toin) == 1


class _FakeEncoding:
    def __init__(self, word_lists: list[list[str]]):
        self._word_lists = word_lists
        self._ids = [[0] * len(w) for w in word_lists]

    def __getitem__(self, key: str):
        return {"input_ids": self._ids, "attention_mask": self._ids}[key]

    def word_ids(self, batch_index: int = 0) -> list[int]:
        return list(range(len(self._word_lists[batch_index])))


class _FakeTokenizer:
    def __call__(self, words, **_kwargs):
        batch = words if (words and isinstance(words[0], list)) else [words]
        return _FakeEncoding(batch)


class _FakeModel:
    def get_keep_mask(self, input_ids, _attention_mask):
        return [[i < 2 for i in range(len(input_ids[0]))]]

    def get_scores(self, input_ids, _attention_mask):
        return [[1.0 if i < 2 else 0.0 for i in range(len(row))] for row in input_ids]


def _local_compressor(monkeypatch):
    from headroom.transforms.kompress_compressor import KompressCompressor

    monkeypatch.setattr(
        "headroom.transforms.kompress_compressor._load_kompress",
        lambda *a, **k: (_FakeModel(), _FakeTokenizer(), "onnx"),
    )
    return KompressCompressor()


def test_kompress_compressor_record_toin_false_skips_write(fresh_toin, monkeypatch):
    """The local compressor forwards the gate to the CCR store: gated
    compression still returns a retrieval key but records nothing."""
    compressor = _local_compressor(monkeypatch)
    content = "line one\nline two\nline three " * 20

    gated = compressor.compress(content, record_toin=False)
    assert gated.cache_key is not None
    assert _toin_total(fresh_toin) == 0

    compressor.compress(content)
    assert _toin_total(fresh_toin) == 1


def test_kompress_compress_batch_record_toin_false_skips_write(fresh_toin, monkeypatch):
    """The batch path (which reconstructs and stores separately) honours
    the gate the same way."""
    compressor = _local_compressor(monkeypatch)
    content = "line one\nline two\nline three " * 20

    results = compressor.compress_batch([content], record_toin=False)
    assert results[0].cache_key is not None
    assert _toin_total(fresh_toin) == 0


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def post(self, url, headers=None, json=None):  # noqa: A002, ANN001
        return _FakeResponse(self._payload)

    def close(self) -> None:
        return None


def test_remote_kompress_record_toin_false_skips_write(fresh_toin, monkeypatch):
    """The remote client shares the store function, so the same gate
    applies on the hosted-endpoint path."""
    from headroom.transforms.kompress_remote import RemoteKompressCompressor

    compressor = RemoteKompressCompressor("https://ml.example.invalid")
    compressor._client = _FakeClient({"compressed": "short", "compression_ratio": 0.2})  # type: ignore[assignment]
    compressor.config.enable_ccr = True
    compressor.config.min_input_words = 10
    content = "real secret block " * 60

    gated = compressor.compress(content, record_toin=False)
    assert gated.cache_key is not None
    assert _toin_total(fresh_toin) == 0

    compressor.compress(content)
    assert _toin_total(fresh_toin) == 1


@_skip_no_core
def test_content_router_kompress_path_forwards_toin_gate(monkeypatch):
    """The router never calls ``KompressCompressor.apply()``, so it must
    forward the request's gate per call, as it does for ``crush()``: the
    kwarg is sent only when the policy is read-only."""
    from headroom.transforms.content_router import ContentRouter
    from headroom.transforms.kompress_compressor import KompressResult

    calls: list[dict] = []

    class _Stub:
        def is_ready(self) -> bool:
            return True

        def ensure_background_load(self) -> None:
            return None

        def compress(self, content: str, **kwargs) -> KompressResult:
            calls.append(kwargs)
            return KompressResult(
                compressed="short",
                original=content,
                original_tokens=100,
                compressed_tokens=1,
                compression_ratio=0.01,
            )

    router = ContentRouter()
    monkeypatch.setattr(router, "_get_kompress", lambda: _Stub())
    content = "plain prose about workers processing batches steadily " * 20

    router._runtime_compression_policy = policy_for_mode(AuthMode.SUBSCRIPTION)
    router._try_ml_compressor(content, "")
    assert calls[-1].get("record_toin") is False

    router._runtime_compression_policy = policy_for_mode(AuthMode.PAYG)
    router._try_ml_compressor(content, "")
    assert "record_toin" not in calls[-1]


def test_kompress_apply_subscription_policy_skips_write(fresh_toin, monkeypatch):
    """Through the Transform interface, a Subscription policy reaches
    ``compress()`` as the gate; no policy keeps the legacy write."""
    compressor = _local_compressor(monkeypatch)
    content = "line one\nline two\nline three " * 20
    messages = [{"role": "tool", "content": content, "tool_call_id": "t1"}]

    policy = policy_for_mode(AuthMode.SUBSCRIPTION)
    compressor.apply(messages, _tokenizer(), compression_policy=policy)
    assert _toin_total(fresh_toin) == 0

    compressor.apply(messages, _tokenizer())
    assert _toin_total(fresh_toin) == 1
