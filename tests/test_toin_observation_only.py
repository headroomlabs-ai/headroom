"""PR-B5 acceptance tests: TOIN observation-only contract.

Pins three guarantees:

1. `get_recommendation()` returns `None` and emits a `DeprecationWarning`
   exactly once per process. The request-time hint API is retired.
2. The aggregation key is `(tenant_key, auth_mode, model_family,
   structure_hash)` — two patterns with the same `structure_hash` but
   different `auth_mode`, `model_family`, or `tenant_key` are tracked
   as distinct rows in the TOIN store. PR-F3 added `tenant_key` so two
   tenants on the same proxy don't cross-pollinate compression
   patterns.
3. Recording a compression event does NOT alter the bytes SmartCrusher
   produces for an identical input. SmartCrusher is deterministic; TOIN
   only observes.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from headroom.telemetry import (
    DEFAULT_AUTH_MODE,
    DEFAULT_MODEL_FAMILY,
    DEFAULT_TENANT_KEY,
    TOINConfig,
    ToolIntelligenceNetwork,
    ToolSignature,
    reset_toin,
)


@pytest.fixture(autouse=True)
def _reset_toin(monkeypatch, tmp_path: Path):
    """Force every test to use a fresh tempfile-backed TOIN."""
    storage = tmp_path / "toin_obs_test.json"
    monkeypatch.setenv("HEADROOM_TOIN_PATH", str(storage))
    reset_toin()
    # Also reset the class-level deprecation flag so each test gets a
    # fresh "one warning" budget. Without this, test ordering would
    # determine whether the warning fires.
    ToolIntelligenceNetwork._DEPRECATION_WARNED = False
    yield
    reset_toin()
    ToolIntelligenceNetwork._DEPRECATION_WARNED = False


# ── Part 1: deprecation surface ────────────────────────────────────────────


def test_get_recommendation_returns_none_with_deprecation_warning():
    """get_recommendation() returns None and emits DeprecationWarning once."""
    toin = ToolIntelligenceNetwork()
    sig = ToolSignature.from_items([{"id": "1", "status": "ok"}])

    # First call: warning fires.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = toin.get_recommendation(sig)

    assert result is None, "PR-B5: get_recommendation must return None"
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1, f"expected 1 DeprecationWarning, got {len(deprecations)}"
    assert "PR-B5" in str(deprecations[0].message)

    # Second call: still None, but warning is suppressed (once-per-process).
    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        result2 = toin.get_recommendation(sig)
    assert result2 is None
    assert all(not issubclass(w.category, DeprecationWarning) for w in caught2)


def test_compression_hint_is_not_publicly_exported():
    """`CompressionHint` is no longer re-exported from `headroom.telemetry`."""
    import headroom.telemetry as telemetry_pkg

    assert not hasattr(telemetry_pkg, "CompressionHint"), (
        "PR-B5: CompressionHint was retired and must not be importable from headroom.telemetry."
    )


# ── Part 2: per-tenant aggregation key ─────────────────────────────────────


def test_aggregation_key_includes_auth_mode_and_model_family():
    """Same structure_hash with different auth_mode/model_family ⇒ distinct patterns."""
    toin = ToolIntelligenceNetwork()
    sig = ToolSignature.from_items([{"id": "1", "score": 99}])

    # Three slices for the same tool signature.
    toin.record_compression(
        tool_signature=sig,
        original_count=10,
        compressed_count=5,
        original_tokens=1000,
        compressed_tokens=500,
        strategy="smart_crusher",
        auth_mode="payg",
        model_family="claude-3-5",
    )
    toin.record_compression(
        tool_signature=sig,
        original_count=10,
        compressed_count=5,
        original_tokens=1000,
        compressed_tokens=500,
        strategy="smart_crusher",
        auth_mode="oauth",
        model_family="claude-3-5",
    )
    toin.record_compression(
        tool_signature=sig,
        original_count=10,
        compressed_count=5,
        original_tokens=1000,
        compressed_tokens=500,
        strategy="smart_crusher",
        auth_mode="payg",
        model_family="gpt-4o",
    )

    sig_hash = sig.structure_hash
    # PR-F3: keys are 4-tuples — `(tenant_key, auth_mode, model_family,
    # sig_hash)`. With no contextvar set the default is "global".
    assert (DEFAULT_TENANT_KEY, "payg", "claude-3-5", sig_hash) in toin._patterns
    assert (DEFAULT_TENANT_KEY, "oauth", "claude-3-5", sig_hash) in toin._patterns
    assert (DEFAULT_TENANT_KEY, "payg", "gpt-4o", sig_hash) in toin._patterns
    # Three distinct slices, each with sample_size=1.
    assert len(toin._patterns) == 3
    for key, pattern in toin._patterns.items():
        assert pattern.tenant_key == key[0]
        assert pattern.auth_mode == key[1]
        assert pattern.model_family == key[2]
        assert pattern.tool_signature_hash == key[3]
        assert pattern.sample_size == 1


def test_aggregation_key_defaults_to_unknown_when_caller_omits_tenant():
    """Callers that don't pass auth_mode/model_family land in the default slice."""
    toin = ToolIntelligenceNetwork()
    sig = ToolSignature.from_items([{"id": "1"}])

    toin.record_compression(
        tool_signature=sig,
        original_count=10,
        compressed_count=5,
        original_tokens=1000,
        compressed_tokens=500,
        strategy="smart_crusher",
    )

    expected_key = (
        DEFAULT_TENANT_KEY,
        DEFAULT_AUTH_MODE,
        DEFAULT_MODEL_FAMILY,
        sig.structure_hash,
    )
    assert expected_key in toin._patterns
    pattern = toin._patterns[expected_key]
    assert pattern.tenant_key == DEFAULT_TENANT_KEY
    assert pattern.auth_mode == DEFAULT_AUTH_MODE
    assert pattern.model_family == DEFAULT_MODEL_FAMILY


def test_storage_round_trip_preserves_aggregation_key(tmp_path: Path):
    """Save/load round-trips the per-tenant aggregation key intact."""
    storage = tmp_path / "toin_roundtrip.json"
    toin1 = ToolIntelligenceNetwork(TOINConfig(storage_path=str(storage)))
    sig = ToolSignature.from_items([{"id": "1"}])

    toin1.record_compression(
        tool_signature=sig,
        original_count=10,
        compressed_count=5,
        original_tokens=1000,
        compressed_tokens=500,
        strategy="smart_crusher",
        auth_mode="oauth",
        model_family="gpt-4o",
        tenant_key="cust_xyz",
    )
    toin1.save()

    toin2 = ToolIntelligenceNetwork(TOINConfig(storage_path=str(storage)))
    key = ("cust_xyz", "oauth", "gpt-4o", sig.structure_hash)
    assert key in toin2._patterns
    assert toin2._patterns[key].tenant_key == "cust_xyz"
    assert toin2._patterns[key].auth_mode == "oauth"
    assert toin2._patterns[key].model_family == "gpt-4o"


# ── PR-F3: per-tenant isolation property ───────────────────────────────────


def test_tenants_do_not_cross_pollinate_patterns():
    """Two tenants compressing identical tool outputs see two isolated pools.

    This is the load-bearing F3 invariant: tenant A's compression
    patterns must be invisible to tenant B's `get_pattern` calls and
    vice-versa. The threat model F3 closes is: a multi-tenant proxy
    that learns a shared pattern pool would cross-pollinate
    compression decisions between distinct customers, and (worse)
    leak structural fingerprints across tenancy boundaries when TOIN
    dumps are shared.
    """
    toin = ToolIntelligenceNetwork()
    sig = ToolSignature.from_items([{"id": "1", "status": "ok"}])
    sig_hash = sig.structure_hash

    # Tenant A: 3 compressions, no retrievals (perfect-compression
    # signal — pattern.skip_compression_recommended will stay False
    # but strategy_success_rate will trend high).
    for _ in range(3):
        toin.record_compression(
            tool_signature=sig,
            original_count=10,
            compressed_count=2,
            original_tokens=1000,
            compressed_tokens=200,
            strategy="smart_crusher",
            tenant_key="tenant_a",
        )

    # Tenant B: 3 compressions PLUS 3 retrievals (over-compressed
    # signal — strategy_success_rate will trend down). If isolation
    # is broken, tenant A's "smart_crusher worked great" signal would
    # contaminate tenant B's pattern.
    for _ in range(3):
        toin.record_compression(
            tool_signature=sig,
            original_count=10,
            compressed_count=2,
            original_tokens=1000,
            compressed_tokens=200,
            strategy="smart_crusher",
            tenant_key="tenant_b",
        )
    for _ in range(3):
        toin.record_retrieval(
            tool_signature_hash=sig_hash,
            retrieval_type="full",
            strategy="smart_crusher",
            tenant_key="tenant_b",
        )

    # Two distinct keys must exist, each with its own dataclass.
    key_a = ("tenant_a", DEFAULT_AUTH_MODE, DEFAULT_MODEL_FAMILY, sig_hash)
    key_b = ("tenant_b", DEFAULT_AUTH_MODE, DEFAULT_MODEL_FAMILY, sig_hash)
    assert key_a in toin._patterns
    assert key_b in toin._patterns

    pattern_a = toin._patterns[key_a]
    pattern_b = toin._patterns[key_b]

    # Compression counts isolated: tenant A saw 3 comps + 0 retrievals.
    assert pattern_a.total_compressions == 3
    assert pattern_a.total_retrievals == 0

    # Tenant B saw 3 comps + 3 retrievals — divergent learning signal.
    assert pattern_b.total_compressions == 3
    assert pattern_b.total_retrievals == 3

    # The success rate diverges: tenant A's "smart_crusher" gets
    # rewarded; tenant B's gets penalized for triggering retrievals.
    rate_a = pattern_a.strategy_success_rates.get("smart_crusher", 0.0)
    rate_b = pattern_b.strategy_success_rates.get("smart_crusher", 0.0)
    assert rate_a > rate_b, (
        f"PR-F3: tenant A and B success rates must diverge (tenant A: {rate_a}, tenant B: {rate_b})"
    )

    # `get_pattern` honors tenant_key — tenant A's lookup returns
    # tenant A's pattern, not tenant B's.
    fetched_a = toin.get_pattern(sig_hash, tenant_key="tenant_a")
    fetched_b = toin.get_pattern(sig_hash, tenant_key="tenant_b")
    assert fetched_a is not None
    assert fetched_b is not None
    assert fetched_a.total_retrievals == 0
    assert fetched_b.total_retrievals == 3


def test_tenant_key_default_is_global_when_no_caller_provides_one():
    """Pre-F3 callers and CLI / batch jobs land in the literal "global" slice.

    Pins the migration story: when no tenant_key is supplied (and no
    contextvar is set), patterns aggregate under the literal
    ``"global"`` namespace, which is exactly where pre-F3 patterns
    live after the legacy 3-tuple → 4-tuple promotion.
    """
    toin = ToolIntelligenceNetwork()
    sig = ToolSignature.from_items([{"id": "1"}])

    toin.record_compression(
        tool_signature=sig,
        original_count=10,
        compressed_count=5,
        original_tokens=1000,
        compressed_tokens=500,
        strategy="smart_crusher",
    )

    keys = list(toin._patterns.keys())
    assert len(keys) == 1
    assert keys[0][0] == DEFAULT_TENANT_KEY == "global"


def test_tenant_key_reads_from_contextvar_when_unspecified():
    """When caller doesn't pass tenant_key, TOIN reads the request ContextVar."""
    from headroom.proxy.tenant_key import set_request_tenant_key

    toin = ToolIntelligenceNetwork()
    sig = ToolSignature.from_items([{"id": "1"}])

    set_request_tenant_key("tenant_from_ctx")
    try:
        toin.record_compression(
            tool_signature=sig,
            original_count=10,
            compressed_count=5,
            original_tokens=1000,
            compressed_tokens=500,
            strategy="smart_crusher",
        )
    finally:
        set_request_tenant_key(None)

    # The contextvar's value made it into the key — not the default
    # "global". This is the actual integration path for the proxy.
    keys = list(toin._patterns.keys())
    assert len(keys) == 1
    assert keys[0][0] == "tenant_from_ctx"


def test_explicit_tenant_key_overrides_contextvar():
    """An explicit tenant_key arg always wins over the ContextVar."""
    from headroom.proxy.tenant_key import set_request_tenant_key

    toin = ToolIntelligenceNetwork()
    sig = ToolSignature.from_items([{"id": "1"}])

    set_request_tenant_key("from_contextvar")
    try:
        toin.record_compression(
            tool_signature=sig,
            original_count=10,
            compressed_count=5,
            original_tokens=1000,
            compressed_tokens=500,
            strategy="smart_crusher",
            tenant_key="explicit_arg",
        )
    finally:
        set_request_tenant_key(None)

    keys = list(toin._patterns.keys())
    assert len(keys) == 1
    assert keys[0][0] == "explicit_arg"


def test_record_does_not_alter_compression_decision():
    """SmartCrusher output is byte-identical regardless of TOIN observation state.

    Calls SmartCrusher twice on the same input — once with TOIN empty,
    once after recording a compression that would have changed the
    pre-B5 hint — and asserts byte equality. This pins the
    observation-only contract: TOIN observes; never mutates.
    """
    smart_crusher_module = pytest.importorskip("headroom.transforms.smart_crusher")
    SmartCrusher = smart_crusher_module.SmartCrusher
    SmartCrusherConfig = smart_crusher_module.SmartCrusherConfig

    cfg = SmartCrusherConfig(
        enabled=True,
        min_items_to_analyze=3,
        min_tokens_to_crush=10,
    )
    crusher = SmartCrusher(config=cfg)

    # 50 low-uniqueness rows so the crusher is willing to compress.
    items = [{"id": i, "status": "ok", "code": 200, "msg": "fine"} for i in range(50)]
    import json as _json

    payload = _json.dumps(items)

    first = crusher.crush(payload)

    # Inject TOIN observations that, pre-B5, would have biased the
    # compressor toward conservative output via get_recommendation().
    toin = ToolIntelligenceNetwork()
    sig = ToolSignature.from_items(items)
    sig_hash = sig.structure_hash
    for _ in range(20):
        toin.record_compression(
            tool_signature=sig,
            original_count=50,
            compressed_count=10,
            original_tokens=1000,
            compressed_tokens=200,
            strategy="smart_crusher",
        )
    for _ in range(15):
        toin.record_retrieval(
            tool_signature_hash=sig_hash,
            retrieval_type="full",
        )

    second = crusher.crush(payload)
    assert first.compressed == second.compressed, (
        "PR-B5: SmartCrusher output must be deterministic regardless of TOIN observation state."
    )


@pytest.mark.parametrize(
    "items",
    [
        # Tiny, mid, and at-threshold inputs covering the conditional
        # paths inside the Rust crusher (lossless tabular, lossy with
        # CCR, pass-through). Spec asks for a hypothesis property test;
        # hypothesis is optional, so we cover the parametrized cases
        # unconditionally and add the property test below behind an
        # importorskip.
        [],
        [{"id": 1}],
        [{"id": i, "status": "ok"} for i in range(8)],
        [{"id": i, "status": "ok", "msg": "fine"} for i in range(50)],
        [{"id": i, "code": 200 + i % 3, "err": ""} for i in range(120)],
    ],
)
def test_smart_crusher_determinism_parametrized(items: list[dict[str, object]]) -> None:
    """Two crush() calls on the same input must return byte-equal output."""
    smart_crusher_module = pytest.importorskip("headroom.transforms.smart_crusher")
    SmartCrusher = smart_crusher_module.SmartCrusher
    SmartCrusherConfig = smart_crusher_module.SmartCrusherConfig
    import json as _json

    crusher = SmartCrusher(config=SmartCrusherConfig(enabled=True))
    payload = _json.dumps(items)
    a = crusher.crush(payload)
    b = crusher.crush(payload)
    assert a.compressed == b.compressed


def test_smart_crusher_determinism_property():
    """Property: any input → byte-stable SmartCrusher output across two calls.

    Skipped if `hypothesis` is not installed (it is not a hard dep of
    Headroom). The parametrized test above covers the deterministic
    surface unconditionally.
    """
    pytest.importorskip("hypothesis")
    from hypothesis import given, settings
    from hypothesis import strategies as st

    smart_crusher_module = pytest.importorskip("headroom.transforms.smart_crusher")
    SmartCrusher = smart_crusher_module.SmartCrusher
    SmartCrusherConfig = smart_crusher_module.SmartCrusherConfig
    crusher = SmartCrusher(config=SmartCrusherConfig(enabled=True))

    @given(
        st.lists(
            st.fixed_dictionaries(
                {
                    "id": st.integers(min_value=0, max_value=10_000),
                    "status": st.sampled_from(["ok", "error", "pending"]),
                }
            ),
            min_size=0,
            max_size=20,
        )
    )
    @settings(max_examples=25, deadline=None)
    def _check(items: list[dict[str, object]]) -> None:
        import json as _json

        payload = _json.dumps(items)
        a = crusher.crush(payload)
        b = crusher.crush(payload)
        assert a.compressed == b.compressed

    _check()
