"""Tests for HuggingFace model-revision pinning (supply-chain integrity).

Model artifacts are pinned to immutable commit SHAs so a changed or compromised
upstream repo cannot be pulled silently. Pinning is centralized in the download
helper so every call site (kompress, memory embedder, image router) inherits it.
"""

from __future__ import annotations

import pytest

from headroom.onnx_runtime import _PINNED_REVISIONS, _resolve_revision


def test_known_repos_are_pinned_to_sha():
    # Every pinned revision must be a full 40-char git SHA, not a branch/tag.
    assert _PINNED_REVISIONS, "expected at least one pinned model repo"
    for repo, sha in _PINNED_REVISIONS.items():
        assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha), (
            f"{repo} is not pinned to a full commit SHA: {sha!r}"
        )


def test_default_kompress_model_is_pinned():
    # The shipping model must be pinned.
    assert "chopratejas/kompress-v2-base" in _PINNED_REVISIONS


def test_resolve_uses_pin_for_known_repo(monkeypatch):
    monkeypatch.delenv("HEADROOM_HF_PIN", raising=False)
    repo = "chopratejas/kompress-v2-base"
    assert _resolve_revision(repo, None) == _PINNED_REVISIONS[repo]


def test_explicit_revision_overrides_pin(monkeypatch):
    monkeypatch.delenv("HEADROOM_HF_PIN", raising=False)
    assert _resolve_revision("chopratejas/kompress-v2-base", "deadbeef") == "deadbeef"


def test_unknown_repo_is_not_pinned(monkeypatch):
    monkeypatch.delenv("HEADROOM_HF_PIN", raising=False)
    assert _resolve_revision("some/unknown-model", None) is None


@pytest.mark.parametrize("value", ["off", "0", "false", "no", "OFF"])
def test_pin_can_be_disabled_via_env(monkeypatch, value):
    monkeypatch.setenv("HEADROOM_HF_PIN", value)
    assert _resolve_revision("chopratejas/kompress-v2-base", None) is None


def test_pin_disabled_still_respects_explicit_revision(monkeypatch):
    monkeypatch.setenv("HEADROOM_HF_PIN", "off")
    assert _resolve_revision("chopratejas/kompress-v2-base", "abc123") == "abc123"


def test_modernbert_tokenizer_base_is_pinned():
    # The Kompress tokenizer (and torch-path encoder) load from this repo; the
    # ONNX export was produced against the pinned snapshot, so it must not float.
    assert "answerdotai/ModernBERT-base" in _PINNED_REVISIONS


def test_tokenizer_loader_passes_pinned_revision(monkeypatch):
    monkeypatch.delenv("HEADROOM_HF_PIN", raising=False)
    from headroom.transforms import kompress_compressor as kc

    calls: list[dict] = []

    class _Tok:
        @staticmethod
        def from_pretrained(repo, **kwargs):
            calls.append({"repo": repo, **kwargs})
            return object()

    kc._load_modernbert_tokenizer(_Tok, allow_download=True)
    assert calls == [
        {
            "repo": "answerdotai/ModernBERT-base",
            "revision": _PINNED_REVISIONS["answerdotai/ModernBERT-base"],
            "local_files_only": True,
        }
    ]


def test_tokenizer_loader_downloads_at_pin_on_cache_miss(monkeypatch):
    monkeypatch.delenv("HEADROOM_HF_PIN", raising=False)
    from headroom.transforms import kompress_compressor as kc

    calls: list[dict] = []

    class _Tok:
        @staticmethod
        def from_pretrained(repo, **kwargs):
            calls.append({"repo": repo, **kwargs})
            if kwargs.get("local_files_only"):
                raise OSError("not cached")
            return object()

    kc._load_modernbert_tokenizer(_Tok, allow_download=True)
    assert [c["local_files_only"] for c in calls] == [True, False]
    assert {c["revision"] for c in calls} == {_PINNED_REVISIONS["answerdotai/ModernBERT-base"]}


def test_tokenizer_loader_floats_when_pin_disabled(monkeypatch):
    monkeypatch.setenv("HEADROOM_HF_PIN", "off")
    from headroom.transforms import kompress_compressor as kc

    calls: list[dict] = []

    class _Tok:
        @staticmethod
        def from_pretrained(repo, **kwargs):
            calls.append(kwargs)
            return object()

    kc._load_modernbert_tokenizer(_Tok, allow_download=True)
    assert calls[0]["revision"] is None


def test_cache_only_pytorch_load_does_not_fetch_encoder(monkeypatch):
    # A cache-only load must not download the pinned encoder on the request
    # path; a missing snapshot defers the load like any other cache miss.
    import sys
    import types

    from headroom.transforms import kompress_compressor as kc

    calls: list[dict] = []

    class _AutoModel:
        @staticmethod
        def from_pretrained(repo, **kwargs):
            calls.append(kwargs)
            if kwargs.get("local_files_only"):
                raise OSError("pinned encoder snapshot not cached")
            raise AssertionError("cache-only load reached the network")

    nn = types.SimpleNamespace(Module=object)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(nn=nn))
    monkeypatch.setitem(sys.modules, "torch.nn", nn)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoModel=_AutoModel, AutoTokenizer=object),
    )
    monkeypatch.setattr(kc, "_kompress_cache", {})

    with pytest.raises(kc.KompressModelNotCached):
        kc._load_kompress_pytorch("chopratejas/kompress-v2-base", allow_download=False)
    assert [c["local_files_only"] for c in calls] == [True]


# ---------------------------------------------------------------------------
# Tokenizer allowlist pins (GH #4939)
#
# Every repository `_load_tokenizer` is allowed to fetch must name an immutable
# commit, or the same model id can be counted with a different vocabulary after
# an upstream push. The table lives beside the allowlist so the two cannot drift.
# ---------------------------------------------------------------------------


def test_every_shipped_tokenizer_repo_is_pinned():
    from headroom.tokenizers import huggingface as hf_mod

    pins = hf_mod._TOKENIZER_REVISIONS
    shipped = hf_mod.ALLOWED_TOKENIZER_REPOS
    # mpt-7b/mpt-30b are gone from the Hub (401/404): nothing to pin, and they
    # fail closed to estimation either way. Everything else must be pinned.
    unpinned = sorted(repo for repo in shipped if repo not in pins)
    assert unpinned == ["mosaicml/mpt-30b", "mosaicml/mpt-7b"], (
        f"a shipped tokenizer repository has no pinned revision: {unpinned}"
    )


def test_every_tokenizer_pin_is_a_full_sha():
    from headroom.tokenizers import huggingface as hf_mod

    assert hf_mod._TOKENIZER_REVISIONS
    for repo, sha in hf_mod._TOKENIZER_REVISIONS.items():
        assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha), (
            f"{repo} is not pinned to a full commit SHA: {sha!r}"
        )


def test_no_tokenizer_pin_is_for_an_unshipped_repo():
    from headroom.tokenizers import huggingface as hf_mod

    stale = sorted(set(hf_mod._TOKENIZER_REVISIONS) - hf_mod.ALLOWED_TOKENIZER_REPOS)
    assert not stale, f"pinned but not shipped (delete the pin): {stale}"


def test_tokenizer_loader_fetches_the_pin_on_both_attempts(monkeypatch):
    monkeypatch.delenv("HEADROOM_HF_PIN", raising=False)
    monkeypatch.delenv("HEADROOM_HF_TOKENIZER_REVISIONS", raising=False)
    monkeypatch.setenv("HEADROOM_HF_TOKENIZER_LOAD_TIMEOUT_SECS", "5")
    import sys
    import types

    from headroom.tokenizers import huggingface as hf_mod

    calls: list[dict] = []

    class _Auto:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append({"name": name, **kwargs})
            if kwargs.get("local_files_only"):
                raise OSError("not cached")
            return object()

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=_Auto))
    hf_mod._load_tokenizer.cache_clear()
    try:
        assert hf_mod._load_tokenizer("Qwen/Qwen2.5-7B") is not None
    finally:
        hf_mod._load_tokenizer.cache_clear()

    pin = hf_mod._TOKENIZER_REVISIONS["Qwen/Qwen2.5-7B"]
    assert [bool(c.get("local_files_only")) for c in calls] == [True, False]
    assert {c["revision"] for c in calls} == {pin}, (
        "the cache-only attempt must use the pin too: a snapshot seeded at "
        "`main` is a different revision"
    )


def test_tokenizer_loader_floats_when_pinning_is_off(monkeypatch):
    """`HEADROOM_HF_PIN=off` floats the tokenizer fetch, not just the ONNX ones."""
    monkeypatch.setenv("HEADROOM_HF_PIN", "off")
    monkeypatch.delenv("HEADROOM_HF_TOKENIZER_REVISIONS", raising=False)
    import sys
    import types

    from headroom.tokenizers import huggingface as hf_mod

    calls: list[dict] = []

    class _Auto:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append(kwargs)
            return object()

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=_Auto))
    hf_mod._load_tokenizer.cache_clear()
    try:
        assert hf_mod._load_tokenizer("Qwen/Qwen2.5-7B") is not None
    finally:
        hf_mod._load_tokenizer.cache_clear()

    assert calls[0]["revision"] is None


def test_operator_revision_pins_an_env_allowlisted_repo(monkeypatch):
    from headroom.tokenizers import huggingface as hf_mod

    monkeypatch.delenv("HEADROOM_HF_PIN", raising=False)
    monkeypatch.setenv(hf_mod._ALLOWLIST_ENV, "acme/internal-tokenizer")
    monkeypatch.setenv(
        hf_mod._REVISIONS_ENV,
        "acme/internal-tokenizer=0123456789abcdef0123456789abcdef01234567",
    )
    assert (
        hf_mod._revision_for("acme/internal-tokenizer")
        == "0123456789abcdef0123456789abcdef01234567"
    )


def test_operator_revision_ignores_malformed_pairs(monkeypatch):
    from headroom.tokenizers import huggingface as hf_mod

    monkeypatch.delenv("HEADROOM_HF_PIN", raising=False)
    monkeypatch.setenv(hf_mod._REVISIONS_ENV, "no-equals-sign, =orphan, repo=")
    assert hf_mod._operator_revisions(hf_mod.os.environ[hf_mod._REVISIONS_ENV]) == {}


def test_default_tokenizer_revision_is_resolved(monkeypatch):
    from headroom.tokenizers import huggingface as hf_mod

    monkeypatch.delenv("HEADROOM_HF_PIN", raising=False)
    monkeypatch.delenv(hf_mod._REVISIONS_ENV, raising=False)
    assert (
        hf_mod._revision_for(hf_mod.DEFAULT_TOKENIZER)
        == (hf_mod._TOKENIZER_REVISIONS[hf_mod.DEFAULT_TOKENIZER])
    )


# ---------------------------------------------------------------------------
# torch-path loaders (MLModelRegistry: SigLIP, sentence-transformers, router)
#
# They funnel through `hf_from_pretrained_local_first`, which is where the pin
# is applied. Covered here rather than per-loader so a fourth loader added
# tomorrow inherits the pin with no test edit.
# ---------------------------------------------------------------------------


def test_local_first_loader_passes_the_pin_to_the_loader(monkeypatch):
    monkeypatch.delenv("HEADROOM_HF_PIN", raising=False)
    from headroom.onnx_runtime import hf_from_pretrained_local_first

    calls: list[dict] = []

    def loader(name, **kwargs):
        calls.append(kwargs)
        return object()

    hf_from_pretrained_local_first(loader, "google/siglip-base-patch16-224", purpose="SigLIP")
    assert calls[0]["revision"] == _PINNED_REVISIONS["google/siglip-base-patch16-224"]
    assert calls[0]["local_files_only"] is True


def test_local_first_loader_omits_revision_for_a_local_directory(monkeypatch, tmp_path):
    """A path has no revision; passing one to a loader pointed at it is at best ignored."""
    monkeypatch.delenv("HEADROOM_HF_PIN", raising=False)
    from headroom.onnx_runtime import hf_from_pretrained_local_first

    calls: list[dict] = []

    def loader(name, **kwargs):
        calls.append(kwargs)
        return object()

    hf_from_pretrained_local_first(loader, str(tmp_path), purpose="local model")
    assert "revision" not in calls[0]


def test_local_first_loader_floats_an_unknown_repo(monkeypatch):
    monkeypatch.delenv("HEADROOM_HF_PIN", raising=False)
    from headroom.onnx_runtime import hf_from_pretrained_local_first

    calls: list[dict] = []
    hf_from_pretrained_local_first(
        lambda name, **kw: calls.append(kw) or object(), "acme/unknown", purpose="x"
    )
    assert calls[0]["revision"] is None


def test_sentence_transformer_default_is_pinned(monkeypatch):
    """The ST default resolves through the bare alias, so pin the alias too."""
    monkeypatch.delenv("HEADROOM_HF_PIN", raising=False)
    from headroom.models.config import ML_MODEL_DEFAULTS
    from headroom.onnx_runtime import _resolve_revision

    assert _resolve_revision(ML_MODEL_DEFAULTS.sentence_transformer, None) is not None


# ---------------------------------------------------------------------------
# Offline upgrade path (GH #4939 review)
#
# A deployment that cached the *default-branch* snapshot before the pin existed
# has the old bytes under its refs, not the pin. Pinning makes that a cache
# miss, so the loader falls back to estimation — silently, once, for the life of
# the process (lru_cache). Correct by design: serving the old vocabulary would
# defeat the pin and quietly change every count. But the operator has to be told
# to pre-seed the SHA first, so this test locks the behaviour in place.
# ---------------------------------------------------------------------------


def test_offline_tolerates_only_the_pinned_snapshot(monkeypatch):
    """An old default-branch snapshot is a miss under a pin, and refusing is the point."""
    monkeypatch.delenv("HEADROOM_HF_TOKENIZER_REVISIONS", raising=False)
    monkeypatch.delenv("HEADROOM_HF_PIN", raising=False)
    import sys
    import types

    from headroom.tokenizers import huggingface as hf_mod

    pin = hf_mod._TOKENIZER_REVISIONS["Qwen/Qwen2.5-7B"]
    calls: list[dict] = []

    class _Auto:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append(kwargs)
            # The cache holds the default branch only: the rev the operator
            # seeded long ago is not the pin, so asking for the pin misses.
            raise OSError("Requested revision not found in cache")

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=_Auto))
    monkeypatch.setenv("HEADROOM_OFFLINE", "1")
    hf_mod._load_tokenizer.cache_clear()
    try:
        # The old snapshot is a miss, and offline means no network fallback, so
        # the answer is None (estimation), never the old vocabulary.
        assert hf_mod._load_tokenizer("Qwen/Qwen2.5-7B") is None
        assert [c["revision"] for c in calls] == [pin]
        assert calls[0]["local_files_only"] is True
    finally:
        hf_mod._load_tokenizer.cache_clear()


def test_a_cache_seeded_at_the_pin_is_used_offline(monkeypatch):
    """The documented fix: pre-seed the SHA, and an offline box keeps real counts."""
    monkeypatch.delenv("HEADROOM_HF_TOKENIZER_REVISIONS", raising=False)
    monkeypatch.delenv("HEADROOM_HF_PIN", raising=False)
    import sys
    import types

    from headroom.tokenizers import huggingface as hf_mod

    pin = hf_mod._TOKENIZER_REVISIONS["Qwen/Qwen2.5-7B"]
    calls: list[dict] = []

    class _Auto:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append(kwargs)
            if kwargs.get("revision") != pin:
                raise OSError("only the pin is cached")
            return "pinned-tokenizer"

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=_Auto))
    monkeypatch.setenv("HEADROOM_OFFLINE", "1")
    hf_mod._load_tokenizer.cache_clear()
    try:
        assert hf_mod._load_tokenizer("Qwen/Qwen2.5-7B") == "pinned-tokenizer"
    finally:
        hf_mod._load_tokenizer.cache_clear()

    assert calls == [{"revision": pin, "trust_remote_code": False, "local_files_only": True}]
