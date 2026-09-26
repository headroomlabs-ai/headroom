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
