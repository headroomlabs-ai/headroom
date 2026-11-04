"""The default relevance embedder loads one pinned snapshot.

fastembed's ``TextEmbedding`` accepts ``revision`` but drops it before
``snapshot_download``, so the old pin never reached the Hub and the model
floated on ``main``. These tests hold the replacement in place: the pinned
SHA reaches every file download and fastembed only gets the resolved
directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from headroom.offline import OFFLINE_ENV
from headroom.onnx_runtime import _PINNED_REVISIONS
from headroom.relevance import embedding
from headroom.relevance.embedding import (
    DEFAULT_MODEL_FILES,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REPO,
    EmbeddingScorer,
)

PIN = _PINNED_REVISIONS[DEFAULT_MODEL_REPO]


@pytest.fixture
def fake_hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict[str, object]]:
    """Stand in for ``huggingface_hub.hf_hub_download``: record each call and
    return the path the real cache layout would give for that revision."""
    import huggingface_hub

    calls: list[dict[str, object]] = []

    def _download(repo_id: str, filename: str, **kwargs: object) -> str:
        calls.append({"repo_id": repo_id, "filename": filename, **kwargs})
        commit = kwargs.get("revision") or "main-commit"
        return str(tmp_path / "snapshots" / str(commit) / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)
    monkeypatch.delenv("HEADROOM_HF_PIN", raising=False)
    return calls


@pytest.fixture
def fake_fastembed(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Install a stub fastembed whose constructor records its kwargs."""
    constructed: list[dict[str, object]] = []

    class _TextEmbedding:
        def __init__(self, **kwargs: object) -> None:
            constructed.append(kwargs)

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=_TextEmbedding))
    return constructed


def test_every_file_is_downloaded_at_the_pinned_revision(fake_hub, fake_fastembed, tmp_path):
    EmbeddingScorer()._get_model()

    assert {c["filename"] for c in fake_hub} == set(DEFAULT_MODEL_FILES)
    assert {c["repo_id"] for c in fake_hub} == {DEFAULT_MODEL_REPO}
    assert {c["revision"] for c in fake_hub} == {PIN}


def test_fastembed_gets_the_snapshot_directory_not_a_revision(fake_hub, fake_fastembed, tmp_path):
    EmbeddingScorer()._get_model()

    assert fake_fastembed == [
        {
            "model_name": DEFAULT_MODEL_NAME,
            "specific_model_path": str(tmp_path / "snapshots" / PIN),
        }
    ]


def test_fastembed_is_constructed_with_the_hub_forced_offline(fake_hub, monkeypatch):
    """If fastembed ever stopped honouring ``specific_model_path`` it would
    try the Hub; forcing it offline makes that a load failure, not an
    unpinned download."""
    from huggingface_hub import constants as hf_constants

    seen: list[tuple[str | None, bool]] = []

    class _TextEmbedding:
        def __init__(self, **kwargs: object) -> None:
            seen.append((os.environ.get("HF_HUB_OFFLINE"), hf_constants.HF_HUB_OFFLINE))

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=_TextEmbedding))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", False)

    EmbeddingScorer()._get_model()

    assert seen == [("1", True)]
    assert os.environ["HF_HUB_OFFLINE"] == "0"
    assert hf_constants.HF_HUB_OFFLINE is False


def test_pin_off_floats_on_main(fake_hub, fake_fastembed, monkeypatch):
    monkeypatch.setenv("HEADROOM_HF_PIN", "off")
    EmbeddingScorer()._get_model()

    assert {c["revision"] for c in fake_hub} == {None}


def test_files_from_two_snapshots_are_refused(monkeypatch, fake_fastembed, tmp_path):
    import huggingface_hub

    def _download(repo_id: str, filename: str, **kwargs: object) -> str:
        commit = "a" if filename == DEFAULT_MODEL_FILES[0] else "b"
        return str(tmp_path / commit / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)
    with pytest.raises(RuntimeError, match="more than one snapshot"):
        EmbeddingScorer()._get_model()
    assert fake_fastembed == []


def test_a_cold_cache_offline_degrades_without_a_download(monkeypatch, fake_fastembed):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    def _download(repo_id: str, filename: str, **kwargs: object) -> str:
        if not kwargs.get("local_files_only"):
            raise AssertionError("the network fallback ran while offline")
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)
    monkeypatch.setenv(OFFLINE_ENV, "1")
    with pytest.raises(RuntimeError, match="is unavailable") as excinfo:
        EmbeddingScorer()._get_model()
    assert fake_fastembed == []
    # The hint names the exact snapshot to seed, not the fastembed cache an
    # older Headroom read.
    assert f"{DEFAULT_MODEL_REPO}@{PIN}" in str(excinfo.value)
    assert f"--revision {PIN}" in str(excinfo.value)


def test_other_models_keep_fastembed_resolution(monkeypatch, fake_hub):
    seen: list[dict[str, str]] = []
    monkeypatch.setattr(embedding, "_load_text_embedding", lambda kwargs: seen.append(kwargs))
    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace())

    EmbeddingScorer("intfloat/e5-small-v2")._get_model()

    assert seen == [{"model_name": "intfloat/e5-small-v2"}]
    assert fake_hub == []


def test_fastembed_loads_specific_model_path_without_a_download(monkeypatch, tmp_path):
    """The contract the fix rests on, against the real fastembed: given
    ``specific_model_path`` it uses that directory and never reaches the Hub."""
    pytest.importorskip("fastembed")
    import fastembed.common.model_management as model_management
    from fastembed import TextEmbedding

    def _no_hub(*args: object, **kwargs: object) -> None:
        raise AssertionError("fastembed reached the Hub despite specific_model_path")

    monkeypatch.setattr(model_management, "snapshot_download", _no_hub)
    model = TextEmbedding(
        model_name=DEFAULT_MODEL_NAME,
        specific_model_path=str(tmp_path),
        cache_dir=str(tmp_path / "unused-cache"),
        lazy_load=True,
    )
    assert Path(model.model._model_dir) == tmp_path
