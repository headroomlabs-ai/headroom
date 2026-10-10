"""The default relevance embedder loads one pinned snapshot, in both languages.

fastembed's ``TextEmbedding`` accepts ``revision`` but drops it before
``snapshot_download``, so the old pin never reached the Hub and the model
floated on ``main``. These tests hold the replacement in place: the pinned
SHA reaches every file download, fastembed only gets the resolved directory,
and the Rust scorer names the same repo, SHA and files.
"""

from __future__ import annotations

import json
import os
import re
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

REPO_ROOT = Path(__file__).resolve().parents[1]
RUST_EMBEDDING = REPO_ROOT / "crates/headroom-core/src/relevance/embedding.rs"
PARITY_FIXTURE = REPO_ROOT / "tests/parity/fixtures/embedding/bge_small_en_v15_pinned.json"
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
    with pytest.raises(RuntimeError, match="is unavailable"):
        EmbeddingScorer()._get_model()
    assert fake_fastembed == []


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


def _rust_const(name: str) -> str:
    match = re.search(rf'const {name}: &str = "([^"]+)";', RUST_EMBEDDING.read_text())
    assert match, f"{name} not found in {RUST_EMBEDDING}"
    return match.group(1)


def test_rust_loads_the_same_snapshot():
    assert _rust_const("DEFAULT_MODEL_REPO") == DEFAULT_MODEL_REPO
    assert _rust_const("DEFAULT_MODEL_REVISION") == PIN
    assert _rust_const("DEFAULT_MODEL_FILE") == DEFAULT_MODEL_FILES[0]
    files = re.search(
        r"const DEFAULT_MODEL_FILES: \[&str; \d+\] = \[(.*?)\];",
        RUST_EMBEDDING.read_text(),
        re.S,
    )
    assert files, "DEFAULT_MODEL_FILES not found"
    entries = [part.strip() for part in files.group(1).split(",") if part.strip()]
    assert entries == ["DEFAULT_MODEL_FILE", *(f'"{name}"' for name in DEFAULT_MODEL_FILES[1:])]


def test_parity_fixture_was_recorded_at_the_pin():
    fixture = json.loads(PARITY_FIXTURE.read_text(encoding="utf-8"))
    assert fixture["repo"] == DEFAULT_MODEL_REPO
    assert fixture["revision"] == PIN
    assert len(fixture["embeddings"]) == len(fixture["texts"])


@pytest.mark.skipif(
    not os.environ.get("RUN_FASTEMBED_TESTS"),
    reason="loads the real model (~67 MB download on a cold cache); set RUN_FASTEMBED_TESTS=1",
)
def test_python_matches_parity_fixture():
    """Python half of the cross-language check; the Rust half is
    ``fastembed_matches_parity_fixture`` in the Rust scorer."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("fastembed")
    fixture = json.loads(PARITY_FIXTURE.read_text(encoding="utf-8"))

    got = EmbeddingScorer()._encode(fixture["texts"])

    for text, row, expected in zip(fixture["texts"], got, fixture["embeddings"]):
        worst = float(np.abs(np.asarray(row, dtype=np.float64) - np.asarray(expected)).max())
        assert worst <= fixture["max_abs_diff"], f"{text!r}: max |python - fixture| = {worst:e}"
