"""Regression tests for lightweight package bootstrap."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import types
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import patch

import headroom._version as version_module


def test_headroom_import_stays_lazy() -> None:
    script = textwrap.dedent(
        """
        import json
        import sys

        import headroom

        print(json.dumps({
            "version": headroom.__version__,
            "cache_loaded": "headroom.cache" in sys.modules,
            "models_registry_loaded": "headroom.models.registry" in sys.modules,
            "memory_loaded": "headroom.memory" in sys.modules,
        }))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )

    data = json.loads(result.stdout.strip())
    # Version is a non-empty string; don't hardcode a specific value.
    assert isinstance(data["version"], str) and data["version"]
    assert data["cache_loaded"] is False
    assert data["models_registry_loaded"] is False
    assert data["memory_loaded"] is False


def test_version_prefers_installed_distribution_metadata() -> None:
    with (
        patch.object(version_module, "_source_root", return_value=None),
        patch.object(version_module, "version", return_value="9.8.7") as package_version,
    ):
        assert version_module.get_version() == "9.8.7"

    package_version.assert_called_once_with("headroom-ai")


def test_version_reports_unknown_when_distribution_metadata_is_missing() -> None:
    with (
        patch.object(version_module, "_source_root", return_value=None),
        patch.object(version_module, "version", side_effect=PackageNotFoundError),
    ):
        assert version_module.get_version() == version_module.UNKNOWN_VERSION


def test_version_prefers_explicit_build_env(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_BUILD_VERSION", "source-build")

    with patch.object(version_module, "version", return_value="9.8.7") as package_version:
        assert version_module.get_version() == "source-build"

    package_version.assert_not_called()


def test_version_label_helpers_only_prefix_release_versions() -> None:
    assert version_module.is_release_version("0.29.0") is True
    assert version_module.is_release_version("v0.29.0") is True
    assert version_module.normalize_release_version("v0.29.0") == "0.29.0"
    assert version_module.is_release_version("source-build+g6266a1d774b5") is False
    assert version_module.is_release_version("source-build+sha.abcdef123456") is False
    assert version_module.is_release_version("6266a1d") is False
    assert version_module.is_release_version("0.29.0+gabcdef0") is False

    assert version_module.format_version_label("0.29.0") == "v0.29.0"
    assert version_module.format_version_label("v0.29.0") == "v0.29.0"
    assert (
        version_module.format_version_label("source-build+sha.abcdef123456")
        == "source-build+sha.abcdef123456"
    )
    assert (
        version_module.format_version_label("source-build+g6266a1d774b5")
        == "source-build+g6266a1d774b5"
    )
    assert version_module.format_version_label("6266a1d") == "6266a1d"
    assert version_module.format_version_label(None) == version_module.UNKNOWN_VERSION


def test_version_uses_packaged_build_metadata(
    monkeypatch,
) -> None:
    build_info = types.ModuleType("headroom._build_info")
    build_info.BUILD_VERSION = "0.29.0+gabcdef0"
    monkeypatch.setitem(sys.modules, "headroom._build_info", build_info)

    with (
        patch.object(version_module, "_source_root", return_value=None),
        patch.object(version_module, "version", return_value="0.29.0") as package_version,
    ):
        assert version_module.get_version() == "0.29.0+gabcdef0"

    package_version.assert_not_called()


def test_observability_version_uses_runtime_version(monkeypatch) -> None:
    from headroom.observability import metrics as metrics_module

    monkeypatch.setattr(
        metrics_module,
        "get_version",
        lambda: "source-build+sha.abcdef123456",
    )

    assert metrics_module._headroom_version() == "source-build+sha.abcdef123456"


def test_version_prefers_source_tree_release_history() -> None:
    with (
        patch.object(version_module, "_source_root", return_value=Path(".")),
        patch.object(version_module, "_source_tree_version", return_value="0.21.17"),
        patch.object(version_module, "version", return_value="0.9.1") as package_version,
    ):
        # Source checkouts are marked -dev so a dev build is never mistaken
        # for the published release.
        assert version_module.get_version() == "0.21.17-dev"

    package_version.assert_not_called()


def test_proxy_package_import_does_not_eagerly_load_server() -> None:
    script = textwrap.dedent(
        """
        import json
        import sys

        import headroom.proxy

        print(json.dumps({
            "server_loaded": "headroom.proxy.server" in sys.modules,
        }))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )

    data = json.loads(result.stdout.strip())
    assert data["server_loaded"] is False


def test_codex_package_import_stays_runtime_only() -> None:
    script = textwrap.dedent(
        """
        import json
        import sys

        import headroom.providers.codex

        print(json.dumps({
            "images_loaded": "headroom.providers.codex.images" in sys.modules,
            "model_metadata_loaded": "headroom.providers.codex.model_metadata" in sys.modules,
            "responses_loaded": "headroom.providers.codex.responses" in sys.modules,
        }))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )

    data = json.loads(result.stdout.strip())
    assert data == {
        "images_loaded": False,
        "model_metadata_loaded": False,
        "responses_loaded": False,
    }


def test_proxy_server_import_skips_litellm_backend() -> None:
    script = textwrap.dedent(
        """
        import json
        import sys

        import headroom.proxy.server

        print(json.dumps({
            "litellm_backend_loaded": "headroom.backends.litellm" in sys.modules,
            "anyllm_backend_loaded": "headroom.backends.anyllm" in sys.modules,
            "litellm_loaded": "litellm" in sys.modules,
        }))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )

    data = json.loads(result.stdout.strip())
    assert data["litellm_backend_loaded"] is False
    assert data["anyllm_backend_loaded"] is False
    assert data["litellm_loaded"] is False


def test_dynamic_detector_import_skips_optional_ml_dependencies(tmp_path: Path) -> None:
    (tmp_path / "spacy.py").write_text("", encoding="utf-8")
    (tmp_path / "numpy.py").write_text("", encoding="utf-8")
    (tmp_path / "torch.py").write_text("", encoding="utf-8")
    sentence_transformers_dir = tmp_path / "sentence_transformers"
    sentence_transformers_dir.mkdir()
    (sentence_transformers_dir / "__init__.py").write_text(
        "import torch\n\nclass SentenceTransformer:\n    pass\n",
        encoding="utf-8",
    )

    script = textwrap.dedent(
        """
        import json
        import sys

        import headroom.cache.dynamic_detector

        print(json.dumps({
            "spacy_loaded": "spacy" in sys.modules,
            "sentence_transformers_loaded": "sentence_transformers" in sys.modules,
            "torch_loaded": "torch" in sys.modules,
        }))
        """
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
    )

    data = json.loads(result.stdout.strip())
    assert data["spacy_loaded"] is False
    assert data["sentence_transformers_loaded"] is False
    assert data["torch_loaded"] is False


def test_compress_spreadsheet_public_import_survives_ort_pin() -> None:
    """`from headroom import compress_spreadsheet` stays eagerly exported, and the
    Windows ORT dylib pin still runs before the `.compress` import.

    The pin (`ensure_ort_dylib_pinned`) was inserted above the eager `.compress`
    import; restoring `compress_spreadsheet` to that line must not reorder it
    relative to the pin. The `__dict__` check distinguishes the eager import from
    the lazy `_LAZY_EXPORTS` fallback, which would also resolve the name.
    """
    script = textwrap.dedent(
        """
        import json

        import headroom
        from headroom import compress_spreadsheet

        print(json.dumps({
            "eager": "compress_spreadsheet" in headroom.__dict__,
            "callable": callable(compress_spreadsheet),
        }))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )

    data = json.loads(result.stdout.strip())
    assert data["eager"] is True
    assert data["callable"] is True

    # ORT pin must precede the `.compress` import, which must still list the helper.
    src = (Path(version_module.__file__).parent / "__init__.py").read_text(encoding="utf-8")
    pin = src.index("ensure_ort_dylib_pinned()")
    compress_import = src.index("from .compress import")
    assert pin < compress_import
    assert "compress_spreadsheet" in src[compress_import : compress_import + 120]
