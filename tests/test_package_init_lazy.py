"""Regression tests for lightweight package bootstrap."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
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


def test_version_prefers_source_tree_release_history() -> None:
    with (
        patch.object(version_module, "_source_root", return_value=Path(".")),
        patch.object(version_module, "_source_tree_version", return_value="0.21.17"),
        patch.object(version_module, "version", return_value="0.9.1") as package_version,
    ):
        assert version_module.get_version() == "0.21.17"

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
