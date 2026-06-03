"""Regression tests for lightweight package bootstrap."""

from __future__ import annotations

import json
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


def test_proxy_package_import_stays_lazy() -> None:
    """Importing ``headroom.proxy`` must not pull in the fastapi-backed server.

    Regression for issue #441: the eager ``from .server import ...`` in
    ``headroom/proxy/__init__.py`` made ``headroom --help`` crash with
    ``ModuleNotFoundError: No module named 'fastapi'`` on base installs without
    the ``[proxy]`` extra, because loading ``headroom.cli.proxy`` imports the
    dependency-free ``headroom.proxy.modes`` and that triggers the package
    ``__init__``.
    """
    script = textwrap.dedent(
        """
        import json
        import sys

        import headroom.proxy
        import headroom.proxy.modes  # what the CLI actually needs

        print(json.dumps({
            "server_loaded": "headroom.proxy.server" in sys.modules,
            "fastapi_loaded": "fastapi" in sys.modules,
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
    assert data["fastapi_loaded"] is False


def test_proxy_package_exports_remain_accessible() -> None:
    """The lazy ``__getattr__`` must still expose ``create_app``/``run_server``."""
    script = textwrap.dedent(
        """
        import sys

        import headroom.proxy

        create_app = headroom.proxy.create_app
        run_server = headroom.proxy.run_server
        assert callable(create_app)
        assert callable(run_server)
        # Accessing the exports must have loaded the server module on demand.
        assert "headroom.proxy.server" in sys.modules

        # Unknown attributes still raise AttributeError.
        try:
            headroom.proxy.does_not_exist
        except AttributeError:
            pass
        else:
            raise AssertionError("expected AttributeError for unknown attribute")

        print("ok")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "ok"


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
