"""Regression tests for lazy import behaviour of heavyweight ML modules.

Two memory wins are covered here:

1. ``headroom.image.trained_router`` must not pull ``torch`` (300-500 MB) into
   ``sys.modules`` at import time. Workers that never touch the trained image
   router (the default ONNX-handled path) therefore never pay the cost.

2. ``headroom.transforms.content_router.eager_load_compressors`` must not
   eagerly load all 8 tree-sitter language parsers (~150-300 MB). Parsers are
   loaded lazily per thread on first use, with opt-in preload via the
   ``HEADROOM_TREE_SITTER_PRELOAD`` env var.

The torch test runs in a subprocess so we get a clean interpreter where
``sys.modules`` is not already polluted by other tests.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Problem 1: torch must not be imported by `headroom.image.trained_router`
# ---------------------------------------------------------------------------


def test_importing_trained_router_does_not_import_torch() -> None:
    """Importing trained_router must NOT pull torch into sys.modules.

    This is checked in a subprocess so unrelated test fixtures cannot
    pre-populate ``sys.modules`` and mask a regression.
    """
    script = textwrap.dedent(
        """
        import sys

        # Sanity: torch is not in sys.modules in a fresh interpreter
        assert "torch" not in sys.modules, (
            f"torch already in sys.modules before our import: "
            f"{[m for m in sys.modules if m.startswith('torch')]}"
        )

        import headroom.image.trained_router  # noqa: F401

        offenders = sorted(
            m for m in sys.modules if m == "torch" or m.startswith("torch.")
        )
        assert not offenders, (
            "Importing trained_router loaded torch eagerly: " + ", ".join(offenders)
        )

        # transformers and PIL are equally heavy; same rule applies.
        for forbidden in ("transformers", "PIL"):
            leaked = sorted(
                m for m in sys.modules if m == forbidden or m.startswith(forbidden + ".")
            )
            assert not leaked, (
                f"Importing trained_router loaded {forbidden} eagerly: " + ", ".join(leaked)
            )

        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, (
        f"Subprocess failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "OK" in result.stdout


def test_image_ml_available_probe_does_not_import_torch() -> None:
    """``_image_ml_available()`` must use find_spec, not actually import torch."""
    script = textwrap.dedent(
        """
        import sys
        from headroom.image.trained_router import _image_ml_available

        _image_ml_available()  # the probe itself
        offenders = sorted(
            m for m in sys.modules if m == "torch" or m.startswith("torch.")
        )
        assert not offenders, (
            "_image_ml_available() leaked torch into sys.modules: " + ", ".join(offenders)
        )
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, (
        f"Subprocess failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# Problem 2: tree-sitter parsers must not be eagerly loaded by eager_load
# ---------------------------------------------------------------------------


try:
    import tree_sitter_language_pack  # noqa: F401

    TREE_SITTER_INSTALLED = True
except ImportError:
    TREE_SITTER_INSTALLED = False


@pytest.mark.skipif(
    not TREE_SITTER_INSTALLED,
    reason="tree-sitter-language-pack not installed",
)
def test_eager_load_compressors_does_not_preload_all_parsers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``eager_load_compressors`` must not load all 8 language parsers.

    The legacy implementation iterated over python/js/ts/go/rust/java/c/cpp
    and called ``_get_parser`` for each one, costing ~150-300 MB of RSS per
    worker. The new behaviour only loads what the operator opts into via
    ``HEADROOM_TREE_SITTER_PRELOAD``.
    """
    from headroom.transforms import code_compressor, content_router

    # Make sure no preload env var leaks into this test
    monkeypatch.delenv("HEADROOM_TREE_SITTER_PRELOAD", raising=False)

    # Reset per-thread parser cache so we can observe what gets loaded.
    if hasattr(code_compressor._tree_sitter_local, "parsers"):
        code_compressor._tree_sitter_local.parsers = {}

    router = content_router.ContentRouter(
        content_router.ContentRouterConfig(enable_code_aware=True),
    )
    status = router.eager_load_compressors()

    parsers = getattr(code_compressor._tree_sitter_local, "parsers", {}) or {}
    assert len(parsers) == 0, "Eager load must not pre-instantiate parsers; got " + ", ".join(
        sorted(parsers.keys())
    )
    # The status should reflect the lazy posture, not a "loaded (N languages)"
    # claim.
    assert status.get("tree_sitter") in {"lazy (load on first use)", "not installed"}


@pytest.mark.skipif(
    not TREE_SITTER_INSTALLED,
    reason="tree-sitter-language-pack not installed",
)
def test_eager_load_respects_opt_in_preload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operators can still opt into preload via HEADROOM_TREE_SITTER_PRELOAD."""
    from headroom.transforms import code_compressor, content_router

    monkeypatch.setenv("HEADROOM_TREE_SITTER_PRELOAD", "python")

    if hasattr(code_compressor._tree_sitter_local, "parsers"):
        code_compressor._tree_sitter_local.parsers = {}

    router = content_router.ContentRouter(
        content_router.ContentRouterConfig(enable_code_aware=True),
    )
    status = router.eager_load_compressors()

    parsers = getattr(code_compressor._tree_sitter_local, "parsers", {}) or {}
    assert "python" in parsers, "Opt-in preload should have loaded python"
    assert len(parsers) == 1, (
        "Only the explicitly-requested language should be preloaded; got "
        + ", ".join(sorted(parsers.keys()))
    )
    assert "tree_sitter" in status
    assert "loaded" in status["tree_sitter"]


def test_maybe_preload_returns_empty_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """The preload helper is a no-op when the env var is unset or empty."""
    from headroom.transforms.content_router import _maybe_preload_tree_sitter_parsers

    monkeypatch.delenv("HEADROOM_TREE_SITTER_PRELOAD", raising=False)
    assert _maybe_preload_tree_sitter_parsers() == []

    monkeypatch.setenv("HEADROOM_TREE_SITTER_PRELOAD", "")
    assert _maybe_preload_tree_sitter_parsers() == []

    monkeypatch.setenv("HEADROOM_TREE_SITTER_PRELOAD", "   ,  ,")
    assert _maybe_preload_tree_sitter_parsers() == []


def test_repo_root_layout() -> None:
    """Sanity: the subprocess REPO_ROOT actually contains the headroom package."""
    assert (REPO_ROOT / "headroom" / "image" / "trained_router.py").is_file()
    assert (REPO_ROOT / "headroom" / "transforms" / "content_router.py").is_file()
    # Subprocess needs to be able to import headroom from this cwd.
    assert (REPO_ROOT / "pyproject.toml").is_file()
