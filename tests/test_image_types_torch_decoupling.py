"""Importing the image compressor must not eagerly import trained_router (torch).

#2513 (secondary): ``from .trained_router import Technique`` at module scope
pulled in torch / transformers, which on Python 3.13+ crashed with
``AttributeError: module 'torch' has no attribute 'compiler'`` before torch had
finished initializing. The routing types now live in a dependency-free module.

The "does not import" checks run in a fresh subprocess so module caching from
other tests can't mask the import graph.
"""

from __future__ import annotations

import subprocess
import sys


def _assert_import_excludes(module_name: str, excluded: str) -> None:
    code = (
        "import sys, importlib\n"
        f"importlib.import_module({module_name!r})\n"
        f"assert {excluded!r} not in sys.modules, {excluded!r} + ' was imported'\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_importing_compressor_does_not_import_trained_router() -> None:
    _assert_import_excludes("headroom.image.compressor", "headroom.image.trained_router")


def test_importing_onnx_router_does_not_import_trained_router() -> None:
    _assert_import_excludes("headroom.image.onnx_router", "headroom.image.trained_router")


def test_image_types_module_does_not_import_torch() -> None:
    _assert_import_excludes("headroom.image.image_types", "torch")


def test_technique_reexports_are_the_same_object() -> None:
    from headroom.image import Technique as via_package
    from headroom.image.image_types import ImageSignals, RouteDecision, Technique
    from headroom.image.trained_router import ImageSignals as tr_signals
    from headroom.image.trained_router import RouteDecision as tr_decision
    from headroom.image.trained_router import Technique as tr_technique

    assert via_package is Technique
    assert tr_technique is Technique
    assert tr_signals is ImageSignals
    assert tr_decision is RouteDecision
    assert Technique.PRESERVE.value == "preserve"
