"""The Kompress ONNX export must build from the same immutable Hub snapshots
production loads, so an exported artifact can never mix floating inputs."""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest

from headroom.onnx_runtime import _PINNED_REVISIONS

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "export_kompress_v2_onnx.py"
SHA = re.compile(r"[0-9a-f]{40}")
BASE = "answerdotai/ModernBERT-base"
CKPT = "chopratejas/kompress-v2-base"


def _load_script():
    spec = importlib.util.spec_from_file_location("export_kompress_v2_onnx", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Call):  # _get_model_class()(...)
        return _call_name(func) + "()"
    return ""


def test_every_hub_load_in_export_script_passes_revision():
    loaders = {"from_pretrained", "hf_hub_download", "snapshot_download", "_get_model_class()"}
    calls = [
        node
        for node in ast.walk(ast.parse(SCRIPT.read_text()))
        if isinstance(node, ast.Call) and _call_name(node) in loaders
    ]
    assert {_call_name(c) for c in calls} >= {
        "from_pretrained",
        "hf_hub_download",
        "_get_model_class()",
    }
    unpinned = [
        f"line {c.lineno}: {_call_name(c)}"
        for c in calls
        if not any(kw.arg == "revision" for kw in c.keywords)
    ]
    assert unpinned == []


def test_export_defaults_to_production_pins_even_with_pinning_disabled(monkeypatch):
    monkeypatch.setenv("HEADROOM_HF_PIN", "off")
    mod = _load_script()
    seen: dict = {}
    monkeypatch.setattr(mod, "export", lambda *a: seen.setdefault("args", a))
    monkeypatch.setattr(sys, "argv", ["export"])

    assert mod.main() == 0
    revision, base_revision = seen["args"][-2:]
    assert revision == _PINNED_REVISIONS[CKPT]
    assert base_revision == _PINNED_REVISIONS[BASE]
    assert SHA.fullmatch(revision) and SHA.fullmatch(base_revision)


@pytest.mark.parametrize(
    "argv",
    [
        ["--revision", "main"],
        ["--base-revision", "v1.0"],
        ["--model-id", "someone/unpinned-model"],
    ],
)
def test_export_rejects_floating_or_missing_revisions(monkeypatch, argv):
    mod = _load_script()
    monkeypatch.setattr(mod, "export", lambda *a: pytest.fail("export ran unpinned"))
    monkeypatch.setattr(sys, "argv", ["export", *argv])
    with pytest.raises(SystemExit, match="40-hex commit SHA"):
        mod.main()


def test_build_core_threads_both_revisions(monkeypatch):
    mod = _load_script()
    calls: list[tuple] = []

    def fake_download(repo, filename, **kwargs):
        calls.append(("download", repo, kwargs.get("revision")))
        return "merged.pt"

    keys = ("encoder_state_dict", "token_head_state_dict", "span_conv_state_dict")
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(load=lambda *a, **k: {k: {} for k in keys})
    )
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=fake_download)
    )

    class _Part:
        def load_state_dict(self, sd, strict):
            return [], []

    class _Core:
        def __init__(self, model_name, **kwargs):
            calls.append(("encoder", model_name, kwargs.get("revision")))
            self.encoder = self.token_head = self.span_conv = _Part()

        def eval(self):
            return self

    from headroom.transforms import kompress_compressor as kc

    monkeypatch.setattr(kc, "_get_model_class", lambda: _Core)

    mod._build_core(CKPT, "a" * 40, "b" * 40)
    assert calls == [("download", CKPT, "a" * 40), ("encoder", BASE, "b" * 40)]
