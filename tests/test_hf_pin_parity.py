"""The HF revision pins must agree across Python and Rust (GH #4939).

Pinning the same artifact in three tables is only useful if they stay in step.
The tokenizer loader, the ONNX/model download helper and the Rust core's
`hf_pins` module all fetch from the Hub, and a repo pinned in one runtime but
floating in the other reproduces the original defect on that code path.

Parsed with a regex rather than a Rust parser: `hf_pins.rs` is a flat static
array of string pairs, and this test fails loudly (not silently) if that shape
changes.
"""

from __future__ import annotations

import re
from pathlib import Path

from headroom.onnx_runtime import _PINNED_REVISIONS
from headroom.tokenizers import huggingface as hf_mod

_PINS_RS = Path(__file__).resolve().parents[1] / "crates" / "headroom-core" / "src" / "hf_pins.rs"

_ENTRY = re.compile(r'\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,?\s*\)')


def _rust_pins() -> dict[str, str]:
    source = _PINS_RS.read_text(encoding="utf-8")
    body = source.split("pub static PINNED_REVISIONS", 1)[1]
    body = body.split("];", 1)[0]
    return dict(_ENTRY.findall(body))


def test_the_rust_table_is_parsed_at_all() -> None:
    """Guards the parser: an empty result would make every test below vacuous."""
    assert len(_rust_pins()) > 60, "hf_pins.rs shape changed; fix the parser"


def test_onnx_pins_match_the_rust_core() -> None:
    """Any repo pinned in both runtimes must agree on the SHA.

    Directional on purpose. Rust pins a strict subset of what Python pins: the
    Python side also owns the torch-path loaders (sentence-transformers, SigLIP,
    the technique router), which the Rust core never loads. The reverse — a repo
    Rust pins but Python floats — is a real defect, and
    `test_no_rust_pin_is_unknown_to_python` covers it.
    """
    rust = _rust_pins()
    disagreeing = {
        repo: (sha, rust[repo])
        for repo, sha in _PINNED_REVISIONS.items()
        if repo in rust and rust[repo] != sha
    }
    assert not disagreeing, (
        f"the same repo is pinned to different commits per runtime: {disagreeing}"
    )


def test_every_rust_pinned_repo_is_also_pinned_in_python() -> None:
    rust = _rust_pins()
    missing = sorted(set(rust) - set(_PINNED_REVISIONS) - set(hf_mod._TOKENIZER_REVISIONS))
    assert not missing, (
        "pinned in hf_pins.rs but in no Python table — the Rust side would pin a "
        f"repo the Python side floats: {missing}"
    )


def test_the_rust_ml_artifacts_are_pinned_in_python_too() -> None:
    """The three paths GH #4939 names are reachable from both runtimes."""
    for repo in (
        "chopratejas/kompress-v2-base",
        "answerdotai/ModernBERT-base",
        "chopratejas/technique-router-onnx",
    ):
        assert repo in _rust_pins(), f"{repo} lost its Rust pin"
        assert repo in _PINNED_REVISIONS, f"{repo} lost its Python pin"
        assert _rust_pins()[repo] == _PINNED_REVISIONS[repo]


def test_tokenizer_pins_match_the_rust_core() -> None:
    rust = _rust_pins()
    disagreeing = {
        repo: (sha, rust[repo])
        for repo, sha in hf_mod._TOKENIZER_REVISIONS.items()
        if repo in rust and rust[repo] != sha
    }
    assert not disagreeing, f"tokenizer pins disagree with hf_pins.rs: {disagreeing}"
