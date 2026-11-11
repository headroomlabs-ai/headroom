"""The Rust Kompress loader reads the snapshots the Python side pinned.

The native proxy's Kompress is cache-only: it never downloads, it reads the
files Python's ``hf_hub_download_local_first`` put in the HuggingFace cache at
``_PINNED_REVISIONS``. ``crates/headroom-core/src/transforms/kompress.rs`` looks
up the same commits, so the two must not drift: a pin bump on one side only
would leave the Rust proxy on a snapshot Python no longer downloads.
"""

from __future__ import annotations

import re
from pathlib import Path

from headroom.onnx_runtime import _PINNED_REVISIONS

RUST_KOMPRESS = (
    Path(__file__).resolve().parents[1] / "crates/headroom-core/src/transforms/kompress.rs"
)


def _rust_const(name: str) -> str:
    match = re.search(rf'pub const {name}: &str = "([^"]+)";', RUST_KOMPRESS.read_text())
    assert match, f"{name} not found in {RUST_KOMPRESS}"
    return match.group(1)


def test_rust_kompress_model_pin_matches_python():
    repo = _rust_const("DEFAULT_MODEL_ID")
    assert repo in _PINNED_REVISIONS, f"{repo} has no Python pin"
    assert _rust_const("DEFAULT_MODEL_REVISION") == _PINNED_REVISIONS[repo]


def test_rust_kompress_tokenizer_pin_matches_python():
    repo = _rust_const("DEFAULT_TOKENIZER_REPO")
    assert repo in _PINNED_REVISIONS, f"{repo} has no Python pin"
    assert _rust_const("DEFAULT_TOKENIZER_REVISION") == _PINNED_REVISIONS[repo]
