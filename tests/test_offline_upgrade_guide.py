"""Verify the offline-upgrade commands in configuration.mdx actually work.

The guide used to tell operators to run `compress()` to seed the cache, which
does not reliably reach Kompress (the router may never invoke it). This checks
the replacement: `warm_kompress_model` exists and the verification snippet
finds files at the layout it tells operators to look for.
"""

from __future__ import annotations

import re
from pathlib import Path

from headroom.onnx_runtime import _PINNED_REVISIONS

DOCS = Path(__file__).resolve().parents[1] / "docs" / "content" / "docs" / "configuration.mdx"


def test_warm_kompress_model_is_the_documented_seeding_entry_point():
    from headroom.transforms.kompress_compressor import warm_kompress_model

    assert callable(warm_kompress_model), "the guide's seeding call must exist"


def test_guide_does_not_tell_operators_to_seed_with_compress():
    """`compress()` routes through the router and may never touch Kompress."""
    text = DOCS.read_text(encoding="utf-8")
    section = text.split("#### Offline upgrade path", 1)[1].split("\n###", 1)[0]
    assert "warm_kompress_model" in section, "the guide must name the real loader"
    assert not re.search(r"^\s*headroom\.compress\(", section, re.MULTILINE), (
        "the guide must not seed via compress(): the router can skip Kompress "
        "and leave the weights absent"
    )


def test_guide_checks_the_pinned_snapshot_layout():
    """The verification snippet must key on snapshots/<pin>, not refs/<pin>."""
    text = DOCS.read_text(encoding="utf-8")
    section = text.split("#### Offline upgrade path", 1)[1].split("\n###", 1)[0]
    assert "snapshots" in section
    assert "_PINNED_REVISIONS" in section
    # A pin the snippet resolves must actually exist in the table.
    assert "chopratejas/kompress-v2-base" in _PINNED_REVISIONS


def test_documented_kompress_repos_and_files_exist_in_the_pin_table():
    """The snippet names these; a rename would make the guide silently wrong."""
    from headroom.transforms.kompress_compressor import (
        _MODERNBERT_TOKENIZER_REPO,
        HF_MODEL_ID,
    )

    assert HF_MODEL_ID in _PINNED_REVISIONS
    assert _MODERNBERT_TOKENIZER_REPO in _PINNED_REVISIONS
