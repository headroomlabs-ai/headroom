"""Regression tests for #1126 — `headroom wrap` instruction injection must read
and write user instruction files as UTF-8, so non-ASCII prose (typographic
quotes, em-dashes) does not crash on a cp1252 (Windows) locale.

The reads use `errors="replace"`, so a stray non-UTF-8 byte (e.g. `0x9d`, which
fails a bare `open()` on *any* locale) cannot abort the wrap either.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from headroom.cli.wrap import (
    _MEMORY_AGENTS_MARKER,
    _RTK_MARKER,
    _inject_memory_agents_md,
    _inject_rtk_instructions,
)

# (inject_fn, marker) for the two prose injectors that share the bug.
INJECTORS = [
    pytest.param(_inject_rtk_instructions, _RTK_MARKER, id="rtk"),
    pytest.param(_inject_memory_agents_md, _MEMORY_AGENTS_MARKER, id="memory_agents"),
]

# A typographic quote / em-dash (valid UTF-8) plus a stray byte that is
# undefined in cp1252 and an invalid UTF-8 start byte.
_EXISTING = "Be in “happy places” — really.\n".encode() + b"legacy \x9d byte\n"


@pytest.mark.parametrize("inject, marker", INJECTORS)
def test_inject_appends_into_file_with_non_ascii_and_stray_byte(inject, marker, tmp_path: Path):
    target = tmp_path / "AGENTS.md"
    target.write_bytes(_EXISTING)

    # Before the fix this raised UnicodeDecodeError reading the existing file.
    assert inject(target) is True

    text = target.read_text(encoding="utf-8", errors="replace")
    assert marker in text
    # The pre-existing prose is preserved (append, not rewrite).
    assert "happy places" in text


@pytest.mark.parametrize("inject, marker", INJECTORS)
def test_inject_creates_file_when_absent(inject, marker, tmp_path: Path):
    target = tmp_path / "nested" / "AGENTS.md"

    assert inject(target) is True
    assert marker in target.read_text(encoding="utf-8")


@pytest.mark.parametrize("inject, marker", INJECTORS)
def test_inject_is_idempotent(inject, marker, tmp_path: Path):
    target = tmp_path / "AGENTS.md"
    target.write_bytes(_EXISTING)

    assert inject(target) is True
    assert inject(target) is True  # marker already present -> no duplicate / no crash
    assert target.read_text(encoding="utf-8", errors="replace").count(marker) == 1
