"""Tests for ``_patch_rtk_hook_absolute_path``.

``rtk init --global --auto-patch`` writes ``~/.claude/hooks/rtk-rewrite.sh``
with a bare ``rtk`` command that depends on PATH lookup. Since
``~/.headroom/bin`` is not automatically added to PATH, that lookup fails
silently and token compression never occurs (see issue #487).

``_patch_rtk_hook_absolute_path`` rewrites bare ``rtk`` tokens in the
generated hook script to the absolute, shell-quoted path of the rtk binary
that Headroom manages.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from headroom.cli.wrap import _patch_rtk_hook_absolute_path


def test_patches_bare_rtk_to_absolute_path(tmp_path: Path) -> None:
    hook_script = tmp_path / "rtk-rewrite.sh"
    hook_script.write_text(
        '#!/bin/sh\nif command -v rtk >/dev/null 2>&1; then\n    exec rtk rewrite "$@"\nfi\n'
    )

    rtk_path = Path("/home/user/.headroom/bin/rtk")
    changed = _patch_rtk_hook_absolute_path(rtk_path, hook_script)

    content = hook_script.read_text()
    quoted = shlex.quote(str(rtk_path))

    assert changed is True
    assert f"exec {quoted} rewrite" in content


def test_quotes_path_containing_spaces(tmp_path: Path) -> None:
    """Paths with spaces (e.g. /Users/Alice Smith/...) must be shell-quoted."""
    hook_script = tmp_path / "rtk-rewrite.sh"
    hook_script.write_text(
        '#!/bin/sh\nif command -v rtk >/dev/null 2>&1; then\n    exec rtk rewrite "$@"\nfi\n'
    )

    rtk_path = Path("/Users/Alice Smith/.headroom/bin/rtk")
    changed = _patch_rtk_hook_absolute_path(rtk_path, hook_script)

    content = hook_script.read_text()
    quoted = shlex.quote(str(rtk_path))

    assert changed is True
    assert f"exec {quoted} rewrite" in content
    # The raw, unquoted path must never appear unescaped in the script.
    assert "exec /Users/Alice Smith/.headroom/bin/rtk rewrite" not in content


def test_idempotent_second_run_is_noop(tmp_path: Path) -> None:
    hook_script = tmp_path / "rtk-rewrite.sh"
    hook_script.write_text('exec rtk rewrite "$@"\n')

    rtk_path = Path("/home/user/.headroom/bin/rtk")

    first = _patch_rtk_hook_absolute_path(rtk_path, hook_script)
    content_after_first = hook_script.read_text()

    second = _patch_rtk_hook_absolute_path(rtk_path, hook_script)
    content_after_second = hook_script.read_text()

    assert first is True
    assert second is False
    assert content_after_first == content_after_second


def test_missing_hook_script_is_noop(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.sh"
    rtk_path = Path("/home/user/.headroom/bin/rtk")

    changed = _patch_rtk_hook_absolute_path(rtk_path, missing)

    assert changed is False
    assert not missing.exists()


def test_does_not_touch_words_containing_rtk(tmp_path: Path) -> None:
    """Tokens like 'rtkfoo' or an already-absolute '/some/path/rtk' are left alone."""
    hook_script = tmp_path / "rtk-rewrite.sh"
    original = '#!/bin/sh\necho rtkfoo\nexec /already/absolute/rtk rewrite "$@"\n'
    hook_script.write_text(original)

    rtk_path = Path("/home/user/.headroom/bin/rtk")
    changed = _patch_rtk_hook_absolute_path(rtk_path, hook_script)

    assert changed is False
    assert hook_script.read_text() == original
