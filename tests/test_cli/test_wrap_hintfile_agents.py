"""Shared hint-file agent tests for `headroom wrap {cline,goose}` (PR-G1).

Cline and Goose are different wrap patterns (cline is proxy-only watcher,
goose launches a child binary) but both inject the RTK guidance into a
*hint file* at the project root — `.clinerules` and `.goosehints` —
through the same code path (`_inject_rtk_instructions` via the shared
`_setup_context_tool_for_agent` helper).

That hint-file plumbing is the same for both, so the tests covering it
parametrize over `(agent, hint_filename)` here. Agent-specific behavior
(goose's env-var fan-out, goose's binary discovery, cline's IDE setup
print-out) lives in `test_wrap_goose.py` / `test_wrap_cline.py`
respectively — those files keep only what genuinely diverges per agent.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli import wrap as wrap_mod
from headroom.cli.main import main

# (subcommand, hint-file basename) — used by every test below.
HINTFILE_AGENTS = [
    pytest.param("cline", ".clinerules", id="cline"),
    pytest.param("goose", ".goosehints", id="goose"),
]


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.mark.parametrize("agent,hintfile", HINTFILE_AGENTS)
def test_prepare_only_injects_rtk_into_hintfile(
    agent: str,
    hintfile: str,
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`wrap <agent> --prepare-only` writes the RTK block to the hint file at cwd."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
        result = runner.invoke(main, ["wrap", agent, "--prepare-only"])

    assert result.exit_code == 0, result.output
    marker = tmp_path / hintfile
    assert marker.exists(), f"{hintfile} should be created"
    content = marker.read_text()
    assert wrap_mod._RTK_MARKER in content
    assert "RTK (Rust Token Killer)" in content


@pytest.mark.parametrize("agent,hintfile", HINTFILE_AGENTS)
def test_prepare_only_idempotent_no_duplicate_block(
    agent: str,
    hintfile: str,
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running prepare-only twice must not duplicate the RTK block in the hint file."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
        runner.invoke(main, ["wrap", agent, "--prepare-only"])
        runner.invoke(main, ["wrap", agent, "--prepare-only"])

    content = (tmp_path / hintfile).read_text()
    assert content.count(wrap_mod._RTK_MARKER) == 1


@pytest.mark.parametrize("agent,hintfile", HINTFILE_AGENTS)
def test_no_context_tool_does_not_create_hintfile(
    agent: str,
    hintfile: str,
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--no-context-tool must not create the hint file and must not invoke rtk."""
    monkeypatch.chdir(tmp_path)

    with patch.object(wrap_mod, "_ensure_rtk_binary") as ensure:
        result = runner.invoke(main, ["wrap", agent, "--prepare-only", "--no-context-tool"])

    assert result.exit_code == 0, result.output
    assert not (tmp_path / hintfile).exists()
    ensure.assert_not_called()


@pytest.mark.parametrize("agent,hintfile", HINTFILE_AGENTS)
def test_preserves_existing_hintfile_content(
    agent: str,
    hintfile: str,
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-existing hint-file content must be preserved when RTK is appended."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    marker_path = tmp_path / hintfile
    original = "# Project conventions\n\nAlways use Python 3.12.\n"
    marker_path.write_text(original)

    with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
        result = runner.invoke(main, ["wrap", agent, "--prepare-only"])

    assert result.exit_code == 0, result.output
    content = marker_path.read_text()
    assert "Always use Python 3.12." in content
    assert wrap_mod._RTK_MARKER in content


# ---------------------------------------------------------------------------
# M4: Ctrl-C during prelude emits a clear "interrupted, marker may be on disk"
# message and exits non-zero (130, the conventional shell signal-130 code).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("agent,hintfile", HINTFILE_AGENTS)
def test_keyboardinterrupt_during_prelude_emits_clear_message(
    agent: str,
    hintfile: str,
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ctrl-C after marker injection but before proxy startup must report clearly."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    def raise_kbd(*args, **kwargs):  # noqa: ANN002, ANN003
        # Simulate the user hitting Ctrl-C right after the prelude wrote the
        # hint-file marker but before _ensure_proxy returns. We trigger via
        # _ensure_rtk_binary side-effect so the marker exists on disk.
        marker_path = tmp_path / hintfile
        marker_path.write_text(wrap_mod.RTK_INSTRUCTIONS_BLOCK)
        raise KeyboardInterrupt

    with patch.object(wrap_mod, "_ensure_rtk_binary", side_effect=raise_kbd):
        result = runner.invoke(main, ["wrap", agent, "--prepare-only"])

    assert result.exit_code == 130
    assert "interrupted" in result.output.lower()
    assert "idempotent" in result.output.lower()
    assert (tmp_path / hintfile).exists()
    assert hintfile in result.output
