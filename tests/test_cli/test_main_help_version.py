"""Tests for top-level help and version aliases."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from headroom.cli.main import main


def test_root_help_short_alias() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["-?"])

    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output
    assert "--version" in result.output


def test_root_version_short_alias() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["-v"])

    assert result.exit_code == 0, result.output
    assert "version" in result.output.lower()


def test_group_help_short_alias() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["wrap", "-?"])

    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output
    assert "claude" in result.output


def test_wrap_subcommand_help_short_alias_beats_passthrough() -> None:
    runner = CliRunner()
    with patch("headroom.cli.wrap.shutil.which") as which_mock:
        result = runner.invoke(main, ["wrap", "claude", "-?"])

    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output
    assert "Launch Claude Code through Headroom proxy." in result.output
    which_mock.assert_not_called()


def test_subcommand_verbose_flag_still_works() -> None:
    runner = CliRunner()
    completed = SimpleNamespace(returncode=0)

    with patch("headroom.cli.wrap.shutil.which", return_value="claude"):
        with patch("headroom.cli.wrap._ensure_proxy", return_value=None):
            with patch("headroom.cli.wrap._setup_rtk", return_value=None):
                with patch("headroom.cli.wrap.subprocess.run", return_value=completed):
                    result = runner.invoke(main, ["wrap", "claude", "-v"])

    assert result.exit_code == 0, result.output
    assert "HEADROOM WRAP: CLAUDE" in result.output
