"""Tests for `headroom xray` Click subcommand."""

from click.testing import CliRunner

from headroom.cli import main


def test_xray_help_shows_subcommand():
    """`headroom --help` should list xray as a subcommand."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "xray" in result.output.lower()


def test_xray_missing_binary_gives_clear_error(monkeypatch):
    """If headroom-xray binary is not found, error message should guide build.

    Patches `_find_binary` directly so a locally-built `target/release/headroom-xray`
    (left over from prior `cargo build`) doesn't cause the test to find a real
    binary and skip the error branch.

    Note: we invoke with no args (not --help) because `--help` is intercepted by
    Click's group-level help dispatch before the command body runs.
    """
    from headroom.cli import xray as xray_mod

    monkeypatch.setattr(xray_mod, "_find_binary", lambda: None)

    runner = CliRunner()
    result = runner.invoke(main, ["xray"])
    assert result.exit_code != 0
    assert "cargo build" in result.output or "cargo install" in result.output
