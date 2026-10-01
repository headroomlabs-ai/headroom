"""A refusal is only useful if the caller turns it into a message.

`UnpinnedDownload` was added as a new `BinaryError` subclass, but three callers
caught a hand-written tuple of subclasses rather than the base, so the new one
escaped as a traceback. These pin the contract at the boundaries a person or a
calling feature actually sees.
"""

from __future__ import annotations

import logging

import pytest
from click.testing import CliRunner

from headroom import binaries


@pytest.fixture
def _refuse(monkeypatch: pytest.MonkeyPatch):
    """Make `resolve` refuse the way an off-registry asset does."""

    def _raise(tool: str):
        raise binaries.UnpinnedDownload(
            f"refusing unpinned download {tool}: no sha256 in the registry. "
            "Pin it, or set HEADROOM_BINARIES_ALLOW_UNVERIFIED=1."
        )

    monkeypatch.setattr(binaries, "resolve", _raise)
    monkeypatch.delenv("HEADROOM_BINARIES_ALLOW_UNVERIFIED", raising=False)


def test_tools_run_reports_a_refusal_instead_of_a_traceback(_refuse) -> None:
    """`headroom tools <name>` must exit 2 with `error: ...`, like every sibling."""
    from headroom.cli.tools import diff_cmd

    result = CliRunner().invoke(diff_cmd, ["--version"])

    assert result.exit_code == 2, result.output
    assert not isinstance(result.exception, binaries.UnpinnedDownload), (
        "the refusal escaped the CLI as a traceback"
    )
    assert "refusing unpinned download" in result.output


def test_tools_install_keeps_going_and_still_sets_an_exit_code(_refuse) -> None:
    """A refusal on one tool must not abort the loop over the others.

    The escape skipped the final `sys.exit(exit_code)` entirely, so the command
    reported success-by-omission for every tool after the first refusal.
    """
    from headroom.cli.tools import tools_group

    result = CliRunner().invoke(tools_group, ["install"])

    assert not isinstance(result.exception, binaries.UnpinnedDownload), (
        "the refusal aborted `tools install`"
    )
    assert result.exit_code == 1, result.output


def test_ensure_cbm_honours_its_documented_none_contract(monkeypatch) -> None:
    """`ensure_cbm` promises "path, or None if the download failed"."""
    from headroom.graph import installer

    monkeypatch.setattr(installer, "get_cbm_path", lambda: None)

    def _raise():
        raise binaries.UnpinnedDownload("refusing unpinned download codebase-memory-mcp")

    monkeypatch.setattr(installer, "download_cbm", _raise)

    assert installer.ensure_cbm() is None


def test_a_tamper_signal_still_propagates_from_ensure_cbm(monkeypatch) -> None:
    """The None contract must not swallow a sha256 mismatch.

    An unpinned asset is a misconfiguration; a mismatch is evidence of tampering
    and has to stay loud rather than becoming a quiet "feature unavailable".
    """
    from headroom.graph import installer

    monkeypatch.setattr(installer, "get_cbm_path", lambda: None)

    def _raise():
        raise binaries.Sha256Mismatch("sha256 mismatch for codebase-memory-mcp")

    monkeypatch.setattr(installer, "download_cbm", _raise)

    with pytest.raises(binaries.Sha256Mismatch):
        installer.ensure_cbm()


def test_the_escape_hatch_warning_is_not_printed_twice(monkeypatch, capsys) -> None:
    """With no handlers configured, logging's lastResort already writes stderr.

    Printing unconditionally as well produced the same sentence twice.
    """
    monkeypatch.setenv("HEADROOM_BINARIES_ALLOW_UNVERIFIED", "1")

    logger = logging.getLogger("headroom.binaries")
    saved, logger.handlers = logger.handlers, []
    saved_propagate, logger.propagate = logger.propagate, False
    try:
        assert binaries._allow_unverified("difft") is True
    finally:
        logger.handlers, logger.propagate = saved, saved_propagate

    stderr = capsys.readouterr().err
    assert stderr.count("WITHOUT sha256 verification") <= 1, (
        f"the warning was emitted more than once:\n{stderr}"
    )
