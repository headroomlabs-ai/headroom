"""agy print-mode MCP version gate (headroom-37g.37).

Older/unknown agy binaries hang on ANY mcpServers entry in --print mode.
`headroom wrap agy` therefore gates print-mode MCP wiring on a runtime
`agy --version` preflight: enable only when >= 1.0.16, otherwise SUPPRESS and
actively PURGE any persisted entries so a prior interactive run can't leave a
config that still hangs. Interactive mode is never gated (the hang is
print-mode-only). No real agy binary is invoked here — everything is mocked.
"""

import subprocess
from unittest import mock

from headroom.cli import wrap
from headroom.cli.wrap import (
    _AGY_PRINT_MODE_MCP_MIN_VERSION,
    _agy_print_mode_mcp_allowed,
    _detect_agy_version,
    _purge_agy_mcp_entries,
)


def _run_result(returncode: int, stdout: str) -> mock.Mock:
    return mock.Mock(returncode=returncode, stdout=stdout)


# -- _detect_agy_version (parse + never-raises) -----------------------------


def test_detect_version_parses_bare_line():
    with mock.patch("subprocess.run", return_value=_run_result(0, "1.0.16\n")):
        assert _detect_agy_version("/usr/bin/agy") == (1, 0, 16)


def test_detect_version_trims_whitespace_and_banner():
    with mock.patch("subprocess.run", return_value=_run_result(0, "  agy version 1.0.16  \n")):
        assert _detect_agy_version("/usr/bin/agy") == (1, 0, 16)


def test_detect_version_takes_last_match_over_wrapper_banner():
    # A wrapper prints its own version first, then the real agy version.
    with mock.patch("subprocess.run", return_value=_run_result(0, "wrapper 9.9.9\nagy 1.0.16\n")):
        assert _detect_agy_version("/usr/bin/agy") == (1, 0, 16)


def test_detect_version_none_on_garbage():
    with mock.patch("subprocess.run", return_value=_run_result(0, "no version here")):
        assert _detect_agy_version("/usr/bin/agy") is None


def test_detect_version_none_on_empty():
    with mock.patch("subprocess.run", return_value=_run_result(0, "")):
        assert _detect_agy_version("/usr/bin/agy") is None


def test_detect_version_none_on_nonzero_exit():
    with mock.patch("subprocess.run", return_value=_run_result(2, "1.0.16")):
        assert _detect_agy_version("/usr/bin/agy") is None


def test_detect_version_none_on_timeout():
    with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired("agy", 1.0)):
        assert _detect_agy_version("/usr/bin/agy") is None


def test_detect_version_none_on_oserror():
    with mock.patch("subprocess.run", side_effect=OSError("boom")):
        assert _detect_agy_version("/usr/bin/agy") is None


def test_detect_version_none_when_bin_missing():
    assert _detect_agy_version(None) is None
    assert _detect_agy_version("") is None


def test_detect_version_uses_short_timeout_and_devnull_stderr():
    with mock.patch("subprocess.run", return_value=_run_result(0, "1.0.16")) as run:
        _detect_agy_version("/usr/bin/agy")
    _args, kwargs = run.call_args
    assert kwargs["timeout"] == 1.0
    assert kwargs["stderr"] is subprocess.DEVNULL


# -- _agy_print_mode_mcp_allowed (the gate) ---------------------------------


def test_gate_allows_print_mode_when_known_good():
    with mock.patch.object(
        wrap, "_detect_agy_version", return_value=_AGY_PRINT_MODE_MCP_MIN_VERSION
    ):
        assert _agy_print_mode_mcp_allowed(("--print", "hi"), "/usr/bin/agy") is True


def test_gate_allows_print_mode_when_newer():
    with mock.patch.object(wrap, "_detect_agy_version", return_value=(1, 1, 0)):
        assert _agy_print_mode_mcp_allowed(("-p", "hi"), "/usr/bin/agy") is True


def test_gate_suppresses_print_mode_when_older():
    with mock.patch.object(wrap, "_detect_agy_version", return_value=(1, 0, 15)):
        assert _agy_print_mode_mcp_allowed(("--print", "hi"), "/usr/bin/agy") is False


def test_gate_suppresses_print_mode_when_unknown():
    with mock.patch.object(wrap, "_detect_agy_version", return_value=None):
        assert _agy_print_mode_mcp_allowed(("--prompt", "hi"), "/usr/bin/agy") is False


def test_gate_allows_interactive_without_version_check():
    # Interactive mode (no print flag) is always allowed and must NOT even
    # spend a version-detection subprocess.
    with mock.patch.object(wrap, "_detect_agy_version") as detect:
        assert _agy_print_mode_mcp_allowed((), "/usr/bin/agy") is True
        assert _agy_print_mode_mcp_allowed(("--model", "x"), "/usr/bin/agy") is True
    detect.assert_not_called()


# -- _purge_agy_mcp_entries (load-bearing: all persisted types removed) ------


def test_purge_targets_all_four_entry_types():
    registrar = mock.Mock()
    with (
        mock.patch.object(wrap, "_disable_tokensave_mcp") as dis_tok,
        mock.patch.object(wrap, "_disable_serena_mcp") as dis_ser,
        mock.patch.object(wrap, "_remove_headroom_installed_retrieve_mcp") as dis_retrieve,
        mock.patch.object(wrap, "_remove_headroom_installed_mcp") as dis_legacy,
    ):
        _purge_agy_mcp_entries(registrar)

    # tokensave + serena removed via the LEDGER-AWARE helpers
    # (so a user-owned entry is never clobbered).
    dis_tok.assert_called_once_with(registrar)
    dis_ser.assert_called_once()
    dis_retrieve.assert_called_once_with(registrar)
    dis_legacy.assert_called_once_with(registrar, wrap._CBM_MCP_SERVER_NAME)
    # Legacy code-graph cleanup remains separate from ledger-owned retrieve.
    unregistered = {c.args[0] for c in registrar.unregister_server.call_args_list}
    assert wrap._CBM_MCP_SERVER_NAME not in unregistered
    assert "headroom" not in unregistered
