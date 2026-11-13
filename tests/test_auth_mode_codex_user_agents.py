"""Codex User-Agent classification regression tests (AQ-0610).

``CLIENT_UA_MAP`` carried a single OpenAI needle, ``codex-cli/``, which no
shipped Codex build actually sends. Every Codex request therefore classified as
``None``, so the Codex-specific fail-open in ``headroom/proxy/helpers.py`` —
which keys on ``client == "codex"`` — never fired and the proxy answered a
compression timeout with 413. Codex treats 413 as fatal, so sessions died at
startup.

The User-Agent strings below are verbatim from ~/.headroom/logs/proxy.log, not
invented. Counts across two rotations at the time of the fix:
    codex-tui/           3,167
    codex-browser-use/   1,338
    Codex Desktop/         368
    codex_cli_rs/           13
    codex-computer-use/      1
"""

from __future__ import annotations

import pytest

from headroom.proxy.auth_mode import classify_client

# Verbatim User-Agents observed on the wire.
OBSERVED_CODEX_USER_AGENTS = [
    "codex-tui/0.147.0 (Mac OS 26.5.1; arm64) Apple_Terminal/470.2 (codex-tui; 0.147.0)",
    "codex-tui/0.148.0-alpha.9 (Mac OS 26.5.1; arm64) xterm-256color",
    "codex_cli_rs/0.147.0 (Mac OS 26.5.1; arm64) Apple_Terminal/470.2",
    "codex_cli_rs/0.148.0-alpha.9 (Mac OS 26.5.1; arm64) unknown",
    "codex-browser-use/0.147.0-alpha.1.2 (Mac OS 26.5.1; arm64) xterm-256color"
    " (codex-browser-use; 0.1.0)",
    "Codex Desktop/0.147.0-alpha.6.5 (Mac OS 26.5.1; arm64) unknown",
    "Codex Desktop/0.148.0-alpha.9 (Mac OS 26.5.1; arm64) unknown (Codex Desktop; 26.810.41047)",
    "codex-computer-use/0.147.0-alpha.1.2 (Mac OS 26.5.1; arm64) unknown",
]


@pytest.mark.parametrize("user_agent", OBSERVED_CODEX_USER_AGENTS)
def test_observed_codex_user_agents_classify_as_codex(user_agent: str) -> None:
    """Every Codex build seen in production must resolve to ``codex``.

    This is what gates the compression-timeout fail-open. A ``None`` here means
    a real Codex session gets a fatal 413 instead of an uncompressed forward.
    """
    assert classify_client({"user-agent": user_agent}) == "codex"


def test_codex_desktop_matches_despite_capitals_and_space() -> None:
    """``Codex Desktop/`` is the only client whose UA contains a space.

    Matching lowercases the UA, so the needle must be lowercase too — a
    ``"Codex Desktop/"`` needle would silently never match.
    """
    assert classify_client({"user-agent": "Codex Desktop/0.148.0-alpha.9 (Mac OS)"}) == "codex"


def test_codex_ua_carrying_a_claude_code_originator_still_classifies_as_codex() -> None:
    """Real UA that embeds both harness names.

    ``codex_cli_rs/... (claude-code; 2.1.233)`` is the Codex binary reporting
    Claude Code as its originator. The Anthropic needles are checked first and
    require ``claude-code/`` with a slash, so the semicolon form must not steal
    this match — the process talking to us is Codex, and Codex is the one that
    dies on a 413.
    """
    ua = "codex_cli_rs/0.147.0 (Mac OS 26.5.1; arm64) Apple_Terminal/470.2 (claude-code; 2.1.233)"

    assert classify_client({"user-agent": ua}) == "codex"


def test_claude_code_user_agents_are_unaffected() -> None:
    """Negative control: the Anthropic needles must keep winning their own UAs."""
    assert classify_client({"user-agent": "claude-cli/2.1.233 (external, cli)"}) == "claude-code"
    assert (
        classify_client({"user-agent": "claude-cli/2.1.233 (external, sdk-py, agent-sdk/0.2.139)"})
        == "claude-code"
    )


def test_unrelated_user_agent_still_returns_none() -> None:
    """Negative control: broadening the Codex needles must not swallow everything."""
    assert classify_client({"user-agent": "Mozilla/5.0 (Macintosh)"}) is None
    assert classify_client({"user-agent": "curl/8.7.1"}) is None
