"""Issue #1779: Remote Control is *silently* disabled behind the proxy.

Claude Code v2.1.196 added a client-side eligibility check that deterministically
disables first-party Remote Control (`/remote-control` / `/rc`) whenever
`ANTHROPIC_BASE_URL` points at a non-`api.anthropic.com` host — which Headroom
always does. The gate is upstream, so Headroom's fix is an *accurate* warning
that:

* states the disable as a fact on v2.1.196+ (never the old hedged "may"),
* fires only for subscription sessions that ever had RC (never API-key / cloud),
* fires only when the installed version is at/after the gate, or unknown,
* co-reports the sibling base-URL gates #746 and #1158.
"""

from __future__ import annotations

import pytest

from headroom.providers.claude.runtime import (
    REMOTE_CONTROL_GATED_MIN_VERSION,
    REMOTE_CONTROL_SIBLING_GATE_NOTE,
    detect_claude_code_version,
    is_custom_anthropic_base_url,
    parse_claude_code_version,
    remote_control_applies_to_auth,
    remote_control_gate_active,
    remote_control_gate_message,
    remote_control_sibling_gate_note,
)

_CUSTOM = "http://127.0.0.1:8787"
_NATIVE = "https://api.anthropic.com"
_GATED = REMOTE_CONTROL_GATED_MIN_VERSION  # (2, 1, 196)
_OLD = (2, 1, 195)


# ---------------------------------------------------------------------------
# Message accuracy — deterministic wording, not "may"
# ---------------------------------------------------------------------------


def test_message_is_accurate_not_hedged() -> None:
    msg = remote_control_gate_message("ANTHROPIC_BASE_URL in shell", version=_GATED)
    # Deterministic: names the exact version and says it "disables" /rc.
    assert "2.1.196" in msg
    assert "disables" in msg
    assert "/remote-control (/rc)" in msg
    # The old hedged phrasing is gone.
    assert "may hide" not in msg
    assert "run Claude without Headroom for sessions that need Remote Control" in msg


def test_message_unknown_version_states_threshold() -> None:
    msg = remote_control_gate_message("ANTHROPIC_BASE_URL in shell", version=None)
    # Without a detected version we state the threshold and let the user
    # self-identify — no false claim about their specific build.
    assert "2.1.196+" in msg
    assert "/rc" in msg
    assert "may hide" not in msg


# ---------------------------------------------------------------------------
# Auth gating — never warn a PAYG / cloud user (RC was never theirs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_key",
    [
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
    ],
)
def test_non_subscription_auth_never_applies(env_key: str) -> None:
    assert remote_control_applies_to_auth({env_key: "something"}) is False
    # And therefore the whole gate is inactive even on a gated version / custom URL.
    assert remote_control_gate_active(_CUSTOM, {env_key: "something"}, _GATED) is False


def test_subscription_auth_applies() -> None:
    assert remote_control_applies_to_auth({}) is True
    assert remote_control_applies_to_auth({"PATH": "/usr/bin"}) is True


def test_blank_api_key_is_not_treated_as_payg() -> None:
    # An empty / whitespace value is "unset" — a subscription session.
    assert remote_control_applies_to_auth({"ANTHROPIC_API_KEY": "   "}) is True
    assert remote_control_gate_active(_CUSTOM, {"ANTHROPIC_API_KEY": ""}, _GATED) is True


# ---------------------------------------------------------------------------
# Version gating — no false alarm on pre-2.1.196 builds
# ---------------------------------------------------------------------------


def test_gate_active_on_gated_version() -> None:
    assert remote_control_gate_active(_CUSTOM, {}, _GATED) is True
    assert remote_control_gate_active(_CUSTOM, {}, (2, 2, 0)) is True


def test_gate_inactive_on_pre_gate_version() -> None:
    # Older Claude Code does not gate RC on the base URL — warning would be false.
    assert remote_control_gate_active(_CUSTOM, {}, _OLD) is False
    assert remote_control_gate_active(_CUSTOM, {}, (1, 0, 0)) is False


def test_gate_active_when_version_unknown() -> None:
    # Unknown version → warn conservatively (the message self-qualifies).
    assert remote_control_gate_active(_CUSTOM, {}, None) is True


def test_gate_inactive_on_native_base_url() -> None:
    assert remote_control_gate_active(_NATIVE, {}, _GATED) is False
    assert remote_control_gate_active(None, {}, _GATED) is False


# ---------------------------------------------------------------------------
# Version parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2.1.196 (Claude Code)", (2, 1, 196)),
        ("claude-code/2.1.200", (2, 1, 200)),
        ("v2.0.0", (2, 0, 0)),
        ("  2.1.196\n", (2, 1, 196)),
        ("no version here", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_claude_code_version(text, expected) -> None:
    assert parse_claude_code_version(text) == expected


def test_detect_claude_code_version_missing_binary_is_none() -> None:
    # A binary that does not exist must never raise — best-effort → None.
    assert detect_claude_code_version("definitely-not-a-real-binary-xyz") is None


def test_detect_claude_code_version_tolerates_proc_without_stdout(monkeypatch) -> None:
    # Regression (CI test failure on PR #1779): a stubbed subprocess result — a
    # SimpleNamespace with only returncode, no stdout/stderr — must not raise
    # AttributeError. detect is best-effort → returns None (version unknown).
    from types import SimpleNamespace

    import headroom._subprocess as _sub

    monkeypatch.setattr(_sub, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    assert detect_claude_code_version("claude") is None


def test_detect_claude_code_version_parses_wrapper_output(monkeypatch) -> None:
    from types import SimpleNamespace

    import headroom._subprocess as _sub

    monkeypatch.setattr(
        _sub,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="2.1.196 (Claude Code)\n", stderr=""),
    )
    assert detect_claude_code_version("claude") == (2, 1, 196)


def test_detect_claude_code_version_nonzero_exit_is_none(monkeypatch) -> None:
    # Review follow-up (PR #1883, @JerrettDavis): a non-zero exit is a detection
    # failure even when the failing command still prints a version-shaped string.
    # Trusting it would emit a false *exact*-version Remote Control warning
    # instead of the self-qualified "2.1.196+ / unknown" path. Must return None.
    from types import SimpleNamespace

    import headroom._subprocess as _sub

    monkeypatch.setattr(
        _sub,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="2.1.196 (Claude Code)\n", stderr=""),
    )
    assert detect_claude_code_version("claude") is None


# ---------------------------------------------------------------------------
# Sibling co-report (#746 / #1158)
# ---------------------------------------------------------------------------


def test_sibling_gate_note_co_reports_746_and_1158() -> None:
    assert "#746" in REMOTE_CONTROL_SIBLING_GATE_NOTE
    assert "#1158" in REMOTE_CONTROL_SIBLING_GATE_NOTE
    assert "--1m" in REMOTE_CONTROL_SIBLING_GATE_NOTE


def test_sibling_note_defaults_claim_active_and_advise_1m() -> None:
    note = remote_control_sibling_gate_note(tool_search_active=True, context_1m_enabled=False)
    assert "#746" in note and "#1158" in note
    assert "keeps it on for this session" in note
    assert "restore with `headroom wrap claude --1m`" in note


def test_sibling_note_does_not_claim_disabled_tool_search_is_on() -> None:
    # Accuracy under opt-outs: --tool-search false means deferral is OFF — the
    # note must say so, not repeat the default "keeps it on" claim.
    note = remote_control_sibling_gate_note(tool_search_active=False, context_1m_enabled=False)
    assert "OFF for this session" in note
    assert "keeps it on" not in note


def test_sibling_note_does_not_advise_1m_already_passed() -> None:
    # Accuracy under opt-ins: with --1m in effect, don't advise adding it.
    note = remote_control_sibling_gate_note(tool_search_active=True, context_1m_enabled=True)
    assert "already restored via --1m" in note
    assert "restore with `headroom wrap claude --1m`" not in note


# ---------------------------------------------------------------------------
# is_custom_anthropic_base_url — string/host edges (Stage-4 matrix)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        # Native host in every spelling: scheme, http-vs-https, trailing slash,
        # port, case, and scheme-less — all NOT custom (host-equality only).
        ("https://api.anthropic.com", False),
        ("http://api.anthropic.com", False),
        ("https://api.anthropic.com/", False),
        ("https://api.anthropic.com:8443", False),
        ("https://API.ANTHROPIC.COM", False),
        ("API.ANTHROPIC.COM", False),
        ("api.anthropic.com:443", False),
        # Lookalike suffix must NOT pass — exact host match, no endswith.
        ("https://api.anthropic.com.evil.com", True),
        # Custom hosts, with and without scheme (scheme-less used to be a
        # silent false-negative: urlparse read the host as a path/scheme).
        ("http://127.0.0.1:8787", True),
        ("127.0.0.1:8787", True),
        ("myproxy.local:8080", True),
        ("evil.com", True),
        ("https://gateway.internal.example", True),
        # Valid IPv6 loopback literal — a real custom host.
        ("http://[::1]:8787", True),
        # Unset / blank — not custom (nothing overrides the default endpoint).
        ("", False),
        ("   ", False),
        (None, False),
        # Malformed values must degrade to "no host -> not custom", never
        # raise: urlparse throws ValueError("Invalid IPv6 URL") on stray
        # brackets, and these strings are user-editable (settings.json /
        # shell). The routing check flags unusable URLs separately.
        ("http://[", False),
        ("[", False),
        ("http://[::1:8787", False),
        ("http://:8080", False),
        ("http://", False),
    ],
)
def test_is_custom_anthropic_base_url_host_edges(value, expected) -> None:
    assert is_custom_anthropic_base_url(value) is expected
