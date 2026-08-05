"""Runtime helpers for Claude-facing integrations."""

from __future__ import annotations

import re
from collections.abc import Mapping
from urllib.parse import urlparse

DEFAULT_API_URL = "https://api.anthropic.com"

# GH #746: Claude Code stops deferring MCP/system tool schemas (materializing
# every one into its context window) when ANTHROPIC_BASE_URL is a custom host
# and ENABLE_TOOL_SEARCH is unset. Every place that points Claude Code at the
# proxy must keep deferral on, so the env key and its default live here as the
# single source of truth shared by `wrap`, `init`, and `install`.
TOOL_SEARCH_ENV = "ENABLE_TOOL_SEARCH"
TOOL_SEARCH_DEFAULT = "true"
REMOTE_CONTROL_BASE_URL_ENV = "ANTHROPIC_BASE_URL"
REMOTE_CONTROL_FEATURE = "Remote Control"

# GH #1779: Claude Code v2.1.196 added a client-side eligibility check that
# DISABLES first-party Remote Control (`/remote-control` / `/rc`, which mirrors a
# local CLI session to claude.ai/code and the mobile apps) whenever
# ANTHROPIC_BASE_URL points at a non-`api.anthropic.com` host. Headroom routes
# through http://127.0.0.1:<port>, so on this version and newer the disable is
# DETERMINISTIC (not "may") — the `/rc` command simply vanishes. The gate is
# upstream in the Claude Code binary and RC's control-plane talks to claude.ai,
# not the API host, so Headroom cannot force it back on; the honest fix is an
# accurate warning at launch/doctor time. This is the same base-URL gating
# family as #746 (on-demand tool loading) and #1158 (1M context window), both of
# which Headroom *can* restore (see the sibling-gate note below).
REMOTE_CONTROL_GATED_MIN_VERSION = (2, 1, 196)

# Auth-mode signals that mean Remote Control was NEVER available for this
# session, so its gate warning must not fire (issue #1779). RC mirrors a local
# CLI session to a claude.ai account — a Claude Pro/Max *subscription* feature.
# API-key (PAYG) callers and cloud IAM/ADC callers (Bedrock / Vertex / Foundry)
# have no claude.ai session to mirror and never saw the `/rc` command. Presence
# of any of these (non-empty) in the effective environment means "not a
# subscription session — stay silent."
REMOTE_CONTROL_NON_SUBSCRIPTION_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
)

# Co-reported alongside the RC gate so the user sees the whole base-URL gating
# family in one place (issue #1779). Unlike RC, Headroom *does* restore these two
# siblings — #746 by default, #1158 on request — which is the point of showing
# them together: RC is the one member of the family Headroom cannot fix.
# This constant describes DEFAULT behaviour and is the right form for `doctor`,
# which cannot see the wrap launch flags. `wrap` knows its flags and must use
# :func:`remote_control_sibling_gate_note` instead, so the note never claims
# tool deferral is on for a session where the user turned it off, nor tells a
# user to pass `--1m` they already passed.
REMOTE_CONTROL_SIBLING_GATE_NOTE = (
    "Same base-URL gate also affects on-demand tool loading "
    "(#746 — `headroom wrap claude` keeps it on by default) and the 1M context "
    "window (#1158 — opt in with `headroom wrap claude --1m`)."
)


def remote_control_sibling_gate_note(*, tool_search_active: bool, context_1m_enabled: bool) -> str:
    """Session-accurate sibling-gate co-report for the wrap launch banner.

    Unlike the flag-blind :data:`REMOTE_CONTROL_SIBLING_GATE_NOTE`, this
    reflects what THIS session actually does (issue #1779 accuracy rule: never
    show the user a claim the session contradicts):

    * ``tool_search_active`` — whether the resolved ``ENABLE_TOOL_SEARCH`` mode
      keeps deferral on (#746). ``False`` when the user chose a falsy mode.
    * ``context_1m_enabled`` — whether ``--1m`` was passed (#1158); if so, don't
      advise adding a flag that is already in effect.
    """
    tool_part = (
        "#746 — Headroom keeps it on for this session"
        if tool_search_active
        else "#746 — OFF for this session per your --tool-search/ENABLE_TOOL_SEARCH setting"
    )
    context_part = (
        "#1158 — already restored via --1m"
        if context_1m_enabled
        else "#1158 — restore with `headroom wrap claude --1m`"
    )
    return (
        "Same base-URL gate also affects on-demand tool loading "
        f"({tool_part}) and the 1M context window ({context_part})."
    )


_CLAUDE_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def _version_str(version: tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in version)


def remote_control_gate_message(source: str, *, version: tuple[int, int, int] | None = None) -> str:
    """Return the Remote Control gate message for Claude warning paths.

    Accuracy matters here (issue #1779): on Claude Code
    :data:`REMOTE_CONTROL_GATED_MIN_VERSION` and newer the disable is
    deterministic, so the wording states it as fact — never "may".

    * ``version`` known and gated → name the exact version and state the
      deterministic disable.
    * ``version`` unknown (``None``) → state the version threshold and let the
      user self-identify, without falsely asserting their build.

    Callers gate on :func:`remote_control_gate_active` first, so a version known
    to be *older* than the threshold never reaches this function.
    """
    source_clean = source.strip() or "this endpoint"
    min_ver = _version_str(REMOTE_CONTROL_GATED_MIN_VERSION)
    if version is not None and version >= REMOTE_CONTROL_GATED_MIN_VERSION:
        lead = (
            f"Claude Code {_version_str(version)} disables the "
            "/remote-control (/rc) command while "
            f"{REMOTE_CONTROL_BASE_URL_ENV} points at a custom endpoint "
            f"({source_clean})."
        )
    else:
        lead = (
            f"Claude Code {min_ver}+ disables the /remote-control (/rc) command "
            f"while {REMOTE_CONTROL_BASE_URL_ENV} points at a custom endpoint "
            f"({source_clean}); if your Claude Code is {min_ver} or newer, /rc "
            "is unavailable in this session."
        )
    return (
        f"{REMOTE_CONTROL_FEATURE}: {lead} "
        "Headroom cannot override this client-side gate — run Claude without "
        "Headroom for sessions that need Remote Control."
    )


def is_custom_anthropic_base_url(value: str | None) -> bool:
    """Return whether ANTHROPIC_BASE_URL is custom from Claude's Remote Control gate view.

    Host-equality only (issue #1779): the scheme, port, path, and trailing
    slash are ignored, and matching is exact — a lookalike such as
    ``api.anthropic.com.evil.com`` is custom. Scheme-less values
    (``myproxy.local:8080``, ``127.0.0.1:8787``) are re-parsed as a network
    location; ``urlparse`` alone reads them as a path (or treats the host as a
    URL *scheme*), yielding no hostname — which silently classified every
    scheme-less custom host as "not custom" and suppressed the warning.
    """
    raw = (value or "").strip()
    if not raw:
        return False
    return _gate_view_host(raw) not in {"", "api.anthropic.com"}


def _gate_view_host(raw: str) -> str:
    """Best-effort host extraction for the Remote Control gate view.

    ``urlparse`` raises ``ValueError`` on bracket-malformed input (e.g. the
    typo'd IPv6 literal ``http://[::1:8787``). These values are user-editable
    (shell env / settings.json), and the doctor path must degrade to "no host"
    rather than crash (issue #1779). Host-less results classify as not-custom;
    the routing check separately flags unusable URLs, so nothing is hidden.
    """
    try:
        host = (urlparse(raw).hostname or "").strip().lower()
    except ValueError:
        return ""
    if not host and "://" not in raw:
        try:
            host = (urlparse(f"//{raw}").hostname or "").strip().lower()
        except ValueError:
            return ""
    return host


def remote_control_applies_to_auth(environ: Mapping[str, object]) -> bool:
    """Return whether this auth mode is one that ever had Remote Control.

    ``False`` for API-key (PAYG) and cloud IAM/ADC sessions — they never saw the
    ``/rc`` command, so the gate warning must stay silent for them (issue
    #1779). See :data:`REMOTE_CONTROL_NON_SUBSCRIPTION_ENV`.
    """
    return not any(
        str(environ.get(key) or "").strip() for key in REMOTE_CONTROL_NON_SUBSCRIPTION_ENV
    )


def parse_claude_code_version(text: str | None) -> tuple[int, int, int] | None:
    """Parse a ``MAJOR.MINOR.PATCH`` version out of ``claude --version`` output.

    ``claude --version`` prints e.g. ``2.1.196 (Claude Code)``. Returns the first
    dotted triple found, or ``None`` when nothing parses (unknown version).
    """
    match = _CLAUDE_VERSION_RE.search(text or "")
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def detect_claude_code_version(claude_bin: str | None = None) -> tuple[int, int, int] | None:
    """Best-effort detection of the installed Claude Code version.

    Runs ``claude --version`` and parses it. Returns ``None`` on any failure
    (binary missing, non-zero exit, timeout, unparseable or absent output) so
    callers fall back to the version-unknown wording rather than crash. Never
    raises. Uses the shared ``headroom._subprocess`` wrapper, which forces
    ``encoding="utf-8"`` under ``text=True`` (the repo's Windows-cp1252 guard).
    """
    import shutil
    import subprocess

    from headroom._subprocess import run

    binary = claude_bin or shutil.which("claude")
    if not binary:
        return None
    try:
        proc = run([binary, "--version"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    # A non-zero exit is a detection failure, per this function's contract: a
    # failing `claude --version` may still print a version-shaped string, and
    # trusting it would surface a false *exact*-version warning instead of the
    # self-qualified "2.1.196+ / unknown" path the callers rely on. getattr with
    # a success default keeps stubbed CompletedProcess objects working.
    if getattr(proc, "returncode", 0) != 0:
        return None
    # getattr, not attribute access: a stubbed CompletedProcess (e.g. a test's
    # SimpleNamespace) may lack stdout/stderr — degrade to "unknown", never raise.
    stdout = getattr(proc, "stdout", "") or ""
    stderr = getattr(proc, "stderr", "") or ""
    return parse_claude_code_version(f"{stdout} {stderr}")


def remote_control_gate_active(
    base_url: str | None,
    environ: Mapping[str, object],
    version: tuple[int, int, int] | None,
) -> bool:
    """Whether to surface the Remote Control gate warning for this session.

    ``True`` only when ALL hold (issue #1779):

    * ``base_url`` is a custom (non-``api.anthropic.com``) endpoint — the gate's
      trigger,
    * the auth mode is one that ever had Remote Control (not API-key / cloud
      IAM) — so PAYG users never see a warning for a feature they never had,
    * the Claude Code version is at or above
      :data:`REMOTE_CONTROL_GATED_MIN_VERSION`, **or** unknown (``None``).

    Returns ``False`` when the version is known to be *older* than the gate — on
    those builds Remote Control is unaffected by a custom base URL, so warning
    would be a false alarm.
    """
    if not is_custom_anthropic_base_url(base_url):
        return False
    if not remote_control_applies_to_auth(environ):
        return False
    if version is not None and version < REMOTE_CONTROL_GATED_MIN_VERSION:
        return False
    return True


def proxy_base_url(port: int) -> str:
    """Return the local proxy base URL used by Claude integrations."""
    return f"http://127.0.0.1:{port}"
