"""Tests for ``should_stamp_codex_client`` — the path-based ``X-Client: codex``
stamp on the Responses endpoint.

The stamp fires only for an unidentified caller on the Responses endpoint, so
Codex Desktop (whose User-Agent isn't a known codex UA) takes the codex
fail-open path instead of being refused with a 413 on a compression timeout.
"""

from __future__ import annotations

from headroom.proxy.auth_mode import classify_client, should_stamp_codex_client

CODEX_DESKTOP_UA = (
    "Codex Desktop/0.140.0-alpha.2 (Mac OS 15.7.7; arm64) unknown (Codex Desktop; 26.609.71450)"
)


def test_unidentified_codex_desktop_on_responses_is_stamped() -> None:
    assert should_stamp_codex_client("/v1/responses", {"user-agent": CODEX_DESKTOP_UA})


def test_stamp_then_classify_yields_codex() -> None:
    # End-to-end of what the HTTP middleware and the WS handler both do:
    # stamp the header, after which classify_client must read "codex".
    headers = {"user-agent": CODEX_DESKTOP_UA}
    assert should_stamp_codex_client("/v1/responses", headers)
    headers["x-client"] = "codex"
    assert classify_client(headers) == "codex"


def test_no_user_agent_on_responses_is_stamped() -> None:
    assert should_stamp_codex_client("/v1/responses", {})


def test_responses_subpath_is_stamped() -> None:
    assert should_stamp_codex_client("/v1/responses/foo", {"user-agent": CODEX_DESKTOP_UA})


def test_other_path_is_not_stamped() -> None:
    # Scoped to the Responses endpoint; unknown callers elsewhere are untouched.
    assert not should_stamp_codex_client("/v1/chat/completions", {"user-agent": CODEX_DESKTOP_UA})


def test_recognized_non_codex_client_is_not_stamped() -> None:
    assert not should_stamp_codex_client("/v1/responses", {"user-agent": "claude-code/1.2.3"})


def test_recognized_codex_cli_is_not_stamped() -> None:
    # Already classifies as codex via UA; no stamp needed.
    assert not should_stamp_codex_client("/v1/responses", {"user-agent": "codex-cli/0.5"})


def test_explicit_x_client_is_not_stamped() -> None:
    assert not should_stamp_codex_client(
        "/v1/responses", {"x-client": "aider", "user-agent": CODEX_DESKTOP_UA}
    )
