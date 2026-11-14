"""A developer's GitHub/Copilot token in the shell must not reach any test.

The Copilot quota poller reads these variables and sends the token to
api.github.com, so ``tests/conftest.py`` removes them for every test.
"""

from __future__ import annotations

import os

import pytest

from headroom.subscription.copilot_quota import discover_github_token

TOKEN_VARS = (
    "GITHUB_COPILOT_GITHUB_TOKEN",
    "GITHUB_TOKEN",
    "COPILOT_GITHUB_TOKEN",
    "GITHUB_COPILOT_API_TOKEN",
    "GH_TOKEN",
)


@pytest.mark.parametrize("name", TOKEN_VARS)
def test_developer_token_is_not_visible_to_tests(name: str) -> None:
    assert name not in os.environ


def test_the_quota_poller_finds_no_token() -> None:
    assert discover_github_token() is None


def test_a_test_can_still_set_its_own_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    assert discover_github_token() == "test-token"
