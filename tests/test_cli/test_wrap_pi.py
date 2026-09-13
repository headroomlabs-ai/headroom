"""Tests for ``headroom wrap pi``."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from click.testing import CliRunner  # type: ignore[import-not-found]

from headroom.cli.main import main


def test_wrap_pi_missing_binary_exits_with_install_hint() -> None:
    with patch("headroom.cli.wrap.shutil.which", return_value=None):
        result = CliRunner().invoke(main, ["wrap", "pi"])

    assert result.exit_code == 1
    assert "npm install -g @earendil-works/pi-coding-agent" in result.output


def test_wrap_pi_starts_proxy_and_points_extension_at_it() -> None:
    captured: dict[str, Any] = {}

    with (
        patch("headroom.cli.wrap.shutil.which", return_value="/usr/local/bin/pi"),
        patch(
            "headroom.cli.wrap._launch_tool",
            side_effect=lambda **kwargs: captured.update(kwargs),
        ),
    ):
        result = CliRunner().invoke(main, ["wrap", "pi", "--", "-p", "fix the bug"])

    assert result.exit_code == 0, result.output
    assert captured["binary"] == "/usr/local/bin/pi"
    assert captured["tool_label"] == "PI"
    assert captured["agent_type"] == "pi"
    assert captured["launch_message"] == "Headroom extension connected"
    assert list(captured["args"]) == ["-p", "fix the bug"]
    assert captured["env"]["HEADROOM_PI_BASE_URL"] == "http://127.0.0.1:8787"
    assert captured["env_vars_display"] == ["HEADROOM_PI_BASE_URL=http://127.0.0.1:8787"]
