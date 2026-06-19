"""Tests for `headroom wrap pi`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli import wrap as wrap_mod
from headroom.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_wrap_pi_launches_with_transient_provider_extension(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pi is wrapped by passing a temporary extension that redirects providers."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        args = kwargs["args"]
        assert isinstance(args, tuple)
        captured["extension_source"] = Path(args[1]).read_text(encoding="utf-8")

    with patch.object(wrap_mod.shutil, "which", return_value="pi"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                result = runner.invoke(
                    main,
                    ["wrap", "pi", "--port", "9000", "--", "-p", "hello"],
                )

    assert result.exit_code == 0, result.output
    assert captured["binary"] == "pi"
    assert captured["tool_label"] == "PI"
    assert captured["agent_type"] == "pi"
    assert captured["args"] == ("-e", captured["args"][1], "-p", "hello")

    extension_source = captured["extension_source"]
    assert isinstance(extension_source, str)
    assert '"anthropic"' in extension_source
    assert '"baseUrl": "http://127.0.0.1:9000"' in extension_source
    assert '"google"' in extension_source
    assert '"baseUrl": "http://127.0.0.1:9000/v1beta"' in extension_source
    assert '"google-vertex"' in extension_source
    assert '"openai"' in extension_source
    assert '"openai-codex"' in extension_source
    assert '"baseUrl": "http://127.0.0.1:9000/v1"' in extension_source
    assert '"X-Headroom-Project":' in extension_source
    assert tmp_path.name in extension_source

    agents_md = tmp_path / "AGENTS.md"
    assert agents_md.exists()
    assert "headroom:rtk-instructions" in agents_md.read_text(encoding="utf-8")


def test_wrap_pi_missing_binary_errors_clearly(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the pi binary is missing the command fails with a clear error."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    with patch.object(wrap_mod.shutil, "which", return_value=None):
        result = runner.invoke(main, ["wrap", "pi", "--no-context-tool"])

    assert result.exit_code == 1
    assert "'pi' not found in PATH" in result.output
