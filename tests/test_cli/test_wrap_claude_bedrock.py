"""`headroom wrap claude` Bedrock-mode routing (CLAUDE_CODE_USE_BEDROCK=1).

Claude Code in Bedrock mode ignores ANTHROPIC_BASE_URL and reads
ANTHROPIC_BEDROCK_BASE_URL instead, signing each request with SigV4. So the
wrap command must (a) point ANTHROPIC_BEDROCK_BASE_URL at the proxy — not
ANTHROPIC_BASE_URL — and (b) start the proxy in re-signing mode
(``bedrock_sign=True``) so the compressed body is re-signed before it reaches
AWS. These tests pin both halves of that contract without launching a real
proxy or Claude binary.

Two layers:
  * ``_start_proxy`` forwards ``--bedrock-sign`` to the proxy subprocess.
  * ``claude()`` sets the Bedrock env key and calls ``_ensure_proxy`` with
    ``bedrock_sign=True``; the default (no Bedrock env) keeps ANTHROPIC_BASE_URL.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from headroom.cli.main import main


def test_start_proxy_forwards_bedrock_sign_flag():
    """``bedrock_sign=True`` adds ``--bedrock-sign`` to the proxy command."""
    from headroom.cli import wrap

    captured: dict = {}

    class _FakeProc:
        returncode = 0

        def poll(self):
            return None

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc()

    with patch("headroom.cli.wrap.subprocess.Popen", side_effect=fake_popen):
        with patch("headroom.cli.wrap._check_proxy", return_value=True):
            with patch("headroom.cli.wrap._get_log_path") as log_path:
                log_path.return_value = MagicMock()
                log_path.return_value.read_text.return_value = ""
                with patch("builtins.open", MagicMock()):
                    wrap._start_proxy(8787, agent_type="claude", bedrock_sign=True)

    assert "--bedrock-sign" in captured["cmd"]


def test_start_proxy_omits_bedrock_sign_by_default():
    from headroom.cli import wrap

    captured: dict = {}

    class _FakeProc:
        returncode = 0

        def poll(self):
            return None

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc()

    with patch("headroom.cli.wrap.subprocess.Popen", side_effect=fake_popen):
        with patch("headroom.cli.wrap._check_proxy", return_value=True):
            with patch("headroom.cli.wrap._get_log_path") as log_path:
                log_path.return_value = MagicMock()
                log_path.return_value.read_text.return_value = ""
                with patch("builtins.open", MagicMock()):
                    wrap._start_proxy(8787, agent_type="claude")

    assert "--bedrock-sign" not in captured["cmd"]


def _run_claude_wrap(monkeypatch, tmp_path, env: dict[str, str]):
    """Invoke ``wrap claude`` with all heavy side-effects stubbed, capturing the
    env passed to the launched Claude subprocess and the _ensure_proxy kwargs."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    for k in (
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
    ):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    captured: dict = {}

    def fake_run(cmd, env=None, **kwargs):
        captured["launch_env"] = env
        captured["launch_cmd"] = cmd

        class _R:
            returncode = 0

        return _R()

    def fake_ensure_proxy(port, no_proxy, **kwargs):
        captured["ensure_kwargs"] = kwargs
        return None

    runner = CliRunner()
    with patch("headroom.cli.wrap.shutil.which", return_value="/usr/bin/claude"):
        with patch("headroom.cli.wrap._ensure_proxy", side_effect=fake_ensure_proxy):
            with patch("headroom.cli.wrap._push_runtime_env"):
                with patch("headroom.cli.wrap._setup_rtk"):
                    with patch("headroom.cli.wrap._setup_headroom_mcp"):
                        with patch("headroom.cli.wrap._setup_serena_mcp"):
                            with patch("headroom.cli.wrap._disable_serena_mcp"):
                                with patch(
                                    "headroom.cli.wrap.subprocess.run", side_effect=fake_run
                                ):
                                    result = runner.invoke(
                                        main,
                                        ["wrap", "claude", "--no-context-tool", "--no-mcp"],
                                    )
    return result, captured


def test_bedrock_mode_sets_bedrock_base_url(monkeypatch, tmp_path):
    result, captured = _run_claude_wrap(
        monkeypatch,
        tmp_path,
        {"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_REGION": "us-west-2"},
    )

    assert result.exit_code == 0, result.output
    launch_env = captured["launch_env"]
    # The whole point: Bedrock mode routes via ANTHROPIC_BEDROCK_BASE_URL, and
    # must NOT set ANTHROPIC_BASE_URL (which Claude Code ignores in Bedrock mode).
    assert launch_env["ANTHROPIC_BEDROCK_BASE_URL"].startswith("http://127.0.0.1:8787")
    assert "ANTHROPIC_BASE_URL" not in launch_env
    # Proxy started in re-signing mode, with the resolved region.
    assert captured["ensure_kwargs"]["bedrock_sign"] is True
    assert captured["ensure_kwargs"]["region"] == "us-west-2"
    assert "Bedrock mode" in result.output


def test_non_bedrock_mode_keeps_anthropic_base_url(monkeypatch, tmp_path):
    result, captured = _run_claude_wrap(monkeypatch, tmp_path, {})

    assert result.exit_code == 0, result.output
    launch_env = captured["launch_env"]
    assert launch_env["ANTHROPIC_BASE_URL"].startswith("http://127.0.0.1:8787")
    assert "ANTHROPIC_BEDROCK_BASE_URL" not in launch_env
    assert captured["ensure_kwargs"]["bedrock_sign"] is False
