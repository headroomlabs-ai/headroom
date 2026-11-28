import json
import os
import stat

import pytest
from conftest import SID
from headroom_claude_mod import launcher


def test_preflight_accepts_companion_with_session_controls(monkeypatch):
    import io
    from types import SimpleNamespace

    payload = {
        "schema_version": 1,
        "service": "headroom-claude-mod",
        "read_only": False,
        "session_controls": True,
    }
    monkeypatch.setattr(
        launcher.urllib.request,
        "build_opener",
        lambda *args: SimpleNamespace(
            open=lambda *args, **kwargs: io.BytesIO(json.dumps(payload).encode())
        ),
    )
    assert launcher.preflight("http://127.0.0.1:8787")["session_controls"] is True


def test_sidebar_can_find_companion_from_absolute_path_launch(monkeypatch, tmp_path):
    runtime = tmp_path / "Scripts"
    runtime.mkdir()
    monkeypatch.setattr(launcher.sysconfig, "get_path", lambda _: str(runtime))
    parent = {"PATH": "existing-tools"}
    env = launcher.child_environment(parent, "http://127.0.0.1:8787", SID)
    assert env["PATH"].split(os.pathsep)[0] == str(runtime)
    assert env["PATH"].endswith(os.pathsep + "existing-tools")
    assert parent == {"PATH": "existing-tools"}


@pytest.mark.parametrize(
    "url",
    [
        "https://remote.example",
        "http://evil.example",
        "http://127.0.0.1/path",
        "http://user:secret@127.0.0.1",
        "http://127.0.0.1?token=secret",
        "file:///etc/passwd",
        "http://127.0.0.1:99999",
    ],
)
def test_bad_endpoints_rejected(url):
    with pytest.raises(ValueError):
        launcher.local_url(url)


def test_header_merge_preserves_auth_and_only_replaces_own_tag():
    parent = {
        "ANTHROPIC_CUSTOM_HEADERS": "Authorization: secret\nX-Foo: bar\nX-HEADROOM-MOD-SESSION: old",
        "HOME": "/home/me",
    }
    result = launcher.child_environment(parent, "http://localhost:8787", SID)
    assert (
        result["ANTHROPIC_CUSTOM_HEADERS"]
        == f"Authorization: secret\nX-Foo: bar\nX-Headroom-Mod-Session: {SID}"
    )
    assert result["HEADROOM_MOD_SESSION_ID"] == SID
    assert result["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
    assert "old" in parent["ANTHROPIC_CUSTOM_HEADERS"]
    assert "x-headroom-session-id" not in result["ANTHROPIC_CUSTOM_HEADERS"].lower()


def test_inherited_upstream_is_never_silently_replaced():
    with pytest.raises(ValueError, match="not overwritten"):
        launcher.child_environment(
            {"ANTHROPIC_BASE_URL": "https://company.example"}, "http://127.0.0.1:8787", SID
        )


@pytest.mark.parametrize(
    "flag", ["CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_FOUNDRY"]
)
def test_provider_bypass_flags_fail_explicitly(flag):
    with pytest.raises(ValueError, match="bypasses"):
        launcher.child_environment({flag: "1"}, "http://127.0.0.1:8787", SID)


def test_malformed_custom_header_fails_without_rewrite():
    with pytest.raises(ValueError, match="Malformed"):
        launcher.child_environment(
            {"ANTHROPIC_CUSTOM_HEADERS": "invalid header"}, "http://127.0.0.1:8787", SID
        )


def test_launch_executes_argv_without_shell_and_propagates_exit(tmp_path, monkeypatch):
    if os.name == "nt":
        pytest.skip("POSIX executable fixture; native Windows live gate is separate")
    capture = tmp_path / "launch.json"
    fake = tmp_path / "fake claude"
    fake.write_text(
        "#!/usr/bin/env python3\nimport os,sys,json\nfrom pathlib import Path\nPath(os.environ['CAPTURE']).write_text(json.dumps({'argv':sys.argv[1:],'sid':os.environ['HEADROOM_MOD_SESSION_ID'],'headers':os.environ['ANTHROPIC_CUSTOM_HEADERS']}))\nsys.exit(7)\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("CAPTURE", str(capture))
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setattr(launcher, "preflight", lambda base: {"version": "test"})
    code = launcher.main(
        [
            "run",
            "--session-id",
            SID,
            "--claude-executable",
            str(fake),
            "--",
            "--model",
            "test model",
            "literal ; & $(not-a-shell)",
        ]
    )
    assert code == 7
    result = json.loads(capture.read_text())
    assert result["argv"] == [
        "--session-id",
        SID,
        "--model",
        "test model",
        "literal ; & $(not-a-shell)",
    ]
    assert result["sid"] == SID


@pytest.mark.parametrize("flag", ["--resume", "--continue", "--session-id=xxx", "--fork-session"])
def test_session_overrides_cannot_bypass_correlation(flag):
    assert launcher.main(["run", "--", flag]) == 2


def test_explicit_port_zero_rejected_but_port_80_preserved():
    with pytest.raises(ValueError):
        launcher.local_url("http://127.0.0.1:0")
    assert launcher.local_url("http://127.0.0.1:80") == "http://127.0.0.1:80"


@pytest.mark.parametrize(
    "flag", ["--cloud", "--desktop", "--bg", "--bare", "--safe-mode", "--print"]
)
def test_sidebar_requires_interactive_local_mode(flag):
    assert launcher.main(["run", "--", flag]) == 2
