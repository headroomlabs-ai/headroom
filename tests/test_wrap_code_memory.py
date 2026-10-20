"""Code-memory MCP is selectable via --code-memory (default serena).

Covers the resolver precedence (selector > deprecated flags > default), the
graceful retirement of the removed ``tokensave`` option, the orchestrator
dispatch for each selection, and that --code-memory is exposed on the
code-memory-capable subcommands (claude/codex/grok) but not others.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from headroom.cli import wrap


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("HEADROOM_CODE_MEMORY", None)
    return env


def test_default_is_serena() -> None:
    with patch.dict(os.environ, _clean_env(), clear=True):
        assert wrap._resolve_code_memory({}) == wrap._CODE_MEMORY_SERENA


def test_selector_env_wins() -> None:
    for val in (wrap._CODE_MEMORY_SERENA, wrap._CODE_MEMORY_NONE):
        with patch.dict(os.environ, {"HEADROOM_CODE_MEMORY": val}):
            # selector beats any legacy flag
            assert wrap._resolve_code_memory({"serena": True, "no_serena": True}) == val


def test_deprecated_flags_map_into_selector() -> None:
    with patch.dict(os.environ, _clean_env(), clear=True):
        assert wrap._resolve_code_memory({"serena": True}) == wrap._CODE_MEMORY_SERENA
        # tokensave is retired: --no-tokensave is now a no-op → default serena
        assert wrap._resolve_code_memory({"no_tokensave": True}) == wrap._CODE_MEMORY_SERENA
        # --no-serena means "no code memory" now that tokensave is gone
        assert wrap._resolve_code_memory({"no_serena": True}) == wrap._CODE_MEMORY_NONE


def test_retired_tokensave_selector_maps_to_serena() -> None:
    # An explicit HEADROOM_CODE_MEMORY=tokensave (or --code-memory tokensave from
    # an old script) degrades gracefully to Serena instead of erroring.
    with patch.dict(os.environ, {"HEADROOM_CODE_MEMORY": "tokensave"}):
        assert wrap._resolve_code_memory({}) == wrap._CODE_MEMORY_SERENA


def test_serena_dashboard_disabled_flips_existing_config(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = tmp_path / ".serena" / "serena_config.yml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        "web_dashboard: true\nweb_dashboard_open_on_launch: true\ngui_log_window: false\n"
    )
    wrap._ensure_serena_dashboard_disabled()
    text = cfg.read_text()
    assert "web_dashboard_open_on_launch: false" in text
    assert "web_dashboard: true" in text  # other keys preserved


def test_serena_dashboard_disabled_handles_unset_value(tmp_path, monkeypatch) -> None:
    """Serena's template ships ``web_dashboard_open_on_launch:`` with no value.

    Requiring a value in the regex treated the key as absent and appended a
    second copy of it. Serena loads this file with ruamel in duplicate-key-strict
    mode, so the duplicate made every later startup die with ``DuplicateKeyError``
    before the MCP handshake ("Connection closed").
    """
    import yaml

    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = tmp_path / ".serena" / "serena_config.yml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        "web_dashboard: true\n"
        "web_dashboard_open_on_launch:\n"
        "\n"
        "gui_log_window: false\n"
        "projects:\n- /home/me/work/api\n"
    )

    wrap._ensure_serena_dashboard_disabled()

    text = cfg.read_text()
    assert text.count("web_dashboard_open_on_launch") == 1, "duplicate key breaks Serena"
    assert "web_dashboard_open_on_launch: false" in text
    assert "\n\ngui_log_window: false\n" in text, "blank line after the key must survive"
    parsed = yaml.safe_load(text)
    assert parsed["web_dashboard_open_on_launch"] is False
    assert parsed["web_dashboard"] is True
    assert parsed["projects"] == ["/home/me/work/api"]

    first = text
    wrap._ensure_serena_dashboard_disabled()
    assert cfg.read_text() == first


def test_serena_dashboard_disabled_keeps_inline_comment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = tmp_path / ".serena" / "serena_config.yml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("projects: []\nweb_dashboard_open_on_launch: true  # keep this\n")

    wrap._ensure_serena_dashboard_disabled()

    text = cfg.read_text()
    assert "web_dashboard_open_on_launch: false  # keep this" in text
    assert text.count("web_dashboard_open_on_launch") == 1


def test_serena_dashboard_disabled_keeps_comment_on_unset_value(tmp_path, monkeypatch) -> None:
    """A comment right after the colon must stay a comment, not become the value.

    ``web_dashboard_open_on_launch: # note`` used to turn into
    ``...: false note``, which YAML reads as the string ``"false note"`` rather
    than the boolean ``False``.
    """
    import yaml

    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = tmp_path / ".serena" / "serena_config.yml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        "projects: []\n"
        "web_dashboard_open_on_launch: # note: leave unset for now\n"
        "gui_log_window: false\n"
    )

    wrap._ensure_serena_dashboard_disabled()

    text = cfg.read_text()
    assert "web_dashboard_open_on_launch: false # note: leave unset for now" in text
    assert yaml.safe_load(text)["web_dashboard_open_on_launch"] is False

    first = text
    wrap._ensure_serena_dashboard_disabled()
    assert cfg.read_text() == first


def test_serena_dashboard_disabled_keeps_comment_glued_to_colon(tmp_path, monkeypatch) -> None:
    """A ``#`` with no space after the colon is still the comment.

    ``web_dashboard_open_on_launch:# note`` does not match a pattern that
    requires the value token to be non-comment text, so the line used to be
    left in place and a second copy of the key was appended.
    """
    import yaml

    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = tmp_path / ".serena" / "serena_config.yml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        "projects: []\n"
        "web_dashboard_open_on_launch:# note: leave unset for now\n"
        "gui_log_window: false\n"
    )

    wrap._ensure_serena_dashboard_disabled()

    text = cfg.read_text()
    assert text.count("web_dashboard_open_on_launch") == 1
    assert "web_dashboard_open_on_launch: false # note: leave unset for now" in text
    assert yaml.safe_load(text)["web_dashboard_open_on_launch"] is False

    first = text
    wrap._ensure_serena_dashboard_disabled()
    assert cfg.read_text() == first


def test_serena_dashboard_disabled_repairs_duplicate_keys(tmp_path, monkeypatch) -> None:
    """Configs already broken by an older Headroom must come back valid.

    A duplicate top-level key makes Serena's ruamel loader raise, so rewriting
    both copies in place would leave the MCP server unable to start.
    """
    import yaml

    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = tmp_path / ".serena" / "serena_config.yml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        "web_dashboard: true\n"
        "web_dashboard_open_on_launch:\n"
        "\n"
        "# other settings\n"
        "gui_log_window: false\n"
        "projects:\n- /home/me/work/api\n"
        "web_dashboard_open_on_launch: false\n"
    )

    wrap._ensure_serena_dashboard_disabled()

    text = cfg.read_text()
    assert text.count("web_dashboard_open_on_launch") == 1, "duplicate key breaks Serena"
    parsed = yaml.safe_load(text)
    assert parsed["web_dashboard_open_on_launch"] is False
    assert parsed["projects"] == ["/home/me/work/api"]
    assert parsed["web_dashboard"] is True

    first = text
    wrap._ensure_serena_dashboard_disabled()
    assert cfg.read_text() == first


def test_serena_dashboard_disabled_ignores_nested_key(tmp_path, monkeypatch) -> None:
    """Only the column-0 key is Serena's setting; an indented one is not.

    A line edit that matches the key name regardless of indentation can
    rewrite an unrelated nested mapping's key, and can mistake it for a
    duplicate of the real top-level setting and delete the real one.
    """
    import yaml

    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = tmp_path / ".serena" / "serena_config.yml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        "extension:\n"
        "  web_dashboard_open_on_launch: true\n"
        "web_dashboard_open_on_launch: true\n"
        "projects: []\n"
    )

    wrap._ensure_serena_dashboard_disabled()

    text = cfg.read_text()
    assert "  web_dashboard_open_on_launch: true\n" in text, "nested key must be left untouched"
    parsed = yaml.safe_load(text)
    assert parsed["web_dashboard_open_on_launch"] is False
    assert parsed["extension"]["web_dashboard_open_on_launch"] is True

    first = text
    wrap._ensure_serena_dashboard_disabled()
    assert cfg.read_text() == first


def test_serena_dashboard_disabled_drops_non_comment_trailing_content(
    tmp_path, monkeypatch
) -> None:
    """Trailing content that is not a comment must not survive the rewrite.

    E.g. an explicit YAML type tag (``!!bool true``) has a value token plus a
    second, space-separated token. Preserving that second token as if it were
    a comment turns the rewritten line into ``false true``, which YAML parses
    as the string ``"false true"`` instead of the boolean ``False``.
    """
    import yaml

    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = tmp_path / ".serena" / "serena_config.yml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("projects: []\nweb_dashboard_open_on_launch: !!bool true\n")

    wrap._ensure_serena_dashboard_disabled()

    text = cfg.read_text()
    assert "web_dashboard_open_on_launch: false\n" in text
    assert yaml.safe_load(text)["web_dashboard_open_on_launch"] is False


def test_serena_config_is_never_created_by_headroom(tmp_path, monkeypatch) -> None:
    """Headroom must NOT pre-empt Serena's own config bootstrap (#2674).

    This is the exact invariant, and it is the reason the outage happened.
    Verified against Serena 1.6.2.dev0 ``serena/config/serena_config.py``: Serena
    autogenerates a complete config only when the path does not exist; once any
    file is there it validates instead, and a missing ``projects`` key is fatal
    (``SerenaConfigError``). Headroom used to write a one-key bootstrap file,
    which killed Serena's MCP handshake on every fresh install.

    Asserting "we write nothing" is stronger than asserting which keys we write:
    it stays correct even if Serena adds a new required key, whereas a test that
    pins our own key list would go on passing while users broke again.
    """
    monkeypatch.setenv("HOME", str(tmp_path))

    wrap._ensure_serena_dashboard_disabled()

    cfg = tmp_path / ".serena" / "serena_config.yml"
    assert not cfg.exists(), "Headroom created a config Serena would have generated itself"


def test_serena_dashboard_disabled_repairs_config_missing_projects(tmp_path, monkeypatch) -> None:
    """Backfill ``projects`` into a config an older Headroom already wrote (#2674).

    Users who ran an affected version have the single-key file on disk, so simply
    not creating new bad files would leave them broken forever.
    """
    import yaml

    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = tmp_path / ".serena" / "serena_config.yml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("web_dashboard_open_on_launch: false\n")

    wrap._ensure_serena_dashboard_disabled()

    parsed = yaml.safe_load(cfg.read_text())
    assert parsed["projects"] == []
    assert parsed["web_dashboard_open_on_launch"] is False


def test_serena_dashboard_disabled_preserves_registered_projects(tmp_path, monkeypatch) -> None:
    """Never clobber Serena's real project registry — it is user data."""
    import yaml

    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = tmp_path / ".serena" / "serena_config.yml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        "# my serena config\nprojects:\n  - /home/me/work/api\n  - /home/me/work/web\n"
        "web_dashboard_open_on_launch: true\n"
    )

    wrap._ensure_serena_dashboard_disabled()

    text = cfg.read_text()
    parsed = yaml.safe_load(text)
    assert parsed["projects"] == ["/home/me/work/api", "/home/me/work/web"]
    assert parsed["web_dashboard_open_on_launch"] is False
    assert "# my serena config" in text  # comments preserved
    assert text.count("projects:") == 1  # no duplicate key


def test_serena_dashboard_disabled_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = tmp_path / ".serena" / "serena_config.yml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("projects: []\nweb_dashboard_open_on_launch: true\n")
    wrap._ensure_serena_dashboard_disabled()
    first = cfg.read_text()
    wrap._ensure_serena_dashboard_disabled()
    assert cfg.read_text() == first


def test_serena_config_required_keys_match_serena_source() -> None:
    """Pin the assumption this fix rests on, against Serena's real source (#2674).

    Skipped unless a Serena checkout is present. When it is, this proves the claim
    the fix depends on — that ``projects`` is the *only* hard-required key and that
    Serena bootstraps only a missing file — rather than trusting a bug report.
    Set ``SERENA_SRC`` to a Serena source tree to enable it.
    """
    import os
    import re
    from pathlib import Path

    src = os.environ.get("SERENA_SRC", "")
    if not src or not (Path(src) / "config" / "serena_config.py").is_file():
        pytest.skip("set SERENA_SRC to a Serena source tree to run this check")

    text = (Path(src) / "config" / "serena_config.py").read_text(encoding="utf-8")
    # Serena bootstraps only when the file is absent — so we must not create one.
    assert re.search(r"if not os\.path\.exists\(config_file_path\)", text)
    # `projects` is the sole fatal omission; everything else has a default.
    fatal = re.findall(r"raise SerenaConfigError\((.*?)\)", text, re.DOTALL)
    projects_fatal = [f for f in fatal if "projects" in f]
    assert projects_fatal, "Serena no longer rejects a missing `projects` key"
    assert "get_value_or_default" in text, "Serena's default-filling path changed"


def test_invalid_env_raises() -> None:
    with patch.dict(os.environ, {"HEADROOM_CODE_MEMORY": "bogus"}):
        try:
            wrap._resolve_code_memory({})
        except click.ClickException:
            pass
        else:  # pragma: no cover
            raise AssertionError("invalid HEADROOM_CODE_MEMORY should raise ClickException")


def _dispatch_calls(selection: str, extra: dict | None = None) -> list[str]:
    """Run the orchestrator with a given selection, recording which setup/disable
    helpers fire (all mocked)."""
    calls: list[str] = []
    env = _clean_env()
    env["HEADROOM_CODE_MEMORY"] = selection
    with (
        patch.dict(os.environ, env, clear=True),
        patch.object(wrap, "_setup_serena_mcp", lambda *a, **k: calls.append("serena")),
        patch.object(
            wrap, "_disable_tokensave_mcp", lambda *a, **k: calls.append("disable_tokensave")
        ),
        patch.object(wrap, "_disable_serena_mcp", lambda *a, **k: calls.append("disable_serena")),
    ):
        wrap._setup_coding_compressor(object(), serena_context="claude-code", **(extra or {}))
    return calls


def test_orchestrator_dispatch() -> None:
    # A legacy tokensave entry is always retired first, then the selection applies.
    assert _dispatch_calls(wrap._CODE_MEMORY_SERENA) == ["disable_tokensave", "serena"]
    assert set(_dispatch_calls(wrap._CODE_MEMORY_NONE)) == {"disable_tokensave", "disable_serena"}


def test_code_memory_option_present_only_on_code_memory_agents() -> None:
    runner = CliRunner()
    for tool in ("claude", "codex", "grok"):
        out = runner.invoke(wrap.wrap, [tool, "--help"]).output
        assert "--code-memory" in out, f"--code-memory missing from `wrap {tool} --help`"
    # aider does not register a code-memory MCP → no flag
    out = runner.invoke(wrap.wrap, ["aider", "--help"]).output
    assert "--code-memory" not in out
