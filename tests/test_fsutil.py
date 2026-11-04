"""Tests for headroom.fsutil — encoding- and newline-safe text I/O (#733)."""

from __future__ import annotations

import os
import stat

import pytest

from headroom import fsutil


def test_write_text_does_not_double_existing_crlf(tmp_path):
    """A string containing \\r\\n must be written verbatim, never as \\r\\r\\n."""
    p = tmp_path / "config.toml"
    fsutil.write_text(p, 'model = "gpt-5"\r\nport = 8787\r\n')
    raw = p.read_bytes()
    assert b"\r\r\n" not in raw
    assert raw == b'model = "gpt-5"\r\nport = 8787\r\n'


def test_write_text_does_not_translate_lf(tmp_path):
    """\\n must stay \\n on every platform (no \\r\\n rewrite)."""
    p = tmp_path / "hook.sh"
    fsutil.write_text(p, "#!/bin/sh\necho hi\n")
    assert p.read_bytes() == b"#!/bin/sh\necho hi\n"


def test_read_text_normalises_crlf(tmp_path):
    """read_text returns universal-newline (\\n) text, so a round trip can't double CRLF."""
    p = tmp_path / "config.toml"
    p.write_bytes(b"a = 1\r\nb = 2\r\n")
    text = fsutil.read_text(p)
    assert text == "a = 1\nb = 2\n"
    fsutil.write_text(p, text)
    assert b"\r" not in p.read_bytes()


def test_read_text_roundtrips_utf8_non_ascii(tmp_path):
    p = tmp_path / "config.toml"
    fsutil.write_text(p, 'project = "比赛/机器人"\n')
    assert fsutil.read_text(p) == 'project = "比赛/机器人"\n'


def test_write_text_leaves_original_intact_when_write_fails(tmp_path, monkeypatch):
    """A failed write must leave the previous file whole, not truncated.

    ``open(path, "w")`` truncates to zero bytes before the new content lands, so
    a crash mid-write used to leave users with a half-written ~/.claude.json.
    """
    p = tmp_path / "settings.json"
    fsutil.write_text(p, '{"permissions": {"allow": ["Bash"]}}\n')

    def boom(src, dst):
        raise OSError("no space left on device")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        fsutil.write_text(p, '{"hooks": {}}\n')

    assert p.read_text() == '{"permissions": {"allow": ["Bash"]}}\n'
    # No .tmp litter left behind in the config directory.
    assert [f.name for f in tmp_path.iterdir()] == ["settings.json"]


def test_write_text_follows_symlink_instead_of_replacing_it(tmp_path):
    """Dotfile managers symlink these configs; os.replace would clobber the link."""
    real = tmp_path / "real.json"
    real.write_text("{}\n")
    link = tmp_path / "settings.json"
    link.symlink_to(real)

    fsutil.write_text(link, '{"env": {}}\n')

    assert link.is_symlink()
    assert real.read_text() == '{"env": {}}\n'


def test_write_text_preserves_existing_mode(tmp_path):
    """mkstemp creates 0600 — an existing 0644 config must not be silently tightened."""
    p = tmp_path / "settings.json"
    p.write_text("{}\n")
    p.chmod(0o644)
    fsutil.write_text(p, '{"a": 1}\n')
    assert stat.S_IMODE(p.stat().st_mode) == 0o644


def test_read_text_falls_back_to_locale_encoding(tmp_path, monkeypatch):
    """A file a tool wrote in the locale encoding (e.g. GBK) still decodes."""
    monkeypatch.setattr(fsutil.locale, "getpreferredencoding", lambda *_: "gbk")
    p = tmp_path / "config.toml"
    p.write_bytes('path = "模型"\n'.encode("gbk"))  # not valid UTF-8
    assert fsutil.read_text(p) == 'path = "模型"\n'


def test_read_text_replace_fallback_never_raises(tmp_path, monkeypatch):
    """When neither UTF-8 nor the locale encoding decodes, fall back to replace."""
    monkeypatch.setattr(fsutil.locale, "getpreferredencoding", lambda *_: "ascii")
    p = tmp_path / "config.toml"
    p.write_bytes(b"\xff\xfe bad bytes")
    # Must not raise; returns *something* decodable.
    assert isinstance(fsutil.read_text(p), str)


def test_read_text_missing_returns_default(tmp_path):
    p = tmp_path / "nope.toml"
    assert fsutil.read_text(p, default="") == ""


def test_read_text_missing_raises_without_default(tmp_path):
    with pytest.raises(OSError):
        fsutil.read_text(tmp_path / "nope.toml")


def test_append_text_preserves_endings(tmp_path):
    p = tmp_path / "AGENTS.md"
    fsutil.write_text(p, "line1\n")
    fsutil.append_text(p, "line2\n")
    assert p.read_bytes() == b"line1\nline2\n"
