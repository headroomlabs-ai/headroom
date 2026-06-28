"""Tests for headroom.fsutil — encoding- and newline-safe text I/O (#733)."""

from __future__ import annotations

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
