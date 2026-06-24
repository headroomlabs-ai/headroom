"""Regression tests for UTF-8 decoding/encoding of headroom-owned assets.

These guard against ``UnicodeDecodeError`` on systems whose default text
encoding is not UTF-8 (e.g. Windows ``cp949``/``cp1252`` locales). Headroom
ships and writes its own templates, JSON state and config files as UTF-8, so
they must be read and written with an explicit ``encoding="utf-8"`` rather than
relying on the platform default codec. See issue #533.
"""

from __future__ import annotations

from pathlib import Path

from headroom.dashboard import TEMPLATES_DIR, get_dashboard_html
from headroom.memory.sync import _load_sync_state, _save_sync_state


def test_dashboard_template_contains_non_ascii() -> None:
    """The bundled template has non-ASCII bytes, so the bug is reproducible."""
    raw = (TEMPLATES_DIR / "dashboard.html").read_bytes()
    assert any(byte > 0x7F for byte in raw), "template expected to contain non-ASCII bytes"


def test_get_dashboard_html_reads_as_utf8(monkeypatch) -> None:
    """get_dashboard_html must decode the template as UTF-8, not the OS default.

    Before the fix, ``read_text()`` used the platform default codec and raised
    ``UnicodeDecodeError`` on non-UTF-8 locales. We assert the explicit encoding
    is passed so the regression cannot silently return (a utf-8 CI host would
    otherwise mask it).
    """
    captured: dict[str, object] = {}
    original = Path.read_text

    def _spy(self: Path, *args: object, **kwargs: object) -> str:
        captured["encoding"] = kwargs.get("encoding")
        return original(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", _spy)

    html = get_dashboard_html()

    assert captured["encoding"] == "utf-8"
    assert html  # non-empty
    # Content must equal an explicit UTF-8 decode of the raw template.
    expected = (TEMPLATES_DIR / "dashboard.html").read_bytes().decode("utf-8")
    assert html == expected


def test_sync_state_round_trips_non_ascii(tmp_path) -> None:
    """JSON sync state with non-ASCII values must survive a save/load round-trip."""
    state_path = tmp_path / "nested" / "sync_state.json"
    state = {"agent": "café", "note": "한국어 메모", "emoji": "🚀"}

    _save_sync_state(state_path, state)

    # Persisted bytes must be valid UTF-8 regardless of the platform default.
    assert state_path.read_bytes().decode("utf-8")
    assert _load_sync_state(state_path) == state
