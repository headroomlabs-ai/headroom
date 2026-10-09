from __future__ import annotations

import logging
from typing import Any

from headroom.proxy import extensions


def test_install_all_skips_failed_extension_and_continues(
    caplog,
    capsys,
    monkeypatch,
) -> None:
    calls: list[str] = []

    def good(app: Any, config: Any) -> None:
        calls.append("good")

    def bad(app: Any, config: Any) -> None:
        calls.append("bad")
        raise RuntimeError("missing optional dependency")

    monkeypatch.setattr(
        extensions,
        "discover",
        lambda: iter([("bad_ext", bad), ("good_ext", good)]),
    )

    with caplog.at_level(logging.WARNING, logger=extensions.log.name):
        installed = extensions.install_all(object(), object(), enabled=["bad_ext", "good_ext"])

    assert installed == ["good_ext"]
    assert calls == ["bad", "good"]
    assert "bad_ext" in capsys.readouterr().err
    assert "failed to install and was skipped" in caplog.text
    assert "proxy extensions skipped due to install errors: bad_ext" in caplog.text


def test_install_all_warns_for_missing_requested_extension(caplog, monkeypatch) -> None:
    monkeypatch.setattr(extensions, "discover", lambda: iter([]))

    with caplog.at_level(logging.WARNING, logger=extensions.log.name):
        installed = extensions.install_all(object(), object(), enabled=["missing_ext"])

    assert installed == []
    assert "proxy extensions requested but not found: missing_ext" in caplog.text


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _discover_warnings(monkeypatch, entry_points: Any) -> list[str]:
    monkeypatch.setattr(extensions.importlib.metadata, "entry_points", entry_points)
    capture = _Capture()
    extensions.log.addHandler(capture)
    try:
        assert list(extensions.discover()) == []
    finally:
        extensions.log.removeHandler(capture)
    return [r.getMessage() for r in capture.records if r.levelno == logging.WARNING]


def test_discover_warns_when_entry_point_enumeration_fails(monkeypatch) -> None:
    """This drops every extension, so it must not hide at DEBUG."""

    def broken(**_: Any) -> Any:
        raise RuntimeError("corrupt dist-info")

    warnings = _discover_warnings(monkeypatch, broken)

    assert len(warnings) == 1
    assert "no proxy extensions will be installed" in warnings[0]
    assert "corrupt dist-info" in warnings[0]


def test_discover_is_quiet_when_nothing_is_installed(monkeypatch) -> None:
    assert _discover_warnings(monkeypatch, lambda **_: []) == []
