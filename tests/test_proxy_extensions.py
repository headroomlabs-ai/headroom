from __future__ import annotations

import logging
from typing import Any

import pytest

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


def test_install_all_aborts_startup_when_fail_closed_extension_fails(
    monkeypatch,
) -> None:
    """A ``fail_closed`` extension's failure must stop startup, not be skipped.

    Skipping an extension that mints the upstream credential would leave the
    proxy serving with the client's own ``Authorization`` relayed upstream
    (issue #4737). Every other extension keeps the skip-and-continue default,
    including the ones installed before the failing one.
    """

    def auth_ext(app: Any, config: Any) -> None:
        raise RuntimeError("headroom-oauth2 misconfigured: token_url must be https")

    auth_ext.fail_closed = True  # type: ignore[attr-defined]

    def good(app: Any, config: Any) -> None:
        raise AssertionError("must not install extensions after a fatal one")

    monkeypatch.setattr(
        extensions,
        "discover",
        lambda: iter([("oauth2", auth_ext), ("good_ext", good)]),
    )

    with pytest.raises(RuntimeError) as excinfo:
        extensions.install_all(object(), object(), enabled=["oauth2", "good_ext"])

    assert "'oauth2'" in str(excinfo.value)
    assert "fail_closed" in str(excinfo.value)
    # The plugin's own message survives as the cause, so the operator sees why.
    assert "token_url must be https" in str(excinfo.value.__cause__)


def test_install_all_skips_fail_closed_extension_when_it_succeeds(
    monkeypatch,
) -> None:
    """The marker only changes the failure path — a healthy install is unaffected."""
    calls: list[str] = []

    def auth_ext(app: Any, config: Any) -> None:
        calls.append("auth")

    auth_ext.fail_closed = True  # type: ignore[attr-defined]
    monkeypatch.setattr(extensions, "discover", lambda: iter([("oauth2", auth_ext)]))

    assert extensions.install_all(object(), object(), enabled=["oauth2"]) == ["oauth2"]
    assert calls == ["auth"]


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
    assert "RuntimeError" in warnings[0]
    assert "corrupt dist-info" not in warnings[0]  # detail stays at DEBUG


def test_discover_is_quiet_when_nothing_is_installed(monkeypatch) -> None:
    assert _discover_warnings(monkeypatch, lambda **_: []) == []
