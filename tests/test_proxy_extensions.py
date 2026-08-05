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
