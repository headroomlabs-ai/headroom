"""A failure applying saved settings is reported without the settings' values.

Saved settings can hold credentials, and an error raised while applying them can
quote a value, so the warning names the file and the failure's type and code
location, never its message, at any level.
"""

from __future__ import annotations

import logging

import pytest

CREDENTIAL_CANARY = "sk-ant-SETTINGS-CANARY-4d18"


def test_settings_apply_failure_warns_without_credential(monkeypatch, tmp_path) -> None:
    from headroom import settings_store
    from headroom.proxy.server import ProxyConfig, create_app

    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "ws"))
    monkeypatch.delenv("HEADROOM_DEBUG_DUMP", raising=False)

    def _apply_fails(_values: object) -> None:
        raise ValueError(f"cannot apply ANTHROPIC_API_KEY={CREDENTIAL_CANARY}")

    monkeypatch.setattr(settings_store, "apply_to_environ", _apply_fails)

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    proxy_logger = logging.getLogger("headroom.proxy")
    handler = _Capture(level=logging.DEBUG)
    old_level = proxy_logger.level
    proxy_logger.addHandler(handler)
    proxy_logger.setLevel(logging.DEBUG)
    try:
        create_app(ProxyConfig(port=0))
    finally:
        proxy_logger.removeHandler(handler)
        proxy_logger.setLevel(old_level)

    warnings = [r for r in records if "Could not apply saved settings" in r.getMessage()]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING
    assert "ValueError" in warnings[0].getMessage()
    assert "settings.json" in warnings[0].getMessage()
    formatter = logging.Formatter()
    for record in records:
        assert CREDENTIAL_CANARY not in formatter.format(record)


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
