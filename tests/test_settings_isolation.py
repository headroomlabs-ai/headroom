"""Shared test fixtures must keep persisted developer settings out of tests."""

from pathlib import Path

import pytest

from headroom import settings_store


@pytest.mark.parametrize("target_ratio", [0.4, 0.6])
def test_settings_reads_and_writes_are_isolated(monkeypatch, tmp_path, target_ratio):
    developer_home = tmp_path / "developer-home"
    developer_settings = developer_home / ".headroom" / "settings.json"
    developer_settings.parent.mkdir(parents=True)
    original = '{"target_ratio": 0.9}\n'
    developer_settings.write_text(original, encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: developer_home)

    # Each test starts empty, regardless of developer settings or prior saves.
    assert settings_store.load() == {}
    settings_store.save({"target_ratio": target_ratio})
    assert settings_store.load() == {"target_ratio": target_ratio}
    assert developer_settings.read_text(encoding="utf-8") == original
