from __future__ import annotations

from pathlib import Path

import click
import pytest

from headroom.providers.copilot.vscode import (
    configure_vscode_proxy_settings,
    remove_vscode_proxy_settings,
    vscode_proxy_url,
    vscode_settings_path,
)


@pytest.mark.parametrize(
    ("platform", "env", "expected"),
    [
        (
            "darwin",
            {"HOME": "/Users/a"},
            "/Users/a/Library/Application Support/Code/User/settings.json",
        ),
        (
            "win32",
            {"APPDATA": "C:/Users/A/AppData/Roaming"},
            "C:/Users/A/AppData/Roaming/Code/User/settings.json",
        ),
        ("linux", {"HOME": "/home/a"}, "/home/a/.config/Code/User/settings.json"),
        ("linux", {"HOME": "/home/a", "XDG_CONFIG_HOME": "/cfg"}, "/cfg/Code/User/settings.json"),
    ],
)
def test_vscode_settings_path_is_cross_platform(
    platform: str, env: dict[str, str], expected: str
) -> None:
    assert str(vscode_settings_path(platform=platform, environ=env)).replace("\\", "/") == expected


def test_proxy_url_carries_project_without_changing_model() -> None:
    assert vscode_proxy_url(8787, "my project") == "http://127.0.0.1:8787/p/my%20project"


def test_configure_update_and_remove_preserve_jsonc_verbatim(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    original = '{\n\t// user comment\n\t"editor.fontSize": 15,\n}\n'
    path.write_text(original, encoding="utf-8")

    assert configure_vscode_proxy_settings(path, "http://127.0.0.1:8787") == "added"
    configured = path.read_text(encoding="utf-8")
    assert "editor.fontSize" in configured
    assert "user comment" in configured
    assert '"github.copilot.advanced.debug.overrideProxyUrl"' in configured
    assert '"github.copilot.advanced.debug.overrideAuthType": "token"' in configured

    assert configure_vscode_proxy_settings(path, "http://127.0.0.1:9999") == "updated"
    assert "9999" in path.read_text(encoding="utf-8")
    assert "8787" not in path.read_text(encoding="utf-8")

    assert remove_vscode_proxy_settings(path) is True
    assert path.read_text(encoding="utf-8") == original
    assert remove_vscode_proxy_settings(path) is False


def test_configure_refuses_malformed_and_unmanaged_override(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    for original in (
        "{broken",
        '{"github.copilot.advanced.debug.overrideProxyUrl":"http://other"}',
    ):
        path.write_text(original, encoding="utf-8")
        with pytest.raises(click.ClickException, match="did not overwrite|refusing"):
            configure_vscode_proxy_settings(path, "http://127.0.0.1:8787")
        assert path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("prefix", [b"", b"\xef\xbb\xbf"])
def test_configure_and_remove_preserve_windows_line_endings_and_bom(
    tmp_path: Path, prefix: bytes
) -> None:
    path = tmp_path / "settings.json"
    original = prefix + b'{\r\n\t"editor.fontSize": 15\r\n}\r\n'
    path.write_bytes(original)

    configure_vscode_proxy_settings(path, "http://127.0.0.1:8787")
    assert path.read_bytes().startswith(prefix + b"{\r\n")
    assert remove_vscode_proxy_settings(path) is True
    assert path.read_bytes() == original


def test_configure_refuses_duplicate_managed_markers(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    original = (
        "{\n"
        "// --- Headroom Copilot proxy ---\n"
        "// --- end Headroom Copilot proxy ---\n"
        "// --- Headroom Copilot proxy ---\n"
        "// --- end Headroom Copilot proxy ---\n"
        "}\n"
    )
    path.write_text(original, encoding="utf-8")

    with pytest.raises(click.ClickException, match="marker block"):
        configure_vscode_proxy_settings(path, "http://127.0.0.1:8787")
    assert path.read_text(encoding="utf-8") == original
