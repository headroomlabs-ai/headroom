from __future__ import annotations

from pathlib import Path

from headroom.mcp_registry.grok import GrokRegistrar
from headroom.mcp_registry.install import build_headroom_spec


def test_grok_registrar_writes_marker_block(tmp_path: Path) -> None:
    registrar = GrokRegistrar(home_dir=tmp_path)
    spec = build_headroom_spec("http://127.0.0.1:9999")

    result = registrar.register_server(spec, force=True)

    assert result.status.value == "registered"
    content = registrar._config_file.read_text(encoding="utf-8")
    assert "[mcp_servers.headroom]" in content
    assert "# --- Headroom MCP server ---" in content


def test_grok_registrar_unregister_removes_marker_block(tmp_path: Path) -> None:
    registrar = GrokRegistrar(home_dir=tmp_path)
    spec = build_headroom_spec()
    registrar.register_server(spec, force=True)

    removed = registrar.unregister_server("headroom")

    assert removed is True
    assert (
        not registrar._config_file.exists() or "Headroom MCP server" not in registrar._read_text()
    )
