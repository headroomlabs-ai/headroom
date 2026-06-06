from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from headroom.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _write_proxy_log(workspace: Path, lines: list[str]) -> None:
    log_dir = workspace / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "proxy.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_support_bundle_default_includes_diagnostics_not_full_logs(
    monkeypatch: pytest.MonkeyPatch,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "headroom"
    workspace.mkdir()
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(workspace))
    monkeypatch.setenv("HEADROOM_PORT", "8787")
    (workspace / "proxy_savings.json").write_text('{"tokens_saved": 123}\n', encoding="utf-8")
    _write_proxy_log(
        workspace,
        [
            "2026-03-07 13:38:31,009 - headroom.proxy - INFO - normal request line sk-secret123456",
            (
                "2026-03-07 13:38:32,009 - headroom.proxy - INFO - [hr_1] PERF "
                "model=gpt-4o msgs=2 tok_before=100 tok_after=70 tok_saved=30 "
                "cache_read=0 cache_write=0 cache_hit_pct=0 opt_ms=12 api_key=sk-secret123456 "
                "transforms=content_router"
            ),
            "2026-03-07 13:38:33,009 - headroom.proxy - INFO - Pipeline complete: 100 -> 70 tokens (saved 30, 30.0% reduction)",
        ],
    )
    output = tmp_path / "bundle.zip"

    result = runner.invoke(
        main, ["support", "bundle", "--output", str(output), "--max-lines", "10"]
    )

    assert result.exit_code == 0, result.output
    assert "Created support bundle" in result.output
    with zipfile.ZipFile(output) as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert "perf-report.txt" in names
        assert "state/proxy_savings.json" in names
        assert "logs/proxy-diagnostics-tail.txt" in names
        assert "logs/full-log-tail-not-included.txt" in names
        assert "logs/proxy-full-tail.redacted.txt" not in names

        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["selected_environment"]["HEADROOM_PORT"] == "8787"
        assert manifest["paths"]["savings"]["exists"] is True

        diagnostics = zf.read("logs/proxy-diagnostics-tail.txt").decode()
        assert "PERF" in diagnostics
        assert "Pipeline complete" in diagnostics
        assert "normal request line" not in diagnostics
        assert "sk-secret123456" not in diagnostics
        assert "<redacted>" in diagnostics


def test_support_bundle_full_log_tail_is_opt_in_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "headroom"
    workspace.mkdir()
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(workspace))
    _write_proxy_log(
        workspace,
        [
            "2026-03-07 13:38:31,009 - headroom.proxy - INFO - Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
            "2026-03-07 13:38:32,009 - headroom.proxy - INFO - arbitrary non-diagnostic line",
        ],
    )
    output = tmp_path / "bundle.zip"

    result = runner.invoke(
        main,
        [
            "support",
            "bundle",
            "--output",
            str(output),
            "--include-full-log-tail",
            "--max-lines",
            "10",
        ],
    )

    assert result.exit_code == 0, result.output
    with zipfile.ZipFile(output) as zf:
        names = set(zf.namelist())
        assert "logs/proxy-full-tail.redacted.txt" in names
        assert "logs/full-log-tail-not-included.txt" not in names

        full_tail = zf.read("logs/proxy-full-tail.redacted.txt").decode()
        assert "arbitrary non-diagnostic line" in full_tail
        assert "abcdefghijklmnopqrstuvwxyz" not in full_tail
        assert "Authorization: Bearer <redacted>" in full_tail
