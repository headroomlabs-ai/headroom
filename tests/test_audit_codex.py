"""Tests for the Codex read-pattern audit (headroom.audit.codex)."""

from __future__ import annotations

import json

import pytest

from headroom.audit.codex import audit_codex, classify_command, render_codex_text, strip_wrappers


class TestClassifier:
    def test_strip_wrappers(self):
        assert strip_wrappers("rtk cat foo.py") == "cat foo.py"
        assert strip_wrappers("rtk proxy sed -n '1,20p' foo.py") == "sed -n '1,20p' foo.py"
        assert strip_wrappers("git status") == "git status"

    @pytest.mark.parametrize(
        ("cmd", "category", "partial"),
        [
            ("cat src/foo.py", "read", False),
            ("sed -n '1,200p' src/foo.py", "read", True),
            ("rtk read src/foo.py --lines 10-50", "read", True),
            ("head -50 src/foo.py", "read", True),
            ("nl headroom/config.py", "read", False),
            ("rg -n 'def apply' headroom/", "search", False),
            ("rtk grep -n pattern .", "search", False),
            ("git diff HEAD~1", "git", False),
            ("apply_patch <<'EOF'\n*** Begin Patch\nEOF", "edit", False),
            ("pytest tests/ -x -q", "build/test", False),
            ("echo hello", "other", False),
        ],
    )
    def test_categories(self, cmd, category, partial):
        cat, _path, is_partial = classify_command(cmd)
        assert cat == category
        if category == "read":
            assert is_partial == partial

    def test_path_extraction_and_workdir(self):
        _, path, _ = classify_command("cat src/foo.py", workdir="/repo")
        assert path == "/repo/src/foo.py"
        _, path, _ = classify_command("cat /abs/foo.py", workdir="/repo")
        assert path == "/abs/foo.py"

    def test_sed_range_not_mistaken_for_path(self):
        _, path, _ = classify_command("sed -n '5,30p' headroom/config.py")
        assert path == "headroom/config.py"

    def test_compound_command_with_read(self):
        cat, path, _ = classify_command("cat foo.py | grep def", workdir="/r")
        assert cat == "read"
        assert path == "/r/foo.py"


def _call(call_id: str, cmd: str, workdir: str = "/repo") -> str:
    return json.dumps(
        {
            "payload": {
                "type": "function_call",
                "name": "exec_command",
                "call_id": call_id,
                "arguments": json.dumps({"cmd": cmd, "workdir": workdir}),
            }
        }
    )


def _output(call_id: str, text: str) -> str:
    return json.dumps(
        {"payload": {"type": "function_call_output", "call_id": call_id, "output": text}}
    )


@pytest.fixture
def codex_dir(tmp_path):
    content = "line\n" * 600  # 3000B — over the maturation floor
    lines = [
        _call("c1", "cat src/foo.py"),
        _output("c1", content),
        _call("c2", "sed -n '1,100p' src/foo.py"),  # partial re-read, same path
        _output("c2", content[:500]),
        _call("c3", "rg -n 'def ' src/"),
        _output("c3", "src/foo.py:1:def x():"),
        _call("c4", "rtk read src/bar.py --lines 1-50"),
        _output("c4", "bar content " * 10),
    ]
    sessions = tmp_path / "sessions" / "2026" / "06"
    sessions.mkdir(parents=True)
    (sessions / "rollout-1.jsonl").write_text("\n".join(lines))
    return tmp_path / "sessions"


class TestAuditCodex:
    def test_metrics(self, codex_dir):
        r = audit_codex(codex_dir)
        assert r.sessions == 1
        assert r.exec_calls == 4
        assert r.read_calls == 3  # c1, c2, c4
        assert r.reads_partial == 2  # c2 (sed range), c4 (--lines)
        assert r.rereads_same_path == 1  # c2 re-reads foo.py
        assert r.distinct_files_read == 2
        assert r.reads_over_floor == 1  # c1 (3000B)
        assert r.calls_by_category["search"] == 1
        assert r.bytes_by_category["read"] > r.bytes_by_category["search"]

    def test_render_runs(self, codex_dir):
        out = render_codex_text(audit_codex(codex_dir))
        assert "codex read-pattern audit" in out
        assert "partial slices" in out

    def test_empty_dir(self, tmp_path):
        assert audit_codex(tmp_path).sessions == 0


class TestCli:
    def test_cli_codex_mode(self, codex_dir):
        from click.testing import CliRunner

        from headroom.cli.main import main

        runner = CliRunner()
        res = runner.invoke(main, ["audit-reads", "--codex", "--path", str(codex_dir)])
        assert res.exit_code == 0, res.output
        assert "codex read-pattern audit" in res.output

        res = runner.invoke(
            main, ["audit-reads", "--codex", "--path", str(codex_dir), "--format", "json"]
        )
        assert res.exit_code == 0
        assert json.loads(res.output)["read_calls"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
