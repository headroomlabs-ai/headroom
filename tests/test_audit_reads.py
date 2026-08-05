"""Tests for the audit-reads traffic audit (headroom.audit.reads)."""

from __future__ import annotations

import json

import pytest

from headroom.audit.reads import audit_reads, render_text

CONTENT = "     1\tdef foo():\n     2\t    return 42\n" * 30  # >512B


def _line(role: str, content, ts: str = "2026-06-09T10:00:00Z") -> str:
    return json.dumps({"message": {"role": role, "content": content}, "timestamp": ts})


def _tool_use(tc_id: str, name: str, inp: dict) -> dict:
    return {"type": "tool_use", "id": tc_id, "name": name, "input": inp}


def _tool_result(tc_id: str, text: str) -> dict:
    return {"type": "tool_result", "tool_use_id": tc_id, "content": text}


@pytest.fixture
def transcript_dir(tmp_path):
    """Synthetic session: read foo.py twice (identical), partial read
    contained in the full read, edit foo.py, then a >5min gap."""
    lines = [
        _line("user", "look at foo.py", "2026-06-09T10:00:00Z"),
        _line(
            "assistant",
            [_tool_use("r1", "Read", {"file_path": "/x/foo.py"})],
            "2026-06-09T10:00:01Z",
        ),
        _line("user", [_tool_result("r1", CONTENT)], "2026-06-09T10:00:02Z"),
        _line(
            "assistant",
            [_tool_use("r2", "Read", {"file_path": "/x/foo.py"})],
            "2026-06-09T10:00:03Z",
        ),
        _line("user", [_tool_result("r2", CONTENT)], "2026-06-09T10:00:04Z"),
        _line(
            "assistant",
            [_tool_use("r3", "Read", {"file_path": "/x/foo.py", "offset": 1, "limit": 2})],
            "2026-06-09T10:00:05Z",
        ),
        # Partial read: a strict substring of the earlier full read.
        _line("user", [_tool_result("r3", CONTENT[: len(CONTENT) // 2])], "2026-06-09T10:00:06Z"),
        _line(
            "assistant",
            [_tool_use("e1", "Edit", {"file_path": "/x/foo.py", "old_string": "a"})],
            "2026-06-09T10:00:07Z",
        ),
        _line("user", [_tool_result("e1", "ok")], "2026-06-09T10:00:08Z"),
        # >5min gap before the next message.
        _line("user", "back from lunch", "2026-06-09T10:20:00Z"),
    ]
    proj = tmp_path / "projects" / "-x-demo"
    proj.mkdir(parents=True)
    (proj / "session1.jsonl").write_text("\n".join(lines))
    return tmp_path / "projects"


class TestAuditReads:
    def test_metrics(self, transcript_dir):
        r = audit_reads(transcript_dir)

        assert r.sessions == 1
        assert r.read_calls == 3
        assert r.dedup_identical_calls == 1  # r2 == r1
        assert r.subset_calls == 1  # r3 ⊂ r1
        # Mechanism rows size each opportunity independently, so a read
        # can appear in more than one: r1 and r3 both precede the edit
        # (stale), and r3 is also a subset of r1. Only identical-repeat
        # excludes from stale (replacing a pointer twice is meaningless).
        assert r.stale_calls == 2
        assert r.gaps_over_5m == 1
        assert r.sessions_with_gap == 1
        assert r.linenum_overhead_bytes > 0
        assert r.class_bytes.get("source code", 0) > 0
        assert r.tool_bytes["Read"] == r.read_bytes
        assert r.reads_per_file_max == 3

    def test_render_text_runs(self, transcript_dir):
        out = render_text(audit_reads(transcript_dir))
        assert "Read opportunity sizing" in out
        assert "identical repeat" in out
        assert "cache-death windows" in out

    def test_json_roundtrip(self, transcript_dir):
        r = audit_reads(transcript_dir)
        data = json.loads(r.to_json())
        assert data["read_calls"] == 3

    def test_malformed_lines_tolerated(self, tmp_path):
        proj = tmp_path / "p"
        proj.mkdir()
        (proj / "bad.jsonl").write_text("not json\n{\n" + _line("user", "hi"))
        r = audit_reads(tmp_path)
        assert r.sessions == 1
        assert r.read_calls == 0

    def test_empty_dir(self, tmp_path):
        r = audit_reads(tmp_path)
        assert r.sessions == 0


class TestMaturationSim:
    def test_metrics(self, transcript_dir):
        from headroom.audit.maturation import simulate_maturation

        r = simulate_maturation(transcript_dir)
        assert r.read_calls == 3
        # r2 and r3 target the already-read foo.py; r3 is partial.
        assert r.rereads_any == 2
        assert r.rereads_partial == 1
        # CONTENT is ~1.2KB — below the 2KB maturation floor — so the
        # big-read metrics stay empty on this fixture.
        assert r.big_reads == 0
        # The edit follows reads of the same file with touch-gap 1.
        assert r.edits_with_prior_read == 1
        assert r.at_risk_edits[1] == 0

    def test_big_read_metrics(self, tmp_path):
        from headroom.audit.maturation import MATURE_FLOOR, simulate_maturation

        big = "x" * (MATURE_FLOOR + 100)
        lines = [
            _line("assistant", [_tool_use("r1", "Read", {"file_path": "/x/big.py"})]),
            _line("user", [_tool_result("r1", big)]),
        ]
        proj = tmp_path / "p"
        proj.mkdir()
        (proj / "s.jsonl").write_text("\n".join(lines))
        r = simulate_maturation(tmp_path)
        assert r.big_reads == 1
        assert r.never_touched_again == 1

    def test_render_runs(self, transcript_dir):
        from headroom.audit.maturation import render_sim_text, simulate_maturation

        out = render_sim_text(simulate_maturation(transcript_dir))
        assert "maturation simulation" in out
        assert "at-risk edits" in out


class TestCli:
    def test_cli_text_and_json(self, transcript_dir):
        from click.testing import CliRunner

        from headroom.cli.main import main

        runner = CliRunner()
        res = runner.invoke(main, ["audit-reads", "--path", str(transcript_dir)])
        assert res.exit_code == 0, res.output
        assert "Read opportunity sizing" in res.output

        res = runner.invoke(
            main, ["audit-reads", "--path", str(transcript_dir), "--format", "json"]
        )
        assert res.exit_code == 0
        assert json.loads(res.output)["sessions"] == 1

    def test_cli_simulate_maturation(self, transcript_dir):
        from click.testing import CliRunner

        from headroom.cli.main import main

        runner = CliRunner()
        res = runner.invoke(
            main, ["audit-reads", "--path", str(transcript_dir), "--simulate-maturation"]
        )
        assert res.exit_code == 0, res.output
        assert "maturation simulation" in res.output

        res = runner.invoke(
            main,
            [
                "audit-reads",
                "--path",
                str(transcript_dir),
                "--simulate-maturation",
                "--format",
                "json",
            ],
        )
        assert res.exit_code == 0
        data = json.loads(res.output)
        assert data["maturation_simulation"]["read_calls"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
