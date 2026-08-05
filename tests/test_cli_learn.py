from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import click
import click.shell_completion as click_shell_completion
import pytest
from click.testing import CliRunner

from headroom.cli.learn import _AgentChoice
from headroom.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class FakeWriter:
    def __init__(self) -> None:
        self.calls: list[tuple[list[object], object, bool]] = []
        self.fail_for: object | None = None

    def write(self, recommendations, project, dry_run: bool):  # noqa: ANN001, ANN201
        self.calls.append((recommendations, project, dry_run))
        if project is self.fail_for:
            raise PermissionError(f"cannot write {project.project_path}")
        return SimpleNamespace(
            dry_run=dry_run,
            content_by_file={
                Path(project.project_path) / "AGENTS.md": "<!-- headroom -->\nRule 1\nRule 2"
            },
        )


class FakePlugin:
    def __init__(self, name: str, display_name: str, projects: list[object]) -> None:
        self.name = name
        self.display_name = display_name
        self._projects = projects
        self.writer = FakeWriter()
        self.scan_calls: list[tuple[object, int]] = []
        self.last_include_subagents: bool | None = None

    def detect(self) -> bool:
        return True

    def create_writer(self) -> FakeWriter:
        return self.writer

    def discover_projects(self) -> list[object]:
        return self._projects

    def scan_project(self, project, max_workers: int = 1, include_subagents: bool = True):  # noqa: ANN001, ANN201
        self.scan_calls.append((project, max_workers))
        self.last_include_subagents = include_subagents
        return [SimpleNamespace(events=["event"], tool_calls=[], failure_count=0)]


class FakeAnalyzer:
    def __init__(self, model: str | None = None) -> None:
        self.model = model
        self.calls: list[tuple[object, list[object]]] = []

    def analyze(self, project, sessions):  # noqa: ANN001, ANN201
        self.calls.append((project, sessions))
        return SimpleNamespace(
            total_sessions=len(sessions),
            total_calls=3,
            total_failures=1,
            failure_rate=1 / 3,
            recommendations=[SimpleNamespace(section="Rules")],
        )


def test_agent_choice_convert_and_shell_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    choice = _AgentChoice()
    monkeypatch.setattr(click, "shell_completion", click_shell_completion)
    monkeypatch.setattr(
        "headroom.learn.registry.get_registry",
        lambda: {"codex": object(), "claude": object()},
    )
    monkeypatch.setattr(
        "headroom.learn.registry.available_agent_names",
        lambda: ["claude", "codex"],
    )

    assert choice.convert("auto", None, None) == "auto"
    assert choice.convert("CODEX", None, None) == "codex"
    with pytest.raises(Exception, match="Unknown agent: bad"):
        choice.convert("bad", None, None)

    completions = choice.shell_complete(None, None, "c")  # type: ignore[arg-type]
    assert [item.value for item in completions] == ["claude", "codex"]
    assert choice.get_metavar(None) == "[auto|<agent>]"  # type: ignore[arg-type]


def test_learn_exits_cleanly_when_model_detection_fails(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner
) -> None:
    monkeypatch.setattr(
        "headroom.learn.analyzer._detect_default_model",
        lambda: (_ for _ in ()).throw(RuntimeError("no model")),
    )

    result = runner.invoke(main, ["learn"], catch_exceptions=False)

    assert result.exit_code == 1
    assert "Error: no model" in result.output


def test_learn_auto_agent_reports_no_detected_plugins(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner
) -> None:
    monkeypatch.setattr("headroom.learn.analyzer._detect_default_model", lambda: "gpt-4o")
    monkeypatch.setattr("headroom.learn.registry.auto_detect_plugins", lambda: [])
    monkeypatch.setattr("headroom.learn.analyzer.SessionAnalyzer", FakeAnalyzer)

    result = runner.invoke(main, ["learn"], catch_exceptions=False)

    assert result.exit_code == 0
    assert "No coding agent data found." in result.output


def test_learn_single_agent_shows_available_projects_when_cwd_missing(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    project = SimpleNamespace(name="demo", project_path=tmp_path / "demo")
    plugin = FakePlugin("codex", "Codex", [project])

    monkeypatch.setattr("headroom.learn.analyzer._detect_default_model", lambda: "gpt-4o")
    monkeypatch.setattr("headroom.learn.registry.get_plugin", lambda name: plugin)
    monkeypatch.setattr("headroom.learn.analyzer.SessionAnalyzer", FakeAnalyzer)

    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(main, ["learn", "--agent", "codex"], catch_exceptions=False)

    assert result.exit_code == 0
    assert "No codex project data found for" in result.output
    assert "Available codex projects:" in result.output
    assert "demo" in result.output


def test_learn_project_lookup_and_apply_flow(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    project_path = tmp_path / "project-a"
    project_path.mkdir()
    matched = SimpleNamespace(name="project-a", project_path=project_path)
    unmatched = SimpleNamespace(name="project-b", project_path=tmp_path / "project-b")
    plugin = FakePlugin("codex", "Codex", [matched, unmatched])
    analyzer = FakeAnalyzer()

    monkeypatch.setattr("headroom.learn.analyzer._detect_default_model", lambda: "gpt-4o")
    monkeypatch.setattr("headroom.learn.registry.get_plugin", lambda name: plugin)
    monkeypatch.setattr("headroom.learn.analyzer.SessionAnalyzer", lambda model=None: analyzer)
    monkeypatch.setattr("os.cpu_count", lambda: 12)

    result = runner.invoke(
        main,
        ["learn", "--agent", "codex", "--project", str(project_path), "--apply", "--workers", "4"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert "Path: " in result.output
    assert "Analyzing with gpt-4o..." in result.output
    assert "Recommendations: 1" in result.output
    assert "[WROTE]" in result.output
    assert "Rule 1" in result.output
    assert plugin.scan_calls == [(matched, 4)]
    assert analyzer.calls[0][0] is matched
    assert plugin.writer.calls[0][2] is False


def test_verbosity_all_apply_aggregates_baselines_across_projects(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    import json as _json

    from headroom.proxy.output_savings import BaselineModel, SavingsLedger

    # Two projects, each with a transcript dir holding a dummy session file
    # (analyze is faked, so contents are irrelevant — only presence matters).
    proj_a_dir = tmp_path / "a"
    proj_b_dir = tmp_path / "b"
    for d in (proj_a_dir, proj_b_dir):
        d.mkdir()
        (d / "s.jsonl").write_text("{}")
    proj_a = SimpleNamespace(name="a", project_path=tmp_path / "src-a", data_path=proj_a_dir)
    proj_b = SimpleNamespace(name="b", project_path=tmp_path / "src-b", data_path=proj_b_dir)
    plugin = FakePlugin("claude", "Claude Code", [proj_a, proj_b])

    # Per-project synthetic baselines. Project A has more samples, so its level
    # must be the one applied.
    base_a = BaselineModel()
    for v in (100, 200, 300):
        base_a.observe("opus|new_user_ask|s|tools", v)
    base_b = BaselineModel()
    base_b.observe("sonnet|unknown|m|notools", 50)

    class _Profile:
        def __init__(self, level: int) -> None:
            self.level = level
            self.confidence = "high"
            self.source = "heuristic"
            self.rationale = "test"
            self.signals: dict[str, object] = {}
            self.learned_at: str | None = None

        def save(self, path: object) -> None:
            Path(str(path)).write_text(_json.dumps({"level": self.level}))

    results = {
        str(proj_a.project_path): (_Profile(1), base_a),
        str(proj_b.project_path): (_Profile(3), base_b),
    }

    def fake_analyze(session_paths, project_path, llm_judge=None):  # noqa: ANN001, ANN201
        return results[project_path]

    monkeypatch.setattr("headroom.learn.registry.get_plugin", lambda name: plugin)
    monkeypatch.setattr("headroom.learn.verbosity.analyze", fake_analyze)
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "ws"))

    result = runner.invoke(
        main,
        ["learn", "--agent", "claude", "--verbosity", "--all", "--apply"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    ledger = SavingsLedger.load(tmp_path / "ws" / "output_savings.json")
    # Aggregated, not last-project-wins: both strata present and totals summed.
    assert ledger.baseline.total_samples == 4
    assert "opus|new_user_ask|s|tools" in ledger.baseline.strata
    assert "sonnet|unknown|m|notools" in ledger.baseline.strata
    assert "across 2 project(s)" in result.output
    # The applied level comes from the project with the most samples (A → 1).
    verbosity = _json.loads((tmp_path / "ws" / "verbosity.json").read_text())
    assert verbosity["level"] == 1


def test_learn_reports_missing_requested_project_and_lists_discovered(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    requested = tmp_path / "missing"
    requested.mkdir()
    discovered = SimpleNamespace(name="project-a", project_path=tmp_path / "project-a")
    plugin = FakePlugin("claude", "Claude Code", [discovered])

    monkeypatch.setattr("headroom.learn.analyzer._detect_default_model", lambda: "gpt-4o")
    monkeypatch.setattr("headroom.learn.registry.get_plugin", lambda name: plugin)
    monkeypatch.setattr("headroom.learn.analyzer.SessionAnalyzer", FakeAnalyzer)

    result = runner.invoke(
        main,
        ["learn", "--agent", "claude", "--project", str(requested)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert f"No project data found for {requested.resolve()}" in result.output
    assert "Available discovered projects:" in result.output
    assert "[claude]" in result.output


def test_learn_analyze_all_uses_default_workers_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    projects_a = [SimpleNamespace(name="a", project_path=tmp_path / "a")]
    projects_b = [SimpleNamespace(name="b", project_path=tmp_path / "b")]
    plugin_a = FakePlugin("codex", "Codex", projects_a)
    plugin_b = FakePlugin("claude", "Claude Code", projects_b)
    analyzer = FakeAnalyzer()

    monkeypatch.setattr("headroom.learn.analyzer._detect_default_model", lambda: "gpt-4o")
    monkeypatch.setattr(
        "headroom.learn.registry.auto_detect_plugins",
        lambda: [plugin_a, plugin_b],
    )
    monkeypatch.setattr("headroom.learn.analyzer.SessionAnalyzer", lambda model=None: analyzer)
    monkeypatch.setattr("os.cpu_count", lambda: 12)

    result = runner.invoke(main, ["learn", "--all"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert "Detected agents: Codex, Claude Code" in result.output
    assert "Total: 2 projects, 2 failures, 2 recommendations" in result.output
    assert plugin_a.scan_calls == [(projects_a[0], 8)]
    assert plugin_b.scan_calls == [(projects_b[0], 8)]


def test_learn_analyze_all_continues_when_one_project_write_fails(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    blocked = SimpleNamespace(name="blocked", project_path=tmp_path / "blocked")
    ok = SimpleNamespace(name="ok", project_path=tmp_path / "ok")
    plugin = FakePlugin("claude", "Claude Code", [blocked, ok])
    plugin.writer.fail_for = blocked
    analyzer = FakeAnalyzer()

    monkeypatch.setattr("headroom.learn.analyzer._detect_default_model", lambda: "gpt-4o")
    monkeypatch.setattr("headroom.learn.registry.get_plugin", lambda name: plugin)
    monkeypatch.setattr("headroom.learn.analyzer.SessionAnalyzer", lambda model=None: analyzer)

    result = runner.invoke(
        main,
        ["learn", "--agent", "claude", "--all", "--apply"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert "Warning: failed to write recommendations" in result.output
    assert str(blocked.project_path) in result.output
    assert "[WROTE]" in result.output
    assert str(ok.project_path / "AGENTS.md") in result.output
    expected_workers = min(os.cpu_count() or 4, 8)
    assert plugin.scan_calls == [(blocked, expected_workers), (ok, expected_workers)]


def test_learn_handles_empty_sessions_and_no_pattern_outputs(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    no_sessions = SimpleNamespace(name="empty", project_path=tmp_path / "empty")
    no_failures = SimpleNamespace(name="clean", project_path=tmp_path / "clean")
    no_actions = SimpleNamespace(name="no-actions", project_path=tmp_path / "no-actions")

    class BranchingPlugin(FakePlugin):
        def scan_project(self, project, max_workers: int = 1, include_subagents: bool = True):  # noqa: ANN001, ANN201
            self.scan_calls.append((project, max_workers))
            if project is no_sessions:
                return []
            return [SimpleNamespace(events=["event"], tool_calls=[], failure_count=0)]

    class BranchingAnalyzer(FakeAnalyzer):
        def analyze(self, project, sessions):  # noqa: ANN001, ANN201
            self.calls.append((project, sessions))
            if project is no_failures:
                return SimpleNamespace(
                    total_sessions=1,
                    total_calls=2,
                    total_failures=0,
                    failure_rate=0.0,
                    recommendations=[],
                )
            return SimpleNamespace(
                total_sessions=1,
                total_calls=2,
                total_failures=1,
                failure_rate=0.5,
                recommendations=[],
            )

    plugin = BranchingPlugin("codex", "Codex", [no_sessions, no_failures, no_actions])
    analyzer = BranchingAnalyzer()

    monkeypatch.setattr("headroom.learn.analyzer._detect_default_model", lambda: "gpt-4o")
    monkeypatch.setattr("headroom.learn.registry.get_plugin", lambda name: plugin)
    monkeypatch.setattr("headroom.learn.analyzer.SessionAnalyzer", lambda model=None: analyzer)

    result = runner.invoke(main, ["learn", "--agent", "codex", "--all"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert "No conversation data found." in result.output
    assert "No failures or patterns found." in result.output
    assert "No actionable patterns found." in result.output


def test_learn_main_only_flag_threads_to_scanner(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    project_path = tmp_path / "proj"
    project_path.mkdir()
    proj = SimpleNamespace(name="proj", project_path=project_path)
    plugin = FakePlugin("codex", "Codex", [proj])

    monkeypatch.setattr("headroom.learn.analyzer._detect_default_model", lambda: "gpt-4o")
    monkeypatch.setattr("headroom.learn.registry.get_plugin", lambda name: plugin)
    monkeypatch.setattr("headroom.learn.analyzer.SessionAnalyzer", FakeAnalyzer)

    # Default: descend into subagent/workflow transcripts.
    result = runner.invoke(main, ["learn", "--agent", "codex", "--all"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert plugin.last_include_subagents is True

    # --main-only restricts to top-level main sessions.
    plugin.last_include_subagents = None
    result = runner.invoke(
        main, ["learn", "--agent", "codex", "--all", "--main-only"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    assert plugin.last_include_subagents is False


class TargetAwareWriter(FakeWriter):
    """A writer that supports --target and surfaces a migration warning."""

    def __init__(self) -> None:
        super().__init__()
        self.context_target: str | None = None

    def set_context_target(self, target: str | None) -> None:
        self.context_target = target

    def write(self, recommendations, project, dry_run: bool):  # noqa: ANN001, ANN201
        self.calls.append((recommendations, project, dry_run))
        return SimpleNamespace(
            dry_run=dry_run,
            content_by_file={
                Path(project.project_path) / "CLAUDE.local.md": "<!-- headroom -->\nRule 1"
            },
            warnings=["Moved Headroom learnings out of CLAUDE.md into CLAUDE.local.md."],
        )


def test_learn_target_threads_to_writer_and_prints_warnings(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    project_path = tmp_path / "proj"
    project_path.mkdir()
    proj = SimpleNamespace(name="proj", project_path=project_path)
    plugin = FakePlugin("claude", "Claude Code", [proj])
    plugin.writer = TargetAwareWriter()

    monkeypatch.setattr("headroom.learn.analyzer._detect_default_model", lambda: "gpt-4o")
    monkeypatch.setattr("headroom.learn.registry.get_plugin", lambda name: plugin)
    monkeypatch.setattr("headroom.learn.analyzer.SessionAnalyzer", FakeAnalyzer)

    result = runner.invoke(
        main,
        [
            "learn",
            "--agent",
            "claude",
            "--project",
            str(project_path),
            "--apply",
            "--target",
            "CLAUDE.md",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    # --target is threaded into the writer...
    assert plugin.writer.context_target == "CLAUDE.md"
    # ...and the writer's warnings are surfaced to the user.
    assert "Moved Headroom learnings" in result.output


def test_learn_target_ignored_for_unsupported_agent(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    project_path = tmp_path / "proj"
    project_path.mkdir()
    proj = SimpleNamespace(name="proj", project_path=project_path)
    # FakePlugin's FakeWriter has no set_context_target, so --target is unsupported.
    plugin = FakePlugin("codex", "Codex", [proj])

    monkeypatch.setattr("headroom.learn.analyzer._detect_default_model", lambda: "gpt-4o")
    monkeypatch.setattr("headroom.learn.registry.get_plugin", lambda name: plugin)
    monkeypatch.setattr("headroom.learn.analyzer.SessionAnalyzer", FakeAnalyzer)

    result = runner.invoke(
        main,
        ["learn", "--agent", "codex", "--project", str(project_path), "--target", "CLAUDE.md"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert "Note: --target is not supported for codex" in result.output
