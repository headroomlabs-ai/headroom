from __future__ import annotations

import json
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

    def analyze(self, project, sessions, on_progress=None):  # noqa: ANN001, ANN201
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


@pytest.mark.parametrize(
    ("args", "env"),
    [
        (["learn"], {"HEADROOM_LEARN_CLI": "agy"}),
        (["learn", "--model", "agy-cli"], {}),
    ],
)
def test_learn_agy_without_unsafe_opt_in_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, args: list[str], env: dict[str, str]
) -> None:
    for var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "HEADROOM_LEARN_CLI",
        "HEADROOM_LEARN_ALLOW_UNSAFE_AGY",
    ):
        monkeypatch.delenv(var, raising=False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    monkeypatch.setattr(
        "headroom.learn.registry.auto_detect_plugins",
        lambda: pytest.fail("sessions must not be scanned without the agy opt-in"),
    )

    result = runner.invoke(main, args, catch_exceptions=False)

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "HEADROOM_LEARN_ALLOW_UNSAFE_AGY=1" in result.output


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


@pytest.mark.parametrize("from_cwd", [False, True])
def test_learn_selects_the_project_a_worktree_was_merged_into(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path, from_cwd: bool
) -> None:
    main_checkout, worktree = tmp_path / "repo", tmp_path / "workspaces" / "ws"
    (worktree / "src").mkdir(parents=True)
    worktree = worktree.resolve()
    merged = SimpleNamespace(
        name="repo", project_path=main_checkout, worktree_paths=[worktree], extra_data_paths=[]
    )
    plugin = FakePlugin("claude", "Claude Code", [merged])
    analyzer = FakeAnalyzer()

    monkeypatch.setattr("headroom.learn.analyzer._detect_default_model", lambda: "gpt-4o")
    monkeypatch.setattr("headroom.learn.registry.get_plugin", lambda name: plugin)
    monkeypatch.setattr("headroom.learn.analyzer.SessionAnalyzer", lambda model=None: analyzer)

    args = ["learn", "--agent", "claude"]
    if from_cwd:
        monkeypatch.chdir(worktree / "src")
    else:
        args += ["--project", str(worktree)]
    result = runner.invoke(main, args, catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert [call[0] for call in plugin.scan_calls] == [merged]


class ProgressEchoingAnalyzer(FakeAnalyzer):
    def analyze(self, project, sessions, on_progress=None):  # noqa: ANN001, ANN201
        self.calls.append((project, sessions))
        if on_progress is not None:
            on_progress("session started")
            on_progress("assistant responding, 5s")
        return SimpleNamespace(
            total_sessions=len(sessions),
            total_calls=3,
            total_failures=1,
            failure_rate=1 / 3,
            recommendations=[SimpleNamespace(section="Rules")],
        )


def test_learn_analyzing_line_gets_progress_detail_appended(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    project_path = tmp_path / "project-a"
    project_path.mkdir()
    matched = SimpleNamespace(name="project-a", project_path=project_path)
    plugin = FakePlugin("codex", "Codex", [matched])
    analyzer = ProgressEchoingAnalyzer()

    monkeypatch.setattr("headroom.learn.analyzer._detect_default_model", lambda: "gpt-4o")
    monkeypatch.setattr("headroom.learn.registry.get_plugin", lambda name: plugin)
    monkeypatch.setattr("headroom.learn.analyzer.SessionAnalyzer", lambda model=None: analyzer)

    result = runner.invoke(
        main,
        ["learn", "--agent", "codex", "--project", str(project_path)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert "  Analyzing with gpt-4o... (session started)" in result.output
    assert "  Analyzing with gpt-4o... (assistant responding, 5s)" in result.output
    # Final result reporting still appears unmodified after the progress lines.
    assert "Recommendations: 1" in result.output


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
    proj_a = SimpleNamespace(
        name="a", project_path=tmp_path / "src-a", data_path=proj_a_dir, extra_data_paths=[]
    )
    proj_b = SimpleNamespace(
        name="b", project_path=tmp_path / "src-b", data_path=proj_b_dir, extra_data_paths=[]
    )
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
    # --apply talks to a local proxy; keep it off whatever is listening on 8787.
    monkeypatch.setattr("urllib.request.urlopen", _no_local_proxy)

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
    # The shaper is available on every channel; the hint must not ask for beta.
    assert "HEADROOM_OUTPUT_SHAPER=1" in result.output
    assert "HEADROOM_ROLLOUT_CHANNEL" not in result.output
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
        def analyze(self, project, sessions, on_progress=None):  # noqa: ANN001, ANN201
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


def test_learn_surfaces_analysis_failure_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    project = SimpleNamespace(name="broken", project_path=tmp_path / "broken")
    plugin = FakePlugin("codex", "Codex", [project])

    class FailingAnalyzer(FakeAnalyzer):
        def analyze(self, project, sessions, *, on_progress=None):  # noqa: ANN001, ANN201
            self.calls.append((project, sessions))
            return SimpleNamespace(
                total_sessions=1,
                total_calls=3,
                total_failures=1,
                failure_rate=1 / 3,
                recommendations=[],
                analysis_error="codex CLI failed (exit 1): Not inside a trusted directory",
            )

    monkeypatch.setattr("headroom.learn.analyzer._detect_default_model", lambda: "codex-cli")
    monkeypatch.setattr("headroom.learn.registry.get_plugin", lambda name: plugin)
    monkeypatch.setattr("headroom.learn.analyzer.SessionAnalyzer", FailingAnalyzer)

    result = runner.invoke(main, ["learn", "--agent", "codex", "--all"])

    assert result.exit_code == 1
    assert "Analysis failed: codex CLI failed (exit 1)" in result.output
    assert "No actionable patterns found." not in result.output


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


@pytest.mark.parametrize(("enabled", "expected"), [(True, "live"), (False, "blocked")])
def test_activate_output_shaper_reports_effective_rollout_decision(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, expected: str
) -> None:
    import urllib.request

    from headroom.cli.learn import _activate_output_shaper

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "rollout": {
                        "features": [
                            {"name": "proxy_output_shaper", "enabled": enabled},
                        ]
                    }
                }
            ).encode()

    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: Response())

    status, port, _ = _activate_output_shaper(9876)

    assert status == expected
    assert port == 9876


def test_activate_output_shaper_handles_malformed_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import urllib.request

    from headroom.cli.learn import _activate_output_shaper

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self) -> bytes:
            return b"not-json"

    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: Response())

    assert _activate_output_shaper(9876) == ("error", 9876, {})


def _no_local_proxy(*args, **kwargs):  # noqa: ANN002, ANN003, ANN201
    import urllib.error

    raise urllib.error.URLError("no proxy in tests")


# Where the stubbed local proxy listens; HEADROOM_PORT points the CLI at it.
_PROXY_PORT = 18787
# The fake proxy's start time and the wall clock it answers /health at.
_STARTED_AT = 1_791_400_000.0
_HEALTH_NOW = "2026-10-08T10:00:00Z"


class _FakeProxy:
    """A local proxy as ``urllib.request.urlopen`` sees it.

    ``health_config`` is the ``/health`` config block (``None`` = unreachable);
    POSTs to ``/admin/runtime-env`` are recorded and accepted.
    """

    def __init__(
        self,
        health_config: dict | None,
        *,
        shaper_allowed: bool = True,
        started_at: float = _STARTED_AT,
    ) -> None:
        self.health_config = health_config
        self.shaper_allowed = shaper_allowed
        self.started_at = started_at
        self.posted: list[dict] = []

    def urlopen(self, request, timeout=None):  # noqa: ANN001, ANN201
        import io
        import urllib.error

        url = request if isinstance(request, str) else request.full_url
        if url.endswith("/health"):
            if self.health_config is None:
                raise urllib.error.URLError("unreachable")
            from datetime import datetime

            now = datetime.fromisoformat(_HEALTH_NOW.replace("Z", "+00:00")).timestamp()
            payload = {
                "timestamp": _HEALTH_NOW,
                "uptime_seconds": round(now - self.started_at, 3),
                "config": self.health_config,
            }
            return io.BytesIO(json.dumps(payload).encode())
        self.posted.append(json.loads(request.data))
        rollout = {"features": [{"name": "proxy_output_shaper", "enabled": self.shaper_allowed}]}
        return io.BytesIO(json.dumps({"applied": self.posted[-1], "rollout": rollout}).encode())


def _apply_learned_level(
    monkeypatch: pytest.MonkeyPatch,
    runner: CliRunner,
    tmp_path: Path,
    proxy: _FakeProxy,
    *,
    level: int = 3,
    pin_record: dict | list | None = None,
    previous_level: int | None = None,
) -> str:
    """Run ``learn --verbosity --apply`` for one project that learns ``level``.

    ``pin_record`` seeds the record an earlier ``--apply`` leaves after pinning,
    ``previous_level`` the verbosity.json it saved. The proxy listens on
    ``_PROXY_PORT``.
    """
    from headroom.proxy.output_savings import BaselineModel

    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    if pin_record is not None:
        (workspace / "verbosity_pin.json").write_text(json.dumps(pin_record), newline="\n")
    if previous_level is not None:
        (workspace / "verbosity.json").write_text(
            json.dumps({"verbosity_level": previous_level}), newline="\n"
        )
    data_dir = tmp_path / "sessions"
    data_dir.mkdir()
    (data_dir / "s.jsonl").write_text("{}")
    project = SimpleNamespace(
        name="p", project_path=tmp_path / "src", data_path=data_dir, extra_data_paths=[]
    )
    baseline = BaselineModel()
    baseline.observe("opus|new_user_ask|s|tools", 100)
    profile = SimpleNamespace(
        level=level,
        confidence="high",
        source="heuristic",
        rationale="test",
        signals={},
        learned_at=None,
        save=lambda path: Path(str(path)).write_text(
            json.dumps({"verbosity_level": level}), newline="\n"
        ),
    )
    monkeypatch.setattr(
        "headroom.learn.registry.get_plugin", lambda name: FakePlugin(name, "Claude", [project])
    )
    monkeypatch.setattr(
        "headroom.learn.verbosity.analyze",
        lambda session_paths, project_path, llm_judge=None: (profile, baseline),
    )
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(workspace))
    monkeypatch.setenv("HEADROOM_PORT", str(_PROXY_PORT))
    monkeypatch.setattr("urllib.request.urlopen", proxy.urlopen)

    result = runner.invoke(
        main,
        ["learn", "--agent", "claude", "--verbosity", "--all", "--apply"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    return result.output


def test_verbosity_apply_pins_the_level_on_a_cache_mode_proxy(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    """Cache mode never reads verbosity.json, so "live" needs the level pinned."""
    proxy = _FakeProxy({"mode": "cache", "runtime_env": {"HEADROOM_VERBOSITY_LEVEL": None}})

    output = _apply_learned_level(monkeypatch, runner, tmp_path, proxy)

    assert proxy.posted == [{"HEADROOM_OUTPUT_SHAPER": "1", "HEADROOM_VERBOSITY_LEVEL": "3"}]
    assert "level 3 is live now (pinned with HEADROOM_VERBOSITY_LEVEL" in output
    assert "HEADROOM_OUTPUT_SHAPER=1 HEADROOM_VERBOSITY_LEVEL=3 before" in output
    assert "re-cache their prompt once" in output


def test_verbosity_apply_leaves_a_token_mode_proxy_reading_the_profile(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    """Token mode reads verbosity.json per request; a pin would only shadow it."""
    proxy = _FakeProxy({"mode": "token", "runtime_env": {"HEADROOM_VERBOSITY_LEVEL": None}})

    output = _apply_learned_level(monkeypatch, runner, tmp_path, proxy)

    assert proxy.posted == [{"HEADROOM_OUTPUT_SHAPER": "1"}]
    assert "level 3 is live now." in output
    assert "export HEADROOM_OUTPUT_SHAPER=1 before" in output


def test_verbosity_apply_does_not_override_an_explicit_level(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    proxy = _FakeProxy({"mode": "cache", "runtime_env": {"HEADROOM_VERBOSITY_LEVEL": "2"}})

    output = _apply_learned_level(monkeypatch, runner, tmp_path, proxy)

    assert proxy.posted == [{"HEADROOM_OUTPUT_SHAPER": "1"}]
    assert "is live now" not in output
    assert "it stays at level 2: HEADROOM_VERBOSITY_LEVEL is set there" in output


def test_verbosity_apply_records_the_pin_it_sets(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    proxy = _FakeProxy({"mode": "cache", "runtime_env": {}, "pid": 4242})

    _apply_learned_level(monkeypatch, runner, tmp_path, proxy)

    record = json.loads((tmp_path / "ws" / "verbosity_pin.json").read_text())
    assert record == {"port": _PROXY_PORT, "pid": 4242, "started_at": _STARTED_AT, "level": 3}


def test_verbosity_apply_replaces_the_pin_an_earlier_apply_set(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    """A re-learned level must not be shadowed by the previous run's own pin."""
    proxy = _FakeProxy(
        {"mode": "cache", "runtime_env": {"HEADROOM_VERBOSITY_LEVEL": "3"}, "pid": 4242}
    )

    output = _apply_learned_level(
        monkeypatch,
        runner,
        tmp_path,
        proxy,
        level=1,
        pin_record={"port": _PROXY_PORT, "pid": 4242, "started_at": _STARTED_AT, "level": 3},
    )

    assert proxy.posted == [{"HEADROOM_OUTPUT_SHAPER": "1", "HEADROOM_VERBOSITY_LEVEL": "1"}]
    assert "level 1 is live now (pinned with HEADROOM_VERBOSITY_LEVEL" in output
    record = json.loads((tmp_path / "ws" / "verbosity_pin.json").read_text())
    assert record == {"port": _PROXY_PORT, "pid": 4242, "started_at": _STARTED_AT, "level": 1}


def test_verbosity_apply_re_pinning_the_same_level_does_not_warn_about_re_caching(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    proxy = _FakeProxy(
        {"mode": "cache", "runtime_env": {"HEADROOM_VERBOSITY_LEVEL": "3"}, "pid": 4242}
    )

    output = _apply_learned_level(
        monkeypatch,
        runner,
        tmp_path,
        proxy,
        pin_record={"port": _PROXY_PORT, "pid": 4242, "started_at": _STARTED_AT, "level": 3},
    )

    assert "level 3 is live now" in output
    assert "re-cache" not in output


@pytest.mark.parametrize(
    "pin_record",
    [
        None,  # the operator pinned the level that was learned before
        # pinned on a proxy since restarted
        {"port": _PROXY_PORT, "pid": 1111, "started_at": _STARTED_AT, "level": 3},
        # a new process that reused the old one's pid on the same port
        {"port": _PROXY_PORT, "pid": 4242, "started_at": _STARTED_AT - 3600, "level": 3},
        # the operator re-pinned after us
        {"port": _PROXY_PORT, "pid": 4242, "started_at": _STARTED_AT, "level": 2},
        # a record from before start times were recorded
        {"port": _PROXY_PORT, "pid": 4242, "level": 3},
        # a record that is valid JSON but not an object
        [_PROXY_PORT, 4242, _STARTED_AT, 3],
    ],
    ids=[
        "no-record",
        "restarted-proxy",
        "reused-pid",
        "changed-since",
        "no-start-time",
        "not-an-object",
    ],
)
def test_verbosity_apply_keeps_a_pin_it_cannot_prove_it_set(
    monkeypatch: pytest.MonkeyPatch,
    runner: CliRunner,
    tmp_path: Path,
    pin_record: dict | list | None,
) -> None:
    proxy = _FakeProxy(
        {"mode": "cache", "runtime_env": {"HEADROOM_VERBOSITY_LEVEL": "3"}, "pid": 4242}
    )

    output = _apply_learned_level(
        monkeypatch, runner, tmp_path, proxy, level=1, pin_record=pin_record, previous_level=3
    )

    assert proxy.posted == [{"HEADROOM_OUTPUT_SHAPER": "1"}]
    assert "it stays at level 3: HEADROOM_VERBOSITY_LEVEL is set there" in output


@pytest.mark.windows_newline
def test_verbosity_pin_record_is_written_with_lf(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    from headroom.cli import learn as learn_cli

    calls: list[dict] = []
    real_write_text = Path.write_text

    def write_text(self, data, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        calls.append(kwargs)
        return real_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write_text)
    process = learn_cli._ProxyProcess(4242, _STARTED_AT)
    assert learn_cli._record_pin(tmp_path, _PROXY_PORT, process, 3) is True

    assert [call.get("newline") for call in calls] == ["\n"]


@pytest.mark.parametrize("raw", ["9", " 04 "])
def test_verbosity_apply_reads_a_pin_the_way_the_proxy_clamps_it(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path, raw: str
) -> None:
    """The proxy steers a pin of 9 (or " 04 ") at level 4, so it matches a learned 4."""
    proxy = _FakeProxy({"mode": "cache", "runtime_env": {"HEADROOM_VERBOSITY_LEVEL": raw}})

    output = _apply_learned_level(monkeypatch, runner, tmp_path, proxy, level=4)

    assert proxy.posted == [{"HEADROOM_OUTPUT_SHAPER": "1"}]
    assert "level 4 is live now." in output
    assert "stays at level" not in output


def test_verbosity_apply_reads_an_unparseable_pin_as_the_default_level(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    """The proxy steers a non-numeric pin at the default level, and so must the CLI."""
    from headroom.proxy.output_shaper import DEFAULT_VERBOSITY_LEVEL

    proxy = _FakeProxy({"mode": "cache", "runtime_env": {"HEADROOM_VERBOSITY_LEVEL": "high"}})

    output = _apply_learned_level(monkeypatch, runner, tmp_path, proxy, level=3)

    assert proxy.posted == [{"HEADROOM_OUTPUT_SHAPER": "1"}]
    assert f"it stays at level {DEFAULT_VERBOSITY_LEVEL}: HEADROOM_VERBOSITY_LEVEL" in output


def test_verbosity_apply_does_not_claim_live_when_the_mode_is_unknown(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    proxy = _FakeProxy(None)

    output = _apply_learned_level(monkeypatch, runner, tmp_path, proxy)

    assert proxy.posted == [{"HEADROOM_OUTPUT_SHAPER": "1"}]
    assert "is live now" not in output
    assert "its mode could not be read" in output


def test_verbosity_apply_on_a_proxy_that_disables_the_shaper(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    proxy = _FakeProxy({"mode": "cache", "runtime_env": {}}, shaper_allowed=False)

    output = _apply_learned_level(monkeypatch, runner, tmp_path, proxy)

    assert "rollout config disables the output shaper" in output
    assert "HEADROOM_DISABLE_FEATURES" in output
    assert "HEADROOM_VERBOSITY_LEVEL=3" in output
    assert "HEADROOM_ROLLOUT_CHANNEL" not in output


def test_verbosity_apply_records_a_pin_the_proxy_applied_while_the_shaper_is_disabled(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    """The proxy stores the pin even when it reports the shaper blocked; own it."""
    proxy = _FakeProxy({"mode": "cache", "runtime_env": {}, "pid": 4242}, shaper_allowed=False)

    output = _apply_learned_level(monkeypatch, runner, tmp_path, proxy)

    assert proxy.posted == [{"HEADROOM_OUTPUT_SHAPER": "1", "HEADROOM_VERBOSITY_LEVEL": "3"}]
    assert "rollout config disables the output shaper" in output
    record = json.loads((tmp_path / "ws" / "verbosity_pin.json").read_text())
    assert record == {"port": _PROXY_PORT, "pid": 4242, "started_at": _STARTED_AT, "level": 3}


def test_verbosity_apply_reports_activation_when_the_pin_cannot_be_recorded(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    proxy = _FakeProxy({"mode": "cache", "runtime_env": {}, "pid": 4242})
    # A directory where the record file should go makes the write fail.
    (tmp_path / "ws" / "verbosity_pin.json").mkdir(parents=True)

    output = _apply_learned_level(monkeypatch, runner, tmp_path, proxy)

    assert proxy.posted == [{"HEADROOM_OUTPUT_SHAPER": "1", "HEADROOM_VERBOSITY_LEVEL": "3"}]
    assert "level 3 is live now (pinned with HEADROOM_VERBOSITY_LEVEL" in output
    assert "To keep it on across restarts" in output
    assert "pin could not be recorded" in output


def test_verbosity_apply_does_not_record_a_pin_without_a_process_identity(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    """No pid means the pin cannot be attributed later, so leave no record."""
    proxy = _FakeProxy({"mode": "cache", "runtime_env": {}})

    output = _apply_learned_level(monkeypatch, runner, tmp_path, proxy)

    assert proxy.posted == [{"HEADROOM_OUTPUT_SHAPER": "1", "HEADROOM_VERBOSITY_LEVEL": "3"}]
    assert not (tmp_path / "ws" / "verbosity_pin.json").exists()
    assert "pin could not be recorded" in output


def test_verbosity_apply_preserves_a_changed_raw_pin_with_the_same_clamped_level(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    proxy = _FakeProxy(
        {"mode": "cache", "pid": 4242, "runtime_env": {"HEADROOM_VERBOSITY_LEVEL": "8"}}
    )
    output = _apply_learned_level(
        monkeypatch,
        runner,
        tmp_path,
        proxy,
        level=3,
        pin_record={"port": _PROXY_PORT, "pid": 4242, "started_at": _STARTED_AT, "level": 4},
    )
    assert proxy.posted == [{"HEADROOM_OUTPUT_SHAPER": "1"}]
    assert "level 4" in output


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "healthy"},  # no config block (e.g. a non-loopback caller's view)
        {"config": "unexpected"},
    ],
    ids=["no-config", "config-not-an-object"],
)
def test_query_proxy_verbosity_without_a_config_block(
    monkeypatch: pytest.MonkeyPatch, payload: dict
) -> None:
    import io

    from headroom.cli.learn import _query_proxy_verbosity

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: io.BytesIO(json.dumps(payload).encode())
    )

    assert _query_proxy_verbosity(_PROXY_PORT) == (None, None, None)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, (4242, _STARTED_AT, None)),
        ({"uptime_seconds": 0.0}, None),  # the proxy has not recorded its start
        ({"uptime_seconds": None}, None),
        ({"timestamp": "not a time"}, None),
        ({"timestamp": None}, None),
    ],
    ids=["identified", "zero-uptime", "no-uptime", "bad-timestamp", "no-timestamp"],
)
def test_proxy_process_needs_a_start_time(overrides: dict, expected: tuple | None) -> None:
    from datetime import datetime

    from headroom.cli.learn import _proxy_process

    now = datetime.fromisoformat(_HEALTH_NOW.replace("Z", "+00:00")).timestamp()
    payload = {"timestamp": _HEALTH_NOW, "uptime_seconds": now - _STARTED_AT, **overrides}

    assert _proxy_process(payload, {"pid": 4242}) == expected
