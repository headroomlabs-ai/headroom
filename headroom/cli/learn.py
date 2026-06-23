"""CLI commands for Headroom Learn — offline failure learning."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

if TYPE_CHECKING:
    from ..learn.base import LearnPlugin

from .main import main


class _AgentChoice(click.ParamType):
    """Dynamic Click type that validates against the plugin registry."""

    name = "agent"

    def get_metavar(self, param: click.Parameter, ctx: click.Context | None = None) -> str | None:
        return "[auto|<agent>]"

    def convert(
        self,
        value: str,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> str:
        if value == "auto":
            return value
        from ..learn.registry import get_registry

        reg = get_registry()
        if value.lower() not in reg:
            available = ", ".join(sorted(reg.keys()))
            self.fail(f"Unknown agent: {value}. Available: auto, {available}", param, ctx)
        return value.lower()

    def shell_complete(
        self,
        ctx: click.Context,
        param: click.Parameter,
        incomplete: str,
    ) -> list[click.shell_completion.CompletionItem]:
        from ..learn.registry import available_agent_names

        names = ["auto"] + available_agent_names()
        return [click.shell_completion.CompletionItem(n) for n in names if n.startswith(incomplete)]


_AGENT_HELP = """Which coding agent to analyze. Auto-detects by default.

\b
Built-in: claude, codex, gemini.
External plugins register via 'headroom.learn_plugin' entry point.
Use 'auto' (default) to scan all detected agents."""


@main.command()
@click.option(
    "--project",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Project directory to analyze. Defaults to current directory.",
)
@click.option(
    "--all",
    "analyze_all",
    is_flag=True,
    default=False,
    help="Analyze all discovered projects.",
)
@click.option(
    "--apply",
    is_flag=True,
    default=False,
    help="Write recommendations to context/memory files (default: dry-run).",
)
@click.option(
    "--target",
    type=str,
    default=None,
    help="Override the context file learnings are written to (Claude Code only). "
    "Path is relative to the project root, or absolute. Defaults to CLAUDE.local.md "
    "(personal, gitignored). Pass CLAUDE.md to write to the team-shared file instead.",
)
@click.option(
    "--agent",
    type=_AgentChoice(),
    default="auto",
    help=_AGENT_HELP,
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="LLM model for analysis (e.g., claude-sonnet-4-6, gpt-4o, gemini/gemini-flash-latest). "
    "Auto-detected from API keys if not specified.",
)
@click.option(
    "--workers",
    "-j",
    type=int,
    default=None,
    help="Parallel workers for session scanning. "
    "Default: auto (min of CPU count, 8). Use 1 for serial.",
)
@click.option(
    "--main-only",
    is_flag=True,
    default=False,
    help="Only scan top-level main sessions, skipping nested subagent/workflow "
    "transcripts (Claude Code). Default scans everything.",
)
@click.option(
    "--verbosity",
    "verbosity_mode",
    is_flag=True,
    default=False,
    help="Learn the user's preferred OUTPUT verbosity from behavioral signals "
    "(interrupts, fast-skips) instead of analyzing failures. Writes the level "
    "the output shaper applies, and seeds the savings baseline. --apply persists.",
)
@click.option(
    "--llm-judge",
    is_flag=True,
    default=False,
    help="With --verbosity: let an LLM override the heuristic level (needs an API key).",
)
def learn(
    project: Path | None,
    analyze_all: bool,
    apply: bool,
    target: str | None,
    agent: str,
    model: str | None,
    workers: int | None,
    main_only: bool,
    verbosity_mode: bool,
    llm_judge: bool,
) -> None:
    """Learn from past tool call failures to prevent future ones.

    Analyzes conversation history using an LLM to find failure patterns
    (wrong paths, missing modules, stubborn retries) and generates context
    that prevents them from recurring.

    Supports multiple coding agents via a plugin architecture. Built-in
    support for Claude Code, Codex, and Gemini CLI. External plugins can
    be installed via pip (entry point: headroom.learn_plugin).

    \b
    Examples:
        headroom learn                        # Auto-detect agent & model
        headroom learn --apply                # Write recommendations
        headroom learn --model gpt-4o         # Use GPT-4o for analysis
        headroom learn --all                  # Analyze all projects
        headroom learn --agent codex --all    # Analyze all Codex sessions
        headroom learn --target CLAUDE.md     # Write to the team-shared file
    """
    import os

    from ..learn.analyzer import SessionAnalyzer, _detect_default_model
    from ..learn.registry import auto_detect_plugins, get_plugin

    max_workers = workers if workers is not None else min(os.cpu_count() or 4, 8)

    # Verbosity learning is a distinct flow: it mines behavioral signals (no
    # failure analysis) and needs no LLM unless --llm-judge is set.
    if verbosity_mode:
        _run_verbosity(
            project=project,
            analyze_all=analyze_all,
            apply=apply,
            agent=agent,
            llm_judge=llm_judge,
            model=model,
        )
        return

    # Resolve model early to fail fast with a clear message
    try:
        resolved_model = model or _detect_default_model()
    except RuntimeError as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1) from None

    analyzer = SessionAnalyzer(model=resolved_model)

    # Determine which agents to scan
    agent_configs: list[tuple[str, LearnPlugin]] = []

    if agent == "auto":
        detected = auto_detect_plugins()
        if not detected:
            click.echo("No coding agent data found.")
            return
        click.echo(f"Detected agents: {', '.join(p.display_name for p in detected)}")
        agent_configs = [(p.name, p) for p in detected]
    else:
        selected = get_plugin(agent)
        agent_configs = [(selected.name, selected)]

    total_projects = 0
    total_failures = 0
    total_recommendations = 0
    matched_projects = 0
    available_projects: list[tuple[str, Path]] = []

    for agent_name, plugin in agent_configs:
        writer = plugin.create_writer()
        if target is not None:
            if hasattr(writer, "set_context_target"):
                writer.set_context_target(target)
            else:
                click.echo(f"Note: --target is not supported for {agent_name}; ignoring.")
        all_projects = plugin.discover_projects()
        if not all_projects:
            continue
        available_projects.extend((agent_name, proj.project_path) for proj in all_projects)

        # Filter to target project(s)
        if analyze_all:
            targets = all_projects
        elif project:
            resolved = project.resolve()
            targets = [p for p in all_projects if p.project_path == resolved]
            if not targets:
                continue
        else:
            cwd = Path.cwd().resolve()
            targets = [p for p in all_projects if p.project_path == cwd]
            if not targets:
                for parent in cwd.parents:
                    targets = [p for p in all_projects if p.project_path == parent]
                    if targets:
                        break
            if not targets and len(agent_configs) == 1:
                click.echo(f"No {agent_name} project data found for {cwd}")
                click.echo("Try: headroom learn --all  or  headroom learn --project <path>")
                click.echo(f"\nAvailable {agent_name} projects:")
                for proj_info in all_projects[:10]:
                    click.echo(f"  {proj_info.name:30s} {proj_info.project_path}")
                return

        for proj in targets:
            matched_projects += 1
            click.echo(f"\n{'=' * 60}")
            click.echo(f"[{agent_name}] {proj.name}")
            click.echo(f"Path: {proj.project_path}")
            click.echo(f"{'=' * 60}")

            try:
                sessions = plugin.scan_project(
                    proj, max_workers=max_workers, include_subagents=not main_only
                )
            except Exception as exc:
                # One unreadable agent/project must not abort the whole
                # cross-agent run; skip it with a warning and continue.
                click.echo(f"  Skipping (could not scan sessions): {exc}")
                continue
            if not sessions:
                click.echo("  No conversation data found.")
                continue

            click.echo(f"  Analyzing with {resolved_model}...")
            result_data = analyzer.analyze(proj, sessions)
            total_projects += 1
            total_failures += result_data.total_failures

            click.echo(
                f"\n  Sessions: {result_data.total_sessions}  |  "
                f"Calls: {result_data.total_calls}  |  "
                f"Failures: {result_data.total_failures} ({result_data.failure_rate:.1%})"
            )

            if result_data.failure_rate == 0 and not result_data.recommendations:
                click.echo("  No failures or patterns found.")
                continue

            recommendations = result_data.recommendations
            if not recommendations:
                click.echo("  No actionable patterns found.")
                continue

            total_recommendations += len(recommendations)
            click.echo(f"  Recommendations: {len(recommendations)}")

            try:
                result = writer.write(recommendations, proj, dry_run=not apply)
            except OSError as e:
                click.echo(
                    f"  Warning: failed to write recommendations for {proj.project_path}: {e}"
                )
                continue

            for warning in getattr(result, "warnings", None) or []:
                click.echo(f"\n  ⚠ {warning}")

            for file_path, content in result.content_by_file.items():
                click.echo(f"\n  {'[WOULD WRITE]' if result.dry_run else '[WROTE]'} {file_path}")
                click.echo(f"  {'─' * 50}")
                for line in content.split("\n"):
                    if line.startswith("<!-- headroom"):
                        continue
                    click.echo(f"  {line}")
                click.echo(f"  {'─' * 50}")

            if result.dry_run:
                click.echo("\n  Dry run — use --apply to write.")

    if project and matched_projects == 0:
        click.echo(f"No project data found for {project.resolve()}")
        if available_projects:
            click.echo("\nAvailable discovered projects:")
            for agent_name, project_path in available_projects[:10]:
                click.echo(f"  [{agent_name}] {project_path}")
        return

    # Summary
    if total_projects > 1:
        click.echo(f"\n{'=' * 60}")
        click.echo(
            f"Total: {total_projects} projects, {total_failures} failures, "
            f"{total_recommendations} recommendations"
        )


def _make_llm_judge(model: str) -> Any:
    """Build an LLM judge callable for verbosity, or None if unavailable.

    The judge gets the behavioral signals and returns (level, rationale). Kept
    best-effort: any failure (no key, parse error) returns None so the caller
    falls back to the heuristic.
    """

    def judge(signals: dict) -> tuple[int, str] | None:
        try:
            import json

            import litellm
        except ImportError:
            return None
        prompt = (
            "You tune how terse an AI coding assistant should be for one user, "
            "from their behavioral signals. Levels: 1=light (skip ceremony), "
            "2=no ceremony+no echo, 3=conclusions only, 4=caveman/fragments. "
            "Users who interrupt often and reply faster than an answer could be "
            "read (fast-skip) want LESS output.\n\n"
            f"Signals: {json.dumps(signals)}\n\n"
            'Return ONLY JSON: {"level": <1-4>, "rationale": "<one sentence>"}'
        )
        try:
            resp = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            text = resp["choices"][0]["message"]["content"]
            start, end = text.find("{"), text.rfind("}")
            data = json.loads(text[start : end + 1])
            return int(data["level"]), str(data.get("rationale", "LLM judgment"))
        except Exception:
            return None

    return judge


def _run_verbosity(
    *,
    project: Path | None,
    analyze_all: bool,
    apply: bool,
    agent: str,
    llm_judge: bool,
    model: str | None,
) -> None:
    """Learn preferred output verbosity from session transcripts."""
    from ..learn.registry import auto_detect_plugins, get_plugin
    from ..learn.verbosity import analyze
    from ..paths import ensure_workspace_dir
    from ..proxy.output_savings import SavingsLedger

    # Verbosity mining reads Claude Code transcripts; restrict to that plugin.
    if agent == "auto":
        plugins = [p for p in auto_detect_plugins() if p.name == "claude"]
        if not plugins:
            click.echo("Verbosity learning currently supports Claude Code transcripts only.")
            return
        plugin = plugins[0]
    else:
        plugin = get_plugin(agent)
        if plugin.name != "claude":
            click.echo("Verbosity learning currently supports Claude Code transcripts only.")
            return

    all_projects = plugin.discover_projects()
    if not all_projects:
        click.echo("No Claude Code project data found.")
        return

    if analyze_all:
        targets = all_projects
    elif project:
        resolved = project.resolve()
        targets = [p for p in all_projects if p.project_path == resolved]
    else:
        cwd = Path.cwd().resolve()
        targets = [p for p in all_projects if p.project_path == cwd]
        if not targets:
            for parent in cwd.parents:
                targets = [p for p in all_projects if p.project_path == parent]
                if targets:
                    break
    if not targets:
        click.echo("No matching project. Try --all or --project <path>.")
        return

    judge = _make_llm_judge(model or "claude-sonnet-4-6") if llm_judge else None

    for proj in targets:
        session_paths = sorted(proj.data_path.glob("*.jsonl"))
        if not session_paths:
            continue
        profile, baseline = analyze(session_paths, str(proj.project_path), llm_judge=judge)
        sig = profile.signals

        click.echo(f"\n{'=' * 60}")
        click.echo(f"Verbosity — {proj.name}")
        click.echo(f"Path: {proj.project_path}")
        click.echo(f"{'=' * 60}")
        click.echo(
            f"  Sessions: {sig.get('sessions')}  human turns: {sig.get('human_msgs')}  "
            f"responses: {sig.get('asst_responses')}"
        )
        click.echo(
            f"  Interrupts:  {sig.get('interrupts')}  "
            f"({sig.get('interrupt_rate', 0):.0%} of turns)   "
            "← push-back signal"
        )
        click.echo(
            f"  Fast-skips:  {sig.get('fast_skips')} / {sig.get('skip_eligible')} long "
            f"answers ({sig.get('fast_skip_rate', 0):.0%} unread)   ← strongest signal"
        )
        click.echo(f"  Echo ratio:  {sig.get('mean_echo_ratio', 0):.1%} of output restated context")
        click.echo(f"\n  Source: {profile.source}")
        click.echo(f"  {profile.rationale}")
        click.echo(
            f"\n  >> Recommended verbosity level: {profile.level} "
            f"(confidence: {profile.confidence})"
        )

        if apply:
            ws = ensure_workspace_dir()
            from datetime import datetime, timezone

            profile.learned_at = datetime.now(timezone.utc).isoformat()
            profile.save(ws / "verbosity.json")
            # Seed the savings baseline: replace baseline, preserve any live
            # treatment/control already accumulated.
            ledger_path = ws / "output_savings.json"
            ledger = SavingsLedger.load(ledger_path)
            ledger.baseline = baseline
            ledger.save(ledger_path)
            click.echo(f"\n  [WROTE] {ws / 'verbosity.json'} (level {profile.level})")
            click.echo(
                f"  [WROTE] {ledger_path} (baseline: {baseline.total_samples} samples, "
                f"{len(baseline.strata)} strata)"
            )
            click.echo(
                "\n  The output shaper now uses this level when "
                "HEADROOM_OUTPUT_SHAPER=1 and HEADROOM_VERBOSITY_LEVEL is unset."
            )
        else:
            click.echo("\n  Dry run — use --apply to persist the level and baseline.")
