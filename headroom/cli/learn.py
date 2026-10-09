"""CLI commands for Headroom Learn — offline failure learning."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import click

if TYPE_CHECKING:
    from ..learn.base import LearnPlugin

from .main import main


def _projects_at(projects: list[Any], path: Path) -> list[Any]:
    """Projects checked out at ``path``, including linked worktrees merged into one.

    ``getattr``: external plugins may return duck-typed projects without the field.
    """
    return [
        p for p in projects if p.project_path == path or path in getattr(p, "worktree_paths", ())
    ]


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
Built-in: claude, codex, gemini, grok.
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
    type=click.IntRange(min=1),
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

    from ..learn.analyzer import SessionAnalyzer, _detect_default_model, _require_agy_opt_in
    from ..learn.registry import auto_detect_plugins, get_plugin

    # Flag-combination validation — reject contradictory/no-op combinations up
    # front rather than letting one flag silently win or be ignored.
    if analyze_all and project is not None:
        raise click.UsageError("--all and --project are mutually exclusive.")
    if llm_judge and not verbosity_mode:
        raise click.UsageError("--llm-judge only applies with --verbosity.")

    max_workers = workers if workers is not None else min(os.cpu_count() or 4, 8)

    # Verbosity learning is a distinct flow: it mines behavioral signals (no
    # failure analysis) and needs no LLM unless --llm-judge is set.
    if verbosity_mode:
        ignored = [
            flag
            for flag, is_set in (
                ("--target", target is not None),
                ("--main-only", main_only),
                ("--workers", workers is not None),
                ("--model", model is not None and not llm_judge),
            )
            if is_set
        ]
        if ignored:
            verb = "is" if len(ignored) == 1 else "are"
            click.echo(f"Note: {', '.join(ignored)} {verb} ignored with --verbosity.")
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
        if resolved_model == "agy-cli":
            _require_agy_opt_in()
    except (RuntimeError, ValueError) as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1) from None

    analyzer = SessionAnalyzer(model=resolved_model)

    def _on_progress(detail: str) -> None:
        # Reuses the exact "  Analyzing with ..." prefix so wrapper UIs that
        # whitelist known stage-line prefixes keep parsing without changes.
        click.echo(f"  Analyzing with {resolved_model}... ({detail})")

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
    total_analysis_failures = 0
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
            # An explicitly-selected agent with no data should say so rather than
            # exiting silently (the auto path aggregates across agents instead).
            if agent != "auto":
                click.echo(f"No {plugin.display_name} project data found.")
            continue
        available_projects.extend((agent_name, proj.project_path) for proj in all_projects)

        # Filter to target project(s)
        if analyze_all:
            targets = all_projects
        elif project:
            resolved = project.resolve()
            targets = _projects_at(all_projects, resolved)
            if not targets:
                continue
        else:
            cwd = Path.cwd().resolve()
            targets = _projects_at(all_projects, cwd)
            if not targets:
                for parent in cwd.parents:
                    targets = _projects_at(all_projects, parent)
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
            result_data = analyzer.analyze(proj, sessions, on_progress=_on_progress)
            total_projects += 1
            total_failures += result_data.total_failures

            click.echo(
                f"\n  Sessions: {result_data.total_sessions}  |  "
                f"Calls: {result_data.total_calls}  |  "
                f"Failures: {result_data.total_failures} ({result_data.failure_rate:.1%})"
            )

            analysis_error = getattr(result_data, "analysis_error", None)
            if analysis_error:
                total_analysis_failures += 1
                click.echo(f"  Analysis failed: {analysis_error}", err=True)
                continue

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

    if total_analysis_failures:
        raise SystemExit(1)


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


def _local_proxy_port(port: int | None = None) -> int:
    import os as _os

    return port if port is not None else int(_os.environ.get("HEADROOM_PORT", "8787"))


def _steered_level(raw: object) -> int | None:
    """The level a proxy steers at for a raw ``HEADROOM_VERBOSITY_LEVEL``.

    Mirrors ``OutputShaperSettings.from_env``: unparseable text falls back to
    the default level, anything else is clamped to 0-4. ``None`` when unset.
    """
    from ..proxy.output_shaper import DEFAULT_VERBOSITY_LEVEL

    if raw is None or str(raw).strip() == "":
        return None
    try:
        level = int(str(raw).strip())
    except ValueError:
        level = DEFAULT_VERBOSITY_LEVEL
    return max(0, min(4, level))


class _ProxyProcess(NamedTuple):
    """Process identity and its raw verbosity pin from the same health snapshot."""

    pid: int
    started_at: float
    verbosity_pin: str | None = None


def _proxy_process(payload: dict[str, Any], config: dict[str, Any]) -> _ProxyProcess | None:
    """The proxy process behind a ``/health`` payload, or ``None`` if unknown.

    The start time is ``timestamp - uptime_seconds``: the proxy computes both
    from ``time.time()`` while building the payload, so the difference is its
    start time to the millisecond, the same on every call.
    """
    from datetime import datetime

    pid = config.get("pid")
    timestamp = payload.get("timestamp")
    uptime = payload.get("uptime_seconds")
    if not isinstance(pid, int) or not isinstance(timestamp, str):
        return None
    # A proxy that has not recorded its start reports 0.0; that is not an identity.
    if not isinstance(uptime, int | float) or uptime <= 0:
        return None
    try:
        now = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None
    runtime = config.get("runtime_env")
    raw_pin = runtime.get("HEADROOM_VERBOSITY_LEVEL") if isinstance(runtime, dict) else None
    return _ProxyProcess(
        pid, round(now - float(uptime), 3), raw_pin if isinstance(raw_pin, str) else None
    )


def _query_proxy_verbosity(
    port: int | None = None,
) -> tuple[str | None, int | None, _ProxyProcess | None]:
    """Best-effort: read a running local proxy's mode, pinned level and process.

    All three come from ``/health``, whose config block the proxy includes for
    loopback callers. Returns ``(mode, level, process)``, each ``None`` when the
    proxy is unreachable or does not report it; ``level`` is the level its
    ``HEADROOM_VERBOSITY_LEVEL`` pins, ``None`` when that is unset.
    """
    import json as _json
    import urllib.error
    import urllib.request

    url = f"http://127.0.0.1:{_local_proxy_port(port)}/health"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            payload = _json.loads(response.read() or b"{}")
    except (urllib.error.URLError, OSError, ValueError):
        return None, None, None
    config = payload.get("config") if isinstance(payload, dict) else None
    if not isinstance(config, dict):
        return None, None, None
    mode = config.get("mode")
    runtime = config.get("runtime_env")
    level = runtime.get("HEADROOM_VERBOSITY_LEVEL") if isinstance(runtime, dict) else None
    return (
        mode if isinstance(mode, str) else None,
        _steered_level(level),
        _proxy_process(payload, config),
    )


#: Records the HEADROOM_VERBOSITY_LEVEL that ``--apply`` hot-pinned, and on
#: which proxy process. Hot overrides die with the process, so a pin is ours to
#: replace only while that same process still holds that value.
_VERBOSITY_PIN_FILE = "verbosity_pin.json"

#: Slack when comparing recorded and current start times, which agree to the
#: millisecond for one process; a different process started this close to the
#: old one would also need its pid, port and level to match.
_START_TIME_TOLERANCE_S = 1.0


def _pin_is_ours(ws: Path, port: int, process: _ProxyProcess | None, level: int | None) -> bool:
    import json as _json

    if process is None or level is None:
        return False
    try:
        record = _json.loads((ws / _VERBOSITY_PIN_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(record, dict):
        return False
    started_at = record.get("started_at")
    return (
        record.get("port") == port
        and record.get("pid") == process.pid
        and record.get("level") == level
        and process.verbosity_pin == str(level)
        and isinstance(started_at, int | float)
        and abs(started_at - process.started_at) <= _START_TIME_TOLERANCE_S
    )


def _record_pin(ws: Path, port: int, process: _ProxyProcess | None, level: int) -> bool:
    """Record a pin ``--apply`` set. Returns False when it could not be recorded.

    Best-effort: the pin is already live on the proxy, so a failed record only
    means a later ``--apply`` treats it as the operator's and leaves it alone.
    """
    import json as _json

    path = ws / _VERBOSITY_PIN_FILE
    try:
        if process is None:
            # Without a process identity the pin cannot be attributed later.
            path.unlink(missing_ok=True)
            return False
        record = {
            "port": port,
            "pid": process.pid,
            "started_at": process.started_at,
            "level": level,
        }
        path.write_text(_json.dumps(record), encoding="utf-8", newline="\n")
    except OSError:
        return False
    return True


class _ShaperActivation(NamedTuple):
    status: str
    port: int
    #: The overrides the proxy reports it applied (empty if it did not answer).
    applied: dict[str, str]


def _activate_output_shaper(
    port: int | None = None, *, verbosity_level: str | None = None
) -> _ShaperActivation:
    """Best-effort: turn the output shaper ON for a running local proxy.

    Writing ``verbosity.json`` is inert on its own — the shaper is a live,
    off-by-default knob, so the learned level does nothing until
    ``HEADROOM_OUTPUT_SHAPER`` is enabled in the proxy that serves traffic.
    When a proxy is already running locally we hot-enable it via
    ``/admin/runtime-env`` (no restart, the same channel ``wrap`` uses), so
    ``--apply`` actually takes effect. ``verbosity_level`` also pins
    ``HEADROOM_VERBOSITY_LEVEL`` in the same request. Returns ``(status, port,
    applied)`` where status is ``"live"`` (enabled on a running proxy),
    ``"blocked"`` (the proxy's rollout config disables the shaper; the proxy
    still applies the overrides), ``"absent"`` (no reachable proxy), or
    ``"error"``.
    """
    import json as _json
    import urllib.error
    import urllib.request

    resolved_port = _local_proxy_port(port)
    overrides = {"HEADROOM_OUTPUT_SHAPER": "1"}
    if verbosity_level is not None:
        overrides["HEADROOM_VERBOSITY_LEVEL"] = verbosity_level
    request = urllib.request.Request(
        f"http://127.0.0.1:{resolved_port}/admin/runtime-env",
        data=_json.dumps(overrides).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            raw_response = response.read()
        payload = _json.loads(raw_response) if raw_response else {}
        applied_raw = payload.get("applied") if isinstance(payload, dict) else None
        applied = (
            {k: v for k, v in applied_raw.items() if isinstance(v, str)}
            if isinstance(applied_raw, dict)
            else {}
        )
        rollout = payload.get("rollout") if isinstance(payload, dict) else None
        if isinstance(rollout, dict):
            decisions = rollout.get("features")
            if isinstance(decisions, list):
                output_shaper = next(
                    (
                        item
                        for item in decisions
                        if isinstance(item, dict) and item.get("name") == "proxy_output_shaper"
                    ),
                    None,
                )
                if isinstance(output_shaper, dict) and not output_shaper.get("enabled", False):
                    return _ShaperActivation("blocked", resolved_port, applied)
        return _ShaperActivation("live", resolved_port, applied)
    except (urllib.error.URLError, OSError):
        # ConnectionRefused (no proxy) or 404 (proxy predates the endpoint).
        return _ShaperActivation("absent", resolved_port, {})
    except ValueError:
        return _ShaperActivation("error", resolved_port, {})


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
    from ..proxy.output_savings import BaselineModel, SavingsLedger

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
        targets = _projects_at(all_projects, resolved)
    else:
        cwd = Path.cwd().resolve()
        targets = _projects_at(all_projects, cwd)
        if not targets:
            for parent in cwd.parents:
                targets = _projects_at(all_projects, parent)
                if targets:
                    break
    if not targets:
        click.echo("No matching project. Try --all or --project <path>.")
        return

    judge = _make_llm_judge(model or "claude-sonnet-4-6") if llm_judge else None

    # Aggregate across all targeted projects. The baseline accumulates so the
    # synthetic control reflects every project's transcripts (not just whichever
    # one happens to be processed last). The applied verbosity level comes from
    # the project with the most samples — the strongest, least noisy signal.
    aggregated = BaselineModel()
    best_profile = None
    best_profile_samples = -1
    analyzed_count = 0

    for proj in targets:
        session_paths = sorted(
            f for d in (proj.data_path, *proj.extra_data_paths) for f in d.glob("*.jsonl")
        )
        if not session_paths:
            continue
        profile, baseline = analyze(session_paths, str(proj.project_path), llm_judge=judge)
        sig = profile.signals
        analyzed_count += 1
        aggregated.merge(baseline)
        if baseline.total_samples > best_profile_samples:
            best_profile_samples = baseline.total_samples
            best_profile = profile

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

    if analyzed_count == 0 or best_profile is None:
        click.echo("\n  No transcripts found in the selected project(s); nothing learned.")
        return

    if apply:
        ws = ensure_workspace_dir()
        from datetime import datetime, timezone

        best_profile.learned_at = datetime.now(timezone.utc).isoformat()
        best_profile.save(ws / "verbosity.json")
        # Seed the savings baseline: replace baseline, preserve any live
        # treatment/control already accumulated.
        ledger_path = ws / "output_savings.json"
        ledger = SavingsLedger.load(ledger_path)
        ledger.baseline = aggregated
        ledger.save(ledger_path)
        click.echo(f"\n  [WROTE] {ws / 'verbosity.json'} (level {best_profile.level})")
        click.echo(
            f"  [WROTE] {ledger_path} (baseline: {aggregated.total_samples} samples, "
            f"{len(aggregated.strata)} strata across {analyzed_count} project(s))"
        )
        # Writing the level is not enough — the shaper is off by default.
        # Make --apply actually take effect: hot-enable a running proxy, and
        # otherwise tell the user exactly how to turn it on.
        level = best_profile.level
        port = _local_proxy_port()
        mode, pinned, process = _query_proxy_verbosity(port)
        # Cache mode never reads verbosity.json: a level that can change while
        # conversations are open would bust their prefix cache, so the proxy
        # steers at its startup level unless HEADROOM_VERBOSITY_LEVEL pins one
        # (resolve_verbosity_level). Pin the learned level there; otherwise the
        # proxy would keep steering at its default while we report this level.
        # An existing pin is the operator's and is left alone, unless the record
        # from an earlier --apply shows this same proxy process holds the value
        # that run pinned.
        ours = _pin_is_ours(ws, port, process, pinned)
        pin = str(level) if mode == "cache" and (pinned is None or ours) else None
        status, shaper_port, applied = _activate_output_shaper(port, verbosity_level=pin)
        # Record the pin whenever the proxy applied it: it stores overrides even
        # when its rollout config disables the shaper ("blocked").
        pin_applied = pin is not None and applied.get("HEADROOM_VERBOSITY_LEVEL") == pin
        pin_recorded = pin_applied and _record_pin(ws, shaper_port, process, level)
        cache_mode_hint = (
            f"in cache mode (the default) also set HEADROOM_VERBOSITY_LEVEL={level}, "
            "because cache mode does not read the learned level"
        )
        if status == "live" and pinned is not None and pin is None and pinned != level:
            click.echo(
                f"\n  ⚠ Output shaper enabled on the running proxy (port {shaper_port}), "
                f"but it stays at level {pinned}: HEADROOM_VERBOSITY_LEVEL is set there "
                "and outranks the learned level."
            )
            click.echo(
                f"    To use level {level}, restart the proxy with HEADROOM_VERBOSITY_LEVEL={level}."
            )
        elif status == "live" and mode is None:
            click.echo(
                f"\n  ⚠ Output shaper enabled on the running proxy (port {shaper_port}), but "
                f"its mode could not be read, so level {level} may not be in use."
            )
            click.echo(
                f"    A proxy in cache mode (the default) keeps its startup level; restart it "
                f"with HEADROOM_OUTPUT_SHAPER=1 HEADROOM_VERBOSITY_LEVEL={level} to be sure."
            )
        elif status == "live":
            how = (
                " (pinned with HEADROOM_VERBOSITY_LEVEL: cache mode does not read the "
                "learned level)"
                if pin
                else ""
            )
            click.echo(
                f"\n  ✓ Output shaper enabled on the running proxy (port {shaper_port}); "
                f"level {level} is live now{how}."
            )
            if pin and pinned != level:
                click.echo(
                    "    Conversations already open re-cache their prompt once, because "
                    "the steering text changed."
                )
            restart_env = "HEADROOM_OUTPUT_SHAPER=1" + (
                f" HEADROOM_VERBOSITY_LEVEL={level}" if mode == "cache" else ""
            )
            click.echo(
                f"    To keep it on across restarts: export {restart_env} "
                "before `headroom wrap ...` or `headroom proxy`."
            )
        elif status == "blocked":
            # proxy_output_shaper is available on every channel, so only an
            # explicit disable (HEADROOM_DISABLE_FEATURES) refuses the hot enable.
            click.echo(
                "\n  ⚠ Level written, but the running proxy's rollout config disables the "
                "output shaper."
            )
            click.echo(
                "    Restart it without proxy_output_shaper in HEADROOM_DISABLE_FEATURES "
                f"and with HEADROOM_OUTPUT_SHAPER=1; {cache_mode_hint}."
            )
        else:
            click.echo(
                "\n  ⚠ Level written, but the output shaper is OFF by default — it is "
                "NOT shaping output yet."
            )
            click.echo(
                "    Enable it: export HEADROOM_OUTPUT_SHAPER=1, then run `headroom wrap ...` "
                "(or restart `headroom proxy`). A token-mode proxy then uses the learned "
                f"level; {cache_mode_hint}."
            )
        if pin_applied and not pin_recorded:
            click.echo(
                f"    Note: the HEADROOM_VERBOSITY_LEVEL={level} pin could not be recorded in "
                f"{ws / _VERBOSITY_PIN_FILE}, so a later --apply will leave it in place as "
                "if you had set it."
            )
    else:
        click.echo("\n  Dry run — use --apply to persist the level and baseline.")
