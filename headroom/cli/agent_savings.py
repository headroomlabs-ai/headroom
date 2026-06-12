"""CLI helpers for agent token-savings profiles."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import click

from headroom.agent_savings import get_agent_savings_profile

from .main import main


@main.command("agent-savings")
@click.option(
    "--profile",
    default="agent-90",
    show_default=True,
    help="Savings profile to render or check.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["shell", "json"]),
    default="shell",
    show_default=True,
    help="Output format for profile environment.",
)
@click.option(
    "--check-perf",
    is_flag=True,
    help="Check recent proxy logs against the profile savings target.",
)
@click.option(
    "--hours",
    type=float,
    default=24.0,
    show_default=True,
    help="Hours of proxy logs to inspect with --check-perf.",
)
@click.option(
    "--accuracy-report",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Headroom eval JSON report proving accuracy preservation.",
)
@click.option(
    "--write-smoke-fixture",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Write deterministic three-agent PERF/eval fixture into workspace dir.",
)
@click.option(
    "--require-agents",
    default="",
    help="Comma-separated clients that must each meet the savings target.",
)
@click.option(
    "--min-accuracy",
    type=float,
    default=0.90,
    show_default=True,
    help="Minimum accepted accuracy preservation rate.",
)
def agent_savings(
    profile: str,
    output_format: str,
    check_perf: bool,
    hours: float,
    accuracy_report: Path | None,
    write_smoke_fixture: Path | None,
    require_agents: str,
    min_accuracy: float,
) -> None:
    """Render or verify Codex/Claude/Cursor token-savings settings."""

    savings_profile = get_agent_savings_profile(profile)
    if write_smoke_fixture is not None:
        eval_path = _write_smoke_fixture(write_smoke_fixture)
        click.echo(f"Wrote agent-90 smoke fixture to {write_smoke_fixture}")
        click.echo(
            "Verify with: HEADROOM_WORKSPACE_DIR="
            f"{write_smoke_fixture} headroom agent-savings --check-perf "
            "--hours 0 --require-agents claude,codex,cursor "
            f"--accuracy-report {eval_path}"
        )
        return

    if check_perf or accuracy_report is not None:
        messages: list[str] = []
        from headroom.perf.analyzer import build_perf_summary, parse_log_files

        if check_perf:
            perf_report = parse_log_files(last_n_hours=hours)
            summary = build_perf_summary(perf_report)
            measured = float(summary.get("savings_pct", 0.0))
            target = savings_profile.target_savings * 100
            if measured < target:
                raise click.ClickException(
                    f"{measured:.1f}% savings below {target:.1f}% target for {savings_profile.name}"
                )
            messages.append(
                f"{measured:.1f}% savings meets {target:.1f}% target for {savings_profile.name}"
            )
            required = _split_required_agents(require_agents)
            if required:
                messages.extend(
                    _check_required_agents(
                        perf_report.perf_records,
                        required,
                        target,
                    )
                )

        if accuracy_report is not None:
            accuracy = _read_accuracy_rate(accuracy_report)
            if accuracy < min_accuracy:
                raise click.ClickException(
                    f"{accuracy * 100:.1f}% accuracy below {min_accuracy * 100:.1f}% target"
                )
            messages.append(
                f"{accuracy * 100:.1f}% accuracy meets {min_accuracy * 100:.1f}% target"
            )

        click.echo("\n".join(messages))
        return

    env = savings_profile.proxy_env()
    if output_format == "json":
        click.echo(json.dumps(env, indent=2, sort_keys=True))
        return

    for key, value in env.items():
        click.echo(f"export {key}={json.dumps(value)}")


def _read_accuracy_rate(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        totals = payload.get("totals")
        if isinstance(totals, dict) and totals.get("accuracy_rate") is not None:
            return float(totals["accuracy_rate"])
        if payload.get("accuracy_preservation_rate") is not None:
            return float(payload["accuracy_preservation_rate"])
    raise click.ClickException(
        f"{path} does not contain totals.accuracy_rate or accuracy_preservation_rate"
    )


def _split_required_agents(raw: str) -> list[str]:
    return [agent.strip().lower() for agent in raw.split(",") if agent.strip()]


def _check_required_agents(
    records: Sequence[object],
    required_agents: list[str],
    target_percent: float,
) -> list[str]:
    messages: list[str] = []
    records_by_agent: dict[str, list[object]] = {}
    for record in records:
        client = str(getattr(record, "client", "") or "").strip().lower()
        if client:
            records_by_agent.setdefault(client, []).append(record)

    missing = [agent for agent in required_agents if agent not in records_by_agent]
    if missing:
        raise click.ClickException("missing required agent traffic: " + ", ".join(missing))

    for agent in required_agents:
        agent_records = records_by_agent[agent]
        before = sum(int(getattr(record, "tokens_before", 0)) for record in agent_records)
        saved = sum(int(getattr(record, "tokens_saved", 0)) for record in agent_records)
        measured = (saved / before * 100) if before > 0 else 0.0
        if measured < target_percent:
            raise click.ClickException(
                f"{agent}: {measured:.1f}% savings below {target_percent:.1f}% target"
            )
        messages.append(f"{agent}: {measured:.1f}% savings meets {target_percent:.1f}% target")
    return messages


def _write_smoke_fixture(workspace: Path) -> Path:
    logs_dir = workspace / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    perf_lines = [
        _perf_line(
            "2026-06-10 10:00:00,000", "hr_smoke_claude", "claude-sonnet", "claude", 1000, 80
        ),
        _perf_line("2026-06-10 10:01:00,000", "hr_smoke_codex", "gpt-5", "codex", 1000, 90),
        _perf_line("2026-06-10 10:02:00,000", "hr_smoke_cursor", "gpt-5", "cursor", 1000, 70),
    ]
    (logs_dir / "proxy.log").write_text("\n".join(perf_lines) + "\n", encoding="utf-8")
    eval_path = workspace / "agent-90-eval.json"
    eval_path.write_text(
        json.dumps(
            {
                "totals": {
                    "cases": 3,
                    "passed": 3,
                    "accuracy_rate": 1.0,
                    "tokens_original": 3000,
                    "tokens_compressed": 240,
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return eval_path


def _perf_line(
    timestamp: str,
    request_id: str,
    model: str,
    client: str,
    before: int,
    after: int,
) -> str:
    saved = before - after
    return (
        f"{timestamp} - headroom.proxy - INFO - [{request_id}] PERF "
        f"model={model} msgs=3 tok_before={before} tok_after={after} "
        f"tok_saved={saved} cache_read=0 cache_write=0 cache_hit_pct=0 "
        f"opt_ms=1 transforms=agent90_smoke client={client}"
    )
