"""`headroom inspect` — view original vs compressed message content.

Headroom already exposes *quantitative* telemetry (token counts, ratios) but no
way to *see* what the compressor changed. This command reads the proxy's
loopback ``/transformations/feed`` endpoint — which carries the pre/post
message snapshots when the proxy runs with ``--log-messages`` — and renders, per
request, the original vs compressed content for each message with the changed
segments highlighted (stdlib unified diff, no new dependencies). See issue #1267.
"""

from __future__ import annotations

import difflib
import json
import os
from typing import Any

import click

from .main import main


def _extract_text(content: Any) -> str:
    """Flatten a message's ``content`` (str or list of blocks) to plain text.

    Handles Anthropic/OpenAI text blocks, nested ``tool_result`` content, and
    falls back to a JSON dump for unrecognized block shapes so nothing is
    silently dropped from the diff.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("content"), (str, list)):
                    parts.append(_extract_text(block.get("content")))
                else:
                    parts.append(json.dumps(block, ensure_ascii=False, sort_keys=True))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _role(msg: Any) -> str:
    return str(msg.get("role", "?")) if isinstance(msg, dict) else "?"


def _content_of(msg: Any) -> Any:
    return msg.get("content") if isinstance(msg, dict) else msg


def _render_request(transformation: dict[str, Any], *, full: bool) -> None:
    rid = transformation.get("request_id") or "?"
    model = transformation.get("model") or "?"
    before = transformation.get("input_tokens_original")
    after = transformation.get("input_tokens_optimized")
    saved = transformation.get("tokens_saved")
    pct = transformation.get("savings_percent")
    transforms = ", ".join(transformation.get("transforms_applied") or []) or "none"

    click.echo(click.style(f"\n━━ {rid}  {model}", bold=True))
    click.echo(f"   tokens {before} → {after}  (saved {saved}, {pct}%)   transforms: {transforms}")

    originals = transformation.get("request_messages") or []
    compressed = transformation.get("compressed_messages") or []
    count = max(len(originals), len(compressed))
    any_change = False

    for i in range(count):
        original = originals[i] if i < len(originals) else {}
        comp = compressed[i] if i < len(compressed) else {}
        original_text = _extract_text(_content_of(original))
        comp_text = _extract_text(_content_of(comp))
        unchanged = original_text == comp_text
        if unchanged and not full:
            continue
        any_change = True
        role = _role(original) if original else _role(comp)
        click.echo(click.style(f"\n  [{i}] {role}", fg="cyan"))
        if unchanged:
            click.echo("    (unchanged)")
            continue
        for line in difflib.unified_diff(
            original_text.splitlines(),
            comp_text.splitlines(),
            fromfile="original",
            tofile="compressed",
            lineterm="",
        ):
            color: str | None = None
            if line.startswith("+") and not line.startswith("+++"):
                color = "green"
            elif line.startswith("-") and not line.startswith("---"):
                color = "red"
            elif line.startswith("@@"):
                color = "yellow"
            click.echo("    " + (click.style(line, fg=color) if color else line))

    if not any_change:
        click.echo(
            "   (no per-message content changes — savings came from "
            "structural / transform-level edits)"
        )


@main.command("inspect")
@click.option(
    "--port",
    "-p",
    default=None,
    type=click.IntRange(1, 65535),
    envvar="HEADROOM_PORT",
    help="Proxy port to query (default: 8787, env: HEADROOM_PORT)",
)
@click.option(
    "--last",
    default=1,
    type=click.IntRange(min=1),
    help="Show the N most recent requests (default: 1)",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format (default: text). json emits the raw feed for offline diffing.",
)
@click.option(
    "--full",
    is_flag=True,
    help="Show unchanged messages too (default: only messages the compressor changed)",
)
def inspect_cmd(port: int | None, last: int, output_format: str, full: bool) -> None:
    """Show original vs compressed content for recent proxy requests.

    \b
    Requires a running proxy started with --log-messages (or --log-file), which
    captures the pre/post-compression message snapshots the diff reads.

    \b
    Examples:
        headroom inspect                 Inspect the most recent request
        headroom inspect --last 5        Inspect the 5 most recent requests
        headroom inspect --full          Include unchanged messages
        headroom inspect --format json   Raw feed for piping into another tool
    """
    from headroom.install.health import probe_json

    resolved_port = port if port is not None else int(os.environ.get("HEADROOM_PORT", "8787"))
    base_url = f"http://127.0.0.1:{resolved_port}"
    payload = probe_json(f"{base_url}/transformations/feed?limit={last}", timeout=5.0)

    if payload is None:
        raise click.ClickException(
            f"No reachable proxy on {base_url}. Start one with `headroom proxy` "
            "(or pass --port to point at a running instance)."
        )
    if not payload.get("log_full_messages"):
        raise click.ClickException(
            "The proxy isn't capturing message content, so there's nothing to diff. "
            "Restart it with `headroom proxy --log-messages`."
        )

    transformations = payload.get("transformations") or []
    if not transformations:
        click.echo("No requests recorded yet. Send traffic through the proxy and retry.")
        return

    if output_format == "json":
        click.echo(json.dumps(transformations, indent=2, ensure_ascii=False))
        return

    # The feed returns oldest→newest; show the most recent first.
    for transformation in reversed(transformations):
        _render_request(transformation, full=full)
