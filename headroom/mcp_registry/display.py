"""Human-readable rendering of MCP registration results.

The CLI ``mcp install`` command, ``init`` flow, and ``wrap claude`` setup
all consume the same ``dict[str, RegisterResult]`` and want to print one
line per agent. They differ only in label format (e.g. ``"  claude:"`` vs
``"  MCP retrieve tool:"``), whether to show ALREADY-registered as a
success line, and which corrective command to suggest on mismatch.

Centralizing those choices here keeps every status branch in one file.
Adding a new :class:`RegisterStatus` member becomes a single edit; the
call sites compose by passing flags rather than re-writing the switch.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import RegisterResult, RegisterStatus

DEFAULT_OVERWRITE_HINT = "headroom mcp install --force"
DEFAULT_RESTART_HINT = "restart the agent if it was already running"


def format_result(
    agent: str,
    result: RegisterResult,
    *,
    label: str | None = None,
    verbose: bool = False,
    overwrite_hint: str = DEFAULT_OVERWRITE_HINT,
    restart_hint: str = DEFAULT_RESTART_HINT,
) -> str | None:
    """Render one ``(agent, result)`` pair as a single display line.

    Returns ``None`` to suppress output (e.g. ALREADY when not ``verbose``).

    Args:
        agent: Stable agent name (used as default label).
        result: Outcome from :func:`install_everywhere` or a registrar.
        label: Override the leading label. Defaults to the agent name.
        verbose: If ``True``, include status lines that are otherwise
            silent (e.g. ALREADY).
        overwrite_hint: Command to suggest when the existing config differs.
        restart_hint: Hint appended to a fresh registration line.
    """
    label = label if label is not None else agent
    status = result.status

    if status == RegisterStatus.REGISTERED:
        return f"  {label}: registered ({restart_hint})"
    if status == RegisterStatus.ALREADY:
        return f"  {label}: already registered" if verbose else None
    if status == RegisterStatus.NOT_DETECTED:
        return f"  {label}: not detected on this system, skipped"
    if status == RegisterStatus.MISMATCH:
        suffix = f" To update: {overwrite_hint}" if overwrite_hint else ""
        return f"  {label}: existing config differs ({result.detail}).{suffix}"
    if status == RegisterStatus.NO_SDK:
        return f"  {label}: MCP SDK missing — install with `pip install 'headroom-ai[mcp]'`"
    # FAILED or any future unhandled status
    return f"  {label}: install failed ({status.value}): {result.detail}"


def format_results(
    results: dict[str, RegisterResult],
    *,
    label_for: Callable[[str], str | None] | None = None,
    verbose: bool = False,
    overwrite_hint: str = DEFAULT_OVERWRITE_HINT,
    restart_hint: str = DEFAULT_RESTART_HINT,
) -> list[str]:
    """Render a results dict to a list of display lines.

    ``label_for`` is an optional ``agent -> label`` mapper. Pass ``None``
    (default) to use the agent name as the label.
    """
    lines: list[str] = []
    for agent, result in results.items():
        label = label_for(agent) if label_for is not None else None
        line = format_result(
            agent,
            result,
            label=label,
            verbose=verbose,
            overwrite_hint=overwrite_hint,
            restart_hint=restart_hint,
        )
        if line is not None:
            lines.append(line)
    return lines


def any_succeeded(results: dict[str, RegisterResult]) -> bool:
    """True when at least one agent ended in REGISTERED or ALREADY."""
    return any(r.ok for r in results.values())
