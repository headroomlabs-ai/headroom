"""Synthetic session fixtures that reproduce known waste patterns.

These build :class:`SessionData` shaped like the real patterns Headroom Learn
must catch, so both unit tests and the RTK-loop eval (``benchmarks/
rtk_loop_learn_eval.py``) drive the analyzer from one source of truth instead
of hand-mocking calls inline.

The headline fixture is the **RTK re-fetch loop**. RTK truncates a shell
command's output; when the truncation drops what the agent needed, the agent
re-runs a *variant* to fetch more. Critically these calls SUCCEED
(``is_error=False``) — the loop is invisible to failure-only analysis, which
is exactly why it was historically under-weighted.
"""

from __future__ import annotations

from .models import ErrorCategory, SessionData, ToolCall


def _tc(
    name: str,
    command: str,
    output: str,
    *,
    msg_index: int,
    is_error: bool = False,
    error_category: ErrorCategory = ErrorCategory.UNKNOWN,
) -> ToolCall:
    """Build a ToolCall, keying input on the field the tool's summary reads."""
    if name.lower() in ("bash", "shell"):
        input_data = {"command": command}
    elif name.lower() in ("read",):
        input_data = {"file_path": command}
    elif name.lower() in ("grep",):
        input_data = {"pattern": command}
    else:
        input_data = {"command": command}
    return ToolCall(
        name=name,
        tool_call_id=f"tc_{msg_index}",
        input_data=input_data,
        output=output,
        is_error=is_error,
        error_category=error_category if is_error else ErrorCategory.UNKNOWN,
        msg_index=msg_index,
        output_bytes=len(output),
    )


def rtk_refetch_loop_session(
    session_id: str = "rtk-loop",
    *,
    repetitions: int = 5,
    bytes_per_call: int = 4000,
) -> SessionData:
    """A session where RTK truncation forces repeated re-fetches of one command.

    The agent greps a large log; RTK rewrites each invocation with an output
    limit. Each call succeeds but returns a truncated window, so the agent
    bumps the limit / shifts the window and re-runs — ``repetitions`` times.
    None of the calls error. The fix a good guardrail should produce: fetch the
    full result up front (e.g., disable RTK truncation for this command, or
    grep into a file and read it once).
    """
    calls: list[ToolCall] = []
    limit = 50
    for i in range(repetitions):
        # Same base command; only the output-limit varies — the RTK signature.
        command = f"grep -rn 'TimeoutError' logs/ | head -{limit}"
        output = "logs/app.log:" + ("x" * (bytes_per_call - 20)) + "\n(truncated)"
        calls.append(_tc("Bash", command, output, msg_index=i * 2))
        limit += 50  # agent asks for more next time — still truncated
    return SessionData(session_id=session_id, tool_calls=calls)


def error_loop_session(
    session_id: str = "error-loop",
    *,
    repetitions: int = 4,
) -> SessionData:
    """A session where the same call fails repeatedly (classic retry loop)."""
    calls: list[ToolCall] = []
    for i in range(repetitions):
        calls.append(
            _tc(
                "Bash",
                "python3 run_tests.py",
                "python3: command not found",
                msg_index=i * 2,
                is_error=True,
                error_category=ErrorCategory.COMMAND_NOT_FOUND,
            )
        )
    return SessionData(session_id=session_id, tool_calls=calls)


def one_off_error_session(session_id: str = "one-off") -> SessionData:
    """A session with a single, non-repeated failure — should NOT be a loop."""
    return SessionData(
        session_id=session_id,
        tool_calls=[
            _tc(
                "Read",
                "/etc/missing.conf",
                "Error: file not found",
                msg_index=0,
                is_error=True,
                error_category=ErrorCategory.FILE_NOT_FOUND,
            ),
            _tc("Bash", "ls -la", "total 8\ndrwxr-xr-x", msg_index=1),
        ],
    )
