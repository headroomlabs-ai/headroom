"""The macOS malloc re-exec must never replace an embedder's process.

``headroom proxy`` re-execs itself once on Darwin to apply two libmalloc knobs
that libmalloc only reads before ``main()`` (#2820). The re-exec rebuilds the
command as ``python -m headroom.cli <argv[1:]>``, which is only a faithful
reconstruction when this process really is the Headroom CLI.

When the ``proxy`` command is invoked *in-process* — pytest's ``CliRunner``, an
embedding application — ``os.execv`` replaces that process instead. The whole
pytest run is destroyed mid-suite with no traceback, and the replacement
Headroom process is handed pytest's own argv.

CI cannot catch this: the tuning is Darwin-only and no CI runner is macOS, so
these tests assert the guard's *logic* on every platform rather than relying on
the re-exec being reachable.
"""

from __future__ import annotations

import sys

import pytest

from headroom.cli import proxy as proxy_cli


@pytest.mark.parametrize(
    "argv0",
    [
        "/usr/local/bin/headroom",
        "/opt/homebrew/bin/headroom",
    ],
)
def test_console_script_is_recognised_as_the_entrypoint(
    argv0: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", [argv0, "proxy"])
    assert proxy_cli._process_is_headroom_cli_entrypoint() is True


def test_module_invocation_is_recognised_as_the_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["/venv/lib/python3.12/site-packages/headroom/cli/__main__.py", "proxy"]
    )
    assert proxy_cli._process_is_headroom_cli_entrypoint() is True


@pytest.mark.parametrize(
    "argv0",
    [
        "/venv/bin/pytest",
        # `python -m pytest` — same basename as a module run, different package.
        "/venv/lib/python3.12/site-packages/pytest/__main__.py",
        "/usr/bin/uvicorn",
        "",
    ],
)
def test_embedders_are_not_mistaken_for_the_entrypoint(
    argv0: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", [argv0, "proxy"])
    assert proxy_cli._process_is_headroom_cli_entrypoint() is False


def test_reexec_does_not_exec_when_embedded(monkeypatch: pytest.MonkeyPatch) -> None:
    """The end-to-end guard: no execv when another program owns the process."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "argv", ["/venv/bin/pytest", "tests/"])
    monkeypatch.delenv("_HEADROOM_MALLOC_TUNED", raising=False)
    for key in proxy_cli._MALLOC_TUNING:
        monkeypatch.delenv(key, raising=False)

    calls: list[object] = []
    monkeypatch.setattr(proxy_cli.os, "execv", lambda *a, **k: calls.append(a))

    proxy_cli._reexec_with_malloc_tuning()

    assert calls == []
    # The loop guard must not be set either: this process never applied the
    # tuning, so a genuine CLI child inheriting the env must still be free to.
    assert "_HEADROOM_MALLOC_TUNED" not in proxy_cli.os.environ


def test_reexec_still_execs_for_a_real_cli_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fix must not disable the feature it is guarding."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "argv", ["/usr/local/bin/headroom", "proxy", "--port", "8787"])
    monkeypatch.delenv("_HEADROOM_MALLOC_TUNED", raising=False)
    for key in proxy_cli._MALLOC_TUNING:
        monkeypatch.delenv(key, raising=False)

    calls: list[tuple] = []
    monkeypatch.setattr(proxy_cli.os, "execv", lambda *a, **k: calls.append(a))

    proxy_cli._reexec_with_malloc_tuning()

    assert len(calls) == 1
    _executable, argv = calls[0]
    assert argv[1:] == ["-m", "headroom.cli", "proxy", "--port", "8787"]
    for key, value in proxy_cli._MALLOC_TUNING.items():
        assert proxy_cli.os.environ[key] == value
