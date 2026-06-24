"""Declarative test-case harness for Docker e2e runners.

Each command gets its own ``run.py`` file that builds a list of ``Case``
objects and calls ``run_cases(cases)``. The harness handles:

* creating a scratch HOME and project directory per case
* dropping the requested shims into a dedicated shim dir
* building a clean PATH that only exposes the shim dir + minimal system dirs
* invoking the ``headroom`` subprocess with the case's argv
* running the case's assertions against stdout / stderr / exit code / files
* reporting pass/fail per case and a final summary

``run_cases`` returns a non-zero exit code if any case fails, so Docker
containers driving it can fail fast.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .assertions import assert_exit, assert_stderr_contains, assert_stdout_contains
from .path_env import with_clean_path
from .shims import ShimBehavior, make_shim

CaseCallback = Callable[["CaseContext"], None]


@dataclass
class CaseContext:
    """Runtime context passed to assertion callbacks."""

    name: str
    home: Path
    project: Path
    shim_dir: Path
    shim_log: Path
    stdout: str
    stderr: str
    exit_code: int


@dataclass
class Case:
    """Declarative specification of a single e2e test case.

    Attributes:
        name: Human-readable identifier, printed on success/failure.
        argv: Arguments passed to ``headroom`` (e.g. ``["init", "-g", "claude"]``).
        shims: Mapping of shim name -> behavior to drop into the shim dir.
        env_extra: Extra env vars layered on top of the clean env.
        expected_exit: Required exit code (default 0).
        expected_stdout_contains: Substrings that must appear on stdout.
        expected_stderr_contains: Substrings that must appear on stderr.
        expected_files: Paths (relative to home or project) that must exist.
                        Use ``{home}/...`` or ``{project}/...`` placeholders.
        extra_assertions: Optional list of callbacks invoked after exit-code /
                          stdout / stderr / file checks pass. Receives a
                          ``CaseContext``. Use for JSON-content assertions,
                          shim-log inspection, etc.
    """

    name: str
    argv: list[str]
    shims: dict[str, ShimBehavior] = field(default_factory=dict)
    env_extra: dict[str, str] = field(default_factory=dict)
    expected_exit: int = 0
    expected_stdout_contains: list[str] = field(default_factory=list)
    expected_stderr_contains: list[str] = field(default_factory=list)
    expected_files: list[str] = field(default_factory=list)
    extra_assertions: list[CaseCallback] = field(default_factory=list)


def _log(message: str) -> None:
    print(f"[e2e] {message}", flush=True)


def _resolve_placeholder(spec: str, *, home: Path, project: Path) -> Path:
    return Path(spec.format(home=str(home), project=str(project)))


def _resolve_headroom_bin(name: str) -> str:
    """Return the absolute path to the headroom binary before PATH is scrubbed.

    ``with_clean_path`` intentionally narrows PATH so agent shims dominate;
    that would also hide the real ``headroom`` binary (typically at
    ``/opt/*venv/bin/headroom`` or similar). Resolving up-front lets the
    subprocess launch even after PATH is cleaned.
    """

    if os.sep in name or (os.altsep and os.altsep in name):
        return name
    import shutil

    resolved = shutil.which(name)
    if resolved:
        return resolved
    # Fall back to the bare name; subprocess will raise a clear
    # FileNotFoundError that the case output surfaces.
    return name


def _run_single(case: Case, headroom_bin: str = "headroom") -> bool:
    """Execute one case. Return True on pass, False on fail."""

    with tempfile.TemporaryDirectory(prefix=f"headroom-e2e-{case.name}-") as temp_raw:
        temp_root = Path(temp_raw)
        home = temp_root / "home"
        project = temp_root / "project"
        shim_dir = temp_root / "bin"
        shim_log = temp_root / "shim-log.jsonl"
        home.mkdir(parents=True)
        project.mkdir(parents=True)

        for shim_name, behavior in case.shims.items():
            make_shim(shim_name, shim_dir, behavior=behavior)

        # Resolve headroom to its absolute path BEFORE mutating PATH so the
        # shim dir can dominate PATH without losing the headroom binary.
        resolved_bin = _resolve_headroom_bin(headroom_bin)

        with with_clean_path([shim_dir]) as env:
            env["HOME"] = str(home)
            env["USERPROFILE"] = str(home)
            env["HEADROOM_E2E_SHIM_LOG"] = str(shim_log)
            env.update(case.env_extra)

            proc = subprocess.run(
                [resolved_bin, *case.argv],
                env=env,
                cwd=str(project),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
            )

        ctx = CaseContext(
            name=case.name,
            home=home,
            project=project,
            shim_dir=shim_dir,
            shim_log=shim_log,
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
        )

        try:
            assert_exit(proc.returncode, case.expected_exit, context=f"case {case.name}")
            for needle in case.expected_stdout_contains:
                assert_stdout_contains(proc.stdout, needle)
            for needle in case.expected_stderr_contains:
                assert_stderr_contains(proc.stderr, needle)
            for spec in case.expected_files:
                path = _resolve_placeholder(spec, home=home, project=project)
                if not path.exists():
                    raise AssertionError(f"Expected file {path} not found")
            for callback in case.extra_assertions:
                callback(ctx)
        except AssertionError as exc:
            _log(f"FAIL {case.name}: {exc}")
            if proc.stdout.strip():
                _log(f"  stdout: {proc.stdout.rstrip()}")
            if proc.stderr.strip():
                _log(f"  stderr: {proc.stderr.rstrip()}")
            return False

        _log(f"PASS {case.name}")
        return True


def _run_in_scratch(
    case: Case,
    *,
    home: Path,
    project: Path,
    shim_dir: Path,
    shim_log: Path,
    headroom_bin: str,
) -> bool:
    """Execute one case inside a pre-existing scratch layout.

    Shims are *added* to ``shim_dir`` (existing shims from prior sequence
    steps are preserved). This enables sequence cases to build up shim state.
    """

    for shim_name, behavior in case.shims.items():
        make_shim(shim_name, shim_dir, behavior=behavior)

    resolved_bin = _resolve_headroom_bin(headroom_bin)

    with with_clean_path([shim_dir]) as env:
        env["HOME"] = str(home)
        env["USERPROFILE"] = str(home)
        env["HEADROOM_E2E_SHIM_LOG"] = str(shim_log)
        env.update(case.env_extra)

        proc = subprocess.run(
            [resolved_bin, *case.argv],
            env=env,
            cwd=str(project),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )

    ctx = CaseContext(
        name=case.name,
        home=home,
        project=project,
        shim_dir=shim_dir,
        shim_log=shim_log,
        stdout=proc.stdout,
        stderr=proc.stderr,
        exit_code=proc.returncode,
    )

    try:
        assert_exit(proc.returncode, case.expected_exit, context=f"case {case.name}")
        for needle in case.expected_stdout_contains:
            assert_stdout_contains(proc.stdout, needle)
        for needle in case.expected_stderr_contains:
            assert_stderr_contains(proc.stderr, needle)
        for spec in case.expected_files:
            path = _resolve_placeholder(spec, home=home, project=project)
            if not path.exists():
                raise AssertionError(f"Expected file {path} not found")
        for callback in case.extra_assertions:
            callback(ctx)
    except AssertionError as exc:
        _log(f"FAIL {case.name}: {exc}")
        if proc.stdout.strip():
            _log(f"  stdout: {proc.stdout.rstrip()}")
        if proc.stderr.strip():
            _log(f"  stderr: {proc.stderr.rstrip()}")
        return False

    _log(f"PASS {case.name}")
    return True


def run_cases(
    cases: list[Case],
    *,
    headroom_bin: str = "headroom",
    fail_fast: bool = False,
) -> int:
    """Run each case in its own scratch dir. Return exit code (0 = all pass)."""

    passed = 0
    failed = 0
    for case in cases:
        ok = _run_single(case, headroom_bin=headroom_bin)
        if ok:
            passed += 1
        else:
            failed += 1
            if fail_fast:
                break

    _log(f"Summary: {passed} passed, {failed} failed, {len(cases)} total")
    return 0 if failed == 0 else 1


def run_case_sequence(
    cases: list[Case],
    *,
    headroom_bin: str = "headroom",
    label: str = "sequence",
    fail_fast: bool = True,
) -> int:
    """Run cases sequentially inside a single shared scratch dir.

    Useful when later cases must observe state left by earlier ones (e.g.
    ``headroom init`` accumulating targets in a shared manifest across
    successive calls).
    """

    passed = 0
    failed = 0
    with tempfile.TemporaryDirectory(prefix=f"headroom-e2e-{label}-") as temp_raw:
        temp_root = Path(temp_raw)
        home = temp_root / "home"
        project = temp_root / "project"
        shim_dir = temp_root / "bin"
        shim_log = temp_root / "shim-log.jsonl"
        home.mkdir(parents=True)
        project.mkdir(parents=True)

        for case in cases:
            ok = _run_in_scratch(
                case,
                home=home,
                project=project,
                shim_dir=shim_dir,
                shim_log=shim_log,
                headroom_bin=headroom_bin,
            )
            if ok:
                passed += 1
            else:
                failed += 1
                if fail_fast:
                    break

    _log(f"Summary ({label}): {passed} passed, {failed} failed, {len(cases)} total")
    return 0 if failed == 0 else 1


# Allow callers to adopt a different exit strategy (e.g. raising) easily.
def main_from_cases(cases: list[Case]) -> None:
    """Convenience entry point for ``run.py`` scripts."""

    code = run_cases(cases)
    sys.exit(code)


__all__ = [
    "Case",
    "CaseContext",
    "main_from_cases",
    "run_case_sequence",
    "run_cases",
]

# Silence unused-import lint for re-exports used by callers.
_ = os
