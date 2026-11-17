"""Nothing in this tree may enable HuggingFace remote code execution (A-1).

``trust_remote_code`` makes a model/dataset repository's own Python run inside
the process that loads it. Headroom loads tokenizers from identifiers that trace
back to proxied request bodies, so a single re-introduced ``=True`` anywhere is
remote code execution in the proxy. This is a tree-wide scan rather than a test
of one module: the tokenizer loader was not the only caller, and the next one
will not be either.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Assembled at runtime so this file contributes no literal occurrence of the
# token to its own scan.
FLAG = "trust_remote" + "_code"

# The value: a quoted string, or a name / number (``True``, ``1``, ``-1``, ``cfg.x``).
# Anything but the exact name ``False`` is treated as enabling the flag.
_VALUE = r"\"[^\"]*\"|'[^']*'|[-+]?[\w.]+"

# Matches the kwarg form (``flag=True``), the mapping form (``"flag": True``)
# and the subscript form (``kwargs["flag"] = True``), capturing whatever it is
# set to up to the next delimiter. The optional quote and bracket matter:
# without them the mapping form slips through, and
# ``from_pretrained(**{"flag": True})`` is exactly how this would come back.
# The value is NOT required to start with a letter: the library tests the flag
# for truthiness, so ``=1``, ``=(True)`` and ``="yes"`` enable it just as
# ``=True`` does. A value on the next line is still a value, because ``\s*``
# spans the newline.
ASSIGNMENT = re.compile(re.escape(FLAG) + r"[\"']?\s*\]?\s*[=:]\s*\(*\s*(" + _VALUE + ")")

# ``setattr(cfg, "flag", True)`` and ``kwargs.setdefault("flag", True)``: the
# name is a string literal followed by a comma, not by ``=`` or ``:``.
# Only calls that set or default the flag count: ``log.info("flag", True)`` does not.
NAMED_ARGUMENT = re.compile(
    r"(?:\b(?:setattr|getattr)\s*\([^,()]*,|\b(?:setdefault|get|pop)\s*\()\s*[\"']"
    + re.escape(FLAG)
    + r"[\"']\s*,\s*\(*\s*("
    + _VALUE
    + ")"
)


def _values(text: str) -> list[tuple[int, str]]:
    """Every ``(line number, value)`` the flag is set to in ``text``."""
    found: list[tuple[int, str]] = []
    for pattern in (ASSIGNMENT, NAMED_ARGUMENT):
        for match in pattern.finditer(text):
            found.append((text.count("\n", 0, match.start()) + 1, match.group(1)))
    return found


def _tracked_files() -> list[Path]:
    """Every git-tracked file, so the scan cannot be dodged by adding a directory.

    git-tracked rather than a filesystem walk: the repository is commonly checked
    out alongside worktrees and virtualenvs that a naive rglob would sweep in.
    Without git (an exported tarball) the package itself is still scanned, so the
    guard does not silently stop running.
    """
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return sorted((ROOT / "headroom").rglob("*.py"))
    return [ROOT / name for name in out.decode().split("\0") if name]


def test_remote_code_is_never_enabled_anywhere_in_the_tree() -> None:
    offenders: list[str] = []

    for path in _tracked_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue  # binary or vanished (submodule, symlink) — nothing to read
        if FLAG not in text:
            continue
        lines = text.splitlines()
        for lineno, value in _values(text):
            if value != "False":
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {lines[lineno - 1].strip()}")

    assert not offenders, (
        f"{FLAG} must be False everywhere — it executes repository-supplied "
        "Python in this process:\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize(
    "line",
    [
        f"model = load(x, {FLAG}=True)",
        f"model = load(x, {FLAG}=1)",
        f"model = load(x, {FLAG}=(True))",
        f'model = load(x, {FLAG}="yes")',
        f"model = load(x, {FLAG}=enabled)",
        f'kwargs["{FLAG}"] = True',
        f'load(**{{"{FLAG}": 1}})',
        f'setattr(cfg, "{FLAG}", True)',
        f'kwargs.setdefault("{FLAG}", True)',
    ],
)
def test_the_scan_flags_every_truthy_way_to_enable_the_flag(line: str) -> None:
    values = [value for _, value in _values(line)]
    assert values and all(value != "False" for value in values), line


@pytest.mark.parametrize(
    "line",
    [
        f'log.info("{FLAG}", True)',
        f'print("{FLAG}", 1)',
        f'logger.debug("{FLAG}", enabled)',
    ],
)
def test_the_scan_ignores_calls_that_do_not_set_the_flag(line: str) -> None:
    assert _values(line) == []


def test_the_scan_flags_a_value_on_the_next_line() -> None:
    text = f"load(\n    x,\n    {FLAG}=\n        True,\n)\n"
    assert [value for _, value in _values(text)] == ["True"]


@pytest.mark.parametrize(
    "line",
    [
        f"model = load(x, {FLAG}=False)",
        f"model = load(x, {FLAG}=(False))",
        f'kwargs["{FLAG}"] = False',
        f'setattr(cfg, "{FLAG}", False)',
    ],
)
def test_the_scan_accepts_an_explicit_false(line: str) -> None:
    assert [value for _, value in _values(line)] == ["False"]


def test_the_tokenizer_loader_sets_the_flag_explicitly() -> None:
    """Explicit beats relying on the library default, which upstream can change."""
    source = (ROOT / "headroom" / "tokenizers" / "huggingface.py").read_text(encoding="utf-8")
    values = [value for _, value in _values(source)]
    assert values, f"tokenizer loader no longer passes {FLAG} explicitly"
    assert set(values) == {"False"}
