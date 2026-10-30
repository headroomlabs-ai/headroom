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

# Matches the kwarg form (``flag=True``), the mapping form (``"flag": True``)
# and the subscript form (``kwargs["flag"] = True``), capturing whatever it is
# set to. The optional quote and bracket matter: without them the mapping form
# slips through, and ``from_pretrained(**{"flag": True})`` is exactly how this
# would come back.
ASSIGNMENT = re.compile(re.escape(FLAG) + r"[\"']?\s*\]?\s*[=:]\s*([A-Za-z_][\w.]*)")


def _tracked_files() -> list[Path]:
    """Every git-tracked file, so the scan cannot be dodged by adding a directory.

    git-tracked rather than a filesystem walk: the repository is commonly checked
    out alongside worktrees and virtualenvs that a naive rglob would sweep in.
    """
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
        pytest.skip(f"git ls-files unavailable: {exc}")
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
        for lineno, line in enumerate(text.splitlines(), start=1):
            for value in ASSIGNMENT.findall(line):
                if value != "False":
                    offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")

    assert not offenders, (
        f"{FLAG} must be False everywhere — it executes repository-supplied "
        "Python in this process:\n" + "\n".join(offenders)
    )


def test_the_tokenizer_loader_sets_the_flag_explicitly() -> None:
    """Explicit beats relying on the library default, which upstream can change."""
    source = (ROOT / "headroom" / "tokenizers" / "huggingface.py").read_text(encoding="utf-8")
    values = ASSIGNMENT.findall(source)
    assert values, f"tokenizer loader no longer passes {FLAG} explicitly"
    assert set(values) == {"False"}
