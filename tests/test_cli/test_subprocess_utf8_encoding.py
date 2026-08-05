"""Guard: every text-mode subprocess call uses the shared wrapper.

On Windows, text-mode ``subprocess`` defaults to the locale codec (cp1252) when
``encoding=`` is omitted. Child output that is UTF-8 (e.g. a repo index printing
symbol names with ``↔``/``—``) then raises ``UnicodeDecodeError: 'charmap'`` in the
reader thread and aborts startup.

The fix is a shared wrapper at ``headroom._subprocess`` that automatically sets
``encoding="utf-8", errors="replace"`` when ``text=True`` or
``universal_newlines=True``.  This test asserts that no raw ``subprocess.run`` /
``subprocess.Popen`` (or similar) call with ``text=True`` exists in the shipped
package — they must all go through the wrapper.
"""

from __future__ import annotations

import ast
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parents[2] / "headroom"
_SKIP = {"_subprocess.py"}
_SUBPROCESS_FUNCS = {"run", "Popen", "check_output", "check_call", "call"}


def _kwarg(call: ast.Call, name: str) -> ast.keyword | None:
    return next((k for k in call.keywords if k.arg == name), None)


def _is_true(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _is_raw_subprocess_call(call: ast.Call) -> bool:
    func = call.func
    return isinstance(func, ast.Attribute) and func.attr in _SUBPROCESS_FUNCS


def _offenders() -> list[str]:
    bad: list[str] = []
    for path in _PACKAGE.rglob("*.py"):
        if path.name in _SKIP:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_raw_subprocess_call(node):
                continue
            text_kw = _kwarg(node, "text")
            un_kw = _kwarg(node, "universal_newlines")
            text_mode = (text_kw is not None and _is_true(text_kw.value)) or (
                un_kw is not None and _is_true(un_kw.value)
            )
            if text_mode:
                rel = path.relative_to(_PACKAGE.parent)
                bad.append(f"{rel}:{node.lineno}")
    return bad


def test_text_mode_subprocess_calls_use_wrapper() -> None:
    offenders = _offenders()
    assert not offenders, (
        "raw subprocess calls with text=True found (use headroom._subprocess wrapper):\n"
        + "\n".join(offenders)
    )


if __name__ == "__main__":  # pragma: no cover - manual run
    test_text_mode_subprocess_calls_use_wrapper()
    print("ok: all text-mode subprocess calls use the shared wrapper")
