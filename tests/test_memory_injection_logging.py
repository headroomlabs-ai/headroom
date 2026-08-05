"""Tests for memory-injection request tags."""

import ast
from pathlib import Path

from headroom.proxy.helpers import log_memory_injection

HANDLER_FILES = [
    Path("headroom/proxy/handlers/anthropic.py"),
    Path("headroom/proxy/handlers/openai.py"),
    Path("headroom/proxy/handlers/gemini.py"),
]


def test_log_memory_injection_marks_only_successful_injection() -> None:
    tags: dict[str, str] = {}

    log_memory_injection(
        request_id="hr_test_memory",
        session_id=None,
        decision="no_eligible_user_turn",
        bytes_injected=0,
        tags=tags,
    )
    assert "memory_injected" not in tags

    log_memory_injection(
        request_id="hr_test_memory",
        session_id=None,
        decision="injected_live_zone_tail",
        bytes_injected=42,
        tags=tags,
    )
    assert tags["memory_injected"] == "true"


def test_successful_handler_injection_logs_pass_tags() -> None:
    missing: list[tuple[Path, int]] = []
    successful_sites = 0

    for file_path in HANDLER_FILES:
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "log_memory_injection":
                continue
            kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
            bytes_injected = kwargs.get("bytes_injected")
            if isinstance(bytes_injected, ast.Constant) and bytes_injected.value == 0:
                continue
            successful_sites += 1
            if "tags" not in kwargs:
                missing.append((file_path, node.lineno))

    assert successful_sites >= 6, "Expected all current provider injection sites"
    assert not missing, "Successful memory injections missing tags: " + ", ".join(
        f"{path}:{line}" for path, line in missing
    )
