"""Tests for the repository-owned architectural guardrail runner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "headroom_guardrails.py"


def _load_guardrails():
    spec = importlib.util.spec_from_file_location("headroom_guardrails", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_guardrail_runner_is_green_on_repo() -> None:
    guardrails = _load_guardrails()

    findings = guardrails.run(ROOT)

    assert findings == []


def test_backend_message_rule_catches_raw_role_content_reconstruction(tmp_path: Path) -> None:
    guardrails = _load_guardrails()
    root = tmp_path
    backend_dir = root / "headroom" / "backends"
    backend_dir.mkdir(parents=True)
    (backend_dir / "litellm.py").write_text(
        """
def convert(messages):
    converted = []
    for msg in messages:
        role = msg.get("role")
        converted.append({"role": role, "content": msg.get("content")})
    return converted
""",
        encoding="utf-8",
    )
    (backend_dir / "anyllm.py").write_text(
        "from headroom.message_contract import preserve_message_fields\n",
        encoding="utf-8",
    )

    findings = guardrails.BackendMessagePreservationRule().check(root)

    assert any(f.rule == "PY001" and "preserve_message_fields" in f.message for f in findings)


def test_positional_restore_rule_catches_optimized_message_index_restore(
    tmp_path: Path,
) -> None:
    guardrails = _load_guardrails()
    path = tmp_path / "headroom" / "proxy" / "handlers"
    path.mkdir(parents=True)
    (path / "openai.py").write_text(
        """
def restore(optimized_messages, original_messages):
    for i, msg in enumerate(optimized_messages):
        msg["reasoning_content"] = original_messages[i]["reasoning_content"]
""",
        encoding="utf-8",
    )

    findings = guardrails.NoPositionalMessageRestoreRule().check(tmp_path)

    assert any(f.rule == "PY002" for f in findings)
