"""The proof table printed for the docs must stay a single, labelled table (#4732).

``benchmarks/index_proof_table.py`` used to print "default (protect_recent=4)"
and "full corpus (protect_recent=0)". Both tables were identical in every row,
because ``ContentRouter`` applies ``protect_recent`` only to content detected as
SOURCE_CODE and every scenario here is a JSON payload
(``tests/test_transforms_content_router.py::
test_protect_recent_does_not_shield_non_code_tool_payloads`` pins that reason).
The second table could therefore never differ, so it was removed.

These tests pin the OUTPUT, not the compression: everything expensive is
monkeypatched away, so they stay fast and offline.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

BENCHMARKS_DIR = Path(__file__).resolve().parents[1] / "benchmarks"

SCENARIOS = (
    "Code search (100 results)",
    "SRE incident debugging",
    "Codebase exploration",
    "GitHub issue triage",
)


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Import the harness with its corpus generators and compression stubbed out."""
    # The harness imports `real_world_agent_benchmark` as a sibling module, which
    # only resolves when it is run as a script with `benchmarks/` on sys.path.
    monkeypatch.syspath_prepend(str(BENCHMARKS_DIR))
    sys.modules.pop("benchmarks.index_proof_table", None)

    module = pytest.importorskip("benchmarks.index_proof_table")

    class _FakeTokenizer:
        _tokenizer = object()

        def __init__(self, model: str | None = None) -> None:
            self.model = model

    numbers = {"before": 1_000, "after": 500}

    def _fake_measure(label: str, tools: list[dict], tok: Any, config: Any) -> dict:
        return {
            "scenario": label,
            "before": numbers["before"],
            "after": numbers["after"],
            "saved": numbers["before"] - numbers["after"],
            "savings_pct": 50.0,
            "transforms": [],
        }

    tool = {"result": [{"path": "src/a.py", "score": 1}]}
    monkeypatch.setattr(module, "seed_everything", lambda seed: seed)
    monkeypatch.setattr(module, "measure", _fake_measure)
    monkeypatch.setattr(module, "OpenAICompatibleTokenCounter", _FakeTokenizer)
    monkeypatch.setattr(module, "generate_github_code_search", lambda *a, **k: tool)
    monkeypatch.setattr(
        module,
        "create_sre_debugging_scenario",
        lambda: types.SimpleNamespace(tools=[tool]),
    )
    monkeypatch.setattr(
        module,
        "create_codebase_exploration_scenario",
        lambda: types.SimpleNamespace(tools=[tool]),
    )
    monkeypatch.setattr(
        module,
        "create_issue_triage_scenario",
        lambda: types.SimpleNamespace(tools=[tool]),
    )
    return module


def test_proof_table_prints_exactly_one_table(
    harness: types.ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A second configuration re-introduces a table that can only duplicate the first."""
    monkeypatch.setattr(sys, "argv", ["index_proof_table.py", "--seed", "20260902"])
    assert harness.main() == 0
    out = capsys.readouterr().out

    assert out.count("=== default (protect_recent=4) ===") == 1, out
    assert "protect_recent=0" not in out


def test_proof_markdown_block_is_the_default_config(
    harness: types.ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The block pasted into the docs must state the config it was measured under."""
    monkeypatch.setattr(sys, "argv", ["index_proof_table.py", "--seed", "20260902"])
    assert harness.main() == 0
    out = capsys.readouterr().out

    assert "Markdown for docs/content/docs/index.mdx (default config" in out
    markdown = out.split("Markdown for docs/content/docs/index.mdx", 1)[1]
    for scenario in SCENARIOS:
        assert f"| {scenario} | 1,000 | 500 | **50%** |" in markdown
