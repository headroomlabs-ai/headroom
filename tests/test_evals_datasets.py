from __future__ import annotations

import json
import sys
import urllib.request
from types import SimpleNamespace
from urllib.error import URLError

import pytest

from headroom.evals import datasets


def install_fake_datasets(
    monkeypatch: pytest.MonkeyPatch,
    mapping: dict[tuple[str, str | None, str | None], list[dict[str, object]]],
) -> list[tuple[str, str | None, str | None]]:
    calls: list[tuple[str, str | None, str | None]] = []

    def fake_load_dataset(name: str, subset: str | None = None, split: str | None = None):
        key = (name, subset, split)
        calls.append(key)
        return mapping[key]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    return calls


def test_check_datasets_installed_errors_without_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "datasets", raising=False)

    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001
        if name == "datasets":
            raise ImportError("missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="HuggingFace datasets required"):
        datasets._check_datasets_installed()


def test_load_hotpotqa_and_natural_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = install_fake_datasets(
        monkeypatch,
        {
            ("hotpotqa/hotpot_qa", "fullwiki", "validation"): [
                {
                    "context": {"title": ["Page A"], "sentences": [["Line 1", "Line 2"]]},
                    "question": "Who?",
                    "answer": "Alice",
                    "type": "bridge",
                    "level": "easy",
                }
            ],
            ("google-research-datasets/natural_questions", "default", "validation"): [
                {"document": {}, "question": {"text": "skip me"}},
                {
                    "document": {
                        "tokens": {
                            "token": ["<p>", "Ada", "Lovelace", "wrote", "notes"],
                            "is_html": [True, False, False, False, False],
                        }
                    },
                    "question": {"text": "Who wrote notes?"},
                    "annotations": {"short_answers": [[{"start_token": 1, "end_token": 3}]]},
                },
            ],
        },
    )

    hotpot = datasets.load_hotpotqa(n=1)
    natural = datasets.load_natural_questions(n=1)

    assert calls == [
        ("hotpotqa/hotpot_qa", "fullwiki", "validation"),
        ("google-research-datasets/natural_questions", "default", "validation"),
    ]
    assert hotpot.name == "HotpotQA"
    assert hotpot.cases[0].context == "## Page A\nLine 1\nLine 2"
    assert hotpot.cases[0].metadata["type"] == "bridge"
    assert natural.name == "Natural_Questions"
    assert natural.cases[0].context == "Ada Lovelace wrote notes"
    assert natural.cases[0].ground_truth == "Ada Lovelace"


def test_load_triviaqa_msmarco_and_squad(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_datasets(
        monkeypatch,
        {
            ("trivia_qa", "rc", "validation"): [
                {"question": "", "search_results": {"search_context": ["unused"]}},
                {
                    "question": "Question 1",
                    "search_results": {"search_context": ["A", "B"]},
                    "answer": {"value": "Answer", "aliases": ["Alias"]},
                },
                {
                    "question": "Question 2",
                    "search_results": {"search_context": []},
                    "entity_pages": {"wiki_context": ["Wiki 1", "Wiki 2"]},
                    "answer": {"normalized_value": "Normalized"},
                },
            ],
            ("microsoft/ms_marco", "v2.1", "validation"): [
                {"query": "", "passages": {"passage_text": ["skip"], "is_selected": [True]}},
                {
                    "query": "Find docs",
                    "passages": {"passage_text": ["Doc 1", "Doc 2"], "is_selected": [True, False]},
                    "answers": ["Primary answer"],
                    "query_type": "description",
                },
            ],
            ("rajpurkar/squad_v2", None, "validation"): [
                {"answers": {"text": []}, "context": "skip", "question": "skip"},
                {
                    "context": "Context",
                    "question": "Question",
                    "answers": {"text": ["First answer"]},
                    "title": "Title",
                },
            ],
        },
    )

    trivia = datasets.load_triviaqa(n=2)
    msmarco = datasets.load_msmarco(n=1)
    squad = datasets.load_squad(n=1)

    assert len(trivia.cases) == 2
    assert trivia.cases[0].context == "A\n\nB"
    assert trivia.cases[1].ground_truth == "Normalized"
    assert trivia.cases[1].metadata["aliases"] == []
    assert msmarco.cases[0].context.startswith("[RELEVANT] Passage 1: Doc 1")
    assert msmarco.cases[0].metadata["num_passages"] == 2
    assert squad.cases[0].ground_truth == "First answer"
    assert squad.cases[0].metadata["title"] == "Title"


def test_load_longbench_narrativeqa_toolbench_codesearchnet_and_humaneval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_datasets(
        monkeypatch,
        {
            ("THUDM/LongBench", "qasper", "test"): [
                {"context": "", "input": "skip"},
                {"context": "Long context", "input": "Question", "answers": ["Truth"]},
            ],
            ("deepmind/narrativeqa", None, "test"): [
                {
                    "document": {"summary": {"text": "Story summary"}, "kind": "movie"},
                    "question": {"text": "What happened?"},
                    "answers": [{"text": "A"}, {"text": "B"}],
                }
            ],
            ("ToolBench/ToolBench", "G1", "test"): [
                {"api_list": [], "query": "skip"},
                {
                    "api_list": [
                        {
                            "api_name": "weather",
                            "api_description": "Get weather",
                            "required_parameters": [{"name": "city"}],
                            "optional_parameters": [{"name": "unit"}],
                        }
                    ],
                    "query": "Weather in SF?",
                    "answer": "Call weather",
                },
            ],
            ("code_search_net", "python", "test"): [
                {"func_code_string": "", "func_documentation_string": "skip"},
                {
                    "func_code_string": "def add(a, b): return a + b",
                    "func_documentation_string": "Add two numbers.",
                    "func_name": "add",
                    "repository_name": "repo",
                },
            ],
            ("openai_humaneval", None, "test"): [
                {"prompt": "", "canonical_solution": "skip"},
                {
                    "task_id": "HumanEval/1",
                    "prompt": "def solve(x):",
                    "canonical_solution": "return x",
                    "entry_point": "solve",
                    "test": "assert solve(1) == 1",
                },
            ],
        },
    )

    longbench = datasets.load_longbench(n=2, task="qasper")
    narrative = datasets.load_narrativeqa(n=1)
    toolbench = datasets.load_toolbench(n=1, category="G1")
    codesearchnet = datasets.load_codesearchnet(n=1, language="python")
    humaneval = datasets.load_humaneval(n=2)

    assert longbench.name == "LongBench_qasper"
    assert longbench.cases[0].metadata["context_length"] == len("Long context")
    assert narrative.cases[0].metadata["all_answers"] == ["A", "B"]
    assert toolbench.cases[0].metadata["num_tools"] == 1
    assert '"name": "weather"' in toolbench.cases[0].context
    assert codesearchnet.cases[0].ground_truth == "Add two numbers."
    assert humaneval.cases[0].id == "humaneval_HumanEval/1"
    assert humaneval.cases[0].metadata["entry_point"] == "solve"


def test_load_longbench_toolbench_and_codesearchnet_wrap_loader_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load_dataset(name: str, subset: str | None = None, split: str | None = None):  # noqa: ANN001
        raise RuntimeError(f"broken {name}:{subset}:{split}")

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))

    with pytest.raises(ValueError, match="Failed to load LongBench task 'gov_report'"):
        datasets.load_longbench(task="gov_report")
    with pytest.raises(ValueError, match="Failed to load ToolBench category 'G2'"):
        datasets.load_toolbench(category="G2")
    with pytest.raises(ValueError, match="Failed to load CodeSearchNet for 'go'"):
        datasets.load_codesearchnet(language="go")


def test_load_bfcl_success_and_download_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    data_lines = "\n".join(
        [
            json.dumps(
                {
                    "id": "case-1",
                    "question": [[{"role": "user", "content": "How is the weather?"}]],
                    "function": [{"name": "weather"}],
                }
            ),
            json.dumps({"question": [123], "function": []}),
        ]
    )
    gt_lines = json.dumps({"id": "case-1", "ground_truth": [{"name": "weather"}]})

    def fake_urlopen(url: str):  # noqa: ANN001
        if "possible_answer/BFCL_v3_simple.json" in url:
            return SimpleNamespace(read=lambda: gt_lines.encode("utf-8"))
        if "BFCL_v3_simple.json" in url:
            return SimpleNamespace(read=lambda: data_lines.encode("utf-8"))
        raise URLError("missing")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    suite = datasets.load_bfcl(n=2, category="simple")
    assert suite.name == "BFCL_simple"
    assert suite.cases[0].query == "How is the weather?"
    assert suite.cases[0].ground_truth == '[{"name": "weather"}]'
    assert suite.cases[0].metadata["num_functions"] == 1

    def failing_urlopen(url: str):  # noqa: ANN001
        raise URLError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", failing_urlopen)
    with pytest.raises(ValueError, match="Failed to download BFCL dataset 'BFCL_v3_parallel.json'"):
        datasets.load_bfcl(category="parallel")


def test_tool_output_samples_custom_dataset_and_probe_generation(tmp_path) -> None:
    tool_outputs = datasets.load_tool_output_samples()
    assert tool_outputs.name == "ToolOutputSamples"
    assert len(tool_outputs.cases) >= 8
    assert tool_outputs.cases[0].ground_truth == "prompt-optimizer"

    custom_path = tmp_path / "custom.jsonl"
    custom_path.write_text(
        json.dumps(
            {"id": "case1", "context": "Context", "query": "Question", "ground_truth": "Answer"}
        )
        + "\n",
        encoding="utf-8",
    )
    custom_suite = datasets.load_custom_dataset(custom_path)
    assert custom_suite.cases[0].id == "case1"

    probes = datasets.generate_retrieval_probes(
        'Alice Smith deployed API on 2024-01-15 at 99.9% confidence for "Launch Ready" and build_id',
        n_probes=5,
    )
    assert "Alice Smith" in probes
    assert "2024-01-15" in probes
    assert "API" in probes
    assert "99.9" in probes
    assert "Launch Ready" in probes


def test_dataset_registry_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    categories = datasets.list_available_datasets()
    assert "hotpotqa" in categories["rag"]
    assert "tool_outputs" in categories["tool_use"]

    seen: list[tuple[str, dict[str, object]]] = []

    def fake_loader(*, n: int = 0, **kwargs):  # noqa: ANN003
        seen.append(("with-n", {"n": n, **kwargs}))
        return "with-n-result"

    def fixed_loader(**kwargs):  # noqa: ANN003
        seen.append(("fixed", kwargs))
        return "fixed-result"

    original_registry = dict(datasets.DATASET_REGISTRY)
    monkeypatch.setattr(
        datasets,
        "DATASET_REGISTRY",
        {
            **original_registry,
            "fake_n": {"loader": fake_loader, "category": "x", "description": "", "default_n": 3},
            "fake_fixed": {
                "loader": fixed_loader,
                "category": "x",
                "description": "",
                "default_n": None,
            },
        },
    )

    assert datasets.load_dataset_by_name("fake_n") == "with-n-result"
    assert datasets.load_dataset_by_name("fake_n", n=7, split="test") == "with-n-result"
    assert datasets.load_dataset_by_name("fake_fixed", path="x") == "fixed-result"
    assert seen == [
        ("with-n", {"n": 3}),
        ("with-n", {"n": 7, "split": "test"}),
        ("fixed", {"path": "x"}),
    ]

    with pytest.raises(ValueError, match="Unknown dataset 'missing'"):
        datasets.load_dataset_by_name("missing")


def test_dataset_loaders_cover_skip_and_limit_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_datasets(
        monkeypatch,
        {
            ("hotpotqa/hotpot_qa", "fullwiki", "validation"): [
                {
                    "context": {"title": ["Page A"], "sentences": [["Line 1"]]},
                    "question": "Q1",
                    "answer": "A1",
                },
                {
                    "context": {"title": ["Page B"], "sentences": [["Line 2"]]},
                    "question": "Q2",
                    "answer": "A2",
                },
            ],
            ("google-research-datasets/natural_questions", "default", "validation"): [
                {
                    "document": {"tokens": {"token": ["x"], "is_html": [False]}},
                    "question": {"text": ""},
                },
                {
                    "document": {"tokens": {"token": ["<b>"], "is_html": [True]}},
                    "question": {"text": "blank context"},
                },
                {
                    "document": {"tokens": {"token": ["Ada", "wrote"], "is_html": [False, False]}},
                    "question": {"text": "Who?"},
                    "annotations": {"short_answers": [[{"start_token": 1, "end_token": 1}]]},
                },
                {
                    "document": {"tokens": {"token": ["Grace"], "is_html": [False]}},
                    "question": {"text": "Ignored by limit"},
                },
            ],
            ("trivia_qa", "rc", "validation"): [
                {"question": "skip", "search_results": {"search_context": []}, "entity_pages": {}},
                {"question": "blank", "search_results": {"search_context": [""]}},
                {
                    "question": "Good 1",
                    "search_results": {"search_context": ["Context 1"]},
                    "answer": {"value": "A1"},
                },
                {
                    "question": "Good 2",
                    "search_results": {"search_context": ["Context 2"]},
                    "answer": {"value": "A2"},
                },
            ],
            ("microsoft/ms_marco", "v2.1", "validation"): [
                {"query": "skip", "passages": {"passage_text": [], "is_selected": []}},
                {
                    "query": "Find one",
                    "passages": {"passage_text": ["Doc 1"], "is_selected": [False]},
                    "answers": [],
                },
                {
                    "query": "Find two",
                    "passages": {"passage_text": ["Doc 2"], "is_selected": [True]},
                    "answers": ["A2"],
                },
            ],
            ("rajpurkar/squad_v2", None, "validation"): [
                {
                    "context": "Context 1",
                    "question": "Q1",
                    "answers": {"text": ["A1"]},
                },
                {
                    "context": "Context 2",
                    "question": "Q2",
                    "answers": {"text": ["A2"]},
                },
            ],
            ("THUDM/LongBench", "qasper", "test"): [
                {"context": "Context 1", "input": "Q1", "answers": ["A1"]},
                {"context": "Has context", "input": ""},
                {"context": "Context 2", "input": "Q2", "answers": ["A2"]},
            ],
            ("deepmind/narrativeqa", None, "test"): [
                {"document": {"summary": {"text": ""}}, "question": {"text": "skip"}},
                {"document": {"summary": {"text": "Story"}}, "question": {"text": ""}},
                {
                    "document": {"summary": {"text": "Story 1"}, "kind": "book"},
                    "question": {"text": "Q1"},
                    "answers": [{"text": "A1"}],
                },
                {
                    "document": {"summary": {"text": "Story 2"}, "kind": "movie"},
                    "question": {"text": "Q2"},
                    "answers": [{"text": "A2"}],
                },
            ],
            ("ToolBench/ToolBench", "G1", "test"): [
                {"api_list": [], "query": "skip"},
                {
                    "api_list": [
                        {
                            "api_name": "weather",
                            "required_parameters": [],
                            "optional_parameters": [],
                        }
                    ],
                    "query": "",
                },
                {
                    "api_list": [
                        {"api_name": "calc", "required_parameters": [], "optional_parameters": []}
                    ],
                    "query": "Good",
                },
            ],
            ("code_search_net", "python", "test"): [
                {
                    "func_code_string": "",
                    "whole_func_string": "",
                    "func_documentation_string": "skip",
                },
                {"whole_func_string": "def alt(): pass", "func_documentation_string": ""},
                {
                    "whole_func_string": "def good(): pass",
                    "func_documentation_string": "Good doc",
                    "func_name": "good",
                    "repository_name": "repo",
                },
                {
                    "whole_func_string": "def ignored(): pass",
                    "func_documentation_string": "Ignored by limit",
                },
            ],
            ("openai_humaneval", None, "test"): [
                {
                    "task_id": "Task/1",
                    "prompt": "def solve():",
                    "canonical_solution": "return 1",
                    "test": "assert solve() == 1",
                },
                {
                    "task_id": "Task/2",
                    "prompt": "def other():",
                    "canonical_solution": "return 2",
                    "test": "assert other() == 2",
                },
            ],
        },
    )

    assert len(datasets.load_hotpotqa(n=1).cases) == 1
    natural = datasets.load_natural_questions(n=1)
    assert len(natural.cases) == 1
    assert natural.cases[0].ground_truth is None
    assert len(datasets.load_triviaqa(n=1).cases) == 1
    msmarco = datasets.load_msmarco(n=1)
    assert len(msmarco.cases) == 1
    assert msmarco.cases[0].ground_truth is None
    assert len(datasets.load_squad(n=1).cases) == 1
    assert len(datasets.load_longbench(n=2, task="qasper").cases) == 1
    assert len(datasets.load_narrativeqa(n=1).cases) == 1
    assert len(datasets.load_toolbench(n=1, category="G1").cases) == 1
    assert len(datasets.load_codesearchnet(n=1, language="python").cases) == 1
    assert len(datasets.load_humaneval(n=1).cases) == 1


def test_load_bfcl_handles_optional_ground_truth_and_question_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_lines = "\n".join(
        [
            json.dumps(
                {
                    "id": "case-1",
                    "question": [123],
                    "function": [{"name": "weather"}],
                }
            ),
            json.dumps({"id": "skip", "function": []}),
            json.dumps(
                {
                    "id": "case-2",
                    "question": [[{"role": "user", "content": "Ignored by limit"}]],
                    "function": [{"name": "time"}],
                }
            ),
        ]
    )

    def fake_urlopen(url: str):  # noqa: ANN001
        if "possible_answer" in url:
            raise URLError("missing ground truth")
        return SimpleNamespace(read=lambda: data_lines.encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    suite = datasets.load_bfcl(n=2, category="simple")
    assert len(suite.cases) == 1
    assert suite.cases[0].query == "[123]"
    assert suite.cases[0].ground_truth is None
