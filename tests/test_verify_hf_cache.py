"""The offline upgrade verifier must accept a minimally-seeded cache.

The guide says one ONNX candidate is enough, because the loader tries them in
order and skips absent ones. The snippet it used to show hard-coded
``onnx/kompress-fp32.onnx`` as required, so a cache seeded with only
``onnx/kompress-int8-wo.onnx`` (which is valid) printed MISSING and exited 1.

These tests exercise the real verifier the guide points at (``headroom
verify-hf-cache``), using the review's exact fixture: only
``onnx/kompress-int8-wo.onnx`` in the pinned Kompress snapshot and only
``tokenizer.json`` in the pinned ModernBERT snapshot.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from headroom.hf_cache_verify import check
from headroom.onnx_runtime import _PINNED_REVISIONS
from headroom.tokenizers.huggingface import _TOKENIZER_REVISIONS
from headroom.transforms.kompress_compressor import (
    _DEFAULT_ONNX_FILENAMES,
    _MODERNBERT_TOKENIZER_REPO,
    HF_MODEL_ID,
)

DOCS = Path(__file__).resolve().parents[1] / "docs" / "content" / "docs" / "configuration.mdx"


def _seed(cache: Path, repo: str, pin: str, filename: str) -> Path:
    path = cache / f"models--{repo.replace('/', '--')}" / "snapshots" / pin / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    return path


def _seed_kompress_minimal(cache: Path) -> None:
    """The review's fixture: one valid ONNX candidate, one tokenizer."""
    _seed(cache, HF_MODEL_ID, _PINNED_REVISIONS[HF_MODEL_ID], "onnx/kompress-int8-wo.onnx")
    _seed(
        cache,
        _MODERNBERT_TOKENIZER_REPO,
        _PINNED_REVISIONS[_MODERNBERT_TOKENIZER_REPO],
        "tokenizer.json",
    )


def _invoke(*args: str):
    from headroom.cli.main import main

    return CliRunner().invoke(main, ["verify-hf-cache", *args])


def test_one_supported_onnx_candidate_is_enough(tmp_path):
    """The regression: only `kompress-int8-wo.onnx` must verify clean."""
    _seed_kompress_minimal(tmp_path)
    assert check(tmp_path) == [], (
        "a cache holding one supported ONNX candidate must verify; requiring "
        "every candidate contradicts the guide and the loader"
    )


def test_the_documented_command_exits_zero(tmp_path):
    """The documented command, not just its helper: this is what operators run."""
    _seed_kompress_minimal(tmp_path)
    result = _invoke(str(tmp_path))
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_each_individual_candidate_satisfies_the_check(tmp_path):
    """Whatever the loader would accept, the verifier must accept."""
    for name in _DEFAULT_ONNX_FILENAMES:
        cache = tmp_path / name.replace("/", "_")
        _seed(cache, HF_MODEL_ID, _PINNED_REVISIONS[HF_MODEL_ID], name)
        _seed(
            cache,
            _MODERNBERT_TOKENIZER_REPO,
            _PINNED_REVISIONS[_MODERNBERT_TOKENIZER_REPO],
            "tokenizer.json",
        )
        assert check(cache) == [], f"{name} is a supported candidate"


def test_an_unsupported_onnx_filename_is_a_miss(tmp_path):
    _seed(tmp_path, HF_MODEL_ID, _PINNED_REVISIONS[HF_MODEL_ID], "onnx/other.onnx")
    _seed(
        tmp_path,
        _MODERNBERT_TOKENIZER_REPO,
        _PINNED_REVISIONS[_MODERNBERT_TOKENIZER_REPO],
        "tokenizer.json",
    )
    missing = check(tmp_path)
    assert missing, "an artifact the loader never looks for must not verify"
    assert any(HF_MODEL_ID in entry for entry in missing)


def test_a_missing_tokenizer_json_is_a_miss(tmp_path):
    _seed(tmp_path, HF_MODEL_ID, _PINNED_REVISIONS[HF_MODEL_ID], "onnx/kompress-int8-wo.onnx")
    assert check(tmp_path), "the Kompress tokenizer is required"


def test_the_counting_tokenizer_for_the_served_model_is_checked(tmp_path):
    """Prevents the estimation fallback the guide describes.

    Seeding Kompress says nothing about whether *counting* for the served model
    can load its tokenizer, which is the other half of the upgrade. The snapshot
    seeded here is the one the runtime resolves to, not the literal argument.
    """
    from headroom.tokenizers.huggingface import get_tokenizer_name

    repo = get_tokenizer_name("Qwen/Qwen2.5-7B")
    _seed_kompress_minimal(tmp_path)
    assert check(tmp_path, model="Qwen/Qwen2.5-7B"), (
        "a cache without the model's counting tokenizer must not verify"
    )

    _seed(tmp_path, repo, _TOKENIZER_REVISIONS[repo], "tokenizer.json")
    assert check(tmp_path, model="Qwen/Qwen2.5-7B") == []


def test_the_model_alias_resolves_to_its_repo(tmp_path):
    """The guide's example uses the alias; both spellings must work."""
    _seed_kompress_minimal(tmp_path)
    pin = _TOKENIZER_REVISIONS["Qwen/Qwen2.5-7B"]
    _seed(tmp_path, "Qwen/Qwen2.5-7B", pin, "tokenizer.json")
    assert check(tmp_path, model="qwen2.5-7b") == []


def test_the_model_check_mirrors_the_runtime_resolution(tmp_path):
    """The verifier must check the repo the *runtime* fetches, not the literal arg.

    `HuggingFaceTokenizer(model)` resolves through `get_tokenizer_name`, which
    matches family prefixes before the allowlist. So `Qwen/Qwen2.5-7B` becomes
    `Qwen/Qwen-7B`. Verifying the literal id would report OK while the runtime
    asked for a snapshot the guide never seeded — the review's exact scenario:
    verifier OK, counting still estimating.
    """
    from headroom.tokenizers.huggingface import get_tokenizer_name

    runtime_repo = get_tokenizer_name("Qwen/Qwen2.5-7B")
    assert runtime_repo == "Qwen/Qwen-7B", "the premise of this test changed"

    _seed_kompress_minimal(tmp_path)
    _seed(tmp_path, runtime_repo, _TOKENIZER_REVISIONS[runtime_repo], "tokenizer.json")

    # Passing the raw id must verify, because the runtime will fetch runtime_repo.
    assert check(tmp_path, model="Qwen/Qwen2.5-7B") == []

    # And the literal id alone must NOT satisfy it — that is the bug.
    other = tmp_path / "literal"
    _seed_kompress_minimal(other)
    _seed(other, "Qwen/Qwen2.5-7B", _TOKENIZER_REVISIONS["Qwen/Qwen2.5-7B"], "tokenizer.json")
    assert check(other, model="Qwen/Qwen2.5-7B"), (
        "seeding only the literal id leaves the runtime's snapshot absent"
    )


def test_the_documented_warm_step_seeds_what_the_runtime_requests(tmp_path, monkeypatch):
    """The guide's warm snippet and `--model` must agree with the runtime.

    The guide warms `get_tokenizer_name(MODEL)`; the verifier resolves a
    `--model` the same way. Assert the two agree for the alias the guide uses,
    so following the guide satisfies the check.
    """
    from headroom.tokenizers.huggingface import get_tokenizer_name

    model_alias = "qwen2.5-7b"
    repo = get_tokenizer_name(model_alias)

    _seed_kompress_minimal(tmp_path)
    _seed(tmp_path, repo, _TOKENIZER_REVISIONS[repo], "tokenizer.json")

    assert check(tmp_path, model=model_alias) == [], (
        "the guide seeds exactly the repo the verifier resolves the alias to"
    )


def test_the_runtime_snapshot_is_the_one_checked_not_the_literal(tmp_path, monkeypatch):
    """Execute the runtime's own lookup and assert the verifier agrees.

    Not just the resolver: exercise the loader call `HuggingFaceTokenizer` makes
    and confirm the repo it asks for is the repo the verifier requires.
    """
    import headroom.tokenizers.huggingface as hf_mod

    requested: list[str] = []

    class _Auto:
        @staticmethod
        def from_pretrained(name, **kwargs):
            requested.append(name)
            raise OSError("stop after recording the request")

    import sys
    import types

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=_Auto))
    monkeypatch.delenv("HEADROOM_HF_PIN", raising=False)
    monkeypatch.delenv("HEADROOM_HF_TOKENIZER_REVISIONS", raising=False)
    monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
    monkeypatch.setenv("HEADROOM_HF_TOKENIZER_LOAD_TIMEOUT_SECS", "5")
    hf_mod._load_tokenizer.cache_clear()
    try:
        counter = hf_mod.HuggingFaceTokenizer("Qwen/Qwen2.5-7B")
        counter.count_text("hello")  # triggers the lazy load
    finally:
        hf_mod._load_tokenizer.cache_clear()

    assert requested, "the runtime never asked the Hub for anything"
    fetched = requested[0]
    assert fetched == "Qwen/Qwen-7B", f"runtime fetched {fetched!r}"

    # Seed exactly that snapshot: the verifier must then pass for that model id.
    _seed_kompress_minimal(tmp_path)
    _seed(tmp_path, fetched, _TOKENIZER_REVISIONS[fetched], "tokenizer.json")
    assert check(tmp_path, model="Qwen/Qwen2.5-7B") == []


# ---------------------------------------------------------------------------
# Misconfiguration paths: a repo the build ships with no pin at all.
#
# These say "this build is broken", not "your cache is incomplete" — a
# deployment that floats on `main` cannot be told what to seed. They are the
# paths an operator hits after a bad downgrade or a hand-edited table, so they
# must report clearly rather than silently passing.
# ---------------------------------------------------------------------------


def test_a_shipped_repo_without_a_pin_is_reported(tmp_path, monkeypatch):
    import headroom.onnx_runtime as onnx_runtime

    monkeypatch.setattr(onnx_runtime, "_PINNED_REVISIONS", {}, raising=True)
    missing = check(tmp_path)
    assert any("no pinned revision" in entry for entry in missing), missing
    assert any(_MODERNBERT_TOKENIZER_REPO in entry for entry in missing), missing


def test_an_unpinned_kompress_model_is_reported(tmp_path, monkeypatch):
    """The model repo's own pin check, not the tokenizer's."""
    import headroom.onnx_runtime as onnx_runtime
    from headroom.transforms.kompress_compressor import _MODERNBERT_TOKENIZER_REPO as tok_repo

    # Keep the tokenizer pinned, drop only the model, so the failure is specific.
    full = dict(onnx_runtime._PINNED_REVISIONS)
    full.pop(HF_MODEL_ID, None)
    assert tok_repo in full, "the tokenizer pin must survive for this test to be specific"
    monkeypatch.setattr(onnx_runtime, "_PINNED_REVISIONS", full)

    missing = check(tmp_path)
    assert any(HF_MODEL_ID in entry and "no pinned revision" in entry for entry in missing), missing


def test_an_unpinned_counting_tokenizer_repo_is_reported(tmp_path, monkeypatch):
    """A repo the runtime resolves to but which is absent from the pin table."""
    import headroom.tokenizers.huggingface as hf_mod

    # `--model` resolves through get_tokenizer_name, so drop the repo it lands on.
    repo = hf_mod.get_tokenizer_name("Qwen/Qwen2.5-7B")
    _seed_kompress_minimal(tmp_path)

    without = dict(hf_mod._TOKENIZER_REVISIONS)
    without.pop(repo)
    monkeypatch.setattr(hf_mod, "_TOKENIZER_REVISIONS", without)

    missing = check(tmp_path, model="Qwen/Qwen2.5-7B")
    assert any("no pinned revision" in entry for entry in missing), missing


def test_the_command_defaults_to_huggingface_hubs_cache(tmp_path, monkeypatch):
    """No path argument uses the cache the process would actually download into."""
    import huggingface_hub.constants as constants

    _seed_kompress_minimal(tmp_path)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))

    result = _invoke()
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_no_kompress_checks_only_the_counting_tokenizer(tmp_path):
    from headroom.tokenizers.huggingface import get_tokenizer_name

    repo = get_tokenizer_name("Qwen/Qwen2.5-7B")
    _seed(tmp_path, repo, _TOKENIZER_REVISIONS[repo], "tokenizer.json")
    assert check(tmp_path, model="Qwen/Qwen2.5-7B", require_kompress=False) == []


def test_no_kompress_alone_is_rejected(tmp_path):
    """`--no-kompress` with no target would print OK over an empty directory."""
    result = _invoke(str(tmp_path), "--no-kompress")
    assert result.exit_code != 0, result.output
    assert "requires --model or --tokenizer-repo" in result.output
    assert "OK" not in result.output


def test_no_kompress_with_a_model_still_works(tmp_path):
    from headroom.tokenizers.huggingface import get_tokenizer_name

    repo = get_tokenizer_name("qwen2.5-7b")
    _seed(tmp_path, repo, _TOKENIZER_REVISIONS[repo], "tokenizer.json")
    result = _invoke(str(tmp_path), "--no-kompress", "--model", "qwen2.5-7b")
    assert result.exit_code == 0, result.output


def test_tokenizer_repo_is_taken_literally(tmp_path):
    """`--tokenizer-repo` names a repository outright; no alias resolution."""
    _seed(tmp_path, "Qwen/Qwen2.5-7B", _TOKENIZER_REVISIONS["Qwen/Qwen2.5-7B"], "tokenizer.json")
    # The alias for this id resolves to Qwen-7B, so a literal check is specific.
    assert check(tmp_path, tokenizer_repo="Qwen/Qwen2.5-7B", require_kompress=False) == []
    assert check(tmp_path, model="Qwen/Qwen2.5-7B", require_kompress=False), (
        "the model path resolves to Qwen-7B, which is not seeded here"
    )


def test_the_command_exits_one_when_something_is_missing(tmp_path):
    result = _invoke(str(tmp_path))
    assert result.exit_code == 1
    assert "MISSING" in result.output


def test_the_command_rejects_a_non_directory(tmp_path):
    result = _invoke(str(tmp_path / "nope"))
    assert result.exit_code == 1
    assert "Not a cache directory" in result.output


def _documented_verify_invocation() -> list[str]:
    """The verify command exactly as the guide writes it, minus the cache path.

    Parsed out of the docs rather than restated, so this test fails if the
    guide's flags drift from what the command accepts (a stale ``--model``
    spelling, a flag that was renamed, an argument that moved).
    """
    import re

    text = DOCS.read_text(encoding="utf-8")
    section = text.split("#### Offline upgrade path", 1)[1].split("\n###", 1)[0]
    match = re.search(r"^headroom verify-hf-cache .+$", section, re.MULTILINE)
    assert match, "the guide must show a `headroom verify-hf-cache` invocation"
    argv = match.group(0).split()
    # argv[0] is `headroom`; drop it and the placeholder cache path.
    assert argv[1] == "verify-hf-cache", argv
    rest = argv[2:]
    assert rest and not rest[0].startswith("-"), (
        f"expected the cache path right after the subcommand: {rest}"
    )
    return ["verify-hf-cache", *rest[1:]]


def _seed_documented_counting_tokenizer(cache: Path) -> str:
    """Seed the counting snapshot the *guide's own --model* resolves to.

    Deriving it means the test tracks the guide: if the documented model changes,
    this seeds the repo the runtime would then ask for, instead of a literal that
    silently stops matching.
    """
    from headroom.tokenizers.huggingface import get_tokenizer_name

    args = _documented_verify_invocation()
    model = None
    for i, arg in enumerate(args):
        if arg == "--model" and i + 1 < len(args):
            model = args[i + 1]
    assert model, f"the guide's invocation has no --model: {args}"
    repo = get_tokenizer_name(model)
    _seed(cache, repo, _TOKENIZER_REVISIONS[repo], "tokenizer.json")
    return repo


def test_the_documented_command_runs_verbatim(tmp_path):
    """Execute the guide's own command line against a correctly-seeded cache.

    The reviewer's fixture, with the documented flags: either supported ONNX
    candidate by itself plus the matching tokenizers. Copying the command by
    hand into each test is how the earlier bug survived — this runs what the
    guide actually says, and seeds the repo its `--model` resolves to.
    """
    _seed_kompress_minimal(tmp_path)
    _seed_documented_counting_tokenizer(tmp_path)

    from headroom.cli.main import main

    args = _documented_verify_invocation()
    result = CliRunner().invoke(main, [args[0], str(tmp_path), *args[1:]])
    assert result.exit_code == 0, f"{args} -> {result.output}"
    assert "OK" in result.output


@pytest.mark.parametrize("candidate", _DEFAULT_ONNX_FILENAMES)
def test_either_candidate_alone_satisfies_the_documented_command(tmp_path, candidate):
    """The reviewer's exact check: one supported ONNX file by itself, mirrored.

    No `.onnx` sibling of any other name, no fp32 fallback — just the one file.
    The counting tokenizer is seeded too, because the documented command passes
    ``--model``.
    """
    cache = tmp_path / "hf-cache"
    _seed(cache, HF_MODEL_ID, _PINNED_REVISIONS[HF_MODEL_ID], candidate)
    _seed(
        cache,
        _MODERNBERT_TOKENIZER_REPO,
        _PINNED_REVISIONS[_MODERNBERT_TOKENIZER_REPO],
        "tokenizer.json",
    )
    _seed(cache, "Qwen/Qwen2.5-7B", _TOKENIZER_REVISIONS["Qwen/Qwen2.5-7B"], "tokenizer.json")
    _seed_documented_counting_tokenizer(cache)
    assert sorted(p.name for p in cache.rglob("*.onnx")) == [Path(candidate).name]
    from headroom.cli.main import main

    args = _documented_verify_invocation()
    result = CliRunner().invoke(main, [args[0], str(cache), *args[1:]])
    assert result.exit_code == 0, f"only {candidate}: {result.output}"


def test_the_documented_command_fails_without_the_counting_tokenizer(tmp_path):
    """The half that prevents the estimation fallback must actually be checked."""
    _seed_kompress_minimal(tmp_path)

    from headroom.cli.main import main

    args = _documented_verify_invocation()
    result = CliRunner().invoke(main, [args[0], str(tmp_path), *args[1:]])
    assert result.exit_code == 1, result.output
    assert "MISSING" in result.output


def test_the_guide_points_at_the_real_command():
    """The docs must reference the command these tests cover, not an inline snippet."""
    section = DOCS.read_text(encoding="utf-8").split("#### Offline upgrade path", 1)[1]
    section = section.split("\n###", 1)[0]
    assert "verify-hf-cache" in section, "the guide must invoke the verifier"
    # The old inline snippet required every candidate by name.
    assert '"onnx/kompress-fp32.onnx"' not in section


@pytest.mark.parametrize("repo", [HF_MODEL_ID, _MODERNBERT_TOKENIZER_REPO])
def test_the_documented_repos_are_pinned(repo):
    """A rename or dropped pin would make the guide silently wrong."""
    assert repo in _PINNED_REVISIONS
