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
    can load its tokenizer, which is the other half of the upgrade.
    """
    _seed_kompress_minimal(tmp_path)
    assert check(tmp_path, model="Qwen/Qwen2.5-7B"), (
        "a cache without the model's counting tokenizer must not verify"
    )

    pin = _TOKENIZER_REVISIONS["Qwen/Qwen2.5-7B"]
    _seed(tmp_path, "Qwen/Qwen2.5-7B", pin, "tokenizer.json")
    assert check(tmp_path, model="Qwen/Qwen2.5-7B") == []


def test_the_model_alias_resolves_to_its_repo(tmp_path):
    """The guide's example uses the alias; both spellings must work."""
    _seed_kompress_minimal(tmp_path)
    pin = _TOKENIZER_REVISIONS["Qwen/Qwen2.5-7B"]
    _seed(tmp_path, "Qwen/Qwen2.5-7B", pin, "tokenizer.json")
    assert check(tmp_path, model="qwen2.5-7b") == []


def test_an_exact_repo_id_is_not_alias_resolved(tmp_path):
    """`Qwen/Qwen2.5-7B` must not silently mean `Qwen/Qwen-7B`.

    `get_tokenizer_name` matches family prefixes before the allowlist, so an
    exact repo id passed through it lands on a different vocabulary. An operator
    naming a repo id means that repo id.
    """
    _seed_kompress_minimal(tmp_path)
    _seed(tmp_path, "Qwen/Qwen-7B", _TOKENIZER_REVISIONS["Qwen/Qwen-7B"], "tokenizer.json")
    assert check(tmp_path, model="Qwen/Qwen2.5-7B"), (
        "a cache seeded with Qwen-7B must not satisfy a check for Qwen2.5-7B"
    )


def test_no_kompress_checks_only_the_counting_tokenizer(tmp_path):
    pin = _TOKENIZER_REVISIONS["Qwen/Qwen2.5-7B"]
    _seed(tmp_path, "Qwen/Qwen2.5-7B", pin, "tokenizer.json")
    assert check(tmp_path, model="Qwen/Qwen2.5-7B", require_kompress=False) == []


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


def test_the_documented_command_runs_verbatim(tmp_path):
    """Execute the guide's own command line against a correctly-seeded cache.

    The reviewer's fixture, with the documented flags: either supported ONNX
    candidate by itself plus the matching tokenizers. Copying the command by
    hand into each test is how the earlier bug survived — this runs what the
    guide actually says.
    """
    _seed_kompress_minimal(tmp_path)
    pin = _TOKENIZER_REVISIONS["Qwen/Qwen2.5-7B"]
    _seed(tmp_path, "Qwen/Qwen2.5-7B", pin, "tokenizer.json")

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
