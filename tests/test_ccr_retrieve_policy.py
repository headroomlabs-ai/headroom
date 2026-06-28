from __future__ import annotations

from pathlib import Path

from headroom.ccr.retrieve_policy import (
    CANONICAL_RULE,
    SKILL_GITHUB_URL,
    classify_retrieve_need,
    render_retrieve_query_description,
    render_retrieve_skill_markdown,
    render_retrieve_system_instructions,
    render_retrieve_tool_description,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_tool_description_carries_canonical_rule() -> None:
    description = render_retrieve_tool_description()
    assert CANONICAL_RULE in description
    assert "raw, original, or complete content" in description


def test_query_description_prefers_targeted_gap() -> None:
    description = render_retrieve_query_description()
    assert "targeted" in description
    assert "concrete gap" in description


def test_system_instructions_warn_against_thoroughness_only_retrieval() -> None:
    instructions = render_retrieve_system_instructions(["abc123"], "headroom_retrieve")
    assert CANONICAL_RULE in instructions
    assert "thorough" in instructions


def test_skill_file_matches_canonical_renderer() -> None:
    skill_path = REPO_ROOT / "headroom" / "skills" / "ccr-literacy" / "SKILL.md"
    assert skill_path.read_text(encoding="utf-8") == render_retrieve_skill_markdown()


def test_static_publication_surfaces_reference_skill_and_rule() -> None:
    llms_text = (REPO_ROOT / "llms.txt").read_text(encoding="utf-8")
    ccr_doc = (REPO_ROOT / "docs" / "content" / "docs" / "ccr.mdx").read_text(encoding="utf-8")
    failure_learning = (
        REPO_ROOT / "docs" / "content" / "docs" / "failure-learning.mdx"
    ).read_text(encoding="utf-8")

    assert SKILL_GITHUB_URL in llms_text
    assert CANONICAL_RULE in llms_text
    assert CANONICAL_RULE in ccr_doc
    assert CANONICAL_RULE in failure_learning


def test_classifier_flags_thoroughness_without_query_as_redundant() -> None:
    assessment = classify_retrieve_need("Be sure the summary did not miss anything.")
    assert assessment.is_redundant
    assert not assessment.should_retrieve


def test_classifier_allows_targeted_or_raw_requests() -> None:
    raw_request = classify_retrieve_need("Show me the original response text.")
    targeted_query = classify_retrieve_need("Check the auth middleware result.", query="auth")

    assert raw_request.should_retrieve
    assert not raw_request.is_redundant
    assert targeted_query.should_retrieve
    assert not targeted_query.is_redundant
