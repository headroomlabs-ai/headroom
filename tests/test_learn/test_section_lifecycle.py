"""Removal invariant for merged sections — the deletion half of #2293.

``test_writer.py`` covers preservation inside a section that the new run
re-emits. The cases here cover the other transition: a section the new run
does not emit at all, which never reaches the same-section merge and so has
to be pruned on the writer's carry-forward path instead. Without that, the
last item of a category can expire and its heading still be pinned in the
file forever.

The learner-level test lives here rather than in ``test_traffic_learner.py``
because it asserts on writer behaviour; it drives the real, unmodified
``_patterns_to_recommendations`` to get there.
"""

from headroom.learn.models import Recommendation, RecommendationTarget
from headroom.learn.writer import _merge_into_file
from headroom.memory.traffic_learner import (
    ExtractedPattern,
    PatternCategory,
    _patterns_to_recommendations,
)


def _rec(section: str, content: str) -> Recommendation:
    return Recommendation(
        target=RecommendationTarget.CONTEXT_FILE,
        section=section,
        content=content,
        confidence=0.8,
        evidence_count=5,
    )


def _block(*sections: tuple[str, str]) -> str:
    parts = ["<!-- headroom:learn:start -->", "## Headroom Learned Patterns", ""]
    for heading, body in sections:
        parts += [f"### {heading}", body, ""]
    parts.append("<!-- headroom:learn:end -->")
    return "\n".join(parts) + "\n"


class TestCarriedSectionLifecycle:
    """The new run's lifecycle signal also governs the sections it omits."""

    def test_absent_tracked_section_is_removed_once_its_items_expire(self, tmp_path):
        """A heading outlives its items unless the carry path is pruned too.

        When a category loses its last pattern the learner stops emitting a
        recommendation for that heading at all, so the same-section merge
        never runs on it. The signal published by the rest of the run has to
        reach the carry path, or the section is pinned forever.
        """
        context_file = tmp_path / "AGENTS.md"
        context_file.write_text(
            _block(
                (
                    "Learned: architecture",
                    "- Handlers live under proxy/handlers <!-- headroom:pattern-id:handlers -->",
                ),
                (
                    "Learned: preference",
                    "- Keep local reviews <!-- headroom:pattern-id:reviews -->",
                ),
            ),
            encoding="utf-8",
        )
        recommendation = _rec(
            "Learned: preference",
            "- Keep local reviews <!-- headroom:pattern-id:reviews -->",
        )
        recommendation.preserve_prior_items = True
        recommendation.active_item_ids = frozenset({"reviews"})

        final = _merge_into_file(context_file, [recommendation])

        assert "### Learned: architecture" not in final
        assert "Handlers live under proxy/handlers" not in final
        assert final.count("Keep local reviews") == 1

    def test_absent_tracked_section_keeps_its_still_active_items(self, tmp_path):
        """Pruning a carried section is per item, not all-or-nothing.

        A category can be missing from a render because of batching while
        still holding live items, so only the ids the run no longer claims
        may go.
        """
        context_file = tmp_path / "AGENTS.md"
        context_file.write_text(
            _block(
                (
                    "Learned: architecture",
                    "- Handlers live under proxy/handlers <!-- headroom:pattern-id:handlers -->\n"
                    "- Build against the vendored SDK <!-- headroom:pattern-id:vendored -->",
                ),
                (
                    "Learned: preference",
                    "- Keep local reviews <!-- headroom:pattern-id:reviews -->",
                ),
            ),
            encoding="utf-8",
        )
        recommendation = _rec(
            "Learned: preference",
            "- Keep local reviews <!-- headroom:pattern-id:reviews -->",
        )
        recommendation.preserve_prior_items = True
        recommendation.active_item_ids = frozenset({"reviews", "handlers"})

        final = _merge_into_file(context_file, [recommendation])

        assert "### Learned: architecture" in final
        assert "Handlers live under proxy/handlers" in final
        assert "Build against the vendored SDK" not in final

    def test_absent_untracked_sections_are_carried_untouched(self, tmp_path):
        """Only id-tagged sections are ours to delete.

        Hand-written headings and prose bodies carry no lifecycle signal,
        and — unlike the same-section merge, where a still-active legacy
        item is re-emitted with an id by the same run — nothing on this path
        would bring them back, so they must survive the prune.
        """
        context_file = tmp_path / "AGENTS.md"
        context_file.write_text(
            _block(
                ("Hand-written notes", "- Deploy from the release branch"),
                ("Project shape", "Prose, deliberately not a markdown list."),
                (
                    "Learned: preference",
                    "- Keep local reviews <!-- headroom:pattern-id:reviews -->",
                ),
            ),
            encoding="utf-8",
        )
        recommendation = _rec(
            "Learned: preference",
            "- Keep local reviews <!-- headroom:pattern-id:reviews -->",
        )
        recommendation.preserve_prior_items = True
        recommendation.active_item_ids = frozenset({"reviews"})

        final = _merge_into_file(context_file, [recommendation])

        assert "Deploy from the release branch" in final
        assert "Prose, deliberately not a markdown list." in final

    def test_absent_tracked_section_survives_a_run_without_a_lifecycle_signal(self, tmp_path):
        """No signal, no deletion.

        A run whose recommendations carry no ``active_item_ids`` cannot
        speak to what is still alive, so the carry path stays conservative
        rather than reading silence as expiry.
        """
        context_file = tmp_path / "AGENTS.md"
        context_file.write_text(
            _block(
                (
                    "Learned: architecture",
                    "- Handlers live under proxy/handlers <!-- headroom:pattern-id:handlers -->",
                ),
                ("Learned: error recovery", "- Search before reading a guessed path"),
            ),
            encoding="utf-8",
        )
        recommendation = _rec(
            "Learned: error recovery",
            "- Search before reading a guessed path",
        )
        assert recommendation.active_item_ids is None

        final = _merge_into_file(context_file, [recommendation])

        assert "### Learned: architecture" in final
        assert "Handlers live under proxy/handlers" in final


class TestTrafficLearnerCategoryLifecycle:
    """End-to-end through the real learner rendering."""

    def test_category_losing_its_last_pattern_drops_its_heading(self, tmp_path):
        """An emptied category leaves the file entirely.

        This is the transition the per-item merge cannot see. Once
        ``architecture`` holds nothing, ``_patterns_to_recommendations``
        emits no recommendation for that heading, so the removal has to
        happen on the writer's carry-forward path instead. The existing
        end-to-end case keeps a second item in the same category, so it
        never reaches this transition.
        """
        keep = ExtractedPattern(
            category=PatternCategory.PREFERENCE,
            content="User prefers terse output",
            importance=0.8,
            evidence_count=3,
        )
        sole_architecture_pattern = ExtractedPattern(
            category=PatternCategory.ARCHITECTURE,
            content="Handlers live under headroom/proxy/handlers",
            importance=0.8,
            evidence_count=3,
        )
        context_file = tmp_path / "AGENTS.md"
        context_file.write_text(
            _merge_into_file(
                context_file,
                _patterns_to_recommendations([keep, sole_architecture_pattern]),
            ),
            encoding="utf-8",
        )
        assert "### Learned: architecture" in context_file.read_text()

        # Next render: architecture has no live pattern left, so the whole
        # category stops being rendered rather than rendering fewer bullets.
        final = _merge_into_file(context_file, _patterns_to_recommendations([keep]))

        assert "User prefers terse output" in final
        assert "### Learned: architecture" not in final
        assert "Handlers live under headroom/proxy/handlers" not in final
