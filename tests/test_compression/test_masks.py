"""Tests for structure mask system."""

import pytest

from headroom.compression.masks import (
    EntropyScore,
    MaskSpan,
    StructureMask,
    apply_mask_to_text,
    compute_entropy_mask,
    mask_to_spans,
)


class TestStructureMask:
    """Tests for StructureMask class."""

    def test_create_mask(self):
        """Test basic mask creation."""
        tokens = ["a", "b", "c", "d"]
        mask = [True, False, False, True]

        sm = StructureMask(tokens=tokens, mask=mask)

        assert len(sm.tokens) == 4
        assert len(sm.mask) == 4
        assert sm.structural_count == 2
        assert sm.compressible_count == 2

    def test_mask_length_mismatch_raises(self):
        """Test that mismatched lengths raise ValueError."""
        tokens = ["a", "b", "c"]
        mask = [True, False]  # Wrong length

        with pytest.raises(ValueError, match="must match"):
            StructureMask(tokens=tokens, mask=mask)

    def test_preservation_ratio(self):
        """Test preservation ratio calculation."""
        tokens = list("abcdefghij")  # 10 tokens
        mask = [True, True, False, False, False, False, False, False, False, False]

        sm = StructureMask(tokens=tokens, mask=mask)

        assert sm.preservation_ratio == 0.2  # 2/10

    def test_empty_mask(self):
        """Test creating empty mask (all compressible)."""
        tokens = list("hello")
        sm = StructureMask.empty(tokens)

        assert all(not m for m in sm.mask)
        assert sm.preservation_ratio == 0.0

    def test_full_mask(self):
        """Test creating full mask (all preserved)."""
        tokens = list("hello")
        sm = StructureMask.full(tokens)

        assert all(m for m in sm.mask)
        assert sm.preservation_ratio == 1.0

    def test_get_structural_tokens(self):
        """Test extracting structural tokens."""
        tokens = ["def", " ", "foo", "(", ")", ":"]
        mask = [True, False, True, True, True, True]

        sm = StructureMask(tokens=tokens, mask=mask)
        structural = sm.get_structural_tokens()

        assert structural == ["def", "foo", "(", ")", ":"]

    def test_get_compressible_tokens(self):
        """Test extracting compressible tokens."""
        tokens = ["def", " ", "foo", "(", ")", ":"]
        mask = [True, False, True, True, True, True]

        sm = StructureMask(tokens=tokens, mask=mask)
        compressible = sm.get_compressible_tokens()

        assert compressible == [" "]

    def test_union_masks(self):
        """Test union of two masks."""
        tokens = list("abcd")
        mask1 = StructureMask(tokens=tokens, mask=[True, False, False, False])
        mask2 = StructureMask(tokens=tokens, mask=[False, False, True, False])

        result = mask1.union(mask2)

        assert result.mask == [True, False, True, False]

    def test_union_different_lengths_raises(self):
        """Test that union of different length masks raises."""
        mask1 = StructureMask(tokens=["a", "b"], mask=[True, False])
        mask2 = StructureMask(tokens=["a", "b", "c"], mask=[True, False, True])

        with pytest.raises(ValueError, match="different lengths"):
            mask1.union(mask2)

    def test_intersection_masks(self):
        """Test intersection of two masks."""
        tokens = list("abcd")
        mask1 = StructureMask(tokens=tokens, mask=[True, True, False, False])
        mask2 = StructureMask(tokens=tokens, mask=[True, False, True, False])

        result = mask1.intersection(mask2)

        assert result.mask == [True, False, False, False]


class TestMaskToSpans:
    """Tests for mask_to_spans function."""

    def test_simple_spans(self):
        """Test converting mask to spans."""
        tokens = list("abcdef")
        mask = StructureMask(
            tokens=tokens,
            mask=[True, True, True, False, False, False],
        )

        spans = mask_to_spans(mask)

        assert len(spans) == 2
        assert spans[0] == MaskSpan(start=0, end=3, is_structural=True)
        assert spans[1] == MaskSpan(start=3, end=6, is_structural=False)

    def test_alternating_spans(self):
        """Test mask with alternating regions."""
        tokens = list("abcdef")
        mask = StructureMask(
            tokens=tokens,
            mask=[True, False, True, False, True, False],
        )

        spans = mask_to_spans(mask)

        assert len(spans) == 6  # Each token is its own span

    def test_empty_mask(self):
        """Test empty mask produces no spans."""
        mask = StructureMask(tokens=[], mask=[])
        spans = mask_to_spans(mask)

        assert spans == []

    def test_span_length(self):
        """Test span length property."""
        span = MaskSpan(start=5, end=15, is_structural=True)
        assert span.length == 10


class TestEntropyScore:
    """Tests for entropy-based preservation."""

    def test_high_entropy_uuid(self):
        """Test that UUIDs have high entropy."""
        uuid = "8f14e45f-ceea-4123-8f14-e45fceea4123"
        score = EntropyScore.compute(uuid, threshold=0.8)

        assert score.value > 0.8
        assert score.should_preserve is True

    def test_low_entropy_repeated(self):
        """Test that repeated text has low entropy."""
        text = "aaaaaaaaaaaaaaaa"
        score = EntropyScore.compute(text, threshold=0.5)

        assert score.value < 0.3
        assert score.should_preserve is False

    def test_normal_text_entropy(self):
        """Test normal text entropy."""
        text = "The quick brown fox"
        score = EntropyScore.compute(text, threshold=0.85)

        # Normal diverse text has high entropy (no repetition)
        assert 0.5 < score.value <= 1.0

    def test_empty_text(self):
        """Test empty text."""
        score = EntropyScore.compute("", threshold=0.5)

        assert score.value == 0.0
        assert score.should_preserve is False

    def test_custom_threshold(self):
        """Test custom threshold."""
        text = "abc123xyz"  # Moderate entropy

        high_threshold = EntropyScore.compute(text, threshold=0.95)
        low_threshold = EntropyScore.compute(text, threshold=0.5)

        # Same value, different preservation decisions
        assert high_threshold.value == low_threshold.value
        assert (
            high_threshold.should_preserve != low_threshold.should_preserve
            or high_threshold.value >= 0.95
            or high_threshold.value < 0.5
        )


class TestComputeEntropyMask:
    """Tests for compute_entropy_mask function."""

    def test_preserves_uuids(self):
        """Test that UUIDs are preserved."""
        tokens = ["user", ":", " ", "8f14e45f-ceea-4123-8f14-e45fceea4123"]
        mask = compute_entropy_mask(tokens, threshold=0.8)

        # Only the UUID token should be preserved
        assert mask.mask[0] is False  # "user"
        assert mask.mask[1] is False  # ":"
        assert mask.mask[2] is False  # " "
        assert mask.mask[3] is True  # UUID

    def test_short_tokens_not_checked(self):
        """Test that short tokens are not checked for entropy."""
        tokens = ["ab", "cd", "ef"]
        mask = compute_entropy_mask(tokens, min_token_length=10)

        # All tokens too short to check
        assert all(not m for m in mask.mask)

    def test_metadata_contains_threshold(self):
        """Test that metadata contains threshold."""
        tokens = ["test"]
        mask = compute_entropy_mask(tokens, threshold=0.9)

        assert mask.metadata["source"] == "entropy"
        assert mask.metadata["threshold"] == 0.9


class TestApplyMaskToText:
    """Tests for apply_mask_to_text function."""

    def test_preserves_structural(self):
        """Test that structural regions are preserved."""
        text = "def foo(): pass"
        tokens = list(text)
        mask = StructureMask(
            tokens=tokens,
            # Preserve "def foo():" (first 10 chars)
            mask=[True] * 10 + [False] * 5,
        )

        def mock_compress(s: str) -> str:
            return "[C]"

        result = apply_mask_to_text(text, mask, mock_compress)

        assert result.startswith("def foo():")
        assert "[C]" in result

    def test_compresses_non_structural(self):
        """Test that non-structural regions are compressed."""
        text = "aaa bbb ccc"
        tokens = list(text)
        mask = StructureMask(
            tokens=tokens,
            mask=[True, True, True, False, False, False, False, True, True, True, True],
        )

        def mock_compress(s: str) -> str:
            return "X"

        result = apply_mask_to_text(text, mask, mock_compress)

        # "aaa" preserved, " bbb " compressed to "X", "ccc" preserved
        assert "aaa" in result
        assert "ccc" in result
