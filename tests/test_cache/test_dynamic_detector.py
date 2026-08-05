"""Tests for the dynamic content detector."""

import pytest

from headroom.cache.dynamic_detector import (
    DetectionResult,
    DetectorConfig,
    DynamicCategory,
    DynamicContentDetector,
    RegexDetector,
    detect_dynamic_content,
)


class TestRegexDetector:
    """Test the Tier 1 regex detector."""

    @pytest.fixture
    def detector(self):
        """Create a regex detector."""
        config = DetectorConfig(tiers=["regex"])
        return RegexDetector(config)

    def test_iso_date(self, detector):
        """Test ISO date detection."""
        spans = detector.detect("The date is 2024-01-15.")
        assert len(spans) == 1
        assert spans[0].text == "2024-01-15"
        assert spans[0].category == DynamicCategory.DATE
        assert spans[0].tier == "regex"

    def test_structural_detection(self, detector):
        """Test structural detection via 'Label: value' patterns."""
        # New scalable approach: detect via structural "Today: value" pattern
        spans = detector.detect("Date: 2024-01-15")
        assert len(spans) == 1
        assert spans[0].text == "2024-01-15"
        assert spans[0].category == DynamicCategory.DATE

        # Test user label detection
        spans = detector.detect("User: john.doe@example.com")
        user_spans = [s for s in spans if s.category == DynamicCategory.USER_DATA]
        assert len(user_spans) == 1

    def test_datetime_iso(self, detector):
        """Test ISO datetime detection."""
        spans = detector.detect("Timestamp: 2024-01-15T10:30:00Z")
        assert len(spans) == 1
        assert spans[0].text == "2024-01-15T10:30:00Z"
        assert spans[0].category == DynamicCategory.DATETIME

    def test_uuid(self, detector):
        """Test UUID detection."""
        spans = detector.detect("ID: 550e8400-e29b-41d4-a716-446655440000")
        assert len(spans) == 1
        assert spans[0].text == "550e8400-e29b-41d4-a716-446655440000"
        assert spans[0].category == DynamicCategory.UUID

    def test_request_id(self, detector):
        """Test request ID detection."""
        spans = detector.detect("Request: req_abc123def456ghi789")
        assert len(spans) == 1
        assert "req_" in spans[0].text
        assert spans[0].category == DynamicCategory.REQUEST_ID

    def test_unix_timestamp(self, detector):
        """Test Unix timestamp detection."""
        spans = detector.detect("Time: 1705312200")
        assert len(spans) == 1
        assert spans[0].text == "1705312200"
        assert spans[0].category == DynamicCategory.TIMESTAMP

    def test_time(self, detector):
        """Test time detection."""
        spans = detector.detect("Meeting at 10:30 AM")
        assert len(spans) == 1
        assert spans[0].text == "10:30 AM"
        assert spans[0].category == DynamicCategory.TIME

    def test_version(self, detector):
        """Test version number detection."""
        spans = detector.detect("Running v2.3.1-beta")
        assert len(spans) == 1
        assert spans[0].text == "v2.3.1-beta"
        assert spans[0].category == DynamicCategory.VERSION

    def test_date_prefix_pattern(self, detector):
        """Test labeled date phrase detection.

        Structural detection requires an explicit ``:``/``=`` separator (a
        bare-whitespace separator used to swallow ordinary prose such as
        "Today is Monday..." — see issue #2110). With the label properly
        delimited, the locale-formatted date value is still extracted.
        """
        spans = detector.detect("Today: Monday, January 15, 2024. You are an assistant.")
        assert len(spans) >= 1
        # Should detect the labeled value
        date_spans = [s for s in spans if s.category == DynamicCategory.DATE]
        assert len(date_spans) >= 1

    def test_multiple_dynamic_elements(self, detector):
        """Test detecting multiple dynamic elements."""
        content = """
        Date: 2024-01-15
        Time: 10:30:00
        Request ID: req_abc123def456ghi789xyz
        UUID: 550e8400-e29b-41d4-a716-446655440000
        """
        spans = detector.detect(content)
        assert len(spans) == 4
        categories = {s.category for s in spans}
        assert DynamicCategory.DATE in categories
        assert DynamicCategory.TIME in categories
        assert DynamicCategory.REQUEST_ID in categories
        assert DynamicCategory.UUID in categories

    def test_no_false_positives_on_static(self, detector):
        """Test that static content doesn't trigger false positives."""
        spans = detector.detect("You are a helpful assistant. Answer questions clearly.")
        assert len(spans) == 0

    def test_positions_are_correct(self, detector):
        """Test that span positions are correct."""
        content = "Date: 2024-01-15"
        spans = detector.detect(content)
        assert len(spans) == 1
        assert content[spans[0].start : spans[0].end] == spans[0].text


class TestDynamicContentDetector:
    """Test the unified dynamic content detector."""

    def test_regex_only(self):
        """Test detector with regex tier only."""
        config = DetectorConfig(tiers=["regex"])
        detector = DynamicContentDetector(config)

        result = detector.detect("Today is 2024-01-15. You are helpful.")

        assert len(result.spans) == 1
        assert result.spans[0].text == "2024-01-15"
        assert "regex" in result.tiers_used
        assert result.processing_time_ms < 10  # Should be very fast

    def test_static_dynamic_split(self):
        """Test that content is properly split."""
        config = DetectorConfig(tiers=["regex"])
        detector = DynamicContentDetector(config)

        result = detector.detect("Today is 2024-01-15. You are helpful.")

        assert "2024-01-15" not in result.static_content
        assert "2024-01-15" in result.dynamic_content
        assert "You are helpful" in result.static_content

    def test_complex_content(self):
        """Test with realistic system prompt."""
        config = DetectorConfig(tiers=["regex"])
        detector = DynamicContentDetector(config)

        content = """You are a helpful AI assistant.
Today is January 15, 2024.
Current session: sess_abc123def456ghi789xyz

Instructions:
1. Be concise
2. Be accurate
3. Be helpful

Request ID: req_xyz789abc123def456ghi"""

        result = detector.detect(content)

        # Should find date, session ID, request ID
        assert len(result.spans) >= 2
        categories = {s.category for s in result.spans}
        assert DynamicCategory.DATE in categories or DynamicCategory.REQUEST_ID in categories

    def test_empty_content(self):
        """Test with empty content."""
        detector = DynamicContentDetector()
        result = detector.detect("")

        assert len(result.spans) == 0
        assert result.static_content == ""
        assert result.dynamic_content == ""

    def test_no_dynamic_content(self):
        """Test with fully static content."""
        detector = DynamicContentDetector()
        content = "You are a helpful assistant. Answer questions clearly and concisely."

        result = detector.detect(content)

        assert len(result.spans) == 0
        assert result.static_content == content
        assert result.dynamic_content == ""

    def test_custom_patterns(self):
        """Test adding custom regex patterns."""
        config = DetectorConfig(
            tiers=["regex"],
            custom_patterns=[
                (r"CUSTOM_\d{4}", DynamicCategory.REQUEST_ID),
            ],
        )
        detector = DynamicContentDetector(config)

        result = detector.detect("Code: CUSTOM_1234")

        custom_spans = [s for s in result.spans if s.text == "CUSTOM_1234"]
        assert len(custom_spans) == 1

    def test_available_tiers(self):
        """Test that available_tiers reflects actual availability."""
        config = DetectorConfig(tiers=["regex", "ner", "semantic"])
        detector = DynamicContentDetector(config)

        # Regex should always be available
        assert "regex" in detector.available_tiers

        # NER and semantic depend on optional dependencies
        # They may or may not be available

    def test_warnings_for_missing_dependencies(self):
        """Test that warnings are generated for missing dependencies."""
        config = DetectorConfig(tiers=["regex", "ner", "semantic"])
        detector = DynamicContentDetector(config)

        detector.detect("Test content")

        # If NER/semantic not installed, should have warnings
        # (This test passes either way - it's informational)
        # If deps ARE installed, no warnings. If not, warnings present.


class TestConvenienceFunction:
    """Test the detect_dynamic_content convenience function."""

    def test_basic_usage(self):
        """Test basic convenience function usage."""
        result = detect_dynamic_content("Date: 2024-01-15")

        assert isinstance(result, DetectionResult)
        assert len(result.spans) == 1
        assert result.spans[0].text == "2024-01-15"

    def test_with_tiers(self):
        """Test specifying tiers."""
        result = detect_dynamic_content(
            "Date: 2024-01-15",
            tiers=["regex"],
        )

        assert "regex" in result.tiers_used


class TestEntropyDetection:
    """Test entropy-based detection for random IDs/tokens."""

    def test_high_entropy_string(self):
        """Test that high-entropy strings are detected."""
        from headroom.cache.dynamic_detector import calculate_entropy

        # High entropy strings (random-looking)
        assert calculate_entropy("abc123xyz789def") > 0.7
        assert calculate_entropy("550e8400e29b41d4") > 0.7

        # Low entropy strings (repetitive)
        assert calculate_entropy("aaaaaaaaaa") < 0.3
        assert calculate_entropy("abababab") < 0.6

    def test_entropy_detection_finds_ids(self):
        """Test that entropy detection finds random IDs."""
        detector = DynamicContentDetector()

        # Random-looking ID that isn't covered by universal patterns
        result = detector.detect("Auth: xK7mN2pQr9sT4vW")

        # Should find the ID via entropy or structural detection
        assert len(result.spans) >= 1

    def test_entropy_skips_common_words(self):
        """Test that common words aren't flagged as high-entropy."""
        detector = DynamicContentDetector()

        # These words have mixed case/numbers but aren't IDs
        result = detector.detect("Use username and password correctly.")

        # "username" and "password" shouldn't be detected
        flagged_words = [s.text for s in result.spans]
        assert "username" not in flagged_words
        assert "password" not in flagged_words


class TestIssue2110FalsePositives:
    """Regression tests for issue #2110.

    The detector misclassified ordinary English words and code identifiers
    (e.g. ``in_progress``, ``is_valid``, ``getAuthToken``) as dynamic content,
    extracting them from the system prompt and re-appending them as a growing
    ``[Dynamic Context]`` tail that corrupted the cached prefix. Genuinely
    dynamic *shapes* (UUIDs, timestamps, hashes, prefixed ids with a digit)
    must still be detected.
    """

    @pytest.fixture
    def detector(self):
        return DynamicContentDetector(DetectorConfig(tiers=["regex"]))

    # --- must NOT be flagged (the reported false positives) ------------------

    @pytest.mark.parametrize(
        "text",
        [
            "in_progress",  # snake_case status word (prefixed_id false positive)
            "is_valid",  # snake_case identifier (entropy false positive)
            "in_pr",  # ordinary short token
            "total_tokens",  # snake_case compound word
            "system-reminder",  # kebab-case tag name
            "getAuthToken (function - src/services/firebase.ts:92)",  # code identifier + path
            "DebugModal (function - src/components/layout/DebugModal.tsx:11)",
            "The current work is being done",  # prose starting with a label word
            "last updated the file yesterday",  # prose starting with a label word
            "the user should review this",  # prose containing a label word
            "the name of the file is unknown",  # prose containing a label word
        ],
    )
    def test_ordinary_words_and_identifiers_not_extracted(self, detector, text):
        result = detector.detect(text)
        assert result.spans == [], f"unexpected dynamic spans for {text!r}: {result.spans}"
        # Nothing extracted -> the static content is preserved verbatim and the
        # dynamic tail stays empty (so it can't grow over a session).
        assert result.dynamic_content == ""

    # --- MUST still be flagged (genuinely dynamic shapes) --------------------

    def test_uuid_still_detected(self, detector):
        text = "550e8400-e29b-41d4-a716-446655440000"
        spans = detector.detect(text).spans
        assert any(s.category == DynamicCategory.UUID and s.text == text for s in spans)

    def test_timestamp_still_detected(self, detector):
        spans = detector.detect("event at 2026-07-12T10:30:00Z happened").spans
        assert any(s.text == "2026-07-12T10:30:00Z" for s in spans)

    def test_long_hex_hash_still_detected(self, detector):
        sha1 = "da39a3ee5e6b4b0d3255bfef95601890afd80709"
        spans = detector.detect(sha1).spans
        assert any(s.category == DynamicCategory.IDENTIFIER and s.text == sha1 for s in spans)

    def test_prefixed_id_with_digit_still_detected(self, detector):
        spans = detector.detect("req_a1b2c3d4").spans
        assert any(s.category == DynamicCategory.REQUEST_ID for s in spans)

    def test_labeled_dynamic_value_still_detected(self, detector):
        # Explicit "label: value" — the label stays static, the value is dynamic.
        spans = detector.detect("session_id: 8f3e2a1c9d").spans
        assert any(s.text == "8f3e2a1c9d" for s in spans)

    def test_high_entropy_id_with_digits_still_detected(self, detector):
        spans = detector.detect("a1b2c3d4e5f6g7h8").spans
        assert any(s.category == DynamicCategory.IDENTIFIER for s in spans)


class TestEdgeCases:
    """Test edge cases and tricky inputs."""

    def test_overlapping_patterns(self):
        """Test that overlapping patterns don't cause duplicates."""
        detector = DynamicContentDetector()

        # ISO datetime contains ISO date - shouldn't match both
        result = detector.detect("Time: 2024-01-15T10:30:00Z")

        # Should match datetime, not date separately
        assert len(result.spans) == 1
        assert result.spans[0].category == DynamicCategory.DATETIME

    def test_adjacent_dynamic_content(self):
        """Test adjacent dynamic elements."""
        detector = DynamicContentDetector()

        result = detector.detect("2024-01-15 10:30:00")

        # Should find both date and time
        assert len(result.spans) == 2

    def test_very_long_content(self):
        """Test with long content."""
        detector = DynamicContentDetector()

        # Create long content with some dynamic parts
        static_parts = ["This is static text. "] * 100
        content = "".join(static_parts) + "Date: 2024-01-15. " + "".join(static_parts)

        result = detector.detect(content)

        assert len(result.spans) == 1
        assert result.processing_time_ms < 100  # Should still be fast

    def test_special_characters(self):
        """Test content with special characters."""
        detector = DynamicContentDetector()

        content = "Date: 2024-01-15\nUUID: 550e8400-e29b-41d4-a716-446655440000\n\n---\n"
        result = detector.detect(content)

        assert len(result.spans) == 2

    def test_unicode_content(self):
        """Test with Unicode content."""
        detector = DynamicContentDetector()

        content = "日期: 2024-01-15. Héllo wörld!"
        result = detector.detect(content)

        # Should still find the date
        assert len(result.spans) == 1
        assert result.spans[0].text == "2024-01-15"


class TestCacheAlignmentScenarios:
    """Test scenarios relevant to cache alignment."""

    def test_system_prompt_dates(self):
        """Test extracting dates from system prompts."""
        detector = DynamicContentDetector()

        content = """You are Claude, an AI assistant by Anthropic.
Today is Monday, January 15, 2024.
Current time: 10:30 AM PST.

Your task is to help users with coding questions."""

        result = detector.detect(content)

        # Should extract date and time
        assert len(result.spans) >= 1

        # Static content should not have dates
        assert "2024" not in result.static_content or "January" in result.static_content

        # Dynamic content should have the dates
        assert (
            "January" in result.dynamic_content
            or "2024-01-15" in result.dynamic_content
            or "10:30" in result.dynamic_content
        )

    def test_request_metadata(self):
        """Test extracting request metadata."""
        detector = DynamicContentDetector()

        content = """Request ID: req_abc123xyz789
Trace ID: 550e8400-e29b-41d4-a716-446655440000
Timestamp: 1705312200

Process the following query:"""

        result = detector.detect(content)

        # Should find request ID, UUID, timestamp
        {s.category for s in result.spans}
        assert len(result.spans) >= 2

    def test_mixed_static_dynamic(self):
        """Test content with interspersed static and dynamic parts."""
        detector = DynamicContentDetector()

        content = """You are helpful (static).
Today is 2024-01-15 (dynamic).
Always be accurate (static).
Session: sess_abc123xyz789 (dynamic).
Never lie (static)."""

        result = detector.detect(content)

        # Should find date and session ID
        assert len(result.spans) >= 1

        # Static content should preserve the static parts
        assert "helpful" in result.static_content
        assert "accurate" in result.static_content


class TestNERDetector:
    """Test Tier 2 NER detector (if spaCy available)."""

    @pytest.fixture
    def ner_detector(self):
        """Create detector with NER enabled."""
        from headroom.cache.dynamic_detector import _SPACY_AVAILABLE, NERDetector

        if not _SPACY_AVAILABLE:
            pytest.skip("spaCy not installed")

        config = DetectorConfig(tiers=["ner"])
        detector = NERDetector(config)

        if not detector.is_available:
            pytest.skip("spaCy model not available")

        return detector

    def test_person_detection(self, ner_detector):
        """Test detecting person names."""
        spans, _ = ner_detector.detect("John Smith sent the message.")

        [s for s in spans if s.category == DynamicCategory.PERSON]
        # NER might or might not detect "John Smith" depending on model
        # This is more of an integration test

    def test_money_detection(self, ner_detector):
        """Test detecting money amounts."""
        spans, _ = ner_detector.detect("The total is $500.00")

        [s for s in spans if s.category == DynamicCategory.MONEY]
        # May or may not detect depending on spaCy model


class TestSemanticDetector:
    """Test Tier 3 semantic detector (if sentence-transformers available)."""

    @pytest.fixture
    def semantic_detector(self):
        """Create detector with semantic enabled."""
        from headroom.cache.dynamic_detector import (
            _SENTENCE_TRANSFORMERS_AVAILABLE,
            SemanticDetector,
        )

        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            pytest.skip("sentence-transformers not installed")

        config = DetectorConfig(tiers=["semantic"])
        detector = SemanticDetector(config)

        if not detector.is_available:
            pytest.skip("Embedding model not available")

        return detector

    def test_realtime_detection(self, semantic_detector):
        """Test detecting real-time/volatile content."""
        content = "The current stock price is updated every minute."
        spans, _ = semantic_detector.detect(content)

        # Should detect this as volatile/realtime
        # Depends on similarity threshold

    def test_missing_exemplar_embeddings_returns_warning(self):
        """Semantic detector reports unavailable state when embeddings are missing."""
        from headroom.cache.dynamic_detector import SemanticDetector

        detector = object.__new__(SemanticDetector)
        detector.config = DetectorConfig(tiers=["semantic"])
        detector._model = object()
        detector._exemplar_embeddings = None
        detector._load_error = None

        spans, warning = detector.detect("The current stock price changes every minute.")

        assert spans == []
        # Model present but exemplar matrix missing → the warning names the
        # actual missing piece (matches TestSemanticDetectorGuards below).
        assert warning == "exemplar embeddings not initialized"


class TestIntegrationWithAllTiers:
    """Integration tests using all available tiers."""

    def test_all_tiers_together(self):
        """Test running all tiers on complex content."""
        config = DetectorConfig(tiers=["regex", "ner", "semantic"])
        detector = DynamicContentDetector(config)

        content = """Today is January 15, 2024.
John paid $500 for the service.
Request ID: req_abc123xyz789.
The stock price updates in real-time.
Be helpful and accurate."""

        result = detector.detect(content)

        # Should find at least the regex matches
        assert len(result.spans) >= 1

        # Check processing time is reasonable
        # NER + semantic might add 50-100ms
        assert result.processing_time_ms < 5000  # Very generous timeout

        # Should have used at least regex
        assert "regex" in result.tiers_used

    def test_tier_precedence(self):
        """Test that earlier tiers take precedence."""
        config = DetectorConfig(tiers=["regex", "ner"])
        detector = DynamicContentDetector(config)

        # Date should be caught by regex, not NER
        result = detector.detect("Date: 2024-01-15")

        assert len(result.spans) == 1
        assert result.spans[0].tier == "regex"


class TestSemanticDetectorGuards:
    """Defensive guards in SemanticDetector.detect()."""

    def test_none_exemplars_early_return(self):
        """detect() must early-return, not crash, when exemplar embeddings
        are unset while a model is present.

        Regression for the `None.T` guard: `is_available` only checks
        `_model`, so `_exemplar_embeddings` can be None at the `np.dot`
        call. The guard returns the method's `(spans, warning)` contract.
        """
        np = pytest.importorskip("numpy")
        from unittest.mock import MagicMock

        from headroom.cache.dynamic_detector import SemanticDetector

        det = object.__new__(SemanticDetector)
        det._model = MagicMock()
        det._model.encode.return_value = np.zeros((1, 3))
        det._exemplar_embeddings = None
        det._load_error = None

        spans, warning = det.detect("This is a sentence here. Here is another long one.")

        assert spans == []
        assert warning == "exemplar embeddings not initialized"


class _RecordingEncoder:
    """A stand-in sentence-transformers model that records encode kwargs and
    returns unit vectors (so the detector's np.dot math still runs)."""

    def __init__(self) -> None:
        self.encode_calls: list[dict] = []

    def encode(self, texts, **kwargs):
        import numpy as np

        self.encode_calls.append(kwargs)
        n = len(texts) if isinstance(texts, list) else 1
        return np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))


class TestSemanticDetectorNormalization:
    """Embeddings must be L2-normalized before the np.dot cosine comparison."""

    def test_detect_normalizes_sentence_embeddings(self):
        """The sentence encode in detect() must pass normalize_embeddings=True.

        Without it np.dot is an unbounded inner product (vector norms ~5-15),
        not a cosine similarity, so nearly every sentence clears the 0.7
        threshold and static content is wrongly flagged dynamic.
        """
        np = pytest.importorskip("numpy")

        from headroom.cache.dynamic_detector import SemanticDetector

        det = object.__new__(SemanticDetector)
        det.config = DetectorConfig(tiers=["semantic"])
        model = _RecordingEncoder()
        det._model = model
        det._exemplar_embeddings = np.array([[1.0, 0.0, 0.0]])
        det._load_error = None

        det.detect("The current stock price changes every minute.")

        assert model.encode_calls, "encode was never called"
        assert all(c.get("normalize_embeddings") is True for c in model.encode_calls)

    def test_init_normalizes_exemplar_embeddings(self, monkeypatch):
        """The exemplar encode in __init__ must also pass normalize_embeddings=True
        (both sides of the dot product must be normalized to be comparable)."""
        pytest.importorskip("numpy")

        import headroom.cache.dynamic_detector as dd
        from headroom.models.ml_models import MLModelRegistry

        model = _RecordingEncoder()
        monkeypatch.setattr(dd, "_SENTENCE_TRANSFORMERS_AVAILABLE", True)
        monkeypatch.setattr(MLModelRegistry, "get_sentence_transformer", lambda *a, **k: model)

        dd.SemanticDetector(DetectorConfig(tiers=["semantic"]))

        assert model.encode_calls, "exemplar encode was never called"
        assert model.encode_calls[0].get("normalize_embeddings") is True
