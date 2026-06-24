"""Tests for HTML extraction evaluation.

These tests verify that the HTML extraction preserves information
that LLMs need to answer questions about web content.

Run with actual LLM calls:
    pytest tests/test_evals/test_html_extraction_eval.py -v -s

Skip LLM calls (just test infrastructure):
    pytest tests/test_evals/test_html_extraction_eval.py -v -k "not llm"
"""

import os

import pytest

# Skip entire module if trafilatura not installed
pytest.importorskip("trafilatura")

from headroom.evals.html_extraction import (
    HTMLEvalCase,
    HTMLEvalResult,
    HTMLEvalSuiteResult,
    HTMLExtractionEvaluator,
    get_sample_eval_cases,
)
from headroom.transforms.html_extractor import HTMLExtractor


class TestHTMLEvalInfrastructure:
    """Tests for evaluation infrastructure (no LLM calls)."""

    def test_sample_cases_available(self):
        """Verify sample evaluation cases are available."""
        cases = get_sample_eval_cases()
        assert len(cases) >= 4
        assert all(isinstance(c, HTMLEvalCase) for c in cases)

    def test_case_categories(self):
        """Verify cases cover different categories."""
        cases = get_sample_eval_cases()
        categories = {c.category for c in cases}
        assert "news" in categories
        assert "docs" in categories
        assert "blog" in categories

    def test_eval_result_properties(self):
        """Test HTMLEvalResult computed properties."""
        result = HTMLEvalResult(
            case_id="test",
            category="news",
            original_html_length=1000,
            extracted_length=300,
            compression_ratio=0.3,
            answer_from_original="Answer A",
            answer_from_extracted="Answer B",
            extracted_score=4.5,
            extracted_reasoning="Good extraction",
        )

        assert result.information_preserved is True  # score >= 4
        assert result.extraction_wins is None  # no baseline

    def test_eval_result_with_baseline(self):
        """Test HTMLEvalResult with baseline comparison."""
        result = HTMLEvalResult(
            case_id="test",
            category="news",
            original_html_length=1000,
            extracted_length=300,
            compression_ratio=0.3,
            answer_from_original="Answer A",
            answer_from_extracted="Answer B",
            answer_from_baseline="Answer C",
            extracted_score=4.5,
            extracted_reasoning="Good extraction",
            baseline_score=3.0,
            baseline_reasoning="Partial extraction",
        )

        assert result.information_preserved is True
        assert result.extraction_wins is True  # 4.5 > 3.0

    def test_suite_result_aggregation(self):
        """Test HTMLEvalSuiteResult aggregation."""
        results = [
            HTMLEvalResult(
                case_id="1",
                category="news",
                original_html_length=1000,
                extracted_length=300,
                compression_ratio=0.3,
                answer_from_original="A",
                answer_from_extracted="B",
                extracted_score=5.0,
                extracted_reasoning="Perfect",
            ),
            HTMLEvalResult(
                case_id="2",
                category="docs",
                original_html_length=800,
                extracted_length=200,
                compression_ratio=0.25,
                answer_from_original="A",
                answer_from_extracted="B",
                extracted_score=4.0,
                extracted_reasoning="Good",
            ),
            HTMLEvalResult(
                case_id="3",
                category="news",
                original_html_length=1200,
                extracted_length=400,
                compression_ratio=0.33,
                answer_from_original="A",
                answer_from_extracted="B",
                extracted_score=3.0,
                extracted_reasoning="Partial",
            ),
        ]

        suite = HTMLEvalSuiteResult(total_cases=3, results=results)

        assert suite.avg_extraction_score == 4.0  # (5+4+3)/3
        assert suite.information_preservation_rate == pytest.approx(66.67, rel=0.1)  # 2/3
        assert suite.avg_compression_ratio == pytest.approx(0.293, rel=0.1)

        summary = suite.summary()
        assert summary["total_cases"] == 3
        assert "by_category" in summary
        assert "news" in summary["by_category"]
        assert "docs" in summary["by_category"]


class TestHTMLExtractionQuality:
    """Tests that verify extraction quality without LLM calls."""

    @pytest.fixture
    def extractor(self):
        return HTMLExtractor()

    def test_extracts_article_content(self, extractor):
        """Test that article content is extracted from sample cases."""
        cases = get_sample_eval_cases()

        for case in cases:
            result = extractor.extract(case.html, url=case.url)

            # Extraction should produce non-empty content
            assert len(result.extracted) > 0

            # Should achieve significant compression
            assert result.compression_ratio < 0.7  # At least 30% reduction

    def test_removes_noise(self, extractor):
        """Test that scripts, styles, nav are removed."""
        cases = get_sample_eval_cases()

        for case in cases:
            result = extractor.extract(case.html, url=case.url)
            extracted = result.extracted.lower()

            # Should not contain JavaScript code patterns
            assert "trackconversion" not in extracted
            assert "var analytics" not in extracted
            assert "function()" not in extracted
            assert "console.log" not in extracted

            # Should not contain CSS
            assert "font-family" not in extracted
            assert "display: block" not in extracted
            assert "font-family: arial" not in extracted

    def test_preserves_key_information(self, extractor):
        """Test that key facts from questions are preserved in extraction."""
        cases = get_sample_eval_cases()

        # Check specific facts that should be preserved
        fact_checks = {
            "news_article_1": ["aria", "march 2024", "$29.99"],
            "documentation_1": ["1000", "api key", "authorization"],
            "blog_post_1": ["200", "customers", "3 years"],
            "product_page_1": ["$1,299.99", "12 hours", "1.4 kg"],
        }

        for case in cases:
            if case.id in fact_checks:
                result = extractor.extract(case.html, url=case.url)
                extracted_lower = result.extracted.lower()

                for fact in fact_checks[case.id]:
                    assert fact.lower() in extracted_lower, (
                        f"Fact '{fact}' missing from {case.id} extraction"
                    )


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
class TestHTMLExtractionWithLLM:
    """Tests that use actual LLM calls for evaluation.

    These tests verify that the extracted content allows LLMs to
    answer questions correctly.
    """

    @pytest.fixture
    def evaluator(self):
        """Create evaluator with OpenAI."""
        return HTMLExtractionEvaluator(
            answer_model="gpt-4o-mini",
            judge_model="gpt-4o-mini",  # Use mini for faster/cheaper tests
            compare_baseline=False,  # Skip baseline for speed
            provider="openai",
        )

    def test_single_case_evaluation(self, evaluator):
        """Test evaluation of a single case."""
        case = get_sample_eval_cases()[0]  # News article

        result = evaluator.evaluate_case(case)

        # Should get a valid score
        assert 1.0 <= result.extracted_score <= 5.0
        assert result.extracted_reasoning != ""

        # Should achieve compression
        assert result.compression_ratio < 0.5

        # Print for manual inspection
        print(f"\nCase: {result.case_id}")
        print(f"Score: {result.extracted_score}/5")
        print(f"Reasoning: {result.extracted_reasoning}")
        print(f"Compression: {(1 - result.compression_ratio) * 100:.1f}%")

    def test_full_suite_evaluation(self, evaluator):
        """Test evaluation of all sample cases."""
        cases = get_sample_eval_cases()

        results = evaluator.evaluate(cases)

        # Should evaluate all cases
        assert results.total_cases == len(cases)
        assert len(results.results) == len(cases)

        # Print summary
        summary = results.summary()
        print(f"\n{'=' * 50}")
        print("HTML Extraction Evaluation Results")
        print(f"{'=' * 50}")
        print(f"Total cases: {summary['total_cases']}")
        print(f"Avg extraction score: {summary['avg_extraction_score']}/5")
        print(f"Information preservation rate: {summary['information_preservation_rate']}%")
        print(f"Avg compression ratio: {summary['avg_compression_ratio']:.1%}")
        print("\nBy category:")
        for cat, stats in summary["by_category"].items():
            print(f"  {cat}: {stats['avg_score']}/5 ({stats['count']} cases)")

        # Should preserve information in most cases
        assert results.information_preservation_rate >= 75.0, (
            f"Information preservation rate too low: {results.information_preservation_rate}%"
        )


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
class TestHTMLvsBaseline:
    """Tests comparing HTMLExtractor vs Kompress baseline."""

    @pytest.fixture
    def evaluator_with_baseline(self):
        """Create evaluator that compares against baseline."""
        return HTMLExtractionEvaluator(
            answer_model="gpt-4o-mini",
            judge_model="gpt-4o-mini",
            compare_baseline=True,
            provider="openai",
        )

    @pytest.mark.skipif(True, reason="Kompress requires GPU, skip in CI")
    def test_extraction_beats_baseline(self, evaluator_with_baseline):
        """Test that HTMLExtractor outperforms Kompress on HTML."""
        cases = get_sample_eval_cases()[:2]  # Just test 2 for speed

        results = evaluator_with_baseline.evaluate(cases)

        if results.extraction_win_rate is not None:
            print(f"\nExtraction win rate: {results.extraction_win_rate}%")
            print(f"Avg extraction score: {results.avg_extraction_score}/5")
            print(f"Avg baseline score: {results.avg_baseline_score}/5")

            # HTMLExtractor should beat Kompress on HTML content
            assert results.avg_extraction_score >= results.avg_baseline_score, (
                "HTMLExtractor should perform at least as well as Kompress on HTML"
            )


class TestEvaluatorConfiguration:
    """Tests for evaluator configuration."""

    def test_lazy_loading(self):
        """Test that components are lazy loaded."""
        evaluator = HTMLExtractionEvaluator()

        # Components should not be loaded yet
        assert evaluator._extractor is None
        assert evaluator._judge_fn is None

    def test_different_providers(self):
        """Test that different providers can be configured."""
        # These should not fail (just create the evaluator)
        HTMLExtractionEvaluator(provider="openai")
        HTMLExtractionEvaluator(provider="anthropic")
        HTMLExtractionEvaluator(provider="litellm")
