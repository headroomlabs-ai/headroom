"""Tests using OSS benchmarks for HTML extraction evaluation.

These tests use established open-source benchmarks to verify that
HTMLExtractor does not lose accuracy:

1. Scrapinghub Article Extraction Benchmark
   - Measures extraction quality (F1 score)
   - Baseline: trafilatura achieves 0.958 F1

2. SQuAD/HotpotQA for QA accuracy preservation
   - Measures whether extraction preserves answer accuracy

Run extraction benchmark only (no API calls):
    pytest tests/test_evals/test_html_oss_benchmarks.py -k "extraction" -v

Run full suite with LLM (requires OPENAI_API_KEY):
    pytest tests/test_evals/test_html_oss_benchmarks.py -v -s
"""

import os

import pytest

# Skip entire module if trafilatura not installed
pytest.importorskip("trafilatura")


class TestExtractionBenchmark:
    """Tests using Scrapinghub Article Extraction Benchmark.

    This is the gold standard for article extraction evaluation.
    No LLM calls required - just measures F1 against ground truth.
    """

    @pytest.fixture
    def extractor(self):
        from headroom.transforms.html_extractor import HTMLExtractor

        return HTMLExtractor()

    def test_benchmark_loads(self):
        """Verify we can load the benchmark dataset."""
        pytest.importorskip("datasets")
        from datasets import load_dataset

        dataset = load_dataset("allenai/scrapinghub-article-extraction-benchmark")
        assert "train" in dataset
        assert len(dataset["train"]) > 0

        # Check expected fields
        sample = dataset["train"][0]
        assert "html" in sample
        assert "articleBody" in sample

    def test_extraction_f1_quick(self, extractor):
        """Quick test: evaluate on 10 samples."""
        pytest.importorskip("datasets")
        from headroom.evals.html_oss_benchmarks import evaluate_scrapinghub_benchmark

        result = evaluate_scrapinghub_benchmark(
            extractor=extractor,
            max_samples=10,
        )

        # Should get reasonable F1 (> 0.8)
        assert result.avg_f1 > 0.8, f"F1 too low: {result.avg_f1}"
        assert result.avg_precision > 0.7
        assert result.avg_recall > 0.7

        # Print results
        print("\nQuick Extraction Benchmark (10 samples):")
        print(f"  Precision: {result.avg_precision:.3f}")
        print(f"  Recall:    {result.avg_recall:.3f}")
        print(f"  F1:        {result.avg_f1:.3f}")
        print(f"  Baseline:  {result.baseline_f1:.3f}")

    def test_extraction_f1_medium(self, extractor):
        """Medium test: evaluate on 50 samples."""
        pytest.importorskip("datasets")
        from headroom.evals.html_oss_benchmarks import evaluate_scrapinghub_benchmark

        result = evaluate_scrapinghub_benchmark(
            extractor=extractor,
            max_samples=50,
        )

        # Should approach baseline performance (0.958)
        # Allow some margin since our extractor may differ slightly
        assert result.avg_f1 > 0.85, f"F1 too low: {result.avg_f1}"

        print("\nMedium Extraction Benchmark (50 samples):")
        print(f"  Precision: {result.avg_precision:.3f}")
        print(f"  Recall:    {result.avg_recall:.3f}")
        print(f"  F1:        {result.avg_f1:.3f}")
        print(f"  Baseline:  {result.baseline_f1:.3f}")
        print(f"  Matches baseline: {result.matches_baseline}")

    @pytest.mark.slow
    def test_extraction_f1_full(self, extractor):
        """Full test: evaluate on all 181 samples."""
        pytest.importorskip("datasets")
        from headroom.evals.html_oss_benchmarks import evaluate_scrapinghub_benchmark

        result = evaluate_scrapinghub_benchmark(
            extractor=extractor,
            max_samples=None,  # All samples
        )

        # Should match or exceed baseline
        assert result.avg_f1 > 0.90, f"F1 too low: {result.avg_f1}"

        print(f"\nFull Extraction Benchmark ({result.total_samples} samples):")
        print(f"  Precision: {result.avg_precision:.3f}")
        print(f"  Recall:    {result.avg_recall:.3f}")
        print(f"  F1:        {result.avg_f1:.3f}")
        print(f"  Baseline:  {result.baseline_f1:.3f}")
        print(f"  Matches baseline: {result.matches_baseline}")
        print(f"  Beats baseline:   {result.beats_baseline}")

    def test_compression_achieved(self, extractor):
        """Verify we achieve meaningful compression."""
        pytest.importorskip("datasets")
        from headroom.evals.html_oss_benchmarks import evaluate_scrapinghub_benchmark

        result = evaluate_scrapinghub_benchmark(
            extractor=extractor,
            max_samples=20,
        )

        # Should achieve significant compression (ratio < 0.5 = 50%+ reduction)
        assert result.avg_compression_ratio < 0.5, (
            f"Compression ratio too high: {result.avg_compression_ratio}"
        )

        print("\nCompression Results:")
        print(f"  Avg compression ratio: {result.avg_compression_ratio:.3f}")
        print(f"  Avg reduction: {(1 - result.avg_compression_ratio) * 100:.1f}%")


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_f1_computation(self):
        from headroom.evals.html_oss_benchmarks import compute_f1

        # Perfect match
        p, r, f1 = compute_f1("hello world", "hello world")
        assert f1 == 1.0

        # Partial match
        p, r, f1 = compute_f1("hello world foo", "hello world bar")
        assert 0.5 < f1 < 1.0

        # No match
        p, r, f1 = compute_f1("foo bar", "hello world")
        assert f1 == 0.0

    def test_exact_match(self):
        from headroom.evals.html_oss_benchmarks import compute_exact_match

        assert compute_exact_match("hello world", "Hello World") is True
        assert compute_exact_match("hello", "hello world") is False


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
class TestQAAccuracyPreservation:
    """Tests that verify QA accuracy is preserved after extraction.

    These tests require an LLM to answer questions, then compare
    accuracy on original HTML vs extracted content.
    """

    @pytest.fixture
    def answer_fn(self):
        """Create an answer function using OpenAI."""
        from openai import OpenAI

        client = OpenAI()

        def answer(context: str, question: str) -> str:
            prompt = f"""Based on the following content, answer the question concisely.

Content:
{context[:4000]}  # Limit context size

Question: {question}

Answer:"""
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            return response.choices[0].message.content or ""

        return answer

    def test_qa_accuracy_squad_quick(self, answer_fn):
        """Quick QA accuracy test on 10 SQuAD questions."""
        pytest.importorskip("datasets")
        from headroom.evals.html_oss_benchmarks import evaluate_qa_accuracy_preservation

        result = evaluate_qa_accuracy_preservation(
            answer_fn=answer_fn,
            max_questions=10,
            dataset_name="squad",
        )

        # Accuracy should be preserved (within 5%)
        assert result.accuracy_preserved, (
            f"Accuracy not preserved: original={result.accuracy_original_html:.3f}, "
            f"extracted={result.accuracy_extracted:.3f}"
        )

        print("\nQA Accuracy (10 questions):")
        print(f"  Original HTML: {result.accuracy_original_html:.3f}")
        print(f"  Extracted:     {result.accuracy_extracted:.3f}")
        print(f"  Preserved:     {result.accuracy_preserved}")

    def test_qa_accuracy_squad_medium(self, answer_fn):
        """Medium QA accuracy test on 30 SQuAD questions."""
        pytest.importorskip("datasets")
        from headroom.evals.html_oss_benchmarks import evaluate_qa_accuracy_preservation

        result = evaluate_qa_accuracy_preservation(
            answer_fn=answer_fn,
            max_questions=30,
            dataset_name="squad",
        )

        assert result.accuracy_preserved

        print("\nQA Accuracy (30 questions):")
        print(f"  Original HTML: {result.accuracy_original_html:.3f}")
        print(f"  Extracted:     {result.accuracy_extracted:.3f}")
        print(f"  Delta:         {result.accuracy_extracted - result.accuracy_original_html:+.3f}")


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
class TestFullBenchmarkSuite:
    """Full benchmark suite combining extraction quality and QA accuracy."""

    @pytest.fixture
    def answer_fn(self):
        from openai import OpenAI

        client = OpenAI()

        def answer(context: str, question: str) -> str:
            prompt = f"""Answer the question based on the content.

Content: {context[:4000]}

Question: {question}

Answer concisely:"""
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            return response.choices[0].message.content or ""

        return answer

    def test_full_suite(self, answer_fn):
        """Run the complete benchmark suite."""
        pytest.importorskip("datasets")
        from headroom.evals.html_oss_benchmarks import run_full_benchmark_suite

        result = run_full_benchmark_suite(
            answer_fn=answer_fn,
            extraction_samples=30,
            qa_questions=20,
        )

        # Print comprehensive results
        print("\n" + "=" * 60)
        print("FULL BENCHMARK SUITE RESULTS")
        print("=" * 60)

        summary = result.summary()

        if result.extraction_result:
            ext = summary["extraction"]
            print("\n📊 Extraction Benchmark:")
            print(f"   Samples:   {ext['total_samples']}")
            print(f"   Precision: {ext['avg_precision']:.3f}")
            print(f"   Recall:    {ext['avg_recall']:.3f}")
            print(f"   F1:        {ext['avg_f1']:.3f} (baseline: {ext['baseline_f1']:.3f})")
            print(f"   Compression: {(1 - ext['avg_compression_ratio']) * 100:.1f}% reduction")

        if result.qa_result:
            qa = summary["qa_accuracy"]
            print("\n📝 QA Accuracy Preservation:")
            print(f"   Questions: {qa['total_questions']}")
            print(f"   Original:  {qa['accuracy_original_html']:.3f}")
            print(f"   Extracted: {qa['accuracy_extracted']:.3f}")
            print(f"   Delta:     {qa['accuracy_delta']:+.3f}")
            print(f"   Preserved: {'✅' if qa['accuracy_preserved'] else '❌'}")

        print(f"\n{'=' * 60}")
        print(f"ALL BENCHMARKS PASSED: {'✅' if summary['all_passed'] else '❌'}")
        print(f"{'=' * 60}\n")

        # Assert all passed
        assert result.all_passed, "Not all benchmarks passed"


class TestBenchmarkInfrastructure:
    """Tests for benchmark infrastructure without running full evals."""

    def test_result_classes(self):
        """Test result dataclasses work correctly."""
        from headroom.evals.html_oss_benchmarks import (
            ExtractionBenchmarkResult,
            QAAccuracyResult,
        )

        ext = ExtractionBenchmarkResult(
            total_samples=100,
            avg_precision=0.95,
            avg_recall=0.92,
            avg_f1=0.935,
            avg_compression_ratio=0.35,
        )
        assert ext.matches_baseline is False  # 0.935 not within 0.02 of 0.958
        assert ext.beats_baseline is False

        qa = QAAccuracyResult(
            total_questions=50,
            accuracy_original_html=0.85,
            accuracy_extracted=0.87,
            accuracy_preserved=True,
            avg_f1_original=0.85,
            avg_f1_extracted=0.87,
            exact_match_original=0.60,
            exact_match_extracted=0.62,
        )
        assert qa.accuracy_preserved is True

    def test_suite_all_passed(self):
        """Test suite pass/fail logic."""
        from headroom.evals.html_oss_benchmarks import (
            ExtractionBenchmarkResult,
            HTMLExtractorBenchmarkSuite,
            QAAccuracyResult,
        )

        # Both pass
        suite = HTMLExtractorBenchmarkSuite(
            extraction_result=ExtractionBenchmarkResult(
                total_samples=100,
                avg_precision=0.95,
                avg_recall=0.92,
                avg_f1=0.935,
                avg_compression_ratio=0.35,
            ),
            qa_result=QAAccuracyResult(
                total_questions=50,
                accuracy_original_html=0.85,
                accuracy_extracted=0.87,
                accuracy_preserved=True,
                avg_f1_original=0.85,
                avg_f1_extracted=0.87,
                exact_match_original=0.60,
                exact_match_extracted=0.62,
            ),
        )
        assert suite.all_passed is True

        # Extraction fails (F1 too low)
        suite_fail = HTMLExtractorBenchmarkSuite(
            extraction_result=ExtractionBenchmarkResult(
                total_samples=100,
                avg_precision=0.7,
                avg_recall=0.7,
                avg_f1=0.7,  # Below 0.90 threshold
                avg_compression_ratio=0.35,
            ),
        )
        assert suite_fail.all_passed is False
