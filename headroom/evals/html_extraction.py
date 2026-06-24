"""Evaluation framework for HTML content extraction.

This module evaluates whether HTMLExtractor preserves the information
that LLMs need to answer questions about web content. We compare:
1. LLM answers from original HTML
2. LLM answers from HTMLExtractor output

Uses LLM-as-judge to score answer quality on a 1-5 scale.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from headroom.transforms.html_extractor import HTMLExtractor

logger = logging.getLogger(__name__)


# HTML Extraction Judge Prompt - optimized for content extraction evaluation
HTML_JUDGE_PROMPT = """You are evaluating an HTML content extraction system.

The system extracts main content from web pages, removing scripts, styles,
navigation, ads, and other noise while preserving the actual article content.

Given a question about a web page, the ground truth answer (from original HTML),
and the system's answer (from extracted content), score the extraction quality:

5 = Perfect: The extracted content answer is semantically equivalent
4 = Mostly correct: Minor details missing but main information preserved
3 = Partially correct: Some key information present, some missing
2 = Mostly incorrect: Significant information loss
1 = Completely wrong: Critical content was removed during extraction

Question: {question}

Answer from Original HTML: {ground_truth}

Answer from Extracted Content: {prediction}

Consider:
- Is the factual information preserved?
- Are key details (names, dates, numbers) maintained?
- Is the answer still complete and useful?

Format your response EXACTLY as:
Reasoning: <your reasoning about information preservation>
Score: <number 1-5>"""


@dataclass
class HTMLEvalCase:
    """A single HTML extraction evaluation case."""

    id: str
    html: str  # Original HTML content
    url: str | None  # Source URL for context
    question: str  # Question about the content
    ground_truth: str  # Expected answer from the original
    category: str = "general"  # news, docs, blog, product, etc.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HTMLEvalResult:
    """Result of a single HTML extraction evaluation."""

    case_id: str
    category: str

    # Content sizes
    original_html_length: int
    extracted_length: int
    compression_ratio: float

    # Answers from different methods
    answer_from_original: str
    answer_from_extracted: str
    answer_from_baseline: str | None = None  # Baseline comparison

    # Judge scores (1-5 scale)
    extracted_score: float = 0.0
    extracted_reasoning: str = ""
    baseline_score: float | None = None
    baseline_reasoning: str | None = None

    # Derived metrics
    @property
    def information_preserved(self) -> bool:
        """True if extraction score >= 4 (mostly correct or better)."""
        return self.extracted_score >= 4.0

    @property
    def extraction_wins(self) -> bool | None:
        """True if extraction beats baseline, None if no baseline."""
        if self.baseline_score is None:
            return None
        return self.extracted_score > self.baseline_score


@dataclass
class HTMLEvalSuiteResult:
    """Aggregated results from HTML extraction evaluation suite."""

    total_cases: int
    results: list[HTMLEvalResult]

    @property
    def avg_extraction_score(self) -> float:
        """Average score for HTMLExtractor (1-5 scale)."""
        if not self.results:
            return 0.0
        return sum(r.extracted_score for r in self.results) / len(self.results)

    @property
    def avg_baseline_score(self) -> float | None:
        """Average score for baseline, None if no baseline tested."""
        baseline_results = [r for r in self.results if r.baseline_score is not None]
        if not baseline_results:
            return None
        return sum(
            r.baseline_score for r in baseline_results if r.baseline_score is not None
        ) / len(baseline_results)

    @property
    def information_preservation_rate(self) -> float:
        """Percentage of cases where extraction score >= 4."""
        if not self.results:
            return 0.0
        preserved = sum(1 for r in self.results if r.information_preserved)
        return preserved / len(self.results) * 100

    @property
    def extraction_win_rate(self) -> float | None:
        """Percentage of cases where extraction beats baseline."""
        comparison_results = [r for r in self.results if r.baseline_score is not None]
        if not comparison_results:
            return None
        wins = sum(1 for r in comparison_results if r.extraction_wins)
        return wins / len(comparison_results) * 100

    @property
    def avg_compression_ratio(self) -> float:
        """Average compression ratio achieved."""
        if not self.results:
            return 0.0
        return sum(r.compression_ratio for r in self.results) / len(self.results)

    def summary(self) -> dict[str, Any]:
        """Return summary statistics."""
        return {
            "total_cases": self.total_cases,
            "avg_extraction_score": round(self.avg_extraction_score, 2),
            "avg_baseline_score": (
                round(self.avg_baseline_score, 2) if self.avg_baseline_score else None
            ),
            "information_preservation_rate": round(self.information_preservation_rate, 1),
            "extraction_win_rate": (
                round(self.extraction_win_rate, 1) if self.extraction_win_rate else None
            ),
            "avg_compression_ratio": round(self.avg_compression_ratio, 3),
            "by_category": self._results_by_category(),
        }

    def _results_by_category(self) -> dict[str, dict[str, Any]]:
        """Break down results by category."""
        categories: dict[str, list[HTMLEvalResult]] = {}
        for r in self.results:
            if r.category not in categories:
                categories[r.category] = []
            categories[r.category].append(r)

        return {
            cat: {
                "count": len(results),
                "avg_score": round(sum(r.extracted_score for r in results) / len(results), 2),
                "preservation_rate": round(
                    sum(1 for r in results if r.information_preserved) / len(results) * 100, 1
                ),
            }
            for cat, results in categories.items()
        }


class HTMLExtractionEvaluator:
    """Evaluates HTML content extraction quality using LLM-as-judge.

    Example:
        evaluator = HTMLExtractionEvaluator(
            answer_model="gpt-4o-mini",
            judge_model="gpt-4o",
        )
        results = evaluator.evaluate(eval_cases)
        print(f"Preservation rate: {results.information_preservation_rate}%")
    """

    def __init__(
        self,
        answer_model: str = "gpt-4o-mini",
        judge_model: str = "gpt-4o",
        compare_baseline: bool = True,
        provider: str = "openai",
    ):
        """Initialize the evaluator.

        Args:
            answer_model: Model for generating answers from content.
            judge_model: Model for judging answer quality.
            compare_baseline: Whether to also test Kompress baseline.
            provider: API provider ("openai", "anthropic", "litellm").
        """
        self.answer_model = answer_model
        self.judge_model = judge_model
        self.compare_baseline = compare_baseline
        self.provider = provider

        # Lazy-loaded components
        self._extractor: HTMLExtractor | None = None
        self._kompress: Any = None
        self._judge_fn: Callable[[str, str, str], tuple[float, str]] | None = None
        self._answer_fn: Any = None

    @property
    def extractor(self) -> HTMLExtractor:
        """Lazy-load HTMLExtractor."""
        if self._extractor is None:
            from headroom.transforms.html_extractor import HTMLExtractor

            self._extractor = HTMLExtractor()
        return self._extractor

    @property
    def kompress(self) -> Any:
        """Lazy-load Kompress compressor for baseline."""
        if self._kompress is None and self.compare_baseline:
            try:
                from headroom.transforms.kompress_compressor import KompressCompressor

                self._kompress = KompressCompressor()
            except ImportError:
                logger.warning("Kompress not available for baseline comparison")
        return self._kompress

    @property
    def judge_fn(self) -> Callable[[str, str, str], tuple[float, str]]:
        """Lazy-load judge function."""
        if self._judge_fn is None:
            self._judge_fn = self._create_judge()
        assert self._judge_fn is not None  # Always set by _create_judge or exception raised
        return self._judge_fn

    def _create_judge(self) -> Callable[[str, str, str], tuple[float, str]]:
        """Create the LLM judge function."""
        if self.provider == "openai":
            try:
                from openai import OpenAI

                client = OpenAI()

                def judge(question: str, ground_truth: str, prediction: str) -> tuple[float, str]:
                    prompt = HTML_JUDGE_PROMPT.format(
                        question=question,
                        ground_truth=ground_truth,
                        prediction=prediction,
                    )
                    response = client.chat.completions.create(
                        model=self.judge_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=200,
                    )
                    return self._parse_judge_response(response.choices[0].message.content or "")

                return judge
            except ImportError:
                raise ImportError(
                    "OpenAI package required. Install with: pip install openai"
                ) from None

        elif self.provider == "anthropic":
            try:
                import anthropic

                anthropic_client = anthropic.Anthropic()

                def judge(question: str, ground_truth: str, prediction: str) -> tuple[float, str]:
                    prompt = HTML_JUDGE_PROMPT.format(
                        question=question,
                        ground_truth=ground_truth,
                        prediction=prediction,
                    )
                    anthropic_response = anthropic_client.messages.create(
                        model=self.judge_model,
                        max_tokens=200,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text = (
                        getattr(anthropic_response.content[0], "text", "")
                        if anthropic_response.content
                        else ""
                    )
                    return self._parse_judge_response(text)

                return judge
            except ImportError:
                raise ImportError(
                    "Anthropic package required. Install with: pip install anthropic"
                ) from None

        else:
            try:
                import litellm

                def judge(question: str, ground_truth: str, prediction: str) -> tuple[float, str]:
                    prompt = HTML_JUDGE_PROMPT.format(
                        question=question,
                        ground_truth=ground_truth,
                        prediction=prediction,
                    )
                    response = litellm.completion(
                        model=self.judge_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=200,
                    )
                    return self._parse_judge_response(response.choices[0].message.content or "")

                return judge
            except ImportError:
                raise ImportError(
                    "LiteLLM package required. Install with: pip install litellm"
                ) from None

    def _parse_judge_response(self, text: str) -> tuple[float, str]:
        """Parse judge response to extract score and reasoning."""
        import re

        reasoning = ""
        score = 3.0  # Default

        for line in text.strip().split("\n"):
            line = line.strip()
            if line.lower().startswith("reasoning:"):
                reasoning = line[len("reasoning:") :].strip()
            elif line.lower().startswith("score:"):
                match = re.search(r"(\d+(?:\.\d+)?)", line)
                if match:
                    score = max(1.0, min(5.0, float(match.group(1))))

        return score, reasoning or text.strip()

    def _get_answer(self, content: str, question: str) -> str:
        """Get LLM answer for a question given content."""
        prompt = f"""Based on the following content, answer the question.

Content:
{content}

Question: {question}

Answer concisely and factually based only on the content provided."""

        if self.provider == "openai":
            from openai import OpenAI

            openai_client = OpenAI()
            openai_response = openai_client.chat.completions.create(
                model=self.answer_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500,
            )
            return openai_response.choices[0].message.content or ""

        elif self.provider == "anthropic":
            import anthropic

            anthropic_client = anthropic.Anthropic()
            anthropic_response = anthropic_client.messages.create(
                model=self.answer_model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            return (
                getattr(anthropic_response.content[0], "text", "")
                if anthropic_response.content
                else ""
            )

        else:
            import litellm

            litellm_response = litellm.completion(
                model=self.answer_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500,
            )
            return litellm_response.choices[0].message.content or ""

    def evaluate_case(self, case: HTMLEvalCase) -> HTMLEvalResult:
        """Evaluate a single HTML extraction case.

        Args:
            case: The evaluation case with HTML, question, and ground truth.

        Returns:
            HTMLEvalResult with scores and metrics.
        """
        # Extract content
        extraction_result = self.extractor.extract(case.html, url=case.url)
        extracted_content = extraction_result.extracted

        # Get answer from extracted content
        answer_from_extracted = self._get_answer(extracted_content, case.question)

        # Get answer from original HTML (for comparison)
        answer_from_original = self._get_answer(case.html, case.question)

        # Judge the extraction quality
        extracted_score, extracted_reasoning = self.judge_fn(
            case.question,
            case.ground_truth,
            answer_from_extracted,
        )

        # Optionally compare with Kompress baseline
        baseline_answer = None
        baseline_score = None
        baseline_reasoning = None

        if self.compare_baseline and self.kompress:
            try:
                baseline_result = self.kompress.compress(case.html)
                baseline_content = baseline_result.compressed
                baseline_answer = self._get_answer(baseline_content, case.question)
                baseline_score, baseline_reasoning = self.judge_fn(
                    case.question,
                    case.ground_truth,
                    baseline_answer,
                )
            except Exception as e:
                logger.warning(f"Baseline comparison failed: {e}")

        return HTMLEvalResult(
            case_id=case.id,
            category=case.category,
            original_html_length=len(case.html),
            extracted_length=len(extracted_content),
            compression_ratio=extraction_result.compression_ratio,
            answer_from_original=answer_from_original,
            answer_from_extracted=answer_from_extracted,
            answer_from_baseline=baseline_answer,
            extracted_score=extracted_score,
            extracted_reasoning=extracted_reasoning,
            baseline_score=baseline_score,
            baseline_reasoning=baseline_reasoning,
        )

    def evaluate(self, cases: list[HTMLEvalCase]) -> HTMLEvalSuiteResult:
        """Evaluate a suite of HTML extraction cases.

        Args:
            cases: List of evaluation cases.

        Returns:
            HTMLEvalSuiteResult with aggregated metrics.
        """
        results = []
        for i, case in enumerate(cases):
            logger.info(f"Evaluating case {i + 1}/{len(cases)}: {case.id}")
            try:
                result = self.evaluate_case(case)
                results.append(result)
                logger.info(
                    f"  Score: {result.extracted_score}/5, "
                    f"Compression: {(1 - result.compression_ratio) * 100:.1f}%"
                )
            except Exception as e:
                logger.error(f"  Failed: {e}")

        return HTMLEvalSuiteResult(total_cases=len(cases), results=results)


# Pre-built evaluation cases for testing
def get_sample_eval_cases() -> list[HTMLEvalCase]:
    """Get sample evaluation cases for testing.

    Returns real HTML structures that test various extraction scenarios.
    """
    return [
        HTMLEvalCase(
            id="news_article_1",
            category="news",
            url="https://example.com/news/tech-announcement",
            html="""<!DOCTYPE html>
<html>
<head>
    <title>Tech Company Announces New AI Product</title>
    <script>var analytics = {track: function(){}};</script>
    <style>body { font-family: Arial; } .ad { display: block; }</style>
</head>
<body>
    <header>
        <nav><a href="/">Home</a> | <a href="/news">News</a> | <a href="/tech">Tech</a></nav>
    </header>
    <div class="ad-banner">Advertisement: Buy our product!</div>
    <article>
        <h1>Tech Company Announces Revolutionary AI Product</h1>
        <p class="byline">By Sarah Johnson | January 15, 2024</p>
        <p>TechCorp announced today the launch of their new AI assistant called "Aria"
        which will be available starting March 2024. The product is priced at $29.99
        per month for individual users.</p>
        <p>CEO John Smith stated: "Aria represents a breakthrough in conversational AI.
        We've trained it on over 100 billion parameters and it achieves 95% accuracy
        on standard benchmarks."</p>
        <p>The company expects to reach 10 million users within the first year.</p>
    </article>
    <aside>
        <h3>Related Articles</h3>
        <ul><li><a href="/article1">Other Tech News</a></li></ul>
    </aside>
    <footer><p>&copy; 2024 News Site. Privacy Policy | Terms</p></footer>
    <script>analytics.track('pageview');</script>
</body>
</html>""",
            question="What is the name of the new AI product and when will it be available?",
            ground_truth="The new AI product is called 'Aria' and will be available starting March 2024.",
        ),
        HTMLEvalCase(
            id="documentation_1",
            category="docs",
            url="https://docs.example.com/api/authentication",
            html="""<!DOCTYPE html>
<html>
<head>
    <title>API Documentation - Authentication</title>
</head>
<body>
    <nav class="docs-sidebar">
        <a href="/docs">Home</a>
        <a href="/docs/quickstart">Quickstart</a>
        <a href="/docs/api">API Reference</a>
    </nav>
    <main class="docs-content">
        <h1>Authentication</h1>
        <p>All API requests require authentication using an API key.</p>
        <h2>Getting Your API Key</h2>
        <p>Sign up at dashboard.example.com to get your API key.
        Free tier includes 1000 requests per day.</p>
        <h2>Using the API Key</h2>
        <p>Include your API key in the Authorization header:</p>
        <pre><code>Authorization: Bearer YOUR_API_KEY</code></pre>
        <h2>Rate Limits</h2>
        <p>Free tier: 1000 requests/day. Pro tier: 100,000 requests/day.
        Enterprise: Unlimited.</p>
    </main>
    <footer>Built with DocsGen v3.0</footer>
</body>
</html>""",
            question="How many requests per day are included in the free tier?",
            ground_truth="The free tier includes 1000 requests per day.",
        ),
        HTMLEvalCase(
            id="blog_post_1",
            category="blog",
            url="https://blog.example.com/lessons-learned",
            html="""<!DOCTYPE html>
<html>
<head>
    <title>5 Lessons I Learned Building My Startup - Personal Blog</title>
    <script src="analytics.js"></script>
</head>
<body>
    <header>
        <h1 class="site-title">John's Tech Blog</h1>
        <nav>Home | About | Contact</nav>
    </header>
    <article class="blog-post">
        <h1>5 Lessons I Learned Building My Startup</h1>
        <p class="meta">Posted on December 10, 2023 by John Doe</p>
        <p>After 3 years of building StartupXYZ, here are my key takeaways:</p>
        <h2>1. Start with a small team</h2>
        <p>We started with just 3 co-founders and stayed lean for the first 18 months.</p>
        <h2>2. Focus on one thing</h2>
        <p>We tried 5 different products before finding product-market fit with our
        current offering - a B2B analytics platform.</p>
        <h2>3. Customer feedback is gold</h2>
        <p>We talked to over 200 potential customers before writing a single line of code.</p>
    </article>
    <section class="comments">
        <h3>Comments (47)</h3>
        <div class="comment">Great post!</div>
    </section>
    <footer>&copy; 2023 John's Blog</footer>
</body>
</html>""",
            question="How many potential customers did they talk to before building the product?",
            ground_truth="They talked to over 200 potential customers before writing a single line of code.",
        ),
        HTMLEvalCase(
            id="product_page_1",
            category="product",
            url="https://store.example.com/laptop-pro",
            html="""<!DOCTYPE html>
<html>
<head>
    <title>Laptop Pro X1 - TechStore</title>
    <script>trackConversion();</script>
</head>
<body>
    <header>
        <nav>Shop | Cart | Account</nav>
        <div class="search-bar"><input placeholder="Search..."></div>
    </header>
    <main class="product-page">
        <h1>Laptop Pro X1</h1>
        <div class="price">$1,299.99</div>
        <div class="specs">
            <h2>Specifications</h2>
            <ul>
                <li>Processor: Intel Core i7-12700H</li>
                <li>RAM: 16GB DDR5</li>
                <li>Storage: 512GB NVMe SSD</li>
                <li>Display: 14" 2K IPS, 120Hz</li>
                <li>Battery: Up to 12 hours</li>
                <li>Weight: 1.4 kg</li>
            </ul>
        </div>
        <div class="description">
            <h2>Description</h2>
            <p>The Laptop Pro X1 is our flagship ultrabook, designed for professionals
            who need power and portability. Featuring the latest 12th gen Intel processor
            and a stunning 2K display.</p>
        </div>
    </main>
    <aside class="recommendations">
        <h3>You might also like</h3>
        <div class="product-card">Other Laptop</div>
    </aside>
    <footer>Free shipping on orders over $50</footer>
</body>
</html>""",
            question="What is the battery life and weight of the Laptop Pro X1?",
            ground_truth="The Laptop Pro X1 has up to 12 hours of battery life and weighs 1.4 kg.",
        ),
    ]
